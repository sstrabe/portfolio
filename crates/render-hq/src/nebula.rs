//! Nebulae and dust around the Galactic Centre: bind group 3 of the trace
//! pass (`wgsl/nebula.wgsl`).
//!
//! Four volumes (the Minispiral, the circumnuclear disk, Sgr A East and a
//! Crab-like pulsar wind nebula; see `nebula/scene.rs` for what and where
//! they are) are generated on the GPU once, here, by `wgsl/nebula_gen.wgsl`:
//!
//! 1. a tiling detail-noise texture for the ray march;
//! 2. per volume: the gas or emission field; for the Galactic Centre gas a
//!    photoionisation pass that marches every voxel to the central cluster;
//!    a velocity field at a quarter of the resolution; and an occupancy
//!    mip (maxima over 8³ blocks) for empty-space skipping;
//! 3. the pulsar wind nebula's totals are summed and read back, to calibrate
//!    its brightness to the Crab's luminosities.
//!
//! Volumes are rgba16float, about 250 MB in all.
//!
//! From inside the cluster the march along each escaping ray depends only
//! on its direction, so it is cached in a cube map around the ship
//! (`wgsl/nebula_cube.wgsl`), rebuilt a band of rows per frame once the
//! ship has moved appreciably. Out among the nebulae every pixel marches.

mod scene;

pub use scene::{earth_direction, galactic, landmarks};

use crate::FrameContext;
use bytemuck::{Pod, Zeroable};
use kerr::planets::C_KM_S;
use kerr::units::{METRES_PER_M, PARSEC};
use kerr::vec3::{self, V3};
use scene::{CLUSTER, DUST_ALBEDO, DUST_ASYMMETRY, Kind, PULSAR, V_ZERO_W_M2_NM, Volume};
use wgpu::util::DeviceExt;

const VOLUME_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Mip level holding the occupancy maxima (8³ blocks).
const OCC_MIP: u32 = 3;
const DETAIL_SIZE: u32 = 64;
/// Velocity fields have a quarter of the resolution.
const VEL_DIV: u32 = 4;
/// Z slices per generation dispatch (keeps each dispatch short).
const SLAB: u32 = 32;
/// Continuum anchors of `nebula.wgsl`, nm.
const ANCHORS: [f64; 4] = [427.5, 527.5, 627.5, 727.5];
/// Extinction slope A_λ ∝ λ^−1.4 (as in `nebula.wgsl`).
const EXT_SLOPE: f64 = -1.4;
/// Faintest line emission (W m⁻² sr⁻¹), dust optical depth and
/// synchrotron (W m⁻² sr⁻¹ nm⁻¹) worth marching through, over one 8³ block.
const FLOOR_LINE: f64 = 3.0e-10;
const FLOOR_TAU: f64 = 2.0e-3;
const FLOOR_SYNCH: f64 = 1.0e-11;
/// Cube map face size (texels) and rows rebuilt per frame once built.
const CUBE_SIZE: u32 = 512;
const CUBE_ROWS_PER_FRAME: u32 = 48;
/// Rebuild the cube once the ship is this far from its centre, pc: the
/// parallax of the nearest structures (at least 1.5 pc away) stays under a
/// texel.
const CUBE_MOVE_PC: f64 = 0.002;
/// Use the cube within this distance of the hole, pc (the empty region the
/// cluster sits in).
const CUBE_RANGE_PC: f64 = 0.08;

/// Mirrors `struct NebVolume` in `nebula.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct VolumeGpu {
    centre: [f32; 4],
    ax: [f32; 4],
    ay: [f32; 4],
    az: [f32; 4],
    emit: [f32; 4],
    ratios: [f32; 4],
    ratios_b: [f32; 4],
    extra: [f32; 4],
    synch: [f32; 4],
    detail: [f32; 4],
}

/// Mirrors `struct NebParams` in `nebula.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct ParamsGpu {
    vols: [VolumeGpu; 4],
    sca: [f32; 4],
    sca2: [f32; 4],
    pulsar: [f32; 4],
    pulsar_l: [f32; 4],
}

/// Mirrors `struct Gen` in `nebula_gen.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct GenGpu {
    centre: [f32; 4],
    ax: [f32; 4],
    ay: [f32; 4],
    az: [f32; 4],
    dims: [u32; 4],
    p: [[f32; 4]; 4],
    occ: [[f32; 4]; 4],
}

/// Mirrors `struct Node` in `nebula_gen.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct NodeGpu {
    pos: [f32; 4],
    tan: [f32; 4],
    nrm: [f32; 4],
    vel: [f32; 4],
}

fn f4(v: V3, w: f64) -> [f32; 4] {
    [v[0] as f32, v[1] as f32, v[2] as f32, w as f32]
}

/// Spectrum of a binned quantity at the anchors: (λ/550)^p.
fn anchors_pow(p: f64) -> [f32; 4] {
    ANCHORS.map(|l| (l / 550.0).powf(p) as f32)
}

pub struct Nebulae {
    layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    _textures: Vec<wgpu::Texture>,
    cube: Cube,
}

/// The cube map of the march from inside the cluster.
struct Cube {
    /// Outputs A-D (see `nebula.wgsl`) as 2D arrays, for writing.
    storage: [wgpu::TextureView; 4],
    state: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    /// The generator's bind groups 0-5 (group 6 is per dispatch).
    groups: Vec<wgpu::BindGroup>,
    /// Centre of the finished cube (M), if any.
    centre: Option<V3>,
    /// A rebuild in progress: centre, face, row.
    pending: Option<(V3, u32, u32)>,
    enabled: bool,
}

/// Mirrors `struct CubeGen` in `nebula_cube.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct CubeGenGpu {
    centre: [f32; 4],
    rows: [u32; 4],
}

struct Generator<'a> {
    device: &'a wgpu::Device,
    module: wgpu::ShaderModule,
    sampler: wgpu::Sampler,
    nodes: wgpu::Buffer,
}

impl Generator<'_> {
    fn pipeline(&self, entry: &str) -> wgpu::ComputePipeline {
        self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: None,
            module: &self.module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        })
    }

    /// Run `entry` over a texture of `dims` voxels (workgroups of 4³, in
    /// slabs of z), with the uniform `u` and the given bindings (1–8).
    fn run(
        &self,
        enc: &mut wgpu::CommandEncoder,
        entry: &str,
        mut u: GenGpu,
        dims: [u32; 3],
        bindings: &[(u32, wgpu::BindingResource)],
    ) {
        let pipeline = self.pipeline(entry);
        let layout = pipeline.get_bind_group_layout(0);
        for z0 in (0..dims[2]).step_by(SLAB as usize) {
            u.dims = [dims[0], dims[1], dims[2], z0];
            let ubuf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("nebula gen"),
                contents: bytemuck::bytes_of(&u),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let mut entries = vec![wgpu::BindGroupEntry { binding: 0, resource: ubuf.as_entire_binding() }];
            entries.extend(bindings.iter().map(|(b, r)| wgpu::BindGroupEntry { binding: *b, resource: r.clone() }));
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(entry),
                layout: &layout,
                entries: &entries,
            });
            let mut pass =
                enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(entry), timestamp_writes: None });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(dims[0].div_ceil(4), dims[1].div_ceil(4), SLAB.min(dims[2] - z0).div_ceil(4));
        }
    }
}

fn texture_3d(
    device: &wgpu::Device,
    label: &str,
    dims: [u32; 3],
    format: wgpu::TextureFormat,
    mips: u32,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: dims[0], height: dims[1], depth_or_array_layers: dims[2] },
        mip_level_count: mips,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    })
}

fn mip_view(t: &wgpu::Texture, level: u32) -> wgpu::TextureView {
    t.create_view(&wgpu::TextureViewDescriptor {
        base_mip_level: level,
        mip_level_count: Some(1),
        ..Default::default()
    })
}

/// The generator's box uniform for a volume (pc).
fn gen_box(v: &Volume) -> GenGpu {
    GenGpu {
        centre: f4(v.centre, v.seed as f64),
        ax: f4(v.axes[0], v.half[0]),
        ay: f4(v.axes[1], v.half[1]),
        az: f4(v.axes[2], v.half[2]),
        p: v.params,
        ..Default::default()
    }
}

impl Nebulae {
    /// `frame` is the renderer's frame uniform (the cube generator reads the
    /// pixel size from it).
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, frame: &wgpu::Buffer) -> Self {
        let volumes = scene::volumes();
        let texture_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D3,
                multisampled: false,
            },
            count: None,
        };
        let sampler_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        };
        let mut entries = vec![
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            sampler_entry(1),
            sampler_entry(2),
        ];
        entries.extend((3..12).map(texture_entry));
        entries.extend((12..16).map(|binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2Array,
                multisampled: false,
            },
            count: None,
        }));
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 16,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
        let layout = device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("nebulae"), entries: &entries });

        let sampler = |label, mode| {
            device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some(label),
                address_mode_u: mode,
                address_mode_v: mode,
                address_mode_w: mode,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            })
        };
        let clamp = sampler("nebula clamp", wgpu::AddressMode::ClampToEdge);
        let repeat = sampler("nebula repeat", wgpu::AddressMode::Repeat);

        let nodes: Vec<NodeGpu> = scene::stream_nodes(28)
            .iter()
            .map(|n| NodeGpu {
                pos: f4(n.pos, n.density / 1.0e4),
                tan: f4(n.tangent, n.half_width),
                nrm: f4(n.normal, n.half_thickness),
                vel: f4(vec3::scale(n.velocity, 1.0e-3), n.stream as f64),
            })
            .collect();
        let generator = Generator {
            device,
            module: device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("nebula gen"),
                source: wgpu::ShaderSource::Wgsl(crate::shaders::nebula_gen().into()),
            }),
            sampler: sampler("nebula gen", wgpu::AddressMode::ClampToEdge),
            nodes: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("stream nodes"),
                contents: bytemuck::cast_slice(&nodes),
                usage: wgpu::BufferUsages::STORAGE,
            }),
        };
        use wgpu::BindingResource as R;

        let detail = texture_3d(device, "nebula detail", [DETAIL_SIZE; 3], wgpu::TextureFormat::Rgba8Unorm, 1);
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("nebula gen") });
        generator.run(
            &mut enc,
            "gen_detail",
            GenGpu::default(),
            [DETAIL_SIZE; 3],
            &[(8, R::TextureView(&mip_view(&detail, 0)))],
        );
        queue.submit([enc.finish()]);

        let dummy = texture_3d(device, "nebula dummy", [1, 1, 1], VOLUME_FORMAT, 1);
        let mut vols: Vec<wgpu::Texture> = Vec::new();
        let mut vels: Vec<wgpu::Texture> = Vec::new();
        let mut sums: Option<(wgpu::Buffer, usize)> = None;
        for v in &volumes {
            let tex = texture_3d(device, v.name, v.dims, VOLUME_FORMAT, OCC_MIP + 1);
            let level0 = mip_view(&tex, 0);
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(v.name) });
            let bx = gen_box(v);
            let nodes_res = generator.nodes.as_entire_binding();
            match (v.kind, &v.light) {
                (Kind::Minispiral | Kind::Cnd, Some((light, occluder))) => {
                    let gas = texture_3d(device, "nebula gas", v.dims, wgpu::TextureFormat::Rg32Float, 1);
                    let gas_view = gas.create_view(&Default::default());
                    let entry = if v.kind == Kind::Minispiral { "gen_minispiral" } else { "gen_cnd" };
                    let mut b: Vec<(u32, R)> = vec![(2, R::TextureView(&gas_view))];
                    if v.kind == Kind::Minispiral {
                        b.push((6, nodes_res.clone()));
                    }
                    generator.run(&mut enc, entry, bx, v.dims, &b);
                    let mut lg = bx;
                    lg.p = [*light, [0.0; 4], [0.0; 4], [0.0; 4]];
                    let occ_view = match occluder {
                        Some((i, scale)) => {
                            let o = &volumes[*i];
                            lg.occ = [
                                f4(o.centre, *scale as f64),
                                f4(vec3::scale(o.axes[0], 1.0 / o.half[0]), 0.0),
                                f4(vec3::scale(o.axes[1], 1.0 / o.half[1]), 0.0),
                                f4(vec3::scale(o.axes[2], 1.0 / o.half[2]), 0.0),
                            ];
                            vols[*i].create_view(&Default::default())
                        }
                        None => dummy.create_view(&Default::default()),
                    };
                    generator.run(
                        &mut enc,
                        "gen_light",
                        lg,
                        v.dims,
                        &[
                            (1, R::TextureView(&level0)),
                            (3, R::TextureView(&gas_view)),
                            (4, R::TextureView(&occ_view)),
                            (5, R::Sampler(&generator.sampler)),
                        ],
                    );
                    queue.submit([enc.finish()]);
                    enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(v.name) });
                }
                (Kind::SgrAEast, _) => {
                    generator.run(&mut enc, "gen_sgra_east", bx, v.dims, &[(1, R::TextureView(&level0))])
                }
                (Kind::Pwn, _) => generator.run(&mut enc, "gen_pwn", bx, v.dims, &[(1, R::TextureView(&level0))]),
                _ => unreachable!("photoionised volumes need light parameters"),
            }

            // Velocity field.
            let vdims = v.dims.map(|d| d / VEL_DIV);
            let vel = texture_3d(device, "nebula velocity", vdims, VOLUME_FORMAT, 1);
            let mut vg = bx;
            let kind = match v.kind {
                Kind::Minispiral => 0.0,
                Kind::Cnd => 1.0,
                Kind::SgrAEast => 2.0,
                Kind::Pwn => 3.0,
            };
            vg.p = [v.velocity[0], v.velocity[1], [0.0; 4], [0.0, 0.0, 0.0, kind]];
            generator.run(
                &mut enc,
                "gen_velocity",
                vg,
                vdims,
                &[(1, R::TextureView(&mip_view(&vel, 0))), (6, nodes_res)],
            );

            // Occupancy maxima.
            let full = tex.create_view(&wgpu::TextureViewDescriptor {
                base_mip_level: 0,
                mip_level_count: Some(1),
                ..Default::default()
            });
            let odims = v.dims.map(|d| d >> OCC_MIP);
            generator.run(
                &mut enc,
                "gen_occupancy",
                bx,
                odims,
                &[(1, R::TextureView(&mip_view(&tex, OCC_MIP))), (3, R::TextureView(&full))],
            );

            if v.optics.calibrate.is_some() {
                let n = (v.dims[0] * v.dims[1]) as usize;
                let size = (n * 8) as u64;
                let buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("nebula sums"),
                    size,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });
                let read = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("nebula sums read"),
                    size,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let pipeline = generator.pipeline("gen_sum");
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("gen_sum"),
                    layout: &pipeline.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry { binding: 3, resource: R::TextureView(&full) },
                        wgpu::BindGroupEntry { binding: 7, resource: buf.as_entire_binding() },
                    ],
                });
                {
                    let mut pass = enc.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&pipeline);
                    pass.set_bind_group(0, &bg, &[]);
                    pass.dispatch_workgroups(v.dims[0].div_ceil(8), v.dims[1].div_ceil(8), 1);
                }
                enc.copy_buffer_to_buffer(&buf, 0, &read, 0, size);
                sums = Some((read, n));
            }
            queue.submit([enc.finish()]);
            vols.push(tex);
            vels.push(vel);
        }

        // Calibrate the pulsar wind nebula to its total luminosities.
        let mut halpha: Vec<f64> = volumes.iter().map(|v| v.optics.halpha.unwrap_or(0.0)).collect();
        let mut synch: Vec<f64> = volumes.iter().map(|v| v.optics.synchrotron).collect();
        if let Some((read, n)) = sums {
            let slice = read.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
            let (mut sr, mut sg) = (0.0f64, 0.0f64);
            if let Ok(data) = slice.get_mapped_range() {
                let v: &[[f32; 2]] = bytemuck::cast_slice(&data[..n * 8]);
                for s in v {
                    sr += s[0] as f64;
                    sg += s[1] as f64;
                }
            }
            read.unmap();
            for (i, v) in volumes.iter().enumerate() {
                if let Some((l_halpha, l_synch)) = v.optics.calibrate {
                    let vox_m3: f64 =
                        (0..3).map(|k| 2.0 * v.half[k] * PARSEC * METRES_PER_M / v.dims[k] as f64).product();
                    // L = 4π ∫ j dV, j = C × channel.
                    halpha[i] = l_halpha / (4.0 * std::f64::consts::PI * sr.max(1e-30) * vox_m3);
                    synch[i] = l_synch / (4.0 * std::f64::consts::PI * sg.max(1e-30) * vox_m3);
                    eprintln!(
                        "nebula {}: Hα {:.3e} W m⁻³ sr⁻¹ per unit, synchrotron {:.3e} W m⁻³ sr⁻¹ nm⁻¹ per unit",
                        v.name, halpha[i], synch[i]
                    );
                }
            }
        }

        let params = params(&volumes, &halpha, &synch);
        let pbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("nebulae"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let views: Vec<wgpu::TextureView> =
            vols.iter().chain(&vels).map(|t| t.create_view(&Default::default())).collect();
        let detail_view = detail.create_view(&Default::default());
        let mut bg_entries = vec![
            wgpu::BindGroupEntry { binding: 0, resource: pbuf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: R::Sampler(&clamp) },
            wgpu::BindGroupEntry { binding: 2, resource: R::Sampler(&repeat) },
            wgpu::BindGroupEntry { binding: 3, resource: R::TextureView(&detail_view) },
        ];
        bg_entries.extend(
            views
                .iter()
                .enumerate()
                .map(|(i, v)| wgpu::BindGroupEntry { binding: 4 + i as u32, resource: R::TextureView(v) }),
        );

        // The cube map and its generator: the trace sources plus an entry
        // point, with a layout derived from what that entry point uses.
        let cube_tex: Vec<wgpu::Texture> = (0..4)
            .map(|_| {
                device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("nebula cube"),
                    size: wgpu::Extent3d { width: CUBE_SIZE, height: CUBE_SIZE, depth_or_array_layers: 6 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba32Float,
                    usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
            })
            .collect();
        let cube_view = |t: &wgpu::Texture, d| {
            t.create_view(&wgpu::TextureViewDescriptor { dimension: Some(d), ..Default::default() })
        };
        let cube_views: Vec<wgpu::TextureView> =
            cube_tex.iter().map(|t| cube_view(t, wgpu::TextureViewDimension::D2Array)).collect();
        let state = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nebula cube state"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut trace_entries = bg_entries.clone();
        trace_entries.extend(
            cube_views
                .iter()
                .enumerate()
                .map(|(i, v)| wgpu::BindGroupEntry { binding: 12 + i as u32, resource: R::TextureView(v) }),
        );
        trace_entries.push(wgpu::BindGroupEntry { binding: 16, resource: state.as_entire_binding() });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("nebulae"),
            layout: &layout,
            entries: &trace_entries,
        });

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("nebula cube"),
            source: wgpu::ShaderSource::Wgsl(crate::shaders::nebula_cube().into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("nebula cube"),
            layout: None,
            module: &module,
            entry_point: Some("cs_neb_cube"),
            compilation_options: Default::default(),
            cache: None,
        });
        let frame_entry = [wgpu::BindGroupEntry { binding: 0, resource: frame.as_entire_binding() }];
        let groups = (0..6)
            .map(|i| {
                let entries: &[wgpu::BindGroupEntry] = match i {
                    0 => &frame_entry,
                    3 => &bg_entries,
                    _ => &[],
                };
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("nebula cube"),
                    layout: &pipeline.get_bind_group_layout(i),
                    entries,
                })
            })
            .collect();
        let storage = std::array::from_fn(|i| cube_view(&cube_tex[i], wgpu::TextureViewDimension::D2Array));

        let mut textures = vols;
        textures.extend(vels);
        textures.push(detail);
        textures.extend(cube_tex);
        // KERR_NEBULA_CUBE=0 marches every pixel (for comparison).
        let enabled = params.pulsar[3] > 0.0 && std::env::var("KERR_NEBULA_CUBE").map_or(true, |v| v != "0");
        let cube = Cube { storage, state, pipeline, groups, centre: None, pending: None, enabled };
        Self { layout, bind_group, _textures: textures, cube }
    }

    pub fn layout(&self) -> &wgpu::BindGroupLayout {
        &self.layout
    }

    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    /// Decide whether the cube map can be used, and whether to rebuild it.
    pub fn update(&mut self, ctx: &FrameContext) {
        let c = &mut self.cube;
        let pos = ctx.world.pilot.position();
        let inside = c.enabled && vec3::norm(pos) < CUBE_RANGE_PC * PARSEC;
        if !inside {
            c.centre = None;
            c.pending = None;
        } else if c.pending.is_none() && c.centre.is_none_or(|o| vec3::norm(vec3::sub(pos, o)) > CUBE_MOVE_PC * PARSEC)
        {
            c.pending = Some((pos, 0, 0));
        }
        let state = match c.centre {
            Some(o) => f4(o, 1.0),
            None => [0.0; 4],
        };
        ctx.queue.write_buffer(&c.state, 0, bytemuck::bytes_of(&state));
    }

    /// The volumes are generated once, in `new`; the cube map is built here,
    /// all at once the first time and then a band of rows per frame.
    pub fn encode(&mut self, enc: &mut wgpu::CommandEncoder, ctx: &FrameContext) {
        let c = &mut self.cube;
        let Some((centre, mut face, mut row)) = c.pending else { return };
        let mut budget = if c.centre.is_none() { 6 * CUBE_SIZE } else { CUBE_ROWS_PER_FRAME };
        let layout = c.pipeline.get_bind_group_layout(6);
        while budget > 0 && face < 6 {
            let rows = budget.min(64).min(CUBE_SIZE - row);
            let u = CubeGenGpu { centre: f4(centre, 0.0), rows: [face, row, rows, CUBE_SIZE] };
            let ubuf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("nebula cube rows"),
                contents: bytemuck::bytes_of(&u),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let mut entries = vec![wgpu::BindGroupEntry { binding: 0, resource: ubuf.as_entire_binding() }];
            entries.extend(c.storage.iter().enumerate().map(|(i, v)| wgpu::BindGroupEntry {
                binding: 1 + i as u32,
                resource: wgpu::BindingResource::TextureView(v),
            }));
            let out = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("nebula cube rows"),
                layout: &layout,
                entries: &entries,
            });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("nebula cube"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&c.pipeline);
                for (i, g) in c.groups.iter().enumerate() {
                    pass.set_bind_group(i as u32, g, &[]);
                }
                pass.set_bind_group(6, &out, &[]);
                pass.dispatch_workgroups(CUBE_SIZE / 8, rows.div_ceil(8), 1);
            }
            budget -= rows;
            row += rows;
            if row >= CUBE_SIZE {
                row = 0;
                face += 1;
            }
        }
        if face >= 6 {
            c.centre = Some(centre);
            c.pending = None;
        } else {
            c.pending = Some((centre, face, row));
        }
    }
}

/// Trace parameters in units of M. `halpha` and `synch` are per metre.
fn params(volumes: &[Volume; 4], halpha: &[f64], synch: &[f64]) -> ParamsGpu {
    let m = METRES_PER_M;
    let mut p = ParamsGpu::default();
    for (i, v) in volumes.iter().enumerate() {
        let o = &v.optics;
        let vox = v.voxel_pc() * PARSEC;
        let e_ha = halpha[i] * m;
        let t_gas = o.tau_gas * m;
        let s = synch[i] * m;
        let r_thr = FLOOR_LINE / (8.0 * vox * e_ha.max(1e-30));
        let g_thr = if s > 0.0 { FLOOR_SYNCH / (8.0 * vox * s) } else { FLOOR_TAU / (8.0 * vox * t_gas.max(1e-30)) };
        let axis = |k: usize| f4(vec3::scale(v.axes[k], 1.0 / (v.half[k] * PARSEC)), v.dims[k] as f64);
        let sigma = o.sigma_kms.hypot(9.0) / C_KM_S;
        p.vols[i] = VolumeGpu {
            centre: f4(vec3::scale(v.centre, PARSEC), vox),
            ax: axis(0),
            ay: axis(1),
            az: axis(2),
            emit: [e_ha as f32, t_gas as f32, (o.tau_ion * m) as f32, s as f32],
            ratios: o.ratios.map(|x| x as f32),
            ratios_b: o.ratios_front.map(|x| x as f32),
            extra: [o.oiii_per_a as f32, if o.scatters { 1.0 } else { 0.0 }, r_thr as f32, g_thr as f32],
            synch: anchors_pow(o.synch_index - 2.0),
            detail: [o.detail as f32, (sigma * sigma) as f32, (1000.0 / C_KM_S) as f32, 0.0],
        };
    }
    // Starlight of the central cluster at the anchors, for dust scattering:
    // ω κ_λ/κ_V L_λ / 4π, per (m per M)².
    let sca: [f32; 4] = std::array::from_fn(|k| {
        let l = ANCHORS[k];
        let lum: f64 = CLUSTER
            .components
            .iter()
            .map(|&(lw, t)| {
                lw * crate::spectrum::planck(l, t) * std::f64::consts::PI / (crate::spectrum::SIGMA_SB * t.powi(4))
            })
            .sum();
        (DUST_ALBEDO * (l / 550.0).powf(EXT_SLOPE) * lum / (4.0 * std::f64::consts::PI * m * m)) as f32
    });
    let core = CLUSTER.core_pc * PARSEC;
    let reach = volumes.iter().map(|v| vec3::norm(v.centre) + vec3::norm(v.half)).fold(0.0, f64::max) * PARSEC;
    p.sca = sca;
    p.sca2 = [DUST_ASYMMETRY as f32, (core * core) as f32, (0.08 * PARSEC) as f32, reach as f32];
    let enabled = std::env::var("KERR_NEBULAE").map_or(true, |v| v != "0");
    p.pulsar = f4(vec3::scale(scene::pwn_centre(), PARSEC), if enabled { 1.0 } else { 0.0 });
    let d = PULSAR.distance_pc * PARSEC * m;
    let l550 = V_ZERO_W_M2_NM * 10f64.powf(-0.4 * PULSAR.v0_mag) * 4.0 * std::f64::consts::PI * d * d;
    let shape = anchors_pow(PULSAR.index - 2.0);
    p.pulsar_l = shape.map(|s| (s as f64 * l550 / (4.0 * std::f64::consts::PI * m * m)) as f32);
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_layouts_match_wgsl() {
        assert_eq!(std::mem::size_of::<VolumeGpu>(), 10 * 16);
        assert_eq!(std::mem::size_of::<ParamsGpu>(), (4 * 10 + 4) * 16);
        assert_eq!(std::mem::size_of::<GenGpu>(), 13 * 16);
        assert_eq!(std::mem::size_of::<NodeGpu>(), 4 * 16);
    }
}
