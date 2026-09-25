//! Optics and post-processing after the trace, at the output resolution:
//!
//! 1. `taa`: temporal accumulation and upscaling of the traced radiance
//!    (the trace is jittered; history is reprojected by the frame-to-frame
//!    Lorentz transformation and accumulates while the view is still);
//! 2. `splat`: every point source's flux at its exact sub-pixel position,
//!    spread over the PSF core; `ghost`: lens ghosts of bright sources;
//! 3. `fft`: the scene convolved with the physical point-spread function
//!    (pupil diffraction per wavelength plus scatter; `optics.rs`) outside
//!    its core, in frequency space on a coarser grid, and the luminance
//!    histogram of the result;
//! 4. `exposure`: metering and eye-like adaptation on the GPU;
//! 5. `post`: sharp core plus wings, rod vision, AgX, display encoding.
//!
//! See the WGSL files of the same names for the physics of each step.

use crate::fft::{self, Grid};
use crate::optics::{self, GHOSTS, Optics, Palette};
use crate::profile::Profiler;
use crate::session::W_PER_FLUX_UNIT;
use crate::{FrameContext, shaders, spectrum};
use bytemuck::{Pod, Zeroable};
use std::sync::{Arc, Mutex};

/// Buffers the post passes read, owned by [`crate::Gpu`].
pub struct Shared<'a> {
    pub frame: &'a wgpu::Buffer,
    pub hq: &'a wgpu::Buffer,
    pub images: &'a wgpu::Buffer,
    pub meta: &'a wgpu::Buffer,
}

/// How the image is taken and shown; the window changes these with keys.
#[derive(Clone, Copy, Debug, Default)]
pub struct Settings {
    pub optics: Optics,
    pub palette: Palette,
    /// Exposure compensation in stops (astrograph: stops over its base
    /// long exposure).
    pub ev: f64,
}

/// Display value (before the tone curve) of the core of the faintest star
/// the dark-adapted eye can see: the upper bound of the exposure.
const FAINT_STAR_PEAK: f64 = 0.06;
/// Display value (AgX saturates near 16) the core of the fifth brightest
/// point source may reach under automatic exposure: bright stars glare
/// white, ten magnitudes of fainter ones stay visible.
const HIGHLIGHT: f32 = 600.0;
/// The astrograph's exposure, in stops over the meter's reading of the
/// scene's background.
const ASTRO_STOPS: f64 = 2.0;
/// Accumulated history weight (≈ frames) while moving, and at most when
/// still.
const HISTORY_MOVING: f32 = 8.0;
const HISTORY_STILL: f32 = 256.0;
/// Hubble-palette channel stretch ([S II], Hα, [O III]) relative to each
/// other: the faint lines are stretched as in the published composites.
const SHO_STRETCH: [f64; 3] = [2.2, 1.0, 2.6];

/// Mirrors `struct PostFrame` in `post_common.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct PostFrame {
    reproject: [[f32; 4]; 4],
    reproject_ship: [[f32; 4]; 4],
    reproject_terrain: [[f32; 4]; 4],
    sizes: [u32; 4],
    grid: [u32; 4],
    grid2: [u32; 4],
    taa: [f32; 4],
    taa2: [f32; 4],
    mode: [u32; 4],
    expo: [f32; 4],
    expo2: [f32; 4],
    optics: [f32; 4],
    sho: [f32; 4],
    ghosts: [[f32; 4]; 16],
}

/// Mirrors `struct FftJob` in `fft.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct FftJob {
    n: [u32; 4],
    valid: [u32; 4],
}

/// Mirrors `struct PsfParams` in `psf.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct PsfParams {
    xyz: [[f32; 4]; 12],
    pupil: [f32; 4],
    dims: [u32; 4],
    scatter: [f32; 4],
    core: [f32; 4],
}

struct Pipelines {
    taa: wgpu::ComputePipeline,
    splat: wgpu::RenderPipeline,
    ghost: wgpu::RenderPipeline,
    psf: wgpu::ComputePipeline,
    down: wgpu::ComputePipeline,
    rows_fwd: wgpu::ComputePipeline,
    cols: wgpu::ComputePipeline,
    rows_inv: wgpu::ComputePipeline,
    exposure: wgpu::ComputePipeline,
    composite: wgpu::RenderPipeline,
}

/// Resources at the output resolution (kept across render-scale changes,
/// so the history survives the adaptive resolution).
struct Targets {
    out: (u32, u32),
    grid: Grid,
    hist: [wgpu::TextureView; 2],
    weight: [wgpu::TextureView; 2],
    scene: wgpu::TextureView,
    spec: wgpu::Buffer,

    job_kernel: wgpu::Buffer,
    splat_bg: wgpu::BindGroup,
    ghost_bg: [wgpu::BindGroup; 2],
    down_bg: wgpu::BindGroup,
    rows_fwd_bg: wgpu::BindGroup,
    cols_bg: wgpu::BindGroup,
    cols_kernel_bg: wgpu::BindGroup,
    rows_inv_bg: wgpu::BindGroup,
    composite_bg: wgpu::BindGroup,
    taa_bg: Option<[wgpu::BindGroup; 2]>,
}

/// Kernel of the current optics, or what it was built for.
#[derive(Clone, Copy, PartialEq)]
struct KernelKey {
    optics: Optics,
    grid: Grid,
    delta: f32,
}

pub struct Post {
    pub settings: Settings,
    bodies: u32,
    pipes: Pipelines,
    frame: wgpu::Buffer,
    hq: wgpu::Buffer,
    images: wgpu::Buffer,
    meta: wgpu::Buffer,
    post_buf: wgpu::Buffer,
    psf_buf: wgpu::Buffer,
    expo_buf: wgpu::Buffer,
    histogram: wgpu::Buffer,
    exposure_bg: wgpu::BindGroup,
    twiddles: wgpu::Buffer,
    sampler: wgpu::Sampler,
    targets: Option<Targets>,
    input: Option<(wgpu::TextureView, wgpu::TextureView, (u32, u32))>,
    kernel: Option<KernelKey>,
    kernel_pending: Option<(KernelKey, wgpu::BindGroup, wgpu::BindGroup, wgpu::Buffer)>,
    uniforms: PostFrame,
    count: u32,
    parity: usize,
    prev_tetrad: Option<kerr::pilot::Tetrad>,
    prev_camera: Option<crate::ship::camera::Pose>,
    /// The planet with terrain tiles last frame: its body axes in the view
    /// axes, and the eye's body-fixed position (km).
    prev_terrain: Option<([kerr::vec3::V3; 3], kerr::vec3::V3)>,
    still_frames: u32,
    last_wall: Option<f64>,
    snap_frames: u32,
    reset: bool,
    last_palette: Palette,
    last_optics: Optics,
    readback: Readback,
    profiler: Option<Profiler>,
}

/// What the meter saw (mirrors `struct Exposure` in `post_common.wgsl`).
#[derive(Clone, Copy, Debug, Default)]
pub struct Metering {
    /// Adapted exposure: display value per unit radiance (W m⁻² sr⁻¹, CIE Y).
    pub exposure: f64,
    /// Metered scene luminance, cd/m².
    pub metered: f64,
    /// 95th percentile luminance of the image, cd/m².
    pub p95: f64,
    /// Peak luminance of the fifth brightest point source, cd/m².
    pub fifth_star: f64,
}

/// The adapted exposure, read back a frame or two late without stalling.
struct Readback {
    slots: Vec<(wgpu::Buffer, Arc<Mutex<bool>>)>,
    value: Arc<Mutex<[f32; 4]>>,
}

impl Post {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        out_format: wgpu::TextureFormat,
        shared: &Shared,
        bodies: u32,
    ) -> Self {
        let module = |label: &str, src: &str| {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            })
        };
        let compute = |label: &str, m: &wgpu::ShaderModule, entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: m,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let render = |label: &str, m: &wgpu::ShaderModule, vs: &str, fs: &str, format, blend| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: None,
                vertex: wgpu::VertexState {
                    module: m,
                    entry_point: Some(vs),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: m,
                    entry_point: Some(fs),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState { format, blend, write_mask: wgpu::ColorWrites::ALL })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let add = |src_factor| wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        };
        let hdr = crate::gpu::HDR_FORMAT;
        let taa_mod = module("taa", &shaders::taa());
        let splat_mod = module("splat", &shaders::splat());
        let psf_mod = module("psf", &shaders::psf());
        let fft_mod = module("fft", &shaders::fft());
        let expo_mod = module("exposure", &shaders::exposure());
        let post_mod = module("post", &shaders::post());
        let pipes = Pipelines {
            taa: compute("taa", &taa_mod, "cs_taa"),
            // dst += src · dst.alpha (the near-field transmittance).
            splat: render("splat", &splat_mod, "vs_splat", "fs_splat", hdr, Some(add(wgpu::BlendFactor::DstAlpha))),
            // Ghosts form inside the lens: nothing in the scene hides them.
            ghost: render("ghost", &splat_mod, "vs_ghost", "fs_ghost", hdr, Some(add(wgpu::BlendFactor::One))),
            psf: compute("psf", &psf_mod, "cs_psf"),
            down: compute("down", &fft_mod, "cs_down"),
            rows_fwd: compute("fft rows", &fft_mod, "cs_rows_fwd"),
            cols: compute("fft columns", &fft_mod, "cs_cols"),
            rows_inv: compute("fft inverse rows", &fft_mod, "cs_rows_inv"),
            exposure: compute("exposure", &expo_mod, "cs_exposure"),
            composite: render("post", &post_mod, "vs_fullscreen", "fs_post", out_format, None),
        };
        let buffer = |label, size, usage| {
            device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage, mapped_at_creation: false })
        };
        use wgpu::BufferUsages as U;
        let post_buf = buffer("post frame", std::mem::size_of::<PostFrame>() as u64, U::UNIFORM | U::COPY_DST);
        let psf_buf = buffer("psf params", std::mem::size_of::<PsfParams>() as u64, U::UNIFORM | U::COPY_DST);
        let expo_buf = buffer("exposure", 16, U::STORAGE | U::COPY_SRC | U::COPY_DST);
        let histogram = buffer("histogram", 256 * 4, U::STORAGE | U::COPY_DST);
        let twiddles = buffer("twiddles", fft::MAX_N as u64 / 2 * 8, U::STORAGE | U::COPY_DST);
        let slots = (0..3)
            .map(|_| (buffer("exposure readback", 16, U::MAP_READ | U::COPY_DST), Arc::new(Mutex::new(false))))
            .collect();
        let exposure_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("exposure"),
            layout: &pipes.exposure.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: shared.frame.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: shared.hq.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: post_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: expo_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: shared.images.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: shared.meta.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 12, resource: histogram.as_entire_binding() },
            ],
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("post linear"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        Self {
            settings: Settings::default(),
            bodies,
            pipes,
            frame: shared.frame.clone(),
            hq: shared.hq.clone(),
            images: shared.images.clone(),
            meta: shared.meta.clone(),
            post_buf,
            psf_buf,
            expo_buf,
            histogram,
            exposure_bg,
            twiddles,
            sampler,
            targets: None,
            input: None,
            kernel: None,
            kernel_pending: None,
            uniforms: PostFrame::default(),
            count: 0,
            parity: 0,
            prev_tetrad: None,
            prev_camera: None,
            prev_terrain: None,
            still_frames: 0,
            last_wall: None,
            snap_frames: 3,
            reset: true,
            last_palette: Palette::True,
            last_optics: Optics::default(),
            readback: Readback { slots, value: Arc::default() },
            profiler: Profiler::new(device, queue, 24),
        }
    }

    /// New trace targets (after a resize or a render-scale change). The
    /// output-resolution history is only rebuilt when the output size
    /// changes.
    pub fn attach(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        trace: &wgpu::TextureView,
        narrowband: &wgpu::TextureView,
        trace_size: (u32, u32),
        out: (u32, u32),
    ) {
        if self.targets.as_ref().is_none_or(|t| t.out != out) {
            self.targets = Some(self.create_targets(device, queue, out));
            self.reset = true;
        }
        self.input = Some((trace.clone(), narrowband.clone(), trace_size));
        let t = self.targets.as_mut().expect("targets");
        let layout = self.pipes.taa.get_bind_group_layout(0);
        let bg = |p: usize| {
            use wgpu::BindingResource as R;
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("taa"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.frame.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: self.post_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: self.expo_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: R::TextureView(trace) },
                    wgpu::BindGroupEntry { binding: 5, resource: R::TextureView(narrowband) },
                    wgpu::BindGroupEntry { binding: 6, resource: R::TextureView(&t.hist[p]) },
                    wgpu::BindGroupEntry { binding: 7, resource: R::TextureView(&t.weight[p]) },
                    wgpu::BindGroupEntry { binding: 8, resource: R::Sampler(&self.sampler) },
                    wgpu::BindGroupEntry { binding: 9, resource: R::TextureView(&t.hist[1 - p]) },
                    wgpu::BindGroupEntry { binding: 10, resource: R::TextureView(&t.weight[1 - p]) },
                    wgpu::BindGroupEntry { binding: 11, resource: R::TextureView(&t.scene) },
                ],
            })
        };
        t.taa_bg = Some([bg(0), bg(1)]);
    }

    fn create_targets(&self, device: &wgpu::Device, queue: &wgpu::Queue, out: (u32, u32)) -> Targets {
        use wgpu::BindingResource as R;
        use wgpu::TextureUsages as T;
        let grid = Grid::for_output(out.0, out.1);
        let texture = |label, (w, h): (u32, u32), format, usage| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        };
        let rgba = wgpu::TextureFormat::Rgba32Float;
        let hist = [0, 1].map(|_| texture("history", out, rgba, T::STORAGE_BINDING | T::TEXTURE_BINDING));
        let weight = [0, 1].map(|_| {
            texture("history weight", out, wgpu::TextureFormat::R32Float, T::STORAGE_BINDING | T::TEXTURE_BINDING)
        });
        let scene = texture("scene", out, rgba, T::STORAGE_BINDING | T::TEXTURE_BINDING | T::RENDER_ATTACHMENT);
        let wide = texture("psf wings", (grid.width, grid.height), rgba, T::STORAGE_BINDING | T::TEXTURE_BINDING);
        let storage = |label, size| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let cells = grid.nx as u64 * grid.ny as u64 * 16;
        let down = storage("grid image", grid.width as u64 * grid.height as u64 * 16);
        let spec = storage("fft", cells);
        let kspec = storage("psf spectrum", cells);
        let job = |valid: [u32; 4]| {
            let j = FftJob { n: [grid.nx, grid.ny, grid.nx.trailing_zeros(), grid.ny.trailing_zeros()], valid };
            let b = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("fft job"),
                size: std::mem::size_of::<FftJob>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&b, 0, bytemuck::bytes_of(&j));
            b
        };
        let job_img = job([grid.width, grid.height, 0, grid.height]);
        let job_kernel = job([grid.nx, grid.ny, 1, grid.ny]);
        queue.write_buffer(&self.twiddles, 0, bytemuck::cast_slice(&fft::twiddles()));

        fn e(binding: u32, resource: wgpu::BindingResource<'_>) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry { binding, resource }
        }
        let bg = |label, pipe: &dyn Fn() -> wgpu::BindGroupLayout, entries: &[wgpu::BindGroupEntry]| {
            device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some(label), layout: &pipe(), entries })
        };
        let p = &self.pipes;
        let splat_bg = bg(
            "splat",
            &|| p.splat.get_bind_group_layout(0),
            &[
                e(0, self.frame.as_entire_binding()),
                e(1, self.hq.as_entire_binding()),
                e(2, self.post_buf.as_entire_binding()),
                e(3, self.expo_buf.as_entire_binding()),
                e(4, self.images.as_entire_binding()),
                e(5, self.meta.as_entire_binding()),
            ],
        );
        let ghost_bg = [0, 1].map(|i| {
            bg(
                "ghost",
                &|| p.ghost.get_bind_group_layout(0),
                &[
                    e(0, self.frame.as_entire_binding()),
                    e(1, self.hq.as_entire_binding()),
                    e(2, self.post_buf.as_entire_binding()),
                    e(3, self.expo_buf.as_entire_binding()),
                    e(4, self.images.as_entire_binding()),
                    e(5, self.meta.as_entire_binding()),
                    e(6, R::TextureView(&hist[i])),
                    e(7, R::Sampler(&self.sampler)),
                ],
            )
        });
        let down_bg = bg(
            "down",
            &|| p.down.get_bind_group_layout(0),
            &[e(2, self.post_buf.as_entire_binding()), e(4, R::TextureView(&scene)), e(5, down.as_entire_binding())],
        );
        let rows_fwd_bg = bg(
            "fft rows",
            &|| p.rows_fwd.get_bind_group_layout(0),
            &[
                e(6, down.as_entire_binding()),
                e(7, spec.as_entire_binding()),
                e(10, self.twiddles.as_entire_binding()),
                e(11, job_img.as_entire_binding()),
            ],
        );
        let cols = |job: &wgpu::Buffer| {
            bg(
                "fft columns",
                &|| p.cols.get_bind_group_layout(0),
                &[
                    e(7, spec.as_entire_binding()),
                    e(8, kspec.as_entire_binding()),
                    e(10, self.twiddles.as_entire_binding()),
                    e(11, job.as_entire_binding()),
                ],
            )
        };
        let cols_bg = cols(&job_img);
        let cols_kernel_bg = cols(&job_kernel);
        let rows_inv_bg = bg(
            "fft inverse rows",
            &|| p.rows_inv.get_bind_group_layout(0),
            &[
                e(5, down.as_entire_binding()),
                e(7, spec.as_entire_binding()),
                e(8, kspec.as_entire_binding()),
                e(9, R::TextureView(&wide)),
                e(10, self.twiddles.as_entire_binding()),
                e(11, job_img.as_entire_binding()),
                e(12, self.histogram.as_entire_binding()),
            ],
        );
        let composite_bg = bg(
            "post",
            &|| p.composite.get_bind_group_layout(0),
            &[
                e(0, self.frame.as_entire_binding()),
                e(2, self.post_buf.as_entire_binding()),
                e(3, self.expo_buf.as_entire_binding()),
                e(4, R::TextureView(&scene)),
                e(5, R::TextureView(&wide)),
                e(6, R::Sampler(&self.sampler)),
                e(7, kspec.as_entire_binding()),
            ],
        );
        Targets {
            out,
            grid,
            hist,
            weight,
            scene,
            spec,

            job_kernel,
            splat_bg,
            ghost_bg,
            down_bg,
            rows_fwd_bg,
            cols_bg,
            cols_kernel_bg,
            rows_inv_bg,
            composite_bg,
            taa_bg: None,
        }
    }

    /// Sub-pixel offset (trace pixels) for the next frame's rays: the
    /// Halton (2, 3) sequence, 16 phases.
    pub fn jitter(&self) -> [f32; 2] {
        let i = self.count % 16 + 1;
        [halton(i, 2) - 0.5, halton(i, 3) - 0.5]
    }

    /// The exposure state as last read back from the GPU.
    pub fn exposure(&self) -> Metering {
        let v = *self.readback.value.lock().unwrap_or_else(|e| e.into_inner());
        Metering { exposure: v[0] as f64, metered: v[1] as f64, p95: v[2] as f64, fifth_star: v[3] as f64 }
    }

    /// Whether the trace must also write the narrowband bins.
    pub fn wants_narrowband(&self) -> bool {
        self.settings.palette == Palette::Hubble
    }

    /// Exposure at which the faintest star of the configured limit is just
    /// visible: the dark-adapted eye, and the ceiling of the adaptation.
    pub fn dark_exposure(&self, world: &kerr::world::World, pixel: f64) -> f64 {
        let sigma = self.settings.optics.core_sigma().max(0.5 * pixel);
        let f_ref = world.cfg.star_flux_ref * W_PER_FLUX_UNIT;
        FAINT_STAR_PEAK * std::f64::consts::TAU * sigma * sigma / (f_ref * spectrum::y_per_watt(5800.0))
    }

    pub fn update(&mut self, ctx: &FrameContext) {
        let Some(t) = &self.targets else { return };
        let Some((_, _, trace)) = self.input else { return };
        let s = self.settings;
        if s.palette != self.last_palette {
            self.reset = true;
            self.last_palette = s.palette;
        }
        if s.optics != self.last_optics {
            self.snap_frames = self.snap_frames.max(2);
            self.last_optics = s.optics;
        }
        let dt = self.last_wall.map_or(1.0 / 60.0, |w| (ctx.wall_time - w).clamp(1e-4, 0.25));
        self.last_wall = Some(ctx.wall_time);

        // g(e'_a, e_b): previous tetrad against the current one.
        let pilot = &ctx.world.pilot;
        let pos = pilot.position();
        let view = *ctx.view;
        let prev = self.prev_tetrad.unwrap_or(view);
        let mut m = [[0.0f32; 4]; 4];
        let mut dev = 0.0f64;
        for (a, row) in m.iter_mut().enumerate() {
            for (b, v) in row.iter_mut().enumerate() {
                let g = ctx.world.kerr.dot(pos, prev[a], view[b]);
                // Identical frames give the Minkowski metric diag(−1, 1, 1, 1).
                let eta = match (a == b, a) {
                    (false, _) => 0.0,
                    (true, 0) => -1.0,
                    _ => 1.0,
                };
                dev = dev.max((g - eta).abs());
                *v = g as f32;
            }
        }
        self.prev_tetrad = Some(view);
        // The ship moves with the camera: its pixels only move when the
        // camera moves relative to it (the chase camera orbited or zoomed).
        let ship_m = crate::ship::camera::reprojection(&self.prev_camera.unwrap_or(ctx.camera), &ctx.camera);
        self.prev_camera = Some(ctx.camera);
        let terrain_m = terrain_reprojection(self.prev_terrain.or(ctx.terrain), ctx.terrain);
        self.prev_terrain = ctx.terrain;
        self.still_frames = if dev < 2e-6 { self.still_frames + 1 } else { 0 };
        let cap = if self.still_frames > 2 {
            (HISTORY_MOVING + self.still_frames as f32).min(HISTORY_STILL)
        } else {
            HISTORY_MOVING
        };
        let gamma = if self.still_frames > 2 { 2.5 } else { 1.25 };

        let pixel_trace = (ctx.hq.view[3] as f64).sqrt();
        let pixel = pixel_trace * trace.1 as f64 / t.out.1 as f64;
        let delta = pixel * t.grid.step as f64;
        let dark = self.dark_exposure(ctx.world, pixel);
        let manual = if s.optics == Optics::Astro { 2f64.powf(ASTRO_STOPS) } else { 0.0 };
        let (tau_light, tau_dark) = match s.optics {
            Optics::Eye => (0.4, 3.0),
            _ => (0.25, 0.8),
        };
        if self.count == 0 {
            // Start where a dark-adapted eye would be.
            ctx.queue.write_buffer(&self.expo_buf, 0, bytemuck::cast_slice(&[dark as f32, 0.0, 0.0, 0.0]));
        }
        let mut ghosts = [[0.0f32; 4]; 16];
        for (i, g) in GHOSTS.iter().enumerate() {
            let tint = optics::ghost_tint(g);
            ghosts[2 * i] = [g.k as f32, g.radius as f32, 0.3 + 0.9 * i as f32, 0.0];
            ghosts[2 * i + 1] = [tint[0] as f32, tint[1] as f32, tint[2] as f32, 0.0];
        }
        // Hubble palette: a flat spectrum keeps the luminance it has in
        // true colour, so exposures carry over.
        let flat_y: f64 = spectrum::xyz_weights().iter().map(|w| w[1]).sum();
        let sho_luma = 0.2126 * SHO_STRETCH[0] + 0.7152 * SHO_STRETCH[1] + 0.0722 * SHO_STRETCH[2];
        let sho = SHO_STRETCH.map(|v| (v * flat_y / sho_luma) as f32);

        self.uniforms = PostFrame {
            reproject: m,
            reproject_ship: ship_m,
            reproject_terrain: terrain_m,
            sizes: [t.out.0, t.out.1, trace.0, trace.1],
            grid: [t.grid.nx, t.grid.ny, t.grid.width, t.grid.height],
            grid2: [t.grid.step, self.count, 0, 0],
            taa: [ctx.hq.view[0], ctx.hq.view[1], cap, if self.reset { 1.0 } else { 0.0 }],
            taa2: [gamma, 0.0, 0.0, 0.0],
            mode: [
                s.optics.index(),
                (s.palette == Palette::Hubble) as u32,
                s.optics.ghosts() as u32,
                (self.snap_frames > 0) as u32,
            ],
            expo: [2f64.powf(s.ev) as f32, 1e-14, dark as f32, dt as f32],
            expo2: [manual as f32, tau_light, tau_dark, HIGHLIGHT],
            optics: [s.optics.core_sigma() as f32, pixel as f32, delta as f32, 0.0],
            sho: [sho[0], sho[1], sho[2], 0.0],
            ghosts,
        };
        ctx.queue.write_buffer(&self.post_buf, 0, bytemuck::bytes_of(&self.uniforms));

        let key = KernelKey { optics: s.optics, grid: t.grid, delta: delta as f32 };
        if self.kernel != Some(key) && self.kernel_pending.as_ref().is_none_or(|p| p.0 != key) {
            self.kernel_pending = Some(self.prepare_kernel(ctx.device, ctx.queue, key));
        }
    }

    /// Parameters and scratch space for building the kernel of `key`.
    fn prepare_kernel(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        key: KernelKey,
    ) -> (KernelKey, wgpu::BindGroup, wgpu::BindGroup, wgpu::Buffer) {
        let t = self.targets.as_ref().expect("targets");
        let g = key.grid;
        let o = key.optics;
        let mut xyz = [[0.0f32; 4]; 12];
        for (k, w) in spectrum::xyz_weights().iter().enumerate() {
            for c in 0..3 {
                xyz[c * 4 + k / 4][k % 4] = w[c] as f32;
            }
        }
        let (mode, blades, rotation, obstruction, vane) = match o.pupil() {
            optics::Pupil::Disc => (0, 0, 0.0, 0.0, 0.0),
            optics::Pupil::Polygon { blades, rotation } => (1, blades, rotation, 0.0, 0.0),
            optics::Pupil::Telescope { obstruction, vane } => (2, 0, 0.0, obstruction, vane),
        };
        let (reach_x, reach_y) = g.reach();
        let taper = reach_x.min(reach_y).min(g.nx / 2).min(g.ny / 2) as f64;
        let delta = key.delta as f64;
        let (fs, theta0) = o.scatter();
        let params = PsfParams {
            xyz,
            pupil: [(o.diameter() * delta) as f32, obstruction as f32, vane as f32, rotation as f32],
            dims: [g.nx, g.ny, mode, blades],
            scatter: [fs as f32, theta0 as f32, delta as f32, taper as f32],
            core: [1.0, 2.5, optics::harvey_norm(theta0, taper * delta, 2.5) as f32, 0.0],
        };
        queue.write_buffer(&self.psf_buf, 0, bytemuck::bytes_of(&params));
        let kimg = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("psf kernel"),
            size: g.nx as u64 * g.ny as u64 * 16,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let psf_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("psf"),
            layout: &self.pipes.psf.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 3, resource: self.psf_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: kimg.as_entire_binding() },
            ],
        });
        let rows_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("psf rows"),
            layout: &self.pipes.rows_fwd.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 6, resource: kimg.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: t.spec.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: self.twiddles.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 11, resource: t.job_kernel.as_entire_binding() },
            ],
        });
        (key, psf_bg, rows_bg, kimg)
    }

    /// Everything after the trace, into `out`.
    pub fn encode(&mut self, enc: &mut wgpu::CommandEncoder, out: &wgpu::TextureView) {
        let Some(t) = &self.targets else { return };
        let Some(taa_bg) = &t.taa_bg else { return };
        let g = t.grid;
        let prof = &mut self.profiler;
        fn compute(
            enc: &mut wgpu::CommandEncoder,
            prof: &mut Option<Profiler>,
            label: &'static str,
            pipe: &wgpu::ComputePipeline,
            bg: &wgpu::BindGroup,
            groups: (u32, u32),
        ) {
            let timestamp_writes = prof.as_mut().and_then(|p| p.compute(label));
            let mut pass =
                enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(label), timestamp_writes });
            pass.set_pipeline(pipe);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(groups.0, groups.1, 1);
        }

        if let Some((key, psf_bg, rows_bg, _kimg)) = self.kernel_pending.take() {
            let p = &self.pipes;
            compute(enc, prof, "psf kernel", &p.psf, &psf_bg, (g.nx.div_ceil(8), g.ny.div_ceil(8)));
            compute(enc, prof, "psf kernel fft rows", &p.rows_fwd, &rows_bg, (g.ny, 1));
            compute(enc, prof, "psf kernel fft columns", &p.cols, &t.cols_kernel_bg, (g.nx / 2 + 1, 1));
            self.kernel = Some(key);
        }
        if self.kernel.is_none() {
            return;
        }
        let p = &self.pipes;
        let (w, h) = t.out;
        compute(enc, prof, "taa", &p.taa, &taa_bg[self.parity], (w.div_ceil(8), h.div_ceil(8)));
        let next = 1 - self.parity;
        {
            let timestamp_writes = prof.as_mut().and_then(|p| p.render("splat + ghosts"));
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("splat"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &t.scene,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&p.splat);
            pass.set_bind_group(0, &t.splat_bg, &[]);
            pass.draw(0..6, 0..self.bodies * 2);
            if self.settings.optics.ghosts() {
                pass.set_pipeline(&p.ghost);
                pass.set_bind_group(0, &t.ghost_bg[next], &[]);
                pass.draw(0..6, 0..self.bodies * 2 * GHOSTS.len() as u32);
            }
        }
        compute(enc, prof, "downsample", &p.down, &t.down_bg, (g.width.div_ceil(8), g.height.div_ceil(8)));
        compute(enc, prof, "fft rows", &p.rows_fwd, &t.rows_fwd_bg, (g.height, 1));
        compute(enc, prof, "fft columns × psf", &p.cols, &t.cols_bg, (g.nx / 2 + 1, 1));
        compute(enc, prof, "fft inverse rows + histogram", &p.rows_inv, &t.rows_inv_bg, (g.height, 1));
        compute(enc, prof, "exposure", &p.exposure, &self.exposure_bg, (1, 1));
        {
            let timestamp_writes = prof.as_mut().and_then(|p| p.render("composite (eye, AgX)"));
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("post"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: out,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&p.composite);
            pass.set_bind_group(0, &t.composite_bg, &[]);
            pass.draw(0..3, 0..1);
        }
        self.encode_readback(enc);
        if let Some(prof) = &mut self.profiler {
            prof.finish(enc);
        }
        self.parity = next;
        self.count = self.count.wrapping_add(1);
        self.reset = false;
        self.snap_frames = self.snap_frames.saturating_sub(1);
    }

    /// Timestamp writes for a pass outside `post` (the trace), when timing.
    pub fn timestamps(&mut self, label: &'static str) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        self.profiler.as_mut().and_then(|p| p.compute(label))
    }

    /// The profiler, when timing: for passes submitted outside the frame's
    /// encoder earlier in the frame (terrain tiles).
    pub fn profiler(&mut self) -> Option<&mut Profiler> {
        self.profiler.as_mut()
    }

    /// Mean GPU milliseconds per pass, when `KERR_GPU_TIMING` is set.
    pub fn timings(&self) -> Vec<(&'static str, f64, u32)> {
        self.profiler.as_ref().map(|p| p.report()).unwrap_or_default()
    }

    /// Copy the exposure into a free read-back slot; it maps after submit.
    fn encode_readback(&self, enc: &mut wgpu::CommandEncoder) {
        let Some((buf, busy)) = self.readback.slots.iter().find(|(_, b)| !*b.lock().unwrap_or_else(|e| e.into_inner()))
        else {
            return;
        };
        *busy.lock().unwrap_or_else(|e| e.into_inner()) = true;
        enc.copy_buffer_to_buffer(&self.expo_buf, 0, buf, 0, 16);
        let (mapped, busy, value) = (buf.clone(), busy.clone(), self.readback.value.clone());
        enc.map_buffer_on_submit(buf, wgpu::MapMode::Read, .., move |result| {
            if result.is_ok()
                && let Ok(data) = mapped.slice(..).get_mapped_range()
            {
                let v: &[f32] = bytemuck::cast_slice(&data);
                *value.lock().unwrap_or_else(|e| e.into_inner()) = [v[0], v[1], v[2], v[3]];
            }
            mapped.unmap();
            *busy.lock().unwrap_or_else(|e| e.into_inner()) = false;
        });
    }
}

/// Element `i` of the Halton sequence in `base`.
fn halton(mut i: u32, base: u32) -> f32 {
    let mut f = 1.0;
    let mut r = 0.0;
    while i > 0 {
        f /= base as f32;
        r += f * (i % base) as f32;
        i /= base;
    }
    r
}

/// TAA's reprojection of terrain pixels (`reproject_terrain` in
/// `post_common.wgsl`): a point at distance D along the current direction n
/// (view axes) is fixed to the rotating planet, so it was seen along
/// n' ∝ M n + t / D, with M the previous view axes against the current ones
/// through the body frame, and t the eye's move in the previous view axes.
/// `prev` and `cur` are (body axes in view components, eye body-fixed km).
fn terrain_reprojection(
    prev: Option<([kerr::vec3::V3; 3], kerr::vec3::V3)>,
    cur: Option<([kerr::vec3::V3; 3], kerr::vec3::V3)>,
) -> [[f32; 4]; 4] {
    let mut m = [[0.0f32; 4]; 4];
    let (Some((bp, ep)), Some((bc, ec))) = (prev, cur) else {
        for (a, row) in m.iter_mut().take(3).enumerate() {
            row[a] = 1.0;
        }
        return m;
    };
    let moved = kerr::vec3::sub(ec, ep);
    for (a, row) in m.iter_mut().take(3).enumerate() {
        for (b, v) in row.iter_mut().take(3).enumerate() {
            *v = (0..3).map(|k| bp[k][a] * bc[k][b]).sum::<f64>() as f32;
        }
        row[3] = (0..3).map(|k| bp[k][a] * moved[k]).sum::<f64>() as f32;
    }
    m
}

#[cfg(test)]
mod terrain_tests {
    use super::*;
    use kerr::vec3::{self, V3};

    /// A fixed ground point seen from two eye positions and orientations:
    /// the reprojection takes its current direction to its previous one.
    #[test]
    fn terrain_reprojection_follows_a_fixed_point() {
        // Body axes in view components: a rotation, then a slightly turned one.
        let axes = |a: f64| -> [V3; 3] {
            let (s, c) = a.sin_cos();
            [[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]]
        };
        let (bp, bc) = (axes(0.3), axes(0.31));
        let (ep, ec) = ([6400.0, 1.0, 2.0], [6400.0, 1.002, 2.001]);
        let point: V3 = [6399.99, 1.05, 2.0];
        let m = terrain_reprojection(Some((bp, ep)), Some((bc, ec)));
        // View-frame direction of a body-fixed offset: components along the
        // view axes are dot products with the body axes' view components.
        let view = |b: &[V3; 3], d: V3| -> V3 { std::array::from_fn(|c| (0..3).map(|k| b[k][c] * d[k]).sum()) };
        let now = view(&bc, vec3::sub(point, ec));
        let then = view(&bp, vec3::sub(point, ep));
        let d = vec3::norm(now);
        let n = vec3::normalize(now);
        let np: V3 = std::array::from_fn(|a| (0..3).map(|b| m[a][b] as f64 * n[b]).sum::<f64>() + m[a][3] as f64 / d);
        let err = vec3::norm(vec3::sub(vec3::normalize(np), vec3::normalize(then)));
        assert!(err < 1e-5, "{err}");
    }
}
