//! Tile generation: where each sample of a tile lies, precisely enough for
//! centimetre terrain.
//!
//! At level 22 a tile's samples are ~4 × 10⁻⁹ apart in face coordinates,
//! below what f32 resolves there (~3 × 10⁻⁸), so the GPU can't take a fine
//! tile's sample directions from `cube` in f32. Instead the CPU expands the
//! tile's points about its centre in f64, to second order in the tile
//! coordinates `(s, t) ∈ [−½, ½]²` (plus the apron):
//!
//! `R·d(s, t) − anchor ≈ o + a_s s + a_t t + ½ b_ss s² + b_st s t + ½ b_tt t²`
//!
//! with `o = R·d(0, 0) − anchor` and the coefficients from central
//! differences across the tile (all in km). The GPU then only adds small
//! numbers. The third-order remainder is ~R θ³/6 for a tile subtending θ:
//! 0.07 mm at level 12, so levels from [`EXPANDED_FROM`] use the expansion
//! and coarser tiles take their directions from `cube` directly.

use super::anchor::{Anchor, OctaveGpu};
use super::atlas::Atlas;
use super::maps::MapKey;
use super::tiles::{MAX_LEVEL, TILE_SAMPLES, TileId};
use crate::shaders;
use bytemuck::{Pod, Zeroable};
use kerr::planets::Planet;
use kerr::vec3::{self, V3};

/// Tiles from this level are generated from the expansion.
pub const EXPANDED_FROM: u8 = 12;

/// Mirrors the GPU's tile frame: the expansion's coefficients in km, xyz
/// (w unused).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TileFrameGpu {
    pub origin: [f32; 4],
    pub a_s: [f32; 4],
    pub a_t: [f32; 4],
    pub b_ss: [f32; 4],
    pub b_st: [f32; 4],
    pub b_tt: [f32; 4],
}

/// Tile `t`'s expansion on a sphere of `radius_km`, relative to `anchor_km`
/// (body-fixed).
pub fn frame(t: TileId, radius_km: f64, anchor_km: V3) -> TileFrameGpu {
    // Points at tile coordinates (s, t) ∈ [−½, ½]² about the centre.
    let p = |s: f64, u: f64| vec3::scale(t.direction(0.5 + s, 0.5 + u), radius_km);
    let c = p(0.0, 0.0);
    let (e, w, n, s) = (p(0.5, 0.0), p(-0.5, 0.0), p(0.0, 0.5), p(0.0, -0.5));
    let (ne, nw, se, sw) = (p(0.5, 0.5), p(-0.5, 0.5), p(0.5, -0.5), p(-0.5, -0.5));
    let f = |v: V3| [v[0] as f32, v[1] as f32, v[2] as f32, 0.0];
    TileFrameGpu {
        origin: f(vec3::sub(c, anchor_km)),
        a_s: f(vec3::sub(e, w)),
        a_t: f(vec3::sub(n, s)),
        // (e − 2c + w)/h² with h = ½, and ½ of it is the s² coefficient.
        b_ss: f(vec3::scale(vec3::add(vec3::sub(e, vec3::scale(c, 2.0)), w), 4.0)),
        b_st: f(vec3::sub(vec3::add(ne, sw), vec3::add(nw, se))),
        b_tt: f(vec3::scale(vec3::add(vec3::sub(n, vec3::scale(c, 2.0)), s), 4.0)),
    }
}

/// The point at tile coordinates `(s, t)` (centred: −½ to ½), in km from
/// the anchor, computed in f32 as the GPU does.
pub fn offset(f: &TileFrameGpu, s: f32, t: f32) -> [f32; 3] {
    std::array::from_fn(|i| {
        f.origin[i]
            + (f.a_s[i] * s + f.a_t[i] * t)
            + (0.5 * f.b_ss[i] * s * s + f.b_st[i] * s * t + 0.5 * f.b_tt[i] * t * t)
    })
}

// --- The generator on the GPU (`wgsl/tile_gen.wgsl`) ------------------------

/// Texels per side of a tile in the atlas: the 129 sample points (edges
/// shared with the neighbours) and a two-texel apron (the parent's cubic
/// reaches two samples out). Must match `TILE_TEXELS` in the shader.
pub const APRON: u32 = 2;
pub const TILE_TEXELS: u32 = TILE_SAMPLES + 1 + 2 * APRON;

/// Atlas layers: 133² × 8 bytes ≈ 140 KB a tile, ~140 MB in all (fewer if
/// the device allows fewer array layers).
pub const ATLAS_LAYERS: u32 = 1024;

/// Most tiles generated per call.
pub const MAX_JOBS: usize = 64;

/// Jobs sit in one uniform buffer at this stride (dynamic offsets).
const JOB_STRIDE: u64 = 256;

/// Nominal sample spacing (km) at `level` on a planet of `radius_km`, as
/// the shader's `tile_spacing`.
pub fn spacing_km(radius_km: f64, level: u8) -> f64 {
    radius_km * std::f64::consts::FRAC_PI_2 / ((1u64 << level) as f64 * TILE_SAMPLES as f64)
}

/// The anchored octaves the refined levels add: one per level from
/// [`EXPANDED_FROM`], at a wavelength of four samples.
pub fn octaves(radius_km: f64, seed: u32) -> impl Iterator<Item = (f64, u32)> {
    (EXPANDED_FROM..=MAX_LEVEL)
        .map(move |level| (1.0 / (4.0 * spacing_km(radius_km, level)), seed.wrapping_add(2000 + level as u32)))
}

/// Mirrors `struct TileGenParams` in `tile_gen.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct ParamsGpu {
    kind: u32,
    seed: u32,
    air: u32,
    octave_count: u32,
    radius: f32,
    relief: f32,
    sea: f32,
    t_eq: f32,
    octaves: [OctaveGpu; 16],
}

/// Mirrors `struct TileJob`, padded to the job stride.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct JobGpu {
    frame: TileFrameGpu,
    /// face, level, x, y
    tile: [u32; 4],
    /// layer, parent's layer, 1 to refine the parent, unused
    slots: [u32; 4],
    pad: [u32; 32],
}

/// What the tiles are of: a planet (identified by `key`) and the anchor of
/// its fine noise.
#[derive(Clone, Copy)]
pub struct Surface<'a> {
    pub key: MapKey,
    pub planet: &'a Planet,
    pub anchor: &'a Anchor,
}

/// The tile atlas and its generator: heights (km, R32Float) and packed
/// material channels (R32Uint) per layer, and each tile's height range.
pub struct TileGen {
    pub height: wgpu::Texture,
    pub material: wgpu::Texture,
    /// Per layer the lowest and highest height inside the tile (mm, i32).
    pub range: wgpu::Buffer,
    params: wgpu::Buffer,
    jobs: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    pub atlas: Atlas,
    /// The planet the atlas holds.
    planet: Option<MapKey>,
}

impl TileGen {
    pub fn new(device: &wgpu::Device) -> Self {
        let layers = ATLAS_LAYERS.min(device.limits().max_texture_array_layers);
        let array = |label, format| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: TILE_TEXELS, height: TILE_TEXELS, depth_or_array_layers: layers },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            })
        };
        let height = array("tile heights", wgpu::TextureFormat::R32Float);
        let material = array("tile materials", wgpu::TextureFormat::R32Uint);
        let buffer = |label, size, usage| {
            device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage, mapped_at_creation: false })
        };
        use wgpu::BufferUsages as U;
        let range = buffer("tile ranges", layers as u64 * 8, U::STORAGE | U::COPY_DST | U::COPY_SRC);
        let params = buffer("tile gen params", std::mem::size_of::<ParamsGpu>() as u64, U::UNIFORM | U::COPY_DST);
        let jobs = buffer("tile jobs", JOB_STRIDE * MAX_JOBS as u64, U::UNIFORM | U::COPY_DST);

        let storage_texture = |binding, format| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::ReadWrite,
                format,
                view_dimension: wgpu::TextureViewDimension::D2Array,
            },
            count: None,
        };
        let uniform = |binding, dynamic| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: dynamic,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tile gen"),
            entries: &[
                uniform(1, false),
                uniform(2, true),
                storage_texture(3, wgpu::TextureFormat::R32Float),
                storage_texture(4, wgpu::TextureFormat::R32Uint),
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let view = |t: &wgpu::Texture| {
            t.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            })
        };
        let (height_view, material_view) = (view(&height), view(&material));
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tile gen"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 1, resource: params.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &jobs,
                        offset: 0,
                        size: std::num::NonZeroU64::new(std::mem::size_of::<JobGpu>() as u64),
                    }),
                },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&height_view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&material_view) },
                wgpu::BindGroupEntry { binding: 5, resource: range.as_entire_binding() },
            ],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tile gen"),
            source: wgpu::ShaderSource::Wgsl(shaders::tile_gen().into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tile gen"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tile gen"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("cs_tile_gen"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self { height, material, range, params, jobs, pipeline, bind_group, atlas: Atlas::new(layers), planet: None }
    }

    /// Generate `tiles` of `surface` (a new planet empties the atlas):
    /// coarse to fine, each refined
    /// tile's parent resident or earlier in the list, at most [`MAX_JOBS`].
    /// Returns the tiles generated and their layers (tiles whose parent is
    /// missing, or that find no free layer, are skipped).
    pub fn generate(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface: Surface,
        tiles: &[TileId],
        profiler: Option<&mut crate::profile::Profiler>,
    ) -> Vec<(TileId, u32)> {
        let Surface { key, planet, anchor } = surface;
        if self.planet != Some(key) {
            self.atlas = Atlas::new(self.atlas.capacity());
            self.planet = Some(key);
        }
        let mut octaves = [OctaveGpu::default(); 16];
        for (slot, o) in octaves.iter_mut().zip(&anchor.octaves) {
            *slot = *o;
        }
        let params = ParamsGpu {
            kind: planet.kind.index(),
            seed: planet.seed,
            air: planet.atmosphere.is_some() as u32,
            octave_count: anchor.octaves.len().min(16) as u32,
            radius: planet.radius_km as f32,
            relief: planet.relief_km as f32,
            sea: planet.sea_level as f32,
            t_eq: planet.equilibrium_temperature as f32,
            octaves,
        };
        queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&params));

        // Parents in use this frame mustn't be evicted by the new tiles.
        for t in tiles {
            if let Some(p) = t.parent() {
                self.atlas.get(p);
            }
        }
        let mut sorted = tiles.to_vec();
        sorted.sort_by_key(|t| t.level);
        let mut done = Vec::new();
        let mut jobs: Vec<JobGpu> = Vec::new();
        for t in sorted.into_iter().take(MAX_JOBS) {
            let refine = t.level >= EXPANDED_FROM;
            let parent_layer = match t.parent() {
                Some(p) if refine => match self.atlas.get(p) {
                    Some(layer) => layer,
                    None => continue,
                },
                _ => 0,
            };
            let Some((layer, _)) = self.atlas.insert(t) else { continue };
            jobs.push(JobGpu {
                frame: if refine { frame(t, planet.radius_km, anchor.origin_km) } else { TileFrameGpu::default() },
                tile: [t.face as u32, t.level as u32, t.x, t.y],
                slots: [layer, parent_layer, refine as u32, 0],
                pad: [0; 32],
            });
            done.push((t, layer));
        }
        for (i, job) in jobs.iter().enumerate() {
            queue.write_buffer(&self.jobs, i as u64 * JOB_STRIDE, bytemuck::bytes_of(job));
            queue.write_buffer(&self.range, job.slots[0] as u64 * 8, bytemuck::cast_slice(&[i32::MAX, i32::MIN]));
        }
        if jobs.is_empty() {
            return done;
        }
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("tile gen") });
        {
            let timestamp_writes = profiler.and_then(|p| p.compute("terrain tiles"));
            let mut pass =
                enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("tile gen"), timestamp_writes });
            pass.set_pipeline(&self.pipeline);
            let groups = TILE_TEXELS.div_ceil(8);
            // One dispatch per tile, coarse first: each dispatch sees the
            // layers the ones before it wrote.
            for i in 0..jobs.len() {
                pass.set_bind_group(0, &self.bind_group, &[(i as u64 * JOB_STRIDE) as u32]);
                pass.dispatch_workgroups(groups, groups, 1);
            }
        }
        queue.submit([enc.finish()]);
        done
    }

    /// A layer's heights (km), row by row, `TILE_TEXELS`² (blocking: for
    /// tools and tests).
    pub fn read_heights(&self, device: &wgpu::Device, queue: &wgpu::Queue, layer: u32) -> Vec<f32> {
        let row = (TILE_TEXELS * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tile readback"),
            size: (row * TILE_TEXELS) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("tile readback") });
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.height,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: layer },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: None },
            },
            wgpu::Extent3d { width: TILE_TEXELS, height: TILE_TEXELS, depth_or_array_layers: 1 },
        );
        queue.submit([enc.finish()]);
        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let mut out = Vec::with_capacity((TILE_TEXELS * TILE_TEXELS) as usize);
        if let Ok(data) = slice.get_mapped_range() {
            let values: &[f32] = bytemuck::cast_slice(&data);
            for y in 0..TILE_TEXELS {
                let start = (y * row / 4) as usize;
                out.extend_from_slice(&values[start..start + TILE_TEXELS as usize]);
            }
        }
        out
    }

    /// The height at sample (i, j) of a layer read with [`Self::read_heights`]
    /// (i, j = 0 … 128 inside the tile).
    pub fn at(heights: &[f32], i: i32, j: i32) -> f32 {
        let k = |c: i32| (c + APRON as i32) as usize;
        heights[k(j) * TILE_TEXELS as usize + k(i)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::cube;

    const RADIUS: f64 = 7234.0;

    /// Near the anchor, every sample (apron included) lands within a
    /// millimetre of the exact f64 point, at every expanded level.
    #[test]
    fn expansion_is_millimetre_exact() {
        let site = cube::direction(3, 0.41, -0.27);
        let half = 0.5 + 1.0 / TILE_SAMPLES as f64;
        for level in EXPANDED_FROM..=MAX_LEVEL {
            let tile = TileId::containing(site, level);
            let anchor = vec3::scale(site, RADIUS + 0.002);
            let fr = frame(tile, RADIUS, anchor);
            let mut worst = 0.0f64;
            for i in 0..=8 {
                for j in 0..=8 {
                    let (s, t) = (-half + 2.0 * half * i as f64 / 8.0, -half + 2.0 * half * j as f64 / 8.0);
                    let exact = vec3::sub(vec3::scale(tile.direction(0.5 + s, 0.5 + t), RADIUS), anchor);
                    let got = offset(&fr, s as f32, t as f32);
                    let err = vec3::norm(vec3::sub(exact, got.map(f64::from)));
                    worst = worst.max(err);
                }
            }
            assert!(worst < 1e-6, "level {level}: {:.3} mm", worst * 1e6);
        }
    }

    /// Neighbouring tiles put their shared edge's samples at the same
    /// points (to well under a sample spacing), so there are no seams.
    #[test]
    fn neighbours_agree_on_edges() {
        let site = cube::direction(0, 0.999_999, 0.2);
        for level in [EXPANDED_FROM, 18, MAX_LEVEL] {
            let a = TileId::containing(site, level);
            let b = a.neighbour(crate::terrain::tiles::Side::East);
            let anchor = vec3::scale(site, RADIUS);
            let (fa, fb) = (frame(a, RADIUS, anchor), frame(b, RADIUS, anchor));
            // a's east edge corners, found among b's corners.
            for t in [-0.5f32, 0.5] {
                let pa = offset(&fa, 0.5, t).map(f64::from);
                let best = [(-0.5f32, -0.5f32), (-0.5, 0.5), (0.5, -0.5), (0.5, 0.5)]
                    .iter()
                    .map(|&(s, u)| vec3::norm(vec3::sub(pa, offset(&fb, s, u).map(f64::from))))
                    .fold(f64::INFINITY, f64::min);
                let spacing = RADIUS * a.angular_size() / TILE_SAMPLES as f64;
                assert!(best < 1e-6 + 1e-3 * spacing, "level {level}: {best} km");
            }
        }
    }

    /// Mirror the shader's structs: `TileFrame` six vec4s, `TileJob` eight
    /// (padded to the stride), `TileGenParams` two vec4s and 16 octaves.
    #[test]
    fn layouts() {
        assert_eq!(std::mem::size_of::<TileFrameGpu>(), 96);
        assert_eq!(std::mem::size_of::<JobGpu>() as u64, JOB_STRIDE);
        assert_eq!(std::mem::offset_of!(JobGpu, slots), 112);
        assert_eq!(std::mem::size_of::<ParamsGpu>(), 32 + 16 * 32);
        assert_eq!(TILE_TEXELS, 133);
    }

    /// One octave per refined level, a wavelength of four samples.
    #[test]
    fn octaves_follow_the_levels() {
        let o: Vec<_> = octaves(RADIUS, 5).collect();
        assert_eq!(o.len(), (MAX_LEVEL - EXPANDED_FROM + 1) as usize);
        assert!(o.len() <= 16);
        for (k, &(freq, _)) in o.iter().enumerate() {
            let level = EXPANDED_FROM + k as u8;
            assert!((1.0 / freq - 4.0 * spacing_km(RADIUS, level)).abs() < 1e-12);
        }
        // Level 22's octave: centimetres.
        let lambda_m = 1000.0 / o[(22 - EXPANDED_FROM) as usize].0;
        assert!((0.05..0.12).contains(&lambda_m), "{lambda_m} m");
    }
}
