//! Terrain on the ray-tracing hardware.
//!
//! Each resident tile's mesh (every [`MESH_STEP`] samples, so triangles are
//! ~6 px across where samples are 3 px; shading reads the full heights) is a
//! BLAS, built when the tile is generated. Vertices are relative to the
//! tile's own centre point on the datum sphere, in body-fixed axes, so a
//! BLAS never depends on the noise anchor. The TLAS, rebuilt each frame,
//! places the drawn tiles at `centre − anchor` (computed in f64 and small
//! near the eye), so rays from the eye stay precise.
//!
//! Skirts hang below each tile's edges so rays can't slip between tiles of
//! different levels.
//!
//! Plants ([`super::plants`]) are instances in the same TLAS: a BLAS per
//! variant, flagged in their custom data ([`PLANT_FLAG`], with their first
//! triangle in the normals table the shader reads) and masked apart
//! ([`PLANT_MASK`]), so ground queries see the tiles alone.

use super::tilegen::TILE_TEXELS;
use super::tiles::{TILE_SAMPLES, TileId};
use crate::shaders;
use bytemuck::{Pod, Zeroable};
use kerr::vec3::{self, V3};
use wgpu::util::DeviceExt;

/// Samples per mesh vertex step. Must match `TILE_MESH_STEP`.
pub const MESH_STEP: u32 = 2;
/// Quads per side of a tile's mesh. Must match `TILE_MESH_N`.
pub const MESH_N: u32 = TILE_SAMPLES / MESH_STEP;
/// Grid vertices, then four skirts of `MESH_N + 1`. Must match
/// `TILE_MESH_VERTS`.
pub const GRID_VERTS: u32 = (MESH_N + 1) * (MESH_N + 1);
pub const MESH_VERTS: u32 = GRID_VERTS + 4 * (MESH_N + 1);
/// Triangles of the grid; the skirts' come after them.
pub const GRID_TRIANGLES: u32 = 2 * MESH_N * MESH_N;
/// Instance masks: tiles, plants. Must match `terrain_rq.wgsl` and
/// `tile_cast.wgsl`.
pub const TILE_MASK: u8 = 0x01;
pub const PLANT_MASK: u8 = 0x02;
/// A plant instance's custom data: this flag and its variant's first
/// triangle in the normals table. Must match `TR_PLANT_FLAG`.
pub const PLANT_FLAG: u32 = 1 << 23;
/// Plant instances in the TLAS at most.
pub const MAX_PLANTS: u32 = 4096;
/// Palm variants (seeds 1…), then tufts of grass.
pub const PALM_VARIANTS: u32 = 4;
pub const TUFT_VARIANTS: u32 = 2;

/// A plant to place: its variant, its foot (body-fixed km from the
/// planet's centre), up there, and the horizontal way it leans.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlantInstance {
    pub variant: u32,
    pub foot_km: V3,
    pub up: V3,
    pub lean: V3,
}

impl PlantInstance {
    /// Its axes (lean, side, up) at wall time `t` (s), swaying in the
    /// trade wind (from the east): about the horizontal square to the wind,
    /// palms ~0.7° at 0.25 Hz, grass ~4° at 0.7 Hz, each in its own phase.
    pub fn axes(&self, t: f64) -> [V3; 3] {
        let grass = self.variant >= PALM_VARIANTS;
        let (amp, freq) = if grass { (0.07, 0.7) } else { (0.012, 0.25) };
        let phase = (vec3::dot(self.foot_km, [12.9898, 78.233, 37.719]) * 43.758).fract() * std::f64::consts::TAU;
        let gust = 1.0 + 0.5 * (0.13 * t + phase).sin();
        let angle = amp * gust * (std::f64::consts::TAU * freq * t + phase).sin();
        let east = vec3::cross([0.0, 0.0, 1.0], self.up);
        let axis = if vec3::norm(east) > 1e-9 { vec3::normalize(vec3::cross(self.up, east)) } else { self.lean };
        let (x, z) = (vec3::rotate(self.lean, axis, angle), vec3::rotate(self.up, axis, angle));
        [x, vec3::cross(z, x), z]
    }
}

/// The plants' meshes on the GPU.
pub struct PlantGeometry {
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    /// Per triangle: normal (object space) and part, for the shader.
    pub normals: wgpu::Buffer,
    /// Per variant: first vertex, vertices, first index, indices.
    ranges: Vec<[u32; 4]>,
    first_triangle: Vec<u32>,
    blas: Vec<wgpu::Blas>,
    built: bool,
}

impl PlantGeometry {
    fn new(device: &wgpu::Device) -> Self {
        let (mut positions, mut indices, mut normals) = (Vec::new(), Vec::new(), Vec::new());
        let (mut ranges, mut first_triangle) = (Vec::new(), Vec::new());
        let meshes = (0..PALM_VARIANTS)
            .map(|v| super::plants::palm(v as u64 + 1))
            .chain((0..TUFT_VARIANTS).map(|v| super::plants::grass_tuft(v as u64 + 1)));
        for m in meshes {
            ranges.push([
                positions.len() as u32,
                m.positions.len() as u32,
                indices.len() as u32,
                m.indices.len() as u32,
            ]);
            first_triangle.push(normals.len() as u32);
            positions.extend(m.positions);
            indices.extend(m.indices);
            normals.extend(m.normals);
        }
        let buffer = |label, contents: &[u8], usage| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents, usage })
        };
        let blas = ranges
            .iter()
            .map(|r| {
                device.create_blas(
                    &wgpu::CreateBlasDescriptor {
                        label: Some("plant"),
                        flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                        update_mode: wgpu::AccelerationStructureUpdateMode::Build,
                    },
                    wgpu::BlasGeometrySizeDescriptors::Triangles { descriptors: vec![Self::size(r)] },
                )
            })
            .collect();
        Self {
            vertices: buffer("plant vertices", bytemuck::cast_slice(&positions), wgpu::BufferUsages::BLAS_INPUT),
            indices: buffer("plant indices", bytemuck::cast_slice(&indices), wgpu::BufferUsages::BLAS_INPUT),
            normals: buffer("plant normals", bytemuck::cast_slice(&normals), wgpu::BufferUsages::STORAGE),
            ranges,
            first_triangle,
            blas,
            built: false,
        }
    }

    fn size(r: &[u32; 4]) -> wgpu::BlasTriangleGeometrySizeDescriptor {
        wgpu::BlasTriangleGeometrySizeDescriptor {
            vertex_format: wgpu::VertexFormat::Float32x3,
            vertex_count: r[1],
            index_format: Some(wgpu::IndexFormat::Uint32),
            index_count: Some(r[3]),
            flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
        }
    }
}
const VERTEX_STRIDE: u64 = 16;

/// The shared index list: grid quads as two triangles, (i, j), (i+1, j),
/// (i+1, j+1) and (i, j), (i+1, j+1), (i, j+1); then each skirt's quads
/// between an edge and its lowered copy (west, east, south, north).
pub fn mesh_indices() -> Vec<u32> {
    let n1 = MESH_N + 1;
    let g = |i: u32, j: u32| j * n1 + i;
    let mut out = Vec::with_capacity((GRID_TRIANGLES as usize + 8 * MESH_N as usize) * 3);
    for j in 0..MESH_N {
        for i in 0..MESH_N {
            out.extend([g(i, j), g(i + 1, j), g(i + 1, j + 1), g(i, j), g(i + 1, j + 1), g(i, j + 1)]);
        }
    }
    // Each skirt: the edge's vertices in order along it, and their copies.
    let edges: [Box<dyn Fn(u32) -> u32>; 4] =
        [Box::new(|k| g(0, k)), Box::new(|k| g(MESH_N, k)), Box::new(|k| g(k, 0)), Box::new(|k| g(k, MESH_N))];
    for (e, edge) in edges.iter().enumerate() {
        let skirt = |k: u32| GRID_VERTS + e as u32 * n1 + k;
        for k in 0..MESH_N {
            out.extend([edge(k), edge(k + 1), skirt(k + 1), edge(k), skirt(k + 1), skirt(k)]);
        }
    }
    out
}

/// Where a hit on primitive `prim` with barycentrics `(u, v)` lies in its
/// tile, as fractions (s, t) ∈ [0, 1]² (a skirt hit maps to its edge).
pub fn hit_st(prim: u32, bary: [f32; 2]) -> [f32; 2] {
    let n = MESH_N as f32;
    if prim < GRID_TRIANGLES {
        let quad = prim / 2;
        let (i, j) = ((quad % MESH_N) as f32, (quad / MESH_N) as f32);
        let (u, v) = (bary[0], bary[1]);
        // Corners of the triangle in grid units, weighted (1 − u − v, u, v).
        let (x, y) = if prim.is_multiple_of(2) { (i + u + v, j + v) } else { (i + u, j + u + v) };
        return [x / n, y / n];
    }
    let k = prim - GRID_TRIANGLES;
    let (edge, along) = (k / (2 * MESH_N), ((k % (2 * MESH_N)) / 2) as f32 + 0.5);
    match edge {
        0 => [0.0, along / n],
        1 => [1.0, along / n],
        2 => [along / n, 0.0],
        _ => [along / n, 1.0],
    }
}

/// A ray's hit: distance (km), atlas layer, and place in the tile.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CastHit {
    pub t_km: f32,
    pub layer: u32,
    pub st: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct CastParams {
    count: u32,
    pad: [u32; 3],
}

pub struct TerrainAccel {
    /// Every layer's mesh: `MESH_VERTS` vertices (xyz km, w unused).
    pub vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    index_count: u32,
    blas: Vec<Option<wgpu::Blas>>,
    pub tlas: wgpu::Tlas,
    pub plants: PlantGeometry,
    /// Instances in the TLAS as last built.
    pub instances: usize,
    cast_pipeline: wgpu::ComputePipeline,
    cast_layout: wgpu::BindGroupLayout,
}

impl TerrainAccel {
    pub fn new(device: &wgpu::Device, layers: u32) -> Self {
        let vertices = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("terrain mesh"),
            size: layers as u64 * MESH_VERTS as u64 * VERTEX_STRIDE,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::BLAS_INPUT,
            mapped_at_creation: false,
        });
        let list = mesh_indices();
        let indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("terrain mesh indices"),
            contents: bytemuck::cast_slice(&list),
            usage: wgpu::BufferUsages::BLAS_INPUT,
        });
        let tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
            label: Some("terrain"),
            max_instances: layers + MAX_PLANTS,
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tile cast"),
            source: wgpu::ShaderSource::Wgsl(shaders::tile_cast().into()),
        });
        let buffer_entry = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let cast_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tile cast"),
            entries: &[
                buffer_entry(1, wgpu::BufferBindingType::Uniform),
                buffer_entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, wgpu::BufferBindingType::Storage { read_only: false }),
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::AccelerationStructure { vertex_return: false },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tile cast"),
            bind_group_layouts: &[Some(&cast_layout)],
            immediate_size: 0,
        });
        let cast_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tile cast"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("cs_tile_cast"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self {
            vertices,
            indices,
            index_count: list.len() as u32,
            blas: (0..layers).map(|_| None).collect(),
            tlas,
            plants: PlantGeometry::new(device),
            instances: 0,
            cast_pipeline,
            cast_layout,
        }
    }

    fn size(&self) -> wgpu::BlasTriangleGeometrySizeDescriptor {
        wgpu::BlasTriangleGeometrySizeDescriptor {
            vertex_format: wgpu::VertexFormat::Float32x3,
            vertex_count: MESH_VERTS,
            index_format: Some(wgpu::IndexFormat::Uint32),
            index_count: Some(self.index_count),
            flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
        }
    }

    /// Build the BLASes of freshly meshed `layers` (after their mesh
    /// dispatches in `enc`).
    pub fn build_blas(&mut self, device: &wgpu::Device, enc: &mut wgpu::CommandEncoder, layers: &[u32]) {
        if layers.is_empty() {
            return;
        }
        let size = self.size();
        for &layer in layers {
            self.blas[layer as usize] = Some(device.create_blas(
                &wgpu::CreateBlasDescriptor {
                    label: Some("terrain tile"),
                    flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                    update_mode: wgpu::AccelerationStructureUpdateMode::Build,
                },
                wgpu::BlasGeometrySizeDescriptors::Triangles { descriptors: vec![size.clone()] },
            ));
        }
        let entries: Vec<wgpu::BlasBuildEntry> = layers
            .iter()
            .filter_map(|&layer| {
                let blas = self.blas[layer as usize].as_ref()?;
                Some(wgpu::BlasBuildEntry {
                    blas,
                    geometry: wgpu::BlasGeometries::TriangleGeometries(vec![wgpu::BlasTriangleGeometry {
                        size: &size,
                        vertex_buffer: &self.vertices,
                        first_vertex: layer * MESH_VERTS,
                        vertex_stride: VERTEX_STRIDE,
                        index_buffer: Some(&self.indices),
                        first_index: Some(0),
                        transform_buffer: None,
                        transform_buffer_offset: None,
                    }]),
                })
            })
            .collect();
        enc.build_acceleration_structures(entries.iter(), std::iter::empty());
    }

    /// Place the `drawn` tiles (with their layers) of a planet of
    /// `radius_km` and the `plants` relative to `anchor_km` and rebuild the
    /// TLAS.
    pub fn build_tlas(
        &mut self,
        enc: &mut wgpu::CommandEncoder,
        (drawn, plants, time_s): (&[(TileId, u32)], &[PlantInstance], f64),
        radius_km: f64,
        anchor_km: V3,
    ) {
        let mut n = 0;
        for &(tile, layer) in drawn {
            let Some(blas) = &self.blas[layer as usize] else { continue };
            let at = vec3::sub(vec3::scale(tile.centre(), radius_km), anchor_km);
            let transform = [
                1.0,
                0.0,
                0.0,
                at[0] as f32, //
                0.0,
                1.0,
                0.0,
                at[1] as f32, //
                0.0,
                0.0,
                1.0,
                at[2] as f32,
            ];
            self.tlas[n] = Some(wgpu::TlasInstance::new(blas, transform, layer, TILE_MASK));
            n += 1;
        }
        // Plants: metres in their meshes, km here; turned so the mesh's x
        // leans their way and z is up.
        let g = &self.plants;
        for p in plants.iter().take(MAX_PLANTS as usize) {
            let [x, y, z] = p.axes(time_s);
            let at = vec3::sub(p.foot_km, anchor_km);
            let s = 1e-3;
            let transform = std::array::from_fn(|k| {
                let (row, col) = (k / 4, k % 4);
                (match col {
                    0 => x[row] * s,
                    1 => y[row] * s,
                    2 => z[row] * s,
                    _ => at[row],
                }) as f32
            });
            let v = p.variant as usize % g.blas.len();
            self.tlas[n] =
                Some(wgpu::TlasInstance::new(&g.blas[v], transform, PLANT_FLAG | g.first_triangle[v], PLANT_MASK));
            n += 1;
        }
        for i in n..self.instances {
            self.tlas[i] = None;
        }
        self.instances = n;
        // The plants' BLASes, once, with the first TLAS.
        let sizes: Vec<_> = g.ranges.iter().map(PlantGeometry::size).collect();
        let entries: Vec<wgpu::BlasBuildEntry> = if g.built {
            Vec::new()
        } else {
            g.blas
                .iter()
                .zip(&g.ranges)
                .zip(&sizes)
                .map(|((blas, r), size)| wgpu::BlasBuildEntry {
                    blas,
                    geometry: wgpu::BlasGeometries::TriangleGeometries(vec![wgpu::BlasTriangleGeometry {
                        size,
                        vertex_buffer: &g.vertices,
                        first_vertex: r[0],
                        vertex_stride: 12,
                        index_buffer: Some(&g.indices),
                        first_index: Some(r[2]),
                        transform_buffer: None,
                        transform_buffer_offset: None,
                    }]),
                })
                .collect()
        };
        enc.build_acceleration_structures(entries.iter(), [&self.tlas]);
        self.plants.built = true;
    }

    /// Cast rays (origin km from the anchor, unit direction, length km)
    /// against the TLAS as last built; blocking (tests, ground queries).
    pub fn cast(&self, device: &wgpu::Device, queue: &wgpu::Queue, rays: &[(V3, V3, f64)]) -> Vec<Option<CastHit>> {
        let pending = self.cast_submit(device, queue, rays);
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        pending.take(device).unwrap_or_else(|| vec![None; rays.len()])
    }

    /// Cast rays as [`TerrainAccel::cast`] does, without waiting: the hits
    /// come from [`PendingCast::take`] a frame or two later.
    pub fn cast_submit(&self, device: &wgpu::Device, queue: &wgpu::Queue, rays: &[(V3, V3, f64)]) -> PendingCast {
        let ready = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        if rays.is_empty() {
            ready.store(true, std::sync::atomic::Ordering::Release);
            let readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("tile cast readback"),
                size: 32,
                usage: wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            return PendingCast { readback, count: 0, ready };
        }
        let f = |v: V3, w: f64| [v[0] as f32, v[1] as f32, v[2] as f32, w as f32];
        let data: Vec<[f32; 4]> = rays.iter().flat_map(|&(o, d, len)| [f(o, len), f(d, 0.0)]).collect();
        let size = (rays.len() * 32) as u64;
        use wgpu::BufferUsages as U;
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tile cast params"),
            contents: bytemuck::bytes_of(&CastParams { count: rays.len() as u32, pad: [0; 3] }),
            usage: U::UNIFORM,
        });
        let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tile cast rays"),
            contents: bytemuck::cast_slice(&data),
            usage: U::STORAGE,
        });
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tile cast hits"),
            size,
            usage: U::STORAGE | U::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tile cast readback"),
            size,
            usage: U::MAP_READ | U::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tile cast"),
            layout: &self.cast_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 1, resource: params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: input.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: output.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::AccelerationStructure(&self.tlas) },
            ],
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("tile cast") });
        {
            let mut pass = enc
                .begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("tile cast"), timestamp_writes: None });
            pass.set_pipeline(&self.cast_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((rays.len() as u32).div_ceil(64), 1, 1);
        }
        enc.copy_buffer_to_buffer(&output, 0, &readback, 0, size);
        queue.submit([enc.finish()]);
        let flag = ready.clone();
        readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| flag.store(r.is_ok(), std::sync::atomic::Ordering::Release));
        PendingCast { readback, count: rays.len(), ready }
    }
}

/// Rays cast without waiting ([`TerrainAccel::cast_submit`]).
pub struct PendingCast {
    readback: wgpu::Buffer,
    count: usize,
    ready: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl PendingCast {
    /// The hits, once the GPU has them (polling without waiting).
    pub fn take(&self, device: &wgpu::Device) -> Option<Vec<Option<CastHit>>> {
        if self.count == 0 {
            return Some(Vec::new());
        }
        let _ = device.poll(wgpu::PollType::Poll);
        if !self.ready.load(std::sync::atomic::Ordering::Acquire) {
            return None;
        }
        let hits = {
            let mapped = self.readback.slice(..).get_mapped_range().ok()?;
            let v: &[[f32; 4]] = bytemuck::cast_slice(&mapped);
            (0..self.count)
                .map(|i| {
                    let (a, b) = (v[2 * i], v[2 * i + 1]);
                    (a[0] >= 0.0).then(|| CastHit {
                        t_km: a[0],
                        layer: a[1].to_bits(),
                        st: hit_st(a[2].to_bits(), [a[3], b[0]]),
                    })
                })
                .collect()
        };
        self.readback.unmap();
        Some(hits)
    }
}

/// Texel (x, y) in a tile's atlas layer of fractions (s, t) across it.
pub fn st_texel(st: [f32; 2]) -> [f32; 2] {
    let apron = ((TILE_TEXELS - TILE_SAMPLES - 1) / 2) as f32;
    st.map(|c| c * TILE_SAMPLES as f32 + apron)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indices_cover_grid_and_skirts() {
        let idx = mesh_indices();
        assert_eq!(idx.len(), 3 * (GRID_TRIANGLES + 8 * MESH_N) as usize);
        assert!(idx.iter().all(|&i| i < MESH_VERTS));
        // Every vertex is used.
        let mut used = vec![false; MESH_VERTS as usize];
        for &i in &idx {
            used[i as usize] = true;
        }
        assert!(used.iter().all(|&u| u));
    }

    /// `hit_st` inverts the index layout: a triangle's barycentric corners
    /// land on its grid vertices.
    #[test]
    fn hits_map_back_into_the_tile() {
        let idx = mesh_indices();
        let n1 = MESH_N + 1;
        for prim in [0, 1, 2, 777, 778, GRID_TRIANGLES - 1] {
            for (k, bary) in [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]].iter().enumerate() {
                let vtx = idx[3 * prim as usize + k];
                let want = [(vtx % n1) as f32 / MESH_N as f32, (vtx / n1) as f32 / MESH_N as f32];
                let got = hit_st(prim, *bary);
                assert!((got[0] - want[0]).abs() < 1e-6 && (got[1] - want[1]).abs() < 1e-6, "{prim} {k}");
            }
        }
        // Skirts map onto their edges.
        let west = hit_st(GRID_TRIANGLES + 5, [0.3, 0.3]);
        assert_eq!(west[0], 0.0);
        let north = hit_st(GRID_TRIANGLES + 6 * MESH_N + 3, [0.3, 0.3]);
        assert_eq!(north[1], 1.0);
    }

    #[test]
    fn texels_of_fractions() {
        assert_eq!(st_texel([0.0, 1.0]), [2.0, 130.0]);
    }
}
