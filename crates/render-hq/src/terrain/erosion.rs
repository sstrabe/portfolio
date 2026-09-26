//! Regional erosion (`wgsl/erosion.wgsl`): a square of ~128 km around the
//! eye on worlds with seas, worn by running water over millions of years
//! (stream-power incision along the drainage and hillslope diffusion), so
//! old land has dendritic valleys and sharp ridges. A planet-wide map
//! would be far too coarse for an island (~20 km cells); this one is
//! 250 m. It's baked on the GPU in one go when a world comes up or the
//! eye has travelled [`REBAKE_KM`], and the tile generator adds its height
//! change inland (nothing below 15 m, so coastlines and beaches stay where
//! the relief puts them, and never cutting land below 40% of its height).

use super::maps::MapKey;
use crate::shaders;
use bytemuck::{Pod, Zeroable};
use kerr::planets::Planet;
use kerr::vec3::{self, V3};
use wgpu::util::DeviceExt;

/// Cells a side, and their size (m).
pub const REGION_N: u32 = 512;
pub const CELL_M: f64 = 250.0;
/// Baked again when the eye is this far (km) from the square's centre.
pub const REBAKE_KM: f64 = 40.0;
/// The run: steps of `DT_YEARS` (4 Myr in all), stream-power erodibility
/// and area exponent, hillslope diffusivity (m²/yr).
const STEPS: u32 = 200;
const DT_YEARS: f32 = 2.0e4;
const ERODIBILITY: f32 = 1.0e-6;
const AREA_EXPONENT: f32 = 0.5;
const DIFFUSIVITY: f32 = 0.3;

/// Mirrors `struct ErosionParams` in `erosion.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct ErosionParams {
    centre: [f32; 4],
    east: [f32; 4],
    north: [f32; 4],
    terrain: [u32; 4],
    planet: [f32; 4],
    stream: [f32; 4],
}

/// Mirrors `struct Region` in `tile_gen.wgsl`: where the square is (unit
/// centre and cell size km; unit east and cells a side, 0 for none; unit
/// north).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct RegionGpu {
    pub centre: [f32; 4],
    pub east: [f32; 4],
    pub north: [f32; 4],
    pub pad: [f32; 4],
}

pub struct RegionErosion {
    /// The height change (km) per cell, and where the square is.
    pub delta: wgpu::Buffer,
    pub region: wgpu::Buffer,
    params: wgpu::Buffer,
    groups: [wgpu::BindGroup; 2],
    pipelines: [wgpu::ComputePipeline; 5],
    /// The planet and centre (body-fixed unit direction) last baked.
    pub baked: Option<(MapKey, V3)>,
}

/// East and north along the ground at unit direction `c` (body-fixed, the
/// spin axis along z).
fn tangent_axes(c: V3) -> (V3, V3) {
    let e = vec3::cross([0.0, 0.0, 1.0], c);
    let east = if vec3::norm(e) > 1e-6 { vec3::normalize(e) } else { [1.0, 0.0, 0.0] };
    (east, vec3::cross(c, east))
}

impl RegionErosion {
    pub fn new(device: &wgpu::Device) -> Self {
        use wgpu::BufferUsages as U;
        let cells = (REGION_N * REGION_N) as u64;
        let buffer = |label, size: u64, usage| {
            device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage, mapped_at_creation: false })
        };
        let h =
            [buffer("erosion heights a", cells * 4, U::STORAGE), buffer("erosion heights b", cells * 4, U::STORAGE)];
        let area = [buffer("erosion area a", cells * 4, U::STORAGE), buffer("erosion area b", cells * 4, U::STORAGE)];
        let initial = buffer("erosion initial", cells * 4, U::STORAGE);
        let receiver = buffer("erosion receivers", cells * 4, U::STORAGE);
        let delta = buffer("erosion delta", cells * 4, U::STORAGE);
        let params = buffer("erosion params", std::mem::size_of::<ErosionParams>() as u64, U::UNIFORM | U::COPY_DST);
        let region = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("erosion region"),
            contents: bytemuck::bytes_of(&RegionGpu::default()),
            usage: U::UNIFORM | U::COPY_DST,
        });
        let storage = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("erosion"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage(2),
                storage(3),
                storage(4),
                storage(5),
                storage(6),
                storage(7),
                storage(8),
            ],
        });
        // Two groups swapping the ping-pong buffers.
        fn e(binding: u32, b: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry { binding, resource: b.as_entire_binding() }
        }
        let group = |g: usize| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("erosion"),
                layout: &layout,
                entries: &[
                    e(1, &params),
                    e(2, &h[g]),
                    e(3, &h[1 - g]),
                    e(4, &initial),
                    e(5, &receiver),
                    e(6, &area[g]),
                    e(7, &area[1 - g]),
                    e(8, &delta),
                ],
            })
        };
        let groups = [group(0), group(1)];
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("erosion"),
            source: wgpu::ShaderSource::Wgsl(shaders::erosion().into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("erosion"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let pipelines = [
            pipeline("cs_erosion_init"),
            pipeline("cs_erosion_receivers"),
            pipeline("cs_erosion_area"),
            pipeline("cs_erosion_step"),
            pipeline("cs_erosion_finish"),
        ];
        Self { delta, region, params, groups, pipelines, baked: None }
    }

    /// Whether the square should be baked (again) for an eye at body-fixed
    /// `eye_km` on `planet` (only worlds with seas are eroded so far).
    pub fn wanted(&self, key: MapKey, planet: &Planet, eye_km: V3) -> bool {
        if planet.kind != kerr::planets::PlanetKind::Ocean {
            return false;
        }
        match self.baked {
            Some((k, c)) => k != key || vec3::norm(vec3::sub(vec3::scale(c, planet.radius_km), eye_km)) > REBAKE_KM,
            None => true,
        }
    }

    /// Bake the square centred under body-fixed `eye_km` (submitted now;
    /// the GPU finishes it before the tiles that read it).
    pub fn bake(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, key: MapKey, planet: &Planet, eye_km: V3) {
        let c = vec3::normalize(eye_km);
        let (east, north) = tangent_axes(c);
        let f = |v: V3, w: f32| [v[0] as f32, v[1] as f32, v[2] as f32, w];
        let params = ErosionParams {
            centre: f(c, CELL_M as f32),
            east: f(east, REGION_N as f32),
            north: f(north, DT_YEARS),
            terrain: [planet.kind.index(), planet.seed, planet.atmosphere.is_some() as u32, 0],
            planet: [
                planet.radius_km as f32,
                planet.relief_km as f32,
                planet.sea_level as f32,
                planet.equilibrium_temperature as f32,
            ],
            stream: [ERODIBILITY, AREA_EXPONENT, DIFFUSIVITY, 0.0],
        };
        queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&params));
        let region = RegionGpu {
            centre: f(c, (CELL_M * 1e-3) as f32),
            east: f(east, REGION_N as f32),
            north: f(north, 0.0),
            pad: [0.0; 4],
        };
        queue.write_buffer(&self.region, 0, bytemuck::bytes_of(&region));
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("erosion") });
        {
            let mut pass =
                enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("erosion"), timestamp_writes: None });
            let groups = REGION_N.div_ceil(8);
            let mut run = |p: usize, g: usize| {
                pass.set_pipeline(&self.pipelines[p]);
                pass.set_bind_group(0, &self.groups[g], &[]);
                pass.dispatch_workgroups(groups, groups, 1);
            };
            run(0, 0);
            for k in 0..STEPS as usize {
                let g = k % 2;
                run(1, g);
                run(2, g);
                run(3, g);
            }
            run(4, STEPS as usize % 2);
        }
        queue.submit([enc.finish()]);
        self.baked = Some((key, c));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The square's axes are orthonormal and along the ground, and the
    /// parameters mirror the shader's struct.
    #[test]
    fn axes_and_layout() {
        for c in [[0.3, -0.5, 0.81], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]] {
            let c = vec3::normalize(c);
            let (e, n) = tangent_axes(c);
            assert!(vec3::dot(e, c).abs() < 1e-12 && vec3::dot(n, c).abs() < 1e-12 && vec3::dot(e, n).abs() < 1e-12);
            assert!((vec3::norm(e) - 1.0).abs() < 1e-12 && (vec3::norm(n) - 1.0).abs() < 1e-12);
        }
        assert_eq!(std::mem::size_of::<ErosionParams>(), 96);
        assert_eq!(std::mem::size_of::<RegionGpu>(), 64);
    }
}
