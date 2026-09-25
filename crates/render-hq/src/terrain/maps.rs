//! Baked planet maps: cube maps over body-fixed directions for what needs
//! more than a point's own terrain to work out. For now the climate
//! (`wgsl/climate.wgsl`): temperature, precipitation with the prevailing
//! winds and rain shadows, and vegetation. One planet at a time has them
//! (the nearest solid world with air); the near field binds them.

use super::cube;
use crate::shaders;
use bytemuck::{Pod, Zeroable};
use kerr::planets::Planet;
use kerr::vec3::V3;
use wgpu::util::DeviceExt;

/// Interior texels per face edge of the climate map (~22 km on an Earth);
/// must match `CLIMATE_N` in `maps.wgsl`.
pub const CLIMATE_N: u32 = 512;

/// Mirrors `struct ClimateParams` in `climate.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct ClimateParams {
    kind: u32,
    seed: u32,
    n: u32,
    air: u32,
    radius: f32,
    relief: f32,
    sea: f32,
    t_eq: f32,
}

/// Which planet the maps hold: (star slot, star generation, planet index).
pub type MapKey = (usize, u32, usize);

pub struct SurfaceMaps {
    climate: wgpu::Texture,
    pub climate_view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    /// The planet baked into the maps, if any.
    pub baked: Option<MapKey>,
}

impl SurfaceMaps {
    pub fn new(device: &wgpu::Device) -> Self {
        let size = CLIMATE_N + 2;
        let climate = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("climate map"),
            size: wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 6 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let climate_view = climate.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("planet maps"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("climate"),
            source: wgpu::ShaderSource::Wgsl(shaders::climate().into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("climate bake"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("climate bake"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("climate bake"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("cs_climate"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self { climate, climate_view, sampler, pipeline, layout, baked: None }
    }

    /// Bake `planet`'s maps (submitted now; the GPU finishes them before
    /// the next frame that reads them).
    pub fn bake(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, key: MapKey, planet: &Planet) {
        let params = ClimateParams {
            kind: planet.kind.index(),
            seed: planet.seed,
            n: CLIMATE_N,
            air: planet.atmosphere.is_some() as u32,
            radius: planet.radius_km as f32,
            relief: planet.relief_km as f32,
            sea: planet.sea_level as f32,
            t_eq: planet.equilibrium_temperature as f32,
        };
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("climate params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let storage_view = self.climate.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("climate bake"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 1, resource: uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&storage_view) },
            ],
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("climate bake") });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("climate bake"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (CLIMATE_N + 2).div_ceil(8);
            pass.dispatch_workgroups(groups, groups, 6);
        }
        queue.submit([enc.finish()]);
        self.baked = Some(key);
    }

    /// The baked climate, read back to the CPU (blocking: for tools such as
    /// the site finder, and tests).
    pub fn read_climate(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> ClimateMap {
        let size = CLIMATE_N + 2;
        let row = (size * 8).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("climate readback"),
            size: (row * size * 6) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("climate readback") });
        enc.copy_texture_to_buffer(
            self.climate.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(size) },
            },
            wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 6 },
        );
        queue.submit([enc.finish()]);
        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let mut texels = Vec::with_capacity((size * size * 6) as usize);
        if let Ok(data) = slice.get_mapped_range() {
            let halves: &[u16] = bytemuck::cast_slice(&data);
            for layer in 0..size * 6 {
                let start = (layer * row / 2) as usize;
                for x in 0..size as usize {
                    let t = &halves[start + 4 * x..start + 4 * x + 4];
                    texels.push(std::array::from_fn(|c| f16_to_f32(t[c])));
                }
            }
        }
        ClimateMap { n: CLIMATE_N as usize, texels }
    }
}

/// The climate cube map on the CPU (see [`SurfaceMaps::read_climate`]).
pub struct ClimateMap {
    n: usize,
    texels: Vec<[f32; 4]>,
}

impl ClimateMap {
    /// (temperature K, precipitation mm/yr, vegetation 0–1, dryness 0–1)
    /// at body-fixed direction `q`: the nearest texel.
    pub fn at(&self, q: V3) -> [f32; 4] {
        let (face, u, v) = cube::face_uv(q);
        let size = self.n + 2;
        let texel = |c: f64| (((c * 0.5 + 0.5) * self.n as f64) as usize + 1).min(self.n);
        self.texels.get((face * size + texel(v)) * size + texel(u)).copied().unwrap_or([0.0; 4])
    }
}

/// IEEE half to single precision.
fn f16_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = ((h >> 10) & 0x1f) as i32;
    let man = (h & 0x3ff) as f32;
    sign * match exp {
        0 => man * 2f32.powi(-24),
        31 => f32::INFINITY,
        _ => (1.0 + man / 1024.0) * 2f32.powi(exp - 15),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halves_decode() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
        // 288.0 (a temperature) and 1500.0 (rain).
        assert_eq!(f16_to_f32(0x5c80), 288.0);
        assert_eq!(f16_to_f32(0x65dc), 1500.0);
    }
}
