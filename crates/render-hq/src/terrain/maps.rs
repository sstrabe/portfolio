//! Baked planet maps: cube maps over body-fixed directions for what needs
//! more than a point's own terrain to work out. For now the climate
//! (`wgsl/climate.wgsl`): temperature, precipitation with the prevailing
//! winds and rain shadows, and vegetation. One planet at a time has them
//! (the nearest solid world with air); the near field binds them.

use crate::shaders;
use bytemuck::{Pod, Zeroable};
use kerr::planets::Planet;
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
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
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
}
