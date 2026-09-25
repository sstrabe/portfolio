//! Terrain heights read back from the GPU (`wgsl/probe.wgsl`).

use crate::shaders;
use bytemuck::{Pod, Zeroable};
use kerr::planets::Planet;
use kerr::vec3::V3;
use wgpu::util::DeviceExt;

/// Mirrors `struct ProbeParams` in `probe.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct ProbeParams {
    kind: u32,
    seed: u32,
    count: u32,
    air: u32,
    radius: f32,
    relief: f32,
    sea: f32,
    t_eq: f32,
    lod: f32,
    pad: [f32; 3],
}

/// The terrain at one body-fixed direction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sample {
    /// Height of what you would stand on or float on, km above the datum
    /// (sea level on worlds with seas).
    pub surface_km: f64,
    /// Height of the solid ground (below the sea where it's flooded).
    pub solid_km: f64,
    /// What fills basins: 0 nothing, 1 water, 2 magma, 3 solid basalt.
    pub fill: u32,
    /// The macro channels (see `terrain_macro` in `terrain.wgsl`).
    pub channels: [f32; 4],
}

pub struct Probe {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

impl Probe {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("terrain probe"),
            source: wgpu::ShaderSource::Wgsl(shaders::probe().into()),
        });
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        use wgpu::BufferBindingType as B;
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrain probe"),
            entries: &[
                entry(1, B::Uniform),
                entry(2, B::Storage { read_only: true }),
                entry(3, B::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("terrain probe"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("terrain probe"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("cs_probe"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self { pipeline, layout }
    }

    /// Terrain at body-fixed unit directions `dirs` (see
    /// [`super::to_body`]) of `planet`, resolved down to `lod_km`. Blocks
    /// until the GPU has answered.
    pub fn sample(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        planet: &Planet,
        dirs: &[V3],
        lod_km: f64,
    ) -> Vec<Sample> {
        if dirs.is_empty() {
            return Vec::new();
        }
        let params = ProbeParams {
            kind: planet.kind.index(),
            seed: planet.seed,
            count: dirs.len() as u32,
            air: planet.atmosphere.is_some() as u32,
            radius: planet.radius_km as f32,
            relief: planet.relief_km as f32,
            sea: planet.sea_level as f32,
            t_eq: planet.equilibrium_temperature as f32,
            lod: lod_km as f32,
            pad: [0.0; 3],
        };
        let init = |label, contents: &[u8], usage| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents, usage })
        };
        use wgpu::BufferUsages as U;
        let uniform = init("probe params", bytemuck::bytes_of(&params), U::UNIFORM);
        let packed: Vec<[f32; 4]> = dirs.iter().map(|d| [d[0] as f32, d[1] as f32, d[2] as f32, 0.0]).collect();
        let input = init("probe directions", bytemuck::cast_slice(&packed), U::STORAGE);
        let size = (dirs.len() * 2 * 16) as u64;
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("probe output"),
            size,
            usage: U::STORAGE | U::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("probe readback"),
            size,
            usage: U::MAP_READ | U::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain probe"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 1, resource: uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: input.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: output.as_entire_binding() },
            ],
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("terrain probe") });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("terrain probe"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((dirs.len() as u32).div_ceil(64), 1, 1);
        }
        enc.copy_buffer_to_buffer(&output, 0, &readback, 0, size);
        queue.submit([enc.finish()]);
        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let out = match slice.get_mapped_range() {
            Ok(data) => {
                let values: &[[f32; 4]] = bytemuck::cast_slice(&data);
                (0..dirs.len())
                    .map(|i| {
                        let (a, m) = (values[2 * i], values[2 * i + 1]);
                        Sample { surface_km: a[0] as f64, solid_km: a[1] as f64, fill: a[2] as u32, channels: m }
                    })
                    .collect()
            }
            Err(_) => Vec::new(),
        };
        readback.unmap();
        out
    }
}
