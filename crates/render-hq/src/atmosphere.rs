//! Atmospheres and clouds: bind group 2 of the trace pass (see
//! `wgsl/atmosphere.wgsl`). Slot `i` describes near-field planet slot `i`.

use crate::FrameContext;
use crate::near::MAX_PLANETS;
use bytemuck::{Pod, Zeroable};

/// Mirrors `struct AtmosphereParams` in `atmosphere.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct AtmosphereGpu {
    pub shape: [f32; 4],
    pub density: [f32; 4],
    pub aerosol: [f32; 4],
    pub clouds: [f32; 4],
}

impl AtmosphereGpu {
    pub fn new(radius_km: f64, a: &kerr::planets::Atmosphere) -> Self {
        let (cov, base, thick, tau) =
            a.clouds.map_or((0.0, 0.0, 0.0, 0.0), |c| (c.coverage, c.base_km, c.thickness_km, c.optical_depth));
        Self {
            shape: [radius_km as f32, a.top_km as f32, a.rayleigh_scale_height_km as f32, a.mie_scale_height_km as f32],
            density: [a.rayleigh_density as f32, a.mie_density as f32, a.ozone as f32, a.methane as f32],
            aerosol: [a.mie_g as f32, a.mie_absorption as f32, a.dust as f32, 1.0],
            clouds: [cov as f32, base as f32, thick as f32, tau as f32],
        }
    }
}

pub struct Atmospheres {
    layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    params: wgpu::Buffer,
}

impl Atmospheres {
    pub fn new(device: &wgpu::Device, _queue: &wgpu::Queue) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("atmospheres"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("atmospheres"),
            size: (MAX_PLANETS * std::mem::size_of::<AtmosphereGpu>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("atmospheres"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() }],
        });
        Self { layout, bind_group, params }
    }

    pub fn layout(&self) -> &wgpu::BindGroupLayout {
        &self.layout
    }

    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    pub fn update(&mut self, ctx: &FrameContext) {
        let sel = ctx.near;
        let data: Vec<AtmosphereGpu> = sel
            .planets
            .iter()
            .map(|p| {
                let planet = p.planet(sel);
                planet
                    .atmosphere
                    .as_ref()
                    .map_or_else(AtmosphereGpu::default, |a| AtmosphereGpu::new(planet.radius_km, a))
            })
            .collect();
        if !data.is_empty() {
            ctx.queue.write_buffer(&self.params, 0, bytemuck::cast_slice(&data));
        }
    }

    /// Work before the trace pass (e.g. lookup tables).
    pub fn encode(&mut self, _enc: &mut wgpu::CommandEncoder, _ctx: &FrameContext) {}
}

#[cfg(test)]
mod tests {
    #[test]
    fn layout_matches_wgsl() {
        assert_eq!(std::mem::size_of::<super::AtmosphereGpu>(), 4 * 16);
    }
}
