//! Nebulae and dust: bind group 3 of the trace pass (see
//! `wgsl/nebula.wgsl`). Placeholder: no resources.

use crate::FrameContext;

pub struct Nebulae {
    layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
}

impl Nebulae {
    pub fn new(device: &wgpu::Device, _queue: &wgpu::Queue) -> Self {
        let layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("nebulae"), entries: &[] });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("nebulae"),
            layout: &layout,
            entries: &[],
        });
        Self { layout, bind_group }
    }

    pub fn layout(&self) -> &wgpu::BindGroupLayout {
        &self.layout
    }

    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    pub fn update(&mut self, _ctx: &FrameContext) {}

    /// Work before the trace pass (e.g. generating volumes).
    pub fn encode(&mut self, _enc: &mut wgpu::CommandEncoder, _ctx: &FrameContext) {}
}
