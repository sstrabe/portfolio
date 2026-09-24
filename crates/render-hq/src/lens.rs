//! Lensing by stellar-mass black holes: bind group 5 of the trace pass (see
//! `wgsl/lens.wgsl`). Placeholder: no resources.

use crate::FrameContext;

pub struct Lensing {
    layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
}

impl Lensing {
    pub fn new(device: &wgpu::Device) -> Self {
        let layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("lensing"), entries: &[] });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lensing"),
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
}
