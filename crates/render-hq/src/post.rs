//! Optics and post-processing after the trace: point sources splatted into
//! the HDR radiance, then exposure, tone mapping and upscaling to the
//! screen (see `wgsl/splat.wgsl`, `wgsl/post.wgsl`).

use crate::FrameContext;
use crate::shaders;

/// Buffers the post passes read, owned by [`crate::Gpu`].
pub struct Shared<'a> {
    pub frame: &'a wgpu::Buffer,
    pub hq: &'a wgpu::Buffer,
    pub images: &'a wgpu::Buffer,
    pub meta: &'a wgpu::Buffer,
}

pub struct Post {
    bodies: u32,
    splat_pipeline: wgpu::RenderPipeline,
    splat_bg: wgpu::BindGroup,
    post_pipeline: wgpu::RenderPipeline,
    post_bg: Option<wgpu::BindGroup>,
    frame: wgpu::Buffer,
    hq: wgpu::Buffer,
    sampler: wgpu::Sampler,
}

impl Post {
    pub fn new(device: &wgpu::Device, out_format: wgpu::TextureFormat, shared: &Shared, bodies: u32) -> Self {
        let module = |label: &str, src: &str| {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            })
        };
        let splat_mod = module("splat", &shaders::splat());
        let post_mod = module("post", &shaders::post());
        let pipeline = |label: &str, m: &wgpu::ShaderModule, vs: &str, fs: &str, format, blend| {
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
        // dst += src · dst.alpha (the near-field transmittance); alpha kept.
        let behind_near_field = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::DstAlpha,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        };
        let splat_pipeline =
            pipeline("splat", &splat_mod, "vs_splat", "fs_splat", crate::gpu::HDR_FORMAT, Some(behind_near_field));
        let post_pipeline = pipeline("post", &post_mod, "vs_fullscreen", "fs_post", out_format, None);
        fn entry(binding: u32, buf: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry { binding, resource: buf.as_entire_binding() }
        }
        let splat_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("splat"),
            layout: &splat_pipeline.get_bind_group_layout(0),
            entries: &[entry(0, shared.frame), entry(1, shared.hq), entry(2, shared.images), entry(3, shared.meta)],
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("hdr"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        Self {
            bodies,
            splat_pipeline,
            splat_bg,
            post_pipeline,
            post_bg: None,
            frame: shared.frame.clone(),
            hq: shared.hq.clone(),
            sampler,
        }
    }

    /// A new HDR target (after a resize or a resolution change).
    pub fn attach(&mut self, device: &wgpu::Device, hdr: &wgpu::TextureView) {
        self.post_bg = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("post"),
            layout: &self.post_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.frame.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.hq.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(hdr) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&self.sampler) },
            ],
        }));
    }

    pub fn update(&mut self, _ctx: &FrameContext) {}

    /// Splat point sources into `hdr`, then tone map it into `out`.
    pub fn encode(&self, enc: &mut wgpu::CommandEncoder, hdr: &wgpu::TextureView, out: &wgpu::TextureView) {
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("splat"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: hdr,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.splat_pipeline);
            pass.set_bind_group(0, &self.splat_bg, &[]);
            pass.draw(0..6, 0..self.bodies * 2);
        }
        let Some(post_bg) = &self.post_bg else { return };
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("post"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: out,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.post_pipeline);
        pass.set_bind_group(0, post_bg, &[]);
        pass.draw(0..3, 0..1);
    }
}
