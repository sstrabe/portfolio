//! WebGPU renderer (through wgpu's browser backend).
//!
//! Passes per frame:
//! 1. `images` compute: every body's direct and around-the-hole image on the
//!    pilot's past light cone (reads the mirrored worldline history).
//! 2. `sky` render: per-pixel backward null geodesics into an HDR target at
//!    a reduced resolution; horizon, lensed galaxy, station panels.
//! 3. `composite` render: upscale + tone map into the swap chain, then the
//!    body-image point sprites additively at full resolution.

use kerr::history::Sample;
use wgpu::util::DeviceExt;

pub use crate::frame::{
    ATLAS_CELL_H, ATLAS_CELL_W, ATLAS_COLS, ATLAS_ROWS, BodyMeta, FrameUniforms, PANEL_SLOTS, PanelUniform,
};

const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

pub struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    srgb_encode: bool,
    render_scale: f32,
    bodies: u32,
    history_cap: u32,

    frame_buf: wgpu::Buffer,
    history_buf: wgpu::Buffer,
    meta_buf: wgpu::Buffer,
    panel_buf: wgpu::Buffer,
    atlas: wgpu::Texture,
    linear: wgpu::Sampler,

    images_pipeline: wgpu::ComputePipeline,
    sky_pipeline: wgpu::RenderPipeline,
    composite_pipeline: wgpu::RenderPipeline,
    sprite_pipeline: wgpu::RenderPipeline,

    images_bg: wgpu::BindGroup,
    sprite_bg: wgpu::BindGroup,
    sky_bg: wgpu::BindGroup,
    hdr: Option<(wgpu::TextureView, wgpu::BindGroup, u32, u32)>,
}

impl Gpu {
    pub async fn new(canvas: web_sys::HtmlCanvasElement, bodies: u32, history_cap: u32) -> Result<Self, String> {
        let (width, height) = (canvas.width().max(1), canvas.height().max(1));
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = wgpu::Backends::BROWSER_WEBGPU;
        let instance = wgpu::Instance::new(desc);
        let surface =
            instance.create_surface(wgpu::SurfaceTarget::Canvas(canvas)).map_err(|e| format!("surface: {e}"))?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            })
            .await
            .map_err(|e| format!("adapter: {e}"))?;
        let mut limits = wgpu::Limits::downlevel_defaults().using_resolution(adapter.limits());
        let history_bytes = (history_cap as u64 + 1) * bodies as u64 * 32;
        limits.max_storage_buffer_binding_size = limits
            .max_storage_buffer_binding_size
            .max(history_bytes.min(adapter.limits().max_storage_buffer_binding_size));
        limits.max_buffer_size = limits.max_buffer_size.max(history_bytes.min(adapter.limits().max_buffer_size));
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("kerr"),
                required_limits: limits,
                ..Default::default()
            })
            .await
            .map_err(|e| format!("device: {e}"))?;

        let mut config =
            surface.get_default_config(&adapter, width, height).ok_or("surface is not supported by the adapter")?;
        config.alpha_mode = wgpu::CompositeAlphaMode::Opaque;
        config.present_mode = wgpu::PresentMode::Fifo;
        surface.configure(&device, &config);
        let srgb_encode = !config.format.is_srgb();

        let frame_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame"),
            size: std::mem::size_of::<FrameUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let history_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("history"),
            size: history_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let meta_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bodies"),
            size: bodies as u64 * std::mem::size_of::<BodyMeta>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let image_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("images"),
            contents: &vec![0u8; bodies as usize * 2 * 48],
            usage: wgpu::BufferUsages::STORAGE,
        });
        let panel_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("panels"),
            size: (PANEL_SLOTS * std::mem::size_of::<PanelUniform>()) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let atlas = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("panel atlas"),
            size: wgpu::Extent3d {
                width: ATLAS_COLS * ATLAS_CELL_W,
                height: ATLAS_ROWS * ATLAS_CELL_H,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let linear = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("linear"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let module = |label: &str, src: &str| {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            })
        };
        let sky_mod = module("sky", shaders::SKY);
        let images_mod = module("images", shaders::IMAGES);
        let sprites_mod = module("sprites", shaders::SPRITES);
        let composite_mod = module("composite", shaders::COMPOSITE);

        let images_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("images"),
            layout: None,
            module: &images_mod,
            entry_point: Some("cs_images"),
            compilation_options: Default::default(),
            cache: None,
        });
        let render = |label: &str, m: &wgpu::ShaderModule, vs: &str, fs: &str, format, blend| {
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
        let additive = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent::OVER,
        };
        let sky_pipeline = render("sky", &sky_mod, "vs_fullscreen", "fs_sky", HDR_FORMAT, None);
        let composite_pipeline =
            render("composite", &composite_mod, "vs_fullscreen", "fs_composite", config.format, None);
        let sprite_pipeline = render("sprites", &sprites_mod, "vs_sprite", "fs_sprite", config.format, Some(additive));

        fn entry(binding: u32, buf: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry { binding, resource: buf.as_entire_binding() }
        }
        let images_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("images"),
            layout: &images_pipeline.get_bind_group_layout(0),
            entries: &[entry(0, &frame_buf), entry(1, &history_buf), entry(2, &meta_buf), entry(3, &image_buf)],
        });
        let sprite_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sprites"),
            layout: &sprite_pipeline.get_bind_group_layout(0),
            entries: &[entry(0, &frame_buf), entry(1, &image_buf), entry(2, &meta_buf)],
        });
        let atlas_view = atlas.create_view(&Default::default());
        let sky_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sky"),
            layout: &sky_pipeline.get_bind_group_layout(0),
            entries: &[
                entry(0, &frame_buf),
                entry(1, &panel_buf),
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&atlas_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&linear) },
            ],
        });

        Ok(Self {
            device,
            queue,
            surface,
            config,
            srgb_encode,
            render_scale: 0.5,
            bodies,
            history_cap,
            frame_buf,
            history_buf,
            meta_buf,
            panel_buf,
            atlas,
            linear,
            images_pipeline,
            sky_pipeline,
            composite_pipeline,
            sprite_pipeline,
            images_bg,
            sprite_bg,
            sky_bg,
            hdr: None,
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        let (width, height) = (width.max(1), height.max(1));
        if width == self.config.width && height == self.config.height {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
        self.hdr = None;
    }

    pub fn set_render_scale(&mut self, scale: f32) {
        let scale = scale.clamp(0.2, 1.0);
        if (scale - self.render_scale).abs() > 1e-3 {
            self.render_scale = scale;
            self.hdr = None;
        }
    }

    pub fn output_size(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }

    /// Size of the ray-traced HDR target.
    pub fn hdr_size(&self) -> (u32, u32) {
        let w = ((self.config.width as f32 * self.render_scale).round() as u32).max(1);
        let h = ((self.config.height as f32 * self.render_scale).round() as u32).max(1);
        (w, h)
    }

    pub fn srgb_encode(&self) -> bool {
        self.srgb_encode
    }

    fn ensure_hdr(&mut self) {
        let (w, h) = self.hdr_size();
        if matches!(self.hdr, Some((_, _, hw, hh)) if hw == w && hh == h) {
            return;
        }
        let tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("hdr"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: HDR_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = tex.create_view(&Default::default());
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("composite"),
            layout: &self.composite_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.frame_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.linear) },
            ],
        });
        self.hdr = Some((view, bg, w, h));
    }

    pub fn write_frame(&self, f: &FrameUniforms) {
        self.queue.write_buffer(&self.frame_buf, 0, bytemuck::bytes_of(f));
    }

    pub fn write_meta(&self, meta: &[BodyMeta]) {
        self.queue.write_buffer(&self.meta_buf, 0, bytemuck::cast_slice(meta));
    }

    pub fn write_panels(&self, panels: &[PanelUniform; PANEL_SLOTS]) {
        self.queue.write_buffer(&self.panel_buf, 0, bytemuck::cast_slice(panels));
    }

    /// Upload one history column (ring slot, or `history_cap` for "now").
    pub fn write_history_column(&self, slot: u32, samples: &[Sample]) {
        let data: Vec<[f32; 4]> = samples
            .iter()
            .flat_map(|s| {
                [
                    [s.pos[0] as f32, s.pos[1] as f32, s.pos[2] as f32, 0.0],
                    [s.vel[0] as f32, s.vel[1] as f32, s.vel[2] as f32, 0.0],
                ]
            })
            .collect();
        let offset = slot as u64 * self.bodies as u64 * 32;
        self.queue.write_buffer(&self.history_buf, offset, bytemuck::cast_slice(&data));
    }

    pub fn history_cap(&self) -> u32 {
        self.history_cap
    }

    /// Copy a 2-D canvas (the panel atlas drawn by the web layer, possibly
    /// through HTML-in-Canvas) into the atlas texture.
    pub fn upload_atlas(&self, canvas: web_sys::HtmlCanvasElement) {
        let w = canvas.width().min(ATLAS_COLS * ATLAS_CELL_W);
        let h = canvas.height().min(ATLAS_ROWS * ATLAS_CELL_H);
        if w == 0 || h == 0 {
            return;
        }
        self.queue.copy_external_image_to_texture(
            &wgpu::CopyExternalImageSourceInfo {
                source: wgpu::ExternalImageSource::HTMLCanvasElement(canvas),
                origin: wgpu::Origin2d::ZERO,
                flip_y: false,
            },
            wgpu::CopyExternalImageDestInfo {
                texture: &self.atlas,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
                color_space: wgpu::PredefinedColorSpace::Srgb,
                premultiplied_alpha: false,
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
    }

    pub fn render(&mut self) {
        self.ensure_hdr();
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.device, &self.config);
                return;
            }
            _ => return,
        };
        let out_view = frame.texture.create_view(&Default::default());
        let Some((hdr_view, composite_bg, _, _)) = &self.hdr else { return };
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });
        {
            let mut pass =
                enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("images"), timestamp_writes: None });
            pass.set_pipeline(&self.images_pipeline);
            pass.set_bind_group(0, &self.images_bg, &[]);
            pass.dispatch_workgroups((self.bodies * 2).div_ceil(64), 1, 1);
        }
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sky"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: hdr_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.sky_pipeline);
            pass.set_bind_group(0, &self.sky_bg, &[]);
            pass.draw(0..3, 0..1);
        }
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("composite"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &out_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.composite_pipeline);
            pass.set_bind_group(0, composite_bg, &[]);
            pass.draw(0..3, 0..1);
            pass.set_pipeline(&self.sprite_pipeline);
            pass.set_bind_group(0, &self.sprite_bg, &[]);
            pass.draw(0..6, 0..self.bodies * 2);
        }
        self.queue.submit([enc.finish()]);
        self.queue.present(frame);
    }
}
