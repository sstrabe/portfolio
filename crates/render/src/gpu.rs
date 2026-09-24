//! The renderer, shared by the browser engine (wgpu's WebGPU backend) and the
//! native desktop app (Vulkan / Metal / DX12).
//!
//! Passes per frame:
//! 1. `images` compute: every body's direct and around-the-hole image on the
//!    pilot's past light cone (reads the mirrored worldline history).
//! 2. `sky` render: per-pixel backward null geodesics into an HDR target at
//!    a reduced resolution; horizon, lensed galaxy, station panels.
//! 3. `composite` render: upscale + tone map into the swap chain, then the
//!    body-image point sprites additively at full resolution.

use kerr::history::Sample;
use std::sync::{Arc, Mutex, MutexGuard};
use wgpu::util::DeviceExt;

pub use crate::frame::{
    ATLAS_CELL_H, ATLAS_CELL_W, ATLAS_COLS, ATLAS_ROWS, BodyMeta, FrameUniforms, PANEL_SLOTS, PanelUniform,
    SPHERE_SLOTS, SphereUniform,
};

const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

pub struct Gpu {
    /// Held for the renderer's lifetime: if the browser's `GPU` object is
    /// garbage collected, Chrome stops delivering buffer-mapping and other
    /// asynchronous events for the device.
    _instance: wgpu::Instance,
    device: wgpu::Device,
    queue: wgpu::Queue,
    output: Output,
    srgb_encode: bool,
    render_scale: f32,
    bodies: u32,
    history_cap: u32,

    frame_buf: wgpu::Buffer,
    history_buf: wgpu::Buffer,
    meta_buf: wgpu::Buffer,
    panel_buf: wgpu::Buffer,
    sphere_buf: wgpu::Buffer,
    /// Station-card atlas; only the web engine uploads into it.
    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
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
    /// Headless mode only: a requested / finished readback of the output.
    capture: Arc<Mutex<Capture>>,
}

/// Where frames go: the page's canvas, or (headless) an offscreen texture.
enum Output {
    Surface { surface: wgpu::Surface<'static>, config: wgpu::SurfaceConfiguration },
    Offscreen { texture: wgpu::Texture, width: u32, height: u32 },
}

#[derive(Default)]
enum Capture {
    #[default]
    Idle,
    Requested,
    Pending,
    Ready(Vec<u8>),
}

const OFFSCREEN_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

fn offscreen_texture(device: &wgpu::Device, width: u32, height: u32) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("offscreen output"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: OFFSCREEN_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

impl Gpu {
    /// Render to `surface`, or with `None` into an offscreen texture of
    /// `size` whose frames can be read back (headless checks and tools).
    pub async fn new(
        instance: wgpu::Instance,
        surface: Option<wgpu::Surface<'static>>,
        size: (u32, u32),
        bodies: u32,
        history_cap: u32,
    ) -> Result<Self, String> {
        let (width, height) = (size.0.max(1), size.1.max(1));
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: surface.as_ref(),
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            })
            .await
            .map_err(|e| format!("adapter: {e}"))?;
        #[cfg(not(target_arch = "wasm32"))]
        {
            let info = adapter.get_info();
            eprintln!("GPU: {} ({:?}, {:?})", info.name, info.backend, info.device_type);
        }
        let supported = adapter.limits();
        let mut limits = wgpu::Limits::downlevel_defaults().using_resolution(supported.clone());
        let history_bytes = (history_cap as u64 + 1) * bodies as u64 * 32;
        limits.max_storage_buffer_binding_size =
            limits.max_storage_buffer_binding_size.max(history_bytes.min(supported.max_storage_buffer_binding_size));
        limits.max_buffer_size = limits.max_buffer_size.max(history_bytes.min(supported.max_buffer_size));

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("kerr"),
                required_limits: limits,
                ..Default::default()
            })
            .await
            .map_err(|e| format!("device: {e}"))?;

        // Report validation errors instead of losing them.
        device.on_uncaptured_error(std::sync::Arc::new(|e: wgpu::Error| log_error(&format!("GPU: {e}"))));

        let output = match surface {
            Some(surface) => {
                let mut config = surface
                    .get_default_config(&adapter, width, height)
                    .ok_or("surface is not supported by the adapter")?;
                config.alpha_mode = wgpu::CompositeAlphaMode::Opaque;
                // Readable (where supported) so the page can save the image.
                if surface.get_capabilities(&adapter).usages.contains(wgpu::TextureUsages::COPY_SRC) {
                    config.usage |= wgpu::TextureUsages::COPY_SRC;
                }
                config.present_mode = wgpu::PresentMode::Fifo;
                surface.configure(&device, &config);
                Output::Surface { surface, config }
            }
            None => Output::Offscreen { texture: offscreen_texture(&device, width, height), width, height },
        };
        let out_format = match &output {
            Output::Surface { config, .. } => config.format,
            Output::Offscreen { .. } => OFFSCREEN_FORMAT,
        };
        let srgb_encode = !out_format.is_srgb();

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
        let sphere_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("spheres"),
            size: (SPHERE_SLOTS * std::mem::size_of::<SphereUniform>()) as u64,
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
        let composite_pipeline = render("composite", &composite_mod, "vs_fullscreen", "fs_composite", out_format, None);
        let sprite_pipeline = render("sprites", &sprites_mod, "vs_sprite", "fs_sprite", out_format, Some(additive));

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
                entry(4, &sphere_buf),
            ],
        });

        Ok(Self {
            _instance: instance,
            device,
            queue,
            output,
            srgb_encode,
            render_scale: 0.5,
            bodies,
            history_cap,
            frame_buf,
            history_buf,
            meta_buf,
            panel_buf,
            sphere_buf,
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
            capture: Arc::default(),
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        let (width, height) = (width.max(1), height.max(1));
        if (width, height) == self.output_size() {
            return;
        }
        match &mut self.output {
            Output::Surface { surface, config } => {
                config.width = width;
                config.height = height;
                surface.configure(&self.device, config);
            }
            Output::Offscreen { texture, width: w, height: h } => {
                *texture = offscreen_texture(&self.device, width, height);
                (*w, *h) = (width, height);
            }
        }
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
        match &self.output {
            Output::Surface { config, .. } => (config.width, config.height),
            Output::Offscreen { width, height, .. } => (*width, *height),
        }
    }

    /// Size of the ray-traced HDR target.
    pub fn hdr_size(&self) -> (u32, u32) {
        let (ow, oh) = self.output_size();
        let w = ((ow as f32 * self.render_scale).round() as u32).max(1);
        let h = ((oh as f32 * self.render_scale).round() as u32).max(1);
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

    pub fn write_spheres(&self, spheres: &[SphereUniform; SPHERE_SLOTS]) {
        self.queue.write_buffer(&self.sphere_buf, 0, bytemuck::cast_slice(spheres));
    }

    pub fn write_panels(&self, panels: &[PanelUniform; PANEL_SLOTS]) {
        self.queue.write_buffer(&self.panel_buf, 0, bytemuck::cast_slice(panels));
    }

    /// Native: block until submitted work (and pending buffer maps) finish.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn wait(&self) {
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
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
    #[cfg(target_arch = "wasm32")]
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
        let (frame, out_view) = match &self.output {
            Output::Surface { surface, config } => match surface.get_current_texture() {
                wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => {
                    let view = t.texture.create_view(&Default::default());
                    (Some(t), view)
                }
                wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                    surface.configure(&self.device, config);
                    return;
                }
                _ => return,
            },
            Output::Offscreen { texture, .. } => (None, texture.create_view(&Default::default())),
        };
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });
        let Some((hdr_view, composite_bg, _, _)) = &self.hdr else { return };
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
        self.encode_capture(&mut enc);
        self.queue.submit([enc.finish()]);
        if let Some(frame) = frame {
            self.queue.present(frame);
        }
    }

    /// Headless mode: ask for the next frame's pixels (RGBA8, sRGB encoded).
    pub fn request_capture(&self) {
        let mut c = self.capture();
        if matches!(*c, Capture::Idle | Capture::Ready(_)) {
            *c = Capture::Requested;
        }
    }

    fn capture(&self) -> MutexGuard<'_, Capture> {
        self.capture.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn take_capture(&self) -> Option<Vec<u8>> {
        let mut c = self.capture();
        match std::mem::take(&mut *c) {
            Capture::Ready(rgba) => Some(rgba),
            other => {
                *c = other;
                None
            }
        }
    }

    fn encode_capture(&self, enc: &mut wgpu::CommandEncoder) {
        let Output::Offscreen { texture, width, height } = &self.output else { return };
        if !matches!(*self.capture(), Capture::Requested) {
            return;
        }
        let (width, height) = (*width, *height);
        let padded = (width as usize * 4).div_ceil(256) * 256;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("capture"),
            size: (padded * height as usize) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        enc.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded as u32),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );
        *self.capture() = Capture::Pending;
        let cell = self.capture.clone();
        let mapped = buffer.clone();
        enc.map_buffer_on_submit(&buffer, wgpu::MapMode::Read, .., move |result| {
            let rgba = result.ok().and_then(|_| {
                let data = mapped.slice(..).get_mapped_range().ok()?;
                let row = width as usize * 4;
                let mut rgba = Vec::with_capacity(8 + row * height as usize);
                rgba.extend_from_slice(&width.to_le_bytes());
                rgba.extend_from_slice(&height.to_le_bytes());
                for y in 0..height as usize {
                    rgba.extend_from_slice(&data[y * padded..y * padded + row]);
                }
                Some(rgba)
            });
            mapped.unmap();
            *cell.lock().unwrap_or_else(|e| e.into_inner()) = rgba.map_or(Capture::Idle, Capture::Ready);
        });
    }
}

fn log_error(msg: &str) {
    #[cfg(target_arch = "wasm32")]
    web_sys::console::error_1(&msg.into());
    #[cfg(not(target_arch = "wasm32"))]
    eprintln!("{msg}");
}
