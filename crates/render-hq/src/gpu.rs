//! Device, targets and the per-frame pass order:
//!
//! 1. feature pre-passes (lookup tables, volumes),
//! 2. `images` compute: every body's lensed images on the past light cone
//!    (shared with the web renderer),
//! 3. `trace` compute: per-pixel spectral ray tracing into the HDR target,
//! 4. post (`post.rs`): temporal accumulation and upscaling, point sources,
//!    lens ghosts, the FFT convolution with the point-spread function,
//!    metering, the eye model and tone mapping, at the output resolution.

use crate::atmosphere::Atmospheres;
use crate::lens::Lensing;
use crate::near::NearField;
use crate::nebula::Nebulae;
use crate::post::{self, Post};
use crate::ship::Ship;
use crate::{FrameContext, HqUniforms, shaders};
use kerr::history::Sample;
use kerr::world::World;
use render::frame::{BodyMeta, FrameUniforms, SPHERE_SLOTS, SphereUniform};
use std::sync::{Arc, Mutex, MutexGuard};
use wgpu::util::DeviceExt;

pub const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;
const OFFSCREEN_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const TRACE_TILE: u32 = 8;

/// Where frames go: a window's surface, or (headless) an offscreen texture.
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

struct Hdr {
    width: u32,
    height: u32,
    core_bg: wgpu::BindGroup,
}

pub struct Gpu {
    _instance: wgpu::Instance,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    output: Output,
    srgb_encode: bool,
    render_scale: f32,
    bodies: u32,
    history_cap: u32,
    frame_index: u32,

    frame_buf: wgpu::Buffer,
    hq_buf: wgpu::Buffer,
    sphere_buf: wgpu::Buffer,
    history_buf: wgpu::Buffer,
    meta_buf: wgpu::Buffer,

    images_pipeline: wgpu::ComputePipeline,
    images_bg: wgpu::BindGroup,
    trace_pipeline: wgpu::ComputePipeline,
    core_layout: wgpu::BindGroupLayout,

    pub near: NearField,
    pub atmo: Atmospheres,
    pub nebula: Nebulae,
    pub ship: Ship,
    pub lens: Lensing,
    pub post: Post,

    hdr: Option<Hdr>,
    capture: Arc<Mutex<Capture>>,
}

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

/// Features the HQ renderer cannot run without.
const REQUIRED_FEATURES: wgpu::Features = wgpu::Features::FLOAT32_FILTERABLE.union(wgpu::Features::FLOAT32_BLENDABLE);

impl Gpu {
    /// Render to `surface`, or with `None` into an offscreen texture of
    /// `size` whose frames can be read back.
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
        let info = adapter.get_info();
        eprintln!("GPU: {} ({:?}, {:?})", info.name, info.backend, info.device_type);
        let missing = REQUIRED_FEATURES - adapter.features();
        if !missing.is_empty() {
            return Err(format!("the GPU lacks required features: {missing:?}"));
        }
        let supported = adapter.limits();
        let mut limits = wgpu::Limits::default().using_resolution(supported.clone());
        let history_bytes = (history_cap as u64 + 1) * bodies as u64 * 32;
        limits.max_storage_buffer_binding_size =
            limits.max_storage_buffer_binding_size.max(history_bytes.min(supported.max_storage_buffer_binding_size));
        limits.max_buffer_size = limits.max_buffer_size.max(history_bytes.min(supported.max_buffer_size));
        limits.max_bind_groups = supported.max_bind_groups.min(8);
        if limits.max_bind_groups < 6 {
            return Err(format!("the GPU supports only {} bind groups (6 needed)", limits.max_bind_groups));
        }
        limits.max_storage_buffers_per_shader_stage = supported.max_storage_buffers_per_shader_stage;
        limits.max_storage_textures_per_shader_stage = supported.max_storage_textures_per_shader_stage;
        limits.max_sampled_textures_per_shader_stage = supported.max_sampled_textures_per_shader_stage;
        limits.max_uniform_buffers_per_shader_stage = supported.max_uniform_buffers_per_shader_stage;
        // The FFT keeps a 2048-point row (or two 1024-point columns) of two
        // complex numbers per point in workgroup memory.
        const FFT_WORKGROUP_BYTES: u32 = 2048 * 16 + 1024;
        if supported.max_compute_workgroup_storage_size < FFT_WORKGROUP_BYTES {
            return Err(format!(
                "the GPU offers {} bytes of workgroup memory ({FFT_WORKGROUP_BYTES} needed)",
                supported.max_compute_workgroup_storage_size
            ));
        }
        limits.max_compute_workgroup_storage_size = supported.max_compute_workgroup_storage_size;

        let timing = if crate::profile::wanted() {
            adapter.features() & wgpu::Features::TIMESTAMP_QUERY
        } else {
            wgpu::Features::empty()
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("kerr hq"),
                required_features: REQUIRED_FEATURES | timing,
                required_limits: limits,
                ..Default::default()
            })
            .await
            .map_err(|e| format!("device: {e}"))?;
        device.on_uncaptured_error(Arc::new(|e: wgpu::Error| eprintln!("GPU: {e}")));

        let output = match surface {
            Some(surface) => {
                let mut config = surface
                    .get_default_config(&adapter, width, height)
                    .ok_or("surface is not supported by the adapter")?;
                config.alpha_mode = wgpu::CompositeAlphaMode::Opaque;
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

        let buffer = |label, size, usage| {
            device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage, mapped_at_creation: false })
        };
        let uniform = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;
        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let frame_buf = buffer("frame", std::mem::size_of::<FrameUniforms>() as u64, uniform);
        let hq_buf = buffer("hq frame", std::mem::size_of::<HqUniforms>() as u64, uniform);
        let sphere_buf = buffer("spheres", (SPHERE_SLOTS * std::mem::size_of::<SphereUniform>()) as u64, uniform);
        let history_buf = buffer("history", history_bytes, storage);
        let meta_buf = buffer("bodies", bodies as u64 * std::mem::size_of::<BodyMeta>() as u64, storage);
        let image_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("images"),
            contents: &vec![0u8; bodies as usize * 2 * 48],
            usage: wgpu::BufferUsages::STORAGE,
        });

        let module = |label: &str, src: &str| {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            })
        };
        let images_mod = module("images", ::shaders::IMAGES);
        let images_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("images"),
            layout: None,
            module: &images_mod,
            entry_point: Some("cs_images"),
            compilation_options: Default::default(),
            cache: None,
        });
        fn entry(binding: u32, buf: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry { binding, resource: buf.as_entire_binding() }
        }
        let images_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("images"),
            layout: &images_pipeline.get_bind_group_layout(0),
            entries: &[entry(0, &frame_buf), entry(1, &history_buf), entry(2, &meta_buf), entry(3, &image_buf)],
        });

        let uniform_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let storage_image = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: HDR_FORMAT,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        };
        let core_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("trace core"),
            entries: &[
                uniform_entry(0),
                uniform_entry(1),
                uniform_entry(2),
                storage_image(3),
                // Narrowband bins for the Hubble palette (`post_common.wgsl`).
                storage_image(4),
            ],
        });
        let near = NearField::new(&device);
        let atmo = Atmospheres::new(&device, &queue);
        let nebula = Nebulae::new(&device, &queue);
        let ship = Ship::new(&device, &queue);
        let lens = Lensing::new(&device);
        let trace_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("trace"),
            bind_group_layouts: &[
                Some(&core_layout),
                Some(near.layout()),
                Some(atmo.layout()),
                Some(nebula.layout()),
                Some(ship.layout()),
                Some(lens.layout()),
            ],
            immediate_size: 0,
        });
        let trace_mod = module("trace", &shaders::trace());
        let trace_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("trace"),
            layout: Some(&trace_layout),
            module: &trace_mod,
            entry_point: Some("cs_trace"),
            compilation_options: Default::default(),
            cache: None,
        });
        let post = Post::new(
            &device,
            &queue,
            out_format,
            &post::Shared { frame: &frame_buf, hq: &hq_buf, images: &image_buf, meta: &meta_buf },
            bodies,
        );

        let _ = 0;
        Ok(Self {
            _instance: instance,
            device,
            queue,
            output,
            srgb_encode: !out_format.is_srgb(),
            render_scale: 1.0,
            bodies,
            history_cap,
            frame_index: 0,
            frame_buf,
            hq_buf,
            sphere_buf,
            history_buf,
            meta_buf,
            images_pipeline,
            images_bg,
            trace_pipeline,
            core_layout,
            near,
            atmo,
            nebula,
            ship,
            lens,
            post,
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

    pub fn history_cap(&self) -> u32 {
        self.history_cap
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

    /// Block until submitted work (and pending buffer maps) finish.
    pub fn wait(&self) {
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
    }

    fn ensure_hdr(&mut self) {
        let (w, h) = self.hdr_size();
        if matches!(&self.hdr, Some(t) if t.width == w && t.height == h) {
            return;
        }
        let target = |label| {
            self.device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: HDR_FORMAT,
                    usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        };
        let view = target("hdr");
        let nb_view = target("narrowband");
        let core_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("trace core"),
            layout: &self.core_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.frame_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.hq_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.sphere_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&nb_view) },
            ],
        });
        let out = self.output_size();
        self.post.attach(&self.device, &self.queue, &view, &nb_view, (w, h), out);
        self.hdr = Some(Hdr { width: w, height: h, core_bg });
    }

    /// Pick the near field for this frame (before building `HqUniforms`).
    pub fn update_near(&mut self, world: &World, pixel_angle: f64) {
        self.near.update(&self.queue, world, pixel_angle);
    }

    /// Update every feature and render a frame. `hq` is completed here with
    /// the target size, frame index and near-field counts.
    pub fn render(&mut self, world: &World, mut hq: HqUniforms, wall_time: f64) {
        self.ensure_hdr();
        let (w, h) = self.hdr_size();
        let [systems, planets] = self.near.counts();
        hq.size = [w, h, self.frame_index, hq.size[3]];
        hq.near = [systems, planets, 0, 0];
        self.queue.write_buffer(&self.hq_buf, 0, bytemuck::bytes_of(&hq));

        let ctx = FrameContext {
            device: &self.device,
            queue: &self.queue,
            world,
            near: &self.near.selection,
            hq: &hq,
            frame_index: self.frame_index,
            wall_time,
        };
        self.atmo.update(&ctx);
        self.nebula.update(&ctx);
        self.ship.update(&ctx);
        self.lens.update(&ctx);
        self.post.update(&ctx);

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
        self.atmo.encode(&mut enc, &ctx);
        self.nebula.encode(&mut enc, &ctx);
        self.ship.encode(&mut enc, &ctx);
        let Some(hdr) = &self.hdr else { return };
        {
            let mut pass =
                enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("images"), timestamp_writes: None });
            pass.set_pipeline(&self.images_pipeline);
            pass.set_bind_group(0, &self.images_bg, &[]);
            pass.dispatch_workgroups((self.bodies * 2).div_ceil(64), 1, 1);
        }
        {
            let timestamp_writes = self.post.timestamps("trace");
            let mut pass =
                enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("trace"), timestamp_writes });
            pass.set_pipeline(&self.trace_pipeline);
            pass.set_bind_group(0, &hdr.core_bg, &[]);
            pass.set_bind_group(1, self.near.bind_group(), &[]);
            pass.set_bind_group(2, self.atmo.bind_group(), &[]);
            pass.set_bind_group(3, self.nebula.bind_group(), &[]);
            pass.set_bind_group(4, self.ship.bind_group(), &[]);
            pass.set_bind_group(5, self.lens.bind_group(), &[]);
            pass.dispatch_workgroups(w.div_ceil(TRACE_TILE), h.div_ceil(TRACE_TILE), 1);
        }
        self.post.encode(&mut enc, &out_view);
        self.encode_capture(&mut enc);
        self.queue.submit([enc.finish()]);
        // Deliver finished read-backs (exposure) without waiting.
        let _ = self.device.poll(wgpu::PollType::Poll);
        if let Some(frame) = frame {
            self.queue.present(frame);
        }
        self.frame_index = self.frame_index.wrapping_add(1);
    }

    /// Headless: ask for the next frame's pixels (RGBA8, sRGB encoded).
    pub fn request_capture(&self) {
        let mut c = self.capture();
        if matches!(*c, Capture::Idle | Capture::Ready(_)) {
            *c = Capture::Requested;
        }
    }

    fn capture(&self) -> MutexGuard<'_, Capture> {
        self.capture.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Width, height and RGBA8 pixels of a finished capture.
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
