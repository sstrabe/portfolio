//! A 2D overlay drawn over the finished image: text, lines, rings, discs
//! and rectangles for the HUD and the map (`wgsl/overlay.wgsl`).
//!
//! Immediate mode: callers add shapes each frame in output pixels (origin
//! top left) with sRGB colours and straight alpha; everything is drawn in
//! one pass after tone mapping and then cleared. Text uses the public-domain
//! 8×8 bitmap font of the `font8x8` crate (ASCII, Latin-1 and Greek), scaled
//! by whole pixels so it stays crisp.

use bytemuck::{Pod, Zeroable};
use font8x8::UnicodeFonts;
use std::collections::HashMap;

/// Colour: sRGB red, green, blue and alpha, 0–1.
pub type Rgba = [f32; 4];

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct Vertex {
    pos: [f32; 2],
    uv: [f32; 2],
    color: [f32; 4],
    /// Shape: 0 solid, 1 glyph (uv in the atlas), 2 ring or disc (uv in
    /// [−1, 1]², inner radius as a fraction in `param`).
    mode: f32,
    param: f32,
    /// Half a pixel in uv units (anti-aliasing of rings).
    aa: f32,
    _pad: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct Screen {
    size: [f32; 2],
    /// 1 when the target stores sRGB (the shader must linearise colours).
    srgb_target: f32,
    _pad: f32,
}

const ATLAS_COLS: u32 = 16;
const GLYPH: u32 = 8;
const MAX_VERTICES: usize = 6 * 20_000;

pub struct Overlay {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    screen: wgpu::Buffer,
    vertices: wgpu::Buffer,
    verts: Vec<Vertex>,
    glyphs: HashMap<char, u32>,
    atlas_rows: u32,
    srgb_target: bool,
}

/// The characters the atlas holds besides ASCII.
const EXTRA: &str = "°·×±µ²γτβΔπθφωαδλ";

fn glyph_bits(c: char) -> Option<[u8; 8]> {
    font8x8::BASIC_FONTS.get(c).or_else(|| font8x8::LATIN_FONTS.get(c)).or_else(|| font8x8::GREEK_FONTS.get(c))
}

impl Overlay {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Self {
        // Font atlas: one 8×8 cell per character, 16 per row.
        let chars: Vec<char> = (32u8..127).map(char::from).chain(EXTRA.chars()).collect();
        let rows = (chars.len() as u32).div_ceil(ATLAS_COLS);
        let (w, h) = (ATLAS_COLS * GLYPH, rows * GLYPH);
        let mut pixels = vec![0u8; (w * h) as usize];
        let mut glyphs = HashMap::new();
        for (i, &c) in chars.iter().enumerate() {
            let Some(bits) = glyph_bits(c) else { continue };
            let (cx, cy) = (i as u32 % ATLAS_COLS * GLYPH, i as u32 / ATLAS_COLS * GLYPH);
            for (y, row) in bits.iter().enumerate() {
                for x in 0..8 {
                    if row >> x & 1 == 1 {
                        pixels[((cy + y as u32) * w + cx + x) as usize] = 255;
                    }
                }
            }
            glyphs.insert(c, i as u32);
        }
        let atlas = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("overlay font"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            atlas.as_image_copy(),
            &pixels,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w), rows_per_image: Some(h) },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("overlay font"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let screen = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("overlay screen"),
            size: std::mem::size_of::<Screen>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let vertices = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("overlay vertices"),
            size: (MAX_VERTICES * std::mem::size_of::<Vertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("overlay"),
            source: wgpu::ShaderSource::Wgsl(crate::shaders::overlay().into()),
        });
        let attributes = wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2, 2 => Float32x4, 3 => Float32x4];
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("overlay"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_overlay"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &attributes,
                })],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_overlay"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let atlas_view = atlas.create_view(&Default::default());
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("overlay"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: screen.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&atlas_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&sampler) },
            ],
        });
        Self {
            pipeline,
            bind_group,
            screen,
            vertices,
            verts: Vec::new(),
            glyphs,
            atlas_rows: rows,
            srgb_target: format.is_srgb(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.verts.is_empty()
    }

    fn quad(&mut self, corners: [[f32; 2]; 4], uv: [[f32; 2]; 4], color: Rgba, mode: f32, param: f32, aa: f32) {
        if self.verts.len() + 6 > MAX_VERTICES {
            return;
        }
        for i in [0usize, 1, 2, 0, 2, 3] {
            self.verts.push(Vertex { pos: corners[i], uv: uv[i], color, mode, param, aa, _pad: 0.0 });
        }
    }

    /// A filled rectangle.
    pub fn rect(&mut self, min: [f32; 2], max: [f32; 2], color: Rgba) {
        let c = [min, [max[0], min[1]], max, [min[0], max[1]]];
        self.quad(c, [[0.0; 2]; 4], color, 0.0, 0.0, 0.0);
    }

    /// A filled triangle.
    pub fn triangle(&mut self, a: [f32; 2], b: [f32; 2], c: [f32; 2], color: Rgba) {
        if self.verts.len() + 3 > MAX_VERTICES {
            return;
        }
        for pos in [a, b, c] {
            self.verts.push(Vertex { pos, uv: [0.0; 2], color, mode: 0.0, param: 0.0, aa: 0.0, _pad: 0.0 });
        }
    }

    /// A filled convex polygon, fanned out from `centre` (which must lie
    /// inside it).
    pub fn fan(&mut self, centre: [f32; 2], outline: &[[f32; 2]], color: Rgba) {
        for i in 0..outline.len() {
            self.triangle(centre, outline[i], outline[(i + 1) % outline.len()], color);
        }
    }

    /// A straight, anti-aliased line `width` pixels wide (thinner than a
    /// pixel it fades instead).
    pub fn line(&mut self, a: [f32; 2], b: [f32; 2], width: f32, color: Rgba) {
        let d = [b[0] - a[0], b[1] - a[1]];
        let len = (d[0] * d[0] + d[1] * d[1]).sqrt();
        if len < 1e-3 || !len.is_finite() {
            return;
        }
        let half = 0.5 * width.max(1.0);
        let color = [color[0], color[1], color[2], color[3] * width.min(1.0)];
        // One pixel of margin on each side for the soft edge.
        let e = half + 1.0;
        let n = [-d[1] / len * e, d[0] / len * e];
        let c = [
            [a[0] + n[0], a[1] + n[1]],
            [b[0] + n[0], b[1] + n[1]],
            [b[0] - n[0], b[1] - n[1]],
            [a[0] - n[0], a[1] - n[1]],
        ];
        self.quad(c, [[0.0, 1.0], [0.0, 1.0], [0.0, -1.0], [0.0, -1.0]], color, 3.0, e, half);
    }

    /// A polyline through `points`.
    pub fn polyline(&mut self, points: &[[f32; 2]], width: f32, color: Rgba) {
        for w in points.windows(2) {
            self.line(w[0], w[1], width, color);
        }
    }

    /// A ring of outer radius `r` and `width` pixels, or a filled disc when
    /// `width` is at least `r`.
    pub fn ring(&mut self, c: [f32; 2], r: f32, width: f32, color: Rgba) {
        let r = r.max(0.5);
        let e = r + 1.0;
        let corners = [[c[0] - e, c[1] - e], [c[0] + e, c[1] - e], [c[0] + e, c[1] + e], [c[0] - e, c[1] + e]];
        let k = e / r;
        let uv = [[-k, -k], [k, -k], [k, k], [-k, k]];
        let inner = ((r - width) / r).max(0.0);
        self.quad(corners, uv, color, 2.0, inner, 0.5 / r);
    }

    pub fn disc(&mut self, c: [f32; 2], r: f32, color: Rgba) {
        self.ring(c, r, r, color);
    }

    /// Width of `s` in pixels at `scale` pixels per font pixel.
    pub fn text_width(s: &str, scale: f32) -> f32 {
        s.chars().count() as f32 * GLYPH as f32 * scale
    }

    /// Text with its top-left corner at `pos`, `scale` pixels per font
    /// pixel (whole numbers stay crisp), with a dark shadow for contrast.
    pub fn text(&mut self, pos: [f32; 2], s: &str, scale: f32, color: Rgba) {
        let shadow = [0.0, 0.0, 0.0, 0.75 * color[3]];
        self.text_plain([pos[0] + scale, pos[1] + scale], s, scale, shadow);
        self.text_plain(pos, s, scale, color);
    }

    fn text_plain(&mut self, pos: [f32; 2], s: &str, scale: f32, color: Rgba) {
        let g = GLYPH as f32 * scale;
        let (aw, ah) = ((ATLAS_COLS * GLYPH) as f32, (self.atlas_rows * GLYPH) as f32);
        for (i, c) in s.chars().enumerate() {
            let Some(&idx) = self.glyphs.get(&c).or_else(|| self.glyphs.get(&'?')) else { continue };
            if c == ' ' {
                continue;
            }
            let x = pos[0] + i as f32 * g;
            let (u0, v0) = ((idx % ATLAS_COLS * GLYPH) as f32 / aw, (idx / ATLAS_COLS * GLYPH) as f32 / ah);
            let (u1, v1) = (u0 + GLYPH as f32 / aw, v0 + GLYPH as f32 / ah);
            let corners = [[x, pos[1]], [x + g, pos[1]], [x + g, pos[1] + g], [x, pos[1] + g]];
            self.quad(corners, [[u0, v0], [u1, v0], [u1, v1], [u0, v1]], color, 1.0, 0.0, 0.0);
        }
    }

    /// Draw everything added since the last call over `target`, then clear.
    pub fn encode(
        &mut self,
        queue: &wgpu::Queue,
        enc: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        size: (u32, u32),
    ) {
        if self.verts.is_empty() {
            return;
        }
        let screen = Screen {
            size: [size.0 as f32, size.1 as f32],
            srgb_target: if self.srgb_target { 1.0 } else { 0.0 },
            _pad: 0.0,
        };
        queue.write_buffer(&self.screen, 0, bytemuck::bytes_of(&screen));
        queue.write_buffer(&self.vertices, 0, bytemuck::cast_slice(&self.verts));
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("overlay"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, self.vertices.slice(..));
        pass.draw(0..self.verts.len() as u32, 0..1);
        drop(pass);
        self.verts.clear();
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn layouts_and_glyphs() {
        assert_eq!(std::mem::size_of::<super::Vertex>(), 48);
        for c in super::EXTRA.chars().chain(['A', 'z', '0', '~']) {
            assert!(super::glyph_bits(c).is_some(), "no glyph for {c:?}");
        }
    }
}
