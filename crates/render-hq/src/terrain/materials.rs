//! The ground's materials at centimetre scale: scanned texture sets (CC0,
//! from Poly Haven; see `assets/ground/README.md`) giving each material its
//! albedo variation, normals, relief, ambient occlusion and roughness. The
//! material's colour itself stays spectral (the biome's reflectance in
//! `planet.wgsl`); a texture only modulates its brightness by the scan's
//! luminance relative to its mean.
//!
//! Textures repeat on the cube-sphere's face grid, the one the terrain tiles
//! lie on: one repeat spans a tile of the material's `repeat_level`, so a
//! texture coordinate is the tile's index modulo a power of two plus its
//! place in the tile, exact at any distance and seamless across tiles (no
//! anchor involved).
//!
//! To add a material, put its maps in `assets/ground/<name>/` and add a line
//! to [`MATERIALS`]; the shaders name it `GM_<SLOT>` (constants generated
//! by [`wgsl_constants`]) and the biomes weigh it in by that name.

use zune_core::bytestream::ZCursor;
use zune_core::colorspace::ColorSpace;
use zune_core::options::DecoderOptions;
use zune_jpeg::JpegDecoder;

/// A ground material: a scanned texture set and how it's laid.
pub struct GroundMaterial {
    /// The shaders' name for it: `GM_<slot>`.
    pub slot: &'static str,
    /// Asset directory under `assets/ground/`.
    pub name: &'static str,
    /// JPEG maps, [`TEXTURE_SIZE`]² each: albedo (sRGB), normals (OpenGL
    /// convention, +y up the image), displacement (grey), and ambient
    /// occlusion / roughness / metalness packed in r, g, b.
    pub maps: [&'static [u8]; 4],
    /// One repeat spans a tile of this level (22 is ≈ 2.4 m on an Earth).
    pub repeat_level: u32,
    /// Height between the displacement map's black and white, m.
    pub relief_m: f32,
}

macro_rules! scanned {
    ($slot:literal, $name:literal, $level:expr, $relief:expr) => {
        GroundMaterial {
            slot: $slot,
            name: $name,
            maps: [
                include_bytes!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/assets/ground/",
                    $name,
                    "/",
                    $name,
                    "_diff_1k.jpg"
                )),
                include_bytes!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/assets/ground/",
                    $name,
                    "/",
                    $name,
                    "_nor_gl_1k.jpg"
                )),
                include_bytes!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/assets/ground/",
                    $name,
                    "/",
                    $name,
                    "_disp_1k.jpg"
                )),
                include_bytes!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/assets/ground/",
                    $name,
                    "/",
                    $name,
                    "_arm_1k.jpg"
                )),
            ],
            repeat_level: $level,
            relief_m: $relief,
        }
    };
}

/// The material library, in texture-array layer order.
pub static MATERIALS: &[GroundMaterial] = &[
    // Coarse beach sand with shell and coral debris (3.9 m scanned).
    scanned!("SAND", "coast_sand_04", 21, 0.03),
    // Damp, compacted beach sand (1.9 m).
    scanned!("WET_SAND", "damp_beach_sand_02", 22, 0.01),
    // Dark fractured rock, standing in for basalt (2.4 m).
    scanned!("ROCK", "dark_rock", 22, 0.12),
    // Soil with stones and gravel, under vegetation (3.2 m).
    scanned!("SOIL", "forest_ground_04", 21, 0.06),
];

/// Wavelengths (m) of the anchored noise that varies the ground's
/// brightness over metres to tens of metres (and clumps the plants), so the
/// textures' repeats don't show; and the seed offset of its octaves.
pub const VARIATION_M: [f64; 4] = [40.0, 12.0, 3.5, 1.0];
const VARIATION_SEED: u32 = 7100;

/// The variation octaves anchored at `anchor_km` (body-fixed) for a planet
/// with `seed`.
pub fn variation_octaves(anchor_km: kerr::vec3::V3, seed: u32) -> [super::anchor::OctaveGpu; 4] {
    let octaves =
        VARIATION_M.iter().enumerate().map(|(i, m)| (1000.0 / m, seed.wrapping_add(VARIATION_SEED + i as u32)));
    let a = super::anchor::Anchor::new(anchor_km, octaves);
    std::array::from_fn(|i| a.octaves[i])
}

/// Texture size (texels a side) of every map.
pub const TEXTURE_SIZE: u32 = 1024;
/// Room in the shaders' parameter table.
pub const MAX_MATERIALS: usize = 8;

/// `const GM_<SLOT>: u32 = <layer>u;` for each material, and `GM_COUNT`.
pub fn wgsl_constants() -> String {
    let mut s = String::from("// Ground material slots (generated from `terrain/materials.rs`).\n");
    for (i, m) in MATERIALS.iter().enumerate() {
        s.push_str(&format!("const GM_{}: u32 = {i}u;\n", m.slot));
    }
    s.push_str(&format!("const GM_COUNT: u32 = {}u;\n", MATERIALS.len()));
    s
}

/// Mirrors `struct GroundMaterials` in `terrain_rq.wgsl`: per material its
/// repeat level, relief (m), the albedo map's mean luminance (linear) and
/// the displacement map's mean (0–1).
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MaterialsGpu {
    pub params: [[f32; 4]; MAX_MATERIALS],
}

/// A decoded map: `TEXTURE_SIZE`² RGB texels.
fn decode(jpeg: &[u8]) -> Result<Vec<[u8; 3]>, String> {
    let options = DecoderOptions::default().jpeg_set_out_colorspace(ColorSpace::RGB);
    let mut decoder = JpegDecoder::new_with_options(ZCursor::new(jpeg), options);
    let pixels = decoder.decode().map_err(|e| format!("{e:?}"))?;
    let info = decoder.info().ok_or("no image info")?;
    let n = (TEXTURE_SIZE * TEXTURE_SIZE) as usize;
    if (info.width as u32, info.height as u32) != (TEXTURE_SIZE, TEXTURE_SIZE) {
        return Err(format!("{}×{}, not {TEXTURE_SIZE}²", info.width, info.height));
    }
    let channels = pixels.len() / n;
    Ok((0..n)
        .map(|i| match channels {
            1 => [pixels[i]; 3],
            _ => [pixels[channels * i], pixels[channels * i + 1], pixels[channels * i + 2]],
        })
        .collect())
}

fn srgb_to_linear(c: u8) -> f32 {
    let c = c as f32 / 255.0;
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

fn linear_to_srgb(c: f32) -> u8 {
    let c = c.clamp(0.0, 1.0);
    let s = if c <= 0.003_130_8 { 12.92 * c } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 };
    (s * 255.0 + 0.5) as u8
}

/// The material's two layers as RGBA8 texels, full size: albedo (sRGB rgb,
/// displacement in alpha) and detail (normal x, normal y, occlusion,
/// roughness); and its parameters for [`MaterialsGpu`].
pub struct Decoded {
    pub albedo: Vec<[u8; 4]>,
    pub detail: Vec<[u8; 4]>,
    pub params: [f32; 4],
}

pub fn decode_material(m: &GroundMaterial) -> Result<Decoded, String> {
    let [diff, nor, disp, arm] = m.maps.map(decode);
    let err = |what: &str, e: String| format!("ground material {} ({what}): {e}", m.name);
    let (diff, nor, disp, arm) = (
        diff.map_err(|e| err("albedo", e))?,
        nor.map_err(|e| err("normals", e))?,
        disp.map_err(|e| err("displacement", e))?,
        arm.map_err(|e| err("ao/roughness", e))?,
    );
    let n = diff.len() as f64;
    let luminance = |p: &[u8; 3]| {
        let [r, g, b] = p.map(srgb_to_linear);
        0.2126 * r + 0.7152 * g + 0.0722 * b
    };
    let y_mean = diff.iter().map(|p| luminance(p) as f64).sum::<f64>() / n;
    let h_mean = disp.iter().map(|p| p[0] as f64 / 255.0).sum::<f64>() / n;
    let albedo = diff.iter().zip(&disp).map(|(c, h)| [c[0], c[1], c[2], h[0]]).collect();
    let detail = nor.iter().zip(&arm).map(|(nm, a)| [nm[0], nm[1], a[0], a[1]]).collect();
    Ok(Decoded { albedo, detail, params: [m.repeat_level as f32, m.relief_m, y_mean as f32, h_mean as f32] })
}

/// Box-filtered mip chain of RGBA8 texels (`srgb`: the rgb average in
/// linear light), from `size`² down to 1².
pub fn mips(level0: &[[u8; 4]], size: u32, srgb: bool) -> Vec<Vec<[u8; 4]>> {
    let mut out = vec![level0.to_vec()];
    let mut s = size as usize;
    while s > 1 {
        let prev = out.last().expect("a level");
        let h = s / 2;
        let next = (0..h * h)
            .map(|k| {
                let (x, y) = (2 * (k % h), 2 * (k / h));
                let taps = [prev[y * s + x], prev[y * s + x + 1], prev[(y + 1) * s + x], prev[(y + 1) * s + x + 1]];
                std::array::from_fn(|c| {
                    if srgb && c < 3 {
                        linear_to_srgb(taps.iter().map(|t| srgb_to_linear(t[c])).sum::<f32>() / 4.0)
                    } else {
                        ((taps.iter().map(|t| t[c] as u32).sum::<u32>() + 2) / 4) as u8
                    }
                })
            })
            .collect();
        out.push(next);
        s = h;
    }
    out
}

/// The library on the GPU: two texture arrays with a layer per material,
/// a repeating trilinear sampler and the parameter table.
pub struct MaterialTextures {
    pub albedo: wgpu::TextureView,
    pub detail: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
    pub params: wgpu::Buffer,
}

impl MaterialTextures {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Result<Self, String> {
        assert!(MATERIALS.len() <= MAX_MATERIALS);
        let decoded = MATERIALS.iter().map(decode_material).collect::<Result<Vec<_>, _>>()?;
        let levels = TEXTURE_SIZE.ilog2() + 1;
        let texture = |label, format| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: TEXTURE_SIZE,
                    height: TEXTURE_SIZE,
                    depth_or_array_layers: MATERIALS.len() as u32,
                },
                mip_level_count: levels,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            })
        };
        let albedo = texture("ground albedo", wgpu::TextureFormat::Rgba8UnormSrgb);
        let detail = texture("ground detail", wgpu::TextureFormat::Rgba8Unorm);
        let upload = |t: &wgpu::Texture, layer: u32, chain: Vec<Vec<[u8; 4]>>| {
            for (level, texels) in chain.iter().enumerate() {
                let s = (TEXTURE_SIZE >> level).max(1);
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: t,
                        mip_level: level as u32,
                        origin: wgpu::Origin3d { x: 0, y: 0, z: layer },
                        aspect: wgpu::TextureAspect::All,
                    },
                    bytemuck::cast_slice(texels),
                    wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4 * s), rows_per_image: Some(s) },
                    wgpu::Extent3d { width: s, height: s, depth_or_array_layers: 1 },
                );
            }
        };
        let mut table = MaterialsGpu { params: [[0.0; 4]; MAX_MATERIALS] };
        for (i, d) in decoded.iter().enumerate() {
            upload(&albedo, i as u32, mips(&d.albedo, TEXTURE_SIZE, true));
            upload(&detail, i as u32, mips(&d.detail, TEXTURE_SIZE, false));
            table.params[i] = d.params;
        }
        let view = |t: &wgpu::Texture| {
            t.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            })
        };
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ground"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        use wgpu::util::DeviceExt;
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ground materials"),
            contents: bytemuck::bytes_of(&table),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        Ok(Self { albedo: view(&albedo), detail: view(&detail), sampler, params })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every map decodes to its full size, and the parameters are sane.
    #[test]
    fn library_decodes() {
        for m in MATERIALS {
            let d = decode_material(m).unwrap();
            let n = (TEXTURE_SIZE * TEXTURE_SIZE) as usize;
            assert_eq!((d.albedo.len(), d.detail.len()), (n, n), "{}", m.name);
            let [level, relief, y, h] = d.params;
            assert!(
                level >= 16.0 && relief > 0.0 && y > 0.005 && y < 0.9 && h > 0.05 && h < 0.95,
                "{}: {:?}",
                m.name,
                d.params
            );
            // Normal maps point out of the surface on average (+z is in the
            // blue channel, not kept; x and y centre on 0.5).
            let mean = |c: usize| d.detail.iter().map(|t| t[c] as f64).sum::<f64>() / n as f64 / 255.0;
            assert!((mean(0) - 0.5).abs() < 0.05 && (mean(1) - 0.5).abs() < 0.05, "{}", m.name);
        }
        let s = wgsl_constants();
        assert!(s.contains("const GM_SAND: u32 = 0u;") && s.contains("const GM_COUNT: u32 = 4u;"));
    }

    /// Mips halve down to one texel and keep the mean (in linear light for
    /// sRGB).
    #[test]
    fn mip_chain() {
        let size = 8;
        let texels: Vec<[u8; 4]> = (0..64).map(|i| [(i * 4) as u8, 128, 0, 255]).collect();
        let chain = mips(&texels, size, false);
        assert_eq!(chain.len(), 4);
        assert_eq!(chain[3].len(), 1);
        let mean = texels.iter().map(|t| t[0] as f64).sum::<f64>() / 64.0;
        assert!((chain[3][0][0] as f64 - mean).abs() <= 2.0);
        let srgb = mips(&[[0, 0, 0, 0], [255, 255, 255, 255], [0, 0, 0, 0], [255, 255, 255, 255]], 2, true);
        // Half of full white in linear light is 188 in sRGB.
        assert!((srgb[1][0][0] as i32 - 188).abs() <= 1, "{:?}", srgb[1][0]);
    }
}
