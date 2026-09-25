//! Atmospheres and clouds: bind group 2 of the trace pass (see
//! `wgsl/atmosphere.wgsl` and `wgsl/clouds.wgsl`). Slot `i` describes
//! near-field planet slot `i`.
//!
//! The sky follows Hillaire (2020), "A Scalable and Production Ready Sky and
//! Atmosphere Rendering Technique", carried in the renderer's 16 spectral
//! bins instead of RGB. Per planet slot, whenever the near-field planet set
//! changes, compute passes (`wgsl/atmo_luts.wgsl`, `wgsl/cloud_gen.wgsl`)
//! build:
//!
//! - the **transmittance** table (256 × 64): density-weighted columns of the
//!   gas, the aerosols and ozone from (h, μ) to the top of the atmosphere.
//!   Transmittance is exp(−Σ β_i(λ) C_i), so one rgba32float texel gives all
//!   16 bins exactly, at a quarter of the fetches of 4 × RGBA16F;
//! - the **multiple-scattering** table Ψ_ms(h, μ_s) (32 × 32, 4 layers of
//!   4 bins): light scattered twice or more, summed as a geometric series
//!   of isotropic second-order scattering;
//! - the **sky irradiance** on a horizontal surface E(h, μ_s) (32 × 32);
//! - the **cloud weather map**: a cube map of coverage, top height, density
//!   and cellularity in the planet's body-fixed frame (6 × 512²).
//!
//! A tileable 128³ Perlin–Worley noise volume for cloud shapes and detail
//! is built once.
//!
//! Optical constants are computed here, in f64, and uploaded per bin as
//! coefficients in km⁻¹ at the reference level (the surface, or a giant's
//! cloud deck):
//! - **Rayleigh** from first principles per gas: σ = 24π³/(N_s²λ⁴)
//!   ((n² − 1)/(n² + 2))² F_K with measured dispersion n(λ) and King
//!   factors F_K, mixed by the planet kind's composition. Earth's air gives
//!   β(550 nm) = 1.15 × 10⁻⁵ m⁻¹ (Bucholtz 1995);
//! - **aerosols (Mie)**: Earth's haze, 4.4 × 10⁻³ km⁻¹ extinction at 550 nm,
//!   with an Ångström exponent that flattens for coarse dust; dust absorbs
//!   blue (iron oxides), so desert skies scatter a reddish tan;
//! - **ozone**: the Chappuis band cross-section (Serdyuchenko et al. 2014)
//!   in a 300 Dobson unit tent profile peaking at 25 km;
//! - **methane**: absorption after Karkoschka (1994), bands at 543, 576,
//!   619, 703, 727 and 790+ nm, mixed like the bulk gas.

use crate::FrameContext;
use crate::near::MAX_PLANETS;
use crate::shaders;
use crate::spectrum::{BIN_WIDTH, BINS, LAMBDA_MIN};
use bytemuck::{Pod, Zeroable};
use kerr::planets::{self, PlanetKind};

pub const TRANSMITTANCE_SIZE: (u32, u32) = (256, 64);
pub const MULTISCATTER_SIZE: u32 = 32;
pub const IRRADIANCE_SIZE: u32 = 32;
pub const CLOUD_MAP_SIZE: u32 = 512;
/// Planets that can have a cloud map at once (the nearest cloudy ones).
pub const CLOUD_MAPS: usize = 8;
pub const NOISE_SIZE: u32 = 128;
/// Texture layers per planet slot for spectral tables (4 × 4 bins).
const LAYERS: u32 = 4;
/// `AtmosphereGpu::ids[2]` of a planet without a cloud map.
pub const NO_CLOUD_MAP: u32 = u32::MAX;

/// Loschmidt's number: molecules per m³ of an ideal gas at 0 °C and 1 atm,
/// the density the refractivities below refer to.
const LOSCHMIDT: f64 = 2.686_78e25;
/// Molecules per m³ of Earth's sea-level air (15 °C, 1 atm): what
/// `Atmosphere::rayleigh_density` = 1 means.
pub const N_AIR: f64 = 2.547e25;
/// Aerosol extinction of Earth's lower atmosphere at 550 nm, km⁻¹.
const MIE_EXTINCTION_550: f64 = 4.4e-3;
/// Molecules per m² in one Dobson unit.
const DOBSON: f64 = 2.687e20;
/// Earth's ozone column (Dobson units), in a tent profile of this half
/// width around this altitude.
const OZONE_COLUMN_DU: f64 = 300.0;
pub const OZONE_PEAK_KM: f64 = 25.0;
pub const OZONE_HALF_WIDTH_KM: f64 = 15.0;
/// Methane column (km-amagat) that makes `methane = 1` look like Neptune:
/// the 727 nm band becomes opaque and 619 nm loses most of its light, while
/// blue is untouched.
const METHANE_COLUMN_KM_AMAGAT: f64 = 4.0;
/// Albedo of a thick cloud deck seen from above (two-stream, τ ≈ 15,
/// g ≈ 0.85), for the light clouds send back into the sky.
const CLOUD_ALBEDO: f64 = 0.55;

/// Gases whose Rayleigh scattering is modelled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gas {
    N2,
    O2,
    Ar,
    Co2,
    H2,
    He,
    Ch4,
}

impl Gas {
    /// Refractivity n − 1 at 0 °C and 1 atm, `lambda` in nm:
    /// N₂ Peck & Khanna (1966), O₂ Bates (1984), Ar Peck & Fisher (1964),
    /// H₂ Peck & Huang (1977), He Mansfield & Peck (1969). CO₂ and CH₄
    /// follow N₂'s dispersion scaled to their measured refractivity at
    /// 589 nm (4.49 × 10⁻⁴ and 4.41 × 10⁻⁴), within ~1% over the visible.
    pub fn refractivity(self, lambda_nm: f64) -> f64 {
        let s2 = (1000.0 / lambda_nm).powi(2);
        let n2 = 1e-8 * (6855.2 + 3_243_157.0 / (144.0 - s2));
        match self {
            Gas::N2 => n2,
            Gas::O2 => 1e-8 * (20_564.8 + 248_089.9 / (40.9 - s2)),
            Gas::Ar => 1e-8 * (6432.135 + 2_860_602.1 / (144.0 - s2)),
            Gas::Co2 => n2 * 1.507,
            Gas::H2 => 0.014_895_6 / (180.7 - s2) + 0.004_903_7 / (92.0 - s2),
            Gas::He => 0.014_700_91 / (423.98 - s2),
            Gas::Ch4 => n2 * 1.480,
        }
    }

    /// King correction factor (6 + 3ρ)/(6 − 7ρ) for the molecule's
    /// depolarisation ρ: N₂ and O₂ Bates (1984), CO₂ Alms et al. (1975),
    /// 1 for atoms; H₂ and CH₄ are nearly isotropic.
    pub fn king_factor(self, lambda_nm: f64) -> f64 {
        let s2 = (1000.0 / lambda_nm).powi(2);
        match self {
            Gas::N2 => 1.034 + 3.17e-4 * s2,
            Gas::O2 => 1.096 + 1.385e-3 * s2 + 1.448e-4 * s2 * s2,
            Gas::Co2 => 1.1364 + 2.53e-3 * s2,
            Gas::H2 => 1.022,
            Gas::Ar | Gas::He | Gas::Ch4 => 1.0,
        }
    }

    /// Rayleigh cross-section per molecule, m²:
    /// σ = 24π³/(N_s² λ⁴) ((n² − 1)/(n² + 2))² F_K, with n measured at the
    /// number density N_s.
    pub fn rayleigh_cross_section(self, lambda_nm: f64) -> f64 {
        let l = lambda_nm * 1e-9;
        let n = 1.0 + self.refractivity(lambda_nm);
        let lorentz = (n * n - 1.0) / (n * n + 2.0);
        24.0 * std::f64::consts::PI.powi(3) / (LOSCHMIDT * LOSCHMIDT * l.powi(4))
            * lorentz
            * lorentz
            * self.king_factor(lambda_nm)
    }
}

/// Mole fractions of the scattering gas of a planet kind: N₂–O₂ air on
/// ocean worlds, N₂–CO₂ on abiotic rocky worlds, CO₂ on desert (Mars-like)
/// and lava (Venus-like) worlds, N₂–CH₄ on icy ones (Titan, Triton), H₂–He
/// on giants.
pub fn composition(kind: PlanetKind) -> &'static [(Gas, f64)] {
    use Gas::*;
    match kind {
        PlanetKind::Ocean => &[(N2, 0.7808), (O2, 0.2095), (Ar, 0.0093), (Co2, 0.0004)],
        PlanetKind::Rocky => &[(N2, 0.95), (Co2, 0.04), (Ar, 0.01)],
        PlanetKind::Desert => &[(Co2, 0.95), (N2, 0.03), (Ar, 0.02)],
        PlanetKind::Lava => &[(Co2, 0.965), (N2, 0.035)],
        PlanetKind::Ice => &[(N2, 0.97), (Ch4, 0.03)],
        PlanetKind::GasGiant => &[(H2, 0.86), (He, 0.136), (Ch4, 0.004)],
        PlanetKind::IceGiant => &[(H2, 0.80), (He, 0.185), (Ch4, 0.015)],
    }
}

/// Rayleigh scattering coefficient of a gas mixture at Earth's sea-level
/// number density, km⁻¹, at `lambda` nm.
pub fn rayleigh_coefficient(gases: &[(Gas, f64)], lambda_nm: f64) -> f64 {
    gases.iter().map(|&(g, x)| x * g.rayleigh_cross_section(lambda_nm)).sum::<f64>() * N_AIR * 1000.0
}

/// Linear interpolation in a table sampled every `step` nm from `first`.
fn table(values: &[f64], first: f64, step: f64, lambda: f64) -> f64 {
    let u = ((lambda - first) / step).clamp(0.0, (values.len() - 1) as f64);
    let i = (u.floor() as usize).min(values.len() - 2);
    let f = u - i as f64;
    values[i] * (1.0 - f) + values[i + 1] * f
}

/// Ozone absorption cross-section, m² per molecule (Chappuis band), from
/// 360 to 830 nm every 10 nm (Serdyuchenko et al. 2014, as tabulated by
/// Bruneton 2017).
pub fn ozone_cross_section(lambda_nm: f64) -> f64 {
    const O3: [f64; 48] = [
        1.18e-27, 2.182e-28, 2.818e-28, 6.636e-28, 1.527e-27, 2.763e-27, 5.52e-27, 8.451e-27, 1.582e-26, 2.316e-26,
        3.669e-26, 4.924e-26, 7.752e-26, 9.016e-26, 1.48e-25, 1.602e-25, 2.139e-25, 2.755e-25, 3.091e-25, 3.5e-25,
        4.266e-25, 4.672e-25, 4.398e-25, 4.701e-25, 5.019e-25, 4.305e-25, 3.74e-25, 3.215e-25, 2.662e-25, 2.238e-25,
        1.852e-25, 1.473e-25, 1.209e-25, 9.423e-26, 7.455e-26, 6.566e-26, 5.105e-26, 4.15e-26, 4.228e-26, 3.237e-26,
        2.451e-26, 2.801e-26, 2.534e-26, 1.624e-26, 1.465e-26, 2.078e-26, 1.383e-26, 7.105e-27,
    ];
    table(&O3, 360.0, 10.0, lambda_nm)
}

/// Methane absorption coefficient, (km-amagat)⁻¹, from 390 to 800 nm every
/// 10 nm: a weak continuum in the blue and the bands of Karkoschka (1994)
/// that make Uranus and Neptune blue-green.
pub fn methane_absorption(lambda_nm: f64) -> f64 {
    const CH4: [f64; 42] = [
        0.0010, 0.0010, 0.0012, 0.0013, 0.0015, 0.0018, 0.0022, 0.0025, 0.0028, 0.0045, // 390–480
        0.0060, 0.0045, 0.0065, 0.0045, 0.0060, 0.030, 0.012, 0.010, 0.030, 0.050, // 490–580
        0.025, 0.045, 0.10, 0.40, 0.12, 0.06, 0.035, 0.06, 0.05, 0.08, // 590–680
        0.12, 0.25, 0.30, 0.90, 2.5, 0.70, 0.15, 0.20, 0.35, 0.55, // 690–780
        1.0, 1.4, // 790–800
    ];
    table(&CH4, 390.0, 10.0, lambda_nm)
}

/// Absorbed fraction of mineral dust's extinction relative to its mean:
/// hematite and goethite absorb strongly below ~550 nm.
fn dust_absorption_shape(lambda_nm: f64) -> f64 {
    0.15 + 2.1 * (-(lambda_nm - 400.0) / 90.0).exp()
}

/// Bin average of `f` (sampled every nanometre).
fn bin_average(k: usize, f: impl Fn(f64) -> f64) -> f64 {
    let lo = LAMBDA_MIN + BIN_WIDTH * k as f64;
    let n = BIN_WIDTH as usize;
    (0..n).map(|i| f(lo + i as f64 + 0.5)).sum::<f64>() / n as f64
}

/// Per-km optical coefficients of an atmosphere at its reference level
/// (surface, or the cloud deck of a giant), per spectral bin.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Coefficients {
    pub rayleigh_scattering: [f64; BINS],
    /// Rayleigh scattering plus the absorption of gases mixed like the bulk
    /// gas (methane).
    pub rayleigh_extinction: [f64; BINS],
    pub mie_scattering: [f64; BINS],
    pub mie_extinction: [f64; BINS],
    /// Ozone absorption at the peak of its layer.
    pub ozone: [f64; BINS],
}

impl Coefficients {
    pub fn new(kind: PlanetKind, a: &planets::Atmosphere) -> Self {
        let gases = composition(kind);
        let mut c = Self {
            rayleigh_scattering: [0.0; BINS],
            rayleigh_extinction: [0.0; BINS],
            mie_scattering: [0.0; BINS],
            mie_extinction: [0.0; BINS],
            ozone: [0.0; BINS],
        };
        // Methane is mixed like the bulk gas, so its absorption per km at
        // the base is the column spread over one scale height.
        let methane = a.methane * METHANE_COLUMN_KM_AMAGAT / a.rayleigh_scale_height_km;
        // Coarse dust scatters nearly greyly; fine haze more in the blue.
        let angstrom = 1.0 - 0.7 * a.dust;
        let ozone_peak = OZONE_COLUMN_DU * DOBSON / (OZONE_HALF_WIDTH_KM * 1000.0) * a.ozone;
        for k in 0..BINS {
            let ray = a.rayleigh_density * bin_average(k, |l| rayleigh_coefficient(gases, l));
            c.rayleigh_scattering[k] = ray;
            c.rayleigh_extinction[k] = ray + methane * bin_average(k, methane_absorption);
            let ext = a.mie_density * MIE_EXTINCTION_550 * bin_average(k, |l| (l / 550.0).powf(-angstrom));
            let absorbed = bin_average(k, |l| {
                (a.mie_absorption * (1.0 + a.dust * (dust_absorption_shape(l) - 1.0))).clamp(0.0, 0.95)
            });
            c.mie_extinction[k] = ext;
            c.mie_scattering[k] = ext * (1.0 - absorbed);
            // Number density (m⁻³) × cross-section (m²) × 1000 m/km.
            c.ozone[k] = ozone_peak * bin_average(k, ozone_cross_section) * 1000.0;
        }
        c
    }
}

/// A spectrum as the shader's `Spectrum` (four vec4s).
pub type SpectrumGpu = [[f32; 4]; 4];

fn spectrum_gpu(s: &[f64; BINS]) -> SpectrumGpu {
    std::array::from_fn(|i| std::array::from_fn(|j| s[4 * i + j] as f32))
}

/// Reflectance of what lies under the air, for the light it sends back up
/// into the sky (multiple scattering, skylight, light under clouds).
fn ground_albedo(kind: PlanetKind) -> f64 {
    match kind {
        PlanetKind::Rocky => 0.15,
        PlanetKind::Ocean => 0.08,
        PlanetKind::Desert => 0.35,
        PlanetKind::Ice => 0.7,
        PlanetKind::Lava => 0.1,
        PlanetKind::GasGiant => 0.5,
        PlanetKind::IceGiant => 0.4,
    }
}

/// Mirrors `struct AtmosphereParams` in `atmo_common.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct AtmosphereGpu {
    pub shape: [f32; 4],
    pub density: [f32; 4],
    pub aerosol: [f32; 4],
    pub clouds: [f32; 4],
    pub extra: [f32; 4],
    pub ids: [u32; 4],
    pub rayleigh_scattering: SpectrumGpu,
    pub rayleigh_extinction: SpectrumGpu,
    pub mie_scattering: SpectrumGpu,
    pub mie_extinction: SpectrumGpu,
    pub ozone: SpectrumGpu,
}

impl AtmosphereGpu {
    /// `cloud_map`: layer of the planet's weather map, or [`NO_CLOUD_MAP`].
    pub fn new(planet: &planets::Planet, a: &planets::Atmosphere, cloud_map: u32) -> Self {
        let (cov, base, thick, tau) =
            a.clouds.map_or((0.0, 0.0, 0.0, 0.0), |c| (c.coverage, c.base_km, c.thickness_km, c.optical_depth));
        let c = Coefficients::new(planet.kind, a);
        let albedo = ground_albedo(planet.kind) * (1.0 - cov) + CLOUD_ALBEDO * cov;
        Self {
            shape: [
                planet.radius_km as f32,
                a.top_km as f32,
                a.rayleigh_scale_height_km as f32,
                a.mie_scale_height_km as f32,
            ],
            density: [a.rayleigh_density as f32, a.mie_density as f32, a.ozone as f32, a.methane as f32],
            aerosol: [a.mie_g as f32, a.mie_absorption as f32, a.dust as f32, 1.0],
            clouds: [cov as f32, base as f32, thick as f32, tau as f32],
            extra: [albedo as f32, OZONE_PEAK_KM as f32, OZONE_HALF_WIDTH_KM as f32, 0.0],
            ids: [planet.seed, planet.kind.index(), cloud_map, 0],
            rayleigh_scattering: spectrum_gpu(&c.rayleigh_scattering),
            rayleigh_extinction: spectrum_gpu(&c.rayleigh_extinction),
            mie_scattering: spectrum_gpu(&c.mie_scattering),
            mie_extinction: spectrum_gpu(&c.mie_extinction),
            ozone: spectrum_gpu(&c.ozone),
        }
    }
}

/// What a slot holds, to notice when its tables must be rebuilt.
#[derive(Clone, Copy, Debug, PartialEq)]
struct SlotKey {
    star: usize,
    generation: u32,
    planet: usize,
}

struct Pass {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    groups: (u32, u32, u32),
}

pub struct Atmospheres {
    layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    params: wgpu::Buffer,
    /// Transmittance, multiple scattering, irradiance, weather maps.
    passes: [Pass; 4],
    keys: Vec<Option<SlotKey>>,
    dirty: bool,
}

fn texture(
    device: &wgpu::Device,
    label: &str,
    size: (u32, u32, u32),
    dimension: wgpu::TextureDimension,
    format: wgpu::TextureFormat,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: size.2 },
        mip_level_count: 1,
        sample_count: 1,
        dimension,
        format,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    })
}

fn view(t: &wgpu::Texture, dimension: wgpu::TextureViewDimension) -> wgpu::TextureView {
    t.create_view(&wgpu::TextureViewDescriptor { dimension: Some(dimension), ..Default::default() })
}

fn tex_entry(binding: u32, v: &wgpu::TextureView) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry { binding, resource: wgpu::BindingResource::TextureView(v) }
}

impl Atmospheres {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        use wgpu::TextureDimension as D;
        use wgpu::TextureFormat as F;
        use wgpu::TextureViewDimension as V;
        let sampled = |binding, view_dimension| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension,
                multisampled: false,
            },
            count: None,
        };
        let sampler_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("atmospheres"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                sampled(1, V::D2Array),
                sampled(2, V::D2Array),
                sampled(3, V::D2Array),
                sampled(4, V::CubeArray),
                sampled(5, V::D3),
                sampler_entry(6),
                sampler_entry(7),
            ],
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("atmospheres"),
            size: (MAX_PLANETS * std::mem::size_of::<AtmosphereGpu>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&params, 0, bytemuck::cast_slice(&[AtmosphereGpu::default(); MAX_PLANETS]));

        let slots = MAX_PLANETS as u32;
        let (tw, th) = TRANSMITTANCE_SIZE;
        let transmittance = texture(device, "atmosphere transmittance", (tw, th, slots), D::D2, F::Rgba32Float);
        let ms = MULTISCATTER_SIZE;
        let multiscatter =
            texture(device, "atmosphere multiple scattering", (ms, ms, slots * LAYERS), D::D2, F::Rgba16Float);
        let irr = IRRADIANCE_SIZE;
        let irradiance =
            texture(device, "atmosphere sky irradiance", (irr, irr, slots * LAYERS), D::D2, F::Rgba16Float);
        let cm = CLOUD_MAP_SIZE;
        let cloud_map = texture(device, "cloud weather maps", (cm, cm, 6 * CLOUD_MAPS as u32), D::D2, F::Rgba8Unorm);
        let ns = NOISE_SIZE;
        let noise = texture(device, "cloud noise", (ns, ns, ns), D::D3, F::Rgba8Unorm);

        let sampler = |label, mode| {
            device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some(label),
                address_mode_u: mode,
                address_mode_v: mode,
                address_mode_w: mode,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            })
        };
        let clamp = sampler("atmosphere clamp", wgpu::AddressMode::ClampToEdge);
        let repeat = sampler("atmosphere repeat", wgpu::AddressMode::Repeat);

        let trans_v = view(&transmittance, V::D2Array);
        let ms_v = view(&multiscatter, V::D2Array);
        let irr_v = view(&irradiance, V::D2Array);
        let map_v = view(&cloud_map, V::D2Array);
        let cube_v = view(&cloud_map, V::CubeArray);
        let noise_v = view(&noise, V::D3);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("atmospheres"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
                tex_entry(1, &trans_v),
                tex_entry(2, &ms_v),
                tex_entry(3, &irr_v),
                tex_entry(4, &cube_v),
                tex_entry(5, &noise_v),
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::Sampler(&clamp) },
                wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::Sampler(&repeat) },
            ],
        });

        // Table passes, one pipeline per entry point of the lookup-table
        // module, each with the bind group layout of what it uses.
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("atmosphere tables"),
            source: wgpu::ShaderSource::Wgsl(shaders::atmosphere_tables().into()),
        });
        let params_entry = || wgpu::BindGroupEntry { binding: 1, resource: params.as_entire_binding() };
        let sampler_bg = || wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::Sampler(&clamp) };
        let pass = |entry: &str, entries: &[wgpu::BindGroupEntry], groups| {
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(entry),
                layout: &pipeline.get_bind_group_layout(0),
                entries,
            });
            Pass { pipeline, bind_group, groups }
        };
        let passes = [
            pass("cs_transmittance", &[params_entry(), tex_entry(2, &trans_v)], (tw / 8, th / 8, slots)),
            // One workgroup of 64 directions per texel.
            pass(
                "cs_multiscatter",
                &[params_entry(), tex_entry(3, &trans_v), tex_entry(4, &ms_v), sampler_bg()],
                (ms, ms, slots),
            ),
            pass(
                "cs_irradiance",
                &[params_entry(), tex_entry(3, &trans_v), tex_entry(5, &ms_v), tex_entry(6, &irr_v), sampler_bg()],
                (irr, irr, slots),
            ),
            pass("cs_cloud_map", &[params_entry(), tex_entry(7, &map_v)], (cm / 8, cm / 8, 6 * slots)),
        ];

        // The noise volume never changes: build it now.
        let noise_pass = pass("cs_cloud_noise", &[tex_entry(8, &view(&noise, V::D3))], (ns / 4, ns / 4, ns / 4));
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("cloud noise") });
        encode_passes(&mut enc, std::slice::from_ref(&noise_pass));
        queue.submit([enc.finish()]);

        Self { layout, bind_group, params, passes, keys: Vec::new(), dirty: true }
    }

    pub fn layout(&self) -> &wgpu::BindGroupLayout {
        &self.layout
    }

    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    /// Upload the parameters when the near-field planets changed.
    pub fn update(&mut self, ctx: &FrameContext) {
        let sel = ctx.near;
        let keys: Vec<Option<SlotKey>> = sel
            .planets
            .iter()
            .map(|p| {
                let s = &sel.systems[p.system].system;
                p.planet(sel).atmosphere.map(|_| SlotKey { star: s.star, generation: s.generation, planet: p.index })
            })
            .collect();
        if keys == self.keys && !self.dirty {
            return;
        }
        self.keys = keys;
        self.dirty = true;
        let mut maps = 0;
        let mut data = [AtmosphereGpu::default(); MAX_PLANETS];
        for (slot, p) in sel.planets.iter().enumerate().take(MAX_PLANETS) {
            let planet = p.planet(sel);
            let Some(a) = &planet.atmosphere else { continue };
            let map = if a.clouds.is_some_and(|c| c.coverage > 0.0) && maps < CLOUD_MAPS {
                maps += 1;
                maps as u32 - 1
            } else {
                NO_CLOUD_MAP
            };
            data[slot] = AtmosphereGpu::new(planet, a, map);
            match std::env::var("KERR_ATMO").as_deref() {
                Ok("off") => data[slot].aerosol[3] = 0.0,
                Ok("noclouds") => data[slot].ids[2] = NO_CLOUD_MAP,
                _ => {}
            }
        }
        ctx.queue.write_buffer(&self.params, 0, bytemuck::cast_slice(&data));
    }

    /// Rebuild the tables and weather maps after the planets changed.
    pub fn encode(&mut self, enc: &mut wgpu::CommandEncoder, _ctx: &FrameContext) {
        if std::mem::take(&mut self.dirty) {
            encode_passes(enc, &self.passes);
        }
    }
}

fn encode_passes(enc: &mut wgpu::CommandEncoder, passes: &[Pass]) {
    let mut cp = enc
        .begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("atmosphere tables"), timestamp_writes: None });
    for p in passes {
        cp.set_pipeline(&p.pipeline);
        cp.set_bind_group(0, &p.bind_group, &[]);
        cp.dispatch_workgroups(p.groups.0, p.groups.1, p.groups.2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn earth() -> planets::Atmosphere {
        planets::Atmosphere {
            top_km: 100.0,
            rayleigh_density: 1.0,
            rayleigh_scale_height_km: 8.0,
            mie_density: 1.0,
            mie_scale_height_km: 1.2,
            mie_g: 0.8,
            mie_absorption: 0.1,
            dust: 0.0,
            ozone: 1.0,
            methane: 0.0,
            clouds: None,
        }
    }

    /// Bin holding `lambda` nm.
    fn bin(lambda: f64) -> usize {
        ((lambda - LAMBDA_MIN) / BIN_WIDTH) as usize
    }

    /// Vertical optical depth from the ground to the top, by quadrature.
    fn zenith_optical_depth(kind: PlanetKind, a: &planets::Atmosphere, k: usize) -> f64 {
        let c = Coefficients::new(kind, a);
        let n = 20_000;
        let dh = a.top_km / n as f64;
        (0..n)
            .map(|i| {
                let h = (i as f64 + 0.5) * dh;
                let ozone = (1.0 - (h - OZONE_PEAK_KM).abs() / OZONE_HALF_WIDTH_KM).max(0.0);
                (c.rayleigh_extinction[k] * (-h / a.rayleigh_scale_height_km).exp()
                    + c.mie_extinction[k] * (-h / a.mie_scale_height_km).exp()
                    + c.ozone[k] * ozone)
                    * dh
            })
            .sum()
    }

    #[test]
    fn layout_matches_wgsl() {
        assert_eq!(std::mem::size_of::<AtmosphereGpu>(), 6 * 16 + 5 * 64);
    }

    /// Earth's air from first principles against Bucholtz (1995): the
    /// cross-section at 550 nm is 4.51 × 10⁻³¹ m², so β = 1.15 × 10⁻⁵ m⁻¹ at
    /// 15 °C. The design notes quote β = (5.8, 13.6, 33.1) × 10⁻⁶ m⁻¹ at
    /// 680, 550 and 440 nm (Bruneton 2008), about 15% above the measured
    /// cross-sections; their spectral shape agrees to 3%.
    #[test]
    fn rayleigh_of_earth_air() {
        let air = composition(PlanetKind::Ocean);
        let sigma: f64 = air.iter().map(|&(g, x)| x * g.rayleigh_cross_section(550.0)).sum();
        assert!((sigma / 4.513e-31 - 1.0).abs() < 0.02, "{sigma}");
        let beta = |l| rayleigh_coefficient(air, l) / 1000.0;
        assert!((beta(550.0) / 1.15e-5 - 1.0).abs() < 0.02, "{}", beta(550.0));
        for (l, reference) in [(680.0, 5.8e-6), (440.0, 33.1e-6)] {
            let shape = beta(l) / beta(550.0) / (reference / 13.6e-6);
            assert!((shape - 1.0).abs() < 0.03, "{l} nm: {shape}");
        }
        // Steeper than λ⁻⁴ in the blue: the refractive index's dispersion.
        assert!(beta(400.0) / beta(550.0) > (550.0f64 / 400.0).powi(4));
    }

    #[test]
    fn rayleigh_per_molecule() {
        let s = |g: Gas| g.rayleigh_cross_section(550.0);
        let co2 = s(Gas::Co2) / s(Gas::N2);
        let h2 = s(Gas::N2) / s(Gas::H2);
        assert!((2.3..2.8).contains(&co2), "CO2/N2 {co2}");
        assert!((3.5..5.5).contains(&h2), "N2/H2 {h2}");
    }

    /// The Rayleigh optical depth of the whole column at 550 nm is 0.097
    /// (Bucholtz 1995): Earth's 2.15 × 10²⁹ molecules/m² are 8.44 km of
    /// sea-level air.
    #[test]
    fn rayleigh_optical_depth_of_earth() {
        let a = planets::Atmosphere { mie_density: 0.0, ozone: 0.0, rayleigh_scale_height_km: 8.44, ..earth() };
        let tau = zenith_optical_depth(PlanetKind::Ocean, &a, bin(550.0));
        assert!((tau / 0.097 - 1.0).abs() < 0.03, "{tau}");
    }

    #[test]
    fn earth_air_at_zenith() {
        let a = earth();
        let t = |l| (-zenith_optical_depth(PlanetKind::Ocean, &a, bin(l))).exp();
        // Clear-sky direct-beam transmittance at 550 nm is about 0.87.
        assert!((0.84..0.91).contains(&t(550.0)), "{}", t(550.0));
        // Blue is scattered out more than red.
        assert!(t(440.0) < t(550.0) && t(550.0) < t(760.0));
        // Ozone: 300 DU with σ ≈ 5 × 10⁻²⁵ m² near 600 nm, τ ≈ 0.035.
        let c = Coefficients::new(PlanetKind::Ocean, &a);
        let o3 = c.ozone[bin(600.0)] * OZONE_HALF_WIDTH_KM;
        assert!((0.028..0.045).contains(&o3), "{o3}");
        assert!((4.0e-25..5.5e-25).contains(&ozone_cross_section(600.0)));
    }

    #[test]
    fn methane_darkens_the_red() {
        let a = planets::Atmosphere { methane: 1.0, ..earth() };
        let c = Coefficients::new(PlanetKind::IceGiant, &a);
        let abs = |l: f64| (c.rayleigh_extinction[bin(l)] - c.rayleigh_scattering[bin(l)]) * a.rayleigh_scale_height_km;
        assert!(
            abs(730.0) > 3.0 && abs(620.0) > 0.4 && abs(450.0) < 0.02,
            "{} {} {}",
            abs(730.0),
            abs(620.0),
            abs(450.0)
        );
    }

    #[test]
    fn dust_scatters_red() {
        let a = planets::Atmosphere { dust: 1.0, mie_absorption: 0.3, ..earth() };
        let c = Coefficients::new(PlanetKind::Desert, &a);
        let albedo = |l: f64| c.mie_scattering[bin(l)] / c.mie_extinction[bin(l)];
        assert!(albedo(420.0) < 0.6 && albedo(700.0) > 0.9, "{} {}", albedo(420.0), albedo(700.0));
    }
}
