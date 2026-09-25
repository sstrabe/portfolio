// ---------------------------------------------------------------------------
// Atmosphere model shared by the lookup-table passes (`atmo_luts.wgsl`) and
// the trace pass (`atmosphere.wgsl`, `clouds.wgsl`): parameters, density
// profiles, phase functions and the lookup-table parameterisations. No
// bindings here (the two modules bind the tables differently).
//
// Lengths are km and coefficients km⁻¹. The atmosphere is three components:
// - the bulk gas (Rayleigh scattering, plus methane absorption as methane is
//   mixed like the gas), exponential with scale height H_R;
// - aerosols (Mie), exponential with scale height H_M;
// - ozone, a tent profile around its peak altitude (absorption only).
// ---------------------------------------------------------------------------

// Mirrors `AtmosphereGpu` in `atmosphere.rs`.
struct AtmosphereParams {
    shape: vec4<f32>,     // planet radius (km), top altitude (km), Rayleigh scale height (km), Mie scale height (km)
    density: vec4<f32>,   // Rayleigh, Mie, ozone, methane (relative to Earth)
    aerosol: vec4<f32>,   // Mie g, Mie absorption fraction, dust colour, present (0/1)
    clouds: vec4<f32>,    // coverage, base (km), thickness (km), vertical optical depth
    extra: vec4<f32>,     // albedo seen from above (ground and clouds), ozone peak (km), ozone half width (km), unused
    ids: vec4<u32>,       // seed, kind, cloud map (ATMO_NO_CLOUDS: none), unused
    rayleigh_scattering: Spectrum,  // at the reference level
    rayleigh_extinction: Spectrum,  // scattering plus methane absorption
    mie_scattering: Spectrum,
    mie_extinction: Spectrum,
    ozone: Spectrum,                // absorption at the ozone peak
}

// The non-spectral part of the parameters: what geometry, density
// profiles and phase functions need (small enough to keep in registers).
struct AtmoGeom {
    shape: vec4<f32>,
    aerosol: vec4<f32>,
    clouds: vec4<f32>,
    extra: vec4<f32>,
    ids: vec4<u32>,
}

fn atmo_geom_of(a: AtmosphereParams) -> AtmoGeom {
    return AtmoGeom(a.shape, a.aerosol, a.clouds, a.extra, a.ids);
}

const ATMO_NO_CLOUDS: u32 = 0xffffffffu;
const ATMO_TRANS_W: f32 = 256.0;
const ATMO_TRANS_H: f32 = 64.0;
const ATMO_MS_SIZE: f32 = 32.0;
const ATMO_IRR_SIZE: f32 = 32.0;

fn atmo_present(a: AtmoGeom) -> bool {
    return a.aerosol.w > 0.5;
}

fn atmo_top(a: AtmoGeom) -> f32 {
    return a.shape.x + a.shape.y;
}

// Relative densities (Rayleigh gas, aerosols, ozone) at altitude h.
fn atmo_density(a: AtmoGeom, h: f32) -> vec3<f32> {
    let hh = max(h, 0.0);
    let ozone = max(1.0 - abs(hh - a.extra.y) / a.extra.z, 0.0);
    return vec3<f32>(exp(-hh / a.shape.z), exp(-hh / a.shape.w), ozone);
}

// Extinction for relative densities d (lookup-table passes; the trace pass
// works four bins at a time, see `atmosphere.wgsl`).
fn atmo_extinction(a: AtmosphereParams, d: vec3<f32>) -> Spectrum {
    return spec_axpy(a.ozone, d.z, spec_axpy(a.mie_extinction, d.y, spec_scale(a.rayleigh_extinction, d.x)));
}

// Transmittance for density-weighted columns (km) of the three components.
fn atmo_transmit_columns(a: AtmosphereParams, c: vec3<f32>) -> Spectrum {
    return spec_transmit(atmo_extinction(a, c));
}

// Rayleigh phase function (per steradian).
fn atmo_phase_rayleigh(c: f32) -> f32 {
    return 3.0 / (16.0 * PI) * (1.0 + c * c);
}

fn atmo_phase_hg(g: f32, c: f32) -> f32 {
    let d = max(1.0 + g * g - 2.0 * g * c, 1e-4);
    return (1.0 - g * g) / (4.0 * PI * d * sqrt(d));
}

// Aerosols: Cornette–Shanks (a Henyey–Greenstein lobe corrected towards
// Mie's polarisation term); dust adds a weak backscattering lobe, as for
// the irregular grains of Martian dust.
fn atmo_phase_mie(a: AtmoGeom, c: f32) -> f32 {
    let g = a.aerosol.x;
    let g2 = g * g;
    let d = max(1.0 + g2 - 2.0 * g * c, 1e-4);
    let cs = 3.0 / (8.0 * PI) * (1.0 - g2) * (1.0 + c * c) / ((2.0 + g2) * d * sqrt(d));
    return mix(cs, atmo_phase_hg(-0.3, c), 0.12 * a.aerosol.z);
}

// Distance from radius r along direction cosine mu to the sphere of radius
// rs (outwards: the far intersection), without cancellation in r² − rs².
fn atmo_dist_to_sphere(r: f32, mu: f32, rs: f32) -> f32 {
    let disc = (rs - r) * (rs + r) + r * r * mu * mu;
    return max(-r * mu + sqrt(max(disc, 0.0)), 0.0);
}

// Does the ray from radius r with direction cosine mu hit the planet?
fn atmo_hits_ground(a: AtmoGeom, r: f32, mu: f32) -> bool {
    let rg = a.shape.x;
    return mu < 0.0 && (rg - r) * (rg + r) + r * r * mu * mu >= 0.0;
}

// Fraction of a sun disc of angular radius `ang` above the planet's horizon
// seen from radius r, the sun at direction cosine mu_s from the zenith.
fn atmo_sun_above_horizon(a: AtmoGeom, r: f32, mu_s: f32, ang: f32) -> f32 {
    let s = min(a.shape.x / r, 1.0);
    // Elevation of the sun above the (dipped) horizon.
    let elevation = asin(clamp(mu_s, -1.0, 1.0)) + acos(s);
    let w = max(ang, 1e-3);
    return smoothstep(-w, w, elevation);
}

// Transmittance table: Bruneton's (2017) parameterisation of rays that
// reach the top of the atmosphere, from radius r at direction cosine mu.
fn atmo_texcoord(x: f32, n: f32) -> f32 {
    return 0.5 / n + x * (1.0 - 1.0 / n);
}

fn atmo_texcoord_inv(u: f32, n: f32) -> f32 {
    return (u - 0.5 / n) / (1.0 - 1.0 / n);
}

fn atmo_trans_uv(a: AtmoGeom, r: f32, mu: f32) -> vec2<f32> {
    let rg = a.shape.x;
    let rt = atmo_top(a);
    let hh = sqrt((rt - rg) * (rt + rg));
    let rr = clamp(r, rg, rt);
    let rho = sqrt(max((rr - rg) * (rr + rg), 0.0));
    let d = atmo_dist_to_sphere(rr, mu, rt);
    let d_min = rt - rr;
    let d_max = rho + hh;
    let x_mu = clamp((d - d_min) / max(d_max - d_min, 1e-6), 0.0, 1.0);
    return vec2<f32>(atmo_texcoord(x_mu, ATMO_TRANS_W), atmo_texcoord(rho / hh, ATMO_TRANS_H));
}

// Inverse of `atmo_trans_uv`: (r, mu).
fn atmo_trans_r_mu(a: AtmoGeom, uv: vec2<f32>) -> vec2<f32> {
    let rg = a.shape.x;
    let rt = atmo_top(a);
    let hh = sqrt((rt - rg) * (rt + rg));
    let x_mu = atmo_texcoord_inv(uv.x, ATMO_TRANS_W);
    let rho = hh * atmo_texcoord_inv(uv.y, ATMO_TRANS_H);
    let r = sqrt(rho * rho + rg * rg);
    let d_min = rt - r;
    let d_max = rho + hh;
    let d = d_min + x_mu * (d_max - d_min);
    var mu = 1.0;
    if (d > 1e-4) {
        mu = clamp((hh * hh - rho * rho - d * d) / (2.0 * r * d), -1.0, 1.0);
    }
    return vec2<f32>(r, mu);
}

// Multiple scattering and sky irradiance tables: sun zenith cosine along
// x, altitude along y (linear, texel centres at the ends of the range).
fn atmo_ms_uv(a: AtmoGeom, r: f32, mu_s: f32, n: f32) -> vec2<f32> {
    let h = clamp((r - a.shape.x) / a.shape.y, 0.0, 1.0);
    return vec2<f32>(atmo_texcoord(mu_s * 0.5 + 0.5, n), atmo_texcoord(h, n));
}

fn atmo_ms_r_mu(a: AtmoGeom, uv: vec2<f32>, n: f32) -> vec2<f32> {
    let mu_s = atmo_texcoord_inv(uv.x, n) * 2.0 - 1.0;
    let h = atmo_texcoord_inv(uv.y, n);
    return vec2<f32>(a.shape.x + h * a.shape.y, mu_s);
}
