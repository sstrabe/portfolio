// ---------------------------------------------------------------------------
// Spectra: 16 bins of 25 nm from 390 to 790 nm (see `spectrum.rs`), held
// as four vec4s so they stay in registers. Radiance is W m⁻² sr⁻¹ nm⁻¹.
//
// A source seen with shift g = ν_obs/ν_emit is sampled at λ_emit = g λ_obs
// and scaled by g⁵ (`spec_shift`); a blackbody at T becomes one at g T.
// ---------------------------------------------------------------------------

const SPEC_L0: f32 = 390.0;
const SPEC_DL: f32 = 25.0;

struct Spectrum {
    a: vec4<f32>,  // bins 0–3   (390–490 nm)
    b: vec4<f32>,  // bins 4–7   (490–590 nm)
    c: vec4<f32>,  // bins 8–11  (590–690 nm)
    d: vec4<f32>,  // bins 12–15 (690–790 nm)
}

fn spec(x: f32) -> Spectrum {
    return Spectrum(vec4<f32>(x), vec4<f32>(x), vec4<f32>(x), vec4<f32>(x));
}

fn spec_add(x: Spectrum, y: Spectrum) -> Spectrum {
    return Spectrum(x.a + y.a, x.b + y.b, x.c + y.c, x.d + y.d);
}

fn spec_sub(x: Spectrum, y: Spectrum) -> Spectrum {
    return Spectrum(x.a - y.a, x.b - y.b, x.c - y.c, x.d - y.d);
}

fn spec_mul(x: Spectrum, y: Spectrum) -> Spectrum {
    return Spectrum(x.a * y.a, x.b * y.b, x.c * y.c, x.d * y.d);
}

fn spec_scale(x: Spectrum, s: f32) -> Spectrum {
    return Spectrum(x.a * s, x.b * s, x.c * s, x.d * s);
}

// x * y + z
fn spec_fma(x: Spectrum, y: Spectrum, z: Spectrum) -> Spectrum {
    return Spectrum(fma(x.a, y.a, z.a), fma(x.b, y.b, z.b), fma(x.c, y.c, z.c), fma(x.d, y.d, z.d));
}

// x * s + z
fn spec_axpy(x: Spectrum, s: f32, z: Spectrum) -> Spectrum {
    return Spectrum(x.a * s + z.a, x.b * s + z.b, x.c * s + z.c, x.d * s + z.d);
}

fn spec_exp(x: Spectrum) -> Spectrum {
    return Spectrum(exp(x.a), exp(x.b), exp(x.c), exp(x.d));
}

// exp(−x): transmittance of optical depth x.
fn spec_transmit(x: Spectrum) -> Spectrum {
    return Spectrum(exp(-x.a), exp(-x.b), exp(-x.c), exp(-x.d));
}

fn spec_mix(x: Spectrum, y: Spectrum, t: f32) -> Spectrum {
    return Spectrum(mix(x.a, y.a, t), mix(x.b, y.b, t), mix(x.c, y.c, t), mix(x.d, y.d, t));
}

fn spec_max(x: Spectrum, y: Spectrum) -> Spectrum {
    return Spectrum(max(x.a, y.a), max(x.b, y.b), max(x.c, y.c), max(x.d, y.d));
}

fn spec_mean(x: Spectrum) -> f32 {
    return dot(x.a + x.b + x.c + x.d, vec4<f32>(0.0625));
}

fn spec_max_value(x: Spectrum) -> f32 {
    let m = max(max(x.a, x.b), max(x.c, x.d));
    return max(max(m.x, m.y), max(m.z, m.w));
}

// Bin k (0–15).
fn spec_get(x: Spectrum, k: u32) -> f32 {
    let i = k & 3u;
    switch (k >> 2u) {
        case 0u: { return x.a[i]; }
        case 1u: { return x.b[i]; }
        case 2u: { return x.c[i]; }
        default: { return x.d[i]; }
    }
}

// Centre wavelengths (nm).
fn spec_lambda() -> Spectrum {
    let o = vec4<f32>(0.5, 1.5, 2.5, 3.5) * SPEC_DL + SPEC_L0;
    return Spectrum(o, o + 4.0 * SPEC_DL, o + 8.0 * SPEC_DL, o + 12.0 * SPEC_DL);
}

// Spectrum from a function sampled at the bin centres: λ^p (e.g. p = −4
// for Rayleigh scattering), normalised to 1 at 550 nm.
fn spec_power_law(p: f32) -> Spectrum {
    let l = spec_lambda();
    let s = 1.0 / 550.0;
    return Spectrum(pow(l.a * s, vec4<f32>(p)), pow(l.b * s, vec4<f32>(p)), pow(l.c * s, vec4<f32>(p)), pow(l.d * s, vec4<f32>(p)));
}

// A Gaussian line of unit integral (per nm) centred at `mu` nm with width
// `sigma` nm, averaged over each bin.
fn spec_line(mu: f32, sigma: f32) -> Spectrum {
    let l = spec_lambda();
    let h = 0.5 * SPEC_DL;
    let k = 1.0 / (sqrt(2.0) * max(sigma, 0.01));
    let f = 0.5 / SPEC_DL;
    // Bin average of the Gaussian: difference of error functions.
    return Spectrum(
        f * (erf4((l.a + h - mu) * k) - erf4((l.a - h - mu) * k)),
        f * (erf4((l.b + h - mu) * k) - erf4((l.b - h - mu) * k)),
        f * (erf4((l.c + h - mu) * k) - erf4((l.c - h - mu) * k)),
        f * (erf4((l.d + h - mu) * k) - erf4((l.d - h - mu) * k)),
    );
}

// Abramowitz–Stegun 7.1.26, |error| < 1.5e-7.
fn erf4(x: vec4<f32>) -> vec4<f32> {
    let s = sign(x);
    let a = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-a * a);
    return s * y;
}

fn planck4(l: vec4<f32>, t: f32) -> vec4<f32> {
    let x = min(1.438777e7 / (l * t), vec4<f32>(80.0));
    // exp(x) − 1 without cancellation for hot sources.
    let em1 = select(exp(x) - 1.0, x * (1.0 + 0.5 * x), x < vec4<f32>(1e-3));
    let l2 = l * l;
    return 1.191042972e20 / (l2 * l2 * l * em1);
}

// Blackbody spectral radiance at temperature t (K).
fn spec_planck(t: f32) -> Spectrum {
    let tt = clamp(t, 300.0, 1.0e8);
    let l = spec_lambda();
    return Spectrum(planck4(l.a, tt), planck4(l.b, tt), planck4(l.c, tt), planck4(l.d, tt));
}

// Spectral shape of a blackbody at t, normalised to unit bolometric
// radiance (so intensity × this conserves energy): π B_λ / (σ T⁴).
fn spec_planck_unit(t: f32) -> Spectrum {
    let tt = clamp(t, 300.0, 1.0e8);
    let t2 = tt * tt;
    return spec_scale(spec_planck(tt), PI / (5.670374e-8 * t2 * t2));
}

// Value of a binned spectrum at wavelength `l` nm (linear between bin
// centres, held flat past the ends).
fn spec_at(x: Spectrum, l: f32) -> f32 {
    let u = clamp((l - SPEC_L0) / SPEC_DL - 0.5, 0.0, 15.0);
    let i = u32(floor(u));
    let j = min(i + 1u, 15u);
    return mix(spec_get(x, i), spec_get(x, j), u - f32(i));
}

// Resample a rest-frame spectrum for an observer who sees it with shift g:
// each observed bin reads the rest-frame spectrum at g λ. Radiance also
// needs the g⁵ factor (`spec_shift`); transmittance and reflectance don't.
fn spec_resample(x: Spectrum, g: f32) -> Spectrum {
    if (abs(g - 1.0) < 1e-4) {
        return x;
    }
    let l = spec_lambda();
    var o: Spectrum;
    for (var i = 0u; i < 4u; i++) {
        o.a[i] = spec_at(x, g * l.a[i]);
        o.b[i] = spec_at(x, g * l.b[i]);
        o.c[i] = spec_at(x, g * l.c[i]);
        o.d[i] = spec_at(x, g * l.d[i]);
    }
    return o;
}

// Rest-frame radiance → observed radiance for shift g.
fn spec_shift(x: Spectrum, g: f32) -> Spectrum {
    let g2 = g * g;
    return spec_scale(spec_resample(x, g), g2 * g2 * g);
}
