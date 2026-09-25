// ---------------------------------------------------------------------------
// The wide part of the point-spread function on the convolution grid: the
// energy each grid pixel receives from a unit source at the origin, in CIE
// X, Y, Z (wrapped so the source sits at texel 0). Port of `optics.rs`.
//
// Diffraction: at wavelength λ the pupil (diameter D) spreads light as
// β² |Â(2π β r)|² / A per square pixel, β = D δ / λ for pixel angle δ.
// It is averaged over each pixel with 4 × 4 jittered samples at each of the
// 16 spectral bins, then weighted by the bins' colour-matching functions.
// Because the pattern scales with λ, red spikes and rings reach further
// than blue ones: the colour fringes.
//
// Scatter: the camera and telescope add a Harvey halo (dust, micro-
// scratches; bluer since scattering grows toward short wavelengths). The
// eye has the CIE disability-glare function instead (scatter in the cornea
// and lens, a real physiological measurement), streaked by the ciliary
// corona and ringed by the lenticular halo (diffraction by the lens-fibre
// lattice, ~9 µm spacing: a coloured ring near 3.5°, red outside).
//
// The core (r < ~2.5 grid px) is left out (`w_core`): it is drawn sharp at
// full resolution. The kernel is tapered before it could wrap around.
// ---------------------------------------------------------------------------

struct PsfParams {
    xyz: array<vec4<f32>, 12>,  // bin weights: rows X0–X3, Y0–Y3, Z0–Z3
    pupil: vec4<f32>,           // D δ (m·rad), obstruction, vane width, blade rotation
    dims: vec4<u32>,            // nx, ny, optics mode, blades
    scatter: vec4<f32>,         // halo fraction, θ₀ (rad), δ (rad), taper radius (grid px)
    core: vec4<f32>,            // core window r₀, r₁ (grid px), halo normalisation, unused
}

@group(0) @binding(3) var<uniform> psf: PsfParams;
@group(0) @binding(4) var<storage, read_write> kimg: array<vec4<f32>>;

fn sinc(x: f32) -> f32 {
    if (abs(x) < 1e-3) {
        return 1.0 - x * x / 6.0;
    }
    return sin(x) / x;
}

// J₁ (rational approximations, Numerical Recipes).
fn bessel_j1(x: f32) -> f32 {
    let ax = abs(x);
    if (ax < 8.0) {
        let y = x * x;
        let a = x * (72362614232.0 + y * (-7895059235.0 + y * (242396853.1 + y * (-2972611.439 + y * (15704.48260 + y * -30.16036606)))));
        let b = 144725228442.0 + y * (2300535178.0 + y * (18583304.74 + y * (99447.43394 + y * (376.9991397 + y))));
        return a / b;
    }
    let z = 8.0 / ax;
    let y = z * z;
    let xx = ax - 2.356194491;
    let p = 1.0 + y * (0.183105e-2 + y * (-0.3516396496e-4 + y * (0.2457520174e-5 + y * -0.240337019e-6)));
    let q = 0.04687499995 + y * (-0.2002690873e-3 + y * (0.8449199096e-5 + y * (-0.88228987e-6 + y * 0.105787412e-6)));
    let v = sqrt(0.636619772 / ax) * (cos(xx) * p - z * sin(xx) * q);
    return select(v, -v, x < 0.0);
}

fn disc_ft(q: vec2<f32>, a: f32) -> f32 {
    let k = length(q) * a;
    if (k < 1e-4) {
        return PI * a * a;
    }
    return TAU * a * a * bessel_j1(k) / k;
}

fn pupil_area() -> f32 {
    switch psf.dims.z {
        case 0u: { return PI / 4.0; }
        case 1u: {
            let n = f32(psf.dims.w);
            return 0.125 * n * sin(TAU / n);
        }
        default: {
            let e = psf.pupil.y;
            return PI / 4.0 * (1.0 - e * e) - 2.0 * psf.pupil.z * (1.0 - e);
        }
    }
}

// Â(q) for the pupil in units of its diameter (see `Pupil::ft`).
fn pupil_ft(q: vec2<f32>) -> vec2<f32> {
    switch psf.dims.z {
        case 0u: {
            return vec2<f32>(disc_ft(q, 0.5), 0.0);
        }
        case 1u: {
            let q2 = dot(q, q);
            if (q2 < 1e-6) {
                return vec2<f32>(pupil_area(), 0.0);
            }
            let n = psf.dims.w;
            let step = TAU / f32(n);
            var sum = vec2<f32>(0.0);
            var va = 0.5 * vec2<f32>(cos(psf.pupil.w), sin(psf.pupil.w));
            for (var m = 1u; m <= n; m++) {
                let ang = psf.pupil.w + step * f32(m);
                let vb = 0.5 * vec2<f32>(cos(ang), sin(ang));
                let e = vb - va;
                let l = length(e);
                let t = e / l;
                let nrm = vec2<f32>(t.y, -t.x);
                let s = dot(q, nrm) * l * sinc(dot(q, t) * l * 0.5);
                let ph = dot(q, 0.5 * (va + vb));
                sum += s * vec2<f32>(sin(ph), cos(ph));
                va = vb;
            }
            return sum / q2;
        }
        default: {
            let e = psf.pupil.y;
            let w = psf.pupil.z;
            var f = vec2<f32>(disc_ft(q, 0.5) - disc_ft(q, 0.5 * e), 0.0);
            let l = 0.5 * (1.0 - e);
            let c = 0.25 * (1.0 + e);
            // Four vanes along ±x, ±y: rectangles l × w centred at c.
            let mx = l * w * sinc(q.x * l * 0.5) * sinc(q.y * w * 0.5);
            let my = l * w * sinc(q.y * l * 0.5) * sinc(q.x * w * 0.5);
            // e^{−iqc} + e^{+iqc} = 2 cos(qc): the opposite vanes pair up.
            f.x -= 2.0 * (mx * cos(q.x * c) + my * cos(q.y * c));
            return f;
        }
    }
}

fn bin_weight(row: u32, k: u32) -> f32 {
    return psf.xyz[row * 4u + k / 4u][k % 4u];
}

fn hash2(p: vec3<u32>) -> vec2<f32> {
    return hash3(p).xy;
}

// CIE glare of the eye, per steradian, at θ in degrees (age 25, average
// pigmentation), split into the lens scatter (the 10/θ³ term, which the
// ciliary corona streaks) and the rest.
fn eye_glare(theta_deg: f32) -> vec2<f32> {
    let t = max(theta_deg, 0.1);
    let age = 1.0 + pow(25.0 / 62.5, 4.0);
    return vec2<f32>(10.0 / (t * t * t), age * (5.0 / (t * t) + 0.05 / t) + 0.00125);
}

// Radial streaks of the ciliary corona: fine, random in azimuth.
fn corona(phi: f32) -> f32 {
    let u = (phi / TAU + 0.5) * 900.0;
    let i = floor(u);
    let f = u - i;
    let a = hash3(vec3<u32>(u32(i) % 900u, 7u, 3u)).x;
    let b = hash3(vec3<u32>((u32(i) + 1u) % 900u, 7u, 3u)).x;
    let n = mix(a, b, f * f * (3.0 - 2.0 * f));
    return 5.0 * n * n * n * n;
}

@compute @workgroup_size(8, 8)
fn cs_psf(@builtin(global_invocation_id) gid: vec3<u32>) {
    let nx = psf.dims.x;
    let ny = psf.dims.y;
    if (gid.x >= nx || gid.y >= ny) {
        return;
    }
    let dx = select(i32(gid.x), i32(gid.x) - i32(nx), gid.x >= nx / 2u);
    let dy = select(i32(gid.y), i32(gid.y) - i32(ny), gid.y >= ny / 2u);
    let r = vec2<f32>(f32(dx), f32(dy));
    let rl = length(r);
    let w_core = smoothstep(psf.core.x, psf.core.y, rl);
    let w_edge = 1.0 - smoothstep(0.75, 1.0, rl / psf.scatter.w);
    let w = w_core * w_edge;
    let idx = gid.y * nx + gid.x;
    if (w <= 0.0) {
        kimg[idx] = vec4<f32>(0.0);
        return;
    }
    let delta = psf.scatter.z;
    let theta = rl * delta;
    let area = pupil_area();
    let mode = psf.dims.z;
    // Scatter halo (camera, telescope), per square pixel at 550 nm.
    let halo = psf.scatter.x * pow(1.0 + pow(theta / psf.scatter.y, 2.0), -1.25) * delta * delta / psf.core.z;
    let glare = eye_glare(degrees(theta)) * delta * delta;
    let phi = atan2(r.y, r.x);

    var acc = vec3<f32>(0.0);
    var norm = vec3<f32>(0.0);
    for (var k = 0u; k < 16u; k++) {
        let lambda = SPEC_L0 + SPEC_DL * (f32(k) + 0.5);
        let beta = psf.pupil.x / (lambda * 1e-9);
        var e = 0.0;
        if (mode == 0u && beta > 2.0) {
            // The eye's Airy rings are far below a pixel: their pixel
            // average, 1 / (π³ β r³).
            e = 1.0 / (PI * PI * PI * beta * rl * rl * rl);
        } else {
            for (var s = 0u; s < 16u; s++) {
                let j = hash2(vec3<u32>(gid.x * 16u + s, gid.y, k * 977u + 11u));
                let o = (vec2<f32>(f32(s % 4u), f32(s / 4u)) + j) * 0.25 - 0.5;
                let f = pupil_ft(TAU * beta * (r + o));
                e += dot(f, f);
            }
            e *= beta * beta / (16.0 * area);
        }
        let l550 = 550.0 / lambda;
        if (mode == 0u) {
            // Corona streaks fade outward; longer wavelengths reach further.
            let reach = exp(-degrees(theta) * l550 / 2.0);
            let g = glare.x * (0.35 + 0.65 * corona(phi) * reach) + glare.y;
            // Lenticular halo: ring at λ / 9.4 µm, 0.3% of the light.
            let th = lambda * 1e-9 / 9.4e-6;
            let sw = 0.12 * th;
            let ring = 0.003 / (TAU * th * sqrt(TAU) * sw) * exp(-0.5 * pow((theta - th) / sw, 2.0)) * delta * delta;
            e = e + g + ring;
        } else {
            let fs = psf.scatter.x * l550 * l550;
            e = e * (1.0 - fs) + halo * l550 * l550;
        }
        let wx = vec3<f32>(bin_weight(0u, k), bin_weight(1u, k), bin_weight(2u, k));
        acc += wx * e;
        norm += wx;
    }
    kimg[idx] = vec4<f32>(acc / norm * w, 0.0);
}
