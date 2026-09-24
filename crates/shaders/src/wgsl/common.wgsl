// ---------------------------------------------------------------------------
// Shared prelude: frame uniforms and Kerr–Schild geometry.
//
// Units: G = c = M = 1. Coordinates (t, x, y, z), spin along +z. All times
// on the GPU are relative to the observer's current coordinate time (the
// observer sits at t = 0), which keeps f32 precision independent of how
// long the session has been running.
//
// The Kerr–Schild formulas mirror `crates/kerr/src/metric.rs`, where they
// are tested against automatic differentiation.
// ---------------------------------------------------------------------------

struct Frame {
    obs: vec4<f32>,      // observer event (0, x, y, z)
    e0: vec4<f32>,       // tetrad: 4-velocity
    e1: vec4<f32>,       //         forward
    e2: vec4<f32>,       //         left
    e3: vec4<f32>,       //         up
    cam: vec4<f32>,      // tan(fov/2), aspect, pixel angle (rad), exposure
    view: vec4<f32>,     // width, height (render target px), wall time (s), star gain
    kerr: vec4<f32>,     // M, a, r+, escape radius
    march: vec4<f32>,    // step fraction, min step, max step, horizon epsilon
    hist: vec4<f32>,     // newest sample time, sample spacing, oldest sample time, lookback limit
    counts: vec4<u32>,   // bodies, ring capacity, newest slot, filled samples
    counts2: vec4<u32>,  // max ray steps, panel count, newton iterations, station count
    screen: vec4<f32>,   // output width, height (px), sRGB-encode flag, unused
    extra: vec4<f32>,    // image-ray escape radius, faintest visible flux, star discs, sky adaptation
}

// Per-body metadata; times relative to the observer.
struct BodyMeta {
    a: vec4<f32>,  // valid-from time, death time, temperature (K), luminosity
    b: vec4<f32>,  // kind (0 station, 1 compact, 2 star), beacon strength, generation, reserved
}

// One (body, image order) slot of the image finder.
struct ImageState {
    dir: vec4<f32>,   // ship-frame direction (forward, left, up), w: 1 if valid
    info: vec4<f32>,  // redshift g, flux g⁴/area, emission time, residual
    warm: vec4<f32>,  // coordinate-space direction for the next warm start, w: generation
}

@group(0) @binding(0) var<uniform> frame: Frame;

const PI: f32 = 3.14159265358979;
const TAU: f32 = 6.28318530717959;

struct Ks {
    r: f32,
    f: f32,
    l: vec3<f32>,
    df: vec3<f32>,
    // Gradients of the three spatial components of l_μ.
    glx: vec3<f32>,
    gly: vec3<f32>,
    glz: vec3<f32>,
}

fn ks_radius(p: vec3<f32>) -> f32 {
    let a2 = frame.kerr.y * frame.kerr.y;
    let rho = dot(p, p) - a2;
    let z2 = p.z * p.z;
    let disc = sqrt(rho * rho + 4.0 * a2 * z2);
    var r2: f32;
    if (rho >= 0.0) {
        r2 = 0.5 * (rho + disc);
    } else {
        r2 = 2.0 * a2 * z2 / max(disc - rho, 1e-20);
    }
    return sqrt(max(r2, 1e-12));
}

fn ks_terms(p: vec3<f32>) -> Ks {
    let a = frame.kerr.y;
    let r = ks_radius(p);
    let r2 = r * r;
    let q = r2 + a * a;
    let sigma = r2 * r2 + a * a * p.z * p.z;
    var k: Ks;
    k.r = r;
    k.f = 2.0 * frame.kerr.x * r2 * r / sigma;
    k.l = vec3<f32>((r * p.x + a * p.y) / q, (r * p.y - a * p.x) / q, p.z / r);
    return k;
}

fn ks_grad(p: vec3<f32>) -> Ks {
    let a = frame.kerr.y;
    let a2 = a * a;
    let r = ks_radius(p);
    let r2 = r * r;
    let d = r2 + a2 * p.z * p.z / r2;
    let dr = vec3<f32>(p.x * r / d, p.y * r / d, p.z * (r2 + a2) / (r * d));
    let sigma = r2 * r2 + a2 * p.z * p.z;
    let f = 2.0 * frame.kerr.x * r2 * r / sigma;
    let r3 = r2 * r;
    var k: Ks;
    k.r = r;
    k.f = f;
    k.df = f * (3.0 * dr / r - 4.0 * r3 * dr / sigma) - vec3<f32>(0.0, 0.0, f * 2.0 * a2 * p.z / sigma);
    let q = r2 + a2;
    k.l = vec3<f32>((r * p.x + a * p.y) / q, (r * p.y - a * p.x) / q, p.z / r);
    let two_r_over_q = 2.0 * r / q;
    // ∂_i l_x = (∂_i r x + r δ_ix + a δ_iy)/q − l_x 2 r ∂_i r / q, etc.
    k.glx = (dr * p.x + vec3<f32>(r, a, 0.0)) / q - k.l.x * two_r_over_q * dr;
    k.gly = (dr * p.y + vec3<f32>(-a, r, 0.0)) / q - k.l.y * two_r_over_q * dr;
    k.glz = vec3<f32>(0.0, 0.0, 1.0 / r) - p.z * dr / r2;
    return k;
}

// g_{μν} v^ν
fn lower(p: vec3<f32>, v: vec4<f32>) -> vec4<f32> {
    let k = ks_terms(p);
    let fl = k.f * (v.x + dot(k.l, v.yzw));
    return vec4<f32>(-v.x + fl, v.yzw + fl * k.l);
}

// g_{μν} u^μ v^ν
fn mdot(p: vec3<f32>, u: vec4<f32>, v: vec4<f32>) -> f32 {
    let k = ks_terms(p);
    let lu = u.x + dot(k.l, u.yzw);
    let lv = v.x + dot(k.l, v.yzw);
    return -u.x * v.x + dot(u.yzw, v.yzw) + k.f * lu * lv;
}

// Future-pointing unit 4-velocity for coordinate velocity dx/dt.
fn four_velocity(p: vec3<f32>, vel: vec3<f32>) -> vec4<f32> {
    let w = vec4<f32>(1.0, vel);
    let n = -mdot(p, w, w);
    return w / sqrt(max(n, 1e-8));
}

// ---------------------------------------------------------------------------
// Null geodesics, Hamiltonian form H = ½ g^{μν} p_μ p_ν (see geodesic.rs).
//
// Ray positions are kept relative to the observer (who sits at the spatial
// origin), so nearby objects stay precise in f32 even far from the hole;
// the metric is evaluated at `frame.obs.yzw + x`.
// ---------------------------------------------------------------------------

// Position relative to the hole of a ray event.
fn abs_pos(x: vec4<f32>) -> vec3<f32> {
    return frame.obs.yzw + x.yzw;
}

struct Phase {
    x: vec4<f32>,  // event
    p: vec4<f32>,  // covariant momentum (p_t is conserved)
}

fn phase_rhs(s: Phase) -> Phase {
    let k = ks_grad(abs_pos(s.x));
    let pp = s.p.yzw;
    let ll = -s.p.x + dot(k.l, pp);
    let fl = k.f * ll;
    var o: Phase;
    o.x = vec4<f32>(-s.p.x + fl, pp - fl * k.l);
    let dl_dot_p = pp.x * k.glx + pp.y * k.gly + pp.z * k.glz;
    o.p = vec4<f32>(0.0, 0.5 * k.df * ll * ll + fl * dl_dot_p);
    return o;
}

fn phase_axpy(a: Phase, h: f32, b: Phase) -> Phase {
    return Phase(a.x + h * b.x, a.p + h * b.p);
}

fn rk4(s: Phase, k1: Phase, h: f32) -> Phase {
    let k2 = phase_rhs(phase_axpy(s, 0.5 * h, k1));
    let k3 = phase_rhs(phase_axpy(s, 0.5 * h, k2));
    let k4 = phase_rhs(phase_axpy(s, h, k3));
    return Phase(
        s.x + h / 6.0 * (k1.x + 2.0 * k2.x + 2.0 * k3.x + k4.x),
        s.p + h / 6.0 * (k1.p + 2.0 * k2.p + 2.0 * k3.p + k4.p),
    );
}

// Past-directed photon for ship-frame direction n = (forward, left, up):
// P = −u + n^a e_a, normalized so the observed frequency is 1.
fn backward_ray(n: vec3<f32>) -> Phase {
    let pu = -frame.e0 + n.x * frame.e1 + n.y * frame.e2 + n.z * frame.e3;
    return Phase(vec4<f32>(0.0), lower(frame.obs.yzw, pu));
}

// Step length in the affine parameter for a spatial advance ∝ r.
fn ray_step(s: Phase, k1: Phase, r: f32) -> f32 {
    let rp = frame.kerr.z;
    let dist = clamp(frame.march.x * max(r - 0.8 * rp, 0.2 * rp), frame.march.y, frame.march.z);
    return dist / max(length(k1.x.yzw), 1e-6);
}

// Ship-frame direction for a screen position in normalized device coords.
fn ndc_to_dir(ndc: vec2<f32>) -> vec3<f32> {
    let t = frame.cam.x;
    return normalize(vec3<f32>(1.0, -ndc.x * t * frame.cam.y, ndc.y * t));
}

// Screen position of a ship-frame direction (w < 0 when behind).
fn dir_to_ndc(n: vec3<f32>) -> vec3<f32> {
    let t = frame.cam.x;
    let fwd = max(n.x, 1e-6);
    return vec3<f32>(-n.y / (fwd * t * frame.cam.y), n.z / (fwd * t), n.x);
}

// ---------------------------------------------------------------------------
// Perceived brightness of a point source: `b` is its flux in units of the
// faintest visible flux. Shared by star sprites and resolved star discs.
fn star_response(b: f32) -> f32 {
    return 0.06 * pow(max(b, 0.0), 0.6);
}

// ---------------------------------------------------------------------------
// Colour: blackbody chromaticity from Planck's law at three wavelengths,
// white-balanced so a 6500 K source is neutral. Doppler/gravitational shifts
// enter as T → g T; bolometric intensity scales as g⁴.
// ---------------------------------------------------------------------------

fn planck(lambda_um: f32, t: f32) -> f32 {
    let x = min(14387.77 / (lambda_um * t), 80.0);
    return 1.0 / (pow(lambda_um, 5.0) * (exp(x) - 1.0));
}

fn blackbody(t: f32) -> vec3<f32> {
    let tt = clamp(t, 400.0, 1.0e6);
    let c = vec3<f32>(planck(0.61, tt), planck(0.55, tt), planck(0.465, tt));
    let wb = vec3<f32>(0.32217, 0.36213, 0.39740);
    let n = c / wb;
    return n / max(n.r, max(n.g, n.b));
}

fn hash3(p: vec3<u32>) -> vec3<f32> {
    var v = p * vec3<u32>(1664525u, 1013904223u, 2654435761u) + vec3<u32>(1013904223u, 1664525u, 374761393u);
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v ^= v >> vec3<u32>(16u);
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    return vec3<f32>(v & vec3<u32>(0xffffffu)) / f32(0x1000000);
}
