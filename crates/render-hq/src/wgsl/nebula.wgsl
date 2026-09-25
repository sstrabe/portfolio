// ---------------------------------------------------------------------------
// Nebulae and dust around the Galactic Centre (hooks called by the far-field
// trace in `far.wgsl`; the volumes are generated at startup by
// `nebula_gen.wgsl`, and `nebula/scene.rs` explains where everything is and
// how bright).
//
// Positions are relative to the hole in units of M (Kerr–Schild
// coordinates, which are Cartesian to ~10⁻⁶ at parsecs, so rays are straight
// here). `g` is the photon's frequency relative to a static observer at
// infinity: gas at rest emitting a line at λ₀ is seen at λ₀/g.
//
// Everything is integrated in the static frame first, per ray: emission
// lines as scalar intensities (each already dimmed by the dust in front at
// its own wavelength) with the intensity-weighted mean and spread of the
// line-of-sight velocity, a continuum (synchrotron, dust-scattered
// starlight, the pulsar) sampled at four anchor wavelengths, and the dust
// optical depth τ_V. `neb_assemble` then builds the observed spectrum: each
// line is a Gaussian at λ₀/(g D) (D the gas's Doppler factor) scaled by g⁴,
// the continuum is read at g λ and scaled by g⁵, and the dust's
// transmission is evaluated at g λ. Lines are much narrower than a 25 nm
// bin, so dimming each by the dust at its centre is exact enough.
// ---------------------------------------------------------------------------

// One volume: a box of voxels at a fixed place.
struct NebVolume {
    centre: vec4<f32>,    // box centre (M, relative to the hole); w: voxel size (M)
    ax: vec4<f32>,        // box axis / half extent (M), so dot(p − centre, ax) ∈ [−1, 1]; w: voxels along it
    ay: vec4<f32>,
    az: vec4<f32>,
    emit: vec4<f32>,      // Hα (W m⁻² sr⁻¹ per M per unit r), τ_V per M per unit g, τ_V per M per unit √r,
                          // synchrotron I_λ at 550 nm (W m⁻² sr⁻¹ nm⁻¹ per M per unit g)
    ratios: vec4<f32>,    // [N II] 6548+6583 / Hα, [S II] 6716+6731 / Hα, [O I] 6300+6363 / Hα, [O III] 4959+5007 / Hβ
    ratios_b: vec4<f32>,  // the same, added per unit of channel b
    extra: vec4<f32>,     // [O III]/Hβ added per unit of channel a, scattering (1 when a is the cluster's light),
                          // occupancy thresholds for r and g
    synch: vec4<f32>,     // synchrotron spectrum at the anchors, relative to 550 nm
    detail: vec4<f32>,    // sub-voxel detail amplitude, turbulent velocity² (c²), velocity unit (c per texel unit), unused
}

struct NebParams {
    vols: array<NebVolume, 4>,
    sca: vec4<f32>,       // cluster light at the anchors for dust scattering: ω κ_λ/κ_V L_λ / (4π) / (m per M)²
    sca2: vec4<f32>,      // Henyey–Greenstein asymmetry, source core radius² (M²), empty radius around the hole (M),
                          // radius enclosing everything (M)
    pulsar: vec4<f32>,    // position (M), 1 when nebulae are drawn
    pulsar_l: vec4<f32>,  // pulsar L_λ / 4π at the anchors (W nm⁻¹ sr⁻¹ per (m per M)²)
}

@group(3) @binding(0) var<uniform> neb: NebParams;
@group(3) @binding(1) var neb_clamp: sampler;
@group(3) @binding(2) var neb_repeat: sampler;
@group(3) @binding(3) var neb_detail: texture_3d<f32>;
@group(3) @binding(4) var neb_vol0: texture_3d<f32>;
@group(3) @binding(5) var neb_vol1: texture_3d<f32>;
@group(3) @binding(6) var neb_vol2: texture_3d<f32>;
@group(3) @binding(7) var neb_vol3: texture_3d<f32>;
@group(3) @binding(8) var neb_vel0: texture_3d<f32>;
@group(3) @binding(9) var neb_vel1: texture_3d<f32>;
@group(3) @binding(10) var neb_vel2: texture_3d<f32>;
@group(3) @binding(11) var neb_vel3: texture_3d<f32>;
// The march from inside the cluster, cached per direction (`nebula_cube.wgsl`):
// A Hβ, [O III], [O I], Hα; B [N II], [S II], τ_V, Σw; C continuum; D Σwv, Σw(v² + σ²).
// Faces are layers of 2D arrays, addressed by `neb_cube_face` (the inverse
// of the generator's `neb_cube_dir`), so lookup and generation agree.
@group(3) @binding(12) var neb_cube_a: texture_2d_array<f32>;
@group(3) @binding(13) var neb_cube_b: texture_2d_array<f32>;
@group(3) @binding(14) var neb_cube_c: texture_2d_array<f32>;
@group(3) @binding(15) var neb_cube_d: texture_2d_array<f32>;
// Cube centre (M, relative to the hole); w: 1 while the cube can be used.
@group(3) @binding(16) var<uniform> neb_cube: vec4<f32>;

const NEB_VOLUMES: u32 = 4u;
const NEB_MAX_STEPS: u32 = 384u;
// Steps are this many voxels where the volume is occupied (jittered per
// pixel; TAA averages the sampling noise).
const NEB_STEP: f32 = 1.5;
// Stop once even the reddest bin is dimmed below 10⁻⁵.
const NEB_TAU_MAX: f32 = 18.0;
// Extinction A_λ/A_V = (λ / 550 nm)^−1.4, the optical slope of the
// Cardelli et al. (1989) law with R_V = 3.1, at the continuum anchors
// (427.5, 527.5, 627.5, 727.5 nm: bins 1, 5, 9, 13) and at the lines
// (Hβ 486.1, [O III] 500.7, [O I] 630.0, Hα 656.3; [S II] 672.4).
const NEB_EXT: f32 = -1.4;
const NEB_ANCHORS: vec4<f32> = vec4<f32>(427.5, 527.5, 627.5, 727.5);
const NEB_K_ANCHORS: vec4<f32> = vec4<f32>(1.4231, 1.0602, 0.8315, 0.6760);
const NEB_K_LINES: vec4<f32> = vec4<f32>(1.1890, 1.1405, 0.8268, 0.7808);
const NEB_K_SII: f32 = 0.7549;

// Rotation between the two detail lookups (about (1, 2, 3), 50°).
const NEB_DETAIL_ROT: mat3x3<f32> = mat3x3<f32>(
    vec3<f32>(0.6683, 0.6652, -0.3329),
    vec3<f32>(-0.5632, 0.7448, 0.3578),
    vec3<f32>(0.4860, -0.0516, 0.8724),
);

// Jitter of the sample positions along a pixel's ray (negative: not set).
var<private> neb_jitter: f32 = -1.0;

struct NebAcc {
    la: vec4<f32>,    // Hβ, [O III], [O I], Hα (W m⁻² sr⁻¹, static frame)
    lb: vec2<f32>,    // [N II], [S II]
    cont: vec4<f32>,  // continuum at the anchors (W m⁻² sr⁻¹ nm⁻¹)
    mv: vec3<f32>,    // Σ w, Σ w v, Σ w (v² + σ²): line-of-sight velocity moments (c), weighted by Hα
    tau: f32,         // dust optical depth at 550 nm
}

struct NebRay {
    u0: vec3<f32>,  // texture coordinates at t = 0
    ud: vec3<f32>,  // their rate of change per M along the ray
    t0: f32,        // entry
    t1: f32,        // exit
}

fn neb_tex(k: u32, u: vec3<f32>) -> vec4<f32> {
    switch k {
        case 0u: { return textureSampleLevel(neb_vol0, neb_clamp, u, 0.0); }
        case 1u: { return textureSampleLevel(neb_vol1, neb_clamp, u, 0.0); }
        case 2u: { return textureSampleLevel(neb_vol2, neb_clamp, u, 0.0); }
        default: { return textureSampleLevel(neb_vol3, neb_clamp, u, 0.0); }
    }
}

// Occupancy of an 8³ block (mip level 3: maxima of r and g).
fn neb_occ(k: u32, c: vec3<i32>) -> vec2<f32> {
    switch k {
        case 0u: { return textureLoad(neb_vol0, c, 3).rg; }
        case 1u: { return textureLoad(neb_vol1, c, 3).rg; }
        case 2u: { return textureLoad(neb_vol2, c, 3).rg; }
        default: { return textureLoad(neb_vol3, c, 3).rg; }
    }
}

fn neb_vel(k: u32, u: vec3<f32>) -> vec3<f32> {
    switch k {
        case 0u: { return textureSampleLevel(neb_vel0, neb_clamp, u, 0.0).xyz; }
        case 1u: { return textureSampleLevel(neb_vel1, neb_clamp, u, 0.0).xyz; }
        case 2u: { return textureSampleLevel(neb_vel2, neb_clamp, u, 0.0).xyz; }
        default: { return textureSampleLevel(neb_vel3, neb_clamp, u, 0.0).xyz; }
    }
}

fn neb_safe_inv(v: vec3<f32>) -> vec3<f32> {
    return 1.0 / select(v, vec3<f32>(1e-30), abs(v) < vec3<f32>(1e-30));
}

// Emission and absorption of volume k at texture coordinates u (position p,
// ray direction rd) over a step dt (M); fp is the pixel footprint there (M).
fn neb_sample(acc: ptr<function, NebAcc>, k: u32, u: vec3<f32>, p: vec3<f32>, rd: vec3<f32>, dt: f32, fp: f32) {
    let v = neb.vols[k];
    let dims = vec3<f32>(v.ax.w, v.ay.w, v.az.w);
    let vox = v.centre.w;
    // Sub-voxel detail: a log-normal density modulation (turbulent gas) and
    // a small warp of the lookup, faded out where it would be smaller than a
    // pixel. Two lookups of the tiling noise texture at incommensurate
    // scales and orientations (tiles of 13.7 and 5.3 voxels) hide its period.
    let q = u * dims;
    let da = textureSampleLevel(neb_detail, neb_repeat, q * (1.0 / 13.7), 0.0);
    let db = textureSampleLevel(neb_detail, neb_repeat, NEB_DETAIL_ROT * q * (1.0 / 5.3) + 0.37, 0.0);
    let a1 = v.detail.x * (1.0 - smoothstep(0.5, 2.0, fp / (2.0 * vox)));
    let a2 = v.detail.x * 0.8 * (1.0 - smoothstep(0.5, 2.0, fp / (0.5 * vox)));
    let uw = u + (vec3<f32>(da.b, da.a, db.b) - 0.5) * (3.0 * a1) / dims;
    let s = neb_tex(k, uw);
    if (max(s.r, s.g) <= 1e-7) {
        return;
    }
    let f = exp(a1 * (da.r + db.r - 1.0) * 1.4 + a2 * (da.g + db.g - 1.0) * 1.4);
    let ion = s.r * f * f;
    let gas = s.g * f;
    let dtau = (v.emit.y * gas + v.emit.z * sqrt(ion)) * dt;
    // Dust in front of this sample, and half of its own.
    let tm = (*acc).tau + 0.5 * dtau;
    let ha = v.emit.x * ion * dt;
    if (ha > 0.0) {
        let tl = exp(-tm * NEB_K_LINES);
        let r = v.ratios + v.ratios_b * s.b;
        let hb = ha / 2.86;
        let oiii = r.w + v.extra.x * s.a;
        (*acc).la += vec4<f32>(hb, hb * oiii, ha * r.z, ha) * tl;
        (*acc).lb += vec2<f32>(ha * r.x * tl.w, ha * r.y * exp(-tm * NEB_K_SII));
        let vlos = -dot(neb_vel(k, u), rd) * v.detail.z;
        let w = ha * tl.w;
        (*acc).mv += w * vec3<f32>(1.0, vlos, vlos * vlos + v.detail.y);
    }
    // Continuum: synchrotron, and starlight from the central cluster
    // scattered once by the dust (Henyey–Greenstein phase function; the
    // cluster's light reaching this voxel is channel a).
    var c = v.synch * (v.emit.w * gas);
    if (v.extra.y > 0.0) {
        let r2 = dot(p, p);
        let mu = dot(p, -rd) * inverseSqrt(max(r2, 1.0));
        let hg = neb.sca2.x;
        let phase = (1.0 - hg * hg) / (4.0 * PI * pow(1.0 + hg * hg - 2.0 * hg * mu, 1.5));
        c += neb.sca * (v.extra.y * s.a * (dtau / dt) * phase / (r2 + neb.sca2.y));
    }
    (*acc).cont += c * exp(-tm * NEB_K_ANCHORS) * dt;
    (*acc).tau += dtau;
}

// March the straight ray ro + t rd for t in [t_lo, t_hi] through all
// volumes, with steps of about a voxel where a volume is occupied, skipping
// empty 8³ blocks and the space between boxes.
fn neb_march(ro: vec3<f32>, rd: vec3<f32>, t_lo: f32, t_hi: f32, jitter: f32) -> NebAcc {
    var acc: NebAcc;
    var rays: array<NebRay, 4>;
    var t_start = 3.0e38;
    var t_end = -1.0;
    for (var k = 0u; k < NEB_VOLUMES; k++) {
        let v = neb.vols[k];
        let rel = ro - v.centre.xyz;
        var r: NebRay;
        r.u0 = vec3<f32>(dot(rel, v.ax.xyz), dot(rel, v.ay.xyz), dot(rel, v.az.xyz)) * 0.5 + 0.5;
        r.ud = vec3<f32>(dot(rd, v.ax.xyz), dot(rd, v.ay.xyz), dot(rd, v.az.xyz)) * 0.5;
        let inv = neb_safe_inv(r.ud);
        let ta = -r.u0 * inv;
        let tb = (1.0 - r.u0) * inv;
        let tn = min(ta, tb);
        let tf = max(ta, tb);
        r.t0 = max(max(tn.x, tn.y), max(tn.z, t_lo));
        r.t1 = min(min(tf.x, tf.y), min(tf.z, t_hi));
        rays[k] = r;
        if (r.t0 < r.t1) {
            t_start = min(t_start, r.t0);
            t_end = max(t_end, r.t1);
        }
    }
    if (t_start >= t_end) {
        return acc;
    }

    let obs = frame.obs.yzw;
    let d0 = length(ro - obs);
    let pix = frame.cam.z;
    // The pulsar: a point source spread over the point-spread core, added
    // when the march passes it (so the dust in front dims it).
    let pp = neb.pulsar.xyz - ro;
    let tp = dot(pp, rd);
    let dist_p = max(length(neb.pulsar.xyz - obs), 1.0);
    let ang = length(pp - rd * tp) / dist_p;
    let sig2 = STAR_SIGMA * STAR_SIGMA + 0.35 * pix * pix;
    var pulsar = 0.0;
    if (tp > t_lo && tp < t_hi && ang * ang < 40.0 * sig2) {
        pulsar = exp(-0.5 * ang * ang / sig2) / (TAU * sig2 * dist_p * dist_p);
    }

    var t = t_start;
    for (var i = 0u; i < NEB_MAX_STEPS && t < t_end; i++) {
        var dt = t_end - t;
        var occupied = 0u;
        for (var k = 0u; k < NEB_VOLUMES; k++) {
            let r = rays[k];
            if (t >= r.t1) {
                continue;
            }
            if (t < r.t0) {
                dt = min(dt, r.t0 - t);
                continue;
            }
            let v = neb.vols[k];
            let u = r.u0 + r.ud * t;
            let cd = max(floor(vec3<f32>(v.ax.w, v.ay.w, v.az.w) * 0.125), vec3<f32>(1.0));
            let cell = clamp(floor(u * cd), vec3<f32>(0.0), cd - 1.0);
            let occ = neb_occ(k, vec3<i32>(cell));
            if (occ.x > v.extra.z || occ.y > v.extra.w) {
                occupied |= 1u << k;
                // About a voxel, longer where voxels are well below a pixel.
                let fp = (d0 + t) * pix;
                dt = min(dt, v.centre.w * clamp(0.5 * fp / v.centre.w, NEB_STEP, 6.0));
            } else {
                let next = (cell + step(vec3<f32>(0.0), r.ud)) / cd;
                let te = (next - u) * neb_safe_inv(r.ud);
                dt = min(dt, max(min(min(te.x, te.y), te.z), 0.0) + 0.01 * v.centre.w);
            }
        }
        dt = max(dt, 1.0);
        if (occupied != 0u) {
            let ts = t + jitter * dt;
            let p = ro + rd * ts;
            let fp = (d0 + ts) * pix;
            for (var k = 0u; k < NEB_VOLUMES; k++) {
                if ((occupied & (1u << k)) != 0u) {
                    neb_sample(&acc, k, rays[k].u0 + rays[k].ud * ts, p, rd, dt, fp);
                }
            }
        }
        if (pulsar > 0.0 && tp < t + dt) {
            acc.cont += neb.pulsar_l * pulsar * exp(-acc.tau * NEB_K_ANCHORS);
            pulsar = 0.0;
        }
        if (acc.tau > NEB_TAU_MAX) {
            break;
        }
        t += dt;
    }
    return acc;
}

// Add a Gaussian line of intensity `i` (W m⁻² sr⁻¹) centred at mu with
// width sigma (nm) to the bins it overlaps (per nm, bin averages).
fn neb_line(bins: ptr<function, array<f32, 16>>, mu: f32, sigma: f32, i: f32) {
    let b = i32(floor((mu - SPEC_L0) / SPEC_DL));
    if (b < -1 || b > 16) {
        return;
    }
    let edges = SPEC_L0 + SPEC_DL * vec4<f32>(f32(b - 1), f32(b), f32(b + 1), f32(b + 2));
    let cdf = 0.5 * erf4((edges - mu) / (sqrt(2.0) * max(sigma, 0.05)));
    for (var j = 0; j < 3; j++) {
        let k = b - 1 + j;
        if (k >= 0 && k < 16) {
            (*bins)[k] += i * (cdf[j + 1] - cdf[j]) / SPEC_DL;
        }
    }
}

// Log–log interpolation of the continuum anchors at wavelength l (nm).
fn neb_cont_at(la: vec4<f32>, xa: vec4<f32>, l: f32) -> f32 {
    let x = log(l);
    var i = 0;
    if (x > xa.y) {
        i = 1;
    }
    if (x > xa.z) {
        i = 2;
    }
    let s = clamp((la[i + 1] - la[i]) / (xa[i + 1] - xa[i]), -8.0, 8.0);
    return exp(la[i] + s * (x - xa[i]));
}

// What an observer with shift g sees of the accumulated emission and dust.
fn neb_assemble(acc: NebAcc, g: f32) -> Medium {
    var m = medium_clear();
    let lines = acc.la.x + acc.la.y + acc.la.z + acc.la.w + acc.lb.x + acc.lb.y;
    let cont = acc.cont.x + acc.cont.y + acc.cont.z + acc.cont.w;
    if (acc.tau <= 0.0 && lines <= 0.0 && cont <= 0.0) {
        return m;
    }
    let gc = clamp(g, 1e-3, 1e3);
    let g2 = gc * gc;
    // Dust at rest absorbs the static-frame wavelength g λ.
    m.T = spec_transmit(spec_scale(spec_power_law(NEB_EXT), acc.tau * pow(gc, NEB_EXT)));
    var bins: array<f32, 16>;
    if (lines > 0.0) {
        let w = max(acc.mv.x, 1e-30);
        let vbar = acc.mv.y / w;
        let sv = sqrt(max(acc.mv.z / w - vbar * vbar, 0.0));
        // Line centres at λ₀ / (g D), widths λ₀ σ_v / g; intensities × g⁴.
        let gl = gc * (1.0 + vbar);
        let s = sv / gc;
        let g4 = g2 * g2;
        let hb = acc.la.x * g4;
        neb_line(&bins, 656.28 / gl, 656.28 * s, acc.la.w * g4);
        neb_line(&bins, 486.13 / gl, 486.13 * s, hb);
        neb_line(&bins, 434.05 / gl, 434.05 * s, 0.468 * hb);
        neb_line(&bins, 410.17 / gl, 410.17 * s, 0.259 * hb);
        neb_line(&bins, 500.68 / gl, 500.68 * s, 0.75 * acc.la.y * g4);
        neb_line(&bins, 495.89 / gl, 495.89 * s, 0.25 * acc.la.y * g4);
        neb_line(&bins, 630.03 / gl, 630.03 * s, 0.75 * acc.la.z * g4);
        neb_line(&bins, 636.38 / gl, 636.38 * s, 0.25 * acc.la.z * g4);
        neb_line(&bins, 658.35 / gl, 658.35 * s, 0.75 * acc.lb.x * g4);
        neb_line(&bins, 654.80 / gl, 654.80 * s, 0.25 * acc.lb.x * g4);
        neb_line(&bins, 671.64 / gl, 671.64 * s, 0.45 * acc.lb.y * g4);
        neb_line(&bins, 673.08 / gl, 673.08 * s, 0.55 * acc.lb.y * g4);
    }
    if (cont > 0.0) {
        let la = log(max(acc.cont, vec4<f32>(1e-35)));
        let xa = log(NEB_ANCHORS);
        let g5 = g2 * g2 * gc;
        for (var k = 0; k < 16; k++) {
            bins[k] += g5 * neb_cont_at(la, xa, gc * (SPEC_L0 + SPEC_DL * (f32(k) + 0.5)));
        }
    }
    m.L = Spectrum(
        vec4<f32>(bins[0], bins[1], bins[2], bins[3]),
        vec4<f32>(bins[4], bins[5], bins[6], bins[7]),
        vec4<f32>(bins[8], bins[9], bins[10], bins[11]),
        vec4<f32>(bins[12], bins[13], bins[14], bins[15]),
    );
    return m;
}

fn neb_pixel_jitter(dir: vec3<f32>) -> f32 {
    if (neb_jitter < 0.0) {
        neb_jitter = hash3(bitcast<vec3<u32>>(dir)).x;
    }
    return neb_jitter;
}

// Along the straight chord a → b (one step of the Kerr trace, going back in
// time from the observer). Only matters when the ship is out among the
// nebulae: chords within the empty region around the hole or outside
// everything return at once.
fn nebula_segment(a: vec3<f32>, b: vec3<f32>, g: f32) -> Medium {
    let r_in2 = neb.sca2.z * neb.sca2.z;
    if (neb.pulsar.w == 0.0 || (dot(a, a) < r_in2 && dot(b, b) < r_in2)) {
        return medium_clear();
    }
    let d = b - a;
    let len = length(d);
    if (len <= 0.0) {
        return medium_clear();
    }
    let rd = d / len;
    let tc = clamp(-dot(a, rd), 0.0, len);
    let closest = a + rd * tc;
    if (dot(closest, closest) > neb.sca2.w * neb.sca2.w) {
        return medium_clear();
    }
    return neb_assemble(neb_march(a, rd, 0.0, len, neb_pixel_jitter(rd)), g);
}

// Cube face and texture coordinates of direction d: (u, v, face).
fn neb_cube_face(d: vec3<f32>) -> vec3<f32> {
    let a = abs(d);
    var st: vec2<f32>;
    var face: f32;
    if (a.x >= a.y && a.x >= a.z) {
        face = select(1.0, 0.0, d.x > 0.0);
        st = select(vec2<f32>(d.z, -d.y), vec2<f32>(-d.z, -d.y), d.x > 0.0) / a.x;
    } else if (a.y >= a.z) {
        face = select(3.0, 2.0, d.y > 0.0);
        st = select(vec2<f32>(d.x, -d.z), vec2<f32>(d.x, d.z), d.y > 0.0) / a.y;
    } else {
        face = select(5.0, 4.0, d.z > 0.0);
        st = select(vec2<f32>(-d.x, -d.y), vec2<f32>(d.x, -d.y), d.z > 0.0) / a.z;
    }
    return vec3<f32>(st * 0.5 + 0.5, face);
}

// From `p` along `dir` out to infinity, after the ray has escaped the hole.
fn nebula_escape(p: vec3<f32>, dir: vec3<f32>, g: f32) -> Medium {
    if (neb.pulsar.w == 0.0) {
        return medium_clear();
    }
    if (neb_cube.w > 0.5) {
        var acc: NebAcc;
        let f = neb_cube_face(dir);
        let face = u32(f.z);
        let b = textureSampleLevel(neb_cube_b, neb_clamp, f.xy, face, 0.0);
        let d = textureSampleLevel(neb_cube_d, neb_clamp, f.xy, face, 0.0);
        acc.la = textureSampleLevel(neb_cube_a, neb_clamp, f.xy, face, 0.0);
        acc.lb = b.xy;
        acc.tau = b.z;
        acc.cont = textureSampleLevel(neb_cube_c, neb_clamp, f.xy, face, 0.0);
        acc.mv = vec3<f32>(b.w, d.xy);
        return neb_assemble(acc, g);
    }
    return neb_assemble(neb_march(p, dir, 0.0, 3.0e38, neb_pixel_jitter(dir)), g);
}
