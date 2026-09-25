// ---------------------------------------------------------------------------
// Far field: the pixel's backward null geodesic through Kerr spacetime
// (horizon, nearby star discs, nebulae along the way), then the distant
// sky it escapes to. Spectral port of the web renderer's `sky.wgsl`.
// ---------------------------------------------------------------------------

@group(0) @binding(2) var<uniform> spheres: Spheres;

const FAR_ESCAPED: u32 = 0u;
const FAR_HORIZON: u32 = 1u;
const FAR_SURFACE: u32 = 2u;
const FAR_LOST: u32 = 3u;

struct FarTrace {
    m: Medium,        // picked up along the way; its T multiplies the sky
    kind: u32,
    dir: vec3<f32>,   // escape direction (coordinates)
    pos: vec3<f32>,   // escape position (relative to the hole, M)
    g: f32,           // frequency shift relative to a static observer at infinity
}

// Stars owned by a near-field system are drawn there instead.
fn sphere_in_near_field(body: f32) -> bool {
    for (var i = 0u; i < hq.near.x; i++) {
        if (systems[i].light.z == body) {
            return true;
        }
    }
    return false;
}

struct DiscHit {
    hit: bool,
    L: Spectrum,
}

// Does the step a → b (observer-relative) hit a nearby star's surface? The
// star is placed where it is when the ray passes it. Discs smaller than a
// pixel spread their light over the pixel.
fn sphere_hit(a: vec4<f32>, b: vec4<f32>, p: vec4<f32>) -> DiscHit {
    var out: DiscHit;
    out.hit = false;
    var best = 2.0;
    let d = b.yzw - a.yzw;
    let len = length(d);
    if (len <= 0.0) {
        return out;
    }
    let dir = d / len;
    for (var j = 0u; j < u32(frame.extra.z); j++) {
        let sp = spheres.items[j];
        if (sphere_in_near_field(sp.star.z)) {
            continue;
        }
        let radius = sp.vel.w;
        var t_hit = sp.center.w;
        var s = -1.0;
        var c = sp.center.xyz;
        for (var it = 0; it < 3; it++) {
            c = sp.center.xyz + sp.vel.xyz * (t_hit - sp.center.w);
            let oc = a.yzw - c;
            let t0 = -dot(oc, dir);
            let miss = length(cross(oc, dir));
            if (miss >= radius || length(oc) <= radius) {
                s = -1.0;
                break;
            }
            s = (t0 - sqrt(radius * radius - miss * miss)) / len;
            t_hit = mix(a.x, b.x, s);
        }
        if (s < 0.0 || s > 1.0 || s >= best) {
            continue;
        }
        best = s;
        out.hit = true;
        let x = a.yzw + d * s;
        let nrm = normalize(x - c);
        let mu = max(dot(nrm, -dir), 0.0);
        let w = four_velocity(frame.obs.yzw + x, sp.vel.xyz);
        let g = clamp(1.0 / max(dot(p, w), 1e-6), 1e-3, 1e3);
        let dist = max(length(c), radius);
        let disc = PI * radius * radius / (dist * dist);
        let limb = (1.0 - 0.6 * (1.0 - mu)) / 0.8;
        let cells = 0.9 + 0.2 * value_noise(nrm * 60.0 + vec3<f32>(f32(j) * 17.0));
        let cover = min(disc / hq.view.w, 1.0);
        out.L = spec_scale(spec_planck(sp.star.x * g), limb * cells * cover);
    }
    return out;
}

fn far_trace(n: vec3<f32>) -> FarTrace {
    var out: FarTrace;
    out.m = medium_clear();
    out.kind = FAR_LOST;
    out.dir = n;
    out.pos = vec3<f32>(0.0);
    out.g = 1.0;
    var s = backward_ray(n);
    let rp = frame.kerr.z;
    let r_esc = max(frame.kerr.w, 2.5 * length(frame.obs.yzw));
    for (var i = 0u; i < frame.counts2.x; i++) {
        let pos = abs_pos(s.x);
        let r = ks_radius(pos);
        if (r - rp < frame.march.w) {
            out.kind = FAR_HORIZON;
            break;
        }
        let k1 = phase_rhs(s);
        let h = ray_step(s, k1, r);
        let nxt = lens_kick(s, rk4(s, k1, h));
        if (lens_captured) {
            out.kind = FAR_HORIZON;
            break;
        }
        out.m = medium_over(out.m, nebula_segment(pos, abs_pos(nxt.x), 1.0 / max(nxt.p.x, 1e-6)));
        if (frame.extra.z > 0.0) {
            let sh = sphere_hit(s.x, nxt.x, nxt.p);
            if (sh.hit) {
                out.kind = FAR_SURFACE;
                out.m = medium_over(out.m, Medium(sh.L, spec(0.0)));
                break;
            }
        }
        s = nxt;
        if (r > r_esc && dot(pos, k1.x.yzw) > 0.0) {
            out.kind = FAR_ESCAPED;
            out.dir = normalize(phase_rhs(s).x.yzw);
            out.pos = abs_pos(s.x);
            out.g = 1.0 / s.p.x;
            break;
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// The distant sky: a galaxy seen from its own nucleus. Intensities are in
// the sky's own units, scaled to radiance by `hq.radiometry.y`.
// ---------------------------------------------------------------------------

const STAR_SIGMA: f32 = 2.5e-4; // intrinsic star blur (rad)

fn cube_uv(d: vec3<f32>) -> vec3<f32> {
    let a = abs(d);
    if (a.x >= a.y && a.x >= a.z) {
        return vec3<f32>(d.y / a.x, d.z / a.x, select(0.0, 1.0, d.x < 0.0));
    }
    if (a.y >= a.z) {
        return vec3<f32>(d.x / a.y, d.z / a.y, select(2.0, 3.0, d.y < 0.0));
    }
    return vec3<f32>(d.x / a.z, d.y / a.z, select(4.0, 5.0, d.z < 0.0));
}

fn cube_dir(face: u32, uv: vec2<f32>) -> vec3<f32> {
    switch face {
        case 0u: { return vec3<f32>(1.0, uv.x, uv.y); }
        case 1u: { return vec3<f32>(-1.0, uv.x, uv.y); }
        case 2u: { return vec3<f32>(uv.x, 1.0, uv.y); }
        case 3u: { return vec3<f32>(uv.x, -1.0, uv.y); }
        case 4u: { return vec3<f32>(uv.x, uv.y, 1.0); }
        default: { return vec3<f32>(uv.x, uv.y, -1.0); }
    }
}

// One layer of point stars, uniform on the sphere (see the web `sky.wgsl`).
fn star_layer(d: vec3<f32>, cells: f32, density: f32, flux_scale: f32, fp: f32, g: f32, seed: u32) -> Spectrum {
    let c = cube_uv(d);
    let face = u32(c.z);
    let grid = (c.xy * 0.5 + 0.5) * cells;
    let cell = floor(grid);
    let ic = vec2<u32>(cell);
    let h = hash3(vec3<u32>(ic.x, ic.y, face + seed * 16u));
    let h2 = hash3(vec3<u32>(ic.y + 911u, ic.x + 17u, face * 7u + seed * 131u + 3u));
    let centre = (cell + 0.5) / cells * 2.0 - 1.0;
    let q = 1.0 + dot(centre, centre);
    let jac = pow(q, -1.5);
    let cell0 = 2.0 / cells;
    let cell_ang = cell0 * pow(q, -0.75);
    let star_dir = normalize(cube_dir(face, (cell + 0.08 + 0.84 * h.yz) / cells * 2.0 - 1.0));
    let ang = length(d - star_dir);
    let sigma2 = STAR_SIGMA * STAR_SIGMA + fp * fp;
    let present = select(0.0, 1.0, h.x < density * jac);
    let flux = flux_scale * pow(h2.x, 5.0) * present;
    let temp = 2600.0 + 26000.0 * pow(h2.y, 3.5);
    let resolved = flux * exp(-0.5 * ang * ang / sigma2) / (TAU * sigma2);
    let mean = 0.3 * density * flux_scale / 6.0 / (cell0 * cell0);
    let w = smoothstep(0.1, 0.25, fp / cell_ang);
    return spec_mix(spec_scale(spec_planck_unit(temp * g), resolved), spec_scale(spec_planck_unit(5200.0 * g), mean), w);
}

fn value_noise(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let o = vec3<u32>(vec3<i32>(i) + vec3<i32>(4096));
    let n000 = hash3(o).x;
    let n100 = hash3(o + vec3<u32>(1u, 0u, 0u)).x;
    let n010 = hash3(o + vec3<u32>(0u, 1u, 0u)).x;
    let n110 = hash3(o + vec3<u32>(1u, 1u, 0u)).x;
    let n001 = hash3(o + vec3<u32>(0u, 0u, 1u)).x;
    let n101 = hash3(o + vec3<u32>(1u, 0u, 1u)).x;
    let n011 = hash3(o + vec3<u32>(0u, 1u, 1u)).x;
    let n111 = hash3(o + vec3<u32>(1u, 1u, 1u)).x;
    return mix(
        mix(mix(n000, n100, u.x), mix(n010, n110, u.x), u.y),
        mix(mix(n001, n101, u.x), mix(n011, n111, u.x), u.y),
        u.z,
    );
}

fn fbm(p: vec3<f32>) -> f32 {
    var s = 0.0;
    var a = 0.5;
    var q = p;
    for (var i = 0; i < 4; i++) {
        s += a * value_noise(q);
        q = q * 2.03 + vec3<f32>(1.7, 9.2, 3.1);
        a *= 0.5;
    }
    return s;
}

fn sky_radiance(dir: vec3<f32>, g: f32, fp: f32) -> Spectrum {
    let gx = normalize(vec3<f32>(0.83, 0.0, 0.56));
    let gz = normalize(vec3<f32>(-0.56, 0.12, 0.83));
    let gy = cross(gz, gx);
    let d = vec3<f32>(dot(dir, gx), dot(dir, gy), dot(dir, gz));
    let lat = asin(clamp(d.z, -1.0, 1.0));
    let gc = clamp(g, 0.02, 40.0);
    let g4 = gc * gc * gc * gc;

    let band = exp(-pow(lat / 0.16, 2.0));
    let dust = smoothstep(0.42, 0.72, fbm(d * 7.0 + vec3<f32>(3.0)));
    let lanes = 1.0 - 0.85 * dust * exp(-pow(lat / 0.07, 2.0));
    let clumps = 0.55 + 0.9 * fbm(d * 18.0);
    let bulge = 0.35 * exp(-pow(lat / 0.55, 2.0));
    var diffuse = (band * clumps * lanes * 1.4 + bulge + 0.04) * 0.008;
    diffuse *= 1.0 + 0.5 * pow(max(d.x, 0.0), 6.0);

    let dens = 0.35 + 0.65 * band;
    var s = spec_scale(spec_planck_unit(4300.0 * g), diffuse);
    s = spec_add(s, star_layer(dir, 24.0, 0.5, 6.0e-4, fp, g, 1u));
    s = spec_add(s, star_layer(dir, 60.0, 0.55, 1.5e-4, fp, g, 2u));
    s = spec_add(s, star_layer(dir, 170.0, 0.6 * dens, 3.0e-5, fp, g, 3u));
    s = spec_add(s, star_layer(dir, 420.0, 0.8 * dens, 6.0e-6, fp, g, 4u));
    return spec_scale(s, g4 * hq.radiometry.y);
}

// What the escaped ray sees: nebulae in front of the galaxy.
fn far_sky(f: FarTrace, fp: f32) -> Spectrum {
    let neb = nebula_escape(f.pos, f.dir, f.g);
    return spec_fma(neb.T, sky_radiance(f.dir, f.g, fp), neb.L);
}
