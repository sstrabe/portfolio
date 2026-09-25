// ---------------------------------------------------------------------------
// Lensing by stellar-mass black holes (hook called on every step of the
// far-field Kerr trace in `far.wgsl`). The model and its derivation are in
// `lens.rs`, whose f64 functions this mirrors:
//
// - each hole works in its rest frame (tetrad `e`, dual basis `w`), where it
//   sits at the spatial origin;
// - along a step's straight chord the weak field turns the photon towards
//   the hole by (2m/b) Δ(sin φ) and shifts the chord's end sideways (Born),
//   counted only within `reach`, so the sum over steps doesn't depend on
//   where they fall;
// - a chord that enters the `bubble` around the hole continues there on the
//   exact Schwarzschild orbit (d²U/dφ² = 3U² − U in isotropic coordinates)
//   and leaves it with the orbit's exit point and direction, or is captured
//   below the photon sphere: the hole's shadow.
//
// `a` is the ray state before a step and `b` after it (events relative to
// the observer, covariant momentum); the result replaces `b`.
// ---------------------------------------------------------------------------

struct Lens {
    event: vec4<f32>,           // the hole's event relative to the observer (t, x, y, z)
    body: vec4<f32>,            // mass, coordinate velocity
    radii: vec4<f32>,           // bubble radius, reach (units of M), unused, unused
    e: array<vec4<f32>, 4>,     // rest-frame tetrad e_a^μ (u, x, y, z)
    w: array<vec4<f32>, 4>,     // dual basis θ^a_μ
}

struct Lenses {
    count: vec4<u32>,
    obs: vec4<f32>,             // product of the holes' lapses at the observer, unused
    items: array<Lens, 8>,
}

@group(5) @binding(0) var<uniform> lenses: Lenses;

const LENS_BUBBLE_STEP: f32 = 0.06;
const LENS_BUBBLE_MAX_STEPS: u32 = 600u;

// Set when a ray falls into a hole (the far-field trace ends it as black).
var<private> lens_captured: bool = false;

fn lens_one_plus(z: f32, b: f32) -> f32 {
    let rho = length(vec2<f32>(z, b));
    return select(1.0 + z / rho, b * b / (rho * (rho - z)), z < 0.0);
}

fn lens_one_minus(z: f32, b: f32) -> f32 {
    let rho = length(vec2<f32>(z, b));
    return select(1.0 - z / rho, b * b / (rho * (rho + z)), z > 0.0);
}

fn lens_rho_plus(z: f32, b: f32) -> f32 {
    let rho = length(vec2<f32>(z, b));
    return select(rho + z, b * b / (rho - z), z < 0.0);
}

fn lens_rho_minus(z: f32, b: f32) -> f32 {
    let rho = length(vec2<f32>(z, b));
    return select(rho - z, b * b / (rho + z), z > 0.0);
}

// Turn towards the mass (rad) and sideways shift at z2 picked up along
// z1..z2 of a line at impact parameter b (see `lens::born`).
fn lens_born(m: f32, b: f32, z1: f32, z2: f32, reach: f32) -> vec2<f32> {
    let zm = sqrt(max(reach * reach - b * b, 0.0));
    let c1 = clamp(z1, -zm, zm);
    let c2 = clamp(z2, -zm, zm);
    if (c2 <= c1 || b <= 0.0) {
        return vec2<f32>(0.0);
    }
    let k = 2.0 * m / b;
    var turn: f32;
    var area: f32;
    if (c1 < 0.0) {
        let f1 = lens_one_plus(c1, b);
        turn = lens_one_plus(c2, b) - f1;
        area = lens_rho_plus(c2, b) - lens_rho_plus(c1, b) - f1 * (c2 - c1);
    } else {
        let g1 = lens_one_minus(c1, b);
        turn = g1 - lens_one_minus(c2, b);
        area = g1 * (c2 - c1) + lens_rho_minus(c2, b) - lens_rho_minus(c1, b);
    }
    return vec2<f32>(k * turn, k * area + k * turn * (z2 - c2));
}

fn lens_u_of(m: f32, rho: f32) -> f32 {
    let q = m / rho;
    return q / ((1.0 + 0.5 * q) * (1.0 + 0.5 * q));
}

// Orbit through the bubble from isotropic radius rho0 at angle psi0 from
// the outward radial: (1, φ at exit, ψ at exit) or (0, ·, ·) if captured.
fn lens_bubble(m: f32, rho0: f32, psi0: f32, rho_exit: f32) -> vec3<f32> {
    let s0 = sin(psi0);
    let c0 = cos(psi0);
    var u = lens_u_of(m, rho0);
    let u_exit = lens_u_of(m, rho_exit);
    if (s0 < 1e-6) {
        return select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0), c0 < 0.0);
    }
    var w = -u * sqrt(max(1.0 - 2.0 * u, 0.0)) * c0 / s0;
    if (w > 0.0 && u < 1.0 / 3.0 && w * w + u * u * (1.0 - 2.0 * u) > 1.0 / 27.0) {
        return vec3<f32>(0.0);
    }
    var phi = 0.0;
    for (var i = 0u; i < LENS_BUBBLE_MAX_STEPS; i++) {
        let h = LENS_BUBBLE_STEP / (1.0 + abs(w) / u);
        let k1 = vec2<f32>(w, u * (3.0 * u - 1.0));
        let y2 = vec2<f32>(u, w) + 0.5 * h * k1;
        let k2 = vec2<f32>(y2.y, y2.x * (3.0 * y2.x - 1.0));
        let y3 = vec2<f32>(u, w) + 0.5 * h * k2;
        let k3 = vec2<f32>(y3.y, y3.x * (3.0 * y3.x - 1.0));
        let y4 = vec2<f32>(u, w) + h * k3;
        let k4 = vec2<f32>(y4.y, y4.x * (3.0 * y4.x - 1.0));
        let yn = vec2<f32>(u, w) + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        if (yn.x >= 0.5) {
            return vec3<f32>(0.0);
        }
        if (yn.y < 0.0 && yn.x <= u_exit) {
            let t = clamp((u - u_exit) / max(u - yn.x, 1e-30), 0.0, 1.0);
            let we = w + t * (yn.y - w);
            let tangential = u_exit * sqrt(1.0 - 2.0 * u_exit);
            return vec3<f32>(1.0, phi + t * h, atan2(tangential, -we));
        }
        u = yn.x;
        w = yn.y;
        phi += h;
    }
    return vec3<f32>(0.0);
}

// Rest-frame components (time, space) of an observer-relative event.
fn lens_to_rest(l: Lens, x: vec4<f32>) -> vec4<f32> {
    let d = x - l.event;
    return vec4<f32>(dot(l.w[0], d), dot(l.w[1], d), dot(l.w[2], d), dot(l.w[3], d));
}

fn lens_from_rest(l: Lens, r: vec4<f32>) -> vec4<f32> {
    return l.event + r.x * l.e[0] + r.y * l.e[1] + r.z * l.e[2] + r.w * l.e[3];
}

// Rotate v by angle `a` towards unit vector t (t ⟂ direction of v).
fn lens_turn(v: vec3<f32>, t: vec3<f32>, a: f32) -> vec3<f32> {
    let n = length(v);
    let d = v / n;
    let tp = normalize(t - dot(t, d) * d);
    return n * (d * cos(a) + tp * sin(a));
}

fn lens_kick(a: Phase, b: Phase) -> Phase {
    var out = b;
    for (var j = 0u; j < lenses.count.x; j++) {
        let l = lenses.items[j];
        let m = l.body.x;
        let bubble = l.radii.x;
        let reach = l.radii.y;
        let ra = lens_to_rest(l, a.x);
        let rb = lens_to_rest(l, out.x);
        let seg = rb.yzw - ra.yzw;
        let len = length(seg);
        if (len <= 0.0) {
            continue;
        }
        let d = seg / len;
        let along = -dot(ra.yzw, d);
        let bv = ra.yzw + along * d;
        let bl = length(bv);
        let z1 = -along;
        let z2 = len - along;
        if (bl >= reach || z2 < -reach || z1 > reach) {
            continue;
        }
        let toward = -bv / max(bl, 1e-30);
        // Momentum components in the rest frame: P_a = e_a · p.
        var pr = vec4<f32>(dot(l.e[0], out.p), dot(l.e[1], out.p), dot(l.e[2], out.p), dot(l.e[3], out.p));
        let z_in = -sqrt(max(bubble * bubble - bl * bl, 0.0));
        if (bl < bubble && z1 < z_in && z_in < z2) {
            // Up to the bubble: weak field; inside: the exact orbit.
            let kb = lens_born(m, bl, z1, z_in, reach);
            let entry = bv + z_in * d + kb.y * toward;
            let dir = normalize(lens_turn(d, toward, kb.x));
            let rho0 = length(entry);
            let radial = entry / rho0;
            let psi0 = atan2(length(cross(radial, dir)), dot(radial, dir));
            let orbit = lens_bubble(m, rho0, psi0, bubble);
            if (orbit.x < 0.5) {
                lens_captured = true;
                return out;
            }
            // The orbit's plane: radial and the perpendicular part of dir.
            var tang = dir - dot(dir, radial) * radial;
            if (length(tang) < 1e-12) {
                tang = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(radial.x) > 0.9);
                tang = tang - dot(tang, radial) * radial;
            }
            tang = normalize(tang);
            let exit_r = radial * cos(orbit.y) + tang * sin(orbit.y);
            let exit_t = -radial * sin(orbit.y) + tang * cos(orbit.y);
            let exit_dir = exit_r * cos(orbit.z) + exit_t * sin(orbit.z);
            // Leave with the rest-frame energy kept; the time spent inside
            // is taken as the straight distance (the Shapiro delay of a few
            // m is far below a pixel's worth of motion).
            let exit_x = exit_r * bubble;
            let t_entry = mix(ra.x, rb.x, (z_in - z1) / len);
            let t_exit = t_entry - length(exit_x - entry);
            let ps = length(pr.yzw);
            pr = vec4<f32>(pr.x, exit_dir * ps);
            out.x = lens_from_rest(l, vec4<f32>(t_exit, exit_x));
        } else {
            let kb = lens_born(m, bl, z1, z2, reach);
            if (kb.x == 0.0 && kb.y == 0.0) {
                continue;
            }
            pr = vec4<f32>(pr.x, lens_turn(pr.yzw, toward, kb.x));
            out.x = lens_from_rest(l, vec4<f32>(rb.x, rb.yzw + kb.y * toward));
        }
        // Back to coordinates: p_μ = P_a θ^a_μ.
        out.p = pr.x * l.w[0] + pr.y * l.w[1] + pr.z * l.w[2] + pr.w * l.w[3];
    }
    return out;
}
