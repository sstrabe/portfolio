// ---------------------------------------------------------------------------
// The pilot's own ship (hook called first by the trace in `trace.wgsl`).
//
// The ship moves with the camera, so it has no aberration or Doppler shift
// and plain perspective is exact for it. `n` is the pixel's direction in
// the view's axes; in the chase view the camera sits behind the ship
// (`ship/camera.rs`) and its axes are `ship.cam_x/y/z` in ship coordinates
// (metres; x forward, y left, z up).
//
// The mesh is traced in software through its BVH (`ship/bvh.rs`): nearest
// hit for the pixel, any hit for the shadow ray towards the star. Lighting:
// the star (shadowed by the ship and by a planet in the way), sunlight
// reflected by the planet below, skylight scaled by the baked ambient
// occlusion, and emission from the drive's plasma, hot radiators and the
// navigation lights.
//
// The reaction-control thrusters' plumes are drawn in front of the hull and
// the sky (`ship_plumes`).
// ---------------------------------------------------------------------------

struct ShipFrame {
    cam_pos: vec4<f32>,     // camera position (m), 1 when the ship is drawn
    cam_x: vec4<f32>,       // camera forward, in ship coordinates
    cam_y: vec4<f32>,       // camera left
    cam_z: vec4<f32>,       // camera up
    bound: vec4<f32>,       // bounding sphere centre (m), radius (m)
    sun_dir: vec4<f32>,     // unit direction to the star, 1 when it lights the ship
    planet_dir: vec4<f32>,  // unit direction to the planet below, 1 when it lights the ship
    glow: vec4<f32>,        // drive power (0–1), wall time (s), unused, unused
    sun: Spectrum,          // irradiance from the star, W m⁻² nm⁻¹
    planet: Spectrum,       // irradiance from the sunlit planet
    sky: Spectrum,          // irradiance from the rest of the sky (per hemisphere)
    rcs: vec4<f32>,         // number of firing RCS nozzles, unused ×3
    jets: array<vec4<f32>, 64>,  // per nozzle: exit (m), strength; exhaust direction, index
}

struct ShipNode {
    lo: vec3<f32>,
    a: u32,         // interior: second child; leaf: first triangle
    hi: vec3<f32>,
    b: u32,         // interior: split axis; leaf: count << 2 | 3
}

@group(4) @binding(0) var<uniform> ship: ShipFrame;
@group(4) @binding(1) var<storage, read> ship_vertices: array<vec4<f32>>;
@group(4) @binding(2) var<storage, read> ship_triangles: array<u32>;
@group(4) @binding(3) var<storage, read> ship_nodes: array<ShipNode>;

// Values of `ship::mesh::Material`.
const MAT_PAINT: u32 = 0u;
const MAT_METAL: u32 = 1u;
const MAT_RADIATOR: u32 = 2u;
const MAT_BELL: u32 = 3u;
const MAT_BELL_INNER: u32 = 4u;
const MAT_FOIL: u32 = 5u;
const MAT_DARK: u32 = 6u;
const MAT_LIGHT: u32 = 7u;

struct ShipHit {
    hit: bool,
    L: Spectrum,
    // RCS plumes in front of the hull (or the scene): their radiance and
    // transmittance.
    plume: Spectrum,
    plume_t: f32,
}

struct ShipHull {
    hit: bool,
    t: f32,
    L: Spectrum,
}

struct ShipRay {
    t: f32,
    tri: u32,
    uv: vec2<f32>,
}

fn ship_vertex(i: u32) -> vec3<f32> {
    return ship_vertices[2u * i].xyz;
}

// Möller–Trumbore: distance and barycentrics, or t < 0.
fn ship_triangle(o: vec3<f32>, d: vec3<f32>, tri: u32) -> vec3<f32> {
    let a = ship_vertex(ship_triangles[3u * tri]);
    let b = ship_vertex(ship_triangles[3u * tri + 1u]);
    let c = ship_vertex(ship_triangles[3u * tri + 2u]);
    let e1 = b - a;
    let e2 = c - a;
    let p = cross(d, e2);
    let det = dot(e1, p);
    if (abs(det) < 1e-9) {
        return vec3<f32>(-1.0);
    }
    let inv = 1.0 / det;
    let s = o - a;
    let u = dot(s, p) * inv;
    if (u < 0.0 || u > 1.0) {
        return vec3<f32>(-1.0);
    }
    let q = cross(s, e1);
    let v = dot(d, q) * inv;
    if (v < 0.0 || u + v > 1.0) {
        return vec3<f32>(-1.0);
    }
    return vec3<f32>(dot(e2, q) * inv, u, v);
}

fn ship_box(lo: vec3<f32>, hi: vec3<f32>, o: vec3<f32>, inv: vec3<f32>, t_max: f32) -> bool {
    let a = (lo - o) * inv;
    let b = (hi - o) * inv;
    let t0 = max(max(min(a.x, b.x), min(a.y, b.y)), max(min(a.z, b.z), 0.0));
    let t1 = min(min(max(a.x, b.x), max(a.y, b.y)), min(max(a.z, b.z), t_max));
    return t0 <= t1;
}

// Nearest hit before t_max (or, with `any`, the first one found).
fn ship_bvh(o: vec3<f32>, d: vec3<f32>, t_max: f32, any: bool) -> ShipRay {
    var best = ShipRay(t_max, 0xffffffffu, vec2<f32>(0.0));
    let inv = 1.0 / select(d, vec3<f32>(1e-12), abs(d) < vec3<f32>(1e-12));
    var stack: array<u32, 32>;
    var sp = 0u;
    var i = 0u;
    for (var guard = 0u; guard < 2048u; guard++) {
        let n = ship_nodes[i];
        var descend = false;
        if (ship_box(n.lo, n.hi, o, inv, best.t)) {
            if ((n.b & 3u) == 3u) {
                let count = n.b >> 2u;
                for (var k = 0u; k < count; k++) {
                    let h = ship_triangle(o, d, n.a + k);
                    if (h.x > 1e-4 && h.x < best.t) {
                        best = ShipRay(h.x, n.a + k, h.yz);
                        if (any) {
                            return best;
                        }
                    }
                }
            } else {
                // Visit the child on the ray's near side first.
                var first = i + 1u;
                var second = n.a;
                if (d[n.b] < 0.0) {
                    first = n.a;
                    second = i + 1u;
                }
                if (sp < 32u) {
                    stack[sp] = second;
                    sp++;
                }
                i = first;
                descend = true;
            }
        }
        if (!descend) {
            if (sp == 0u) {
                break;
            }
            sp--;
            i = stack[sp];
        }
    }
    return best;
}

// Gold's reflectance (the multi-layer insulation's outer film): low in the
// blue, high from the yellow on.
fn ship_gold() -> Spectrum {
    return spec_axpy(spec_ramp(470.0, 580.0), 0.6, spec(0.33));
}

fn ship_ggx(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, rough: f32) -> f32 {
    let h = normalize(v + l);
    let a2 = max(rough * rough * rough * rough, 1e-5);
    let nh = max(dot(n, h), 0.0);
    let d = a2 / (PI * pow(nh * nh * (a2 - 1.0) + 1.0, 2.0));
    let nv = max(dot(n, v), 1e-3);
    let nl = max(dot(n, l), 1e-3);
    let k = rough * rough * 0.5;
    let g = nv / (nv * (1.0 - k) + k) * nl / (nl * (1.0 - k) + k);
    return d * g / (4.0 * nv * nl);
}

fn ship_schlick(f0: Spectrum, c: f32) -> Spectrum {
    let x = pow(1.0 - clamp(c, 0.0, 1.0), 5.0);
    return spec_axpy(spec_sub(spec(1.0), f0), x, f0);
}

struct ShipSurface {
    albedo: Spectrum,  // diffuse
    f0: Spectrum,      // specular at normal incidence
    rough: f32,
    emission: Spectrum,
}

fn ship_surface(mat: u32, p: vec3<f32>) -> ShipSurface {
    var s: ShipSurface;
    s.emission = spec(0.0);
    s.f0 = spec(0.04);
    s.rough = 0.5;
    let power = ship.glow.x;
    switch (mat) {
        case MAT_PAINT: {
            // White panels with faint seams every 1.6 m and an orange band
            // around the crew section.
            let seam = smoothstep(0.02, 0.0, abs(fract(p.x / 1.6) - 0.5) - 0.48);
            let band = step(abs(p.x - 17.0), 0.35);
            s.albedo = spec_mix(spec(0.78 - 0.15 * seam), spec_axpy(spec_ramp(560.0, 610.0), 0.6, spec(0.1)), band);
            s.rough = 0.45;
        }
        case MAT_METAL: {
            s.albedo = spec(0.0);
            s.f0 = spec(0.6);
            s.rough = 0.32;
        }
        case MAT_RADIATOR: {
            s.albedo = spec(0.07);
            s.rough = 0.7;
            // Hot when the drive runs: the radiators shed its waste heat.
            s.emission = spec_scale(spec_planck(700.0 + 500.0 * power), 0.9 * power);
        }
        case MAT_BELL: {
            s.albedo = spec(0.12);
            s.f0 = spec(0.3);
            s.rough = 0.45;
        }
        case MAT_BELL_INNER: {
            s.albedo = spec(0.05);
            // Lit by the plasma: a very hot, diluted blackbody.
            s.emission = spec_scale(spec_planck(9000.0), 2.0e-3 * power);
        }
        case MAT_FOIL: {
            s.albedo = spec(0.0);
            s.f0 = ship_gold();
            // Crinkled film: roughness varies over the surface.
            s.rough = 0.2 + 0.25 * value_noise(p * 2.3);
        }
        case MAT_LIGHT: {
            s.albedo = spec(0.1);
            // Port red, starboard green, blinking once a second.
            let on = step(0.85, fract(ship.glow.y));
            let col = spec_mix(spec_bump(530.0, 25.0), spec_bump(640.0, 20.0), step(0.0, p.y));
            s.emission = spec_scale(col, 400.0 * on);
        }
        default: {
            s.albedo = spec(0.035);
            s.rough = 0.3;
        }
    }
    return s;
}

fn ship_light(s: ShipSurface, n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, e: Spectrum) -> Spectrum {
    let nl = dot(n, l);
    if (nl <= 0.0) {
        return spec(0.0);
    }
    let spec_term = spec_scale(ship_schlick(s.f0, dot(normalize(v + l), v)), ship_ggx(n, v, l, s.rough));
    let brdf = spec_add(spec_scale(s.albedo, 1.0 / PI), spec_term);
    return spec_scale(spec_mul(brdf, e), nl);
}

// ---------------------------------------------------------------------------
// RCS plumes. A firing nozzle blows a jet that spreads at about 25° and
// thins as it widens (density ∝ 1/width², the flow being conserved). The
// gas itself is nearly invisible; droplets of unburnt propellant scatter
// light, sunlight forward more than back (Henyey–Greenstein, g = 0.6),
// and where the puff is thick, light scattered many times makes it look
// white from any side, like a cloud (an even 0.2 per steradian on top).
// The gas glows faintly right at the nozzle where it is still hot. Each jet
// is sampled across where the ray passes closest to its axis, over a
// window a few widths wide, in front of whatever the ray hits.
// ---------------------------------------------------------------------------

const JET_W0: f32 = 0.07;       // core width at the exit, m
const JET_SPREAD: f32 = 0.3;    // widening per metre along the jet
const JET_K: f32 = 60.0;        // extinction at the core of a full jet at the exit, per m

struct Plume {
    L: Spectrum,
    T: f32,
}

fn ship_plumes(o: vec3<f32>, d: vec3<f32>, t_max: f32) -> Plume {
    var out = Plume(spec(0.0), 1.0);
    let g = 0.6;
    let mu = dot(ship.sun_dir.xyz, d);
    let hg = 0.5 * (1.0 - g * g) / (4.0 * PI * pow(1.0 + g * g - 2.0 * g * mu, 1.5)) + 0.2;
    var src = spec_scale(spec_add(ship.sky, ship.planet), 1.0 / (4.0 * PI));
    if (ship.sun_dir.w > 0.5) {
        src = spec_axpy(ship.sun, hg, src);
    }
    let hot = spec_planck(2400.0);
    let count = min(u32(ship.rcs.x + 0.5), 32u);
    for (var j = 0u; j < count; j++) {
        let a = ship.jets[2u * j];
        let b = ship.jets[2u * j + 1u];
        let p = a.xyz;
        let strength = a.w;
        let ax = b.xyz;
        let len = 1.5 + 4.5 * strength;
        // Closest approach of the ray to the jet's axis.
        let w0 = o - p;
        let cb = dot(d, ax);
        let sin2 = max(1.0 - cb * cb, 1e-4);
        let dd = dot(d, w0);
        let e = dot(ax, w0);
        let s_axis = clamp((e - cb * dd) / sin2, 0.0, len);
        let t_c = dot(p + ax * s_axis - o, d);
        let width = JET_W0 + JET_SPREAD * s_axis;
        let miss = length(o + d * t_c - (p + ax * s_axis));
        if (miss > 3.0 * width + 0.2) {
            continue;
        }
        let half = min(3.0 * width / sqrt(sin2), len);
        let t0 = max(t_c - half, 0.0);
        let t1 = min(t_c + half, t_max);
        if (t1 <= t0) {
            continue;
        }
        let steps = 10;
        let dt = (t1 - t0) / f32(steps);
        for (var i = 0; i < steps; i++) {
            let x = o + d * (t0 + (f32(i) + 0.5) * dt) - p;
            let along = dot(x, ax);
            if (along < 0.0 || along > len) {
                continue;
            }
            let r2 = max(dot(x, x) - along * along, 0.0);
            let w = JET_W0 + JET_SPREAD * along;
            let fade = 1.0 - smoothstep(0.4 * len, len, along);
            let flicker = 0.55 + 0.9 * value_noise(vec3<f32>(along * 5.0 - ship.glow.y * 40.0, sqrt(r2) * 8.0, b.w * 3.7));
            let k = JET_K * strength * (JET_W0 / w) * (JET_W0 / w) * exp(-r2 / (w * w)) * fade * flicker;
            let dtau = k * dt;
            let glow = spec_scale(hot, 0.003 * exp(-along / 0.12));
            out.L = spec_axpy(spec_add(src, glow), out.T * dtau, out.L);
            out.T *= exp(-dtau);
        }
    }
    return out;
}

fn ship_trace(n: vec3<f32>) -> ShipHit {
    var h: ShipHit;
    h.hit = false;
    h.L = spec(0.0);
    h.plume = spec(0.0);
    h.plume_t = 1.0;
    if (ship.cam_pos.w < 0.5) {
        return h;
    }
    let o = ship.cam_pos.xyz;
    let d = normalize(n.x * ship.cam_x.xyz + n.y * ship.cam_y.xyz + n.z * ship.cam_z.xyz);
    let hull = ship_hull(o, d);
    h.hit = hull.hit;
    h.L = hull.L;
    if (ship.rcs.x > 0.5) {
        let plume = ship_plumes(o, d, select(1e9, hull.t, hull.hit));
        h.plume = plume.L;
        h.plume_t = plume.T;
    }
    return h;
}

fn ship_hull(o: vec3<f32>, d: vec3<f32>) -> ShipHull {
    var h = ShipHull(false, 1e9, spec(0.0));
    // Bounding sphere.
    let oc = o - ship.bound.xyz;
    let b = dot(oc, d);
    let c = dot(oc, oc) - ship.bound.w * ship.bound.w;
    if (b > 0.0 && c > 0.0 || b * b - c < 0.0) {
        return h;
    }
    let r = ship_bvh(o, d, 1e9, false);
    if (r.tri == 0xffffffffu) {
        return h;
    }
    let i0 = ship_triangles[3u * r.tri];
    let i1 = ship_triangles[3u * r.tri + 1u];
    let i2 = ship_triangles[3u * r.tri + 2u];
    let w = vec3<f32>(1.0 - r.uv.x - r.uv.y, r.uv.x, r.uv.y);
    let a0 = ship_vertices[2u * i0 + 1u];
    let a1 = ship_vertices[2u * i1 + 1u];
    let a2 = ship_vertices[2u * i2 + 1u];
    var nrm = normalize(a0.xyz * w.x + a1.xyz * w.y + a2.xyz * w.z);
    if (dot(nrm, d) > 0.0) {
        nrm = -nrm;
    }
    let ao = ship_vertices[2u * i0].w * w.x + ship_vertices[2u * i1].w * w.y + ship_vertices[2u * i2].w * w.z;
    let mat = bitcast<u32>(a0.w);
    let p = o + r.t * d;
    let v = -d;
    let s = ship_surface(mat, p);

    var l = s.emission;
    if (ship.sun_dir.w > 0.5) {
        let shadow = ship_bvh(p + nrm * 0.02, ship.sun_dir.xyz, 1e9, true);
        if (shadow.tri == 0xffffffffu) {
            l = spec_add(l, ship_light(s, nrm, v, ship.sun_dir.xyz, ship.sun));
        }
    }
    if (ship.planet_dir.w > 0.5) {
        // The planet is a broad source: soften by wrapping the cosine.
        let wrap = max(dot(nrm, ship.planet_dir.xyz) * 0.7 + 0.3, 0.0);
        l = spec_add(l, spec_scale(spec_mul(spec_add(s.albedo, s.f0), ship.planet), wrap * ao / PI));
    }
    l = spec_add(l, spec_scale(spec_mul(spec_add(s.albedo, spec_scale(s.f0, 0.5)), ship.sky), ao / PI));
    h.hit = true;
    h.t = r.t;
    h.L = l;
    return h;
}
