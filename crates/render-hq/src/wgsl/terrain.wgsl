// ---------------------------------------------------------------------------
// Procedural terrain of solid planets: pure functions of a planet's
// parameters and a body-fixed unit direction, shared by the trace pass and
// the surface-map generator.
//
// The height field is split by wavelength at `TerrainParams.split` (about
// R/160, 40 km on an Earth): the *macro* part (continents, plates, basins,
// big craters, climate channels) is what a surface map stores; the *detail*
// part (smaller mountains, craters, dunes, crevasses) is always evaluated
// here, down to the pixel footprint. Heights are km above the planet's
// datum radius R, which is sea level on worlds with seas (water, magma or
// flooded mare plains fill everything below 0).
//
// Noise is lattice gradient noise with an integer hash (no textures), so
// every planet is determined by its seed alone.
// ---------------------------------------------------------------------------

// Values of `Planet.ids.x` (and the probe's `kind`), matching `kerr::planets::PlanetKind`.
const KIND_ROCKY: u32 = 0u;
const KIND_OCEAN: u32 = 1u;
const KIND_DESERT: u32 = 2u;
const KIND_ICE: u32 = 3u;
const KIND_LAVA: u32 = 4u;
const KIND_GAS_GIANT: u32 = 5u;
const KIND_ICE_GIANT: u32 = 6u;

struct TerrainParams {
    kind: u32,
    seed: u32,
    radius: f32,   // km
    relief: f32,   // peak-to-trough, km
    sea: f32,      // fraction of the relief below the datum
    t_eq: f32,     // equilibrium temperature, K
    air: f32,      // 1 with an atmosphere
    split: f32,    // macro/detail wavelength split, km
    lo: f32,       // lowest possible height, km
    hi: f32,       // highest possible height, km
    liquid: u32,   // what fills basins below 0: 0 nothing, 1 water, 2 magma, 3 solid basalt (maria)
    threshold: f32,  // continent threshold (ocean worlds)
}

const FILL_NONE: u32 = 0u;
const FILL_WATER: u32 = 1u;
const FILL_MAGMA: u32 = 2u;
const FILL_BASALT: u32 = 3u;

// Octave-to-octave rotation (orthonormal, scaled by 2 in use): breaks up
// the lattice's axis alignment.
const TN_ROT = mat3x3<f32>(
    vec3<f32>(0.00, 0.80, 0.60),
    vec3<f32>(-0.80, 0.36, -0.48),
    vec3<f32>(-0.60, -0.48, 0.64),
);

fn tn_hash(v: u32) -> u32 {
    var x = v;
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}

fn tn_unit(h: u32) -> f32 {
    return f32(h >> 8u) * (1.0 / 16777216.0);
}

fn tn_grad(h: u32, f: vec3<f32>) -> f32 {
    let g = vec3<f32>(vec3<u32>(h, h >> 10u, h >> 20u) & vec3<u32>(1023u)) * (2.0 / 1023.0) - 1.0;
    return dot(g, f);
}

// Gradient noise, roughly in [−1, 1] (standard deviation ≈ 0.2).
fn tn_noise(p: vec3<f32>, seed: u32) -> f32 {
    let i = floor(p);
    let f = p - i;
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let c = bitcast<vec3<u32>>(vec3<i32>(i));
    let x0 = c.x * 0x8da6b343u;
    let x1 = x0 + 0x8da6b343u;
    let y0 = c.y * 0xd8163841u;
    let y1 = y0 + 0xd8163841u;
    let z0 = c.z * 0xcb1ab31fu + seed * 0x9e3779b9u;
    let z1 = z0 + 0xcb1ab31fu;
    let n000 = tn_grad(tn_hash(x0 + y0 + z0), f);
    let n100 = tn_grad(tn_hash(x1 + y0 + z0), f - vec3<f32>(1.0, 0.0, 0.0));
    let n010 = tn_grad(tn_hash(x0 + y1 + z0), f - vec3<f32>(0.0, 1.0, 0.0));
    let n110 = tn_grad(tn_hash(x1 + y1 + z0), f - vec3<f32>(1.0, 1.0, 0.0));
    let n001 = tn_grad(tn_hash(x0 + y0 + z1), f - vec3<f32>(0.0, 0.0, 1.0));
    let n101 = tn_grad(tn_hash(x1 + y0 + z1), f - vec3<f32>(1.0, 0.0, 1.0));
    let n011 = tn_grad(tn_hash(x0 + y1 + z1), f - vec3<f32>(0.0, 1.0, 1.0));
    let n111 = tn_grad(tn_hash(x1 + y1 + z1), f - vec3<f32>(1.0, 1.0, 1.0));
    return mix(
        mix(mix(n000, n100, u.x), mix(n010, n110, u.x), u.y),
        mix(mix(n001, n101, u.x), mix(n011, n111, u.x), u.y),
        u.z,
    ) * 1.6;
}

// Fractional Brownian motion with `octaves` octaves (the last one faded
// in by its fractional part, so level of detail changes continuously).
fn tn_fbm(p: vec3<f32>, seed: u32, octaves: f32, gain: f32) -> f32 {
    var s = 0.0;
    var a = 1.0;
    var q = p;
    let n = i32(ceil(max(octaves, 0.0)));
    for (var i = 0; i < n; i++) {
        let w = clamp(octaves - f32(i), 0.0, 1.0);
        s += a * w * tn_noise(q, seed + u32(i));
        q = TN_ROT * q * 2.02;
        a *= gain;
    }
    return s;
}

// Ridged multifractal (Musgrave): sharp crests, each octave weighted by
// the one before so detail gathers on ridges and valleys stay smooth.
// Returns about [0, 1.5].
fn tn_ridged(p: vec3<f32>, seed: u32, octaves: f32, gain: f32) -> f32 {
    var s = 0.0;
    var a = 1.0;
    var w = 1.0;
    var q = p;
    let n = i32(ceil(max(octaves, 0.0)));
    for (var i = 0; i < n; i++) {
        let fade = clamp(octaves - f32(i), 0.0, 1.0);
        var r = 1.0 - abs(tn_noise(q, seed + 17u * u32(i)));
        r = r * r;
        s += a * r * w * fade;
        w = clamp(r * 1.6, 0.0, 1.0);
        q = TN_ROT * q * 2.03;
        a *= gain;
    }
    return s;
}

fn tn_warp(p: vec3<f32>, seed: u32, octaves: f32) -> vec3<f32> {
    return vec3<f32>(
        tn_fbm(p, seed, octaves, 0.5),
        tn_fbm(p + vec3<f32>(5.2, 1.3, 7.1), seed + 101u, octaves, 0.5),
        tn_fbm(p + vec3<f32>(2.7, 9.2, 4.4), seed + 202u, octaves, 0.5),
    );
}

// Octaves of a series starting at wavelength `lambda0` (km) down to
// `lambda_min` (km), keeping each octave's wavelength ≥ 2 footprints.
fn tn_octaves(lambda0: f32, lambda_min: f32) -> f32 {
    return clamp(log2(lambda0 / max(lambda_min, 1e-3)), 0.0, 14.0);
}

// Inverse of the standard normal CDF (Acklam-free logistic fit, ±0.02
// in z): sets thresholds that flood a given fraction of a field.
fn tn_probit(p: f32) -> f32 {
    let q = clamp(p, 0.001, 0.999);
    return log(q / (1.0 - q)) / 1.702;
}

fn terrain_params(kind: u32, seed: u32, radius: f32, relief: f32, sea: f32, t_eq: f32, air: bool) -> TerrainParams {
    var tp: TerrainParams;
    tp.kind = kind;
    tp.seed = tn_hash(seed ^ 0x51ed270bu);
    tp.radius = radius;
    tp.relief = max(relief, 0.0);
    tp.sea = sea;
    tp.t_eq = t_eq;
    tp.air = select(0.0, 1.0, air);
    tp.split = radius / 160.0;
    tp.lo = -sea * tp.relief;
    tp.hi = (1.0 - sea) * tp.relief;
    tp.liquid = FILL_NONE;
    if (sea > 0.0) {
        switch (kind) {
            case KIND_OCEAN: { tp.liquid = FILL_WATER; }
            case KIND_LAVA: { tp.liquid = FILL_MAGMA; }
            default: {
                // Liquid water needs air and a temperate climate; otherwise
                // the low plains are old lava floods (maria).
                let temperate = air && t_eq > 230.0 && t_eq < 330.0;
                tp.liquid = select(FILL_BASALT, FILL_WATER, temperate);
            }
        }
    }
    // Ocean area grows with the sea level: 0.55 → 66 %, 0.75 → 81 % (Earth 71 %).
    tp.threshold = 0.22 * tn_probit(0.25 + 0.75 * sea);
    return tp;
}

fn tn_rand3(h: u32) -> vec3<f32> {
    let a = tn_hash(h);
    let b = tn_hash(a ^ 0x68bc21ebu);
    let c = tn_hash(b ^ 0x02e5be93u);
    return vec3<f32>(tn_unit(a), tn_unit(b), tn_unit(c));
}

// ---------------------------------------------------------------------------
// Impact craters: one candidate per cell of a 3D lattice of spacing `s`
// (km), centred within the middle of the cell so its bowl and ejecta
// never cross into a neighbour (one lookup per size class). The sphere of
// influence cuts the surface in a circle of random size, and equal
// probability per cell over halving sizes gives the observed N(>D) ∝ D⁻².
// Shapes follow Pike (1977): simple bowls (depth ≈ D/5) below ~15 km,
// shallower complex craters (depth ≈ 1.04 D^0.3 km) with central peaks
// above; rims are ~1/4 of the depth; old craters are softened and filled.
// Returns (height, freshness of ejecta at this point).
// ---------------------------------------------------------------------------
fn tn_crater(x: vec3<f32>, s: f32, seed: u32, prob: f32, age_bias: f32) -> vec2<f32> {
    let cell = floor(x / s);
    let c = bitcast<vec3<u32>>(vec3<i32>(cell));
    let h = tn_hash(c.x * 0x8da6b343u + c.y * 0xd8163841u + c.z * 0xcb1ab31fu + seed * 0x9e3779b9u);
    if (tn_unit(h) > prob) {
        return vec2<f32>(0.0);
    }
    let r3 = tn_rand3(h);
    let centre = (cell + 0.35 + 0.3 * r3) * s;
    let rad = s * (0.07 + 0.1 * r3.x * r3.y);
    let d = length(x - centre);
    let rho = d / rad;
    if (rho > 2.2) {
        return vec2<f32>(0.0);
    }
    let diam = 2.0 * rad;
    let age = clamp(r3.z + age_bias, 0.0, 1.0);
    var depth = min(0.2 * diam, 1.04 * pow(diam, 0.3)) * (1.0 - 0.75 * age);
    let rim = 0.25 * depth * (1.0 - 0.6 * age);
    // Bowl (flattened for complex craters), rim crest, ejecta ∝ ρ⁻³.
    let complex_w = smoothstep(10.0, 30.0, diam);
    let bowl = mix(rho * rho - 1.0, smoothstep(0.35, 1.0, rho) - 1.0, complex_w);
    var z = 0.0;
    if (rho < 1.0) {
        z = depth * bowl + rim * smoothstep(0.75, 1.0, rho);
        z += complex_w * 0.5 * depth * exp(-rho * rho / 0.02);
    } else {
        let e = rim * pow(rho, -3.0);
        z = e * (1.0 - smoothstep(1.6, 2.2, rho));
    }
    let fresh = (1.0 - age) * (1.0 - smoothstep(1.0, 2.2, rho)) * step(0.55, 1.0 - age);
    return vec2<f32>(z, fresh);
}

// Sum of crater size classes with cell sizes from `s_max` down to `s_min`
// (km); the smallest class fades in with the footprint.
fn tn_craters(x: vec3<f32>, s_max: f32, s_min: f32, seed: u32, prob: f32, age_bias: f32) -> vec2<f32> {
    var out = vec2<f32>(0.0);
    var s = s_max;
    for (var i = 0u; i < 12u; i++) {
        if (s < s_min) {
            break;
        }
        let fade = clamp(s / s_min - 1.0, 0.0, 1.0);
        let c = tn_crater(x, s, seed + 31u * i, prob, age_bias);
        out = vec2<f32>(out.x + fade * c.x, max(out.y, fade * c.y));
        s *= 0.5;
    }
    return out;
}

// ---------------------------------------------------------------------------
// Tectonics, lite: 14–23 plates as a spherical Voronoi diagram of jittered
// Fibonacci points (in a warped direction, so boundaries are ragged), each
// rotating about its own Euler pole. The relative velocity across the
// nearest boundary tells convergent (+, mountain belts, trenches, island
// arcs) from divergent (−, mid-ocean ridges and rifts).
// Returns (angular distance to the boundary, convergence −1…1, plate hash).
// ---------------------------------------------------------------------------
fn tn_plates(q: vec3<f32>, seed: u32) -> vec3<f32> {
    let n = 14u + seed % 10u;
    var best = -2.0;
    var second = -2.0;
    var i1 = 0u;
    var i2 = 0u;
    var s1 = vec3<f32>(0.0, 0.0, 1.0);
    var s2 = vec3<f32>(0.0, 0.0, 1.0);
    let fn_ = f32(n);
    for (var i = 0u; i < n; i++) {
        let r = tn_rand3(seed * 7919u + i);
        let z = 1.0 - (f32(i) + 0.2 + 0.6 * r.x) * 2.0 / fn_;
        let phi = f32(i) * 2.39996 + 6.2831 * r.y * 0.15 + f32(seed % 97u);
        let rr = sqrt(max(1.0 - z * z, 0.0));
        let s = vec3<f32>(rr * cos(phi), rr * sin(phi), z);
        let d = dot(q, s);
        if (d > best) {
            second = best;
            s2 = s1;
            i2 = i1;
            best = d;
            s1 = s;
            i1 = i;
        } else if (d > second) {
            second = d;
            s2 = s;
            i2 = i;
        }
    }
    let axis = s2 - s1;
    let la = max(length(axis), 1e-6);
    // Distance to the bisecting plane through the centre (the boundary).
    let dist = abs(dot(q, axis)) / la;
    // Euler vectors: random axes, 0.5–1 in arbitrary units.
    let w1 = (tn_rand3(seed ^ (i1 * 0x51u + 7u)) * 2.0 - 1.0);
    let w2 = (tn_rand3(seed ^ (i2 * 0x51u + 7u)) * 2.0 - 1.0);
    let v_rel = cross(w1 - w2, q);
    let nb = normalize(axis - dot(axis, q) * q + vec3<f32>(1e-7));
    let conv = clamp(dot(v_rel, nb), -1.0, 1.0);
    return vec3<f32>(dist, conv, f32(tn_hash(seed + i1) >> 8u) / 16777216.0);
}

// Terraces (mesas, benches): flat treads and steep risers of height `step`.
fn tn_terrace(h: f32, step_h: f32, sharp: f32) -> f32 {
    let k = h / step_h;
    let f = floor(k);
    let t = k - f;
    return (f + smoothstep(0.5 - 0.5 / sharp, 0.5 + 0.5 / sharp, t)) * step_h;
}

// ---------------------------------------------------------------------------
// Macro terrain: vec4(height km, a, b, c) with kind-specific channels
//   ocean:  orogeny, moisture, rock (0 granite … 1 basalt)
//   rocky:  ejecta freshness, highland brightness, 0
//   desert: sand, playa, strata/ferric variation
//   ice:    lineae stain, chaos, 0
//   lava:   heat, flows, 0
// `lod` (km) limits the wavelengths to what a pixel resolves.
// ---------------------------------------------------------------------------
fn terrain_macro(tp: TerrainParams, q: vec3<f32>, lod: f32) -> vec4<f32> {
    switch (tp.kind) {
        case KIND_OCEAN: { return tn_macro_ocean(tp, q, lod); }
        case KIND_DESERT: { return tn_macro_desert(tp, q, lod); }
        case KIND_ICE: { return tn_macro_ice(tp, q, lod); }
        case KIND_LAVA: { return tn_macro_lava(tp, q, lod); }
        default: { return tn_macro_rocky(tp, q, lod); }
    }
}

// Octaves from a base frequency f0 (cycles per radius) down to the split.
fn tn_macro_octaves(tp: TerrainParams, f0: f32, lod: f32) -> f32 {
    return tn_octaves(tp.radius / f0, max(2.0 * lod, tp.split));
}

fn tn_latitude(q: vec3<f32>) -> f32 {
    return asin(clamp(q.z, -1.0, 1.0));
}

fn tn_macro_ocean(tp: TerrainParams, q: vec3<f32>, lod: f32) -> vec4<f32> {
    let k = tp.relief / 16.0;
    let seed = tp.seed;
    let warp = tn_warp(q * 1.3, seed + 1u, min(3.0, tn_macro_octaves(tp, 1.3, lod))) * 0.28;
    let qc = q + warp;
    let c = tn_fbm(qc * 1.7, seed + 2u, tn_macro_octaves(tp, 1.7, lod), 0.5);
    let e = c - tp.threshold;

    let pl = tn_plates(normalize(q + 0.35 * warp), seed + 3u);
    let belt = exp(-pl.x * pl.x / (0.03 * 0.03));
    let conv = max(pl.y, 0.0);
    let div = max(-pl.y, 0.0);
    // Mountain belts along convergent boundaries, and a weaker background
    // of old, eroded ranges.
    let old = smoothstep(0.55, 1.1, tn_ridged(qc * 3.0, seed + 4u, min(2.0, tn_macro_octaves(tp, 3.0, lod)), 0.5));
    let orog = clamp(conv * belt * 2.0 + 0.5 * old, 0.0, 1.0);
    let ridge_oct = tn_macro_octaves(tp, 12.0, lod);
    let ridges = tn_ridged(qc * 12.0, seed + 5u, ridge_oct, 0.5);

    let land_w = smoothstep(-0.012, 0.012, e);
    let land = k * (0.07 * smoothstep(0.0, 0.015, e) + 1.5 * smoothstep(0.0, 0.45, e) - 0.02);
    // Shelf to ~130 m, slope, abyssal plain at ~4.5 km (Earth's hypsometry).
    let sea = -k * (0.02 + 0.11 * smoothstep(0.0, -0.025, e) + 4.3 * smoothstep(-0.03, -0.14, e));
    let mid_ridge = div * exp(-pl.x * pl.x / (0.018 * 0.018)) * k * 2.0 * smoothstep(-0.05, -0.15, e);
    let trench = conv * exp(-pow((pl.x - 0.02) / 0.008, 2.0)) * k * 3.5 * smoothstep(-0.03, -0.12, e);
    // Fold belts: ridges parallel to the plate boundary (~25 km apart,
    // bent by noise), mixed with ridged noise for the crests.
    let fold_phase = pl.x / 0.0037 + 2.0 * tn_noise(qc * 20.0, seed + 11u);
    let fold = 1.0 - abs(sin(fold_phase * PI));
    let fold_w = 1.0 - smoothstep(0.5, 1.0, 2.0 * lod / (0.0037 * tp.radius));
    let mountains = orog * k * mix(2.5, 5.5, land_w) * (0.3 + 0.4 * ridges + 0.35 * fold * fold * belt * fold_w);
    // Uplands and plains: hills where a low-frequency field is high.
    let upland = smoothstep(-0.15, 0.35, tn_fbm(qc * 3.0 + vec3<f32>(7.7), seed + 12u, 2.0, 0.5));
    let hills = k * mix(0.25, 1.2, upland) * tn_fbm(qc * 8.0, seed + 6u, tn_macro_octaves(tp, 8.0, lod), 0.5);
    var h = mix(sea + mid_ridge - trench, land, land_w) + mountains + hills * mix(0.3, 1.0, land_w);
    let orog_out = clamp(orog + 0.35 * upland, 0.0, 1.0);
    // Hotspot island chains rise from the sea floor.
    let hot = tn_hotspots(tp, q);
    h += hot.x;

    // Hadley, Ferrel and polar cells: wet at the equator and near 60°,
    // dry near 30° and at the poles; drier far inland.
    let lat = tn_latitude(q);
    let cells = 0.5 + 0.5 * cos(6.0 * lat);
    let moist = clamp(0.25 + 0.6 * cells - 1.8 * max(e - 0.08, 0.0) + 0.5 * tn_fbm(qc * 5.0, seed + 7u, 2.0, 0.5), 0.0, 1.0);
    let rock = smoothstep(0.1, 0.35, tn_fbm(qc * 4.0, seed + 8u, 2.0, 0.5) + 0.25 * conv * belt);
    // Volcanic islands are basalt.
    return vec4<f32>(h, orog_out, moist, max(rock, hot.y));
}

// ---------------------------------------------------------------------------
// Hotspot island chains (Hawaii, the Societies, the Marquesas): a mantle
// plume under a moving plate builds a line of shield volcanoes, the
// youngest over the plume, each older one farther along the plate's
// motion. Shields rise ~10 km from the sea floor (Mauna Kea stands 4.2 km
// above the sea on a 5 km deep floor) with slopes of a few degrees,
// stretched along rift zones; with age they sink (the lithosphere cools),
// erode into radial valleys and drown, leaving reefs.
// Returns (height to add, km; basalt share 0–1; reef 0–1).
// ---------------------------------------------------------------------------
fn tn_hotspots(tp: TerrainParams, q: vec3<f32>) -> vec3<f32> {
    let n = 3u + tp.seed % 4u;
    var add = 0.0;
    var basalt = 0.0;
    var reef = 0.0;
    for (var i = 0u; i < n; i++) {
        let r = tn_rand3(tp.seed * 131u + i * 7u + 3u);
        let z = r.x * 1.4 - 0.7;
        let phi = r.y * TAU;
        let spot = vec3<f32>(sqrt(1.0 - z * z) * cos(phi), sqrt(1.0 - z * z) * sin(phi), z);
        // Islands every ~130 km along the chain, 8 of them.
        let spacing = (110.0 + 50.0 * r.z) / tp.radius;
        // Skip chains far from q (the whole chain lies within 9 steps).
        if (dot(q, spot) < cos(9.0 * spacing + 0.03)) {
            continue;
        }
        let side = tn_rand3(tp.seed * 977u + i) * 2.0 - 1.0;
        let along = normalize(side - dot(side, spot) * spot);
        for (var k = 0u; k < 8u; k++) {
            let rk = tn_rand3(tp.seed * 313u + i * 17u + k);
            // Older islands lie further along −along, a little off line.
            let a = f32(k) * spacing * (0.8 + 0.4 * rk.x);
            let off = (rk.y - 0.5) * 0.6 * spacing;
            let across = cross(spot, along);
            let c = normalize(spot * cos(a) - along * sin(a) + across * off);
            let d_ang = acos(clamp(dot(q, c), -1.0, 1.0));
            let d = d_ang * tp.radius;
            if (d > 220.0) {
                continue;
            }
            let age = f32(k) * (0.9 + 0.5 * rk.z);  // million years
            // Rift zones stretch the shield along a random direction.
            let rift = normalize(cross(c, tn_rand3(tp.seed * 71u + i * 13u + k) * 2.0 - 1.0));
            let t = q - c * dot(q, c);
            let along_rift = dot(t, rift) * tp.radius;
            let across_rift = dot(t, cross(c, rift)) * tp.radius;
            let w = 30.0 + 10.0 * rk.x + 3.0 * age;
            let rho = sqrt(pow(along_rift / (1.6 * w), 2.0) + pow(across_rift / w, 2.0));
            // Amplitude above the floor: 10 km young, sinking ~0.35 km/Myr
            // and wearing down.
            let amp = max(10.0 * exp(-age / 9.0) - 0.35 * age, 0.0);
            var shield = amp * exp(-pow(rho, 1.4));
            // Radial valleys cut into the older, rain-washed flanks.
            if (age > 1.0) {
                let bearing = atan2(across_rift, along_rift);
                let n_valleys = 14.0 + 6.0 * rk.y;
                let v = pow(0.5 + 0.5 * cos(n_valleys * bearing + 3.0 * tn_noise(q * tp.radius / 8.0, tp.seed + k)), 6.0);
                shield *= 1.0 - min(0.08 * age, 0.5) * v * smoothstep(0.15, 0.6, rho) * smoothstep(1.8, 0.8, rho);
            }
            // A caldera on the young summits.
            shield -= 0.15 * amp * exp(-pow(rho / 0.08, 2.0)) * step(age, 2.0);
            add = max(add, shield);
            basalt = max(basalt, smoothstep(1.6, 0.8, rho) * step(0.5, amp));
            // Fringing reef and atoll rings where the flank meets the sea,
            // in the tropics.
            let tropical = smoothstep(0.55, 0.4, abs(q.z));
            reef = max(reef, tropical * exp(-pow((rho - 1.1 - 0.05 * age) / 0.12, 2.0)) * smoothstep(0.5, 3.0, age));
        }
    }
    return vec3<f32>(add, basalt, reef);
}

fn tn_macro_rocky(tp: TerrainParams, q: vec3<f32>, lod: f32) -> vec4<f32> {
    let seed = tp.seed;
    let warp = tn_warp(q * 1.1, seed + 1u, min(2.0, tn_macro_octaves(tp, 1.1, lod))) * 0.3;
    let qc = q + warp;
    // Highland/lowland dichotomy plus rolling terrain.
    let b = tn_fbm(qc * 1.3, seed + 2u, tn_macro_octaves(tp, 1.3, lod), 0.55);
    // With a fill level, the lowest ~1.5 × `sea` of the area is flooded
    // (maria or seas); otherwise the terrain sits in the middle of its range.
    var h = 0.5 * (tp.lo + tp.hi) + tp.relief * 0.45 * b;
    if (tp.sea > 0.0) {
        h = tp.relief * 0.45 * (b - 0.3 * tn_probit(min(1.5 * tp.sea, 0.6)));
    }
    // Big basins and craters; fewer where an atmosphere erodes them.
    let x = q * tp.radius;
    let prob = mix(0.9, 0.45, tp.air);
    let cr = tn_craters(x, tp.radius * 0.6, max(4.0 * tp.split, 4.0 * lod), seed + 9u, prob, 0.15 + 0.3 * tp.air);
    h += cr.x;
    let bright = tn_fbm(qc * 3.0, seed + 3u, 2.0, 0.5);
    return vec4<f32>(h, cr.y, bright, 0.0);
}

fn tn_macro_desert(tp: TerrainParams, q: vec3<f32>, lod: f32) -> vec4<f32> {
    let seed = tp.seed;
    let warp = tn_warp(q * 1.4, seed + 1u, min(3.0, tn_macro_octaves(tp, 1.4, lod))) * 0.3;
    let qc = q + warp;
    let b = tn_fbm(qc * 1.5, seed + 2u, tn_macro_octaves(tp, 1.5, lod), 0.5);
    var h = tp.relief * (0.12 + 0.5 * smoothstep(-0.45, 0.5, b));
    // Plateaus cut into mesas and benches.
    let plateau = smoothstep(0.05, 0.2, b);
    h = mix(h, tn_terrace(h, 0.09 * tp.relief, 4.0), plateau);
    // Dry basins: flat playa floors.
    let basin = smoothstep(-0.15, -0.35, b);
    h = mix(h, tp.relief * 0.1, basin * 0.85);
    // A few shield volcanoes (Tharsis-like); height ∝ 1/g would make them
    // huge on small worlds, so they are capped by the relief.
    for (var i = 0u; i < 3u; i++) {
        let r = tn_rand3(seed * 13u + i);
        let z = r.x * 1.4 - 0.7;
        let phi = r.y * 6.2831;
        let c = vec3<f32>(sqrt(1.0 - z * z) * cos(phi), sqrt(1.0 - z * z) * sin(phi), z);
        let d = acos(clamp(dot(q, c), -1.0, 1.0)) / (0.04 + 0.05 * r.z);
        h += tp.relief * 0.5 * r.z * exp(-d * d) * step(0.4, r.z);
    }
    let x = q * tp.radius;
    let cr = tn_craters(x, tp.radius * 0.4, max(4.0 * tp.split, 4.0 * lod), seed + 9u, 0.35, 0.5);
    h += cr.x;
    let sand = smoothstep(0.0, 0.6, tn_fbm(qc * 3.0, seed + 5u, 3.0, 0.5) + 0.3 * basin + 0.2) * (1.0 - plateau * 0.8);
    let strata = tn_fbm(qc * 6.0, seed + 6u, 2.0, 0.5);
    return vec4<f32>(h, sand, basin, strata);
}

fn tn_macro_ice(tp: TerrainParams, q: vec3<f32>, lod: f32) -> vec4<f32> {
    let seed = tp.seed;
    let qc = q + tn_warp(q * 2.0, seed + 1u, min(2.0, tn_macro_octaves(tp, 2.0, lod))) * 0.2;
    // Smooth ice sheets.
    var h = tp.relief * (0.35 + 0.2 * tn_fbm(qc * 1.5, seed + 2u, min(4.0, tn_macro_octaves(tp, 1.5, lod)), 0.45));
    // Lineae: long curving double ridges from cracking of the shell.
    var lin = 0.0;
    var f = 4.0;
    for (var i = 0u; i < 4u; i++) {
        // Narrower lines at higher frequencies, fading out below a pixel.
        let fade = 1.0 - smoothstep(0.3, 1.0, 40.0 * lod * f / tp.radius);
        let l = 1.0 - abs(tn_noise(qc * f + vec3<f32>(3.1 * f32(i)), seed + 3u + i));
        lin = max(lin, pow(l, 40.0) * fade * (1.0 - 0.15 * f32(i)));
        f *= 1.9;
    }
    h += tp.relief * 0.04 * lin;
    // Chaos terrain: disrupted patches.
    let chaos = smoothstep(0.25, 0.4, tn_fbm(qc * 3.0, seed + 5u, 2.0, 0.5));
    let x = q * tp.radius;
    let cr = tn_craters(x, tp.radius * 0.3, max(4.0 * tp.split, 4.0 * lod), seed + 9u, 0.2, 0.3);
    h += cr.x * 0.5;
    return vec4<f32>(h, clamp(lin, 0.0, 1.0), chaos, 0.0);
}

fn tn_macro_lava(tp: TerrainParams, q: vec3<f32>, lod: f32) -> vec4<f32> {
    let seed = tp.seed;
    let qc = q + tn_warp(q * 1.5, seed + 1u, min(3.0, tn_macro_octaves(tp, 1.5, lod))) * 0.3;
    let b = tn_fbm(qc * 1.6, seed + 2u, tn_macro_octaves(tp, 1.6, lod), 0.5);
    let e = b - 0.22 * tn_probit(tp.sea * 0.9 + 0.05);
    var h = mix(tp.lo * 0.6, tp.hi * 0.45, smoothstep(-0.3, 0.4, e));
    // Shield volcanoes.
    var heat = 0.0;
    for (var i = 0u; i < 4u; i++) {
        let r = tn_rand3(seed * 17u + i);
        let z = r.x * 1.8 - 0.9;
        let phi = r.y * 6.2831;
        let c = vec3<f32>(sqrt(1.0 - z * z) * cos(phi), sqrt(1.0 - z * z) * sin(phi), z);
        let d = acos(clamp(dot(q, c), -1.0, 1.0)) / (0.03 + 0.04 * r.z);
        let cone = exp(-d);
        h += tp.hi * 0.5 * cone * (1.0 - 0.6 * exp(-d * d * 30.0));
        heat = max(heat, exp(-d * d * 20.0));
    }
    let flows = tn_ridged(qc * 10.0, seed + 3u, min(3.0, tn_macro_octaves(tp, 10.0, lod)), 0.5);
    h += tp.relief * 0.05 * flows;
    heat = max(heat, smoothstep(0.02, -0.02, e) * 0.6);
    return vec4<f32>(h, heat, flows, 0.0);
}

// ---------------------------------------------------------------------------
// Detail below the split wavelength, on top of macro `m`, down to `lod`.
// ---------------------------------------------------------------------------
fn terrain_detail(tp: TerrainParams, q: vec3<f32>, m: vec4<f32>, lod: f32) -> f32 {
    let lambda_min = max(2.0 * lod, 0.012);
    if (lambda_min >= tp.split) {
        return 0.0;
    }
    let oct = tn_octaves(tp.split, lambda_min);
    let p = q * (tp.radius / tp.split);
    let seed = tp.seed + 1000u;
    let x = q * tp.radius;
    switch (tp.kind) {
        case KIND_OCEAN: {
            let k = tp.relief / 16.0;
            // Ridged in mountain belts, gentler rolling elsewhere.
            let orog = m.y;
            // Terrain slopes stay roughly constant from octave to octave
            // (amplitude ∝ wavelength): ~0.1 in uplands, ~0.4 in ranges.
            let rough = mix(0.12, 3.0, orog * orog) * k * smoothstep(-0.3, 0.2, m.x);
            let r = tn_ridged(p, seed, oct, 0.5) - 0.5;
            let f = tn_fbm(p, seed + 50u, oct, 0.5);
            return rough * mix(f, r, orog) + 0.06 * k * f;
        }
        case KIND_DESERT: {
            var h = 0.12 * tp.relief / 6.0 * tn_fbm(p, seed, oct, 0.5);
            // Transverse dunes in sand seas: asymmetric waves (gentle
            // stoss, steep lee) of ~1.5 km wavelength, crests bent by noise.
            if (m.y > 0.05 && lod < 0.6) {
                let wdir = normalize(vec3<f32>(0.8, 0.6, 0.0));
                let bend = tn_noise(x / 8.0, seed + 5u) * 3.0;
                let phase = dot(x, wdir) / 1.5 + bend;
                let t = fract(phase);
                let dune = mix(t / 0.8, (1.0 - t) / 0.2, step(0.8, t));
                let fade = 1.0 - smoothstep(0.25, 0.6, lod);
                h += 0.08 * m.y * dune * fade * (0.6 + 0.4 * tn_noise(x / 3.0, seed + 6u));
            }
            return h;
        }
        case KIND_ICE: {
            // Smooth sheets with crevasse fields and small ridges.
            let cr = tn_ridged(p * 2.0, seed, oct - 1.0, 0.45);
            return tp.relief * (0.004 * tn_fbm(p, seed + 3u, oct, 0.5) + mix(0.002, 0.02, m.z) * (cr - 0.5));
        }
        case KIND_LAVA: {
            return 0.08 * tp.relief / 5.0 * (tn_ridged(p, seed, oct, 0.5) - 0.4);
        }
        default: {
            // Small craters over rolling regolith.
            let prob = mix(0.9, 0.35, tp.air);
            let cr = tn_craters(x, 2.0 * tp.split, 6.0 * lod, seed + 9u, prob, 0.1 + 0.4 * tp.air);
            return cr.x + 0.1 * tp.relief / 12.0 * tn_fbm(p, seed, oct, 0.5);
        }
    }
}

// Solid height (km) at q: macro plus detail, kept inside [lo, hi].
fn terrain_solid(tp: TerrainParams, q: vec3<f32>, m: vec4<f32>, lod: f32) -> f32 {
    return clamp(m.x + terrain_detail(tp, q, m, lod), tp.lo, tp.hi);
}

// Height of the visible surface: the solid ground or, over basins, the
// surface of whatever fills them (at 0).
fn terrain_surface(tp: TerrainParams, solid: f32) -> f32 {
    return select(solid, max(solid, 0.0), tp.liquid != FILL_NONE);
}
