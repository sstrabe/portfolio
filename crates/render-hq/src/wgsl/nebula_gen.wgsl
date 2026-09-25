// ---------------------------------------------------------------------------
// Procedural generation of the nebula volumes, run once at startup (see
// `nebula.rs` for the pass order and `nebula/scene.rs` for the physics
// behind every number passed in).
//
// Each volume is a box of voxels. Positions here are in parsecs relative to
// Sgr A*, in the renderer's coordinates. Final volumes hold four channels:
//   r  ionised emission measure density, (n_e / 10⁴ cm⁻³)² (PWN and SNR:
//      relative, calibrated later)
//   g  gas (dust) density n_H / n_scale; for the pulsar wind nebula the
//      synchrotron emissivity instead
//   b  low-ionisation weight: ionisation fronts and radiative shocks, where
//      [S II], [N II] and [O I] are strong
//   a  the Galactic Centre volumes: transmission of the central cluster's
//      light (for dust scattering); the remnants: high-excitation weight
//      ([O III])
// ---------------------------------------------------------------------------

struct Gen {
    centre: vec4<f32>,      // box centre relative to the hole (pc); w: seed
    ax: vec4<f32>,          // box axes (unit), w: half extent (pc)
    ay: vec4<f32>,
    az: vec4<f32>,
    dims: vec4<u32>,        // size of the texture written; w: first z slice of this dispatch
    p0: vec4<f32>,          // structure parameters (see each entry point)
    p1: vec4<f32>,
    p2: vec4<f32>,
    p3: vec4<f32>,
    occ_centre: vec4<f32>,  // light pass: a second volume that also absorbs (w: its gas scale / ours, 0: none)
    occ_ax: vec4<f32>,      // its axes / half extents
    occ_ay: vec4<f32>,
    occ_az: vec4<f32>,
}

// A node of a gas stream (the minispiral's arms).
struct Node {
    pos: vec4<f32>,  // centre (pc), gas density n_H / n_scale
    tan: vec4<f32>,  // direction of flow, half width in the sheet (pc)
    nrm: vec4<f32>,  // sheet normal, half thickness (pc)
    vel: vec4<f32>,  // velocity (1000 km/s), stream id
}

@group(0) @binding(0) var<uniform> gen: Gen;
@group(0) @binding(1) var out_vol: texture_storage_3d<rgba16float, write>;
@group(0) @binding(2) var out_gas: texture_storage_3d<rg32float, write>;
@group(0) @binding(3) var in_vol: texture_3d<f32>;
@group(0) @binding(4) var in_occ: texture_3d<f32>;
@group(0) @binding(5) var lin: sampler;
@group(0) @binding(6) var<storage, read> nodes: array<Node>;
@group(0) @binding(7) var<storage, read_write> sums: array<vec2<f32>>;
@group(0) @binding(8) var out_detail: texture_storage_3d<rgba8unorm, write>;

// ---------------------------------------------------------------------------
// Noise
// ---------------------------------------------------------------------------

fn pcg3(v0: vec3<u32>) -> vec3<u32> {
    var v = v0 * 1664525u + 1013904223u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v ^= v >> vec3<u32>(16u);
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    return v;
}

// Three uniform numbers in [0, 1) for lattice point c.
fn rand3(c: vec3<i32>, seed: u32) -> vec3<f32> {
    let h = pcg3(bitcast<vec3<u32>>(c) + vec3<u32>(seed * 747796405u, seed * 2891336453u + 1u, seed * 277803737u + 7u));
    return vec3<f32>(h >> vec3<u32>(8u)) / 16777216.0;
}

fn lattice_grad(c: vec3<i32>, period: i32, seed: u32) -> vec3<f32> {
    var q = c;
    if (period > 0) {
        q = ((c % vec3<i32>(period)) + vec3<i32>(period)) % vec3<i32>(period);
    }
    return normalize(rand3(q, seed) * 2.0 - 1.0 + vec3<f32>(1e-4));
}

// Gradient noise, roughly in [−1, 1]; periodic with `period` lattice cells
// when period > 0.
fn gnoise(p: vec3<f32>, period: i32, seed: u32) -> f32 {
    let fl = floor(p);
    let i = vec3<i32>(fl);
    let f = p - fl;
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    var n: array<f32, 8>;
    for (var k = 0; k < 8; k++) {
        let o = vec3<i32>(k & 1, (k >> 1) & 1, (k >> 2) & 1);
        n[k] = dot(lattice_grad(i + o, period, seed), f - vec3<f32>(o));
    }
    let x0 = mix(mix(n[0], n[1], u.x), mix(n[2], n[3], u.x), u.y);
    let x1 = mix(mix(n[4], n[5], u.x), mix(n[6], n[7], u.x), u.y);
    return 1.6 * mix(x0, x1, u.z);
}

fn fbm(p: vec3<f32>, octaves: i32, seed: u32) -> f32 {
    var s = 0.0;
    var a = 0.5;
    var q = p;
    var norm = 0.0;
    for (var i = 0; i < octaves; i++) {
        s += a * gnoise(q, 0, seed + u32(i) * 31u);
        norm += a;
        q = q * 2.03 + vec3<f32>(1.7, 9.2, 3.1);
        a *= 0.5;
    }
    return s / norm;
}

// Ridged multifractal in [0, 1]: sharp crests where the noise crosses zero,
// the look of filaments and sheets seen edge on.
fn ridged(p: vec3<f32>, octaves: i32, seed: u32) -> f32 {
    var s = 0.0;
    var a = 0.5;
    var q = p;
    var norm = 0.0;
    var w = 1.0;
    for (var i = 0; i < octaves; i++) {
        var r = 1.0 - abs(gnoise(q, 0, seed + u32(i) * 57u));
        r = r * r * w;
        w = clamp(r * 1.5, 0.0, 1.0);
        s += a * r;
        norm += a;
        q = q * 2.07 + vec3<f32>(5.3, 1.1, 7.7);
        a *= 0.5;
    }
    return s / norm;
}

// Vector noise for domain warping.
fn warp(p: vec3<f32>, seed: u32) -> vec3<f32> {
    return vec3<f32>(fbm(p, 3, seed), fbm(p + vec3<f32>(31.4, 7.1, 2.7), 3, seed + 3u), fbm(p + vec3<f32>(3.9, 41.3, 13.1), 3, seed + 7u));
}

// Cellular noise: distances to the nearest and second-nearest feature
// point, and a random number for the nearest one's cell.
fn worley(p: vec3<f32>, seed: u32) -> vec3<f32> {
    let fl = floor(p);
    let i = vec3<i32>(fl);
    var f1 = 9.0;
    var f2 = 9.0;
    var id = 0.0;
    for (var z = -1; z <= 1; z++) {
        for (var y = -1; y <= 1; y++) {
            for (var x = -1; x <= 1; x++) {
                let c = i + vec3<i32>(x, y, z);
                let h = rand3(c, seed);
                let d = length(vec3<f32>(c) + h - p);
                if (d < f1) {
                    f2 = f1;
                    f1 = d;
                    id = fract(h.x * 17.0 + h.y * 3.0);
                } else if (d < f2) {
                    f2 = d;
                }
            }
        }
    }
    return vec3<f32>(f1, f2, id);
}

// ---------------------------------------------------------------------------
// Voxel geometry
// ---------------------------------------------------------------------------

fn seed() -> u32 {
    return u32(gen.centre.w);
}

// Voxel id of this invocation, or a negative x when out of range.
fn voxel_id(gid: vec3<u32>) -> vec3<i32> {
    let id = gid + vec3<u32>(0u, 0u, gen.dims.w);
    if (any(id >= gen.dims.xyz)) {
        return vec3<i32>(-1);
    }
    return vec3<i32>(id);
}

// Box coordinates in [−1, 1] of a voxel centre.
fn voxel_local(id: vec3<i32>) -> vec3<f32> {
    return (vec3<f32>(id) + 0.5) / vec3<f32>(gen.dims.xyz) * 2.0 - 1.0;
}

// Box coordinates in parsecs along the box axes.
fn local_pc(l: vec3<f32>) -> vec3<f32> {
    return l * vec3<f32>(gen.ax.w, gen.ay.w, gen.az.w);
}

// Position relative to the hole (pc).
fn world_pc(l: vec3<f32>) -> vec3<f32> {
    let s = local_pc(l);
    return gen.centre.xyz + s.x * gen.ax.xyz + s.y * gen.ay.xyz + s.z * gen.az.xyz;
}

// A box-local vector in world axes.
fn local_to_world(v: vec3<f32>) -> vec3<f32> {
    return v.x * gen.ax.xyz + v.y * gen.ay.xyz + v.z * gen.az.xyz;
}

// Texture coordinates in [0, 1] of a world position (pc) in this box.
fn box_uv(x: vec3<f32>) -> vec3<f32> {
    let d = x - gen.centre.xyz;
    return vec3<f32>(dot(d, gen.ax.xyz) / gen.ax.w, dot(d, gen.ay.xyz) / gen.ay.w, dot(d, gen.az.xyz) / gen.az.w) * 0.5
        + 0.5;
}

fn occ_uv(x: vec3<f32>) -> vec3<f32> {
    let d = x - gen.occ_centre.xyz;
    return vec3<f32>(dot(d, gen.occ_ax.xyz), dot(d, gen.occ_ay.xyz), dot(d, gen.occ_az.xyz)) * 0.5 + 0.5;
}

// 1 inside the box, falling to 0 over the outer 8% so nothing is cut off
// by a box face.
fn edge_fade(l: vec3<f32>) -> f32 {
    let m = max(max(abs(l.x), abs(l.y)), abs(l.z));
    return 1.0 - smoothstep(0.92, 0.99, m);
}

fn inside01(u: vec3<f32>) -> bool {
    return all(u >= vec3<f32>(0.0)) && all(u <= vec3<f32>(1.0));
}

// ---------------------------------------------------------------------------
// Detail noise for the ray march: 64³, tiling. r: two voxel-scale octaves,
// g: two sub-voxel octaves, b and a: a warp vector (see `nebula.wgsl`).
// ---------------------------------------------------------------------------

@compute @workgroup_size(4, 4, 4)
fn gen_detail(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = vec3<f32>(gid) + 0.5;
    let sd = seed();
    let r = 0.7 * gnoise(p / 16.0, 4, sd + 1u) + 0.3 * gnoise(p / 8.0, 8, sd + 2u);
    let g = 0.7 * gnoise(p / 4.0, 16, sd + 3u) + 0.3 * gnoise(p / 2.0, 32, sd + 4u);
    let b = gnoise(p / 8.0, 8, sd + 5u);
    let a = gnoise(p / 8.0, 8, sd + 6u);
    textureStore(out_detail, gid, clamp(vec4<f32>(r, g, b, a) * 0.5 + 0.5, vec4<f32>(0.0), vec4<f32>(1.0)));
}

// ---------------------------------------------------------------------------
// The minispiral (Sgr A West): gas streams along the nodes, plus a tenuous
// gas filling the cavity. Writes gas density n_H / n_scale; the light pass
// decides what is ionised.
//   p0: diffuse density, radius of the empty centre (pc), outer radius of the
//       diffuse gas (pc), unused
// ---------------------------------------------------------------------------

struct StreamHit {
    dens: f32,
    tangent: vec3<f32>,
    vel: vec3<f32>,
}

// Gas density of the streams at x (pc), the flow direction and velocity of
// the densest one.
fn streams_at(x: vec3<f32>) -> StreamHit {
    var out: StreamHit;
    out.dens = 0.0;
    out.tangent = vec3<f32>(1.0, 0.0, 0.0);
    out.vel = vec3<f32>(0.0);
    var best = 0.0;
    var cur_id = -1.0;
    var cur_max = 0.0;
    let n = arrayLength(&nodes);
    for (var i = 0u; i + 1u < n; i++) {
        let a = nodes[i];
        let b = nodes[i + 1u];
        if (a.vel.w != b.vel.w) {
            continue;
        }
        if (a.vel.w != cur_id) {
            out.dens += cur_max;
            cur_max = 0.0;
            cur_id = a.vel.w;
        }
        let ab = b.pos.xyz - a.pos.xyz;
        let s = clamp(dot(x - a.pos.xyz, ab) / max(dot(ab, ab), 1e-12), 0.0, 1.0);
        let c = a.pos.xyz + ab * s;
        let t = normalize(mix(a.tan.xyz, b.tan.xyz, s));
        var nr = mix(a.nrm.xyz, b.nrm.xyz, s);
        nr = normalize(nr - dot(nr, t) * t);
        let bn = cross(t, nr);
        let w = mix(a.tan.w, b.tan.w, s);
        let h = mix(a.nrm.w, b.nrm.w, s);
        let o = x - c;
        let q = pow(dot(o, t) / w, 2.0) + pow(dot(o, nr) / h, 2.0) + pow(dot(o, bn) / w, 2.0);
        let d = mix(a.pos.w, b.pos.w, s) * exp(-q);
        cur_max = max(cur_max, d);
        if (d > best) {
            best = d;
            out.tangent = t;
            out.vel = mix(a.vel.xyz, b.vel.xyz, s);
        }
    }
    out.dens += cur_max;
    return out;
}

@compute @workgroup_size(4, 4, 4)
fn gen_minispiral(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = voxel_id(gid);
    if (id.x < 0) {
        return;
    }
    let lv = voxel_local(id);
    let x = world_pc(lv);
    let sd = seed();
    let r = length(x);
    // Wavy edges: warp the position before measuring distance to a stream.
    let xw = x + 0.045 * warp(x * 4.0, sd);
    let st = streams_at(xw);
    // Striations along the flow (the arms are streaky in radio and Paα
    // maps): noise stretched fivefold along the stream.
    let q = xw * 16.0;
    let qa = q - 0.8 * dot(q, st.tangent) * st.tangent;
    let stri = ridged(qa + 0.4 * warp(qa * 0.3, sd + 11u), 4, sd + 5u);
    let clumps = exp(0.7 * fbm(xw * 7.0, 4, sd + 9u));
    let window = smoothstep(gen.p0.y, 1.6 * gen.p0.y, r) * edge_fade(lv);
    let n = st.dens * (0.25 + 1.3 * stri * stri) * clumps * window;
    // The cavity's tenuous ionised gas carries little dust (it is sputtered
    // in the hot, ionised cavity).
    let diffuse = gen.p0.x * exp(1.2 * fbm(x * 2.5, 4, sd + 13u)) * (1.0 - smoothstep(0.7 * gen.p0.z, gen.p0.z, r)) * window;
    textureStore(out_gas, id, vec4<f32>(max(n + diffuse, 0.0), max(n, 0.0), 0.0, 0.0));
}

// ---------------------------------------------------------------------------
// The circumnuclear disk, in its own frame (x, y in the disk plane, z along
// its axis). Gas density n_H / n_scale.
//   p0: inner radius (pc), scale height at the inner edge (pc), flare
//       (pc per pc), warp (pc per pc)
//   p1: Western Arc direction in the disk plane (x, y), arc density, unused
//   p2: inter-clump density, clump peak density, clump cell size (pc),
//       outer decline length (pc)
//   p3: centre offset (x, y, pc; the disk is lopsided), radius where the
//       outer decline starts (pc), filament density
// ---------------------------------------------------------------------------

@compute @workgroup_size(4, 4, 4)
fn gen_cnd(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = voxel_id(gid);
    if (id.x < 0) {
        return;
    }
    let sd = seed();
    let lv = voxel_local(id);
    let l = local_pc(lv);
    let xy = l.xy - gen.p3.xy;
    let rho = max(length(xy), 1e-4);
    let dir = xy / rho;
    let ephi = vec3<f32>(-dir.y, dir.x, 0.0);
    // Ragged, lopsided inner edge.
    let rin = gen.p0.x * (1.0 + 0.12 * fbm(vec3<f32>(dir * 1.3, 0.5), 3, sd + 1u));
    let inner = smoothstep(rin - 0.1, rin + 0.12, rho);
    let outer = exp(-pow(max(rho - gen.p3.z, 0.0) / gen.p2.w, 2.0));
    // Warped and flaring: sin(φ − 0.8) from the in-plane direction.
    let z0 = gen.p0.w * rho * (dir.y * 0.697 - dir.x * 0.717);
    let h = gen.p0.y + gen.p0.z * max(rho - rin, 0.0);
    let dz = l.z - z0;
    let vert = exp(-pow(dz / h, 2.0));
    let base = inner * outer * vert;

    // Clumps sheared by the differential rotation: compress coordinates
    // along the orbit so cells stretch into arcs.
    let p = l;
    let q = (p - 0.65 * dot(p, ephi) * ephi) / gen.p2.z;
    let wq = q + 0.35 * warp(q * 0.7, sd + 2u);
    let wv = worley(wq, sd + 3u);
    let amp = 0.2 + 0.8 * wv.z * wv.z;
    let clump = amp * exp(-pow(wv.x / 0.33, 2.0));
    let fil = ridged(q * 1.6 + 0.5 * warp(q * 0.5, sd + 4u), 4, sd + 5u);
    var n = base * (gen.p2.x * exp(0.8 * fbm(q * 0.8, 3, sd + 6u)) + gen.p2.y * clump * clump + gen.p3.w * fil * fil * fil);

    // The Western Arc: a ridge of denser gas along the inner edge on the
    // side facing west, which the central cluster ionises.
    let rim = exp(-pow((rho - rin - 0.06) / 0.09, 2.0)) * exp(-pow(dz / (1.6 * h), 2.0));
    let side = smoothstep(0.1, 0.8, dot(dir, normalize(gen.p1.xy)));
    let arc_tex = 0.35 + 0.9 * ridged(p * 9.0 + warp(p * 2.0, sd + 7u), 3, sd + 8u);
    n += gen.p1.z * rim * side * arc_tex;
    n = max(n * edge_fade(lv), 0.0);
    textureStore(out_gas, id, vec4<f32>(n, n, 0.0, 0.0));
}

// ---------------------------------------------------------------------------
// Photoionisation by the central cluster (the minispiral and the CND). For
// each voxel, march to the centre: ionising photons per steradian that are
// left after the recombinations on the way,
//   N = Q/4π − ∫ α_B n² r² dr,
// decide which fraction of the voxel is ionised (an ionisation-bounded skin
// whose emission measure is set by the ionising flux, not the density).
// The same march gives the dust's transmission of the cluster's starlight.
// Input: all gas (r) and the gas that carries dust (g); output g is the
// latter.
//   p0: Q/(4π α_B) in (10⁴ cm⁻³)² pc³, n_scale / 10⁴ cm⁻³, τ_V per
//       (10⁴ cm⁻³ pc), radius of the source region (pc)
// ---------------------------------------------------------------------------

@compute @workgroup_size(4, 4, 4)
fn gen_light(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = voxel_id(gid);
    if (id.x < 0) {
        return;
    }
    let gd = textureLoad(in_vol, id, 0).rg;
    let gas = gd.x;
    if (gas <= 0.0) {
        textureStore(out_vol, id, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }
    let x = world_pc(voxel_local(id));
    let r = length(x);
    let vox = 2.0 * min(min(gen.ax.w / f32(gen.dims.x), gen.ay.w / f32(gen.dims.y)), gen.az.w / f32(gen.dims.z));
    let r0 = gen.p0.w;
    let steps = 112;
    let r_top = max(r - 0.5 * vox, r0);
    let dr = (r_top - r0) / f32(steps);
    var rec = 0.0;
    var col = 0.0;
    for (var i = 0; i < steps; i++) {
        let rr = r_top - (f32(i) + 0.5) * dr;
        let p = x * (rr / r);
        var nn = vec2<f32>(0.0);
        let u = box_uv(p);
        if (inside01(u)) {
            nn += textureSampleLevel(in_vol, lin, u, 0.0).rg;
        }
        if (gen.occ_centre.w > 0.0) {
            let uo = occ_uv(p);
            if (inside01(uo)) {
                nn += gen.occ_centre.w * textureSampleLevel(in_occ, lin, uo, 0.0).g;
            }
        }
        let n4 = nn * gen.p0.y;
        rec += n4.x * n4.x * rr * rr * dr;
        col += n4.y * dr;
    }
    let n4 = gas * gen.p0.y;
    let demand = n4 * n4 * r * r * vox;
    let avail = gen.p0.x - rec;
    let xion = clamp(avail / max(demand, 1e-30), 0.0, 1.0);
    // (n_e / 10⁴)², with helium adding 10% to n_e.
    let ion = xion * 1.1 * n4 * n4;
    // Where the ionising photons run out: the partially ionised front.
    let front = xion * exp(-max(avail, 0.0) / (3.0 * max(demand, 1e-30)));
    let light = exp(-col * gen.p0.z);
    textureStore(out_vol, id, vec4<f32>(ion, gd.y, front, light));
}

// ---------------------------------------------------------------------------
// Sgr A East: the supernova remnant's radiative shock front, a few thin
// wrinkled sheets on an ellipsoid (the Veil-like filaments are these sheets
// seen edge on). Box axes are the ellipsoid's.
//   p0: sheet half width (in units of the semi-axes), sheet count, unused,
//       brightening towards the cloud it runs into
//   p1: direction to that cloud (box axes), unused
//   p2: semi-axes (pc), dust per unit emission
// ---------------------------------------------------------------------------

@compute @workgroup_size(4, 4, 4)
fn gen_sgra_east(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = voxel_id(gid);
    if (id.x < 0) {
        return;
    }
    let sd = seed();
    let lv = voxel_local(id);
    let q = local_pc(lv) / gen.p2.xyz;
    let rho = length(q);
    let d = q / max(rho, 1e-6);
    var ion = 0.0;
    var hi = 0.0;
    for (var i = 0; i < i32(gen.p0.y); i++) {
        let si = sd + u32(i) * 17u;
        let rs = 1.0 - 0.035 * f32(i) + 0.05 * fbm(d * 2.2, 3, si) + 0.014 * fbm(d * 9.0, 3, si + 3u);
        let x = (rho - rs) / gen.p0.x;
        let sheet = exp(-x * x);
        let cover = smoothstep(-0.05, 0.4, fbm(d * 2.8, 3, si + 7u) + select(0.0, 0.15, i == 0));
        let e = sheet * cover;
        ion += e;
        // [O III] leads the Balmer lines slightly outwards, strongest where
        // the shock is fast.
        let fast = smoothstep(-0.2, 0.5, fbm(d * 1.6, 2, si + 9u));
        hi += e * smoothstep(-0.4, 1.2, x) * fast;
    }
    let a = clamp(hi / max(ion, 1e-6), 0.0, 1.0);
    let cloud = 1.0 + gen.p0.w * smoothstep(0.2, 0.9, dot(d, gen.p1.xyz));
    // Fine braided texture within the sheets.
    let braid = 0.3 + 1.2 * ridged(q * 16.0 + 0.5 * warp(q * 3.0, sd + 21u), 3, sd + 23u);
    ion *= cloud * braid * edge_fade(lv);
    textureStore(out_vol, id, vec4<f32>(ion, gen.p2.w * ion, 1.0 - a, a));
}

// ---------------------------------------------------------------------------
// A Crab-like pulsar wind nebula. Box axes are the nebula's: x along the
// pulsar's spin axis (the long axis), y and z across.
//   p0: semi-axes (pc), radius of the filament cage (fraction of the
//       semi-axes)
//   p1: cell counts of the coarse and fine filament networks and of the
//       Rayleigh–Taylor fingers, finger length (fraction)
//   p2: torus radius, torus half width, jet length, jet radius (pc)
// ---------------------------------------------------------------------------

@compute @workgroup_size(4, 4, 4)
fn gen_pwn(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = voxel_id(gid);
    if (id.x < 0) {
        return;
    }
    let sd = seed();
    let lv = voxel_local(id);
    let l = local_pc(lv);
    let q = l / gen.p0.xyz;
    let qw = q + 0.07 * warp(q * 2.2, sd);
    let rho = length(qw);
    let d = qw / max(rho, 1e-6);
    let cage = gen.p0.w;
    // Filaments sit at different depths within the cage.
    let rj = rho + 0.05 * gnoise(d * 5.0, 0, sd + 2u);

    // The cage: a network of filaments along Voronoi cell edges on the
    // sphere of directions, coarse and fine.
    let v1 = worley(d * gen.p1.x, sd + 3u);
    let v2 = worley(d * gen.p1.y + vec3<f32>(11.0), sd + 4u);
    let net = exp(-pow((v1.y - v1.x) / 0.07, 2.0)) + 0.55 * exp(-pow((v2.y - v2.x) / 0.06, 2.0));
    let shell = smoothstep(cage - 0.2, cage - 0.06, rj) * (1.0 - smoothstep(cage + 0.06, cage + 0.14, rj));

    // Rayleigh–Taylor fingers: the light synchrotron bubble pushes on the
    // heavier ejecta, which fall inwards in fingers capped by blobs.
    let v3 = worley(d * gen.p1.z, sd + 5u);
    let has = step(0.35, fract(v3.z * 7.31));
    let len = gen.p1.w * (0.3 + 0.7 * fract(v3.z * 13.7));
    let top = cage;
    let tip = top - len;
    let along = smoothstep(tip - 0.01, tip + 0.03, rj) * (1.0 - smoothstep(top - 0.02, top + 0.05, rj));
    let taper = 0.55 + 0.45 * clamp((top - rj) / max(len, 1e-3), 0.0, 1.0);
    let core = exp(-pow(v3.x / (0.1 * taper), 2.0));
    let cap = exp(-pow((rj - tip) / 0.025, 2.0)) * exp(-pow(v3.x / 0.16, 2.0));
    let finger = has * (core * along + 0.8 * cap);

    var dens = (net * shell + finger) * (0.5 + 0.5 * fbm(qw * 9.0, 3, sd + 6u));
    dens *= 0.5 + 0.8 * ridged(qw * 24.0, 3, sd + 7u);
    let fade = edge_fade(lv);
    let ion = dens * dens * fade;

    // Synchrotron nebula: fills the cage, brighter inwards, fibrous along
    // the toroidal magnetic field, with the equatorial torus and the jet
    // along the spin axis.
    let inside = 1.0 - smoothstep(cage - 0.12, cage + 0.02, rj);
    let body = inside * (0.35 + 0.65 * exp(-dot(q, q) / 0.3));
    let rad = length(l.yz);
    let ephi = vec3<f32>(0.0, -l.z, l.y) / max(rad, 1e-3);
    let fq = l * 8.0;
    let fqa = fq - 0.75 * dot(fq, ephi) * ephi;
    let fib = 0.45 + 1.1 * ridged(fqa + 0.4 * warp(fqa * 0.4, sd + 8u), 4, sd + 9u);
    let torus = exp(-(pow(rad - gen.p2.x, 2.0) + l.x * l.x) / (gen.p2.y * gen.p2.y));
    let jet = exp(-pow(rad / gen.p2.w, 2.0)) * smoothstep(0.05, 0.15, l.x) * (1.0 - smoothstep(0.6 * gen.p2.z, gen.p2.z, l.x));
    let synch = (body * fib + 2.0 * torus * (0.6 + 0.8 * fib) + 1.2 * jet) * fade;

    // Dense filament cores are cool and weakly ionised ([S II], [O I]);
    // the outer skins are highly excited ([O III]).
    let low = smoothstep(0.25, 0.8, dens);
    let hi = clamp(smoothstep(cage - 0.05, cage + 0.1, rj) + 0.3 * (1.0 - low), 0.0, 1.0);
    textureStore(out_vol, id, vec4<f32>(ion, synch, low, hi));
}

// ---------------------------------------------------------------------------
// Velocity fields (1000 km/s, world axes) at a quarter of the resolution.
//   p3.w: kind (0 minispiral, 1 CND, 2 SNR, 3 PWN)
//   p0: kind 0: GM (1000 km/s)² pc; kind 1: rotation speed; kind 2: speed at
//       the shell; kind 3: 1/age (1000 km/s per pc)
//   p1: rotation axis (kinds 0, 1); kind 2: semi-axes (pc)
// ---------------------------------------------------------------------------

@compute @workgroup_size(4, 4, 4)
fn gen_velocity(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = voxel_id(gid);
    if (id.x < 0) {
        return;
    }
    let l = voxel_local(id);
    let x = world_pc(l);
    let kind = u32(gen.p3.w);
    var v = vec3<f32>(0.0);
    let r = max(length(x), 1e-3);
    let axis = gen.p1.xyz;
    let around = cross(axis, x);
    let around_dir = around / max(length(around), 1e-4);
    if (kind == 0u) {
        let st = streams_at(x);
        let rot = around_dir * sqrt(gen.p0.x / r);
        let w = clamp(st.dens * 4.0, 0.0, 1.0);
        v = mix(rot, st.vel, w);
    } else if (kind == 1u) {
        v = around_dir * gen.p0.x;
    } else if (kind == 2u) {
        v = gen.p0.x * local_to_world(local_pc(l) / gen.p1.xyz);
    } else {
        v = gen.p0.x * (x - gen.centre.xyz);
    }
    textureStore(out_vol, id, vec4<f32>(v, 0.0));
}

// ---------------------------------------------------------------------------
// Occupancy for empty-space skipping: mip level 3 holds, per 8³ block, the
// largest value of r and g over the block and a one-voxel apron (trilinear
// filtering reaches that far).
// ---------------------------------------------------------------------------

@compute @workgroup_size(4, 4, 4)
fn gen_occupancy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = voxel_id(gid);
    if (id.x < 0) {
        return;
    }
    let full = vec3<i32>(textureDimensions(in_vol, 0));
    let lo = max(id * 8 - vec3<i32>(1), vec3<i32>(0));
    let hi = min(id * 8 + vec3<i32>(9), full);
    var m = vec2<f32>(0.0);
    for (var z = lo.z; z < hi.z; z++) {
        for (var y = lo.y; y < hi.y; y++) {
            for (var x = lo.x; x < hi.x; x++) {
                m = max(m, textureLoad(in_vol, vec3<i32>(x, y, z), 0).rg);
            }
        }
    }
    textureStore(out_vol, id, vec4<f32>(m, 0.0, 0.0));
}

// ---------------------------------------------------------------------------
// Column sums of r and g, for calibrating total luminosities.
// ---------------------------------------------------------------------------

@compute @workgroup_size(8, 8, 1)
fn gen_sum(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = vec3<u32>(textureDimensions(in_vol, 0));
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }
    var s = vec2<f32>(0.0);
    for (var z = 0u; z < dims.z; z++) {
        s += textureLoad(in_vol, vec3<u32>(gid.xy, z), 0).rg;
    }
    sums[gid.y * dims.x + gid.x] = s;
}
