// ---------------------------------------------------------------------------
// Anchored lattice noise (twin of `terrain/anchor.rs`; keep them in step):
// centimetre octaves stay exact near the camera. Per octave the CPU gives
// the anchor's lattice cell (wrapped i32) and its fraction in the cell;
// here only small offsets `d` (km) from the anchor are added, so the
// lattice coordinate splits exactly into cell and fraction. Values hash
// the integer cell, so any anchor gives the same noise.
// ---------------------------------------------------------------------------

struct AnchorOctave {
    cell: vec4<i32>,       // anchor's cell x, y, z; w the octave's seed
    frac_freq: vec4<f32>,  // anchor's fraction x, y, z; w cells per km
}

struct AncLattice {
    cell: vec3<i32>,
    frac: vec3<f32>,
}

fn anc_lattice(o: AnchorOctave, d_km: vec3<f32>) -> AncLattice {
    let x = o.frac_freq.xyz + d_km * o.frac_freq.w;
    let fl = floor(x);
    // i32 addition wraps, as `wrapping_add` does on the CPU.
    return AncLattice(o.cell.xyz + vec3<i32>(fl), x - fl);
}

// PCG-style 3-D hash (Jarzynski & Olano 2020, "pcg3d").
fn anc_pcg3d(v0: vec3<u32>) -> vec3<u32> {
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

// The gradient at a lattice corner: components in [−1, 1].
fn anc_gradient(cell: vec3<i32>, seed: u32) -> vec3<f32> {
    let c = bitcast<vec3<u32>>(cell);
    let h = anc_pcg3d(vec3<u32>(c.x ^ seed, c.y, c.z ^ ((seed << 16u) | (seed >> 16u))));
    return vec3<f32>(h >> vec3<u32>(8u)) / 8388607.5 - 1.0;
}

fn anc_corner(cell: vec3<i32>, f: vec3<f32>, corner: vec3<i32>, seed: u32) -> f32 {
    return dot(anc_gradient(cell + corner, seed), f - vec3<f32>(corner));
}

// Gradient noise (roughly −1 to 1) with quintic fades.
fn anc_noise(cell: vec3<i32>, f: vec3<f32>, seed: u32) -> f32 {
    let w = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let x0 = mix(anc_corner(cell, f, vec3<i32>(0, 0, 0), seed), anc_corner(cell, f, vec3<i32>(1, 0, 0), seed), w.x);
    let x1 = mix(anc_corner(cell, f, vec3<i32>(0, 1, 0), seed), anc_corner(cell, f, vec3<i32>(1, 1, 0), seed), w.x);
    let x2 = mix(anc_corner(cell, f, vec3<i32>(0, 0, 1), seed), anc_corner(cell, f, vec3<i32>(1, 0, 1), seed), w.x);
    let x3 = mix(anc_corner(cell, f, vec3<i32>(0, 1, 1), seed), anc_corner(cell, f, vec3<i32>(1, 1, 1), seed), w.x);
    return mix(mix(x0, x1, w.y), mix(x2, x3, w.y), w.z);
}

// One octave's noise at `d_km` from the anchor.
fn anchored_noise(o: AnchorOctave, d_km: vec3<f32>) -> f32 {
    let l = anc_lattice(o, d_km);
    return anc_noise(l.cell, l.frac, bitcast<u32>(o.cell.w));
}
