// ---------------------------------------------------------------------------
// Boulders (`terrain/rocks.rs`): rocks on the ground, part of the terrain's
// heightfield (so they cast shadows, hide the sky and can be stood on).
// The tile generator raises them at one level (`tile_gen.wgsl`); the trace
// finds the same ones at a hit to give them rock's material
// (`terrain_rq.wgsl`).
//
// They sit on the cube face's grid at `cell` level (cells ~3 m): a place on
// a tile is a cell index and a fraction in it, exact from the tile's index
// and its place in it, and the same for a point on the ground and on top
// of a boulder (no height involved). A cell may hold one boulder (the
// share `density` of them do), jittered, under half a cell in radius, so
// the four cells nearest a point hold every boulder that reaches it and
// the field is continuous.
// ---------------------------------------------------------------------------

// How many cells hold a boulder, from a tile's material channels
// (mountainousness, moisture, basalt share, land): a few on land, more
// where the rock is basalt and in the mountains.
fn boulder_density(m: vec4<f32>) -> f32 {
    return (0.01 + 0.05 * m.z + 0.25 * m.x * m.x) * smoothstep(0.3, 0.7, m.w);
}

// Place `st` of `tile` (face, level, x, y) on the face grid at `cell_level`:
// the cell (face-wide index) and the fraction in it.
fn boulder_cell(tile: vec4<u32>, st: vec2<f32>, cell_level: u32) -> vec4<f32> {
    let level = tile.y;
    var g: vec2<f32>;
    var c0 = vec2<i32>(0);
    if (level >= cell_level) {
        let k = level - cell_level;
        let n = f32(1u << k);
        let within = (vec2<f32>(vec2<u32>(tile.z, tile.w) & vec2<u32>((1u << k) - 1u)) + st) / n;
        c0 = vec2<i32>(vec2<u32>(tile.z, tile.w) >> vec2<u32>(k));
        g = within;
    } else {
        let n = 1u << (cell_level - level);
        c0 = vec2<i32>(vec2<u32>(tile.z, tile.w) * n);
        g = st * f32(n);
    }
    let fl = floor(g);
    return vec4<f32>(vec2<f32>(c0) + fl, g - fl);
}

// How high (km) the boulders rise at `st` of `tile`, and how much of the
// point is boulder (0–1). `cell_km`: a cell's size.
fn boulders_at(tile: vec4<u32>, st: vec2<f32>, cell_level: u32, cell_km: f32, density: f32, seed: u32) -> vec2<f32> {
    let cf = boulder_cell(tile, st, cell_level);
    let cell = vec2<i32>(cf.xy);
    let f = cf.zw;
    let base = vec2<i32>(floor(f - 0.5));
    var top = 0.0;
    var mask = 0.0;
    for (var k = 0; k < 4; k++) {
        let c = base + vec2<i32>(k & 1, k >> 1);
        let id = bitcast<vec2<u32>>(cell + c);
        let hs = anc_pcg3d(vec3<u32>(id.x ^ seed, id.y, tile.x ^ (seed << 3u)));
        let r = vec3<f32>(hs >> vec3<u32>(8u)) / 16777216.0;
        if (r.x >= density) {
            continue;
        }
        let hs2 = anc_pcg3d(hs ^ vec3<u32>(0x68bc21ebu));
        let r2 = vec3<f32>(hs2 >> vec3<u32>(8u)) / 16777216.0;
        let radius = 0.1 + 0.36 * r.y * r.y;
        // Jittered so the boulder stays inside its cell's reach.
        let centre = vec2<f32>(c) + 0.5 + (r2.xy - 0.5) * (0.98 - 2.0 * radius);
        var off = f - centre;
        // Elongated along a random direction, with a lumpy outline.
        let a = 6.2831853 * r2.z;
        let dir = vec2<f32>(cos(a), sin(a));
        let e = 0.6 + 0.4 * fract(r.z * 7.31);
        off = vec2<f32>(dot(off, dir), dot(off, vec2<f32>(-dir.y, dir.x)) / e);
        let ang = atan2(off.y, off.x);
        let lump = 1.0 + 0.18 * sin(3.0 * ang + 6.2831853 * r.z) + 0.1 * sin(5.0 * ang + 6.2831853 * r2.x);
        let q = length(off) / (radius * lump);
        if (q < 1.0) {
            // A blocky dome, tilted.
            let tilt = 1.0 + 0.35 * dot(off / radius, vec2<f32>(r2.y - 0.5, r2.x - 0.5));
            let dome = pow(1.0 - q * q, 0.4) * radius * (0.35 + 0.35 * r.z) * tilt;
            top = max(top, dome * cell_km);
            mask = max(mask, 1.0 - smoothstep(0.9, 1.0, q));
        }
    }
    return vec2<f32>(top, mask);
}
