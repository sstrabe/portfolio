// ---------------------------------------------------------------------------
// The regional erosion's square (`terrain/erosion.rs`) for the shaders that
// read it: the tile generator, the probe and the trace. Each declares
// `region` (a Region uniform) and `region_cells` (per cell: height change,
// km; distance from the shore, m) in its own bind group; the lookups here
// read them.
// ---------------------------------------------------------------------------

struct Region {
    centre: vec4<f32>,  // unit direction of the centre (body-fixed), cell size (km)
    east: vec4<f32>,    // unit, cells a side (0: no square)
    north: vec4<f32>,   // unit, unused
    planet: vec4<f32>,  // the planet it's for: seed (bits), radius (km), unused ×2
}

// Bilinear lookup of the square's cells at body-fixed `q` on the planet
// with `seed` and `radius` (km): (height change km, shore distance m, 1
// inside / 0 outside, or on another planet).
fn region_lookup(q: vec3<f32>, seed: u32, radius: f32) -> vec3<f32> {
    let n = region.east.w;
    if (n < 2.0 || bitcast<u32>(region.planet.x) != seed || abs(region.planet.y - radius) > 0.5) {
        return vec3<f32>(0.0);
    }
    let c = region.centre.xyz;
    let cq = dot(q, c);
    if (cq < 0.9) {
        return vec3<f32>(0.0);
    }
    // Onto the square's plane, as the bake laid its cells.
    let p = (q / cq - c) * radius / region.centre.w;
    let x = dot(p, region.east.xyz) + 0.5 * n - 0.5;
    let y = dot(p, region.north.xyz) + 0.5 * n - 0.5;
    if (x < 0.0 || y < 0.0 || x >= n - 1.0 || y >= n - 1.0) {
        return vec3<f32>(0.0);
    }
    let ni = i32(n);
    let b = vec2<i32>(i32(floor(x)), i32(floor(y)));
    let f = vec2<f32>(x, y) - vec2<f32>(b);
    let k = b.y * ni + b.x;
    let v = mix(mix(region_cells[k], region_cells[k + 1], f.x), mix(region_cells[k + ni], region_cells[k + ni + 1], f.x), f.y);
    return vec3<f32>(v, 1.0);
}

// The erosion's height change (km) at `q`, where the sampling (`lod`, km)
// is fine enough to hold its valleys.
fn region_delta(q: vec3<f32>, seed: u32, radius: f32, lod: f32) -> f32 {
    if (lod > 8.0 * region.centre.w) {
        return 0.0;
    }
    return region_lookup(q, seed, radius).x;
}

// How far (m) `q` is from the shore, where the square knows (−1 outside).
fn region_shore(q: vec3<f32>, seed: u32, radius: f32) -> f32 {
    let r = region_lookup(q, seed, radius);
    return select(-1.0, r.y, r.z > 0.5);
}
