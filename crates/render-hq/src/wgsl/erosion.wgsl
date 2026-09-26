// ---------------------------------------------------------------------------
// Regional erosion (`terrain/erosion.rs`): a square of the planet around
// the eye, N² cells on the tangent plane at its centre, worn down by
// running water over millions of years: stream-power incision (erosion ∝
// √(drainage area) × slope, towards each cell's steepest-descent
// neighbour) and hillslope diffusion, with the sea as the base level. The
// drainage area builds up downstream by one cell a step. What it takes
// off inland, faded out near the coast and the square's edges, is the
// height change the tile generator adds.
// ---------------------------------------------------------------------------

struct ErosionParams {
    centre: vec4<f32>,   // unit direction of the centre (body-fixed), cell size (m)
    east: vec4<f32>,     // unit, cells a side
    north: vec4<f32>,    // unit, time step (years)
    terrain: vec4<u32>,  // kind, seed, air, unused
    planet: vec4<f32>,   // radius (km), relief, sea, t_eq
    stream: vec4<f32>,   // erodibility K, area exponent m, diffusivity (m²/yr), unused
}

@group(0) @binding(1) var<uniform> ep: ErosionParams;
@group(0) @binding(2) var<storage, read_write> h_in: array<f32>;
@group(0) @binding(3) var<storage, read_write> h_out: array<f32>;
@group(0) @binding(4) var<storage, read_write> initial: array<f32>;
@group(0) @binding(5) var<storage, read_write> receiver: array<u32>;
@group(0) @binding(6) var<storage, read_write> area_in: array<f32>;
@group(0) @binding(7) var<storage, read_write> area_out: array<f32>;
// Per cell: the height change (km) and the distance from the shore (m).
@group(0) @binding(8) var<storage, read_write> delta: array<vec2<f32>>;

fn er_n() -> i32 {
    return i32(ep.east.w);
}

fn er_index(x: i32, y: i32) -> u32 {
    return u32(y * er_n() + x);
}

fn er_cell(gid: vec3<u32>) -> vec2<i32> {
    return vec2<i32>(gid.xy);
}

const ER_OFFSETS: array<vec2<i32>, 8> = array<vec2<i32>, 8>(
    vec2<i32>(-1, 0), vec2<i32>(1, 0), vec2<i32>(0, -1), vec2<i32>(0, 1),
    vec2<i32>(-1, -1), vec2<i32>(1, -1), vec2<i32>(-1, 1), vec2<i32>(1, 1),
);

// The planet's height (m) at cell (x, y), from the analytic terrain.
@compute @workgroup_size(8, 8)
fn cs_erosion_init(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = er_cell(gid);
    let n = er_n();
    if (c.x >= n || c.y >= n) {
        return;
    }
    let tp = terrain_params(ep.terrain.x, ep.terrain.y, ep.planet.x, ep.planet.y, ep.planet.z, ep.planet.w, ep.terrain.z != 0u);
    let cell_km = ep.centre.w * 0.001;
    let off = (vec2<f32>(c) - 0.5 * f32(n) + 0.5) * cell_km;
    let q = normalize(ep.centre.xyz * ep.planet.x + off.x * ep.east.xyz + off.y * ep.north.xyz);
    let lod = cell_km;
    let m = terrain_macro(tp, q, lod);
    let h = terrain_solid(tp, q, m, lod) * 1000.0;
    let i = er_index(c.x, c.y);
    h_in[i] = h;
    h_out[i] = h;
    initial[i] = h;
    area_in[i] = 1.0;
    area_out[i] = 1.0;
}

// Each cell's steepest-descent neighbour (itself in a pit or the sea).
@compute @workgroup_size(8, 8)
fn cs_erosion_receivers(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = er_cell(gid);
    let n = er_n();
    if (c.x >= n || c.y >= n) {
        return;
    }
    let i = er_index(c.x, c.y);
    let h = h_in[i];
    var best = 0.0;
    var r = i;
    if (h > 0.0) {
        for (var k = 0; k < 8; k++) {
            let o = c + ER_OFFSETS[k];
            if (o.x < 0 || o.y < 0 || o.x >= n || o.y >= n) {
                continue;
            }
            let j = er_index(o.x, o.y);
            let drop = (h - h_in[j]) / select(1.0, 1.41421356, k >= 4);
            if (drop > best) {
                best = drop;
                r = j;
            }
        }
    }
    receiver[i] = r;
}

// Drainage area (cells): one's own plus what flows in from upstream.
@compute @workgroup_size(8, 8)
fn cs_erosion_area(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = er_cell(gid);
    let n = er_n();
    if (c.x >= n || c.y >= n) {
        return;
    }
    let i = er_index(c.x, c.y);
    var a = 1.0;
    for (var k = 0; k < 8; k++) {
        let o = c + ER_OFFSETS[k];
        if (o.x < 0 || o.y < 0 || o.x >= n || o.y >= n) {
            continue;
        }
        let j = er_index(o.x, o.y);
        if (receiver[j] == i) {
            a += area_in[j];
        }
    }
    area_out[i] = a;
}

// One time step of incision and diffusion.
@compute @workgroup_size(8, 8)
fn cs_erosion_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = er_cell(gid);
    let n = er_n();
    if (c.x >= n || c.y >= n) {
        return;
    }
    let i = er_index(c.x, c.y);
    let h = h_in[i];
    // The sea and the square's rim hold still (base level, boundary).
    if (h <= 0.0 || c.x == 0 || c.y == 0 || c.x == n - 1 || c.y == n - 1) {
        h_out[i] = h;
        return;
    }
    let cell = ep.centre.w;
    let dt = ep.north.w;
    let r = receiver[i];
    var incision = 0.0;
    if (r != i) {
        let rc = vec2<i32>(i32(r) % n, i32(r) / n);
        let dist = cell * select(1.0, 1.41421356, rc.x != c.x && rc.y != c.y);
        let slope = max(h - h_in[r], 0.0) / dist;
        incision = ep.stream.x * pow(area_out[i] * cell * cell, ep.stream.y) * slope;
    }
    let lap = (h_in[er_index(c.x - 1, c.y)] + h_in[er_index(c.x + 1, c.y)] + h_in[er_index(c.x, c.y - 1)]
        + h_in[er_index(c.x, c.y + 1)] - 4.0 * h) / (cell * cell);
    var hn = h + dt * (ep.stream.z * lap - incision);
    // Never cut below where the water goes.
    if (r != i) {
        hn = max(hn, h_in[r] + 1e-3);
    }
    h_out[i] = hn;
}

// Distance from the shore (cells) of the eroded land, spread from the sea
// a cell a step (area_in/area_out reused, the erosion done): start at 0 in
// the sea (final heights in h_in), unknown on land.
@compute @workgroup_size(8, 8)
fn cs_shore_init(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = er_cell(gid);
    let n = er_n();
    if (c.x >= n || c.y >= n) {
        return;
    }
    let i = er_index(c.x, c.y);
    area_out[i] = select(1e9, 0.0, h_in[i] <= 0.0);
}

@compute @workgroup_size(8, 8)
fn cs_shore_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = er_cell(gid);
    let n = er_n();
    if (c.x >= n || c.y >= n) {
        return;
    }
    let i = er_index(c.x, c.y);
    var d = area_in[i];
    for (var k = 0; k < 8; k++) {
        let o = c + ER_OFFSETS[k];
        if (o.x < 0 || o.y < 0 || o.x >= n || o.y >= n) {
            continue;
        }
        d = min(d, area_in[er_index(o.x, o.y)] + select(1.0, 1.41421356, k >= 4));
    }
    area_out[i] = d;
}

// The height change (km), faded out towards the square's edges, and the
// distance from the shore (m, up to 2 km). Run with the group whose h_out
// holds the final heights and area_in the shore distances.
@compute @workgroup_size(8, 8)
fn cs_erosion_finish(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = er_cell(gid);
    let n = er_n();
    if (c.x >= n || c.y >= n) {
        return;
    }
    let i = er_index(c.x, c.y);
    let edge = f32(min(min(c.x, c.y), min(n - 1 - c.x, n - 1 - c.y))) / f32(n);
    let change = (h_out[i] - initial[i]) * 0.001 * smoothstep(0.0, 0.1, edge);
    delta[i] = vec2<f32>(change, min(area_in[i] * ep.centre.w, 2000.0));
}
