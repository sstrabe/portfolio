// ---------------------------------------------------------------------------
// Terrain tile generation (see `terrain/tilegen.rs`): one tile per dispatch
// into a layer of the tile atlas, (128 + 1 + 2·2)² texels: sample points
// s = i/128 (i = 0 … 128, edges shared with the neighbours) and a two-texel
// apron.
//
// * Coarse tiles (level < TILE_REFINE_FROM) evaluate the planet's terrain
//   (`terrain_macro`, `terrain_solid`) at their sample directions, with
//   detail down to wavelengths of four samples.
// * Finer tiles refine their parent: its heights upsampled (Catmull-Rom),
//   plus one octave of anchored noise at this level's wavelength (four
//   samples), so each level adds exactly the detail its samples can hold.
//   Positions come from the tile's expansion about its centre, relative to
//   the anchor (`TileFrame`): small numbers only, exact to ~1 mm.
//
// A tile's edges lie on its parent's sample lines, where the cubic reduces
// to one dimension along the line, so neighbouring tiles agree on their
// shared edges (across cube faces too).
// ---------------------------------------------------------------------------

struct TileGenParams {
    kind: u32,
    seed: u32,
    air: u32,
    octave_count: u32,
    radius: f32,
    relief: f32,
    sea: f32,
    t_eq: f32,
    // One per refined level, from TILE_REFINE_FROM.
    octaves: array<AnchorOctave, 16>,
    // Boulders (`boulders.wgsl`): the level that raises them, their cells'
    // level, the hash seed (bits), unused.
    boulders: vec4<f32>,
}

struct TileFrame {
    origin: vec4<f32>,
    a_s: vec4<f32>,
    a_t: vec4<f32>,
    b_ss: vec4<f32>,
    b_st: vec4<f32>,
    b_tt: vec4<f32>,
}

struct TileJob {
    frame: TileFrame,
    tile: vec4<u32>,    // face, level, x, y
    slots: vec4<u32>,   // layer, parent's layer, 1 to refine the parent, unused
    centre: vec4<f32>,  // unit direction of the tile's centre (body fixed), planet radius (km)
}

// Must match `terrain/tilegen.rs`.
const TILE_SAMPLES: f32 = 128.0;
const TILE_APRON: i32 = 2;
const TILE_TEXELS: u32 = 133u;
const TILE_REFINE_FROM: u32 = 12u;

@group(0) @binding(1) var<uniform> tg: TileGenParams;
@group(0) @binding(2) var<uniform> job: TileJob;
@group(0) @binding(3) var tile_height: texture_storage_2d_array<r32float, read_write>;
@group(0) @binding(4) var tile_material: texture_storage_2d_array<r32uint, read_write>;
// Per layer: the lowest and highest height inside the tile, in mm.
@group(0) @binding(5) var<storage, read_write> tile_range: array<atomic<i32>>;
// Mesh vertices for the ray-tracing hardware (see `terrain/rt.rs`):
// TILE_MESH_VERTS per layer, xyz in km relative to the tile's centre point
// on the datum sphere, body-fixed axes.
@group(0) @binding(6) var<storage, read_write> tile_vertices: array<vec4<f32>>;

// Must match `terrain/rt.rs`.
const TILE_MESH_STEP: i32 = 2;
const TILE_MESH_N: u32 = 64u;
const TILE_MESH_VERTS: u32 = 4485u;

// The regional erosion (`region.wgsl`, `terrain/erosion.rs`).
@group(0) @binding(7) var<uniform> region: Region;
@group(0) @binding(8) var<storage, read> region_cells: array<vec2<f32>>;

// Nominal sample spacing (km) at a level.
fn tile_spacing(level: u32) -> f32 {
    return tg.radius * 1.5707963 / (f32(1u << level) * TILE_SAMPLES);
}

// The point at centred tile coordinates (s, t), km from the anchor.
fn tile_offset(f: TileFrame, s: f32, t: f32) -> vec3<f32> {
    return f.origin.xyz + (f.a_s.xyz * s + f.a_t.xyz * t)
        + (0.5 * f.b_ss.xyz * s * s + f.b_st.xyz * s * t + 0.5 * f.b_tt.xyz * t * t);
}

// Catmull-Rom weights for the taps at −1, 0, 1, 2 (exactly (0, 1, 0, 0) at
// t = 0).
fn tile_cubic(t: f32) -> vec4<f32> {
    let t2 = t * t;
    let t3 = t2 * t;
    return vec4<f32>(
        -0.5 * t3 + t2 - 0.5 * t,
        1.5 * t3 - 2.5 * t2 + 1.0,
        -1.5 * t3 + 2.0 * t2 + 0.5 * t,
        0.5 * t3 - 0.5 * t2,
    );
}

fn tile_texel(c: vec2<i32>) -> vec2<i32> {
    return clamp(c, vec2<i32>(0), vec2<i32>(i32(TILE_TEXELS) - 1));
}

// The parent's height at texel coordinates `p` (texel k at p = k).
fn parent_height(layer: u32, p: vec2<f32>) -> f32 {
    let base = floor(p);
    let wx = tile_cubic(p.x - base.x);
    let wy = tile_cubic(p.y - base.y);
    let b = vec2<i32>(base);
    var h = 0.0;
    for (var j = 0; j < 4; j++) {
        if (wy[j] == 0.0) {
            continue;
        }
        var row = 0.0;
        for (var i = 0; i < 4; i++) {
            if (wx[i] != 0.0) {
                row += wx[i] * textureLoad(tile_height, tile_texel(b + vec2<i32>(i - 1, j - 1)), layer).x;
            }
        }
        h += wy[j] * row;
    }
    return h;
}

// Gullies: channels running down the slope. Each lattice cell near the
// point (the 2×2×2 nearest, anchored) has a jittered centre that lays a
// stripe pattern across the downhill direction, faded with distance, so
// every level carves rills into its parent's slopes (and, as the parent's
// slopes hold the coarser gullies, they branch along them). `across` is a
// unit vector along the ground, square to the slope. Roughly −1 to 1.
fn tile_gullies(o: AnchorOctave, d: vec3<f32>, across: vec3<f32>) -> f32 {
    let l = anc_lattice(o, d);
    let seed = bitcast<u32>(o.cell.w) ^ 0x9e3779b9u;
    let base = vec3<i32>(floor(l.frac - 0.5));
    var sum = 0.0;
    var wsum = 0.0;
    for (var k = 0; k < 8; k++) {
        let c = base + vec3<i32>(k & 1, (k >> 1) & 1, (k >> 2) & 1);
        let hsh = anc_pcg3d(bitcast<vec3<u32>>(l.cell + c) ^ vec3<u32>(seed, seed * 3u, seed * 7u));
        let jitter = vec3<f32>(hsh >> vec3<u32>(8u)) / 16777216.0;
        let off = l.frac - (vec3<f32>(c) + jitter);
        let w = max(1.0 - dot(off, off) / 2.25, 0.0);
        sum += w * w * cos(6.2831853 * dot(off, across));
        wsum += w * w;
    }
    return sum / max(wsum, 1e-4);
}

// The parent's material channels at texel coordinates `p` (bilinear).
fn parent_material(layer: u32, p: vec2<f32>) -> vec4<f32> {
    let base = floor(p);
    let f = p - base;
    let b = vec2<i32>(base);
    let m00 = unpack4x8unorm(textureLoad(tile_material, tile_texel(b), layer).x);
    let m10 = unpack4x8unorm(textureLoad(tile_material, tile_texel(b + vec2<i32>(1, 0)), layer).x);
    let m01 = unpack4x8unorm(textureLoad(tile_material, tile_texel(b + vec2<i32>(0, 1)), layer).x);
    let m11 = unpack4x8unorm(textureLoad(tile_material, tile_texel(b + vec2<i32>(1, 1)), layer).x);
    return mix(mix(m00, m10, f.x), mix(m01, m11, f.x), f.y);
}

@compute @workgroup_size(8, 8)
fn cs_tile_gen(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= TILE_TEXELS || gid.y >= TILE_TEXELS) {
        return;
    }
    let tp = terrain_params(tg.kind, tg.seed, tg.radius, tg.relief, tg.sea, tg.t_eq, tg.air != 0u);
    let level = job.tile.y;
    let layer = job.slots.x;
    // Sample index: 0 … 128 inside the tile.
    let ij = vec2<i32>(gid.xy) - TILE_APRON;
    let st = vec2<f32>(ij) / TILE_SAMPLES;
    let spacing = tile_spacing(level);
    var h: f32;
    // Material: mountainousness, moisture, rock, land (see `terrain_macro`).
    var m: vec4<f32>;
    if (job.slots.z == 0u) {
        // Coarse: the planet's terrain, down to four-sample wavelengths.
        let uv = -1.0 + 2.0 * (vec2<f32>(job.tile.zw) + st) / f32(1u << level);
        let q = csph_dir(job.tile.x, uv);
        let lod = 2.0 * spacing;
        let mac = terrain_macro(tp, q, lod);
        // The planet's terrain, worn by the regional erosion where it has run.
        h = terrain_solid(tp, q, mac, lod) + region_delta(q, tg.seed, tg.radius, lod);
        m = vec4<f32>(saturate(mac.y), saturate(mac.z), saturate(mac.w), smoothstep(-0.3, 0.2, mac.x));
    } else {
        // Refine: the parent upsampled, plus this level's octave.
        let quadrant = vec2<f32>(job.tile.zw & vec2<u32>(1u)) * 0.5 * TILE_SAMPLES;
        let p = quadrant + 0.5 * vec2<f32>(ij) + f32(TILE_APRON);
        h = parent_height(job.slots.y, p);
        m = parent_material(job.slots.y, p);
        let d = tile_offset(job.frame, st.x - 0.5, st.y - 0.5);
        let o = tg.octaves[min(level - TILE_REFINE_FROM, 15u)];
        // The parent's slope here (km per km) along the ground, from its
        // heights a sample either side (the same texels a neighbouring
        // tile reads, so shared edges still agree).
        let f = job.frame;
        let dh = vec2<f32>(
            parent_height(job.slots.y, p + vec2<f32>(0.5, 0.0)) - parent_height(job.slots.y, p - vec2<f32>(0.5, 0.0)),
            parent_height(job.slots.y, p + vec2<f32>(0.0, 0.5)) - parent_height(job.slots.y, p - vec2<f32>(0.0, 0.5)),
        ) / (2.0 * spacing);
        let up = normalize(job.centre.xyz * job.centre.w + (d - f.origin.xyz));
        let slope_dir = dh.x * normalize(f.a_s.xyz) + dh.y * normalize(f.a_t.xyz);
        let slope = length(slope_dir);
        // Slopes stay roughly constant from octave to octave (as in
        // `terrain_detail`): ridged ranges rough, lowlands gentle.
        let k = tp.relief / 16.0;
        var rough = (mix(0.12, 3.0, m.x * m.x) * m.w + 0.06) * k * (4.0 * spacing / tp.split);
        if (tp.liquid == FILL_WATER) {
            // Beaches and the shelf stay smooth (the waves keep them so;
            // the sand's own texture gives the centimetres).
            rough *= mix(0.1, 1.0, smoothstep(0.004, 0.015, abs(h - 0.0015)));
        }
        // On slopes the detail runs in gullies down them; on the level it
        // stays plain noise.
        var detail = anchored_noise(o, d);
        let erode = smoothstep(0.04, 0.3, slope) * smoothstep(0.0, 0.02, h);
        if (erode > 0.0) {
            let across = normalize(cross(up, slope_dir));
            detail = mix(detail, 1.4 * tile_gullies(o, d, across), erode);
        }
        h += rough * detail;
        // Boulders, raised once, at the level that resolves them.
        if (level == u32(tg.boulders.x)) {
            let cell_level = u32(tg.boulders.y);
            let cell_km = tg.radius * 1.5707963 / f32(1u << cell_level);
            h += boulders_at(job.tile, st, cell_level, cell_km, boulder_density(m), bitcast<u32>(tg.boulders.z)).x;
        }
    }
    h = clamp(h, tp.lo, tp.hi);
    textureStore(tile_height, gid.xy, layer, vec4<f32>(h));
    textureStore(tile_material, gid.xy, layer, vec4<u32>(pack4x8unorm(m)));
    if (all(ij >= vec2<i32>(0)) && all(ij <= vec2<i32>(i32(TILE_SAMPLES)))) {
        let mm = i32(round(h * 1e6));
        atomicMin(&tile_range[2u * layer], mm);
        atomicMax(&tile_range[2u * layer + 1u], mm);
    }
}

// The tile's point at mesh vertex (i, j) (every TILE_MESH_STEP samples),
// relative to its centre point on the datum sphere, and the local up.
fn tile_mesh_point(i: u32, j: u32) -> array<vec3<f32>, 2> {
    let level = job.tile.y;
    let st = vec2<f32>(f32(i), f32(j)) / f32(TILE_MESH_N);
    let r = job.centre.w;
    let c = job.centre.xyz * r;
    var local: vec3<f32>;
    if (job.slots.z == 0u) {
        // Coarse tiles: straight from the cube (km-sized triangles; f32's
        // ~0.5 m here is plenty).
        let n = f32(1u << level);
        let uv = -1.0 + 2.0 * (vec2<f32>(job.tile.zw) + st) / n;
        local = r * csph_dir(job.tile.x, uv) - c;
    } else {
        // Fine tiles: the f64 expansion without its anchor offset.
        let f = job.frame;
        let s = st.x - 0.5;
        let t = st.y - 0.5;
        local = (f.a_s.xyz * s + f.a_t.xyz * t) + (0.5 * f.b_ss.xyz * s * s + f.b_st.xyz * s * t + 0.5 * f.b_tt.xyz * t * t);
    }
    return array<vec3<f32>, 2>(local, normalize(c + local));
}

// The tile's mesh for the BLAS: a (TILE_MESH_N + 1)² grid, then skirts
// hanging below the west, east, south and north edges (so rays can't slip
// between tiles of different levels).
@compute @workgroup_size(8, 8)
fn cs_tile_mesh(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n1 = TILE_MESH_N + 1u;
    if (gid.x >= n1 || gid.y >= n1) {
        return;
    }
    let layer = job.slots.x;
    let texel = vec2<i32>(gid.xy) * TILE_MESH_STEP + TILE_APRON;
    let h = textureLoad(tile_height, texel, layer).x;
    let pu = tile_mesh_point(gid.x, gid.y);
    let v = pu[0] + h * pu[1];
    let base = layer * TILE_MESH_VERTS;
    tile_vertices[base + gid.y * n1 + gid.x] = vec4<f32>(v, 0.0);
    // Skirts: deep enough to cover the steps between levels.
    let skirt = v - 8.0 * tile_spacing(job.tile.y) * f32(TILE_MESH_STEP) * pu[1];
    let grid = n1 * n1;
    if (gid.x == 0u) {
        tile_vertices[base + grid + gid.y] = vec4<f32>(skirt, 0.0);
    }
    if (gid.x == TILE_MESH_N) {
        tile_vertices[base + grid + n1 + gid.y] = vec4<f32>(skirt, 0.0);
    }
    if (gid.y == 0u) {
        tile_vertices[base + grid + 2u * n1 + gid.x] = vec4<f32>(skirt, 0.0);
    }
    if (gid.y == TILE_MESH_N) {
        tile_vertices[base + grid + 3u * n1 + gid.x] = vec4<f32>(skirt, 0.0);
    }
}
