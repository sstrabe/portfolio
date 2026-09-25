// ---------------------------------------------------------------------------
// Terrain tiles traced by the ray-tracing hardware (`terrain/rt.rs`,
// `terrain/field.rs`), for the planet that has them.
//
// The TLAS places the tiles relative to the noise anchor in the planet's
// body-fixed axes, so a ray is taken there from the eye's offset, which the
// CPU computes in f64: the f32 planet centre alone is ~0.5 m coarse on an
// Earth. The sea is the exact sphere at the datum, solved from the eye's
// altitude. At a hit the atlas gives the height and material channels at
// full resolution, and the normal comes from its slopes along the
// triangle's own ground directions.
// ---------------------------------------------------------------------------

struct TerrainView {
    eye: vec4<f32>,     // the pilot relative to the anchor (body-fixed km), altitude above the datum (km)
    up: vec4<f32>,      // body-fixed unit up at the pilot, planet radius (km)
    anchor: vec4<f32>,  // body-fixed unit direction of the anchor, its distance from the centre (km)
    ids: vec4<u32>,     // planet slot with tiles (0xffffffff: none), unused, unused, unused
}

@group(1) @binding(4) var<uniform> terrain_view: TerrainView;
@group(1) @binding(5) var tile_heights: texture_2d_array<f32>;
@group(1) @binding(6) var tile_materials: texture_2d_array<u32>;
@group(1) @binding(7) var terrain_tlas: acceleration_structure;
@group(1) @binding(8) var<storage, read> terrain_mesh: array<vec4<f32>>;

// Must match `terrain/tilegen.rs` and `terrain/rt.rs`.
const TR_SAMPLES: f32 = 128.0;
const TR_APRON: f32 = 2.0;
const TR_TEXELS: f32 = 133.0;
const TR_MESH_N: u32 = 64u;
const TR_MESH_VERTS: u32 = 4485u;
const TR_GRID_TRIANGLES: u32 = 8192u;

fn terrain_on(p: Planet) -> bool {
    return terrain_view.ids.x == p.ids.w;
}

// Bilinear height (km) at texel coordinates `x` of a layer.
fn tr_height(layer: u32, x: vec2<f32>) -> f32 {
    let c = clamp(x, vec2<f32>(0.0), vec2<f32>(TR_TEXELS - 1.001));
    let b = vec2<i32>(floor(c));
    let f = c - floor(c);
    let h00 = textureLoad(tile_heights, b, layer, 0).x;
    let h10 = textureLoad(tile_heights, b + vec2<i32>(1, 0), layer, 0).x;
    let h01 = textureLoad(tile_heights, b + vec2<i32>(0, 1), layer, 0).x;
    let h11 = textureLoad(tile_heights, b + vec2<i32>(1, 1), layer, 0).x;
    return mix(mix(h00, h10, f.x), mix(h01, h11, f.x), f.y);
}

// Bilinear material channels (mountainousness, moisture, rock, land).
fn tr_material(layer: u32, x: vec2<f32>) -> vec4<f32> {
    let c = clamp(x, vec2<f32>(0.0), vec2<f32>(TR_TEXELS - 1.001));
    let b = vec2<i32>(floor(c));
    let f = c - floor(c);
    let m00 = unpack4x8unorm(textureLoad(tile_materials, b, layer, 0).x);
    let m10 = unpack4x8unorm(textureLoad(tile_materials, b + vec2<i32>(1, 0), layer, 0).x);
    let m01 = unpack4x8unorm(textureLoad(tile_materials, b + vec2<i32>(0, 1), layer, 0).x);
    let m11 = unpack4x8unorm(textureLoad(tile_materials, b + vec2<i32>(1, 1), layer, 0).x);
    return mix(mix(m00, m10, f.x), mix(m01, m11, f.x), f.y);
}

// Where a hit lies in its tile, as fractions (s, t) (twin of `rt::hit_st`;
// a skirt hit maps to its edge).
fn tr_hit_st(prim: u32, bary: vec2<f32>) -> vec2<f32> {
    let n = f32(TR_MESH_N);
    if (prim < TR_GRID_TRIANGLES) {
        let quad = prim / 2u;
        let ij = vec2<f32>(f32(quad % TR_MESH_N), f32(quad / TR_MESH_N));
        let uv = select(vec2<f32>(bary.x, bary.x + bary.y), vec2<f32>(bary.x + bary.y, bary.y), prim % 2u == 0u);
        return (ij + uv) / n;
    }
    let k = prim - TR_GRID_TRIANGLES;
    let edge = k / (2u * TR_MESH_N);
    let along = (f32((k % (2u * TR_MESH_N)) / 2u) + 0.5) / n;
    switch (edge) {
        case 0u: { return vec2<f32>(0.0, along); }
        case 1u: { return vec2<f32>(1.0, along); }
        case 2u: { return vec2<f32>(along, 0.0); }
        default: { return vec2<f32>(along, 1.0); }
    }
}

// The ray o + u dir (planet-centred inertial km) against the tiles and
// the sea; `hit` false when neither is met within `u_max`.
fn terrain_surface_hit(p: Planet, o: vec3<f32>, dir: vec3<f32>, u_max: f32, fp: f32, scale: f32) -> SurfaceHit {
    var h: SurfaceHit;
    h.hit = false;
    let tv = terrain_view;
    let tp = planet_terrain(p);
    let bdir = normalize(planet_body(p, dir, 0.0));
    let r = tv.up.w;
    let alt = tv.eye.w;

    // The sea: |(r + alt) up + u d| = r, without forming (r + alt)².
    var u_sea = 3.0e38;
    if (tp.liquid == FILL_WATER && alt > 0.0) {
        let b = (r + alt) * dot(tv.up.xyz, bdir);
        let disc = b * b - alt * (2.0 * r + alt);
        if (disc >= 0.0 && b < 0.0) {
            // The nearer root, without cancellation.
            u_sea = alt * (2.0 * r + alt) / (sqrt(disc) - b);
        }
    }
    var rq: ray_query;
    rayQueryInitialize(&rq, terrain_tlas, RayDesc(RAY_FLAG_FORCE_OPAQUE, 0xffu, 0.0, u_max, tv.eye.xyz, bdir));
    rayQueryProceed(&rq);
    let hit = rayQueryGetCommittedIntersection(&rq);
    let land = hit.kind != RAY_QUERY_INTERSECTION_NONE;
    let u_land = select(3.0e38, hit.t, land);
    let u = min(u_land, u_sea);
    // Neither met (u_max can be unbounded for rays passing the planet).
    if (u >= 3.0e38 || u > u_max) {
        return h;
    }
    h.hit = true;
    h.u = u;
    h.time = -u / scale;
    h.pos = o + u * dir;
    let up = normalize(tv.anchor.xyz * tv.anchor.w + (tv.eye.xyz + u * bdir));
    h.body = up;
    let graze = max(abs(dot(bdir, up)), 0.08);
    h.lod = max(fp * u / graze, 1e-6);
    h.normal = normalize(planet_inertial(p, up, h.time));

    if (u_sea <= u_land) {
        // Water, as deep as the ray finds the floor below it (straight down
        // from the slant), or the planet's terrain where no tile is under.
        if (land) {
            h.height = -(u_land - u_sea) * max(abs(dot(bdir, up)), 0.02);
        } else {
            h.height = min(planet_height(tp, up, h.lod).x, -1e-3);
        }
        h.terrain = vec4<f32>(h.height, 0.0, 0.5, 0.0);
        return h;
    }

    let layer = hit.instance_custom_data;
    let st = tr_hit_st(hit.primitive_index, hit.barycentrics);
    let x = st * TR_SAMPLES + TR_APRON;
    h.height = tr_height(layer, x);
    let m = tr_material(layer, x);
    h.terrain = vec4<f32>(h.height, m.x, m.y, m.z);

    // Ground directions and sample spacing from the mesh quad under the
    // hit, slopes (km per sample) from the full-resolution heights.
    let q = min(vec2<u32>(st * f32(TR_MESH_N)), vec2<u32>(TR_MESH_N - 1u));
    let row = TR_MESH_N + 1u;
    let base = layer * TR_MESH_VERTS + q.y * row + q.x;
    let v00 = terrain_mesh[base].xyz;
    let ds = terrain_mesh[base + 1u].xyz - v00;
    let dt = terrain_mesh[base + row].xyz - v00;
    let step = f32(TR_SAMPLES) / f32(TR_MESH_N);
    let ts = (ds - dot(ds, up) * up) / step;
    let tt = (dt - dot(dt, up) * up) / step;
    let dhs = 0.5 * (tr_height(layer, x + vec2<f32>(1.0, 0.0)) - tr_height(layer, x - vec2<f32>(1.0, 0.0)));
    let dht = 0.5 * (tr_height(layer, x + vec2<f32>(0.0, 1.0)) - tr_height(layer, x - vec2<f32>(0.0, 1.0)));
    var nb = normalize(cross(ts + dhs * up, tt + dht * up));
    if (dot(nb, up) < 0.0) {
        nb = -nb;
    }
    h.normal = normalize(planet_inertial(p, nb, h.time));
    return h;
}
