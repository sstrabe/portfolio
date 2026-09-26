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
    ids: vec4<u32>,     // planet slot with tiles (0xffffffff: none); boulders: the level that raises them, their cells' level, seed
    // Anchored noise varying the ground's brightness over metres to tens
    // of metres (`terrain::materials::VARIATION_M`, coarse first).
    variation: array<AnchorOctave, 4>,
}

@group(1) @binding(4) var<uniform> terrain_view: TerrainView;
@group(1) @binding(5) var tile_heights: texture_2d_array<f32>;
@group(1) @binding(6) var tile_materials: texture_2d_array<u32>;
@group(1) @binding(7) var terrain_tlas: acceleration_structure;
@group(1) @binding(8) var<storage, read> terrain_mesh: array<vec4<f32>>;

// The ground's scanned materials (`terrain/materials.rs`), a layer each:
// albedo (sRGB rgb, height in alpha) and detail (normal x, y in OpenGL's
// convention, ambient occlusion, roughness).
struct GroundMaterials {
    // Repeat level, relief (m), the albedo's mean luminance, the height's
    // mean.
    m: array<vec4<f32>, 8>,
}
@group(1) @binding(9) var ground_albedo: texture_2d_array<f32>;
@group(1) @binding(10) var ground_detail: texture_2d_array<f32>;
@group(1) @binding(11) var ground_sampler: sampler;
@group(1) @binding(12) var<uniform> ground_materials: GroundMaterials;
// The tile in each atlas layer: face, level, x, y.
@group(1) @binding(13) var<uniform> layer_tiles: array<vec4<u32>, 1024>;

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
    h.tiled = true;
    h.u = u;
    h.time = -u / scale;
    h.pos = o + u * dir;
    h.local = tv.eye.xyz + u * bdir;
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

    tr_tile_surface(p, &h, hit.instance_custom_data, hit.primitive_index, hit.barycentrics);
    return h;
}

// Height, material channels and normal of a tile hit from the atlas (the
// hit's `body`, `time` and `lod` already set).
fn tr_tile_surface(p: Planet, hp: ptr<function, SurfaceHit>, layer: u32, prim: u32, bary: vec2<f32>) {
    var h = *hp;
    let up = h.body;
    let st = tr_hit_st(prim, bary);
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
    h.layer = layer;
    h.st = st;
    h.jitter = anchored_noise(terrain_view.variation[1], h.local) + 0.6 * anchored_noise(terrain_view.variation[2], h.local);
    // On tiles fine enough to hold them, the boulders there.
    let tile = layer_tiles[layer];
    if (tile.y >= terrain_view.ids.y) {
        let cell_level = terrain_view.ids.z;
        let cell_km = terrain_view.up.w * 1.5707963 / f32(1u << cell_level);
        h.boulder = boulders_at(tile, st, cell_level, cell_km, boulder_density(m), terrain_view.ids.w).y;
    }
    h.axis_s = normalize(ts);
    h.axis_t = normalize(tt);
    *hp = h;
}

// Where a material's texture is at place `st` of `tile` (face, level, x,
// y): one repeat spans a tile of `repeat` level on the face grid, so the
// coordinate is exact and continuous across tiles at any distance.
fn tr_ground_uv(tile: vec4<u32>, st: vec2<f32>, repeat: u32) -> vec2<f32> {
    let level = tile.y;
    if (level >= repeat) {
        let n = 1u << min(level - repeat, 31u);
        let within = vec2<u32>(tile.z, tile.w) & vec2<u32>(n - 1u);
        return (vec2<f32>(within) + st) / f32(n);
    }
    // Whole repeats per tile: the tile's index drops out.
    return st * f32(1u << min(repeat - level, 31u));
}

// The scanned materials' centimetre detail at a tile hit, blended by the
// surface's make-up (`Material.ground`) and by height (the higher texel
// of two materials shows, as rock stands out of sand): brightness, normal
// and occlusion.
fn terrain_micro(p: Planet, h: SurfaceHit, mat: Material) -> Micro {
    var out = Micro(1.0, h.normal, 1.0);
    if (!h.tiled || h.height < 0.0) {
        return out;
    }
    // Patches of lighter and darker ground over metres to tens of metres,
    // stronger where plants grow (their clumps and gaps), faded out where
    // a pixel spans a wavelength.
    var patches = 0.0;
    let amp = array<f32, 4>(0.10, 0.08, 0.07, 0.05);
    for (var k = 0u; k < 4u; k++) {
        let o = terrain_view.variation[k];
        let fade = smoothstep(2.0, 4.0, 1.0 / (o.frac_freq.w * max(h.lod, 1e-7)));
        if (fade > 0.0) {
            patches += fade * amp[k] * anchored_noise(o, h.local);
        }
    }
    out.albedo = exp((1.0 + 2.5 * mat.cover) * patches);
    let tile = layer_tiles[h.layer];
    let radius = terrain_view.up.w;
    var weight_sum = 0.0;
    var bright = 0.0;
    var nts = vec2<f32>(0.0);
    var ao = 0.0;
    for (var i = 0u; i < GM_COUNT; i++) {
        let w = mat.ground[i];
        if (w < 0.02) {
            continue;
        }
        let gm = ground_materials.m[i];
        let repeat = u32(gm.x);
        // One repeat's size, and the mip whose texels match the footprint.
        let span = radius * 1.5707963 / f32(1u << repeat);
        if (h.lod > span) {
            continue;
        }
        let lod = log2(max(h.lod * 1024.0 / span, 1.0));
        // Past the coarse mip the detrended detail is nil.
        if (lod >= 7.0) {
            continue;
        }
        let uv = tr_ground_uv(tile, h.st, repeat);
        let a = textureSampleLevel(ground_albedo, ground_sampler, uv, i, lod);
        let d = textureSampleLevel(ground_detail, ground_sampler, uv, i, lod);
        // The scan's large blotches, from a coarse mip (texels an eighth of
        // a repeat across) are taken out, so its repeats don't show: the
        // texture gives the small scales and the anchored noise the large.
        let coarse = max(lod, 7.0);
        let a_low = textureSampleLevel(ground_albedo, ground_sampler, uv, i, coarse);
        let d_low = textureSampleLevel(ground_detail, ground_sampler, uv, i, coarse);
        let wh = w * exp(4.0 * (a.a - a_low.a));
        let luma = vec3<f32>(0.2126, 0.7152, 0.0722);
        weight_sum += wh;
        bright += wh * dot(a.rgb, luma) / max(dot(a_low.rgb, luma), 1e-3);
        nts += wh * 2.0 * (d.xy - d_low.xy);
        ao += wh * d.z / max(d_low.z, 0.2);
    }
    if (weight_sum <= 0.0) {
        return out;
    }
    // Materials left out (too far to resolve, or slight, or hidden under
    // plants) keep their share of the plain surface.
    let total = max(weight_sum, 1e-6);
    let share = saturate(weight_sum / (weight_sum + max(1.0 - weight_sum, 0.0)));
    out.albedo *= mix(1.0, clamp(bright / total, 0.0, 2.5), share);
    out.ao = mix(1.0, ao / total, share);
    let t = nts / total * share;
    // Tangent frame: the texture's x along +s, its y (up the image) along
    // −t, both on the tile's shading normal.
    let n0 = normalize(planet_body(p, h.normal, h.time));
    let ax = normalize(h.axis_s - dot(h.axis_s, n0) * n0);
    let ay = normalize(-h.axis_t + dot(h.axis_t, n0) * n0);
    let nz = sqrt(max(1.0 - dot(t, t), 0.0));
    let nb = normalize(t.x * ax + t.y * ay + nz * n0);
    out.normal = normalize(planet_inertial(p, nb, h.time));
    return out;
}

// How much of the sun the terrain hides from a tile hit: a ray towards a
// point on the sun's disc, a different one each frame and pixel (TAA
// averages them into penumbrae), stopping at the first tile it meets.
fn terrain_shadow(p: Planet, h: SurfaceHit, sun: SunLight) -> f32 {
    let up = h.body;
    let s = normalize(planet_body(p, sun.dir, h.time));
    if (dot(s, up) < -0.2) {
        return 1.0;
    }
    // A point on the disc, uniformly.
    let seed = tn_hash(bitcast<u32>(h.local.x) ^ tn_hash(bitcast<u32>(h.local.y) ^ tn_hash(bitcast<u32>(h.local.z) ^ hq.size.z)));
    let r1 = f32(seed & 0xffffu) / 65535.0;
    let r2 = f32(seed >> 16u) / 65535.0;
    let t1 = normalize(cross(s, select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), abs(s.z) > 0.9)));
    let t2 = cross(s, t1);
    let a = sun.angular_radius * sqrt(r1);
    let phi = 6.2831853 * r2;
    let d = normalize(s + a * (cos(phi) * t1 + sin(phi) * t2));
    // Off the surface by a little, more for distant hits (f32 positions).
    let o = h.local + up * (1e-5 + 1e-4 * h.u);
    var rq: ray_query;
    rayQueryInitialize(&rq, terrain_tlas, RayDesc(RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_TERMINATE_ON_FIRST_HIT, 0xffu, 0.0, 100.0, o, d));
    rayQueryProceed(&rq);
    let hit = rayQueryGetCommittedIntersection(&rq);
    return select(0.0, 1.0, hit.kind != RAY_QUERY_INTERSECTION_NONE);
}

// The share of the sky the terrain hides from a tile hit: one
// cosine-weighted ray over the shading normal per frame and pixel (TAA
// averages them), out to 1 km (the enclosure close by matters most).
fn terrain_sky_occlusion(p: Planet, h: SurfaceHit) -> f32 {
    let n = normalize(planet_body(p, h.normal, h.time));
    let seed = tn_hash(bitcast<u32>(h.local.z) ^ tn_hash(bitcast<u32>(h.local.x) ^ tn_hash(bitcast<u32>(h.local.y) ^ (hq.size.z * 747796405u))));
    let r1 = f32(seed & 0xffffu) / 65535.0;
    let r2 = f32(seed >> 16u) / 65535.0;
    let t1 = normalize(cross(n, select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), abs(n.z) > 0.9)));
    let t2 = cross(n, t1);
    let rr = sqrt(r1);
    let phi = 6.2831853 * r2;
    let d = normalize(rr * cos(phi) * t1 + rr * sin(phi) * t2 + sqrt(max(1.0 - r1, 0.0)) * n);
    let o = h.local + h.body * (1e-5 + 1e-4 * h.u);
    var rq: ray_query;
    rayQueryInitialize(&rq, terrain_tlas, RayDesc(RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_TERMINATE_ON_FIRST_HIT, 0xffu, 0.0, 1.0, o, d));
    rayQueryProceed(&rq);
    let hit = rayQueryGetCommittedIntersection(&rq);
    return select(0.0, 1.0, hit.kind != RAY_QUERY_INTERSECTION_NONE);
}

// The terrain mirrored by the sea at a tile-sea hit: one reflection ray per
// frame and pixel off a wave facet drawn from the wind's slope
// distribution (Cox & Munk, as `ocean_glint`), so TAA blurs the mirror as
// the waves do. The land it meets is lit by the sun (with its shadow ray
// skipped) and the sky.
fn terrain_reflection(p: Planet, h: SurfaceHit, view: vec3<f32>, sun: SunLight, wind: f32) -> TerrainReflection {
    var out: TerrainReflection;
    out.hit = false;
    if (!h.tiled) {
        return out;
    }
    let up = h.body;
    let v = normalize(planet_body(p, view, h.time));
    let seed = tn_hash(bitcast<u32>(h.local.y) ^ tn_hash(bitcast<u32>(h.local.z) ^ tn_hash(bitcast<u32>(h.local.x) ^ (hq.size.z * 2891336453u))));
    let r1 = max(f32(seed & 0xffffu) / 65535.0, 1e-6);
    let r2 = f32(seed >> 16u) / 65535.0;
    // Gaussian slopes (Box–Muller), variance σ²/2 per axis.
    let sigma = sqrt(0.5 * (0.003 + 5.12e-3 * max(wind, 0.5)));
    let g = sqrt(-2.0 * log(r1)) * vec2<f32>(cos(6.2831853 * r2), sin(6.2831853 * r2)) * sigma;
    let t1 = normalize(cross(up, select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), abs(up.z) > 0.9)));
    let t2 = cross(up, t1);
    let n = normalize(up - g.x * t1 - g.y * t2);
    var r = reflect(-v, n);
    if (dot(r, up) <= 1e-3) {
        r = reflect(-v, up);
    }
    let o = h.local + up * (1e-5 + 1e-4 * h.u);
    var rq: ray_query;
    rayQueryInitialize(&rq, terrain_tlas, RayDesc(RAY_FLAG_FORCE_OPAQUE, 0xffu, 0.0, 50.0, o, r));
    rayQueryProceed(&rq);
    let hit = rayQueryGetCommittedIntersection(&rq);
    if (hit.kind == RAY_QUERY_INTERSECTION_NONE) {
        return out;
    }
    let tv = terrain_view;
    var m: SurfaceHit;
    m.hit = true;
    m.tiled = true;
    m.u = h.u + hit.t;
    m.time = h.time;
    m.local = o + hit.t * r;
    let bpos = tv.anchor.xyz * tv.anchor.w + m.local;
    m.body = normalize(bpos);
    m.pos = planet_inertial(p, bpos, m.time);
    m.lod = max(h.lod, 1e-5);
    tr_tile_surface(p, &m, hit.instance_custom_data, hit.primitive_index, hit.barycentrics);
    let mat = planet_material(planet_terrain(p), m, p.detail.y > 0.5);
    let e_sun = spec_mul(sun.irradiance, atmo_sun_transmittance(p, m.pos, sun));
    let e_sky = atmo_sky_irradiance(p, m.pos, m.normal, sun);
    let nl = max(dot(m.normal, sun.dir), 0.0);
    out.hit = true;
    out.L = spec_mul(mat.albedo, spec_add(spec_scale(e_sun, nl / PI), spec_scale(e_sky, 1.0 / PI)));
    return out;
}
