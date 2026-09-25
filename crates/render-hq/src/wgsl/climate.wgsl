// ---------------------------------------------------------------------------
// Climate of a planet, baked into a cube map over body-fixed directions
// (`terrain/maps.rs`): annual mean temperature, precipitation, and from
// them the vegetation (Whittaker's biomes). Every texel is independent, so
// the bake is one dispatch.
//
// * Temperature: the surface temperature by latitude (as `planet.wgsl`)
//   less 6.5 K/km of altitude.
// * Precipitation: a latitude profile (the ITCZ's rains, the dry
//   subtropical highs near 27°, the storm tracks near 50°, dry poles)
//   scaled by the moisture the prevailing wind brings: trade winds from the
//   east below 30°, westerlies to 60°, polar easterlies beyond. The bake
//   walks 3000 km upwind: the sea refills the air, land slowly dries it,
//   rising ground rains it out (windward coasts are wet), and air that
//   has crossed mountains is dry (rain shadows: the sunny lee coasts).
// * Vegetation from temperature and rain: none on ice and in deserts,
//   sparse in tundra and dry grassland, dense in forests; `dryness` tints
//   it from green to straw (savanna, steppe).
// ---------------------------------------------------------------------------

struct ClimateParams {
    kind: u32,
    seed: u32,
    n: u32,        // interior texels per face edge
    air: u32,
    radius: f32,
    relief: f32,
    sea: f32,
    t_eq: f32,
}

@group(0) @binding(1) var<uniform> cp: ClimateParams;
@group(0) @binding(2) var climate_out: texture_storage_2d_array<rgba16float, write>;

fn climate_height(tp: TerrainParams, q: vec3<f32>, lod: f32) -> f32 {
    return terrain_macro(tp, normalize(q), lod).x;
}

@compute @workgroup_size(8, 8)
fn cs_climate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = cp.n + 2u;
    if (gid.x >= size || gid.y >= size) {
        return;
    }
    let face = gid.z;
    // Texel x = 1 … n are the face; 0 and n + 1 the apron beyond its edges.
    let uv = ((vec2<f32>(gid.xy) - 0.5) / f32(cp.n)) * 2.0 - 1.0;
    let q = csph_dir(face, uv);
    let tp = terrain_params(cp.kind, cp.seed, cp.radius, cp.relief, cp.sea, cp.t_eq, cp.air != 0u);
    let texel_km = 1.6 * cp.radius / f32(cp.n);
    let h = climate_height(tp, q, texel_km);

    let lat = asin(clamp(q.z, -1.0, 1.0));
    let alat = abs(lat) * 57.29578;
    let greenhouse = select(1.0, 1.13, cp.air != 0u);
    let t_air = tp.t_eq * greenhouse + 18.0 - 45.0 * q.z * q.z - 6.5 * max(h, 0.0);

    // Rain by latitude (mm/yr).
    let p_lat = 350.0 + 2400.0 * exp(-pow(alat / 12.0, 2.0)) + 1000.0 * exp(-pow((alat - 45.0) / 15.0, 2.0))
        - 250.0 * exp(-pow((alat - 27.0) / 7.0, 2.0)) - 250.0 * smoothstep(65.0, 85.0, alat);

    // The wind comes from the east in the tropics and polar caps, from the
    // west in between; walk upwind from 3000 km away to here.
    let east = normalize(cross(vec3<f32>(0.0, 0.0, 1.0), q) + vec3<f32>(1e-6, 0.0, 0.0));
    let from_east = alat < 30.0 || alat > 60.0;
    let upwind = select(-east, east, from_east);
    let step_km = 150.0;
    var moisture = 1.0;
    var last_h = 0.0;
    // Steps since the air left the sea: onshore maritime air rains on the
    // coast it reaches (the trade-wind coasts are wet).
    var inland = 20.0;
    for (var k = 20; k >= 1; k--) {
        let p = normalize(q + upwind * (f32(k) * step_km / cp.radius));
        let hk = climate_height(tp, p, step_km);
        if (hk < 0.0 && tp.liquid == FILL_WATER) {
            moisture = min(1.0, moisture + 0.35);
            last_h = 0.0;
            inland = 0.0;
        } else {
            inland += 1.0;
            let lift = max(hk - last_h, 0.0);
            let wet = moisture * min(1.0, lift / 1.2) * 0.6;
            // Land dries the air slowly (plants give back much of the rain).
            moisture = (moisture - wet) * 0.985;
            last_h = max(hk, 0.0);
        }
    }
    // Rising ground here rains out what the air carries (windward slopes).
    let lift_here = max(max(h, 0.0) - last_h, 0.0);
    let orographic = min(lift_here / 0.8, 1.5);
    let ocean = h < 0.0 && tp.liquid == FILL_WATER;
    let maritime = 800.0 * moisture * exp(-inland / 2.5) * smoothstep(70.0, 40.0, alat);
    var rain = select(p_lat * moisture * (1.0 + 2.5 * orographic) + maritime, p_lat + 800.0 * smoothstep(70.0, 40.0, alat), ocean);
    rain = max(rain, 30.0);

    // Vegetation (Whittaker, simplified): cold or dry limits it.
    let t_c = t_air - 273.15;
    let warmth = smoothstep(-8.0, 6.0, t_c);
    let water = smoothstep(100.0, 700.0, rain);
    // Over the sea these are what a shore at sea level would have, so
    // coasts don't blend towards bare ground.
    let veg = warmth * water;
    // Dryness: straw-coloured grass where rain is scarce or seasonal.
    let dryness = 1.0 - smoothstep(500.0, 1600.0, rain);
    textureStore(climate_out, vec2<i32>(gid.xy), i32(face), vec4<f32>(t_air, rain, veg, dryness));
}
