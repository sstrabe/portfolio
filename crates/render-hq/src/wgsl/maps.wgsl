// ---------------------------------------------------------------------------
// Baked planet maps in the trace (see `terrain/maps.rs`): one planet at a
// time has them (`Planet.detail.y` = 1). Cube maps over body-fixed
// directions with a one-texel apron, sampled bilinearly.
// ---------------------------------------------------------------------------

@group(1) @binding(2) var climate_map: texture_2d_array<f32>;
@group(1) @binding(3) var map_sampler: sampler;

// Must match `CLIMATE_N` in `terrain/maps.rs`.
const CLIMATE_N: f32 = 512.0;

// (temperature K, precipitation mm/yr, vegetation 0–1, dryness 0–1) at
// body-fixed direction q.
fn map_climate(q: vec3<f32>) -> vec4<f32> {
    let f = csph_face_uv(q);
    let uv = ((f.yz * 0.5 + 0.5) * CLIMATE_N + 1.0) / (CLIMATE_N + 2.0);
    return textureSampleLevel(climate_map, map_sampler, uv, i32(f.x + 0.5), 0.0);
}
