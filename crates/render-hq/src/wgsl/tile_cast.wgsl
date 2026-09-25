// ---------------------------------------------------------------------------
// Rays against the terrain's TLAS (`terrain/rt.rs`), read back by the CPU:
// tests now, ground queries (landing, walking) later. Positions are km
// from the noise anchor in body-fixed axes, as the TLAS places the tiles.
// ---------------------------------------------------------------------------

struct CastParams {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(1) var<uniform> cast_params: CastParams;
// Per ray: (origin, t_max), (direction, unused).
@group(0) @binding(2) var<storage, read> cast_rays: array<vec4<f32>>;
// Per ray: (t or −1 for a miss, layer, primitive, barycentric u as bits),
// (barycentric v, unused …).
@group(0) @binding(3) var<storage, read_write> cast_out: array<vec4<f32>>;
@group(0) @binding(4) var terrain_tlas: acceleration_structure;

@compute @workgroup_size(64)
fn cs_tile_cast(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= cast_params.count) {
        return;
    }
    let a = cast_rays[2u * i];
    let b = cast_rays[2u * i + 1u];
    var rq: ray_query;
    rayQueryInitialize(&rq, terrain_tlas, RayDesc(RAY_FLAG_FORCE_OPAQUE, 0xffu, 0.0, a.w, a.xyz, b.xyz));
    rayQueryProceed(&rq);
    let hit = rayQueryGetCommittedIntersection(&rq);
    if (hit.kind == RAY_QUERY_INTERSECTION_NONE) {
        cast_out[2u * i] = vec4<f32>(-1.0, 0.0, 0.0, 0.0);
        cast_out[2u * i + 1u] = vec4<f32>(0.0);
        return;
    }
    cast_out[2u * i] = vec4<f32>(hit.t, bitcast<f32>(hit.instance_custom_data), bitcast<f32>(hit.primitive_index), hit.barycentrics.x);
    cast_out[2u * i + 1u] = vec4<f32>(hit.barycentrics.y, 0.0, 0.0, 0.0);
}
