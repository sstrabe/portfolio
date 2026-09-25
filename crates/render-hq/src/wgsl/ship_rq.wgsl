// ---------------------------------------------------------------------------
// The ship's mesh traced by the GPU's ray-tracing hardware: a BLAS in a
// one-instance TLAS, built once in `ship/mod.rs`. Same answers as the
// software BVH in `ship_bvh.wgsl` (nearest hit, or any hit for shadows;
// the triangle index and barycentrics address the same buffers).
// ---------------------------------------------------------------------------

@group(4) @binding(4) var ship_tlas: acceleration_structure;

fn ship_bvh(o: vec3<f32>, d: vec3<f32>, t_max: f32, any: bool) -> ShipRay {
    var rq: ray_query;
    let flags = select(RAY_FLAG_FORCE_OPAQUE, RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_TERMINATE_ON_FIRST_HIT, any);
    rayQueryInitialize(&rq, ship_tlas, RayDesc(flags, 0xffu, 1e-4, t_max, o, d));
    rayQueryProceed(&rq);
    let hit = rayQueryGetCommittedIntersection(&rq);
    if (hit.kind == RAY_QUERY_INTERSECTION_NONE) {
        return ShipRay(t_max, 0xffffffffu, vec2<f32>(0.0));
    }
    return ShipRay(hit.t, hit.primitive_index, hit.barycentrics);
}
