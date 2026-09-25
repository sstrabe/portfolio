// ---------------------------------------------------------------------------
// The cube-sphere (twin of `terrain/cube.rs`; keep them in step): face f
// has normal n, right r and up u (r × u = n); (u, v) ∈ [−1, 1]² maps to
// normalize(n + tan(πu/4) r + tan(πv/4) u).
// ---------------------------------------------------------------------------

fn cube_axes(face: u32) -> mat3x3<f32> {
    // Columns: normal, right, up.
    switch (face) {
        case 0u: { return mat3x3<f32>(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0)); }
        case 1u: { return mat3x3<f32>(vec3<f32>(-1.0, 0.0, 0.0), vec3<f32>(0.0, -1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0)); }
        case 2u: { return mat3x3<f32>(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(-1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0)); }
        case 3u: { return mat3x3<f32>(vec3<f32>(0.0, -1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0)); }
        case 4u: { return mat3x3<f32>(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(-1.0, 0.0, 0.0)); }
        default: { return mat3x3<f32>(vec3<f32>(0.0, 0.0, -1.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0)); }
    }
}

// Unit direction of face `face` at `uv` (may lie a little outside [−1, 1]).
fn cube_dir(face: u32, uv: vec2<f32>) -> vec3<f32> {
    let m = cube_axes(face);
    let t = tan(0.78539816 * uv);
    return normalize(m[0] + t.x * m[1] + t.y * m[2]);
}

// The face a direction lies on, and its (u, v) there: vec3(face, u, v).
fn cube_face_uv(d: vec3<f32>) -> vec3<f32> {
    let a = abs(d);
    var face = 0u;
    if (a.x >= a.y && a.x >= a.z) {
        face = select(1u, 0u, d.x >= 0.0);
    } else if (a.y >= a.z) {
        face = select(3u, 2u, d.y >= 0.0);
    } else {
        face = select(5u, 4u, d.z >= 0.0);
    }
    let m = cube_axes(face);
    let z = dot(d, m[0]);
    let uv = atan(vec2<f32>(dot(d, m[1]), dot(d, m[2])) / z) / 0.78539816;
    return vec3<f32>(f32(face), uv);
}
