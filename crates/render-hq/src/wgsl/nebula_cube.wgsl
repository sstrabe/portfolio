// ---------------------------------------------------------------------------
// The nebulae seen from inside the cluster, cached in a cube map.
//
// From the cluster (within `neb.sca2.z` of the hole) every structure is
// ≳ 1 pc away while the ship moves by thousandths of a parsec, so the march
// of `nebula_escape` depends only on the direction. It is run once per
// texel of a cube map centred on the ship, storing the static-frame
// accumulation (`NebAcc`: line intensities, continuum anchors, dust depth,
// velocity moments) so each pixel still applies its own Doppler shift in
// `neb_assemble`. The map is rebuilt a band of rows per frame when the
// ship has moved appreciably (`nebula.rs`).
//
// Compiled with the trace sources (for `neb_march`); its own outputs are in
// group 6.
// ---------------------------------------------------------------------------

struct CubeGen {
    centre: vec4<f32>,   // ship position (M, relative to the hole)
    rows: vec4<u32>,     // face, first row, rows, face size
}

@group(6) @binding(0) var<uniform> cube_gen: CubeGen;
@group(6) @binding(1) var cube_out_a: texture_storage_2d_array<rgba32float, write>;
@group(6) @binding(2) var cube_out_b: texture_storage_2d_array<rgba32float, write>;
@group(6) @binding(3) var cube_out_c: texture_storage_2d_array<rgba32float, write>;
@group(6) @binding(4) var cube_out_d: texture_storage_2d_array<rgba32float, write>;

// Direction of texel (s, t) ∈ [0, 1]² on cube face `face` (WebGPU order).
fn neb_cube_dir(face: u32, st: vec2<f32>) -> vec3<f32> {
    let sc = st.x * 2.0 - 1.0;
    let tc = st.y * 2.0 - 1.0;
    switch (face) {
        case 0u: { return normalize(vec3<f32>(1.0, -tc, -sc)); }
        case 1u: { return normalize(vec3<f32>(-1.0, -tc, sc)); }
        case 2u: { return normalize(vec3<f32>(sc, 1.0, tc)); }
        case 3u: { return normalize(vec3<f32>(sc, -1.0, -tc)); }
        case 4u: { return normalize(vec3<f32>(sc, -tc, 1.0)); }
        default: { return normalize(vec3<f32>(-sc, -tc, -1.0)); }
    }
}

@compute @workgroup_size(8, 8)
fn cs_neb_cube(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = cube_gen.rows.w;
    let row = cube_gen.rows.y + gid.y;
    if (gid.x >= size || gid.y >= cube_gen.rows.z || row >= size) {
        return;
    }
    let face = cube_gen.rows.x;
    let dir = neb_cube_dir(face, (vec2<f32>(f32(gid.x), f32(row)) + 0.5) / f32(size));
    // Two marches with independent jitter per texel: the cube is static,
    // so unlike the per-pixel march no TAA averages its sampling pattern,
    // and one shared offset would draw the voxel grid's planes around the
    // ship as lines across the sky.
    let h = hash3(vec3<u32>(gid.x, row, face * 977u + 13u));
    let m0 = neb_march(cube_gen.centre.xyz, dir, 0.0, 3.0e38, h.x);
    let m1 = neb_march(cube_gen.centre.xyz, dir, 0.0, 3.0e38, h.y);
    var acc: NebAcc;
    acc.la = 0.5 * (m0.la + m1.la);
    acc.lb = 0.5 * (m0.lb + m1.lb);
    acc.cont = 0.5 * (m0.cont + m1.cont);
    acc.mv = 0.5 * (m0.mv + m1.mv);
    acc.tau = 0.5 * (m0.tau + m1.tau);
    let at = vec2<u32>(gid.x, row);
    textureStore(cube_out_a, at, face, acc.la);
    textureStore(cube_out_b, at, face, vec4<f32>(acc.lb, acc.tau, acc.mv.x));
    textureStore(cube_out_c, at, face, acc.cont);
    textureStore(cube_out_d, at, face, vec4<f32>(acc.mv.yz, 0.0, 0.0));
}
