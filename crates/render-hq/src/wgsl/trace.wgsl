// ---------------------------------------------------------------------------
// The trace pass: one invocation per pixel, front to back:
//   1. the ship (moves with the camera),
//   2. the near field (star systems in their rest frames),
//   3. the far field (Kerr geodesic, nebulae, the distant sky).
// Output is linear sRGB radiance (W m⁻² sr⁻¹, CIE weighted) in rgba32float;
// alpha is the near-field transmittance, so point sources splatted later
// are hidden behind planets. When `hq.size.w & HQ_NARROWBAND` is set, the
// three narrowband bins ([S II], Hα, [O III]) also go to `nb_out` for the
// Hubble palette (`post_common.wgsl`).
// ---------------------------------------------------------------------------

@group(0) @binding(3) var hdr_out: texture_storage_2d<rgba32float, write>;
@group(0) @binding(4) var nb_out: texture_storage_2d<rgba32float, write>;

// Escape directions of the workgroup's 8×8 pixels, for the lensed pixel
// footprint on the sky (a compute shader has no derivatives).
var<workgroup> escape_dirs: array<vec4<f32>, 64>;

fn footprint(li: u32, lid: vec2<u32>, own: vec3<f32>) -> f32 {
    var fp = 0.0;
    var n = 0.0;
    let dx = select(li + 1u, li - 1u, lid.x == 7u);
    let dy = select(li + 8u, li - 8u, lid.y == 7u);
    let ex = escape_dirs[dx];
    let ey = escape_dirs[dy];
    if (ex.w > 0.5) {
        fp += dot(ex.xyz - own, ex.xyz - own);
        n += 1.0;
    }
    if (ey.w > 0.5) {
        fp += dot(ey.xyz - own, ey.xyz - own);
        n += 1.0;
    }
    let base = 0.25 * frame.cam.z;
    if (n == 0.0) {
        return frame.cam.z;
    }
    return max(sqrt(2.0 * fp / n), base);
}

@compute @workgroup_size(8, 8)
fn cs_trace(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let inside = gid.x < hq.size.x && gid.y < hq.size.y;
    let px = vec2<f32>(gid.xy) + 0.5 + hq.view.xy;
    let ndc = vec2<f32>(px.x / f32(hq.size.x) * 2.0 - 1.0, 1.0 - px.y / f32(hq.size.y) * 2.0);
    let n = ndc_to_dir(ndc);

    var near = NearResult(medium_clear(), !inside);
    if (inside) {
        let ship = ship_trace(n);
        if (ship.hit) {
            near = NearResult(Medium(ship.L, spec(0.0)), true);
        } else {
            near = near_field(n, frame.cam.z);
        }
    }
    var far: FarTrace;
    far.m = medium_clear();
    far.kind = FAR_LOST;
    if (!near.opaque) {
        far = far_trace(n);
    }
    let escaped = !near.opaque && far.kind == FAR_ESCAPED;
    escape_dirs[li] = vec4<f32>(far.dir, select(0.0, 1.0, escaped));
    workgroupBarrier();
    if (!inside) {
        return;
    }
    var far_l = far.m.L;
    if (escaped) {
        far_l = spec_fma(far.m.T, far_sky(far, footprint(li, lid.xy, far.dir)), far_l);
    }
    let total = spec_fma(near.m.T, far_l, near.m.L);
    textureStore(hdr_out, gid.xy, vec4<f32>(spec_to_rgb(total), spec_mean(near.m.T)));
    if ((hq.size.w & HQ_NARROWBAND) != 0u) {
        // Bins 11 (665–690 nm), 10 (640–665 nm), 4 (490–515 nm).
        textureStore(nb_out, gid.xy, vec4<f32>(total.c.w, total.c.z, total.b.x, 0.0));
    }
}
