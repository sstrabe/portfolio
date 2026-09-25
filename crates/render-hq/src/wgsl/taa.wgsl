// ---------------------------------------------------------------------------
// Temporal accumulation and upscaling of the traced radiance, at the output
// resolution.
//
// Every frame the trace samples each of its pixels at a new sub-pixel
// offset (the jitter, a Halton sequence). This pass gathers the 3 × 3
// traced samples around each output pixel, weighted by their distance to
// its centre, and blends them into the history reprojected from the
// previous frame. The camera mostly rotates and the far field lies at
// infinity, so the reprojection is the Lorentz transformation between the
// two frames' tetrads (rotation plus aberration); the near field moves by
// well under a pixel per frame. The history is clipped to the variance box
// of the current neighbourhood, which rejects what reprojection gets
// wrong, and accumulates up to many frames when the view is still
// (progressive refinement). Blending happens on tone-compressed values
// c/(1 + e L) so single blazing samples cannot dominate.
//
// Point sources are not in this image: they are splatted afterwards, sharp,
// at their exact positions, so they never smear or ghost.
// ---------------------------------------------------------------------------

@group(0) @binding(3) var<storage, read> expo: Exposure;
@group(0) @binding(4) var trace_tex: texture_2d<f32>;
@group(0) @binding(5) var nb_tex: texture_2d<f32>;
@group(0) @binding(6) var hist_prev: texture_2d<f32>;
@group(0) @binding(7) var weight_prev: texture_2d<f32>;
@group(0) @binding(8) var lin: sampler;
@group(0) @binding(9) var hist_next: texture_storage_2d<rgba32float, write>;
@group(0) @binding(10) var weight_next: texture_storage_2d<r32float, write>;
@group(0) @binding(11) var scene: texture_storage_2d<rgba32float, write>;

fn compress(c: vec3<f32>, e: f32) -> vec3<f32> {
    return c * e / (1.0 + max(luma(c * e), 0.0));
}

fn expand(c: vec3<f32>, e: f32) -> vec3<f32> {
    return c / (e * max(1.0 - max(luma(c), 0.0), 1e-4));
}

fn traced(p: vec2<i32>) -> vec4<f32> {
    let c = textureLoad(trace_tex, p, 0);
    if (pf.mode.y == 1u) {
        let nb = textureLoad(nb_tex, p, 0);
        return vec4<f32>(nb.xyz * pf.sho.xyz, c.w);
    }
    return c;
}

// Catmull–Rom history lookup from 5 bilinear taps (the corner taps of the
// 4 × 4 footprint contribute little and are dropped).
fn history_at(uv: vec2<f32>) -> vec4<f32> {
    let size = vec2<f32>(pf.sizes.xy);
    let p = uv * size;
    let c = floor(p - 0.5) + 0.5;
    let f = p - c;
    let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    let w3 = f * f * (-0.5 + 0.5 * f);
    let w12 = w1 + w2;
    let t0 = (c - 1.0) / size;
    let t3 = (c + 2.0) / size;
    let t12 = (c + w2 / w12) / size;
    var s = textureSampleLevel(hist_prev, lin, vec2<f32>(t12.x, t0.y), 0.0) * (w12.x * w0.y);
    s += textureSampleLevel(hist_prev, lin, vec2<f32>(t0.x, t12.y), 0.0) * (w0.x * w12.y);
    s += textureSampleLevel(hist_prev, lin, t12, 0.0) * (w12.x * w12.y);
    s += textureSampleLevel(hist_prev, lin, vec2<f32>(t3.x, t12.y), 0.0) * (w3.x * w12.y);
    s += textureSampleLevel(hist_prev, lin, vec2<f32>(t12.x, t3.y), 0.0) * (w12.x * w3.y);
    let w = w12.x * w0.y + w0.x * w12.y + w12.x * w12.y + w3.x * w12.y + w12.x * w3.y;
    return s / w;
}

// Pull `h` toward the box centre until it lies inside the box.
fn clip_box(h: vec3<f32>, lo: vec3<f32>, hi: vec3<f32>) -> vec3<f32> {
    let c = 0.5 * (hi + lo);
    let e = 0.5 * (hi - lo) + 1e-7;
    let d = h - c;
    let u = abs(d / e);
    let m = max(u.x, max(u.y, u.z));
    return select(h, c + d / m, m > 1.0);
}

@compute @workgroup_size(8, 8)
fn cs_taa(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out = pf.sizes.xy;
    if (gid.x >= out.x || gid.y >= out.y) {
        return;
    }
    let rs = vec2<i32>(pf.sizes.zw);
    let r = vec2<f32>(pf.sizes.zw) / vec2<f32>(out);
    let px = vec2<f32>(gid.xy) + 0.5;
    let c = px * r;
    let jit = pf.taa.xy;
    let e = max(expo.value, 1e-30);
    let i0 = vec2<i32>(round(c - 0.5 - jit));

    var sum = vec4<f32>(0.0);
    var wsum = 0.0;
    var m1 = vec3<f32>(0.0);
    var m2 = vec3<f32>(0.0);
    var ship_w = 0.0;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let i = clamp(i0 + vec2<i32>(dx, dy), vec2<i32>(0), rs - 1);
            let s = vec2<f32>(i) + 0.5 + jit;
            // Distance in output pixels; a Blackman–Harris-like Gaussian.
            let d = (s - c) / r;
            let w = exp(-2.29 * dot(d, d));
            let t = traced(i);
            let y = compress(t.rgb, e);
            sum += vec4<f32>(y, max(t.a, 0.0)) * w;
            ship_w += select(0.0, w, t.a < -0.5);
            wsum += w;
            m1 += y;
            m2 += y * y;
        }
    }
    let cur = sum / wsum;
    let mean = m1 / 9.0;
    let sigma = sqrt(max(m2 / 9.0 - mean * mean, vec3<f32>(0.0)));
    let gamma = pf.taa2.x;

    // Where this pixel's direction was on the previous frame's screen: for
    // the sky through the Lorentz transformation between the frames, for
    // the ship through the camera's turn relative to it.
    let n = ndc_to_dir(out_ndc(px));
    var np: vec3<f32>;
    if (ship_w > 0.5 * wsum) {
        let ms = pf.reproject_ship;
        np = vec3<f32>(dot(ms[0].xyz, n), dot(ms[1].xyz, n), dot(ms[2].xyz, n));
    } else {
        let h = vec4<f32>(-1.0, n);
        let hp = vec4<f32>(dot(pf.reproject[0], h), dot(pf.reproject[1], h), dot(pf.reproject[2], h), dot(pf.reproject[3], h));
        np = hp.yzw / hp.x;
    }
    let q = dir_to_ndc(np);
    let uv = vec2<f32>(q.x * 0.5 + 0.5, 0.5 - q.y * 0.5);
    let valid = q.z > 0.0 && all(uv >= vec2<f32>(0.0)) && all(uv <= vec2<f32>(1.0)) && pf.taa.w < 0.5;

    var res = cur;
    var hw = 0.0;
    if (valid) {
        let hist = history_at(uv);
        hw = min(textureSampleLevel(weight_prev, lin, uv, 0.0).x, pf.taa.z);
        let hy = clip_box(compress(max(hist.rgb, vec3<f32>(0.0)), e), mean - gamma * sigma, mean + gamma * sigma);
        let ha = clamp(hist.a, 0.0, 1.0);
        res = (vec4<f32>(hy, ha) * hw + cur * wsum) / (hw + wsum);
    }
    let lin_rgb = expand(res.rgb, e);
    let o = vec4<f32>(lin_rgb, clamp(res.a, 0.0, 1.0));
    textureStore(hist_next, gid.xy, o);
    textureStore(weight_next, gid.xy, vec4<f32>(hw + wsum, 0.0, 0.0, 0.0));
    textureStore(scene, gid.xy, o);
}
