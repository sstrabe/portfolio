// ---------------------------------------------------------------------------
// Metering and adaptation, entirely on the GPU (no read-back stall).
//
// Scene: the histogram holds the log luminance of what reaches the eye or
// sensor (the image after the point-spread function, so glare around a
// bright star raises the adaptation near it, as in a real eye), centre
// weighted. The meter log-averages the bright part of the scene: pixels
// within 8 stops of the 95th percentile, up to the 99th. Deep black sky
// beside a sunlit planet is then ignored, as a photographer's meter or an
// eye fixating the planet would, and a star's disc filling the centre sets
// the exposure; that 95th percentile may not exceed 0.5 (under white).
// The average maps to a key value that falls from middle
// grey (0.18) in daylight to 0.045 in scotopic light: night scenes look
// dark, not grey.
//
// Point sources: stars are far smaller than a pixel, so they barely move a
// histogram, yet the eye fixating them (or a photographer protecting
// highlights) adapts to them. The fifth brightest point on screen may peak
// at `pf.expo2.w` (AgX white is ~16; brighter cores stay white while their
// glare and spikes grow) and no brighter.
//
// Bounds: never more exposure than the dark-adapted limit (the faintest
// visible star just visible). Adaptation runs in log space: quickly toward
// brighter light, slowly toward darkness (rods take minutes; shortened for
// flight). The astrograph uses a fixed exposure instead.
// ---------------------------------------------------------------------------

@group(0) @binding(3) var<storage, read_write> expo: Exposure;
@group(0) @binding(4) var<storage, read> images: array<ImageState>;
@group(0) @binding(5) var<storage, read> bodies: array<BodyMeta>;
@group(0) @binding(12) var<storage, read_write> histogram: array<atomic<u32>, 256>;

// Display value of the 95th percentile at most (AgX shows 0.5 as ~0.68).
const P95_DISPLAY: f32 = 0.5;

var<workgroup> counts: array<u32, 256>;
var<workgroup> brightest: array<vec4<f32>, 256>;

// Peak radiance (CIE Y, W m⁻² sr⁻¹) of point image `ii`'s splatted core,
// or 0 when it is off screen or drawn as a resolved disc.
fn point_peak(ii: u32) -> f32 {
    let st = images[ii];
    let bm = bodies[ii / 2u];
    if (st.dir.w < 0.5 || bm.b.x < 0.5) {
        return 0.0;
    }
    let ndc = dir_to_ndc(st.dir.xyz);
    if (ndc.z <= 0.02 || abs(ndc.x) > 1.0 || abs(ndc.y) > 1.0) {
        return 0.0;
    }
    let g = st.info.x;
    let theta = bm.b.w * sqrt(st.info.y / max(bm.a.w * g * g * g * g, 1e-30));
    if (theta > 1.5 * frame.cam.z) {
        return 0.0;
    }
    let flux = st.info.y * hq.radiometry.x;
    let y = spec_luma(spec_scale(spec_planck_unit(bm.a.z * g), flux));
    let sigma = max(pf.optics.x, 0.5 * pf.optics.y);
    return y / (TAU * sigma * sigma);
}

// Insert v into a descending top-4 list.
fn top4(t: vec4<f32>, v: f32) -> vec4<f32> {
    if (v <= t.w) {
        return t;
    }
    if (v > t.x) {
        return vec4<f32>(v, t.xyz);
    }
    if (v > t.y) {
        return vec4<f32>(t.x, v, t.yz);
    }
    if (v > t.z) {
        return vec4<f32>(t.xy, v, t.z);
    }
    return vec4<f32>(t.xyz, v);
}

@compute @workgroup_size(256)
fn cs_exposure(@builtin(local_invocation_index) li: u32) {
    counts[li] = atomicLoad(&histogram[li]);
    atomicStore(&histogram[li], 0u);
    var t = vec4<f32>(0.0);
    let n = arrayLength(&images);
    for (var i = li; i < n; i += 256u) {
        t = top4(t, point_peak(i));
    }
    brightest[li] = t;
    workgroupBarrier();
    if (li != 0u) {
        return;
    }

    // The fifth brightest point: drop the four largest of all candidates.
    var best = vec4<f32>(0.0);
    var fifth = 0.0;
    for (var i = 0u; i < 256u; i++) {
        let c = brightest[i];
        for (var k = 0u; k < 4u; k++) {
            let v = c[k];
            if (v > best.w) {
                fifth = max(fifth, best.w);
                best = top4(best, v);
            } else {
                fifth = max(fifth, v);
            }
        }
    }

    var total = 0u;
    for (var i = 0u; i < HIST_BINS; i++) {
        total += counts[i];
    }
    if (total == 0u) {
        return;
    }
    let bin_width = HIST_RANGE / f32(HIST_BINS);
    // The 95th percentile, then log-average from 8 stops below it to the
    // 99th percentile.
    var below = 0.0;
    var l95 = HIST_MIN;
    for (var i = 0u; i < HIST_BINS; i++) {
        if (below < 0.95 * f32(total)) {
            l95 = HIST_MIN + (f32(i) + 0.5) * bin_width;
        }
        below += f32(counts[i]);
    }
    let hi = 0.99 * f32(total);
    below = 0.0;
    var wsum = 0.0;
    var lsum = 0.0;
    for (var i = 0u; i < HIST_BINS; i++) {
        let c = f32(counts[i]);
        let l = HIST_MIN + (f32(i) + 0.5) * bin_width;
        let take = max(min(below + c, hi) - below, 0.0);
        below += c;
        if (l >= l95 - 8.0) {
            wsum += take;
            lsum += take * l;
        }
    }
    let lavg = exp2(lsum / max(wsum, 1.0));
    let cd = lavg * LM_PER_W;
    let tk = clamp((log2(cd) / log2(10.0) + 3.0) / 4.0, 0.0, 1.0);
    let key = 0.18 * exp2(-2.0 * (1.0 - tk));
    // ... and the bright part of the subject (the 95th percentile) stays
    // below white: a sunlit planet or a star's disc filling the view keeps
    // its shading instead of burning out.
    var target_value = min(key / lavg, P95_DISPLAY / exp2(l95));
    if (fifth > 0.0) {
        target_value = min(target_value, pf.expo2.w / fifth);
    }
    target_value = clamp(target_value, pf.expo.y, pf.expo.z) * pf.expo.x;
    if (pf.expo2.x > 0.0) {
        target_value = pf.expo2.x * pf.expo.x;
    }
    var v = target_value;
    if (pf.mode.w == 0u && expo.value > 0.0) {
        let now = log2(expo.value);
        let goal = log2(target_value);
        // Lower exposure = brighter scene = light adaptation (fast).
        let tau = select(pf.expo2.z, pf.expo2.y, goal < now);
        v = exp2(now + (goal - now) * (1.0 - exp(-pf.expo.w / tau)));
    }
    expo.value = v;
    expo.adapted = cd;
    expo.p95 = exp2(l95) * LM_PER_W;
    expo.highlight = fifth * LM_PER_W;
}
