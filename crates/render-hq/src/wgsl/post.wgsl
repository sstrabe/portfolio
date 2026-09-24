// ---------------------------------------------------------------------------
// Exposure, tone mapping and upscaling of the HDR radiance to the screen.
//
// Placeholder: fixed exposure from the CPU, ACES curve, bilinear upscale.
// ---------------------------------------------------------------------------

@group(0) @binding(2) var hdr_tex: texture_2d<f32>;
@group(0) @binding(3) var hdr_smp: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_fullscreen(@builtin(vertex_index) i: u32) -> VsOut {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    let p = uv * 2.0 - 1.0;
    return VsOut(vec4<f32>(p, 0.0, 1.0), vec2<f32>(uv.x, 1.0 - uv.y));
}

fn aces(x: vec3<f32>) -> vec3<f32> {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn srgb_encode(c: vec3<f32>) -> vec3<f32> {
    let lo = c * 12.92;
    let hi = 1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(hi, lo, c <= vec3<f32>(0.0031308));
}

@fragment
fn fs_post(in: VsOut) -> @location(0) vec4<f32> {
    let hdr = max(textureSample(hdr_tex, hdr_smp, in.uv).rgb, vec3<f32>(0.0)) * hq.radiometry.z;
    var c = aces(hdr);
    let q = in.uv * 2.0 - 1.0;
    c *= 1.0 - 0.09 * dot(q, q);
    if (frame.screen.z > 0.5) {
        c = srgb_encode(c);
    }
    let n = hash3(vec3<u32>(vec2<u32>(in.pos.xy), hq.size.z)).x - 0.5;
    c += vec3<f32>(n / 255.0);
    return vec4<f32>(c, 1.0);
}
