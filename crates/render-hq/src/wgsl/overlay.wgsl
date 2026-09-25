// ---------------------------------------------------------------------------
// The 2D overlay (HUD, map): quads in output pixels, drawn over the finished
// image with premultiplied alpha. Colours arrive in sRGB; a target that
// stores sRGB gets them linearised so they appear as given.
// ---------------------------------------------------------------------------

struct Screen {
    size: vec2<f32>,
    srgb_target: f32,
    pad: f32,
}

@group(0) @binding(0) var<uniform> screen: Screen;
@group(0) @binding(1) var font: texture_2d<f32>;
@group(0) @binding(2) var font_smp: sampler;

struct VsIn {
    @location(0) pos: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
    @location(3) shape: vec4<f32>,  // mode, then per mode: ring (inner radius, soft edge), line (extent, half width)
}

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) @interpolate(flat) shape: vec4<f32>,
}

@vertex
fn vs_overlay(v: VsIn) -> VsOut {
    let ndc = vec2<f32>(v.pos.x / screen.size.x * 2.0 - 1.0, 1.0 - v.pos.y / screen.size.y * 2.0);
    return VsOut(vec4<f32>(ndc, 0.0, 1.0), v.uv, v.color, v.shape);
}

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    return select(pow((c + 0.055) / 1.055, vec3<f32>(2.4)), c / 12.92, c <= vec3<f32>(0.04045));
}

@fragment
fn fs_overlay(in: VsOut) -> @location(0) vec4<f32> {
    var a = in.color.a;
    let mode = u32(in.shape.x + 0.5);
    if (mode == 1u) {
        a *= textureSampleLevel(font, font_smp, in.uv, 0.0).r;
    } else if (mode == 2u) {
        let r = length(in.uv);
        let aa = in.shape.z;
        a *= (1.0 - smoothstep(1.0 - aa, 1.0 + aa, r)) * smoothstep(in.shape.y - aa, in.shape.y + aa, r);
        if (in.shape.y <= 0.0) {
            a = in.color.a * (1.0 - smoothstep(1.0 - aa, 1.0 + aa, r));
        }
    } else if (mode == 3u) {
        // Line: uv.y runs ±1 across the quad, which is `shape.y` pixels
        // from the centre line to each edge; the line is `shape.z` wide
        // each way.
        let d = abs(in.uv.y) * in.shape.y;
        a *= clamp(in.shape.z + 0.5 - d, 0.0, 1.0);
    }
    var rgb = in.color.rgb;
    if (screen.srgb_target > 0.5) {
        rgb = srgb_to_linear(rgb);
    }
    return vec4<f32>(rgb * a, a);
}
