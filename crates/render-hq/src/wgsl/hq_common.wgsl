// ---------------------------------------------------------------------------
// HQ renderer prelude (after `common.wgsl` and `spectrum.wgsl`): per-frame
// uniforms beyond `Frame`, and the group-0 bindings of the trace pass.
//
// Bind groups of the trace pass, one per feature so features can be
// developed independently:
//   0  core: frame, HQ frame, nearby star discs, output image, narrowband image
//   1  near field: star systems and planets          (near.wgsl, planet.wgsl)
//   2  atmospheres and clouds                        (atmosphere.wgsl)
//   3  nebulae                                       (nebula.wgsl)
//   4  the ship                                      (ship.wgsl)
//   5  lensing by stellar-mass black holes           (lens.wgsl)
// ---------------------------------------------------------------------------

struct HqFrame {
    size: vec4<u32>,        // trace width, height, frame index, flags (HQ_*)
    view: vec4<f32>,        // jitter x, y (px), wall time (s), pixel solid angle (sr)
    radiometry: vec4<f32>,  // W/m² per flux unit (L☉/M²), sky radiance scale, exposure, point-source σ (rad)
    near: vec4<u32>,        // star systems, planets, unused, unused
    units: vec4<f32>,       // km per M, seconds per M, c (km/s), unused
    rgb: array<vec4<f32>, 12>,  // spectrum → linear sRGB: rows R0–R3, G0–G3, B0–B3
}

// `hq.size.w` flags.
const HQ_NARROWBAND: u32 = 1u;  // the trace also writes the narrowband bins
const HQ_DEBUG_NAN: u32 = 2u;   // mark non-finite pixels in magenta

// A nearby star close enough to show a disc (far field, Kerr traced).
struct Sphere {
    center: vec4<f32>,  // xyz relative to the observer (M), emission time
    vel: vec4<f32>,     // coordinate velocity, radius (M)
    star: vec4<f32>,    // temperature (K), luminosity (L☉), body index, unused
}

struct Spheres {
    items: array<Sphere, 16>,
}

@group(0) @binding(1) var<uniform> hq: HqFrame;

fn spec_to_rgb(s: Spectrum) -> vec3<f32> {
    return vec3<f32>(
        dot(s.a, hq.rgb[0]) + dot(s.b, hq.rgb[1]) + dot(s.c, hq.rgb[2]) + dot(s.d, hq.rgb[3]),
        dot(s.a, hq.rgb[4]) + dot(s.b, hq.rgb[5]) + dot(s.c, hq.rgb[6]) + dot(s.d, hq.rgb[7]),
        dot(s.a, hq.rgb[8]) + dot(s.b, hq.rgb[9]) + dot(s.c, hq.rgb[10]) + dot(s.d, hq.rgb[11]),
    );
}

// Luminance-like Y (sRGB luma of the linear colour).
fn spec_luma(s: Spectrum) -> f32 {
    return dot(spec_to_rgb(s), vec3<f32>(0.2126, 0.7152, 0.0722));
}
