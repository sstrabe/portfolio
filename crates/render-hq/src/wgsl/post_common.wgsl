// ---------------------------------------------------------------------------
// Post-processing prelude (after `hq_common.wgsl`): the uniforms every
// post pass shares, the adapted exposure the GPU keeps, and colour helpers.
//
// Post passes, in order (see `post.rs`):
//   taa      temporal accumulation and upscaling of the traced radiance
//   splat    point sources as energy at their exact sub-pixel positions
//   ghost    lens ghosts of the brightest sources (camera)
//   fft      downsample, 2-D FFT, × PSF spectrum, inverse FFT, histogram
//   exposure metering and adaptation from the histogram
//   post     the sharp image plus the convolved wings, eye model, AgX
// ---------------------------------------------------------------------------

struct PostFrame {
    // Rows of g(e'_a, e_b): the previous frame's tetrad against this one's.
    // A direction n seen now was seen at n' = (M h).yzw / (M h).x with
    // h = (−1, n): a Lorentz transformation (rotation and aberration).
    reproject: array<vec4<f32>, 4>,
    // The ship's pixels (marked by the trace with alpha −1) move with the
    // camera, not the sky: rows are the previous camera axes in the current
    // ones, so n' = (m₀·n, m₁·n, m₂·n). Row 3 is unused.
    reproject_ship: array<vec4<f32>, 4>,
    sizes: vec4<u32>,     // output width, height; trace width, height
    grid: vec4<u32>,      // FFT nx, ny; grid image width, height
    grid2: vec4<u32>,     // output px per grid px, frame count, unused, unused
    taa: vec4<f32>,       // jitter x, y (trace px), history weight cap, reset (1)
    taa2: vec4<f32>,      // clipping box half-width (σ), unused ×3
    mode: vec4<u32>,      // optics (0 eye, 1 camera, 2 astro), palette (0 true, 1 SHO), ghosts, snap
    expo: vec4<f32>,      // exposure multiplier, min exposure, dark-adapted max exposure, dt (s)
    expo2: vec4<f32>,     // astrograph gain over the metered background (0: automatic), light τ (s), dark τ (s), fifth-brightest point's peak display value
    optics: vec4<f32>,    // core σ (rad), output pixel angle (rad), grid pixel angle (rad), unused
    sho: vec4<f32>,       // Hubble palette gains for [S II], Hα, [O III]; unused
    ghosts: array<vec4<f32>, 16>,  // pairs: (k, radius rad, rotation, unused), (tint rgb, unused)
}

// The exposure the GPU adapts; `post.rs` reads it back (a frame or two late)
// for the window title and `hq.radiometry.z`.
struct Exposure {
    value: f32,       // display value per unit radiance (W m⁻² sr⁻¹, CIE Y)
    adapted: f32,     // metered scene luminance, cd/m²
    p95: f32,         // 95th percentile luminance, cd/m²
    highlight: f32,   // peak luminance of the fifth brightest point source, cd/m²
}

@group(0) @binding(2) var<uniform> pf: PostFrame;

const EYE: u32 = 0u;
const CAMERA: u32 = 1u;
const ASTRO: u32 = 2u;

// Luminous efficacy: CIE Y in W m⁻² sr⁻¹ (ȳ peaks at 1) → cd/m².
const LM_PER_W: f32 = 683.0;

// Histogram of log₂ luminance (CIE Y, W m⁻² sr⁻¹).
const HIST_BINS: u32 = 256u;
const HIST_MIN: f32 = -34.0;
const HIST_RANGE: f32 = 64.0;

fn rgb_to_xyz(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        dot(c, vec3<f32>(0.4124564, 0.3575761, 0.1804375)),
        dot(c, vec3<f32>(0.2126729, 0.7151522, 0.0721750)),
        dot(c, vec3<f32>(0.0193339, 0.1191920, 0.9503041)),
    );
}

fn xyz_to_rgb(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        dot(c, vec3<f32>(3.2404542, -1.5371385, -0.4985314)),
        dot(c, vec3<f32>(-0.9692660, 1.8760108, 0.0415560)),
        dot(c, vec3<f32>(0.0556434, -0.2040259, 1.0572252)),
    );
}

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126729, 0.7151522, 0.0721750));
}

// Output pixel ↔ ship-frame direction.
fn out_ndc(px: vec2<f32>) -> vec2<f32> {
    let s = vec2<f32>(pf.sizes.xy);
    return vec2<f32>(px.x / s.x * 2.0 - 1.0, 1.0 - px.y / s.y * 2.0);
}

fn ndc_to_out(ndc: vec2<f32>) -> vec2<f32> {
    let s = vec2<f32>(pf.sizes.xy);
    return vec2<f32>((ndc.x * 0.5 + 0.5) * s.x, (0.5 - ndc.y * 0.5) * s.y);
}

// Colour of a spectrum in the current palette: CIE (true colour) or the
// three narrowband bins mapped to R, G, B.
fn palette_rgb(s: Spectrum) -> vec3<f32> {
    if (pf.mode.y == 1u) {
        return vec3<f32>(s.c.w, s.c.z, s.b.x) * pf.sho.xyz;
    }
    return spec_to_rgb(s);
}
