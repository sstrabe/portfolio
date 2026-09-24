// ---------------------------------------------------------------------------
// Point sources (every body's lensed images from the `images` pass) added
// to the output-resolution scene as energy, at their exact sub-pixel
// positions: each pixel receives the source's flux times the fraction of
// the point-spread core that falls inside it (the Gaussian core integrated
// over the pixel square with error functions), divided by the pixel's
// solid angle. The sum over pixels is the flux, whatever the resolution
// or the sub-pixel position, so stars neither shimmer nor change brightness
// as they drift. The core has the optics' own width (never below half a
// pixel, the reconstruction limit); the rest of the PSF — rings, spikes,
// halo — comes from the FFT convolution of the whole image afterwards.
//
// Blending multiplies by the destination alpha (the near-field
// transmittance): planets hide stars.
// ---------------------------------------------------------------------------

@group(0) @binding(3) var<storage, read> expo: Exposure;
@group(0) @binding(4) var<storage, read> images: array<ImageState>;
@group(0) @binding(5) var<storage, read> bodies: array<BodyMeta>;

struct SplatOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) @interpolate(flat) centre: vec2<f32>,  // output px
    @location(1) @interpolate(flat) rgb: vec3<f32>,     // flux / pixel solid angle
    @location(2) @interpolate(flat) sigma: f32,         // output px
}

// Point-image flux (W/m²) and spectrum, or zero flux when it is not a
// visible point. `resolved_fade` drops sources the trace draws as discs.
struct PointSource {
    ndc: vec3<f32>,
    flux: f32,
    s: Spectrum,         // unit-bolometric spectral shape × flux
    theta: f32,          // angular radius of the body's disc, rad
}

fn point_source(ii: u32) -> PointSource {
    var p: PointSource;
    p.flux = 0.0;
    let st = images[ii];
    let bm = bodies[ii / 2u];
    if (st.dir.w < 0.5 || bm.b.x < 0.5) {
        return p;
    }
    p.ndc = dir_to_ndc(st.dir.xyz);
    if (p.ndc.z <= 0.02) {
        return p;
    }
    let g = st.info.x;
    p.flux = st.info.y * hq.radiometry.x;
    p.theta = bm.b.w * sqrt(st.info.y / max(bm.a.w * g * g * g * g, 1e-30));
    p.s = spec_scale(spec_planck_unit(bm.a.z * g), p.flux);
    return p;
}

@vertex
fn vs_splat(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> SplatOut {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
    );
    var o: SplatOut;
    o.pos = vec4<f32>(0.0, 0.0, 2.0, 1.0);
    let p = point_source(ii);
    if (p.flux <= 0.0) {
        return o;
    }
    // Stars resolved into discs are drawn by the ray tracer instead (the
    // trace's pixel is `frame.cam.z` across).
    let fade = 1.0 - smoothstep(0.5, 1.5, p.theta / frame.cam.z);
    if (fade <= 0.0) {
        return o;
    }
    let pix = pf.optics.y;
    o.rgb = palette_rgb(p.s) * fade / (pix * pix);
    o.sigma = max(pf.optics.x / pix, 0.5);
    o.centre = ndc_to_out(p.ndc.xy);
    // Out to where the core falls below ~1e-4 of white.
    let peak = luma(o.rgb) * expo.value / (TAU * o.sigma * o.sigma + 1.0);
    let reach = o.sigma * min(sqrt(2.0 * log(max(peak / 1e-4, 1.0))), 9.0) + 1.5;
    let c = corners[vi];
    let at = o.centre + c * reach;
    o.pos = vec4<f32>(out_ndc(at), 0.0, 1.0);
    return o;
}

@fragment
fn fs_splat(in: SplatOut) -> @location(0) vec4<f32> {
    // Fraction of the Gaussian inside this pixel's square.
    let lo = floor(in.pos.xy) - in.centre;
    let k = 1.0 / (sqrt(2.0) * in.sigma);
    let e = erf4(vec4<f32>(lo, lo + 1.0) * k);
    let f = 0.25 * (e.z - e.x) * (e.w - e.y);
    return vec4<f32>(in.rgb * f, 0.0);
}

// ---------------------------------------------------------------------------
// Lens ghosts (camera only): light reflected off one coated lens surface,
// back off another and on to the sensor forms a defocused image of the
// iris, placed on the line through the image centre at a fixed multiple of
// the source's offset (negative multiples mirror it through the centre).
// Each carries the product of the two surfaces' reflectances, tinted by
// their antireflection coatings (`optics.rs`), times the source's flux,
// spread evenly over the heptagon — so only the Sun and the brightest stars
// leave visible ghosts. A source hidden behind a planet makes none.
// ---------------------------------------------------------------------------

@group(0) @binding(6) var trans_tex: texture_2d<f32>;
@group(0) @binding(7) var lin: sampler;

const GHOST_COUNT: u32 = 7u;
const IRIS_BLADES: f32 = 7.0;

struct GhostOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) local: vec2<f32>,                      // in units of the ghost radius
    @location(1) @interpolate(flat) rgb: vec3<f32>,     // radiance
    @location(2) @interpolate(flat) rot: f32,
}

@vertex
fn vs_ghost(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> GhostOut {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
    );
    var o: GhostOut;
    o.pos = vec4<f32>(0.0, 0.0, 2.0, 1.0);
    if (pf.mode.z == 0u) {
        return o;
    }
    let p = point_source(ii / GHOST_COUNT);
    let g = ii % GHOST_COUNT;
    let edge = max(abs(p.ndc.x), abs(p.ndc.y));
    if (p.flux <= 0.0 || edge > 1.3) {
        return o;
    }
    // Visibility of the source: transmittance over its disc.
    let size = vec2<f32>(pf.sizes.xy);
    let uv = ndc_to_out(p.ndc.xy) / size;
    let rad = 0.7 * p.theta / pf.optics.y / size;
    var vis = 0.0;
    var taps = array<vec2<f32>, 5>(
        vec2<f32>(0.0), vec2<f32>(1.0, 0.0), vec2<f32>(-1.0, 0.0), vec2<f32>(0.0, 1.0), vec2<f32>(0.0, -1.0),
    );
    for (var i = 0u; i < 5u; i++) {
        let at = clamp(uv + taps[i] * rad, vec2<f32>(0.0), vec2<f32>(1.0));
        vis += textureSampleLevel(trans_tex, lin, at, 0.0).a;
    }
    vis *= 0.2 * (1.0 - smoothstep(0.95, 1.3, edge));
    let gk = pf.ghosts[2u * g];
    let tint = pf.ghosts[2u * g + 1u].rgb;
    let r = gk.y;
    let area = 0.5 * IRIS_BLADES * r * r * sin(TAU / IRIS_BLADES);
    o.rgb = palette_rgb(p.s) * tint * (vis / area);
    if (luma(o.rgb) * expo.value < 2e-4) {
        return o;
    }
    let c = corners[vi] * 1.08;
    o.local = c;
    o.rot = gk.z;
    let t = frame.cam.x;
    o.pos = vec4<f32>(gk.x * p.ndc.xy + c * r * vec2<f32>(1.0 / (t * frame.cam.y), 1.0 / t), 0.0, 1.0);
    return o;
}

@fragment
fn fs_ghost(in: GhostOut) -> @location(0) vec4<f32> {
    // Signed distance to the heptagon (circumradius 1), rotated.
    let a = atan2(in.local.y, in.local.x) - in.rot;
    let sector = TAU / IRIS_BLADES;
    let phi = a - sector * (floor(a / sector) + 0.5);
    let d = length(in.local) * cos(phi) - cos(0.5 * sector);
    let cover = 1.0 - smoothstep(-0.05, 0.02, d);
    // Defocused ghosts are a little brighter toward their rims.
    let rim = 0.78 + 0.45 * dot(in.local, in.local);
    return vec4<f32>(in.rgb * cover * rim, 0.0);
}
