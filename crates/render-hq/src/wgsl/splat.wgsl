// ---------------------------------------------------------------------------
// Point sources (every body's lensed images from the `images` pass) added
// to the HDR radiance as energy: flux F spread by a point-spread function
// of angular width σ, so ∫ radiance dΩ = F. Blending multiplies by the
// destination alpha (near-field transmittance): planets hide stars.
//
// Placeholder optics: a Gaussian core plus a faint wide halo.
// ---------------------------------------------------------------------------

@group(0) @binding(2) var<storage, read> images: array<ImageState>;
@group(0) @binding(3) var<storage, read> bodies: array<BodyMeta>;

struct SplatOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) offset: vec2<f32>,  // from the image centre, rad
    @location(1) rgb: vec3<f32>,     // flux (W/m², CIE weighted linear sRGB)
    @location(2) sigma: f32,         // core width (rad)
}

const HALO_FRACTION: f32 = 0.03;
const HALO_WIDTH: f32 = 12.0;

@vertex
fn vs_splat(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> SplatOut {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
    );
    var o: SplatOut;
    o.pos = vec4<f32>(0.0, 0.0, 2.0, 1.0);
    let st = images[ii];
    let bm = bodies[ii / 2u];
    if (st.dir.w < 0.5 || bm.b.x < 0.5) {
        return o;
    }
    let ndc = dir_to_ndc(st.dir.xyz);
    if (ndc.z <= 0.02) {
        return o;
    }
    let g = st.info.x;
    var flux = st.info.y * hq.radiometry.x;
    // Stars resolved into discs are drawn by the ray tracer instead.
    let px = frame.cam.z;
    let theta = bm.b.w * sqrt(st.info.y / max(bm.a.w * g * g * g * g, 1e-30));
    flux *= 1.0 - smoothstep(0.5, 1.5, theta / px);
    if (flux <= 0.0) {
        return o;
    }
    let s = spec_scale(spec_planck_unit(bm.a.z * g), flux);
    o.rgb = spec_to_rgb(s);
    o.sigma = max(hq.radiometry.w, 0.6 * px);
    // Out to where the halo drops below ~1e-3 of the exposure's white.
    let peak = spec_luma(s) * hq.radiometry.z * HALO_FRACTION / (TAU * pow(o.sigma * HALO_WIDTH, 2.0));
    let reach = o.sigma * max(4.0, HALO_WIDTH * sqrt(2.0 * log(max(peak / 1e-3, 1.0))));
    let c = corners[vi];
    o.offset = c * reach;
    let t = frame.cam.x;
    o.pos = vec4<f32>(ndc.xy + c * reach * vec2<f32>(1.0 / (t * frame.cam.y), 1.0 / t), 0.0, 1.0);
    return o;
}

@fragment
fn fs_splat(in: SplatOut) -> @location(0) vec4<f32> {
    let r2 = dot(in.offset, in.offset);
    let s2 = in.sigma * in.sigma;
    let h2 = s2 * HALO_WIDTH * HALO_WIDTH;
    let core = (1.0 - HALO_FRACTION) * exp(-0.5 * r2 / s2) / (TAU * s2);
    let halo = HALO_FRACTION * exp(-0.5 * r2 / h2) / (TAU * h2);
    return vec4<f32>(in.rgb * (core + halo), 0.0);
}
