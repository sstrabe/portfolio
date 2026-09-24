// ---------------------------------------------------------------------------
// The final image: the sharp core of the point-spread function at full
// resolution plus its convolved wings, exposed, seen by the eye model, tone
// mapped with AgX and encoded for the display.
//
// Image = η · scene + wide, per XYZ channel: `scene` is the accumulated
// trace with the point sources splatted into it, `wide` is the scene
// convolved with the PSF outside its core (FFT, coarser grid), and η is the
// fraction of the light the core keeps (1 − the wide kernel's sum). Energy
// is conserved whatever the optics.
// ---------------------------------------------------------------------------

@group(0) @binding(3) var<storage, read> expo: Exposure;
@group(0) @binding(4) var scene_tex: texture_2d<f32>;
@group(0) @binding(5) var wide_tex: texture_2d<f32>;
@group(0) @binding(6) var lin: sampler;
@group(0) @binding(7) var<storage, read> kspec: array<vec4<f32>>;

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

// AgX (Sobotka), in the form Blender and three.js use: into a wider
// working space (inset), log2 encoding over 16.5 stops around middle grey,
// a sigmoid, back out (outset). Bright saturated light desaturates toward
// white the way film and eyes see it, instead of skewing hue as per-channel
// curves do. Returns display-linear sRGB.
fn agx(c: vec3<f32>) -> vec3<f32> {
    let srgb_to_2020 = mat3x3<f32>(
        vec3<f32>(0.6274, 0.0691, 0.0164),
        vec3<f32>(0.3293, 0.9195, 0.0880),
        vec3<f32>(0.0433, 0.0113, 0.8956),
    );
    let inset = mat3x3<f32>(
        vec3<f32>(0.856627153315983, 0.137318972929847, 0.11189821299995),
        vec3<f32>(0.0951212405381588, 0.761241990602591, 0.0767994186031903),
        vec3<f32>(0.0482516061458583, 0.101439036467562, 0.811302368396859),
    );
    let outset = mat3x3<f32>(
        vec3<f32>(1.1271005818144368, -0.1413297634984383, -0.14132976349843826),
        vec3<f32>(-0.11060664309660323, 1.157823702216272, -0.11060664309660294),
        vec3<f32>(-0.016493938717834573, -0.016493938717834257, 1.2519364065950405),
    );
    let from_2020 = mat3x3<f32>(
        vec3<f32>(1.6605, -0.1246, -0.0182),
        vec3<f32>(-0.5876, 1.1329, -0.1006),
        vec3<f32>(-0.0728, -0.0083, 1.1187),
    );
    let min_ev = -12.47393;
    let max_ev = 4.026069;
    var x = inset * (srgb_to_2020 * c);
    x = clamp((log2(max(x, vec3<f32>(1e-10))) - min_ev) / (max_ev - min_ev), vec3<f32>(0.0), vec3<f32>(1.0));
    let x2 = x * x;
    let x4 = x2 * x2;
    x = 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
    x = pow(max(outset * x, vec3<f32>(0.0)), vec3<f32>(2.2));
    return clamp(from_2020 * x, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn srgb_encode(c: vec3<f32>) -> vec3<f32> {
    let lo = c * 12.92;
    let hi = 1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(hi, lo, c <= vec3<f32>(0.0031308));
}

// Rod vision. Below a few cd/m² the cones give way to the rods, which
// see no colour and peak at 507 nm instead of 555 nm (the Purkinje shift:
// blues brighten, reds go dark). Scotopic luminance from XYZ (a linear fit
// to V′(λ)), shown with the slight blue cast night scenes are perceived
// with; the photopic share follows the luminance, log-linearly from
// 0.003 to 3 cd/m².
fn mesopic(xyz: vec3<f32>, rgb: vec3<f32>) -> vec3<f32> {
    let cd = max(xyz.y, 0.0) * LM_PER_W;
    let m = smoothstep(-2.5, 0.5, log2(max(cd, 1e-12)) / log2(10.0));
    let rod = max(-0.702 * xyz.x + 1.039 * xyz.y + 0.433 * xyz.z, 0.0) / 0.77;
    let tint = vec3<f32>(0.86, 0.97, 1.22) / 0.968;
    return mix(rod * tint, rgb, m);
}

@fragment
fn fs_post(in: VsOut) -> @location(0) vec4<f32> {
    let p = vec2<i32>(in.pos.xy);
    let sharp = rgb_to_xyz(textureLoad(scene_tex, p, 0).rgb);
    let s = f32(pf.grid2.x);
    let wuv = in.pos.xy / (s * vec2<f32>(pf.grid.zw));
    let wide = textureSampleLevel(wide_tex, lin, wuv, 0.0).xyz;
    let eta = 1.0 - kspec[0].xyz;
    let xyz = eta * sharp + wide;
    var c = xyz_to_rgb(xyz);
    if (pf.mode.x == EYE) {
        c = mesopic(xyz, c);
    } else {
        // Natural vignetting of a wide-angle lens (a retrofocus design
        // keeps it well under the cos⁴ law).
        let n = ndc_to_dir(out_ndc(in.pos.xy));
        c *= mix(1.0, n.x * n.x, 0.6);
    }
    c = agx(max(c, vec3<f32>(0.0)) * expo.value);
    if (frame.screen.z > 0.5) {
        c = srgb_encode(c);
    }
    let d = hash3(vec3<u32>(vec2<u32>(in.pos.xy), pf.grid2.y)).x - 0.5;
    c += vec3<f32>(d / 255.0);
    return vec4<f32>(c, 1.0);
}
