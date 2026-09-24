// ---------------------------------------------------------------------------
// Point-spread sprites for body images, drawn at full output resolution on
// top of the tone-mapped sky. Colour is the blackbody at the shifted
// temperature g·T; total energy is the lensed, beamed flux g⁴/area.
// ---------------------------------------------------------------------------

@group(0) @binding(1) var<storage, read> images: array<ImageState>;
@group(0) @binding(2) var<storage, read> bodies: array<BodyMeta>;

struct SpriteOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) local: vec2<f32>,
    @location(1) color: vec3<f32>,
    @location(2) peak: f32,
}

@vertex
fn vs_sprite(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> SpriteOut {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
    );
    var o: SpriteOut;
    o.pos = vec4<f32>(0.0, 0.0, 2.0, 1.0);
    o.local = vec2<f32>(0.0);
    o.color = vec3<f32>(0.0);
    o.peak = 0.0;
    let st = images[ii];
    if (st.dir.w < 0.5) {
        return o;
    }
    let ndc = dir_to_ndc(st.dir.xyz);
    if (ndc.z <= 0.02) {
        return o;
    }
    let bm = bodies[ii / 2u];
    let g = st.info.x;
    var e = st.info.y * frame.screen.w;
    if (bm.b.x < 0.5) {
        e *= bm.b.y;
    }
    if (e < 2.0e-3) {
        return o;
    }
    if (bm.b.x < 0.5) {
        // Station beacons: a fixed cyan hue, still Doppler shifted.
        o.color = vec3<f32>(0.3, 0.85, 1.0) * mix(vec3<f32>(1.0), blackbody(6500.0 * g), 0.6);
    } else {
        o.color = blackbody(bm.a.z * g);
    }
    let sigma = clamp(0.75 + 0.5 * log2(1.0 + e), 0.75, 14.0);
    o.peak = 6.0 * e / (TAU * sigma * sigma);
    let c = corners[vi];
    o.local = c * 3.0;
    o.pos = vec4<f32>(ndc.xy + c * 3.0 * sigma * 2.0 / frame.screen.xy, 0.0, 1.0);
    return o;
}

@fragment
fn fs_sprite(in: SpriteOut) -> @location(0) vec4<f32> {
    let i = in.peak * exp(-0.5 * dot(in.local, in.local));
    let c = in.color * (1.0 - exp(-i));
    return vec4<f32>(c, 0.0);
}
