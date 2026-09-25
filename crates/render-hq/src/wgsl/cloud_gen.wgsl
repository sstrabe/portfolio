// ---------------------------------------------------------------------------
// Cloud textures, built on the GPU:
//
// - a tileable 128³ noise volume (once): Perlin–Worley in R (billowy cloud
//   masses) and Worley fBm at three frequencies in G, B, A (cellular
//   detail), as in Guerrilla's Nubis;
// - per cloudy planet, a weather cube map (6 × 512², body fixed) of
//   coverage (R), cloud-top height as a fraction of the layer (G), density
//   (B) and cellularity (A), following the general circulation:
//   * the ITCZ: a band of deep convection near the equator;
//   * the subtropical highs (~15–35°): mostly clear, with decks of
//     cellular marine stratocumulus;
//   * the mid-latitude storm tracks (~40–65°): cyclones as log-spiral
//     twists of the pattern, counter-clockwise in the northern hemisphere,
//     along with frontal bands stretched by the jets;
//   * polar stratus.
// ---------------------------------------------------------------------------

@group(0) @binding(7) var map_out: texture_storage_2d_array<rgba8unorm, write>;
@group(0) @binding(8) var noise_out: texture_storage_3d<rgba8unorm, write>;

fn cg_hash(c: vec3<i32>, period: i32) -> vec3<f32> {
    let q = ((c % vec3<i32>(period)) + vec3<i32>(period)) % vec3<i32>(period);
    return hash3(vec3<u32>(q) + vec3<u32>(7u, 131u, 977u));
}

// Tileable gradient noise in [−1, 1] (period in cells).
fn cg_perlin(p: vec3<f32>, period: i32) -> f32 {
    let i = vec3<i32>(floor(p));
    let f = fract(p);
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    var n: array<f32, 8>;
    for (var k = 0; k < 8; k++) {
        let o = vec3<i32>(k & 1, (k >> 1) & 1, (k >> 2) & 1);
        let g = normalize(cg_hash(i + o, period) * 2.0 - 1.0 + vec3<f32>(1e-4));
        n[k] = dot(g, f - vec3<f32>(o));
    }
    let x0 = mix(mix(n[0], n[1], u.x), mix(n[2], n[3], u.x), u.y);
    let x1 = mix(mix(n[4], n[5], u.x), mix(n[6], n[7], u.x), u.y);
    return mix(x0, x1, u.z);
}

// Tileable Worley noise: 1 − distance to the nearest feature point.
fn cg_worley(p: vec3<f32>, period: i32) -> f32 {
    let i = vec3<i32>(floor(p));
    let f = fract(p);
    var d = 1.0;
    for (var z = -1; z <= 1; z++) {
        for (var y = -1; y <= 1; y++) {
            for (var x = -1; x <= 1; x++) {
                let o = vec3<i32>(x, y, z);
                let fp = vec3<f32>(o) + cg_hash(i + o, period);
                d = min(d, length(fp - f));
            }
        }
    }
    return 1.0 - d;
}

fn cg_worley_fbm(p: vec3<f32>, period: i32) -> f32 {
    return cg_worley(p, period) * 0.625 + cg_worley(p * 2.0, period * 2) * 0.25 + cg_worley(p * 4.0, period * 4) * 0.125;
}

@compute @workgroup_size(4, 4, 4)
fn cs_cloud_noise(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = (vec3<f32>(gid) + 0.5) / 128.0;
    // Perlin fBm with 4, 8, 16 cells over the tile.
    var perlin = 0.0;
    var amp = 0.5;
    var cells = 4;
    for (var o = 0; o < 3; o++) {
        perlin += amp * cg_perlin(x * f32(cells), cells);
        amp *= 0.5;
        cells *= 2;
    }
    perlin = clamp(perlin * 0.9 + 0.5, 0.0, 1.0);
    let w4 = cg_worley_fbm(x * 4.0, 4);
    // Perlin–Worley: Worley's billows filled in by Perlin (Nubis).
    let pw = clamp(w4 + perlin * (1.0 - w4) - 0.35, 0.0, 1.0) / 0.65;
    let w8 = cg_worley_fbm(x * 8.0, 8);
    let w16 = cg_worley_fbm(x * 16.0, 16);
    let w32 = cg_worley_fbm(x * 32.0, 32);
    textureStore(noise_out, gid, vec4<f32>(pw, w8, w16, w32));
}

// --- Weather maps ----------------------------------------------------------

fn cg_value(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let o = vec3<u32>(vec3<i32>(i) + vec3<i32>(4096));
    let a = mix(hash3(o).x, hash3(o + vec3<u32>(1u, 0u, 0u)).x, u.x);
    let b = mix(hash3(o + vec3<u32>(0u, 1u, 0u)).x, hash3(o + vec3<u32>(1u, 1u, 0u)).x, u.x);
    let c = mix(hash3(o + vec3<u32>(0u, 0u, 1u)).x, hash3(o + vec3<u32>(1u, 0u, 1u)).x, u.x);
    let d = mix(hash3(o + vec3<u32>(0u, 1u, 1u)).x, hash3(o + vec3<u32>(1u, 1u, 1u)).x, u.x);
    return mix(mix(a, b, u.y), mix(c, d, u.y), u.z);
}

// fBm in [0, 1] (mean ≈ 0.5).
fn cg_fbm(p: vec3<f32>, octaves: i32) -> f32 {
    var s = 0.0;
    var amp = 0.5;
    var q = p;
    var norm = 0.0;
    for (var o = 0; o < octaves; o++) {
        s += amp * cg_value(q);
        norm += amp;
        amp *= 0.5;
        q = q * 2.03 + vec3<f32>(17.1, 3.7, 9.3);
    }
    return s / norm;
}

// Direction of texel (s, t) on cube face f (Vulkan / WebGPU face order).
fn cg_cube_dir(face: u32, st: vec2<f32>) -> vec3<f32> {
    let sc = st.x * 2.0 - 1.0;
    let tc = st.y * 2.0 - 1.0;
    switch (face) {
        case 0u: { return normalize(vec3<f32>(1.0, -tc, -sc)); }
        case 1u: { return normalize(vec3<f32>(-1.0, -tc, sc)); }
        case 2u: { return normalize(vec3<f32>(sc, 1.0, tc)); }
        case 3u: { return normalize(vec3<f32>(sc, -1.0, -tc)); }
        case 4u: { return normalize(vec3<f32>(sc, -tc, 1.0)); }
        default: { return normalize(vec3<f32>(-sc, -tc, -1.0)); }
    }
}

fn cg_rotate(v: vec3<f32>, axis: vec3<f32>, angle: f32) -> vec3<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return v * c + cross(axis, v) * s + axis * dot(axis, v) * (1.0 - c);
}

const CG_CYCLONES: u32 = 14u;

@compute @workgroup_size(8, 8)
fn cs_cloud_map(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot = gid.z / 6u;
    let face = gid.z % 6u;
    let a = atmo_geom_of(lut_params[slot]);
    if (!atmo_present(a) || a.ids.z == ATMO_NO_CLOUDS) {
        return;
    }
    let seed = a.ids.x;
    let so = hash3(vec3<u32>(seed, seed ^ 0x5bd1e995u, 77u)) * 200.0;
    let b0 = cg_cube_dir(face, (vec2<f32>(gid.xy) + 0.5) / 512.0);
    let lat0 = asin(clamp(b0.z, -1.0, 1.0));

    // Cyclones: twist the pattern around seeded centres in the storm
    // tracks, strongest at the centre, giving log-spiral arms.
    var b = b0;
    var storm = 0.0;
    for (var k = 0u; k < CG_CYCLONES; k++) {
        let h = hash3(vec3<u32>(seed, k, 911u));
        let hemi = select(-1.0, 1.0, (k & 1u) == 0u);
        let lat = hemi * mix(0.6, 1.1, h.x);
        let lon = h.y * TAU;
        let c = vec3<f32>(cos(lat) * cos(lon), cos(lat) * sin(lon), sin(lat));
        let radius = mix(0.12, 0.26, h.z);
        let d = acos(clamp(dot(b, c), -1.0, 1.0));
        if (d < 3.0 * radius) {
            let twist = hemi * 5.0 * exp(-d / (0.45 * radius));
            b = normalize(cg_rotate(b, c, twist));
            storm = max(storm, exp(-(d / radius) * (d / radius)));
        }
    }
    // Zonal jets shear the pattern east–west and stretch it along latitude.
    let lat = asin(clamp(b.z, -1.0, 1.0));
    let shear = 0.35 * sin(3.0 * lat) + 0.15 * sin(7.0 * lat);
    let bz = cg_rotate(b, vec3<f32>(0.0, 0.0, 1.0), shear);
    let q = vec3<f32>(bz.x, bz.y, bz.z * 1.8) * 3.0 + so;
    let warp = vec3<f32>(cg_fbm(q * 1.3, 4), cg_fbm(q * 1.3 + 31.7, 4), cg_fbm(q * 1.3 + 71.3, 4)) - 0.5;
    let base = cg_fbm(q + warp * 1.6, 7);

    // Belts of the general circulation (Hadley, Ferrel, polar cells).
    let itcz_lat = 0.1 * (hash3(vec3<u32>(seed, 5u, 5u)).x - 0.5);
    let al = abs(lat0);
    let itcz = exp(-pow((lat0 - itcz_lat) / 0.09, 2.0));
    let subtropic = exp(-pow((al - 0.45) / 0.14, 2.0));
    let track = exp(-pow((al - 0.9) / 0.22, 2.0));
    let polar = smoothstep(1.1, 1.35, al);
    let bias = 0.22 * itcz - 0.2 * subtropic + 0.1 * track + 0.05 * polar + 0.18 * storm;

    let cover_param = a.clouds.x;
    let threshold = 0.5 + 0.35 * (0.5 - cover_param);
    let cov = clamp((base + bias - threshold) * 3.0 + 0.5, 0.0, 1.0);

    // Marine stratocumulus: cellular decks in the subtropics.
    let deck = subtropic * smoothstep(0.45, 0.65, cg_fbm(q * 0.7 + 13.0, 4));
    let coverage = max(cov, deck * 0.8 * cover_param * 1.4);
    // Deep convection at the ITCZ and in cyclones; low tops in the decks.
    let top = clamp(0.55 + 0.45 * max(itcz, storm) - 0.3 * subtropic + 0.3 * (cg_fbm(q * 2.0 + 5.0, 3) - 0.5), 0.2, 1.0);
    let density = clamp(0.6 + 0.4 * max(itcz, storm) + 0.5 * (base - 0.5), 0.3, 1.0);
    let cells = clamp(deck * 1.5 + 0.2 * subtropic, 0.0, 1.0);
    textureStore(map_out, gid.xy, a.ids.z * 6u + face, vec4<f32>(coverage, top, density, cells));
}
