// ---------------------------------------------------------------------------
// Per-pixel backward null geodesics from the pilot's eye.
//
// Each pixel's ray ends on the horizon (black), on the distant sky (a
// procedural galaxy seen with gravitational + Doppler shift g and intensity
// ∝ g⁴), or on a station panel. Panels are flat cards moving with their
// station; the ray meets them at its own coordinate time, so they are
// lensed, aberrated, delayed and Doppler shifted like everything else.
// ---------------------------------------------------------------------------

struct Panel {
    center: vec4<f32>,  // centre at the reference event (xyz), reference time
    vel: vec4<f32>,     // coordinate velocity (xyz), active flag
    right: vec4<f32>,   // half-width vector (xyz), atlas u0
    up: vec4<f32>,      // half-height vector (xyz), atlas v0
    atlas: vec4<f32>,   // atlas du, dv, brightness, highlight
}

struct Panels {
    items: array<Panel, 8>,
}

@group(0) @binding(1) var<uniform> panels: Panels;
@group(0) @binding(2) var atlas_tex: texture_2d<f32>;
@group(0) @binding(3) var atlas_smp: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) ndc: vec2<f32>,
}

@vertex
fn vs_fullscreen(@builtin(vertex_index) i: u32) -> VsOut {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    let p = uv * 2.0 - 1.0;
    return VsOut(vec4<f32>(p, 0.0, 1.0), p);
}

struct PanelHit {
    hit: bool,
    color: vec3<f32>,
}

// Does the step a → b cross a panel? `p` is the photon momentum there.
fn panel_hit(a: vec4<f32>, b: vec4<f32>, p: vec4<f32>) -> PanelHit {
    var out: PanelHit;
    out.hit = false;
    var best = 2.0;
    for (var j = 0u; j < frame.counts2.y; j++) {
        let pn = panels.items[j];
        if (pn.vel.w < 0.5) {
            continue;
        }
        let ca = pn.center.xyz + pn.vel.xyz * (a.x - pn.center.w);
        let cb = pn.center.xyz + pn.vel.xyz * (b.x - pn.center.w);
        let nrm = normalize(cross(pn.right.xyz, pn.up.xyz));
        let da = dot(a.yzw - ca, nrm);
        let db = dot(b.yzw - cb, nrm);
        if (da * db > 0.0 || da == db) {
            continue;
        }
        let s = da / (da - db);
        if (s >= best) {
            continue;
        }
        let q = mix(a.yzw - ca, b.yzw - cb, s);
        let u = dot(q, pn.right.xyz) / dot(pn.right.xyz, pn.right.xyz);
        let v = dot(q, pn.up.xyz) / dot(pn.up.xyz, pn.up.xyz);
        if (abs(u) > 1.04 || abs(v) > 1.06) {
            continue;
        }
        best = s;
        out.hit = true;
        let x = mix(a.yzw, b.yzw, s);
        let w = four_velocity(x, pn.vel.xyz);
        let g = clamp(1.0 / max(dot(p, w), 1e-4), 0.05, 20.0);
        let shift = mix(vec3<f32>(1.0), blackbody(6500.0 * g), 0.65) * min(pow(g, 3.0), 12.0);
        if (da < 0.0) {
            // Back of the card.
            out.color = vec3<f32>(0.012, 0.014, 0.02) * shift;
            continue;
        }
        if (abs(u) > 1.0 || abs(v) > 1.0) {
            // Glowing frame, brighter when highlighted.
            out.color = vec3<f32>(0.35, 0.8, 1.0) * (0.6 + 1.4 * pn.atlas.w) * shift;
            continue;
        }
        let uv = vec2<f32>(pn.right.w + (u * 0.5 + 0.5) * pn.atlas.x, pn.up.w + (0.5 - v * 0.5) * pn.atlas.y);
        let texel = textureSampleLevel(atlas_tex, atlas_smp, uv, 0.0);
        // sRGB texel → linear emission.
        let lin = pow(texel.rgb, vec3<f32>(2.2));
        out.color = (lin * pn.atlas.z + vec3<f32>(0.004, 0.006, 0.012)) * shift;
    }
    return out;
}

// ---------------------------------------------------------------------------
// The distant sky: a galaxy seen from its own nucleus.
// ---------------------------------------------------------------------------

const STAR_SIGMA: f32 = 2.5e-4; // intrinsic star blur (rad)

fn cube_uv(d: vec3<f32>) -> vec3<f32> {
    let a = abs(d);
    if (a.x >= a.y && a.x >= a.z) {
        return vec3<f32>(d.y / a.x, d.z / a.x, select(0.0, 1.0, d.x < 0.0));
    }
    if (a.y >= a.z) {
        return vec3<f32>(d.x / a.y, d.z / a.y, select(2.0, 3.0, d.y < 0.0));
    }
    return vec3<f32>(d.x / a.z, d.y / a.z, select(4.0, 5.0, d.z < 0.0));
}

// Direction of cube-face coordinates (inverse of `cube_uv`).
fn cube_dir(face: u32, uv: vec2<f32>) -> vec3<f32> {
    switch face {
        case 0u: { return vec3<f32>(1.0, uv.x, uv.y); }
        case 1u: { return vec3<f32>(-1.0, uv.x, uv.y); }
        case 2u: { return vec3<f32>(uv.x, 1.0, uv.y); }
        case 3u: { return vec3<f32>(uv.x, -1.0, uv.y); }
        case 4u: { return vec3<f32>(uv.x, uv.y, 1.0); }
        default: { return vec3<f32>(uv.x, uv.y, -1.0); }
    }
}

// One layer of point stars scattered over the sphere through a cube-face
// grid. Cells near face edges and corners cover less sky (solid angle
// ∝ (1 + u² + v²)^(-3/2)), so the chance that a cell holds a star is scaled
// by that factor: the result is uniform on the sphere, with no trace of the
// cube. `fp` is the pixel footprint on the source sphere (rad), which
// lensing may stretch or squeeze.
fn star_layer(d: vec3<f32>, cells: f32, density: f32, flux_scale: f32, fp: f32, g: f32, seed: u32) -> vec3<f32> {
    let c = cube_uv(d);
    let face = u32(c.z);
    let grid = (c.xy * 0.5 + 0.5) * cells;
    let cell = floor(grid);
    let ic = vec2<u32>(cell);
    let h = hash3(vec3<u32>(ic.x, ic.y, face + seed * 16u));
    let h2 = hash3(vec3<u32>(ic.y + 911u, ic.x + 17u, face * 7u + seed * 131u + 3u));
    let centre = (cell + 0.5) / cells * 2.0 - 1.0;
    let q = 1.0 + dot(centre, centre);
    let jac = pow(q, -1.5);
    // Angular side of a face-centre cell, and of this one.
    let cell0 = 2.0 / cells;
    let cell_ang = cell0 * pow(q, -0.75);
    // True angular distance to the star (isotropic, unlike grid distance).
    let star_dir = normalize(cube_dir(face, (cell + 0.08 + 0.84 * h.yz) / cells * 2.0 - 1.0));
    let ang = length(d - star_dir);
    let sigma2 = STAR_SIGMA * STAR_SIGMA + fp * fp;
    let present = select(0.0, 1.0, h.x < density * jac);
    let flux = flux_scale * pow(h2.x, 5.0) * present;
    let temp = 2600.0 + 26000.0 * pow(h2.y, 3.5);
    let resolved = flux * exp(-0.5 * ang * ang / sigma2) / (TAU * sigma2);
    // Only the owning cell is evaluated, so a layer is drawn as resolved
    // stars only while their blur is well inside a cell; beyond that it
    // fades to the mean intensity of its (now unresolved) stars, which is
    // the same in every direction.
    let mean = 0.3 * density * flux_scale / 6.0 / (cell0 * cell0);
    let w = smoothstep(0.1, 0.25, fp / cell_ang);
    let resolved_col = blackbody(temp * g) * resolved;
    let mean_col = blackbody(5200.0 * g) * mean;
    return mix(resolved_col, mean_col, w);
}

fn value_noise(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let o = vec3<u32>(vec3<i32>(i) + vec3<i32>(4096));
    let n000 = hash3(o).x;
    let n100 = hash3(o + vec3<u32>(1u, 0u, 0u)).x;
    let n010 = hash3(o + vec3<u32>(0u, 1u, 0u)).x;
    let n110 = hash3(o + vec3<u32>(1u, 1u, 0u)).x;
    let n001 = hash3(o + vec3<u32>(0u, 0u, 1u)).x;
    let n101 = hash3(o + vec3<u32>(1u, 0u, 1u)).x;
    let n011 = hash3(o + vec3<u32>(0u, 1u, 1u)).x;
    let n111 = hash3(o + vec3<u32>(1u, 1u, 1u)).x;
    return mix(
        mix(mix(n000, n100, u.x), mix(n010, n110, u.x), u.y),
        mix(mix(n001, n101, u.x), mix(n011, n111, u.x), u.y),
        u.z,
    );
}

fn fbm(p: vec3<f32>) -> f32 {
    var s = 0.0;
    var a = 0.5;
    var q = p;
    for (var i = 0; i < 4; i++) {
        s += a * value_noise(q);
        q = q * 2.03 + vec3<f32>(1.7, 9.2, 3.1);
        a *= 0.5;
    }
    return s;
}

fn sky_radiance(dir: vec3<f32>, g: f32, fp: f32) -> vec3<f32> {
    // The galactic plane is tilted with respect to the hole's spin axis.
    let gx = normalize(vec3<f32>(0.83, 0.0, 0.56));
    let gz = normalize(vec3<f32>(-0.56, 0.12, 0.83));
    let gy = cross(gz, gx);
    let d = vec3<f32>(dot(dir, gx), dot(dir, gy), dot(dir, gz));
    let lat = asin(clamp(d.z, -1.0, 1.0));
    let gc = clamp(g, 0.02, 40.0);
    let g4 = gc * gc * gc * gc;

    // Diffuse starlight: disc band with dust lanes, plus the bulge around us.
    let band = exp(-pow(lat / 0.16, 2.0));
    let dust = smoothstep(0.42, 0.72, fbm(d * 7.0 + vec3<f32>(3.0)));
    let lanes = 1.0 - 0.85 * dust * exp(-pow(lat / 0.07, 2.0));
    let clumps = 0.55 + 0.9 * fbm(d * 18.0);
    let bulge = 0.35 * exp(-pow(lat / 0.55, 2.0));
    var diffuse = (band * clumps * lanes * 1.4 + bulge + 0.04) * 0.008;
    diffuse *= 1.0 + 0.5 * pow(max(d.x, 0.0), 6.0);
    let diffuse_col = blackbody(4300.0 * g) * diffuse;

    // Point stars in three magnitude classes; denser near the plane.
    let dens = 0.35 + 0.65 * band;
    var stars = star_layer(dir, 24.0, 0.5, 6.0e-4, fp, g, 1u);
    stars += star_layer(dir, 60.0, 0.55, 1.5e-4, fp, g, 2u);
    stars += star_layer(dir, 170.0, 0.6 * dens, 3.0e-5, fp, g, 3u);
    stars += star_layer(dir, 420.0, 0.8 * dens, 6.0e-6, fp, g, 4u);

    return (diffuse_col + stars) * g4;
}

// ---------------------------------------------------------------------------

@fragment
fn fs_sky(in: VsOut) -> @location(0) vec4<f32> {
    let n = ndc_to_dir(in.ndc);
    var s = backward_ray(n);
    let rp = frame.kerr.z;
    let r_esc = max(frame.kerr.w, 2.5 * length(frame.obs.yzw));
    // 0 sky, 1 horizon, 2 panel, 3 ran out of steps (photon-ring orbits)
    var kind = 3u;
    var color = vec3<f32>(0.0);
    var esc_dir = n;
    var g = 1.0;
    for (var i = 0u; i < frame.counts2.x; i++) {
        let pos = s.x.yzw;
        let r = ks_radius(pos);
        if (r - rp < frame.march.w) {
            kind = 1u;
            break;
        }
        let k1 = phase_rhs(s);
        let h = ray_step(s, k1, r);
        let nxt = rk4(s, k1, h);
        if (frame.counts2.y > 0u) {
            let ph = panel_hit(s.x, nxt.x, nxt.p);
            if (ph.hit) {
                kind = 2u;
                color = ph.color;
                break;
            }
        }
        s = nxt;
        if (r > r_esc && dot(pos, k1.x.yzw) > 0.0) {
            kind = 0u;
            esc_dir = normalize(phase_rhs(s).x.yzw);
            g = 1.0 / s.p.x;
            break;
        }
    }
    // Footprint of this pixel on the celestial sphere (lensing included).
    let fp = max(length(fwidth(esc_dir)), 0.25 * frame.cam.z);
    if (kind == 0u) {
        color = sky_radiance(esc_dir, g, fp);
    }
    return vec4<f32>(color, 1.0);
}
