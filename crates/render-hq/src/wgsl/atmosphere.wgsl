// ---------------------------------------------------------------------------
// Atmospheres and clouds (hooks called by `near.wgsl` and `planet.wgsl`).
//
// Positions are planet-centred and inertial (km). Parameters for the planet
// in slot `p.ids.w` are `atmospheres[p.ids.w]` (see `atmosphere.rs`); the
// model and the table layouts are in `atmo_common.wgsl`, the tables are
// built by `atmo_luts.wgsl`, clouds are in `clouds.wgsl`.
//
// The view ray is marched through the shell per pixel (Hillaire 2020 from
// space): sunlight at each sample comes from the transmittance table
// (planet shadow included, so the terminator's reddening, the twilight
// wedge and the limb glow need no special cases), light scattered more
// than once from the multiple-scattering table. Samples are spaced evenly
// in air density rather than distance, so a ray straight down puts half its
// samples in the lowest scale height and a limb ray clusters them around
// its perigee.
//
// This code runs inside the one big trace kernel, whose register use sets
// the occupancy of every pixel: spectral coefficients are read from the
// storage buffer where they are used, and the march works on four bins at
// a time.
// ---------------------------------------------------------------------------

@group(2) @binding(0) var<storage, read> atmospheres: array<AtmosphereParams>;
@group(2) @binding(1) var atmo_trans_lut: texture_2d_array<f32>;
@group(2) @binding(2) var atmo_ms_lut: texture_2d_array<f32>;
@group(2) @binding(3) var atmo_irr_lut: texture_2d_array<f32>;
@group(2) @binding(4) var cloud_maps: texture_cube_array<f32>;
@group(2) @binding(5) var cloud_noise: texture_3d<f32>;
@group(2) @binding(6) var atmo_clamp: sampler;
@group(2) @binding(7) var atmo_repeat: sampler;

// Most samples of a view ray through the air (split over the parts of the
// ray on either side of its perigee and around the clouds).
const AIR_STEPS: f32 = 20.0;

fn atmo_geom(slot: u32) -> AtmoGeom {
    return AtmoGeom(
        atmospheres[slot].shape,
        atmospheres[slot].aerosol,
        atmospheres[slot].clouds,
        atmospheres[slot].extra,
        atmospheres[slot].ids,
    );
}

// Transmittance for density-weighted columns c (km).
fn atmo_columns_transmit(slot: u32, c: vec3<f32>) -> Spectrum {
    return Spectrum(
        exp(-(atmospheres[slot].rayleigh_extinction.a * c.x + atmospheres[slot].mie_extinction.a * c.y
            + atmospheres[slot].ozone.a * c.z)),
        exp(-(atmospheres[slot].rayleigh_extinction.b * c.x + atmospheres[slot].mie_extinction.b * c.y
            + atmospheres[slot].ozone.b * c.z)),
        exp(-(atmospheres[slot].rayleigh_extinction.c * c.x + atmospheres[slot].mie_extinction.c * c.y
            + atmospheres[slot].ozone.c * c.z)),
        exp(-(atmospheres[slot].rayleigh_extinction.d * c.x + atmospheres[slot].mie_extinction.d * c.y
            + atmospheres[slot].ozone.d * c.z)),
    );
}

fn atmo_columns(g: AtmoGeom, slot: u32, r: f32, mu: f32) -> vec3<f32> {
    return textureSampleLevel(atmo_trans_lut, atmo_clamp, atmo_trans_uv(g, r, mu), slot, 0.0).xyz;
}

// Transmittance from radius r to the top of the atmosphere along direction
// cosine mu (held at the horizon's value below it).
fn atmo_air_transmittance(g: AtmoGeom, slot: u32, r: f32, mu: f32) -> Spectrum {
    return atmo_columns_transmit(slot, atmo_columns(g, slot, r, mu));
}

// Transmittance of sunlight through the air from radius r (inside the
// atmosphere) with the sun at zenith cosine mu_s, including the part of
// the sun's disc hidden by the planet.
fn atmo_sun_air(g: AtmoGeom, slot: u32, r: f32, mu_s: f32, ang: f32) -> Spectrum {
    let vis = atmo_sun_above_horizon(g, r, mu_s, ang);
    if (vis <= 0.0) {
        return spec(0.0);
    }
    return spec_scale(atmo_air_transmittance(g, slot, r, mu_s), vis);
}

// Skylight on a horizontal surface per unit solar irradiance.
fn atmo_irradiance(g: AtmoGeom, slot: u32, r: f32, mu_s: f32) -> Spectrum {
    let uv = atmo_ms_uv(g, r, mu_s, ATMO_IRR_SIZE);
    let l = slot * 4u;
    return Spectrum(
        textureSampleLevel(atmo_irr_lut, atmo_clamp, uv, l, 0.0),
        textureSampleLevel(atmo_irr_lut, atmo_clamp, uv, l + 1u, 0.0),
        textureSampleLevel(atmo_irr_lut, atmo_clamp, uv, l + 2u, 0.0),
        textureSampleLevel(atmo_irr_lut, atmo_clamp, uv, l + 3u, 0.0),
    );
}

// A straight ray x0 + t dir seen from its perigee: distance tp to it,
// radius rp there.
struct AtmoRay {
    x0: vec3<f32>,
    dir: vec3<f32>,
    tp: f32,
    rp: f32,
}

fn atmo_ray(x0: vec3<f32>, dir: vec3<f32>) -> AtmoRay {
    let tp = -dot(x0, dir);
    return AtmoRay(x0, dir, tp, length(x0 + tp * dir));
}

// Altitude at t along the ray, without cancellation near the perigee.
fn atmo_ray_height(g: AtmoGeom, ray: AtmoRay, t: f32) -> f32 {
    let s = t - ray.tp;
    let r = sqrt(ray.rp * ray.rp + s * s);
    return (ray.rp - g.shape.x) + s * s / (r + ray.rp);
}

// Where along the ray (on the side `side` = ±1 of the perigee) the altitude
// is h.
fn atmo_ray_at_height(g: AtmoGeom, ray: AtmoRay, h: f32, side: f32) -> f32 {
    let dh = h - (ray.rp - g.shape.x);
    let q = sqrt(max(dh * (g.shape.x + h + ray.rp), 0.0));
    return ray.tp + side * q;
}

// What one air sample needs, shared by the four groups of bins.
struct AirSample {
    d: vec3<f32>,    // relative densities
    col: vec3<f32>,  // columns towards the sun
    vis: f32,        // visible fraction of the sun's disc
    dt: f32,
    pr: f32,         // Rayleigh phase
    pm: f32,         // aerosol phase
    uv: vec2<f32>,   // multiple-scattering table coordinates
    layer: u32,
}

struct Chunk {
    L: vec4<f32>,
    T: vec4<f32>,
}

// One step for four bins: in-scattering σ_s (T_sun P + Ψ_ms), integrated
// exactly over the step for constant coefficients (Hillaire's
// energy-conserving form).
fn atmo_air_chunk(
    m: Chunk,
    ext_r: vec4<f32>,
    ext_m: vec4<f32>,
    oz: vec4<f32>,
    sca_r: vec4<f32>,
    sca_m: vec4<f32>,
    s: AirSample,
    k: u32,
) -> Chunk {
    let ms = textureSampleLevel(atmo_ms_lut, atmo_clamp, s.uv, s.layer + k, 0.0);
    let sig_t = ext_r * s.d.x + ext_m * s.d.y + oz * s.d.z;
    let ts = exp(-(ext_r * s.col.x + ext_m * s.col.y + oz * s.col.z)) * s.vis;
    let src = sca_r * (s.d.x * (ts * s.pr + ms)) + sca_m * (s.d.y * (ts * s.pm + ms));
    let e = exp(-sig_t * s.dt);
    return Chunk(m.L + m.T * (1.0 - e) / max(sig_t, vec4<f32>(1e-9)) * src, m.T * e);
}

// Light scattered by the air along [t0, t1] of the ray, where the ray's
// altitude changes monotonically (`side` −1: descending, +1: ascending),
// per unit solar irradiance. Sample boundaries are evenly spaced in
// exp(−h/H_s).
fn atmo_air_piece(g: AtmoGeom, slot: u32, ray: AtmoRay, t0: f32, t1: f32, side: f32, sun: SunLight) -> Medium {
    let one = Chunk(vec4<f32>(0.0), vec4<f32>(1.0));
    var ca = one;
    var cb = one;
    var cc = one;
    var cd = one;
    if (t1 - t0 >= 1e-4) {
        let c = dot(ray.dir, sun.dir);
        let pr = atmo_phase_rayleigh(c);
        let pm = atmo_phase_mie(g, c);
        let hs = 1.5 * g.shape.z;
        let rho0 = exp(-atmo_ray_height(g, ray, t0) / hs);
        let rho1 = exp(-atmo_ray_height(g, ray, t1) / hs);
        let n = clamp(ceil(AIR_STEPS * sqrt(abs(rho1 - rho0))), 3.0, AIR_STEPS);
        let ni = u32(n);
        var ta = t0;
        for (var i = 1u; i <= ni; i++) {
            var tb = t1;
            if (i < ni) {
                let rho = mix(rho0, rho1, f32(i) / n);
                tb = clamp(atmo_ray_at_height(g, ray, -hs * log(rho), side), ta, t1);
            }
            let t = 0.5 * (ta + tb);
            let x = ray.x0 + t * ray.dir;
            let h = atmo_ray_height(g, ray, t);
            let r = g.shape.x + h;
            let mu_s = dot(x, sun.dir) / r;
            let s = AirSample(
                atmo_density(g, h),
                atmo_columns(g, slot, r, mu_s),
                atmo_sun_above_horizon(g, r, mu_s, sun.angular_radius),
                tb - ta,
                pr,
                pm,
                atmo_ms_uv(g, r, mu_s, ATMO_MS_SIZE),
                slot * 4u,
            );
            ca = atmo_air_chunk(
                ca,
                atmospheres[slot].rayleigh_extinction.a,
                atmospheres[slot].mie_extinction.a,
                atmospheres[slot].ozone.a,
                atmospheres[slot].rayleigh_scattering.a,
                atmospheres[slot].mie_scattering.a,
                s,
                0u,
            );
            cb = atmo_air_chunk(
                cb,
                atmospheres[slot].rayleigh_extinction.b,
                atmospheres[slot].mie_extinction.b,
                atmospheres[slot].ozone.b,
                atmospheres[slot].rayleigh_scattering.b,
                atmospheres[slot].mie_scattering.b,
                s,
                1u,
            );
            cc = atmo_air_chunk(
                cc,
                atmospheres[slot].rayleigh_extinction.c,
                atmospheres[slot].mie_extinction.c,
                atmospheres[slot].ozone.c,
                atmospheres[slot].rayleigh_scattering.c,
                atmospheres[slot].mie_scattering.c,
                s,
                2u,
            );
            cd = atmo_air_chunk(
                cd,
                atmospheres[slot].rayleigh_extinction.d,
                atmospheres[slot].mie_extinction.d,
                atmospheres[slot].ozone.d,
                atmospheres[slot].rayleigh_scattering.d,
                atmospheres[slot].mie_scattering.d,
                s,
                3u,
            );
            ta = tb;
        }
    }
    return Medium(Spectrum(ca.L, cb.L, cc.L, cd.L), Spectrum(ca.T, cb.T, cc.T, cd.T));
}

// The air along [t0, t1] (split at the perigee).
fn atmo_air(g: AtmoGeom, slot: u32, ray: AtmoRay, t0: f32, t1: f32, sun: SunLight) -> Medium {
    if (t1 <= t0) {
        return medium_clear();
    }
    var m: Medium;
    if (ray.tp <= t0) {
        m = atmo_air_piece(g, slot, ray, t0, t1, 1.0, sun);
    } else if (ray.tp >= t1) {
        m = atmo_air_piece(g, slot, ray, t0, t1, -1.0, sun);
    } else {
        m = medium_over(
            atmo_air_piece(g, slot, ray, t0, ray.tp, -1.0, sun),
            atmo_air_piece(g, slot, ray, ray.tp, t1, 1.0, sun),
        );
    }
    m.L = spec_mul(m.L, sun.irradiance);
    return m;
}

// Light scattered towards the eye along o + u dir for u in [entry, u_max]
// (the part inside the atmosphere), and the transmittance to what lies at
// u_max (the surface when `hit_surface`, else whatever is behind).
//
// Clouds are marched on their own; the air is then split at the clouds'
// transmittance-weighted depth: air in front, the clouds, air behind them
// (the usual aerial-perspective approximation, exact when the cloud is a
// thin layer).
fn atmo_segment(p: Planet, o: vec3<f32>, dir: vec3<f32>, u_max: f32, hit_surface: bool, sun: SunLight, fp: f32) -> Medium {
    let slot = p.ids.w;
    let g = atmo_geom(slot);
    if (!atmo_present(g)) {
        return medium_clear();
    }
    let span = sphere_span(o, dir, atmo_top(g));
    let u0 = max(span.x, 0.0);
    let u1 = min(span.y, u_max);
    if (span.x > span.y || u1 <= u0) {
        return medium_clear();
    }
    // Work from the entry point, where coordinates are small and precise.
    let ray = atmo_ray(o + u0 * dir, dir);
    let len = u1 - u0;
    let cl = clouds_march(p, g, ray, u0, len, sun, fp);
    if (cl.T >= 0.999) {
        return atmo_air(g, slot, ray, 0.0, len, sun);
    }
    var m = medium_over(atmo_air(g, slot, ray, 0.0, cl.depth, sun), Medium(cl.L, spec(cl.T)));
    if (cl.T > 0.002) {
        m = medium_over(m, atmo_air(g, slot, ray, cl.depth, len, sun));
    } else {
        m.T = spec(0.0);
    }
    return m;
}

// Transmittance of sunlight reaching planet-centred position x (includes
// the planet's own shadow and cloud shadows).
fn atmo_sun_transmittance(p: Planet, x: vec3<f32>, sun: SunLight) -> Spectrum {
    let g = atmo_geom(p.ids.w);
    if (!atmo_present(g)) {
        if (dot(x, sun.dir) < 0.0 && length(x - dot(x, sun.dir) * sun.dir) < p.centre.w) {
            return spec(0.0);
        }
        return spec(1.0);
    }
    var xs = x;
    let rt = atmo_top(g);
    if (dot(x, x) > rt * rt) {
        // Outside: start where the sunward ray enters the atmosphere.
        let span = sphere_span(x, sun.dir, rt);
        if (span.x > span.y || span.y < 0.0) {
            return spec(1.0);
        }
        xs = x + max(span.x, 0.0) * sun.dir;
    }
    let r = max(length(xs), g.shape.x);
    let mu_s = dot(xs, sun.dir) / length(xs);
    let t = atmo_sun_air(g, p.ids.w, r, mu_s, sun.angular_radius);
    return spec_scale(t, clouds_sun_transmittance(p, g, xs, sun));
}

// Irradiance on a surface at x with normal `nrm` from the sky (skylight,
// excluding the direct sun), W m⁻² nm⁻¹. The tables give it on a horizontal
// surface; a tilted one sees the fraction (1 + n·up)/2 of the sky. Under
// clouds the sky is replaced by the light diffusing through them.
fn atmo_sky_irradiance(p: Planet, x: vec3<f32>, nrm: vec3<f32>, sun: SunLight) -> Spectrum {
    let slot = p.ids.w;
    let g = atmo_geom(slot);
    if (!atmo_present(g)) {
        return spec(0.0);
    }
    let r = length(x);
    let up = x / r;
    let mu_s = dot(up, sun.dir);
    let rr = clamp(r, g.shape.x, atmo_top(g));
    let sky = atmo_irradiance(g, slot, rr, mu_s);
    let tilt = 0.5 * (1.0 + dot(nrm, up));
    let cl = clouds_diffuse(p, g, x);
    if (cl.y <= 0.0) {
        return spec_scale(spec_mul(sky, sun.irradiance), tilt);
    }
    // Two-stream diffuse transmission of the cloud column (g ≈ 0.85):
    // t = 1 / (1 + ¾ (1 − g) τ); what is not transmitted directly arrives
    // diffuse.
    let top_r = g.shape.x + g.clouds.y + g.clouds.z;
    let sun_top = atmo_sun_air(g, slot, top_r, mu_s, sun.angular_radius);
    let t_diffuse = 1.0 / (1.0 + 0.1125 * cl.x);
    let direct = exp(-cl.x / max(mu_s, 0.05));
    let through = max(mu_s, 0.0) * max(t_diffuse - direct, 0.0);
    let e = spec_axpy(sun_top, through, spec_scale(sky, t_diffuse));
    return spec_scale(spec_mul(e, sun.irradiance), tilt);
}
