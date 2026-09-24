// ---------------------------------------------------------------------------
// Planet surfaces and rings (hooks called by `near.wgsl`).
//
// All positions are planet-centred and inertial (km); `planet_body` turns
// them into body-fixed coordinates at rest-frame time t. A ray is o + u dir
// with |dir| = 1; rest-frame time along it is t = −u / scale.
//
// Placeholder: a smooth sphere with a flat albedo per planet kind.
// ---------------------------------------------------------------------------

struct SurfaceHit {
    hit: bool,
    u: f32,             // distance along the ray (km)
    time: f32,          // rest-frame time of the hit (km of light travel, ≤ 0)
    pos: vec3<f32>,     // planet-centred inertial position (km)
    normal: vec3<f32>,  // shading normal (inertial)
}

fn planet_surface_hit(p: Planet, o: vec3<f32>, dir: vec3<f32>, u_max: f32, fp: f32, scale: f32) -> SurfaceHit {
    var h: SurfaceHit;
    h.hit = false;
    let span = sphere_span(o, dir, p.centre.w);
    if (span.x > span.y || span.y < 0.0 || span.x > u_max) {
        return h;
    }
    h.hit = true;
    h.u = max(span.x, 0.0);
    h.time = -h.u / scale;
    h.pos = o + h.u * dir;
    h.normal = normalize(h.pos);
    return h;
}

fn planet_albedo(kind: u32) -> Spectrum {
    switch (kind) {
        case KIND_OCEAN: { return spec_mix(spec(0.06), spec_power_law(-2.0), 0.08); }
        case KIND_DESERT: { return spec_mix(spec(0.15), spec_power_law(2.5), 0.2); }
        case KIND_ICE: { return spec(0.7); }
        case KIND_LAVA: { return spec(0.08); }
        case KIND_GAS_GIANT: { return spec_mix(spec(0.4), spec_power_law(1.5), 0.2); }
        case KIND_ICE_GIANT: { return spec_mix(spec(0.3), spec_power_law(-3.0), 0.2); }
        default: { return spec(0.2); }
    }
}

// Radiance leaving the surface towards `view` (unit, towards the eye).
fn planet_surface_radiance(p: Planet, h: SurfaceHit, view: vec3<f32>, sun: SunLight) -> Spectrum {
    let ndl = max(dot(h.normal, sun.dir), 0.0);
    let t_sun = atmo_sun_transmittance(p, h.pos, sun);
    let direct = spec_scale(spec_mul(sun.irradiance, t_sun), ndl / PI);
    let sky = spec_scale(atmo_sky_irradiance(p, h.pos, h.normal, sun), 1.0 / PI);
    return spec_mul(planet_albedo(p.ids.x), spec_add(direct, sky));
}

struct RingHit {
    hit: bool,
    u: f32,
    L: Spectrum,
    T: Spectrum,
}

// Crossing of the ring plane before `u_max`.
fn planet_rings(p: Planet, o: vec3<f32>, dir: vec3<f32>, u_max: f32, sun: SunLight, scale: f32) -> RingHit {
    var r: RingHit;
    r.hit = false;
    if (p.rings.z <= 0.0) {
        return r;
    }
    let nrm = p.spin.xyz;
    let dn = dot(dir, nrm);
    if (abs(dn) < 1e-8) {
        return r;
    }
    let u = -dot(o, nrm) / dn;
    if (u < 0.0 || u > u_max) {
        return r;
    }
    let x = o + u * dir;
    let rad = length(x);
    if (rad < p.rings.x || rad > p.rings.y) {
        return r;
    }
    let tau = p.rings.z / max(abs(dn), 0.05);
    let a = 1.0 - exp(-tau);
    r.hit = true;
    r.u = u;
    r.T = spec(1.0 - a);
    r.L = spec_scale(sun.irradiance, a * 0.5 * abs(dot(sun.dir, nrm)) / PI);
    return r;
}
