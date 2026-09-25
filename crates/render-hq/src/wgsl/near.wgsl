// ---------------------------------------------------------------------------
// Near field: star systems close to the pilot, traced in each system's
// rest frame.
//
// Over a few AU spacetime near the pilot is flat to ~10⁻⁶ of a pixel, so a
// pixel's past light ray is a straight line in the pilot's local frame.
// Each system moves with velocity β relative to the pilot; a pure boost
// takes the ray into the system's rest frame, where it leaves the observer
// event (the origin) with direction d' and rest-frame time t' = −σ after
// rest-frame distance σ (km). Frequencies there are 1/g of the observed
// ones, so rest-frame radiance is brought back with `spec_shift(L, g)`:
// aberration, Doppler shift and beaming are exact.
//
// In the rest frame the star sits still and planets move slowly (≲10⁻⁴ c),
// each evaluated at the ray's own time: centre(t') = centre(0) + v t'.
// Coordinates are kilometres relative to the observer event; c = 1, so
// times are in km of light travel.
// ---------------------------------------------------------------------------

struct StarSystem {
    beta: vec4<f32>,   // direction of the system's velocity in the ship frame (forward, left, up), |β|
    boost: vec4<f32>,  // γ, 1 − |β| (computed without cancellation), unused, unused
    star: vec4<f32>,   // star centre in the rest frame (km, relative to the observer), radius (km)
    light: vec4<f32>,  // temperature (K), luminosity (W), body index, 1 when the disc is drawn here
    range: vec4<u32>,  // first planet slot, planet count, unused, unused
}

struct Planet {
    centre: vec4<f32>,  // centre at rest-frame time 0 (km, relative to the observer), radius (km)
    vel: vec4<f32>,     // velocity in the rest frame (units of c), gravitational parameter (km³/s²)
    spin: vec4<f32>,    // rotation axis (unit), rotation angle at time 0 (rad)
    axis0: vec4<f32>,   // body-fixed x axis at angle 0 (unit, ⟂ spin), angular speed (rad per km of light travel)
    surface: vec4<f32>, // relief (km), sea level (fraction of relief), equilibrium temperature (K), atmosphere top (km, 0: none)
    rings: vec4<f32>,   // inner radius (km), outer radius (km), optical depth (0: none), dust fraction
    ids: vec4<u32>,     // kind (PlanetKind), seed, system slot, own slot
    detail: vec4<f32>,  // surface wind speed (m/s), unused, unused, unused
}

@group(1) @binding(0) var<storage, read> systems: array<StarSystem>;
@group(1) @binding(1) var<storage, read> planets: array<Planet>;

// Values of `Planet.ids.x`, matching `kerr::planets::PlanetKind`.
const KIND_ROCKY: u32 = 0u;
const KIND_OCEAN: u32 = 1u;
const KIND_DESERT: u32 = 2u;
const KIND_ICE: u32 = 3u;
const KIND_LAVA: u32 = 4u;
const KIND_GAS_GIANT: u32 = 5u;
const KIND_ICE_GIANT: u32 = 6u;

// Light from a system's star as seen at a planet.
struct SunLight {
    dir: vec3<f32>,          // unit, from the planet towards the star
    angular_radius: f32,     // of the star's disc (rad)
    irradiance: Spectrum,    // at normal incidence above the atmosphere, W m⁻² nm⁻¹
    temperature: f32,
}

// Radiance picked up and transmittance along a stretch of ray. Radiance is
// what arrives at the start of the stretch; transmittance multiplies
// whatever lies behind it.
struct Medium {
    L: Spectrum,
    T: Spectrum,
}

fn medium_clear() -> Medium {
    return Medium(spec(0.0), spec(1.0));
}

// Put `back` behind `front`.
fn medium_over(front: Medium, back: Medium) -> Medium {
    return Medium(spec_fma(front.T, back.L, front.L), spec_mul(front.T, back.T));
}

struct RestRay {
    d: vec3<f32>,  // unit direction in the rest frame
    g: f32,        // observed / emitted frequency
}

// The pixel ray (ship-frame direction n) in a system's rest frame.
fn rest_ray(n: vec3<f32>, s: StarSystem) -> RestRay {
    let b = s.beta.w;
    if (b < 1e-9) {
        return RestRay(n, 1.0);
    }
    let bh = s.beta.xyz;
    let gam = s.boost.x;
    let omb = s.boost.y;
    let c = dot(n, bh);
    // 1 + cos, from a chord so it stays accurate when n ≈ −β̂.
    let opc = 0.5 * dot(n + bh, n + bh);
    // Emitted frequency / observed = γ (1 + β·n).
    let inv_g = gam * (omb + b * opc);
    // x' = n⊥ + γ (cos + β) β̂ with cos + β = (1 + cos) − (1 − β).
    let xp = (n - c * bh) + gam * (opc - omb) * bh;
    return RestRay(normalize(xp), 1.0 / inv_g);
}

// Nearest intersection u ≥ 0 of o + u d (|d| = 1) with a sphere of radius
// r at the origin: (entry, exit), or entry > exit when missed. Written to
// avoid squaring the (possibly huge) distance to the centre.
fn sphere_span(o: vec3<f32>, d: vec3<f32>, r: f32) -> vec2<f32> {
    let b = dot(o, d);
    let perp = o - b * d;
    let h2 = r * r - dot(perp, perp);
    if (h2 < 0.0) {
        return vec2<f32>(1.0, -1.0);
    }
    let h = sqrt(h2);
    return vec2<f32>(-b - h, -b + h);
}

// Planet-centred frame at rest-frame time t (km): `x` is a planet-centred
// inertial position, the result is body fixed (rotates with the planet).
fn planet_body(p: Planet, x: vec3<f32>, t: f32) -> vec3<f32> {
    let angle = p.spin.w + p.axis0.w * t;
    let s = p.spin.xyz;
    let e1 = p.axis0.xyz * cos(angle) + cross(s, p.axis0.xyz) * sin(angle);
    let e2 = cross(s, e1);
    return vec3<f32>(dot(x, e1), dot(x, e2), dot(x, s));
}

// Inverse of `planet_body` for directions.
fn planet_inertial(p: Planet, b: vec3<f32>, t: f32) -> vec3<f32> {
    let angle = p.spin.w + p.axis0.w * t;
    let s = p.spin.xyz;
    let e1 = p.axis0.xyz * cos(angle) + cross(s, p.axis0.xyz) * sin(angle);
    let e2 = cross(s, e1);
    return b.x * e1 + b.y * e2 + b.z * s;
}

fn sun_light(sys: StarSystem, p: Planet) -> SunLight {
    let to_star = sys.star.xyz - p.centre.xyz;
    let d = max(length(to_star), 1.0);
    let rs = sys.star.w;
    var sun: SunLight;
    sun.dir = to_star / d;
    sun.angular_radius = asin(min(rs / d, 1.0));
    // E_λ = π B_λ(T) (R/d)².
    sun.irradiance = spec_scale(spec_planck(sys.light.x), PI * (rs / d) * (rs / d));
    sun.temperature = sys.light.x;
    return sun;
}

// A star's photosphere: blackbody with limb darkening and granulation.
fn star_surface(sys: StarSystem, nrm: vec3<f32>, view: vec3<f32>) -> Spectrum {
    let mu = max(dot(nrm, view), 0.0);
    let limb = (1.0 - 0.6 * (1.0 - mu)) / 0.8;
    let cells = 0.9 + 0.2 * value_noise(nrm * 60.0 + vec3<f32>(sys.light.z * 0.37));
    return spec_scale(spec_planck(sys.light.x), limb * cells);
}

// One planet crossed by the rest-frame ray x(σ) = σ d, planet centre
// c(σ) = c0 − v σ. In planet-centred coordinates the ray is o + u dir with
// o = −c0 and dir ∝ d + v; `u_max` limits it (km along dir).
fn planet_trace(sys: StarSystem, p: Planet, d: vec3<f32>, sigma_max: f32, fp: f32) -> Hit {
    let dv = d + p.vel.xyz;
    let scale = length(dv);
    let dir = dv / scale;
    let o = -p.centre.xyz;
    let u_max = sigma_max * scale;
    let sun = sun_light(sys, p);

    var out: Hit;
    out.opaque = false;
    out.sigma = sigma_max;

    let surf = planet_surface_hit(p, o, dir, u_max, fp, scale);
    let u_end = select(u_max, surf.u, surf.hit);
    let atmo = atmo_segment(p, o, dir, u_end, surf.hit, sun, fp);
    var back = medium_clear();
    if (surf.hit) {
        // Under an opaque cloud deck the surface isn't seen: skip shading it.
        var surf_l = spec(0.0);
        if (spec_max_value(atmo.T) > 1e-4) {
            surf_l = planet_surface_radiance(p, surf, -dir, sun);
        }
        back = Medium(surf_l, spec(0.0));
        out.opaque = true;
        out.sigma = surf.u / scale;
    }
    var m = medium_over(atmo, back);
    let ring = planet_rings(p, o, dir, u_end, sun, scale);
    if (ring.hit) {
        let rm = Medium(ring.L, ring.T);
        // Rings lie outside the atmosphere: in front of it if the ray
        // crosses the ring plane before reaching the planet.
        let atmo_entry = sphere_span(o, dir, p.centre.w + p.surface.w).x;
        if (ring.u < atmo_entry || !surf.hit) {
            if (ring.u < atmo_entry) {
                m = medium_over(rm, m);
            } else {
                m = medium_over(m, rm);
            }
        }
    }
    out.m = m;
    return out;
}

struct Hit {
    m: Medium,
    opaque: bool,
    sigma: f32,  // rest-frame distance where the ray stopped
}

const MAX_PLANETS_PER_SYSTEM: u32 = 8u;

// Radiance at direction d from a sub-pixel planet: flux
// E_sun p (R/r)² Φ(α) (Lambert sphere phase, geometric albedo p) spread as
// a Gaussian of the pixel's width; faded out as the disc gets resolved.
fn planet_glint_point(sys: StarSystem, p: Planet, d: vec3<f32>, fp: f32) -> Spectrum {
    let c = p.centre.xyz;
    let r = length(c);
    let ang_radius = p.centre.w / max(r, 1.0);
    let resolved = smoothstep(0.3, 1.0, ang_radius / fp);
    if (resolved >= 1.0) {
        return spec(0.0);
    }
    let dir = c / r;
    let cos_t = dot(d, dir);
    let s2 = fp * fp;
    let t2 = 2.0 * (1.0 - cos_t);
    if (t2 > 16.0 * s2) {
        return spec(0.0);
    }
    let sun = sun_light(sys, p);
    // Phase angle at the planet between the star and the observer.
    let alpha = acos(clamp(dot(sun.dir, -dir), -1.0, 1.0));
    let phase = (sin(alpha) + (PI - alpha) * cos(alpha)) / PI;
    let albedo = select(0.3, 0.5, p.ids.x == KIND_GAS_GIANT || p.ids.x == KIND_ICE_GIANT || p.ids.x == KIND_ICE);
    let flux = albedo * ang_radius * ang_radius * phase * (1.0 - resolved);
    return spec_scale(sun.irradiance, flux * exp(-0.5 * t2 / s2) / (TAU * s2));
}

// Everything of one system along the rest-frame ray direction d.
fn system_trace(sys: StarSystem, d: vec3<f32>, fp: f32) -> Hit {
    var out: Hit;
    out.m = medium_clear();
    out.opaque = false;
    out.sigma = 3.0e38;

    // The star occludes what lies behind it.
    var sigma_star = 3.0e38;
    if (sys.light.w > 0.5) {
        let span = sphere_span(-sys.star.xyz, d, sys.star.w);
        if (span.x <= span.y && span.y > 0.0) {
            sigma_star = max(span.x, 0.0);
        }
    }

    // Planets whose bounds the ray crosses, nearest first.
    var order: array<u32, 8>;
    var entry: array<f32, 8>;
    var count = 0u;
    let first = sys.range.x;
    let n = min(sys.range.y, MAX_PLANETS_PER_SYSTEM);
    for (var i = 0u; i < n; i++) {
        let p = planets[first + i];
        let bound = max(p.centre.w + p.surface.w, p.rings.y);
        let dv = normalize(d + p.vel.xyz);
        let span = sphere_span(-p.centre.xyz, dv, bound);
        if (span.x > span.y || span.y < 0.0 || span.x > sigma_star) {
            continue;
        }
        var j = count;
        while (j > 0u && entry[j - 1u] > span.x) {
            order[j] = order[j - 1u];
            entry[j] = entry[j - 1u];
            j--;
        }
        order[j] = first + i;
        entry[j] = span.x;
        count++;
    }

    // Planets smaller than the pixel: their reflected sunlight as a point
    // source spread over the footprint (energy conserving), so from afar a
    // giant shows as a bright star and an Earth as a faint one.
    for (var i = 0u; i < n; i++) {
        let g = planet_glint_point(sys, planets[first + i], d, fp);
        out.m.L = spec_add(out.m.L, g);
    }

    var limit = sigma_star;
    for (var k = 0u; k < count; k++) {
        if (entry[k] > limit) {
            break;
        }
        let h = planet_trace(sys, planets[order[k]], d, limit, fp);
        out.m = medium_over(out.m, h.m);
        if (h.opaque) {
            out.opaque = true;
            out.sigma = h.sigma;
            return out;
        }
    }
    if (sigma_star < 3.0e38) {
        let x = sigma_star * d - sys.star.xyz;
        let nrm = normalize(x);
        out.m = medium_over(out.m, Medium(star_surface(sys, nrm, -d), spec(0.0)));
        out.opaque = true;
        out.sigma = sigma_star;
    }
    return out;
}

struct NearResult {
    m: Medium,
    opaque: bool,
}

// All nearby systems along ship-frame direction n, in the observer's frame.
// `fp` is the pixel's angular footprint (rad).
fn near_field(n: vec3<f32>, fp: f32) -> NearResult {
    var out = NearResult(medium_clear(), false);
    for (var i = 0u; i < hq.near.x; i++) {
        let sys = systems[i];
        let r = rest_ray(n, sys);
        // Aberration widens the footprint by g in the rest frame (a
        // blueshifted view packs more of the rest-frame sky per pixel).
        let h = system_trace(sys, r.d, fp * r.g);
        let seen = Medium(spec_shift(h.m.L, r.g), spec_resample(h.m.T, r.g));
        out.m = medium_over(out.m, seen);
        if (h.opaque) {
            out.opaque = true;
            break;
        }
    }
    return out;
}
