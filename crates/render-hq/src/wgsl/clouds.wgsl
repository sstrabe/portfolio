// ---------------------------------------------------------------------------
// Clouds: a layer between base and base + thickness, fixed to the rotating
// planet (`planet_body`). Density is the weather cube map (coverage, top
// height, density, cellularity; see `cloud_gen.wgsl`) shaped by the tiled
// Perlin–Worley volume at two scales and eroded by Worley detail whose
// octaves fade out as the pixel footprint grows.
//
// Lighting per sample:
// - sunlight through the air (transmittance table) and through the clouds
//   towards the sun (a 4-sample light march, so tall clouds shade their
//   neighbours at low sun);
// - a two-lobe Henyey–Greenstein phase function with Wrenninge's
//   multiple-scattering octaves (extinction, contribution and anisotropy
//   halved per octave), and a light "powder" darkening of sunlit edges seen
//   with the sun behind the viewer;
// - skylight from the irradiance table and light reflected from below.
// Cloud droplets scatter conservatively and greyly in the visible, so the
// cloud's own extinction is a scalar.
// ---------------------------------------------------------------------------

// Tile periods (km) of the noise volume: cloud masses, a finer octave,
// marine stratocumulus cells, and erosion detail.
const CLOUD_MASS_KM: f32 = 90.0;
const CLOUD_MASS2_KM: f32 = 23.0;
const CLOUD_CELL_KM: f32 = 220.0;
const CLOUD_DETAIL_KM: f32 = 9.0;
const CLOUD_MAX_STEPS: f32 = 32.0;

fn clouds_present(a: AtmoGeom) -> bool {
    return a.ids.z != ATMO_NO_CLOUDS && a.clouds.x > 0.0 && a.clouds.z > 0.0;
}

// Extinction (km⁻¹) at body-fixed position q (km), height fraction hf in
// the layer; `fp_km` is the pixel footprint there (km, large to skip
// detail).
fn clouds_density(a: AtmoGeom, q: vec3<f32>, hf: f32, fp_km: f32, full: bool) -> f32 {
    if (hf <= 0.0 || hf >= 1.0) {
        return 0.0;
    }
    let w = textureSampleLevel(cloud_maps, atmo_clamp, q, i32(a.ids.z), 0.0);
    let cov = w.r;
    if (cov < 0.01 || hf >= w.g) {
        return 0.0;
    }
    // Height profile: a soft base, tops rounded below the local top height.
    let prof = smoothstep(0.0, 0.15, hf) * (1.0 - smoothstep(0.5 * w.g, w.g, hf));
    var mass = textureSampleLevel(cloud_noise, atmo_repeat, q / CLOUD_MASS_KM, 0.0).r;
    if (full) {
        mass = 0.65 * mass + 0.35 * textureSampleLevel(cloud_noise, atmo_repeat, q / CLOUD_MASS2_KM, 0.0).r;
    }
    var base = mass;
    if (full && w.a > 0.01) {
        let cells = textureSampleLevel(cloud_noise, atmo_repeat, q / CLOUD_CELL_KM, 0.0).g;
        base = mix(mass, 0.35 + 0.65 * cells * cells, w.a);
    }
    var d = clamp((base - (1.0 - cov)) / max(cov, 0.1), 0.0, 1.0);
    d = clamp(d - (1.0 - prof), 0.0, 1.0);
    if (d <= 0.0) {
        return 0.0;
    }
    let fade = 1.0 - smoothstep(0.8, 4.0, fp_km);
    if (full && fade > 0.0) {
        let n = textureSampleLevel(cloud_noise, atmo_repeat, q / CLOUD_DETAIL_KM, 0.0);
        let e = 0.45 * fade * dot(n.gba, vec3<f32>(0.5, 0.3, 0.2));
        d = clamp((d - e) / (1.0 - e), 0.0, 1.0);
    }
    return d * w.b * 1.6 * a.clouds.w / a.clouds.z;
}

fn clouds_phase(c: f32, k: f32) -> f32 {
    return 0.75 * atmo_phase_hg(0.8 * k, c) + 0.25 * atmo_phase_hg(-0.35 * k, c);
}

struct CloudHit {
    L: Spectrum,
    T: f32,
    depth: f32,  // transmittance-weighted distance of the cloud light (from the ray's start)
}

// Parts of [0, len] of the ray inside the cloud layer (up to two).
fn clouds_spans(a: AtmoGeom, ray: AtmoRay, len: f32) -> vec4<f32> {
    let none = vec4<f32>(1.0, 0.0, 1.0, 0.0);
    let rb = a.shape.x + a.clouds.y;
    let rt = rb + a.clouds.z;
    if (ray.rp >= rt) {
        return none;
    }
    let qo = sqrt((rt - ray.rp) * (rt + ray.rp));
    var s = vec4<f32>(ray.tp - qo, ray.tp + qo, 1.0, 0.0);
    if (ray.rp < rb) {
        let qi = sqrt((rb - ray.rp) * (rb + ray.rp));
        s = vec4<f32>(ray.tp - qo, ray.tp - qi, ray.tp + qi, ray.tp + qo);
    }
    return clamp(s, vec4<f32>(0.0), vec4<f32>(len));
}

// March the cloud layer along the ray (x0 at global distance u0 from the
// eye, so rest-frame time is −(u0 + t)).
//
// Only scalars are accumulated per step: sunlight through the air and the
// ambient light are spectra taken once, where the ray first meets cloud
// (the lit surface that dominates what is seen), while the planet's shadow
// is applied per step. This keeps the march cheap and its register use low.
fn clouds_march(p: Planet, a: AtmoGeom, ray: AtmoRay, u0: f32, len: f32, sun: SunLight, fp: f32) -> CloudHit {
    var hit = CloudHit(spec(0.0), 1.0, len);
    if (!clouds_present(a)) {
        return hit;
    }
    let spans = clouds_spans(a, ray, len);
    let l0 = max(spans.y - spans.x, 0.0);
    let l1 = max(spans.w - spans.z, 0.0);
    if (l0 + l1 <= 0.0) {
        return hit;
    }
    let rb = a.shape.x + a.clouds.y;
    let thick = a.clouds.z;
    let rt = rb + thick;
    let c = dot(ray.dir, sun.dir);
    let ph0 = clouds_phase(c, 1.0);
    let ph1 = 0.5 * clouds_phase(c, 0.5);
    let ph2 = 0.25 * clouds_phase(c, 0.25);
    let powder_w = 0.3 * (1.0 - c);
    let jitter = hash3(bitcast<vec3<u32>>(ray.dir)).x;
    // About 8 steps through the layer's thickness, fewer (longer) for long
    // grazing paths.
    let n_total = clamp(ceil((l0 + l1) / (thick / 8.0)), 6.0, CLOUD_MAX_STEPS);
    let dt = (l0 + l1) / n_total;
    var first = -1.0;
    var sun_w = 0.0;
    var amb_w = 0.0;
    var wsum = 0.0;
    var dsum = 0.0;
    for (var k = 0.0; k < n_total; k += 1.0) {
        // Position along the concatenation of the two spans.
        let s = (k + jitter) * dt;
        let t = select(spans.x + s, spans.z + (s - l0), s > l0);
        let x = ray.x0 + t * ray.dir;
        let r = length(x);
        let hf = (r - rb) / thick;
        let u = u0 + t;
        let sigma = clouds_density(a, planet_body(p, x, -u), hf, fp * u, true);
        if (sigma <= 1e-4) {
            continue;
        }
        if (first < 0.0) {
            first = t;
        }
        let mu_s = dot(x, sun.dir) / r;
        let vis = atmo_sun_above_horizon(a, r, mu_s, sun.angular_radius);
        var oct = 0.0;
        if (vis > 0.0) {
            // Light march towards the sun, quadratically spaced.
            let ls = min(atmo_dist_to_sphere(r, mu_s, rt), 20.0);
            var tau = 0.0;
            for (var j = 0.0; j < 3.0; j += 1.0) {
                let xs = x + ls * (j + 0.5) * (j + 0.5) / 9.0 * sun.dir;
                let d = clouds_density(a, planet_body(p, xs, -u), (length(xs) - rb) / thick, 1e3, false);
                tau += d * ls * (2.0 * j + 1.0) / 9.0;
            }
            let beer = exp(-tau);
            let powder = mix(1.0, 1.0 - exp(-2.0 * tau - 0.1), powder_w);
            oct = vis * (ph0 * beer * powder + ph1 * exp(-0.5 * tau) + ph2 * exp(-0.25 * tau));
        }
        let e = exp(-sigma * dt);
        let wgt = hit.T * (1.0 - e);
        sun_w += wgt * oct;
        amb_w += wgt * (0.4 + 0.6 * hf);
        wsum += wgt;
        dsum += wgt * t;
        hit.T *= e;
        if (hit.T < 0.005) {
            hit.T = 0.0;
            break;
        }
    }
    if (wsum <= 0.0) {
        return hit;
    }
    hit.depth = dsum / wsum;
    // Spectra at the first cloud met: sunlight through the air, skylight
    // from above and sunlight reflected from below the layer (isotropic,
    // half a sphere each: σ E / 2π).
    let x = ray.x0 + first * ray.dir;
    let r = length(x);
    let mu_s = dot(x, sun.dir) / r;
    let slot = p.ids.w;
    let ts = atmo_air_transmittance(a, slot, r, mu_s);
    let sky = atmo_irradiance(a, slot, r, mu_s);
    let below = spec_scale(atmo_sun_air(a, slot, rb, mu_s, sun.angular_radius), max(mu_s, 0.0) * a.extra.x);
    let amb = spec_scale(spec_add(sky, below), amb_w / (2.0 * PI));
    hit.L = spec_mul(sun.irradiance, spec_axpy(ts, sun_w, amb));
    return hit;
}

// Transmittance of sunlight through the cloud layer to x (planet centred).
// Body-fixed coordinates are taken at rest-frame time 0: the surface hook
// doesn't pass its time, and the planet turns by ω·u over the light travel
// time u, far below a pixel whenever the planet is resolved.
fn clouds_sun_transmittance(p: Planet, a: AtmoGeom, x: vec3<f32>, sun: SunLight) -> f32 {
    if (!clouds_present(a)) {
        return 1.0;
    }
    let r = length(x);
    let mu_s = dot(x, sun.dir) / r;
    let rb = a.shape.x + a.clouds.y;
    let thick = a.clouds.z;
    var tau = 0.0;
    for (var j = 0.0; j < 3.0; j += 1.0) {
        let hf = (j + 0.5) / 3.0;
        let rj = rb + hf * thick;
        if (r >= rj) {
            continue;
        }
        let xj = x + atmo_dist_to_sphere(r, mu_s, rj) * sun.dir;
        let mu_j = dot(xj, sun.dir) / rj;
        tau += clouds_density(a, planet_body(p, xj, 0.0), hf, 1e3, true) * thick / 3.0 / max(mu_j, 0.03);
    }
    return exp(-tau);
}

// Vertical optical depth of the cloud column above x, and whether there is
// a cloud layer above x at all (0/1).
fn clouds_diffuse(p: Planet, a: AtmoGeom, x: vec3<f32>) -> vec2<f32> {
    if (!clouds_present(a)) {
        return vec2<f32>(0.0);
    }
    let r = length(x);
    let rb = a.shape.x + a.clouds.y;
    if (r >= rb + a.clouds.z) {
        return vec2<f32>(0.0);
    }
    let up = x / r;
    var tau = 0.0;
    for (var j = 0.0; j < 3.0; j += 1.0) {
        let hf = (j + 0.5) / 3.0;
        let q = planet_body(p, up * (rb + hf * a.clouds.z), 0.0);
        tau += clouds_density(a, q, hf, 1e3, true) * a.clouds.z / 3.0;
    }
    return vec2<f32>(tau, 1.0);
}
