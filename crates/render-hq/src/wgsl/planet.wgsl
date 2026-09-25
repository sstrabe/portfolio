// ---------------------------------------------------------------------------
// Planet surfaces and rings (hooks called by `near.wgsl`).
//
// All positions are planet-centred and inertial (km); `planet_body` turns
// them into body-fixed coordinates at rest-frame time t. A ray is o + u dir
// with |dir| = 1; rest-frame time along it is t = −u / scale.
//
// Solid worlds: the ray meets the datum sphere (sea level on worlds with
// seas); the terrain of `terrain.wgsl` sets the material, and its gradient
// the shading normal, down to the pixel footprint. Relief is at most a few
// parts in a thousand of the radius, so from orbit its silhouette on the
// limb stays within a pixel or two.
// Giants: the visible surface is the cloud deck at the planet's radius,
// banded by the zonal jets; the atmosphere hook adds the air above it.
// Reflectances are spectra (16 bins) built from laboratory shapes: iron
// oxides' blue absorption, chlorophyll's green peak and red edge, water's
// absorption spectrum, snow's near-flat white.
// ---------------------------------------------------------------------------

struct SurfaceHit {
    hit: bool,
    u: f32,             // distance along the ray (km)
    time: f32,          // rest-frame time of the hit (km of light travel, ≤ 0)
    pos: vec3<f32>,     // planet-centred inertial position (km)
    normal: vec3<f32>,  // shading normal (inertial)
    body: vec3<f32>,    // body-fixed unit direction of the point
    terrain: vec4<f32>, // macro terrain channels (see `terrain_macro`)
    height: f32,        // solid height above the datum (km)
    lod: f32,           // pixel footprint on the ground (km)
    // Terrain tiles (`terrain_rq.wgsl`): whether the hit is on them or their
    // sea, where (km from the anchor, body-fixed), and how much of the sun
    // the terrain hides from it (0 lit, 1 in shadow; zero by default).
    tiled: bool,
    local: vec3<f32>,
    shadow: f32,
    // The share of the sky the terrain hides (0 open; zero by default).
    sky_occlusion: f32,
}

fn planet_terrain(p: Planet) -> TerrainParams {
    return terrain_params(p.ids.x, p.ids.y, p.centre.w, p.surface.x, p.surface.y, p.surface.z, p.surface.w > 0.0);
}

fn planet_is_giant(p: Planet) -> bool {
    return p.ids.x == KIND_GAS_GIANT || p.ids.x == KIND_ICE_GIANT;
}

// Solid height (km) and macro channels at body-fixed direction q.
fn planet_height(tp: TerrainParams, q: vec3<f32>, lod: f32) -> vec4<f32> {
    let m = terrain_macro(tp, q, lod);
    return vec4<f32>(terrain_solid(tp, q, m, lod), m.yzw);
}

fn planet_surface_hit(p: Planet, o: vec3<f32>, dir: vec3<f32>, u_max: f32, fp: f32, scale: f32) -> SurfaceHit {
    // The planet with terrain tiles: its real relief, traced by hardware
    // (`terrain_rq.wgsl`), where the tiles or its sea are met.
    if (terrain_on(p)) {
        let t = terrain_surface_hit(p, o, dir, u_max, fp, scale);
        if (t.hit) {
            return t;
        }
    }
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
    let up = normalize(h.pos);
    h.normal = up;
    h.body = normalize(planet_body(p, h.pos, h.time));
    // Footprint on the ground, stretched where the view grazes it.
    let graze = max(abs(dot(dir, up)), 0.08);
    h.lod = max(fp * h.u / graze, 1e-3);
    if (planet_is_giant(p)) {
        return h;
    }
    let tp = planet_terrain(p);
    let c = planet_height(tp, h.body, h.lod);
    h.height = c.x;
    let m0 = terrain_macro(tp, h.body, h.lod);
    h.terrain = vec4<f32>(m0.x, c.yzw);
    // Shading normal from the height gradient, except over open water
    // (whose roughness is handled statistically by the glint).
    if (!(tp.liquid == FILL_WATER && c.x < 0.0)) {
        let d = max(h.lod, 0.02);
        let ang = d / tp.radius;
        let t1 = normalize(cross(h.body, select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), abs(h.body.z) > 0.9)));
        let t2 = cross(h.body, t1);
        let s0 = terrain_surface(tp, c.x);
        let s1 = terrain_surface(tp, planet_height(tp, normalize(h.body + ang * t1), h.lod).x);
        let s2 = terrain_surface(tp, planet_height(tp, normalize(h.body + ang * t2), h.lod).x);
        let nb = normalize(h.body - (s1 - s0) / d * t1 - (s2 - s0) / d * t2);
        h.normal = normalize(planet_inertial(p, nb, h.time));
    }
    return h;
}

// --- Spectral reflectances --------------------------------------------------

// 0 below l0 nm, 1 above l1, smooth in between.
fn spec_ramp(l0: f32, l1: f32) -> Spectrum {
    let l = spec_lambda();
    let a = vec4<f32>(l0);
    let b = vec4<f32>(l1);
    return Spectrum(smoothstep(a, b, l.a), smoothstep(a, b, l.b), smoothstep(a, b, l.c), smoothstep(a, b, l.d));
}

// A Gaussian bump of height 1 at mu nm.
fn spec_bump(mu: f32, sigma: f32) -> Spectrum {
    let l = spec_lambda();
    let k = -0.5 / (sigma * sigma);
    return Spectrum(
        exp(k * (l.a - mu) * (l.a - mu)),
        exp(k * (l.b - mu) * (l.b - mu)),
        exp(k * (l.c - mu) * (l.c - mu)),
        exp(k * (l.d - mu) * (l.d - mu)),
    );
}

fn refl_basalt() -> Spectrum {
    return spec_axpy(spec_ramp(400.0, 800.0), 0.03, spec(0.055));
}

fn refl_granite() -> Spectrum {
    return spec_axpy(spec_ramp(400.0, 750.0), 0.1, spec(0.2));
}

// Ferric soils and sand: Fe³⁺ absorbs the blue (Mars, deserts).
fn refl_ferric(strength: f32) -> Spectrum {
    return spec_axpy(spec_ramp(470.0, 630.0), 0.32 * strength, spec(0.12 + 0.1 * (1.0 - strength)));
}

// Leaves: chlorophyll's small green peak and the red edge at ~700 nm.
fn refl_vegetation(dryness: f32) -> Spectrum {
    let green = spec_axpy(spec_bump(550.0, 28.0), 0.06, spec(0.035));
    let leaf = spec_axpy(spec_ramp(690.0, 740.0), 0.4, green);
    // Dry grass: yellower, weaker chlorophyll bands. Seen from above,
    // savanna and steppe are olive-brown (scattered trees and shrubs, and
    // their shadows), so some leaf stays in the mix.
    let dry = spec_axpy(spec_ramp(480.0, 650.0), 0.13, spec(0.06));
    return spec_mix(leaf, dry, 0.8 * dryness);
}

// Soils of a living world: pale yellow quartz sand where arid (iron-oxide
// coatings take the blue), dark brown loam where wet (organic matter).
fn refl_soil(dryness: f32) -> Spectrum {
    let sand = spec_axpy(spec_ramp(420.0, 600.0), 0.24, spec(0.17));
    let loam = spec_axpy(spec_ramp(450.0, 700.0), 0.1, spec(0.06));
    return spec_mix(loam, sand, dryness);
}

fn refl_snow() -> Spectrum {
    return spec_axpy(spec_ramp(600.0, 800.0), -0.08, spec(0.9));
}

// Glacier and sea ice: bluish (red absorbed in the ice).
fn refl_ice() -> Spectrum {
    return spec_axpy(spec_ramp(420.0, 750.0), -0.25, spec(0.72));
}

fn refl_salt() -> Spectrum {
    return spec(0.65);
}

// Pure-water absorption (m⁻¹, Pope & Fry 1997; Kou et al. 1993 in the red)
// at the bin centres.
fn water_absorption() -> Spectrum {
    return Spectrum(
        vec4<f32>(0.0066, 0.0048, 0.0092, 0.0114),
        vec4<f32>(0.0204, 0.0487, 0.0565, 0.0894),
        vec4<f32>(0.2224, 0.298, 0.34, 0.425),
        vec4<f32>(0.65, 1.69, 2.61, 2.4),
    );
}

// Diffuse reflectance just below the sea surface over water `depth_m`
// deep above a sandy floor: R∞ = 0.33 b_b / a for deep water (Morel &
// Prieur), blending to the floor with attenuation K ≈ a + b_b both ways.
fn water_reflectance(depth_m: f32) -> Spectrum {
    let a = water_absorption();
    // Backscatter of seawater (∝ λ^−4.32) plus a little from fine particles.
    let bb = spec_add(spec_scale(spec_power_law(-4.32), 0.5 * 0.0019), spec(0.0004));
    let deep = spec_scale(spec_mul(bb, Spectrum(1.0 / (a.a + bb.a), 1.0 / (a.b + bb.b), 1.0 / (a.c + bb.c), 1.0 / (a.d + bb.d))), 0.33);
    let k = spec_scale(spec_add(a, bb), 2.0 * max(depth_m, 0.0));
    let floor_t = spec_transmit(k);
    return spec_fma(floor_t, spec_sub(refl_ferric(0.4), deep), deep);
}

// --- Reflection models ------------------------------------------------------

// Oren–Nayar (roughness σ = 0.35), per unit albedo and irradiance.
fn brdf_oren_nayar(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>) -> f32 {
    let nl = max(dot(n, l), 0.0);
    let nv = max(dot(n, v), 1e-3);
    let s2 = 0.35 * 0.35;
    let a = 1.0 - 0.5 * s2 / (s2 + 0.33);
    let b = 0.45 * s2 / (s2 + 0.09);
    let lp = normalize(l - nl * n + vec3<f32>(1e-6));
    let vp = normalize(v - nv * n + vec3<f32>(1e-6));
    let cphi = max(dot(lp, vp), 0.0);
    let ti = acos(nl);
    let tr = acos(nv);
    let alpha = max(ti, tr);
    let beta = min(ti, tr);
    return (a + b * cphi * sin(alpha) * tan(beta)) / PI;
}

// Airless regolith: Lommel–Seeliger with Hapke's opposition surge, scaled
// to match Lambert at normal incidence and emergence.
fn brdf_regolith(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>) -> f32 {
    let mu0 = max(dot(n, l), 0.0);
    let mu = max(dot(n, v), 1e-3);
    let g = acos(clamp(dot(v, l), -1.0, 1.0));
    let surge = 1.0 / (1.0 + tan(0.5 * g) / 0.06);
    return (2.0 / (mu0 + mu)) * 0.5 * (1.0 + surge) / PI;
}

fn fresnel_water(c: f32) -> f32 {
    let f0 = 0.02;
    let x = 1.0 - clamp(c, 0.0, 1.0);
    return f0 + (1.0 - f0) * x * x * x * x * x;
}

// Sun glint off a wind-roughened sea (Cox & Munk 1954): facet slopes are
// Gaussian with total variance σ² = 0.003 + 5.12e-3 U (U wind, m/s), which
// is a Beckmann distribution; returns radiance per unit solar irradiance.
fn ocean_glint(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, wind: f32) -> f32 {
    let s2 = 0.003 + 5.12e-3 * max(wind, 0.5);
    let hv = normalize(v + l);
    let ch = max(dot(n, hv), 1e-3);
    let c2 = ch * ch;
    let tan2 = (1.0 - c2) / c2;
    let d = exp(-tan2 / s2) / (PI * s2 * c2 * c2);
    let nv = max(dot(n, v), 0.05);
    let nl = max(dot(n, l), 0.0);
    // Smith masking for rough seas at grazing angles (rational fit).
    let g = min(1.0, 2.0 * ch * min(nv, nl) / max(dot(v, hv), 1e-3));
    return fresnel_water(dot(v, hv)) * d * g / (4.0 * nv);
}

// --- Solid worlds ----------------------------------------------------------

struct Material {
    albedo: Spectrum,
    water: bool,      // open water: glint + subsurface instead of the BRDF
    depth_m: f32,     // water depth
    regolith: bool,
    emission: Spectrum,
}

// Surface temperature (K) from the equilibrium temperature, the
// greenhouse effect, latitude and the lapse rate.
fn surface_temperature(tp: TerrainParams, q: vec3<f32>, h: f32) -> f32 {
    let lat_s = q.z * q.z;
    let greenhouse = select(1.0, 1.13, tp.air > 0.5);
    let t = tp.t_eq * greenhouse + 18.0 - 45.0 * lat_s;
    return t - 6.5 * max(h, 0.0) * tp.air;
}

fn material_ocean_world(tp: TerrainParams, q: vec3<f32>, h: f32, m: vec4<f32>, baked: bool) -> Material {
    var mat: Material;
    mat.emission = spec(0.0);
    // Weather and currents make the ice and snow lines ragged: a few K of
    // noise on the temperature.
    let wobble = 6.0 * tn_fbm(q * 9.0, tp.seed + 41u, 4.0, 0.55);
    // The baked climate (winds, rain shadows) when this planet has it.
    var climate = vec4<f32>(0.0);
    if (baked) {
        climate = map_climate(q);
    }
    let temp = select(surface_temperature(tp, q, h), climate.x - 6.5 * max(h - max(m.x, 0.0), 0.0) * tp.air, baked) + wobble;
    if (h < 0.0) {
        // Pack ice below about −2 °C, broken into floes near its edge.
        let floes = tn_fbm(q * 60.0, tp.seed + 43u, 3.0, 0.6);
        if (temp + 3.0 * floes < 268.0) {
            mat.albedo = spec_mix(refl_ice(), refl_snow(), 0.5);
            return mat;
        }
        mat.water = true;
        mat.depth_m = -h * 1000.0;
        return mat;
    }
    let moist = m.z;
    let rock = spec_mix(refl_granite(), refl_basalt(), m.w);
    // Vegetation where it is warm and wet; desert where dry; snow where
    // cold. Mountains are green below the treeline (the temperature sees to
    // that): only the cores of the highest ranges are bare, and rock shows
    // through the thin soils of dry ranges.
    let bare = smoothstep(0.75, 1.0, m.y);
    var veg_w = smoothstep(0.1, 0.4, moist) * smoothstep(266.0, 280.0, temp) * (1.0 - bare);
    var dryness = 1.0 - smoothstep(0.3, 0.7, moist);
    if (baked) {
        // Vegetation from the climate.
        veg_w = climate.z * smoothstep(262.0, 272.0, temp) * (1.0 - bare);
        dryness = climate.w;
    }
    var a = spec_mix(refl_soil(dryness), rock, max(bare, smoothstep(0.4, 0.9, m.y) * dryness));
    a = spec_mix(a, refl_vegetation(dryness), veg_w);
    let snow = smoothstep(271.0, 262.0, temp + 3.0 * (moist - 0.5));
    mat.albedo = spec_mix(a, refl_snow(), snow);
    return mat;
}

fn material_desert(q: vec3<f32>, m: vec4<f32>) -> Material {
    var mat: Material;
    mat.emission = spec(0.0);
    let rock = spec_mix(refl_ferric(0.75), refl_basalt(), 0.35 + 0.3 * m.w);
    var a = spec_mix(rock, refl_ferric(1.0), m.y);
    a = spec_mix(a, refl_salt(), 0.7 * smoothstep(0.4, 0.9, m.z));
    // Thin polar frost caps.
    a = spec_mix(a, refl_snow(), smoothstep(0.93, 0.98, abs(q.z)));
    mat.albedo = a;
    return mat;
}

fn material_ice(m: vec4<f32>) -> Material {
    var mat: Material;
    mat.emission = spec(0.0);
    // Lineae stained red-brown by salts and radiation-processed organics.
    let stain = spec_axpy(spec_ramp(450.0, 700.0), 0.25, spec(0.2));
    var a = spec_mix(refl_snow(), refl_ice(), 0.4 + 0.3 * m.z);
    mat.albedo = spec_mix(a, stain, clamp(0.8 * m.y + 0.4 * m.z, 0.0, 0.9));
    return mat;
}

fn material_lava(tp: TerrainParams, h: f32, m: vec4<f32>) -> Material {
    var mat: Material;
    mat.albedo = refl_basalt();
    mat.emission = spec(0.0);
    // Molten seas (below the datum) and fresh flows glow as blackbodies;
    // a thin chilled skin cools most of the surface.
    let molten = select(0.0, 1.0, tp.liquid == FILL_MAGMA && h < 0.0);
    let glow = max(molten, smoothstep(0.75, 1.0, m.y) * smoothstep(0.6, 1.1, m.z));
    if (glow > 0.0) {
        let t = mix(900.0, 1350.0, max(m.y, 0.5 * molten));
        mat.emission = spec_scale(spec_planck(t), 0.9 * glow);
        mat.albedo = spec_scale(mat.albedo, 1.0 - 0.5 * glow);
    }
    return mat;
}

fn material_rocky(tp: TerrainParams, h: f32, m: vec4<f32>) -> Material {
    var mat: Material;
    mat.emission = spec(0.0);
    mat.regolith = tp.air < 0.5;
    // Highlands (anorthosite-like) against dark mare basalt; fresh ejecta
    // bright; weathering reddens it slowly.
    let high = spec_axpy(spec_ramp(400.0, 780.0), 0.08, spec(0.14 + 0.06 * m.z));
    var a = high;
    if (tp.liquid == FILL_BASALT && h < 0.02) {
        a = refl_basalt();
    }
    a = spec_mix(a, spec(0.32), 0.7 * m.y);
    if (tp.liquid == FILL_WATER && h < 0.0) {
        mat.water = true;
        mat.depth_m = -h * 1000.0;
    }
    mat.albedo = a;
    return mat;
}

fn planet_material(tp: TerrainParams, hit: SurfaceHit, baked: bool) -> Material {
    switch (tp.kind) {
        case KIND_OCEAN: { return material_ocean_world(tp, hit.body, hit.height, hit.terrain, baked); }
        case KIND_DESERT: { return material_desert(hit.body, hit.terrain); }
        case KIND_ICE: { return material_ice(hit.terrain); }
        case KIND_LAVA: { return material_lava(tp, hit.height, hit.terrain); }
        default: { return material_rocky(tp, hit.height, hit.terrain); }
    }
}

// --- Giants -------------------------------------------------------------------

// The cloud deck of a giant: zones and belts from the zonal jets, sheared
// and eddying at their edges, with a few long-lived anticyclonic ovals.
// Returns the spectral albedo at body-fixed direction q.
fn giant_albedo(p: Planet, q: vec3<f32>, lod_rad: f32) -> Spectrum {
    let seed = p.ids.y;
    let lat = asin(clamp(q.z, -1.0, 1.0));
    let lon = atan2(q.y, q.x);
    let oct = clamp(log2(0.08 / max(lod_rad, 1e-5)), 1.0, 6.0);
    // Turbulence stretched along latitude by the shear.
    let s = vec3<f32>(q.x, q.y, q.z * 5.0);
    let turb = tn_fbm(s * 3.0 + tn_warp(s * 1.5, seed + 3u, 2.0) * 0.8, seed + 4u, oct, 0.55);
    var ovals = 0.0;
    var oval_col = 0.0;
    for (var i = 0u; i < 4u; i++) {
        let r = tn_rand3(seed * 31u + i);
        let olat = (r.x - 0.5) * 1.2;
        let olon = r.y * TAU;
        let size = mix(0.04, 0.12, r.z * r.z);
        var dl = lon - olon;
        dl = dl - TAU * round(dl / TAU);
        let e = (dl * cos(olat) / (1.8 * size)) * (dl * cos(olat) / (1.8 * size)) + ((lat - olat) / size) * ((lat - olat) / size);
        let w = exp(-e * 1.5);
        ovals = max(ovals, w);
        oval_col = select(oval_col, r.z, w >= ovals);
    }
    let jets = 7.0 + f32(seed % 5u);
    let band = sin(lat * jets + 0.6 * turb + 0.4 * sin(lat * 3.1 + f32(seed % 7u)));
    let zone = smoothstep(-0.25, 0.35, band);
    if (p.ids.x == KIND_ICE_GIANT) {
        // Muted, nearly uniform decks with faint bands and bright methane
        // cirrus near the dark spots.
        let a = spec(0.52 + 0.05 * zone + 0.08 * smoothstep(0.55, 0.9, turb));
        return spec_mix(a, spec(0.35), 0.6 * ovals);
    }
    // Ammonia-ice zones (white-cream) and belts coloured by chromophores
    // (tholin-like browns absorbing the blue).
    let zone_col = spec_axpy(spec_ramp(400.0, 600.0), 0.12, spec(0.58));
    let belt_col = spec_axpy(spec_ramp(430.0, 680.0), 0.3, spec(0.16));
    var a = spec_mix(belt_col, zone_col, zone);
    a = spec_mix(a, spec_scale(belt_col, 0.8), 0.25 * smoothstep(0.2, 0.9, turb) * (1.0 - zone));
    // Red ovals (the Great Red Spot's colour) or white ones.
    let red = spec_axpy(spec_ramp(520.0, 660.0), 0.35, spec(0.18));
    a = spec_mix(a, spec_mix(zone_col, red, step(0.5, oval_col)), 0.8 * ovals);
    // Hazy, darker polar regions.
    return spec_scale(a, 1.0 - 0.45 * smoothstep(0.9, 1.35, abs(lat)));
}

// --- Radiance ---------------------------------------------------------------

// Sunlight lost to a ring's shadow on the way to x.
fn ring_shadow(p: Planet, x: vec3<f32>, sun: SunLight) -> f32 {
    if (p.rings.z <= 0.0) {
        return 1.0;
    }
    let nrm = p.spin.xyz;
    let ds = dot(sun.dir, nrm);
    if (abs(ds) < 1e-4) {
        return 1.0;
    }
    let t = -dot(x, nrm) / ds;
    if (t <= 0.0) {
        return 1.0;
    }
    let r = length(x + t * sun.dir);
    return exp(-ring_depth(p, r) / abs(ds));
}

// Radiance leaving the surface towards `view` (unit, towards the eye).
fn planet_surface_radiance(p: Planet, h: SurfaceHit, view: vec3<f32>, sun: SunLight) -> Spectrum {
    let up = normalize(h.pos);
    let t_sun = spec_scale(atmo_sun_transmittance(p, h.pos, sun), ring_shadow(p, h.pos, sun) * (1.0 - h.shadow));
    let e_sun = spec_mul(sun.irradiance, t_sun);
    let e_sky = spec_scale(atmo_sky_irradiance(p, h.pos, h.normal, sun), 1.0 - h.sky_occlusion);
    if (planet_is_giant(p)) {
        let a = giant_albedo(p, h.body, h.lod / p.centre.w);
        // Minnaert-darkened limb of a deep, forward-scattering cloud deck.
        let mu0 = max(dot(up, sun.dir), 0.0);
        let mu = max(dot(up, view), 1e-3);
        let k = 0.8;
        let minnaert = pow(max(mu0 * mu, 1e-4), k - 1.0) * mu0;
        return spec_mul(a, spec_add(spec_scale(e_sun, minnaert / PI), spec_scale(e_sky, 1.0 / PI)));
    }
    let tp = planet_terrain(p);
    let mat = planet_material(tp, h, p.detail.y > 0.5);
    if (mat.water) {
        let mu0 = max(dot(up, sun.dir), 0.0);
        let f_sun = fresnel_water(mu0);
        // Light entering the sea, the water-leaving radiance it becomes, and
        // the sky and sun mirrored by the waves.
        let e_in = spec_add(spec_scale(e_sun, mu0 * (1.0 - f_sun)), spec_scale(e_sky, 0.94));
        let below = spec_scale(spec_mul(water_reflectance(mat.depth_m), e_in), 0.54 / PI);
        let wind = p.detail.x;
        let glint = spec_scale(e_sun, ocean_glint(up, view, sun.dir, wind));
        let sky = spec_scale(e_sky, fresnel_water(max(dot(up, view), 0.0)) / PI);
        return spec_add(spec_add(below, glint), sky);
    }
    let f = select(brdf_oren_nayar(h.normal, view, sun.dir), brdf_regolith(h.normal, view, sun.dir), mat.regolith);
    // The shading normal can face the sun where the true sphere doesn't:
    // no sunlight below the geometric horizon.
    let nl = max(dot(h.normal, sun.dir), 0.0) * smoothstep(-0.02, 0.02, dot(up, sun.dir));
    let direct = spec_scale(e_sun, f * nl);
    let ambient = spec_scale(e_sky, 1.0 / PI);
    return spec_fma(mat.albedo, spec_add(direct, ambient), mat.emission);
}

// --- Rings --------------------------------------------------------------------

// Normal optical depth at radius r: ringlets and gaps from a 1D noise in
// radius, a Cassini-like division and a fading inner edge.
fn ring_depth(p: Planet, r: f32) -> f32 {
    let inner = p.rings.x;
    let outer = p.rings.y;
    if (r < inner || r > outer) {
        return 0.0;
    }
    let x = (r - inner) / (outer - inner);
    let seed = p.ids.y;
    var tau = p.rings.z;
    // Denser middle, thin inner "C ring", tenuous outer edge.
    tau *= smoothstep(0.0, 0.25, x) * 0.7 + 0.3;
    tau *= 1.0 - 0.6 * smoothstep(0.85, 1.0, x);
    // Ringlets at several radial scales.
    let n = tn_fbm(vec3<f32>(x * 40.0, f32(seed % 97u), 0.5), seed + 71u, 5.0, 0.6);
    tau *= clamp(0.75 + 1.8 * n, 0.05, 2.0);
    // A broad gap (at a resonance) and a narrow one.
    let gap = tn_rand3(seed + 5u);
    let g1 = mix(0.55, 0.72, gap.x);
    tau *= 1.0 - 0.95 * exp(-pow((x - g1) / 0.018, 2.0));
    tau *= 1.0 - 0.8 * exp(-pow((x - mix(0.82, 0.93, gap.y)) / 0.004, 2.0));
    return tau;
}

struct RingHit {
    hit: bool,
    u: f32,
    L: Spectrum,
    T: Spectrum,
}

// Crossing of the ring plane before `u_max`: a thin slab of particles in
// single scattering (lit face or seen through), in the planet's shadow
// where the planet hides the sun.
fn planet_rings(p: Planet, o: vec3<f32>, dir: vec3<f32>, u_max: f32, sun: SunLight, scale: f32) -> RingHit {
    var r: RingHit;
    r.hit = false;
    if (p.rings.z <= 0.0) {
        return r;
    }
    let nrm = p.spin.xyz;
    let dn = dot(dir, nrm);
    if (abs(dn) < 1e-6) {
        return r;
    }
    let u = -dot(o, nrm) / dn;
    if (u < 0.0 || u > u_max) {
        return r;
    }
    let x = o + u * dir;
    let tau = ring_depth(p, length(x));
    if (tau <= 1e-4) {
        return r;
    }
    let mu = max(abs(dn), 1e-3);
    let mu0 = max(abs(dot(sun.dir, nrm)), 1e-3);
    let lit_side = dot(sun.dir, nrm) * dn < 0.0;
    // Particles: water ice (bright, backscattering like icy moons) darkened
    // and reddened by dust, which scatters forwards.
    let dust = p.rings.w;
    let ice = spec_axpy(spec_ramp(400.0, 650.0), 0.15, spec(0.55));
    let dusty = spec_axpy(spec_ramp(420.0, 700.0), 0.12, spec(0.05));
    let albedo = spec_mix(ice, dusty, dust);
    let c = dot(-dir, sun.dir);
    let phase = mix(atmo_phase_hg(-0.35, -c), atmo_phase_hg(0.6, -c), dust) * 4.0 * PI;
    var s = 0.0;
    if (lit_side) {
        s = mu0 / (4.0 * (mu + mu0)) * (1.0 - exp(-tau * (1.0 / mu + 1.0 / mu0)));
    } else if (abs(mu - mu0) > 1e-3) {
        s = mu0 / (4.0 * (mu - mu0)) * (exp(-tau / mu) - exp(-tau / mu0));
    } else {
        s = tau / (4.0 * mu) * exp(-tau / mu);
    }
    // The planet's shadow on the rings.
    let shadow = sphere_span(x, sun.dir, p.centre.w);
    let lit = select(1.0, 0.0, shadow.x <= shadow.y && shadow.y > 0.0);
    r.hit = true;
    r.u = u;
    r.T = spec(exp(-tau / mu));
    r.L = spec_scale(spec_mul(albedo, sun.irradiance), max(s, 0.0) * phase * lit / PI);
    return r;
}
