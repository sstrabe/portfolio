// ---------------------------------------------------------------------------
// Atmospheres and clouds (hooks called by `near.wgsl` and `planet.wgsl`).
//
// Positions are planet-centred and inertial (km). Parameters for the planet
// in slot `p.ids.w` are `atmospheres[p.ids.w]` (see `atmosphere.rs`).
//
// Placeholder: vacuum.
// ---------------------------------------------------------------------------

// Mirrors `AtmosphereGpu` in `atmosphere.rs`; densities relative to Earth.
struct AtmosphereParams {
    shape: vec4<f32>,     // planet radius (km), top altitude (km), Rayleigh scale height (km), Mie scale height (km)
    density: vec4<f32>,   // Rayleigh, Mie, ozone, methane
    aerosol: vec4<f32>,   // Mie g, Mie absorption fraction, dust colour, present (0/1)
    clouds: vec4<f32>,    // coverage, base (km), thickness (km), optical depth
}

@group(2) @binding(0) var<storage, read> atmospheres: array<AtmosphereParams>;

// Light scattered towards the eye along o + u dir for u in [entry, u_max]
// (the part inside the atmosphere), and the transmittance to what lies at
// u_max (the surface when `hit_surface`, else whatever is behind).
fn atmo_segment(p: Planet, o: vec3<f32>, dir: vec3<f32>, u_max: f32, hit_surface: bool, sun: SunLight, fp: f32) -> Medium {
    return medium_clear();
}

// Transmittance of sunlight reaching planet-centred position x (includes
// the planet's own shadow and cloud shadows).
fn atmo_sun_transmittance(p: Planet, x: vec3<f32>, sun: SunLight) -> Spectrum {
    if (dot(x, sun.dir) < 0.0 && length(x - dot(x, sun.dir) * sun.dir) < p.centre.w) {
        return spec(0.0);
    }
    return spec(1.0);
}

// Irradiance on a surface at x with normal `nrm` from the sky (skylight,
// excluding the direct sun), W m⁻² nm⁻¹.
fn atmo_sky_irradiance(p: Planet, x: vec3<f32>, nrm: vec3<f32>, sun: SunLight) -> Spectrum {
    return spec(0.0);
}
