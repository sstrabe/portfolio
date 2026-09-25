// ---------------------------------------------------------------------------
// Without ray-tracing hardware there are no terrain tiles in the trace
// (the raster fallback is still to come): planets keep their datum-sphere
// surfaces (`planet_surface_hit`).
// ---------------------------------------------------------------------------

fn terrain_on(p: Planet) -> bool {
    return false;
}

fn terrain_shadow(p: Planet, h: SurfaceHit, sun: SunLight) -> f32 {
    return 0.0;
}

fn terrain_sky_occlusion(p: Planet, h: SurfaceHit) -> f32 {
    return 0.0;
}

fn terrain_reflection(p: Planet, h: SurfaceHit, view: vec3<f32>, sun: SunLight, wind: f32) -> TerrainReflection {
    var out: TerrainReflection;
    out.hit = false;
    return out;
}

fn terrain_surface_hit(p: Planet, o: vec3<f32>, dir: vec3<f32>, u_max: f32, fp: f32, scale: f32) -> SurfaceHit {
    var h: SurfaceHit;
    h.hit = false;
    return h;
}
