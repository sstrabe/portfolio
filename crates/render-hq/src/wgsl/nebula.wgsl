// ---------------------------------------------------------------------------
// Nebulae and dust (hooks called by the far-field trace in `far.wgsl`).
//
// Positions are relative to the hole in units of M (Kerr–Schild
// coordinates, which are nearly Cartesian this far out). `g` is the photon's
// frequency relative to a static observer at infinity: a source at rest in
// these coordinates is seen shifted by g.
//
// Placeholder: empty space.
// ---------------------------------------------------------------------------

// Along the straight chord a → b (one step of the Kerr trace, going back in
// time from the observer).
fn nebula_segment(a: vec3<f32>, b: vec3<f32>, g: f32) -> Medium {
    return medium_clear();
}

// From `p` along `dir` out to infinity, after the ray has escaped the hole.
fn nebula_escape(p: vec3<f32>, dir: vec3<f32>, g: f32) -> Medium {
    return medium_clear();
}
