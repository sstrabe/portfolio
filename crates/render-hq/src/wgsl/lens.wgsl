// ---------------------------------------------------------------------------
// Lensing by stellar-mass black holes (hook called on every step of the
// far-field Kerr trace in `far.wgsl`).
//
// `a` is the ray state before a step and `b` after it; the hook may bend
// `b.p` (and move `b.x`) for compact objects near the chord. Positions are
// relative to the observer (see `abs_pos`).
//
// Placeholder: only Sagittarius A* bends light.
// ---------------------------------------------------------------------------

fn lens_kick(a: Phase, b: Phase) -> Phase {
    return b;
}
