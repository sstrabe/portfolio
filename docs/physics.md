# Physics notes

Units are geometrized (`G = c = 1`), with lengths and times in units of the
hole's mass `M`. For Sagittarius A\* (4.3 × 10⁶ M☉), one `M` of time is
`GM/c³ ≈ 21 s` and one `M` of length is about 6.4 × 10⁶ km. The hole spins
with `a = 0.94 M`.

This page separates what is exact (up to numerical integration error) from
what is modelled.

## Spacetime and coordinates — exact

Kerr in Cartesian Kerr–Schild form, `g = η + f l⊗l`
([`metric.rs`](../crates/kerr/src/metric.rs)). The coordinates are regular
across the future horizon, and every `t = const` slice is spacelike. That
makes `t` a valid global clock for the whole cluster, including bodies that
are falling in. The analytic gradients of `r`, `f` and `l` are
cross-checked against forward-mode dual numbers. The GPU shaders use the same
formulas.

## Bodies — exact geodesics plus perturbations

Every star, compact object and station is integrated in Hamiltonian form,
`H = ½ g^{μν} p_μ p_ν`, with coordinate time as the independent variable. The
integrator is adaptive Dormand–Prince 5(4) with a relative tolerance of 10⁻⁹
([`geodesic.rs`](../crates/kerr/src/geodesic.rs),
[`integrate.rs`](../crates/kerr/src/integrate.rs)). Without perturbations,
`E`, `L_z`, the Carter constant and the mass shell are conserved to about
10⁻⁹ over thousands of `M` (tested).

Perturbations enter as 4-forces projected orthogonal to the 4-velocity, so
`u·u = −1` holds exactly and the geodesic part is untouched:

* **Mutual pulls (modelled).** Each body feels a Plummer-softened,
  Newtonian-strength pull from the heaviest bodies. Each source is evaluated
  at its **retarded time**, `t − t_ret = |x − x_src(t_ret)|`, and then
  **extrapolated forward with its retarded velocity**. For uniform motion this
  gives the present position, as for the Liénard–Wiechert field of a
  uniformly moving charge; linearized gravity behaves the same way. So
  influence travels at `c` without the spurious aberration of naive
  retardation. Limitations:
  * the light-travel delay uses flat-space distance in Kerr–Schild
    coordinates (no Shapiro delay in the retardation);
  * the pulls are refreshed once per history tick (2.5 `M`) and held
    constant in between;
  * only the 64 heaviest bodies act as sources;
  * star masses are exaggerated (up to 2 × 10⁻⁴ `M`) so the effect is
    visible.
* **Radiation reaction (modelled).** Compact objects feel the 2.5PN
  Iyer–Will acceleration (harmonic gauge), evaluated with Kerr–Schild
  position and coordinate velocity. It reproduces the Peters–Mathews flux for
  circular orbits (tested) and makes tight orbits shrink and eventually
  plunge. Near the hole it is a qualitative "kludge" in the spirit of
  numerical-kludge EMRI waveforms, not a self-force calculation. Masses of
  0.4–1.5 % of `M` make the decay visible within minutes.

Stations feel neither perturbation, so they stay on their orbits. Their
orbits are still real geodesics, and inclined ones precess through frame
dragging.

When a star crosses the horizon it stops being integrated. Its slot is only
reused after its last light has left the stored history window, so its image
keeps freezing and reddening near the shadow for as long as that light is
still arriving.

## The pilot — exact

The ship follows an accelerated timelike worldline
([`pilot.rs`](../crates/kerr/src/pilot.rs)):

```
du^μ/dτ   = −Γ^μ_{αβ} u^α u^β + A^i e_i^μ
de_i^μ/dτ = −Γ^μ_{αβ} u^α e_i^β + A^i u^μ     (Fermi–Walker)
```

Thrust is a proper acceleration in the ship frame, so the speed approaches
`c` but never reaches it. The spatial axes are Fermi–Walker transported, like
gyroscopes, plus the pilot's commanded rotation. They are re-orthonormalized
against the metric after every step.

**The clock.** Each frame turns wall-clock time into pilot proper time `Δτ`.
The pilot is integrated over `Δτ`, and the cluster is then advanced to the
pilot's new coordinate time. High `γ`, or hovering deep in the well, makes
`dt/dτ` large and the cluster races ahead. If `Δt` would exceed the per-frame
compute budget, `Δτ` shrinks instead: the "clock limiter" shown in the HUD.
The physics is never coarsened; your own clock just runs slower than the wall
clock. A freely falling pilot's `dt/dτ` stays finite at the horizon in
Kerr–Schild time (about 1.5 for a drop from rest). The big speed-ups come
from high speed or from hovering.

## Seeing — exact rays, modelled appearance

A pixel looking in ship-frame direction `n` traces the past-directed null
geodesic with `P = −u + n^a e_a`. Its observed frequency is normalized to 1,
so anything the ray meets with 4-velocity `w` has `g = ν_obs/ν_emit = 1/(P·w)`
([`lensing.rs`](../crates/kerr/src/lensing.rs),
[`common.wgsl`](../crates/shaders/src/wgsl/common.wgsl)).

* **Sky.** Rays run until they are captured (`r − r₊ < 0.03`) or escape past
  `max(120, 2.5 r_obs)`. The small bending left beyond that radius is
  neglected. The galaxy is procedural, seen with colour temperature `g·T` and
  bolometric intensity `∝ g⁴`. Aberration and beaming come from the tetrad.
* **Bodies.** An image is a ray that meets the body's worldline at the ray's
  own coordinate time, using the Hermite-interpolated history. That includes
  light-travel time, Shapiro delay and the body's motion. Images are found by
  Newton iteration on the closest-approach miss vector. The angle the ray
  sweeps around the hole selects the order: 0 for the direct image, 1 for the
  image from around the far side. Flux is `L g⁴ / |J₁ × J₂|`, where `J` is the
  Jacobian of the ray bundle at the source. This matches the point-lens
  magnification (tested). Colour is a blackbody at `g·T` (Planck's law at
  three wavelengths, white-balanced to 6500 K).
* **Station cards.** Cards are flat panels at the station's emission event,
  moving with its velocity near that event, and always turned to face the
  pilot. The same per-pixel rays hit them, so they are lensed and Doppler
  shifted. The linear motion and the billboard orientation are modelling
  choices.
* The GPU uses `f32` RK4 with steps proportional to `r`. The CPU uses `f64`,
  and the station screen positions handed to the page come from the CPU
  solver.

## Realistic mode (desktop)

`WorldConfig::sgr_a` uses the physical star model:

- **Masses:** Salpeter, 0.5–40 M☉.
- **Main sequence:** R ∝ M^0.8 below 1 M☉ and M^0.57 above. L follows a
  piecewise power law in mass (index 2.3, 4 or 3.5 depending on the range).
- **Red giants:** 8% of stars, 10–60 R☉ at 3600–4800 K.
- **Temperature:** from L and R.

Weak-field pulls use the real masses, so they are genuinely tiny. They are
refreshed every 16 history ticks (~3 × 10⁴ M), far shorter than any orbital
timescale there.

- **Stars as points.** Point flux is `L g⁴ / area` in L☉/M². It is compared
  with the flux of a 7th-magnitude star, then raised by an eye-like
  adaptation: the fifth-brightest star, Doppler boost included, sits 10⁴
  times above the visibility limit.
- **Stars as discs.** A star is drawn as a disc once its angular radius
  exceeds a third of a pixel. The disc sits on the flat-space past light cone
  and moves with the star, has limb darkening (u = 0.6), and has the same
  total flux as the point it replaces.
