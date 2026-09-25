//! Lensing by the cluster's stellar-mass black holes: bind group 5 of the
//! trace pass. `wgsl/lens.wgsl` runs the model below per ray step; the
//! functions here are its f64 mirror, used by the tests.
//!
//! # Model
//!
//! Each hole is a Schwarzschild lens of mass `m` on its orbit. Everything
//! happens in the hole's rest frame: an orthonormal tetrad at the hole's
//! event (the Kerr–Schild metric of Sagittarius A* there, boosted by the
//! hole's 4-velocity), so aberration by the moving lens and the photon's
//! energy change on deflection come out of the frame change exactly. Over
//! the region that matters (well under an AU) Sgr A*'s field is uniform to
//! ~10⁻⁸, so the frame is flat apart from the hole.
//!
//! The main Kerr integrator steps ~5 % of the distance to Sgr A*, tens of
//! AU near the cluster, while a 10 M☉ hole's Schwarzschild radius is 30 km.
//! So after each step the chord is checked against every nearby hole, and
//! the hole's effect on it is added analytically:
//!
//! * **Weak field (Born).** Along a straight line with impact parameter `b`
//!   the linearised field turns the ray towards the hole by
//!   `dα = (2m/b) d(sin φ)`, `sin φ = z/√(z² + b²)`, `z` the distance past
//!   closest approach; the total is Einstein's `4m/b`. The part picked up
//!   along the chord is applied as a rotation of the rest-frame momentum
//!   (null condition and rest-frame energy kept) plus the matching sideways
//!   shift of the chord's end, so the result doesn't depend on where the
//!   steps fall.
//! * **Reach.** The field is only felt within `reach = 4m/(ε θ_px)` of the
//!   hole, where it deflects by less than `ε = 0.1` pixel; the cut is
//!   continuous (the deflection falls to zero at `b = reach`).
//! * **Bubble (strong field).** Inside `r_b = m √(15π/(4 ε θ_px))`,
//!   between 50 and 2000 m, the Born angle's first correction
//!   `(15π/4)(m/b)²` could exceed ε pixels, so rays that enter are
//!   integrated exactly: the Schwarzschild orbit equation
//!   `d²u/dφ² = 3m u² − u` (`u = 1/r`) in the orbit's plane, entering and
//!   leaving in isotropic coordinates, whose conformally flat space makes
//!   coordinate angles the angles static observers measure. Rays that fall
//!   below `r = 2m` (or orbit the photon sphere too long) are captured:
//!   the shadow, `b < 3√3 m`. Outside the bubble the Born kick carries on
//!   along the exit ray, so the two regimes join without a seam.
//! * **Observer.** A pilot inside a hole's field is taken as static there:
//!   pixel directions are rest-frame angles (conformal flatness again), and
//!   the photon's frequency is blueshifted by the lapse.
//!
//! Holes whose reach subtends less than half a pixel are skipped: the
//! little they do stays inside one pixel. Each hole's position is taken on
//! the pilot's past light cone (flat estimate) and moved along its velocity
//! to the ray's own time, like the star discs.
//!
//! Star point images (the `images` pass) are found by the Sgr A* solver and
//! are not lensed by these holes; the galaxy, nebulae and star discs are.

use crate::FrameContext;
use bytemuck::{Pod, Zeroable};
use kerr::cluster::BodyKind;
use kerr::pilot::orthonormalize;
use kerr::vec3::{self, V3, V4};
use kerr::world::World;
use render::frame::BodyMeta;

pub const MAX_LENSES: usize = 8;
/// Largest deflection error allowed, in pixels.
pub const TOLERANCE_PX: f64 = 0.1;
/// Bubble radius limits, units of the hole's mass.
pub const BUBBLE_MIN: f64 = 50.0;
pub const BUBBLE_MAX: f64 = 2000.0;
/// Orbit-equation step: at most this in φ, or this fraction of `u`.
pub const BUBBLE_STEP: f64 = 0.06;
pub const BUBBLE_MAX_STEPS: usize = 600;

/// Mirrors `struct Lens` in `lens.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct LensGpu {
    /// The hole's event relative to the observer: (t, x, y, z).
    pub event: [f32; 4],
    /// Mass, then coordinate velocity dx/dt.
    pub body: [f32; 4],
    /// Bubble radius, reach (rest frame, units of M), unused, unused.
    pub radii: [f32; 4],
    /// Rest-frame tetrad `e_a^μ` (columns u, x, y, z).
    pub e: [[f32; 4]; 4],
    /// Dual basis `θ^a_μ` (columns), `θ^a(e_b) = δ^a_b`.
    pub w: [[f32; 4]; 4],
}

/// Mirrors `struct Lenses` in `lens.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct LensesGpu {
    pub count: [u32; 4],
    /// Product of the holes' lapses at the observer, unused ×3.
    pub obs: [f32; 4],
    pub items: [LensGpu; MAX_LENSES],
}

/// Bubble radius and reach of a hole of mass `m` for a pixel of
/// `pixel_angle` radians.
pub fn radii(m: f64, pixel_angle: f64) -> (f64, f64) {
    let eps = TOLERANCE_PX * pixel_angle;
    let bubble = m * (15.0 * std::f64::consts::PI / (4.0 * eps)).sqrt().clamp(BUBBLE_MIN, BUBBLE_MAX);
    let reach = (4.0 * m / eps).max(2.0 * bubble);
    (bubble, reach)
}

/// `1 + z/ρ` and `1 − z/ρ`, `ρ = √(z² + b²)`, without cancellation.
fn one_plus(z: f64, b: f64) -> f64 {
    let rho = z.hypot(b);
    if z < 0.0 { b * b / (rho * (rho - z)) } else { 1.0 + z / rho }
}

fn one_minus(z: f64, b: f64) -> f64 {
    let rho = z.hypot(b);
    if z > 0.0 { b * b / (rho * (rho + z)) } else { 1.0 - z / rho }
}

/// `ρ + z` and `ρ − z` without cancellation.
fn rho_plus(z: f64, b: f64) -> f64 {
    let rho = z.hypot(b);
    if z < 0.0 { b * b / (rho - z) } else { rho + z }
}

fn rho_minus(z: f64, b: f64) -> f64 {
    let rho = z.hypot(b);
    if z > 0.0 { b * b / (rho + z) } else { rho - z }
}

/// Weak-field deflection picked up along the straight stretch `z1..z2` of a
/// line passing a mass `m` at impact parameter `b`, counting only what lies
/// within `reach` of it: the turn towards the mass (rad) and the sideways
/// shift at `z2` (towards the mass) relative to the undeflected line.
pub fn born(m: f64, b: f64, z1: f64, z2: f64, reach: f64) -> (f64, f64) {
    let zm = (reach * reach - b * b).max(0.0).sqrt();
    let (c1, c2) = (z1.clamp(-zm, zm), z2.clamp(-zm, zm));
    if c2 <= c1 || b <= 0.0 {
        return (0.0, 0.0);
    }
    let k = 2.0 * m / b;
    // ∫ (sin φ − sin φ₁) dz, written with whichever of 1 ± sin φ is small.
    let (turn, area) = if c1 < 0.0 {
        let f1 = one_plus(c1, b);
        (one_plus(c2, b) - f1, rho_plus(c2, b) - rho_plus(c1, b) - f1 * (c2 - c1))
    } else {
        let g1 = one_minus(c1, b);
        (g1 - one_minus(c2, b), g1 * (c2 - c1) + rho_minus(c2, b) - rho_minus(c1, b))
    };
    (k * turn, k * area + k * turn * (z2 - c2))
}

/// `m/r` (Schwarzschild `r`) at isotropic radius `rho`.
fn u_of(m: f64, rho: f64) -> f64 {
    let q = m / rho;
    q / ((1.0 + 0.5 * q) * (1.0 + 0.5 * q))
}

/// Where a ray leaves a hole's bubble, in the plane of its orbit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Passage {
    Captured,
    /// Leaves at polar angle `phi` from the entry point, moving at angle
    /// `psi` from the outward radial (towards increasing `phi`).
    Exit {
        phi: f64,
        psi: f64,
    },
}

/// A ray starting at isotropic radius `rho0` at angle `psi0` from the
/// outward radial (0 ≤ ψ ≤ π, turning towards increasing φ), followed
/// through the Schwarzschild field of mass `m` until it reaches isotropic
/// radius `rho_exit` moving outwards. Integrates `d²U/dφ² = 3U² − U` for
/// `U = m/r` with RK4 steps of at most `BUBBLE_STEP` in φ and in `ln U`.
pub fn bubble(m: f64, rho0: f64, psi0: f64, rho_exit: f64) -> Passage {
    let (s0, c0) = psi0.sin_cos();
    let mut u = u_of(m, rho0);
    let u_exit = u_of(m, rho_exit);
    if s0 < 1e-6 {
        return if c0 < 0.0 { Passage::Captured } else { Passage::Exit { phi: 0.0, psi: 0.0 } };
    }
    // dU/dφ = −U √(1 − 2U) cot ψ (static-frame angle ψ).
    let mut w = -u * (1.0 - 2.0 * u).max(0.0).sqrt() * c0 / s0;
    // Inbound from outside the photon sphere below the critical impact
    // parameter (1/b² = W² + U²(1 − 2U) > 1/27): captured.
    if w > 0.0 && u < 1.0 / 3.0 && w * w + u * u * (1.0 - 2.0 * u) > 1.0 / 27.0 {
        return Passage::Captured;
    }
    let f = |y: [f64; 2]| [y[1], y[0] * (3.0 * y[0] - 1.0)];
    let mut phi = 0.0;
    for _ in 0..BUBBLE_MAX_STEPS {
        let h = BUBBLE_STEP / (1.0 + w.abs() / u);
        let y = [u, w];
        let k1 = f(y);
        let k2 = f([u + 0.5 * h * k1[0], w + 0.5 * h * k1[1]]);
        let k3 = f([u + 0.5 * h * k2[0], w + 0.5 * h * k2[1]]);
        let k4 = f([u + h * k3[0], w + h * k3[1]]);
        let un = u + h / 6.0 * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]);
        let wn = w + h / 6.0 * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]);
        if un >= 0.5 {
            return Passage::Captured;
        }
        if wn < 0.0 && un <= u_exit {
            let t = ((u - u_exit) / (u - un).max(1e-300)).clamp(0.0, 1.0);
            let we = w + t * (wn - w);
            let tangential = u_exit * (1.0 - 2.0 * u_exit).sqrt();
            return Passage::Exit { phi: phi + t * h, psi: tangential.atan2(-we) };
        }
        (u, w) = (un, wn);
        phi += h;
    }
    Passage::Captured
}

/// Lapse `√(−g_tt)` of a mass `m` at isotropic radius `rho`.
fn lapse(m: f64, rho: f64) -> f64 {
    let q = 0.5 * m / rho.max(0.5 * m);
    ((1.0 - q) / (1.0 + q)).max(1e-3)
}

fn f4(v: V4) -> [f32; 4] {
    v.map(|x| x as f32)
}

/// The holes that can visibly bend light this frame, nearest in effect
/// first, each in its rest frame relative to the observer.
pub fn select(world: &World, pixel_angle: f64) -> LensesGpu {
    let k = &world.kerr;
    let cluster = &world.cluster;
    let obs = world.pilot.position();
    let t_obs = world.pilot.x[0];
    let mut found: Vec<(f64, LensGpu, f64)> = Vec::new();
    for (i, body) in cluster.bodies.iter().enumerate() {
        if !body.alive || body.params.kind != BodyKind::Compact {
            continue;
        }
        let m = body.params.mass;
        let (bubble, reach) = radii(m, pixel_angle);
        let dist = vec3::norm(vec3::sub(body.position(), obs));
        if reach < 0.5 * pixel_angle * (dist - reach) {
            continue;
        }
        // On the past light cone (flat estimate; the ray's own time is
        // used on the GPU).
        let mut t_ret = t_obs - dist;
        let mut sample = cluster.sample_at(i, t_ret);
        for _ in 0..4 {
            let Some(s) = sample else { break };
            t_ret = t_obs - vec3::norm(vec3::sub(s.pos, obs));
            sample = cluster.sample_at(i, t_ret.min(cluster.t));
        }
        let Some(s) = sample else { continue };
        let Some(u) = k.four_velocity(s.pos, s.vel) else { continue };
        let mut e = [u, [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];
        orthonormalize(k, s.pos, &mut e);
        let w: [V4; 4] = std::array::from_fn(|a| {
            let low = k.lower(s.pos, e[a]);
            if a == 0 { low.map(|x| -x) } else { low }
        });
        let rel: V3 = vec3::sub(s.pos, obs);
        let event = [t_ret - t_obs, rel[0], rel[1], rel[2]];
        // Observer's distance in the rest frame (the hole sits at its
        // spatial origin).
        let rho_obs = vec3::norm(std::array::from_fn(|a| -(0..4).map(|mu| w[a + 1][mu] * event[mu]).sum::<f64>()));
        let gpu = LensGpu {
            event: f4(event),
            body: [m as f32, s.vel[0] as f32, s.vel[1] as f32, s.vel[2] as f32],
            radii: [bubble as f32, reach as f32, 0.0, 0.0],
            e: e.map(f4),
            w: w.map(f4),
        };
        found.push((reach / dist.max(1e-30), gpu, lapse(m, rho_obs)));
    }
    found.sort_by(|a, b| b.0.total_cmp(&a.0));
    found.truncate(MAX_LENSES);
    let mut out = LensesGpu { count: [found.len() as u32, 0, 0, 0], obs: [1.0, 0.0, 0.0, 0.0], ..Default::default() };
    for (slot, (_, gpu, lapse)) in found.into_iter().enumerate() {
        out.items[slot] = gpu;
        out.obs[0] *= lapse as f32;
    }
    out
}

/// Give the holes their shadow's radius (`3√3 m`) as body radius, so their
/// point-source glow fades out once the shadow is resolved and the ray
/// tracer draws it (see `splat.wgsl`).
pub fn mark_shadows(world: &World, meta: &mut [BodyMeta]) {
    for (b, meta) in world.cluster.bodies.iter().zip(meta) {
        if b.params.kind == BodyKind::Compact {
            meta.b[3] = (27f64.sqrt() * b.params.mass) as f32;
        }
    }
}

pub struct Lensing {
    layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    buffer: wgpu::Buffer,
}

impl Lensing {
    pub fn new(device: &wgpu::Device) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("lensing"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lenses"),
            size: std::mem::size_of::<LensesGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lensing"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: buffer.as_entire_binding() }],
        });
        Self { layout, bind_group, buffer }
    }

    pub fn layout(&self) -> &wgpu::BindGroupLayout {
        &self.layout
    }

    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    pub fn update(&mut self, ctx: &FrameContext) {
        let lenses = select(ctx.world, (ctx.hq.view[3] as f64).sqrt());
        ctx.queue.write_buffer(&self.buffer, 0, bytemuck::bytes_of(&lenses));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kerr::Kerr;
    use kerr::lensing::{Observer, RayConfig, RayEnd, trace_to_sky};
    use kerr::pilot::Pilot;
    use std::f64::consts::PI;

    #[test]
    fn layout_matches_wgsl() {
        assert_eq!(std::mem::size_of::<LensGpu>(), 176);
        assert_eq!(std::mem::size_of::<LensesGpu>(), 32 + MAX_LENSES * 176);
    }

    /// Exact Schwarzschild deflection from the weak-field series
    /// (Keeton & Petters 2005), valid for b ≫ m.
    fn series(m: f64, b: f64) -> f64 {
        let x = m / b;
        4.0 * x + 15.0 * PI / 4.0 * x * x + 128.0 / 3.0 * x.powi(3) + 3465.0 * PI / 64.0 * x.powi(4)
    }

    /// Total deflection of a ray arriving from far away with impact
    /// parameter `b` (m = 1), by the model: Born outside the bubble, the
    /// orbit equation inside. Splits the approach into `chords` straight
    /// pieces, re-deriving the line after every kick as the shader does.
    fn model(b: f64, bubble_r: f64, reach: f64, chords: usize) -> Option<f64> {
        // 2-D: the lens at the origin, the ray moving along +x at y = b.
        let far = 1.0e7;
        let mut pos = [-far, b];
        let mut dir = [1.0f64, 0.0];
        let chord = 2.0 * far / chords as f64;
        for _ in 0..chords + 1 {
            let end = [pos[0] + chord * dir[0], pos[1] + chord * dir[1]];
            // Closest approach of this chord's line.
            let along = -(pos[0] * dir[0] + pos[1] * dir[1]);
            let bv = [pos[0] + along * dir[0], pos[1] + along * dir[1]];
            let bl = bv[0].hypot(bv[1]);
            let (z1, z2) = (-along, chord - along);
            let z_in = -(bubble_r * bubble_r - bl * bl).max(0.0).sqrt();
            let toward = [-bv[0] / bl, -bv[1] / bl];
            let rotate = |d: [f64; 2], a: f64| {
                let s = if d[0] * toward[1] - d[1] * toward[0] > 0.0 { a } else { -a };
                [d[0] * s.cos() - d[1] * s.sin(), d[0] * s.sin() + d[1] * s.cos()]
            };
            if bl < bubble_r && z1 < 0.0 && z_in < z2 {
                let (turn, shift) = born(1.0, bl, z1, z_in, reach);
                let entry = [bv[0] + z_in * dir[0] + shift * toward[0], bv[1] + z_in * dir[1] + shift * toward[1]];
                let d = rotate(dir, turn);
                let rho0 = entry[0].hypot(entry[1]);
                let radial = [entry[0] / rho0, entry[1] / rho0];
                let psi0 = (radial[0] * d[1] - radial[1] * d[0]).abs().atan2(radial[0] * d[0] + radial[1] * d[1]);
                // Orientation of the orbit (sense of increasing φ).
                let sense = if radial[0] * d[1] - radial[1] * d[0] >= 0.0 { 1.0 } else { -1.0 };
                let Passage::Exit { phi, psi } = bubble(1.0, rho0, psi0, bubble_r) else { return None };
                let a = radial[1].atan2(radial[0]) + sense * phi;
                let out = [a.cos(), a.sin()];
                let dir_angle = a + sense * psi;
                pos = [bubble_r * out[0], bubble_r * out[1]];
                dir = [dir_angle.cos(), dir_angle.sin()];
                continue;
            }
            let (turn, shift) = born(1.0, bl, z1, z2, reach);
            dir = rotate(dir, turn);
            pos = [end[0] + shift * toward[0], end[1] + shift * toward[1]];
            if pos[0] > far {
                break;
            }
        }
        Some(dir[1].atan2(dir[0]).abs())
    }

    #[test]
    fn born_adds_up_to_einstein_angle() {
        let (m, b) = (1.0, 300.0);
        let (turn, _) = born(m, b, -1e9, 1e9, f64::INFINITY);
        assert!((turn - 4.0 * m / b).abs() < 1e-12 * turn, "{turn}");
        // Split anywhere, the pieces sum to the whole.
        let cuts = [-1e9, -5e3, -1.0, 0.0, 17.0, 4e4, 1e9];
        let sum: f64 = cuts.windows(2).map(|w| born(m, b, w[0], w[1], 1e5).0).sum();
        assert!((sum - born(m, b, -1e9, 1e9, 1e5).0).abs() < 1e-15, "{sum}");
        // The shift at the end of a long stretch through closest approach is
        // the thin-lens kink: (turn) × distance past the lens.
        let (turn, shift) = born(m, b, -1e8, 1e8, f64::INFINITY);
        assert!((shift / (turn * 1e8) - 1.0).abs() < 1e-4, "{shift}");
        // The cut at `reach` is continuous.
        assert!(born(m, 999.999, -1e9, 1e9, 1000.0).0 < 1e-5);
    }

    #[test]
    fn orbit_equation_matches_the_exact_deflection() {
        // Enter from far away (isotropic radius R), leave at R.
        for b in [20.0f64, 50.0, 200.0, 1000.0] {
            let r = 1.0e5;
            let psi0 = PI - (b / r).asin();
            let Passage::Exit { phi, psi } = bubble(1.0, r, psi0, r) else { panic!("captured at b = {b}") };
            // Deflection: the direction's angle (polar angle + ψ) at exit
            // minus at entry (ψ₀ at polar angle 0), plus what the weak field
            // adds outside R on either side, (2m/b)(1 − sin φ) ≈ m b/R².
            let turned = phi + psi - psi0 + 2.0 * b / (r * r);
            let err = (turned - series(1.0, b)).abs() / series(1.0, b);
            assert!(err < 2e-3 * (20.0 / b).powi(3) + 2e-5, "b = {b}: {turned} vs {}", series(1.0, b));
        }
    }

    #[test]
    fn shadow_edge_is_the_critical_impact_parameter() {
        let bc = 27f64.sqrt();
        let r = 1.0e4;
        let at = |b: f64| bubble(1.0, r, PI - (b / r).asin(), r);
        assert_eq!(at(bc * 0.999), Passage::Captured);
        assert!(matches!(at(bc * 1.01), Passage::Exit { .. }));
    }

    #[test]
    fn model_is_seamless_across_the_bubble_edge() {
        let (rb, reach) = (300.0, f64::INFINITY);
        let mut last = None;
        for i in 0..40 {
            let b = rb * (0.8 + 0.01 * i as f64);
            let d = model(b, rb, reach, 1).unwrap();
            let exact = series(1.0, b);
            // Born beyond the bubble misses (15π/4)(m/b)²; nothing else.
            assert!((d - exact).abs() < 1.2 * 15.0 * PI / 4.0 / (b * b), "b = {b}: {d} vs {exact}");
            if let Some(prev) = last {
                // Deflection falls smoothly with b: no jump at the edge.
                let step: f64 = prev - d;
                assert!(step > 0.0 && step < 2.0 * 4.0 / (b * b) * 0.01 * rb, "b = {b}: step {step}");
            }
            last = Some(d);
        }
    }

    #[test]
    fn model_does_not_depend_on_the_steps() {
        for b in [30.0, 250.0, 2000.0] {
            let one = model(b, 300.0, 1e5, 1).unwrap();
            for chords in [3, 20, 1001] {
                let many = model(b, 300.0, 1e5, chords).unwrap();
                assert!((many - one).abs() < 1e-3 * one, "b = {b}, {chords} chords: {many} vs {one}");
            }
        }
    }

    /// Independent check against the Kerr crate's geodesic tracer (a = 0),
    /// in the strong field.
    #[test]
    fn model_matches_a_schwarzschild_geodesic() {
        let k = Kerr::new(1.0, 0.0);
        let dist = 1.0e5;
        let pilot = Pilot::new(&k, [-dist, 0.0, 0.0], [0.0; 3], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]).unwrap();
        let obs = Observer { x: pilot.x, e: pilot.e };
        let cfg = RayConfig { escape_radius: 2.0e6, max_steps: 400_000, step: 0.01, ..Default::default() };
        for b in [6.0, 10.0, 40.0] {
            let sin = b * (1.0 - 2.0 / dist).sqrt() / dist;
            let n = [(1.0 - sin * sin).sqrt(), sin, 0.0];
            let RayEnd::Sky { dir, .. } = trace_to_sky(&k, &obs, n, &cfg) else { panic!("b = {b}") };
            // The pilot's left is +y or −y; the deflection is about the
            // direction at the pilot, which is along ±y.
            let side = pilot.e[2][2].signum();
            let geodesic = (-side * dir[1]).atan2(dir[0]) + sin.asin();
            let modelled = model(b, 300.0, 1e6, 1).unwrap();
            // Unbound turns > π wrap; compare modulo 2π.
            let diff = (geodesic - modelled + PI).rem_euclid(2.0 * PI) - PI;
            assert!(diff.abs() < 2e-3, "b = {b}: geodesic {geodesic}, model {modelled}");
        }
    }

    #[test]
    fn radii_scale_with_the_pixel() {
        let m = 10.0 * kerr::units::MSUN;
        let (rb, reach) = radii(m, 1.4e-3);
        assert!(rb > 250.0 * m && rb < 350.0 * m, "{}", rb / m);
        assert!((reach / m - 4.0 / (0.1 * 1.4e-3)).abs() < 1.0);
    }
}
