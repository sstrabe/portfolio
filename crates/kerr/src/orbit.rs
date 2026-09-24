//! Initial conditions for orbits.

use crate::metric::Kerr;
use crate::rng::Rng;
use crate::vec3::{self, V3};

/// Position and coordinate velocity `dx/dt` in Kerr–Schild coordinates.
#[derive(Clone, Copy, Debug)]
pub struct Launch {
    pub pos: V3,
    pub vel: V3,
}

/// Angular frequency `dφ/dt` of a circular equatorial geodesic at
/// Boyer–Lindquist radius `r` (the same `r` as Kerr–Schild).
pub fn circular_omega(k: &Kerr, r: f64, prograde: bool) -> f64 {
    let sm = k.m.sqrt();
    if prograde { sm / (r.powf(1.5) + k.a * sm) } else { -sm / (r.powf(1.5) - k.a * sm) }
}

/// Innermost stable circular orbit radius.
pub fn isco(k: &Kerr, prograde: bool) -> f64 {
    let a = k.a / k.m;
    let z1 = 1.0 + (1.0 - a * a).cbrt() * ((1.0 + a).cbrt() + (1.0 - a).cbrt());
    let z2 = (3.0 * a * a + z1 * z1).sqrt();
    let s = ((3.0 - z1) * (3.0 + z1 + 2.0 * z2)).sqrt();
    k.m * if prograde { 3.0 + z2 - s } else { 3.0 + z2 + s }
}

/// Exact circular equatorial orbit. In Kerr–Schild Cartesian coordinates
/// these are rigid rotations about `z`: `x + i y = (r + i a) e^{iφ}`,
/// `φ = φ₀ + Ω t`.
pub fn circular_equatorial(k: &Kerr, r: f64, phase: f64, prograde: bool) -> Launch {
    let om = circular_omega(k, r, prograde);
    let (s, c) = phase.sin_cos();
    let pos = [r * c - k.a * s, r * s + k.a * c, 0.0];
    Launch { pos, vel: [-om * pos[1], om * pos[0], 0.0] }
}

/// A near-circular orbit of radius `r` in a plane with the given
/// inclination to the equator and longitude of ascending node. Exact for
/// `a = 0`; for spinning holes the orbit is mildly eccentric and precesses
/// (Lense–Thirring), which is the point.
pub fn inclined_circular(k: &Kerr, r: f64, inclination: f64, node: f64, phase: f64) -> Launch {
    let node_dir = [node.cos(), node.sin(), 0.0];
    let normal = vec3::rotate([0.0, 0.0, 1.0], node_dir, inclination);
    let in_plane = vec3::cross(normal, node_dir);
    let radial = vec3::add(vec3::scale(node_dir, phase.cos()), vec3::scale(in_plane, phase.sin()));
    let tangent = vec3::cross(normal, radial);
    // Frame dragging felt by the orbit scales with the projected spin.
    let eff = Kerr::new(k.m, k.a * inclination.cos().abs());
    let prograde = inclination.cos() >= 0.0;
    let speed = r * circular_omega(&eff, r, prograde).abs();
    Launch { pos: vec3::scale(radial, r), vel: vec3::scale(tangent, speed) }
}

/// Random bound orbit for a cluster star. Radii are drawn from a density
/// profile `n(r) ∝ r^{-γ}` (a Bahcall–Wolf-like cusp for `γ ≈ 1.75`), with
/// isotropic orientation and a spread of speeds around circular. Orbits whose
/// Newtonian periapsis would be closer than `min_periapsis` are redrawn.
pub fn random_cluster_orbit(k: &Kerr, rng: &mut Rng, r_min: f64, r_max: f64, gamma: f64, min_periapsis: f64) -> Launch {
    loop {
        // Inverse CDF of r² n(r) ∝ r^{2-γ} on [r_min, r_max].
        let e = 3.0 - gamma;
        let u = rng.uniform();
        let r = (r_min.powf(e) + u * (r_max.powf(e) - r_min.powf(e))).powf(1.0 / e);
        let radial = rng.unit_vector();
        let tangent = vec3::any_orthogonal(radial);
        let tangent = vec3::rotate(tangent, radial, rng.range(0.0, std::f64::consts::TAU));
        let v_circ = (k.m / r).sqrt();
        let speed = v_circ * rng.range(0.55, 1.25);
        let tilt = rng.normal() * 0.35;
        let dir = vec3::normalize(vec3::add(vec3::scale(tangent, tilt.cos()), vec3::scale(radial, tilt.sin())));
        let vel = vec3::scale(dir, speed);
        // Newtonian orbit elements for the periapsis check.
        let h = vec3::norm(vec3::cross(vec3::scale(radial, r), vel));
        let energy = 0.5 * speed * speed - k.m / r;
        if energy >= 0.0 {
            continue;
        }
        let a_semi = -k.m / (2.0 * energy);
        let ecc = (1.0 - h * h / (k.m * a_semi)).max(0.0).sqrt();
        let peri = a_semi * (1.0 - ecc);
        let apo = a_semi * (1.0 + ecc);
        if peri < min_periapsis || apo > 1.6 * r_max {
            continue;
        }
        return Launch { pos: vec3::scale(radial, r), vel };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geodesic;
    use crate::integrate::{Dopri5, Outcome, Tolerance};

    fn stepper() -> Dopri5 {
        Dopri5 { tol: Tolerance { abs: 1e-11, rel: 1e-11 }, h_min: 1e-6, h_max: 50.0, max_steps: 1_000_000 }
    }

    #[test]
    fn isco_known_values() {
        assert!((isco(&Kerr::new(1.0, 0.0), true) - 6.0).abs() < 1e-12);
        assert!((isco(&Kerr::new(1.0, 0.0), false) - 6.0).abs() < 1e-12);
        assert!((isco(&Kerr::new(1.0, 0.999999), true) - 1.0).abs() < 0.03);
        assert!((isco(&Kerr::new(1.0, 0.999999), false) - 9.0).abs() < 1e-3);
    }

    #[test]
    fn circular_orbit_stays_circular_and_on_phase() {
        for (a, r, pro) in [(0.0, 10.0, true), (0.9, 6.0, true), (0.9, 12.0, false), (0.5, 40.0, true)] {
            let k = Kerr::new(1.0, a);
            let l = circular_equatorial(&k, r, 0.3, pro);
            let u = k.four_velocity(l.pos, l.vel).unwrap();
            let mut s = geodesic::body_state(&k, l.pos, u, 0.0);
            let om = circular_omega(&k, r, pro);
            let period = std::f64::consts::TAU / om.abs();
            let mut h = 0.5;
            let out =
                stepper().integrate(&mut |y: &[f64; 8]| geodesic::body_rhs(&k, y, [0.0; 4]), &mut s, period, &mut h);
            assert_eq!(out, Outcome::Reached);
            let rr = k.radius(geodesic::position(&s));
            assert!((rr - r).abs() < 1e-6 * r, "a={a} r={r}: drifted to {rr}");
            let d = vec3::norm(vec3::sub(geodesic::position(&s), l.pos));
            assert!(d < 1e-5 * r, "a={a} r={r}: phase error {d}");
        }
    }
}
