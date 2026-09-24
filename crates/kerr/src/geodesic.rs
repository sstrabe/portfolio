//! Geodesic motion in Hamiltonian form, `H = ½ g^{μν} p_μ p_ν`.
//!
//! With the Kerr–Schild split `g^{μν} = η^{μν} − f l^μ l^ν` and
//! `L = l^μ p_μ = −p_t + l_i p_i` the equations are
//!
//! ```text
//! dt/dλ   = −p_t + f L
//! dx^i/dλ =  p_i − f L l_i
//! dp_t/dλ =  0
//! dp_i/dλ =  ½ ∂_i f L² + f L p_j ∂_i l_j
//! ```
//!
//! For massive bodies `λ` is proper time and `H = −½`; for light `H = 0`.

use crate::metric::Kerr;
use crate::vec3::V3;

/// Covariant momentum `[p_t, p_x, p_y, p_z]`.
pub type Momentum = [f64; 4];

/// `dx^μ/dλ` and `dp_μ/dλ` at a phase-space point.
#[derive(Clone, Copy, Debug)]
pub struct Flow {
    pub dx: [f64; 4],
    pub dp: [f64; 4],
}

pub fn flow(k: &Kerr, x: V3, p: Momentum) -> Flow {
    let g = k.grad(x);
    let ll = -p[0] + g.l[0] * p[1] + g.l[1] * p[2] + g.l[2] * p[3];
    let fl = g.f * ll;
    let mut dp = [0.0; 4];
    for i in 0..3 {
        let dl_dot_p = g.dl[0][i] * p[1] + g.dl[1][i] * p[2] + g.dl[2][i] * p[3];
        dp[i + 1] = 0.5 * g.df[i] * ll * ll + fl * dl_dot_p;
    }
    Flow { dx: [-p[0] + fl, p[1] - fl * g.l[0], p[2] - fl * g.l[1], p[3] - fl * g.l[2]], dp }
}

/// `H = ½ g^{μν} p_μ p_ν`.
pub fn hamiltonian(k: &Kerr, x: V3, p: Momentum) -> f64 {
    let t = k.terms(x);
    let ll = -p[0] + t.l[0] * p[1] + t.l[1] * p[2] + t.l[2] * p[3];
    0.5 * (-p[0] * p[0] + p[1] * p[1] + p[2] * p[2] + p[3] * p[3] - t.f * ll * ll)
}

/// Conserved energy at infinity `E = −p_t` (per unit rest mass).
pub fn energy(p: Momentum) -> f64 {
    -p[0]
}

/// Conserved axial angular momentum `L_z = p_φ = x p_y − y p_x`.
pub fn angular_momentum_z(x: V3, p: Momentum) -> f64 {
    x[0] * p[2] - x[1] * p[1]
}

/// Carter constant `Q = p_θ² + cos²θ (a²(μ² − E²) + L_z² / sin²θ)`, written
/// with Kerr–Schild Cartesian quantities. `mu2` is the squared rest mass
/// (1 for bodies, 0 for photons).
///
/// `p_θ` is extracted from the Cartesian momentum through the Boyer–Lindquist
/// relations, which hold unchanged in Kerr–Schild form because the two
/// charts share `r` and `θ`.
pub fn carter_constant(k: &Kerr, x: V3, p: Momentum, mu2: f64) -> f64 {
    let a = k.a;
    let r = k.radius(x);
    let cos_th = (x[2] / r).clamp(-1.0, 1.0);
    let sin2 = (1.0 - cos_th * cos_th).max(1e-300);
    let e = -p[0];
    let lz = angular_momentum_z(x, p);
    // ∂(x,y,z)/∂θ at fixed r and Kerr–Schild φ:
    // x + i y = (r + i a) e^{iφ} sin θ, z = r cos θ.
    let sin_th = sin2.sqrt();
    let rho_xy = (x[0] * x[0] + x[1] * x[1]).sqrt();
    let (cphi_mix, sphi_mix) = if rho_xy > 0.0 { (x[0] / rho_xy, x[1] / rho_xy) } else { (1.0, 0.0) };
    // (x, y) = sqrt(r² + a²) sin θ (cos(φ + β), sin(φ + β)) with tan β = a / r,
    // so ∂(x, y)/∂θ = sqrt(r² + a²) cos θ (cos(φ+β), sin(φ+β)).
    let q = (r * r + a * a).sqrt();
    let dx_dth = q * cos_th * cphi_mix;
    let dy_dth = q * cos_th * sphi_mix;
    let dz_dth = -r * sin_th;
    let p_th = p[1] * dx_dth + p[2] * dy_dth + p[3] * dz_dth;
    p_th * p_th + cos_th * cos_th * (a * a * (mu2 - e * e) + lz * lz / sin2)
}

/// Phase-space state of a massive body integrated in coordinate time:
/// `[x, y, z, p_t, p_x, p_y, p_z, τ]`.
pub type BodyState = [f64; 8];

pub fn position(s: &BodyState) -> V3 {
    [s[0], s[1], s[2]]
}

pub fn momentum(s: &BodyState) -> Momentum {
    [s[3], s[4], s[5], s[6]]
}

/// Contravariant 4-velocity `u^μ = g^{μν} p_ν` of a body state.
pub fn four_velocity(k: &Kerr, s: &BodyState) -> [f64; 4] {
    k.raise(position(s), momentum(s))
}

/// Coordinate velocity `dx/dt`.
pub fn coordinate_velocity(k: &Kerr, s: &BodyState) -> V3 {
    let u = four_velocity(k, s);
    [u[1] / u[0], u[2] / u[0], u[3] / u[0]]
}

/// Build a body state from position and 4-velocity.
pub fn body_state(k: &Kerr, x: V3, u: [f64; 4], tau: f64) -> BodyState {
    let p = k.lower(x, u);
    [x[0], x[1], x[2], p[0], p[1], p[2], p[3], tau]
}

/// Converts a coordinate 3-acceleration perturbation `a^i` (as it would
/// appear in `d²x^i/dt²`) into a covariant 4-force per unit mass that is
/// orthogonal to `u`, so the mass shell `u·u = −1` is preserved exactly.
pub fn coordinate_accel_to_force(k: &Kerr, x: V3, u: [f64; 4], acc: V3) -> [f64; 4] {
    let ut2 = u[0] * u[0];
    let f_up = [0.0, ut2 * acc[0], ut2 * acc[1], ut2 * acc[2]];
    let u_low = k.lower(x, u);
    // F⊥ = F + (F·u) u, with u·u = −1.
    let fu = u_low[1] * f_up[1] + u_low[2] * f_up[2] + u_low[3] * f_up[3];
    let perp = [fu * u[0], f_up[1] + fu * u[1], f_up[2] + fu * u[2], f_up[3] + fu * u[3]];
    k.lower(x, perp)
}

/// Right-hand side `d(state)/dt` for a body with an additional covariant
/// 4-force `force` (per unit mass, with respect to proper time).
/// Returns `None` if `dt/dτ ≤ 0`, which only happens for invalid states.
pub fn body_rhs(k: &Kerr, s: &BodyState, force: [f64; 4]) -> Option<BodyState> {
    let fl = flow(k, position(s), momentum(s));
    let dt_dtau = fl.dx[0];
    if dt_dtau.is_nan() || dt_dtau <= 0.0 {
        return None;
    }
    let inv = 1.0 / dt_dtau;
    Some([
        fl.dx[1] * inv,
        fl.dx[2] * inv,
        fl.dx[3] * inv,
        (fl.dp[0] + force[0]) * inv,
        (fl.dp[1] + force[1]) * inv,
        (fl.dp[2] + force[2]) * inv,
        (fl.dp[3] + force[3]) * inv,
        inv,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flow_matches_numerical_hamiltonian_derivatives() {
        let k = Kerr::new(1.0, 0.9);
        let x = [6.0, -2.5, 1.5];
        let p = [-0.95, 0.1, 0.3, -0.05];
        let fl = flow(&k, x, p);
        let h = 1e-6;
        for i in 0..3 {
            let mut xp = x;
            let mut xm = x;
            xp[i] += h;
            xm[i] -= h;
            let d = (hamiltonian(&k, xp, p) - hamiltonian(&k, xm, p)) / (2.0 * h);
            assert!((fl.dp[i + 1] + d).abs() < 1e-8, "dp[{i}]");
        }
        for mu in 0..4 {
            let mut pp = p;
            let mut pm = p;
            pp[mu] += h;
            pm[mu] -= h;
            let d = (hamiltonian(&k, x, pp) - hamiltonian(&k, x, pm)) / (2.0 * h);
            assert!((fl.dx[mu] - d).abs() < 1e-8, "dx[{mu}]");
        }
    }

    #[test]
    fn force_is_orthogonal_to_velocity() {
        let k = Kerr::new(1.0, 0.7);
        let x = [8.0, 3.0, -1.0];
        let u = k.four_velocity(x, [0.1, -0.3, 0.05]).unwrap();
        let f = coordinate_accel_to_force(&k, x, u, [1e-3, -2e-3, 5e-4]);
        let fu = f[0] * u[0] + f[1] * u[1] + f[2] * u[2] + f[3] * u[3];
        assert!(fu.abs() < 1e-15);
    }

    #[test]
    fn generic_orbit_conserves_e_lz_carter_and_mass_shell() {
        use crate::integrate::{Dopri5, Outcome, Tolerance};
        let k = Kerr::new(1.0, 0.95);
        let x = [9.0, 2.0, 5.0];
        let u = k.four_velocity(x, [-0.05, 0.28, 0.12]).unwrap();
        let mut s = body_state(&k, x, u, 0.0);
        let c0 = (energy(momentum(&s)), angular_momentum_z(x, momentum(&s)), carter_constant(&k, x, momentum(&s), 1.0));
        let h0 = hamiltonian(&k, x, momentum(&s));
        assert!((h0 + 0.5).abs() < 1e-12);
        let d = Dopri5 { tol: Tolerance { abs: 1e-11, rel: 1e-11 }, h_min: 1e-8, h_max: 20.0, max_steps: 1_000_000 };
        let mut h = 0.1;
        let out = d.integrate(&mut |y: &BodyState| body_rhs(&k, y, [0.0; 4]), &mut s, 5000.0, &mut h);
        assert_eq!(out, Outcome::Reached);
        let (xp, p) = (position(&s), momentum(&s));
        assert!((energy(p) - c0.0).abs() < 1e-9);
        assert!((angular_momentum_z(xp, p) - c0.1).abs() < 1e-8);
        assert!((carter_constant(&k, xp, p, 1.0) - c0.2).abs() < 1e-7 * c0.2.abs().max(1.0), "Q drift");
        assert!((hamiltonian(&k, xp, p) + 0.5).abs() < 1e-9);
    }
}
