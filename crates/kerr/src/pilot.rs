//! The visitor's ship: an accelerated timelike worldline carrying an
//! orthonormal tetrad.
//!
//! * `e[0]` is the 4-velocity `u`; `e[1]` forward, `e[2]` left, `e[3]` up
//!   (a right-handed spatial triad).
//! * Thrust is a proper acceleration `A` given in the ship frame, so the
//!   ship can approach but never reach `c`: `u` stays unit timelike.
//! * The spatial axes are Fermi–Walker transported (they behave like
//!   gyroscopes) plus the pilot's commanded rotation.
//!
//! ```text
//! du^μ/dτ   = −Γ^μ_{αβ} u^α u^β + a^μ,             a = A^i e_i
//! de_i^μ/dτ = −Γ^μ_{αβ} u^α e_i^β + (a·e_i) u^μ
//! ```

use crate::integrate::rk4_step;
use crate::metric::{Gamma, Kerr};
use crate::vec3::{self, V3, V4};

pub type Tetrad = [V4; 4];

pub const FORWARD: usize = 1;
pub const LEFT: usize = 2;
pub const UP: usize = 3;

#[derive(Clone, Debug)]
pub struct Pilot {
    /// Event `(t, x, y, z)`.
    pub x: V4,
    pub e: Tetrad,
    /// Proper time.
    pub tau: f64,
}

/// Commands for one frame, in the ship frame.
#[derive(Clone, Copy, Debug, Default)]
pub struct Command {
    /// Proper acceleration along (forward, left, up), units of `1/M`.
    pub accel: V3,
    /// Angular velocity about (forward, left, up) in rad per unit proper time.
    pub spin: V3,
}

impl Pilot {
    /// A ship at `pos` moving with coordinate velocity `vel`, oriented so that
    /// forward points along `look` (projected into its rest frame) and up is
    /// as close as possible to `up_hint`.
    pub fn new(k: &Kerr, pos: V3, vel: V3, look: V3, up_hint: V3) -> Option<Self> {
        let u = k.four_velocity(pos, vel)?;
        let x = [0.0, pos[0], pos[1], pos[2]];
        let fwd = [0.0, look[0], look[1], look[2]];
        let up = [0.0, up_hint[0], up_hint[1], up_hint[2]];
        let left_guess = vec3::cross(up_hint, look);
        let left = [0.0, left_guess[0], left_guess[1], left_guess[2]];
        let mut e = [u, fwd, left, up];
        orthonormalize(k, pos, &mut e);
        // Make sure the triad is right-handed: forward × left = up.
        let mut p = Self { x, e, tau: 0.0 };
        if p.local_handedness(k) < 0.0 {
            p.e[LEFT] = p.e[LEFT].map(|v| -v);
        }
        Some(p)
    }

    pub fn position(&self) -> V3 {
        vec3::spatial(self.x)
    }

    /// `dt/dτ` — how fast the rest of the universe runs relative to the ship.
    pub fn time_dilation(&self) -> f64 {
        self.e[0][0]
    }

    /// Speed relative to the local static-at-infinity grid is not defined
    /// inside the ergosphere, so report the speed relative to the local
    /// normal observer of the `t = const` slicing, `v = √(1 − 1/γ²)` with
    /// `γ = −n·u`.
    pub fn normal_gamma(&self, k: &Kerr) -> f64 {
        let t = k.terms(self.position());
        // n_μ = −α ∇_μ t with α = 1/√(−g^{tt}) = 1/√(1 + f).
        let alpha = 1.0 / (1.0 + t.f).sqrt();
        alpha * self.e[0][0]
    }

    fn local_handedness(&self, k: &Kerr) -> f64 {
        // Sign of ε(u, e1, e2, e3) via the determinant of the component
        // matrix (the metric determinant is −1 in Kerr–Schild form).
        let _ = k;
        det4(&self.e)
    }

    /// Advance by proper time `dtau` under a command.
    pub fn step(&mut self, k: &Kerr, cmd: &Command, dtau: f64) {
        let r = k.radius(self.position());
        let accel = vec3::norm(cmd.accel);
        // Resolve the curvature scale (move at most ~3% of r per substep in
        // coordinates) and the thrust (rapidity change ≲ 0.5 per substep).
        // Needlessly small steps are harmful at high γ: renormalizing u
        // cancels terms of order γ², so rounding error accumulates per step.
        let h_max = (0.03 * r.max(1.0) / self.e[0][0].max(1.0)).min(0.5 / (1.0 + 0.5 * accel));
        let n = (dtau.abs() / h_max).ceil().clamp(1.0, 2000.0) as usize;
        let h = dtau / n as f64;
        for _ in 0..n {
            self.substep(k, cmd, h);
        }
        self.tau += dtau;
    }

    fn substep(&mut self, k: &Kerr, cmd: &Command, h: f64) {
        let mut y = [0.0; 20];
        y[..4].copy_from_slice(&self.x);
        for a in 0..4 {
            y[4 + 4 * a..8 + 4 * a].copy_from_slice(&self.e[a]);
        }
        let acc = cmd.accel;
        let mut f = |y: &[f64; 20]| {
            let pos = [y[1], y[2], y[3]];
            let gam = k.christoffel(pos);
            let u = [y[4], y[5], y[6], y[7]];
            let mut out = [0.0; 20];
            out[..4].copy_from_slice(&u);
            let mut a4 = [0.0; 4];
            for i in 0..3 {
                for mu in 0..4 {
                    a4[mu] += acc[i] * y[8 + 4 * i + mu];
                }
            }
            let du = transport(&gam, u, u);
            for mu in 0..4 {
                out[4 + mu] = du[mu] + a4[mu];
            }
            for i in 0..3 {
                let e = [y[8 + 4 * i], y[9 + 4 * i], y[10 + 4 * i], y[11 + 4 * i]];
                let de = transport(&gam, u, e);
                for mu in 0..4 {
                    out[8 + 4 * i + mu] = de[mu] + acc[i] * u[mu];
                }
            }
            out
        };
        let y = rk4_step(&mut f, &y, h);
        self.x.copy_from_slice(&y[..4]);
        for a in 0..4 {
            self.e[a].copy_from_slice(&y[4 + 4 * a..8 + 4 * a]);
        }
        self.rotate(cmd.spin, h);
        orthonormalize(k, self.position(), &mut self.e);
    }

    /// Rotate the spatial triad with angular velocity `w` (ship frame) for
    /// proper time `h`.
    pub fn rotate(&mut self, w: V3, h: f64) {
        let ang = vec3::norm(w) * h;
        if ang == 0.0 {
            return;
        }
        let axis = vec3::normalize(w);
        // Column i of the rotation matrix = image of basis vector i.
        let cols = [
            vec3::rotate([1.0, 0.0, 0.0], axis, ang),
            vec3::rotate([0.0, 1.0, 0.0], axis, ang),
            vec3::rotate([0.0, 0.0, 1.0], axis, ang),
        ];
        let old = [self.e[1], self.e[2], self.e[3]];
        for i in 0..3 {
            let mut v = [0.0; 4];
            for j in 0..3 {
                for mu in 0..4 {
                    v[mu] += cols[i][j] * old[j][mu];
                }
            }
            self.e[i + 1] = v;
        }
    }

    /// Components, in the ship frame (forward, left, up), of a coordinate
    /// displacement `d` taken on the ship's `t = const` slice.
    pub fn local_components(&self, k: &Kerr, d: V3) -> V3 {
        let p = self.position();
        let v = [0.0, d[0], d[1], d[2]];
        [k.dot(p, v, self.e[1]), k.dot(p, v, self.e[2]), k.dot(p, v, self.e[3])]
    }

    /// Velocity of a body with 4-velocity `w` (at the ship's event)
    /// measured in the ship frame.
    pub fn relative_velocity(&self, k: &Kerr, w: V4) -> V3 {
        let p = self.position();
        let gamma = -k.dot(p, w, self.e[0]);
        [k.dot(p, w, self.e[1]) / gamma, k.dot(p, w, self.e[2]) / gamma, k.dot(p, w, self.e[3]) / gamma]
    }

    /// Replace the 4-velocity (e.g. when docking), keeping the attitude as
    /// close as possible.
    pub fn set_velocity(&mut self, k: &Kerr, u: V4) {
        self.e[0] = u;
        orthonormalize(k, self.position(), &mut self.e);
    }
}

fn transport(gam: &Gamma, u: V4, v: V4) -> V4 {
    let mut out = [0.0; 4];
    for mu in 0..4 {
        let mut s = 0.0;
        for a in 0..4 {
            for b in 0..4 {
                s += gam[mu][a][b] * u[a] * v[b];
            }
        }
        out[mu] = -s;
    }
    out
}

/// Gram–Schmidt with respect to the metric at `p`; `e[0]` is normalized as
/// timelike, the rest as spacelike and orthogonal to all previous legs.
pub fn orthonormalize(k: &Kerr, p: V3, e: &mut Tetrad) {
    let n0 = (-k.dot(p, e[0], e[0])).sqrt();
    e[0] = e[0].map(|v| v / n0);
    for i in 1..4 {
        let mut v = e[i];
        // Timelike leg has norm −1: v ← v + (v·u) u.
        let vu = k.dot(p, v, e[0]);
        for mu in 0..4 {
            v[mu] += vu * e[0][mu];
        }
        for j in 1..i {
            let vj = k.dot(p, v, e[j]);
            for mu in 0..4 {
                v[mu] -= vj * e[j][mu];
            }
        }
        let n = k.dot(p, v, v).sqrt();
        e[i] = v.map(|c| c / n);
    }
}

fn det4(m: &Tetrad) -> f64 {
    let mut det = 0.0;
    for c in 0..4 {
        let minor: Vec<[f64; 3]> = (1..4)
            .map(|r| {
                let mut row = [0.0; 3];
                let mut k = 0;
                for cc in 0..4 {
                    if cc != c {
                        row[k] = m[r][cc];
                        k += 1;
                    }
                }
                row
            })
            .collect();
        let d3 = minor[0][0] * (minor[1][1] * minor[2][2] - minor[1][2] * minor[2][1])
            - minor[0][1] * (minor[1][0] * minor[2][2] - minor[1][2] * minor[2][0])
            + minor[0][2] * (minor[1][0] * minor[2][1] - minor[1][1] * minor[2][0]);
        det += if c % 2 == 0 { m[0][c] * d3 } else { -m[0][c] * d3 };
    }
    det
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geodesic;
    use crate::integrate::{Dopri5, Tolerance};
    use crate::orbit;

    fn check_orthonormal(k: &Kerr, p: &Pilot) {
        for a in 0..4 {
            for b in 0..4 {
                let d = k.dot(p.position(), p.e[a], p.e[b]);
                let expect = if a != b {
                    0.0
                } else if a == 0 {
                    -1.0
                } else {
                    1.0
                };
                assert!((d - expect).abs() < 1e-9, "e{a}·e{b} = {d}");
            }
        }
    }

    #[test]
    fn free_fall_matches_hamiltonian_geodesic() {
        let k = Kerr::new(1.0, 0.9);
        let l = orbit::inclined_circular(&k, 15.0, 0.6, 0.3, 1.0);
        let vel = vec3::scale(l.vel, 0.9);
        let mut pilot = Pilot::new(&k, l.pos, vel, [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]).unwrap();
        check_orthonormal(&k, &pilot);
        let tau = 150.0;
        pilot.step(&k, &Command::default(), tau);
        check_orthonormal(&k, &pilot);

        // Same initial data through the Hamiltonian equations in proper time.
        let u = k.four_velocity(l.pos, vel).unwrap();
        let p = k.lower(l.pos, u);
        let mut y = [0.0, l.pos[0], l.pos[1], l.pos[2], p[0], p[1], p[2], p[3]];
        let d = Dopri5 { tol: Tolerance { abs: 1e-12, rel: 1e-12 }, h_min: 1e-8, h_max: 5.0, max_steps: 1_000_000 };
        let mut h = 0.1;
        d.integrate(
            &mut |y: &[f64; 8]| {
                let fl = geodesic::flow(&k, [y[1], y[2], y[3]], [y[4], y[5], y[6], y[7]]);
                Some([fl.dx[0], fl.dx[1], fl.dx[2], fl.dx[3], fl.dp[0], fl.dp[1], fl.dp[2], fl.dp[3]])
            },
            &mut y,
            tau,
            &mut h,
        );
        for mu in 0..4 {
            assert!((pilot.x[mu] - y[mu]).abs() < 1e-5, "x[{mu}]: {} vs {}", pilot.x[mu], y[mu]);
        }
    }

    #[test]
    fn thrust_gives_requested_proper_acceleration_and_never_reaches_c() {
        let k = Kerr::new(1.0, 0.5);
        let mut pilot = Pilot::new(&k, [60.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]).unwrap();
        let cmd = Command { accel: [0.2, 0.0, 0.0], spin: [0.0; 3] };
        // Short burn: rapidity grows by ≈ A·τ minus gravity's share.
        pilot.step(&k, &cmd, 10.0);
        check_orthonormal(&k, &pilot);
        let g = pilot.normal_gamma(&k);
        assert!(g > 2.5, "γ = {g}");
        // Long burn: γ keeps growing, v < 1 always (u stays timelike by construction).
        pilot.step(&k, &cmd, 20.0);
        let g2 = pilot.normal_gamma(&k);
        assert!(g2 > g && g2.is_finite());
        check_orthonormal(&k, &pilot);
    }

    #[test]
    fn coasting_at_extreme_speed_conserves_energy() {
        let k = Kerr::new(1.0, 0.9);
        let mut p = Pilot::new(&k, [5e6, 1e5, 3e5], [0.0; 3], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]).unwrap();
        // Burn to γ ≈ 4 × 10⁴, then coast for a few hundred frames.
        let burn = Command { accel: [0.06, 0.0, 0.0], spin: [0.0; 3] };
        for _ in 0..376 {
            p.step(&k, &burn, 0.5);
        }
        let e0 = -k.lower(p.position(), p.e[0])[0];
        assert!(e0 > 1e4, "E = {e0}");
        for _ in 0..600 {
            p.step(&k, &Command::default(), 0.78);
        }
        let e = -k.lower(p.position(), p.e[0])[0];
        assert!((e / e0 - 1.0).abs() < 1e-4, "energy drifted by {}", e / e0 - 1.0);
    }

    #[test]
    fn rotation_keeps_handedness() {
        let k = Kerr::new(1.0, 0.0);
        let mut pilot = Pilot::new(&k, [30.0, 0.0, 0.0], [0.0, 0.1, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]).unwrap();
        let before = det4(&pilot.e);
        pilot.step(&k, &Command { accel: [0.0; 3], spin: [0.3, -0.2, 0.5] }, 5.0);
        check_orthonormal(&k, &pilot);
        assert!(det4(&pilot.e).signum() == before.signum());
        // Pure yaw turns forward towards left.
        let mut p2 = Pilot::new(&k, [30.0, 0.0, 0.0], [0.0; 3], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]).unwrap();
        let left = p2.e[LEFT];
        p2.rotate([0.0, 0.0, 1.0], std::f64::consts::FRAC_PI_2);
        for mu in 0..4 {
            assert!((p2.e[FORWARD][mu] - left[mu]).abs() < 1e-12);
        }
    }
}
