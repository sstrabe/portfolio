//! Kerr metric in Cartesian Kerr–Schild coordinates.
//!
//! Index conventions: 4-vectors are `[t, x, y, z]`, spatial gradients are
//! `[∂x, ∂y, ∂z]` (the metric is stationary so `∂t = 0`). Signature `−+++`.

use crate::dual::Dual3;
use crate::vec3::V3;

pub type Mat4 = [[f64; 4]; 4];
pub type Gamma = [[[f64; 4]; 4]; 4];

const ETA: [f64; 4] = [-1.0, 1.0, 1.0, 1.0];

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Kerr {
    /// Mass `M` (1 in simulation units, 0 gives Minkowski space).
    pub m: f64,
    /// Spin parameter `a = J / M`, `|a| < M`.
    pub a: f64,
}

/// Scalar field `f` and null covector `l_i` of the Kerr–Schild form.
#[derive(Clone, Copy, Debug)]
pub struct KsTerms {
    pub r: f64,
    pub f: f64,
    /// Spatial components of `l_μ` (`l_t = 1`).
    pub l: V3,
}

/// [`KsTerms`] plus spatial gradients.
#[derive(Clone, Copy, Debug)]
pub struct KsGrad {
    pub r: f64,
    pub dr: V3,
    pub f: f64,
    pub df: V3,
    pub l: V3,
    /// `dl[j][i] = ∂_i l_j`.
    pub dl: [V3; 3],
}

impl Kerr {
    pub const fn new(m: f64, a: f64) -> Self {
        Self { m, a }
    }

    /// Outer horizon radius `r₊ = M + √(M² − a²)`.
    pub fn r_plus(&self) -> f64 {
        self.m + (self.m * self.m - self.a * self.a).max(0.0).sqrt()
    }

    /// Outer ergosurface radius at polar angle given by `cos θ = z / r`.
    pub fn r_ergo(&self, cos_theta: f64) -> f64 {
        self.m + (self.m * self.m - self.a * self.a * cos_theta * cos_theta).max(0.0).sqrt()
    }

    /// Kerr–Schild radius `r(x, y, z)`: the positive root of
    /// `r⁴ − (R² − a²) r² − a² z² = 0`.
    pub fn radius(&self, p: V3) -> f64 {
        let a2 = self.a * self.a;
        let rho = p[0] * p[0] + p[1] * p[1] + p[2] * p[2] - a2;
        let z2 = p[2] * p[2];
        let disc = (rho * rho + 4.0 * a2 * z2).sqrt();
        // Two algebraically equal forms, picked to avoid cancellation.
        let r2 = if rho >= 0.0 { 0.5 * (rho + disc) } else { 2.0 * a2 * z2 / (disc - rho) };
        r2.max(0.0).sqrt()
    }

    pub fn terms(&self, p: V3) -> KsTerms {
        let a = self.a;
        let r = self.radius(p);
        let r2 = r * r;
        let q = r2 + a * a;
        let sigma = r2 * r2 + a * a * p[2] * p[2];
        KsTerms {
            r,
            f: 2.0 * self.m * r2 * r / sigma,
            l: [(r * p[0] + a * p[1]) / q, (r * p[1] - a * p[0]) / q, p[2] / r],
        }
    }

    /// Analytic gradients of `r`, `f`, `l_i`. These formulas are mirrored in
    /// the WGSL shaders; see the tests for the dual-number cross-check.
    pub fn grad(&self, p: V3) -> KsGrad {
        let a = self.a;
        let a2 = a * a;
        let (x, y, z) = (p[0], p[1], p[2]);
        let r = self.radius(p);
        let r2 = r * r;
        let d = r2 + a2 * z * z / r2;
        let dr = [x * r / d, y * r / d, z * (r2 + a2) / (r * d)];
        let sigma = r2 * r2 + a2 * z * z;
        let f = 2.0 * self.m * r2 * r / sigma;
        let r3 = r2 * r;
        let df = [
            f * (3.0 * dr[0] / r - 4.0 * r3 * dr[0] / sigma),
            f * (3.0 * dr[1] / r - 4.0 * r3 * dr[1] / sigma),
            f * (3.0 * dr[2] / r - (4.0 * r3 * dr[2] + 2.0 * a2 * z) / sigma),
        ];
        let q = r2 + a2;
        let l = [(r * x + a * y) / q, (r * y - a * x) / q, z / r];
        let mut dl = [[0.0; 3]; 3];
        for i in 0..3 {
            let dx = if i == 0 { 1.0 } else { 0.0 };
            let dy = if i == 1 { 1.0 } else { 0.0 };
            let dz = if i == 2 { 1.0 } else { 0.0 };
            let two_r_dr_over_q = 2.0 * r * dr[i] / q;
            dl[0][i] = (dr[i] * x + r * dx + a * dy) / q - l[0] * two_r_dr_over_q;
            dl[1][i] = (dr[i] * y + r * dy - a * dx) / q - l[1] * two_r_dr_over_q;
            dl[2][i] = dz / r - z * dr[i] / r2;
        }
        KsGrad { r, dr, f, df, l, dl }
    }

    /// Reference implementation of the Kerr–Schild terms with automatic
    /// differentiation. Slow; used for testing.
    pub fn terms_dual(&self, p: V3) -> (Dual3, Dual3, [Dual3; 3]) {
        let a = self.a;
        let x = Dual3::var(p[0], 0);
        let y = Dual3::var(p[1], 1);
        let z = Dual3::var(p[2], 2);
        let a2 = a * a;
        let rho = x * x + y * y + z * z - a2;
        let disc = (rho * rho + 4.0 * a2 * (z * z)).sqrt();
        let r2 = if rho.v >= 0.0 { (rho + disc) * 0.5 } else { (z * z) * (2.0 * a2) / (disc - rho) };
        let r = r2.sqrt();
        let f = (r2 * r) * (2.0 * self.m) / (r2 * r2 + (z * z) * a2);
        let q = r2 + a2;
        let l = [(r * x + y * a) / q, (r * y - x * a) / q, z / r];
        (r, f, l)
    }

    /// Covariant metric `g_{μν}`.
    pub fn metric(&self, p: V3) -> Mat4 {
        let t = self.terms(p);
        let lo = [1.0, t.l[0], t.l[1], t.l[2]];
        let mut g = [[0.0; 4]; 4];
        for mu in 0..4 {
            for nu in 0..4 {
                g[mu][nu] = t.f * lo[mu] * lo[nu];
            }
            g[mu][mu] += ETA[mu];
        }
        g
    }

    /// Contravariant metric `g^{μν}`.
    pub fn inverse(&self, p: V3) -> Mat4 {
        let t = self.terms(p);
        let up = [-1.0, t.l[0], t.l[1], t.l[2]];
        let mut g = [[0.0; 4]; 4];
        for mu in 0..4 {
            for nu in 0..4 {
                g[mu][nu] = -t.f * up[mu] * up[nu];
            }
            g[mu][mu] += ETA[mu];
        }
        g
    }

    /// Christoffel symbols `Γ^α_{μν}` (index order `[α][μ][ν]`).
    pub fn christoffel(&self, p: V3) -> Gamma {
        let k = self.grad(p);
        let lo = [1.0, k.l[0], k.l[1], k.l[2]];
        // dg[s][μ][ν] = ∂_s g_{μν}; s = 0 (time) vanishes.
        let mut dg = [[[0.0; 4]; 4]; 4];
        for i in 0..3 {
            let dlo = [0.0, k.dl[0][i], k.dl[1][i], k.dl[2][i]];
            for mu in 0..4 {
                for nu in 0..4 {
                    dg[i + 1][mu][nu] = k.df[i] * lo[mu] * lo[nu] + k.f * (dlo[mu] * lo[nu] + lo[mu] * dlo[nu]);
                }
            }
        }
        let up = [-1.0, k.l[0], k.l[1], k.l[2]];
        let mut ginv = [[0.0; 4]; 4];
        for mu in 0..4 {
            for nu in 0..4 {
                ginv[mu][nu] = -k.f * up[mu] * up[nu];
            }
            ginv[mu][mu] += ETA[mu];
        }
        // Γ_{βμν} with the first index lowered.
        let mut low = [[[0.0; 4]; 4]; 4];
        for b in 0..4 {
            for mu in 0..4 {
                for nu in mu..4 {
                    let v = 0.5 * (dg[mu][b][nu] + dg[nu][b][mu] - dg[b][mu][nu]);
                    low[b][mu][nu] = v;
                    low[b][nu][mu] = v;
                }
            }
        }
        let mut gamma = [[[0.0; 4]; 4]; 4];
        for al in 0..4 {
            for mu in 0..4 {
                for nu in mu..4 {
                    let mut s = 0.0;
                    for b in 0..4 {
                        s += ginv[al][b] * low[b][mu][nu];
                    }
                    gamma[al][mu][nu] = s;
                    gamma[al][nu][mu] = s;
                }
            }
        }
        gamma
    }

    /// `g_{μν} v^ν`.
    pub fn lower(&self, p: V3, v: [f64; 4]) -> [f64; 4] {
        let t = self.terms(p);
        let lv = v[0] + t.l[0] * v[1] + t.l[1] * v[2] + t.l[2] * v[3];
        let fl = t.f * lv;
        [-v[0] + fl, v[1] + fl * t.l[0], v[2] + fl * t.l[1], v[3] + fl * t.l[2]]
    }

    /// `g^{μν} w_ν`.
    pub fn raise(&self, p: V3, w: [f64; 4]) -> [f64; 4] {
        let t = self.terms(p);
        let lw = -w[0] + t.l[0] * w[1] + t.l[1] * w[2] + t.l[2] * w[3];
        let fl = t.f * lw;
        [-w[0] + fl, w[1] - fl * t.l[0], w[2] - fl * t.l[1], w[3] - fl * t.l[2]]
    }

    /// `g_{μν} u^μ v^ν`.
    pub fn dot(&self, p: V3, u: [f64; 4], v: [f64; 4]) -> f64 {
        let t = self.terms(p);
        let lu = u[0] + t.l[0] * u[1] + t.l[1] * u[2] + t.l[2] * u[3];
        let lv = v[0] + t.l[0] * v[1] + t.l[1] * v[2] + t.l[2] * v[3];
        -u[0] * v[0] + u[1] * v[1] + u[2] * v[2] + u[3] * v[3] + t.f * lu * lv
    }

    /// Future-directed unit 4-velocity for coordinate 3-velocity `v = dx/dt`.
    /// Returns `None` if `(1, v)` is not timelike at `p`.
    pub fn four_velocity(&self, p: V3, v: V3) -> Option<[f64; 4]> {
        let w = [1.0, v[0], v[1], v[2]];
        let n = -self.dot(p, w, w);
        (n > 0.0).then(|| {
            let ut = 1.0 / n.sqrt();
            [ut, ut * v[0], ut * v[1], ut * v[2]]
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const POINTS: [V3; 6] = [
        [7.0, -3.0, 2.0],
        [2.1, 0.4, -1.3],
        [-30.0, 12.0, 40.0],
        [0.3, 0.2, 1.8],
        [1.2, -0.7, 0.05],
        [150.0, 3.0, -0.001],
    ];

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * (1.0 + a.abs().max(b.abs()))
    }

    #[test]
    fn radius_solves_quartic() {
        for a in [0.0, 0.5, 0.94, 0.999] {
            let k = Kerr::new(1.0, a);
            for p in POINTS {
                let r = k.radius(p);
                let big_r2 = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
                let resid = r.powi(4) - (big_r2 - a * a) * r * r - a * a * p[2] * p[2];
                assert!(resid.abs() < 1e-9 * big_r2 * big_r2, "a={a} p={p:?} resid={resid}");
            }
        }
    }

    #[test]
    fn analytic_gradient_matches_dual_numbers() {
        for a in [0.0, 0.3, 0.9, 0.998] {
            let k = Kerr::new(1.0, a);
            for p in POINTS {
                let g = k.grad(p);
                let (r, f, l) = k.terms_dual(p);
                assert!(close(g.r, r.v, 1e-12));
                assert!(close(g.f, f.v, 1e-12));
                for i in 0..3 {
                    assert!(close(g.dr[i], r.d[i], 1e-9), "dr a={a} p={p:?} i={i}");
                    assert!(close(g.df[i], f.d[i], 1e-9), "df a={a} p={p:?} i={i}");
                    for j in 0..3 {
                        assert!(close(g.l[j], l[j].v, 1e-12));
                        assert!(
                            close(g.dl[j][i], l[j].d[i], 1e-9),
                            "dl[{j}][{i}] a={a} p={p:?}: {} vs {}",
                            g.dl[j][i],
                            l[j].d[i]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn inverse_is_inverse() {
        let k = Kerr::new(1.0, 0.9);
        for p in POINTS {
            let g = k.metric(p);
            let gi = k.inverse(p);
            for a in 0..4 {
                for b in 0..4 {
                    let s: f64 = (0..4).map(|c| gi[a][c] * g[c][b]).sum();
                    let expect = if a == b { 1.0 } else { 0.0 };
                    assert!((s - expect).abs() < 1e-12, "p={p:?} [{a}][{b}] = {s}");
                }
            }
        }
    }

    #[test]
    fn l_is_null() {
        let k = Kerr::new(1.0, 0.7);
        for p in POINTS {
            let t = k.terms(p);
            let n = -1.0 + t.l[0] * t.l[0] + t.l[1] * t.l[1] + t.l[2] * t.l[2];
            assert!(n.abs() < 1e-12);
        }
    }

    #[test]
    fn christoffel_matches_finite_difference_metric() {
        let k = Kerr::new(1.0, 0.8);
        let p = [5.0, -2.0, 3.0];
        let gam = k.christoffel(p);
        let h = 1e-5;
        let mut dg = [[[0.0; 4]; 4]; 4];
        for i in 0..3 {
            let mut pp = p;
            let mut pm = p;
            pp[i] += h;
            pm[i] -= h;
            let gp = k.metric(pp);
            let gm = k.metric(pm);
            for mu in 0..4 {
                for nu in 0..4 {
                    dg[i + 1][mu][nu] = (gp[mu][nu] - gm[mu][nu]) / (2.0 * h);
                }
            }
        }
        let gi = k.inverse(p);
        for al in 0..4 {
            for mu in 0..4 {
                for nu in 0..4 {
                    let mut s = 0.0;
                    for b in 0..4 {
                        s += 0.5 * gi[al][b] * (dg[mu][b][nu] + dg[nu][b][mu] - dg[b][mu][nu]);
                    }
                    assert!((s - gam[al][mu][nu]).abs() < 1e-8, "Γ[{al}][{mu}][{nu}]");
                }
            }
        }
    }

    #[test]
    fn raise_lower_roundtrip() {
        let k = Kerr::new(1.0, 0.6);
        let p = [3.0, 1.0, -2.0];
        let v = [1.3, 0.2, -0.4, 0.1];
        let back = k.raise(p, k.lower(p, v));
        for i in 0..4 {
            assert!((back[i] - v[i]).abs() < 1e-12);
        }
    }
}
