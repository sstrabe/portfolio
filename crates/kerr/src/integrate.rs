//! Explicit Runge–Kutta integrators over fixed-size state arrays.

/// Result of an adaptive integration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    Reached,
    /// The right-hand side refused to evaluate (e.g. left the domain).
    Failed,
    /// The step budget ran out before the target was reached.
    Budget,
}

/// Classic fourth-order Runge–Kutta step.
pub fn rk4_step<const N: usize>(f: &mut impl FnMut(&[f64; N]) -> [f64; N], y: &[f64; N], h: f64) -> [f64; N] {
    let k1 = f(y);
    let k2 = f(&lin(y, &[(0.5 * h, &k1)]));
    let k3 = f(&lin(y, &[(0.5 * h, &k2)]));
    let k4 = f(&lin(y, &[(h, &k3)]));
    let mut out = *y;
    for i in 0..N {
        out[i] += h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
    out
}

#[inline]
fn lin<const N: usize>(y: &[f64; N], terms: &[(f64, &[f64; N])]) -> [f64; N] {
    let mut out = *y;
    for (c, k) in terms {
        for i in 0..N {
            out[i] += c * k[i];
        }
    }
    out
}

// Dormand–Prince 5(4) tableau.
const A21: f64 = 1.0 / 5.0;
const A31: f64 = 3.0 / 40.0;
const A32: f64 = 9.0 / 40.0;
const A41: f64 = 44.0 / 45.0;
const A42: f64 = -56.0 / 15.0;
const A43: f64 = 32.0 / 9.0;
const A51: f64 = 19372.0 / 6561.0;
const A52: f64 = -25360.0 / 2187.0;
const A53: f64 = 64448.0 / 6561.0;
const A54: f64 = -212.0 / 729.0;
const A61: f64 = 9017.0 / 3168.0;
const A62: f64 = -355.0 / 33.0;
const A63: f64 = 46732.0 / 5247.0;
const A64: f64 = 49.0 / 176.0;
const A65: f64 = -5103.0 / 18656.0;
const B1: f64 = 35.0 / 384.0;
const B3: f64 = 500.0 / 1113.0;
const B4: f64 = 125.0 / 192.0;
const B5: f64 = -2187.0 / 6784.0;
const B6: f64 = 11.0 / 84.0;
// Error coefficients: b − b*.
const E1: f64 = 71.0 / 57600.0;
const E3: f64 = -71.0 / 16695.0;
const E4: f64 = 71.0 / 1920.0;
const E5: f64 = -17253.0 / 339200.0;
const E6: f64 = 22.0 / 525.0;
const E7: f64 = -1.0 / 40.0;

/// Tolerances for [`Dopri5`].
#[derive(Clone, Copy, Debug)]
pub struct Tolerance {
    pub abs: f64,
    pub rel: f64,
}

/// Adaptive Dormand–Prince 5(4) integrator for autonomous systems
/// `dy/ds = f(y)` (the independent variable does not appear explicitly,
/// which holds for everything in a stationary spacetime).
pub struct Dopri5 {
    pub tol: Tolerance,
    pub h_min: f64,
    pub h_max: f64,
    pub max_steps: usize,
}

impl Dopri5 {
    /// One trial step; returns the 5th-order solution and the scaled error
    /// norm (≤ 1 means accept).
    fn trial<const N: usize>(
        &self,
        f: &mut impl FnMut(&[f64; N]) -> Option<[f64; N]>,
        y: &[f64; N],
        k1: &[f64; N],
        h: f64,
    ) -> Option<([f64; N], [f64; N], f64)> {
        let k2 = f(&lin(y, &[(h * A21, k1)]))?;
        let k3 = f(&lin(y, &[(h * A31, k1), (h * A32, &k2)]))?;
        let k4 = f(&lin(y, &[(h * A41, k1), (h * A42, &k2), (h * A43, &k3)]))?;
        let k5 = f(&lin(y, &[(h * A51, k1), (h * A52, &k2), (h * A53, &k3), (h * A54, &k4)]))?;
        let k6 = f(&lin(y, &[(h * A61, k1), (h * A62, &k2), (h * A63, &k3), (h * A64, &k4), (h * A65, &k5)]))?;
        let y5 = lin(y, &[(h * B1, k1), (h * B3, &k3), (h * B4, &k4), (h * B5, &k5), (h * B6, &k6)]);
        let k7 = f(&y5)?;
        let mut err2 = 0.0;
        for i in 0..N {
            let e = h * (E1 * k1[i] + E3 * k3[i] + E4 * k4[i] + E5 * k5[i] + E6 * k6[i] + E7 * k7[i]);
            let sc = self.tol.abs + self.tol.rel * y[i].abs().max(y5[i].abs());
            err2 += (e / sc) * (e / sc);
        }
        Some((y5, k7, (err2 / N as f64).sqrt()))
    }

    /// Integrate `y` over an interval of length `span` (may be negative),
    /// starting with step `h` (sign ignored). On return `h` holds a good
    /// step size to continue with.
    pub fn integrate<const N: usize>(
        &self,
        f: &mut impl FnMut(&[f64; N]) -> Option<[f64; N]>,
        y: &mut [f64; N],
        span: f64,
        h: &mut f64,
    ) -> Outcome {
        if span == 0.0 {
            return Outcome::Reached;
        }
        let dir = span.signum();
        let mut left = span.abs();
        let mut step = h.abs().clamp(self.h_min, self.h_max);
        let Some(mut k1) = f(y) else { return Outcome::Failed };
        for _ in 0..self.max_steps {
            let last = step >= left;
            let hs = if last { left } else { step };
            let Some((y5, k7, err)) = self.trial(f, y, &k1, dir * hs) else {
                if step <= self.h_min {
                    return Outcome::Failed;
                }
                step = (step * 0.25).max(self.h_min);
                continue;
            };
            if err <= 1.0 || hs <= self.h_min {
                *y = y5;
                k1 = k7;
                left -= hs;
                let grow = if err == 0.0 { 5.0 } else { (0.9 * err.powf(-0.2)).clamp(0.2, 5.0) };
                if !last {
                    step = (hs * grow).clamp(self.h_min, self.h_max);
                }
                if last || left <= 0.0 {
                    *h = step;
                    return Outcome::Reached;
                }
            } else {
                step = (hs * (0.9 * err.powf(-0.25)).clamp(0.1, 0.9)).max(self.h_min);
            }
        }
        *h = step;
        Outcome::Budget
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dopri5_harmonic_oscillator() {
        let d = Dopri5 { tol: Tolerance { abs: 1e-12, rel: 1e-12 }, h_min: 1e-9, h_max: 1.0, max_steps: 100_000 };
        let mut y = [1.0, 0.0];
        let mut h = 0.1;
        let out = d.integrate(&mut |y: &[f64; 2]| Some([y[1], -y[0]]), &mut y, 10.0, &mut h);
        assert_eq!(out, Outcome::Reached);
        assert!((y[0] - 10f64.cos()).abs() < 1e-9);
        assert!((y[1] + 10f64.sin()).abs() < 1e-9);
        // And back again.
        let out = d.integrate(&mut |y: &[f64; 2]| Some([y[1], -y[0]]), &mut y, -10.0, &mut h);
        assert_eq!(out, Outcome::Reached);
        assert!((y[0] - 1.0).abs() < 1e-9 && y[1].abs() < 1e-9);
    }

    #[test]
    fn rk4_exponential() {
        let mut y = [1.0];
        for _ in 0..100 {
            y = rk4_step(&mut |y: &[f64; 1]| [y[0]], &y, 0.01);
        }
        assert!((y[0] - 1f64.exp()).abs() < 1e-9);
    }
}
