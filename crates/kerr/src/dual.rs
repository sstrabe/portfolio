//! Forward-mode dual numbers carrying a spatial gradient.
//!
//! Used as an independent reference for the hand-derived analytic metric
//! gradients in [`crate::metric`] (the GPU shaders reuse those analytic
//! formulas, so they need a trustworthy check).

use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dual3 {
    pub v: f64,
    pub d: [f64; 3],
}

impl Dual3 {
    pub const fn constant(v: f64) -> Self {
        Self { v, d: [0.0; 3] }
    }

    /// The `i`-th coordinate as an independent variable.
    pub const fn var(v: f64, i: usize) -> Self {
        let mut d = [0.0; 3];
        d[i] = 1.0;
        Self { v, d }
    }

    pub fn sqrt(self) -> Self {
        let s = self.v.sqrt();
        let k = 0.5 / s;
        Self { v: s, d: [self.d[0] * k, self.d[1] * k, self.d[2] * k] }
    }

    fn map(self, v: f64, k: f64) -> Self {
        Self { v, d: [self.d[0] * k, self.d[1] * k, self.d[2] * k] }
    }
}

impl Add for Dual3 {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Self { v: self.v + o.v, d: [self.d[0] + o.d[0], self.d[1] + o.d[1], self.d[2] + o.d[2]] }
    }
}

impl Sub for Dual3 {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        Self { v: self.v - o.v, d: [self.d[0] - o.d[0], self.d[1] - o.d[1], self.d[2] - o.d[2]] }
    }
}

impl Mul for Dual3 {
    type Output = Self;
    fn mul(self, o: Self) -> Self {
        Self {
            v: self.v * o.v,
            d: [
                self.d[0] * o.v + self.v * o.d[0],
                self.d[1] * o.v + self.v * o.d[1],
                self.d[2] * o.v + self.v * o.d[2],
            ],
        }
    }
}

impl Div for Dual3 {
    type Output = Self;
    fn div(self, o: Self) -> Self {
        let inv = 1.0 / o.v;
        let v = self.v * inv;
        Self { v, d: [(self.d[0] - v * o.d[0]) * inv, (self.d[1] - v * o.d[1]) * inv, (self.d[2] - v * o.d[2]) * inv] }
    }
}

impl Neg for Dual3 {
    type Output = Self;
    fn neg(self) -> Self {
        self.map(-self.v, -1.0)
    }
}

impl Add<f64> for Dual3 {
    type Output = Self;
    fn add(self, o: f64) -> Self {
        Self { v: self.v + o, d: self.d }
    }
}

impl Sub<f64> for Dual3 {
    type Output = Self;
    fn sub(self, o: f64) -> Self {
        Self { v: self.v - o, d: self.d }
    }
}

impl Mul<f64> for Dual3 {
    type Output = Self;
    fn mul(self, o: f64) -> Self {
        self.map(self.v * o, o)
    }
}

impl Mul<Dual3> for f64 {
    type Output = Dual3;
    fn mul(self, o: Dual3) -> Dual3 {
        o * self
    }
}
