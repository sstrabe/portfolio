//! Local inertial frames: measuring things near a moving body the way an
//! observer riding with it would.
//!
//! Simulation coordinates (Kerr–Schild, centred on the hole) are not a
//! body's rest frame. Near a star 1000 AU out, the metric makes coordinate
//! lengths differ from proper ones by `~M/r ≈ 10⁻⁵`, and the star's
//! motion (≈ 0.003c) adds Lorentz contraction and a shift of simultaneity.
//! Across a planet that is ~100 m of disagreement about where its surface
//! is. A [`LocalFrame`] is an orthonormal tetrad at one event: its time axis
//! is the body's 4-velocity and its space axes are the coordinate axes made
//! orthonormal to it (Gram–Schmidt under the metric there), so its
//! components are proper times and lengths and its axes stay parallel to
//! the simulation's to within `v/c`.

use crate::metric::Kerr;
use crate::pilot::{Tetrad, orthonormalize};
use crate::vec3::{V3, V4};

#[derive(Clone, Copy, Debug)]
pub struct LocalFrame {
    /// Where the metric is evaluated.
    pub at: V3,
    /// `e[0]` is the 4-velocity; `e[1..4]` the space axes.
    pub e: Tetrad,
}

impl LocalFrame {
    /// The rest frame of 4-velocity `u` at position `at`.
    pub fn new(k: &Kerr, at: V3, u: V4) -> Self {
        let mut e = [u, [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];
        orthonormalize(k, at, &mut e);
        Self { at, e }
    }

    /// Time and position in this frame of a coordinate displacement `d`.
    pub fn components(&self, k: &Kerr, d: V4) -> (f64, V3) {
        let dot = |a: V4| k.dot(self.at, a, d);
        (-dot(self.e[0]), [dot(self.e[1]), dot(self.e[2]), dot(self.e[3])])
    }

    /// The coordinate displacement of time `t` and position `x` in this
    /// frame.
    pub fn displacement(&self, t: f64, x: V3) -> V4 {
        std::array::from_fn(|m| t * self.e[0][m] + x[0] * self.e[1][m] + x[1] * self.e[2][m] + x[2] * self.e[3][m])
    }

    /// Velocity in this frame of something with 4-velocity `w`.
    pub fn velocity(&self, k: &Kerr, w: V4) -> V3 {
        let (g, p) = self.components(k, w);
        [p[0] / g, p[1] / g, p[2] / g]
    }

    /// 4-velocity of something moving at `v` in this frame.
    pub fn four_velocity(&self, v: V3) -> V4 {
        let gamma = 1.0 / (1.0 - (v[0] * v[0] + v[1] * v[1] + v[2] * v[2])).max(1e-300).sqrt();
        self.displacement(gamma, [gamma * v[0], gamma * v[1], gamma * v[2]])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vec3;

    /// Components and displacements invert each other, lengths are proper
    /// ones, and a frame's own 4-velocity is at rest in it.
    #[test]
    fn round_trips_and_proper_lengths() {
        let k = Kerr::new(1.0, 0.9);
        let at = [18000.0, -6000.0, 2500.0];
        let u = k.four_velocity(at, [0.004, 0.002, -0.001]).unwrap();
        let f = LocalFrame::new(&k, at, u);
        let d = f.displacement(0.3, [1e-3, -2e-3, 5e-4]);
        let (t, x) = f.components(&k, d);
        assert!((t - 0.3).abs() < 1e-15 && vec3::norm(vec3::sub(x, [1e-3, -2e-3, 5e-4])) < 1e-15);
        // A purely spatial displacement in the frame has the metric length
        // of its components.
        let s = f.displacement(0.0, [3e-3, 4e-3, 0.0]);
        assert!((k.dot(at, s, s).sqrt() - 5e-3).abs() < 1e-17);
        assert!(vec3::norm(f.velocity(&k, u)) < 1e-15);
        let w = f.four_velocity([0.1, -0.2, 0.05]);
        assert!(vec3::norm(vec3::sub(f.velocity(&k, w), [0.1, -0.2, 0.05])) < 1e-14);
        assert!((k.dot(at, w, w) + 1.0).abs() < 1e-12);
        // The space axes stay close to the coordinate axes (within ~v and
        // the metric's ~M/r).
        assert!(f.e[1][1] > 0.99 && f.e[2][2] > 0.99 && f.e[3][3] > 0.99);
    }
}
