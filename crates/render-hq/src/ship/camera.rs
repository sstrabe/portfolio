//! First person or a third-person chase camera.
//!
//! The chase camera rides along with the ship (it shares the ship's
//! 4-velocity), so the ship shows no aberration or Doppler shift and plain
//! perspective from the camera's position is exact for it. For everything
//! else the camera's offset of tens of metres is negligible (the nearest
//! thing, a planet's surface, is hundreds of km away), so the rest of the
//! trace keeps using the pilot's event and only takes the camera's
//! orientation: the view is rendered with the tetrad rotated onto the
//! camera's axes (see [`ChaseCamera::view_tetrad`]).

use kerr::pilot::Tetrad;
use kerr::vec3::{self, V3};

/// Where the camera looks from, in ship-frame metres, and its axes
/// (forward, left, up) as ship-frame unit vectors.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pose {
    pub pos: V3,
    pub axes: [V3; 3],
}

#[derive(Clone, Copy, Debug)]
pub struct ChaseCamera {
    /// Third person (the ship is drawn) or first person (it isn't).
    pub chase: bool,
    /// Orbit angles (rad): yaw to the left of straight behind, pitch
    /// above the ship's horizontal plane.
    pub yaw: f64,
    pub pitch: f64,
    /// Distance from the orbit centre, m.
    pub distance: f64,
}

/// Orbit centre (ship frame, m): the middle of the hull, a little above
/// the axis.
pub const TARGET: V3 = [-1.0, 0.0, 1.5];
/// The view is tilted up by this much from looking straight at the orbit
/// centre, so the ship sits in the lower part of the frame and the scene
/// ahead stays in view.
const LIFT: f64 = 0.16;
/// A three-quarter view from behind and to the left: the hull, the
/// radiators and the drive all show.
pub const DEFAULT_YAW: f64 = 0.6;
pub const DEFAULT_PITCH: f64 = 0.25;
pub const DEFAULT_DISTANCE: f64 = 82.0;
pub const MIN_DISTANCE: f64 = 36.0;
pub const MAX_DISTANCE: f64 = 400.0;

impl Default for ChaseCamera {
    fn default() -> Self {
        Self { chase: false, yaw: DEFAULT_YAW, pitch: DEFAULT_PITCH, distance: DEFAULT_DISTANCE }
    }
}

impl ChaseCamera {
    /// Turn the orbit by mouse-like deltas (rad), keeping the camera off
    /// the poles.
    pub fn orbit(&mut self, d_yaw: f64, d_pitch: f64) {
        self.yaw = (self.yaw + d_yaw).rem_euclid(std::f64::consts::TAU);
        self.pitch = (self.pitch + d_pitch).clamp(-1.45, 1.45);
    }

    /// Scale the distance (e.g. by a scroll wheel).
    pub fn zoom(&mut self, factor: f64) {
        self.distance = (self.distance * factor).clamp(MIN_DISTANCE, MAX_DISTANCE);
    }

    pub fn reset(&mut self) {
        (self.yaw, self.pitch, self.distance) = (DEFAULT_YAW, DEFAULT_PITCH, DEFAULT_DISTANCE);
    }

    pub fn pose(&self) -> Pose {
        if !self.chase {
            return Pose { pos: [0.0; 3], axes: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] };
        }
        let look = look_axes(self.yaw, self.pitch);
        let pos = vec3::axpy(TARGET, -self.distance, look[0]);
        Pose { pos, axes: look_axes(self.yaw, self.pitch - LIFT) }
    }

    /// The pilot's tetrad turned onto the camera's axes: e₀ is kept and
    /// each spatial axis becomes the ship-frame combination the camera
    /// axis is made of, so it stays orthonormal.
    pub fn view_tetrad(&self, e: &Tetrad) -> Tetrad {
        let axes = self.pose().axes;
        let mut out = *e;
        for (i, a) in axes.iter().enumerate() {
            out[i + 1] = std::array::from_fn(|m| a[0] * e[1][m] + a[1] * e[2][m] + a[2] * e[3][m]);
        }
        out
    }
}

/// Camera axes looking along (yaw, −pitch): forward, left, up.
fn look_axes(yaw: f64, pitch: f64) -> [V3; 3] {
    let (sy, cy) = yaw.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    let fwd = [cp * cy, cp * sy, -sp];
    let up = [sp * cy, sp * sy, cp];
    [fwd, vec3::cross(up, fwd), up]
}

/// TAA's reprojection of ship pixels, from a direction `n` in the current
/// camera axes to the previous frame's (up to scale, as rows applied to
/// `n`). Row a is the previous camera axis a in the current axes, plus
/// the parallax of the camera's move for points on the plane through the
/// orbit centre facing the camera: a homography, exact at the centre and
/// close across the hull. Rotation alone, as before, misplaced the history
/// by the orbit angle whenever the chase camera was orbited, so it was
/// rejected and the hull's edges flickered.
pub fn reprojection(prev: &Pose, cur: &Pose) -> [[f32; 4]; 4] {
    // A point X = c + s d on the current ray d = Σ n_b axis_b lies on the
    // plane when s = depth / n_forward; seen from before it's along
    // A_prev (X − c_prev) ∝ A_prev d + (n_forward / depth) A_prev (c − c_prev).
    let depth = vec3::dot(vec3::sub(TARGET, cur.pos), cur.axes[0]).max(1.0);
    let moved = vec3::sub(cur.pos, prev.pos);
    let mut m = [[0.0f32; 4]; 4];
    for (a, row) in m.iter_mut().take(3).enumerate() {
        for (b, v) in row.iter_mut().take(3).enumerate() {
            *v = vec3::dot(prev.axes[a], cur.axes[b]) as f32;
        }
        row[0] += (vec3::dot(prev.axes[a], moved) / depth) as f32;
    }
    m
}

/// Components in the ship frame of a vector given in camera axes.
pub fn to_ship(axes: &[V3; 3], v: V3) -> V3 {
    std::array::from_fn(|m| v[0] * axes[0][m] + v[1] * axes[1][m] + v[2] * axes[2][m])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axes_are_orthonormal_and_right_handed() {
        for (y, p) in [(0.0, 0.0), (0.7, 0.3), (-2.0, -1.2), (3.0, 1.4)] {
            let a = look_axes(y, p);
            for i in 0..3 {
                assert!((vec3::norm(a[i]) - 1.0).abs() < 1e-12);
                assert!(vec3::dot(a[i], a[(i + 1) % 3]).abs() < 1e-12);
            }
            let c = vec3::cross(a[0], a[1]);
            assert!(vec3::norm(vec3::sub(c, a[2])) < 1e-12);
        }
    }

    /// Orbiting the chase camera: the reprojection takes the orbit centre's
    /// direction now to its direction on the previous frame exactly.
    #[test]
    fn reprojection_follows_the_orbit() {
        let prev = ChaseCamera { chase: true, ..Default::default() };
        let mut cur = prev;
        cur.orbit(0.05, -0.03);
        cur.zoom(1.1);
        let (p, c) = (prev.pose(), cur.pose());
        let m = reprojection(&p, &c);
        let in_axes = |pose: &Pose, x: V3| {
            let d = vec3::sub(x, pose.pos);
            vec3::normalize(std::array::from_fn(|a| vec3::dot(pose.axes[a], d)))
        };
        let check = |x: V3, tol: f64| {
            let n = in_axes(&c, x);
            let np: V3 = std::array::from_fn(|a| (0..3).map(|b| m[a][b] as f64 * n[b]).sum());
            let err = vec3::norm(vec3::sub(vec3::normalize(np), in_axes(&p, x)));
            assert!(err < tol, "{x:?}: {err}");
        };
        check(TARGET, 1e-6);
        // Hull points off the plane: within a fraction of the camera's turn.
        check([12.0, 0.0, 1.0], 0.2 * 0.06);
        check([-15.0, 3.0, 0.0], 0.2 * 0.06);
    }

    /// The default chase view is behind and above the ship, looking
    /// forward and down at it.
    #[test]
    fn chase_pose() {
        let cam = ChaseCamera { chase: true, ..Default::default() };
        let p = cam.pose();
        assert!(p.pos[0] < -40.0 && p.pos[2] > 10.0, "{:?}", p.pos);
        assert!(p.axes[0][0] > 0.7 && p.axes[0][2] < 0.0);
        let to_ship = vec3::normalize(vec3::sub(TARGET, p.pos));
        // The ship's centre is below the middle of the view.
        assert!(vec3::dot(to_ship, p.axes[2]) < -0.1);
    }
}
