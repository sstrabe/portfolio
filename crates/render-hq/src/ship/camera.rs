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
//!
//! Near a planet the chase camera keeps the horizon level: down is where
//! gravity pulls, the orbit turns about the local vertical, and the camera
//! holds still in the planet's frame while the ship turns in view (as in
//! KSP). Elsewhere it turns with the ship.

use kerr::pilot::Tetrad;
use kerr::vec3::{self, V3};
use kerr::world::Vertical;

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
    /// The direction the camera looks at the orbit centre from, as axes
    /// (forward, left, up); it sits behind the centre along −forward. In
    /// the ship frame, turned about its own axes (a trackball, so every
    /// direction is reachable and there are no poles to flip over); near a
    /// planet in the planet's frame, level (see [`Level`]).
    pub orbit: [V3; 3],
    /// Distance from the orbit centre, m.
    pub distance: f64,
    level: Option<Level>,
    /// Whether [`ChaseCamera::follow`] has run: the first levelling is
    /// immediate, later ones ease the roll out.
    followed: bool,
}

/// The planet frame a level orbit is held in.
#[derive(Clone, Copy, Debug)]
struct Level {
    vertical: Vertical,
    /// Roll (rad) about the view axis still to ease out after levelling.
    roll: f64,
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
/// Time constant of the roll easing out when the camera levels, s.
const ROLL_EASE_S: f64 = 0.4;

impl Default for ChaseCamera {
    fn default() -> Self {
        Self {
            chase: false,
            orbit: look_axes(DEFAULT_YAW, DEFAULT_PITCH),
            distance: DEFAULT_DISTANCE,
            level: None,
            followed: false,
        }
    }
}

impl ChaseCamera {
    /// Turn the orbit by mouse-like deltas (rad): yaw about the camera's
    /// own up, or near a planet about the local vertical (positive turns it
    /// left), pitch about its own left (positive looks further down). No
    /// limits: it goes over the top or underneath and on round.
    pub fn orbit(&mut self, d_yaw: f64, d_pitch: f64) {
        let [f, l, u] = self.orbit;
        let axis = self.level.map_or(u, |lv| lv.vertical.up);
        let (f, l) = (vec3::rotate(f, axis, d_yaw), vec3::rotate(l, axis, d_yaw));
        let f = vec3::rotate(f, l, d_pitch);
        // Keep the axes orthonormal as small errors add up.
        let f = vec3::normalize(f);
        let l = vec3::normalize(vec3::axpy(l, -vec3::dot(l, f), f));
        self.orbit = [f, l, vec3::cross(f, l)];
    }

    /// Scale the distance (e.g. by a scroll wheel).
    pub fn zoom(&mut self, factor: f64) {
        self.distance = (self.distance * factor).clamp(MIN_DISTANCE, MAX_DISTANCE);
    }

    pub fn reset(&mut self) {
        self.distance = DEFAULT_DISTANCE;
        match &mut self.level {
            // Behind the ship's heading on the horizon (its nose flattened
            // onto it, or its top when the nose points up or down).
            Some(lv) => {
                let (up, axes) = (lv.vertical.up, lv.vertical.axes);
                let flat = |v: V3| {
                    let v = from_ship(&axes, v);
                    vec3::axpy(v, -vec3::dot(v, up), up)
                };
                let nose = flat([1.0, 0.0, 0.0]);
                let h = vec3::normalize(if vec3::norm(nose) > 0.3 { nose } else { flat([0.0, 0.0, 1.0]) });
                let basis = [h, vec3::cross(up, h), up];
                self.orbit = look_axes(DEFAULT_YAW, DEFAULT_PITCH).map(|a| combine(&basis, a));
                lv.roll = 0.0;
            }
            None => self.orbit = look_axes(DEFAULT_YAW, DEFAULT_PITCH),
        }
    }

    /// Follow the local vertical each frame (`None` away from planets):
    /// level the orbit on arriving near a planet, carry it round as up
    /// turns while the ship moves over the planet, and hand it back to the
    /// ship frame on leaving.
    pub fn follow(&mut self, vertical: Option<Vertical>, dt: f64) {
        let first = !self.followed;
        self.followed = true;
        match (self.level, vertical) {
            (Some(lv), Some(v)) if lv.vertical.planet == v.planet => {
                // The turn taking the old up to the new one: parallel
                // transport, so the view keeps its place over the ground.
                let (a, b) = (lv.vertical.up, v.up);
                let axis = vec3::cross(a, b);
                let s = vec3::norm(axis);
                if s > 1e-15 {
                    let (axis, angle) = (vec3::scale(axis, 1.0 / s), s.atan2(vec3::dot(a, b)));
                    self.orbit = self.orbit.map(|x| vec3::rotate(x, axis, angle));
                }
                self.orbit = levelled(self.orbit, v.up, false).0;
                let roll = lv.roll * (-dt / ROLL_EASE_S).exp();
                self.level = Some(Level { vertical: v, roll: if roll.abs() < 1e-4 { 0.0 } else { roll } });
            }
            (_, Some(v)) => {
                let ship = self.ship_orbit();
                let (orbit, roll) = levelled(ship.map(|a| from_ship(&v.axes, a)), v.up, true);
                self.orbit = orbit;
                self.level = Some(Level { vertical: v, roll: if first { 0.0 } else { roll } });
            }
            (Some(_), None) => {
                self.orbit = self.ship_orbit();
                self.level = None;
            }
            (None, None) => {}
        }
    }

    /// The orbit's axes in the ship frame, with any roll still easing out.
    fn ship_orbit(&self) -> [V3; 3] {
        let Some(lv) = self.level else { return self.orbit };
        let [f, l, u] = self.orbit.map(|a| combine(&lv.vertical.axes, a));
        [f, vec3::rotate(l, f, lv.roll), vec3::rotate(u, f, lv.roll)]
    }

    pub fn pose(&self) -> Pose {
        if !self.chase {
            return Pose { pos: [0.0; 3], axes: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] };
        }
        let [f, l, u] = self.ship_orbit();
        let pos = vec3::axpy(TARGET, -self.distance, f);
        // Tilted up by LIFT about the camera's left.
        Pose { pos, axes: [vec3::rotate(f, l, -LIFT), l, vec3::rotate(u, l, -LIFT)] }
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

/// `axes` (forward, left, up) rolled about forward so left is horizontal
/// (perpendicular to `up`), with the roll (rad) that turns the levelled
/// axes back to the given ones. `upright`: the view's up on up's side;
/// else the smaller roll (so a view turned over the top stays so).
fn levelled(axes: [V3; 3], up: V3, upright: bool) -> ([V3; 3], f64) {
    let [f, l, _] = axes;
    let side = vec3::cross(up, f);
    // Looking straight up or down: keep left, flattened.
    let l_new = if vec3::norm(side) > 1e-6 {
        let side = vec3::normalize(side);
        if upright || vec3::dot(side, l) >= 0.0 { side } else { vec3::scale(side, -1.0) }
    } else {
        vec3::normalize(vec3::axpy(l, -vec3::dot(l, up), up))
    };
    let f = vec3::normalize(vec3::axpy(f, -vec3::dot(f, l_new), l_new));
    let u_new = vec3::cross(f, l_new);
    let roll = vec3::dot(l, u_new).atan2(vec3::dot(l, l_new));
    ([f, l_new, u_new], roll)
}

/// Σ v_j axes_j: a vector given in `axes` in the frame the axes are
/// written in.
fn combine(axes: &[V3; 3], v: V3) -> V3 {
    std::array::from_fn(|m| v[0] * axes[0][m] + v[1] * axes[1][m] + v[2] * axes[2][m])
}

/// A ship-frame vector in orthonormal `axes` (given in the ship frame).
fn from_ship(axes: &[V3; 3], v: V3) -> V3 {
    axes.map(|a| vec3::dot(a, v))
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

    /// The orbit goes over the top and on round with no limit or flip: a
    /// full turn in pitch (or yaw) comes back to the start, and the axes
    /// stay orthonormal and right-handed all the way.
    #[test]
    fn orbit_has_no_limits() {
        let mut cam = ChaseCamera { chase: true, ..Default::default() };
        let start = cam.orbit;
        let steps = 400;
        let mut prev = start;
        for k in 0..steps {
            cam.orbit(0.0, std::f64::consts::TAU / steps as f64);
            let a = cam.orbit;
            for i in 0..3 {
                assert!((vec3::norm(a[i]) - 1.0).abs() < 1e-9 && vec3::dot(a[i], a[(i + 1) % 3]).abs() < 1e-9, "{k}");
            }
            assert!(vec3::norm(vec3::sub(vec3::cross(a[0], a[1]), a[2])) < 1e-9);
            // Each step turns the view by the step alone: no jumps.
            for (now, before) in a.iter().zip(&prev) {
                assert!(vec3::norm(vec3::sub(*now, *before)) < 0.02, "{k}");
            }
            prev = a;
        }
        for (now, then) in cam.orbit.iter().zip(&start) {
            assert!(vec3::norm(vec3::sub(*now, *then)) < 1e-6);
        }
        for _ in 0..steps {
            cam.orbit(std::f64::consts::TAU / steps as f64, 0.0);
        }
        assert!(vec3::norm(vec3::sub(cam.orbit[0], start[0])) < 1e-6);
    }

    /// A ship turned by `angle` about `axis` (ship-frame components of the
    /// planet frame's axes) near a planet, at `up` (planet frame).
    fn vertical(axis: V3, angle: f64, up: V3) -> Vertical {
        let axis = vec3::normalize(axis);
        let axes = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]].map(|a| vec3::rotate(a, axis, angle));
        let planet = kerr::local::PlanetRef { star: 3, generation: 0, planet: 1 };
        Vertical { planet, axes, up: vec3::normalize(up) }
    }

    /// The pose's axes in the planet frame.
    fn in_planet(cam: &ChaseCamera, v: &Vertical) -> [V3; 3] {
        cam.pose().axes.map(|a| from_ship(&v.axes, a))
    }

    fn close(a: [V3; 3], b: [V3; 3], tol: f64) -> bool {
        a.iter().zip(&b).all(|(x, y)| vec3::norm(vec3::sub(*x, *y)) < tol)
    }

    /// Near a planet the view is level (its left is horizontal, its up is
    /// on the sky's side), orbiting keeps it level, and it stays put in
    /// the planet's frame while the ship turns.
    #[test]
    fn level_near_a_planet() {
        let mut cam = ChaseCamera { chase: true, ..Default::default() };
        let up = [0.3, -0.5, 0.8];
        let v = vertical([1.0, 2.0, -0.5], 0.9, up);
        cam.follow(Some(v), 1.0 / 60.0);
        let level = |cam: &ChaseCamera, v: &Vertical| {
            let [_, l, u] = in_planet(cam, v);
            vec3::dot(l, v.up).abs() < 1e-9 && vec3::dot(u, v.up) > 0.0
        };
        assert!(level(&cam, &v));
        cam.orbit(0.7, -0.4);
        assert!(level(&cam, &v));
        let before = in_planet(&cam, &v);
        let turned = vertical([-0.2, 1.0, 0.4], 2.1, up);
        cam.follow(Some(turned), 1.0 / 60.0);
        assert!(close(in_planet(&cam, &turned), before, 1e-9));
    }

    /// Moving over the planet carries the view round with the vertical;
    /// a view turned over the top stays so (no flip back).
    #[test]
    fn carried_round_with_the_vertical() {
        let mut cam = ChaseCamera { chase: true, ..Default::default() };
        let mut v = vertical([0.0, 0.0, 1.0], 0.3, [0.0, 0.0, 1.0]);
        cam.follow(Some(v), 1.0 / 60.0);
        cam.orbit(0.0, 1.2 + std::f64::consts::FRAC_PI_2);
        let over = |cam: &ChaseCamera, v: &Vertical| vec3::dot(in_planet(cam, v)[2], v.up) < 0.0;
        assert!(over(&cam, &v));
        let mut prev = in_planet(&cam, &v);
        for k in 1..=200 {
            let a = k as f64 * 0.01;
            v.up = [a.sin(), 0.0, a.cos()];
            cam.follow(Some(v), 1.0 / 60.0);
            let now = in_planet(&cam, &v);
            assert!(close(now, prev, 0.012), "{k}");
            assert!(over(&cam, &v) && vec3::dot(now[1], v.up).abs() < 1e-9, "{k}");
            prev = now;
        }
    }

    /// Arriving near a planet levels the view without a jump: the roll
    /// eases out over a second or so.
    #[test]
    fn levelling_eases_in() {
        let mut cam = ChaseCamera { chase: true, ..Default::default() };
        cam.follow(None, 1.0 / 60.0);
        cam.orbit(0.3, 0.2);
        let v = vertical([1.0, 0.2, 0.1], 0.8, [0.1, 0.9, 0.3]);
        let before = cam.pose().axes;
        cam.follow(Some(v), 1.0 / 60.0);
        assert!(close(cam.pose().axes, before, 1e-9));
        for _ in 0..300 {
            cam.follow(Some(v), 1.0 / 60.0);
        }
        let [_, l, u] = in_planet(&cam, &v);
        assert!(vec3::dot(l, v.up).abs() < 1e-9 && vec3::dot(u, v.up) > 0.0);
        // Leaving hands the orbit back to the ship frame where it was.
        let at = cam.pose().axes;
        cam.follow(None, 1.0 / 60.0);
        assert!(close(cam.pose().axes, at, 1e-9));
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
