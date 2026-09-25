//! Planetary terrain on the CPU side: the planets' body-fixed frame as the
//! shaders use it, and reading terrain heights back from the GPU.
//!
//! The terrain itself is defined once, in `wgsl/terrain.wgsl`. The CPU
//! asks the GPU for it ([`probe::Probe`]) rather than keeping a second copy
//! that could drift from what is drawn.

pub mod anchor;
pub mod atlas;
pub mod cube;
pub mod field;
pub mod maps;
pub mod probe;
pub mod rt;
pub mod tilegen;
pub mod tiles;

use kerr::vec3::{self, V3};

/// The body-fixed axes of a planet with spin axis `spin` rotated by
/// `angle`, as `planet_body` in `near.wgsl` builds them: `e1` starts at
/// `any_orthogonal(spin)` and turns with the planet; `e2 = spin × e1`.
pub fn body_axes(spin: V3, angle: f64) -> [V3; 3] {
    let s = vec3::normalize(spin);
    let a0 = vec3::any_orthogonal(s);
    let (sin, cos) = angle.sin_cos();
    let e1 = vec3::add(vec3::scale(a0, cos), vec3::scale(vec3::cross(s, a0), sin));
    [e1, vec3::cross(s, e1), s]
}

/// Body-fixed coordinates of planet-frame vector `v` (the shaders' `q`).
pub fn to_body(axes: &[V3; 3], v: V3) -> V3 {
    axes.map(|a| vec3::dot(v, a))
}

/// Planet-frame vector of body-fixed coordinates `q`.
pub fn from_body(axes: &[V3; 3], q: V3) -> V3 {
    vec3::add(vec3::add(vec3::scale(axes[0], q[0]), vec3::scale(axes[1], q[1])), vec3::scale(axes[2], q[2]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn body_axes_are_a_rotating_frame() {
        let spin = vec3::normalize([0.2, -0.3, 0.93]);
        let a = body_axes(spin, 0.7);
        for i in 0..3 {
            assert!((vec3::norm(a[i]) - 1.0).abs() < 1e-12);
            assert!(vec3::dot(a[i], a[(i + 1) % 3]).abs() < 1e-12);
        }
        assert!(vec3::norm(vec3::sub(vec3::cross(a[0], a[1]), a[2])) < 1e-12);
        let v = [0.3, 0.5, -0.8];
        assert!(vec3::norm(vec3::sub(from_body(&a, to_body(&a, v)), v)) < 1e-12);
        // A quarter turn later the ground has moved: e1 went where e2 was.
        let b = body_axes(spin, 0.7 + std::f64::consts::FRAC_PI_2);
        assert!(vec3::norm(vec3::sub(b[0], a[1])) < 1e-12);
    }
}
