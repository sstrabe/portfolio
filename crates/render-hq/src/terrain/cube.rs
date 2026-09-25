//! The cube-sphere: a sphere's directions as six square faces, the layout
//! every planet-scale map and terrain tile uses (`wgsl/cube.wgsl` is the
//! GPU twin; keep them in step).
//!
//! Face `f` looks along its normal `n`, with `right` and `up` spanning it
//! (`right × up = n`). A point `(u, v) ∈ [−1, 1]²` on a face maps to the
//! direction `normalize(n + tan(πu/4) right + tan(πv/4) up)`: the tangent
//! warp keeps texel areas within ~1.4 × of each other, against 5 × for a
//! plain cube.

use kerr::vec3::{self, V3};

/// Faces: +X, −X, +Y, −Y, +Z, −Z as (normal, right, up).
pub const FACES: [[V3; 3]; 6] = [
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
    [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
    [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
];

/// Unit direction of face `face` at `(u, v)` (u, v may lie a little
/// outside [−1, 1], for aprons: the warp continues smoothly).
pub fn direction(face: usize, u: f64, v: f64) -> V3 {
    let [n, r, up] = FACES[face];
    let (a, b) = ((std::f64::consts::FRAC_PI_4 * u).tan(), (std::f64::consts::FRAC_PI_4 * v).tan());
    vec3::normalize(vec3::add(n, vec3::add(vec3::scale(r, a), vec3::scale(up, b))))
}

/// The face a direction lies on and its `(u, v)` there.
pub fn face_uv(d: V3) -> (usize, f64, f64) {
    let face = (0..6).max_by(|&a, &b| vec3::dot(d, FACES[a][0]).total_cmp(&vec3::dot(d, FACES[b][0]))).unwrap_or(0);
    let [n, r, up] = FACES[face];
    let z = vec3::dot(d, n);
    let (a, b) = (vec3::dot(d, r) / z, vec3::dot(d, up) / z);
    (face, a.atan() / std::f64::consts::FRAC_PI_4, b.atan() / std::f64::consts::FRAC_PI_4)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn faces_are_right_handed() {
        for [n, r, up] in FACES {
            assert!(vec3::norm(vec3::sub(vec3::cross(r, up), n)) < 1e-12);
        }
    }

    #[test]
    fn round_trips_and_covers_the_sphere() {
        for face in 0..6 {
            for &(u, v) in &[(0.0, 0.0), (0.3, -0.7), (-0.99, 0.99), (1.0, 0.2)] {
                let d = direction(face, u, v);
                let (f, u2, v2) = face_uv(d);
                // On an edge either face may claim it; the direction agrees.
                assert!(vec3::norm(vec3::sub(direction(f, u2, v2), d)) < 1e-12, "{face} {u} {v}");
                if u.abs() < 1.0 && v.abs() < 1.0 {
                    assert!(f == face && (u2 - u).abs() < 1e-12 && (v2 - v).abs() < 1e-12);
                }
            }
        }
    }

    /// Crossing a face edge is continuous: stepping just past it lands next
    /// to where the neighbouring face starts.
    #[test]
    fn edges_are_continuous() {
        for face in 0..6 {
            for &(u, v) in &[(1.0, 0.3), (-1.0, -0.5), (0.2, 1.0), (-0.6, -1.0)] {
                let inside = direction(face, u * 0.999_999, v * 0.999_999);
                let outside = direction(face, u * 1.000_001, v * 1.000_001);
                assert!(vec3::norm(vec3::sub(inside, outside)) < 1e-5);
            }
        }
    }

    /// Texel areas vary by less than 1.5 × over a face.
    #[test]
    fn warp_keeps_areas_even() {
        let n = 64;
        let area = |i: usize, j: usize| {
            let uv = |k: usize| -1.0 + 2.0 * k as f64 / n as f64;
            let (a, b, c) =
                (direction(0, uv(i), uv(j)), direction(0, uv(i + 1), uv(j)), direction(0, uv(i), uv(j + 1)));
            vec3::norm(vec3::cross(vec3::sub(b, a), vec3::sub(c, a)))
        };
        let (mut lo, mut hi) = (f64::INFINITY, 0.0f64);
        for i in 0..n {
            for j in 0..n {
                let a = area(i, j);
                (lo, hi) = (lo.min(a), hi.max(a));
            }
        }
        assert!(hi / lo < 1.5, "{}", hi / lo);
    }
}
