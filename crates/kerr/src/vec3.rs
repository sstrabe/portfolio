//! Tiny fixed-size vector helpers. Deliberately dependency free.

pub type V3 = [f64; 3];
pub type V4 = [f64; 4];

#[inline]
pub fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
pub fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
pub fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[inline]
pub fn axpy(a: V3, s: f64, b: V3) -> V3 {
    [a[0] + s * b[0], a[1] + s * b[1], a[2] + s * b[2]]
}

#[inline]
pub fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
pub fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

#[inline]
pub fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}

#[inline]
pub fn normalize(a: V3) -> V3 {
    let n = norm(a);
    if n > 0.0 { scale(a, 1.0 / n) } else { a }
}

/// Any unit vector orthogonal to `a` (which must be non-zero).
pub fn any_orthogonal(a: V3) -> V3 {
    let helper = if a[0].abs() < 0.8 * norm(a) { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
    normalize(cross(a, helper))
}

/// Rotate `v` about unit `axis` by `angle` (Rodrigues).
pub fn rotate(v: V3, axis: V3, angle: f64) -> V3 {
    let (s, c) = angle.sin_cos();
    let k = axis;
    add(add(scale(v, c), scale(cross(k, v), s)), scale(k, dot(k, v) * (1.0 - c)))
}

#[inline]
pub fn spatial(x: V4) -> V3 {
    [x[1], x[2], x[3]]
}
