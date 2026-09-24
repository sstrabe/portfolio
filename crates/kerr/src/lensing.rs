//! Seeing along the past light cone.
//!
//! A pixel looking in ship-frame direction `n` receives a photon whose
//! 4-momentum is proportional to `u + d`, `d = −n`. We integrate the
//! past-directed null geodesic with initial momentum `P = −u + n` (so
//! `−P·u = 1` fixes the observed frequency to one) backward into the past.
//!
//! * Redshift of anything the ray meets with 4-velocity `w`:
//!   `g = ν_obs / ν_emit = 1 / (P·w)`; for the distant sky `g = 1 / P_t`.
//! * An *image* of a body is a ray that intersects the body's worldline
//!   (history lookup at the ray's own coordinate time `t`, so light travel
//!   time, Shapiro delay and the body's motion are all included). Images are
//!   found with a Newton iteration on the 2-D miss vector at closest
//!   approach; the ray's swept angle around the hole selects the image
//!   order (0: direct, 1: around the other side).
//! * The Jacobian of the miss vector with respect to the viewing angles
//!   gives the lensing magnification: a point source's flux scales as
//!   `g⁴ / |J₁ × J₂|` (flat space: `|J₁ × J₂| = D²`).

use crate::history::Sample;
use crate::metric::Kerr;
use crate::pilot::{FORWARD, LEFT, Tetrad, UP};
use crate::vec3::{self, V3, V4};

#[derive(Clone, Copy, Debug)]
pub struct Observer {
    pub x: V4,
    pub e: Tetrad,
}

impl Observer {
    pub fn position(&self) -> V3 {
        vec3::spatial(self.x)
    }

    /// Ship-frame direction (forward, left, up) of a coordinate direction
    /// `d` treated as the spatial part of a past-pointing null vector.
    pub fn local_direction(&self, k: &Kerr, d: V3) -> V3 {
        let d = vec3::normalize(d);
        let p = self.position();
        let v = [-1.0, d[0], d[1], d[2]];
        let n = [k.dot(p, v, self.e[FORWARD]), k.dot(p, v, self.e[LEFT]), k.dot(p, v, self.e[UP])];
        vec3::normalize(n)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RayConfig {
    /// Spatial step as a fraction of the current radius.
    pub step: f64,
    pub min_step: f64,
    pub max_step: f64,
    pub max_steps: usize,
    /// Rays that get within this coordinate distance of `r₊` are captured.
    pub horizon_eps: f64,
    pub escape_radius: f64,
}

impl Default for RayConfig {
    fn default() -> Self {
        Self { step: 0.04, min_step: 0.01, max_step: 12.0, max_steps: 4000, horizon_eps: 0.03, escape_radius: 3000.0 }
    }
}

/// Initial position and past-directed covariant momentum of the ray
/// seen in ship-frame direction `n` (unit).
pub fn backward_ray(k: &Kerr, obs: &Observer, n: V3) -> (V4, V4) {
    let e = &obs.e;
    let mut p_up = [0.0; 4];
    for mu in 0..4 {
        p_up[mu] = -e[0][mu] + n[0] * e[FORWARD][mu] + n[1] * e[LEFT][mu] + n[2] * e[UP][mu];
    }
    (obs.x, k.lower(obs.position(), p_up))
}

#[derive(Clone, Copy, Debug)]
struct RayState {
    x: V4,
    p: V4,
}

fn ray_rhs(k: &Kerr, s: &RayState) -> RayState {
    let fl = crate::geodesic::flow(k, vec3::spatial(s.x), s.p);
    RayState { x: fl.dx, p: fl.dp }
}

fn ray_step(k: &Kerr, s: &RayState, h: f64) -> RayState {
    let add = |a: &RayState, c: f64, b: &RayState| {
        let mut o = *a;
        for i in 0..4 {
            o.x[i] += c * b.x[i];
            o.p[i] += c * b.p[i];
        }
        o
    };
    let k1 = ray_rhs(k, s);
    let k2 = ray_rhs(k, &add(s, 0.5 * h, &k1));
    let k3 = ray_rhs(k, &add(s, 0.5 * h, &k2));
    let k4 = ray_rhs(k, &add(s, h, &k3));
    let mut o = *s;
    for i in 0..4 {
        o.x[i] += h / 6.0 * (k1.x[i] + 2.0 * k2.x[i] + 2.0 * k3.x[i] + k4.x[i]);
        o.p[i] += h / 6.0 * (k1.p[i] + 2.0 * k2.p[i] + 2.0 * k3.p[i] + k4.p[i]);
    }
    o
}

/// Where a ray ends up.
#[derive(Clone, Copy, Debug)]
pub enum RayEnd {
    /// Reached the distant sky, arriving from coordinate direction `dir`
    /// with redshift `g` relative to a static source at infinity.
    Sky {
        dir: V3,
        g: f64,
    },
    Horizon,
    Exhausted,
}

/// Walks a backward ray, calling `visit(prev, next, psi_prev, psi_next)`
/// for every step; `visit` returns `false` to stop early.
fn march(k: &Kerr, x0: V4, p0: V4, cfg: &RayConfig, mut visit: impl FnMut(&V4, &V4, &V4, f64, f64) -> bool) -> RayEnd {
    let r_plus = k.r_plus();
    let mut s = RayState { x: x0, p: p0 };
    let mut psi = 0.0;
    for _ in 0..cfg.max_steps {
        let pos = vec3::spatial(s.x);
        let r = k.radius(pos);
        if r - r_plus < cfg.horizon_eps {
            return RayEnd::Horizon;
        }
        let d = ray_rhs(k, &s);
        let speed = vec3::norm(vec3::spatial(d.x)).max(1e-12);
        let dist = (cfg.step * (r - 0.8 * r_plus).max(0.2 * r_plus)).clamp(cfg.min_step, cfg.max_step);
        let next = ray_step(k, &s, dist / speed);
        let npos = vec3::spatial(next.x);
        let dpsi = vec3::norm(vec3::cross(pos, npos)).atan2(vec3::dot(pos, npos));
        if !visit(&s.x, &next.x, &next.p, psi, psi + dpsi) {
            return RayEnd::Exhausted;
        }
        psi += dpsi;
        s = next;
        let nr = k.radius(npos);
        if nr > cfg.escape_radius && vec3::dot(npos, vec3::spatial(d.x)) > 0.0 {
            let dir = vec3::normalize(vec3::spatial(ray_rhs(k, &s).x));
            return RayEnd::Sky { dir, g: 1.0 / s.p[0] };
        }
    }
    RayEnd::Exhausted
}

pub fn trace_to_sky(k: &Kerr, obs: &Observer, n: V3, cfg: &RayConfig) -> RayEnd {
    let (x, p) = backward_ray(k, obs, n);
    march(k, x, p, cfg, |_, _, _, _, _| true)
}

/// Closest approach of a backward ray to a body's worldline.
#[derive(Clone, Copy, Debug)]
pub struct Approach {
    /// Ray position minus body position at the moment of closest approach.
    pub miss: V3,
    pub t: f64,
    pub pos: V3,
    pub body: Sample,
    /// Past-directed photon momentum there (observer-normalized).
    pub p: V4,
    /// Coordinate path length from the observer.
    pub path: f64,
}

/// Search a backward ray for its closest approach to `target(t)`, only
/// considering the part of the ray whose swept angle lies in `window`.
pub fn closest_approach(
    k: &Kerr,
    obs: &Observer,
    n: V3,
    target: &dyn Fn(f64) -> Option<Sample>,
    window: (f64, f64),
    t_min: f64,
    cfg: &RayConfig,
) -> Option<Approach> {
    let (x0, p0) = backward_ray(k, obs, n);
    let mut best: Option<(f64, Approach)> = None;
    let mut path = 0.0;
    let mut prev: Option<(V3, Sample)> = None;
    let mut p_prev = p0;
    march(k, x0, p0, cfg, |a, b, pb, psi0, psi1| {
        let seg = vec3::norm(vec3::sub(vec3::spatial(*b), vec3::spatial(*a)));
        let in_window = psi1 >= window.0 && psi0 <= window.1;
        let result = (|| {
            if b[0] < t_min {
                return false;
            }
            if psi0 > window.1 {
                return false;
            }
            if !in_window {
                prev = None;
                return true;
            }
            let Some(sb) = target(b[0]) else {
                prev = None;
                return true;
            };
            let db = vec3::sub(vec3::spatial(*b), sb.pos);
            let (da, sa) = match prev {
                Some(v) => v,
                None => match target(a[0]) {
                    Some(sa) => (vec3::sub(vec3::spatial(*a), sa.pos), sa),
                    None => {
                        prev = Some((db, sb));
                        return true;
                    }
                },
            };
            let delta = vec3::sub(db, da);
            let dd = vec3::dot(delta, delta);
            let s = if dd > 0.0 { (-vec3::dot(da, delta) / dd).clamp(0.0, 1.0) } else { 0.0 };
            let miss = vec3::axpy(da, s, delta);
            let dist = vec3::norm(miss);
            if best.as_ref().is_none_or(|(bd, _)| dist < *bd) {
                let lerp = |u: f64, v: f64| u + s * (v - u);
                let mut p = [0.0; 4];
                let mut pos = [0.0; 3];
                let mut body = Sample::default();
                for i in 0..4 {
                    p[i] = lerp(p_prev[i], pb[i]);
                }
                for i in 0..3 {
                    pos[i] = lerp(a[i + 1], b[i + 1]);
                    body.pos[i] = lerp(sa.pos[i], sb.pos[i]);
                    body.vel[i] = lerp(sa.vel[i], sb.vel[i]);
                }
                best = Some((dist, Approach { miss, t: lerp(a[0], b[0]), pos, body, p, path: path + s * seg }));
            }
            prev = Some((db, sb));
            // Once well past the closest approach the ray is only moving away.
            !best.as_ref().is_some_and(|(bd, _)| vec3::norm(db) > 3.0 * bd + 20.0 && psi1 > window.0 + 0.2)
        })();
        path += seg;
        p_prev = *pb;
        result
    });
    best.map(|(_, a)| a)
}

/// A lensed image of a body.
#[derive(Clone, Copy, Debug)]
pub struct Image {
    /// Ship-frame direction (forward, left, up).
    pub dir: V3,
    /// Frequency ratio `ν_obs / ν_emit`.
    pub g: f64,
    /// `|J₁ × J₂|`: transverse area at the source per steradian at the
    /// observer. Point-source flux ∝ `g⁴ / area`; angular size ≈ `R / √area`.
    pub area: f64,
    /// Emission event and the body's state then.
    pub t_emit: f64,
    pub body: Sample,
    pub path: f64,
    pub residual: f64,
}

/// Swept-angle window for image order `k`.
pub fn order_window(order: u8) -> (f64, f64) {
    let pi = std::f64::consts::PI;
    match order {
        0 => (0.0, pi),
        _ => (pi, 2.0 * pi),
    }
}

/// Initial guess for an image of `order`, from the body's present position.
pub fn image_guess(k: &Kerr, obs: &Observer, body_pos: V3, order: u8) -> V3 {
    let op = obs.position();
    let direct = obs.local_direction(k, vec3::sub(body_pos, op));
    if order == 0 {
        return direct;
    }
    let to_hole = obs.local_direction(k, vec3::scale(op, -1.0));
    let r_obs = vec3::norm(op);
    let r_src = vec3::norm(body_pos);
    let shadow = (5.3 * k.m / r_obs).min(1.0).asin();
    let einstein = (4.0 * k.m * r_src / (r_obs * (r_obs + r_src))).sqrt();
    let alpha = (1.08 * shadow).max(einstein);
    // Rotate from the hole direction away from the direct image.
    let axis = vec3::cross(to_hole, direct);
    let axis = if vec3::norm(axis) < 1e-9 { vec3::any_orthogonal(to_hole) } else { vec3::normalize(axis) };
    vec3::rotate(to_hole, axis, -alpha)
}

#[derive(Clone, Copy, Debug)]
pub struct ImageSearch {
    pub max_iter: usize,
    /// Accept when the miss distance is below `tol_abs + tol_rel · path`.
    pub tol_abs: f64,
    pub tol_rel: f64,
    pub eps: f64,
    pub max_turn: f64,
}

impl Default for ImageSearch {
    fn default() -> Self {
        Self { max_iter: 10, tol_abs: 1e-3, tol_rel: 1e-5, eps: 2e-6, max_turn: 0.25 }
    }
}

/// Newton iteration for the image of `target` of the given order.
#[allow(clippy::too_many_arguments)]
pub fn find_image(
    k: &Kerr,
    obs: &Observer,
    target: &dyn Fn(f64) -> Option<Sample>,
    guess: V3,
    order: u8,
    t_min: f64,
    ray: &RayConfig,
    search: &ImageSearch,
) -> Option<Image> {
    let window = order_window(order);
    let mut n = vec3::normalize(guess);
    let approach = |n: V3| closest_approach(k, obs, n, target, window, t_min, ray);
    let mut a0 = approach(n)?;
    for _ in 0..search.max_iter {
        let tol = search.tol_abs + search.tol_rel * a0.path;
        let res = vec3::norm(a0.miss);
        if res < tol {
            return make_image(k, obs, n, &a0, window, t_min, ray, search);
        }
        let t1 = vec3::any_orthogonal(n);
        let t2 = vec3::cross(n, t1);
        let eps = search.eps.max(1e-9);
        let a1 = approach(vec3::normalize(vec3::axpy(n, eps, t1)))?;
        let a2 = approach(vec3::normalize(vec3::axpy(n, eps, t2)))?;
        let j1 = vec3::scale(vec3::sub(a1.miss, a0.miss), 1.0 / eps);
        let j2 = vec3::scale(vec3::sub(a2.miss, a0.miss), 1.0 / eps);
        let (a11, a12, a22) = (vec3::dot(j1, j1), vec3::dot(j1, j2), vec3::dot(j2, j2));
        let det = a11 * a22 - a12 * a12;
        if det.abs() < 1e-300 {
            return None;
        }
        let b1 = -vec3::dot(j1, a0.miss);
        let b2 = -vec3::dot(j2, a0.miss);
        let mut d1 = (a22 * b1 - a12 * b2) / det;
        let mut d2 = (a11 * b2 - a12 * b1) / det;
        let turn = (d1 * d1 + d2 * d2).sqrt();
        if turn > search.max_turn {
            d1 *= search.max_turn / turn;
            d2 *= search.max_turn / turn;
        }
        let mut step = 1.0;
        // Backtracking: accept the first step that reduces the miss.
        loop {
            let cand = vec3::normalize(vec3::add(n, vec3::add(vec3::scale(t1, step * d1), vec3::scale(t2, step * d2))));
            if let Some(a) = approach(cand)
                && (vec3::norm(a.miss) < res || step < 0.05)
            {
                n = cand;
                a0 = a;
                break;
            }
            step *= 0.5;
            if step < 0.05 {
                return None;
            }
        }
    }
    let tol = search.tol_abs + search.tol_rel * a0.path;
    (vec3::norm(a0.miss) < tol).then(|| make_image(k, obs, n, &a0, window, t_min, ray, search))?
}

/// Builds the image record. The magnification uses the cross-section of
/// the ray bundle at the *fixed* emission point (the body's motion must not
/// enter the solid-angle mapping; it acts through `g` alone).
#[allow(clippy::too_many_arguments)]
fn make_image(
    k: &Kerr,
    obs: &Observer,
    n: V3,
    a: &Approach,
    window: (f64, f64),
    t_min: f64,
    ray: &RayConfig,
    search: &ImageSearch,
) -> Option<Image> {
    let fixed = a.body.pos;
    let target = move |_t: f64| Some(Sample { pos: fixed, vel: [0.0; 3] });
    let eps = search.eps.max(1e-9);
    let t1 = vec3::any_orthogonal(n);
    let t2 = vec3::cross(n, t1);
    let miss = |d: V3| closest_approach(k, obs, vec3::normalize(d), &target, window, t_min, ray).map(|a| a.miss);
    let m0 = miss(n)?;
    let j1 = vec3::scale(vec3::sub(miss(vec3::axpy(n, eps, t1))?, m0), 1.0 / eps);
    let j2 = vec3::scale(vec3::sub(miss(vec3::axpy(n, eps, t2))?, m0), 1.0 / eps);
    let w = k.four_velocity(a.body.pos, a.body.vel).unwrap_or([1.0, 0.0, 0.0, 0.0]);
    let pw = a.p[0] * w[0] + a.p[1] * w[1] + a.p[2] * w[2] + a.p[3] * w[3];
    Some(Image {
        dir: n,
        g: 1.0 / pw,
        area: vec3::norm(vec3::cross(j1, j2)),
        t_emit: a.t,
        body: a.body,
        path: a.path,
        residual: vec3::norm(a.miss),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pilot::Pilot;

    fn static_observer(k: &Kerr, pos: V3, look: V3) -> Observer {
        let p = Pilot::new(k, pos, [0.0; 3], look, [0.0, 0.0, 1.0]).unwrap();
        Observer { x: p.x, e: p.e }
    }

    #[test]
    fn flat_space_images_point_at_retarded_position() {
        let k = Kerr::new(0.0, 0.0);
        let o = [-10.0, 2.0, 3.0];
        let obs = static_observer(&k, o, [1.0, 0.0, 0.0]);
        let v = [0.0, 0.4, 0.1];
        let x_now = [30.0, 5.0, -4.0];
        let target = move |t: f64| Some(Sample { pos: vec3::axpy(x_now, t, v), vel: v });
        let guess = image_guess(&k, &obs, x_now, 0);
        let img = find_image(&k, &obs, &target, guess, 0, -1e9, &RayConfig::default(), &ImageSearch::default())
            .expect("image");
        // Retarded time: |x(t) − o| = −t.
        let mut t = -30.0;
        for _ in 0..50 {
            t = -vec3::norm(vec3::sub(vec3::axpy(x_now, t, v), o));
        }
        let expect = obs.local_direction(&k, vec3::sub(vec3::axpy(x_now, t, v), o));
        assert!(vec3::norm(vec3::sub(img.dir, expect)) < 1e-5, "{:?} vs {:?}", img.dir, expect);
        assert!((img.t_emit - t).abs() < 1e-3);
        // Doppler shift of a receding source: g = 1 / (γ (1 + v·n̂)).
        let pos = vec3::sub(vec3::axpy(x_now, t, v), o);
        let nhat = vec3::normalize(pos);
        let vv = vec3::dot(v, v);
        let expect_g = 1.0 / ((1.0 / (1.0 - vv).sqrt()) * (1.0 + vec3::dot(v, nhat)));
        assert!((img.g - expect_g).abs() < 1e-6, "g = {} vs {}", img.g, expect_g);
        // Flat space: area = D².
        let d = vec3::norm(pos);
        assert!((img.area / (d * d) - 1.0).abs() < 1e-3, "area {} vs {}", img.area, d * d);
    }

    #[test]
    fn weak_lensing_matches_point_mass_lens_equation() {
        let k = Kerr::new(1.0, 0.0);
        let dl = 500.0;
        let src = [500.0, 50.0, 0.0];
        let obs = static_observer(&k, [-dl, 0.0, 0.0], [1.0, 0.0, 0.0]);
        let target = move |_t: f64| Some(Sample { pos: src, vel: [0.0; 3] });
        let ray = RayConfig { step: 0.02, max_step: 4.0, ..Default::default() };
        // Reference: thin-lens equation β = θ − (D_LS/D_S) α(D_L |θ|) with the
        // Schwarzschild deflection to third order,
        // α = 4M/b + 15πM²/(4b²) + 128M³/(3b³).
        let beta = (50.0f64 / 1000.0).atan();
        let alpha = |b: f64| 4.0 / b + 15.0 * std::f64::consts::PI / (4.0 * b * b) + 128.0 / (3.0 * b * b * b);
        let map = |th: f64| th - 0.5 * th.signum() * alpha(dl * th.abs());
        let solve = |beta: f64, lo: f64, hi: f64| {
            let (mut lo, mut hi) = (lo, hi);
            for _ in 0..200 {
                let mid = 0.5 * (lo + hi);
                if (map(mid) - beta) * (map(lo) - beta) <= 0.0 { hi = mid } else { lo = mid }
            }
            0.5 * (lo + hi)
        };
        for (order, lo, hi) in [(0u8, 0.02, 0.3), (1u8, -0.3, -0.02)] {
            let expect = solve(beta, lo, hi);
            let guess = image_guess(&k, &obs, src, order);
            let img = find_image(&k, &obs, &target, guess, order, -1e9, &ray, &ImageSearch::default())
                .unwrap_or_else(|| panic!("order {order} image"));
            // Angle from the optical axis (towards the lens), signed along +y (left).
            let theta = img.dir[1].atan2(img.dir[0]);
            assert!((theta / expect - 1.0).abs() < 0.01, "order {order}: θ = {theta}, expected {expect}");
            // Axisymmetric lens magnification μ = (θ/β) dθ/dβ.
            let db = 1e-6;
            let dth = (solve(beta + db, lo, hi) - solve(beta - db, lo, hi)) / (2.0 * db);
            let mu = (expect / beta * dth).abs();
            let d = 1000.0f64;
            let mu_ray = d * d / img.area;
            assert!((mu_ray / mu - 1.0).abs() < 0.05, "order {order}: μ = {mu_ray} vs {mu}");
            assert!((img.g - 1.0).abs() < 1e-3);
        }
    }

    #[test]
    fn shadow_and_sky() {
        let k = Kerr::new(1.0, 0.9);
        let obs = static_observer(&k, [-40.0, 0.0, 3.0], [1.0, 0.0, 0.0]);
        let cfg = RayConfig::default();
        assert!(matches!(trace_to_sky(&k, &obs, [1.0, 0.0, 0.0], &cfg), RayEnd::Horizon));
        match trace_to_sky(&k, &obs, [-1.0, 0.0, 0.0], &cfg) {
            RayEnd::Sky { dir, g } => {
                assert!(dir[0] < -0.99);
                // Static observer deep in the well sees the sky blueshifted.
                let expect = 1.0 / (1.0 - 2.0 / 40.1f64).sqrt();
                assert!((g / expect - 1.0).abs() < 0.01, "g = {g}");
            }
            other => panic!("{other:?}"),
        }
    }
}
