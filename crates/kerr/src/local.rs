//! Flight inside a star system: the gravity of the nearest star and its
//! planets on the ship, the solid bodies it can run into, and Kepler
//! elements for telemetry and guidance.
//!
//! # Physics
//!
//! A star and its planets are far too light to curve spacetime noticeably
//! (`GM/(r c²)` is 2 × 10⁻⁶ at the Sun's surface, 10⁻⁹ in low Earth orbit),
//! so their pull is added to the ship's Kerr geodesic as a weak-field
//! perturbation, the same way the cluster adds the stars' mutual pulls:
//! as a 4-acceleration orthogonal to `u`, next to the thrust.
//!
//! * **Field.** Newtonian `g = −Σ GM d/|d|³` of the star and its planets
//!   (uniform-sphere field `−GM d/R³` inside a body), evaluated at the
//!   ship's event. The star moves on its Kerr geodesic; between refreshes
//!   (see [`Local::valid_until`]) its worldline is a Taylor expansion to
//!   second order, with its geodesic acceleration taken from the Kerr
//!   equations, so a ship falling with the star stays with it. Planets sit
//!   at the star plus their Kepler offsets at the same coordinate time, as
//!   [`crate::planets`] and the renderer place them.
//! * **Any speed.** In the star's rest frame `w`, a ship with relative
//!   speed `v` and Lorentz factor `γ` gets
//!   `F = g∥ + γ²(1 + v²) g⊥`, then `a = F + (F·u) u`. This is the
//!   linearised field of a static mass acting on a test particle: the
//!   rapidity changes at `g∥` (conserving `γ(1 + Φ)`), and the path bends
//!   at `(1 + v²) g⊥ / v`, which is Newton's rate at low speed and twice it
//!   for light. At the few km/s of orbital flight it is Newton's law in the
//!   simulation's coordinate time, which is also what the planets' Kepler
//!   orbits use, so a ship orbiting a planet and the planet agree to
//!   `O(v²)` ≈ 10⁻⁹.
//! * **Range.** Beyond the star's Hill sphere around the hole,
//!   `r_H = r (m / 3M)^{1/3}` (about 4 AU for a Sun at 1000 AU), the hole's
//!   tide dominates. The local field is faded out between 3 and 6 `r_H`,
//!   where the star's pull is already ≲ 1/30 of the tide it would compete
//!   with; beyond that the ship follows the plain Kerr geodesic.
//! * **Not modelled:** the gravity of stars other than the nearest (a
//!   neighbour's pull on the ship is ≲ 10⁻¹⁵ /M; the star's own weak-field
//!   pull from the cluster is shared with the ship), atmospheric drag,
//!   terrain relief (the surface is the sphere of the planet's mean radius)
//!   and the stellar-mass black holes (they are not stars; the ship feels
//!   only Sgr A*'s geometry near them).

use crate::cluster::{BodyKind, Cluster};
use crate::geodesic;
use crate::metric::Kerr;
use crate::pilot::Field;
use crate::planets::{self, C_KM_S, KM_PER_M, Planet, System};
use crate::units::SECONDS_PER_M;
use crate::vec3::{self, V3, V4};

/// The field is full strength inside this many Hill radii of the star...
const FADE_IN: f64 = 3.0;
/// ...and off beyond this many.
const FADE_OUT: f64 = 6.0;
/// Substeps resolve `1/ORBIT_STEPS` of a radian of an orbit at the ship's
/// distance from each body (a circular orbit gets `2π · 50` substeps)...
const ORBIT_STEPS: f64 = 50.0;
/// ...unless the ship moves so fast that gravity changes its velocity by
/// less than this fraction per substep.
const KICK: f64 = 1e-3;
/// Largest relative error allowed in the extrapolated source positions
/// (relative to the ship's distance from them) before a refresh.
const EXTRAPOLATION_ERROR: f64 = 1e-6;
/// Coordinate time between refreshes, units of M.
const REFRESH: (f64, f64) = (2.0, 5000.0);

/// Gravitational parameter `GM` in km³/s² to units of M (with `G = c = 1`
/// this is the mass in units of the hole's).
pub fn gm_to_m(gm_km: f64) -> f64 {
    gm_km * SECONDS_PER_M * SECONDS_PER_M / (KM_PER_M * KM_PER_M * KM_PER_M)
}

/// A planet of a particular star (slots are reused, hence the generation).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PlanetRef {
    pub star: usize,
    pub generation: u32,
    pub planet: usize,
}

/// A body of a [`Local`] system.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BodyRef {
    Star,
    Planet(usize),
}

/// Where a path first touched a body.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Contact {
    pub body: BodyRef,
    /// The event of contact.
    pub x: V4,
}

/// Osculating two-body orbit (Newtonian, units of M and c).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Elements {
    pub semi_major: f64,
    pub eccentricity: f64,
    /// Closest and farthest distance from the centre (∞ if unbound).
    pub periapsis: f64,
    pub apoapsis: f64,
    /// Orbital period (∞ if unbound).
    pub period: f64,
}

impl Elements {
    pub fn of(mu: f64, r: V3, v: V3) -> Self {
        let d = vec3::norm(r);
        let energy = 0.5 * vec3::dot(v, v) - mu / d;
        let h = vec3::norm(vec3::cross(r, v));
        let e = (1.0 + 2.0 * energy * h * h / (mu * mu)).max(0.0).sqrt();
        if energy >= 0.0 {
            let periapsis = h * h / (mu * (1.0 + e));
            return Self {
                semi_major: f64::INFINITY,
                eccentricity: e,
                periapsis,
                apoapsis: f64::INFINITY,
                period: f64::INFINITY,
            };
        }
        let a = -mu / (2.0 * energy);
        Self {
            semi_major: a,
            eccentricity: e,
            periapsis: a * (1.0 - e),
            apoapsis: a * (1.0 + e),
            period: std::f64::consts::TAU * (a * a * a / mu).sqrt(),
        }
    }

    pub fn bound(&self) -> bool {
        self.period.is_finite()
    }
}

/// Mass and size of a planet in units of M.
#[derive(Clone, Copy, Debug)]
struct Mass {
    mu: f64,
    radius: f64,
    /// Hill radius around the star.
    hill: f64,
}

/// A snapshot of one star system for the ship: the star's worldline near
/// the snapshot time and its planets.
#[derive(Clone, Debug)]
pub struct Local {
    pub star: usize,
    pub generation: u32,
    /// The star's planets, if it has any.
    pub system: Option<System>,
    /// The star's `GM` and radius, units of M.
    pub star_mu: f64,
    pub star_radius: f64,
    /// The star's Hill radius around the hole, units of M.
    pub hill: f64,
    t0: f64,
    x0: V3,
    v0: V3,
    a0: V3,
    /// The star's own weak-field pull from the cluster (shared by the ship).
    pert: V3,
    planets: Vec<Mass>,
    /// Coordinate time up to which the extrapolated star stays accurate
    /// for a ship where the snapshot was taken.
    pub valid_until: f64,
}

/// Hill radius of body `i` around the hole, units of M.
fn hill_radius(cluster: &Cluster, i: usize) -> f64 {
    let b = &cluster.bodies[i];
    vec3::norm(b.position()) * (b.params.mass / 3.0).cbrt()
}

fn is_star(cluster: &Cluster, i: usize) -> bool {
    let b = &cluster.bodies[i];
    b.alive && b.params.kind == BodyKind::Star && b.params.radius > 0.0
}

/// The star whose field reaches `pos` (the one with `pos` deepest inside
/// its range, in Hill radii), if any.
pub fn select(cluster: &Cluster, pos: V3) -> Option<usize> {
    (0..cluster.len())
        .filter(|&i| is_star(cluster, i))
        .map(|i| (i, vec3::norm(vec3::sub(cluster.bodies[i].position(), pos)) / hill_radius(cluster, i)))
        .filter(|&(_, x)| x < FADE_OUT)
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(i, _)| i)
}

/// Lower bound on the coordinate time before a ship at `pos` moving at
/// coordinate speed `speed` can enter any star's range.
pub fn time_to_range(cluster: &Cluster, pos: V3, speed: f64) -> f64 {
    let gap = (0..cluster.len())
        .filter(|&i| is_star(cluster, i))
        .map(|i| vec3::norm(vec3::sub(cluster.bodies[i].position(), pos)) - FADE_OUT * hill_radius(cluster, i))
        .fold(f64::INFINITY, f64::min);
    gap.max(0.0) / speed.max(1e-12)
}

impl Local {
    /// Snapshot of star `star` at the cluster's current time, for a ship at
    /// `ship` (which sets how long the snapshot stays accurate).
    pub fn new(k: &Kerr, cluster: &Cluster, star: usize, ship: V3) -> Self {
        let b = &cluster.bodies[star];
        let system = planets::system(cluster.cfg.seed, star, b);
        // With planets, use the system's own GM so the Kepler orbits and the
        // ship's field agree exactly.
        let star_mu = system.as_ref().map_or(b.params.mass, |s| gm_to_m(s.star_gm()));
        let planets = system.as_ref().map_or_else(Vec::new, |s| {
            s.planets
                .iter()
                .map(|p| Mass {
                    mu: gm_to_m(p.gm()),
                    radius: p.radius_km / KM_PER_M,
                    hill: p.hill_radius_km(s.star_mass_kg) / KM_PER_M,
                })
                .collect()
        });
        let s = b.state;
        let x0 = b.position();
        let v0 = geodesic::coordinate_velocity(k, &s);
        let a0 = vec3::add(geodesic_acceleration(k, &s), b.pert);
        let mut local = Self {
            star,
            generation: b.generation,
            system,
            star_mu,
            star_radius: b.params.radius,
            hill: hill_radius(cluster, star),
            t0: cluster.t,
            x0,
            v0,
            a0,
            pert: b.pert,
            planets,
            valid_until: cluster.t,
        };
        // Error of the second-order extrapolation: jerk ~ 3 a ω, error
        // ≈ jerk Δt³ / 6, kept below EXTRAPOLATION_ERROR of the distance to
        // the nearest source.
        let jerk = 3.0 * vec3::norm(a0) * vec3::norm(v0) / vec3::norm(x0).max(1.0) + 1e-300;
        let near = local
            .sources(cluster.t)
            .map(|(c, _, r)| vec3::norm(vec3::sub(ship, c)).max(r))
            .fold(f64::INFINITY, f64::min);
        let dt = (6.0 * EXTRAPOLATION_ERROR * near / jerk).cbrt().clamp(REFRESH.0, REFRESH.1);
        local.valid_until = cluster.t + dt;
        local
    }

    /// Star position and coordinate velocity at coordinate time `t`.
    pub fn star_state(&self, t: f64) -> (V3, V3) {
        let dt = t - self.t0;
        let x = vec3::axpy(vec3::axpy(self.x0, dt, self.v0), 0.5 * dt * dt, self.a0);
        (x, vec3::axpy(self.v0, dt, self.a0))
    }

    pub fn planet_count(&self) -> usize {
        self.planets.len()
    }

    pub fn planet(&self, i: usize) -> Option<&Planet> {
        self.system.as_ref().and_then(|s| s.planets.get(i))
    }

    pub fn planet_ref(&self, i: usize) -> PlanetRef {
        PlanetRef { star: self.star, generation: self.generation, planet: i }
    }

    /// Whether this snapshot is of `p`'s star.
    pub fn holds(&self, p: PlanetRef) -> bool {
        self.star == p.star && self.generation == p.generation && p.planet < self.planets.len()
    }

    /// `GM` and radius of planet `i`, units of M.
    pub fn planet_mass(&self, i: usize) -> (f64, f64) {
        (self.planets[i].mu, self.planets[i].radius)
    }

    /// Planet `i`'s position and coordinate velocity at coordinate time `t`.
    pub fn planet_state(&self, i: usize, t: f64) -> (V3, V3) {
        let (xs, vs) = self.star_state(t);
        let (off, vel) = self.system.as_ref().expect("planets").planet_state(i, t);
        (vec3::axpy(xs, 1.0 / KM_PER_M, off), vec3::axpy(vs, 1.0 / C_KM_S, vel))
    }

    /// Position and coordinate velocity of a body at `t`.
    pub fn body_state(&self, body: BodyRef, t: f64) -> (V3, V3) {
        match body {
            BodyRef::Star => self.star_state(t),
            BodyRef::Planet(i) => self.planet_state(i, t),
        }
    }

    /// (centre, GM, radius) of every body at `t`, star first.
    fn sources(&self, t: f64) -> impl Iterator<Item = (V3, f64, f64)> + '_ {
        let star = std::iter::once((self.star_state(t).0, self.star_mu, self.star_radius));
        star.chain(
            (0..self.planets.len())
                .map(move |i| (self.planet_state(i, t).0, self.planets[i].mu, self.planets[i].radius)),
        )
    }

    /// Strength of the field at distance `d` from the star (1 inside
    /// `FADE_IN` Hill radii, 0 beyond `FADE_OUT`).
    fn weight(&self, d: f64) -> f64 {
        let x = ((d / self.hill - FADE_IN) / (FADE_OUT - FADE_IN)).clamp(0.0, 1.0);
        1.0 - x * x * (3.0 - 2.0 * x)
    }

    /// Coordinate acceleration of the field at event `(t, pos)`, leaving
    /// out planet `skip`.
    pub fn gravity_except(&self, t: f64, pos: V3, skip: Option<usize>) -> V3 {
        let w = self.weight(vec3::norm(vec3::sub(pos, self.star_state(t).0)));
        if w == 0.0 {
            return [0.0; 3];
        }
        let mut acc = self.pert;
        for (j, (c, mu, radius)) in self.sources(t).enumerate() {
            if skip.is_some_and(|i| j == i + 1) {
                continue;
            }
            let d = vec3::sub(pos, c);
            let r = vec3::norm(d).max(radius);
            acc = vec3::axpy(acc, -mu / (r * r * r), d);
        }
        vec3::scale(acc, w)
    }

    pub fn gravity(&self, t: f64, pos: V3) -> V3 {
        self.gravity_except(t, pos, None)
    }

    /// The body whose gravity dominates at `pos`: a planet inside its Hill
    /// sphere, else the star. Returns it with the distance to its centre.
    pub fn dominant(&self, t: f64, pos: V3) -> (BodyRef, f64) {
        for i in 0..self.planets.len() {
            let d = vec3::norm(vec3::sub(pos, self.planet_state(i, t).0));
            if d < self.planets[i].hill {
                return (BodyRef::Planet(i), d);
            }
        }
        (BodyRef::Star, vec3::norm(vec3::sub(pos, self.star_state(t).0)))
    }

    /// `GM` and radius of a body.
    pub fn mass_of(&self, body: BodyRef) -> (f64, f64) {
        match body {
            BodyRef::Star => (self.star_mu, self.star_radius),
            BodyRef::Planet(i) => self.planet_mass(i),
        }
    }

    /// Whether the field reaches `pos` at all.
    pub fn reaches(&self, t: f64, pos: V3) -> bool {
        self.weight(vec3::norm(vec3::sub(pos, self.star_state(t).0))) > 0.0
    }

    /// Where the straight path from event `x0` to `x1` first touches a
    /// body (relative motion is linear over the path), if it does.
    pub fn contact(&self, x0: V4, x1: V4) -> Option<Contact> {
        let (p0, p1) = (vec3::spatial(x0), vec3::spatial(x1));
        let mut best: Option<(f64, BodyRef)> = None;
        let bodies = std::iter::once(BodyRef::Star).chain((0..self.planets.len()).map(BodyRef::Planet));
        for body in bodies {
            let radius = self.mass_of(body).1;
            let a = vec3::sub(p0, self.body_state(body, x0[0]).0);
            let b = vec3::sub(p1, self.body_state(body, x1[0]).0);
            let Some(s) = segment_enters_sphere(a, b, radius) else { continue };
            if best.is_none_or(|(s0, _)| s < s0) {
                best = Some((s, body));
            }
        }
        best.map(|(s, body)| Contact { body, x: std::array::from_fn(|m| x0[m] + s * (x1[m] - x0[m])) })
    }
}

/// Fraction along `a → b` where the segment first reaches distance
/// `radius` from the origin (0 if it starts inside).
fn segment_enters_sphere(a: V3, b: V3, radius: f64) -> Option<f64> {
    let r2 = radius * radius;
    let c = vec3::dot(a, a) - r2;
    if c <= 0.0 {
        return Some(0.0);
    }
    let d = vec3::sub(b, a);
    let qa = vec3::dot(d, d);
    let qb = vec3::dot(a, d);
    if qb >= 0.0 || qa == 0.0 {
        return None;
    }
    let disc = qb * qb - qa * c;
    if disc < 0.0 {
        return None;
    }
    let s = c / (-qb + disc.sqrt());
    (s <= 1.0).then_some(s)
}

/// `d²x/dt²` of a free-falling body, from the directional derivative of its
/// coordinate velocity along the Hamiltonian flow.
fn geodesic_acceleration(k: &Kerr, s: &geodesic::BodyState) -> V3 {
    let Some(rate) = geodesic::body_rhs(k, s, [0.0; 4]) else { return [0.0; 3] };
    let eps = 1.0;
    let shifted = |sign: f64| -> V3 {
        let y: geodesic::BodyState = std::array::from_fn(|i| s[i] + sign * eps * rate[i]);
        geodesic::coordinate_velocity(k, &y)
    };
    vec3::scale(vec3::sub(shifted(1.0), shifted(-1.0)), 0.5 / eps)
}

impl Field for Local {
    fn accel(&self, k: &Kerr, x: V4, u: V4) -> V4 {
        let (t, pos) = (x[0], vec3::spatial(x));
        let g = self.gravity(t, pos);
        if g == [0.0; 3] {
            return [0.0; 4];
        }
        let terms = k.terms(pos);
        // The star's rest frame at the ship.
        let vs = self.star_state(t).1;
        let ws = [1.0, vs[0], vs[1], vs[2]];
        let w = ws.map(|c| c / (-terms.dot(ws, ws)).sqrt());
        let g4 = [0.0, g[0], g[1], g[2]];
        let gw = terms.dot(g4, w);
        let g4: V4 = std::array::from_fn(|m| g4[m] + gw * w[m]);
        // Ship velocity relative to the star frame, and the split of g.
        let gamma = -terms.dot(u, w);
        let rel: V4 = std::array::from_fn(|m| u[m] / gamma - w[m]);
        let v2 = terms.dot(rel, rel).max(0.0);
        let (par, perp) = if v2 > 1e-40 {
            let s = terms.dot(g4, rel) / v2;
            let par: V4 = rel.map(|c| s * c);
            (par, std::array::from_fn(|m| g4[m] - par[m]))
        } else {
            ([0.0; 4], g4)
        };
        let boost = gamma * gamma * (1.0 + v2);
        let f: V4 = std::array::from_fn(|m| par[m] + boost * perp[m]);
        let fu = terms.dot(f, u);
        std::array::from_fn(|m| f[m] + fu * u[m])
    }

    fn max_substep(&self, _k: &Kerr, x: V4, u: V4) -> f64 {
        let (t, pos) = (x[0], vec3::spatial(x));
        if !self.reaches(t, pos) {
            return f64::INFINITY;
        }
        let v = [u[1] / u[0], u[2] / u[0], u[3] / u[0]];
        let bodies = std::iter::once(BodyRef::Star).chain((0..self.planets.len()).map(BodyRef::Planet));
        let mut dt = f64::INFINITY;
        for body in bodies {
            let (mu, radius) = self.mass_of(body);
            let (c, vc) = self.body_state(body, t);
            let d = vec3::norm(vec3::sub(pos, c)).max(radius);
            let orbit = (d * d * d / mu).sqrt() / ORBIT_STEPS;
            let fast = KICK * vec3::norm(vec3::sub(v, vc)) * d * d / mu;
            dt = dt.min(orbit.max(fast));
        }
        dt / u[0].max(1.0)
    }

    fn hit(&self, x0: V4, x1: V4) -> bool {
        self.contact(x0, x1).is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gm_conversion_matches_the_hole_mass() {
        // The Sun's GM in units of M is 1/4.3e6.
        let sun = gm_to_m(planets::G_KM * planets::SUN_MASS_KG);
        assert!((sun * 4.3e6 - 1.0).abs() < 1e-3, "{sun}");
    }

    #[test]
    fn elements_of_circular_and_escape_orbits() {
        let mu = 2.0;
        let e = Elements::of(mu, [3.0, 0.0, 0.0], [0.0, (mu / 3.0f64).sqrt(), 0.0]);
        assert!((e.periapsis - 3.0).abs() < 1e-12 && (e.apoapsis - 3.0).abs() < 1e-12);
        assert!((e.period - std::f64::consts::TAU * (27.0 / mu).sqrt()).abs() < 1e-9);
        let esc = Elements::of(mu, [3.0, 0.0, 0.0], [0.0, 1.01 * (2.0 * mu / 3.0f64).sqrt(), 0.0]);
        assert!(!esc.bound() && (esc.periapsis - 3.0).abs() < 1e-9);
    }

    #[test]
    fn segments_enter_spheres() {
        assert_eq!(segment_enters_sphere([2.0, 0.0, 0.0], [3.0, 0.0, 0.0], 1.0), None);
        assert_eq!(segment_enters_sphere([2.0, 0.0, 0.0], [1.5, 0.0, 0.0], 1.0), None);
        let s = segment_enters_sphere([2.0, 0.0, 0.0], [-2.0, 0.0, 0.0], 1.0).unwrap();
        assert!((s - 0.25).abs() < 1e-12);
        assert_eq!(segment_enters_sphere([0.5, 0.0, 0.0], [3.0, 0.0, 0.0], 1.0), Some(0.0));
        assert_eq!(segment_enters_sphere([2.0, 2.0, 0.0], [-2.0, 2.0, 0.0], 1.0), None);
    }
}
