//! The nuclear star cluster: every body follows a Kerr geodesic, nudged by
//!
//! * weak-field pulls from the other massive bodies, evaluated on each
//!   receiver's past light cone: the source is looked up at its retarded
//!   time and its position extrapolated forward with its retarded velocity
//!   (the gravitational analogue of the Liénard–Wiechert result that the
//!   field of a uniformly moving source points at its "present" position),
//!   and
//! * 2.5PN radiation reaction (Iyer–Will, harmonic gauge) for compact
//!   objects on tight orbits, so they inspiral.
//!
//! Both perturbations enter as 4-forces orthogonal to the 4-velocity, so the
//! mass shell is preserved and the geodesic part stays exact. The weak-field
//! pull is refreshed once per history tick and held fixed in between
//! (operator splitting); radiation reaction is evaluated at every stage.

use crate::geodesic::{self, BodyState};
use crate::history::{HistoryRing, Sample};
use crate::integrate::{Dopri5, Outcome, Tolerance};
use crate::metric::Kerr;
use crate::orbit::{self, Launch};
use crate::rng::Rng;
use crate::units;
use crate::vec3::{self, V3};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BodyKind {
    Station,
    Compact,
    Star,
}

#[derive(Clone, Copy, Debug)]
pub struct BodyParams {
    pub kind: BodyKind,
    /// Mass in units of the black hole mass.
    pub mass: f64,
    /// Photospheric temperature in kelvin (rendering).
    pub temperature: f64,
    /// Luminosity (rendering): in solar luminosities for physical stars,
    /// arbitrary for the stylized cluster.
    pub luminosity: f64,
    /// Radius in units of `M` (0: a point, never resolved on screen).
    pub radius: f64,
    pub feels_perturbations: bool,
    pub radiates: bool,
}

#[derive(Clone, Debug)]
pub struct Body {
    pub params: BodyParams,
    pub state: BodyState,
    /// Step-size hint for the adaptive integrator.
    pub h: f64,
    /// Cached weak-field coordinate acceleration from the other bodies.
    pub pert: V3,
    pub alive: bool,
    /// Worldline history is meaningful from this coordinate time on.
    pub valid_from: f64,
    /// Coordinate time of capture/escape (`+∞` while alive).
    pub died_at: f64,
    /// Incremented whenever the slot is reused for a new body.
    pub generation: u32,
}

impl Body {
    pub fn position(&self) -> V3 {
        geodesic::position(&self.state)
    }
    /// Whether the worldline exists at coordinate time `t`.
    pub fn exists_at(&self, t: f64) -> bool {
        t >= self.valid_from && t <= self.died_at
    }
}

#[derive(Clone, Debug)]
pub struct ClusterConfig {
    pub seed: u64,
    pub stars: usize,
    pub compact_objects: usize,
    pub r_min: f64,
    pub r_max: f64,
    pub cusp_gamma: f64,
    /// Star masses are log-uniform in this range (units of M).
    pub star_mass: (f64, f64),
    /// How stars are made: stylized (exaggerated masses, arbitrary
    /// brightness, point-like) or physical.
    pub star_model: StarModel,
    pub compact_mass: (f64, f64),
    pub compact_radius: (f64, f64),
    pub history_dt: f64,
    pub history_len: usize,
    /// Plummer softening length of the weak-field pull.
    pub softening: f64,
    /// Bodies at least this massive act as sources.
    pub source_mass_min: f64,
    /// At most this many (the most massive) sources are used.
    pub max_sources: usize,
    /// Refresh the weak-field pulls every this many history ticks.
    pub perturbation_every: usize,
    pub radiation_reaction: bool,
    pub escape_radius: f64,
}

/// How cluster stars are populated.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StarModel {
    /// Exaggerated masses so mutual pulls are visible; point sources.
    Stylized,
    /// Real stars: masses from a Salpeter mass function in `mass_msun`,
    /// main-sequence radii and luminosities, with a share of red giants.
    Physical { mass_msun: (f64, f64), giant_fraction: f64 },
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self {
            seed: 0x5EED_CAFE,
            stars: 384,
            compact_objects: 4,
            r_min: 9.0,
            r_max: 260.0,
            cusp_gamma: 1.75,
            star_mass: (2e-6, 2e-4),
            star_model: StarModel::Stylized,
            compact_mass: (4e-3, 1.5e-2),
            compact_radius: (9.0, 16.0),
            history_dt: 2.5,
            history_len: 384,
            softening: 0.6,
            source_mass_min: 2e-5,
            max_sources: 64,
            perturbation_every: 1,
            radiation_reaction: true,
            escape_radius: 900.0,
        }
    }
}

/// A body to place on a prescribed orbit (used for stations).
#[derive(Clone, Copy, Debug)]
pub struct Placement {
    pub launch: Launch,
    pub params: BodyParams,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ClusterEvent {
    Captured { body: usize, t: f64 },
    Escaped { body: usize, t: f64 },
    Respawned { body: usize, t: f64 },
}

pub struct Cluster {
    pub kerr: Kerr,
    pub cfg: ClusterConfig,
    pub bodies: Vec<Body>,
    /// Current coordinate time of every live body.
    pub t: f64,
    pub history: HistoryRing,
    pub events: Vec<ClusterEvent>,
    rng: Rng,
    stepper: Dopri5,
    sources: Vec<usize>,
    ticks: usize,
}

impl Cluster {
    pub fn new(kerr: Kerr, cfg: ClusterConfig, placements: &[Placement]) -> Self {
        let mut rng = Rng::new(cfg.seed);
        let n = placements.len() + cfg.compact_objects + cfg.stars;
        // Tolerance, not a cap, sets the step: far-out orbits take huge steps.
        let stepper = Dopri5 { tol: Tolerance { abs: 1e-9, rel: 1e-9 }, h_min: 1e-7, h_max: 1e12, max_steps: 50_000 };
        let mut bodies = Vec::with_capacity(n);
        for p in placements {
            bodies.push(make_body(&kerr, p.launch, p.params));
        }
        for _ in 0..cfg.compact_objects {
            let (launch, params) = random_compact(&kerr, &cfg, &mut rng);
            bodies.push(make_body(&kerr, launch, params));
        }
        for _ in 0..cfg.stars {
            let (launch, params) = random_star(&kerr, &cfg, &mut rng);
            bodies.push(make_body(&kerr, launch, params));
        }
        let history = HistoryRing::new(n, cfg.history_len, cfg.history_dt);
        let mut c = Self {
            kerr,
            cfg,
            bodies,
            t: 0.0,
            history,
            events: Vec::new(),
            rng,
            stepper,
            sources: Vec::new(),
            ticks: 0,
        };
        c.fill_initial_history();
        c.refresh_sources();
        c.update_perturbations();
        c
    }

    /// How far back in coordinate time the stored history reaches.
    pub fn lookback(&self) -> f64 {
        (self.history.capacity() - 1) as f64 * self.history.dt()
    }

    pub fn len(&self) -> usize {
        self.bodies.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bodies.is_empty()
    }

    /// Live position/velocity sample of a body.
    pub fn current_sample(&self, i: usize) -> Sample {
        let b = &self.bodies[i];
        Sample { pos: b.position(), vel: geodesic::coordinate_velocity(&self.kerr, &b.state) }
    }

    /// Position/velocity of body `i` at coordinate time `t ≤ self.t`, or
    /// `None` outside its recorded worldline.
    pub fn sample_at(&self, i: usize, t: f64) -> Option<Sample> {
        let b = &self.bodies[i];
        if !b.exists_at(t) || t > self.t + 1e-9 {
            return None;
        }
        let now = b.alive.then(|| (self.t, self.current_sample(i)));
        self.history.at(i, t, now)
    }

    /// Integrate backward from t = 0 to fill the history grid, so the past
    /// light cone is populated from the first frame.
    fn fill_initial_history(&mut self) {
        let cap = self.history.capacity();
        let dt = self.history.dt();
        let mut columns = vec![vec![Sample::default(); self.bodies.len()]; cap];
        for i in 0..self.bodies.len() {
            let (samples, valid) = self.backward_samples(i, cap);
            self.bodies[i].valid_from = -((valid.max(1) - 1) as f64) * dt;
            for (age, s) in samples.into_iter().enumerate() {
                columns[age][i] = s;
            }
        }
        for age in (0..cap).rev() {
            let col = &columns[age];
            self.history.push(-(age as f64) * dt, |b| col[b]);
        }
    }

    /// Samples of body `i`'s pure-geodesic past at `t − age·dt` for
    /// `age = 0..count`; also returns how many are valid.
    fn backward_samples(&self, i: usize, count: usize) -> (Vec<Sample>, usize) {
        let k = self.kerr;
        let dt = self.history.dt();
        let mut s = self.bodies[i].state;
        let mut h = 1.0;
        let mut out = Vec::with_capacity(count);
        let mut valid = 0;
        let rhs = |y: &BodyState| geodesic::body_rhs(&k, y, [0.0; 4]);
        for age in 0..count {
            if age > 0 && valid == age {
                let mut f = rhs;
                if self.stepper.integrate(&mut f, &mut s, -dt, &mut h) == Outcome::Reached
                    && k.radius(geodesic::position(&s)) > k.r_plus() * 1.05
                {
                    valid += 1;
                }
            } else if age == 0 {
                valid = 1;
            }
            out.push(Sample { pos: geodesic::position(&s), vel: geodesic::coordinate_velocity(&k, &s) });
        }
        (out, valid)
    }

    fn refresh_sources(&mut self) {
        let mut src: Vec<usize> =
            (0..self.bodies.len()).filter(|&i| self.bodies[i].params.mass >= self.cfg.source_mass_min).collect();
        src.sort_by(|&a, &b| self.bodies[b].params.mass.total_cmp(&self.bodies[a].params.mass));
        src.truncate(self.cfg.max_sources);
        self.sources = src;
    }

    /// Retarded, velocity-extrapolated position of source `j` as seen from
    /// event `(t, x)`.
    pub fn retarded_source(&self, j: usize, t: f64, x: V3) -> Option<(V3, f64)> {
        let mut t_ret = t - vec3::norm(vec3::sub(x, self.bodies[j].position()));
        let mut s = self.sample_at(j, t_ret)?;
        for _ in 0..6 {
            let next = t - vec3::norm(vec3::sub(x, s.pos));
            let conv = (next - t_ret).abs() < 1e-6;
            t_ret = next;
            s = self.sample_at(j, t_ret)?;
            if conv {
                break;
            }
        }
        Some((vec3::axpy(s.pos, t - t_ret, s.vel), t_ret))
    }

    fn update_perturbations(&mut self) {
        let eps2 = self.cfg.softening * self.cfg.softening;
        let t = self.t;
        let mut acc = vec![[0.0; 3]; self.bodies.len()];
        for (i, a) in acc.iter_mut().enumerate() {
            let b = &self.bodies[i];
            if !b.alive || !b.params.feels_perturbations {
                continue;
            }
            let x = b.position();
            for &j in &self.sources {
                if j == i {
                    continue;
                }
                let Some((xs, _)) = self.retarded_source(j, t, x) else { continue };
                let d = vec3::sub(xs, x);
                let r2 = vec3::dot(d, d) + eps2;
                *a = vec3::axpy(*a, self.bodies[j].params.mass / (r2 * r2.sqrt()), d);
            }
        }
        for (b, a) in self.bodies.iter_mut().zip(acc) {
            b.pert = a;
        }
    }

    /// Advance every body to coordinate time `t_target`.
    pub fn advance_to(&mut self, t_target: f64) {
        while self.t < t_target - 1e-12 {
            let t_tick = self.history.t_next();
            let t_next = t_target.min(t_tick);
            self.integrate_all(t_next - self.t);
            self.t = t_next;
            if t_next >= t_tick - 1e-9 {
                self.t = t_tick;
                let snapshot: Vec<Sample> = (0..self.bodies.len()).map(|i| self.current_sample(i)).collect();
                self.history.push(t_tick, |b| snapshot[b]);
                self.recycle();
                self.ticks += 1;
                if self.ticks.is_multiple_of(self.cfg.perturbation_every.max(1)) {
                    self.update_perturbations();
                }
            }
        }
    }

    fn integrate_all(&mut self, span: f64) {
        if span <= 0.0 {
            return;
        }
        let k = self.kerr;
        let rr = self.cfg.radiation_reaction;
        let r_plus = k.r_plus();
        let t_end = self.t + span;
        for i in 0..self.bodies.len() {
            let b = &mut self.bodies[i];
            if !b.alive {
                continue;
            }
            let pert = b.pert;
            let params = b.params;
            let mut f = |y: &BodyState| {
                let x = geodesic::position(y);
                let u = k.raise(x, geodesic::momentum(y));
                let mut acc = pert;
                if rr && params.radiates {
                    acc = vec3::add(acc, radiation_reaction_accel(&k, x, u, params.mass));
                }
                let force = if acc == [0.0; 3] { [0.0; 4] } else { geodesic::coordinate_accel_to_force(&k, x, u, acc) };
                geodesic::body_rhs(&k, y, force)
            };
            let out = self.stepper.integrate(&mut f, &mut b.state, span, &mut b.h);
            let r = k.radius(b.position());
            if out != Outcome::Reached || r < r_plus * 1.01 {
                b.alive = false;
                b.died_at = t_end;
                self.events.push(ClusterEvent::Captured { body: i, t: t_end });
            } else if r > self.cfg.escape_radius {
                b.alive = false;
                b.died_at = t_end;
                self.events.push(ClusterEvent::Escaped { body: i, t: t_end });
            }
        }
    }

    /// Reuse slots of bodies whose light has entirely left the stored
    /// history window, so their final images (e.g. freezing and reddening
    /// at the horizon) stay visible for as long as they are on the past
    /// light cone of anyone inside the cluster.
    fn recycle(&mut self) {
        let lookback = self.lookback();
        let mut changed = false;
        for i in 0..self.bodies.len() {
            let b = &self.bodies[i];
            if b.alive || b.params.kind == BodyKind::Station || self.t - b.died_at < lookback {
                continue;
            }
            let (launch, params) = match b.params.kind {
                BodyKind::Compact => random_compact(&self.kerr, &self.cfg, &mut self.rng),
                _ => random_star(&self.kerr, &self.cfg, &mut self.rng),
            };
            let generation = b.generation + 1;
            let mut nb = make_body(&self.kerr, launch, params);
            nb.generation = generation;
            self.bodies[i] = nb;
            let (samples, valid) = self.backward_samples(i, self.history.len());
            for (age, s) in samples.into_iter().enumerate() {
                self.history.set(i, age, s);
            }
            self.bodies[i].valid_from = self.t - ((valid.max(1) - 1) as f64) * self.history.dt();
            self.events.push(ClusterEvent::Respawned { body: i, t: self.t });
            changed = true;
        }
        if changed {
            self.refresh_sources();
        }
    }

    /// Specific orbital energy `E = −p_t` of a body.
    pub fn energy(&self, i: usize) -> f64 {
        geodesic::energy(geodesic::momentum(&self.bodies[i].state))
    }
}

fn make_body(k: &Kerr, launch: Launch, params: BodyParams) -> Body {
    let u = k.four_velocity(launch.pos, launch.vel).expect("launch velocity must be timelike");
    let r = k.radius(launch.pos);
    Body {
        params,
        state: geodesic::body_state(k, launch.pos, u, 0.0),
        h: 0.05 * r.powf(1.5),
        pert: [0.0; 3],
        alive: true,
        valid_from: 0.0,
        died_at: f64::INFINITY,
        generation: 0,
    }
}

fn log_uniform(rng: &mut Rng, (lo, hi): (f64, f64)) -> f64 {
    (lo.ln() + rng.uniform() * (hi.ln() - lo.ln())).exp()
}

fn random_star(k: &Kerr, cfg: &ClusterConfig, rng: &mut Rng) -> (Launch, BodyParams) {
    let launch = orbit::random_cluster_orbit(k, rng, cfg.r_min, cfg.r_max, cfg.cusp_gamma, cfg.r_min * 0.8);
    let (mass, temperature, luminosity, radius) = match cfg.star_model {
        StarModel::Stylized => {
            let mass = log_uniform(rng, cfg.star_mass);
            // Heavier stars run hotter and brighter (loosely main-sequence-like).
            let x = (mass.ln() - cfg.star_mass.0.ln()) / (cfg.star_mass.1.ln() - cfg.star_mass.0.ln());
            let temperature = 2800.0 * (14.0f64).powf(x * (0.7 + 0.3 * rng.uniform()));
            (mass, temperature, 0.25 + 3.0 * x * x + 0.3 * rng.uniform(), 0.0)
        }
        StarModel::Physical { mass_msun, giant_fraction } => physical_star(rng, mass_msun, giant_fraction),
    };
    let params = BodyParams {
        kind: BodyKind::Star,
        mass,
        temperature,
        luminosity,
        radius,
        feels_perturbations: true,
        radiates: false,
    };
    (launch, params)
}

/// A star drawn from a Salpeter mass function (dN/dm ∝ m^−2.35) with
/// main-sequence mass–radius and mass–luminosity relations, or with
/// probability `giant_fraction` a red giant. Returns (mass in M,
/// temperature K, luminosity L☉, radius in M).
fn physical_star(rng: &mut Rng, (lo, hi): (f64, f64), giant_fraction: f64) -> (f64, f64, f64, f64) {
    let e = -1.35;
    let m = (lo.powf(e) + rng.uniform() * (hi.powf(e) - lo.powf(e))).powf(1.0 / e);
    let (radius_rsun, lum) = if rng.uniform() < giant_fraction {
        let r = rng.range(10.0, 60.0);
        let t: f64 = rng.range(3600.0, 4800.0);
        (r, r * r * (t / 5772.0).powi(4))
    } else {
        let r = if m < 1.0 { m.powf(0.8) } else { m.powf(0.57) };
        let l = if m < 0.43 {
            0.23 * m.powf(2.3)
        } else if m < 2.0 {
            m.powi(4)
        } else {
            1.4 * m.powf(3.5)
        };
        (r, l)
    };
    let temperature = 5772.0 * (lum / (radius_rsun * radius_rsun)).powf(0.25);
    (m * units::MSUN, temperature, lum, radius_rsun * units::RSUN)
}

fn random_compact(k: &Kerr, cfg: &ClusterConfig, rng: &mut Rng) -> (Launch, BodyParams) {
    let r = rng.range(cfg.compact_radius.0, cfg.compact_radius.1);
    let launch = orbit::inclined_circular(
        k,
        r,
        rng.range(0.0, 0.9),
        rng.range(0.0, std::f64::consts::TAU),
        rng.range(0.0, std::f64::consts::TAU),
    );
    let params = BodyParams {
        kind: BodyKind::Compact,
        mass: log_uniform(rng, cfg.compact_mass),
        temperature: 60_000.0,
        luminosity: 1.5,
        radius: 0.0,
        feels_perturbations: true,
        radiates: true,
    };
    (launch, params)
}

/// 2.5PN radiation-reaction acceleration (Iyer & Will 1993, harmonic gauge)
/// of a body of mass `q·M` relative to the hole:
///
/// `a = (8/5) η m²/r³ [(3v² + 17m/(3r)) ṙ n − (v² + 3m/r) v]`
///
/// with `m = M(1 + q)`, `η = q/(1 + q)²`. For circular orbits it reproduces
/// the Peters–Mathews energy flux `dE/dt = −(32/5) η² m⁵ / r⁵`.
pub fn radiation_reaction_accel(k: &Kerr, x: V3, u: [f64; 4], q: f64) -> V3 {
    let m = k.m * (1.0 + q);
    let eta = q / ((1.0 + q) * (1.0 + q));
    let v = [u[1] / u[0], u[2] / u[0], u[3] / u[0]];
    let r = vec3::norm(x);
    let n = vec3::scale(x, 1.0 / r);
    let v2 = vec3::dot(v, v);
    let rdot = vec3::dot(n, v);
    let c = 1.6 * eta * m * m / (r * r * r);
    let a_n = c * (3.0 * v2 + 17.0 * m / (3.0 * r)) * rdot;
    let a_v = -c * (v2 + 3.0 * m / r);
    vec3::add(vec3::scale(n, a_n), vec3::scale(v, a_v))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn station(launch: Launch) -> Placement {
        Placement {
            launch,
            params: BodyParams {
                kind: BodyKind::Station,
                mass: 0.0,
                temperature: 6500.0,
                luminosity: 1.0,
                radius: 0.0,
                feels_perturbations: false,
                radiates: false,
            },
        }
    }

    fn small_cfg() -> ClusterConfig {
        ClusterConfig { stars: 24, compact_objects: 2, history_len: 64, ..Default::default() }
    }

    #[test]
    fn stations_follow_exact_geodesics() {
        let k = Kerr::new(1.0, 0.9);
        let launch = orbit::circular_equatorial(&k, 40.0, 0.0, true);
        let mut c = Cluster::new(k, small_cfg(), &[station(launch)]);
        let e0 = c.energy(0);
        c.advance_to(300.0);
        let r = k.radius(c.bodies[0].position());
        assert!((r - 40.0).abs() < 1e-6, "station drifted to r = {r}");
        assert!((c.energy(0) - e0).abs() < 1e-10);
        // History reproduces the analytic circular orbit.
        let om = orbit::circular_omega(&k, 40.0, true);
        for t in [150.0, 217.3, 280.0, 299.0] {
            let s = c.sample_at(0, t).unwrap();
            let expect = orbit::circular_equatorial(&k, 40.0, om * t, true).pos;
            assert!(vec3::norm(vec3::sub(s.pos, expect)) < 1e-4, "t = {t}");
        }
    }

    #[test]
    fn perturbations_act_but_stay_weak() {
        let k = Kerr::new(1.0, 0.5);
        let mut c = Cluster::new(k, small_cfg(), &[]);
        let any = c.bodies.iter().any(|b| b.pert != [0.0; 3]);
        assert!(any, "some body should feel a pull");
        for b in &c.bodies {
            let r = vec3::norm(b.position());
            // Much weaker than the hole's own pull.
            assert!(vec3::norm(b.pert) < 0.2 / (r * r), "{:?}", b.pert);
        }
        c.advance_to(50.0);
        assert!((c.t - 50.0).abs() < 1e-12);
    }

    #[test]
    fn retarded_source_of_uniform_motion_is_present_position() {
        // With M = 0 the space is flat and bodies move uniformly, so the
        // velocity-extrapolated retarded position is exactly the present one.
        let k = Kerr::new(0.0, 0.0);
        let cfg = ClusterConfig { stars: 0, compact_objects: 0, history_len: 64, ..Default::default() };
        let mk = |pos: V3, vel: V3| station(Launch { pos, vel });
        let mut c = Cluster::new(k, cfg, &[mk([20.0, 0.0, 0.0], [0.0, 0.3, 0.1]), mk([-5.0, 3.0, 0.0], [0.0; 3])]);
        c.advance_to(40.0);
        let (x, t_ret) = c.retarded_source(0, c.t, [-5.0, 3.0, 0.0]).unwrap();
        let present = c.bodies[0].position();
        assert!(vec3::norm(vec3::sub(x, present)) < 1e-6);
        let emitted = c.sample_at(0, t_ret).unwrap().pos;
        let light_time = vec3::norm(vec3::sub(emitted, [-5.0, 3.0, 0.0]));
        assert!((c.t - t_ret - light_time).abs() < 1e-6);
    }

    #[test]
    fn radiation_reaction_matches_peters_for_circular_orbits() {
        let k = Kerr::new(1.0, 0.0);
        let r = 200.0;
        let q = 1e-3;
        let launch = orbit::circular_equatorial(&k, r, 0.0, true);
        let u = k.four_velocity(launch.pos, launch.vel).unwrap();
        let a = radiation_reaction_accel(&k, launch.pos, u, q);
        // Power per unit reduced mass, compared with Peters' formula.
        let de_dt = vec3::dot(a, launch.vel);
        let m: f64 = 1.0 + q;
        let eta = q / (m * m);
        let peters = -(32.0 / 5.0) * eta * m.powi(4) / r.powi(5);
        assert!((de_dt / peters - 1.0).abs() < 0.03, "{de_dt} vs {peters}");
    }

    #[test]
    fn compact_objects_inspiral() {
        let k = Kerr::new(1.0, 0.7);
        let cfg = ClusterConfig {
            stars: 0,
            compact_objects: 0,
            history_len: 16,
            radiation_reaction: true,
            ..Default::default()
        };
        let launch = orbit::circular_equatorial(&k, 10.0, 0.0, true);
        let mut p = station(launch);
        p.params.kind = BodyKind::Compact;
        p.params.radiates = true;
        p.params.mass = 0.02;
        let mut c = Cluster::new(k, cfg, &[p]);
        let e0 = c.energy(0);
        c.advance_to(2000.0);
        let r = k.radius(c.bodies[0].position());
        assert!(c.energy(0) < e0, "energy must be radiated away");
        assert!(r < 10.0 - 0.05, "orbit should shrink, r = {r}");
    }
}
