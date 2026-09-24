//! Per-frame orchestration: the pilot's proper time drives everything.
//!
//! Each frame the wall-clock interval becomes an interval of the pilot's
//! proper time `Δτ`. The pilot's worldline is integrated over `Δτ`, which
//! yields the coordinate-time interval `Δt` (≈ `u^t Δτ`), and the whole
//! cluster is then advanced to the pilot's new coordinate time. High speed
//! or a deep dive makes `u^t` large and the universe visibly fast-forwards.
//! If `Δt` would exceed the per-frame compute budget, `Δτ` is reduced
//! instead (the "clock limiter"): physics is never approximated, the pilot's
//! own clock just runs slower than the wall clock.

use crate::cluster::{BodyKind, BodyParams, Cluster, ClusterConfig, ClusterEvent, Placement};
use crate::geodesic;
use crate::lensing::{self, ImageSearch, Observer, RayConfig};
use crate::metric::Kerr;
use crate::orbit;
use crate::pilot::{Command, FORWARD, LEFT, Pilot, UP};
use crate::vec3::{self, V3};

#[derive(Clone, Copy, Debug)]
pub struct StationSpec {
    pub radius: f64,
    pub inclination: f64,
    pub node: f64,
    pub phase: f64,
    /// Half-size of the station's panel in units of `M`.
    pub size: f64,
}

#[derive(Clone, Debug)]
pub struct WorldConfig {
    pub spin: f64,
    pub cluster: ClusterConfig,
    pub stations: Vec<StationSpec>,
    /// Pilot proper time (units of `M`) per wall-clock second.
    pub time_scale: f64,
    /// Compute budget: coordinate time the cluster may advance per frame.
    pub max_dt_per_frame: f64,
    /// Proper acceleration of the main engine, units of `1/M`.
    pub thrust: f64,
    pub boost_factor: f64,
    /// Maximum turn rate, radians per wall second.
    pub turn_rate: f64,
    /// The autopilot parks this far in front of a station's card.
    pub dock_standoff: f64,
    /// Docking tolerance around the parking point.
    pub dock_distance: f64,
    pub dock_speed: f64,
    /// While close to or docked at a station, aim this far to its right so
    /// the card sits left of the content panel.
    pub dock_view_yaw: f64,
    pub autopilot_cruise: f64,
    pub start_station: usize,
    /// Without stations: radius of the circular orbit the ship starts on.
    pub start_radius: f64,
    /// Rendering: point-source flux (L☉/M² for physical stars) that maps to
    /// the faintest visible star.
    pub star_flux_ref: f64,
    /// Rendering: adapt exposure to the brightest stars in view, like an eye
    /// or a camera, instead of a fixed dark-adapted exposure.
    pub auto_exposure: bool,
}

impl Default for WorldConfig {
    fn default() -> Self {
        Self {
            spin: 0.94,
            cluster: ClusterConfig::default(),
            stations: default_stations(6),
            time_scale: 8.0,
            max_dt_per_frame: 12.0,
            thrust: 0.012,
            boost_factor: 6.0,
            turn_rate: 1.4,
            dock_standoff: 2.6,
            dock_distance: 0.5,
            dock_speed: 0.01,
            dock_view_yaw: 0.32,
            autopilot_cruise: 0.45,
            start_station: 0,
            start_radius: 60.0,
            // Tuned by eye for the stylized cluster.
            star_flux_ref: 1.8e-6,
            auto_exposure: false,
        }
    }
}

impl WorldConfig {
    /// Sagittarius A* at its real scale: 4.3 million solar masses, real
    /// stars (Salpeter masses, main-sequence and giant radii and
    /// luminosities) from ~100 AU out to ~0.1 pc, stellar-mass black holes
    /// that barely inspiral, and no stations. From most places the hole is
    /// smaller than a pixel; you see it by what it does to the light behind
    /// it, or by flying in.
    pub fn sgr_a(stars: usize, seed: u64) -> Self {
        use crate::units::{AU, MSUN, SECONDS_PER_M, flux_of_magnitude};
        Self {
            spin: 0.9,
            cluster: ClusterConfig {
                seed,
                stars,
                compact_objects: 8,
                r_min: 100.0 * AU,
                r_max: 20_000.0 * AU,
                cusp_gamma: 1.75,
                star_mass: (0.5 * MSUN, 40.0 * MSUN),
                star_model: crate::cluster::StarModel::Physical { mass_msun: (0.5, 40.0), giant_fraction: 0.08 },
                compact_mass: (5.0 * MSUN, 30.0 * MSUN),
                compact_radius: (150.0 * AU, 1500.0 * AU),
                history_dt: 2000.0,
                history_len: 512,
                softening: 2.0 * AU,
                source_mass_min: 15.0 * MSUN,
                max_sources: 16,
                // Pulls between real stars change on orbital timescales of
                // ~10⁵ M or more; refreshing every ~3×10⁴ M is plenty.
                perturbation_every: 16,
                radiation_reaction: true,
                escape_radius: 200_000.0 * AU,
            },
            stations: Vec::new(),
            // 1000× real time: a year of ship time is ~9 hours of wall time.
            time_scale: 1000.0 / SECONDS_PER_M,
            max_dt_per_frame: 40_000.0,
            thrust: 0.012,
            boost_factor: 5.0,
            start_radius: 800.0 * AU,
            // Faintest visible: apparent magnitude 7, a little past what the
            // eye sees on Earth.
            star_flux_ref: flux_of_magnitude(7.0),
            auto_exposure: true,
            ..Self::default()
        }
    }
}

/// A spread of stable, mildly inclined orbits between r = 30 and 80 M.
pub fn default_stations(n: usize) -> Vec<StationSpec> {
    (0..n)
        .map(|i| {
            let f = if n > 1 { i as f64 / (n - 1) as f64 } else { 0.0 };
            StationSpec {
                radius: 30.0 + 50.0 * f,
                inclination: [0.0, 0.35, -0.25, 0.6, 0.15, -0.5, 0.45, 0.8][i % 8],
                node: 1.9 * i as f64,
                phase: 2.4 * i as f64 + 0.5,
                size: 1.3,
            }
        })
        .collect()
}

/// Player input for one frame (all components in `[-1, 1]`).
#[derive(Clone, Copy, Debug, Default)]
pub struct Input {
    /// Thrust along (forward, left, up).
    pub thrust: V3,
    /// Rotation about (forward, left, up): roll, pitch, yaw.
    pub turn: V3,
    pub boost: bool,
    /// Null the velocity relative to the nearest station.
    pub brake: bool,
    /// Station index to fly to, or `-1`.
    pub autopilot: i32,
    pub undock: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PilotStatus {
    Free,
    Autopilot(usize),
    Docked { station: usize, offset: V3 },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WorldEvent {
    HorizonCrossed,
    Docked(usize),
    Undocked(usize),
    StarCaptured(usize),
}

/// Where a station appears on the pilot's sky right now.
#[derive(Clone, Copy, Debug, Default)]
pub struct StationView {
    pub visible: bool,
    /// Ship-frame direction (forward, left, up) of the primary image.
    pub dir: V3,
    /// Angular radius of the panel, radians.
    pub angular_radius: f64,
    pub g: f64,
    /// Relative point-source flux (`g⁴ / area`, normalized to 1 at 10 M).
    pub flux: f64,
    /// Present distance measured in the ship frame.
    pub distance: f64,
    /// Emission event and the station's state then (for in-scene panels).
    pub t_emit: f64,
    pub emit_pos: V3,
    pub emit_vel: V3,
    /// Coordinate light-travel time of the image.
    pub delay: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Telemetry {
    pub tau: f64,
    pub t: f64,
    pub r: f64,
    pub dt_dtau: f64,
    pub gamma: f64,
    pub speed: f64,
    /// Fraction of the requested proper time actually simulated this frame.
    pub clock_rate: f64,
    pub status: i32,
    pub station: i32,
    pub station_distance: f64,
    pub station_speed: f64,
    pub r_plus: f64,
    pub spin: f64,
    pub alive_bodies: usize,
    /// Proper time until the ship reaches the horizon if it coasts on its
    /// current (straight-line) heading, or infinity when it would miss.
    pub impact_in: f64,
}

pub struct World {
    pub kerr: Kerr,
    pub cfg: WorldConfig,
    pub cluster: Cluster,
    pub pilot: Pilot,
    pub status: PilotStatus,
    pub views: Vec<StationView>,
    pub events: Vec<WorldEvent>,
    pub ray: RayConfig,
    search: ImageSearch,
    /// Last image directions as coordinate vectors, for warm starts.
    guesses: Vec<Option<V3>>,
    last_clock_rate: f64,
}

pub const STATION_TEMPERATURE: f64 = 6500.0;

/// Longest proper-time interval over which a guidance command is held.
const CONTROL_STEP: f64 = 0.4;

impl World {
    pub fn new(cfg: WorldConfig) -> Self {
        let kerr = Kerr::new(1.0, cfg.spin);
        let placements: Vec<Placement> = cfg
            .stations
            .iter()
            .map(|s| Placement {
                launch: orbit::inclined_circular(&kerr, s.radius, s.inclination, s.node, s.phase),
                params: BodyParams {
                    kind: BodyKind::Station,
                    mass: 0.0,
                    temperature: STATION_TEMPERATURE,
                    luminosity: 1.0,
                    radius: 0.0,
                    feels_perturbations: false,
                    radiates: false,
                },
            })
            .collect();
        let cluster = Cluster::new(kerr, cfg.cluster.clone(), &placements);
        let n = cfg.stations.len();
        let pilot = start_pilot(&kerr, &cluster, cfg.start_station.min(n.saturating_sub(1)), n, cfg.start_radius);
        Self {
            kerr,
            cluster,
            pilot,
            status: PilotStatus::Free,
            views: vec![StationView::default(); n],
            events: Vec::new(),
            ray: RayConfig::default(),
            search: ImageSearch { max_iter: 6, ..Default::default() },
            guesses: vec![None; n],
            last_clock_rate: 1.0,
            cfg,
        }
    }

    pub fn station_count(&self) -> usize {
        self.cfg.stations.len()
    }

    pub fn observer(&self) -> Observer {
        Observer { x: self.pilot.x, e: self.pilot.e }
    }

    /// Advance one frame of `wall_dt` seconds.
    pub fn step(&mut self, wall_dt: f64, input: &Input) {
        self.events.clear();
        let wall_dt = wall_dt.clamp(0.0, 0.1);
        let mut dtau = wall_dt * self.cfg.time_scale;
        let requested = dtau;

        if input.autopilot >= 0 && (input.autopilot as usize) < self.station_count() {
            match self.status {
                PilotStatus::Docked { station, .. } if station == input.autopilot as usize => {}
                _ => self.status = PilotStatus::Autopilot(input.autopilot as usize),
            }
        } else if input.autopilot < 0 && matches!(self.status, PilotStatus::Autopilot(_)) {
            self.status = PilotStatus::Free;
        }
        if input.undock
            && let PilotStatus::Docked { station, .. } = self.status
        {
            self.status = PilotStatus::Free;
            self.events.push(WorldEvent::Undocked(station));
        }

        // Clock limiter.
        let dt_dtau = self.pilot.time_dilation().max(1.0);
        if dtau * dt_dtau > self.cfg.max_dt_per_frame {
            dtau = self.cfg.max_dt_per_frame / dt_dtau;
        }
        self.last_clock_rate = if requested > 0.0 { dtau / requested } else { 1.0 };

        match self.status {
            PilotStatus::Docked { station, offset } => {
                let w = geodesic::four_velocity(&self.kerr, &self.cluster.bodies[station].state);
                let t_new = self.pilot.x[0] + w[0] * dtau;
                self.cluster.advance_to(t_new);
                let pos = vec3::add(
                    self.cluster.bodies[station].position(),
                    from_orbital_frame(&self.kerr, &self.cluster.bodies[station].state, offset),
                );
                let w = geodesic::four_velocity(&self.kerr, &self.cluster.bodies[station].state);
                self.pilot.x = [self.cluster.t, pos[0], pos[1], pos[2]];
                self.pilot.set_velocity(&self.kerr, w);
                self.pilot.tau += dtau;
                let per_tau = wall_dt / dtau.max(1e-9);
                let spin = if input.turn == [0.0; 3] {
                    let (d, _) = self.relative(station);
                    self.aim(d, per_tau)
                } else {
                    vec3::scale(input.turn, self.cfg.turn_rate * per_tau)
                };
                self.pilot.rotate(spin, dtau);
            }
            _ => {
                // Guidance is re-evaluated at least every CONTROL_STEP of
                // proper time so it stays stable at high time scales.
                let guided = input.brake || matches!(self.status, PilotStatus::Autopilot(_));
                let chunks = if guided { (dtau / CONTROL_STEP).ceil().max(1.0) as usize } else { 1 };
                let h = dtau / chunks as f64;
                for _ in 0..chunks {
                    let cmd = self.command(input, wall_dt / chunks as f64, h);
                    self.pilot.step(&self.kerr, &cmd, h);
                    self.cluster.advance_to(self.pilot.x[0]);
                    self.check_docking();
                    if matches!(self.status, PilotStatus::Docked { .. }) {
                        break;
                    }
                }
            }
        }

        for ev in self.cluster.events.drain(..) {
            if let ClusterEvent::Captured { body, .. } = ev {
                self.events.push(WorldEvent::StarCaptured(body));
            }
        }

        let r = self.kerr.radius(self.pilot.position());
        if r < self.kerr.r_plus() * 1.02 {
            self.events.push(WorldEvent::HorizonCrossed);
            self.respawn();
        }
        self.update_views();
    }

    fn respawn(&mut self) {
        let n = self.station_count();
        let s = self.cfg.start_station.min(n.saturating_sub(1));
        let t = self.cluster.t;
        let mut p = start_pilot(&self.kerr, &self.cluster, s, n, self.cfg.start_radius);
        p.x[0] = t;
        p.tau = self.pilot.tau;
        self.pilot = p;
        self.status = PilotStatus::Free;
        self.guesses.iter_mut().for_each(|g| *g = None);
    }

    /// 4-velocity of the observer at rest in the t = const slicing.
    fn normal_observer(&self) -> [f64; 4] {
        let t = self.kerr.terms(self.pilot.position());
        let alpha = 1.0 / (1.0 + t.f).sqrt();
        [alpha * (1.0 + t.f), -alpha * t.f * t.l[0], -alpha * t.f * t.l[1], -alpha * t.f * t.l[2]]
    }

    fn nearest_station(&self) -> Option<(usize, f64)> {
        let pos = self.pilot.position();
        (0..self.station_count())
            .map(|i| (i, vec3::norm(vec3::sub(self.cluster.bodies[i].position(), pos))))
            .min_by(|a, b| a.1.total_cmp(&b.1))
    }

    /// Ship-frame offset and relative velocity of a station (velocity of
    /// the ship relative to the station).
    fn relative(&self, i: usize) -> (V3, V3) {
        let b = &self.cluster.bodies[i];
        let d = self.pilot.local_components(&self.kerr, vec3::sub(b.position(), self.pilot.position()));
        let w = geodesic::four_velocity(&self.kerr, &b.state);
        let v = vec3::scale(self.pilot.relative_velocity(&self.kerr, w), -1.0);
        (d, v)
    }

    fn command(&mut self, input: &Input, wall_dt: f64, dtau: f64) -> Command {
        let a_max = self.cfg.thrust * if input.boost { self.cfg.boost_factor } else { 1.0 };
        // Turn rates are specified per wall second so steering feels the same
        // regardless of how fast proper time runs.
        let per_tau = if dtau > 0.0 { wall_dt / dtau } else { 0.0 };
        let mut spin = vec3::scale(input.turn, self.cfg.turn_rate * per_tau);
        let mut accel = vec3::scale(input.thrust, a_max);

        if input.brake {
            // Null the velocity relative to the nearest station, or without
            // stations relative to the local rest frame (the normal
            // observer of the t = const slicing).
            let v = match self.nearest_station() {
                Some((i, _)) => self.relative(i).1,
                None => vec3::scale(self.pilot.relative_velocity(&self.kerr, self.normal_observer()), -1.0),
            };
            // Brake over a few proper-time units, without exceeding boost thrust.
            let rapidity = vec3::norm(v).min(0.999_999_999).atanh();
            let dir = vec3::scale(v, -1.0 / vec3::norm(v).max(1e-300));
            let a_brake = self.cfg.thrust * self.cfg.boost_factor;
            accel = vec3::scale(dir, (rapidity * 0.5).min(a_brake));
        }

        if let PilotStatus::Autopilot(i) = self.status {
            let (d, v) = self.relative(i);
            // Park in front of the station's card, on the line of sight.
            let dist = vec3::norm(d).max(1e-9);
            let to_park = vec3::scale(d, 1.0 - self.cfg.dock_standoff / dist);
            let park_dist = vec3::norm(to_park);
            let a_ap = self.cfg.thrust * self.cfg.boost_factor;
            // Brake so as to arrive at rest: v = √(2 a s), with margin.
            let v_des = self.cfg.autopilot_cruise.min((1.2 * a_ap * park_dist).sqrt());
            let dv = vec3::sub(vec3::scale(to_park, v_des / park_dist.max(1e-9)), v);
            accel = clamp_len(vec3::scale(dv, 0.6), a_ap);
            if input.turn == [0.0; 3] {
                spin = self.aim(d, per_tau);
            }
        }
        Command { accel, spin }
    }

    /// Spin that turns the nose towards ship-frame direction `d` (offset
    /// to the right when close, see [`WorldConfig::dock_view_yaw`]).
    fn aim(&self, d: V3, per_tau: f64) -> V3 {
        let dist = vec3::norm(d);
        let near = (1.0 - (dist - self.cfg.dock_standoff) / 10.0).clamp(0.0, 1.0);
        let dir = vec3::rotate(vec3::scale(d, 1.0 / dist.max(1e-9)), [0.0, 0.0, 1.0], -self.cfg.dock_view_yaw * near);
        let turn = vec3::cross([1.0, 0.0, 0.0], dir);
        let ang = vec3::norm(turn).atan2(dir[0]);
        let max_rate = self.cfg.turn_rate * per_tau;
        let rate = (2.0 * ang * per_tau).min(max_rate);
        if vec3::norm(turn) > 1e-9 {
            vec3::scale(vec3::normalize(turn), rate)
        } else if dir[0] < 0.0 {
            [0.0, 0.0, max_rate]
        } else {
            [0.0; 3]
        }
    }

    fn check_docking(&mut self) {
        let candidates: Vec<usize> = match self.status {
            PilotStatus::Autopilot(i) => vec![i],
            PilotStatus::Free => (0..self.station_count()).collect(),
            PilotStatus::Docked { .. } => return,
        };
        for i in candidates {
            let (d, v) = self.relative(i);
            let park_error = (vec3::norm(d) - self.cfg.dock_standoff).abs();
            if park_error < self.cfg.dock_distance && vec3::norm(v) < self.cfg.dock_speed {
                let st = &self.cluster.bodies[i];
                let offset = to_orbital_frame(&self.kerr, &st.state, vec3::sub(self.pilot.position(), st.position()));
                self.status = PilotStatus::Docked { station: i, offset };
                let w = geodesic::four_velocity(&self.kerr, &self.cluster.bodies[i].state);
                self.pilot.set_velocity(&self.kerr, w);
                self.events.push(WorldEvent::Docked(i));
                return;
            }
        }
    }

    /// Solve for each station's primary image on the pilot's past light cone.
    fn update_views(&mut self) {
        let obs = self.observer();
        let k = self.kerr;
        let t_min = self.cluster.history.t_oldest();
        for i in 0..self.station_count() {
            let cluster = &self.cluster;
            let target = |t: f64| cluster.sample_at(i, t);
            let guess = match self.guesses[i] {
                Some(v) => coord_to_local(&obs, &k, v),
                None => lensing::image_guess(&k, &obs, cluster.bodies[i].position(), 0),
            };
            let img = lensing::find_image(&k, &obs, &target, guess, 0, t_min, &self.ray, &self.search);
            let (d, _) = self.relative(i);
            let size = self.cfg.stations[i].size;
            self.views[i] = match img {
                Some(img) => {
                    self.guesses[i] = Some(local_to_coord(&obs, img.dir));
                    let area = img.area.max(1e-12);
                    StationView {
                        visible: true,
                        dir: img.dir,
                        angular_radius: (size / area.sqrt()).min(1.5),
                        g: img.g,
                        flux: img.g.powi(4) * 100.0 / area,
                        distance: vec3::norm(d),
                        t_emit: img.t_emit,
                        emit_pos: img.body.pos,
                        emit_vel: img.body.vel,
                        delay: obs.x[0] - img.t_emit,
                    }
                }
                None => {
                    self.guesses[i] = None;
                    StationView { visible: false, distance: vec3::norm(d), ..Default::default() }
                }
            };
        }
    }

    pub fn telemetry(&self) -> Telemetry {
        let gamma = self.pilot.normal_gamma(&self.kerr);
        let (status, station) = match self.status {
            PilotStatus::Free => (0, -1),
            PilotStatus::Autopilot(i) => (1, i as i32),
            PilotStatus::Docked { station, .. } => (2, station as i32),
        };
        let (sd, sv) = match self.status {
            PilotStatus::Free => self.nearest_station().map(|(i, _)| self.relative(i)).unwrap_or_default(),
            PilotStatus::Autopilot(i) | PilotStatus::Docked { station: i, .. } => self.relative(i),
        };
        Telemetry {
            tau: self.pilot.tau,
            t: self.pilot.x[0],
            r: self.kerr.radius(self.pilot.position()),
            dt_dtau: self.pilot.time_dilation(),
            gamma,
            speed: (1.0 - 1.0 / (gamma * gamma)).max(0.0).sqrt(),
            clock_rate: self.last_clock_rate,
            status,
            station,
            station_distance: vec3::norm(sd),
            station_speed: vec3::norm(sv),
            r_plus: self.kerr.r_plus(),
            spin: self.kerr.a,
            alive_bodies: self.cluster.bodies.iter().filter(|b| b.alive).count(),
            impact_in: self.impact_in(),
        }
    }

    /// Straight-line estimate of the proper time left before the horizon.
    /// Anything passing within a few M of the centre is treated as a hit
    /// (photons are captured below an impact parameter of about 5 M).
    fn impact_in(&self) -> f64 {
        let u = self.pilot.e[0];
        let pos = self.pilot.position();
        let v = [u[1] / u[0], u[2] / u[0], u[3] / u[0]];
        let speed = vec3::norm(v);
        let along = -vec3::dot(pos, v) / speed.max(1e-12);
        if along <= 0.0 {
            return f64::INFINITY;
        }
        let miss = vec3::norm(vec3::add(pos, vec3::scale(v, along / speed)));
        if miss > 6.0 * self.kerr.m {
            return f64::INFINITY;
        }
        (along / speed) / self.pilot.time_dilation().max(1.0)
    }
}

/// Radial / along-track / orbit-normal basis of a body, so a docked ship
/// keeps its place relative to the station as the station orbits.
fn orbital_frame(k: &Kerr, s: &geodesic::BodyState) -> [V3; 3] {
    let p = geodesic::position(s);
    let v = geodesic::coordinate_velocity(k, s);
    let r = vec3::normalize(p);
    let n = vec3::normalize(vec3::cross(p, v));
    [r, vec3::cross(n, r), n]
}

fn to_orbital_frame(k: &Kerr, s: &geodesic::BodyState, d: V3) -> V3 {
    let f = orbital_frame(k, s);
    [vec3::dot(d, f[0]), vec3::dot(d, f[1]), vec3::dot(d, f[2])]
}

fn from_orbital_frame(k: &Kerr, s: &geodesic::BodyState, c: V3) -> V3 {
    let f = orbital_frame(k, s);
    vec3::add(vec3::add(vec3::scale(f[0], c[0]), vec3::scale(f[1], c[1])), vec3::scale(f[2], c[2]))
}

fn clamp_len(v: V3, max: f64) -> V3 {
    let n = vec3::norm(v);
    if n > max { vec3::scale(v, max / n) } else { v }
}

/// Coordinate-space vector `Σ n_a e_a` (spatial part) for warm starts.
fn local_to_coord(obs: &Observer, n: V3) -> V3 {
    let e = &obs.e;
    let mut v = [0.0; 3];
    for i in 0..3 {
        v[i] = n[0] * e[FORWARD][i + 1] + n[1] * e[LEFT][i + 1] + n[2] * e[UP][i + 1];
    }
    v
}

fn coord_to_local(obs: &Observer, k: &Kerr, v: V3) -> V3 {
    let p = obs.position();
    let w = [0.0, v[0], v[1], v[2]];
    // Remove the component along u so we measure a purely spatial direction.
    let n = [k.dot(p, w, obs.e[FORWARD]), k.dot(p, w, obs.e[LEFT]), k.dot(p, w, obs.e[UP])];
    vec3::normalize(n)
}

/// Park the ship a short way outside station `s`, co-moving with it and
/// looking past it towards the hole.
fn start_pilot(k: &Kerr, cluster: &Cluster, s: usize, n_stations: usize, radius: f64) -> Pilot {
    if n_stations == 0 {
        // A circular orbit slightly above the equator, facing prograde with
        // the hole off to the side: thrusting straight ahead at realistic
        // scale would otherwise dive into it within seconds.
        let pos = [radius, 0.0, 0.1 * radius];
        let v = (k.m / vec3::norm(pos)).sqrt();
        let look = vec3::normalize([-0.35, 1.0, -0.05]);
        return Pilot::new(k, pos, [0.0, v, 0.0], look, [0.0, 0.0, 1.0]).expect("valid start");
    }
    let st = &cluster.bodies[s];
    let sp = st.position();
    let radial = vec3::normalize(sp);
    let vel = geodesic::coordinate_velocity(k, &st.state);
    let normal = vec3::normalize(vec3::cross(sp, vel));
    let pos = vec3::add(sp, vec3::add(vec3::scale(radial, 7.0), vec3::scale(normal, 1.6)));
    let look = vec3::normalize(vec3::sub(vec3::scale(sp, 0.0), pos));
    // Aim between the station and the hole.
    let look = vec3::normalize(vec3::add(look, vec3::scale(vec3::normalize(vec3::sub(sp, pos)), 0.6)));
    // Match the station's velocity so it hangs in view.
    Pilot::new(k, pos, vel, look, normal).expect("valid start")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_world() -> World {
        let cfg = WorldConfig {
            cluster: ClusterConfig { stars: 32, compact_objects: 2, history_len: 128, ..Default::default() },
            stations: default_stations(3),
            ..Default::default()
        };
        World::new(cfg)
    }

    #[test]
    fn full_thrust_ahead_from_the_realistic_start_misses_the_hole() {
        let mut w = World::new(WorldConfig::sgr_a(16, 1));
        assert!(w.telemetry().impact_in.is_infinite());
        let input = Input { thrust: [1.0, 0.0, 0.0], autopilot: -1, ..Default::default() };
        for _ in 0..900 {
            w.step(1.0 / 60.0, &input);
            assert!(!w.events.contains(&WorldEvent::HorizonCrossed));
        }
        assert!(w.telemetry().gamma > 50.0);
        // Pointing straight at the hole is flagged.
        let mut dive = World::new(WorldConfig::sgr_a(16, 1));
        let inward = vec3::scale(vec3::normalize(dive.pilot.position()), -0.5);
        let p = Pilot::new(&dive.kerr, dive.pilot.position(), inward, inward, [0.0, 0.0, 1.0]).unwrap();
        dive.pilot = p;
        assert!(dive.telemetry().impact_in.is_finite());
    }

    #[test]
    fn proper_time_drives_the_clock() {
        let mut w = small_world();
        let t0 = w.pilot.x[0];
        w.step(1.0 / 60.0, &Input { autopilot: -1, ..Default::default() });
        let tel = w.telemetry();
        assert!(tel.tau > 0.0);
        assert!((w.cluster.t - w.pilot.x[0]).abs() < 1e-9);
        // Coordinate time runs at least as fast as proper time here.
        assert!(w.pilot.x[0] - t0 >= tel.tau * 0.99);
    }

    #[test]
    fn start_station_is_in_view() {
        let mut w = small_world();
        w.step(1.0 / 60.0, &Input { autopilot: -1, ..Default::default() });
        let v = &w.views[0];
        assert!(v.visible, "station image should be found");
        assert!(v.dir[0] > 0.5, "station should be ahead: {:?}", v.dir);
        assert!(v.delay > 0.0 && v.delay < 20.0);
    }

    #[test]
    fn autopilot_docks() {
        let mut w = small_world();
        let input = Input { autopilot: 0, ..Default::default() };
        let mut docked = false;
        for _ in 0..(60 * 120) {
            w.step(1.0 / 60.0, &input);
            if matches!(w.status, PilotStatus::Docked { station: 0, .. }) {
                docked = true;
                break;
            }
        }
        assert!(docked, "telemetry: {:?}", w.telemetry());
        // Docked ships ride along with the station.
        let before = w.telemetry().station_distance;
        for _ in 0..120 {
            w.step(1.0 / 60.0, &input);
        }
        let after = w.telemetry();
        assert!((after.station_distance / before - 1.0).abs() < 1e-3);
        assert!(after.station_speed < 1e-6);
    }

    #[test]
    fn autopilot_docks_at_high_time_scale() {
        let mut w = small_world();
        w.cfg.time_scale = 40.0;
        let input = Input { autopilot: 1, ..Default::default() };
        let docked = (0..20 * 150).any(|_| {
            w.step(1.0 / 20.0, &input);
            matches!(w.status, PilotStatus::Docked { station: 1, .. })
        });
        assert!(docked, "telemetry: {:?}", w.telemetry());
        let t = w.telemetry();
        assert!((t.station_distance - w.cfg.dock_standoff).abs() < w.cfg.dock_distance);
    }

    #[test]
    fn runs_without_stations() {
        let cfg = WorldConfig {
            cluster: ClusterConfig { stars: 16, compact_objects: 1, history_len: 64, ..Default::default() },
            stations: Vec::new(),
            ..Default::default()
        };
        let mut w = World::new(cfg);
        let input = Input { thrust: [1.0, 0.0, 0.0], autopilot: 0, brake: true, ..Default::default() };
        for _ in 0..120 {
            w.step(1.0 / 60.0, &input);
        }
        assert!(w.views.is_empty());
        assert_eq!(w.status, PilotStatus::Free);
        assert!(w.telemetry().tau > 0.0);
    }

    #[test]
    fn falling_into_the_hole_respawns() {
        let mut w = small_world();
        // Released from rest: zero angular momentum, so it plunges.
        w.pilot = Pilot::new(&w.kerr, [0.0, 25.0, 4.0], [0.0; 3], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]).unwrap();
        w.pilot.x[0] = w.cluster.t;
        let input = Input { autopilot: -1, ..Default::default() };
        let mut crossed = false;
        let mut max_dilation: f64 = 1.0;
        for _ in 0..(30 * 60) {
            w.step(1.0 / 30.0, &input);
            if w.events.contains(&WorldEvent::HorizonCrossed) {
                crossed = true;
                break;
            }
            max_dilation = max_dilation.max(w.pilot.time_dilation());
        }
        assert!(crossed);
        // In horizon-penetrating Kerr–Schild time a free faller's dt/dτ stays
        // finite (≈ 1.5 at the horizon for a drop from rest); it is hovering
        // or high speed that makes the universe race ahead.
        assert!(max_dilation > 1.3, "max dt/dτ = {max_dilation}");
        assert!(w.kerr.radius(w.pilot.position()) > 10.0);
    }
}
