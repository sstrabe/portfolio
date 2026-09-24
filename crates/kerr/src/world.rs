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
use crate::local::{self, BodyRef, Elements, Local, PlanetRef};
use crate::metric::Kerr;
use crate::orbit;
use crate::pilot::{Command, FORWARD, LEFT, Pilot, Stepped, UP};
use crate::planets::{self, C_KM_S, KM_PER_M, PlanetKind};
use crate::units::{self, SECONDS_PER_M};
use crate::vec3::{self, V3, V4};

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
    /// Physical stars and their planets pull on the ship, and the ship can
    /// land on planets (see [`crate::local`]).
    pub local_gravity: bool,
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
            local_gravity: false,
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
            local_gravity: true,
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
    Docked {
        station: usize,
        offset: V3,
    },
    /// Flying to a planet and into a circular orbit around it.
    Orbit(PlanetRef),
    /// Resting on a planet's surface, turning with it; `offset` is the
    /// position relative to the centre in the planet's rotating frame.
    Landed {
        planet: PlanetRef,
        offset: V3,
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WorldEvent {
    HorizonCrossed,
    Docked(usize),
    Undocked(usize),
    StarCaptured(usize),
    /// Touched down on (or crashed into) a planet at this speed, km/s.
    Landed {
        speed: f64,
    },
    /// Flew into a star and was put back outside it.
    StarContact,
    OrbitReached(PlanetRef),
    AutopilotOff,
    /// The throttle was reset for a new gravity well.
    Throttle(f64),
    /// The time warp was lowered to this (× real time) to keep orbits
    /// watchable.
    WarpLimited(f64),
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
    /// Fraction of the engine's thrust the flight controls command.
    pub throttle: f64,
    /// Acceleration of the engine at that throttle (without boost), in g.
    pub thrust_g: f64,
    /// Time warp in use (× real time) and the most allowed here.
    pub warp: f64,
    pub warp_limit: f64,
    /// The planet being flown to, landed on or targeted, else the nearest
    /// one in the local system.
    pub planet: Option<PlanetTelemetry>,
    /// Phase of the orbit autopilot while it flies.
    pub autopilot: Option<&'static str>,
}

/// The ship relative to a planet.
#[derive(Clone, Copy, Debug)]
pub struct PlanetTelemetry {
    pub planet: PlanetRef,
    pub kind: PlanetKind,
    pub radius_km: f64,
    pub altitude_km: f64,
    /// Speed relative to the planet's centre and its radial part, km/s.
    pub speed_km_s: f64,
    pub vertical_km_s: f64,
    /// Osculating two-body orbit: periapsis and apoapsis altitudes (km)
    /// and period (s); apoapsis and period are infinite if unbound.
    pub periapsis_km: f64,
    pub apoapsis_km: f64,
    pub period_s: f64,
    pub targeted: bool,
    pub landed: bool,
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
    /// The star system whose gravity acts on the ship, if any.
    pub local: Option<Local>,
    /// Snapshot of the target planet's system when it isn't `local`.
    target_system: Option<Local>,
    /// Coordinate time up to which the snapshots are accurate.
    local_until: f64,
    /// Fraction of [`WorldConfig::thrust`] the flight controls command.
    pub throttle: f64,
    /// Planet for the orbit autopilot (cycled with `Tab` on the desktop).
    pub planet_target: Option<PlanetRef>,
    /// Plane of the orbit the autopilot flies into (coordinate normal).
    orbit_normal: V3,
    /// Planet whose vicinity the ship is in (sets the default throttle).
    well: Option<PlanetRef>,
    warp_limited: bool,
    time_scale_now: f64,
}

pub const STATION_TEMPERATURE: f64 = 6500.0;

/// Longest proper-time interval over which a guidance command is held.
const CONTROL_STEP: f64 = 0.4;
/// Near a body, the time warp is capped so that an orbit at the ship's
/// distance takes at least this many wall-clock seconds.
const WALL_SECONDS_PER_ORBIT: f64 = 5.0;
/// The orbit autopilot warps time so the rest of the trip takes about
/// this many wall-clock seconds (within the cap above).
const AUTOPILOT_WALL_SECONDS: f64 = 8.0;
/// Share of the boosted engine the orbit autopilot plans its braking with.
const BRAKE_SHARE: f64 = 0.6;
/// Orbit autopilot gains: velocity (rapidity) error → acceleration, and
/// the slope of the desired approach speed near the target (units of 1/M).
const VELOCITY_GAIN: f64 = 0.5;
const POSITION_GAIN: f64 = 0.1;
/// `Tab` and the orbit autopilot look for planets within this range.
const TARGET_RANGE_AU: f64 = 2000.0;
/// Near a planet (inside this many radii) the throttle defaults to a
/// value suited to its surface gravity.
const WELL_RADII: (f64, f64) = (20.0, 25.0);
const MIN_THROTTLE: f64 = 1e-7;
/// A landed ship rests this far above the mean surface, km.
const LANDED_HEIGHT_KM: f64 = 0.01;

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
            local: None,
            target_system: None,
            local_until: f64::NEG_INFINITY,
            throttle: 1.0,
            planet_target: None,
            orbit_normal: [0.0, 0.0, 1.0],
            well: None,
            warp_limited: false,
            time_scale_now: cfg.time_scale,
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
        // Manual thrust takes over from the orbit autopilot, or lifts off.
        if input.thrust != [0.0; 3] {
            match self.status {
                PilotStatus::Orbit(_) => {
                    self.status = PilotStatus::Free;
                    self.events.push(WorldEvent::AutopilotOff);
                }
                PilotStatus::Landed { .. } => self.status = PilotStatus::Free,
                _ => {}
            }
        }

        self.sync_local();
        self.update_well();
        self.time_scale_now = self.time_scale(wall_dt);
        let mut dtau = wall_dt * self.time_scale_now;
        let requested = dtau;

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
            PilotStatus::Landed { planet, offset } => self.ride_surface(planet, offset, dtau, wall_dt, input),
            _ => self.fly(input, wall_dt, dtau),
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

    /// Integrate the ship over `dtau` of proper time: thrust, guidance and
    /// the local star system's gravity.
    fn fly(&mut self, input: &Input, wall_dt: f64, dtau: f64) {
        let stations = self.station_count() > 0;
        // Guidance is re-evaluated at least every CONTROL_STEP of proper
        // time so it stays stable at high time scales.
        let guided = input.brake || matches!(self.status, PilotStatus::Autopilot(_) | PilotStatus::Orbit(_));
        let mut left = dtau;
        while left > 0.0 {
            self.sync_local();
            let ut = self.pilot.time_dilation().max(1e-9);
            let mut h = left;
            if self.cfg.local_gravity {
                h = h.min(((self.local_until - self.pilot.x[0]) / ut).max(1e-6));
            }
            if guided {
                h = h.min(CONTROL_STEP);
            }
            if left - h < 1e-9 * dtau {
                h = left;
            }
            let cmd = self.command(input, wall_dt * h / dtau, h);
            let stepped = match &self.local {
                Some(field) if self.cfg.local_gravity => self.pilot.step_in(&self.kerr, &cmd, h, field),
                _ => {
                    self.pilot.step(&self.kerr, &cmd, h);
                    Stepped::Done
                }
            };
            left -= h;
            if stations {
                self.cluster.advance_to(self.pilot.x[0]);
                self.check_docking();
                if matches!(self.status, PilotStatus::Docked { .. }) {
                    break;
                }
            }
            if let Stepped::Hit { from, to } = stepped {
                self.touch(from, to);
                break;
            }
            if let PilotStatus::Orbit(p) = self.status {
                self.check_orbit(p);
            }
        }
        self.cluster.advance_to(self.pilot.x[0]);
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
        self.local_until = f64::NEG_INFINITY;
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
        let a_max = self.cfg.thrust * self.throttle * if input.boost { self.cfg.boost_factor } else { 1.0 };
        // Turn rates are specified per wall second so steering feels the same
        // regardless of how fast proper time runs.
        let per_tau = if dtau > 0.0 { wall_dt / dtau } else { 0.0 };
        let mut spin = vec3::scale(input.turn, self.cfg.turn_rate * per_tau);
        let mut accel = vec3::scale(input.thrust, a_max);

        if input.brake {
            // Null the velocity relative to the nearest station, or without
            // stations relative to the local rest frame: the planet or star
            // whose gravity dominates, else the normal observer of the
            // t = const slicing.
            let v = match self.nearest_station() {
                Some((i, _)) => self.relative(i).1,
                None => {
                    let w = self.local_frame().unwrap_or_else(|| self.normal_observer());
                    vec3::scale(self.pilot.relative_velocity(&self.kerr, w), -1.0)
                }
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

        if let PilotStatus::Orbit(p) = self.status {
            match self.orbit_guidance(p, per_tau) {
                Some((a, s)) => {
                    accel = a;
                    if input.turn == [0.0; 3] {
                        spin = s;
                    }
                }
                None => {
                    self.status = PilotStatus::Free;
                    self.events.push(WorldEvent::AutopilotOff);
                }
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
            _ => return,
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
            PilotStatus::Orbit(_) => (3, -1),
            PilotStatus::Landed { .. } => (4, -1),
        };
        let (sd, sv) = match self.status {
            PilotStatus::Autopilot(i) | PilotStatus::Docked { station: i, .. } => self.relative(i),
            _ => self.nearest_station().map(|(i, _)| self.relative(i)).unwrap_or_default(),
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
            throttle: self.throttle,
            thrust_g: self.cfg.thrust * self.throttle / units::G0,
            warp: self.time_scale_now * SECONDS_PER_M,
            warp_limit: self.warp_limit() * SECONDS_PER_M,
            planet: self.planet_telemetry(),
            autopilot: self.autopilot_phase(),
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

/// Flight near planets: the local system, throttle, time warp, the orbit
/// autopilot, landing and telemetry.
impl World {
    /// Refresh the star-system snapshots once the pilot's clock passes the
    /// time up to which they are accurate (or they were invalidated).
    fn sync_local(&mut self) {
        if !self.cfg.local_gravity {
            return;
        }
        let t = self.pilot.x[0];
        if t < self.local_until {
            return;
        }
        self.cluster.advance_to(t);
        let k = self.kerr;
        let pos = self.pilot.position();
        self.local = local::select(&self.cluster, pos).map(|i| Local::new(&k, &self.cluster, i, pos));
        let wanted = self.flight_planet().or(self.planet_target);
        self.target_system = wanted
            .filter(|&p| self.valid(p) && !self.local.as_ref().is_some_and(|l| l.star == p.star))
            .map(|p| Local::new(&k, &self.cluster, p.star, pos));
        let mut until = match &self.local {
            Some(l) => l.valid_until,
            None => t + local::time_to_range(&self.cluster, pos).max(1.0),
        };
        if let Some(ts) = &self.target_system {
            until = until.min(ts.valid_until);
        }
        self.local_until = until;
    }

    /// The planet the ship is flying to or resting on.
    fn flight_planet(&self) -> Option<PlanetRef> {
        match self.status {
            PilotStatus::Orbit(p) | PilotStatus::Landed { planet: p, .. } => Some(p),
            _ => None,
        }
    }

    /// Whether `p`'s star still exists (slots are reused).
    fn valid(&self, p: PlanetRef) -> bool {
        self.cluster.bodies.get(p.star).is_some_and(|b| b.alive && b.generation == p.generation)
    }

    /// The snapshot holding planet `p`, if it is current.
    pub fn system_of(&self, p: PlanetRef) -> Option<&Local> {
        self.local.iter().chain(self.target_system.iter()).find(|l| l.holds(p))
    }

    /// Position and coordinate velocity of the ship relative to the centre
    /// of planet `i` of `l`.
    fn relative_to(&self, l: &Local, i: usize) -> (V3, V3) {
        let (xp, vp) = l.planet_state(i, self.pilot.x[0]);
        let u = self.pilot.e[0];
        let v = [u[1] / u[0], u[2] / u[0], u[3] / u[0]];
        (vec3::sub(self.pilot.position(), xp), vec3::sub(v, vp))
    }

    /// 4-velocity of the body whose gravity dominates at the ship.
    fn local_frame(&self) -> Option<V4> {
        let l = self.local.as_ref()?;
        let (t, pos) = (self.pilot.x[0], self.pilot.position());
        if !l.reaches(t, pos) {
            return None;
        }
        let (body, _) = l.dominant(t, pos);
        self.kerr.four_velocity(pos, l.body_state(body, t).1)
    }

    /// Reset the throttle when the ship enters or leaves a planet's
    /// vicinity: near a planet full thrust (17,000 g) is useless, so the
    /// throttle drops to the power of ten that gives at least 1.5 times the
    /// surface gravity; away from planets it goes back to full.
    fn update_well(&mut self) {
        let Some(l) = &self.local else {
            if self.well.take().is_some() {
                self.throttle = 1.0;
                self.events.push(WorldEvent::Throttle(1.0));
            }
            return;
        };
        let (t, pos) = (self.pilot.x[0], self.pilot.position());
        let well = (0..l.planet_count())
            .find(|&i| {
                let (_, radius) = l.planet_mass(i);
                let radii = if self.well == Some(l.planet_ref(i)) { WELL_RADII.1 } else { WELL_RADII.0 };
                vec3::norm(vec3::sub(pos, l.planet_state(i, t).0)) < radii * radius
            })
            .map(|i| (l.planet_ref(i), l.planet_mass(i)));
        if well.map(|w| w.0) == self.well {
            return;
        }
        self.throttle = match well {
            Some((_, (mu, radius))) => {
                let g = mu / (radius * radius);
                10f64.powf((1.5 * g / self.cfg.thrust).log10().ceil()).clamp(MIN_THROTTLE, 1.0)
            }
            None => 1.0,
        };
        self.well = well.map(|w| w.0);
        self.events.push(WorldEvent::Throttle(self.throttle));
    }

    /// Multiply the throttle by `factor` (clamped to 10⁻⁷–1); returns it.
    pub fn scale_throttle(&mut self, factor: f64) -> f64 {
        self.throttle = (self.throttle * factor).clamp(MIN_THROTTLE, 1.0);
        self.throttle
    }

    /// Largest time scale (M per wall second) allowed where the ship is: an
    /// orbit at its distance from the dominant body must take at least
    /// [`WALL_SECONDS_PER_ORBIT`], so low orbits stay watchable. Real time
    /// is always allowed. Unlimited outside star systems.
    pub fn warp_limit(&self) -> f64 {
        let Some(l) = self.local.as_ref().filter(|_| self.cfg.local_gravity) else { return f64::INFINITY };
        let (t, pos) = (self.pilot.x[0], self.pilot.position());
        if !l.reaches(t, pos) {
            return f64::INFINITY;
        }
        let (body, d) = l.dominant(t, pos);
        let (mu, radius) = l.mass_of(body);
        let d = d.max(radius);
        let period = std::f64::consts::TAU * (d * d * d / mu).sqrt();
        (period / WALL_SECONDS_PER_ORBIT).max(1.0 / SECONDS_PER_M)
    }

    /// Time scale for this frame: the pilot's, lowered to the local limit
    /// (which also lowers the setting), or raised by the orbit autopilot.
    fn time_scale(&mut self, wall_dt: f64) -> f64 {
        let cap = self.warp_limit();
        if self.cfg.time_scale > cap {
            self.cfg.time_scale = cap;
            if !self.warp_limited {
                self.events.push(WorldEvent::WarpLimited(cap * SECONDS_PER_M));
            }
            self.warp_limited = true;
        } else if self.cfg.time_scale < 0.99 * cap {
            self.warp_limited = false;
        }
        let mut ts = self.cfg.time_scale;
        if let PilotStatus::Orbit(p) = self.status {
            // Enough warp for the rest of the trip to take a few seconds,
            // within the cap and with a few guidance updates per frame.
            let auto = self.autopilot_time_to_go(p) / AUTOPILOT_WALL_SECONDS;
            ts = ts.max(auto.min(cap).min(20.0 * CONTROL_STEP / wall_dt.max(1e-3)));
        }
        ts
    }

    /// Radius of the parking orbit around planet `i` of `l`, units of M.
    fn parking_radius(l: &Local, i: usize) -> f64 {
        let planet = l.planet(i).expect("planet");
        (planet.radius_km + local::parking_altitude_km(planet)) / KM_PER_M
    }

    /// Rough proper time the orbit autopilot still needs: a boosted
    /// brachistochrone to the parking orbit plus one orbit to settle.
    fn autopilot_time_to_go(&self, p: PlanetRef) -> f64 {
        let Some(l) = self.system_of(p) else { return 0.0 };
        let (mu, _) = l.planet_mass(p.planet);
        let rc = Self::parking_radius(l, p.planet);
        let s = (vec3::norm(self.relative_to(l, p.planet).0) - rc).max(0.0);
        let a = BRAKE_SHARE * self.cfg.thrust * self.cfg.boost_factor;
        2.0 * (1.0 + 0.5 * a * s).acosh() / a + std::f64::consts::TAU * (rc * rc * rc / mu).sqrt()
    }

    fn autopilot_phase(&self) -> Option<&'static str> {
        let PilotStatus::Orbit(p) = self.status else { return None };
        let l = self.system_of(p)?;
        let d = vec3::norm(self.relative_to(l, p.planet).0) / Self::parking_radius(l, p.planet);
        Some(if d > 30.0 {
            "transfer"
        } else if d > 1.5 {
            "approach"
        } else {
            "orbit insertion"
        })
    }

    /// The nearest planet of the nearest star system with planets within
    /// [`TARGET_RANGE_AU`].
    fn nearest_planet(&self) -> Option<PlanetRef> {
        let pos = self.pilot.position();
        let bodies = &self.cluster.bodies;
        let range = TARGET_RANGE_AU * units::AU;
        let (_, sys) = planets::nearby(self.cluster.cfg.seed, bodies, pos, range).into_iter().next()?;
        let star = bodies[sys.star].position();
        let t = self.cluster.t;
        let dist = |i: usize| vec3::norm(vec3::sub(vec3::axpy(star, 1.0 / KM_PER_M, sys.planet_state(i, t).0), pos));
        let i = (0..sys.planets.len()).min_by(|&a, &b| dist(a).total_cmp(&dist(b)))?;
        Some(PlanetRef { star: sys.star, generation: bodies[sys.star].generation, planet: i })
    }

    /// Target the next planet (outwards) of the targeted system, or the
    /// nearest planet if none is targeted. Retargets a flying autopilot.
    pub fn cycle_target(&mut self) -> Option<PlanetRef> {
        let next = match self.planet_target.filter(|&p| self.valid(p)) {
            Some(p) => {
                let body = &self.cluster.bodies[p.star];
                let n = planets::system(self.cluster.cfg.seed, p.star, body).map_or(1, |s| s.planets.len());
                PlanetRef { planet: (p.planet + 1) % n, ..p }
            }
            None => self.nearest_planet()?,
        };
        self.planet_target = Some(next);
        self.local_until = f64::NEG_INFINITY;
        if matches!(self.status, PilotStatus::Orbit(_)) {
            self.engage(next);
        }
        Some(next)
    }

    /// Engage the orbit autopilot on the target (or the nearest planet),
    /// or disengage it. Returns the planet when engaged.
    pub fn toggle_orbit_autopilot(&mut self) -> Option<PlanetRef> {
        if matches!(self.status, PilotStatus::Orbit(_)) {
            self.status = PilotStatus::Free;
            return None;
        }
        let p = self.planet_target.filter(|&p| self.valid(p)).or_else(|| self.nearest_planet())?;
        self.planet_target = Some(p);
        self.engage(p).then_some(p)
    }

    /// Fly to `p` and into a circular orbit. The orbit's plane is the one
    /// the ship already moves in around the planet if it has real angular
    /// momentum there, else the planet's equator (tilted to contain the
    /// ship).
    fn engage(&mut self, p: PlanetRef) -> bool {
        let saved = self.status;
        self.status = PilotStatus::Orbit(p);
        self.local_until = f64::NEG_INFINITY;
        self.sync_local();
        let Some(l) = self.system_of(p) else {
            self.status = saved;
            return false;
        };
        let (r, v) = self.relative_to(l, p.planet);
        let (mu, _) = l.planet_mass(p.planet);
        let d = vec3::norm(r);
        let h = vec3::cross(r, v);
        let axis = l.planet(p.planet).expect("planet").spin_axis;
        let rhat = vec3::normalize(r);
        let in_plane = vec3::axpy(axis, -vec3::dot(axis, rhat), rhat);
        self.orbit_normal = if vec3::norm(h) > 0.3 * d * (mu / d).sqrt() {
            vec3::normalize(h)
        } else if vec3::norm(in_plane) > 1e-3 {
            vec3::normalize(in_plane)
        } else {
            vec3::any_orthogonal(rhat)
        };
        true
    }

    /// Guidance for the orbit autopilot: thrust and spin (ship frame).
    ///
    /// It tracks a desired velocity field around the planet, in rapidity so
    /// it also holds on relativistic transfers: towards the parking circle
    /// in its plane (radial and out-of-plane parts) at a speed from which
    /// the boosted engine can still stop, plus the circular-orbit speed
    /// `√(GM/ρ)` along the orbit. The feed-forward cancels the planet's
    /// gravity (and the star's tide) against the centripetal acceleration
    /// of that motion, so on the circle the engine is off.
    fn orbit_guidance(&self, p: PlanetRef, per_tau: f64) -> Option<(V3, V3)> {
        let l = self.system_of(p)?;
        let i = p.planet;
        let k = &self.kerr;
        let (t, pos) = (self.pilot.x[0], self.pilot.position());
        let (xp, vp) = l.planet_state(i, t);
        let (mu, radius) = l.planet_mass(i);
        let rc = Self::parking_radius(l, i);
        let to_planet = self.pilot.local_components(k, vec3::sub(xp, pos));
        let r = vec3::scale(to_planet, -1.0);
        let v = vec3::scale(self.pilot.relative_velocity(k, k.four_velocity(pos, vp)?), -1.0);
        let g_rel = vec3::sub(l.gravity(t, pos), l.gravity_except(t, xp, Some(i)));
        let g = self.pilot.local_components(k, g_rel);
        let n = vec3::normalize(self.pilot.local_components(k, self.orbit_normal));

        let z = vec3::dot(r, n);
        let rho_v = vec3::axpy(r, -z, n);
        let rho = vec3::norm(rho_v);
        let rhat = if rho > 1e-9 * rc { vec3::scale(rho_v, 1.0 / rho) } else { vec3::any_orthogonal(n) };
        let along = vec3::cross(n, rhat);

        let a_full = self.cfg.thrust * self.cfg.boost_factor;
        let a_b = BRAKE_SHARE * a_full;
        // Approach rapidity with a distance s to go: √(2as + c²) − c is
        // ≈ POSITION_GAIN·s near the target and never needs more than a_b
        // of braking; acosh(1 + a s) is the exact relativistic limit.
        let c = a_b / POSITION_GAIN;
        let law = |s: f64| -> f64 {
            let m = s.abs();
            -s.signum() * ((2.0 * a_b * m + c * c).sqrt() - c).min((1.0 + a_b * m).acosh())
        };
        let rho_c = rho.max(radius);
        let v_t = (mu / rho_c).sqrt();
        let want = vec3::add(
            vec3::add(vec3::scale(rhat, law(rho - rc)), vec3::scale(n, law(z))),
            vec3::scale(along, v_t.atanh()),
        );
        let speed = vec3::norm(v);
        let have = if speed > 0.0 { vec3::scale(v, speed.min(1.0 - 1e-16).atanh() / speed) } else { [0.0; 3] };
        let feed = vec3::sub(vec3::scale(rhat, -v_t * v_t / rho_c), g);
        let accel = clamp_len(vec3::axpy(feed, VELOCITY_GAIN, vec3::sub(want, have)), a_full);

        // Face the planet on the way in; in orbit face along it with the
        // planet below.
        let spin = if rho < 4.0 * rc && z.abs() < rc {
            self.orient(along, r, per_tau)
        } else {
            self.orient(to_planet, [0.0, 0.0, 1.0], per_tau)
        };
        Some((accel, spin))
    }

    /// Spin that turns the nose towards ship-frame direction `fwd` and,
    /// once close, rolls the top towards `up`.
    fn orient(&self, fwd: V3, up: V3, per_tau: f64) -> V3 {
        let f = vec3::normalize(fwd);
        let axis = vec3::cross([1.0, 0.0, 0.0], f);
        let ang = vec3::norm(axis).atan2(f[0]);
        let max_rate = self.cfg.turn_rate * per_tau;
        let mut spin = if vec3::norm(axis) > 1e-9 {
            vec3::scale(vec3::normalize(axis), (2.0 * ang * per_tau).min(max_rate))
        } else if f[0] < 0.0 {
            [0.0, 0.0, max_rate]
        } else {
            [0.0; 3]
        };
        if ang < 0.3 && up[1].hypot(up[2]) > 1e-9 {
            // Rolling by +θ about forward turns the top towards −left.
            let roll = (-up[1]).atan2(up[2]);
            spin[0] += (2.0 * roll * per_tau).clamp(-max_rate, max_rate);
        }
        spin
    }

    /// Hand control back once the osculating orbit is the parking circle.
    fn check_orbit(&mut self, p: PlanetRef) {
        let Some(l) = self.system_of(p) else { return };
        let i = p.planet;
        let (mu, radius) = l.planet_mass(i);
        let planet = l.planet(i).expect("planet");
        let rc = Self::parking_radius(l, i);
        let floor = radius + planet.atmosphere.map_or(0.0, |a| a.top_km) / KM_PER_M;
        let (r, v) = self.relative_to(l, i);
        let el = Elements::of(mu, r, v);
        if el.bound()
            && el.periapsis > floor
            && el.apoapsis - el.periapsis < 0.0005 * rc
            && (el.semi_major - rc).abs() < 0.001 * rc
        {
            self.status = PilotStatus::Free;
            self.events.push(WorldEvent::OrbitReached(p));
        }
    }

    /// The substep from `from` to `to` entered a body: land on a planet,
    /// or get put back outside a star.
    fn touch(&mut self, from: V4, to: V4) {
        let Some(l) = &self.local else { return };
        let Some(c) = l.contact(from, to) else { return };
        let t = c.x[0];
        let (centre, vel) = l.body_state(c.body, t);
        let n = vec3::normalize(vec3::sub(vec3::spatial(c.x), centre));
        let u = self.pilot.e[0];
        let v_ship = [u[1] / u[0], u[2] / u[0], u[3] / u[0]];
        match c.body {
            BodyRef::Planet(i) => {
                let p = l.planet_ref(i);
                let (_, radius) = l.planet_mass(i);
                let planet = l.planet(i).expect("planet");
                let r = vec3::scale(n, radius + LANDED_HEIGHT_KM / KM_PER_M);
                let offset = vec3::rotate(r, planet.spin_axis, -planet.rotation(t * SECONDS_PER_M));
                let (pos, v_surface) = surface(l, i, offset, t);
                let speed = vec3::norm(vec3::sub(v_ship, v_surface)) * C_KM_S;
                self.place(t, pos, v_surface);
                self.status = PilotStatus::Landed { planet: p, offset };
                self.events.push(WorldEvent::Landed { speed });
            }
            BodyRef::Star => {
                let pos = vec3::axpy(centre, 10.0 * l.star_radius, n);
                self.place(t, pos, vel);
                self.status = PilotStatus::Free;
                self.events.push(WorldEvent::StarContact);
            }
        }
    }

    fn place(&mut self, t: f64, pos: V3, vel: V3) {
        self.pilot.x = [t, pos[0], pos[1], pos[2]];
        if let Some(u) = self.kerr.four_velocity(pos, vel) {
            self.pilot.set_velocity(&self.kerr, u);
        }
    }

    /// Ride on a planet's surface for `dtau`, turning with it.
    fn ride_surface(&mut self, p: PlanetRef, offset: V3, dtau: f64, wall_dt: f64, input: &Input) {
        let Some(l) = self.system_of(p) else {
            self.status = PilotStatus::Free;
            return self.fly(input, wall_dt, dtau);
        };
        let planet = l.planet(p.planet).expect("planet");
        let omega = std::f64::consts::TAU / planet.rotation_period_s * SECONDS_PER_M;
        let t = self.pilot.x[0];
        let (pos, vel) = surface(l, p.planet, offset, t);
        let ut = self.kerr.four_velocity(pos, vel).map_or(1.0, |u| u[0]);
        let t1 = t + ut * dtau;
        let (pos1, vel1) = surface(l, p.planet, offset, t1);
        // The ship's axes turn with the ground, plus the pilot's turning.
        let per_tau = wall_dt / dtau.max(1e-9);
        let turn = vec3::scale(self.pilot.local_components(&self.kerr, planet.spin_axis), omega * ut);
        let spin = vec3::axpy(turn, self.cfg.turn_rate * per_tau, input.turn);
        self.place(t1, pos1, vel1);
        self.pilot.tau += dtau;
        self.pilot.rotate(spin, dtau);
        self.cluster.advance_to(t1);
    }

    fn planet_telemetry(&self) -> Option<PlanetTelemetry> {
        let pos = self.pilot.position();
        let t = self.pilot.x[0];
        let nearest = || {
            let l = self.local.as_ref()?;
            let dist = |i: usize| vec3::norm(vec3::sub(l.planet_state(i, t).0, pos));
            (0..l.planet_count()).min_by(|&a, &b| dist(a).total_cmp(&dist(b))).map(|i| l.planet_ref(i))
        };
        let p =
            self.flight_planet().or(self.planet_target.filter(|&p| self.system_of(p).is_some())).or_else(nearest)?;
        let l = self.system_of(p)?;
        let planet = l.planet(p.planet)?;
        let (mu, radius) = l.planet_mass(p.planet);
        let (r, v) = self.relative_to(l, p.planet);
        let d = vec3::norm(r);
        let el = Elements::of(mu, r, v);
        Some(PlanetTelemetry {
            planet: p,
            kind: planet.kind,
            radius_km: planet.radius_km,
            altitude_km: (d - radius) * KM_PER_M,
            speed_km_s: vec3::norm(v) * C_KM_S,
            vertical_km_s: vec3::dot(r, v) / d * C_KM_S,
            periapsis_km: (el.periapsis - radius) * KM_PER_M,
            apoapsis_km: (el.apoapsis - radius) * KM_PER_M,
            period_s: el.period * SECONDS_PER_M,
            targeted: self.planet_target == Some(p),
            landed: matches!(self.status, PilotStatus::Landed { .. }),
        })
    }
}

/// Position and coordinate velocity of the surface point at `offset`
/// (planet-fixed) of planet `i` at coordinate time `t`.
fn surface(l: &Local, i: usize, offset: V3, t: f64) -> (V3, V3) {
    let planet = l.planet(i).expect("planet");
    let (xp, vp) = l.planet_state(i, t);
    let r = vec3::rotate(offset, planet.spin_axis, planet.rotation(t * SECONDS_PER_M));
    let omega = std::f64::consts::TAU / planet.rotation_period_s * SECONDS_PER_M;
    (vec3::add(xp, r), vec3::axpy(vp, omega, vec3::cross(planet.spin_axis, r)))
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

    /// A realistic world and the rocky planet in it where the star's tide
    /// is weakest (so low orbits are Keplerian to ~10⁻⁶ and drift shows
    /// numerical error, not physics).
    fn planet_world() -> (World, PlanetRef) {
        let mut w = World::new(WorldConfig::sgr_a(400, 1));
        let seed = w.cluster.cfg.seed;
        let systems = planets::nearby(seed, &w.cluster.bodies, w.pilot.position(), f64::INFINITY);
        // Tide over gravity at the surface: 2 (R/a)³ M★/m.
        let tide = |s: &planets::System, p: &planets::Planet| {
            2.0 * (p.radius_km / p.orbit.a_km).powi(3) * s.star_mass_kg / p.mass_kg
        };
        let (_, sys, i) = systems
            .iter()
            .flat_map(|(_, s)| s.planets.iter().enumerate().map(move |(i, p)| (s, i, p)))
            .filter(|(_, _, p)| !p.kind.is_giant())
            .map(|(s, i, p)| (tide(s, p), s, i))
            .min_by(|a, b| a.0.total_cmp(&b.0))
            .expect("a rocky planet");
        let p = PlanetRef { star: sys.star, generation: w.cluster.bodies[sys.star].generation, planet: i };
        w.planet_target = Some(p);
        (w, p)
    }

    /// Put the ship at `offset` (units of M) from planet `p`'s centre with
    /// velocity `vel` relative to it, facing along `vel` (or the planet).
    fn put_near(w: &mut World, p: PlanetRef, offset: V3, vel: V3) {
        let t = w.cluster.t;
        let l = Local::new(&w.kerr, &w.cluster, p.star, w.pilot.position());
        let (xp, vp) = l.planet_state(p.planet, t);
        let look = if vec3::norm(vel) > 0.0 { vel } else { vec3::scale(offset, -1.0) };
        let pos = vec3::add(xp, offset);
        let mut pilot = Pilot::new(&w.kerr, pos, vec3::add(vp, vel), look, vec3::any_orthogonal(look)).unwrap();
        pilot.x[0] = t;
        w.pilot = pilot;
        w.local_until = f64::NEG_INFINITY;
    }

    fn circular(w: &World, p: PlanetRef, altitude_km: f64) -> (V3, V3) {
        let l = Local::new(&w.kerr, &w.cluster, p.star, w.pilot.position());
        let (mu, radius) = l.planet_mass(p.planet);
        let r = radius + altitude_km / KM_PER_M;
        let axis = l.planet(p.planet).unwrap().spin_axis;
        let radial = vec3::any_orthogonal(axis);
        (vec3::scale(radial, r), vec3::scale(vec3::cross(axis, radial), (mu / r).sqrt()))
    }

    #[test]
    fn low_orbit_stays_bounded() {
        let (mut w, p) = planet_world();
        let (r, v) = circular(&w, p, 300.0);
        put_near(&mut w, p, r, v);
        // Ask for far too much warp: it is capped so an orbit takes ≈ 5 s.
        w.cfg.time_scale = 1e9;
        let input = Input { autopilot: -1, ..Default::default() };
        w.step(1.0 / 60.0, &input);
        let t0 = w.telemetry().planet.expect("planet telemetry");
        assert!(w.cfg.time_scale < 1e5, "warp not capped: {}", w.cfg.time_scale * SECONDS_PER_M);
        assert!((t0.altitude_km - 300.0).abs() < 0.1 && t0.apoapsis_km - t0.periapsis_km < 0.1, "{t0:?}");
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        let start = w.pilot.x[0];
        let mut orbits = 0.0;
        while orbits < 4.0 {
            w.step(1.0 / 60.0, &input);
            let t = w.telemetry().planet.unwrap();
            lo = lo.min(t.altitude_km);
            hi = hi.max(t.altitude_km);
            orbits = (w.pilot.x[0] - start) * SECONDS_PER_M / t0.period_s;
        }
        let t = w.telemetry().planet.unwrap();
        assert!(w.status == PilotStatus::Free);
        assert!(lo > 299.5 && hi < 300.5, "altitude range {lo}..{hi} km");
        // Semi-major axis (energy) and period drift.
        assert!((t.period_s / t0.period_s - 1.0).abs() < 1e-5, "period {} -> {}", t0.period_s, t.period_s);
        eprintln!("4 orbits: altitude {lo:.2}..{hi:.2} km, period {:.3} -> {:.3} s", t0.period_s, t.period_s);
    }

    #[test]
    fn orbit_autopilot_flies_in_from_an_au() {
        let (mut w, p) = planet_world();
        let l = Local::new(&w.kerr, &w.cluster, p.star, w.pilot.position());
        // 1 AU from the planet, out of its orbital plane (away from the
        // star), at rest in the star's frame.
        let orbit = l.planet(p.planet).unwrap().orbit;
        let normal = vec3::normalize(vec3::cross(orbit.p, orbit.q));
        let vp = l.planet_state(p.planet, w.cluster.t).1;
        let vs = l.star_state(w.cluster.t).1;
        put_near(&mut w, p, vec3::scale(normal, units::AU), vec3::sub(vs, vp));
        assert_eq!(w.toggle_orbit_autopilot(), Some(p));
        let input = Input { autopilot: -1, ..Default::default() };
        let mut reached = None;
        for frame in 0..60 * 60 {
            w.step(1.0 / 60.0, &input);
            if w.events.contains(&WorldEvent::OrbitReached(p)) {
                reached = Some(frame);
                break;
            }
            let crashed = w.events.iter().any(|e| matches!(e, WorldEvent::Landed { .. } | WorldEvent::StarContact));
            assert!(!crashed, "crashed: {:?}", w.telemetry());
        }
        let frame = reached.unwrap_or_else(|| panic!("no orbit after 60 s: {:?}", w.telemetry()));
        let t = w.telemetry().planet.unwrap();
        let want = local::parking_altitude_km(w.system_of(p).unwrap().planet(p.planet).unwrap());
        assert!((t.periapsis_km / want - 1.0).abs() < 0.03 && (t.apoapsis_km / want - 1.0).abs() < 0.03, "{t:?}");
        assert!(frame < 60 * 30, "took {} s", frame / 60);
        eprintln!("orbit from 1 AU after {:.1} s of wall time: {t:?}", frame as f64 / 60.0);
        // It stays up on its own.
        for _ in 0..600 {
            w.step(1.0 / 60.0, &input);
        }
        let after = w.telemetry().planet.unwrap();
        assert!(w.status == PilotStatus::Free && (after.periapsis_km / want - 1.0).abs() < 0.1, "{after:?}");
    }

    #[test]
    fn orbit_autopilot_crosses_the_cluster() {
        let mut w = World::new(WorldConfig::sgr_a(400, 1));
        let p = w.toggle_orbit_autopilot().expect("a planet in range");
        // Hundreds of AU away: the transfer is relativistic (γ in the
        // hundreds) and the warp is high until the approach.
        let input = Input { autopilot: -1, ..Default::default() };
        let mut max_gamma: f64 = 1.0;
        let reached = (0..60 * 60).any(|_| {
            w.step(1.0 / 60.0, &input);
            max_gamma = max_gamma.max(w.telemetry().gamma);
            w.events.contains(&WorldEvent::OrbitReached(p))
        });
        assert!(reached && max_gamma > 10.0, "max γ {max_gamma}: {:?}", w.telemetry());
    }

    #[test]
    fn falling_ship_lands_and_turns_with_the_planet() {
        let (mut w, p) = planet_world();
        let (r, _) = circular(&w, p, 2000.0);
        put_near(&mut w, p, r, [0.0; 3]);
        w.cfg.time_scale = 1e9;
        let input = Input { autopilot: -1, ..Default::default() };
        let landed = (0..60 * 60).any(|_| {
            w.step(1.0 / 60.0, &input);
            matches!(w.status, PilotStatus::Landed { .. })
        });
        assert!(landed, "{:?}", w.telemetry());
        let t = w.telemetry().planet.unwrap();
        assert!(t.landed && (t.altitude_km - LANDED_HEIGHT_KM).abs() < 1e-3, "{t:?}");
        for _ in 0..120 {
            w.step(1.0 / 60.0, &input);
        }
        let t = w.telemetry().planet.unwrap();
        assert!((t.altitude_km - LANDED_HEIGHT_KM).abs() < 1e-3 && t.vertical_km_s.abs() < 1e-3, "{t:?}");
        // Thrusting up lifts off.
        let up = vec3::normalize(w.pilot.local_components(&w.kerr, r));
        let lift = Input { thrust: up, boost: true, autopilot: -1, ..Default::default() };
        w.step(1.0 / 60.0, &lift);
        assert_eq!(w.status, PilotStatus::Free);
    }

    #[test]
    fn gravity_is_off_far_from_star_systems() {
        let mut with = World::new(WorldConfig::sgr_a(16, 1));
        let mut without = World::new(WorldConfig { local_gravity: false, ..WorldConfig::sgr_a(16, 1) });
        let input = Input { thrust: [1.0, 0.0, 0.0], autopilot: -1, ..Default::default() };
        for _ in 0..300 {
            with.step(1.0 / 60.0, &input);
            without.step(1.0 / 60.0, &input);
        }
        assert!(with.local.is_none() && with.telemetry().planet.is_none());
        assert!(with.warp_limit().is_infinite() && with.throttle == 1.0);
        let d = vec3::norm(vec3::sub(with.pilot.position(), without.pilot.position()));
        assert!(d < 1e-9 * vec3::norm(with.pilot.position()), "paths differ by {d}");
    }
}
