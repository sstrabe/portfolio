//! Where a flight begins.

use kerr::geodesic;
use kerr::local::{self, PlanetRef};
use kerr::pilot::Pilot;
use kerr::planets::{self, C_KM_S, KM_PER_M, PlanetKind};
use kerr::units::{PARSEC, SECONDS_PER_M};
use kerr::vec3::{self, V3};
use kerr::world::World;
use render_hq::Gpu;
use render_hq::terrain::{self, probe::Probe};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Start {
    /// On a circular orbit 800 AU from the hole, facing prograde.
    Cluster,
    /// Next to the nearest planet of a kind (by default 420 km above an
    /// Earth-like world at dawn, looking at the sunrise over its limb).
    Planet(PlanetStart),
    /// Facing one of the nebulae (`render_hq::nebula::landmarks`): from the
    /// cluster start, or at rest `distance_pc` from it on the side facing
    /// Earth.
    Look { target: usize, distance_pc: Option<f64> },
    /// Standing on the Earth-like world.
    Ground(GroundStart),
}

/// `ground[:SITE][:LAT[:HOUR[:VIEW[:HEIGHT]]]]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GroundStart {
    pub site: Site,
    pub latitude_deg: f64,
    /// Local solar time, hours (12 is noon).
    pub hour: f64,
    pub view: GroundView,
    /// Eye height above the ground (or the sea), m.
    pub height_m: f64,
}

impl Default for GroundStart {
    fn default() -> Self {
        Self { site: Site::Here, latitude_deg: 20.0, hour: 15.0, view: GroundView::Horizon, height_m: 1.7 }
    }
}

/// Where to stand, relative to the spot at the latitude and hour.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Site {
    /// On the spot, land or sea.
    Here,
    /// On the nearest land.
    Land,
    /// On the nearest coast, a little inland, looking out to sea.
    Coast,
    /// On the shore of the tallest young volcanic island in the tropics
    /// (a hotspot chain's newest, like Hawaii's Big Island), at the local
    /// time asked for.
    Island,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GroundView {
    /// Level, 5° down, the sun on the right.
    Horizon,
    /// Towards the sun's azimuth, level.
    Sun,
    /// 45° down, away from the sun (the ground lit from behind you).
    Down,
    /// 30° up, away from the sun.
    Sky,
}

impl Start {
    /// Parse `--look NAME[@PC]`.
    pub fn look(s: &str) -> Result<Self, String> {
        let (name, distance_pc) = match s.split_once('@') {
            Some((n, d)) => (n, Some(d.parse::<f64>().map_err(|e| format!("--look {s}: {e}"))?)),
            None => (s, None),
        };
        let places = render_hq::nebula::landmarks();
        let target = places
            .iter()
            .position(|(n, _)| *n == name)
            .ok_or_else(|| format!("unknown nebula {name:?} (one of {})", places.map(|(n, _)| n).join(", ")))?;
        Ok(Self::Look { target, distance_pc })
    }
}

/// `planet[:KIND[:ALTITUDE[:VIEW]]]`, e.g. `planet:desert:2000:day` or
/// `planet:ringed:3r:disc`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlanetStart {
    pub kind: KindFilter,
    /// Kilometres, or planet radii when `in_radii`.
    pub altitude: f64,
    pub in_radii: bool,
    pub view: View,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KindFilter {
    Kind(PlanetKind),
    /// A giant with rings.
    Ringed,
    /// The nearest planet of any kind.
    Any,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum View {
    /// Over the dawn terminator, looking at the sunrise over the limb.
    Dawn,
    /// Over the morning side, looking at the sun's specular point (glint).
    Day,
    /// Low sun from the side, looking along the horizon (relief on the limb).
    Limb,
    /// Straight down, sun at the top of the image.
    Nadir,
    /// Over the night side, looking towards the terminator.
    Night,
    /// The whole planet seen from 30° off the sun's direction.
    Disc,
}

impl Default for PlanetStart {
    fn default() -> Self {
        Self { kind: KindFilter::Kind(PlanetKind::Ocean), altitude: 420.0, in_radii: false, view: View::Dawn }
    }
}

impl std::str::FromStr for Start {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        let mut parts = s.split(':');
        match parts.next() {
            Some("cluster") if parts.clone().next().is_none() => return Ok(Self::Cluster),
            Some("planet") => {}
            Some("ground") => return parse_ground(s, parts).map(Self::Ground),
            _ => {
                return Err(format!(
                    "unknown start {s:?} (cluster, planet[:KIND[:ALTITUDE[:VIEW]]] or ground[:LAT[:HOUR[:VIEW[:HEIGHT]]]])"
                ));
            }
        }
        let mut p = PlanetStart::default();
        let mut altitude_given = false;
        if let Some(k) = parts.next() {
            p.kind = match k {
                "ocean" => KindFilter::Kind(PlanetKind::Ocean),
                "rocky" => KindFilter::Kind(PlanetKind::Rocky),
                "desert" => KindFilter::Kind(PlanetKind::Desert),
                "ice" => KindFilter::Kind(PlanetKind::Ice),
                "lava" => KindFilter::Kind(PlanetKind::Lava),
                "gas" => KindFilter::Kind(PlanetKind::GasGiant),
                "icegiant" => KindFilter::Kind(PlanetKind::IceGiant),
                "ringed" => KindFilter::Ringed,
                "any" => KindFilter::Any,
                _ => {
                    return Err(format!(
                        "unknown planet kind {k:?} (ocean, rocky, desert, ice, lava, gas, icegiant, ringed, any)"
                    ));
                }
            };
        }
        if let Some(a) = parts.next() {
            let (num, radii) = a.strip_suffix('r').map_or((a, false), |n| (n, true));
            p.altitude = num.parse().map_err(|e| format!("altitude {a:?}: {e}"))?;
            p.in_radii = radii;
            altitude_given = true;
        }
        if !altitude_given
            && matches!(p.kind, KindFilter::Ringed | KindFilter::Kind(PlanetKind::GasGiant | PlanetKind::IceGiant))
        {
            (p.altitude, p.in_radii) = (2.0, true);
        }
        if let Some(v) = parts.next() {
            p.view = match v {
                "dawn" => View::Dawn,
                "day" => View::Day,
                "limb" => View::Limb,
                "nadir" => View::Nadir,
                "night" => View::Night,
                "disc" => View::Disc,
                _ => return Err(format!("unknown view {v:?} (dawn, day, limb, nadir, night, disc)")),
            };
        }
        if parts.next().is_some() {
            return Err(format!("too many parts in {s:?}"));
        }
        Ok(Self::Planet(p))
    }
}

fn parse_ground<'a>(s: &str, parts: impl Iterator<Item = &'a str>) -> Result<GroundStart, String> {
    let mut g = GroundStart::default();
    let num = |p: &str, what: &str| p.parse::<f64>().map_err(|e| format!("{what} {p:?} in {s:?}: {e}"));
    let mut parts = parts.peekable();
    if let Some(site) = parts.peek().and_then(|p| match *p {
        "here" => Some(Site::Here),
        "land" => Some(Site::Land),
        "coast" => Some(Site::Coast),
        "island" => Some(Site::Island),
        _ => None,
    }) {
        g.site = site;
        parts.next();
    }
    if let Some(p) = parts.next() {
        g.latitude_deg = num(p, "latitude")?.clamp(-89.0, 89.0);
    }
    if let Some(p) = parts.next() {
        g.hour = num(p, "hour")?.rem_euclid(24.0);
    }
    if let Some(p) = parts.next() {
        g.view = match p {
            "horizon" => GroundView::Horizon,
            "sun" => GroundView::Sun,
            "down" => GroundView::Down,
            "sky" => GroundView::Sky,
            _ => return Err(format!("unknown ground view {p:?} (horizon, sun, down, sky)")),
        };
    }
    if let Some(p) = parts.next() {
        g.height_m = num(p, "height")?.max(0.05);
    }
    if parts.next().is_some() {
        return Err(format!("too many parts in {s:?}"));
    }
    Ok(g)
}

/// Move the pilot to `start`; returns a description of where. Standing on
/// the ground needs the GPU (the terrain is defined in its shaders).
pub fn apply(world: &mut World, start: Start, gpu: Option<&Gpu>) -> Result<Option<String>, String> {
    match start {
        Start::Cluster => Ok(None),
        Start::Planet(p) => near_planet(world, p).map(Some),
        Start::Look { target, distance_pc } => look_at(world, target, distance_pc).map(Some),
        Start::Ground(g) => on_ground(world, g, gpu.ok_or("standing on a planet needs the GPU")?).map(Some),
    }
}

/// Stand on the Earth-like world (the default of [`PlanetStart`]) at a
/// latitude and local solar time: land there, `height_m` above the terrain
/// as the GPU evaluates it (or above the sea), looking as `g.view` says.
fn on_ground(world: &mut World, g: GroundStart, gpu: &Gpu) -> Result<String, String> {
    let (sys, i) = find_planet(world, KindFilter::Kind(PlanetKind::Ocean))?;
    let planet = sys.planets[i].clone();
    let body = &world.cluster.bodies[sys.star];
    let pref = PlanetRef { star: sys.star, generation: body.generation, planet: i };
    let probe = Probe::new(&gpu.device);
    // An island site is fixed on the ground: find it first, then let the
    // planet turn until it's the hour asked for there.
    let island = if g.site == Site::Island {
        let (q, hour_now) = find_island(world, &sys, i, &probe, gpu)?;
        let solar_day_s = solar_day(&planet, &sys);
        let wait_h = (g.hour - hour_now).rem_euclid(24.0);
        let t = world.cluster.t + wait_h / 24.0 * solar_day_s / SECONDS_PER_M;
        world.cluster.advance_to(t);
        world.pilot.x[0] = t;
        Some(q)
    } else {
        None
    };
    let body = &world.cluster.bodies[sys.star];
    // Fly in to a few radii, co-moving, so the physics takes up this system.
    let (off_km, vel_km_s) = sys.planet_state(i, world.cluster.t);
    let away = vec3::scale(vec3::normalize(vec3::scale(off_km, -1.0)), 3.0 * planet.radius_km);
    let at = vec3::add(body.position(), vec3::scale(vec3::add(off_km, away), 1.0 / KM_PER_M));
    let vel = vec3::add(geodesic::coordinate_velocity(&world.kerr, &body.state), vec3::scale(vel_km_s, 1.0 / C_KM_S));
    let mut pilot = Pilot::new(&world.kerr, at, vel, away, planet.spin_axis).ok_or("could not place the ship")?;
    pilot.x[0] = world.pilot.x[0];
    pilot.tau = world.pilot.tau;
    world.pilot = pilot;
    world.teleported();
    let l = world.system_of(pref).ok_or("the planet's system did not become the local one")?;

    // The site, in the planet's frame: latitude and hour angle from the
    // subsolar meridian (afternoon is east, where the ground turns to).
    let rel = l.planet_relative(i, world.pilot.x);
    let tc = rel.centre[0];
    let spin = planet.spin_axis;
    let to_star = vec3::normalize(vec3::scale(off_km, -1.0));
    let noon = vec3::normalize(vec3::axpy(to_star, -vec3::dot(to_star, spin), spin));
    let east = vec3::cross(spin, noon);
    let (lat, h) = (g.latitude_deg.to_radians(), ((g.hour - 12.0) * 15.0).to_radians());
    let up = vec3::add(
        vec3::scale(vec3::add(vec3::scale(noon, h.cos()), vec3::scale(east, h.sin())), lat.cos()),
        vec3::scale(spin, lat.sin()),
    );
    let angle = planet.rotation(tc * SECONDS_PER_M);
    let axes = terrain::body_axes(spin, angle);
    // The island's summit, where it is now.
    let up = island.map_or(up, |q| terrain::from_body(&axes, q));
    let terrain_at = |dirs: &[V3]| {
        let q: Vec<V3> = dirs.iter().map(|&d| terrain::to_body(&axes, d)).collect();
        probe.sample(&gpu.device, &gpu.queue, &planet, &q, 1e-4)
    };
    let dry = |s: &terrain::probe::Sample| s.fill != 1 || s.solid_km > 0.002;
    // For land or a coast, the nearest dry candidate on a 25° cap around
    // the spot (a Fibonacci spiral, nearest first).
    let mut sea_side = None;
    let up = match g.site {
        Site::Here => up,
        Site::Land | Site::Coast | Site::Island => {
            // Around an island summit look for its own shore (a cap of
            // ~150 km); elsewhere within 25°.
            let cap = if g.site == Site::Island { 150.0 / planet.radius_km } else { 25f64.to_radians() };
            let cands = cap_points(up, cap, 20_000);
            let samples = terrain_at(&cands);
            let land = samples.iter().position(dry).ok_or("no land near the spot")?;
            if g.site == Site::Land {
                cands[land]
            } else {
                // The nearest sea to that land, then the shore between them
                // by bisection, and 30 m back inland.
                let l = cands[land];
                let sea = cands
                    .iter()
                    .zip(&samples)
                    .filter(|(_, s)| !dry(s))
                    .map(|(c, _)| *c)
                    .max_by(|a, b| vec3::dot(*a, l).total_cmp(&vec3::dot(*b, l)))
                    .ok_or("no sea near the land")?;
                let (mut lo, mut hi) = (0.0, 1.0);
                for _ in 0..40 {
                    let mid = 0.5 * (lo + hi);
                    if dry(&terrain_at(&[slerp(l, sea, mid)])[0]) {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                let span = vec3::dot(l, sea).clamp(-1.0, 1.0).acos() * planet.radius_km;
                let back = (lo - 0.03 / span.max(1e-6)).max(0.0);
                sea_side = Some(slerp(l, sea, 1.0));
                slerp(l, sea, back)
            }
        }
    };
    let ground = terrain_at(&[up]);
    let ground = ground.first().ok_or("the terrain probe returned nothing")?;
    let r_km = planet.radius_km + ground.surface_km + g.height_m / 1000.0;
    let offset = vec3::rotate(vec3::scale(up, r_km / KM_PER_M), spin, -angle);

    // Face the view, then set down (landing keeps the attitude).
    let horizontal = |d: V3| vec3::normalize(vec3::axpy(d, -vec3::dot(d, up), up));
    let sun_h = horizontal(to_star);
    let tilt = |d: V3, deg: f64| {
        let a = deg.to_radians();
        vec3::add(vec3::scale(d, a.cos()), vec3::scale(up, a.sin()))
    };
    let away_sun = vec3::scale(sun_h, -1.0);
    let look = match (sea_side, g.view) {
        (Some(sea), GroundView::Horizon) => tilt(horizontal(vec3::sub(sea, up)), -5.0),
        (_, GroundView::Horizon) => tilt(vec3::cross(up, sun_h), -5.0),
        (_, GroundView::Sun) => sun_h,
        (_, GroundView::Down) => tilt(away_sun, -45.0),
        (_, GroundView::Sky) => tilt(away_sun, 30.0),
    };
    let coord = |v: V3| vec3::spatial(rel.frame.displacement(0.0, v));
    let here = world.pilot.position();
    let mut pilot = Pilot::new(&world.kerr, here, vel, coord(look), coord(up)).ok_or("could not turn the ship")?;
    pilot.x = world.pilot.x;
    pilot.tau = world.pilot.tau;
    world.pilot = pilot;
    if !world.land(pref, offset) {
        return Err("could not land".into());
    }
    world.cfg.time_scale = 1.0 / SECONDS_PER_M;
    let sun_elevation = vec3::dot(to_star, up).asin().to_degrees();
    let what = match (ground.fill, ground.solid_km < 0.0) {
        (1, true) => format!("on the sea ({:.0} m deep)", -ground.solid_km * 1000.0),
        _ => format!("on land {:.0} m above sea level", ground.surface_km * 1000.0),
    };
    let what = match g.site {
        Site::Coast => format!("{what}, 30 m from the shore, facing the sea"),
        Site::Island => format!("{what} on a young volcanic island, 30 m from the shore, facing the sea"),
        _ => what,
    };
    Ok(format!(
        "standing {what} on a {:?} world of {:.0} km radius, latitude {:.1}°, {:.1} h local time, sun {sun_elevation:.0}° up ({:?} view)",
        planet.kind, planet.radius_km, g.latitude_deg, g.hour, g.view
    ))
}

/// The summit (body-fixed direction) of the tallest young hotspot island
/// within 25° of the equator, and its local solar time now (hours).
fn find_island(world: &World, sys: &planets::System, i: usize, probe: &Probe, gpu: &Gpu) -> Result<(V3, f64), String> {
    let planet = &sys.planets[i];
    let tropics = cap_points([0.0, 0.0, 1.0], std::f64::consts::PI, 400_000)
        .into_iter()
        .filter(|q| q[2].abs() < 25f64.to_radians().sin())
        .collect::<Vec<_>>();
    let samples = probe.sample(&gpu.device, &gpu.queue, planet, &tropics, 5.0);
    let (q, _) = tropics
        .iter()
        .zip(&samples)
        .filter(|(_, s)| s.surface_km > 0.0 && s.hotspot[0] > 1.0)
        .max_by(|a, b| a.1.hotspot[0].total_cmp(&b.1.hotspot[0]))
        .ok_or("no volcanic islands in the tropics of this world")?;
    // Local solar time now: the angle between the site and the subsolar
    // meridian, in the planet frame.
    let t = world.cluster.t;
    let angle = planet.rotation(t * SECONDS_PER_M);
    let up = terrain::from_body(&terrain::body_axes(planet.spin_axis, angle), *q);
    let (off_km, _) = sys.planet_state(i, t);
    let to_star = vec3::normalize(vec3::scale(off_km, -1.0));
    let spin = planet.spin_axis;
    let noon = vec3::normalize(vec3::axpy(to_star, -vec3::dot(to_star, spin), spin));
    let east = vec3::cross(spin, noon);
    let h = vec3::dot(up, east).atan2(vec3::dot(up, noon));
    Ok((*q, (12.0 + h.to_degrees() / 15.0).rem_euclid(24.0)))
}

/// Length of the planet's solar day, seconds.
fn solar_day(planet: &planets::Planet, sys: &planets::System) -> f64 {
    let sidereal = planet.rotation_period_s;
    let year = planet.orbit.period_s(sys.star_gm());
    1.0 / (1.0 / sidereal - 1.0 / year)
}

/// `n` unit directions spread evenly over a cap of angular radius `radius`
/// around `centre`, nearest the centre first (a Fibonacci spiral).
fn cap_points(centre: V3, radius: f64, n: usize) -> Vec<V3> {
    let a = vec3::any_orthogonal(centre);
    let b = vec3::cross(centre, a);
    let golden = std::f64::consts::PI * (3.0 - 5f64.sqrt());
    (0..n)
        .map(|k| {
            // Equal-area steps in 1 − cos θ.
            let theta = (1.0 - (1.0 - radius.cos()) * (k as f64 + 0.5) / n as f64).acos();
            let phi = golden * k as f64;
            let side = vec3::add(vec3::scale(a, phi.cos()), vec3::scale(b, phi.sin()));
            vec3::add(vec3::scale(centre, theta.cos()), vec3::scale(side, theta.sin()))
        })
        .collect()
}

/// Great-circle interpolation between unit vectors.
fn slerp(a: V3, b: V3, t: f64) -> V3 {
    let omega = vec3::dot(a, b).clamp(-1.0, 1.0).acos();
    if omega < 1e-12 {
        return a;
    }
    let s = omega.sin();
    vec3::add(vec3::scale(a, ((1.0 - t) * omega).sin() / s), vec3::scale(b, (t * omega).sin() / s))
}

/// The nearest planet matching `kind` (for oceans, the Earth analogue: a
/// temperate ocean world around a Sun-like star, or failing that the
/// nearest planet with an atmosphere).
fn find_planet(world: &World, kind: KindFilter) -> Result<(planets::System, usize), String> {
    let seed = world.cluster.cfg.seed;
    let systems = planets::nearby(seed, &world.cluster.bodies, world.pilot.position(), f64::INFINITY);
    let find = |pred: &dyn Fn(&planets::Planet) -> bool| {
        systems.iter().find_map(|(_, s)| s.planets.iter().position(pred).map(|i| (s, i)))
    };
    let found = match kind {
        KindFilter::Kind(PlanetKind::Ocean) => systems
            .iter()
            .filter(|(_, s)| (5000.0..6500.0).contains(&s.star_temperature))
            .find_map(|(_, s)| {
                s.planets
                    .iter()
                    .position(|p| p.kind == PlanetKind::Ocean && (245.0..275.0).contains(&p.equilibrium_temperature))
                    .map(|i| (s, i))
            })
            .or_else(|| find(&|p| p.kind == PlanetKind::Ocean && (245.0..275.0).contains(&p.equilibrium_temperature)))
            .or_else(|| find(&|p| p.kind == PlanetKind::Ocean))
            .or_else(|| find(&|p| p.atmosphere.is_some() && !p.kind.is_giant())),
        KindFilter::Kind(k) => find(&|p| p.kind == k),
        KindFilter::Ringed => find(&|p| p.rings.is_some()),
        KindFilter::Any => find(&|_| true),
    };
    found.map(|(s, i)| (s.clone(), i)).ok_or(format!("no such planet ({kind:?}) in this cluster"))
}

/// Turn to face a nebula, first moving `distance_pc` from it towards Earth
/// (at rest) if given. Galactic north is up.
fn look_at(world: &mut World, target: usize, distance_pc: Option<f64>) -> Result<String, String> {
    let (name, centre_pc) = render_hq::nebula::landmarks()[target];
    let centre = vec3::scale(centre_pc, PARSEC);
    let (pos, vel) = match distance_pc {
        Some(d) => (vec3::axpy(centre, d * PARSEC, render_hq::nebula::earth_direction()), [0.0; 3]),
        None => {
            let u = world.pilot.e[0];
            (world.pilot.position(), [u[1] / u[0], u[2] / u[0], u[3] / u[0]])
        }
    };
    let look = vec3::sub(centre, pos);
    let mut pilot =
        Pilot::new(&world.kerr, pos, vel, look, render_hq::nebula::galactic().z).ok_or("could not place the ship")?;
    pilot.x[0] = world.pilot.x[0];
    pilot.tau = world.pilot.tau;
    world.pilot = pilot;
    Ok(format!("facing {name}, {:.2} pc away", vec3::norm(look) / PARSEC))
}

/// Put the ship on a circular orbit around the nearest planet matching
/// `start` (for the default, an ocean world, or failing that the nearest
/// planet with an atmosphere), at real-time warp. The orbit runs along the
/// horizontal part of the view direction (prograde when looking straight
/// down), so views of the horizon keep flying towards what they show.
fn near_planet(world: &mut World, start: PlanetStart) -> Result<String, String> {
    let (sys, i) = find_planet(world, start.kind)?;
    let planet = &sys.planets[i];
    let body = &world.cluster.bodies[sys.star];
    let (off_km, vel_km_s) = sys.planet_state(i, world.cluster.t);
    let radius = planet.radius_km;
    let altitude_km = if start.in_radii { start.altitude * radius } else { start.altitude };
    let (pos, look, up) = view_geometry(start.view, off_km, planet.spin_axis, radius, radius + altitude_km);
    let at = vec3::add(body.position(), vec3::scale(vec3::add(off_km, pos), 1.0 / KM_PER_M));
    let planet_vel =
        vec3::add(geodesic::coordinate_velocity(&world.kerr, &body.state), vec3::scale(vel_km_s, 1.0 / C_KM_S));
    // Circular speed (the star's tide and the other planets perturb it by
    // ≲ 10⁻⁵).
    let r_km = radius + altitude_km;
    let v_circ = (local::gm_to_m(planet.gm()) * KM_PER_M / r_km).sqrt();
    let radial = vec3::normalize(pos);
    let mut along = vec3::axpy(look, -vec3::dot(look, radial), radial);
    if vec3::norm(along) < 0.1 {
        along = vec3::cross(planet.spin_axis, radial);
    }
    let vel = vec3::axpy(planet_vel, v_circ, vec3::normalize(along));
    let mut pilot = Pilot::new(&world.kerr, at, vel, look, up).ok_or("could not place the ship")?;
    pilot.x[0] = world.pilot.x[0];
    pilot.tau = world.pilot.tau;
    world.pilot = pilot;
    world.cfg.time_scale = 1.0 / SECONDS_PER_M;
    Ok(format!(
        "orbiting {altitude_km:.0} km above a {:?} world of {radius:.0} km radius at {:.2} km/s, {:.2} AU from a {:.0} K star ({:?} view)",
        planet.kind,
        v_circ * C_KM_S,
        vec3::norm(off_km) / planets::AU_KM,
        sys.star_temperature,
        start.view,
    ))
}

/// Ship position relative to the planet's centre (km), look direction and
/// up hint for a view from distance `r` of a planet of radius `radius` at
/// `off_km` from its star.
fn view_geometry(view: View, off_km: V3, spin: V3, radius: f64, r: f64) -> (V3, V3, V3) {
    let to_star = vec3::normalize(vec3::scale(off_km, -1.0));
    let mut side = vec3::cross(spin, to_star);
    if vec3::norm(side) < 1e-3 {
        side = vec3::any_orthogonal(to_star);
    }
    let side = vec3::normalize(side);
    // Unit direction from the centre at angle `theta` from the subsolar
    // point towards the dawn terminator (`side`).
    let at_angle = |theta: f64| vec3::add(vec3::scale(to_star, theta.cos()), vec3::scale(side, theta.sin()));
    // Angle of the horizon below the local horizontal.
    let dip = (radius / r).min(1.0).acos();
    match view {
        View::Dawn => {
            // Look at the sun on the horizon, tilted down to put the limb in view.
            let a = 0.55 * dip;
            let look = vec3::sub(vec3::scale(to_star, a.cos()), vec3::scale(side, a.sin()));
            (vec3::scale(side, r), look, side)
        }
        View::Day => {
            // Sun 35° up; find the point whose normal bisects the directions
            // to the sun and to the ship.
            // It lies between the nadir and the subsolar point: bisect on the
            // angle from the nadir.
            let theta = 55f64.to_radians();
            let d = at_angle(theta);
            let ship = vec3::scale(d, r);
            let (mut lo, mut hi) = (0.0, theta);
            for _ in 0..60 {
                let mid = 0.5 * (lo + hi);
                let n = at_angle(theta - mid);
                let to_eye = vec3::normalize(vec3::sub(ship, vec3::scale(n, radius)));
                if vec3::dot(n, to_eye) > vec3::dot(n, to_star) {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            let n = at_angle(theta - 0.5 * (lo + hi));
            let look = vec3::normalize(vec3::sub(vec3::scale(n, radius), ship));
            (ship, look, d)
        }
        View::Limb => {
            // Sun 12° up and to the right; look along the horizon.
            let d = at_angle(78f64.to_radians());
            let across = vec3::normalize(vec3::cross(d, to_star));
            let a = 0.7 * dip;
            let look = vec3::sub(vec3::scale(across, a.cos()), vec3::scale(d, a.sin()));
            (vec3::scale(d, r), look, d)
        }
        View::Nadir => {
            let d = at_angle(60f64.to_radians());
            (vec3::scale(d, r), vec3::scale(d, -1.0), to_star)
        }
        View::Night => {
            let d = at_angle(125f64.to_radians());
            let toward = vec3::normalize(vec3::axpy(to_star, -vec3::dot(to_star, d), d));
            let a = dip + 0.25;
            let look = vec3::sub(vec3::scale(toward, a.cos()), vec3::scale(d, a.sin()));
            (vec3::scale(d, r), look, d)
        }
        View::Disc => {
            // 30° from the sun, raised 25° out of the equator (and ring) plane.
            let d = at_angle(30f64.to_radians());
            let d =
                vec3::normalize(vec3::axpy(vec3::axpy(d, -vec3::dot(d, spin), spin), 25f64.to_radians().tan(), spin));
            (vec3::scale(d, r), vec3::scale(d, -1.0), spin)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_planet_starts() {
        assert_eq!("cluster".parse::<Start>(), Ok(Start::Cluster));
        assert_eq!("planet".parse::<Start>(), Ok(Start::Planet(PlanetStart::default())));
        let Ok(Start::Planet(p)) = "planet:ringed".parse::<Start>() else { panic!() };
        assert_eq!((p.kind, p.altitude, p.in_radii, p.view), (KindFilter::Ringed, 2.0, true, View::Dawn));
        let Ok(Start::Planet(p)) = "planet:desert:1500:day".parse::<Start>() else { panic!() };
        assert_eq!((p.kind, p.altitude, p.in_radii), (KindFilter::Kind(PlanetKind::Desert), 1500.0, false));
        assert_eq!(p.view, View::Day);
        assert!("planet:moon".parse::<Start>().is_err());
        assert!("planet:ice:3r:disc:x".parse::<Start>().is_err());
        assert_eq!("ground".parse::<Start>(), Ok(Start::Ground(GroundStart::default())));
        let Ok(Start::Ground(g)) = "ground:-33:7.5:down:40".parse::<Start>() else { panic!() };
        assert_eq!((g.latitude_deg, g.hour, g.view, g.height_m), (-33.0, 7.5, GroundView::Down, 40.0));
        let Ok(Start::Ground(g)) = "ground:coast:10".parse::<Start>() else { panic!() };
        assert_eq!((g.site, g.latitude_deg, g.hour), (Site::Coast, 10.0, 15.0));
        assert!("ground:10:12:sideways".parse::<Start>().is_err());
    }

    /// The glint view looks down at a point on the surface where the sun's
    /// mirror image is.
    #[test]
    fn day_view_looks_at_the_specular_point() {
        let off = [1.0e8, 0.0, 0.0];
        let (radius, r) = (6000.0, 6420.0);
        let (ship, look, _) = view_geometry(View::Day, off, [0.0, 0.0, 1.0], radius, r);
        // Distance along `look` to the sphere.
        let b = vec3::dot(ship, look);
        let c = vec3::dot(ship, ship) - radius * radius;
        let t = -b - (b * b - c).sqrt();
        let s = vec3::axpy(ship, t, look);
        let n = vec3::normalize(s);
        let to_star = [-1.0, 0.0, 0.0];
        let refl = vec3::axpy(look, -2.0 * vec3::dot(look, n), n);
        assert!(vec3::dot(refl, to_star) > 0.9999, "{refl:?}");
    }

    /// How the ocean world's land divides among climates, by the aridity
    /// index `climate.wgsl` uses, against Earth's (UNEP: drylands are ~41 %
    /// of the land, 7 % hyper-arid, 11 % arid, 15 % semi-arid, 9 % dry
    /// sub-humid). Needs a GPU, so it's run by hand:
    /// `cargo test -p desktop --release -- --ignored --nocapture climate`.
    #[test]
    #[ignore = "needs a GPU"]
    fn ocean_world_climate() {
        let world = crate::world(384);
        let (n, cap) = (world.cluster.len() as u32, world.cluster.history.capacity() as u32);
        let gpu = pollster::block_on(Gpu::new(crate::instance(), None, (64, 64), n, cap)).unwrap();
        let (sys, i) = find_planet(&world, KindFilter::Kind(PlanetKind::Ocean)).unwrap();
        let planet = &sys.planets[i];
        let mut maps = terrain::maps::SurfaceMaps::new(&gpu.device);
        maps.bake(&gpu.device, &gpu.queue, (0, 0, i), planet);
        let climate = maps.read_climate(&gpu.device, &gpu.queue);
        let dirs = cap_points([0.0, 0.0, 1.0], std::f64::consts::PI, 200_000);
        let samples = Probe::new(&gpu.device).sample(&gpu.device, &gpu.queue, planet, &dirs, 20.0);

        // Classes: cold (below −8 °C), hyper-arid, arid, semi-arid, dry
        // sub-humid, humid; and per latitude band the mean rain and cover.
        let names = ["cold", "hyper-arid", "arid", "semi-arid", "dry sub-humid", "humid"];
        let mut classes = [0usize; 6];
        let mut bands = [[0.0f64; 3]; 4];
        let mut land = 0;
        for (q, s) in dirs.iter().zip(&samples) {
            if s.surface_km <= 0.0 {
                continue;
            }
            land += 1;
            let [t, rain, veg, _] = climate.at(*q);
            let t_c = t - 273.15;
            // Potential evaporation as `climate.wgsl` has it.
            let aridity = rain / (250.0 + 40.0 * t_c).clamp(100.0, 1800.0);
            let class = match aridity {
                _ if t_c < -8.0 => 0,
                a if a < 0.05 => 1,
                a if a < 0.2 => 2,
                a if a < 0.5 => 3,
                a if a < 0.65 => 4,
                _ => 5,
            };
            classes[class] += 1;
            let lat = q[2].abs().asin().to_degrees();
            let band = &mut bands[[15.0, 35.0, 60.0, 90.0].iter().position(|&b| lat < b).unwrap_or(3)];
            *band = [band[0] + 1.0, band[1] + rain as f64, band[2] + veg as f64];
        }
        println!("{} land of {} samples ({:.0} %)", land, dirs.len(), 100.0 * land as f64 / dirs.len() as f64);
        for (name, c) in names.iter().zip(classes) {
            println!("{name:>14}: {:5.1} %", 100.0 * c as f64 / land as f64);
        }
        for (b, name) in bands.iter().zip(["0–15°", "15–35°", "35–60°", "60–90°"]) {
            println!(
                "{name:>7}: land {:5.1} %, rain {:5.0} mm/yr, cover {:.2}",
                100.0 * b[0] / land as f64,
                b[1] / b[0],
                b[2] / b[0]
            );
        }
        let dry = classes[1..5].iter().sum::<usize>() as f64 / land as f64;
        assert!((0.2..0.6).contains(&dry), "drylands {:.0} % of the land", 100.0 * dry);
        assert!(classes[5] as f64 / land as f64 > 0.25, "too little humid land");
    }

    /// The tile generator on the GPU, at the showcase island: coarse tiles
    /// agree with the probe, refined tiles stay near their parents, and
    /// neighbours agree on their shared edge. Needs a GPU, so it's run by
    /// hand: `cargo test -p desktop --release -- --ignored --nocapture tiles`.
    #[test]
    #[ignore = "needs a GPU"]
    fn tiles_on_the_gpu() {
        use render_hq::terrain::anchor::Anchor;
        use render_hq::terrain::tilegen::{self, TileGen};
        use render_hq::terrain::tiles::{Side, TileId};
        let world = crate::world(384);
        let (n, cap) = (world.cluster.len() as u32, world.cluster.history.capacity() as u32);
        let gpu = pollster::block_on(Gpu::new(crate::instance(), None, (64, 64), n, cap)).unwrap();
        let (sys, i) = find_planet(&world, KindFilter::Kind(PlanetKind::Ocean)).unwrap();
        let planet = &sys.planets[i];
        let probe = Probe::new(&gpu.device);
        let (site, _) = find_island(&world, &sys, i, &probe, &gpu).unwrap();
        let anchor = Anchor::new(vec3::scale(site, planet.radius_km), tilegen::octaves(planet.radius_km, planet.seed));
        let mut tile_gen = TileGen::new(&gpu.device);
        let target = TileId::containing(site, 16);
        let east = target.neighbour(Side::East);
        let mut tiles = Vec::new();
        for t in [target, east] {
            let mut at = Some(t);
            while let Some(a) = at {
                if !tiles.contains(&a) {
                    tiles.push(a);
                }
                at = a.parent();
            }
        }
        let done = tile_gen.generate(&gpu.device, &gpu.queue, (0, 0, i), planet, &anchor, &tiles, None);
        assert_eq!(done.len(), tiles.len(), "every tile generated");
        let layer = |t: TileId| done.iter().find(|d| d.0 == t).map(|d| d.1).unwrap();
        let heights = |t: TileId| tile_gen.read_heights(&gpu.device, &gpu.queue, layer(t));

        // Coarse tiles evaluate the planet's terrain, as the probe does.
        let coarse = TileId::containing(site, 11);
        let hc = heights(coarse);
        let points: Vec<(i32, i32)> = (0..=8).flat_map(|a| (0..=8).map(move |b| (16 * a, 16 * b))).collect();
        let dirs: Vec<V3> = points.iter().map(|&(a, b)| coarse.direction(a as f64 / 128.0, b as f64 / 128.0)).collect();
        let lod = 2.0 * tilegen::spacing_km(planet.radius_km, 11);
        let probed = probe.sample(&gpu.device, &gpu.queue, planet, &dirs, lod);
        let worst = points
            .iter()
            .zip(&probed)
            .map(|(&(a, b), s)| (TileGen::at(&hc, a, b) as f64 - s.solid_km).abs())
            .fold(0.0, f64::max);
        println!("level 11 vs probe: worst {:.3} m", worst * 1000.0);
        assert!(worst < 1e-3, "{worst} km");

        // A refined tile stays near its parent: they differ by this level's
        // octave only, at the parent's own samples.
        let parent = target.parent().unwrap();
        let (ht, hp) = (heights(target), heights(parent));
        let (qx, qy) = ((target.x & 1) as i32 * 64, (target.y & 1) as i32 * 64);
        let mut diff = 0.0f32;
        for a in (0..=128).step_by(2) {
            for b in (0..=128).step_by(2) {
                diff = diff.max((TileGen::at(&ht, a, b) - TileGen::at(&hp, qx + a / 2, qy + b / 2)).abs());
            }
        }
        println!("level 16 vs its parent: up to {:.3} m", diff * 1000.0);
        assert!(diff < 0.02, "{diff} km");

        // Neighbours agree on their shared edge.
        let he = heights(east);
        let mut edge = 0.0f32;
        for b in 0..=128 {
            let theirs = if east.face == target.face { TileGen::at(&he, 0, b) } else { continue };
            edge = edge.max((TileGen::at(&ht, 128, b) - theirs).abs());
        }
        println!("shared edge: up to {:.3} mm", edge * 1e6);
        assert!(edge < 1e-6, "{edge} km");
        let (lo, hi) = ht.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(l, h), &v| (l.min(v), h.max(v)));
        println!("level 16 tile at the island: {lo:.4} to {hi:.4} km");
    }
}
