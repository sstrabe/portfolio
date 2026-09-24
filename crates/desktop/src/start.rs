//! Where a flight begins.

use kerr::geodesic;
use kerr::pilot::Pilot;
use kerr::planets::{self, C_KM_S, KM_PER_M, PlanetKind};
use kerr::units::SECONDS_PER_M;
use kerr::vec3::{self, V3};
use kerr::world::World;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Start {
    /// On a circular orbit 800 AU from the hole, facing prograde.
    Cluster,
    /// Next to the nearest planet of a kind (by default 420 km above an
    /// Earth-like world at dawn, looking at the sunrise over its limb).
    Planet(PlanetStart),
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
            _ => return Err(format!("unknown start {s:?} (cluster or planet[:KIND[:ALTITUDE[:VIEW]]])")),
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

/// Move the pilot to `start`; returns a description of where.
pub fn apply(world: &mut World, start: Start) -> Result<Option<String>, String> {
    match start {
        Start::Cluster => Ok(None),
        Start::Planet(p) => near_planet(world, p).map(Some),
    }
}

/// Park near the nearest planet matching `start` (for the default, an
/// ocean world, or failing that the nearest planet with an atmosphere),
/// co-moving with it, at real-time warp.
fn near_planet(world: &mut World, start: PlanetStart) -> Result<String, String> {
    let seed = world.cluster.cfg.seed;
    let systems = planets::nearby(seed, &world.cluster.bodies, world.pilot.position(), f64::INFINITY);
    let find = |pred: &dyn Fn(&planets::Planet) -> bool| {
        systems.iter().find_map(|(_, s)| s.planets.iter().position(pred).map(|i| (s, i)))
    };
    let found = match start.kind {
        KindFilter::Kind(PlanetKind::Ocean) => {
            find(&|p| p.kind == PlanetKind::Ocean).or_else(|| find(&|p| p.atmosphere.is_some() && !p.kind.is_giant()))
        }
        KindFilter::Kind(k) => find(&|p| p.kind == k),
        KindFilter::Ringed => find(&|p| p.rings.is_some()),
        KindFilter::Any => find(&|_| true),
    };
    let (sys, i) = found.ok_or(format!("no such planet ({:?}) in this cluster", start.kind))?;
    let planet = &sys.planets[i];
    let body = &world.cluster.bodies[sys.star];
    let (off_km, vel_km_s) = sys.planet_state(i, world.cluster.t);
    let radius = planet.radius_km;
    let altitude_km = if start.in_radii { start.altitude * radius } else { start.altitude };
    let (pos, look, up) = view_geometry(start.view, off_km, planet.spin_axis, radius, radius + altitude_km);
    let at = vec3::add(body.position(), vec3::scale(vec3::add(off_km, pos), 1.0 / KM_PER_M));
    let vel = vec3::add(geodesic::coordinate_velocity(&world.kerr, &body.state), vec3::scale(vel_km_s, 1.0 / C_KM_S));
    let mut pilot = Pilot::new(&world.kerr, at, vel, look, up).ok_or("could not place the ship")?;
    pilot.x[0] = world.pilot.x[0];
    pilot.tau = world.pilot.tau;
    world.pilot = pilot;
    world.cfg.time_scale = 1.0 / SECONDS_PER_M;
    Ok(format!(
        "{altitude_km:.0} km above a {:?} world of {radius:.0} km radius, {:.2} AU from a {:.0} K star ({:?} view)",
        planet.kind,
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
            let d = at_angle(30f64.to_radians());
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
}
