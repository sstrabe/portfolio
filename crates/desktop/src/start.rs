//! Where a flight begins.

use kerr::geodesic;
use kerr::local;
use kerr::pilot::Pilot;
use kerr::planets::{self, C_KM_S, KM_PER_M, PlanetKind};
use kerr::units::SECONDS_PER_M;
use kerr::vec3;
use kerr::world::World;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Start {
    /// On a circular orbit 800 AU from the hole, facing prograde.
    Cluster,
    /// In a low circular orbit around the nearest Earth-like world, at the
    /// terminator, heading into the sunrise over its limb.
    Planet,
}

impl std::str::FromStr for Start {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "cluster" => Ok(Self::Cluster),
            "planet" => Ok(Self::Planet),
            _ => Err(format!("unknown start {s:?} (cluster or planet)")),
        }
    }
}

/// Move the pilot to `start`; returns a description of where.
pub fn apply(world: &mut World, start: Start) -> Result<Option<String>, String> {
    match start {
        Start::Cluster => Ok(None),
        Start::Planet => above_planet(world, 420.0).map(Some),
    }
}

/// Put the ship `altitude_km` above the terminator of the nearest ocean
/// world (or, failing that, the nearest planet with an atmosphere), on a
/// circular orbit heading towards the star so the sun rises over the limb
/// ahead, at real-time warp.
fn above_planet(world: &mut World, altitude_km: f64) -> Result<String, String> {
    let seed = world.cluster.cfg.seed;
    let systems = planets::nearby(seed, &world.cluster.bodies, world.pilot.position(), f64::INFINITY);
    let find = |pred: &dyn Fn(&planets::Planet) -> bool| {
        systems.iter().find_map(|(_, s)| s.planets.iter().position(pred).map(|i| (s, i)))
    };
    let (sys, i) = find(&|p| p.kind == PlanetKind::Ocean)
        .or_else(|| find(&|p| p.atmosphere.is_some() && !p.kind.is_giant()))
        .ok_or("no planet with an atmosphere in this cluster")?;
    let planet = &sys.planets[i];
    let body = &world.cluster.bodies[sys.star];
    let (off_km, vel_km_s) = sys.planet_state(i, world.cluster.t);
    let to_star = vec3::normalize(vec3::scale(off_km, -1.0));
    let mut side = vec3::cross(planet.spin_axis, to_star);
    if vec3::norm(side) < 1e-3 {
        side = vec3::any_orthogonal(to_star);
    }
    let side = vec3::normalize(side);
    let r = planet.radius_km + altitude_km;
    // Look at the sun on the horizon, tilted down to put the limb in view.
    let dip = (planet.radius_km / r).acos();
    let a = 0.55 * dip;
    let look = vec3::sub(vec3::scale(to_star, a.cos()), vec3::scale(side, a.sin()));
    let at = vec3::add(body.position(), vec3::scale(vec3::axpy(off_km, r, side), 1.0 / KM_PER_M));
    // Circular speed around the planet (the star's tide and the other
    // planets perturb it by ≲ 10⁻⁵), towards the star.
    let v_circ = (local::gm_to_m(planet.gm()) * KM_PER_M / r).sqrt();
    let planet_vel =
        vec3::add(geodesic::coordinate_velocity(&world.kerr, &body.state), vec3::scale(vel_km_s, 1.0 / C_KM_S));
    let vel = vec3::axpy(planet_vel, v_circ, to_star);
    let mut pilot = Pilot::new(&world.kerr, at, vel, look, side).ok_or("could not place the ship")?;
    pilot.x[0] = world.pilot.x[0];
    pilot.tau = world.pilot.tau;
    world.pilot = pilot;
    world.cfg.time_scale = 1.0 / SECONDS_PER_M;
    Ok(format!(
        "orbiting {altitude_km:.0} km above a {:?} world of {:.0} km radius at {:.2} km/s, {:.2} AU from a {:.0} K star",
        planet.kind,
        planet.radius_km,
        v_circ * C_KM_S,
        vec3::norm(off_km) / planets::AU_KM,
        sys.star_temperature
    ))
}
