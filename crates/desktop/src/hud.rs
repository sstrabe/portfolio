//! Names, numbers and notes for the HUD and the terminal.

use crate::input::ENGINE;
use kerr::local::PlanetRef;
use kerr::planets::{self, AU_KM, PlanetKind};
use kerr::units::G0;
use kerr::world::{PlanetTelemetry, Telemetry, World, WorldEvent};

pub fn kind_name(kind: PlanetKind) -> &'static str {
    match kind {
        PlanetKind::Rocky => "rocky world",
        PlanetKind::Ocean => "ocean world",
        PlanetKind::Desert => "desert world",
        PlanetKind::Ice => "ice world",
        PlanetKind::Lava => "lava world",
        PlanetKind::GasGiant => "gas giant",
        PlanetKind::IceGiant => "ice giant",
    }
}

/// "ocean world 2", numbering planets from the star outwards.
pub fn planet_name(world: &World, p: PlanetRef) -> String {
    let body = &world.cluster.bodies[p.star];
    let kind = world.system_of(p).and_then(|l| l.planet(p.planet).map(|pl| pl.kind)).or_else(|| {
        planets::system(world.cluster.cfg.seed, p.star, body).and_then(|s| s.planets.get(p.planet).map(|pl| pl.kind))
    });
    kind.map_or_else(|| "planet".into(), |k| format!("{} {}", kind_name(k), p.planet + 1))
}

pub fn km(x: f64) -> String {
    match x.abs() {
        a if a < 100.0 => format!("{x:.1} km"),
        a if a < 1e6 => format!("{x:.0} km"),
        a if a < 100.0 * AU_KM => format!("{:.2} AU", x / AU_KM),
        _ => format!("{:.0} AU", x / AU_KM),
    }
}

pub fn seconds(s: f64) -> String {
    match s {
        s if s < 120.0 => format!("{s:.0} s"),
        s if s < 7200.0 => format!("{:.1} min", s / 60.0),
        s if s < 2.0 * 86400.0 => format!("{:.1} h", s / 3600.0),
        s if s < 2.0 * 3.156e7 => format!("{:.1} d", s / 86400.0),
        s => format!("{:.1} yr", s / 3.156e7),
    }
}

/// Altitude, speed and orbit relative to a planet.
pub fn planet(world: &World, p: &PlanetTelemetry) -> String {
    let name = planet_name(world, p.planet);
    let name = if p.targeted { format!("▸ {name}") } else { name };
    if p.landed {
        return format!("landed on {name}");
    }
    let mut s = format!("{name} · alt {} · {:.2} km/s", km(p.altitude_km), p.speed_km_s);
    if p.period_s.is_finite() {
        s += &format!(" · orbit {} ({}–{})", seconds(p.period_s), km(p.periapsis_km), km(p.apoapsis_km));
    } else {
        s += " · unbound";
    }
    if p.periapsis_km < 0.0 && p.vertical_km_s < 0.0 {
        s += " · ⚠ ON A COLLISION COURSE";
    }
    s
}

/// Acceleration, in g.
pub fn gees(g: f64) -> String {
    if g < 10.0 { format!("{g:.2} g") } else { format!("{g:.0} g") }
}

/// The engine's thrust limit and the acceleration at full throttle.
pub fn throttle(t: &Telemetry) -> String {
    format!("thrust limit {:.0e} (full throttle {})", t.throttle, gees(ENGINE * t.thrust_g))
}

/// A short note for a world event, if it deserves one.
pub fn note(world: &World, ev: &WorldEvent) -> Option<String> {
    Some(match *ev {
        WorldEvent::HorizonCrossed => "you crossed the event horizon: respawned".into(),
        WorldEvent::Landed { speed } if speed < 0.1 => format!("touched down at {:.0} m/s", speed * 1000.0),
        WorldEvent::Landed { speed } => format!("hit the ground at {speed:.2} km/s (the hull held)"),
        WorldEvent::StarContact => "flew into the star: moved back out to 10 stellar radii".into(),
        WorldEvent::OrbitReached(p) => format!("in orbit around {}: autopilot off", planet_name(world, p)),
        WorldEvent::AutopilotOff => "autopilot off".into(),
        WorldEvent::Throttle(x) => {
            format!("thrust limit {x:.0e}: full throttle {}", gees(ENGINE * world.cfg.thrust * x / G0))
        }
        WorldEvent::WarpLimited(w) => format!("time warp lowered to ×{w:.0} to keep orbits watchable"),
        _ => return None,
    })
}
