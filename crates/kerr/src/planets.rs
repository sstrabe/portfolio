//! Planetary systems around cluster stars.
//!
//! Planets are generated on demand from a star's slot and generation, so a
//! star always has the same planets and nothing is stored per frame. They
//! move on Keplerian orbits in the star's rest frame: at 10³–10⁴ AU from
//! Sagittarius A* the hole's tide only matters through the Hill radius
//! `r (m / 3M)^{1/3}` (about 4 AU for a Sun-like star at 1000 AU), inside
//! which every orbit is placed.
//!
//! Units here are SI-flavoured: kilometres, seconds, kilograms. Everything
//! else in the crate uses units of the hole's mass `M`; [`KM_PER_M`]
//! converts.

use crate::cluster::{Body, BodyKind};
use crate::rng::Rng;
use crate::units;
use crate::vec3::{self, V3};

/// Kilometres per unit of `M` (6.35 million km for Sgr A*).
pub const KM_PER_M: f64 = units::METRES_PER_M / 1000.0;
pub const AU_KM: f64 = 1.495_978_707e8;
pub const EARTH_RADIUS_KM: f64 = 6371.0;
pub const SUN_RADIUS_KM: f64 = 6.957e5;
pub const EARTH_MASS_KG: f64 = 5.972e24;
pub const SUN_MASS_KG: f64 = 1.989e30;
pub const SUN_LUMINOSITY_W: f64 = 3.828e26;
/// Newton's constant in km³ kg⁻¹ s⁻².
pub const G_KM: f64 = 6.674_30e-20;
/// Speed of light, km/s.
pub const C_KM_S: f64 = 299_792.458;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PlanetKind {
    /// Bare or thinly veiled rock.
    Rocky,
    /// Earth-like: mostly ocean, continents, clouds.
    Ocean,
    /// Dry, dusty, rust coloured.
    Desert,
    /// Frozen surface of ice and snow.
    Ice,
    /// Close to its star, with molten seas glowing on the night side.
    Lava,
    /// Jupiter-like banded hydrogen giant.
    GasGiant,
    /// Neptune-like, blue from methane absorbing red light.
    IceGiant,
}

impl PlanetKind {
    pub fn is_giant(self) -> bool {
        matches!(self, Self::GasGiant | Self::IceGiant)
    }

    /// Stable small integer for the GPU.
    pub fn index(self) -> u32 {
        self as u32
    }
}

/// Optical makeup of an atmosphere, relative to Earth's where noted.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Atmosphere {
    /// Altitude where the atmosphere is treated as ending, km.
    pub top_km: f64,
    /// Rayleigh (molecular) scattering: density at the surface relative to
    /// Earth's sea-level air, and its exponential scale height.
    pub rayleigh_density: f64,
    pub rayleigh_scale_height_km: f64,
    /// Aerosols and dust (Mie scattering), relative to Earth's.
    pub mie_density: f64,
    pub mie_scale_height_km: f64,
    /// Henyey–Greenstein asymmetry of the aerosols.
    pub mie_g: f64,
    /// Fraction of aerosol extinction that is absorption.
    pub mie_absorption: f64,
    /// Aerosol colour: 0 neutral haze, 1 red-brown dust.
    pub dust: f64,
    /// Ozone column relative to Earth's (absorbs orange, keeps the sky blue
    /// at twilight).
    pub ozone: f64,
    /// Methane column in units that make 1 look like Neptune (absorbs red).
    pub methane: f64,
    pub clouds: Option<Clouds>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Clouds {
    /// Fraction of the sky covered, 0–1.
    pub coverage: f64,
    /// Cloud base and thickness, km.
    pub base_km: f64,
    pub thickness_km: f64,
    /// Vertical optical depth of a fully cloudy column.
    pub optical_depth: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rings {
    pub inner_km: f64,
    pub outer_km: f64,
    /// Normal optical depth of the densest part.
    pub optical_depth: f64,
}

/// A Keplerian orbit around the star, oriented in the simulation's
/// coordinate axes (which the star's rest frame shares to within v/c).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Orbit {
    pub a_km: f64,
    pub e: f64,
    /// Unit vectors towards periapsis and 90° ahead of it in the orbit.
    pub p: V3,
    pub q: V3,
    /// Mean anomaly at t = 0.
    pub mean_anomaly0: f64,
}

impl Orbit {
    /// Mean motion (rad/s) for a star of gravitational parameter `gm`.
    pub fn mean_motion(&self, gm: f64) -> f64 {
        (gm / (self.a_km * self.a_km * self.a_km)).sqrt()
    }

    pub fn period_s(&self, gm: f64) -> f64 {
        std::f64::consts::TAU / self.mean_motion(gm)
    }

    /// Position (km) and velocity (km/s) relative to the star at time
    /// `t_s` seconds.
    pub fn state(&self, gm: f64, t_s: f64) -> (V3, V3) {
        let n = self.mean_motion(gm);
        let m = (self.mean_anomaly0 + n * t_s).rem_euclid(std::f64::consts::TAU);
        let e = self.e;
        let mut ea = if e < 0.8 { m } else { std::f64::consts::PI };
        for _ in 0..30 {
            let f = ea - e * ea.sin() - m;
            let d = f / (1.0 - e * ea.cos());
            ea -= d;
            if d.abs() < 1e-14 {
                break;
            }
        }
        let (s, c) = ea.sin_cos();
        let b = (1.0 - e * e).sqrt();
        let x = self.a_km * (c - e);
        let y = self.a_km * b * s;
        let rate = n / (1.0 - e * c);
        let vx = -self.a_km * s * rate;
        let vy = self.a_km * b * c * rate;
        let pos = vec3::add(vec3::scale(self.p, x), vec3::scale(self.q, y));
        let vel = vec3::add(vec3::scale(self.p, vx), vec3::scale(self.q, vy));
        (pos, vel)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Planet {
    pub kind: PlanetKind,
    /// Seed for the procedural surface.
    pub seed: u32,
    pub radius_km: f64,
    pub mass_kg: f64,
    pub orbit: Orbit,
    /// Sidereal rotation period, s.
    pub rotation_period_s: f64,
    /// Rotation axis (unit, coordinate axes).
    pub spin_axis: V3,
    /// Rotation angle at t = 0.
    pub rotation0: f64,
    /// Peak-to-trough terrain relief, km (0 for giants).
    pub relief_km: f64,
    /// Fraction of the terrain height range under water (or lava).
    pub sea_level: f64,
    /// Radiative equilibrium temperature, K.
    pub equilibrium_temperature: f64,
    pub atmosphere: Option<Atmosphere>,
    pub rings: Option<Rings>,
}

impl Planet {
    /// Gravitational parameter, km³/s².
    pub fn gm(&self) -> f64 {
        G_KM * self.mass_kg
    }

    /// Rotation angle about [`Planet::spin_axis`] at `t_s`.
    pub fn rotation(&self, t_s: f64) -> f64 {
        (self.rotation0 + std::f64::consts::TAU * t_s / self.rotation_period_s).rem_euclid(std::f64::consts::TAU)
    }

    /// Where gravity of the planet dominates the star's (Hill sphere), km.
    pub fn hill_radius_km(&self, star_mass_kg: f64) -> f64 {
        self.orbit.a_km * (1.0 - self.orbit.e) * (self.mass_kg / (3.0 * star_mass_kg)).cbrt()
    }
}

/// A star's planets, with the star's properties in the same units.
#[derive(Clone, Debug, PartialEq)]
pub struct System {
    /// Body slot of the star in the cluster.
    pub star: usize,
    pub generation: u32,
    pub star_mass_kg: f64,
    pub star_radius_km: f64,
    pub star_temperature: f64,
    pub star_luminosity_w: f64,
    pub planets: Vec<Planet>,
}

impl System {
    pub fn star_gm(&self) -> f64 {
        G_KM * self.star_mass_kg
    }

    /// Planet `i`'s position (km) and velocity (km/s) relative to the star
    /// at coordinate time `t` (units of M).
    pub fn planet_state(&self, i: usize, t: f64) -> (V3, V3) {
        self.planets[i].orbit.state(self.star_gm(), t * units::SECONDS_PER_M)
    }

    /// Stellar flux at distance `d_km`, W/m².
    pub fn flux_at(&self, d_km: f64) -> f64 {
        let d_m = d_km * 1000.0;
        self.star_luminosity_w / (4.0 * std::f64::consts::PI * d_m * d_m)
    }
}

/// The planets of the star in slot `star` (deterministic in the world
/// `seed`, the slot and the body's generation), or `None` for giants,
/// compact objects, stations and stars whose Hill sphere is too small.
pub fn system(seed: u64, star: usize, body: &Body) -> Option<System> {
    let p = &body.params;
    if p.kind != BodyKind::Star || p.radius <= 0.0 {
        return None;
    }
    let mass_kg = p.mass / units::MSUN * SUN_MASS_KG;
    let radius_km = p.radius * KM_PER_M;
    let lum = p.luminosity;
    // Giants have swallowed their inner systems; very massive stars are too
    // short-lived to have formed planets.
    if radius_km > 3.0 * SUN_RADIUS_KM || mass_kg > 3.0 * SUN_MASS_KG {
        return None;
    }
    let mut rng = Rng::new(seed ^ (star as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ ((body.generation as u64) << 48));
    if rng.uniform() > 0.75 {
        return None;
    }
    // Orbits must stay well inside the Hill sphere around the hole, taken
    // at the star's current distance.
    let r_km = vec3::norm(body.position()) * KM_PER_M;
    let hill_km = r_km * (p.mass / 3.0).cbrt();
    let a_max = (0.4 * hill_km).min(40.0 * AU_KM);
    let a_min = (0.03 * AU_KM * lum.sqrt()).max(4.0 * radius_km);
    if a_min > a_max {
        return None;
    }

    // A common orbital plane, with small inclinations about it.
    let normal = rng.unit_vector();
    let count = 1 + (rng.uniform() * 6.0) as usize;
    let snow_line_au = 2.7 * lum.sqrt();
    let mut planets = Vec::with_capacity(count);
    let mut a = a_min * rng.range(1.0, 2.5);
    for i in 0..count {
        if a > a_max {
            break;
        }
        let a_au = a / AU_KM;
        let insolation = lum / (a_au * a_au);
        let kind = choose_kind(&mut rng, insolation, a_au > snow_line_au);
        planets.push(make_planet(&mut rng, kind, a, normal, insolation, seed as u32 ^ (star as u32 * 131 + i as u32)));
        a *= rng.range(1.5, 2.2);
    }
    if planets.is_empty() {
        return None;
    }
    Some(System {
        star,
        generation: body.generation,
        star_mass_kg: mass_kg,
        star_radius_km: radius_km,
        star_temperature: p.temperature,
        star_luminosity_w: lum * SUN_LUMINOSITY_W,
        planets,
    })
}

fn choose_kind(rng: &mut Rng, insolation: f64, beyond_snow_line: bool) -> PlanetKind {
    use PlanetKind::*;
    let x = rng.uniform();
    if beyond_snow_line {
        return if x < 0.45 {
            GasGiant
        } else if x < 0.7 {
            IceGiant
        } else {
            Ice
        };
    }
    match insolation {
        s if s > 4.0 => {
            if x < 0.2 {
                GasGiant
            } else {
                Lava
            }
        }
        s if s > 1.8 => {
            if x < 0.6 {
                Desert
            } else {
                Rocky
            }
        }
        s if s > 0.3 => {
            if x < 0.55 {
                Ocean
            } else if x < 0.8 {
                Rocky
            } else {
                Desert
            }
        }
        _ => {
            if x < 0.6 {
                Ice
            } else {
                Rocky
            }
        }
    }
}

fn make_planet(rng: &mut Rng, kind: PlanetKind, a_km: f64, normal: V3, insolation: f64, seed: u32) -> Planet {
    use PlanetKind::*;
    let (r_lo, r_hi) = match kind {
        Rocky => (0.4, 1.6),
        Ocean => (0.8, 1.5),
        Desert => (0.5, 1.3),
        Ice => (0.3, 1.2),
        Lava => (0.6, 1.8),
        GasGiant => (8.0, 12.5),
        IceGiant => (3.4, 4.6),
    };
    let radius_re = rng.range(r_lo, r_hi);
    let mass_me = match kind {
        GasGiant => rng.range(60.0, 1200.0),
        IceGiant => rng.range(12.0, 22.0),
        _ => radius_re.powf(3.7),
    };

    // Orbit: nearly circular, a few degrees off the common plane.
    let tilt = rng.range(0.0, 3.0f64.to_radians());
    let node = vec3::rotate(vec3::any_orthogonal(normal), normal, rng.range(0.0, std::f64::consts::TAU));
    let plane_normal = vec3::rotate(normal, node, tilt);
    let arg = rng.range(0.0, std::f64::consts::TAU);
    let p = vec3::normalize(vec3::rotate(node, plane_normal, arg));
    let q = vec3::cross(plane_normal, p);
    let orbit =
        Orbit { a_km, e: 0.1 * rng.uniform().powi(2), p, q, mean_anomaly0: rng.range(0.0, std::f64::consts::TAU) };

    let obliquity = if kind.is_giant() { rng.range(0.0, 0.5) } else { rng.range(0.0, 0.45) };
    let spin_axis = vec3::rotate(plane_normal, vec3::any_orthogonal(plane_normal), obliquity);
    let rotation_period_h = if kind.is_giant() { rng.range(9.0, 17.0) } else { rng.range(10.0, 40.0) };
    let (relief_km, sea_level) = match kind {
        Ocean => (rng.range(6.0, 10.0), rng.range(0.55, 0.75)),
        Rocky => (rng.range(4.0, 12.0), if rng.uniform() < 0.3 { rng.range(0.05, 0.3) } else { 0.0 }),
        Desert => (rng.range(3.0, 9.0), 0.0),
        Ice => (rng.range(2.0, 6.0), 0.0),
        Lava => (rng.range(3.0, 7.0), rng.range(0.2, 0.45)),
        GasGiant | IceGiant => (0.0, 0.0),
    };
    let rings = (kind.is_giant() && rng.uniform() < if kind == GasGiant { 0.4 } else { 0.25 }).then(|| {
        let r = radius_re * EARTH_RADIUS_KM;
        let inner = r * rng.range(1.25, 1.6);
        Rings { inner_km: inner, outer_km: inner + r * rng.range(0.5, 1.2), optical_depth: rng.range(0.3, 1.5) }
    });
    Planet {
        kind,
        seed,
        radius_km: radius_re * EARTH_RADIUS_KM,
        mass_kg: mass_me * EARTH_MASS_KG,
        orbit,
        rotation_period_s: rotation_period_h * 3600.0,
        spin_axis,
        rotation0: rng.range(0.0, std::f64::consts::TAU),
        relief_km,
        sea_level,
        equilibrium_temperature: 278.6 * insolation.powf(0.25),
        atmosphere: make_atmosphere(rng, kind),
        rings,
    }
}

fn make_atmosphere(rng: &mut Rng, kind: PlanetKind) -> Option<Atmosphere> {
    use PlanetKind::*;
    let earth = Atmosphere {
        top_km: 100.0,
        rayleigh_density: 1.0,
        rayleigh_scale_height_km: 8.0,
        mie_density: 1.0,
        mie_scale_height_km: 1.2,
        mie_g: 0.8,
        mie_absorption: 0.1,
        dust: 0.0,
        ozone: 1.0,
        methane: 0.0,
        clouds: None,
    };
    let clouds = |rng: &mut Rng, lo: f64, hi: f64| {
        Some(Clouds {
            coverage: rng.range(lo, hi),
            base_km: rng.range(1.5, 3.0),
            thickness_km: rng.range(2.0, 8.0),
            optical_depth: rng.range(8.0, 30.0),
        })
    };
    match kind {
        Ocean => Some(Atmosphere {
            rayleigh_density: rng.range(0.7, 1.6),
            mie_density: rng.range(0.5, 2.0),
            ozone: rng.range(0.4, 1.3),
            clouds: clouds(rng, 0.35, 0.7),
            ..earth
        }),
        Rocky if rng.uniform() < 0.5 => Some(Atmosphere {
            rayleigh_density: rng.range(0.05, 0.6),
            mie_density: rng.range(0.2, 1.5),
            ozone: 0.0,
            clouds: clouds(rng, 0.0, 0.25),
            ..earth
        }),
        Desert => Some(Atmosphere {
            rayleigh_density: rng.range(0.2, 1.2),
            mie_density: rng.range(3.0, 10.0),
            mie_scale_height_km: rng.range(1.5, 4.0),
            mie_g: 0.7,
            mie_absorption: 0.3,
            dust: rng.range(0.6, 1.0),
            ozone: 0.0,
            clouds: clouds(rng, 0.0, 0.12),
            ..earth
        }),
        Ice if rng.uniform() < 0.4 => {
            Some(Atmosphere { rayleigh_density: rng.range(0.02, 0.3), mie_density: 0.3, ozone: 0.0, ..earth })
        }
        Lava => Some(Atmosphere {
            rayleigh_density: rng.range(0.2, 0.6),
            mie_density: rng.range(2.0, 5.0),
            mie_absorption: 0.4,
            dust: 0.3,
            ozone: 0.0,
            ..earth
        }),
        GasGiant => Some(Atmosphere {
            top_km: 500.0,
            rayleigh_density: rng.range(2.0, 5.0),
            rayleigh_scale_height_km: 27.0,
            mie_density: rng.range(1.0, 3.0),
            mie_scale_height_km: 20.0,
            mie_g: 0.7,
            mie_absorption: 0.2,
            ozone: 0.0,
            methane: rng.range(0.0, 0.2),
            clouds: None,
            ..earth
        }),
        IceGiant => Some(Atmosphere {
            top_km: 600.0,
            rayleigh_density: rng.range(3.0, 6.0),
            rayleigh_scale_height_km: 45.0,
            mie_density: rng.range(0.5, 1.5),
            mie_scale_height_km: 30.0,
            mie_g: 0.7,
            mie_absorption: 0.1,
            ozone: 0.0,
            methane: rng.range(0.7, 1.3),
            clouds: None,
            ..earth
        }),
        _ => None,
    }
}

/// Systems whose star is within `range` (units of M) of `pos`, nearest
/// first.
pub fn nearby(seed: u64, bodies: &[Body], pos: V3, range: f64) -> Vec<(f64, System)> {
    let mut out: Vec<(f64, System)> = bodies
        .iter()
        .enumerate()
        .filter(|(_, b)| b.alive && b.params.kind == BodyKind::Star)
        .filter_map(|(i, b)| {
            let d = vec3::norm(vec3::sub(b.position(), pos));
            (d < range).then(|| system(seed, i, b).map(|s| (d, s))).flatten()
        })
        .collect();
    out.sort_by(|a, b| a.0.total_cmp(&b.0));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::{World, WorldConfig};

    #[test]
    fn kepler_orbits_conserve_energy_and_close() {
        let orbit = Orbit { a_km: AU_KM, e: 0.3, p: [1.0, 0.0, 0.0], q: [0.0, 0.6, 0.8], mean_anomaly0: 0.4 };
        let gm = G_KM * SUN_MASS_KG;
        let energy = |t: f64| {
            let (x, v) = orbit.state(gm, t);
            0.5 * vec3::dot(v, v) - gm / vec3::norm(x)
        };
        let e0 = energy(0.0);
        assert!((e0 + gm / (2.0 * AU_KM)).abs() < 1e-9 * e0.abs());
        for t in [1e5, 3.3e6, 2e7] {
            assert!((energy(t) / e0 - 1.0).abs() < 1e-9);
        }
        let period = orbit.period_s(gm);
        assert!((period / 3.156e7 - 1.0).abs() < 2e-3, "a year: {period}");
        let (a, _) = orbit.state(gm, 1234.0);
        let (b, _) = orbit.state(gm, 1234.0 + period);
        assert!(vec3::norm(vec3::sub(a, b)) < 1e-3);
    }

    #[test]
    fn systems_are_deterministic_and_inside_the_hill_sphere() {
        let w = World::new(WorldConfig::sgr_a(400, 1));
        let bodies = &w.cluster.bodies;
        let mut count = 0;
        let mut kinds = std::collections::HashSet::new();
        for (i, b) in bodies.iter().enumerate() {
            let Some(s) = system(7, i, b) else { continue };
            assert_eq!(Some(&s), system(7, i, b).as_ref());
            count += 1;
            let r_km = vec3::norm(b.position()) * KM_PER_M;
            let hill = r_km * (b.params.mass / 3.0).cbrt();
            for p in &s.planets {
                kinds.insert(p.kind);
                assert!(p.orbit.a_km * (1.0 + p.orbit.e) < 0.45 * hill);
                assert!(p.orbit.a_km > s.star_radius_km * 3.0);
                assert!(p.radius_km > 1000.0 && p.radius_km < 1.0e5);
            }
        }
        assert!(count > 100, "only {count} systems");
        for k in [PlanetKind::Ocean, PlanetKind::GasGiant, PlanetKind::Lava, PlanetKind::Ice] {
            assert!(kinds.contains(&k), "no {k:?} planets");
        }
    }

    #[test]
    fn nearby_is_sorted() {
        let w = World::new(WorldConfig::sgr_a(200, 3));
        let near = nearby(1, &w.cluster.bodies, w.pilot.position(), 5000.0 * units::AU);
        assert!(!near.is_empty());
        assert!(near.windows(2).all(|p| p[0].0 <= p[1].0));
    }
}
