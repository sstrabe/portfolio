//! The map (M): the cluster, the star systems and the ship's orbit, in the
//! hole's coordinates, seen by a camera that orbits a focus. The scale
//! runs from a planet's low orbits out to the whole cluster.
//!
//! The ship's orbit is the osculating conic around the body whose gravity
//! dominates (as in the HUD), drawn around that body where it is now.
//! Stars and planets are placed where they are at the ship's coordinate
//! time, without light-travel delay.

use crate::flight::{self, DIM, Nav, PANEL, PROGRADE, TARGET, TEXT, WARN};
use crate::input::DRAG_RAD_PER_PX;
use kerr::cluster::BodyKind;
use kerr::local::PlanetRef;
use kerr::planets::{AU_KM, KM_PER_M, PlanetKind, System};
use kerr::units::AU;
use kerr::vec3::{self, V3};
use kerr::world::{Reference, Target, World};
use render_hq::overlay::{Overlay, Rgba};
use std::collections::HashMap;
use std::f64::consts::{PI, TAU};

const ORBIT: Rgba = [0.35, 0.65, 1.0, 0.95];
const SHIP: Rgba = [1.0, 0.62, 0.15, 1.0];
const RINGS: Rgba = [0.45, 0.55, 0.7, 0.28];
const BACKGROUND: [f64; 3] = [0.002, 0.003, 0.007];
const MIN_DISTANCE: f64 = 1e-5;
const MAX_DISTANCE: f64 = 5e7;
const FOV: f64 = 0.8;

/// What can be clicked or focused on the map.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Pick {
    Ship,
    Hole,
    Star(usize),
    Planet(PlanetRef),
}

pub struct Map {
    pub on: bool,
    focus: Pick,
    yaw: f64,
    pitch: f64,
    /// Camera distance from the focus, M.
    distance: f64,
    systems: HashMap<(usize, u32), Option<System>>,
    /// Where each pickable thing was drawn last frame.
    picks: Vec<([f32; 2], Pick)>,
}

impl Default for Map {
    fn default() -> Self {
        Self {
            on: false,
            focus: Pick::Hole,
            yaw: 0.6,
            pitch: 0.5,
            distance: 1e4,
            systems: HashMap::new(),
            picks: Vec::new(),
        }
    }
}

/// A perspective camera looking at a point.
struct Camera {
    eye: V3,
    right: V3,
    up: V3,
    fwd: V3,
    /// Focal length and centre, pixels.
    f: f64,
    c: [f64; 2],
    near: f64,
    size: [f64; 2],
}

impl Camera {
    fn new(focus: V3, yaw: f64, pitch: f64, distance: f64, size: (u32, u32)) -> Self {
        let (sy, cy) = yaw.sin_cos();
        let (sp, cp) = pitch.sin_cos();
        let back = [cp * cy, cp * sy, sp];
        let fwd = vec3::scale(back, -1.0);
        let right = vec3::normalize(vec3::cross(fwd, [0.0, 0.0, 1.0]));
        let (w, h) = (size.0 as f64, size.1.max(1) as f64);
        Self {
            eye: vec3::axpy(focus, distance, back),
            right,
            up: vec3::cross(right, fwd),
            fwd,
            f: 0.5 * h / (0.5 * FOV).tan(),
            c: [0.5 * w, 0.5 * h],
            near: distance * 1e-3,
            size: [w, h],
        }
    }

    fn view(&self, p: V3) -> V3 {
        let q = vec3::sub(p, self.eye);
        [vec3::dot(q, self.right), vec3::dot(q, self.up), vec3::dot(q, self.fwd)]
    }

    fn screen(&self, v: V3) -> [f32; 2] {
        [(self.c[0] + v[0] / v[2] * self.f) as f32, (self.c[1] - v[1] / v[2] * self.f) as f32]
    }

    fn project(&self, p: V3) -> Option<[f32; 2]> {
        let v = self.view(p);
        (v[2] > self.near).then(|| self.screen(v))
    }

    fn visible(&self, p: [f32; 2], margin: f32) -> bool {
        p[0] > -margin && p[1] > -margin && p[0] < self.size[0] as f32 + margin && p[1] < self.size[1] as f32 + margin
    }

    /// Pixels spanned by a length `l` at `p`.
    fn pixels(&self, l: f64, p: V3) -> f64 {
        let z = self.view(p)[2];
        if z > self.near { l / z * self.f } else { f64::INFINITY }
    }

    /// Whether a sphere (centre, radius) hides `p`.
    fn hidden(&self, p: V3, spheres: &[(V3, f64)]) -> bool {
        let to = vec3::sub(p, self.eye);
        let l = vec3::norm(to);
        let u = vec3::scale(to, 1.0 / l.max(1e-300));
        spheres.iter().any(|&(c, r)| {
            let oc = vec3::sub(c, self.eye);
            let tca = vec3::dot(oc, u);
            let d2 = vec3::dot(oc, oc) - tca * tca;
            d2 < r * r && {
                let t0 = tca - (r * r - d2).sqrt();
                t0 > 0.0 && t0 < l * (1.0 - 1e-6)
            }
        })
    }

    /// A 3D polyline, clipped at the near plane, leaving out what the
    /// spheres hide.
    fn polyline(&self, o: &mut Overlay, points: &[V3], width: f32, color: Rgba, spheres: &[(V3, f64)]) {
        let views: Vec<V3> = points.iter().map(|&p| self.view(p)).collect();
        for (i, pair) in views.windows(2).enumerate() {
            let mid = vec3::scale(vec3::add(points[i], points[i + 1]), 0.5);
            if !spheres.is_empty() && self.hidden(mid, spheres) {
                continue;
            }
            let (mut a, mut b) = (pair[0], pair[1]);
            if a[2] <= self.near && b[2] <= self.near {
                continue;
            }
            if a[2] <= self.near || b[2] <= self.near {
                let t = (self.near - a[2]) / (b[2] - a[2]);
                let m = vec3::add(a, vec3::scale(vec3::sub(b, a), t));
                if a[2] <= self.near {
                    a = m;
                } else {
                    b = m;
                }
            }
            let (sa, sb) = (self.screen(a), self.screen(b));
            let (w, h) = (self.size[0] as f32, self.size[1] as f32);
            let off = (sa[0] < 0.0 && sb[0] < 0.0)
                || (sa[1] < 0.0 && sb[1] < 0.0)
                || (sa[0] > w && sb[0] > w)
                || (sa[1] > h && sb[1] > h);
            if !off {
                o.line(sa, sb, width, color);
            }
        }
    }
}

/// Colour of a blackbody at `t` kelvin (an sRGB fit to the Planckian
/// locus).
fn blackbody(t: f64) -> Rgba {
    let t = (t / 100.0).clamp(10.0, 400.0);
    let r = if t <= 66.0 { 255.0 } else { 329.7 * (t - 60.0).powf(-0.1332) };
    let g = if t <= 66.0 { 99.47 * t.ln() - 161.12 } else { 288.12 * (t - 60.0).powf(-0.07551) };
    let b = if t >= 66.0 {
        255.0
    } else if t <= 19.0 {
        0.0
    } else {
        138.52 * (t - 10.0).ln() - 305.04
    };
    let c = |x: f64| (x.clamp(0.0, 255.0) / 255.0) as f32;
    [c(r), c(g), c(b), 1.0]
}

fn planet_color(kind: PlanetKind) -> Rgba {
    match kind {
        PlanetKind::Rocky => [0.66, 0.6, 0.55, 1.0],
        PlanetKind::Ocean => [0.3, 0.58, 1.0, 1.0],
        PlanetKind::Desert => [0.9, 0.7, 0.42, 1.0],
        PlanetKind::Ice => [0.82, 0.92, 1.0, 1.0],
        PlanetKind::Lava => [1.0, 0.42, 0.16, 1.0],
        PlanetKind::GasGiant => [0.92, 0.78, 0.56, 1.0],
        PlanetKind::IceGiant => [0.52, 0.84, 0.92, 1.0],
    }
}

fn alpha(c: Rgba, a: f32) -> Rgba {
    [c[0], c[1], c[2], c[3] * a]
}

/// Points of a conic with focus at the origin: semi-latus rectum `p`,
/// eccentricity vector direction `pe` (unit), `q` 90° ahead in the orbit,
/// out to distance `r_max` when open.
fn conic(p: f64, e: f64, pe: V3, q: V3, r_max: f64) -> Vec<V3> {
    let limit = if e < 1.0 {
        PI
    } else {
        // Where r = r_max: 1 + e cos ν = p / r_max.
        ((p / r_max - 1.0) / e).clamp(-1.0, 1.0).acos()
    };
    let n = 256;
    (0..=n)
        .map(|i| {
            let nu = -limit + 2.0 * limit * i as f64 / n as f64;
            let r = p / (1.0 + e * nu.cos());
            vec3::add(vec3::scale(pe, r * nu.cos()), vec3::scale(q, r * nu.sin()))
        })
        .collect()
}

/// A round length near `x` (1, 2 or 5 times a power of ten).
fn round_length(x: f64) -> f64 {
    let k = 10f64.powf(x.log10().floor());
    [1.0, 2.0, 5.0, 10.0].into_iter().map(|m| m * k).take_while(|&v| v <= x).last().unwrap_or(k)
}

impl Map {
    /// Open the map centred on the body the ship flies around, far enough
    /// out to show its orbit.
    pub fn open(&mut self, nav: &Nav) {
        self.on = true;
        self.reset(nav);
    }

    pub fn reset(&mut self, nav: &Nav) {
        let r = vec3::norm(nav.rel.offset);
        self.focus = match nav.rel.reference {
            Reference::Hole => Pick::Hole,
            Reference::Star(i) => Pick::Star(i),
            Reference::Planet(p) => Pick::Planet(p),
        };
        let e = &nav.elements;
        let extent = if e.bound() { e.apoapsis.max(r) } else { 2.0 * r };
        self.distance = (2.5 * extent).max(6.0 * nav.rel.radius).clamp(MIN_DISTANCE, MAX_DISTANCE);
        self.pitch = 0.5;
    }

    pub fn orbit(&mut self, drag: [f64; 2]) {
        self.yaw -= drag[0] * DRAG_RAD_PER_PX;
        self.pitch = (self.pitch + drag[1] * DRAG_RAD_PER_PX).clamp(-1.5, 1.5);
    }

    pub fn zoom(&mut self, notches: f64) {
        self.distance = (self.distance * 1.25f64.powf(-notches)).clamp(MIN_DISTANCE, MAX_DISTANCE);
    }

    /// The thing drawn nearest `cursor` (within a few pixels), last frame.
    pub fn pick(&self, cursor: [f32; 2], ui: f32) -> Option<Pick> {
        let d2 = |p: [f32; 2]| (p[0] - cursor[0]).powi(2) + (p[1] - cursor[1]).powi(2);
        let reach = (12.0 * ui).powi(2);
        self.picks.iter().filter(|(p, _)| d2(*p) < reach).min_by(|a, b| d2(a.0).total_cmp(&d2(b.0))).map(|&(_, k)| k)
    }

    /// Centre on `pick`, from a distance that suits it.
    pub fn focus_on(&mut self, world: &World, pick: Pick) {
        self.focus = pick;
        let d = match pick {
            Pick::Ship => self.distance,
            Pick::Hole => 2.5 * vec3::norm(world.pilot.position()),
            Pick::Star(i) => {
                match self.system(world, i).map(|s| s.planets.iter().map(|p| p.orbit.a_km).fold(0.0, f64::max)) {
                    Some(a) if a > 0.0 => 3.0 * a / KM_PER_M,
                    _ => 20.0 * AU,
                }
            }
            Pick::Planet(p) => 30.0 * self.planet_radius(world, p).unwrap_or(1e4 / KM_PER_M),
        };
        self.distance = d.clamp(MIN_DISTANCE, MAX_DISTANCE);
    }

    /// Focus on the next of: the ship, the body it flies around, the
    /// target and its star, the hole.
    pub fn next_focus(&mut self, world: &World, nav: &Nav) -> Pick {
        let mut list = vec![Pick::Ship];
        match nav.rel.reference {
            Reference::Planet(p) => list.extend([Pick::Planet(p), Pick::Star(p.star)]),
            Reference::Star(i) => list.push(Pick::Star(i)),
            Reference::Hole => {}
        }
        match nav.target.as_ref().map(|t| t.target) {
            Some(Target::Planet(p)) => list.extend([Pick::Planet(p), Pick::Star(p.star)]),
            Some(Target::Hole) | None => {}
        }
        list.push(Pick::Hole);
        list.dedup();
        let i = list.iter().position(|&p| p == self.focus).map_or(0, |i| (i + 1) % list.len());
        self.focus_on(world, list[i]);
        list[i]
    }

    pub fn name(world: &World, pick: Pick) -> String {
        match pick {
            Pick::Ship => "your ship".into(),
            Pick::Hole => "Sgr A*".into(),
            Pick::Star(i) => flight::star_name(world, i),
            Pick::Planet(p) => crate::hud::planet_name(world, p),
        }
    }

    pub fn background() -> [f64; 3] {
        BACKGROUND
    }

    fn system(&mut self, world: &World, i: usize) -> Option<&System> {
        let body = world.cluster.bodies.get(i)?;
        let key = (i, body.generation);
        self.systems
            .entry(key)
            .or_insert_with(|| {
                (body.alive && body.params.kind == BodyKind::Star)
                    .then(|| kerr::planets::system(world.cluster.cfg.seed, i, body))
                    .flatten()
            })
            .as_ref()
    }

    /// Where `pick` is now.
    fn position(&mut self, world: &World, pick: Pick) -> Option<V3> {
        match pick {
            Pick::Ship => Some(world.pilot.position()),
            Pick::Hole => Some([0.0; 3]),
            Pick::Star(i) => world.cluster.bodies.get(i).filter(|b| b.alive).map(|b| b.position()),
            Pick::Planet(p) => flight::planet_state(world, p).map(|s| s.0),
        }
    }

    fn planet_radius(&mut self, world: &World, p: PlanetRef) -> Option<f64> {
        self.system(world, p.star).and_then(|s| s.planets.get(p.planet)).map(|pl| pl.radius_km / KM_PER_M)
    }

    pub fn draw(
        &mut self,
        o: &mut Overlay,
        world: &World,
        nav: &Nav,
        size: (u32, u32),
        ui: f32,
        cursor: Option<[f32; 2]>,
    ) {
        let s = flight::text_scale(ui);
        let focus = self.position(world, self.focus).unwrap_or_else(|| {
            self.focus = Pick::Ship;
            world.pilot.position()
        });
        let cam = Camera::new(focus, self.yaw, self.pitch, self.distance, size);
        let t = world.pilot.x[0];
        let distance = self.distance;
        let mut picks: Vec<([f32; 2], Pick)> = Vec::new();
        // Labels: anchor, radius of what they name (pixels), text, colour.
        let mut labels: Vec<([f32; 2], f32, String, Rgba)> = Vec::new();
        let hovered = cursor.and_then(|c| self.pick(c, ui));
        let target = world.planet_target();
        let hole_targeted = world.target == Some(Target::Hole);
        let local_star = nav.star.as_ref().and(world.local.as_ref()).map(|l| l.star);
        let ship = world.pilot.position();
        let body = vec3::sub(ship, nav.rel.offset);
        let mut spheres = Vec::new();
        if nav.rel.reference != Reference::Hole {
            spheres.push((body, nav.rel.radius));
        }
        if let Some(p) = target
            && let (Some((x, _)), Some(r)) = (flight::planet_state(world, p), self.planet_radius(world, p))
        {
            spheres.push((x, r));
        }
        spheres.retain(|&(c, r)| cam.pixels(r, c) > 4.0);

        // Distance rings around the hole, in its equatorial plane.
        for k in -1..=5 {
            let r = 10f64.powi(k) * AU;
            let px = cam.pixels(r, [0.0; 3]);
            if !(25.0..=20.0 * cam.size[0]).contains(&px) || self.distance < 0.1 * r {
                continue;
            }
            let ring: Vec<V3> =
                (0..=128).map(|i| (TAU * i as f64 / 128.0).sin_cos()).map(|(s, c)| [r * c, r * s, 0.0]).collect();
            cam.polyline(o, &ring, ui, RINGS, &[]);
            let side = self.yaw - 0.6;
            if let Some(p) =
                cam.project([r * side.cos(), r * side.sin(), 0.0]).filter(|&p| px > 60.0 && cam.visible(p, 0.0))
            {
                let text = if k < 0 { "0.1 AU".to_string() } else { format!("{} AU", 10u64.pow(k as u32)) };
                labels.push((p, 0.0, text, alpha(RINGS, 2.5)));
            }
        }

        // The hole.
        if let Some(p) = cam.project([0.0; 3]) {
            let r = (cam.pixels(world.kerr.r_plus(), [0.0; 3]) as f32).max(4.0 * ui);
            o.disc(p, r + 2.0 * ui, [1.0, 0.55, 0.2, 0.9]);
            o.disc(p, r, [0.0, 0.0, 0.0, 1.0]);
            if hole_targeted {
                o.ring(p, r + 7.0 * ui, 1.5 * ui, TARGET);
            }
            picks.push((p, Pick::Hole));
            labels.push((p, r, "Sgr A*".into(), if hole_targeted { TARGET } else { WARN }));
        }

        // Stars, compact objects and planetary systems.
        for i in 0..world.cluster.bodies.len() {
            let b = &world.cluster.bodies[i];
            if !b.alive || b.params.kind == BodyKind::Station {
                continue;
            }
            let star_pos = match &world.local {
                Some(l) if l.star == i => l.star_state(t).0,
                _ => b.position(),
            };
            let Some(p) = cam.project(star_pos) else { continue };
            let (params, generation) = (b.params, b.generation);
            let mut show_label = Some(i) == local_star || target.is_some_and(|tg| tg.star == i);
            let has_planets = self.system(world, i).is_some_and(|s| !s.planets.is_empty());
            if let Some(sys) = self.system(world, i) {
                let extent = sys.planets.iter().map(|pl| pl.orbit.a_km).fold(0.0, f64::max) / KM_PER_M;
                let px = cam.pixels(extent, star_pos);
                if px > 6.0 {
                    show_label |= px > 60.0;
                    for (j, pl) in sys.planets.iter().enumerate() {
                        let pref = PlanetRef { star: i, generation, planet: j };
                        let targeted = target == Some(pref);
                        let o_px = cam.pixels(pl.orbit.a_km / KM_PER_M, star_pos);
                        if o_px < 3.0 {
                            continue;
                        }
                        let orb = pl.orbit;
                        let b_axis = (1.0 - orb.e * orb.e).sqrt();
                        let ring: Vec<V3> = (0..=128)
                            .map(|k| {
                                let ea = TAU * k as f64 / 128.0;
                                let x = orb.a_km * (ea.cos() - orb.e) / KM_PER_M;
                                let y = orb.a_km * b_axis * ea.sin() / KM_PER_M;
                                vec3::add(star_pos, vec3::add(vec3::scale(orb.p, x), vec3::scale(orb.q, y)))
                            })
                            .collect();
                        // A planet's own orbit fades out when looking at the planet
                        // from close by.
                        let fade = (distance / (0.05 * pl.orbit.a_km / KM_PER_M)).clamp(0.2, 1.0) as f32;
                        let color = if targeted { TARGET } else { alpha(planet_color(pl.kind), 0.4 * fade) };
                        cam.polyline(o, &ring, if targeted { 1.8 * ui } else { ui }, color, &spheres);
                        let pos = match world.system_of(pref) {
                            Some(l) => l.planet_state(j, t).0,
                            None => vec3::axpy(star_pos, 1.0 / KM_PER_M, sys.planet_state(j, t).0),
                        };
                        if let Some(pp) = cam.project(pos).filter(|&pp| cam.visible(pp, 20.0)) {
                            let r = (cam.pixels(pl.radius_km / KM_PER_M, pos) as f32).max(3.0 * ui);
                            o.disc(pp, r, planet_color(pl.kind));
                            if targeted {
                                o.ring(pp, r + 5.0 * ui, 1.5 * ui, TARGET);
                            }
                            picks.push((pp, Pick::Planet(pref)));
                            let is_ref = nav.rel.reference == Reference::Planet(pref);
                            // Other planets are named once their whole orbit is in view.
                            let whole = o_px > 40.0 && distance > 0.3 * pl.orbit.a_km / KM_PER_M;
                            if whole || targeted || is_ref || hovered == Some(Pick::Planet(pref)) {
                                let name = crate::hud::planet_name(world, pref);
                                labels.push((pp, r, name, if targeted { TARGET } else { alpha(TEXT, 0.85) }));
                            }
                        }
                    }
                }
            }
            if !cam.visible(p, 10.0) {
                continue;
            }
            match params.kind {
                BodyKind::Compact => {
                    o.ring(p, 3.5 * ui, 1.5 * ui, [0.7, 0.45, 1.0, 0.9]);
                }
                _ => {
                    let lum = (2.0 + 0.6 * params.luminosity.max(1e-2).log10()).clamp(1.3, 4.5) as f32 * ui;
                    let r = (cam.pixels(params.radius, star_pos) as f32).max(lum);
                    o.disc(p, r, blackbody(params.temperature));
                    if has_planets {
                        o.ring(p, r + 3.5 * ui, ui, [0.6, 0.8, 1.0, 0.45]);
                    }
                }
            }
            picks.push((p, Pick::Star(i)));
            if show_label || hovered == Some(Pick::Star(i)) {
                labels.push((p, 4.0 * ui, flight::star_name(world, i), alpha(TEXT, 0.7)));
            }
        }

        // The ship's orbit around its reference body, drawn around where
        // that body is now.
        let (r, v, mu) = (nav.rel.offset, nav.rel.velocity, nav.rel.mu);
        let h = vec3::cross(r, v);
        if vec3::norm(h) > 1e-300 && vec3::norm(r) > 0.0 {
            let d = vec3::norm(r);
            let ev = vec3::scale(
                vec3::sub(vec3::scale(r, vec3::dot(v, v) - mu / d), vec3::scale(v, vec3::dot(r, v))),
                1.0 / mu,
            );
            let e = vec3::norm(ev);
            let p = vec3::dot(h, h) / mu;
            let pe = if e > 1e-9 { vec3::scale(ev, 1.0 / e) } else { vec3::normalize(r) };
            let q = vec3::normalize(vec3::cross(h, pe));
            let pts: Vec<V3> = conic(p, e, pe, q, 3.0 * d.max(p))
                .into_iter()
                .filter(|x| vec3::norm(*x) >= nav.rel.radius)
                .map(|x| vec3::add(body, x))
                .collect();
            cam.polyline(o, &pts, 2.0 * ui, ORBIT, &spheres);
            let alt = |x: f64| match nav.rel.reference {
                Reference::Planet(_) => flight::distance(x - nav.rel.radius),
                _ => flight::distance(x),
            };
            // (A circular orbit has neither.)
            let rp = p / (1.0 + e);
            let at = |x: V3| Some(x).filter(|&x| !cam.hidden(x, &spheres)).and_then(|x| cam.project(x));
            if e > 0.005
                && rp > nav.rel.radius
                && let Some(sp) = at(vec3::axpy(body, rp, pe))
            {
                o.disc(sp, 3.5 * ui, ORBIT);
                labels.push((sp, 3.0 * ui, format!("Pe {}", alt(rp)), ORBIT));
            }
            if e > 0.005 && e < 1.0 {
                let ra = p / (1.0 - e);
                if let Some(sp) = at(vec3::axpy(body, -ra, pe)) {
                    o.disc(sp, 3.5 * ui, ORBIT);
                    labels.push((sp, 3.0 * ui, format!("Ap {}", alt(ra)), ORBIT));
                }
            }
        }

        // The ship, with a line along its nose.
        if let Some(sp) = cam.project(ship) {
            let e1 = world.pilot.e[1];
            let nose = vec3::normalize([e1[1], e1[2], e1[3]]);
            if let Some(tip) = cam.project(vec3::axpy(ship, 0.08 * self.distance, nose)) {
                o.line(sp, tip, 2.0 * ui, SHIP);
            }
            o.disc(sp, 5.0 * ui, [0.0, 0.0, 0.0, 0.8]);
            o.ring(sp, 5.0 * ui, 2.0 * ui, SHIP);
            picks.push((sp, Pick::Ship));
            labels.push((sp, 5.0 * ui, "ship".into(), SHIP));
        }
        if let Some(pick) = hovered
            && let Some(&(p, _)) = picks.iter().find(|(_, k)| *k == pick)
        {
            o.ring(p, 9.0 * ui, 1.5 * ui, alpha(TEXT, 0.9));
        }
        for (p, r, text, color) in &labels {
            o.text([p[0] + r * 0.7 + 4.0 * ui, p[1] - r * 0.7 - 8.0 * s], text, s, *color);
        }
        self.picks = picks;

        // Scale bar, focus and hints.
        let (w, hgt) = (cam.size[0] as f32, cam.size[1] as f32);
        let per_px = self.distance / cam.f;
        let want = 160.0 * ui as f64 * per_px * KM_PER_M;
        let (len, label) = if want >= 0.1 * AU_KM {
            let au = round_length(want / AU_KM);
            (au * AU_KM, format!("{au} AU"))
        } else {
            let km = round_length(want);
            (km, format!("{km} km"))
        };
        let bar = (len / KM_PER_M / per_px) as f32;
        let (x1, y) = (w - 16.0 * ui, hgt - 24.0 * ui);
        o.rect([x1 - bar, y], [x1, y + 2.0 * ui], TEXT);
        o.rect([x1 - bar, y - 6.0 * ui], [x1 - bar + 2.0 * ui, y + 2.0 * ui], TEXT);
        o.rect([x1 - 2.0 * ui, y - 6.0 * ui], [x1, y + 2.0 * ui], TEXT);
        o.text([x1 - bar, y - 14.0 * s], &label, s, TEXT);

        let lines = [
            (format!("MAP: {}", Self::name(world, self.focus)), PROGRADE),
            ("right drag turns, wheel zooms".into(), DIM),
            ("F focus, Home reset, M back".into(), DIM),
            ("click a planet or Sgr A*: target".into(), DIM),
            ("ringed stars have planets".into(), DIM),
        ];
        let line = 10.0 * s;
        let width = lines.iter().map(|(l, _)| Overlay::text_width(l, s)).fold(0.0, f32::max);
        let (x1, y0) = (w - 8.0 * ui, 22.0 * ui);
        o.rect([x1 - width - 12.0 * ui, y0 - 6.0 * ui], [x1, y0 + lines.len() as f32 * line + 4.0 * ui], PANEL);
        for (i, (text, color)) in lines.iter().enumerate() {
            o.text([x1 - width - 6.0 * ui, y0 + i as f32 * line], text, s, *color);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_lengths() {
        assert_eq!(round_length(3.7), 2.0);
        assert_eq!(round_length(7.0), 5.0);
        assert_eq!(round_length(1234.0), 1000.0);
    }

    /// A circular orbit's conic is a circle through the ship.
    #[test]
    fn conic_is_the_orbit() {
        let pts = conic(2.0, 0.0, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 10.0);
        assert!(pts.iter().all(|p| (vec3::norm(*p) - 2.0).abs() < 1e-12));
        let open = conic(1.0, 1.5, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 10.0);
        assert!(open.iter().all(|p| vec3::norm(*p) <= 10.0 + 1e-9));
        assert!((vec3::norm(open[128]) - 0.4).abs() < 1e-12);
    }

    #[test]
    fn camera_projects_the_focus_to_the_centre() {
        let cam = Camera::new([5.0, -3.0, 2.0], 0.3, 0.4, 10.0, (800, 600));
        let p = cam.project([5.0, -3.0, 2.0]).unwrap();
        assert!((p[0] - 400.0).abs() < 1e-3 && (p[1] - 300.0).abs() < 1e-3);
        // +z is up on screen.
        let up = cam.project([5.0, -3.0, 3.0]).unwrap();
        assert!(up[1] < 300.0);
    }
}
