//! The flight HUD: where the ship is and where it is going. A navball
//! (the ship's nose in the middle, the horizon of the body it flies
//! around, and the prograde, normal, radial and target markers), the same
//! markers over the view, and the readouts.

use crate::hud;
use crate::input::{Controls, ENGINE, SasMode};
use kerr::cluster::BodyKind;
use kerr::local::{Elements, PlanetRef};
use kerr::planets::{self, C_KM_S, KM_PER_M};
use kerr::units::SECONDS_PER_M;
use kerr::vec3::{self, V3};
use kerr::world::{PilotStatus, Reference, Relative, World};
use render_hq::overlay::{Overlay, Rgba};
use render_hq::session::View;
use std::f64::consts::{PI, TAU};

pub const PROGRADE: Rgba = [0.86, 0.95, 0.25, 1.0];
pub const NORMAL: Rgba = [0.85, 0.4, 0.98, 1.0];
pub const RADIAL: Rgba = [0.35, 0.85, 1.0, 1.0];
pub const TARGET: Rgba = [1.0, 0.38, 0.72, 1.0];
pub const TEXT: Rgba = [0.92, 0.94, 0.96, 1.0];
pub const DIM: Rgba = [0.6, 0.64, 0.7, 1.0];
pub const WARN: Rgba = [1.0, 0.5, 0.3, 1.0];
pub const PANEL: Rgba = [0.02, 0.03, 0.05, 0.6];
const ON: Rgba = [0.4, 0.95, 0.5, 1.0];
const NOSE: Rgba = [1.0, 0.62, 0.15, 1.0];
const SKY: Rgba = [0.2, 0.42, 0.72, 0.92];
const GROUND: Rgba = [0.5, 0.33, 0.2, 0.92];

/// The target (a planet or Sgr A*) as seen from the ship.
pub struct Target {
    pub target: kerr::world::Target,
    pub name: String,
    /// Apparent direction, ship frame.
    pub dir: V3,
    /// Distance to its centre, M.
    pub distance: f64,
    /// Rate at which that distance shrinks, c.
    pub closing: f64,
}

/// The ship's situation for the HUD and for SAS, from the world.
pub struct Nav {
    pub rel: Relative,
    pub name: String,
    pub elements: Elements,
    /// Ship-frame unit vectors: away from the body's centre, along the
    /// velocity relative to it, along the orbit's normal, and radially out
    /// in the orbit's plane (perpendicular to the velocity).
    pub up: Option<V3>,
    pub prograde: Option<V3>,
    pub normal: Option<V3>,
    pub radial: Option<V3>,
    pub target: Option<Target>,
    /// Apparent directions of the hole and the local star.
    pub hole: V3,
    pub star: Option<(V3, String)>,
}

impl Nav {
    pub fn new(world: &World) -> Self {
        let (k, pilot) = (&world.kerr, &world.pilot);
        let pos = pilot.position();
        let rel = world.reference();
        let unit = |v: V3| (vec3::norm(v) > 1e-300).then(|| vec3::normalize(v));
        let r = pilot.local_components(k, rel.offset);
        let v = rel.ship_velocity;
        let prograde = (vec3::norm(v) > 1e-12).then(|| vec3::normalize(v));
        let normal = prograde.and_then(|_| unit(vec3::cross(r, v)));
        let radial = match (prograde, normal) {
            (Some(p), Some(n)) => Some(vec3::normalize(vec3::cross(p, n))),
            _ => None,
        };
        let star = world.local.as_ref().filter(|_| rel.reference != Reference::Hole).map(|l| {
            let (x, _) = l.star_state(pilot.x[0]);
            (pilot.sky_direction(k, vec3::sub(x, pos)), star_name(world, l.star))
        });
        Self {
            elements: Elements::of(rel.mu, rel.offset, rel.velocity),
            name: reference_name(world, rel.reference),
            up: unit(r),
            prograde,
            normal,
            radial,
            target: target(world),
            hole: pilot.sky_direction(k, vec3::scale(pos, -1.0)),
            star,
            rel,
        }
    }

    /// Where the SAS mode wants the nose, if that exists now.
    pub fn hold(&self, mode: SasMode) -> Option<V3> {
        let neg = |v: Option<V3>| v.map(|v| vec3::scale(v, -1.0));
        let target = self.target.as_ref().map(|t| t.dir);
        match mode {
            SasMode::Stability => None,
            SasMode::Prograde => self.prograde,
            SasMode::Retrograde => neg(self.prograde),
            SasMode::Normal => self.normal,
            SasMode::AntiNormal => neg(self.normal),
            SasMode::RadialOut => self.radial,
            SasMode::RadialIn => neg(self.radial),
            SasMode::Target => target,
            SasMode::AntiTarget => neg(target),
        }
    }

    /// Speed relative to the reference body, c.
    pub fn speed(&self) -> f64 {
        vec3::norm(self.rel.ship_velocity)
    }

    /// What the ship is doing, in capitals.
    pub fn situation(&self, world: &World) -> String {
        match world.status {
            PilotStatus::Landed { planet, .. } => return format!("LANDED on {}", hud::planet_name(world, planet)),
            PilotStatus::Docked { .. } => return "DOCKED".into(),
            PilotStatus::Orbit(p) => return format!("AUTOPILOT to {}", hud::planet_name(world, p)),
            PilotStatus::HoleOrbit => return "AUTOPILOT to Sgr A*".into(),
            _ => {}
        }
        let e = &self.elements;
        let falling = vec3::dot(self.rel.offset, self.rel.velocity) < 0.0;
        let hole = self.rel.reference == Reference::Hole;
        if e.bound() && e.periapsis > self.rel.radius * if hole { 3.0 } else { 1.0 } {
            format!("ORBITING {}", self.name)
        } else if e.bound() || (falling && e.periapsis < self.rel.radius) {
            format!("{} {}", if hole { "PLUNGING INTO" } else { "SUB-ORBITAL over" }, self.name)
        } else if falling {
            format!("FLYBY of {}", self.name)
        } else {
            format!("ESCAPING {}", self.name)
        }
    }
}

/// Position and coordinate velocity of planet `p` now.
pub fn planet_state(world: &World, p: PlanetRef) -> Option<(V3, V3)> {
    let t = world.pilot.x[0];
    if let Some(l) = world.system_of(p) {
        return Some(l.planet_state(p.planet, t));
    }
    let body = world.cluster.bodies.get(p.star).filter(|b| b.alive && b.generation == p.generation)?;
    let sys = planets::system(world.cluster.cfg.seed, p.star, body)?;
    if p.planet >= sys.planets.len() {
        return None;
    }
    let (off, vel) = sys.planet_state(p.planet, t);
    let vs = kerr::geodesic::coordinate_velocity(&world.kerr, &body.state);
    Some((vec3::axpy(body.position(), 1.0 / KM_PER_M, off), vec3::axpy(vs, 1.0 / C_KM_S, vel)))
}

fn target(world: &World) -> Option<Target> {
    let target = world.target?;
    let (k, pilot) = (&world.kerr, &world.pilot);
    // The hole's frame here is that of static observers.
    let (x, v) = match target {
        kerr::world::Target::Planet(p) => planet_state(world, p)?,
        kerr::world::Target::Hole => ([0.0; 3], [0.0; 3]),
    };
    let d = vec3::sub(x, pilot.position());
    let w = k.four_velocity(pilot.position(), v)?;
    let geometric = vec3::normalize(pilot.local_components(k, d));
    Some(Target {
        target,
        name: hud::target_name(world, target),
        dir: pilot.sky_direction(k, d),
        distance: vec3::norm(d),
        closing: -vec3::dot(pilot.relative_velocity(k, w), geometric),
    })
}

pub fn reference_name(world: &World, r: Reference) -> String {
    match r {
        Reference::Hole => "Sgr A*".into(),
        Reference::Star(i) => star_name(world, i),
        Reference::Planet(p) => hud::planet_name(world, p),
    }
}

/// "G star 17", "black hole 3".
pub fn star_name(world: &World, i: usize) -> String {
    let b = &world.cluster.bodies[i];
    match b.params.kind {
        BodyKind::Compact => format!("black hole {i}"),
        BodyKind::Station => format!("station {i}"),
        BodyKind::Star => format!("{} star {i}", spectral_class(b.params.temperature)),
    }
}

pub fn spectral_class(t: f64) -> char {
    match t {
        t if t >= 30_000.0 => 'O',
        t if t >= 10_000.0 => 'B',
        t if t >= 7_500.0 => 'A',
        t if t >= 6_000.0 => 'F',
        t if t >= 5_200.0 => 'G',
        t if t >= 3_700.0 => 'K',
        _ => 'M',
    }
}

/// A speed or a velocity component, given as a fraction of c.
pub fn speed(v: f64) -> String {
    let km_s = v * C_KM_S;
    match km_s {
        s if s < 1.0 => format!("{:.1} m/s", s * 1000.0),
        s if s < 1000.0 => format!("{s:.2} km/s"),
        s if v < 0.2 => format!("{s:.0} km/s"),
        _ => format!("{v:.5} c"),
    }
}

/// The ship's speed: as a speed while that says something, else as its
/// Lorentz factor.
pub fn motion(v: f64, gamma: f64) -> String {
    match gamma {
        g if g < 1.5 => speed(v),
        g if g < 1000.0 => format!("gamma {g:.2}"),
        g => format!("gamma {g:.3e}"),
    }
}

/// A distance in units of M.
pub fn distance(m: f64) -> String {
    hud::km(m * KM_PER_M)
}

/// Markers of directions relative to the reference body and the target.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Marker {
    Prograde,
    Retrograde,
    Normal,
    AntiNormal,
    RadialOut,
    RadialIn,
    Target,
    AntiTarget,
}

impl Marker {
    fn color(self) -> Rgba {
        match self {
            Marker::Prograde | Marker::Retrograde => PROGRADE,
            Marker::Normal | Marker::AntiNormal => NORMAL,
            Marker::RadialOut | Marker::RadialIn => RADIAL,
            Marker::Target | Marker::AntiTarget => TARGET,
        }
    }
}

fn with_alpha(c: Rgba, a: f32) -> Rgba {
    [c[0], c[1], c[2], c[3] * a]
}

/// Draw a marker centred on `p`, `r` pixels in radius.
pub fn marker(o: &mut Overlay, m: Marker, p: [f32; 2], r: f32, alpha: f32) {
    let c = with_alpha(m.color(), alpha);
    let w = (r * 0.22).max(1.2);
    let at = |dx: f32, dy: f32| [p[0] + dx, p[1] + dy];
    let tick = |o: &mut Overlay, ang: f32, from: f32, to: f32| {
        let (s, co) = ang.sin_cos();
        o.line(at(co * from, s * from), at(co * to, s * to), w, c);
    };
    let x = |o: &mut Overlay| {
        let k = r * 0.55;
        o.line(at(-k, -k), at(k, k), w, c);
        o.line(at(-k, k), at(k, -k), w, c);
    };
    use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI as PI32};
    match m {
        Marker::Prograde => {
            o.ring(p, r, w, c);
            for ang in [-FRAC_PI_2, 0.0, PI32] {
                tick(o, ang, r, r * 1.7);
            }
            o.disc(p, w, c);
        }
        Marker::Retrograde => {
            // An X through the ring, reaching past it, so it can't be
            // mistaken for prograde.
            o.ring(p, r, w, c);
            let k = r * 1.3;
            o.line(at(-k, -k), at(k, k), w, c);
            o.line(at(-k, k), at(k, -k), w, c);
        }
        Marker::Normal | Marker::AntiNormal => {
            let s = if m == Marker::Normal { 1.0 } else { -1.0 };
            let v = |ang: f32| at(r * 1.2 * ang.cos(), -s * r * 1.2 * ang.sin());
            let pts = [v(FRAC_PI_2), v(FRAC_PI_2 + 2.0 * PI32 / 3.0), v(FRAC_PI_2 + 4.0 * PI32 / 3.0), v(FRAC_PI_2)];
            o.polyline(&pts, w, c);
            o.disc(p, w, c);
        }
        Marker::RadialOut | Marker::RadialIn => {
            o.ring(p, r, w, c);
            for i in 0..4 {
                let ang = FRAC_PI_4 + i as f32 * FRAC_PI_2;
                if m == Marker::RadialOut {
                    tick(o, ang, r, r * 1.7);
                } else {
                    tick(o, ang, r * 0.35, r);
                }
            }
        }
        Marker::Target | Marker::AntiTarget => {
            o.ring(p, r, w, c);
            for i in 0..4 {
                tick(o, i as f32 * FRAC_PI_2, r, r * 1.6);
            }
            if m == Marker::Target {
                o.disc(p, w, c);
            } else {
                x(o);
            }
        }
    }
}

/// Pixels per font pixel for text at display scale `ui` (whole numbers
/// keep the bitmap font crisp).
pub fn text_scale(ui: f32) -> f32 {
    (1.6 * ui).round().max(1.0)
}

/// Everything besides the world that the HUD shows.
pub struct Extras<'a> {
    pub fps: f64,
    pub note: Option<&'a str>,
    pub help: bool,
    /// The map is up: no markers over the view.
    pub map: bool,
    /// The mouse, for highlighting the button under it.
    pub cursor: Option<[f32; 2]>,
}

/// What a HUD button does.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Button {
    Sas,
    Rcs,
    Mode(SasMode),
}

/// A button's place on screen, for clicks.
#[derive(Clone, Copy, Debug)]
pub struct ButtonRect {
    pub min: [f32; 2],
    pub max: [f32; 2],
    pub button: Button,
}

impl ButtonRect {
    pub fn contains(&self, p: [f32; 2]) -> bool {
        p[0] >= self.min[0] && p[0] <= self.max[0] && p[1] >= self.min[1] && p[1] <= self.max[1]
    }
}

/// The button at `p`, if any.
pub fn button_at(buttons: &[ButtonRect], p: [f32; 2]) -> Option<Button> {
    buttons.iter().find(|b| b.contains(p)).map(|b| b.button)
}

/// Draw the HUD for a `size` image; `ui` is the display's scale factor.
/// Returns where its buttons are.
pub fn draw(
    o: &mut Overlay,
    world: &World,
    nav: &Nav,
    controls: &Controls,
    view: &View,
    ui: f32,
    extras: &Extras,
) -> Vec<ButtonRect> {
    let (w, h) = (view.size.0 as f32, view.size.1 as f32);
    let s = text_scale(ui);
    if !extras.map {
        view_markers(o, nav, view, ui, s);
    }
    let r = (78.0 * ui).round();
    let centre = [(w * 0.5).round(), h - r - (14.0 * ui).round()];
    navball(o, nav, centre, r, ui);
    gauges(o, world, nav, controls, centre, r, s);
    let buttons = sas_buttons(o, nav, controls, centre, r, s, ui, extras.cursor);
    readouts(o, world, nav, s, ui);

    let line = 10.0 * s;
    let fps = format!("{:.0} fps", extras.fps);
    o.text([w - Overlay::text_width(&fps, s * 0.5) - 8.0 * ui, 6.0 * ui], &fps, s * 0.5, DIM);
    if let Some(note) = extras.note {
        let x = (w - Overlay::text_width(note, s)) * 0.5;
        o.rect([x - 8.0 * ui, 8.0 * ui], [w - x + 8.0 * ui, 8.0 * ui + line + 6.0 * ui], PANEL);
        o.text([x, 8.0 * ui + 5.0 * ui], note, s, TEXT);
    }
    if extras.help {
        help(o, view, ui);
    }
    buttons
}

/// SAS and RCS switches and the SAS modes, in a column left of the
/// throttle as in KSP: stability; prograde, retrograde; normal,
/// anti-normal; radial out, radial in; target, anti-target. A mode whose
/// direction doesn't exist now (no target, no motion) is dimmed.
#[allow(clippy::too_many_arguments)]
fn sas_buttons(
    o: &mut Overlay,
    nav: &Nav,
    controls: &Controls,
    c: [f32; 2],
    r: f32,
    s: f32,
    ui: f32,
    cursor: Option<[f32; 2]>,
) -> Vec<ButtonRect> {
    use SasMode::*;
    let size = (13.0 * s).round();
    let gap = (3.0 * ui).round().max(2.0);
    let rows: [&[Button]; 6] = [
        &[Button::Sas, Button::Rcs],
        &[Button::Mode(Stability)],
        &[Button::Mode(Prograde), Button::Mode(Retrograde)],
        &[Button::Mode(Normal), Button::Mode(AntiNormal)],
        &[Button::Mode(RadialOut), Button::Mode(RadialIn)],
        &[Button::Mode(Target), Button::Mode(AntiTarget)],
    ];
    // Right of the column: the throttle gauge and its percentage.
    let right = c[0] - r - 10.0 * s - 5.0 * s - 4.0 * 8.0 * s - 14.0 * s;
    let left = right - 2.0 * size - gap;
    let top = c[1] + r - 6.0 * size - 5.0 * gap;
    let active = [0.16, 0.5, 0.24, 0.9];
    let mut out = Vec::new();
    let mut tip = None;
    for (i, row) in rows.iter().enumerate() {
        let y = top + i as f32 * (size + gap);
        let width = if row.len() == 1 { 2.0 * size + gap } else { size };
        for (j, &button) in row.iter().enumerate() {
            let x = left + j as f32 * (size + gap);
            let rect = ButtonRect { min: [x, y], max: [x + width, y + size], button };
            let hover = cursor.is_some_and(|p| rect.contains(p));
            let on = match button {
                Button::Sas => controls.sas,
                Button::Rcs => controls.rcs,
                Button::Mode(m) => controls.sas && controls.sas_mode == m,
            };
            o.rect(rect.min, rect.max, if on { active } else { PANEL });
            if hover {
                let e = ui.max(1.0);
                let edge = [1.0, 1.0, 1.0, 0.7];
                o.rect(rect.min, [rect.max[0], rect.min[1] + e], edge);
                o.rect([rect.min[0], rect.max[1] - e], rect.max, edge);
                o.rect(rect.min, [rect.min[0] + e, rect.max[1]], edge);
                o.rect([rect.max[0] - e, rect.min[1]], rect.max, edge);
            }
            let mid = [x + 0.5 * width, y + 0.5 * size];
            match button {
                Button::Sas | Button::Rcs => {
                    let label = if button == Button::Sas { "SAS" } else { "RCS" };
                    let ts = (s * 0.5).max(1.0);
                    let tw = Overlay::text_width(label, ts);
                    o.text([mid[0] - 0.5 * tw, mid[1] - 4.0 * ts], label, ts, if on { TEXT } else { DIM });
                }
                Button::Mode(Stability) => {
                    let k = 0.3 * size;
                    let col = if on { TEXT } else { DIM };
                    o.ring(mid, k, 1.5 * ui, col);
                    o.line([mid[0] - 1.8 * k, mid[1]], [mid[0] + 1.8 * k, mid[1]], 1.5 * ui, col);
                    o.line([mid[0], mid[1] - 1.5 * k], [mid[0], mid[1] - k], 1.5 * ui, col);
                }
                Button::Mode(m) => {
                    let alpha = if nav.hold(m).is_some() { 1.0 } else { 0.3 };
                    marker(o, mode_marker(m), mid, 0.26 * size, alpha);
                }
            }
            if hover {
                tip = Some((
                    [left, top - 10.0 * s],
                    match button {
                        Button::Sas => format!("SAS (T): {}", if controls.sas { "on" } else { "off" }),
                        Button::Rcs => format!("RCS (R): {}", if controls.rcs { "on" } else { "off" }),
                        Button::Mode(m) => format!("hold {} ({})", m.name(), mode_key(m)),
                    },
                ));
            }
            out.push(rect);
        }
    }
    if let Some((p, text)) = tip {
        let ts = (s * 0.5).max(1.0);
        let tw = Overlay::text_width(&text, ts);
        o.rect([p[0] - 3.0, p[1] - 3.0], [p[0] + tw + 3.0, p[1] + 8.0 * ts + 3.0], PANEL);
        o.text(p, &text, ts, TEXT);
    }
    out
}

fn mode_marker(m: SasMode) -> Marker {
    match m {
        SasMode::Prograde | SasMode::Stability => Marker::Prograde,
        SasMode::Retrograde => Marker::Retrograde,
        SasMode::Normal => Marker::Normal,
        SasMode::AntiNormal => Marker::AntiNormal,
        SasMode::RadialOut => Marker::RadialOut,
        SasMode::RadialIn => Marker::RadialIn,
        SasMode::Target => Marker::Target,
        SasMode::AntiTarget => Marker::AntiTarget,
    }
}

/// The number key that selects a mode.
fn mode_key(m: SasMode) -> char {
    match m {
        SasMode::Stability => '1',
        SasMode::Prograde => '2',
        SasMode::Retrograde => '3',
        SasMode::Normal => '4',
        SasMode::AntiNormal => '5',
        SasMode::RadialOut => '6',
        SasMode::RadialIn => '7',
        SasMode::Target => '8',
        SasMode::AntiTarget => '9',
    }
}

/// Markers over the view where the directions are on screen; the target
/// gets an arrow at the edge when it is not.
fn view_markers(o: &mut Overlay, nav: &Nav, view: &View, ui: f32, s: f32) {
    let (w, h) = (view.size.0 as f32, view.size.1 as f32);
    let on_screen = |p: [f32; 2]| p[0] >= 0.0 && p[0] <= w && p[1] >= 0.0 && p[1] <= h;
    let neg = |v: V3| vec3::scale(v, -1.0);
    let r = 9.0 * ui;
    let mut marks = Vec::new();
    if let Some(p) = nav.prograde {
        marks.push((Marker::Prograde, p));
        marks.push((Marker::Retrograde, neg(p)));
    }
    for (m, dir) in marks {
        if let Some(p) = view.project(dir).filter(|&p| on_screen(p)) {
            marker(o, m, p, r, 0.9);
        }
    }
    let label = |o: &mut Overlay, p: [f32; 2], text: &str, color: Rgba| {
        o.text([p[0] + 14.0 * ui, p[1] - 4.0 * s], text, s, color);
    };
    if let Some(p) = view.project(nav.hole).filter(|&p| on_screen(p)) {
        o.ring(p, 6.0 * ui, 1.5 * ui, with_alpha(WARN, 0.8));
        label(o, p, "Sgr A*", with_alpha(WARN, 0.9));
    }
    if let Some((dir, name)) = &nav.star
        && let Some(p) = view.project(*dir).filter(|&p| on_screen(p))
    {
        o.ring(p, 7.0 * ui, 1.5 * ui, with_alpha(TEXT, 0.6));
        label(o, p, name, with_alpha(TEXT, 0.8));
    }
    let Some(t) = &nav.target else { return };
    let text = format!("{} {}", t.name, distance(t.distance));
    match view.project(t.dir).filter(|&p| on_screen(p)) {
        Some(p) => {
            marker(o, Marker::Target, p, r, 0.95);
            label(o, p, &text, TARGET);
        }
        None => {
            // An arrow at the edge of the screen, pointing towards it.
            let v = view.to_view(t.dir);
            let mut d = [-v[1] as f32, -v[2] as f32];
            let n = (d[0] * d[0] + d[1] * d[1]).sqrt();
            d = if n > 1e-6 { [d[0] / n, d[1] / n] } else { [0.0, 1.0] };
            let m = 48.0 * ui;
            let k = ((w * 0.5 - m) / d[0].abs().max(1e-6)).min((h * 0.5 - m) / d[1].abs().max(1e-6));
            let p = [w * 0.5 + d[0] * k, h * 0.5 + d[1] * k];
            let a = 14.0 * ui;
            let side = [-d[1] * a * 0.5, d[0] * a * 0.5];
            let tip = [p[0] + d[0] * a, p[1] + d[1] * a];
            o.triangle(tip, [p[0] + side[0], p[1] + side[1]], [p[0] - side[0], p[1] - side[1]], TARGET);
            let tw = Overlay::text_width(&text, s);
            let tx = (p[0] - d[0] * 20.0 * ui - tw * 0.5 * (1.0 + d[0])).clamp(4.0, w - tw - 4.0);
            let ty = (p[1] - d[1] * 20.0 * ui - 4.0 * s).clamp(4.0, h - 12.0 * s);
            o.text([tx, ty], &text, s, TARGET);
        }
    }
}

/// A point on the navball for ship-frame direction `d` (front half).
fn ball(c: [f32; 2], r: f32, d: V3) -> [f32; 2] {
    [c[0] - r * d[1] as f32, c[1] - r * d[2] as f32]
}

/// A curve of ship-frame directions on the navball, clipped to its front.
fn ball_curve(o: &mut Overlay, c: [f32; 2], r: f32, dirs: &[V3], width: f32, color: Rgba) {
    for pair in dirs.windows(2) {
        let (p, d) = (pair[0], pair[1]);
        let (a, b) = match (p[0] > 0.0, d[0] > 0.0) {
            (true, true) => (p, d),
            (false, false) => continue,
            (front, _) => {
                let t = p[0] / (p[0] - d[0]);
                let rim = vec3::normalize(vec3::add(p, vec3::scale(vec3::sub(d, p), t)));
                if front { (p, rim) } else { (rim, d) }
            }
        };
        o.line(ball(c, r, a), ball(c, r, b), width, color);
    }
}

/// Unit vectors perpendicular to `up` (and to each other), the first as
/// close to `hint` as possible.
fn plane_axes(up: V3, hint: Option<V3>) -> (V3, V3) {
    let h = hint.map(|h| vec3::axpy(h, -vec3::dot(h, up), up)).filter(|h| vec3::norm(*h) > 1e-6);
    let e1 = vec3::normalize(h.unwrap_or_else(|| vec3::any_orthogonal(up)));
    (e1, vec3::cross(up, e1))
}

fn navball(o: &mut Overlay, nav: &Nav, c: [f32; 2], r: f32, ui: f32) {
    // Sky and ground: the part of the front half on the nose's side of the
    // horizon is convex (half a disc plus half an ellipse), so it can be
    // filled as a fan around the centre over the other colour.
    match nav.up {
        Some(up) => {
            let a = vec3::cross(up, [1.0, 0.0, 0.0]);
            let (base, inner) = if up[0] > 0.0 { (GROUND, SKY) } else { (SKY, GROUND) };
            if vec3::norm(a) < 1e-6 {
                o.disc(c, r, inner);
            } else {
                let a = vec3::normalize(a);
                let mut b = vec3::cross(up, a);
                if b[0] < 0.0 {
                    b = vec3::scale(b, -1.0);
                }
                o.disc(c, r, base);
                let n = 48;
                let mut outline: Vec<[f32; 2]> = (0..=n)
                    .map(|i| {
                        let t = PI * i as f64 / n as f64;
                        ball(c, r, vec3::add(vec3::scale(a, t.cos()), vec3::scale(b, t.sin())))
                    })
                    .collect();
                let start = (-a[2]).atan2(-a[1]) + PI;
                let pb = [-b[1], -b[2]];
                let mid = start + 0.5 * PI;
                let sign = if mid.cos() * pb[0] + mid.sin() * pb[1] < 0.0 { 1.0 } else { -1.0 };
                for i in 1..n {
                    let ang = start + sign * PI * i as f64 / n as f64;
                    outline.push([c[0] + r * ang.cos() as f32, c[1] + r * ang.sin() as f32]);
                }
                o.fan(c, &outline, inner);
            }
            // Horizon, lines of latitude and meridians (north along the
            // orbit's normal).
            let (e1, e2) = plane_axes(up, nav.normal);
            let circle = |lat: f64| -> Vec<V3> {
                (0..=96)
                    .map(|i| {
                        let t = TAU * i as f64 / 96.0;
                        let h = vec3::add(vec3::scale(e1, t.cos()), vec3::scale(e2, t.sin()));
                        vec3::add(vec3::scale(up, lat.sin()), vec3::scale(h, lat.cos()))
                    })
                    .collect()
            };
            for lat in [-60.0f64, -30.0, 30.0, 60.0] {
                ball_curve(o, c, r, &circle(lat.to_radians()), ui, [1.0, 1.0, 1.0, 0.25]);
            }
            for k in 0..4 {
                let psi = PI * k as f64 / 4.0;
                let h = vec3::add(vec3::scale(e1, psi.cos()), vec3::scale(e2, psi.sin()));
                let meridian: Vec<V3> = (0..=96)
                    .map(|i| {
                        let t = TAU * i as f64 / 96.0;
                        vec3::add(vec3::scale(up, t.cos()), vec3::scale(h, t.sin()))
                    })
                    .collect();
                ball_curve(o, c, r, &meridian, ui, [1.0, 1.0, 1.0, if k == 0 { 0.4 } else { 0.18 }]);
            }
            ball_curve(o, c, r, &circle(0.0), 2.0 * ui, [1.0, 1.0, 1.0, 0.85]);
        }
        None => o.disc(c, r, SKY),
    }

    let neg = |v: Option<V3>| v.map(|v| vec3::scale(v, -1.0));
    let target = nav.target.as_ref().map(|t| t.dir);
    let marks = [
        (Marker::Prograde, nav.prograde),
        (Marker::Retrograde, neg(nav.prograde)),
        (Marker::Normal, nav.normal),
        (Marker::AntiNormal, neg(nav.normal)),
        (Marker::RadialOut, nav.radial),
        (Marker::RadialIn, neg(nav.radial)),
        (Marker::Target, target),
        (Marker::AntiTarget, neg(target)),
    ];
    for (m, dir) in marks {
        let Some(d) = dir.filter(|d| d[0] > 0.0) else { continue };
        let fade = ((d[0] / 0.25) as f32).min(1.0);
        marker(o, m, ball(c, r, d), 6.0 * ui, fade);
    }

    // The ship's nose.
    let k = ui;
    let pts =
        [[-20.0, 0.0], [-8.0, 0.0], [0.0, 8.0], [8.0, 0.0], [20.0, 0.0]].map(|p| [c[0] + p[0] * k, c[1] + p[1] * k]);
    o.polyline(&pts, 3.0 * ui, NOSE);
    o.disc(c, 2.0 * ui, NOSE);
    o.ring(c, r + 3.0 * ui, 4.0 * ui, [0.12, 0.13, 0.15, 1.0]);
}

/// Speed above the navball, throttle to its left, SAS and RCS to its
/// right.
fn gauges(o: &mut Overlay, world: &World, nav: &Nav, controls: &Controls, c: [f32; 2], r: f32, s: f32) {
    let line = 10.0 * s;
    let top = c[1] - r - 3.0 * s;
    let text = format!("{}  {}", nav.name, motion(nav.speed(), nav.rel.gamma));
    let tw = Overlay::text_width(&text, s);
    o.rect([c[0] - tw * 0.5 - 6.0, top - line - 4.0], [c[0] + tw * 0.5 + 6.0, top + 2.0], PANEL);
    o.text([c[0] - tw * 0.5, top - line], &text, s, PROGRADE);

    // Throttle.
    let bw = 5.0 * s;
    let x0 = c[0] - r - 10.0 * s - bw;
    let (y0, y1) = (c[1] - r + line + 4.0, c[1] + r);
    o.rect([x0 - 2.0, y0 - 2.0], [x0 + bw + 2.0, y1 + 2.0], PANEL);
    let fill = y1 - (y1 - y0) * controls.throttle as f32;
    o.rect([x0, fill], [x0 + bw, y1], NOSE);
    // The percentage rides beside the top of the fill.
    let pct = format!("{:.0}%", controls.throttle * 100.0);
    let ty = (fill - 4.0 * s).clamp(y0, y1 - 8.0 * s);
    o.text([x0 - Overlay::text_width(&pct, s) - 3.0 * s, ty], &pct, s, NOSE);
    o.text([x0 + bw * 0.5 - Overlay::text_width("THR", s) * 0.5, y0 - line], "THR", s, DIM);

    // Flight systems.
    let t = world.telemetry();
    let x = c[0] + r + 8.0 * s;
    let mut y = c[1] - r + 2.0;
    let mut row = |o: &mut Overlay, text: &str, color: Rgba| {
        o.text([x, y], text, s, color);
        y += line;
    };
    let sas = if controls.sas { format!("SAS {}", controls.sas_mode.name()) } else { "SAS off".into() };
    row(o, &sas, if controls.sas { ON } else { DIM });
    row(o, if controls.rcs { "RCS on" } else { "RCS off" }, if controls.rcs { ON } else { DIM });
    let g = ENGINE * t.thrust_g * controls.throttle;
    row(o, &format!("accel {}", hud::gees(g)), if g > 0.0 { NOSE } else { DIM });
    row(o, &format!("limit {:.0e}", t.throttle), DIM);
    let warp = if t.warp < 1.5 { "warp 1x".to_string() } else { format!("warp {:.0}x", t.warp) };
    let capped = t.warp >= 0.99 * t.warp_limit && t.warp > 1.5;
    row(o, &warp, if capped { WARN } else { DIM });
}

/// The panel at the top left.
fn readouts(o: &mut Overlay, world: &World, nav: &Nav, s: f32, ui: f32) {
    let t = world.telemetry();
    let rel = &nav.rel;
    let e = &nav.elements;
    let mut lines: Vec<(String, Rgba)> = vec![(nav.situation(world), TEXT)];
    let r = vec3::norm(rel.offset);
    let vertical = vec3::dot(rel.velocity, rel.offset) / r.max(1e-300);
    let sign = match vertical * C_KM_S * 1000.0 {
        v if v >= 0.05 => "+",
        v if v <= -0.05 => "-",
        _ => "",
    };
    let vs = format!("{sign}{}", speed(vertical.abs()));
    match rel.reference {
        Reference::Planet(_) => {
            lines.push((format!("alt {}   vert {vs}", distance(r - rel.radius)), TEXT));
            if e.bound() {
                let (pe, ap) = (distance(e.periapsis - rel.radius), distance(e.apoapsis - rel.radius));
                lines.push((format!("Pe {pe}   Ap {ap}   T {}", hud::seconds(e.period * SECONDS_PER_M)), DIM));
            } else {
                lines.push((format!("Pe {}   escape", distance(e.periapsis - rel.radius)), DIM));
            }
        }
        _ => {
            lines.push((format!("r {}   vert {vs}", distance(r)), TEXT));
            if e.bound() {
                let (pe, ap) = (distance(e.periapsis), distance(e.apoapsis));
                lines.push((format!("Pe {pe}   Ap {ap}   T {}", hud::seconds(e.period * SECONDS_PER_M)), DIM));
            } else {
                lines.push((format!("Pe {}   unbound", distance(e.periapsis)), DIM));
            }
        }
    }
    if let Some(tg) = &nav.target {
        let closing = if tg.closing >= 0.0 { "closing" } else { "opening" };
        lines.push((format!("target {}", tg.name), TARGET));
        lines.push((format!("  {} away, {closing} at {}", distance(tg.distance), speed(tg.closing.abs())), TARGET));
    }
    lines.push((
        format!("ship time {}   universe {}", hud::seconds(t.tau * SECONDS_PER_M), hud::seconds(t.t * SECONDS_PER_M)),
        DIM,
    ));
    if t.gamma > 1.001 && rel.reference != Reference::Hole {
        lines.push((format!("gamma {:.4} in the hole's frame", t.gamma), DIM));
    }
    if let Some(phase) = t.autopilot {
        lines.push((format!("autopilot: {phase}"), ON));
    }
    if rel.reference != Reference::Hole
        && e.periapsis < rel.radius
        && vertical < 0.0
        && !matches!(world.status, PilotStatus::Landed { .. })
    {
        lines.push(("IMPACT COURSE".into(), WARN));
    }
    let wall_to_impact = t.impact_in / (world.cfg.time_scale * t.clock_rate.max(1e-6));
    if wall_to_impact < 60.0 {
        lines.push((format!("HEADING INTO THE HOLE: {wall_to_impact:.0} s (B brakes)"), WARN));
    }
    if t.clock_rate < 0.98 {
        lines.push((format!("clock limited to {:.0}%", t.clock_rate * 100.0), WARN));
    }

    let line = 10.0 * s;
    let pad = 6.0 * ui;
    let width = lines.iter().map(|(l, _)| Overlay::text_width(l, s)).fold(0.0, f32::max);
    let (x, y) = (8.0 * ui, 8.0 * ui);
    o.rect([x, y], [x + width + 2.0 * pad, y + lines.len() as f32 * line + 2.0 * pad - 2.0 * s], PANEL);
    for (i, (text, color)) in lines.iter().enumerate() {
        o.text([x + pad, y + pad + i as f32 * line], text, s, *color);
    }
}

const HELP: [(&str, &str); 27] = [
    ("W / S", "pitch (W: nose down)"),
    ("A / D", "yaw"),
    ("Q / E", "roll"),
    ("T", "SAS on / off (or the buttons"),
    ("", "left of the navball)"),
    ("1 ... 9", "SAS: hold, prograde, retrograde,"),
    ("", "normal, anti-normal, radial out,"),
    ("", "radial in, target, anti-target"),
    ("Shift / Ctrl", "throttle up / down"),
    ("Z / X", "full throttle / cut"),
    ("[ / ]", "thrust limit / 10, x 10"),
    ("R", "RCS on / off"),
    ("H / N  J / L  I / K", "RCS: fwd/back, left/right, up/down"),
    ("B (hold)", "brake to the local frame"),
    ("O", "orbit autopilot to the target"),
    ("Tab", "next target: planets, then Sgr A*"),
    (",  .  /", "time warp down, up, 1x"),
    ("M", "map (click a planet or Sgr A*)"),
    ("F", "map: focus ship, body, target, hole"),
    ("right drag, wheel", "turn and zoom the camera or map"),
    ("Home", "reset the camera or map"),
    ("V", "chase camera / first person"),
    ("P / Y", "optics / Hubble palette"),
    ("PgUp PgDn Bksp", "exposure up, down, reset"),
    ("F2", "hide the HUD"),
    ("F11", "fullscreen"),
    ("Ctrl+Q", "quit      (F1 closes this)"),
];

/// The list of controls, over the middle of the view.
pub fn help(o: &mut Overlay, view: &View, ui: f32) {
    let (w, h) = (view.size.0 as f32, view.size.1 as f32);
    let s = text_scale(ui);
    let line = 10.0 * s;
    let key_w = HELP.iter().map(|(k, _)| Overlay::text_width(k, s)).fold(0.0, f32::max) + 3.0 * s * 8.0 / 2.0;
    let desc_w = HELP.iter().map(|(_, d)| Overlay::text_width(d, s)).fold(0.0, f32::max);
    let (bw, bh) = (key_w + desc_w + 4.0 * line, (HELP.len() + 2) as f32 * line + 2.0 * line);
    let (x, y) = (((w - bw) * 0.5).max(0.0), ((h - bh) * 0.5).max(0.0));
    o.rect([x, y], [x + bw, y + bh], [0.02, 0.03, 0.05, 0.85]);
    o.text([x + 2.0 * line, y + line], "CONTROLS", s, PROGRADE);
    for (i, (k, d)) in HELP.iter().enumerate() {
        let yy = y + (i as f32 + 3.0) * line;
        o.text([x + 2.0 * line, yy], k, s, NOSE);
        o.text([x + 2.0 * line + key_w, yy], d, s, TEXT);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use winit::keyboard::KeyCode;

    /// From a low circular orbit: SAS turns the nose onto prograde, and a
    /// burn there raises the apoapsis and leaves the periapsis where it was.
    #[test]
    fn prograde_burn_raises_the_orbit() {
        let mut world = crate::world(384, "planet".parse().unwrap()).unwrap();
        let mut controls = Controls::default();
        controls.key(KeyCode::Digit2, true);
        assert_eq!(controls.sas_mode, SasMode::Prograde);
        let dt = 1.0 / 60.0;
        for frame in 0..240 {
            if frame == 180 {
                controls.key(KeyCode::KeyZ, true);
            }
            let hold = Nav::new(&world).hold(controls.sas_mode);
            world.step(dt, &controls.sample(dt, hold));
            if frame == 179 {
                let nose = Nav::new(&world).prograde.unwrap();
                assert!(nose[0] > 0.999, "not on prograde after 3 s: {nose:?}");
            }
        }
        let nav = Nav::new(&world);
        assert!(matches!(nav.rel.reference, Reference::Planet(_)));
        let alt = |r: f64| (r - nav.rel.radius) * KM_PER_M;
        let (pe, ap) = (alt(nav.elements.periapsis), alt(nav.elements.apoapsis));
        assert!((pe - 420.0).abs() < 30.0 && ap > 1000.0, "Pe {pe:.0} km, Ap {ap:.0} km");
        assert!(nav.situation(&world).starts_with("ORBITING"));
    }
}
