//! Keyboard and mouse → the world's per-frame [`Input`], with flight
//! controls like Kerbal Space Program's.
//!
//! * W/S pitch (W puts the nose down), A/D yaw, Q/E roll. The ship turns
//!   with inertia: keys spin it up, and SAS (T) spins it down again when
//!   they are let go; without SAS it keeps turning. SAS can also hold the
//!   nose on prograde, retrograde, normal, radial or the target (1–9).
//! * The main engine has a throttle that stays where it is set: Shift
//!   opens it, Ctrl closes it, Z is full and X is cut. `[`/`]` change the
//!   engine's thrust limit by factors of ten.
//! * RCS (R) translates: H/N forward/back, J/L left/right, I/K up/down.
//!
//! Ship axes are (forward, left, up); positive pitch is nose down,
//! positive yaw turns left, positive roll rolls right.

use kerr::vec3::{self, V3};
use kerr::world::Input;
use std::collections::HashSet;
use winit::keyboard::KeyCode;

/// Main engine thrust at full throttle, in units of the world's thrust
/// limit (`World::throttle` × `WorldConfig::thrust`).
pub const ENGINE: f64 = 2.0;
/// RCS thrust, same units.
pub const RCS: f64 = 0.25;
/// Throttle change per second while Shift or Ctrl is held.
const THROTTLE_RATE: f64 = 0.8;
/// Angular acceleration of the ship, in maximum turn rates per second:
/// from rest to full rate in half a second.
const SPIN_ACCEL: f64 = 2.0;
/// Must match `WorldConfig::turn_rate` (rad per wall second).
const TURN_RATE: f64 = 1.4;
/// Map and chase-camera orbiting, rad per pixel dragged.
pub const DRAG_RAD_PER_PX: f64 = 0.006;

/// What SAS does when no rotation key is held.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SasMode {
    /// Stop turning and hold the attitude.
    Stability,
    Prograde,
    Retrograde,
    Normal,
    AntiNormal,
    RadialOut,
    RadialIn,
    Target,
    AntiTarget,
}

impl SasMode {
    pub fn name(self) -> &'static str {
        match self {
            SasMode::Stability => "stability",
            SasMode::Prograde => "prograde",
            SasMode::Retrograde => "retrograde",
            SasMode::Normal => "normal",
            SasMode::AntiNormal => "anti-normal",
            SasMode::RadialOut => "radial out",
            SasMode::RadialIn => "radial in",
            SasMode::Target => "target",
            SasMode::AntiTarget => "anti-target",
        }
    }
}

/// Things the window itself should do in response to a key.
pub enum Action {
    ToggleFullscreen,
    /// Esc: close the help or the map, else leave fullscreen.
    Back,
    Quit,
    /// Multiply the time warp by this factor (0: back to real time).
    Warp(f64),
    /// Multiply the engine's thrust limit by this factor.
    ThrustLimit(f64),
    /// Engage or release the orbit autopilot.
    OrbitAutopilot,
    /// Target the next planet (then Sgr A*).
    NextTarget,
    /// Cycle eye → camera → astrograph.
    NextOptics,
    /// Toggle true colour / the Hubble palette.
    TogglePalette,
    /// Change the exposure compensation by this many stops (0: reset).
    Exposure(f64),
    /// First person / chase camera.
    ToggleChase,
    ToggleMap,
    ToggleHud,
    ToggleHelp,
    /// Map: centre on the next thing.
    MapFocus,
    /// Put the chase camera or the map view back to its default.
    ResetView,
    /// A flight-control change worth a note (SAS, RCS, throttle).
    Note(String),
}

pub struct Controls {
    keys: HashSet<KeyCode>,
    /// Main engine throttle, 0–1.
    pub throttle: f64,
    pub sas: bool,
    pub sas_mode: SasMode,
    pub rcs: bool,
    /// Rate of turn (roll, pitch, yaw) as a fraction of the maximum.
    spin: V3,
    /// What the reaction control did last frame: angular acceleration
    /// and translation, as shares of the most it can give.
    rcs_torque: V3,
    rcs_force: V3,
    /// Right mouse button held: dragging orbits the camera.
    dragging: bool,
    drag: [f64; 2],
    wheel: f64,
}

impl Default for Controls {
    fn default() -> Self {
        Self {
            keys: HashSet::new(),
            throttle: 0.0,
            sas: true,
            sas_mode: SasMode::Stability,
            rcs: false,
            spin: [0.0; 3],
            rcs_torque: [0.0; 3],
            rcs_force: [0.0; 3],
            dragging: false,
            drag: [0.0; 2],
            wheel: 0.0,
        }
    }
}

impl Controls {
    pub fn key(&mut self, code: KeyCode, pressed: bool) -> Option<Action> {
        if !pressed {
            self.keys.remove(&code);
            return None;
        }
        let ctrl = self.held(&[KeyCode::ControlLeft, KeyCode::ControlRight]);
        let repeat = !self.keys.insert(code);
        if repeat {
            return None;
        }
        use KeyCode::*;
        let sas = |mode: SasMode| Some(mode);
        let mode = match code {
            Digit1 => sas(SasMode::Stability),
            Digit2 => sas(SasMode::Prograde),
            Digit3 => sas(SasMode::Retrograde),
            Digit4 => sas(SasMode::Normal),
            Digit5 => sas(SasMode::AntiNormal),
            Digit6 => sas(SasMode::RadialOut),
            Digit7 => sas(SasMode::RadialIn),
            Digit8 => sas(SasMode::Target),
            Digit9 => sas(SasMode::AntiTarget),
            _ => None,
        };
        if let Some(mode) = mode {
            return Some(Action::Note(self.set_sas_mode(mode)));
        }
        match code {
            F11 => Some(Action::ToggleFullscreen),
            Escape => Some(Action::Back),
            KeyQ if ctrl => Some(Action::Quit),
            Period | Equal | NumpadAdd => Some(Action::Warp(2.0)),
            Comma | Minus | NumpadSubtract => Some(Action::Warp(0.5)),
            Slash => Some(Action::Warp(0.0)),
            BracketRight => Some(Action::ThrustLimit(10.0)),
            BracketLeft => Some(Action::ThrustLimit(0.1)),
            KeyT => Some(Action::Note(self.toggle_sas())),
            KeyR => Some(Action::Note(self.toggle_rcs())),
            KeyZ => {
                self.throttle = 1.0;
                None
            }
            KeyX => {
                self.throttle = 0.0;
                None
            }
            KeyO => Some(Action::OrbitAutopilot),
            Tab => Some(Action::NextTarget),
            KeyM => Some(Action::ToggleMap),
            KeyF => Some(Action::MapFocus),
            KeyP => Some(Action::NextOptics),
            KeyY => Some(Action::TogglePalette),
            KeyV => Some(Action::ToggleChase),
            Home => Some(Action::ResetView),
            F1 => Some(Action::ToggleHelp),
            F2 => Some(Action::ToggleHud),
            PageUp => Some(Action::Exposure(1.0)),
            PageDown => Some(Action::Exposure(-1.0)),
            Backspace => Some(Action::Exposure(0.0)),
            _ => None,
        }
    }

    /// Switch SAS on or off; returns a note saying which.
    pub fn toggle_sas(&mut self) -> String {
        self.sas = !self.sas;
        if self.sas { format!("SAS on: {}", self.sas_mode.name()) } else { "SAS off".into() }
    }

    pub fn toggle_rcs(&mut self) -> String {
        self.rcs = !self.rcs;
        if self.rcs { "RCS on" } else { "RCS off" }.into()
    }

    /// Hold the nose on `mode` (switching SAS on).
    pub fn set_sas_mode(&mut self, mode: SasMode) -> String {
        self.sas = true;
        self.sas_mode = mode;
        format!("SAS: {}", mode.name())
    }

    /// Right mouse button pressed or released.
    pub fn right_button(&mut self, pressed: bool) {
        self.dragging = pressed;
    }

    /// Cursor moved by this many pixels.
    pub fn mouse_motion(&mut self, dx: f64, dy: f64) {
        if self.dragging {
            self.drag[0] += dx;
            self.drag[1] += dy;
        }
    }

    /// Pixels dragged with the right button since the last call.
    pub fn take_drag(&mut self) -> [f64; 2] {
        std::mem::take(&mut self.drag)
    }

    /// Mouse wheel notches (or 60 px of touchpad scrolling) since the last
    /// call.
    pub fn wheel(&mut self, notches: f64) {
        self.wheel += notches;
    }

    pub fn take_wheel(&mut self) -> f64 {
        std::mem::take(&mut self.wheel)
    }

    pub fn release_all(&mut self) {
        self.keys.clear();
        self.dragging = false;
    }

    /// Angular acceleration (roll, pitch, yaw) and translation the
    /// reaction control gave in the last frame, shares −1…1 of its most.
    pub fn rcs_command(&self) -> (V3, V3) {
        (self.rcs_torque, self.rcs_force)
    }

    /// Stop turning at once (when an autopilot takes over).
    pub fn stop_spin(&mut self) {
        self.spin = [0.0; 3];
    }

    fn held(&self, keys: &[KeyCode]) -> bool {
        keys.iter().any(|k| self.keys.contains(k))
    }

    fn axis(&self, pos: &[KeyCode], neg: &[KeyCode]) -> f64 {
        (self.held(pos) as i32 - self.held(neg) as i32) as f64
    }

    /// Input for a frame of `dt` seconds. `hold` is the ship-frame
    /// direction the SAS mode wants the nose on, if it has one and it
    /// exists (no target, no motion: SAS just holds the attitude).
    pub fn sample(&mut self, dt: f64, hold: Option<V3>) -> Input {
        use KeyCode::*;
        let up = self.held(&[ShiftLeft, ShiftRight]) as i32 as f64;
        let down = self.held(&[ControlLeft, ControlRight]) as i32 as f64;
        self.throttle = (self.throttle + (up - down) * THROTTLE_RATE * dt).clamp(0.0, 1.0);

        let command = [self.axis(&[KeyE], &[KeyQ]), self.axis(&[KeyW], &[KeyS]), self.axis(&[KeyA], &[KeyD])];
        let target = match (self.sas, hold) {
            (true, Some(dir)) => aim(dir),
            (true, None) => [0.0; 3],
            // Without SAS the ship keeps turning.
            (false, _) => self.spin,
        };
        let step = SPIN_ACCEL * dt;
        let before = self.spin;
        for i in 0..3 {
            let want = if command[i] != 0.0 { command[i] } else { target[i] };
            self.spin[i] += (want - self.spin[i]).clamp(-step, step);
            if self.spin[i].abs() < 1e-4 && command[i] == 0.0 {
                self.spin[i] = 0.0;
            }
        }

        let rcs = if self.rcs {
            [self.axis(&[KeyH], &[KeyN]), self.axis(&[KeyJ], &[KeyL]), self.axis(&[KeyI], &[KeyK])]
        } else {
            [0.0; 3]
        };
        // The thrusters fire while the rotation speeds up or slows down.
        self.rcs_torque = std::array::from_fn(|i| if step > 0.0 { (self.spin[i] - before[i]) / step } else { 0.0 });
        self.rcs_force = rcs;
        Input {
            thrust: vec3::axpy(vec3::scale(rcs, RCS), self.throttle * ENGINE, [1.0, 0.0, 0.0]),
            turn: self.spin,
            brake: self.held(&[KeyB]),
            autopilot: -1,
            ..Default::default()
        }
    }
}

/// Rate of turn (fractions of the maximum) that brings the nose onto
/// ship-frame direction `dir` without overshooting: as fast as the ship
/// can still stop in the angle left, and no roll.
fn aim(dir: V3) -> V3 {
    let dir = vec3::normalize(dir);
    let axis = vec3::cross([1.0, 0.0, 0.0], dir);
    let angle = vec3::norm(axis).atan2(dir[0]);
    if angle < 1e-4 {
        return [0.0; 3];
    }
    // Stopping from rate ω (rad/s) at acceleration α takes ω²/2α of angle;
    // 0.7 of that leaves a margin for the frame-by-frame update.
    let accel = SPIN_ACCEL * TURN_RATE;
    let rate = (0.7 * (2.0 * accel * angle).sqrt()).min(4.0 * angle) / TURN_RATE;
    let axis = if vec3::norm(axis) > 1e-9 { vec3::normalize(axis) } else { [0.0, 0.0, 1.0] };
    vec3::scale([0.0, axis[1], axis[2]], rate.min(1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A nose held off to the left and above turns left (positive yaw) and
    /// up (negative pitch), and settles on it without overshooting.
    #[test]
    fn sas_settles_on_a_direction() {
        let mut c = Controls::default();
        let mut dir = vec3::normalize([1.0, 1.0, 0.5]);
        let dt = 1.0 / 60.0;
        let first = c.sample(dt, Some(dir)).turn;
        assert!(first[1] < 0.0 && first[2] > 0.0, "{first:?}");
        let mut min_forward: f64 = 1.0;
        for _ in 0..600 {
            let spin = c.sample(dt, Some(dir)).turn;
            // Turn the direction as the ship would (the world applies the
            // same rotation to the ship's axes).
            let w = vec3::scale(spin, TURN_RATE * dt);
            if vec3::norm(w) > 0.0 {
                dir = vec3::rotate(dir, vec3::normalize(w), -vec3::norm(w));
            }
            min_forward = min_forward.min(dir[0]);
        }
        assert!(dir[0] > 0.9999, "{dir:?}");
        assert!(vec3::norm(c.spin) < 1e-3);
        assert!(min_forward > 0.5);
    }

    #[test]
    fn throttle_keys() {
        let mut c = Controls::default();
        c.key(KeyCode::KeyZ, true);
        assert_eq!(c.sample(0.1, None).thrust, [ENGINE, 0.0, 0.0]);
        c.key(KeyCode::KeyX, true);
        c.key(KeyCode::ShiftLeft, true);
        let t = c.sample(0.5, None).thrust[0];
        assert!((t - 0.4 * ENGINE).abs() < 1e-12, "{t}");
        // RCS only fires when it is on.
        c.key(KeyCode::KeyJ, true);
        assert_eq!(c.sample(0.0, None).thrust[1], 0.0);
        c.key(KeyCode::KeyR, true);
        assert_eq!(c.sample(0.0, None).thrust[1], RCS);
    }
}
