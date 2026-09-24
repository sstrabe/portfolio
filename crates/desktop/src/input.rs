//! Keyboard and mouse → the world's per-frame [`Input`]. Same flight
//! bindings as the web version (`web/src/immersive/input.ts`).
//!
//! Ship axes are (forward, left, up); positive pitch is nose down, positive
//! yaw turns left, positive roll lifts the left wing.

use kerr::world::Input;
use std::collections::HashSet;
use winit::keyboard::KeyCode;

const MOUSE_RAD_PER_PX: f64 = 0.0035;
/// Must match `WorldConfig::turn_rate` (rad per wall second).
const TURN_RATE: f64 = 1.4;

/// Things the window itself should do in response to a key.
pub enum Action {
    ToggleFullscreen,
    LeaveFullscreen,
    Quit,
}

#[derive(Default)]
pub struct Controls {
    keys: HashSet<KeyCode>,
    dragging: bool,
    dx: f64,
    dy: f64,
}

impl Controls {
    pub fn key(&mut self, code: KeyCode, pressed: bool) -> Option<Action> {
        if !pressed {
            self.keys.remove(&code);
            return None;
        }
        let ctrl = self.keys.contains(&KeyCode::ControlLeft) || self.keys.contains(&KeyCode::ControlRight);
        self.keys.insert(code);
        match code {
            KeyCode::F11 => Some(Action::ToggleFullscreen),
            KeyCode::Escape => Some(Action::LeaveFullscreen),
            KeyCode::KeyQ if ctrl => Some(Action::Quit),
            _ => None,
        }
    }

    pub fn mouse_button(&mut self, pressed: bool) {
        self.dragging = pressed;
    }

    pub fn mouse_motion(&mut self, dx: f64, dy: f64) {
        if self.dragging {
            self.dx += dx;
            self.dy += dy;
        }
    }

    pub fn release_all(&mut self) {
        self.keys.clear();
        self.dragging = false;
    }

    fn axis(&self, pos: &[KeyCode], neg: &[KeyCode]) -> f64 {
        let p = pos.iter().any(|k| self.keys.contains(k)) as i32;
        let n = neg.iter().any(|k| self.keys.contains(k)) as i32;
        (p - n) as f64
    }

    /// Input for a frame of `dt` seconds. Dragging turns by an angle
    /// proportional to the distance moved.
    pub fn sample(&mut self, dt: f64) -> Input {
        use KeyCode::*;
        let k = if dt > 0.0 { MOUSE_RAD_PER_PX / (dt * TURN_RATE) } else { 0.0 };
        let input = Input {
            thrust: [
                self.axis(&[KeyW], &[KeyS]),
                self.axis(&[KeyA], &[KeyD]),
                self.axis(&[KeyR, Space], &[KeyF, KeyC]),
            ],
            turn: [
                self.axis(&[KeyE], &[KeyQ]),
                self.axis(&[ArrowDown], &[ArrowUp]) + self.dy * k,
                self.axis(&[ArrowLeft], &[ArrowRight]) - self.dx * k,
            ],
            boost: self.keys.contains(&ShiftLeft) || self.keys.contains(&ShiftRight),
            autopilot: -1,
            ..Default::default()
        };
        self.dx = 0.0;
        self.dy = 0.0;
        input
    }
}
