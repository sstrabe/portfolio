//! Keyboard and mouse → the world's per-frame [`Input`].
//!
//! Click to capture the mouse; while captured, moving it turns the ship
//! like any first-person game (right looks right, down looks down; `I`
//! inverts the vertical axis). Ship axes are (forward, left, up); positive
//! pitch is nose down, positive yaw turns left.

use kerr::world::Input;
use std::collections::HashSet;
use winit::keyboard::KeyCode;

const MOUSE_RAD_PER_PX: f64 = 0.0025;
/// Must match `WorldConfig::turn_rate` (rad per wall second).
const TURN_RATE: f64 = 1.4;

/// Things the window itself should do in response to a key.
pub enum Action {
    ToggleFullscreen,
    /// Esc: release the mouse, then leave fullscreen.
    Release,
    Quit,
    /// Multiply the time warp by this factor.
    Warp(f64),
    InvertY(bool),
}

#[derive(Default)]
pub struct Controls {
    keys: HashSet<KeyCode>,
    pub captured: bool,
    invert_y: bool,
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
        let repeat = !self.keys.insert(code);
        if repeat {
            return None;
        }
        match code {
            KeyCode::F11 => Some(Action::ToggleFullscreen),
            KeyCode::Escape => Some(Action::Release),
            KeyCode::KeyQ if ctrl => Some(Action::Quit),
            KeyCode::Period | KeyCode::Equal | KeyCode::NumpadAdd => Some(Action::Warp(2.0)),
            KeyCode::Comma | KeyCode::Minus | KeyCode::NumpadSubtract => Some(Action::Warp(0.5)),
            KeyCode::KeyI => {
                self.invert_y = !self.invert_y;
                Some(Action::InvertY(self.invert_y))
            }
            _ => None,
        }
    }

    pub fn mouse_motion(&mut self, dx: f64, dy: f64) {
        if self.captured {
            self.dx += dx;
            self.dy += dy;
        }
    }

    pub fn release_all(&mut self) {
        self.keys.clear();
    }

    fn axis(&self, pos: &[KeyCode], neg: &[KeyCode]) -> f64 {
        let p = pos.iter().any(|k| self.keys.contains(k)) as i32;
        let n = neg.iter().any(|k| self.keys.contains(k)) as i32;
        (p - n) as f64
    }

    /// Input for a frame of `dt` seconds. The mouse turns the ship by an
    /// angle proportional to how far it moved.
    pub fn sample(&mut self, dt: f64) -> Input {
        use KeyCode::*;
        let k = if dt > 0.0 { MOUSE_RAD_PER_PX / (dt * TURN_RATE) } else { 0.0 };
        let dy = if self.invert_y { -self.dy } else { self.dy };
        let input = Input {
            thrust: [
                self.axis(&[KeyW], &[KeyS]),
                self.axis(&[KeyA], &[KeyD]),
                self.axis(&[KeyR, Space], &[KeyF, KeyC]),
            ],
            turn: [
                self.axis(&[KeyE], &[KeyQ]),
                // Mouse down → nose down (positive pitch); ↑ → nose up.
                self.axis(&[ArrowDown], &[ArrowUp]) + dy * k,
                // Mouse right → turn right (negative yaw).
                self.axis(&[ArrowLeft], &[ArrowRight]) - self.dx * k,
            ],
            boost: self.keys.contains(&ShiftLeft) || self.keys.contains(&ShiftRight),
            brake: self.keys.contains(&KeyX),
            autopilot: -1,
            ..Default::default()
        };
        self.dx = 0.0;
        self.dy = 0.0;
        input
    }
}
