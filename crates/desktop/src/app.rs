//! The windowed app (winit).

use crate::Options;
use crate::input::{Action, Controls};
use kerr::units::{AU, SECONDS_PER_M};
use kerr::vec3;
use render::{Gpu, Session};
use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::PhysicalKey;
use winit::window::{CursorGrabMode, Fullscreen, Window, WindowId};

/// Lowest ray-tracing resolution the adaptive scaler may pick.
const MIN_SCALE: f32 = 0.5;

pub fn run(options: Options) -> Result<(), String> {
    let event_loop = EventLoop::new().map_err(|e| format!("event loop: {e}"))?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App { options, state: None, error: None };
    event_loop.run_app(&mut app).map_err(|e| format!("event loop: {e}"))?;
    app.error.map_or(Ok(()), Err)
}

struct App {
    options: Options,
    state: Option<State>,
    error: Option<String>,
}

struct State {
    window: Arc<Window>,
    session: Session,
    controls: Controls,
    last: Instant,
    /// Adaptive resolution of the per-pixel ray tracer.
    scale: f32,
    avg_ms: f64,
    title_at: Instant,
    note: Option<(String, Instant)>,
}

impl State {
    fn new(event_loop: &ActiveEventLoop, o: &Options) -> Result<Self, String> {
        let attrs = Window::default_attributes()
            .with_title("Kerr Nucleus")
            .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 720.0));
        let window = Arc::new(event_loop.create_window(attrs).map_err(|e| format!("window: {e}"))?);
        let instance = crate::instance();
        let surface = instance.create_surface(window.clone()).map_err(|e| format!("surface: {e}"))?;
        let size = window.inner_size();
        let world = crate::world(o.stars);
        let (n, cap) = (world.cluster.len() as u32, world.cluster.history.capacity() as u32);
        let gpu = pollster::block_on(Gpu::new(instance, Some(surface), (size.width, size.height), n, cap))?;
        let mut session = Session::new(world, gpu);
        session.fov_deg = o.fov;
        let scale = o.scale.unwrap_or(1.0);
        session.gpu.set_render_scale(scale);
        let now = Instant::now();
        Ok(Self {
            window,
            session,
            controls: Controls::default(),
            last: now,
            scale,
            avg_ms: 16.0,
            title_at: now,
            note: Some(("click to steer with the mouse".into(), now)),
        })
    }

    fn capture_mouse(&mut self, on: bool) {
        let grabbed = if on {
            self.window
                .set_cursor_grab(CursorGrabMode::Locked)
                .or_else(|_| self.window.set_cursor_grab(CursorGrabMode::Confined))
                .is_ok()
        } else {
            let _ = self.window.set_cursor_grab(CursorGrabMode::None);
            false
        };
        self.window.set_cursor_visible(!grabbed);
        self.controls.captured = grabbed;
    }

    fn notify(&mut self, text: impl Into<String>) {
        self.note = Some((text.into(), Instant::now()));
        self.title_at = Instant::now() - std::time::Duration::from_secs(1);
    }

    fn redraw(&mut self, adaptive: bool) {
        let now = Instant::now();
        let dt = now.duration_since(self.last).as_secs_f64().min(0.1);
        self.last = now;
        let input = self.controls.sample(dt);
        self.session.frame(dt, &input);
        if self.session.events().iter().any(|e| matches!(e, kerr::world::WorldEvent::HorizonCrossed)) {
            self.notify("you crossed the event horizon: respawned");
        }

        self.avg_ms = self.avg_ms * 0.95 + dt * 1000.0 * 0.05;
        if adaptive {
            if self.avg_ms > 20.0 && self.scale > MIN_SCALE {
                self.scale = (self.scale * 0.98).max(MIN_SCALE);
            } else if self.avg_ms < 14.0 && self.scale < 1.0 {
                self.scale = (self.scale * 1.01).min(1.0);
            }
            self.session.gpu.set_render_scale(self.scale);
        }
        if now.duration_since(self.title_at).as_secs_f64() > 0.25 {
            self.title_at = now;
            self.window.set_title(&self.title());
        }
    }

    fn title(&self) -> String {
        let w = &self.session.world;
        let t = w.telemetry();
        let pos = w.pilot.position();
        let nearest = w
            .cluster
            .bodies
            .iter()
            .filter(|b| b.alive)
            .map(|b| vec3::norm(vec3::sub(b.position(), pos)))
            .fold(f64::INFINITY, f64::min);
        let warp = w.cfg.time_scale * SECONDS_PER_M;
        let mut s = format!(
            "τ {} · universe {} · ×{:.0} warp · {} · r {} · nearest star {} · {:.0} fps",
            duration(t.tau),
            duration(t.t),
            warp,
            speed(t.speed, t.gamma),
            distance(t.r),
            distance(nearest),
            1000.0 / self.avg_ms,
        );
        let wall_to_impact = t.impact_in / (w.cfg.time_scale * t.clock_rate.max(1e-6));
        if wall_to_impact < 60.0 {
            s = format!("⚠ HEADING INTO THE HOLE: {wall_to_impact:.0} s (X brakes) · {s}");
        }
        if t.clock_rate < 0.98 {
            s += &format!(" · clock limited to {:.0}%", t.clock_rate * 100.0);
        }
        if let Some((note, at)) = &self.note
            && at.elapsed().as_secs_f64() < 4.0
        {
            s = format!("{note} · {s}");
        }
        s
    }
}

/// Human-readable span of `m` units of M.
fn duration(m: f64) -> String {
    let s = m * SECONDS_PER_M;
    match s {
        s if s < 120.0 => format!("{s:.0} s"),
        s if s < 7200.0 => format!("{:.1} min", s / 60.0),
        s if s < 2.0 * 86400.0 => format!("{:.1} h", s / 3600.0),
        s if s < 2.0 * 3.156e7 => format!("{:.1} d", s / 86400.0),
        s => format!("{:.1} yr", s / 3.156e7),
    }
}

fn distance(m: f64) -> String {
    let au = m / AU;
    if au < 0.1 { format!("{:.0} R☉", m / kerr::units::RSUN) } else { format!("{au:.1} AU") }
}

fn speed(v: f64, gamma: f64) -> String {
    if gamma < 1.1 { format!("v {v:.4}c") } else { format!("γ {gamma:.3e}") }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        match State::new(event_loop, &self.options) {
            Ok(s) => self.state = Some(s),
            Err(e) => {
                self.error = Some(e);
                event_loop.exit();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(s) = &mut self.state else { return };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => s.session.gpu.resize(size.width, size.height),
            WindowEvent::Focused(false) => {
                s.controls.release_all();
                s.capture_mouse(false);
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let PhysicalKey::Code(code) = event.physical_key else { return };
                match s.controls.key(code, event.state == ElementState::Pressed) {
                    Some(Action::Quit) => event_loop.exit(),
                    Some(Action::ToggleFullscreen) => {
                        let fs = s.window.fullscreen().is_none().then_some(Fullscreen::Borderless(None));
                        s.window.set_fullscreen(fs);
                    }
                    Some(Action::Release) => {
                        if s.controls.captured {
                            s.capture_mouse(false);
                        } else {
                            s.window.set_fullscreen(None);
                        }
                    }
                    Some(Action::Warp(f)) => {
                        let ts = &mut s.session.world.cfg.time_scale;
                        let real = 1.0 / SECONDS_PER_M;
                        *ts = (*ts * f).clamp(real, 1e7 * real);
                        let warp = *ts * SECONDS_PER_M;
                        s.notify(format!("time warp ×{warp:.0}"));
                    }
                    Some(Action::InvertY(on)) => s.notify(if on { "mouse Y inverted" } else { "mouse Y normal" }),
                    None => {}
                }
            }
            WindowEvent::MouseInput { button: MouseButton::Left, state: ElementState::Pressed, .. } => {
                if !s.controls.captured {
                    s.capture_mouse(true);
                }
            }
            WindowEvent::RedrawRequested => s.redraw(self.options.scale.is_none()),
            _ => {}
        }
    }

    fn device_event(&mut self, _el: &ActiveEventLoop, _id: DeviceId, event: DeviceEvent) {
        if let (Some(s), DeviceEvent::MouseMotion { delta }) = (&mut self.state, event) {
            s.controls.mouse_motion(delta.0, delta.1);
        }
    }

    fn about_to_wait(&mut self, _el: &ActiveEventLoop) {
        if let Some(s) = &self.state {
            s.window.request_redraw();
        }
    }
}
