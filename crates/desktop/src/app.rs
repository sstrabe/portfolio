//! The windowed app (winit).

use crate::Options;
use crate::flight::{self, Nav};
use crate::hud;
use crate::input::{Action, Controls, DRAG_RAD_PER_PX};
use crate::map::{Map, Pick};
use kerr::units::SECONDS_PER_M;
use kerr::world::{PilotStatus, Target, WorldEvent};
use render_hq::{Gpu, Session};
use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::PhysicalKey;
use winit::window::{Fullscreen, Window, WindowId};

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
    map: Map,
    hud: bool,
    help: bool,
    cursor: Option<[f32; 2]>,
    last_click: Option<(Instant, [f32; 2])>,
    /// Where the HUD's buttons were drawn last frame.
    buttons: Vec<flight::ButtonRect>,
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
        let world = crate::world(o.stars, o.start)?;
        let (n, cap) = (world.cluster.len() as u32, world.cluster.history.capacity() as u32);
        let gpu = pollster::block_on(Gpu::new(instance, Some(surface), (size.width, size.height), n, cap))?;
        let mut session = Session::new(world, gpu);
        session.fov_deg = o.fov;
        session.gpu.post.settings = o.optics;
        session.gpu.ship.camera.chase = o.chase;
        let scale = o.scale.unwrap_or(1.0);
        session.gpu.set_render_scale(scale);
        let now = Instant::now();
        Ok(Self {
            window,
            session,
            controls: Controls::default(),
            map: Map::default(),
            hud: true,
            help: false,
            cursor: None,
            last_click: None,
            buttons: Vec::new(),
            last: now,
            scale,
            avg_ms: 16.0,
            title_at: now,
            note: Some(("F1: controls   M: map".into(), now)),
        })
    }

    fn notify(&mut self, text: impl Into<String>) {
        self.note = Some((text.into(), Instant::now()));
    }

    fn ui(&self) -> f32 {
        self.window.scale_factor() as f32
    }

    fn redraw(&mut self, adaptive: bool) {
        let now = Instant::now();
        let dt = now.duration_since(self.last).as_secs_f64().min(0.1);
        self.last = now;

        // The mouse turns and zooms the map or the chase camera.
        let drag = self.controls.take_drag();
        let wheel = self.controls.take_wheel();
        if self.map.on {
            self.map.orbit(drag);
            self.map.zoom(wheel);
        } else {
            let cam = &mut self.session.gpu.ship.camera;
            cam.orbit(-drag[0] * DRAG_RAD_PER_PX, drag[1] * DRAG_RAD_PER_PX);
            cam.zoom(1.15f64.powf(-wheel));
        }

        // SAS steers unless an autopilot does.
        let world = &self.session.world;
        let autopilot =
            matches!(world.status, PilotStatus::Orbit(_) | PilotStatus::HoleOrbit | PilotStatus::Autopilot(_));
        let hold = if self.controls.sas && !autopilot { Nav::new(world).hold(self.controls.sas_mode) } else { None };
        let input = self.controls.sample(dt, hold);
        let (torque, force) = self.controls.rcs_command();
        self.session.gpu.ship.set_rcs(torque, force);
        self.session.advance(dt, &input);
        self.session.gpu.ship.power = self.controls.throttle;
        let world = &self.session.world;
        let mut landed = false;
        let mut note = None;
        for ev in self.session.events() {
            landed |= matches!(ev, WorldEvent::Landed { .. });
            note = hud::note(world, ev).or(note);
        }
        if landed {
            // Otherwise the engine lifts off again at once.
            self.controls.throttle = 0.0;
        }
        if let Some(note) = note {
            self.notify(note);
        }

        let nav = Nav::new(&self.session.world);
        let view = self.session.view();
        let ui = self.ui();
        let note = self.note.as_ref().filter(|(_, at)| at.elapsed().as_secs_f64() < 4.0).map(|(n, _)| n.clone());
        let Session { world, gpu, .. } = &mut self.session;
        if self.map.on {
            self.map.draw(&mut gpu.overlay, world, &nav, view.size, ui, self.cursor);
        }
        let extras = flight::Extras {
            fps: 1000.0 / self.avg_ms,
            note: note.as_deref(),
            help: self.help,
            map: self.map.on,
            cursor: self.cursor,
        };
        self.buttons.clear();
        if self.hud {
            self.buttons = flight::draw(&mut gpu.overlay, world, &nav, &self.controls, &view, ui, &extras);
        } else if self.help {
            flight::help(&mut gpu.overlay, &view, ui);
        }
        if self.map.on {
            self.session.gpu.render_overlay(Map::background());
        } else {
            self.session.render();
        }

        self.avg_ms = self.avg_ms * 0.95 + dt * 1000.0 * 0.05;
        if adaptive && !self.map.on {
            if self.avg_ms > 20.0 && self.scale > MIN_SCALE {
                self.scale = (self.scale * 0.98).max(MIN_SCALE);
            } else if self.avg_ms < 14.0 && self.scale < 1.0 {
                self.scale = (self.scale * 1.01).min(1.0);
            }
            self.session.gpu.set_render_scale(self.scale);
        }
        if now.duration_since(self.title_at).as_secs_f64() > 0.5 {
            self.title_at = now;
            let fps = 1000.0 / self.avg_ms;
            let title = format!("Kerr Nucleus · {} · {fps:.0} fps", nav.situation(&self.session.world));
            self.window.set_title(&title);
        }
    }

    fn action(&mut self, event_loop: &ActiveEventLoop, action: Action) {
        match action {
            Action::Quit => event_loop.exit(),
            Action::ToggleFullscreen => {
                let fs = self.window.fullscreen().is_none().then_some(Fullscreen::Borderless(None));
                self.window.set_fullscreen(fs);
            }
            Action::Back => {
                if self.help {
                    self.help = false;
                } else if self.map.on {
                    self.map.on = false;
                } else {
                    self.window.set_fullscreen(None);
                }
            }
            Action::Warp(f) => {
                let world = &mut self.session.world;
                let real = 1.0 / SECONDS_PER_M;
                let limit = world.warp_limit();
                let wanted = if f == 0.0 { real } else { (world.cfg.time_scale * f).clamp(real, 1e7 * real) };
                world.cfg.time_scale = wanted.min(limit);
                let warp = world.cfg.time_scale * SECONDS_PER_M;
                let capped = if wanted > limit { " (the most allowed this close)" } else { "" };
                self.notify(format!("time warp {warp:.0}x{capped}"));
            }
            Action::ThrustLimit(f) => {
                let world = &mut self.session.world;
                world.scale_throttle(f);
                let t = world.telemetry();
                self.notify(hud::throttle(&t));
            }
            Action::OrbitAutopilot => {
                let world = &mut self.session.world;
                let was_on = matches!(world.status, PilotStatus::Orbit(_) | PilotStatus::HoleOrbit);
                let note = match world.toggle_orbit_autopilot() {
                    Some(t) => {
                        // The engine and the rotation controls would take
                        // over again at once.
                        self.controls.throttle = 0.0;
                        self.controls.stop_spin();
                        format!("autopilot: into orbit around {}", hud::target_name(world, t))
                    }
                    None if was_on => "autopilot off".into(),
                    None => "the autopilot can't reach the target".into(),
                };
                self.notify(note);
            }
            Action::NextTarget => {
                let world = &mut self.session.world;
                let note = match world.cycle_target() {
                    Some(t) => format!("target: {}", hud::target_name(world, t)),
                    None => "nothing to target".into(),
                };
                self.notify(note);
            }
            Action::NextOptics => {
                let o = &mut self.session.gpu.post.settings;
                o.optics = o.optics.next();
                let note = format!("optics: {:?}", o.optics).to_lowercase();
                self.notify(note);
            }
            Action::TogglePalette => {
                use render_hq::optics::Palette;
                let o = &mut self.session.gpu.post.settings;
                o.palette = if o.palette == Palette::True { Palette::Hubble } else { Palette::True };
                let note =
                    if o.palette == Palette::Hubble { "Hubble palette ([S II], Hα, [O III])" } else { "true colour" };
                self.notify(note);
            }
            Action::Exposure(stops) => {
                let o = &mut self.session.gpu.post.settings;
                o.ev = if stops == 0.0 { 0.0 } else { o.ev + stops };
                let note = format!("exposure {:+.0} EV", o.ev);
                self.notify(note);
            }
            Action::ToggleChase => {
                let cam = &mut self.session.gpu.ship.camera;
                cam.chase = !cam.chase;
                let note = if cam.chase { "chase camera (right drag turns it, wheel zooms)" } else { "first person" };
                self.notify(note);
            }
            Action::ToggleMap => {
                if self.map.on {
                    self.map.on = false;
                } else {
                    self.map.open(&Nav::new(&self.session.world));
                }
            }
            Action::MapFocus => {
                if self.map.on {
                    let world = &self.session.world;
                    let pick = self.map.next_focus(world, &Nav::new(world));
                    let note = format!("map: {}", Map::name(world, pick));
                    self.notify(note);
                }
            }
            Action::ResetView => {
                if self.map.on {
                    self.map.reset(&Nav::new(&self.session.world));
                } else {
                    self.session.gpu.ship.camera.reset();
                }
            }
            Action::ToggleHud => self.hud = !self.hud,
            Action::ToggleHelp => self.help = !self.help,
            Action::Note(text) => self.notify(text),
        }
    }

    /// Left click: on the map, a planet becomes the target, and a double
    /// click centres the map on whatever was clicked.
    fn click(&mut self) {
        if let Some(button) = self.cursor.and_then(|p| flight::button_at(&self.buttons, p)) {
            let note = match button {
                flight::Button::Sas => self.controls.toggle_sas(),
                flight::Button::Rcs => self.controls.toggle_rcs(),
                flight::Button::Mode(m) => self.controls.set_sas_mode(m),
            };
            self.notify(note);
            return;
        }
        let Some(cursor) = self.cursor.filter(|_| self.map.on) else { return };
        let now = Instant::now();
        let double = self.last_click.is_some_and(|(at, p)| {
            now.duration_since(at).as_secs_f64() < 0.4 && (p[0] - cursor[0]).abs() + (p[1] - cursor[1]).abs() < 8.0
        });
        self.last_click = Some((now, cursor));
        let Some(pick) = self.map.pick(cursor, self.ui()) else { return };
        let world = &mut self.session.world;
        let note = if double {
            self.map.focus_on(world, pick);
            format!("map: {}", Map::name(world, pick))
        } else if let Some(target) = match pick {
            Pick::Planet(p) => Some(Target::Planet(p)),
            Pick::Hole => Some(Target::Hole),
            _ => None,
        } && world.set_target(target)
        {
            format!("target: {} (O flies there)", hud::target_name(world, target))
        } else {
            format!("{} (double click to centre)", Map::name(world, pick))
        };
        self.notify(note);
    }
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
            WindowEvent::Focused(false) => s.controls.release_all(),
            WindowEvent::KeyboardInput { event, is_synthetic, .. } => {
                // On focus, Windows reports keys it believes are held as
                // synthetic presses; stale state there once held a strafe
                // key down forever. Only real presses count.
                if is_synthetic && event.state == ElementState::Pressed {
                    return;
                }
                let PhysicalKey::Code(code) = event.physical_key else { return };
                if let Some(action) = s.controls.key(code, event.state == ElementState::Pressed) {
                    s.action(event_loop, action);
                }
            }
            WindowEvent::CursorMoved { position, .. } => s.cursor = Some([position.x as f32, position.y as f32]),
            WindowEvent::CursorLeft { .. } => s.cursor = None,
            WindowEvent::MouseInput { button: MouseButton::Right, state, .. } => {
                s.controls.right_button(state == ElementState::Pressed);
            }
            WindowEvent::MouseInput { button: MouseButton::Left, state: ElementState::Pressed, .. } => s.click(),
            WindowEvent::MouseWheel { delta, .. } => {
                let notches = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y as f64,
                    MouseScrollDelta::PixelDelta(p) => p.y / 60.0,
                };
                s.controls.wheel(notches);
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
