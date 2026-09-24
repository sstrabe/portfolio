//! The windowed app (winit).

use crate::Options;
use crate::input::{Action, Controls};
use render::{Gpu, Session};
use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::PhysicalKey;
use winit::window::{Fullscreen, Window, WindowId};

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
    frames: u32,
    title_at: Instant,
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
        let scale = o.scale.unwrap_or(0.7);
        session.gpu.set_render_scale(scale);
        let now = Instant::now();
        Ok(Self {
            window,
            session,
            controls: Controls::default(),
            last: now,
            scale,
            avg_ms: 16.0,
            frames: 0,
            title_at: now,
        })
    }

    fn redraw(&mut self, adaptive: bool) {
        let now = Instant::now();
        let dt = now.duration_since(self.last).as_secs_f64().min(0.1);
        self.last = now;
        let input = self.controls.sample(dt);
        self.session.frame(dt, &input);
        for ev in self.session.events() {
            if let kerr::world::WorldEvent::HorizonCrossed = ev {
                println!("You crossed the event horizon. Respawned.");
            }
        }

        self.frames += 1;
        self.avg_ms = self.avg_ms * 0.95 + dt * 1000.0 * 0.05;
        if adaptive {
            if self.avg_ms > 20.0 && self.scale > 0.25 {
                self.scale = (self.scale * 0.97).max(0.25);
            } else if self.avg_ms < 14.0 && self.scale < 1.0 {
                self.scale = (self.scale * 1.01).min(1.0);
            }
            self.session.gpu.set_render_scale(self.scale);
        }
        if now.duration_since(self.title_at).as_secs_f64() > 0.25 {
            let t = self.session.world.telemetry();
            self.window.set_title(&format!(
                "Kerr Nucleus · τ {:.1} M · t {:.1} M · dt/dτ {:.3} · v {:.4}c · γ {:.3} · r {:.2} M · {:.0} fps · {:.0}% res",
                t.tau,
                t.t,
                t.dt_dtau,
                t.speed,
                t.gamma,
                t.r,
                1000.0 / self.avg_ms,
                self.scale * 100.0,
            ));
            self.title_at = now;
        }
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
            WindowEvent::KeyboardInput { event, .. } => {
                let PhysicalKey::Code(code) = event.physical_key else { return };
                match s.controls.key(code, event.state == ElementState::Pressed) {
                    Some(Action::Quit) => event_loop.exit(),
                    Some(Action::ToggleFullscreen) => {
                        let fs = s.window.fullscreen().is_none().then_some(Fullscreen::Borderless(None));
                        s.window.set_fullscreen(fs);
                    }
                    Some(Action::LeaveFullscreen) => s.window.set_fullscreen(None),
                    None => {}
                }
            }
            WindowEvent::MouseInput { button: MouseButton::Left, state, .. } => {
                s.controls.mouse_button(state == ElementState::Pressed);
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
