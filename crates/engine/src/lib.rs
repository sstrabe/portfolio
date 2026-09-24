//! Browser entry point. The web layer creates an [`Engine`] for a canvas,
//! calls [`Engine::frame`] from `requestAnimationFrame` with the input
//! state, and reads back the stations' apparent screen positions to float
//! (or composite) the portfolio content over them.

#[cfg(target_arch = "wasm32")]
pub use web::*;

#[cfg(target_arch = "wasm32")]
mod web {
    use kerr::world::{Input, World, WorldEvent};
    use render::frame::StationScreen;
    use render::{Gpu, Session, world_config};
    use wasm_bindgen::prelude::*;

    /// Floats per station in [`Engine::stations`].
    pub const STATION_STRIDE: usize = StationScreen::STRIDE;

    #[wasm_bindgen]
    pub struct Engine {
        session: Session,
    }

    /// Create the engine for `canvas` (its `width`/`height` must already be
    /// set in device pixels). `stars` sets the cluster size.
    #[wasm_bindgen(js_name = createEngine)]
    pub async fn create_engine(
        canvas: web_sys::HtmlCanvasElement,
        station_count: u32,
        stars: u32,
        seed: u32,
    ) -> Result<Engine, JsValue> {
        Engine::build(Some(canvas), (0, 0), station_count, stars, seed).await
    }

    /// An engine that renders offscreen at `width` × `height` and exposes its
    /// frames through [`Engine::capture`], for automated visual checks in
    /// environments that cannot present WebGPU canvases.
    #[wasm_bindgen(js_name = createHeadlessEngine)]
    pub async fn create_headless_engine(
        width: u32,
        height: u32,
        station_count: u32,
        stars: u32,
        seed: u32,
    ) -> Result<Engine, JsValue> {
        Engine::build(None, (width, height), station_count, stars, seed).await
    }

    impl Engine {
        async fn build(
            canvas: Option<web_sys::HtmlCanvasElement>,
            size: (u32, u32),
            station_count: u32,
            stars: u32,
            seed: u32,
        ) -> Result<Engine, JsValue> {
            console_error_panic_hook::set_once();
            let world = World::new(world_config(station_count.max(1), stars, seed));
            let n = world.cluster.len() as u32;
            let cap = world.cluster.history.capacity() as u32;
            let size = canvas.as_ref().map_or(size, |c| (c.width(), c.height()));
            let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
            desc.backends = wgpu::Backends::BROWSER_WEBGPU;
            let instance = wgpu::Instance::new(desc);
            let surface = match canvas {
                Some(c) => Some(
                    instance
                        .create_surface(wgpu::SurfaceTarget::Canvas(c))
                        .map_err(|e| JsValue::from_str(&format!("surface: {e}")))?,
                ),
                None => None,
            };
            let gpu = Gpu::new(instance, surface, size, n, cap).await.map_err(|e| JsValue::from_str(&e))?;
            Ok(Engine { session: Session::new(world, gpu) })
        }
    }

    #[wasm_bindgen]
    impl Engine {
        /// Resize the swap chain (device pixels).
        pub fn resize(&mut self, width: u32, height: u32) {
            self.session.gpu.resize(width, height);
        }

        /// Fraction of the output resolution used for ray tracing.
        #[wasm_bindgen(js_name = setRenderScale)]
        pub fn set_render_scale(&mut self, scale: f32) {
            self.session.gpu.set_render_scale(scale);
        }

        #[wasm_bindgen(js_name = setFov)]
        pub fn set_fov(&mut self, degrees: f64) {
            self.session.fov_deg = degrees.clamp(30.0, 120.0);
        }

        #[wasm_bindgen(js_name = setExposure)]
        pub fn set_exposure(&mut self, exposure: f64) {
            self.session.exposure = exposure.clamp(0.05, 20.0);
        }

        /// Pilot proper time per wall-clock second (units of M).
        #[wasm_bindgen(js_name = setTimeScale)]
        pub fn set_time_scale(&mut self, scale: f64) {
            self.session.world.cfg.time_scale = scale.clamp(0.0, 200.0);
        }

        /// Station whose panel frame should glow (-1 for none).
        #[wasm_bindgen(js_name = setHighlight)]
        pub fn set_highlight(&mut self, station: i32) {
            self.session.highlight = station;
        }

        #[wasm_bindgen(js_name = stationCount)]
        pub fn station_count(&self) -> u32 {
            self.session.world.station_count() as u32
        }

        /// Copy the panel atlas canvas (4 × 2 cells of 512 × 320 px).
        #[wasm_bindgen(js_name = uploadPanelAtlas)]
        pub fn upload_panel_atlas(&self, canvas: web_sys::HtmlCanvasElement) {
            self.session.gpu.upload_atlas(canvas);
        }

        /// Advance by `wall_dt` seconds and render.
        ///
        /// `input`: thrust (forward, left, up), turn (roll, pitch, yaw),
        /// boost, brake, autopilot station (or -1), undock.
        pub fn frame(&mut self, wall_dt: f64, input: &[f32]) {
            let get = |i: usize| input.get(i).copied().unwrap_or(0.0) as f64;
            let inp = Input {
                thrust: [get(0), get(1), get(2)],
                turn: [get(3), get(4), get(5)],
                boost: get(6) > 0.5,
                brake: get(7) > 0.5,
                autopilot: input.get(8).map(|v| *v as i32).unwrap_or(-1),
                undock: get(9) > 0.5,
            };
            self.session.frame(wall_dt, &inp);
        }

        /// Per station: visible, ndc x, ndc y, on screen, angular radius (ndc
        /// units of the vertical half-height), g, relative flux, distance,
        /// light delay, emission time.
        pub fn stations(&self) -> Vec<f32> {
            self.session.stations().iter().flat_map(|s| s.to_array()).collect()
        }

        /// τ, t, r, dt/dτ, γ, v, clock rate, status (0 free, 1 autopilot,
        /// 2 docked), station, distance, relative speed, r₊, a, live bodies.
        pub fn telemetry(&self) -> Vec<f64> {
            let t = self.session.world.telemetry();
            vec![
                t.tau,
                t.t,
                t.r,
                t.dt_dtau,
                t.gamma,
                t.speed,
                t.clock_rate,
                t.status as f64,
                t.station as f64,
                t.station_distance,
                t.station_speed,
                t.r_plus,
                t.spin,
                t.alive_bodies as f64,
            ]
        }

        /// Headless engines: request the next frame's pixels.
        #[wasm_bindgen(js_name = requestCapture)]
        pub fn request_capture(&self) {
            self.session.gpu.request_capture();
        }

        /// Headless engines: `[width u32 LE, height u32 LE, ...RGBA8]` once a
        /// requested capture is ready.
        pub fn capture(&self) -> Option<Vec<u8>> {
            self.session.gpu.take_capture()
        }

        /// Events since the last frame as (code, argument) pairs: 1 horizon
        /// crossed, 2 docked, 3 undocked, 4 star captured.
        pub fn events(&self) -> Vec<i32> {
            self.session
                .events()
                .iter()
                .flat_map(|ev| match *ev {
                    WorldEvent::HorizonCrossed => [1, -1],
                    WorldEvent::Docked(i) => [2, i as i32],
                    WorldEvent::Undocked(i) => [3, i as i32],
                    WorldEvent::StarCaptured(i) => [4, i as i32],
                })
                .collect()
        }
    }
}
