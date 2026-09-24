//! Browser entry point. The web layer creates an [`Engine`] for a canvas,
//! calls [`Engine::frame`] from `requestAnimationFrame` with the input
//! state, and reads back the stations' apparent screen positions to float
//! (or composite) the portfolio content over them.

#[cfg(target_arch = "wasm32")]
mod gpu;

pub mod frame;

#[cfg(target_arch = "wasm32")]
pub use web::*;

#[cfg(target_arch = "wasm32")]
mod web {
    use crate::frame::{self, StationScreen};
    use crate::gpu::{self, Gpu, PANEL_SLOTS};
    use kerr::cluster::ClusterConfig;
    use kerr::world::{Input, World, WorldConfig, WorldEvent, default_stations};
    use wasm_bindgen::prelude::*;

    /// Floats per station in [`Engine::stations`].
    pub const STATION_STRIDE: usize = StationScreen::STRIDE;

    #[wasm_bindgen]
    pub struct Engine {
        world: World,
        gpu: Gpu,
        wall_time: f64,
        fov_deg: f64,
        exposure: f64,
        highlight: i32,
        uploaded_newest: f64,
        generations: Vec<u32>,
        stations: Vec<f32>,
        events: Vec<i32>,
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
        console_error_panic_hook::set_once();
        let defaults = WorldConfig::default();
        let cfg = WorldConfig {
            stations: default_stations(station_count.clamp(1, gpu::ATLAS_COLS * gpu::ATLAS_ROWS) as usize),
            cluster: ClusterConfig {
                stars: stars as usize,
                seed: seed as u64 ^ 0x5EED_CAFE,
                ..defaults.cluster.clone()
            },
            ..defaults
        };
        let world = World::new(cfg);
        let n = world.cluster.len() as u32;
        let cap = world.cluster.history.capacity() as u32;
        let gpu = Gpu::new(canvas, n, cap).await.map_err(|e| JsValue::from_str(&e))?;
        let mut engine = Engine {
            generations: world.cluster.bodies.iter().map(|b| b.generation).collect(),
            stations: vec![0.0; world.station_count() * STATION_STRIDE],
            world,
            gpu,
            wall_time: 0.0,
            fov_deg: 75.0,
            exposure: 1.0,
            highlight: -1,
            uploaded_newest: f64::NEG_INFINITY,
            events: Vec::new(),
        };
        engine.upload_all_history();
        Ok(engine)
    }

    #[wasm_bindgen]
    impl Engine {
        /// Resize the swap chain (device pixels).
        pub fn resize(&mut self, width: u32, height: u32) {
            self.gpu.resize(width, height);
        }

        /// Fraction of the output resolution used for ray tracing.
        #[wasm_bindgen(js_name = setRenderScale)]
        pub fn set_render_scale(&mut self, scale: f32) {
            self.gpu.set_render_scale(scale);
        }

        #[wasm_bindgen(js_name = setFov)]
        pub fn set_fov(&mut self, degrees: f64) {
            self.fov_deg = degrees.clamp(30.0, 120.0);
        }

        #[wasm_bindgen(js_name = setExposure)]
        pub fn set_exposure(&mut self, exposure: f64) {
            self.exposure = exposure.clamp(0.05, 20.0);
        }

        /// Pilot proper time per wall-clock second (units of M).
        #[wasm_bindgen(js_name = setTimeScale)]
        pub fn set_time_scale(&mut self, scale: f64) {
            self.world.cfg.time_scale = scale.clamp(0.0, 200.0);
        }

        /// Station whose panel frame should glow (-1 for none).
        #[wasm_bindgen(js_name = setHighlight)]
        pub fn set_highlight(&mut self, station: i32) {
            self.highlight = station;
        }

        #[wasm_bindgen(js_name = stationCount)]
        pub fn station_count(&self) -> u32 {
            self.world.station_count() as u32
        }

        /// Copy the panel atlas canvas (4 × 2 cells of 512 × 320 px).
        #[wasm_bindgen(js_name = uploadPanelAtlas)]
        pub fn upload_panel_atlas(&self, canvas: web_sys::HtmlCanvasElement) {
            self.gpu.upload_atlas(canvas);
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
            self.wall_time += wall_dt;
            self.world.step(wall_dt, &inp);
            self.events.clear();
            for ev in &self.world.events {
                let (code, arg) = match *ev {
                    WorldEvent::HorizonCrossed => (1, -1),
                    WorldEvent::Docked(i) => (2, i as i32),
                    WorldEvent::Undocked(i) => (3, i as i32),
                    WorldEvent::StarCaptured(i) => (4, i as i32),
                };
                self.events.extend([code, arg]);
            }
            self.sync_history();
            self.upload_frame();
            self.gpu.render();
        }

        /// Per station: visible, ndc x, ndc y, on screen, angular radius (ndc
        /// units of the vertical half-height), g, relative flux, distance,
        /// light delay, emission time.
        pub fn stations(&self) -> Vec<f32> {
            self.stations.clone()
        }

        /// τ, t, r, dt/dτ, γ, v, clock rate, status (0 free, 1 autopilot,
        /// 2 docked), station, distance, relative speed, r₊, a, live bodies.
        pub fn telemetry(&self) -> Vec<f64> {
            let t = self.world.telemetry();
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

        /// Events since the last frame as (code, argument) pairs: 1 horizon
        /// crossed, 2 docked, 3 undocked, 4 star captured.
        pub fn events(&self) -> Vec<i32> {
            self.events.clone()
        }
    }

    impl Engine {
        fn upload_all_history(&mut self) {
            let h = &self.world.cluster.history;
            for slot in 0..h.capacity() {
                self.gpu.write_history_column(slot as u32, h.column(slot));
            }
            self.uploaded_newest = h.t_newest();
        }

        /// Mirror new history ticks (and whole columns after respawns).
        fn sync_history(&mut self) {
            let respawned = self.world.cluster.bodies.iter().zip(&self.generations).any(|(b, g)| b.generation != *g);
            let h = &self.world.cluster.history;
            let ticks = ((h.t_newest() - self.uploaded_newest) / h.dt()).round() as i64;
            if respawned || ticks >= h.capacity() as i64 {
                self.generations = self.world.cluster.bodies.iter().map(|b| b.generation).collect();
                self.upload_all_history();
            } else {
                for age in (0..ticks.max(0) as usize).rev() {
                    let slot = h.slot(age);
                    self.gpu.write_history_column(slot as u32, h.column(slot));
                }
                self.uploaded_newest = h.t_newest();
            }
            let cluster = &self.world.cluster;
            let now: Vec<_> = (0..cluster.len()).map(|i| cluster.current_sample(i)).collect();
            self.gpu.write_history_column(self.gpu.history_cap(), &now);
        }

        fn upload_frame(&mut self) {
            let (out_w, out_h) = self.gpu.output_size();
            let (hdr_w, hdr_h) = self.gpu.hdr_size();
            let params = frame::FrameParams {
                fov_deg: self.fov_deg,
                exposure: self.exposure,
                wall_time: self.wall_time,
                hdr: (hdr_w, hdr_h),
                out: (out_w, out_h),
                srgb_encode: self.gpu.srgb_encode(),
                panel_slots: PANEL_SLOTS as u32,
                highlight: self.highlight,
            };
            let built = frame::build(&self.world, &params);
            self.gpu.write_frame(&built.uniforms);
            self.gpu.write_meta(&built.meta);
            self.gpu.write_panels(&built.panels);
            self.stations = built.stations.iter().flat_map(|s| s.to_array()).collect();
        }
    }
}
