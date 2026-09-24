//! One running nucleus: the physics world plus the renderer, advanced a frame
//! at a time. Shared by the browser engine and the desktop app.

use crate::frame::{self, ATLAS_COLS, ATLAS_ROWS, PANEL_SLOTS, StationScreen};
use crate::gpu::Gpu;
use kerr::cluster::ClusterConfig;
use kerr::world::{Input, World, WorldConfig, WorldEvent, default_stations};

/// World settings for `stations` stations (at most 8; the desktop build
/// has none) and `stars` stars.
pub fn world_config(stations: u32, stars: u32, seed: u32) -> WorldConfig {
    let defaults = WorldConfig::default();
    WorldConfig {
        stations: default_stations(stations.min(ATLAS_COLS * ATLAS_ROWS) as usize),
        cluster: ClusterConfig { stars: stars as usize, seed: seed as u64 ^ 0x5EED_CAFE, ..defaults.cluster.clone() },
        ..defaults
    }
}

pub struct Session {
    pub world: World,
    pub gpu: Gpu,
    pub fov_deg: f64,
    pub exposure: f64,
    /// Station whose card frame glows (-1 for none).
    pub highlight: i32,
    wall_time: f64,
    uploaded_newest: f64,
    generations: Vec<u32>,
    stations: Vec<StationScreen>,
}

impl Session {
    /// `gpu` must have been created for `world`'s body count and history
    /// capacity.
    pub fn new(world: World, gpu: Gpu) -> Self {
        let mut s = Self {
            generations: world.cluster.bodies.iter().map(|b| b.generation).collect(),
            stations: Vec::new(),
            world,
            gpu,
            fov_deg: 75.0,
            exposure: 1.0,
            highlight: -1,
            wall_time: 0.0,
            uploaded_newest: f64::NEG_INFINITY,
        };
        s.upload_all_history();
        s
    }

    /// Advance by `wall_dt` seconds of wall-clock time and render.
    pub fn frame(&mut self, wall_dt: f64, input: &Input) {
        self.wall_time += wall_dt;
        self.world.step(wall_dt, input);
        self.sync_history();
        self.upload_frame();
        self.gpu.render();
    }

    /// Where each station appears on screen this frame.
    pub fn stations(&self) -> &[StationScreen] {
        &self.stations
    }

    /// What happened during the last frame.
    pub fn events(&self) -> &[WorldEvent] {
        &self.world.events
    }

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
        let params = frame::FrameParams {
            fov_deg: self.fov_deg,
            exposure: self.exposure,
            wall_time: self.wall_time,
            hdr: self.gpu.hdr_size(),
            out: self.gpu.output_size(),
            srgb_encode: self.gpu.srgb_encode(),
            panel_slots: PANEL_SLOTS as u32,
            highlight: self.highlight,
        };
        let built = frame::build(&self.world, &params);
        self.gpu.write_frame(&built.uniforms);
        self.gpu.write_meta(&built.meta);
        self.gpu.write_panels(&built.panels);
        self.stations = built.stations;
    }
}
