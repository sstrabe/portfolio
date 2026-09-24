//! Platform-independent translation of the world state into GPU uniforms,
//! body metadata, in-scene panels and the station screen positions handed
//! to the web layer. Kept free of wgpu so it can be unit tested natively.

use bytemuck::{Pod, Zeroable};
use kerr::cluster::BodyKind;
use kerr::lensing::Observer;
use kerr::pilot::UP;
use kerr::vec3::{self, V3};
use kerr::world::World;

pub const PANEL_SLOTS: usize = 8;
pub const ATLAS_COLS: u32 = 4;
pub const ATLAS_ROWS: u32 = 2;
pub const ATLAS_CELL_W: u32 = 512;
pub const ATLAS_CELL_H: u32 = 320;

/// Stations farther than this (in M) get no in-scene panel.
pub const PANEL_RANGE: f64 = 160.0;
pub const MAX_RAY_STEPS: u32 = 420;
pub const NEWTON_ITERATIONS: u32 = 2;
pub const SPRITE_GAIN: f32 = 2000.0;

/// Mirrors `struct Frame` in `common.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct FrameUniforms {
    pub obs: [f32; 4],
    pub e0: [f32; 4],
    pub e1: [f32; 4],
    pub e2: [f32; 4],
    pub e3: [f32; 4],
    pub cam: [f32; 4],
    pub view: [f32; 4],
    pub kerr: [f32; 4],
    pub march: [f32; 4],
    pub hist: [f32; 4],
    pub counts: [u32; 4],
    pub counts2: [u32; 4],
    pub screen: [f32; 4],
}

/// Mirrors `struct BodyMeta` in `common.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct BodyMeta {
    pub a: [f32; 4],
    pub b: [f32; 4],
}

/// Mirrors `struct Panel` in `sky.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct PanelUniform {
    pub center: [f32; 4],
    pub vel: [f32; 4],
    pub right: [f32; 4],
    pub up: [f32; 4],
    pub atlas: [f32; 4],
}

/// What the web layer needs to place content over a station.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StationScreen {
    pub visible: bool,
    pub ndc: [f32; 2],
    pub on_screen: bool,
    /// Angular radius in units of the vertical half field of view.
    pub radius: f32,
    pub g: f32,
    pub flux: f32,
    pub distance: f32,
    pub delay: f32,
    pub t_emit: f32,
}

impl StationScreen {
    pub const STRIDE: usize = 10;

    pub fn to_array(&self) -> [f32; Self::STRIDE] {
        [
            self.visible as u8 as f32,
            self.ndc[0],
            self.ndc[1],
            self.on_screen as u8 as f32,
            self.radius,
            self.g,
            self.flux,
            self.distance,
            self.delay,
            self.t_emit,
        ]
    }
}

pub struct FrameParams {
    pub fov_deg: f64,
    pub exposure: f64,
    pub wall_time: f64,
    pub hdr: (u32, u32),
    pub out: (u32, u32),
    pub srgb_encode: bool,
    pub panel_slots: u32,
    pub highlight: i32,
}

pub struct Built {
    pub uniforms: FrameUniforms,
    pub meta: Vec<BodyMeta>,
    pub panels: [PanelUniform; PANEL_SLOTS],
    pub stations: Vec<StationScreen>,
}

fn f4(v: [f64; 4]) -> [f32; 4] {
    v.map(|x| x as f32)
}

fn f3w(v: V3, w: f64) -> [f32; 4] {
    [v[0] as f32, v[1] as f32, v[2] as f32, w as f32]
}

/// Screen position of a ship-frame direction (forward, left, up).
pub fn project(dir: V3, tan_half: f64, aspect: f64) -> Option<[f64; 2]> {
    (dir[0] > 1e-6).then(|| [-dir[1] / (dir[0] * tan_half * aspect), dir[2] / (dir[0] * tan_half)])
}

pub fn build(world: &World, p: &FrameParams) -> Built {
    let cluster = &world.cluster;
    let h = &cluster.history;
    let t_obs = world.pilot.x[0];
    let e = &world.pilot.e;
    let tan_half = (p.fov_deg.to_radians() * 0.5).tan();
    let aspect = p.out.0 as f64 / p.out.1.max(1) as f64;
    let pos = world.pilot.position();

    let uniforms = FrameUniforms {
        obs: [0.0, pos[0] as f32, pos[1] as f32, pos[2] as f32],
        e0: f4(e[0]),
        e1: f4(e[1]),
        e2: f4(e[2]),
        e3: f4(e[3]),
        cam: [tan_half as f32, aspect as f32, (2.0 * tan_half / p.hdr.1.max(1) as f64) as f32, p.exposure as f32],
        view: [p.hdr.0 as f32, p.hdr.1 as f32, p.wall_time as f32, 0.0],
        kerr: [world.kerr.m as f32, world.kerr.a as f32, world.kerr.r_plus() as f32, 120.0],
        march: [0.05, 0.02, 40.0, world.ray.horizon_eps as f32],
        hist: [
            (h.t_newest() - t_obs) as f32,
            h.dt() as f32,
            (h.t_oldest() - t_obs) as f32,
            (h.t_oldest() - t_obs) as f32,
        ],
        counts: [cluster.len() as u32, h.capacity() as u32, h.newest_slot() as u32, h.len() as u32],
        counts2: [MAX_RAY_STEPS, p.panel_slots, NEWTON_ITERATIONS, world.station_count() as u32],
        screen: [p.out.0 as f32, p.out.1 as f32, if p.srgb_encode { 1.0 } else { 0.0 }, SPRITE_GAIN],
    };

    let meta = cluster
        .bodies
        .iter()
        .map(|b| {
            let kind = match b.params.kind {
                BodyKind::Station => 0.0,
                BodyKind::Compact => 1.0,
                BodyKind::Star => 2.0,
            };
            let died = if b.died_at.is_finite() { (b.died_at - t_obs) as f32 } else { 1.0e30 };
            BodyMeta {
                a: [(b.valid_from - t_obs) as f32, died, b.params.temperature as f32, b.params.luminosity as f32],
                b: [kind, 0.0, b.generation as f32, 0.0],
            }
        })
        .collect();

    let obs = world.observer();
    let stations: Vec<StationScreen> = world
        .views
        .iter()
        .map(|v| {
            let ndc = if v.visible { project(v.dir, tan_half, aspect) } else { None };
            StationScreen {
                visible: v.visible,
                ndc: ndc.map(|n| [n[0] as f32, n[1] as f32]).unwrap_or([0.0, 0.0]),
                on_screen: ndc.is_some_and(|n| n[0].abs() <= 1.05 && n[1].abs() <= 1.05),
                radius: (v.angular_radius.tan() / tan_half) as f32,
                g: v.g as f32,
                flux: v.flux as f32,
                distance: v.distance as f32,
                delay: v.delay as f32,
                t_emit: (v.t_emit - t_obs) as f32,
            }
        })
        .collect();

    Built { uniforms, meta, panels: panels(world, &obs, t_obs, p.highlight), stations }
}

/// In-scene panels for the nearest visible stations, each a card facing the
/// pilot at the station's emission event.
fn panels(world: &World, obs: &Observer, t_obs: f64, highlight: i32) -> [PanelUniform; PANEL_SLOTS] {
    let mut out = [PanelUniform::default(); PANEL_SLOTS];
    let mut order: Vec<usize> = (0..world.station_count())
        .filter(|&i| world.views[i].visible && world.views[i].distance < PANEL_RANGE)
        .collect();
    order.sort_by(|&a, &b| world.views[a].distance.total_cmp(&world.views[b].distance));
    let up_hint = vec3::spatial(obs.e[UP]);
    for (slot, &i) in order.iter().take(PANEL_SLOTS).enumerate() {
        let v = &world.views[i];
        let half_w = world.cfg.stations[i].size;
        let half_h = half_w * ATLAS_CELL_H as f64 / ATLAS_CELL_W as f64;
        let normal = vec3::normalize(vec3::sub(obs.position(), v.emit_pos));
        let mut right = vec3::cross(up_hint, normal);
        if vec3::norm(right) < 1e-6 {
            right = vec3::any_orthogonal(normal);
        }
        let right = vec3::normalize(right);
        let up = vec3::cross(normal, right);
        let col = i as u32 % ATLAS_COLS;
        let row = i as u32 / ATLAS_COLS;
        out[slot] = PanelUniform {
            center: f3w(v.emit_pos, v.t_emit - t_obs),
            vel: f3w(v.emit_vel, 1.0),
            right: f3w(vec3::scale(right, half_w), col as f64 / ATLAS_COLS as f64),
            up: f3w(vec3::scale(up, half_h), row as f64 / ATLAS_ROWS as f64),
            atlas: [
                1.0 / ATLAS_COLS as f32,
                1.0 / ATLAS_ROWS as f32,
                1.6,
                if highlight == i as i32 { 1.0 } else { 0.0 },
            ],
        };
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use kerr::cluster::ClusterConfig;
    use kerr::world::{Input, WorldConfig, default_stations};

    #[test]
    fn uniform_layout_matches_wgsl() {
        // 13 vec4s; WGSL uniform structs of vec4 members have no padding.
        assert_eq!(std::mem::size_of::<FrameUniforms>(), 13 * 16);
        assert_eq!(std::mem::size_of::<BodyMeta>(), 32);
        assert_eq!(std::mem::size_of::<PanelUniform>(), 80);
    }

    #[test]
    fn start_station_projects_near_the_centre_with_a_panel() {
        let cfg = WorldConfig {
            cluster: ClusterConfig { stars: 16, compact_objects: 1, history_len: 96, ..Default::default() },
            stations: default_stations(4),
            ..Default::default()
        };
        let mut w = World::new(cfg);
        w.step(1.0 / 60.0, &Input { autopilot: -1, ..Default::default() });
        let params = FrameParams {
            fov_deg: 75.0,
            exposure: 1.0,
            wall_time: 0.0,
            hdr: (640, 360),
            out: (1280, 720),
            srgb_encode: true,
            panel_slots: PANEL_SLOTS as u32,
            highlight: 0,
        };
        let b = build(&w, &params);
        let s = b.stations[0];
        assert!(s.visible && s.on_screen, "{s:?}");
        assert!(s.radius > 0.01);
        assert!(b.panels[0].vel[3] == 1.0);
        assert!(b.panels[0].atlas[3] == 1.0);
        assert_eq!(b.meta.len(), w.cluster.len());
        // Every live body's history reaches back to the oldest sample.
        assert!(b.uniforms.hist[2] < -100.0);
    }
}
