//! One running nucleus for the desktop: the physics world plus the HQ
//! renderer, advanced a frame at a time.

use crate::gpu::Gpu;
use crate::{HQ_NARROWBAND, HqUniforms, spectrum};
use kerr::planets::{C_KM_S, KM_PER_M, SUN_LUMINOSITY_W};
use kerr::units::{METRES_PER_M, SECONDS_PER_M};
use kerr::vec3::{self, V3};
use kerr::world::{Input, World, WorldEvent};
use render::frame;

/// W/m² per unit of the renderer's flux (L☉ / M², no 4π).
pub const W_PER_FLUX_UNIT: f64 = SUN_LUMINOSITY_W / (4.0 * std::f64::consts::PI * METRES_PER_M * METRES_PER_M);
/// Angular width of the point-spread core the procedural sky was
/// calibrated with, rad.
pub const PSF_SIGMA: f64 = 6.0e-4;
/// Display value of the core of the faintest visible star in that
/// calibration (`sky_scale`).
const FAINT_STAR_PEAK: f64 = 0.06;

pub struct Session {
    pub world: World,
    pub gpu: Gpu,
    pub fov_deg: f64,
    wall_time: f64,
    uploaded_newest: f64,
    generations: Vec<u32>,
}

impl Session {
    /// `gpu` must have been created for `world`'s body count and history
    /// capacity.
    pub fn new(world: World, gpu: Gpu) -> Self {
        let mut s = Self {
            generations: world.cluster.bodies.iter().map(|b| b.generation).collect(),
            world,
            gpu,
            fov_deg: 75.0,
            wall_time: 0.0,
            uploaded_newest: f64::NEG_INFINITY,
        };
        s.upload_all_history();
        s
    }

    /// Advance by `wall_dt` seconds of wall-clock time and render, the
    /// drive glowing with the commanded thrust.
    pub fn frame(&mut self, wall_dt: f64, input: &Input) {
        self.advance(wall_dt, input);
        let thrust = input.thrust.iter().map(|x| x * x).sum::<f64>().sqrt().min(1.0);
        self.gpu.ship.power = thrust * if input.boost { 1.0 } else { 0.4 };
        self.render();
    }

    /// Advance the world by `wall_dt` seconds of wall-clock time.
    pub fn advance(&mut self, wall_dt: f64, input: &Input) {
        self.wall_time += wall_dt;
        // The terrain the renderer has read back is the ground the physics
        // lands and stands on.
        let ground = self.gpu.near.terrain.ground.snapshot();
        self.world.set_ground(ground.map(|g| Box::new(g) as Box<dyn kerr::world::Ground>));
        self.world.step(wall_dt, input);
        self.sync_history();
    }

    /// Render the world as it is now (with whatever the overlay holds).
    pub fn render(&mut self) {
        let params = frame::FrameParams {
            fov_deg: self.fov_deg,
            exposure: 1.0,
            // Exposure is metered on the GPU (`post.rs`); the web's
            // flux-based adaptation is not used here.
            flux_ref: self.world.cfg.star_flux_ref,
            wall_time: self.wall_time,
            hdr: self.gpu.hdr_size(),
            out: self.gpu.output_size(),
            srgb_encode: self.gpu.srgb_encode(),
            panel_slots: 0,
            highlight: -1,
        };
        let mut built = frame::build(&self.world, &params);
        // The chase camera turns the view (its offset of tens of metres
        // matters only for the ship itself, see `ship/camera.rs`).
        let view = self.gpu.ship.camera.view_tetrad(&self.world.pilot.e);
        let e = |v: [f64; 4]| v.map(|x| x as f32);
        (built.uniforms.e1, built.uniforms.e2, built.uniforms.e3) = (e(view[1]), e(view[2]), e(view[3]));
        self.gpu.write_frame(&built.uniforms);
        crate::lens::mark_shadows(&self.world, &mut built.meta);
        self.gpu.write_meta(&built.meta);
        self.gpu.write_spheres(&built.spheres);
        let pixel_angle = built.uniforms.cam[2] as f64;
        self.gpu.update_near(&self.world, &view, pixel_angle);

        let sigma = PSF_SIGMA.max(0.6 * pixel_angle);
        let jitter = self.gpu.post.jitter();
        let mut flags = if self.gpu.post.wants_narrowband() { HQ_NARROWBAND } else { 0 };
        if std::env::var_os("KERR_DEBUG_NAN").is_some() {
            flags |= crate::HQ_DEBUG_NAN;
        }
        let hq = HqUniforms {
            size: [0, 0, 0, flags],
            view: [jitter[0], jitter[1], self.wall_time as f32, (pixel_angle * pixel_angle) as f32],
            radiometry: [
                W_PER_FLUX_UNIT as f32,
                self.sky_scale(sigma) as f32,
                self.exposure() as f32,
                self.gpu.post.settings.optics.core_sigma() as f32,
            ],
            near: [0; 4],
            units: [KM_PER_M as f32, SECONDS_PER_M as f32, C_KM_S as f32, 0.0],
            rgb: spectrum::rgb_weight_rows(),
        };
        self.gpu.render(&self.world, &view, hq, self.wall_time);
    }

    /// How ship-frame directions map to the output image this frame.
    pub fn view(&self) -> View {
        View {
            axes: self.gpu.ship.camera.pose().axes,
            tan_half: (self.fov_deg.to_radians() * 0.5).tan(),
            size: self.gpu.output_size(),
        }
    }

    /// What happened during the last frame.
    pub fn events(&self) -> &[WorldEvent] {
        &self.world.events
    }

    /// The exposure the GPU adapted to (display value per unit radiance, a
    /// frame or two old), or before the first read-back the dark-adapted
    /// one.
    pub fn exposure(&self) -> f64 {
        match self.gpu.post.exposure().exposure {
            e if e > 0.0 => e,
            _ => self.gpu.post.dark_exposure(&self.world, self.pixel_angle()),
        }
    }

    /// What the GPU's meter saw last.
    pub fn metering(&self) -> crate::post::Metering {
        self.gpu.post.exposure()
    }

    /// Angle of an output pixel, rad.
    fn pixel_angle(&self) -> f64 {
        let tan_half = (self.fov_deg.to_radians() * 0.5).tan();
        2.0 * tan_half / self.gpu.output_size().1.max(1) as f64
    }

    /// Scale from the procedural sky's units to radiance, chosen so the
    /// galaxy looks as it did in the web renderer at the dark-adapted
    /// exposure.
    fn sky_scale(&self, sigma: f64) -> f64 {
        let dark = self.world.cfg.star_flux_ref * W_PER_FLUX_UNIT;
        0.85 * dark * spectrum::y_per_watt(5800.0)
            / (FAINT_STAR_PEAK * std::f64::consts::TAU * sigma * sigma * spectrum::y_per_watt(4800.0))
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
}

/// The view's orientation and field, for placing things over the image.
#[derive(Clone, Copy, Debug)]
pub struct View {
    /// The view's axes (forward, left, up) in the ship frame.
    pub axes: [V3; 3],
    pub tan_half: f64,
    /// Output size, pixels.
    pub size: (u32, u32),
}

impl View {
    /// Components along the view's axes of a ship-frame direction: the
    /// same in first person, turned for the chase camera.
    pub fn to_view(&self, dir: V3) -> V3 {
        self.axes.map(|a| vec3::dot(dir, a))
    }

    /// Output pixel at which a ship-frame direction appears, or `None`
    /// when it is behind the view.
    pub fn project(&self, dir: V3) -> Option<[f32; 2]> {
        let (w, h) = (self.size.0 as f64, self.size.1.max(1) as f64);
        let [x, y] = frame::project(self.to_view(dir), self.tan_half, w / h)?;
        Some([((x + 1.0) * 0.5 * w) as f32, ((1.0 - y) * 0.5 * h) as f32])
    }
}
