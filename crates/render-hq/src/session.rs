//! One running nucleus for the desktop: the physics world plus the HQ
//! renderer, advanced a frame at a time.

use crate::gpu::Gpu;
use crate::{HqUniforms, spectrum};
use kerr::planets::{C_KM_S, KM_PER_M, SUN_LUMINOSITY_W};
use kerr::units::{METRES_PER_M, SECONDS_PER_M};
use kerr::world::{Input, World, WorldEvent};
use render::frame;

/// W/m² per unit of the renderer's flux (L☉ / M², no 4π).
pub const W_PER_FLUX_UNIT: f64 = SUN_LUMINOSITY_W / (4.0 * std::f64::consts::PI * METRES_PER_M * METRES_PER_M);
/// Angular width of the point-spread core, rad (never below 0.6 px).
pub const PSF_SIGMA: f64 = 6.0e-4;
/// Display value (before the tone curve) of the core of the faintest
/// visible star.
const FAINT_STAR_PEAK: f64 = 0.06;
/// Display value a sunlit planet surface adapts to.
const DAYLIGHT_KEY: f64 = 0.2;

pub struct Session {
    pub world: World,
    pub gpu: Gpu,
    pub fov_deg: f64,
    /// Exposure multiplier on top of the automatic adaptation.
    pub exposure_bias: f64,
    wall_time: f64,
    /// Adapted faintest-visible point-source flux (L☉/M²).
    flux_ref: f64,
    /// Smoothed exposure (display value per unit radiance).
    exposure: f64,
    uploaded_newest: f64,
    generations: Vec<u32>,
}

impl Session {
    /// `gpu` must have been created for `world`'s body count and history
    /// capacity.
    pub fn new(world: World, gpu: Gpu) -> Self {
        let flux_ref = frame::adapted_flux_ref(&world);
        let mut s = Self {
            generations: world.cluster.bodies.iter().map(|b| b.generation).collect(),
            world,
            gpu,
            fov_deg: 75.0,
            exposure_bias: 1.0,
            wall_time: 0.0,
            flux_ref,
            exposure: 0.0,
            uploaded_newest: f64::NEG_INFINITY,
        };
        s.upload_all_history();
        s
    }

    /// Advance by `wall_dt` seconds of wall-clock time and render.
    pub fn frame(&mut self, wall_dt: f64, input: &Input) {
        self.wall_time += wall_dt;
        self.world.step(wall_dt, input);
        let target = frame::adapted_flux_ref(&self.world);
        let k = (wall_dt / 0.8).min(1.0);
        self.flux_ref = (self.flux_ref.ln() + (target.ln() - self.flux_ref.ln()) * k).exp();
        self.sync_history();

        let params = frame::FrameParams {
            fov_deg: self.fov_deg,
            exposure: 1.0,
            flux_ref: self.flux_ref,
            wall_time: self.wall_time,
            hdr: self.gpu.hdr_size(),
            out: self.gpu.output_size(),
            srgb_encode: self.gpu.srgb_encode(),
            panel_slots: 0,
            highlight: -1,
        };
        let built = frame::build(&self.world, &params);
        self.gpu.write_frame(&built.uniforms);
        self.gpu.write_meta(&built.meta);
        self.gpu.write_spheres(&built.spheres);
        let pixel_angle = built.uniforms.cam[2] as f64;
        self.gpu.update_near(&self.world, pixel_angle);

        let sigma = PSF_SIGMA.max(0.6 * pixel_angle);
        let target = self.target_exposure(sigma) * self.exposure_bias;
        self.exposure = if self.exposure > 0.0 {
            (self.exposure.ln() + (target.ln() - self.exposure.ln()) * k).exp()
        } else {
            target
        };
        let hq = HqUniforms {
            size: [0; 4],
            view: [0.0, 0.0, self.wall_time as f32, (pixel_angle * pixel_angle) as f32],
            radiometry: [W_PER_FLUX_UNIT as f32, self.sky_scale(sigma) as f32, self.exposure as f32, PSF_SIGMA as f32],
            near: [0; 4],
            units: [KM_PER_M as f32, SECONDS_PER_M as f32, C_KM_S as f32, 0.0],
            rgb: spectrum::rgb_weight_rows(),
        };
        self.gpu.render(&self.world, hq, self.wall_time);
    }

    /// What happened during the last frame.
    pub fn events(&self) -> &[WorldEvent] {
        &self.world.events
    }

    pub fn exposure(&self) -> f64 {
        self.exposure
    }

    /// Adapt to the stars (the faintest visible one shows as a dim point),
    /// or, when a sunlit planet fills much of the view, to daylight, which
    /// hides the stars as it does in photographs from orbit.
    fn target_exposure(&self, sigma: f64) -> f64 {
        let f_ref = self.flux_ref * W_PER_FLUX_UNIT;
        let stars = FAINT_STAR_PEAK * std::f64::consts::TAU * sigma * sigma / (f_ref * spectrum::y_per_watt(5800.0));
        let sel = &self.gpu.near.selection;
        let Some(p) = sel.planets.iter().max_by(|a, b| a.angular_radius.total_cmp(&b.angular_radius)) else {
            return stars;
        };
        let sys = &sel.systems[p.system].system;
        let flux = sys.flux_at(vec3_dist(sys, p, sel));
        let day_luma = flux * spectrum::y_per_watt(sys.star_temperature) * 0.3 / std::f64::consts::PI;
        let day = DAYLIGHT_KEY / day_luma.max(1e-30);
        let w = smooth(0.01, 0.15, p.angular_radius);
        (stars.ln() + (day.min(stars).ln() - stars.ln()) * w).exp()
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

/// Distance from a selected planet to its star, km.
fn vec3_dist(sys: &kerr::planets::System, p: &crate::near::SelectedPlanet, sel: &crate::near::Selection) -> f64 {
    let s = sel.systems[p.system].gpu.star;
    let c = p.gpu.centre;
    let d = [(s[0] - c[0]) as f64, (s[1] - c[1]) as f64, (s[2] - c[2]) as f64];
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(sys.star_radius_km)
}

fn smooth(lo: f64, hi: f64, x: f64) -> f64 {
    let t = ((x - lo) / (hi - lo)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}
