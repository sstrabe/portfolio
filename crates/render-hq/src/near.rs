//! Near field: the star systems close to the pilot (bind group 1 of the
//! trace pass, see `wgsl/near.wgsl`).
//!
//! Each frame the nearest systems are picked and expressed in their own
//! rest frames. The rest frame's axes are the pilot's (forward, left, up)
//! carried over by the pure boost between the two 4-velocities, so the
//! shader only needs the relative velocity β to boost a pixel's ray. The
//! star and planets are placed relative to the observer event in f64 and
//! only then rounded to f32, so they stay precise however far from the hole
//! the pilot is.

use bytemuck::{Pod, Zeroable};
use kerr::cluster::Body;
use kerr::geodesic;
use kerr::planets::{self, C_KM_S, KM_PER_M, System};
use kerr::units::{AU, SECONDS_PER_M};
use kerr::vec3::{self, V3, V4};
use kerr::world::World;

pub const MAX_SYSTEMS: usize = 2;
pub const MAX_PLANETS: usize = 16;
/// Systems whose star is farther than this (units of M) are left to the
/// far field: their planets are far below a pixel.
pub const SYSTEM_RANGE: f64 = 80.0 * AU;

/// Mirrors `struct StarSystem` in `near.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct SystemGpu {
    pub beta: [f32; 4],
    pub boost: [f32; 4],
    pub star: [f32; 4],
    pub light: [f32; 4],
    pub range: [u32; 4],
}

/// Mirrors `struct Planet` in `near.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct PlanetGpu {
    pub centre: [f32; 4],
    pub vel: [f32; 4],
    pub spin: [f32; 4],
    pub axis0: [f32; 4],
    pub surface: [f32; 4],
    pub rings: [f32; 4],
    pub ids: [u32; 4],
}

pub struct SelectedSystem {
    pub system: System,
    pub gpu: SystemGpu,
    /// Distance from the pilot to the star, km (coordinate estimate).
    pub distance_km: f64,
}

pub struct SelectedPlanet {
    /// Index into [`Selection::systems`].
    pub system: usize,
    /// Index into that system's `planets`.
    pub index: usize,
    pub gpu: PlanetGpu,
    /// Distance from the pilot to the planet's centre, km.
    pub distance_km: f64,
    /// Angular radius of the planet as seen from the pilot (rest frame).
    pub angular_radius: f64,
}

impl SelectedPlanet {
    pub fn planet<'a>(&self, sel: &'a Selection) -> &'a planets::Planet {
        &sel.systems[self.system].system.planets[self.index]
    }
}

/// What the near field holds this frame. GPU slot `i` of the planet
/// buffer is `planets[i]`.
#[derive(Default)]
pub struct Selection {
    pub systems: Vec<SelectedSystem>,
    pub planets: Vec<SelectedPlanet>,
}

/// Pick the systems near the pilot and express them in their rest frames.
/// `pixel_angle` (rad) decides whether a star's disc is drawn here.
pub fn select(world: &World, pixel_angle: f64) -> Selection {
    let k = &world.kerr;
    let pilot = &world.pilot;
    let pos = pilot.position();
    let e = &pilot.e;
    let t = world.cluster.t;
    let seed = world.cluster.cfg.seed;
    let mut sel = Selection::default();
    for (dist, system) in planets::nearby(seed, &world.cluster.bodies, pos, SYSTEM_RANGE).into_iter().take(MAX_SYSTEMS)
    {
        let body: &Body = &world.cluster.bodies[system.star];
        let u = geodesic::four_velocity(k, &body.state);
        let dot = |a: V4, b: V4| k.dot(pos, a, b);
        // Relative velocity of the star in the pilot's frame.
        let gamma = -dot(u, e[0]);
        let beta: V3 = std::array::from_fn(|i| dot(u, e[i + 1]) / gamma);
        let b = vec3::norm(beta);
        // Rest-frame axes: f_i = e_i + (u + e₀)(u·e_i)/(1 + γ).
        let f: [V4; 3] = std::array::from_fn(|i| {
            let c = dot(u, e[i + 1]) / (1.0 + gamma);
            std::array::from_fn(|m| e[i + 1][m] + c * (u[m] + e[0][m]))
        });
        // Rest-frame components of a coordinate displacement (spatial part).
        let rest = |v: V4| -> V3 { std::array::from_fn(|i| dot(f[i], v)) };
        let rest_time = |v: V4| -dot(u, v);

        let d_star: V4 = [0.0, body.position()[0] - pos[0], body.position()[1] - pos[1], body.position()[2] - pos[2]];
        let star_km = vec3::scale(rest(d_star), KM_PER_M);
        let star_dist = vec3::norm(star_km);
        let disc_drawn = system.star_radius_km / star_dist.max(1.0) > pixel_angle / 3.0;
        let first = sel.planets.len() as u32;
        let slot = sel.systems.len();

        for (i, p) in system.planets.iter().enumerate() {
            if sel.planets.len() >= MAX_PLANETS {
                break;
            }
            let (off_km, vel_km_s) = system.planet_state(i, t);
            let off: V4 = [0.0, off_km[0] / KM_PER_M, off_km[1] / KM_PER_M, off_km[2] / KM_PER_M];
            let event: V4 = std::array::from_fn(|m| d_star[m] + off[m]);
            let x_km = vec3::scale(rest(event), KM_PER_M);
            let t_km = rest_time(event) * KM_PER_M;
            let v_rest = vec3::scale(rest([0.0, vel_km_s[0], vel_km_s[1], vel_km_s[2]]), 1.0 / C_KM_S);
            // Centre at rest-frame time 0.
            let centre = vec3::axpy(x_km, -t_km, v_rest);
            let spin = vec3::normalize(rest([0.0, p.spin_axis[0], p.spin_axis[1], p.spin_axis[2]]));
            let a0c = vec3::any_orthogonal(p.spin_axis);
            let a0 = rest([0.0, a0c[0], a0c[1], a0c[2]]);
            let axis0 = vec3::normalize(vec3::axpy(a0, -vec3::dot(a0, spin), spin));
            let omega_s = std::f64::consts::TAU / p.rotation_period_s;
            // Rotation angle at the planet's event, then back to time 0.
            let angle = p.rotation(t * SECONDS_PER_M) - omega_s * t_km / C_KM_S;
            let atmo_top = p.atmosphere.map_or(0.0, |a| a.top_km);
            let (ri, ro, rt) = p.rings.map_or((0.0, 0.0, 0.0), |r| (r.inner_km, r.outer_km, r.optical_depth));
            let distance_km = vec3::norm(centre);
            sel.planets.push(SelectedPlanet {
                system: slot,
                index: i,
                distance_km,
                angular_radius: (p.radius_km / distance_km.max(p.radius_km)).asin(),
                gpu: PlanetGpu {
                    centre: f4(centre, p.radius_km),
                    vel: f4(v_rest, p.gm()),
                    spin: f4(spin, angle.rem_euclid(std::f64::consts::TAU)),
                    axis0: f4(axis0, omega_s / C_KM_S),
                    surface: [
                        p.relief_km as f32,
                        p.sea_level as f32,
                        p.equilibrium_temperature as f32,
                        atmo_top as f32,
                    ],
                    rings: [ri as f32, ro as f32, rt as f32, 0.0],
                    ids: [p.kind.index(), p.seed, slot as u32, sel.planets.len() as u32],
                },
            });
        }
        let count = sel.planets.len() as u32 - first;
        // 1 − |β| without cancellation: 1/(γ²(1 + |β|)).
        let one_minus_b = 1.0 / (gamma * gamma * (1.0 + b));
        let gpu = SystemGpu {
            beta: f4(if b > 0.0 { vec3::scale(beta, 1.0 / b) } else { [1.0, 0.0, 0.0] }, b),
            boost: [gamma as f32, one_minus_b as f32, 0.0, 0.0],
            star: f4(star_km, system.star_radius_km),
            light: [
                system.star_temperature as f32,
                system.star_luminosity_w as f32,
                system.star as f32,
                if disc_drawn { 1.0 } else { 0.0 },
            ],
            range: [first, count, 0, 0],
        };
        sel.systems.push(SelectedSystem { system, gpu, distance_km: dist * KM_PER_M });
    }
    sel
}

fn f4(v: V3, w: f64) -> [f32; 4] {
    [v[0] as f32, v[1] as f32, v[2] as f32, w as f32]
}

/// GPU side: storage buffers for the systems and planets.
pub struct NearField {
    layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    systems: wgpu::Buffer,
    planets: wgpu::Buffer,
    pub selection: Selection,
}

impl NearField {
    pub fn new(device: &wgpu::Device) -> Self {
        let storage = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("near field"),
            entries: &[storage(0), storage(1)],
        });
        let buffer = |label, size| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let systems = buffer("systems", (MAX_SYSTEMS * std::mem::size_of::<SystemGpu>()) as u64);
        let planets = buffer("planets", (MAX_PLANETS * std::mem::size_of::<PlanetGpu>()) as u64);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("near field"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: systems.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: planets.as_entire_binding() },
            ],
        });
        Self { layout, bind_group, systems, planets, selection: Selection::default() }
    }

    pub fn layout(&self) -> &wgpu::BindGroupLayout {
        &self.layout
    }

    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    pub fn update(&mut self, queue: &wgpu::Queue, world: &World, pixel_angle: f64) {
        self.selection = select(world, pixel_angle);
        let systems: Vec<SystemGpu> = self.selection.systems.iter().map(|s| s.gpu).collect();
        let planets: Vec<PlanetGpu> = self.selection.planets.iter().map(|p| p.gpu).collect();
        if !systems.is_empty() {
            queue.write_buffer(&self.systems, 0, bytemuck::cast_slice(&systems));
        }
        if !planets.is_empty() {
            queue.write_buffer(&self.planets, 0, bytemuck::cast_slice(&planets));
        }
    }

    /// (systems, planets) for `HqFrame.near`.
    pub fn counts(&self) -> [u32; 2] {
        [self.selection.systems.len() as u32, self.selection.planets.len() as u32]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kerr::world::WorldConfig;

    #[test]
    fn layouts_match_wgsl() {
        assert_eq!(std::mem::size_of::<SystemGpu>(), 5 * 16);
        assert_eq!(std::mem::size_of::<PlanetGpu>(), 7 * 16);
    }

    /// A pilot at rest next to a planet sees it where the coordinates put
    /// it, with no Doppler shift, and the rest-frame axes stay orthonormal
    /// when the pilot moves fast.
    #[test]
    fn rest_frame_placement() {
        let mut w = World::new(WorldConfig::sgr_a(300, 1));
        let seed = w.cluster.cfg.seed;
        let (star, system) = w
            .cluster
            .bodies
            .iter()
            .enumerate()
            .find_map(|(i, b)| planets::system(seed, i, b).map(|s| (i, s)))
            .expect("a system");
        let body = &w.cluster.bodies[star];
        let (off_km, _) = system.planet_state(0, w.cluster.t);
        let r = system.planets[0].radius_km;
        // Park 3 planet radii from planet 0, co-moving with the star.
        let away = vec3::scale(vec3::normalize(off_km), 3.0 * r);
        let at = vec3::add(body.position(), vec3::scale(vec3::add(off_km, away), 1.0 / KM_PER_M));
        let vel = geodesic::coordinate_velocity(&w.kerr, &body.state);
        let look = vec3::scale(away, -1.0);
        let mut pilot = kerr::pilot::Pilot::new(&w.kerr, at, vel, look, vec3::any_orthogonal(look)).unwrap();
        pilot.x[0] = w.pilot.x[0];
        w.pilot = pilot;
        let sel = select(&w, 1e-3);
        let p = sel.planets.iter().find(|p| p.gpu.ids[2] == 0 && p.index == 0).expect("planet 0 selected");
        let c = p.gpu.centre;
        // Straight ahead (forward axis), 3 radii away.
        let d = (c[0] as f64, c[1] as f64, c[2] as f64);
        assert!((d.0 / (3.0 * r) - 1.0).abs() < 1e-3, "{d:?}");
        assert!(d.1.abs() < 1e-3 * r && d.2.abs() < 1e-3 * r, "{d:?}");
        assert!(sel.systems[0].gpu.beta[3] < 1e-6);
        assert!(p.angular_radius > 0.3);
    }
}
