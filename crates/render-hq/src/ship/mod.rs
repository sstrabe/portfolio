//! The pilot's ship: bind group 4 of the trace pass (see `wgsl/ship.wgsl`).
//!
//! The mesh (`mesh.rs`) and its BVH (`bvh.rs`) are built once and uploaded;
//! each frame the chase camera's pose (`camera.rs`) and the light falling on
//! the ship are computed here, in the ship's frame:
//! - the nearest system's star, E_λ = π B_λ(T) (R/d)², unless a planet
//!   hides it;
//! - that planet's sunlight reflected towards the ship: E_sun × albedo ×
//!   (R_p/d_p)² × the Lambert sphere's phase function;
//! - a faint skylight from the rest of the cluster.

pub mod bvh;
pub mod camera;
pub mod mesh;

use crate::FrameContext;
use crate::spectrum;
use bytemuck::{Pod, Zeroable};
use kerr::planets::KM_PER_M;
use kerr::vec3::{self, V3};
use wgpu::util::DeviceExt;

pub use camera::ChaseCamera;

/// Mirrors `struct ShipFrame` in `ship.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct ShipFrameGpu {
    cam_pos: [f32; 4],
    cam_x: [f32; 4],
    cam_y: [f32; 4],
    cam_z: [f32; 4],
    bound: [f32; 4],
    sun_dir: [f32; 4],
    planet_dir: [f32; 4],
    glow: [f32; 4],
    sun: [[f32; 4]; 4],
    planet: [[f32; 4]; 4],
    sky: [[f32; 4]; 4],
}

/// Irradiance from the rest of the sky per hemisphere, W m⁻² nm⁻¹: the
/// cluster's thousands of bright stars give roughly a full moon's light.
const SKY_IRRADIANCE: f64 = 2.0e-4;
/// Bond albedo assumed for sunlight reflected by a planet.
const PLANET_ALBEDO: f64 = 0.3;

pub struct Ship {
    layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    uniform: wgpu::Buffer,
    bound: (V3, f64),
    /// The camera: first person or chasing the ship.
    pub camera: ChaseCamera,
    /// Drive power 0–1 for the plasma and radiator glow (set by the session).
    pub power: f64,
}

fn spectrum_rows(s: [f64; spectrum::BINS]) -> [[f32; 4]; 4] {
    std::array::from_fn(|r| std::array::from_fn(|c| s[4 * r + c] as f32))
}

fn f4(v: V3, w: f64) -> [f32; 4] {
    [v[0] as f32, v[1] as f32, v[2] as f32, w as f32]
}

impl Ship {
    pub fn new(device: &wgpu::Device, _queue: &wgpu::Queue) -> Self {
        let (mesh, bvh) = mesh::ship();
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
            label: Some("ship"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage(1),
                storage(2),
                storage(3),
            ],
        });
        let init = |label, contents: &[u8], usage| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents, usage })
        };
        let uniform = init(
            "ship frame",
            bytemuck::bytes_of(&ShipFrameGpu::default()),
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let vertices = init("ship vertices", bytemuck::cast_slice(&mesh.vertices), wgpu::BufferUsages::STORAGE);
        let triangles = init("ship triangles", bytemuck::cast_slice(&mesh.triangles), wgpu::BufferUsages::STORAGE);
        let nodes = init("ship bvh", bytemuck::cast_slice(&bvh.gpu_nodes()), wgpu::BufferUsages::STORAGE);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ship"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: vertices.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: triangles.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: nodes.as_entire_binding() },
            ],
        });
        Self { layout, bind_group, uniform, bound: mesh.bounding_sphere(), camera: ChaseCamera::default(), power: 0.0 }
    }

    pub fn layout(&self) -> &wgpu::BindGroupLayout {
        &self.layout
    }

    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    pub fn update(&mut self, ctx: &FrameContext) {
        let world = ctx.world;
        let pilot = &world.pilot;
        let pose = self.camera.pose();
        let mut u = ShipFrameGpu {
            cam_pos: f4(pose.pos, if self.camera.chase { 1.0 } else { 0.0 }),
            cam_x: f4(pose.axes[0], 0.0),
            cam_y: f4(pose.axes[1], 0.0),
            cam_z: f4(pose.axes[2], 0.0),
            bound: f4(self.bound.0, self.bound.1),
            glow: [self.power as f32, ctx.wall_time as f32, 0.0, 0.0],
            sky: spectrum_rows(
                spectrum::blackbody(5500.0).map(|b| b / spectrum::planck(550.0, 5500.0) * SKY_IRRADIANCE),
            ),
            ..Default::default()
        };
        if self.camera.chase
            && let Some(sel) = ctx.near.systems.first()
        {
            let sys = &sel.system;
            let pos = pilot.position();
            let star = world.cluster.bodies[sys.star].position();
            // Ship-frame vectors (km) to the star and to the nearest planet.
            let local = |p: V3| vec3::scale(pilot.local_components(&world.kerr, vec3::sub(p, pos)), KM_PER_M);
            let to_star = local(star);
            let d_star = vec3::norm(to_star);
            let sun_dir = vec3::scale(to_star, 1.0 / d_star);
            let ratio = sys.star_radius_km / d_star;
            let sun = spectrum::blackbody(sys.star_temperature).map(|b| std::f64::consts::PI * b * ratio * ratio);
            let mut lit = true;
            let nearest = ctx
                .near
                .planets
                .iter()
                .filter(|p| p.system == 0)
                .min_by(|a, b| a.distance_km.total_cmp(&b.distance_km));
            if let Some(p) = nearest {
                let planet = p.planet(ctx.near);
                let (off_km, _) = sys.planet_state(p.index, world.cluster.t);
                let centre = local(vec3::add(star, vec3::scale(off_km, 1.0 / KM_PER_M)));
                let d = vec3::norm(centre);
                let r = planet.radius_km;
                // The planet hides the star when the sunward ray passes
                // within its radius (ahead of the ship).
                let along = vec3::dot(centre, sun_dir);
                if along > 0.0 && vec3::norm(vec3::axpy(centre, -along, sun_dir)) < r {
                    lit = false;
                }
                let dir = vec3::scale(centre, 1.0 / d);
                // Phase angle at the planet between the star and the ship.
                let from_planet_to_sun = vec3::normalize(vec3::sub(to_star, centre));
                let alpha = vec3::dot(from_planet_to_sun, vec3::scale(dir, -1.0)).clamp(-1.0, 1.0).acos();
                let phase = (alpha.sin() + (std::f64::consts::PI - alpha) * alpha.cos()) / std::f64::consts::PI;
                let k = PLANET_ALBEDO * (r / d.max(r)).powi(2) * phase;
                u.planet = spectrum_rows(sun.map(|e| e * k));
                u.planet_dir = f4(dir, if k > 0.0 { 1.0 } else { 0.0 });
            }
            u.sun = spectrum_rows(sun);
            u.sun_dir = f4(sun_dir, if lit { 1.0 } else { 0.0 });
        }
        ctx.queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&u));
    }

    pub fn encode(&mut self, _enc: &mut wgpu::CommandEncoder, _ctx: &FrameContext) {}
}

#[cfg(test)]
mod tests {
    #[test]
    fn layouts_match_wgsl() {
        assert_eq!(std::mem::size_of::<super::ShipFrameGpu>(), (8 + 12) * 16);
        assert_eq!(std::mem::size_of::<super::bvh::NodeGpu>(), 32);
        assert_eq!(std::mem::size_of::<super::mesh::Vertex>(), 32);
    }
}
