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
//!
//! The reaction-control thrusters fire for the rotation and translation the
//! pilot commands ([`Ship::set_rcs`]); the shader draws their plumes.

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
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
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
    /// Number of firing nozzles, unused ×3.
    rcs: [f32; 4],
    /// Per firing nozzle: exit (m) and strength 0–1, exhaust direction and
    /// the nozzle's index.
    jets: [[f32; 4]; 2 * MAX_JETS],
}

/// Irradiance from the rest of the sky per hemisphere, W m⁻² nm⁻¹: the
/// cluster's thousands of bright stars give roughly a full moon's light.
const SKY_IRRADIANCE: f64 = 2.0e-4;
/// Bond albedo assumed for sunlight reflected by a planet.
const PLANET_ALBEDO: f64 = 0.3;
/// Most nozzles drawn firing at once (all of them).
const MAX_JETS: usize = 32;
/// A plume takes this long to clear after its valve closes, s.
const JET_FADE_S: f64 = 0.08;
/// Nozzles firing at less than this share of full thrust stay closed.
const JET_THRESHOLD: f64 = 0.2;
/// The ship's centre of mass (ship frame, m): the full propellant tank.
const CENTRE_OF_MASS: V3 = [-2.5, 0.0, 0.0];

pub struct Ship {
    layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    uniform: wgpu::Buffer,
    bound: (V3, f64),
    /// The camera: first person or chasing the ship.
    pub camera: ChaseCamera,
    /// Drive power 0–1 for the plasma and radiator glow (set by the session).
    pub power: f64,
    nozzles: Vec<mesh::Nozzle>,
    /// Unit torque of each nozzle about the centre of mass, or zero for one
    /// that pushes through it.
    torques: Vec<V3>,
    /// Commanded and drawn strength of each nozzle, 0–1.
    wanted: Vec<f64>,
    firing: Vec<f64>,
    last_wall: Option<f64>,
    /// The mesh in the ray-tracing hardware's acceleration structures, kept
    /// alive while the bind group uses them (with ray queries only).
    _accel: Option<(wgpu::Blas, wgpu::Tlas)>,
}

fn spectrum_rows(s: [f64; spectrum::BINS]) -> [[f32; 4]; 4] {
    std::array::from_fn(|r| std::array::from_fn(|c| s[4 * r + c] as f32))
}

fn f4(v: V3, w: f64) -> [f32; 4] {
    [v[0] as f32, v[1] as f32, v[2] as f32, w as f32]
}

impl Ship {
    /// With `rt`, the mesh also goes into a BLAS in a one-instance TLAS
    /// for hardware ray queries (`ship_rq.wgsl`).
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, rt: bool) -> Self {
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
        let mut entries = vec![
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
        ];
        if rt {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::AccelerationStructure { vertex_return: false },
                count: None,
            });
        }
        let layout = device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("ship"), entries: &entries });
        let init = |label, contents: &[u8], usage| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents, usage })
        };
        let uniform = init(
            "ship frame",
            bytemuck::bytes_of(&ShipFrameGpu::zeroed()),
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let mesh_usage =
            if rt { wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::BLAS_INPUT } else { wgpu::BufferUsages::STORAGE };
        let vertices = init("ship vertices", bytemuck::cast_slice(&mesh.vertices), mesh_usage);
        let triangles = init("ship triangles", bytemuck::cast_slice(&mesh.triangles), mesh_usage);
        let nodes = init("ship bvh", bytemuck::cast_slice(&bvh.gpu_nodes()), wgpu::BufferUsages::STORAGE);
        let accel = rt.then(|| build_acceleration(device, queue, &vertices, &triangles, &mesh));
        let mut entries = vec![
            wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: vertices.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: triangles.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: nodes.as_entire_binding() },
        ];
        if let Some((_, tlas)) = &accel {
            entries.push(wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::AccelerationStructure(tlas),
            });
        }
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ship"),
            layout: &layout,
            entries: &entries,
        });
        let nozzles = mesh::rcs_nozzles();
        let torques = unit_torques(&nozzles);
        let n = nozzles.len();
        Self {
            layout,
            bind_group,
            uniform,
            bound: mesh.bounding_sphere(),
            camera: ChaseCamera::default(),
            power: 0.0,
            nozzles,
            torques,
            wanted: vec![0.0; n],
            firing: vec![0.0; n],
            last_wall: None,
            _accel: accel,
        }
    }

    /// Fire the reaction control for this frame: `torque` about the ship's
    /// (forward, left, up) axes and `force` along them, each component a
    /// share −1…1 of what the thrusters can give. A nozzle fires as far as
    /// its own torque and thrust point the wanted way; opposite nozzles
    /// pair up, so their pushes (or twists) cancel.
    pub fn set_rcs(&mut self, torque: V3, force: V3) {
        self.wanted = mix(&self.nozzles, &self.torques, torque, force);
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
            ..Zeroable::zeroed()
        };
        // Plumes linger a moment after the valves close.
        let dt = self.last_wall.map_or(0.0, |w| (ctx.wall_time - w).clamp(0.0, 0.25));
        self.last_wall = Some(ctx.wall_time);
        let decay = (-dt / JET_FADE_S).exp();
        let mut count = 0;
        for (i, nz) in self.nozzles.iter().enumerate() {
            self.firing[i] = self.wanted[i].max(self.firing[i] * decay);
            if self.firing[i] > 0.02 && count < MAX_JETS {
                u.jets[2 * count] = f4(nz.exit, self.firing[i]);
                u.jets[2 * count + 1] = f4(nz.dir, i as f64);
                count += 1;
            }
        }
        u.rcs = [count as f32, 0.0, 0.0, 0.0];
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

/// The ship's mesh as a BLAS (positions are the first 12 bytes of each
/// 32-byte vertex) in a TLAS with one instance at the identity, built now.
fn build_acceleration(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    vertices: &wgpu::Buffer,
    triangles: &wgpu::Buffer,
    mesh: &mesh::Mesh,
) -> (wgpu::Blas, wgpu::Tlas) {
    let size = wgpu::BlasTriangleGeometrySizeDescriptor {
        vertex_format: wgpu::VertexFormat::Float32x3,
        vertex_count: mesh.vertices.len() as u32,
        index_format: Some(wgpu::IndexFormat::Uint32),
        index_count: Some(3 * mesh.triangles.len() as u32),
        flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
    };
    let blas = device.create_blas(
        &wgpu::CreateBlasDescriptor {
            label: Some("ship"),
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        },
        wgpu::BlasGeometrySizeDescriptors::Triangles { descriptors: vec![size.clone()] },
    );
    let mut tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
        label: Some("ship"),
        max_instances: 1,
        flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
        update_mode: wgpu::AccelerationStructureUpdateMode::Build,
    });
    let identity = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    tlas[0] = Some(wgpu::TlasInstance::new(&blas, identity, 0, 0xff));
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("ship acceleration") });
    enc.build_acceleration_structures(
        [&wgpu::BlasBuildEntry {
            blas: &blas,
            geometry: wgpu::BlasGeometries::TriangleGeometries(vec![wgpu::BlasTriangleGeometry {
                size: &size,
                vertex_buffer: vertices,
                first_vertex: 0,
                vertex_stride: std::mem::size_of::<mesh::Vertex>() as u64,
                index_buffer: Some(triangles),
                first_index: Some(0),
                transform_buffer: None,
                transform_buffer_offset: None,
            }]),
        }],
        [&tlas],
    );
    queue.submit([enc.finish()]);
    (blas, tlas)
}

/// Direction of each nozzle's torque about the centre of mass, or zero for
/// one that pushes (nearly) through it.
fn unit_torques(nozzles: &[mesh::Nozzle]) -> Vec<V3> {
    let arms: Vec<V3> =
        nozzles.iter().map(|nz| vec3::cross(vec3::sub(nz.exit, CENTRE_OF_MASS), vec3::scale(nz.dir, -1.0))).collect();
    let longest = arms.iter().map(|t| vec3::norm(*t)).fold(0.0, f64::max);
    arms.iter().map(|&t| if vec3::norm(t) > 0.1 * longest { vec3::normalize(t) } else { [0.0; 3] }).collect()
}

/// Strength 0–1 of each nozzle for a wanted torque and force (see
/// [`Ship::set_rcs`]).
fn mix(nozzles: &[mesh::Nozzle], torques: &[V3], torque: V3, force: V3) -> Vec<f64> {
    nozzles
        .iter()
        .zip(torques)
        .map(|(nz, t)| {
            let s = vec3::dot(*t, torque) - vec3::dot(nz.dir, force);
            if s > JET_THRESHOLD { s.min(1.0) } else { 0.0 }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Net force and torque (about the centre of mass) of the nozzles
    /// firing at `strength`, per unit thrust.
    fn net(nozzles: &[mesh::Nozzle], strength: &[f64]) -> (V3, V3) {
        nozzles.iter().zip(strength).fold(([0.0; 3], [0.0; 3]), |(f, t), (nz, &s)| {
            let push = vec3::scale(nz.dir, -s);
            (vec3::add(f, push), vec3::add(t, vec3::cross(vec3::sub(nz.exit, CENTRE_OF_MASS), push)))
        })
    }

    /// A turn about one axis twists about that axis with little push or
    /// twist elsewhere; a push along one axis pushes that way without
    /// turning the ship much.
    #[test]
    fn rcs_mixing_turns_and_pushes_the_right_way() {
        let nozzles = mesh::rcs_nozzles();
        let torques = unit_torques(&nozzles);
        for axis in 0..3 {
            let mut want = [0.0; 3];
            want[axis] = 1.0;
            let (f, t) = net(&nozzles, &mix(&nozzles, &torques, want, [0.0; 3]));
            eprintln!("turn {axis}: force {f:?}, torque {t:?}");
            let off = (0..3).filter(|&k| k != axis).map(|k| t[k].abs()).fold(0.0, f64::max);
            assert!(t[axis] > 0.0 && off < 0.1 * t[axis], "turn {axis}: torque {t:?}");
            assert!(vec3::norm(f) < 0.05 * t[axis], "turn {axis}: force {f:?} vs torque {t:?}");

            let (f, t) = net(&nozzles, &mix(&nozzles, &torques, [0.0; 3], want));
            eprintln!("push {axis}: force {f:?}, torque {t:?}");
            let off = (0..3).filter(|&k| k != axis).map(|k| f[k].abs()).fold(0.0, f64::max);
            assert!(f[axis] > 0.0 && off < 0.1 * f[axis], "push {axis}: force {f:?}");
            // Lever arms are metres: compare the twist with the push times
            // the ship's half-length.
            assert!(vec3::norm(t) < 0.5 * 26.0 * f[axis], "push {axis}: torque {t:?} vs force {f:?}");
        }
    }

    #[test]
    fn layouts_match_wgsl() {
        assert_eq!(std::mem::size_of::<super::ShipFrameGpu>(), (8 + 12 + 1 + 2 * super::MAX_JETS) * 16);
        assert_eq!(std::mem::size_of::<super::bvh::NodeGpu>(), 32);
        assert_eq!(std::mem::size_of::<super::mesh::Vertex>(), 32);
    }
}
