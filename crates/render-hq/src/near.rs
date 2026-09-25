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
//!
//! Planets of the system the physics is flying in are placed with the
//! physics' own functions ([`kerr::local::Local::planet_relative`], and for
//! a landed ship its body-fixed offset), so the ground is drawn exactly
//! where the ship stands on it.

use bytemuck::{Pod, Zeroable};
use kerr::cluster::Body;
use kerr::geodesic;
use kerr::local::PlanetRef;
use kerr::planets::{self, C_KM_S, KM_PER_M, System};
use kerr::units::{AU, SECONDS_PER_M};
use kerr::vec3::{self, V3, V4};
use kerr::world::World;
use wgpu::util::DeviceExt;

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
    pub detail: [f32; 4],
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
    /// The centre relative to the pilot (km, f64; the GPU copy is rounded
    /// to f32): in the planet's own rest frame for the system the physics
    /// is flying in, else in the system's.
    pub centre_km: V3,
    /// Distance from the pilot to the planet's centre, km.
    pub distance_km: f64,
    /// Angular radius of the planet as seen from the pilot (rest frame).
    pub angular_radius: f64,
    /// The planet's body-fixed axes now, in the same axes as `centre_km`
    /// (as `planet_body` in `near.wgsl` builds them, in f64).
    pub body_axes: [V3; 3],
}

impl SelectedPlanet {
    /// Where the pilot is in body-fixed coordinates: km from the centre.
    pub fn pilot_body_km(&self) -> V3 {
        crate::terrain::to_body(&self.body_axes, vec3::scale(self.centre_km, -1.0))
    }
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
/// `e` is the view's tetrad (the pilot's, or turned to a chase camera).
pub fn select(world: &World, e: &kerr::pilot::Tetrad, pixel_angle: f64) -> Selection {
    let k = &world.kerr;
    let pilot = &world.pilot;
    let pos = pilot.position();
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
        // The physics' snapshot of this system, if it is flying in it.
        let local = world.system_of(PlanetRef { star: system.star, generation: body.generation, planet: 0 });
        let fix = world.surface_fix();

        for (i, p) in system.planets.iter().enumerate() {
            if sel.planets.len() >= MAX_PLANETS {
                break;
            }
            let omega_s = std::f64::consts::TAU / p.rotation_period_s;
            let a0c = vec3::any_orthogonal(p.spin_axis);
            let spatial = |v: V3| -> V4 { [0.0, v[0], v[1], v[2]] };
            // The planet event (relative to the pilot), its rest-frame time,
            // its velocity, its spin axis and body x axis, and its rotation
            // angle at that event.
            let (x_km, t_km, v_rest, spin_v, a0, angle) = match local {
                Some(l) => {
                    // As the physics measures it, in the planet's own rest
                    // frame (the shader's planet-centred coordinates); a
                    // landed ship's place comes from its body-fixed offset.
                    let (pos, centre_event, frame) = match fix {
                        Some((fp, pt)) if fp.star == system.star && fp.planet == i => (pt.pos, pt.centre, pt.frame),
                        _ => {
                            let r = l.planet_relative(i, pilot.x);
                            (r.pos, r.centre, r.frame)
                        }
                    };
                    // The planet frame's axes: the system frame's, boosted by
                    // the planet's velocity (no rotation), so the shader's
                    // first-order ray transform lines up with them.
                    let u_p = frame.e[0];
                    let gamma_p = rest_time(u_p);
                    let h: [V4; 3] = std::array::from_fn(|j| {
                        let c = dot(u_p, f[j]) / (1.0 + gamma_p);
                        std::array::from_fn(|m| f[j][m] + c * (u_p[m] + u[m]))
                    });
                    let planet_axes = |v: V4| -> V3 { std::array::from_fn(|j| dot(h[j], v)) };
                    let to_centre = frame.displacement(0.0, vec3::scale(pos, -1.0));
                    (
                        vec3::scale(planet_axes(to_centre), KM_PER_M),
                        // Simultaneous with the pilot in the planet's frame.
                        0.0,
                        vec3::scale(rest(u_p), 1.0 / gamma_p),
                        planet_axes(frame.displacement(0.0, p.spin_axis)),
                        planet_axes(frame.displacement(0.0, a0c)),
                        p.rotation(centre_event[0] * SECONDS_PER_M),
                    )
                }
                None => {
                    let (off_km, vel_km_s) = system.planet_state(i, t);
                    let off: V4 = [0.0, off_km[0] / KM_PER_M, off_km[1] / KM_PER_M, off_km[2] / KM_PER_M];
                    let event: V4 = std::array::from_fn(|m| d_star[m] + off[m]);
                    let t_km = rest_time(event) * KM_PER_M;
                    (
                        vec3::scale(rest(event), KM_PER_M),
                        t_km,
                        vec3::scale(rest(spatial(vel_km_s)), 1.0 / C_KM_S),
                        rest(spatial(p.spin_axis)),
                        rest(spatial(a0c)),
                        // Rotation angle at the planet's event.
                        p.rotation(t * SECONDS_PER_M) - omega_s * t_km / C_KM_S,
                    )
                }
            };
            // Centre at rest-frame time 0.
            let centre = vec3::axpy(x_km, -t_km, v_rest);
            let spin = vec3::normalize(spin_v);
            let axis0 = vec3::normalize(vec3::axpy(a0, -vec3::dot(a0, spin), spin));
            let atmo_top = p.atmosphere.map_or(0.0, |a| a.top_km);
            let (ri, ro, rt, rd) =
                p.rings.map_or((0.0, 0.0, 0.0, 0.0), |r| (r.inner_km, r.outer_km, r.optical_depth, r.dust));
            let distance_km = vec3::norm(centre);
            let (sin, cos) = angle.sin_cos();
            let e1 = vec3::add(vec3::scale(axis0, cos), vec3::scale(vec3::cross(spin, axis0), sin));
            sel.planets.push(SelectedPlanet {
                body_axes: [e1, vec3::cross(spin, e1), spin],
                system: slot,
                index: i,
                centre_km: centre,
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
                    rings: [ri as f32, ro as f32, rt as f32, rd as f32],
                    ids: [p.kind.index(), p.seed, slot as u32, sel.planets.len() as u32],
                    detail: [p.wind_speed_m_s as f32, 0.0, 0.0, 0.0],
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
    /// Baked maps (climate) of the nearest solid world with air.
    pub maps: crate::terrain::maps::SurfaceMaps,
    /// Terrain tiles around the pilot on that world.
    pub terrain: crate::terrain::field::TerrainField,
    /// Where the tiles are relative to the pilot (`terrain_rq.wgsl`).
    terrain_view: wgpu::Buffer,
}

/// Mirrors `struct TerrainView` in `terrain_rq.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct TerrainViewGpu {
    eye: [f32; 4],
    up: [f32; 4],
    anchor: [f32; 4],
    ids: [u32; 4],
}

impl TerrainViewGpu {
    /// No planet has tiles in the trace.
    fn none() -> Self {
        Self { eye: [0.0; 4], up: [0.0; 4], anchor: [0.0; 4], ids: [u32::MAX, 0, 0, 0] }
    }
}

/// Maps are baked for a solid world with air once the pilot is within this
/// many of its radii.
const MAPS_RANGE_RADII: f64 = 50.0;

impl NearField {
    /// `rt`: the terrain gets BLASes for hardware ray queries.
    pub fn new(device: &wgpu::Device, rt: bool) -> Self {
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
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty,
            count: None,
        };
        let array = |sample_type| wgpu::BindingType::Texture {
            sample_type,
            view_dimension: wgpu::TextureViewDimension::D2Array,
            multisampled: false,
        };
        // With ray queries, the terrain tiles (`terrain_rq.wgsl`).
        let rt_entries: Vec<wgpu::BindGroupLayoutEntry> = if rt {
            vec![
                entry(5, array(wgpu::TextureSampleType::Float { filterable: false })),
                entry(6, array(wgpu::TextureSampleType::Uint)),
                entry(7, wgpu::BindingType::AccelerationStructure { vertex_return: false }),
                entry(
                    8,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
            ]
        } else {
            Vec::new()
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("near field"),
            entries: &[
                storage(0),
                storage(1),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                entry(
                    4,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
            ]
            .into_iter()
            .chain(rt_entries)
            .collect::<Vec<_>>(),
        });
        let maps = crate::terrain::maps::SurfaceMaps::new(device);
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
        let terrain_view = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("terrain view"),
            contents: bytemuck::bytes_of(&TerrainViewGpu::none()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let terrain = crate::terrain::field::TerrainField::new(device, rt);
        let array_view = |t: &wgpu::Texture| {
            t.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            })
        };
        let (heights, materials) = (array_view(&terrain.tile_gen.height), array_view(&terrain.tile_gen.material));
        let mut entries = vec![
            wgpu::BindGroupEntry { binding: 0, resource: systems.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: planets.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&maps.climate_view) },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&maps.sampler) },
            wgpu::BindGroupEntry { binding: 4, resource: terrain_view.as_entire_binding() },
        ];
        if let Some(accel) = &terrain.tile_gen.accel {
            entries.extend([
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&heights) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&materials) },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::AccelerationStructure(&accel.tlas),
                },
                wgpu::BindGroupEntry { binding: 8, resource: accel.vertices.as_entire_binding() },
            ]);
        }
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("near field"),
            layout: &layout,
            entries: &entries,
        });
        Self { layout, bind_group, systems, planets, terrain_view, selection: Selection::default(), maps, terrain }
    }

    /// Where the tiles are for the trace, for a pilot at `eye` (body-fixed
    /// km) on `planet` in GPU slot `slot`: offsets in f64, then f32.
    fn terrain_view(&self, planet: &planets::Planet, eye: V3, slot: u32) -> TerrainViewGpu {
        let tiles_in = self.terrain.tile_gen.accel.as_ref().is_some_and(|a| a.instances > 0);
        let Some(anchor) = self.terrain.anchor_km().filter(|_| tiles_in) else { return TerrainViewGpu::none() };
        let f = |v: V3, w: f64| [v[0] as f32, v[1] as f32, v[2] as f32, w as f32];
        let r = vec3::norm(eye);
        TerrainViewGpu {
            eye: f(vec3::sub(eye, anchor), r - planet.radius_km),
            up: f(vec3::scale(eye, 1.0 / r), planet.radius_km),
            anchor: f(vec3::normalize(anchor), vec3::norm(anchor)),
            ids: [slot, 0, 0, 0],
        }
    }

    /// The planet whose terrain tiles are in the trace: its body axes in
    /// the view axes and the pilot's body-fixed position (km).
    pub fn terrain_pose(&self) -> Option<([V3; 3], V3)> {
        let key = self.maps.baked?;
        self.terrain.tile_gen.accel.as_ref().filter(|a| a.instances > 0)?;
        let p = self.selection.planets.iter().find(|p| {
            let sys = &self.selection.systems[p.system].system;
            (sys.star, sys.generation, p.index) == key
        })?;
        Some((p.body_axes, p.pilot_body_km()))
    }

    pub fn layout(&self) -> &wgpu::BindGroupLayout {
        &self.layout
    }

    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    pub fn update(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        world: &World,
        e: &kerr::pilot::Tetrad,
        pixel_angle: f64,
        profiler: Option<&mut crate::profile::Profiler>,
    ) {
        self.selection = select(world, e, pixel_angle);
        // Bake the maps of the nearest solid world with air in range, once.
        let nearest = self
            .selection
            .planets
            .iter()
            .filter(|p| {
                let planet = p.planet(&self.selection);
                !planet.kind.is_giant()
                    && planet.atmosphere.is_some()
                    && p.distance_km < MAPS_RANGE_RADII * planet.radius_km
            })
            .min_by(|a, b| a.distance_km.total_cmp(&b.distance_km))
            .map(|p| {
                let sys = &self.selection.systems[p.system].system;
                ((sys.star, sys.generation, p.index), p.planet(&self.selection).clone())
            });
        if let Some((key, planet)) = nearest
            && self.maps.baked != Some(key)
        {
            self.maps.bake(device, queue, key, &planet);
        }
        let baked = self.maps.baked;
        let mut view = TerrainViewGpu::none();
        // Keep that world's terrain tiles filled around the pilot.
        if let Some(key) = baked
            && let Some(p) = self.selection.planets.iter().find(|p| {
                let sys = &self.selection.systems[p.system].system;
                (sys.star, sys.generation, p.index) == key
            })
        {
            let planet = p.planet(&self.selection).clone();
            let eye = p.pilot_body_km();
            self.terrain.update(device, queue, (key, &planet), eye, pixel_angle, profiler);
            view = self.terrain_view(&planet, eye, p.gpu.ids[3]);
        }
        queue.write_buffer(&self.terrain_view, 0, bytemuck::bytes_of(&view));
        let systems: Vec<SystemGpu> = self.selection.systems.iter().map(|s| s.gpu).collect();
        let planets: Vec<PlanetGpu> = self
            .selection
            .planets
            .iter()
            .map(|p| {
                let sys = &self.selection.systems[p.system].system;
                let mut gpu = p.gpu;
                gpu.detail[1] = if baked == Some((sys.star, sys.generation, p.index)) { 1.0 } else { 0.0 };
                gpu
            })
            .collect();
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
        assert_eq!(std::mem::size_of::<PlanetGpu>(), 8 * 16);
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
        let sel = select(&w, &w.pilot.e, 1e-3);
        let p = sel.planets.iter().find(|p| p.gpu.ids[2] == 0 && p.index == 0).expect("planet 0 selected");
        let c = p.gpu.centre;
        // Straight ahead (forward axis), 3 radii away.
        let d = (c[0] as f64, c[1] as f64, c[2] as f64);
        assert!((d.0 / (3.0 * r) - 1.0).abs() < 1e-3, "{d:?}");
        assert!(d.1.abs() < 1e-3 * r && d.2.abs() < 1e-3 * r, "{d:?}");
        assert!(sel.systems[0].gpu.beta[3] < 1e-6);
        assert!(p.angular_radius > 0.3);
    }

    /// Physics and rendering agree on where the ground is: a ship landed
    /// 10 m up (from its body-fixed offset) and one flying 2 km up (from
    /// its position) are drawn exactly that high, to 1 cm. Measured the old
    /// way, in simulation coordinates, they were 77–109 m off.
    #[test]
    fn ground_is_where_the_physics_puts_it() {
        let mut w = World::new(WorldConfig::sgr_a(300, 1));
        let seed = w.cluster.cfg.seed;
        let (star, system) = w
            .cluster
            .bodies
            .iter()
            .enumerate()
            .find_map(|(i, b)| planets::system(seed, i, b).filter(|s| s.planets.len() > 1).map(|s| (i, s)))
            .expect("a system");
        let i = 1;
        let body = &w.cluster.bodies[star];
        let pref = PlanetRef { star, generation: body.generation, planet: i };
        let r_km = system.planets[i].radius_km;
        let (off_km, vel_km_s) = system.planet_state(i, w.cluster.t);
        let pvel = vec3::add(geodesic::coordinate_velocity(&w.kerr, &body.state), vec3::scale(vel_km_s, 1.0 / C_KM_S));
        // Fly in to 3 radii, co-moving, so the system becomes the local one.
        let dir = vec3::normalize([0.3, -0.8, 0.5]);
        let at =
            vec3::add(body.position(), vec3::scale(vec3::add(off_km, vec3::scale(dir, 3.0 * r_km)), 1.0 / KM_PER_M));
        let mut pilot = kerr::pilot::Pilot::new(&w.kerr, at, pvel, dir, vec3::any_orthogonal(dir)).unwrap();
        pilot.x[0] = w.pilot.x[0];
        w.pilot = pilot;
        w.teleported();
        let altitude = |w: &World| {
            let sel = select(w, &w.pilot.e, 1e-3);
            let p = sel.planets.iter().find(|p| p.gpu.ids[2] == 0 && p.index == i).expect("planet selected");
            (vec3::norm(p.centre_km) - r_km) * 1000.0
        };
        // Landed 10 m up, a few directions.
        for d in [[1.0, 0.0, 0.0], [0.0, -1.0, 0.3], [-0.4, 0.2, -0.9]] {
            let offset = vec3::scale(vec3::normalize(d), (r_km + 0.01) / KM_PER_M);
            assert!(w.land(pref, offset));
            w.step(1.0 / 60.0, &kerr::world::Input { autopilot: -1, ..Default::default() });
            let h = altitude(&w);
            assert!((h - 10.0).abs() < 0.01, "landed: drawn {h} m up");
        }
        // Flying: whatever the physics' telemetry says.
        w.status = kerr::world::PilotStatus::Free;
        let offset = vec3::scale(dir, (r_km + 2.0) / KM_PER_M);
        assert!(w.land(pref, offset));
        w.status = kerr::world::PilotStatus::Free;
        let t = w.telemetry().planet.expect("planet telemetry");
        let h = altitude(&w);
        assert!((h - t.altitude_km * 1000.0).abs() < 0.01 && (h - 2000.0).abs() < 1.0, "flying: {h} m vs {t:?}");
    }
}
