//! The ship's hull: a procedural triangle mesh in ship-frame metres
//! (x forward, y left, z up), about 52 m long.
//!
//! The layout follows what a fusion torch ship needs rather than film
//! convention: a slim crew fuselage in front, far from the drive; a
//! propellant tank wrapped in gold multi-layer insulation; an open truss
//! carrying four large radiators (a drive this powerful has to dump its waste
//! heat, and radiators are the only way to do it in vacuum); the drive and
//! its magnetic-nozzle bell at the back. Radiators sit edge-on to the
//! exhaust and in an X so they see little of each other.
//!
//! Parts are built from grids (lathes around an axis, flat slabs), each
//! with smooth normals across its own faces and hard edges where parts meet.
//! Ambient occlusion is baked per vertex by ray casting against the ship's
//! own BVH: the ship is rigid, so this is exact for its self-shadowing of a
//! uniform sky and costs nothing per pixel.

use super::bvh::Bvh;
use bytemuck::{Pod, Zeroable};
use kerr::vec3::{self, V3};

/// Material ids (`Vertex::material`), matching the `MAT_*` constants in
/// `ship.wgsl`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Material {
    /// White painted aluminium panels, with markings.
    Paint = 0,
    /// Bare brushed metal (titanium and aluminium alloys).
    Metal = 1,
    /// High-emissivity ceramic radiator panels (glow when hot).
    Radiator = 2,
    /// Refractory outer skin of the nozzle bell.
    Bell = 3,
    /// Inside of the bell and the reaction chamber, lit by the plasma.
    BellInner = 4,
    /// Gold-coated multi-layer insulation over the propellant tank.
    Foil = 5,
    /// Black anodised parts and glass.
    Dark = 6,
    /// Navigation lights.
    Light = 7,
}

/// Mirrors the two `vec4`s per vertex read by `ship.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct Vertex {
    pub pos: [f32; 3],
    /// Fraction of the sky the vertex sees past the ship itself (baked).
    pub ao: f32,
    pub normal: [f32; 3],
    pub material: u32,
}

pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub triangles: Vec<[u32; 3]>,
}

impl Mesh {
    /// Centre and radius of a sphere holding every vertex.
    pub fn bounding_sphere(&self) -> (V3, f64) {
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for v in &self.vertices {
            for i in 0..3 {
                lo[i] = lo[i].min(v.pos[i] as f64);
                hi[i] = hi[i].max(v.pos[i] as f64);
            }
        }
        let c = vec3::scale(vec3::add(lo, hi), 0.5);
        let r = self.vertices.iter().map(|v| vec3::norm(vec3::sub(pos(v), c))).fold(0.0, f64::max);
        (c, r)
    }

    /// Put the triangles in `order` (the BVH's leaf order), so a triangle's
    /// index is the same for the software BVH and the hardware structure.
    pub fn reorder(&mut self, order: &[u32]) {
        self.triangles = order.iter().map(|&i| self.triangles[i as usize]).collect();
    }
}

fn pos(v: &Vertex) -> V3 {
    [v.pos[0] as f64, v.pos[1] as f64, v.pos[2] as f64]
}

const X: V3 = [1.0, 0.0, 0.0];
const Y: V3 = [0.0, 1.0, 0.0];
const Z: V3 = [0.0, 0.0, 1.0];

/// Segments around the fuselage and the bell.
const SEG: usize = 48;
/// Segments around small rings and the dish.
const SEG_SMALL: usize = 32;

/// A frame for surfaces of revolution: points are `o + x a + r (sy cos φ u
/// + sz sin φ v)`, with `u × v = a`.
#[derive(Clone, Copy)]
struct Axis {
    o: V3,
    a: V3,
    u: V3,
    v: V3,
    sy: f64,
    sz: f64,
}

impl Axis {
    fn x(sy: f64, sz: f64) -> Self {
        Self { o: [0.0; 3], a: X, u: Y, v: Z, sy, sz }
    }

    fn along(o: V3, a: V3) -> Self {
        let a = vec3::normalize(a);
        let u = vec3::normalize(vec3::any_orthogonal(a));
        Self { o, a, u, v: vec3::cross(a, u), sy: 1.0, sz: 1.0 }
    }

    fn point(&self, x: f64, r: f64, phi: f64) -> V3 {
        let radial = vec3::add(vec3::scale(self.u, self.sy * phi.cos()), vec3::scale(self.v, self.sz * phi.sin()));
        vec3::add(vec3::axpy(self.o, x, self.a), vec3::scale(radial, r))
    }
}

#[derive(Default)]
struct Builder {
    vertices: Vec<Vertex>,
    triangles: Vec<[u32; 3]>,
}

impl Builder {
    /// A `rows × cols` grid of points (row major). Quads between
    /// neighbours become two triangles facing along ∂/∂row × ∂/∂col (or the
    /// opposite with `flip`); `wrap` joins the last column to the first.
    /// Normals average the grid's own faces, weighted by area.
    fn grid(&mut self, pts: &[V3], rows: usize, cols: usize, wrap: bool, flip: bool, mat: Material) {
        assert_eq!(pts.len(), rows * cols);
        let base = self.vertices.len() as u32;
        let mut normals = vec![[0.0; 3]; pts.len()];
        let quads_c = if wrap { cols } else { cols - 1 };
        let mut add = |tri: [usize; 3], out: &mut Vec<[u32; 3]>| {
            let t = if flip { [tri[0], tri[2], tri[1]] } else { tri };
            let n = vec3::cross(vec3::sub(pts[t[1]], pts[t[0]]), vec3::sub(pts[t[2]], pts[t[0]]));
            if vec3::norm(n) < 1e-12 {
                return;
            }
            for &i in &t {
                normals[i] = vec3::add(normals[i], n);
            }
            out.push([base + t[0] as u32, base + t[1] as u32, base + t[2] as u32]);
        };
        for r in 0..rows - 1 {
            for c in 0..quads_c {
                let c1 = (c + 1) % cols;
                let (a, b, cc, d) = (r * cols + c, (r + 1) * cols + c, (r + 1) * cols + c1, r * cols + c1);
                add([a, b, d], &mut self.triangles);
                add([b, cc, d], &mut self.triangles);
            }
        }
        for (p, n) in pts.iter().zip(&normals) {
            let n = if vec3::norm(*n) > 0.0 { vec3::normalize(*n) } else { X };
            self.vertices.push(Vertex {
                pos: p.map(|x| x as f32),
                ao: 1.0,
                normal: n.map(|x| x as f32),
                material: mat as u32,
            });
        }
    }

    /// Revolve a profile of (x, r) points, smooth along it. Profiles run
    /// from front to back (decreasing x) to face outwards; `flip` faces the
    /// axis instead.
    fn lathe(&mut self, ax: &Axis, profile: &[(f64, f64)], seg: usize, flip: bool, mat: Material) {
        let mut pts = Vec::with_capacity(profile.len() * seg);
        for &(x, r) in profile {
            for j in 0..seg {
                pts.push(ax.point(x, r, std::f64::consts::TAU * j as f64 / seg as f64));
            }
        }
        self.grid(&pts, profile.len(), seg, true, flip, mat);
    }

    /// Revolve a polyline with a hard edge at every corner.
    fn lathe_creased(&mut self, ax: &Axis, profile: &[(f64, f64)], seg: usize, mat: Material) {
        for w in profile.windows(2) {
            self.lathe(ax, w, seg, false, mat);
        }
    }

    /// A flat bilinear patch `p00 + s (p10 − p00) + t (p01 − p00)` (with
    /// `p11` for the far corner), `nu × nv` quads, facing `out`.
    #[allow(clippy::too_many_arguments)]
    fn patch(&mut self, p00: V3, p10: V3, p01: V3, p11: V3, nu: usize, nv: usize, out: V3, mat: Material) {
        let mut pts = Vec::with_capacity((nu + 1) * (nv + 1));
        for i in 0..=nu {
            let s = i as f64 / nu as f64;
            let a = vec3::add(vec3::scale(p00, 1.0 - s), vec3::scale(p10, s));
            let b = vec3::add(vec3::scale(p01, 1.0 - s), vec3::scale(p11, s));
            for j in 0..=nv {
                let t = j as f64 / nv as f64;
                pts.push(vec3::add(vec3::scale(a, 1.0 - t), vec3::scale(b, t)));
            }
        }
        let facing = vec3::cross(vec3::sub(p10, p00), vec3::sub(p01, p00));
        self.grid(&pts, nu + 1, nv + 1, false, vec3::dot(facing, out) < 0.0, mat);
    }

    /// A plate of `thickness` whose mid-plane is the patch (p00, p10, p01,
    /// p11) with unit normal `n`: faces of material `face`, rims of `rim`.
    #[allow(clippy::too_many_arguments)]
    fn slab(&mut self, c: [V3; 4], n: V3, thickness: f64, nu: usize, nv: usize, face: Material, rim: Material) {
        let h = vec3::scale(n, 0.5 * thickness);
        let top = c.map(|p| vec3::add(p, h));
        let bot = c.map(|p| vec3::sub(p, h));
        self.patch(top[0], top[1], top[2], top[3], nu, nv, n, face);
        self.patch(bot[0], bot[1], bot[2], bot[3], nu, nv, vec3::scale(n, -1.0), face);
        let centre = vec3::scale(c.iter().fold([0.0; 3], |s, p| vec3::add(s, *p)), 0.25);
        // Rims: edges 00–10, 10–11, 11–01, 01–00.
        for (i, j) in [(0, 1), (1, 3), (3, 2), (2, 0)] {
            let mid = vec3::scale(vec3::add(c[i], c[j]), 0.5);
            let out = vec3::sub(mid, centre);
            self.patch(bot[i], bot[j], top[i], top[j], 1, 1, out, rim);
        }
    }

    /// A box with centre `c`, unit axes `ax` and half extents `h`.
    fn cuboid(&mut self, c: V3, ax: [V3; 3], h: [f64; 3], mat: Material) {
        for k in 0..3 {
            let (i, j) = ((k + 1) % 3, (k + 2) % 3);
            for s in [-1.0, 1.0] {
                let fc = vec3::axpy(c, s * h[k], ax[k]);
                let du = vec3::scale(ax[i], h[i]);
                let dv = vec3::scale(ax[j], h[j]);
                let p = |a: f64, b: f64| vec3::add(fc, vec3::add(vec3::scale(du, a), vec3::scale(dv, b)));
                self.patch(p(-1.0, -1.0), p(1.0, -1.0), p(-1.0, 1.0), p(1.0, 1.0), 1, 1, vec3::scale(ax[k], s), mat);
            }
        }
    }

    /// A box beam from `a` to `b` with square section of half width `w`,
    /// its sides aligned with `side` (projected perpendicular to the beam).
    fn beam(&mut self, a: V3, b: V3, w: f64, side: V3, mat: Material) {
        let d = vec3::sub(b, a);
        let len = vec3::norm(d);
        let t = vec3::scale(d, 1.0 / len);
        let s = vec3::normalize(vec3::axpy(side, -vec3::dot(side, t), t));
        let u = vec3::cross(t, s);
        self.cuboid(vec3::scale(vec3::add(a, b), 0.5), [t, s, u], [0.5 * len, w, w], mat);
    }

    fn finish(self) -> Mesh {
        Mesh { vertices: self.vertices, triangles: self.triangles }
    }
}

/// Elliptical section of the fuselage: a little wider than tall.
const HULL_SY: f64 = 1.08;
const HULL_SZ: f64 = 0.92;
/// Radiator roots, around the x axis from +y towards +z.
pub const RADIATOR_ANGLES_DEG: [f64; 4] = [30.0, 150.0, 210.0, 330.0];

/// The ship, with baked ambient occlusion and triangles in BVH order.
pub fn ship() -> (Mesh, Bvh) {
    let mut mesh = geometry();
    let bvh = Bvh::build(&mesh);
    mesh.reorder(&bvh.order);
    bake_ambient_occlusion(&mut mesh, &bvh, 64, 10.0);
    (mesh, bvh)
}

fn geometry() -> Mesh {
    use Material::*;
    let mut b = Builder::default();
    let hull = Axis::x(HULL_SY, HULL_SZ);
    let round = Axis::x(1.0, 1.0);

    // Crew fuselage: an ogive nose flowing into the cabin section.
    b.lathe(
        &hull,
        &[
            (26.0, 0.0),
            (25.9, 0.36),
            (25.6, 0.78),
            (25.1, 1.2),
            (24.3, 1.66),
            (23.2, 2.1),
            (21.8, 2.5),
            (20.1, 2.8),
            (18.1, 2.98),
            (15.8, 3.05),
            (13.6, 3.05),
        ],
        SEG,
        false,
        Paint,
    );
    b.lathe_creased(&hull, &[(13.6, 3.05), (13.6, 3.2), (12.8, 3.2), (12.8, 3.05)], SEG, Metal);
    b.lathe(&hull, &[(12.8, 3.05), (9.0, 3.05), (5.2, 3.05), (3.6, 2.96), (2.6, 2.7), (1.9, 2.2)], SEG, false, Paint);

    // Propellant tank in gold insulation, with metal straps.
    let (tank_r, fwd, aft, dome) = (3.35, 0.4, -5.4, 2.1);
    let mut tank = Vec::new();
    for i in 0..=8 {
        let a = std::f64::consts::FRAC_PI_2 * (1.0 - i as f64 / 8.0);
        tank.push((fwd + dome * a.sin(), tank_r * a.cos()));
    }
    for i in 1..=8 {
        let a = std::f64::consts::FRAC_PI_2 * i as f64 / 8.0;
        tank.push((aft - dome * a.sin(), tank_r * a.cos()));
    }
    b.lathe(&round, &tank, SEG, false, Foil);
    for x in [-0.4, -2.5, -4.6] {
        let r = tank_r + 0.05;
        b.lathe_creased(
            &round,
            &[(x + 0.14, tank_r - 0.05), (x + 0.14, r), (x - 0.14, r), (x - 0.14, tank_r - 0.05)],
            SEG,
            Metal,
        );
    }

    // Truss: central tube, four longerons at the radiator roots, frames.
    b.lathe(&round, &[(-6.0, 0.75), (-15.3, 0.75)], SEG_SMALL, false, Metal);
    let radial = |deg: f64| {
        let p = deg.to_radians();
        vec3::add(vec3::scale(Y, p.cos()), vec3::scale(Z, p.sin()))
    };
    for deg in RADIATOR_ANGLES_DEG {
        let r = radial(deg);
        b.beam(vec3::axpy([-6.6, 0.0, 0.0], 1.35, r), vec3::axpy([-15.2, 0.0, 0.0], 1.35, r), 0.14, r, Metal);
    }
    for x in [-7.0, -9.8, -12.6] {
        b.lathe_creased(
            &round,
            &[(x + 0.1, 1.15), (x + 0.1, 1.55), (x - 0.1, 1.55), (x - 0.1, 1.15), (x + 0.1, 1.15)],
            SEG_SMALL,
            Metal,
        );
    }

    // Radiators: swept plates edge-on to the exhaust, heat pipes along the
    // span (drawn by the shader), a metal frame round the rim.
    for deg in RADIATOR_ANGLES_DEG {
        let r = radial(deg);
        let n = vec3::cross(X, r);
        let at = |x: f64, rho: f64| vec3::axpy([x, 0.0, 0.0], rho, r);
        let corners = [at(-6.8, 1.45), at(-15.3, 1.45), at(-10.9, 13.2), at(-15.3, 13.2)];
        b.slab(corners, n, 0.12, 10, 14, Radiator, Metal);
        // Navigation light on the trailing tip.
        b.cuboid(at(-15.25, 13.34), [X, r, n], [0.14, 0.14, 0.1], Light);
    }

    // Drive housing, then the nozzle bell (outer skin, inner surface,
    // lip) and the reaction chamber window at its throat.
    b.lathe(
        &hull,
        &[
            (-14.4, 0.75),
            (-14.9, 1.9),
            (-15.3, 2.45),
            (-15.7, 2.62),
            (-17.3, 2.62),
            (-17.8, 2.35),
            (-18.2, 1.8),
            (-18.45, 1.3),
        ],
        SEG,
        false,
        Metal,
    );
    let (throat, exit_x, exit_r, skin) = (1.25, -26.0, 4.35, 0.09);
    let bell = |off: f64| -> Vec<(f64, f64)> {
        (0..=15)
            .map(|i| {
                let s = i as f64 / 15.0;
                (-18.45 + (exit_x + 18.45) * s, throat + (exit_r - throat) * (1.0 - (1.0 - s).powi(2)) - off)
            })
            .collect()
    };
    b.lathe(&round, &bell(0.0), SEG, false, Bell);
    b.lathe(&round, &bell(skin), SEG, true, BellInner);
    b.lathe_creased(&round, &[(exit_x, exit_r), (exit_x, exit_r - skin)], SEG, Bell);
    b.lathe(&round, &[(-18.5, throat - skin), (-18.5, 0.0)], SEG, false, BellInner);

    // Reaction-control thruster blocks, fore and aft.
    for x in [9.2, -16.5] {
        for k in 0..4 {
            let phi = (45.0 + 90.0 * k as f64).to_radians();
            let dir = vec3::normalize([0.0, HULL_SY * phi.cos(), HULL_SZ * phi.sin()]);
            let rim = if x > 0.0 { 3.05 } else { 2.62 };
            let rho = rim * (HULL_SY * phi.cos()).hypot(HULL_SZ * phi.sin());
            let t = vec3::cross(X, dir);
            b.cuboid(vec3::axpy([x, 0.0, 0.0], rho + 0.12, dir), [X, t, dir], [0.36, 0.28, 0.22], Metal);
        }
    }
    // Anti-collision strobes on the drive housing.
    for s in [1.0, -1.0] {
        b.cuboid([-16.5, 0.0, s * (2.62 * HULL_SZ + 0.08)], [X, Y, Z], [0.12, 0.12, 0.08], Light);
    }

    // High-gain antenna: a dish on a mast behind the cabin, looking up and
    // back.
    b.beam([6.2, 0.0, 2.6], [6.2, 0.0, 4.35], 0.09, X, Metal);
    let dish_axis = vec3::normalize([-0.45, 0.0, 1.0]);
    let dish = Axis::along([6.2, 0.0, 4.45], dish_axis);
    let (dish_r, depth) = (1.35, 0.32);
    let dish_profile = |off: f64| -> Vec<(f64, f64)> {
        (0..=8)
            .map(|i| {
                let r = dish_r * i as f64 / 8.0;
                (depth * (r / dish_r).powi(2) - off, r)
            })
            .collect()
    };
    // Rows run outwards and towards +axis, so the concave side faces the
    // axis direction unflipped.
    b.lathe(&dish, &dish_profile(0.0), SEG_SMALL, false, Paint);
    b.lathe(&dish, &dish_profile(0.05), SEG_SMALL, true, Metal);
    b.beam(dish.o, vec3::axpy(dish.o, 0.95, dish_axis), 0.035, X, Dark);

    b.finish()
}

/// Per-vertex ambient visibility: the cosine-weighted fraction of `rays`
/// hemisphere rays that leave the ship, with hits beyond `reach` metres
/// counting partly (obscurance), so wide open areas don't turn grey.
pub fn bake_ambient_occlusion(mesh: &mut Mesh, bvh: &Bvh, rays: usize, reach: f64) {
    let tris: Vec<[V3; 3]> = mesh.triangles.iter().map(|t| t.map(|i| pos(&mesh.vertices[i as usize]))).collect();
    for (vi, v) in mesh.vertices.iter_mut().enumerate() {
        let n = [v.normal[0] as f64, v.normal[1] as f64, v.normal[2] as f64];
        let o = vec3::axpy(pos(v), 0.02, n);
        let t1 = vec3::normalize(vec3::any_orthogonal(n));
        let t2 = vec3::cross(n, t1);
        let side = (rays as f64).sqrt().ceil() as usize;
        let mut seen = 0.0;
        let mut total = 0.0;
        for k in 0..rays {
            // Stratified cosine-weighted directions, rotated per vertex.
            let jitter = hash2(vi as u32, k as u32);
            let u = ((k % side) as f64 + jitter.0) / side as f64;
            let w = ((k / side) as f64 + jitter.1) / side as f64;
            let r = u.sqrt();
            let phi = std::f64::consts::TAU * w;
            let d = vec3::add(
                vec3::scale(n, (1.0 - u).max(0.0).sqrt()),
                vec3::add(vec3::scale(t1, r * phi.cos()), vec3::scale(t2, r * phi.sin())),
            );
            total += 1.0;
            seen += match bvh.intersect(&tris, o, d, reach) {
                Some(hit) => (hit.t / reach).powi(2),
                None => 1.0,
            };
        }
        v.ao = (seen / total) as f32;
    }
}

fn hash2(a: u32, b: u32) -> (f64, f64) {
    let mut x = a.wrapping_mul(0x9E37_79B9) ^ b.wrapping_mul(0x85EB_CA6B);
    x ^= x >> 15;
    x = x.wrapping_mul(0x2C1B_3C6D);
    x ^= x >> 12;
    let y = x.wrapping_mul(0x297A_2D39) ^ (x >> 16);
    ((x >> 8) as f64 / (1u32 << 24) as f64, (y >> 8) as f64 / (1u32 << 24) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vertex_layout_matches_wgsl() {
        assert_eq!(std::mem::size_of::<Vertex>(), 32);
    }

    /// A few thousand triangles, all indices valid, about 52 m long.
    #[test]
    fn mesh_is_sane() {
        let m = geometry();
        assert!((3000..14000).contains(&m.triangles.len()), "{} triangles", m.triangles.len());
        assert!(m.triangles.iter().flatten().all(|&i| (i as usize) < m.vertices.len()));
        let (c, r) = m.bounding_sphere();
        assert!(r > 20.0 && r < 32.0, "radius {r}");
        assert!(vec3::norm(c) < 10.0, "centre {c:?}");
        assert!(m.vertices.iter().all(|v| (v.normal.iter().map(|x| x * x).sum::<f32>() - 1.0).abs() < 1e-3));
    }

    /// Normals point out of the hull: most of the ship sees much of the sky
    /// (inward normals would see only the inside of a closed part).
    #[test]
    fn normals_face_outwards() {
        let (m, _) = ship();
        let open = m.vertices.iter().filter(|v| v.ao > 0.35).count();
        // Parts tucked between the radiators and inside the truss see less
        // (about a fifth of the vertices).
        assert!(open as f64 > 0.75 * m.vertices.len() as f64, "{open} of {}", m.vertices.len());
        // The top of the cabin sees nearly the whole sky.
        let top = m
            .vertices
            .iter()
            .filter(|v| v.pos[0] > 15.0 && v.pos[0] < 20.0 && v.normal[2] > 0.95)
            .map(|v| v.ao)
            .fold(1.0f32, f32::min);
        assert!(top > 0.9, "cabin top ao {top}");
    }
}
