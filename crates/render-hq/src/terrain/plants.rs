//! Plants as geometry: procedural meshes traced by the ray-tracing hardware
//! as instances beside the terrain tiles (`rt.rs`), shaded as bark and
//! leaves (`terrain_rq.wgsl`).
//!
//! The first is a coconut palm: a tapering trunk that curves as it grows,
//! flared at the foot, and a crown of arching fronds, each a folded blade
//! tapering to its tip, with a few dead ones hanging below. Meshes are in
//! metres, z up from the foot; instances scale them to the TLAS's km.

use kerr::vec3::{self, V3};

/// A plant's mesh: vertices (m), triangles, and per triangle its normal
/// (unit, object space) and which part it is.
#[derive(Clone, Debug, Default)]
pub struct PlantMesh {
    pub positions: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
    /// xyz: the triangle's normal; w: its part ([`BARK`], [`LEAF`],
    /// [`DEAD_LEAF`]).
    pub normals: Vec<[f32; 4]>,
}

/// Parts, for shading.
pub const BARK: f32 = 0.0;
pub const LEAF: f32 = 1.0;
pub const DEAD_LEAF: f32 = 2.0;

impl PlantMesh {
    fn vertex(&mut self, p: V3) -> u32 {
        self.positions.push([p[0] as f32, p[1] as f32, p[2] as f32]);
        self.positions.len() as u32 - 1
    }

    fn triangle(&mut self, a: u32, b: u32, c: u32, part: f32) {
        let [pa, pb, pc] = [a, b, c].map(|i| self.positions[i as usize].map(f64::from));
        let n = vec3::cross(vec3::sub(pb, pa), vec3::sub(pc, pa));
        let len = vec3::norm(n);
        if len < 1e-12 {
            return;
        }
        let n = vec3::scale(n, 1.0 / len);
        self.indices.extend([a, b, c]);
        self.normals.push([n[0] as f32, n[1] as f32, n[2] as f32, part]);
    }

    fn quad(&mut self, a: u32, b: u32, c: u32, d: u32, part: f32) {
        self.triangle(a, b, c, part);
        self.triangle(a, c, d, part);
    }

    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// The highest point and the farthest reach from the trunk's axis (m).
    pub fn extent(&self) -> (f32, f32) {
        let top = self.positions.iter().map(|p| p[2]).fold(0.0, f32::max);
        let reach = self.positions.iter().map(|p| p[0].hypot(p[1])).fold(0.0, f32::max);
        (top, reach)
    }
}

/// A small deterministic generator (splitmix64).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }
    fn range(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.next()
    }
}

/// A coconut palm, one of a family by `seed`: 7–12 m tall.
pub fn palm(seed: u64) -> PlantMesh {
    let mut rng = Rng(seed.wrapping_mul(0x2545_f491_4f6c_dd1d) ^ 0x51ed);
    let mut m = PlantMesh::default();
    let height = rng.range(7.0, 12.0);
    let lean = rng.range(0.08, 0.35);
    let sway = rng.range(-0.08, 0.08);
    // The trunk's axis: leaning more as it rises (it grew towards the
    // light), with a little S.
    let axis =
        |t: f64| -> V3 { [height * (lean * t * t + sway * (std::f64::consts::TAU * t).sin() * 0.3), 0.0, height * t] };
    let radius = |t: f64| 0.16 + 0.07 * (1.0 - t) + 0.14 * (-(t * 30.0)).exp();
    let (rings, around) = (18, 9);
    let mut ring_start = Vec::new();
    for i in 0..=rings {
        let t = i as f64 / rings as f64;
        let c = axis(t);
        // The ring square to the axis.
        let tangent = vec3::normalize(vec3::sub(axis(t + 1e-3), axis(t - 1e-3)));
        let side = vec3::normalize(vec3::cross([0.0, 1.0, 0.0], tangent));
        let other = vec3::cross(tangent, side);
        ring_start.push(m.positions.len() as u32);
        for k in 0..around {
            let a = std::f64::consts::TAU * k as f64 / around as f64;
            // Old leaf scars ridge the trunk.
            let r = radius(t) * (1.0 + 0.04 * ((t * rings as f64 * 2.0).fract() - 0.5));
            let p = vec3::add(c, vec3::add(vec3::scale(side, r * a.cos()), vec3::scale(other, r * a.sin())));
            m.vertex(p);
        }
    }
    for i in 0..rings {
        for k in 0..around {
            let (a, b) = (ring_start[i] + k, ring_start[i] + (k + 1) % around);
            let (c, d) = (ring_start[i + 1] + (k + 1) % around, ring_start[i + 1] + k);
            m.quad(a, b, c, d, BARK);
        }
    }
    let top = axis(1.0);
    // The crown: live fronds arching out and down, dead ones hanging.
    let fronds = rng.range(13.0, 18.0) as usize;
    for f in 0..fronds + 4 {
        let dead = f >= fronds;
        let az = std::f64::consts::TAU * (f as f64 + rng.range(-0.3, 0.3)) / fronds as f64;
        let rise = if dead { rng.range(-1.3, -0.9) } else { rng.range(-0.2, 0.9) };
        let len = if dead { rng.range(1.8, 2.8) } else { rng.range(3.6, 5.2) };
        let droop = if dead { 0.2 } else { rng.range(0.9, 1.5) };
        let dir_h = [az.cos(), az.sin(), 0.0];
        let part = if dead { DEAD_LEAF } else { LEAF };
        frond(&mut m, top, dir_h, rise, len, droop, part);
    }
    m
}

/// A frond from `base`: its midrib leaves along horizontal `dir_h` at
/// `rise` (rad above level), `len` m long, bending down by `droop`; its
/// blade is folded along the midrib (a shallow V) and tapers to the tip.
fn frond(m: &mut PlantMesh, base: V3, dir_h: V3, rise: f64, len: f64, droop: f64, part: f32) {
    let segs = 10;
    let up = [0.0, 0.0, 1.0];
    let side = vec3::normalize(vec3::cross(up, dir_h));
    let mut rib = Vec::new();
    let mut ang = rise;
    let mut p = base;
    for i in 0..=segs {
        rib.push((p, ang));
        let step = len / segs as f64;
        let d = vec3::add(vec3::scale(dir_h, ang.cos()), vec3::scale(up, ang.sin()));
        p = vec3::axpy(p, step, d);
        ang -= droop / segs as f64 * (1.0 + i as f64 / segs as f64);
    }
    let mut prev: Option<[u32; 3]> = None;
    for (i, &(p, a)) in rib.iter().enumerate() {
        let s = i as f64 / segs as f64;
        let width = 0.75 * (std::f64::consts::PI * s.powf(0.7)).sin().max(0.0) + 0.02;
        // The blade's halves fold up from the midrib.
        let d = vec3::add(vec3::scale(dir_h, a.cos()), vec3::scale(up, a.sin()));
        let normal = vec3::normalize(vec3::cross(d, side));
        let fold = vec3::scale(normal, 0.25 * width);
        let left = vec3::add(vec3::axpy(p, width, side), fold);
        let right = vec3::add(vec3::axpy(p, -width, side), fold);
        let ids = [m.vertex(left), m.vertex(p), m.vertex(right)];
        if let Some(q) = prev {
            m.quad(q[0], q[1], ids[1], ids[0], part);
            m.quad(q[1], q[2], ids[2], ids[1], part);
        }
        prev = Some(ids);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Palms are palm-sized, closed-form (every index valid, one normal a
    /// triangle, unit length) and differ by seed.
    #[test]
    fn palms() {
        for seed in 0..6 {
            let p = palm(seed);
            let (top, reach) = p.extent();
            assert!((6.0..16.0).contains(&top) && (2.5..9.0).contains(&reach), "{seed}: {top} {reach}");
            assert_eq!(p.normals.len(), p.triangle_count());
            assert!(p.indices.iter().all(|&i| (i as usize) < p.positions.len()));
            assert!(p.normals.iter().all(|n| ((n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt() - 1.0).abs() < 1e-4));
            assert!(p.triangle_count() < 4000, "{}", p.triangle_count());
            assert!(p.normals.iter().any(|n| n[3] == BARK) && p.normals.iter().any(|n| n[3] == LEAF));
        }
        assert_ne!(palm(1).extent(), palm(2).extent());
    }
}
