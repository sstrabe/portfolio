//! A bounding volume hierarchy over the ship's triangles: used on the CPU
//! to bake ambient occlusion, and uploaded for the software ray tracer in
//! `ship_sw.wgsl` when the GPU has no hardware ray queries.
//!
//! Binary, built top-down with the surface area heuristic over 12 bins per
//! axis, flattened depth first: an interior node's first child follows it,
//! the second is at `a`. Leaves hold up to 4 triangles.

// Index loops mirror the per-axis box maths.
#![allow(clippy::needless_range_loop)]

use super::mesh::Mesh;
use bytemuck::{Pod, Zeroable};
use kerr::vec3::{self, V3};

const BINS: usize = 12;
const LEAF_MAX: usize = 4;
/// `NodeGpu::b` for a leaf: count << 2 | LEAF.
const LEAF: u32 = 3;

/// Mirrors `struct ShipNode` in `ship_sw.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct NodeGpu {
    pub lo: [f32; 3],
    /// Interior: index of the second child. Leaf: first triangle.
    pub a: u32,
    pub hi: [f32; 3],
    /// Interior: split axis (0–2). Leaf: triangle count << 2 | 3.
    pub b: u32,
}

#[derive(Clone, Copy)]
struct Aabb {
    lo: V3,
    hi: V3,
}

impl Aabb {
    const EMPTY: Self = Self { lo: [f64::INFINITY; 3], hi: [f64::NEG_INFINITY; 3] };

    fn grow(&mut self, p: V3) {
        for i in 0..3 {
            self.lo[i] = self.lo[i].min(p[i]);
            self.hi[i] = self.hi[i].max(p[i]);
        }
    }

    fn union(mut self, o: &Aabb) -> Self {
        self.grow(o.lo);
        self.grow(o.hi);
        self
    }

    fn area(&self) -> f64 {
        let d = vec3::sub(self.hi, self.lo);
        if d[0] < 0.0 {
            return 0.0;
        }
        2.0 * (d[0] * d[1] + d[1] * d[2] + d[2] * d[0])
    }

    /// Entry distance of the ray, if it meets the box before `t_max`.
    fn hit(&self, o: V3, inv: V3, t_max: f64) -> Option<f64> {
        let (mut t0, mut t1) = (0.0f64, t_max);
        for i in 0..3 {
            let a = (self.lo[i] - o[i]) * inv[i];
            let b = (self.hi[i] - o[i]) * inv[i];
            t0 = t0.max(a.min(b));
            t1 = t1.min(a.max(b));
        }
        (t0 <= t1).then_some(t0)
    }
}

pub struct Bvh {
    pub nodes: Vec<NodeGpu>,
    /// Original index of each triangle, in leaf order.
    pub order: Vec<u32>,
    boxes: Vec<Aabb>,
}

pub struct Hit {
    pub t: f64,
    pub triangle: usize,
}

impl Bvh {
    pub fn build(mesh: &Mesh) -> Self {
        let p = |i: u32| {
            let v = mesh.vertices[i as usize].pos;
            [v[0] as f64, v[1] as f64, v[2] as f64]
        };
        let prims: Vec<(Aabb, V3)> = mesh
            .triangles
            .iter()
            .map(|t| {
                let mut b = Aabb::EMPTY;
                t.iter().for_each(|&i| b.grow(p(i)));
                (b, vec3::scale(vec3::add(b.lo, b.hi), 0.5))
            })
            .collect();
        let mut bvh = Bvh { nodes: Vec::new(), order: (0..prims.len() as u32).collect(), boxes: Vec::new() };
        let mut order = std::mem::take(&mut bvh.order);
        bvh.split(&prims, &mut order, 0);
        bvh.order = order;
        bvh
    }

    /// Build the subtree over `idx` (which starts at `first` in the final
    /// triangle order); returns its node index.
    fn split(&mut self, prims: &[(Aabb, V3)], idx: &mut [u32], first: usize) -> usize {
        let node = self.nodes.len();
        let bounds = idx.iter().fold(Aabb::EMPTY, |b, &i| b.union(&prims[i as usize].0));
        self.nodes.push(NodeGpu::default());
        self.boxes.push(bounds);
        let leaf = |s: &mut Self| {
            s.nodes[node].a = first as u32;
            s.nodes[node].b = (idx.len() as u32) << 2 | LEAF;
            node
        };
        if idx.len() <= LEAF_MAX {
            return leaf(self);
        }
        // Surface area heuristic over centroid bins.
        let mut cb = Aabb::EMPTY;
        idx.iter().for_each(|&i| cb.grow(prims[i as usize].1));
        let mut best = (f64::INFINITY, 0, 0.0);
        for axis in 0..3 {
            let (lo, hi) = (cb.lo[axis], cb.hi[axis]);
            if hi - lo < 1e-9 {
                continue;
            }
            let bin_of = |c: V3| (((c[axis] - lo) / (hi - lo) * BINS as f64) as usize).min(BINS - 1);
            let mut bins = [(Aabb::EMPTY, 0usize); BINS];
            for &i in idx.iter() {
                let b = &mut bins[bin_of(prims[i as usize].1)];
                b.0 = b.0.union(&prims[i as usize].0);
                b.1 += 1;
            }
            for cut in 1..BINS {
                let (l, r) = bins.split_at(cut);
                let side = |s: &[(Aabb, usize)]| s.iter().fold((Aabb::EMPTY, 0), |(a, n), b| (a.union(&b.0), n + b.1));
                let ((la, ln), (ra, rn)) = (side(l), side(r));
                if ln == 0 || rn == 0 {
                    continue;
                }
                let cost = la.area() * ln as f64 + ra.area() * rn as f64;
                if cost < best.0 {
                    best = (cost, axis, lo + (hi - lo) * cut as f64 / BINS as f64);
                }
            }
        }
        let (cost, axis, at) = best;
        // Splitting costs a box test; keep a leaf if it isn't worth it.
        if idx.len() <= 8 && (!cost.is_finite() || cost >= bounds.area() * idx.len() as f64) {
            return leaf(self);
        }
        let mid = if cost.is_finite() {
            let mut m = 0;
            for k in 0..idx.len() {
                if prims[idx[k] as usize].1[axis] < at {
                    idx.swap(k, m);
                    m += 1;
                }
            }
            m
        } else {
            idx.len() / 2
        };
        let mid = if mid == 0 || mid == idx.len() { idx.len() / 2 } else { mid };
        let (l, r) = idx.split_at_mut(mid);
        self.split(prims, l, first);
        let right = self.split(prims, r, first + mid);
        self.nodes[node].a = right as u32;
        self.nodes[node].b = axis as u32;
        let b = &self.boxes[node];
        self.nodes[node].lo = b.lo.map(|x| x as f32);
        self.nodes[node].hi = b.hi.map(|x| x as f32);
        node
    }

    /// Node boxes for the GPU (filled for leaves too).
    pub fn gpu_nodes(&self) -> Vec<NodeGpu> {
        self.nodes
            .iter()
            .zip(&self.boxes)
            .map(|(n, b)| NodeGpu { lo: b.lo.map(|x| x as f32), hi: b.hi.map(|x| x as f32), ..*n })
            .collect()
    }

    /// Nearest hit of the ray `o + t d` (t < `t_max`) with the triangles
    /// `tris`, given in leaf order.
    pub fn intersect(&self, tris: &[[V3; 3]], o: V3, d: V3, t_max: f64) -> Option<Hit> {
        let inv = d.map(|x| if x.abs() > 1e-30 { 1.0 / x } else { 1e30f64.copysign(x) });
        let mut best: Option<Hit> = None;
        let mut stack = vec![0usize];
        while let Some(i) = stack.pop() {
            let limit = best.as_ref().map_or(t_max, |h| h.t);
            if self.boxes[i].hit(o, inv, limit).is_none() {
                continue;
            }
            let n = &self.nodes[i];
            if n.b & 3 == LEAF {
                for k in n.a as usize..(n.a + (n.b >> 2)) as usize {
                    if let Some(t) = triangle(&tris[k], o, d)
                        && t < best.as_ref().map_or(t_max, |h| h.t)
                    {
                        best = Some(Hit { t, triangle: k });
                    }
                }
            } else {
                let (near, far) = if d[n.b as usize] >= 0.0 { (i + 1, n.a as usize) } else { (n.a as usize, i + 1) };
                stack.push(far);
                stack.push(near);
            }
        }
        best
    }
}

/// Möller–Trumbore, two sided; distance along `d` (not normalised).
fn triangle(t: &[V3; 3], o: V3, d: V3) -> Option<f64> {
    let e1 = vec3::sub(t[1], t[0]);
    let e2 = vec3::sub(t[2], t[0]);
    let p = vec3::cross(d, e2);
    let det = vec3::dot(e1, p);
    if det.abs() < 1e-14 {
        return None;
    }
    let inv = 1.0 / det;
    let s = vec3::sub(o, t[0]);
    let u = vec3::dot(s, p) * inv;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = vec3::cross(s, e1);
    let v = vec3::dot(d, q) * inv;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = vec3::dot(e2, q) * inv;
    (t > 1e-6).then_some(t)
}

#[cfg(test)]
mod tests {
    use super::super::mesh;
    use super::*;

    /// The BVH finds the same nearest hits as testing every triangle.
    #[test]
    fn matches_brute_force() {
        let (m, bvh) = mesh::ship();
        let tris: Vec<[V3; 3]> =
            m.triangles.iter().map(|t| t.map(|i| m.vertices[i as usize].pos.map(|x| x as f64))).collect();
        let mut hits = 0;
        for k in 0..400 {
            let a = k as f64 * 2.399_963;
            let z = 1.0 - 2.0 * (k as f64 + 0.5) / 400.0;
            let dir = [(1.0 - z * z).sqrt() * a.cos(), (1.0 - z * z).sqrt() * a.sin(), z];
            let o = vec3::scale(dir, -60.0);
            let d = vec3::normalize(vec3::sub([(k % 7) as f64 * 3.0 - 9.0, (k % 5) as f64 - 2.0, 0.0], o));
            let brute = tris.iter().filter_map(|t| triangle(t, o, d)).fold(f64::INFINITY, f64::min);
            let fast = bvh.intersect(&tris, o, d, 1e9).map_or(f64::INFINITY, |h| h.t);
            assert!((brute - fast).abs() < 1e-9 || brute == fast, "ray {k}: {brute} vs {fast}");
            hits += brute.is_finite() as usize;
        }
        assert!(hits > 100, "only {hits} hits");
        // Every triangle sits in exactly one leaf.
        let mut seen = vec![0; m.triangles.len()];
        for n in bvh.nodes.iter().filter(|n| n.b & 3 == LEAF) {
            for k in n.a..n.a + (n.b >> 2) {
                seen[k as usize] += 1;
            }
        }
        assert!(seen.iter().all(|&c| c == 1));
    }
}
