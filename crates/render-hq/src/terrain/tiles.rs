//! Addressing for the terrain tile pyramid: a quadtree on each face of the
//! cube-sphere (`cube.rs`).
//!
//! Tile `(face, level, x, y)` covers `u ∈ [−1 + 2x/2^level, −1 + 2(x+1)/2^level]`
//! and likewise `v` with `y`. Level 0 is a whole face (a quarter of a
//! great circle on a side); each level halves the side, so with
//! [`TILE_SAMPLES`] samples per side a tile at level 22 on an Earth has
//! samples ~2.4 cm apart. Because every face uses the same tangent warp,
//! tile edges on neighbouring faces line up exactly at every level, so a
//! tile's neighbours are tiles of the same level even across face edges.
//!
//! [`select`] picks the tiles to draw: it refines the quadtree while a
//! tile's sample spacing looks larger than a given number of pixels from
//! the eye, and drops tiles hidden below the horizon.

use super::cube;
use kerr::vec3::{self, V3};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};
use std::f64::consts::FRAC_PI_2;

/// Samples along a tile's side (the apron comes on top).
pub const TILE_SAMPLES: u32 = 128;

/// The finest level: ~6 mm samples on an Earth.
pub const MAX_LEVEL: u8 = 24;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TileId {
    pub face: u8,
    pub level: u8,
    pub x: u32,
    pub y: u32,
}

/// A tile's sides, in its face's (u, v).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    /// −u
    West,
    /// +u
    East,
    /// −v
    South,
    /// +v
    North,
}

pub const SIDES: [Side; 4] = [Side::West, Side::East, Side::South, Side::North];

impl TileId {
    pub fn root(face: u8) -> Self {
        Self { face, level: 0, x: 0, y: 0 }
    }

    pub fn roots() -> [Self; 6] {
        std::array::from_fn(|f| Self::root(f as u8))
    }

    /// Tiles per side of a face at this tile's level.
    fn span(self) -> u32 {
        1 << self.level
    }

    pub fn parent(self) -> Option<Self> {
        (self.level > 0).then(|| Self { face: self.face, level: self.level - 1, x: self.x / 2, y: self.y / 2 })
    }

    /// The four children: (x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)
    /// at the next level.
    pub fn children(self) -> [Self; 4] {
        let (level, x, y) = (self.level + 1, 2 * self.x, 2 * self.y);
        [(0, 0), (1, 0), (0, 1), (1, 1)].map(|(i, j)| Self { face: self.face, level, x: x + i, y: y + j })
    }

    /// Whether `other` is this tile or lies inside it.
    pub fn contains_tile(self, other: Self) -> bool {
        other.face == self.face
            && other.level >= self.level
            && other.x >> (other.level - self.level) == self.x
            && other.y >> (other.level - self.level) == self.y
    }

    /// The (u, v) at fractions (s, t) ∈ [0, 1]² across the tile.
    pub fn uv(self, s: f64, t: f64) -> (f64, f64) {
        let size = 2.0 / self.span() as f64;
        (-1.0 + size * (self.x as f64 + s), -1.0 + size * (self.y as f64 + t))
    }

    /// Unit direction at fractions (s, t) across the tile (may lie outside
    /// [0, 1] for aprons).
    pub fn direction(self, s: f64, t: f64) -> V3 {
        let (u, v) = self.uv(s, t);
        cube::direction(self.face as usize, u, v)
    }

    pub fn centre(self) -> V3 {
        self.direction(0.5, 0.5)
    }

    /// The corners, counter-clockwise from (u0, v0).
    pub fn corners(self) -> [V3; 4] {
        [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)].map(|(s, t)| self.direction(s, t))
    }

    /// Angle (rad) from the centre to the farthest corner.
    pub fn angular_radius(self) -> f64 {
        let c = self.centre();
        self.corners().iter().map(|&k| vec3::dot(c, k).clamp(-1.0, 1.0).acos()).fold(0.0, f64::max)
    }

    /// Nominal side length as an angle (rad): a quarter great circle per
    /// face, halved per level (the warp keeps the real side within ~1.2 ×).
    pub fn angular_size(self) -> f64 {
        FRAC_PI_2 / self.span() as f64
    }

    /// The tile at `level` containing (u, v) on `face`.
    pub fn at(face: u8, u: f64, v: f64, level: u8) -> Self {
        let span = 1u32 << level;
        let index = |c: f64| (((c + 1.0) * 0.5 * span as f64).floor().max(0.0) as u32).min(span - 1);
        Self { face, level, x: index(u), y: index(v) }
    }

    /// The tile at `level` containing direction `d`.
    pub fn containing(d: V3, level: u8) -> Self {
        let (face, u, v) = cube::face_uv(d);
        Self::at(face as u8, u, v, level)
    }

    /// The tile of the same level across `side`, on the neighbouring face
    /// when the side is a face edge.
    pub fn neighbour(self, side: Side) -> Self {
        let span = self.span();
        let (dx, dy): (i64, i64) = match side {
            Side::West => (-1, 0),
            Side::East => (1, 0),
            Side::South => (0, -1),
            Side::North => (0, 1),
        };
        let (nx, ny) = (self.x as i64 + dx, self.y as i64 + dy);
        if (0..span as i64).contains(&nx) && (0..span as i64).contains(&ny) {
            return Self { x: nx as u32, y: ny as u32, ..self };
        }
        // Across a face edge: the middle of the tile just beyond it, found
        // through its direction.
        let d = self.direction(0.5 + dx as f64, 0.5 + dy as f64);
        Self::containing(d, self.level)
    }
}

/// What [`select`] refines for.
pub struct Query<'a> {
    /// The eye, body-fixed km from the planet's centre.
    pub eye_km: V3,
    pub radius_km: f64,
    /// Terrain lies within this of the datum.
    pub relief_km: f64,
    /// Radians per pixel.
    pub pixel_angle: f64,
    /// Refine until a tile's samples are at most this many pixels apart.
    pub max_px: f64,
    pub max_tiles: usize,
    /// A tile's heights (lowest, highest; km from the datum) where known.
    pub band: &'a dyn Fn(TileId) -> Option<(f64, f64)>,
}

/// The tiles to draw: the quadtree refined until each tile's sample
/// spacing, seen from the eye, is at most `max_px` pixels, leaving out
/// tiles below the horizon. At most `max_tiles` (refinement stops once the
/// budget would be exceeded, nearest tiles first). The result is a set of
/// leaves: no tile in it contains another.
///
/// A tile's distance combines how far its footprint is along the ground
/// with how far the eye is above or below its band of heights (the
/// planet's whole relief where the band isn't known yet), so standing on a
/// mountain refines the ground underfoot as finely as standing on a beach.
pub fn select(q: &Query) -> Vec<TileId> {
    let (radius_km, relief_km) = (q.radius_km, q.relief_km);
    let r_eye = vec3::norm(q.eye_km);
    let up = vec3::scale(q.eye_km, 1.0 / r_eye.max(1e-9));
    let altitude = r_eye - radius_km;
    let (low, high) = (radius_km - relief_km, radius_km + relief_km);
    // Angle from the eye's nadir to the farthest visible terrain: over the
    // horizon of the lowest ground, up to the highest peaks behind it.
    let horizon = if r_eye > low { (low / r_eye).acos() + (low / high).acos() } else { std::f64::consts::PI };
    // Angle between unit vectors, accurate when tiny.
    let angle = |a: V3, b: V3| vec3::norm(vec3::cross(a, b)).atan2(vec3::dot(a, b));
    let visible = |t: TileId| angle(up, t.centre()) <= horizon + t.angular_radius();
    let distance = |t: TileId| {
        let along = radius_km * (angle(up, t.centre()) - t.angular_radius()).max(0.0);
        let (lo, hi) = (q.band)(t).unwrap_or((-relief_km, relief_km));
        let vertical = if altitude > hi {
            altitude - hi
        } else if altitude < lo {
            lo - altitude
        } else {
            0.0
        };
        along.hypot(vertical).max(1e-6)
    };
    let (max_px, pixel_angle, max_tiles) = (q.max_px, q.pixel_angle, q.max_tiles);
    let wants_split = |t: TileId| {
        let spacing = radius_km * t.angular_size() / TILE_SAMPLES as f64;
        t.level < MAX_LEVEL && spacing / distance(t) > max_px * pixel_angle
    };

    // Split nearest-first while the budget allows. Distances are positive,
    // so their bit patterns order like the values.
    let mut leaves: HashSet<TileId> = TileId::roots().into_iter().filter(|&t| visible(t)).collect();
    let mut queue = BinaryHeap::new();
    let offer = |queue: &mut BinaryHeap<_>, t: TileId| {
        if wants_split(t) {
            queue.push((Reverse(distance(t).to_bits()), t));
        }
    };
    for &t in &leaves {
        offer(&mut queue, t);
    }
    while let Some((_, t)) = queue.pop() {
        let kids: Vec<TileId> = t.children().into_iter().filter(|&k| visible(k)).collect();
        if leaves.len() - 1 + kids.len() > max_tiles {
            break;
        }
        leaves.remove(&t);
        for k in kids {
            leaves.insert(k);
            offer(&mut queue, k);
        }
    }
    let mut out: Vec<TileId> = leaves.into_iter().collect();
    out.sort();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const EARTH: f64 = 6371.0;

    fn samples() -> Vec<TileId> {
        let mut out = TileId::roots().to_vec();
        for (face, level, x, y) in [(0, 1, 1, 0), (2, 3, 7, 5), (4, 5, 0, 31), (5, 8, 200, 17), (3, 20, 1 << 19, 12345)]
        {
            out.push(TileId { face, level, x, y });
        }
        out
    }

    #[test]
    fn parents_and_children_agree() {
        for t in samples() {
            for c in t.children() {
                assert_eq!(c.parent(), Some(t));
                assert!(t.contains_tile(c) && !c.contains_tile(t));
            }
            // The children tile the parent: their corners are the parent's
            // corners and its edge midpoints.
            let [a, b, c, d] = t.children();
            let (u0, v0) = t.uv(0.0, 0.0);
            let (u1, v1) = t.uv(1.0, 1.0);
            assert_eq!(a.uv(0.0, 0.0), (u0, v0));
            assert_eq!(d.uv(1.0, 1.0), (u1, v1));
            assert_eq!(b.uv(0.0, 0.0), a.uv(1.0, 0.0));
            assert_eq!(c.uv(0.0, 0.0), a.uv(0.0, 1.0));
        }
        assert_eq!(TileId::root(3).parent(), None);
    }

    #[test]
    fn containing_finds_the_tile() {
        for t in samples() {
            let d = t.direction(0.3, 0.8);
            assert_eq!(TileId::containing(d, t.level), t);
        }
    }

    /// Neighbours are mutual, and share an edge: two corners in common.
    #[test]
    fn neighbours_share_edges_across_faces() {
        let mut edge_crossings = 0;
        for t in samples() {
            for side in SIDES {
                let n = t.neighbour(side);
                assert_eq!(n.level, t.level);
                assert_ne!(n, t);
                edge_crossings += (n.face != t.face) as usize;
                assert!(SIDES.iter().any(|&s| n.neighbour(s) == t), "{t:?} {side:?} → {n:?}");
                let shared = t
                    .corners()
                    .iter()
                    .filter(|&&a| n.corners().iter().any(|&b| vec3::norm(vec3::sub(a, b)) < 1e-12))
                    .count();
                assert_eq!(shared, 2, "{t:?} {side:?} → {n:?}");
            }
        }
        assert!(edge_crossings >= 24, "{edge_crossings}");
    }

    fn query<'a>(eye: V3, pixel: f64, band: &'a dyn Fn(TileId) -> Option<(f64, f64)>) -> Query<'a> {
        Query { eye_km: eye, radius_km: EARTH, relief_km: 9.0, pixel_angle: pixel, max_px: 2.0, max_tiles: 1500, band }
    }

    /// On a 5 km summit the ground underfoot is refined as finely as at
    /// sea level once the tiles' heights are known (and nearly so before).
    #[test]
    fn selection_on_a_summit() {
        let foot = cube::direction(1, -0.4, 0.1);
        let eye = vec3::scale(foot, EARTH + 5.2017);
        let finest_under = |band: &dyn Fn(TileId) -> Option<(f64, f64)>| {
            let tiles = select(&query(eye, 1e-3, band));
            check_leaves(&tiles);
            tiles.iter().find(|t| t.contains_tile(TileId::containing(foot, MAX_LEVEL))).map(|t| t.level).unwrap()
        };
        let known = finest_under(&|_| Some((5.195, 5.2005)));
        let unknown = finest_under(&|_| None);
        assert!(known >= 20, "{known}");
        assert!(unknown >= 16, "{unknown}");
    }

    fn check_leaves(tiles: &[TileId]) {
        for (i, a) in tiles.iter().enumerate() {
            for b in &tiles[i + 1..] {
                assert!(!a.contains_tile(*b) && !b.contains_tile(*a), "{a:?} {b:?}");
            }
        }
    }

    /// Standing on the ground: centimetre tiles underfoot, coarse ones far
    /// off, nothing from the far side, within budget.
    #[test]
    fn selection_on_the_ground() {
        let foot = cube::direction(2, 0.3, -0.2);
        let eye = vec3::scale(foot, EARTH + 0.0017);
        let pixel = 1e-3;
        let tiles = select(&query(eye, pixel, &|_| None));
        check_leaves(&tiles);
        assert!(tiles.len() <= 1500);
        let under =
            tiles.iter().find(|t| t.contains_tile(TileId::containing(foot, MAX_LEVEL))).expect("tile underfoot");
        let spacing_m = 1000.0 * EARTH * under.angular_size() / TILE_SAMPLES as f64;
        assert!(spacing_m < 0.01, "{under:?}: {spacing_m} m");
        let far = TileId::containing(vec3::scale(foot, -1.0), 0);
        assert!(!tiles.iter().any(|t| far.contains_tile(*t)));
        assert!(tiles.iter().any(|t| t.level <= 6), "coarse tiles far away");
    }

    /// From orbit: few, coarse tiles, but the visible hemisphere is covered.
    #[test]
    fn selection_from_orbit() {
        let eye = [0.0, 0.0, EARTH + 20_000.0];
        let tiles = select(&query(eye, 1e-3, &|_| None));
        check_leaves(&tiles);
        assert!(tiles.iter().all(|t| t.level <= 6), "{:?}", tiles.iter().map(|t| t.level).max());
        // Every direction the eye can see lies in a selected tile.
        for k in 0..200 {
            let a = k as f64 * 2.399;
            let theta = 1.2 * (k as f64 / 200.0);
            let d = [theta.sin() * a.cos(), theta.sin() * a.sin(), theta.cos()];
            assert!(tiles.iter().any(|t| t.contains_tile(TileId::containing(d, MAX_LEVEL))), "{d:?}");
        }
    }
}
