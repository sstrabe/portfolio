//! Tile generation: where each sample of a tile lies, precisely enough for
//! centimetre terrain.
//!
//! At level 22 a tile's samples are ~4 × 10⁻⁹ apart in face coordinates,
//! below what f32 resolves there (~3 × 10⁻⁸), so the GPU can't take a fine
//! tile's sample directions from `cube` in f32. Instead the CPU expands the
//! tile's points about its centre in f64, to second order in the tile
//! coordinates `(s, t) ∈ [−½, ½]²` (plus the apron):
//!
//! `R·d(s, t) − anchor ≈ o + a_s s + a_t t + ½ b_ss s² + b_st s t + ½ b_tt t²`
//!
//! with `o = R·d(0, 0) − anchor` and the coefficients from central
//! differences across the tile (all in km). The GPU then only adds small
//! numbers. The third-order remainder is ~R θ³/6 for a tile subtending θ:
//! 0.07 mm at level 12, so levels from [`EXPANDED_FROM`] use the expansion
//! and coarser tiles take their directions from `cube` directly.

use super::cube;
use super::tiles::TileId;
use kerr::vec3::{self, V3};

/// Tiles from this level are generated from the expansion.
pub const EXPANDED_FROM: u8 = 12;

/// Mirrors the GPU's tile frame: the expansion's coefficients in km, xyz
/// (w unused).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TileFrameGpu {
    pub origin: [f32; 4],
    pub a_s: [f32; 4],
    pub a_t: [f32; 4],
    pub b_ss: [f32; 4],
    pub b_st: [f32; 4],
    pub b_tt: [f32; 4],
}

/// Tile `t`'s expansion on a sphere of `radius_km`, relative to `anchor_km`
/// (body-fixed).
pub fn frame(t: TileId, radius_km: f64, anchor_km: V3) -> TileFrameGpu {
    // Points at tile coordinates (s, t) ∈ [−½, ½]² about the centre.
    let p = |s: f64, u: f64| vec3::scale(t.direction(0.5 + s, 0.5 + u), radius_km);
    let c = p(0.0, 0.0);
    let (e, w, n, s) = (p(0.5, 0.0), p(-0.5, 0.0), p(0.0, 0.5), p(0.0, -0.5));
    let (ne, nw, se, sw) = (p(0.5, 0.5), p(-0.5, 0.5), p(0.5, -0.5), p(-0.5, -0.5));
    let f = |v: V3| [v[0] as f32, v[1] as f32, v[2] as f32, 0.0];
    TileFrameGpu {
        origin: f(vec3::sub(c, anchor_km)),
        a_s: f(vec3::sub(e, w)),
        a_t: f(vec3::sub(n, s)),
        // (e − 2c + w)/h² with h = ½, and ½ of it is the s² coefficient.
        b_ss: f(vec3::scale(vec3::add(vec3::sub(e, vec3::scale(c, 2.0)), w), 4.0)),
        b_st: f(vec3::sub(vec3::add(ne, sw), vec3::add(nw, se))),
        b_tt: f(vec3::scale(vec3::add(vec3::sub(n, vec3::scale(c, 2.0)), s), 4.0)),
    }
}

/// The point at tile coordinates `(s, t)` (centred: −½ to ½), in km from
/// the anchor, computed in f32 as the GPU does.
pub fn offset(f: &TileFrameGpu, s: f32, t: f32) -> [f32; 3] {
    std::array::from_fn(|i| {
        f.origin[i]
            + (f.a_s[i] * s + f.a_t[i] * t)
            + (0.5 * f.b_ss[i] * s * s + f.b_st[i] * s * t + 0.5 * f.b_tt[i] * t * t)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::tiles::{MAX_LEVEL, TILE_SAMPLES};

    const RADIUS: f64 = 7234.0;

    /// Near the anchor, every sample (apron included) lands within a
    /// millimetre of the exact f64 point, at every expanded level.
    #[test]
    fn expansion_is_millimetre_exact() {
        let site = cube::direction(3, 0.41, -0.27);
        let half = 0.5 + 1.0 / TILE_SAMPLES as f64;
        for level in EXPANDED_FROM..=MAX_LEVEL {
            let tile = TileId::containing(site, level);
            let anchor = vec3::scale(site, RADIUS + 0.002);
            let fr = frame(tile, RADIUS, anchor);
            let mut worst = 0.0f64;
            for i in 0..=8 {
                for j in 0..=8 {
                    let (s, t) = (-half + 2.0 * half * i as f64 / 8.0, -half + 2.0 * half * j as f64 / 8.0);
                    let exact = vec3::sub(vec3::scale(tile.direction(0.5 + s, 0.5 + t), RADIUS), anchor);
                    let got = offset(&fr, s as f32, t as f32);
                    let err = vec3::norm(vec3::sub(exact, got.map(f64::from)));
                    worst = worst.max(err);
                }
            }
            assert!(worst < 1e-6, "level {level}: {:.3} mm", worst * 1e6);
        }
    }

    /// Neighbouring tiles put their shared edge's samples at the same
    /// points (to well under a sample spacing), so there are no seams.
    #[test]
    fn neighbours_agree_on_edges() {
        let site = cube::direction(0, 0.999_999, 0.2);
        for level in [EXPANDED_FROM, 18, MAX_LEVEL] {
            let a = TileId::containing(site, level);
            let b = a.neighbour(crate::terrain::tiles::Side::East);
            let anchor = vec3::scale(site, RADIUS);
            let (fa, fb) = (frame(a, RADIUS, anchor), frame(b, RADIUS, anchor));
            // a's east edge corners, found among b's corners.
            for t in [-0.5f32, 0.5] {
                let pa = offset(&fa, 0.5, t).map(f64::from);
                let best = [(-0.5f32, -0.5f32), (-0.5, 0.5), (0.5, -0.5), (0.5, 0.5)]
                    .iter()
                    .map(|&(s, u)| vec3::norm(vec3::sub(pa, offset(&fb, s, u).map(f64::from))))
                    .fold(f64::INFINITY, f64::min);
                let spacing = RADIUS * a.angular_size() / TILE_SAMPLES as f64;
                assert!(best < 1e-6 + 1e-3 * spacing, "level {level}: {best} km");
            }
        }
    }

    /// Mirrors `struct TileFrame` in the shader (six vec4s).
    #[test]
    fn frame_layout() {
        assert_eq!(std::mem::size_of::<TileFrameGpu>(), 96);
    }
}
