//! Boulders scattered on the ground (`wgsl/boulders.wgsl`): the face-grid
//! cells they sit in and the tile level that raises them into the
//! heightfield.

use super::tilegen::{EXPANDED_FROM, spacing_km};
use super::tiles::{MAX_LEVEL, TILE_SAMPLES};

/// About how big a cell is (m): at most one boulder a cell, up to 0.46
/// cell in radius.
pub const CELL_M: f64 = 3.0;
const SEED: u32 = 7300;

/// The face-grid level whose tiles are the cells: the one nearest
/// [`CELL_M`] across.
pub fn cell_level(radius_km: f64) -> u8 {
    let tile_m = |l: u8| spacing_km(radius_km, l) * TILE_SAMPLES as f64 * 1000.0;
    (0..=MAX_LEVEL)
        .min_by(|&a, &b| (tile_m(a) / CELL_M).ln().abs().total_cmp(&(tile_m(b) / CELL_M).ln().abs()))
        .unwrap_or(MAX_LEVEL)
}

/// The tile level that raises them: the first whose samples are at most
/// 0.25 m apart, so the smallest spans a few.
pub fn level(radius_km: f64) -> u8 {
    (EXPANDED_FROM..=MAX_LEVEL).find(|&l| spacing_km(radius_km, l) <= 0.25e-3).unwrap_or(MAX_LEVEL)
}

/// The boulders' hash seed for a planet.
pub fn seed(planet_seed: u32) -> u32 {
    planet_seed.wrapping_add(SEED)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// On an Earth the boulders come in at level 19 (17 cm samples), in
    /// cells of level 22 (2.4 m).
    #[test]
    fn boulder_levels() {
        assert_eq!(level(6371.0), 19);
        assert_eq!(cell_level(6371.0), 22);
        assert!(spacing_km(7234.0, level(7234.0)) <= 0.25e-3);
    }
}
