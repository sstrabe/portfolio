//! The terrain field: keeps the tile atlas filled around the eye for the
//! planet that has the baked maps.
//!
//! Each frame it anchors the noise near the eye (re-anchoring when the eye
//! has moved [`REANCHOR_KM`]; tiles made from another anchor stay valid,
//! the noise being anchor-independent), selects the tiles the view needs
//! ([`tiles::select`]), generates missing ones within a budget, coarse and
//! near first, and lists what to draw: each selected tile, or its nearest
//! resident ancestor while it is still missing.

use super::anchor::Anchor;
use super::ground::GroundCache;
use super::maps::MapKey;
use super::tilegen::{self, Surface, TileGen};
use super::tiles::{self, TileId};
use kerr::planets::Planet;
use kerr::vec3::{self, V3};
use std::collections::HashSet;

/// Tiles generated per frame at most.
pub const TILE_BUDGET: usize = 24;

/// Tiles selected for a view at most: well inside the atlas, so the tiles
/// in view never evict each other.
pub const MAX_SELECTED: usize = 900;

/// Refine until a tile's samples are at most this many pixels apart.
pub const MAX_PX: f64 = 3.0;

/// Re-anchor when the eye is this far (km) from the anchor: well within
/// the ~16,000 cells where the finest octaves stay exact.
pub const REANCHOR_KM: f64 = 0.1;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FieldStats {
    pub selected: usize,
    pub generated: usize,
    pub resident: usize,
    /// Finest level among the tiles drawn.
    pub finest: u8,
    /// Selected tiles still drawn from an ancestor.
    pub standing_in: usize,
}

pub struct TerrainField {
    pub tile_gen: TileGen,
    anchor: Option<(MapKey, Anchor)>,
    /// What to draw this frame: tiles and their atlas layers.
    pub drawn: Vec<(TileId, u32)>,
    pub stats: FieldStats,
    /// The ground around the eye on the CPU (for landing and walking).
    pub ground: GroundCache,
}

impl TerrainField {
    /// `rt`: build BLASes and a TLAS of the drawn tiles.
    pub fn new(device: &wgpu::Device, rt: bool) -> Self {
        Self {
            tile_gen: TileGen::new(device, rt),
            anchor: None,
            drawn: Vec::new(),
            stats: FieldStats::default(),
            ground: GroundCache::default(),
        }
    }

    /// The anchor of the noise and of the TLAS (body-fixed km).
    pub fn anchor_km(&self) -> Option<V3> {
        self.anchor.as_ref().map(|(_, a)| a.origin_km)
    }

    /// Update for an eye at `eye_km` (body-fixed, from the centre of the
    /// planet, which `key` identifies) and a view of `pixel_angle` rad per
    /// pixel.
    pub fn update(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        (key, planet): (MapKey, &Planet),
        eye_km: V3,
        pixel_angle: f64,
        profiler: Option<&mut crate::profile::Profiler>,
    ) {
        let reanchor = match &self.anchor {
            Some((k, a)) => *k != key || vec3::norm(vec3::sub(a.origin_km, eye_km)) > REANCHOR_KM,
            None => true,
        };
        if reanchor {
            let octaves = tilegen::octaves(planet.radius_km, planet.seed);
            self.anchor = Some((key, Anchor::new(eye_km, octaves)));
        }
        let Some((_, anchor)) = &self.anchor else { return };

        let r = planet.radius_km;
        let tile_gen = &self.tile_gen;
        let band = |t: TileId| tile_gen.band(t);
        let mut wanted = tiles::select(&tiles::Query {
            eye_km,
            radius_km: r,
            relief_km: planet.relief_km,
            pixel_angle,
            max_px: MAX_PX,
            max_tiles: MAX_SELECTED,
            band: &band,
        });
        // Nearest first.
        let distance = |t: &TileId| vec3::norm(vec3::sub(vec3::scale(t.centre(), r), eye_km));
        wanted.sort_by(|a, b| distance(a).total_cmp(&distance(b)));

        self.tile_gen.atlas.begin_frame();
        let plan = self.tile_gen.atlas.plan(&wanted, TILE_BUDGET);
        let made = self.tile_gen.generate(device, queue, Surface { key, planet, anchor }, &plan, profiler);

        self.drawn.clear();
        let mut shown_already = HashSet::new();
        let mut standing_in = 0;
        for &t in &wanted {
            if let Some((shown, layer)) = self.tile_gen.atlas.get_or_ancestor(t) {
                standing_in += (shown != t) as usize;
                if shown_already.insert(shown) {
                    self.drawn.push((shown, layer));
                }
            }
        }
        self.tile_gen.poll_ranges(device, queue);
        self.ground.update(device, queue, key, &self.tile_gen, &self.drawn, vec3::normalize(eye_km));
        if let Some(accel) = &mut self.tile_gen.accel {
            let mut enc =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("terrain tlas") });
            accel.build_tlas(&mut enc, &self.drawn, r, anchor.origin_km);
            queue.submit([enc.finish()]);
        }
        self.stats = FieldStats {
            selected: wanted.len(),
            generated: made.len(),
            resident: self.tile_gen.atlas.resident(),
            finest: self.drawn.iter().map(|d| d.0.level).max().unwrap_or(0),
            standing_in,
        };
    }
}
