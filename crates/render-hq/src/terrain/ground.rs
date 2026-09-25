//! The ground under the eye, on the CPU: heights of the tiles around it
//! read back from the atlas without stalling, and interpolated on the very
//! triangles the ray-tracing hardware traces (`rt.rs`), so what the physics
//! stands on is what is drawn (to ~1 mm on fine tiles).
//!
//! A tile's heights depend only on its id (the noise is anchor-independent),
//! so a tile read once stays valid until the planet changes.

use super::rt::{MESH_N, MESH_STEP};
use super::tilegen::{APRON, TILE_TEXELS, TileGen};
use super::tiles::{SIDES, TileId};
use super::{cube, maps::MapKey};
use kerr::vec3::V3;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Tiles kept on the CPU at most.
const MAX_CACHED: usize = 64;
/// Read-backs in flight at most.
const MAX_PENDING: usize = 9;

/// A tile's mesh-vertex heights (km), `(MESH_N + 1)²`, row by row.
type MeshHeights = Arc<Vec<f32>>;

struct Pending {
    tile: TileId,
    buffer: wgpu::Buffer,
    ready: Arc<AtomicBool>,
}

#[derive(Default)]
pub struct GroundCache {
    planet: Option<MapKey>,
    /// The planet's spin axis, for the physics' body-fixed convention.
    spin: V3,
    tiles: HashMap<TileId, MeshHeights>,
    pending: Vec<Pending>,
}

impl GroundCache {
    /// Collect finished read-backs and ask for the tiles around `eye_dir`
    /// (body-fixed unit vector): the finest drawn tile under it and its
    /// neighbours at that level where they are drawn too.
    pub fn update(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        (key, spin): (MapKey, V3),
        tile_gen: &TileGen,
        drawn: &[(TileId, u32)],
        eye_dir: V3,
    ) {
        if self.planet != Some(key) {
            *self = Self { planet: Some(key), ..Default::default() };
        }
        self.spin = spin;
        let _ = device.poll(wgpu::PollType::Poll);
        let mut still = Vec::new();
        for p in self.pending.drain(..) {
            if !p.ready.load(Ordering::Acquire) {
                still.push(p);
                continue;
            }
            if let Ok(data) = p.buffer.slice(..).get_mapped_range() {
                let row = (TILE_TEXELS * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) as usize / 4;
                let texels: &[f32] = bytemuck::cast_slice(&data);
                let n1 = MESH_N as usize + 1;
                let at = |i: usize, j: usize| {
                    let (x, y) = (i * MESH_STEP as usize + APRON as usize, j * MESH_STEP as usize + APRON as usize);
                    texels[y * row + x]
                };
                let heights: Vec<f32> = (0..n1 * n1).map(|k| at(k % n1, k / n1)).collect();
                self.tiles.insert(p.tile, Arc::new(heights));
            }
            p.buffer.unmap();
        }
        self.pending = still;

        let Some(&(under, _)) = drawn
            .iter()
            .filter(|(t, _)| t.contains_tile(TileId::containing(eye_dir, t.level.max(1))))
            .max_by_key(|(t, _)| t.level)
        else {
            return;
        };
        let wanted: Vec<(TileId, u32)> = std::iter::once(under)
            .chain(SIDES.iter().map(|&s| under.neighbour(s)))
            .filter_map(|t| drawn.iter().find(|d| d.0 == t).copied())
            .collect();
        for (tile, layer) in wanted.iter().copied() {
            if self.pending.len() >= MAX_PENDING
                || self.tiles.contains_key(&tile)
                || self.pending.iter().any(|p| p.tile == tile)
            {
                continue;
            }
            self.pending.push(read_layer(device, queue, tile_gen, tile, layer));
        }
        if self.tiles.len() > MAX_CACHED {
            let keep: Vec<TileId> = wanted.iter().map(|w| w.0).collect();
            self.tiles.retain(|t, _| keep.contains(t) || t.level + 2 < under.level);
            while self.tiles.len() > MAX_CACHED {
                let Some(&drop) = self.tiles.keys().find(|t| !keep.contains(t)) else { break };
                self.tiles.remove(&drop);
            }
        }
    }

    /// The ground's height (km above the datum) at body-fixed unit
    /// direction `q`, from the finest cached tile holding it.
    pub fn height_km(&self, q: V3) -> Option<f64> {
        height_in(&self.tiles, q)
    }

    /// A copy for the physics: the heights are shared, so it's cheap.
    pub fn snapshot(&self) -> Option<GroundSnapshot> {
        let (star, generation, planet) = self.planet?;
        (!self.tiles.is_empty()).then(|| GroundSnapshot {
            planet: kerr::local::PlanetRef { star, generation, planet },
            axes: super::body_axes(self.spin, 0.0),
            tiles: self.tiles.clone(),
        })
    }
}

/// The ground as the physics sees it ([`kerr::world::Ground`]).
pub struct GroundSnapshot {
    planet: kerr::local::PlanetRef,
    /// The renderer's body axes at rotation angle 0: the physics' body-fixed
    /// directions (planet-frame axes turned back by the rotation) are
    /// turned into the renderer's with these.
    axes: [V3; 3],
    tiles: HashMap<TileId, MeshHeights>,
}

impl kerr::world::Ground for GroundSnapshot {
    fn planet(&self) -> kerr::local::PlanetRef {
        self.planet
    }

    fn height_km(&self, q: V3) -> Option<f64> {
        height_in(&self.tiles, super::to_body(&self.axes, q))
    }
}

fn height_in(tiles: &HashMap<TileId, MeshHeights>, q: V3) -> Option<f64> {
    let (face, u, v) = cube::face_uv(q);
    let (tile, heights) = tiles
        .iter()
        .filter(|(t, _)| t.contains_tile(TileId::at(face as u8, u, v, t.level)))
        .max_by_key(|(t, _)| t.level)?;
    Some(mesh_height(*tile, heights, u, v))
}

/// Height on a tile's mesh at face coordinates (u, v): the grid quad's two
/// triangles as `rt::mesh_indices` splits them.
fn mesh_height(tile: TileId, heights: &[f32], u: f64, v: f64) -> f64 {
    let (u0, v0) = tile.uv(0.0, 0.0);
    let (u1, v1) = tile.uv(1.0, 1.0);
    let n = MESH_N as f64;
    let x = ((u - u0) / (u1 - u0) * n).clamp(0.0, n);
    let y = ((v - v0) / (v1 - v0) * n).clamp(0.0, n);
    let (i, j) = ((x.floor() as usize).min(MESH_N as usize - 1), (y.floor() as usize).min(MESH_N as usize - 1));
    let (fx, fy) = (x - i as f64, y - j as f64);
    let n1 = MESH_N as usize + 1;
    let h = |a: usize, b: usize| heights[(j + b) * n1 + i + a] as f64;
    if fx >= fy {
        // (i, j), (i + 1, j), (i + 1, j + 1)
        h(0, 0) + fx * (h(1, 0) - h(0, 0)) + fy * (h(1, 1) - h(1, 0))
    } else {
        // (i, j), (i + 1, j + 1), (i, j + 1)
        h(0, 0) + fy * (h(0, 1) - h(0, 0)) + fx * (h(1, 1) - h(0, 1))
    }
}

/// Start reading a layer's heights back.
fn read_layer(device: &wgpu::Device, queue: &wgpu::Queue, tile_gen: &TileGen, tile: TileId, layer: u32) -> Pending {
    let row = (TILE_TEXELS * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ground read-back"),
        size: (row * TILE_TEXELS) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("ground read-back") });
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &tile_gen.height,
            mip_level: 0,
            origin: wgpu::Origin3d { x: 0, y: 0, z: layer },
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: None },
        },
        wgpu::Extent3d { width: TILE_TEXELS, height: TILE_TEXELS, depth_or_array_layers: 1 },
    );
    queue.submit([enc.finish()]);
    let ready = Arc::new(AtomicBool::new(false));
    let flag = ready.clone();
    buffer.slice(..).map_async(wgpu::MapMode::Read, move |r| flag.store(r.is_ok(), Ordering::Release));
    Pending { tile, buffer, ready }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The interpolation reproduces the vertex heights and is continuous
    /// across the quad's diagonal.
    #[test]
    fn mesh_height_matches_vertices() {
        let tile = TileId { face: 2, level: 10, x: 300, y: 511 };
        let n1 = MESH_N as usize + 1;
        let heights: Vec<f32> = (0..n1 * n1).map(|k| ((k % n1) as f32 * 0.37).sin() + (k / n1) as f32 * 0.01).collect();
        let (u0, v0) = tile.uv(0.0, 0.0);
        let (u1, v1) = tile.uv(1.0, 1.0);
        let uv = |x: f64, y: f64| (u0 + (u1 - u0) * x / MESH_N as f64, v0 + (v1 - v0) * y / MESH_N as f64);
        for (i, j) in [(0, 0), (5, 7), (63, 64), (64, 64), (20, 0)] {
            let (u, v) = uv(i as f64, j as f64);
            let h = mesh_height(tile, &heights, u, v);
            assert!((h - heights[j * n1 + i] as f64).abs() < 1e-5, "{i} {j}");
        }
        // Along the diagonal both triangles agree.
        let (u, v) = uv(10.3, 10.3);
        let (ua, va) = uv(10.3 + 1e-9, 10.3);
        let (ub, vb) = uv(10.3, 10.3 + 1e-9);
        let (a, b, c) = (
            mesh_height(tile, &heights, u, v),
            mesh_height(tile, &heights, ua, va),
            mesh_height(tile, &heights, ub, vb),
        );
        assert!((a - b).abs() < 1e-6 && (a - c).abs() < 1e-6);
    }
}
