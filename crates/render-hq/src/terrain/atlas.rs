//! The tile atlas's bookkeeping: which tile lives in which layer of the
//! tile texture arrays, and which tiles to generate next.
//!
//! Layers are recycled least recently used first, never one drawn this
//! frame. While a tile is missing, its nearest resident ancestor stands in
//! (so generation goes coarse to fine: a parent is always there first).
//! Each frame generates at most a budget of tiles, coarsest and then
//! nearest first.

use super::tiles::TileId;
use std::collections::HashMap;

#[derive(Clone, Copy, Debug)]
struct Slot {
    layer: u32,
    last_used: u64,
}

pub struct Atlas {
    capacity: u32,
    slots: HashMap<TileId, Slot>,
    free: Vec<u32>,
    frame: u64,
}

impl Atlas {
    pub fn new(capacity: u32) -> Self {
        Self { capacity, slots: HashMap::new(), free: (0..capacity).rev().collect(), frame: 1 }
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    pub fn resident(&self) -> usize {
        self.slots.len()
    }

    pub fn begin_frame(&mut self) {
        self.frame += 1;
    }

    /// The layer holding `t`, marking it used this frame.
    pub fn get(&mut self, t: TileId) -> Option<u32> {
        let frame = self.frame;
        self.slots.get_mut(&t).map(|s| {
            s.last_used = frame;
            s.layer
        })
    }

    /// The layer holding `t`, without marking it used.
    pub fn peek(&self, t: TileId) -> Option<u32> {
        self.slots.get(&t).map(|s| s.layer)
    }

    /// The layer of `t` or, if it isn't resident, of its nearest resident
    /// ancestor, with the tile found (marked used this frame).
    pub fn get_or_ancestor(&mut self, t: TileId) -> Option<(TileId, u32)> {
        let mut at = Some(t);
        while let Some(a) = at {
            if let Some(layer) = self.get(a) {
                return Some((a, layer));
            }
            at = a.parent();
        }
        None
    }

    /// A layer for `t` (already resident: its own), evicting the least
    /// recently used tile not used this frame if the atlas is full. Returns
    /// the layer and the evicted tile, or `None` if every layer is in use
    /// this frame.
    pub fn insert(&mut self, t: TileId) -> Option<(u32, Option<TileId>)> {
        if let Some(layer) = self.get(t) {
            return Some((layer, None));
        }
        let (layer, evicted) = match self.free.pop() {
            Some(layer) => (layer, None),
            None => {
                let (&old, slot) = self
                    .slots
                    .iter()
                    .filter(|(_, s)| s.last_used < self.frame)
                    .min_by_key(|(k, s)| (s.last_used, std::cmp::Reverse(k.level)))?;
                let layer = slot.layer;
                self.slots.remove(&old);
                (layer, Some(old))
            }
        };
        self.slots.insert(t, Slot { layer, last_used: self.frame });
        Some((layer, evicted))
    }

    /// The tiles to generate this frame for drawing `wanted` (ordered
    /// nearest first): missing tiles and their missing ancestors, coarsest
    /// first, then in the order wanted; at most `budget`.
    pub fn plan(&self, wanted: &[TileId], budget: usize) -> Vec<TileId> {
        let mut missing: Vec<(u8, usize, TileId)> = Vec::new();
        for (order, &t) in wanted.iter().enumerate() {
            let mut at = Some(t);
            while let Some(a) = at {
                if self.slots.contains_key(&a) {
                    break;
                }
                missing.push((a.level, order, a));
                at = a.parent();
            }
        }
        missing.sort();
        let mut out: Vec<TileId> = Vec::with_capacity(budget);
        for (_, _, t) in missing {
            if out.len() == budget {
                break;
            }
            if !out.contains(&t) {
                out.push(t);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tile(level: u8, x: u32) -> TileId {
        TileId { face: 1, level, x, y: 0 }
    }

    #[test]
    fn evicts_the_least_recently_used() {
        let mut atlas = Atlas::new(3);
        for x in 0..3 {
            assert_eq!(atlas.insert(tile(4, x)).unwrap().1, None);
            atlas.begin_frame();
        }
        // Touch the oldest; the next oldest goes.
        atlas.get(tile(4, 0));
        atlas.begin_frame();
        let (layer, evicted) = atlas.insert(tile(4, 9)).unwrap();
        assert_eq!(evicted, Some(tile(4, 1)));
        assert_eq!(atlas.get(tile(4, 9)), Some(layer));
        assert_eq!(atlas.get(tile(4, 1)), None);
        assert_eq!(atlas.resident(), 3);
    }

    #[test]
    fn never_evicts_what_this_frame_uses() {
        let mut atlas = Atlas::new(2);
        atlas.insert(tile(3, 0));
        atlas.insert(tile(3, 1));
        assert_eq!(atlas.insert(tile(3, 2)), None);
        atlas.begin_frame();
        atlas.get(tile(3, 0));
        assert_eq!(atlas.insert(tile(3, 2)).unwrap().1, Some(tile(3, 1)));
    }

    #[test]
    fn ancestors_stand_in() {
        let mut atlas = Atlas::new(8);
        let root = TileId::root(1);
        atlas.insert(root);
        let deep = TileId { face: 1, level: 5, x: 17, y: 3 };
        assert_eq!(atlas.get_or_ancestor(deep), Some((root, 0)));
        let mid = deep.parent().unwrap().parent().unwrap();
        let (layer, _) = atlas.insert(mid).unwrap();
        assert_eq!(atlas.get_or_ancestor(deep), Some((mid, layer)));
        assert_eq!(atlas.get_or_ancestor(TileId::root(2)), None);
    }

    #[test]
    fn plans_coarse_first_within_budget() {
        let mut atlas = Atlas::new(64);
        atlas.insert(TileId::root(1));
        let a = TileId { face: 1, level: 3, x: 5, y: 2 };
        let b = TileId { face: 1, level: 2, x: 0, y: 0 };
        let plan = atlas.plan(&[a, b], 10);
        // a's missing ancestors (levels 1, 2) and b come before a; no
        // duplicates; the resident root isn't planned.
        assert_eq!(plan.first().map(|t| t.level), Some(1));
        assert!(plan.iter().position(|&t| t == a) > plan.iter().position(|&t| t == b));
        assert!(!plan.contains(&TileId::root(1)));
        let mut sorted = plan.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), plan.len());
        assert_eq!(atlas.plan(&[a, b], 2).len(), 2);
    }
}
