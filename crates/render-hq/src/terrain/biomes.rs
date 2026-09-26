//! Biomes: what the land of a living world is made of, as a table.
//!
//! Each biome says where it is, as soft windows on seven things known at
//! every point (temperature, moisture from the baked climate, height above
//! the sea, steepness, ruggedness, whether the rock is basalt or granite,
//! and the distance from the shore), and what it is: its spectral reflectance, the scanned ground
//! materials ([`super::materials`]) its surface shows, and how much of the
//! ground its plants cover. The shader weighs every biome by the product of
//! its windows times its precedence and blends them, so biomes shade into
//! each other where their windows overlap and a narrow one (a beach) wins
//! over a broad one (the forest behind it) by precedence.
//!
//! To add a biome, add a row to [`BIOMES`] (and, if it needs one, a
//! material to the library); nothing in the shaders changes.

use super::materials::{MATERIALS, MAX_MATERIALS};
use crate::spectrum::{self, BINS};

/// Spectral reflectance at the renderer's bins.
pub type Spectrum = [f32; BINS];

/// Uniform reflectance `x`.
pub fn flat(x: f32) -> Spectrum {
    [x; BINS]
}

fn smoothstep(a: f64, b: f64, x: f64) -> f64 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// 0 below `l0` nm, 1 above `l1`, smooth between (`spec_ramp`).
pub fn ramp(l0: f64, l1: f64) -> Spectrum {
    std::array::from_fn(|k| smoothstep(l0, l1, spectrum::centre(k)) as f32)
}

/// A Gaussian bump of height 1 at `mu` nm (`spec_bump`).
pub fn bump(mu: f64, sigma: f64) -> Spectrum {
    std::array::from_fn(|k| (-0.5 * ((spectrum::centre(k) - mu) / sigma).powi(2)).exp() as f32)
}

/// x·s + z.
pub fn axpy(x: Spectrum, s: f32, z: Spectrum) -> Spectrum {
    std::array::from_fn(|k| x[k] * s + z[k])
}

/// x + (y − x)·t.
pub fn mix(x: Spectrum, y: Spectrum, t: f32) -> Spectrum {
    std::array::from_fn(|k| x[k] + (y[k] - x[k]) * t)
}

pub fn scale(x: Spectrum, s: f32) -> Spectrum {
    x.map(|v| v * s)
}

// Reflectances (twins of `refl_*` in `planet.wgsl`).

pub fn basalt() -> Spectrum {
    axpy(ramp(400.0, 800.0), 0.03, flat(0.055))
}

pub fn granite() -> Spectrum {
    axpy(ramp(400.0, 750.0), 0.1, flat(0.2))
}

/// Leaves (chlorophyll's green peak and red edge), yellowing as they dry.
pub fn vegetation(dryness: f32) -> Spectrum {
    let green = axpy(bump(550.0, 28.0), 0.06, flat(0.035));
    let leaf = axpy(ramp(690.0, 740.0), 0.4, green);
    let dry = axpy(ramp(480.0, 650.0), 0.13, flat(0.06));
    mix(leaf, dry, 0.8 * dryness)
}

/// Dark loam where wet, pale quartz sand where arid.
pub fn soil(dryness: f32) -> Spectrum {
    let sand = axpy(ramp(420.0, 600.0), 0.24, flat(0.17));
    let loam = axpy(ramp(450.0, 700.0), 0.1, flat(0.06));
    mix(loam, sand, dryness)
}

/// Quartz and coral grains, pale gold.
pub fn beach_sand() -> Spectrum {
    axpy(ramp(420.0, 620.0), 0.16, flat(0.38))
}

pub fn snow() -> Spectrum {
    axpy(ramp(600.0, 800.0), -0.08, flat(0.9))
}

/// A soft window on one variable: 1 inside `[lo, hi]`, fading to 0 over
/// `soft` beyond each end.
#[derive(Clone, Copy, Debug)]
pub struct Window {
    pub lo: f32,
    pub hi: f32,
    pub soft: f32,
}

const OPEN: f32 = 1e9;

impl Window {
    /// Anything.
    pub const ANY: Window = Window { lo: -OPEN, hi: OPEN, soft: 1.0 };
    pub const fn between(lo: f32, hi: f32, soft: f32) -> Self {
        Self { lo, hi, soft }
    }
    pub const fn above(lo: f32, soft: f32) -> Self {
        Self { lo, hi: OPEN, soft }
    }
    pub const fn below(hi: f32, soft: f32) -> Self {
        Self { lo: -OPEN, hi, soft }
    }
    /// Its weight at `x`, as `biome_window` computes it (in a form that
    /// stays defined for the open ends, where lo − soft rounds to lo).
    pub fn at(&self, x: f32) -> f32 {
        let s = |t: f32| {
            let t = t.clamp(0.0, 1.0);
            t * t * (3.0 - 2.0 * t)
        };
        s((x - (self.lo - self.soft)) / self.soft) * s((self.hi + self.soft - x) / self.soft)
    }
}

/// A biome: where it is and what it's made of.
pub struct Biome {
    pub name: &'static str,
    /// Surface temperature, K.
    pub temperature: Window,
    /// The climate's moisture: the share of the ground plants could cover
    /// (0 desert, 1 rainforest).
    pub moisture: Window,
    /// Height above the sea, m.
    pub height: Window,
    /// 1 − cos(slope): 0 level, 0.3 about 45°.
    pub steepness: Window,
    /// Mountainousness (0 lowland, 1 the cores of ranges).
    pub ruggedness: Window,
    /// Rock type, 0 granite to 1 basalt.
    pub basalt: Window,
    /// Distance from the shore (m), where the regional erosion's square
    /// knows it (0 elsewhere, so beaches there go by height alone).
    pub shore: Window,
    /// Spectral reflectance (the ground and its plants, as seen from above).
    pub albedo: fn() -> Spectrum,
    /// Scanned ground materials its surface shows, by slot name, with
    /// their shares.
    pub ground: &'static [(&'static str, f32)],
    /// The share of the ground its plants cover.
    pub cover: f32,
    /// Precedence over biomes whose windows overlap it.
    pub weight: f32,
}

impl Biome {
    /// Its windows, in the shader's order.
    pub fn windows(&self) -> [Window; WINDOWS] {
        [self.temperature, self.moisture, self.height, self.steepness, self.ruggedness, self.basalt, self.shore]
    }
}

const ANY: Window = Window::ANY;
/// Warm enough for plants.
const GROWING: Window = Window::above(272.0, 8.0);

/// The biomes of living worlds.
pub static BIOMES: &[Biome] = &[
    Biome {
        name: "forest",
        temperature: GROWING,
        moisture: Window::above(0.6, 0.2),
        height: ANY,
        steepness: ANY,
        ruggedness: ANY,
        basalt: ANY,
        shore: ANY,
        // Seen from above: mostly canopy, a little soil between.
        albedo: || mix(soil(0.3), vegetation(0.1), 0.85),
        ground: &[("SOIL", 1.0)],
        cover: 0.9,
        weight: 1.0,
    },
    Biome {
        name: "grassland and savanna",
        temperature: GROWING,
        moisture: Window::between(0.25, 0.6, 0.15),
        height: ANY,
        steepness: ANY,
        ruggedness: ANY,
        basalt: ANY,
        shore: ANY,
        // Olive-tan: dry grass, scattered trees and bare ground.
        albedo: || mix(soil(0.65), vegetation(0.7), 0.45),
        ground: &[("SOIL", 1.0)],
        cover: 0.5,
        weight: 1.0,
    },
    Biome {
        name: "shrubland and steppe",
        temperature: GROWING,
        moisture: Window::between(0.08, 0.25, 0.08),
        height: ANY,
        steepness: ANY,
        ruggedness: ANY,
        basalt: ANY,
        shore: ANY,
        albedo: || mix(soil(0.85), vegetation(0.9), 0.2),
        ground: &[("SOIL", 0.7), ("SAND", 0.3)],
        cover: 0.2,
        weight: 1.0,
    },
    Biome {
        name: "desert",
        temperature: ANY,
        moisture: Window::below(0.08, 0.06),
        height: ANY,
        steepness: ANY,
        ruggedness: ANY,
        basalt: ANY,
        shore: ANY,
        albedo: || soil(1.0),
        ground: &[("SAND", 0.6), ("SOIL", 0.4)],
        cover: 0.02,
        weight: 1.0,
    },
    Biome {
        name: "tundra",
        temperature: Window::between(262.0, 272.0, 4.0),
        moisture: Window::above(0.08, 0.06),
        height: ANY,
        steepness: ANY,
        ruggedness: ANY,
        basalt: ANY,
        shore: ANY,
        albedo: || mix(vegetation(0.6), soil(0.4), 0.5),
        ground: &[("SOIL", 1.0)],
        cover: 0.5,
        weight: 1.0,
    },
    Biome {
        name: "snow and ice",
        temperature: Window::below(262.0, 9.0),
        moisture: ANY,
        height: ANY,
        steepness: ANY,
        ruggedness: ANY,
        basalt: ANY,
        shore: ANY,
        albedo: snow,
        ground: &[],
        cover: 0.0,
        weight: 3.0,
    },
    Biome {
        name: "bare basalt highlands",
        temperature: ANY,
        moisture: ANY,
        height: ANY,
        steepness: ANY,
        ruggedness: Window::above(0.85, 0.1),
        basalt: Window::above(0.5, 0.2),
        shore: ANY,
        albedo: basalt,
        ground: &[("ROCK", 1.0)],
        cover: 0.0,
        weight: 3.0,
    },
    Biome {
        name: "bare granite highlands",
        temperature: ANY,
        moisture: ANY,
        height: ANY,
        steepness: ANY,
        ruggedness: Window::above(0.85, 0.1),
        basalt: Window::below(0.5, 0.2),
        shore: ANY,
        albedo: granite,
        ground: &[("ROCK", 1.0)],
        cover: 0.0,
        weight: 3.0,
    },
    Biome {
        name: "dry rocky ranges",
        temperature: ANY,
        moisture: Window::below(0.3, 0.15),
        height: ANY,
        steepness: ANY,
        ruggedness: Window::above(0.6, 0.2),
        basalt: ANY,
        shore: ANY,
        albedo: || mix(granite(), basalt(), 0.4),
        ground: &[("ROCK", 0.7), ("SOIL", 0.3)],
        cover: 0.1,
        weight: 1.5,
    },
    Biome {
        name: "basalt cliffs",
        temperature: ANY,
        moisture: ANY,
        height: ANY,
        steepness: Window::above(0.35, 0.1),
        ruggedness: ANY,
        basalt: Window::above(0.5, 0.2),
        shore: ANY,
        albedo: basalt,
        ground: &[("ROCK", 1.0)],
        cover: 0.0,
        weight: 4.0,
    },
    Biome {
        name: "granite cliffs",
        temperature: ANY,
        moisture: ANY,
        height: ANY,
        steepness: Window::above(0.35, 0.1),
        ruggedness: ANY,
        basalt: Window::below(0.5, 0.2),
        shore: ANY,
        albedo: granite,
        ground: &[("ROCK", 1.0)],
        cover: 0.0,
        weight: 4.0,
    },
    Biome {
        name: "sand beach",
        temperature: Window::above(280.0, 5.0),
        moisture: ANY,
        height: Window::below(3.5, 1.0),
        steepness: Window::below(0.05, 0.04),
        ruggedness: ANY,
        basalt: ANY,
        shore: Window::below(60.0, 40.0),
        albedo: beach_sand,
        ground: &[("SAND", 1.0)],
        cover: 0.0,
        weight: 30.0,
    },
    Biome {
        name: "wet sand",
        temperature: Window::above(280.0, 5.0),
        moisture: ANY,
        height: Window::below(0.5, 0.3),
        steepness: Window::below(0.06, 0.04),
        ruggedness: ANY,
        basalt: ANY,
        shore: Window::below(60.0, 40.0),
        albedo: || scale(beach_sand(), 0.55),
        ground: &[("WET_SAND", 1.0)],
        cover: 0.0,
        weight: 200.0,
    },
    Biome {
        name: "shingle beach",
        temperature: Window::below(280.0, 5.0),
        moisture: ANY,
        height: Window::below(3.0, 1.0),
        steepness: Window::below(0.05, 0.04),
        ruggedness: ANY,
        basalt: ANY,
        shore: Window::below(60.0, 40.0),
        albedo: || mix(granite(), basalt(), 0.5),
        ground: &[("ROCK", 0.5), ("SOIL", 0.5)],
        cover: 0.0,
        weight: 30.0,
    },
];

/// Room in the shaders' biome table.
pub const MAX_BIOMES: usize = 24;

/// The windows a biome has, in the shader's order.
pub const WINDOWS: usize = 7;

/// Mirrors `struct Biome` in `planet.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BiomeGpu {
    /// Temperature, moisture, height, steepness, ruggedness, basalt, shore:
    /// lo, hi, soft, unused.
    pub windows: [[f32; 4]; WINDOWS],
    pub albedo: [f32; BINS],
    /// Share of each ground material, by slot.
    pub ground: [f32; MAX_MATERIALS],
    /// Cover, weight, unused, unused.
    pub cover_weight: [f32; 4],
}

/// Mirrors `struct Biomes` in `planet.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BiomesGpu {
    /// Count, unused ×3.
    pub count: [u32; 4],
    pub biomes: [BiomeGpu; MAX_BIOMES],
}

/// The table for the GPU. Panics on a ground material that isn't in the
/// library (a typo in a row), so it fails at start-up, not silently.
pub fn table() -> BiomesGpu {
    assert!(BIOMES.len() <= MAX_BIOMES, "too many biomes");
    let mut t: BiomesGpu = bytemuck::Zeroable::zeroed();
    t.count[0] = BIOMES.len() as u32;
    for (slot, b) in t.biomes.iter_mut().zip(BIOMES) {
        let w = |w: Window| [w.lo, w.hi, w.soft, 0.0];
        slot.windows = b.windows().map(w);
        slot.albedo = (b.albedo)();
        for &(name, share) in b.ground {
            let i = MATERIALS
                .iter()
                .position(|m| m.slot == name)
                .unwrap_or_else(|| panic!("biome {}: no ground material {name}", b.name));
            slot.ground[i] += share;
        }
        slot.cover_weight = [b.cover, b.weight, 0.0, 0.0];
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The table builds (every material named exists), windows are soft,
    /// and reflectances are physical.
    #[test]
    fn table_is_sound() {
        let t = table();
        assert_eq!(t.count[0] as usize, BIOMES.len());
        for b in BIOMES {
            for w in b.windows() {
                assert!(w.soft > 0.0 && w.lo <= w.hi, "{}", b.name);
            }
            assert!((b.albedo)().iter().all(|&r| (0.0..=1.0).contains(&r)), "{}", b.name);
            assert!(b.weight > 0.0 && (0.0..=1.0).contains(&b.cover), "{}", b.name);
        }
        assert_eq!(std::mem::size_of::<BiomeGpu>(), 224);
    }

    /// The twins match the shader's recipes at a few bins (values worked
    /// out from `planet.wgsl`).
    #[test]
    fn recipes() {
        // Basalt: 0.055 + 0.03·ramp(400, 800): flat-ish, dark.
        let b = basalt();
        assert!((b[0] - 0.055).abs() < 0.002 && (b[15] - 0.085).abs() < 0.003, "{b:?}");
        // Leaves reflect more at 765 nm (past the red edge) than at 665.
        let v = vegetation(0.0);
        assert!(v[15] > 5.0 * v[11], "{v:?}");
    }

    /// A beach point is mostly beach; the forest behind it mostly forest.
    #[test]
    fn beach_beats_forest_where_both_fit() {
        let weight =
            |b: &Biome, v: [f32; WINDOWS]| b.weight * b.windows().iter().zip(v).map(|(w, x)| w.at(x)).product::<f32>();
        let share = |name: &str, v: [f32; WINDOWS]| {
            let total: f32 = BIOMES.iter().map(|b| weight(b, v)).sum();
            BIOMES.iter().filter(|b| b.name == name).map(|b| weight(b, v)).sum::<f32>() / total
        };
        // 298 K, wet, 2 m up, level, lowland, basalt, 20 m from the shore.
        assert!(share("sand beach", [298.0, 0.8, 2.0, 0.01, 0.1, 0.9, 20.0]) > 0.9);
        assert!(share("forest", [298.0, 0.8, 40.0, 0.05, 0.1, 0.9, 500.0]) > 0.9);
        assert!(share("wet sand", [298.0, 0.8, 0.2, 0.01, 0.1, 0.9, 5.0]) > 0.8);
        assert!(share("basalt cliffs", [298.0, 0.8, 80.0, 0.6, 0.3, 0.9, 300.0]) > 0.7);
        // A valley floor just above the sea but a kilometre inland is no
        // beach.
        assert!(share("sand beach", [298.0, 0.8, 2.0, 0.01, 0.1, 0.9, 1000.0]) < 0.05);
    }
}
