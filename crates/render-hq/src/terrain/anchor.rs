//! The precision anchor: lattice noise with centimetre cells on a planet
//! thousands of km across, stable as the camera moves.
//!
//! An octave of lattice noise at `f` cells per km needs the lattice
//! coordinate `p·f`: for 1 cm cells on an Earth that is ~6 × 10⁸, which f32
//! can't hold to a fraction of a cell. So the CPU keeps an anchor near the
//! camera (body-fixed km, f64) and gives the GPU, per octave, the anchor's
//! lattice cell as integers and its place within that cell. The GPU only
//! handles small offsets `d` from the anchor: the coordinate is
//! `cell + (frac + d·f)`, whose integer and fractional parts it can split
//! exactly while `d·f` stays small. Values come from hashing the integer
//! cell (wrapped to 32 bits), so a point's noise doesn't depend on which
//! anchor it was reached from: moving the anchor changes nothing.
//!
//! Precision: the fraction's error is about ulp(d·f) ≈ 6 × 10⁻⁸ d·f cells,
//! so an octave is exact to 10⁻³ cell within ~16,000 cells of the anchor
//! (160 m for 1 cm cells). Farther out a cm octave is far below a pixel
//! and is faded out anyway.
//!
//! [`noise`] is the CPU twin of the GPU's `anchored_noise`; keep them in
//! step.

use kerr::vec3::V3;

/// One octave as the GPU gets it: the anchor's lattice cell (wrapped to
/// i32) and seed, and its place in the cell with the frequency.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct OctaveGpu {
    /// Cell x, y, z; w the octave's hash seed.
    pub cell: [i32; 4],
    /// Fraction x, y, z in [0, 1); w cells per km.
    pub frac_freq: [f32; 4],
}

#[derive(Clone, Debug)]
pub struct Anchor {
    /// Body-fixed km from the planet's centre.
    pub origin_km: V3,
    pub octaves: Vec<OctaveGpu>,
}

impl Anchor {
    /// An anchor at `origin_km` for octaves at the given frequencies
    /// (cells per km) and seeds.
    pub fn new(origin_km: V3, octaves: impl IntoIterator<Item = (f64, u32)>) -> Self {
        let octaves = octaves
            .into_iter()
            .map(|(freq, seed)| {
                let x = origin_km.map(|c| c * freq);
                let cell = x.map(f64::floor);
                OctaveGpu {
                    // Wrapping to 32 bits keeps cell + k consistent: the
                    // hash sees the same bits either way.
                    cell: [cell[0] as i64 as i32, cell[1] as i64 as i32, cell[2] as i64 as i32, seed as i32],
                    frac_freq: [(x[0] - cell[0]) as f32, (x[1] - cell[1]) as f32, (x[2] - cell[2]) as f32, freq as f32],
                }
            })
            .collect();
        Self { origin_km, octaves }
    }
}

/// The lattice cell and fraction of a point `d_km` from the anchor, in f32
/// as the GPU computes it.
pub fn lattice(o: &OctaveGpu, d_km: [f32; 3]) -> ([i32; 3], [f32; 3]) {
    let mut cell = [0; 3];
    let mut frac = [0.0; 3];
    for i in 0..3 {
        let x = o.frac_freq[i] + d_km[i] * o.frac_freq[3];
        let fl = x.floor();
        cell[i] = o.cell[i].wrapping_add(fl as i32);
        frac[i] = x - fl;
    }
    (cell, frac)
}

/// PCG-style 3-D hash (Jarzynski & Olano 2020, "pcg3d").
fn pcg3d(v: [u32; 3]) -> [u32; 3] {
    let mut v = v.map(|c| c.wrapping_mul(1_664_525).wrapping_add(1_013_904_223));
    v[0] = v[0].wrapping_add(v[1].wrapping_mul(v[2]));
    v[1] = v[1].wrapping_add(v[2].wrapping_mul(v[0]));
    v[2] = v[2].wrapping_add(v[0].wrapping_mul(v[1]));
    v = v.map(|c| c ^ (c >> 16));
    v[0] = v[0].wrapping_add(v[1].wrapping_mul(v[2]));
    v[1] = v[1].wrapping_add(v[2].wrapping_mul(v[0]));
    v[2] = v[2].wrapping_add(v[0].wrapping_mul(v[1]));
    v
}

/// The gradient at a lattice corner: components in [−1, 1].
fn gradient(cell: [i32; 3], seed: u32) -> [f32; 3] {
    let h = pcg3d([cell[0] as u32 ^ seed, cell[1] as u32, cell[2] as u32 ^ seed.rotate_left(16)]);
    h.map(|c| (c >> 8) as f32 / 8_388_607.5 - 1.0)
}

/// Gradient noise (roughly −1 to 1) at lattice cell `cell` and fraction
/// `f`, with quintic fades.
pub fn noise(cell: [i32; 3], f: [f32; 3], seed: u32) -> f32 {
    let fade = |t: f32| t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
    let w = f.map(fade);
    let mut acc = [[0.0f32; 2]; 4];
    for (k, slot) in acc.iter_mut().enumerate() {
        let (j, l) = (k & 1, k >> 1);
        for (i, v) in slot.iter_mut().enumerate() {
            let corner = [i as i32, j as i32, l as i32];
            let g = gradient(
                [cell[0].wrapping_add(corner[0]), cell[1].wrapping_add(corner[1]), cell[2].wrapping_add(corner[2])],
                seed,
            );
            *v = (0..3).map(|a| g[a] * (f[a] - corner[a] as f32)).sum();
        }
    }
    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
    let x = acc.map(|[a, b]| lerp(a, b, w[0]));
    lerp(lerp(x[0], x[1], w[1]), lerp(x[2], x[3], w[1]), w[2])
}

/// Noise of one octave at `d_km` from the anchor.
pub fn anchored(o: &OctaveGpu, d_km: [f32; 3]) -> f32 {
    let (cell, frac) = lattice(o, d_km);
    noise(cell, frac, o.cell[3] as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A point on an Earth-sized planet, far from any axis.
    const SITE: V3 = [3_904.123_456_789, -2_871.987_654_321, 4_012.555_555_555];

    /// The reference: lattice coordinates in f64 from the absolute point.
    fn reference(p: V3, freq: f64, seed: u32) -> f32 {
        let x = p.map(|c| c * freq);
        let cell = x.map(f64::floor);
        let c = [cell[0] as i64 as i32, cell[1] as i64 as i32, cell[2] as i64 as i32];
        noise(c, [(x[0] - cell[0]) as f32, (x[1] - cell[1]) as f32, (x[2] - cell[2]) as f32], seed)
    }

    fn offset(a: V3, b: V3) -> [f32; 3] {
        [(b[0] - a[0]) as f32, (b[1] - a[1]) as f32, (b[2] - a[2]) as f32]
    }

    /// Centimetre cells resolve against the f64 reference near the anchor,
    /// where f32 absolute coordinates are hopeless.
    #[test]
    fn centimetre_octaves_resolve() {
        let freq = 1e5; // 1 cm cells
        let anchor = Anchor::new(SITE, [(freq, 7)]);
        let o = &anchor.octaves[0];
        let mut worst = 0.0f32;
        let mut naive_worst = 0.0f32;
        for k in 0..400 {
            // Points up to ~120 m away, stepping ~1 cm.
            let s = k as f64 * 3e-4 + 1e-5 * (k % 7) as f64;
            let p = [SITE[0] + s, SITE[1] - 0.7 * s, SITE[2] + 0.3 * s];
            let want = reference(p, freq, 7);
            worst = worst.max((anchored(o, offset(SITE, p)) - want).abs());
            let naive = p.map(|c| c as f32 * freq as f32);
            let fl = naive.map(f32::floor);
            let cell = [fl[0] as i64 as i32, fl[1] as i64 as i32, fl[2] as i64 as i32];
            let got = noise(cell, [naive[0] - fl[0], naive[1] - fl[1], naive[2] - fl[2]], 7);
            naive_worst = naive_worst.max((got - want).abs());
        }
        assert!(worst < 2e-3, "anchored error {worst}");
        assert!(naive_worst > 0.1, "naive f32 unexpectedly fine: {naive_worst}");
    }

    /// The same point through two anchors gives the same noise: moving the
    /// anchor doesn't make the terrain swim.
    #[test]
    fn anchors_agree() {
        let a = [SITE[0] + 0.03, SITE[1] + 0.01, SITE[2] - 0.02];
        let freqs = [(1.0, 1), (37.0, 2), (1e3, 3), (1e5, 4)];
        let (one, two) = (Anchor::new(SITE, freqs), Anchor::new(a, freqs));
        for k in 0..50 {
            let p = [SITE[0] + 0.001 * k as f64, SITE[1] + 0.0007 * k as f64, SITE[2] - 0.0004 * k as f64];
            for (o1, o2) in one.octaves.iter().zip(&two.octaves) {
                let (n1, n2) = (anchored(o1, offset(SITE, p)), anchored(o2, offset(a, p)));
                assert!((n1 - n2).abs() < 2e-3, "{} cells/km: {n1} vs {n2}", o1.frac_freq[3]);
            }
        }
    }

    /// Noise is continuous across cell boundaries and spans its range.
    #[test]
    fn noise_is_continuous() {
        let (mut lo, mut hi) = (0.0f32, 0.0f32);
        for k in 0..2000 {
            let t = k as f32 * 0.013;
            let cell = [t.floor() as i32, 5, -3];
            let f = [t - t.floor(), 0.37, 0.61];
            let n = noise(cell, f, 11);
            let t2 = t + 1e-4;
            let n2 = noise([t2.floor() as i32, 5, -3], [t2 - t2.floor(), 0.37, 0.61], 11);
            assert!((n - n2).abs() < 1e-2, "{t}");
            (lo, hi) = (lo.min(n), hi.max(n));
        }
        assert!(lo < -0.2 && hi > 0.2, "{lo} {hi}");
    }
}
