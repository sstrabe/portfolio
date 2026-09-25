//! The optics the image is seen through: an eye, a camera or a telescope.
//!
//! Each has an entrance pupil whose Fraunhofer diffraction pattern is the
//! point-spread function (PSF): at wavelength λ, a source's light arrives
//! at angle θ from its image with density
//!
//! ```text
//! PSF_λ(θ) = |Â(2π θ / λ)|² / (λ² A)          (per steradian)
//! ```
//!
//! where Â is the Fourier transform of the pupil and A its area. Parseval's
//! theorem makes ∫ PSF dΩ = 1 exactly, so the PSF only redistributes
//! energy. Straight pupil edges (iris blades, spider vanes) throw light
//! into spikes perpendicular to them, falling off as θ⁻²; because the
//! pattern scales with λ, the spikes and rings carry colour fringes.
//! Scattering by dust and micro-scratches on the optics (or, in the eye,
//! by the lens and cornea) adds a wide halo.
//!
//! This module holds the pupil transforms in double precision (the shader
//! `wgsl/psf.wgsl` is their port and builds the kernel on the GPU), the
//! parameters of the three optical systems, and the lens-ghost table.
//!
//! Scale: the pupils are small (a fraction of a millimetre for the camera
//! and the telescope), so that the diffraction pattern spans a few pixels,
//! as it does in a stopped-down wide-angle shot on a high-resolution sensor
//! (the reference photographs of the Sun over Earth's limb). Larger pupils
//! would push the spikes and fringes below a pixel.

use crate::spectrum;
use std::f64::consts::PI;

/// What the pilot sees through. `P` cycles them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Optics {
    /// The dark-adapting human eye: a round pupil, glare and the ciliary
    /// corona, rod vision (desaturation, Purkinje shift) in dim light.
    Eye,
    /// A camera with a 7-bladed iris: 14 spikes, lens ghosts, auto exposure.
    #[default]
    Camera,
    /// A telescope with a secondary mirror on four spider vanes (4 spikes),
    /// fixed long exposures, true colour or the Hubble palette.
    Astro,
}

impl Optics {
    pub fn next(self) -> Self {
        match self {
            Self::Eye => Self::Camera,
            Self::Camera => Self::Astro,
            Self::Astro => Self::Eye,
        }
    }

    pub fn index(self) -> u32 {
        self as u32
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Eye => "eye",
            Self::Camera => "camera",
            Self::Astro => "astrograph",
        }
    }

    pub fn pupil(self) -> Pupil {
        match self {
            Self::Eye => Pupil::Disc,
            Self::Camera => Pupil::Polygon { blades: 7, rotation: 0.3 },
            Self::Astro => Pupil::Telescope { obstruction: 0.33, vane: 0.01 },
        }
    }

    /// Entrance pupil diameter, m.
    pub fn diameter(self) -> f64 {
        match self {
            // A photopic eye pupil. Its diffraction is far below a pixel;
            // the eye's glare comes from scattering instead.
            Self::Eye => 3.0e-3,
            Self::Camera => 0.4e-3,
            Self::Astro => 0.3e-3,
        }
    }

    /// Fraction of the light scattered into the wide halo, and the halo's
    /// angular core θ₀ (rad), for the camera and the telescope (the eye's
    /// glare follows the CIE disability-glare function in the shader).
    pub fn scatter(self) -> (f64, f64) {
        match self {
            Self::Eye => (0.0, 1.0e-2),
            Self::Camera => (0.012, 4.0e-3),
            Self::Astro => (0.004, 3.0e-3),
        }
    }

    /// Angular σ of the point-spread core (the central lobe fitted by a
    /// Gaussian, σ ≈ 0.42 λ/D at 550 nm), rad. The eye's is set by its
    /// aberrations: about one arcminute across.
    pub fn core_sigma(self) -> f64 {
        match self {
            Self::Eye => 1.2e-4,
            _ => 0.42 * 550e-9 / self.diameter(),
        }
    }

    pub fn ghosts(self) -> bool {
        self == Self::Camera
    }
}

/// Colours the image is shown in.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Palette {
    /// CIE 1931 colour matching: what an eye or a colour sensor records.
    #[default]
    True,
    /// The Hubble (SHO) palette of narrowband filters: [S II] 672 nm as
    /// red, Hα 656 nm as green, [O III] 501 nm as blue.
    Hubble,
}

/// Shapes of entrance pupil, in units of the pupil diameter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Pupil {
    Disc,
    /// A regular polygon (straight iris blades) inscribed in the unit
    /// diameter circle.
    Polygon {
        blades: u32,
        rotation: f64,
    },
    /// A circular mirror with a central obstruction (diameter fraction) held
    /// by four vanes of the given width, crossing along the axes.
    Telescope {
        obstruction: f64,
        vane: f64,
    },
}

type C = [f64; 2];

fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-4 { 1.0 - x * x / 6.0 } else { x.sin() / x }
}

/// Bessel function J₁ (for the disc's transform).
fn bessel_j1(x: f64) -> f64 {
    // Power series is fine in f64 for the arguments used in tests; the
    // shader uses a rational approximation instead.
    if x.abs() < 12.0 {
        let mut term = x / 2.0;
        let mut sum = term;
        let q = -x * x / 4.0;
        for k in 1..60 {
            term *= q / (k as f64 * (k as f64 + 1.0));
            sum += term;
            if term.abs() < 1e-17 * sum.abs() {
                break;
            }
        }
        sum
    } else {
        // Hankel asymptotic expansion (three terms).
        let z = 8.0 / x.abs();
        let y = z * z;
        let xx = x.abs() - 0.75 * PI;
        let p = 1.0 + y * (0.183_105e-2 + y * (-0.351_639_649_6e-4 + y * (0.245_752_017_4e-5 - y * 0.240_337_019e-6)));
        let q = 0.046_874_999_95
            + y * (-0.200_269_087_3e-3 + y * (0.844_919_909_6e-5 + y * (-0.882_289_87e-6 + y * 0.105_787_412e-6)));
        let v = (std::f64::consts::FRAC_2_PI / x.abs()).sqrt() * (xx.cos() * p - z * xx.sin() * q);
        if x < 0.0 { -v } else { v }
    }
}

/// Transform of a disc of radius `a` (real).
fn disc_ft(q: C, a: f64) -> f64 {
    let k = (q[0] * q[0] + q[1] * q[1]).sqrt() * a;
    if k < 1e-6 { PI * a * a } else { 2.0 * PI * a * a * bessel_j1(k) / k }
}

/// Transform of a w × l rectangle along unit axis `d` (length l along d),
/// centred at `c`.
fn rect_ft(q: C, d: C, l: f64, w: f64, c: C) -> C {
    let qd = q[0] * d[0] + q[1] * d[1];
    let qp = -q[0] * d[1] + q[1] * d[0];
    let m = l * w * sinc(qd * l / 2.0) * sinc(qp * w / 2.0);
    let ph = q[0] * c[0] + q[1] * c[1];
    [m * ph.cos(), -m * ph.sin()]
}

impl Pupil {
    /// Area in units of the diameter squared.
    pub fn area(&self) -> f64 {
        match *self {
            Pupil::Disc => PI / 4.0,
            Pupil::Polygon { blades, .. } => {
                let n = blades as f64;
                0.5 * n * 0.25 * (2.0 * PI / n).sin()
            }
            Pupil::Telescope { obstruction: e, vane } => PI / 4.0 * (1.0 - e * e) - 4.0 * vane * (1.0 - e) / 2.0,
        }
    }

    /// Â(q) = ∫ e^{−i q·x} d²x over the pupil (x in diameters).
    pub fn ft(&self, q: C) -> C {
        match *self {
            Pupil::Disc => [disc_ft(q, 0.5), 0.0],
            Pupil::Polygon { blades, rotation } => {
                // Divergence theorem: Â = (i/|q|²) ∮ (q·n) e^{−i q·x} ds;
                // each straight edge integrates to a sinc.
                let q2 = q[0] * q[0] + q[1] * q[1];
                if q2 < 1e-8 {
                    return [self.area(), 0.0];
                }
                let n = blades as usize;
                let v = |m: usize| {
                    let a = rotation + 2.0 * PI * m as f64 / n as f64;
                    [0.5 * a.cos(), 0.5 * a.sin()]
                };
                let mut sum = [0.0, 0.0];
                for m in 0..n {
                    let (a, b) = (v(m), v((m + 1) % n));
                    let e = [b[0] - a[0], b[1] - a[1]];
                    let l = (e[0] * e[0] + e[1] * e[1]).sqrt();
                    let t = [e[0] / l, e[1] / l];
                    let nrm = [t[1], -t[0]];
                    let mid = [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0];
                    let s = (q[0] * nrm[0] + q[1] * nrm[1]) * l * sinc((q[0] * t[0] + q[1] * t[1]) * l / 2.0);
                    let ph = q[0] * mid[0] + q[1] * mid[1];
                    // i · e^{−iφ} = sin φ + i cos φ
                    sum[0] += s * ph.sin();
                    sum[1] += s * ph.cos();
                }
                [sum[0] / q2, sum[1] / q2]
            }
            Pupil::Telescope { obstruction: e, vane } => {
                let mut re = disc_ft(q, 0.5) - disc_ft(q, 0.5 * e);
                let mut im = 0.0;
                let l = 0.5 * (1.0 - e);
                let c = 0.25 * (1.0 + e);
                for d in [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]] {
                    let r = rect_ft(q, d, l, vane, [c * d[0], c * d[1]]);
                    re -= r[0];
                    im -= r[1];
                }
                [re, im]
            }
        }
    }

    /// Diffracted energy per square pixel at `r` pixels from the image,
    /// for a pixel scale β = D δ / λ (pupil diameter over wavelength times
    /// pixel angle): β² |Â(2π β r)|² / A. Sums to 1 over the plane.
    pub fn psf(&self, beta: f64, r: C) -> f64 {
        let f = self.ft([2.0 * PI * beta * r[0], 2.0 * PI * beta * r[1]]);
        beta * beta * (f[0] * f[0] + f[1] * f[1]) / self.area()
    }
}

/// Fraction of the light within `theta_max` of a Harvey-type scatter halo
/// ∝ (1 + (θ/θ₀)²)^(−s/2), times 1/(π θ₀²): the normalisation of the halo
/// density. The shader uses s = 2.5.
pub fn harvey_norm(theta0: f64, theta_max: f64, s: f64) -> f64 {
    let u = (theta_max / theta0).powi(2);
    PI * theta0 * theta0 * (1.0 - (1.0 + u).powf(1.0 - s / 2.0)) / (s / 2.0 - 1.0)
}

/// Reflectance of a glass surface (n = 1.52) with a single-layer MgF₂
/// antireflection coating (n = 1.38), a quarter wave thick at `design_nm`,
/// at normal incidence: thin-film interference makes it lowest at the
/// design wavelength, so reflections take a purple or green tint.
pub fn coating_reflectance(lambda_nm: f64, design_nm: f64) -> f64 {
    let (n0, n1, n2) = (1.0, 1.38, 1.52);
    let r01 = (n0 - n1) / (n0 + n1);
    let r12 = (n1 - n2) / (n1 + n2);
    // Phase 2δ with δ = 2π n₁ d / λ and d = design / (4 n₁).
    let two_delta = PI * design_nm / lambda_nm;
    let e = [two_delta.cos(), -two_delta.sin()];
    let num = [r01 + r12 * e[0], r12 * e[1]];
    let den = [1.0 + r01 * r12 * e[0], r01 * r12 * e[1]];
    (num[0] * num[0] + num[1] * num[1]) / (den[0] * den[0] + den[1] * den[1])
}

/// One lens ghost: light reflected back and forth between two coated
/// surfaces, imaged as a defocused copy of the iris.
#[derive(Clone, Copy, Debug)]
pub struct Ghost {
    /// Position along the line through the image centre: the ghost sits at
    /// `k` times the source's offset (negative: mirrored).
    pub k: f64,
    /// Angular radius, rad.
    pub radius: f64,
    /// Design wavelengths of the two reflecting surfaces' coatings, nm.
    pub coatings: [f64; 2],
}

pub const GHOSTS: [Ghost; 7] = [
    Ghost { k: -0.35, radius: 0.020, coatings: [520.0, 560.0] },
    Ghost { k: -0.62, radius: 0.045, coatings: [480.0, 600.0] },
    Ghost { k: -1.05, radius: 0.090, coatings: [550.0, 540.0] },
    Ghost { k: -1.45, radius: 0.030, coatings: [600.0, 610.0] },
    Ghost { k: 0.42, radius: 0.012, coatings: [500.0, 530.0] },
    Ghost { k: 0.72, radius: 0.060, coatings: [570.0, 470.0] },
    Ghost { k: -0.18, radius: 0.140, coatings: [530.0, 530.0] },
];

/// Per-channel (linear sRGB) fraction of a white source's energy that a
/// ghost carries: the product of the two coatings' reflectances, averaged
/// over each channel's spectral weight.
pub fn ghost_tint(g: &Ghost) -> [f64; 3] {
    let w = spectrum::xyz_weights();
    let mut xyz = [0.0; 3];
    let mut norm = [0.0; 3];
    for (k, wk) in w.iter().enumerate() {
        let l = spectrum::centre(k);
        let r = coating_reflectance(l, g.coatings[0]) * coating_reflectance(l, g.coatings[1]);
        for c in 0..3 {
            xyz[c] += r * wk[c];
            norm[c] += wk[c];
        }
    }
    let xyz = [xyz[0] / norm[0], xyz[1] / norm[1], xyz[2] / norm[2]];
    // Equal-energy XYZ (1,1,1) is white; map the ratios through XYZ → sRGB
    // relative to that white.
    let to_rgb = |v: [f64; 3]| spectrum::xyz_to_rgb(v);
    let white = to_rgb([1.0, 1.0, 1.0]);
    let t = to_rgb(xyz);
    [0, 1, 2].map(|c| (t[c] / white[c]).max(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Brute-force Â(q) by summing over a fine grid of the pupil.
    fn brute(p: &Pupil, q: C) -> C {
        let inside = |x: f64, y: f64| -> bool {
            let r = (x * x + y * y).sqrt();
            match *p {
                Pupil::Disc => r <= 0.5,
                Pupil::Polygon { blades, rotation } => {
                    let n = blades as f64;
                    let a = (y.atan2(x) - rotation).rem_euclid(2.0 * PI / n) - PI / n;
                    r * a.cos() <= 0.5 * (PI / n).cos()
                }
                Pupil::Telescope { obstruction, vane } => {
                    let in_vane = (x.abs() <= vane / 2.0 || y.abs() <= vane / 2.0) && r <= 0.5;
                    r <= 0.5 && r >= 0.5 * obstruction && !in_vane
                }
            }
        };
        let n = 1200;
        let h = 1.0 / n as f64;
        let mut s = [0.0, 0.0];
        for i in 0..n {
            for j in 0..n {
                let (x, y) = (-0.5 + (i as f64 + 0.5) * h, -0.5 + (j as f64 + 0.5) * h);
                if inside(x, y) {
                    let ph = q[0] * x + q[1] * y;
                    s[0] += ph.cos() * h * h;
                    s[1] -= ph.sin() * h * h;
                }
            }
        }
        s
    }

    const PUPILS: [Pupil; 4] = [
        Pupil::Disc,
        Pupil::Polygon { blades: 7, rotation: 0.3 },
        Pupil::Polygon { blades: 6, rotation: 0.0 },
        Pupil::Telescope { obstruction: 0.33, vane: 0.025 },
    ];

    #[test]
    fn pupil_transforms_match_brute_force_integration() {
        for p in PUPILS {
            let a = p.ft([0.0, 0.0]);
            assert!((a[0] / p.area() - 1.0).abs() < 1e-9, "{p:?}");
            for q in [[3.0, 1.0], [-7.0, 12.0], [25.0, -4.0], [0.0, 40.0]] {
                let (f, b) = (p.ft(q), brute(&p, q));
                let err = ((f[0] - b[0]).powi(2) + (f[1] - b[1]).powi(2)).sqrt();
                assert!(err < 3e-3 * p.area(), "{p:?} q={q:?}: {f:?} vs {b:?}");
            }
        }
    }

    #[test]
    fn psf_carries_unit_energy() {
        // Sum over a 512² pixel grid, 3×3 samples per pixel: what is missing
        // is the light diffracted beyond 256 px, about 1/(β r) of a percent.
        for p in PUPILS {
            let beta = 0.8;
            let half = 256i32;
            let mut sum = 0.0;
            for y in -half..half {
                for x in -half..half {
                    let mut e = 0.0;
                    for sy in 0..3 {
                        for sx in 0..3 {
                            let r = [x as f64 + (sx as f64 - 1.0) / 3.0, y as f64 + (sy as f64 - 1.0) / 3.0];
                            e += p.psf(beta, r);
                        }
                    }
                    sum += e / 9.0;
                }
            }
            assert!(sum < 1.001 && sum > 0.985, "{p:?}: {sum}");
        }
    }

    #[test]
    fn bessel_matches_tabulated_values() {
        for (x, j) in [(1.0, 0.440_050_585_7), (3.831_705_97, 0.0), (10.0, 0.043_472_746_2), (20.0, 0.066_833_124_6)] {
            assert!((bessel_j1(x) - j).abs() < 2e-7, "J1({x}) = {}", bessel_j1(x));
        }
    }

    #[test]
    fn harvey_halo_normalises() {
        // Numerical ∫ 2πθ (1+(θ/θ₀)²)^(−1.25) dθ.
        let (t0, tmax) = (4e-3, 1.0);
        let n = 200_000;
        let h = tmax / n as f64;
        let s: f64 = (0..n)
            .map(|i| {
                let t = (i as f64 + 0.5) * h;
                2.0 * PI * t * (1.0 + (t / t0).powi(2)).powf(-1.25) * h
            })
            .sum();
        assert!((s / harvey_norm(t0, tmax, 2.5) - 1.0).abs() < 1e-3);
    }

    #[test]
    fn coatings_suppress_their_design_wavelength() {
        let r = |l| coating_reflectance(l, 550.0);
        assert!(r(550.0) < 0.015 && r(420.0) > r(550.0) && r(700.0) > r(550.0));
        // Uncoated glass reflects 4.3%.
        let bare = ((1.52f64 - 1.0) / 2.52).powi(2);
        assert!(r(550.0) < bare / 3.0);
        for g in &GHOSTS {
            let t = ghost_tint(g);
            assert!(t.iter().all(|&v| (0.0..1e-3).contains(&v)), "{t:?}");
        }
    }
}
