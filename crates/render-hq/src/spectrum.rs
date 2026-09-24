//! Spectral sampling shared by the CPU and the shaders.
//!
//! Light is carried as 16 bins of 25 nm from 390 to 790 nm. The bin edges
//! are chosen so the nebular lines land in separate bins: Hβ 486 and
//! [O III] 501/496 in 465–490 and 490–515, Hα 656 (with [N II] 658) in
//! 640–665 and [S II] 672 in 665–690. That is what a narrowband (Hubble
//! palette) rendering needs; true colour integrates the bins against the
//! CIE 1931 colour-matching functions.
//!
//! Radiance is in W m⁻² sr⁻¹ nm⁻¹ (bin average). A source moving with
//! Doppler/gravitational factor g = ν_obs/ν_emit is seen at
//! λ_emit = g λ_obs with I_λ,obs(λ) = g⁵ I_λ,emit(g λ); a blackbody at T
//! is then exactly a blackbody at g T.

pub const BINS: usize = 16;
pub const LAMBDA_MIN: f64 = 390.0;
pub const BIN_WIDTH: f64 = 25.0;

/// Centre wavelength of bin `k`, nm.
pub fn centre(k: usize) -> f64 {
    LAMBDA_MIN + BIN_WIDTH * (k as f64 + 0.5)
}

/// CIE 1931 2° colour-matching functions, multi-lobe fit of Wyman, Sloan &
/// Shirley (2013), accurate to about 1% of the peak.
pub fn cmf(lambda: f64) -> [f64; 3] {
    fn g(x: f64, mu: f64, s1: f64, s2: f64) -> f64 {
        let t = (x - mu) / if x < mu { s1 } else { s2 };
        (-0.5 * t * t).exp()
    }
    let x = 1.056 * g(lambda, 599.8, 37.9, 31.0) + 0.362 * g(lambda, 442.0, 16.0, 26.7)
        - 0.065 * g(lambda, 501.1, 20.4, 26.2);
    let y = 0.821 * g(lambda, 568.8, 46.9, 40.5) + 0.286 * g(lambda, 530.9, 16.3, 31.1);
    let z = 1.217 * g(lambda, 437.0, 11.8, 36.0) + 0.681 * g(lambda, 459.0, 26.0, 13.8);
    [x, y, z]
}

/// CIE XYZ → linear sRGB (D65).
const XYZ_TO_RGB: [[f64; 3]; 3] = [
    [3.240_454_2, -1.537_138_5, -0.498_531_4],
    [-0.969_266_0, 1.876_010_8, 0.041_556_0],
    [0.055_643_4, -0.204_025_9, 1.057_225_2],
];

/// Bin-averaged colour-matching functions times the bin width: XYZ of a
/// spectrum is `Σ_k S_k w_k`.
pub fn xyz_weights() -> [[f64; 3]; BINS] {
    let mut w = [[0.0; 3]; BINS];
    for (k, wk) in w.iter_mut().enumerate() {
        let lo = LAMBDA_MIN + BIN_WIDTH * k as f64;
        let steps = 50;
        for i in 0..steps {
            let c = cmf(lo + BIN_WIDTH * (i as f64 + 0.5) / steps as f64);
            for j in 0..3 {
                wk[j] += c[j] * BIN_WIDTH / steps as f64;
            }
        }
    }
    w
}

/// Per-bin weights giving linear sRGB directly (possibly negative for
/// saturated wavelengths, as sRGB cannot show them).
pub fn rgb_weights() -> [[f64; 3]; BINS] {
    let xyz = xyz_weights();
    let mut out = [[0.0; 3]; BINS];
    for k in 0..BINS {
        for c in 0..3 {
            out[k][c] = (0..3).map(|j| XYZ_TO_RGB[c][j] * xyz[k][j]).sum();
        }
    }
    out
}

/// Planck's law, W m⁻² sr⁻¹ nm⁻¹.
pub fn planck(lambda_nm: f64, t: f64) -> f64 {
    let x = (1.438_777e7 / (lambda_nm * t)).min(700.0);
    1.191_042_972e20 / (lambda_nm.powi(5) * x.exp_m1())
}

/// Stefan–Boltzmann constant, W m⁻² K⁻⁴.
pub const SIGMA_SB: f64 = 5.670_374e-8;

/// Binned spectrum of a blackbody at `t`.
pub fn blackbody(t: f64) -> [f64; BINS] {
    std::array::from_fn(|k| planck(centre(k), t))
}

/// XYZ of a binned spectrum.
pub fn to_xyz(s: &[f64; BINS]) -> [f64; 3] {
    let w = xyz_weights();
    let mut out = [0.0; 3];
    for k in 0..BINS {
        for j in 0..3 {
            out[j] += s[k] * w[k][j];
        }
    }
    out
}

/// Luminance-like Y of a blackbody whose bolometric flux is 1 W/m²
/// (the Y per watt of such light).
pub fn y_per_watt(t: f64) -> f64 {
    let s = blackbody(t);
    // Bolometric: ∫ B_λ dλ = σT⁴/π.
    to_xyz(&s)[1] / (SIGMA_SB * t.powi(4) / std::f64::consts::PI)
}

/// GPU layout of the conversion table: three rows (R, G, B) of 16 weights.
pub fn rgb_weight_rows() -> [[f32; 4]; 12] {
    let w = rgb_weights();
    let mut out = [[0.0f32; 4]; 12];
    for c in 0..3 {
        for k in 0..BINS {
            out[c * 4 + k / 4][k % 4] = w[k][c] as f32;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cmf_integrals_are_close_to_the_tabulated_ones() {
        // ∫ x̄ dλ ≈ ∫ ȳ dλ ≈ ∫ z̄ dλ ≈ 106.9 nm for the 1931 observer (the
        // 390–790 range misses a little of each).
        let w = xyz_weights();
        for j in 0..3 {
            let s: f64 = w.iter().map(|k| k[j]).sum();
            assert!((s / 106.9 - 1.0).abs() < 0.05, "channel {j}: {s}");
        }
    }

    #[test]
    fn a_6500_k_blackbody_is_nearly_neutral() {
        let s = blackbody(6504.0);
        let w = rgb_weights();
        let rgb: Vec<f64> = (0..3).map(|c| (0..BINS).map(|k| s[k] * w[k][c]).sum()).collect();
        let max = rgb.iter().cloned().fold(0.0, f64::max);
        for (c, v) in rgb.iter().enumerate() {
            assert!((v / max - 1.0).abs() < 0.08, "channel {c}: {rgb:?}");
        }
        // Cool stars are red, hot ones blue.
        let red = blackbody(3000.0);
        let blue = blackbody(20000.0);
        let r = |s: &[f64; BINS], c: usize| (0..BINS).map(|k| s[k] * w[k][c]).sum::<f64>();
        assert!(r(&red, 0) > 2.0 * r(&red, 2));
        assert!(r(&blue, 2) > r(&blue, 0));
    }

    #[test]
    fn planck_matches_the_sun() {
        // The Sun's surface at 500 nm: about 2.6e4 W m⁻² sr⁻¹ nm⁻¹.
        let b = planck(500.0, 5772.0);
        assert!((b / 2.62e4 - 1.0).abs() < 0.02, "{b}");
        // Visible fraction of sunlight luminance: Y per W is sensible.
        let y = y_per_watt(5772.0);
        assert!(y > 0.1 && y < 0.6, "{y}");
    }
}
