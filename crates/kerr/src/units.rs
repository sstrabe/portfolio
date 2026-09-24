//! Physical units for a hole of Sagittarius A*'s mass.
//!
//! The simulation works in geometrized units of the hole's mass `M`
//! (`G = c = 1`). For Sgr A*, `M = 4.3 × 10⁶ M☉`, so:

/// Hole mass in solar masses.
pub const SGR_A_MASS_MSUN: f64 = 4.3e6;
/// One `M` of length in metres (`GM/c²`).
pub const METRES_PER_M: f64 = 1.476_625e3 * SGR_A_MASS_MSUN;
/// One `M` of time in seconds (`GM/c³`).
pub const SECONDS_PER_M: f64 = METRES_PER_M / 299_792_458.0;
/// A solar mass in units of `M`.
pub const MSUN: f64 = 1.0 / SGR_A_MASS_MSUN;
/// Astronomical unit in `M`.
pub const AU: f64 = 1.495_978_707e11 / METRES_PER_M;
/// Parsec in `M`.
pub const PARSEC: f64 = 3.085_677_581e16 / METRES_PER_M;
/// Solar radius in `M`.
pub const RSUN: f64 = 6.957e8 / METRES_PER_M;
/// Standard gravity as a proper acceleration in units of `1/M`.
pub const G0: f64 = 9.806_65 * METRES_PER_M / (299_792_458.0 * 299_792_458.0);
/// Apparent bolometric magnitude of the Sun seen from 1 AU.
pub const SUN_MAG_1AU: f64 = -26.83;

/// Flux (in the renderer's `L☉ / M²` units) of a source of apparent
/// magnitude `mag`.
pub fn flux_of_magnitude(mag: f64) -> f64 {
    10f64.powf(-0.4 * (mag - SUN_MAG_1AU)) / (AU * AU)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sgr_a_scales() {
        assert!((SECONDS_PER_M - 21.2).abs() < 0.1);
        assert!((AU - 23.56).abs() < 0.05);
        assert!((PARSEC / 4.86e6 - 1.0).abs() < 0.01);
        assert!((RSUN - 0.1096).abs() < 0.001);
        // A Sun-like star at 1 AU is magnitude −26.8.
        assert!((flux_of_magnitude(SUN_MAG_1AU) * AU * AU - 1.0).abs() < 1e-12);
    }
}
