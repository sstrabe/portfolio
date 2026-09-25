//! Where the nebulae around Sgr A* are, how big, how they move and how
//! bright they are. Everything here is in parsecs relative to the hole, in
//! the renderer's coordinates, and SI radiometry; `nebula.rs` turns it into
//! GPU parameters (units of M).
//!
//! **Orientation.** The Galactic frame is the one the procedural sky uses
//! (`sky_radiance` in `far.wgsl`): `x` points away from the Sun (the Sun is
//! 8.18 kpc along −x), `y` towards Galactic longitude 90° and `z` to the
//! north Galactic pole. Structures are placed from their positions on the
//! sky as seen from Earth (arcseconds east and north of Sgr A*) plus a depth
//! along the line of sight. At Sgr A*, Galactic north lies at position angle
//! −58.6° (the pole at α = 192.86°, δ = +27.13° seen from α = 266.42°,
//! δ = −29.01°), so increasing longitude runs 31.4° east of north.
//!
//! **What is here** (sizes are the real ones; the inner few parsecs of the
//! Galaxy, Genzel, Eisenhauer & Gillessen 2010, Rev. Mod. Phys. 82, 3121):
//! - the Minispiral (Sgr A West): ionised streams falling towards Sgr A*
//!   within ~1.5 pc, lit by the central cluster;
//! - the circumnuclear disk (CND): a clumpy molecular torus from 1.5 to
//!   ~4 pc whose inner edge, ionised by the same cluster, is the Western Arc;
//! - Sgr A East: the supernova remnant, ~7 × 9 pc, centred ~2.5 pc behind
//!   Sgr A* and ~2 pc east of it on the sky;
//! - a Crab-like pulsar wind nebula (fictional) 7 pc from the hole, placed
//!   above the disk where it can be seen from the cluster.
//!
//! From inside the Galactic Centre there is none of the ~30 magnitudes of
//! visual extinction that hide all of this from Earth; only the local dust
//! (the CND, the dust mixed into the gas) dims it.

use kerr::vec3::{self, V3};

/// Distance of the Sun from Sgr A* (GRAVITY Collaboration 2019), pc.
pub const SUN_DISTANCE_PC: f64 = 8178.0;
/// One arcsecond on the sky at Sgr A*, pc.
pub const PC_PER_ARCSEC: f64 = SUN_DISTANCE_PC * std::f64::consts::PI / 648_000.0;
/// Position angle of increasing Galactic longitude at Sgr A*, degrees east
/// of north.
pub const LONGITUDE_PA_DEG: f64 = 31.4;

/// Case B Hα emissivity at 10⁴ K: 4π j / (n_e n_p), erg cm³ s⁻¹
/// (Osterbrock & Ferland 2006).
pub const HALPHA_EMISSIVITY_CGS: f64 = 3.56e-25;
/// Case B recombination coefficient at 10⁴ K, cm³ s⁻¹.
pub const ALPHA_B: f64 = 2.59e-13;
/// Visual extinction per hydrogen nucleus for Galactic dust,
/// A_V / N_H = 5.3 × 10⁻²² mag cm² (Bohlin, Savage & Drake 1978 with
/// R_V = 3.1), as an optical depth: τ_V per N_H in cm².
pub const TAU_V_PER_H: f64 = 5.3e-22 / 1.0857;
/// Centimetres per parsec.
pub const CM_PER_PC: f64 = 3.085_677_581e18;
const LSUN_W: f64 = 3.828e26;
/// G in pc (km/s)² per solar mass.
const G_PC: f64 = 4.3009e-3;

/// The Galactic frame in renderer coordinates.
#[derive(Clone, Copy, Debug)]
pub struct Galactic {
    /// Away from the Sun (from the Sun towards Sgr A*).
    pub x: V3,
    /// Towards increasing Galactic longitude.
    pub y: V3,
    /// The north Galactic pole.
    pub z: V3,
}

pub fn galactic() -> Galactic {
    let x = vec3::normalize([0.83, 0.0, 0.56]);
    let z = vec3::normalize([-0.56, 0.12, 0.83]);
    Galactic { x, y: vec3::cross(z, x), z }
}

/// Unit vector from Sgr A* towards the Sun (and Earth).
pub fn earth_direction() -> V3 {
    vec3::scale(galactic().x, -1.0)
}

/// Sky east and north (unit vectors in the plane of the sky at Sgr A*).
pub fn east_north() -> (V3, V3) {
    let g = galactic();
    let (s, c) = LONGITUDE_PA_DEG.to_radians().sin_cos();
    // l̂ = s E + c N, b̂ = −c E + s N, so E = s l̂ − c b̂ and N = c l̂ + s b̂.
    (vec3::sub(vec3::scale(g.y, s), vec3::scale(g.z, c)), vec3::add(vec3::scale(g.y, c), vec3::scale(g.z, s)))
}

/// Position (pc from Sgr A*) of a point seen `east` and `north` arcseconds
/// from Sgr A* on Earth's sky, `depth` pc behind it.
pub fn sky(east: f64, north: f64, depth: f64) -> V3 {
    let (e, n) = east_north();
    let s = vec3::add(vec3::scale(e, east * PC_PER_ARCSEC), vec3::scale(n, north * PC_PER_ARCSEC));
    vec3::axpy(s, depth, galactic().x)
}

/// A position from Galactic longitude/latitude (degrees, as seen from Sgr
/// A*, longitude 0 away from the Sun) and distance (pc).
pub fn galactic_position(lon_deg: f64, lat_deg: f64, r_pc: f64) -> V3 {
    let g = galactic();
    let (sl, cl) = lon_deg.to_radians().sin_cos();
    let (sb, cb) = lat_deg.to_radians().sin_cos();
    let d = vec3::add(vec3::add(vec3::scale(g.x, cb * cl), vec3::scale(g.y, cb * sl)), vec3::scale(g.z, sb));
    vec3::scale(d, r_pc)
}

/// Angular momentum direction of the circumnuclear disk. It rotates with
/// the Galaxy (its positive-longitude side recedes from Earth, like Galactic
/// rotation, whose angular momentum points to the south pole) and is tilted
/// 25° out of the Galactic plane, its near side to the north, so that from
/// Earth it is seen ~65° from face on, an ellipse of axis ratio ~0.4
/// elongated along the plane. With this tilt the Northern and Eastern Arms,
/// as drawn on the sky, orbit in the same sense as the disk.
pub fn cnd_axis() -> V3 {
    let g = galactic();
    let t = 25f64.to_radians();
    vec3::normalize(vec3::sub(vec3::scale(g.z, -t.cos()), vec3::scale(g.x, t.sin())))
}

/// Which physics fills a volume (the generator entry point).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Minispiral,
    Cnd,
    SgrAEast,
    Pwn,
}

/// Line and continuum physics of one volume, per unit of its texture
/// channels (see `nebula_gen.wgsl` for what the channels hold).
#[derive(Clone, Copy, Debug)]
pub struct Optics {
    /// Hα, W m⁻² sr⁻¹ per metre per unit of channel r; `None`: calibrated
    /// on the GPU to the total luminosities in `calibrate`.
    pub halpha: Option<f64>,
    /// τ_V per metre per unit of channel g.
    pub tau_gas: f64,
    /// τ_V per metre per unit of √r (dust in the emitting filaments).
    pub tau_ion: f64,
    /// Synchrotron I_λ(550 nm), W m⁻² sr⁻¹ nm⁻¹ per metre per unit of g.
    pub synchrotron: f64,
    /// Synchrotron spectral index α (F_ν ∝ ν^−α).
    pub synch_index: f64,
    /// [N II] 6548+6583 / Hα, [S II] 6716+6731 / Hα, [O I] 6300+6363 / Hα,
    /// [O III] 4959+5007 / Hβ.
    pub ratios: [f64; 4],
    /// Added per unit of channel b (fronts and radiative shocks).
    pub ratios_front: [f64; 4],
    /// [O III] / Hβ added per unit of channel a.
    pub oiii_per_a: f64,
    /// Channel a is the central cluster's light: the dust scatters it.
    pub scatters: bool,
    /// Turbulent (and thermal) velocity dispersion, km/s.
    pub sigma_kms: f64,
    /// Amplitude of the sub-voxel detail added while marching.
    pub detail: f64,
    /// Total Hα luminosity (W) and synchrotron L_λ(550 nm) (W/nm) to
    /// calibrate to.
    pub calibrate: Option<(f64, f64)>,
}

/// One box of voxels and what fills it.
#[derive(Clone, Debug)]
pub struct Volume {
    pub name: &'static str,
    pub kind: Kind,
    /// Box centre, pc from the hole.
    pub centre: V3,
    /// Box axes (unit, orthonormal).
    pub axes: [V3; 3],
    /// Half extents along the axes, pc.
    pub half: [f64; 3],
    /// Voxels along the axes (multiples of 8, for the occupancy blocks).
    pub dims: [u32; 3],
    pub seed: u32,
    /// Generator parameters p0–p3 (see the entry point in `nebula_gen.wgsl`).
    pub params: [[f32; 4]; 4],
    /// Photoionisation pass parameters (p0 of `gen_light`) and the volume
    /// whose gas also shadows this one, with its density scale relative to
    /// ours.
    pub light: Option<([f32; 4], Option<(usize, f32)>)>,
    /// Velocity field parameters (p0, p1; kind in p3.w of `gen_velocity`).
    pub velocity: [[f32; 4]; 2],
    pub optics: Optics,
}

impl Volume {
    /// Voxel edge (pc), the mean over the axes.
    pub fn voxel_pc(&self) -> f64 {
        (0..3).map(|i| 2.0 * self.half[i] / self.dims[i] as f64).sum::<f64>() / 3.0
    }
}

/// The central cluster, as the source that lights the gas and dust. The
/// ~100 young O, Wolf–Rayet and He I emission-line stars of the central
/// parsec (≈2 × 10⁷ L☉ at ~30,000 K) and the old red giants
/// (≈10⁷ L☉ at ~4,000 K) within a couple of parsecs. The simulated cluster
/// (≲0.1 pc) is only its innermost part.
pub struct Cluster {
    /// (luminosity W, temperature K) of each component.
    pub components: [(f64, f64); 2],
    /// Hydrogen-ionising photons per second: ≈2 × 10⁵⁰ s⁻¹, the
    /// recombination rate of the ionised gas inferred from its free-free
    /// emission (Lacy et al. 1980; Genzel et al. 2010).
    pub q_lyc: f64,
    /// Radius of the source region (pc): the young stars are spread over
    /// ~0.5 pc.
    pub core_pc: f64,
}

pub const CLUSTER: Cluster =
    Cluster { components: [(2.0e7 * LSUN_W, 30_000.0), (1.0e7 * LSUN_W, 4_000.0)], q_lyc: 2.0e50, core_pc: 0.3 };

/// Dust scattering: albedo and Henyey–Greenstein asymmetry of Galactic
/// dust in the optical (Draine 2003).
pub const DUST_ALBEDO: f64 = 0.6;
pub const DUST_ASYMMETRY: f64 = 0.6;

/// The pulsar at the centre of the wind nebula, like the Crab pulsar:
/// V ≈ 16.5 at 2.0 kpc behind A_V ≈ 1.6, so V₀ ≈ 14.9, with a nearly flat
/// optical spectrum F_ν ∝ ν^−0.1 (Sollerman et al. 2000).
pub struct Pulsar {
    pub v0_mag: f64,
    pub distance_pc: f64,
    pub index: f64,
}

pub const PULSAR: Pulsar = Pulsar { v0_mag: 14.9, distance_pc: 2000.0, index: 0.1 };

/// V-band zero point: F_λ(550 nm) of a V = 0 source, W m⁻² nm⁻¹.
pub const V_ZERO_W_M2_NM: f64 = 3.63e-11;

/// Hα surface brightness per unit emission measure: W m⁻² sr⁻¹ per metre
/// per (n_e n_p / 10⁸ cm⁻⁶), the unit of channel r of the photoionised
/// volumes.
fn halpha_per_ion() -> f64 {
    HALPHA_EMISSIVITY_CGS * 1.0e8 / (4.0 * std::f64::consts::PI) * 100.0 * 1.0e-3
}

/// τ_V per metre for n_H = `n` cm⁻³ with `dgr` times the Galactic dust to
/// gas ratio.
fn tau_per_m(n: f64, dgr: f64) -> f64 {
    TAU_V_PER_H * n * dgr * 100.0
}

/// Kepler speed (km/s) at r (pc) on an orbit of semi-major axis a (pc)
/// around `m` solar masses.
fn vis_viva(m: f64, r: f64, a: f64) -> f64 {
    (G_PC * m * (2.0 / r - 1.0 / a)).max(0.01 * G_PC * m / r).sqrt()
}

/// Mass that the minispiral's streams orbit: Sgr A* plus the stars within
/// ~1 pc (≈10⁶ M☉, Schödel et al. 2009), as one point mass.
pub const MINISPIRAL_MASS_MSUN: f64 = 5.0e6;

/// A stream of the minispiral: control points on Earth's sky (arcsec east,
/// arcsec north, pc behind Sgr A*) in the direction of flow, the orbit's
/// semi-major axis (pc), and gas density n_H (cm⁻³), half width and half
/// thickness (pc) at its outer and inner ends.
pub struct Stream {
    pub points: &'static [[f64; 3]],
    pub semi_major_pc: f64,
    pub density: [f64; 2],
    pub half_width: [f64; 2],
    pub half_thickness: [f64; 2],
}

/// The Northern and Eastern Arms (the Bar is where they meet south of
/// Sgr A*), traced from the radio and Paα maps (Lo & Claussen 1983; Zhao et
/// al. 2009). Both are tidally stretched streams on eccentric orbits
/// (e ≈ 0.8) that pass ~0.1 pc from Sgr A* at a few hundred km/s, orbiting
/// in the CND's sense; their depths follow from that. Densities ~10⁴ cm⁻³
/// (Zhao et al. 2009); where a stream's gas is too dense for the cluster to
/// ionise all of it, only the side facing the cluster glows. The Western
/// Arc is the CND's inner edge (`gen_cnd`).
pub const STREAMS: [Stream; 2] = [
    Stream {
        // Northern Arm.
        points: &[
            [10.0, 42.0, -0.55],
            [10.0, 32.0, -0.45],
            [8.5, 22.0, -0.32],
            [6.0, 13.0, -0.17],
            [3.5, 5.5, -0.04],
            [1.5, 0.5, 0.06],
            [-2.0, -3.0, 0.1],
            [-7.0, -4.5, 0.08],
            [-12.0, -2.5, 0.02],
        ],
        semi_major_pc: 1.0,
        density: [1.5e4, 2.5e4],
        half_width: [0.2, 0.07],
        half_thickness: [0.06, 0.035],
    },
    Stream {
        // Eastern Arm.
        points: &[
            [34.0, -3.0, 0.0],
            [27.0, -7.0, 0.08],
            [19.0, -9.5, 0.14],
            [11.0, -8.5, 0.17],
            [5.0, -5.0, 0.15],
            [0.0, -3.5, 0.12],
            [-6.0, -6.0, 0.1],
            [-12.0, -11.0, 0.06],
        ],
        semi_major_pc: 1.1,
        density: [1.2e4, 2.0e4],
        half_width: [0.14, 0.06],
        half_thickness: [0.05, 0.03],
    },
];

/// One node of a stream for the generator.
#[derive(Clone, Copy, Debug)]
pub struct StreamNode {
    pub pos: V3,
    pub density: f64,
    pub tangent: V3,
    pub half_width: f64,
    pub normal: V3,
    pub half_thickness: f64,
    /// Velocity, km/s.
    pub velocity: V3,
    pub stream: usize,
}

/// Catmull–Rom point between control points.
fn catmull_rom(p: &[V3], t: f64) -> V3 {
    let n = p.len() - 1;
    let s = (t * n as f64).clamp(0.0, n as f64 - 1e-9);
    let i = s.floor() as usize;
    let u = s - i as f64;
    let p0 = p[i.saturating_sub(1)];
    let p1 = p[i];
    let p2 = p[(i + 1).min(n)];
    let p3 = p[(i + 2).min(n)];
    let (u2, u3) = (u * u, u * u * u);
    std::array::from_fn(|k| {
        0.5 * (2.0 * p1[k]
            + (p2[k] - p0[k]) * u
            + (2.0 * p0[k] - 5.0 * p1[k] + 4.0 * p2[k] - p3[k]) * u2
            + (3.0 * p1[k] - p0[k] - 3.0 * p2[k] + p3[k]) * u3)
    })
}

/// The streams resampled into nodes, with Kepler speeds along the flow.
/// Each stream is a ribbon whose broad side faces Earth (the arms look wide
/// on the sky); its density fades in over the first nodes so it blends into
/// the CND it comes from.
pub fn stream_nodes(per_stream: usize) -> Vec<StreamNode> {
    let x = galactic().x;
    let mut out = Vec::new();
    for (id, s) in STREAMS.iter().enumerate() {
        let pts: Vec<V3> = s.points.iter().map(|p| sky(p[0], p[1], p[2])).collect();
        for i in 0..per_stream {
            let f = i as f64 / (per_stream - 1) as f64;
            let pos = catmull_rom(&pts, f);
            let h = 1e-3;
            let tangent =
                vec3::normalize(vec3::sub(catmull_rom(&pts, (f + h).min(1.0)), catmull_rom(&pts, (f - h).max(0.0))));
            let normal = vec3::normalize(vec3::axpy(x, -vec3::dot(x, tangent), tangent));
            let r = vec3::norm(pos);
            let speed = vis_viva(MINISPIRAL_MASS_MSUN, r, s.semi_major_pc);
            let fade = (f / 0.12).min(1.0);
            let lerp = |a: [f64; 2]| a[0] + (a[1] - a[0]) * f;
            out.push(StreamNode {
                pos,
                density: lerp(s.density) * fade * fade * (3.0 - 2.0 * fade),
                tangent,
                half_width: lerp(s.half_width),
                normal,
                half_thickness: lerp(s.half_thickness),
                velocity: vec3::scale(tangent, speed),
                stream: id,
            });
        }
    }
    out
}

/// Seconds per year.
const YEAR_S: f64 = 3.156e7;
/// Kilometres per parsec.
const KM_PER_PC: f64 = CM_PER_PC * 1.0e-5;

/// Minispiral gas scale: channel g is n_H / 10⁴ cm⁻³.
const MINISPIRAL_N: f64 = 1.0e4;
/// CND gas scale: channel g is n_H / 10⁵ cm⁻³.
const CND_N: f64 = 1.0e5;

/// Photoionisation parameters for a volume with gas scale `n_scale` and
/// dust-to-gas ratio `dgr` (relative to Galactic).
fn light_params(n_scale: f64, dgr: f64) -> [f32; 4] {
    let k = ALPHA_B * 1.0e8 * CM_PER_PC.powi(3);
    let budget = CLUSTER.q_lyc / (4.0 * std::f64::consts::PI * k);
    let tau = TAU_V_PER_H * 1.0e4 * CM_PER_PC * dgr;
    [budget as f32, (n_scale / 1.0e4) as f32, tau as f32, 0.05]
}

fn f4(v: V3, w: f64) -> [f32; 4] {
    [v[0] as f32, v[1] as f32, v[2] as f32, w as f32]
}

/// All volumes, in the order of the shader's bindings.
pub fn volumes() -> [Volume; 4] {
    let g = galactic();
    let cnd_n = cnd_axis();
    let (east, _) = east_north();
    [minispiral(&g, cnd_n), cnd(&g, cnd_n, east), sgr_a_east(&g, east), pwn(&g)]
}

/// Sgr A West. Photoionised gas at ~10⁴ K with the line ratios of the
/// minispiral: low excitation ([Ne III]/[Ne II] and [Ar III]/[Ar II] are
/// small: the ionising stars are ~30,000 K, Lutz et al. 1996), so [O III]
/// is weak, while the twice-solar metallicity makes [N II] strong. About
/// half the Galactic dust-to-gas ratio (dust is being destroyed in the
/// streams). The streams move at up to ~600 km/s near Sgr A*; lines are
/// ~25 km/s wide besides.
fn minispiral(g: &Galactic, cnd_n: V3) -> Volume {
    let gm = G_PC * MINISPIRAL_MASS_MSUN * 1.0e-6;
    Volume {
        name: "minispiral",
        kind: Kind::Minispiral,
        centre: vec3::add(vec3::scale(g.y, 0.2), vec3::scale(g.z, -0.1)),
        axes: [g.y, g.z, g.x],
        half: [1.7, 1.5, 0.9],
        dims: [256, 224, 136],
        seed: 11,
        // Diffuse ionised gas of ~300 cm⁻³ fills the cavity out to 0.9 pc;
        // nothing within 0.08 pc of the hole (inside the simulated
        // cluster).
        params: [[0.03, 0.08, 0.9, 0.0], [0.0; 4], [0.0; 4], [0.0; 4]],
        light: Some((light_params(MINISPIRAL_N, 0.5), None)),
        velocity: [[gm as f32, 0.0, 0.0, 0.0], f4(cnd_n, 0.0)],
        optics: Optics {
            halpha: Some(halpha_per_ion()),
            tau_gas: tau_per_m(MINISPIRAL_N, 0.5),
            tau_ion: 0.0,
            synchrotron: 0.0,
            synch_index: 0.0,
            ratios: [0.55, 0.12, 0.01, 0.3],
            ratios_front: [0.4, 0.5, 0.08, 0.0],
            oiii_per_a: 0.0,
            scatters: true,
            sigma_kms: 27.0,
            detail: 0.55,
            calibrate: None,
        },
    }
}

/// The circumnuclear disk: inner radius 1.5 pc, most of its mass within
/// ~3 pc and a tail to ~4 pc, a few tenths of a parsec thick, warped and
/// lopsided, rotating at ~110 km/s (Güsten et al. 1987; Christopher et al.
/// 2005). Clumps of 10⁵–10⁶ cm⁻³ in a 10³–10⁴ cm⁻³ inter-clump medium
/// (Requena-Torres et al. 2012), a few × 10⁴ M☉ in all: A_V of tens to
/// hundreds of magnitudes through a clump, so the disk is black against the
/// Galaxy except where the cluster lights it. The minispiral's gas shadows
/// it from the cluster.
fn cnd(g: &Galactic, n: V3, east: V3) -> Volume {
    let e1 = g.y;
    let e2 = vec3::cross(n, e1);
    let west = vec3::scale(east, -1.0);
    let w = [vec3::dot(west, e1), vec3::dot(west, e2)];
    Volume {
        name: "cnd",
        kind: Kind::Cnd,
        centre: [0.0; 3],
        axes: [e1, e2, n],
        half: [4.5, 4.5, 1.2],
        dims: [256, 256, 64],
        seed: 23,
        params: [
            [1.5, 0.15, 0.1, 0.06],
            [w[0] as f32, w[1] as f32, 0.15, 0.0],
            [0.03, 6.0, 0.3, 1.2],
            [0.15, -0.1, 2.3, 0.25],
        ],
        light: Some((light_params(CND_N, 1.0), Some((0, (MINISPIRAL_N / CND_N) as f32)))),
        velocity: [[0.11, 0.0, 0.0, 0.0], f4(n, 0.0)],
        optics: Optics {
            halpha: Some(halpha_per_ion()),
            tau_gas: tau_per_m(CND_N, 1.0),
            tau_ion: 0.0,
            synchrotron: 0.0,
            synch_index: 0.0,
            ratios: [0.55, 0.15, 0.01, 0.2],
            ratios_front: [0.4, 0.6, 0.1, 0.0],
            oiii_per_a: 0.0,
            scatters: true,
            sigma_kms: 22.0,
            detail: 0.6,
            calibrate: None,
        },
    }
}

/// Sgr A East: an elongated shell of 9 × 7 pc along the Galactic plane
/// (the radio shell, e.g. Maeda et al. 2002), centred ~52″ east and ~8″
/// north of Sgr A* on the sky and 2.5 pc behind it, so Sgr A* lies inside
/// the shell near its front, and the Minispiral in front of its emission
/// (as the free-free absorption shows). The shell runs into the "50 km/s"
/// molecular cloud to the east, where it is brightest.
///
/// Its famous emission is radio synchrotron from GeV electrons; extended to
/// the optical at its spectral index (S_ν ∝ ν^−1, ~200 Jy at 1 GHz) it
/// would be V ≈ 17.5 spread over 12 arcmin², far below anything visible,
/// and the electrons that would radiate optical synchrotron cool in far
/// less than the remnant's age. So in visible light it is what old remnants
/// look like: the radiative shock front, thin wrinkled sheets that show as
/// Veil-like filaments where seen edge on. Face on a ~150 km/s radiative
/// shock into the ~10² cm⁻³ gas of the Galactic Centre emits about half an
/// Hα photon per hydrogen atom swept up: I(Hα) = n₀ v_s ½ hν / 4π ≈
/// 2 × 10⁻⁷ W m⁻² sr⁻¹, many times that where the sheets are edge on.
/// Shock line ratios: [S II]/Hα ≈ 0.5–1 (the classic criterion), strong
/// [N II] at the Galactic Centre's metallicity, [O III] from the faster
/// shocks (Raymond 1979; Fesen, Blair & Kirshner 1985).
fn sgr_a_east(g: &Galactic, east: V3) -> Volume {
    let semi: [f64; 3] = [4.5, 3.5, 3.5];
    let sheet = 0.012;
    let a_mean = (semi[0] * semi[1] * semi[2]).cbrt();
    let n0 = 100.0e6;
    let v_s = 150.0e3;
    let photon = 6.626e-34 * 2.998e8 / 656.28e-9;
    let face_on = n0 * v_s * 0.5 * photon / (4.0 * std::f64::consts::PI);
    let per_unit = face_on / (std::f64::consts::PI.sqrt() * sheet * a_mean * CM_PER_PC * 0.01);
    let axes = [g.y, g.z, g.x];
    let cloud = vec3::normalize([vec3::dot(east, axes[0]), vec3::dot(east, axes[1]), vec3::dot(east, axes[2])]);
    Volume {
        name: "sgr-a-east",
        kind: Kind::SgrAEast,
        centre: sky(52.0, 8.0, 2.5),
        axes,
        half: [5.0, 4.0, 4.0],
        dims: [256, 208, 208],
        seed: 37,
        params: [
            [sheet as f32, 3.0, 0.0, 1.5],
            f4(cloud, 0.0),
            [semi[0] as f32, semi[1] as f32, semi[2] as f32, 1.0],
            [0.0; 4],
        ],
        light: None,
        // Expanding at ~300 km/s at the shell.
        velocity: [[0.3, 0.0, 0.0, 0.0], [semi[0] as f32, semi[1] as f32, semi[2] as f32, 0.0]],
        optics: Optics {
            halpha: Some(per_unit),
            // Swept-up dust, mostly destroyed by the shock: τ_V ~ 0.02 face
            // on through a sheet.
            tau_gas: 0.02 / (std::f64::consts::PI.sqrt() * sheet * a_mean * CM_PER_PC * 0.01),
            tau_ion: 0.0,
            synchrotron: 0.0,
            synch_index: 0.0,
            ratios: [0.9, 0.6, 0.12, 0.5],
            ratios_front: [0.3, 0.3, 0.1, 0.0],
            oiii_per_a: 6.0,
            scatters: false,
            sigma_kms: 40.0,
            detail: 0.5,
            calibrate: None,
        },
    }
}

/// Where the (fictional) pulsar wind nebula is: 7 pc from Sgr A*, 38°
/// above the Galactic plane (over 30° above the CND, which would hide it
/// otherwise), ~25° above the direction the ship faces at the start.
pub fn pwn_centre() -> V3 {
    galactic_position(95.0, 38.0, 7.0)
}

/// A pulsar wind nebula like the Crab: 4.4 × 2.9 pc (the Crab's size at
/// 2.0 kpc), its long axis along the pulsar's spin axis, expanding
/// homologously since it formed 972 years ago (v = r / age: ~1500 km/s
/// across, ~2200 km/s along the long axis, as the Crab's filaments).
/// A filament cage with Rayleigh–Taylor fingers around a synchrotron nebula
/// with a torus and a jet.
///
/// Brightness is calibrated to the Crab's totals, corrected for its
/// foreground extinction (A_V ≈ 1.6), which this one lacks: V ≈ 8.4 → 6.8,
/// most of it synchrotron, i.e. F_λ(550 nm) ≈ 6.4 × 10⁻¹⁴ W m⁻² nm⁻¹ at
/// 2.0 kpc; optical index α ≈ 0.8 (F_ν ∝ ν^−α; Véron-Cetty & Woltjer
/// 1993). The filaments radiate ~1% of the synchrotron power in lines,
/// L(Hα) ≈ 2 × 10³⁵ erg/s (Davidson & Fesen 1985). Photoionised by the
/// synchrotron light: [O III] strong in the outer skins, [S II], [N II] and
/// [O I] strong in the dense cores; some dust in the filaments, which shows
/// as dark threads against the synchrotron glow (Fesen & Blair 1990).
fn pwn(g: &Galactic) -> Volume {
    let centre = pwn_centre();
    let v = vec3::normalize(centre);
    let u1 = vec3::normalize(vec3::cross(v, g.z));
    let u2 = vec3::cross(u1, v);
    let (s40, c40) = 40f64.to_radians().sin_cos();
    let (s25, c25) = 25f64.to_radians().sin_cos();
    let major0 = vec3::add(vec3::scale(u1, c40), vec3::scale(u2, s40));
    let major = vec3::normalize(vec3::add(vec3::scale(major0, c25), vec3::scale(v, s25)));
    let m2 = vec3::normalize(vec3::cross(major, v));
    let m3 = vec3::cross(major, m2);
    let semi = [2.2, 1.45, 1.45];
    let d_m = 2000.0 * CM_PER_PC * 0.01;
    let area = 4.0 * std::f64::consts::PI * d_m * d_m;
    let synch = 0.9 * V_ZERO_W_M2_NM * 10f64.powf(-0.4 * 6.8) * area;
    let age_s = 972.0 * YEAR_S;
    Volume {
        name: "pwn",
        kind: Kind::Pwn,
        centre,
        axes: [major, m2, m3],
        half: [2.4, 1.6, 1.6],
        dims: [256, 176, 176],
        seed: 53,
        params: [
            [semi[0] as f32, semi[1] as f32, semi[2] as f32, 0.86],
            [4.0, 9.0, 14.0, 0.3],
            [0.3, 0.1, 0.9, 0.06],
            [0.0; 4],
        ],
        light: None,
        velocity: [[(KM_PER_PC / age_s * 1.0e-3) as f32, 0.0, 0.0, 0.0], [0.0; 4]],
        optics: Optics {
            halpha: None,
            tau_gas: 0.0,
            // τ_V ≈ 0.3 through the densest filament cores (~0.03 pc).
            tau_ion: 0.3 / (0.03 * CM_PER_PC * 0.01),
            synchrotron: 0.0,
            synch_index: 0.8,
            ratios: [0.6, 0.35, 0.05, 1.5],
            ratios_front: [0.5, 0.5, 0.2, 0.0],
            oiii_per_a: 7.0,
            scatters: false,
            sigma_kms: 30.0,
            detail: 0.35,
            calibrate: Some((2.0e28, synch)),
        },
    }
}

/// Named places for pointing the ship (`--look`): centres, pc from Sgr A*.
pub fn landmarks() -> [(&'static str, V3); 5] {
    let v = volumes();
    [("sgra", [0.0; 3]), ("minispiral", [0.0; 3]), ("cnd", [0.0; 3]), ("sgra-east", v[2].centre), ("pwn", v[3].centre)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sky_axes_are_right_handed_and_away_from_earth() {
        let (e, n) = east_north();
        assert!(vec3::dot(e, n).abs() < 1e-12);
        // East × north points away from the observer.
        let away = vec3::cross(e, n);
        assert!(vec3::dot(away, galactic().x) > 0.999);
        // Galactic north lies at PA −58.6°: mostly west and a little north.
        let b = galactic().z;
        assert!(vec3::dot(b, e) < -0.8 && vec3::dot(b, n) > 0.5);
    }

    #[test]
    fn streams_fall_in_and_corotate_with_the_disk() {
        let nodes = stream_nodes(24);
        let axis = cnd_axis();
        for id in 0..STREAMS.len() {
            let s: Vec<_> = nodes.iter().filter(|n| n.stream == id).collect();
            let l: f64 = s.iter().map(|n| vec3::dot(vec3::cross(n.pos, n.velocity), axis)).sum();
            assert!(l > 0.0, "stream {id} counter-rotates");
            let r0 = vec3::norm(s[0].pos);
            let rmin = s.iter().map(|n| vec3::norm(n.pos)).fold(f64::MAX, f64::min);
            assert!(r0 > 1.2 && rmin < 0.2 && rmin > 0.08, "stream {id}: {r0} {rmin}");
            let vmax = s.iter().map(|n| vec3::norm(n.velocity)).fold(0.0, f64::max);
            assert!(vmax > 300.0 && vmax < 900.0, "{vmax}");
        }
    }

    #[test]
    fn everything_fits_its_box() {
        for v in volumes() {
            for i in 0..3 {
                assert_eq!(v.dims[i] % 8, 0, "{}", v.name);
            }
        }
        let ms = &volumes()[0];
        for n in stream_nodes(24) {
            let d = vec3::sub(n.pos, ms.centre);
            for i in 0..3 {
                assert!(vec3::dot(d, ms.axes[i]).abs() + n.half_width < ms.half[i], "{:?}", n.pos);
            }
        }
    }

    #[test]
    fn the_pwn_is_clear_of_the_disk_plane() {
        let d = vec3::normalize(pwn_centre());
        let elevation = vec3::dot(d, cnd_axis()).abs().asin().to_degrees();
        assert!(elevation > 25.0, "{elevation}");
    }

    #[test]
    fn minispiral_arms_are_about_as_bright_as_the_radio_says() {
        // An ionisation-bounded skin facing the cluster 0.5 pc away has an
        // emission measure Q / (4π r² α_B), a few 10⁶ pc cm⁻⁶, as measured
        // (Sgr A West turns optically thick to free-free near 1 GHz).
        let r = 0.5 * CM_PER_PC;
        let em = CLUSTER.q_lyc / (4.0 * std::f64::consts::PI * r * r * ALPHA_B) / CM_PER_PC;
        assert!(em > 3e6 && em < 3e7, "{em}");
    }
}
