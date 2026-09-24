//! Physics core for the relativistic galactic nucleus.
//!
//! Everything lives in geometrized units (`G = c = 1`) with lengths and times
//! measured in units of the central black hole mass `M` (the metric keeps `M`
//! as a parameter so the flat limit `M = 0` can be tested).
//!
//! The spacetime is Kerr in Cartesian Kerr–Schild coordinates
//! `(t, x, y, z)`, spin along `+z`:
//!
//! ```text
//! g_{μν} = η_{μν} + f l_μ l_ν,     g^{μν} = η^{μν} − f l^μ l^ν
//! f   = 2 M r³ / (r⁴ + a² z²)
//! l_μ = (1, (r x + a y)/(r² + a²), (r y − a x)/(r² + a²), z / r)
//! ```
//!
//! where `r` is the Boyer–Lindquist-like radius defined implicitly by
//! `(x² + y²)/(r² + a²) + z²/r² = 1`. The coordinates are horizon
//! penetrating for future-directed (infalling) worldlines, and every
//! constant-`t` slice is spacelike, so `t` is used as the global simulation
//! clock for the whole cluster.
//!
//! Modules:
//! * [`metric`]: metric, inverse, analytic gradients and Christoffel symbols.
//! * [`geodesic`]: Hamiltonian geodesic equations and conserved quantities.
//! * [`integrate`]: Dormand–Prince 5(4) and RK4 steppers.
//! * [`orbit`]: initial conditions for bound orbits.
//! * [`history`]: sampled worldline history with Hermite interpolation.
//! * [`cluster`]: the N-body star cluster (geodesics + retarded weak-field
//!   perturbations + 2.5PN radiation reaction).
//! * [`pilot`]: the visitor's accelerated worldline and Fermi–Walker tetrad.
//! * [`lensing`]: backward null-geodesic tracing and image finding on the
//!   observer's past light cone.
//! * [`world`]: ties it together behind a per-frame API.
//! * [`units`]: conversions for a hole of Sagittarius A*'s mass.

// Index loops mirror the tensor notation in the docs and the WGSL ports.
#![allow(clippy::needless_range_loop)]

pub mod cluster;
pub mod dual;
pub mod geodesic;
pub mod history;
pub mod integrate;
pub mod lensing;
pub mod metric;
pub mod orbit;
pub mod pilot;
pub mod rng;
pub mod units;
pub mod vec3;
pub mod world;

pub use metric::Kerr;
