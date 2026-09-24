//! Rendering of the Kerr nucleus from the pilot's past light cone.
//!
//! * [`frame`] turns the physics world into GPU uniforms and the stations'
//!   screen positions (platform independent, unit tested).
//! * [`gpu`] owns the wgpu device and the passes.
//! * [`session`] runs a world and its renderer frame by frame.

pub mod frame;
pub mod gpu;
pub mod session;

pub use gpu::Gpu;
pub use session::{Session, world_config};
