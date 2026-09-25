//! The desktop's high-quality renderer (native only; the web keeps
//! `render`).
//!
//! One compute pass traces every pixel spectrally, front to back: the ship,
//! the star systems near the pilot (each in its own rest frame, so
//! aberration and Doppler shifts are exact), then the Kerr geodesic out to
//! nebulae and the distant galaxy. Point sources from the shared `images`
//! pass are splatted as energy afterwards, and the optics turn radiance into
//! an image. See `gpu.rs` for the pass order and `wgsl/hq_common.wgsl` for
//! which bind group belongs to which feature.

pub mod atmosphere;
pub mod fft;
pub mod gpu;
pub mod lens;
pub mod near;
pub mod nebula;
pub mod optics;
pub mod overlay;
pub mod post;
pub mod profile;
pub mod session;
pub mod shaders;
pub mod ship;
pub mod spectrum;

pub use gpu::Gpu;
pub use session::Session;

use bytemuck::{Pod, Zeroable};

/// Mirrors `struct HqFrame` in `hq_common.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct HqUniforms {
    pub size: [u32; 4],
    pub view: [f32; 4],
    pub radiometry: [f32; 4],
    pub near: [u32; 4],
    pub units: [f32; 4],
    pub rgb: [[f32; 4]; 12],
}

/// `HqUniforms::size[3]` flag: the trace also writes the narrowband bins
/// (mirrors `HQ_NARROWBAND` in `hq_common.wgsl`).
pub const HQ_NARROWBAND: u32 = 1;
/// `HqUniforms::size[3]` flag: mark non-finite pixels (KERR_DEBUG_NAN).
pub const HQ_DEBUG_NAN: u32 = 2;

/// What features see when they update for a frame.
pub struct FrameContext<'a> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub world: &'a kerr::world::World,
    pub near: &'a near::Selection,
    pub hq: &'a HqUniforms,
    pub frame_index: u32,
    pub wall_time: f64,
    /// The view's tetrad: the pilot's, turned onto the camera's axes in the
    /// chase view.
    pub view: &'a kerr::pilot::Tetrad,
}

#[cfg(test)]
mod tests {
    #[test]
    fn hq_uniform_layout_matches_wgsl() {
        assert_eq!(std::mem::size_of::<super::HqUniforms>(), (5 + 12) * 16);
    }
}
