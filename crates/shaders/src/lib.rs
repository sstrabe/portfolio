//! WGSL has no `#include`, so every pipeline's module is the shared prelude
//! (`common.wgsl`: frame uniforms, Kerr–Schild geometry, null geodesics,
//! blackbody colours) followed by its own entry points.

macro_rules! module {
    ($($file:literal),+) => {
        concat!($(include_str!(concat!("wgsl/", $file)), "\n"),+)
    };
}

/// The shared prelude on its own (for renderers that assemble their own
/// modules on top of it).
pub const COMMON: &str = include_str!("wgsl/common.wgsl");

/// Full-screen ray tracer: shadow, lensed sky, in-scene station panels.
pub const SKY: &str = module!("sky_header.wgsl", "common.wgsl", "sky.wgsl");
/// Compute pass that finds each body's images on the past light cone.
pub const IMAGES: &str = module!("common.wgsl", "history.wgsl", "images.wgsl");
/// Instanced point sprites for the body images.
pub const SPRITES: &str = module!("common.wgsl", "sprites.wgsl");
/// Upscale + tone mapping.
pub const COMPOSITE: &str = module!("common.wgsl", "composite.wgsl");

pub const ALL: [(&str, &str); 4] = [("sky", SKY), ("images", IMAGES), ("sprites", SPRITES), ("composite", COMPOSITE)];

#[cfg(test)]
mod tests {
    #[test]
    fn shaders_validate() {
        for (name, src) in super::ALL {
            let module = match naga::front::wgsl::parse_str(src) {
                Ok(m) => m,
                Err(e) => panic!("{name}: {}", e.emit_to_string(src)),
            };
            let mut v =
                naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all());
            if let Err(e) = v.validate(&module) {
                panic!("{name}: {}", e.emit_to_string(src));
            }
        }
    }
}
