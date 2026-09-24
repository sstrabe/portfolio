//! WGSL modules of the HQ renderer, assembled from the shared Kerr prelude
//! (`shaders::COMMON`) and the files in `src/wgsl/`. Module-scope
//! declarations in WGSL may appear in any order, so each feature keeps its
//! code in its own file.

macro_rules! wgsl {
    ($($file:literal),+) => {
        concat!($(include_str!(concat!("wgsl/", $file)), "\n"),+)
    };
}

const PRELUDE: &str = wgsl!("spectrum.wgsl", "hq_common.wgsl");

fn assemble(parts: &[&str]) -> String {
    let mut s = String::from(::shaders::COMMON);
    s.push('\n');
    s.push_str(PRELUDE);
    for p in parts {
        s.push_str(p);
    }
    s
}

/// The per-pixel trace pass.
pub fn trace() -> String {
    assemble(&[wgsl!(
        "near.wgsl",
        "planet.wgsl",
        "atmosphere.wgsl",
        "nebula.wgsl",
        "ship.wgsl",
        "lens.wgsl",
        "far.wgsl",
        "trace.wgsl"
    )])
}

const POST_PRELUDE: &str = wgsl!("post_common.wgsl");

/// Temporal accumulation and upscaling.
pub fn taa() -> String {
    assemble(&[POST_PRELUDE, wgsl!("taa.wgsl")])
}

/// Point sources splatted into the scene, and lens ghosts.
pub fn splat() -> String {
    assemble(&[POST_PRELUDE, wgsl!("splat.wgsl")])
}

/// The point-spread function's kernel on the convolution grid.
pub fn psf() -> String {
    assemble(&[wgsl!("psf.wgsl")])
}

/// Downsampling, FFT convolution and the luminance histogram.
pub fn fft() -> String {
    assemble(&[POST_PRELUDE, wgsl!("fft.wgsl")])
}

/// Metering and adaptation.
pub fn exposure() -> String {
    assemble(&[POST_PRELUDE, wgsl!("exposure.wgsl")])
}

/// The eye model, tone mapping and display encoding.
pub fn post() -> String {
    assemble(&[POST_PRELUDE, wgsl!("post.wgsl")])
}

pub fn all() -> [(&'static str, String); 7] {
    [
        ("trace", trace()),
        ("taa", taa()),
        ("splat", splat()),
        ("psf", psf()),
        ("fft", fft()),
        ("exposure", exposure()),
        ("post", post()),
    ]
}

#[cfg(test)]
mod tests {
    #[test]
    fn shaders_validate() {
        for (name, src) in super::all() {
            let module = match naga::front::wgsl::parse_str(&src) {
                Ok(m) => m,
                Err(e) => panic!("{name}: {}", e.emit_to_string(&src)),
            };
            let mut v =
                naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all());
            if let Err(e) = v.validate(&module) {
                panic!("{name}: {}", e.emit_to_string(&src));
            }
        }
    }
}
