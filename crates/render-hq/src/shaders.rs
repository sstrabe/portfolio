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

/// Point sources splatted into the HDR radiance.
pub fn splat() -> String {
    assemble(&[wgsl!("splat.wgsl")])
}

/// Optics, tone mapping and upscaling.
pub fn post() -> String {
    assemble(&[wgsl!("post.wgsl")])
}

pub fn all() -> [(&'static str, String); 3] {
    [("trace", trace()), ("splat", splat()), ("post", post())]
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
