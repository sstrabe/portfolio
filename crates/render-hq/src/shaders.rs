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

/// The per-pixel trace pass; with `rt`, geometry is traced with hardware
/// ray queries (`EXPERIMENTAL_RAY_QUERY`) instead of in software.
pub fn trace(rt: bool) -> String {
    let body = assemble(&[
        wgsl!(
            "near.wgsl",
            "terrain.wgsl",
            "planet.wgsl",
            "atmo_common.wgsl",
            "atmosphere.wgsl",
            "clouds.wgsl",
            "nebula.wgsl",
            "ship.wgsl",
            "lens.wgsl",
            "far.wgsl",
            "trace.wgsl"
        ),
        if rt { wgsl!("ship_rq.wgsl") } else { wgsl!("ship_bvh.wgsl") },
    ]);
    // An `enable` directive has to come before any declaration.
    if rt { format!("enable wgpu_ray_query;\n{body}") } else { body }
}

/// Atmosphere lookup tables and cloud textures (entry points in
/// `atmo_luts.wgsl` and `cloud_gen.wgsl`). Built on the Kerr prelude alone:
/// these passes bind their own resources in group 0.
pub fn atmosphere_tables() -> String {
    let mut s = String::from(::shaders::COMMON);
    s.push('\n');
    s.push_str(wgsl!("spectrum.wgsl", "atmo_common.wgsl", "atmo_luts.wgsl", "cloud_gen.wgsl"));
    s
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

/// Startup generation of the nebula volumes (standalone: its own bindings
/// at group 0).
pub fn nebula_gen() -> String {
    wgsl!("nebula_gen.wgsl").to_string()
}

/// The nebulae seen from inside the cluster, cached per direction (the trace
/// sources plus `nebula_cube.wgsl`'s entry point).
pub fn nebula_cube() -> String {
    let mut s = trace(false);
    s.push_str(wgsl!("nebula_cube.wgsl"));
    s
}

/// The 2D overlay (standalone: no shared prelude).
pub fn overlay() -> String {
    wgsl!("overlay.wgsl").to_string()
}

/// Terrain heights at given directions, read back by the CPU. Built on
/// the Kerr prelude alone, binding its own resources in group 0.
pub fn probe() -> String {
    let mut s = String::from(::shaders::COMMON);
    s.push('\n');
    s.push_str(wgsl!("terrain.wgsl", "probe.wgsl"));
    s
}

pub fn all() -> [(&'static str, String); 13] {
    [
        ("probe", probe()),
        ("overlay", overlay()),
        ("trace", trace(false)),
        ("trace (ray queries)", trace(true)),
        ("nebula cube", nebula_cube()),
        ("atmosphere tables", atmosphere_tables()),
        ("nebula_gen", nebula_gen()),
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
