//! Native desktop build of the Kerr nucleus: the black hole, the star cluster
//! and your ship, on Vulkan, Metal or DX12. No portfolio stations or content.
//!
//! ```text
//! kerr-nucleus [--stars N] [--fov DEG] [--scale S]
//! kerr-nucleus --headless 1280x720 [--seconds S] [--out shot.png]
//! ```

mod app;
mod headless;
mod input;

use kerr::world::World;

pub struct Options {
    pub stars: u32,
    pub fov: f64,
    pub scale: Option<f32>,
    pub headless: Option<(u32, u32)>,
    pub seconds: f64,
    pub out: String,
}

const USAGE: &str = "\
kerr-nucleus: fly through a star cluster around a spinning black hole

USAGE:
    kerr-nucleus [OPTIONS]

OPTIONS:
    --stars N           Cluster size (default 384)
    --fov DEG           Vertical field of view (default 75)
    --scale S           Fixed ray-tracing resolution scale 0.2–1 (default: adaptive)
    --headless WxH      Render offscreen and save a PNG instead of opening a window
    --seconds S         Headless: seconds of flight before the shot (default 2)
    --out FILE          Headless: output path (default kerr-nucleus.png)
    -h, --help          Show this help

CONTROLS:
    W/S thrust, A/D strafe, Space/C up/down, drag or arrows to turn, Q/E roll,
    Shift boost, F11 fullscreen, Esc leave fullscreen, Ctrl+Q quit.
    Telemetry is shown in the window title.

ENVIRONMENT:
    WGPU_BACKEND=vulkan|metal|dx12  Force a graphics backend
";

fn parse() -> Result<Options, String> {
    let mut o =
        Options { stars: 384, fov: 75.0, scale: None, headless: None, seconds: 2.0, out: "kerr-nucleus.png".into() };
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut value = |name: &str| args.next().ok_or(format!("{name} needs a value"));
        match a.as_str() {
            "-h" | "--help" => {
                print!("{USAGE}");
                std::process::exit(0);
            }
            "--stars" => o.stars = value("--stars")?.parse().map_err(|e| format!("--stars: {e}"))?,
            "--fov" => o.fov = value("--fov")?.parse().map_err(|e| format!("--fov: {e}"))?,
            "--scale" => o.scale = Some(value("--scale")?.parse().map_err(|e| format!("--scale: {e}"))?),
            "--headless" => {
                let v = value("--headless")?;
                let (w, h) = v.split_once('x').ok_or("--headless expects WIDTHxHEIGHT")?;
                o.headless = Some((w.parse().map_err(|_| "bad width")?, h.parse().map_err(|_| "bad height")?));
            }
            "--seconds" => o.seconds = value("--seconds")?.parse().map_err(|e| format!("--seconds: {e}"))?,
            "--out" => o.out = value("--out")?,
            other => return Err(format!("unknown option {other}\n\n{USAGE}")),
        }
    }
    Ok(o)
}

fn main() {
    let result = parse().and_then(|o| if o.headless.is_some() { headless::run(&o) } else { app::run(o) });
    if let Err(e) = result {
        eprintln!("kerr-nucleus: {e}");
        std::process::exit(1);
    }
}

/// Graphics instance for the platform's native backends (overridable with
/// `WGPU_BACKEND`).
pub fn instance() -> wgpu::Instance {
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = wgpu::Backends::from_env().unwrap_or(wgpu::Backends::PRIMARY);
    wgpu::Instance::new(desc)
}

/// The cluster with no stations: nothing but the hole, the stars and you.
pub fn world(stars: u32) -> World {
    World::new(render::world_config(0, stars, 1))
}
