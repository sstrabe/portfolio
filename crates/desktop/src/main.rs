//! Native desktop build of the Kerr nucleus: Sagittarius A* at its real
//! scale, its star cluster and your ship, on Vulkan, Metal or DX12. No
//! portfolio stations or content.
//!
//! ```text
//! kerr-nucleus [--stars N] [--fov DEG] [--scale S] [--start cluster|planet[:KIND[:ALTITUDE[:VIEW]]]] [--look NEBULA[@PC]]
//! kerr-nucleus --headless 1280x720 [--seconds S] [--out shot.png]
//! kerr-nucleus [--optics eye|camera|astro] [--palette true|hubble] [--ev STOPS]
//! ```

mod app;
mod headless;
mod hud;
mod input;
mod start;

use kerr::world::World;

pub struct Options {
    pub stars: u32,
    pub fov: f64,
    pub scale: Option<f32>,
    pub headless: Option<(u32, u32)>,
    pub seconds: f64,
    pub out: String,
    /// Headless: start this many stellar radii from the nearest star.
    pub near_star: Option<f64>,
    /// Headless: thrust forward (with boost) for the whole flight.
    pub burn: bool,
    /// Headless: engage the orbit autopilot on the nearest planet.
    pub autopilot: bool,
    pub start: start::Start,
    pub optics: render_hq::post::Settings,
}

const USAGE: &str = "\
kerr-nucleus: fly through a star cluster around a spinning black hole

USAGE:
    kerr-nucleus [OPTIONS]

OPTIONS:
    --stars N           Cluster size (default 1500)
    --fov DEG           Vertical field of view (default 75)
    --scale S           Fixed ray-tracing resolution scale 0.2–1 (default: adaptive)
    --start WHERE       cluster (default): orbiting 800 AU from the hole;
                        planet: in a 420 km orbit around the nearest Earth-like
                        world, heading into the sunrise over its limb;
                        planet:KIND[:ALTITUDE[:VIEW]] for other planets and views:
                          KIND ocean, rocky, desert, ice, lava, gas, icegiant,
                               ringed or any (nearest of that kind)
                          ALTITUDE km, or planet radii with an r suffix (2r)
                          VIEW dawn, day (sun glint), limb, nadir, night, disc
    --look NEBULA[@PC]  Face a nebula (sgra, minispiral, cnd, sgra-east, pwn) instead;
                        with @PC, from PC parsecs away on Earth's side, at rest
    --optics KIND       eye, camera (default) or astro (a telescope)
    --palette P         true (default) or hubble (narrowband [S II], Hα, [O III])
    --ev STOPS          Exposure compensation (astro: over its base exposure)
    --headless WxH      Render offscreen and save a PNG instead of opening a window
    --seconds S         Headless: seconds of flight before the shot (default 2)
    --out FILE          Headless: output path (default kerr-nucleus.png)
    --near-star K       Headless: start K stellar radii from the nearest star
    --burn              Headless: thrust forward with boost during the flight
    --autopilot         Headless: fly into orbit around the nearest planet
    -h, --help          Show this help

CONTROLS:
    Click to steer with the mouse (Esc releases it), I inverts mouse Y.
    W/S thrust, A/D strafe, Space/C up/down, arrows turn, Q/E roll,
    Shift boost (5x), X brake (to the local rest frame: the planet or star
    whose gravity dominates, else the hole's frame),
    [ / ] or the mouse wheel: throttle down / up by 10x (the engine gives
    17,000 g at full throttle; near a planet the throttle resets to a
    power of ten above its surface gravity, and back to full away from it),
    O orbit autopilot: fly to the targeted (else nearest) planet and into a
    circular orbit above its atmosphere; O again or any thrust takes over,
    Tab target the next planet of the system,
    , / . halve / double the time warp (near a body it is capped so an
    orbit takes at least 5 s), F11 fullscreen, Ctrl+Q quit.
    P cycles eye, camera and astrograph; H toggles the Hubble palette;
    PageDown / PageUp exposure down / up a stop; Backspace resets it.
    Telemetry is shown in the window title.

PLANETS:
    Stars and planets pull on the ship with Newtonian gravity added to the
    Kerr geodesic. Flying into a planet lands you on it (thrust lifts off);
    flying into a star puts you back outside it.

SCALE:
    Real Sagittarius A*: 4.3 million solar masses, stars from ~100 AU out to
    ~20,000 AU. From that far the hole is smaller than a pixel: look for the
    lensing of the stars behind it, or fly in. Distances are huge; accelerate
    to high gamma and time dilation makes the trips short in ship time.

ENVIRONMENT:
    WGPU_BACKEND=vulkan|metal|dx12  Force a graphics backend
";

fn parse() -> Result<Options, String> {
    let mut o = Options {
        stars: 384,
        fov: 75.0,
        scale: None,
        headless: None,
        seconds: 2.0,
        out: "kerr-nucleus.png".into(),
        near_star: None,
        burn: false,
        autopilot: false,
        start: start::Start::Cluster,
        optics: Default::default(),
    };
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
            "--near-star" => {
                o.near_star = Some(value("--near-star")?.parse().map_err(|e| format!("--near-star: {e}"))?)
            }
            "--burn" => o.burn = true,
            "--autopilot" => o.autopilot = true,
            "--start" => o.start = value("--start")?.parse()?,
            "--look" => o.start = start::Start::look(&value("--look")?)?,
            "--optics" => {
                o.optics.optics = match value("--optics")?.as_str() {
                    "eye" => render_hq::optics::Optics::Eye,
                    "camera" => render_hq::optics::Optics::Camera,
                    "astro" => render_hq::optics::Optics::Astro,
                    other => return Err(format!("unknown optics {other:?} (eye, camera or astro)")),
                }
            }
            "--palette" => {
                o.optics.palette = match value("--palette")?.as_str() {
                    "true" => render_hq::optics::Palette::True,
                    "hubble" => render_hq::optics::Palette::Hubble,
                    other => return Err(format!("unknown palette {other:?} (true or hubble)")),
                }
            }
            "--ev" => o.optics.ev = value("--ev")?.parse().map_err(|e| format!("--ev: {e}"))?,
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

/// Sagittarius A* at its real scale: the hole, its stars and you, placed
/// according to `start`.
pub fn world(stars: u32, start: start::Start) -> Result<World, String> {
    let mut world = World::new(kerr::world::WorldConfig::sgr_a(stars as usize, 1));
    if let Some(place) = start::apply(&mut world, start)? {
        eprintln!("start: {place}");
    }
    Ok(world)
}
