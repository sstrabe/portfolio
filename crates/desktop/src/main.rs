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
mod flight;
mod headless;
mod hud;
mod input;
mod map;
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
    /// Headless: start this many Schwarzschild radii from the nearest
    /// stellar-mass black hole.
    pub near_hole: Option<f64>,
    /// Headless: thrust forward (with boost) for the whole flight.
    pub burn: bool,
    /// Headless: engage the orbit autopilot on the nearest planet.
    pub autopilot: bool,
    /// Start with the chase camera (the ship in view).
    pub chase: bool,
    /// Headless: pitch and translate with the RCS during the flight.
    pub rcs: bool,
    /// Headless: draw the HUD, the map, the list of controls.
    pub hud: bool,
    pub map: bool,
    pub help: bool,
    pub start: start::Start,
    pub optics: render_hq::post::Settings,
}

const USAGE: &str = "\
kerr-nucleus: fly through a star cluster around a spinning black hole

USAGE:
    kerr-nucleus [OPTIONS]

OPTIONS:
    --stars N           Cluster size (default 384)
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
                        ground[:SITE][:LAT[:HOUR[:VIEW[:HEIGHT]]]]: standing on the
                          Earth-like world at latitude LAT (degrees, default 20) at
                          local solar time HOUR (default 15), HEIGHT metres above the
                          ground (default 1.7); SITE here (default), land (the nearest
                          land) or coast (the nearest shore, facing the sea); VIEW
                          horizon (the sun on the right, or the sea), sun, down, sky
    --look NEBULA[@PC]  Face a nebula (sgra, minispiral, cnd, sgra-east, pwn) instead;
                        with @PC, from PC parsecs away on Earth's side, at rest
    --optics KIND       eye, camera (default) or astro (a telescope)
    --palette P         true (default) or hubble (narrowband [S II], Hα, [O III])
    --ev STOPS          Exposure compensation (astro: over its base exposure)
    --headless WxH      Render offscreen and save a PNG instead of opening a window
    --seconds S         Headless: seconds of flight before the shot (default 2)
    --out FILE          Headless: output path (default kerr-nucleus.png)
    --near-star K       Headless: start K stellar radii from the nearest star
    --near-hole K       Headless: start K Schwarzschild radii from the nearest
                        stellar-mass black hole, facing it (50: an Einstein ring
                        about 20° across around a 5° shadow)
    --burn              Headless: thrust forward with boost during the flight
    --autopilot         Headless: fly into orbit around the nearest planet (else Sgr A*)
    --chase             Start with the chase camera (the ship in view)
    --rcs               Headless: pitch down and translate forward on the RCS
                        (its plumes show while the turn speeds up: --seconds 0.3)
    --hud               Headless: draw the HUD over the shot
    --map               Headless: shoot the map instead
    --help-overlay      Headless: draw the list of controls
    -h, --help          Show this help

CONTROLS (like Kerbal Space Program; F1 shows them in the window):
    W/S pitch (W: nose down), A/D yaw, Q/E roll. The ship turns with
    inertia (the RCS fires as it speeds up or slows down); SAS (T) stops it
    turning, and 1-9 (or the buttons left of the navball) make SAS hold the
    nose on:
    1 attitude, 2 prograde, 3 retrograde, 4 normal, 5 anti-normal,
    6 radial out, 7 radial in, 8 target, 9 anti-target.
    Shift/Ctrl throttle up/down, Z full, X cut. [ / ] divide or multiply
    the engine's thrust limit by 10 (at the limit of 1, full throttle is
    35,000 g; near a planet the limit resets to a power of ten above its
    surface gravity, and back to 1 away from it).
    R toggles RCS: H/N forward/back, J/L left/right, I/K up/down.
    B (hold) brakes to the local rest frame: the planet or star whose
    gravity dominates, else the hole's frame.
    O orbit autopilot: fly to the targeted (else nearest) planet and into a
    circular orbit above its atmosphere; O again or the throttle takes over.
    Tab targets the next planet of the system, then Sgr A* (50 M out, where
    the shadow spans about 12 degrees); O with no planet in range flies there.
    M map: right drag turns it, the wheel zooms, F cycles the focus (ship,
    the body you orbit, the target, its star, the hole), a click on a planet
    targets it, a double click centres on anything, Home resets it.
    V chase camera / first person (right drag turns the chase camera, the
    wheel zooms it, Home resets it).
    , / . halve / double the time warp (near a body it is capped so an
    orbit takes at least 5 s), / back to real time.
    P cycles eye, camera and astrograph; Y toggles the Hubble palette;
    PageDown / PageUp exposure down / up a stop; Backspace resets it.
    F2 hides the HUD, F11 fullscreen, Esc closes the help or the map,
    Ctrl+Q quits.

HUD:
    Top left: what the ship is doing and its orbit around the body whose
    gravity dominates (a planet inside its Hill sphere, a star whose field
    reaches the ship, else Sgr A*), and the target. The navball shows the
    ship's nose in the middle, that body's horizon, and the prograde (yellow),
    normal (purple), radial (cyan) and target (pink) markers; prograde,
    retrograde, the target, the local star and Sgr A* are also marked over
    the view. At high speed aberration crowds the stars towards prograde.

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
        near_hole: None,
        burn: false,
        autopilot: false,
        chase: false,
        rcs: false,
        hud: false,
        map: false,
        help: false,
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
            "--near-hole" => {
                o.near_hole = Some(value("--near-hole")?.parse().map_err(|e| format!("--near-hole: {e}"))?)
            }
            "--burn" => o.burn = true,
            "--autopilot" => o.autopilot = true,
            "--chase" => o.chase = true,
            "--hud" => o.hud = true,
            "--rcs" => o.rcs = true,
            "--map" => o.map = true,
            "--help-overlay" => o.help = true,
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

/// Sagittarius A* at its real scale: the hole, its stars and you (on the
/// cluster start; see [`start::apply`] for the others).
pub fn world(stars: u32) -> World {
    World::new(kerr::world::WorldConfig::sgr_a(stars as usize, 1))
}
