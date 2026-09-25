# Kerr Nucleus — an interactive portfolio

A portfolio set inside a relativistic galactic nucleus. A spinning (Kerr) black
hole sits at the centre of a star cluster. Every star follows an exact Kerr
geodesic, the stars pull on each other with weak, light-speed-delayed forces,
and tight orbits decay by emitting gravitational waves. You fly a small ship
through this spacetime. Portfolio projects are stations on real orbits that
you can fly to and dock with.

Everything you see is traced back along your past light cone, so
gravitational lensing, aberration, Doppler colour shifts and forward beaming
all come out of the same physics. The simulation clock is your ship's proper
time: fly fast and the rest of the universe runs ahead of you.

A plain HTML version of the same content is baked into the page. Crawlers,
browsers without WebGPU and anyone who clicks **Plain version** get that
instead.

> **Content is placeholder.** Edit `web/src/content/portfolio.ts`. That one
> file drives the plain page and the stations.

## Layout

```
crates/kerr      Physics core (pure Rust, tested natively)
crates/shaders   WGSL sources, validated natively with naga
crates/render    wgpu renderer + per-frame session for the web (its frame
                 building is shared with the desktop)
crates/render-hq The desktop's spectral compute renderer: planets, atmospheres,
                 nebulae, the ship, small-hole lensing, physical optics
crates/engine    wasm-bindgen front end for the web page
crates/desktop   Native desktop app (no portfolio content)
web/             Vite + TypeScript: content, plain page, immersive UI
scripts/         build-wasm.sh
docs/physics.md  What is simulated, how, and which parts are approximations
docs/planets.md  Design of the planets and atmospheres
```

### How the pieces fit

* **`kerr`** covers the Kerr metric in Cartesian Kerr–Schild coordinates
  (with analytic gradients checked against automatic differentiation),
  Hamiltonian geodesics, the N-body cluster, the pilot's thrusting worldline
  with a Fermi–Walker tetrad, and past-light-cone image finding with lensing
  magnification and redshift. `World::step` advances everything by one frame
  of the pilot's proper time.
* **`engine`** mirrors the cluster's worldline history into GPU buffers and
  runs three passes each frame:
  1. A compute pass that finds every body's direct image and its image from
     around the far side of the hole.
  2. A per-pixel null-geodesic ray tracer for the shadow, the lensed galaxy
     and the station cards, which are ray traced into the scene.
  3. Tone mapping, followed by point-spread sprites for the bodies.

  Each frame it returns the stations' apparent screen positions to the page.
* **`web`** pins HTML labels to those positions, lists the stations for the
  autopilot, and opens the full project content when you dock. It also paints
  the station cards that the engine renders into the scene. On browsers that
  support the [HTML-in-Canvas](https://github.com/WICG/html-in-canvas)
  proposal (currently behind a flag in Chromium), those cards are real HTML
  drawn with `drawElementImage`. Everywhere else they are drawn with Canvas 2D.

## Running locally

Requirements: Rust (stable) with the `wasm32-unknown-unknown` target,
`wasm-bindgen-cli` **0.2.128** (it must match the `wasm-bindgen` crate), and
Node 20+.

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.128 --locked

cd web
npm install
npm run wasm      # builds the engine into web/src/wasm/pkg
npm run dev       # http://localhost:5173
npm run build     # static site in web/dist (deployable anywhere)
```

URL switches:

| Query        | Effect                                                       |
| ------------ | ------------------------------------------------------------ |
| `?plain`     | Force the plain page (remembered via the "Plain version" link) |
| `?immersive` | Force the WebGPU nucleus (also overrides reduced-motion)     |
| `?noGui=off` | Show the HTML GUI (see feature flags below)                  |
| `?debug`     | Exposes `window.kerrDebug` (engine, controls, snapshot)      |
| `?debug&headless` | Renders offscreen and captures frames. For automated checks in headless browsers, which cannot display WebGPU canvases |

Feature flags live in `web/src/flags.ts`. Each flag has a default and can be
overridden from the URL with `?name=on|off`:

| Flag    | Default | Effect |
| ------- | ------- | ------ |
| `noGui` | on      | Hides all HTML controls in the immersive mode (title bar, station list, labels, HUD, dock panel, toasts, help, touch pad), leaving only the rendered scene. Keyboard and mouse flight, `1`–`8` autopilot and docking still work. Station cards drawn into the scene are part of the world and stay. |

The immersive mode starts automatically when WebGPU is available, unless the
visitor prefers reduced motion or last chose the plain version.

## Desktop app

`crates/desktop` builds `kerr-nucleus`. It's a native window with the same
physics on Vulkan, Metal or DX12, set at Sagittarius A*'s real scale, drawn
by its own renderer (`crates/render-hq`). It has no stations or portfolio
content.

- **The hole:** 4.3 million solar masses. One `M` of time is 21 s, and 1 AU is
  23.6 M.
- **The stars:** 384 real stars by default (`--stars N` for more), from about
  100 AU out to 20,000 AU. They
  have Salpeter masses, main-sequence and red-giant radii, luminosities and
  temperatures, plus stellar-mass black holes. Close stars are ray traced as
  limb-darkened discs with granulation; distant ones are points.
- **Planets:** most dwarf stars have 1–6 procedural planets (rocky, ocean,
  desert, ice, lava, gas and ice giants, some with rings) on Kepler orbits,
  with terrain, oceans, spectral atmospheres and clouds.
- **Nebulae:** the Minispiral, the circumnuclear disk, Sgr A East and a
  Crab-like pulsar wind nebula, at their real places and brightnesses.
- **What you see:** light is carried in 16 wavelength bins, so Doppler shifts,
  Rayleigh scattering and emission lines are exact. Brightness is physical;
  a camera (or eye, or telescope) model turns it into an image, with
  diffraction spikes, lens ghosts and exposure that adapts to the scene. From
  most places Sgr A* itself is smaller than a pixel: you find it by the
  lensing of what's behind it, or by flying in. The cluster's stellar-mass
  holes lens too.
- **Travel:** distances are real and the speed limit is `c`. Accelerate to
  high γ and time dilation makes the trips short in ship time. Your clock
  drives the simulation, and `,` / `.` warp it. It starts at 1000× real time.

```sh
cargo run -p desktop --release                  # window
cargo run -p desktop --release -- --help        # options
cargo run -p desktop --release -- --headless 1920x1080 --seconds 5 --out shot.png
cargo run -p desktop --release -- --start planet --chase          # in orbit, ship in view
cargo run -p desktop --release -- --start planet:ringed:3r:disc   # a ringed giant
cargo run -p desktop --release -- --look pwn@5 --optics astro --palette hubble
cargo run -p desktop --release -- --headless 1280x720 --near-hole 50 --out hole.png
```

Controls work like Kerbal Space Program's (`F1` lists them in the window):
- `W`/`S` pitch (`W` puts the nose down), `A`/`D` yaw, `Q`/`E` roll. The
  ship turns with inertia. SAS (`T`) stops it turning, and `1`–`9` make SAS
  hold the nose on attitude, prograde, retrograde, normal, anti-normal,
  radial out, radial in, the target or away from it.
- The main engine has a throttle that stays where you set it: `Shift` /
  `Ctrl` open and close it, `Z` is full and `X` cuts it. `[` / `]` scale
  the engine's thrust limit by 10. At a limit of 1, full throttle is
  35,000 g; near a planet the limit resets to the power of ten just above
  its surface gravity, and back to 1 away from it.
- `R` toggles RCS: `H`/`N` forward/back, `J`/`L` left/right, `I`/`K` up/down.
- `B` (held) brakes to the local rest frame: the planet or star whose
  gravity dominates, else the hole's frame.
- `O` flies to the targeted (else nearest) planet and into a circular orbit
  above its atmosphere, warping time on the way. `O` again or the throttle
  takes over. `Tab` targets the next planet of the system.
- `M` opens the map: the cluster, the star systems (ringed stars have
  planets) and your orbit. Right drag turns it, the wheel zooms from low
  orbit out to the whole cluster, `F` cycles the focus, clicking a planet
  targets it, and double-clicking centres on anything.
- `,` / `.` halve or double the time warp, and `/` returns to real time.
  Near a body the warp is capped so an orbit takes at least 5 s.
- `V` switches between first person and the chase camera; right drag turns
  the chase camera and the wheel zooms it. `Home` resets the camera or map.
- `P` cycles the optics: eye, camera, astrograph (telescope). `Y` toggles
  the Hubble palette. `PageUp` / `PageDown` change the exposure by a stop,
  `Backspace` resets it.
- `F2` hides the HUD, `F11` toggles fullscreen and `Ctrl+Q` quits.

The HUD shows what the ship is doing relative to the body whose gravity
dominates (a planet inside its Hill sphere, a star whose field reaches you,
else Sgr A*): altitude, speed, periapsis, apoapsis and period, and the
target. The navball has the ship's nose in the middle, that body's horizon,
and the prograde, normal, radial and target markers. Prograde, retrograde,
the target, the local star and Sgr A* are marked over the view too. At high
speed the stars crowd towards prograde (aberration), which can make it look
as if you are flying backwards; the prograde marker shows which way you
really move.

Stars and planets pull on the ship (Newtonian gravity on top of the Kerr
geodesic). Flying into a planet lands you on it; thrust lifts off. Flying
into a star puts you back outside it. `--start planet` begins in a 420 km
orbit around an Earth-like world, heading into the sunrise.

The ray-tracing resolution adapts between
50% and 100% unless you pass `--scale`. Set `WGPU_BACKEND=vulkan|metal|dx12`
to pick a backend. CI builds binaries for Linux, Windows and macOS as
workflow artifacts.

**RT cores.** The desktop app doesn't use hardware ray tracing yet. The light
paths are curved and are integrated step by step on the shader cores; stars,
planets and atmospheres are tested analytically. The ship is the one triangle
mesh, traced through a software BVH; hardware ray queries for it are a
possible next step.

Environment switches for checking the renderer: `KERR_GPU_TIMING=1` prints
per-pass GPU times after a headless run, `KERR_DEBUG_NAN=1` marks non-finite
pixels in magenta, `KERR_ATMO=off|noclouds`, `KERR_NEBULAE=0` and
`KERR_NEBULA_CUBE=0` turn features off.

## Controls

| Input | Action |
| ----- | ------ |
| `W` / `S` | Thrust forward / back |
| `A` / `D` | Strafe |
| `Space` / `C` | Thrust up / down |
| Click, then mouse; arrow keys | Turn (`Esc` releases the mouse, `I` inverts up/down; on touch, drag the sky) |
| `Q` / `E` | Roll |
| `Shift` | Boost (6× thrust) |
| `X` | Match velocity with the nearest station |
| `1`–`8`, click a label | Autopilot there and dock |
| `Esc` | Cancel autopilot / undock |
| `P` | Save a photo |
| `H` | Help |

## Tests

```sh
cargo test --workspace          # physics, frame layout, WGSL validation
cargo clippy --workspace --all-targets
cargo clippy -p engine --target wasm32-unknown-unknown
cd web && npm run typecheck
```

The physics tests check, among other things:

* conservation of energy, axial angular momentum, the Carter constant and
  the mass shell on a generic inclined orbit;
* circular orbits returning on phase after one period;
* lensing against the point-mass lens equation, including the second image
  and its magnification;
* the flat-space limit of image finding (retarded position, Doppler factor);
* the Peters–Mathews flux from radiation reaction;
* retarded, velocity-extrapolated sources in uniform motion pointing at the
  source's present position;
* autopilot docking, including at high time scales;
* respawning after crossing the horizon.

## Editing content

`web/src/content/portfolio.ts` holds the owner, a short bio and links, and up
to seven projects. Station 0 is the home station (about and contact), and each
project gets the next station. Accent colours tint the stations' cards and
labels.
