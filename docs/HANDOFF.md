# Handoff: Kerr Nucleus

This is a snapshot for the next working session, which runs on the owner's
laptop so the app can be tested in a real window. Read it together with
`README.md` (how to build and run) and `docs/physics.md` (what is simulated).

- **Branch:** `claude/kerr-spacetime-portfolio-safh9l`. It is the only
  branch on the remote. There is no `main`, so a PR can't be opened yet.
  Keep pushing to this branch.
- **Last commit at handoff:** `c7ce9a1` ("Start facing prograde and warn
  before diving into the hole").
- **CI:** green on every pushed commit:
  - fmt, clippy (native and wasm), tests;
  - web build;
  - desktop builds for Linux, Windows and macOS, uploaded as
    `kerr-nucleus-<OS>` artifacts.

---

## 1. What the project is

It started as an interactive portfolio set inside a relativistic galactic
nucleus, and grew a standalone desktop app. It has two front ends over one
physics core and one renderer.

| Front end | What it is | Portfolio content |
| --- | --- | --- |
| **Web** (`web/` plus `crates/engine`, Rust → Wasm on WebGPU) | The original portfolio. Stations on real orbits carry project cards you can dock with. A plain HTML fallback is prerendered. | Yes. Placeholder content lives in `web/src/content/portfolio.ts`. |
| **Desktop** (`crates/desktop`, binary `kerr-nucleus`, wgpu on Vulkan / Metal / DX12) | A flight sim at the real scale of Sagittarius A\* | **None, by explicit request.** Keep it that way. |

**The owner's current priority is desktop visual quality.** The owner says
the web version "will be limited of course".

### Physics (crate `kerr`, pure Rust, f64)

- **The hole:** the Kerr metric in Cartesian Kerr–Schild form. Analytic
  gradients are checked against dual numbers.
- **Stars:** every star follows an exact geodesic, integrated in Hamiltonian
  form with an adaptive DOPRI5 solver.
- **Perturbations**, applied as 4-forces:
  - weak mutual pulls between stars, evaluated at the retarded time and
    extrapolated with the source's velocity;
  - 2.5PN radiation reaction on the compact objects.
- **The pilot:**
  - thrust is a proper acceleration, with a Fermi–Walker-transported frame;
  - the simulation clock is the pilot's proper time;
  - a "clock limiter" caps the coordinate time simulated per frame.
- **Image finding:** backward null geodesics with Newton iteration find
  each body's direct image and its image from around the far side of the
  hole, with lensing magnification and the redshift g.
- **`units.rs`:** Sgr A\* units, where M = 21.2 s = 6.4 × 10⁶ km and
  1 AU = 23.56 M. It also converts apparent magnitude to flux.
- **`WorldConfig::sgr_a`**, the desktop world:
  - 1,500 physical stars: Salpeter masses from 0.5 to 40 M☉,
    main-sequence and giant radii, luminosities and temperatures;
  - 8 compact objects;
  - star orbits from 100 to 20,000 AU; the pilot starts at 800 AU;
  - time scale 1000× real time; exposure adapts like an eye.

### Rendering (crate `render`, shaders in crate `shaders`)

`render` is shared by web and desktop. Per frame:

1. **`images` compute pass:** finds each body's images on the past light
   cone, reading the worldline history mirrored to the GPU.
2. **`sky` pass:** a per-pixel backward null geodesic, integrated with f32
   RK4 relative to the observer. It draws:
   - the horizon;
   - the procedural lensed galaxy;
   - station panels (web only);
   - up to 16 nearby stars as ray-traced discs.
3. **`composite` pass:** ACES tone mapping and upscaling, then the
   point-spread sprites for body images.

`session.rs` runs one frame: step the world, adapt exposure, sync history,
upload uniforms, render.

---

## 2. Working with the owner

- **Style:** they are direct and don't want padding.
  - Never add "if this is happening to you" or "if you are considering…"
    paragraphs.
  - Don't offer things that weren't asked for unless they're the obvious
    next step.
- **Sub-agents:** they asked for them to be used to speed up large work.
- **Realism:** they want realistic scale and physics, and they accept what
  realism implies (tiny hole from far away, aberration at high γ) as long
  as it's explained.
- **Commits:**
  - End every commit message with:
    ```
    Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
    Claude-Session: https://claude.ai/code/session_01M3Xv8txLS6qRyBrVrJrzVs
    ```
    Replace the session URL if the new session gives different attribution
    lines.
  - Don't put model names in commits, PRs or code.

---

## 3. Build, run, test

```sh
# Desktop (the main target now)
cargo run -p desktop --release
cargo run -p desktop --release -- --help
cargo run -p desktop --release -- --headless 1920x1080 --seconds 5 --out shot.png
cargo run -p desktop --release -- --headless 1280x720 --near-star 8 --out star.png   # disc close-up
cargo run -p desktop --release -- --headless 1280x720 --burn --seconds 6 --out burn.png  # high-γ view

# Checks (all must pass; CI enforces them)
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo clippy -p engine --target wasm32-unknown-unknown -- -D warnings
cargo test --workspace            # kerr has 34 tests; render and shaders validate layouts and WGSL

# Web
cd web && npm install && npm run wasm && npm run dev    # needs wasm-bindgen-cli 0.2.128 exactly
```

- **Desktop controls:**
  - Click to capture the mouse. Look is FPS-style (right looks right, down
    looks down); `Esc` releases and `I` inverts Y.
  - `W`/`S`, `A`/`D`, `Space`/`C` thrust; arrow keys turn; `Q`/`E` roll.
  - `Shift` boosts 5×, and `X` brakes to the local rest frame.
  - `,` / `.` halve or double the time warp (between 1× and 10⁷× real time).
  - `F11` toggles fullscreen and `Ctrl+Q` quits.
  - Telemetry goes to the window title: τ, universe time, warp, v or γ, r,
    nearest star, fps, the clock limiter, and a warning when you're
    heading into the hole.
- **`WGPU_BACKEND`** set to `vulkan`, `metal` or `dx12` forces a backend.
- **Headless testing without a GPU** (how the cloud session tested): use
  Chromium's SwiftShader Vulkan driver.
  `VK_ICD_FILENAMES=<chromium dir>/vk_swiftshader_icd.json WGPU_BACKEND=vulkan LD_LIBRARY_PATH=<chromium dir>`.
  On the laptop you can just run the window.
- **Web headless check:** `?immersive&debug&headless&noGui=off`. Headless
  Chromium can't present WebGPU canvases, so this renders offscreen and
  captures frames. Without the GUI, headless rAF throttling stalls it, hence
  `noGui=off`.

---

## 4. Pitfalls already hit (don't repeat them)

- **Cluster step size:** the cluster's DOPRI5 `h_max` must stay huge
  (1e12). A small `h_max` (it was 40) made the realistic world take more
  than 10 minutes to build.
- **Pilot substeps:** size them by curvature and thrust
  (`0.03 r / uᵗ`, `0.5 / (1 + 0.5 a)`). Too many tiny substeps at γ > 10⁴
  made γ drift through renormalization rounding. A regression test covers it.
- **f32 precision:** far from the hole, ray positions are relative to the
  observer (`abs_pos`). Keep everything near the camera observer-relative.
- **HDR overflow:** f16 HDR overflows if exposure gain is applied late.
  Sky adaptation is applied inside the sky shader (`frame.extra.w`).
- **Moving spheres:** they must be placed at the ray's hit time (iterate),
  not at a segment's mid-time. Segments can be hundreds of M long.
- **WGSL:** `meta` is a reserved word.
- **Web vs native:** the web build can't use native-only wgpu features.
  Hardware ray queries (`EXPERIMENTAL_RAY_QUERY`) are native only (Vulkan
  for sure; check wgpu 30.0.1 for DX12/Metal).
- **`pkill`:** `pkill -f <name>` can kill your own shell. Use a bracket
  pattern such as `"[k]err-nucleus"`.
- **Start heading:** the realistic start used to face the hole, so `W` meant
  diving into it at γ≈100 within about 15 s. It now faces prograde. Keep the
  dive warning (`Telemetry::impact_in`).

---

## 5. Open feedback from the owner

1. **Inverted controls:** reported, then fixed with FPS mouse-look plus the
   `I` toggle. Which axis felt wrong was never confirmed; ask when they test
   the window.
2. **"You can't see what you are":** no ship model or third-person view yet.
3. **Blurry:** improved (full-resolution start, magnitude-based stars, star
   discs). The owner said the updated scene "looks better".
4. **Stars fading when flying straight:** explained as aberration and
   Doppler at high γ. Exposure now includes Doppler boost.
5. **Their question about segmenting.** Could the ray tracer skip curvature
   when a long segment grazes a stellar-mass black hole?
   - Yes, and currently worse: **light isn't bent by the compact objects at
     all**. The only lens is Sgr A\*, and step lengths scale with distance
     from it.
   - Segment-vs-sphere *hit tests* are exact; only the bending is missing.
   - The planned fix is local bubbles, below.

---

## 6. The approved plan (not started): make the desktop look like the reference images

The owner shared two reference images and said **"Do all the things you
listed"**:
- **Earth limb from orbit:** sun starburst, lens ghosts, blue atmosphere
  limb, ocean sun glint, clouds.
- **The Crab Nebula** (Hubble composite).

Two questions were asked and **not yet answered**:
- **Which GPU is the desktop target?** On the laptop you can just look.
- **Planets:** procedural only, or also a literal Earth with NASA Blue
  Marble textures?

Default to procedural until told otherwise.

### Architecture decided so far

- **A new native-only renderer crate** (working name `render-hq`) for the
  desktop. It uses compute passes and reuses `kerr`, the `images` pass and
  the Kerr WGSL. The web keeps `render` unchanged, so the web stays safe
  while the desktop uses native features.
- **Spectral rendering.**
  - Carry about 32 wavelength bins (380–780 nm, 12.5 nm) through the trace
    and convert with the CIE colour-matching functions at the end.
    Narrowband separation of Hα 656 nm from [S II] 672 nm needs that bin
    width.
  - A source's spectrum is evaluated at the emitted wavelength
    λ_e = g·λ_o, and I_λ,obs = g⁵ I_λ,emit(g λ_o).
  - A blackbody stays analytic. Lines are Gaussians integrated over bins.
    Smooth spectra such as the atmosphere can use about 8 samples,
    interpolated.
- **Local bubbles.**
  - Near a star system or a small black hole, the Kerr ray hands off to a
    local tracer in that object's rest frame, built from an orthonormal
    tetrad at the object and boosted by its 4-velocity. Spacetime there is
    flat to about 10⁻¹².
  - The ray returns to the Kerr integrator when it leaves the bubble.
  - Only a handful of bubbles are relevant per frame (planets are
    sub-pixel beyond about 1 AU), so they are culled on the CPU and looped
    over linearly. No BVH is needed for them.
  - Planets move about 2 Earth radii per AU of light travel, so evaluate
    their positions at the ray's own time.
- **Where the RT cores go:**
  - the ship: primary visibility, shadows and self-reflections;
  - later, close-range terrain.

  Use wgpu ray queries on Vulkan (plus DX12/Metal if wgpu 30 supports them),
  with a small software BVH in WGSL as fallback. The ship moves with the
  camera, so it has no aberration and plain perspective is exact for it.
  Stars, planets, atmospheres and nebulae are analytic or volumetric and
  don't need the RT cores.
- **Bind groups:** one per feature to limit merge conflicts between
  parallel agents:
  - 0: frame, history, images, spectra;
  - 1: bubbles, planets and atmosphere LUTs;
  - 2: nebula volumes;
  - 3: ship and acceleration structure.

  Native can raise the bind-group limit if needed.

### Work items

1. **Optics and post-processing.** This is the biggest visible win, so do
   it first.
   - Star images and background stars become energy splatted at sub-pixel
     positions instead of glow sprites.
   - TAA with rotation reprojection and temporal upscaling, plus progressive
     accumulation when the view is still.
   - **FFT convolution with a physical point spread function:**
     - the aperture's diffraction pattern, computed per wavelength (spikes
       with colour fringes);
     - a scatter halo;
     - an energy-conserving normalisation.
   - Screen-space lens ghosts (aperture-shaped, coating tints).
   - AgX tone mapping.
   - An eye model with mesopic/scotopic desaturation of faint light.
   - An astrophotography mode: exposure time, true colour or the Hubble
     palette.
2. **Planetary systems.**
   - **Physics side (`kerr`):**
     - some dwarf stars get 1–6 planets on Keplerian orbits in the star's
       frame;
     - keep them inside the Hill radius (about 4 AU for 1 M☉ at 1000 AU);
     - planet types: rocky / ocean-world, desert, ice, lava, and gas giants
       with optional rings.
   - **Rendering side:**
     - terrain height functions;
     - oceans with Cox–Munk sun glint;
     - a cloud layer with shadows (volumetric clouds up close later);
     - banded gas giants and ring shadows.
3. **Atmospheres.** Use the Hillaire 2020 lookup tables: transmittance,
   multi-scattering, sky-view and aerial perspective. Include Rayleigh
   (λ⁻⁴), Mie and ozone, with parameters per planet type. From space, ray
   march the atmosphere using the lookup tables.
4. **Nebulae.**
   - Volumes stored as 3D textures of about 256³ (Hα, [O III], [S II]/[N II],
     dust) and generated procedurally on the GPU:
     - a shell with Rayleigh–Taylor fingers;
     - domain-warped ridged noise;
     - extra detail added while ray marching.
   - A synchrotron continuum, a pulsar, and line widths from the expansion
     speed.
   - **Placement at real positions:**
     - the Minispiral (Sgr A West);
     - the dark dusty circumnuclear disk (CND, 1.5–4 pc);
     - Sgr A East (a real supernova remnant, about 7 × 9 pc);
     - a Crab-like pulsar wind nebula as a showcase (fictional).

     All of these lie beyond the current 0.1 pc cluster, so the world
     extends outward.
   - The eye sees these faint. The camera mode shows the Hubble look.
5. **Your ship.** A procedural mesh, a third-person chase camera (`V`),
   physically based materials, lighting from the brightest image sources
   (from the `images` buffer), and ray-query shadows.
6. **Stellar-mass black hole lensing.** Local bubbles with their own null
   geodesic integration. The bubble radius must make the deflection left
   outside it sub-pixel (about 1 M for 10 M☉); the smooth option is a
   thin-lens kick outside the bubble.

### Suggested order and parallelisation

- **You:** first, the `render-hq` skeleton:
  - the compute trace loop with hook functions in separate WGSL files;
  - the spectral accumulator;
  - the bubble hand-off;
  - bind-group ownership as above;
  - desktop switched over to the new crate.
- **Then parallel agents in worktrees:**
  - A: optics and post-processing;
  - B: nebulae;
  - C: planetary systems (physics and surfaces);
  - D: atmospheres;
  - E: ship and RT;
  - F: small-black-hole lensing.

  Merge each in turn and run the full check suite before every push.
- **Test in the window on the laptop** after each merge. The owner will
  judge by eye.

---

## 7. Repo map

```
crates/kerr      physics core (metric, geodesic, integrate, orbit, history, cluster,
                 pilot, lensing, world, units, rng, dual, vec3)
crates/shaders   WGSL: common, sky(_header), images, history, sprites, composite
crates/render    gpu.rs (wgpu setup/passes), frame.rs (uniform building, spheres,
                 exposure), session.rs (per-frame loop)
crates/engine    wasm-bindgen wrapper for the web
crates/desktop   main.rs (options), app.rs (winit window, title HUD), input.rs, headless.rs
web/             Vite + TS; src/flags.ts (noGui flag, default on), src/immersive/*,
                 src/content/portfolio.ts
docs/            physics.md, this file
.github/workflows/ci.yml
```
