# Procedural planets and atmospheres: design

This is the design for planetary systems in the desktop renderer (`render-hq`,
see `HANDOFF.md` §6). Nothing here is implemented yet.

- **Target GPU:** RTX 3060 Laptop, 6 GB VRAM, second-generation RT cores.
  About 16 ms per frame at 1080p, tracing at roughly 0.67× resolution with
  temporal upscaling.

Everything is deterministic from `(cluster seed, star id)`. Generation runs in
four layers, each derived from the one before:

```
star  ──►  1. system (orbits, masses, composition)                    CPU, µs
      ──►  2. planet state (climate, volatiles, atmosphere recipe)    CPU, µs
      ──►  3. surface maps (plates, height, erosion, biomes, clouds)  GPU compute, ~1 s/planet, progressive
      ──►  4. atmosphere optics (spectral LUTs)                       GPU compute, ~10 ms/planet
render: local-bubble ray tracer (terrain, ocean, clouds, atmosphere, rings), spectral
```

Layers 1–2 are pure physics and live in `kerr` (`systems.rs`), with unit
tests. Layers 3–4 and rendering live in the renderer.

---

## 1. Planetary systems

**Which stars have planets.** Nearly all dwarf stars from F to M do: draw
0–7 planets (Kepler statistics give about one or more per star). The
environment then trims them:

- **Hill radius in the SMBH's tidal field:**
  `r_H = d · (m★ / 3 M_BH)^(1/3)`.
  - For a 1 M☉ star at 1000 AU, r_H ≈ 4.3 AU.
  - Prograde orbits are kept only inside `0.4 r_H`. At 100 AU that is
    about 0.17 AU, so systems deep in the cluster are compact.
- **Stellar encounters.** The nuclear cluster is dense, so outer orbits are
  truncated further and eccentricities and inclinations are raised.
  Rayleigh-distributed e and i, with σ scaled by the local stellar density.
- **Age.**
  - The old population (about 8 Gyr, metal-rich, [Fe/H] ≈ +0.2) has mature
    systems.
  - The young S-stars (B-type, about 6–10 Myr) have only molten
    protoplanets and debris discs.

**Formation-flavoured placement**, a cheap stand-in for population synthesis:

- **Snow line:** `r_snow ≈ 2.7 AU · (m★/M☉)²`.
- **Solid surface density:** `Σ ∝ Z · r^-1.5`, multiplied by about 4 beyond
  the snow line.
- **Spacing:** place planets outward at a mutual Hill spacing `K ~ 𝒩(20, 6)`.
  Neighbours get similar masses ("peas in a pod").
- **Mass:** start from the isolation mass. Cores above about 10 M⊕ that
  formed beyond the snow line become gas giants (the probability rises
  with Z).
- **Hot Jupiters:** about 1% of systems migrate a giant to 0.03–0.1 AU.
- **Water mass fraction:** 0.01–0.3% inside the snow line, 10–50% beyond it
  (water worlds).
- **Core mass fraction:** 0.2–0.45.

**Radius from mass** (Chen & Kipping 2017):

| Class | Radius relation |
| --- | --- |
| Terran, M < 2 M⊕ | R ∝ M^0.279 |
| Neptunian, up to 0.41 M_J | R ∝ M^0.589 |
| Jovian | R ∝ M^-0.044 |

Terran radii are adjusted by the water and core fractions.

**Spin:**

- rotation period log-uniform, 8 h to 5 d;
- obliquity with σ ≈ 20°, occasionally large;
- **tidal locking** when `t_lock ∝ a⁶ ω Q / (m★² k₂ R³)` is shorter than
  the age. This is common for M-dwarf habitable-zone planets and leads to
  "eyeball" worlds.

**Moons:** giants get 0–5 large moons, which go through the same rocky
pipeline. Terrestrial planets get a large moon about 10% of the time.

**Motion.** Orbits are Keplerian in the star's rest frame. The renderer
evaluates them at the ray's own local time: light takes about 500 s to
cross 1 AU, and in that time a planet moves about 2 Earth radii.

---

## 2. Planet state: climate, volatiles, atmosphere recipe

**Equilibrium temperature:** `T_eq = T★ √(R★ / 2a) (1 − A)^¼`. Iterate with
the albedo A from step 3's first pass.

**Does it keep an atmosphere?** Use the "cosmic shoreline" (Zahnle & Catling
2017): lifetime XUV dose `I_XUV` against `v_esc⁴`, calibrated so that:

- Mercury and the Moon sit below the line (bare);
- Mars sits near it (thin);
- Earth, Venus and Titan sit above it.

M dwarfs give about 100× the XUV dose per unit bolometric dose. There's an
optional knob for Sgr A\*'s past active (AGN) phase, which would have
stripped more.

**Surface pressure.**
- Volatile inventory ∝ M.
- `p_s = g · M_atm / 4πR²`.
- Log-normal scatter of σ = 1 dex around that.

**Composition class**, chosen by temperature regime, water and life:

| Regime | Atmosphere | Examples |
| --- | --- | --- |
| Temperate, liquid water, age > 1 Gyr, life roll | N₂–O₂ + H₂O, trace O₃ | Earth |
| Temperate, abiotic | N₂–CO₂ | early Earth |
| Runaway greenhouse (inside the inner HZ edge) | CO₂ 10–100 bar, H₂SO₄ clouds | Venus |
| Cold, thin | CO₂ ~0.01 bar + dust | Mars |
| Cold, thick, CH₄ present | N₂–CH₄ + tholin haze | Titan |
| Lava world (T_eq > 1500 K) | Na/SiO vapour, thin | K2-141b |
| Sub-Neptune / giant | H₂–He + CH₄, cloud decks by T_eq | Jupiter, Neptune |

**Habitable zone** (Kopparapu): `0.95–1.67 · √(L/L☉)` AU.

**Greenhouse.** Grey optical depth τ from the CO₂, H₂O and CH₄ partial
pressures, then `T_s = T_eq (1 + ¾τ)^¼`.

**Latitudinal climate.** A 1-D diffusive energy-balance model (North 1975)
on 90 latitude bands:

- inputs: annual-mean insolation for the obliquity, plus ice-albedo
  feedback;
- outputs: the zonal-mean temperature and the ice-line latitude.

Tidally locked planets use the angle from the substellar point instead of
latitude.

**Giant cloud class** (Sudarsky):

| Class | T_eq | Clouds | Look |
| --- | --- | --- | --- |
| I | < 150 K | NH₃ | Jupiter bands |
| II | < 250 K | H₂O | white |
| III | 350–800 K | cloudless | deep blue (Rayleigh + CH₄) |
| IV | > 900 K | Na/K absorption | dark |
| V | > 1400 K | silicate | grey-brown |

---

## 3. Surface maps (GPU compute, cube-sphere)

Everything is stored as 6 cube faces with a tangent-warped cube mapping,
which gives nearly uniform texel area.

### 3a. Tectonics-lite

1. **Plates:** 8–40 seeds spread over the sphere by Fibonacci sampling with
   jitter. Take a spherical Voronoi, then grow the plates by noisy flood
   fill so the boundaries are ragged rather than straight Voronoi edges.
2. **Plate type and motion.** Each plate is continental or oceanic, filled
   up to the planet's continental crust fraction (Earth: 0.4), and gets an
   Euler pole and angular speed.
3. **Boundaries.** At each boundary texel, take the relative velocity:
   - its normal component gives convergent (+) or divergent (−);
   - its tangential component gives a transform boundary.
4. **Elevation field.**
   - Base height: continent +0.4 km, ocean −4 km. Ocean floor deepens with
     the age since the ridge (√age).
   - Convergent boundaries, by the two sides meeting:

     | Plates meeting | Features |
     | --- | --- |
     | continent–continent | wide high range (Himalaya) |
     | ocean–continent | trench plus coastal range (Andes) |
     | ocean–ocean | trench plus island arc |

   - Divergent boundaries: mid-ocean ridges and rift valleys.
   - Distance fields to each boundary class come from jump flooding.
5. **Tectonic regime.**
   - With liquid water and M > 0.5 M⊕: plate tectonics, as above.
   - Otherwise a stagnant lid (Mars, Venus): 1–3 "plates" and no ranges, but
     large volcanic shields (height ∝ 1/g, Olympus Mons) and dichotomy basins.

### 3b. Detail and erosion

- **Macro heightmap:** 6 × 1024² R16F (12.6 MB), or 2048² for the planet
   you're visiting.
- **Detail noise** within a band, weighted by the tectonic fields:
  - ridged multifractal where the "orogeny" field is high;
  - smooth fBm on plains;
  - derivative-damped fBm (Quilez's "eroded" noise) to fake drainage.
- **Hydraulic erosion** on the macro map, only for planets with liquid water
  now or in the past:
  - 200–500 iterations of a grid-based model: water, sediment, thermal
    slumping;
  - about 0.3 s on the 3060 at 1024².
- **Impact craters**, with the density set by the age of the surface
  (resurfacing, erosion, atmosphere):
  - sizes follow `N(>D) ∝ D⁻²`;
  - shapes change with size: simple bowls, then central peaks, then
    multi-ring basins;
  - craters are degraded by age;
  - craters are shaped by a closed-form height function with an ejecta
    blanket.
- **Sea level:** solve so the ocean volume matches the water inventory from
  the hypsometric curve. Water worlds can drown every continent.

### 3c. Climate and biomes, per texel

- **Temperature:** the EBM zonal mean, minus a 6.5 K/km lapse rate, plus
  ocean moderation (a blurred land mask).
- **Precipitation** comes from:
  - Hadley/Ferrel/polar cells: wet at the ITCZ and around 60°, dry around
    30°;
  - prevailing winds (trades, westerlies, sign set by the rotation
    direction);
  - an orographic rain shadow, marched upwind over the heightmap;
  - distance to the ocean along the wind.
- **Materials** come from a Whittaker diagram of (temperature, precipitation)
  plus height and slope. Each material has a **16-sample reflectance
  spectrum over 200–1600 nm**, so the red edge, water absorption and iron
  oxides come out right under any Doppler shift:
  - basalt, granite and regolith;
  - iron-oxide desert, sand, salt flats;
  - snow and ice (bright, falling in the near infrared);
  - lava (emissive blackbody plus crust);
  - **vegetation**, only if life is present: chlorophyll dips at 450 and
    680 nm, a green bump, and the **red edge** at 700 nm. Pigments shift
    with the host star; for M-dwarf planets that is flagged as speculative.
- **Ocean:**
  - the absorption spectrum of pure water (Pope & Fry) plus depth, which
    gives shallow turquoise and deep navy;
  - whitecaps as a function of wind speed.

### 3d. Clouds

- **Coverage and type map:** 6 × 2048² RGBA8 (100 MB), holding coverage,
  type, top height and optical depth. It is built from:
  - ITCZ and storm-track bands from the circulation model;
  - **cyclones**: log-spiral warps of curl noise at seeded centres in the
    storm tracks, handed by hemisphere;
  - **marine stratocumulus decks**: cellular Worley noise over cold
    upwelling water on western coasts (the km-scale texture in the Earth
    reference photo);
  - orographic cloud on windward slopes;
  - polar stratus.
- **3D detail:** a tiled 128³ Perlin–Worley texture plus a 32³ erosion
  texture, as in Guerrilla's Nubis.
- **Advection:** zonal winds shift the pattern slowly; cyclones drift and
  decay.
- **Condensate by class:** H₂O (Earth), CO₂ ice (cold), H₂SO₄ (Venus: a
  global 100% deck with the unknown UV absorber, so it looks yellowish),
  CH₄ (Titan).

### 3e. Gas giants

- **2D flow simulation** on a 2048 × 1024 lat–long grid:
  - vorticity–streamfunction form, forced towards alternating zonal jets
    (the number of jets scales with rotation rate and radius, following
    Rhines);
  - about 3000 steps at load time, around 1–2 s on the GPU;
  - it produces real shear turbulence, festoons and long-lived ovals
    (Great Red Spots).
- **Dye fields** are advected through the flow:
  - zones are bright (NH₃ ice);
  - belts are dark (NH₄SH with chromophores);
  - Sudarsky classes swap the palette.
- **Stored as** 2048 × 1024 RGBA16F (16 MB), with procedural detail on top.
- **Rings**, for about 30% of giants:
  - they lie between 1.1 R and the ice Roche limit, `2.44 R (ρ_p/ρ_ice)^⅓`;
  - the radial optical-depth profile has gaps at moon resonances;
  - composition is water ice (bright) or dusty silicate (red);
  - they cast shadows on the planet and receive the planet's shadow.

---

## 4. Atmosphere optics: the whole nine yards

This follows Hillaire (2020), "A Scalable and Production Ready Sky and
Atmosphere Rendering Technique", extended from RGB to spectral and to
per-planet gas mixtures.

**Spectral samples.** Use 16 log-spaced wavelengths over **200–1600 nm**, not
just the visible range. A fast flyby Doppler shifts the visible band to
emitted wavelengths `λ_e = g λ_o`, so g ∈ [0.5, 2] stays inside the table.
Blueshifted views then correctly show the UV look (ozone's Hartley band
darkens the planet). Outside the range, Rayleigh is extrapolated
analytically.

**Rayleigh scattering from first principles, per gas species.**
For each species with mole fraction x:

```
σ_R(λ) = 24π³/(N_s² λ⁴) · ((n(λ)²−1)/(n(λ)²+2))² · F_K(λ)
```

- `n(λ)` comes from the dispersion formula (Peck & Fisher for N₂/air,
  plus O₂, CO₂, Ar, H₂, He and CH₄).
- `F_K` is the King depolarization factor: N₂ 1.034, O₂ 1.096, CO₂ 1.15,
  about 1 for monatomic gases.
- The mixture gives `β_R(λ, h) = Σ x σ_R · n_0 e^(−h/H)`, with:
  - `n_0 = p_s/(k T_s)`;
  - scale height `H = k T / (μ m_u g)`: Earth 8.4 km, Mars 11 km,
    Titan 20 km.

**Checks:**
- Earth must reproduce β_R = (5.8, 13.6, 33.1) × 10⁻⁶ m⁻¹ at
  680 / 550 / 440 nm to within 3%.
- A CO₂ atmosphere scatters about 2.5× more per molecule than N₂.
- H₂ scatters about 4× less than N₂.

**Aerosols and hazes (Mie):**

| Aerosol | Parameters | Effect |
| --- | --- | --- |
| Earth sulfate / sea salt | β_s ≈ 4 × 10⁻⁶ m⁻¹, H = 1.2 km, g = 0.76, single-scattering albedo ≈ 0.9 | haze |
| Mars dust | r ≈ 1.5 µm, iron-oxide imaginary index (absorbs blue), strong forward scattering | butterscotch day sky, **blue sunset** |
| Titan tholin haze | strong blue absorption, H ≈ 40–60 km | orange |
| Venus H₂SO₄ deck | 45–70 km, τ ≈ 30 | featureless yellow-white |

The phase function is Cornette–Shanks, or a two-lobe Henyey–Greenstein for
dust.

**Absorbers:**
- **Ozone**, only in O₂ atmospheres:
  - a tent profile peaking at about 25 km;
  - Chappuis band cross-section, about 5 × 10⁻²⁵ m² at 600 nm (this gives
    the deep-blue zenith at twilight);
  - Hartley band in the UV.
- **CH₄ bands** at 619, 727 and 890 nm: why Uranus and Neptune are
  cyan and blue.
- **H₂O vapour:** negligible in the visible, included for the near-IR
  samples.

**Lookup tables, per planet, generated when a system activates:**

| LUT | Size | Spectral packing |
| --- | --- | --- |
| Transmittance (μ, h) | 256 × 64 | 4 × RGBA16F (16 λ) |
| Multiple scattering (μ_sun, h) | 32 × 32 | 4 × RGBA16F |
| Sky view (per frame, only inside the atmosphere) | 192 × 108 | 4 × RGBA16F |
| Aerial perspective (per frame, near the ground) | 32³ froxels | 4 × RGBA16F |

The total is under 5 MB per planet, and each takes about 10 ms to generate.

**Rendering from space** (the limb in the reference photo):
- March the view ray through the atmosphere shell per pixel, 32–48 samples,
  distributed exponentially in height.
- At each sample: sun transmittance from the LUT, multiple scattering from
  its LUT, phase functions for Rayleigh and Mie, and the cloud shadow term.
- The limb glow, the terminator's reddening and the twilight wedge all come
  out of this with no special cases.

**Light sources:**
- the host star as a **limb-darkened disc of finite size**, with a
  blackbody spectrum at T★ (soft terminator and correct sunsets);
- **the cluster's night sky:** in the Galactic Centre hundreds of stars
  outshine Venus. Their summed irradiance (from the `images` pass)
  gives a real, faintly lit night side, and it is also used for the eye's
  adaptation.

---

## 5. Rendering inside the local bubble

**Frame.** The Kerr ray enters the system's bubble and is transformed into
the star's rest frame. Positions are in km relative to the planet, which
gives f32 precision of about 0.5 m at Earth's radius. The camera tetrad is
boosted per pixel, so aberration and Doppler stay exact.

**Per ray, in order:**
1. Bounding tests: planets (top of the atmosphere), ring planes, the star's
   disc, moons.
2. **Surface intersection:**
   - Far away, the ground is a sphere shaded with normals from the macro
     heightmap plus detail, which is exact enough for orbit views.
   - Below about 50 km of altitude, and near the horizon, use **RT cores**:
     - cube-sphere chunked LOD patches of 65 × 65 vertices near the camera;
     - a BLAS per chunk, refit as chunks split and merge, all in one TLAS;
     - hardware ray queries for primary hits and **shadow rays** (mountain
       shadows at low sun).
   - Fallback without ray queries: sphere tracing of the height function
     with a Lipschitz bound.
3. **Clouds:** march the shell, 32–64 steps, skipping empty space using the
   coverage map. Lighting:
   - two-lobe Henyey–Greenstein;
   - Wrenninge multiple-scattering octaves;
   - a "powder" term;
   - a light march toward the sun of about 6 samples.

   The atmosphere is integrated in segments before, between and after the
   clouds.
4. **Shading:**
   - **Ocean:** GGX with Cox–Munk slope variance `σ² = 0.003 + 5.12×10⁻³ U`
     (U = wind speed, m/s). This gives the gold sun-glint streak, plus
     Fresnel, and subsurface colour from the water absorption spectrum.
   - **Land:** Oren–Nayar with the material's spectral albedo.
   - **Airless regolith:** Hapke, including the opposition surge.
   - **Snow:** a forward-scattering lobe.
   - **Lighting terms:** terrain shadows (ray query), cloud shadows (the
     cloud map sampled along the sun direction), sky irradiance from the
     LUT, and cluster starlight at night.
   - **Lava worlds** add blackbody emission, which is Doppler shifted like
     everything else.
5. The spectral radiance goes into the renderer's accumulator, together
   with the transmittance to anything behind.

**Budget on the 3060 Laptop** at about 1280 × 720 traced and upscaled to
1080p:
- atmosphere about 2 ms, clouds 3–5 ms, terrain and shading 2–3 ms;
- the remaining ~6 ms go to the Kerr trace, TAA and the FFT optics.

**VRAM for the active system** (1–2 detailed planets):

| Item | Size |
| --- | --- |
| Heightmap | 13–50 MB |
| Materials and climate | 25 MB |
| Cloud map | 100 MB |
| Noise volumes | 10 MB |
| Terrain BLASes | ~50 MB |
| Gas-giant flow | 16 MB |
| LUTs | < 5 MB |

That is about 300–500 MB in total. Other planets in the system use a 256²
per face preview, a few MB each.

**Streaming.** Generate on approach to a star, nearest planet first:
- 128² per face at once (milliseconds);
- the full resolution over about 1 s on the async compute queue.

Only planets that cover more than a pixel get anything beyond an analytic
point.

---

## 6. Validation

**Numeric tests (`cargo test`):**
- Earth β_R and scale height;
- T_eq(Earth) = 255 K;
- Hill radii;
- the mass–radius relation continuous at the class boundaries;
- LUT energy conservation (a zero-albedo ground in a white-scattering
  atmosphere loses no energy to multiple scattering);
- the cosmic shoreline classifies the Solar System correctly.

**Reference renders** (headless, `--planet <preset>`), compared by eye:

| Preset | Must show |
| --- | --- |
| Earth analogue | the reference limb shot |
| Mars | butterscotch sky, blue sunset |
| Titan | orange haze, blue upper haze layer |
| Venus | yellow-white deck |
| Jupiter | bands, Great-Red-Spot-like oval |
| Neptune | blue from CH₄ |
| Saturn-like | ring shadows |

These presets sit alongside the random generator.

---

## 7. Work split for agents

1. **`kerr::systems`:** layers 1–2 with the numeric tests. CPU only;
   can start now.
2. **Surface generation:** tectonics, erosion, craters, climate, biomes,
   spectral materials (compute shaders, headless dump of the maps).
3. **Atmosphere:** the spectral Rayleigh/Mie/absorber model, the LUTs, and
   the space-view march.
4. **Clouds:** weather maps, the volumetric shell, shadows.
5. **Gas giants and rings.**
6. **RT terrain:** chunked LOD, BLAS/TLAS, ray-query and sphere-tracing
   paths.

Items 2–6 plug into the `render-hq` local-bubble hook (HANDOFF §6), so that
skeleton comes first.
