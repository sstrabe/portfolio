# Ground-level terrain: how it's built and how to extend it

The desktop renderer (`crates/render-hq`) draws walkable ground down to
centimetres on worlds with air. This note is the map of that machinery and
the recipes for adding to it. The plan and its progress are in
`docs/HANDOFF.md` (the high-fidelity terrain row); the planet-scale design is
in `docs/planets.md`.

## The pieces

| Piece | Where | What it does |
|---|---|---|
| Analytic terrain | `wgsl/terrain.wgsl` | Heights and macro channels from noise: plates, hotspot islands (`tn_hotspots`), coast profiles (`terrain_coast`). Point-evaluable anywhere. |
| Climate map | `terrain/maps.rs`, `wgsl/climate.wgsl` | A cube map of temperature, rain, vegetation and dryness, baked per planet. |
| Regional erosion | `terrain/erosion.rs`, `wgsl/erosion.wgsl` | Stream-power erosion over a ~128 km square around the eye (125 m cells), added inland by the coarse tiles. |
| Tile pyramid | `terrain/tiles.rs`, `tilegen.rs`, `wgsl/tile_gen.wgsl` | Cube-sphere quadtree tiles (128² samples) generated on the GPU: coarse ones from the analytic terrain, finer ones refining their parent with anchored octaves, gullies and boulders. |
| Precision anchor | `terrain/anchor.rs`, `wgsl/anchor.wgsl` | Lattice noise exact to 1 mm near the eye on a planet thousands of km across. |
| Ray tracing | `terrain/rt.rs`, `wgsl/terrain_rq.wgsl` | A BLAS per tile, plants as instances, a TLAS rebuilt each frame; primary, shadow, sky and reflection rays. |
| Materials | `terrain/materials.rs`, `assets/ground/` | Scanned PBR texture sets laid on the face grid. |
| Biomes | `terrain/biomes.rs`, `planet.wgsl` | What the land is made of, as a table of rows. |
| Boulders | `terrain/rocks.rs`, `wgsl/boulders.wgsl` | Faceted rocks in the heightfield. |
| Plants | `terrain/plants.rs` | Procedural meshes (palms, grass tufts) placed around the eye by rays cast down. |
| Ground for physics | `terrain/ground.rs` | Tile heights read back to the CPU; `kerr::world::Ground`. |

## Adding a ground material

1. Put its four maps in `crates/render-hq/assets/ground/<name>/`, as 1024²
   JPEGs named like the others (`<name>_diff_1k.jpg`, `_nor_gl_1k.jpg`,
   `_disp_1k.jpg`, `_arm_1k.jpg`). Poly Haven's CC0 sets come in exactly
   this form; note the source in the directory's `README.md`.
2. Add a line to `MATERIALS` in `terrain/materials.rs`:
   `scanned!("SLOT", "<name>", repeat_level, relief_m)`. `repeat_level` sets
   the size of one repeat: level 22 is a tile of ~2.4 m on an Earth, 21 is
   twice that.
3. The shaders now have `GM_SLOT`. Up to 8 materials fit
   (`MAX_MATERIALS`).

The texture only varies the brightness, normals and ambient occlusion. The
colour is the biome's spectral reflectance, so a scan of any tint works.

## Adding a biome

Add a row to `BIOMES` in `terrain/biomes.rs`. A row has:

* **Where:** soft windows (`Window::between`, `above`, `below`, or `ANY`)
  on temperature (K), moisture (0 desert to 1 rainforest, from the
  climate), height above the sea (m), steepness (1 − cos slope),
  ruggedness (0 lowland to 1 the cores of ranges) and rock type (0 granite
  to 1 basalt).
* **What:** a spectral reflectance, built from the helpers there (`ramp`,
  `bump`, `mix`, and recipes like `vegetation(dryness)`, `soil(dryness)`,
  `beach_sand()`). Also the ground materials it shows (by slot name, with
  shares), its plant cover, and its precedence `weight`.

The shader weighs every row by the product of its windows times its
weight and blends them. Narrow biomes (a beach) win over broad ones (the
forest behind) by a larger weight. A row naming a missing material fails at
start-up. Up to 24 rows fit (`MAX_BIOMES`). The tests in `biomes.rs` show
how to check a point comes out as the biome you meant.

## Adding a plant

1. Write a mesh generator in `terrain/plants.rs` returning a `PlantMesh`
   (metres, z up from the foot, one normal and part per triangle; see
   `palm` and `grass_tuft`). Parts pick the shading: bark, leaf, dead leaf
   or grass (`plant_material` in `planet.wgsl`; leaves are translucent).
2. Add its variants to the list in `PlantGeometry::new` (`terrain/rt.rs`),
   after the existing ones, and give it a variant range like
   `PALM_VARIANTS` and `TUFT_VARIANTS`.
3. Give it candidate sites (cells of the face grid, hashed so they're the
   same from anywhere, as `palm_candidates` and `grass_candidates` do) and
   a rule in `plants_from_hits` for the ground it takes (height above the
   sea, for now).
4. Its sway comes from `PlantInstance::axes`.

Plants are placed within a range of the eye and stay under
`MAX_PLANTS` instances.

## Rules that keep it working

* **Seams:** a tile's heights must depend only on the parent's heights and
  material at the point, world position, and exact face-grid coordinates.
  Its shared edges then agree bit for bit with its neighbours'
  (`tiles_on_the_gpu`, a by-hand GPU test, checks this). The parent's
  slope from texels a sample either side is safe; anything else needs the
  same care.
* **Precision:** anything finer than a metre uses the anchored lattice
  (relative to the anchor, never absolute f32 positions) or face-grid
  coordinates built from tile indices.
* **The physics sees tiles:** landing and walking read tile heights back
  (`ground.rs`), so anything added to tile heights is solid ground.
  Anything drawn but not in the tiles (plants) is not solid.
* **Checks before a commit:** fmt, clippy (native and wasm), and all tests.
  Also run the by-hand GPU tests (`cargo test -p desktop --release --
  --ignored --nocapture`) after changing tile generation or the ground.
