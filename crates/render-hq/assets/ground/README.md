# Ground texture sets

Scanned PBR texture sets for the ground's centimetre-scale detail
(`src/terrain/materials.rs`), from [Poly Haven](https://polyhaven.com),
released under **CC0** (public domain; no attribution required). Each set is
the 1k JPEG download: albedo (`diff`), OpenGL normals (`nor_gl`),
displacement (`disp`) and AO/roughness/metalness (`arm`).

| Directory | Poly Haven asset | Used as |
|---|---|---|
| `coast_sand_04` | [Coast Sand 04](https://polyhaven.com/a/coast_sand_04) | dry beach sand |
| `damp_beach_sand_02` | [Damp Beach Sand 02](https://polyhaven.com/a/damp_beach_sand_02) | wet sand |
| `dark_rock` | [Dark Rock](https://polyhaven.com/a/dark_rock) | basalt |
| `forest_ground_04` | [Forest Ground 04](https://polyhaven.com/a/forest_ground_04) | soil under vegetation |

To add a set: put its four maps (1024², the same file names) in a directory
here and add a line to `MATERIALS` in `src/terrain/materials.rs`.
