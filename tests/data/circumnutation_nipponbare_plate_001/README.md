# `circumnutation_nipponbare_plate_001` test fixture

## Purpose

Real-data input fixture for circumnutation `kinematics.compute` reference-value sanity tests (OpenSpec change `add-circumnutation-tier0-kinematics`, PR #2). This is **the same plate** that produced the kinematic numbers in `docs/circumnutation/preliminary_results_2026-05-07.md` §3.2 / §3.5 / §4.1 — committing the proofread `.slp` here lets Tier 0 (and subsequent tier PRs #5–#7) be regression-tested against the only plate that has been kinematically characterized end-to-end.

Subsequent tier PRs (Tier 1 temporal CWT, Tier 2 `ψ_g`, Tier 3 spatial wavelet) all reference this plate as the empirical anchor — they will reuse this fixture rather than committing a duplicate.

## Imaging geometry

- One agar plate, 6 plants per plate, top-down imaging.
- ~48 h imaging window at 5-min intervals → 575 frames.
- Single-node skeleton (`r0`) — only the primary-root tip is annotated and tracked. The pipeline supports multi-node skeletons too, but this fixture is the minimal case (one node = the tip).
- Image filenames carry datetime stamps (e.g. `_set1_day1_20250913-165722_001.tif`); the per-frame timestamp / source-filename metadata is intentionally NOT shipped with this fixture (deferred to issue #186, same as the KitaakeX fixture).

## Acquisition context

- Experiment: `20250917_Suyash_Patil_CMTN_Nipponbare_0.8PG_GA4vsTZT`
- Acquisition date: 2025-09-13 — 2025-09-15
- Researcher: Suyash Patil (Talmo Lab / Busch Lab, Salk Institute, Harnessing Plants Initiative)
- Genotype: Nipponbare (rice, *Oryza sativa* cv. Nipponbare)
- Treatment: MOCK (control)
- Imaging substrate: 1/2 MS (0.60% Phytagel) — note: the experiment name `Nipponbare_0.8PG_GA4vsTZT` refers to other plates in this experiment at 0.8% PG; plate 001 specifically is at 0.60% PG per its source `Plate001_META.csv`.
- Plate-level metadata also includes 3 sibling plates (Nipponbare × 2.5 µM GA₄ plate 002, Nipponbare MOCK rep plate 003, Nipponbare × 0.2 µM TZT plate 004) — only plate 1 is used for this fixture; the others are not committed.

## Contents

| File | Size | Tracked via | Description |
|---|---|---|---|
| `plate_001_greyscale.tracked_proofread.slp` | 362 KB | Git LFS | Tracked + proofread SLEAP predictions: 575 frames, 6 tracks (`track_0` … `track_5`), single-node skeleton, HDF5 video backend (the .h5 file itself is NOT shipped — sleap-io only opens the .slp directly for trait extraction in our pipeline; image-based code paths require the `.h5` separately). The proofread file contains `instance_type=0` (user-corrected) instances where available, `instance_type=1` (predicted) otherwise — per `preliminary_results_2026-05-07.md` §3.1, user-corrected values take precedence. |
| `fixture_metadata.csv` | <1 KB | (regular text file) | Synthesized plate-level metadata, single row, `plant_qr_code="plate_001"`. Provides `Series.sample_uid` / `Series.timepoint` lookup support for circumnutation pipeline tests. |
| `README.md` | this file | (regular text file) | Fixture documentation. |

## Conversion provenance — source → fixture

### `plate_001_greyscale.tracked_proofread.slp`

Copied **verbatim, no editing** from:
```
\\multilab-na.ad.salk.edu\hpi_dev\users\eberrigan\circumnutation\20250917_Suyash_Patil_CMTN_Nipponbare_0.8PG_GA4vsTZT\runs\run_20250917_201037\plate_001_greyscale.tracked_proofread.slp
```

Originated as the output of an external SLEAP tracking step on the corresponding `plate_001_greyscale.h5` (untracked predictions in `plate_001_greyscale.slp` were tracked downstream into `plate_001_greyscale.tracked.slp`, then user-proofread to produce `plate_001_greyscale.tracked_proofread.slp`). The full pipeline that produces the `.h5` from raw `.tif` files (issue #187) is upstream of sleap-roots and not run here.

### `fixture_metadata.csv`

Synthesized **by hand** (one-time, NOT a script — script-driven fixture generation defeats the purpose of frozen test data). Source row from `\\multilab-na.ad.salk.edu\hpi_dev\users\eberrigan\circumnutation\20250917_Suyash_Patil_CMTN_Nipponbare_0.8PG_GA4vsTZT\Plate001_META.csv`:

```
DATA_Location,\\pbiob-centos.salk.edu\groot-data\Suyash\AA_Suyash_MDATA\Circumnutation project\Images every 5 minutes\Nipponbare_0.8PG_GA4vsTZT
Experiment_Name,Nipponbare_0.8PG_GA4vsTZT
Plate_ID,1
Genotype,Nipponbare
Treatment,MOCK
Media,1/2 MS
Phytagel concentration,0.60%
Number of seedlings,6
Number of frames,575
Time between scans,5 minutes
Start frame,_set1_day1_20250913-165722_001
End frame,_set1_day1_20250915-164856_001
```

Mapped to the existing repo CSV convention so `Series.get_metadata` lookups work without modification (mirrors the KitaakeX fixture at `tests/data/circumnutation_plate/fixture_metadata.csv`):

| Source column | Fixture column | Transformation |
|---|---|---|
| `Plate_ID=1` | `plant_qr_code="plate_001"` | Reformatted as `plate_{Plate_ID:03d}`. **Legacy column-name caveat**: the column is named `plant_qr_code` (the existing convention from cylinder pipelines) but the value here is a PLATE identifier, not a plant identifier. Renaming the column to something like `series_qr_code` is tracked in #163. |
| `Genotype` | `genotype` | Pass-through, lowercased column name. |
| `Treatment` | `treatment` | Pass-through, lowercased column name. |
| `Number of seedlings=6` | `number_of_plants_cylinder=6` | Reused the existing repo column name. **Misnomer caveat**: there is no cylinder here (this is a plate). Renaming the column is tracked in #163. |
| (none) | `timepoint=0` | Invented for the fixture — treats this single time-point experiment as `t=0`. The source's `Number of frames=575` and `Time between scans=5 minutes` are not absorbed into `fixture_metadata.csv` (per-frame timing is deferred to #186). |
| `Media`, `Phytagel concentration`, `Number of frames`, `Time between scans`, `Start frame`, `End frame`, `DATA_Location`, `Experiment_Name` | (dropped) | Not needed by `Series.get_metadata` lookups for the trait-extraction tests. The cadence (5 min) is hard-coded in the test `CircumnutationInputs(cadence_s=300.0)` constructions that consume this fixture. |

## Known limitations

- **Per-frame metadata NOT shipped.** The source `run_20250917_201037/plate_001_metadata.csv` (575 rows, one per frame, with `filename, datetime, frame, datetime_str` columns) is **deferred to issue #186** (per-frame metadata accessor on `Series`). This fixture's pipeline tests use integer `frame` indexing only and pass cadence (5 min) explicitly via `CircumnutationInputs(cadence_s=300.0)`.
- **HDF5 video file (`plate_001_greyscale.h5`, ~10.0 GB) NOT shipped.** Out of scope for this fixture — sleap-roots' pipeline path opens the .slp directly and does not require the underlying video file. Image-display code paths (e.g. `Series.plot`) require the .h5 and are not exercised by this fixture.
- **`plant_qr_code` is a misnomer for plate-level data.** The column carries plate identifiers (`plate_001`); the legacy name is preserved for compatibility with the existing `Series.get_metadata` lookup. Rename tracked in #163.
- **`number_of_plants_cylinder` is a misnomer.** Same legacy reason — the column happens to carry a meaningful "number of plants on this plate" value (6), but the schema name is wrong for plate-tracking experiments. Rename tracked in #163.
- **Only plate 1 of 4 is committed.** The full experiment has 4 plates (Nipponbare MOCK, Nipponbare GA₄, Nipponbare MOCK rep, Nipponbare TZT). Multi-plate batch testing for circumnutation aggregation is exercised with synthetic `.slp` files in later PRs, not multi-plate real data.

## Related issues

- **#197** — circumnutation umbrella epic.
- **#198 / #200** — circumnutation foundation (PR #1) and its sub-issue.
- **#129 / #190** — TrackedTipPipeline (PR #190's fixture at `tests/data/circumnutation_plate/` is the KitaakeX precedent that this fixture mirrors).
- **#163** — broaden CSV column conventions (will fix `plant_qr_code` and `number_of_plants_cylinder` naming).
- **#186** — per-frame metadata accessor on `Series` (will let downstream consumers load the per-frame timestamp CSV).
- **#187** — preprocessing helpers (image folder → `.h5` + per-frame metadata CSV). Captures the upstream of what produced the source `.h5` and `_metadata.csv` files.
- **#188** — generic source-META → sleap-roots-CSV converter. Will eventually replace the by-hand synthesis described above.
