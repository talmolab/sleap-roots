# `circumnutation_plate` test fixture

## Purpose

Real-data input fixture for `TrackedTipPipeline` integration tests (issue #129, OpenSpec change `add-tracked-tip-pipeline`). Exercises the full pipeline path on tracked SLEAP predictions from a real circumnutation plate experiment.

## Imaging geometry

- One agar plate, 6 plants per plate, top-down imaging.
- ~5h imaging window at ~10-min intervals → 311 frames.
- Single-node skeleton (`['r0']`) — only the primary-root tip is annotated and tracked. The pipeline supports multi-node skeletons too, but this fixture is the minimal case (one node = the tip).
- Image filenames carry datetime stamps (e.g. `_set1_day1_20250730-212631_001.tif`); the per-frame timestamp / source-filename metadata is intentionally NOT shipped with this fixture (deferred to issue #186).

## Acquisition context

- Experiment: `20250819_Suyash_Patil_CMTN_Kitx_vs_Hk1-3_07-30-25`
- Acquisition date: 2025-07-30 — 2025-07-31
- Researcher: Suyash Patil (Talmo Lab / Busch Lab, Salk Institute, Harnessing Plants Initiative)
- Genotype: KitaakeX (rice, _Oryza sativa_ Kitaake-X background)
- Treatment: MOCK (control)
- Imaging substrate: 1/2 MS (0.6% Phytagel)
- Plate-level metadata also includes 3 sibling plates (KitaakeX × 1uM PAC, hk1-3 × MOCK, hk1-3 × 1uM GA3) — only plate 1 is used for this fixture; the others are not committed.

## Contents

| File | Size | Tracked via | Description |
|---|---|---|---|
| `plate_001_greyscale.tracked.slp` | 184 KB | Git LFS | Tracked SLEAP predictions: 311 frames, 6 tracks (`track_0` … `track_5`), single-node skeleton, HDF5 video backend (the .h5 file itself is NOT shipped — sleap-io only opens the .slp directly for trait extraction in our pipeline; image-based code paths require the `.h5` separately). All instances are tracked (zero untracked anywhere — verified during the brainstorm for #129). |
| `fixture_metadata.csv` | <1 KB | (regular text file) | Synthesized plate-level metadata, single row, `plant_qr_code="plate_001"`. Provides `Series.sample_uid` / `Series.timepoint` lookup support for the WITH-CSV integration test. |
| `README.md` | this file | (regular text file) | Fixture documentation. |

## Conversion provenance — source → fixture

### `plate_001_greyscale.tracked.slp`

Copied **verbatim, no editing** from:
```
Z:\users\eberrigan\circumnutation\20250819_Suyash_Patil_CMTN_Kitx_vs_Hk1-3_07-30-25\run_20250827_091833\plate_001_greyscale.tracked.slp
```

Originated as the output of an external SLEAP tracking step on the corresponding `plate_001_greyscale.h5` (untracked predictions in `plate_001_greyscale.slp` were tracked downstream into the `.tracked.slp`). The full pipeline that produces the `.h5` from raw `.tif` files (issue #187) is upstream of sleap-roots and not run here.

### `fixture_metadata.csv`

Synthesized **by hand** (one-time, NOT a script — script-driven fixture generation defeats the purpose of frozen test data). Source row from `Z:\users\eberrigan\circumnutation\20250819_Suyash_Patil_CMTN_Kitx_vs_Hk1-3_07-30-25\CMTN_KITXvsHK1-3_META.csv`, plate 1:

```
plate_number,treatment,num_plants,accesion,num_images,experiment_start,growth_media
1,MOCK,6,KitaakeX,311,,1/2 MS (0.6% Phytagel)
```

Mapped to the existing repo CSV convention so `Series.get_metadata` lookups work without modification:

| Source column | Fixture column | Transformation |
|---|---|---|
| `plate_number=1` | `plant_qr_code="plate_001"` | Reformatted as `plate_{plate_number:03d}`. **Legacy column-name caveat**: the column is named `plant_qr_code` (the existing convention from cylinder pipelines) but the value here is a PLATE identifier, not a plant identifier. Renaming the column to something like `series_qr_code` is tracked in #163. |
| `accesion` | `genotype` | Renamed (the source's `accesion` is the cultivar/accession, semantically equivalent to sleap-roots' `genotype` lookup). |
| `treatment` | `treatment` | Pass-through. |
| `num_plants=6` | `number_of_plants_cylinder=6` | Reused the existing repo column name. **Misnomer caveat**: there is no cylinder here (this is a plate). Renaming the column is tracked in #163. |
| (none) | `timepoint=0` | Invented for the fixture — treats this single time-point experiment as `t=0`. The source's `experiment_start` field is empty in CMTN_META and we are not modeling absolute datetime in this fixture. |
| `num_images, experiment_start, growth_media` | (dropped) | Not needed by `Series.get_metadata` lookups for this PR's tests. |

## Known limitations

- **Per-frame metadata NOT shipped.** The source `run_20250827_091833/plate_001_metadata.csv` (311 rows, one per frame, with `filename, datetime, frame, datetime_str` columns) is **deferred to issue #186** (per-frame metadata accessor on `Series`). This fixture's pipeline tests use integer `frame` indexing only.
- **HDF5 video file (`plate_001_greyscale.h5`, ~5.7 GB) NOT shipped.** Out of scope for this fixture — sleap-roots' pipeline path opens the .slp directly and does not require the underlying video file. Image-display code paths (e.g. `Series.plot`) require the .h5 and are not exercised by this fixture.
- **`plant_qr_code` is a misnomer for plate-level data.** The column carries plate identifiers (`plate_001`); the legacy name is preserved for compatibility with the existing `Series.get_metadata` lookup. Rename tracked in #163.
- **`number_of_plants_cylinder` is a misnomer.** Same legacy reason — the column happens to carry a meaningful "number of plants on this plate" value (6), but the schema name is wrong for plate-tracking experiments. Rename tracked in #163.
- **Only plate 1 of 4 is committed.** The full experiment has 4 plates (KitaakeX MOCK, hk1-3 GA3, hk1-3 MOCK, KitaakeX PAC). Multi-plate batch testing for `compute_batch_tracked_tip_traits` is exercised with synthetic `.slp` files, not multi-plate real data.

## Related issues

- **#129** — TrackedTipPipeline (this fixture's primary consumer).
- **#163** — broaden CSV column conventions (will fix `plant_qr_code` and `number_of_plants_cylinder` naming).
- **#168** — backfill test-data README documentation. This README ships in-PR with #129 (no documentation debt added).
- **#186** — per-frame metadata accessor on `Series` (will let downstream consumers load the per-frame timestamp CSV).
- **#187** — preprocessing helpers (image folder → `.h5` + per-frame metadata CSV). Captures the upstream of what produced the source `.h5` and `_metadata.csv` files.
- **#188** — generic source-META → sleap-roots-CSV converter. Will eventually replace the by-hand synthesis described above.
