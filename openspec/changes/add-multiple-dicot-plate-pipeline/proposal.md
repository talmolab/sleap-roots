## Why

Plate-based dicot experiments (e.g. the Medicago MK22 dataset) record one frame per timepoint with multiple plants per plate. The existing `MultipleDicotPipeline` is tuned for cylinder scans (~72 rotational frames per series, aggregated to summary stats) and drops frames whose detected plant count differs from the expected count. That frame-drop behavior is catastrophic on plates: one frame == one timepoint, and losing a frame can lose the whole series. Plates also need **per-plant scalar output** (one row per plant per frame) rather than cross-frame summary stats, so that a future timelapse analysis works without architectural changes. Issue [#126](https://github.com/talmolab/sleap-roots/issues/126) asks for a dedicated `MultipleDicotPlatePipeline` class.

This proposal covers PR 1 of a 3-PR decomposition laid out in `docs/superpowers/specs/2026-04-16-multiple-dicot-plate-pipeline-design.md`: the core pipeline class with primary + lateral root handling. Tertiary roots are deferred to PR 2, configurable filtering thresholds to PR 3, and standardization of multi-plant JSON output / plate-depth trait / plate-aware CSV column names across other pipelines to follow-up issues A/B/C/D/E/F (filed during pre-merge — see tasks).

## What Changes

- **NEW** `MultipleDicotPlatePipeline` class in `sleap_roots/trait_pipelines.py`, extending `Pipeline` via `@attrs.define`.
  - `define_traits()` returns TraitDef DAG: `primary_pts_no_nans`, `lateral_pts_no_nans`, `detected_count`, `plant_associations_dict`, `plant_id_order`.
  - `get_initial_frame_traits(series, frame_idx)` returns `{primary_pts, lateral_pts, primary_sleap_idxs, lateral_sleap_idxs, expected_count}`, where the `*_sleap_idxs` lists preserve original SLEAP instance indices through the downstream `filter_roots_with_nans` collapse.
  - `compute_plate_traits(series, ...)` runs the frame loop and emits a per-series dict with a flat `plants` list (one entry per plant per frame).
  - `compute_batch_plate_traits(all_series, ...)` concatenates per-series CSV rows across multiple series.
  - Frame loop is built from day one to support >1 frame per series (future timelapse) without architecture changes.
  - **Skips** `filter_plants_with_unexpected_ct` entirely — plates keep all detected plants regardless of expected count, surfacing the discrepancy via `expected_count` and `detected_count` columns.
- **NEW** `argsort_primaries_by_base_x` helper in `sleap_roots/points.py`. Takes the output of `associate_lateral_to_primary` and returns primary instance indices sorted left-to-right by base-node (node 0) x-coordinate. Stable sort; empty input → empty list.
- **Per-plant trait output** reuses `DicotPipeline` trait names **unchanged** (no renaming): `primary_length`, `lateral_count`, `lateral_lengths`, `network_length`, `primary_base_tip_dist`, etc. — emits the full `DicotPipeline.csv_traits` set per plant.
- **Deterministic `plant_id`** assigned left-to-right by primary base x; output ALSO emits `primary_sleap_idx` and `lateral_sleap_idxs` so consumers can map any plant back to the source `.slp` file.
- **Output** per-series dict structure (JSON-serialized via a plate-aware extension of `NumpyArrayEncoder` that emits NaN as JSON `null`):
  - Top-level: `schema_version` (int = 1), `units` (str = "pixels"), `series`, `group`, `qc_fail`, `expected_count`, `plants` (list).
  - Per plant: `frame`, `plant_id`, `primary_sleap_idx`, `lateral_sleap_idxs`, `primary_points`, `lateral_points`, `expected_count`, `detected_count`, `count_validated` (bool), `count_mismatch` (bool), `traits` (full DicotPipeline output).
  - CSV: metadata columns (`series, frame, plant_id, primary_sleap_idx, expected_count, detected_count`) followed by the full `DicotPipeline.csv_traits` set. `count_validated` / `count_mismatch` are NOT CSV columns (consumers derive from raw expected/detected).
- **Warning log on count mismatch**: when any plant row has `count_mismatch=True`, the pipeline MUST emit one `warnings.warn(..., UserWarning)` per series naming detected vs expected counts. Transferred from #125's scope-split comment to #126.
- **JSON NaN emission**: the plate JSON encoder MUST emit Python / numpy NaN values as JSON `null` (not the non-standard bare `NaN` token that Python's default `json.dumps` produces). Written JSON files MUST round-trip through strict RFC-8259 parsers (tested by asserting `"NaN"` does not appear as a literal in the written file content).
- **NEW** capability spec: `openspec/specs/multiple-dicot-plate-pipeline/` (orthogonal addition; distinct from the existing `multiple-dicot-pipeline` cylinder capability).
- **NEW** top-level re-export: `MultipleDicotPlatePipeline` added to `sleap_roots/__init__.py` alongside `DicotPipeline`, `MultipleDicotPipeline`, etc., so users can `from sleap_roots import MultipleDicotPlatePipeline`.

**Explicitly NOT in this PR** (deferred per design doc):

- Tertiary root support → PR 2 (issue to be filed).
- Configurable filtering thresholds (`min_primary_length_px`, `node_score_threshold`, etc.) → PR 3 (issue to be filed).
- Raw points + deterministic plant_id in other multi-plant pipelines (`MultipleDicotPipeline`, `MultiplePrimaryRootPipeline`) → follow-up issue A.
- Raw points in single-plant pipeline JSON (`DicotPipeline`, `YoungerMonocotPipeline`, `OlderMonocotPipeline`) → follow-up issue B.
- Plate visualization / viewer → follow-up issue C (related to #128).
- Real plate `.slp` fixture tests (MK22 dataset) → follow-up issue D.
- Generalizing cylinder-conventional CSV column names read by `Series.qc_fail` (`qc_cylinder`), `Series.expected_count` (`number_of_plants_cylinder`) for plates → follow-up issue E.
- Plate-specific "depth" trait matching #126's "max y-extent" semantics (PR 1 substitutes `primary_base_tip_dist`) → follow-up issue F.

**Not breaking.** Pipeline base class untouched. No changes to existing `MultipleDicotPipeline`, `DicotPipeline`, or any other pipeline. No changes to `filter_plants_with_unexpected_ct`, `filter_roots_with_nans`, `associate_lateral_to_primary`, or existing Series methods.

## Impact

- **Affected specs**: new capability `multiple-dicot-plate-pipeline` (ADDED requirements for the pipeline class, the helper, and output format). No modifications to existing specs.
- **Affected code**:
  - `sleap_roots/trait_pipelines.py` — add `MultipleDicotPlatePipeline` class AND a plate-aware JSON encoder (either extending `NumpyArrayEncoder` to emit NaN→null or adding a sibling encoder class).
  - `sleap_roots/points.py` — add `argsort_primaries_by_base_x` helper.
  - `sleap_roots/__init__.py` — add `MultipleDicotPlatePipeline` to the top-level re-exports (parity with existing pipelines).
  - `tests/test_points.py` — unit tests for the helper.
  - `tests/test_trait_pipelines.py` — unit + integration tests for the pipeline (synthetic `.slp` round-trip via `sio.save_slp` + `Series.load`; real plate fixtures deferred to issue D).
- **Source of truth for architecture**: `docs/superpowers/specs/2026-04-16-multiple-dicot-plate-pipeline-design.md` — this proposal references it; do not duplicate. See that document for decisions D1, D2 (incl. zero-laterals handling), D3, D4, D5 + D5b (incl. JSON-NaN, schema_version, units, count_mismatch/count_validated flags), D6, D7, the SLEAP instance index mapping mechanism, the `primary_root_depth` substitution rationale, and the 8 follow-up issues (PR 2, PR 3, A, B, C, D, E, F).
- **Reproducibility**: all new traits are in pixel units (no DPI conversion); JSON output is self-contained (includes raw points + original SLEAP indices + trait values) so analyses can be rerun from the JSON alone if the `.slp` file is ever lost.
- **Dependencies**: unblocked by issue [#125](https://github.com/talmolab/sleap-roots/issues/125) (optional `expected_count`), which landed in PR [#155](https://github.com/talmolab/sleap-roots/pull/155).
