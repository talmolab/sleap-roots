# Add `TrackedTipPipeline` (Workstream 2 of the 2026-04-23 timelapse design)

## Why

Issue #129 needs a root-agnostic pipeline that consumes **tracked** SLEAP `.slp` predictions and emits per-track tip-trajectory data plus a minimum-viable substrate of per-track geometric scalars. The existing `Series` accessors (`get_primary_points` / `get_lateral_points` / `get_crown_points`) call `.numpy()` on labels and DROP track identity — they cannot be reused for track-aware analysis.

This pipeline is the **substrate** for downstream tip-aware analyses (circumnutation, growth-rate, gravitropism). Its scope is deliberately frozen: trajectory rows + 5 per-track scalars (`n_frames_tracked`, `n_frames_total`, `tracking_coverage`, `tip_trajectory_length`, `tip_displacement_net`). Velocity, curvature, and circumnutation traits NEVER belong here — they live in separate downstream pipeline classes that REUSE this pipeline's `Series.get_tracked_tips` accessor and trajectory output as their input substrate. Keeping `TrackedTipPipeline` minimal makes it reusable across many downstream pipelines without coupling to any single analysis's opinions.

Workstream 1 (PR #171, archived) shipped the metadata layer this pipeline depends on (`Series.sample_uid`, `Series.timepoint`, `Series.get_metadata`). Workstream 3 (`TimeDiffPipeline`, #170) depends on this pipeline emitting per-row-over-time output.

## What Changes

### NEW pipeline class

- **NEW** `sleap_roots/tracked_tip_pipeline.py` — `TrackedTipPipeline(Pipeline)` class. Lives in its own file (NOT appended to `trait_pipelines.py`). Starts the per-pipeline-module pattern; the existing `trait_pipelines.py` megafile (3763 lines, 8 pipelines) split is tracked in **#189**.
- **NEW** `TrackedTipPipeline.compute_tracked_tip_traits(series, write_csv=False, write_json=False, output_dir=".", emit_trajectories=True, ...)` — single-series method. Returns a per-series result dict; optionally writes one summary CSV, one trajectory CSV (suppressed when `emit_trajectories=False`), and one JSON file.
- **NEW** `TrackedTipPipeline.compute_batch_tracked_tip_traits(all_series, ...)` — batch method mirroring the existing `compute_batch_plate_traits` pattern in [trait_pipelines.py:3327](sleap_roots/trait_pipelines.py#L3327). Walks the input list, calls the per-series method, concatenates per-series DataFrames, writes batch CSVs + a JSON list of per-series dicts.

### NEW Series accessor + validation helpers

- **NEW** `Series.get_tracked_tips(root_type: Optional[Literal["primary", "lateral", "crown"]] = None) -> pd.DataFrame` — returns long-format `(track_id, frame, tip_x, tip_y)` rows, sorted by `(track_id, frame)`. Iterates per-instance (not per-frame-stack) because tracker output does NOT preserve positional ordering across frames. Auto-detects `root_type` from whichever of `primary_path` / `lateral_path` / `crown_path` is populated. Raises `ValueError` on (a) zero or >1 paths populated with `root_type=None`, (b) any instance with `inst.track is None` or empty `inst.track.name`.
- **NEW** module-level `validate_tracked_slp(slp_path: Union[str, Path]) -> None` in `series.py` — opens the .slp, asserts every instance has a non-empty track, raises `ValueError` listing offending frame indices.
- **NEW** module-level `validate_series_for_tracked_tip(series: Series, root_type: Optional[str] = None) -> None` in `series.py` — composite check: validates the relevant `<root_type>_path`, asserts skeleton has ≥1 node, calls `validate_tracked_slp` on the resolved path.

### Trait functions — DAG-A composition (no new trait module)

The pipeline's TraitDef DAG reuses existing tested trait functions DIRECTLY. **No `tip_kinematics.py` module is created.** The DAG provides per-track input slicing (`track_first_xy`, `track_last_xy`); existing functions plug in unchanged:

- `tip_displacement_net` — TraitDef with `fn=bases.get_base_tip_dist`, `input_traits=["track_first_xy", "track_last_xy"]`. Single-frame returns `0.0` naturally (xy[0] == xy[-1]).
- `tip_trajectory_length` — TraitDef with `fn=lengths.get_root_lengths`, `input_traits=["track_xy"]`. Single-frame returns `NaN` (codebase's NaN-on-empty-segments convention from `get_root_lengths`).
- `tracking_coverage` — TraitDef with inline lambda `fn=lambda nt, ntot: nt/ntot if ntot else np.nan`, `input_traits=["n_frames_tracked", "n_frames_total"]`.

Single-frame edge-case asymmetry (`tip_displacement_net=0.0`, `tip_trajectory_length=NaN`) is **deliberate and documented** — the cost of using `get_root_lengths` directly without a wrapper. Users who want either filtered or coerced apply `fillna(0.0)` post-hoc; users who want only "real trajectories" filter on `n_frames_tracked > 1`.

### Output contract

**Two tables, both emitted in one call:**

1. **Trajectory rows** (`<series>.tracked_tip_trajectories.csv` — one row per `(track_id, frame)`):
   ```
   series, sample_uid, timepoint, track_id, frame, tip_x, tip_y
   ```
2. **Track summary rows** (`<series>.tracked_tip_traits.csv` — one row per `track_id`):
   ```
   series, sample_uid, timepoint, track_id, n_frames_tracked, n_frames_total,
   tracking_coverage, tip_trajectory_length, tip_displacement_net
   ```

**JSON** (`<series>.tracked_tip_traits.json`): single dict with `schema_version: 1`, `pipeline: "TrackedTipPipeline"`, structured `units` dict (`lengths: "pixels"`, `ratios: "dimensionless"`, `counts: "dimensionless"`, `time: "unspecified"`), top-level scalars (`series`, `sample_uid`, `timepoint`), and both tables under `{"tracks": [...], "trajectories": [...]}`. Top-level scalars do NOT repeat inside per-row arrays in JSON (they DO repeat in CSV — PR #165's CSV/JSON asymmetry).

**Batch**: `tracked_tip_batch_traits.csv` (concatenated summary), `tracked_tip_batch_trajectories.csv` (concatenated trajectory), `tracked_tip_batch_traits.json` (list of per-series dicts). Mirrors PR #165's `compute_batch_plate_traits` pattern.

**`emit_trajectories: bool = True` kwarg** on both compute methods — when `False`, skips writing the trajectory CSV and omits the `trajectories` array from JSON. At realistic data scale (~7,500 rows for a 4-plate experiment, ~750 KB) the kwarg is rarely needed, but it costs nothing.

### Real-data test fixture

- **NEW** `tests/data/circumnutation_plate/plate_001_greyscale.tracked.slp` — 184 KB tracked predictions sliced from `Z:\users\eberrigan\circumnutation\20250819_Suyash_Patil_CMTN_Kitx_vs_Hk1-3_07-30-25\run_20250827_091833\plate_001_greyscale.tracked.slp` (verified 2026-05-06: 311 frames, 6 tracks named `track_0..track_5`, single-node skeleton `['r0']`, HDF5 video backend, all instances tracked). Committed via Git LFS.
- **NEW** `tests/data/circumnutation_plate/fixture_metadata.csv` — synthesized BY HAND (one-time, not a script) from `CMTN_KITXvsHK1-3_META.csv`'s plate-1 row. Single row keyed `plant_qr_code="plate_001"`. Schema follows existing `plant_qr_code`-keyed convention (legacy column name; rename tracked in #163).
- **NEW** `tests/data/circumnutation_plate/README.md` — fixture documentation per the standard template (issue #168). Covers: source provenance, conversion steps, the `plant_qr_code` legacy-name caveat (value `"plate_001"` is a plate identifier, not a plant identifier), why per-frame metadata is NOT shipped (deferred to #186), related issues.

## Impact

### Affected specs

- **NEW capability** `tracked-tip-pipeline` — ADDED requirements covering: pipeline class location, output contract (CSV + JSON for per-series and batch), `emit_trajectories` kwarg, edge cases (zero tracks, single-frame, partial tracking, multiple paths populated, untracked instance, single-node skeleton, multi-node skeleton, track-order non-determinism), `Series.get_tracked_tips` accessor, validation helpers, trait DAG composition.

No existing capabilities are MODIFIED by this change — `series-metadata` (PR #171) and `multiple-dicot-plate-pipeline` (PR #165) stay as-is. The new accessor + validators on `Series` are additive (no observable contract change to existing `Series` methods).

### Affected code

- `sleap_roots/tracked_tip_pipeline.py` — NEW FILE (~150 lines). Contains `TrackedTipPipeline` class with `traits` list (DAG-A topology) and the two `compute_*` methods.
- `sleap_roots/series.py` — EXTEND. Add `Series.get_tracked_tips` method (~50 lines including auto-detection + per-instance iteration), `validate_tracked_slp`, `validate_series_for_tracked_tip` (~30 lines combined).
- `sleap_roots/__init__.py` — re-exports `TrackedTipPipeline` (top-level), `validate_tracked_slp`, `validate_series_for_tracked_tip`.
- `tests/test_tracked_tip_pipeline.py` — NEW FILE. Synthetic-`.slp` integration tests for the pipeline + DAG composition tests.
- `tests/test_series.py` — EXTEND. New tests for `get_tracked_tips` (auto-detect, ordering, untracked-instance error, single-node skeleton, multi-node skeleton) + the two validators.
- `tests/data/circumnutation_plate/` — NEW directory with fixture .slp + fixture_metadata.csv + README.md (committed via Git LFS).
- No existing code is modified — additive throughout.

### Source design doc

[`docs/superpowers/specs/2026-04-23-timelapse-diff-and-tip-kinematics-design.md`](docs/superpowers/specs/2026-04-23-timelapse-diff-and-tip-kinematics-design.md) § Workstream 2.

### Dependencies

- **Depends on**: PR #171 (Workstream 1 metadata layer — `Series.sample_uid`, `Series.timepoint`, `Series.get_metadata`). Already merged.
- **Depends on**: PR #165 (`MultipleDicotPlatePipeline`) — output-contract pattern this pipeline mirrors.
- **Unblocks**: #170 (`TimeDiffPipeline`) consuming tracked-tip per-track-over-time output. The plate-intra-series and tracked-tip-intra-series cases of #170 land cleanly once this pipeline ships.
- **Substrate for**: future circumnutation pipeline (literature-driven design once assembled), future growth-rate / gravitropism pipelines.

## Known limitations / follow-ups

These are **explicit deferrals** filed as separate issues, NOT introduced by this PR:

- **#186** — Per-frame metadata accessor on `Series` (`get_per_frame_metadata`, companion CSV). Required before any pipeline emits real-time-per-frame columns. This PR uses integer `frame` only.
- **#187** — Preprocessing helpers (image folder → `.h5` + per-frame metadata CSV). Captures the upstream data-prep workflow.
- **#188** — Generic source-META → sleap-roots-CSV converter. Captures the experiment-metadata side of data-prep. The fixture CSV in this PR is synthesized by hand pending this helper.
- **#189** — Megafile split: refactor `trait_pipelines.py` into per-pipeline modules. This PR's `tracked_tip_pipeline.py` starts the pattern.
- **`get_root_lengths` NaN-on-empty-segments behavior** — single-frame `tip_trajectory_length=NaN` is the codebase's existing convention. A future refactor could fix this (return 0.0) but would touch existing pipelines. Out of scope for #129. Documented in the spec.
- **Pipeline class location follows new convention** — `sleap_roots/tracked_tip_pipeline.py`, NOT `sleap_roots/trait_pipelines.py`. Tests use both top-level (`from sleap_roots import TrackedTipPipeline`) and module-level imports.
- **Track-order non-determinism across frames** — verified during brainstorm: frame 0 instances are `[track_0, 1, 2, 3, 4, 5]` but frame 1 is `[track_0, 3, 4, 2, 1, 5]`. `Series.get_tracked_tips` MUST iterate per-instance and read `inst.track.name` per instance — never rely on positional ordering. Codified as a regression test.
- **Single-node skeleton support** — verified on the circumnutation fixture (`['r0']`). The `[-1]` slice convention from `tips.get_tips` works because `pts[:, -1, :]` of a `(n_inst, 1, 2)` array degenerates to `(n_inst, 0, :)` which IS the only point. Codified as a regression test using a multi-node-skeleton synthetic .slp alongside the single-node real fixture.
