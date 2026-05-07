# Tasks: Add `TrackedTipPipeline` (Workstream 2 of 2026-04-23 design, issue #129)

**Source of truth for architecture**: `docs/superpowers/specs/2026-04-23-timelapse-diff-and-tip-kinematics-design.md` § Workstream 2.

**Workflow guarantees**:

- Strict TDD: each subsection writes failing tests BEFORE implementation. Test-helper additions (e.g. extending the synthetic-`.slp` builder) land atomically with the tests that need them.
- All commands use `uv run` (`uv run pytest`, `uv run black`, `uv run pydocstyle`).
- No implementation merged until every follow-up issue (#186 / #187 / #188 / #189) is linked in the PR body.
- Substrate scope is FROZEN — no velocity / curvature / circumnutation traits in this PR. They live in separate downstream pipeline classes (filed as future work).

## 1. Pre-flight: copy real fixture into `tests/data/` (Git LFS)

This lands FIRST because the integration tests in §11 depend on it. No Python code yet — just commit the fixture files so `git lfs ls-files` shows them tracked.

- [x] 1.1 Verify Git LFS hooks installed: `git lfs ls-files | head -3` shows existing `.slp` / `.h5` files in `tests/data/`. If not, install with `git lfs install`.
- [x] 1.2 Confirm `*.slp` is already tracked by LFS: `git check-attr -a tests/data/multiple_arabidopsis_11do/6039_1.primary.predictions.slp` shows `filter: lfs`. If not, fix `.gitattributes` first.
- [x] 1.3 Create `tests/data/circumnutation_plate/` directory.
- [x] 1.4 Copy `Z:\users\eberrigan\circumnutation\20250819_Suyash_Patil_CMTN_Kitx_vs_Hk1-3_07-30-25\run_20250827_091833\plate_001_greyscale.tracked.slp` → `tests/data/circumnutation_plate/plate_001_greyscale.tracked.slp` (184 KB).
- [x] 1.5 Synthesize `tests/data/circumnutation_plate/fixture_metadata.csv` with one row:
  ```
  plant_qr_code,genotype,treatment,number_of_plants_cylinder,timepoint
  plate_001,KitaakeX,MOCK,6,0
  ```
- [x] 1.6 Write `tests/data/circumnutation_plate/README.md` per the standard template (issue #168). Required sections:
  - Purpose (input fixture for `TrackedTipPipeline` integration tests)
  - Imaging geometry (plate, 6 plants, ~5h imaging at ~10-min intervals — derived from filename datetime stamps)
  - Acquisition context (Suyash Patil, KitaakeX vs hk1-3, MOCK treatment, 2025-07-30)
  - Contents (the .slp file, the synthesized CSV — explicit list)
  - Conversion provenance (which source file mapped to which fixture file; what columns were renamed/dropped)
  - Known limitations (per-frame CSV NOT shipped — deferred to #186; column-name caveat — `plant_qr_code` value `"plate_001"` is a plate identifier, NOT a plant identifier, legacy convention; #163 will rename)
  - Related issues (#129, #163, #168, #186, #187, #188)
- [x] 1.7 `git add tests/data/circumnutation_plate/` and verify `git status` shows the .slp as `LFS`. Commit: `git commit -m "tests: add circumnutation_plate fixture for TrackedTipPipeline (#129)"`. DO NOT push until the PR is opened.

## 2. Write failing tests for `Series.get_tracked_tips` (TDD red phase 1)

These tests come FIRST — `get_tracked_tips` is the substrate accessor on which the pipeline + every downstream consumer depends.

- [x] 2.1 Locate or extend a synthetic-`.slp`-builder helper in `tests/conftest.py` or `tests/test_series.py` that constructs a tracked `.slp` via `sio.save_slp(...)` with `sio.Track` objects attached to instances via `inst.track = sio.Track(name="track_N")`. **Critical**: this helper accepts `n_frames`, `n_tracks`, `track_positions` (per-frame, per-track xy), `skeleton_node_names` (default `["r0"]` for single-node, configurable for multi-node), and `instance_order` (per-frame list of track names — supports the regression test for non-deterministic positional ordering across frames).
- [x] 2.2 Add `test_get_tracked_tips_returns_long_dataframe` — synthetic 3-frame .slp with 2 tracks → DataFrame with columns `["track_id", "frame", "tip_x", "tip_y"]`, 6 rows total.
- [x] 2.3 Add `test_get_tracked_tips_sorted_by_track_then_frame` — synthetic 3-frame .slp where instance positional order is RANDOMIZED per frame (frame 0: `[track_a, track_b]`; frame 1: `[track_b, track_a]`) → output DataFrame's `(track_id, frame)` column is monotonically sorted within each track group. **This is the regression test for the brainstorm-verified track-order non-determinism.**
- [x] 2.4 Add `test_get_tracked_tips_auto_detects_root_type_from_populated_path` — `Series.load(primary_path=...)` (lateral/crown empty) → `series.get_tracked_tips()` (no kwarg) succeeds and reads from primary.
- [x] 2.5 Add `test_get_tracked_tips_raises_when_multiple_paths_populated_no_root_type` — `Series.load(primary_path=..., lateral_path=...)` → `series.get_tracked_tips()` raises `ValueError` mentioning `root_type`.
- [x] 2.6 Add `test_get_tracked_tips_raises_when_zero_paths_populated` — `Series.load()` (none populated) → `ValueError` mentioning `root_type` and the missing path kwargs.
- [x] 2.7 Add `test_get_tracked_tips_raises_on_untracked_instance` — synthetic .slp where one instance has `inst.track = None` → `ValueError` mentioning the offending frame index AND `sleap.ai/tracking`.
- [x] 2.8 Add `test_get_tracked_tips_raises_on_empty_track_name` — synthetic .slp where `inst.track = sio.Track(name="")` → same `ValueError` as 2.7 (empty-name treated as untracked).
- [x] 2.9 Add `test_get_tracked_tips_single_node_skeleton` — synthetic .slp with skeleton `['r0']` (one node) → output `tip_x`, `tip_y` are the node's coordinates (`pts[-1]` is `pts[0]`).
- [x] 2.10 Add `test_get_tracked_tips_multi_node_skeleton` — synthetic .slp with skeleton `['base', 'mid', 'tip']` → output `tip_x`, `tip_y` are the LAST node's coordinates (the convention).
- [x] 2.11 Run `uv run pytest tests/test_series.py -k get_tracked_tips -x` — all 9 FAIL with `AttributeError: 'Series' object has no attribute 'get_tracked_tips'`.

## 3. Implement `Series.get_tracked_tips` (TDD green phase 1)

- [x] 3.1 Add `get_tracked_tips(self, root_type: Optional[Literal["primary", "lateral", "crown"]] = None) -> pd.DataFrame` method to `Series` class in `sleap_roots/series.py`. Implementation:
  - Resolve `root_type` (auto-detect when `None`); raise `ValueError` on zero / >1 populated paths.
  - Get the labels object for the resolved root type (`self.primary_labels` / `self.lateral_labels` / `self.crown_labels`).
  - Iterate `for frame_idx in range(len(labels)):` then `for inst in labels[frame_idx].instances:`. Note: existing accessors use `lf.user_instances + lf.unused_predictions` — confirm whether this matches what we want for tracked predictions (see open question in §3.4).
  - For each instance: assert `inst.track is not None and inst.track.name`; raise `ValueError` listing offending frame index + `sleap.ai/tracking` URL otherwise.
  - Append row dict `{"track_id": inst.track.name, "frame": frame_idx, "tip_x": inst.numpy()[-1, 0], "tip_y": inst.numpy()[-1, 1]}`.
  - Return `pd.DataFrame(rows).sort_values(["track_id", "frame"]).reset_index(drop=True)`.
- [x] 3.2 Update `Series` class docstring to mention `get_tracked_tips`. Add Google-style docstring to the new method covering Args / Returns / Raises.
- [x] 3.3 Verify via `uv run pytest tests/test_series.py -k get_tracked_tips -v` — all 9 pass.
- [x] 3.4 **Open question for code review**: which instance set to iterate — `lf.instances` (all), `lf.predicted_instances`, or `lf.user_instances + lf.unused_predictions` (existing convention)? Verify on the real fixture: `uv run --no-project python -c "import sleap_io as sio; lbls = sio.load_slp(r'tests/data/circumnutation_plate/plate_001_greyscale.tracked.slp'); print(len(lbls.labeled_frames[0].instances), len(lbls.labeled_frames[0].predicted_instances), len(lbls.labeled_frames[0].user_instances))"`. Document the choice in the docstring.
- [x] 3.5 Run `uv run pytest tests/ -x` — full suite green (no regressions on existing accessors).

## 4. Write failing tests for `validate_tracked_slp` + `validate_series_for_tracked_tip` (TDD red phase 2)

- [x] 4.1 Add `test_validate_tracked_slp_passes_on_fully_tracked` — synthetic .slp where every instance has `inst.track is not None` → `validate_tracked_slp(path)` returns `None`, no raise.
- [x] 4.2 Add `test_validate_tracked_slp_raises_on_untracked_instance` — synthetic .slp with one untracked instance → `ValueError` listing the offending frame index AND `sleap.ai/tracking`.
- [x] 4.3 Add `test_validate_tracked_slp_lists_all_offending_frames` — synthetic .slp with 3 untracked instances across 3 different frames → error message lists all 3 frame indices.
- [x] 4.4 Add `test_validate_series_for_tracked_tip_resolves_root_type` — `Series.load(primary_path=tracked_slp)` → `validate_series_for_tracked_tip(series)` (no `root_type`) succeeds.
- [x] 4.5 Add `test_validate_series_for_tracked_tip_explicit_root_type` — `Series.load(primary_path=tracked_slp, lateral_path=other_tracked)` → `validate_series_for_tracked_tip(series, root_type="primary")` succeeds; `root_type="lateral"` validates the other path.
- [x] 4.6 Add `test_validate_series_for_tracked_tip_raises_on_zero_or_multiple_paths_no_root_type` — same pattern as `get_tracked_tips` errors.
- [x] 4.7 Run `uv run pytest tests/test_series.py -k validate -x` — all 6 FAIL with `ImportError` (functions don't exist yet).

## 5. Implement `validate_tracked_slp` + `validate_series_for_tracked_tip` (TDD green phase 2)

- [x] 5.1 Add `validate_tracked_slp(slp_path: Union[str, Path]) -> None` as a module-level function in `sleap_roots/series.py` (NOT a method on `Series` — it operates on a path string before any `Series` is constructed). Implementation: open via `sio.load_slp`, walk frames + instances, collect frames with any untracked instance, raise `ValueError` if any.
- [x] 5.2 Add `validate_series_for_tracked_tip(series: Series, root_type: Optional[str] = None) -> None` as a module-level function. Implementation: resolve root_type, check the relevant `<root_type>_path` attribute is set, assert skeleton has ≥1 node, call `validate_tracked_slp` on the resolved path.
- [x] 5.3 Re-export both from `sleap_roots/__init__.py`.
- [x] 5.4 Run `uv run pytest tests/test_series.py -k validate -v` — all 6 pass.
- [x] 5.5 Run `uv run pytest tests/ -x` — full suite green.

## 6. Write failing tests for `TrackedTipPipeline` DAG composition (TDD red phase 3)

These tests exercise the pipeline class without writing files — pure compute method, in-memory result dict.

- [x] 6.1 Create `tests/test_tracked_tip_pipeline.py`. Add a synthetic-`.slp`-builder helper or import from conftest.
- [x] 6.2 Add `test_compute_tracked_tip_traits_returns_expected_dict_keys` — synthetic 5-frame .slp with 2 tracks → result dict has top-level keys `{schema_version, pipeline, units, series, sample_uid, timepoint, tracks, trajectories}`.
- [x] 6.3 Add `test_compute_tracked_tip_traits_schema_version_is_1` — `result["schema_version"] == 1`.
- [x] 6.4 Add `test_compute_tracked_tip_traits_units_dict` — `result["units"] == {"lengths": "pixels", "ratios": "dimensionless", "counts": "dimensionless", "time": "unspecified"}`.
- [x] 6.5 Add `test_compute_tracked_tip_traits_tracks_table_one_row_per_track_id` — synthetic 2-track .slp → `len(result["tracks"]) == 2`. Each row has keys `{track_id, n_frames_tracked, n_frames_total, tracking_coverage, tip_trajectory_length, tip_displacement_net}`.
- [x] 6.6 Add `test_compute_tracked_tip_traits_trajectories_table_one_row_per_track_frame` — synthetic 5-frame × 2-track .slp where every instance is tracked → `len(result["trajectories"]) == 10`.
- [x] 6.7 Add `test_tip_trajectory_length_straight_line` — synthetic .slp where one track moves in a straight line of known length (e.g. (0,0) → (3,0) → (6,0) → (10,0)) → `tip_trajectory_length == 10.0`.
- [x] 6.8 Add `test_tip_displacement_net_straight_line` — same straight-line track → `tip_displacement_net == 10.0`.
- [x] 6.9 Add `test_tip_displacement_net_round_trip_returns_to_origin` — track (0,0) → (5,0) → (0,0) → `tip_displacement_net == 0.0`, `tip_trajectory_length == 10.0`. **Validates the geometric distinction.**
- [x] 6.10 Add `test_tip_displacement_net_right_angle_path` — track (0,0) → (3,0) → (3,4) → `tip_trajectory_length == 7.0`, `tip_displacement_net == 5.0` (3-4-5 triangle).
- [x] 6.11 Add `test_tracking_coverage_full` — every frame tracked → `tracking_coverage == 1.0`.
- [x] 6.12 Add `test_tracking_coverage_partial` — track present in 3 of 5 frames → `tracking_coverage == 0.6`.
- [x] 6.13 Add `test_compute_tracked_tip_traits_single_frame_track_zero_displacement_nan_length` — track present in 1 frame only → `tip_displacement_net == 0.0`, `np.isnan(tip_trajectory_length)`, `n_frames_tracked == 1`. **Codifies the documented asymmetry.**
- [x] 6.14 Add `test_compute_tracked_tip_traits_zero_tracks` — synthetic .slp with NO tracked instances anywhere → `len(result["tracks"]) == 0`, `len(result["trajectories"]) == 0`, no crash.
- [x] 6.15 Add `test_compute_tracked_tip_traits_emits_sample_uid_and_timepoint_from_csv` — `Series.load(primary_path=..., csv_path=metadata_csv, sample_uid="plate_X")` where the CSV has a row with `plant_qr_code="plate_X", timepoint=2.5` → `result["sample_uid"] == "plate_X"`, `result["timepoint"] == 2.5`.
- [x] 6.16 Add `test_compute_tracked_tip_traits_no_csv_defaults_sample_uid_to_series_name` — no CSV → `result["sample_uid"] == result["series"]`, `np.isnan(result["timepoint"])`.
- [x] 6.17 Add `test_track_id_values_match_inst_track_name` — synthetic .slp with custom track names `["alpha", "beta"]` → `set(row["track_id"] for row in result["tracks"]) == {"alpha", "beta"}`.
- [x] 6.18 Run `uv run pytest tests/test_tracked_tip_pipeline.py -x` — all FAIL with `ImportError`.

## 7. Implement `TrackedTipPipeline` class (TDD green phase 3)

- [x] 7.1 Create `sleap_roots/tracked_tip_pipeline.py`. Imports: `from sleap_roots.trait_pipelines import Pipeline, TraitDef, _json_sanitize, NumpyArrayEncoder`; `from sleap_roots.lengths import get_root_lengths`; `from sleap_roots.bases import get_base_tip_dist`.
- [x] 7.2 Define `_TRACKED_TIP_UNITS = {"lengths": "pixels", "ratios": "dimensionless", "counts": "dimensionless", "time": "unspecified"}`.
- [x] 7.3 Define class `TrackedTipPipeline(Pipeline)` with `traits` list per the design doc DAG-A topology:
  - `track_xy` (input)
  - `n_frames_tracked`, `n_frames_total` (inputs)
  - `track_first_xy = lambda xy: xy[0]` (`input_traits=["track_xy"]`)
  - `track_last_xy = lambda xy: xy[-1]` (`input_traits=["track_xy"]`)
  - `tip_displacement_net = get_base_tip_dist` (`input_traits=["track_first_xy", "track_last_xy"]`)
  - `tip_trajectory_length = get_root_lengths` (`input_traits=["track_xy"]`)
  - `tracking_coverage = lambda nt, ntot: nt / ntot if ntot else np.nan` (`input_traits=["n_frames_tracked", "n_frames_total"]`)
- [x] 7.4 Implement `compute_tracked_tip_traits(self, series, *, write_csv=False, write_json=False, output_dir=".", emit_trajectories=True, csv_summary_suffix=".tracked_tip_traits.csv", csv_trajectory_suffix=".tracked_tip_trajectories.csv", json_suffix=".tracked_tip_traits.json")`. Algorithm:
  1. Validate via `validate_series_for_tracked_tip(series)` — raises early on bad inputs.
  2. Get `df = series.get_tracked_tips()` (already frame-sorted).
  3. Build `n_frames_total = len(series)`.
  4. Build `result` dict skeleton with top-level scalars (`schema_version=1`, `pipeline="TrackedTipPipeline"`, `units=_TRACKED_TIP_UNITS`, `series=series.series_name`, `sample_uid=series.sample_uid`, `timepoint=series.timepoint`, `tracks=[]`, `trajectories=[]`).
  5. For each `track_id, group in df.groupby("track_id")`:
     - `track_xy = group[["tip_x", "tip_y"]].values` (frame-sorted because the DataFrame is sorted).
     - `n_frames_tracked = len(group)`.
     - Run the TraitDef DAG with these inputs.
     - Append per-track summary row to `result["tracks"]`.
  6. If `emit_trajectories`: append every `(track_id, frame, tip_x, tip_y)` row to `result["trajectories"]` (already in frame-sorted order from the groupby).
  7. Write outputs per `write_csv` / `write_json` flags.
- [x] 7.5 Implement `_build_tracked_tip_dataframes(result, emit_trajectories) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]` — returns (summary_df, trajectory_df) for CSV write. Trajectory df is `None` when `emit_trajectories=False`.
- [x] 7.6 Add Google-style docstrings to class + every method.
- [x] 7.7 Re-export `TrackedTipPipeline` from `sleap_roots/__init__.py`.
- [x] 7.8 Run `uv run pytest tests/test_tracked_tip_pipeline.py -v` — all tests from §6 pass.
- [x] 7.9 Run `uv run pytest tests/ -x` — full suite green.

## 8. Write failing tests for CSV/JSON file emission (TDD red phase 4)

- [x] 8.1 Add `test_compute_tracked_tip_traits_writes_summary_csv` — `compute_tracked_tip_traits(series, write_csv=True, output_dir=tmp_path)` → `<series>.tracked_tip_traits.csv` exists, parsed columns are `["series", "sample_uid", "timepoint", "track_id", "n_frames_tracked", "n_frames_total", "tracking_coverage", "tip_trajectory_length", "tip_displacement_net"]`.
- [x] 8.2 Add `test_compute_tracked_tip_traits_writes_trajectory_csv` — same call → `<series>.tracked_tip_trajectories.csv` exists, parsed columns are `["series", "sample_uid", "timepoint", "track_id", "frame", "tip_x", "tip_y"]`.
- [x] 8.3 Add `test_compute_tracked_tip_traits_writes_json_with_both_tables` — `write_json=True` → `<series>.tracked_tip_traits.json` exists, parses to a dict with `tracks` and `trajectories` arrays at top level (NOT nested under another key).
- [x] 8.4 Add `test_compute_tracked_tip_traits_emit_trajectories_false_skips_trajectory_csv` — `emit_trajectories=False, write_csv=True` → summary CSV exists but trajectory CSV does NOT exist (`tmp_path` doesn't contain it).
- [x] 8.5 Add `test_compute_tracked_tip_traits_emit_trajectories_false_omits_trajectories_in_json` — same kwargs with `write_json=True` → JSON's top-level dict has `tracks` array but `trajectories` array is empty `[]` (or omitted; design choice — pick one and codify).
- [x] 8.6 Add `test_compute_tracked_tip_traits_csv_repeats_series_sample_uid_timepoint_per_row` — CSV has `series`, `sample_uid`, `timepoint` value on EVERY row (CSV/JSON asymmetry).
- [x] 8.7 Add `test_compute_tracked_tip_traits_json_top_level_scalars_not_in_per_row` — JSON's `tracks[i]` and `trajectories[i]` rows do NOT contain `series`, `sample_uid`, `timepoint` keys (those are top-level).
- [x] 8.8 Add `test_compute_tracked_tip_traits_json_nan_to_null` — synthetic single-frame track → JSON's `tracks[0]["tip_trajectory_length"]` is `null` (verifies `_json_sanitize` is wired).
- [x] 8.9 Run `uv run pytest tests/test_tracked_tip_pipeline.py -k write -x` — all FAIL.

## 9. Implement file emission (TDD green phase 4)

- [x] 9.1 Implement `_build_tracked_tip_dataframes` (see §7.5). Repeat top-level `series` / `sample_uid` / `timepoint` on every row in both DataFrames.
- [x] 9.2 Wire `write_csv` / `write_json` / `emit_trajectories` in `compute_tracked_tip_traits`. JSON sanitize via `_json_sanitize(result)` from `trait_pipelines.py`.
- [x] 9.3 Decide: `emit_trajectories=False` makes JSON's `trajectories` an empty list `[]` (consistent with `tracks` being empty for zero-tracks case) — codify in spec scenario from §8.5.
- [x] 9.4 Run `uv run pytest tests/test_tracked_tip_pipeline.py -k write -v` — all 8 pass.
- [x] 9.5 Run `uv run pytest tests/ -x` — full suite green.

## 10. Write failing tests for batch method (TDD red phase 5)

- [x] 10.1 Add `test_compute_batch_tracked_tip_traits_concatenates_summary` — 3 synthetic Series with 2 tracks each → batch summary CSV has 6 rows (2 × 3).
- [x] 10.2 Add `test_compute_batch_tracked_tip_traits_concatenates_trajectories` — 3 synthetic Series → batch trajectory CSV has Σ(per-series tracked instances) rows.
- [x] 10.3 Add `test_compute_batch_tracked_tip_traits_writes_json_list` — batch JSON parses to a `list` of per-series dicts (length 3), each shaped like `compute_tracked_tip_traits` output.
- [x] 10.4 Add `test_compute_batch_tracked_tip_traits_emit_trajectories_false` — batch with `emit_trajectories=False` → batch trajectory CSV does NOT exist; summary CSV does.
- [x] 10.5 Add `test_compute_batch_tracked_tip_traits_empty_input` — `compute_batch_tracked_tip_traits([])` → empty DataFrame, empty JSON list, no crash.
- [x] 10.6 Run `uv run pytest tests/test_tracked_tip_pipeline.py -k batch -x` — all 5 FAIL.

## 11. Implement batch method (TDD green phase 5)

- [x] 11.1 Implement `compute_batch_tracked_tip_traits(self, all_series, *, write_csv=False, write_json=False, output_dir=".", csv_summary_name="tracked_tip_batch_traits.csv", csv_trajectory_name="tracked_tip_batch_trajectories.csv", json_name="tracked_tip_batch_traits.json", emit_trajectories=True)`. Mirrors `compute_batch_plate_traits` at [trait_pipelines.py:3327](sleap_roots/trait_pipelines.py#L3327): walks all series, calls per-series method, collects per-series result dicts into a list, concatenates per-series summary DataFrames into one, ditto trajectory DataFrames.
- [x] 11.2 Run `uv run pytest tests/test_tracked_tip_pipeline.py -k batch -v` — all 5 pass.
- [x] 11.3 Run `uv run pytest tests/ -x` — full suite green.

## 12. Real-fixture integration tests (TDD red + green merged — fixture already committed in §1)

These tests use the real `tests/data/circumnutation_plate/` fixture. No new implementation needed at this point; if they fail, the failure is in the pipeline or the fixture (fix appropriately).

- [x] 12.1 Add `test_real_fixture_with_csv_metadata` — `Series.load(primary_path="tests/data/circumnutation_plate/plate_001_greyscale.tracked.slp", csv_path="tests/data/circumnutation_plate/fixture_metadata.csv", sample_uid="plate_001")` → `TrackedTipPipeline().compute_tracked_tip_traits(series)`. Assertions:
  - `result["sample_uid"] == "plate_001"`.
  - `result["timepoint"] == 0.0`.
  - `len(result["tracks"]) == 6`.
  - Every track has `0 <= tracking_coverage <= 1`.
  - `len(result["trajectories"]) == 1866` (311 frames × 6 tracked instances per frame, verified during brainstorm).
  - All output column names present (no missing).
- [x] 12.2 Add `test_real_fixture_no_csv_metadata` — `Series.load(primary_path=..., series_name="plate_001")` (NO `csv_path`). Same pipeline call. Assertions:
  - `result["sample_uid"] == "plate_001"` (defaults from `series_name` per Workstream 1).
  - `np.isnan(result["timepoint"])`.
  - `len(result["tracks"]) == 6`, `len(result["trajectories"]) == 1866`.
  - Every numeric trait value matches its corresponding row in `test_real_fixture_with_csv_metadata` (only the metadata changes, not the geometry).
- [x] 12.3 Add `test_real_fixture_track_id_values` — assert `set(row["track_id"] for row in result["tracks"]) == {"track_0", "track_1", "track_2", "track_3", "track_4", "track_5"}` (verified during brainstorm).
- [x] 12.4 Add `test_real_fixture_track_order_non_determinism_handled` — explicit assert on the brainstorm-verified observation: walk frames 0 and 1 of the real .slp directly via sleap-io, confirm positional order differs across the two frames, then assert that the pipeline output is correctly grouped by `track_id`. **Regression test for the per-instance iteration requirement.**
- [x] 12.5 Run `uv run pytest tests/test_tracked_tip_pipeline.py::test_real_fixture -v` — all 4 pass on Linux + Windows + macOS in CI.
- [x] 12.6 Run `uv run pytest tests/ -x` — full suite green.

## 13. Lint, format, docstrings

- [x] 13.1 `uv run black sleap_roots tests` — format all touched files.
- [x] 13.2 `uv run pydocstyle --convention=google sleap_roots/tracked_tip_pipeline.py sleap_roots/series.py` — Google docstrings on every public function / method / class.
- [x] 13.3 Verify imports are sorted (Black handles this; pydocstyle confirms).
- [ ] 13.4 Run `uv run pytest tests/ --cov=sleap_roots --cov-report=term-missing` — coverage maintained or increased; new files have ≥90% coverage.

## 14. Documentation

- [ ] 14.1 Add a tutorial section to `docs/` (existing `docs/tutorials/` directory) showing end-to-end usage. _Deferred to a follow-up PR — existing tutorials are full Jupyter notebooks (notebooks/*.ipynb) which is heavier than this substrate-focused PR warrants. Docstrings + the mermaid graph below provide canonical reference._
- [x] 14.2 Add CHANGELOG entry under `Added`: pipeline, accessor, validators. Under `Internal`: started per-pipeline-module pattern (#189 tracks the rest). _CHANGELOG is at `docs/changelog.md` (lowercase)._
- [x] 14.3 Generate `notebooks/TrackedTipPipeline_Mermaid_Graph.md` — DAG visualization mirroring the existing `DicotPipeline_Mermaid_Graph.md` and friends. Built directly from `TrackedTipPipeline().traits` via the same idiom in `notebooks/Pipeline_mermaid_diagrams.ipynb`. 8 distinct trait nodes.

## 15. OpenSpec validation + final checks

- [x] 15.1 Run `openspec validate add-tracked-tip-pipeline --strict` — clean (no errors).
- [x] 15.2 Run `openspec show add-tracked-tip-pipeline --json --deltas-only > /tmp/deltas.json` — verify the deltas describe ADDED requirements only (no MODIFIED or REMOVED).
- [ ] 15.3 `git diff --stat` — verify the size matches the proposal's `Affected code` estimate (no surprise files modified).

## 16. PR + review

- [ ] 16.1 Open PR against `main`. Title: `feat: TrackedTipPipeline for tracked-tip kinematic substrate (Workstream 2 of #129) (#129)`.
- [ ] 16.2 PR body links: #129 (closes), #170 (unblocks plate-intra-series and tracked-tip cases), #186 (deferred), #187 (deferred), #188 (deferred), #189 (per-pipeline-module pattern starts here).
- [ ] 16.3 Run `/review-pr` (Phase 3.5 self-review) before requesting human review.
- [ ] 16.4 Run `/pre-merge` after review approvals.
- [ ] 16.5 Verify CI green on Linux + Windows + macOS, Python 3.11.
- [ ] 16.6 Address any review feedback. New comments → new failing test → green test → push.

## 17. Post-merge: archive the OpenSpec change

- [ ] 17.1 After merge, run `openspec archive add-tracked-tip-pipeline` — moves the change folder under `openspec/changes/archive/YYYY-MM-DD-add-tracked-tip-pipeline/` and creates the `openspec/specs/tracked-tip-pipeline/spec.md`.
- [ ] 17.2 Commit and push the archive: `docs: archive add-tracked-tip-pipeline after PR #N merge`.
