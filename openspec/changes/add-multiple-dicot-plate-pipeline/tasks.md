# Tasks: Add `MultipleDicotPlatePipeline`

**Source of truth for architecture**: `docs/superpowers/specs/2026-04-16-multiple-dicot-plate-pipeline-design.md` — this document describes the TraitDef DAG, per-frame/per-plant compute flow, output structures, SLEAP instance index mapping mechanism, and per-plant trait set. Do NOT duplicate that content into task descriptions; reference it instead.

**Workflow guarantees**:

- Strict TDD: every subsection writes failing tests BEFORE the implementation that makes them pass. Each test is given a name and exactly-what-it-verifies up front.
- All commands use `uv run` (`uv run pytest`, `uv run black`, `uv run pydocstyle`) — never bare tool names.
- No implementation is merged until every follow-up issue in section 5 is filed and linked in the PR body.

## 1. Write failing unit tests for `argsort_primaries_by_base_x`

- [ ] 1.1 Add `test_argsort_primaries_by_base_x_basic` to `tests/test_points.py`. Verifies three primaries at base x=[100, 50, 200] return index order `[1, 0, 2]`.
- [ ] 1.2 Add `test_argsort_primaries_by_base_x_single_plant` to `tests/test_points.py`. Verifies a one-entry dict returns a one-element list with that key.
- [ ] 1.3 Add `test_argsort_primaries_by_base_x_empty` to `tests/test_points.py`. Verifies empty dict input returns `[]` without raising.
- [ ] 1.4 Add `test_argsort_primaries_by_base_x_identical_x` to `tests/test_points.py`. Verifies identical x-values preserve insertion order (stable sort tiebreak).
- [ ] 1.5 Run `uv run pytest tests/test_points.py::test_argsort_primaries_by_base_x_basic -x` and confirm it FAILS with an `AttributeError` / `ImportError` (function does not yet exist).

## 2. Implement `argsort_primaries_by_base_x`

- [ ] 2.1 Add `argsort_primaries_by_base_x(plant_associations_dict: dict) -> List[int]` to `sleap_roots/points.py`. Google-style docstring. Stable sort via `np.argsort(..., kind="stable")`. Empty input → empty list.
- [ ] 2.2 Run `uv run pytest tests/test_points.py -k argsort_primaries_by_base_x -v` — all four tests pass.
- [ ] 2.3 Run `uv run pytest tests/test_points.py -x` — full file passes (no regressions).

## 3. Write failing tests for `MultipleDicotPlatePipeline` (TDD)

**Setup helper**: before writing pipeline tests, add a test helper (or pytest fixture) at the top of `tests/test_trait_pipelines.py` that builds a synthetic `.slp` file on disk using `sio.save_slp` + `sio.Labels` (same idiom as `tests/test_pixel_units.py`). The helper takes per-frame primary and lateral point arrays and returns a `Series` via `Series.load`. This is the mechanism every integration test in this section uses — **do NOT bypass it by constructing `initial_frame_traits` directly, because the `compute_plate_traits(series)` frame loop must be exercised end-to-end through the Series interface.**

### 3a. Unit tests (targeting the TraitDef DAG and `get_initial_frame_traits`)

- [ ] 3a.1 Add `test_multiple_dicot_plate_pipeline_define_traits` — instantiate the pipeline and assert `[t.name for t in pipeline.define_traits()]` contains (in order) `primary_pts_no_nans`, `lateral_pts_no_nans`, `detected_count`, `plant_associations_dict`, `plant_id_order`. Additionally assert:
  - `detected_count` is `scalar=True, include_in_csv=True`
  - No TraitDef uses `filter_plants_with_unexpected_ct` (plates skip count-filtering per design D2)
  - `plant_id_order` uses `argsort_primaries_by_base_x` with input `["plant_associations_dict"]`
- [ ] 3a.2 Add `test_multiple_dicot_plate_pipeline_get_initial_frame_traits_keys` — synthetic Series, assert returned dict has keys `{primary_pts, lateral_pts, primary_sleap_idxs, lateral_sleap_idxs, expected_count}`.
- [ ] 3a.3 Add `test_multiple_dicot_plate_pipeline_get_initial_frame_traits_sleap_idxs` — synthetic primaries of shape `(4, 6, 2)` with instance 1 all-NaN; assert `primary_sleap_idxs == [0, 2, 3]` (skips NaN primary by original SLEAP index, NOT post-filter index).

### 3b. Integration tests (round-trip through synthetic `.slp` via `Series.load`)

- [ ] 3b.1 Add `test_multiple_dicot_plate_pipeline_basic`. Build a synthetic `.slp` with 3 primaries at known base-x `[200, 50, 100]` plus laterals. Call `compute_plate_traits(series)`. Assert `plants` length == 3, `plant_id` order is left-to-right (primary at x=50 has `plant_id=0`, x=100 has `plant_id=1`, x=200 has `plant_id=2`), `primary_sleap_idx` values map back to original SLEAP indices (1, 2, 0 for the left-to-right order), every plant row has `detected_count == 3`, every `traits` dict contains DicotPipeline trait names **unchanged** (`primary_length`, `lateral_count`, `network_length`, `primary_base_tip_dist` all present — NO `primary_root_length`-style renames).
- [ ] 3b.2 Add `test_multiple_dicot_plate_pipeline_sleap_idx_traceability`. Build a synthetic `.slp` with 3 primaries where SLEAP index 1 is all-NaN. Call `compute_plate_traits(series)`. Assert `len(plants) == 2`, `{p["primary_sleap_idx"] for p in plants} == {0, 2}`, NOT `{0, 1}`.
- [ ] 3b.3 Add `test_multiple_dicot_plate_pipeline_expected_count_none`. Build a synthetic `.slp` and load the Series WITHOUT a `csv_path` so `series.expected_count` is `np.nan`. Call `compute_plate_traits(series)`. Assert no exception; assert every `plants[i]["expected_count"]` is `np.nan` (or `None` after JSON round-trip); assert `detected_count` is a non-negative int.
- [ ] 3b.4 Add `test_multiple_dicot_plate_pipeline_expected_count_mismatch`. Build a synthetic `.slp` with 3 detected primaries. Load the Series with a CSV specifying `expected_count=2`. Call `compute_plate_traits(series)`. Assert `len(plants) == 3` (no dropping), every plant row has `expected_count == 2` and `detected_count == 3`.
- [ ] 3b.5 Add `test_multiple_dicot_plate_pipeline_empty_frame`. Build a synthetic `.slp` where every primary instance in frame 0 is all-NaN. Call `compute_plate_traits(series)`. Assert `plants == []`, no exception.
- [ ] 3b.6 Add `test_multiple_dicot_plate_pipeline_timelapse_shape`. Build a synthetic `.slp` with 2 frames, each containing the same 3 primaries. Call `compute_plate_traits(series)`. Assert `len(plants) == 6` (3 × 2). Assert exactly 3 rows have `frame == 0` and 3 rows have `frame == 1`.
- [ ] 3b.7 Add `test_multiple_dicot_plate_pipeline_csv_output`. Build a synthetic `.slp` with 2 plants. Call `compute_plate_traits(series, write_csv=True, output_dir=tmp_path)`. Read the CSV with `pd.read_csv`. Assert `list(df.columns)[0:6] == ["series", "frame", "plant_id", "primary_sleap_idx", "expected_count", "detected_count"]`. Assert `list(df.columns)[6:] == DicotPipeline().csv_traits`. Assert no column name contains `_root_` as an infix. Assert `"lateral_sleap_idxs"` is NOT a column.
- [ ] 3b.8 Add `test_multiple_dicot_plate_pipeline_json_output`. Call `compute_plate_traits(series, write_json=True, output_dir=tmp_path)`. Read back via `json.load`. Assert top-level keys `{series, group, qc_fail, expected_count, plants}`. Assert each plant entry has `primary_points` as a list of `[x, y]` pairs, `lateral_points` as a list of lists of `[x, y]` pairs, `primary_sleap_idx` as int, `lateral_sleap_idxs` as list of ints. Assert `traits` dict keys include DicotPipeline trait names unchanged.
- [ ] 3b.9 Add `test_compute_batch_plate_traits`. Build two synthetic `.slp` Series (2 plants × 1 frame, 3 plants × 1 frame). Call `compute_batch_plate_traits([seriesA, seriesB])`. Assert returned DataFrame has 5 rows, `series` column matches expectations for rows 0-1 vs 2-4. Call again with `write_json=True` and assert the resulting JSON file is a list of 2 per-series dicts.
- [ ] 3b.10 Run `uv run pytest tests/test_trait_pipelines.py -k multiple_dicot_plate_pipeline -v` — confirm all 11 tests (3a + 3b) FAIL with `AttributeError` / `ImportError` (class does not yet exist).

## 4. Implement `MultipleDicotPlatePipeline`

Implement in `sleap_roots/trait_pipelines.py`. Keep the implementation minimal — only what the tests in section 3 require.

- [ ] 4.1 Add `MultipleDicotPlatePipeline` class with `@attrs.define` decorator. `define_traits()` returns the 5-element TraitDef DAG described in section 3a.1. Google-style docstring at class level noting: (a) plates skip count-filter (D2), (b) `plant_id` is a left-to-right ordering paired with original SLEAP indices (D3), (c) per-plant traits reuse DicotPipeline trait names unchanged (D6).
- [ ] 4.2 Implement `get_initial_frame_traits(plant, frame_idx)` to return the 5-key dict. Compute `primary_sleap_idxs` / `lateral_sleap_idxs` via the validity-mask-before-filter pattern: `primary_sleap_idxs = [i for i, r in enumerate(primary_pts_raw) if not np.isnan(r).any()]` (and likewise for lateral). See design doc § "SLEAP instance index mapping".
- [ ] 4.3 Implement `compute_plate_traits(series, write_csv=False, write_json=False, output_dir=".", csv_suffix=".plate_traits.csv", json_suffix=".plate_traits.json")`. Frame loop follows design doc § "Compute flow (per series)". For each plant, instantiate a nested `DicotPipeline()`, call `compute_frame_traits({"primary_pts": assoc["primary_points"][None, ...], "lateral_pts": assoc["lateral_points"]})`, store the full returned dict as `traits` in the plant row. Map post-filter lateral indices back to original SLEAP indices by matching each entry in `assoc["lateral_points"]` against `lateral_pts_no_nans` via `is_line_valid` + `np.array_equal`, then looking up `initial["lateral_sleap_idxs"][match_idx]`.
- [ ] 4.4 Implement JSON serialization via `NumpyArrayEncoder` at `sleap_roots/trait_pipelines.py:123`. JSON structure matches section 3b.8.
- [ ] 4.5 Implement CSV flattening: build per-plant rows where each row contains the 6 metadata columns + each DicotPipeline csv-trait (scalar as-is; non-scalar via `get_summary` → `{name}_{min,max,...}` suffixes from `sleap_roots/summary.py`). Use `pd.DataFrame(rows).to_csv(path, index=False)`.
- [ ] 4.6 Implement `compute_batch_plate_traits(all_series, write_csv=False, write_json=False, output_dir=".", csv_name="plate_batch_traits.csv", json_name="plate_batch_traits.json")`. Loop over series, concatenate per-series DataFrames, optionally write CSV/JSON.
- [ ] 4.7 Run `uv run pytest tests/test_trait_pipelines.py -k multiple_dicot_plate_pipeline -v` — all 11 tests pass.
- [ ] 4.8 Run `uv run pytest tests/test_points.py tests/test_trait_pipelines.py -x` — no regressions in existing points / pipeline tests.

## 5. File follow-up GitHub issues BEFORE the PR merges

Seven issues must be filed and verified before the PR is marked ready for review. Each `gh issue create` command below is followed by a `gh issue list --search` verification gate.

- [ ] 5.1 File issue **PR 2** — "Add tertiary root support to `MultipleDicotPlatePipeline`". Body references this PR and design doc section on tertiary→primary direct association (D4).
  ```
  gh issue create --title "Add tertiary root support to MultipleDicotPlatePipeline (PR 2 of #126)" \
    --body "Tracks PR 2 of #126. See docs/superpowers/specs/2026-04-16-multiple-dicot-plate-pipeline-design.md § D4. Scope: Series.get_tertiary_points, tertiary_path attribute, reuse associate_lateral_to_primary with tertiary input, emit tertiary columns."
  ```
- [ ] 5.2 File issue **PR 3** — "Add configurable filtering thresholds to `MultipleDicotPlatePipeline`".
  ```
  gh issue create --title "Add configurable filtering thresholds to MultipleDicotPlatePipeline (PR 3 of #126)" \
    --body "Tracks PR 3 of #126. Add min_primary_length_px, min_lateral_length_px, node_score_threshold, primary_angle_filter as constructor kwargs with plate-specific defaults."
  ```
- [ ] 5.3 File issue **A** — "Standardize multi-plant pipeline JSON output + deterministic plant_id".
  ```
  gh issue create --title "Standardize multi-plant pipeline JSON output (follow-up A from #126)" \
    --body "Apply plate JSON format (raw points, SLEAP instance indices, left-to-right sorted plant_id) to MultipleDicotPipeline and MultiplePrimaryRootPipeline. Spawned by #126."
  ```
- [ ] 5.4 File issue **B** — "Include raw points in single-plant pipeline JSON".
  ```
  gh issue create --title "Include raw points in single-plant pipeline JSON (follow-up B from #126)" \
    --body "Apply to DicotPipeline, YoungerMonocotPipeline, OlderMonocotPipeline for self-contained analysis artifacts. Spawned by #126."
  ```
- [ ] 5.5 File issue **C** — "Plate visualization / viewer".
  ```
  gh issue create --title "Plate visualization / viewer for MultipleDicotPlatePipeline JSON (follow-up C from #126)" \
    --body "Consume plate pipeline JSON to render colored per-plant overlays. Related to #128. Spawned by #126."
  ```
- [ ] 5.6 File issue **D** — "Real plate `.slp` fixture tests".
  ```
  gh issue create --title "Real plate .slp fixture tests for MultipleDicotPlatePipeline (follow-up D from #126)" \
    --body "Add MK22 dataset integration tests once fixtures are available. Synthetic-only tests land in PR 1 of #126."
  ```
- [ ] 5.7 File issue **E** — "Generalize `qc_fail` for plates".
  ```
  gh issue create --title "Generalize Series.qc_fail for plate data (follow-up E from #126)" \
    --body "Series.qc_fail currently reads CSV column qc_cylinder at sleap_roots/series.py:196-208. Plates need either a qc_plate column or a unified qc column. PR 1 of #126 emits qc_fail unchanged; document the cylinder-specific interpretation in the meantime."
  ```

### Verification gate

- [ ] 5.8 Run the verification command and confirm all 7 issue numbers come back:
  ```
  gh issue list --search "#126 in:body follow-up" --state open --limit 20 --json number,title
  ```
  If any of the 7 titles is missing, re-file before proceeding.
- [ ] 5.9 Copy the 7 issue numbers into the PR body (section "Follow-up issues filed during this PR").

## 6. Pre-merge validation

- [ ] 6.1 Run `openspec validate add-multiple-dicot-plate-pipeline --strict` and resolve any issues.
- [ ] 6.2 Run `uv run black --check sleap_roots/ tests/` — formatting clean.
- [ ] 6.3 Run `uv run pydocstyle sleap_roots/points.py sleap_roots/trait_pipelines.py` — docstrings clean for the new additions.
- [ ] 6.4 Run `uv run pytest tests/ -x` — full test suite passes.
- [ ] 6.5 Invoke `/review-pr` or equivalent self-review on the branch before opening the PR.
- [ ] 6.6 Open the PR; body MUST reference issue #126 and link the 7 follow-up issue numbers from 5.9.

## Dependencies

- Section 2 depends on section 1.
- Section 3 (tests-first) must be written before section 4 (implementation).
- Section 4 depends on section 2 (uses `argsort_primaries_by_base_x`).
- Section 5 can start any time but all 7 issues MUST exist before section 6.6 (PR ready for review).
- Section 6 depends on sections 2, 4, and 5.

## Notes

- **Source-of-truth reference**: `docs/superpowers/specs/2026-04-16-multiple-dicot-plate-pipeline-design.md` (committed on this branch). The commit amending it with trait-naming, SLEAP-index-mapping, and `qc_fail` corrections is `8b02698`.
- **Why no renaming of trait names**: the project-wide convention uses `primary_*`, `lateral_*`, `network_*`, `crown_*` prefixes; no `_root_` infix anywhere. Renaming to `primary_root_length` etc. would have created a parallel dialect that conflicts with `DicotPipeline.primary_length` at `sleap_roots/trait_pipelines.py:1425`. See design doc § D6 and § "Per-plant traits (PR 1)".
- **Why JSON includes raw points**: the plate JSON is intended as a self-contained analysis artifact (design D5). The plate viewer follow-up (issue C) will consume this JSON directly — no dependency on the source `.slp` at visualization time. `~30 KB` overhead per plate is acceptable.
- **Why synthetic `.slp` round-trip (not in-memory mocks)**: `Series.get_primary_points` / `.get_lateral_points` have non-trivial behavior around user_instances + unused_predictions stacking and NaN placeholder injection. Integration tests MUST go through `Series.load` to exercise that code path. The `tests/test_pixel_units.py` module demonstrates the synthetic-`.slp` idiom via `sio.save_slp`.
