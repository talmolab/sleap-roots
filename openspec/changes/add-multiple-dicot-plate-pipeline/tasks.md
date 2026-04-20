# Tasks: Add `MultipleDicotPlatePipeline`

**Source of truth for architecture**: [`docs/superpowers/specs/2026-04-16-multiple-dicot-plate-pipeline-design.md`](../../../docs/superpowers/specs/2026-04-16-multiple-dicot-plate-pipeline-design.md) on the `feature/multiple-dicot-plate-pipeline-126` branch — this document describes the TraitDef DAG, per-frame/per-plant compute flow, output structures, SLEAP instance index mapping mechanism, per-plant trait set, zero-laterals handling (D2), JSON-NaN-to-null emission (D5), count_mismatch/count_validated flags (D5b), and the `primary_root_depth` → `primary_base_tip_dist` substitution. Do NOT duplicate that content into task descriptions; reference it instead.

**Workflow guarantees**:

- Strict TDD: every subsection writes failing tests BEFORE the implementation that makes them pass. Each test is given a name and exactly-what-it-verifies up front.
- All commands use `uv run` (`uv run pytest`, `uv run black`, `uv run pydocstyle`) — never bare tool names.
- No implementation is merged until every follow-up issue in section 5 is filed and linked in the PR body.

## 1. Write failing unit tests for `argsort_primaries_by_base_x`

- [x] 1.1 Add `test_argsort_primaries_by_base_x_basic` to `tests/test_points.py`. Verifies three primaries at base x=[100, 50, 200] return index order `[1, 0, 2]`.
- [x] 1.2 Add `test_argsort_primaries_by_base_x_single_plant` to `tests/test_points.py`. Verifies a one-entry dict returns a one-element list with that key.
- [x] 1.3 Add `test_argsort_primaries_by_base_x_empty` to `tests/test_points.py`. Verifies empty dict input returns `[]` without raising.
- [x] 1.4 Add `test_argsort_primaries_by_base_x_identical_x` to `tests/test_points.py`. Verifies identical x-values preserve insertion order (stable sort tiebreak).
- [x] 1.5 Run `uv run pytest tests/test_points.py::test_argsort_primaries_by_base_x_basic -x` and confirm it FAILS with an `AttributeError` / `ImportError` (function does not yet exist). **Result**: collection error `ImportError: cannot import name 'argsort_primaries_by_base_x' from 'sleap_roots.points'` — red phase confirmed.

## 2. Implement `argsort_primaries_by_base_x`

- [x] 2.1 Add `argsort_primaries_by_base_x(plant_associations_dict: dict) -> List[int]` to `sleap_roots/points.py`. Google-style docstring. Stable sort via `np.argsort(..., kind="stable")`. Empty input → empty list.
- [x] 2.2 Run `uv run pytest tests/test_points.py -k argsort_primaries_by_base_x -v` — all four tests pass. **Result**: 4 passed, 92 deselected.
- [x] 2.3 Run `uv run pytest tests/test_points.py -x` — full file passes (no regressions). **Result**: 96 passed.

## 3. Write failing tests for `MultipleDicotPlatePipeline` (TDD)

**Setup helper**: before writing pipeline tests, add a test helper (or pytest fixture) at the top of `tests/test_trait_pipelines.py` that builds a synthetic `.slp` file on disk using `sio.save_slp` + `sio.Labels` (same idiom as `tests/test_pixel_units.py`). The helper takes per-frame primary and lateral point arrays and returns a `Series` via `Series.load`. This is the mechanism every integration test in this section uses — **do NOT bypass it by constructing `initial_frame_traits` directly, because the `compute_plate_traits(series)` frame loop must be exercised end-to-end through the Series interface.**

### 3a. Unit tests (targeting the TraitDef DAG and `get_initial_frame_traits`)

- [x] 3a.1 Add `test_multiple_dicot_plate_pipeline_define_traits` — instantiate the pipeline and assert:
  - `isinstance(pipeline, Pipeline) is True` (Req 1 scenario "Class is importable and instantiable")
  - `[t.name for t in pipeline.define_traits()]` contains (in order) `primary_pts_no_nans`, `lateral_pts_no_nans`, `detected_count`, `plant_associations_dict`, `plant_id_order`
  - `detected_count` is `scalar=True, include_in_csv=True`
  - No TraitDef uses `filter_plants_with_unexpected_ct` (plates skip count-filtering per design D2)
  - `plant_id_order` uses `argsort_primaries_by_base_x` with input `["plant_associations_dict"]`
- [x] 3a.1b Add `test_multiple_dicot_plate_pipeline_top_level_import` — verify `from sleap_roots import MultipleDicotPlatePipeline` succeeds AND the imported symbol `is sleap_roots.trait_pipelines.MultipleDicotPlatePipeline`.
- [x] 3a.2 Add `test_multiple_dicot_plate_pipeline_get_initial_frame_traits_keys` — synthetic Series, assert returned dict has keys `{primary_pts, lateral_pts, primary_sleap_idxs, lateral_sleap_idxs, expected_count}`.
- [x] 3a.3 Add `test_multiple_dicot_plate_pipeline_get_initial_frame_traits_sleap_idxs` — synthetic primaries of shape `(4, 6, 2)` with instance 1 all-NaN; assert:
  - `primary_sleap_idxs == [0, 2, 3]` (skips NaN primary by original SLEAP index, NOT post-filter index)
  - `initial["primary_pts"].shape == (4, 6, 2)` (raw pre-filter shape preserved — Req 1 scenario c explicitly requires this)

### 3b. Integration tests (round-trip through synthetic `.slp` via `Series.load`)

- [x] 3b.1 Add `test_multiple_dicot_plate_pipeline_basic`. Build a synthetic `.slp` with 3 primaries at known base-x `[200, 50, 100]` plus laterals. Call `compute_plate_traits(series)`. Assert `plants` length == 3, `plant_id` order is left-to-right (primary at x=50 has `plant_id=0`, x=100 has `plant_id=1`, x=200 has `plant_id=2`), `primary_sleap_idx` values map back to original SLEAP indices (1, 2, 0 for the left-to-right order), every plant row has `detected_count == 3`, every `traits` dict contains DicotPipeline trait names **unchanged** (`primary_length`, `lateral_count`, `network_length`, `primary_base_tip_dist` all present — NO `primary_root_length`-style renames).
- [x] 3b.2 Add `test_multiple_dicot_plate_pipeline_sleap_idx_traceability`. Build a synthetic `.slp` with 3 primaries where SLEAP index 1 is all-NaN. Call `compute_plate_traits(series)`. Assert `len(plants) == 2`, `{p["primary_sleap_idx"] for p in plants} == {0, 2}`, NOT `{0, 1}`.
- [x] 3b.3 Add `test_multiple_dicot_plate_pipeline_expected_count_none`. Build a synthetic `.slp` and load the Series WITHOUT a `csv_path` so `series.expected_count` is `np.nan`. Use pytest's `caplog` fixture at WARNING level. Call `compute_plate_traits(series)`. Assertions (use `math.isnan` / `pd.isna`, NOT `is np.nan` — identity comparison fails on `float('nan')` returned by `json.load`):
  - no exception raised
  - every `plants[i]` has `pd.isna(plants[i]["expected_count"])` is True in memory
  - every `plants[i]["count_validated"] is False` AND `plants[i]["count_mismatch"] is False` (unknown ≠ mismatch)
  - every `plants[i]["detected_count"]` is a non-negative int
  - top-level `pd.isna(result["qc_fail"])` is True (cylinder-specific; plate Series without qc_cylinder column → nan)
  - `caplog.records` contains no WARNING records from logger `sleap_roots.trait_pipelines` (spec scenario "Missing expected_count yields both flags False and logs nothing")
- [x] 3b.4 Add `test_multiple_dicot_plate_pipeline_expected_count_mismatch`. Build a synthetic `.slp` with 3 detected primaries. Load the Series with a CSV specifying `expected_count=2`. Use pytest's `caplog` fixture at WARNING level. Call `compute_plate_traits(series)`. Assertions:
  - `len(plants) == 3` (no dropping)
  - every plant row: `expected_count == 2` AND `detected_count == 3`
  - every `plants[i]["count_mismatch"] is True` AND `plants[i]["count_validated"] is False`
  - `caplog.records` contains at least one WARNING record from `sleap_roots.trait_pipelines` whose message contains both `"3"` (detected) and `"2"` (expected) AND `"frame 0"` (or equivalent frame identifier)
  - **NOTE on mechanism change**: the pipeline uses `logging.warning`, NOT `warnings.warn(UserWarning)`, because batch processors commonly suppress `UserWarning`. This was a round-2 review finding. Tests use `caplog`, not `pytest.warns`.
- [x] 3b.4b Add `test_multiple_dicot_plate_pipeline_expected_count_match`. Same setup but `expected_count=3`. Use `caplog` at WARNING level. Assert: every `plants[i]["count_validated"] is True`, `plants[i]["count_mismatch"] is False`, `caplog.records` contains no WARNING records from `sleap_roots.trait_pipelines`.
- [x] 3b.5 Add `test_multiple_dicot_plate_pipeline_empty_frame`. Build a synthetic `.slp` where every primary instance in frame 0 is all-NaN. Note: the synthetic `.slp` builder helper MUST use a skeleton with **≥3 nodes** (to avoid crashing `get_node_ind` — see issue #150) AND must not rely on `Series.get_primary_points`'s NaN-placeholder `(1, 2, 2)` shape (the test constructs explicit all-NaN instances with the matching node count). Call `compute_plate_traits(series)`. Assert `plants == []`, no exception.
- [x] 3b.5b Add `test_multiple_dicot_plate_pipeline_zero_laterals`. Build a synthetic `.slp` with ONE primary at known coordinates and ZERO lateral instances. Call `compute_plate_traits(series)`. Assertions (Req 3 scenario "Zero-laterals plant yields lateral_count == 0"):
  - `plants[0]["traits"]["lateral_count"] == 0` (NOT 1)
  - `plants[0]["traits"]["lateral_lengths"]` is an empty array (`shape[0] == 0`), NOT `[nan]`
  - `plants[0]["traits"]["network_length"] == plants[0]["traits"]["primary_length"]` (laterals contribute 0, not NaN)
  - `plants[0]["lateral_sleap_idxs"] == []`
- [x] 3b.5c Add `test_multiple_dicot_plate_pipeline_duplicate_lateral_coords`. Build a synthetic `.slp` with 1 primary and 2 laterals whose `(n_nodes, 2)` node coordinates are **bit-identical** (copy the same numpy array to both instances). Call `compute_plate_traits(series)`. Assertions (Req 3 scenario "Duplicate lateral coordinates disambiguate back to distinct SLEAP indices"):
  - `len(plants) == 1`
  - `len(plants[0]["lateral_sleap_idxs"]) == 2` (both laterals present)
  - `set(plants[0]["lateral_sleap_idxs"]) == {0, 1}` (distinct SLEAP indices, NOT `{0, 0}` or `{1, 1}`)
  - This test fails if the implementation uses `np.array_equal`-first-match back-mapping; passes only if SLEAP indices are tracked alongside the distance-based association.
- [x] 3b.6 Add `test_multiple_dicot_plate_pipeline_timelapse_shape`. Build a synthetic `.slp` with 2 frames, each containing the same 3 primaries. Call `compute_plate_traits(series)`. Assertions (Req 3 scenario "Multi-frame series produces N_plants × N_frames rows grouped by frame"):
  - `len(plants) == 6`
  - `plants[0]["frame"] == plants[1]["frame"] == plants[2]["frame"] == 0`
  - `plants[3]["frame"] == plants[4]["frame"] == plants[5]["frame"] == 1`
  - within each frame, `plant_id` values are `[0, 1, 2]` in left-to-right order
- [x] 3b.7 Add `test_multiple_dicot_plate_pipeline_csv_output`. Build a synthetic `.slp` with 2 plants. Call `compute_plate_traits(series, write_csv=True, output_dir=tmp_path)`. Read via `df = pd.read_csv(tmp_path / f"{series.series_name}.plate_traits.csv")`. Assertions:
  - `list(df.columns)[0:6] == ["series", "frame", "plant_id", "primary_sleap_idx", "expected_count", "detected_count"]`
  - `list(df.columns)[6:] == DicotPipeline().csv_traits` (ordered equality)
  - `set(df.columns[6:]) == set(DicotPipeline().csv_traits)` (set equality as diagnostic when order differs)
  - no column name contains `_root_` as an infix
  - `"lateral_sleap_idxs"` is NOT a column
  - `"count_validated"` is NOT a column
  - `"count_mismatch"` is NOT a column
- [x] 3b.7b Add `test_multiple_dicot_plate_pipeline_csv_missing_expected_count`. Series loaded without CSV (`expected_count=NaN`), **1 detected primary**. After `compute_plate_traits(series, write_csv=True, output_dir=tmp_path)` + `pd.read_csv`, assert:
  - `pd.isna(df.loc[0, "expected_count"])` is True
  - `df.loc[0, "detected_count"] == 1` (1 detected primary → 1 row → `detected_count == 1`)
  - `"count_validated"` is NOT in `df.columns`
  - `"count_mismatch"` is NOT in `df.columns`
- [x] 3b.8 Add `test_multiple_dicot_plate_pipeline_json_output`. Call `compute_plate_traits(series, write_json=True, output_dir=tmp_path)`. Read back via `json.load`. Assertions (Req 4 "Written JSON is self-contained and round-trips"):
  - top-level keys `{schema_version, units, series, group, qc_fail, expected_count, plants}`
  - `result["schema_version"] == 1`
  - `result["units"] == {"lengths": "pixels", "angles": "degrees", "counts": "unitless", "ratios": "dimensionless"}` (structured units object; NOT the single-string `"pixels"`)
  - each plant entry has `primary_points` as list-of-`[x,y]`, `lateral_points` as list-of-lists-of-`[x,y]`, `primary_sleap_idx` as int, `lateral_sleap_idxs` as list of ints
  - each plant entry has `count_validated` and `count_mismatch` as JSON-native booleans (verified via `isinstance(plants[i]["count_validated"], bool)` — NOT `int`)
  - `traits` dict keys include DicotPipeline trait names unchanged
- [x] 3b.8b Add `test_multiple_dicot_plate_pipeline_json_rfc8259_valid_with_nested_nan`. Setup: one zero-laterals plant (deterministically produces NaN deep in `plants[0]["traits"]["lateral_angles_distal"]` per [angle.py:70-71](../../../sleap_roots/angle.py#L70-L71)) AND Series loaded without CSV (so top-level `expected_count` is NaN). After `compute_plate_traits(series, write_json=True, output_dir=tmp_path)`:
  - the call MUST NOT raise (sanitizer converts all NaN to None before json.dump; `allow_nan=False` defense-in-depth does NOT fire in the happy path)
  - `"NaN" not in tmp_path.joinpath(f"{series.series_name}.plate_traits.json").read_text()` (string-grep assertion)
  - strict RFC-8259 parse does NOT raise:
    ```python
    def _raise_on_constant(s): raise ValueError(f"bare constant {s!r}")
    json.loads(tmp_path.joinpath(f"{series.series_name}.plate_traits.json").read_text(),
               parse_constant=_raise_on_constant)
    ```
  - after re-loading: `loaded["expected_count"] is None` (top-level NaN → null)
  - AND `loaded["plants"][0]["traits"]["lateral_angles_distal"] is None` (deep NaN → null; exercises the recursive sanitizer path)
  - AND `loaded["plants"][0]["lateral_sleap_idxs"] == []` AND `loaded["plants"][0]["lateral_points"] == []` (zero-laterals NaN placeholder NOT serialized)
- [x] 3b.9 Add `test_compute_batch_plate_traits`. Build two synthetic `.slp` Series (2 plants × 1 frame, 3 plants × 1 frame). Call `compute_batch_plate_traits([seriesA, seriesB])`. Assert returned DataFrame has 5 rows, `series` column matches expectations for rows 0-1 vs 2-4. Call again with `write_json=True, output_dir=tmp_path, json_name="batch.json"` and assert `json.loads((tmp_path / "batch.json").read_text())` is a list of 2 per-series dicts, each with `{schema_version, units, series, group, qc_fail, expected_count, plants}` keys (matching the single-series format). Also assert the batch JSON passes the same strict-RFC-8259 parse check as 3b.8b.
- [x] 3b.10 Run `uv run pytest tests/test_trait_pipelines.py -k multiple_dicot_plate_pipeline -v` — confirm all **18** tests (4 in 3a: 3a.1, 3a.1b, 3a.2, 3a.3; plus 14 in 3b: 3b.1, 3b.2, 3b.3, 3b.4, 3b.4b, 3b.5, 3b.5b, 3b.5c, 3b.6, 3b.7, 3b.7b, 3b.8, 3b.8b, 3b.9) FAIL with `AttributeError` / `ImportError` (class does not yet exist). Paste the failing-test summary into the PR body under a `## TDD evidence` heading so reviewers can verify TDD discipline.

## 4. Implement `MultipleDicotPlatePipeline`

Implement in `sleap_roots/trait_pipelines.py`. Keep the implementation minimal — only what the tests in section 3 require.

- [x] 4.1 Add `MultipleDicotPlatePipeline` class with `@attrs.define` decorator. `define_traits()` returns the 5-element TraitDef DAG described in section 3a.1. Google-style docstring at class level MUST note:
  - (a) plates skip count-filter (D2)
  - (b) `plant_id` is a left-to-right ordering paired with original SLEAP indices (D3); `plant_id` is NOT stable across SLEAP model re-prediction
  - (c) per-plant traits reuse DicotPipeline trait names unchanged (D6)
  - (d) `primary_base_tip_dist` is the substituted "depth" trait for PR 1; a dedicated max-y-extent trait is tracked in follow-up F (document #126's original `primary_root_depth` definition alongside this)
  - (e) `qc_fail` inherits `Series.qc_fail`'s cylinder-specific semantics (reads CSV column `qc_cylinder`); tracked in follow-up E
  - (f) `expected_count` inherits `Series.expected_count`'s cylinder-specific column name (`number_of_plants_cylinder`); tracked in follow-up E
  - (g) `filter_roots_with_nans` drops any primary with even one NaN node (whole-root filter); compounds with plant_id fragility
  - (h) **NO back-mapping key is stable across SLEAP model re-prediction.** Both `plant_id` AND `primary_sleap_idx` can shift if predictions at the confidence threshold flip between runs. For cross-run alignment (e.g., matching plants across re-predicted timelapse frames), use spatial matching (nearest-base-x within tolerance) rather than either identifier.
- [x] 4.2 Implement `get_initial_frame_traits(plant, frame_idx)` to return the 5-key dict. Compute `primary_sleap_idxs` / `lateral_sleap_idxs` via the validity-mask-before-filter pattern: `primary_sleap_idxs = [i for i, r in enumerate(primary_pts_raw) if not np.isnan(r).any()]` (and likewise for lateral). See design doc § "SLEAP instance index mapping".
- [x] 4.3 Implement `compute_plate_traits(series, write_csv=False, write_json=False, output_dir=".", csv_suffix=".plate_traits.csv", json_suffix=".plate_traits.json")`. Frame loop follows design doc § "Compute flow (per series)".
  - **Zero-laterals detection (Req 3 zero-laterals scenario, design D2)**: before invoking the nested DicotPipeline, check `if assoc["lateral_points"].shape[0] == 1 and not is_line_valid(assoc["lateral_points"][0])`. If true, pass `np.empty((0, assoc["lateral_points"].shape[1], 2))` as `lateral_pts` to the nested DicotPipeline (so `lateral_count` computes to 0 and `lateral_lengths` is an empty array, NOT `[nan]`).
  - **SLEAP index back-mapping (Req 3 duplicate-coord scenario)**: do NOT rely on `np.array_equal` first-match. Instead, track SLEAP indices alongside the distance-based association: in `compute_plate_traits`, rebuild the primary-lookup loop using the same `LineString.distance` logic as `associate_lateral_to_primary` ([points.py:511-593](../../../sleap_roots/points.py#L511-L593)) — but this time maintain a `lateral_sleap_idx_by_primary: Dict[int, List[int]]` map. For each valid lateral at post-filter index `i` (SLEAP index `initial["lateral_sleap_idxs"][i]`), compute the nearest primary and append the SLEAP index to that primary's list. Add an in-code comment "KEEP IN SYNC WITH associate_lateral_to_primary in sleap_roots/points.py" to flag the duplication for future maintainers (a follow-up to refactor onto a shared helper is beyond PR 1 scope).
  - **Count-flag derivation (Req 3 count_mismatch scenario, design D5b)**: compute `count_validated` and `count_mismatch` booleans per plant from `expected_count` / `detected_count`; emit as JSON fields (NOT CSV columns). Semantics: `count_validated = expected_count is not None and not math.isnan(expected_count) and int(round(expected_count)) == detected_count`; `count_mismatch = ... and int(round(expected_count)) != detected_count`. Both `False` when expected_count is unknown.
  - **Per-frame log on mismatch (design D5b)**: use a module-level `logger = logging.getLogger(__name__)` (from `import logging` at the top of trait_pipelines.py). For each frame where `detected_count != expected_count AND expected_count is resolvable`, emit `logger.warning(f"MultipleDicotPlatePipeline: {series.series_name} frame {frame_idx} detected {detected_count} primaries but expected {expected_count}; no plants dropped")`. Dedupe at (series, frame) granularity (not per-plant-row). Use `logging.warning`, NOT `warnings.warn(UserWarning)` — batch processors routinely filter `UserWarning` and would lose the signal silently.
  - **Top-level metadata**: include `schema_version: 1` and the structured `units` object (see task 4.4) in the in-memory return dict.
- [x] 4.4 Implement JSON serialization with pre-serialization NaN sanitization.
  - Add a private helper `_json_sanitize(obj)` in `sleap_roots/trait_pipelines.py` that recursively walks `obj` and returns a new structure where (a) `dict` keys/values are visited recursively, (b) `list`/`tuple` elements are visited recursively, (c) `np.ndarray` is converted via `.tolist()` and the resulting nested lists are then walked recursively, (d) any scalar `float('nan')` / `np.floating` NaN (detected via `isinstance(x, (float, np.floating)) and not np.isfinite(x) and np.isnan(x)`) is replaced with `None`, (e) `np.int64` / `np.integer` is cast to `int`, (f) all other values pass through unchanged.
  - **Why recursive sanitization instead of `JSONEncoder.default()` subclass**: CPython's `json.JSONEncoder` has a fast path for native `float` (including `np.float64`, which subclasses `float`) that BYPASSES `default()`. With `allow_nan=True`, bare `NaN` is emitted; with `allow_nan=False`, `ValueError` is raised BEFORE `default()` is called. `default()` hook alone cannot intercept scalar NaN. This was empirically confirmed in openspec-review round 2 (Subagent 2).
  - Invoke: `sanitized = _json_sanitize(result); json.dump(sanitized, f, cls=NumpyArrayEncoder, allow_nan=False, ensure_ascii=False, indent=4)`. `allow_nan=False` is defense-in-depth: if the sanitizer misses any NaN, the write raises `ValueError` with a clear error rather than producing invalid JSON.
  - Structured `units` object (design D5): `{"lengths": "pixels", "angles": "degrees", "counts": "unitless", "ratios": "dimensionless"}`. A single-string `"units": "pixels"` would be factually wrong because `DicotPipeline.lateral_angles_distal` etc. are emitted in degrees by `sleap_roots/angle.py`.
  - **Zero-laterals JSON cleanup**: before sanitization, post-process the per-plant `lateral_points` entries: if the entry came from the zero-laterals NaN placeholder (detected in 4.3), replace with `[]` (empty list) and ensure `lateral_sleap_idxs` is also `[]`. This ensures the written JSON has `"lateral_points": []` and `"lateral_sleap_idxs": []` for zero-laterals plants (per Req 4 zero-laterals scenario).
- [x] 4.5 Implement CSV flattening: build per-plant rows where each row contains the 6 metadata columns (`series`, `frame`, `plant_id`, `primary_sleap_idx`, `expected_count`, `detected_count`) + each DicotPipeline csv-trait (scalar as-is; non-scalar via `get_summary` → `{name}_{min,max,...}` suffixes from `sleap_roots/summary.py`). Use `pd.DataFrame(rows).to_csv(path, index=False)` — pandas renders `np.nan` / `None` as empty cells by default. `count_validated` and `count_mismatch` MUST NOT appear as CSV columns (JSON only).
- [x] 4.6 Implement `compute_batch_plate_traits(all_series, write_csv=False, write_json=False, output_dir=".", csv_name="plate_batch_traits.csv", json_name="plate_batch_traits.json")`. Loop over series, concatenate per-series DataFrames, optionally write CSV/JSON (JSON uses the same plate encoder from 4.4).
- [x] 4.7 Add `MultipleDicotPlatePipeline` to `sleap_roots/__init__.py` top-level re-exports, parity with `DicotPipeline`, `MultipleDicotPipeline`, etc. (Req 1 scenario "Class is importable from the top-level sleap_roots namespace"; Subagent 2 Finding I7).
- [x] 4.8 Run `uv run pytest tests/test_trait_pipelines.py tests/test_points.py -k "multiple_dicot_plate_pipeline or argsort_primaries_by_base_x" -v` — all 18 new tests pass.
- [x] 4.9 Run `uv run pytest tests/ -x` — **full** test suite passes (broadened from the previous narrow scope per Subagent 1 Finding; catches ripple effects in `test_series.py`, `test_summary.py`, etc. before section 5 files follow-up issues).

## 5. File follow-up GitHub issues BEFORE the PR merges

Eight issues must be filed and verified before the PR is marked ready for review. Each `gh issue create` command below includes `follow-up of #126` in the body so the verification query in 5.8 matches all of them.

- [ ] 5.1 File issue **PR 2** — "Add tertiary root support to `MultipleDicotPlatePipeline`". Body references this PR and design doc section on tertiary→primary direct association (D4).
  ```
  gh issue create --title "Add tertiary root support to MultipleDicotPlatePipeline (PR 2 of #126)" \
    --body "Follow-up of #126 (PR 2 of 3). See docs/superpowers/specs/2026-04-16-multiple-dicot-plate-pipeline-design.md § D4. Scope: Series.get_tertiary_points, tertiary_path attribute, reuse associate_lateral_to_primary with tertiary input, emit tertiary columns. Note: once PR 2 lands, network_length semantics change (tertiaries add into the sum); downstream consumers of PR 1's network_length column must be aware."
  ```
- [ ] 5.2 File issue **PR 3** — "Add configurable filtering thresholds to `MultipleDicotPlatePipeline`".
  ```
  gh issue create --title "Add configurable filtering thresholds to MultipleDicotPlatePipeline (PR 3 of #126)" \
    --body "Follow-up of #126 (PR 3 of 3). Add min_primary_length_px, min_lateral_length_px, node_score_threshold, primary_angle_filter as constructor kwargs with plate-specific defaults."
  ```
- [ ] 5.3 File issue **A** — "Standardize multi-plant pipeline JSON output + deterministic plant_id".
  ```
  gh issue create --title "Standardize multi-plant pipeline JSON output (follow-up A from #126)" \
    --body "Follow-up of #126. Apply plate JSON format (raw points, SLEAP instance indices, left-to-right sorted plant_id, schema_version, units, NaN->null encoding) to MultipleDicotPipeline and MultiplePrimaryRootPipeline."
  ```
- [ ] 5.4 File issue **B** — "Include raw points in single-plant pipeline JSON".
  ```
  gh issue create --title "Include raw points in single-plant pipeline JSON (follow-up B from #126)" \
    --body "Follow-up of #126. Apply to DicotPipeline, YoungerMonocotPipeline, OlderMonocotPipeline for self-contained analysis artifacts."
  ```
- [ ] 5.5 File issue **C** — "Plate visualization / viewer".
  ```
  gh issue create --title "Plate visualization / viewer for MultipleDicotPlatePipeline JSON (follow-up C from #126)" \
    --body "Follow-up of #126. Consume plate pipeline JSON (schema_version=1, units=pixels) to render colored per-plant overlays with plant_id labels near primary base nodes. Related to #128."
  ```
- [ ] 5.6 File issue **D** — "Real plate `.slp` fixture tests".
  ```
  gh issue create --title "Real plate .slp fixture tests for MultipleDicotPlatePipeline (follow-up D from #126)" \
    --body "Follow-up of #126. Add MK22 dataset integration tests once fixtures are available. Synthetic-only tests land in PR 1. Related to #119 (plate test data)."
  ```
- [ ] 5.7 File issue **E** — "Generalize cylinder-conventional CSV column names for plates".
  ```
  gh issue create --title "Generalize cylinder-conventional CSV column names in Series (follow-up E from #126)" \
    --body "Follow-up of #126. Three Series properties read cylinder-named CSV columns that plates also have to use: Series.qc_fail reads qc_cylinder (series.py:196-208), Series.expected_count reads number_of_plants_cylinder (series.py:165-180). Plate CSV schemas should be able to use plate-named columns (qc_plate, number_of_plants_plate) or a unified name. Options: add plate-aware fallback column names, or add constructor kwargs for column-name resolution."
  ```
- [ ] 5.7b File issue **F** — "Plate-specific depth trait (max y-extent)".
  ```
  gh issue create --title "Plate-specific depth trait matching #126's max y-extent definition (follow-up F from #126)" \
    --body "Follow-up of #126. Issue #126 defines primary_root_depth as 'max y-extent (deepest node y - base node y)'. PR 1 substitutes primary_base_tip_dist (Euclidean base-to-tip) as the closest existing DicotPipeline trait; see design doc § 'Deviation from #126 on primary_root_depth'. This issue tracks introducing a dedicated max-y-extent or pure-y-depth scalar after real plate data informs which variant is biologically correct."
  ```
- [ ] 5.7c Post a comment on #126 documenting the trait-name mapping (Subagent 3 Finding I1) so the original requester can see the substitution and push back if needed.
  ```
  gh issue comment 126 --body "$(cat <<'EOF'
## PR 1 trait-name mapping — summary for the requester

PR 1 (in branch `feature/multiple-dicot-plate-pipeline-126`) reuses `DicotPipeline` trait names unchanged to preserve the project-wide naming convention (`primary_*`, `lateral_*`, `network_*` prefixes; no `_root_` infix). Mapping from this issue's original acceptance criteria:

| Requested column | Emitted trait | Notes |
|---|---|---|
| `primary_root_length` | `primary_length` | Direct rename; same scalar. |
| `primary_root_depth` | `primary_base_tip_dist` | **Semantic substitution** — Euclidean base-to-tip distance, NOT the "max y-extent" originally specified. Tracked in follow-up F for a dedicated max-y-extent trait. |
| `lateral_root_count` | `lateral_count` | Direct rename. |
| `avg_lateral_root_length` | `lateral_lengths_mean` | Auto-derived from non-scalar `lateral_lengths` via CSV stat-suffix mechanism. |
| `max_lateral_root_length` | `lateral_lengths_max` | Same mechanism. |
| `total_lateral_root_length` | (no direct trait) | Derive as `network_length - primary_length` OR wait for tertiary support in PR 2. Not in PR 1's CSV. |
| `total_root_network_length` | `network_length` | `network_length` = `primary_length + sum(lateral_lengths)`; tertiaries added in PR 2. |

If the `_root_`-infixed names are actually required for external downstream scripts, please comment here and PR 1's scope can reopen — the cost is maintaining parallel trait-name dialects project-wide.
EOF
)"
  ```

### Verification gate

- [ ] 5.8 Run the verification command and confirm exactly 8 new issues come back. The query matches any open issue whose body contains `#126` (all 8 bodies include the phrase `Follow-up of #126` OR `PR 2 of #126` / `PR 3 of #126`):
  ```
  gh issue list --state open --search "follow-up #126 in:body" --limit 20 --json number,title,body
  gh issue list --state open --search "Tracks PR of #126 in:body" --limit 20 --json number,title,body
  ```
  Since no single `gh issue list` query reliably matches all 8 bodies (PR 2 / PR 3 bodies use "PR 2 of #126" / "PR 3 of #126"; A-F use "Follow-up of #126"), the simpler approach is to enumerate open issues whose body mentions #126:
  ```
  gh issue list --state open --search "#126 in:body" --limit 50 --json number,title,body
  ```
  Manually confirm **exactly 8** distinct new issues in the output (one each for PR 2, PR 3, A, B, C, D, E, F — excluding #126 itself). If the count is not 8, re-file any missing ones before proceeding.
- [ ] 5.9 Copy the 8 issue numbers into the PR body under a `## Follow-up issues filed during this PR` heading with each issue's title and number.

## 6. Pre-merge validation

- [ ] 6.1 Run `openspec validate add-multiple-dicot-plate-pipeline --strict` and resolve any issues.
- [ ] 6.2 Run `uv run black --check sleap_roots/ tests/` — formatting clean.
- [ ] 6.3 Run `uv run pydocstyle sleap_roots/points.py sleap_roots/trait_pipelines.py` — docstrings clean for the new additions.
- [ ] 6.4 Run `uv run pytest tests/ -x` — full test suite passes.
- [ ] 6.5 Invoke `/review-pr` or equivalent self-review on the branch before opening the PR.
- [ ] 6.6 Open the PR; body MUST reference issue #126, link the 8 follow-up issue numbers from 5.9, and include the `## TDD evidence` section from task 3b.10.

## Dependencies

- Section 2 depends on section 1.
- Section 3 (tests-first) must be written before section 4 (implementation). Section 1-2 (helper) is a natural commit boundary independent of section 3-4 (pipeline).
- Section 4 depends on section 2 (uses `argsort_primaries_by_base_x`).
- Section 5 can start any time but all 8 issues MUST exist and the #126 comment MUST be posted before section 6.6 (PR ready for review).
- Section 6 depends on sections 2, 4, and 5.

## Notes

- **Source-of-truth reference**: [`docs/superpowers/specs/2026-04-16-multiple-dicot-plate-pipeline-design.md`](../../../docs/superpowers/specs/2026-04-16-multiple-dicot-plate-pipeline-design.md) on branch `feature/multiple-dicot-plate-pipeline-126`. Amendment commits are visible via `git log docs/superpowers/specs/2026-04-16-multiple-dicot-plate-pipeline-design.md`.
- **Why no renaming of trait names**: the project-wide convention uses `primary_*`, `lateral_*`, `network_*`, `crown_*` prefixes; no `_root_` infix anywhere. Renaming to `primary_root_length` etc. would have created a parallel dialect that conflicts with `DicotPipeline.primary_length` at [trait_pipelines.py:1425](../../../sleap_roots/trait_pipelines.py#L1425). See design doc § D6 and § "Per-plant traits (PR 1)".
- **Why `primary_base_tip_dist` substitutes for `primary_root_depth`**: #126 originally defined `primary_root_depth` as "max y-extent (deepest node y − base node y)". No existing DicotPipeline trait matches this exactly. PR 1 substitutes `primary_base_tip_dist` (Euclidean base-to-tip distance) as the closest existing scalar, tracked in follow-up F (task 5.7b) for proper resolution. This substitution is documented via a comment on #126 (task 5.7c).
- **Why JSON includes raw points**: the plate JSON is intended as a self-contained analysis artifact (design D5). The plate viewer follow-up (issue C) will consume this JSON directly — no dependency on the source `.slp` at visualization time. `~30 KB` overhead per plate is acceptable.
- **Why synthetic `.slp` round-trip (not in-memory mocks)**: `Series.get_primary_points` / `.get_lateral_points` have non-trivial behavior around user_instances + unused_predictions stacking and NaN placeholder injection. Integration tests go through `Series.load` to exercise that code path. `tests/test_pixel_units.py` demonstrates the synthetic-`.slp` idiom via `sio.save_slp`. **Caveat**: PR 1's synthetic-`.slp` builder constructs user-instance labels, which exercises the `user_instances` branch of `Series.get_primary_points` but NOT the `unused_predictions` branch; that branch is covered by real-fixture tests in follow-up D.
- **Why skeletons must have ≥3 nodes**: issue #150 documents a `get_node_ind` crash on roots with <3 nodes. All synthetic primaries in section 3 tests use ≥3 nodes (matching the 6-node convention in `test_pixel_units.py`). This is a known cross-cutting constraint, not a PR 1 bug.
- **Why JSON NaN → null**: default `json.dumps` emits bare `NaN` tokens that violate RFC 8259 and break non-Python consumers (JavaScript `JSON.parse`, Go `encoding/json`, `jq`). Since the plate JSON is the input to the plate viewer (follow-up C) and potentially other cross-language tools, PR 1 MUST emit NaN as JSON `null` via a plate-aware encoder. See design doc § D5 and spec Req 4 scenario "Written JSON is RFC-8259-valid".
