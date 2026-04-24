# Tasks: Add Series metadata layer (sample_uid, timepoint, get_metadata)

**Source of truth for architecture**: `docs/superpowers/specs/2026-04-23-timelapse-diff-and-tip-kinematics-design.md` § Workstream 1 (PR #171). Implementation plan with task-level detail: `docs/superpowers/plans/2026-04-24-metadata-layer.md`.

**Workflow guarantees**:

- Strict TDD: each subsection writes failing tests BEFORE implementation.
- All commands use `uv run` (`uv run pytest`, `uv run black`, `uv run pydocstyle`).
- No implementation is merged until every follow-up issue in section 5 is filed and linked in the PR body.

## 1. Write failing tests for `Series.get_metadata`

- [ ] 1.1 Add `test_series_get_metadata_no_csv` — Series without `csv_path` → `get_metadata("anything")` returns `np.nan`.
- [ ] 1.2 Add `test_series_get_metadata_missing_column` — CSV exists but lacks the requested column → NaN.
- [ ] 1.3 Add `test_series_get_metadata_no_matching_row` — CSV has the column but no row matches sample_uid → NaN.
- [ ] 1.4 Add `test_series_get_metadata_matches_row` — CSV has matching row → returns the column value.
- [ ] 1.5 Add `test_series_get_metadata_plant_id_composite_lookup` — CSV has a `plant_id` column; composite lookup returns the correct row.
- [ ] 1.6 Add `test_series_get_metadata_plant_id_ignored_when_no_column` — plant_id argument silently ignored when CSV lacks `plant_id` column.
- [ ] 1.7 Run `uv run pytest tests/test_series.py -k get_metadata -x` — confirm all 6 tests FAIL with `AttributeError: 'Series' object has no attribute 'get_metadata'`.

## 2. Implement `Series.get_metadata`

- [ ] 2.1 Add `get_metadata(column, plant_id=None)` to `Series`. Also add `sample_uid` attrs field + `Series.load` kwarg (prerequisite — `get_metadata` reads `self.sample_uid`). Refactor `expected_count`, `group`, `qc_fail` as thin wrappers.
- [ ] 2.2 Run `uv run pytest tests/test_series.py -k get_metadata -v` — all 6 pass.
- [ ] 2.3 Run `uv run pytest tests/test_series.py -x` — existing tests still pass (regression check for refactored properties).

## 3. Write failing tests for `Series.sample_uid` + `Series.timepoint`

- [ ] 3.1 Add `test_series_sample_uid_defaults_to_series_name` — no kwarg → `sample_uid == series_name`.
- [ ] 3.2 Add `test_series_sample_uid_explicit_kwarg` — explicit kwarg → `sample_uid == kwarg value`.
- [ ] 3.3 Add `test_series_sample_uid_shared_across_series` — two Series with different `series_name` but same `sample_uid`.
- [ ] 3.4 Add `test_series_timepoint_from_csv` — CSV has timepoint column → property returns value.
- [ ] 3.5 Add `test_series_timepoint_no_csv` → NaN.
- [ ] 3.6 Run `uv run pytest tests/test_series.py -k "sample_uid or timepoint" -x` — timepoint tests FAIL with `AttributeError`; sample_uid tests pass (field landed with Task 2.1 as prerequisite).

## 4. Implement `Series.timepoint`

- [ ] 4.1 Add `timepoint` property wrapping `get_metadata("timepoint")`.
- [ ] 4.2 Run `uv run pytest tests/test_series.py -k "sample_uid or timepoint" -v` — all pass.
- [ ] 4.3 Run `uv run pytest tests/test_series.py -x` — full file still passes.

## 5. Write failing tests for `sleap_roots/metadata.py`

- [ ] 5.1 Create `tests/test_metadata.py` with tests below.
- [ ] 5.2 Add `test_build_metadata_csv_canonical_column_order` — verify canonical order with subset of standard columns.
- [ ] 5.3 Add `test_build_metadata_csv_omits_unused_columns` — columns not present in any row are omitted.
- [ ] 5.4 Add `test_build_metadata_csv_raises_on_missing_plant_qr_code` — `ValueError` with "plant_qr_code" in message.
- [ ] 5.5 Add `test_build_metadata_csv_returns_path` — function returns the Path of the written file.
- [ ] 5.6 Add `test_build_metadata_csv_accepts_str_path` — accepts str path (not just Path).
- [ ] 5.7 Add `test_build_metadata_csv_extras_sorted` — non-canonical columns appear sorted at the end.
- [ ] 5.8 Add `test_infer_timepoints_from_filenames_named_groups` — correct dict output.
- [ ] 5.9 Add `test_infer_timepoints_from_filenames_missing_named_groups` — `ValueError` with `series_name` / `timepoint` in message.
- [ ] 5.10 Add `test_infer_timepoints_from_filenames_skips_non_matches` — garbage stems skipped silently.
- [ ] 5.11 Add `test_infer_timepoints_from_filenames_casts_to_float` — timepoint values are cast to float.
- [ ] 5.12 Run `uv run pytest tests/test_metadata.py -x` — all FAIL with `ImportError`.

## 6. Implement `sleap_roots/metadata.py`

- [ ] 6.1 Create `sleap_roots/metadata.py` with `build_metadata_csv` and `infer_timepoints_from_filenames`.
- [ ] 6.2 Add re-exports to `sleap_roots/__init__.py`.
- [ ] 6.3 Run `uv run pytest tests/test_metadata.py -v` — all pass.

## 7. Write failing tests for plate pipeline `sample_uid`/`timepoint` emission

- [ ] 7.1 Add `test_multiple_dicot_plate_pipeline_emits_sample_uid_and_timepoint_top_level` — result dict has both keys at top level.
- [ ] 7.2 Add `test_multiple_dicot_plate_pipeline_emits_sample_uid_and_timepoint_per_plant` — every plant row has both keys.
- [ ] 7.3 Add `test_multiple_dicot_plate_pipeline_csv_column_positions` — CSV column 1 = `sample_uid`, column 2 = `timepoint`.
- [ ] 7.4 Add `test_multiple_dicot_plate_pipeline_timepoint_nan_without_csv` — no CSV → NaN at top-level + every plant row.
- [ ] 7.5 Update `test_multiple_dicot_plate_pipeline_csv_output` to assert the new column order (positions 0-7 metadata, 8+ traits).
- [ ] 7.6 Run `uv run pytest tests/test_multiple_dicot_plate_pipeline.py -k "sample_uid or timepoint or csv" -x` — 4 new tests FAIL; updated test 7.5 FAILS.

## 8. Implement plate pipeline emission

- [ ] 8.1 Modify `compute_plate_traits` in `trait_pipelines.py` to add `sample_uid` + `timepoint` to the top-level result dict.
- [ ] 8.2 Modify `_build_plant_row` to add `sample_uid` + `timepoint` to every plant-row dict.
- [ ] 8.3 Modify `_build_plate_dataframe` `meta_cols` list to insert `sample_uid` at position 1 and `timepoint` at position 2.
- [ ] 8.4 Run `uv run pytest tests/test_multiple_dicot_plate_pipeline.py -x` — all 20+ tests pass (20 existing + 4 new).
- [ ] 8.5 Run `uv run pytest tests/ -x` — full suite green (~449+ tests).

## 9. File follow-up issues BEFORE PR merges

- [ ] 9.1 File issue: "Emit `sample_uid` + `timepoint` in remaining pipelines (follow-up of #169)". Covers DicotPipeline, MultipleDicotPipeline, MultiplePrimaryRootPipeline, PrimaryRootPipeline, LateralRootPipeline, YoungerMonocotPipeline, OlderMonocotPipeline. Related to #159 and #170.
- [ ] 9.2 Post a comment on #169 linking the OpenSpec change + this PR.
- [ ] 9.3 Verification gate: `gh issue list --state open --search "#169 in:body" --limit 20 --json number,title` — confirm the follow-up issue is present.
- [ ] 9.4 Copy follow-up issue number into PR body under `## Follow-up issues filed during this PR`.

## 10. Pre-merge validation

- [ ] 10.1 Run `openspec validate add-sample-uid-timepoint-metadata --strict` — passes.
- [ ] 10.2 Run `uv run black --check sleap_roots/ tests/` — clean.
- [ ] 10.3 Run `uv run pydocstyle --convention=google sleap_roots/` — clean.
- [ ] 10.4 Run `uv run pytest tests/ -x` — full suite passes.
- [ ] 10.5 Invoke `/review-pr` on the branch before opening the PR.
- [ ] 10.6 Open PR; body references issue #169, links the follow-up from 9.1, includes `## TDD evidence` from tasks 1.7 / 3.6 / 5.12 / 7.6.
