# Tasks: Add Series metadata layer (sample_uid, timepoint, get_metadata)

**Source of truth for architecture**: `docs/superpowers/specs/2026-04-23-timelapse-diff-and-tip-kinematics-design.md` § Workstream 1 (PR #171). Implementation plan with task-level detail: `docs/superpowers/plans/2026-04-24-metadata-layer.md`.

**Workflow guarantees**:

- Strict TDD: each subsection writes failing tests BEFORE implementation. Test-file-structure changes (e.g. extending `_build_synthetic_slp` with a new kwarg) land atomically with the tests that need them.
- All commands use `uv run` (`uv run pytest`, `uv run black`, `uv run pydocstyle`).
- No implementation is merged until every follow-up issue in section 9 is filed and linked in the PR body.

## 1. Write failing tests for `Series.sample_uid` + `__attrs_post_init__` defaulting (TDD red phase 1)

These tests come FIRST — the sample_uid attrs field is a prerequisite for `get_metadata` (Section 3), so its tests must be authored before any sample_uid/get_metadata implementation. This fixes the TDD-ordering violation flagged in openspec-review round 1.

- [ ] 1.1 Add `test_series_sample_uid_defaults_to_series_name` — `Series.load(series_name="plant1")` without `sample_uid` kwarg → `series.sample_uid == "plant1"`.
- [ ] 1.2 Add `test_series_sample_uid_explicit_kwarg` — `Series.load(series_name="plant1_day0", sample_uid="plant1")` → `series.sample_uid == "plant1"`.
- [ ] 1.3 Add `test_series_sample_uid_empty_string_falls_through` — `Series.load(..., sample_uid="")` → `sample_uid == series_name` (empty string treated as "not set").
- [ ] 1.4 Add `test_series_sample_uid_shared_across_series` — two Series with distinct series_name, same sample_uid kwarg.
- [ ] 1.5 Add `test_series_sample_uid_direct_construction_defaults` — `Series(series_name="test_video")` (bypass `Series.load`) → `series.sample_uid == "test_video"` (populated by `__attrs_post_init__`).
- [ ] 1.6 Run `uv run pytest tests/test_series.py -k sample_uid -x` — confirm all 5 FAIL with `TypeError` (unknown kwarg) or `AttributeError` (field doesn't exist).

## 2. Implement `Series.sample_uid` attrs field + `Series.load` kwarg + `__attrs_post_init__`

- [ ] 2.1 Add `sample_uid: Optional[str] = None` attrs field to `Series`. Add `__attrs_post_init__(self)` method that sets `self.sample_uid = self.series_name` when `self.sample_uid` is `None` or empty string.
- [ ] 2.2 Add `sample_uid: Optional[str] = None` kwarg to `Series.load`. In the `return cls(...)` call, pass `sample_uid=sample_uid`. The `__attrs_post_init__` handles defaulting.
- [ ] 2.3 Run `uv run pytest tests/test_series.py -k sample_uid -v` — all 5 pass.
- [ ] 2.4 Run `uv run pytest tests/ -x` — full suite green (no regressions introduced by the new field).

## 3. Write failing tests for `Series.get_metadata` (TDD red phase 2)

- [ ] 3.1 Add `test_series_get_metadata_no_csv` — no csv_path → NaN.
- [ ] 3.2 Add `test_series_get_metadata_missing_column` — CSV exists but lacks requested column → NaN.
- [ ] 3.3 Add `test_series_get_metadata_no_matching_row` — CSV has column but no row matches sample_uid → NaN.
- [ ] 3.4 Add `test_series_get_metadata_matches_row` — CSV has matching row → returns column value.
- [ ] 3.5 Add `test_series_get_metadata_plant_id_composite_lookup` — CSV has `plant_id` column, composite lookup returns correct row.
- [ ] 3.6 Add `test_series_get_metadata_plant_id_ignored_when_no_column_emits_warning` — CSV has no `plant_id` column, `plant_id=99` silently ignored AND `caplog` captures a WARNING record from `sleap_roots.series`. Second call on same Series: NO second warning (one-shot dedup).
- [ ] 3.7 Add `test_series_get_metadata_plant_id_none_equivalent_to_omitted` — explicit `plant_id=None` returns same value as omitted argument.
- [ ] 3.8 Add `test_series_get_metadata_multiple_matches_first_row` — CSV has 2 rows with same `plant_qr_code` (no `plant_id` column), `get_metadata` returns first row's value (`.iloc[0]` semantics).
- [ ] 3.9 Run `uv run pytest tests/test_series.py -k get_metadata -x` — all 7 FAIL with `AttributeError: 'Series' object has no attribute 'get_metadata'`.

## 4. Implement `Series.get_metadata` + refactor existing properties

- [ ] 4.1 Add module-level `logger = logging.getLogger(__name__)` to `sleap_roots/series.py` (if not already present).
- [ ] 4.2 Add `get_metadata(self, column, plant_id=None)` method. Include `_warned_missing_plant_id_column: bool` instance flag (set in `__attrs_post_init__` to `False`) for the one-shot warning dedup.
- [ ] 4.3 Refactor `expected_count`, `group`, `qc_fail` as thin wrappers around `get_metadata`. **Preserve the existing `print(...)` calls** in the wrappers (not inside `get_metadata`) so `test_expected_count_error`'s stdout assertion stays green. Alternative: remove the print and update `test_expected_count_error` — choose the preserve path for minimum disruption.
- [ ] 4.4 Run `uv run pytest tests/test_series.py -k get_metadata -v` — all 7 pass.
- [ ] 4.5 Run `uv run pytest tests/test_series.py -x` — existing tests still pass (specifically `test_expected_count`, `test_qc_cylinder`, `test_expected_count_error` which rely on stdout/return values).
- [ ] 4.6 Run `uv run pytest tests/ -x` — full suite green.

## 5. Write failing tests for `Series.timepoint` (TDD red phase 3)

- [ ] 5.1 Add `test_series_timepoint_from_csv_numeric` — CSV with `timepoint=2` (int column) → `series.timepoint == 2.0`, `isinstance(series.timepoint, float)` True.
- [ ] 5.2 Add `test_series_timepoint_no_csv` → `np.isnan(series.timepoint)` True.
- [ ] 5.3 Add `test_series_timepoint_string_float_parses` — CSV with `timepoint="3.5"` (string column) → `series.timepoint == 3.5`.
- [ ] 5.4 Add `test_series_timepoint_raises_on_non_numeric` — CSV with `timepoint="2024-03-15"` → `ValueError` with message containing series_name + `"timepoint"` + `"2024-03-15"`.
- [ ] 5.5 Run `uv run pytest tests/test_series.py -k timepoint -x` — all 4 FAIL with `AttributeError`.

## 6. Implement `Series.timepoint` with numeric coercion

- [ ] 6.1 Add `timepoint` property. Implementation: `value = self.get_metadata("timepoint"); return float(value) if not pd.isna(value) else np.nan`. Wrap the `float(value)` call in try/except — if `ValueError`, raise with a descriptive message per spec.
- [ ] 6.2 Run `uv run pytest tests/test_series.py -k timepoint -v` — all 4 pass.
- [ ] 6.3 Run `uv run pytest tests/ -x` — full suite green.

## 7. Write failing tests for `sleap_roots/metadata.py` (TDD red phase 4)

- [ ] 7.1 Create `tests/test_metadata.py` with tests 7.2-7.12.
- [ ] 7.2 Add `test_build_metadata_csv_canonical_column_order` — verify canonical order.
- [ ] 7.3 Add `test_build_metadata_csv_omits_unused_columns` — canonical columns absent in rows are omitted.
- [ ] 7.4 Add `test_build_metadata_csv_raises_on_missing_plant_qr_code` — `ValueError` with "plant_qr_code" in message.
- [ ] 7.5 Add `test_build_metadata_csv_returns_path` — returns the written Path, file exists.
- [ ] 7.6 Add `test_build_metadata_csv_accepts_str_path` — accepts str paths.
- [ ] 7.7 Add `test_build_metadata_csv_extras_sorted` — non-canonical columns appear sorted at end.
- [ ] 7.8 Add `test_build_metadata_csv_overwrites_existing` — writes over an existing file without error.
- [ ] 7.9 Add `test_infer_timepoints_from_filenames_named_groups` — correct dict output.
- [ ] 7.10 Add `test_infer_timepoints_from_filenames_missing_named_groups` — `ValueError` with `series_name` / `timepoint` in message.
- [ ] 7.11 Add `test_infer_timepoints_from_filenames_skips_non_matches_with_warning` — garbage stems skipped AND `caplog` (logger `sleap_roots.metadata`, WARNING level) contains a record mentioning the skipped path.
- [ ] 7.12 Add `test_infer_timepoints_from_filenames_casts_to_float` — integer timepoint → float output.
- [ ] 7.13 Run `uv run pytest tests/test_metadata.py -x` — all FAIL with `ImportError`.

## 8. Implement `sleap_roots/metadata.py`

- [ ] 8.1 Create `sleap_roots/metadata.py` with module-level `logger = logging.getLogger(__name__)`, `build_metadata_csv`, and `infer_timepoints_from_filenames`. `infer_timepoints_from_filenames` MUST call `logger.warning(...)` for each skipped path (pattern mismatch OR float-cast failure).
- [ ] 8.2 Add re-exports to `sleap_roots/__init__.py`.
- [ ] 8.3 Run `uv run pytest tests/test_metadata.py -v` — all 11 pass.
- [ ] 8.4 Run `uv run pytest tests/ -x` — full suite green.

## 9. Extend the `_build_synthetic_slp` test helper to accept `sample_uid` + `csv_text` with timepoint

This is a prerequisite for Section 10's plate-pipeline tests. Extending the helper atomically (tests + helper change in one commit) keeps the TDD discipline intact: we don't create tests that fail on HELPER signature issues before the signature lands.

- [ ] 9.1 Locate `_build_synthetic_slp` in `tests/test_multiple_dicot_plate_pipeline.py` (around line 66). Current signature is `(tmp_path, series_name, primary_pts_per_frame, lateral_pts_per_frame, csv_content=None)`.
- [ ] 9.2 Add `sample_uid: Optional[str] = None` kwarg to `_build_synthetic_slp`, pass-through to `Series.load`.
- [ ] 9.3 Locate the `_plate_csv` helper (around line 353). Add a sibling helper `_plate_csv_with_timepoint(series_name, expected_count, genotype, timepoint, qc_cylinder=0)` that emits a CSV text with a `timepoint` column. Call the existing `_plate_csv` internally if helpful (DRY).
- [ ] 9.4 Run the existing plate-pipeline tests to verify the `_build_synthetic_slp` kwarg addition doesn't break anything: `uv run pytest tests/test_multiple_dicot_plate_pipeline.py -x`. Expected: all existing tests still pass.

## 10. Write failing tests for plate pipeline `sample_uid`/`timepoint` emission + schema_version=2 + units["time"] (TDD red phase 5)

- [ ] 10.1 Add `test_multiple_dicot_plate_pipeline_emits_sample_uid_and_timepoint_top_level` — top-level dict has `sample_uid` AND `timepoint` keys with expected values.
- [ ] 10.2 Add `test_multiple_dicot_plate_pipeline_emits_sample_uid_and_timepoint_per_plant` — every plant dict has both keys.
- [ ] 10.3 Add `test_multiple_dicot_plate_pipeline_csv_column_positions` — `list(df.columns)[0:8] == ["series", "sample_uid", "timepoint", "frame", "plant_id", "primary_sleap_idx", "expected_count", "detected_count"]`.
- [ ] 10.4 Add `test_multiple_dicot_plate_pipeline_timepoint_nan_without_csv` — no CSV → top-level NaN + every plant NaN + `sample_uid` defaults to series_name.
- [ ] 10.5 Add `test_multiple_dicot_plate_pipeline_schema_version_bumped_to_2` — `result["schema_version"] == 2`.
- [ ] 10.6 Add `test_multiple_dicot_plate_pipeline_units_has_time_family` — `result["units"]["time"] == "unspecified"` (default; follow-up work will allow configuration).
- [ ] 10.7 **Update** existing test `test_multiple_dicot_plate_pipeline_csv_output` — change `list(df.columns)[0:6]` assertion to `list(df.columns)[0:8]` with the new 8-element list. Change `list(df.columns)[6:]` to `list(df.columns)[8:]`. Change `set(df.columns[6:])` to `set(df.columns[8:])`.
- [ ] 10.8 **Update** existing test `test_multiple_dicot_plate_pipeline_json_output` — change the top-level key-set assertion from `{"schema_version", "units", "series", "group", "qc_fail", "expected_count", "plants"}` to `{"schema_version", "units", "series", "sample_uid", "timepoint", "group", "qc_fail", "expected_count", "plants"}`. Change `result["schema_version"] == 1` to `== 2`. Add `"time": "unspecified"` to the units dict assertion.
- [ ] 10.9 **Update** existing test `test_compute_batch_plate_traits` — per-series dicts now have 9 top-level keys; update the key-set assertion accordingly. Update `schema_version` assertion to 2.
- [ ] 10.10 **Update** existing test `test_multiple_dicot_plate_pipeline_json_rfc8259_valid_with_nested_nan` if it asserts exact top-level keys — verify and update if needed. Also add an assertion that a NaN `timepoint` round-trips through the JSON as `null` (not `NaN`).
- [ ] 10.11 Run `uv run pytest tests/test_multiple_dicot_plate_pipeline.py -k "sample_uid or timepoint or csv_column_positions or schema_version or units_has_time or csv_output or json_output or compute_batch or rfc8259" -x` — 6 new tests FAIL (`KeyError`, `AssertionError`). 4 updated tests FAIL (assertion mismatch).

## 11. Implement plate pipeline emission

- [ ] 11.1 Add `"time": "unspecified"` to `_PLATE_UNITS` in `sleap_roots/trait_pipelines.py`.
- [ ] 11.2 Update `compute_plate_traits` result-dict initialization: change `"schema_version": 1` → `"schema_version": 2`, add `"sample_uid": str(series.sample_uid)` and `"timepoint": series.timepoint` keys.
- [ ] 11.3 Thread `sample_uid` and `timepoint` into `_build_plant_row` — either as explicit arguments OR compute inside (use `series.sample_uid` / `series.timepoint` since the `series` object is already a parameter). Return them as keys in the plant-row dict.
- [ ] 11.4 **CRITICAL**: update `_build_plate_dataframe`'s row-dict construction (around line 3213-3220 in `trait_pipelines.py`) to explicitly include `"sample_uid": plant["sample_uid"]` and `"timepoint": plant["timepoint"]`. The `meta_cols` list alone does not make the columns appear in the DataFrame — the row dict must contain them.
- [ ] 11.5 Update `meta_cols` list in `_build_plate_dataframe` from 6 to 8 entries: `["series", "sample_uid", "timepoint", "frame", "plant_id", "primary_sleap_idx", "expected_count", "detected_count"]`.
- [ ] 11.6 Run `uv run pytest tests/test_multiple_dicot_plate_pipeline.py -x` — all 20+ tests pass (existing 20 + 6 new + 4 updated).
- [ ] 11.7 Run `uv run pytest tests/ -x` — full suite green.

## 12. File follow-up issues BEFORE PR merges

- [ ] 12.1 File issue: "Emit `sample_uid` + `timepoint` in remaining pipelines (follow-up of #169)". Covers `DicotPipeline`, `MultipleDicotPipeline`, `MultiplePrimaryRootPipeline`, `PrimaryRootPipeline`, `LateralRootPipeline`, `YoungerMonocotPipeline`, `OlderMonocotPipeline`. Blocks #170's cylinder-inter-series case.
- [ ] 12.2 File issue: "Coerce `plant_qr_code` CSV column to string dtype on read". Documents pre-existing silent no-match behavior on pure-numeric QR codes.
- [ ] 12.3 File issue: "Add `strict=True` mode to `Series.get_metadata` that raises on missing rows". Separates "row missing" from "value NaN".
- [ ] 12.4 File issue: "Add lazy `_metadata_df` cache to Series to avoid per-access CSV reads".
- [ ] 12.5 File issue: "Populate `_PLATE_UNITS['time']` from a pipeline kwarg or CSV column". Removes the `"unspecified"` default.
- [ ] 12.6 Post a comment on #169 linking this PR and the 5 filed follow-ups.
- [ ] 12.7 Post a comment on #170 explicitly noting that only plate-intra-series and tracked-tip cases are unblocked by this PR; cylinder-inter-series is blocked on #12.1's follow-up.
- [ ] 12.8 Post a comment on #163 clarifying the implementation order — this PR (Workstream 1) lands first with `plant_qr_code` as the CSV lookup key; #163 then adds the `sample_uid`-column fallback.
- [ ] 12.9 Verification gate: `gh issue list --state open --search "#169 in:body" --limit 20 --json number,title` — confirm all 5 follow-ups (12.1-12.5) present.
- [ ] 12.10 Copy the 5 issue numbers into PR body under `## Follow-up issues filed during this PR`.

## 13. Pre-merge validation

- [ ] 13.1 Run `openspec validate add-sample-uid-timepoint-metadata --strict` — passes.
- [ ] 13.2 Run `uv run black --check sleap_roots/ tests/` — clean.
- [ ] 13.3 Run `uv run pydocstyle --convention=google sleap_roots/` — clean.
- [ ] 13.4 Run `uv run pytest tests/ -x` — full suite passes.
- [ ] 13.5 Invoke `/review-pr` on the branch before opening the PR.
- [ ] 13.6 Open PR; body references #169, links the 5 follow-ups from 12.1-12.5, includes `## TDD evidence` from tasks 1.6 / 3.9 / 5.5 / 7.13 / 10.11.
