# Tasks: Add Series metadata layer (sample_uid, timepoint, get_metadata)

**Source of truth for architecture**: `docs/superpowers/specs/2026-04-23-timelapse-diff-and-tip-kinematics-design.md` § Workstream 1 (PR #171). Implementation plan with task-level detail: `docs/superpowers/plans/2026-04-24-metadata-layer.md`.

**Workflow guarantees**:

- Strict TDD: each subsection writes failing tests BEFORE implementation. Test-file-structure changes (e.g. extending `_build_synthetic_slp` with a new kwarg) land atomically with the tests that need them.
- All commands use `uv run` (`uv run pytest`, `uv run black`, `uv run pydocstyle`).
- No implementation is merged until every follow-up issue in section 12 is filed and linked in the PR body.

## 1. Write failing tests for `Series.sample_uid` + `__attrs_post_init__` defaulting (TDD red phase 1)

These tests come FIRST — the sample_uid attrs field is a prerequisite for `get_metadata` (Section 3), so its tests must be authored before any sample_uid/get_metadata implementation. This fixes the TDD-ordering violation flagged in openspec-review round 1.

- [x] 1.1 Add `test_series_sample_uid_defaults_to_series_name` — `Series.load(series_name="plant1")` without `sample_uid` kwarg → `series.sample_uid == "plant1"`.
- [x] 1.2 Add `test_series_sample_uid_explicit_kwarg` — `Series.load(series_name="plant1_day0", sample_uid="plant1")` → `series.sample_uid == "plant1"`.
- [x] 1.3 Add `test_series_sample_uid_empty_string_falls_through` — `Series.load(..., sample_uid="")` → `sample_uid == series_name` (empty string treated as "not set").
- [x] 1.4 Add `test_series_sample_uid_shared_across_series` — two Series with distinct series_name, same sample_uid kwarg.
- [x] 1.5 Add `test_series_sample_uid_direct_construction_defaults` — `Series(series_name="test_video")` (bypass `Series.load`) → `series.sample_uid == "test_video"` (populated by `__attrs_post_init__`).
- [x] 1.6 Add `test_series_sample_uid_str_coercion` — `Series(series_name="x", sample_uid=1002)` → `series.sample_uid == "1002"` AND `isinstance(series.sample_uid, str) is True`. Rationale: CSV `plant_qr_code` columns may infer as int, but `sample_uid` MUST be str so `df["plant_qr_code"].astype(str) == self.sample_uid` semantics are predictable upstream of any future CSV-dtype follow-up.
- [x] 1.7 Run `uv run pytest tests/test_series.py -k sample_uid -x` — confirm all 6 FAIL with `TypeError` (unknown kwarg) or `AttributeError` (field doesn't exist).

## 2. Implement `Series.sample_uid` attrs field + `Series.load` kwarg + `__attrs_post_init__`

- [x] 2.1 Inspect `sleap_roots/series.py` for an existing `__attrs_post_init__`. **Critical**: if one exists (likely tied to `expected_count`/`group` initialization), the new logic MUST be appended to that method, NOT replace it. Replacing breaks initialization invariants the existing properties depend on.
- [x] 2.2 Add `sample_uid: Optional[str] = None` attrs field to `Series`. Add (or extend) `__attrs_post_init__(self)` so it: (a) sets `self.sample_uid = self.series_name` when `self.sample_uid` is `None` or empty string, then (b) coerces to str via `self.sample_uid = str(self.sample_uid)` (handles int/numpy-int kwarg). Also initialize `self._warned_missing_plant_id_column = False` here for the get_metadata one-shot dedup (Section 4).
- [x] 2.3 Add `sample_uid: Optional[str] = None` kwarg to `Series.load`. In the `return cls(...)` call, pass `sample_uid=sample_uid`. The `__attrs_post_init__` handles defaulting + coercion.
- [x] 2.4 Run `uv run pytest tests/test_series.py -k sample_uid -v` — all 6 pass.
- [x] 2.5 Run `uv run pytest tests/ -x` — full suite green (no regressions introduced by the new field).

## 3. Write failing tests for `Series.get_metadata` (TDD red phase 2)

- [x] 3.1 Add `test_series_get_metadata_no_csv` — no csv_path → NaN.
- [x] 3.2 Add `test_series_get_metadata_missing_column` — CSV exists but lacks requested column → NaN.
- [x] 3.3 Add `test_series_get_metadata_no_matching_row` — CSV has column but no row matches sample_uid → NaN.
- [x] 3.4 Add `test_series_get_metadata_matches_row` — CSV has matching row → returns column value.
- [x] 3.5 Add `test_series_get_metadata_plant_id_composite_lookup` — CSV has `plant_id` column, composite lookup returns correct row.
- [x] 3.6 Add `test_series_get_metadata_plant_id_ignored_when_no_column_emits_warning` — CSV has no `plant_id` column, `plant_id=99` silently ignored AND `caplog` captures a WARNING record from `sleap_roots.series`. Second call on same Series: NO second warning (one-shot dedup).
- [x] 3.7 Add `test_series_get_metadata_plant_id_none_equivalent_to_omitted` — explicit `plant_id=None` returns same value as omitted argument.
- [x] 3.8 Add `test_series_get_metadata_multiple_matches_first_row` — CSV has 2 rows with same `plant_qr_code` (no `plant_id` column), `get_metadata` returns first row's value (`.iloc[0]` semantics).
- [x] 3.9 Run `uv run pytest tests/test_series.py -k get_metadata -x` — all 8 FAIL with `AttributeError: 'Series' object has no attribute 'get_metadata'`.

## 4. Implement `Series.get_metadata` + refactor existing properties

- [x] 4.1 Add module-level `logger = logging.getLogger(__name__)` to `sleap_roots/series.py` (if not already present).
- [x] 4.2 Add `get_metadata(self, column, plant_id=None)` method. The `_warned_missing_plant_id_column` flag is initialized in `__attrs_post_init__` (Section 2.2) — DO NOT declare it as an `attrs.field(...)` (it's runtime-only mutation state, not user-settable construction state) and DO NOT use a plain class attribute in a slotted attrs class (slotted classes forbid undeclared instance attributes; the assignment in `__attrs_post_init__` works because attrs adds the slot for any name set in post-init when `slots=True`-equivalent semantics are in play; verify with `uv run python -c "import attrs; @attrs.define\\nclass C: x: int = 0\\nc = C(); c.y = 1"` — should NOT raise; if your attrs version DOES enforce slots strictly, add `_warned_missing_plant_id_column: bool = attrs.field(default=False, init=False, repr=False)` to the class definition instead).
- [x] 4.3 Refactor `expected_count`, `group`, `qc_fail` as thin wrappers around `get_metadata`. **Preserve the existing `print(...)` calls** in the wrappers (not inside `get_metadata`) so `test_expected_count_error`'s stdout assertion stays green. Alternative: remove the print and update `test_expected_count_error` — choose the preserve path for minimum disruption.
- [x] 4.4 Run `uv run pytest tests/test_series.py -k get_metadata -v` — all 8 pass.
- [x] 4.5 Run `uv run pytest tests/test_series.py -x` — existing tests still pass (specifically `test_expected_count`, `test_qc_cylinder`, `test_expected_count_error` which rely on stdout/return values).
- [x] 4.6 Run `uv run pytest tests/ -x` — full suite green.

## 5. Write failing tests for `Series.timepoint` (TDD red phase 3)

- [x] 5.1 Add `test_series_timepoint_from_csv_numeric` — CSV with `timepoint=2` (int column) → `series.timepoint == 2.0`, `isinstance(series.timepoint, float)` True.
- [x] 5.2 Add `test_series_timepoint_no_csv` → `np.isnan(series.timepoint)` True.
- [x] 5.3 Add `test_series_timepoint_string_float_parses` — CSV with `timepoint="3.5"` (string column) → `series.timepoint == 3.5`.
- [x] 5.4 Add `test_series_timepoint_raises_on_non_numeric` — CSV with `timepoint="2024-03-15"` → `ValueError` with message containing series_name + `"timepoint"` + `"2024-03-15"`.
- [x] 5.5 Run `uv run pytest tests/test_series.py -k timepoint -x` — all 4 FAIL with `AttributeError`.

## 6. Implement `Series.timepoint` with numeric coercion

- [x] 6.1 Add `timepoint` property. Implementation: `value = self.get_metadata("timepoint"); return float(value) if not pd.isna(value) else np.nan`. Wrap the `float(value)` call in try/except — if `ValueError`, raise with a descriptive message per spec.
- [x] 6.2 Run `uv run pytest tests/test_series.py -k timepoint -v` — all 4 pass.
- [x] 6.3 Run `uv run pytest tests/ -x` — full suite green.

## 7. Write failing tests for `sleap_roots/metadata.py` (TDD red phase 4)

- [x] 7.1 Create `tests/test_metadata.py` with tests 7.2-7.13.
- [x] 7.2 Add `test_build_metadata_csv_canonical_column_order` — verify canonical order.
- [x] 7.3 Add `test_build_metadata_csv_omits_unused_columns` — canonical columns absent in rows are omitted.
- [x] 7.4 Add `test_build_metadata_csv_raises_on_missing_plant_qr_code` — `ValueError` with "plant_qr_code" in message.
- [x] 7.5 Add `test_build_metadata_csv_returns_path` — returns the written Path, file exists.
- [x] 7.6 Add `test_build_metadata_csv_accepts_str_path` — accepts str paths.
- [x] 7.7 Add `test_build_metadata_csv_extras_sorted` — non-canonical columns appear sorted at end.
- [x] 7.8 Add `test_build_metadata_csv_overwrites_existing` — writes over an existing file without error.
- [x] 7.9 Add `test_infer_timepoints_from_filenames_named_groups` — correct dict output.
- [x] 7.10 Add `test_infer_timepoints_from_filenames_missing_named_groups` — `ValueError` with `series_name` / `timepoint` in message.
- [x] 7.11 Add `test_infer_timepoints_from_filenames_skips_non_matches_with_warning` — garbage stems skipped AND `caplog` (logger `sleap_roots.metadata`, WARNING level) contains a record mentioning the skipped path.
- [x] 7.12 Add `test_infer_timepoints_from_filenames_skips_non_numeric_with_warning` — pattern matches but `timepoint` group is `"abc"` (or empty) → path skipped, `caplog` records WARNING from `sleap_roots.metadata` mentioning the path AND the float-cast failure reason ("could not convert" or similar). Verifies float-cast failure path is logged separately from pattern mismatch.
- [x] 7.13 Add `test_infer_timepoints_from_filenames_casts_to_float` — integer timepoint → float output.
- [x] 7.14 Run `uv run pytest tests/test_metadata.py -x` — all FAIL with `ImportError`.

## 8. Implement `sleap_roots/metadata.py`

- [x] 8.1 Create `sleap_roots/metadata.py` with module-level `logger = logging.getLogger(__name__)`, `build_metadata_csv`, and `infer_timepoints_from_filenames`. `infer_timepoints_from_filenames` MUST call `logger.warning(...)` for each skipped path (pattern mismatch OR float-cast failure — separate log messages with distinct reasons).
- [x] 8.2 Add re-exports to `sleap_roots/__init__.py`.
- [x] 8.3 Run `uv run pytest tests/test_metadata.py -v` — all 12 pass.
- [x] 8.4 Run `uv run pytest tests/ -x` — full suite green.

## 9. Extend the `_build_synthetic_slp` test helper to accept `sample_uid` + `csv_text` with timepoint

This is a prerequisite for Section 10's plate-pipeline tests. Extending the helper atomically (tests + helper change in one commit) keeps the TDD discipline intact: we don't create tests that fail on HELPER signature issues before the signature lands.

- [x] 9.1 Locate `_build_synthetic_slp` in `tests/test_multiple_dicot_plate_pipeline.py` (around line 66). Current signature is `(tmp_path, series_name, primary_pts_per_frame, lateral_pts_per_frame, csv_content=None)`.
- [x] 9.2 Add `sample_uid: Optional[str] = None` kwarg to `_build_synthetic_slp`, pass-through to `Series.load`.
- [x] 9.3 Locate the `_plate_csv` helper (around line 353). Add a sibling helper `_plate_csv_with_timepoint(series_name, expected_count, genotype, timepoint, qc_cylinder=0)` that emits a CSV text with a `timepoint` column. Call the existing `_plate_csv` internally if helpful (DRY).
- [x] 9.4 Run the existing plate-pipeline tests to verify the `_build_synthetic_slp` kwarg addition doesn't break anything: `uv run pytest tests/test_multiple_dicot_plate_pipeline.py -x`. Expected: all existing tests still pass.

## 10. Write failing tests for plate pipeline `sample_uid`/`timepoint` emission + schema_version=2 + units["time"] (TDD red phase 5)

- [x] 10.1 Add `test_multiple_dicot_plate_pipeline_emits_sample_uid_and_timepoint_top_level` — top-level dict has `sample_uid` AND `timepoint` keys with expected values.
- [x] 10.2 Add `test_multiple_dicot_plate_pipeline_emits_sample_uid_and_timepoint_per_plant` — every plant dict has both keys.
- [x] 10.3 Add `test_multiple_dicot_plate_pipeline_csv_column_positions` — `list(df.columns)[0:8] == ["series", "sample_uid", "timepoint", "frame", "plant_id", "primary_sleap_idx", "expected_count", "detected_count"]`.
- [x] 10.4 Add `test_multiple_dicot_plate_pipeline_timepoint_nan_without_csv` — no CSV → top-level NaN + every plant NaN + `sample_uid` defaults to series_name.
- [x] 10.5 Add `test_multiple_dicot_plate_pipeline_schema_version_bumped_to_2` — `result["schema_version"] == 2`.
- [x] 10.6 Add `test_multiple_dicot_plate_pipeline_units_has_time_family` — `result["units"]["time"] == "unspecified"` (default; follow-up work will allow configuration).
- [x] 10.7 Add `test_multiple_dicot_plate_pipeline_unspecified_time_emits_warning_when_timepoint_non_nan` — Series with non-NaN `timepoint` (CSV provides `timepoint=3`) → exactly ONE WARNING record from logger `sleap_roots.trait_pipelines` mentioning `"unspecified"` and `"timepoint"` and the series_name. Second call: same one-shot semantics (verify warning count via `caplog`).
- [x] 10.8 Add `test_multiple_dicot_plate_pipeline_unspecified_time_no_warning_when_timepoint_nan` — Series without CSV → `timepoint` is NaN → NO warning emitted (NaN is not informative; no point reminding user about unit).
- [x] 10.9 **Update** existing test `test_multiple_dicot_plate_pipeline_csv_output` — change `list(df.columns)[0:6]` assertion to `list(df.columns)[0:8]` with the new 8-element list. Change `list(df.columns)[6:]` to `list(df.columns)[8:]`. Change `set(df.columns[6:])` to `set(df.columns[8:])`.
- [x] 10.10 **Update** existing test `test_multiple_dicot_plate_pipeline_csv_missing_expected_count` — add assertions that `df.loc[0, "sample_uid"] == df.loc[0, "series"]` (sample_uid defaults to series_name when no kwarg) AND `pd.isna(df.loc[0, "timepoint"])` (no CSV → NaN timepoint). Also update column-count assertion if the test counts metadata columns.
- [x] 10.11 **Update** existing test `test_multiple_dicot_plate_pipeline_json_output` — change the top-level key-set assertion from `{"schema_version", "units", "series", "group", "qc_fail", "expected_count", "plants"}` to `{"schema_version", "units", "series", "sample_uid", "timepoint", "group", "qc_fail", "expected_count", "plants"}` (8 keys → 9 keys, not counting `schema_version`/`units`; total 9 top-level keys). Change `result["schema_version"] == 1` to `== 2`. Add `"time": "unspecified"` to the units dict assertion.
- [x] 10.12 **Update** existing test `test_compute_batch_plate_traits` — per-series dicts now have 9 top-level keys; update the key-set assertion accordingly. Update `schema_version` assertion to 2. Add assertions that each per-series dict has `units["time"] == "unspecified"`.
- [x] 10.13 **Update** existing test `test_multiple_dicot_plate_pipeline_zero_frames` (around line 546 of `tests/test_multiple_dicot_plate_pipeline.py`) — currently asserts `result["schema_version"] == 1`; change to `== 2`. Also add assertions for the new top-level keys (`sample_uid` defaults to series_name, `timepoint` is NaN since no CSV with frames). Without this update, the empty-Series path silently ships with the wrong schema_version assertion.
- [x] 10.14 **Update** existing test `test_multiple_dicot_plate_pipeline_json_rfc8259_valid_with_nested_nan` if it asserts exact top-level keys — verify and update if needed. Add an assertion that a NaN `timepoint` round-trips through the JSON as `null` (not `NaN`) AND that `loaded["timepoint"] is None` when the source Series has no CSV.
- [x] 10.15 Run `uv run pytest tests/test_multiple_dicot_plate_pipeline.py -k "sample_uid or timepoint or csv_column_positions or schema_version or units_has_time or unspecified_time or csv_output or csv_missing_expected_count or json_output or compute_batch or rfc8259 or zero_frames" -x` — 8 new tests FAIL (`KeyError`, `AssertionError`). 6 updated tests FAIL (assertion mismatch).

## 11. Implement plate pipeline emission

- [x] 11.1 Add `"time": "unspecified"` to `_PLATE_UNITS` in `sleap_roots/trait_pipelines.py`.
- [x] 11.2 Update `compute_plate_traits` result-dict initialization: change `"schema_version": 1` → `"schema_version": 2`, and resolve `sample_uid_resolved = str(series.sample_uid)` and `timepoint_resolved = series.timepoint` ONCE near the top of the method (single property access). Add `"sample_uid": sample_uid_resolved` and `"timepoint": timepoint_resolved` keys to the top-level result dict.
- [x] 11.3 **CRITICAL — performance contract**: thread `sample_uid_resolved` and `timepoint_resolved` into `_build_plant_row` as **explicit arguments** (NOT computed inside via `series.sample_uid` / `series.timepoint`). At plate-timelapse scale (10 series × 100 frames × 6 plants = 6000 rows), per-plant property accesses each trigger a CSV re-read absent caching — so 6000 redundant reads. Compute-once-pass-down is the contract spec'd in Requirement "compute_plate_traits SHALL emit a per-series dict ...". Update `_build_plant_row` signature to accept `sample_uid: str` and `timepoint: float`; populate `row_dict["sample_uid"] = sample_uid` and `row_dict["timepoint"] = timepoint`.
- [x] 11.4 **CRITICAL — DataFrame columns**: update `_build_plate_dataframe`'s row-dict construction (around line 3213-3220 in `trait_pipelines.py`) to explicitly include `"sample_uid": plant["sample_uid"]` and `"timepoint": plant["timepoint"]`. The `meta_cols` list alone does not make the columns appear in the DataFrame — the row dict must contain them.
- [x] 11.5 Update `meta_cols` list in `_build_plate_dataframe` from 6 to 8 entries: `["series", "sample_uid", "timepoint", "frame", "plant_id", "primary_sleap_idx", "expected_count", "detected_count"]`.
- [x] 11.6 Add the one-shot `units["time"] == "unspecified"` warning in `compute_plate_traits`: when the resolved `timepoint_resolved` is non-NaN AND the units dict has `"time": "unspecified"`, emit `logger.warning(...)` ONCE per call (use a local boolean to dedupe within the method scope; the spec scenario asserts exactly one warning per call). Reference Issue #169 / follow-up §12.5 in the warning message.
- [x] 11.7 Run `uv run pytest tests/test_multiple_dicot_plate_pipeline.py -x` — all 20+ tests pass (existing 20 + 8 new + 6 updated).
- [x] 11.8 Run `uv run pytest tests/ -x` — full suite green.

## 12. File follow-up issues BEFORE PR merges

- [x] 12.1 File issue: "Emit `sample_uid` + `timepoint` in remaining pipelines (follow-up of #169)". Filed as #173.
- [x] 12.2 File issue: "Coerce `plant_qr_code` CSV column to string dtype on read". Filed as #174.
- [x] 12.3 File issue: "Add `strict=True` mode to `Series.get_metadata` that raises on missing rows". Filed as #175.
- [x] 12.4 File issue: "Add lazy `_metadata_df` cache to Series to avoid per-access CSV reads". Filed as #176.
- [x] 12.5 File issue: "Populate `_PLATE_UNITS['time']` from a pipeline kwarg or CSV column". Filed as #177.
- [x] 12.6 Post a comment on #169 linking this PR and the 5 filed follow-ups (#173, #174, #175, #176, #177).
- [x] 12.7 Post a comment on #170 explicitly noting that only plate-intra-series and tracked-tip cases are unblocked by this PR; cylinder-inter-series is blocked on #173.
- [x] 12.8 Post a comment on #163 clarifying the implementation order — this PR (Workstream 1) lands first with `plant_qr_code` as the CSV lookup key; #163 then adds the `sample_uid`-column fallback.
- [x] 12.9 Verification gate confirmed: `gh issue list --search "#169" --limit 20` shows all 5 follow-ups (#173, #174, #175, #176, #177) present and open.
- [ ] 12.10 Copy the 5 issue numbers into PR body under `## Follow-up issues filed during this PR` (deferred until PR is opened).

## 13. Pre-merge validation

- [x] 13.1 Run `openspec validate add-sample-uid-timepoint-metadata --strict` — passes.
- [x] 13.2 Run `uv run black --check sleap_roots/ tests/` — clean.
- [x] 13.3 Run `uv run pydocstyle --convention=google sleap_roots/` — clean.
- [x] 13.4 Run `uv run pytest tests/ -x` — full suite passes.
- [x] 13.5 Invoke `/review-pr` on the branch before opening the PR.
- [x] 13.6 Open PR; body references #169, links the 5 follow-ups from 12.1-12.5, includes `## TDD evidence` from tasks 1.7 / 3.9 / 5.5 / 7.14 / 10.15.
