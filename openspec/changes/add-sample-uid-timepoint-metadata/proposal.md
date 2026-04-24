## Why

The existing CSV-metadata pattern on `Series` (`expected_count`, `group`, `qc_fail`) hardcodes three property-specific lookups against the `plant_qr_code` column. Three new concerns from the timelapse design (issue #169, Workstream 1 of the 2026-04-23 design) need the same lookup mechanism:

1. `sample_uid` — a cross-scan stable identity (the semantic rename of the role `plant_qr_code` has been implicitly filling). Not every workflow needs this; when unset it defaults to `series_name`.
2. `timepoint` — a numeric time-axis value needed for cross-series diffs in `TimeDiffPipeline` (Workstream 3 of the same design).
3. A generic `get_metadata(column, plant_id=None)` accessor so future properties don't each need their own lookup boilerplate.

Without this layer, Workstream 2 (TrackedTipPipeline, issue #129) and Workstream 3 (TimeDiffPipeline, issue #170) have no clean way to consume per-timepoint metadata.

## What Changes

- **NEW** `Series.load(..., sample_uid: Optional[str] = None)` kwarg. Defaults to `series_name` when unset — preserves today's behavior for every existing workflow.
- **NEW** `Series.sample_uid: str` attribute (attrs field), initialized from the kwarg or `series_name`.
- **NEW** `Series.get_metadata(column: str, plant_id: Optional[int] = None) -> Any` method. Looks up `df[df["plant_qr_code"] == self.sample_uid]`. If `plant_id` is given AND the CSV has a `plant_id` column, composite lookup returns the `(sample_uid, plant_id)` row. If `plant_id` is given but the CSV has no `plant_id` column, the argument is silently ignored and sample-uid-only lookup is used. Returns `np.nan` when the column is missing or no row matches.
- **NEW** `Series.timepoint: Union[float, int]` property — thin wrapper `return self.get_metadata("timepoint")`.
- **REFACTORED** `Series.expected_count`, `Series.group`, `Series.qc_fail` become thin wrappers around `get_metadata()`. Observable behavior unchanged.
- **NEW** `sleap_roots/metadata.py` module with two pure-function helpers:
  - `build_metadata_csv(rows: List[Dict], path: Union[str, Path]) -> Path`
  - `infer_timepoints_from_filenames(slp_paths: List[Path], pattern: str) -> Dict[str, float]`
- **NEW** top-level re-exports of both helpers from `sleap_roots.__init__`.
- **MODIFIED** `MultipleDicotPlatePipeline.compute_plate_traits` output:
  - Top-level dict gains `sample_uid` and `timepoint` keys.
  - Every per-plant row gains `sample_uid` and `timepoint` keys.
  - CSV emits `sample_uid` and `timepoint` columns right after `series` (positions 1 and 2), shifting the remaining metadata columns by 2.

**Not breaking.** Existing `Series` API is preserved (`expected_count`, `group`, `qc_fail` keep their signatures and return values). Existing `MultipleDicotPlatePipeline` consumers who read CSV by column NAME continue to work; consumers who read CSV by column POSITION will see columns shifted — the plate pipeline is new enough (PR #165) that no published downstream scripts exist. CSV column rename to `sample_uid` / `timepoint` (from `plant_qr_code` / n/a) is explicitly deferred to #163.

## Impact

- **Affected specs**: new capability `series-metadata` (ADDED requirements for `sample_uid`, `timepoint`, `get_metadata`, CSV-builder helpers, pipeline output emission). No modifications to existing specs.
- **Affected code**:
  - `sleap_roots/series.py` — new kwarg, new method, new property, refactor of three existing properties.
  - `sleap_roots/metadata.py` — new module.
  - `sleap_roots/__init__.py` — re-export additions.
  - `sleap_roots/trait_pipelines.py` — plate pipeline output additions (~10 lines).
  - `tests/test_series.py` — new test coverage; regression coverage for refactored properties.
  - `tests/test_metadata.py` — new test file.
  - `tests/test_multiple_dicot_plate_pipeline.py` — new emission tests; update existing CSV column-order assertions.
- **Source design doc**: `docs/superpowers/specs/2026-04-23-timelapse-diff-and-tip-kinematics-design.md` § Workstream 1. PR #171.
- **Dependencies**: Unblocks #129 (TrackedTipPipeline) and #170 (TimeDiffPipeline).
