## Why

The existing CSV-metadata pattern on `Series` (`expected_count`, `group`, `qc_fail`) hardcodes three property-specific lookups against the `plant_qr_code` column. Three new concerns from the timelapse design (issue #169, Workstream 1 of the 2026-04-23 design) need the same lookup mechanism:

1. `sample_uid` — a cross-scan stable identity (the semantic rename of the role `plant_qr_code` has been implicitly filling). Not every workflow needs this; when unset it defaults to `series_name`.
2. `timepoint` — a numeric time-axis value needed for cross-series diffs in `TimeDiffPipeline` (Workstream 3 of the same design).
3. A generic `get_metadata(column, plant_id=None)` accessor so future properties don't each need their own lookup boilerplate.

Without this layer, Workstream 2 (TrackedTipPipeline, issue #129) and Workstream 3 (TimeDiffPipeline, issue #170) have no clean way to consume per-timepoint metadata.

## What Changes

- **NEW** `Series.load(..., sample_uid: Optional[str] = None)` kwarg. Defaults to `series_name` when unset — preserves today's behavior for every existing workflow.
- **NEW** `Series.sample_uid: str` attribute (attrs field), initialized from the kwarg or `series_name`. `__attrs_post_init__` additionally coerces non-string values via `str(...)` so `df["plant_qr_code"] == self.sample_uid` semantics are predictable when callers pass int kwargs.
- **NEW** `Series.get_metadata(column: str, plant_id: Optional[int] = None) -> Any` method. Looks up `df[df["plant_qr_code"] == self.sample_uid]`. If `plant_id` is given AND the CSV has a `plant_id` column, composite lookup returns the `(sample_uid, plant_id)` row. If `plant_id` is given but the CSV has no `plant_id` column, the argument is silently ignored and sample-uid-only lookup is used. Returns `np.nan` when the column is missing or no row matches.
- **NEW** `Series.timepoint: float` property — calls `self.get_metadata("timepoint")`, returns `np.nan` for the missing-CSV / missing-column / no-row paths, coerces numeric values (int, numpy ints, floats, float-castable strings like `"3.5"`) to `float`, and raises `ValueError` with a descriptive message naming the series + column + raw value when a matching row contains a non-numeric string (e.g. a date `"2024-03-15"`). Failing loudly at the metadata layer is intentional — silent-NaN coercion would mask wrong inputs and let downstream `TimeDiffPipeline` arithmetic produce nonsensical deltas.
- **REFACTORED** `Series.expected_count`, `Series.group`, `Series.qc_fail` become thin wrappers around `get_metadata()`. Observable behavior unchanged.
- **NEW** `sleap_roots/metadata.py` module with two pure-function helpers:
  - `build_metadata_csv(rows: List[Dict], path: Union[str, Path]) -> Path`
  - `infer_timepoints_from_filenames(slp_paths: List[Path], pattern: str) -> Dict[str, float]`
- **NEW** top-level re-exports of both helpers from `sleap_roots.__init__`.
- **MODIFIED** `MultipleDicotPlatePipeline.compute_plate_traits` output:
  - Top-level dict gains `sample_uid` and `timepoint` keys.
  - Every per-plant row gains `sample_uid` and `timepoint` keys.
  - CSV emits `sample_uid` and `timepoint` columns right after `series` (positions 1 and 2), shifting the remaining metadata columns by 2.
  - **Performance contract**: `sample_uid` and `timepoint` are resolved ONCE in `compute_plate_traits` (single property access on `series`) and threaded into `_build_plant_row` as explicit arguments. Per-plant rows MUST NOT call `series.sample_uid` / `series.timepoint` (or `series.get_metadata`) inside any per-frame or per-plant loop — at plate-timelapse scale (10 series × 100 frames × 6 plants = 6000 rows), per-row property access would trigger 6000 redundant CSV reads. Compute-once-pass-down is spec'd as a normative requirement.

- **MODIFIED** existing `Series(...)` direct construction now defaults `sample_uid` via `__attrs_post_init__` — necessary because existing test fixtures and production call sites construct `Series(...)` directly (bypassing `Series.load`). Without this, `sample_uid=None` → lookup fails → `expected_count` etc. silently return NaN where they used to return values.
- **MODIFIED** `MultipleDicotPlatePipeline` capability spec — CSV column order goes from 6 metadata columns to 8, top-level dict gains 2 keys, per-plant rows gain 2 keys. `schema_version` bumps to **2** (positional shift is non-additive). See `specs/multiple-dicot-plate-pipeline/spec.md` in this change for the full MODIFIED requirements.
- **NEW** `"time"` entry in `_PLATE_UNITS` so the plate JSON carries the unit for `timepoint` (default `"unspecified"`; user can override via pipeline kwarg or CSV column — see follow-up issue below). Without this, two collaborators recording `timepoint=3` in days vs seconds silently produce incompatible data. While `units["time"] == "unspecified"` and a non-NaN `timepoint` is emitted, `compute_plate_traits` emits a one-shot WARNING per call to surface the missing-unit hazard at runtime — silent "unspecified" is the failure mode the warning is designed to prevent.

**Partially breaking for CSV positional readers.** The plate CSV column order shifts by 2 positions (new columns at 1 and 2, old columns slide right). `schema_version` bumps to 2. Consumers who read CSV by column NAME continue to work unchanged. Consumers who read CSV by column POSITION will need to update — PR #165 is days old, no known downstream scripts exist. CSV column rename (`plant_qr_code` → `sample_uid`, etc.) is explicitly deferred to #163.

**`schema_version` is JSON-only**, by design. The in-memory dict and the JSON file both carry `schema_version=2`, but the per-plant CSV does not — adding the version as a column would repeat the value on every row (wasteful) and as a header comment line would break `pandas.read_csv` defaults. The migration path for CSV consumers is therefore to read by column NAME (`df["primary_length"]`), which is the standard pandas pattern; positional reads (`df.iloc[:, 6]`) cannot programmatically detect the version change and will silently misread under the new layout. Documented in `compute_plate_traits` docstring and the `multiple-dicot-plate-pipeline` capability spec.

**`Series.expected_count` / `.group` / `.qc_fail` observable behavior preserved byte-for-byte for the default-kwarg path** (i.e. when `Series.load` / `Series(...)` is called without an explicit `sample_uid` kwarg — which is the only path any existing code uses). If a caller passes an explicit `sample_uid` kwarg, the lookup key changes from `series_name` to `sample_uid` — this is the desired semantic for cross-scan identity but IS a behavior change. Documented in the spec's "sample_uid" requirement.

## Impact

- **Affected specs**:
  - **NEW capability** `series-metadata` (ADDED requirements for `sample_uid`, `timepoint`, `get_metadata`, wrapper-refactor regression, CSV-builder helpers). Pipeline emission is explicitly NOT a requirement of this capability.
  - **MODIFIED** `multiple-dicot-plate-pipeline` capability (CSV column order, top-level dict shape, per-plant row shape, `schema_version` bump, `units` dict gains a `time` entry).
- **Affected code**:
  - `sleap_roots/series.py` — new kwarg, new attrs field, new `__attrs_post_init__`, new method, new property, refactor of three existing properties. New `logger = logging.getLogger(__name__)` at module top for the `plant_id`-ignored warning.
  - `sleap_roots/metadata.py` — new module with `build_metadata_csv` + `infer_timepoints_from_filenames` (both with `logger.warning` on skip).
  - `sleap_roots/__init__.py` — re-export additions.
  - `sleap_roots/trait_pipelines.py` — plate pipeline output additions: `_PLATE_UNITS` gains `"time"` key, `compute_plate_traits` result dict gains `sample_uid` + `timepoint` + `schema_version=2`, `_build_plant_row` gains two keys, `_build_plate_dataframe` builds row dicts that include `sample_uid` + `timepoint`, `meta_cols` grows by 2. ~30 lines total.
  - `tests/test_series.py` — new test coverage for `get_metadata`, `sample_uid`, `timepoint`; **existing tests** `test_expected_count_error` (stdout assertion no longer valid because refactored `get_metadata` drops the `print`) and `test_expected_count`/`test_qc_cylinder` (fixtures use direct `Series(...)` which now defaults `sample_uid` via `__attrs_post_init__`) are **updated in-place** — NOT rewritten.
  - `tests/test_metadata.py` — new test file.
  - `tests/test_multiple_dicot_plate_pipeline.py` — new emission tests; **4 existing tests updated**: `test_multiple_dicot_plate_pipeline_csv_output` (column order), `test_multiple_dicot_plate_pipeline_json_output` (top-level key set), `test_compute_batch_plate_traits` (per-series dict key set), `test_multiple_dicot_plate_pipeline_json_rfc8259_valid_with_nested_nan` (potentially affected if it asserts exact key sets).
- **Source design doc**: `docs/superpowers/specs/2026-04-23-timelapse-diff-and-tip-kinematics-design.md` § Workstream 1. PR #171.
- **Dependencies**:
  - **Fully unblocks #129** (TrackedTipPipeline consumes `Series.sample_uid` / `Series.timepoint` directly, not via inner-pipeline output).
  - **Partially unblocks #170** (TimeDiffPipeline): the plate-intra-series case (`time_col="frame"`) and the tracked-tip case (via #129) are unblocked. The **cylinder-inter-series case** (`time_col="timepoint"` with identity `["sample_uid"]`) is NOT fully unblocked here because `MultipleDicotPipeline`, `DicotPipeline`, `MultiplePrimaryRootPipeline`, etc. do not yet emit `sample_uid` + `timepoint`. That's tracked as a dedicated follow-up issue filed during this PR (see `tasks.md` § 9.1). #170's proposal can land against the plate-intra-series case only; cylinder-case tests wait for the follow-up.

## Known limitations / follow-ups tracked during this PR

- **CSV `plant_qr_code` column dtype coercion**: if a user writes a CSV where `plant_qr_code` values are pure integers (e.g. `1002`), pandas infers `int64` dtype and `df["plant_qr_code"] == self.sample_uid` (str) silently matches no rows. Pre-existing behavior (inherited by the `get_metadata` refactor); not worsened. Follow-up issue: "Coerce `plant_qr_code` CSV column to string dtype on read to prevent silent no-match on pure-numeric QR codes."
- **NaN-in-cell vs no-matching-row collapse**: both return `np.nan`, losing the distinction between "row exists, value unrecorded" and "no row for this sample". Follow-up issue: "Add a `strict=True` mode to `Series.get_metadata` that raises on missing rows."
- **CSV read on every `get_metadata` call**: no caching. At plate-timelapse scale (1000s of property accesses per series) this is measurable. Follow-up issue: "Add lazy `_metadata_df` cache on Series."
- **Emit `sample_uid` + `timepoint` in remaining pipelines** (cylinder, single-plant, lateral, crown, tracked-tip): see `tasks.md` § 9.1.
- **`_PLATE_UNITS["time"]` population mechanism**: this PR ships with default `"unspecified"`. A future follow-up adds a pipeline kwarg (`time_unit="days"`) or a `timepoint_unit` CSV column to populate it explicitly.
