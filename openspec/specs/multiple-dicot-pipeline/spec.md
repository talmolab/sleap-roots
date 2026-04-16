# multiple-dicot-pipeline Specification

## Purpose
TBD - created by archiving change make-expected-count-optional. Update Purpose after archive.
## Requirements
### Requirement: `filter_plants_with_unexpected_ct` MUST accept `None` as "no expected count"

The `filter_plants_with_unexpected_ct` function in `sleap_roots/points.py` SHALL accept `expected_count: Optional[float] = None`. When `expected_count` is `None`, the function MUST return both `primary_pts` and `lateral_pts` unchanged without performing any count check. The `None` path MUST be observationally equivalent to the existing `np.nan` "skip filtering" sentinel: both inputs yield pass-through output, and neither triggers the numeric-mismatch empty-array branch.

#### Scenario: Calling with `expected_count=None` returns inputs unchanged

- **Given** `primary_pts` of shape `(5, 10, 2)` populated with random values
- **And** `lateral_pts` of shape `(5, 10, 2)` populated with random values
- **When** `filter_plants_with_unexpected_ct(primary_pts, lateral_pts, expected_count=None)` is called
- **Then** the returned primary array is element-wise equal to the input `primary_pts`
- **And** the returned lateral array is element-wise equal to the input `lateral_pts`
- **And** no exception is raised

#### Scenario: `None` does not implicitly require matching primary/lateral counts

- **Given** `primary_pts` of shape `(5, 10, 2)` and `lateral_pts` of shape `(3, 10, 2)` — different instance counts
- **When** `filter_plants_with_unexpected_ct(primary_pts, lateral_pts, expected_count=None)` is called
- **Then** the returned primary array has shape `(5, 10, 2)` — all 5 instances preserved
- **And** the returned lateral array has shape `(3, 10, 2)` — all 3 instances preserved
- **And** neither array is the empty `(0, 10, 2)` placeholder

#### Scenario: `expected_count` defaults to `None` when omitted

- **Given** `primary_pts` of shape `(5, 10, 2)` and `lateral_pts` of shape `(5, 10, 2)`
- **When** `filter_plants_with_unexpected_ct(primary_pts, lateral_pts)` is called with exactly two positional arguments
- **Then** the returned arrays are element-wise equal to the inputs
- **And** no `TypeError` is raised for a missing `expected_count` argument

### Requirement: Numeric `expected_count` filter semantics MUST be preserved

This requirement captures the **existing** behavior of `filter_plants_with_unexpected_ct` for numeric `expected_count` values as a regression baseline for the new `multiple-dicot-pipeline` capability. Because no prior capability spec existed, these pre-existing semantics are expressed here as ADDED requirements so the broadened contract (Requirement 1) can coexist with them verifiably. The behavior MUST be preserved byte-for-byte to keep `MultipleDicotPipeline`'s drop-on-mismatch aggregation intact:

- `np.nan` skips filtering and returns inputs unchanged.
- A finite number equal to `round(expected_count) == len(primary_pts)` returns inputs unchanged.
- A finite number with `round(expected_count) != len(primary_pts)` returns empty `(0, n_nodes, 2)` arrays for both primary and lateral.
- Rounding MUST use Python's built-in `round()` (banker's rounding, half-to-even): `round(2.5) == 2`, `round(3.5) == 4`. This rule is pinned to prevent a future refactor (e.g. `int(round(...))` or `math.floor(x + 0.5)`) from silently changing drop/keep decisions on half-integer inputs.
- A non-numeric, non-`None` value (e.g. a string) raises `ValueError`.

#### Scenario: `np.nan` expected_count skips filtering

- **Given** `primary_pts` of shape `(5, 10, 2)`, `lateral_pts` of shape `(5, 10, 2)`, and `expected_count=np.nan`
- **When** `filter_plants_with_unexpected_ct` is called
- **Then** both output arrays are element-wise equal to the inputs

#### Scenario: Matching numeric expected_count passes arrays through

- **Given** `primary_pts` of shape `(5, 10, 2)`, `lateral_pts` of shape `(5, 10, 2)`, and `expected_count=5.0`
- **When** `filter_plants_with_unexpected_ct` is called
- **Then** both output arrays are element-wise equal to the inputs

#### Scenario: Mismatching numeric expected_count produces empty arrays

- **Given** `primary_pts` of shape `(5, 10, 2)`, `lateral_pts` of shape `(5, 10, 2)`, and `expected_count=3.0`
- **When** `filter_plants_with_unexpected_ct` is called
- **Then** the returned primary array has shape `(0, 10, 2)`
- **And** the returned lateral array has shape `(0, 10, 2)`

#### Scenario: Non-numeric, non-None expected_count raises ValueError

- **Given** `primary_pts` of shape `(5, 10, 2)`, `lateral_pts` of shape `(5, 10, 2)`, and `expected_count="not a float"`
- **When** `filter_plants_with_unexpected_ct` is called
- **Then** a `ValueError` is raised

#### Scenario: `expected_count=0` with empty primary_pts passes through

- **Given** `primary_pts` of shape `(0, 10, 2)` (no primary-root instances) and `lateral_pts` of shape `(0, 10, 2)`
- **And** `expected_count=0`
- **When** `filter_plants_with_unexpected_ct` is called
- **Then** the returned primary array has shape `(0, 10, 2)` (pass-through, because `round(0) == 0 == len(primary_pts)`)
- **And** the returned lateral array has shape `(0, 10, 2)`
- **And** no `ValueError` is raised

#### Scenario: `expected_count=0` with non-empty primary_pts produces empty arrays

- **Given** `primary_pts` of shape `(3, 10, 2)`, `lateral_pts` of shape `(3, 10, 2)`, and `expected_count=0`
- **When** `filter_plants_with_unexpected_ct` is called
- **Then** the returned primary array has shape `(0, 10, 2)` (mismatch branch, `round(0) == 0 != 3`)
- **And** the returned lateral array has shape `(0, 10, 2)`

#### Scenario: Half-integer expected_count uses Python banker's rounding

- **Given** `primary_pts` of shape `(2, 10, 2)`, `lateral_pts` of shape `(2, 10, 2)`, and `expected_count=2.5`
- **When** `filter_plants_with_unexpected_ct` is called
- **Then** both output arrays are element-wise equal to the inputs (because `round(2.5) == 2 == len(primary_pts)`)
- **And given** `primary_pts` of shape `(4, 10, 2)`, `lateral_pts` of shape `(4, 10, 2)`, and `expected_count=3.5`
- **When** `filter_plants_with_unexpected_ct` is called
- **Then** both output arrays are element-wise equal to the inputs (because `round(3.5) == 4 == len(primary_pts)`)

### Requirement: `MultipleDicotPipeline` MUST run end-to-end when `Series.expected_count` is unknown

When `MultipleDicotPipeline` is executed against a `Series` whose `expected_count` property evaluates to `np.nan` (because the series was loaded without a `csv_path`, or the CSV does not list that series), the pipeline MUST complete without error and MUST NOT silently zero out plant-point arrays at the frame level. Specifically, at least one frame with a non-empty primary-root prediction MUST retain its detected primary instances after the count-filter TraitDef runs.

Because the per-frame trait `primary_pts_expected_plant_ct` is declared `include_in_csv=False` on `MultipleDicotPipeline` and is therefore discarded after each frame's aggregation inside `compute_multiple_dicots_traits`, the per-frame assertion MUST be verified by invoking `pipeline.compute_frame_traits(pipeline.get_initial_frame_traits(series, frame_idx))` directly against the instantiated pipeline — not by inspecting the return value of `compute_multiple_dicots_traits(series)`.

#### Scenario: `compute_multiple_dicots_traits` completes without error on a series loaded without a CSV

- **Given** a `Series` loaded from `tests/data/multiple_arabidopsis_11do/` via `Series.load(series_name=..., primary_path=..., lateral_path=...)` with no `csv_path` argument
- **And** `np.isnan(series.expected_count)` is `True`
- **When** `MultipleDicotPipeline().compute_multiple_dicots_traits(series)` is called
- **Then** the call returns a dict whose key set is exactly `{"series", "group", "qc_fail", "traits", "summary_stats"}`
- **And** no `ValueError` is raised by the underlying `filter_plants_with_unexpected_ct` call

#### Scenario: Per-frame `primary_pts_expected_plant_ct` is non-empty for a frame with primary predictions

- **Given** the same `Series` (loaded without `csv_path`) and a `MultipleDicotPipeline()` instance
- **When** for each `frame_idx` in `range(len(series))`, the test calls `frame_traits = pipeline.compute_frame_traits(pipeline.get_initial_frame_traits(series, frame_idx))`
- **Then** for at least one `frame_idx`, `frame_traits["primary_pts_expected_plant_ct"].shape[0]` is greater than or equal to 1
- **And** the test does NOT depend on `frame_idx == 0` specifically — the loop guarantees robustness against per-sample labeling drift where frame 0 happens to have no primary-root predictions

