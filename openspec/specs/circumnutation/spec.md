# circumnutation Specification

## Purpose
TBD - created by archiving change add-circumnutation-foundation. Update Purpose after archive.
## Requirements
### Requirement: Package layout
The system SHALL provide a `sleap_roots.circumnutation` sub-package whose import-tree is complete from this PR onward — every module name referenced anywhere in `docs/circumnutation/roadmap.md` exists as an importable Python module from PR #1 forward, even if its functions raise `NotImplementedError` until later PRs land. The package SHALL contain:

- 7 contract modules: `__init__.py`, `_constants.py`, `_types.py`, `_io.py`, `units.py`, `_noise.py`, `_geometry.py`
- 1 implementation module: `kinematics` (implemented from PR #2 onward; see Requirement: Tier 0 raw kinematic traits)
- 9 stub modules: `qc`, `synthetic`, `temporal_cwt`, `psi_g`, `midline`, `spatial_cwt`, `parametric`, `plotting`, `pipeline`

Each stub module SHALL define exactly one canonical callable per the table below, raising `NotImplementedError(f"PR #{N} — see docs/circumnutation/roadmap.md")` with the documented PR number. Each stub callable SHALL also have a complete Google-style docstring (Args, Returns, Raises) so `pydocstyle --convention=google` and the mkdocstrings auto-generated reference pages remain informative. Stubs whose tier PR will compose with the typed `ConstantsT` override-bag SHALL include `constants=None` as a forward-compatible keyword parameter so callers do not get `TypeError` before `NotImplementedError`.

| Stub module | Canonical callable | PR # |
|---|---|---|
| `qc` | `compute(trajectory_df, constants=None)` | 3 |
| `synthetic` | `generate_trajectory(...)` (pure-pixel; no `px_per_mm`) | 4 |
| `temporal_cwt` | `compute_scaleogram(x, cadence_s, constants=None)` | 5 |
| `psi_g` | `compute_psi_g(x, y, constants=None)` | 7 |
| `midline` | `reconstruct(x, y, cadence_s, constants=None)` | 8 |
| `spatial_cwt` | `compute_scaleogram(kappa, ds, constants=None)` | 9 |
| `parametric` | `compute(tier3_df, R_px, omega, Delta_phi)` | 11 |
| `plotting` | `scaleogram(scaleogram_result, out_path)` | 16 |
| `pipeline` | `compute_traits(inputs, constants=None)` | 14 |

The `kinematics` module SHALL be importable on the same terms as the other modules (clean import, namespaced logger) and SHALL expose `compute(trajectory_df, constants=None)` per Requirement: Tier 0 raw kinematic traits. Unlike the stub modules, calling `kinematics.compute` with a valid input SHALL NOT raise `NotImplementedError`.

The `_noise` and `_geometry` modules are private (underscore-prefixed) shared internals; their canonical callables are documented under Requirement: Tier 0 helper modules.

#### Scenario: All stub modules import cleanly
- **WHEN** a user runs `import sleap_roots.circumnutation as c; import sleap_roots.circumnutation.kinematics, sleap_roots.circumnutation.qc, sleap_roots.circumnutation.synthetic, sleap_roots.circumnutation.temporal_cwt, sleap_roots.circumnutation.psi_g, sleap_roots.circumnutation.midline, sleap_roots.circumnutation.spatial_cwt, sleap_roots.circumnutation.parametric, sleap_roots.circumnutation.plotting, sleap_roots.circumnutation.pipeline`
- **THEN** every import succeeds
- **AND** no exception is raised at module-import time

#### Scenario: Calling each remaining stub raises NotImplementedError with the correct PR number
- **GIVEN** the package imported as in the previous scenario
- **WHEN** the canonical callable in each of the 9 remaining stub modules (`qc`, `synthetic`, `temporal_cwt`, `psi_g`, `midline`, `spatial_cwt`, `parametric`, `plotting`, `pipeline`) is invoked (parameters per the table above; `NotImplementedError` fires before any argument check)
- **THEN** `NotImplementedError` is raised
- **AND** the exception message matches the regex `r"^PR #\d+ — see docs/circumnutation/roadmap\.md$"`
- **AND** the captured PR number equals the one in the table for that module

#### Scenario: `kinematics.compute` no longer raises NotImplementedError
- **GIVEN** a valid `trajectory_df` (8 row-identity columns + `frame`, `tip_x`, `tip_y`; ≥ 1 row)
- **WHEN** `sleap_roots.circumnutation.kinematics.compute(trajectory_df)` is invoked
- **THEN** the call returns a `pandas.DataFrame` (the Tier 0 per-plant output) without raising `NotImplementedError`

#### Scenario: Stubs accept `constants=None` where the table prescribes it
- **GIVEN** the stubs listed in the table above whose canonical callable includes `constants=None`
- **WHEN** a caller invokes that stub with `constants=...` keyword argument (any value)
- **THEN** `NotImplementedError` is raised (not `TypeError`)

#### Scenario: `synthetic.generate_trajectory` has no `px_per_mm` parameter
- **WHEN** `inspect.signature(sleap_roots.circumnutation.synthetic.generate_trajectory)` is inspected
- **THEN** the parameter list does not contain `px_per_mm`
- **AND** the docstring confirms the generator emits pure-pixel trajectories (callers compose `convert_to_mm()` if they want mm output)

#### Scenario: `import sleap_roots` succeeds without raising
- **WHEN** a user runs `import sleap_roots`
- **THEN** no exception is raised
- **AND** `sleap_roots.CircumnutationInputs` is accessible
- **AND** `sleap_roots.convert_to_mm` is accessible

### Requirement: CircumnutationInputs data class
The system SHALL provide an `attrs`-based `CircumnutationInputs` class capturing `(trajectory_df: pd.DataFrame, cadence_s: float, R_px: Optional[float] = None, run_id: Optional[str] = None)`. It SHALL validate at construction:

- `trajectory_df` is a `pandas.DataFrame` and is non-empty (≥ 1 row)
- `trajectory_df` contains the eight row-identity columns required by the trait CSV schema (Requirement: Trait CSV row-identity schema)
- `trajectory_df` contains the three per-frame columns `frame`, `tip_x`, `tip_y` (the foundation does not assert finiteness of `tip_x`/`tip_y` per row — that is a tier-PR concern — but their presence is mandatory)
- `cadence_s` is a positive finite float: `not isinstance(value, bool)` (Python booleans are int-subclass and SHALL be rejected explicitly), AND `math.isfinite(float(value))`, AND `float(value) > 0`. String-convertible numeric inputs (e.g. `cadence_s="300"`) are coerced to `float` before validation via an attrs converter.
- if `R_px` is set, the same rules apply: bool-rejected, finite, positive, string-convertible

It SHALL be re-exported from `sleap_roots/__init__.py`. The class SHALL NOT accept any `px_per_mm` parameter — calibration is a downstream concern handled by `convert_to_mm()`.

#### Scenario: Valid construction
- **GIVEN** a DataFrame containing all eight row-identity columns AND the three per-frame columns (`frame`, `tip_x`, `tip_y`) and at least one row
- **WHEN** `CircumnutationInputs(trajectory_df=df, cadence_s=300.0, R_px=2.4, run_id="plate_001")` is called
- **THEN** the instance is created without exception

#### Scenario: Missing row-identity column
- **GIVEN** a DataFrame missing the `plate_id` column
- **WHEN** `CircumnutationInputs(...)` is called
- **THEN** a `ValueError` is raised
- **AND** the exception message names the missing column

#### Scenario: Missing per-frame column
- **GIVEN** a DataFrame missing one of `frame`, `tip_x`, or `tip_y`
- **WHEN** `CircumnutationInputs(...)` is called
- **THEN** a `ValueError` is raised
- **AND** the exception message names the missing column

#### Scenario: Empty trajectory DataFrame
- **GIVEN** a DataFrame with all eight row-identity columns but zero rows
- **WHEN** `CircumnutationInputs(trajectory_df=df, cadence_s=300.0)` is called
- **THEN** a `ValueError` is raised
- **AND** the exception message indicates the DataFrame is empty

#### Scenario: Invalid cadence_s (zero, negative, NaN, infinity)
- **WHEN** `CircumnutationInputs(trajectory_df=df, cadence_s=v)` is called for any of `v ∈ {0.0, -1.0, float('nan'), float('inf'), float('-inf')}`
- **THEN** in each case a `ValueError` is raised
- **AND** the exception message names the `cadence_s` field

#### Scenario: cadence_s as Python bool is rejected
- **WHEN** `CircumnutationInputs(trajectory_df=df, cadence_s=True)` is called
- **THEN** a `ValueError` is raised
- **AND** the exception message names the `cadence_s` field

#### Scenario: Invalid R_px (zero, negative, NaN, infinity, bool)
- **WHEN** `CircumnutationInputs(trajectory_df=df, cadence_s=300.0, R_px=v)` is called for any of `v ∈ {0.0, -2.4, float('nan'), float('inf'), True}`
- **THEN** in each case a `ValueError` is raised
- **AND** the exception message names the `R_px` field
- **AND** `CircumnutationInputs(trajectory_df=df, cadence_s=300.0, R_px=None)` succeeds

#### Scenario: Importable from top-level
- **WHEN** a user runs `from sleap_roots import CircumnutationInputs`
- **THEN** the import succeeds

### Requirement: Trait CSV row-identity schema
Every per-plant trait CSV SHALL begin with the eight columns `(series, sample_uid, timepoint, plate_id, plant_id, track_id, genotype, treatment)` in that order, ahead of any trait columns. Today `plant_id` SHALL be populated identically to `track_id`; both columns SHALL exist so future divergence is non-breaking. `genotype` SHALL be populated from Series-level metadata where available (the `series-metadata` capability, PR #171), NaN otherwise. **`plate_id` and `treatment` SHALL be populated as NaN today** — no upstream produces them; the schema reserves them for future upstream metadata work. The DataFrame SHALL be sorted via `pandas.DataFrame.sort_values(by=['series', 'sample_uid', 'plate_id', 'plant_id', 'track_id'])`, where string columns sort lexicographically and integer columns (`track_id`) sort numerically.

The CSV-row builder `build_per_plant_template` SHALL key duplicate-row detection on the 5-tuple `(series, sample_uid, plate_id, plant_id, track_id)` only, NOT the full 8-tuple. If the same 5-tuple has conflicting values in `timepoint`, `genotype`, or `treatment` across the trajectory frames (a sign of upstream join error), `build_per_plant_template` SHALL raise `ValueError` rather than emit duplicate rows.

#### Scenario: Schema columns exist with correct dtypes
- **GIVEN** a DataFrame produced by the foundation's CSV-row builder for `CircumnutationInputs` containing 6 tracks
- **THEN** the first 8 columns are `series`, `sample_uid`, `timepoint`, `plate_id`, `plant_id`, `track_id`, `genotype`, `treatment`
- **AND** `track_id` has integer dtype
- **AND** `plant_id` is column-wise equal to `track_id`
- **AND** `series`, `sample_uid`, `genotype`, `treatment`, `plate_id` are object dtype (allowing NaN + string)

#### Scenario: Sort order is numeric for track_id
- **GIVEN** a DataFrame with `track_id ∈ {2, 10}` (and identical other identity columns)
- **WHEN** the foundation sorts the DataFrame
- **THEN** the row with `track_id=2` precedes the row with `track_id=10`
- **AND** the sort is NOT lexicographic (which would put `10` before `2`)

#### Scenario: Conflicting per-frame metadata raises ValueError
- **GIVEN** a `trajectory_df` where the same `(series, sample_uid, plate_id, plant_id, track_id)` 5-tuple has different `genotype` (or `treatment` or `timepoint`) values across frames
- **WHEN** `build_per_plant_template(inputs)` is called
- **THEN** a `ValueError` is raised
- **AND** the exception message names the offending 5-tuple AND the column whose values conflict

#### Scenario: track_id with NaN raises clear error
- **GIVEN** a `trajectory_df` whose `track_id` column contains NaN in at least one row
- **WHEN** `build_per_plant_template(inputs)` is called
- **THEN** a `ValueError` is raised
- **AND** the exception message names the `track_id` field (rather than a cryptic pandas `IntCastingNaNError`)

### Requirement: Pure-pixel pipeline output convention
The pipeline SHALL never accept `px_per_mm` as a parameter and SHALL never emit `[mm]` columns directly. Every length-bearing trait SHALL be expressed in pixels (`px`, `px²`, `px/frame`, `px/hr`, `px·hr⁻¹`); time in `hr` or `s`; angles in `rad`; rates in `hr⁻¹`; ratios as dimensionless (`—`); booleans as `bool`; integer counts as `int`; categorical strings as `string`. Internal CWT, ridge extraction, and derivative computations SHALL operate in pixels. This convention matches `TrackedTipPipeline`'s `lengths: "pixels"` declaration in `_TRACKED_TIP_UNITS` (`sleap_roots/tracked_tip_pipeline.py`).

#### Scenario: Pipeline output is calibration-independent
- **GIVEN** the same `CircumnutationInputs` (no `px_per_mm` parameter exists)
- **WHEN** the foundation's CSV-row builder produces a per-plant DataFrame for any future tier's traits
- **THEN** every numeric column has a unit string in the documented vocabulary
- **AND** no column has a unit string of `mm`, `mm²`, `mm/hr`, or any other mm-bearing unit

### Requirement: convert_to_mm utility
The system SHALL provide a `sleap_roots.circumnutation.units.convert_to_mm(traits_df: pd.DataFrame, units: dict[str, str], px_per_mm: float) -> tuple[pd.DataFrame, dict[str, str]]` pure function. It SHALL: (a) return a NEW DataFrame and units dict (input arguments not mutated), (b) for every column whose unit string is `px`, `px²`, `px/frame`, `px/hr`, or `px·hr⁻¹`, scale the values by the appropriate power of `1/px_per_mm` and rename the column with `_mm`-suffix replacing the `_px`-suffix (also updating the unit string), (c) pass non-px columns and their units through unchanged, (d) detect and raise `ValueError` when a `_px`→`_mm` rename would collide with an existing `_mm`-named column in the input (silent data loss prevention), (e) validate `px_per_mm` is a positive finite float (rejects 0, negative, NaN, ±inf, Python bool) and raise `ValueError` otherwise. The function SHALL be re-exported from `sleap_roots/__init__.py`.

#### Scenario: Identity at px_per_mm = 1.0
- **GIVEN** a 1-row DataFrame with column `length_px = 47.24` and units `{"length_px": "px"}`
- **WHEN** `convert_to_mm(df, units, px_per_mm=1.0)` is called
- **THEN** the returned DataFrame has column `length_mm = 47.24`
- **AND** the returned units has `{"length_mm": "mm"}`
- **AND** the input DataFrame and units dict are unchanged

#### Scenario: 1200 DPI conversion
- **GIVEN** a 1-row DataFrame with `length_px = 47.24` and units `{"length_px": "px"}`
- **WHEN** `convert_to_mm(df, units, px_per_mm=47.24)` is called
- **THEN** the returned DataFrame has column `length_mm = 1.0` (within IEEE float tolerance)
- **AND** the returned units has `{"length_mm": "mm"}`

#### Scenario: Velocity unit conversions
- **GIVEN** a DataFrame with columns `v_long_px_per_hr`, `v_total_px_per_frame` and matching unit strings
- **WHEN** `convert_to_mm(df, units, px_per_mm=47.24)` is called
- **THEN** the returned columns are `v_long_mm_per_hr`, `v_total_mm_per_frame` with values divided by `47.24`
- **AND** units strings are updated to `mm/hr`, `mm/frame`

#### Scenario: Non-px columns pass through
- **GIVEN** a DataFrame with columns `T_nutation_hr` (unit `hr`), `is_nutating` (unit `bool`), `B_balance_number` (unit `—`)
- **WHEN** `convert_to_mm(df, units, px_per_mm=47.24)` is called
- **THEN** all three columns retain their names, values, and unit strings unchanged

#### Scenario: Invalid px_per_mm (zero, negative, NaN, infinity)
- **WHEN** `convert_to_mm(df, units, px_per_mm=v)` is called for any of `v ∈ {0.0, -1.0, float('nan'), float('inf'), float('-inf')}`
- **THEN** a `ValueError` is raised
- **AND** the exception message names the `px_per_mm` field

#### Scenario: Rename collision raises ValueError
- **GIVEN** a DataFrame with BOTH `length_px` AND `length_mm` columns (the latter from a prior conversion)
- **WHEN** `convert_to_mm(df, units, px_per_mm=47.24)` is called
- **THEN** a `ValueError` is raised
- **AND** the exception message names both the source column (`length_px`) and the target column (`length_mm`) that would collide

### Requirement: Units sidecar JSON
For every per-plant trait CSV the system SHALL write a sibling `traits_per_plant.units.json` mapping each column name to a unit string. Every column (numeric, boolean, string) SHALL be present. The unit-string vocabulary for sidecar values is `sleap_roots.circumnutation._constants.PIPELINE_UNIT_VOCABULARY` (pixel-based and calibration-independent units only — no mm-based units). The writer SHALL validate every unit string against `PIPELINE_UNIT_VOCABULARY` BEFORE writing and SHALL raise `ValueError` naming the offending column/unit pair if any value is out-of-vocabulary. The JSON file SHALL be written with `encoding="utf-8"` so non-ASCII unit symbols (`²`, `⁻`, `·`) round-trip on Windows.

#### Scenario: Sidecar exists and parses
- **WHEN** the foundation writes a CSV via `_io.write_per_plant_csv`
- **THEN** a sibling `traits_per_plant.units.json` exists in the same directory
- **AND** it parses as valid JSON
- **AND** every column from the CSV is a key in the JSON mapping
- **AND** every value is a string in `PIPELINE_UNIT_VOCABULARY`

#### Scenario: UTF-8 round-trip with non-ASCII unit
- **GIVEN** a units dict containing `{"helix_signed_area": "px²"}`
- **WHEN** the foundation writes the sidecar to disk
- **AND** loads it back via `_io.read_units_sidecar`
- **THEN** the round-tripped dict contains `{"helix_signed_area": "px²"}` byte-for-byte unchanged

#### Scenario: Writing with invalid unit raises ValueError
- **GIVEN** a units dict containing a unit string not in `PIPELINE_UNIT_VOCABULARY` (e.g. `{"length_px": "mm"}` or `{"length_px": "kg"}`)
- **WHEN** `_io.write_per_plant_csv(out_path, df, units, run_metadata)` is called
- **THEN** a `ValueError` is raised
- **AND** the exception message names the offending column and the invalid unit string
- **AND** no CSV or sidecar files are written

#### Scenario: Writer rejects units dict that doesn't cover every column
- **GIVEN** a DataFrame with columns `["a", "b"]` and a units dict `{"a": "px"}` (missing `"b"`)
- **WHEN** `_io.write_per_plant_csv(out_path, df, units, run_metadata)` is called
- **THEN** a `ValueError` is raised
- **AND** the exception message names the missing column(s)
- **AND** no CSV or sidecar files are written

#### Scenario: Writer rejects units dict with extra keys not in the DataFrame
- **GIVEN** a DataFrame with columns `["a"]` and a units dict `{"a": "px", "b": "hr"}` (`"b"` not in df)
- **WHEN** `_io.write_per_plant_csv(out_path, df, units, run_metadata)` is called
- **THEN** a `ValueError` is raised
- **AND** the exception message names the extra key(s)
- **AND** no CSV or sidecar files are written

### Requirement: Run-metadata sidecar
For every per-plant CSV the system SHALL write a sibling `run_metadata.json` capturing: `input_path`, `sleap_roots_git_sha`, `sleap_roots_version`, `sleap_io_version`, `numpy_version`, `scipy_version`, `pandas_version`, `python_version`, `platform`, `timestamp` (ISO 8601 UTC), `run_id`, `_schema_version`, `_constants_version`, `_constants_snapshot`. The `_constants_snapshot` SHALL be a JSON-serializable mapping from every name in `_constants.py` to its value at write time. The numpy / scipy / pandas / platform fields support numerical reproducibility (IEEE float rounding can differ between numpy versions and across BLAS implementations).

#### Scenario: Run-metadata sidecar contains required fields
- **WHEN** the foundation writes a CSV
- **THEN** a sibling `run_metadata.json` exists in the same directory
- **AND** every key listed above is present and non-null (except `run_id` which may be null)
- **AND** `_schema_version` and `_constants_version` are integers
- **AND** `_constants_snapshot` contains every name in `_constants.py` with its current value
- **AND** `numpy_version`, `scipy_version`, `pandas_version` are present as version strings (e.g. `"2.3.4"`, `"1.16.3"`, `"2.2.0"`) or `"unknown"` if the dependency could not be imported
- **AND** `platform` is a non-empty string (the value of `platform.platform()`)

#### Scenario: Constants snapshot reflects override
- **GIVEN** a custom `ConstantsT` override passed to the writer with `BAND_POWER_NOISE_RATIO = 4`
- **WHEN** the foundation writes the metadata
- **THEN** `_constants_snapshot["BAND_POWER_NOISE_RATIO"] == 4`
- **AND** the default-valued constants reflect the unmodified defaults

### Requirement: Module-level constants
The system SHALL expose all overridable defaults as module-level named constants in `sleap_roots/circumnutation/_constants.py`. The set SHALL include at minimum: `NOISE_MASK_K`, `LGZ_STEADY_STATE_RESIDUAL_MAX`, `NYQUIST_RATIO_MAX`, `SG_D2_AGREEMENT_MAX`, `LGZ_NMIN_RESOLVABLE`, `COI_FRACTION_MAX`, `BAND_POWER_NOISE_RATIO`, `WAVELET_DEFAULT_TEMPORAL`, `WAVELET_DEFAULT_SPATIAL`, `SG_WINDOW_SHORT`, `SG_DEGREE`, `SG_WINDOW_DETREND`, `OUTLIER_STEP_RATIO`, `GROWTH_AXIS_RELIABILITY_K`, `_SCHEMA_VERSION`, `_CONSTANTS_VERSION`. The values SHALL match the defaults in `docs/circumnutation/roadmap.md` cross-cutting concern CC-2; `_SCHEMA_VERSION` and `_CONSTANTS_VERSION` SHALL each be `1` initially. The module SHALL also expose `PIPELINE_UNIT_VOCABULARY` (px-based + calibration-independent units, the closed sidecar vocabulary), `CONVERTED_UNIT_VOCABULARY` (mm-based units produced by `convert_to_mm`), and `VALID_UNIT_VOCABULARY` (their union), plus `ROW_IDENTITY_UNITS` (the canonical units dict for the eight row-identity columns).

#### Scenario: All required constants are importable with correct types
- **WHEN** a user runs `from sleap_roots.circumnutation import _constants`
- **THEN** every name listed above is an attribute of `_constants`
- **AND** each value matches the documented default in `roadmap.md` CC-2
- **AND** `_constants._SCHEMA_VERSION` and `_constants._CONSTANTS_VERSION` are integers equal to `1`
- **AND** `_constants.PIPELINE_UNIT_VOCABULARY`, `_constants.CONVERTED_UNIT_VOCABULARY`, `_constants.VALID_UNIT_VOCABULARY`, `_constants.ROW_IDENTITY_UNITS` are all importable

### Requirement: Per-module logger convention
Every module in `sleap_roots/circumnutation/` SHALL declare a module-level logger via `logger = logging.getLogger(__name__)`. The package SHALL NOT call `logging.basicConfig` or otherwise configure handlers at import time. No log records SHALL be emitted at package import time.

#### Scenario: Module loggers are namespaced
- **GIVEN** `import logging; import sleap_roots.circumnutation.kinematics`
- **THEN** `logging.getLogger("sleap_roots.circumnutation.kinematics")` returns the same logger object that the module uses

#### Scenario: No handlers added on import
- **GIVEN** a fresh Python process
- **WHEN** `caplog` captures all records and `import sleap_roots.circumnutation` is executed
- **THEN** the root logger's handlers list is unchanged from before the import
- **AND** no log records are emitted during import

### Requirement: Tier 0 raw kinematic traits
The system SHALL provide `sleap_roots.circumnutation.kinematics.compute(trajectory_df: pd.DataFrame, constants: Optional[ConstantsT] = None) -> pd.DataFrame`. The function SHALL accept the canonical `(trajectory_df, constants=None)` signature locked by the foundation's Package layout requirement — `cadence_s` SHALL NOT appear in the signature (the function emits cadence-independent units only).

It SHALL return a per-plant `pandas.DataFrame` with one row per unique `(series, sample_uid, plate_id, plant_id, track_id)` 5-tuple in the input, sorted via the same convention as `_io._build_per_plant_template_from_df`. The DataFrame columns SHALL be (in this order):

1. The 8 row-identity columns in their declared order: `series`, `sample_uid`, `timepoint`, `plate_id`, `plant_id`, `track_id`, `genotype`, `treatment`.
2. The 9 Tier 0 trait columns and 1 boolean flag, in this order:

| Column | Unit | Definition |
|---|---|---|
| `v_total_median_px_per_frame` | `px/frame` | `np.nanmedian(\|Δxy_i\| / Δframe_i)` over consecutive present frames — median per-frame step magnitude (gap-aware). |
| `v_long_signed_median_px_per_frame` | `px/frame` | `np.nanmedian((Δxy_i · û_g) / Δframe_i)` where `û_g = (xy[-1] − xy[0]) / D` is the unit growth-axis vector — signed median longitudinal component. |
| `v_long_abs_median_px_per_frame` | `px/frame` | `np.nanmedian(\|Δxy_i · û_g\| / Δframe_i)` — absolute median longitudinal component. |
| `v_lat_signed_median_px_per_frame` | `px/frame` | `np.nanmedian((Δxy_i · û_lat) / Δframe_i)` where `û_lat = (−û_g[1], û_g[0])` — signed median lateral component (expected ≈ 0 by symmetry; serves as a sanity-check trait). |
| `v_lat_abs_median_px_per_frame` | `px/frame` | `np.nanmedian(\|Δxy_i · û_lat\| / Δframe_i)` — absolute median lateral component. |
| `long_lat_ratio` | `—` | `v_long_abs_median_px_per_frame / v_lat_abs_median_px_per_frame`; NaN when `v_lat_abs_median_px_per_frame == 0`. |
| `path_displacement_ratio` | `—` | `L / D` where `L = sum(\|Δxy_i\|)` over consecutive present frames and `D = \|xy[-1] − xy[0]\|`; NaN when `D == 0` exactly. |
| `angular_amplitude` | `rad` | `np.nanmax(ψ_g) − np.nanmin(ψ_g)` where `ψ_g = _geometry.compute_psi_g(x, y) = np.unwrap(np.arctan2(dx, dy))` per Bastien-Meroz 2016 Eq. 20 / theory.md §3.5 (note argument order: `dx` first, then `dy`); rotation-invariant peak-to-peak extent. |
| `principal_axis_angle` | `rad` | `np.arctan2(û_g[1], û_g[0])`; full range `(−π, π]`; image-y-downward convention (a root growing image-down reads as `+π/2`). |
| `growth_axis_unreliable` | `bool` | True iff the growth axis is unreliable per Requirement: Growth-axis reliability gate. |

The function SHALL emit only units in `PIPELINE_UNIT_VOCABULARY` — specifically `px/frame`, `—`, `rad`, `bool` for the new columns. NO `_mm`-suffixed columns and NO `_px_per_hr` columns SHALL be emitted by `kinematics.compute`.

The function SHALL process each track via the algorithm documented in `design.md` D5: (a) drop NaN rows on `tip_x`/`tip_y` BEFORE any `np.diff` or arithmetic — this NaN-then-sort ordering is load-bearing because `np.diff` of a NaN-bearing array propagates NaN to two adjacent diffs and `np.sum` (used to compute path length `L`) would silently return NaN; (b) sort the remaining rows by `frame`; (c) if fewer than 2 frames remain, emit NaN for all 9 trait columns and `False` for `growth_axis_unreliable`; (d) compute gap-aware per-frame velocities `Δxy / Δframe`; (e) compute the magnitude, signed and absolute longitudinal/lateral components, ratios, `angular_amplitude` via `_geometry.compute_psi_g`, and `principal_axis_angle`; (f) apply the reliability gate. The path-length sum `L = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))` uses `np.sum` (not `np.nansum`); the dropna precondition guarantees no NaN reaches this sum.

The function SHALL accept `constants=None` (default) by constructing the default `ConstantsT()`, OR a caller-supplied `ConstantsT` override. The constants consumed are `SG_WINDOW_SHORT`, `SG_DEGREE` (passed to `_noise.compute_sg_residual_xy`), and `GROWTH_AXIS_RELIABILITY_K` (the gate threshold multiplier).

The function SHALL validate `trajectory_df` is a `pandas.DataFrame` containing the 8 row-identity columns plus `frame`, `tip_x`, `tip_y` columns. On validation failure it SHALL raise `ValueError` whose message names the offending issue (missing column name, wrong type, etc.) — matching the foundation's permissive-but-clear-error style.

#### Scenario: Straight-line track yields exact analytical values
- **GIVEN** a 100-frame trajectory_df with `tip_x = frame * 1.0` and `tip_y = 0.0` for a single track (unit horizontal velocity)
- **WHEN** `kinematics.compute(trajectory_df)` is called
- **THEN** the returned DataFrame has exactly 1 row
- **AND** `v_total_median_px_per_frame == 1.0` within IEEE float tolerance
- **AND** `v_long_signed_median_px_per_frame == 1.0` and `v_long_abs_median_px_per_frame == 1.0`
- **AND** `v_lat_signed_median_px_per_frame == 0.0` and `v_lat_abs_median_px_per_frame == 0.0`
- **AND** `long_lat_ratio` is `NaN` (denominator zero)
- **AND** `path_displacement_ratio == 1.0` exactly (`L == D`)
- **AND** `angular_amplitude == 0.0` (ψ_g is constant)
- **AND** `principal_axis_angle == 0.0`
- **AND** `growth_axis_unreliable == False`

#### Scenario: Image-y-downward straight-line track reports `principal_axis_angle ≈ π/2`
- **GIVEN** a 100-frame trajectory_df with `tip_x = 0.0` and `tip_y = frame * 1.0` for a single track (unit downward velocity in image space)
- **WHEN** `kinematics.compute(trajectory_df)` is called
- **THEN** `principal_axis_angle == math.pi / 2` within IEEE float tolerance

#### Scenario: Pure-noise track triggers gate and NaNs rotation-dependent traits
- **GIVEN** a 100-frame trajectory_df where `tip_x` and `tip_y` are i.i.d. `N(0, 1)` drawn via `np.random.default_rng(0).normal(0, 1, size=(2, 100))` for a single track (concrete seed for cross-platform test determinism)
- **WHEN** `kinematics.compute(trajectory_df)` is called
- **THEN** `growth_axis_unreliable == True`
- **AND** the 6 rotation-dependent columns (`v_long_signed_median_px_per_frame`, `v_long_abs_median_px_per_frame`, `v_lat_signed_median_px_per_frame`, `v_lat_abs_median_px_per_frame`, `long_lat_ratio`, `principal_axis_angle`) are all `NaN`
- **AND** the 3 rotation-invariant traits (`v_total_median_px_per_frame`, `path_displacement_ratio`, `angular_amplitude`) are finite (not `NaN`)

#### Scenario: Circular trajectory yields `angular_amplitude ≈ 2π`
- **GIVEN** a 100-frame trajectory_df with `tip_x = 10·cos(2π·frame/100) + frame * 0.1` and `tip_y = 10·sin(2π·frame/100)` for a single track. The circle radius `R = 10` is chosen so the circle's per-frame velocity magnitude `R·(2π/100) ≈ 0.628` dominates the horizontal drift (`0.1` per frame); this ensures `dx = −R·sin(t)·dt + drift` swings through both signs over one revolution and `ψ_g = atan2(dx, dy)` sweeps the full 2π range. A smaller `R` with the same drift would leave `dx` always positive and `ψ_g` would not complete a revolution.
- **WHEN** `kinematics.compute(trajectory_df)` is called
- **THEN** `angular_amplitude` is in the range `[2π - 0.5, 2π + 0.5]` (one full revolution within tolerance)
- **AND** `growth_axis_unreliable == False` (D ≈ 10 px from the horizontal drift dominates the smooth-trajectory SG residual)

#### Scenario: NaN rows are dropped BEFORE diff (ordering is load-bearing)
- **GIVEN** a 100-frame straight-line trajectory_df with `tip_x = frame * 1.0` and `tip_y = 0.0` where 10 random rows (seeded `np.random.default_rng(0)`) have `tip_x = NaN`
- **WHEN** `kinematics.compute(trajectory_df)` is called
- **THEN** the trait values match the no-NaN-row case (scenario "Straight-line track yields exact analytical values") within IEEE float tolerance — specifically `v_total_median_px_per_frame == 1.0` exactly (gap-aware diff: removing 10 NaN rows produces a remaining 90-row contiguous-or-gapped track, each per-frame step normalized by Δframe yields 1.0)
- **AND** `path_displacement_ratio == 1.0` exactly (the NaN-then-sort ordering ensures `np.sum` of the present-row step magnitudes returns a finite value, not NaN — if the diff happened before dropna, NaN would propagate through `np.sum` and contaminate the ratio)

#### Scenario: Frame gaps are handled gap-aware
- **GIVEN** a straight-line trajectory_df at velocity 1 px/frame, but with frames `[40..50)` missing (10-frame gap)
- **WHEN** `kinematics.compute(trajectory_df)` is called
- **THEN** `v_total_median_px_per_frame == 1.0` exactly (gap-aware per-frame velocity: the big jump across the gap is divided by `Δframe = 11`, normalizing to 1 px/frame)

#### Scenario: Track with fewer than 2 valid frames emits NaN traits without raising
- **GIVEN** a trajectory_df with a single track that has only 1 row (or all rows have NaN coords)
- **WHEN** `kinematics.compute(trajectory_df)` is called
- **THEN** the returned DataFrame has 1 row for that track
- **AND** all 9 trait columns are `NaN`
- **AND** `growth_axis_unreliable == False` (cannot judge reliability with fewer than 2 frames)
- **AND** no exception is raised

#### Scenario: Zero net displacement yields NaN ratios and `growth_axis_unreliable=True`
- **GIVEN** a trajectory_df for a single track whose start and end coordinates are identical (`xy[-1] == xy[0]`)
- **WHEN** `kinematics.compute(trajectory_df)` is called
- **THEN** `path_displacement_ratio` is `NaN`
- **AND** `growth_axis_unreliable == True`
- **AND** the 6 rotation-dependent columns are `NaN`
- **AND** the rotation-invariant `v_total_median_px_per_frame` and `angular_amplitude` are finite (not NaN)

#### Scenario: Output DataFrame columns are in the specified order
- **GIVEN** a valid trajectory_df with 6 tracks
- **WHEN** `kinematics.compute(trajectory_df)` is called
- **THEN** the returned DataFrame has exactly 18 columns
- **AND** the first 8 columns are `series`, `sample_uid`, `timepoint`, `plate_id`, `plant_id`, `track_id`, `genotype`, `treatment` in that order
- **AND** the next 10 columns are `v_total_median_px_per_frame`, `v_long_signed_median_px_per_frame`, `v_long_abs_median_px_per_frame`, `v_lat_signed_median_px_per_frame`, `v_lat_abs_median_px_per_frame`, `long_lat_ratio`, `path_displacement_ratio`, `angular_amplitude`, `principal_axis_angle`, `growth_axis_unreliable` in that order

#### Scenario: Output column units are within `PIPELINE_UNIT_VOCABULARY`
- **GIVEN** the output of `kinematics.compute(trajectory_df)` and the units-mapping dict for the 10 new columns
- **WHEN** every unit string is checked against `sleap_roots.circumnutation._constants.PIPELINE_UNIT_VOCABULARY`
- **THEN** every unit string is a member of the vocabulary
- **AND** no unit string is `mm`-bearing or `px/hr`-bearing

#### Scenario: ConstantsT override changes the gate threshold
- **GIVEN** a trajectory_df constructed to produce a known `D` and a known SG-residual value, by either of two equivalent test-construction patterns:
  - (Recipe A — preferred for unit tests) a `monkeypatch`-injected `_noise.compute_sg_residual_xy` that returns a fixed `1.0`, paired with a 6-frame trajectory `xy = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)]` yielding `D = 5.0` exactly
  - (Recipe B — for integration coverage) a 100-frame trajectory with `x_smooth = np.linspace(0, 5, 100)`, `y_smooth = np.zeros(100)`, plus `np.random.default_rng(0).normal(0, 0.7, size=(2, 100))` added to xy; the smooth+noise combination yields `D ≈ 5` and SG-residual `≈ 1` deterministically
- **WHEN** `kinematics.compute(trajectory_df, constants=ConstantsT(GROWTH_AXIS_RELIABILITY_K=3))` is called
- **THEN** `growth_axis_unreliable == False` (because `5 > 3 * 1`)
- **AND** when called with `ConstantsT(GROWTH_AXIS_RELIABILITY_K=10)` instead, `growth_axis_unreliable == True` (because `5 < 10 * 1`)

#### Scenario: Invalid trajectory_df raises ValueError
- **WHEN** `kinematics.compute(None)` is called
- **THEN** a `ValueError` is raised whose message indicates the input is not a `pandas.DataFrame`
- **AND** `kinematics.compute(df_missing_tip_x)` similarly raises a `ValueError` naming the missing column

### Requirement: Growth-axis reliability gate
For each track, the system SHALL compute net displacement `D = ‖xy[-1] − xy[0]‖` (in pixels) and a local SG-residual noise estimate via `_noise.compute_sg_residual_xy(x, y, window=constants.SG_WINDOW_SHORT, degree=constants.SG_DEGREE)`. It SHALL set `growth_axis_unreliable = (D < constants.GROWTH_AXIS_RELIABILITY_K * sg_residual_xy_local)` — strict less-than. The threshold multiplier is configurable via `ConstantsT.GROWTH_AXIS_RELIABILITY_K` with documented default `10`.

When `growth_axis_unreliable == True`, the 6 rotation-dependent trait columns (`v_long_signed_median_px_per_frame`, `v_long_abs_median_px_per_frame`, `v_lat_signed_median_px_per_frame`, `v_lat_abs_median_px_per_frame`, `long_lat_ratio`, `principal_axis_angle`) SHALL be set to `NaN`. The 3 rotation-invariant traits (`v_total_median_px_per_frame`, `path_displacement_ratio`, `angular_amplitude`) SHALL NOT be NaN'd by the gate (they may still be NaN for other documented reasons — e.g. `path_displacement_ratio` when `D == 0`, or all traits when fewer than 2 valid frames).

The Tier 0 module SHALL be the sole emitter of the `growth_axis_unreliable` column on the per-plant trait DataFrame; subsequent tier PRs (e.g. PR #3 QC) SHALL NOT re-emit this column, but MAY compose with it (e.g. by including it as a clause in a composite QC trait like `track_is_clean`).

The local SG residual computation SHALL use the same `_noise.compute_sg_residual_xy` helper that PR #3 (QC) will use to emit the canonical `sg_residual_xy` trait, so the gate value and the canonical trait are guaranteed-identical from identical inputs.

#### Scenario: Gate fires below threshold
- **GIVEN** a trajectory_df constructed via the dual-recipe pattern documented in "ConstantsT override changes the gate threshold" — `D = 5 px` paired with a known/fixed `sg_residual = 1.0 px` (either via `monkeypatch` of `_noise.compute_sg_residual_xy` or via a smooth-line-plus-σ=0.7-noise construction)
- **WHEN** `kinematics.compute(trajectory_df)` is called (default `GROWTH_AXIS_RELIABILITY_K=10`)
- **THEN** `growth_axis_unreliable == True`
- **AND** the 6 rotation-dependent columns are NaN

#### Scenario: Gate does not fire above threshold
- **GIVEN** a trajectory_df constructed via the same dual-recipe pattern — `D = 100 px` paired with a fixed `sg_residual = 1.0 px`
- **WHEN** `kinematics.compute(trajectory_df)` is called (default K=10)
- **THEN** `growth_axis_unreliable == False`
- **AND** the 6 rotation-dependent columns carry finite (non-NaN) values

#### Scenario: Gate threshold is strict less-than at boundary
- **GIVEN** a trajectory_df with `D = 10.0` and SG residual fixed to `1.0` (via monkeypatch as in the dual-recipe pattern), at the exact boundary `D == K * residual`
- **WHEN** `kinematics.compute(trajectory_df)` is called (K=10)
- **THEN** `growth_axis_unreliable == False` (strict less-than per the spec — at equality, the axis is judged reliable)

#### Scenario: Rotation-invariant traits survive the gate
- **GIVEN** any trajectory_df where the gate fires (`growth_axis_unreliable == True`)
- **WHEN** `kinematics.compute(trajectory_df)` is called
- **THEN** `v_total_median_px_per_frame`, `path_displacement_ratio`, and `angular_amplitude` are NOT set to NaN by the gate
- **AND** they are finite values reflecting the actual track kinematics

### Requirement: Tier 0 helper modules
The system SHALL provide two private helper modules in `sleap_roots/circumnutation/` to host computations shared across Tier 0 (PR #2) and future tier PRs (#3, #7):

**`_noise.py`** SHALL define `compute_sg_residual_xy(x: np.ndarray, y: np.ndarray, window: int, degree: int) -> float`. The function SHALL apply `scipy.signal.savgol_filter` to `x` and `y` independently using the given `window` and `degree`, compute the standard deviation of residuals for each, and return their quadrature sum: `sqrt(std(x - x_smooth)^2 + std(y - y_smooth)^2)`. When `len(x) < window`, the function SHALL return `np.nan` and log a `DEBUG` record naming the short-input case rather than raising. The function SHALL be deterministic — identical inputs SHALL produce identical outputs.

**`_geometry.py`** SHALL define `compute_psi_g(x: np.ndarray, y: np.ndarray) -> np.ndarray`. The function SHALL compute `dx = np.diff(x)`, `dy = np.diff(y)`, then `psi = np.arctan2(dx, dy)` (note argument order: `dx` first, then `dy` — this matches Bastien-Meroz 2016 Eq. 20 verbatim and `docs/circumnutation/theory.md` §3.5's explicit instruction *"The pipeline must use `atan2(dx/dt, dy/dt)` and unwrap the result"*), then return `np.unwrap(psi)`. The return shape is `(len(x) − 1,)`. When `len(x) < 2`, the function SHALL return an empty 1-D array `np.array([])`. The argument order is convention-critical: PR #7's `handedness` trait (`theory.md` §7.3) defines `+1 = counterclockwise (left-handed in image frame)` as the sign of mean `dψ_g/dt`, which requires the BM-Eq.-20 convention for the handedness sign to be correct in the published literature sense.

Both modules SHALL declare module-level loggers via `logger = logging.getLogger(__name__)`. Both module names SHALL be underscore-prefixed indicating they are private internals — they are not re-exported from the package's `__init__.py`.

#### Scenario: `_noise.compute_sg_residual_xy` returns zero for a polynomial of degree ≤ SG_DEGREE
- **GIVEN** `x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]` and `y = [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]` (linear and quadratic respectively) and `window=5, degree=3`
- **WHEN** `compute_sg_residual_xy(x, y, window=5, degree=3)` is called
- **THEN** the return is `0.0` within IEEE float tolerance (SG of degree 3 fits these polynomials exactly)

#### Scenario: `_noise.compute_sg_residual_xy` recovers approximate σ on noisy data
- **GIVEN** `x_smooth = np.linspace(0, 100, 1000)`, `y_smooth = np.zeros(1000)`, `noise = np.random.default_rng(0).normal(0, 1.0, size=(2, 1000))`, `x = x_smooth + noise[0]`, `y = y_smooth + noise[1]`
- **WHEN** `compute_sg_residual_xy(x, y, window=5, degree=3)` is called
- **THEN** the return is within `[1.0, 1.6]` (the quadrature sum is `sqrt(σ_x^2 + σ_y^2) = sqrt(2) ≈ 1.41`; tolerance accounts for SG's slight under-estimate of σ)

#### Scenario: `_noise.compute_sg_residual_xy` returns NaN for short inputs
- **WHEN** `compute_sg_residual_xy(np.array([1.0, 2.0]), np.array([3.0, 4.0]), window=5, degree=3)` is called
- **THEN** the return is `np.nan`
- **AND** no exception is raised

#### Scenario: `_geometry.compute_psi_g` returns a constant for a straight-line track
- **GIVEN** `x = np.arange(100, dtype=float)` (constant velocity in +x direction) and `y = np.zeros(100)`
- **WHEN** `compute_psi_g(x, y)` is called
- **THEN** the return array has length 99 and all elements equal `math.pi / 2` (within IEEE float tolerance — because `atan2(dx=1, dy=0) = π/2`)
- **AND** for the orthogonal case `x = np.zeros(100)`, `y = np.arange(100, dtype=float)` (constant velocity in +y direction = image-down), `compute_psi_g(x, y)` returns all elements equal to `0.0` (because `atan2(dx=0, dy=1) = 0`)

#### Scenario: `_geometry.compute_psi_g` returns monotonic unwrapped angles spanning ≈2π for a closed circular trajectory
- **GIVEN** `t = np.linspace(0, 2*np.pi, 100)`, `x = np.cos(t)`, `y = np.sin(t)` (one full revolution in `(x, y)` parametric space)
- **WHEN** `compute_psi_g(x, y)` is called
- **THEN** the return is a length-99 array that is strictly monotonic (either entirely increasing or entirely decreasing — the direction is convention-dependent; under `atan2(dx, dy)` with `x = cos(t), y = sin(t)`, ψ_g decreases monotonically because the velocity-direction angle rotates by `−2π` over the revolution)
- **AND** the difference between consecutive elements is bounded (no `±2π` discontinuities — i.e., the unwrap worked)
- **AND** the total absolute span (`abs(return[-1] − return[0])`) is approximately `2π` (one full revolution).

#### Scenario: `_geometry.compute_psi_g` returns empty for too-short input
- **WHEN** `compute_psi_g(np.array([1.0]), np.array([2.0]))` is called
- **THEN** the return is an empty 1-D array (`shape == (0,)`)
- **AND** no exception is raised

### Requirement: Tier 0 input-validation boundary
The system SHALL validate at the entry of `kinematics.compute` that `trajectory_df` is a `pandas.DataFrame` and that it contains the eight row-identity columns plus `frame`, `tip_x`, `tip_y`. On validation failure, `ValueError` SHALL be raised with a message naming the offending field or type. Validation MAY be implemented by delegating to the foundation's `_types._validate_trajectory_df` helper, OR by re-implementing the equivalent checks inline; either choice is acceptable as long as the error messages match the foundation's permissive-but-clear-error style.

Beyond the foundation's column-presence and DataFrame-type checks, Tier 0 makes the following deliberate scoping choices on input quality (documented as non-goals of THIS PR; deferred to future tier PRs or pre-validation by the caller):

- **`±inf` in `tip_x` / `tip_y`** is NOT validated. `pandas.DataFrame.dropna(subset=["tip_x", "tip_y"])` retains `±inf` rows (NaN-only filter), and `np.linalg.norm` of an `inf` vector returns `inf`. Tier 0 propagates `inf` through trait computations; resulting traits may be `inf` or `NaN` depending on the propagation path. **Rationale**: SLEAP predictions never emit `±inf`; this would only happen via corrupted upstream data. PR #3 QC's `frac_outlier_steps` is the right place to detect and gate on this.
- **Duplicate `(track_id, frame)` rows** are NOT detected. The `sort_values("frame")` is stable, so duplicate-frame rows keep their input order; `np.diff(frame)` then produces `Δframe = 0` for the duplicate pair, leading to `Δxy/0 = inf` or `nan/0 = nan`. **Rationale**: duplicate `(track_id, frame)` indicates an upstream data error (`TrackedTipPipeline` never produces them — each instance has a unique `(track_id, frame)` by construction). Pre-validation is the caller's responsibility.
- **Non-contiguous `Δframe = 0` from sort instability** cannot occur because `np.diff` on a non-decreasing `frame` array produces `Δframe ≥ 0`, and `Δframe = 0` only happens with duplicates (handled above).
- **Non-integer `frame` columns** (e.g., timestamps as floats) are NOT rejected — `frame` may be any numeric type. The "per-frame" semantic of the emitted velocity columns becomes "per-sample" under non-integer frames; for the Nipponbare/KitaakeX fixtures (integer-frame), this is moot.

#### Scenario: `±inf` in tip_x propagates without raising
- **GIVEN** a trajectory_df with one row having `tip_x = float('inf')` (the foundation's `CircumnutationInputs` validator does not check finiteness per row, per its docstring at `_types.py:60`)
- **WHEN** `kinematics.compute(trajectory_df)` is called
- **THEN** no exception is raised
- **AND** the trait values for that track are either `inf` or `NaN` (propagation-dependent), reflecting the documented "Tier 0 does not validate ±inf" non-goal

#### Scenario: Duplicate `(track_id, frame)` rows do not raise and propagate non-finite values opportunistically
- **GIVEN** a trajectory_df where the same `(track_id, frame)` 2-tuple appears in two rows (e.g., the upstream join double-emitted a frame)
- **WHEN** `kinematics.compute(trajectory_df)` is called
- **THEN** no exception is raised
- **AND** the resulting `Δframe = 0` divide-by-zero MAY produce non-finite (`inf` or `NaN`) values in velocity-related traits — whether the non-finite value reaches the final trait depends on the projection (`[0, 0] / 0 = NaN`; `[nonzero, ·] / 0 = ±inf`) AND on whether the corrupted step survives the `np.nanmedian` aggregation. Tier 0 does NOT guarantee that the contamination is observable in the emitted traits; it guarantees only that no exception is raised. PR #3 QC's `frac_outlier_steps` is the right place for explicit duplicate-frame detection.
- **AND** the documented behavior is "Tier 0 does not detect duplicate frames — PR #3 QC's `frac_outlier_steps` is the right place"

### Requirement: Per-plant template helper for raw-DataFrame callers
The system SHALL expose a private helper `_build_per_plant_template_from_df(df: pd.DataFrame) -> pd.DataFrame` in `sleap_roots/circumnutation/_io.py`. The helper SHALL implement the same drop-duplicates + sort + dtype-coercion logic that the existing public `build_per_plant_template(inputs: CircumnutationInputs)` function uses. The public function SHALL be refactored to a one-line wrapper that returns `_build_per_plant_template_from_df(inputs.trajectory_df)`.

The helper exists so tier modules whose canonical signature takes a raw `trajectory_df` (today: `kinematics.compute`; later: `qc.compute`, `parametric.compute`) can compose the row-identity template without wrapping the DataFrame in a `CircumnutationInputs` purely to satisfy an API. The helper SHALL enforce the same validations the public function enforces — `track_id` integer-coercible (raise `ValueError` naming the field if NaN), same 5-tuple-conflict check on `timepoint` / `genotype` / `treatment` with the same error message format.

The helper's existence and behavior SHALL be tested via direct import; the public `build_per_plant_template` SHALL continue to pass all of its existing foundation tests unchanged (regression preserved).

#### Scenario: Helper produces identical output to public wrapper
- **GIVEN** a `CircumnutationInputs` instance and `inputs.trajectory_df` (a `pd.DataFrame`)
- **WHEN** `_build_per_plant_template_from_df(inputs.trajectory_df)` and `build_per_plant_template(inputs)` are both called
- **THEN** the two return values are column-for-column equal (via `pandas.DataFrame.equals`)

#### Scenario: Helper enforces the integer `track_id` constraint with NaN
- **GIVEN** a raw trajectory `pd.DataFrame` whose `track_id` column contains `NaN` in at least one row
- **WHEN** `_build_per_plant_template_from_df(df)` is called
- **THEN** a `ValueError` is raised
- **AND** the exception message names the `track_id` field (matching the public function's behavior)

#### Scenario: Helper enforces the 5-tuple-conflict check
- **GIVEN** a raw trajectory `pd.DataFrame` where the same `(series, sample_uid, plate_id, plant_id, track_id)` 5-tuple has different `genotype` values across frames
- **WHEN** `_build_per_plant_template_from_df(df)` is called
- **THEN** a `ValueError` is raised
- **AND** the exception message names the offending 5-tuple AND the column whose values conflict (matching the public function's behavior)

#### Scenario: Public wrapper preserves foundation API
- **WHEN** a caller invokes `build_per_plant_template(inputs)` for any valid `CircumnutationInputs`
- **THEN** the return value is identical (column-for-column equal) to the return value of the same call BEFORE this PR's refactor — i.e., all existing foundation tests on `build_per_plant_template` pass without modification

