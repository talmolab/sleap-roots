## ADDED Requirements

### Requirement: Package layout
The system SHALL provide a `sleap_roots.circumnutation` sub-package whose import-tree is complete from this PR onward — every module name referenced anywhere in `docs/circumnutation/roadmap.md` exists as an importable Python module from PR #1 forward, even if its functions raise `NotImplementedError` until later PRs land. The package SHALL contain:

- 5 contract modules: `__init__.py`, `_constants.py`, `_types.py`, `_io.py`, `units.py`
- 10 stub modules: `kinematics`, `qc`, `synthetic`, `temporal_cwt`, `psi_g`, `midline`, `spatial_cwt`, `parametric`, `plotting`, `pipeline`

Each stub module SHALL define exactly one canonical callable per the table below, raising `NotImplementedError(f"PR #{N} — see docs/circumnutation/roadmap.md")` with the documented PR number. Each stub callable SHALL also have a complete Google-style docstring (Args, Returns, Raises) so `pydocstyle --convention=google` and the mkdocstrings auto-generated reference pages remain informative. Stubs whose tier PR will compose with the typed `ConstantsT` override-bag SHALL include `constants=None` as a forward-compatible keyword parameter so callers do not get `TypeError` before `NotImplementedError`.

| Stub module | Canonical callable | PR # |
|---|---|---|
| `kinematics` | `compute(trajectory_df, constants=None)` | 2 |
| `qc` | `compute(trajectory_df, constants=None)` | 3 |
| `synthetic` | `generate_trajectory(...)` (pure-pixel; no `px_per_mm`) | 4 |
| `temporal_cwt` | `compute_scaleogram(x, cadence_s, constants=None)` | 5 |
| `psi_g` | `compute_psi_g(x, y, constants=None)` | 7 |
| `midline` | `reconstruct(x, y, cadence_s, constants=None)` | 8 |
| `spatial_cwt` | `compute_scaleogram(kappa, ds, constants=None)` | 9 |
| `parametric` | `compute(tier3_df, R_px, omega, Delta_phi)` | 11 |
| `plotting` | `scaleogram(scaleogram_result, out_path)` | 16 |
| `pipeline` | `compute_traits(inputs, constants=None)` | 14 |

#### Scenario: All stub modules import cleanly
- **WHEN** a user runs `import sleap_roots.circumnutation as c; import sleap_roots.circumnutation.kinematics, sleap_roots.circumnutation.qc, sleap_roots.circumnutation.synthetic, sleap_roots.circumnutation.temporal_cwt, sleap_roots.circumnutation.psi_g, sleap_roots.circumnutation.midline, sleap_roots.circumnutation.spatial_cwt, sleap_roots.circumnutation.parametric, sleap_roots.circumnutation.plotting, sleap_roots.circumnutation.pipeline`
- **THEN** every import succeeds
- **AND** no exception is raised at module-import time

#### Scenario: Calling each stub raises NotImplementedError with the correct PR number
- **GIVEN** the package imported as in the previous scenario
- **WHEN** the canonical callable in each stub module is invoked (parameters per the table above; NotImplementedError fires before any argument check)
- **THEN** `NotImplementedError` is raised
- **AND** the exception message matches the regex `r"^PR #\d+ — see docs/circumnutation/roadmap\.md$"`
- **AND** the captured PR number equals the one in the table for that module

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
