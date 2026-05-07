## ADDED Requirements

### Requirement: Package layout
The system SHALL provide a `sleap_roots.circumnutation` package whose import-tree is complete from this PR onward — i.e., every module name referenced anywhere in `docs/circumnutation/roadmap.md` exists as an importable Python module from PR #1 forward, even if its functions raise `NotImplementedError` until later PRs land. The package SHALL contain:

- `__init__.py` (public API)
- `_constants.py` (named constants)
- `_types.py` (data classes)
- `_io.py` (calibration + units-sidecar I/O)
- Stub modules `kinematics`, `qc`, `synthetic`, `temporal_cwt`, `psi_g`, `midline`, `spatial_cwt`, `parametric`, `plotting`, `pipeline` whose top-level callables raise `NotImplementedError("PR #N — see docs/circumnutation/roadmap.md")`.

#### Scenario: All stub modules import cleanly
- **WHEN** a user runs `import sleap_roots.circumnutation as c; import sleap_roots.circumnutation.kinematics, sleap_roots.circumnutation.qc, sleap_roots.circumnutation.synthetic, sleap_roots.circumnutation.temporal_cwt, sleap_roots.circumnutation.psi_g, sleap_roots.circumnutation.midline, sleap_roots.circumnutation.spatial_cwt, sleap_roots.circumnutation.parametric, sleap_roots.circumnutation.plotting, sleap_roots.circumnutation.pipeline`
- **THEN** every import succeeds
- **AND** no exception is raised at module-import time

#### Scenario: Calling a stub raises NotImplementedError
- **GIVEN** the package imported as in the previous scenario
- **WHEN** the user calls any top-level callable in a stub module (e.g. `kinematics.compute(...)`)
- **THEN** `NotImplementedError` is raised
- **AND** the exception message contains the substring `"PR #"` and the substring `"docs/circumnutation/roadmap.md"`

### Requirement: CircumnutationInputs data class
The system SHALL provide an `attrs`-based `CircumnutationInputs` class capturing `(trajectory_df: pd.DataFrame, px_per_mm: Optional[float], cadence_s: float, R_mm: Optional[float], run_id: Optional[str])`. It SHALL validate at construction: `trajectory_df` contains the row-identity columns required by the trait CSV schema (Requirement: Row-identity schema below); `cadence_s > 0`; if `px_per_mm` is set, `px_per_mm > 0`; if `R_mm` is set, `R_mm > 0`. It SHALL be re-exported from `sleap_roots/__init__.py`.

#### Scenario: Valid construction
- **GIVEN** a DataFrame containing all eight row-identity columns and `cadence_s = 300.0`
- **WHEN** `CircumnutationInputs(trajectory_df=df, cadence_s=300.0, px_per_mm=47.24, R_mm=0.05, run_id="plate_001")` is called
- **THEN** the instance is created without exception

#### Scenario: Missing row-identity column
- **GIVEN** a DataFrame missing the `plate_id` column
- **WHEN** `CircumnutationInputs(...)` is called
- **THEN** a `ValueError` is raised
- **AND** the exception message names the missing column

#### Scenario: Importable from top-level
- **WHEN** a user runs `from sleap_roots import CircumnutationInputs`
- **THEN** the import succeeds

### Requirement: Trait CSV row-identity schema
Every per-plant trait CSV SHALL begin with the eight columns `(series, sample_uid, timepoint, plate_id, plant_id, track_id, genotype, treatment)` in that order, ahead of any trait columns. Today `plant_id` and `track_id` SHALL be populated identically; both columns SHALL exist so that future divergence is non-breaking. `genotype` and `treatment` SHALL be populated from Series-level metadata where available (the `series-metadata` capability), NaN otherwise. The DataFrame SHALL be sorted lexicographically by `(series, sample_uid, plate_id, plant_id, track_id)`.

#### Scenario: Schema columns exist with correct dtypes
- **GIVEN** a DataFrame produced by the foundation's CSV-row builder for a `CircumnutationInputs` containing 6 tracks
- **THEN** the first 8 columns are `series`, `sample_uid`, `timepoint`, `plate_id`, `plant_id`, `track_id`, `genotype`, `treatment`
- **AND** `track_id` has integer dtype
- **AND** `plant_id` is equal column-wise to `track_id`
- **AND** rows are sorted lexicographically by `(series, sample_uid, plate_id, plant_id, track_id)`

### Requirement: Calibration contract
The pipeline SHALL accept `px_per_mm: Optional[float]` per-call. When `px_per_mm` is `None` or `NaN`, every `[mm]` trait emitted in any tier SHALL be `NaN` AND the QC trait `calibration_present` SHALL be `False`. When `px_per_mm` is a positive finite float, `[mm]` traits SHALL convert from internal pixel units at trait-emission time only; internal CWT, ridge extraction, and derivative computations SHALL operate in pixels. The serialization pattern (sidecar JSON vs CSV header vs attrs metadata vs per-row column) SHALL match exactly what `TrackedTipPipeline` uses, so a single downstream loader can read both pipelines' outputs.

#### Scenario: Calibration provided
- **GIVEN** `CircumnutationInputs(..., px_per_mm=47.24)` and a future trait emission stub returning a `[mm]` trait
- **WHEN** the foundation's `_io.write_per_plant_csv(...)` writes the CSV and units sidecar
- **THEN** the units sidecar JSON contains `px_per_mm: 47.24` (or whatever path matches `TrackedTipPipeline`'s pattern)
- **AND** loading the CSV via the existing `TrackedTipPipeline`-output loader recovers `px_per_mm = 47.24`

#### Scenario: Calibration omitted
- **GIVEN** `CircumnutationInputs(..., px_per_mm=None)`
- **WHEN** any tier's trait function emits a `[mm]` trait
- **THEN** the value is `NaN`
- **AND** the QC trait `calibration_present` in the same row is `False`
- **AND** the run completes without raising an exception

### Requirement: Units sidecar JSON
For every per-plant trait CSV the system SHALL write a sibling `traits_per_plant.units.json` mapping each column name to a unit string. Numeric trait columns MUST be present in the mapping. Boolean and string columns MAY be present with empty unit strings. The unit-string vocabulary is `{"mm", "mm/hr", "hr", "rad", "px", "px/frame", "px/hr", "—"}`.

#### Scenario: Units sidecar exists and is valid
- **WHEN** the foundation writes a CSV
- **THEN** a sibling `traits_per_plant.units.json` exists in the same directory
- **AND** it parses as valid JSON
- **AND** every numeric column from the CSV is a key in the JSON mapping
- **AND** every value is a string in the documented vocabulary

### Requirement: Module-level constants
The system SHALL expose all overridable defaults as module-level named constants in `sleap_roots/circumnutation/_constants.py`. The constant set SHALL include at minimum: `NOISE_MASK_K`, `LGZ_STEADY_STATE_RESIDUAL_MAX`, `NYQUIST_RATIO_MAX`, `SG_D2_AGREEMENT_MAX`, `LGZ_NMIN_RESOLVABLE`, `COI_FRACTION_MAX`, `BAND_POWER_NOISE_RATIO`, `WAVELET_DEFAULT_TEMPORAL`, `WAVELET_DEFAULT_SPATIAL`, `SG_WINDOW_SHORT`, `SG_DEGREE`, `SG_WINDOW_DETREND`, `OUTLIER_STEP_RATIO`, `GROWTH_AXIS_RELIABILITY_K`. The values SHALL match the defaults in `docs/circumnutation/roadmap.md` cross-cutting concern CC-2.

#### Scenario: All required constants are importable
- **WHEN** a user runs `from sleap_roots.circumnutation import _constants`
- **THEN** every name listed above is an attribute of `_constants`
- **AND** each value matches the documented default in `roadmap.md` CC-2

### Requirement: Per-module logger convention
Every module in `sleap_roots/circumnutation/` SHALL declare a module-level logger via `logger = logging.getLogger(__name__)`. The package SHALL NOT call `logging.basicConfig` or otherwise configure handlers at import time.

#### Scenario: Module loggers are namespaced
- **GIVEN** `import logging; import sleap_roots.circumnutation.kinematics`
- **THEN** `logging.getLogger("sleap_roots.circumnutation.kinematics")` returns the same logger object that the module uses
- **AND** the package import does not add handlers to the root logger
