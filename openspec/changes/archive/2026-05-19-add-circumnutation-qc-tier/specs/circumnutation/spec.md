# Spec delta — add-circumnutation-qc-tier

## MODIFIED Requirements

### Requirement: Package layout
The system SHALL provide a `sleap_roots.circumnutation` sub-package whose import-tree is complete from this PR onward — every module name referenced anywhere in `docs/circumnutation/roadmap.md` exists as an importable Python module from PR #1 forward, even if its functions raise `NotImplementedError` until later PRs land. The package SHALL contain:

- 7 contract modules: `__init__.py`, `_constants.py`, `_types.py`, `_io.py`, `units.py`, `_noise.py`, `_geometry.py`
- 2 implementation modules: `kinematics` (implemented from PR #2 onward; see Requirement: Tier 0 raw kinematic traits) and `qc` (implemented from PR #3 onward; see Requirement: QC tier per-track quality traits)
- 8 stub modules: `synthetic`, `temporal_cwt`, `psi_g`, `midline`, `spatial_cwt`, `parametric`, `plotting`, `pipeline`

Each stub module SHALL define exactly one canonical callable per the table below, raising `NotImplementedError(f"PR #{N} — see docs/circumnutation/roadmap.md")` with the documented PR number. Each stub callable SHALL also have a complete Google-style docstring (Args, Returns, Raises) so `pydocstyle --convention=google` and the mkdocstrings auto-generated reference pages remain informative. Stubs whose tier PR will compose with the typed `ConstantsT` override-bag SHALL include `constants=None` as a forward-compatible keyword parameter so callers do not get `TypeError` before `NotImplementedError`.

| Stub module | Canonical callable | PR # |
|---|---|---|
| `synthetic` | `generate_trajectory(...)` (pure-pixel; no `px_per_mm`) | 4 |
| `temporal_cwt` | `compute_scaleogram(x, cadence_s, constants=None)` | 5 |
| `psi_g` | `compute_psi_g(x, y, constants=None)` | 7 |
| `midline` | `reconstruct(x, y, cadence_s, constants=None)` | 8 |
| `spatial_cwt` | `compute_scaleogram(kappa, ds, constants=None)` | 9 |
| `parametric` | `compute(tier3_df, R_px, omega, Delta_phi)` | 11 |
| `plotting` | `scaleogram(scaleogram_result, out_path)` | 16 |
| `pipeline` | `compute_traits(inputs, constants=None)` | 14 |

The `kinematics` module SHALL be importable on the same terms as the other modules (clean import, namespaced logger) and SHALL expose `compute(trajectory_df, constants=None)` per Requirement: Tier 0 raw kinematic traits. The `qc` module SHALL be importable on the same terms and SHALL expose `compute(trajectory_df, constants=None)` per Requirement: QC tier per-track quality traits. Unlike the stub modules, calling `kinematics.compute` or `qc.compute` with a valid input SHALL NOT raise `NotImplementedError`.

The `_noise` and `_geometry` modules are private (underscore-prefixed) shared internals; their canonical callables are documented under Requirement: Tier 0 helper modules.

#### Scenario: All stub modules import cleanly
- **WHEN** a user runs `import sleap_roots.circumnutation as c; import sleap_roots.circumnutation.kinematics, sleap_roots.circumnutation.qc, sleap_roots.circumnutation.synthetic, sleap_roots.circumnutation.temporal_cwt, sleap_roots.circumnutation.psi_g, sleap_roots.circumnutation.midline, sleap_roots.circumnutation.spatial_cwt, sleap_roots.circumnutation.parametric, sleap_roots.circumnutation.plotting, sleap_roots.circumnutation.pipeline`
- **THEN** every import succeeds
- **AND** no exception is raised at module-import time

#### Scenario: Calling each remaining stub raises NotImplementedError with the correct PR number
- **GIVEN** the package imported as in the previous scenario
- **WHEN** the canonical callable in each of the 8 remaining stub modules (`synthetic`, `temporal_cwt`, `psi_g`, `midline`, `spatial_cwt`, `parametric`, `plotting`, `pipeline`) is invoked (parameters per the table above; `NotImplementedError` fires before any argument check)
- **THEN** `NotImplementedError` is raised
- **AND** the exception message matches the regex `r"^PR #\d+ — see docs/circumnutation/roadmap\.md$"`
- **AND** the captured PR number equals the one in the table for that module

#### Scenario: `kinematics.compute` no longer raises NotImplementedError
- **GIVEN** a valid `trajectory_df` (8 row-identity columns + `frame`, `tip_x`, `tip_y`; ≥ 1 row)
- **WHEN** `sleap_roots.circumnutation.kinematics.compute(trajectory_df)` is invoked
- **THEN** the call returns a `pandas.DataFrame` (the Tier 0 per-plant output) without raising `NotImplementedError`

#### Scenario: `qc.compute` no longer raises NotImplementedError
- **GIVEN** a valid `trajectory_df` (8 row-identity columns + `frame`, `tip_x`, `tip_y`; ≥ 1 row)
- **WHEN** `sleap_roots.circumnutation.qc.compute(trajectory_df)` is invoked
- **THEN** the call returns a `pandas.DataFrame` (the QC tier per-plant output) without raising `NotImplementedError`

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

### Requirement: Module-level constants
The system SHALL expose all overridable defaults as module-level named constants in `sleap_roots/circumnutation/_constants.py`. The set SHALL include at minimum: `NOISE_MASK_K`, `LGZ_STEADY_STATE_RESIDUAL_MAX`, `NYQUIST_RATIO_MAX`, `SG_D2_AGREEMENT_MAX`, `SG_MSD_AGREEMENT_MAX`, `D2_MSD_AGREEMENT_MAX`, `LGZ_NMIN_RESOLVABLE`, `COI_FRACTION_MAX`, `BAND_POWER_NOISE_RATIO`, `WAVELET_DEFAULT_TEMPORAL`, `WAVELET_DEFAULT_SPATIAL`, `SG_WINDOW_SHORT`, `SG_DEGREE`, `SG_WINDOW_DETREND`, `OUTLIER_STEP_RATIO`, `FRAC_OUTLIER_STEPS_MAX`, `WORST_STEP_RATIO_MAX`, `GROWTH_AXIS_RELIABILITY_K`, `_SCHEMA_VERSION`, `_CONSTANTS_VERSION`. The values SHALL match the defaults in `docs/circumnutation/roadmap.md` cross-cutting concern CC-2 and `docs/circumnutation/theory.md` §7.6 (for the QC-tier-introduced thresholds: `FRAC_OUTLIER_STEPS_MAX = 0.05`, `WORST_STEP_RATIO_MAX = 5`, `SG_MSD_AGREEMENT_MAX = 1.5`, `D2_MSD_AGREEMENT_MAX = 1.5`); `_SCHEMA_VERSION` SHALL be `1` (unchanged from PR #1) and `_CONSTANTS_VERSION` SHALL be `2` (bumped from `1` in this PR per the version-sentinel contract — the constants set grew). The module SHALL also expose `PIPELINE_UNIT_VOCABULARY` (px-based + calibration-independent units, the closed sidecar vocabulary), `CONVERTED_UNIT_VOCABULARY` (mm-based units produced by `convert_to_mm`), and `VALID_UNIT_VOCABULARY` (their union), plus `ROW_IDENTITY_UNITS` (the canonical units dict for the eight row-identity columns).

The `ConstantsT` typed override-bag SHALL include corresponding fields for every overridable constant above, so callers can override per-call via `ConstantsT(FRAC_OUTLIER_STEPS_MAX=0.1)` etc. `_default_constants_snapshot()` SHALL emit every constant name in the set above into the run-metadata sidecar, including the four new QC-tier constants.

#### Scenario: All required constants are importable with correct types
- **WHEN** a user runs `from sleap_roots.circumnutation import _constants`
- **THEN** every name listed above is an attribute of `_constants`
- **AND** each value matches the documented default in `roadmap.md` CC-2 and `theory.md` §7.6
- **AND** `_constants._SCHEMA_VERSION` is the integer `1`
- **AND** `_constants._CONSTANTS_VERSION` is the integer `2`
- **AND** `_constants.PIPELINE_UNIT_VOCABULARY`, `_constants.CONVERTED_UNIT_VOCABULARY`, `_constants.VALID_UNIT_VOCABULARY`, `_constants.ROW_IDENTITY_UNITS` are all importable

#### Scenario: New QC-tier constants are overridable via ConstantsT
- **GIVEN** a custom `ConstantsT(FRAC_OUTLIER_STEPS_MAX=0.10, WORST_STEP_RATIO_MAX=10, SG_MSD_AGREEMENT_MAX=2.0, D2_MSD_AGREEMENT_MAX=2.0)`
- **WHEN** the instance is inspected
- **THEN** each overridden field reflects its caller-supplied value
- **AND** unoverridden fields reflect their module-level defaults

#### Scenario: Constants snapshot includes the four new QC constants
- **WHEN** `_default_constants_snapshot()` is called
- **THEN** the returned mapping contains `FRAC_OUTLIER_STEPS_MAX`, `WORST_STEP_RATIO_MAX`, `SG_MSD_AGREEMENT_MAX`, `D2_MSD_AGREEMENT_MAX` with their default values

### Requirement: Tier 0 helper modules
The system SHALL provide two private helper modules in `sleap_roots/circumnutation/` to host computations shared across Tier 0 (PR #2), the QC tier (PR #3), and future tier PRs (#7):

**`_noise.py`** SHALL define three callables for SLEAP-localization-noise estimation, each deterministic (identical inputs → identical outputs):

1. `compute_sg_residual_xy(x: np.ndarray, y: np.ndarray, window: int, degree: int) -> float`. The function SHALL apply `scipy.signal.savgol_filter` to `x` and `y` independently using the given `window` and `degree`, compute the standard deviation of residuals for each, and return their quadrature sum: `sqrt(std(x - x_smooth)^2 + std(y - y_smooth)^2)`. When `len(x) < window`, the function SHALL return `np.nan` and log a `DEBUG` record naming the short-input case rather than raising.

2. `compute_d2_residual_xy(x: np.ndarray, y: np.ndarray) -> float`. The function SHALL compute `delta2_x = x[2:] - 2*x[1:-1] + x[:-2]` (second-difference array) and similarly `delta2_y`, then return `sqrt(std(delta2_x)**2 + std(delta2_y)**2) / sqrt(6)`. The `1/sqrt(6)` normalization derives from the white-noise propagation rule `Var(Δ²x) = 6σ²` (theory.md §7.6 / preliminary_results.md §3.3). When `len(x) < 3`, the function SHALL return `np.nan` and log a `DEBUG` record naming the short-input case rather than raising.

3. `compute_msd_residual_xy(x: np.ndarray, y: np.ndarray, window: int, degree: int, lag: int = 1) -> float`. The function SHALL first SG-detrend `x` and `y` separately using `scipy.signal.savgol_filter(..., window_length=window, polyorder=degree)`, compute residuals `x_res = x - x_smooth` (and similarly `y_res`), then compute the 2D MSD at the given `lag` as `msd = mean((x_res[lag:] - x_res[:-lag])**2 + (y_res[lag:] - y_res[:-lag])**2)`, and return `sqrt(msd / 4.0)`. The factor of 4 (NOT 2) is the 2D MSD ↔ σ² relationship `MSD(τ→0) = 4σ²` from Michalet 2010 `Phys. Rev. E` 82:041914 / theory.md §7.6. When `len(x) < window + lag`, the function SHALL return `np.nan` and log a `DEBUG` record naming the short-input case rather than raising.

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

#### Scenario: `_noise.compute_d2_residual_xy` returns zero for a linear signal
- **GIVEN** `x = np.linspace(0, 99, 100)` (perfectly linear; second differences are identically zero) and `y = np.zeros(100)`
- **WHEN** `compute_d2_residual_xy(x, y)` is called
- **THEN** the return is `0.0` within IEEE float tolerance

#### Scenario: `_noise.compute_d2_residual_xy` recovers approximate σ on noisy data
- **GIVEN** `x_smooth = np.linspace(0, 100, 1000)`, `y_smooth = np.zeros(1000)`, `noise = np.random.default_rng(0).normal(0, 1.0, size=(2, 1000))`, `x = x_smooth + noise[0]`, `y = y_smooth + noise[1]`
- **WHEN** `compute_d2_residual_xy(x, y)` is called
- **THEN** the return is within `[1.2, 1.8]` (the d2 estimator is unbiased for i.i.d. noise on a linear signal: `std(Δ²x) = sqrt(6)·σ_x`, so the quadrature sum `sqrt(std(Δ²x)² + std(Δ²y)²) / sqrt(6) = sqrt(σ_x² + σ_y²) = sqrt(2) ≈ 1.41`)

#### Scenario: `_noise.compute_d2_residual_xy` returns NaN for short inputs
- **WHEN** `compute_d2_residual_xy(np.array([1.0, 2.0]), np.array([3.0, 4.0]))` is called
- **THEN** the return is `np.nan`
- **AND** a `DEBUG` log record is emitted naming the short-input case
- **AND** no exception is raised

#### Scenario: `_noise.compute_msd_residual_xy` returns approximately zero for a smooth signal
- **GIVEN** `x = np.linspace(0, 99, 100)` (smooth linear; SG-detrend residual is ≈ 0) and `y = np.zeros(100)` and `window=5, degree=3, lag=1`
- **WHEN** `compute_msd_residual_xy(x, y, window=5, degree=3, lag=1)` is called
- **THEN** the return is `≤ 1e-6` (SG residual is numerically zero; MSD of ~zero residuals is ~zero; σ ≈ 0)

#### Scenario: `_noise.compute_msd_residual_xy` recovers approximate σ on noisy data
- **GIVEN** `x_smooth = np.linspace(0, 100, 1000)`, `y_smooth = np.zeros(1000)`, `noise = np.random.default_rng(0).normal(0, 1.0, size=(2, 1000))`, `x = x_smooth + noise[0]`, `y = y_smooth + noise[1]`
- **WHEN** `compute_msd_residual_xy(x, y, window=5, degree=3, lag=1)` is called
- **THEN** the return is within `[1.0, 2.0]` (MSD-extrapolation at lag=1 of SG-detrended residuals; for i.i.d. unit-σ noise on independent x and y, `MSD(τ=1) = 4σ² = 4` so `σ_MSD = sqrt(4/4) = 1`; tolerance allows for SG-detrend slight under-estimate and lag-1 stochastic variation)

#### Scenario: `_noise.compute_msd_residual_xy` returns NaN for short inputs
- **WHEN** `compute_msd_residual_xy(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), np.array([0.0, 0.0, 0.0, 0.0, 0.0]), window=5, degree=3, lag=1)` is called (len=5 < window+lag = 6)
- **THEN** the return is `np.nan`
- **AND** a `DEBUG` log record is emitted naming the short-input case
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

### Requirement: Growth-axis reliability gate
For each track, the system SHALL compute net displacement `D = ‖xy[-1] − xy[0]‖` (in pixels) and a local SG-residual noise estimate via `_noise.compute_sg_residual_xy(x, y, window=constants.SG_WINDOW_SHORT, degree=constants.SG_DEGREE)`. It SHALL set `growth_axis_unreliable = (D < constants.GROWTH_AXIS_RELIABILITY_K * sg_residual_xy_local)` — strict less-than. The threshold multiplier is configurable via `ConstantsT.GROWTH_AXIS_RELIABILITY_K` with documented default `10`.

When `growth_axis_unreliable == True`, the 6 rotation-dependent trait columns (`v_long_signed_median_px_per_frame`, `v_long_abs_median_px_per_frame`, `v_lat_signed_median_px_per_frame`, `v_lat_abs_median_px_per_frame`, `long_lat_ratio`, `principal_axis_angle`) SHALL be set to `NaN`. The 3 rotation-invariant traits (`v_total_median_px_per_frame`, `path_displacement_ratio`, `angular_amplitude`) SHALL NOT be NaN'd by the gate (they may still be NaN for other documented reasons — e.g. `path_displacement_ratio` when `D == 0`, or all traits when fewer than 2 valid frames).

Both the Tier 0 module AND the QC tier module SHALL emit the `growth_axis_unreliable` column on their respective per-plant trait DataFrames. The two emissions SHALL be element-wise equal as `bool` dtype by construction, because both tiers compute the flag via the same formula on the same inputs through the shared `_noise.compute_sg_residual_xy` helper after applying the same `xy = subset[["tip_x","tip_y"]].to_numpy(dtype=float)` cast. PR #14 pipeline composition MAY coalesce or drop one column (either choice is safe because they are equal). This rule SUPERSEDES the previous "sole emitter" wording from PR #2 (rationale: QC's standalone usefulness requires the bool column for downstream `df[~df.growth_axis_unreliable]` filtering — see PR #3 design.md D5). The equality contract is governed by Requirement: QC tier growth_axis_unreliable equality with Tier 0.

The local SG residual computation SHALL use the same `_noise.compute_sg_residual_xy` helper that the QC tier uses to emit the canonical `sg_residual_xy` trait, so the gate value and the canonical trait are guaranteed-identical from identical inputs.

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

#### Scenario: Both tiers emit the column with equal values
- **GIVEN** a valid `trajectory_df` (any synthetic or real fixture)
- **WHEN** both `kinematics.compute(df)` and `qc.compute(df)` are invoked with default constants
- **THEN** both returned DataFrames contain a `growth_axis_unreliable` column
- **AND** both columns are `bool` dtype with no NaN values
- **AND** the two columns are element-wise equal (`(kinematics_result["growth_axis_unreliable"] == qc_result["growth_axis_unreliable"]).all()` is True)

## ADDED Requirements

### Requirement: QC tier per-track quality traits
The system SHALL provide `sleap_roots.circumnutation.qc.compute(trajectory_df: pd.DataFrame, constants: Optional[ConstantsT] = None) -> pd.DataFrame`. The function SHALL accept the canonical `(trajectory_df, constants=None)` signature locked by the foundation's Package layout requirement — `cadence_s` SHALL NOT appear in the signature (the function emits cadence-independent traits only).

It SHALL return a per-plant `pandas.DataFrame` with one row per unique `(series, sample_uid, plate_id, plant_id, track_id)` 5-tuple in the input, sorted via the same convention as `_io._build_per_plant_template_from_df`. The DataFrame columns SHALL be (in this order):

1. The 8 row-identity columns in their declared order: `series`, `sample_uid`, `timepoint`, `plate_id`, `plant_id`, `track_id`, `genotype`, `treatment`.
2. The 11 QC trait columns, in this order:

| Column | Unit | Definition |
|---|---|---|
| `sg_residual_xy` | `px` | `_noise.compute_sg_residual_xy(x, y, window=constants.SG_WINDOW_SHORT, degree=constants.SG_DEGREE)` |
| `d2_noise_xy` | `px` | `_noise.compute_d2_residual_xy(x, y)` |
| `msd_noise_xy` | `px` | `_noise.compute_msd_residual_xy(x, y, window=constants.SG_WINDOW_SHORT, degree=constants.SG_DEGREE, lag=1)` |
| `sg_d2_agreement` | `—` | `max(sg_residual_xy, d2_noise_xy) / min(sg_residual_xy, d2_noise_xy)`; NaN if either operand is NaN |
| `sg_msd_agreement` | `—` | `max(sg_residual_xy, msd_noise_xy) / min(sg_residual_xy, msd_noise_xy)`; NaN if either operand is NaN |
| `d2_msd_agreement` | `—` | `max(d2_noise_xy, msd_noise_xy) / min(d2_noise_xy, msd_noise_xy)`; NaN if either operand is NaN |
| `frac_outlier_steps` | `—` | `count(\|Δxy_i\| > constants.OUTLIER_STEP_RATIO · median(\|Δxy_i\|)) / n_steps` over consecutive present frames (gap-aware); NaN when `median(\|Δxy_i\|) == 0` (stationary track) |
| `worst_step_ratio` | `—` | `max(\|Δxy_i\|) / median(\|Δxy_i\|)`; NaN when `median(\|Δxy_i\|) == 0` |
| `growth_axis_unreliable` | `bool` | True iff the growth axis is unreliable per Requirement: QC tier growth_axis_unreliable equality with Tier 0 |
| `track_is_clean` | `bool` | composite per Requirement: QC tier track_is_clean and qc_failure_reason composition |
| `qc_failure_reason` | `string` | stable-ordered comma-separated failure clauses per the same requirement; `""` when `track_is_clean == True` |

The function SHALL emit only units in `PIPELINE_UNIT_VOCABULARY` — specifically `px` for the 3 noise estimators, `—` for the 5 dimensionless ratios/fractions, `bool` for `growth_axis_unreliable` and `track_is_clean`, `string` for `qc_failure_reason`. NO `_mm`-suffixed columns and NO `_px_per_hr` columns SHALL be emitted by `qc.compute`.

The function SHALL process each track via the algorithm documented in `design.md` D7: (a) drop NaN rows on `tip_x` / `tip_y` BEFORE any `np.diff` or arithmetic (NaN-then-sort ordering matching Tier 0); (b) sort the remaining rows by `frame`; (c) ALWAYS compute `growth_axis_unreliable` via the same `_noise.compute_sg_residual_xy` helper and gate formula as Tier 0 (preserving the equality contract — see Requirement: QC tier growth_axis_unreliable equality with Tier 0); (d) if fewer than `constants.SG_WINDOW_SHORT` (default 5) frames remain after NaN-drop, emit the 8 numeric traits + 3 pairwise agreements all as NaN, `track_is_clean = False`, `qc_failure_reason = "qc_inputs_insufficient"` (single sentinel reason, NOT comma-concatenated with other clauses), and skip to the merge step; (e) otherwise compute the 3 noise estimators, 3 pairwise agreements, and 2 outlier-step traits per the table above; (f) compose `track_is_clean` and `qc_failure_reason` per Requirement: QC tier track_is_clean and qc_failure_reason composition.

The xy array SHALL be cast to `dtype=float` explicitly (`xy = subset[["tip_x", "tip_y"]].to_numpy(dtype=float)`) so that input dtype variation (int, float32, object) cannot perturb the equality contract with Tier 0 — both tiers feed identically-typed arrays to `_noise.compute_sg_residual_xy`.

The function SHALL accept `constants=None` (default) by constructing the default `ConstantsT()`, OR a caller-supplied `ConstantsT` override. The constants consumed are `SG_WINDOW_SHORT`, `SG_DEGREE`, `OUTLIER_STEP_RATIO`, `GROWTH_AXIS_RELIABILITY_K`, `SG_D2_AGREEMENT_MAX`, `SG_MSD_AGREEMENT_MAX`, `D2_MSD_AGREEMENT_MAX`, `FRAC_OUTLIER_STEPS_MAX`, `WORST_STEP_RATIO_MAX`.

After per-track computation, the function SHALL merge the per-track trait dictionaries onto the foundation's per-plant template via the 5-tuple key (`_io._build_per_plant_template_from_df`) and re-select columns to enforce the declared 19-column order `ROW_IDENTITY_COLUMNS + _QC_TRAIT_COLUMNS`. This re-selection step is load-bearing: the groupby key uses the 5-tuple `_IDENTITY_5_TUPLE` which omits `timepoint`, but the template carries all 8 row-identity columns; without the final re-selection `timepoint` could drop or shift.

#### Scenario: Output DataFrame columns are in the specified order
- **GIVEN** a valid `trajectory_df` with 6 tracks
- **WHEN** `qc.compute(trajectory_df)` is called
- **THEN** the returned DataFrame has exactly 19 columns
- **AND** the first 8 columns are `series`, `sample_uid`, `timepoint`, `plate_id`, `plant_id`, `track_id`, `genotype`, `treatment` in that order
- **AND** the next 11 columns are `sg_residual_xy`, `d2_noise_xy`, `msd_noise_xy`, `sg_d2_agreement`, `sg_msd_agreement`, `d2_msd_agreement`, `frac_outlier_steps`, `worst_step_ratio`, `growth_axis_unreliable`, `track_is_clean`, `qc_failure_reason` in that order

#### Scenario: Output column units are within `PIPELINE_UNIT_VOCABULARY`
- **GIVEN** the output of `qc.compute(trajectory_df)` and the units-mapping dict for the 11 new columns
- **WHEN** every unit string is checked against `sleap_roots.circumnutation._constants.PIPELINE_UNIT_VOCABULARY`
- **THEN** every unit string is a member of the vocabulary
- **AND** no unit string is `mm`-bearing or `px/hr`-bearing
- **AND** the 3 noise estimators use `px`, the 5 dimensionless traits use `—`, the 2 booleans use `bool`, `qc_failure_reason` uses `string`

#### Scenario: Clean straight-line track yields zero residuals and `track_is_clean == True`
- **GIVEN** a 100-frame track with `tip_x = frame * 1.0`, `tip_y = 0.0` (perfectly linear, no noise)
- **WHEN** `qc.compute(trajectory_df)` is called with default constants
- **THEN** the returned row has `sg_residual_xy ≈ 0.0` and `d2_noise_xy ≈ 0.0` and `msd_noise_xy ≈ 0.0` (smooth signal → all residuals near zero)
- **AND** `frac_outlier_steps == 0.0` (all steps equal — no outliers)
- **AND** `worst_step_ratio ≈ 1.0` (all steps equal)
- **AND** `growth_axis_unreliable == False` (D = 99 ≫ K · residual)
- **AND** `track_is_clean == True`
- **AND** `qc_failure_reason == ""`

#### Scenario: Pure-noise track fires growth_axis_unreliable and `track_is_clean == False`
- **GIVEN** a 100-frame track with `tip_x, tip_y = np.random.default_rng(0).normal(0, 1, size=(2, 100))` (i.i.d. noise around origin, no growth)
- **WHEN** `qc.compute(trajectory_df)` is called with default constants
- **THEN** `growth_axis_unreliable == True` (small D vs SG-residual)
- **AND** `track_is_clean == False`
- **AND** `qc_failure_reason` contains `"growth_axis_unreliable"` as one of its comma-separated clauses

#### Scenario: Short-track gate fires for `len < SG_WINDOW_SHORT`
- **GIVEN** a track with only 3 frames (`len < SG_WINDOW_SHORT = 5`)
- **WHEN** `qc.compute(trajectory_df)` is called
- **THEN** the 8 numeric traits + 3 pairwise agreements are NaN
- **AND** `track_is_clean == False`
- **AND** `qc_failure_reason == "qc_inputs_insufficient"` (literally; NOT comma-concatenated with other clauses even if `growth_axis_unreliable=True` would have fired)

#### Scenario: Single-frame track (n=1) emits NaN traits without raising
- **GIVEN** a `trajectory_df` with a single track that has only 1 row after NaN-drop
- **WHEN** `qc.compute(trajectory_df)` is called
- **THEN** the returned DataFrame has 1 row for that track
- **AND** the 8 numeric traits + 3 pairwise agreements are NaN
- **AND** `growth_axis_unreliable == False` (`D = NaN`, `D == 0.0` is False, `not isnan(NaN)` is False — both gate clauses False)
- **AND** `track_is_clean == False`
- **AND** `qc_failure_reason == "qc_inputs_insufficient"`
- **AND** no exception is raised

#### Scenario: Zero-displacement closed-loop track triggers gate
- **GIVEN** a track of length ≥ 5 whose start and end coordinates are identical (`xy[-1] == xy[0]`)
- **WHEN** `qc.compute(trajectory_df)` is called
- **THEN** `growth_axis_unreliable == True` (D == 0.0)
- **AND** `track_is_clean == False`
- **AND** `qc_failure_reason` contains `"growth_axis_unreliable"`

#### Scenario: Invalid trajectory_df raises ValueError
- **WHEN** `qc.compute(None)` is called
- **THEN** a `ValueError` is raised whose message indicates the input is not a `pandas.DataFrame`
- **AND** `qc.compute(df_missing_tip_x)` similarly raises a `ValueError` naming the missing column
- **AND** `qc.compute(df_missing_plate_id)` raises a `ValueError` naming the missing row-identity column

#### Scenario: ConstantsT override changes per-clause thresholds
- **GIVEN** a `trajectory_df` constructed so that `sg_d2_agreement = 1.7` (above default `SG_D2_AGREEMENT_MAX = 1.5`)
- **WHEN** `qc.compute(trajectory_df)` is called with default constants
- **THEN** `qc_failure_reason` contains `"sg_d2_agreement_high"`
- **AND** when called again with `constants=ConstantsT(SG_D2_AGREEMENT_MAX=2.0)`, `qc_failure_reason` does NOT contain `"sg_d2_agreement_high"` (threshold loosened; clause does not fire)

### Requirement: QC tier track_is_clean and qc_failure_reason composition
The system SHALL compose `track_is_clean` and `qc_failure_reason` per track via the formulas below.

`track_is_clean` SHALL be the AND of six clauses:

```
track_is_clean = (
    NOT growth_axis_unreliable
  AND sg_d2_agreement   < constants.SG_D2_AGREEMENT_MAX   (default 1.5)
  AND sg_msd_agreement  < constants.SG_MSD_AGREEMENT_MAX  (default 1.5)
  AND d2_msd_agreement  < constants.D2_MSD_AGREEMENT_MAX  (default 1.5)
  AND frac_outlier_steps < constants.FRAC_OUTLIER_STEPS_MAX (default 0.05)
  AND worst_step_ratio   < constants.WORST_STEP_RATIO_MAX   (default 5)
)
```

Comparisons against NaN return False in Python/NumPy, so any NaN-bearing trait fires its clause (i.e., `not (NaN < threshold)` evaluates to True). Net effect: any NaN-laden track gets `track_is_clean = False` and gets the relevant clause(s) appended to `qc_failure_reason`.

`qc_failure_reason` SHALL be a string column. When `track_is_clean == True`, the value SHALL be the empty string `""` (NOT NaN). When `track_is_clean == False`, the value SHALL be the comma-separated (`", "` with a single space) list of failure-clause names in the stable canonical order encoded as the module-level `tuple` `_FAILURE_CLAUSE_ORDER`:

```python
_FAILURE_CLAUSE_ORDER: tuple = (
    "qc_inputs_insufficient",       # short-track sentinel (overrides all other clauses)
    "growth_axis_unreliable",
    "sg_d2_agreement_high",
    "sg_msd_agreement_high",
    "d2_msd_agreement_high",
    "frac_outlier_steps_high",
    "worst_step_ratio_high",
)
```

`qc_inputs_insufficient` SHALL be treated as a sentinel, NOT as a regular clause that can co-occur with others. When the short-track gate fires (per Requirement: QC tier per-track quality traits, step d), `qc_failure_reason` SHALL be LITERALLY `"qc_inputs_insufficient"` with no other clauses appended even if `growth_axis_unreliable = True` would have fired. The other 6 clauses MAY co-occur and SHALL concatenate via `", ".join(...)` in the declared tuple order.

#### Scenario: All clauses clean → empty failure reason
- **GIVEN** a track where `track_is_clean == True` (all 6 clauses satisfied)
- **WHEN** `qc.compute(trajectory_df)` is called
- **THEN** `qc_failure_reason == ""` (empty string, NOT NaN)

#### Scenario: Single clause failure → single-clause failure reason
- **GIVEN** a track where exactly one of the 6 clauses fails (e.g., `worst_step_ratio = 7.0 > 5`)
- **WHEN** `qc.compute(trajectory_df)` is called
- **THEN** `qc_failure_reason == "worst_step_ratio_high"`
- **AND** `track_is_clean == False`

#### Scenario: Multiple clause failures → comma-separated reason in stable order
- **GIVEN** a track constructed to fire three clauses simultaneously. Construction recipe (parallel to PR #2's dual-recipe pattern): take a 30-frame track with `tip_x = np.random.default_rng(0).normal(0, 1, 30)`, `tip_y = np.random.default_rng(1).normal(0, 1, 30)` (pure-noise, no growth → `growth_axis_unreliable=True`); inject 4 outlier frames (`tip_x[10:14] = 100`) so `frac_outlier_steps > 0.05`; the noise structure + outliers also pushes `sg_d2_agreement` above 1.5. Equivalent monkeypatch alternative: `monkeypatch.setattr("sleap_roots.circumnutation._noise.compute_d2_residual_xy", lambda *a, **k: 1.7 * sg_value)` to force exactly `sg_d2_agreement = 1.7`.
- **WHEN** `qc.compute(trajectory_df)` is called
- **THEN** `qc_failure_reason == "growth_axis_unreliable, sg_d2_agreement_high, frac_outlier_steps_high"` (clauses in `_FAILURE_CLAUSE_ORDER` order)

#### Scenario: Short-track gate produces sentinel single-clause reason
- **GIVEN** a 3-frame track (below `SG_WINDOW_SHORT = 5`)
- **WHEN** `qc.compute(trajectory_df)` is called
- **THEN** `qc_failure_reason == "qc_inputs_insufficient"` (literally; NOT comma-concatenated with `"growth_axis_unreliable"` even if that condition would have fired)

#### Scenario: Threshold override via ConstantsT changes failure-clause firing
- **GIVEN** a track whose `worst_step_ratio = 6.0`
- **WHEN** `qc.compute(trajectory_df)` is called with `constants=ConstantsT(WORST_STEP_RATIO_MAX=10)`
- **THEN** the `worst_step_ratio_high` clause does NOT appear in `qc_failure_reason`
- **AND** when called with default constants (`WORST_STEP_RATIO_MAX=5`), it DOES appear

### Requirement: QC tier growth_axis_unreliable equality with Tier 0
The system SHALL emit `growth_axis_unreliable` as a column of the QC tier per-plant DataFrame, with values element-wise equal as `bool` dtype to the same column emitted by Tier 0's `kinematics.compute` on the same `trajectory_df` input. The equality is by construction: both tiers compute the flag via `(D == 0.0) or (not math.isnan(sg_residual) and D < constants.GROWTH_AXIS_RELIABILITY_K * sg_residual)` where `sg_residual = _noise.compute_sg_residual_xy(xy[:,0], xy[:,1], window=constants.SG_WINDOW_SHORT, degree=constants.SG_DEGREE)`, `D = float(np.linalg.norm(xy[-1] - xy[0]))`, and `xy = subset[["tip_x", "tip_y"]].to_numpy(dtype=float)` after the same NaN-drop and sort-by-frame.

The column SHALL be `bool` dtype (no NaN values) in BOTH outputs — tracks where the computation is degenerate (`len < 2` after dropna) SHALL emit `False`, matching Tier 0's `_emit_nan_row` precedent.

This requirement deliberately reverses the previous CC-5 rule (in `docs/circumnutation/roadmap.md`) that *"Tier 0 emits `growth_axis_unreliable`; QC does NOT re-emit a duplicate column."* Rationale: QC's output is more useful standalone when `growth_axis_unreliable` is a first-class bool column rather than only a string-embedded clause in `qc_failure_reason`. The duplicated column between Tier 0 and QC outputs is a cost; equality-by-construction makes that cost trivial (coalesce or drop in PR #14 pipeline composition).

#### Scenario: Equality on Nipponbare fixture
- **GIVEN** the Nipponbare plate 001 fixture loaded via `Series.get_tracked_tips()` (per `tests/data/circumnutation_nipponbare_plate_001/`) and enriched with the 4 missing identity columns (`plate_id`, `plant_id = track_id`, `genotype = "Nipponbare"`, `treatment = "MOCK"`)
- **WHEN** both `kinematics.compute(df)` and `qc.compute(df)` are called with default constants
- **THEN** `kinematics_result["growth_axis_unreliable"].dtype == qc_result["growth_axis_unreliable"].dtype == np.dtype("bool")`
- **AND** `(kinematics_result["growth_axis_unreliable"] == qc_result["growth_axis_unreliable"]).all()` is True (element-wise equal for all 6 tracks)

#### Scenario: Equality holds under int / float32 / object dtype input
- **GIVEN** a `trajectory_df` whose `tip_x` and `tip_y` columns are dtype `int` or `float32` rather than the canonical `float64`
- **WHEN** both `kinematics.compute(df)` and `qc.compute(df)` are called
- **THEN** the `growth_axis_unreliable` columns are still element-wise equal between the two outputs (because both tiers apply `to_numpy(dtype=float)` before passing arrays to the shared helper)

#### Scenario: Equality holds for closed-loop tracks (D == 0)
- **GIVEN** a track of length ≥ 2 where `xy[-1] == xy[0]` (closed loop)
- **WHEN** both `kinematics.compute(df)` and `qc.compute(df)` are called
- **THEN** both report `growth_axis_unreliable == True` for that track (the `D == 0.0` disjunct fires in both implementations)

#### Scenario: Equality holds for ultra-short tracks (n < 2)
- **GIVEN** a track that has only 1 row after NaN-drop
- **WHEN** both `kinematics.compute(df)` and `qc.compute(df)` are called
- **THEN** both report `growth_axis_unreliable == False` for that track (matching `_emit_nan_row` precedent in Tier 0; matching D7 step 2 path in QC)

### Requirement: QC tier input-validation boundary
The system SHALL validate at the entry of `qc.compute` that `trajectory_df` is a `pandas.DataFrame` and that it contains the eight row-identity columns plus `frame`, `tip_x`, `tip_y`. On validation failure, `ValueError` SHALL be raised with a message naming the offending field or type. Validation MAY be implemented by delegating to the foundation's `_types._validate_trajectory_df` helper, OR by re-implementing the equivalent checks inline; either choice is acceptable as long as the error messages match the foundation's permissive-but-clear-error style (matching the precedent established by Tier 0 in Requirement: Tier 0 input-validation boundary).

Beyond the foundation's column-presence and DataFrame-type checks, the QC tier inherits Tier 0's deliberate scoping choices on input quality (documented as non-goals of this PR; deferred to future tier PRs or pre-validation by the caller):

- **`±inf` in `tip_x` / `tip_y`** is NOT validated. The QC tier propagates `inf` through trait computations; resulting traits MAY be `inf` or `NaN` depending on the propagation path. In the worst case (>50% of steps `inf`), `frac_outlier_steps` may silently pass at `0.0` because `median` of mostly-`inf` is `inf` and `inf > 2 * inf == False`. **Rationale**: SLEAP predictions never emit `±inf`; this would only happen via corrupted upstream data. A future QC-level `±inf` detector is deliberately out of scope for PR #3 (file follow-up if needed).
- **Duplicate `(track_id, frame)` rows** are NOT detected. Same rationale as Tier 0: upstream data integrity is the caller's responsibility.

The QC tier additionally documents one tier-specific non-goal:

- **Stationary tracks** (all step magnitudes identically zero, so `median(\|Δxy_i\|) == 0`) yield NaN for `frac_outlier_steps` and `worst_step_ratio`. Under NaN-comparison semantics this fires BOTH `frac_outlier_steps_high` AND `worst_step_ratio_high` clauses in `qc_failure_reason`. **Rationale**: a stationary track IS a QC-failing track in any reasonable sense; emitting two clauses is correct semantically (the track failed two thresholds). A dedicated `stationary_track` clause is reserved for a future PR if a downstream user wants finer-grained categorization.

#### Scenario: ±inf in tip_x propagates without raising
- **GIVEN** a `trajectory_df` with one row having `tip_x = float('inf')` (the foundation's `CircumnutationInputs` validator does not check finiteness per row)
- **WHEN** `qc.compute(trajectory_df)` is called
- **THEN** no exception is raised
- **AND** the trait values for that track are either `inf` or `NaN` (propagation-dependent)

#### Scenario: Duplicate `(track_id, frame)` rows do not raise
- **GIVEN** a `trajectory_df` where the same `(track_id, frame)` 2-tuple appears in two rows
- **WHEN** `qc.compute(trajectory_df)` is called
- **THEN** no exception is raised
- **AND** the `Δframe = 0` divide-by-zero MAY produce non-finite values in step-magnitude-derived traits

#### Scenario: Stationary track fires both outlier clauses
- **GIVEN** a 100-frame track with `tip_x = 5.0` constant and `tip_y = 3.0` constant (all step magnitudes are zero, so `median = 0`)
- **WHEN** `qc.compute(trajectory_df)` is called
- **THEN** `frac_outlier_steps` is `NaN` and `worst_step_ratio` is `NaN`
- **AND** `track_is_clean == False`
- **AND** `qc_failure_reason` contains both `"frac_outlier_steps_high"` AND `"worst_step_ratio_high"`
