# Spec delta — add-circumnutation-synthetic-generator

## MODIFIED Requirements

### Requirement: Package layout
The system SHALL provide a `sleap_roots.circumnutation` sub-package whose import-tree is complete from this PR onward — every module name referenced anywhere in `docs/circumnutation/roadmap.md` exists as an importable Python module from PR #1 forward, even if its functions raise `NotImplementedError` until later PRs land. The package SHALL contain:

- 7 contract modules: `__init__.py`, `_constants.py`, `_types.py`, `_io.py`, `units.py`, `_noise.py`, `_geometry.py`
- 3 implementation modules: `kinematics` (implemented from PR #2 onward; see Requirement: Tier 0 raw kinematic traits), `qc` (implemented from PR #3 onward; see Requirement: QC tier per-track quality traits), and `synthetic` (implemented from PR #4 onward; see Requirement: Synthetic trajectory generator)
- 7 stub modules: `temporal_cwt`, `psi_g`, `midline`, `spatial_cwt`, `parametric`, `plotting`, `pipeline`

Each stub module SHALL define exactly one canonical callable per the table below, raising `NotImplementedError(f"PR #{N} — see docs/circumnutation/roadmap.md")` with the documented PR number. Each stub callable SHALL also have a complete Google-style docstring (Args, Returns, Raises) so `pydocstyle --convention=google` and the mkdocstrings auto-generated reference pages remain informative. Stubs whose tier PR will compose with the typed `ConstantsT` override-bag SHALL include `constants=None` as a forward-compatible keyword parameter so callers do not get `TypeError` before `NotImplementedError`.

| Stub module | Canonical callable | PR # |
|---|---|---|
| `temporal_cwt` | `compute_scaleogram(x, cadence_s, constants=None)` | 5 |
| `psi_g` | `compute_psi_g(x, y, constants=None)` | 7 |
| `midline` | `reconstruct(x, y, cadence_s, constants=None)` | 8 |
| `spatial_cwt` | `compute_scaleogram(kappa, ds, constants=None)` | 9 |
| `parametric` | `compute(tier3_df, R_px, omega, Delta_phi)` | 11 |
| `plotting` | `scaleogram(scaleogram_result, out_path)` | 16 |
| `pipeline` | `compute_traits(inputs, constants=None)` | 14 |

The `kinematics` module SHALL be importable on the same terms as the other modules (clean import, namespaced logger) and SHALL expose `compute(trajectory_df, constants=None)` per Requirement: Tier 0 raw kinematic traits. The `qc` module SHALL be importable on the same terms and SHALL expose `compute(trajectory_df, constants=None)` per Requirement: QC tier per-track quality traits. The `synthetic` module SHALL be importable on the same terms and SHALL expose `generate_trajectory(...)` per Requirement: Synthetic trajectory generator. Unlike the stub modules, calling `kinematics.compute`, `qc.compute`, or `synthetic.generate_trajectory` with a valid input SHALL NOT raise `NotImplementedError`.

The `_noise` and `_geometry` modules are private (underscore-prefixed) shared internals; their canonical callables are documented under Requirement: Tier 0 helper modules.

#### Scenario: All stub modules import cleanly
- **WHEN** a user runs `import sleap_roots.circumnutation as c; import sleap_roots.circumnutation.kinematics, sleap_roots.circumnutation.qc, sleap_roots.circumnutation.synthetic, sleap_roots.circumnutation.temporal_cwt, sleap_roots.circumnutation.psi_g, sleap_roots.circumnutation.midline, sleap_roots.circumnutation.spatial_cwt, sleap_roots.circumnutation.parametric, sleap_roots.circumnutation.plotting, sleap_roots.circumnutation.pipeline`
- **THEN** every import succeeds
- **AND** no exception is raised at module-import time

#### Scenario: Calling each remaining stub raises NotImplementedError with the correct PR number
- **GIVEN** the package imported as in the previous scenario
- **WHEN** the canonical callable in each of the 7 remaining stub modules (`temporal_cwt`, `psi_g`, `midline`, `spatial_cwt`, `parametric`, `plotting`, `pipeline`) is invoked (parameters per the table above; `NotImplementedError` fires before any argument check)
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

#### Scenario: `synthetic.generate_trajectory` no longer raises NotImplementedError
- **WHEN** `sleap_roots.circumnutation.synthetic.generate_trajectory()` is invoked with all-default kwargs
- **THEN** the call returns a `pandas.DataFrame` (the per-frame trajectory output) without raising `NotImplementedError`
- **AND** the DataFrame has `SYNTHETIC_N_FRAMES` rows (default 575) and the documented 11-column schema per Requirement: Synthetic trajectory generator

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
The system SHALL expose all overridable defaults as module-level named constants in `sleap_roots/circumnutation/_constants.py`. The set SHALL include at minimum: `NOISE_MASK_K`, `LGZ_STEADY_STATE_RESIDUAL_MAX`, `NYQUIST_RATIO_MAX`, `SG_D2_AGREEMENT_MAX`, `SG_MSD_AGREEMENT_MAX`, `D2_MSD_AGREEMENT_MAX`, `LGZ_NMIN_RESOLVABLE`, `COI_FRACTION_MAX`, `BAND_POWER_NOISE_RATIO`, `WAVELET_DEFAULT_TEMPORAL`, `WAVELET_DEFAULT_SPATIAL`, `SG_WINDOW_SHORT`, `SG_DEGREE`, `SG_WINDOW_DETREND`, `OUTLIER_STEP_RATIO`, `FRAC_OUTLIER_STEPS_MAX`, `WORST_STEP_RATIO_MAX`, `GROWTH_AXIS_RELIABILITY_K`, `SYNTHETIC_T_NUTATION_S`, `SYNTHETIC_AMPLITUDE_PX`, `SYNTHETIC_GROWTH_RATE_PX_PER_FRAME`, `SYNTHETIC_NOISE_SIGMA_PX`, `SYNTHETIC_CADENCE_S`, `SYNTHETIC_N_FRAMES`, `SYNTHETIC_GROWTH_AXIS_ANGLE_RAD`, `_SCHEMA_VERSION`, `_CONSTANTS_VERSION`. The values SHALL match the defaults in `docs/circumnutation/roadmap.md` cross-cutting concern CC-2 and `docs/circumnutation/theory.md` §7.6 (for the QC-tier-introduced thresholds: `FRAC_OUTLIER_STEPS_MAX = 0.05`, `WORST_STEP_RATIO_MAX = 5`, `SG_MSD_AGREEMENT_MAX = 1.5`, `D2_MSD_AGREEMENT_MAX = 1.5`) and `docs/circumnutation/preliminary_results_2026-05-07.md` §1, §3.4, §4.1, §4.3 (for the synthetic-generator defaults: `SYNTHETIC_T_NUTATION_S = 3333.0`, `SYNTHETIC_AMPLITUDE_PX = 10.0`, `SYNTHETIC_GROWTH_RATE_PX_PER_FRAME = 4.29`, `SYNTHETIC_NOISE_SIGMA_PX = 2.0`, `SYNTHETIC_CADENCE_S = 300.0`, `SYNTHETIC_N_FRAMES = 575`, `SYNTHETIC_GROWTH_AXIS_ANGLE_RAD = math.pi / 2`); `_SCHEMA_VERSION` SHALL be `1` (unchanged from PR #1) and `_CONSTANTS_VERSION` SHALL be `3` (bumped from `2` in this PR per the version-sentinel contract — the constants set grew by 7). The module SHALL also expose `PIPELINE_UNIT_VOCABULARY` (px-based + calibration-independent units, the closed sidecar vocabulary), `CONVERTED_UNIT_VOCABULARY` (mm-based units produced by `convert_to_mm`), and `VALID_UNIT_VOCABULARY` (their union), plus `ROW_IDENTITY_UNITS` (the canonical units dict for the eight row-identity columns).

The `ConstantsT` typed override-bag SHALL include corresponding fields for every overridable constant above, so callers can override per-call via `ConstantsT(SYNTHETIC_AMPLITUDE_PX=20.0)` etc. `_default_constants_snapshot()` SHALL emit every constant name in the set above into the run-metadata sidecar, including the seven new synthetic-generator constants.

#### Scenario: All required constants are importable with correct types
- **WHEN** a user runs `from sleap_roots.circumnutation import _constants`
- **THEN** every name listed above is an attribute of `_constants`
- **AND** each value matches the documented default in `roadmap.md` CC-2, `theory.md` §7.6, and `preliminary_results_2026-05-07.md` §1/§3.4/§4.1/§4.3
- **AND** `_constants._SCHEMA_VERSION` is the integer `1`
- **AND** `_constants._CONSTANTS_VERSION` is the integer `3`
- **AND** `_constants.PIPELINE_UNIT_VOCABULARY`, `_constants.CONVERTED_UNIT_VOCABULARY`, `_constants.VALID_UNIT_VOCABULARY`, `_constants.ROW_IDENTITY_UNITS` are all importable

#### Scenario: New QC-tier constants are overridable via ConstantsT
- **GIVEN** a custom `ConstantsT(FRAC_OUTLIER_STEPS_MAX=0.10, WORST_STEP_RATIO_MAX=10, SG_MSD_AGREEMENT_MAX=2.0, D2_MSD_AGREEMENT_MAX=2.0)`
- **WHEN** the instance is inspected
- **THEN** each overridden field reflects its caller-supplied value
- **AND** unoverridden fields reflect their module-level defaults

#### Scenario: New synthetic-generator constants are overridable via ConstantsT
- **GIVEN** a custom `ConstantsT(SYNTHETIC_T_NUTATION_S=1800.0, SYNTHETIC_AMPLITUDE_PX=20.0, SYNTHETIC_GROWTH_RATE_PX_PER_FRAME=3.0, SYNTHETIC_NOISE_SIGMA_PX=1.0, SYNTHETIC_CADENCE_S=60.0, SYNTHETIC_N_FRAMES=200, SYNTHETIC_GROWTH_AXIS_ANGLE_RAD=0.0)`
- **WHEN** the instance is inspected
- **THEN** each of the seven overridden fields reflects its caller-supplied value
- **AND** unoverridden fields reflect their module-level defaults

#### Scenario: Constants snapshot includes the four QC constants and the seven synthetic constants
- **WHEN** `_default_constants_snapshot()` is called
- **THEN** the returned mapping contains `FRAC_OUTLIER_STEPS_MAX`, `WORST_STEP_RATIO_MAX`, `SG_MSD_AGREEMENT_MAX`, `D2_MSD_AGREEMENT_MAX` with their default values
- **AND** the returned mapping contains `SYNTHETIC_T_NUTATION_S`, `SYNTHETIC_AMPLITUDE_PX`, `SYNTHETIC_GROWTH_RATE_PX_PER_FRAME`, `SYNTHETIC_NOISE_SIGMA_PX`, `SYNTHETIC_CADENCE_S`, `SYNTHETIC_N_FRAMES`, `SYNTHETIC_GROWTH_AXIS_ANGLE_RAD` with their default values

## ADDED Requirements

### Requirement: Synthetic trajectory generator
The system SHALL provide `sleap_roots.circumnutation.synthetic.generate_trajectory(*, ...) -> pd.DataFrame`. The function SHALL accept the canonical `generate_trajectory(...)` signature locked by the foundation's Package layout requirement — `px_per_mm` SHALL NOT appear in the signature (the function emits pure-pixel trajectories per CC-3). All parameters SHALL be keyword-only (`*,` enforced at signature top); positional invocation SHALL raise `TypeError`.

The function SHALL realize Rivière 2022 Eq. 4 in **parametric closed form** (not literal Eq. 5 ODE forward integration). The apex propagates along the growth axis at velocity `v_growth_per_s = growth_rate_px_per_frame / cadence_s`; transverse nutation contributes `A_lat · sin(handedness · ω · t + initial_phase_rad)` with `A_lat = amplitude_px / 2` and `ω = 2π / T_nutation_s`; iid Gaussian localization noise is added per-axis with `σ_per_axis = noise_sigma_px / √2` so that the QC tier's xy-quadrature noise estimators (`sg_residual_xy`, `d2_noise_xy`, `msd_noise_xy`) recover `noise_sigma_px` directly. The growth-axis unit vector is `u_g = (cos(growth_axis_angle_rad), sin(growth_axis_angle_rad))` and the lateral unit vector is `u_lat = (-u_g[1], u_g[0])` (standard CCW 90° rotation in math axes; under the image-y-down convention this displays as visually clockwise on screen).

The function SHALL return a `pandas.DataFrame` with exactly 11 columns and exactly `n_frames` rows. The columns SHALL be (in this order):

1. The 8 row-identity columns in their declared order: `series`, `sample_uid`, `timepoint`, `plate_id`, `plant_id`, `track_id`, `genotype`, `treatment`. The DataFrame contains one unique row-identity 5-tuple (single-track output per call).
2. `frame` (int64) — values `[0, n_frames)` strictly monotonic ascending; `frame.iloc[0] == 0` and `frame.iloc[-1] == n_frames - 1`.
3. `tip_x` (float64) — pure-pixel; the closed-form longitudinal + transverse + noise sum per the math above.
4. `tip_y` (float64) — pure-pixel; same.

The function SHALL emit dtypes per the contract: `frame.dtype == np.int64`; `tip_x.dtype == tip_y.dtype == np.float64`; `plant_id.dtype == track_id.dtype == np.int64`; `series`, `sample_uid`, `timepoint`, `plate_id`, `genotype`, `treatment` all `object` dtype. When `genotype=None` (or `treatment=None`) is passed, the resulting column contains `np.nan` values in `object`-dtype storage (NOT literal string `"None"`); `df["genotype"].isna().all()` returns `True` in this case.

**Closed-form math equivalences** (load-bearing for round-trip tests):

- In the noise-free case (`noise_sigma_px == 0`), the Tier 0 trait `v_long_signed_median_px_per_frame` recovers `growth_rate_px_per_frame` exactly when `amplitude_px == 0` (pure linear; growth-axis inference is exact). With nutation, recovery is within ≈0.5% because the kinematics-inferred net-displacement growth-axis matches `growth_axis_angle_rad` up to a small phase-sampling offset.
- In the noise-free case, the Tier 0 trait `angular_amplitude` (peak-to-peak ψ_g) equals **`2 · arctan(amplitude_px · ω / (2 · v_growth_per_s))`** (the exact relation, NOT the small-angle approximation `amplitude_px · ω / v_growth_per_s` which over-estimates by ~13% in the plate-001 empirical regime).
- The QC tier's xy-quadrature noise estimators recover `noise_sigma_px` within ±15% (matches theory.md §8 spatial tolerance and absorbs the documented SG under-bias).
- For default plate-001 parameters, `kinematics.compute(df)["growth_axis_unreliable"]` is `False` (safety margin ≈ 123× over the gate threshold; `D ≈ 2467 px`, `K · noise ≈ 20 px`).

**Determinism contract (CC-6).** `random_state` SHALL accept `int`, `np.random.Generator`, or `None`. A single internal call to `np.random.default_rng(random_state)` handles all three idiomatically. Same `int` seed SHALL produce bit-identical `tip_x` / `tip_y` arrays across two calls AND across OSs on 64-bit platforms (relying on numpy's PCG64 stability per NEP 19; sleap-roots' supported CI matrix — Ubuntu / Windows / macOS x86_64 + Apple Silicon arm64 — is 100% 64-bit). The legacy `np.random.RandomState` API SHALL be rejected with `TypeError`.

When `noise_sigma_px == 0.0` exactly, the implementation SHALL NOT call `np.random.default_rng(random_state)` or `rng.normal(...)` — the RNG path SHALL be short-circuited. This decouples output determinism from `random_state` in noise-free mode AND preserves caller-supplied Generator state unchanged: a caller passing `random_state=my_rng` with `noise_sigma_px=0` SHALL get `my_rng` back in unchanged state (`my_rng.bit_generator.state` SHALL be equal before and after the call as a deep dict comparison).

**Handedness convention (load-bearing for PR #7's `handedness` trait).** `handedness` SHALL be `+1` or `-1` (exactly; integers only; bool rejected). Default `+1` corresponds to counterclockwise per BM2016 §"Constant principal direction of growth" + `theory.md` §3.5 / §7.3. With `handedness = +1` AND `noise_sigma_px = 0`, `_geometry.compute_psi_g(tip_x, tip_y)` SHALL return an unwrapped ψ_g time series whose mean derivative is positive (`np.mean(np.diff(psi_g)) > 0`). With `handedness = -1`, the mean derivative SHALL be negative.

**Rivière correspondence (documented in module docstring; not exposed in the API):** the API parameters map to Rivière 2022 quantities via `v_growth_per_s = ε̇₀ · R`, `amplitude_px = 2 · ΔL · δ̇₀ / ω = R · Δφ` (Eq. 1), `ω = 2π / T_nutation_s`. The Rivière 6-tuple `(L_gz, ΔL, δ̇₀, ε̇₀, ω, R)` is degenerate at the tip-trajectory level (only 3 aggregate combinations are observable); PR #12 wraps `generate_trajectory` with a Rivière-named translation helper once PR #9 / PR #11 land spatial-CWT recovery of `L_gz` / `ΔL` individually.

**ConstantsT resolution-order.** Seven user-facing parameters (`n_frames`, `cadence_s`, `amplitude_px`, `T_nutation_s`, `growth_rate_px_per_frame`, `noise_sigma_px`, `growth_axis_angle_rad`) SHALL default to `None` in the signature. At call time, the resolution order SHALL be:

1. **Call-site kwarg wins.** If the parameter is explicitly passed (not `None`), use that value.
2. **`constants` parameter overrides module-level default.** If the parameter is `None` AND `constants` is a `ConstantsT` instance, use `constants.SYNTHETIC_<name>`.
3. **Module-level default.** If both above are absent, use the module-level constant directly (which equals the `ConstantsT()` field default).

The remaining parameters (`handedness`, `x0_px`, `y0_px`, `initial_phase_rad`, `random_state`, `constants`, and the 8 identity columns) keep direct literal defaults because they are NOT `ConstantsT`-overridable.

**Strict input validation at the boundary.** Every numeric parameter SHALL be validated for type, finiteness, sign, and bool-rejection. `isinstance(value, (int, np.integer))` for integer parameters; `isinstance(value, (float, int, np.floating, np.integer))` for float parameters (with `bool` then explicitly rejected because `True`/`False` are `int` subclasses). String inputs SHALL be rejected with `TypeError`; the `generate_trajectory` function does NOT coerce string-typed numerics (unlike `CircumnutationInputs.cadence_s` which has an attrs converter). On validation failure the function SHALL raise `ValueError` or `TypeError` whose message names the offending field.

#### Scenario: Default call returns 575-row DataFrame with the documented schema
- **WHEN** `synthetic.generate_trajectory()` is called with all-default kwargs
- **THEN** the returned DataFrame has exactly 575 rows
- **AND** the columns are `["series", "sample_uid", "timepoint", "plate_id", "plant_id", "track_id", "genotype", "treatment", "frame", "tip_x", "tip_y"]` in that order
- **AND** `df["frame"].dtype == np.dtype("int64")`, `df["tip_x"].dtype == df["tip_y"].dtype == np.dtype("float64")`, `df["plant_id"].dtype == df["track_id"].dtype == np.dtype("int64")`
- **AND** `df["series"].dtype == df["sample_uid"].dtype == df["timepoint"].dtype == df["plate_id"].dtype == df["genotype"].dtype == df["treatment"].dtype == np.dtype("object")`
- **AND** `df["frame"].iloc[0] == 0 and df["frame"].iloc[-1] == 574`
- **AND** `(np.diff(df["frame"]) == 1).all()`
- **AND** `df["plant_id"].equals(df["track_id"])`
- **AND** the default `growth_axis_angle_rad = math.pi / 2` corresponds to the apex propagating in the `+y` screen direction (visually downward) under the image-y-down coordinate convention per `theory.md` §2.1 — i.e., `u_g = (cos(π/2), sin(π/2)) = (0, 1)` aligns the apex motion with the standard rice-plate convention of roots growing downward on a vertically-mounted plate

#### Scenario: `None` for `genotype` and `treatment` becomes NaN, not literal "None"
- **WHEN** `synthetic.generate_trajectory(genotype=None, treatment=None)` is called
- **THEN** `df["genotype"].isna().all() == True` AND `df["genotype"].dtype == np.dtype("object")`
- **AND** `(df["genotype"] == "None").any() == False` (no literal `"None"` strings)
- **AND** the same holds for `treatment`

#### Scenario: Positional invocation raises TypeError
- **WHEN** `synthetic.generate_trajectory(575)` is called (positional, not keyword)
- **THEN** a `TypeError` is raised

#### Scenario: Same int seed produces bit-identical output across two calls
- **WHEN** `synthetic.generate_trajectory(random_state=42, ...)` is called twice with identical kwargs
- **THEN** the two returned DataFrames have `tip_x` arrays that are bit-identical via `np.array_equal` (not just `np.allclose`)
- **AND** the `tip_y` arrays are also bit-identical
- **AND** the assertion holds across operating systems on 64-bit platforms (Ubuntu / Windows / macOS x86_64 + Apple Silicon arm64)

#### Scenario: int seed and default_rng(seed) produce identical output
- **GIVEN** two calls with `random_state=42` and `random_state=np.random.default_rng(42)` respectively (otherwise identical kwargs)
- **WHEN** both are invoked
- **THEN** the two returned DataFrames have bit-identical `tip_x` and `tip_y` arrays

#### Scenario: `noise_sigma_px=0` short-circuits RNG and preserves Generator state
- **GIVEN** `rng = np.random.default_rng(42)` and `state_before = rng.bit_generator.state`
- **WHEN** `synthetic.generate_trajectory(noise_sigma_px=0, random_state=rng, ...)` is called
- **THEN** the returned DataFrame's `tip_x` matches the output of a separate call with `random_state=None` (RNG path not entered; output determined only by closed-form math)
- **AND** `rng.bit_generator.state == state_before` (deep dict comparison) — caller-supplied Generator state unchanged

#### Scenario: Different seeds produce different output
- **WHEN** `synthetic.generate_trajectory(random_state=0, ...)` and `synthetic.generate_trajectory(random_state=1, ...)` are called with otherwise identical kwargs and `noise_sigma_px > 0`
- **THEN** the two returned DataFrames' `tip_x` arrays differ (`not np.allclose`)

#### Scenario: Recovered longitudinal velocity equals input growth rate in pure-linear case
- **GIVEN** `synthetic.generate_trajectory(amplitude_px=0, growth_rate_px_per_frame=4.29, noise_sigma_px=0, growth_axis_angle_rad=math.pi/2, ...)` (pure linear, no nutation, no noise)
- **WHEN** `kinematics.compute(df)["v_long_signed_median_px_per_frame"].iloc[0]` is computed
- **THEN** the value equals `4.29` within IEEE float tolerance (`abs(value - 4.29) < 1e-9`; loosened from `1e-10` per /openspec-review round-1 TDD reviewer B3 to accommodate cross-platform BLAS rounding through the `(growth_rate / cadence_s) · (i · cadence_s)` reintroduction)
- **AND** `kinematics.compute(df)["long_lat_ratio"].iloc[0]` is `NaN` (pure-linear trajectory yields `v_lat_abs_median == 0`, triggering the documented kinematics NaN contract)

#### Scenario: Recovered angular amplitude matches exact arctan formula in noise-free case
- **GIVEN** `synthetic.generate_trajectory(amplitude_px=10.0, T_nutation_s=3333.0, growth_rate_px_per_frame=4.29, cadence_s=300.0, n_frames=575, noise_sigma_px=0, ...)` (plate-001-matching defaults; noise-free)
- **WHEN** `kinematics.compute(df)["angular_amplitude"].iloc[0]` is computed
- **THEN** the value matches the exact analytical prediction `2 * arctan(amplitude_px * omega / (2 * v_growth_per_s))` (≈ 1.17 rad for these inputs) within ±15% (theory.md §8 spatial tolerance)
- **AND** `kinematics.compute(df)["growth_axis_unreliable"].iloc[0] == False`

#### Scenario: Noise round-trips via QC's xy-quadrature estimators with documented per-estimator bias
- **GIVEN** `synthetic.generate_trajectory(noise_sigma_px=2.0, random_state=42, ...)` with otherwise plate-001-matching defaults
- **WHEN** `qc.compute(df)` is computed
- **THEN** the QC tier's xy-quadrature noise estimators recover `noise_sigma_px` modulo the documented per-estimator bias factors empirically calibrated on the closed-form trajectory (Copilot review #5; §3.7 canary capture): `sg_residual_xy` recovers `≈ 0.65 × noise_sigma_px` within ±25%, `d2_noise_xy` recovers `≈ 0.95 × noise_sigma_px` within ±25% (near-unbiased), `msd_noise_xy` recovers `≈ 0.61 × noise_sigma_px` within ±25%. The bias factors are multiplicative (linear in `noise_sigma_px`), so doubling the input noise approximately doubles each estimator's output
- **AND** with default ConstantsT thresholds, `df_qc["d2_msd_agreement"].iloc[0]` lands at `≈ 0.95/0.61 ≈ 1.55` — slightly above the default `D2_MSD_AGREEMENT_MAX = 1.5` (matching the structural borderline PR #3 observed on plate 001 at `d2_msd_agreement = 1.537`). To assert `track_is_clean == True`, callers SHALL loosen the agreement thresholds via `ConstantsT(SG_D2_AGREEMENT_MAX=2.0, SG_MSD_AGREEMENT_MAX=2.0, D2_MSD_AGREEMENT_MAX=2.0)` — the documented escape per design.md
- **AND** when called with the loosened ConstantsT, `df_qc["track_is_clean"].iloc[0] == True` and `df_qc["qc_failure_reason"].iloc[0] == ""`

#### Scenario: handedness=+1 yields positive mean dψ_g/dt
- **GIVEN** `synthetic.generate_trajectory(handedness=+1, noise_sigma_px=0, ...)` (noise-free for unambiguous determinism)
- **WHEN** `_geometry.compute_psi_g(df["tip_x"].to_numpy(), df["tip_y"].to_numpy())` is computed and the mean first-difference `np.mean(np.diff(psi_g))` is evaluated
- **THEN** the result is positive (`> 0`)

#### Scenario: handedness=-1 yields negative mean dψ_g/dt
- **GIVEN** `synthetic.generate_trajectory(handedness=-1, noise_sigma_px=0, ...)`
- **WHEN** the same evaluation is performed
- **THEN** the result is negative (`< 0`)

#### Scenario: handedness=+1 curl-sign agrees with ψ_g sign
- **GIVEN** `synthetic.generate_trajectory(handedness=+1, noise_sigma_px=0, ...)` and the computed `ψ_g = _geometry.compute_psi_g(tip_x, tip_y)`
- **WHEN** the curl-sign of the trajectory is computed independently of `_geometry.compute_psi_g` as `sign(mean(diff(tip_x)[1:] * diff(diff(tip_y)) - diff(tip_y)[1:] * diff(diff(tip_x))))`
- **THEN** the curl-sign is positive (`> 0`)
- **AND** `sign(mean(diff(ψ_g)))` is also positive
- **AND** the two signs agree (both `+1`) — this cross-check guards against a future refactor of `_geometry.compute_psi_g` that inverts the `atan2(dx, dy)` argument order silently inverting PR #7's handedness trait convention

#### Scenario: handedness=-1 curl-sign agrees with ψ_g sign
- **GIVEN** `synthetic.generate_trajectory(handedness=-1, noise_sigma_px=0, ...)` and the computed `ψ_g = _geometry.compute_psi_g(tip_x, tip_y)`
- **WHEN** the curl-sign is computed via the same finite-difference formula as the previous scenario
- **THEN** the curl-sign is negative (`< 0`)
- **AND** `sign(mean(diff(ψ_g)))` is also negative
- **AND** the two signs agree (both `-1`) — combined with the `handedness=+1` scenario, this locks the convention chain `handedness input → trajectory rotation → atan2(dx, dy) sign` against silent inversion

#### Scenario: `ConstantsT` override propagates when kwarg omitted
- **GIVEN** `synthetic.generate_trajectory(amplitude_px=None, noise_sigma_px=0, constants=ConstantsT(SYNTHETIC_AMPLITUDE_PX=20.0))` (kwarg omitted; constants override active)
- **WHEN** `kinematics.compute(df)["angular_amplitude"].iloc[0]` is computed
- **THEN** the recovered value matches the exact analytical prediction using `amplitude_px=20.0` (i.e., `2 * arctan(10.0 * omega / (2 * v_growth_per_s))`) within ±15%

#### Scenario: Explicit kwarg overrides ConstantsT override (resolution-order kwarg-wins)
- **GIVEN** `synthetic.generate_trajectory(amplitude_px=15.0, noise_sigma_px=0, constants=ConstantsT(SYNTHETIC_AMPLITUDE_PX=20.0))` (BOTH kwarg AND constants override are set; kwarg should win)
- **WHEN** `kinematics.compute(df)["angular_amplitude"].iloc[0]` is computed
- **THEN** the recovered value matches the exact analytical prediction using `amplitude_px=15.0` (NOT 20.0; kwarg won the resolution race) within ±15%

#### Scenario: Invalid n_frames raises with named field
- **WHEN** `synthetic.generate_trajectory(n_frames=v)` is called for any `v ∈ {0, -1, True, 1.5, "100", np.nan, np.inf}`
- **THEN** in each case a `ValueError` or `TypeError` is raised
- **AND** the exception message contains the string `"n_frames"`

#### Scenario: Invalid cadence_s raises with named field; no string coercion
- **WHEN** `synthetic.generate_trajectory(cadence_s=v)` is called for any `v ∈ {0.0, -1.0, np.nan, np.inf, -np.inf, True, "300"}`
- **THEN** in each case a `ValueError` or `TypeError` is raised
- **AND** the exception message contains the string `"cadence_s"`
- **AND** `"300"` specifically raises `TypeError` (no implicit string-to-float coercion, unlike `CircumnutationInputs.cadence_s`)

#### Scenario: Invalid handedness raises with named field
- **WHEN** `synthetic.generate_trajectory(handedness=v)` is called for any `v ∈ {0, 2, -2, 1.0, True, "+1", None}`
- **THEN** in each case a `ValueError` or `TypeError` is raised (only `+1` or `-1` integer are valid)
- **AND** the exception message contains the string `"handedness"`

#### Scenario: Invalid random_state raises with named field
- **WHEN** `synthetic.generate_trajectory(random_state=v)` is called for any `v ∈ {1.5, "42", np.random.RandomState(0)}`
- **THEN** in each case a `TypeError` is raised
- **AND** the exception message contains the string `"random_state"`
- **AND** specifically `np.random.RandomState(0)` is rejected (legacy API; only the modern `Generator` API is accepted per the determinism contract)

#### Scenario: Reference-fixture agreement (Layer-1 sanity)
- **GIVEN** the plate 001 fixture (`tests/data/circumnutation_nipponbare_plate_001/plate_001_greyscale.tracked_proofread.slp`) loaded via `sio.load_slp` and enriched into a `trajectory_df` per PR #2's reference-value test pattern; AND `synth_df = synthetic.generate_trajectory()` with all-default kwargs
- **WHEN** `qc.compute(real_df)["sg_residual_xy"].median()` (median across 6 tracks) and `qc.compute(synth_df)["sg_residual_xy"].iloc[0]` are both computed
- **THEN** the two values agree within ±15% (`abs(real - synth) / real < 0.15`)
- **AND** the per-frame mean total step magnitude (`np.linalg.norm(np.diff(xy, axis=0), axis=1).mean()` per track, then median across 6 tracks for the real fixture; single-track value for synth) also agrees within ±15%
- **AND** `kinematics.compute(real_df)["growth_axis_unreliable"]` and `kinematics.compute(synth_df)["growth_axis_unreliable"]` are both False (no track flagged by the gate)
- **AND** if the fixture file is absent at the documented path, the test SHALL skip rather than fail (`pytest.skip` semantics; the fixture is committed via Git LFS but may be unavailable in environments where LFS pointers were not resolved)
