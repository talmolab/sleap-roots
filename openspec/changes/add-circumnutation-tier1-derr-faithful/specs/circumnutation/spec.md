# circumnutation Specification — PR #6 delta

## MODIFIED Requirements

### Requirement: Package layout
The system SHALL provide a `sleap_roots.circumnutation` sub-package whose import-tree is complete from this PR onward — every module name referenced anywhere in `docs/circumnutation/roadmap.md` exists as an importable Python module from PR #1 forward, even if its functions raise `NotImplementedError` until later PRs land. The package SHALL contain:

- 7 contract modules: `__init__.py`, `_constants.py`, `_types.py`, `_io.py`, `units.py`, `_noise.py`, `_geometry.py`
- 5 implementation modules: `kinematics` (implemented from PR #2 onward; see Requirement: Tier 0 raw kinematic traits), `qc` (implemented from PR #3 onward; see Requirement: QC tier per-track quality traits), `synthetic` (implemented from PR #4 onward; see Requirement: Synthetic trajectory generator), `temporal_cwt` (implemented from PR #5 onward; see Requirement: Temporal CWT scaleogram API and Requirement: Temporal CWT ridge API), and `nutation` (implemented from PR #6 onward; see Requirement: Tier 1 nutation trait emission API)
- 6 stub modules: `psi_g`, `midline`, `spatial_cwt`, `parametric`, `plotting`, `pipeline`

Each stub module SHALL define exactly one canonical callable per the table below, raising `NotImplementedError(f"PR #{N} — see docs/circumnutation/roadmap.md")` with the documented PR number. Each stub callable SHALL also have a complete Google-style docstring (Args, Returns, Raises) so `pydocstyle --convention=google` and the mkdocstrings auto-generated reference pages remain informative. Stubs whose tier PR will compose with the typed `ConstantsT` override-bag SHALL include `constants=None` as a forward-compatible keyword parameter so callers do not get `TypeError` before `NotImplementedError`.

| Stub module | Canonical callable | PR # |
|---|---|---|
| `psi_g` | `compute_psi_g(x, y, constants=None)` | 7 |
| `midline` | `reconstruct(x, y, cadence_s, constants=None)` | 8 |
| `spatial_cwt` | `compute_scaleogram(kappa, ds, constants=None)` | 9 |
| `parametric` | `compute(tier3_df, R_px, omega, Delta_phi)` | 11 |
| `plotting` | `scaleogram(scaleogram_result, out_path)` | 16 |
| `pipeline` | `compute_traits(inputs, constants=None)` | 14 |

The `kinematics` module SHALL be importable on the same terms as the other modules (clean import, namespaced logger) and SHALL expose `compute(trajectory_df, constants=None)` per Requirement: Tier 0 raw kinematic traits. The `qc` module SHALL be importable on the same terms and SHALL expose `compute(trajectory_df, constants=None)` per Requirement: QC tier per-track quality traits. The `synthetic` module SHALL be importable on the same terms and SHALL expose `generate_trajectory(...)` per Requirement: Synthetic trajectory generator. The `temporal_cwt` module SHALL be importable on the same terms and SHALL expose `compute_scaleogram(x, cadence_s, constants=None) -> ScaleogramResult` and `extract_ridge(scaleogram_result, constants=None) -> RidgeResult` per Requirement: Temporal CWT scaleogram API and Requirement: Temporal CWT ridge API, AND SHALL ALSO expose `smooth_ridge(ridge_result, window=None, constants=None) -> RidgeResult` per Requirement: Temporal CWT ridge-continuity smoothing API. The `nutation` module SHALL be importable on the same terms and SHALL expose `compute(trajectory_df, cadence_s, coordinate="lateral", constants=None) -> pd.DataFrame` per Requirement: Tier 1 nutation trait emission API. Unlike the stub modules, calling `kinematics.compute`, `qc.compute`, `synthetic.generate_trajectory`, `temporal_cwt.compute_scaleogram`, or `nutation.compute` with a valid input SHALL NOT raise `NotImplementedError`.

The `_noise` and `_geometry` modules are private (underscore-prefixed) shared internals; their canonical callables are documented under Requirement: Tier 0 helper modules.

**Scope note on PR #6 addition-vs-transition.** The `nutation` module is NEWLY created in PR #6 — it was never a stub module in PR #1–#5, and therefore does not appear in the stub-callable table. The implementation-module count grows from 4 (PR #5 baseline: kinematics, qc, synthetic, temporal_cwt) to 5 by ADDITION of `nutation`, not by transition from a prior stub. The stub-module count stays at 6 (psi_g, midline, spatial_cwt, parametric, plotting, pipeline). This is the first PR in the program to grow the implementation set without shrinking the stub set.

#### Scenario: All stub modules import cleanly
- **WHEN** a user runs `import sleap_roots.circumnutation as c; import sleap_roots.circumnutation.kinematics, sleap_roots.circumnutation.qc, sleap_roots.circumnutation.synthetic, sleap_roots.circumnutation.temporal_cwt, sleap_roots.circumnutation.nutation, sleap_roots.circumnutation.psi_g, sleap_roots.circumnutation.midline, sleap_roots.circumnutation.spatial_cwt, sleap_roots.circumnutation.parametric, sleap_roots.circumnutation.plotting, sleap_roots.circumnutation.pipeline`
- **THEN** every import succeeds
- **AND** no exception is raised at module-import time

#### Scenario: Calling each remaining stub raises NotImplementedError with the correct PR number
- **GIVEN** the package imported as in the previous scenario
- **WHEN** the canonical callable in each of the 6 remaining stub modules (`psi_g`, `midline`, `spatial_cwt`, `parametric`, `plotting`, `pipeline`) is invoked (parameters per the table above; `NotImplementedError` fires before any argument check)
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

#### Scenario: `temporal_cwt.compute_scaleogram` no longer raises NotImplementedError
- **GIVEN** a valid 1-D float64 ndarray `x` of length ≥ 9 with all-finite values, and a positive finite `cadence_s` (e.g., `np.linspace(0, 100, 32) * 0.1` and `cadence_s = 300.0`)
- **WHEN** `sleap_roots.circumnutation.temporal_cwt.compute_scaleogram(x, cadence_s)` is invoked
- **THEN** the call returns a `ScaleogramResult` (the temporal CWT scaleogram output) without raising `NotImplementedError`

#### Scenario: `temporal_cwt.extract_ridge` is callable on a valid ScaleogramResult without raising
- **GIVEN** a valid `ScaleogramResult` produced by `compute_scaleogram(x, 300.0)` on a length-≥9 finite array
- **WHEN** `sleap_roots.circumnutation.temporal_cwt.extract_ridge(scaleogram_result)` is invoked
- **THEN** the call returns a `RidgeResult` without raising any exception
- **AND** since `extract_ridge` is a NEW public symbol introduced by PR #5 (not a transition from a prior stub), it does not appear in the stub-callable table — its callability contract is locked here in the MODIFIED Package layout requirement for symmetry with `compute_scaleogram` (per /openspec-review round-2 reviewer N-I1)

#### Scenario: `nutation.compute` is callable on a valid trajectory_df without raising
- **GIVEN** a valid `trajectory_df` (8 row-identity columns + `frame`, `tip_x`, `tip_y`; ≥ 9 rows for at least one track) and a positive finite `cadence_s` (e.g., 300.0)
- **WHEN** `sleap_roots.circumnutation.nutation.compute(trajectory_df, cadence_s=300.0)` is invoked
- **THEN** the call returns a `pandas.DataFrame` (the Tier 1 per-plant nutation trait output) without raising `NotImplementedError`
- **AND** since `nutation` is a NEW module introduced by PR #6 (not a transition from a prior stub), it does not appear in the stub-callable table — its callability contract is locked here in the MODIFIED Package layout requirement for symmetry with `kinematics.compute`, `qc.compute`, `synthetic.generate_trajectory`, and `temporal_cwt.compute_scaleogram`

#### Scenario: `temporal_cwt.smooth_ridge` is callable on a valid RidgeResult without raising
- **GIVEN** a valid `RidgeResult` produced by `extract_ridge(compute_scaleogram(x, 300.0))` on a length-≥9 finite array
- **WHEN** `sleap_roots.circumnutation.temporal_cwt.smooth_ridge(ridge_result)` is invoked
- **THEN** the call returns a `RidgeResult` (with smoothed `periods_s`) without raising any exception
- **AND** since `smooth_ridge` is a NEW public symbol introduced by PR #6 (closing GitHub issue #214 — ridge-tracking continuity post-filter; not a transition from a prior stub), it does not appear in the stub-callable table — its callability contract is locked here in the MODIFIED Package layout requirement for symmetry with `compute_scaleogram` and `extract_ridge`

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
The system SHALL expose all overridable defaults as module-level named constants in `sleap_roots/circumnutation/_constants.py`. The set SHALL include at minimum: `NOISE_MASK_K`, `LGZ_STEADY_STATE_RESIDUAL_MAX`, `NYQUIST_RATIO_MAX`, `SG_D2_AGREEMENT_MAX`, `SG_MSD_AGREEMENT_MAX`, `D2_MSD_AGREEMENT_MAX`, `LGZ_NMIN_RESOLVABLE`, `COI_FRACTION_MAX`, `BAND_POWER_NOISE_RATIO`, `WAVELET_DEFAULT_TEMPORAL`, `WAVELET_DEFAULT_SPATIAL`, `SG_WINDOW_SHORT`, `SG_DEGREE`, `SG_WINDOW_DETREND`, `OUTLIER_STEP_RATIO`, `FRAC_OUTLIER_STEPS_MAX`, `WORST_STEP_RATIO_MAX`, `GROWTH_AXIS_RELIABILITY_K`, `SYNTHETIC_T_NUTATION_S`, `SYNTHETIC_AMPLITUDE_PX`, `SYNTHETIC_GROWTH_RATE_PX_PER_FRAME`, `SYNTHETIC_NOISE_SIGMA_PX`, `SYNTHETIC_CADENCE_S`, `SYNTHETIC_N_FRAMES`, `SYNTHETIC_GROWTH_AXIS_ANGLE_RAD`, `COI_EFOLDING_FACTOR`, `CWT_SCALE_COUNT_DEFAULT`, `CWT_PERIOD_MIN_NYQUIST_FACTOR`, `CWT_PERIOD_MAX_SIGNAL_FRACTION`, `RIDGE_CONTINUITY_FILTER_WINDOW`, `NOISE_FLOOR_OUT_OF_BAND_FACTOR`, `BAND_POWER_BAND_LOW_FACTOR`, `BAND_POWER_BAND_HIGH_FACTOR`, `DERR_EXPECTED_PERIOD_S`, `TEMPORAL_NYQUIST_RATIO_MAX`, `_SCHEMA_VERSION`, `_CONSTANTS_VERSION`. The values SHALL match the defaults in `docs/circumnutation/roadmap.md` cross-cutting concern CC-2 and `docs/circumnutation/theory.md` §7.6 (for the QC-tier-introduced thresholds: `FRAC_OUTLIER_STEPS_MAX = 0.05`, `WORST_STEP_RATIO_MAX = 5`, `SG_MSD_AGREEMENT_MAX = 1.5`, `D2_MSD_AGREEMENT_MAX = 1.5`) and `docs/circumnutation/preliminary_results_2026-05-07.md` §1, §3.4, §4.1, §4.3 (for the synthetic-generator defaults: `SYNTHETIC_T_NUTATION_S = 3333.0`, `SYNTHETIC_AMPLITUDE_PX = 10.0`, `SYNTHETIC_GROWTH_RATE_PX_PER_FRAME = 4.29`, `SYNTHETIC_NOISE_SIGMA_PX = 2.0`, `SYNTHETIC_CADENCE_S = 300.0`, `SYNTHETIC_N_FRAMES = 575`, `SYNTHETIC_GROWTH_AXIS_ANGLE_RAD = math.pi / 2`) and the wavelet-aware step-response derivation in PR #5's archived `design.md` D3 (for the CWT-machinery defaults: `COI_EFOLDING_FACTOR = math.sqrt(1.5)` calibrated for `cmor1.5-1.0` per `√B` envelope e-folding; `CWT_SCALE_COUNT_DEFAULT = 64`; `CWT_PERIOD_MIN_NYQUIST_FACTOR = 2.0`; `CWT_PERIOD_MAX_SIGNAL_FRACTION = 0.25`) and PR #6's `design.md` D4/D7/D8/D9 (for the Tier 1 / threshold defaults: `RIDGE_CONTINUITY_FILTER_WINDOW = 5` per GitHub issue #214 + Mallat 1999 §4.4.2; `NOISE_FLOOR_OUT_OF_BAND_FACTOR = 5.0` per CC-8 verbatim; `BAND_POWER_BAND_LOW_FACTOR = 0.5` and `BAND_POWER_BAND_HIGH_FACTOR = 2.0` per theory.md §7.2 `[0.5T, 2T]` band; `DERR_EXPECTED_PERIOD_S = 3333.0` per preliminary_results §4.4 Nipponbare empirical anchor; `TEMPORAL_NYQUIST_RATIO_MAX = 0.25` per theory.md §6.5 "10-min still works" empirical anchor); `_SCHEMA_VERSION` SHALL be `1` (unchanged from PR #1) and `_CONSTANTS_VERSION` SHALL be `5` (bumped from `4` in this PR per the version-sentinel contract — the constants set grew by 6). The module SHALL also expose `PIPELINE_UNIT_VOCABULARY` (px-based + calibration-independent units, the closed sidecar vocabulary), `CONVERTED_UNIT_VOCABULARY` (mm-based units produced by `convert_to_mm`), and `VALID_UNIT_VOCABULARY` (their union), plus `ROW_IDENTITY_UNITS` (the canonical units dict for the eight row-identity columns).

The `ConstantsT` typed override-bag SHALL include corresponding fields for every overridable constant above, so callers can override per-call via `ConstantsT(RIDGE_CONTINUITY_FILTER_WINDOW=11, NOISE_FLOOR_OUT_OF_BAND_FACTOR=3.0)` etc. `_default_constants_snapshot()` SHALL emit every constant name in the set above into the run-metadata sidecar, including the six new PR #6 Tier 1 / threshold constants. `len(_default_constants_snapshot())` SHALL be `35` exactly after PR #6 (29 pre-PR-#6 baseline + 6 PR #6 additions).

**Scope note on `NYQUIST_RATIO_MAX` ↔ `TEMPORAL_NYQUIST_RATIO_MAX` reciprocal docstring touch.** PR #6 ADDS a reciprocal docstring cross-reference between the existing `NYQUIST_RATIO_MAX` (locked as the SPATIAL cadence-Nyquist threshold per `theory.md` §6.5 spatial example "5.83 / 65 ≈ 9.0%, well below the conservative 25% threshold") and the NEW `TEMPORAL_NYQUIST_RATIO_MAX` (locked as the TEMPORAL cadence-Nyquist threshold per `theory.md` §6.5 temporal example "5-min comfortable, 10-min still works, 30-min aliases"). Both defaults are `0.25` — the dimensional separation lives in the constant NAMES + docstrings, NOT in different values. This is an internal-docstring-only touch with no behavioral change; PR #5's earlier touch of `NYQUIST_RATIO_MAX` docstring (adding a cross-reference to `CWT_PERIOD_MAX_SIGNAL_FRACTION`) is preserved. PR #6 enumerates the change here so the touch is explicitly scope-bounded and not surprising to reviewers of the `_constants.py` diff.

#### Scenario: All required constants are importable with correct types
- **WHEN** a user runs `from sleap_roots.circumnutation import _constants`
- **THEN** every name listed above is an attribute of `_constants`
- **AND** each value matches the documented default in `roadmap.md` CC-2, `theory.md` §6.5/§7.2/§7.6, `preliminary_results_2026-05-07.md` §1/§3.4/§4.1/§4.3/§4.4, PR #5's archived `design.md` D3/D7, and PR #6's `design.md` D4/D7/D8/D9
- **AND** `_constants._SCHEMA_VERSION` is the integer `1`
- **AND** `_constants._CONSTANTS_VERSION` is the integer `5`
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

#### Scenario: New CWT-machinery constants are overridable via ConstantsT
- **GIVEN** a custom `ConstantsT(COI_EFOLDING_FACTOR=math.sqrt(2.0), CWT_SCALE_COUNT_DEFAULT=128, CWT_PERIOD_MIN_NYQUIST_FACTOR=4.0, CWT_PERIOD_MAX_SIGNAL_FRACTION=0.5)`
- **WHEN** the instance is inspected
- **THEN** each of the four overridden fields reflects its caller-supplied value
- **AND** unoverridden fields reflect their module-level defaults

#### Scenario: New nutation/Tier 1 constants are overridable via ConstantsT
- **GIVEN** a custom `ConstantsT(RIDGE_CONTINUITY_FILTER_WINDOW=11, NOISE_FLOOR_OUT_OF_BAND_FACTOR=3.0, BAND_POWER_BAND_LOW_FACTOR=0.25, BAND_POWER_BAND_HIGH_FACTOR=4.0, DERR_EXPECTED_PERIOD_S=7200.0, TEMPORAL_NYQUIST_RATIO_MAX=0.20)`
- **WHEN** the instance is inspected
- **THEN** each of the six overridden fields reflects its caller-supplied value
- **AND** unoverridden fields reflect their module-level defaults

#### Scenario: Constants snapshot includes the four QC constants, the seven synthetic constants, the four CWT-machinery constants, and the six nutation/Tier 1 constants
- **WHEN** `_default_constants_snapshot()` is called
- **THEN** the returned mapping contains `FRAC_OUTLIER_STEPS_MAX`, `WORST_STEP_RATIO_MAX`, `SG_MSD_AGREEMENT_MAX`, `D2_MSD_AGREEMENT_MAX` with their default values
- **AND** the returned mapping contains `SYNTHETIC_T_NUTATION_S`, `SYNTHETIC_AMPLITUDE_PX`, `SYNTHETIC_GROWTH_RATE_PX_PER_FRAME`, `SYNTHETIC_NOISE_SIGMA_PX`, `SYNTHETIC_CADENCE_S`, `SYNTHETIC_N_FRAMES`, `SYNTHETIC_GROWTH_AXIS_ANGLE_RAD` with their default values
- **AND** the returned mapping contains `COI_EFOLDING_FACTOR`, `CWT_SCALE_COUNT_DEFAULT`, `CWT_PERIOD_MIN_NYQUIST_FACTOR`, `CWT_PERIOD_MAX_SIGNAL_FRACTION` with their default values
- **AND** the returned mapping contains `RIDGE_CONTINUITY_FILTER_WINDOW`, `NOISE_FLOOR_OUT_OF_BAND_FACTOR`, `BAND_POWER_BAND_LOW_FACTOR`, `BAND_POWER_BAND_HIGH_FACTOR`, `DERR_EXPECTED_PERIOD_S`, `TEMPORAL_NYQUIST_RATIO_MAX` with their default values
- **AND** `len(_default_constants_snapshot()) == 35` exactly (29 pre-PR-#6 baseline + 6 PR #6 additions)

#### Scenario: NYQUIST_RATIO_MAX docstring cross-references CWT_PERIOD_MAX_SIGNAL_FRACTION
- **WHEN** `sleap_roots.circumnutation._constants.NYQUIST_RATIO_MAX` is inspected via `inspect.getdoc(...)` or via `_constants.__dict__["NYQUIST_RATIO_MAX"].__doc__` (note: module-level constants do not carry docstrings natively; the cross-reference lives in the module-level constant's surrounding docstring or in a `__doc__`-attached commentary block defined at the same point — implementation may use either pattern as long as `grep "CWT_PERIOD_MAX_SIGNAL_FRACTION" sleap_roots/circumnutation/_constants.py` shows the cross-reference text within ±5 lines of `NYQUIST_RATIO_MAX:` declaration)
- **THEN** the cross-reference text mentions `CWT_PERIOD_MAX_SIGNAL_FRACTION` AND notes the numerical-equality-but-semantically-distinct relationship
- **AND** `CWT_PERIOD_MAX_SIGNAL_FRACTION`'s docstring reciprocally references `NYQUIST_RATIO_MAX`

#### Scenario: NYQUIST_RATIO_MAX and TEMPORAL_NYQUIST_RATIO_MAX cross-reference each other reciprocally
- **WHEN** `grep "TEMPORAL_NYQUIST_RATIO_MAX" sleap_roots/circumnutation/_constants.py` and `grep "NYQUIST_RATIO_MAX" sleap_roots/circumnutation/_constants.py` are inspected
- **THEN** the docstring or surrounding commentary on `NYQUIST_RATIO_MAX` mentions `TEMPORAL_NYQUIST_RATIO_MAX` AND identifies it as the temporal sibling (spatial-vs-temporal dimensional separation)
- **AND** the docstring or surrounding commentary on `TEMPORAL_NYQUIST_RATIO_MAX` reciprocally references `NYQUIST_RATIO_MAX` AND identifies it as the spatial sibling
- **AND** both constants have the same default value (`0.25`) but the dimensional separation lives in the constant NAMES + docstrings, NOT in different values

## ADDED Requirements

### Requirement: Tier 1 nutation trait emission API
The system SHALL provide `sleap_roots.circumnutation.nutation.compute(trajectory_df: pd.DataFrame, cadence_s: float, coordinate: str = "lateral", constants: Optional[ConstantsT] = None) -> pd.DataFrame`. The function SHALL accept the documented signature with `cadence_s` as an explicit positional parameter (mirroring `temporal_cwt.compute_scaleogram`'s precedent — `nutation` is the first cadence-consuming tier; `cadence_s` is NOT carried via `trajectory_df.attrs`). The `coordinate` parameter SHALL accept exactly one of `{"lateral", "x", "y"}`, with default `"lateral"` per `docs/circumnutation/roadmap.md` CC-7.

The function SHALL emit a per-track DataFrame whose rows correspond to the unique 5-tuple `(series, sample_uid, plate_id, plant_id, track_id)` derived from `trajectory_df` via `groupby(_IDENTITY_5_TUPLE, dropna=False, sort=False)`. The returned DataFrame SHALL contain the 8 row-identity columns (per Requirement: Trait CSV row-identity schema) followed by exactly 8 trait columns in this declared order: `T_nutation_median: float64`, `T_nutation_iqr: float64`, `A_nutation_envelope_max: float64`, `band_power_ratio: float64`, `noise_floor_estimate: float64`, `is_nutating: bool`, `period_residual_vs_derr_reference: float64`, `cadence_nyquist_ratio: float64`.

The function SHALL compose the existing PR #5 `temporal_cwt.compute_scaleogram` + `extract_ridge` primitives with the new PR #6 `temporal_cwt.smooth_ridge` primitive (per Requirement: Temporal CWT ridge-continuity smoothing API), the new `_geometry.project_to_growth_axis_perpendicular` lateral-projection helper, the new `_noise.compute_sg_detrended` Savitzky-Golay detrending helper (window = `SG_WINDOW_DETREND = 23`, polynomial_order = `SG_DEGREE = 3` per `preliminary_results_2026-05-07.md` §3.4), and the new `_noise.compute_fourier_noise_floor` Fourier noise-floor helper (per `roadmap.md` CC-8: median amplitude over `f > NOISE_FLOOR_OUT_OF_BAND_FACTOR / T_nutation_median`). The 9-step per-track pipeline is documented in `design.md` D5.

The function SHALL emit NaN for exactly 3 strictly biological-meaning-dependent traits when `is_nutating == False`: `T_nutation_median`, `T_nutation_iqr`, `A_nutation_envelope_max`. The other 5 trait columns SHALL be ALWAYS POPULATED (never NaN-gated by `is_nutating`): `is_nutating` (the gate boolean itself), `band_power_ratio` (the SNR-like precursor), `noise_floor_estimate` (the noise precursor), `period_residual_vs_derr_reference` (ridge-of-noise diagnostic — informs WHERE the spectral peak landed even on noise-driven inputs), `cadence_nyquist_ratio` (engineering diagnostic — answers "could we have observed nutation if it were present?" independent of biology). This split prevents downstream consumers from being unable to distinguish "no biological oscillation" from "cadence aliasing" from "ridge-of-noise" via trait inspection alone.

The function SHALL be deterministic per CC-6: same input → bit-identical 7 float columns + `is_nutating` boolean across calls in the same process (`atol=0`) AND identical to within `atol=1e-6` across Ubuntu / Windows / macOS CI runners (tolerance loosened from PR #5's `atol=1e-9` baseline per PR #6 `design.md` Round-2 S6: PR #6 composes 4 unverified scipy paths on top of PR #5's verified pywt path — `scipy.fft.rfft`, `scipy.ndimage.median_filter`, `scipy.signal.savgol_filter`, `scipy.stats.iqr` — and `atol=1e-6` is scientifically irrelevant for these traits per CC-6's "either 1e-9 OR documented looser"). The function SHALL log exactly one `logger.debug` message at the start (after input validation) whose text begins with `"nutation.compute("` and contains the tokens `n_tracks=`, `coordinate=`, `cadence_s=`. No INFO, WARNING, or ERROR log records SHALL be emitted on the happy path.

The function SHALL validate inputs strictly: `trajectory_df` validation delegates to `_validate_trajectory_df` (per Requirement: Tier 0 input-validation boundary); `cadence_s` validation mirrors PR #5's `_validate_cadence_s` (Python int/float or numpy integer/floating; explicit `bool` and `np.bool_` rejection; positive finite); `coordinate` SHALL be in `{"lateral", "x", "y"}` else ValueError; `constants` SHALL be None or `ConstantsT` else TypeError. The stationary-track failure path (`_geometry.project_to_growth_axis_perpendicular` returns `np.full(n, np.nan)` per the graceful-NaN policy) SHALL produce an all-NaN trait row with `is_nutating=False` rather than raising or emitting `np.RuntimeWarning("All-NaN slice encountered")`.

#### Scenario: nutation.compute returns a DataFrame with the documented column order and dtypes
- **GIVEN** a valid `trajectory_df` with 6 tracks (the Nipponbare proofread fixture) and `cadence_s = 300.0`
- **WHEN** `nutation.compute(trajectory_df, cadence_s=300.0)` is invoked
- **THEN** the returned object is a `pandas.DataFrame`
- **AND** the column order is: 8 row-identity columns from `ROW_IDENTITY_COLUMNS` followed by the 8 trait columns in declared order `["T_nutation_median", "T_nutation_iqr", "A_nutation_envelope_max", "band_power_ratio", "noise_floor_estimate", "is_nutating", "period_residual_vs_derr_reference", "cadence_nyquist_ratio"]`
- **AND** trait dtypes are: 7 float64 (`T_nutation_median`, `T_nutation_iqr`, `A_nutation_envelope_max`, `band_power_ratio`, `noise_floor_estimate`, `period_residual_vs_derr_reference`, `cadence_nyquist_ratio`) and 1 bool (`is_nutating`)
- **AND** row-identity 5-tuple `(series, sample_uid, plate_id, plant_id, track_id)` uniqueness holds: `df[list(_IDENTITY_5_TUPLE)].duplicated().sum() == 0`

#### Scenario: nutation.compute NaN-gates only 3 strictly biological-meaning-dependent traits when is_nutating is False
- **GIVEN** a noise-only input constructed via `synthetic.generate_trajectory(amplitude_px=0.0, noise_sigma_px=1.0, n_frames=1024, cadence_s=300, random_state=0)`
- **WHEN** `nutation.compute(trajectory_df, cadence_s=300.0)` is invoked on the converted trajectory_df
- **THEN** `result.is_nutating.iloc[0] == False`
- **AND** `np.isnan(result.T_nutation_median.iloc[0])` AND `np.isnan(result.T_nutation_iqr.iloc[0])` AND `np.isnan(result.A_nutation_envelope_max.iloc[0])` (the 3 NaN-gated traits)
- **AND** `np.isfinite(result.band_power_ratio.iloc[0])` AND `np.isfinite(result.noise_floor_estimate.iloc[0])` AND `np.isfinite(result.period_residual_vs_derr_reference.iloc[0])` AND `np.isfinite(result.cadence_nyquist_ratio.iloc[0])` (the 4 always-populated diagnostic + precursor traits remain finite)

#### Scenario: nutation.compute rejects invalid coordinate value with ValueError naming the field
- **WHEN** `nutation.compute(trajectory_df, cadence_s=300.0, coordinate=v)` is invoked for any of `v ∈ {"", "X", 1, None, "longitudinal"}`
- **THEN** `ValueError` is raised
- **AND** the exception message contains the substring `"coordinate"`

#### Scenario: nutation.compute rejects invalid cadence_s value with ValueError naming the field
- **WHEN** `nutation.compute(trajectory_df, cadence_s=v)` is invoked for any of `v ∈ {0, -1.0, float("nan"), float("inf"), float("-inf")}`
- **THEN** `ValueError` is raised
- **AND** the exception message contains the substring `"cadence_s"`

#### Scenario: nutation.compute rejects invalid cadence_s type with TypeError naming the field
- **WHEN** `nutation.compute(trajectory_df, cadence_s=v)` is invoked for any of `v ∈ {True, np.bool_(True), "300", [300.0]}`
- **THEN** `TypeError` is raised
- **AND** the exception message contains the substring `"cadence_s"`

#### Scenario: nutation.compute is deterministic across runs and across OSs
- **GIVEN** a valid `trajectory_df` and `cadence_s = 300.0`
- **WHEN** `nutation.compute(trajectory_df, cadence_s=300.0)` is invoked twice in the same Python process
- **THEN** the 7 float trait columns are bit-identical at `atol=0`
- **AND** the `is_nutating` boolean column is equal
- **AND** the captured 3-value canary at `[T_nutation_median, band_power_ratio, noise_floor_estimate]` on `synthetic.generate_trajectory(random_state=0, n_frames=575, T_nutation_s=3333, cadence_s=300, noise_sigma_px=0.5)` matches the hardcoded expected values to within `atol=1e-6` across Ubuntu / Windows / macOS CI runners AT THE TIME OF PR-MERGE; canary values are regression-detection sentinels and MAY be re-captured (in a follow-up commit cross-referencing this scenario) if upstream BLAS / scipy / pywt / numpy semantics legitimately shift after merge

#### Scenario: nutation.compute emits exactly one DEBUG logger record on the happy path
- **GIVEN** a valid `trajectory_df` and `cadence_s = 300.0` and `caplog.set_level(logging.DEBUG)`
- **WHEN** `nutation.compute(trajectory_df, cadence_s=300.0)` is invoked
- **THEN** exactly one log record at level `DEBUG` is emitted by the logger `sleap_roots.circumnutation.nutation`
- **AND** the record's message starts with `"nutation.compute("`
- **AND** the record's message contains each of the tokens `"n_tracks="`, `"coordinate="`, `"cadence_s="`
- **AND** no `INFO` / `WARNING` / `ERROR` / `CRITICAL` records are emitted

#### Scenario: nutation.compute handles stationary tracks gracefully via NaN trait emission
- **GIVEN** a `trajectory_df` containing one track whose `(tip_x, tip_y)` net displacement is zero (e.g., a closed-loop trajectory with `x[-1]==x[0] AND y[-1]==y[0]` but varying intermediate frames) and `cadence_s = 300.0`
- **WHEN** `nutation.compute(trajectory_df, cadence_s=300.0, coordinate="lateral")` is invoked
- **THEN** the call returns a `pandas.DataFrame` without raising
- **AND** `result.is_nutating.iloc[0] == False`
- **AND** the 3 NaN-gated traits (`T_nutation_median`, `T_nutation_iqr`, `A_nutation_envelope_max`) are NaN
- **AND** no `np.RuntimeWarning("All-NaN slice encountered")` is emitted (the implementation short-circuits the rest of the pipeline when `_geometry.project_to_growth_axis_perpendicular` returns an all-NaN signal)

#### Scenario: Layer-2 Derr forensic-match acceptance on the Nipponbare proofread fixture
- **GIVEN** the 6 tracks of `tests/data/circumnutation_nipponbare_plate_001/plate_001_greyscale.tracked_proofread.slp` loaded via `Series.load(...).get_tracked_tips()` filtered by `track_id ∈ {0, 1, 2, 3, 4, 5}` and `cadence_s = 300.0`
- **WHEN** `nutation.compute(trajectory_df_for_track_i, cadence_s=300.0, coordinate="lateral")` is invoked for each track and the resulting `period_residual_vs_derr_reference` and `is_nutating` values are collected
- **THEN** the median of the 6 per-track `period_residual_vs_derr_reference` values satisfies `abs(np.median(per_track_residuals)) < 0.02` (CC-7 median enforcement: the median across tracks must agree with Derr's published 3333 s reference to within ±2%, robust to per-plant biological variance)
- **AND** at least 4 of the 6 tracks satisfy `abs(period_residual_vs_derr_reference) < 0.05 AND is_nutating == True` (per-track count check acknowledging biological variance — allows 2 outlier tracks)

#### Scenario: GitHub issue #214 acceptance — ridge-continuity post-filter reduces T_nutation_iqr on plate-001
- **GIVEN** the 6 tracks of the Nipponbare proofread fixture and `cadence_s = 300.0`
- **WHEN** for each track, both `T_nutation_iqr_raw` (using `extract_ridge` only) and `T_nutation_iqr_post_filter` (using `extract_ridge` + `smooth_ridge`) are computed separately
- **THEN** at least 5 of the 6 tracks satisfy `T_nutation_iqr_post_filter < T_nutation_iqr_raw` — confirming the median-window post-filter materially reduces ridge-jitter-induced IQR inflation
- **AND** if fewer than 5 tracks improve, per-track values are recorded in the GREEN-phase Reconciliation Appendix and the threshold may be re-anchored to ≥3 of 6 with documentation

### Requirement: Temporal CWT ridge-continuity smoothing API
The system SHALL provide `sleap_roots.circumnutation.temporal_cwt.smooth_ridge(ridge_result: RidgeResult, window: Optional[int] = None, constants: Optional[ConstantsT] = None) -> RidgeResult`. The function SHALL apply a median-filter post-filter to `ridge_result.periods_s` to suppress scale-grid hopping artifacts in PR #5's per-frame argmax ridge — closing GitHub issue [#214](https://github.com/talmolab/sleap-roots/issues/214) (Mallat 1999 §4.4.2 ridge-continuity baseline).

The function SHALL median-filter ONLY the `periods_s` field (using `scipy.ndimage.median_filter(...mode='nearest')`). The other 4 fields (`frame_indices`, `amplitudes`, `powers`, `in_coi`) SHALL be carried through to the returned `RidgeResult` unchanged. Rationale: issue #214's acceptance is period-IQR-focused; `A_nutation_envelope_max` is a PEAK statistic computed from `amplitudes` and would be distorted by smoothing without a corresponding accuracy benefit; `in_coi` is a function of the original ridge scale indices and would require access to the COI mask grid (not available from RidgeResult alone) to recompute; `powers = amplitudes²` is a tautology that doesn't change under period-smoothing; `frame_indices = np.arange(n_frames)` is similarly invariant.

The function SHALL accept `window` as a `kwargs`-precedence override (when not None); otherwise SHALL use `resolved_constants.RIDGE_CONTINUITY_FILTER_WINDOW` (default `5`). The function SHALL emit NO trait values (composition with `nutation.compute` is documented under Requirement: Tier 1 nutation trait emission API). The function SHALL be deterministic: same input → identical `RidgeResult` output at `atol=0`. The function SHALL log exactly one `logger.debug` message at the start (after input validation) whose text begins with `"smooth_ridge("` and contains tokens `n_frames=` and `window=`.

The function SHALL validate inputs strictly: `ridge_result` SHALL be a `RidgeResult` instance (anything else raises `TypeError`); `window` SHALL be an `int` (when not None); `window >= 1` (raise `ValueError` if zero or negative); `window` SHALL be odd (raise `ValueError` if even — `scipy.ndimage.median_filter` requires odd-size for symmetric neighborhood); `constants` SHALL be `None` or a `ConstantsT` instance.

#### Scenario: smooth_ridge returns a RidgeResult with smoothed periods_s and unchanged other fields
- **GIVEN** a valid `RidgeResult` produced by `extract_ridge(compute_scaleogram(x, 300.0))` on a length-≥9 finite array
- **WHEN** `smooth_ridge(ridge_result)` is invoked (with default `window=None` → resolved to `RIDGE_CONTINUITY_FILTER_WINDOW=5`)
- **THEN** the returned object is a `RidgeResult` instance
- **AND** `np.array_equal(result.amplitudes, ridge_result.amplitudes)` (carried through unchanged)
- **AND** `np.array_equal(result.powers, ridge_result.powers)` (carried through unchanged)
- **AND** `np.array_equal(result.in_coi, ridge_result.in_coi)` (carried through unchanged)
- **AND** `np.array_equal(result.frame_indices, ridge_result.frame_indices)` (carried through unchanged)
- **AND** `np.array_equal(result.periods_s, scipy.ndimage.median_filter(ridge_result.periods_s, size=5, mode='nearest'))` (the smoothed periods)

#### Scenario: smooth_ridge accepts window override via positional kwarg
- **GIVEN** a valid `RidgeResult`
- **WHEN** `smooth_ridge(ridge_result, window=11)` is invoked
- **THEN** `result.periods_s == scipy.ndimage.median_filter(ridge_result.periods_s, size=11, mode='nearest')`
- **AND** the other 4 fields are carried through unchanged

#### Scenario: smooth_ridge accepts window override via constants
- **GIVEN** a valid `RidgeResult`
- **WHEN** `smooth_ridge(ridge_result, constants=ConstantsT(RIDGE_CONTINUITY_FILTER_WINDOW=11))` is invoked
- **THEN** `result.periods_s == scipy.ndimage.median_filter(ridge_result.periods_s, size=11, mode='nearest')` (the constants override takes effect when `window` kwarg is None)

#### Scenario: smooth_ridge rejects non-RidgeResult input with TypeError
- **WHEN** `smooth_ridge(x)` is invoked with `x` equal to `None` OR `{}` OR `(1, 2, 3)` OR an arbitrary numpy ndarray
- **THEN** `TypeError` is raised
- **AND** the exception message references the expected type `RidgeResult`

#### Scenario: smooth_ridge rejects non-positive or even window with ValueError
- **WHEN** `smooth_ridge(ridge_result, window=v)` is invoked for any of `v ∈ {0, -1, 4}` (zero, negative, even)
- **THEN** `ValueError` is raised
- **AND** the exception message contains the substring `"window"` and identifies the offending condition (non-positive or even)

#### Scenario: smooth_ridge emits exactly one DEBUG logger record on the happy path
- **GIVEN** a valid `RidgeResult` and `caplog.set_level(logging.DEBUG)`
- **WHEN** `smooth_ridge(ridge_result)` is invoked
- **THEN** exactly one log record at level `DEBUG` is emitted by the logger `sleap_roots.circumnutation.temporal_cwt`
- **AND** the record's message starts with `"smooth_ridge("`
- **AND** the record's message contains each of the tokens `"n_frames="`, `"window="`
- **AND** no `INFO` / `WARNING` / `ERROR` / `CRITICAL` records are emitted

#### Scenario: smooth_ridge is deterministic
- **GIVEN** a valid `RidgeResult`
- **WHEN** `smooth_ridge(ridge_result)` is invoked twice
- **THEN** `np.array_equal(result1.periods_s, result2.periods_s)` is True at `atol=0`
- **AND** all other 4 fields are equal under `np.array_equal` at `atol=0`
