# circumnutation Specification — PR #5 delta

## MODIFIED Requirements

### Requirement: Package layout
The system SHALL provide a `sleap_roots.circumnutation` sub-package whose import-tree is complete from this PR onward — every module name referenced anywhere in `docs/circumnutation/roadmap.md` exists as an importable Python module from PR #1 forward, even if its functions raise `NotImplementedError` until later PRs land. The package SHALL contain:

- 7 contract modules: `__init__.py`, `_constants.py`, `_types.py`, `_io.py`, `units.py`, `_noise.py`, `_geometry.py`
- 4 implementation modules: `kinematics` (implemented from PR #2 onward; see Requirement: Tier 0 raw kinematic traits), `qc` (implemented from PR #3 onward; see Requirement: QC tier per-track quality traits), `synthetic` (implemented from PR #4 onward; see Requirement: Synthetic trajectory generator), and `temporal_cwt` (implemented from PR #5 onward; see Requirement: Temporal CWT scaleogram API and Requirement: Temporal CWT ridge API)
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

The `kinematics` module SHALL be importable on the same terms as the other modules (clean import, namespaced logger) and SHALL expose `compute(trajectory_df, constants=None)` per Requirement: Tier 0 raw kinematic traits. The `qc` module SHALL be importable on the same terms and SHALL expose `compute(trajectory_df, constants=None)` per Requirement: QC tier per-track quality traits. The `synthetic` module SHALL be importable on the same terms and SHALL expose `generate_trajectory(...)` per Requirement: Synthetic trajectory generator. The `temporal_cwt` module SHALL be importable on the same terms and SHALL expose `compute_scaleogram(x, cadence_s, constants=None) -> ScaleogramResult` and `extract_ridge(scaleogram_result, constants=None) -> RidgeResult` per Requirement: Temporal CWT scaleogram API and Requirement: Temporal CWT ridge API. Unlike the stub modules, calling `kinematics.compute`, `qc.compute`, `synthetic.generate_trajectory`, or `temporal_cwt.compute_scaleogram` with a valid input SHALL NOT raise `NotImplementedError`.

The `_noise` and `_geometry` modules are private (underscore-prefixed) shared internals; their canonical callables are documented under Requirement: Tier 0 helper modules.

#### Scenario: All stub modules import cleanly
- **WHEN** a user runs `import sleap_roots.circumnutation as c; import sleap_roots.circumnutation.kinematics, sleap_roots.circumnutation.qc, sleap_roots.circumnutation.synthetic, sleap_roots.circumnutation.temporal_cwt, sleap_roots.circumnutation.psi_g, sleap_roots.circumnutation.midline, sleap_roots.circumnutation.spatial_cwt, sleap_roots.circumnutation.parametric, sleap_roots.circumnutation.plotting, sleap_roots.circumnutation.pipeline`
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
The system SHALL expose all overridable defaults as module-level named constants in `sleap_roots/circumnutation/_constants.py`. The set SHALL include at minimum: `NOISE_MASK_K`, `LGZ_STEADY_STATE_RESIDUAL_MAX`, `NYQUIST_RATIO_MAX`, `SG_D2_AGREEMENT_MAX`, `SG_MSD_AGREEMENT_MAX`, `D2_MSD_AGREEMENT_MAX`, `LGZ_NMIN_RESOLVABLE`, `COI_FRACTION_MAX`, `BAND_POWER_NOISE_RATIO`, `WAVELET_DEFAULT_TEMPORAL`, `WAVELET_DEFAULT_SPATIAL`, `SG_WINDOW_SHORT`, `SG_DEGREE`, `SG_WINDOW_DETREND`, `OUTLIER_STEP_RATIO`, `FRAC_OUTLIER_STEPS_MAX`, `WORST_STEP_RATIO_MAX`, `GROWTH_AXIS_RELIABILITY_K`, `SYNTHETIC_T_NUTATION_S`, `SYNTHETIC_AMPLITUDE_PX`, `SYNTHETIC_GROWTH_RATE_PX_PER_FRAME`, `SYNTHETIC_NOISE_SIGMA_PX`, `SYNTHETIC_CADENCE_S`, `SYNTHETIC_N_FRAMES`, `SYNTHETIC_GROWTH_AXIS_ANGLE_RAD`, `COI_EFOLDING_FACTOR`, `CWT_SCALE_COUNT_DEFAULT`, `CWT_PERIOD_MIN_NYQUIST_FACTOR`, `CWT_PERIOD_MAX_SIGNAL_FRACTION`, `_SCHEMA_VERSION`, `_CONSTANTS_VERSION`. The values SHALL match the defaults in `docs/circumnutation/roadmap.md` cross-cutting concern CC-2 and `docs/circumnutation/theory.md` §7.6 (for the QC-tier-introduced thresholds: `FRAC_OUTLIER_STEPS_MAX = 0.05`, `WORST_STEP_RATIO_MAX = 5`, `SG_MSD_AGREEMENT_MAX = 1.5`, `D2_MSD_AGREEMENT_MAX = 1.5`) and `docs/circumnutation/preliminary_results_2026-05-07.md` §1, §3.4, §4.1, §4.3 (for the synthetic-generator defaults: `SYNTHETIC_T_NUTATION_S = 3333.0`, `SYNTHETIC_AMPLITUDE_PX = 10.0`, `SYNTHETIC_GROWTH_RATE_PX_PER_FRAME = 4.29`, `SYNTHETIC_NOISE_SIGMA_PX = 2.0`, `SYNTHETIC_CADENCE_S = 300.0`, `SYNTHETIC_N_FRAMES = 575`, `SYNTHETIC_GROWTH_AXIS_ANGLE_RAD = math.pi / 2`) and the wavelet-aware step-response derivation in this change's `design.md` D3 (for the CWT-machinery defaults: `COI_EFOLDING_FACTOR = math.sqrt(1.5)` calibrated for `cmor1.5-1.0` per `√B` envelope e-folding; `CWT_SCALE_COUNT_DEFAULT = 64`; `CWT_PERIOD_MIN_NYQUIST_FACTOR = 2.0`; `CWT_PERIOD_MAX_SIGNAL_FRACTION = 0.25`); `_SCHEMA_VERSION` SHALL be `1` (unchanged from PR #1) and `_CONSTANTS_VERSION` SHALL be `4` (bumped from `3` in this PR per the version-sentinel contract — the constants set grew by 4). The module SHALL also expose `PIPELINE_UNIT_VOCABULARY` (px-based + calibration-independent units, the closed sidecar vocabulary), `CONVERTED_UNIT_VOCABULARY` (mm-based units produced by `convert_to_mm`), and `VALID_UNIT_VOCABULARY` (their union), plus `ROW_IDENTITY_UNITS` (the canonical units dict for the eight row-identity columns).

The `ConstantsT` typed override-bag SHALL include corresponding fields for every overridable constant above, so callers can override per-call via `ConstantsT(COI_EFOLDING_FACTOR=math.sqrt(2.0))` etc. `_default_constants_snapshot()` SHALL emit every constant name in the set above into the run-metadata sidecar, including the four new CWT-machinery constants.

**Scope note on `NYQUIST_RATIO_MAX` docstring touch.** PR #5 ALSO modifies the docstring of the existing `NYQUIST_RATIO_MAX` constant to add a cross-reference to the new `CWT_PERIOD_MAX_SIGNAL_FRACTION` constant (numerically equal at `0.25` but semantically distinct — `NYQUIST_RATIO_MAX` is a QC alias-protection threshold for the cadence-Nyquist gate, while `CWT_PERIOD_MAX_SIGNAL_FRACTION` is the CWT scale-range upper bound). This is an internal-docstring-only touch with no behavioral change; it is enumerated here in the spec delta so the touch is explicitly scope-bounded and not surprising to reviewers of the `_constants.py` diff.

#### Scenario: All required constants are importable with correct types
- **WHEN** a user runs `from sleap_roots.circumnutation import _constants`
- **THEN** every name listed above is an attribute of `_constants`
- **AND** each value matches the documented default in `roadmap.md` CC-2, `theory.md` §7.6, `preliminary_results_2026-05-07.md` §1/§3.4/§4.1/§4.3, and this change's `design.md` D3/D7
- **AND** `_constants._SCHEMA_VERSION` is the integer `1`
- **AND** `_constants._CONSTANTS_VERSION` is the integer `4`
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

#### Scenario: Constants snapshot includes the four QC constants, the seven synthetic constants, and the four CWT-machinery constants
- **WHEN** `_default_constants_snapshot()` is called
- **THEN** the returned mapping contains `FRAC_OUTLIER_STEPS_MAX`, `WORST_STEP_RATIO_MAX`, `SG_MSD_AGREEMENT_MAX`, `D2_MSD_AGREEMENT_MAX` with their default values
- **AND** the returned mapping contains `SYNTHETIC_T_NUTATION_S`, `SYNTHETIC_AMPLITUDE_PX`, `SYNTHETIC_GROWTH_RATE_PX_PER_FRAME`, `SYNTHETIC_NOISE_SIGMA_PX`, `SYNTHETIC_CADENCE_S`, `SYNTHETIC_N_FRAMES`, `SYNTHETIC_GROWTH_AXIS_ANGLE_RAD` with their default values
- **AND** the returned mapping contains `COI_EFOLDING_FACTOR`, `CWT_SCALE_COUNT_DEFAULT`, `CWT_PERIOD_MIN_NYQUIST_FACTOR`, `CWT_PERIOD_MAX_SIGNAL_FRACTION` with their default values

#### Scenario: NYQUIST_RATIO_MAX docstring cross-references CWT_PERIOD_MAX_SIGNAL_FRACTION
- **WHEN** `sleap_roots.circumnutation._constants.NYQUIST_RATIO_MAX` is inspected via `inspect.getdoc(...)` or via `_constants.__dict__["NYQUIST_RATIO_MAX"].__doc__` (note: module-level constants do not carry docstrings natively; the cross-reference lives in the module-level constant's surrounding docstring or in a `__doc__`-attached commentary block defined at the same point — implementation may use either pattern as long as `grep "CWT_PERIOD_MAX_SIGNAL_FRACTION" sleap_roots/circumnutation/_constants.py` shows the cross-reference text within ±5 lines of `NYQUIST_RATIO_MAX:` declaration)
- **THEN** the cross-reference text mentions `CWT_PERIOD_MAX_SIGNAL_FRACTION` AND notes the numerical-equality-but-semantically-distinct relationship
- **AND** `CWT_PERIOD_MAX_SIGNAL_FRACTION`'s docstring reciprocally references `NYQUIST_RATIO_MAX`

## ADDED Requirements

### Requirement: Temporal CWT scaleogram API
The system SHALL provide `sleap_roots.circumnutation.temporal_cwt.compute_scaleogram(x: np.ndarray, cadence_s: float, constants: Optional[ConstantsT] = None) -> ScaleogramResult`. The function SHALL accept the canonical `(x, cadence_s, constants=None)` signature locked by the foundation's Package layout requirement. The function SHALL compute a temporal Continuous Wavelet Transform using the `cmor1.5-1.0` mother wavelet by default (overridable via `constants.WAVELET_DEFAULT_TEMPORAL`), at log-spaced scales over an auto-derived period range `[constants.CWT_PERIOD_MIN_NYQUIST_FACTOR * cadence_s, constants.CWT_PERIOD_MAX_SIGNAL_FRACTION * len(x) * cadence_s]`, returning a frozen `ScaleogramResult` containing the complex-valued scaleogram, scales axis, period axis (derived via `pywt.scale2frequency` round-trip for wavelet-agnostic correctness), frequency axis, cone-of-influence boolean mask (computed via wavelet-aware `constants.COI_EFOLDING_FACTOR * scale` per `design.md` D3), the resolved `cadence_s`, and the resolved wavelet name.

The function SHALL emit NO trait values (PR #6 owns trait emission). The function SHALL be deterministic per CC-6: same input → bit-identical scaleogram across calls in the same process AND identical to within `atol=1e-9` across Ubuntu / Windows / macOS CI runners. The function SHALL log exactly one `logger.debug` message at the start (after input validation) whose text begins with `"compute_scaleogram("` and contains the named tokens `n_frames=`, `cadence_s=`, `n_scales=`, `period_min_s=`, `period_max_s=`, `wavelet=`. No INFO, WARNING, or ERROR log records SHALL be emitted on the happy path.

The function SHALL validate inputs strictly: `x` SHALL be a 1-D `np.ndarray` (coercible from integer/float dtypes; rejecting `complex` and `object`); `x` SHALL be all-finite (no NaN, no ±inf); `len(x)` SHALL be ≥ `MIN_FRAMES_REQUIRED` derived at call time as `int(math.floor(constants.CWT_PERIOD_MIN_NYQUIST_FACTOR / constants.CWT_PERIOD_MAX_SIGNAL_FRACTION)) + 1` (= `9` at defaults); positive-finite guards SHALL fire on `constants.CWT_PERIOD_MAX_SIGNAL_FRACTION` and `constants.CWT_PERIOD_MIN_NYQUIST_FACTOR`; `cadence_s` SHALL be a Python `int` or `float` (rejecting `bool` subtype of `int` and `str`); `cadence_s > 0` and `math.isfinite(cadence_s)`; `constants` SHALL be `None` or a `ConstantsT` instance. Invalid inputs SHALL raise `ValueError` or `TypeError` with the offending field name embedded in the message.

The `ScaleogramResult` class SHALL be an `@attrs.define(frozen=True, slots=False, kw_only=True)` container with exactly the following fields (in this order): `scaleogram: np.ndarray` (shape `(n_scales, n_frames)`, dtype `complex128`); `scales: np.ndarray` (shape `(n_scales,)`, dtype `float64`, monotonically increasing); `periods_s: np.ndarray` (shape `(n_scales,)`, dtype `float64`); `frequencies_hz: np.ndarray` (shape `(n_scales,)`, dtype `float64`, equal to `1.0 / periods_s` to within numerical precision); `coi_mask: np.ndarray` (shape `(n_scales, n_frames)`, dtype `bool`; `True` indicates inside-COI = unreliable); `cadence_s: float`; `wavelet: str`.

#### Scenario: compute_scaleogram returns a ScaleogramResult with the documented field shapes and dtypes
- **GIVEN** a valid 1-D `float64` ndarray `x` of length 575 with all-finite values, and `cadence_s = 300.0`
- **WHEN** `compute_scaleogram(x, 300.0)` is invoked
- **THEN** the returned object is a `ScaleogramResult` instance
- **AND** `result.scaleogram.shape == (CWT_SCALE_COUNT_DEFAULT, 575)` and `dtype == np.complex128`
- **AND** `result.scales.shape == (CWT_SCALE_COUNT_DEFAULT,)`, `dtype == np.float64`, strictly monotonically increasing
- **AND** `result.periods_s.shape == (CWT_SCALE_COUNT_DEFAULT,)` and `dtype == np.float64`
- **AND** `np.allclose(result.frequencies_hz * result.periods_s, 1.0, atol=1e-12)`
- **AND** `result.coi_mask.shape == result.scaleogram.shape` and `dtype == bool`
- **AND** `result.cadence_s == 300.0` and `result.wavelet == "cmor1.5-1.0"`

#### Scenario: ScaleogramResult is a frozen attrs class with the seven documented fields
- **WHEN** `attrs.fields(ScaleogramResult)` is inspected
- **THEN** the field names are exactly `("scaleogram", "scales", "periods_s", "frequencies_hz", "coi_mask", "cadence_s", "wavelet")` in that order
- **AND** attempting `result.scaleogram = new_array` raises `attrs.exceptions.FrozenInstanceError`

#### Scenario: compute_scaleogram rejects non-finite x with ValueError naming the field
- **WHEN** `compute_scaleogram(x, 300.0)` is invoked with `x` containing NaN OR `+inf` OR `-inf` at any index
- **THEN** `ValueError` is raised
- **AND** the exception message contains the substring `"x"` and identifies the non-finite condition

#### Scenario: compute_scaleogram rejects malformed x with ValueError or TypeError naming the field
- **WHEN** `compute_scaleogram(x, 300.0)` is invoked with `x.ndim != 1` (e.g., 2-D ndarray) OR `x.dtype == np.complex128` OR `x.dtype == object`
- **THEN** `ValueError` or `TypeError` is raised
- **AND** the exception message contains the substring `"x"` and identifies the offending shape or dtype

#### Scenario: compute_scaleogram rejects too-short x with ValueError naming the field
- **WHEN** `compute_scaleogram(x, 300.0)` is invoked with `len(x) < MIN_FRAMES_REQUIRED` (where MIN_FRAMES_REQUIRED = `int(math.floor(CWT_PERIOD_MIN_NYQUIST_FACTOR / CWT_PERIOD_MAX_SIGNAL_FRACTION)) + 1` = 9 at defaults)
- **THEN** `ValueError` is raised
- **AND** the exception message contains the substring `"x"` (or `"MIN_FRAMES"`) and reports the actual length

#### Scenario: compute_scaleogram accepts x at the exact MIN_FRAMES_REQUIRED floor without raising
- **GIVEN** a 1-D float64 ndarray `x` of length exactly `MIN_FRAMES_REQUIRED` (= 9 at defaults) with all-finite values
- **WHEN** `compute_scaleogram(x, 300.0)` is invoked
- **THEN** the call returns a `ScaleogramResult` without raising (paired positive boundary contract for the "too-short x" negative scenario above, per /openspec-review round-2 reviewer N-I3 — bidirectional contract on the documented floor)
- **AND** the returned `scaleogram.shape == (CWT_SCALE_COUNT_DEFAULT, 9)`

#### Scenario: compute_scaleogram rejects invalid cadence_s value with ValueError naming the field
- **WHEN** `compute_scaleogram(x, cadence_s)` is invoked with `cadence_s` equal to `0` OR `-1.0` OR `float("nan")` OR `float("inf")` OR `float("-inf")`
- **THEN** `ValueError` is raised
- **AND** the exception message contains the substring `"cadence_s"` and the offending value

#### Scenario: compute_scaleogram rejects invalid cadence_s type with TypeError naming the field
- **WHEN** `compute_scaleogram(x, cadence_s)` is invoked with `cadence_s` equal to `True` (Python bool) OR `np.bool_(True)` (numpy bool scalar — must be explicitly guarded since `np.bool_` is a numpy scalar subclass of `int`) OR `"300"` (str) OR `[300.0]` (list)
- **THEN** `TypeError` is raised
- **AND** the exception message contains the substring `"cadence_s"` and identifies the offending type

#### Scenario: compute_scaleogram is deterministic across runs
- **GIVEN** a valid `x` and `cadence_s`
- **WHEN** `compute_scaleogram(x, cadence_s)` is invoked twice in the same Python process
- **THEN** `np.array_equal(result1.scaleogram, result2.scaleogram)` is True at `atol=0`
- **AND** the captured 3-value canary at `[scale_idx_at_target, [coi_interior_indices]]` matches the hardcoded expected complex values to within `atol=1e-9` across Ubuntu / Windows / macOS CI runners AT THE TIME OF PR-MERGE; canary values are regression-detection sentinels and MAY be re-captured (in a follow-up commit cross-referencing this scenario) if upstream BLAS / pywt / numpy semantics legitimately shift after merge

#### Scenario: compute_scaleogram emits exactly one DEBUG logger record on the happy path
- **GIVEN** a valid `x` and `cadence_s` and `caplog.set_level(logging.DEBUG)`
- **WHEN** `compute_scaleogram(x, cadence_s)` is invoked
- **THEN** exactly one log record at level `DEBUG` is emitted by the logger `sleap_roots.circumnutation.temporal_cwt`
- **AND** the record's message starts with `"compute_scaleogram("`
- **AND** the record's message contains each of the tokens `"n_frames="`, `"cadence_s="`, `"n_scales="`, `"period_min_s="`, `"period_max_s="`, `"wavelet="`
- **AND** no `INFO` / `WARNING` / `ERROR` / `CRITICAL` records are emitted

#### Scenario: Proofread fixture (Nipponbare plate-001 6 tracks) does not raise and produces shape-correct output
- **GIVEN** the 6 tracks of `tests/data/circumnutation_nipponbare_plate_001/plate_001_greyscale.tracked_proofread.slp` loaded via `Series.load(...).get_tracked_tips()` (each track has 575 finite-float64 frames with zero NaN and zero frame gaps, per the pre-design empirical verification)
- **WHEN** `compute_scaleogram(track_x, cadence_s=300.0)` is invoked for each track
- **THEN** the call does not raise
- **AND** the returned `ScaleogramResult` has `scaleogram.shape == (CWT_SCALE_COUNT_DEFAULT, 575)`
- **AND** `coi_mask.shape == scaleogram.shape`
- **AND** the COI fraction at the scale nearest to period 3333 s (`scale_idx_at_target = int(np.argmin(np.abs(result.periods_s - 3333.0)))`) is well below the `COI_FRACTION_MAX = 0.5` reliability threshold; concretely, `result.coi_mask[scale_idx_at_target, :].mean() < 0.10` (~2× the empirically-measured 4.87% at the `√1.5` factor — see `design.md` D3 arithmetic)

### Requirement: Temporal CWT ridge API
The system SHALL provide `sleap_roots.circumnutation.temporal_cwt.extract_ridge(scaleogram_result: ScaleogramResult, constants: Optional[ConstantsT] = None) -> RidgeResult`. The function SHALL extract a per-frame ridge from the input scaleogram via deterministic per-frame argmax of `|scaleogram|` along the scale axis (numpy's documented tie-breaking returns the smallest index on equal values).

The function SHALL emit NO trait values. The function SHALL be deterministic: same `ScaleogramResult` input → identical `RidgeResult` output at `atol=0`. The function SHALL log exactly one `logger.debug` message at the start (after input validation) whose text begins with `"extract_ridge("` and contains tokens `n_scales=` and `n_frames=`. No INFO, WARNING, or ERROR log records SHALL be emitted on the happy path.

The function SHALL validate inputs strictly: `scaleogram_result` SHALL be a `ScaleogramResult` instance (anything else raises `TypeError`); empty `ScaleogramResult` (with `n_scales == 0` OR `n_frames == 0`) SHALL raise `ValueError`; `constants` SHALL be `None` or a `ConstantsT` instance.

The `RidgeResult` class SHALL be an `@attrs.define(frozen=True, slots=False, kw_only=True)` container with exactly the following fields (in this order): `frame_indices: np.ndarray` (shape `(n_frames,)`, dtype `int64`, equal to `np.arange(n_frames, dtype=np.int64)`); `periods_s: np.ndarray` (shape `(n_frames,)`, dtype `float64`; indexed by frame, NOT by scale — value at index `i` is the period AT THE RIDGE for frame `i`); `amplitudes: np.ndarray` (shape `(n_frames,)`, dtype `float64`, equal to `|C|` at the ridge cell); `powers: np.ndarray` (shape `(n_frames,)`, dtype `float64`, equal to `amplitudes ** 2 = |C|²` — redundant by construction, intentionally preserved per `design.md` D2 + Round-2 reviewer R2-N3 + Round-3 reviewer R3-N1, so downstream consumers can consume either `|C|` or `|C|²` without recomputing); `in_coi: np.ndarray` (shape `(n_frames,)`, dtype `bool`; `True` iff `scaleogram_result.coi_mask[ridge_scale_idx, frame_idx]`). The ridge SHALL NOT be pre-COI-masked; PR #6's trait emission applies the mask per the COI-masked language in `theory.md` §7.2.

#### Scenario: extract_ridge returns a RidgeResult with the documented field shapes and dtypes
- **GIVEN** a valid `ScaleogramResult` produced by `compute_scaleogram(x, 300.0)` where `len(x) == 575`
- **WHEN** `extract_ridge(scaleogram_result)` is invoked
- **THEN** the returned object is a `RidgeResult` instance
- **AND** `result.frame_indices.shape == (575,)` and `dtype == np.int64` and `np.array_equal(result.frame_indices, np.arange(575, dtype=np.int64))`
- **AND** `result.periods_s.shape == (575,)` and `dtype == np.float64`
- **AND** `result.amplitudes.shape == (575,)` and `dtype == np.float64` and `(result.amplitudes >= 0).all()`
- **AND** `result.powers.shape == (575,)` and `dtype == np.float64`
- **AND** `np.allclose(result.powers, result.amplitudes ** 2)` (redundancy preservation)
- **AND** `result.in_coi.shape == (575,)` and `dtype == bool`

#### Scenario: RidgeResult is a frozen attrs class with the five documented fields preserving the powers redundancy
- **WHEN** `attrs.fields(RidgeResult)` is inspected
- **THEN** the field names are exactly `("frame_indices", "periods_s", "amplitudes", "powers", "in_coi")` in that order
- **AND** attempting `ridge.amplitudes = new_array` raises `attrs.exceptions.FrozenInstanceError`
- **AND** `powers` is present (not removed in a future PR without a BREAKING-change spec revision)

#### Scenario: extract_ridge rejects non-ScaleogramResult input with TypeError
- **WHEN** `extract_ridge(x)` is invoked with `x` equal to `None` OR `{}` OR `(1, 2, 3)` OR `np.zeros((10, 10))`
- **THEN** `TypeError` is raised
- **AND** the exception message references the expected type `ScaleogramResult`

#### Scenario: extract_ridge rejects empty ScaleogramResult with ValueError
- **GIVEN** a `ScaleogramResult` constructed with `n_scales == 0` (scaleogram shape `(0, n_frames)`) OR `n_frames == 0` (scaleogram shape `(n_scales, 0)`)
- **WHEN** `extract_ridge(scaleogram_result)` is invoked
- **THEN** `ValueError` is raised
- **AND** the exception message references the empty-axis condition (e.g., `"n_scales == 0"` or `"n_frames == 0"`)

#### Scenario: extract_ridge rejects invalid constants type with TypeError
- **GIVEN** a valid `ScaleogramResult`
- **WHEN** `extract_ridge(scaleogram_result, constants=invalid)` is invoked with `invalid` not `None` and not a `ConstantsT` instance (e.g., `42`, `"foo"`, `{}`)
- **THEN** `TypeError` is raised
- **AND** the exception message references `"constants"`

#### Scenario: extract_ridge emits exactly one DEBUG logger record on the happy path
- **GIVEN** a valid `ScaleogramResult` and `caplog.set_level(logging.DEBUG)`
- **WHEN** `extract_ridge(scaleogram_result)` is invoked
- **THEN** exactly one log record at level `DEBUG` is emitted by the logger `sleap_roots.circumnutation.temporal_cwt`
- **AND** the record's message starts with `"extract_ridge("`
- **AND** the record's message contains each of the tokens `"n_scales="`, `"n_frames="`
- **AND** no `INFO` / `WARNING` / `ERROR` / `CRITICAL` records are emitted

#### Scenario: extract_ridge is deterministic
- **GIVEN** a valid `ScaleogramResult`
- **WHEN** `extract_ridge(scaleogram_result)` is invoked twice
- **THEN** `np.array_equal(result1.periods_s, result2.periods_s)` is True at `atol=0`
- **AND** the same holds for `amplitudes`, `powers`, `in_coi`, `frame_indices`
