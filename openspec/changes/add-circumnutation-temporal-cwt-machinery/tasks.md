# Tasks for add-circumnutation-temporal-cwt-machinery

TDD-ordered. Tests precede implementation per `superpowers:test-driven-development`. **Do not push commits from section 2 (red phase — tests only) without section 3 (implementation) in the same push** — the new test file imports `compute_scaleogram`, `extract_ridge`, `ScaleogramResult`, `RidgeResult`, `_coi_boundary_samples` from `temporal_cwt.py` which section 3 implements, and the foundation test migration (§2.I) flips `temporal_cwt` semantics in `STUBS_WITH_CONSTANTS_KWARG`. The suite is expected to be red between §2.x and §3.x; PR is one logical unit.

## 1. Pre-flight (no code changes; confirms ground-truth)

- [ ] 1.1 Confirm the existing fixture used by §2.H.1 is present: `tests/data/circumnutation_nipponbare_plate_001/plate_001_greyscale.tracked_proofread.slp` (committed via Git LFS in PR #2). Run `uv run python -c "from pathlib import Path; from sleap_roots.series import Series; s = Series.load(series_name='plate_001', primary_path=str(Path('tests/data/circumnutation_nipponbare_plate_001/plate_001_greyscale.tracked_proofread.slp'))); df = s.get_tracked_tips(); print(len(df), df['tip_x'].isna().sum())"` → expect **3450 rows** (575 × 6 tracks) and 0 NaN. Note: this 3450 figure depends on PR #2's `Series.get_tracked_tips` dedup-by-instance-type fix; the raw-LFS file contains 3906 instance rows before dedup (per /openspec-review round-1 Issue-S3 note). If your `len(df)` is 3906 instead of 3450, your `sleap-io` / `sleap_roots` version predates the PR #2 fix.
- [ ] 1.2 Run baseline: `uv run pytest tests/ -x` — confirm green BEFORE any changes. Foundation + Tier 0 + QC + synthetic tests must be passing at HEAD.
- [ ] 1.3 Run baseline coverage: `uv run pytest tests/ --cov=sleap_roots --cov-report=term --cov-fail-under=84` — confirm starting at 84%+ baseline.
- [ ] 1.4 Confirm `pywavelets>=1.5` is already in `pyproject.toml` runtime deps (PR #1 added it). Verify with `uv run python -c "import pywt; print(pywt.__version__); print(pywt.scale2frequency('cmor1.5-1.0', 1.0))"` → expect `1.0` (center_freq for cmor1.5-1.0).

## 2. Tests (write before implementation — TDD red phase)

- [ ] 2.1 Create `tests/test_circumnutation_temporal_cwt.py` with module docstring referencing the spec delta + `theory.md` §3.4/§6.5/§7.2/§7.6 + `design.md` sections D1–D9 + the Reconciliation Appendix (3 rounds of critical review).

  **Test-file imports** (per design.md "Test-file imports" section): the test module SHALL import:
  - `import math`
  - `from pathlib import Path`
  - `import attrs` (used by §2.A.5 / §2.A.10 for `attrs.has`, `attrs.fields`, `attrs.exceptions.FrozenInstanceError` — per /openspec-review round-1 TDD-I1)
  - `import numpy as np`
  - `import numpy.testing as npt`
  - `import pandas as pd`
  - `import pytest`
  - `from sleap_roots.circumnutation import synthetic`
  - `from sleap_roots.circumnutation._constants import (ConstantsT, _default_constants_snapshot, _CONSTANTS_VERSION, _SCHEMA_VERSION, COI_EFOLDING_FACTOR, CWT_SCALE_COUNT_DEFAULT, CWT_PERIOD_MIN_NYQUIST_FACTOR, CWT_PERIOD_MAX_SIGNAL_FRACTION, WAVELET_DEFAULT_TEMPORAL, FRAC_OUTLIER_STEPS_MAX, WORST_STEP_RATIO_MAX, SG_MSD_AGREEMENT_MAX, D2_MSD_AGREEMENT_MAX)`
  - `from sleap_roots.circumnutation.temporal_cwt import (compute_scaleogram, extract_ridge, ScaleogramResult, RidgeResult, _coi_boundary_samples)`
  - `from sleap_roots.series import Series`

  `pywt` and `scipy.stats` are NOT imported (pywt is internal to `temporal_cwt`; the chi-square test was replaced by max-fraction dispersion per round-3 R3-B1).

### 2.A — Schema/structural tests

- [ ] 2.A.1 Test `compute_scaleogram_returns_ScaleogramResult`: `result = compute_scaleogram(np.linspace(0, 100, 575), 300.0); assert isinstance(result, ScaleogramResult)`.

- [ ] 2.A.2 Test `ScaleogramResult_fields_and_shapes`: assert `result.scaleogram.shape == (CWT_SCALE_COUNT_DEFAULT, 575)`, `result.scaleogram.dtype == np.complex128`, `result.scales.shape == (CWT_SCALE_COUNT_DEFAULT,)`, `result.scales.dtype == np.float64`, `result.periods_s.shape == (CWT_SCALE_COUNT_DEFAULT,)`, `result.coi_mask.shape == result.scaleogram.shape`, `result.coi_mask.dtype == bool`.

- [ ] 2.A.3 Test `scales_strictly_monotonic_increasing`: `assert (np.diff(result.scales) > 0).all()`.

- [ ] 2.A.4 Test `periods_s_frequencies_hz_inverse_relation`: `assert np.allclose(result.frequencies_hz * result.periods_s, 1.0, atol=1e-12)`.

- [ ] 2.A.5 Test `ScaleogramResult_is_frozen_attrs_class`: `assert attrs.has(ScaleogramResult)`; assert field order via `[f.name for f in attrs.fields(ScaleogramResult)] == ["scaleogram", "scales", "periods_s", "frequencies_hz", "coi_mask", "cadence_s", "wavelet"]`; assert `with pytest.raises(attrs.exceptions.FrozenInstanceError): result.cadence_s = 999.0`.

- [ ] 2.A.6 Test `extract_ridge_returns_RidgeResult`: `result = compute_scaleogram(...); ridge = extract_ridge(result); assert isinstance(ridge, RidgeResult)`.

- [ ] 2.A.7 Test `RidgeResult_fields_and_shapes`: assert `ridge.frame_indices.shape == (575,)`, `ridge.frame_indices.dtype == np.int64`, `np.array_equal(ridge.frame_indices, np.arange(575, dtype=np.int64))`; same shape and float64 dtype for `periods_s`, `amplitudes`, `powers`; `ridge.in_coi.shape == (575,)` and `dtype == bool`.

- [ ] 2.A.8 Test `RidgeResult_powers_equals_amplitudes_squared`: `assert np.allclose(ridge.powers, ridge.amplitudes ** 2, atol=1e-15)` (the redundancy preservation).

- [ ] 2.A.9 Test `RidgeResult_amplitudes_non_negative`: `assert (ridge.amplitudes >= 0).all()`.

- [ ] 2.A.10 Test `RidgeResult_is_frozen_attrs_class`: field order `["frame_indices", "periods_s", "amplitudes", "powers", "in_coi"]`; frozen-instance error on assignment.

- [ ] 2.A.11 Test `caplog_no_warning_or_error_on_default_call`: with `caplog.set_level(logging.WARNING)`, call `compute_scaleogram(x, 300.0)` and `extract_ridge(result)`; assert no WARNING/ERROR records.

- [ ] 2.A.12 Test `caplog_debug_messages_contain_required_tokens`: with `caplog.set_level(logging.DEBUG)`, call both functions; assert at least one DEBUG record from `compute_scaleogram` starts with `"compute_scaleogram("` and contains tokens `n_frames=`, `cadence_s=`, `n_scales=`, `period_min_s=`, `period_max_s=`, `wavelet=`; at least one DEBUG record from `extract_ridge` starts with `"extract_ridge("` and contains `n_scales=`, `n_frames=`.

- [ ] 2.A.13 Test `scalar_field_values` (per /openspec-review round-1 TDD-I9): `x = np.linspace(0, 100, 64); result = compute_scaleogram(x, 300.0)`; assert `result.cadence_s == 300.0` (exact equality, dtype float) AND `result.wavelet == "cmor1.5-1.0"` (string). Covers the spec scenario at "compute_scaleogram returns a ScaleogramResult..." which asserts `result.cadence_s == 300.0` and `result.wavelet == "cmor1.5-1.0"` — neither was previously covered by §2.A.1–§2.A.12.

### 2.B — Determinism tests (CC-6)

- [ ] 2.B.1 Test `compute_scaleogram_same_input_bit_identical_in_process`: call twice on same input; assert `np.array_equal(result1.scaleogram, result2.scaleogram)` at `atol=0`. Same for `coi_mask`, `scales`, `periods_s`, `frequencies_hz`.

- [ ] 2.B.2 Test `extract_ridge_same_input_bit_identical_in_process`: call twice on same `ScaleogramResult`; assert `np.array_equal(ridge1.periods_s, ridge2.periods_s)` at `atol=0`. Same for `amplitudes`, `powers`, `in_coi`, `frame_indices`.

- [ ] 2.B.3 **Canary test** `cross_os_canary_at_atol_1e_9`: generate `x = synthetic.generate_trajectory(random_state=0, n_frames=128, T_nutation_s=3333, cadence_s=300, noise_sigma_px=0)["tip_x"].to_numpy()`; call `result = compute_scaleogram(x, 300.0)`; compute `scale_idx_at_target = int(np.argmin(np.abs(result.periods_s - 3333.0)))`; check 3 hardcoded complex values at `result.scaleogram[scale_idx_at_target, [frame_idx_0, frame_idx_mid, frame_idx_last_interior]]` (specific COI-interior indices captured via the repo-resident script at `scripts/circumnutation/capture_temporal_cwt_canary.py` — see §3.5) match the hardcoded expected at `atol=1e-9, rtol=0`. Include an inline `# TODO(PR #5 §3.5): replace placeholder with vault-captured values from running scripts/circumnutation/capture_temporal_cwt_canary.py` comment at the assertion site so a RED-phase failure self-documents the next step (per TDD round-2 R2-I2).

  **Canary purpose (not an oracle, per design.md D5).** The 3 hardcoded complex values are a REGRESSION DETECTOR for future pywt / numpy / BLAS drift, NOT a correctness oracle. They are captured from the GREEN-phase implementation via `c:\vaults\sleap-roots\circumnutation\scripts\capture_temporal_cwt_canary.py` and locked in; bit-identical reproduction across CI matrix (Ubuntu / Windows / macOS) proves determinism is preserved. RED-phase ships with `np.full(3, np.nan, dtype=np.complex128)` placeholder + `# TODO: replace via vault capture script on GREEN-phase` comment per design.md D5 + TDD round-3 reviewer R3-N1 (NOT `pytest.skip`, which would silently pass CI).

  **Test docstring** (per architecture round-3 R3-I1 + scientific-rigor round-2 R2-N1) MUST echo the capture script's provenance header (date, OS/BLAS/pywt/numpy versions, full `ConstantsT()` snapshot, run parameters, git SHA of `synthetic.py`).

### 2.C — Parameter recovery via independent analytical + synthetic oracles

- [ ] 2.C.1 Test `analytical_recovery_at_n_frames_1024` (parametrize over `T ∈ {1500, 3333, 7200}` s): construct `t = np.arange(1024) * 300.0`, `x = np.sin(2 * np.pi * t / T) * 10.0`; call `result = compute_scaleogram(x, 300.0); ridge = extract_ridge(result)`; **assert `(~ridge.in_coi).sum() > 0`** (precondition guard per /openspec-review round-1 TDD-I5 — prevents empty-slice RuntimeWarning if the COI saturates the ridge); assert `abs(np.median(ridge.periods_s[~ridge.in_coi]) - T) / T < 0.05` (±5% tolerance per round-3 R3-B2 pin at n_frames=1024).

- [ ] 2.C.2 Test `synthetic_recovery_at_n_frames_575` (parametrize over `T ∈ {1500, 3333, 7200}` s): `x = synthetic.generate_trajectory(T_nutation_s=T, n_frames=575, cadence_s=300, noise_sigma_px=0)["tip_x"].to_numpy()`; same CWT+ridge; **assert `(~ridge.in_coi).sum() > 0`** (precondition guard, per /openspec-review round-1 TDD-I5); assert `abs(np.median(ridge.periods_s[~ridge.in_coi]) - T) / T < 0.10` (±10% tolerance — absorbs the n=575 scale-grid discreteness).

### 2.D — COI mask correctness

- [ ] 2.D.1 Test `coi_mask_cell_by_cell_via_shared_helper` (atol=0 round-trip via `_coi_boundary_samples`): construct `x = np.linspace(0, 100, 512)`; `result = compute_scaleogram(x, 1.0)` (cadence=1 simplifies the math); for each scale `s` in `result.scales`: compute `boundary = _coi_boundary_samples(s, COI_EFOLDING_FACTOR)` (same helper the impl uses); assert `result.coi_mask[i_scale, :min(boundary, 512)].all()` AND `result.coi_mask[i_scale, max(0, 512-boundary):].all()` AND `~result.coi_mask[i_scale, boundary:512-boundary].any()` (when `2*boundary < 512`). The test reuses the SAME helper, so floating-point rounding is identical.

- [ ] 2.D.2 Test `coi_fraction_increases_with_scale`: `coi_fractions = result.coi_mask.mean(axis=1)`; assert `(np.diff(coi_fractions) >= -1e-12).all()` (monotonically non-decreasing — coarse periods have wider COI).

- [ ] 2.D.3 Test `coi_mask_small_at_smallest_scale_when_signal_long_enough`: with `n_frames=575` and `cadence_s=300`, the smallest scale (`period ≈ 600 s` → scale ≈ 2 samples) has `coi_half_width ≈ √1.5 · 2 ≈ 2.45`. Per /openspec-review round-1 TDD-I4, derive the expected mask coverage via the shared helper (`expected_boundary = _coi_boundary_samples(float(result.scales[0]), COI_EFOLDING_FACTOR)`) and assert `result.coi_mask[0, :].mean() == (2 * expected_boundary) / 575` (when `2 * expected_boundary < 575`) AND `result.coi_mask[0, :].mean() < 0.05`. This avoids the brittle hardcoded `6/575` magic ratio.

- [ ] 2.D.4 Test `coi_mask_saturates_at_largest_scale`: with `n_frames=575` and `period_max=43125 s` → scale ≈ 144; `coi_half_width ≈ √1.5 · 144 ≈ 176` → boundary=176; total COI = `2 · 176 = 352` of 575 → `coi_mask[-1, :].mean() > 0.5` (largest scale is mostly COI as expected).

### 2.E — Ridge sanity

- [ ] 2.E.1 Test `ridge_concentrated_for_single_frequency_input`: `x = np.sin(2*np.pi*np.arange(1024)*300/3333) * 10`; `result = compute_scaleogram(x, 300.0)`; `ridge = extract_ridge(result)`; `interior_mask = ~ridge.in_coi`; assert `interior_mask.sum() > 0`; `ridge_scale_idx = np.argmax(np.abs(result.scaleogram), axis=0)`; assert mode-fraction `np.bincount(ridge_scale_idx[interior_mask]).max() / interior_mask.sum() >= 0.85` (most COI-interior frames pick the same scale; derivation in design.md D9 §2.E.1). Per /openspec-review round-1 TDD-I8, the test docstring MUST record the empirically-measured mode-fraction value captured at GREEN-phase (placeholder until then) so a future drift below 0.85 produces an informative failure message of the form "mode-fraction dropped from 0.97 [recorded GREEN-phase] to 0.83 < 0.85" rather than the bare "0.83 < 0.85".

- [ ] 2.E.2 Test `ridge_dispersed_for_pure_noise_input` (per round-3 R3-B1, replaces chi-square uniformity test): `x = np.random.default_rng(0).standard_normal(1024)`; `result = compute_scaleogram(x, 300.0); ridge = extract_ridge(result)`; `interior_mask = ~ridge.in_coi`; compute `ridge_scale_idx` (same as 2.E.1); assert `np.bincount(ridge_scale_idx[interior_mask], minlength=CWT_SCALE_COUNT_DEFAULT).max() / interior_mask.sum() < 0.5` (no single scale captures more than half of COI-interior frames — non-degeneracy dispersion check; seed-stream-independent across reasonable numpy upgrades).

- [ ] 2.E.3 Test `in_coi_field_correctly_flagged_at_edges`: `result = compute_scaleogram(np.linspace(0, 100, 128), 300.0); ridge = extract_ridge(result)`; assert `ridge.in_coi[0:3].all()` (first 3 frames are in COI for any scale's COI band ≥ 1) AND `ridge.in_coi[-3:].all()` (last 3 frames similarly).

- [ ] 2.E.4 Test `in_coi_consistent_with_scaleogram_coi_mask`: for the ridge produced by `extract_ridge(result)`, verify `ridge.in_coi[i] == result.coi_mask[ridge_scale_idx[i], i]` for all `i`.

### 2.F — Validation/errors (30 parametrized ids)

- [ ] 2.F.1 `compute_scaleogram_x_invalid` parametrized over 9 ids:
  - `x` contains NaN → ValueError matching `"x"`
  - `x` contains +inf → ValueError matching `"x"`
  - `x` contains -inf → ValueError matching `"x"`
  - `x.dtype == np.complex128` → ValueError matching `"x"` or `"dtype"`
  - `x.ndim == 2` (e.g., `np.zeros((10, 10))`) → ValueError matching `"x"` or `"shape"`
  - `x.dtype == object` (Python objects array) → ValueError matching `"x"` or `"dtype"`
  - `len(x) == 0` → ValueError matching `"x"` or `"length"`
  - `len(x) == 1` → ValueError matching `"x"` or `"length"`
  - `len(x) == 8` (= MIN_FRAMES_REQUIRED - 1 at defaults) → ValueError matching `"x"` or `"length"`

- [ ] 2.F.2 `compute_scaleogram_cadence_s_invalid` parametrized over 9 ids (bumped from 8 per /openspec-review round-1 Code-I1 — explicit `np.bool_(True)` guard test):
  - `cadence_s = 0` → ValueError matching `"cadence_s"`
  - `cadence_s = -1.0` → ValueError matching `"cadence_s"`
  - `cadence_s = float("nan")` → ValueError matching `"cadence_s"`
  - `cadence_s = float("inf")` → ValueError matching `"cadence_s"`
  - `cadence_s = float("-inf")` → ValueError matching `"cadence_s"`
  - `cadence_s = True` (Python bool subtype of int) → TypeError matching `"cadence_s"`
  - `cadence_s = np.bool_(True)` (numpy bool scalar — subclass of `int` per numpy; must be explicitly guarded by the `_check_float_finite`-equivalent helper because it satisfies `isinstance(np.bool_(True), int)` on some numpy versions) → TypeError matching `"cadence_s"`
  - `cadence_s = "300"` (str) → TypeError matching `"cadence_s"`
  - `cadence_s = [300.0]` (list) → TypeError matching `"cadence_s"`

- [ ] 2.F.3 `compute_scaleogram_constants_type_invalid` parametrized over 3 ids: `constants=42`, `constants="foo"`, `constants={}` → all raise TypeError matching `"constants"`.

- [ ] 2.F.4 `compute_scaleogram_constants_field_invalid` parametrized over 2 ids (per design.md D8 positive-finite guards):
  - `constants=ConstantsT(CWT_PERIOD_MAX_SIGNAL_FRACTION=0.0)` → ValueError matching `"CWT_PERIOD_MAX_SIGNAL_FRACTION"`
  - `constants=ConstantsT(CWT_PERIOD_MIN_NYQUIST_FACTOR=-1.0)` → ValueError matching `"CWT_PERIOD_MIN_NYQUIST_FACTOR"`

- [ ] 2.F.5 `extract_ridge_input_type_invalid` parametrized over 3 ids:
  - `extract_ridge(None)` → TypeError matching `"ScaleogramResult"`
  - `extract_ridge({})` → TypeError matching `"ScaleogramResult"`
  - `extract_ridge((1, 2, 3))` → TypeError matching `"ScaleogramResult"`

- [ ] 2.F.6 `extract_ridge_empty_scaleogram` parametrized over 2 ids (per design.md D8). **Full ScaleogramResult construction must be explicit** (per /openspec-review round-1 TDD-I13) — provide a `_make_empty_scaleogram_result(empty_axis: str) -> ScaleogramResult` fixture helper:
  ```python
  def _make_empty_scaleogram_result(empty_axis: str) -> ScaleogramResult:
      if empty_axis == "scales":
          shape = (0, 10)
          scales = np.empty(0, dtype=np.float64)
          periods_s = np.empty(0, dtype=np.float64)
          frequencies_hz = np.empty(0, dtype=np.float64)
          coi_mask = np.empty((0, 10), dtype=bool)
      elif empty_axis == "frames":
          shape = (10, 0)
          scales = np.linspace(2.0, 50.0, 10, dtype=np.float64)
          periods_s = scales * 300.0
          frequencies_hz = 1.0 / periods_s
          coi_mask = np.empty((10, 0), dtype=bool)
      return ScaleogramResult(
          scaleogram=np.zeros(shape, dtype=np.complex128),
          scales=scales, periods_s=periods_s, frequencies_hz=frequencies_hz,
          coi_mask=coi_mask, cadence_s=300.0, wavelet="cmor1.5-1.0",
      )
  ```
  - `extract_ridge(_make_empty_scaleogram_result("scales"))` → ValueError matching `"n_scales"` or `"empty"`
  - `extract_ridge(_make_empty_scaleogram_result("frames"))` → ValueError matching `"n_frames"` or `"empty"`

- [ ] 2.F.7 `extract_ridge_constants_invalid` parametrized over 3 ids (per round-3 R3-I5 — symmetric with compute_scaleogram): `constants=42`, `constants="foo"`, `constants={}` → all raise TypeError matching `"constants"`.

- [ ] 2.F.8 `compute_scaleogram_x_at_min_frames_succeeds` (positive boundary test, per /openspec-review round-1 TDD-I11): `x = np.linspace(0.0, 1.0, 9, dtype=np.float64); result = compute_scaleogram(x, 300.0); assert result.scaleogram.shape == (CWT_SCALE_COUNT_DEFAULT, 9)`. Asserts that the documented `MIN_FRAMES_REQUIRED = 9` floor is achievable (the negative case `len(x) == 8` is in §2.F.1).

  **Total: 9 + 9 + 3 + 2 + 3 + 2 + 3 + 1 = 32 ids** (up from 30 with the np.bool_ + n=9-positive additions).

### 2.G — ConstantsT override + 2-tier resolution-order

- [ ] 2.G.1 Test `module_default_equals_ConstantsT_default`: call `r1 = compute_scaleogram(x, 300.0)` and `r2 = compute_scaleogram(x, 300.0, constants=ConstantsT())`; assert `np.array_equal(r1.scaleogram, r2.scaleogram)` (validates the 2-tier resolution: `constants or ConstantsT()` produces the same output as the default).

- [ ] 2.G.2 Test `override_CWT_SCALE_COUNT_DEFAULT`: `result = compute_scaleogram(x, 300.0, constants=ConstantsT(CWT_SCALE_COUNT_DEFAULT=32))`; assert `result.scaleogram.shape[0] == 32`.

- [ ] 2.G.3 Test `override_COI_EFOLDING_FACTOR_approximately_doubles_mask` (per round-3 R3-I2): `r1 = compute_scaleogram(x, 300.0, constants=ConstantsT())` (factor=√1.5); `r2 = compute_scaleogram(x, 300.0, constants=ConstantsT(COI_EFOLDING_FACTOR=2*math.sqrt(1.5)))`; assert `1.7 < r2.coi_mask.mean() / r1.coi_mask.mean() < 2.0` (not exactly 2 because the mask saturates at the highest scales where `2·half_width ≥ n_frames`; empirically 1.94).

- [ ] 2.G.4 Test `constants_version_is_4`: `assert _CONSTANTS_VERSION == 4`.

- [ ] 2.G.5 Test `default_constants_snapshot_includes_all_required_keys` (per round-2 R2-I6 + round-3 R3-N2): assert the snapshot is a set-superset of the required keys:
  ```python
  snapshot_keys = set(_default_constants_snapshot().keys())
  required_pr5 = {"COI_EFOLDING_FACTOR", "CWT_SCALE_COUNT_DEFAULT",
                  "CWT_PERIOD_MIN_NYQUIST_FACTOR", "CWT_PERIOD_MAX_SIGNAL_FRACTION"}
  required_pr4 = {"SYNTHETIC_T_NUTATION_S", "SYNTHETIC_AMPLITUDE_PX",
                  "SYNTHETIC_GROWTH_RATE_PX_PER_FRAME", "SYNTHETIC_NOISE_SIGMA_PX",
                  "SYNTHETIC_CADENCE_S", "SYNTHETIC_N_FRAMES",
                  "SYNTHETIC_GROWTH_AXIS_ANGLE_RAD"}
  required_pr3 = {"FRAC_OUTLIER_STEPS_MAX", "WORST_STEP_RATIO_MAX",
                  "SG_MSD_AGREEMENT_MAX", "D2_MSD_AGREEMENT_MAX"}
  assert snapshot_keys >= required_pr5 | required_pr4 | required_pr3
  ```

- [ ] 2.G.6 Test `min_frames_required_tracks_constants_override` (per /openspec-review round-1 TDD-I12): under a non-default `ConstantsT`, the derived MIN_FRAMES_REQUIRED must update correctly.
  ```python
  # With CWT_PERIOD_MIN_NYQUIST_FACTOR=4.0 and CWT_PERIOD_MAX_SIGNAL_FRACTION=0.25,
  # floor(4.0/0.25)+1 = 17 frames required.
  custom = ConstantsT(CWT_PERIOD_MIN_NYQUIST_FACTOR=4.0)
  with pytest.raises(ValueError, match=r"x|MIN_FRAMES|length"):
      compute_scaleogram(np.linspace(0.0, 1.0, 16), 300.0, constants=custom)
  result = compute_scaleogram(np.linspace(0.0, 1.0, 17), 300.0, constants=custom)
  assert result.scaleogram.shape[1] == 17
  ```

- [ ] 2.G.7 Test `nyquist_ratio_max_docstring_cross_references_cwt_period_max_signal_fraction` (per /openspec-review round-1 Spec-B2): grep for the cross-reference text in `_constants.py`.
  ```python
  import inspect
  from sleap_roots.circumnutation import _constants
  source = inspect.getsource(_constants)
  # NYQUIST_RATIO_MAX declaration at line X must be within ±10 lines of
  # a comment / docstring mentioning CWT_PERIOD_MAX_SIGNAL_FRACTION.
  # Implementation may use either an inline comment or a docstring attached
  # via __doc__-set; the test checks both patterns.
  nyquist_pos = source.find("NYQUIST_RATIO_MAX:")
  cwt_pos = source.find("CWT_PERIOD_MAX_SIGNAL_FRACTION")
  assert nyquist_pos > -1 and cwt_pos > -1
  # The two declarations / cross-references must appear in close proximity (within 1500 chars
  # of each other), indicating intentional cross-referencing rather than accidental colocation.
  assert abs(nyquist_pos - cwt_pos) < 1500
  # Verify the substring "CWT_PERIOD_MAX_SIGNAL_FRACTION" appears within ~10 lines of NYQUIST_RATIO_MAX (and vice versa)
  nearby_to_nyquist = source[max(0, nyquist_pos - 800):nyquist_pos + 800]
  assert "CWT_PERIOD_MAX_SIGNAL_FRACTION" in nearby_to_nyquist
  nearby_to_cwt = source[max(0, cwt_pos - 800):cwt_pos + 800]
  assert "NYQUIST_RATIO_MAX" in nearby_to_cwt
  ```

### 2.H — Reference-fixture sanity

- [ ] 2.H.1 Test `proofread_fixture_constraint_satisfaction` parametrized over the 6 tracks (`pytest.param(track_id, id=f"track_{track_id}")` for `track_id ∈ {0, 1, 2, 3, 4, 5}`) loaded from `tests/data/circumnutation_nipponbare_plate_001/plate_001_greyscale.tracked_proofread.slp` via `Series.load(...).get_tracked_tips()` filtered by `track_id`. For each track:
  - (a) Regression guard on D8 constraints: `assert np.isfinite(x).all()`, `assert len(x) >= 9`, `assert x.dtype.kind in "if"` (integer or float coercible to float64).
  - (b) `result = compute_scaleogram(x, cadence_s=300.0)` does not raise.
  - (c) `ridge = extract_ridge(result)` does not raise.
  - (d) `assert result.scaleogram.shape == (CWT_SCALE_COUNT_DEFAULT, 575)`.
  - (e) `assert result.coi_mask.shape == result.scaleogram.shape`.
  - (f) Scale range covers biologically-plausible nutation periods: `assert result.periods_s.min() < 1000.0` AND `assert result.periods_s.max() > 10000.0`.
  - (g) COI fraction at target scale (`scale_idx_at_target = int(np.argmin(np.abs(result.periods_s - 3333.0)))`): `assert result.coi_mask[scale_idx_at_target, :].mean() < 0.10` (measured 4.87% at √1.5 factor — see design.md D3 arithmetic).
  - (h) Regression-detector sanity (revised at GREEN-phase from "plausibility-band median ridge period"): assert `np.isfinite(ridge.periods_s).all()`, `np.isfinite(ridge.amplitudes).all()`, `(ridge.amplitudes >= 0).all()`, and `(ridge.amplitudes[~ridge.in_coi] > 0).all()`. **Why not the original `1000 < median < 10000 s` biological-plausibility band**: empirical observation during GREEN-phase showed that proofread `tip_x` carries substantial lateral drift (~70-170 px peak-to-peak after linear detrend, vs ~10 px expected nutation amplitude). The CWT correctly identifies this dominant low-frequency drift at the longest available scales — not a bug, just the expected CWT response to a multi-scale signal where low-frequency content dominates. Proper nutation-period recovery requires the LATERAL coordinate projection per theory.md CC-7, which is PR #6's `coordinate="lateral"` parameter; PR #5 does not own that preprocessing. The (h) softening keeps the "catch shape-correct garbage" intent (catches NaN propagation, all-zero scaleogram, negative-amplitude bugs) without making biological-correctness claims that depend on preprocessing outside PR #5's scope.

- [ ] 2.H.2 Test `synthetic_layer_1_sanity`: `df = synthetic.generate_trajectory(T_nutation_s=3333, n_frames=575, cadence_s=300, noise_sigma_px=2, random_state=0)`; `x = df["tip_x"].to_numpy()`; `result = compute_scaleogram(x, 300.0); ridge = extract_ridge(result)`; assert `abs(np.median(ridge.periods_s[~ridge.in_coi]) - 3333.0) / 3333.0 < 0.10` (±10% tolerance; NOT the Derr forensic match — that's PR #6).

### 2.I — Foundation-test migration (mirrors PR #4 split-table refactor)

- [ ] 2.I.1 Edit `tests/test_circumnutation_foundation.py` `STUB_MODULES` (lines 35–43): drop `("temporal_cwt", "compute_scaleogram", 5)` → 7 → **6** entries remaining. Verify `test_stub_module_imports_cleanly` and `test_stub_callable_raises_with_correct_pr` automatically drop the temporal_cwt parametrize id. The remaining 6 are `("psi_g", "compute_psi_g", 7)`, `("midline", "reconstruct", 8)`, `("spatial_cwt", "compute_scaleogram", 9)`, `("parametric", "compute", 11)`, `("plotting", "scaleogram", 16)`, `("pipeline", "compute_traits", 14)`.

- [ ] 2.I.2 Edit `STUBS_WITH_CONSTANTS_KWARG` (lines 821–827): drop `("temporal_cwt", "compute_scaleogram")` → 5 → **4** entries remaining. Verify `test_stub_accepts_constants_kwarg` automatically drops the temporal_cwt parametrize id.

- [ ] 2.I.3 Edit `IMPLEMENTATIONS_WITH_CONSTANTS_KWARG` (lines 833–837): add `("temporal_cwt", "compute_scaleogram")` → 3 → **4** entries. Update the comment to note PR #5's addition mirroring PR #4's pattern.

- [ ] 2.I.4 Extend `test_implementation_accepts_constants_kwarg` (line 855): add an `elif module_name == "temporal_cwt":` branch in the dispatcher. Build a minimal valid 9-element float64 array (`x = np.linspace(0, 100, 16)`); call `fn(x=x, cadence_s=300.0, constants=ConstantsT())`; assert returned object `isinstance(_, ScaleogramResult)`.

- [ ] 2.I.5 Rename `test_schema_version_is_1_and_constants_version_is_3` → `test_schema_version_is_1_and_constants_version_is_4` (line 204). Update assertion at line 211: `_constants._CONSTANTS_VERSION == 4`. Update docstring at line 205: `"_CONSTANTS_VERSION is 4 (bumped in PR #5)"`.

- [ ] 2.I.6 Append to the comment block at lines 26-34: a new line reading `"PR #5 (add-circumnutation-temporal-cwt-machinery) further reduces the stub count from 7 to 6 and the STUBS_WITH_CONSTANTS_KWARG count from 5 to 4, since temporal_cwt is now an implementation module."`

- [ ] 2.I.7 **Add `"temporal_cwt"` to the explicit module list in `test_module_logger_is_namespaced`'s parametrize** (around lines 745-749 of `tests/test_circumnutation_foundation.py`). The test parametrizes over `[name for name, _, _ in STUB_MODULES] + [<explicit list>]`; since §2.I.1 removes `temporal_cwt` from `STUB_MODULES`, the logger-namespace coverage would SILENTLY VANISH without this explicit re-addition. **This is the exact regression Copilot caught for `synthetic` in PR #4** (see test_circumnutation_foundation.py:745-747 comment). Append `"temporal_cwt"` to the explicit list with a comment line: `# Added in PR #5: temporal_cwt (now an implementation module, not a stub)`.

## 3. Implementation (GREEN phase — make red tests pass)

- [ ] 3.1 Edit `sleap_roots/circumnutation/_constants.py`:
  - Below the wavelet-basis defaults block, add 4 new module-level UPPER_SNAKE constants:
    - `COI_EFOLDING_FACTOR: float = math.sqrt(1.5)` with docstring citing `design.md` D3 derivation (`√B` envelope e-folding for cmor1.5-1.0; cross-reference to `WAVELET_DEFAULT_TEMPORAL`; "override when overriding the wavelet")
    - `CWT_SCALE_COUNT_DEFAULT: int = 64` with docstring citing Derr Sept-2025 pilot density
    - `CWT_PERIOD_MIN_NYQUIST_FACTOR: float = 2.0` with docstring citing Nyquist
    - `CWT_PERIOD_MAX_SIGNAL_FRACTION: float = 0.25` with docstring citing Torrence-Compo n/4 + cross-reference to `NYQUIST_RATIO_MAX` (numerically equal at 0.25, semantically distinct)
  - Add the reciprocal cross-reference paragraph to `NYQUIST_RATIO_MAX`'s docstring.
  - Extend `ConstantsT` with 4 new fields (defaults sourced from the module-level constants).
  - Extend `_default_constants_snapshot()` with 4 new keys.
  - Bump `_CONSTANTS_VERSION` from `3` to `4`. Update its docstring with a paragraph noting PR #5's contribution alongside PR #3 / PR #4.

- [ ] 3.2 Rewrite `sleap_roots/circumnutation/temporal_cwt.py` from scratch (replacing the 41-line stub):
  - Module docstring referencing the spec delta + `theory.md` §3.4/§6.5/§7.2/§7.6 + `design.md` D1–D9 + scope discipline (no trait emission).
  - Imports: `import logging`, `import math`, `from typing import Optional`; `import attrs`, `import numpy as np`, `import pywt`; relative imports `from ._constants import ConstantsT, COI_EFOLDING_FACTOR, CWT_SCALE_COUNT_DEFAULT, CWT_PERIOD_MIN_NYQUIST_FACTOR, CWT_PERIOD_MAX_SIGNAL_FRACTION, WAVELET_DEFAULT_TEMPORAL`.
  - Module logger: `logger = logging.getLogger(__name__)`.
  - Define `@attrs.define(frozen=True, slots=False, kw_only=True)` `ScaleogramResult` with the 7 documented fields (see design.md D1).
  - Define `@attrs.define(frozen=True, slots=False, kw_only=True)` `RidgeResult` with the 5 documented fields (see design.md D2); include `powers = amplitudes ** 2` redundancy preservation docstring.
  - Private helper `_coi_boundary_samples(scale: float, coi_factor: float) -> int`: returns `int(math.ceil(coi_factor * scale))`. Docstring notes test-importability per design.md D3.
  - Private helper `_validate_x(x) -> np.ndarray`: 1-D ndarray coercion, dtype float64 (reject complex/object), finite check, length check (uses MIN_FRAMES_REQUIRED from `_derive_min_frames_required`). Returns the coerced array.
  - Private helper `_validate_cadence_s(cadence_s) -> float`: PR #4's `_check_float_finite` semantics — accept Python `int`/`float`, accept numpy `np.integer`/`np.floating` scalars, **explicitly reject `bool` (Python) AND `np.bool_` (numpy scalar)** by an isinstance check that precedes the int/float check, reject `str`/`list`/`complex`/tuple, require positive finite (`math.isfinite(float(value))` and `float(value) > 0`). Returns coerced `float`. **The `np.bool_` guard is load-bearing** per /openspec-review round-1 Code-I1 because `np.bool_` is a numpy scalar subclass of `int` on some numpy versions, so an int-check-only path would accept it.
  - Private helper `_check_constants(constants) -> ConstantsT`: None → `ConstantsT()`; else must be `ConstantsT` instance (TypeError). **Named `_check_constants` for DRY consistency with `synthetic._check_constants`** per /openspec-review round-1 Code-I2.
  - Private helper `_derive_min_frames_required(constants: ConstantsT) -> int`: positive-finite guards on `CWT_PERIOD_MAX_SIGNAL_FRACTION` and `CWT_PERIOD_MIN_NYQUIST_FACTOR` (raise ValueError naming offending field); return `int(math.floor(NYQUIST_FACTOR / SIGNAL_FRACTION)) + 1`.
  - Private helper `_log_spaced_scales(n_frames: int, cadence_s: float, constants: ConstantsT) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]`: derives `period_min_s`, `period_max_s`, `scale_min`, `scale_max` via `pywt.scale2frequency` round-trip; returns `(scales, periods_s, frequencies_hz, wavelet_name)`.
  - Private helper `_make_coi_mask(scales: np.ndarray, n_frames: int, coi_factor: float) -> np.ndarray`: bool array as per design.md D3.
  - Public `compute_scaleogram(x, cadence_s, constants=None) -> ScaleogramResult`: validates → derives scale axis → calls `pywt.cwt(x, scales, wavelet)` → builds COI mask → emits the single `logger.debug` per design.md "Logger emissions" subsection → returns `ScaleogramResult`.
  - Public `extract_ridge(scaleogram_result, constants=None) -> RidgeResult`: validates → ridge_scale_idx = argmax(|scaleogram|, axis=0) → derives all 5 RidgeResult fields → emits the single `logger.debug` → returns `RidgeResult`.

- [ ] 3.3 Run `uv run pytest tests/test_circumnutation_temporal_cwt.py -x --no-header --tb=short` — most tests should pass except §2.B.3 (canary).

- [ ] 3.4 Run `uv run pytest tests/test_circumnutation_foundation.py -x --no-header --tb=short` — confirm all foundation tests pass with the migration changes.

### 3.5 Canary value capture (one-time)

- [ ] 3.5.1 Write `scripts/circumnutation/capture_temporal_cwt_canary.py` **in the repo** (NOT in the vault — per /openspec-review round-1 TDD-I10, the capture script must be reproducible by any contributor, CI runner, or code-review subagent, not just Elizabeth's dev machine). Add a brief module docstring referencing this tasks.md §3.5 and the design.md "Capture script header / provenance" subsection. The script's exact imports (per /openspec-review round-2 reviewer R2-I3 — make the private-API access explicit so the script doesn't silently re-derive `math.ceil(math.sqrt(1.5) * scale)` inline if `_coi_boundary_samples` evolves):
  ```python
  from sleap_roots.circumnutation import synthetic
  from sleap_roots.circumnutation.temporal_cwt import (
      compute_scaleogram,
      _coi_boundary_samples,  # private helper imported intentionally for COI-interior derivation
  )
  from sleap_roots.circumnutation._constants import (
      _default_constants_snapshot,
      COI_EFOLDING_FACTOR,
  )
  ```
  The script then:
  - Generates `df = synthetic.generate_trajectory(random_state=0, n_frames=128, T_nutation_s=3333, cadence_s=300, noise_sigma_px=0)`.
  - Calls `result = compute_scaleogram(df["tip_x"].to_numpy(), 300.0)`.
  - Computes `scale_idx_at_target = int(np.argmin(np.abs(result.periods_s - 3333.0)))`.
  - Chooses 3 COI-interior frame indices via explicit derivation: `boundary = _coi_boundary_samples(float(result.scales[scale_idx_at_target]), COI_EFOLDING_FACTOR)`; then `frame_indices = [boundary + 2, 64, 128 - boundary - 2]` (2 frames inside the COI-interior on each edge plus the geometric middle frame). The script asserts `0 < frame_indices[0] < frame_indices[1] < frame_indices[2] < 128` and `result.coi_mask[scale_idx_at_target, frame_indices].any() == False` before recording values.
  - Prints a header containing: capture date (ISO 8601); machine fingerprint (OS / BLAS impl / pywt version / numpy version via `numpy.__version__`, `pywt.__version__`, `np.__config__.show()`); full `ConstantsT()` snapshot via `_default_constants_snapshot()`; run parameters (`n_frames=128, T_nutation_s=3333, cadence_s=300, noise_sigma_px=0, random_state=0`); git commit SHA of `sleap_roots/circumnutation/synthetic.py` via `git log -1 --format=%H sleap_roots/circumnutation/synthetic.py`.
  - Prints the 3 captured complex values in a copy-paste-ready Python literal format for §2.B.3 hardcode (e.g., `expected_canary = np.array([complex(a, b), complex(c, d), complex(e, f)], dtype=np.complex128)`).
  - Supports `--out tests/test_circumnutation_temporal_cwt.canary.json` flag (optional reproducibility-helper per /openspec-review round-1 TDD-S3) writing the captured values + provenance header as JSON; default behavior prints to stdout for copy-paste.

- [ ] 3.5.2 Run the capture script: `uv run python scripts/circumnutation/capture_temporal_cwt_canary.py`. Copy the 3 complex values into `tests/test_circumnutation_temporal_cwt.py` §2.B.3 (replacing the `np.full(3, np.nan, dtype=np.complex128)` placeholder). Copy the provenance header into the test docstring.

- [ ] 3.5.3 Re-run `uv run pytest tests/test_circumnutation_temporal_cwt.py::test_cross_os_canary_at_atol_1e_9 -x` — expect green.

## 4. Documentation

- [ ] 4.1 Update `docs/changelog.md` under `## [Unreleased]` / `### Added`:
  ```markdown
  - **circumnutation**: temporal CWT machinery (`sleap_roots.circumnutation.temporal_cwt.compute_scaleogram`, `extract_ridge`, `ScaleogramResult`, `RidgeResult`) using the `cmor1.5-1.0` mother wavelet; log-spaced 64-scale default with wavelet-aware `√B·scale ≈ 1.225·scale` COI mask (first concrete realization of the QC tier's `coi_fraction_t1`); deterministic across OSs at `atol=1e-9` (CC-6); no trait emission (PR #6 will compose on top). 4 new defaults in `_constants.py` + `ConstantsT`; `_CONSTANTS_VERSION` 3 → 4. See `openspec/changes/add-circumnutation-temporal-cwt-machinery/`.
  ```

- [ ] 4.2 No `docs/circumnutation/theory.md` or `docs/circumnutation/roadmap.md` edits inside this PR. Roadmap row #5 status `⬜ → ✅` happens post-merge during `cleanup-merged` per §7.

## 5. Verification (gates before opening PR)

- [ ] 5.1 Full test suite: `uv run pytest tests/ -x` — all tests pass.

- [ ] 5.2 Module coverage: `uv run pytest tests/test_circumnutation_temporal_cwt.py --cov=sleap_roots --cov-report=term-missing` — 100% coverage on `sleap_roots/circumnutation/temporal_cwt.py` (use `--cov=sleap_roots` whole-package to avoid the numpy 2.x + scipy + coverage interaction noted in the user prompt's hard constraints).

- [ ] 5.3 Project-wide coverage: `uv run pytest tests/ --cov=sleap_roots --cov-fail-under=84` — coverage holds at 84%+.

- [ ] 5.4 Black format check: `uv run black --check sleap_roots tests` — no formatting issues.

- [ ] 5.5 Pydocstyle: `uv run pydocstyle --convention=google sleap_roots/` — no docstring violations.

- [ ] 5.6 uv lock check: `uv lock --check` — lockfile in sync (no new deps; expect clean).

- [ ] 5.7 mkdocs build: `uv run mkdocs build` — confirm the auto-generated reference pages for `temporal_cwt` (compute_scaleogram, extract_ridge, ScaleogramResult, RidgeResult) render without errors.

- [ ] 5.8 OpenSpec strict validation: `openspec validate add-circumnutation-temporal-cwt-machinery --strict` — no validation errors.

- [ ] 5.9 Pre-merge skill: `/pre-merge` — runs the full gate checklist + smoke-tests.

## 6. PR open

- [ ] 6.1 Stage files: `git add sleap_roots/circumnutation/_constants.py sleap_roots/circumnutation/temporal_cwt.py tests/test_circumnutation_foundation.py tests/test_circumnutation_temporal_cwt.py docs/changelog.md openspec/changes/add-circumnutation-temporal-cwt-machinery/`

- [ ] 6.2 Commit message (Conventional Commits style): `feat(circumnutation): temporal CWT machinery for Layer-1 + Tier 1 composition (PR #5)`. Body cites the OpenSpec change-id, the design.md sections, the 3-round critical-review reconciliation, the Layer-1 sanity status (synthetic + proofread fixture), and the COI factor empirical verification.

- [ ] 6.3 Draft GitHub sub-issue body at `c:\vaults\sleap-roots\circumnutation\github_issues\issue_add-circumnutation-temporal-cwt-machinery.md` referencing GitHub epic #197 + roadmap row #5 + theory.md anchors + the new follow-up issue body for ridge-tracking continuity (PR #6).

- [ ] 6.4 Show Elizabeth both drafts (issue body + PR body) BEFORE posting; await her authorization to post via `gh issue create` and `gh pr create` (per repo hard constraint: drafts go to vault first).

- [ ] 6.5 After Elizabeth's go-ahead: `gh issue create` (using the vault draft) → record the issue number. Then `gh pr create --title "feat(circumnutation): temporal CWT machinery (#<issue>)"` with body from vault.

- [ ] 6.6 File the NEW follow-up issue "circumnutation: per-frame argmax ridge can hop discontinuously between scales — add ridge-tracking continuity post-filter in PR #6" per design.md Follow-up Issues. Draft body at `c:\vaults\sleap-roots\circumnutation\github_issues\issue_ridge_continuity_post_filter.md`.

## 6.5. Copilot review reconciliation

- [ ] 6.5.1 After PR opens, GitHub Copilot will auto-review. Invoke `/copilot-review` to view inline comments via the GraphQL query.
- [ ] 6.5.2 Verify each Copilot finding empirically (per `superpowers:receiving-code-review`). Distinguish legitimate bugs from style noise.
- [ ] 6.5.3 Land Copilot fixes in a reconciliation commit: `fix(circumnutation): address GitHub Copilot review on PR #<n>` with body summarizing each finding + resolution.

## 7. Self-review (post-Copilot, before merge)

- [ ] 7.1 Invoke `/review-pr` — launches the 5-subagent post-PR review (code quality / security / TDD / docs / architecture).
- [ ] 7.2 Address BLOCKING + IMPORTANT findings inline; defer the rest with rationale.
- [ ] 7.3 Post the review verdict as a PR comment via `gh pr review --comment` with the `> **Verdict: APPROVE** (posted as comment — cannot approve your own PR)` banner.

## 8. Post-merge cleanup

- [ ] 8.1 Invoke `/cleanup-merged` — handles `git fetch origin --prune`, branch cleanup, `/openspec:archive` invocation, and the verify checklist.

- [ ] 8.2 `/openspec:archive` (invoked from `/cleanup-merged`) runs `openspec archive add-circumnutation-temporal-cwt-machinery --yes` to fold the requirements into `openspec/specs/circumnutation/spec.md` and move the change folder to `openspec/changes/archive/YYYY-MM-DD-add-circumnutation-temporal-cwt-machinery/`.

- [ ] 8.3 Update `docs/circumnutation/roadmap.md` row PR #5: `⬜ → ✅` with issue + PR cross-links. Mirror the format used for PR #1 / #2 / #3 / #4.

- [ ] 8.4 Update `docs/changelog.md`: confirm the `[Unreleased] / ### Added` entry from §4.1 is present and accurate.

- [ ] 8.5 Update the Notion task page (https://www.notion.so/Circumnutation-work-sleap-roots-3494a67a7667818db2aeee80d76efdd7) with a new `Status update — YYYY-MM-DD` section. Draft to vault first at `c:\vaults\sleap-roots\circumnutation\notion_status_update_YYYY-MM-DD.md`; show Elizabeth before posting via the Notion MCP.
