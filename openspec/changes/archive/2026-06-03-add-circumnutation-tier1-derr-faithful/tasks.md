# Tasks for add-circumnutation-tier1-derr-faithful

TDD-ordered. Tests precede implementation per `superpowers:test-driven-development`. **Do not push commits from section 2 (red phase — tests only) without section 3 (implementation) in the same push** — the new test file imports `nutation.compute`, the new `_geometry.project_to_growth_axis_perpendicular`, the new `_noise.compute_sg_detrended` / `compute_fourier_noise_floor`, and the new `temporal_cwt.smooth_ridge` which section 3 implements; the foundation test migration (§2.I) adds `nutation` to `IMPLEMENTATIONS_WITH_CONSTANTS_KWARG`. The suite is expected to be red between §2.x and §3.x; PR is one logical unit.

## 1. Pre-flight (no code changes; confirms ground-truth)

- [ ] 1.1 Confirm the existing fixture used by §2.H is present: `tests/data/circumnutation_nipponbare_plate_001/plate_001_greyscale.tracked_proofread.slp`. Run `uv run python -c "from pathlib import Path; from sleap_roots.series import Series; s = Series.load(series_name='plate_001', primary_path=str(Path('tests/data/circumnutation_nipponbare_plate_001/plate_001_greyscale.tracked_proofread.slp'))); df = s.get_tracked_tips(); print(len(df), df['tip_x'].isna().sum())"` → expect **3450 rows** (575 × 6 tracks) and 0 NaN. (PR #5 verified this pre-design; reconfirm pre-RED.)
- [ ] 1.2 Run baseline: `uv run pytest tests/ -x` — confirm green BEFORE any changes. Foundation + Tier 0 + QC + synthetic + temporal_cwt tests must be passing at HEAD.
- [ ] 1.3 Run baseline coverage: `uv run pytest tests/ --cov=sleap_roots --cov-report=term --cov-fail-under=84` — confirm starting at 84%+ baseline (per CC-6 gate).
- [ ] 1.4 Verify `len(_default_constants_snapshot())` == 29 baseline (PR #5 left this at 29). After PR #6 we expect 35 (= 29 + 6 new constants).

## 2. Tests (write before implementation — TDD red phase)

- [ ] 2.1 Create `tests/test_circumnutation_nutation.py` with module docstring referencing the spec delta + `theory.md` §3.5/§6.5/§7.2/§7.6 + `roadmap.md` CC-7/CC-8 + `design.md` sections D1–D9 + the Reconciliation Appendices (2 rounds of critical review).

  **Test-file imports** (per design.md "Test-file imports" section, S2'' round-2 + S1 round-1 incorporations):
  - `import logging`
  - `import math`
  - `from pathlib import Path`
  - `import attrs`
  - `import numpy as np`
  - `import numpy.testing as npt`
  - `import pandas as pd`
  - `import pytest`
  - `import scipy.fft`
  - `import scipy.ndimage`
  - `import scipy.signal`
  - `import scipy.stats`
  - `from sleap_roots.circumnutation import _constants, _geometry, _noise, _types, nutation, synthetic, temporal_cwt`
  - `from sleap_roots.circumnutation._constants import (BAND_POWER_BAND_HIGH_FACTOR, BAND_POWER_BAND_LOW_FACTOR, BAND_POWER_NOISE_RATIO, DERR_EXPECTED_PERIOD_S, NOISE_FLOOR_OUT_OF_BAND_FACTOR, NYQUIST_RATIO_MAX, RIDGE_CONTINUITY_FILTER_WINDOW, SG_DEGREE, SG_WINDOW_DETREND, TEMPORAL_NYQUIST_RATIO_MAX, ConstantsT, _CONSTANTS_VERSION, _default_constants_snapshot)`
  - `from sleap_roots.series import Series`
  - Module-level test-only literal: `_DERR_MATCH_TOLERANCE_FOR_TEST: float = 0.05` (the per-track tolerance per S5 round-1; S5 round-2 + Sci-B3 added a separate median-±2% assertion — see §2.H.3)

### 2.A — Schema/structural tests

- [ ] 2.A.1 Test `nutation_compute_returns_dataframe`: `df = nutation.compute(_minimal_trajectory_df(), cadence_s=300.0); assert isinstance(df, pd.DataFrame)`.

- [ ] 2.A.2 Test `nutation_compute_returns_8_trait_columns_in_declared_order`: assert `list(df.columns[8:]) == ["T_nutation_median", "T_nutation_iqr", "A_nutation_envelope_max_px", "band_power_ratio", "noise_floor_estimate", "is_nutating", "period_residual_vs_derr_reference", "cadence_nyquist_ratio"]`.

- [ ] 2.A.3 Test `nutation_compute_includes_identity_columns_in_declared_order`: assert `list(df.columns[:8]) == list(ROW_IDENTITY_COLUMNS)`.

- [ ] 2.A.4 Test `trait_dtypes_correct` (parametrize over 8 trait columns): the 7 float traits are `float64`; `is_nutating` is `bool`. Single parametrize per trait → 8 ids.

- [ ] 2.A.5 Test `row_identity_5_tuple_uniqueness`: assert `df[list(_IDENTITY_5_TUPLE)].duplicated().sum() == 0`.

- [ ] 2.A.6 Test `caplog_debug_message_token_containment`: with `caplog.set_level(logging.DEBUG)`, call `nutation.compute(_minimal_trajectory_df(), cadence_s=300.0)`; assert at least one DEBUG record starts with `"nutation.compute("` AND contains tokens `n_tracks=`, `coordinate=`, `cadence_s=`.

- [ ] 2.A.7 Test `caplog_no_warning_or_error_on_happy_path`: with `caplog.set_level(logging.WARNING)`, call `nutation.compute(...)`; assert no WARNING/ERROR records.

### 2.B — Determinism + canary (CC-6 + S6 round-2)

- [ ] 2.B.1 Test `same_input_bit_identical_in_process`: call `nutation.compute(...)` twice on same input; assert per-column equality at `atol=0` for the 7 float trait columns AND `is_nutating` boolean equality.

- [ ] 2.B.2 **Canary test** `cross_os_canary_at_atol_1e_6` (S6 round-2): generate `df_synth = synthetic.generate_trajectory(random_state=0, n_frames=575, T_nutation_s=3333, cadence_s=300, noise_sigma_px=0.5)`; convert to `trajectory_df` schema; call `result_df = nutation.compute(trajectory_df, cadence_s=300.0)`; assert `result_df` columns `T_nutation_median`, `band_power_ratio`, `noise_floor_estimate` match the hardcoded 3-value tuple at `atol=1e-6, rtol=0, equal_nan=False`.

  **Canary purpose (S6 round-2 documented rationale):** loosened from PR #5's 1e-9 to 1e-6 because PR #6 adds 4 unverified scipy paths on top of PR #5's verified pywt path: `scipy.fft.rfft` (used in noise_floor + band_power), `scipy.ndimage.median_filter` (smooth_ridge), `scipy.signal.savgol_filter` (compute_sg_detrended), `scipy.stats.iqr` (T_nutation_iqr). 1e-6 cushion is scientifically irrelevant for these traits per CC-6's "either 1e-9 OR documented looser". RED-phase ships with `np.full(3, np.nan)` placeholder + `# TODO: replace via vault capture script on GREEN-phase` comment; `equal_nan=False` is load-bearing to guarantee RED-phase failure.

  **Test docstring** MUST echo the capture script's provenance header (date, OS/BLAS/pywt/scipy/numpy versions, full `ConstantsT()` snapshot, run parameters, git SHA of `nutation.py`).

### 2.C — Parameter recovery via independent analytical + synthetic oracles

- [ ] 2.C.1 Test `analytical_recovery_at_n_frames_1024` (parametrize `T ∈ {2000, 3333, 4500}` s; T set per Sci-B1 round-2): construct `t = np.arange(1024) * 300.0`, `x = np.sin(2 * np.pi * t / T) * 10.0`; build minimal `trajectory_df` with `tip_x = x` and `tip_y = np.zeros(1024)`; `result = nutation.compute(trajectory_df, cadence_s=300.0, coordinate="x")` (coordinate="x" to bypass lateral projection on this synthetic 1-axis signal); assert `(result.T_nutation_median.iloc[0] - T) / T < 0.05` (±5%); assert `result.is_nutating.iloc[0] == True`. 3 periods × 2 assertions = 6 ids.

  **Rationale (TDD-B2 round-1 + Sci-B1 round-2)**: T=1000 (original draft) is too close to CWT period_min floor of 600 s and would fail recovery the way PR #5 R3-B2 did. T=6666 (round-1 fix) sits at 96% of the SG-detrend window (6900 s) and would have its amplitude partially suppressed, flipping `is_nutating` to False. T=4500 (round-2 fix) sits at 65% of the SG-detrend window, comfortably away from both edges.

- [ ] 2.C.2 Test `synthetic_recovery_at_n_frames_575` (parametrize `T ∈ {2000, 3333, 4500}` s): `df_synth = synthetic.generate_trajectory(T_nutation_s=T, n_frames=575, cadence_s=300, noise_sigma_px=0)`; convert to `trajectory_df`; call `nutation.compute(trajectory_df, cadence_s=300.0)`; assert `(result.T_nutation_median.iloc[0] - T) / T < 0.10` (±10% tolerance — absorbs n=575 scale-grid discreteness). 3 ids.

  **TDD-I1 round-2 note:** synthetic generator produces a pure sinusoid with no growth-axis drift, so SG-detrend on this signal removes a fraction of the signal amplitude. If GREEN-phase observes T recovery failing, switch to either (a) `constants=ConstantsT(SG_WINDOW_DETREND=1)` to bypass detrending on the synth oracle, or (b) larger `noise_sigma_px` so the residual carries the period. Record in GREEN-phase Reconciliation Appendix.

- [ ] 2.C.3 Test `noise_only_input_gates_is_nutating_to_false` (TDD-B1 round-1): `df_synth = synthetic.generate_trajectory(amplitude_px=0.0, noise_sigma_px=1.0, n_frames=1024, cadence_s=300, random_state=0)`; convert to `trajectory_df`; call `nutation.compute(trajectory_df, cadence_s=300.0)`; assert `result.is_nutating.iloc[0] == False`; assert `np.isnan(result.T_nutation_median.iloc[0])` AND `np.isnan(result.T_nutation_iqr.iloc[0])` AND `np.isnan(result.A_nutation_envelope_max_px.iloc[0])` (the 3 NaN-gated traits per S4 round-1). 2 ids.

  **Rationale (TDD-B1 round-1):** original draft used `T_nutation_s=None`, but synthetic.py:415-416 resolves `None` to default 3333.0 s → produces a sinusoid, NOT noise-only. Correct noise-only path is `amplitude_px=0.0` (synthetic.py:329 docstring confirms 0.0 is valid).

### 2.D — `smooth_ridge` field-pass-through + #214 acceptance

- [ ] 2.D.1 Test `smooth_ridge_periods_only` parametrized over 4 fields: construct a hand-crafted `RidgeResult` with known scale-hopping in `periods_s`; call `smoothed = smooth_ridge(raw_ridge)`; assert `np.array_equal(smoothed.amplitudes, raw_ridge.amplitudes)` (carried through unchanged) AND `np.array_equal(smoothed.powers, raw_ridge.powers)` AND `np.array_equal(smoothed.in_coi, raw_ridge.in_coi)` AND `np.array_equal(smoothed.frame_indices, raw_ridge.frame_indices)` AND NOT `np.array_equal(smoothed.periods_s, raw_ridge.periods_s)` (smoothed != raw). 5 ids.

- [ ] 2.D.2 Test `smooth_ridge_window_default_from_constants`: call `smooth_ridge(raw_ridge)` with no window → use `RIDGE_CONTINUITY_FILTER_WINDOW = 5`. Verify by checking smoothed_periods equals `scipy.ndimage.median_filter(raw_ridge.periods_s, size=5, mode='nearest')`. 1 id.

- [ ] 2.D.3 Test `smooth_ridge_window_override`: call `smooth_ridge(raw_ridge, window=11)`; verify smoothed_periods equals `scipy.ndimage.median_filter(raw_ridge.periods_s, size=11, mode='nearest')`. 1 id.

- [ ] 2.D.4 Test `smooth_ridge_constants_override`: call `smooth_ridge(raw_ridge, constants=ConstantsT(RIDGE_CONTINUITY_FILTER_WINDOW=11))`; verify same smoothed_periods as window=11 explicit. 1 id.

- [ ] 2.D.5 Test `issue_214_acceptance_per_track` (cross-references §2.H.4): on each of the 6 Nipponbare proofread tracks, compute `raw_ridge` and `smoothed_ridge` separately, assert `scipy.stats.iqr(smoothed_ridge.periods_s[~smoothed_ridge.in_coi]) < scipy.stats.iqr(raw_ridge.periods_s[~raw_ridge.in_coi])` for ≥5 of 6 tracks. 1 id (aggregated; per-track values logged via informative-failure message).

### 2.E — `band_power_ratio` + `is_nutating` sanity + NaN-gating + factor sensitivity

- [ ] 2.E.1 Test `pure_noise_input_gates_is_nutating_to_false` (cross-references §2.C.3): same setup; assert `result.band_power_ratio.iloc[0] < BAND_POWER_NOISE_RATIO * result.noise_floor_estimate.iloc[0]` (i.e., the gate triggers). 1 id.

- [ ] 2.E.2 Test `pure_sinusoid_input_gates_is_nutating_to_true`: pure sinusoid input → assert `result.is_nutating.iloc[0] == True`. 1 id.

- [ ] 2.E.3 Test `noise_floor_estimate_finite_and_non_negative`: assert `np.isfinite(result.noise_floor_estimate.iloc[0])` AND `result.noise_floor_estimate.iloc[0] >= 0`. 1 id.

- [ ] 2.E.4 Test `band_power_ratio_finite_and_in_unit_interval`: assert `np.isfinite(result.band_power_ratio.iloc[0])` AND `0 <= result.band_power_ratio.iloc[0] <= 1`. 1 id.

- [ ] 2.E.5 Test `nan_gating_when_is_nutating_false` (parametrize over the 3 NaN-gated traits per S4 round-1 + TDD-B1 round-2): on the noise-only input, assert each of `T_nutation_median`, `T_nutation_iqr`, `A_nutation_envelope_max_px` is NaN. 3 ids.

- [ ] 2.E.6 Test `always_populated_when_is_nutating_false` (5 traits per S4 round-1): assert `is_nutating`, `band_power_ratio`, `noise_floor_estimate`, `period_residual_vs_derr_reference`, `cadence_nyquist_ratio` are all finite (NOT NaN) on the noise-only input. 1 id with 5-column finite-value check.

- [ ] 2.E.7 Test `noise_floor_factor_sensitivity` (S7 deferred from round-1; TDD-I2 round-2): on Nipponbare plate-001 track_0 lateral signal, parametrize `factor ∈ {3, 5, 7}` via `ConstantsT(NOISE_FLOOR_OUT_OF_BAND_FACTOR=factor)`; assert at least one factor produces `is_nutating == True` (the GREEN-phase decision picks the most-robust value; recorded in GREEN-phase Reconciliation Appendix). 3 ids.

### 2.F — Validation/errors (≥30 ids per TDD-I1 round-2)

- [ ] 2.F.1 `nutation_compute_trajectory_df_invalid` (delegates to `_validate_trajectory_df`; parametrize over 6 representative failure modes): `df_not_dataframe={}`, `df_missing_identity_column`, `df_missing_tip_x`, `df_dtype_mismatch_on_tip_x`, `df_empty`, `df_non_finite_tip_x` → all raise ValueError or TypeError naming the field. 6 ids.

- [ ] 2.F.2 `nutation_compute_cadence_s_invalid` parametrized over 9 ids (mirroring PR #5 §2.F.2 + S8' round-2 explicit cadence_s parameter): `cadence_s = 0`, `-1.0`, `float("nan")`, `float("inf")`, `float("-inf")`, `True` (Python bool), `np.bool_(True)`, `"300"` (str), `[300.0]` (list) → all raise ValueError or TypeError naming `cadence_s`. 9 ids.

- [ ] 2.F.3 `nutation_compute_coordinate_invalid` (parametrize 5 ids): `""`, `"X"`, `1`, `None`, `"longitudinal"` → all raise ValueError matching `"coordinate"`. 5 ids.

- [ ] 2.F.4 `nutation_compute_constants_type_invalid` (parametrize 3 ids): `42`, `"foo"`, `{}` → all raise TypeError matching `"constants"`. 3 ids.

- [ ] 2.F.5 `geometry_project_to_growth_axis_perpendicular_validation` parametrize 4 ids: `len(x) != len(y)` → ValueError; `x contains NaN` → ValueError; `y contains NaN` → ValueError; `zero net displacement (x[0]==x[-1] AND y[0]==y[-1] with constant intermediate)` → returns `np.full(n, np.nan)` (NOT raise per Architecture-I3 round-1 graceful-NaN policy). 4 ids.

- [ ] 2.F.6 `noise_compute_fourier_noise_floor_validation` parametrize 2 ids: `len(x) < 2` → NaN; `factor / t_nut > nyquist_freq` (empty out-of-band) → NaN. 2 ids.

- [ ] 2.F.7 `nutation_compute_stationary_track_fallback` (TDD-B2 round-2 closed-loop trajectory): construct trajectory_df with a closed-loop track (`x[-1] == x[0] AND y[-1] == y[0]` with varying intermediate frames forming a circle); call `nutation.compute(...)`; assert `result.is_nutating.iloc[0] == False`; assert 3 NaN-gated traits are NaN; assert no `np.RuntimeWarning("All-NaN slice encountered")` per Architecture-I3 round-2 (verify by `caplog` + `warnings.catch_warnings`). 1 id.

- [ ] 2.F.8 `smooth_ridge_input_type_invalid` parametrize 3 ids: `smooth_ridge(None)`, `smooth_ridge({})`, `smooth_ridge((1,2,3))` → all raise TypeError matching `"RidgeResult"`. 3 ids.

- [ ] 2.F.9 `smooth_ridge_window_invalid` parametrize 3 ids: `window=0` → ValueError; `window=-1` → ValueError; `window=4` (even) → ValueError. 3 ids.

  **Total §2.F**: 6+9+5+3+4+2+1+3+3 = **36 ids** (≥30 per TDD-I1 round-2 + S8' round-2 explicit cadence_s coverage).

### 2.G — `ConstantsT` override + 2-tier resolution + `coordinate=` parameter sensitivity

- [ ] 2.G.1 Test `module_default_equals_ConstantsT_default`: `r1 = nutation.compute(df, cadence_s=300.0)`; `r2 = nutation.compute(df, cadence_s=300.0, constants=ConstantsT())`; assert per-column equality (validates 2-tier resolution: `constants or ConstantsT()` produces same output as default).

- [ ] 2.G.2 Test `override_RIDGE_CONTINUITY_FILTER_WINDOW` parametrize 2 ids (`window=1` → no smoothing; `window=11` → heavy smoothing); assert `T_nutation_iqr` decreases monotonically from window=1 to window=11.

- [ ] 2.G.3 Test `override_NOISE_FLOOR_OUT_OF_BAND_FACTOR` (1 id): with `ConstantsT(NOISE_FLOOR_OUT_OF_BAND_FACTOR=10.0)` on a sinusoid input, verify `noise_floor_estimate` is computed over a tighter high-frequency band.

- [ ] 2.G.4 Test `override_BAND_POWER_BAND_LOW_HIGH_FACTOR` parametrize 2 ids: override LOW=0.25 / HIGH=4.0 → verify `band_power_ratio` changes (band is wider).

- [ ] 2.G.5 Test `override_DERR_EXPECTED_PERIOD_S` (1 id): `ConstantsT(DERR_EXPECTED_PERIOD_S=7200.0)` → verify `period_residual_vs_derr_reference` recomputes against 7200 not 3333.

- [ ] 2.G.6 Test `override_TEMPORAL_NYQUIST_RATIO_MAX_in_snapshot` (S2'' round-2; 1 id): `ConstantsT(TEMPORAL_NYQUIST_RATIO_MAX=0.5)` → `_default_constants_snapshot(custom)["TEMPORAL_NYQUIST_RATIO_MAX"] == 0.5`.

- [ ] 2.G.7 Test `constants_version_is_5`: `assert _CONSTANTS_VERSION == 5`. 1 id.

- [ ] 2.G.8 Test `default_constants_snapshot_includes_all_pr6_keys`: assert `set(_default_constants_snapshot().keys()) >= EXPECTED_PR6_CONSTANTS` where `EXPECTED_PR6_CONSTANTS = {*EXPECTED_PR5_CONSTANTS, "RIDGE_CONTINUITY_FILTER_WINDOW", "NOISE_FLOOR_OUT_OF_BAND_FACTOR", "BAND_POWER_BAND_LOW_FACTOR", "BAND_POWER_BAND_HIGH_FACTOR", "DERR_EXPECTED_PERIOD_S", "TEMPORAL_NYQUIST_RATIO_MAX"}` (6 PR #6 entries). Also assert `len(_default_constants_snapshot()) == 35` exactly (Architecture-B2 round-2 hard-anchor). 1 id with 2 assertions.

- [ ] 2.G.9 Test `coordinate_default_is_lateral`: `r1 = nutation.compute(df, cadence_s=300.0)`; `r2 = nutation.compute(df, cadence_s=300.0, coordinate="lateral")`; assert per-column equality. 1 id.

- [ ] 2.G.10 Test `coordinate_x_differs_from_lateral_on_diagonal_track` (TDD-I3 round-2; 1 id): construct synthetic with `growth_axis_angle_rad=math.pi/4` (diagonal growth axis); call `nutation.compute(df, cadence_s=300.0, coordinate="lateral")` and `coordinate="x"`; assert `T_nutation_median` differs by ≥1% between the two (lateral projection extracts the perpendicular component which is different from raw x).

- [ ] 2.G.11 Test `nyquist_ratio_max_temporal_nyquist_ratio_max_reciprocal_cross_reference_in_docstrings` (S2'' round-2): grep `_constants.py` source for the reciprocal docstring cross-reference. 1 id.

### 2.H — Reference-fixture sanity

- [ ] 2.H.1 Test `proofread_fixture_per_track_emission` parametrize over 6 tracks (`pytest.param(track_id, id=f"track_{track_id}")` for `track_id ∈ {0,1,2,3,4,5}`) loaded from the Nipponbare proofread `.slp`. For each track: (a) `result = nutation.compute(trajectory_df, cadence_s=300.0)`; (b) assert returned DataFrame has 1 row + 8 trait columns + 8 identity columns; (c) assert dtypes correct. 6 ids.

- [ ] 2.H.2 Test `proofread_fixture_plausibility_band_after_sg_detrend` (S1 round-1 + Sci-B1 round-2): parametrize 6 tracks; for each track call `nutation.compute(trajectory_df, cadence_s=300.0, coordinate="lateral")`; assert `1000 < result.T_nutation_median.iloc[0] < 10000` (plausibility band; PR #5 GREEN-phase had to soften this on raw `tip_x` without lateral projection; PR #6's lateral projection + SG-detrend should make this achievable). 6 ids. If any track fails, record per-track values in GREEN-phase Reconciliation Appendix.

- [ ] 2.H.3 Test `layer_2_derr_forensic_match_two_part_assertion` (S5 round-1 + Sci-B3 round-2): on the 6 Nipponbare proofread tracks with `coordinate="lateral"`, compute `per_track_residuals = [nutation.compute(trajectory_df_for_track_i, cadence_s=300.0).period_residual_vs_derr_reference.iloc[0] for i in range(6)]` AND `per_track_is_nutating = [...is_nutating.iloc[0] for i in range(6)]`. Assert TWO conditions:
  ```python
  # CC-7 median enforcement:
  assert abs(np.median(per_track_residuals)) < 0.02, (
      f"Layer-2 Derr forensic-match: median residual {np.median(per_track_residuals):.4f} "
      f"exceeds CC-7 ±2% target. Per-track residuals: {per_track_residuals}. "
      f"Per-track is_nutating: {per_track_is_nutating}.")

  # Per-track count check acknowledging biological variance:
  within_5pct_count = sum(
      abs(r) < _DERR_MATCH_TOLERANCE_FOR_TEST and nut
      for r, nut in zip(per_track_residuals, per_track_is_nutating))
  assert within_5pct_count >= 4, (
      f"Layer-2 Derr forensic-match: only {within_5pct_count}/6 tracks "
      f"within ±5% AND is_nutating==True. "
      f"Per-track residuals: {per_track_residuals}. "
      f"Per-track is_nutating: {per_track_is_nutating}.")
  ```
  1 aggregated id with informative-failure messages. **Semantic note per Sci-I1 round-2:** when `is_nutating == False` on a track, `period_residual_vs_derr_reference` is still POPULATED per S4 (it's a "ridge-of-noise" diagnostic); the per-track count's AND-conjunction correctly excludes such tracks; the median in (1) uses ALL 6 residuals because the median is robust enough to handle 1-2 outliers.

- [ ] 2.H.4 Test `issue_214_acceptance_aggregate` (cross-references §2.D.5): on 6 plate-001 tracks, count tracks where `T_nutation_iqr_post_filter < T_nutation_iqr_raw`; assert count ≥ 5. 1 id. If the count is <5, log per-track raw vs smoothed IQR + record in GREEN-phase Reconciliation Appendix.

  **Total §2.H**: 6+6+1+1 = **14 ids**.

### 2.I — Foundation-test migration

- [ ] 2.I.1 Edit `tests/test_circumnutation_foundation.py` `IMPLEMENTATIONS_WITH_CONSTANTS_KWARG` (around line 840): add `("nutation", "compute")` → 4 → **5** entries: kinematics, qc, synthetic, temporal_cwt, **nutation**.

- [ ] 2.I.2 Extend `test_implementation_accepts_constants_kwarg` (around line 855): add an `elif module_name == "nutation":` branch. Build a minimal valid trajectory_df (8 row-identity columns + frame + tip_x + tip_y + ≥1 row); call `fn(trajectory_df=df, cadence_s=300.0, constants=ConstantsT())`; assert returned object `isinstance(_, pd.DataFrame)`.

- [ ] 2.I.3 Rename `test_schema_version_is_1_and_constants_version_is_4` → `test_schema_version_is_1_and_constants_version_is_5` (line 206 — canonical name verified via Architecture-B1 round-1). Update assertion at line 213: `_constants._CONSTANTS_VERSION == 5`. Update docstring at line 207: `"_CONSTANTS_VERSION is 5 (bumped in PR #6)"`.

- [ ] 2.I.4 Append to the comment block at lines 26-34: a new line reading `"PR #6 (add-circumnutation-tier1-derr-faithful) adds nutation as the 5th implementation module (newly created, not a stub transition) and bumps _CONSTANTS_VERSION 4 → 5 by adding 6 Tier 1 / threshold defaults."`

- [ ] 2.I.5 **Add `"nutation"` to the explicit module list in `test_module_logger_is_namespaced`'s parametrize** (around lines 745-749). This is the **Copilot-precedent fix** from PR #4 and PR #5 — without explicit re-addition, the logger-namespace coverage would silently vanish if the parametrize iterates only over `STUB_MODULES`. Append `"nutation"` to the explicit list with a comment line: `# Added in PR #6: nutation (newly created implementation module; never a stub)`.

- [ ] 2.I.6 **No `STUB_MODULES` change** (no stub removed; `nutation` was never a stub). **No `STUBS_WITH_CONSTANTS_KWARG` change** (same reason). Verify by reviewing the post-edit grep against both lists.

## 3. Implementation (GREEN phase — make red tests pass)

- [ ] 3.1 Edit `sleap_roots/circumnutation/_constants.py`:
  - Below the PR #5 CWT-machinery defaults block, add 6 new module-level UPPER_SNAKE constants:
    - `RIDGE_CONTINUITY_FILTER_WINDOW: int = 5` (docstring cites #214 + design.md D4 + cross-reference to `WAVELET_DEFAULT_TEMPORAL`)
    - `NOISE_FLOOR_OUT_OF_BAND_FACTOR: float = 5.0` (docstring cites CC-8 + S7 GREEN-phase-revisit note)
    - `BAND_POWER_BAND_LOW_FACTOR: float = 0.5` (docstring cites theory.md §7.2 `[0.5T, 2T]`)
    - `BAND_POWER_BAND_HIGH_FACTOR: float = 2.0` (docstring cites theory.md §7.2)
    - `DERR_EXPECTED_PERIOD_S: float = 3333.0` (docstring cites preliminary_results §4.4 + Derr Sept-2025 pilot PDF; mentions ConstantsT-override path for non-rice species; cross-link to TEMPORAL_NYQUIST_RATIO_MAX multi-plate validation follow-up issue)
    - `TEMPORAL_NYQUIST_RATIO_MAX: float = 0.25` (docstring cites theory.md §6.5 + S2'' round-2 reconciliation; mentions multi-plate validation follow-up issue; reciprocally cross-references `NYQUIST_RATIO_MAX`)
  - Add the reciprocal cross-reference paragraph to `NYQUIST_RATIO_MAX`'s docstring: "dimensional separation from TEMPORAL_NYQUIST_RATIO_MAX — this constant is the SPATIAL cadence-Nyquist threshold (px/px); see TEMPORAL_NYQUIST_RATIO_MAX for the temporal sibling".
  - Extend `ConstantsT` with 6 new fields (defaults sourced from the module-level constants; mirror the PR #5 `attrs.field(default=..., validator=instance_of(...), converter=...)` pattern).
  - Extend `_default_constants_snapshot()` with 6 new keys.
  - Bump `_CONSTANTS_VERSION` from `4` to `5`. Update its docstring with a paragraph noting PR #6's contribution: `"PR #6 (add-circumnutation-tier1-derr-faithful) bumped this 4 → 5 by adding six new Tier 1 default constants (RIDGE_CONTINUITY_FILTER_WINDOW, NOISE_FLOOR_OUT_OF_BAND_FACTOR, BAND_POWER_BAND_LOW_FACTOR, BAND_POWER_BAND_HIGH_FACTOR, DERR_EXPECTED_PERIOD_S, TEMPORAL_NYQUIST_RATIO_MAX) to the overridable defaults set."`

- [ ] 3.2 Edit `sleap_roots/circumnutation/_geometry.py`:
  - Add `project_to_growth_axis_perpendicular(x, y) -> np.ndarray` per design.md D2.
  - Algorithm: validate (length match, finite, length ≥ 2); compute `net_length = math.hypot(x[-1]-x[0], y[-1]-y[0])`; if zero → return `np.full(len(x), np.nan, dtype=np.float64)` (Architecture-I3 round-1 graceful-NaN); otherwise compute growth-axis unit vector + perpendicular unit vector + center positions + project → return 1D float64 array of length n.
  - Add brief debug log on the zero-net-displacement path: `logger.debug("project_to_growth_axis_perpendicular: zero net displacement, returning all-NaN")`.

- [ ] 3.3 Edit `sleap_roots/circumnutation/_noise.py`:
  - Add `compute_sg_detrended(x, window, polynomial_order) -> np.ndarray` per design.md D3 (S1 round-1).
    - Algorithm: validate `len(x) >= window`, `window` odd ≥ 3, `polynomial_order < window`; return `x - scipy.signal.savgol_filter(x, window_length=window, polyorder=polynomial_order, mode='nearest')`.
  - Add `compute_fourier_noise_floor(x, cadence_s, t_nutation_median_s, factor) -> float` per design.md D3.
    - Algorithm: if `len(x) < 2` → NaN; `spectrum = np.abs(scipy.fft.rfft(x))`; `freqs = scipy.fft.rfftfreq(len(x), d=cadence_s)`; `f_cut = factor / t_nutation_median_s`; `band_mask = freqs > f_cut`; if not any → NaN; return `float(np.median(spectrum[band_mask]))`.
  - Add `import scipy.fft`, `import scipy.signal` if not already imported.

- [ ] 3.4 Edit `sleap_roots/circumnutation/temporal_cwt.py`:
  - Add `smooth_ridge(ridge_result, window=None, constants=None) -> RidgeResult` per design.md D4 (Q4).
  - Algorithm: resolve `constants or ConstantsT()`; if `window is None` → `window = resolved.RIDGE_CONTINUITY_FILTER_WINDOW`; validate `window` is int, ≥ 1, odd; emit debug log starting `"smooth_ridge("`; `smoothed_periods = scipy.ndimage.median_filter(ridge_result.periods_s, size=window, mode='nearest')`; return new `RidgeResult` with smoothed periods + carried-through other fields.
  - Add `import scipy.ndimage` if not already imported.

- [ ] 3.5 Create `sleap_roots/circumnutation/nutation.py` (new module):
  - Module docstring referencing the spec delta + `theory.md` §3.5/§6.5/§7.2/§7.6 + `roadmap.md` CC-7/CC-8 + `design.md` D1–D9.
  - Imports: `import logging`, `from typing import Any, Optional`; `import numpy as np`, `import pandas as pd`, `import scipy.fft`, `import scipy.stats`; relative imports `from ._constants import ConstantsT, BAND_POWER_NOISE_RATIO, BAND_POWER_BAND_LOW_FACTOR, BAND_POWER_BAND_HIGH_FACTOR, DERR_EXPECTED_PERIOD_S, NOISE_FLOOR_OUT_OF_BAND_FACTOR, RIDGE_CONTINUITY_FILTER_WINDOW, SG_DEGREE, SG_WINDOW_DETREND`; `from . import _geometry, _noise, temporal_cwt`; `from ._types import _validate_trajectory_df, ROW_IDENTITY_COLUMNS, _IDENTITY_5_TUPLE, _TIP_X_COLUMN, _TIP_Y_COLUMN`.
  - Module logger: `logger = logging.getLogger(__name__)`.
  - Module-level constants: `_NUTATION_TRAIT_COLUMNS: tuple[str, ...]` (8 traits in declared order); `_COORDINATE_CHOICES: frozenset[str] = frozenset({"lateral", "x", "y"})`.
  - Private helper `_check_cadence_s(cadence_s) -> float`: mirror PR #5's `_validate_cadence_s` semantics (positive finite, reject bool/np.bool_/str/list, accept int/float/np.integer/np.floating).
  - Private helper `_check_constants(constants) -> ConstantsT`: None → `ConstantsT()`; else must be `ConstantsT` (TypeError).
  - Private helper `_check_coordinate(coordinate) -> str`: must be in `_COORDINATE_CHOICES`; else ValueError.
  - Private helper `_select_signal(group, coordinate) -> np.ndarray`: dispatch to `_geometry.project_to_growth_axis_perpendicular(x, y)` for "lateral"; raw `tip_x` for "x"; raw `tip_y` for "y".
  - Private helper `_compute_band_power_ratio(x, cadence_s, t_nutation_median_s, constants) -> float`: per design.md D7 algorithm.
  - Private helper `_compute_one_track(group, cadence_s, coordinate, constants) -> dict`: implements the 9-step sequencing per design.md D5 pseudocode (S8' round-2 + S1 round-1 + S4 round-1 + S2 + S3 + Sci-I1 round-2 + S5 round-2 + S6 round-2 incorporated):
    1. Project + SG-detrend
    2. CWT compute_scaleogram + extract_ridge + smooth_ridge
    3. T_nutation_median + T_nutation_iqr from COI-masked smoothed periods
    4. A_nutation_envelope_max_px from COI-masked raw ridge amplitudes
    5. noise_floor_estimate from FFT
    6. band_power_ratio from FFT
    7. is_nutating gate
    8. Derived traits (period_residual_vs_derr_reference, cadence_nyquist_ratio) — both ALWAYS computed (S4 round-2)
    9. NaN-gate 3 traits ONLY (T_nutation_median, T_nutation_iqr, A_nutation_envelope_max_px) when is_nutating==False
  - Public `compute(trajectory_df, cadence_s, coordinate="lateral", constants=None) -> pd.DataFrame`:
    - Validate inputs (delegate to `_validate_trajectory_df` for trajectory_df; explicit `_check_cadence_s` + `_check_coordinate` + `_check_constants`)
    - Resolve `constants = _check_constants(constants)`
    - Emit single `logger.debug` starting `"nutation.compute("` containing tokens `n_tracks=`, `coordinate=`, `cadence_s=`
    - Per-track loop via `groupby(_IDENTITY_5_TUPLE, dropna=False, sort=False)`; for each group call `_compute_one_track(group, cadence_s=cadence_s, coordinate=coordinate, constants=constants)`; catch ValueError-from-`_geometry` → emit all-NaN trait row with `is_nutating=False`
    - Build per-row dict `{**identity, **traits}` and assemble `pd.DataFrame`
    - Enforce column order: 8 identity columns + 8 trait columns
    - Enforce dtypes: 7 float64 + 1 bool
    - Return the DataFrame

- [ ] 3.6 Run `uv run pytest tests/test_circumnutation_nutation.py -x --no-header --tb=short` — most tests should pass except §2.B.2 (canary).

- [ ] 3.7 Run `uv run pytest tests/test_circumnutation_foundation.py -x --no-header --tb=short` — confirm all foundation tests pass with the migration changes.

### 3.8 Canary value capture (one-time)

- [ ] 3.8.1 Write `scripts/circumnutation/capture_nutation_canary.py` **in the repo** (NOT in the vault, per PR #5 §3.5 precedent). Script generates `df = synthetic.generate_trajectory(random_state=0, n_frames=575, T_nutation_s=3333, cadence_s=300, noise_sigma_px=0.5)`; converts to trajectory_df; calls `result_df = nutation.compute(trajectory_df, cadence_s=300.0)`; prints provenance header (date, OS, BLAS/scipy/pywt/numpy versions, full `ConstantsT()` snapshot, run parameters, git SHA of `nutation.py`) + 3 canary values (`T_nutation_median`, `band_power_ratio`, `noise_floor_estimate`) in copy-paste-ready format. Supports `--out tests/test_circumnutation_nutation.canary.json` flag.

- [ ] 3.8.2 Run the capture: `uv run python scripts/circumnutation/capture_nutation_canary.py`. Copy the 3 values into §2.B.2 (replacing `np.full(3, np.nan)` placeholder). Copy provenance header into the test docstring.

- [ ] 3.8.3 Re-run `uv run pytest tests/test_circumnutation_nutation.py::test_cross_os_canary_at_atol_1e_6 -x` — expect green.

## 4. Documentation

- [ ] 4.1 Update `docs/changelog.md` under `## [Unreleased]` / `### Added`:
  ```markdown
  - **circumnutation**: Tier 1 nutation trait emission (`sleap_roots.circumnutation.nutation.compute`) producing 8 traits per `theory.md` §7.2 + §7.6 + §6.5: `T_nutation_median`, `T_nutation_iqr`, `A_nutation_envelope_max_px`, `band_power_ratio`, `noise_floor_estimate`, `is_nutating`, `period_residual_vs_derr_reference`, `cadence_nyquist_ratio`. Composes PR #5 CWT primitives + new `_geometry.project_to_growth_axis_perpendicular` (CC-7 lateral projection) + new `_noise.compute_sg_detrended` (per `preliminary_results §3.4`) + new `_noise.compute_fourier_noise_floor` (CC-8) + new `temporal_cwt.smooth_ridge` (closes #214, ridge-tracking continuity post-filter). `is_nutating` boolean gates NaN-emission of 3 strictly biological traits per scientific-honesty principle (S4); 5 diagnostic traits remain populated. Layer-2 Derr forensic-match enforced on Nipponbare plate-001 proofread fixture via two-part assertion: median across 6 tracks within ±2% AND ≥4 of 6 tracks within ±5%. 6 new defaults in `_constants.py` + `ConstantsT`; `_CONSTANTS_VERSION` 4 → 5. See `openspec/changes/add-circumnutation-tier1-derr-faithful/`.
  ```

- [ ] 4.2 No `docs/circumnutation/theory.md` or `docs/circumnutation/roadmap.md` edits inside this PR. Roadmap row #6 status `⬜ → ✅` happens post-merge during `cleanup-merged` per §8.

## 5. Verification (gates before opening PR)

- [ ] 5.1 Full test suite: `uv run pytest tests/ -x` — all tests pass.

- [ ] 5.2 Module coverage: `uv run pytest tests/test_circumnutation_nutation.py --cov=sleap_roots --cov-report=term-missing` — 100% coverage on `sleap_roots/circumnutation/nutation.py` AND on the added helpers (`_geometry.project_to_growth_axis_perpendicular`, `_noise.compute_sg_detrended`, `_noise.compute_fourier_noise_floor`, `temporal_cwt.smooth_ridge`). Use `--cov=sleap_roots` whole-package to avoid the numpy 2.x + scipy + coverage interaction.

- [ ] 5.3 Project-wide coverage: `uv run pytest tests/ --cov=sleap_roots --cov-fail-under=84` — coverage holds at 84%+.

- [ ] 5.4 Black format check: `uv run black --check sleap_roots tests` — no formatting issues.

- [ ] 5.5 Pydocstyle: `uv run pydocstyle --convention=google sleap_roots/` — no docstring violations.

- [ ] 5.6 uv lock check: `uv lock --check` — lockfile in sync (no new deps).

- [ ] 5.7 mkdocs build: `uv run mkdocs build` — confirm auto-generated reference pages for `nutation` (compute) + new helpers render without errors.

- [ ] 5.8 OpenSpec strict validation: `openspec validate add-circumnutation-tier1-derr-faithful --strict` — no validation errors.

- [ ] 5.9 Pre-merge skill: `/pre-merge` — runs the full gate checklist + smoke-tests.

## 6. PR open

- [ ] 6.1 Stage files: `git add sleap_roots/circumnutation/_constants.py sleap_roots/circumnutation/_geometry.py sleap_roots/circumnutation/_noise.py sleap_roots/circumnutation/nutation.py sleap_roots/circumnutation/temporal_cwt.py tests/test_circumnutation_foundation.py tests/test_circumnutation_nutation.py docs/changelog.md scripts/circumnutation/capture_nutation_canary.py openspec/changes/add-circumnutation-tier1-derr-faithful/`

- [ ] 6.2 Commit message (Conventional Commits): `feat(circumnutation): Tier 1 Derr-faithful trait emission (PR #6)`. Body cites the OpenSpec change-id, the design.md sections, the 2-round critical-review reconciliation, the Layer-2 Derr forensic-match acceptance status, and closes #214.

- [ ] 6.3 Draft GitHub sub-issue body at `c:\vaults\sleap-roots\circumnutation\github_issues\issue_add-circumnutation-tier1-derr-faithful.md` referencing epic #197 + roadmap row #6 + closes #214 + theory.md anchors.

- [ ] 6.4 Draft TWO new follow-up issue bodies at `c:\vaults\sleap-roots\circumnutation\github_issues\`:
  - `issue_derr_layer_2_upgrade_pending_raw_data.md`: "circumnutation: upgrade period_residual_vs_derr_reference to spectral-shape distance when Derr provides raw scaleogram numerics"
  - `issue_temporal_nyquist_threshold_validation.md`: "circumnutation: validate TEMPORAL_NYQUIST_RATIO_MAX from literature + multi-plate data" (mirror #205-#208 pattern)

- [ ] 6.5 Show Elizabeth all drafts (sub-issue body + 2 follow-up issue bodies + PR body) BEFORE posting; await per-item authorization (per repo hard constraint: drafts go to vault first, shown individually).

- [ ] 6.6 After Elizabeth's go-ahead: `gh issue create` per draft (sub-issue first → record number; then 2 follow-up issues → record numbers). Then `gh pr create --title "feat(circumnutation): Tier 1 Derr-faithful trait emission (#<issue>)"` with body referencing the 3 issues + closes #214.

## 6.5. Copilot review reconciliation

- [ ] 6.5.1 After PR opens, GitHub Copilot will auto-review. Invoke `/copilot-review`.
- [ ] 6.5.2 Verify each Copilot finding empirically. Distinguish legitimate bugs from style noise.
- [ ] 6.5.3 Land Copilot fixes in a reconciliation commit: `fix(circumnutation): address GitHub Copilot review on PR #<n>` with body summarizing each finding + resolution. Expect 2-3 rounds (PR #5 needed 3 rounds — Copilot re-reviews on every fixup push).

## 7. Self-review (post-Copilot, before merge)

- [ ] 7.1 Invoke `/review-pr` — launches the 5-subagent post-PR review (code quality / security / TDD / docs / architecture).
- [ ] 7.2 Address BLOCKING + IMPORTANT findings inline; defer the rest with rationale.
- [ ] 7.3 Post the review verdict as a PR comment via `gh pr review --comment` with the `> **Verdict: APPROVE** (posted as comment — cannot approve your own PR)` banner.

## 8. Post-merge cleanup

- [ ] 8.1 Invoke `/cleanup-merged` — handles `git fetch origin --prune`, branch cleanup, `/openspec:archive` invocation, and the verify checklist.

- [ ] 8.2 `/openspec:archive` (invoked from `/cleanup-merged`) runs `openspec archive add-circumnutation-tier1-derr-faithful --yes` to fold the requirements into `openspec/specs/circumnutation/spec.md` and move the change folder to `openspec/changes/archive/YYYY-MM-DD-add-circumnutation-tier1-derr-faithful/`.

- [ ] 8.3 Update `docs/circumnutation/roadmap.md` row PR #6: `⬜ → ✅` with issue + PR cross-links + #214 closed reference. Mirror the format used for PR #1 / #2 / #3 / #4 / #5.

- [ ] 8.4 Update `docs/changelog.md`: confirm the `[Unreleased] / ### Added` entry from §4.1 is present and accurate.

- [ ] 8.5 Update the Notion task page (https://www.notion.so/Circumnutation-work-sleap-roots-3494a67a7667818db2aeee80d76efdd7) with a new `Status update — YYYY-MM-DD` section. Draft to vault first at `c:\vaults\sleap-roots\circumnutation\notion_status_update_YYYY-MM-DD.md`; show Elizabeth before posting via the Notion MCP.
