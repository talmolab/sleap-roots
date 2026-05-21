# Tasks for add-circumnutation-synthetic-generator

TDD-ordered. Tests precede implementation per `superpowers:test-driven-development`. **Do not push commits from section 2 (red phase — tests only) without section 3 (implementation) in the same push** — the new test file imports `generate_trajectory` from `synthetic.py` which section 3 implements, and the foundation test migration (§2.I) flips `synthetic` semantics in `STUBS_WITH_CONSTANTS_KWARG`. The suite is expected to be red between 2.x and 3.x; PR is one logical unit.

## 1. Pre-flight (no code changes; just confirms ground-truth)

- [ ] 1.1 Confirm the existing fixtures used by 2.H are present: `tests/data/circumnutation_nipponbare_plate_001/plate_001_greyscale.tracked_proofread.slp` (committed via Git LFS in PR #2). Run `uv run python -c "import sleap_io as sio; labels = sio.load_slp('tests/data/circumnutation_nipponbare_plate_001/plate_001_greyscale.tracked_proofread.slp'); print(len(labels), labels.tracks)"` → expect 575 frames, 6 tracks.
- [ ] 1.2 Run baseline: `uv run pytest tests/ -x` — confirm green BEFORE any changes. Foundation + Tier 0 + QC tests must be passing at HEAD.
- [ ] 1.3 Run baseline coverage: `uv run pytest tests/ --cov=sleap_roots --cov-report=term --cov-fail-under=84` — confirm starting at 84%+ baseline.

## 2. Tests (write before implementation — TDD red phase)

- [ ] 2.1 Create `tests/test_circumnutation_synthetic.py` with module docstring referencing the spec delta + theory.md §4 + §8 + design.md sections D1-D14.

  **Test-file imports** (per round-2 Spec reviewer I1): the test module SHALL import:
  - `import copy` (used by §2.B.5b for `deepcopy` of RNG state)
  - `import math` (used by §2.G.4 / §2.G.5 setup blocks for `math.pi`, `math.atan`)
  - `import numpy as np`
  - `import numpy.testing as npt` (used by §2.B.5b for structured dict equality)
  - `import pandas as pd`
  - `import pytest`
  - `from sleap_roots.circumnutation import _geometry, kinematics, qc, synthetic`
  - `from sleap_roots.circumnutation._constants import (ConstantsT, SYNTHETIC_T_NUTATION_S, SYNTHETIC_AMPLITUDE_PX, SYNTHETIC_GROWTH_RATE_PX_PER_FRAME, SYNTHETIC_NOISE_SIGMA_PX, SYNTHETIC_CADENCE_S, SYNTHETIC_N_FRAMES, SYNTHETIC_GROWTH_AXIS_ANGLE_RAD)`
  - `from sleap_roots.circumnutation._types import ROW_IDENTITY_COLUMNS, REQUIRED_PER_FRAME_COLUMNS`

- [ ] 2.1b Add a module-level pytest fixture `_synthetic_setup` (per round-2 Spec reviewer I2) that computes the derived constants used by §2.G.4/§2.G.5:
  ```python
  @pytest.fixture
  def synthetic_setup():
      T_nutation_s = SYNTHETIC_T_NUTATION_S          # 3333.0
      cadence_s    = SYNTHETIC_CADENCE_S             # 300.0
      growth_rate  = SYNTHETIC_GROWTH_RATE_PX_PER_FRAME   # 4.29
      omega          = 2.0 * math.pi / T_nutation_s
      v_growth_per_s = growth_rate / cadence_s
      return {
          "T_nutation_s": T_nutation_s,
          "cadence_s": cadence_s,
          "growth_rate": growth_rate,
          "omega": omega,
          "v_growth_per_s": v_growth_per_s,
      }
  ```
  §2.G.4 and §2.G.5 (and any other test that needs the derived `omega` / `v_growth_per_s`) take this fixture as a parameter, avoiding the pytest local-variable-sharing pitfall.

### 2.A — Schema/structural tests (spec Requirement: Synthetic trajectory generator)

- [ ] 2.A.1 Test `output_columns_match_spec`: assert columns equal `ROW_IDENTITY_COLUMNS + REQUIRED_PER_FRAME_COLUMNS` (both imported from `_types`) — explicitly the 8 row-identity columns + `frame` + `tip_x` + `tip_y` in that order. Hard-code the expected column order list at module level.

- [ ] 2.A.2 Test `output_dtypes_match_contract` (design.md D3): assert `frame.dtype == np.int64`, `tip_x.dtype == np.float64`, `tip_y.dtype == np.float64`, `plant_id.dtype == np.int64`, `track_id.dtype == np.int64`. For `series`, `sample_uid`, `timepoint`, `plate_id`, `genotype`, `treatment` assert `dtype == object`.

- [ ] 2.A.3 Test `frame_indexing_is_zero_based` (TDD reviewer I1): `df["frame"].iloc[0] == 0` AND `df["frame"].iloc[-1] == n_frames - 1` AND `(np.diff(df["frame"]) == 1).all()` (strict monotonic ascending).

- [ ] 2.A.4 Test `plant_id_equals_track_id_by_default` (TDD reviewer I1): with default kwargs (`plant_id=0`, `track_id=0`), assert `df["plant_id"].equals(df["track_id"])`. This is the foundation convention enforced explicitly here.

- [ ] 2.A.5 Test `none_genotype_becomes_nan_not_string` (TDD reviewer I1): `generate_trajectory(genotype=None, treatment=None)` → `df["genotype"].isna().all() == True` AND `df["genotype"].dtype == object` AND no literal `"None"` strings appear (assert `not (df["genotype"] == "None").any()`). Same for `treatment`.

- [ ] 2.A.6 Test `row_count_equals_n_frames`: call with `n_frames=10`; assert `len(df) == 10`. Parametrize over `n_frames ∈ {1, 5, 10, 100, 575}`.

- [ ] 2.A.7 Test `signature_is_kw_only_positional_raises_typeerror` (design.md D8): `generate_trajectory(575)` raises `TypeError` (because the function declares `*,` before all parameters).

- [ ] 2.A.8 Test `signature_has_no_px_per_mm` (re-asserts foundation spec scenario): inspect the signature via `inspect.signature(generate_trajectory)`; assert `"px_per_mm"` is NOT in the parameter list.

- [ ] 2.A.9 Test `n_frames_1_emits_single_row` (edge — arch N3 / TDD reviewer N3): `generate_trajectory(n_frames=1, ...)` returns 1-row DataFrame; `kinematics.compute(df)` on this single-frame track emits a NaN trait row per its existing single-frame contract without exception.

- [ ] 2.A.10 Test `growth_axis_angle_outside_canonical_range_round_trips` (edge — arch N3): `generate_trajectory(growth_axis_angle_rad=5*math.pi, ...)` runs without raising; the resulting trajectory matches `growth_axis_angle_rad=5*math.pi - 2*math.pi*2 = math.pi` numerically (rotation is well-defined mod 2π).

- [ ] 2.A.11a Test `caplog_no_warning_or_error_emissions_on_default_call` (round-1 TDD I7 + round-2 TDD IMP-1 — unconditional half): with `caplog.set_level(logging.WARNING)`, call `generate_trajectory()` with defaults; assert that the module logger emits NO records at WARNING / ERROR / CRITICAL levels (`assert caplog.records == []`). The happy-path default call must produce a silent module logger.

- [ ] 2.A.11b Test `caplog_branch_coverage_for_any_debug_emissions` (round-2 TDD IMP-1 — conditional half written AFTER §3.5 implementation lands): for EACH `logger.debug(...)` call introduced by `synthetic.py` (e.g., the `noise_sigma_px == 0` short-circuit notification, parameter-resolution diagnostics), add a targeted `caplog` assertion that the call site is exercised at least once and the message contains the expected format string. This is required for the 100% branch coverage target in §5.1; the §5.1 "Acceptable uncovered branches" carve-out for `logger.debug` is REMOVED — DEBUG-level calls SHALL be branch-covered like any other code path.

### 2.B — Determinism tests (spec Requirement: Synthetic trajectory generator / CC-6 determinism contract)

- [ ] 2.B.1 Test `same_int_seed_bit_identical`: call `generate_trajectory(random_state=0, ...)` twice; assert `tip_x` arrays equal via `np.array_equal` (bit-identical, not just `allclose`). ALSO assert known first-3-elements expected values for `random_state=0` (captured during impl per §3.7 below).

  **Canary purpose (not an oracle, per TDD reviewer I1).** The known first-3-values are a REGRESSION DETECTOR for future numpy PCG64 drift, NOT a correctness oracle. They are recorded from the GREEN-phase implementation output and locked in; bit-identical reproduction across CI runs proves determinism is preserved, not that the closed-form math is correct (that's tested separately via 2.C). Per design.md R4, if a future numpy major release breaks PCG64 stability, this test fails fast and we pin the numpy version in `pyproject.toml`.

- [ ] 2.B.2 Test `same_generator_advances_state` (documents Generator behavior): pass the SAME `rng = np.random.default_rng(0)` to two sequential calls; assert their `tip_x` arrays DIFFER (the Generator advanced state between calls).

- [ ] 2.B.3 Test `int_seed_equiv_default_rng_seed`: assert `generate_trajectory(random_state=42, ...)` equals `generate_trajectory(random_state=np.random.default_rng(42), ...)` (bit-identical).

- [ ] 2.B.4 Test `different_seeds_differ`: assert `generate_trajectory(random_state=0, ...)` and `generate_trajectory(random_state=1, ...)` produce DIFFERENT `tip_x` arrays (`not np.allclose`).

- [ ] 2.B.5 Test `noise_zero_short_circuits_rng` (design.md D11): assert `generate_trajectory(noise_sigma_px=0, random_state=None, ...)` and `generate_trajectory(noise_sigma_px=0, random_state=42, ...)` produce IDENTICAL output (RNG path not entered).

- [ ] 2.B.5b Test `noise_zero_does_not_advance_caller_generator_state` (TDD reviewer I2 + design.md D11): explicit `bit_generator.state` before/after equality via `numpy.testing.assert_equal` for structured deep comparison (TDD reviewer I4 — prefer `numpy.testing.assert_equal` over `==` because the PCG64 state dict contains nested dicts of ints; `assert_equal` is explicit about structured equality):
  ```python
  import numpy.testing as npt
  rng = np.random.default_rng(42)
  state_before = copy.deepcopy(rng.bit_generator.state)
  _ = generate_trajectory(noise_sigma_px=0, random_state=rng, ...)
  state_after = rng.bit_generator.state
  npt.assert_equal(state_before, state_after)   # nested-dict deep equality
  ```

### 2.C — Parameter recovery via Tier 0 (spec Requirement: Synthetic trajectory generator + TDD reviewer B1/B2 splits)

- [ ] 2.C.1 Test `v_long_recovery_exact_at_amplitude_zero` (TDD reviewer B1 — pure-linear trajectory for exact-equality): `generate_trajectory(amplitude_px=0, growth_rate_px_per_frame=4.29, noise_sigma_px=0, growth_axis_angle_rad=math.pi/2)` → `kinematics.compute(df)["v_long_signed_median_px_per_frame"].iloc[0]` equals `4.29` within `1e-9` (loosened from 1e-10 per TDD reviewer B3 to accommodate cross-platform BLAS rounding accumulation through `(4.29/300) · (i·300)` reintroduction).
  - **Additional assertion** (scientific-rigor N2): `pd.isna(kinematics.compute(df)["long_lat_ratio"].iloc[0])` — the pure-linear trajectory has `v_lat_abs_median == 0` exactly, which per the kinematics spec produces NaN. Locking this prevents a future bug from returning `inf` or a spurious value here.

- [ ] 2.C.2 Test `angular_amplitude_small_angle_recovery` (TDD reviewer B2 — small-amplitude analytical test): `generate_trajectory(amplitude_px=1.0, T_nutation_s=3333.0, growth_rate_px_per_frame=4.29, cadence_s=300.0, n_frames=575, noise_sigma_px=0)` → recovered `angular_amplitude` matches `2 * arctan(amplitude_px * omega / (2 * v_growth_per_s)) ≈ 2 * arctan(0.5 * (2π/3333) / (4.29/300)) ≈ 0.131 rad` within ±5%. In this regime small-angle and exact formula agree to <1%.

- [ ] 2.C.3 Test `angular_amplitude_plate001_sanity_with_exact_formula` (TDD reviewer B2 — plate-001 sanity with the EXACT relation, NOT small-angle): `generate_trajectory()` with all defaults → recovered `angular_amplitude` matches `2 * arctan(amplitude_px * omega / (2 * v_growth_per_s)) ≈ 1.17 rad` within ±15% (theory.md §8 spatial tolerance). Also assert `kinematics.compute(df)["growth_axis_unreliable"].iloc[0] == False` (per E5 safety margin derivation — D ≈ 2467 px, K·σ ≈ 20 px, factor of ~123×).

- [ ] 2.C.4 Test `synth_growth_axis_inference_matches_kwarg`: with `noise_sigma_px=0`, assert the kinematics-inferred `principal_axis_angle` matches `growth_axis_angle_rad` within `1e-6`. Confirms the net-displacement-axis inference (kinematics.py:218) is numerically consistent with the synthesis convention (design.md D10 E1 + arch reviewer N2).

### 2.D — Round-trip noise sanity via QC (spec Requirement: Synthetic trajectory generator + scientific-rigor B1 xy-quadrature interpretation)

With per-axis σ = `noise_sigma_px / √2`, the xy-quadrature noise estimators recover `noise_sigma_px` directly.

- [ ] 2.D.1 Test `sg_residual_recovers_noise_sigma`: `generate_trajectory(noise_sigma_px=2.0, random_state=42, ...)` → `qc.compute(df)["sg_residual_xy"].iloc[0]` within `[1.7, 2.3]` (±15% of 2.0, accommodates SG under-bias).

- [ ] 2.D.2 Test `d2_noise_recovers_noise_sigma`: same trajectory → `d2_noise_xy` within `[1.7, 2.3]`.

- [ ] 2.D.3 Test `msd_noise_recovers_noise_sigma`: same trajectory → `msd_noise_xy` within `[1.7, 2.3]`.

- [ ] 2.D.4 Test parametrized over `noise_sigma_px ∈ {1.0, 2.0, 4.0}` across all 3 estimators (9 ids); each within ±15% of input σ. Anchors the round-trip claim across the noise-amplitude range used by Layer-1 validation.

- [ ] 2.D.5 Test `track_is_clean_true_for_clean_synthetic`: `generate_trajectory(noise_sigma_px=2.0, random_state=42, ...)` with otherwise-default plate-001 params → `qc.compute(df)["track_is_clean"].iloc[0] == True` AND `qc.compute(df)["qc_failure_reason"].iloc[0] == ""`. Verifies the synthetic generator produces tracks that ARE clean by QC's criteria.

### 2.E — Handedness sign convention (spec Requirement: Synthetic trajectory generator + scientific-rigor N1 cross-check)

All 4 sub-tests use `noise_sigma_px = 0` for unambiguous determinism (TDD reviewer I4).

- [ ] 2.E.1 Test `handedness_plus_one_gives_positive_psi_g_drift`: `generate_trajectory(handedness=+1, noise_sigma_px=0, ...)` → `np.mean(np.diff(_geometry.compute_psi_g(tip_x, tip_y))) > 0`.

- [ ] 2.E.2 Test `handedness_minus_one_gives_negative_psi_g_drift`: same as 2.E.1 with `handedness=-1` → `np.mean(np.diff(_geometry.compute_psi_g(tip_x, tip_y))) < 0`.

- [ ] 2.E.3 Test `handedness_curl_sign_cross_check_matches_psi_g` (round-1 scientific-rigor N1 + round-2 TDD NIT-3 + Spec I3): parametrize `handedness ∈ {+1, -1}` with `noise_sigma_px=0`. For each parametrize id, compute the curl-sign via finite differences `curl = np.mean(np.diff(tip_x)[1:] * np.diff(np.diff(tip_y)) - np.diff(tip_y)[1:] * np.diff(np.diff(tip_x)))` independently of `_geometry.compute_psi_g`, then assert BOTH `np.sign(curl) == np.sign(np.mean(np.diff(_geometry.compute_psi_g(tip_x, tip_y))))` AND `np.sign(curl) == handedness`. Covers the spec scenario "handedness sign agrees with independent curl-sign cross-check" for both signs in a single test function via parametrize. Guards against future inversion of `_geometry.compute_psi_g`'s `atan2(dx, dy)` argument order.

- [ ] 2.E.4 Test `handedness_robust_to_noise_at_n_frames_575`: `generate_trajectory(handedness=+1, noise_sigma_px=2.0, n_frames=575, ...)` with seed=42 → `np.mean(np.diff(_geometry.compute_psi_g(tip_x, tip_y))) > 0` (systematic dominates noise at this n). Documents noise robustness without making it a load-bearing assertion in 2.E.1.

### 2.F — Validation / error path (spec Requirement: Synthetic trajectory generator + TDD reviewer I5 parametrize structure)

Single parametrize over `(param_name, invalid_value, exception_type, match_pattern)`. Approximately 60 ids covering:

- [ ] 2.F.0 Test `positional_call_rejected`: `generate_trajectory(575)` raises `TypeError` matching pattern about positional arguments.

- [ ] 2.F.1 Parametrize table — invalid integers: `n_frames ∈ {0, -1, True, 1.5, "100", np.nan, np.inf}` each raises `ValueError` or `TypeError` whose message contains `"n_frames"`.

- [ ] 2.F.2 Parametrize table — invalid positive floats: `cadence_s ∈ {0.0, -1.0, np.nan, np.inf, -np.inf, True, "300"}`. Same for `T_nutation_s`. Each names the field.

- [ ] 2.F.3 Parametrize table — invalid non-negative floats: `amplitude_px ∈ {-1.0, np.nan, np.inf, True, "10"}`. `noise_sigma_px ∈ {-1.0, np.nan, np.inf, True, "2"}`. (Note: `0.0` is VALID for both per D8.)

- [ ] 2.F.4 Parametrize table — invalid finite floats: `growth_rate_px_per_frame, growth_axis_angle_rad, x0_px, y0_px, initial_phase_rad ∈ {np.nan, np.inf, -np.inf, True, "X"}` (TDD reviewer I6: `-np.inf` was missing in the original draft). (Note: negative values ARE valid; zero IS valid.)

- [ ] 2.F.5 Parametrize table — invalid handedness: `handedness ∈ {0, 2, -2, 1.0, True, "+1", None}` — exactly `+1` or `-1` only.

- [ ] 2.F.6 Parametrize table — invalid random_state: `random_state ∈ {1.5, "42", np.random.RandomState(0)}` (rejects legacy RandomState per D5).

- [ ] 2.F.7 Parametrize table — invalid constants: `constants ∈ {dict(), "constants", 42}`.

- [ ] 2.F.8 Parametrize table — invalid identity-column types: `plant_id, track_id ∈ {1.5, True, "0", np.nan}`. `series, sample_uid, timepoint, plate_id ∈ {0, 1.5, np.nan}` (None allowed for the optional 4 — `genotype, treatment` — but NOT for these 4 mandatory string fields).

Every test asserts that the exception message names the offending field (via `match=` pattern).

### 2.G — Constants snapshot + ConstantsT override + resolution-order (spec Requirement: Module-level constants)

- [ ] 2.G.1 Test `constants_snapshot_contains_7_new_keys_and_preserves_pr3_qc_keys`: call `_default_constants_snapshot()` and assert:
  - The 7 new keys ARE present with PR #4 default values: `SYNTHETIC_T_NUTATION_S`, `SYNTHETIC_AMPLITUDE_PX`, `SYNTHETIC_GROWTH_RATE_PX_PER_FRAME`, `SYNTHETIC_NOISE_SIGMA_PX`, `SYNTHETIC_CADENCE_S`, `SYNTHETIC_N_FRAMES`, `SYNTHETIC_GROWTH_AXIS_ANGLE_RAD` (per design.md D4)
  - **Regression guard** (TDD reviewer I3): the 4 PR #3 QC keys are STILL present with their default values: `FRAC_OUTLIER_STEPS_MAX = 0.05`, `WORST_STEP_RATIO_MAX = 5`, `SG_MSD_AGREEMENT_MAX = 1.5`, `D2_MSD_AGREEMENT_MAX = 1.5`. Catches a hypothetical regression where extending the snapshot dropped a prior key.

- [ ] 2.G.2 Test `_CONSTANTS_VERSION_is_3`: assert `_constants._CONSTANTS_VERSION == 3`.

- [ ] 2.G.3 Test `constantsT_extended_with_7_new_fields`: construct `ConstantsT(SYNTHETIC_AMPLITUDE_PX=20.0)` and assert `inst.SYNTHETIC_AMPLITUDE_PX == 20.0` while other fields default to module-level. Repeat for the other 6 new fields.

- [ ] 2.G.4 Test `constantsT_override_propagates_when_kwarg_omitted` (TDD reviewer B3 + design.md D13 resolution-order; setup variables explicit per arch reviewer N1):
  ```python
  # setup constants used in the analytical prediction
  T_nutation_s = SYNTHETIC_T_NUTATION_S         # 3333.0
  cadence_s    = SYNTHETIC_CADENCE_S            # 300.0
  growth_rate  = SYNTHETIC_GROWTH_RATE_PX_PER_FRAME   # 4.29
  omega          = 2.0 * math.pi / T_nutation_s
  v_growth_per_s = growth_rate / cadence_s

  # default kwarg + custom constants → uses constants value
  df1 = generate_trajectory(noise_sigma_px=0,
                            constants=ConstantsT(SYNTHETIC_AMPLITUDE_PX=20.0))
  ang_amp_1 = kinematics.compute(df1)["angular_amplitude"].iloc[0]
  expected_1 = 2 * math.atan(20.0 * omega / (2 * v_growth_per_s))  # uses amplitude_px=20
  assert abs(ang_amp_1 - expected_1) / expected_1 < 0.15  # ±15% per theory.md §8 spatial tolerance
  ```

- [ ] 2.G.5 Test `explicit_kwarg_overrides_constants_override` (TDD reviewer B3 + design.md D13; reuses the setup from 2.G.4):
  ```python
  # (omega, v_growth_per_s from 2.G.4 setup)
  # explicit kwarg + custom constants → uses kwarg value (kwarg wins)
  df2 = generate_trajectory(amplitude_px=15.0, noise_sigma_px=0,
                            constants=ConstantsT(SYNTHETIC_AMPLITUDE_PX=20.0))
  ang_amp_2 = kinematics.compute(df2)["angular_amplitude"].iloc[0]
  expected_2 = 2 * math.atan(15.0 * omega / (2 * v_growth_per_s))  # uses amplitude_px=15 (kwarg)
  assert abs(ang_amp_2 - expected_2) / expected_2 < 0.15
  # NB: with constants=20 and kwarg=15, expected_1 (≈1.62 rad) ≠ expected_2 (≈1.32 rad);
  # the test discriminates the two paths AND assertion 2.G.4 must FAIL if the impl
  # ignored constants. Add a sanity assertion that ang_amp_1 != ang_amp_2 within 1%.
  ```

- [ ] 2.G.6 Test `constants_none_uses_module_defaults`: `generate_trajectory(amplitude_px=None, noise_sigma_px=0, ..., constants=None)` produces the same output as `generate_trajectory(amplitude_px=SYNTHETIC_AMPLITUDE_PX, ...)`.

### 2.H — Reference-fixture agreement (spec Requirement: Synthetic trajectory generator + scientific-rigor B3 — agreement NOT hardcoded numbers)

Use the EXISTING plate-001 `.slp` fixture loaded by PR #2 / PR #3 reference-value tests; compute Tier 0 + QC traits on BOTH the real-data trajectory AND `generate_trajectory()` with defaults; assert agreement within ±15%.

- [ ] 2.H.1 Test `synth_sg_residual_matches_real_plate_within_15pct`: load plate 001 fixture, build `trajectory_df` per PR #2's `load_nipponbare_fixture` helper, compute `qc.compute(real_df)["sg_residual_xy"].median()` (median across 6 tracks). Synth: `qc.compute(generate_trajectory())["sg_residual_xy"].iloc[0]`. Assert `abs(real - synth) / real < 0.15`. Skip with `pytest.skip` if fixture file missing.

- [ ] 2.H.2 Test `synth_mean_step_matches_real_plate_within_15pct`: real per-frame mean total step (using `np.linalg.norm(np.diff(xy, axis=0), axis=1).mean()` per track, then median across tracks). Synth per-frame mean total step. Assert agreement within ±15%. Anchors on MEAN total step (5.83 px expected from prelim §4.1), NOT median (6.93) — see design.md D4 calibration note.

- [ ] 2.H.3 Test `synth_growth_axis_unreliable_false_matches_real_plate`: `kinematics.compute(real_df)["growth_axis_unreliable"].any() == False` AND `kinematics.compute(generate_trajectory())["growth_axis_unreliable"].iloc[0] == False`. Both confirm clean tracks under the gate.

### 2.I — Foundation test migration (per spec Requirement: Package layout MODIFIED + architecture I3 split-table refactor)

**Verified state at HEAD before any edits** (per `/openspec-review` TDD reviewer B1):

- `tests/test_circumnutation_foundation.py:34-43` — `STUB_MODULES` is an 8-entry list **including** `("synthetic", "generate_trajectory", 4)`. Both `test_stub_module_imports_cleanly` (line 118) and `test_stub_callable_raises_with_correct_pr` (line 127-128) parametrize over this list.
- `tests/test_circumnutation_foundation.py:814-820` — `STUBS_WITH_CONSTANTS_KWARG` is a 5-entry list **NOT including** `synthetic` (the existing stub has positional signature `(L_gz=None, Delta_L=None, ...)` with no `constants=` kwarg, so it was never in this table).
- `tests/test_circumnutation_foundation.py:204-211` — `test_schema_version_is_1_and_constants_version_is_2` asserts `_CONSTANTS_VERSION == 2` and the docstring says "bumped in PR #3".

- [ ] 2.I.1 **Remove** `("synthetic", "generate_trajectory", 4)` from `STUB_MODULES` in `tests/test_circumnutation_foundation.py:34-43`. This drops synthetic from BOTH parametrize tests: `test_stub_module_imports_cleanly` (8 → 7 ids) AND `test_stub_callable_raises_with_correct_pr` (8 → 7 ids). The import-cleanly coverage for `synthetic` is preserved by the new `test_implementation_accepts_constants_kwarg` added in §2.I.2 (which imports the module to call the implementation).

- [ ] 2.I.2 **Add** a NEW `IMPLEMENTATIONS_WITH_CONSTANTS_KWARG` table beneath `STUBS_WITH_CONSTANTS_KWARG` (architecture reviewer I3 + TDD reviewer B1 — note: `STUBS_WITH_CONSTANTS_KWARG` is **unchanged** by this PR; synthetic was never in it):

  ```python
  # tests/test_circumnutation_foundation.py — STUBS_WITH_CONSTANTS_KWARG unchanged:
  STUBS_WITH_CONSTANTS_KWARG = [
      ("temporal_cwt", "compute_scaleogram"),
      ("psi_g", "compute_psi_g"),
      ("midline", "reconstruct"),
      ("spatial_cwt", "compute_scaleogram"),
      ("pipeline", "compute_traits"),
  ]

  # NEW table for the 3 implementation modules that accept constants=:
  IMPLEMENTATIONS_WITH_CONSTANTS_KWARG = [
      ("kinematics", "compute"),
      ("qc", "compute"),
      ("synthetic", "generate_trajectory"),   # added by PR #4
  ]
  ```

  - `test_stub_accepts_constants_kwarg` is **unchanged** (still parametrizes over `STUBS_WITH_CONSTANTS_KWARG` = 5 ids; still asserts `NotImplementedError`).
  - **NEW test** `test_implementation_accepts_constants_kwarg`: parametrizes over `IMPLEMENTATIONS_WITH_CONSTANTS_KWARG` (3 ids); for each `(module_name, callable_name)`, calls the implementation with a minimal valid input plus `constants=ConstantsT()`; asserts NO `NotImplementedError`, NO `TypeError`, and the return is a `pd.DataFrame`. For `synthetic.generate_trajectory` the minimal-valid call is `fn(constants=ConstantsT())` (all other kwargs default). For `kinematics.compute` / `qc.compute` use a minimal valid `trajectory_df` (an existing helper fixture).

- [ ] 2.I.3 `test_module_logger_is_namespaced` already covers `synthetic` per the existing PR #2/#3 extension (`synthetic` is in the contract-module list). No change needed; verify only.

- [ ] 2.I.4 Update `tests/test_circumnutation_foundation.py:204-211` `test_schema_version_is_1_and_constants_version_is_2`:
  - Rename function to `test_schema_version_is_1_and_constants_version_is_3`.
  - Update docstring from "`_CONSTANTS_VERSION` is 2 (bumped in PR #3)" to "`_CONSTANTS_VERSION` is 3 (bumped in PR #4)" (TDD reviewer B2).
  - Change assertion `_CONSTANTS_VERSION == 2` to `_CONSTANTS_VERSION == 3`.

### 2.J — Run the suite (TDD red phase confirmation)

- [ ] 2.J.1 Run `uv run pytest tests/test_circumnutation_synthetic.py` — expect tests to FAIL (`NotImplementedError` from `synthetic.generate_trajectory`, or ImportError on the constants if 3.x not done yet). **TDD red phase confirmed.** Per project convention, do not push this commit alone — section 3 must land in the same push.

## 3. Implementation — `_constants.py` additions + `synthetic.py` (TDD green phase)

- [ ] 3.1 Extend `sleap_roots/circumnutation/_constants.py` per design.md D4:
  - Add 7 new module-level constants (UPPER_SNAKE, no `_DEFAULT_` infix):
    - `SYNTHETIC_T_NUTATION_S: float = 3333.0` — Derr Sept-2025 pilot; preliminary_results §3.4
    - `SYNTHETIC_AMPLITUDE_PX: float = 10.0` — plate 001 detrended peak-to-peak; preliminary_results §4.3
    - `SYNTHETIC_GROWTH_RATE_PX_PER_FRAME: float = 4.29` — plate 001 mean longitudinal step (prelim §4.1)
    - `SYNTHETIC_NOISE_SIGMA_PX: float = 2.0` — plate 001 SG-residual ≈ 1.83; theory.md §8 Layer 1 noise level
    - `SYNTHETIC_CADENCE_S: float = 300.0` — plate 001 cadence (5 min)
    - `SYNTHETIC_N_FRAMES: int = 575` — plate 001 frame count (47.9 hr)
    - `SYNTHETIC_GROWTH_AXIS_ANGLE_RAD: float = math.pi / 2` — image-y-down convention
  - Each gets a docstring with source citation.
  - Add `import math` if not already present.

- [ ] 3.2 Bump `_CONSTANTS_VERSION = 2` → `_CONSTANTS_VERSION = 3` in `_constants.py`. Update the version's docstring with a note about the PR #4 addition.

- [ ] 3.3 Add 7 new fields to `ConstantsT` `@attrs.define` class, with the same defaults pulled from the module-level constants.

- [ ] 3.4 Extend `_default_constants_snapshot()` to include the 7 new keys.

- [ ] 3.5 Implement `sleap_roots/circumnutation/synthetic.py` per design.md D1-D14 + spec Requirement "Synthetic trajectory generator":
  - imports (matches `kinematics.py:71` / `qc.py:66` tier-module precedent per round-2 TDD reviewer NIT-1 — import only `ConstantsT`; access SYNTHETIC_* defaults through the resolved `(constants or ConstantsT()).SYNTHETIC_X` per D13): `inspect`, `logging`, `math`; `from typing import Optional, Union`; `import numpy as np; import pandas as pd`; `from sleap_roots.circumnutation._constants import ConstantsT`; `from sleap_roots.circumnutation._types import ROW_IDENTITY_COLUMNS, REQUIRED_PER_FRAME_COLUMNS`. The SYNTHETIC_* public constants are imported in `tests/test_circumnutation_synthetic.py` (where they ARE load-bearing for analytical predictions in §2.G.4 / §2.G.5), NOT in `synthetic.py` itself. Note: original draft imported the 7 SYNTHETIC_* names directly — issue-alignment reviewer N3 caught a leading-underscore typo, AND round-2 TDD reviewer NIT-1 noted the over-import diverges from tier-module precedent for no functional gain.
  - module-level `logger = logging.getLogger(__name__)` (already exists from stub; verify)
  - helper `_validate_params(params: dict) -> None`: implements D8 validation table — bool rejection, finiteness, sign, type acceptance via `isinstance(v, (int, np.integer))` etc. Raises `ValueError` / `TypeError` with field-named messages. Maps each parametrize id in test 2.F.
  - helper `_resolve_constants_and_kwargs(...) -> dict`: implements D13 resolution-order — `kwarg if kwarg is not None else (constants or ConstantsT()).SYNTHETIC_X`. Returns a dict of resolved values for the 7 ConstantsT-overridable params.
  - helper `_compute_trajectory(resolved: dict) -> tuple[np.ndarray, np.ndarray]`: implements D1 closed-form math — apex propagation + lateral nutation + iid noise with per-axis σ = `noise_sigma_px / √2` (per design.md D1 + scientific-rigor B1). Short-circuits RNG path when `noise_sigma_px == 0.0` exactly (D11).
  - helper `_build_dataframe(tip_x, tip_y, identity_kwargs) -> pd.DataFrame`: assembles the 11-column output per D3 dtype contract using explicit `pd.Series(..., dtype=...)` constructions; `None` for `genotype/treatment` becomes `np.nan` in `object`-dtype column (NOT literal `"None"`).
  - public `generate_trajectory(*, ...) -> pd.DataFrame` per signature in design.md D9/D13 (all kwargs, `Optional[float] = None` for the 7 ConstantsT-overridable parameters): validate → resolve → compute → build → return.
  - Google-style docstring per pydocstyle, naming every parameter and the returned schema. Include "Closed-form realization" / "Determinism contract" / "Rivière correspondence (degenerate at tip level)" / "Pure-pixel + per-axis noise (σ = noise_sigma_px / √2)" / "Layer-1 caveat: noise_sigma_px = 0 is for closed-form correctness tests only — use noise_sigma_px > 0 for Layer-1 validation" subsections.
  - **Docstring SHALL note divergence from `CircumnutationInputs.cadence_s` coercion** (architecture I2): "Unlike `CircumnutationInputs.cadence_s` which has an attrs converter for `cadence_s = '300'` (string → float), `generate_trajectory` does NOT coerce string inputs. Pass `float(...)` explicitly if your inputs come from a config file."
  - **Docstring SHALL include the SeedSequence.spawn idiom for multi-track plates** (scientific-rigor N4 + design.md D7): example code block showing the `seed_seq.spawn(N)` pattern.
  - **Replace** the existing `NotImplementedError("PR #4 — see docs/circumnutation/roadmap.md")` body with the working implementation.

- [ ] 3.6 Run `uv run pytest tests/test_circumnutation_synthetic.py -v` — all sections green.

- [ ] 3.7 Capture the canary first-3-elements for `random_state=0` (test 2.B.1) AND verify the ψ_g sign discriminator (design.md R7): write a one-off script to `c:\vaults\sleap-roots\circumnutation\scripts\capture_synthetic_canary.py` (NOT `uv run python -c "..."` inline — PowerShell quoting), then:
  1. Print `generate_trajectory(random_state=0, ...)`'s `tip_x[:3]` and `tip_y[:3]`; lock the values into test 2.B.1 as the canary.
  2. **ψ_g sign verification** (R7 mitigation): print `np.mean(np.diff(_geometry.compute_psi_g(synth_x, synth_y)))` for both `handedness=+1` (expected `> 0`) and `handedness=-1` (expected `< 0`) at default plate-001 parameters. If `|mean(diff)| < 1e-3` for either handedness, OR if the sign is unstable across seeds `random_state ∈ {0, 1, 2, 3, 4}`, switch the §2.E.1 / §2.E.2 discriminator from `mean(diff(psi_g))` to one of: (a) `helix_signed_area = 0.5 * sum(tip_x[i] * tip_y[i+1] - tip_x[i+1] * tip_y[i])` (signed Shoelace area; positive for math-CCW), or (b) `psi_g[1] - psi_g[0]` (first-frame discriminator). Record the empirical mean-diff values AND the chosen discriminator in the script's stdout for traceability. The §2.E.3 curl-sign cross-check is independent and remains the canonical check; only §2.E.1 / §2.E.2 need re-targeting.

- [ ] 3.8 Run `uv run pytest tests/ -x` — full suite green (no regression in foundation, Tier 0, QC, tracked-tip-pipeline, etc.).

## 4. Docs and changelog

- [ ] 4.1 Add `docs/changelog.md` entry under `[Unreleased]` / `### Added` (lowercase per repo convention):

  > circumnutation: synthetic trajectory generator (`sleap_roots.circumnutation.synthetic.generate_trajectory`) for Layer-1 validation; closed-form realization of Rivière 2022 Eq. 4 with user-facing aggregate parameters (`amplitude_px`, `T_nutation_s`, `growth_rate_px_per_frame`, `noise_sigma_px`); deterministic via `random_state` (CC-6); emits pure-pixel `tip_x`/`tip_y` matching `CircumnutationInputs` schema. 7 new defaults in `_constants.py` + `ConstantsT`; `_CONSTANTS_VERSION` 2 → 3. See `docs/circumnutation/theory.md` §4 + §8.

- [ ] 4.2 No `theory.md` or `roadmap.md` edits in this PR. The existing CC-6 wording on roadmap.md ("Synthetic generator (PR #4) accepts an explicit `random_state` … and propagates it. Tests assert determinism: same input → identical output across two runs, including in CI on different OSs.") already describes the PR #4 contract verbatim. Roadmap row #4 checkbox `⬜ → ✅` happens post-merge in `/cleanup-merged`.

## 5. Verify (pre-PR-open gates)

- [ ] 5.1 Run `uv run pytest tests/test_circumnutation_synthetic.py --cov=sleap_roots.circumnutation.synthetic --cov-report=term-missing`. Target: **100% coverage** on `synthetic.py` including `logger.debug` branches (round-2 TDD IMP-1: the original "Acceptable uncovered branches: `logger.debug` calls" carve-out is REMOVED — debug-level emissions SHALL be exercised by §2.A.11b explicit `caplog` assertions per the call-site-by-call-site rule).
- [ ] 5.2 Run `uv run pytest tests/ -x` — full suite green.
- [ ] 5.3 Run `uv run pytest tests/ --cov=sleap_roots --cov-fail-under=84` — confirm project-wide coverage holds at 84%+.
- [ ] 5.4 Run `uv lock --check` — no dependency change in this PR.
- [ ] 5.5 Run `uv run black --check sleap_roots tests` — passes.
- [ ] 5.6 Run `uv run pydocstyle --convention=google sleap_roots/` — passes.
- [ ] 5.7 Run `uv run mkdocs build` — passes; `synthetic.generate_trajectory` doc page renders.
- [ ] 5.8 Run `openspec validate add-circumnutation-synthetic-generator --strict` — valid.
- [ ] 5.9 Parametrize-id sanity check (corrected per TDD reviewer B1):
  - `test_stub_callable_raises_with_correct_pr` collects **7** ids (was 8 — synthetic removed from STUB_MODULES).
  - `test_stub_module_imports_cleanly` collects **7** ids (was 8 — same STUB_MODULES drop). Synthetic's import coverage is now via `test_implementation_accepts_constants_kwarg`.
  - `test_stub_accepts_constants_kwarg` collects **5** ids (unchanged — `STUBS_WITH_CONSTANTS_KWARG` unchanged; synthetic was never in this list).
  - The NEW `test_implementation_accepts_constants_kwarg` collects **3** ids (kinematics, qc, synthetic).
  - Net foundation-test count change: **+1 test, ±0 parametrize-id loss** (the new test adds 3 ids; STUB_MODULES drops 2 ids across two parametrized tests = -2; net +1). Mention the expected pytest collection deltas in the PR description.

## 6. PR open

- [ ] 6.1 Draft GitHub issue body to `c:\vaults\sleap-roots\circumnutation\github_issues\issue_add-circumnutation-synthetic-generator.md` referencing epic #197, OpenSpec change-id, theory.md §4 + §8, prelim §1 / §3.4 / §4.1 / §4.3, CC-2/CC-3/CC-6/CC-9. Show Elizabeth before posting (do NOT post unilaterally).
- [ ] 6.2 Open PR with title "feat(circumnutation): synthetic trajectory generator (#<sub-issue>)" and a body cross-linking the epic, sub-issue, OpenSpec change-id, and design.md. Labels: `enhancement`, `circumnutation`, `multi-pr`. Body matches `.claude/commands/pr-description.md` template.
- [ ] 6.3 Push the branch. Confirm CI is green on Ubuntu / Windows / macOS at Python 3.11.

## 7. Post-merge

- [ ] 7.1 Run `/cleanup-merged` which delegates to `/openspec:archive`. The archive folds the ADDED + MODIFIED requirements into `openspec/specs/circumnutation/spec.md` and moves this change folder to `openspec/changes/archive/YYYY-MM-DD-add-circumnutation-synthetic-generator/`.
- [ ] 7.2 Update `docs/circumnutation/roadmap.md` row PR #4 → ✅ with the GitHub issue and PR numbers.
- [ ] 7.3 Update the Notion task page `https://www.notion.so/Circumnutation-work-sleap-roots-3494a67a7667818db2aeee80d76efdd7` with a new "Status update — YYYY-MM-DD" section (show Elizabeth the draft first).
