# Tasks for add-circumnutation-qc-tier

TDD-ordered. Tests precede implementation per `superpowers:test-driven-development`. **Do not push commits from section 2 (red phase — tests only) without sections 3 and 4 (implementations) in the same push** — the new test file imports `compute_d2_residual_xy` and `compute_msd_residual_xy` from `_noise.py` which section 3 adds, and asserts `qc.compute` returns a DataFrame which section 4 implements. The suite is expected to be red between 2.x and 4.x; PR is one logical unit.

## 1. Pre-flight (no code changes; just confirms ground-truth)

- [ ] 1.1 Confirm `tests/data/circumnutation_nipponbare_plate_001/plate_001_greyscale.tracked_proofread.slp` is present (committed via Git LFS in PR #2). Run `uv run python -c "import sleap_io as sio; labels = sio.load_slp('tests/data/circumnutation_nipponbare_plate_001/plate_001_greyscale.tracked_proofread.slp'); print(len(labels), labels.tracks)"` → expect 575 frames, 6 tracks.
- [ ] 1.2 Run baseline: `uv run pytest tests/ -x` — confirm green BEFORE any changes. Foundation + Tier 0 tests must be passing at HEAD.
- [ ] 1.3 Run baseline coverage: `uv run pytest tests/ --cov=sleap_roots --cov-report=term --cov-fail-under=84` — confirm we're starting at the 84% baseline.

## 2. Tests (write before implementation — TDD red phase)

- [ ] 2.1 Create `tests/test_circumnutation_qc.py` with module docstring referencing the spec deltas + theory.md §7.6 + design.md sections D1-D9.

### 2.A — Trait set schema and structural tests (spec Requirement: QC tier per-track quality traits)

- [ ] 2.A.1 Test `compute_returns_per_plant_dataframe`: build a 6-track trajectory_df via the existing `valid_trajectory_df` pattern (from `test_circumnutation_foundation.py`), call `qc.compute(df)`, assert return is `pd.DataFrame` with exactly 6 rows.
- [ ] 2.A.2 Test `output_columns_match_spec`: assert columns = the 8 row-identity columns (first, in declared order) + the 11 new columns in the documented order: `sg_residual_xy`, `d2_noise_xy`, `msd_noise_xy`, `sg_d2_agreement`, `sg_msd_agreement`, `d2_msd_agreement`, `frac_outlier_steps`, `worst_step_ratio`, `growth_axis_unreliable`, `track_is_clean`, `qc_failure_reason`. Hard-code the expected column order list at module level.
- [ ] 2.A.3 Test `unit_columns_match_vocabulary`: assert every emitted trait column's unit string (from the module-level `_QC_TRAIT_UNITS` dict imported from `qc.py`) is in `PIPELINE_UNIT_VOCABULARY`. Cover: `px` for the 3 estimators, `—` for the 5 dimensionless, `bool` for `growth_axis_unreliable` and `track_is_clean`, `string` for `qc_failure_reason`. **Add explicit negative assertions** (per spec scenario "Output column units are within PIPELINE_UNIT_VOCABULARY"): `assert not any('mm' in u for u in units.values())` AND `assert not any('px/hr' in u for u in units.values())`.
- [ ] 2.A.4 Test `track_id_is_integer_and_plant_id_equals_track_id`: same invariants as the foundation/Tier-0 tests, just on the QC output DataFrame.
- [ ] 2.A.5 Test `output_sort_order_is_numeric`: identity columns sorted by the 5-tuple, `track_id` numeric (not lexicographic).
- [ ] 2.A.6 Test `timepoint_column_preserved`: with input rows carrying `timepoint = "T0"`, confirm the output DataFrame still has `timepoint == "T0"` for every row (regression guard against the `_IDENTITY_5_TUPLE` re-selection bug — see spec Requirement: QC tier per-track quality traits + design.md D7 merge step).
- [ ] 2.A.7 Test `invalid_trajectory_df_raises_valueerror`: parametrize over non-DataFrame types (`None`, `[1, 2, 3]`, `{"frame": []}`, `np.array([1.0])`) — each MUST raise `ValueError` whose message mentions "DataFrame". Also assert `qc.compute(df_missing_tip_x)` raises `ValueError` whose message names `tip_x` explicitly. Maps to Requirement "QC tier input-validation boundary".

- [ ] 2.A.8 Test `qc_compute_no_longer_raises_not_implemented_error`: mirror PR #2's precedent for `kinematics.compute`. Call `qc.compute(valid_trajectory_df)` with a 1-row valid input; assert return is a `pd.DataFrame` and no `NotImplementedError` is raised. Maps to MODIFIED Package layout scenario "`qc.compute` no longer raises NotImplementedError".

- [ ] 2.A.9 Test `duplicate_track_id_frame_rows_do_not_raise`: build `trajectory_df` where the same `(track_id=0, frame=5)` 2-tuple appears in two rows (simulating an upstream join error); assert `qc.compute(df)` returns without raising. Trait values may be non-finite (inf or NaN) per the spec's documented non-goal. Maps to spec scenario "Duplicate `(track_id, frame)` rows do not raise".

- [ ] 2.A.10 Test `inf_in_tip_x_propagates_without_raising`: build `trajectory_df` with one row having `tip_x = float('inf')`; assert `qc.compute(df)` returns without raising. The track's trait values may be `inf` or `NaN` (propagation-dependent). Maps to spec scenario "±inf in tip_x propagates without raising".

### 2.B — Synthetic exact-value tests (spec Requirement: QC tier per-track quality traits / scenarios)

- [ ] 2.B.1 Test `clean_straight_line_track`: 100-frame track with `tip_x = frame * 1.0`, `tip_y = 0.0`. Expected:
  - `sg_residual_xy ≈ 0.0`, `d2_noise_xy ≈ 0.0`, `msd_noise_xy ≈ 0.0` (smooth signal; SG of degree 3 fits linear exactly; d2 of linear is identically zero)
  - `frac_outlier_steps == 0.0`, `worst_step_ratio ≈ 1.0`
  - `growth_axis_unreliable == False`
  - **DISCOVERED-DURING-IMPL (TDD refinement)**: the initial spec called for "all 3 `*_agreement_high` clauses fire" on the rationale `0/0 = NaN → not (NaN < threshold) = True`. In practice, the SG filter on a perfectly linear signal produces residuals at floating-point precision (~1e-16, not exactly 0), so `_pairwise_agreement` may compute a finite-but-meaningless ratio (e.g. 4.81) for some pairs rather than returning NaN. The qc.py "Caveat: noiseless input" docstring documents this. **Revised assertion**: assert `track_is_clean=False` AND **at least one** `*_agreement_high` clause fires (not specifying which). Synthetic-noise tracks (test 2.B.2 with σ > 0) are the canonical clean-track case; the noiseless case is documented edge behavior, not a positive test of the clean-path.

- [ ] 2.B.2 Test `clean_track_with_uniform_noise`: 100-frame track with `tip_x = np.arange(100) + np.random.default_rng(0).normal(0, 1, 100)`, `tip_y = np.random.default_rng(1).normal(0, 1, 100)` (linear growth + i.i.d. unit-σ noise on both x and y). Expected:
  - `sg_residual_xy ∈ [1.0, 1.6]`, `d2_noise_xy ∈ [1.2, 1.8]`, `msd_noise_xy ∈ [1.0, 2.0]` (per the per-helper spec scenarios)
  - 3 pairwise agreements all `< 1.5` (well under threshold for clean data)
  - `frac_outlier_steps < 0.05`, `worst_step_ratio < 5`
  - `growth_axis_unreliable == False` (`D ≈ 99 ≫ K · noise`)
  - `track_is_clean == True`, `qc_failure_reason == ""`

- [ ] 2.B.3 Test `pure_noise_track`: 100-frame track with `tip_x, tip_y = np.random.default_rng(0).normal(0, 1, size=(2, 100))`. Expected:
  - `growth_axis_unreliable == True` (small D vs noise)
  - `track_is_clean == False`
  - `qc_failure_reason` contains `"growth_axis_unreliable"`

- [ ] 2.B.4 Test `short_track_triggers_gate`: 3-frame track with non-zero displacement (e.g., `tip_x = [0.0, 1.0, 2.0]`, `tip_y = [0.0, 0.0, 0.0]`). Expected:
  - 8 numeric traits + 3 pairwise agreements all NaN
  - `growth_axis_unreliable` value matches what Tier 0's logic would produce. For n=3 the algorithm enters the full path (n ≥ 2, NOT the n<2 short-circuit), then `sg = _noise.compute_sg_residual_xy(...)` returns NaN because `len = 3 < SG_WINDOW_SHORT = 5` (helper's own short-input branch). Thus the gate evaluates `(D == 0.0) OR (not math.isnan(NaN) and ...)` = `(D == 0.0) OR False` = `D == 0.0`. For this non-zero-displacement track, `growth_axis_unreliable == False`. (Verifying the equality path matches `kinematics.py:218`.)
  - `track_is_clean == False`
  - `qc_failure_reason == "qc_inputs_insufficient"` (literally, NOT comma-concatenated with other clauses)

- [ ] 2.B.5 Test `single_frame_track`: 1-frame track. Expected:
  - All 8 numeric traits + 3 agreements NaN
  - `growth_axis_unreliable == False` (n < 2 path: D=NaN, both gate clauses False)
  - `track_is_clean == False`
  - `qc_failure_reason == "qc_inputs_insufficient"`
  - No exception raised

- [ ] 2.B.6 Test `closed_loop_track`: 10-frame track where `xy[-1] == xy[0]` (D=0). Expected:
  - `growth_axis_unreliable == True` (D=0 disjunct fires)
  - `track_is_clean == False`
  - `qc_failure_reason` contains `"growth_axis_unreliable"`

- [ ] 2.B.7 Test `stationary_track`: 100-frame track with `tip_x == 5.0` and `tip_y == 3.0` constant (all step magnitudes zero, `median == 0`). Expected:
  - `frac_outlier_steps` is NaN, `worst_step_ratio` is NaN
  - `track_is_clean == False`
  - `qc_failure_reason` contains BOTH `"frac_outlier_steps_high"` AND `"worst_step_ratio_high"` (per spec scenario "Stationary track fires both outlier clauses")
  - `growth_axis_unreliable == True` (D=0)

- [ ] 2.B.8 Test `outlier_step_fires_worst_step_ratio_clause`: 100-frame clean track + a single injected `tip_x[50] = 1000.0` outlier. Expected:
  - `worst_step_ratio > 5` (the injected step has magnitude ~1000 vs median ~1)
  - `qc_failure_reason` contains `"worst_step_ratio_high"`
  - `track_is_clean == False`

- [ ] 2.B.9 Test `frac_outlier_steps_fires_when_many_outliers`: construct a track with > 5% of steps > 2× median. Expected:
  - `frac_outlier_steps > 0.05`
  - `qc_failure_reason` contains `"frac_outlier_steps_high"`
  - `track_is_clean == False`

- [ ] 2.B.10 Test `nan_rows_dropped_before_diff`: 100-frame clean noisy track (per 2.B.2) with 10 random rows (`np.random.default_rng(2).choice(100, 10, replace=False)`) having `tip_x = NaN`. Expected: trait values match the no-NaN-row case within float tolerance. NaN-then-sort ordering invariant validated.

- [ ] 2.B.11 Test `n_5_boundary_msd_returns_nan` (**load-bearing D8 case**): 5-frame straight-line-with-noise track (`tip_x = np.arange(5) + rng.normal(0, 1, 5)`, `tip_y = rng.normal(0, 1, 5)`). At exactly `len = SG_WINDOW_SHORT = 5`, the short-track gate does NOT fire (gate is `len < SG_WINDOW_SHORT`, strict less-than). Expected:
  - `sg_residual_xy` finite (SG can run at len = window = 5)
  - `d2_noise_xy` finite (d2 only needs len ≥ 3)
  - `msd_noise_xy == NaN` (MSD needs len ≥ window + lag = 6; helper returns NaN + DEBUG log)
  - `sg_d2_agreement` finite
  - `sg_msd_agreement == NaN`, `d2_msd_agreement == NaN`
  - `track_is_clean == False`
  - `qc_failure_reason` contains BOTH `"sg_msd_agreement_high"` AND `"d2_msd_agreement_high"` (per design.md D8 final paragraph: "MSD may legitimately return NaN for a track of length exactly 5...this is expected: a 5-frame track has too few samples for the MSD method")
  - DOES NOT contain `"qc_inputs_insufficient"` (the short-track gate doesn't fire at n=5)

- [ ] 2.B.12 Test `n_6_boundary_all_estimators_finite`: 6-frame clean noisy track. At `len = SG_WINDOW_SHORT + lag = 6`, MSD's minimum is satisfied. Expected: all three noise estimators finite, all three pairwise agreements finite, no `*_high` clauses fire from estimator-NaN propagation (other clauses may or may not fire based on the actual values; assert agreements specifically are non-NaN).

- [ ] 2.B.13 Test `empty_after_dropna_yields_qc_inputs_insufficient`: build `trajectory_df` for a single track where every row has `tip_x = NaN`. After dropna, `len(subset) == 0`. Expected:
  - the returned DataFrame has 1 row for that track
  - 8 numeric traits + 3 pairwise agreements all NaN
  - `growth_axis_unreliable == False` (matches Tier 0's `_emit_nan_row` for the same case)
  - `track_is_clean == False`
  - `qc_failure_reason == "qc_inputs_insufficient"`
  - No exception raised

### 2.C — Per-helper synthetic tests (spec Requirement: Tier 0 helper modules)

- [ ] 2.C.1 Test `compute_sg_residual_xy_unchanged_by_pr3`: assert PR #2's existing SG-residual scenarios still pass (regression — confirm we didn't break the existing helper).

- [ ] 2.C.2 Test `compute_d2_residual_xy_linear_signal_zero`: per spec scenario "_noise.compute_d2_residual_xy returns zero for a linear signal" — `x = np.linspace(0, 99, 100)`, `y = np.zeros(100)` → return `0.0` within IEEE float tolerance.

- [ ] 2.C.3 Test `compute_d2_residual_xy_noisy_signal_recovers_sigma`: per spec scenario "_noise.compute_d2_residual_xy recovers approximate σ on noisy data" — `np.random.default_rng(0).normal(0, 1, size=(2, 1000))` on linear x signal → return within `[1.2, 1.8]`.

- [ ] 2.C.4 Test `compute_d2_residual_xy_short_input_returns_nan_with_debug_log`: pass len-2 arrays → return `np.nan` AND `caplog.at_level(logging.DEBUG)` captures the short-input message.

- [ ] 2.C.5 Test `compute_msd_residual_xy_smooth_signal_near_zero`: per spec scenario "_noise.compute_msd_residual_xy returns approximately zero for a smooth signal" — linear x, zero y → return `≤ 1e-6`.

- [ ] 2.C.6 Test `compute_msd_residual_xy_noisy_signal_recovers_sigma`: per spec scenario "_noise.compute_msd_residual_xy recovers approximate σ on noisy data" — `rng(0).normal(0, 1, (2, 1000))` on linear x → return within `[1.0, 2.0]`. **Factor-of-4 verification**: assert the return is in this range and NOT in `[2.0, 4.0]` (which would happen if the implementation used factor-of-2 instead of factor-of-4 by mistake).

- [ ] 2.C.7 Test `compute_msd_residual_xy_short_input_returns_nan_with_debug_log`: pass len-5 arrays with `window=5, lag=1` (so `len < window+lag = 6`) → return `np.nan` AND DEBUG log emitted.

### 2.D — Pairwise agreement tests (spec Requirement: QC tier per-track quality traits)

- [ ] 2.D.1 Test `pairwise_agreement_when_estimators_agree`: synthetic noisy track where all three estimators converge → all 3 agreements near 1.0 (∈ [1.0, 1.5]).

- [ ] 2.D.2 Test `pairwise_agreement_when_one_estimator_is_nan`: track where SG returns valid but d2 returns NaN (impossible in practice but constructible by mocking). Use `monkeypatch.setattr("sleap_roots.circumnutation._noise.compute_d2_residual_xy", lambda *a, **k: float("nan"))` to force d2 to return NaN regardless of input (matches the monkeypatch precedent in `test_circumnutation_kinematics.py:488-562`). With a clean noisy track input, expect: `sg_d2_agreement` is NaN; `d2_msd_agreement` is NaN; both corresponding `*_high` clauses fire; `qc_failure_reason` contains both.

### 2.E — `track_is_clean` + `qc_failure_reason` composition (spec Requirement: QC tier track_is_clean and qc_failure_reason composition)

- [ ] 2.E.1 Test `track_is_clean_all_clauses_pass`: clean noisy synthetic (per 2.B.2) → `track_is_clean == True`, `qc_failure_reason == ""`.

- [ ] 2.E.2 Test `qc_failure_reason_single_clause`: track tuned to fail exactly one clause (e.g., one extreme outlier step → only `worst_step_ratio_high` fires) → `qc_failure_reason == "worst_step_ratio_high"`.

- [ ] 2.E.3 Test `qc_failure_reason_multi_clause_stable_order`: track failing 3 specific clauses (growth_axis_unreliable + sg_d2_agreement_high + frac_outlier_steps_high) → `qc_failure_reason == "growth_axis_unreliable, sg_d2_agreement_high, frac_outlier_steps_high"` (clauses in `_FAILURE_CLAUSE_ORDER` order).

- [ ] 2.E.4 Test `qc_failure_reason_short_track_sentinel`: 3-frame track → `qc_failure_reason == "qc_inputs_insufficient"` LITERALLY (not "qc_inputs_insufficient, growth_axis_unreliable").

- [ ] 2.E.5 Test `_FAILURE_CLAUSE_ORDER_tuple_is_canonical`: import `qc._FAILURE_CLAUSE_ORDER` directly, assert it equals the documented tuple (7 entries in the specified order).

- [ ] 2.E.6 Test `track_is_clean_excludes_growth_axis_unreliable`: track where ONLY `growth_axis_unreliable == True` and all other clauses pass → `track_is_clean == False`, `qc_failure_reason == "growth_axis_unreliable"`. Confirms the D5 design decision (growth_axis_unreliable is in the composite).

### 2.F — Growth-axis equality contract (spec Requirement: QC tier growth_axis_unreliable equality with Tier 0)

- [ ] 2.F.1 Test `growth_axis_unreliable_equality_synthetic`: build 6 synthetic tracks spanning the full gate-behavior space (clean / pure-noise / closed-loop / single-frame / 3-frame-short / 10-frame-medium). Assert `kinematics.compute(df)["growth_axis_unreliable"].equals(qc.compute(df)["growth_axis_unreliable"])` AND `kinematics_result["growth_axis_unreliable"].dtype == qc_result["growth_axis_unreliable"].dtype == np.dtype("bool")`.

- [ ] 2.F.2 Test `growth_axis_unreliable_equality_under_int_dtype`: same as 2.F.1 but with `tip_x`, `tip_y` cast to int via `astype(int)`. Confirms the `to_numpy(dtype=float)` cast in QC preserves the equality contract.

- [ ] 2.F.3 Test `growth_axis_unreliable_equality_under_float32_dtype`: same as 2.F.1 but with `tip_x`, `tip_y` cast to float32. Same conclusion.

- [ ] 2.F.4 Test `growth_axis_unreliable_dtype_is_bool_no_nan`: assert `qc.compute(df)["growth_axis_unreliable"].dtype == np.dtype("bool")` AND `~qc.compute(df)["growth_axis_unreliable"].isna().any()`. Same for `track_is_clean`.

### 2.G — ConstantsT override parametric (spec Requirements: Module-level constants, QC tier per-track quality traits)

- [ ] 2.G.1 Test parametrized over `(constant_name, override_value, expected_clause_in_reason)` for each of the 6 threshold constants. For each row, construct a synthetic track that puts the underlying trait at a value that crosses the default threshold, then call once with default constants (asserts clause fires) and once with `ConstantsT(<constant>=loose_value)` (asserts clause does NOT fire). The parametrize table SHALL include explicitly the spec scenario "ConstantsT override changes per-clause thresholds" pair: a track tuned so `sg_d2_agreement = 1.7`; default `SG_D2_AGREEMENT_MAX=1.5` fires `sg_d2_agreement_high`; override `ConstantsT(SG_D2_AGREEMENT_MAX=2.0)` suppresses it.
- [ ] 2.G.2 Test `constants_snapshot_contains_4_new_keys`: call `_default_constants_snapshot()` and assert `FRAC_OUTLIER_STEPS_MAX`, `WORST_STEP_RATIO_MAX`, `SG_MSD_AGREEMENT_MAX`, `D2_MSD_AGREEMENT_MAX` are all present.
- [ ] 2.G.3 Test `_CONSTANTS_VERSION_is_2`: assert `_constants._CONSTANTS_VERSION == 2`.
- [ ] 2.G.4 Test `qc_compute_accepts_constants_kwarg`: assert `qc.compute(valid_trajectory_df, constants=ConstantsT())` returns a `pd.DataFrame` without raising `TypeError` or `NotImplementedError`. Closes the coverage gap left by §2.I.2 removing `qc` from `STUBS_WITH_CONSTANTS_KWARG` in the foundation test (after that removal, `test_stub_accepts_constants_kwarg` no longer covers `qc.compute`, but `qc.compute` is now an *implementation* that MUST accept the kwarg).

### 2.H — KitaakeX smoke + Nipponbare reference value test

- [ ] 2.H.1 Test `kitaakex_smoke`: load `tests/data/circumnutation_plate/plate_001_greyscale.tracked.slp` via `TrackedTipPipeline`; enrich (`plate_id="plate_001"`, `plant_id = track_id`, `genotype="KitaakeX"`, `treatment="MOCK"`); call `qc.compute(enriched_df)`. Assert:
  - return is a DataFrame with exactly 6 rows + 19 columns
  - all 3 noise estimators finite for all 6 tracks
  - all 3 pairwise agreements finite
  - explicit units sidecar validation: combine `default_units_for_template(template)` with `_QC_TRAIT_UNITS`; assert `all(u in PIPELINE_UNIT_VOCABULARY for u in units.values())`; round-trip via `_io.write_per_plant_csv` to `tmp_path` + read-back the sidecar JSON via `_io.read_units_sidecar` AND explicitly assert that all 11 QC trait column names are keys in the read-back dict AND their values match `_QC_TRAIT_UNITS` (byte-for-byte unchanged, including non-ASCII `—`).

- [ ] 2.H.2 Test `nipponbare_reference_values`: load `tests/data/circumnutation_nipponbare_plate_001/plate_001_greyscale.tracked_proofread.slp`; enrich (`plate_id="plate_001"`, `plant_id = track_id`, `genotype="Nipponbare"`, `treatment="MOCK"`); call `qc.compute(enriched_df)`. Assert per-track median values fall within tolerance ranges locked during impl (section 4.4 below):

  **Important — anchor on per-track medians, NOT median-of-medians or quotient-of-medians.** Prelim §4.2 reports `sg = 1.83 px median across 6 tracks` and `d2 = 2.67 px median across 6 tracks` — these are the medians OF the per-track median noise estimates. The §7.6-quoted `sg_d2_agreement ≈ 1.46` is the *quotient of the two reported medians* (`2.67 / 1.83 = 1.459`), NOT the median of the per-track agreement values. This test asserts on the per-track values (each track gets its own sg/d2/msd estimate) and computes statistics from those — DO NOT fall into PR #2's median-of-means trap. Concrete tolerances locked in section 4.4 after one calibration run.

  - `median(per_track_sg_residual_xy) ∈ [TODO_LOWER, TODO_UPPER]` — captured in section 4.4; expected anchor ≈ 1.83; ±20% tolerance (single-plate population)
  - `median(per_track_d2_noise_xy) ∈ [TODO_LOWER, TODO_UPPER]` — expected anchor ≈ 2.67; ±20%
  - `median(per_track_msd_noise_xy) ∈ [TODO_LOWER, TODO_UPPER]` — no prelim anchor; captured at impl-time; ±20% post-capture
  - `median(per_track_sg_d2_agreement) ∈ [TODO_LOWER, TODO_UPPER]` — expected anchor near 1.46 but with looser tolerance because this is median-of-quotients (could differ by 10-20% from §7.6's quotient-of-medians value); ±25%
  - All 6 tracks: `growth_axis_unreliable == False` (Nipponbare is a healthy plate)
  - At least 5 of 6 tracks: `track_is_clean == True` (one outlier tolerated for population effects)
  - Equality contract — assert ALL of:
    - `qc_result["growth_axis_unreliable"].dtype == np.dtype("bool")` (matches spec scenario "Equality on Nipponbare fixture")
    - `kinematics_result["growth_axis_unreliable"].dtype == np.dtype("bool")`
    - `(kinematics_result["growth_axis_unreliable"] == qc_result["growth_axis_unreliable"]).all()` is True (element-wise equal)

### 2.I — Foundation test migration (per spec Requirement: Package layout MODIFIED)

- [ ] 2.I.1 Remove `("qc", "compute", 3)` from `STUB_MODULES` in `tests/test_circumnutation_foundation.py` (line ~32-42). Parametrize-id count drops 9 → 8.
- [ ] 2.I.2 Remove `("qc", "compute")` from `STUBS_WITH_CONSTANTS_KWARG` in `tests/test_circumnutation_foundation.py` (line ~810-817). Parametrize-id count drops 6 → 5.
- [ ] 2.I.3 Extend `test_module_logger_is_namespaced` to include `qc` as an implementation module (so the logger-namespace assertion still covers it).
- [ ] 2.I.4 Update `test_schema_and_constants_versions_are_integers_equal_to_one` (line ~203-210): rename to `test_schema_version_is_1_and_constants_version_is_2`; change assertion `_CONSTANTS_VERSION == 1` to `_CONSTANTS_VERSION == 2`.

### 2.J — Run the suite (TDD red phase confirmation)

- [ ] 2.J.1 Run `uv run pytest tests/test_circumnutation_qc.py` — expect tests to FAIL (ImportError on `compute_d2_residual_xy` / `compute_msd_residual_xy`, or `NotImplementedError` from `qc.compute`). **TDD red phase confirmed.** Per project convention, do not push this commit alone — sections 3 and 4 must land in the same push.

## 3. Implementation — `_noise.py` extensions and `_constants.py` additions (TDD green phase, part 1)

- [ ] 3.1 Extend `sleap_roots/circumnutation/_noise.py` with two new public functions:
  - `compute_d2_residual_xy(x: np.ndarray, y: np.ndarray) -> float`: compute `delta2_x = x[2:] - 2*x[1:-1] + x[:-2]` and similarly for `y`; return `sqrt(std(delta2_x)**2 + std(delta2_y)**2) / sqrt(6)`. Handle `len(x) < 3` per the spec scenario: return `np.nan` and `logger.debug(...)` with the short-input case named. Google-style docstring referencing theory.md §7.6 + prelim §3.3.
  - `compute_msd_residual_xy(x, y, window, degree, lag=1) -> float`: SG-detrend x and y via `savgol_filter`; compute residuals; compute 2D MSD at the given `lag` as `mean((x_res[lag:] - x_res[:-lag])**2 + (y_res[lag:] - y_res[:-lag])**2)`; return `sqrt(msd / 4.0)`. **The factor of 4 (NOT 2) is load-bearing — see design.md D8 and theory.md §7.6.** Handle `len(x) < window + lag` per the spec scenario: return `np.nan` and `logger.debug(...)`. Google-style docstring referencing Michalet 2010.
- [ ] 3.2 Add 4 new constants to `sleap_roots/circumnutation/_constants.py`:
  - `FRAC_OUTLIER_STEPS_MAX: float = 0.05` — theory.md §7.6 clause threshold
  - `WORST_STEP_RATIO_MAX: float = 5` — theory.md §7.6 clause threshold
  - `SG_MSD_AGREEMENT_MAX: float = 1.5` — CC-10 (inherited from SG_D2_AGREEMENT_MAX)
  - `D2_MSD_AGREEMENT_MAX: float = 1.5` — CC-10 (inherited from SG_D2_AGREEMENT_MAX)
  - Each gets a docstring with source citation
- [ ] 3.3 Bump `_CONSTANTS_VERSION = 1` → `_CONSTANTS_VERSION = 2` in `_constants.py`. Update the version's docstring with a note about the PR #3 addition.
- [ ] 3.4 Add 4 new fields to `ConstantsT` `@attrs.define` class, with the same defaults pulled from the module-level constants.
- [ ] 3.5 Extend `_default_constants_snapshot()` to include the 4 new keys.

## 4. Implementation — `qc.py` (TDD green phase, part 2)

- [ ] 4.1 Implement `sleap_roots/circumnutation/qc.py` per design.md D7 (full algorithm) and the spec Requirement "QC tier per-track quality traits":
  - imports: `logging`, `math`, `numpy`, `pandas`, `_constants`, `_io._build_per_plant_template_from_df`, `_io._IDENTITY_5_TUPLE`, `_noise`, `_types.ROW_IDENTITY_COLUMNS`, `_types._validate_trajectory_df`
  - module-level `logger = logging.getLogger(__name__)` (already exists from stub; verify)
  - module-level `_QC_TRAIT_COLUMNS: tuple = (...)` — declared order of the 11 new columns
  - module-level `_QC_TRAIT_UNITS: dict = {...}` — units mapping for the 11 new columns
  - module-level `_FAILURE_CLAUSE_ORDER: tuple = (...)` — 7 entries in the spec-declared order
  - helper `_emit_short_track_row(growth_axis_unreliable: bool) -> dict[str, Any]`: returns the NaN/sentinel row when short-track gate fires
  - helper `_compute_one_track(group: pd.DataFrame, constants: ConstantsT) -> dict[str, Any]`: implements D7 steps 1-9
  - helper `_compose_track_is_clean_and_reason(traits: dict, constants: ConstantsT) -> tuple[bool, str]`: implements the composition formula from spec Requirement "QC tier track_is_clean and qc_failure_reason composition"
  - public `compute(trajectory_df, constants=None) -> pd.DataFrame`: **drop the `=None` default on `trajectory_df`** (the current stub at `qc.py:20` uses `trajectory_df=None` so the foundation test's `fn()` invocation works against `NotImplementedError`; PR #2 removed it for `kinematics.compute` at `kinematics.py:307` and we mirror that). Input validation → resolve constants → build template → per-track loop via groupby → merge + re-select columns enforcing 19-column order
  - Google-style docstring per pydocstyle, naming all 11 emitted columns and their units. Include "Coordinate convention" / "Pure-pixel + cadence-independent emission" / "Composes with Tier 0" subsections matching kinematics.py's docstring style.
  - **Docstring SHALL include a "Caveat: noiseless input" note**: *"Noiseless synthetic data (perfectly smooth trajectories) makes all three noise estimators return 0.0, which yields `0/0 = NaN` for all three pairwise agreements. Under NaN-comparison semantics this fires the three `*_agreement_high` clauses and `track_is_clean = False`. Real SLEAP-tracked data always carries ≳ 0.1 px quantization noise so this is benign in practice, but synthetic smoke tests should expect this behavior."* This is the documented edge case from test 2.B.1.
  - replace the existing `NotImplementedError("PR #3 — see docs/circumnutation/roadmap.md")` body with the working implementation

- [ ] 4.2 Run `uv run pytest tests/test_circumnutation_qc.py -v` — all sections green except 2.H.2 (Nipponbare placeholders still `TODO_*`).

- [ ] 4.3 Calibrate the Nipponbare tolerances:
  - Write a one-off script to `c:\vaults\sleap-roots\circumnutation\scripts\calibrate_qc_tolerances.py` (NOT `uv run python -c "..."` inline — PowerShell on Windows has unpredictable quoting behavior for multi-line `-c` arguments). The script loads the Nipponbare fixture, calls `qc.compute`, prints the per-track median values for `sg_residual_xy`, `d2_noise_xy`, `msd_noise_xy`, and the median of per-track `sg_d2_agreement`. Run via `uv run python c:\vaults\sleap-roots\circumnutation\scripts\calibrate_qc_tolerances.py`.
  - **Sanity floor**: assert the captured `median(per_track_sg_residual_xy)` falls within ±50% of the prelim §4.2 anchor of 1.83 px; similarly for d2 vs 2.67 px. If either differs by more than ±50%, STOP and investigate before locking — would indicate impl bug.
  - **Once sanity-floor passes**: write the captured values as `value ± 20%` (medians) and `value ± 25%` (sg_d2_agreement which is more variable as median-of-quotients) into test 2.H.2. Replace `TODO_*` placeholders with concrete numeric bounds.
  - Re-run test 2.H.2 — green.

- [ ] 4.4 Run `uv run pytest tests/test_circumnutation_qc.py -v` — all green.

- [ ] 4.5 Run `uv run pytest tests/ -x` — full suite green (no regression in foundation, Tier 0, tracked-tip-pipeline, etc.).

## 5. Docs and changelog

- [ ] 5.1 Update `docs/circumnutation/theory.md` §7.6 (line ~504 cross-tier ownership note): replace the SHALL clause *"PR #3's QC implementation SHALL NOT re-emit a duplicate `growth_axis_unreliable` column — each trait is emitted by exactly one tier"* with the equality-by-construction wording per design.md D5. Add a footnote on `track_is_clean` in the §7.6 trait table pointing to `openspec/specs/circumnutation/spec.md` as the canonical formula (6 clauses, not 3).

- [ ] 5.2 Update `docs/circumnutation/roadmap.md`:
  - Row PR #3 description: lose `cadence_nyquist_ratio` and the redundant `growth_axis_unreliable` mention (former deferred to PR #6; latter is owned per the equality-by-construction rule).
  - Row PR #6 description: gain `cadence_nyquist_ratio` mention under CC-7 / §6.5.
  - CC-5 step 3: replace "Tier 0 emits; QC does NOT re-emit" with the new wording (per design.md D5 + the Migration Plan in design.md).
  - CC-10: leave unchanged.

- [ ] 5.3 Add `docs/changelog.md` entry under "Added" (matching foundation/Tier 0 entry style):
  - "QC tier per-track quality emission (`sleap_roots.circumnutation.qc.compute`): 11 traits (3 noise estimators, 3 pairwise agreements, 2 outlier-step diagnostics, `growth_axis_unreliable`, `track_is_clean`, `qc_failure_reason`); extends `_noise.py` with `compute_d2_residual_xy` and `compute_msd_residual_xy`; promotes 4 new threshold constants to `_constants.py`. See `docs/circumnutation/theory.md` §7.6."

## 6. Verify (pre-PR-open gates)

- [ ] 6.1 Run `uv run pytest tests/test_circumnutation_qc.py --cov=sleap_roots.circumnutation.qc --cov=sleap_roots.circumnutation._noise --cov-report=term-missing`. Target: 100% coverage on `qc.py` and the two new `_noise.py` functions. Acceptable uncovered branches: `logger.debug` calls (testable via `caplog`; prefer testing).
- [ ] 6.2 Run `uv run pytest tests/ -x` — full suite green.
- [ ] 6.3 Run `uv run pytest tests/ --cov=sleap_roots --cov-fail-under=84` — confirm project-wide coverage doesn't regress below 84%.
- [ ] 6.4 Run `uv lock --check` — no dependency change in this PR.
- [ ] 6.5 Run `uv run black --check sleap_roots tests` — passes.
- [ ] 6.6 Run `uv run pydocstyle --convention=google sleap_roots/` — passes.
- [ ] 6.7 Run `uv run mkdocs build` — passes; `qc.compute` doc page renders.
- [ ] 6.8 Run `openspec validate add-circumnutation-qc-tier --strict` — valid.
- [ ] 6.9 Parametrize-id sanity check: confirm `test_stub_callable_raises_with_correct_pr` now collects 8 parametrize cases (was 9), `test_stub_module_imports_cleanly` collects 8 (was 9), and `test_stub_accepts_constants_kwarg` collects 5 (was 6). For `test_module_logger_is_namespaced` the count is **net-zero (was 15, still 15)**: §2.I.1 removes `qc` from STUB_MODULES (one fewer there) but §2.I.3 adds `qc`, `_noise`, `_geometry` to the contract-module list — net `9+6 = 15 → 8+7 = 15` (the +1 to the contract list is `qc`; `_noise` and `_geometry` were already in the contract list from PR #2 per the test convention). Mention the expected pytest collection deltas in the PR description.

## 7. PR open

- [ ] 7.1 Draft GitHub issue body to `c:\vaults\sleap-roots\circumnutation\github_issues\issue_add-circumnutation-qc-tier.md` referencing epic #197, OpenSpec change-id, theory.md §7.6, prelim §3.3 / §4.2, CC-2/CC-3/CC-5/CC-9/CC-10. Show Elizabeth before posting (do NOT post unilaterally).
- [ ] 7.2 Draft 4 follow-up issue bodies to `c:\vaults\sleap-roots\circumnutation\github_issues\`:
  - `issue_pairwise_agreement_threshold_validation.md` (Issue α — design.md follow-up cluster 1)
  - `issue_outlier_step_threshold_validation.md` (Issue β — cluster 2)
  - `issue_msd_lag_selection.md` (Issue γ — cluster 3)
  - `issue_qc_inf_input_detection.md` (Issue δ — `±inf` input detection for `qc.compute`. Raised by `/openspec-review` scientific-rigor reviewer: the pathological case where >50% of step magnitudes are `inf` lets `frac_outlier_steps` silently pass at 0.0 because `median = inf` and `inf > 2 * inf == False`. Proposal: extend `qc.compute` to detect non-finite `tip_x`/`tip_y` inputs and either fire a dedicated `non_finite_inputs` clause in `qc_failure_reason` or warn at the `_types._validate_trajectory_df` boundary. Out of scope for PR #3 because it would require either (i) extending the foundation validator (cross-tier impact) or (ii) adding a per-row finiteness scan in `qc.py` (changes the public API contract about ±inf propagation). Defer to a follow-up PR. Issue body should describe both options and request triage input.)
  - Each issue body: (a) references epic #197 + this PR's design.md; (b) describes the experimental design or proposal (fixtures + sweep + success criteria for α/β/γ; option-A/option-B trade-off for δ); (c) notes the relevant constants are `ConstantsT`-overridable (α/β/γ) or that the API contract change scope is the blocker (δ).
- [ ] 7.3 Open PR with title "feat(circumnutation): QC tier per-track quality traits (#<sub-issue>)" and a body cross-linking the epic, sub-issue, OpenSpec change-id, and design.md. Labels: `enhancement`, `circumnutation`, `multi-pr`. Body matches `.claude/commands/pr-description.md` template.
- [ ] 7.4 Push the branch. Confirm CI is green on Ubuntu / Windows / macOS at Python 3.11.

## 8. Post-merge

- [ ] 8.1 Run `/cleanup-merged` which delegates to `/openspec:archive`. The archive folds the ADDED + MODIFIED requirements into `openspec/specs/circumnutation/spec.md` and moves this change folder to `openspec/changes/archive/YYYY-MM-DD-add-circumnutation-qc-tier/`.
- [ ] 8.2 Update `docs/circumnutation/roadmap.md` row PR #3 → ✅ with the GitHub issue and PR numbers.
- [ ] 8.3 Update the Notion task page `https://www.notion.so/Circumnutation-work-sleap-roots-3494a67a7667818db2aeee80d76efdd7` with a new "Status update — YYYY-MM-DD" section (show Elizabeth the draft first).
- [ ] 8.4 Post the 3 follow-up issues (α/β/γ) to GitHub after Elizabeth approves the drafts.
