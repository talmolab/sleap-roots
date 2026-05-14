# Design: add-circumnutation-qc-tier

## Context

This is PR #3 in the circumnutation analysis program tracked by `docs/circumnutation/roadmap.md`. Foundation (PR #1) shipped contracts; Tier 0 (PR #2) shipped raw kinematic traits + the `_noise.compute_sg_residual_xy` helper. This PR fills in the QC tier — track-level signal-quality flags that downstream tiers and aggregation will compose with.

Theory anchors: `docs/circumnutation/theory.md` §7.6 (QC trait table; methodological note on noise estimators); `docs/circumnutation/preliminary_results_2026-05-07.md` §3.3 (two noise-estimator formulas) and §4.2 (plate 001 reference values: sg ≈ 1.83 px median, d2 ≈ 2.67 px median, sg_d2_agreement ≈ 1.46). Cross-cutting concerns: CC-2 (constants); CC-3 (pure-pixel + cadence-independent); CC-5 (composition with Tier 0's `growth_axis_unreliable`); CC-9 (logging); **CC-10 (all three independent noise estimators in Phase 1)**.

## Goals / Non-Goals

**Goals:**

- Implement `sleap_roots.circumnutation.qc.compute(trajectory_df, constants=None)` matching the canonical signature locked by the foundation's Package layout requirement.
- Emit **11 trait columns** capturing track-level signal quality: 3 independent SLEAP-localization-noise estimators (CC-10) + 3 pairwise agreements + 2 outlier-step traits + `growth_axis_unreliable` + `track_is_clean` composite + `qc_failure_reason` diagnostic.
- Extend `_noise.py` with two sibling helpers (`compute_d2_residual_xy`, `compute_msd_residual_xy`) so all three estimators share a single source of truth — same DRY discipline as PR #2.
- Honor the pure-pixel + cadence-independent contract (CC-3): QC output uses `px`, `—`, `bool`, `string` only — no `cadence_s` input, no `mm`-bearing columns.
- Lock the empirical-validation follow-up work as three focused GitHub issues so the threshold defaults can be refined as multi-plate / multi-genotype data lands.

**Non-Goals:**

- No `coi_fraction_t1`, `coi_fraction_t3`, `is_nutating`, `noise_floor_estimate` — all depend on Tier 1 (PR #6) or Tier 3 (PR #9–10) outputs that don't exist yet. Emitted by their owning tiers when those tiers land.
- No `cadence_nyquist_ratio` — both §6.5 definitions depend on Tier 1/3 quantities; the temporal flavor would also require breaking the cadence-independent signature contract. Deferred to PR #6 (which already has a `cadence_s` input via `CircumnutationInputs`).
- No pipeline composition of QC + Tier 0 + Tier 1+ outputs — that's PR #14.
- No per-genotype aggregation of QC traits — that's PR #15.
- No new test fixtures — reuse `tests/data/circumnutation_nipponbare_plate_001/` (PR #2) and `tests/data/circumnutation_plate/` (KitaakeX, foundation).

## Decisions

### D1. Emit all three independent noise estimators (CC-10 directive)

`theory.md` §7.6 lists `sg_residual_xy` and `d2_noise_xy` as core, with `msd_noise_xy` labeled *"Optional Phase 1+ trait — not recommended for Phase 1 unless requested. The existing two-estimator agreement at 1.46 already establishes data quality, and a third estimator is incremental rather than structural."* Roadmap CC-10 supersedes that read with: *"Per Elizabeth's directive: all three independent noise estimators are emitted in Phase 1."*

**This is a policy override, not a methodological refutation of §7.6.** §7.6 is correct that the SG/d2 agreement at 1.46 already validates data quality structurally — adding MSD is incremental. CC-10 elects the incremental gain anyway, on the grounds that:

- Schema stability: emitting all three now avoids a future PR that adds the column and reshapes the trait CSV downstream. Tier-1 onwards will assume the schema is stable.
- Methodological breadth: MSD extrapolation is the SPT-literature canonical estimator (Michalet 2010 *Phys. Rev. E* 82:041914). Including it makes the QC tier defensible against a reviewer familiar with that tradition, even if it adds no SLEAP-specific information.
- Marginal cost: one extra `_noise.py` helper (~30 LOC), two extra agreement columns, ~6 extra tests.

**Risk:** SG and MSD share structure (both SG-detrend the signal before measurement) → may be MORE correlated than SG and d2 → `sg_msd_agreement` may be systematically closer to 1 than the other pairs. Empirically testable (Issue α below); doesn't change today's design.

**Helper layout:** all three estimators live in the existing `_noise.py` module — single source of truth for the formulas. `qc.py` calls them; future tier PRs can compose. No inline formula in `qc.py`. All three helpers follow the existing `compute_sg_residual_xy` convention: return `float`, NaN + `logger.debug` for short-input cases (no exception raised). MSD-formula contract anchored explicitly in D8 below.

### D2. Defer `cadence_nyquist_ratio` to PR #6

`theory.md` §6.5 defines two flavors: temporal (`cadence_s / (T_nutation/2)`) and spatial (`per_frame_step / spatial_wavelength`). The temporal flavor needs `cadence_s` as input — which would break QC's cadence-independent canonical signature `compute(trajectory_df, constants=None)`. The spatial flavor needs `spatial_wavelength` which comes from PR #9–10 (Tier 3).

**Alternatives considered:**

- *Hard-code Derr's pilot reference period (3300s).* Rejected: ships a misleading trait value if rice differs from the Arabidopsis-pilot period. The whole point of `T_nutation_median` (Tier 1) is to measure this per-track.
- *NaN-stub the column to lock the schema.* Rejected: emits a permanently-NaN column for several PRs until PR #14 (pipeline) composes Tier 1's value back in. Schema-stability gain marginal because PR #14 hasn't shipped.

**Decision:** PR #3 omits the column. PR #6 (Tier 1) emits it once `T_nutation_median` exists, using `CircumnutationInputs.cadence_s` (which Tier 1's signature already accepts). Roadmap row #3 description currently lists `cadence_nyquist_ratio` as PR #3 scope; this PR's task list includes updating roadmap row #3 and row #6 to reflect the move.

### D3. Composite `track_is_clean` includes `NOT growth_axis_unreliable`

`theory.md` §7.6 defines `track_is_clean` as the AND of three conditions: `sg_d2_agreement < 1.5`, `frac_outlier_steps < 0.05`, `worst_step_ratio < 5`. That definition predates PR #2's decision to move `growth_axis_unreliable` to Tier 0.

CC-10 expands the agreement clause to all three pairwise agreements. PR #3 further extends with `NOT growth_axis_unreliable`. Final formula:

```
track_is_clean = (
    sg_d2_agreement   < SG_D2_AGREEMENT_MAX  (1.5)
  AND sg_msd_agreement  < SG_MSD_AGREEMENT_MAX  (1.5)
  AND d2_msd_agreement  < D2_MSD_AGREEMENT_MAX  (1.5)
  AND frac_outlier_steps < FRAC_OUTLIER_STEPS_MAX  (0.05)
  AND worst_step_ratio   < WORST_STEP_RATIO_MAX  (5)
  AND NOT growth_axis_unreliable
)
```

Rationale: a track with `growth_axis_unreliable = True` has six rotation-dependent Tier 0 trait columns NaN'd — it is NOT a fully usable track even if its localization-noise quality is high. The composite "is this track usable" semantic stays coherent. The per-clause traits remain available for users who want to filter differently (e.g., select tracks with `growth_axis_unreliable=True` but otherwise clean kinematic-noise quality).

**NaN propagation:** comparisons against NaN return False in Python/NumPy, so any NaN-bearing trait fires its clause. Net effect: a track with NaN inputs gets `track_is_clean = False` and gets the relevant clause(s) in `qc_failure_reason`. If the per-track short-track gate fires upstream (too few frames for any estimator), the entire numeric trait row is NaN and `qc_failure_reason = "qc_inputs_insufficient"` (single-clause override).

### D4. `qc_failure_reason` is a stable-ordered comma-separated string

Three candidate formats considered: (a) categorical single-cause string, (b) comma-separated multi-cause string, (c) bitmask integer.

**Decision: (b) comma-separated.** Empty string `""` when `track_is_clean = True`; otherwise the comma-separated list of failure-clauses in a stable canonical order encoded as a module-level `tuple` constant `_FAILURE_CLAUSE_ORDER`. Example: `"growth_axis_unreliable, sg_d2_agreement_high, frac_outlier_steps_high"`.

Rationale: human-readable in the CSV; preserves multi-cause information that (a) loses; programmatically splittable via `.str.split(", ")` if a user wants set membership; clause-name stability is enforced by the `_FAILURE_CLAUSE_ORDER` constant. Rejected (a) because real tracks often fail for multiple co-occurring reasons; rejected (c) because downstream analysis is more often done in pandas than in bitwise-aware code.

**Locked `_FAILURE_CLAUSE_ORDER` tuple** (module-level, private; enforces stable output ordering):

```python
_FAILURE_CLAUSE_ORDER: tuple = (
    "qc_inputs_insufficient",      # short-track sentinel (overrides all others)
    "growth_axis_unreliable",
    "sg_d2_agreement_high",
    "sg_msd_agreement_high",
    "d2_msd_agreement_high",
    "frac_outlier_steps_high",
    "worst_step_ratio_high",
)
```

**`qc_inputs_insufficient` is a sentinel, NOT just another clause.** When the short-track gate fires (D7 step 3, `len(subset) < SG_WINDOW_SHORT`), the reason string is literally `"qc_inputs_insufficient"` and no other clauses are appended, even if `growth_axis_unreliable=True` would have fired. The other 6 clauses can co-occur and concatenate via `", ".join(...)` in declared tuple order.

**Clause naming convention:** each clause named after the threshold crossed (`sg_d2_agreement_high`, `frac_outlier_steps_high`, `worst_step_ratio_high`) or the upstream boolean condition (`growth_axis_unreliable`, `qc_inputs_insufficient`). The `_high` suffix indicates "value exceeded its threshold"; the bare boolean clauses don't take suffixes.

### D5. Include `growth_axis_unreliable` as a column in QC's output (revise roadmap CC-5)

`docs/circumnutation/roadmap.md` cross-cutting concern CC-5 (as updated by PR #2) says: *"Tier 0 (PR #2) emits `growth_axis_unreliable` as a bool flag on the per-plant trait DataFrame; QC tier (PR #3) composes with it (e.g., as a clause in `track_is_clean`) but does NOT re-emit a duplicate column."* That constraint was carried forward from PR #2's design.md D2 architectural cleanliness ("each trait emitted by exactly one tier").

**Reversed in this PR.** QC's output is more useful standalone if `growth_axis_unreliable` is a first-class column:

- A user inspecting `track_is_clean = False, qc_failure_reason = "growth_axis_unreliable, sg_d2_agreement_high"` and wanting to bool-filter on the growth-axis cause needs the bool column. String-parsing `qc_failure_reason.str.contains("growth_axis_unreliable")` is brittle (would also match a future clause like `growth_axis_unreliable_extreme` if one were added).
- The composite `track_is_clean` references the flag's value internally; surfacing only the *string* mention via `qc_failure_reason` but not the bool itself is inconsistent.
- The "each trait emitted by exactly one tier" principle was originally about avoiding the *runtime* circular dependency (Tier 0's gate needing QC's `sg_residual_xy`). The shared `_noise.compute_sg_residual_xy` helper solved that. The remaining "no duplicate column" was architectural cleanliness, not correctness — and it forces tight coupling between QC's output and Tier 0's output for any downstream consumer.

**Equality contract:** QC recomputes `growth_axis_unreliable` locally using the *same* `_noise.compute_sg_residual_xy` helper and the *same* gate formula as Tier 0 (`D < K * sg_residual`). Numerically identical to Tier 0's value by construction, on the same `trajectory_df` input. A regression test on the Nipponbare fixture asserts element-wise equality of `kinematics.compute(df)["growth_axis_unreliable"]` and `qc.compute(df)["growth_axis_unreliable"]`.

**Dtype invariant (load-bearing for the equality test).** The `growth_axis_unreliable` column is `bool` dtype with NO NaN values in either Tier 0's or QC's output. For tracks where computation is degenerate (len < 2 after dropna), both emitters return `False` (matching `kinematics._emit_nan_row` precedent). This means the equality test can use either `(a == b).all()` or `a.equals(b)` — both produce the same result because there are no NaN values to differentiate `==` semantics from `.equals()` semantics. The test SHALL assert dtype equality (`a.dtype == b.dtype == np.dtype("bool")`) before asserting value equality.

**Input-dtype invariant (load-bearing for numerical equality).** Both Tier 0 (`kinematics.py:181`) and QC SHALL cast `xy = subset[["tip_x","tip_y"]].to_numpy(dtype=float)` with an explicit `dtype=float` — NOT bare `to_numpy()`. The foundation validator (`_types._validate_trajectory_df`) does not pin float64 on `tip_x`/`tip_y`; a caller passing int columns or float32 would feed differently-typed arrays to `compute_sg_residual_xy`, breaking last-bit equality. The explicit cast is what makes the "same inputs → same outputs through the same helper" claim hold.

**Roadmap CC-5 update:** this PR updates roadmap.md CC-5 step 3 from "Tier 0 emits; QC does NOT re-emit" to "Both Tier 0 and QC emit `growth_axis_unreliable`. Values are numerically identical by construction (same shared `_noise` helper, identical inputs after `to_numpy(dtype=float)` cast). PR #14 pipeline composition may coalesce or drop one."

**Theory.md §7.6 update:** the cross-tier ownership note added in PR #2 (`theory.md:504`) contains a SHALL statement *"PR #3's QC implementation SHALL NOT re-emit a duplicate `growth_axis_unreliable` column — each trait is emitted by exactly one tier."* This SHALL is reversed by D5; the migration plan below includes editing §7.6 to reflect the new ownership rule.

### D6. Promote four threshold constants to `_constants.py` + `ConstantsT`

Per CC-2 (*"every constant is overridable via the pipeline class init or per-call kwarg"*), this PR adds:

| Constant | Value | Source |
|---|---|---|
| `FRAC_OUTLIER_STEPS_MAX` | `0.05` | `theory.md` §7.6 `track_is_clean` clause |
| `WORST_STEP_RATIO_MAX` | `5` | `theory.md` §7.6 `track_is_clean` clause |
| `SG_MSD_AGREEMENT_MAX` | `1.5` | CC-10 (inherited from `SG_D2_AGREEMENT_MAX`) |
| `D2_MSD_AGREEMENT_MAX` | `1.5` | CC-10 (inherited from `SG_D2_AGREEMENT_MAX`) |

`_CONSTANTS_VERSION` bumps `1 → 2` per the version-sentinel contract. `_default_constants_snapshot()` extended to emit all four into the run-metadata sidecar. MSD lag stays hard-coded at 1 inside `_noise.compute_msd_residual_xy` (theory.md §7.6 explicitly fixes lag=1 for Phase 1; multi-lag extrapolation is Issue γ follow-up).

Three new pairwise-agreement constants share the same default `1.5` — open invitation to drift if they're maintained independently. Mitigation: design.md (this doc) records the inheritance reason ("from `SG_D2_AGREEMENT_MAX`") so future bumps to one constant can flag whether the others should bump too. Issue α tracks empirical re-validation of all three pairs as a unit.

**Alternative considered:** reuse the existing `SG_D2_AGREEMENT_MAX` for all three pairs (no new agreement constants). Rejected because Elizabeth selected the "Promote all four — separate threshold per pair" option in brainstorm, on the rationale that easy-to-tune-one-pair-independently is a clearer affordance than implicit-shared-default. The drift risk is mitigated by the inheritance documentation.

### D7. Per-track algorithm

```
For each track in trajectory_df.groupby(_IDENTITY_5_TUPLE, sort=False):
  1. subset = group.dropna(subset=["tip_x","tip_y"]).sort_values("frame")
  2. ALWAYS compute growth_axis_unreliable using the SAME formula and inputs
     as Tier 0, to preserve the equality contract in D5:
       If len(subset) < 2:
         D = NaN; sg = NaN; growth_axis_unreliable = False  (matches Tier 0)
         (For len = 1: D = norm(empty) is undefined; we set D = NaN.
          Comparison `D == 0.0` with NaN yields False; second clause has
          `not math.isnan(sg)` = False; net: growth_axis_unreliable = False.)
       Else:
         xy = subset[["tip_x","tip_y"]].to_numpy(dtype=float)
              # explicit dtype=float: load-bearing for equality contract — see D5.
         D = float(np.linalg.norm(xy[-1] - xy[0]))
         sg = _noise.compute_sg_residual_xy(
                  xy[:,0], xy[:,1],
                  window=constants.SG_WINDOW_SHORT,
                  degree=constants.SG_DEGREE,
              )  # may return NaN for len < constants.SG_WINDOW_SHORT
         growth_axis_unreliable = (D == 0.0) or (
             not math.isnan(sg) and D < constants.GROWTH_AXIS_RELIABILITY_K * sg
         )

  3. If len(subset) < constants.SG_WINDOW_SHORT (=5 by default):
       Emit row with the 8 numeric noise/outlier traits = NaN, the 3 pairwise
       agreements = NaN, growth_axis_unreliable per step 2, track_is_clean=False,
       qc_failure_reason = "qc_inputs_insufficient" (single sentinel reason
       overrides all per-clause firing per D4; growth_axis_unreliable=True
       still composes via track_is_clean=False but the reason string is
       LITERALLY "qc_inputs_insufficient" with no other clauses appended).
       SKIP to merge step.
       (Below this length, the SG-based estimators (sg, msd) cannot run by
       construction, so the noise-estimator stack is structurally crippled
       even though d2 alone could compute. Single clear failure reason is
       more useful than half-NaN trait row with 4 firing clauses.)
  4. Reuse xy = (already cast to float in step 2);
       frame = subset["frame"].to_numpy(dtype=float)
  5. Three noise estimators (sg already computed in step 2):
       d2  = _noise.compute_d2_residual_xy(xy[:,0], xy[:,1])
       msd = _noise.compute_msd_residual_xy(
                 xy[:,0], xy[:,1],
                 window=constants.SG_WINDOW_SHORT,
                 degree=constants.SG_DEGREE,
                 lag=1,
             )
       (each may return NaN if its own length minimum isn't met; d2 needs
       len ≥ 3, msd needs len ≥ constants.SG_WINDOW_SHORT + 1; sg from step 2
       has the len ≥ constants.SG_WINDOW_SHORT minimum which is already
       guaranteed because we passed step 3.)
  6. Three pairwise agreements:
       sg_d2  = max(sg, d2)  / min(sg, d2)     (NaN if either operand is NaN)
       sg_msd = max(sg, msd) / min(sg, msd)
       d2_msd = max(d2, msd) / min(d2, msd)
  7. Step magnitudes (gap-aware diff, matching Tier 0's convention):
       delta_xy = np.diff(xy, axis=0)
       delta_frame = np.diff(frame)
       with np.errstate(divide="ignore", invalid="ignore"):
           steps = np.linalg.norm(delta_xy / delta_frame[:,None], axis=1)
       median = float(np.nanmedian(steps))
       If median == 0 (stationary track, or numerically all-zero steps):
           frac_outlier_steps = NaN; worst_step_ratio = NaN
           (Per D3's NaN-propagation rule, this fires BOTH
           "frac_outlier_steps_high" AND "worst_step_ratio_high" clauses.
           track_is_clean = False. Documented as the expected behavior; if
           a "stationary_track" dedicated clause is wanted later, it's a
           future PR addition.)
       Else:
           frac_outlier_steps = float(
               (steps > constants.OUTLIER_STEP_RATIO * median).mean()
           )
           worst_step_ratio = float(np.nanmax(steps) / median)
  8. Compose track_is_clean and qc_failure_reason per D3 + D4.
  9. Return the 11-trait dict.

  Merge step (outside the per-track loop, mirroring kinematics.py:417-422):
       template = _build_per_plant_template_from_df(trajectory_df)
       result = template.merge(trait_df, on=list(_IDENTITY_5_TUPLE), how="left")
       result = result[list(ROW_IDENTITY_COLUMNS) + list(_QC_TRAIT_COLUMNS)]
              # FINAL re-selection step: groupby uses the 5-tuple, but the
              # template carries all 8 row-identity columns including
              # `timepoint` (which is NOT in _IDENTITY_5_TUPLE). The
              # re-selection here enforces the declared 19-column order
              # and ensures `timepoint` doesn't drop or shift.
```

Merge per-track dicts onto the foundation's per-plant template (`_io._build_per_plant_template_from_df`) via the 5-tuple key. Same shape as `kinematics.compute`'s output.

### D8. MSD-noise-estimator formula (contract anchor)

The MSD-extrapolation method (Michalet 2010 *Phys. Rev. E* 82:041914) gives, for an i.i.d. localization-noise process added to a smooth stationary signal, the relationship `MSD(τ → 0) = 4σ²` in 2D (= 2σ² per dimension × 2 dimensions). Inverting at the smallest available lag (τ=1 frame):

```
σ_MSD = sqrt(MSD(τ=1) / 4)
where MSD(τ=1) = mean( (x_res[t+1] - x_res[t])² + (y_res[t+1] - y_res[t])² )
and x_res, y_res are SG-detrended residuals using window=constants.SG_WINDOW_SHORT,
                                                 degree=constants.SG_DEGREE
```

The factor-of-4 (NOT 2) is a frequent footgun: the 1D MSD formula is `MSD = 2σ²`; the 2D version aggregates both dimensions into a single MSD value, giving `MSD = 4σ²`. The implementation MUST use 4 to match `theory.md:526`'s explicit "`MSD(τ → 0) = 4σ²`" claim.

**Length minimum:** `compute_msd_residual_xy` requires `len(x) ≥ window + lag` so the SG filter can run AND at least one squared-displacement pair exists. For `window=5, lag=1` that's `len ≥ 6`. Below that, returns NaN + `logger.debug`. With the D7 step 3 short-track gate at `len < SG_WINDOW_SHORT (=5)`, valid algorithm entry guarantees `len ≥ 5` but does NOT guarantee `len ≥ 6` — so MSD may legitimately return NaN for a track of length exactly 5. NaN propagates through the agreements; `sg_msd_agreement_high` and `d2_msd_agreement_high` clauses fire; `track_is_clean=False`. This is expected: a 5-frame track has too few samples for the MSD method to be statistically meaningful, and the QC tier flags it correctly.

### D9. Output schema (locked)

8 row-identity columns + 11 trait columns, in this order:

| # | Column | Unit |
|---|---|---|
| 1-8 | `series`, `sample_uid`, `timepoint`, `plate_id`, `plant_id`, `track_id`, `genotype`, `treatment` | per `ROW_IDENTITY_UNITS` |
| 9 | `sg_residual_xy` | `px` |
| 10 | `d2_noise_xy` | `px` |
| 11 | `msd_noise_xy` | `px` |
| 12 | `sg_d2_agreement` | `—` |
| 13 | `sg_msd_agreement` | `—` |
| 14 | `d2_msd_agreement` | `—` |
| 15 | `frac_outlier_steps` | `—` |
| 16 | `worst_step_ratio` | `—` |
| 17 | `growth_axis_unreliable` | `bool` |
| 18 | `track_is_clean` | `bool` |
| 19 | `qc_failure_reason` | `string` |

Every unit is in `PIPELINE_UNIT_VOCABULARY`. No `mm`-bearing columns. No `cadence_s` parameter on the canonical signature.

## Risks / Trade-offs

**R1. Threshold defaults have weak empirical anchors.** `SG_D2_AGREEMENT_MAX=1.5`, `FRAC_OUTLIER_STEPS_MAX=0.05`, `WORST_STEP_RATIO_MAX=5` all derive from single-plate (plate 001 Nipponbare) observations. `SG_MSD_AGREEMENT_MAX=1.5` and `D2_MSD_AGREEMENT_MAX=1.5` are inherited from `SG_D2_AGREEMENT_MAX` without their own empirical measurement. **Mitigation:** ConstantsT exposes every threshold for per-call override; three focused follow-up issues (α/β/γ below) track the empirical-validation work; design.md documents the weakness explicitly so future-Elizabeth or a reviewer can decide to update defaults.

**R2. Duplicate `growth_axis_unreliable` column.** Both Tier 0's and QC's outputs carry the same column. **Mitigation:** the shared `_noise` helper guarantees numerical equality by construction; a regression test on the Nipponbare fixture asserts element-wise equality; PR #14 (pipeline composition) coalesces or drops one (trivial because they're equal). Documented in CC-5 update.

**R3. SG and MSD estimators share structure** (both SG-detrend before measurement), so `sg_msd_agreement` may be systematically closer to 1 than `sg_d2_agreement` or `d2_msd_agreement`. **Mitigation:** Issue α empirical sweep characterizes the typical distribution per pair; if `sg_msd_agreement` is systematically near 1 and adds no QC discrimination, a future PR may either tighten its threshold or drop it.

**R4. NaN-clause-firing semantics may surprise.** A track where the SG estimator returns NaN (track too short for SG window) will fire `sg_d2_agreement_high` and `sg_msd_agreement_high` clauses simultaneously (because `NaN < threshold` is False, so the clause guard `not (NaN < threshold)` is True). **Mitigation:** the short-track gate at the top of the algorithm (step 2) catches "nothing computable" and uses a single `qc_inputs_insufficient` clause instead. Borderline cases (some estimators succeed, others fail) still fire their per-clause failures, which is correct semantically.

**R5. `±inf` propagation through outlier-step computation.** Tier 0's spec (Requirement: Tier 0 input-validation boundary) commits to "`±inf` propagates without raising" for `tip_x`/`tip_y` values. QC inherits this: `np.linalg.norm` of an `±inf` step yields `inf`; `np.nanmedian` of mostly-finite steps with one `inf` is the finite median; `inf > 2 * finite_median` is True so `frac_outlier_steps` correctly increments; `worst_step_ratio = inf / finite_median = inf` which fires `worst_step_ratio_high` → `track_is_clean=False`. The pathological case is `>50%` of steps being `inf` → `median = inf` → `inf > inf = False` → `frac_outlier_steps = 0` (silent pass). **Mitigation:** documented as a known non-goal; the upstream data integrity is the caller's responsibility (no SLEAP prediction emits `±inf`). A QC-level `±inf` detector is deliberately out of scope for PR #3 — file a follow-up issue if needed.

## Migration Plan

This PR is additive — no breaking changes to existing capabilities. Foundation + Tier 0 outputs unchanged.

**Foundation test migration (`tests/test_circumnutation_foundation.py`):**

- Remove `("qc", "compute", 3)` from `STUB_MODULES` (line ~32-42). Parametrize-id count for `test_stub_module_imports_cleanly` and `test_stub_callable_raises_with_correct_pr` drops **9 → 8**.
- Remove `("qc", "compute")` from `STUBS_WITH_CONSTANTS_KWARG` (line ~810-817). Parametrize-id count for `test_stub_accepts_constants_kwarg` drops **6 → 5**.
- Extend the contract-module list in `test_module_logger_is_namespaced` to include `qc` (now an implementation module).
- **Update `test_schema_and_constants_versions_are_integers_equal_to_one`** (line ~203-210) — currently asserts `_CONSTANTS_VERSION == 1`. After D6 bumps the version, this assertion MUST update to `== 2` (and probably should be renamed to `test_schema_version_is_1_and_constants_version_is_2` for clarity, mirroring the PR #1 naming intent).
- `EXPECTED_CONSTANTS` in the foundation test uses `hasattr` rather than equality of the full set, so adding 4 new constants does NOT break it. `_default_constants_snapshot()` test still passes because new constants get added to the snapshot fn (per D6).

**`_noise.py` extension (consistency with existing `compute_sg_residual_xy` precedent):**

- `compute_d2_residual_xy` MUST emit `logger.debug(...)` when `len(x) < 3` before returning NaN, matching the SG precedent (`_noise.py:70-76`).
- `compute_msd_residual_xy` MUST emit `logger.debug(...)` when `len(x) < window + lag` before returning NaN, matching the SG precedent.

**`docs/circumnutation/roadmap.md` updates:**

- Row PR #3: scope description loses `cadence_nyquist_ratio` and `growth_axis_unreliable` (the former deferred to PR #6, the latter is Tier-0-owned-with-QC-re-emission per D5).
- Row PR #6: scope description gains `cadence_nyquist_ratio` (under CC-7 / §6.5).
- CC-5 step 3: updated to reflect both-tiers-emit-with-equality-by-construction per D5 (replaces the "QC does NOT re-emit" wording).
- CC-10: unchanged (already describes the three-estimator design).

**`docs/circumnutation/theory.md` updates:**

- §7.6 cross-tier ownership note (line ~504): EDIT the SHALL clause that currently says *"PR #3's QC implementation SHALL NOT re-emit a duplicate `growth_axis_unreliable` column"* to reflect D5's reversal. New wording: *"Both Tier 0 and QC emit `growth_axis_unreliable`; values are numerically identical by construction because both tiers call the shared `_noise.compute_sg_residual_xy` helper with identically-typed inputs."*
- §7.6 trait table: add a footnote on `track_is_clean` pointing to `openspec/specs/circumnutation/spec.md` as the canonical formula (the §7.6 prose lists only the 3 original clauses; the canonical formula now has 6 clauses per D3).

**Nipponbare reference-value test specification (avoid PR #2's median-of-means trap):**

- Prelim §4.2 reports `sg = 1.83 px median` and `d2 = 2.67 px median` as **per-track medians across 6 tracks**. The `sg_d2_agreement ≈ 1.46` quoted in §7.6 is the *quotient of the two reported per-track medians* (`2.67 / 1.83 = 1.459`), NOT the median of the 6 per-track ratios.
- The Nipponbare reference-value test in PR #3 SHALL anchor on the **per-track median values** (≈ 1.83 and ≈ 2.67) and the **median of the 6 per-track `sg_d2_agreement` values** (which may differ from `1.46` by 10-20% on heavy-tailed distributions). Concrete tolerance ranges are locked during impl after one calibration run (similar to PR #2's section 4.4 calibration step), with the prelim §4.2 values used as a sanity floor only.

## Follow-up Issues (drafted to vault before PR opens; Elizabeth posts)

Three focused GitHub issues, one per experiment cluster. Each references epic #197 and this PR's design.md.

**Issue α — Empirically validate pairwise-agreement thresholds for circumnutation QC.** Sweep all three pairwise agreements (`sg_d2_agreement`, `sg_msd_agreement`, `d2_msd_agreement`) across plate 001 (Nipponbare), the KitaakeX fixture, and any subsequently-shipped CMTN plates (002 GA₄, 003 MOCK rep, 004 TZT once they land). Compare against the current default of 1.5 for all three. Report distribution medians + 95th percentiles per pair. Update defaults if measured 95th percentile differs from 1.5 by more than ±0.3 on a representative pool.

**Issue β — Empirically validate outlier-step thresholds for circumnutation QC.** Sweep `FRAC_OUTLIER_STEPS_MAX`, `WORST_STEP_RATIO_MAX`, and the underlying `OUTLIER_STEP_RATIO` multiplier (currently `2`) across the same fixture set. Especially important once low-growth-mutant data is available — these tracks may legitimately exhibit higher outlier rates without being "noisy". Identify the natural break-point between clean and noisy tracks in the empirical distribution; consider distinguishing genotype-aware thresholds.

**Issue γ — MSD lag selection for circumnutation noise estimation.** Compare single-lag (`lag=1`, current default per theory.md §7.6) against Michalet-2010-style multi-lag extrapolation across lags `{1, 2, 3, 4, 5}` on plate 001 and KitaakeX. Report whether the multi-lag approach gives systematically tighter agreement with `sg_residual_xy` and `d2_noise_xy`. If multi-lag wins, change `_noise.compute_msd_residual_xy` default and document the methodology in `docs/circumnutation/theory.md` §7.6.

## Open Questions

None blocking. The empirical-anchor weakness (R1) is the largest unknown, and it is explicitly tracked via Issues α/β/γ as documented follow-up work rather than as a blocker.
