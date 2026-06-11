# Design: add-circumnutation-tier3c-traits (PR #10)

**Date**: 2026-06-10
**Author**: eberrigan (+ Claude)
**Epic**: #197 — circumnutation analysis program
**Upstream (merged)**: PR #8 midline (#227), PR #9 spatial CWT (#231)
**Blocks-on**: #230 (L_gz/L_c tip-trail transfer) — see Scope
**Evidence**: `docs/circumnutation/investigations/2026-06-10-tier3c-traveling-wave/report.md`

## Context

Tier 3c is the **trait-emission** tier that turns PR #9's spatial-CWT machinery into
per-plant traits, and the first consumer of that machinery. Its headline output is
`traveling_wave_residual` — the program's central falsifiable test of the QPB
steady-traveling-wave hypothesis λ_spatial = v · T_nutation (theory.md §4.7).

It mirrors the Tier 1 (`nutation.py`) / Tier 2 (`psi_g.py`) trait-emission template, NOT the
machinery shape of Tier 3a/3b. Public API:

```python
def compute(trajectory_df: pd.DataFrame, cadence_s: float, constants=None) -> pd.DataFrame
```

## Scope (D1)

**Reduced scope: traveling-wave validation traits only.** PR #9 empirically established (all 6
plate-001 tracks) that the §7.4 |κ|-envelope-peak recipe for `L_gz`/`L_c` does NOT transfer to
top-view tip-trail κ(s); those traits + the L_gz growth-zone mask are descoped to research
issue **#230**. PR #10 therefore ships ONLY the 3 traits the λ(s_a) ridge supports and operates
on the **full reconstructed trail (no mask)**.

- **Deliverable**: `lambda_spatial_median_px`, `traveling_wave_residual`, `lambda_spatial_variation`.
- **Blocked on #230 (omitted entirely, not reserved as NaN)**: `L_gz_estimate`, `L_c_estimate`,
  `B_balance_number`, `L_gz_steady_state_residual`, `L_gz_resolvable`, the L_gz mask.
  Rationale: #230 may redefine these for tip-trail geometry; committing names/units now would
  pre-commit a definition that may change, and all-NaN columns are ambiguous. Matches how
  nutation/psi_g emit only what they compute.

## Goals / Non-Goals

- **Goals**: emit the 3 λ-based traits + diagnostics per track; correct unit + cgau2-calibration
  reconciliation; COI reliability gating; self-contained composition with Tier 0/1; real-data
  validation on plate-001.
- **Non-Goals**: L_gz/L_c/B (→ #230); the growth-zone mask; pipeline DAG wiring (→ PR #14);
  parametric fits (→ PR #11); plotting (stub remains).

## Architecture

New flat module `sleap_roots/circumnutation/traveling_wave.py` (sibling of nutation.py,
psi_g.py). Public `compute`; private `_compute_one_track`, `_check_constants`,
`_check_cadence_s` (reuse the established helpers / patterns).

### Composition (recompute internally) — D-composition

`compute` is self-contained: it calls `kinematics.compute(trajectory_df)` (Tier 0) and
`nutation.compute(trajectory_df, cadence_s, coordinate="lateral")` (Tier 1) once each, indexes
the results by track, and reads `v_total_median_px_per_frame`, `T_nutation_median`,
`is_nutating`. This keeps the `(trajectory_df, cadence_s, constants)` signature identical to its
siblings, guarantees operand consistency, and gets the `is_nutating` gate for free.

> **Forward note to PR #14 (pipeline DAG):** Tier 3c RE-RUNS Tier 0 and Tier 1 internally. In a
> full pipeline run those tiers are also computed standalone, so the DAG MUST dedup — compute
> Tier 0/Tier 1 once and route their outputs into Tier 3c — rather than calling
> `traveling_wave.compute` naively. This redundancy is intentional for standalone correctness.

### Per-track chain

```
x, y (tip trajectory, sorted by frame, NaN-dropped)
  → midline.reconstruct(x, y, cadence_s)                      # MidlineResult
  → resample_curvature(curvature_px_inv, arc_length_px, velocity_sub_noise_mask)
  → compute_scaleogram(kappa_uniform, ds)                     # cgau2
  → extract_ridge(scaleogram)                                 # wavelengths_px, in_coi, ...
  → interior = ~in_coi                                        # COI gate
  → λ_cal[i] = calibrate(wavelengths_px[i])                   # true px (per-λ)
  → λ statistics over interior
```

If `midline`/`resample` is degenerate, or the ridge cannot be formed (<9 samples), or the
in-COI fraction exceeds `COI_FRACTION_MAX`, the spatial λ-traits are NaN for that track.

## Trait schema (D2, revised by D4)

8 `ROW_IDENTITY_COLUMNS` + 6 trait columns, in declared order:

| # | Column | dtype | units | definition |
|---|--------|-------|-------|------------|
| 1 | `lambda_spatial_median_px` | float64 | px | `median(λ_cal[interior])` |
| 2 | `lambda_spatial_variation` | float64 | — | `MAD(λ_cal[interior]) / median(λ_cal[interior])` (robust fractional spread; 0 = uniform) |
| 3 | `traveling_wave_residual` | float64 | — | `|λ_med − lambda_expected| / lambda_expected` |
| 4 | `lambda_expected_px` | float64 | px | `v · T_frames`, `T_frames = T_nutation_median / cadence_s` |
| 5 | `lambda_spatial_mad_px` | float64 | px | `median(|λ_cal[interior] − median|)` (numerator of #2) |
| 6 | `coi_valid_fraction` | float64 | — | `interior.sum() / interior.size` |

`MAD` = median absolute deviation about the median (the robust, orientation-invariant spread;
see D4). `λ_med` in #3 is `lambda_spatial_median_px` (#1).

### NaN-gating

- **Spatial-availability gate** → NaN on #1,#2,#3,#4,#5 when: `midline` degenerate, OR `resample`
  degenerate, OR ridge cannot form (insufficient samples), OR
  `(in_coi.sum() / in_coi.size) > COI_FRACTION_MAX` (= `1 − coi_valid_fraction > 0.5`).
  `coi_valid_fraction` (#6) is populated whenever a ridge exists (NaN only on full degeneracy),
  so it diagnoses why a row gated.
- **`is_nutating` gate** → flows through `lambda_expected_px`: a non-nutating track has
  `T_nutation_median = NaN` ⇒ #4 = NaN ⇒ #3 = NaN. The pure-spatial traits #1, #2 remain valid
  (λ exists whether or not temporal nutation was detected).

## Key reconciliations (the §7.4 handoff contract + the corrections)

### Unit reconciliation (compare in px)
λ_spatial is px; `v_total_median_px_per_frame` is px/frame; `T_nutation_median` is **seconds**
(no suffix; the "hr" in stale trait tables is post-conversion — do NOT ×3600). Convert T to
frames: `T_frames = T_nutation_median / cadence_s`, so `lambda_expected_px = v · T_frames`
(matches §6.4's 5.83 × 3333/300 ≈ 65 px). Pure-pixel pipeline (CC-3): no `[mm]`, no `px_per_mm`.

### cgau2 calibration (D3) — DEVIATION from §7.4 handoff
The handoff (note 2) claimed the cgau2 bias "largely cancels" in the residual ratio. **This is
empirically false** (investigation FINDING 1): `v·T_frames` is a true-px prediction with no
cgau2 bias, so the naive `|λ_raw − v·T|/(v·T)` is **mixed-domain** and wrong (it only looks
small — raw median 0.054 — because the +13% over-report accidentally nudges λ_raw toward v·T).
**Decision:** compute ONE calibrated λ array in true px and use it for ALL three λ-traits
(`lambda_spatial_median_px`, `traveling_wave_residual`, `lambda_spatial_variation`). Honest
residual: median 0.147, range [0.087, 0.177] on plate-001 → **QPB holds to ~9–18%**.

Calibration lookup: `circumnutation_spatial_cwt_calibration.json` gives `ratio = λ_reported/λ_true`
at sampled `(n, ds, λ_true)`. A consumer knows `λ_reported`, so interpolate `ratio` as a function
of `λ_reported` and divide: `λ_cal = λ_reported / ratio_interp(λ_reported)`, applied per-λ.

Observed `λ_reported` reaches ~142 px but the current table covers only [21, 91] px → clamped
extrapolation for 2 tracks. **PR #10 extends the table** (re-run `capture_spatial_coi_factor.py`
with `λ_true` up to ~150 px). Capture is deterministic ⇒ existing entries reproduce identically
⇒ PR #9's tolerance-based calibration test stays green; new entries only extend coverage.

### COI reliability gate
`extract_ridge` ships the raw, un-COI-masked ridge. PR #10 gates on the in-COI fraction using
the existing `COI_FRACTION_MAX = 0.5`. On real data the in-COI fraction is 0.02–0.14, so the
gate protects degenerate/short trails without binding good ones.

### `lambda_spatial_variation` (D4) — DEVIATION from §7.4 (rename + redefinition)
theory.md §7.4 named this `apex_basal_period_consistency` and described it as "uniform λ(s) along
trail; large variation = H1 violation". The investigation (FINDING 2/3) showed an apex-vs-basal
*difference* is unmeasurable here: the s_a→0 / s_a→max extremes are edge/scale-rail artifacts
(track-0 apex λ ≈ 432 px), and the apex orientation is ambiguous (resample's s_a=0 is frame-0 =
oldest tissue; §6.2 sign-polarity warning). Implement the **uniformity** intent as an
orientation-invariant, outlier-robust spread `MAD(λ_cal)/median(λ_cal)` (real data 0.13–0.37),
and **rename** to `lambda_spatial_variation` (plainer; honestly a spread, not a directional
comparison). CV (std/mean) and IQR/median were rejected as unstable/outlier-driven on the spiky
per-position ridge.

## Constants (D5)
No new constants. Reuse `COI_FRACTION_MAX` (already defined in `_constants.py` + `ConstantsT`,
previously unused — PR #10 is its first consumer). Continuous residual + MAD/median need no
thresholds. **`_CONSTANTS_VERSION` stays 6** (bump only for a real new constant).

## Determinism (D6)
The composed stack = SG smoothing (midline) + pywt cgau2 CWT (spatial) + np.interp (calibration)
+ argmax ridge + medians, AND the recomputed Tier 1 (which already composes scipy fft/signal/
ndimage/stats). Tier 3c therefore inherits Tier 1's looseness; target **atol = 1e-6** (nutation's
value), re-measured on a real-data canary during implementation — measure, don't cargo-cult.

## Input-validation boundary
Reuse `_validate_trajectory_df` (8 identity cols + `frame`, `tip_x`, `tip_y`; non-empty;
DataFrame) and `_check_cadence_s` (positive finite). `constants` validated as None or ConstantsT.

## Testing strategy (TDD)
- Synthetic: a known-λ trajectory → λ_spatial recovers it within calibration tolerance;
  uniform-λ synthetic → small `lambda_spatial_variation`; non-nutating → residual NaN, λ valid.
- Gating: degenerate/short track → spatial traits NaN, `coi_valid_fraction` diagnostic;
  forced low-COI → gated.
- Schema: exact 8+6 columns in order, dtypes, 5-tuple groupby, per-plant merge.
- Determinism: two-run canary at the measured atol.
- **Real plate-001 (all 6 tracks)**: `traveling_wave_residual` finite and ~0.09–0.18 (QPB),
  `lambda_spatial_variation` ~0.13–0.37, all gates pass — the scientific cross-check.
- Calibration-table extension: regenerated JSON covers observed λ; PR #9 calibration test green.

## Risks / Trade-offs
- Recompute redundancy (Tier 0/1 twice) → mitigated by the PR #14 DAG; documented forward-note.
- Extending the PR #9 calibration artifact touches a committed deliverable → mitigated by
  determinism (existing entries unchanged) + re-running PR #9's test.
- λ-calibration extrapolation beyond the (extended) table → bounded; documented limitation.

## Open Questions
- Exact `λ_true` grid for the table extension (decide during impl from the observed λ range).
- Final atol value (measure during impl; expect 1e-6).

## Decisions log (D1–D7)
- **D1** Reduced scope (λ-traits only); blocked L_gz traits omitted, not reserved.
- **Composition** Recompute Tier 0/1 internally (self-contained); forward-note for PR #14 dedup.
- **Module** `traveling_wave.py`.
- **D2** 3 core + 3 diagnostics (revised after D4: dropped apex/basal px, added `lambda_spatial_mad_px`).
- **D3** Calibrate λ to true px (single λ for median + residual); "bias cancels" handoff claim
  overturned; extend calibration table to ~150 px.
- **D4** `lambda_spatial_variation` = MAD/median (orientation-invariant); rename from
  `apex_basal_period_consistency`.
- **D5** No new constants; `_CONSTANTS_VERSION` stays 6.
- **D6** atol target 1e-6, re-measured.
- **D7** QPB holds to ~9–18% on plate-001 (real scientific result).
