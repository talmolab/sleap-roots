## Context

Full design + the two-round critical-review reconciliation log live at
`docs/superpowers/specs/2026-06-10-add-circumnutation-tier3c-traits-design.md`; the grounding
real-data evidence at `docs/circumnutation/investigations/2026-06-10-tier3c-traveling-wave/report.md`.
This file captures the decisions an implementer/reviewer needs without re-reading those.

## Goals / Non-Goals

- **Goals:** emit the 3 λ-based traits + 3 diagnostics per track; correct unit + cgau2-calibration
  reconciliation; COI gating; self-contained Tier 0/1 composition; real-data validation.
- **Non-Goals:** `L_gz`/`L_c`/`B_balance_number` + the growth-zone mask (→ #230); pipeline DAG
  wiring (→ PR #14); parametric (→ PR #11); plotting (stub remains).

## Decisions

- **D1 reduced scope** — ship only `lambda_spatial_median_px`, `traveling_wave_residual`,
  `lambda_spatial_variation` (+ diagnostics `lambda_expected_px`, `lambda_spatial_mad_px`,
  `coi_valid_fraction`). The 5 L_gz traits are **omitted** (not NaN-reserved); blocked on #230.
- **Module** `traveling_wave.py`; ADDITION (never a stub) → impl count 8 → 9; stubs unchanged.
- **Composition** — recompute Tier 0 (`kinematics.compute`) + Tier 1
  (`nutation.compute(coordinate="lateral")`) internally; **merge operands on the full
  `_IDENTITY_5_TUPLE`** (with int64 coercion — NOT `track_id` alone, NOT `.at[key]`; CR-1/CR2-1).
  Redundant by design → forward-note for the PR #14 DAG to dedup.
- **D3 calibration** — the §7.4 "bias cancels in the ratio" handoff claim is empirically FALSE;
  compute ONE calibrated λ (true px, via the **n-averaged** `ratio(λ)` curve — mean across
  `n ∈ {200,400,600}` per `λ_true`; see Round-3 reconciliation — strictly-increasing `λ_reported_mean`
  → well-posed `np.interp`) and use it for all three λ-traits. Extend the calibration artifact
  append-only to ~150 px (new `λ_true` knots for ALL three n; existing 18 rows + provenance frozen
  byte-for-byte; regression test pins them). No test currently reads the JSON.
- **D4 `lambda_spatial_variation`** — `MAD(λ_cal)/median(λ_cal)` over COI-valid positions
  (orientation-ROBUST spread; renamed from `apex_basal_period_consistency`, which implied an
  apex-vs-basal difference that is edge-artifact-dominated + orientation-ambiguous).
- **D5 constants** — none new; reuse `COI_FRACTION_MAX`. `_CONSTANTS_VERSION` stays 6.
- **Gating** — spatial-availability gate (degenerate midline/resample, caught CWT raise, or
  in-COI fraction > `COI_FRACTION_MAX`) NaNs #1–#5; `is_nutating`/`v` flow through
  `lambda_expected_px` to NaN #3/#4 (division guarded: NaN when `v` non-finite or
  `lambda_expected_px ≤ 0`). `coi_valid_fraction` is finite iff a ridge formed.
- **D6 determinism** — target `atol=1e-6` (inherits the recomputed Tier 1 scipy looseness);
  re-measured on a real-data canary covering the full chain incl. the np.interp calibration.

## Risks / Trade-offs

- Recompute redundancy (Tier 0/1 twice) → mitigated by the PR #14 DAG; forward-noted.
- Extending the PR #9 calibration artifact → append-only merge + byte-identical regression test
  removes the env-drift blast radius.
- λ-calibration via piecewise-linear-on-extended-table (vs a smooth fit) chosen for auditability +
  reuse of the deterministic PR #9 generator.

## Deviation discipline

Two §7.4 handoff claims overturned (bias-cancels; apex-vs-basal). theory.md edits (§7.4 rows +
handoff notes 2/4 + the dead-name scope note) + new **Appendix B(6)** preserving the originals are
enumerated in `tasks.md` §8.2; roadmap + changelog propagation in §8.3.

## OpenSpec-review reconciliation (5-subagent review)

- **Packaging (BLOCKING):** the cgau2 calibration data must reach the production module at runtime,
  but the JSON lives in `tests/data/` (not in the wheel). Resolution: ship a committed in-package
  `_CGAU2_LAMBDA_CALIBRATION` literal (the n-averaged `ratio(λ)` curve incl. the extension; see
  Round-3), validated against the authoritative JSON by a sync test; the module never reads
  `tests/data` at runtime.
- **Calibration regeneration (BLOCKING):** the append-only mode must **load the existing JSON and
  pass `provenance` + the 18 rows through verbatim** (measure only the new `λ_true` knots for all
  three n), so the byte-for-byte freeze is mechanical, not environment-dependent. No PR #9 test reads
  the JSON.
- **Silent-NaN join (IMPORTANT):** Tier 0/1 return int64 keys; raw `track_id` may be float64 →
  pandas merge yields silent all-NaN (not `KeyError`). Coerce per-track keys to int64; the test
  asserts FINITE operands for healthy tracks.
- **Issue hygiene (BLOCKING):** draft the PR #10 GitHub issue to the vault → user OK → post →
  backfill roadmap line 146 (task 0.1).
- **Coverage additions:** `_TRAVELING_WAVE_TRAIT_UNITS` mapping + units-vocabulary test (#222);
  scenarios/tests for DEBUG-log, constants rejection, cadence value/type, all-NaN-tip/single-frame,
  COI-gate boundary, calibration-literal sync; real-data test `skipif`-gated; tasks 1.1+1.2 land in
  one commit; synthetic-recovery params + tolerance pinned; px⁻¹ vocabulary token confirmed
  not-needed (no px⁻¹ column emitted).

## Round-3 review reconciliation (3rd critical-review pass)

The fresh-eyes pass surfaced scientific gaps the engineering-focused rounds missed; all reconciled:
- **Calibration n-dependence (decision: average across n).** The cgau2 ratio scatters ~7% across
  `n ∈ {200,400,600}`, non-monotonically; fixing n=400 injected a ~±5% systematic ≈ the residual
  signal. Resolution: the consumer uses a single **n-averaged** `ratio(λ)` curve (`_CGAU2_LAMBDA_CALIBRATION`),
  the extension measures the new λ_true knots for ALL three n, and the ~±5% systematic is DOCUMENTED
  (spec + theory) so track-to-track residual differences within it are not over-interpreted.
- **D7 provisional (4/6 clamp, not 2).** 4 of 6 real tracks currently clamp-extrapolate; D7's ~9–18%
  is restated as provisional pending the post-extension re-measurement; the real-data test asserts the
  post-extension residual.
- **`lambda_spatial_variation` noise floor.** Measured on a uniform-λ synthetic (task 7.3); values
  within the argmax-quantization floor are "consistent with uniform" — the trait is a spread
  diagnostic, not a calibrated H1 test below the floor.
- **Determinism = argmax index equality.** The cross-OS canary pins the integer ridge scale-index
  array equal (tie-flips are discrete scale-step jumps, not atol-bounded); the float atol is measured
  and may be looser than 1e-6.
- **`traveling_wave_residual` interpretability regime.** Documented: meaningful only when
  `lambda_expected_px ≳ one wavelength` (a tiny-but-positive v·T gives a large-but-finite artifact).
- **"orientation-robust" not "orientation-invariant."** The MAD/median statistic is reversal-invariant
  but the COI-interior selection depends on the resample `s_a=0` anchor — wording softened.
- **Mechanics:** append new calibration rows at the END (existing 18 a contiguous prefix); regression
  compares by `(n, λ_true)` key; the literal is generated from full-precision JSON tokens; §6 precedes
  §4.2/4.3; `lambda_spatial_mad_px` named among the is_nutating-independent spatial traits.

## Round-4 review reconciliation (focused science pass)

- **Noise-floor claim CORRECTED (round-3 over-correction).** Empirically, `lambda_spatial_variation`
  on a NOISE-FREE uniform-λ synthetic reads ≈ 0 (there is NO argmax-quantization floor; the round-3
  "~0.13 floor" hypothesis was wrong — that number came from the fixture's default localization noise).
  Reframed: the trait correctly reads ≈0 when λ is uniform; real-data 0.13–0.37 is genuine
  ridge-localization scatter that grows with noise, interpreted relative to the noise level — not a
  quantization floor, not pure biology. Task 7.3 now asserts ≈0 on a `noise_sigma_px=0` trail.
- **Determinism field CORRECTED.** `SpatialRidgeResult` exposes no integer scale-index; the determinism
  contract asserts exact-equality on the PUBLIC `wavelengths_px[interior]` (1:1 with the argmax index)
  instead. The spatial-λ ridge is exact cross-OS (modulo a tie-flip); the 1e-6 atol budget covers only
  the v·T-derived columns (`lambda_expected_px`, `traveling_wave_residual`).
- **n-averaging verified SOUND** (re-computed): the averaged `λ_reported_mean` axis is strictly
  increasing; plate-001 median 0.142, range [0.094, 0.182] (unchanged vs n=400); averaging an
  unstructured non-monotone n-scatter is defensible (it is deterministic quantization/edge structure,
  not usable signal). Supersedes the superpowers-design CR-7/CR2-4 "n=400-only" reasoning.
