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
  compute ONE calibrated λ (true px, via the n=400 calibration slice, strictly-increasing
  `λ_reported` → well-posed `np.interp`) and use it for all three λ-traits. Extend the calibration
  artifact append-only to ~150 px (existing 18 rows + provenance frozen byte-for-byte; regression
  test pins them). No test currently reads the JSON.
- **D4 `lambda_spatial_variation`** — `MAD(λ_cal)/median(λ_cal)` over COI-valid positions
  (orientation-invariant robust spread; renamed from `apex_basal_period_consistency`, which implied
  an apex-vs-basal difference that is edge-artifact-dominated + orientation-ambiguous).
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
  `_CGAU2_LAMBDA_CALIBRATION_N400` literal (the n=400 slice incl. the extension), validated against
  the authoritative JSON by a sync test; the module never reads `tests/data` at runtime.
- **Calibration regeneration (BLOCKING):** the append-only mode must **load the existing JSON and
  pass `provenance` + the 18 rows through verbatim** (measure only the new n=400 knots), so the
  byte-for-byte freeze is mechanical, not environment-dependent. No PR #9 test reads the JSON.
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
