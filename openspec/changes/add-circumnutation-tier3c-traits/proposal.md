## Why

Tier 3c is the trait-emission tier that turns PR #9's spatial-CWT machinery into per-plant
traits, and the first consumer of that machinery. Its headline output is
`traveling_wave_residual` — the program's central falsifiable test of the QPB
steady-traveling-wave hypothesis `λ_spatial = v · T_nutation` (theory.md §4.7). A pre-spec
real-data investigation on all 6 Nipponbare plate-001 proofread tracks grounded the trait
definitions and overturned two §7.4 "Handoff to PR #10" claims (see Impact). Design + evidence:
`docs/superpowers/specs/2026-06-10-add-circumnutation-tier3c-traits-design.md` and
`docs/circumnutation/investigations/2026-06-10-tier3c-traveling-wave/report.md`.

## What Changes

- Add a new `sleap_roots.circumnutation.traveling_wave` implementation module exposing
  `compute(trajectory_df, cadence_s, constants=None) -> pd.DataFrame`, mirroring the Tier 1/Tier 2
  trait-emission template (5-tuple groupby, 8 row-identity columns, per-plant template merge).
  This is an ADDITION (never a stub) — implementation-module count grows 8 → 9; the 3 remaining
  stubs (`parametric`, `plotting`, `pipeline`) are unchanged.
- Emit **6 float64 trait columns**: `lambda_spatial_median_px`, `lambda_spatial_variation`,
  `traveling_wave_residual`, `lambda_expected_px`, `lambda_spatial_mad_px`, `coi_valid_fraction`.
- **Reduced scope (per PR #9 descope → #230):** ship ONLY the 3 λ-based traits + diagnostics on
  the full reconstructed trail (no L_gz mask). The 5 `L_gz`/`L_c`-dependent traits
  (`L_gz_estimate`, `L_c_estimate`, `B_balance_number`, `L_gz_steady_state_residual`,
  `L_gz_resolvable`) and the growth-zone mask remain **blocked on #230** and are **omitted**
  (not reserved as NaN columns).
- Compose self-contained: recompute Tier 0 (`kinematics.compute`) and Tier 1
  (`nutation.compute(coordinate="lateral")`) internally and join on the full `_IDENTITY_5_TUPLE`.
  Documented redundancy for the PR #14 pipeline DAG to dedup.
- Apply the cgau2 calibration map (`tests/data/circumnutation_spatial_cwt_calibration.json`) to λ
  in true px; apply the spatial COI gate (reuse existing `COI_FRACTION_MAX`). **No new constants;
  `_CONSTANTS_VERSION` stays 6.**
- **Extend the calibration artifact** (append-only) to cover the observed λ range (~150 px) via
  `scripts/circumnutation/capture_spatial_coi_factor.py`, freezing the existing 18 rows +
  provenance byte-for-byte, with a regression test asserting they are unchanged.
- **Deviation records (theory.md):** correct the now-false §7.4 handoff-note-2 ("bias cancels in
  the ratio") and handoff-note-4 (apex-vs-basal), rename the §7.4 trait
  `apex_basal_period_consistency` → `lambda_spatial_variation` (redefined as an orientation-
  invariant robust spread `MAD/median`), update the three §7.4 trait rows, and add **Appendix
  B(6)** preserving the original wording. Propagate the rename + reduced-scope cleanup to
  `roadmap.md` line 146 and `docs/changelog.md`.

## Impact

- **Affected specs:** `circumnutation` — MODIFY "Package layout" (add `traveling_wave`,
  8→9 impl, addition scope note, callability scenario, import scenario); ADD "Tier 3c
  traveling-wave trait emission API".
- **Affected code (new):** `sleap_roots/circumnutation/traveling_wave.py`;
  `tests/test_circumnutation_traveling_wave.py`.
- **Affected code (edits):** `scripts/circumnutation/capture_spatial_coi_factor.py` (append-only
  merge mode); `tests/test_circumnutation_foundation.py` (add `traveling_wave` to the
  namespaced-logger list, the `IMPLEMENTATIONS_WITH_CONSTANTS_KWARG` list, and a dedicated
  `traveling_wave` branch in `test_implementation_accepts_constants_kwarg`).
- **Affected docs:** `theory.md` (§7.4 rows + handoff notes 2/4 + new Appendix B(6));
  `roadmap.md` (line 146 rename + descope cleanup, add `traveling_wave` to the module list);
  `docs/changelog.md`.
- **Affected data:** `tests/data/circumnutation_spatial_cwt_calibration.json` (append-only
  extension; existing 18 rows + provenance preserved byte-for-byte; no test reads it today — a new
  regression test pins the existing rows).
- **Scientific result (D7):** on plate-001, `traveling_wave_residual ≈ 0.10–0.18` (QPB holds to
  ~9–18%, calibrated) and `lambda_spatial_variation ≈ 0.13–0.37`.
- **Blocked / follow-up:** #230 (L_gz/L_c tip-trail transfer). No downstream consumer expects the
  omitted columns (verified against spec.md + roadmap + theory).
