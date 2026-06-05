## Why

Program PR #7 (epic #197, `docs/circumnutation/roadmap.md`) graduates the
`psi_g` stub into the **Tier 2 Bastien–Meroz ψ_g** module. ψ_g(t) — the
unwrapped velocity-direction angle of the apical tip — is, under hypotheses
H1–H3 (theory.md §3.4–3.5, Bastien & Meroz 2016 Eq. 20), the *direct estimator
of the differential-growth oscillator*, not a numerically convenient detrended
quantity. Tier 2 emits the trajectory-intrinsic ψ_g traits (theory.md §7.3) and
cross-checks ψ_g recovery against Tier 0's spatial `principal_axis_angle`.

## What Changes

- Graduate `psi_g` from stub to implementation: replace the
  `NotImplementedError` stub with
  `psi_g.compute(trajectory_df, cadence_s, constants=None) -> pd.DataFrame`
  (renames the stub callable `compute_psi_g` → `compute`, matching
  `kinematics.compute` / `qc.compute` / `nutation.compute`). **BREAKING** for the
  stub contract only (no released consumer — `psi_g` raised `NotImplementedError`).
- Emit **4** self-contained Tier 2 trait columns: `T_psig_median_s` (s),
  `delta_E_amplitude_proxy_px_per_frame` (px/frame), `handedness` (int
  {−1,0,+1}), `helix_signed_area_px2` (px²). The 5th §7.3 trait
  `psig_long_consistency` (cross-tier T_psig↔T_nutation correlation) is **deferred**
  to a follow-up issue — it structurally couples Tier 2 to Tier 1.
- Add geometry helper `_geometry.compute_signed_area(x, y) -> float` (y-down
  Shoelace; sign agrees with `handedness`).
- Reuse the locked `_geometry.compute_psi_g` (PR #2 atan2 convention) and the
  `temporal_cwt` primitives (PR #5/#6) — compose, do not reinvent.
- **No new `ConstantsT` fields**; `_CONSTANTS_VERSION` stays **5**. No
  `PIPELINE_UNIT_VOCABULARY` change (all 4 units already present).
- Cross-tier consistency validation against `principal_axis_angle` on the
  Nipponbare plate-001 fixture (GREEN-phase reconciliation) + a synthetic
  convention-lock (RED).
- **Three recorded deviations from theory.md §7.3 / §6.3** (theory.md is patched
  in this PR to match, preserving the original wording in an Appendix B
  correction note): (1) `handedness` drops §7.3's literal "COI-masked range" (COI
  is a CWT-edge concept that mis-applies to a raw angular displacement and
  introduced a sign-flip bug); (2) `delta_E` is **px/frame**, not §7.3's
  px·hr⁻¹ (drops the cadence factor — Tier 0 velocity convention; `px/s` ∉ the
  units vocabulary); (3) ψ_g conditioning is **SG-detrend** (the SG residual),
  not §6.3's literal "smooth" (the residual is the oscillation a period-CWT
  needs; no public smoothing primitive exists; reuses Tier 1's primitive).

## Impact

- **Affected specs:** `circumnutation` — MODIFIED `Requirement: Package layout`
  (stub→impl transition for `psi_g`); ADDED `Requirement: Tier 2 ψ_g trait
  emission API`.
- **Affected code:** `sleap_roots/circumnutation/psi_g.py` (stub → impl),
  `sleap_roots/circumnutation/_geometry.py` (new `compute_signed_area`),
  `tests/test_circumnutation_psi_g.py` (new), `tests/test_circumnutation_foundation.py`
  (stub-table migration), `docs/circumnutation/theory.md` (§7.3 handedness +
  delta_E rows, §6.3 conditioning note, + Appendix B corrections preserving the
  original wording).
- **Consumed test fixture (read-only, Git LFS):**
  `tests/data/circumnutation_nipponbare_plate_001/plate_001_greyscale.tracked_proofread.slp`
  for the cross-tier reconciliation test.
- **Deferred trait ownership:** `psig_long_consistency` (the 5th §7.3 trait,
  cross-tier Tier 1 × Tier 2 correlation) is folded into the already-roadmapped
  **PR #13** Layer-3 work (`T_nutation ↔ T_psig ±5%`), which owns both the trait
  and the consistency test — **no separate follow-up issue** (avoids a duplicate
  of PR #13). This PR only files the **PR #7 tracking issue**.
