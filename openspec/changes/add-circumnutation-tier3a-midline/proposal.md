## Why

Program PR #8 (epic #197, `docs/circumnutation/roadmap.md`) graduates the
`midline` stub into the **Tier 3a tip-trail-as-midline reconstruction**
machinery. For an apically-growing organ whose tissue past the elongation zone
does not reshape, the curve of past tip positions IS the organ midline
(theory.md §6.1); reconstructing it parameterized by arc length `s(τ)=∫|v|dσ`,
with per-frame curvature `κ` and tip speed `|v|`, is the substrate PR #9
(spatial CWT on `κ(s)`, `L_gz`/`L_c`) and PR #10 (Tier 3 trait emission) compose
on. This is a **machinery** PR mirroring PR #5 `temporal_cwt` (input-validating,
frozen `attrs` Result, determinism contract, **no trait emission**), NOT the
per-track `compute()` shape of PR #6/#7.

## What Changes

- Graduate `midline` from stub to implementation: replace the
  `NotImplementedError` stub with
  `midline.reconstruct(x, y, cadence_s, sg_window=None, constants=None) -> MidlineResult`
  (the callable KEEPS its name `reconstruct` — no rename). **BREAKING** for the
  stub contract only (no released consumer — `midline` raised
  `NotImplementedError`). The signature adds `sg_window=None` to the PR #1
  stub-table signature `reconstruct(x, y, cadence_s, constants=None)`.
- New frozen `MidlineResult` dataclass (`@attrs.define(frozen=True, slots=False,
  kw_only=True)`, mirroring `ScaleogramResult`/`RidgeResult`) with 7 per-frame
  arrays (`frame_indices`, `x_smooth_px`, `y_smooth_px`, `speed_px_per_frame`,
  `arc_length_px`, `curvature_px_inv`, `velocity_sub_noise_mask`) and 6
  provenance scalars (`cadence_s`, `sg_window`, `sg_degree`,
  `sigma_v_px_per_frame`, `noise_mask_k`, `is_degenerate`). **Frame-parameterized
  / cadence-independent** (velocity in **px/frame**, the program convention;
  `px/s` is deliberately not used — theory §10 Appendix B).
- New shared SG-derivative primitive
  `_noise.compute_sg_derivative(x, window, polynomial_order, deriv, delta=1.0, mode="interp")`
  (covers deriv 0/1/2 from one `savgol_filter` polynomial; reuses
  `compute_sg_detrended`'s window/order boundary-validation pattern).
- New pure-geometry curvature helper
  `_geometry.compute_path_curvature(x_dot, y_dot, x_ddot, y_ddot) -> np.ndarray`
  (`κ = (ẋÿ−ẏẍ)/|v|³`, px⁻¹), sibling to `compute_psi_g`/`compute_signed_area`,
  with the load-bearing sign convention pinned by an **absolute hand-built-input
  test** and a y-down disambiguation in the docstring.
- Reuse the imported `temporal_cwt._validate_cadence_s` (do not re-implement);
  compose `scipy.signal.savgol_filter` + `scipy.integrate.cumulative_trapezoid`
  (module-qualified imports per the program rule; convert `_noise.py`'s lone
  bare `from scipy.signal import savgol_filter` in the same PR).
- **No new `ConstantsT` fields**; `_CONSTANTS_VERSION` stays **5** (reuses
  `NOISE_MASK_K`, `SG_WINDOW_SHORT`, `SG_DEGREE`; the two new helpers are
  functions, not constants). No `PIPELINE_UNIT_VOCABULARY` change.
- Determinism canary `scripts/circumnutation/capture_midline_canary.py` locking a
  **closed-form circle oracle** (`κ ≡ 1/R`) plus a synthetic-generator drift
  detector, asserted cross-OS at `atol=1e-9` (measured full-pipeline ULP
  propagation ≈1e-14; PR #6/#7's looser 1e-6 was a coverage argument for a
  4-path scipy stack and does not transfer to this narrow savgol+cumtrapz path).
- Validation on the **real plate-001** Nipponbare proofread fixture (curvature
  finite & physically bounded on unmasked frames; arc length monotonic; mask
  fraction in a recorded band; cross-tier `arc_length_px[-1] ≈` Tier 0 path
  length within ~5%) — NOT just the synthetic generator (PR #7 lesson).

**Two deliberate scope splits (documented so PR #9/#10 inherit a clean handoff):**

1. **No `L_gz` growth-zone mask** (CC-1) — PR #8 computes `κ(s)` along the FULL
   trail; the `L_gz` apical-region mask is built in PR #10 after PR #9 detects
   the peak. `velocity_sub_noise_mask` (per-FRAME, time-domain `|v|≤k·σ_v`) is a
   DIFFERENT object from the `L_gz` mask (per-ARC-LENGTH, apical region) — they
   share the `σ_v`/`NOISE_MASK_K` vocabulary but must not be conflated.
2. **No uniform-arc-length resampling** — PR #8 emits `curvature_px_inv` on the
   native NON-UNIFORM `arc_length_px` grid; **PR #9 owns** the resample to the
   uniform `(kappa, ds)` grid its locked stub
   `spatial_cwt.compute_scaleogram(kappa, ds, constants=None)` requires (where
   the `ds`/spatial-Nyquist decision intrinsically lives).

## Impact

- **Affected specs:** `circumnutation` — MODIFIED `Requirement: Package layout`
  (stub→impl transition for `midline`: impl modules 6→7, stub modules 5→4);
  ADDED `Requirement: Tier 3a midline reconstruction API` (the `reconstruct`
  contract + `MidlineResult` + the two new helpers, with the
  `compute_path_curvature` absolute sign-anchor scenario). The frozen
  `Requirement: Tier 0 helper modules` is NOT modified (the new helpers are
  spec'd inside the new Tier 3a requirement, per the PR #7 `compute_signed_area`
  precedent).
- **Affected code:** `sleap_roots/circumnutation/midline.py` (stub → impl),
  `sleap_roots/circumnutation/_noise.py` (new `compute_sg_derivative` +
  bare-import cleanup), `sleap_roots/circumnutation/_geometry.py` (new
  `compute_path_curvature`), `scripts/circumnutation/capture_midline_canary.py`
  (new), `tests/test_circumnutation_midline.py` (new),
  `tests/test_circumnutation_foundation.py` (stub-table migration),
  `docs/circumnutation/theory.md` (§6.2 y-down curvature-sign clarification +
  Appendix B correction note preserving the original "left turn" wording),
  `docs/changelog.md`.
- **Consumed test fixture (read-only, Git LFS):**
  `tests/data/circumnutation_nipponbare_plate_001/plate_001_greyscale.tracked_proofread.slp`
  for the real-data + cross-tier reconciliation tests.
- **Deferred to later PRs (no separate follow-up issue):** uniform-arc-length
  resampling + `ds` (PR #9); `L_gz` growth-zone mask + Tier 3 trait emission
  (PR #10). This PR files only the **PR #8 tracking issue**.
