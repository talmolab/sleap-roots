# Design — add-circumnutation-tier3a-midline (Tier 3a midline)

> Full design rationale + the 2-round critical-review reconciliation log lives in
> `docs/superpowers/specs/2026-06-08-add-circumnutation-tier3a-midline-design.md`.
> This file is the OpenSpec-format summary of the load-bearing decisions.

## Context

Program PR #8 (epic #197) graduates the `midline` stub into the Tier 3a
tip-trail-as-midline reconstruction machinery (theory.md §6.1/§6.2). It is a
**machinery** PR mirroring the merged PR #5 `temporal_cwt` (input-validating,
frozen `attrs` Result, determinism contract, **no trait emission**) — NOT the
per-track `compute()` shape of PR #6/#7. The design was hardened across 2 rounds
of multi-reviewer critical review (5 reviewers round 1, all numerically verified
on real plate-001), which overturned the empirical premise of the velocity-mask
prose, switched units to px/frame, tightened the determinism floor, and moved
stationary-detection to the raw input — all reconciled below.

## Goals / Non-Goals

- **Goals:** implement `reconstruct(...) -> MidlineResult`; arc length `s(τ)`,
  curvature `κ(s)`, speed `|v|`, velocity-bandpass mask; SG smoothing before
  differentiation; reuse not reinvent; strict TDD; no new constants; determinism
  canary; real-plate-001 + cross-tier validation.
- **Non-Goals:** the `L_gz` growth-zone mask (CC-1 → PR #10); uniform-arc-length
  resampling + `ds` (→ PR #9); any trait column / `trajectory_df` / groupby; any
  `_CONSTANTS_VERSION` bump or `PIPELINE_UNIT_VOCABULARY` change; spatial CWT,
  parametric, pipeline tiers.

## Decisions

- **D1 — Public API:** `midline.reconstruct(x, y, cadence_s, sg_window=None,
  constants=None) -> MidlineResult` (keeps the callable name `reconstruct`).
  `sg_window` adds to the PR #1 stub signature; the window/degree asymmetry
  (window per-call, degree from `SG_DEGREE`) is deliberate — window is the
  smoothing-scale knob callers sweep (`smooth_ridge` precedent), degree is a
  fixed methodological choice. `cadence_s` validated via the imported
  `temporal_cwt._validate_cadence_s` and stored as provenance, but the core
  outputs are **cadence-independent** (frame-parameterized).
- **D2 — `MidlineResult`** frozen `attrs`: 7 per-frame arrays + 6 provenance
  scalars (§4 of the full draft). **One** `curvature_px_inv` array (curvature is
  parameterization-invariant, so `κ_path(τ)=κ(s(τ))` are bit-identical; pairing
  with `frame_indices` = time view, with `arc_length_px` = arc view, the latter
  non-uniformly sampled). The `is_degenerate` flag + provenance scalars are a
  **deliberate divergence** from `ScaleogramResult`/`RidgeResult` (which carry
  only data) — an all-NaN Result + explicit flag is the right degenerate output
  for a per-track reconstruction primitive (justified like `RidgeResult.powers`).
  Also adds **`eq=False`** to the attrs decorator (a second deliberate improvement
  over the template): with ndarray fields the generated `__eq__` on `r1 == r2` is
  ill-defined (a multi-element array has no unambiguous truth value; `==` can raise
  `ValueError: ambiguous truth value`); `eq=False` makes `==` identity and forces
  field-by-field `np.array_equal` comparison (the determinism-test pattern).
- **D3 — Units = px/frame (program convention):** velocity `speed_px_per_frame`,
  arc length integrated with `dx=1` frame (px), curvature `px⁻¹` (time-unit
  invariant). `px/s` deliberately NOT used (theory §10 Appendix B). *Alternative:*
  px/s — rejected (program-rejected; absent from the units vocabulary).
- **D4 — SG analytic derivatives:** one `savgol_filter` polynomial yields deriv=0
  (smoothed x,y), deriv=1 (ẋ,ẏ), deriv=2 (ẍ,ÿ) — self-consistent, the textbook
  "SG before differentiation" (§6.2). New shared primitive
  `_noise.compute_sg_derivative(x, window, polynomial_order, deriv, delta=1.0,
  mode="interp")` reuses `compute_sg_detrended`'s validation; **additionally**
  validates `0 ≤ deriv ≤ polynomial_order` (scipy silently returns zeros for
  `deriv>order` — a silent-wrong-answer hazard). *Alternative:* smooth + np.gradient
  — rejected (two-stage, nested 2nd-deriv amplifies noise, less self-consistent).
- **D5 — Arc length = cumulative trapezoid of |v|:**
  `scipy.integrate.cumulative_trapezoid(speed, dx=1.0, initial=0)` — self-consistent
  with the SG-derived speed. *Alternative:* cumulative segment distance — rejected
  (decoupled from the derivative; immaterial ~2.5% difference on real data).
- **D6 — Curvature helper in `_geometry`:** pure-geometry
  `compute_path_curvature(ẋ, ẏ, ẍ, ÿ) -> κ` (`(ẋÿ−ẏẍ)/|v|³`), sibling to
  `compute_psi_g`/`compute_signed_area`. Sign **anchored by an absolute
  hand-built test** (`[1],[0],[0],[1] → +1.0`; CCW circle `→ +1/R`; CW `→ −1/R`)
  — NOT an internal-agreement test. **y-down note (load-bearing):** the formula
  is the standard y-up math formula; in the y-down image frame `+κ` is a
  visual/clockwise-right turn (anchor on the sign, not the word — `_geometry.py`
  precedent). **Cross-helper collision (publication-risk, per scientific review):**
  the ψ_g family (`compute_psi_g`/`compute_signed_area`/`handedness`) uses the
  swapped `atan2(dx, dy)`, so for the SAME loop `sign(κ) == −handedness` — documented
  in the docstring + theory §6.2 patch + pinned by a cross-helper sign test, so a
  PR #9/#10 chirality trait can't silently invert. Length-mismatch → ValueError
  (sibling guard).
- **D7 — σ_v = std(speed), self-contained:** mask `velocity_sub_noise_mask` True ⇔
  `speed ≤ NOISE_MASK_K·σ_v` (matches `coi_mask` polarity; `κ[~mask]`).
  **Empirically measured on real plate-001:** masks ~40–60% of frames. This is the
  *data-specific behavior*, NOT a settled identity (per scientific review): the rice
  tip moves only ~3× the localization-noise floor, and `std(speed)` happens to land
  near `√2·σ_pos` HERE — a coincidence of plate-001's parameters, not a general law
  (`std(speed)` is statistical-relative-to-the-spread and largely SNR-insensitive in
  this regime). *Round-1 corrected the false "flags ~nothing" prose.* The resulting
  ~50%-sparse, non-uniform `κ(s)` makes gap-handling a PR #9 concern (its spatial-CWT
  validity depends on it); PR #10 may refine the σ_v definition when the mask is first
  consumed for a trait. PR #8 only emits the mask as data + provenance (nothing
  irreversible is baked in).
- **D8 — Split degenerate policy:** RAISE (field-named) on contract violations
  (non-finite NaN/±inf — rejected not dropped, since SG/cumtrapz need uniform
  spacing; wrong dtype/shape; `len(x)≠len(y)`; bad `cadence_s`/`sg_window`); the
  non-finite check precedes the degenerate gate. GRACEFUL all-NaN `MidlineResult`
  (`is_degenerate=True`, no raise/warning) for `n=0`, `n<sg_window`, and
  **raw-stationary** (`ptp(x)==0 and ptp(y)==0`, detected on the RAW input because
  post-SG speed is float dust, never exactly 0). Degenerate gate returns BEFORE
  any `np.std`/`hypot`/`cumtrapz` (`n==0` first disjunct — `np.ptp([])` raises).
  Curvature computed under `errstate(divide,invalid,over)` + post-`~isfinite→NaN`
  sweep (no inf, no RuntimeWarning). On the graceful path the float arrays are NaN,
  `frame_indices=arange(n)`, and `velocity_sub_noise_mask=np.zeros(n, bool)` (all-False
  — a bool array can't hold NaN). ALL field-named validation runs first and
  unconditionally, so `n=0`-with-bad-`cadence_s` RAISES (validation wins over the
  graceful path). **Deliberate gate-divergence (R4-BLOCKING-3):** `reconstruct` gates
  stationarity on `ptp(x)==0 and ptp(y)==0` whereas `_geometry.project_to_growth_axis
  _perpendicular` gates on zero NET displacement — different physical questions ("did
  the tip move at all" vs. "is there a growth axis"). A closed loop (e.g. a full
  circle: net-disp 0 but `ptp=2R`) HAS a well-defined curvature midline but no growth
  axis, so midline correctly uses `ptp`; the canary circle exercises exactly this
  healthy-on-net-zero path.
- **D9 — Determinism (CC-6):** same-process atol=0; cross-OS **atol=1e-9, rtol=0**
  (measured full-pipeline ULP propagation ≈1e-14; well-conditioned savgol lstsq
  `cond(A)≈11–92`). *Alternative:* 1e-6 (PR #6/#7) — rejected (a coverage argument
  for a 4-path scipy stack; would mask ~100%-of-range curvature regressions).
  Canary locks a **closed-form circle (`κ≡1/R`)** + the synthetic generator.
- **D10 — No new constant, no version bump (stays 5).** `mode="interp"` is a
  documented default, not a field. The two helpers are functions.

## Risks / Trade-offs

- **Velocity mask flags a data-dependent ~40–60% of real frames** → sub-noise frames SHOULD be masked,
  but the fraction is data-dependent and SNR-insensitive (not a noise-discriminating
  signal); PR #9's spatial CWT inherits a ~50%-sparse, non-uniform `κ(s)` and OWNS
  gap-handling (its spectral validity depends on it). PR #10 may refine σ_v.
- **Cross-helper sign collision** `sign(κ) == −handedness` → documented + tested so a
  PR #9/#10 chirality trait doesn't invert (publication risk).
- **Informational (not fixed here):** the PR #9 `spatial_cwt` stub's actual signature
  carries extra `wavelet=None, scale_range=None` kwargs beyond the spec stub-table's
  `compute_scaleogram(kappa, ds, constants=None)` — a pre-existing spec↔code drift in
  PR #9's stub, out of PR #8's scope; flagged for whoever writes PR #9.
- **Cross-OS determinism:** float fields hold to `atol=1e-9` (~1e-14 headroom);
  `frame_indices` (int) + `is_degenerate` (bool) exact.
- **spec↔theory drift:** the y-down curvature-sign note is recorded in the new
  spec requirement AND a theory.md §6.2 patch (+ Appendix B) so they don't
  contradict.
- **`_noise.py` scipy-import cleanup** touches all 5 existing `savgol_filter(` call
  sites across 3 functions
  (`compute_sg_detrended`/`compute_sg_residual_xy`/`compute_msd_residual_xy`);
  mechanical, consumers (`kinematics`/`qc`/`nutation`/`psi_g`) re-run in the
  per-pair gate.

## Migration Plan

`midline` had no released consumer (it raised `NotImplementedError`). The
stub→impl change (callable name unchanged) updates the foundation stub tables +
the spec Package-layout prose ATOMICALLY with the first non-raising commit (PR #7
BLOCKING precedent) so the foundation suite never goes red. theory.md §6.2 is
patched in the same PR. Rollback = revert the PR (no schema/state migration).

## Open Questions

- Real plate-001 cross-tier tolerance + mask-fraction band are captured at GREEN
  against pre-committed floors (finite; monotonic; `max|κ[~mask]| < 1 px⁻¹`;
  `0.1 < mask_frac < 0.85`; arc within ~5% of Tier 0 path length, plus the robust
  `arc ≤ L` invariant) with the
  observed values recorded — auditable, not self-fulfilling (PR #7 discipline).
- Uniform-arc-length resampling + `ds` is owned by PR #9 (the `(kappa, ds)` stub
  presumes a uniform grid); `L_gz` mask + Tier 3 traits by PR #10. No separate
  follow-up issue.
