# Design — add-circumnutation-tier2-psi-g (Tier 2 ψ_g)

> Full design rationale + the 3-round review reconciliation log lives in
> `docs/superpowers/specs/2026-06-05-add-circumnutation-tier2-psi-g-design.md`.
> This file is the OpenSpec-format summary of the load-bearing decisions.

## Context

Program PR #7 (epic #197) graduates the `psi_g` stub into the Tier 2
Bastien–Meroz ψ_g module. ψ_g(t) = unwrapped `atan2(dx, dy)` of the apical tip
(theory.md §3.5, Eq. 20) is the direct estimator of the differential-growth
oscillator under H1–H3. The module mirrors the merged Tier 1 `nutation.py`
shape and reuses the locked `_geometry.compute_psi_g` (PR #2) + `temporal_cwt`
primitives (PR #5/#6). The design was hardened across 3 rounds of multi-reviewer
critical review (11 reviewer-passes); two reviews surfaced reachable bugs
(min-length crash, stationary-tip spurious period) and a sign-convention error,
all reconciled below.

## Goals / Non-Goals

- **Goals:** emit the 4 self-contained §7.3 traits; reuse not reinvent; no new
  constants; cross-tier consistency check vs Tier 0 `principal_axis_angle`;
  strict TDD with RED→GREEN commit pairs.
- **Non-Goals:** `psig_long_consistency` (couples to Tier 1 — deferred to a
  follow-up issue); CSV/sidecar wiring; spatial/midline/parametric tiers; any
  `_CONSTANTS_VERSION` bump or `PIPELINE_UNIT_VOCABULARY` change.

## Decisions

- **D1 — Public API:** `psi_g.compute(trajectory_df, cadence_s, constants=None)`,
  mirroring `nutation.compute` minus the `coordinate` kwarg (ψ_g is inherently 2-D
  from raw `tip_x`/`tip_y`). Renames the stub callable `compute_psi_g` → `compute`.
- **D2 — Scope = 4 self-contained traits:** `T_psig_median_s`,
  `delta_E_amplitude_proxy_px_per_frame`, `handedness`, `helix_signed_area_px2`.
  `psig_long_consistency` deferred. *Alternative considered:* ship all 5 with an
  internal T_nutation recompute or an optional `nutation_df` — rejected as
  cross-tier coupling beyond a single-tier PR.
- **D3 — Units:** period in seconds (`s`); speed proxy in **px/frame** (Tier 0's
  cadence-independent velocity convention). Column names adopt the **#222** target
  suffix convention (`_s`, `_px_per_frame`, `_px2`) — note this *knowingly differs*
  from merged Tier 1's un-suffixed `T_nutation_median` (also seconds), i.e. PR #7
  adopts the #222 suffix style ahead of the program-wide retrofit rather than
  copying Tier 1's current naming. *Alternative:* px/s — rejected (`px/s` ∉
  `PIPELINE_UNIT_VOCABULARY`; would force a foundation change). All 4 units already
  in the vocabulary.
- **D4 — ψ_g conditioning = SG-detrend** (`compute_sg_detrended`, window 23) for
  the `T_psig` CWT path only. *Alternative:* SG-smooth-only (theory §6.3 literal)
  — rejected: no public smoothing primitive exists, and smooth-only leaves
  gravitropic drift that biases `T_psig` (~39% of longest-scale COI-interior
  unmasked). Detrend reuses the existing primitive and matches Tier 1.
- **D5 — `handedness` = net rotation over all finite frames, no COI:**
  `int(np.sign(ψ_g[-1] − ψ_g[0]))` with a `1e-9 rad` numerical-zero guard.
  **Deviation from theory.md §7.3's literal "COI-masked range"** (theory patched
  this PR). *Why:* COI-masking coupled handedness to the conditioned CWT ridge and
  introduced a non-contiguity sign-flip bug (per-frame argmax → COI gaps → endpoint
  diff across a gap → wrong sign). COI is a CWT-edge concept; a raw angular
  displacement has no edge contamination. Makes "conditioning affects only
  `T_psig`" literally true and the `1e-9` guard defensible (raw atan2 ~`1e-12`
  reproducible).
- **D6 — `helix_signed_area_px2`:** new `_geometry.compute_signed_area`, y-down
  **negated** Shoelace `0.5·Σ(x_{i+1}·y_i − x_i·y_{i+1})`, applied to the
  **growth-detrended** tip trajectory (per-axis linear-trend subtraction in
  `psi_g.compute`). *Why detrend (post-approval self-review fix):* the raw
  Shoelace of an open growing root is dominated by the growth ribbon —
  `sign(raw_area) == handedness` held only **1/6** on real plate-001; after the
  per-axis linear growth-detrend it holds **≥5/6**. The helper itself stays a
  pure raw Shoelace (closed-orbit sign pinned by the absolute anchor); the
  detrend is applied by the caller. The synthetic generator is a planar
  wobble-on-growth (no true 2-D circulation → ~0 detrended area), so the
  sign-agreement is validated on real plate-001 data, not the synthetic. So
  `sign(area) ==
  handedness`. The load-bearing fact is the **sign-agreement invariant**, not any
  single orbit's chirality label (the word "counterclockwise" is frame-ambiguous —
  anchor on the `dψ_g/dt` sign instead). Two worked anchors, both verified against
  the real `compute_psi_g`/`compute_signed_area`: (i) the orbit `x=cos t,
  y=−sin t` → net `dψ_g/dt > 0` → `handedness=+1`, standard Shoelace `−3.14`,
  negated area `+3.14` → `sign(area)=+1=handedness`; (ii) the absolute hand-built
  anchor `[0,1,1,0],[0,0,1,1]` → net `Δψ_g=−π` → `handedness=−1`, negated area
  `−1.0` → `sign(area)=−1=handedness`. Pinned by the absolute-anchor test
  (`compute_signed_area([0,1,1,0],[0,0,1,1]) == −1.0`) AND an agreement test.
- **D7 — Min-length + zero-energy guards:** `T_psig` requires `N ≥ 24` (ψ_g len ≥
  `SG_WINDOW_DETREND=23`); `3 ≤ N < 24` → `T_psig=NaN`, raw traits defined.
  Stationary/straight-growth (zero detrended energy) → `T_psig=NaN` via a
  `np.allclose(detrended, 0.0)` guard (else `argmax`-over-zeros yields a spurious
  `2·cadence` period with no warning). Mirrors `nutation.py:418-428` for the
  CWT short-input path.
- **D8 — Cross-tier check:** reconciled identity `circular_mean(ψ_g) ≈ π/2 −
  principal_axis_angle` (from `atan2(a,b)=π/2−atan2(b,a)`), branch-cut-safe via
  `wrap_to_pi`. Two-fixture RED convention-lock (angle identity at `amplitude=0`;
  handedness at `amplitude>0` — they cannot share a fixture: oscillation biases
  `circular_mean` ~1.7e-3) + a plate-001 GREEN-phase reconciliation (skip NaN
  `principal_axis_angle` tracks; `≥N/6` count; tolerance captured at GREEN).
- **D9 — No gate, ungated emission:** Tier 2 never reads `is_nutating`; downstream
  `LEFT JOIN` masks non-nutating tracks. `handedness` ⟂ `is_nutating` (directional
  sign vs existence/SNR gate).

## Risks / Trade-offs

- **ψ_g ramp degeneracy** → a pure circular orbit makes ψ_g ramp (no period);
  real growth-dominated roots give oscillating ψ_g. Documented as the H1 regime
  assumption.
- **Cross-OS determinism (CC-6):** float traits inherit Tier 1's `atol=1e-6`
  floor (SG-detrend→scipy-CWT); integer `handedness` exact (CWT-free).
- **spec↔theory drift:** the §7.3 deviation is recorded in BOTH the ADDED spec
  requirement rationale AND a theory.md §7.3 patch (this PR) so they don't
  contradict.
- **Coverage honesty:** two round-2 defensive branches (post-detrend finite-check,
  `except ValueError`) are unreachable given the length guard → `# pragma: no
  cover` with invariant comments, not contrived tests.

## Migration Plan

`psi_g` had no released consumer (it raised `NotImplementedError`). The stub→impl
rename `compute_psi_g` → `compute` updates only the foundation stub tables. Theory
doc §7.3 is patched in the same PR. Rollback = revert the PR (no schema/state
migration).

## Open Questions

- The plate-001 GREEN-phase tolerance + pass-count are captured at GREEN, not
  pre-known (documented as reconciliation, mirroring `_DERR_MATCH_*`), but MUST
  clear a pre-committed floor (`N ≥ 2`, `tol ≤ 0.35 rad`) and record the observed
  deviation distribution so the pass is auditable, not self-fulfilling.
- `psig_long_consistency` is deferred to the already-roadmapped **PR #13**
  Layer-3 (`T_nutation ↔ T_psig ±5%`), which is resolved to own **both** the §7.3
  trait emission AND the cross-tier consistency test (single-sourced — no separate
  follow-up issue, to avoid fragmenting ownership). The implementation home (new
  cross-tier module vs `psi_g.compute` optional `nutation_df`) is PR #13's
  decision.
