# Design — `add-circumnutation-tier2-psi-g` (Program PR #7, Tier 2)

**Status:** brainstorm draft, revised after 4-reviewer critical pass (pre-OpenSpec-proposal)
**Date:** 2026-06-05
**Branch:** `add-circumnutation-tier2-psi-g`
**Epic:** #197 · **Roadmap row:** PR #7 (`docs/circumnutation/roadmap.md`)
**Theory anchors:** `docs/circumnutation/theory.md` §3.5 (ψ_g, Bastien & Meroz 2016 Eq. 20), §6.3 (tier structure + SG conditioning), §7.3 (Tier 2 trait table), §3.7 (trochoid / H1-failure)

> **Revision note.** §13 is a reconciliation log mapping every BLOCKING/IMPORTANT
> finding from the 4-reviewer critical pass to its resolution here. Two
> brainstorm decisions were reversed on review evidence: (a) ψ_g conditioning is
> now **SG-detrend** (not smooth-only) — only a detrend primitive exists and
> smooth-only risks gravitropic-drift bias; (b) `delta_E` is now **px/frame**
> (not px/s) — Tier 0's velocity convention, and px/s is absent from the units
> vocabulary.

---

## 1. Purpose

Emit **Tier 2 Bastien–Meroz ψ_g traits** per track. ψ_g(t) is the unwrapped
velocity-direction angle of the apical tip — under hypotheses H1–H3 it is the
*direct estimator of the differential-growth oscillator* (theory §3.5), not a
numerically-convenient detrended quantity. Applying the temporal CWT to ψ_g(t)
recovers the spectral content of that oscillator.

PR #7 ships a self-contained, single-tier module mirroring the merged Tier 1
module (`nutation.py`, PR #6). It reuses the locked geometry helper
`_geometry.compute_psi_g` (PR #2, atan2 convention) and the temporal-CWT
primitives (`temporal_cwt`, PR #5). It introduces **no new `ConstantsT` fields**
and emits **4 of the 5** §7.3 traits; the cross-tier `psig_long_consistency`
trait is deferred (§7).

## 2. Scope

### In scope
- New public API `sleap_roots.circumnutation.psi_g.compute(trajectory_df, cadence_s, constants=None) -> pd.DataFrame`, **graduating `psi_g` from stub to implementation** (renames the stub callable `compute_psi_g` → `compute`, matching `kinematics.compute`/`qc.compute`/`nutation.compute`).
- New geometry helper `_geometry.compute_signed_area(x, y) -> float` (y-down-corrected Shoelace), sibling to `compute_psi_g`.
- 4 Tier 2 trait columns (§4).
- Cross-tier empirical consistency validation against Tier 0 `principal_axis_angle` (§6).
- Foundation-test migration (§9): move `psi_g` out of stub tables into the implementation tables.

### Out of scope (explicit non-goals)
- `psig_long_consistency` (§7.3 trait #2) — requires Tier 1's `T_nutation`; deferred to a follow-up issue (§7 has the pre-drafted title + scope).
- Any new `ConstantsT` field / `_CONSTANTS_VERSION` bump (stays 5).
- Any `PIPELINE_UNIT_VOCABULARY` change — all 4 trait units (`s`, `px/frame`, `int`, `px²`) already exist in it.
- Spatial CWT, midline, parametric tiers (PRs #8+).
- Multi-plate empirical validation beyond plate-001 (cf. #220).
- Pipeline orchestration / CSV+sidecar emission wiring (later PRs).

## 3. Public API & module shape

```python
def compute(
    trajectory_df: pd.DataFrame,
    cadence_s: float,
    constants: Optional[ConstantsT] = None,
) -> pd.DataFrame:
    ...
```

Differences from `nutation.compute`:
- **No `coordinate` kwarg** (and no `_check_coordinate`). ψ_g is computed from the
  raw 2D tip trajectory (`tip_x`, `tip_y`) via `atan2(dx, dy)`; there is no
  1D-projection choice.

Shared with `nutation.compute`:
- Per-track `groupby(list(_IDENTITY_5_TUPLE), dropna=False, sort=False)` →
  `_compute_one_track` → identity+traits row dict → per-plant template merge via
  `_build_per_plant_template_from_df` → identity dtype coercion (loud `ValueError`)
  → declared column order `ROW_IDENTITY_COLUMNS + _PSIG_TRAIT_COLUMNS`.

**Validators (disclosed reuse vs new — Reviewer-2 F2).** `_validate_trajectory_df`
is shared (`_types.py`) and reused; it already enforces `tip_x`/`tip_y` column
*presence* for all callers (per-row *finiteness* is not enforced — handled in
§3.1). For `cadence_s`, reuse the importable `temporal_cwt._validate_cadence_s`
rather than copying nutation's private `_check_cadence_s`. For `constants`,
`psi_g` defines a small `_check_constants` that (a) accepts `None`/`ConstantsT`,
(b) validates the SG fields it consumes (`SG_WINDOW_DETREND` odd & >
`SG_DEGREE`, `SG_DEGREE` ≥ 0) with field-named `ValueError`s, and (c) defers CWT
field validation to `compute_scaleogram` (which runs `_validate_cwt_constants`).
It does **not** call `_validate_nutation_constants` (those are Tier-1 fields
psi_g doesn't use).

`constants` is threaded to `compute_sg_detrended` (SG window/degree) and
`temporal_cwt.compute_scaleogram` (CWT fields) so overrides propagate — this is
why `psi_g.compute` joins `IMPLEMENTATIONS_WITH_CONSTANTS_KWARG`.

### 3.1 `_compute_one_track` pipeline

Operates on raw `tip_x`/`tip_y`. **Conditioning affects only `T_psig_median_s`**;
`handedness`, `delta_E`, `helix_signed_area`, and the cross-tier check all use
the *raw* ψ_g / raw tip coordinates.

1. **Finite-frame extraction + degenerate short-circuit.** Drop rows with
   non-finite `tip_x`/`tip_y` (Reviewer-2 F7: `compute_psi_g` has no NaN guard,
   so mask first). Let `N` = count of finite frames. If `N < 3` (ψ_g needs ≥2
   velocity samples for a derivative), return graceful values:
   `T_psig_median_s=NaN`, `delta_E_amplitude_proxy_px_per_frame=NaN`,
   `handedness=0`, `helix_signed_area_px2=NaN`. A track that is *entirely*
   non-finite is the `N=0` case of this branch (mirrors nutation's all-NaN
   short-circuit). The `<3` count is over **post-finite-mask** rows.
2. **ψ_g(t).** `psi_g = _geometry.compute_psi_g(tip_x, tip_y)` — length `N−1`,
   unwrapped, locked `atan2(dx, dy)` convention. **Reused, never redefined.**
3. **Condition for the CWT (T_psig only).** SG-**detrend** ψ_g:
   `psi_g_detrended = _noise.compute_sg_detrended(psi_g, window=SG_WINDOW_DETREND (=23), polynomial_order=SG_DEGREE (=3))`
   (reuses the existing residual helper — Reviewer-1 F2 / Reviewer-2 F1).
   Removes slow gravitropic drift in the velocity direction; preserves the
   ~1-period nutation oscillation (window ≈ 2 periods, Tier-1 precedent).
4. **Temporal CWT chain** on `psi_g_detrended` (identical primitive composition
   to `nutation._compute_one_track`):
   `compute_scaleogram → extract_ridge → smooth_ridge`.
5. **Emit 4 traits** (§4).

## 4. Trait set (§7.3, restricted to the 4 self-contained traits)

Naming: explicit unit suffixes (issue #222). Periods in seconds (consistent with
Tier 1 `T_nutation_median`); the speed proxy in **px/frame** (Tier 0's
cadence-independent velocity convention — see Reviewer-1 F3 / units-vocabulary
gap). All four units (`s`, `px/frame`, `int`, `px²`) are already in
`PIPELINE_UNIT_VOCABULARY`.

| Column | dtype | Units | Definition |
|---|---|---|---|
| `T_psig_median_s` | float64 | s | `np.nanmedian` of the **smoothed-ridge** periods over **COI-interior** frames (`~smooth_ridge(...).in_coi`), mirroring `nutation.py`'s smoothed-ridge statistic. Guarded: empty-interior **or** all-NaN interior → NaN with **no** `RuntimeWarning` (Reviewer-2 F4). |
| `delta_E_amplitude_proxy_px_per_frame` | float64 | px·frame⁻¹ | `np.median(√(dx²+dy²))` over **all finite** velocity samples (`dx,dy = np.diff` of finite tip coords). **No** COI mask (§7.3 omits it for this kinematic median); **no** `/cadence_s` (px/frame, Tier-0 convention). `= (L/2R)·ΔĖ`, Eq. 21. |
| `handedness` | int64 | int | `int(np.sign(Δψ_g_interior))` where `Δψ_g_interior = ψ_g[last COI-interior] − ψ_g[first COI-interior]` (the **net** unwrapped rotation across COI-interior frames = mean dψ_g/dt × span; avoids the N−2 second-diff alignment, Reviewer-2 F6). Determinism guard (Reviewer-4 F6): `|Δψ_g_interior| < 1e-9 rad → 0` (numerical-zero hygiene matching the CWT determinism atol — **not** a physical deadband). `+1` = counterclockwise. `0` = no net rotation / degenerate / empty interior. |
| `helix_signed_area_px2` | float64 | px² | `_geometry.compute_signed_area(tip_x, tip_y)`; **y-down-corrected** Shoelace so `sign(area) == handedness` (§5). Independent confirmation of handedness. |

Declared order: the 8 `ROW_IDENTITY_COLUMNS` then the 4 trait columns above.

**COI semantics (Reviewer-2 F3 / Reviewer-4 F4).** "COI-interior" uses the
**per-frame** boolean `RidgeResult.in_coi` (shape `(N−1,)`), exactly as
`nutation.py` does (`ridge.periods_s[~ridge.in_coi]`). It does **not** use
`ScaleogramResult.coi_mask`, which is 2-D `(n_scales, N−1)`. `T_psig_median_s`
masks with `~smooth_ridge.in_coi`; `handedness` takes ψ_g endpoints at the first
and last `~in_coi` indices. `delta_E` uses no mask.

**No `is_nutating`-style gate** (decision D1). Tier 2 is self-contained — it never
reads Tier 1's `is_nutating`. Traits are emitted ungated with the same documented
"ridge-of-noise" caveat Tier 1 carries; masking non-nutating tracks is a
downstream `LEFT JOIN` of `is_nutating` on the shared 5-tuple identity — the same
composition point where the deferred `psig_long_consistency` will live.

### Why `handedness` and `is_nutating` are orthogonal
`is_nutating` (Tier 1) is an *existence/SNR gate* — "is there a coherent periodic
oscillation above the Fourier noise floor?". `handedness` (Tier 2) is a
*directional sign*. A non-nutating track can have `handedness=±1` (slow one-way
gravitropic curve); a clean back-and-forth wobble can have `handedness=0`
(net-zero circulation). Different questions → Tier 2 does not gate on
`is_nutating`.

### Degenerate / edge-case table (intentional NaN-vs-0.0 distinction — Reviewer-4 F5)

| Condition | `T_psig_median_s` | `delta_E…_px_per_frame` | `handedness` | `helix_signed_area_px2` |
|---|---|---|---|---|
| `< 3` finite frames (incl. all-NaN track) | NaN | NaN | 0 | NaN |
| Stationary tip, ≥3 frames (zero displacement) | NaN | 0.0 | 0 | 0.0 |
| All-COI ridge (no interior) | NaN (guarded) | (defined) | 0 (empty interior) | (defined) |
| Unmatched per-plant template row | NaN (`fillna`) | NaN (`fillna`) | 0 (`fillna(0)`) | NaN (`fillna`) |

The NaN-vs-0.0 split is deliberate: **too-short → NaN** (undefined); **stationary
but long-enough → 0.0** (a genuinely measured zero area / zero speed).

**dtype enforcement (Reviewer-2 F5).** Extend nutation's enforcement loop with a
new branch: `handedness → fillna(0).astype(np.int64)`, alongside `is_nutating →
fillna(False).astype(bool)` (not present here) — concretely psi_g's loop is
`int64` for `handedness`, `float64` for the other three. `0` is a *meaningful*
handedness value, so the `fillna(0)` collapse on unmatched rows is only safe
under the same documented 5-tuple merge invariant nutation relies on (template is
built from the same `trajectory_df` → no unmatched rows in practice; the path is
defensive).

## 5. Geometry helper: `_geometry.compute_signed_area(x, y)`

Lives next to `compute_psi_g` in `_geometry.py` (single home for
convention-critical geometry).

- **y-down-corrected Shoelace:** `A = 0.5 * Σ_i (x_{i+1}·y_i − x_i·y_{i+1})`
  — i.e. the **negative** of the standard `0.5·Σ(x_i·y_{i+1} − x_{i+1}·y_i)`.
  This negation is required so that positive area corresponds to
  counterclockwise-as-a-viewer-sees and therefore to `handedness=+1` under the
  same image-y-down `atan2(dx, dy)` convention `compute_psi_g` encodes.
  **Verified numerically** (Reviewer-1 F6): a screen-CCW orbit yields
  `handedness=+1` and standard-Shoelace `= −3.14`; the negation flips it to
  `+3.14` so `sign(area) == handedness`.
- Returns `float`. `< 3` points → `0.0` (degenerate polygon) at the helper level;
  `psi_g._compute_one_track` maps the `<3`-frame case to NaN *before* calling, so
  the trait reports NaN for too-short tracks.
- NaN-robust: non-finite coordinates → NaN area (caller short-circuits first).
- **Sign is load-bearing** and pinned by two *independent* tests (Reviewer-3 F6 —
  break the joint-flip degeneracy):
  1. **Absolute anchor:** a hand-built vertex sequence with screen-obvious
     orientation (e.g. `x=[0,1,1,0], y=[0,0,1,1]`), assert `compute_signed_area`
     returns the hand-computed sign — independent of any ψ_g machinery.
  2. **Agreement:** a synthetic CCW orbit (`generate_trajectory(handedness=+1)`)
     yields **both** `handedness=+1` **and** `helix_signed_area_px2 > 0`.

## 6. Cross-tier consistency check (validation, not a trait)

ψ_g and `principal_axis_angle` use different atan2 conventions — ψ_g =
`atan2(dx, dy)`, `principal_axis_angle` = `atan2(uy, ux)` (`kinematics.py:269`).
Via `atan2(a, b) = π/2 − atan2(b, a)`, a track growing along unit axis **u**
satisfies the reconciled identity:

> **circular_mean(ψ_g) ≈ π/2 − `principal_axis_angle`  (mod 2π)**

where `circular_mean(ψ_g) = atan2(mean(sin ψ_g), mean(cos ψ_g))`.

**Branch-cut-safe comparison (Reviewer-1 F1).** Assert on the *wrapped* angular
distance, never arithmetic closeness:
`abs(wrap_to_pi(circular_mean(ψ_g) − (π/2 − principal_axis_angle))) < tol`,
with `wrap_to_pi(d) = (d + π) mod 2π − π`.

Two tests with **distinct honesty levels** (Reviewer-3 F2/F3):

1. **Synthetic convention-lock — the true RED test (tight, derivable tolerance).**
   `generate_trajectory` with a *planted* growth-axis angle θ (and planted
   `handedness`); on a noise-free trajectory assert
   `wrap_to_pi(circular_mean(ψ_g) − (π/2 − θ)) ≈ 0` to `atol≈1e-6`, and
   `handedness` equals the planted rotation sign. Include a fixture whose θ lands
   the RHS `π/2 − θ` **outside** `(−π, π]` (e.g. θ ≈ −2.0 rad) to exercise the
   wrap. This pins the convention before any real data. Tests call
   `generate_trajectory(handedness=±1, growth_axis_angle_rad=θ)` directly (the
   shared `_minimal_trajectory_df` hard-codes `handedness=+1` and doesn't forward
   it — Reviewer-3 F1).
2. **Real plate-001 — GREEN-phase fixture-sanity / reconciliation (loose,
   post-hoc band).** Over the 6 tracks of
   `circumnutation_nipponbare_plate_001/…proofread.slp` (mirror the test-local
   `_load_proofread_track_df` loader pattern — Reviewer-4 F9), run
   `kinematics.compute` for `principal_axis_angle` and assert the reconciled
   identity. **Critical guards:**
   - **Skip NaN'd tracks** (Reviewer-3 F2): `principal_axis_angle` is NaN when
     `growth_axis_unreliable` fires (`kinematics.py:218-221,284-290`). Assert on
     the surviving subset with a `≥ N-of-6` count clause (mirrors nutation's
     `test_2H3` `≥3/6` pattern), N pinned at GREEN.
   - **Tolerance is a documented GREEN-phase constant** (e.g.
     `_PSIG_AXIS_RECONCILE_TOL_RAD`), captured from a real run, with a
     "GREEN-phase Reconciliation" docstring — exactly the `_DERR_MATCH_*`
     precedent (`test_circumnutation_nutation.py:101-118`). It is **not** a RED
     assertion against a pre-known threshold.

## 7. Deferred: `psig_long_consistency` (pre-drafted follow-up issue)

§7.3 trait #2, "correlation between `T_psig` and `T_nutation` across the CWT
range" (H1 diagnostic, §3.7 trochoid signature) — structurally couples Tier 2 to
Tier 1's ridge. Deferring keeps PR #7 a clean single-tier module (decision D1).

**Follow-up issue to stage in the vault during proposal (Reviewer-4 F11):**
- **Title:** `circumnutation: psig_long_consistency cross-tier T_psig↔T_nutation correlation (Tier 1 × Tier 2 CWT co-registration)`
- **Scope bullets:** (a) co-register the Tier 1 lateral-coordinate ridge and the
  Tier 2 ψ_g ridge on a common frame/period grid; (b) emit
  `psig_long_consistency` as their correlation over the COI-interior, overlapping
  period range; (c) H1-failure (§3.7) interpretation note; (d) decide ownership
  (new cross-tier module vs an extension of `psi_g.compute` taking optional
  `nutation_df`).

Until then, the downstream `LEFT JOIN` of `is_nutating` is the cross-tier
composition mechanism.

## 8. Constants & versioning

**No new `ConstantsT` fields.** Conditioning reuses `SG_WINDOW_DETREND=23`,
`SG_DEGREE=3`; the CWT reuses the PR #5 fields. `handedness` (sign), `delta_E`
(median), `helix_signed_area` (Shoelace) need none. Therefore
**`_CONSTANTS_VERSION` stays 5**; `test_schema_version_is_1_and_constants_version_is_5`
is **untouched** (no-op confirmation in tasks.md — Reviewer-4 F3). No MODIFY to
the *Module-level constants* requirement.

## 9. Foundation-test migration (PR #6 precedent — Reviewer-2 F8 / Reviewer-4 F1/F2)

Graduating `psi_g` from stub to implementation touches the foundation contract:
- **Remove** `psi_g` from the stub tables: `STUB_MODULES` (drop the
  `("psi_g", "compute_psi_g", 7)` entry) and `STUBS_WITH_CONSTANTS_KWARG` (drop
  `("psi_g", "compute_psi_g")`).
- **Add** `("psi_g", "compute")` to `IMPLEMENTATIONS_WITH_CONSTANTS_KWARG`, **and**
  add the matching `elif module_name == "psi_g":` branch in
  `test_implementation_accepts_constants_kwarg` that builds a valid
  `trajectory_df` and asserts a DataFrame is returned.
- **Add** `psi_g` to the `test_module_logger_is_namespaced` parametrization
  (module declares `logger = logging.getLogger(__name__)`; assert
  `logger.name == "sleap_roots.circumnutation.psi_g"`).
- The stub-`NotImplementedError` scenarios in the foundation spec/tests no longer
  enumerate `psi_g` (§11 MODIFIED Package layout).

## 10. TDD plan (RED → GREEN commit pairs, per #222 retrospective)

Strict red-green-refactor: each scope unit ships a failing test commit
(`test: … (TDD red)`) then an implementation commit (`feat:/fix: … (TDD green)`).
No bundling impl + large test file into one commit (the PR #6 3b7b03e mistake).
Fixup commits ship a test alongside any substantive code change.

Commit-pair units (ordered; ordering validated red-greenable — Reviewer-3 F7):
1. `_geometry.compute_signed_area` + the **two** sign tests (absolute anchor +
   `<3`-point/`NaN` edge), §5.
2. `psi_g.compute` schema/structure: returns DataFrame, 8 identity + 4 trait
   columns in declared order, dtypes (3 float64 + 1 int64). Stub emits the
   degenerate-table default row (so units 2 and 6 share default-emission code).
3. Input-validation boundary: non-DataFrame, invalid `trajectory_df`, bad
   `cadence_s` (reuse `temporal_cwt._validate_cadence_s`), bad `constants`
   (psi_g `_check_constants`, incl. even-`SG_WINDOW_DETREND` override → field-named
   `ValueError`).
4. ψ_g pipeline wiring: `compute_psi_g → compute_sg_detrended → CWT chain`
   produces a finite `T_psig_median_s` on a clean synthetic nutating track;
   assert **no** `RuntimeWarning` on the all-COI path (Reviewer-2 F4).
5. Per-trait math: `T_psig_median_s` (known-period synthetic),
   `delta_E_amplitude_proxy_px_per_frame` (known constant-speed synthetic),
   `handedness` (planted CW/CCW via `generate_trajectory(handedness=±1)`), and the
   handedness↔area agreement.
6. Degenerate/edge cases (the §4 table) via **direct construction** — 2-row
   DataFrames and NaN-injection (à la `test_2F7b`), not `_minimal_trajectory_df`
   (Reviewer-3 F4).
7. Cross-tier: (a) synthetic convention-lock RED test incl. the branch-cut
   fixture; (b) plate-001 GREEN-phase reconciliation with NaN-skip + `≥N/6`
   count (Reviewer-3 F2/F3).
8. Foundation-test migration (§9) + multi-track integration test.

**Coverage (Reviewer-2 F5 / Reviewer-3 F5).** Target ≥ 90% on `psi_g.py` and the
new `_geometry` helper; project gate ≥ 84%. The `fillna`/identity-dtype-coerce
defensive branches are **unreachable via the public API** (template derives from
the same `trajectory_df`); mirror `nutation.py`'s handling — either `# pragma: no
cover` with a comment citing the invariant, or a direct unit test that calls the
merge/coerce path with a hand-built mismatched template. Do not chase coverage
with contrived public-API tests.

## 11. Spec deltas (preview for `/openspec:proposal`)

- **MODIFIED** `Requirement: Package layout` (Reviewer-4 F1/F2): move `psi_g`
  from the stub-module set to the implementation set (recount: 6 implementations,
  5 stubs); remove `psi_g` from the stub-callable table; update the two stub
  scenarios (`Calling each remaining stub raises NotImplementedError`,
  `import … cleanly`) to drop `psi_g`; add a `psi_g.compute callable without
  raising` scenario mirroring the `nutation.compute` one. The
  `IMPLEMENTATIONS_WITH_CONSTANTS_KWARG` + logger-namespace migrations (§9) are
  part of this delta.
- **ADDED** `Requirement: Tier 2 ψ_g trait emission API` — `psi_g.compute`
  (signature, the 4 traits with COI semantics, the degenerate table, the ungated
  self-contained policy, no cross-tier input) and `_geometry.compute_signed_area`
  (y-down Shoelace + sign convention). Scenarios encode: schema/order/dtypes;
  each trait's definition; the degenerate table; the handedness↔area sign lock
  (absolute + agreement); the cross-tier reconciled identity (synthetic RED +
  plate-001 GREEN); the **happy-path logging contract** (§ below); CC-6
  determinism.
- **No MODIFY** to *Module-level constants* (§8) or *Per-module logger
  convention* (§ below adds a scenario within the new Tier-2 requirement, not a
  change to the convention requirement).

**Logging contract (CC-9 — Reviewer-4 F7).** `psi_g.compute` emits exactly one
`DEBUG` record on the happy path, message prefixed `"psi_g.compute("` with tokens
`n_tracks=` and `cadence_s=`, and **no** `INFO`/`WARNING` — mirroring
`nutation.compute` (`nutation.py:697-702`). A scenario asserts this.

## 12. Open risks / reviewer flags (confirmed during review)

- **ψ_g ramp degeneracy** (confirmed numerically, Reviewer-1 F5). A *pure*
  circular orbit makes ψ_g ramp monotonically → no oscillatory period; `T_psig`
  ill-defined. Real growth-dominated roots give an *oscillating* ψ_g (zero net
  ramp for symmetric wobble) → CWT recovers the period. Near-circular-but-drifting
  orbits give a ramp-with-ripple → CWT recovers a period **and** `handedness=±1`
  (intended). Documented as the H1 regime assumption (§3.4).
- **Off-by-one bookkeeping** (Reviewer-2 F6). ψ_g is length `N−1`; `in_coi` is
  length `N−1`; the second diff dψ_g/dt would be `N−2` — avoided by computing
  `handedness` from ψ_g **endpoints** over COI-interior frames (§4), so no
  `N−2`↔`N−1` mismatch.
- **Cross-OS determinism** (CC-6 — Reviewer-4 F6). `handedness` is integer and
  cannot use `atol`; a near-zero net rotation could flip ±1 across OSes. Guarded
  by the `1e-9 rad` numerical-zero band (§4) — determinism hygiene, distinct from
  the (rejected) physical deadband. The 3 float traits follow the established
  cross-OS `atol` expectation. `np.nanmedian` of an even-length COI array is
  deterministic.
- **Sign-convention drift.** `compute_signed_area`'s negation and
  `compute_psi_g`'s atan2 order are jointly load-bearing; the *absolute-anchor*
  test (§5) — not just the agreement test — is the guard against a silent joint
  flip.

## 13. Reconciliation log (4-reviewer critical pass)

**Brainstorm decisions reversed on review evidence**
- *Conditioning* smooth-only → **SG-detrend** (`compute_sg_detrended`, win 23).
  Reviewer-1 F2 (gravitropic-drift bias; "COI handles it" overstated, ~39% of
  longest-scale interior unmasked) + Reviewer-2 F1 (no smooth primitive exists).
  Verified: only `compute_sg_detrended`/`compute_sg_residual_xy` exist.
- *delta_E unit* px/s → **px/frame** (`delta_E_amplitude_proxy_px_per_frame`).
  Reviewer-4 F8 (px/s ∉ `PIPELINE_UNIT_VOCABULARY`) + Reviewer-1 F3. Verified:
  vocab has `px/frame`, `px/hr`, `s`, `px²`, `int` — not `px/s`. Tier 0 emits all
  velocities in px/frame.

**BLOCKING findings**
- R1-F6 Shoelace sign backwards → §5 y-down negation, `sign(area)==handedness`,
  verified numerically; absolute-anchor + agreement tests.
- R2-F1 no SG-smooth primitive → resolved by SG-detrend (above).
- R3-F2 plate-001 `principal_axis_angle` NaN-gated → §6 NaN-skip + `≥N/6` count.
- R3-F3 plate-001 tolerance was TDD-after → §6 relabeled GREEN-phase
  reconciliation; synthetic convention-lock is the true RED test.
- R4-F1/F2 `MODIFY Package layout` required (stub→impl) → §11 (was "MODIFY none").

**IMPORTANT findings**
- R1-F1 branch-cut comparison → §6 `wrap_to_pi` distance + branch-cut fixture.
- R2-F2 validators not shared → §3 discloses reuse of
  `temporal_cwt._validate_cadence_s` + a small psi_g `_check_constants`.
- R2-F3 `coi_mask` is 2-D → §4 uses per-frame `ridge.in_coi`.
- R2-F4 all-NaN `nanmedian` warning → §4 guarded; §10 unit-4 asserts no warning.
- R2-F6 off-by-one → §4 handedness via ψ_g endpoints (no `N−2` array).
- R2-F7 raw-tip NaN handling → §3.1 finite-mask + graceful row.
- R3-F4 degenerate tests need direct construction → §10 unit 6.
- R3-F5 unreachable defensive branches vs ≥90% → §10 coverage note (pragma or
  targeted helper test).
- R3-F6 sign test joint-flip-degenerate → §5 absolute anchor + agreement (two
  independent tests).
- R4-F4 underspecified terms pinned: "COI-interior" = `~ridge.in_coi`; "empirical
  tolerance" = documented GREEN-phase constant + count; "graceful values" =
  §4 table.
- R4-F6 CC-6 handedness determinism → §4 `1e-9 rad` numerical-zero band.
- R4-F7 CC-9 logging → §11 logging contract scenario.

**MINOR / confirmed-correct (no change)**
- R1-F4 handedness sign self-consistent with locked convention (confirmed).
- R1-F5 ramp degeneracy correctly characterized (confirmed numerically).
- R2-F5 dtype loop extends with `handedness→int64` branch; `fillna(0)` invariant
  documented (§4).
- R3-F1 `generate_trajectory(handedness=±1)` is controllable; tests call it
  directly (§6/§10).
- R3-F7 commit-unit ordering sound.
- R4-F3 `_CONSTANTS_VERSION` stays 5, test untouched (§8).
- R4-F5 NaN-vs-0.0 distinction pinned (§4 table).
- R4-F9 `_load_proofread_track_df` is test-local → "mirror the loader pattern".
- R4-F10 `compute_signed_area` justified (consumed now), not speculative.
- R4-F11 follow-up issue pre-drafted (§7).
