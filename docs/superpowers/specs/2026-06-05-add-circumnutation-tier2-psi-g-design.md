# Design — `add-circumnutation-tier2-psi-g` (Program PR #7, Tier 2)

**Status:** brainstorm draft, revised after 2 rounds of 4-reviewer critical pass (pre-OpenSpec-proposal)
**Date:** 2026-06-05
**Branch:** `add-circumnutation-tier2-psi-g`
**Epic:** #197 · **Roadmap row:** PR #7 (`docs/circumnutation/roadmap.md`)
**Theory anchors:** `docs/circumnutation/theory.md` §3.5 (ψ_g, Bastien & Meroz 2016 Eq. 20), §6.3 (tier structure + SG conditioning), §7.3 (Tier 2 trait table), §3.7 (trochoid / H1-failure)

> **Revision note.** §13 is the reconciliation log for both review rounds. Three
> brainstorm/round-1 decisions changed on evidence: (a) ψ_g conditioning →
> **SG-detrend** (only a detrend primitive exists; smooth-only risks
> gravitropic-drift bias); (b) `delta_E` → **px/frame** (px/s absent from the
> units vocabulary; Tier 0 convention); (c) `handedness` → **net rotation over
> all finite frames, no COI** (round-2: COI-masking coupled handedness to the
> conditioned CWT ridge and introduced a non-contiguity sign-flip bug — a
> deliberate deviation from §7.3's literal "COI-masked", justified in §4/§13).

---

## 1. Purpose

Emit **Tier 2 Bastien–Meroz ψ_g traits** per track. ψ_g(t) is the unwrapped
velocity-direction angle of the apical tip — under hypotheses H1–H3 it is the
*direct estimator of the differential-growth oscillator* (theory §3.5). Applying
the temporal CWT to ψ_g(t) recovers the spectral content of that oscillator.

PR #7 ships a self-contained, single-tier module mirroring the merged Tier 1
module (`nutation.py`, PR #6). It reuses the locked geometry helper
`_geometry.compute_psi_g` (PR #2, atan2 convention) and the temporal-CWT
primitives (`temporal_cwt`, PR #5). It introduces **no new `ConstantsT` fields**
and emits **4 of the 5** §7.3 traits; `psig_long_consistency` is deferred (§7).

**Conditioning isolation (now literally exact).** Only `T_psig_median_s` uses the
conditioned (SG-detrended) ψ_g and the CWT. The other three traits use **raw**
inputs, and each a *different* raw input: `handedness` uses the raw unwrapped
ψ_g; `delta_E` uses raw velocity samples `(dx, dy)`; `helix_signed_area` uses raw
coordinates `(x, y)`. None of these three touches the CWT or the conditioned
signal — so a track too short for the CWT still yields all three.

## 2. Scope

### In scope
- New public API `sleap_roots.circumnutation.psi_g.compute(trajectory_df, cadence_s, constants=None) -> pd.DataFrame`, **graduating `psi_g` from stub to implementation** (renames the stub callable `compute_psi_g` → `compute`, matching `kinematics.compute`/`qc.compute`/`nutation.compute`).
- New geometry helper `_geometry.compute_signed_area(x, y) -> float` (y-down-corrected Shoelace), sibling to `compute_psi_g`.
- 4 Tier 2 trait columns (§4).
- Cross-tier empirical consistency validation against Tier 0 `principal_axis_angle` (§6).
- Min-length handling for the CWT/`T_psig` path (§3.1) — explicit, mirroring `nutation.py`'s two-layer guard.
- Foundation-test migration (§9): move `psi_g` out of stub tables into the implementation tables.

### Out of scope (explicit non-goals)
- `psig_long_consistency` (§7.3 trait #2) — requires Tier 1's `T_nutation`; deferred to a pre-drafted follow-up issue (§7).
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

Differences from `nutation.compute`: **no `coordinate` kwarg** (and no
`_check_coordinate`) — ψ_g comes from the raw 2D tip trajectory via
`atan2(dx, dy)`; there is no 1D-projection choice.

Shared with `nutation.compute`: per-track
`groupby(list(_IDENTITY_5_TUPLE), dropna=False, sort=False)` →
`_compute_one_track` → per-plant template merge via
`_build_per_plant_template_from_df` → identity dtype coercion (loud `ValueError`)
→ declared column order `ROW_IDENTITY_COLUMNS + _PSIG_TRAIT_COLUMNS`.

**Validators (disclosed reuse vs new).** `_validate_trajectory_df` is shared
(`_types.py`) and reused; it already enforces `tip_x`/`tip_y` column *presence*
for all callers (per-row finiteness is handled in §3.1). For `cadence_s`, reuse
the importable `temporal_cwt._validate_cadence_s` (module-level) rather than
copying nutation's private `_check_cadence_s`. For `constants`, `psi_g` defines a
small `_check_constants` that accepts `None`/`ConstantsT`, validates the SG fields
it consumes (`SG_WINDOW_DETREND` odd & > `SG_DEGREE`; `SG_DEGREE` ≥ 0) with
field-named `ValueError`s, and defers CWT-field validation to `compute_scaleogram`
(which runs `_validate_cwt_constants`). It does **not** call
`_validate_nutation_constants` (Tier-1 fields psi_g doesn't use).

`constants` is threaded to `compute_sg_detrended` and `compute_scaleogram` so
overrides propagate — hence `psi_g.compute` joins
`IMPLEMENTATIONS_WITH_CONSTANTS_KWARG`.

### 3.1 `_compute_one_track` pipeline

Operates on raw `tip_x`/`tip_y`. **Only `T_psig_median_s` uses the conditioned
signal + CWT; the other three are raw and CWT-free.**

1. **Finite-frame extraction.** Drop rows with non-finite `tip_x`/`tip_y`
   (`compute_psi_g` has no NaN guard — Reviewer-2/r1-F7). Let `N` = count of
   finite frames.
2. **Degenerate short-circuit (`N < 3`).** ψ_g needs ≥2 velocity samples; return
   the all-degenerate row `{T_psig_median_s=NaN,
   delta_E_amplitude_proxy_px_per_frame=NaN, handedness=0,
   helix_signed_area_px2=NaN}`. (`N=0`, the all-non-finite track, is this case.)
   The `<3` count is over **post-finite-mask** rows.
3. **ψ_g(t).** `psi_g = _geometry.compute_psi_g(tip_x, tip_y)` — length `N−1`,
   unwrapped, locked `atan2(dx, dy)`. **Reused, never redefined.**
4. **Raw, CWT-free traits (always emitted for `N ≥ 3`):**
   - `handedness = int(np.sign(psi_g[-1] − psi_g[0]))` (net unwrapped rotation
     over **all** finite frames; no COI), with the `1e-9 rad` zero-guard (§4).
   - `delta_E = np.median(√(dx² + dy²))` over finite velocity samples
     (`dx,dy = np.diff` of finite tip coords).
   - `helix = _geometry.compute_signed_area(tip_x, tip_y)`.
5. **`T_psig_median_s` (CWT path; needs `N ≥ 24` — see min-length note):**
   - If `len(psi_g) < SG_WINDOW_DETREND (=23)` → `T_psig_median_s = NaN`
     (skip the CWT entirely). `compute_sg_detrended` returns all-NaN for
     `len < window` (`_noise.py`), and feeding all-NaN to `compute_scaleogram`
     raises — so this guard is mandatory.
   - Else: `psi_g_detrended = compute_sg_detrended(psi_g, window=SG_WINDOW_DETREND,
     polynomial_order=SG_DEGREE)`. If `not np.isfinite(psi_g_detrended).any()` →
     `T_psig_median_s = NaN` (mirrors `nutation.py:418-419`).
   - Else `try`: `compute_scaleogram → extract_ridge → smooth_ridge`;
     `T_psig_median_s = nanmedian` of `smooth_ridge.periods_s[~smooth_ridge.in_coi]`,
     **guarded** so an empty-interior **or** all-NaN slice → NaN with **no**
     `RuntimeWarning` (mirrors `nutation.py:438-442`). `except ValueError` (signal
     too short for the CWT scale grid) → `T_psig_median_s = NaN` (mirrors
     `nutation.py:422-428`).

**Min-length note.** A finite `T_psig_median_s` requires `len(ψ_g)=N−1 ≥
SG_WINDOW_DETREND=23`, i.e. **`N ≥ 24`**. The CWT itself also has a floor
(`compute_scaleogram` requires `len ≥ MIN_FRAMES_REQUIRED = 9` and raises
otherwise). The two-layer guard above (finite-check + `try/except ValueError`) is
the exact merged-Tier-1 precedent. For `3 ≤ N < 24`, `T_psig_median_s` is NaN
while `handedness`/`delta_E`/`helix` are fully defined (step 4 is CWT-free).

## 4. Trait set (§7.3, restricted to the 4 self-contained traits)

Naming: explicit unit suffixes (#222). Period in seconds (matches Tier 1
`T_nutation_median`); the speed proxy in **px/frame** (Tier 0's cadence-independent
velocity convention). All four units (`s`, `px/frame`, `int`, `px²`) are already
in `PIPELINE_UNIT_VOCABULARY`.

| Column | dtype | Units | Definition |
|---|---|---|---|
| `T_psig_median_s` | float64 | s | `np.nanmedian` of `smooth_ridge.periods_s` over **COI-interior** frames (`~smooth_ridge.in_coi`), mirroring `nutation.py`. Empty/all-NaN interior → NaN, no `RuntimeWarning`. NaN when `N < 24` (§3.1). |
| `delta_E_amplitude_proxy_px_per_frame` | float64 | px·frame⁻¹ | `np.median(√(dx²+dy²))` over **all finite** velocity samples. No COI; no `/cadence_s` (px/frame). `= (L/2R)·ΔĖ`, Eq. 21. |
| `handedness` | int64 | int | `int(np.sign(psi_g[-1] − psi_g[0]))` — sign of **net unwrapped ψ_g rotation over all finite frames** (= mean dψ_g/dt × span). Determinism zero-guard: `|psi_g[-1] − psi_g[0]| < 1e-9 rad → 0`. `+1` = counterclockwise. `0` = no net rotation / degenerate. **No COI** (deviation from §7.3 — see note). |
| `helix_signed_area_px2` | float64 | px² | `_geometry.compute_signed_area(tip_x, tip_y)`; **y-down-corrected** Shoelace so `sign(area) == handedness` (§5). Independent confirmation of handedness. |

Declared order: 8 `ROW_IDENTITY_COLUMNS` then the 4 trait columns above.

**Why `handedness` drops the §7.3 "COI-masked range" (round-2 deviation).** §7.3
reads "sign of mean dψ_g/dt over COI-masked range." Round-2 review showed the COI
mask (a) is sourced from the CWT of the *detrended* signal, so COI-masking would
couple `handedness` to conditioning (breaking the isolation in §1) and to the
CWT min-length floor (short tracks with obvious curl would read `handedness=0`
for a spectral reason); and (b) `ridge.in_coi` interior is **not contiguous**
(per-frame argmax selects different scales → COI gaps), so an endpoint-difference
across a masked gap can report the *wrong* sign (demonstrated counterexample).
COI is a CWT-edge-reliability concept; a raw angular displacement (atan2 of
velocity) has no edge contamination, so COI-masking it is a category error —
§7.3 itself omits COI for the sibling `delta_E` kinematic median. Net rotation
over all finite frames is gap-immune, conditioning-independent, defined for
`N ≥ 3`, and makes the `1e-9 rad` zero-guard defensible (raw atan2 differences
are ~`1e-12` reproducible cross-OS, well below `1e-9`). Recorded as an
intentional deviation in the proposal.

**COI semantics.** Only `T_psig_median_s` uses COI — via the **per-frame**
`RidgeResult.in_coi` (shape `(N−1,)`), exactly as `nutation.py` does, **not** the
2-D `ScaleogramResult.coi_mask` `(n_scales, N−1)`.

**No `is_nutating`-style gate** (decision D1). Tier 2 is self-contained — never
reads Tier 1's `is_nutating`. Traits emitted ungated; downstream QC `LEFT JOIN`s
`is_nutating` on the shared 5-tuple — the same composition point as the deferred
`psig_long_consistency`.

### Why `handedness` and `is_nutating` are orthogonal
`is_nutating` is an *existence/SNR gate*; `handedness` is a *directional sign*. A
non-nutating track can have `handedness=±1` (slow gravitropic curl); a clean
back-and-forth wobble can have `handedness=0` (net-zero circulation). Different
questions → Tier 2 does not gate on `is_nutating`.

### Degenerate / edge-case table

| Condition | `T_psig_median_s` | `delta_E…_px_per_frame` | `handedness` | `helix_signed_area_px2` |
|---|---|---|---|---|
| `N < 3` finite frames (incl. all-NaN track) | NaN | NaN | 0 | NaN |
| `3 ≤ N < 24` (too short to SG-detrend/CWT) | NaN | **defined** | **defined (±1/0)** | **defined** |
| Stationary tip, `N ≥ 24` (zero displacement) | NaN | 0.0 | 0 | 0.0 |
| `N ≥ 24`, normal | (defined) | (defined) | (defined) | (defined) |
| Unmatched per-plant template row | NaN (`fillna`) | NaN (`fillna`) | 0 (`fillna(0)`) | NaN (`fillna`) |

NaN-vs-0.0 is deliberate: **too-short → NaN** (undefined); **stationary but
long-enough → 0.0** (a genuinely measured zero). With handedness now COI-free,
there is **no** "empty-interior" handedness failure mode.

**dtype enforcement.** psi_g's enforcement loop is `int64` for `handedness`,
`float64` for the other three (replaces nutation's `is_nutating→bool`
special-case with a `handedness → fillna(0).astype(np.int64)` branch). `0` is a
*meaningful* handedness value, so `fillna(0)` on unmatched rows is only safe under
the documented 5-tuple merge invariant (template built from the same
`trajectory_df` → no unmatched rows in practice; the path is defensive).

## 5. Geometry helper: `_geometry.compute_signed_area(x, y)`

Next to `compute_psi_g` in `_geometry.py` (single home for convention-critical
geometry).

- **y-down-corrected Shoelace:** `A = 0.5 * Σ_i (x_{i+1}·y_i − x_i·y_{i+1})`
  (cyclic) — the **negative** of standard `0.5·Σ(x_i·y_{i+1} − x_{i+1}·y_i)`.
  The negation makes positive area correspond to counterclockwise-as-a-viewer-sees
  and therefore to `handedness=+1` under the same y-down `atan2(dx, dy)`
  convention. **Verified numerically (both rounds):** a screen-CCW orbit gives
  `handedness=+1` and standard Shoelace `−3.14`; the negation → `+3.14` so
  `sign(area)==handedness`.
- Returns `float`. `< 3` points → `0.0` (degenerate polygon) **at the helper
  level**; `_compute_one_track` maps the `N<3`-frame case to NaN *before* calling,
  so the trait reports NaN for too-short tracks. (Helper-level scenario asserts
  `0.0`; trait-level scenario asserts NaN — kept distinct, not collapsed.)
- NaN-robust: non-finite coords → NaN area (caller short-circuits first).
- **Sign is load-bearing**, pinned by two *independent* tests (break the
  joint-flip degeneracy):
  1. **Absolute anchor (concrete RED number):** for the hand-built path
     `x=[0,1,1,0], y=[0,0,1,1]`, assert `compute_signed_area(x, y) == -1.0`
     exactly (standard Shoelace gives `+1.0`; the negation flips it). On that same
     path `compute_psi_g` gives net `Δψ_g = −π` → `handedness=−1`, so
     `sign(area)==handedness==−1`. Verified by review.
  2. **Agreement:** a synthetic CCW orbit (`generate_trajectory(handedness=+1)`)
     yields **both** `handedness=+1` **and** `helix_signed_area_px2 > 0`.

## 6. Cross-tier consistency check (validation, not a trait)

ψ_g = `atan2(dx, dy)`; `principal_axis_angle` = `atan2(uy, ux)`
(`kinematics.py:269`). Via `atan2(a, b) = π/2 − atan2(b, a)`, a track growing
along unit axis **u** satisfies:

> **circular_mean(ψ_g) ≈ π/2 − `principal_axis_angle`  (mod 2π)**

with `circular_mean(ψ_g) = atan2(mean(sin ψ_g), mean(cos ψ_g))`.

**Branch-cut-safe comparison.** Assert on the *wrapped* distance:
`abs(wrap_to_pi(circular_mean(ψ_g) − (π/2 − principal_axis_angle))) < tol`,
`wrap_to_pi(d) = (d + π) mod 2π − π` (realized interval `[−π, π)`; immaterial for
a `|distance| < tol` check).

Two tests with **distinct honesty levels**:

1. **Synthetic convention-lock — the true RED test.** `generate_trajectory` with
   a *planted* growth-axis angle θ and planted `handedness`; on a noise-free
   trajectory assert `abs(wrap_to_pi(circular_mean(ψ_g) − (π/2 − θ))) < 1e-6`
   **(angle identity only)** and `handedness == planted sign`. Include a fixture
   with θ ≈ −2.0 rad so `π/2 − θ` lands outside `[−π, π)` and exercises the wrap.
   Tests call `generate_trajectory(handedness=±1, growth_axis_angle_rad=θ)`
   directly (the shared `_minimal_trajectory_df` hard-codes `handedness=+1`).
   **The `1e-6` atol applies ONLY to the angle identity + handedness sign — NOT
   to any period magnitude.**
2. **Real plate-001 — GREEN-phase fixture-sanity / reconciliation.** Over the 6
   tracks of `…proofread.slp` (mirror the test-local `_load_proofread_track_df`
   loader pattern; tracks are 575 frames ≫ 24), run `kinematics.compute` for
   `principal_axis_angle` and assert the reconciled identity. **Guards:**
   - **Skip NaN'd tracks:** `principal_axis_angle` is NaN when
     `growth_axis_unreliable` fires (`kinematics.py:218-221,284-290`); assert on
     the surviving subset with a `≥ N-of-6` count clause (mirrors nutation's
     `test_2H3` `≥3/6`), N pinned at GREEN.
   - **Tolerance is a documented GREEN-phase constant** (e.g.
     `_PSIG_AXIS_RECONCILE_TOL_RAD`), captured from a real run, "GREEN-phase
     Reconciliation" docstring — the `_DERR_MATCH_*` precedent
     (`test_circumnutation_nutation.py:101-118`). Not a RED assertion against a
     pre-known threshold.

## 7. Deferred: `psig_long_consistency` (pre-drafted follow-up issue)

§7.3 trait #2, "correlation between `T_psig` and `T_nutation` across the CWT
range" (H1 diagnostic, §3.7) — structurally couples Tier 2 to Tier 1's ridge.
Deferring keeps PR #7 a clean single-tier module (decision D1).

**Follow-up issue to stage in the vault during proposal:**
- **Title:** `circumnutation: psig_long_consistency cross-tier T_psig↔T_nutation correlation (Tier 1 × Tier 2 CWT co-registration)`
- **Scope bullets:** (a) co-register the Tier 1 lateral-coordinate ridge and the
  Tier 2 ψ_g ridge on a common frame/period grid; (b) emit
  `psig_long_consistency` as their correlation over the COI-interior overlapping
  period range; (c) H1-failure (§3.7) interpretation note; (d) ownership decision
  (new cross-tier module vs `psi_g.compute` optional `nutation_df`).

Until then, the downstream `LEFT JOIN` of `is_nutating` is the cross-tier
composition mechanism.

## 8. Constants & versioning

**No new `ConstantsT` fields.** Conditioning reuses `SG_WINDOW_DETREND=23`,
`SG_DEGREE=3`; the CWT reuses the PR #5 fields. `handedness`/`delta_E`/`helix`
need none. **`_CONSTANTS_VERSION` stays 5**;
`test_schema_version_is_1_and_constants_version_is_5` is **untouched** (no-op
confirmation in tasks.md). No MODIFY to the *Module-level constants* requirement.

## 9. Foundation-test migration (PR #6 precedent)

Graduating `psi_g` stub→impl touches the foundation contract:
- **Remove** from `STUB_MODULES` (drop `("psi_g", "compute_psi_g", 7)`) and
  `STUBS_WITH_CONSTANTS_KWARG` (drop `("psi_g", "compute_psi_g")`). Verified these
  exact entries exist.
- **Add** `("psi_g", "compute")` to `IMPLEMENTATIONS_WITH_CONSTANTS_KWARG`, **and**
  the matching `elif module_name == "psi_g":` branch in
  `test_implementation_accepts_constants_kwarg` (builds a valid `trajectory_df`,
  asserts a DataFrame).
- **Add** `psi_g` to the explicit `test_module_logger_is_namespaced` list — note
  removing `psi_g` from `STUB_MODULES` *de-covers* it from that parametrization
  (which derives from `STUB_MODULES`), so the explicit add is mandatory (same
  regression PR #4/#5/#6 handled). Assert
  `logger.name == "sleap_roots.circumnutation.psi_g"`.
- The stub-`NotImplementedError` scenarios no longer enumerate `psi_g` (§11). No
  executable stub-count assertion exists to update (only comments) — verified.
- Add a cheap unit test asserting all 4 psi_g trait units ∈
  `PIPELINE_UNIT_VOCABULARY` (RED guard for the §4 units claim).

## 10. TDD plan (RED → GREEN commit pairs, per #222 retrospective)

Strict red-green-refactor: failing-test commit (`test: … (TDD red)`) then
implementation commit (`feat:/fix: … (TDD green)`). No bundling impl + large test
file. Fixup commits ship a test alongside any substantive code change.

Commit-pair units (ordered; red-greenable):
1. `_geometry.compute_signed_area` + the **two** sign tests (absolute anchor
   asserting `== -1.0`; `<3`-point/`NaN` edge), §5.
2. `psi_g.compute` schema/structure: returns DataFrame, 8 identity + 4 trait
   columns in declared order, dtypes (3 float64 + 1 int64). Stub emits the
   degenerate-table default row (units 2 & 6 share default-emission code).
3. Input-validation boundary: non-DataFrame, invalid `trajectory_df`, bad
   `cadence_s` (reuse `temporal_cwt._validate_cadence_s`), bad `constants`
   (even-`SG_WINDOW_DETREND` override → field-named `ValueError`).
4. Raw CWT-free traits: `handedness` (planted CW/CCW via
   `generate_trajectory(handedness=±1)`), `delta_E` (known constant-speed
   synthetic), handedness↔area agreement. These pass for `N ≥ 3`.
5. `T_psig_median_s` pipeline: `compute_psi_g → compute_sg_detrended → CWT chain`
   on a clean known-period synthetic (`N ≥ 24`); assert recovered period within
   **±10 %** (cite nutation `test_2C2`: SG-detrend distorts noise-free recovery to
   ~5 %, so ±5 % is too tight — ±10 % keeps it honest RED→GREEN, **not**
   TDD-after). Assert **no** `RuntimeWarning` on the all-COI path.
6. Degenerate/edge cases (the §4 table) via **direct construction** — incl. a
   `3 ≤ N < 24` track (e.g. 15 frames) asserting `T_psig=NaN` while
   `handedness`/`delta_E`/`helix` are finite **and no exception is raised**; a
   2-row track; NaN-injection (à la `test_2F7b`); a stationary `N≥24` track.
7. Cross-tier: (a) synthetic convention-lock RED (incl. the θ≈−2.0 branch-cut
   fixture, `1e-6` on the angle identity only); (b) plate-001 GREEN-phase
   reconciliation with NaN-skip + `≥N/6` count.
8. Foundation-test migration (§9) + the units-vocabulary guard + a multi-track
   integration test.

**Coverage.** Target ≥ 90 % on `psi_g.py` and the new `_geometry` helper; project
gate ≥ 84 %. The `fillna`/identity-dtype-coerce branches are unreachable via the
public API (template derives from the same `trajectory_df`); mirror `nutation.py`
— `# pragma: no cover` citing the invariant, or a direct unit test on the
merge/coerce path. The new min-length / `try-except` branches **are** reachable
and are covered by unit 6.

## 11. Spec deltas (preview for `/openspec:proposal`)

- **MODIFIED** `Requirement: Package layout` — **full-paste the entire updated
  requirement** (the archiver replaces the whole requirement; partial deltas drop
  detail). Edits: move `psi_g` from the stub set to the implementation set
  (recount **6 implementation / 5 stub** — the *inverse* of PR #6, which grew impl
  without shrinking the stub set differently); delete the `psi_g` row from the
  stub-callable table; keep `psi_g` in the "imports cleanly" scenario (still
  importable, now as impl); drop `psi_g` from the "Calling each remaining stub
  raises NotImplementedError" scenario (now **5** remaining stubs); add a
  `psi_g.compute callable without raising` scenario mirroring `nutation.compute`.
  The `IMPLEMENTATIONS_WITH_CONSTANTS_KWARG` + logger-namespace migrations (§9)
  are part of this delta.
- **ADDED** `Requirement: Tier 2 ψ_g trait emission API` — `psi_g.compute` (the 4
  traits, conditioning isolation, no cross-tier input, min-length behavior) and
  `_geometry.compute_signed_area`. **Each topic rendered as a discrete
  `#### Scenario:`** (this is a large set — budget it in tasks.md):
  schema/order/dtypes; each trait's definition; the degenerate table
  (parametrized over its rows, incl. the `3≤N<24` band); the handedness↔area
  **absolute-anchor** lock AND the agreement lock (two scenarios, §5); the
  synthetic convention-lock RED **with the branch-cut fixture**; the plate-001
  GREEN with NaN-skip + `≥N/6`; validation errors (bad `cadence_s`, even
  `SG_WINDOW_DETREND`); the **one-DEBUG-record logging contract**; the
  no-`RuntimeWarning` all-COI scenario; CC-6 determinism; the helper-level `0.0`
  vs trait-level `NaN` distinction.
- **No MODIFY** to *Module-level constants* (§8) or *Per-module logger
  convention* (the logging scenario lives inside the new Tier-2 requirement).

**Logging contract (CC-9).** `psi_g.compute` emits exactly one `DEBUG` record on
the happy path, prefixed `"psi_g.compute("` with tokens `n_tracks=` and
`cadence_s=` (no `coordinate=` token — psi_g has no such kwarg), and **no**
`INFO`/`WARNING` — mirroring `nutation.compute` (`nutation.py:697-702`).

## 12. Open risks / reviewer flags (confirmed across both rounds)

- **ψ_g ramp degeneracy** (confirmed numerically). A *pure* circular orbit makes
  ψ_g ramp monotonically → no oscillatory period; `T_psig` ill-defined. Real
  growth-dominated roots give an *oscillating* ψ_g (zero net ramp for symmetric
  wobble) → CWT recovers the period; near-circular-but-drifting orbits give
  ramp-with-ripple → CWT recovers a period **and** `handedness=±1` (intended).
  Documented as the H1 regime assumption (§3.4).
- **Off-by-one bookkeeping.** ψ_g is length `N−1`. `handedness` uses ψ_g
  **endpoints** (no second diff, no `N−2` array). `T_psig` uses `ridge.in_coi`
  (also length `N−1`). No length mismatch anywhere.
- **SG-detrend passband** (confirmed numerically). Window 23 at cadence 300 s ≈
  2× the Derr 3333 s period: retention ≈1.22 at the nutation period, ≈0.009 at 5×
  (gravitropic drift). The nutation band survives; drift is removed.
- **Cross-OS determinism (CC-6).** Float traits (`T_psig`, `delta_E`, `helix`)
  inherit Tier 1's **`atol=1e-6`** cross-OS floor (the SG-detrend→scipy-CWT stack;
  spec.md:1163). `handedness` is integer and CWT-free — its `1e-9 rad` zero-guard
  is safe because the net-rotation endpoint difference is a raw-atan2 quantity
  (~`1e-12` reproducible), well clear of `1e-9`; zeroing `<1e-9 rad` of net
  angular drift is scientifically harmless (numerical-zero hygiene, not a physical
  deadband). `np.nanmedian` of an even-length array is deterministic.
- **Sign-convention drift.** `compute_signed_area`'s negation and
  `compute_psi_g`'s atan2 order are jointly load-bearing; the *absolute-anchor*
  test (§5, `== -1.0`) — not just the agreement test — guards a silent joint flip.

## 13. Reconciliation log (2 rounds × 4 reviewers)

**Decisions changed on review evidence**
- *Conditioning* smooth-only → **SG-detrend** (`compute_sg_detrended`, win 23).
  R1 (gravitropic-drift bias; no smooth primitive exists). Verified.
- *delta_E unit* px/s → **px/frame**. R1 (px/s ∉ vocabulary; Tier 0 convention).
  Verified.
- *handedness* COI-masked → **net rotation over all finite frames, no COI**. R2
  (COI coupled handedness to the conditioned ridge + non-contiguity sign-flip bug
  + short-track coupling). Deliberate §7.3 deviation, justified §4.

**Round-1 BLOCKING/IMPORTANT** — all reconciled (Shoelace y-down negation §5;
SG-detrend replaces nonexistent smooth primitive; plate-001 NaN-skip + GREEN-phase
relabel §6; `MODIFY Package layout` §11; branch-cut wrap §6; validator reuse §3;
per-frame `ridge.in_coi` not 2-D mask §4; all-NaN nanmedian guard §3.1; raw-tip
NaN mask §3.1; degenerate direct-construction tests §10; unreachable-branch
coverage policy §10; absolute + agreement sign tests §5; CC-6 `1e-9` guard §12;
CC-9 logging §11; pre-drafted follow-up §7).

**Round-2 BLOCKING/IMPORTANT**
- R2 *min-length gap* (3 of 4 reviewers): SG-detrend all-NaN for `N<24` →
  `compute_scaleogram` raises → uncaught crash for `3≤N<24`. **Fixed** §3.1 step 5
  (finite-check + `try/except ValueError`, mirroring `nutation.py:418-428`) +
  degenerate-table row + unit-6 test.
- R2 *handedness COI provenance contradiction + non-contiguity sign-flip*: fixed
  by the all-finite-frames redefinition (above).
- R2 *T_psig synthetic tolerance*: noise-free recovery ≈5 % error through
  SG-detrend → unit-5 tolerance **±10 %** (not `1e-6`); `1e-6` applies only to the
  angle identity. **Fixed** §6/§10.
- R2 *determinism anchor*: float floor is `1e-6` (Tier 1), not `1e-9`; handedness
  `1e-9` defensible because it's CWT-free raw atan2. **Fixed** §12.
- R2 *spec-delta completeness*: ADDED requirement needs per-trait scenarios
  rendered; MODIFIED Package layout must full-paste with 6-impl/5-stub recount.
  **Captured** §11.
- R2 *"raw ψ_g" imprecision*: partitioned — handedness/cross-tier use raw ψ_g,
  delta_E uses raw velocities, helix uses raw coords. **Fixed** §1/§3.1.

**Confirmed correct (no change)**: Shoelace negation formula (abs anchor `−1.0`);
`compute_psi_g→compute` rename has no dangling refs (`_geometry.compute_psi_g` is
a different, retained symbol); §9 foundation edit list exact & complete; dtype
loop shape; §6 atan2/NaN-gate/`generate_trajectory(growth_axis_angle_rad, handedness)`
verified; validator reuse; no `compute_signed_area` collision; `_CONSTANTS_VERSION`
5 untouched; SG-detrend passband; spec-delta MODIFIED+ADDED split; CC-3 pure-pixel;
no 5th-trait leakage.
