# Design — `add-circumnutation-tier3a-midline` (Program PR #8, Tier 3a)

**Status:** brainstorm draft, revised after critical-review round 1 (5 reviewers, numerically verified on real plate-001)
**Date:** 2026-06-08
**Branch:** `add-circumnutation-tier3a-midline`
**Epic:** #197 · **Roadmap row:** PR #8 (`docs/circumnutation/roadmap.md`)
**Theory anchors:** `docs/circumnutation/theory.md` §6.1 (tip-trail-as-midline identity), §6.2 (trajectory curvature + SG smoothing before differentiation + velocity mask), §6.3 (tier structure), §4 (Rivière 2022/2025 QPB context)

> **Revision note (round 1).** §13 is the reconciliation log. Round 1 (numerically verified on real plate-001) changed five load-bearing things: (a) **units → px/frame** (program convention; `px/s` was deliberately rejected, theory §10 App B); (b) the velocity-mask **rationale was empirically false** — std(speed) flags ~40–60% of real frames (NOT "~nothing"), which is *correct* (rice tip moves only ~3× the localization-noise floor) and coincides with a principled noise-floor on plate-001, so the std(speed) choice holds but the prose+test are corrected; (c) **cross-OS atol → 1e-9** (measured full-pipeline ULP propagation ≈1e-14, not the hand-waved 1e-6); (d) **stationary detection moved to the raw input** (SG dust means speed is never exactly 0); (e) canary gains a **closed-form circle oracle**. This is a **machinery** PR mirroring PR #5 `temporal_cwt` (input-validating, frozen `attrs` Result, determinism contract, **no trait emission**) — with one deliberate divergence: a graceful `is_degenerate` flag (§4, §13/I1).

---

## 1. Purpose

Implement the **tip-trail-as-midline reconstruction** primitive (theory §6.1): for an apically-growing organ whose tissue past the elongation zone does not reshape, the curve of past tip positions **is** the organ midline. `midline.reconstruct(x, y, cadence_s, …)` returns that midline parameterized by arc length `s(τ) = ∫|v|dσ`, together with per-frame trajectory curvature `κ`, tip speed `|v|`, and a velocity-bandpass mask flagging sub-noise frames where the curvature denominator `|v|³` would blow up.

PR #8 is the reconstruction substrate that PR #9 (spatial CWT on `κ(s)`, `L_gz` peak-finder, `L_c` decay fit) and PR #10 (Tier 3 trait emission, applies the `L_gz` growth-zone mask) compose on top of. PR #8 emits **NO traits**, builds **NO `L_gz` mask** (CC-1), and does **NO** uniform-arc-length resampling (deferred to PR #9; §2 handoff).

## 2. Scope

### In scope
- New public API `sleap_roots.circumnutation.midline.reconstruct(x, y, cadence_s, sg_window=None, constants=None) -> MidlineResult`, **graduating `midline` from stub to implementation** (the callable keeps its name `reconstruct` — no rename).
- New frozen dataclass `MidlineResult` (`@attrs.define(frozen=True, slots=False, kw_only=True)`), mirroring `ScaleogramResult`/`RidgeResult` (with the deliberate `is_degenerate` divergence, §4).
- New shared SG-derivative primitive `_noise.compute_sg_derivative(x, window, polynomial_order, deriv, delta=1.0, mode="interp")` (covers deriv 0/1/2 from one `savgol_filter` polynomial).
- New pure-geometry curvature helper `_geometry.compute_path_curvature(x_dot, y_dot, x_ddot, y_ddot) -> np.ndarray`, sibling to `compute_psi_g`/`compute_signed_area`, sign pinned by an absolute hand-built-input test, with a y-down disambiguation in the docstring.
- Reuse of `temporal_cwt._validate_cadence_s` (imported, not re-implemented — PR #7 precedent).
- Determinism canary `scripts/circumnutation/capture_midline_canary.py` (analytic circle oracle + synthetic-generator drift detector).
- Foundation-test migration (§9) + spec-prose migration (§9b).
- **theory.md §6.2 patch**: a y-down sign-convention clarification + Appendix B(Corrections) line, so the curvature sign anchor doesn't contradict the §6.2 "left turn" prose for PR #9/#10 readers (§13/R1-I1; the doc already carries an Appendix B — PR #7 precedent).

### Out of scope (explicit non-goals)
- The `L_gz` growth-zone mask (CC-1 — constructed in PR #10). **Disambiguation (load-bearing):** `velocity_sub_noise_mask` is a **per-FRAME, time-domain** noise-floor mask (`|v| ≤ k·σ_v`, theory §6.2); the `L_gz` mask is a **per-ARC-LENGTH, apical-region** mask built from PR #9's detected peak. They share the `σ_v`/`NOISE_MASK_K` vocabulary but are **different objects in different domains** — PR #9/#10 MUST NOT reuse one as the other.
- Uniform-arc-length resampling of `κ(s)` (deferred to PR #9). **PR #8→#9 handoff contract:** PR #9's locked stub `spatial_cwt.compute_scaleogram(kappa, ds, constants=None)` takes a scalar grid spacing `ds`, i.e. it presumes curvature already on a uniform-`ds` grid. PR #8 deliberately emits `curvature_px_inv` on the **native non-uniform** `arc_length_px` grid; **PR #9 owns the resample** that converts `(MidlineResult.curvature_px_inv, MidlineResult.arc_length_px)` → `(kappa_uniform, ds)` (where the `ds`/spatial-Nyquist `NYQUIST_RATIO_MAX` decision intrinsically lives). `arc_length_px` (non-uniform) + `curvature_px_inv` is the complete, sufficient input for that resample — no field is missing.
- Any trait column emission, `trajectory_df` ingestion, or 5-tuple groupby.
- Any new `ConstantsT` field / `_CONSTANTS_VERSION` bump (stays 5).
- Spatial CWT, parametric tiers, pipeline orchestration, CSV/sidecar wiring (later PRs).
- Multi-plate empirical validation beyond plate-001 (cf. #220).

## 3. Public API & module shape

```python
def reconstruct(
    x: np.ndarray,
    y: np.ndarray,
    cadence_s: float,
    sg_window: Optional[int] = None,
    constants: Optional[ConstantsT] = None,
) -> MidlineResult:
    ...
```

- `x`, `y`, `cadence_s` **required positional** (matching `compute_scaleogram(x, cadence_s, …)`); the PR #1 stub's all-`None` defaults (a foundation-probe affordance) are dropped on graduation.
- `sg_window=None` → `constants.SG_WINDOW_SHORT` (5). **`sg_window` is kept (not dropped to `ConstantsT` only):** it matches the existing stub code's parameter and the user-specified signature, and `smooth_ridge(ridge_result, window=None, …)` is the package precedent for a per-call SG-window knob. The **window/degree asymmetry** (window is a param; degree is not) is deliberate: the smoothing *scale* (window) is the knob a caller realistically sweeps to trade noise-suppression vs. spatial resolution of `κ`; the polynomial *degree* is a fixed methodological choice (`SG_DEGREE=3`) shared across the package's SG helpers. Degree comes from `constants.SG_DEGREE`.
- `cadence_s` is validated and stored, but the core outputs (`arc_length_px`, `curvature_px_inv`, the mask) are **cadence-independent** (frame-parameterized, §4) — `cadence_s` is provenance + a hook for a caller's optional physical-time conversion. (Tier 0 `kinematics.compute` drops `cadence_s` entirely for the same cadence-independence reason; PR #8 keeps it because the stub signature locks it.)
- `constants=None` → `ConstantsT()` via `_check_constants`.

## 4. `MidlineResult` dataclass

Frozen `attrs` container, immutable-by-convention (same as `ScaleogramResult`). **Frame-parameterized, cadence-independent** (round-1 R2-B1): velocity is **px/frame** (the program's cadence-independent convention, matching Tier 0 velocity and Tier 2 `delta_E_..._px_per_frame`; `px/s` was deliberately rejected per theory §10 Appendix B). Arc length integrates speed over the frame index (`dx=1`), giving px; curvature is parameterization-invariant (px⁻¹ regardless of time unit).

Per-frame arrays, length `n` (= `len(x)`), index `i` ↔ frame `i`:

| field | dtype | unit | definition |
|---|---|---|---|
| `frame_indices` | int64 | — | `np.arange(n)` |
| `x_smooth_px` | float64 | px | SG deriv=0 of `x` |
| `y_smooth_px` | float64 | px | SG deriv=0 of `y` |
| `speed_px_per_frame` | float64 | px/frame | `√(ẋ² + ẏ²)`, ẋ/ẏ = SG deriv=1 (delta=1 frame) |
| `arc_length_px` | float64 | px | `scipy.integrate.cumulative_trapezoid(speed_px_per_frame, dx=1.0, initial=0)`; monotonic non-decreasing, `s[0]=0` |
| `curvature_px_inv` | float64 | px⁻¹ | `_geometry.compute_path_curvature(ẋ, ẏ, ẍ, ÿ)` (ẍ/ÿ = SG deriv=2); non-finite entries swept to NaN (§5) |
| `velocity_sub_noise_mask` | bool | — | `True ⇔ speed_px_per_frame ≤ NOISE_MASK_K · σ_v` (flagged sub-noise; consumers use `κ[~mask]`) |

**Single curvature array (reconciles the stub docstring's `κ_path`/`κ_arc`).** Curvature is a parameterization-invariant geometric scalar; the identity `κ_path(τ)=κ(s(τ))` (§6.1) makes the time-domain and arc-length-domain curvatures **bit-identical numbers**, so storing two arrays would be pure redundancy (worse than `RidgeResult.powers`, which is at least algebraically derived, not identical). `MidlineResult` stores **one** `curvature_px_inv`: pair it with `frame_indices` for the time view (`κ_path`), or with `arc_length_px` for the arc view (`κ_arc`). **The arc view is non-uniformly sampled** (speed varies per frame) — a consumer computing `dκ/ds` (PR #9's decay fit) MUST use `arc_length_px` as the axis, not assume uniform spacing. The PR #1 stub docstring (which named two return fields `κ_path`/`κ_arc`) is **updated** by this PR's implementation to describe the single array + two views.

Scalars (provenance / reproducibility):

| field | type | meaning |
|---|---|---|
| `cadence_s` | float | resolved cadence (provenance; not used in the cadence-independent core) |
| `sg_window` | int | resolved SG window actually used |
| `sg_degree` | int | resolved SG degree (`constants.SG_DEGREE`) |
| `sigma_v_px_per_frame` | float | `np.std(speed_px_per_frame, ddof=0)` used for the mask (NaN on degenerate) |
| `noise_mask_k` | float | `constants.NOISE_MASK_K` used |
| `is_degenerate` | bool | `True` when SG could not be applied (`n < sg_window`, `n=0`) or the **raw** track is stationary (§5) → per-frame arrays are all-NaN |

**Deliberate divergence from the PR #5 Result shape (R2-I1).** `ScaleogramResult`/`RidgeResult` carry *only data fields* — degenerate signaling in this package otherwise lives in the *trait-emitting* `compute` modules (`psi_g._all_degenerate_traits`, `nutation`'s NaN-gate), and the machinery layer *raises* on contract violations. `MidlineResult` instead carries a graceful `is_degenerate` flag + all-NaN payload. This is a **deliberate, documented divergence**: an all-NaN `MidlineResult` is the right degenerate output for a per-track reconstruction primitive (an exception would force every future per-track caller to wrap try/except), and an explicit boolean is cleaner for consumers than probing `np.isnan(curvature).all()`. The package already accepts justified Result-field choices (cf. `RidgeResult.powers` "intentional and locked" redundancy); this divergence is justified the same way.

## 5. Algorithm (`reconstruct` orchestration)

1. **Resolve** constants (`_check_constants(constants) or ConstantsT()`) and `sg_window`.
2. **Validate inputs (RAISE, field-named — §7):** `x`/`y` 1-D real-numeric finite ndarrays of equal length; `cadence_s` via `temporal_cwt._validate_cadence_s`; `sg_window` positive odd int with `SG_DEGREE < sg_window` (the `compute_sg_derivative` validator is reused). **The non-finite check precedes the degenerate gate** (a short all-NaN track raises, it does not return graceful-NaN — §7).
3. **Degenerate gate (GRACEFUL — returns BEFORE any `np.std`/`hypot`/`cumtrapz`):** if `n == 0`, **or** `n < sg_window`, **or** the raw track is stationary (`np.ptp(x) == 0 and np.ptp(y) == 0`, mirroring `_geometry.project_to_growth_axis_perpendicular`'s zero-net-displacement precedent), return an all-NaN `MidlineResult` with `is_degenerate=True`, arrays length `n`, `sigma_v_px_per_frame=NaN` **assigned literally** (never call `np.std` on an empty/degenerate array — `np.std([])` emits a RuntimeWarning, R3-I1). This gate is what makes the "stationary → all-NaN" contract realizable: after SG, speed is floating-point *dust* (~1e-18), never exactly 0, so stationarity must be detected on the **raw** input (R3-B1). **Disjunct order is load-bearing:** `n == 0` MUST be the first disjunct so Python `or` short-circuits before `np.ptp(x)` (which raises `ValueError` on an empty array, R2-Q2).
4. **SG derivatives:** `x_smooth, ẋ, ẍ = compute_sg_derivative(x, w, d, deriv∈{0,1,2}, delta=1.0)`; same for `y`. (deriv=0 ignores `delta`.)
5. **Speed:** `speed = np.hypot(ẋ, ẏ)` (px/frame).
6. **Arc length:** `scipy.integrate.cumulative_trapezoid(speed, dx=1.0, initial=0)` (px).
7. **Curvature:** `κ = _geometry.compute_path_curvature(ẋ, ẏ, ẍ, ÿ)` (px⁻¹). Computed under `np.errstate(divide="ignore", invalid="ignore", over="ignore")` (the `over` category is included so even non-physical huge-magnitude input cannot emit `RuntimeWarning('overflow encountered in square')` from the `|v|³` term — R2-I1); **then a post-division `κ[~np.isfinite(κ)] = np.nan` sweep** guarantees no `inf`/`-inf` leaks at the underflow/overflow corner (R3-B2). Note: large *finite* κ at near-stationary frames is left intact in the raw array (those frames are flagged by `velocity_sub_noise_mask`; on real plate-001 every `|κ|>1` frame is masked, and `max|κ[~mask]| ≈ 0.09–0.17`), so consumers use `κ[~mask]`.
8. **Velocity mask:** `σ_v = np.std(speed, ddof=0)`; `mask = speed <= NOISE_MASK_K * σ_v`.
9. **Assemble** the frozen `MidlineResult`; `logger.debug` a one-line lazy-%-formatted summary (CC-9).

No code path emits an `np.RuntimeWarning` (verified round 1–2: the `n=0` early return avoids `np.std([])`; the `errstate(divide,invalid,over)`+`~isfinite` sweep avoids divide/invalid/overflow warnings and `inf`; `np.std` on a non-empty all-NaN array does not warn). A `warnings.simplefilter("error")` test pins this for the n=0, stationary, and a deliberately-large-magnitude curvature input.

## 6. New shared helpers

### 6.1 `_noise.compute_sg_derivative(x, window, polynomial_order, deriv, delta=1.0, mode="interp")`

Thin `scipy.signal.savgol_filter(x, window, polynomial_order, deriv=deriv, delta=delta, mode=mode)` wrapper. **Param named `polynomial_order`** for consistency with the sibling `compute_sg_detrended(x, window, polynomial_order)` (R2-M1). Reuses the exact window/`polynomial_order` boundary-validation (int/odd/positive window; non-negative int order `< window`), and **additionally** validates `0 ≤ deriv ≤ polynomial_order`. **Corrected rationale (R3-I2):** for `deriv > polynomial_order` scipy *silently returns all-zeros* (a silent-wrong-answer hazard, worse than a raise), and for `deriv < 0` it raises an opaque `factorial()` error; the validator converts both into field-named errors. Returns `np.full(len(x), np.nan)` (length-preserving) when `len(x) < window`, mirroring `compute_sg_detrended`'s short-input contract.

- `mode="interp"` default (the derivative-appropriate boundary policy: edge windows are fit by `np.polyfit`/lstsq and the derivative evaluated there). Kept as a parameter (not hardcoded like `compute_sg_detrended`'s `"nearest"`) because this helper is an explicitly *general* SG-derivative primitive (`deriv`/`delta` are already parameters); `"interp"` is the only mode `reconstruct` uses.
- Lives in `_noise.py` next to the other SG wrappers; adding a NEW function does not touch existing `_noise` consumers.
- **scipy import discipline (R5-I3):** this PR converts `_noise.py` to module-qualified `import scipy.signal` and updates the existing three call sites from bare `savgol_filter(` to `scipy.signal.savgol_filter(` (removing the lone `from scipy.signal import savgol_filter` violation), and `reconstruct`'s orchestrator uses `import scipy.integrate` / `scipy.integrate.cumulative_trapezoid`. The existing-helper rename is mechanical; its consumers (`kinematics`, `qc`, `nutation`, `psi_g`) are re-run in the per-pair gate.

### 6.2 `_geometry.compute_path_curvature(x_dot, y_dot, x_ddot, y_ddot) -> np.ndarray`

Pure geometry — returns `κ` (px⁻¹):

```
κ = (ẋ·ÿ − ẏ·ẍ) / (ẋ² + ẏ²)^{3/2}
```

This is the *literal standard math curvature formula* (theory §6.2 verbatim). Raises `ValueError` on length-mismatched input arrays (matching the `compute_psi_g`/`compute_signed_area` sibling guard, R2-M2). Length = `len(x_dot)`, float64.

**Sign convention (load-bearing), pinned by absolute hand-built anchors** (numerically verified round 0/1, not internal-agreement tests):
- `compute_path_curvature([1.0],[0.0],[0.0],[1.0]) == +1.0` (unit velocity +x, unit acceleration +y).
- Math-CCW unit circle `(cos t, sin t)` → `κ = +1/R`; clockwise `(cos t, −sin t)` → `−1/R`.

**y-down disambiguation (R1-I1 — must be in the docstring).** theory §6.2 labels `κ>0` as "left turn"; that is the **standard y-up math convention** this formula encodes. In the **y-down image frame the pipeline actually runs in**, `+κ` corresponds to a **clockwise/visual-right turn as displayed** (a tip moving +x and curving toward screen-top — a visual left turn — gives `κ = −1`). Per the `_geometry.py` precedent we **anchor on the formula sign, not the word** (exactly as `compute_signed_area`'s docstring spells out "+1 is clockwise in y-up math axes and counterclockwise as displayed in the y-down image frame"). PR #9/#10, which consume `κ`'s sign for chirality/`L_gz` work, inherit this anchored convention; theory §6.2 is patched (§2) with the same y-down note so a theory-only reader does not build an inverted sign trait.

- `|v| = 0` exactly → `κ = NaN` (errstate-guarded); non-finite results swept to NaN by the caller (§5 step 7). The helper itself never emits a RuntimeWarning.

## 7. Degenerate / error handling (split policy)

- **Raise** `TypeError`/`ValueError` (field-named, CC-1): `x`/`y` not ndarray; not 1-D; complex/object/non-numeric dtype; **non-finite (NaN/±inf) — rejected, not dropped**, because SG and `cumulative_trapezoid` assume uniform frame spacing and dropping NaN frames would silently break it; `len(x) != len(y)`; `cadence_s` invalid; `sg_window` invalid. **Order:** the non-finite check precedes the degenerate gate, so a short *all-NaN* track raises (it does not return graceful-NaN).
- **Graceful all-NaN `MidlineResult`** (`is_degenerate=True`, no raise, no warning): `n == 0`; `n < sg_window`; **raw-stationary** (`ptp(x)==0 and ptp(y)==0`). Traced from `n=0` up (PR #7 SG-window-exceeds-signal lesson): `n=0` → length-0 arrays (gate returns before `cumulative_trapezoid([])`, which would raise, and before `np.std([])`, which would warn); `0 < n < sg_window` → all-NaN length-`n`; `n ≥ sg_window` non-stationary → real reconstruction.

## 8. Determinism (CC-6)

- **Same process:** bit-identical (`atol=0`) — `savgol_filter`, `cumulative_trapezoid`, `np.std(ddof=0)`, `np.hypot` all verified deterministic in-process. No `argmax` anywhere in the `MidlineResult` path (no tie-break risk; peak-finding is PR #9).
- **Cross-OS (Ubuntu/Windows/macOS): `atol=1e-9, rtol=0`** — matching PR #5's verified baseline, **not** PR #6/#7's 1e-6 (R5-B1, **revised**). PR #6's 1e-6 was a *coverage* argument for its 4-path scipy trait stack (fft+signal+ndimage+stats), not a measured floor, and does not transfer. Measured for PR #8's narrow savgol+cumtrapz stack: `savgol_coeffs` solves a *well-conditioned* lstsq (`cond(A) ≈ 11.6` at window 5/deg 3; ≈92 at window 11); cross-LAPACK-driver coefficient differences ≈1e-18; full-pipeline ULP-perturbation propagation through `deriv0/1/2 → speed → cumtrapz → κ=(ẋÿ−ẏẍ)/|v|³` ≈ **1.6e-14**, and ≈3.5e-14 even in the worst near-stationary case (`|v|³` down to ~2.5e-11). `1e-9` leaves ~5 orders of headroom while still catching real regressions (1e-6 could mask ~100%-of-range curvature errors). If a CI runner later exceeds 1e-9, widen *then* with the failing diff captured (PR #5 canary procedure).
- **Canary** `scripts/circumnutation/capture_midline_canary.py` locks **two** inputs (R5-B2): (a) a **closed-form analytic oracle** — a pure circle radius `R`, where `κ ≡ 1/R` exactly (verified interior-frame recovery error ≈1.8e-5 vs 1/R; a self-evident expected value, and the only input that exercises *sustained* constant-sign curvature — the synthetic generator cannot); (b) the **synthetic generator** `generate_trajectory(random_state=0, n_frames=128, …)` as a drift detector. Both asserted within the cross-OS `atol=1e-9`.
- **scipy imports** module-qualified throughout the new code (§6.1).

## 9. Foundation-test migration (ATOMIC with the first non-raising commit)

PR #7's BLOCKING finding: the migration MUST land in the same commit that makes `reconstruct` stop raising `NotImplementedError`. Edits to `tests/test_circumnutation_foundation.py` (line numbers verified round 1):

1. **Remove** `("midline", "reconstruct", 8)` from `STUB_MODULES` (line 50) → 4 remaining stubs.
2. **Remove** `("midline", "reconstruct")` from `STUBS_WITH_CONSTANTS_KWARG` (line 850).
3. **Add** `("midline", "reconstruct")` to `IMPLEMENTATIONS_WITH_CONSTANTS_KWARG` (lines 859–866) with a trailing `# added by PR #8 (stub→impl)` comment matching the per-row convention.
4. **Add** `"midline"` to the explicit list in `test_module_logger_is_namespaced` (lines 745–777).
5. **Add** an `elif module_name == "midline":` branch to `test_implementation_accepts_constants_kwarg` (before the `else` at line 967) constructing **raw 1-D float64 arrays** (NOT a DataFrame — midline is the first `IMPLEMENTATIONS_WITH_CONSTANTS_KWARG` entry whose callable takes arrays): e.g. `n=8` finite `x`/`y`, `reconstruct(x, y, 300.0, constants=ConstantsT())`, assert a `MidlineResult` is returned.
6. **Append a PR #8 paragraph to the `STUB_MODULES` block comment** (lines 38–48) documenting the midline stub→impl move (the four sub-moves above + no `_CONSTANTS_VERSION` bump), matching every prior transition PR's in-file audit-trail paragraph (R4-BLOCKING).

Counts after PR #8: implementation modules 6 → **7**; stub modules 5 → **4**.

## 9b. Spec-prose migration (MODIFIED Package layout) — discrete sub-edits

The spec-prose migration is **independent** of the test-file migration (§9) and is where round-1 found gaps (R4-IMPORTANT). The MODIFIED `Package layout` requirement (`openspec/specs/circumnutation/spec.md` lines 6–100) needs ALL of:
- (a) impl-module list (line 10): add `midline` ("implemented from PR #8 onward; see Requirement: Tier 3a midline reconstruction API") → 7.
- (b) stub-module list (line 11): drop `midline` → "4 stub modules: `spatial_cwt`, `parametric`, `plotting`, `pipeline`".
- (c) stub-callable table (line 17): remove the `midline` row.
- (d) "remaining stub" scenario (lines 36–41): change "5 remaining stub modules (`midline`, …)" → "4 remaining stub modules (`spatial_cwt`, `parametric`, `plotting`, `pipeline`)".
- (e) trailing "no longer raises" prose (line 23): add `midline.reconstruct` to the enumeration of valid-input non-raising callables.
- (f) line-25 helper pointer: add a parenthetical routing `_noise.compute_sg_derivative` / `_geometry.compute_path_curvature` to the new Tier 3a requirement.
- (g) a new `#### Scenario: midline.reconstruct is callable on valid arrays without raising` (mirroring the `temporal_cwt`/`nutation`/`psi_g` callability scenarios).
- (h) a `**Scope note on PR #8 stub-to-implementation transition.**` paragraph (mirroring the PR #6/#7 scope notes).

**Spec home for the two new helpers (R4-IMPORTANT).** Per the PR #7 `compute_signed_area` precedent, the helpers are spec'd **inside the new ADDED "Tier 3a midline reconstruction API" requirement** — with a discrete `#### Scenario:` for `compute_path_curvature`'s **absolute sign anchor** (load-bearing) and one for `compute_sg_derivative`'s deriv-range/short-input behavior. The frozen **"Tier 0 helper modules" requirement is NOT modified** (that would wrongly imply the helpers shipped in PR #2/#7).

## 10. Validation / test plan (TDD, RED→GREEN pairs)

- **`_noise.compute_sg_derivative`** (unit, isolated): polynomial-exactness (deriv of a degree-≤order polynomial recovered to machine precision — verified: `2t²+3t+1` → deriv1=`4t+3`, deriv2=`4`); `delta` scaling; deriv-range validation (incl. the silent-zeros `deriv>order` case); short-input all-NaN; window/order validation reuse.
- **`_geometry.compute_path_curvature`** (unit, isolated): **absolute sign anchor** (`[1],[0],[0],[1] → +1.0`); straight line `κ≈0`; circle radius `R` → `κ≈+1/R` (CCW) / `−1/R` (CW); `|v|=0 → NaN`; length-mismatch `ValueError`; no RuntimeWarning.
- **`midline.reconstruct`**: arc-length monotonicity + `s[0]=0`; speed = `√(ẋ²+ẏ²)`; mask polarity; every degenerate `n` from 0 up + raw-stationary + non-finite-raise-vs-graceful order; no-RuntimeWarning under `warnings.simplefilter("error")` for n=0/stationary; determinism (same-process atol=0; canary cross-OS atol=1e-9 incl. the circle oracle).
- **Real plate-001** (`tests/data/circumnutation_nipponbare_plate_001/plate_001_greyscale.tracked_proofread.slp`, 575×6, via `Series.load(series_name="plate_001", primary_path=…).get_tracked_tips()`, track_id strings `"track_<i>"`): per real track —
  - `arc_length_px` monotonic, `curvature_px_inv` finite (no inf/NaN on non-degenerate frames);
  - **`max|curvature_px_inv[~velocity_sub_noise_mask]|` below a physical floor** (e.g. `< 1 px⁻¹`; observed ≈0.09–0.17) — the *unmasked* array is the plausible one, the full raw array is not (R1-M1);
  - **mask fraction in a sane band and recorded** (observed 0.38–0.61; assert e.g. `0.1 < frac < 0.75` and log the value) — this replaces the false "flags ~nothing" expectation (R1-B1).
- **Cross-tier acceptance** (R1-M2): `arc_length_px[-1]` ≈ Tier 0 path length within a documented tolerance (observed within 2.1–3.1%; tolerance ~5%). Tier 0's `L = Σ‖diff(xy)‖` (kinematics.py) is recomputed/recovered; the test imports `kinematics`. **Caveat to document:** Tier 0 NaN-drops gap frames before summing, so on a track *with* dropouts the two path lengths diverge by the gap chords — plate-001 has no gaps, so it holds here.

## 11. Constants / units / version

- **No new `ConstantsT` field, no `_CONSTANTS_VERSION` bump (stays 5).** Reuses `NOISE_MASK_K`, `SG_WINDOW_SHORT`, `SG_DEGREE`. `mode="interp"` is a documented default, not a field. The two helpers are *functions*, not constants. PR #7 lesson: the version tracks the constants payload, not the PR count.
- **Units (R2-B1, revised):** physical fields use **program-convention** units — `px` (`x_smooth_px`, `arc_length_px`), `px/frame` (`speed_px_per_frame`, `sigma_v_px_per_frame`; in `PIPELINE_UNIT_VOCABULARY`, and matching Tier 2's `delta_E_..._px_per_frame` precedent), `px⁻¹` (`curvature_px_inv`). **`px/s` is NOT used** (theory §10 Appendix B rejected it). `MidlineResult` fields are not CSV trait columns, so `PIPELINE_UNIT_VOCABULARY` (which gates *sidecar* unit strings) does not bind them; `px⁻¹` has no vocabulary token because no trait column currently carries it — a later PR emitting midline-derived *trait columns* owns any vocabulary addition.

## 12. Risks / trade-offs

- **The velocity mask flags ~40–60% of real frames — this is correct, not over-aggressive (R1-B1, corrected).** On plate-001 the rice tip moves only ~5.5 px/frame while SLEAP localization noise is ~1.85 px (velocity-noise ~2.6 px/frame), so ~half the frames genuinely have sub-2σ motion (theory §6.2's exact target). std(speed) coincides with a principled noise-floor (√2·σ_pos) here (both ~40–50%), validating the self-contained std(speed) choice. The downstream consequence — PR #9's spatial CWT sees a ~50%-sparse `κ(s)` — is the *honest* signal (unreliable frames SHOULD be masked); PR #9/#10 own coping with sparsity. PR #10 may revisit the σ_v definition when the mask is first *consumed* for traits.
- **Curvature noise amplification:** 2nd derivatives amplify ~1-px tip noise; mitigated by SG-smoothing before differentiation (§6.2) + the velocity mask. PR #8 does not consume `κ` for traits; the real-data `max|κ[~mask]|` test guards gross blow-up.
- **Cross-OS determinism:** float fields hold to `atol=1e-9` (measured ~1e-14 headroom); `frame_indices` (int) and `is_degenerate` (bool) exact.
- **spec↔theory consistency:** the single-`κ`-array decision, the two-mask disambiguation, and the y-down curvature-sign note are recorded here, in the spec rationale, AND (sign) in a theory.md §6.2 patch, so no artifact contradicts another.

## 13. Reconciliation log (critical-review rounds 1–2)

Round 1: five reviewers (numerical/sign, API/dataclass, degenerate/edge, scope/migration/spec, determinism), all numerically verified on real plate-001. Round 2: one holistic re-review verifying the fixes hold + no new contradictions (no BLOCKING; px/frame consistency, gate ordering, AND-stationary logic, inf-sweep, determinism, and §9/§9b line numbers all confirmed solid).

**BLOCKING**
- **R1-B1 (mask rationale empirically false):** §12 claimed std(speed) "flags ~nothing." Measured: 38–61% on real plate-001. *Resolution:* the masking is *correct* (tip moves only ~3× the noise floor; std(speed) ≈ noise-floor here) — kept the std(speed) choice, **rewrote §12** to state the measured reality, and **added a §10 real-data test** asserting the observed mask-fraction band + `max|κ[~mask]|<1 px⁻¹`. The premise changed; the decision (validated) stood.
- **R2-B1 (`px/s` violates program convention):** §4 used `_px_per_s`. *Resolution:* **switched to px/frame** (frame-parameterized arc length, `dx=1.0`; curvature is time-unit-invariant). `cadence_s` retained as provenance only. Updated §3/§4/§11.
- **R3-B1 (stationary undetectable post-SG):** SG dust ⇒ speed never exactly 0 ⇒ `is_degenerate`/mask/curvature-NaN claims unrealizable. *Resolution:* **detect stationarity on the RAW input** (`ptp(x)==0 and ptp(y)==0`) in the degenerate gate (§5 step 3).
- **R3-B2 (`|v|=0`-only guard lets near-stationary κ blow up; inf leaks):** *Resolution:* kept the errstate guard, **added a post-division `κ[~isfinite]=NaN` sweep** (no inf), documented that large *finite* κ at low-speed frames is intentionally retained and flagged by the mask (§5 step 7).
- **R4-BLOCKING (`STUB_MODULES` block-comment audit trail):** *Resolution:* added §9 item 6 (PR #8 comment paragraph + per-row `# added by PR #8`).
- **R5-B1 (atol=1e-6 too loose):** measured full-pipeline ULP propagation ≈1e-14. *Resolution:* **tightened cross-OS to `atol=1e-9, rtol=0`** with the measurement in §8; documented why PR #6's 1e-6 doesn't transfer.
- **R5-B2 (canary lacks analytic oracle):** *Resolution:* canary now locks a **closed-form circle (`κ≡1/R`)** plus the synthetic generator (§8).

**IMPORTANT**
- **R1-I1 (theory §6.2 "+ = left turn" inverted in y-down):** *Resolution:* added y-down disambiguation to the `compute_path_curvature` docstring (§6.2) + a **theory.md §6.2 patch** (§2 in-scope).
- **R2-B2 / R4 (PR #9 resample handoff):** *Resolution:* explicit handoff contract in §2 (PR #9 owns `(curvature_px_inv, arc_length_px) → (kappa_uniform, ds)`).
- **R2-I1 (`is_degenerate` diverges from PR #5 Result shape):** *Resolution:* added the "deliberate divergence" justification (§4).
- **R2-I2 (`sg_window` YAGNI / window-vs-degree asymmetry):** *Resolution:* kept `sg_window` (stub-code + `smooth_ridge` precedent), **justified the asymmetry** (§3).
- **R2-I3 (single-array vs stub docstring):** *Resolution:* documented the single-array decision + non-uniform arc view + stub-docstring update (§4).
- **R3-I1 (`np.std([])` warns on n=0):** *Resolution:* degenerate gate returns before any `np.std`/`hypot`/`cumtrapz`, `sigma_v` set literally NaN (§5 step 3).
- **R3-I2 (deriv-validator rationale wrong):** *Resolution:* corrected the rationale (scipy silently zeros for `deriv>order`) (§6.1).
- **R4-IMPORTANT (spec-prose migration under-enumerated; helper spec home):** *Resolution:* added **§9b** enumerating all 8 Package-layout sub-edits + the new-helper spec home (inside the Tier 3a requirement, NOT the frozen Tier 0 helper requirement).
- **R5-I3 (scipy import discipline):** *Resolution:* module-qualified imports for new code + convert `_noise.py`'s existing bare `savgol_filter` import (§6.1).

**MINOR**
- **R1-M1** (test `max|κ[~mask]|`, not just "finite") → §10. **R1-M2** (cross-tier ~5% tol + gap caveat) → §10. **R2-M1** (`polynomial_order` naming) → §6.1. **R2-M2** (length-mismatch guard) → §6.2/§10. **R3-M1** (non-finite-before-length order) → §7. **R5-M4** (`ddof=0`; interp edges use polyfit/lstsq; no argmax) → §4/§8.

**Round 2**
- **R2-I1 (errstate `over` category):** `errstate(divide,invalid)` doesn't suppress overflow warnings from `|v|³` on non-physical huge input. *Resolution:* added `over="ignore"` (§5 step 7) + a large-magnitude no-warning test (§5/§10); softened the unconditional "no RuntimeWarning" prose.
- **R2-Q2 (disjunct order):** `np.ptp([])` raises, so `n==0` must be the first `or` disjunct. *Resolution:* documented the load-bearing order (§5 step 3).

**CONFIRMED OK (no change):** no `_CONSTANTS_VERSION` bump; §9 items 1–5 line-accurate; counts impl 6→7 / stub 5→4; `_geometry` 4-derivative pure-geometry split; trapezoid-vs-segment immaterial; `frame_indices` int64; circle-canary recovery ≈1.78e-5 + bit-identical repeat; §9b 8-edit enumeration complete (line-32 import scenario correctly needs no edit); theory §6.2 patch well-scoped; `_noise.py` scipy-import conversion adequately scoped.

### Round 3 — /openspec-review (5-subagent proposal review)

Verdict across reviewers: APPROVE / 9-of-10 / implementation-ready; the PR #7 BLOCKING (dropped sibling scenarios) is ABSENT (all 13 preserved, counts correct), `--strict` valid. **No BLOCKING.** IMPORTANT findings reconciled into the OpenSpec change files (proposal/design/tasks/spec):
- **R5-sign-collision (highest, publication-risk):** ψ_g family's swapped `atan2(dx,dy)` ⇒ `sign(κ) == −handedness`. *Resolution:* spec `compute_path_curvature` paragraph + a new cross-helper sign scenario + tasks §2.1 cross-helper test + theory §6.2 patch (§9.1) all now state `sign(κ) == −handedness`.
- **R4-cadence-independence-test-gap:** spec scenario had no test. *Resolution:* added the RED cadence-independence test to tasks §5.1.
- **R4/R5-canary chicken-and-egg + tolerance conflation:** *Resolution:* tasks §7.1 hardcodes `_MIDLINE_CANARY_EXPECTED` (à la `_CANARY_EXPECTED_VALUES`) and SEPARATES the oracle `κ≈1/R` (loose `atol≈1e-3`) from the cross-OS reproducibility `atol=1e-9`; spec determinism scenario updated.
- **R4-mask-band-headroom:** 0.75 upper bound too thin. *Resolution:* widened to 0.85 in spec + tasks §8.1.
- **R5-mask-over-rationalization / R5-cross-tier-data-dependence:** *Resolution:* softened spec + design D7/Risks prose (data-specific, SNR-insensitive, not a general identity); spec cross-tier scenario gained the robust `arc ≤ L` invariant + `σ_pos` recording.
- MINOR: tasks corrected "5 call sites across 3 functions"; elif line ref ~937 (not 967) + explicit `cadence_s=300.0`; §1 gate → full suite (kinematics is a `_noise` consumer); §8 `skipif(fixture.exists())`; §1.1 `deriv=0` + `deriv==order` cases; §6.1 parametrize `n∈{0..5}`; `sg_window/sg_degree/noise_mask_k` value asserts; §5/§6 RED-vs-contract-locking note; spatial_cwt extra-kwargs drift noted as informational. `openspec validate --strict` re-passed after edits.

### Round 4 — user-requested second adversarial pass on the reconciled change files

3 reviewers (numeric sign-verification, reconciliation-landed verification, completeness critic). **`sign(κ) == −handedness` VERIFIED TRUE** (exact, proven `dψ_g/dt = −κ·|v|`; the spec scenario is correctly scoped to a single-signed circle — tightened the docstring requirement to add the per-frame identity + single-signed scope). The completeness critic found 3 real defects prior rounds missed:
- **R4-B1 (spec self-contradiction):** the degenerate path said "all arrays NaN" but `velocity_sub_noise_mask` is `bool` — `np.full(n, nan, dtype=bool)` silently yields all-`True` (verified). *Resolution:* spec + tasks §6.1 now specify `velocity_sub_noise_mask = np.zeros(n, dtype=bool)` (all-`False`) on the degenerate path; the §6.1 assertion is `mask.dtype==bool and not mask.any()` (NOT `np.isnan(mask)`).
- **R4-B2 (canary non-reproducible):** circle `R`/span/center unspecified. *Resolution:* pinned `R=50.0`, `theta=np.linspace(0, 2π, 128, endpoint=False)`, center `(0,0)` in spec + tasks §7.1.
- **R4-B3 (undocumented gate divergence):** the full-circle canary is net-zero-displacement; `reconstruct` gates on `ptp` while `project_to_growth_axis_perpendicular` gates on net displacement. *Resolution:* documented the deliberate divergence in design D8 + a spec note ("closed loop has a curvature midline but no growth axis").
- **R4-I1 (`MidlineResult.__eq__`):** ndarray `__eq__` is fragile/ambiguous. *Resolution:* added `eq=False` to the attrs decorator (deliberate improvement over the `ScaleogramResult` template) + field-by-field `np.array_equal` in the determinism test (tasks §7.1). (Note: a single-field `==` did not reproduce a raise in testing, so the rationale is softened to "ambiguous/can raise," not "always raises.")
- **R4-I2 (synthetic canary):** DataFrame→array extraction unstated (a `pd.Series` isn't an ndarray → would trip the type guard). *Resolution:* tasks §7.1 spells out `df["tip_x"].to_numpy(dtype=np.float64)` + the `noise_sigma_px=0.5` RNG-path/NEP-19 note.
- **R4-I3 (validation ordering):** `n=0` + bad `cadence_s` precedence undefined. *Resolution:* spec states ALL field-named validation runs first unconditionally → validation wins over the graceful path.
- **R4-MINOR:** constructed exact-`|v|=0` + constructed inf-pre-sweep inputs added to §2.1/§6.1 for honest coverage (not pragma).
- **Reconciliation-verifier** confirmed all openspec-review edits landed in spec.md/tasks.md and flagged 3 stale spots in design.md (`0.75`→`0.85`; "3 call sites"→"5 across 3 functions"; softened Risks lead) — all fixed. `openspec validate --strict` re-passed; stale-value grep clean.

## 14. Open questions

- Real plate-001 cross-tier tolerance + mask-fraction band are captured/asserted at GREEN against pre-committed floors (finite, monotonic, `max|κ[~mask]|<1`, `0.1<frac<0.75`, arc within ~5%) and the observed values recorded — auditable, not self-fulfilling (PR #7 GREEN-phase discipline).
- Whether to opportunistically convert `compute_sg_detrended`'s bare `savgol_filter` call during the `_noise.py` import cleanup, or grandfather it — leaning convert (mechanical, same PR, removes the lone violation), decision recorded in §13/R5-I3.
