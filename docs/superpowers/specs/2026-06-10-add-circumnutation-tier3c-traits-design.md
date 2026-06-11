# Design: add-circumnutation-tier3c-traits (PR #10)

**Date**: 2026-06-10
**Author**: eberrigan (+ Claude)
**Epic**: #197 — circumnutation analysis program
**Upstream (merged)**: PR #8 midline (#227), PR #9 spatial CWT (#231)
**Blocks-on**: #230 (L_gz/L_c tip-trail transfer) — see Scope
**Evidence**: `docs/circumnutation/investigations/2026-06-10-tier3c-traveling-wave/report.md`

## Context

Tier 3c is the **trait-emission** tier that turns PR #9's spatial-CWT machinery into
per-plant traits, and the first consumer of that machinery. Its headline output is
`traveling_wave_residual` — the program's central falsifiable test of the QPB
steady-traveling-wave hypothesis λ_spatial = v · T_nutation (theory.md §4.7).

It mirrors the Tier 1 (`nutation.py`) / Tier 2 (`psi_g.py`) trait-emission template, NOT the
machinery shape of Tier 3a/3b. Public API:

```python
def compute(trajectory_df: pd.DataFrame, cadence_s: float, constants=None) -> pd.DataFrame
```

## Scope (D1)

**Reduced scope: traveling-wave validation traits only.** PR #9 empirically established (all 6
plate-001 tracks) that the §7.4 |κ|-envelope-peak recipe for `L_gz`/`L_c` does NOT transfer to
top-view tip-trail κ(s); those traits + the L_gz growth-zone mask are descoped to research
issue **#230**. PR #10 therefore ships ONLY the 3 traits the λ(s_a) ridge supports and operates
on the **full reconstructed trail (no mask)**.

- **Deliverable**: `lambda_spatial_median_px`, `traveling_wave_residual`, `lambda_spatial_variation`.
- **Blocked on #230 (omitted entirely, not reserved as NaN)**: `L_gz_estimate`, `L_c_estimate`,
  `B_balance_number`, `L_gz_steady_state_residual`, `L_gz_resolvable`, the L_gz mask.
  Rationale: #230 may redefine these for tip-trail geometry; committing names/units now would
  pre-commit a definition that may change, and all-NaN columns are ambiguous. Matches how
  nutation/psi_g emit only what they compute.

## Goals / Non-Goals

- **Goals**: emit the 3 λ-based traits + diagnostics per track; correct unit + cgau2-calibration
  reconciliation; COI reliability gating; self-contained composition with Tier 0/1; real-data
  validation on plate-001.
- **Non-Goals**: L_gz/L_c/B (→ #230); the growth-zone mask; pipeline DAG wiring (→ PR #14);
  parametric fits (→ PR #11); plotting (stub remains).

## Architecture

New flat module `sleap_roots/circumnutation/traveling_wave.py` (sibling of nutation.py,
psi_g.py). Public `compute`; private `_compute_one_track`, `_check_constants`,
`_check_cadence_s` (reuse the established helpers / patterns).

### Composition (recompute internally) — D-composition

`compute` is self-contained: it calls `kinematics.compute(trajectory_df, constants=resolved)`
(Tier 0) and `nutation.compute(trajectory_df, cadence_s, coordinate="lateral", constants=resolved)`
(Tier 1) once each, **joins both results to the per-track loop on the full `_IDENTITY_5_TUPLE`**
(`series, sample_uid, plate_id, plant_id, track_id` — NOT `track_id` alone; `track_id` is not
unique across plates/samples), and reads `v_total_median_px_per_frame`, `T_nutation_median`,
`is_nutating`. This keeps the `(trajectory_df, cadence_s, constants)` signature identical to its
siblings, guarantees operand consistency, and gets the `is_nutating` gate for free. The
resolved `constants` MUST flow into both recomputed tiers so a caller's override is honored
consistently. `coordinate="lateral"` is hardcoded (not exposed): the QPB residual is only
defined against the lateral nutation period (CC-7); exposing it would silently invalidate the
residual (mirrors psi_g's deliberate omission of `coordinate`).

> **Forward note to PR #14 (pipeline DAG):** Tier 3c RE-RUNS Tier 0 and Tier 1 internally. In a
> full pipeline run those tiers are also computed standalone, so the DAG MUST dedup — compute
> Tier 0/Tier 1 once and route their outputs into Tier 3c — rather than calling
> `traveling_wave.compute` naively. This redundancy is intentional for standalone correctness.

### Per-track chain

```
x, y (tip trajectory, sorted by frame, NaN-dropped)
  → midline.reconstruct(x, y, cadence_s)                      # MidlineResult
  → resample_curvature(curvature_px_inv, arc_length_px, velocity_sub_noise_mask)
  → compute_scaleogram(kappa_uniform, ds)                     # cgau2
  → extract_ridge(scaleogram)                                 # wavelengths_px, in_coi, ...
  → interior = ~in_coi                                        # COI gate
  → λ_cal[i] = calibrate(wavelengths_px[i])                   # true px (per-λ)
  → λ statistics over interior
```

If `midline`/`resample` is degenerate, or the ridge cannot be formed (<9 samples), or the
in-COI fraction exceeds `COI_FRACTION_MAX`, the spatial λ-traits are NaN for that track.

## Trait schema (D2, revised by D4)

8 `ROW_IDENTITY_COLUMNS` + 6 trait columns, in declared order:

| # | Column | dtype | units | definition |
|---|--------|-------|-------|------------|
| 1 | `lambda_spatial_median_px` | float64 | px | `median(λ_cal[interior])` |
| 2 | `lambda_spatial_variation` | float64 | — | `MAD(λ_cal[interior]) / median(λ_cal[interior])` (robust fractional spread; 0 = uniform) |
| 3 | `traveling_wave_residual` | float64 | — | `|λ_med − lambda_expected| / lambda_expected` |
| 4 | `lambda_expected_px` | float64 | px | `v · T_frames`, `T_frames = T_nutation_median / cadence_s` |
| 5 | `lambda_spatial_mad_px` | float64 | px | `median(|λ_cal[interior] − median|)` (numerator of #2) |
| 6 | `coi_valid_fraction` | float64 | — | `interior.sum() / interior.size` |

`MAD` = median absolute deviation about the median (the robust, orientation-invariant spread;
see D4). `λ_med` in #3 is `lambda_spatial_median_px` (#1).

### NaN-gating

- **Spatial-availability gate** → NaN on #1,#2,#3,#4,#5 when: `midline` degenerate, OR `resample`
  degenerate, OR ridge cannot form (insufficient samples), OR
  `(in_coi.sum() / in_coi.size) > COI_FRACTION_MAX` (= `1 − coi_valid_fraction > 0.5`).
  `coi_valid_fraction` (#6) is populated whenever a ridge exists (NaN only on full degeneracy),
  so it diagnoses why a row gated.
- **`is_nutating` gate** → flows through `lambda_expected_px`: a non-nutating track has
  `T_nutation_median = NaN` ⇒ #4 = NaN ⇒ #3 = NaN. The pure-spatial traits #1, #2 remain valid
  (λ exists whether or not temporal nutation was detected).

## Key reconciliations (the §7.4 handoff contract + the corrections)

### Unit reconciliation (compare in px)
λ_spatial is px; `v_total_median_px_per_frame` is px/frame; `T_nutation_median` is **seconds**
(no suffix; the "hr" in stale trait tables is post-conversion — do NOT ×3600). Convert T to
frames: `T_frames = T_nutation_median / cadence_s`, so `lambda_expected_px = v · T_frames`
(matches §6.4's 5.83 × 3333/300 ≈ 65 px). Pure-pixel pipeline (CC-3): no `[mm]`, no `px_per_mm`.

### cgau2 calibration (D3) — DEVIATION from §7.4 handoff
The handoff (note 2) claimed the cgau2 bias "largely cancels" in the residual ratio. **This is
empirically false** (investigation FINDING 1): `v·T_frames` is a true-px prediction with no
cgau2 bias, so the naive `|λ_raw − v·T|/(v·T)` is **mixed-domain** and wrong (it only looks
small — raw median 0.054 — because the +13% over-report accidentally nudges λ_raw toward v·T).
**Decision:** compute ONE calibrated λ array in true px and use it for ALL three λ-traits
(`lambda_spatial_median_px`, `traveling_wave_residual`, `lambda_spatial_variation`). Honest
residual: median 0.147, range [0.087, 0.177] on plate-001 → **QPB holds to ~9–18%**.

Calibration lookup: `circumnutation_spatial_cwt_calibration.json` gives `ratio = λ_reported/λ_true`
at sampled `(n, ds, λ_true)`. A consumer knows `λ_reported`, so interpolate `ratio` as a function
of `λ_reported` and divide: `λ_cal = λ_reported / ratio_interp(λ_reported)`, applied per-λ.

Observed `λ_reported` reaches ~142 px but the current table covers only [21, 91] px → clamped
extrapolation for 2 tracks. **PR #10 extends the table** (re-run `capture_spatial_coi_factor.py`
with `λ_true` up to ~150 px). Capture is deterministic ⇒ existing entries reproduce identically
⇒ PR #9's tolerance-based calibration test stays green; new entries only extend coverage.

### COI reliability gate
`extract_ridge` ships the raw, un-COI-masked ridge. PR #10 gates on the in-COI fraction using
the existing `COI_FRACTION_MAX = 0.5`. On real data the in-COI fraction is 0.02–0.14, so the
gate protects degenerate/short trails without binding good ones.

### `lambda_spatial_variation` (D4) — DEVIATION from §7.4 (rename + redefinition)
theory.md §7.4 named this `apex_basal_period_consistency` and described it as "uniform λ(s) along
trail; large variation = H1 violation". The investigation (FINDING 2/3) showed an apex-vs-basal
*difference* is unmeasurable here: the s_a→0 / s_a→max extremes are edge/scale-rail artifacts
(track-0 apex λ ≈ 432 px), and the apex orientation is ambiguous (resample's s_a=0 is frame-0 =
oldest tissue; §6.2 sign-polarity warning). Implement the **uniformity** intent as an
orientation-invariant, outlier-robust spread `MAD(λ_cal)/median(λ_cal)` (real data 0.13–0.37),
and **rename** to `lambda_spatial_variation` (plainer; honestly a spread, not a directional
comparison). CV (std/mean) and IQR/median were rejected as unstable/outlier-driven on the spiky
per-position ridge.

## Constants (D5)
No new constants. Reuse `COI_FRACTION_MAX` (already defined in `_constants.py` + `ConstantsT`,
previously unused — PR #10 is its first consumer). Continuous residual + MAD/median need no
thresholds. **`_CONSTANTS_VERSION` stays 6** (bump only for a real new constant).

## Determinism (D6)
The composed stack = SG smoothing (midline) + pywt cgau2 CWT (spatial) + np.interp (calibration)
+ argmax ridge + medians, AND the recomputed Tier 1 (which already composes scipy fft/signal/
ndimage/stats). Tier 3c therefore inherits Tier 1's looseness; target **atol = 1e-6** (nutation's
value), re-measured on a real-data canary during implementation — measure, don't cargo-cult.

## Input-validation boundary
Reuse `_validate_trajectory_df` (8 identity cols + `frame`, `tip_x`, `tip_y`; non-empty;
DataFrame) and `_check_cadence_s` (positive finite). `constants` validated as None or ConstantsT.

## Testing strategy (TDD)
- Synthetic: a known-λ trajectory → λ_spatial recovers it within calibration tolerance;
  uniform-λ synthetic → small `lambda_spatial_variation`; non-nutating → residual NaN, λ valid.
- Gating: degenerate/short track → spatial traits NaN, `coi_valid_fraction` diagnostic;
  forced low-COI → gated.
- Schema: exact 8+6 columns in order, dtypes, 5-tuple groupby, per-plant merge.
- Determinism: two-run canary at the measured atol.
- **Real plate-001 (all 6 tracks)**: `traveling_wave_residual` finite and ~0.10–0.18 (QPB),
  `lambda_spatial_variation` ~0.13–0.37, all gates pass — the scientific cross-check.
- Calibration-table extension: regenerated JSON covers observed λ; PR #9 calibration test green.

## Risks / Trade-offs
- Recompute redundancy (Tier 0/1 twice) → mitigated by the PR #14 DAG; documented forward-note.
- Extending the PR #9 calibration artifact touches a committed deliverable → mitigated by
  determinism (existing entries unchanged) + re-running PR #9's test.
- λ-calibration extrapolation beyond the (extended) table → bounded; documented limitation.

## Open Questions
- Exact `λ_true` grid for the table extension (decide during impl from the observed λ range).
- Final atol value (measure during impl; expect 1e-6).

## Decisions log (D1–D7)
- **D1** Reduced scope (λ-traits only); blocked L_gz traits omitted, not reserved.
- **Composition** Recompute Tier 0/1 internally (self-contained); forward-note for PR #14 dedup.
- **Module** `traveling_wave.py`.
- **D2** 3 core + 3 diagnostics (revised after D4: dropped apex/basal px, added `lambda_spatial_mad_px`).
- **D3** Calibrate λ to true px (single λ for median + residual); "bias cancels" handoff claim
  overturned; extend calibration table to ~150 px.
- **D4** `lambda_spatial_variation` = MAD/median (orientation-invariant); rename from
  `apex_basal_period_consistency`.
- **D5** No new constants; `_CONSTANTS_VERSION` stays 6.
- **D6** atol target 1e-6, re-measured.
- **D7** QPB holds to ~9–18% on plate-001 (real scientific result; see CR-9 re: extrapolation).

## Critical-review reconciliation (round 1, 2026-06-10)

Five adversarial subagents reviewed this design; numbers re-verified on plate-001. Resolutions
below are authoritative for the implementer and become tasks in the proposal.

### CR-1 [BLOCKING] Join recomputed Tier 0/1 on the full 5-tuple, not `track_id`
`track_id` is not unique across plates/samples; joining Tier 0/1 operands by `track_id` alone
would pull the wrong plate's `v`/`T`/`is_nutating` (silent-wrong) or crash on duplicate labels —
undetectable by the single-plate plate-001 fixture. **Resolution (revised per round-2, see
CR2-1):** do NOT use `set_index(...).at[key]` (dtype-fragile — see CR2-1). Instead **merge** the
operand columns onto the per-track `trait_df` on `list(_IDENTITY_5_TUPLE)`, applying the same
int64 coercion-with-raise guard CR-5 mandates for the template merge, to BOTH sides. Concretely:
`trait_df = trait_df.merge(tier0[[*_IDENTITY_5_TUPLE, "v_total_median_px_per_frame"]], on=..., how="left").merge(tier1[[*_IDENTITY_5_TUPLE, "T_nutation_median", "is_nutating"]], on=..., how="left")`
after coercing `track_id`/`plant_id` to int64 on all three frames. **Add a multi-plate test**
(≥2 plates, overlapping `track_id`s, **with float64 `track_id`** to exercise the coercion) — the
regression the single-plate fixture alone can't catch.

### CR-2 [BLOCKING] `_compute_one_track` never raises — guard + try/except → all-NaN row
`midline.reconstruct` raises `ValueError` on non-finite tips; `compute_scaleogram` raises
`ValueError` on <`MIN_SAMPLES_REQUIRED` (9) samples, non-finite κ, or non-positive `ds`. The
linear chain as drawn would crash all of `compute()` on one bad track. **Resolution** (mirrors
`nutation._compute_one_track` lines 398–428 verbatim in shape):
1. Sort the track by `frame` and drop non-finite `tip_x`/`tip_y` rows **before** `reconstruct`
   (so it never sees NaN and never raises on finiteness — mirrors `psi_g.py:239`).
2. `mr = reconstruct(...)`; if `mr.is_degenerate` → return `_all_nan_spatial_traits()`.
3. `rs = resample_curvature(...)`; if `rs.is_degenerate` → return `_all_nan_spatial_traits()`.
4. Guard-before-call: only call `compute_scaleogram` when `not rs.is_degenerate`.
5. **AND** wrap `compute_scaleogram`/`extract_ridge` in `try/except ValueError → _all_nan_spatial_traits()`
   (defense-in-depth, exact sibling pattern).
Add a named helper `_all_nan_spatial_traits()` returning all 6 columns (with the CR-3 rule for
#6), so every gate path returns a complete dict and the per-plant left-merge always yields one
row per 5-tuple. **Invariant (state explicitly):** `_compute_one_track` always returns a full
6-key trait dict and never raises.

### CR-3 [BLOCKING] `coi_valid_fraction` (#6) value pinned across all gate paths
| Path | ridge formed? | #1–#5 | #6 `coi_valid_fraction` |
|---|---|---|---|
| midline degenerate | no | NaN | **NaN** |
| resample degenerate | no | NaN | **NaN** |
| scaleogram/ridge raised (caught) | no | NaN | **NaN** |
| low-COI gate fired (`coi_valid_fraction < 1−COI_FRACTION_MAX`) | yes | NaN | **the actual fraction** (finite) |
| healthy | yes | finite | the actual fraction |
So `#6` is finite **iff a ridge formed**; it disambiguates only the low-COI-vs-healthy case (the
three no-ridge paths collapse to one all-NaN row). The design prose claim that #6 "diagnoses why
a row gated" is **scoped to that distinction only** (corrected). `#6` is the one
not-spatial-gated diagnostic (mirrors nutation's always-populated precursors).

### CR-4 [IMPORTANT] Residual division guard (`v` finite, `lambda_expected > 0`)
A stationary/degenerate track can give `v ≈ 0` or non-finite ⇒ `lambda_expected_px = v·T_frames`
≤ 0 or non-finite ⇒ `traveling_wave_residual = |λ−0|/0 = inf/nan` with a RuntimeWarning.
**Resolution:** gate #3 and #4 to NaN when `v` is non-finite **or** `lambda_expected_px ≤ 0`
(explicit guard before the division; mirrors nutation's `T > 0` guards). So #4 = NaN when
`is_nutating==False` (T NaN) **or** `v` non-finite/zero.

### CR-5 [IMPORTANT] Emission tail specified verbatim from siblings
Add an "Emission" step: `groupby(list(_IDENTITY_5_TUPLE), dropna=False, sort=False)`;
`template = _build_per_plant_template_from_df(trajectory_df)`; pre-merge dtype-coercion guard on
`track_id`/`plant_id` that **raises** on cast failure (prevents silent all-NaN merge);
`template.merge(trait_df, on=list(_IDENTITY_5_TUPLE), how="left")`; final column order =
8 `ROW_IDENTITY_COLUMNS` + 6 trait columns. **All 6 traits are float64** — no bool/int special
case (unlike nutation's `is_nutating`/psi_g's `handedness`); a single
`for col in _TRAIT_COLUMNS: result[col] = result[col].astype(np.float64)` loop. NaN from an
unmatched left-merge row is already the correct "no trait" sentinel for float columns.

### CR-6 [IMPORTANT] Validation helpers
Reuse `_validate_trajectory_df` (with the `isinstance(... pd.DataFrame)` pre-check the siblings
duplicate) and import `temporal_cwt._validate_cadence_s` (psi_g's choice — avoid a third private
copy). `_check_constants`: validate `COI_FRACTION_MAX` locally (float in (0, 1]); defer the
Tier-1/Tier-0 fields to the inner `nutation.compute`/`kinematics.compute` (which re-validate
their own). `COI_FRACTION_MAX` **verified present** in `_constants.py` (line 135) + `ConstantsT`
(line 539) — `_CONSTANTS_VERSION` stays 6 confirmed.

### CR-7 [IMPORTANT] Calibration consumer: one monotone `ratio(λ_true)` curve
The JSON is a `(n, ds, λ_true)` grid with **3 `λ_reported` per `λ_true`** (n∈{200,400,600}),
non-monotone in `λ_reported` → interpolating `ratio` vs `λ_reported` rides a sawtooth (~5%
jitter) and is ill-posed for `np.interp`. **Resolution:** collapse to a single monotone curve —
for the consumer, use the `n` slice nearest the real ridge length (real `n_int` 279–414 ⇒ the
**n=400** rows), build `ratio(λ_true)` from that slice, and invert
`λ_true = λ_reported / ratio` via interpolation on the strictly-increasing `(λ_reported_n400 → ratio)`
mapping. (Robustness checked: averaging across n shifts the residual median only 0.147→0.142.)
Document the chosen `n` and the monotone construction. **Do NOT assert** "calibrating λ ÷r ≡
inflating v·T ×r" as strictly equal — it holds only for a constant scalar r; with a per-λ `r` it
is approximate. State the operative rule plainly: compute one calibrated λ array in true px, use
it for all three λ-traits.

### CR-8 [IMPORTANT] Calibration regeneration = append-only merge (zero blast radius)
**No test reads the JSON** (verified — only a comment in `test_circumnutation_spatial_cwt.py:689`;
the gate is a hardcoded `[1.00, 1.25]` band), so regeneration cannot break a PR #9 test. The real
risk is **environment drift**: `capture_spatial_coi_factor.py` rewrites the whole file incl. the
provenance block (`capture_date_iso`, numpy/pywt/BLAS versions), and `pywt.cwt`/`scale2frequency`
are library-version-sensitive (PR #9's own canary flags cgau2 as cross-OS-unproven) — a full
re-run on a different env could silently shift the existing 18 rows. **Resolution:** add an
**append-only / merge mode** to `capture_spatial_coi_factor.py` that adds only the new `λ_true`
rows (≥ ~150 px coverage) and **freezes the existing provenance + 18 rows byte-for-byte**. Add a
regression test asserting the pre-existing `(n, λ_true)` rows are unchanged (nothing currently
protects them). This mechanically guarantees the "existing entries unchanged" claim instead of
resting it on env-determinism.

### CR-9 [IMPORTANT] D7 honesty: extend table BEFORE locking the headline
Both endpoints of the advertised residual range [0.087, 0.177] are the two tracks (0,1) whose
`λ_reported` (120–142 px) currently **clamp-extrapolate** beyond the [21,91] table. **Resolution:**
extend the calibration table (CR-8) **first**, then re-measure; state D7 as "≈0.10–0.18 (re-confirmed
post-extension)"; the real-data test asserts the residual is **finite and < 0.30** (a generous
band), NOT a pinned [0.087, 0.177] that encodes an extrapolation artifact.

### CR-10 [IMPORTANT] Logging (CC-9) + foundation test migrations
Add `logger = logging.getLogger(__name__)` and a single top-of-`compute`
`logger.debug("traveling_wave.compute(n_tracks=%d, cadence_s=%.6f)", …)` matching the exact
sibling pattern (siblings emit only DEBUG — no INFO/per-plate scheme; do not invent one). Module
docstring follows the sibling shape (callable summary + per-trait bullets + Anchors to this
design, the investigation report, theory.md §4.7/§6.4/§7.4). **Foundation test migrations
(tasks):** add `"traveling_wave"` to `test_module_logger_is_namespaced`'s module list
(`tests/test_circumnutation_foundation.py:786–801`); add the `traveling_wave.compute` callability
entry + any `IMPLEMENTATIONS_WITH_CONSTANTS_KWARG`-style table entry (verify exact locations in
impl). `traveling_wave` was never a stub (only `parametric`/`plotting`/`pipeline` remain), so it
is a pure ADDITION — no `STUB_MODULES` removal (mirrors PR #6's `nutation` addition note).

### CR-11 [IMPORTANT] Units subsection (all in-vocabulary; no `_constants` change)
Per-column units: `lambda_spatial_median_px`=px, `lambda_spatial_variation`=—,
`traveling_wave_residual`=—, `lambda_expected_px`=px, `lambda_spatial_mad_px`=px,
`coi_valid_fraction`=—. **Verified** `px` and `—` are already in `PIPELINE_UNIT_VOCABULARY`
(`_constants.py:431`) → no vocabulary addition, no `_CONSTANTS_VERSION` change. The `px⁻¹`
concern PR #9 deferred does **not** fire (no trait column has px⁻¹ units; `spatial_freqs_px_inv`
is an internal Result field, not a trait/sidecar column). Declare the 6 column→unit pairs in the
tier's trait-units mapping so the sidecar writer validates.

### CR-12 [BLOCKING] Enumerate exact theory.md edits + Appendix B(6)
Per deviation discipline (B(3)/B(5) convention: **preserve the original wording for provenance**),
the proposal's tasks MUST include these precise `theory.md` edits:
1. §7.4 `lambda_spatial_median` row (line 504): rename → `lambda_spatial_median_px`, unit `mm`→`px`,
   strike "basal of $L_{gz}$" (no mask, D1), note calibration applied.
2. §7.4 `traveling_wave_residual` row (line 505): reflect calibrated λ + `T_frames = T_nutation/cadence_s`
   + honest ~0.10–0.18 result (§4.7 anchor stays correct).
3. §7.4 `apex_basal_period_consistency` row (line 507): rename → `lambda_spatial_variation`,
   redefine as `MAD(λ_cal)/median(λ_cal)` robust spread; weaken the directional-H1 anchor.
4. §7.4 "Handoff to PR #10" note 2 (line 512): correct the now-false "bias largely cancels" claim
   in place (point to B(6)).
5. §7.4 "Handoff to PR #10" note 4 (line 514): correct the apex-vs-basal / "pin apex = s_a→0"
   instruction in place (overturned; point to the uniformity-spread definition).
6. §7.4 PR #9 scope-note (lines 485–495): update the dead name `apex_basal_period_consistency`
   → `lambda_spatial_variation`.
7. **NEW Appendix B(6)**: preserve verbatim the ORIGINAL handoff-note-2 "bias cancels" wording and
   the ORIGINAL `apex_basal_period_consistency` name+definition; then state the corrections (calibrate
   λ in true px for all 3 traits; rename→`lambda_spatial_variation`=MAD/median; honest residual
   0.10–0.18). Cite this design + the investigation report (as B(4)/B(5) cite theirs).
**roadmap.md**: rename both `apex_basal_period_consistency` occurrences in the PR #10 row (line 146)
→ `lambda_spatial_variation`; add `traveling_wave` to the module enumeration. **changelog.md**: the
PR #10 entry announces the rename (leave the historical PR #9 entry as-is). **PR #10 issue draft**
uses the new name.

### CR-13 [IMPORTANT] OpenSpec deltas
- **MODIFY "Package layout"**: add `traveling_wave` as an addition-shape new implementation module
  (impl count 8→9; stub count unchanged), with a "Scope note on PR #10 addition" mirroring PR #6's
  `nutation` note, plus a callability scenario (`traveling_wave.compute` importable/callable).
- **ADD "Tier 3c traveling-wave trait emission API"**: lock the
  `compute(trajectory_df, cadence_s, constants=None) -> pd.DataFrame` signature, the 8+6 column
  schema in order, dtypes (all float64 traits), the two NaN-gates, the COI gate
  (`coi_valid_fraction < 1−COI_FRACTION_MAX`), and the calibration application. Render each behavior
  as a discrete `#### Scenario:`.

### Verified reviewer claims (no change needed)
Unit reconciliation sound (T is seconds, no ×3600); "raw residual is mixed-domain, calibrated is
honest" correct; MAD computed about the median; all 6 column suffixes correct; CC-3/CC-1 honored;
COI gate **direction** correct.

## Critical-review reconciliation (round 2, 2026-06-10)

Round 2 verified the CR-1..CR-13 fixes against code + re-ran the calibration on real data. Most
are SOUND (CR-3/4/5/6/7/8/9/11/12/13 confirmed; n=400 slice `λ_reported` strictly increasing →
np.interp well-posed; n=400 residual median 0.142 vs 0.147 — immaterial; append-only regeneration
proven bit-identical on shared rows; theory.md line numbers 504/505/507/512/514/485–495 verified;
last Appendix B entry is B(5) → new one is B(6)). Remaining items:

### CR2-1 [BLOCKING] The CR-1 `.at[key]` lookup is dtype-fragile → use merge
`_validate_trajectory_df` does NOT constrain `track_id`/`plant_id` dtype, but the recomputed Tier
0/1 frames coerce them to int64 (`_io.py:164`). A raw float64 `track_id` makes the groupby key
`(..., 0.0)` miss the int64 index `0` → `KeyError`, uncaught (CR-2 wraps only the CWT calls) →
`compute()` crashes. **Resolution:** replaced `.at[key]` with a merge-on-`_IDENTITY_5_TUPLE` +
int64 coercion (fixed in CR-1 above). The multi-plate test MUST use a **float64 `track_id`**
fixture to exercise this path.

### CR2-2 [IMPORTANT/HIGH] CR-10 foundation-test edits — exact locations (else CI breaks)
CR-10 named only 1 of 3 required edits and mis-cited its line. Pin all three in
`tests/test_circumnutation_foundation.py`:
1. `test_module_logger_is_namespaced` parametrize list — append `"traveling_wave"` **after line
   801** (the `spatial_cwt` entry; the "786–801" range in CR-10 was wrong — 786 is `nutation`).
2. `IMPLEMENTATIONS_WITH_CONSTANTS_KWARG` list (**line ~882–891**) — append
   `("traveling_wave", "compute")` after the `spatial_cwt` entry (line ~890).
3. `test_implementation_accepts_constants_kwarg` (the if/elif chain ~921–1037) — add a **dedicated
   `elif module_name == "traveling_wave":` branch** (mirror the `nutation` branch ~936–961: a
   ≥64-frame single-track df, `fn(df, 300.0, constants=ConstantsT())`, assert DataFrame). The
   generic `else` branch calls `fn(df, constants=...)` with **no `cadence_s`** → `TypeError`. This
   is the most likely missed CI break. (`STUB_MODULES`/`STUBS_WITH_CONSTANTS_KWARG`: no edit —
   `traveling_wave` was never a stub; confirmed.) Verify exact line numbers at impl time.

### CR2-3 [IMPORTANT] Synthetic known-λ test: small-amplitude construction
The synthetic generator's spatial λ = `growth_rate_px_per_frame · (T_nutation_s / cadence_s)` =
exactly `v·T_frames` (synthetic.py:475–500) — so a known-λ recovery test IS constructible. BUT at
the default `amplitude_px=10` the trail's arc-length-per-period exceeds the pure-drift `v·T` by the
lateral excursion → κ(s)'s spatial period is 5–15% longer than `growth_rate·T_frames`. **The
synthetic λ-recovery test MUST use small amplitude relative to drift** (`amplitude_px ≪
growth_rate·T_frames`) so arc-length ≈ drift and the a-priori λ is correct — OR assert against the
numerically-integrated arc-length-per-period. Note: synthetic λ and `lambda_expected` are coupled
by construction, so a *large-residual* (QPB-violating) case is NOT synthesizable — the real-data
plate-001 test covers the non-trivial-residual case. State both in the Testing strategy.

### CR2-4 [MINOR] Table extension covers n=400 only
The consumer reads only the n=400 slice (CR-7), so the append-only extension generates the new
`λ_true` rows (≥ ~140–150 px) **for n=400 only** (sufficient and minimal). A single
`λ_true=140` row already pushes `λ_reported` to ~157 px, covering track-0's 142.5 — confirmed.

### CR2-5 [MINOR] CR-2 wording: non-finite-dropped, not just NaN
The pre-`reconstruct` filter drops **non-finite** tips (`np.isfinite(tip_x) & np.isfinite(tip_y)`,
per psi_g.py:239) — ±inf as well as NaN (`midline._validate_xy` rejects ±inf too). The per-track
diagram's "NaN-dropped" → "non-finite-dropped (np.isfinite mask)".

### CR2-6 [MINOR] roadmap.md line 146 also strike the descoped L_gz claims
Beyond the rename (CR-12), roadmap line 146 still lists `B_balance_number`,
`L_gz_steady_state_residual`, `L_gz_resolvable` as PR #10 deliverables and says "Applies the L_gz
growth-zone mask" — contradicting D1's descope. The roadmap edit MUST also remove those
descoped traits + the mask claim so the row matches D1 + §7.4.

### CR2-7 [MINOR] Note the smooth-fit calibration alternative + rationale
A smooth `ratio(λ)` least-squares fit would remove the per-knot sawtooth and extrapolate
gracefully (making table extension optional). The chosen piecewise-linear-on-extended-table is
preferred for **auditability** (every prediction traces to a measured grid point) + **reuse of the
committed deterministic PR #9 generator**. State this so reviewers don't re-raise it.

### CR2-8 [MINOR] Determinism canary covers the full chain
The two-run determinism canary runs the **full composed chain including the np.interp calibration
on the extended table** (not just the raw ridge), guarding against a non-deterministic
table-extension. (`np.interp` itself is bit-reproducible cross-OS; the binding risk is the
inherited Tier-1 scipy stack + cgau2 CWT — hence measure, don't cargo-cult.)

### CR2-9 [MINOR] CR-13 import scenario
Add `sleap_roots.circumnutation.traveling_wave` to the spec's "modules import cleanly" scenario
import list (`openspec/specs/circumnutation/spec.md:34`) — every impl module currently appears
there.

## Rounds 3–4 supersession (read this — it overrides earlier calibration entries)

Two later critical-review rounds (openspec-review + a fresh-eyes pass, then a focused science pass)
changed three decisions recorded above. Where this section conflicts with CR-7 / CR2-4 / CR-9 / the
round-2 summary, **this section wins** (the authoritative current decisions also live in the
change-folder `design.md` Round-3 / Round-4 sections and in the spec/tasks):

- **Calibration is now n-AVERAGED, not the n=400 slice.** SUPERSEDES CR-7 ("use the n=400 rows") and
  CR2-4 ("extension for n=400 only"). The cgau2 ratio scatters ~7% across `n ∈ {200,400,600}`
  non-monotonically; fixing n=400 injected a ~±5% systematic ≈ the residual signal. The consumer uses
  a single n-averaged `ratio(λ)` curve (in-package literal `_CGAU2_LAMBDA_CALIBRATION`, NOT
  `_CGAU2_LAMBDA_CALIBRATION_N400`); the append-only extension measures the new `λ_true` knots for ALL
  three n; the ~±5% systematic is documented. Verified: averaged axis strictly increasing; plate-001
  median 0.142, range [0.094, 0.182] (unchanged).
- **D7 is provisional.** 4 of 6 real tracks currently clamp-extrapolate (not 2); the ~9–18% headline is
  re-confirmed only after the post-extension re-measurement. SUPERSEDES the settled-sounding D7 / CR-9
  `[0.087, 0.177]` phrasing above (that range is retired — do NOT pin it).
- **`lambda_spatial_variation` has NO argmax-quantization floor.** A noise-free uniform-λ synthetic
  reads ≈0; real-data 0.13–0.37 is genuine ridge-localization scatter (grows with noise), interpreted
  relative to the noise level — not a quantization floor, not pure biology.
- **Determinism asserts `wavelengths_px[interior]` exact-equality** (the integer scale index is not a
  public `SpatialRidgeResult` field); the 1e-6 atol covers only the v·T-derived columns.
- **Packaging:** the calibration data ships as the in-package literal (wheel-safe); the module never
  reads `tests/data` at runtime.
