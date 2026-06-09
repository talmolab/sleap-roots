# Design: Tier 3b spatial CWT machinery (PR #9, `add-circumnutation-tier3b-spatial-cwt`)

**Date**: 2026-06-09
**Status**: approved (revised scope after empirical investigation)
**Author**: Elizabeth Berrigan (eberrigan) + Claude

## Context

PR #9 of the multi-PR circumnutation program (epic #197; roadmap row #9). It
graduates the `spatial_cwt` stub into an implementation module — a **machinery**
PR mirroring PR #5 `temporal_cwt` and PR #8 `midline` (input-validating, frozen
`attrs` Results, a determinism contract, **no trait emission, no DataFrame, no
mask**). Its input is PR #8's `MidlineResult` (`curvature_px_inv` on the native
NON-uniform `arc_length_px` grid); PR #9 owns the uniform-arc-length resample
(the locked PR #8→#9 handoff), runs a cgau2 spatial CWT, and extracts the
dominant spatial wavelength ridge λ(s_a).

## Goals / Non-Goals

- **Goals:** resample κ(s) onto a uniform grid; cgau2 spatial scaleogram + COI;
  spatial ridge → λ(s_a) (the steady-traveling-wave wavelength). Deterministic,
  cross-OS reproducible, pure-pixel.
- **Non-Goals (deferred):** trait columns / `trajectory_df` / groupby (PR #10);
  the `L_gz` growth-zone mask (CC-1, PR #10); **L_gz / L_c growth-zone-structure
  detection** (descoped — see "Empirical investigation" + follow-up issue).

## Empirical investigation that reshaped scope

Before locking the spec we ran `_scratch/2026-06-09-lgz-localizer-comparison/`
(documented `research:investigate`) to test how `detect_growth_zone` should
localize `L_gz` (peak of the |κ(s)| envelope, theory §7.4) — Hilbert vs cgau2
CWT-power vs smoothed-|κ| — on the 6 Nipponbare plate-001 **proofread** tracks +
an analytic oracle.

**Finding (decisive):** the §7.4 premise — "L_gz = peak of |κ(s)| envelope, L_c =
basal exponential decay" — **does not transfer to top-view tip-trail κ(s)**.

- The oracle confirms the localizers work *when an apical envelope peak exists*
  (Hilbert/smoothed ~1-bin accurate on clean/sparse; Hilbert edge-fails under
  noise; CWT-power noise-robust but +23–46 px biased).
- On real data the structure is absent: trail spans ≈ 3050–3527 px, the growth
  zone (~29–65 px) is ~1 % of the trail, and the global |κ| envelope argmax lands
  **basal (48–100 % of span)** on every track, every method. Apical/basal Hilbert
  envelope ratio = 0.32/1.25/0.56/0.67/0.77/0.91 → for 5/6 tracks apical curvature
  amplitude is *lower* than basal — the opposite of the predicted localized peak.
- Raw κ(s_a) is a **roughly steady-amplitude oscillation over the whole trail** —
  the QPB steady-traveling-wave signature (§4.7), directly visible in the plots,
  independent of envelope method or edge artifacts.

This is exactly what theory **§6.5** warned: "the recent trail cannot be inverted
to recover ε̇(s_a)… trail-as-midline holds only past the growth zone." The §7.4
definitions come from Rivière/Averrhoa **whole-organ side-view** κ(s,t) over a
short growth zone; they do not transfer to top-view multi-period tip-trail data.
Synthetic-only would have confirmed the wrong premise (PR #7 helix / PR #8 mask
lesson).

**Consequence:** the data SUPPORTS the spatial scaleogram + λ_spatial (a steady
wave has a clean dominant wavelength — what cgau2 CWT extracts), and PR #10's
central falsifiable trait `traveling_wave_residual = |λ_spatial − v·T|/(v·T)`
remains deliverable. It is specifically the **Meroz growth-zone-structure traits
(L_gz, L_c, B = L_gz/L_c)** that don't transfer → descoped to a follow-up.

## Decisions (brainstorm D1–D10 + scope pivot)

- **D1 — API:** separate public `resample_curvature` (entry, raw arrays);
  `compute_scaleogram(kappa, ds, constants=None)` stays the locked pure-CWT stub
  signature (drop the stub file's `wavelet=`/`scale_range=` kwargs; derive from
  constants, matching temporal_cwt). Entry-takes-arrays / downstream-takes-Result
  symmetry with temporal_cwt.
- **D2 — mask gaps:** drop `velocity_sub_noise_mask` frames (κ blows up at
  sub-noise |v|), then `np.interp` survivors onto the uniform grid; guards →
  graceful all-NaN. (Empirically validated; no new dep; tight determinism.)
- **D3 — ds:** median of positive surviving Δ(arc_length); scale range derives
  from it via spatial-named CWT constants.
- **D4 — 3 funcs / 3 Results** (revised after pivot): `resample_curvature →
  ResampleResult`, `compute_scaleogram → SpatialScaleogramResult`, `extract_ridge
  → SpatialRidgeResult`. (Was `detect_growth_zone/GrowthZoneResult` — descoped.)
- **D6 — constants + version bump 5→6:** add `SPATIAL_COI_EFOLDING_FACTOR`
  (cgau2, empirically measured via step-response), `CWT_WAVELENGTH_MIN_NYQUIST_FACTOR`,
  `CWT_WAVELENGTH_MAX_SIGNAL_FRACTION`; reuse `CWT_SCALE_COUNT_DEFAULT`,
  `WAVELET_DEFAULT_SPATIAL="cgau2"`.
- **D9 — apex-origin axis:** emit `s_a = arc_max − s` (apex at 0) on the position
  axis so PR #10 can do apex-vs-basal λ-consistency. λ itself is orientation-
  invariant. Pin the apex end with the oracle.
- **D10 — units:** pure-pixel (px, px⁻¹) Result fields; defer the `px⁻¹`
  `PIPELINE_UNIT_VOCABULARY` token to PR #10 (PR #9 emits no sidecar/CSV).
- **Descoped by the investigation:** D5 (Hilbert envelope), D7 (curve_fit L_c),
  D8 (resolvability v-window), `detect_growth_zone`, `GrowthZoneResult`,
  `L_gz_resolvable`. `LGZ_NMIN_RESOLVABLE` stays in `_constants` for the future PR
  but is no longer consumed by PR #9.

## Public API

```python
resample_curvature(curvature_px_inv, arc_length_px, velocity_sub_noise_mask=None,
                   constants=None) -> ResampleResult        # entry, raw arrays
compute_scaleogram(kappa, ds, constants=None) -> SpatialScaleogramResult  # locked stub
extract_ridge(scaleogram_result, constants=None) -> SpatialRidgeResult    # λ(s_a) ridge
```

Direct spatial twin of `temporal_cwt` (`compute_scaleogram` + `extract_ridge`),
plus the `resample_curvature` entry helper for the non-uniform→uniform handoff.

## Result classes (all `@attrs.define(frozen=True, slots=False, kw_only=True, eq=False)`)

- **`ResampleResult`**: `kappa_uniform` (px⁻¹), `s_a_uniform_px` (apex-origin),
  `ds` (px), `n_unmasked` (int), `arc_span_px` (float), `is_degenerate` (bool).
- **`SpatialScaleogramResult`**: `scaleogram` (complex128 `(n_scales, n_samples)`),
  `scales` (float64), `wavelengths_px`, `spatial_freqs_px_inv`, `coi_mask` (bool),
  `ds`, `wavelet` (str).
- **`SpatialRidgeResult`**: `position_indices` (int64), `wavelengths_px` (λ at
  ridge per position), `amplitudes`, `powers`, `in_coi` (bool). Mirrors temporal
  `RidgeResult` (frame→position, period_s→wavelength_px).

## Algorithms

- **resample_curvature:** drop masked + non-finite frames; `s_a = arc_max − s`
  over survivors; `ds = median(positive Δs_a)`; uniform grid `[0, arc_span]` step
  `ds`; `kappa_uniform = np.interp(grid, s_a_sorted, kappa_sorted)`. Guards (min
  unmasked count, min arc-span) → graceful all-NaN `is_degenerate=True`.
- **compute_scaleogram:** log-spaced scales via `pywt.scale2frequency("cgau2", …)`
  (wavelet-agnostic), λ_min = `CWT_WAVELENGTH_MIN_NYQUIST_FACTOR·ds`, λ_max =
  `CWT_WAVELENGTH_MAX_SIGNAL_FRACTION·n·ds`, `CWT_SCALE_COUNT_DEFAULT` scales;
  `pywt.cwt(kappa, scales, "cgau2")` → complex128; COI via
  `SPATIAL_COI_EFOLDING_FACTOR` (spatial sibling of `_make_coi_mask`/
  `_coi_boundary_samples`).
- **extract_ridge:** per-position argmax over scales → λ at ridge; amplitudes =
  |W| at ridge; powers = amplitudes²; `in_coi` from the scaleogram COI mask.
  Deterministic (no random tie-break).

## Constants & versioning

`_CONSTANTS_VERSION` 5 → 6 (three real new CWT constants). All three added to
`ConstantsT` + `_default_constants_snapshot()`. `SPATIAL_COI_EFOLDING_FACTOR`
default is the empirically-measured cgau2 e-folding factor (capture script
`scripts/circumnutation/capture_spatial_coi_factor.py`, mirroring PR #5's
cmor calibration; the cmor `COI_EFOLDING_FACTOR=√1.5` docstring already defers
the cgau2 factor to PR #9).

## Determinism (CC-6) & analytic oracle

No `curve_fit` (descoped) → the whole stack (`np.interp` + `pywt.cwt(cgau2)` +
argmax ridge) holds at **`atol=1e-9, rtol=0`** cross-OS, like PR #5. Canary
`scripts/circumnutation/capture_spatial_cwt_canary.py`. Oracle: a planted **pure
sinusoid of known λ** on a uniform grid → assert `extract_ridge` recovers λ at
interior (COI-dodging) positions; scaleogram cells canaried at interior indices.

## Degenerate / raise policy (PR #8 split)

Field-named validation runs first, unconditionally → raises (`TypeError`/
`ValueError`) on bad arrays, non-finite, length mismatch, bad `ds`, bad
constants. Degenerate gate runs only on valid input → graceful all-NaN Result,
never raises, never `RuntimeWarning` (`errstate` + `~isfinite` sweep).
Module-qualified imports (`import pywt`, `import scipy.signal`).

## Foundation-test migration (atomic with first non-raising commit)

Remove `("spatial_cwt","compute_scaleogram",9)` from `STUB_MODULES` +
`STUBS_WITH_CONSTANTS_KWARG`; add to `IMPLEMENTATIONS_WITH_CONSTANTS_KWARG`
(array-typed branch like midline) + the explicit `test_module_logger_is_namespaced`
list (PR #9 comment block); add `elif module_name == "spatial_cwt":`; bump the
`_CONSTANTS_VERSION` assertion 5→6; update the STUB_MODULES block-comment.
Counts: impl 7→8, stub 4→3.

## TDD outline (one RED→GREEN per scope unit)

constants + version → ResampleResult/resample (mask-drop, ds, degenerate) →
SpatialScaleogramResult/compute_scaleogram (scale axis, cgau2 CWT, COI) →
SpatialRidgeResult/extract_ridge (ridge λ, COI status, determinism) → analytic
sinusoid oracle + determinism canary → real plate-001 sanity (λ_spatial plausible
+ ~uniform along trail, per the investigation) → foundation migration (atomic).

## Recorded deviations (deviation discipline)

- **theory.md note (§7.4 / §6.3):** the L_gz/L_c growth-zone-structure traits are
  NOT measurable from top-view tip-trail κ(s) as written (empirical, this PR);
  the steady-traveling-wave λ_spatial is the measurable spatial quantity. Original
  §7.4 text preserved in Appendix B; changelog "Why" note.
- **Pure-pixel (CC-3):** λ/wavelengths in px, curvature/freq in px⁻¹ (theory shows
  mm) — same deliberate deviation as PR #8 px/frame.

## Follow-ups (vault-drafted GitHub issues — post per lazy-issue workflow)

1. **PR #9 tracking issue** (`add-circumnutation-tier3b-spatial-cwt`; parent #197;
   labels enhancement, circumnutation, multi-pr).
2. **L_gz/L_c tip-trail transfer research issue** — redefine or relocate the
   growth-zone-structure traits for top-view tip-trail κ(s); references this
   investigation. Blocks the L_gz-dependent parts of PR #10.

## Open questions

- The empirical `SPATIAL_COI_EFOLDING_FACTOR` value (measured during GREEN).
- PR #10 scope adjustment (L_gz mask + L_gz traits blocked; traveling_wave_residual
  + apex_basal_period_consistency remain) — tracked, not changed by PR #9.
- Multi-plate validation of λ_spatial (plate-001 only here; cf. #202).
```
