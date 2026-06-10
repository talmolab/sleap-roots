## Context

PR #9 spatial-CWT machinery (epic #197, roadmap #9, issue #229). Spatial sibling
of PR #5 `temporal_cwt`; input is PR #8 `MidlineResult`. Full brainstorm +
empirical investigation: `docs/superpowers/specs/2026-06-09-add-circumnutation-tier3b-spatial-cwt-design.md`.
Investigation artifacts: `_scratch/2026-06-09-lgz-localizer-comparison/` (local).

## Goals / Non-Goals

- **Goals:** uniform-arc-length resample of κ(s); cgau2 spatial scaleogram + COI;
  spatial ridge → λ(s_a). Deterministic, cross-OS, pure-pixel, no trait emission.
- **Non-Goals:** trait columns / DataFrame / groupby (PR #10); `L_gz` mask (CC-1,
  PR #10); **`L_gz`/`L_c` detection** (descoped — premise doesn't transfer, #230).

## Decisions

- **D1 API:** separate `resample_curvature` entry (raw arrays — no `spatial_cwt→
  midline` import edge, unit-testable, optional mask = caller control);
  `compute_scaleogram(kappa, ds, constants=None)` keeps the locked stub signature
  (drop the stub file's `wavelet=`/`scale_range=` kwargs; derive from constants).
  Entry-takes-arrays / downstream-takes-Result symmetry with `temporal_cwt`.
- **D2 mask gaps:** drop `velocity_sub_noise_mask` frames (κ=(ẋÿ−ẏẍ)/|v|³ blows up
  at sub-noise |v|), then `np.interp` survivors; guards → graceful all-NaN. No new
  dep; tight determinism. Empirically validated on real plate-001.
- **D3 ds:** median of positive surviving Δ(arc_length); λ-range derives from it.
- **D4 topology:** 3 funcs / 3 frozen Results (`ResampleResult`,
  `SpatialScaleogramResult`, `SpatialRidgeResult`) — direct twin of
  `temporal_cwt` (`compute_scaleogram` + `extract_ridge`) plus the resample entry.
- **D6 constants (5→6):** `SPATIAL_COI_EFOLDING_FACTOR` (cgau2, empirically
  measured via step-response — the cmor `COI_EFOLDING_FACTOR=√1.5` docstring
  already defers the cgau2 factor to PR #9), `CWT_WAVELENGTH_MIN_NYQUIST_FACTOR`,
  `CWT_WAVELENGTH_MAX_SIGNAL_FRACTION` (spatial siblings of `CWT_PERIOD_*`).
- **D9 apex axis:** emit `s_a = arc_max − s` (apex at 0) so PR #10 can do
  apex-vs-basal λ-consistency; λ itself is orientation-invariant. Pin with oracle.
- **D10 units:** pure-pixel (px, px⁻¹) Result fields; defer the `px⁻¹`
  `PIPELINE_UNIT_VOCABULARY` token to PR #10 (no sidecar/CSV here).

## Empirical finding driving the scope (the decisive part)

`research:investigate` on the 6 plate-001 proofread tracks: the §7.4 "L_gz = peak
of |κ(s)| envelope, L_c = basal exp decay" premise does NOT transfer to top-view
tip-trail κ(s). Trail spans ≈3050–3527 px; growth zone (~29–65 px) ≈1 % of trail;
|κ| envelope global-argmax lands basal (48–100 % of span) for all three candidate
methods (Hilbert / cgau2 CWT-power / smoothed-|κ|); apical/basal Hilbert ratio
< 1 on 5/6 tracks. Raw κ(s_a) is a steady-amplitude oscillation over the whole
trail (QPB steady traveling wave, §4.7) — as theory §6.5 warned ("recent trail
cannot be inverted"). Oracle confirms the methods work when an apical peak exists,
so the data — not the method — lacks the structure. → ship scaleogram + λ_spatial
(supported); descope `L_gz`/`L_c` to #230.

## Determinism (CC-6)

No `curve_fit` (descoped) → `np.interp` + `pywt.cwt(cgau2)` + argmax-ridge holds
at `atol=1e-9, rtol=0` cross-OS (like PR #5). Analytic pure-sinusoid-of-known-λ
oracle (recover λ at interior COI-dodging positions) + a captured canary at
interior `[scale_idx, positions]`.

## Risks / Trade-offs

- **cgau2 COI factor unknown at design time** → measured empirically in GREEN
  (task 6.1); `SPATIAL_COI_EFOLDING_FACTOR` default set from the measurement.
- **L_gz/L_c descope ripples to PR #10** → its `L_gz`-dependent traits + mask are
  blocked on #230, but `traveling_wave_residual` + `apex_basal_period_consistency`
  remain deliverable from λ(s_a). Documented as a scope note, not changed here.
- **Single-plate evidence** (plate-001) → multi-plate validation deferred (#220 multi-plate backlog; #202 the broader sweep).

## Migration Plan

Foundation-test migration atomic with the first non-raising commit (impl 7→8,
stub 4→3; `_CONSTANTS_VERSION` assertion 5→6 in BOTH `test_circumnutation_foundation.py`
AND `test_circumnutation_temporal_cwt.py::test_2G4`, same commit). `theory.md`
§7.4/§6.3 patched with the deviation AND the §6.5 trait-status table rows for
`L_gz`/`L_c` flipped from "✓ Measurable" to "✗ — see #230" (resolving the
§6.5-vs-§7.4 internal contradiction); original §7.4 preserved in Appendix B.
`roadmap.md` row #9 / CC-1 / row #10 reconciled to the descope.

## Open Questions

- Measured `SPATIAL_COI_EFOLDING_FACTOR` value (resolved in GREEN).
- PR #10 scope adjustment (tracked via #230; not changed by PR #9).
