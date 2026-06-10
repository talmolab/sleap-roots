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
  (task 1.1); `SPATIAL_COI_EFOLDING_FACTOR` default set from the measurement. The
  same capture also characterizes the cgau2 `scale2frequency` wavelength
  calibration: a λ- AND n-dependent offset (≈+5–20%, from grid quantization +
  center-freq convention — NOT a single factor), so the λ-recovery oracle tolerance
  is a measured band per-(n, scale-count), not a speculative ±5%. See Handoff to
  PR #10 #2 for why we ship the convention value rather than baking a correction.
  Use the **impulse 1/e** stimulus for the COI factor (≈1.33–1.35), not a literal
  step (round-3 finding).
- **L_gz/L_c descope ripples to PR #10** → its `L_gz`-dependent traits + mask are
  blocked on #230, but `traveling_wave_residual` + `apex_basal_period_consistency`
  remain deliverable from λ(s_a). Documented as a scope note, not changed here.
- **Single-plate evidence** (plate-001) → multi-plate validation deferred (#220 multi-plate backlog; #202 the broader sweep).

## Handoff to PR #10 (must be documented now to protect the central trait)

PR #10's `traveling_wave_residual = |λ_spatial − v·T| / (v·T)` is the program's
central falsifiable trait. PR #9 ships λ_spatial; FOUR handoffs MUST be written
into `theory.md` §7.4 (next to `traveling_wave_residual`) so PR #10 cannot
silently corrupt it (round-3/4 scientific review). Round-4 verified the formula
below is dimensionally correct AND its operand units match the actual emitted
trait units in the code:

1. **Unit reconciliation (dimensional landmine).** Emitted units, NOT the stale
   §7.x trait-table annotations: `wavelengths_px` = **px**; Tier-0
   `v_total_median_px_per_frame` = **px/frame** (kinematics.py); the temporal ridge
   `periods_s` = **seconds**; **`T_nutation_median` is emitted in SECONDS too**
   (nutation.py `nanmedian(periods_s)`, column has no suffix) — the §7.2/§7.4 trait
   table's "hr" annotation is STALE (confirmed by Appendix B(3), where `T_psig`
   was corrected to seconds "for consistency with Tier 1"); do NOT multiply by
   3600. So `v·T` naively is `px·s/frame`, NOT px. The correct comparison is
   `λ_spatial[px]  vs  v[px/frame] · (T_nutation_median[s] / cadence_s[s/frame])`
   (i.e. `T_frames = periods_s / cadence_s`; theory §6.4 uses T in **frames**:
   `5.83 px/frame × (3333 s / 300 s) ≈ 65 px` ✓). §7.4's **mm** λ annotation is
   post-`convert_to_mm` only — compare both sides in the SAME unit.
2. **cgau2 wavelength calibration (do NOT bake a correction into `wavelengths_px`).**
   `wavelengths_px` is the honest `pywt.scale2frequency("cgau2", …)` convention
   value. Round-3/4 probing shows it over-reports the true px wavelength by a
   **λ- AND n-dependent** band (≈+5–20% over λ∈[20,80]px, n∈{200,400,600}) driven
   by the 64-scale log-grid quantization + cgau2's center-freq convention — so it
   is NOT a single calibratable constant. PR #9 ships the convention value and
   **commits the task-1.1 measurement as a machine-readable calibration artifact**
   (`scripts/circumnutation/` emits a small JSON/CSV `{n, scale_count, lambda_true,
   lambda_reported}` map) — NOT prose-only — so PR #10's absolute-λ consumers
   (`lambda_spatial_median`) and the oracle literals are reproducible, not hand-typed.
   For `traveling_wave_residual` the bias cancels via two-sided same-axis comparison
   (compare in scale space, OR raise `CWT_SCALE_COUNT_DEFAULT`, OR widen tolerance);
   for absolute-λ traits it does NOT cancel and the map must be applied. A +10%
   one-sided bias could flip "confirmed" vs "refuted." NOT added as a constant
   (a single number would lie).
3. **COI reliability gate.** `extract_ridge` ships the raw per-position ridge
   (correct — mirrors temporal). PR #10 MUST apply a spatial `COI_FRACTION_MAX`-style
   gate (NaN λ_spatial when too few `~in_coi` positions survive), exactly as the
   temporal tier does, so a sparse/short trail does not ship a few-sample median as
   authoritative. **Any λ_spatial appearing in PR #9's own artifacts/canaries is
   UNGATED and not authoritative;** PR #9's real-data test asserts a HARD minimum
   `~in_coi` count floor (not record-only) as an interim guard.
4. **`apex_basal_period_consistency` — the λ-offset does NOT cancel here.** This
   trait compares λ at the apex (`s_a → 0`, short λ) vs the base (`s_a → max`, long
   λ) on the SAME scaleogram. Because the cgau2 over-report is **λ-dependent**, apex
   and basal positions get DIFFERENT fractional offsets, so a genuinely uniform-λ
   trail shows SPURIOUS apex-vs-basal variation — a false H1-violation generator.
   PR #10 MUST apply the calibration map (#2) per-λ BEFORE comparing apex vs basal,
   and MUST pin the direction: **apex = `s_a → 0`, base = `s_a → max`** (do NOT infer
   apex from raw `arc_length`; recall the PR #8 `sign(κ) = −handedness` polarity
   warning for any chirality trait).

## Migration Plan

Foundation-test migration atomic with the first non-raising commit (impl 7→8,
stub 4→3; `_CONSTANTS_VERSION` assertion 5→6 in BOTH `test_circumnutation_foundation.py`
AND `test_circumnutation_temporal_cwt.py::test_2G4`, same commit). `theory.md`
§7.4/§6.3 patched with the deviation AND the §6.6 trait-status table rows
("What this means for the trait list") for `L_gz`/`L_c` flipped from "✓ Measurable"
to "✗ — see #230" (resolving the §6.6-vs-§7.4 internal contradiction); original
§7.4 preserved in Appendix B.
`roadmap.md` row #9 / CC-1 / row #10 reconciled to the descope.

## Open Questions

- Measured `SPATIAL_COI_EFOLDING_FACTOR` value (resolved in GREEN).
- PR #10 scope adjustment (tracked via #230; not changed by PR #9).
