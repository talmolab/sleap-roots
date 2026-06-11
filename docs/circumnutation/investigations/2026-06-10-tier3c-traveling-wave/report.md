# Investigation: Tier 3c traveling-wave traits — real-data grounding

**Date**: 2026-06-10
**Author**: eberrigan (+ Claude)
**Context**: pre-spec evidence for PR #10 (`add-circumnutation-tier3c-traits`), epic #197.
**Status**: complete — decisions fed into the OpenSpec proposal + `theory.md` deviation record.

## Why this investigation
PR #10 emits the program's central falsifiable trait, `traveling_wave_residual`
(the QPB test λ_spatial = v·T). PR #7/#8/#9 each shipped a theory premise that turned out
empirically false on real roots (helix; velocity-mask; L_gz |κ|-envelope peak). Per that
lesson, the Tier 3c trait definitions were grounded in real data **before** the spec locked,
on all 6 Nipponbare plate-001 proofread tracks
(`tests/data/circumnutation_nipponbare_plate_001/plate_001_greyscale.tracked_proofread.slp`).

Per-track chain (cadence_s = 300 s, `coordinate="lateral"`):
`midline.reconstruct → spatial_cwt.resample_curvature → compute_scaleogram → extract_ridge`
→ COI gate (`~in_coi`) → cgau2 calibration
(`tests/data/circumnutation_spatial_cwt_calibration.json`) → λ statistics; composed with
Tier 0 `v_total_median_px_per_frame` and Tier 1 `T_nutation_median` / `is_nutating`.

## Pass 1 — per-track operands
`lam_exp = v · T_frames`, `T_frames = T_nutation_median / cadence_s`.
`res_* = |λ − lam_exp| / lam_exp`. `_raw` = honest cgau2 value; `_cal` = calibrated to true px.

| track | n_int | coi_frac | v | T_s | T_fr | is_nut | lam_exp | lam_raw | lam_cal | res_raw | res_cal |
|--|--|--|--|--|--|--|--|--|--|--|--|
| 0 | 279 | 0.858 | 7.06 | 4917 | 16.4 | T | 115.7 | 142.5 | 125.7 | 0.232 | 0.087 |
| 1 | 350 | 0.902 | 6.77 | 4011 | 13.4 | T | 90.5  | 120.4 | 106.2 | 0.329 | 0.173 |
| 2 | 414 | 0.961 | 7.03 | 3748 | 12.5 | T | 87.8  | 84.7  | 75.0  | 0.036 | 0.146 |
| 3 | 385 | 0.977 | 7.01 | 4595 | 15.3 | T | 107.3 | 103.4 | 91.3  | 0.036 | 0.149 |
| 4 | 377 | 0.967 | 6.86 | 4011 | 13.4 | T | 91.7  | 85.1  | 75.5  | 0.072 | 0.177 |
| 5 | 389 | 0.926 | 6.85 | 4011 | 13.4 | T | 91.7  | 93.1  | 82.2  | 0.016 | 0.103 |

- All 6 tracks nutate; COI-valid fraction 0.86–0.98 (in-COI fraction 0.02–0.14 ≪ `COI_FRACTION_MAX=0.5`).
- cgau2 over-reports λ by mean factor 1.13 (calibration shrinks λ_median by ×0.884).

## Pass 2 — orientation-invariant λ(s_a) spread
MAD/median = robust fractional spread of the ridge wavelength over COI-valid positions.

| track | MAD/median | CV (std/mean) | IQR/median |
|--|--|--|--|
| 0 | 0.368 | 0.800 | 2.179 |
| 1 | 0.308 | 0.872 | 2.069 |
| 2 | 0.130 | 0.861 | 0.199 |
| 3 | 0.362 | 1.001 | 2.403 |
| 4 | 0.188 | 0.873 | 0.403 |
| 5 | 0.270 | 0.877 | 1.645 |

The CWT ridge λ(s_a) is intrinsically spiky (per-position argmax jumps scales), so CV and
IQR/median are unstable/outlier-driven; **MAD/median (0.13–0.37) is the only robust measure.**
Directional apex(s_a→0)-vs-basal(s_a→max) window ratios ranged 0.22–3.22 — edge-artifact-
dominated (e.g. track-0 apex λ ≈ 432 px, >3× its own median) — and the apex orientation is
ambiguous (resample's s_a=0 is frame-0 = oldest tissue, possibly opposite the growing tip;
§6.2 sign-polarity warning). A symmetric spread sidesteps both.

## Findings → decisions

1. **D3 — the §7.4 handoff's "bias cancels in the ratio" is FALSE.** `v·T_frames` is a true-px
   prediction with no cgau2 bias, so `|λ_raw − v·T|/(v·T)` is **mixed-domain** and wrong; it
   only *looks* small (median 0.054) because the +13% over-report accidentally nudges λ_raw
   toward v·T. The honest residual calibrates λ first → **median 0.147, range [0.087, 0.177]**.
   **Decision: compute one calibrated λ (true px) and use it for both `lambda_spatial_median_px`
   and `traveling_wave_residual`.** Recorded as a deviation in `theory.md` Appendix B.

2. **D7 — QPB holds to ~9–18%** on all 6 tracks (calibrated). A real biological agreement with
   λ = v·T; not a clean ≈0, not a falsification.

3. **D4 — the H1 trait is an orientation-invariant spread, not an apex-vs-basal difference.**
   Implement "is λ(s) uniform along the trail?" as **MAD(λ_cal)/median(λ_cal)** over COI-valid
   positions (continuous; 0 = uniform, larger = more variation). Renamed
   `apex_basal_period_consistency` → **`lambda_spatial_variation`** (the value is a spread, and
   the name is plainer). Recorded as a rename in `theory.md`.

4. **D3 calibration range — extend the table.** Observed λ_reported reaches ~142 px but the
   calibration JSON covers only [21, 91] px, so tracks 0/1 currently rely on clamped
   extrapolation. PR #10 regenerates the artifact via `capture_spatial_coi_factor.py` with
   `lambda_true` up to ~150 px (deterministic → existing entries unchanged → PR #9's
   tolerance-based test stays green).

5. **D5 — no new constants.** `COI_FRACTION_MAX=0.5` already exists (unused until now);
   MAD/median + continuous residual need no thresholds. `_CONSTANTS_VERSION` stays 6.

## Reproduce
The analysis scripts and the per-track operands are committed beside this report so the result
is reproducible from the repo (the raw working scratch folder remains gitignored per convention):

- `scripts/measure_tracks.py` — pass 1 (operands + residuals)
- `scripts/measure_spread.py` — pass 2 (orientation-invariant spread)
- `operands.csv` — distilled pass-1 table

Run with `uv run python docs/circumnutation/investigations/2026-06-10-tier3c-traveling-wave/scripts/measure_tracks.py`
(the scripts locate the repo root via `pyproject.toml`, so they run from any cwd). Requires the
Git-LFS proofread fixture `tests/data/circumnutation_nipponbare_plate_001/`.
