# Circumnutation pipeline — implementation roadmap

This is the umbrella tracking doc for adding circumnutation analysis to sleap-roots. It is the *only* place the full multi-PR program lives. It does not duplicate the theoretical content (in `theory.md`) or the empirical feasibility numbers (in `preliminary_results_2026-05-07.md`); it points to them.

Each numbered PR below has its own OpenSpec change-id and its own GitHub issue. **OpenSpec changes are written just-in-time**, when the corresponding PR is opened — not all up front. **GitHub issues are drafted lazily** to `c:\vaults\sleap-roots\circumnutation\github_issues\` as we approach each PR. The status checkboxes here are the source of truth for what's done, in flight, and queued.

---

## Vision

Tracked SLEAP tip trajectories from `TrackedTipPipeline` enter; per-plant scalar circumnutation traits (period, amplitude, handedness, growth-zone length, balance number, traveling-wave residual) and per-plant time-series CSVs and diagnostic plots come out. Trait list, units, and source citations are fixed by `theory.md` §7. Validation is three-layer: synthetic, Derr-pilot regression, cross-tier consistency.

## Inputs and outputs

- **Input:** per-track tip trajectories `(series, sample_uid, timepoint, track_id, frame, tip_x, tip_y)` plus optional `R_px` (root cross-section radius in pixels, used by Tier 4). **No `px_per_mm`** — the pipeline is pure-pixel per CC-3 below; calibration is handled by the downstream `convert_to_mm()` utility, never inside the pipeline. Today, in this codebase, `track_id` and `plant_id` are 1:1 — a track *is* a plant. The schema reserves both columns so that a future change (e.g., multi-track-per-plant scenarios) is non-breaking.
- **Output:** `traits_per_plant.csv` (one row per `(series, sample_uid, plate_id, plant_id, track_id)`), `traits_per_plant.units.json`, `trajectory_per_plant/<id>.csv`, `plots/*.png`, `run_metadata.json`.

## Architecture (link only — not duplicated here)

Five computation tiers + QC tier, full description in `theory.md` §6.3. Trait list with units and source citations in `theory.md` §7. Empirical feasibility numbers (noise floor, period, amplitude, SNR, DPI ambiguity, Nyquist comfort) in `preliminary_results_2026-05-07.md`.

## Cross-cutting concerns

These apply across many PRs and need to be designed into the foundation rather than retro-fitted.

### CC-1. L_gz mask is applied AFTER detection, not before.

The growth-zone mask (apical region of the tip-trail, length ≈ `L_gz`) is *not* known at midline-reconstruction time. The correct algorithm is:

1. Smooth `(x, y)`; reconstruct full midline with arc-length `s(τ)`; compute `κ(s)` along the *full* trail. **No mask.** (PR #8.)
2. Spatial CWT on the full `κ(s)`; peak-find `L_gz_estimate` from the `|κ(s)|` envelope; fit exponential decay basal of the peak for `L_c_estimate`. (PR #9.)
3. *Now* construct the growth-zone mask using the detected `L_gz`, and apply it to compute downstream traits (`λ_spatial_median`, `traveling_wave_residual`) on the basal-of-peak portion only. (PR #10.)

PR #8's "growth-zone mask" responsibility is removed; PR #10 owns it.

### CC-2. Configuration strategy: module-level named constants.

Match `tracked_tip_pipeline.py` convention: tuple/dict module-level constants (e.g., `_TRACKED_TIP_UNITS`, `_VALID_ROOT_TYPES`, `_SCHEMA_VERSION`). One `_constants.py` per tier OR top-level if shared. Magic numbers to name (non-exhaustive — flesh out in PR #1):

| Constant | Default | Source |
|---|---|---|
| `NOISE_MASK_K` | 2 | `theory.md` §6.2 (κ-mask: \|v\| > 2σ_v) |
| `LGZ_STEADY_STATE_RESIDUAL_MAX` | 0.2 | `theory.md` §7.4 |
| `NYQUIST_RATIO_MAX` | 0.25 | `theory.md` §6.5 |
| `SG_D2_AGREEMENT_MAX` | 1.5 | `theory.md` §7.6 (clean-track threshold) |
| `LGZ_NMIN_RESOLVABLE` | 5 | `theory.md` §6.4 |
| `COI_FRACTION_MAX` | 0.5 | `theory.md` §7.6 |
| `BAND_POWER_NOISE_RATIO` | 3 | `theory.md` §7.6 (`is_nutating` threshold) |
| `WAVELET_DEFAULT_TEMPORAL` | `"cmor1.5-1.0"` | Forensic match to Derr Sept-2025 oracle |
| `WAVELET_DEFAULT_SPATIAL` | `"cgau2"` | Rivière 2022 §"Kinematics: fine elongation measurements" |
| `SG_WINDOW_SHORT` | 5 | `preliminary_results.md` §3.3 |
| `SG_DEGREE` | 3 | `preliminary_results.md` §3.3 |
| `SG_WINDOW_DETREND` | 23 | `preliminary_results.md` §3.4 (≈ 2 nutation periods) |
| `OUTLIER_STEP_RATIO` | 2 | `preliminary_results.md` §4.1 (median multiplier) |

Each constant is overridable via the pipeline class init or per-call kwarg, but defaults flow from one place.

### CC-3. Pure-pixel pipeline convention; downstream `convert_to_mm()` utility.

The pipeline NEVER takes `px_per_mm` as a parameter and NEVER emits `[mm]` columns. Every length-bearing trait is in pixels (`px`, `px²`, `px/frame`, `px/hr`, `px·hr⁻¹`); time in `hr` or `s`; angles in `rad`; rates in `hr⁻¹`; ratios as dimensionless. This matches `TrackedTipPipeline`'s convention exactly — `_TRACKED_TIP_UNITS` declares `lengths: "pixels"`. Calibration is always a downstream concern.

Users who want mm output compose `sleap_roots.circumnutation.units.convert_to_mm(traits_df, units, px_per_mm)` on the trait DataFrame. This pure function lives in PR #1 (foundation), is fully tested in isolation, and never affects the pipeline output. The `R` parameter for Tier 4 (Bastien-Meroz) becomes `R_px` — the user provides their root cross-section radius in pixels using whatever calibration they trust; the formula `δ̇₀ = ω·R·Δφ / (2·ΔL)` cancels dimensions because `R/ΔL` is a length ratio, so `R_px / ΔL_px` is dimensionless and `δ̇₀` is `hr⁻¹` regardless of calibration.

This decouples the DPI ambiguity (PR #19) from the pipeline entirely. The pipeline can ship and run on plate 001 today without resolving DPI; the resolved value flows into `convert_to_mm()` calls downstream.

### CC-4. Row identity in trait CSV.

To make per-plate / per-genotype / per-plant aggregation feasible without re-engineering, the per-plant trait CSV row-identity columns are:

```
series, sample_uid, timepoint, plate_id, plant_id, track_id, genotype, treatment
```

Today `plant_id == track_id` everywhere; both columns are populated identically. The schema reserves both so a future divergence (multi-track-per-plant, e.g. measuring a plant via tip + a side-marker) is a non-breaking change. `genotype` and `treatment` are populated from `Series` metadata if available, NaN otherwise. PR #1 task: confirm `TrackedTipPipeline` output exposes `plate_id` and the metadata fields, and document upstream-dependency contract if it doesn't.

### CC-5. Growth-axis edge case (low-displacement plants).

Some real tracks have net displacement comparable to per-frame noise (mutants, stressed plants, very early imaging windows). The growth axis defined as `(x_N - x_1) / |x_N - x_1|` is meaningless in that regime. PR #2 must:

1. Compute net displacement `D = |x_N - x_1|` (in pixels).
2. If `D < GROWTH_AXIS_RELIABILITY_K * sg_residual_xy` (recommend `K = 10`), set `growth_axis_unreliable = True` and set the rotation-dependent traits (`v_long_median`, `v_lat_median`, `long_lat_ratio`, `principal_axis_angle`) to NaN.
3. The QC tier emits `growth_axis_unreliable` as a bool flag.

### CC-6. CWT determinism / reproducibility.

- Synthetic generator (PR #4) accepts an explicit `random_state` (numpy `Generator` or seed `int`) and propagates it.
- `pywavelets.cwt` is deterministic — confirm in PR #5.
- The ridge-extraction algorithm (PR #5) MUST be deterministic — no random tie-breaking, no shuffling. Document in the function docstring.
- Tests assert determinism: same input → identical output across two runs, including in CI on different OSs.

### CC-7. Tier 1 coordinate choice for Derr regression.

Derr's Sept-2025 oracle signal was a 1D coordinate but he didn't document which. Forensic reading (`preliminary_results.md` and the input PDF analysis): the ~80 px non-monotonic drift + ~10 px oscillation matches the **lateral coordinate** (perpendicular to growth axis), not raw image-x or image-y. Resolution:

- **Default:** `coordinate="lateral"` for Tier 1 (uses Tier 0's growth-axis estimate).
- **Configurable:** `coordinate=` accepts `["x", "y", "longitudinal", "lateral"]`.
- **Layer-2 regression test (PR #6):** uses `coordinate="lateral"` and asserts `T = 3333 s ± 2%`. If the assertion fails, log the result for `coordinate="x"`, `"y"` as a diagnostic.
- **Close-out PR #21** (verify-equations) confirms with Julien which coordinate his pilot used and locks in the test target.

### CC-8. `is_nutating` threshold: `noise_floor_estimate` definition.

`is_nutating` is defined in `theory.md` §7.6 as `band_power_ratio > BAND_POWER_NOISE_RATIO * noise_floor_estimate`. Pin the noise-floor definition:

- `noise_floor_estimate` = median of the Tier-1 Fourier amplitude spectrum over frequencies `> 5 × (1 / T_nutation_median)` (out-of-band Fourier floor).
- Rationale: signal-band-relative, no dependence on calibration, matches Derr's plot's empty-spectrum region.

Implementation in PR #6.

### CC-9. Logging convention.

Per-module logger via `logger = logging.getLogger(__name__)`, matching `tracked_tip_pipeline.py`. Per-plate progress messages at `INFO`; per-plant inside-plate messages at `DEBUG`. PR #1 sets up; subsequent PRs use.

### CC-10. Number of independent noise estimators in QC tier (Phase 1).

Per Elizabeth's directive (this conversation): all three independent noise estimators are emitted in Phase 1 — `sg_residual_xy`, `d2_noise_xy`, `msd_noise_xy`. Pairwise agreement traits: `sg_d2_agreement`, `sg_msd_agreement`, `d2_msd_agreement`. `track_is_clean` requires all three pairwise agreements ≤ 1.5. Implementation in PR #3.

## PR / issue split

Status: ⬜ not started | 🟡 in flight | ✅ merged. Issue links populated as drafts land in the vault.

Each row is one PR, one OpenSpec change-id (`add-circumnutation-<topic>` for code PRs; `verify-…` for close-out docs PRs), and one GitHub issue.

### Phase 1 — main pipeline

| # | OpenSpec change-id | Scope | Issue draft | Status |
|---|---|---|---|---|
| 1 | `add-circumnutation-foundation` | Deps (`pywavelets`, explicit `scipy`); package skeleton; `CircumnutationInputs` data class; calibration contract (CC-3 investigation); module-constants pattern (CC-2); row-identity schema (CC-4); units sidecar; logging (CC-9); `__init__.py` re-exports. | ⬜ | ⬜ |
| 2 | `add-circumnutation-tier0-kinematics` | Tier 0 traits (theory.md §7.1); growth-axis edge case (CC-5); plate 001 sanity tests. | ⬜ | ⬜ |
| 3 | `add-circumnutation-qc-tier` | All QC traits (theory.md §7.6) including `msd_noise_xy` (CC-10); `cadence_nyquist_ratio` (theory.md §6.5); `growth_axis_unreliable`; `qc_failure_reason`. | ⬜ | ⬜ |
| 4 | `add-circumnutation-synthetic-generator` | `synthetic.generate_trajectory` integrating Rivière 2022 Eq. 4 forward; `random_state` propagation (CC-6); sanity tests. | ⬜ | ⬜ |
| 5 | `add-circumnutation-temporal-cwt-machinery` | `temporal_cwt.compute_scaleogram` + `extract_ridge` + COI mask; deterministic per CC-6; no trait emission. | ⬜ | ⬜ |
| 6 | `add-circumnutation-tier1-derr-faithful` | Tier 1 trait emission (theory.md §7.2); `coordinate=` parameter (CC-7); `noise_floor_estimate` definition (CC-8); Layer-2 Derr regression. | ⬜ | ⬜ |
| 7 | `add-circumnutation-tier2-psi-g` | `ψ_g(t)` unwrap + smooth (theory.md §3.5); Tier 2 trait emission (theory.md §7.3); cross-tier consistency check. | ⬜ | ⬜ |
| 8 | `add-circumnutation-tier3a-midline` | Tip-trail-as-midline reconstruction (theory.md §6.1); arc-length `s(τ)`; `κ(s)`; `v(t)`; SG smoothing of `(x,y)`. **No growth-zone mask** (CC-1). | ⬜ | ⬜ |
| 9 | `add-circumnutation-tier3b-spatial-cwt` | Spatial CWT (`cgau2` default); `L_gz` peak-finder; `L_c` exponential-decay fit. **No mask** (CC-1). | ⬜ | ⬜ |
| 10 | `add-circumnutation-tier3c-traits` | Tier-3 trait emission (theory.md §7.4) including `B_balance_number`, `L_gz_steady_state_residual`, `L_gz_resolvable`, `traveling_wave_residual`, `apex_basal_period_consistency`. **Applies the L_gz growth-zone mask** (CC-1). | ⬜ | ⬜ |
| 11 | `add-circumnutation-tier4-parametric` | `gamma_over_beta` alias + `delta_dot_0_estimate` (theory.md §7.5); Phase-2 NaN columns (`beta`, `gamma`, `theta_p`). | ⬜ | ⬜ |
| 12 | `add-circumnutation-layer1-validation` | Synthetic-data parameterized suite; Averrhoa-scale + rice-scale; both handedness signs; tolerances per theory.md §8 Layer 1. | ⬜ | ⬜ |
| 13 | `add-circumnutation-layer3-cross-tier` | Explicit Layer-3 cross-tier consistency tests as a coherent class (`T_nutation ↔ T_psig ±5%`, `λ_spatial ↔ v·T ±10%`). | ⬜ | ⬜ |
| 14 | `add-circumnutation-pipeline` | `CircumnutationPipeline` DAG composition; pickle-ability; integration test on plate 001; CSV writer with full row-identity (CC-4). | ⬜ | ⬜ |
| 15 | `add-circumnutation-per-genotype-aggregation` | `aggregate_by_genotype()` per theory.md §7.7 (median ± IQR; `n_plants_passing_qc`; exclusion reasons). Schema-aware: groups by `(genotype, treatment)`, reports per-`plate_id` separately. | ⬜ | ⬜ |
| 16 | `add-circumnutation-plots` | Scaleograms (Tier 1 + Tier 3); trail overlay (κ-color-coded); 6-up plate panel; `--no-plots` flag. | ⬜ | ⬜ |
| 17 | `add-circumnutation-cli` | `sleap-roots circumnutation analyze`. | ⬜ | ⬜ |
| 18 | `add-circumnutation-user-guide` | `pipeline_guide.md`; cookbook entry; mkdocs nav; `docs/changelog.md` entry. | ⬜ | ⬜ |

### Phase 1.5 — close-out (parallel; not blocking Phase 1 merge)

| # | OpenSpec change-id (or N/A) | Scope | Issue draft | Status |
|---|---|---|---|---|
| 19 | N/A (coordination + docs) | Resolve DPI calibration with Suyash; document resolved value(s) + per-plate variation in new `docs/circumnutation/calibration.md`. Until resolved, all `[mm]` traits are scale-tentative — pipeline runs in pixels internally so this never blocks merging code. | ⬜ | ⬜ |
| 20 | `verify-rice-l-gz-citation` | Confirm or replace Iijima & Kato 2007 (theory.md §6.4 + §11). Update `theory.md` reference. Small docs PR. | ⬜ | ⬜ |
| 21 | `verify-equation-attributions` | Append a verification log to `theory.md` Appendix B confirming BM2016 Eqs. 13/14/20/21/23, Rivière 2022 Eqs. 1–5, Meroz 2026 Eqs. 1/5/6/7 against published PDFs. Coordinate with Julien on the Tier 1 coordinate choice (CC-7) while at it. Small docs PR. | ⬜ | ⬜ |
| 22 | `add-taylor-r-pipeline-cross-validation` | Python re-implementation of LOESS centerline + drift correction + natural-spline + peak-finding (Taylor 2021 PNAS supplementary R). Regression test on `taylor_2021_pnas_loess_spline/data/all_tip_tracking_combined.csv` matching published amplitudes. NOT a Tier in the main pipeline — this is a *cross-validation reference implementation*. | ⬜ | ⬜ |

### Beyond Phase 1.5 (out of this roadmap)

- Multi-node SLEAP skeleton tracking upstream → unlocks the spatial profile traits the current tip-only pipeline can't measure (theory.md §6.6).
- Full `(β, γ, θ_p)` identification with gravitropism stimulus.
- tsfresh exploratory feature extraction layer.
- 3D reconstruction.
- AFM / cell-wall biology workstream.
- Live monitoring during timelapse (post-hoc analysis only in Phase 1).
- Multi-plate analysis: re-run on plates 002 (GA₄), 003 (MOCK rep), 004 (TZT) once Phase 1 is stable on plate 001.

## Open scientific blockers (live-tracked)

These are flagged in `theory.md` and `preliminary_results.md`. The pipeline runs around them; resolution moves Phase-1 deliverables from "scale-tentative" to "publication-defensible." Tracked as PRs #19–22 above. None blocks Phase-1 code merging.

## How to execute

1. Pick the next ⬜ row.
2. Draft the GitHub issue body to `c:\vaults\sleap-roots\circumnutation\github_issues\issue_<change-id>.md`. Reference the OpenSpec change-id, the relevant theory.md section, the relevant `preliminary_results.md` section if applicable, the cross-cutting concerns it touches.
3. Open the GitHub issue (Elizabeth posts).
4. Scaffold the OpenSpec change at `openspec/changes/<change-id>/` with `proposal.md`, optional `design.md`, `tasks.md`, `specs/circumnutation/spec.md`. Validate with `openspec validate <change-id> --strict`.
5. TDD-implement per `tasks.md`. Open PR linking issue + change-id.
6. After merge, archive with `openspec archive <change-id> --yes` to fold the requirements into `openspec/specs/circumnutation/spec.md`.
7. Update this roadmap's status checkbox.
