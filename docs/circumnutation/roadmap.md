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
2. Spatial CWT on the full `κ(s)`; extract the dominant spatial-wavelength ridge `λ(s_a)`. (PR #9.)
3. *Now* (once `L_gz` is available) construct the growth-zone mask and apply it to compute downstream traits (`λ_spatial_median`, `traveling_wave_residual`) on the basal-of-peak portion only. (PR #10.)

**⚠ PR #9 update (empirical, 2026-06-09):** step 2's original plan — "peak-find `L_gz_estimate` from the `|κ(s)|` envelope; fit exponential decay basal of the peak for `L_c_estimate`" — was found NOT to transfer to top-view tip-trail κ(s) (the trail is a steady-amplitude oscillation over ~3000 px; the growth zone is ~1 % of the trail; the |κ| envelope global-argmax lands basal on all 6 plate-001 tracks — exactly the §6.5 "recent trail cannot be inverted" limit). PR #9 therefore ships **resample + cgau2 scaleogram + λ_spatial ridge** only; `L_gz`/`L_c` detection (and hence the `L_gz` growth-zone mask in step 3) are **descoped to follow-up research issue #230**. PR #8's "growth-zone mask" responsibility is removed; PR #10 owns it once #230 resolves.

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

**`plant_id` semantics, populating responsibility, and future divergence.** `plant_id` is meant to be a **stable identifier for the physical plant entity**, decoupled from the tracker-assigned `track_id`. The foundation contract is intentionally minimal: it requires the column to exist (and tests assert that callers conventionally set `plant_id = track_id` today), but it does **not** enforce equality at runtime. Per-plant aggregation in tier PRs (`aggregate_by_genotype`, see PR #15) keys on `plant_id` rather than `track_id` so the aggregation stays correct under any of the three foreseeable divergence cases:

1. **Track fragmentation.** The SLEAP tracker loses a plant mid-run and re-IDs it, so two `track_id` values bind to one physical plant. An adapter / cleanup step (a future tier PR or a manual proofread upstream) sets the two fragments to the same `plant_id` while leaving their `track_id`s distinct.
2. **Multi-track-per-plant.** A future model tracks the tip + a side-marker (or any second landmark) and emits two tracks per plant. Same one-`plant_id`-many-`track_id`s pattern.
3. **External plate-position metadata.** Suyash's CMTN plates have 6 plants in known positions. A future ingest step can map "position 1 (top-left) → `plant_id="plate001_WT_rep1"`, position 2 → `plant_id="plate001_WT_rep2"`, ..." onto the trajectory DataFrame so `plant_id` carries treatment-meaningful identity rather than tracker-internal numbering.

**Populating responsibility is upstream of the foundation.** `TrackedTipPipeline` does not currently emit `plant_id`, `plate_id`, `genotype`, or `treatment`. Callers feeding `TrackedTipPipeline` output into `CircumnutationInputs` must enrich the DataFrame with these four columns before construction; the convention today is `df["plant_id"] = df["track_id"]` and the other three set to `NaN` when unavailable. A future tier PR or upstream extension (e.g., a per-frame-metadata follow-up to `series-metadata`/PR #171) may automate this enrichment. The foundation requires the full 8-column schema rather than synthesizing defaults at validation time so the row-identity contract stays explicit and any divergence shows up as a deliberate caller decision.

**Implementation already supports divergence.** `_io.build_per_plant_template` drops duplicates on the full 8-tuple and sorts by `(series, sample_uid, plate_id, plant_id, track_id)`. If a caller populates `plant_id != track_id`, the template just produces more rows (one per unique 8-tuple). No code change is needed for divergence to work mechanically; only the spec definition needs to evolve when the convention does.

### CC-5. Growth-axis edge case (low-displacement plants).

Some real tracks have net displacement comparable to per-frame noise (mutants, stressed plants, very early imaging windows). The growth axis defined as `(x_N - x_1) / |x_N - x_1|` is meaningless in that regime. PR #2 must:

1. Compute net displacement `D = |x_N - x_1|` (in pixels).
2. If `D < GROWTH_AXIS_RELIABILITY_K * sg_residual_xy` (recommend `K = 10`), set `growth_axis_unreliable = True` and set the **6 rotation-dependent traits** (`v_long_signed_median`, `v_long_abs_median`, `v_lat_signed_median`, `v_lat_abs_median`, `long_lat_ratio`, `principal_axis_angle`) to NaN. The 4-trait NaN-set originally documented here was expanded to 6 in PR #2 when the signed/abs split was adopted for the velocity traits (see PR #2's design.md D1 for the trait-list expansion rationale).
3. **BOTH Tier 0 (PR #2) AND QC tier (PR #3) emit `growth_axis_unreliable` as a bool column on their respective per-plant trait DataFrames.** Values are element-wise equal as `bool` dtype by construction because both tiers compute the flag via the same gate formula on the same inputs through the shared `sleap_roots.circumnutation._noise.compute_sg_residual_xy` helper after applying the same `to_numpy(dtype=float)` cast. The original "no re-emission" rule (carried from PR #2 per CC-5 step 3 archive) is REVERSED by PR #3 (see PR #3 design.md D5): QC's standalone usefulness requires the bool column for downstream `df[~df.growth_axis_unreliable]` filtering instead of brittle string-parsing of `qc_failure_reason`. PR #14 pipeline composition may coalesce or drop one (either is safe). Spec contract: `openspec/specs/circumnutation/spec.md` Requirement "Growth-axis reliability gate" (modified by PR #3) and Requirement "QC tier growth_axis_unreliable equality with Tier 0" (added by PR #3).

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
| 1 | `add-circumnutation-foundation` | Deps (`pywavelets`, explicit `scipy`); package skeleton; `CircumnutationInputs` data class; pure-pixel `convert_to_mm` utility (CC-3); module-constants pattern (CC-2); row-identity schema (CC-4); units sidecar + run-metadata sidecar with provenance; logging (CC-9); `__init__.py` re-exports. [#198](https://github.com/talmolab/sleap-roots/issues/198) · [PR #200](https://github.com/talmolab/sleap-roots/pull/200) | ✅ | ✅ |
| 2 | `add-circumnutation-tier0-kinematics` | Tier 0 traits (theory.md §7.1); growth-axis edge case (CC-5); plate 001 sanity tests; bonus upstream fix to `Series.get_tracked_tips` for proofread `.slp` dedup (per prelim §3.1). [#201](https://github.com/talmolab/sleap-roots/issues/201) · [PR #203](https://github.com/talmolab/sleap-roots/pull/203) · follow-up [#202](https://github.com/talmolab/sleap-roots/issues/202) (`K=10` sensitivity sweep) | ✅ | ✅ |
| 3 | `add-circumnutation-qc-tier` | QC traits per theory.md §7.6: 3 noise estimators (sg/d2/msd per CC-10), 3 pairwise agreements, 2 outlier-step diagnostics, `growth_axis_unreliable` (re-emitted per design.md D5; equal-by-construction to Tier 0's), `track_is_clean` composite, `qc_failure_reason` stable-ordered comma-separated diagnostic. `cadence_nyquist_ratio` moved to PR #6 (depends on Tier 1's `T_nutation_median`). [#204](https://github.com/talmolab/sleap-roots/issues/204) · [PR #209](https://github.com/talmolab/sleap-roots/pull/209) · follow-ups [#205](https://github.com/talmolab/sleap-roots/issues/205) (α pairwise-agreement thresholds) · [#206](https://github.com/talmolab/sleap-roots/issues/206) (β outlier-step thresholds) · [#207](https://github.com/talmolab/sleap-roots/issues/207) (γ MSD lag selection) · [#208](https://github.com/talmolab/sleap-roots/issues/208) (δ ±inf detection) | ✅ | ✅ |
| 4 | `add-circumnutation-synthetic-generator` | `synthetic.generate_trajectory` integrating Rivière 2022 Eq. 4 forward (closed-form per §4.4); `random_state` propagation (CC-6); per-axis σ = `noise_sigma_px/√2` xy-quadrature interpretation; user-facing aggregate parameters (Rivière 6-tuple deferred to PR #12 wrapper). [#211](https://github.com/talmolab/sleap-roots/issues/211) · [PR #210](https://github.com/talmolab/sleap-roots/pull/210) | ✅ | ✅ |
| 5 | `add-circumnutation-temporal-cwt-machinery` | `temporal_cwt.compute_scaleogram` + `extract_ridge` + COI mask; deterministic per CC-6; no trait emission. Wavelet-aware `COI_EFOLDING_FACTOR = math.sqrt(1.5)` for cmor1.5-1.0 (empirically verified via step-response across cmor0.5/1.0/1.5/2.0). [#212](https://github.com/talmolab/sleap-roots/issues/212) · [PR #213](https://github.com/talmolab/sleap-roots/pull/213) · follow-up [#214](https://github.com/talmolab/sleap-roots/issues/214) (ridge-tracking continuity per Mallat 1999 §4.4.2, deferred to PR #6) | ✅ | ✅ |
| 6 | `add-circumnutation-tier1-derr-faithful` | Tier 1 trait emission (theory.md §7.2); `coordinate=` parameter (CC-7); `noise_floor_estimate` definition (CC-8); Layer-2 Derr regression; **`cadence_nyquist_ratio` (moved from PR #3; depends on Tier 1's `T_nutation_median`)** per theory.md §6.5. **Closes #214** via `temporal_cwt.smooth_ridge` median-filter post-filter (Mallat 1999 §4.4.2). [#215](https://github.com/talmolab/sleap-roots/issues/215) · [PR #216](https://github.com/talmolab/sleap-roots/pull/216) · follow-ups [#217](https://github.com/talmolab/sleap-roots/issues/217) (Derr forensic-match upgrade pending raw scaleogram numerics) · [#218](https://github.com/talmolab/sleap-roots/issues/218) (TEMPORAL_NYQUIST_RATIO_MAX + DERR_EXPECTED_PERIOD_S multi-plate validation) · [#219](https://github.com/talmolab/sleap-roots/issues/219) (Layer-2 ±2% acceptance preprocessing improvements) · [#220](https://github.com/talmolab/sleap-roots/issues/220) (#214 multi-plate empirical validation) · [#222](https://github.com/talmolab/sleap-roots/issues/222) (program-wide `_px`/`_s`/`_rad` suffix convention + 6 deferred round-2 review findings) | ✅ | ✅ |
| 7 | `add-circumnutation-tier2-psi-g` | `ψ_g(t)` via locked `_geometry.compute_psi_g` (theory.md §3.5); 4 self-contained Tier 2 traits (theory.md §7.3): `T_psig_median_s` (SG-detrend→CWT), `delta_E_amplitude_proxy_px_per_frame`, `handedness` (COI-free net rotation), `helix_signed_area_px2` (**growth-detrended** Shoelace — drift-robust fix found in pre-PR self-review, ≥5/6 vs 1/6 raw on plate-001); new `_geometry.compute_signed_area`; cross-tier consistency vs Tier 0 `principal_axis_angle` (all 6 plate-001 tracks within 1.8°). No new constants (`_CONSTANTS_VERSION` stays 5). 3 recorded theory deviations (handedness COI-free; delta_E px/frame; SG-detrend not literal "smooth") + theory.md §7.3/§6.3 patched (Appendix B). `psig_long_consistency` (5th §7.3 trait) **deferred to PR #13** (Layer-3 cross-tier, owns both trait + test). [#224](https://github.com/talmolab/sleap-roots/issues/224) · [PR #225](https://github.com/talmolab/sleap-roots/pull/225) | ✅ | ✅ |
| 8 | `add-circumnutation-tier3a-midline` | Tip-trail-as-midline reconstruction (theory.md §6.1); `midline.reconstruct(x, y, cadence_s, sg_window=None, constants=None) -> MidlineResult` (frozen `attrs`, `eq=False`) emitting arc-length `s(τ)` (`cumulative_trapezoid`), per-frame curvature `κ` (theory.md §6.2), tip speed, and a per-frame velocity sub-noise mask. **Machinery** PR mirroring PR #5 (input-validating, frozen Result, determinism — **no trait emission**). SG smoothing/differentiation from one polynomial per coordinate (new `_noise.compute_sg_derivative`, deriv 0/1/2); curvature via new sign-anchored `_geometry.compute_path_curvature` (`κ=(ẋÿ−ẏẍ)/|v|³`; absolute anchor + the cross-helper `sign(κ)=−handedness` note, `dψ_g/dt=−κ·|v|`). **Frame-parameterized / cadence-independent** (px/frame). **No `L_gz` mask** (CC-1, → PR #10); **no uniform-`s` resample** (→ PR #9 owns `(kappa, ds)`); no new constants (`_CONSTANTS_VERSION` stays 5). Cross-OS determinism canary `atol=1e-9` (closed-form circle oracle + synthetic; reproduced on all 3 BLAS backends); real plate-001 validation (curvature bounded on unmasked frames, mask-fraction band, cross-tier `arc ≤` Tier 0 path length). theory.md §6.2 y-down curvature-sign clarification patched (Appendix B(4)). [#226](https://github.com/talmolab/sleap-roots/issues/226) · [PR #227](https://github.com/talmolab/sleap-roots/pull/227) | ✅ | ✅ |
| 9 | `add-circumnutation-tier3b-spatial-cwt` | Spatial CWT machinery (`cgau2` default): `resample_curvature` (non-uniform→uniform κ(s), the PR #8 handoff), `compute_scaleogram`, `extract_ridge` (λ_spatial ridge). Frozen Results, determinism canary, **no mask, no trait emission**. **`L_gz`/`L_c` DESCOPED** — the §7.4 |κ|-envelope-peak premise does not transfer to top-view tip-trail κ(s) (steady traveling wave; §6.5 inversion limit; empirically confirmed on all 6 plate-001 tracks) → follow-up research [#230](https://github.com/talmolab/sleap-roots/issues/230). 3 new constants (`SPATIAL_COI_EFOLDING_FACTOR`, `CWT_WAVELENGTH_{MIN_NYQUIST,MAX_SIGNAL}_*`); `_CONSTANTS_VERSION` 5→6. [#229](https://github.com/talmolab/sleap-roots/issues/229) · [PR #231](https://github.com/talmolab/sleap-roots/pull/231) | ✅ | ✅ |
| 10 | `add-circumnutation-tier3c-traits` | Tier 3c traveling-wave validation traits (theory.md §7.4): `lambda_spatial_median_px`, `traveling_wave_residual` (the central falsifiable QPB test λ=v·T), `lambda_spatial_variation` (orientation-robust `MAD/median` spread — **renamed from `apex_basal_period_consistency`**, §7.4 handoff note 4 overturned), + diagnostics (`lambda_expected_px`, `lambda_spatial_mad_px`, `coi_valid_fraction`). First consumer of PR #9's λ(s_a) ridge: cgau2 calibration applied in true px (n-averaged across n∈{200,400,600}, shipped as an in-package literal; **§7.4 handoff note 2 "bias cancels" overturned** — recorded in Appendix B(6)) + the §7.4 frame-domain unit handoff. New module `traveling_wave.py` (9th impl module, by ADDITION; self-contained Tier 0/1 recompute, redundant-by-design for the PR #14 DAG). **Reduced scope (PR #9 → #230):** `L_gz_estimate`/`L_c_estimate`/`B_balance_number`/`L_gz_steady_state_residual`/`L_gz_resolvable` + the L_gz growth-zone mask are **blocked on [#230](https://github.com/talmolab/sleap-roots/issues/230)** and omitted (not NaN-reserved). Calibration table extended append-only to ≥140 px. No new constants (`_CONSTANTS_VERSION` stays 6). QPB holds ~9–21% on plate-001 (provisional). [#232](https://github.com/talmolab/sleap-roots/issues/232) · [PR #233](https://github.com/talmolab/sleap-roots/pull/233) | ✅ | ✅ |
| 11 | `add-circumnutation-tier4-parametric` | **⏸ DEFERRED (2026-06-11) — revisit after [#230](https://github.com/talmolab/sleap-roots/issues/230) + a Phase-2 gravitropism-stimulus experiment.** Every Tier 4 trait is currently blocked: `gamma_over_beta` IS the `L_c` alias (theory §7.5; Meroz 2026 Eq. 6) → blocked on #230 (L_c not measurable from top-view tip-trail κ(s), per the PR #9/#10 descope); `delta_dot_0_estimate` (theory §7.4: `δ̇₀ = ωRΔφ/(2ΔL)`, `ΔL = L_c`) → blocked on #230 + needs an external root-radius `R`; `beta`/`gamma`/`theta_p` are **Phase 2** (need a gravitropism stimulus to disentangle β from γ). Rather than ship a NaN-only PR, **PR #14 (pipeline DAG) is promoted ahead of PR #11** — it is independently deliverable, gives the program its first end-to-end CSV, and resolves the Tier 0/1 recompute redundancy PR #10 deferred to it. Full deferral rationale + the revisit-handoff: `c:\vaults\sleap-roots\circumnutation\new_session_prompt_pr11.md`. ~~`gamma_over_beta` alias + `delta_dot_0_estimate` (theory.md §7.5); Phase-2 NaN columns (`beta`, `gamma`, `theta_p`).~~ | ⬜ | ⏸ |
| 12 | `add-circumnutation-layer1-validation` | Synthetic-data parameterized suite; Averrhoa-scale + rice-scale; both handedness signs; tolerances per theory.md §8 Layer 1. | ⬜ | ⬜ |
| 13 | `add-circumnutation-layer3-cross-tier` | Explicit Layer-3 cross-tier consistency tests as a coherent class (`T_nutation ↔ T_psig ±5%`, `λ_spatial ↔ v·T ±10%`). | ⬜ | ⬜ |
| 14 | `add-circumnutation-pipeline` | Promoted ahead of PR #11 (2026-06-11; Tier 4 blocked on #230/Phase 2 — see the PR #11 row). `CircumnutationPipeline` — a **sequential merge-orchestrator** (NOT the per-frame networkx `TraitDef` DAG the stub docstring/this row originally said: the circumnutation tiers are per-track `DataFrame→DataFrame` functions, not per-frame nodes; the documented tier-dependency order is honored as a fixed call order). Composes the 5 tiers into one **46-column** per-plant frame + CSV + 2 sidecars; pickle-able; plate-001 integration test; full row-identity (CC-4). **Dedups the Tier 0/Tier 1 recompute** PR #10 `traveling_wave.compute` does redundant-by-design, via optional precomputed-frames `tier0_df`/`tier1_df` kwargs. Adds the `_NUTATION_TRAIT_UNITS`/`_PSIG_TRAIT_UNITS` maps the units sidecar needs (#222 units-map portion only). **Provenance: `cadence_s`/`R_px`/`run_id` now captured in `run_metadata.json`** (additive `_io.gather_run_metadata` kwargs) so a run is reproducible from its sidecars alone. No new constants (`_CONSTANTS_VERSION` stays 6). [#234](https://github.com/talmolab/sleap-roots/issues/234) · [PR #235](https://github.com/talmolab/sleap-roots/pull/235) | ✅ | ✅ |
| 15 | `add-circumnutation-per-genotype-aggregation` | `aggregate_by_genotype(per_plant_df, units) -> (per_genotype_df, units)` per theory.md §7.7 (median ± IQR; `n_plants_passing_qc`; exclusion reasons). NEW `aggregation` module (impl 10 → 11 by ADDITION). Groups by `(plate_id, genotype, treatment)`, plates not pooled (`dropna=False`); **73-column** output. `principal_axis_angle` dropped (wrapping circular angle); `helix_signed_area_px2` aggregated by magnitude; `growth_axis_unreliable` dropped. Fails loud on a track↔plant 1:1 violation (two-level per-plant collapse deferred to a follow-up). New `_io.write_per_genotype_csv`/`read_per_genotype_csv` + shared `_validate_units_coverage`. No new constants (`_CONSTANTS_VERSION` stays 6). [#236](https://github.com/talmolab/sleap-roots/issues/236) · [PR #240](https://github.com/talmolab/sleap-roots/pull/240) | ✅ | ✅ |
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
