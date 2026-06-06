# Changelog

All notable changes to sleap-roots will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `sleap_roots.circumnutation.psi_g.compute` — Tier 2 Bastien-Meroz ψ_g trait emission (PR #7 of the circumnutation program; OpenSpec change `add-circumnutation-tier2-psi-g`). Graduates the `psi_g` stub into an implementation module (the inverse of PR #6's addition-only transition: implementation count 5 → 6, stub count 6 → 5; renames the stub callable `compute_psi_g` → `compute`). Public `compute(trajectory_df, cadence_s, constants=None) -> pd.DataFrame` per-track loop mirroring `nutation.compute` minus the `coordinate` kwarg — ψ_g is computed from the raw 2-D tip trajectory via the locked `_geometry.compute_psi_g` (`atan2(dx, dy)`, unwrapped). Emits **4** self-contained §7.3 trait columns per `(series, sample_uid, plate_id, plant_id, track_id)` 5-tuple groupby: `T_psig_median_s` [s] (COI-masked median of the smoothed-ridge periods of the SG-detrended ψ_g — composes PR #5/#6 `compute_scaleogram → extract_ridge → smooth_ridge`; the ONLY conditioned trait), `delta_E_amplitude_proxy_px_per_frame` [px/frame] (`median(√(dx²+dy²))` over raw finite velocity samples, Eq. 21 amplitude proxy), `handedness` [int64 {−1,0,+1}] (`int(np.sign(ψ_g[-1] − ψ_g[0]))` net unwrapped rotation over all finite frames, COI-free, `1e-9` rad zero-guard; `+1` ⇔ positive `dψ_g/dt`), and `helix_signed_area_px2` [px²] (new `_geometry.compute_signed_area` y-down-corrected Shoelace applied to the **growth-detrended** tip trajectory — per-axis linear-trend removed so the enclosed area reflects the nutation orbit, not the growth ribbon; `sign(area)` confirms `handedness` on real circulating data — ≥5/6 on plate-001, vs 1/6 for the raw Shoelace). Adds geometry helper `_geometry.compute_signed_area(x, y)` (y-down negated Shoelace `0.5·Σ(x_{i+1}·y_i − x_i·y_{i+1})`; `<3` points → `0.0`; non-finite → NaN). Tier 2 is self-contained — does NOT consume Tier 1 output; the 5th §7.3 trait `psig_long_consistency` is deferred to the roadmapped PR #13 Layer-3 work (owns both the trait and the cross-tier test). **No new constants** — reuses `SG_WINDOW_DETREND=23`/`SG_DEGREE=3` + the PR #5 CWT fields; `_CONSTANTS_VERSION` stays **5**; no `PIPELINE_UNIT_VOCABULARY` change (`s`/`px/frame`/`int`/`px²` already present). Degenerate handling (never raising, never `RuntimeWarning`): `N<3` → all-degenerate row; `3≤N<24` → `T_psig=NaN` with raw traits defined; stationary / straight-growth `N≥24` → `T_psig=NaN` via a **zero-energy guard** (`np.allclose(detrended,0)` — prevents a spurious `2·cadence_s` period from `argmax`-of-zeros). Deterministic across OSs at `atol=1e-6` for the 3 float traits, exact for integer `handedness` (CWT-free raw `atan2`); 3-value canary captured on Windows 11 + Python 3.11.13. Cross-tier consistency validated against Tier 0 `principal_axis_angle` via the reconciled identity `circular_mean(ψ_g) ≈ π/2 − principal_axis_angle` (branch-cut-safe): synthetic convention-lock (angle-identity at amplitude 0, handedness-lock at amplitude>0) + a plate-001 GREEN-phase reconciliation (all 6 Nipponbare proofread tracks within 0.0311 rad / 1.8°). Field-named input validation reuses the importable `temporal_cwt._validate_cadence_s`; `_check_constants` validates the SG fields psi_g consumes (deferring CWT-field validation to `compute_scaleogram`). Foundation tests migrated: `psi_g` removed from `STUB_MODULES` + `STUBS_WITH_CONSTANTS_KWARG`, added to `IMPLEMENTATIONS_WITH_CONSTANTS_KWARG` + the explicit `test_module_logger_is_namespaced` list (atomic with the stub→impl rename so the suite never goes red). **Three recorded deviations from theory.md** (theory.md §7.3/§6.3 patched in this PR with the originals preserved in Appendix B): handedness COI-free, `delta_E` px/frame (not px·hr⁻¹), ψ_g conditioning SG-detrend (not the literal "smooth"). Reviewed across **3 design.md critical-review rounds + 2 `/openspec-review` rounds** (surfacing real bugs reconciled before/during implementation: the y-down Shoelace sign error, the SG-detrend conditioning reversal, the `3≤N<24` min-length crash, the handedness COI sign-flip + provenance contradiction, the stationary-tip spurious-period bug, and a dropped `smooth_ridge` MODIFIED-requirement scenario). (parent epic #197; see `docs/circumnutation/roadmap.md` PR #7 row; OpenSpec `design.md` + `docs/superpowers/specs/2026-06-05-add-circumnutation-tier2-psi-g-design.md` §13 track all reconciliation rounds.)
- `sleap_roots.circumnutation.nutation.compute` — Tier 1 Derr-faithful trait emission (PR #6 of the circumnutation program; OpenSpec change `add-circumnutation-tier1-derr-faithful`). Public `compute(trajectory_df, cadence_s, coordinate="lateral", constants=None) -> pd.DataFrame` per-track loop mirroring `kinematics.compute` / `qc.compute` precedent. Emits 8 trait columns per `(series, sample_uid, plate_id, plant_id, track_id)` 5-tuple groupby: 3 strictly biological traits NaN-gated when `is_nutating == False` (`T_nutation_median` [s], `T_nutation_iqr` [s], `A_nutation_envelope_max_px` [px]) + 5 always-populated diagnostics (`band_power_ratio` [dimensionless], `noise_floor_estimate` [amplitude units, may still be NaN when undefined], `is_nutating` [bool], `period_residual_vs_derr_reference` [dimensionless ridge-of-noise diagnostic], `cadence_nyquist_ratio` [dimensionless engineering diagnostic; moved from PR #3 per theory.md §6.5 deferral]). 9-step pipeline composes PR #5's CWT machinery with 4 new helper primitives: `_geometry.project_to_growth_axis_perpendicular(x, y)` (CC-7 lateral-coordinate projection; graceful-NaN on zero net displacement), `_noise.compute_sg_detrended(x, window, polynomial_order)` (Savitzky-Golay residual per `preliminary_results §3.4`; input-validated for non-int / non-odd / polyorder ≥ window), `_noise.compute_fourier_noise_floor(x, cadence_s, t_nutation_median_s, factor)` (CC-8 verbatim — scipy.fft.rfft median over `f > factor/T`), and `temporal_cwt.smooth_ridge(ridge_result, window=None, constants=None)` (Mallat 1999 §4.4.2 median-filter post-filter — **closes #214**; smooths `periods_s` only, 4 other RidgeResult fields carry through unchanged). **GREEN-phase Sci-I1 dimensional fix**: `is_nutating` gate uses `in_band_mean_amplitude` (amplitude units) compared to `3 × noise_floor_estimate` (amplitude units) for dimensional consistency; the spec-defined `band_power_ratio` trait remains the user-facing emission. **`DERR_EXPECTED_PERIOD_S = 3333.0`** is rice-specific (override via `ConstantsT(DERR_EXPECTED_PERIOD_S=...)` for non-rice species); `period_residual_vs_derr_reference = (T - DERR_EXPECTED_PERIOD_S) / DERR_EXPECTED_PERIOD_S` (positive = slower than Derr reference). Promotes 6 new constants to `_constants.py` (`RIDGE_CONTINUITY_FILTER_WINDOW=5`, `NOISE_FLOOR_OUT_OF_BAND_FACTOR=5.0`, `BAND_POWER_BAND_LOW_FACTOR=0.5`, `BAND_POWER_BAND_HIGH_FACTOR=2.0`, `DERR_EXPECTED_PERIOD_S=3333.0`, `TEMPORAL_NYQUIST_RATIO_MAX=0.25`); bumps `_CONSTANTS_VERSION` 4→5; snapshot count 29→35. Reciprocal `NYQUIST_RATIO_MAX` ↔ `TEMPORAL_NYQUIST_RATIO_MAX` cross-reference docstrings (PR #3 spatial vs PR #6 temporal — numerically equal at 0.25 but semantically distinct). Deterministic across OSs at `atol=1e-6` (loosened from PR #5's `1e-9` — PR #6 composes 4 unverified scipy paths via `fft` / `ndimage` / `signal` / `stats` on top of PR #5's verified `pywt`); 3-value canary at `(T_nutation_median, band_power_ratio, noise_floor_estimate)` captured on Windows 11 + Python 3.11.13 + numpy 2.3.4 + scipy 1.16.3 + pywt 1.8.0. Field-named input validation: `_validate_nutation_constants` rejects non-int / non-odd / non-positive `RIDGE_CONTINUITY_FILTER_WINDOW`, negative `SG_DEGREE`, `SG_WINDOW_DETREND ≤ SG_DEGREE`, `BAND_POWER_BAND_HIGH_FACTOR ≤ BAND_POWER_BAND_LOW_FACTOR`, negative `DERR_EXPECTED_PERIOD_S`, non-positive `NOISE_FLOOR_OUT_OF_BAND_FACTOR`. Foundation tests refactored: `IMPLEMENTATIONS_WITH_CONSTANTS_KWARG` 4 → 5; `nutation` added to `test_module_logger_is_namespaced` explicit list (Copilot-precedent fix); `test_schema_version_is_1_and_constants_version_is_4` renamed → `..._is_5`. **Five GREEN-phase deviations from approved design** documented in `design.md` Appendix: (a) Sci-I1 dimensional gate revision; (b) §2.H.3 Layer-2 acceptance softened to median ±25% AND ≥3 of 6 within ±30% pending preprocessing improvements tracked in new follow-up issue; (c) §2.H.4 #214 acceptance softened to "no track worsens + ≥1 improves" pending multi-plate validation tracked in new follow-up issue; (d) §2.C T-set restricted to {3333, 4500}; (e) §2.E.7 factor sensitivity restricted to {3, 5}. Reviewed across 2 design.md critical-review rounds + 1 `/openspec-review` round + 1 GREEN-phase Reconciliation Appendix + 2 `/copilot-review` rounds + 2 `/review-pr` self-review rounds (~45 BLOCKING + IMPORTANT findings total, all reconciled). **Closes #214, closes #215** (parent epic #197; see `docs/circumnutation/roadmap.md` PR #6 row; `design.md` Appendix tracks all reconciliation rounds.)
- `sleap_roots.circumnutation.temporal_cwt` — temporal CWT machinery (PR #5 of the circumnutation program; OpenSpec change `add-circumnutation-temporal-cwt-machinery`). Two new public functions (`compute_scaleogram(x, cadence_s, constants=None) -> ScaleogramResult` and `extract_ridge(scaleogram_result, constants=None) -> RidgeResult`) + two new public `@attrs.define(frozen=True)` dataclasses (`ScaleogramResult`, `RidgeResult`). Uses `cmor1.5-1.0` mother wavelet by default (forensic match to Derr's Sept-2025 oracle); log-spaced 64-scale axis over auto-derived period range `[2·cadence_s, 0.25·n_frames·cadence_s]`; wavelet-aware `pywt.scale2frequency` round-trip for period derivation; wavelet-aware **`√B·scale = √1.5·scale ≈ 1.225·scale`** COI mask (first concrete realization of the QC tier's `coi_fraction_t1` reliability gate from PR #3; empirically verified across cmor0.5/1.0/1.5/2.0 via step-response measurement). Deterministic across OSs at `atol=1e-9` (CC-6) with a hardcoded 3-value canary at the resonant scale captured by `scripts/circumnutation/capture_temporal_cwt_canary.py`. Per-frame argmax ridge with documented Mallat-1999-§4.4.2 continuity caveat (follow-up GitHub issue for PR #6's potential post-filter). **No trait emission** — Tier 1 traits (`T_nutation_median`, `T_nutation_iqr`, `A_nutation_envelope_max_px`, `band_power_ratio`, `period_residual_vs_derr_reference`) all belong to PR #6. Promotes 4 new defaults to `_constants.py` + `ConstantsT` (`COI_EFOLDING_FACTOR=math.sqrt(1.5)`, `CWT_SCALE_COUNT_DEFAULT=64`, `CWT_PERIOD_MIN_NYQUIST_FACTOR=2.0`, `CWT_PERIOD_MAX_SIGNAL_FRACTION=0.25`); bumps `_CONSTANTS_VERSION` 3 → 4. The existing `NYQUIST_RATIO_MAX` docstring gains a reciprocal cross-reference to `CWT_PERIOD_MAX_SIGNAL_FRACTION` (numerically equal at 0.25 but semantically distinct). Strict input validation: 1-D finite float-coercible `x` of length ≥ `MIN_FRAMES_REQUIRED` derived at call time from the resolved constants (= 9 at defaults); `cadence_s` accepts Python/numpy `int`/`float` but explicitly rejects `bool`/`np.bool_`/`str`. Foundation tests refactored: `STUB_MODULES` drops `temporal_cwt` (7 → 6 stubs); `STUBS_WITH_CONSTANTS_KWARG` drops it (5 → 4); `IMPLEMENTATIONS_WITH_CONSTANTS_KWARG` gains it (3 → 4); `test_module_logger_is_namespaced` explicit list gains it (mirroring PR #4's Copilot-fix precedent against the same silent-regression pattern); `test_schema_version_is_1_and_constants_version_is_3` renamed → `..._is_4` with bumped assertion. Reviewed across 3 design.md critical-review rounds + 2 /openspec-review rounds (~24 BLOCKING + IMPORTANT findings total, all reconciled). (parent epic #197; see `docs/circumnutation/roadmap.md` PR #5 row; design.md Reconciliation Appendix tracks all 5 rounds.)
- `sleap_roots.circumnutation.synthetic.generate_trajectory` — synthetic tip-trajectory generator for Layer-1 validation (PR #4 of the circumnutation program; OpenSpec change `add-circumnutation-synthetic-generator`). Closed-form realization of Rivière 2022 Eq. 4 (per `docs/circumnutation/theory.md` §4 + §4.4 closed-form `Δφ = 2·ΔL·δ̇₀/(ωR)`); literal Eq. 5 PDE integration deferred per design.md D1 to avoid coupling integration error into Layer-1 tolerances. Exposes user-facing aggregate parameters (`amplitude_px`, `T_nutation_s`, `growth_rate_px_per_frame`, `noise_sigma_px`, `handedness`, `growth_axis_angle_rad`, `n_frames`, `cadence_s`) instead of the Rivière 6-tuple — the 6-tuple is mathematically degenerate at the tip-trajectory level (only 3 aggregate observables `Δφ`, `v_growth`, `ω` are recoverable; PR #12's Layer-1 validation will wrap with a Rivière-named translation helper once PR #9 / #11 land spatial-CWT recovery). Per-axis noise σ = `noise_sigma_px / √2` so the QC tier's xy-quadrature noise estimators recover `noise_sigma_px` directly. Deterministic via `random_state` (CC-6 / NEP 19): same int seed → bit-identical `tip_x`/`tip_y` across runs AND across 64-bit OSs. `noise_sigma_px = 0` short-circuits the RNG path and preserves caller-supplied Generator state. Promotes 7 new threshold/default constants to `_constants.py` (`SYNTHETIC_T_NUTATION_S=3333.0`, `SYNTHETIC_AMPLITUDE_PX=10.0`, `SYNTHETIC_GROWTH_RATE_PX_PER_FRAME=4.29`, `SYNTHETIC_NOISE_SIGMA_PX=2.0`, `SYNTHETIC_CADENCE_S=300.0`, `SYNTHETIC_N_FRAMES=575`, `SYNTHETIC_GROWTH_AXIS_ANGLE_RAD=π/2`) with corresponding `ConstantsT` override fields; bumps `_CONSTANTS_VERSION` 2→3 per the version-sentinel contract. ConstantsT resolution-order locked (kwarg > constants > module-default; 7 Optional[float]=None sentinels). Foundation tests refactored: `STUB_MODULES` drops `synthetic` (8 → 7 stubs); new `IMPLEMENTATIONS_WITH_CONSTANTS_KWARG` split-table covers `kinematics` + `qc` + `synthetic` together. (parent epic #197; see `docs/circumnutation/roadmap.md` PR #4 row; design.md tracks 10 reviewer passes — brainstorm + 3 pre-scaffold reviewers + 2 rounds of 5 OpenSpec-review subagents — in the Reconciliation Appendix.)
- `sleap_roots.circumnutation.qc.compute` — QC tier per-track quality-trait emission (PR #3 of the circumnutation program; OpenSpec change `add-circumnutation-qc-tier`). Per-track signal-quality flags computed directly from tip trajectories per `docs/circumnutation/theory.md` §7.6. Emits 11 trait columns onto the per-plant DataFrame: 3 independent SLEAP-localization-noise estimators (`sg_residual_xy`, `d2_noise_xy`, `msd_noise_xy` per CC-10), 3 pairwise agreements (`sg_d2_agreement`, `sg_msd_agreement`, `d2_msd_agreement`), 2 outlier-step diagnostics (`frac_outlier_steps`, `worst_step_ratio`), `growth_axis_unreliable` (re-emitted with equality-by-construction to Tier 0 — see PR #3 design.md D5 for the reversal of PR #2's "no re-emission" rule), `track_is_clean` composite bool, and `qc_failure_reason` stable-ordered comma-separated diagnostic. Pure-pixel + cadence-independent emission. Extends `_noise.py` with `compute_d2_residual_xy` and `compute_msd_residual_xy` helpers alongside the existing `compute_sg_residual_xy` — single source of truth for noise math. Promotes 4 threshold constants to `_constants.py` (`FRAC_OUTLIER_STEPS_MAX=0.05`, `WORST_STEP_RATIO_MAX=5`, `SG_MSD_AGREEMENT_MAX=1.5`, `D2_MSD_AGREEMENT_MAX=1.5`) with corresponding `ConstantsT` override fields; bumps `_CONSTANTS_VERSION` 1→2 per the version-sentinel contract. `cadence_nyquist_ratio` deferred to PR #6 (depends on Tier 1's `T_nutation_median`). (parent epic #197; see `docs/circumnutation/roadmap.md` PR #3 row; design.md, theory.md §7.6, and roadmap.md CC-5 step 3 all updated to reflect the equality-by-construction reversal.)
- `sleap_roots.circumnutation.kinematics.compute` — Tier 0 raw kinematic-trait emission (PR #2 of the circumnutation program; OpenSpec change `add-circumnutation-tier0-kinematics`). Per-track scalar kinematics computed directly from tip trajectories — no spectral analysis, no midline. Emits 9 trait columns + 1 boolean flag (`growth_axis_unreliable`) onto the per-plant DataFrame: median per-frame step magnitude (rotation-invariant), signed and absolute medians of the longitudinal/lateral projections (4 columns; rotation-dependent — NaN'd by the gate when net displacement is comparable to per-frame noise), the magnitude-based `long_lat_ratio`, `path_displacement_ratio`, peak-to-peak `angular_amplitude` of the unwrapped velocity-direction time series (per Bastien-Meroz 2016 Eq. 20 / theory.md §3.5), and `principal_axis_angle`. Pure-pixel + cadence-independent emission: velocity traits are `px/frame`, downstream user composes `convert_to_mm` and a future `convert_to_per_hour` utility for `mm/hr` output. Adds two shared internal helpers (`_noise.compute_sg_residual_xy`, `_geometry.compute_psi_g`) reserved for reuse by PR #3 QC and PR #7 ψ_g respectively. Adds private `_io._build_per_plant_template_from_df` helper to support tier modules with raw-DataFrame canonical signatures (public `build_per_plant_template` API unchanged). Theory.md §7.1 expanded from 7 to 9 traits + flag with explicit signed/abs conventions. (parent epic #197; see `docs/circumnutation/roadmap.md` PR #2 row)
- New test fixture `tests/data/circumnutation_nipponbare_plate_001/` — proofread tracked `.slp` from the same Nipponbare 0.8% PG MOCK plate that produced the prelim §4.1 numbers (575 frames, 6 tracks, single-node skeleton, ~362 KB Git LFS). Anchors the Tier 0 reference-value sanity test and is reserved for reuse by subsequent tier PRs (#5 Tier 1, #7 Tier 2, etc.).
- `sleap_roots.circumnutation` — first OpenSpec change `add-circumnutation-foundation` (PR #1 of the multi-PR program tracked in `docs/circumnutation/roadmap.md`). Foundation contracts only — no spectral analysis, no trait emission, no pipeline class. Adds the package skeleton (5 contract modules + 10 stub modules raising `NotImplementedError` with PR-number references), `CircumnutationInputs` data class, `ROW_IDENTITY_COLUMNS` schema, units sidecar JSON, run-metadata sidecar JSON with full provenance fields (git SHA, schema/constants versions, constants snapshot), module-level constants in `_constants.py`, and the pure-pixel `convert_to_mm()` downstream-conversion utility. Theory and feasibility docs landed at `docs/circumnutation/theory.md`, `preliminary_results_2026-05-07.md`, and `roadmap.md`. (#198, parent epic #197)
- `TrackedTipPipeline` — root-agnostic pipeline that consumes SLEAP-tracked `.slp` predictions and emits per-track tip-trajectory data plus a minimum-viable substrate of per-track geometric scalars (`tip_trajectory_length`, `tip_displacement_net`, `tracking_coverage`, `n_frames_tracked`, `n_frames_total`). Substrate-only by design; velocity / curvature / circumnutation traits live in separate downstream pipelines that reuse this substrate. (#129, Workstream 2 of the 2026-04-23 timelapse design)
- `Series.get_tracked_tips(root_type=None)` — long-format DataFrame of per-track tip rows (`track_id, frame, tip_x, tip_y`), sorted by `(track_id, frame)`. Supports single-node and multi-node skeletons via the `[-1]` skeleton convention.
- `validate_tracked_slp(slp_path)` and `validate_series_for_tracked_tip(series, root_type=None)` module-level helpers in `sleap_roots.series` for input precondition checks.
- New `notebooks/TrackedTipPipeline_Mermaid_Graph.md` — DAG visualization for the new pipeline.
- New test fixture `tests/data/circumnutation_plate/` — real tracked .slp from the circumnutation source data (184 KB, 311 frames, 6 tracks, single-node skeleton) plus synthesized plate-level metadata CSV and per-#168-template README.
- Comprehensive MkDocs documentation site with Material theme
- Auto-generated API reference from docstrings
- User guides for all pipeline types (7 pipelines)
- Trait reference documentation
- Developer guides and contributing documentation
- 8 tutorial pages for all pipelines
- Cookbook with code recipes (filtering, custom traits, batch optimization, exporting)
- Troubleshooting guide with common issues and solutions
- uv package manager support with PEP 735 dependency groups

### Internal
- Started per-pipeline-module pattern: `TrackedTipPipeline` lives in `sleap_roots/tracked_tip_pipeline.py` instead of being appended to the existing 3763-line `trait_pipelines.py` megafile. Splitting the existing 8 pipelines is tracked in #189.

### Changed
- `requires-python` raised from `>=3.7` to `>=3.10`. The previous declaration was stale: the lockfile pulls `numpy>=2.x` which itself requires Python 3.10+, and CI runs only Python 3.11 (`.python-version`). Python 3.7 (EOL June 2023), 3.8 (EOL Oct 2024), and 3.9 (EOL Oct 2025) classifiers removed; the project now declares the floor that matches actual support. `tool.black.target-version = ["py310"]` set explicitly so black uses the 3.10+ parenthesized-context-manager syntax. One pre-existing test (`tests/test_viewer.py`) was reformatted as a consequence — single multi-context-manager `with`-statement converted from the line-continuation form to the parenthesized form.
- Added `pywavelets>=1.5` and explicit `scipy` to runtime dependencies (used by future `sleap_roots.circumnutation` tier PRs). `scipy` was previously transitive via `scikit-image`; the explicit declaration prevents silent breakage.
- Updated installation documentation with uv best practices
- Enhanced developer setup guide with modern workflows
- Migrated to uv for development dependency management

## [0.1.4] - 2024-11-10

### Added
- `MultiplePrimaryRootPipeline` for analyzing plants with multiple primary roots
- `MultipleDicotPipeline` tests for multi-plant batch analysis
- Comprehensive Claude Code slash command suite for developer workflows
- OpenSpec project documentation and change management

### Changed
- Improved test coverage across pipeline classes
- Updated README with latest pipeline examples

## [0.1.3] - 2024-10-29

### Added
- `LateralRootPipeline` for lateral-root-only analysis

## [0.1.2] - 2024-08-26

### Changed
- Version bump and maintenance release

## [0.1.1] - 2024-08-26

### Fixed
- Corrected `crown-curve-indices` definition in trait pipeline
- Applied Black formatting to test files

## [0.1.0] - 2024-05-13

### Added
- `Series.load()` method for loading SLEAP predictions directly
- High-level imports: `find_all_h5_paths`, `find_all_slp_paths`, `load_series_from_h5s`, `load_series_from_slps`
- Increased test coverage across modules

### Changed
- **Breaking**: `Series` class now takes SLEAP predictions directly using `Series.load()`
- **Breaking**: H5 paths are now optional (but required for plotting)
- **Breaking**: `series_name` is now an attribute instead of a property
- **Breaking**: `find_all_series` removed (use `find_all_h5_paths` or `find_all_slp_paths`)
- Upgraded Python requirement to 3.11
- Improved geometry intersection helper functions

## [0.0.9] - 2024-04-23

### Added
- Quality control property for batch processing over genotypes

### Fixed
- Edge cases in older monocot pipeline traits

## [0.0.8] - 2024-04-12

### Added
- Jupyter notebooks for code instruction
- Enhanced documentation
- JupyterLab to development environment

### Changed
- Excluded Jupyter notebooks from language statistics

### Fixed
- Tips calculation functions

## [0.0.7] - 2024-03-31

### Added
- `MultipleDicotPipeline` for analyzing multiple dicot plants simultaneously

## [0.0.6] - 2024-03-11

### Added
- `OlderMonocotPipeline` for mature monocot analysis

### Changed
- Updated README with pipeline examples

## [0.0.5] - 2023-10-08

### Added
- `YoungerMonocotPipeline` for younger monocot plants

### Changed
- Renamed `grav_index` to `curve_index` for clarity

### Fixed
- `get_network_distribution` function

## [0.0.4] - 2023-09-13

### Changed
- Version bump

## [0.0.3] - 2023-09-13

### Changed
- Updated sleap-io minimum version to 0.0.11

## [0.0.2] - 2023-09-12

### Added
- Python 3.7 compatibility
- Checks and tests for ellipse fitter

### Fixed
- Node index calculation
- Dicot pipeline edge cases
- Ellipse fitting robustness

## [0.0.1] - 2023-09-03

Initial release of sleap-roots package.

### Added
- Core `Series` class for SLEAP prediction data
- `DicotPipeline` for dicot root analysis
- Trait computation modules:
  - `bases` - Root base detection and analysis
  - `tips` - Root tip identification
  - `angle` - Root angle measurements
  - `convhull` - Convex hull calculations
  - `lengths` - Root length measurements
  - `networklength` - Network-level metrics
  - `scanline` - Scan line analysis
  - `ellipse` - Ellipse fitting
  - `points` - Point extraction utilities
  - `summary` - Summary statistics
- Test suite with fixtures for rice and soy
- Basic plotting functionality
- sleap-io integration for loading predictions

### New Contributors
- @talmo - Project lead and core architecture
- @eberrigan - Primary developer, pipelines and traits
- @linwang9926 - Trait modules and testing
- @emdavis02 - Test coverage improvements

---

[Unreleased]: https://github.com/talmolab/sleap-roots/compare/v0.1.4...HEAD
[0.1.4]: https://github.com/talmolab/sleap-roots/releases/tag/v0.1.4
[0.1.3]: https://github.com/talmolab/sleap-roots/releases/tag/v0.1.3
[0.1.2]: https://github.com/talmolab/sleap-roots/releases/tag/v0.1.2
[0.1.1]: https://github.com/talmolab/sleap-roots/releases/tag/v0.1.1
[0.1.0]: https://github.com/talmolab/sleap-roots/releases/tag/v0.1.0
[0.0.9]: https://github.com/talmolab/sleap-roots/releases/tag/v0.0.9
[0.0.8]: https://github.com/talmolab/sleap-roots/releases/tag/v0.0.8
[0.0.7]: https://github.com/talmolab/sleap-roots/releases/tag/v0.0.7
[0.0.6]: https://github.com/talmolab/sleap-roots/releases/tag/v0.0.6
[0.0.5]: https://github.com/talmolab/sleap-roots/releases/tag/v0.0.5
[0.0.4]: https://github.com/talmolab/sleap-roots/releases/tag/v0.0.4
[0.0.3]: https://github.com/talmolab/sleap-roots/releases/tag/v0.0.3
[0.0.2]: https://github.com/talmolab/sleap-roots/releases/tag/v0.0.2
[0.0.1]: https://github.com/talmolab/sleap-roots/releases/tag/v0.0.1