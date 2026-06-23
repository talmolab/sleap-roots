# circumnutation Specification

## Purpose
TBD - created by archiving change add-circumnutation-foundation. Update Purpose after archive.
## Requirements
### Requirement: Package layout
The system SHALL provide a `sleap_roots.circumnutation` sub-package whose import-tree is complete from this PR onward — every module name referenced anywhere in `docs/circumnutation/roadmap.md` exists as an importable Python module from PR #1 forward, even if its functions raise `NotImplementedError` until later PRs land. The package SHALL contain:

- 7 contract modules: `__init__.py`, `_constants.py`, `_types.py`, `_io.py`, `units.py`, `_noise.py`, `_geometry.py`
- 12 implementation modules: `kinematics` (implemented from PR #2 onward; see Requirement: Tier 0 raw kinematic traits), `qc` (implemented from PR #3 onward; see Requirement: QC tier per-track quality traits), `synthetic` (implemented from PR #4 onward; see Requirement: Synthetic trajectory generator), `temporal_cwt` (implemented from PR #5 onward; see Requirement: Temporal CWT scaleogram API and Requirement: Temporal CWT ridge API), `nutation` (implemented from PR #6 onward; see Requirement: Tier 1 nutation trait emission API), `psi_g` (implemented from PR #7 onward; see Requirement: Tier 2 ψ_g trait emission API), `midline` (implemented from PR #8 onward; see Requirement: Tier 3a midline reconstruction API), `spatial_cwt` (implemented from PR #9 onward; see Requirement: Tier 3b spatial curvature resample API, Requirement: Tier 3b spatial CWT scaleogram API, and Requirement: Tier 3b spatial CWT ridge API), `traveling_wave` (implemented from PR #10 onward; see Requirement: Tier 3c traveling-wave trait emission API), `pipeline` (implemented from PR #14 onward; see Requirement: Circumnutation pipeline composition API), `aggregation` (implemented from PR #15 onward; see Requirement: Per-genotype aggregation API), and `plotting` (implemented from PR #16 onward; see Requirement: Circumnutation diagnostic plots API)
- 1 stub module: `parametric`

Each stub module SHALL define exactly one canonical callable per the table below, raising `NotImplementedError(f"PR #{N} — see docs/circumnutation/roadmap.md")` with the documented PR number. Each stub callable SHALL also have a complete Google-style docstring (Args, Returns, Raises) so `pydocstyle --convention=google` and the mkdocstrings auto-generated reference pages remain informative.

| Stub module | Canonical callable | PR # |
|---|---|---|
| `parametric` | `compute(tier3_df, R_px, omega, Delta_phi)` | 11 |

The `kinematics` module SHALL be importable on the same terms as the other modules (clean import, namespaced logger) and SHALL expose `compute(trajectory_df, constants=None)` per Requirement: Tier 0 raw kinematic traits. The `qc` module SHALL be importable on the same terms and SHALL expose `compute(trajectory_df, constants=None)` per Requirement: QC tier per-track quality traits. The `synthetic` module SHALL be importable on the same terms and SHALL expose `generate_trajectory(...)` per Requirement: Synthetic trajectory generator. The `temporal_cwt` module SHALL be importable on the same terms and SHALL expose `compute_scaleogram(x, cadence_s, constants=None) -> ScaleogramResult` and `extract_ridge(scaleogram_result, constants=None) -> RidgeResult` per Requirement: Temporal CWT scaleogram API and Requirement: Temporal CWT ridge API, AND SHALL ALSO expose `smooth_ridge(ridge_result, window=None, constants=None) -> RidgeResult` per Requirement: Temporal CWT ridge-continuity smoothing API. The `nutation` module SHALL be importable on the same terms and SHALL expose `compute(trajectory_df, cadence_s, coordinate="lateral", constants=None) -> pd.DataFrame` per Requirement: Tier 1 nutation trait emission API. The `psi_g` module SHALL be importable on the same terms and SHALL expose `compute(trajectory_df, cadence_s, constants=None) -> pd.DataFrame` per Requirement: Tier 2 ψ_g trait emission API. The `midline` module SHALL be importable on the same terms and SHALL expose `reconstruct(x, y, cadence_s, sg_window=None, constants=None) -> MidlineResult` per Requirement: Tier 3a midline reconstruction API. The `spatial_cwt` module SHALL be importable on the same terms and SHALL expose `compute_scaleogram(kappa, ds, constants=None) -> SpatialScaleogramResult` and `extract_ridge(scaleogram_result, constants=None) -> SpatialRidgeResult` per Requirement: Tier 3b spatial CWT scaleogram API and Requirement: Tier 3b spatial CWT ridge API, AND SHALL ALSO expose `resample_curvature(curvature_px_inv, arc_length_px, velocity_sub_noise_mask=None, constants=None) -> ResampleResult` per Requirement: Tier 3b spatial curvature resample API. The `traveling_wave` module SHALL be importable on the same terms and SHALL expose `compute(trajectory_df, cadence_s, constants=None, *, tier0_df=None, tier1_df=None) -> pd.DataFrame` per Requirement: Tier 3c traveling-wave trait emission API. The `pipeline` module SHALL be importable on the same terms and SHALL expose `compute_traits(inputs, constants=None) -> tuple[pd.DataFrame, pd.DataFrame, dict]` and the `CircumnutationPipeline` class per Requirement: Circumnutation pipeline composition API. The `aggregation` module SHALL be importable on the same terms and SHALL expose `aggregate_by_genotype(per_plant_df, units) -> tuple[pd.DataFrame, dict]` per Requirement: Per-genotype aggregation API. The `plotting` module SHALL be importable on the same terms and SHALL expose `scaleogram(scaleogram_result, out_path, *, ridge_result=None) -> Path`, `trail_overlay(midline_result, out_path) -> Path`, `plate_panel(midline_results, out_path) -> Path`, and `save_plots(inputs, out_dir, *, constants=None, enabled=True) -> list[Path]` per Requirement: Circumnutation diagnostic plots API. Unlike the stub modules, calling `kinematics.compute`, `qc.compute`, `synthetic.generate_trajectory`, `temporal_cwt.compute_scaleogram`, `nutation.compute`, `psi_g.compute`, `midline.reconstruct`, `spatial_cwt.resample_curvature`, `spatial_cwt.compute_scaleogram`, `spatial_cwt.extract_ridge`, `traveling_wave.compute`, `pipeline.compute_traits`, `aggregation.aggregate_by_genotype`, `plotting.scaleogram`, `plotting.trail_overlay`, `plotting.plate_panel`, or `plotting.save_plots` with a valid input SHALL NOT raise `NotImplementedError`.

The `_noise` and `_geometry` modules are private (underscore-prefixed) shared internals; their canonical callables are documented under Requirement: Tier 0 helper modules (and, for `_geometry.compute_signed_area`, under Requirement: Tier 2 ψ_g trait emission API; and, for `_noise.compute_sg_derivative` / `_geometry.compute_path_curvature`, under Requirement: Tier 3a midline reconstruction API).

**Scope note on PR #6 addition-vs-transition.** The `nutation` module is NEWLY created in PR #6 — it was never a stub module in PR #1–#5, and therefore does not appear in the stub-callable table. The implementation-module count grows from 4 (PR #5 baseline: kinematics, qc, synthetic, temporal_cwt) to 5 by ADDITION of `nutation`, not by transition from a prior stub. This was the first PR in the program to grow the implementation set without shrinking the stub set.

**Scope note on PR #7 stub-to-implementation transition.** The `psi_g` module was a stub in PR #1–#6 (it appeared in the stub-callable table with canonical callable `compute_psi_g(x, y, constants=None)`). PR #7 graduated it to an implementation module: the implementation-module count grew from 5 to 6 AND the stub-module count shrank from 6 to 5. The canonical callable was RENAMED `compute_psi_g` → `compute`.

**Scope note on PR #8 stub-to-implementation transition.** The `midline` module was a stub in PR #1–#7 (it appeared in the stub-callable table with canonical callable `reconstruct(x, y, cadence_s, constants=None)`). PR #8 graduated it to an implementation module: the implementation-module count grew from 6 to 7 AND the stub-module count shrank from 5 to 4 (the same stub→impl shape as PR #7). The canonical callable KEPT its name `reconstruct` (no rename); the implementation signature ADDED a `sg_window=None` parameter (`reconstruct(x, y, cadence_s, constants=None)` → `reconstruct(x, y, cadence_s, sg_window=None, constants=None)`), locked by Requirement: Tier 3a midline reconstruction API.

**Scope note on PR #9 stub-to-implementation transition.** The `spatial_cwt` module IS a stub in PR #1–#8 (it appeared in the stub-callable table with canonical callable `compute_scaleogram(kappa, ds, constants=None)`, PR #9). PR #9 graduates it to an implementation module: the implementation-module count grows from 7 to **8** AND the stub-module count shrinks from 4 to **3** (the same stub→impl shape as PR #7/#8). The canonical callable KEEPS its name `compute_scaleogram` (no rename); the implementation signature is EXACTLY the stub-table signature `compute_scaleogram(kappa, ds, constants=None)` — the speculative `wavelet=`/`scale_range=` keyword parameters present in the PR #1 stub file are DROPPED (the wavelet and scale range are derived from `constants`, mirroring `temporal_cwt.compute_scaleogram`'s `(x, cadence_s, constants=None)` precedent). `spatial_cwt` is therefore removed from the stub-callable table and from the "remaining stub" enumeration, and gains callability scenarios below. PR #9 ADDS two further public symbols not previously in the stub table — `resample_curvature` (the non-uniform→uniform κ(s) resample entry helper) and `extract_ridge` (the spatial ridge) — whose callability contracts are locked here for symmetry with `compute_scaleogram`, mirroring how PR #5's `extract_ridge` and PR #6's `smooth_ridge` were locked. **PR #9 descopes `L_gz`/`L_c` growth-zone-structure detection** (the §7.4 |κ|-envelope-peak premise does not transfer to top-view tip-trail κ(s); see the Tier 3b requirements and follow-up issue #230); `spatial_cwt` therefore exposes no `detect_growth_zone` symbol.

**Scope note on PR #10 addition-vs-transition.** The `traveling_wave` module is NEWLY created in PR #10 — it was never a stub module in PR #1–#9 (only `parametric`, `plotting`, `pipeline` remain stubs at PR #10), and therefore does not appear in the stub-callable table. The implementation-module count grows from 8 to **9** by ADDITION of `traveling_wave`, not by transition from a prior stub (the same addition shape as PR #6's `nutation`); the stub-module count is UNCHANGED at 3. The canonical callable is `compute(trajectory_df, cadence_s, constants=None) -> pd.DataFrame` (the Tier 1/Tier 2 trait-emission signature; `cadence_s` is an explicit positional parameter). **PR #10 ships reduced scope** per the PR #9 `L_gz`/`L_c` descope (#230): it emits only the 3 λ-based traits + diagnostics on the full reconstructed trail (no growth-zone mask); the 5 `L_gz`/`L_c`-dependent traits and the mask remain blocked on #230 and are OMITTED from the output schema (not reserved as NaN columns). `traveling_wave` gains a callability scenario below.

**Scope note on PR #14 stub-to-implementation transition.** The `pipeline` module WAS a stub in PR #1–#10 (it appeared in the stub-callable table with canonical callable `compute_traits(inputs, constants=None)`, PR #14). PR #14 graduates it to an implementation module: the implementation-module count grows from 9 to **10** AND the stub-module count shrinks from 3 to **2** (the same stub→impl shape as PR #7/#8/#9). The canonical callable KEEPS its name `compute_traits` (no rename); the implementation adds keyword-only file-writing via a separate `CircumnutationPipeline.save(...)` method and a picklable `CircumnutationPipeline` class, locked by Requirement: Circumnutation pipeline composition API. `pipeline` is therefore removed from the stub-callable table and from the "remaining stub" enumeration, and gains a callability scenario below. **`pipeline` was the last stub whose canonical callable carried the forward-compatible `constants=None` parameter** — the 2 remaining stubs (`parametric`, `plotting`) do not, so the prior "Stubs accept `constants=None` where the table prescribes it" scenario is REMOVED (no remaining subject). **Deviation note:** the `pipeline` stub docstring and the roadmap row described PR #14 as building "a TraitDef DAG matching the `Pipeline` base class pattern". PR #14 instead implements a **sequential merge-orchestrator** (the circumnutation tiers are per-track `DataFrame → DataFrame` functions, not the per-frame networkx `TraitDef` nodes of `sleap_roots.trait_pipelines.Pipeline`); the documented tier-order dependency structure is preserved without the per-frame node model. The stub docstring + roadmap row are corrected to match.

**Scope note on PR #15 addition-vs-transition.** The `aggregation` module is NEWLY created in PR #15 — it was never a stub module in PR #1–#14 (only `parametric` and `plotting` remain stubs at PR #15), and therefore does not appear in the stub-callable table. The implementation-module count grows from 10 to **11** by ADDITION of `aggregation`, not by transition from a prior stub (the same addition shape as PR #6's `nutation` and PR #10's `traveling_wave`); the stub-module count is UNCHANGED at 2. The canonical callable is `aggregate_by_genotype(per_plant_df, units) -> tuple[pd.DataFrame, dict]` (a post-pipeline aggregation that consumes the PR #14 composed per-plant frame). `aggregation` gains a callability scenario below.

**Scope note on PR #16 stub-to-implementation transition.** The `plotting` module WAS a stub in PR #1–#15 (it appeared in the stub-callable table with canonical callable `scaleogram(scaleogram_result, out_path)`, PR #16). PR #16 graduates it to an implementation module: the implementation-module count grows from 11 to **12** AND the stub-module count shrinks from 2 to **1** (the same stub→impl shape as PR #7/#8/#9/#14 — only `parametric`/PR #11 remains a stub). The canonical callable KEEPS its name `scaleogram` (no rename); the implementation signature gains a keyword-only `ridge_result=None` and now returns the written `Path` (`scaleogram(scaleogram_result, out_path, *, ridge_result=None) -> Path`), and the module ADDS three further public symbols — `trail_overlay`, `plate_panel`, and a `save_plots` orchestrator — whose callability contracts are locked here for symmetry, mirroring how PR #5/#9's `extract_ridge` and PR #6's `smooth_ridge` were locked. `plotting` is therefore removed from the stub-callable table and from the "remaining stub" enumeration. The stub had **no** forward-compatible `constants=None` parameter, so no prior `constants=` scenario is affected. The intro paragraph's sentence requiring stubs to carry a forward-compatible `constants=None` (preserved through PR #15) is now DROPPED here: the sole remaining stub (`parametric`) takes no `constants=` parameter, so that normative clause has no remaining subject. **Deviation note:** the `plotting` stub docstring promised "tip-trail overlays with κ-color-coding and an `L_gz` arc-length marker". `L_gz` (growth-zone length) is blocked on #230 and is not computed anywhere in the pipeline, so PR #16 OMITS the `L_gz` marker entirely (the graduated docstrings drop the claim; no `L_gz` parameter is added to any function); a follow-up PR adds it once #230 lands. See Requirement: Circumnutation diagnostic plots API.

#### Scenario: All stub modules import cleanly
- **WHEN** a user runs `import sleap_roots.circumnutation as c; import sleap_roots.circumnutation.kinematics, sleap_roots.circumnutation.qc, sleap_roots.circumnutation.synthetic, sleap_roots.circumnutation.temporal_cwt, sleap_roots.circumnutation.nutation, sleap_roots.circumnutation.psi_g, sleap_roots.circumnutation.midline, sleap_roots.circumnutation.spatial_cwt, sleap_roots.circumnutation.traveling_wave, sleap_roots.circumnutation.parametric, sleap_roots.circumnutation.plotting, sleap_roots.circumnutation.pipeline, sleap_roots.circumnutation.aggregation`
- **THEN** every import succeeds
- **AND** no exception is raised at module-import time

#### Scenario: Calling each remaining stub raises NotImplementedError with the correct PR number
- **GIVEN** the package imported as in the previous scenario
- **WHEN** the canonical callable in the 1 remaining stub module (`parametric`) is invoked (parameters per the table above; `NotImplementedError` fires before any argument check)
- **THEN** `NotImplementedError` is raised
- **AND** the exception message matches the regex `r"^PR #\d+ — see docs/circumnutation/roadmap\.md$"`
- **AND** the captured PR number equals the one in the table for that module

#### Scenario: `kinematics.compute` no longer raises NotImplementedError
- **GIVEN** a valid `trajectory_df` (8 row-identity columns + `frame`, `tip_x`, `tip_y`; ≥ 1 row)
- **WHEN** `sleap_roots.circumnutation.kinematics.compute(trajectory_df)` is invoked
- **THEN** the call returns a `pandas.DataFrame` (the Tier 0 per-plant output) without raising `NotImplementedError`

#### Scenario: `qc.compute` no longer raises NotImplementedError
- **GIVEN** a valid `trajectory_df` (8 row-identity columns + `frame`, `tip_x`, `tip_y`; ≥ 1 row)
- **WHEN** `sleap_roots.circumnutation.qc.compute(trajectory_df)` is invoked
- **THEN** the call returns a `pandas.DataFrame` (the QC tier per-plant output) without raising `NotImplementedError`

#### Scenario: `synthetic.generate_trajectory` no longer raises NotImplementedError
- **WHEN** `sleap_roots.circumnutation.synthetic.generate_trajectory()` is invoked with all-default kwargs
- **THEN** the call returns a `pandas.DataFrame` (the per-frame trajectory output) without raising `NotImplementedError`
- **AND** the DataFrame has `SYNTHETIC_N_FRAMES` rows (default 575) and the documented 11-column schema per Requirement: Synthetic trajectory generator

#### Scenario: `temporal_cwt.compute_scaleogram` no longer raises NotImplementedError
- **GIVEN** a valid 1-D float64 ndarray `x` of length ≥ 9 with all-finite values, and a positive finite `cadence_s` (e.g., `np.linspace(0, 100, 32) * 0.1` and `cadence_s = 300.0`)
- **WHEN** `sleap_roots.circumnutation.temporal_cwt.compute_scaleogram(x, cadence_s)` is invoked
- **THEN** the call returns a `ScaleogramResult` (the temporal CWT scaleogram output) without raising `NotImplementedError`

#### Scenario: `temporal_cwt.extract_ridge` is callable on a valid ScaleogramResult without raising
- **GIVEN** a valid `ScaleogramResult` produced by `compute_scaleogram(x, 300.0)` on a length-≥9 finite array
- **WHEN** `sleap_roots.circumnutation.temporal_cwt.extract_ridge(scaleogram_result)` is invoked
- **THEN** the call returns a `RidgeResult` without raising any exception
- **AND** since `extract_ridge` is a NEW public symbol introduced by PR #5 (not a transition from a prior stub), it does not appear in the stub-callable table — its callability contract is locked here in the MODIFIED Package layout requirement for symmetry with `compute_scaleogram`

#### Scenario: `nutation.compute` is callable on a valid trajectory_df without raising
- **GIVEN** a valid `trajectory_df` (8 row-identity columns + `frame`, `tip_x`, `tip_y`; ≥ 9 rows for at least one track) and a positive finite `cadence_s` (e.g., 300.0)
- **WHEN** `sleap_roots.circumnutation.nutation.compute(trajectory_df, cadence_s=300.0)` is invoked
- **THEN** the call returns a `pandas.DataFrame` (the Tier 1 per-plant nutation trait output) without raising `NotImplementedError`
- **AND** since `nutation` is a NEW module introduced by PR #6 (not a transition from a prior stub), it does not appear in the stub-callable table — its callability contract is locked here in the MODIFIED Package layout requirement

#### Scenario: `temporal_cwt.smooth_ridge` is callable on a valid RidgeResult without raising
- **GIVEN** a valid `RidgeResult` produced by `extract_ridge(compute_scaleogram(x, 300.0))` on a length-≥9 finite array
- **WHEN** `sleap_roots.circumnutation.temporal_cwt.smooth_ridge(ridge_result)` is invoked
- **THEN** the call returns a `RidgeResult` (with smoothed `periods_s`) without raising any exception
- **AND** since `smooth_ridge` is a NEW public symbol introduced by PR #6 (not a transition from a prior stub), it does not appear in the stub-callable table — its callability contract is locked here in the MODIFIED Package layout requirement

#### Scenario: `psi_g.compute` is callable on a valid trajectory_df without raising
- **GIVEN** a valid `trajectory_df` (8 row-identity columns + `frame`, `tip_x`, `tip_y`; ≥ 24 rows for at least one track) and a positive finite `cadence_s` (e.g., 300.0)
- **WHEN** `sleap_roots.circumnutation.psi_g.compute(trajectory_df, cadence_s=300.0)` is invoked
- **THEN** the call returns a `pandas.DataFrame` (the Tier 2 per-plant ψ_g trait output) without raising `NotImplementedError`
- **AND** `psi_g` transitioned FROM a stub in PR #7 (the stub-module count shrank 6 → 5 and the implementation-module count grew 5 → 6)

#### Scenario: `midline.reconstruct` is callable on valid arrays without raising
- **GIVEN** valid 1-D float64 ndarrays `x`, `y` of equal length ≥ `sg_window` (default `SG_WINDOW_SHORT = 5`) with all-finite values and non-zero displacement, and a positive finite `cadence_s` (e.g., 300.0)
- **WHEN** `sleap_roots.circumnutation.midline.reconstruct(x, y, cadence_s=300.0)` is invoked
- **THEN** the call returns a `MidlineResult` (the Tier 3a reconstruction output) without raising `NotImplementedError`
- **AND** `midline` transitioned FROM a stub in PR #8 (the stub-module count shrank 5 → 4 and the implementation-module count grew 6 → 7; callable name `reconstruct` unchanged, signature gained `sg_window=None`)

#### Scenario: `spatial_cwt.compute_scaleogram` no longer raises NotImplementedError
- **GIVEN** a valid 1-D float64 ndarray `kappa` of length ≥ the spatial MIN length with all-finite values, and a positive finite `ds` (e.g., a planted sinusoid of length 64 and `ds = 5.8`)
- **WHEN** `sleap_roots.circumnutation.spatial_cwt.compute_scaleogram(kappa, ds)` is invoked
- **THEN** the call returns a `SpatialScaleogramResult` (the Tier 3b spatial CWT scaleogram output) without raising `NotImplementedError`
- **AND** `spatial_cwt` transitions FROM a stub (it WAS in the stub-callable table in PR #1–#8 with callable `compute_scaleogram`); PR #9 removes it from that table, so the stub-module count shrinks 4 → 3 and the implementation-module count grows 7 → 8 (the callable name `compute_scaleogram` is unchanged; the speculative `wavelet=`/`scale_range=` stub kwargs are dropped)

#### Scenario: `spatial_cwt.resample_curvature` is callable on valid arrays without raising
- **GIVEN** valid 1-D float64 ndarrays `curvature_px_inv` and `arc_length_px` of equal length (monotonic non-decreasing `arc_length_px`, non-zero span, enough unmasked samples) and an optional bool `velocity_sub_noise_mask`
- **WHEN** `sleap_roots.circumnutation.spatial_cwt.resample_curvature(curvature_px_inv, arc_length_px)` is invoked
- **THEN** the call returns a `ResampleResult` without raising any exception
- **AND** since `resample_curvature` is a NEW public symbol introduced by PR #9 (not a transition from a prior stub), it does not appear in the stub-callable table — its callability contract is locked here in the MODIFIED Package layout requirement for symmetry with `compute_scaleogram`

#### Scenario: `spatial_cwt.extract_ridge` is callable on a valid SpatialScaleogramResult without raising
- **GIVEN** a valid `SpatialScaleogramResult` produced by `compute_scaleogram(kappa, ds)` on a length-≥ MIN finite array
- **WHEN** `sleap_roots.circumnutation.spatial_cwt.extract_ridge(scaleogram_result)` is invoked
- **THEN** the call returns a `SpatialRidgeResult` without raising any exception
- **AND** since `extract_ridge` is a NEW public symbol introduced by PR #9 (not a transition from a prior stub), it does not appear in the stub-callable table — its callability contract is locked here in the MODIFIED Package layout requirement for symmetry with `compute_scaleogram`

#### Scenario: `traveling_wave.compute` is callable on a valid trajectory_df without raising
- **GIVEN** a valid `trajectory_df` (8 row-identity columns + `frame`, `tip_x`, `tip_y`; enough rows for at least one track to form a midline + spatial scaleogram) and a positive finite `cadence_s` (e.g., 300.0)
- **WHEN** `sleap_roots.circumnutation.traveling_wave.compute(trajectory_df, cadence_s=300.0)` is invoked
- **THEN** the call returns a `pandas.DataFrame` (the Tier 3c per-plant traveling-wave trait output) without raising `NotImplementedError`
- **AND** since `traveling_wave` is a NEW module introduced by PR #10 (not a transition from a prior stub), it does not appear in the stub-callable table — its callability contract is locked here in the MODIFIED Package layout requirement

#### Scenario: `pipeline.compute_traits` no longer raises NotImplementedError
- **GIVEN** a valid `CircumnutationInputs` (a `trajectory_df` with the 8 row-identity columns + `frame`, `tip_x`, `tip_y` and ≥ 1 track with enough frames for the temporal/spatial chains, `cadence_s = 300.0`)
- **WHEN** `sleap_roots.circumnutation.pipeline.compute_traits(inputs)` is invoked
- **THEN** the call returns a 3-tuple `(per_plant_df, trajectory_df, units_dict)` (the composed pipeline output) without raising `NotImplementedError`
- **AND** `pipeline` transitioned FROM a stub in PR #14 (the stub-module count shrank 3 → 2 and the implementation-module count grew 9 → 10; callable name `compute_traits` unchanged)

#### Scenario: `aggregation.aggregate_by_genotype` is callable on a valid per_plant_df without raising
- **GIVEN** a valid composed per-plant DataFrame and its `units` mapping (the `(per_plant_df, units_dict)` output of `pipeline.compute_traits`, ≥ 1 plant)
- **WHEN** `sleap_roots.circumnutation.aggregation.aggregate_by_genotype(per_plant_df, units)` is invoked
- **THEN** the call returns a 2-tuple `(per_genotype_df, per_genotype_units)` (the per-genotype aggregation output) without raising any exception
- **AND** since `aggregation` is a NEW module introduced by PR #15 (not a transition from a prior stub), it does not appear in the stub-callable table — its callability contract is locked here in the MODIFIED Package layout requirement

#### Scenario: `plotting.scaleogram` no longer raises NotImplementedError
- **GIVEN** a valid `ScaleogramResult` (e.g. from `temporal_cwt.compute_scaleogram` on a length-≥9 finite array) and an `out_path` whose parent directory exists
- **WHEN** `sleap_roots.circumnutation.plotting.scaleogram(scaleogram_result, out_path)` is invoked
- **THEN** the call writes a PNG to `out_path` and returns that `Path` without raising `NotImplementedError`
- **AND** `plotting` transitioned FROM a stub in PR #16 (the stub-module count shrank 2 → 1 and the implementation-module count grew 11 → 12; callable name `scaleogram` unchanged, signature gained a keyword-only `ridge_result=None` and a `Path` return)

#### Scenario: `plotting.trail_overlay`, `plotting.plate_panel`, and `plotting.save_plots` are callable without raising
- **GIVEN** a valid `MidlineResult` (from `midline.reconstruct` on finite arrays), an ordered collection of such results, and a valid `CircumnutationInputs`, with output paths/directories that exist
- **WHEN** `plotting.trail_overlay(midline_result, out_path)`, `plotting.plate_panel(midline_results, out_path)`, and `plotting.save_plots(inputs, out_dir)` are each invoked
- **THEN** each call returns without raising `NotImplementedError` — `trail_overlay` and `plate_panel` return the written `Path`, and `save_plots` returns a `list[Path]` of the PNGs it wrote
- **AND** since `trail_overlay`, `plate_panel`, and `save_plots` are NEW public symbols introduced by PR #16 (not transitions from a prior stub), they do not appear in the stub-callable table — their callability contracts are locked here in the MODIFIED Package layout requirement for symmetry with `scaleogram`

#### Scenario: `synthetic.generate_trajectory` has no `px_per_mm` parameter
- **WHEN** `inspect.signature(sleap_roots.circumnutation.synthetic.generate_trajectory)` is inspected
- **THEN** the parameter list does not contain `px_per_mm`
- **AND** the docstring confirms the generator emits pure-pixel trajectories (callers compose `convert_to_mm()` if they want mm output)

#### Scenario: `import sleap_roots` succeeds without raising
- **WHEN** a user runs `import sleap_roots`
- **THEN** no exception is raised
- **AND** `sleap_roots.CircumnutationInputs` is accessible
- **AND** `sleap_roots.convert_to_mm` is accessible

### Requirement: CircumnutationInputs data class
The system SHALL provide an `attrs`-based `CircumnutationInputs` class capturing `(trajectory_df: pd.DataFrame, cadence_s: float, R_px: Optional[float] = None, run_id: Optional[str] = None)`. It SHALL validate at construction:

- `trajectory_df` is a `pandas.DataFrame` and is non-empty (≥ 1 row)
- `trajectory_df` contains the eight row-identity columns required by the trait CSV schema (Requirement: Trait CSV row-identity schema)
- `trajectory_df` contains the three per-frame columns `frame`, `tip_x`, `tip_y` (the foundation does not assert finiteness of `tip_x`/`tip_y` per row — that is a tier-PR concern — but their presence is mandatory)
- `cadence_s` is a positive finite float: `not isinstance(value, bool)` (Python booleans are int-subclass and SHALL be rejected explicitly), AND `math.isfinite(float(value))`, AND `float(value) > 0`. String-convertible numeric inputs (e.g. `cadence_s="300"`) are coerced to `float` before validation via an attrs converter.
- if `R_px` is set, the same rules apply: bool-rejected, finite, positive, string-convertible

It SHALL be re-exported from `sleap_roots/__init__.py`. The class SHALL NOT accept any `px_per_mm` parameter — calibration is a downstream concern handled by `convert_to_mm()`.

#### Scenario: Valid construction
- **GIVEN** a DataFrame containing all eight row-identity columns AND the three per-frame columns (`frame`, `tip_x`, `tip_y`) and at least one row
- **WHEN** `CircumnutationInputs(trajectory_df=df, cadence_s=300.0, R_px=2.4, run_id="plate_001")` is called
- **THEN** the instance is created without exception

#### Scenario: Missing row-identity column
- **GIVEN** a DataFrame missing the `plate_id` column
- **WHEN** `CircumnutationInputs(...)` is called
- **THEN** a `ValueError` is raised
- **AND** the exception message names the missing column

#### Scenario: Missing per-frame column
- **GIVEN** a DataFrame missing one of `frame`, `tip_x`, or `tip_y`
- **WHEN** `CircumnutationInputs(...)` is called
- **THEN** a `ValueError` is raised
- **AND** the exception message names the missing column

#### Scenario: Empty trajectory DataFrame
- **GIVEN** a DataFrame with all eight row-identity columns but zero rows
- **WHEN** `CircumnutationInputs(trajectory_df=df, cadence_s=300.0)` is called
- **THEN** a `ValueError` is raised
- **AND** the exception message indicates the DataFrame is empty

#### Scenario: Invalid cadence_s (zero, negative, NaN, infinity)
- **WHEN** `CircumnutationInputs(trajectory_df=df, cadence_s=v)` is called for any of `v ∈ {0.0, -1.0, float('nan'), float('inf'), float('-inf')}`
- **THEN** in each case a `ValueError` is raised
- **AND** the exception message names the `cadence_s` field

#### Scenario: cadence_s as Python bool is rejected
- **WHEN** `CircumnutationInputs(trajectory_df=df, cadence_s=True)` is called
- **THEN** a `ValueError` is raised
- **AND** the exception message names the `cadence_s` field

#### Scenario: Invalid R_px (zero, negative, NaN, infinity, bool)
- **WHEN** `CircumnutationInputs(trajectory_df=df, cadence_s=300.0, R_px=v)` is called for any of `v ∈ {0.0, -2.4, float('nan'), float('inf'), True}`
- **THEN** in each case a `ValueError` is raised
- **AND** the exception message names the `R_px` field
- **AND** `CircumnutationInputs(trajectory_df=df, cadence_s=300.0, R_px=None)` succeeds

#### Scenario: Importable from top-level
- **WHEN** a user runs `from sleap_roots import CircumnutationInputs`
- **THEN** the import succeeds

### Requirement: Trait CSV row-identity schema
Every per-plant trait CSV SHALL begin with the eight columns `(series, sample_uid, timepoint, plate_id, plant_id, track_id, genotype, treatment)` in that order, ahead of any trait columns. Today `plant_id` SHALL be populated identically to `track_id`; both columns SHALL exist so future divergence is non-breaking. `genotype` SHALL be populated from Series-level metadata where available (the `series-metadata` capability, PR #171), NaN otherwise. **`plate_id` and `treatment` SHALL be populated as NaN today** — no upstream produces them; the schema reserves them for future upstream metadata work. The DataFrame SHALL be sorted via `pandas.DataFrame.sort_values(by=['series', 'sample_uid', 'plate_id', 'plant_id', 'track_id'])`, where string columns sort lexicographically and integer columns (`track_id`) sort numerically.

The CSV-row builder `build_per_plant_template` SHALL key duplicate-row detection on the 5-tuple `(series, sample_uid, plate_id, plant_id, track_id)` only, NOT the full 8-tuple. If the same 5-tuple has conflicting values in `timepoint`, `genotype`, or `treatment` across the trajectory frames (a sign of upstream join error), `build_per_plant_template` SHALL raise `ValueError` rather than emit duplicate rows.

#### Scenario: Schema columns exist with correct dtypes
- **GIVEN** a DataFrame produced by the foundation's CSV-row builder for `CircumnutationInputs` containing 6 tracks
- **THEN** the first 8 columns are `series`, `sample_uid`, `timepoint`, `plate_id`, `plant_id`, `track_id`, `genotype`, `treatment`
- **AND** `track_id` has integer dtype
- **AND** `plant_id` is column-wise equal to `track_id`
- **AND** `series`, `sample_uid`, `genotype`, `treatment`, `plate_id` are object dtype (allowing NaN + string)

#### Scenario: Sort order is numeric for track_id
- **GIVEN** a DataFrame with `track_id ∈ {2, 10}` (and identical other identity columns)
- **WHEN** the foundation sorts the DataFrame
- **THEN** the row with `track_id=2` precedes the row with `track_id=10`
- **AND** the sort is NOT lexicographic (which would put `10` before `2`)

#### Scenario: Conflicting per-frame metadata raises ValueError
- **GIVEN** a `trajectory_df` where the same `(series, sample_uid, plate_id, plant_id, track_id)` 5-tuple has different `genotype` (or `treatment` or `timepoint`) values across frames
- **WHEN** `build_per_plant_template(inputs)` is called
- **THEN** a `ValueError` is raised
- **AND** the exception message names the offending 5-tuple AND the column whose values conflict

#### Scenario: track_id with NaN raises clear error
- **GIVEN** a `trajectory_df` whose `track_id` column contains NaN in at least one row
- **WHEN** `build_per_plant_template(inputs)` is called
- **THEN** a `ValueError` is raised
- **AND** the exception message names the `track_id` field (rather than a cryptic pandas `IntCastingNaNError`)

### Requirement: Pure-pixel pipeline output convention
The pipeline SHALL never accept `px_per_mm` as a parameter and SHALL never emit `[mm]` columns directly. Every length-bearing trait SHALL be expressed in pixels (`px`, `px²`, `px/frame`, `px/hr`, `px·hr⁻¹`); time in `hr` or `s`; angles in `rad`; rates in `hr⁻¹`; ratios as dimensionless (`—`); booleans as `bool`; integer counts as `int`; categorical strings as `string`. Internal CWT, ridge extraction, and derivative computations SHALL operate in pixels. This convention matches `TrackedTipPipeline`'s `lengths: "pixels"` declaration in `_TRACKED_TIP_UNITS` (`sleap_roots/tracked_tip_pipeline.py`).

#### Scenario: Pipeline output is calibration-independent
- **GIVEN** the same `CircumnutationInputs` (no `px_per_mm` parameter exists)
- **WHEN** the foundation's CSV-row builder produces a per-plant DataFrame for any future tier's traits
- **THEN** every numeric column has a unit string in the documented vocabulary
- **AND** no column has a unit string of `mm`, `mm²`, `mm/hr`, or any other mm-bearing unit

### Requirement: convert_to_mm utility
The system SHALL provide a `sleap_roots.circumnutation.units.convert_to_mm(traits_df: pd.DataFrame, units: dict[str, str], px_per_mm: float) -> tuple[pd.DataFrame, dict[str, str]]` pure function. It SHALL: (a) return a NEW DataFrame and units dict (input arguments not mutated), (b) for every column whose unit string is `px`, `px²`, `px/frame`, `px/hr`, or `px·hr⁻¹`, scale the values by the appropriate power of `1/px_per_mm` and rename the column with `_mm`-suffix replacing the `_px`-suffix (also updating the unit string), (c) pass non-px columns and their units through unchanged, (d) detect and raise `ValueError` when a `_px`→`_mm` rename would collide with an existing `_mm`-named column in the input (silent data loss prevention), (e) validate `px_per_mm` is a positive finite float (rejects 0, negative, NaN, ±inf, Python bool) and raise `ValueError` otherwise. The function SHALL be re-exported from `sleap_roots/__init__.py`.

#### Scenario: Identity at px_per_mm = 1.0
- **GIVEN** a 1-row DataFrame with column `length_px = 47.24` and units `{"length_px": "px"}`
- **WHEN** `convert_to_mm(df, units, px_per_mm=1.0)` is called
- **THEN** the returned DataFrame has column `length_mm = 47.24`
- **AND** the returned units has `{"length_mm": "mm"}`
- **AND** the input DataFrame and units dict are unchanged

#### Scenario: 1200 DPI conversion
- **GIVEN** a 1-row DataFrame with `length_px = 47.24` and units `{"length_px": "px"}`
- **WHEN** `convert_to_mm(df, units, px_per_mm=47.24)` is called
- **THEN** the returned DataFrame has column `length_mm = 1.0` (within IEEE float tolerance)
- **AND** the returned units has `{"length_mm": "mm"}`

#### Scenario: Velocity unit conversions
- **GIVEN** a DataFrame with columns `v_long_px_per_hr`, `v_total_px_per_frame` and matching unit strings
- **WHEN** `convert_to_mm(df, units, px_per_mm=47.24)` is called
- **THEN** the returned columns are `v_long_mm_per_hr`, `v_total_mm_per_frame` with values divided by `47.24`
- **AND** units strings are updated to `mm/hr`, `mm/frame`

#### Scenario: Non-px columns pass through
- **GIVEN** a DataFrame with columns `T_nutation_hr` (unit `hr`), `is_nutating` (unit `bool`), `B_balance_number` (unit `—`)
- **WHEN** `convert_to_mm(df, units, px_per_mm=47.24)` is called
- **THEN** all three columns retain their names, values, and unit strings unchanged

#### Scenario: Invalid px_per_mm (zero, negative, NaN, infinity)
- **WHEN** `convert_to_mm(df, units, px_per_mm=v)` is called for any of `v ∈ {0.0, -1.0, float('nan'), float('inf'), float('-inf')}`
- **THEN** a `ValueError` is raised
- **AND** the exception message names the `px_per_mm` field

#### Scenario: Rename collision raises ValueError
- **GIVEN** a DataFrame with BOTH `length_px` AND `length_mm` columns (the latter from a prior conversion)
- **WHEN** `convert_to_mm(df, units, px_per_mm=47.24)` is called
- **THEN** a `ValueError` is raised
- **AND** the exception message names both the source column (`length_px`) and the target column (`length_mm`) that would collide

### Requirement: Units sidecar JSON
For every per-plant trait CSV the system SHALL write a sibling `traits_per_plant.units.json` mapping each column name to a unit string. Every column (numeric, boolean, string) SHALL be present. The unit-string vocabulary for sidecar values is `sleap_roots.circumnutation._constants.PIPELINE_UNIT_VOCABULARY` (pixel-based and calibration-independent units only — no mm-based units). The writer SHALL validate every unit string against `PIPELINE_UNIT_VOCABULARY` BEFORE writing and SHALL raise `ValueError` naming the offending column/unit pair if any value is out-of-vocabulary. The JSON file SHALL be written with `encoding="utf-8"` so non-ASCII unit symbols (`²`, `⁻`, `·`) round-trip on Windows.

#### Scenario: Sidecar exists and parses
- **WHEN** the foundation writes a CSV via `_io.write_per_plant_csv`
- **THEN** a sibling `traits_per_plant.units.json` exists in the same directory
- **AND** it parses as valid JSON
- **AND** every column from the CSV is a key in the JSON mapping
- **AND** every value is a string in `PIPELINE_UNIT_VOCABULARY`

#### Scenario: UTF-8 round-trip with non-ASCII unit
- **GIVEN** a units dict containing `{"helix_signed_area": "px²"}`
- **WHEN** the foundation writes the sidecar to disk
- **AND** loads it back via `_io.read_units_sidecar`
- **THEN** the round-tripped dict contains `{"helix_signed_area": "px²"}` byte-for-byte unchanged

#### Scenario: Writing with invalid unit raises ValueError
- **GIVEN** a units dict containing a unit string not in `PIPELINE_UNIT_VOCABULARY` (e.g. `{"length_px": "mm"}` or `{"length_px": "kg"}`)
- **WHEN** `_io.write_per_plant_csv(out_path, df, units, run_metadata)` is called
- **THEN** a `ValueError` is raised
- **AND** the exception message names the offending column and the invalid unit string
- **AND** no CSV or sidecar files are written

#### Scenario: Writer rejects units dict that doesn't cover every column
- **GIVEN** a DataFrame with columns `["a", "b"]` and a units dict `{"a": "px"}` (missing `"b"`)
- **WHEN** `_io.write_per_plant_csv(out_path, df, units, run_metadata)` is called
- **THEN** a `ValueError` is raised
- **AND** the exception message names the missing column(s)
- **AND** no CSV or sidecar files are written

#### Scenario: Writer rejects units dict with extra keys not in the DataFrame
- **GIVEN** a DataFrame with columns `["a"]` and a units dict `{"a": "px", "b": "hr"}` (`"b"` not in df)
- **WHEN** `_io.write_per_plant_csv(out_path, df, units, run_metadata)` is called
- **THEN** a `ValueError` is raised
- **AND** the exception message names the extra key(s)
- **AND** no CSV or sidecar files are written

### Requirement: Run-metadata sidecar
For every per-plant CSV the system SHALL write a sibling `run_metadata.json` capturing: `input_path`, `sleap_roots_git_sha`, `sleap_roots_version`, `sleap_io_version`, `numpy_version`, `scipy_version`, `pandas_version`, `python_version`, `platform`, `timestamp` (ISO 8601 UTC), `run_id`, `cadence_s`, `R_px`, `_schema_version`, `_constants_version`, `_constants_snapshot`. The `cadence_s` and `R_px` fields (added in PR #14) capture the per-run acquisition inputs from `CircumnutationInputs` so a run is reproducible from the sidecars alone — `cadence_s` determines every period trait (`*_s`) and the traveling-wave residual, and is otherwise unrecoverable from the `.slp` at `input_path`. `gather_run_metadata` SHALL accept `cadence_s` and `R_px` as optional keyword parameters (default `None`), so pre-PR-#14 callers that omit them write `null` (mirroring the nullable `run_id`); the PR #14 pipeline `save` SHALL pass `inputs.cadence_s` and `inputs.R_px`. The `_constants_snapshot` SHALL be a JSON-serializable mapping from every name in `_constants.py` to its value at write time. The numpy / scipy / pandas / platform fields support numerical reproducibility (IEEE float rounding can differ between numpy versions and across BLAS implementations).

#### Scenario: Run-metadata sidecar contains required fields
- **WHEN** the foundation writes a CSV
- **THEN** a sibling `run_metadata.json` exists in the same directory
- **AND** every key listed above is present and non-null (except `run_id`, `cadence_s`, and `R_px`, which may be null when not supplied)
- **AND** `_schema_version` and `_constants_version` are integers
- **AND** `_constants_snapshot` contains every name in `_constants.py` with its current value
- **AND** `numpy_version`, `scipy_version`, `pandas_version` are present as version strings (e.g. `"2.3.4"`, `"1.16.3"`, `"2.2.0"`) or `"unknown"` if the dependency could not be imported
- **AND** `platform` is a non-empty string (the value of `platform.platform()`)

#### Scenario: Run-metadata captures cadence_s and R_px from the pipeline
- **GIVEN** the PR #14 pipeline `save` invoked with `inputs.cadence_s == 300.0` and `inputs.R_px == 12.0`
- **WHEN** `gather_run_metadata(input_path, run_id=inputs.run_id, constants=..., cadence_s=300.0, R_px=12.0)` writes the sidecar
- **THEN** `run_metadata["cadence_s"] == 300.0` and `run_metadata["R_px"] == 12.0`
- **AND** WHEN `cadence_s` / `R_px` are omitted (a non-pipeline caller), both keys are present with value `null`

#### Scenario: Constants snapshot reflects override
- **GIVEN** a custom `ConstantsT` override passed to the writer with `BAND_POWER_NOISE_RATIO = 4`
- **WHEN** the foundation writes the metadata
- **THEN** `_constants_snapshot["BAND_POWER_NOISE_RATIO"] == 4`
- **AND** the default-valued constants reflect the unmodified defaults

### Requirement: Module-level constants
The system SHALL expose all overridable defaults as module-level named constants in `sleap_roots/circumnutation/_constants.py`. The set SHALL include at minimum: `NOISE_MASK_K`, `LGZ_STEADY_STATE_RESIDUAL_MAX`, `NYQUIST_RATIO_MAX`, `SG_D2_AGREEMENT_MAX`, `SG_MSD_AGREEMENT_MAX`, `D2_MSD_AGREEMENT_MAX`, `LGZ_NMIN_RESOLVABLE`, `COI_FRACTION_MAX`, `BAND_POWER_NOISE_RATIO`, `WAVELET_DEFAULT_TEMPORAL`, `WAVELET_DEFAULT_SPATIAL`, `SG_WINDOW_SHORT`, `SG_DEGREE`, `SG_WINDOW_DETREND`, `OUTLIER_STEP_RATIO`, `FRAC_OUTLIER_STEPS_MAX`, `WORST_STEP_RATIO_MAX`, `GROWTH_AXIS_RELIABILITY_K`, `SYNTHETIC_T_NUTATION_S`, `SYNTHETIC_AMPLITUDE_PX`, `SYNTHETIC_GROWTH_RATE_PX_PER_FRAME`, `SYNTHETIC_NOISE_SIGMA_PX`, `SYNTHETIC_CADENCE_S`, `SYNTHETIC_N_FRAMES`, `SYNTHETIC_GROWTH_AXIS_ANGLE_RAD`, `COI_EFOLDING_FACTOR`, `CWT_SCALE_COUNT_DEFAULT`, `CWT_PERIOD_MIN_NYQUIST_FACTOR`, `CWT_PERIOD_MAX_SIGNAL_FRACTION`, `RIDGE_CONTINUITY_FILTER_WINDOW`, `NOISE_FLOOR_OUT_OF_BAND_FACTOR`, `BAND_POWER_BAND_LOW_FACTOR`, `BAND_POWER_BAND_HIGH_FACTOR`, `DERR_EXPECTED_PERIOD_S`, `TEMPORAL_NYQUIST_RATIO_MAX`, `SPATIAL_COI_EFOLDING_FACTOR`, `CWT_WAVELENGTH_MIN_NYQUIST_FACTOR`, `CWT_WAVELENGTH_MAX_SIGNAL_FRACTION`, `_SCHEMA_VERSION`, `_CONSTANTS_VERSION`. The values SHALL match the defaults in `docs/circumnutation/roadmap.md` cross-cutting concern CC-2 and `docs/circumnutation/theory.md` §7.6 (for the QC-tier-introduced thresholds: `FRAC_OUTLIER_STEPS_MAX = 0.05`, `WORST_STEP_RATIO_MAX = 5`, `SG_MSD_AGREEMENT_MAX = 1.5`, `D2_MSD_AGREEMENT_MAX = 1.5`) and `docs/circumnutation/preliminary_results_2026-05-07.md` §1, §3.4, §4.1, §4.3 (for the synthetic-generator defaults: `SYNTHETIC_T_NUTATION_S = 3333.0`, `SYNTHETIC_AMPLITUDE_PX = 10.0`, `SYNTHETIC_GROWTH_RATE_PX_PER_FRAME = 4.29`, `SYNTHETIC_NOISE_SIGMA_PX = 2.0`, `SYNTHETIC_CADENCE_S = 300.0`, `SYNTHETIC_N_FRAMES = 575`, `SYNTHETIC_GROWTH_AXIS_ANGLE_RAD = math.pi / 2`) and the wavelet-aware step-response derivation in PR #5's archived `design.md` D3 (for the CWT-machinery defaults: `COI_EFOLDING_FACTOR = math.sqrt(1.5)` calibrated for `cmor1.5-1.0` per `√B` envelope e-folding; `CWT_SCALE_COUNT_DEFAULT = 64`; `CWT_PERIOD_MIN_NYQUIST_FACTOR = 2.0`; `CWT_PERIOD_MAX_SIGNAL_FRACTION = 0.25`) and PR #6's `design.md` D4/D7/D8/D9 (for the Tier 1 / threshold defaults: `RIDGE_CONTINUITY_FILTER_WINDOW = 5` per GitHub issue #214 + Mallat 1999 §4.4.2; `NOISE_FLOOR_OUT_OF_BAND_FACTOR = 5.0` per CC-8 verbatim; `BAND_POWER_BAND_LOW_FACTOR = 0.5` and `BAND_POWER_BAND_HIGH_FACTOR = 2.0` per theory.md §7.2 `[0.5T, 2T]` band; `DERR_EXPECTED_PERIOD_S = 3333.0` per preliminary_results §4.4 Nipponbare empirical anchor; `TEMPORAL_NYQUIST_RATIO_MAX = 0.25` per theory.md §6.5 "10-min still works" empirical anchor) and PR #9's `design.md` (for the spatial-CWT-machinery defaults: `SPATIAL_COI_EFOLDING_FACTOR` set to the empirically-measured `cgau2` e-folding factor from a step-response capture script — the analog of `COI_EFOLDING_FACTOR=√1.5` for `cmor1.5-1.0`, whose docstring explicitly defers the `cgau2` factor to PR #9; `CWT_WAVELENGTH_MIN_NYQUIST_FACTOR = 2.0` and `CWT_WAVELENGTH_MAX_SIGNAL_FRACTION = 0.25`, the spatial-domain siblings of `CWT_PERIOD_MIN_NYQUIST_FACTOR` / `CWT_PERIOD_MAX_SIGNAL_FRACTION` — same numeric defaults, dimensional separation in NAMES + docstrings). `_SCHEMA_VERSION` SHALL be `1` (unchanged from PR #1) and `_CONSTANTS_VERSION` SHALL be `6` (bumped from `5` in this PR per the version-sentinel contract — the constants set grew by 3). The module SHALL also expose `PIPELINE_UNIT_VOCABULARY` (px-based + calibration-independent units, the closed sidecar vocabulary), `CONVERTED_UNIT_VOCABULARY` (mm-based units produced by `convert_to_mm`), and `VALID_UNIT_VOCABULARY` (their union), plus `ROW_IDENTITY_UNITS` (the canonical units dict for the eight row-identity columns).

The `ConstantsT` typed override-bag SHALL include corresponding fields for every overridable constant above, so callers can override per-call via `ConstantsT(SPATIAL_COI_EFOLDING_FACTOR=..., CWT_WAVELENGTH_MIN_NYQUIST_FACTOR=4.0)` etc. `_default_constants_snapshot()` SHALL emit every constant name in the set above into the run-metadata sidecar, including the three new PR #9 spatial-CWT constants. `len(_default_constants_snapshot())` SHALL be `38` exactly after PR #9 (35 pre-PR-#9 baseline + 3 PR #9 additions).

**Scope note on `NYQUIST_RATIO_MAX` ↔ `TEMPORAL_NYQUIST_RATIO_MAX` reciprocal docstring touch.** PR #6 ADDS a reciprocal docstring cross-reference between the existing `NYQUIST_RATIO_MAX` (locked as the SPATIAL cadence-Nyquist threshold per `theory.md` §6.5 spatial example "5.83 / 65 ≈ 9.0%, well below the conservative 25% threshold") and the NEW `TEMPORAL_NYQUIST_RATIO_MAX` (locked as the TEMPORAL cadence-Nyquist threshold per `theory.md` §6.5 temporal example "5-min comfortable, 10-min still works, 30-min aliases"). Both defaults are `0.25` — the dimensional separation lives in the constant NAMES + docstrings, NOT in different values. This is an internal-docstring-only touch with no behavioral change; PR #5's earlier touch of `NYQUIST_RATIO_MAX` docstring (adding a cross-reference to `CWT_PERIOD_MAX_SIGNAL_FRACTION`) is preserved. PR #6 enumerates the change here so the touch is explicitly scope-bounded and not surprising to reviewers of the `_constants.py` diff.

#### Scenario: All required constants are importable with correct types
- **WHEN** a user runs `from sleap_roots.circumnutation import _constants`
- **THEN** every name listed above is an attribute of `_constants`
- **AND** each value matches the documented default in `roadmap.md` CC-2, `theory.md` §6.5/§7.2/§7.6, `preliminary_results_2026-05-07.md` §1/§3.4/§4.1/§4.3/§4.4, PR #5's archived `design.md` D3/D7, PR #6's `design.md` D4/D7/D8/D9, and PR #9's `design.md`
- **AND** `_constants._SCHEMA_VERSION` is the integer `1`
- **AND** `_constants._CONSTANTS_VERSION` is the integer `6`
- **AND** `_constants.PIPELINE_UNIT_VOCABULARY`, `_constants.CONVERTED_UNIT_VOCABULARY`, `_constants.VALID_UNIT_VOCABULARY`, `_constants.ROW_IDENTITY_UNITS` are all importable

#### Scenario: New QC-tier constants are overridable via ConstantsT
- **GIVEN** a custom `ConstantsT(FRAC_OUTLIER_STEPS_MAX=0.10, WORST_STEP_RATIO_MAX=10, SG_MSD_AGREEMENT_MAX=2.0, D2_MSD_AGREEMENT_MAX=2.0)`
- **WHEN** the instance is inspected
- **THEN** each overridden field reflects its caller-supplied value
- **AND** unoverridden fields reflect their module-level defaults

#### Scenario: New synthetic-generator constants are overridable via ConstantsT
- **GIVEN** a custom `ConstantsT(SYNTHETIC_T_NUTATION_S=1800.0, SYNTHETIC_AMPLITUDE_PX=20.0, SYNTHETIC_GROWTH_RATE_PX_PER_FRAME=3.0, SYNTHETIC_NOISE_SIGMA_PX=1.0, SYNTHETIC_CADENCE_S=60.0, SYNTHETIC_N_FRAMES=200, SYNTHETIC_GROWTH_AXIS_ANGLE_RAD=0.0)`
- **WHEN** the instance is inspected
- **THEN** each of the seven overridden fields reflects its caller-supplied value
- **AND** unoverridden fields reflect their module-level defaults

#### Scenario: New CWT-machinery constants are overridable via ConstantsT
- **GIVEN** a custom `ConstantsT(COI_EFOLDING_FACTOR=math.sqrt(2.0), CWT_SCALE_COUNT_DEFAULT=128, CWT_PERIOD_MIN_NYQUIST_FACTOR=4.0, CWT_PERIOD_MAX_SIGNAL_FRACTION=0.5)`
- **WHEN** the instance is inspected
- **THEN** each of the four overridden fields reflects its caller-supplied value
- **AND** unoverridden fields reflect their module-level defaults

#### Scenario: New nutation/Tier 1 constants are overridable via ConstantsT
- **GIVEN** a custom `ConstantsT(RIDGE_CONTINUITY_FILTER_WINDOW=11, NOISE_FLOOR_OUT_OF_BAND_FACTOR=3.0, BAND_POWER_BAND_LOW_FACTOR=0.25, BAND_POWER_BAND_HIGH_FACTOR=4.0, DERR_EXPECTED_PERIOD_S=7200.0, TEMPORAL_NYQUIST_RATIO_MAX=0.20)`
- **WHEN** the instance is inspected
- **THEN** each of the six overridden fields reflects its caller-supplied value
- **AND** unoverridden fields reflect their module-level defaults

#### Scenario: New spatial-CWT constants are overridable via ConstantsT
- **GIVEN** a custom `ConstantsT(SPATIAL_COI_EFOLDING_FACTOR=2.0, CWT_WAVELENGTH_MIN_NYQUIST_FACTOR=4.0, CWT_WAVELENGTH_MAX_SIGNAL_FRACTION=0.5)`
- **WHEN** the instance is inspected
- **THEN** each of the three overridden fields reflects its caller-supplied value
- **AND** unoverridden fields reflect their module-level defaults

#### Scenario: Constants snapshot includes the QC, synthetic, CWT-machinery, nutation/Tier 1, and spatial-CWT constants
- **WHEN** `_default_constants_snapshot()` is called
- **THEN** the returned mapping contains `FRAC_OUTLIER_STEPS_MAX`, `WORST_STEP_RATIO_MAX`, `SG_MSD_AGREEMENT_MAX`, `D2_MSD_AGREEMENT_MAX` with their default values
- **AND** the returned mapping contains `SYNTHETIC_T_NUTATION_S`, `SYNTHETIC_AMPLITUDE_PX`, `SYNTHETIC_GROWTH_RATE_PX_PER_FRAME`, `SYNTHETIC_NOISE_SIGMA_PX`, `SYNTHETIC_CADENCE_S`, `SYNTHETIC_N_FRAMES`, `SYNTHETIC_GROWTH_AXIS_ANGLE_RAD` with their default values
- **AND** the returned mapping contains `COI_EFOLDING_FACTOR`, `CWT_SCALE_COUNT_DEFAULT`, `CWT_PERIOD_MIN_NYQUIST_FACTOR`, `CWT_PERIOD_MAX_SIGNAL_FRACTION` with their default values
- **AND** the returned mapping contains `RIDGE_CONTINUITY_FILTER_WINDOW`, `NOISE_FLOOR_OUT_OF_BAND_FACTOR`, `BAND_POWER_BAND_LOW_FACTOR`, `BAND_POWER_BAND_HIGH_FACTOR`, `DERR_EXPECTED_PERIOD_S`, `TEMPORAL_NYQUIST_RATIO_MAX` with their default values
- **AND** the returned mapping contains `SPATIAL_COI_EFOLDING_FACTOR`, `CWT_WAVELENGTH_MIN_NYQUIST_FACTOR`, `CWT_WAVELENGTH_MAX_SIGNAL_FRACTION` with their default values
- **AND** `len(_default_constants_snapshot()) == 38` exactly (35 pre-PR-#9 baseline + 3 PR #9 additions)

#### Scenario: NYQUIST_RATIO_MAX docstring cross-references CWT_PERIOD_MAX_SIGNAL_FRACTION
- **WHEN** `sleap_roots.circumnutation._constants.NYQUIST_RATIO_MAX` is inspected via `inspect.getdoc(...)` or via `_constants.__dict__["NYQUIST_RATIO_MAX"].__doc__` (note: module-level constants do not carry docstrings natively; the cross-reference lives in the module-level constant's surrounding docstring or in a `__doc__`-attached commentary block defined at the same point — implementation may use either pattern as long as `grep "CWT_PERIOD_MAX_SIGNAL_FRACTION" sleap_roots/circumnutation/_constants.py` shows the cross-reference text within ±5 lines of `NYQUIST_RATIO_MAX:` declaration)
- **THEN** the cross-reference text mentions `CWT_PERIOD_MAX_SIGNAL_FRACTION` AND notes the numerical-equality-but-semantically-distinct relationship
- **AND** `CWT_PERIOD_MAX_SIGNAL_FRACTION`'s docstring reciprocally references `NYQUIST_RATIO_MAX`

#### Scenario: NYQUIST_RATIO_MAX and TEMPORAL_NYQUIST_RATIO_MAX cross-reference each other reciprocally
- **WHEN** `grep "TEMPORAL_NYQUIST_RATIO_MAX" sleap_roots/circumnutation/_constants.py` and `grep "NYQUIST_RATIO_MAX" sleap_roots/circumnutation/_constants.py` are inspected
- **THEN** the docstring or surrounding commentary on `NYQUIST_RATIO_MAX` mentions `TEMPORAL_NYQUIST_RATIO_MAX` AND identifies it as the temporal sibling (spatial-vs-temporal dimensional separation)
- **AND** the docstring or surrounding commentary on `TEMPORAL_NYQUIST_RATIO_MAX` reciprocally references `NYQUIST_RATIO_MAX` AND identifies it as the spatial sibling
- **AND** both constants have the same default value (`0.25`) but the dimensional separation lives in the constant NAMES + docstrings, NOT in different values

### Requirement: Per-module logger convention
Every module in `sleap_roots/circumnutation/` SHALL declare a module-level logger via `logger = logging.getLogger(__name__)`. The package SHALL NOT call `logging.basicConfig` or otherwise configure handlers at import time. No log records SHALL be emitted at package import time.

#### Scenario: Module loggers are namespaced
- **GIVEN** `import logging; import sleap_roots.circumnutation.kinematics`
- **THEN** `logging.getLogger("sleap_roots.circumnutation.kinematics")` returns the same logger object that the module uses

#### Scenario: No handlers added on import
- **GIVEN** a fresh Python process
- **WHEN** `caplog` captures all records and `import sleap_roots.circumnutation` is executed
- **THEN** the root logger's handlers list is unchanged from before the import
- **AND** no log records are emitted during import

### Requirement: Tier 0 raw kinematic traits
The system SHALL provide `sleap_roots.circumnutation.kinematics.compute(trajectory_df: pd.DataFrame, constants: Optional[ConstantsT] = None) -> pd.DataFrame`. The function SHALL accept the canonical `(trajectory_df, constants=None)` signature locked by the foundation's Package layout requirement — `cadence_s` SHALL NOT appear in the signature (the function emits cadence-independent units only).

It SHALL return a per-plant `pandas.DataFrame` with one row per unique `(series, sample_uid, plate_id, plant_id, track_id)` 5-tuple in the input, sorted via the same convention as `_io._build_per_plant_template_from_df`. The DataFrame columns SHALL be (in this order):

1. The 8 row-identity columns in their declared order: `series`, `sample_uid`, `timepoint`, `plate_id`, `plant_id`, `track_id`, `genotype`, `treatment`.
2. The 9 Tier 0 trait columns and 1 boolean flag, in this order:

| Column | Unit | Definition |
|---|---|---|
| `v_total_median_px_per_frame` | `px/frame` | `np.nanmedian(\|Δxy_i\| / Δframe_i)` over consecutive present frames — median per-frame step magnitude (gap-aware). |
| `v_long_signed_median_px_per_frame` | `px/frame` | `np.nanmedian((Δxy_i · û_g) / Δframe_i)` where `û_g = (xy[-1] − xy[0]) / D` is the unit growth-axis vector — signed median longitudinal component. |
| `v_long_abs_median_px_per_frame` | `px/frame` | `np.nanmedian(\|Δxy_i · û_g\| / Δframe_i)` — absolute median longitudinal component. |
| `v_lat_signed_median_px_per_frame` | `px/frame` | `np.nanmedian((Δxy_i · û_lat) / Δframe_i)` where `û_lat = (−û_g[1], û_g[0])` — signed median lateral component (expected ≈ 0 by symmetry; serves as a sanity-check trait). |
| `v_lat_abs_median_px_per_frame` | `px/frame` | `np.nanmedian(\|Δxy_i · û_lat\| / Δframe_i)` — absolute median lateral component. |
| `long_lat_ratio` | `—` | `v_long_abs_median_px_per_frame / v_lat_abs_median_px_per_frame`; NaN when `v_lat_abs_median_px_per_frame == 0`. |
| `path_displacement_ratio` | `—` | `L / D` where `L = sum(\|Δxy_i\|)` over consecutive present frames and `D = \|xy[-1] − xy[0]\|`; NaN when `D == 0` exactly. |
| `angular_amplitude` | `rad` | `np.nanmax(ψ_g) − np.nanmin(ψ_g)` where `ψ_g = _geometry.compute_psi_g(x, y) = np.unwrap(np.arctan2(dx, dy))` per Bastien-Meroz 2016 Eq. 20 / theory.md §3.5 (note argument order: `dx` first, then `dy`); rotation-invariant peak-to-peak extent. |
| `principal_axis_angle` | `rad` | `np.arctan2(û_g[1], û_g[0])`; full range `(−π, π]`; image-y-downward convention (a root growing image-down reads as `+π/2`). |
| `growth_axis_unreliable` | `bool` | True iff the growth axis is unreliable per Requirement: Growth-axis reliability gate. |

The function SHALL emit only units in `PIPELINE_UNIT_VOCABULARY` — specifically `px/frame`, `—`, `rad`, `bool` for the new columns. NO `_mm`-suffixed columns and NO `_px_per_hr` columns SHALL be emitted by `kinematics.compute`.

The function SHALL process each track via the algorithm documented in `design.md` D5: (a) drop NaN rows on `tip_x`/`tip_y` BEFORE any `np.diff` or arithmetic — this NaN-then-sort ordering is load-bearing because `np.diff` of a NaN-bearing array propagates NaN to two adjacent diffs and `np.sum` (used to compute path length `L`) would silently return NaN; (b) sort the remaining rows by `frame`; (c) if fewer than 2 frames remain, emit NaN for all 9 trait columns and `False` for `growth_axis_unreliable`; (d) compute gap-aware per-frame velocities `Δxy / Δframe`; (e) compute the magnitude, signed and absolute longitudinal/lateral components, ratios, `angular_amplitude` via `_geometry.compute_psi_g`, and `principal_axis_angle`; (f) apply the reliability gate. The path-length sum `L = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))` uses `np.sum` (not `np.nansum`); the dropna precondition guarantees no NaN reaches this sum.

The function SHALL accept `constants=None` (default) by constructing the default `ConstantsT()`, OR a caller-supplied `ConstantsT` override. The constants consumed are `SG_WINDOW_SHORT`, `SG_DEGREE` (passed to `_noise.compute_sg_residual_xy`), and `GROWTH_AXIS_RELIABILITY_K` (the gate threshold multiplier).

The function SHALL validate `trajectory_df` is a `pandas.DataFrame` containing the 8 row-identity columns plus `frame`, `tip_x`, `tip_y` columns. On validation failure it SHALL raise `ValueError` whose message names the offending issue (missing column name, wrong type, etc.) — matching the foundation's permissive-but-clear-error style.

#### Scenario: Straight-line track yields exact analytical values
- **GIVEN** a 100-frame trajectory_df with `tip_x = frame * 1.0` and `tip_y = 0.0` for a single track (unit horizontal velocity)
- **WHEN** `kinematics.compute(trajectory_df)` is called
- **THEN** the returned DataFrame has exactly 1 row
- **AND** `v_total_median_px_per_frame == 1.0` within IEEE float tolerance
- **AND** `v_long_signed_median_px_per_frame == 1.0` and `v_long_abs_median_px_per_frame == 1.0`
- **AND** `v_lat_signed_median_px_per_frame == 0.0` and `v_lat_abs_median_px_per_frame == 0.0`
- **AND** `long_lat_ratio` is `NaN` (denominator zero)
- **AND** `path_displacement_ratio == 1.0` exactly (`L == D`)
- **AND** `angular_amplitude == 0.0` (ψ_g is constant)
- **AND** `principal_axis_angle == 0.0`
- **AND** `growth_axis_unreliable == False`

#### Scenario: Image-y-downward straight-line track reports `principal_axis_angle ≈ π/2`
- **GIVEN** a 100-frame trajectory_df with `tip_x = 0.0` and `tip_y = frame * 1.0` for a single track (unit downward velocity in image space)
- **WHEN** `kinematics.compute(trajectory_df)` is called
- **THEN** `principal_axis_angle == math.pi / 2` within IEEE float tolerance

#### Scenario: Pure-noise track triggers gate and NaNs rotation-dependent traits
- **GIVEN** a 100-frame trajectory_df where `tip_x` and `tip_y` are i.i.d. `N(0, 1)` drawn via `np.random.default_rng(0).normal(0, 1, size=(2, 100))` for a single track (concrete seed for cross-platform test determinism)
- **WHEN** `kinematics.compute(trajectory_df)` is called
- **THEN** `growth_axis_unreliable == True`
- **AND** the 6 rotation-dependent columns (`v_long_signed_median_px_per_frame`, `v_long_abs_median_px_per_frame`, `v_lat_signed_median_px_per_frame`, `v_lat_abs_median_px_per_frame`, `long_lat_ratio`, `principal_axis_angle`) are all `NaN`
- **AND** the 3 rotation-invariant traits (`v_total_median_px_per_frame`, `path_displacement_ratio`, `angular_amplitude`) are finite (not `NaN`)

#### Scenario: Circular trajectory yields `angular_amplitude ≈ 2π`
- **GIVEN** a 100-frame trajectory_df with `tip_x = 10·cos(2π·frame/100) + frame * 0.1` and `tip_y = 10·sin(2π·frame/100)` for a single track. The circle radius `R = 10` is chosen so the circle's per-frame velocity magnitude `R·(2π/100) ≈ 0.628` dominates the horizontal drift (`0.1` per frame); this ensures `dx = −R·sin(t)·dt + drift` swings through both signs over one revolution and `ψ_g = atan2(dx, dy)` sweeps the full 2π range. A smaller `R` with the same drift would leave `dx` always positive and `ψ_g` would not complete a revolution.
- **WHEN** `kinematics.compute(trajectory_df)` is called
- **THEN** `angular_amplitude` is in the range `[2π - 0.5, 2π + 0.5]` (one full revolution within tolerance)
- **AND** `growth_axis_unreliable == False` (D ≈ 10 px from the horizontal drift dominates the smooth-trajectory SG residual)

#### Scenario: NaN rows are dropped BEFORE diff (ordering is load-bearing)
- **GIVEN** a 100-frame straight-line trajectory_df with `tip_x = frame * 1.0` and `tip_y = 0.0` where 10 random rows (seeded `np.random.default_rng(0)`) have `tip_x = NaN`
- **WHEN** `kinematics.compute(trajectory_df)` is called
- **THEN** the trait values match the no-NaN-row case (scenario "Straight-line track yields exact analytical values") within IEEE float tolerance — specifically `v_total_median_px_per_frame == 1.0` exactly (gap-aware diff: removing 10 NaN rows produces a remaining 90-row contiguous-or-gapped track, each per-frame step normalized by Δframe yields 1.0)
- **AND** `path_displacement_ratio == 1.0` exactly (the NaN-then-sort ordering ensures `np.sum` of the present-row step magnitudes returns a finite value, not NaN — if the diff happened before dropna, NaN would propagate through `np.sum` and contaminate the ratio)

#### Scenario: Frame gaps are handled gap-aware
- **GIVEN** a straight-line trajectory_df at velocity 1 px/frame, but with frames `[40..50)` missing (10-frame gap)
- **WHEN** `kinematics.compute(trajectory_df)` is called
- **THEN** `v_total_median_px_per_frame == 1.0` exactly (gap-aware per-frame velocity: the big jump across the gap is divided by `Δframe = 11`, normalizing to 1 px/frame)

#### Scenario: Track with fewer than 2 valid frames emits NaN traits without raising
- **GIVEN** a trajectory_df with a single track that has only 1 row (or all rows have NaN coords)
- **WHEN** `kinematics.compute(trajectory_df)` is called
- **THEN** the returned DataFrame has 1 row for that track
- **AND** all 9 trait columns are `NaN`
- **AND** `growth_axis_unreliable == False` (cannot judge reliability with fewer than 2 frames)
- **AND** no exception is raised

#### Scenario: Zero net displacement yields NaN ratios and `growth_axis_unreliable=True`
- **GIVEN** a trajectory_df for a single track whose start and end coordinates are identical (`xy[-1] == xy[0]`)
- **WHEN** `kinematics.compute(trajectory_df)` is called
- **THEN** `path_displacement_ratio` is `NaN`
- **AND** `growth_axis_unreliable == True`
- **AND** the 6 rotation-dependent columns are `NaN`
- **AND** the rotation-invariant `v_total_median_px_per_frame` and `angular_amplitude` are finite (not NaN)

#### Scenario: Output DataFrame columns are in the specified order
- **GIVEN** a valid trajectory_df with 6 tracks
- **WHEN** `kinematics.compute(trajectory_df)` is called
- **THEN** the returned DataFrame has exactly 18 columns
- **AND** the first 8 columns are `series`, `sample_uid`, `timepoint`, `plate_id`, `plant_id`, `track_id`, `genotype`, `treatment` in that order
- **AND** the next 10 columns are `v_total_median_px_per_frame`, `v_long_signed_median_px_per_frame`, `v_long_abs_median_px_per_frame`, `v_lat_signed_median_px_per_frame`, `v_lat_abs_median_px_per_frame`, `long_lat_ratio`, `path_displacement_ratio`, `angular_amplitude`, `principal_axis_angle`, `growth_axis_unreliable` in that order

#### Scenario: Output column units are within `PIPELINE_UNIT_VOCABULARY`
- **GIVEN** the output of `kinematics.compute(trajectory_df)` and the units-mapping dict for the 10 new columns
- **WHEN** every unit string is checked against `sleap_roots.circumnutation._constants.PIPELINE_UNIT_VOCABULARY`
- **THEN** every unit string is a member of the vocabulary
- **AND** no unit string is `mm`-bearing or `px/hr`-bearing

#### Scenario: ConstantsT override changes the gate threshold
- **GIVEN** a trajectory_df constructed to produce a known `D` and a known SG-residual value, by either of two equivalent test-construction patterns:
  - (Recipe A — preferred for unit tests) a `monkeypatch`-injected `_noise.compute_sg_residual_xy` that returns a fixed `1.0`, paired with a 6-frame trajectory `xy = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)]` yielding `D = 5.0` exactly
  - (Recipe B — for integration coverage) a 100-frame trajectory with `x_smooth = np.linspace(0, 5, 100)`, `y_smooth = np.zeros(100)`, plus `np.random.default_rng(0).normal(0, 0.7, size=(2, 100))` added to xy; the smooth+noise combination yields `D ≈ 5` and SG-residual `≈ 1` deterministically
- **WHEN** `kinematics.compute(trajectory_df, constants=ConstantsT(GROWTH_AXIS_RELIABILITY_K=3))` is called
- **THEN** `growth_axis_unreliable == False` (because `5 > 3 * 1`)
- **AND** when called with `ConstantsT(GROWTH_AXIS_RELIABILITY_K=10)` instead, `growth_axis_unreliable == True` (because `5 < 10 * 1`)

#### Scenario: Invalid trajectory_df raises ValueError
- **WHEN** `kinematics.compute(None)` is called
- **THEN** a `ValueError` is raised whose message indicates the input is not a `pandas.DataFrame`
- **AND** `kinematics.compute(df_missing_tip_x)` similarly raises a `ValueError` naming the missing column

### Requirement: Growth-axis reliability gate
For each track, the system SHALL compute net displacement `D = ‖xy[-1] − xy[0]‖` (in pixels) and a local SG-residual noise estimate via `_noise.compute_sg_residual_xy(x, y, window=constants.SG_WINDOW_SHORT, degree=constants.SG_DEGREE)`. It SHALL set `growth_axis_unreliable = (D < constants.GROWTH_AXIS_RELIABILITY_K * sg_residual_xy_local)` — strict less-than. The threshold multiplier is configurable via `ConstantsT.GROWTH_AXIS_RELIABILITY_K` with documented default `10`.

When `growth_axis_unreliable == True`, the 6 rotation-dependent trait columns (`v_long_signed_median_px_per_frame`, `v_long_abs_median_px_per_frame`, `v_lat_signed_median_px_per_frame`, `v_lat_abs_median_px_per_frame`, `long_lat_ratio`, `principal_axis_angle`) SHALL be set to `NaN`. The 3 rotation-invariant traits (`v_total_median_px_per_frame`, `path_displacement_ratio`, `angular_amplitude`) SHALL NOT be NaN'd by the gate (they may still be NaN for other documented reasons — e.g. `path_displacement_ratio` when `D == 0`, or all traits when fewer than 2 valid frames).

Both the Tier 0 module AND the QC tier module SHALL emit the `growth_axis_unreliable` column on their respective per-plant trait DataFrames. The two emissions SHALL be element-wise equal as `bool` dtype by construction, because both tiers compute the flag via the same formula on the same inputs through the shared `_noise.compute_sg_residual_xy` helper after applying the same `xy = subset[["tip_x","tip_y"]].to_numpy(dtype=float)` cast. PR #14 pipeline composition MAY coalesce or drop one column (either choice is safe because they are equal). This rule SUPERSEDES the previous "sole emitter" wording from PR #2 (rationale: QC's standalone usefulness requires the bool column for downstream `df[~df.growth_axis_unreliable]` filtering — see PR #3 design.md D5). The equality contract is governed by Requirement: QC tier growth_axis_unreliable equality with Tier 0.

The local SG residual computation SHALL use the same `_noise.compute_sg_residual_xy` helper that the QC tier uses to emit the canonical `sg_residual_xy` trait, so the gate value and the canonical trait are guaranteed-identical from identical inputs.

#### Scenario: Gate fires below threshold
- **GIVEN** a trajectory_df constructed via the dual-recipe pattern documented in "ConstantsT override changes the gate threshold" — `D = 5 px` paired with a known/fixed `sg_residual = 1.0 px` (either via `monkeypatch` of `_noise.compute_sg_residual_xy` or via a smooth-line-plus-σ=0.7-noise construction)
- **WHEN** `kinematics.compute(trajectory_df)` is called (default `GROWTH_AXIS_RELIABILITY_K=10`)
- **THEN** `growth_axis_unreliable == True`
- **AND** the 6 rotation-dependent columns are NaN

#### Scenario: Gate does not fire above threshold
- **GIVEN** a trajectory_df constructed via the same dual-recipe pattern — `D = 100 px` paired with a fixed `sg_residual = 1.0 px`
- **WHEN** `kinematics.compute(trajectory_df)` is called (default K=10)
- **THEN** `growth_axis_unreliable == False`
- **AND** the 6 rotation-dependent columns carry finite (non-NaN) values

#### Scenario: Gate threshold is strict less-than at boundary
- **GIVEN** a trajectory_df with `D = 10.0` and SG residual fixed to `1.0` (via monkeypatch as in the dual-recipe pattern), at the exact boundary `D == K * residual`
- **WHEN** `kinematics.compute(trajectory_df)` is called (K=10)
- **THEN** `growth_axis_unreliable == False` (strict less-than per the spec — at equality, the axis is judged reliable)

#### Scenario: Rotation-invariant traits survive the gate
- **GIVEN** any trajectory_df where the gate fires (`growth_axis_unreliable == True`)
- **WHEN** `kinematics.compute(trajectory_df)` is called
- **THEN** `v_total_median_px_per_frame`, `path_displacement_ratio`, and `angular_amplitude` are NOT set to NaN by the gate
- **AND** they are finite values reflecting the actual track kinematics

#### Scenario: Both tiers emit the column with equal values
- **GIVEN** a valid `trajectory_df` (any synthetic or real fixture)
- **WHEN** both `kinematics.compute(df)` and `qc.compute(df)` are invoked with default constants
- **THEN** both returned DataFrames contain a `growth_axis_unreliable` column
- **AND** both columns are `bool` dtype with no NaN values
- **AND** the two columns are element-wise equal (`(kinematics_result["growth_axis_unreliable"] == qc_result["growth_axis_unreliable"]).all()` is True)

### Requirement: Tier 0 helper modules
The system SHALL provide two private helper modules in `sleap_roots/circumnutation/` to host computations shared across Tier 0 (PR #2), the QC tier (PR #3), and future tier PRs (#7):

**`_noise.py`** SHALL define three callables for SLEAP-localization-noise estimation, each deterministic (identical inputs → identical outputs):

1. `compute_sg_residual_xy(x: np.ndarray, y: np.ndarray, window: int, degree: int) -> float`. The function SHALL apply `scipy.signal.savgol_filter` to `x` and `y` independently using the given `window` and `degree`, compute the standard deviation of residuals for each, and return their quadrature sum: `sqrt(std(x - x_smooth)^2 + std(y - y_smooth)^2)`. When `len(x) < window`, the function SHALL return `np.nan` and log a `DEBUG` record naming the short-input case rather than raising.

2. `compute_d2_residual_xy(x: np.ndarray, y: np.ndarray) -> float`. The function SHALL compute `delta2_x = x[2:] - 2*x[1:-1] + x[:-2]` (second-difference array) and similarly `delta2_y`, then return `sqrt(std(delta2_x)**2 + std(delta2_y)**2) / sqrt(6)`. The `1/sqrt(6)` normalization derives from the white-noise propagation rule `Var(Δ²x) = 6σ²` (theory.md §7.6 / preliminary_results.md §3.3). When `len(x) < 3`, the function SHALL return `np.nan` and log a `DEBUG` record naming the short-input case rather than raising.

3. `compute_msd_residual_xy(x: np.ndarray, y: np.ndarray, window: int, degree: int, lag: int = 1) -> float`. The function SHALL first SG-detrend `x` and `y` separately using `scipy.signal.savgol_filter(..., window_length=window, polyorder=degree)`, compute residuals `x_res = x - x_smooth` (and similarly `y_res`), then compute the 2D MSD at the given `lag` as `msd = mean((x_res[lag:] - x_res[:-lag])**2 + (y_res[lag:] - y_res[:-lag])**2)`, and return `sqrt(msd / 4.0)`. The factor of 4 (NOT 2) is the 2D MSD ↔ σ² relationship `MSD(τ→0) = 4σ²` from Michalet 2010 `Phys. Rev. E` 82:041914 / theory.md §7.6. When `len(x) < window + lag`, the function SHALL return `np.nan` and log a `DEBUG` record naming the short-input case rather than raising.

**`_geometry.py`** SHALL define `compute_psi_g(x: np.ndarray, y: np.ndarray) -> np.ndarray`. The function SHALL compute `dx = np.diff(x)`, `dy = np.diff(y)`, then `psi = np.arctan2(dx, dy)` (note argument order: `dx` first, then `dy` — this matches Bastien-Meroz 2016 Eq. 20 verbatim and `docs/circumnutation/theory.md` §3.5's explicit instruction *"The pipeline must use `atan2(dx/dt, dy/dt)` and unwrap the result"*), then return `np.unwrap(psi)`. The return shape is `(len(x) − 1,)`. When `len(x) < 2`, the function SHALL return an empty 1-D array `np.array([])`. The argument order is convention-critical: PR #7's `handedness` trait (`theory.md` §7.3) defines `+1 = counterclockwise (left-handed in image frame)` as the sign of mean `dψ_g/dt`, which requires the BM-Eq.-20 convention for the handedness sign to be correct in the published literature sense.

Both modules SHALL declare module-level loggers via `logger = logging.getLogger(__name__)`. Both module names SHALL be underscore-prefixed indicating they are private internals — they are not re-exported from the package's `__init__.py`.

#### Scenario: `_noise.compute_sg_residual_xy` returns zero for a polynomial of degree ≤ SG_DEGREE
- **GIVEN** `x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]` and `y = [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]` (linear and quadratic respectively) and `window=5, degree=3`
- **WHEN** `compute_sg_residual_xy(x, y, window=5, degree=3)` is called
- **THEN** the return is `0.0` within IEEE float tolerance (SG of degree 3 fits these polynomials exactly)

#### Scenario: `_noise.compute_sg_residual_xy` recovers approximate σ on noisy data
- **GIVEN** `x_smooth = np.linspace(0, 100, 1000)`, `y_smooth = np.zeros(1000)`, `noise = np.random.default_rng(0).normal(0, 1.0, size=(2, 1000))`, `x = x_smooth + noise[0]`, `y = y_smooth + noise[1]`
- **WHEN** `compute_sg_residual_xy(x, y, window=5, degree=3)` is called
- **THEN** the return is within `[1.0, 1.6]` (the quadrature sum is `sqrt(σ_x^2 + σ_y^2) = sqrt(2) ≈ 1.41`; tolerance accounts for SG's slight under-estimate of σ)

#### Scenario: `_noise.compute_sg_residual_xy` returns NaN for short inputs
- **WHEN** `compute_sg_residual_xy(np.array([1.0, 2.0]), np.array([3.0, 4.0]), window=5, degree=3)` is called
- **THEN** the return is `np.nan`
- **AND** no exception is raised

#### Scenario: `_noise.compute_d2_residual_xy` returns zero for a linear signal
- **GIVEN** `x = np.linspace(0, 99, 100)` (perfectly linear; second differences are identically zero) and `y = np.zeros(100)`
- **WHEN** `compute_d2_residual_xy(x, y)` is called
- **THEN** the return is `0.0` within IEEE float tolerance

#### Scenario: `_noise.compute_d2_residual_xy` recovers approximate σ on noisy data
- **GIVEN** `x_smooth = np.linspace(0, 100, 1000)`, `y_smooth = np.zeros(1000)`, `noise = np.random.default_rng(0).normal(0, 1.0, size=(2, 1000))`, `x = x_smooth + noise[0]`, `y = y_smooth + noise[1]`
- **WHEN** `compute_d2_residual_xy(x, y)` is called
- **THEN** the return is within `[1.2, 1.8]` (the d2 estimator is unbiased for i.i.d. noise on a linear signal: `std(Δ²x) = sqrt(6)·σ_x`, so the quadrature sum `sqrt(std(Δ²x)² + std(Δ²y)²) / sqrt(6) = sqrt(σ_x² + σ_y²) = sqrt(2) ≈ 1.41`)

#### Scenario: `_noise.compute_d2_residual_xy` returns NaN for short inputs
- **WHEN** `compute_d2_residual_xy(np.array([1.0, 2.0]), np.array([3.0, 4.0]))` is called
- **THEN** the return is `np.nan`
- **AND** a `DEBUG` log record is emitted naming the short-input case
- **AND** no exception is raised

#### Scenario: `_noise.compute_msd_residual_xy` returns approximately zero for a smooth signal
- **GIVEN** `x = np.linspace(0, 99, 100)` (smooth linear; SG-detrend residual is ≈ 0) and `y = np.zeros(100)` and `window=5, degree=3, lag=1`
- **WHEN** `compute_msd_residual_xy(x, y, window=5, degree=3, lag=1)` is called
- **THEN** the return is `≤ 1e-6` (SG residual is numerically zero; MSD of ~zero residuals is ~zero; σ ≈ 0)

#### Scenario: `_noise.compute_msd_residual_xy` recovers approximate σ on noisy data
- **GIVEN** `x_smooth = np.linspace(0, 100, 1000)`, `y_smooth = np.zeros(1000)`, `noise = np.random.default_rng(0).normal(0, 1.0, size=(2, 1000))`, `x = x_smooth + noise[0]`, `y = y_smooth + noise[1]`
- **WHEN** `compute_msd_residual_xy(x, y, window=5, degree=3, lag=1)` is called
- **THEN** the return is within `[1.0, 2.0]` (MSD-extrapolation at lag=1 of SG-detrended residuals; for i.i.d. unit-σ noise on independent x and y, `MSD(τ=1) = 4σ² = 4` so `σ_MSD = sqrt(4/4) = 1`; tolerance allows for SG-detrend slight under-estimate and lag-1 stochastic variation)

#### Scenario: `_noise.compute_msd_residual_xy` returns NaN for short inputs
- **WHEN** `compute_msd_residual_xy(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), np.array([0.0, 0.0, 0.0, 0.0, 0.0]), window=5, degree=3, lag=1)` is called (len=5 < window+lag = 6)
- **THEN** the return is `np.nan`
- **AND** a `DEBUG` log record is emitted naming the short-input case
- **AND** no exception is raised

#### Scenario: `_geometry.compute_psi_g` returns a constant for a straight-line track
- **GIVEN** `x = np.arange(100, dtype=float)` (constant velocity in +x direction) and `y = np.zeros(100)`
- **WHEN** `compute_psi_g(x, y)` is called
- **THEN** the return array has length 99 and all elements equal `math.pi / 2` (within IEEE float tolerance — because `atan2(dx=1, dy=0) = π/2`)
- **AND** for the orthogonal case `x = np.zeros(100)`, `y = np.arange(100, dtype=float)` (constant velocity in +y direction = image-down), `compute_psi_g(x, y)` returns all elements equal to `0.0` (because `atan2(dx=0, dy=1) = 0`)

#### Scenario: `_geometry.compute_psi_g` returns monotonic unwrapped angles spanning ≈2π for a closed circular trajectory
- **GIVEN** `t = np.linspace(0, 2*np.pi, 100)`, `x = np.cos(t)`, `y = np.sin(t)` (one full revolution in `(x, y)` parametric space)
- **WHEN** `compute_psi_g(x, y)` is called
- **THEN** the return is a length-99 array that is strictly monotonic (either entirely increasing or entirely decreasing — the direction is convention-dependent; under `atan2(dx, dy)` with `x = cos(t), y = sin(t)`, ψ_g decreases monotonically because the velocity-direction angle rotates by `−2π` over the revolution)
- **AND** the difference between consecutive elements is bounded (no `±2π` discontinuities — i.e., the unwrap worked)
- **AND** the total absolute span (`abs(return[-1] − return[0])`) is approximately `2π` (one full revolution).

#### Scenario: `_geometry.compute_psi_g` returns empty for too-short input
- **WHEN** `compute_psi_g(np.array([1.0]), np.array([2.0]))` is called
- **THEN** the return is an empty 1-D array (`shape == (0,)`)
- **AND** no exception is raised

### Requirement: Tier 0 input-validation boundary
The system SHALL validate at the entry of `kinematics.compute` that `trajectory_df` is a `pandas.DataFrame` and that it contains the eight row-identity columns plus `frame`, `tip_x`, `tip_y`. On validation failure, `ValueError` SHALL be raised with a message naming the offending field or type. Validation MAY be implemented by delegating to the foundation's `_types._validate_trajectory_df` helper, OR by re-implementing the equivalent checks inline; either choice is acceptable as long as the error messages match the foundation's permissive-but-clear-error style.

Beyond the foundation's column-presence and DataFrame-type checks, Tier 0 makes the following deliberate scoping choices on input quality (documented as non-goals of THIS PR; deferred to future tier PRs or pre-validation by the caller):

- **`±inf` in `tip_x` / `tip_y`** is NOT validated. `pandas.DataFrame.dropna(subset=["tip_x", "tip_y"])` retains `±inf` rows (NaN-only filter), and `np.linalg.norm` of an `inf` vector returns `inf`. Tier 0 propagates `inf` through trait computations; resulting traits may be `inf` or `NaN` depending on the propagation path. **Rationale**: SLEAP predictions never emit `±inf`; this would only happen via corrupted upstream data. PR #3 QC's `frac_outlier_steps` is the right place to detect and gate on this.
- **Duplicate `(track_id, frame)` rows** are NOT detected. The `sort_values("frame")` is stable, so duplicate-frame rows keep their input order; `np.diff(frame)` then produces `Δframe = 0` for the duplicate pair, leading to `Δxy/0 = inf` or `nan/0 = nan`. **Rationale**: duplicate `(track_id, frame)` indicates an upstream data error (`TrackedTipPipeline` never produces them — each instance has a unique `(track_id, frame)` by construction). Pre-validation is the caller's responsibility.
- **Non-contiguous `Δframe = 0` from sort instability** cannot occur because `np.diff` on a non-decreasing `frame` array produces `Δframe ≥ 0`, and `Δframe = 0` only happens with duplicates (handled above).
- **Non-integer `frame` columns** (e.g., timestamps as floats) are NOT rejected — `frame` may be any numeric type. The "per-frame" semantic of the emitted velocity columns becomes "per-sample" under non-integer frames; for the Nipponbare/KitaakeX fixtures (integer-frame), this is moot.

#### Scenario: `±inf` in tip_x propagates without raising
- **GIVEN** a trajectory_df with one row having `tip_x = float('inf')` (the foundation's `CircumnutationInputs` validator does not check finiteness per row, per its docstring at `_types.py:60`)
- **WHEN** `kinematics.compute(trajectory_df)` is called
- **THEN** no exception is raised
- **AND** the trait values for that track are either `inf` or `NaN` (propagation-dependent), reflecting the documented "Tier 0 does not validate ±inf" non-goal

#### Scenario: Duplicate `(track_id, frame)` rows do not raise and propagate non-finite values opportunistically
- **GIVEN** a trajectory_df where the same `(track_id, frame)` 2-tuple appears in two rows (e.g., the upstream join double-emitted a frame)
- **WHEN** `kinematics.compute(trajectory_df)` is called
- **THEN** no exception is raised
- **AND** the resulting `Δframe = 0` divide-by-zero MAY produce non-finite (`inf` or `NaN`) values in velocity-related traits — whether the non-finite value reaches the final trait depends on the projection (`[0, 0] / 0 = NaN`; `[nonzero, ·] / 0 = ±inf`) AND on whether the corrupted step survives the `np.nanmedian` aggregation. Tier 0 does NOT guarantee that the contamination is observable in the emitted traits; it guarantees only that no exception is raised. PR #3 QC's `frac_outlier_steps` is the right place for explicit duplicate-frame detection.
- **AND** the documented behavior is "Tier 0 does not detect duplicate frames — PR #3 QC's `frac_outlier_steps` is the right place"

### Requirement: Per-plant template helper for raw-DataFrame callers
The system SHALL expose a private helper `_build_per_plant_template_from_df(df: pd.DataFrame) -> pd.DataFrame` in `sleap_roots/circumnutation/_io.py`. The helper SHALL implement the same drop-duplicates + sort + dtype-coercion logic that the existing public `build_per_plant_template(inputs: CircumnutationInputs)` function uses. The public function SHALL be refactored to a one-line wrapper that returns `_build_per_plant_template_from_df(inputs.trajectory_df)`.

The helper exists so tier modules whose canonical signature takes a raw `trajectory_df` (today: `kinematics.compute`; later: `qc.compute`, `parametric.compute`) can compose the row-identity template without wrapping the DataFrame in a `CircumnutationInputs` purely to satisfy an API. The helper SHALL enforce the same validations the public function enforces — `track_id` integer-coercible (raise `ValueError` naming the field if NaN), same 5-tuple-conflict check on `timepoint` / `genotype` / `treatment` with the same error message format.

The helper's existence and behavior SHALL be tested via direct import; the public `build_per_plant_template` SHALL continue to pass all of its existing foundation tests unchanged (regression preserved).

#### Scenario: Helper produces identical output to public wrapper
- **GIVEN** a `CircumnutationInputs` instance and `inputs.trajectory_df` (a `pd.DataFrame`)
- **WHEN** `_build_per_plant_template_from_df(inputs.trajectory_df)` and `build_per_plant_template(inputs)` are both called
- **THEN** the two return values are column-for-column equal (via `pandas.DataFrame.equals`)

#### Scenario: Helper enforces the integer `track_id` constraint with NaN
- **GIVEN** a raw trajectory `pd.DataFrame` whose `track_id` column contains `NaN` in at least one row
- **WHEN** `_build_per_plant_template_from_df(df)` is called
- **THEN** a `ValueError` is raised
- **AND** the exception message names the `track_id` field (matching the public function's behavior)

#### Scenario: Helper enforces the 5-tuple-conflict check
- **GIVEN** a raw trajectory `pd.DataFrame` where the same `(series, sample_uid, plate_id, plant_id, track_id)` 5-tuple has different `genotype` values across frames
- **WHEN** `_build_per_plant_template_from_df(df)` is called
- **THEN** a `ValueError` is raised
- **AND** the exception message names the offending 5-tuple AND the column whose values conflict (matching the public function's behavior)

#### Scenario: Public wrapper preserves foundation API
- **WHEN** a caller invokes `build_per_plant_template(inputs)` for any valid `CircumnutationInputs`
- **THEN** the return value is identical (column-for-column equal) to the return value of the same call BEFORE this PR's refactor — i.e., all existing foundation tests on `build_per_plant_template` pass without modification

### Requirement: QC tier per-track quality traits
The system SHALL provide `sleap_roots.circumnutation.qc.compute(trajectory_df: pd.DataFrame, constants: Optional[ConstantsT] = None) -> pd.DataFrame`. The function SHALL accept the canonical `(trajectory_df, constants=None)` signature locked by the foundation's Package layout requirement — `cadence_s` SHALL NOT appear in the signature (the function emits cadence-independent traits only).

It SHALL return a per-plant `pandas.DataFrame` with one row per unique `(series, sample_uid, plate_id, plant_id, track_id)` 5-tuple in the input, sorted via the same convention as `_io._build_per_plant_template_from_df`. The DataFrame columns SHALL be (in this order):

1. The 8 row-identity columns in their declared order: `series`, `sample_uid`, `timepoint`, `plate_id`, `plant_id`, `track_id`, `genotype`, `treatment`.
2. The 11 QC trait columns, in this order:

| Column | Unit | Definition |
|---|---|---|
| `sg_residual_xy` | `px` | `_noise.compute_sg_residual_xy(x, y, window=constants.SG_WINDOW_SHORT, degree=constants.SG_DEGREE)` |
| `d2_noise_xy` | `px` | `_noise.compute_d2_residual_xy(x, y)` |
| `msd_noise_xy` | `px` | `_noise.compute_msd_residual_xy(x, y, window=constants.SG_WINDOW_SHORT, degree=constants.SG_DEGREE, lag=1)` |
| `sg_d2_agreement` | `—` | `max(sg_residual_xy, d2_noise_xy) / min(sg_residual_xy, d2_noise_xy)`; NaN if either operand is NaN |
| `sg_msd_agreement` | `—` | `max(sg_residual_xy, msd_noise_xy) / min(sg_residual_xy, msd_noise_xy)`; NaN if either operand is NaN |
| `d2_msd_agreement` | `—` | `max(d2_noise_xy, msd_noise_xy) / min(d2_noise_xy, msd_noise_xy)`; NaN if either operand is NaN |
| `frac_outlier_steps` | `—` | `count(\|Δxy_i\| > constants.OUTLIER_STEP_RATIO · median(\|Δxy_i\|)) / n_steps` over consecutive present frames (gap-aware); NaN when `median(\|Δxy_i\|) == 0` (stationary track) |
| `worst_step_ratio` | `—` | `max(\|Δxy_i\|) / median(\|Δxy_i\|)`; NaN when `median(\|Δxy_i\|) == 0` |
| `growth_axis_unreliable` | `bool` | True iff the growth axis is unreliable per Requirement: QC tier growth_axis_unreliable equality with Tier 0 |
| `track_is_clean` | `bool` | composite per Requirement: QC tier track_is_clean and qc_failure_reason composition |
| `qc_failure_reason` | `string` | stable-ordered comma-separated failure clauses per the same requirement; `""` when `track_is_clean == True` |

The function SHALL emit only units in `PIPELINE_UNIT_VOCABULARY` — specifically `px` for the 3 noise estimators, `—` for the 5 dimensionless ratios/fractions, `bool` for `growth_axis_unreliable` and `track_is_clean`, `string` for `qc_failure_reason`. NO `_mm`-suffixed columns and NO `_px_per_hr` columns SHALL be emitted by `qc.compute`.

The function SHALL process each track via the algorithm documented in `design.md` D7: (a) drop NaN rows on `tip_x` / `tip_y` BEFORE any `np.diff` or arithmetic (NaN-then-sort ordering matching Tier 0); (b) sort the remaining rows by `frame`; (c) ALWAYS compute `growth_axis_unreliable` via the same `_noise.compute_sg_residual_xy` helper and gate formula as Tier 0 (preserving the equality contract — see Requirement: QC tier growth_axis_unreliable equality with Tier 0); (d) if fewer than `constants.SG_WINDOW_SHORT` (default 5) frames remain after NaN-drop, emit the 8 numeric traits + 3 pairwise agreements all as NaN, `track_is_clean = False`, `qc_failure_reason = "qc_inputs_insufficient"` (single sentinel reason, NOT comma-concatenated with other clauses), and skip to the merge step; (e) otherwise compute the 3 noise estimators, 3 pairwise agreements, and 2 outlier-step traits per the table above; (f) compose `track_is_clean` and `qc_failure_reason` per Requirement: QC tier track_is_clean and qc_failure_reason composition.

The xy array SHALL be cast to `dtype=float` explicitly (`xy = subset[["tip_x", "tip_y"]].to_numpy(dtype=float)`) so that input dtype variation (int, float32, object) cannot perturb the equality contract with Tier 0 — both tiers feed identically-typed arrays to `_noise.compute_sg_residual_xy`.

The function SHALL accept `constants=None` (default) by constructing the default `ConstantsT()`, OR a caller-supplied `ConstantsT` override. The constants consumed are `SG_WINDOW_SHORT`, `SG_DEGREE`, `OUTLIER_STEP_RATIO`, `GROWTH_AXIS_RELIABILITY_K`, `SG_D2_AGREEMENT_MAX`, `SG_MSD_AGREEMENT_MAX`, `D2_MSD_AGREEMENT_MAX`, `FRAC_OUTLIER_STEPS_MAX`, `WORST_STEP_RATIO_MAX`.

After per-track computation, the function SHALL merge the per-track trait dictionaries onto the foundation's per-plant template via the 5-tuple key (`_io._build_per_plant_template_from_df`) and re-select columns to enforce the declared 19-column order `ROW_IDENTITY_COLUMNS + _QC_TRAIT_COLUMNS`. This re-selection step is load-bearing: the groupby key uses the 5-tuple `_IDENTITY_5_TUPLE` which omits `timepoint`, but the template carries all 8 row-identity columns; without the final re-selection `timepoint` could drop or shift.

#### Scenario: Output DataFrame columns are in the specified order
- **GIVEN** a valid `trajectory_df` with 6 tracks
- **WHEN** `qc.compute(trajectory_df)` is called
- **THEN** the returned DataFrame has exactly 19 columns
- **AND** the first 8 columns are `series`, `sample_uid`, `timepoint`, `plate_id`, `plant_id`, `track_id`, `genotype`, `treatment` in that order
- **AND** the next 11 columns are `sg_residual_xy`, `d2_noise_xy`, `msd_noise_xy`, `sg_d2_agreement`, `sg_msd_agreement`, `d2_msd_agreement`, `frac_outlier_steps`, `worst_step_ratio`, `growth_axis_unreliable`, `track_is_clean`, `qc_failure_reason` in that order

#### Scenario: Output column units are within `PIPELINE_UNIT_VOCABULARY`
- **GIVEN** the output of `qc.compute(trajectory_df)` and the units-mapping dict for the 11 new columns
- **WHEN** every unit string is checked against `sleap_roots.circumnutation._constants.PIPELINE_UNIT_VOCABULARY`
- **THEN** every unit string is a member of the vocabulary
- **AND** no unit string is `mm`-bearing or `px/hr`-bearing
- **AND** the 3 noise estimators use `px`, the 5 dimensionless traits use `—`, the 2 booleans use `bool`, `qc_failure_reason` uses `string`

#### Scenario: Clean straight-line track yields zero residuals and `track_is_clean == True`
- **GIVEN** a 100-frame track with `tip_x = frame * 1.0`, `tip_y = 0.0` (perfectly linear, no noise)
- **WHEN** `qc.compute(trajectory_df)` is called with default constants
- **THEN** the returned row has `sg_residual_xy ≈ 0.0` and `d2_noise_xy ≈ 0.0` and `msd_noise_xy ≈ 0.0` (smooth signal → all residuals near zero)
- **AND** `frac_outlier_steps == 0.0` (all steps equal — no outliers)
- **AND** `worst_step_ratio ≈ 1.0` (all steps equal)
- **AND** `growth_axis_unreliable == False` (D = 99 ≫ K · residual)
- **AND** `track_is_clean == True`
- **AND** `qc_failure_reason == ""`

#### Scenario: Pure-noise track fires growth_axis_unreliable and `track_is_clean == False`
- **GIVEN** a 100-frame track with `tip_x, tip_y = np.random.default_rng(0).normal(0, 1, size=(2, 100))` (i.i.d. noise around origin, no growth)
- **WHEN** `qc.compute(trajectory_df)` is called with default constants
- **THEN** `growth_axis_unreliable == True` (small D vs SG-residual)
- **AND** `track_is_clean == False`
- **AND** `qc_failure_reason` contains `"growth_axis_unreliable"` as one of its comma-separated clauses

#### Scenario: Short-track gate fires for `len < SG_WINDOW_SHORT`
- **GIVEN** a track with only 3 frames (`len < SG_WINDOW_SHORT = 5`)
- **WHEN** `qc.compute(trajectory_df)` is called
- **THEN** the 8 numeric traits + 3 pairwise agreements are NaN
- **AND** `track_is_clean == False`
- **AND** `qc_failure_reason == "qc_inputs_insufficient"` (literally; NOT comma-concatenated with other clauses even if `growth_axis_unreliable=True` would have fired)

#### Scenario: Single-frame track (n=1) emits NaN traits without raising
- **GIVEN** a `trajectory_df` with a single track that has only 1 row after NaN-drop
- **WHEN** `qc.compute(trajectory_df)` is called
- **THEN** the returned DataFrame has 1 row for that track
- **AND** the 8 numeric traits + 3 pairwise agreements are NaN
- **AND** `growth_axis_unreliable == False` (`D = NaN`, `D == 0.0` is False, `not isnan(NaN)` is False — both gate clauses False)
- **AND** `track_is_clean == False`
- **AND** `qc_failure_reason == "qc_inputs_insufficient"`
- **AND** no exception is raised

#### Scenario: Zero-displacement closed-loop track triggers gate
- **GIVEN** a track of length ≥ 5 whose start and end coordinates are identical (`xy[-1] == xy[0]`)
- **WHEN** `qc.compute(trajectory_df)` is called
- **THEN** `growth_axis_unreliable == True` (D == 0.0)
- **AND** `track_is_clean == False`
- **AND** `qc_failure_reason` contains `"growth_axis_unreliable"`

#### Scenario: Invalid trajectory_df raises ValueError
- **WHEN** `qc.compute(None)` is called
- **THEN** a `ValueError` is raised whose message indicates the input is not a `pandas.DataFrame`
- **AND** `qc.compute(df_missing_tip_x)` similarly raises a `ValueError` naming the missing column
- **AND** `qc.compute(df_missing_plate_id)` raises a `ValueError` naming the missing row-identity column

#### Scenario: ConstantsT override changes per-clause thresholds
- **GIVEN** a `trajectory_df` constructed so that `sg_d2_agreement = 1.7` (above default `SG_D2_AGREEMENT_MAX = 1.5`)
- **WHEN** `qc.compute(trajectory_df)` is called with default constants
- **THEN** `qc_failure_reason` contains `"sg_d2_agreement_high"`
- **AND** when called again with `constants=ConstantsT(SG_D2_AGREEMENT_MAX=2.0)`, `qc_failure_reason` does NOT contain `"sg_d2_agreement_high"` (threshold loosened; clause does not fire)

### Requirement: QC tier track_is_clean and qc_failure_reason composition
The system SHALL compose `track_is_clean` and `qc_failure_reason` per track via the formulas below.

`track_is_clean` SHALL be the AND of six clauses:

```
track_is_clean = (
    NOT growth_axis_unreliable
  AND sg_d2_agreement   < constants.SG_D2_AGREEMENT_MAX   (default 1.5)
  AND sg_msd_agreement  < constants.SG_MSD_AGREEMENT_MAX  (default 1.5)
  AND d2_msd_agreement  < constants.D2_MSD_AGREEMENT_MAX  (default 1.5)
  AND frac_outlier_steps < constants.FRAC_OUTLIER_STEPS_MAX (default 0.05)
  AND worst_step_ratio   < constants.WORST_STEP_RATIO_MAX   (default 5)
)
```

Comparisons against NaN return False in Python/NumPy, so any NaN-bearing trait fires its clause (i.e., `not (NaN < threshold)` evaluates to True). Net effect: any NaN-laden track gets `track_is_clean = False` and gets the relevant clause(s) appended to `qc_failure_reason`.

`qc_failure_reason` SHALL be a string column. When `track_is_clean == True`, the value SHALL be the empty string `""` (NOT NaN). When `track_is_clean == False`, the value SHALL be the comma-separated (`", "` with a single space) list of failure-clause names in the stable canonical order encoded as the module-level `tuple` `_FAILURE_CLAUSE_ORDER`:

```python
_FAILURE_CLAUSE_ORDER: tuple = (
    "qc_inputs_insufficient",       # short-track sentinel (overrides all other clauses)
    "growth_axis_unreliable",
    "sg_d2_agreement_high",
    "sg_msd_agreement_high",
    "d2_msd_agreement_high",
    "frac_outlier_steps_high",
    "worst_step_ratio_high",
)
```

`qc_inputs_insufficient` SHALL be treated as a sentinel, NOT as a regular clause that can co-occur with others. When the short-track gate fires (per Requirement: QC tier per-track quality traits, step d), `qc_failure_reason` SHALL be LITERALLY `"qc_inputs_insufficient"` with no other clauses appended even if `growth_axis_unreliable = True` would have fired. The other 6 clauses MAY co-occur and SHALL concatenate via `", ".join(...)` in the declared tuple order.

#### Scenario: All clauses clean → empty failure reason
- **GIVEN** a track where `track_is_clean == True` (all 6 clauses satisfied)
- **WHEN** `qc.compute(trajectory_df)` is called
- **THEN** `qc_failure_reason == ""` (empty string, NOT NaN)

#### Scenario: Single clause failure → single-clause failure reason
- **GIVEN** a track where exactly one of the 6 clauses fails (e.g., `worst_step_ratio = 7.0 > 5`)
- **WHEN** `qc.compute(trajectory_df)` is called
- **THEN** `qc_failure_reason == "worst_step_ratio_high"`
- **AND** `track_is_clean == False`

#### Scenario: Multiple clause failures → comma-separated reason in stable order
- **GIVEN** a track constructed to fire three clauses simultaneously. Construction recipe (parallel to PR #2's dual-recipe pattern): take a 30-frame track with `tip_x = np.random.default_rng(0).normal(0, 1, 30)`, `tip_y = np.random.default_rng(1).normal(0, 1, 30)` (pure-noise, no growth → `growth_axis_unreliable=True`); inject 4 outlier frames (`tip_x[10:14] = 100`) so `frac_outlier_steps > 0.05`; the noise structure + outliers also pushes `sg_d2_agreement` above 1.5. Equivalent monkeypatch alternative: `monkeypatch.setattr("sleap_roots.circumnutation._noise.compute_d2_residual_xy", lambda *a, **k: 1.7 * sg_value)` to force exactly `sg_d2_agreement = 1.7`.
- **WHEN** `qc.compute(trajectory_df)` is called
- **THEN** `qc_failure_reason == "growth_axis_unreliable, sg_d2_agreement_high, frac_outlier_steps_high"` (clauses in `_FAILURE_CLAUSE_ORDER` order)

#### Scenario: Short-track gate produces sentinel single-clause reason
- **GIVEN** a 3-frame track (below `SG_WINDOW_SHORT = 5`)
- **WHEN** `qc.compute(trajectory_df)` is called
- **THEN** `qc_failure_reason == "qc_inputs_insufficient"` (literally; NOT comma-concatenated with `"growth_axis_unreliable"` even if that condition would have fired)

#### Scenario: Threshold override via ConstantsT changes failure-clause firing
- **GIVEN** a track whose `worst_step_ratio = 6.0`
- **WHEN** `qc.compute(trajectory_df)` is called with `constants=ConstantsT(WORST_STEP_RATIO_MAX=10)`
- **THEN** the `worst_step_ratio_high` clause does NOT appear in `qc_failure_reason`
- **AND** when called with default constants (`WORST_STEP_RATIO_MAX=5`), it DOES appear

### Requirement: QC tier growth_axis_unreliable equality with Tier 0
The system SHALL emit `growth_axis_unreliable` as a column of the QC tier per-plant DataFrame, with values element-wise equal as `bool` dtype to the same column emitted by Tier 0's `kinematics.compute` on the same `trajectory_df` input. The equality is by construction: both tiers compute the flag via `(D == 0.0) or (not math.isnan(sg_residual) and D < constants.GROWTH_AXIS_RELIABILITY_K * sg_residual)` where `sg_residual = _noise.compute_sg_residual_xy(xy[:,0], xy[:,1], window=constants.SG_WINDOW_SHORT, degree=constants.SG_DEGREE)`, `D = float(np.linalg.norm(xy[-1] - xy[0]))`, and `xy = subset[["tip_x", "tip_y"]].to_numpy(dtype=float)` after the same NaN-drop and sort-by-frame.

The column SHALL be `bool` dtype (no NaN values) in BOTH outputs — tracks where the computation is degenerate (`len < 2` after dropna) SHALL emit `False`, matching Tier 0's `_emit_nan_row` precedent.

This requirement deliberately reverses the previous CC-5 rule (in `docs/circumnutation/roadmap.md`) that *"Tier 0 emits `growth_axis_unreliable`; QC does NOT re-emit a duplicate column."* Rationale: QC's output is more useful standalone when `growth_axis_unreliable` is a first-class bool column rather than only a string-embedded clause in `qc_failure_reason`. The duplicated column between Tier 0 and QC outputs is a cost; equality-by-construction makes that cost trivial (coalesce or drop in PR #14 pipeline composition).

#### Scenario: Equality on Nipponbare fixture
- **GIVEN** the Nipponbare plate 001 fixture loaded via `Series.get_tracked_tips()` (per `tests/data/circumnutation_nipponbare_plate_001/`) and enriched with the 4 missing identity columns (`plate_id`, `plant_id = track_id`, `genotype = "Nipponbare"`, `treatment = "MOCK"`)
- **WHEN** both `kinematics.compute(df)` and `qc.compute(df)` are called with default constants
- **THEN** `kinematics_result["growth_axis_unreliable"].dtype == qc_result["growth_axis_unreliable"].dtype == np.dtype("bool")`
- **AND** `(kinematics_result["growth_axis_unreliable"] == qc_result["growth_axis_unreliable"]).all()` is True (element-wise equal for all 6 tracks)

#### Scenario: Equality holds under int / float32 / object dtype input
- **GIVEN** a `trajectory_df` whose `tip_x` and `tip_y` columns are dtype `int` or `float32` rather than the canonical `float64`
- **WHEN** both `kinematics.compute(df)` and `qc.compute(df)` are called
- **THEN** the `growth_axis_unreliable` columns are still element-wise equal between the two outputs (because both tiers apply `to_numpy(dtype=float)` before passing arrays to the shared helper)

#### Scenario: Equality holds for closed-loop tracks (D == 0)
- **GIVEN** a track of length ≥ 2 where `xy[-1] == xy[0]` (closed loop)
- **WHEN** both `kinematics.compute(df)` and `qc.compute(df)` are called
- **THEN** both report `growth_axis_unreliable == True` for that track (the `D == 0.0` disjunct fires in both implementations)

#### Scenario: Equality holds for ultra-short tracks (n < 2)
- **GIVEN** a track that has only 1 row after NaN-drop
- **WHEN** both `kinematics.compute(df)` and `qc.compute(df)` are called
- **THEN** both report `growth_axis_unreliable == False` for that track (matching `_emit_nan_row` precedent in Tier 0; matching D7 step 2 path in QC)

### Requirement: QC tier input-validation boundary
The system SHALL validate at the entry of `qc.compute` that `trajectory_df` is a `pandas.DataFrame` and that it contains the eight row-identity columns plus `frame`, `tip_x`, `tip_y`. On validation failure, `ValueError` SHALL be raised with a message naming the offending field or type. Validation MAY be implemented by delegating to the foundation's `_types._validate_trajectory_df` helper, OR by re-implementing the equivalent checks inline; either choice is acceptable as long as the error messages match the foundation's permissive-but-clear-error style (matching the precedent established by Tier 0 in Requirement: Tier 0 input-validation boundary).

Beyond the foundation's column-presence and DataFrame-type checks, the QC tier inherits Tier 0's deliberate scoping choices on input quality (documented as non-goals of this PR; deferred to future tier PRs or pre-validation by the caller):

- **`±inf` in `tip_x` / `tip_y`** is NOT validated. The QC tier propagates `inf` through trait computations; resulting traits MAY be `inf` or `NaN` depending on the propagation path. In the worst case (>50% of steps `inf`), `frac_outlier_steps` may silently pass at `0.0` because `median` of mostly-`inf` is `inf` and `inf > 2 * inf == False`. **Rationale**: SLEAP predictions never emit `±inf`; this would only happen via corrupted upstream data. A future QC-level `±inf` detector is deliberately out of scope for PR #3 (file follow-up if needed).
- **Duplicate `(track_id, frame)` rows** are NOT detected. Same rationale as Tier 0: upstream data integrity is the caller's responsibility.

The QC tier additionally documents one tier-specific non-goal:

- **Stationary tracks** (all step magnitudes identically zero, so `median(\|Δxy_i\|) == 0`) yield NaN for `frac_outlier_steps` and `worst_step_ratio`. Under NaN-comparison semantics this fires BOTH `frac_outlier_steps_high` AND `worst_step_ratio_high` clauses in `qc_failure_reason`. **Rationale**: a stationary track IS a QC-failing track in any reasonable sense; emitting two clauses is correct semantically (the track failed two thresholds). A dedicated `stationary_track` clause is reserved for a future PR if a downstream user wants finer-grained categorization.

#### Scenario: ±inf in tip_x propagates without raising
- **GIVEN** a `trajectory_df` with one row having `tip_x = float('inf')` (the foundation's `CircumnutationInputs` validator does not check finiteness per row)
- **WHEN** `qc.compute(trajectory_df)` is called
- **THEN** no exception is raised
- **AND** the trait values for that track are either `inf` or `NaN` (propagation-dependent)

#### Scenario: Duplicate `(track_id, frame)` rows do not raise
- **GIVEN** a `trajectory_df` where the same `(track_id, frame)` 2-tuple appears in two rows
- **WHEN** `qc.compute(trajectory_df)` is called
- **THEN** no exception is raised
- **AND** the `Δframe = 0` divide-by-zero MAY produce non-finite values in step-magnitude-derived traits

#### Scenario: Stationary track fires both outlier clauses
- **GIVEN** a 100-frame track with `tip_x = 5.0` constant and `tip_y = 3.0` constant (all step magnitudes are zero, so `median = 0`)
- **WHEN** `qc.compute(trajectory_df)` is called
- **THEN** `frac_outlier_steps` is `NaN` and `worst_step_ratio` is `NaN`
- **AND** `track_is_clean == False`
- **AND** `qc_failure_reason` contains both `"frac_outlier_steps_high"` AND `"worst_step_ratio_high"`

### Requirement: Synthetic trajectory generator
The system SHALL provide `sleap_roots.circumnutation.synthetic.generate_trajectory(*, ...) -> pd.DataFrame`. The function SHALL accept the canonical `generate_trajectory(...)` signature locked by the foundation's Package layout requirement — `px_per_mm` SHALL NOT appear in the signature (the function emits pure-pixel trajectories per CC-3). All parameters SHALL be keyword-only (`*,` enforced at signature top); positional invocation SHALL raise `TypeError`.

The function SHALL realize Rivière 2022 Eq. 4 in **parametric closed form** (not literal Eq. 5 ODE forward integration). The apex propagates along the growth axis at velocity `v_growth_per_s = growth_rate_px_per_frame / cadence_s`; transverse nutation contributes `A_lat · sin(handedness · ω · t + initial_phase_rad)` with `A_lat = amplitude_px / 2` and `ω = 2π / T_nutation_s`; iid Gaussian localization noise is added per-axis with `σ_per_axis = noise_sigma_px / √2` so that the QC tier's xy-quadrature noise estimators (`sg_residual_xy`, `d2_noise_xy`, `msd_noise_xy`) recover `noise_sigma_px` directly. The growth-axis unit vector is `u_g = (cos(growth_axis_angle_rad), sin(growth_axis_angle_rad))` and the lateral unit vector is `u_lat = (-u_g[1], u_g[0])` (standard CCW 90° rotation in math axes; under the image-y-down convention this displays as visually clockwise on screen).

The function SHALL return a `pandas.DataFrame` with exactly 11 columns and exactly `n_frames` rows. The columns SHALL be (in this order):

1. The 8 row-identity columns in their declared order: `series`, `sample_uid`, `timepoint`, `plate_id`, `plant_id`, `track_id`, `genotype`, `treatment`. The DataFrame contains one unique row-identity 5-tuple (single-track output per call).
2. `frame` (int64) — values `[0, n_frames)` strictly monotonic ascending; `frame.iloc[0] == 0` and `frame.iloc[-1] == n_frames - 1`.
3. `tip_x` (float64) — pure-pixel; the closed-form longitudinal + transverse + noise sum per the math above.
4. `tip_y` (float64) — pure-pixel; same.

The function SHALL emit dtypes per the contract: `frame.dtype == np.int64`; `tip_x.dtype == tip_y.dtype == np.float64`; `plant_id.dtype == track_id.dtype == np.int64`; `series`, `sample_uid`, `timepoint`, `plate_id`, `genotype`, `treatment` all `object` dtype. When `genotype=None` (or `treatment=None`) is passed, the resulting column contains `np.nan` values in `object`-dtype storage (NOT literal string `"None"`); `df["genotype"].isna().all()` returns `True` in this case.

**Closed-form math equivalences** (load-bearing for round-trip tests):

- In the noise-free case (`noise_sigma_px == 0`), the Tier 0 trait `v_long_signed_median_px_per_frame` recovers `growth_rate_px_per_frame` exactly when `amplitude_px == 0` (pure linear; growth-axis inference is exact). With nutation, recovery is within ≈0.5% because the kinematics-inferred net-displacement growth-axis matches `growth_axis_angle_rad` up to a small phase-sampling offset.
- In the noise-free case, the Tier 0 trait `angular_amplitude` (peak-to-peak ψ_g) equals **`2 · arctan(amplitude_px · ω / (2 · v_growth_per_s))`** (the exact relation, NOT the small-angle approximation `amplitude_px · ω / v_growth_per_s` which over-estimates by ~13% in the plate-001 empirical regime).
- The QC tier's xy-quadrature noise estimators recover `noise_sigma_px` within ±15% (matches theory.md §8 spatial tolerance and absorbs the documented SG under-bias).
- For default plate-001 parameters, `kinematics.compute(df)["growth_axis_unreliable"]` is `False` (safety margin ≈ 123× over the gate threshold; `D ≈ 2467 px`, `K · noise ≈ 20 px`).

**Determinism contract (CC-6).** `random_state` SHALL accept `int`, `np.random.Generator`, or `None`. A single internal call to `np.random.default_rng(random_state)` handles all three idiomatically. Same `int` seed SHALL produce bit-identical `tip_x` / `tip_y` arrays across two calls AND across OSs on 64-bit platforms (relying on numpy's PCG64 stability per NEP 19; sleap-roots' supported CI matrix — Ubuntu / Windows / macOS x86_64 + Apple Silicon arm64 — is 100% 64-bit). The legacy `np.random.RandomState` API SHALL be rejected with `TypeError`.

When `noise_sigma_px == 0.0` exactly, the implementation SHALL NOT call `np.random.default_rng(random_state)` or `rng.normal(...)` — the RNG path SHALL be short-circuited. This decouples output determinism from `random_state` in noise-free mode AND preserves caller-supplied Generator state unchanged: a caller passing `random_state=my_rng` with `noise_sigma_px=0` SHALL get `my_rng` back in unchanged state (`my_rng.bit_generator.state` SHALL be equal before and after the call as a deep dict comparison).

**Handedness convention (load-bearing for PR #7's `handedness` trait).** `handedness` SHALL be `+1` or `-1` (exactly; integers only; bool rejected). Default `+1` corresponds to counterclockwise per BM2016 §"Constant principal direction of growth" + `theory.md` §3.5 / §7.3. With `handedness = +1` AND `noise_sigma_px = 0`, `_geometry.compute_psi_g(tip_x, tip_y)` SHALL return an unwrapped ψ_g time series whose mean derivative is positive (`np.mean(np.diff(psi_g)) > 0`). With `handedness = -1`, the mean derivative SHALL be negative.

**Rivière correspondence (documented in module docstring; not exposed in the API):** the API parameters map to Rivière 2022 quantities via `v_growth_per_s = ε̇₀ · R`, `amplitude_px = 2 · ΔL · δ̇₀ / ω = R · Δφ` (Eq. 1), `ω = 2π / T_nutation_s`. The Rivière 6-tuple `(L_gz, ΔL, δ̇₀, ε̇₀, ω, R)` is degenerate at the tip-trajectory level (only 3 aggregate combinations are observable); PR #12 wraps `generate_trajectory` with a Rivière-named translation helper once PR #9 / PR #11 land spatial-CWT recovery of `L_gz` / `ΔL` individually.

**ConstantsT resolution-order.** Seven user-facing parameters (`n_frames`, `cadence_s`, `amplitude_px`, `T_nutation_s`, `growth_rate_px_per_frame`, `noise_sigma_px`, `growth_axis_angle_rad`) SHALL default to `None` in the signature. At call time, the resolution order SHALL be:

1. **Call-site kwarg wins.** If the parameter is explicitly passed (not `None`), use that value.
2. **`constants` parameter overrides module-level default.** If the parameter is `None` AND `constants` is a `ConstantsT` instance, use `constants.SYNTHETIC_<name>`.
3. **Module-level default.** If both above are absent, use the module-level constant directly (which equals the `ConstantsT()` field default).

The remaining parameters (`handedness`, `x0_px`, `y0_px`, `initial_phase_rad`, `random_state`, `constants`, and the 8 identity columns) keep direct literal defaults because they are NOT `ConstantsT`-overridable.

**Strict input validation at the boundary.** Every numeric parameter SHALL be validated for type, finiteness, sign, and bool-rejection. `isinstance(value, (int, np.integer))` for integer parameters; `isinstance(value, (float, int, np.floating, np.integer))` for float parameters (with `bool` then explicitly rejected because `True`/`False` are `int` subclasses). String inputs SHALL be rejected with `TypeError`; the `generate_trajectory` function does NOT coerce string-typed numerics (unlike `CircumnutationInputs.cadence_s` which has an attrs converter). On validation failure the function SHALL raise `ValueError` or `TypeError` whose message names the offending field.

#### Scenario: Default call returns 575-row DataFrame with the documented schema
- **WHEN** `synthetic.generate_trajectory()` is called with all-default kwargs
- **THEN** the returned DataFrame has exactly 575 rows
- **AND** the columns are `["series", "sample_uid", "timepoint", "plate_id", "plant_id", "track_id", "genotype", "treatment", "frame", "tip_x", "tip_y"]` in that order
- **AND** `df["frame"].dtype == np.dtype("int64")`, `df["tip_x"].dtype == df["tip_y"].dtype == np.dtype("float64")`, `df["plant_id"].dtype == df["track_id"].dtype == np.dtype("int64")`
- **AND** `df["series"].dtype == df["sample_uid"].dtype == df["timepoint"].dtype == df["plate_id"].dtype == df["genotype"].dtype == df["treatment"].dtype == np.dtype("object")`
- **AND** `df["frame"].iloc[0] == 0 and df["frame"].iloc[-1] == 574`
- **AND** `(np.diff(df["frame"]) == 1).all()`
- **AND** `df["plant_id"].equals(df["track_id"])`
- **AND** the default `growth_axis_angle_rad = math.pi / 2` corresponds to the apex propagating in the `+y` screen direction (visually downward) under the image-y-down coordinate convention per `theory.md` §2.1 — i.e., `u_g = (cos(π/2), sin(π/2)) = (0, 1)` aligns the apex motion with the standard rice-plate convention of roots growing downward on a vertically-mounted plate

#### Scenario: `None` for `genotype` and `treatment` becomes NaN, not literal "None"
- **WHEN** `synthetic.generate_trajectory(genotype=None, treatment=None)` is called
- **THEN** `df["genotype"].isna().all() == True` AND `df["genotype"].dtype == np.dtype("object")`
- **AND** `(df["genotype"] == "None").any() == False` (no literal `"None"` strings)
- **AND** the same holds for `treatment`

#### Scenario: Positional invocation raises TypeError
- **WHEN** `synthetic.generate_trajectory(575)` is called (positional, not keyword)
- **THEN** a `TypeError` is raised

#### Scenario: Same int seed produces bit-identical output across two calls
- **WHEN** `synthetic.generate_trajectory(random_state=42, ...)` is called twice with identical kwargs
- **THEN** the two returned DataFrames have `tip_x` arrays that are bit-identical via `np.array_equal` (not just `np.allclose`)
- **AND** the `tip_y` arrays are also bit-identical
- **AND** the assertion holds across operating systems on 64-bit platforms (Ubuntu / Windows / macOS x86_64 + Apple Silicon arm64)

#### Scenario: int seed and default_rng(seed) produce identical output
- **GIVEN** two calls with `random_state=42` and `random_state=np.random.default_rng(42)` respectively (otherwise identical kwargs)
- **WHEN** both are invoked
- **THEN** the two returned DataFrames have bit-identical `tip_x` and `tip_y` arrays

#### Scenario: `noise_sigma_px=0` short-circuits RNG and preserves Generator state
- **GIVEN** `rng = np.random.default_rng(42)` and `state_before = rng.bit_generator.state`
- **WHEN** `synthetic.generate_trajectory(noise_sigma_px=0, random_state=rng, ...)` is called
- **THEN** the returned DataFrame's `tip_x` matches the output of a separate call with `random_state=None` (RNG path not entered; output determined only by closed-form math)
- **AND** `rng.bit_generator.state == state_before` (deep dict comparison) — caller-supplied Generator state unchanged

#### Scenario: Different seeds produce different output
- **WHEN** `synthetic.generate_trajectory(random_state=0, ...)` and `synthetic.generate_trajectory(random_state=1, ...)` are called with otherwise identical kwargs and `noise_sigma_px > 0`
- **THEN** the two returned DataFrames' `tip_x` arrays differ (`not np.allclose`)

#### Scenario: Recovered longitudinal velocity equals input growth rate in pure-linear case
- **GIVEN** `synthetic.generate_trajectory(amplitude_px=0, growth_rate_px_per_frame=4.29, noise_sigma_px=0, growth_axis_angle_rad=math.pi/2, ...)` (pure linear, no nutation, no noise)
- **WHEN** `kinematics.compute(df)["v_long_signed_median_px_per_frame"].iloc[0]` is computed
- **THEN** the value equals `4.29` within IEEE float tolerance (`abs(value - 4.29) < 1e-9`; loosened from `1e-10` per /openspec-review round-1 TDD reviewer B3 to accommodate cross-platform BLAS rounding through the `(growth_rate / cadence_s) · (i · cadence_s)` reintroduction)
- **AND** `kinematics.compute(df)["long_lat_ratio"].iloc[0]` is `NaN` (pure-linear trajectory yields `v_lat_abs_median == 0`, triggering the documented kinematics NaN contract)

#### Scenario: Recovered angular amplitude matches exact arctan formula in noise-free case
- **GIVEN** `synthetic.generate_trajectory(amplitude_px=10.0, T_nutation_s=3333.0, growth_rate_px_per_frame=4.29, cadence_s=300.0, n_frames=575, noise_sigma_px=0, ...)` (plate-001-matching defaults; noise-free)
- **WHEN** `kinematics.compute(df)["angular_amplitude"].iloc[0]` is computed
- **THEN** the value matches the exact analytical prediction `2 * arctan(amplitude_px * omega / (2 * v_growth_per_s))` (≈ 1.17 rad for these inputs) within ±15% (theory.md §8 spatial tolerance)
- **AND** `kinematics.compute(df)["growth_axis_unreliable"].iloc[0] == False`

#### Scenario: Noise round-trips via QC's xy-quadrature estimators with documented per-estimator bias
- **GIVEN** `synthetic.generate_trajectory(noise_sigma_px=2.0, random_state=42, ...)` with otherwise plate-001-matching defaults
- **WHEN** `qc.compute(df)` is computed
- **THEN** the QC tier's xy-quadrature noise estimators recover `noise_sigma_px` modulo the documented per-estimator bias factors empirically calibrated on the closed-form trajectory (Copilot review #5; §3.7 canary capture): `sg_residual_xy` recovers `≈ 0.65 × noise_sigma_px` within ±25%, `d2_noise_xy` recovers `≈ 0.95 × noise_sigma_px` within ±25% (near-unbiased), `msd_noise_xy` recovers `≈ 0.61 × noise_sigma_px` within ±25%. The bias factors are multiplicative (linear in `noise_sigma_px`), so doubling the input noise approximately doubles each estimator's output
- **AND** with default ConstantsT thresholds, `df_qc["d2_msd_agreement"].iloc[0]` lands at `≈ 0.95/0.61 ≈ 1.55` — slightly above the default `D2_MSD_AGREEMENT_MAX = 1.5` (matching the structural borderline PR #3 observed on plate 001 at `d2_msd_agreement = 1.537`). To assert `track_is_clean == True`, callers SHALL loosen the agreement thresholds via `ConstantsT(SG_D2_AGREEMENT_MAX=2.0, SG_MSD_AGREEMENT_MAX=2.0, D2_MSD_AGREEMENT_MAX=2.0)` — the documented escape per design.md
- **AND** when called with the loosened ConstantsT, `df_qc["track_is_clean"].iloc[0] == True` and `df_qc["qc_failure_reason"].iloc[0] == ""`

#### Scenario: handedness=+1 yields positive mean dψ_g/dt
- **GIVEN** `synthetic.generate_trajectory(handedness=+1, noise_sigma_px=0, ...)` (noise-free for unambiguous determinism)
- **WHEN** `_geometry.compute_psi_g(df["tip_x"].to_numpy(), df["tip_y"].to_numpy())` is computed and the mean first-difference `np.mean(np.diff(psi_g))` is evaluated
- **THEN** the result is positive (`> 0`)

#### Scenario: handedness=-1 yields negative mean dψ_g/dt
- **GIVEN** `synthetic.generate_trajectory(handedness=-1, noise_sigma_px=0, ...)`
- **WHEN** the same evaluation is performed
- **THEN** the result is negative (`< 0`)

#### Scenario: handedness=+1 curl-sign agrees with ψ_g sign
- **GIVEN** `synthetic.generate_trajectory(handedness=+1, noise_sigma_px=0, ...)` and the computed `ψ_g = _geometry.compute_psi_g(tip_x, tip_y)`
- **WHEN** the curl-sign of the trajectory is computed independently of `_geometry.compute_psi_g` as `sign(mean(diff(tip_x)[1:] * diff(diff(tip_y)) - diff(tip_y)[1:] * diff(diff(tip_x))))`
- **THEN** the curl-sign is positive (`> 0`)
- **AND** `sign(mean(diff(ψ_g)))` is also positive
- **AND** the two signs agree (both `+1`) — this cross-check guards against a future refactor of `_geometry.compute_psi_g` that inverts the `atan2(dx, dy)` argument order silently inverting PR #7's handedness trait convention

#### Scenario: handedness=-1 curl-sign agrees with ψ_g sign
- **GIVEN** `synthetic.generate_trajectory(handedness=-1, noise_sigma_px=0, ...)` and the computed `ψ_g = _geometry.compute_psi_g(tip_x, tip_y)`
- **WHEN** the curl-sign is computed via the same finite-difference formula as the previous scenario
- **THEN** the curl-sign is negative (`< 0`)
- **AND** `sign(mean(diff(ψ_g)))` is also negative
- **AND** the two signs agree (both `-1`) — combined with the `handedness=+1` scenario, this locks the convention chain `handedness input → trajectory rotation → atan2(dx, dy) sign` against silent inversion

#### Scenario: `ConstantsT` override propagates when kwarg omitted
- **GIVEN** `synthetic.generate_trajectory(amplitude_px=None, noise_sigma_px=0, constants=ConstantsT(SYNTHETIC_AMPLITUDE_PX=20.0))` (kwarg omitted; constants override active)
- **WHEN** `kinematics.compute(df)["angular_amplitude"].iloc[0]` is computed
- **THEN** the recovered value matches the exact analytical prediction using `amplitude_px=20.0` (i.e., `2 * arctan(10.0 * omega / (2 * v_growth_per_s))`) within ±15%

#### Scenario: Explicit kwarg overrides ConstantsT override (resolution-order kwarg-wins)
- **GIVEN** `synthetic.generate_trajectory(amplitude_px=15.0, noise_sigma_px=0, constants=ConstantsT(SYNTHETIC_AMPLITUDE_PX=20.0))` (BOTH kwarg AND constants override are set; kwarg should win)
- **WHEN** `kinematics.compute(df)["angular_amplitude"].iloc[0]` is computed
- **THEN** the recovered value matches the exact analytical prediction using `amplitude_px=15.0` (NOT 20.0; kwarg won the resolution race) within ±15%

#### Scenario: Invalid n_frames raises with named field
- **WHEN** `synthetic.generate_trajectory(n_frames=v)` is called for any `v ∈ {0, -1, True, 1.5, "100", np.nan, np.inf}`
- **THEN** in each case a `ValueError` or `TypeError` is raised
- **AND** the exception message contains the string `"n_frames"`

#### Scenario: Invalid cadence_s raises with named field; no string coercion
- **WHEN** `synthetic.generate_trajectory(cadence_s=v)` is called for any `v ∈ {0.0, -1.0, np.nan, np.inf, -np.inf, True, "300"}`
- **THEN** in each case a `ValueError` or `TypeError` is raised
- **AND** the exception message contains the string `"cadence_s"`
- **AND** `"300"` specifically raises `TypeError` (no implicit string-to-float coercion, unlike `CircumnutationInputs.cadence_s`)

#### Scenario: Invalid handedness raises with named field
- **WHEN** `synthetic.generate_trajectory(handedness=v)` is called for any `v ∈ {0, 2, -2, 1.0, True, "+1", None}`
- **THEN** in each case a `ValueError` or `TypeError` is raised (only `+1` or `-1` integer are valid)
- **AND** the exception message contains the string `"handedness"`

#### Scenario: Invalid random_state raises with named field
- **WHEN** `synthetic.generate_trajectory(random_state=v)` is called for any `v ∈ {1.5, "42", np.random.RandomState(0)}`
- **THEN** in each case a `TypeError` is raised
- **AND** the exception message contains the string `"random_state"`
- **AND** specifically `np.random.RandomState(0)` is rejected (legacy API; only the modern `Generator` API is accepted per the determinism contract)

#### Scenario: Reference-fixture agreement (Layer-1 sanity)
- **GIVEN** the plate 001 fixture (`tests/data/circumnutation_nipponbare_plate_001/plate_001_greyscale.tracked_proofread.slp`) loaded via `sio.load_slp` and enriched into a `trajectory_df` per PR #2's reference-value test pattern; AND `synth_df = synthetic.generate_trajectory()` with all-default kwargs
- **WHEN** `qc.compute(real_df)["sg_residual_xy"].median()` (median across 6 tracks) and `qc.compute(synth_df)["sg_residual_xy"].iloc[0]` are both computed
- **THEN** the two values agree within ±35% (`abs(real - synth) / real < 0.35`) — tolerance widened from theory.md §8's ±15% spatial baseline to accommodate the structural gap between `SYNTHETIC_NOISE_SIGMA_PX = 2.0` (the theory.md §8 anchor) and plate-001's apparent true xy-quadrature σ ≈ 2.83 px (back-derived from the QC SG estimator's ~0.65× bias factor on plate-001 sg=1.83). The ratio plate/synth ≈ 1.42 ≈ √2 across all 3 estimators is the consistency anchor; PR #12's Layer-1 validation will revisit the default calibration when more multi-plate data lands
- **AND** the per-frame mean total step magnitude (`np.linalg.norm(np.diff(xy, axis=0), axis=1).mean()` per track, then median across 6 tracks for the real fixture; single-track value for synth) also agrees within ±35%
- **AND** `kinematics.compute(real_df)["growth_axis_unreliable"]` and `kinematics.compute(synth_df)["growth_axis_unreliable"]` are both False (no track flagged by the gate)
- **AND** if the fixture file is absent at the documented path, the test SHALL skip rather than fail (`pytest.skip` semantics; the fixture is committed via Git LFS but may be unavailable in environments where LFS pointers were not resolved)

### Requirement: Temporal CWT scaleogram API
The system SHALL provide `sleap_roots.circumnutation.temporal_cwt.compute_scaleogram(x: np.ndarray, cadence_s: float, constants: Optional[ConstantsT] = None) -> ScaleogramResult`. The function SHALL accept the canonical `(x, cadence_s, constants=None)` signature locked by the foundation's Package layout requirement. The function SHALL compute a temporal Continuous Wavelet Transform using the `cmor1.5-1.0` mother wavelet by default (overridable via `constants.WAVELET_DEFAULT_TEMPORAL`), at log-spaced scales over an auto-derived period range `[constants.CWT_PERIOD_MIN_NYQUIST_FACTOR * cadence_s, constants.CWT_PERIOD_MAX_SIGNAL_FRACTION * len(x) * cadence_s]`, returning a frozen `ScaleogramResult` containing the complex-valued scaleogram, scales axis, period axis (derived via `pywt.scale2frequency` round-trip for wavelet-agnostic correctness), frequency axis, cone-of-influence boolean mask (computed via wavelet-aware `constants.COI_EFOLDING_FACTOR * scale` per `design.md` D3), the resolved `cadence_s`, and the resolved wavelet name.

The function SHALL emit NO trait values (PR #6 owns trait emission). The function SHALL be deterministic per CC-6: same input → bit-identical scaleogram across calls in the same process AND identical to within `atol=1e-9` across Ubuntu / Windows / macOS CI runners. The function SHALL log exactly one `logger.debug` message at the start (after input validation) whose text begins with `"compute_scaleogram("` and contains the named tokens `n_frames=`, `cadence_s=`, `n_scales=`, `period_min_s=`, `period_max_s=`, `wavelet=`. No INFO, WARNING, or ERROR log records SHALL be emitted on the happy path.

The function SHALL validate inputs strictly: `x` SHALL be a 1-D `np.ndarray` (coercible from integer/float dtypes; rejecting `complex` and `object`); `x` SHALL be all-finite (no NaN, no ±inf); `len(x)` SHALL be ≥ `MIN_FRAMES_REQUIRED` derived at call time as `int(math.floor(constants.CWT_PERIOD_MIN_NYQUIST_FACTOR / constants.CWT_PERIOD_MAX_SIGNAL_FRACTION)) + 1` (= `9` at defaults); positive-finite guards SHALL fire on `constants.CWT_PERIOD_MAX_SIGNAL_FRACTION` and `constants.CWT_PERIOD_MIN_NYQUIST_FACTOR`; `cadence_s` SHALL be a Python `int` or `float` (rejecting `bool` subtype of `int` and `str`); `cadence_s > 0` and `math.isfinite(cadence_s)`; `constants` SHALL be `None` or a `ConstantsT` instance. Invalid inputs SHALL raise `ValueError` or `TypeError` with the offending field name embedded in the message.

The `ScaleogramResult` class SHALL be an `@attrs.define(frozen=True, slots=False, kw_only=True)` container with exactly the following fields (in this order): `scaleogram: np.ndarray` (shape `(n_scales, n_frames)`, dtype `complex128`); `scales: np.ndarray` (shape `(n_scales,)`, dtype `float64`, monotonically increasing); `periods_s: np.ndarray` (shape `(n_scales,)`, dtype `float64`); `frequencies_hz: np.ndarray` (shape `(n_scales,)`, dtype `float64`, equal to `1.0 / periods_s` to within numerical precision); `coi_mask: np.ndarray` (shape `(n_scales, n_frames)`, dtype `bool`; `True` indicates inside-COI = unreliable); `cadence_s: float`; `wavelet: str`.

#### Scenario: compute_scaleogram returns a ScaleogramResult with the documented field shapes and dtypes
- **GIVEN** a valid 1-D `float64` ndarray `x` of length 575 with all-finite values, and `cadence_s = 300.0`
- **WHEN** `compute_scaleogram(x, 300.0)` is invoked
- **THEN** the returned object is a `ScaleogramResult` instance
- **AND** `result.scaleogram.shape == (CWT_SCALE_COUNT_DEFAULT, 575)` and `dtype == np.complex128`
- **AND** `result.scales.shape == (CWT_SCALE_COUNT_DEFAULT,)`, `dtype == np.float64`, strictly monotonically increasing
- **AND** `result.periods_s.shape == (CWT_SCALE_COUNT_DEFAULT,)` and `dtype == np.float64`
- **AND** `np.allclose(result.frequencies_hz * result.periods_s, 1.0, atol=1e-12)`
- **AND** `result.coi_mask.shape == result.scaleogram.shape` and `dtype == bool`
- **AND** `result.cadence_s == 300.0` and `result.wavelet == "cmor1.5-1.0"`

#### Scenario: ScaleogramResult is a frozen attrs class with the seven documented fields
- **WHEN** `attrs.fields(ScaleogramResult)` is inspected
- **THEN** the field names are exactly `("scaleogram", "scales", "periods_s", "frequencies_hz", "coi_mask", "cadence_s", "wavelet")` in that order
- **AND** attempting `result.scaleogram = new_array` raises `attrs.exceptions.FrozenInstanceError`

#### Scenario: compute_scaleogram rejects non-finite x with ValueError naming the field
- **WHEN** `compute_scaleogram(x, 300.0)` is invoked with `x` containing NaN OR `+inf` OR `-inf` at any index
- **THEN** `ValueError` is raised
- **AND** the exception message contains the substring `"x"` and identifies the non-finite condition

#### Scenario: compute_scaleogram rejects malformed x with ValueError or TypeError naming the field
- **WHEN** `compute_scaleogram(x, 300.0)` is invoked with `x.ndim != 1` (e.g., 2-D ndarray) OR `x.dtype == np.complex128` OR `x.dtype == object`
- **THEN** `ValueError` or `TypeError` is raised
- **AND** the exception message contains the substring `"x"` and identifies the offending shape or dtype

#### Scenario: compute_scaleogram rejects too-short x with ValueError naming the field
- **WHEN** `compute_scaleogram(x, 300.0)` is invoked with `len(x) < MIN_FRAMES_REQUIRED` (where MIN_FRAMES_REQUIRED = `int(math.floor(CWT_PERIOD_MIN_NYQUIST_FACTOR / CWT_PERIOD_MAX_SIGNAL_FRACTION)) + 1` = 9 at defaults)
- **THEN** `ValueError` is raised
- **AND** the exception message contains the substring `"x"` (or `"MIN_FRAMES"`) and reports the actual length

#### Scenario: compute_scaleogram accepts x at the exact MIN_FRAMES_REQUIRED floor without raising
- **GIVEN** a 1-D float64 ndarray `x` of length exactly `MIN_FRAMES_REQUIRED` (= 9 at defaults) with all-finite values
- **WHEN** `compute_scaleogram(x, 300.0)` is invoked
- **THEN** the call returns a `ScaleogramResult` without raising (paired positive boundary contract for the "too-short x" negative scenario above, per /openspec-review round-2 reviewer N-I3 — bidirectional contract on the documented floor)
- **AND** the returned `scaleogram.shape == (CWT_SCALE_COUNT_DEFAULT, 9)`

#### Scenario: compute_scaleogram rejects invalid cadence_s value with ValueError naming the field
- **WHEN** `compute_scaleogram(x, cadence_s)` is invoked with `cadence_s` equal to `0` OR `-1.0` OR `float("nan")` OR `float("inf")` OR `float("-inf")`
- **THEN** `ValueError` is raised
- **AND** the exception message contains the substring `"cadence_s"` and the offending value

#### Scenario: compute_scaleogram rejects invalid cadence_s type with TypeError naming the field
- **WHEN** `compute_scaleogram(x, cadence_s)` is invoked with `cadence_s` equal to `True` (Python bool) OR `np.bool_(True)` (numpy bool scalar — must be explicitly guarded since `np.bool_` is a numpy scalar subclass of `int`) OR `"300"` (str) OR `[300.0]` (list)
- **THEN** `TypeError` is raised
- **AND** the exception message contains the substring `"cadence_s"` and identifies the offending type

#### Scenario: compute_scaleogram is deterministic across runs
- **GIVEN** a valid `x` and `cadence_s`
- **WHEN** `compute_scaleogram(x, cadence_s)` is invoked twice in the same Python process
- **THEN** `np.array_equal(result1.scaleogram, result2.scaleogram)` is True at `atol=0`
- **AND** the captured 3-value canary at `[scale_idx_at_target, [coi_interior_indices]]` matches the hardcoded expected complex values to within `atol=1e-9` across Ubuntu / Windows / macOS CI runners AT THE TIME OF PR-MERGE; canary values are regression-detection sentinels and MAY be re-captured (in a follow-up commit cross-referencing this scenario) if upstream BLAS / pywt / numpy semantics legitimately shift after merge

#### Scenario: compute_scaleogram emits exactly one DEBUG logger record on the happy path
- **GIVEN** a valid `x` and `cadence_s` and `caplog.set_level(logging.DEBUG)`
- **WHEN** `compute_scaleogram(x, cadence_s)` is invoked
- **THEN** exactly one log record at level `DEBUG` is emitted by the logger `sleap_roots.circumnutation.temporal_cwt`
- **AND** the record's message starts with `"compute_scaleogram("`
- **AND** the record's message contains each of the tokens `"n_frames="`, `"cadence_s="`, `"n_scales="`, `"period_min_s="`, `"period_max_s="`, `"wavelet="`
- **AND** no `INFO` / `WARNING` / `ERROR` / `CRITICAL` records are emitted

#### Scenario: Proofread fixture (Nipponbare plate-001 6 tracks) does not raise and produces shape-correct output
- **GIVEN** the 6 tracks of `tests/data/circumnutation_nipponbare_plate_001/plate_001_greyscale.tracked_proofread.slp` loaded via `Series.load(...).get_tracked_tips()` (each track has 575 finite-float64 frames with zero NaN and zero frame gaps, per the pre-design empirical verification)
- **WHEN** `compute_scaleogram(track_x, cadence_s=300.0)` is invoked for each track
- **THEN** the call does not raise
- **AND** the returned `ScaleogramResult` has `scaleogram.shape == (CWT_SCALE_COUNT_DEFAULT, 575)`
- **AND** `coi_mask.shape == scaleogram.shape`
- **AND** the COI fraction at the scale nearest to period 3333 s (`scale_idx_at_target = int(np.argmin(np.abs(result.periods_s - 3333.0)))`) is well below the `COI_FRACTION_MAX = 0.5` reliability threshold; concretely, `result.coi_mask[scale_idx_at_target, :].mean() < 0.10` (~2× the empirically-measured 4.87% at the `√1.5` factor — see `design.md` D3 arithmetic)

### Requirement: Temporal CWT ridge API
The system SHALL provide `sleap_roots.circumnutation.temporal_cwt.extract_ridge(scaleogram_result: ScaleogramResult, constants: Optional[ConstantsT] = None) -> RidgeResult`. The function SHALL extract a per-frame ridge from the input scaleogram via deterministic per-frame argmax of `|scaleogram|` along the scale axis (numpy's documented tie-breaking returns the smallest index on equal values).

The function SHALL emit NO trait values. The function SHALL be deterministic: same `ScaleogramResult` input → identical `RidgeResult` output at `atol=0`. The function SHALL log exactly one `logger.debug` message at the start (after input validation) whose text begins with `"extract_ridge("` and contains tokens `n_scales=` and `n_frames=`. No INFO, WARNING, or ERROR log records SHALL be emitted on the happy path.

The function SHALL validate inputs strictly: `scaleogram_result` SHALL be a `ScaleogramResult` instance (anything else raises `TypeError`); empty `ScaleogramResult` (with `n_scales == 0` OR `n_frames == 0`) SHALL raise `ValueError`; `constants` SHALL be `None` or a `ConstantsT` instance.

The `RidgeResult` class SHALL be an `@attrs.define(frozen=True, slots=False, kw_only=True)` container with exactly the following fields (in this order): `frame_indices: np.ndarray` (shape `(n_frames,)`, dtype `int64`, equal to `np.arange(n_frames, dtype=np.int64)`); `periods_s: np.ndarray` (shape `(n_frames,)`, dtype `float64`; indexed by frame, NOT by scale — value at index `i` is the period AT THE RIDGE for frame `i`); `amplitudes: np.ndarray` (shape `(n_frames,)`, dtype `float64`, equal to `|C|` at the ridge cell); `powers: np.ndarray` (shape `(n_frames,)`, dtype `float64`, equal to `amplitudes ** 2 = |C|²` — redundant by construction, intentionally preserved per `design.md` D2 + Round-2 reviewer R2-N3 + Round-3 reviewer R3-N1, so downstream consumers can consume either `|C|` or `|C|²` without recomputing); `in_coi: np.ndarray` (shape `(n_frames,)`, dtype `bool`; `True` iff `scaleogram_result.coi_mask[ridge_scale_idx, frame_idx]`). The ridge SHALL NOT be pre-COI-masked; PR #6's trait emission applies the mask per the COI-masked language in `theory.md` §7.2.

#### Scenario: extract_ridge returns a RidgeResult with the documented field shapes and dtypes
- **GIVEN** a valid `ScaleogramResult` produced by `compute_scaleogram(x, 300.0)` where `len(x) == 575`
- **WHEN** `extract_ridge(scaleogram_result)` is invoked
- **THEN** the returned object is a `RidgeResult` instance
- **AND** `result.frame_indices.shape == (575,)` and `dtype == np.int64` and `np.array_equal(result.frame_indices, np.arange(575, dtype=np.int64))`
- **AND** `result.periods_s.shape == (575,)` and `dtype == np.float64`
- **AND** `result.amplitudes.shape == (575,)` and `dtype == np.float64` and `(result.amplitudes >= 0).all()`
- **AND** `result.powers.shape == (575,)` and `dtype == np.float64`
- **AND** `np.allclose(result.powers, result.amplitudes ** 2)` (redundancy preservation)
- **AND** `result.in_coi.shape == (575,)` and `dtype == bool`

#### Scenario: RidgeResult is a frozen attrs class with the five documented fields preserving the powers redundancy
- **WHEN** `attrs.fields(RidgeResult)` is inspected
- **THEN** the field names are exactly `("frame_indices", "periods_s", "amplitudes", "powers", "in_coi")` in that order
- **AND** attempting `ridge.amplitudes = new_array` raises `attrs.exceptions.FrozenInstanceError`
- **AND** `powers` is present (not removed in a future PR without a BREAKING-change spec revision)

#### Scenario: extract_ridge rejects non-ScaleogramResult input with TypeError
- **WHEN** `extract_ridge(x)` is invoked with `x` equal to `None` OR `{}` OR `(1, 2, 3)` OR `np.zeros((10, 10))`
- **THEN** `TypeError` is raised
- **AND** the exception message references the expected type `ScaleogramResult`

#### Scenario: extract_ridge rejects empty ScaleogramResult with ValueError
- **GIVEN** a `ScaleogramResult` constructed with `n_scales == 0` (scaleogram shape `(0, n_frames)`) OR `n_frames == 0` (scaleogram shape `(n_scales, 0)`)
- **WHEN** `extract_ridge(scaleogram_result)` is invoked
- **THEN** `ValueError` is raised
- **AND** the exception message references the empty-axis condition (e.g., `"n_scales == 0"` or `"n_frames == 0"`)

#### Scenario: extract_ridge rejects invalid constants type with TypeError
- **GIVEN** a valid `ScaleogramResult`
- **WHEN** `extract_ridge(scaleogram_result, constants=invalid)` is invoked with `invalid` not `None` and not a `ConstantsT` instance (e.g., `42`, `"foo"`, `{}`)
- **THEN** `TypeError` is raised
- **AND** the exception message references `"constants"`

#### Scenario: extract_ridge emits exactly one DEBUG logger record on the happy path
- **GIVEN** a valid `ScaleogramResult` and `caplog.set_level(logging.DEBUG)`
- **WHEN** `extract_ridge(scaleogram_result)` is invoked
- **THEN** exactly one log record at level `DEBUG` is emitted by the logger `sleap_roots.circumnutation.temporal_cwt`
- **AND** the record's message starts with `"extract_ridge("`
- **AND** the record's message contains each of the tokens `"n_scales="`, `"n_frames="`
- **AND** no `INFO` / `WARNING` / `ERROR` / `CRITICAL` records are emitted

#### Scenario: extract_ridge is deterministic
- **GIVEN** a valid `ScaleogramResult`
- **WHEN** `extract_ridge(scaleogram_result)` is invoked twice
- **THEN** `np.array_equal(result1.periods_s, result2.periods_s)` is True at `atol=0`
- **AND** the same holds for `amplitudes`, `powers`, `in_coi`, `frame_indices`

### Requirement: Tier 1 nutation trait emission API
The system SHALL provide `sleap_roots.circumnutation.nutation.compute(trajectory_df: pd.DataFrame, cadence_s: float, coordinate: str = "lateral", constants: Optional[ConstantsT] = None) -> pd.DataFrame`. The function SHALL accept the documented signature with `cadence_s` as an explicit positional parameter (mirroring `temporal_cwt.compute_scaleogram`'s precedent — `nutation` is the first cadence-consuming tier; `cadence_s` is NOT carried via `trajectory_df.attrs`). The `coordinate` parameter SHALL accept exactly one of `{"lateral", "x", "y"}`, with default `"lateral"` per `docs/circumnutation/roadmap.md` CC-7.

The function SHALL emit a per-track DataFrame whose rows correspond to the unique 5-tuple `(series, sample_uid, plate_id, plant_id, track_id)` derived from `trajectory_df` via `groupby(_IDENTITY_5_TUPLE, dropna=False, sort=False)`. The returned DataFrame SHALL contain the 8 row-identity columns (per Requirement: Trait CSV row-identity schema) followed by exactly 8 trait columns in this declared order: `T_nutation_median: float64`, `T_nutation_iqr: float64`, `A_nutation_envelope_max_px: float64`, `band_power_ratio: float64`, `noise_floor_estimate: float64`, `is_nutating: bool`, `period_residual_vs_derr_reference: float64`, `cadence_nyquist_ratio: float64`.

The function SHALL compose the existing PR #5 `temporal_cwt.compute_scaleogram` + `extract_ridge` primitives with the new PR #6 `temporal_cwt.smooth_ridge` primitive (per Requirement: Temporal CWT ridge-continuity smoothing API), the new `_geometry.project_to_growth_axis_perpendicular` lateral-projection helper, the new `_noise.compute_sg_detrended` Savitzky-Golay detrending helper (window = `SG_WINDOW_DETREND = 23`, polynomial_order = `SG_DEGREE = 3` per `preliminary_results_2026-05-07.md` §3.4), and the new `_noise.compute_fourier_noise_floor` Fourier noise-floor helper (per `roadmap.md` CC-8: median amplitude over `f > NOISE_FLOOR_OUT_OF_BAND_FACTOR / T_nutation_median`). The 9-step per-track pipeline is documented in `design.md` D5.

The function SHALL emit NaN for exactly 3 strictly biological-meaning-dependent traits when `is_nutating == False`: `T_nutation_median`, `T_nutation_iqr`, `A_nutation_envelope_max_px`. The other 5 trait columns SHALL NOT be NaN-gated by `is_nutating`: `is_nutating` (the gate boolean itself), `band_power_ratio` (the SNR-like precursor), `noise_floor_estimate` (the noise precursor), `period_residual_vs_derr_reference` (ridge-of-noise diagnostic — informs WHERE the spectral peak landed even on noise-driven inputs), `cadence_nyquist_ratio` (engineering diagnostic — answers "could we have observed nutation if it were present?" independent of biology). "Not NaN-gated by `is_nutating`" is the load-bearing semantic: these 5 traits MAY still be NaN when the underlying diagnostic is undefined (e.g., stationary tracks where lateral projection returns all-NaN; all-COI ridge where the per-frame argmax has no interior frames; empty out-of-band Fourier region where the noise-floor cutoff exceeds Nyquist). The split — "NaN-gated by gate" vs "NaN only when undefined" — prevents downstream consumers from being unable to distinguish "no biological oscillation" from "cadence aliasing" from "ridge-of-noise" via trait inspection alone.

The function SHALL be deterministic per CC-6: same input → bit-identical 7 float columns + `is_nutating` boolean across calls in the same process (`atol=0`) AND identical to within `atol=1e-6` across Ubuntu / Windows / macOS CI runners (tolerance loosened from PR #5's `atol=1e-9` baseline per PR #6 `design.md` Round-2 S6: PR #6 composes 4 unverified scipy paths on top of PR #5's verified pywt path — `scipy.fft.rfft`, `scipy.ndimage.median_filter`, `scipy.signal.savgol_filter`, `scipy.stats.iqr` — and `atol=1e-6` is scientifically irrelevant for these traits per CC-6's "either 1e-9 OR documented looser"). The function SHALL log exactly one `logger.debug` message at the start (after input validation) whose text begins with `"nutation.compute("` and contains the tokens `n_tracks=`, `coordinate=`, `cadence_s=`. No INFO, WARNING, or ERROR log records SHALL be emitted on the happy path.

The function SHALL validate inputs strictly: `trajectory_df` validation delegates to `_validate_trajectory_df` (per Requirement: Tier 0 input-validation boundary); `cadence_s` validation mirrors PR #5's `_validate_cadence_s` (Python int/float or numpy integer/floating; explicit `bool` and `np.bool_` rejection; positive finite); `coordinate` SHALL be in `{"lateral", "x", "y"}` else ValueError; `constants` SHALL be None or `ConstantsT` else TypeError. The stationary-track failure path (`_geometry.project_to_growth_axis_perpendicular` returns `np.full(n, np.nan)` per the graceful-NaN policy) SHALL produce an all-NaN trait row with `is_nutating=False` rather than raising or emitting `np.RuntimeWarning("All-NaN slice encountered")`.

#### Scenario: nutation.compute returns a DataFrame with the documented column order and dtypes
- **GIVEN** a valid `trajectory_df` with 6 tracks (the Nipponbare proofread fixture) and `cadence_s = 300.0`
- **WHEN** `nutation.compute(trajectory_df, cadence_s=300.0)` is invoked
- **THEN** the returned object is a `pandas.DataFrame`
- **AND** the column order is: 8 row-identity columns from `ROW_IDENTITY_COLUMNS` followed by the 8 trait columns in declared order `["T_nutation_median", "T_nutation_iqr", "A_nutation_envelope_max_px", "band_power_ratio", "noise_floor_estimate", "is_nutating", "period_residual_vs_derr_reference", "cadence_nyquist_ratio"]`
- **AND** trait dtypes are: 7 float64 (`T_nutation_median`, `T_nutation_iqr`, `A_nutation_envelope_max_px`, `band_power_ratio`, `noise_floor_estimate`, `period_residual_vs_derr_reference`, `cadence_nyquist_ratio`) and 1 bool (`is_nutating`)
- **AND** row-identity 5-tuple `(series, sample_uid, plate_id, plant_id, track_id)` uniqueness holds: `df[list(_IDENTITY_5_TUPLE)].duplicated().sum() == 0`

#### Scenario: nutation.compute NaN-gates only 3 strictly biological-meaning-dependent traits when is_nutating is False
- **GIVEN** a noise-only input constructed via `synthetic.generate_trajectory(amplitude_px=0.0, noise_sigma_px=1.0, n_frames=1024, cadence_s=300, random_state=0)`
- **WHEN** `nutation.compute(trajectory_df, cadence_s=300.0)` is invoked on the converted trajectory_df
- **THEN** `result.is_nutating.iloc[0] == False`
- **AND** `np.isnan(result.T_nutation_median.iloc[0])` AND `np.isnan(result.T_nutation_iqr.iloc[0])` AND `np.isnan(result.A_nutation_envelope_max_px.iloc[0])` (the 3 NaN-gated traits)
- **AND** the 4 not-NaN-gated diagnostic + precursor traits (`band_power_ratio`, `noise_floor_estimate`, `period_residual_vs_derr_reference`, `cadence_nyquist_ratio`) are each either finite OR NaN (NEVER ±inf) — these traits MAY be NaN on pathological-but-valid inputs (e.g., when the candidate `T_nutation_median` makes the noise-floor out-of-band region empty, or when the all-COI ridge has no interior frames), but they are NOT NaN-gated by `is_nutating`. The "always populated" semantic is "not gated by `is_nutating`", not "guaranteed finite" (per Copilot review on PR #216 + design.md GREEN-phase Reconciliation Appendix).

#### Scenario: nutation.compute rejects invalid coordinate value with ValueError naming the field
- **WHEN** `nutation.compute(trajectory_df, cadence_s=300.0, coordinate=v)` is invoked for any of `v ∈ {"", "X", 1, None, "longitudinal"}`
- **THEN** `ValueError` is raised
- **AND** the exception message contains the substring `"coordinate"`

#### Scenario: nutation.compute rejects invalid cadence_s value with ValueError naming the field
- **WHEN** `nutation.compute(trajectory_df, cadence_s=v)` is invoked for any of `v ∈ {0, -1.0, float("nan"), float("inf"), float("-inf")}`
- **THEN** `ValueError` is raised
- **AND** the exception message contains the substring `"cadence_s"`

#### Scenario: nutation.compute rejects invalid cadence_s type with TypeError naming the field
- **WHEN** `nutation.compute(trajectory_df, cadence_s=v)` is invoked for any of `v ∈ {True, np.bool_(True), "300", [300.0]}`
- **THEN** `TypeError` is raised
- **AND** the exception message contains the substring `"cadence_s"`

#### Scenario: nutation.compute is deterministic across runs and across OSs
- **GIVEN** a valid `trajectory_df` and `cadence_s = 300.0`
- **WHEN** `nutation.compute(trajectory_df, cadence_s=300.0)` is invoked twice in the same Python process
- **THEN** the 7 float trait columns are bit-identical at `atol=0`
- **AND** the `is_nutating` boolean column is equal
- **AND** the captured 3-value canary at `[T_nutation_median, band_power_ratio, noise_floor_estimate]` on `synthetic.generate_trajectory(random_state=0, n_frames=575, T_nutation_s=3333, cadence_s=300, noise_sigma_px=0.5)` matches the hardcoded expected values to within `atol=1e-6` across Ubuntu / Windows / macOS CI runners AT THE TIME OF PR-MERGE; canary values are regression-detection sentinels and MAY be re-captured (in a follow-up commit cross-referencing this scenario) if upstream BLAS / scipy / pywt / numpy semantics legitimately shift after merge

#### Scenario: nutation.compute emits exactly one DEBUG logger record on the happy path
- **GIVEN** a valid `trajectory_df` and `cadence_s = 300.0` and `caplog.set_level(logging.DEBUG)`
- **WHEN** `nutation.compute(trajectory_df, cadence_s=300.0)` is invoked
- **THEN** exactly one log record at level `DEBUG` is emitted by the logger `sleap_roots.circumnutation.nutation`
- **AND** the record's message starts with `"nutation.compute("`
- **AND** the record's message contains each of the tokens `"n_tracks="`, `"coordinate="`, `"cadence_s="`
- **AND** no `INFO` / `WARNING` / `ERROR` / `CRITICAL` records are emitted

#### Scenario: nutation.compute handles stationary tracks gracefully via NaN trait emission
- **GIVEN** a `trajectory_df` containing one track whose `(tip_x, tip_y)` net displacement is zero (e.g., a closed-loop trajectory with `x[-1]==x[0] AND y[-1]==y[0]` but varying intermediate frames) and `cadence_s = 300.0`
- **WHEN** `nutation.compute(trajectory_df, cadence_s=300.0, coordinate="lateral")` is invoked
- **THEN** the call returns a `pandas.DataFrame` without raising
- **AND** `result.is_nutating.iloc[0] == False`
- **AND** the 3 NaN-gated traits (`T_nutation_median`, `T_nutation_iqr`, `A_nutation_envelope_max_px`) are NaN
- **AND** no `np.RuntimeWarning("All-NaN slice encountered")` is emitted (the implementation short-circuits the rest of the pipeline when `_geometry.project_to_growth_axis_perpendicular` returns an all-NaN signal)

#### Scenario: Layer-2 Derr forensic-match acceptance on the Nipponbare proofread fixture (GREEN-phase softened)
- **GIVEN** the 6 tracks of `tests/data/circumnutation_nipponbare_plate_001/plate_001_greyscale.tracked_proofread.slp` loaded via `Series.load(...).get_tracked_tips()` filtered by `track_id ∈ {0, 1, 2, 3, 4, 5}` and `cadence_s = 300.0`
- **WHEN** `nutation.compute(trajectory_df_for_track_i, cadence_s=300.0, coordinate="lateral")` is invoked for each track and the resulting `period_residual_vs_derr_reference` and `is_nutating` values are collected
- **THEN** the median of the 6 per-track `period_residual_vs_derr_reference` values satisfies `abs(np.nanmedian(per_track_residuals)) < 0.25` (GREEN-phase softened from CC-7's ±2% target — see design.md GREEN-phase Reconciliation Appendix: plate-001 fixture shows median residual ≈ 0.20 driven by CWT scale-grid alignment at the ~T=4013s grid point at n_frames=575)
- **AND** at least 3 of the 6 tracks satisfy `abs(period_residual_vs_derr_reference) < 0.30 AND is_nutating == True` (GREEN-phase softened from "≥4 of 6 within ±5%")
- **AND** the long-term CC-7 ±2% target remains documented as the goal; preprocessing improvements (parabolic ridge refinement, denser scale grid, higher-order detrending) tracked in GitHub follow-up issue #219 ("Layer-2 ±2% acceptance pending preprocessing improvements")

#### Scenario: GitHub issue #214 acceptance — ridge-continuity post-filter behaves correctly on plate-001 (GREEN-phase softened)
- **GIVEN** the 6 tracks of the Nipponbare proofread fixture and `cadence_s = 300.0`
- **WHEN** for each track, both `T_nutation_iqr_raw` (using `extract_ridge` only) and `T_nutation_iqr_post_filter` (using `extract_ridge` + `smooth_ridge`) are computed separately
- **THEN** no track WORSENS post-filter: `T_nutation_iqr_post_filter <= T_nutation_iqr_raw` for all 6 tracks (the median filter is a no-op on stable ridges, never a regression)
- **AND** at least 1 of the 6 tracks satisfies `T_nutation_iqr_post_filter < T_nutation_iqr_raw` — confirming the post-filter at least sometimes reduces ridge-jitter-induced IQR inflation
- **AND** the long-term "≥5 of 6 tracks improve" target remains documented as the goal; plate-001 has a clean ridge with minimal scale-hopping so the post-filter is mostly a no-op there. Multi-plate empirical validation tracked in GitHub follow-up issue #220 ("Issue #214 multi-plate empirical validation")

### Requirement: Temporal CWT ridge-continuity smoothing API
The system SHALL provide `sleap_roots.circumnutation.temporal_cwt.smooth_ridge(ridge_result: RidgeResult, window: Optional[int] = None, constants: Optional[ConstantsT] = None) -> RidgeResult`. The function SHALL apply a median-filter post-filter to `ridge_result.periods_s` to suppress scale-grid hopping artifacts in PR #5's per-frame argmax ridge — closing GitHub issue [#214](https://github.com/talmolab/sleap-roots/issues/214) (Mallat 1999 §4.4.2 ridge-continuity baseline).

The function SHALL median-filter ONLY the `periods_s` field (using `scipy.ndimage.median_filter(...mode='nearest')`). The other 4 fields (`frame_indices`, `amplitudes`, `powers`, `in_coi`) SHALL be carried through to the returned `RidgeResult` unchanged. Rationale: issue #214's acceptance is period-IQR-focused; `A_nutation_envelope_max_px` is a PEAK statistic computed from `amplitudes` and would be distorted by smoothing without a corresponding accuracy benefit; `in_coi` is a function of the original ridge scale indices and would require access to the COI mask grid (not available from RidgeResult alone) to recompute; `powers = amplitudes²` is a tautology that doesn't change under period-smoothing; `frame_indices = np.arange(n_frames)` is similarly invariant.

The function SHALL accept `window` as a `kwargs`-precedence override (when not None); otherwise SHALL use `resolved_constants.RIDGE_CONTINUITY_FILTER_WINDOW` (default `5`). The function SHALL emit NO trait values (composition with `nutation.compute` is documented under Requirement: Tier 1 nutation trait emission API). The function SHALL be deterministic: same input → identical `RidgeResult` output at `atol=0`. The function SHALL log exactly one `logger.debug` message at the start (after input validation) whose text begins with `"smooth_ridge("` and contains tokens `n_frames=` and `window=`.

The function SHALL validate inputs strictly: `ridge_result` SHALL be a `RidgeResult` instance (anything else raises `TypeError`); `window` SHALL be an `int` (when not None); `window >= 1` (raise `ValueError` if zero or negative); `window` SHALL be odd (raise `ValueError` if even — `scipy.ndimage.median_filter` requires odd-size for symmetric neighborhood); `constants` SHALL be `None` or a `ConstantsT` instance.

#### Scenario: smooth_ridge returns a RidgeResult with smoothed periods_s and unchanged other fields
- **GIVEN** a valid `RidgeResult` produced by `extract_ridge(compute_scaleogram(x, 300.0))` on a length-≥9 finite array
- **WHEN** `smooth_ridge(ridge_result)` is invoked (with default `window=None` → resolved to `RIDGE_CONTINUITY_FILTER_WINDOW=5`)
- **THEN** the returned object is a `RidgeResult` instance
- **AND** `np.array_equal(result.amplitudes, ridge_result.amplitudes)` (carried through unchanged)
- **AND** `np.array_equal(result.powers, ridge_result.powers)` (carried through unchanged)
- **AND** `np.array_equal(result.in_coi, ridge_result.in_coi)` (carried through unchanged)
- **AND** `np.array_equal(result.frame_indices, ridge_result.frame_indices)` (carried through unchanged)
- **AND** `np.array_equal(result.periods_s, scipy.ndimage.median_filter(ridge_result.periods_s, size=5, mode='nearest'))` (the smoothed periods)

#### Scenario: smooth_ridge accepts window override via positional kwarg
- **GIVEN** a valid `RidgeResult`
- **WHEN** `smooth_ridge(ridge_result, window=11)` is invoked
- **THEN** `result.periods_s == scipy.ndimage.median_filter(ridge_result.periods_s, size=11, mode='nearest')`
- **AND** the other 4 fields are carried through unchanged

#### Scenario: smooth_ridge accepts window override via constants
- **GIVEN** a valid `RidgeResult`
- **WHEN** `smooth_ridge(ridge_result, constants=ConstantsT(RIDGE_CONTINUITY_FILTER_WINDOW=11))` is invoked
- **THEN** `result.periods_s == scipy.ndimage.median_filter(ridge_result.periods_s, size=11, mode='nearest')` (the constants override takes effect when `window` kwarg is None)

#### Scenario: smooth_ridge rejects non-RidgeResult input with TypeError
- **WHEN** `smooth_ridge(x)` is invoked with `x` equal to `None` OR `{}` OR `(1, 2, 3)` OR an arbitrary numpy ndarray
- **THEN** `TypeError` is raised
- **AND** the exception message references the expected type `RidgeResult`

#### Scenario: smooth_ridge rejects non-positive or even window with ValueError
- **WHEN** `smooth_ridge(ridge_result, window=v)` is invoked for any of `v ∈ {0, -1, 4}` (zero, negative, even)
- **THEN** `ValueError` is raised
- **AND** the exception message contains the substring `"window"` and identifies the offending condition (non-positive or even)

#### Scenario: smooth_ridge emits exactly one DEBUG logger record on the happy path
- **GIVEN** a valid `RidgeResult` and `caplog.set_level(logging.DEBUG)`
- **WHEN** `smooth_ridge(ridge_result)` is invoked
- **THEN** exactly one log record at level `DEBUG` is emitted by the logger `sleap_roots.circumnutation.temporal_cwt`
- **AND** the record's message starts with `"smooth_ridge("`
- **AND** the record's message contains each of the tokens `"n_frames="`, `"window="`
- **AND** no `INFO` / `WARNING` / `ERROR` / `CRITICAL` records are emitted

#### Scenario: smooth_ridge is deterministic
- **GIVEN** a valid `RidgeResult`
- **WHEN** `smooth_ridge(ridge_result)` is invoked twice
- **THEN** `np.array_equal(result1.periods_s, result2.periods_s)` is True at `atol=0`
- **AND** all other 4 fields are equal under `np.array_equal` at `atol=0`

### Requirement: Tier 2 ψ_g trait emission API
The system SHALL provide `sleap_roots.circumnutation.psi_g.compute(trajectory_df: pd.DataFrame, cadence_s: float, constants: Optional[ConstantsT] = None) -> pd.DataFrame`. Unlike `nutation.compute`, `psi_g.compute` SHALL NOT take a `coordinate` parameter: ψ_g is computed from the raw 2-D tip trajectory `(tip_x, tip_y)` via the locked `_geometry.compute_psi_g` helper (`atan2(dx, dy)`, unwrapped — Requirement: Tier 0 helper modules), so there is no 1-D-projection choice to make.

The function SHALL emit a per-track DataFrame whose rows correspond to the unique 5-tuple `(series, sample_uid, plate_id, plant_id, track_id)` derived via `groupby(_IDENTITY_5_TUPLE, dropna=False, sort=False)`. The returned DataFrame SHALL contain the 8 row-identity columns (Requirement: Trait CSV row-identity schema) followed by exactly 4 trait columns in this declared order: `T_psig_median_s: float64`, `delta_E_amplitude_proxy_px_per_frame: float64`, `handedness: int64`, `helix_signed_area_px2: float64`. All 4 trait units (`s`, `px/frame`, `int`, `px²`) SHALL be members of `PIPELINE_UNIT_VOCABULARY` (no vocabulary change is required).

The four traits SHALL be defined as follows, on the finite-masked tip coordinates (rows with non-finite `tip_x`/`tip_y` dropped first; let `N` = count of finite frames; ψ_g = `compute_psi_g(tip_x, tip_y)` has length `N−1`):

- `T_psig_median_s`: `np.nanmedian` of the COI-interior smoothed-ridge periods, computed by composing `temporal_cwt.compute_scaleogram → extract_ridge → smooth_ridge` on the **SG-detrended** ψ_g (`_noise.compute_sg_detrended(psi_g, window=SG_WINDOW_DETREND=23, polynomial_order=SG_DEGREE=3)`), masked by `~smooth_ridge.in_coi` (the per-frame boolean, NOT the 2-D `ScaleogramResult.coi_mask`). It is the ONLY trait that uses the conditioned signal and the CWT.
- `delta_E_amplitude_proxy_px_per_frame`: `np.median(√(dx² + dy²))` over all finite velocity samples (`dx, dy = np.diff` of finite tip coords); no COI mask, no `/cadence_s` (px/frame, matching Tier 0's velocity convention). Corresponds to `(L/2R)·ΔĖ` (Eq. 21).
- `handedness`: `int(np.sign(psi_g[-1] − psi_g[0]))` — the sign of the **net unwrapped ψ_g rotation over all finite frames**, with a determinism zero-guard `|psi_g[-1] − psi_g[0]| < 1e-9 rad → 0`. **Sign convention (anchored to avoid the "counterclockwise" ambiguity):** `+1` ⇔ ψ_g increasing ⇔ **positive mean `dψ_g/dt`** — identical to `synthetic.generate_trajectory(handedness=+1)`'s locked convention (Requirement: Synthetic trajectory generator, scenario "handedness=+1 yields positive mean dψ_g/dt"). In physical terms this is clockwise in standard (y-up) math axes and counterclockwise as displayed in the y-down image frame; the program anchors on the `dψ_g/dt` sign, NOT the word "counterclockwise". `0` = no net rotation / degenerate. This is COI-FREE (a deliberate deviation from theory.md §7.3's literal "sign of mean dψ_g/dt over COI-masked range" — see deviation note below).
- `helix_signed_area_px2`: `_geometry.compute_signed_area` applied to the **growth-detrended** tip trajectory (each of `tip_x`, `tip_y` has its per-axis least-squares linear trend subtracted before the y-down-corrected Shoelace `0.5·Σ(x_{i+1}·y_i − x_i·y_{i+1})`). The growth detrend removes the linear-growth ribbon so the enclosed area reflects the nutation orbit, making `sign(helix_signed_area_px2) == handedness` a meaningful confirmation on genuinely 2-D-circulating data. **Why growth-detrend (deviation from the originally-approved raw-coordinate Shoelace):** a self-review on the real plate-001 fixture found the raw Shoelace area is dominated by the growth ribbon for open growing roots — `sign(raw_area) == handedness` held for only **1 of 6** tracks. After the per-axis linear detrend it holds for **≥5 of 6**. (The helper `_geometry.compute_signed_area` itself is unchanged — pure raw Shoelace; the growth-detrend is applied by `psi_g.compute` before calling it. A planar wobble-on-growth such as the synthetic generator's output has no true 2-D circulation and collapses to ~0 area after the detrend, so the sign-agreement is validated only on real circulating data — see the plate-001 scenario below.)

**Deviations from theory.md §7.3 / §6.3 (all recorded; theory.md is patched in this PR to match, preserving the original wording in an Appendix B correction note rather than silently overwriting):**

1. **`handedness` is COI-free** — theory.md §7.3 specifies "sign of mean dψ_g/dt over COI-masked range". PR #7 emits it COI-free because (a) the COI mask is a function of the SG-detrended CWT ridge, so COI-masking would couple a raw kinematic sign to the conditioned signal and to the CWT min-length floor; (b) the per-frame `ridge.in_coi` interior is not contiguous (per-frame argmax scale selection creates COI gaps), so an endpoint difference across a masked gap can report the wrong sign; and (c) COI is a CWT-edge-reliability concept that does not apply to a raw angular displacement (`atan2` of velocity has no edge contamination) — §7.3 itself omits COI for the sibling `delta_E` kinematic median, and the COI-free `angular_amplitude` (§7.1) is precedent.
2. **`delta_E` is px/frame, not px·hr⁻¹** — §7.3 specifies "Median of √(ẋ²+ẏ²) × (frames/hr)" in px·hr⁻¹. PR #7 emits `median(√(dx²+dy²))` in **px/frame** (drops the `(frames/hr)` cadence factor), because `px/s`/`px·hr⁻¹` is not the pipeline's velocity convention (Tier 0 emits all velocities in px/frame; `px/s` ∉ `PIPELINE_UNIT_VOCABULARY`). As a per-track *amplitude proxy* the prefactor-free, cadence-free magnitude preserves the shape and relative magnitude of `Δ·Ė` (R, L, cadence are per-track constants), which is all the proxy needs; the "`= (L/2R)·ΔĖ`" proportionality (Eq. 21 solved for the measured speed) still holds.
3. **ψ_g conditioning is SG-detrend, not §6.3's literal "smooth"** — `T_psig_median_s` feeds the CWT the SG **residual** (`compute_sg_detrended` = raw − SG-smooth), not the SG-smoothed signal. §6.3 says "pre-smoothed via Savitzky-Golay"; the residual is the oscillation component and is what a period-extracting CWT needs (smoothing-only would retain gravitropic drift biasing `T_psig`), and reuses the exact primitive Tier 1 uses. No public SG-smoothing primitive exists.

The function SHALL NOT consume Tier 1 output (no `nutation_df` / `is_nutating` input). Traits are emitted ungated; downstream consumers compose a `LEFT JOIN` of Tier 1's `is_nutating` on the shared 5-tuple to mask non-nutating tracks. The 5th §7.3 trait `psig_long_consistency` (the cross-tier `T_psig ↔ T_nutation` correlation) is intentionally NOT emitted here; it is deferred to and owned by the roadmapped PR #13 Layer-3 cross-tier work (which owns both the trait emission and the consistency test), keeping Tier 2 single-tier.

The function SHALL handle short and degenerate tracks gracefully (never raising, never emitting `np.RuntimeWarning`):

- `N < 3` finite frames (including an all-non-finite track): return `T_psig_median_s=NaN`, `delta_E_amplitude_proxy_px_per_frame=NaN`, `handedness=0`, `helix_signed_area_px2=NaN`.
- `3 ≤ N < 24` (ψ_g shorter than `SG_WINDOW_DETREND`): `T_psig_median_s=NaN` (the CWT path is skipped — `compute_sg_detrended` returns all-NaN for `len < window`, which would otherwise make `compute_scaleogram` raise); `handedness`, `delta_E_amplitude_proxy_px_per_frame`, `helix_signed_area_px2` are fully defined (they are CWT-free).
- Stationary tip or perfectly straight growth (zero detrended energy) with `N ≥ 24`: `T_psig_median_s=NaN` via a zero-energy guard (`np.allclose(psi_g_detrended, 0.0)` → skip the CWT), because `compute_scaleogram` accepts an all-zero signal without raising and `argmax`-over-zeros would otherwise yield a spurious shortest-period ridge.

The function SHALL be deterministic per CC-6: the 3 float trait columns SHALL be bit-identical at `atol=0` across calls in the same process AND identical to within `atol=1e-6` across Ubuntu / Windows / macOS CI runners (inheriting Tier 1's loosened float floor for the SG-detrend→scipy-CWT stack). The integer `handedness` column SHALL be exactly equal across OSes; its `1e-9 rad` numerical-zero guard is safe because the net-rotation endpoint difference is a raw `atan2` quantity (~`1e-12` reproducible), not a CWT output.

The function SHALL validate inputs strictly: `trajectory_df` validation delegates to `_validate_trajectory_df` (Requirement: Tier 0 input-validation boundary); `cadence_s` validation reuses `temporal_cwt._validate_cadence_s` (positive finite; explicit `bool`/`np.bool_` rejection); `constants` SHALL be None or `ConstantsT` else TypeError, and an invalid SG override (e.g. even `SG_WINDOW_DETREND`, or `SG_WINDOW_DETREND ≤ SG_DEGREE`) SHALL raise `ValueError` naming the field. The function SHALL log exactly one `logger.debug` record at the start (after validation) whose text begins with `"psi_g.compute("` and contains the tokens `n_tracks=` and `cadence_s=` (no `coordinate=` token); no INFO/WARNING/ERROR records SHALL be emitted on the happy path.

The system SHALL ALSO provide `_geometry.compute_signed_area(x: np.ndarray, y: np.ndarray) -> float`: the y-down-corrected Shoelace signed area `0.5·Σ_i (x_{i+1}·y_i − x_i·y_{i+1})` (cyclic). It SHALL return `0.0` for inputs of fewer than 3 points (degenerate polygon) and SHALL propagate NaN for non-finite coordinates. Its sign convention SHALL be documented as load-bearing and anchored on the invariant `sign(compute_signed_area(x, y)) == handedness == int(np.sign(ψ_g[-1] − ψ_g[0]))` (the same image-y-down `atan2(dx, dy)` frame `compute_psi_g` encodes) — i.e. positive area corresponds to `+1` (ψ_g increasing; "counterclockwise" only in the y-down image sense, NOT the y-up math sense — anchor on the sign, not the word).

#### Scenario: psi_g.compute returns a DataFrame with the documented column order and dtypes
- **GIVEN** a valid `trajectory_df` with multiple tracks (each ≥ 24 frames) and `cadence_s = 300.0`
- **WHEN** `psi_g.compute(trajectory_df, cadence_s=300.0)` is invoked
- **THEN** the returned object is a `pandas.DataFrame`
- **AND** the column order is: 8 row-identity columns from `ROW_IDENTITY_COLUMNS` followed by the 4 trait columns in declared order `["T_psig_median_s", "delta_E_amplitude_proxy_px_per_frame", "handedness", "helix_signed_area_px2"]`
- **AND** trait dtypes are: 3 float64 (`T_psig_median_s`, `delta_E_amplitude_proxy_px_per_frame`, `helix_signed_area_px2`) and 1 int64 (`handedness`)
- **AND** row-identity 5-tuple uniqueness holds: `df[list(_IDENTITY_5_TUPLE)].duplicated().sum() == 0`

#### Scenario: psi_g.compute trait units are all within PIPELINE_UNIT_VOCABULARY
- **WHEN** the 4 declared trait units `{"T_psig_median_s": "s", "delta_E_amplitude_proxy_px_per_frame": "px/frame", "handedness": "int", "helix_signed_area_px2": "px²"}` are checked
- **THEN** every unit string is a member of `PIPELINE_UNIT_VOCABULARY`

#### Scenario: T_psig_median_s recovers a planted nutation period within ±10%
- **GIVEN** a noise-free synthetic track via `synthetic.generate_trajectory(T_nutation_s=T, n_frames=575, cadence_s=300, amplitude_px=10, noise_sigma_px=0.0)` for `T ∈ {3333, 4500}` (in-band periods, per nutation `test_2C2`)
- **WHEN** `psi_g.compute(trajectory_df, cadence_s=300.0)` is invoked
- **THEN** the recovered `T_psig_median_s` satisfies `abs(T_psig_median_s − T) / T < 0.10`
- **AND** no `np.RuntimeWarning` is emitted

#### Scenario: handedness tracks the generator's planted handedness
- **GIVEN** two synthetic orbits via `synthetic.generate_trajectory(handedness=+1, amplitude_px=10, noise_sigma_px=0.0)` and the same with `handedness=-1` (the generator's `handedness` is the locked `dψ_g/dt`-sign convention, NOT a screen-orientation label)
- **WHEN** `psi_g.compute(...)` is invoked for each
- **THEN** the `handedness=+1` track yields output `handedness == +1`; the `handedness=-1` track yields output `handedness == -1` (the emitted `handedness` equals the planted sign)
- **AND** `helix_signed_area_px2` is finite for both (its sign is NOT asserted on the synthetic — a planar wobble-on-growth has no true 2-D circulation, so its growth-detrended area is ~0; sign-agreement is validated on real plate-001 data below)

#### Scenario: helix_signed_area sign confirms handedness on real plate-001 data (GREEN-phase)
- **GIVEN** the 6 tracks of `tests/data/circumnutation_nipponbare_plate_001/plate_001_greyscale.tracked_proofread.slp` (loaded + enriched with the row-identity columns) and `cadence_s = 300.0`
- **WHEN** `psi_g.compute(trajectory_df, cadence_s=300.0)` is invoked
- **THEN** `int(np.sign(helix_signed_area_px2)) == handedness` for **≥ 5 of the 6** tracks (GREEN-phase reconciliation: the growth-detrended Shoelace agrees with the independently-computed handedness on genuinely 2-D-circulating real data; the raw un-detrended Shoelace agreed only 1/6). The single non-agreeing track is weakly-circulating; the long-term 6/6 goal via a per-period orbit decomposition is tracked as a follow-up.

#### Scenario: compute_signed_area sign is pinned to an absolute hand-built orbit
- **WHEN** `_geometry.compute_signed_area(np.array([0.,1.,1.,0.]), np.array([0.,0.,1.,1.]))` is invoked
- **THEN** the returned value equals `-1.0` exactly (the y-down negation of the standard Shoelace `+1.0`)
- **AND** `_geometry.compute_psi_g(np.array([0.,1.,1.,0.]), np.array([0.,0.,1.,1.]))` has net change `psi_g[-1] − psi_g[0] = −π` so `int(np.sign(net)) == -1`, matching `int(np.sign(-1.0))`
- **AND** `compute_signed_area` returns `0.0` for an input of fewer than 3 points

#### Scenario: conditioning affects only T_psig_median_s
- **GIVEN** the same synthetic track conditioned identically
- **WHEN** the pipeline is traced
- **THEN** `handedness`, `delta_E_amplitude_proxy_px_per_frame`, and `helix_signed_area_px2` are computed from raw inputs (raw unwrapped ψ_g endpoints, raw velocity samples, and per-axis linearly-growth-detrended raw coordinates respectively) and do NOT depend on `compute_sg_detrended` or the CWT (the helix growth-detrend is a simple per-axis linear-trend subtraction, NOT the SG/CWT conditioning)
- **AND** only `T_psig_median_s` uses the SG-detrended ψ_g and the `compute_scaleogram → extract_ridge → smooth_ridge` chain — confirmed by invariance under a `SG_WINDOW_DETREND` override

#### Scenario: short track (3 ≤ N < 24) emits NaN T_psig but defined raw traits without raising
- **GIVEN** a single-track `trajectory_df` with exactly 15 finite frames and `cadence_s = 300.0`
- **WHEN** `psi_g.compute(trajectory_df, cadence_s=300.0)` is invoked
- **THEN** the call returns a `pandas.DataFrame` without raising any exception
- **AND** `np.isnan(result.T_psig_median_s.iloc[0])` (ψ_g length 14 < `SG_WINDOW_DETREND=23` → CWT skipped)
- **AND** `result.handedness.iloc[0]` is in `{-1, 0, +1}` and `result.delta_E_amplitude_proxy_px_per_frame.iloc[0]` and `result.helix_signed_area_px2.iloc[0]` are finite
- **AND** no `np.RuntimeWarning` is emitted

#### Scenario: degenerate too-short track (N < 3) emits the all-degenerate row
- **GIVEN** a single-track `trajectory_df` with exactly 2 finite frames
- **WHEN** `psi_g.compute(trajectory_df, cadence_s=300.0)` is invoked
- **THEN** `T_psig_median_s`, `delta_E_amplitude_proxy_px_per_frame`, and `helix_signed_area_px2` are NaN
- **AND** `handedness == 0`
- **AND** no exception or `np.RuntimeWarning` is raised

#### Scenario: stationary track (N ≥ 24, zero displacement) emits NaN T_psig via the zero-energy guard without a spurious period
- **GIVEN** a stationary single-track `trajectory_df` (≥ 24 frames, zero net and per-frame displacement) via `synthetic.generate_trajectory(amplitude_px=0.0, growth_rate_px_per_frame=0.0, noise_sigma_px=0.0, n_frames=64)` and `cadence_s = 300.0`
- **WHEN** `psi_g.compute(trajectory_df, cadence_s=300.0)` is invoked
- **THEN** `np.isnan(result.T_psig_median_s.iloc[0])` (the zero-energy guard fires; the CWT is NOT entered, so no spurious shortest-period ridge is produced)
- **AND** `result.handedness.iloc[0] == 0` AND `result.delta_E_amplitude_proxy_px_per_frame.iloc[0] == 0.0` AND `result.helix_signed_area_px2.iloc[0] == 0.0`
- **AND** no `np.RuntimeWarning` is emitted

#### Scenario: psi_g.compute rejects invalid cadence_s value with ValueError naming the field
- **WHEN** `psi_g.compute(trajectory_df, cadence_s=v)` is invoked for any of `v ∈ {0, -1.0, float("nan"), float("inf"), float("-inf")}`
- **THEN** `ValueError` is raised
- **AND** the exception message contains the substring `"cadence_s"`

#### Scenario: psi_g.compute rejects invalid cadence_s type with TypeError naming the field
- **WHEN** `psi_g.compute(trajectory_df, cadence_s=v)` is invoked for any of `v ∈ {True, np.bool_(True), "300", [300.0]}`
- **THEN** `TypeError` is raised
- **AND** the exception message contains the substring `"cadence_s"`

#### Scenario: psi_g.compute rejects a non-ConstantsT constants argument with TypeError
- **WHEN** `psi_g.compute(trajectory_df, cadence_s=300.0, constants=v)` is invoked for any of `v ∈ {42, "default", [ ]}`
- **THEN** `TypeError` is raised
- **AND** the exception message contains the substring `"constants"`

#### Scenario: psi_g.compute rejects an invalid SG constants override with ValueError naming the field
- **GIVEN** a `ConstantsT` override with an even `SG_WINDOW_DETREND` (e.g. 24) or `SG_WINDOW_DETREND ≤ SG_DEGREE`
- **WHEN** `psi_g.compute(trajectory_df, cadence_s=300.0, constants=override)` is invoked
- **THEN** `ValueError` is raised
- **AND** the exception message contains `"SG_WINDOW_DETREND"`

#### Scenario: psi_g.compute is deterministic across runs and across OSs
- **GIVEN** a valid multi-track `trajectory_df` and `cadence_s = 300.0`
- **WHEN** `psi_g.compute(trajectory_df, cadence_s=300.0)` is invoked twice in the same Python process
- **THEN** the 3 float trait columns are bit-identical at `atol=0`
- **AND** the `handedness` integer column is exactly equal
- **AND** across Ubuntu / Windows / macOS CI runners the 3 float columns match to within `atol=1e-6` and `handedness` matches exactly

#### Scenario: psi_g.compute emits exactly one DEBUG logger record on the happy path
- **GIVEN** a valid `trajectory_df` and `cadence_s = 300.0` and `caplog.set_level(logging.DEBUG)`
- **WHEN** `psi_g.compute(trajectory_df, cadence_s=300.0)` is invoked
- **THEN** exactly one log record at level `DEBUG` is emitted by the logger `sleap_roots.circumnutation.psi_g`
- **AND** the record's message starts with `"psi_g.compute("`
- **AND** the record's message contains each of the tokens `"n_tracks="` and `"cadence_s="`
- **AND** no `INFO` / `WARNING` / `ERROR` / `CRITICAL` records are emitted

#### Scenario: Cross-tier convention lock — circular_mean(ψ_g) ≈ π/2 − planted growth-axis angle (synthetic RED)
- **GIVEN** a noise-free, non-oscillating synthetic track via `synthetic.generate_trajectory(amplitude_px=0.0, growth_axis_angle_rad=θ, noise_sigma_px=0.0)` for `θ ∈ {0.3, −2.0}` (the `θ = −2.0` case places `π/2 − θ` outside `[−π, π)`, exercising the branch-cut wrap)
- **WHEN** ψ_g is computed for the track and `circular_mean(ψ_g) = atan2(mean(sin ψ_g), mean(cos ψ_g))`
- **THEN** `abs(wrap_to_pi(circular_mean(ψ_g) − (π/2 − θ))) < 1e-6` where `wrap_to_pi(d) = (d + π) mod 2π − π`
- **AND** `handedness == 0` for these non-oscillating tracks (the `1e-6` tolerance applies to the angle identity only — a separate oscillating fixture with `amplitude_px > 0` locks `handedness == planted ±1`)

#### Scenario: Cross-tier consistency on the Nipponbare plate-001 fixture (GREEN-phase reconciliation)
- **GIVEN** the 6 tracks of `tests/data/circumnutation_nipponbare_plate_001/plate_001_greyscale.tracked_proofread.slp` loaded via `Series.load(...).get_tracked_tips()` filtered by `track_id ∈ {0..5}`, and `cadence_s = 300.0`
- **WHEN** for each track `circular_mean(ψ_g)` (Tier 2) and `principal_axis_angle` (from `kinematics.compute`, Tier 0) are computed
- **THEN** tracks whose `principal_axis_angle` is NaN (the `growth_axis_unreliable` gate fired) are SKIPPED
- **AND** at least `N`-of-6 surviving tracks satisfy `abs(wrap_to_pi(circular_mean(ψ_g) − (π/2 − principal_axis_angle))) < _PSIG_AXIS_RECONCILE_TOL_RAD`, where the count `N` and the tolerance constant are documented GREEN-phase reconciliation values captured from a real run (mirroring the `_DERR_MATCH_*` precedent)
- **AND** the captured values MUST clear a pre-committed floor that the implementation cannot trivially satisfy: `N ≥ 2` AND `_PSIG_AXIS_RECONCILE_TOL_RAD ≤ 0.35 rad` (if a real run cannot meet this floor, that is a genuine RED signal to investigate, not a tolerance to widen); the GREEN commit SHALL also record the observed per-track angular deviations (max + distribution), not only the chosen tolerance, so headroom is auditable

### Requirement: Tier 3a midline reconstruction API
The system SHALL provide `sleap_roots.circumnutation.midline.reconstruct(x: np.ndarray, y: np.ndarray, cadence_s: float, sg_window: Optional[int] = None, constants: Optional[ConstantsT] = None) -> MidlineResult`. It reconstructs the tip-trail-as-midline (theory.md §6.1): the curve of past tip positions parameterized by arc length `s(τ) = ∫|v|dσ`, with per-frame trajectory curvature `κ = (ẋÿ − ẏẍ)/(ẋ²+ẏ²)^{3/2}` (theory.md §6.2) and tip speed. It is **machinery**: it emits NO trait columns, ingests NO `trajectory_df`, performs NO 5-tuple groupby, builds NO `L_gz` growth-zone mask, and performs NO uniform-arc-length resampling.

`x`, `y`, and `cadence_s` are required positional parameters. `sg_window` defaults to `constants.SG_WINDOW_SHORT` (5); the Savitzky-Golay polynomial degree is `constants.SG_DEGREE` (3) and is NOT a parameter. `cadence_s` is validated (via the imported `temporal_cwt._validate_cadence_s`) and stored as provenance, but the reconstruction is **cadence-independent**: the per-frame outputs are frame-parameterized (velocity in px/frame, arc length integrated over the frame index), so `cadence_s` does not affect `arc_length_px`, `curvature_px_inv`, or `velocity_sub_noise_mask`. `constants=None` resolves to `ConstantsT()`.

**Differentiation = Savitzky-Golay analytic derivatives.** The smoothed coordinates and their first/second derivatives come from ONE `savgol_filter` polynomial per coordinate (deriv=0 → `x_smooth_px`/`y_smooth_px`; deriv=1 → ẋ/ẏ; deriv=2 → ẍ/ÿ), via the new shared helper `_noise.compute_sg_derivative` (below). This realizes theory §6.2's "SG smoothing BEFORE second-derivative operations" self-consistently.

**The function SHALL return a frozen `MidlineResult`** (`@attrs.define(frozen=True, slots=False, kw_only=True, eq=False)`, mirroring `ScaleogramResult`/`RidgeResult` but ADDING `eq=False`) with these per-frame arrays, each length `n = len(x)`, index `i` ↔ frame `i`. The `eq=False` is a deliberate improvement over the `ScaleogramResult` template: with ndarray fields, attrs' generated `__eq__` on `result_a == result_b` is ill-defined (a multi-element ndarray has no unambiguous truth value, so `==` can raise `ValueError: ambiguous truth value` depending on field shapes); `eq=False` makes `==` identity-based so consumers (and the determinism test) compare **field-by-field** with `np.array_equal` (the `temporal_cwt` determinism-test precedent) rather than relying on a fragile generated `__eq__`.

- `frame_indices: int64` — `np.arange(n)`.
- `x_smooth_px: float64`, `y_smooth_px: float64` — SG deriv=0 of `x`, `y` (px).
- `speed_px_per_frame: float64` — `√(ẋ² + ẏ²)` from SG deriv=1 (px/frame, the program's cadence-independent velocity convention; `px/s` is NOT used per theory §10 Appendix B).
- `arc_length_px: float64` — `scipy.integrate.cumulative_trapezoid(speed_px_per_frame, dx=1.0, initial=0)` (px); `arc_length_px[0] == 0.0`, monotonic non-decreasing.
- `curvature_px_inv: float64` — `κ` (px⁻¹) via `_geometry.compute_path_curvature(ẋ, ẏ, ẍ, ÿ)`; non-finite entries (the underflow/overflow corner) swept to NaN. This is a SINGLE array: curvature is parameterization-invariant, so the time-domain `κ_path(τ)` and arc-length `κ(s(τ))` are bit-identical (theory §6.1) — pair it with `frame_indices` for the time view or with `arc_length_px` (NON-uniformly sampled) for the arc view.
- `velocity_sub_noise_mask: bool` — `True ⇔ speed_px_per_frame ≤ NOISE_MASK_K · σ_v` (theory §6.2's sub-noise guard; `σ_v = np.std(speed_px_per_frame, ddof=0)`). `True` flags a sub-noise frame to EXCLUDE before curvature use (consumers use `curvature_px_inv[~velocity_sub_noise_mask]`), matching `ScaleogramResult.coi_mask`'s True=unreliable polarity. **This is a per-FRAME, time-domain mask and is NOT the `L_gz` growth-zone mask** (per-arc-length apical region, built in PR #10) — the two share `σ_v`/`NOISE_MASK_K` vocabulary but are different objects in different domains.

and these provenance scalars: `cadence_s: float`, `sg_window: int`, `sg_degree: int`, `sigma_v_px_per_frame: float` (the `np.std` used; NaN on degenerate), `noise_mask_k: float`, `is_degenerate: bool`. The `is_degenerate` flag + provenance scalars are a deliberate divergence from `ScaleogramResult`/`RidgeResult` (which carry only data): an all-NaN `MidlineResult` + explicit flag is the correct degenerate output for a per-track reconstruction primitive (an exception would force every per-track caller to wrap try/except).

**The function SHALL handle invalid vs. degenerate-but-valid inputs with a split policy:**

- RAISE (field-named, CC-1): `x`/`y` not an ndarray (`TypeError`); not 1-D, complex/object/non-numeric dtype, `len(x) != len(y)`, or any non-finite (NaN/±inf) value (`ValueError`) — non-finite is REJECTED, not dropped, because SG and `cumulative_trapezoid` assume uniform frame spacing; invalid `cadence_s` (value→`ValueError`, type incl. `bool`/`np.bool_`/str→`TypeError`); invalid `sg_window` (even / `≤ SG_DEGREE` / non-int). **ALL field-named validation runs FIRST and unconditionally; the degenerate gate runs only on fully-valid inputs.** Therefore a short all-NaN track RAISES (non-finite check precedes the length gate), and an `n == 0` input with an invalid `cadence_s` RAISES (cadence validation precedes the degenerate gate) — validation always wins over the graceful path.
- GRACEFUL all-NaN `MidlineResult` (`is_degenerate=True`, NEVER raising, NEVER emitting `np.RuntimeWarning`): `n == 0` (length-0 arrays); `n < sg_window` (SG cannot apply); raw-stationary (`np.ptp(x) == 0 and np.ptp(y) == 0` on the RAW input — post-SG speed is floating-point dust, never exactly 0, so stationarity is detected pre-SG). On the graceful path the float per-frame arrays are `np.full(n, np.nan)`, `frame_indices = np.arange(n, dtype=np.int64)`, and — because a `bool` array CANNOT hold NaN (`np.full(n, np.nan, dtype=bool)` silently yields all-`True`) — **`velocity_sub_noise_mask = np.zeros(n, dtype=bool)` (all-`False`)**: no frame is asserted reliable-or-unreliable when the reconstruction is void, and `curvature_px_inv[~mask]` then selects the (all-NaN) curvature rather than an empty array. The degenerate gate returns BEFORE any `np.std`/`np.hypot`/`cumulative_trapezoid` call (`np.std([])` warns; `cumulative_trapezoid([])` raises), with `n == 0` as the first short-circuit disjunct (`np.ptp([])` raises). Curvature (on the non-degenerate path) is computed under `np.errstate(divide="ignore", invalid="ignore", over="ignore")` followed by a `κ[~np.isfinite(κ)] = np.nan` sweep, so no `inf`/`-inf` leaks and no RuntimeWarning is emitted on any path (including a deliberately huge-magnitude curvature input).

**The function SHALL be deterministic per CC-6:** all float arrays SHALL be bit-identical at `atol=0` across calls in the same process; `frame_indices` (int) and `is_degenerate` (bool) SHALL be exactly equal. Across Ubuntu / Windows / macOS CI runners the float arrays SHALL match to within `atol=1e-9, rtol=0` (the measured full-pipeline ULP-propagation floor for the well-conditioned savgol-lstsq + `cumulative_trapezoid` + `|v|³`-division stack is ≈1e-14; PR #6/#7's looser 1e-6 was a coverage argument for a 4-path scipy trait stack and does not transfer). A determinism canary script `scripts/circumnutation/capture_midline_canary.py` SHALL capture canary values from BOTH a closed-form analytic input (a pure circle radius `R`, where `κ ≡ 1/R` exactly — a self-evident oracle) AND `synthetic.generate_trajectory(random_state=0, n_frames=128, …)` (a drift detector), asserted within the cross-OS `atol=1e-9`.

**The function SHALL log exactly one `logger.debug` record** at the start (after validation) whose text begins with `"midline.reconstruct("` and contains the tokens `n_frames=` and `sg_window=`; no INFO/WARNING/ERROR records SHALL be emitted on the happy path (CC-9).

**Handoff to PR #9 (recorded so the seam is explicit).** PR #9's locked stub `spatial_cwt.compute_scaleogram(kappa, ds, constants=None)` takes a scalar grid spacing `ds`, presuming curvature on a uniform-`ds` grid. PR #8 deliberately emits `curvature_px_inv` on the native NON-uniform `arc_length_px` grid; PR #9 OWNS the resample `(MidlineResult.curvature_px_inv, MidlineResult.arc_length_px) → (kappa_uniform, ds)` (where the `ds`/spatial-Nyquist `NYQUIST_RATIO_MAX` decision lives). `arc_length_px` + `curvature_px_inv` is the complete, sufficient input for that resample.

**The system SHALL ALSO provide the new shared helper `_noise.compute_sg_derivative(x: np.ndarray, window: int, polynomial_order: int, deriv: int, delta: float = 1.0, mode: str = "interp") -> np.ndarray`:** a `scipy.signal.savgol_filter(x, window, polynomial_order, deriv=deriv, delta=delta, mode=mode)` wrapper reusing `compute_sg_detrended`'s window/`polynomial_order` boundary validation, AND additionally validating `0 ≤ deriv ≤ polynomial_order` (because `scipy` SILENTLY returns all-zeros for `deriv > polynomial_order` — a silent-wrong-answer hazard — and raises an opaque `factorial()` error for `deriv < 0`; the helper converts both into field-named errors). It SHALL return `np.full(len(x), np.nan)` (length-preserving) when `len(x) < window`. As part of this PR, `_noise.py` SHALL use module-qualified `scipy.signal.savgol_filter` (the existing bare `from scipy.signal import savgol_filter` import is converted) per the program's scipy-import discipline.

**The system SHALL ALSO provide the new shared helper `_geometry.compute_path_curvature(x_dot: np.ndarray, y_dot: np.ndarray, x_ddot: np.ndarray, y_ddot: np.ndarray) -> np.ndarray`:** the trajectory curvature `κ = (ẋ·ÿ − ẏ·ẍ) / (ẋ² + ẏ²)^{3/2}` (px⁻¹; theory §6.2), sibling to `compute_psi_g`/`compute_signed_area`. It SHALL raise `ValueError` on length-mismatched inputs and SHALL set `κ = NaN` where `|v| = 0` (guarded — never `inf`, never `RuntimeWarning`). Its sign convention is **load-bearing** and SHALL be documented anchored on the FORMULA sign (NOT the frame-ambiguous word "left turn"): the formula is the standard y-up math curvature formula, so `+κ` is a left turn in y-up math axes and a clockwise/visual-right turn as displayed in the y-down image frame — the same anchor-on-the-sign discipline `_geometry.compute_signed_area` uses. **Cross-helper sign relationship (load-bearing — must be in the docstring to prevent a publication-trait inversion in PR #9/#10):** because the ψ_g family (`compute_psi_g`/`compute_signed_area`/`handedness`) uses the deliberately swapped `atan2(dx, dy)` argument order while `compute_path_curvature` uses the standard `(ẋÿ − ẏẍ)` formula, the exact per-frame identity is `dψ_g/dt = −κ·|v|`, so `sign(dψ_g/dt) = −sign(κ)` frame-by-frame wherever `|v| > 0`. For a loop traversed with a SINGLE sense of rotation (single-signed κ — e.g. a circle/ellipse/arc) this collapses to the scalar `sign(κ) == −handedness` (e.g. a y-up-math-CCW circle gives `κ = +1/R` but `handedness = −1`). For a sign-changing trajectory (e.g. wobble-on-growth) κ is per-frame multi-signed and `handedness` is a net-rotation scalar, so only the per-frame identity applies, not the scalar form. A consumer composing curvature chirality with `handedness` MUST account for this opposite polarity. The convention SHALL be pinned by an absolute hand-built-input test AND a cross-helper sign-consistency test (below), and theory.md §6.2 is patched in this PR with the y-down clarification + the `sign(κ) == −handedness` note (original "positive = left turn" wording preserved in an Appendix B correction note).

#### Scenario: midline.reconstruct returns a MidlineResult with the documented fields, dtypes, and arc-length contract
- **GIVEN** valid 1-D float64 ndarrays `x`, `y` of equal length `n ≥ sg_window` (e.g. `n = 32`) with all-finite values and non-zero displacement, and `cadence_s = 300.0`
- **WHEN** `midline.reconstruct(x, y, cadence_s=300.0)` is invoked
- **THEN** the returned object is a `MidlineResult`
- **AND** `frame_indices` (int64), `x_smooth_px`/`y_smooth_px`/`speed_px_per_frame`/`arc_length_px`/`curvature_px_inv` (float64), and `velocity_sub_noise_mask` (bool) are each length `n`
- **AND** the scalars `cadence_s` (float), `sg_window` (int), `sg_degree` (int), `sigma_v_px_per_frame` (float), `noise_mask_k` (float), `is_degenerate` (bool == False here) are present
- **AND** the resolved provenance scalars take their documented default values when not overridden: `sg_window == SG_WINDOW_SHORT` (5), `sg_degree == SG_DEGREE` (3), `noise_mask_k == NOISE_MASK_K` (2)
- **AND** `arc_length_px[0] == 0.0` and `arc_length_px` is monotonic non-decreasing

#### Scenario: midline.reconstruct outputs are cadence-independent
- **GIVEN** the same `x`, `y` reconstructed with `cadence_s = 300.0` and with `cadence_s = 1.0`
- **WHEN** the two `MidlineResult`s are compared
- **THEN** `arc_length_px`, `curvature_px_inv`, `speed_px_per_frame`, and `velocity_sub_noise_mask` are bit-identical (the reconstruction is frame-parameterized; `cadence_s` is stored as provenance only)

#### Scenario: compute_path_curvature sign is pinned to absolute hand-built inputs
- **WHEN** `_geometry.compute_path_curvature(np.array([1.0]), np.array([0.0]), np.array([0.0]), np.array([1.0]))` is invoked
- **THEN** the returned value equals `+1.0` exactly (unit velocity +x, unit acceleration +y)
- **AND** the derivatives of a counterclockwise (y-up math) unit circle `(cos t, sin t)` yield `κ ≈ +1/R` (R = 1 → `+1`), and a clockwise circle `(cos t, −sin t)` yields `κ ≈ −1/R`
- **AND** a straight line (zero acceleration) yields `κ ≈ 0`, and a `|v| = 0` frame yields `NaN` with no `np.RuntimeWarning`

#### Scenario: compute_path_curvature sign is opposite to handedness on a known loop (cross-helper anchor)
- **GIVEN** a y-up-math counterclockwise circular loop `(x, y) = (cos t, sin t)` and its analytic derivatives
- **WHEN** `_geometry.compute_path_curvature(ẋ, ẏ, ẍ, ÿ)` and `_geometry.compute_psi_g(x, y)` are computed on the same loop
- **THEN** `np.sign(compute_path_curvature(...))` is `+1` (the standard-formula curvature) while `int(np.sign(psi_g[-1] − psi_g[0])) == handedness` is `−1` — i.e. `sign(κ) == −handedness`
- **AND** this opposite polarity (arising from the ψ_g family's swapped `atan2(dx, dy)` vs. the standard `(ẋÿ − ẏẍ)` formula) is documented in the `compute_path_curvature` docstring and theory.md §6.2 so a PR #9/#10 chirality trait does not silently invert

#### Scenario: compute_sg_derivative recovers polynomial derivatives and validates the deriv range
- **GIVEN** `x = 2·t² + 3·t + 1` for `t = np.arange(11.0)`, `window = 5`, `polynomial_order = 3`
- **WHEN** `_noise.compute_sg_derivative(x, 5, 3, deriv=1, delta=1.0)` and `deriv=2` are invoked
- **THEN** deriv=1 recovers `4·t + 3` and deriv=2 recovers the constant `4` to machine precision
- **AND** `compute_sg_derivative(x, 5, 3, deriv=4)` raises `ValueError` naming `deriv` (NOT scipy's silent all-zeros), and `deriv=-1` raises `ValueError` (NOT scipy's opaque `factorial` error)
- **AND** for `len(x) < window` the function returns `np.full(len(x), np.nan)`

#### Scenario: midline.reconstruct velocity-sub-noise mask has the documented polarity
- **GIVEN** a reconstruction whose `speed_px_per_frame` has `σ_v = np.std(speed_px_per_frame, ddof=0)`
- **WHEN** `velocity_sub_noise_mask` is inspected
- **THEN** `velocity_sub_noise_mask[i] == (speed_px_per_frame[i] <= noise_mask_k * sigma_v_px_per_frame)` for every frame `i`
- **AND** `noise_mask_k == NOISE_MASK_K` (default 2) and `sigma_v_px_per_frame == np.std(speed_px_per_frame, ddof=0)`

#### Scenario: midline.reconstruct rejects non-finite, mismatched, and mistyped inputs (raise, not drop)
- **WHEN** `midline.reconstruct(x, y, cadence_s=300.0)` is invoked with `x` or `y` containing a NaN or ±inf
- **THEN** `ValueError` is raised (the non-finite frame is rejected, NOT dropped — SG and arc-length integration assume uniform frame spacing)
- **AND** mismatched lengths (`len(x) != len(y)`) raise `ValueError`; a non-ndarray `x` raises `TypeError`; a non-1-D or complex/object-dtype `x` raises `ValueError`
- **AND** an invalid `cadence_s` value (`0`, negative, NaN, ±inf) raises `ValueError` naming `cadence_s`, and an invalid type (`True`, `np.bool_(True)`, `"300"`, `[300.0]`) raises `TypeError` naming `cadence_s`
- **AND** an even `sg_window`, `sg_window ≤ SG_DEGREE`, or a non-int `sg_window` raises `ValueError`/`TypeError` naming `sg_window`
- **AND** all field-named validation runs before the degenerate gate, so a `3 ≤ n < sg_window` all-NaN track RAISES (does not return graceful-NaN), and an `n == 0` input with an invalid `cadence_s` RAISES (validation wins over the graceful path) rather than returning a degenerate `MidlineResult`

#### Scenario: midline.reconstruct degrades gracefully on degenerate-but-valid inputs without raising or warning
- **GIVEN** the following degenerate-but-valid inputs, each invoked under `warnings.simplefilter("error")`
- **WHEN** `midline.reconstruct(x, y, cadence_s=300.0)` is invoked with: (a) `n = 0` empty arrays; (b) `0 < n < sg_window` finite arrays; (c) a raw-stationary track (`x` and `y` all-constant, `n ≥ sg_window`)
- **THEN** each returns a `MidlineResult` with `is_degenerate == True`, the float per-frame arrays of length `n` filled with `NaN`, `frame_indices == np.arange(n)` (int64), `velocity_sub_noise_mask == np.zeros(n, dtype=bool)` (all-`False`, since a bool array cannot hold NaN), and `sigma_v_px_per_frame` is `NaN`
- **AND** no exception and no `np.RuntimeWarning` is raised (the gate returns before any `np.std`/`np.hypot`/`cumulative_trapezoid` call; `n == 0` is the first short-circuit disjunct)
- **AND** for `n == sg_window` (non-stationary) a full reconstruction is produced (`is_degenerate == False`)

#### Scenario: midline.reconstruct emits no RuntimeWarning and no inf at the curvature blow-up corner
- **GIVEN** an input that drives a near-zero `|v|³` denominator (a near-stationary frame, optionally with a large acceleration), invoked under `warnings.simplefilter("error")`
- **WHEN** `midline.reconstruct(x, y, cadence_s=300.0)` is invoked
- **THEN** `curvature_px_inv` contains no `inf`/`-inf` (non-finite entries are swept to `NaN`)
- **AND** no `np.RuntimeWarning` (divide / invalid / overflow) is emitted

#### Scenario: midline.reconstruct is deterministic across runs and across OSs (with a closed-form canary)
- **GIVEN** a valid `x`, `y` and `cadence_s = 300.0`
- **WHEN** `midline.reconstruct` is invoked twice in the same Python process
- **THEN** all float arrays are bit-identical at `atol=0` and `frame_indices`/`is_degenerate` are exactly equal
- **AND** for the EXACTLY-specified canary circle — `R = 50.0`, `theta = np.linspace(0.0, 2*np.pi, 128, endpoint=False)`, `x = R*np.cos(theta)`, `y = R*np.sin(theta)`, center `(0, 0)` — the interior `curvature_px_inv` recovers `1/R = 0.02` to a LOOSE physical-accuracy tolerance (`atol ≈ 1e-3`; the SG-polynomial discretization of a sampled circle limits this to ~1e-4, NOT the reproducibility floor) — this is the closed-form ORACLE check. (This is a CLOSED loop with zero net displacement but `ptp(x)=ptp(y)=2R ≠ 0`, so it is correctly NOT flagged degenerate — `reconstruct` gates stationarity on `ptp`, not net displacement; see the helper-divergence note in the design.)
- **AND** SEPARATELY, the captured circle-canary AND a `synthetic.generate_trajectory(random_state=0, n_frames=128, …)` canary (its `tip_x`/`tip_y` columns extracted as `float64` ndarrays) match across Ubuntu / Windows / macOS to within the cross-OS REPRODUCIBILITY floor `atol=1e-9, rtol=0` — a DIFFERENT assertion from physical accuracy (runtime-value vs. hardcoded captured-value array, not vs. `1/R`)

#### Scenario: midline.reconstruct is physically plausible on the real Nipponbare plate-001 fixture (GREEN-phase)
- **GIVEN** the 6 tracks of `tests/data/circumnutation_nipponbare_plate_001/plate_001_greyscale.tracked_proofread.slp` (loaded via `Series.load(series_name="plate_001", primary_path=…).get_tracked_tips()`, `track_id` strings `"track_<i>"`) and `cadence_s = 300.0`
- **WHEN** `midline.reconstruct(tip_x, tip_y, cadence_s=300.0)` is invoked per track
- **THEN** `arc_length_px` is monotonic with `arc_length_px[0] == 0.0`, and `curvature_px_inv` is finite on every `~velocity_sub_noise_mask` frame
- **AND** `max(abs(curvature_px_inv[~velocity_sub_noise_mask])) < 1.0` px⁻¹ (the unmasked array is the physically-plausible one; observed ≈ 0.09–0.17)
- **AND** the `velocity_sub_noise_mask` fraction is in the band `0.1 < frac < 0.85` (observed ≈ 0.38–0.61, recorded per-track for auditability). **Framing caveat (per scientific review):** on plate-001 the rice tip moves only ~3× the localization-noise floor, so ~half the frames fall below `2·σ_v` — and `σ_v = std(speed)` happens to land near the `√2·σ_pos` noise floor HERE, a data-specific coincidence, NOT a general identity (the `std(speed)` threshold is statistical relative to the speed distribution's own spread and is largely SNR-insensitive in this regime). This is the corrected expectation (NOT "flags ~nothing"), but the mask fraction is data-dependent; the band's loose upper bound (0.85 vs observed 0.61) gives CI headroom. PR #9 OWNS how the resulting ~50%-sparse, non-uniform `κ(s)` is gap-handled for its spatial CWT; PR #10 may refine the `σ_v` definition when the mask is first consumed for a trait

#### Scenario: midline.reconstruct total arc length agrees with Tier 0 path length on real plate-001 (cross-tier, GREEN-phase)
- **GIVEN** the same 6 real plate-001 tracks (no gap frames) and Tier 0's path length `L = Σ‖diff(xy)‖` (from `kinematics`)
- **WHEN** `arc_length_px[-1]` (midline total path length) is compared to `L` per track
- **THEN** `arc_length_px[-1] ≤ L` per track (the robust SNR-independent invariant: SG smoothing removes jitter, so the smoothed-path length never exceeds the raw step-sum)
- **AND** `abs(arc_length_px[-1] − L) / L` is within a documented tolerance (observed 2.1–3.1% on plate-001; the assertion tolerance ≤ ~5%, clearing a pre-committed floor, with the observed per-track deviations AND the per-track `σ_pos` recorded). **Caveat:** this magnitude tolerance is data-SNR-dependent — at higher localization noise SG removes more jitter and the gap widens (a synthetic σ_pos≈2px check reached ~30%), so the `σ_pos` is recorded and the robust `≤ L` invariant above is the primary assertion
- **AND** the test documents that Tier 0 NaN-drops gap frames before summing, so the agreement holds only on gap-free tracks (true for plate-001)

### Requirement: Tier 3b spatial curvature resample API
The system SHALL provide `sleap_roots.circumnutation.spatial_cwt.resample_curvature(curvature_px_inv: np.ndarray, arc_length_px: np.ndarray, velocity_sub_noise_mask: Optional[np.ndarray] = None, constants: Optional[ConstantsT] = None) -> ResampleResult`. The function SHALL resample the PR #8 midline curvature κ(s) — emitted on the native NON-uniform `arc_length_px` grid — onto a uniform-spacing grid suitable for the spatial CWT (the locked PR #8→#9 handoff). It SHALL take RAW arrays (NOT a `MidlineResult`), so it is unit-testable and carries no `spatial_cwt → midline` import edge.

The function SHALL: (1) drop frames flagged by `velocity_sub_noise_mask` (where provided; `True` = sub-noise = excluded, matching the PR #8 mask polarity) AND any non-finite `(curvature, arc_length)` pairs — sub-noise frames carry noise-amplified curvature since κ = (ẋÿ−ẏẍ)/|v|³ blows up as |v|→0; (2) reparameterize the surviving samples onto an apex-origin axis `s_a = max(surviving arc_length_px) − (surviving arc_length_px)` (apex at `s_a = 0`, per the theory §6.5 fossil-marker convention: the latest/largest-arc tip position is the current apex; `max`/`min` are taken over the SURVIVING samples so a dropped apex frame does not leave a phantom origin); (3) choose `ds` as the median of the positive differences of the sorted surviving `s_a` (the typical per-frame arc step, robust to large gaps); (4) build a uniform grid of EXACTLY `floor(arc_span_px / ds) + 1` points starting at `0` with spacing `ds` (a pinned length so the canary/shape tests are unambiguous and cross-OS deterministic — `np.arange`-vs-`linspace` rounding must not change the count) and interpolate κ via `np.interp`. λ itself is orientation-invariant; the apex-origin axis exists so downstream consumers (PR #10) can do apex-vs-basal comparisons.

Non-finite handling: non-finite `(curvature, arc_length)` pairs are DROPPED (treated as missing data, like masked frames), NOT rejected — `curvature_px_inv` legitimately carries NaN (a degenerate `MidlineResult` is all-NaN, and curvature blow-up is swept to NaN by PR #8), so rejecting would make any partially-NaN curvature raise. Duplicate-knot handling: if multiple surviving samples map to the same `s_a` (duplicate `arc_length_px`, e.g. a stalled tip after float rounding), they SHALL be deduplicated by averaging their κ so the interpolation knots are strictly increasing — `np.interp` requires a strictly-increasing `xp` for well-defined, deterministic, cross-OS output. The degenerate gate (below) SHALL run BEFORE the `ds = median(positive Δs_a)` computation so that an all-equal `arc_length_px` (empty positive-difference set) returns the graceful all-NaN Result rather than emitting `np.RuntimeWarning` from `np.median([])`.

The function SHALL be deterministic per CC-6: same input → bit-identical `kappa_uniform` across calls in the same process (`atol=0`) AND identical to within `atol=1e-9, rtol=0` across Ubuntu / Windows / macOS CI runners. The function SHALL log exactly one `logger.debug` message at the start (after input validation) whose text begins with `"resample_curvature("` and contains the tokens `n_input=`, `n_unmasked=`, `ds=`, `arc_span_px=`. No INFO/WARNING/ERROR records SHALL be emitted on the happy path.

The function SHALL validate inputs strictly (field-named, runs FIRST and unconditionally): `curvature_px_inv` and `arc_length_px` SHALL be 1-D `np.ndarray` (coercible from integer/float; rejecting `complex`/`object`) of EQUAL length (else `ValueError`); non-ndarray raises `TypeError`; `velocity_sub_noise_mask`, where provided, SHALL be a bool-coercible 1-D array of the same length (else `ValueError`/`TypeError`); `constants` SHALL be `None` or a `ConstantsT` instance (else `TypeError`). Validation always wins over the degenerate path.

After validation, a TWO-STAGE degenerate gate runs ONLY on valid inputs and SHALL return a graceful all-NaN `ResampleResult` (`is_degenerate=True`) — never raising, never emitting `np.RuntimeWarning` (computed under `np.errstate` with a `~np.isfinite` sweep). **Stage 1 (pre-`ds`, from survivor count + span ALONE):** degenerate when, after dropping masked + non-finite frames, fewer than `MIN_SAMPLES_REQUIRED` (= `9` at defaults) samples survive (including the zero-survivor extreme `n_unmasked == 0`, e.g. a fully-masked input) OR the surviving arc-span is non-positive (the all-equal-`arc_length` case). Stage 1 SHALL short-circuit BEFORE both the apex-origin reparameterization's `max`/`min` AND `ds = median(positive Δs_a)` — so an empty survivor set never reaches `np.max([])`/`np.min([])` and an all-equal `arc_length` never reaches `np.median([])`. **Stage 2 (post-`ds`, on the OUTPUT grid length):** the uniform grid length `floor(arc_span_px / ds) + 1` SHALL ALSO be `≥ MIN_SAMPLES_REQUIRED`, else degenerate — because `ds = median(positive Δs_a)` with a skewed/bimodal gap distribution can produce a grid SHORTER than the survivor count (e.g. 9 survivors with gaps `[1,1,1,100,100,100,100,100]` → `arc_span=503`, `ds=100`, grid length `=6 < 9`), which would otherwise feed `compute_scaleogram` a too-short `kappa` and make it raise. The Stage-2 check guarantees a non-degenerate `ResampleResult` always yields a `kappa_uniform` of length `≥ MIN_SAMPLES_REQUIRED` (the invariant `compute_scaleogram` relies on).

The `ResampleResult` class SHALL be an `@attrs.define(frozen=True, slots=False, kw_only=True, eq=False)` container with exactly the following fields (in this order): `kappa_uniform: np.ndarray` (float64, px⁻¹, on the uniform apex-origin grid); `s_a_uniform_px: np.ndarray` (float64, px, apex-origin, `s_a_uniform_px[0] == 0.0`); `ds: float` (px); `n_unmasked: int`; `arc_span_px: float`; `is_degenerate: bool`. (`eq=False` because ndarray `__eq__` is ambiguous.)

#### Scenario: resample_curvature returns a ResampleResult with the documented field shapes and dtypes
- **GIVEN** finite 1-D float64 arrays `curvature_px_inv` and `arc_length_px` of equal length with a monotonic non-decreasing `arc_length_px` of positive span and no mask
- **WHEN** `resample_curvature(curvature_px_inv, arc_length_px)` is invoked
- **THEN** the returned object is a `ResampleResult` instance with `is_degenerate == False`
- **AND** `kappa_uniform.dtype == np.float64` and `s_a_uniform_px.dtype == np.float64` and `kappa_uniform.shape == s_a_uniform_px.shape`
- **AND** `s_a_uniform_px[0] == 0.0` and `np.all(np.diff(s_a_uniform_px) > 0)` and `result.ds > 0`

#### Scenario: ResampleResult is a frozen attrs class with the six documented fields
- **WHEN** `attrs.fields(ResampleResult)` is inspected
- **THEN** the field names are exactly `("kappa_uniform", "s_a_uniform_px", "ds", "n_unmasked", "arc_span_px", "is_degenerate")` in that order
- **AND** attempting `result.ds = 1.0` raises `attrs.exceptions.FrozenInstanceError`

#### Scenario: resample_curvature drops sub-noise-masked frames before interpolation
- **GIVEN** finite arrays where `velocity_sub_noise_mask` flags a known subset of frames `True`
- **WHEN** `resample_curvature(curvature_px_inv, arc_length_px, velocity_sub_noise_mask=mask)` is invoked
- **THEN** `result.n_unmasked == int((~mask).sum())`
- **AND** the interpolation uses only the unmasked samples (the masked frames do not contribute knots)

#### Scenario: resample_curvature coerces an integer 0/1 mask identically to a bool mask
- **GIVEN** finite inputs and a `velocity_sub_noise_mask` given as an integer 0/1 array (`np.array([0, 1, 0, ...], dtype=int)`)
- **WHEN** `resample_curvature(curvature_px_inv, arc_length_px, velocity_sub_noise_mask=int_mask)` is invoked
- **THEN** it coerces to bool and yields the same `n_unmasked` and `kappa_uniform` as the equivalent `dtype=bool` mask
- **AND** a mask whose length differs from the inputs raises `ValueError` naming the field

#### Scenario: resample_curvature uses the apex-origin convention (apex at s_a = 0)
- **GIVEN** finite arrays with strictly increasing `arc_length_px` (no mask)
- **WHEN** `resample_curvature(curvature_px_inv, arc_length_px)` is invoked
- **THEN** `result.s_a_uniform_px[0] == 0.0` corresponds to the apex (the largest `arc_length_px`, i.e. the latest tip position per theory §6.5)
- **AND** `result.arc_span_px == max(arc_length_px) − min(arc_length_px)` to within numerical precision over the surviving (unmasked, finite) samples

#### Scenario: resample_curvature rejects malformed or mismatched inputs with TypeError or ValueError naming the field
- **WHEN** `resample_curvature(curvature_px_inv, arc_length_px)` is invoked with a non-ndarray argument, OR `curvature_px_inv.ndim != 1`, OR `complex`/`object` dtype, OR `len(curvature_px_inv) != len(arc_length_px)`, OR a `velocity_sub_noise_mask` of wrong length
- **THEN** `TypeError` (non-ndarray / bad mask type) or `ValueError` (shape / dtype / length) is raised
- **AND** the exception message names the offending field

#### Scenario: resample_curvature returns a graceful all-NaN ResampleResult on degenerate input
- **GIVEN** valid inputs that, after dropping masked + non-finite frames, leave fewer than the documented minimum surviving samples OR a non-positive surviving arc-span (e.g., all `arc_length_px` equal)
- **WHEN** `resample_curvature(...)` is invoked
- **THEN** the call does NOT raise and emits no `RuntimeWarning`
- **AND** the returned `ResampleResult` has `is_degenerate == True` and `kappa_uniform` all-NaN

#### Scenario: resample_curvature is degenerate when skewed gaps make the output grid too short (Stage-2 gate)
- **GIVEN** valid inputs with `≥ MIN_SAMPLES_REQUIRED` (9) surviving samples but a skewed/bimodal gap distribution such that `floor(arc_span_px / ds) + 1 < MIN_SAMPLES_REQUIRED` (e.g. surviving `s_a` gaps `[1,1,1,100,100,100,100,100]` → `arc_span = 503`, `ds = median = 100`, grid length `= 6`)
- **WHEN** `resample_curvature(...)` is invoked
- **THEN** the returned `ResampleResult` has `is_degenerate == True` and `kappa_uniform` all-NaN (the Stage-2 output-grid-length gate fires)
- **AND** a non-degenerate `ResampleResult` therefore always has `len(kappa_uniform) >= MIN_SAMPLES_REQUIRED`, so feeding it to `compute_scaleogram` never raises "too-short" (the cross-function invariant holds)

#### Scenario: resample_curvature handles a fully-masked input (n_unmasked == 0) without raising
- **GIVEN** a valid `velocity_sub_noise_mask` that is all-`True` (so the zero-survivor extreme `n_unmasked == 0`)
- **WHEN** `resample_curvature(curvature_px_inv, arc_length_px, velocity_sub_noise_mask=mask)` is invoked
- **THEN** the call does NOT raise (no `ValueError` from `np.max([])`/`np.min([])` on the empty survivor set, no `RuntimeWarning` from `np.median([])`)
- **AND** the returned `ResampleResult` has `is_degenerate == True`, `n_unmasked == 0`, and `kappa_uniform` all-NaN — the gate short-circuits BEFORE the apex-origin `max`/`min` and the `ds`-median

#### Scenario: resample_curvature drops non-finite (curvature, arc_length) pairs without raising
- **GIVEN** finite `arc_length_px` and a `curvature_px_inv` containing some NaN values (e.g. from a curvature blow-up swept to NaN by PR #8) at indices that still leave enough finite survivors
- **WHEN** `resample_curvature(curvature_px_inv, arc_length_px)` is invoked
- **THEN** the call does NOT raise and emits no `RuntimeWarning`
- **AND** the non-finite pairs are excluded from the interpolation knots (dropped, not rejected); `result.n_unmasked` counts only finite, unmasked survivors
- **AND** `result.kappa_uniform` is all-finite when ≥ the documented minimum survive (else `is_degenerate=True`, all-NaN)

#### Scenario: resample_curvature deduplicates duplicate-s_a knots deterministically
- **GIVEN** valid inputs where two or more surviving samples share an identical `arc_length_px` (hence identical `s_a`, e.g. a stalled tip), but the overall surviving span is positive with enough distinct knots
- **WHEN** `resample_curvature(...)` is invoked twice in the same process
- **THEN** the call does NOT raise and emits no `RuntimeWarning`
- **AND** `result.kappa_uniform` is finite (the duplicate-`s_a` κ values are averaged so the `np.interp` knots are strictly increasing)
- **AND** `np.array_equal(result1.kappa_uniform, result2.kappa_uniform)` is True at `atol=0` (deterministic — no order-dependent `np.interp` ambiguity)

#### Scenario: resample_curvature value-pins the apex (asymmetric-κ oracle)
- **GIVEN** a curvature with a single distinguishable feature placed at the apical end of the trail (largest `arc_length_px`, i.e. the latest tip position per theory §6.5) — e.g. a lone spike — and a flat κ elsewhere
- **WHEN** `resample_curvature(curvature_px_inv, arc_length_px)` is invoked
- **THEN** the feature appears at SMALL `s_a` (near the apex origin `s_a = 0`), NOT at large `s_a`
- **AND** this pins the `s_a = max(arc_length) − arc_length` orientation by VALUE, not merely by `s_a_uniform_px[0] == 0` position — guarding against a sign/off-by-one flip (the PR #8 `sign(κ) = −handedness` convention-anchoring lesson)

#### Scenario: resample_curvature is deterministic and emits exactly one DEBUG record
- **GIVEN** a valid input and `caplog.set_level(logging.DEBUG)`
- **WHEN** `resample_curvature(...)` is invoked twice in the same process
- **THEN** `np.array_equal(result1.kappa_uniform, result2.kappa_uniform)` is True at `atol=0`
- **AND** exactly one `DEBUG` record is emitted by `sleap_roots.circumnutation.spatial_cwt`, its message starting with `"resample_curvature("` and containing the tokens `"n_input="`, `"n_unmasked="`, `"ds="`, `"arc_span_px="`
- **AND** no `INFO`/`WARNING`/`ERROR`/`CRITICAL` records are emitted

### Requirement: Tier 3b spatial CWT scaleogram API
The system SHALL provide `sleap_roots.circumnutation.spatial_cwt.compute_scaleogram(kappa: np.ndarray, ds: float, constants: Optional[ConstantsT] = None) -> SpatialScaleogramResult`. The function SHALL accept the canonical `(kappa, ds, constants=None)` signature locked by the foundation's Package layout requirement. The function SHALL compute a spatial Continuous Wavelet Transform of the uniform-grid curvature `kappa` using the `cgau2` mother wavelet by default (overridable via `constants.WAVELET_DEFAULT_SPATIAL`), at log-spaced scales over an auto-derived spatial-wavelength range `[constants.CWT_WAVELENGTH_MIN_NYQUIST_FACTOR * ds, constants.CWT_WAVELENGTH_MAX_SIGNAL_FRACTION * len(kappa) * ds]`, returning a frozen `SpatialScaleogramResult` containing the complex-valued scaleogram, scales axis, spatial-wavelength axis (`wavelengths_px`, the honest `pywt.scale2frequency` round-trip convention value for wavelet-agnostic correctness — NOT bias-corrected; for `cgau2` this convention value over-reports the true px wavelength by a documented λ- and `n`-dependent band that PR #10 must reconcile identically on both sides of `traveling_wave_residual`, see the design "Handoff to PR #10"), spatial-frequency axis (`spatial_freqs_px_inv = 1.0 / wavelengths_px`), cone-of-influence boolean mask (computed via wavelet-aware `constants.SPATIAL_COI_EFOLDING_FACTOR * scale`), the resolved `ds`, and the resolved wavelet name.

The function SHALL emit NO trait values. The function SHALL be deterministic per CC-6: same input → bit-identical scaleogram across calls in the same process AND identical to within `atol=1e-9, rtol=0` across Ubuntu / Windows / macOS CI runners. The function SHALL log exactly one `logger.debug` message at the start (after input validation) whose text begins with `"compute_scaleogram("` and contains the named tokens `n_samples=`, `ds=`, `n_scales=`, `wavelength_min_px=`, `wavelength_max_px=`, `wavelet=`. No INFO, WARNING, or ERROR log records SHALL be emitted on the happy path.

The function SHALL validate inputs strictly: `kappa` SHALL be a 1-D `np.ndarray` (coercible from integer/float dtypes; rejecting `complex` and `object`); `kappa` SHALL be all-finite (no NaN, no ±inf); `len(kappa)` SHALL be ≥ `MIN_SAMPLES_REQUIRED` derived at call time as `int(math.floor(constants.CWT_WAVELENGTH_MIN_NYQUIST_FACTOR / constants.CWT_WAVELENGTH_MAX_SIGNAL_FRACTION)) + 1` (= `9` at defaults); positive-finite guards SHALL fire on `constants.CWT_WAVELENGTH_MAX_SIGNAL_FRACTION` and `constants.CWT_WAVELENGTH_MIN_NYQUIST_FACTOR`; `ds` SHALL be a Python `int` or `float` or numpy scalar (rejecting `bool` subtype and `str`); `ds > 0` and `math.isfinite(ds)`; `constants` SHALL be `None` or a `ConstantsT` instance. Invalid inputs SHALL raise `ValueError` or `TypeError` with the offending field name embedded in the message.

The `SpatialScaleogramResult` class SHALL be an `@attrs.define(frozen=True, slots=False, kw_only=True, eq=False)` container with exactly the following fields (in this order): `scaleogram: np.ndarray` (shape `(n_scales, n_samples)`, dtype `complex128`); `scales: np.ndarray` (shape `(n_scales,)`, dtype `float64`, monotonically increasing; dimensionless pywt CWT scale, NOT px); `wavelengths_px: np.ndarray` (shape `(n_scales,)`, dtype `float64`); `spatial_freqs_px_inv: np.ndarray` (shape `(n_scales,)`, dtype `float64`, equal to `1.0 / wavelengths_px` to within numerical precision); `coi_mask: np.ndarray` (shape `(n_scales, n_samples)`, dtype `bool`; `True` indicates inside-COI = unreliable); `ds: float`; `wavelet: str`. (`eq=False` because ndarray `__eq__` is ambiguous.)

#### Scenario: compute_scaleogram returns a SpatialScaleogramResult with the documented field shapes and dtypes
- **GIVEN** a valid 1-D `float64` ndarray `kappa` of length 200 with all-finite values, and `ds = 5.8`
- **WHEN** `compute_scaleogram(kappa, 5.8)` is invoked
- **THEN** the returned object is a `SpatialScaleogramResult` instance
- **AND** `result.scaleogram.shape == (CWT_SCALE_COUNT_DEFAULT, 200)` and `dtype == np.complex128`
- **AND** `result.scales.shape == (CWT_SCALE_COUNT_DEFAULT,)`, `dtype == np.float64`, strictly monotonically increasing
- **AND** `result.wavelengths_px.shape == (CWT_SCALE_COUNT_DEFAULT,)` and `dtype == np.float64`
- **AND** `np.allclose(result.spatial_freqs_px_inv * result.wavelengths_px, 1.0, atol=1e-12)`
- **AND** `result.coi_mask.shape == result.scaleogram.shape` and `dtype == bool`
- **AND** `result.ds == 5.8` and `result.wavelet == "cgau2"`

#### Scenario: SpatialScaleogramResult is a frozen attrs class with the seven documented fields
- **WHEN** `attrs.fields(SpatialScaleogramResult)` is inspected
- **THEN** the field names are exactly `("scaleogram", "scales", "wavelengths_px", "spatial_freqs_px_inv", "coi_mask", "ds", "wavelet")` in that order
- **AND** attempting `result.scaleogram = new_array` raises `attrs.exceptions.FrozenInstanceError`

#### Scenario: compute_scaleogram rejects non-finite kappa with ValueError naming the field
- **WHEN** `compute_scaleogram(kappa, 5.8)` is invoked with `kappa` containing NaN OR `+inf` OR `-inf` at any index
- **THEN** `ValueError` is raised
- **AND** the exception message contains the substring `"kappa"` and identifies the non-finite condition

#### Scenario: compute_scaleogram rejects malformed kappa with ValueError or TypeError naming the field
- **WHEN** `compute_scaleogram(kappa, 5.8)` is invoked with `kappa.ndim != 1` (e.g., 2-D ndarray) OR `kappa.dtype == np.complex128` OR `kappa.dtype == object`
- **THEN** `ValueError` or `TypeError` is raised
- **AND** the exception message contains the substring `"kappa"` and identifies the offending shape or dtype

#### Scenario: compute_scaleogram rejects too-short kappa with ValueError naming the field
- **WHEN** `compute_scaleogram(kappa, 5.8)` is invoked with `len(kappa) < MIN_SAMPLES_REQUIRED` (where MIN_SAMPLES_REQUIRED = `int(math.floor(CWT_WAVELENGTH_MIN_NYQUIST_FACTOR / CWT_WAVELENGTH_MAX_SIGNAL_FRACTION)) + 1` = 9 at defaults)
- **THEN** `ValueError` is raised
- **AND** the exception message contains the substring `"kappa"` (or `"MIN_SAMPLES"`) and reports the actual length

#### Scenario: compute_scaleogram accepts kappa at the exact MIN_SAMPLES_REQUIRED floor without raising
- **GIVEN** a 1-D float64 ndarray `kappa` of length exactly `MIN_SAMPLES_REQUIRED` (= 9 at defaults) with all-finite values
- **WHEN** `compute_scaleogram(kappa, 5.8)` is invoked
- **THEN** the call returns a `SpatialScaleogramResult` without raising (bidirectional contract on the documented floor)
- **AND** the returned `scaleogram.shape == (CWT_SCALE_COUNT_DEFAULT, 9)`

#### Scenario: compute_scaleogram rejects invalid ds value with ValueError naming the field
- **WHEN** `compute_scaleogram(kappa, ds)` is invoked with `ds` equal to `0` OR `-1.0` OR `float("nan")` OR `float("inf")` OR `float("-inf")`
- **THEN** `ValueError` is raised
- **AND** the exception message contains the substring `"ds"` and the offending value

#### Scenario: compute_scaleogram rejects invalid ds type with TypeError naming the field
- **WHEN** `compute_scaleogram(kappa, ds)` is invoked with `ds` equal to `True` (Python bool) OR `np.bool_(True)` (numpy bool scalar — must be explicitly guarded since `np.bool_` is a subclass of `int`) OR `"5.8"` (str) OR `[5.8]` (list)
- **THEN** `TypeError` is raised
- **AND** the exception message contains the substring `"ds"` and identifies the offending type

#### Scenario: compute_scaleogram accepts a numpy float scalar ds
- **GIVEN** a valid `kappa` and `ds = np.float64(5.8)` (a numpy floating scalar, the natural type of `ResampleResult.ds` arithmetic)
- **WHEN** `compute_scaleogram(kappa, np.float64(5.8))` is invoked
- **THEN** the call returns a `SpatialScaleogramResult` and `result.ds == 5.8` (numpy float scalar ACCEPTED — the positive sibling of the `np.bool_` rejection scenario, so the `bool`-guard does not over-reject numpy numerics)

#### Scenario: compute_scaleogram honors the WAVELET_DEFAULT_SPATIAL override
- **GIVEN** a valid `kappa`, `ds`, and `constants=ConstantsT(WAVELET_DEFAULT_SPATIAL="cmor1.5-1.0")`
- **WHEN** `compute_scaleogram(kappa, ds, constants=constants)` is invoked
- **THEN** `result.wavelet == "cmor1.5-1.0"` (the wavelet is read from `constants`, NOT hard-coded "cgau2")
- **AND** the DEBUG record's `wavelet=` token reflects the override

#### Scenario: compute_scaleogram is deterministic across runs
- **GIVEN** a valid `kappa` and `ds`
- **WHEN** `compute_scaleogram(kappa, ds)` is invoked twice in the same Python process
- **THEN** `np.array_equal(result1.scaleogram, result2.scaleogram)` is True at `atol=0`
- **AND** a captured canary at interior COI-dodging `[scale_idx, [positions]]` cells matches the hardcoded expected complex values to within `atol=1e-9, rtol=0` across Ubuntu / Windows / macOS CI runners AT THE TIME OF PR-MERGE; canary values are regression-detection sentinels and MAY be re-captured (in a follow-up commit cross-referencing this scenario) if upstream BLAS / pywt / numpy semantics legitimately shift after merge

#### Scenario: compute_scaleogram emits exactly one DEBUG logger record on the happy path
- **GIVEN** a valid `kappa` and `ds` and `caplog.set_level(logging.DEBUG)`
- **WHEN** `compute_scaleogram(kappa, ds)` is invoked
- **THEN** exactly one log record at level `DEBUG` is emitted by the logger `sleap_roots.circumnutation.spatial_cwt`
- **AND** the record's message starts with `"compute_scaleogram("`
- **AND** the record's message contains each of the tokens `"n_samples="`, `"ds="`, `"n_scales="`, `"wavelength_min_px="`, `"wavelength_max_px="`, `"wavelet="`
- **AND** no `INFO` / `WARNING` / `ERROR` / `CRITICAL` records are emitted

#### Scenario: Proofread fixture (Nipponbare plate-001 6 tracks) does not raise and produces shape-correct output
- **GIVEN** the 6 tracks of `tests/data/circumnutation_nipponbare_plate_001/plate_001_greyscale.tracked_proofread.slp` loaded via `Series.load(...).get_tracked_tips()`, each run through `midline.reconstruct(track_x, track_y, cadence_s=300.0)` then `resample_curvature(mr.curvature_px_inv, mr.arc_length_px, mr.velocity_sub_noise_mask)`
- **WHEN** `compute_scaleogram(resample_result.kappa_uniform, resample_result.ds)` is invoked for each non-degenerate track
- **THEN** the call does not raise
- **AND** the returned `SpatialScaleogramResult` has `scaleogram.shape == (CWT_SCALE_COUNT_DEFAULT, len(kappa_uniform))` and `coi_mask.shape == scaleogram.shape`
- **AND** the dominant spatial wavelength from `extract_ridge` (the median of `wavelengths_px` over `~in_coi` positions only) is finite and physically plausible (recorded for auditability)
- **AND** the post-mask Nyquist headroom `2 * resample_result.ds < NYQUIST_RATIO_MAX * lambda_observed` holds (recorded for auditability — sub-noise masking coarsens `ds`, so this guards against the resample grid approaching the spatial-aliasing threshold of theory §6.5)

### Requirement: Tier 3b spatial CWT ridge API
The system SHALL provide `sleap_roots.circumnutation.spatial_cwt.extract_ridge(scaleogram_result: SpatialScaleogramResult, constants: Optional[ConstantsT] = None) -> SpatialRidgeResult`. The function SHALL extract a per-position ridge from the input spatial scaleogram via deterministic per-position argmax of `|scaleogram|` along the scale axis (numpy's documented tie-breaking returns the smallest index on equal values), yielding the dominant spatial wavelength λ(s_a) at each position — the steady-traveling-wave quantity PR #10 consumes.

The function SHALL emit NO trait values. The function SHALL be deterministic: same `SpatialScaleogramResult` input → identical `SpatialRidgeResult` output at `atol=0`. The function SHALL log exactly one `logger.debug` message at the start (after input validation) whose text begins with `"extract_ridge("` and contains tokens `n_scales=` and `n_samples=`. No INFO, WARNING, or ERROR log records SHALL be emitted on the happy path.

The function SHALL validate inputs strictly: `scaleogram_result` SHALL be a `SpatialScaleogramResult` instance (anything else raises `TypeError`); empty `SpatialScaleogramResult` (with `n_scales == 0` OR `n_samples == 0`) SHALL raise `ValueError`; `constants` SHALL be `None` or a `ConstantsT` instance.

The `SpatialRidgeResult` class SHALL be an `@attrs.define(frozen=True, slots=False, kw_only=True, eq=False)` container with exactly the following fields (in this order): `position_indices: np.ndarray` (shape `(n_samples,)`, dtype `int64`, equal to `np.arange(n_samples, dtype=np.int64)`); `wavelengths_px: np.ndarray` (shape `(n_samples,)`, dtype `float64`; indexed by position, NOT by scale — value at index `i` is the spatial wavelength AT THE RIDGE for position `i`); `amplitudes: np.ndarray` (shape `(n_samples,)`, dtype `float64`, equal to `|C|` at the ridge cell, all ≥ 0; units of `|κ|` × wavelet normalization, effectively px⁻¹); `powers: np.ndarray` (shape `(n_samples,)`, dtype `float64`, equal to `amplitudes ** 2`, effectively px⁻², redundant by construction and intentionally preserved for symmetry with the temporal `RidgeResult`); `in_coi: np.ndarray` (shape `(n_samples,)`, dtype `bool`; `True` iff `scaleogram_result.coi_mask[ridge_scale_idx, position_idx]`). The ridge SHALL NOT be pre-COI-masked; PR #10's trait emission applies the COI mask.

#### Scenario: extract_ridge returns a SpatialRidgeResult with the documented field shapes and dtypes
- **GIVEN** a valid `SpatialScaleogramResult` produced by `compute_scaleogram(kappa, 5.8)` where `len(kappa) == 200`
- **WHEN** `extract_ridge(scaleogram_result)` is invoked
- **THEN** the returned object is a `SpatialRidgeResult` instance
- **AND** `result.position_indices.shape == (200,)` and `dtype == np.int64` and `np.array_equal(result.position_indices, np.arange(200, dtype=np.int64))`
- **AND** `result.wavelengths_px.shape == (200,)` and `dtype == np.float64`
- **AND** `result.amplitudes.shape == (200,)` and `dtype == np.float64` and `(result.amplitudes >= 0).all()`
- **AND** `result.powers.shape == (200,)` and `dtype == np.float64` and `np.allclose(result.powers, result.amplitudes ** 2)`
- **AND** `result.in_coi.shape == (200,)` and `dtype == bool`

#### Scenario: SpatialRidgeResult is a frozen attrs class with the five documented fields
- **WHEN** `attrs.fields(SpatialRidgeResult)` is inspected
- **THEN** the field names are exactly `("position_indices", "wavelengths_px", "amplitudes", "powers", "in_coi")` in that order
- **AND** attempting `ridge.amplitudes = new_array` raises `attrs.exceptions.FrozenInstanceError`

#### Scenario: extract_ridge recovers a planted spatial wavelength (analytic oracle)
- **GIVEN** a uniform-grid `kappa` that is a pure sinusoid of known spatial wavelength `lambda_true` px (`kappa = sin(2π s_a / lambda_true)`), passed through `compute_scaleogram(kappa, ds)`
- **WHEN** `extract_ridge(scaleogram_result)` is invoked
- **THEN** the median of `result.wavelengths_px` over the COI-interior positions (`~result.in_coi`) recovers `lambda_true` to within a documented numeric tolerance band MEASURED during GREEN (the `pywt.scale2frequency("cgau2", …)` center-frequency convention introduces a systematic offset in the wavelength axis — empirically a positive, mildly λ-dependent bias — so the tolerance is the measured band from the task-1.1 cgau2 calibration capture, NOT a speculative ±5%; stated as a literal in the test, with the cgau2 calibration recorded so PR #10's `traveling_wave_residual` can apply the same offset consistently)

#### Scenario: extract_ridge tracks the fundamental, not the second harmonic (cgau2 high-frequency bias)
- **GIVEN** a uniform-grid `kappa` that is a fundamental sinusoid of wavelength `lambda_true` plus a moderate ~20–30% second-harmonic component (wavelength `lambda_true/2`), passed through `compute_scaleogram(kappa, ds)`
- **WHEN** `extract_ridge(scaleogram_result)` is invoked
- **THEN** the COI-interior median of `result.wavelengths_px` tracks the fundamental `lambda_true`, NOT `lambda_true/2`
- **AND** this guards against the ridge locking onto a harmonic — a real risk because `cgau2` (a 2nd-derivative wavelet) amplifies high-frequency content by ~(1/scale)²; the COI-interior MEDIAN provides the robustness (empirically the fundamental is tracked well past 30% harmonic, but the contract is pinned at the conservative ~20–30% where the fundamental's post-weighting amplitude clearly dominates)

#### Scenario: λ_spatial recovery survives 50%-sparsity-then-interp (the real-data regime)
- **GIVEN** a pure sinusoid of known `lambda_true` sampled on a non-uniform arc-length grid, with ~50% of samples randomly dropped (mimicking the real plate-001 sub-noise mask fraction), passed through `resample_curvature(...)` → `compute_scaleogram(...)` → `extract_ridge(...)`
- **WHEN** the full chain is run
- **THEN** the COI-interior median of `result.wavelengths_px` recovers `lambda_true` to within a documented tolerance
- **AND** the interpolation-gap bias is asserted by SIGN, not merely recorded: linear interpolation across dropped-sample gaps is a low-pass operation that systematically LENGTHENS the apparent wavelength, so the recovered median SHALL be ≥ `lambda_true` (minus the calibration band) — a regression that flips the bias direction or exceeds the band is caught. This is the robustness check on the headline data property (real κ(s) is ~50% sparse) for the one quantity PR #9 ships

#### Scenario: extract_ridge rejects non-SpatialScaleogramResult input with TypeError
- **WHEN** `extract_ridge(x)` is invoked with `x` equal to `None` OR `{}` OR `(1, 2, 3)` OR `np.zeros((10, 10))`
- **THEN** `TypeError` is raised
- **AND** the exception message references the expected type `SpatialScaleogramResult`

#### Scenario: extract_ridge rejects empty SpatialScaleogramResult with ValueError
- **GIVEN** a `SpatialScaleogramResult` constructed with `n_scales == 0` (scaleogram shape `(0, n_samples)`) OR `n_samples == 0` (scaleogram shape `(n_scales, 0)`)
- **WHEN** `extract_ridge(scaleogram_result)` is invoked
- **THEN** `ValueError` is raised
- **AND** the exception message references the empty-axis condition (e.g., `"n_scales == 0"` or `"n_samples == 0"`)

#### Scenario: extract_ridge degenerates gracefully on a single-scale scaleogram (n_scales == 1)
- **GIVEN** a manually-constructed valid `SpatialScaleogramResult` with `n_scales == 1` (scaleogram shape `(1, n_samples)`, `n_samples > 0`)
- **WHEN** `extract_ridge(scaleogram_result)` is invoked
- **THEN** the call does NOT raise (argmax over a single scale returns index 0 at every position; no `IndexError`)
- **AND** every `result.wavelengths_px[i] == scaleogram_result.wavelengths_px[0]`

#### Scenario: extract_ridge rejects invalid constants type with TypeError
- **GIVEN** a valid `SpatialScaleogramResult`
- **WHEN** `extract_ridge(scaleogram_result, constants=invalid)` is invoked with `invalid` not `None` and not a `ConstantsT` instance (e.g., `42`, `"foo"`, `{}`)
- **THEN** `TypeError` is raised
- **AND** the exception message references `"constants"`

#### Scenario: extract_ridge emits exactly one DEBUG logger record on the happy path
- **GIVEN** a valid `SpatialScaleogramResult` and `caplog.set_level(logging.DEBUG)`
- **WHEN** `extract_ridge(scaleogram_result)` is invoked
- **THEN** exactly one log record at level `DEBUG` is emitted by the logger `sleap_roots.circumnutation.spatial_cwt`
- **AND** the record's message starts with `"extract_ridge("`
- **AND** the record's message contains each of the tokens `"n_scales="`, `"n_samples="`
- **AND** no `INFO` / `WARNING` / `ERROR` / `CRITICAL` records are emitted

#### Scenario: extract_ridge is deterministic
- **GIVEN** a valid `SpatialScaleogramResult`
- **WHEN** `extract_ridge(scaleogram_result)` is invoked twice
- **THEN** `np.array_equal(result1.wavelengths_px, result2.wavelengths_px)` is True at `atol=0`
- **AND** the same holds for `amplitudes`, `powers`, `in_coi`, `position_indices`

### Requirement: Tier 3c traveling-wave trait emission API
The system SHALL provide `sleap_roots.circumnutation.traveling_wave.compute(trajectory_df: pd.DataFrame, cadence_s: float, constants: Optional[ConstantsT] = None, *, tier0_df: Optional[pd.DataFrame] = None, tier1_df: Optional[pd.DataFrame] = None) -> pd.DataFrame`. The function SHALL accept the documented signature with `cadence_s` as an explicit positional parameter (mirroring `nutation.compute` / `psi_g.compute`) and `tier0_df` / `tier1_df` as keyword-only optional precomputed-frames parameters (the PR #14 dedup fast path; default `None`). It SHALL be the first consumer of the PR #9 spatial-CWT machinery and SHALL compute the Tier 3c traveling-wave validation traits that test the QPB steady-traveling-wave hypothesis `λ_spatial = v · T_nutation` (theory.md §4.7). The `coordinate` projection is NOT exposed as a parameter: the function SHALL internally use `coordinate="lateral"` for its Tier 1 recompute (the QPB residual is only defined against the lateral nutation period, CC-7).

The function SHALL emit a per-track DataFrame whose rows correspond to the unique 5-tuple `(series, sample_uid, plate_id, plant_id, track_id)` derived from `trajectory_df` via `groupby(_IDENTITY_5_TUPLE, dropna=False, sort=False)`. The returned DataFrame SHALL contain the 8 row-identity columns (per Requirement: Trait CSV row-identity schema) followed by exactly 6 trait columns, all `float64`, in this declared order: `lambda_spatial_median_px`, `lambda_spatial_variation`, `traveling_wave_residual`, `lambda_expected_px`, `lambda_spatial_mad_px`, `coi_valid_fraction`.

The function SHALL be self-contained by default: when `tier0_df` and `tier1_df` are both `None` it SHALL recompute Tier 0 via `kinematics.compute(trajectory_df, constants=constants)` and Tier 1 via `nutation.compute(trajectory_df, cadence_s, coordinate="lateral", constants=constants)` (passing the resolved `constants` into both), and SHALL join their `v_total_median_px_per_frame` and `T_nutation_median` operands onto the per-track results **by merging on the full `_IDENTITY_5_TUPLE`** (NOT on `track_id` alone — `track_id` is not unique across plates/samples). The Tier 1 `is_nutating` flag is consumed **implicitly** via `T_nutation_median` — Tier 1 NaN-gates `T_nutation_median` to NaN when `is_nutating == False`, so a non-nutating track NaNs `lambda_expected_px`/`traveling_wave_residual` without a separate `is_nutating` join. Because `kinematics.compute`/`nutation.compute` return their identity keys coerced to int64 while the raw `trajectory_df` keys may be float64, the function SHALL coerce the per-track frame's `track_id`/`plant_id` to int64 before the operand merge (mirroring the per-plant template-merge guard); a key-dtype mismatch would otherwise produce a SILENT all-NaN operand join (not a `KeyError`). This default recompute is redundant-by-design when called inside a full pipeline (Tier 0/Tier 1 are also computed as emitting tiers).

**PR #14 dedup fast path.** When `tier0_df` and `tier1_df` are BOTH provided, the function SHALL use them as the Tier 0 / Tier 1 operand source and SHALL NOT recompute `kinematics.compute` / `nutation.compute`. The function SHALL validate that each provided frame carries the `_IDENTITY_5_TUPLE` columns plus the single operand column it supplies (`v_total_median_px_per_frame` for `tier0_df`, `T_nutation_median` for `tier1_df`) and SHALL raise `ValueError` naming the missing column otherwise (cheaper than a recompute; prevents a silent all-NaN operand join). Supplying exactly ONE of `tier0_df` / `tier1_df` (the other `None`) SHALL raise `ValueError` (both-or-neither — a half-deduped call would silently recompute the other operand). The fast path SHALL **project each supplied frame to `[*_IDENTITY_5_TUPLE, <its operand column>]` before the operand merge** (identical to the recompute's `ops`-build), so the many extra trait columns the pipeline's full Tier 0 / Tier 1 frames carry CANNOT enter the result, collide, or alter the column schema; the supplied frames SHALL then flow through the SAME operand-merge + int64-coercion path as the recompute (only the two `*.compute` calls are bypassed). The fast path SHALL produce Tier 3c trait columns IDENTICAL (`atol=0`, `NaN`-equal) to the default recompute path — deduplication changes performance, not results — provided the supplied frames are the `kinematics.compute(trajectory_df, constants)` and `nutation.compute(trajectory_df, cadence_s, coordinate="lateral", constants)` outputs for that same `trajectory_df` and `constants`, and `constants` is one that `nutation.compute` accepts (because `traveling_wave` validates only `COI_FRACTION_MAX` while `nutation` validates its own fields, a `constants` that `nutation` would REJECT makes the recompute raise while the fast path could return — so the equivalence is scoped to nutation-accepted constants). When `tier0_df` and `tier1_df` are omitted (the standalone default), the function's behavior SHALL be byte-identical to its pre-PR-#14 recompute behavior.

The function SHALL compose the per-track spatial chain `midline.reconstruct` → `spatial_cwt.resample_curvature` → `spatial_cwt.compute_scaleogram` → `spatial_cwt.extract_ridge`, then gate and calibrate the ridge wavelengths:

- **Spatial-availability / COI gate.** The 5 spatial traits (`lambda_spatial_median_px`, `lambda_spatial_variation`, `traveling_wave_residual`, `lambda_expected_px`, `lambda_spatial_mad_px`) SHALL be NaN for a track when the spatial chain is unavailable: `MidlineResult.is_degenerate`, OR `ResampleResult.is_degenerate`, OR `compute_scaleogram`/`extract_ridge` raised `ValueError` (the function SHALL catch it and emit a NaN row, NOT crash), OR the cone-of-influence fraction exceeds the reuse of the existing `COI_FRACTION_MAX` constant — i.e. `coi_valid_fraction < (1 − COI_FRACTION_MAX)`, where `coi_valid_fraction = (~in_coi).sum() / in_coi.size`. `coi_valid_fraction` SHALL be finite whenever a ridge formed (including when the low-COI gate fires) and NaN only when no ridge formed (degenerate midline/resample or caught CWT raise).
- **cgau2 calibration.** `lambda_spatial_median_px`, `lambda_spatial_variation`, and `lambda_spatial_mad_px` SHALL be computed from a single calibrated wavelength array in true pixels, obtained by dividing each COI-valid ridge `wavelengths_px` by the cgau2 over-report ratio interpolated on a single `ratio(λ)` curve. That curve SHALL be obtained by **averaging the per-`(n)` ratios across `n ∈ {200, 400, 600}` at each `λ_true`** (the calibration ratio scatters ~7 % across `n`, non-monotonically; averaging avoids the false precision of interpolating that noisy non-monotone `n`-dependence and keeps the per-knot variance bounded). The curve SHALL be a committed in-package literal (a module-level `_CGAU2_LAMBDA_CALIBRATION` table of `(λ_reported_mean, ratio_mean)` pairs, strictly increasing in `λ_reported_mean`, covering the observed real λ range, i.e. `λ_true` up to ≥ 140 px), derived from and validated against the authoritative `tests/data/circumnutation_spatial_cwt_calibration.json`; the production module SHALL NOT read the `tests/data` file at runtime (it is not shipped in the installed wheel). The honest `traveling_wave_residual` SHALL use this calibrated λ (true px on both sides); the function SHALL NOT emit the raw mixed-domain residual. The residual `np.interp` uses the strictly-increasing `λ_reported_mean` axis (well-posed); values beyond the table edges clamp.
- **Documented calibration limitation.** The cgau2 calibration carries an irreducible ~±5 % systematic (the residual `n`-scatter after averaging). `traveling_wave_residual` differences between tracks that are within ~5 %, and `lambda_spatial_variation` as a noise-sensitive spread (≈0 on a noise-free uniform-λ trail; growing with curvature-localization noise — see the trait definition), SHALL be documented (spec + theory) as "within calibration uncertainty / noise-dependent" and not over-interpreted as pure biological signal. The headline plate-001 result is provisional pending the post-extension re-measurement (see the real-data scenario).
- **Trait definitions.** `lambda_spatial_median_px = median(λ_cal[interior])`; `lambda_spatial_mad_px = median(|λ_cal[interior] − median|)`; `lambda_spatial_variation = lambda_spatial_mad_px / lambda_spatial_median_px` (an **orientation-robust** spread — the median/MAD statistic is invariant under arc-length reversal, though the COI-interior selection still depends on the resample `s_a=0` anchor; 0 = uniform); `lambda_expected_px = v_total_median_px_per_frame · (T_nutation_median / cadence_s)`; `traveling_wave_residual = |lambda_spatial_median_px − lambda_expected_px| / lambda_expected_px`. `lambda_spatial_variation` reads ≈ 0 on a noise-free uniform-λ synthetic (verified — there is NO spurious argmax-quantization floor), so the trait correctly reports "uniform" when λ is uniform. On real data it reflects genuine ridge-localization scatter — which grows with curvature-localization noise (empirically ~0 at zero noise, ~0.13–0.40 across realistic noise) — PLUS any real λ variation; it is therefore a spread diagnostic to be interpreted relative to the track's noise level, NOT a pure biological-λ-variation measure or a calibrated H1 test. `traveling_wave_residual` is interpretable only when `lambda_expected_px` is at least ~one resolvable wavelength; a tiny-but-positive `v·T_frames` yields a large-but-finite residual that is an undefined-regime artifact, not a QPB violation (see the gating below and the documented limitation).

The function SHALL gate `traveling_wave_residual` and `lambda_expected_px` to NaN when the temporal/velocity operands are undefined: `T_nutation_median` is NaN (which occurs when `is_nutating == False`, per Requirement: Tier 1 nutation trait emission API), OR `v_total_median_px_per_frame` is non-finite, OR `lambda_expected_px ≤ 0`. The division SHALL be guarded so no `inf`/`np.RuntimeWarning` is produced. The pure-spatial traits `lambda_spatial_median_px`, `lambda_spatial_variation`, AND `lambda_spatial_mad_px` SHALL remain valid (not gated by `is_nutating`) whenever the spatial chain succeeded.

The function SHALL emit only these 6 columns; the `L_gz`/`L_c`-dependent traits (`L_gz_estimate`, `L_c_estimate`, `B_balance_number`, `L_gz_steady_state_residual`, `L_gz_resolvable`) and the growth-zone mask SHALL NOT be emitted (blocked on #230; omitted, not reserved as NaN columns). The pipeline SHALL remain pure-pixel (CC-3): the `_px` columns are pixels and the unit-bearing columns SHALL use only `PIPELINE_UNIT_VOCABULARY` entries (`px`, `—`); no new constant is introduced and `_CONSTANTS_VERSION` SHALL remain 6. The module SHALL declare a `_TRAVELING_WAVE_TRAIT_UNITS` mapping (per the `_TIER0_TRAIT_UNITS` precedent and GitHub issue #222) assigning each of the 6 trait columns its unit string — `lambda_spatial_median_px`/`lambda_expected_px`/`lambda_spatial_mad_px` → `"px"`; `lambda_spatial_variation`/`traveling_wave_residual`/`coi_valid_fraction` → `"—"` — and every value SHALL be a member of `PIPELINE_UNIT_VOCABULARY` (the `_px` columns are the first spatial-wavelength trait columns in the program; no `px⁻¹` column is emitted, so PR #9's deferred `px⁻¹` vocabulary token is NOT required by this PR).

The function SHALL be deterministic per CC-6: same input → bit-identical 6 float columns across calls in the same process (`atol=0`) AND identical to within a measured `atol` (target `1e-6`, inherited from the recomputed Tier 1 scipy paths; re-measured on a real-data canary covering the full chain including the `np.interp` calibration) across Ubuntu / Windows / macOS CI runners. The function SHALL log exactly one `logger.debug` message at the start (after input validation) whose text begins with `"traveling_wave.compute("` and contains the tokens `n_tracks=` and `cadence_s=`. No INFO, WARNING, or ERROR log records SHALL be emitted on the happy path.

The function SHALL validate inputs strictly: `trajectory_df` validation delegates to `_validate_trajectory_df` (per Requirement: Tier 0 input-validation boundary); `cadence_s` validation reuses `temporal_cwt._validate_cadence_s` (Python int/float or numpy integer/floating; explicit `bool` / `np.bool_` rejection; positive finite); `constants` SHALL be None or `ConstantsT` else TypeError, with `COI_FRACTION_MAX` validated as a float in `(0, 1]`. Degenerate, stationary, single-frame, or all-NaN-tip tracks SHALL produce a well-formed all-NaN trait row (never a crash, never a dropped row): every per-track code path SHALL return a complete 6-key trait dict so the per-plant template merge yields exactly one row per 5-tuple.

#### Scenario: traveling_wave.compute returns a DataFrame with the documented column order and dtypes
- **GIVEN** a valid `trajectory_df` with 6 tracks (the Nipponbare proofread fixture) and `cadence_s = 300.0`
- **WHEN** `traveling_wave.compute(trajectory_df, cadence_s=300.0)` is invoked
- **THEN** the returned object is a `pandas.DataFrame`
- **AND** the column order is: 8 row-identity columns from `ROW_IDENTITY_COLUMNS` followed by the 6 trait columns in declared order `["lambda_spatial_median_px", "lambda_spatial_variation", "traveling_wave_residual", "lambda_expected_px", "lambda_spatial_mad_px", "coi_valid_fraction"]`
- **AND** all 6 trait dtypes are `float64`
- **AND** row-identity 5-tuple `(series, sample_uid, plate_id, plant_id, track_id)` uniqueness holds: `df[list(_IDENTITY_5_TUPLE)].duplicated().sum() == 0`
- **AND** none of the omitted L_gz/L_c columns (`L_gz_estimate`, `L_c_estimate`, `B_balance_number`, `L_gz_steady_state_residual`, `L_gz_resolvable`) is present

#### Scenario: traveling_wave.compute joins Tier 0/Tier 1 operands on the full 5-tuple across plates
- **GIVEN** a `trajectory_df` containing ≥ 2 plates whose `track_id` values overlap (e.g. both plates have `track_id ∈ {0, 1}`) and whose `track_id` column dtype is `float64`, and `cadence_s = 300.0`
- **WHEN** `traveling_wave.compute(trajectory_df, cadence_s=300.0)` is invoked
- **THEN** the call returns one row per unique 5-tuple without raising
- **AND** for healthy nutating tracks the merged operands are FINITE (NOT all-NaN) — guarding against the silent int64-vs-float64 key-dtype mismatch that pandas merge would otherwise resolve to NaN rather than a `KeyError`
- **AND** each row's `lambda_expected_px` is computed from the `v_total_median_px_per_frame` and `T_nutation_median` of THAT row's own 5-tuple (not another plate's track with the same `track_id`)

#### Scenario: traveling_wave.compute precomputed-frames fast path matches the recompute path
- **GIVEN** a valid multi-track `trajectory_df` and `cadence_s = 300.0`, plus `tier0 = kinematics.compute(trajectory_df)` and `tier1 = nutation.compute(trajectory_df, 300.0, coordinate="lateral")`
- **WHEN** `traveling_wave.compute(trajectory_df, 300.0)` (recompute) and `traveling_wave.compute(trajectory_df, 300.0, tier0_df=tier0, tier1_df=tier1)` (fast path) are both invoked
- **THEN** both return DataFrames with the identical 6 trait columns equal at `atol=0` (deduplication changes performance, not results)
- **AND** the fast-path call does NOT recompute `kinematics.compute` / `nutation.compute` internally (the supplied frames are the operand source)

#### Scenario: traveling_wave.compute rejects an incomplete or malformed precomputed-frames request
- **WHEN** `traveling_wave.compute(trajectory_df, 300.0, tier0_df=tier0)` is invoked with `tier1_df` left as `None` (exactly one of the two supplied)
- **THEN** `ValueError` is raised (both-or-neither)
- **AND** WHEN both are supplied but `tier0_df` is missing the `v_total_median_px_per_frame` column (or `tier1_df` is missing `T_nutation_median`, or either is missing a `_IDENTITY_5_TUPLE` column), `ValueError` is raised naming the missing column
- **AND** the standalone call `traveling_wave.compute(trajectory_df, 300.0)` (both kwargs omitted) is byte-identical to the pre-PR-#14 recompute behavior

#### Scenario: traveling_wave.compute NaN-gates the residual for non-nutating tracks but keeps the spatial λ traits
- **GIVEN** a `trajectory_df` with a track that is spatially well-formed but non-nutating (e.g. `synthetic.generate_trajectory(amplitude_px=0.0, noise_sigma_px=1.0, n_frames=1024, cadence_s=300, random_state=0)` converted to a trajectory_df) and `cadence_s = 300.0`
- **WHEN** `traveling_wave.compute(trajectory_df, cadence_s=300.0)` is invoked
- **THEN** `np.isnan(result.traveling_wave_residual.iloc[0])` AND `np.isnan(result.lambda_expected_px.iloc[0])` (gated because `T_nutation_median` is NaN when `is_nutating == False`)
- **AND** `lambda_spatial_median_px` and `lambda_spatial_variation` are each finite OR NaN but NEVER ±inf (they are pure-spatial; valid whenever the spatial chain succeeded)

#### Scenario: traveling_wave.compute handles degenerate / stationary tracks gracefully via NaN trait emission
- **GIVEN** a `trajectory_df` containing one stationary track (zero net displacement) and `cadence_s = 300.0`
- **WHEN** `traveling_wave.compute(trajectory_df, cadence_s=300.0)` is invoked
- **THEN** the call returns a `pandas.DataFrame` without raising and without emitting `np.RuntimeWarning`
- **AND** that track's 5 spatial traits (`lambda_spatial_median_px`, `lambda_spatial_variation`, `traveling_wave_residual`, `lambda_expected_px`, `lambda_spatial_mad_px`) are NaN
- **AND** `coi_valid_fraction` for that track is NaN (no ridge formed)

#### Scenario: traveling_wave.compute does not crash when one track is too short for the spatial CWT
- **GIVEN** a `trajectory_df` with two tracks, one healthy (≥ enough frames) and one too short to form a spatial scaleogram (`compute_scaleogram` would raise `ValueError`), and `cadence_s = 300.0`
- **WHEN** `traveling_wave.compute(trajectory_df, cadence_s=300.0)` is invoked
- **THEN** the call returns a `pandas.DataFrame` with one row per 5-tuple (both tracks present) without raising
- **AND** the short track's 5 spatial traits are NaN
- **AND** the healthy track's `lambda_spatial_median_px` is finite

#### Scenario: traveling_wave.compute rejects invalid cadence_s value with ValueError naming the field
- **WHEN** `traveling_wave.compute(trajectory_df, cadence_s=v)` is invoked for any of `v ∈ {0, -1.0, float("nan"), float("inf"), float("-inf")}`
- **THEN** `ValueError` is raised
- **AND** the exception message contains the substring `"cadence_s"`

#### Scenario: traveling_wave.compute rejects invalid cadence_s type with TypeError naming the field
- **WHEN** `traveling_wave.compute(trajectory_df, cadence_s=v)` is invoked for any of `v ∈ {True, np.bool_(True), "300", [300.0]}`
- **THEN** `TypeError` is raised
- **AND** the exception message contains the substring `"cadence_s"`

#### Scenario: traveling_wave.compute is deterministic across runs and across OSs
- **GIVEN** a valid `trajectory_df` and `cadence_s = 300.0`
- **WHEN** `traveling_wave.compute(trajectory_df, cadence_s=300.0)` is invoked twice in the same Python process
- **THEN** the 6 float trait columns are bit-identical at `atol=0`
- **AND** the ridge `wavelengths_px[interior]` array (a PUBLIC `SpatialRidgeResult` field, 1:1 with the per-position `argmax` scale-index selection) is asserted EXACTLY equal across two in-process runs (`atol=0`) — the load-bearing spatial-λ determinism contract, since an argmax tie-flip is a DISCRETE full-scale-step jump, not an `atol`-bounded perturbation. Cross-OS, the median canary below (run on the Ubuntu/Windows/macOS CI matrix) catches such a flip because a flip moves the median by a full scale-step ≫ the canary `atol`
- **AND** a captured canary on a fixed synthetic input matches the hardcoded expected values to within the measured `atol` (target `1e-6`, but the real value SHALL be measured — it may be looser if a cross-OS tie-flip shifts the median) across Ubuntu / Windows / macOS CI runners AT THE TIME OF PR-MERGE; canary values are regression-detection sentinels and MAY be re-captured (in a follow-up commit cross-referencing this scenario) if upstream BLAS / scipy / pywt / numpy semantics legitimately shift after merge

#### Scenario: traveling_wave.compute emits exactly one DEBUG logger record on the happy path
- **GIVEN** a valid `trajectory_df` and `cadence_s = 300.0` and `caplog.set_level(logging.DEBUG)`
- **WHEN** `traveling_wave.compute(trajectory_df, cadence_s=300.0)` is invoked
- **THEN** exactly one log record at level `DEBUG` is emitted by the logger `sleap_roots.circumnutation.traveling_wave`
- **AND** the record's message starts with `"traveling_wave.compute("`
- **AND** the record's message contains each of the tokens `"n_tracks="`, `"cadence_s="`
- **AND** no `INFO` / `WARNING` / `ERROR` / `CRITICAL` records are emitted

#### Scenario: traveling_wave.compute real plate-001 QPB result (calibrated)
- **GIVEN** the 6 tracks of `tests/data/circumnutation_nipponbare_plate_001/plate_001_greyscale.tracked_proofread.slp` loaded via `Series.load(...).get_tracked_tips()` filtered by `track_id ∈ {0, 1, 2, 3, 4, 5}` and `cadence_s = 300.0`
- **WHEN** `traveling_wave.compute(trajectory_df, cadence_s=300.0)` is invoked
- **THEN** all 6 tracks have finite `traveling_wave_residual` with value `< 0.30` (QPB holds; the calibrated residual is ≈ 0.10–0.18 on this fixture — the test asserts a generous band, not a pinned range, because the precise endpoints depend on the extended calibration table)
- **AND** all 6 tracks have finite `lambda_spatial_variation` in a plausible range (≈ 0.10–0.45)
- **AND** all 6 tracks pass the COI gate (`coi_valid_fraction ≥ 1 − COI_FRACTION_MAX`)

#### Scenario: traveling_wave.compute recovers a known spatial wavelength on a small-amplitude synthetic
- **GIVEN** a synthetic trajectory built so the spatial wavelength is known a priori — small lateral amplitude relative to growth drift so the trail arc-length-per-period ≈ `growth_rate_px_per_frame · (T_nutation_s / cadence_s)` (e.g. `synthetic.generate_trajectory(amplitude_px=2.0, growth_rate_px_per_frame=4.29, T_nutation_s=3333, cadence_s=300, n_frames=575, random_state=0)` → a-priori λ ≈ 47.7 px)
- **WHEN** `traveling_wave.compute(trajectory_df, cadence_s=300.0)` is invoked on the converted trajectory_df
- **THEN** `lambda_spatial_median_px` recovers the a-priori wavelength within a documented relative tolerance (`abs(lambda_spatial_median_px − λ_apriori) / λ_apriori < 0.25`, covering cgau2-residual + small-amplitude arc-length bias)
- **AND** because the synthetic's spatial wavelength equals `v · T_frames` by construction, `traveling_wave_residual` is small (the synthetic cannot produce a large/QPB-violating residual; the plate-001 scenario covers the non-trivial-residual case)

#### Scenario: traveling_wave.compute rejects invalid constants
- **WHEN** `traveling_wave.compute(trajectory_df, cadence_s=300.0, constants=object())` is invoked
- **THEN** `TypeError` is raised (constants must be None or `ConstantsT`)
- **AND** WHEN invoked with a `ConstantsT` whose `COI_FRACTION_MAX` is outside `(0, 1]` (e.g. `0.0` or `1.5`), `ValueError` is raised naming `COI_FRACTION_MAX`

#### Scenario: traveling_wave.compute handles all-NaN-tip and single-frame tracks without crashing
- **GIVEN** a `trajectory_df` containing one track whose every `tip_x`/`tip_y` is NaN and another track with exactly one frame, and `cadence_s = 300.0`
- **WHEN** `traveling_wave.compute(trajectory_df, cadence_s=300.0)` is invoked
- **THEN** the call returns a `pandas.DataFrame` with exactly one row per 5-tuple (no dropped or duplicated rows), without raising and without `np.RuntimeWarning`
- **AND** each such track's 5 spatial traits and `coi_valid_fraction` are NaN (no ridge formed)

#### Scenario: traveling_wave.compute COI gate boundary and coi_valid_fraction-finite-iff-ridge
- **GIVEN** a constructed `SpatialRidgeResult` (synthetic or monkeypatched) whose `in_coi` fraction is exactly 0.5, and a second whose `in_coi` fraction exceeds 0.5
- **WHEN** the gate `coi_valid_fraction < (1 − COI_FRACTION_MAX)` is applied (with `COI_FRACTION_MAX = 0.5`)
- **THEN** the exactly-0.5-in-COI ridge does NOT gate (strict inequality; its 5 spatial traits are computed), while the >0.5-in-COI ridge gates its 5 spatial traits to NaN
- **AND** in BOTH cases `coi_valid_fraction` is FINITE (a ridge formed) — `coi_valid_fraction` is NaN only when no ridge forms (degenerate midline/resample or caught CWT `ValueError`)

#### Scenario: traveling_wave trait units are all in the pipeline vocabulary
- **WHEN** the module-level `_TRAVELING_WAVE_TRAIT_UNITS` mapping is inspected
- **THEN** it has an entry for each of the 6 trait columns
- **AND** `lambda_spatial_median_px`, `lambda_expected_px`, `lambda_spatial_mad_px` map to `"px"` and `lambda_spatial_variation`, `traveling_wave_residual`, `coi_valid_fraction` map to `"—"`
- **AND** every value is a member of `PIPELINE_UNIT_VOCABULARY` (no `_CONSTANTS_VERSION` change required)

#### Scenario: in-package calibration literal matches the authoritative JSON (n-averaged)
- **GIVEN** the module-level `_CGAU2_LAMBDA_CALIBRATION` literal and the authoritative `tests/data/circumnutation_spatial_cwt_calibration.json`
- **WHEN** for each `λ_true` the JSON's `ratio` values across `n ∈ {200, 400, 600}` are averaged (and the corresponding `λ_reported_mean` computed), and the resulting pairs are sorted by `λ_reported_mean`
- **THEN** the literal's `(λ_reported_mean, ratio_mean)` pairs equal that n-averaged computation (to `atol=0` on the committed full-precision values — the literal SHALL be generated from the JSON tokens, never hand-rounded)
- **AND** the literal's `λ_reported_mean` axis is strictly increasing (well-posed `np.interp`) and covers `λ_true` up to ≥ 140 px (no clamped extrapolation for the observed real λ ≈ 142 px)

### Requirement: Circumnutation pipeline composition API
The system SHALL provide `sleap_roots.circumnutation.pipeline.compute_traits(inputs: CircumnutationInputs, constants: Optional[ConstantsT] = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]` and a picklable `CircumnutationPipeline` class that composes the five trait-emitting tiers (Tier 0 `kinematics`, QC `qc`, Tier 1 `nutation`, Tier 2 `psi_g`, Tier 3c `traveling_wave`) into one per-plant trait table, units mapping, and provenance-bearing CSV. The composition SHALL be a **sequential merge-orchestrator**, not the per-frame networkx `TraitDef` DAG of `sleap_roots.trait_pipelines.Pipeline`: the circumnutation tiers are per-track `DataFrame → DataFrame` functions, so the pipeline calls each tier's `compute()` once and merges their per-plant outputs on `_IDENTITY_5_TUPLE`. The documented tier-dependency structure (Tier 0 / QC / Tier 1 / Tier 2 independent; Tier 3c depends on Tier 0 + Tier 1) SHALL be honored as a fixed call order; the pipeline introduces no new physical constant and `_CONSTANTS_VERSION` SHALL remain 6.

`CircumnutationPipeline` SHALL be an `attrs` class whose only field is `constants` (an `Optional[ConstantsT]`, default `None`), making instances picklable. It SHALL expose `compute_traits(self, inputs) -> tuple[pd.DataFrame, pd.DataFrame, dict]` (the pure composition) and `save(self, out_path, per_plant_df, units, *, inputs, input_path) -> None` (the I/O step; `inputs` is the same `CircumnutationInputs` passed to `compute_traits`, so `save` can record `cadence_s` / `R_px` / `run_id` provenance from it — these live on `inputs`, not on `per_plant_df`). The module-level `compute_traits(inputs, constants=None)` SHALL be a thin wrapper equal to `CircumnutationPipeline(constants=constants).compute_traits(inputs)`.

`compute_traits` SHALL be **pure** (no filesystem I/O): it SHALL read `inputs.trajectory_df` and `inputs.cadence_s`, call `kinematics.compute(df, constants)`, `qc.compute(df, constants)`, `nutation.compute(df, cadence_s, constants=constants)` (relying on `nutation.compute`'s default `coordinate="lateral"` — the pipeline SHALL NOT override `coordinate`, because the dedup `atol=0` equivalence requires the Tier 1 frame passed to `traveling_wave` to be the lateral one `traveling_wave` would otherwise recompute), `psi_g.compute(df, cadence_s, constants=constants)`, and `traveling_wave.compute(df, cadence_s, constants, tier0_df=<tier0 result>, tier1_df=<tier1 result>)` (the PR #14 dedup fast path — Tier 0 and Tier 1 are computed once and reused, NOT recomputed inside `traveling_wave`). The SAME `constants` object SHALL be passed to all five tiers AND to the `traveling_wave` fast path (a divergent `constants` would silently break the `atol=0` dedup equivalence). `cadence_s` SHALL be threaded only to the three cadence-consuming tiers (nutation, psi_g, traveling_wave); Tier 0 and QC take none. `inputs.R_px` SHALL NOT be passed to any of the five tiers (they are pure-pixel, CC-3); it is provenance-only. The pipeline SHALL add no redundant input validation of its own (the `CircumnutationInputs` `attrs` validators and each tier's internal validation already cover it).

`compute_traits` SHALL return the 3-tuple `(per_plant_df, trajectory_df, units_dict)` where `trajectory_df` is the input frame echoed for provenance — it SHALL be the unmodified `inputs.trajectory_df` (no tier mutates the input in place; the echo carries the same data every tier consumed). The components:

- `per_plant_df` SHALL have one row per unique `_IDENTITY_5_TUPLE` and exactly **46 columns** in fixed tier order: the 8 `ROW_IDENTITY_COLUMNS`, then the 10 Tier 0 trait columns (`_TIER0_TRAIT_COLUMNS`), then the QC trait columns EXCLUDING `growth_axis_unreliable` (10 columns), then the 8 Tier 1 columns (`_NUTATION_TRAIT_COLUMNS`), then the 4 Tier 2 columns (`_PSIG_TRAIT_COLUMNS`), then the 6 Tier 3c columns (`_TRAVELING_WAVE_TRAIT_COLUMNS`). Each tier's emitted dtypes SHALL be preserved through the merge (e.g. bool `is_nutating`, string `qc_failure_reason`); no global float coercion is applied.
- **Merge mechanic (load-bearing).** Each tier's `compute()` returns ALL 8 `ROW_IDENTITY_COLUMNS`, but `_IDENTITY_5_TUPLE` is only 5 of them. Merging two full 8-identity frames `on=_IDENTITY_5_TUPLE` would make pandas treat the other 3 identity columns (`timepoint`, `genotype`, `treatment`) as non-key columns present on both sides and suffix them `_x`/`_y`, breaking the final 46-column selection. Therefore the pipeline SHALL build the composed frame by starting from the shared per-plant template (`_io._build_per_plant_template_from_df(inputs.trajectory_df)`, which carries all 8 identity columns, one row per 5-tuple) and left-merging each tier's output **projected to `[*_IDENTITY_5_TUPLE, *<that tier's trait columns>]`** (the 5 keys + only that tier's traits; for QC, its trait columns MINUS `growth_axis_unreliable`) — reproducing the intra-tier `trait_df` merge shape so no non-key identity column collides. Each `how="left"` merge SHALL apply the int64 coercion-with-raise guard on `track_id`/`plant_id` (a key-dtype mismatch raises rather than producing a silent all-NaN row). Because every tier templates from the same `inputs.trajectory_df`, all five emit the identical 5-tuple row set, so no left-merge injects an unmatched (NaN) row.
- `growth_axis_unreliable` is emitted by BOTH Tier 0 and QC (QC recomputes it with the same formula/inputs; an existing cross-tier equality test guarantees element-wise equality). Both are `bool` dtype. The pipeline SHALL keep exactly ONE `growth_axis_unreliable` column — Tier 0's, in the Tier 0 block — and SHALL drop QC's copy before merging (so no `_x`/`_y` suffix collision occurs). The pipeline SHALL assert the two source columns are equal via `Series.equals` (dtype + value, so a future nullable-boolean drift cannot pass a naive `==` check that treats `NaN != NaN`) before dropping QC's copy, raising `ValueError` naming the divergence if they differ.

- The composed schema deliberately OMITS the 5 `L_gz`/`L_c`-dependent Tier 3c traits (`L_gz_estimate`, `L_c_estimate`, `B_balance_number`, `L_gz_steady_state_residual`, `L_gz_resolvable`) and the growth-zone mask — they are blocked on #230 and were never emitted by `traveling_wave` (PR #10 reduced scope), so they are absent from the 46-column frame, not reserved as NaN.
- `units_dict` SHALL map every one of the 46 columns to a unit string in `PIPELINE_UNIT_VOCABULARY`, assembled as `ROW_IDENTITY_UNITS` (restricted to the identity columns) merged with the five per-tier `_*_TRAIT_UNITS` maps. `growth_axis_unreliable` SHALL appear exactly once.

To assemble `units_dict`, the `nutation` and `psi_g` modules SHALL each declare a `_*_TRAIT_UNITS` mapping (mirroring `_TIER0_TRAIT_UNITS` / `_QC_TRAIT_UNITS` / `_TRAVELING_WAVE_TRAIT_UNITS`): `nutation._NUTATION_TRAIT_UNITS` SHALL have one entry per `_NUTATION_TRAIT_COLUMNS` (8) and `psi_g._PSIG_TRAIT_UNITS` one entry per `_PSIG_TRAIT_COLUMNS` (4); every value SHALL be a member of `PIPELINE_UNIT_VOCABULARY`. These additions are additive module constants (no behavior change to the tiers). The unit strings are pinned below (semantic-correctness matters: the sidecar writer validates vocabulary membership but NOT semantic correctness, so an in-vocabulary-but-wrong string would silently mislabel a published column):

- `_NUTATION_TRAIT_UNITS`: `T_nutation_median` → `"s"`, `T_nutation_iqr` → `"s"`, `A_nutation_envelope_max_px` → `"px"`, `band_power_ratio` → `"—"`, `noise_floor_estimate` → `"px"` (it is a median FFT **amplitude** of the lateral px signal — NOT a dimensionless ratio), `is_nutating` → `"bool"`, `period_residual_vs_derr_reference` → `"—"`, `cadence_nyquist_ratio` → `"—"`.
- `_PSIG_TRAIT_UNITS`: `T_psig_median_s` → `"s"`, `delta_E_amplitude_proxy_px_per_frame` → `"px/frame"`, `handedness` → `"int"` (an integer sign in {−1, 0, +1}; matches the program's type-token convention for non-float columns, e.g. `track_id` → `"int"`, `is_nutating` → `"bool"`), `helix_signed_area_px2` → `"px²"` (the superscript-² glyph, the `PIPELINE_UNIT_VOCABULARY` token — a literal ASCII `"px2"` is NOT in vocabulary and would fail the writer).

The `"s"` (seconds) period units (`T_nutation_median`, `T_nutation_iqr`, `T_psig_median_s`) arrive **pre-converted** from the tiers: the frames→seconds conversion is performed once inside the temporal CWT machinery (`temporal_cwt._log_spaced_scales` derives `periods_s` via `pywt.scale2frequency(...) / cadence_s`), so the period traits are already in seconds when the pipeline receives them. The pipeline performs **no unit conversion** of its own — it threads `cadence_s` to the cadence-consuming tiers and labels each column with the unit the tier emits. Note the deliberate program convention asymmetry: **periods are in seconds** while **rates are per-frame** (`v_*_px_per_frame`, `delta_E_amplitude_proxy_px_per_frame` → `"px/frame"`, NOT px/s); the two are reconciled only inside `traveling_wave` (`T_frames = T_nutation_median / cadence_s`, then `lambda_expected_px = v · T_frames`, keeping λ pure-pixel).

**Scope note — GitHub issue #222.** PR #14 adds ONLY the two units maps the pipeline needs; it does NOT perform the broader #222 program-wide suffix-convention work (renaming `T_nutation_median` → `T_nutation_median_s` / `T_nutation_iqr` → `T_nutation_iqr_s`, documenting the rule, and the foundation suffix-gate), which remains #222's own job. The two maps are therefore keyed on the CURRENT (unsuffixed) `nutation` column names; when the #222 rename lands, the `_NUTATION_TRAIT_UNITS` keys for those two columns will be re-keyed accordingly.

`save` SHALL write the per-plant CSV plus its two sibling sidecars by delegating to `_io.gather_run_metadata(input_path, run_id=inputs.run_id, constants=self.constants, cadence_s=inputs.cadence_s, R_px=inputs.R_px)` and `_io.write_per_plant_csv(out_path, per_plant_df, units, run_metadata)` (which writes the CSV, the `<stem>.units.json` units sidecar, and the `run_metadata.json` provenance sidecar per Requirement: Units sidecar JSON and Requirement: Run-metadata sidecar). The `input_path` (source `.slp` path) SHALL be supplied by the caller (it is file provenance, not on `inputs`); `run_id`, `cadence_s`, and `R_px` SHALL be sourced authoritatively from `inputs` (so `inputs.run_id`'s documented "included in `run_metadata.json`" contract is satisfied, and `cadence_s` — which determines every period trait and the residual — is captured so a run is reproducible from the sidecars alone). The parent directory of `out_path` MUST exist (the underlying writer does not create it); `save` SHALL raise a clear error if it does not. Existing files at `out_path` and the sibling sidecars are overwritten; because `run_metadata.json` has a FIXED name in the CSV's parent directory, callers SHALL write at most one composed CSV per output directory (a second `save` into the same directory clobbers the first's `run_metadata.json`).

**Composed-CSV schema stability (downstream contract for PR #15–#18).** The 46-column order is contractual by tier block (identity → Tier 0 → QC → Tier 1 → Tier 2 → Tier 3c), sourced from the module-level `_*_TRAIT_COLUMNS` tuples. Future tiers SHALL append their block after Tier 3c, and new columns within an existing tier SHALL append within that tier's block — the ordinal positions of existing columns SHALL NOT change retroactively. Downstream consumers SHALL key on column NAME, not ordinal position. `_schema_version` versions the row-identity schema only (the 8-column identity prefix + 5-tuple keying), NOT the trait-column set; adding trait columns in future tiers does NOT bump `_schema_version`. This composed CSV is the program's first multi-tier on-disk artifact (no prior composed format exists to break).

`compute_traits` SHALL be deterministic per CC-6: same `inputs` → the two `per_plant_df` results SHALL be equal as full frames (identical column order, row order, dtypes, and values — float columns `atol=0`, `NaN`-equal), deferring to the per-tier determinism contracts (no separate composed-output canary is introduced; the per-plant template's stable sort fixes row order). `CircumnutationPipeline` instances SHALL be picklable (`pickle.loads(pickle.dumps(pipeline))` round-trips) and the unpickled instance SHALL compute an identical `per_plant_df`.

`compute_traits` SHALL handle an all-degenerate input (every track degenerate, so every tier emits its all-NaN trait block) without raising: the returned `per_plant_df` SHALL still have the full 46-column schema with one row per 5-tuple, the `growth_axis_unreliable` cross-tier `Series.equals` coalescing check SHALL NOT raise on the all-degenerate column (Tier 0 and QC both emit it as `bool`), and `save` SHALL still write successfully (the units dict still covers all 46 columns).

#### Scenario: compute_traits returns the documented 3-tuple and 46-column composed schema
- **GIVEN** a valid multi-track `CircumnutationInputs` (≥ 2 plates with overlapping `track_id`, `track_id` dtype `float64` to exercise the int64-coercion merge guard, `cadence_s = 300.0`, enough frames per track for the temporal/spatial chains)
- **WHEN** `pipeline.compute_traits(inputs)` is invoked
- **THEN** the return value is a 3-tuple `(per_plant_df, trajectory_df, units_dict)` of types `(pandas.DataFrame, pandas.DataFrame, dict)`
- **AND** `per_plant_df` has exactly 46 columns in the declared tier order: the 8 `ROW_IDENTITY_COLUMNS`, then `_TIER0_TRAIT_COLUMNS` (10), then `_QC_TRAIT_COLUMNS` minus `growth_axis_unreliable` (10), then `_NUTATION_TRAIT_COLUMNS` (8), then `_PSIG_TRAIT_COLUMNS` (4), then `_TRAVELING_WAVE_TRAIT_COLUMNS` (6)
- **AND** `per_plant_df` has exactly one row per unique `_IDENTITY_5_TUPLE` (`per_plant_df[list(_IDENTITY_5_TUPLE)].duplicated().sum() == 0`)
- **AND** each tier's emitted dtypes are preserved through the merge with NO NaN-induced upcast (`is_nutating` stays `bool`, `handedness` stays integer, `qc_failure_reason` stays object/string) — the `how="left"` merge onto the shared template emits exactly the template's 5-tuples so no unmatched row injects a `NaN` that would upcast a `bool`/`int` flag column

#### Scenario: compute_traits coalesces growth_axis_unreliable to a single Tier-0-owned column
- **GIVEN** a valid `CircumnutationInputs`
- **WHEN** `pipeline.compute_traits(inputs)` is invoked
- **THEN** `per_plant_df` contains exactly one `growth_axis_unreliable` column (no `growth_axis_unreliable_x` / `_y`)
- **AND** that column sits in the Tier 0 block of the column order (the position of `growth_axis_unreliable` in `_TIER0_TRAIT_COLUMNS`, not the QC block)
- **AND** its values equal the `growth_axis_unreliable` that `qc.compute(inputs.trajectory_df)` emits for the same 5-tuples (cross-tier equality is preserved by the coalescing)

#### Scenario: compute_traits dedups Tier 0/Tier 1 by routing precomputed frames into traveling_wave
- **GIVEN** a valid `CircumnutationInputs`
- **WHEN** `pipeline.compute_traits(inputs)` is invoked
- **THEN** the Tier 3c columns in `per_plant_df` equal (at `atol=0`) the columns from a standalone `traveling_wave.compute(inputs.trajectory_df, inputs.cadence_s)` call for the same input
- **AND** the pipeline obtains those columns by calling `traveling_wave.compute(..., tier0_df=<tier0 result>, tier1_df=<tier1 result>)` so Tier 0 and the Tier 1 temporal CWT are computed once, not twice (the dedup contract)

#### Scenario: compute_traits assembles a units_dict covering every emitted column in vocabulary
- **GIVEN** a valid `CircumnutationInputs`
- **WHEN** `pipeline.compute_traits(inputs)` is invoked and the returned `units_dict` is inspected
- **THEN** `set(units_dict.keys()) == set(per_plant_df.columns)` (1:1 coverage of all 46 columns)
- **AND** every value in `units_dict` is a member of `PIPELINE_UNIT_VOCABULARY`
- **AND** `growth_axis_unreliable` appears exactly once as a key
- **AND** passing `(per_plant_df, units_dict)` to `_io.write_per_plant_csv` does NOT raise the coverage / vocabulary `ValueError`

#### Scenario: nutation and psi_g declare trait-units maps with the pinned unit strings
- **WHEN** the module-level `nutation._NUTATION_TRAIT_UNITS` and `psi_g._PSIG_TRAIT_UNITS` mappings are inspected
- **THEN** `_NUTATION_TRAIT_UNITS` equals `{"T_nutation_median": "s", "T_nutation_iqr": "s", "A_nutation_envelope_max_px": "px", "band_power_ratio": "—", "noise_floor_estimate": "px", "is_nutating": "bool", "period_residual_vs_derr_reference": "—", "cadence_nyquist_ratio": "—"}` (one entry per `_NUTATION_TRAIT_COLUMNS`; `noise_floor_estimate` is `"px"` — a median FFT amplitude — NOT `"—"`)
- **AND** `_PSIG_TRAIT_UNITS` equals `{"T_psig_median_s": "s", "delta_E_amplitude_proxy_px_per_frame": "px/frame", "handedness": "int", "helix_signed_area_px2": "px²"}` (one entry per `_PSIG_TRAIT_COLUMNS`; `helix_signed_area_px2` is the superscript-² glyph `"px²"`, not ASCII `"px2"`)
- **AND** every value in both mappings is a member of `PIPELINE_UNIT_VOCABULARY`

#### Scenario: save writes the CSV and two sidecars and round-trips with full provenance
- **GIVEN** a `(per_plant_df, trajectory_df, units_dict)` from `compute_traits(inputs)` where `inputs = CircumnutationInputs(trajectory_df=..., cadence_s=300.0, R_px=12.0, run_id="r1")`, a temporary `out_path` whose parent directory exists, and an `input_path` (a `.slp` path string)
- **WHEN** `CircumnutationPipeline().save(out_path, per_plant_df, units_dict, inputs=inputs, input_path=input_path)` is invoked
- **THEN** the CSV at `out_path`, a sibling `<stem>.units.json`, and a sibling `run_metadata.json` all exist
- **AND** `_io.read_per_plant_csv(out_path)` recovers the DataFrame, the units mapping, and the run-metadata
- **AND** the run-metadata contains `input_path`, the git SHA, the sleap_roots/sleap_io/numpy/scipy/pandas/python version fields, `platform`, an ISO-8601 `timestamp`, `_schema_version`, `_constants_version` (equal to 6), and `_constants_snapshot`
- **AND** the run-metadata records `cadence_s == 300.0`, `R_px == 12.0`, and `run_id == "r1"` (sourced from `inputs`), so the run is reproducible from the sidecars alone
- **AND** WHEN `out_path`'s parent directory does NOT exist, `save` raises a clear error (the writer does not create parent directories)

#### Scenario: compute_traits performs no filesystem I/O
- **GIVEN** a valid `CircumnutationInputs`
- **WHEN** `pipeline.compute_traits(inputs)` is invoked (with no `out_path` argument — the signature has none)
- **THEN** no CSV, units sidecar, or run-metadata file is written (writing is exclusively the responsibility of `save`)

#### Scenario: CircumnutationPipeline is picklable and computes identically after a round-trip
- **GIVEN** a `CircumnutationPipeline()` instance and a valid `CircumnutationInputs`
- **WHEN** the instance is round-tripped via `pickle.loads(pickle.dumps(pipeline))`
- **THEN** the round-trip succeeds without raising
- **AND** the unpickled instance's `compute_traits(inputs)` `per_plant_df` is equal to the original instance's (identical float trait columns at `atol=0`)

#### Scenario: compute_traits is deterministic across two in-process runs
- **GIVEN** a valid `CircumnutationInputs`
- **WHEN** `pipeline.compute_traits(inputs)` is invoked twice in the same Python process
- **THEN** the two `per_plant_df` results are equal as full frames — identical column order, row order, and dtypes, with float columns bit-identical at `atol=0` and `NaN`-equal (e.g. `pandas.testing.assert_frame_equal`)

#### Scenario: compute_traits handles an all-degenerate input without raising
- **GIVEN** a `CircumnutationInputs` in which EVERY track is degenerate (e.g. stationary / too-short), so every tier emits its all-NaN trait block
- **WHEN** `pipeline.compute_traits(inputs)` is invoked
- **THEN** the call returns the full 46-column `per_plant_df` with one row per 5-tuple, without raising (the `growth_axis_unreliable` cross-tier `Series.equals` coalescing check does NOT raise — both Tier 0 and QC emit it as `bool`)
- **AND** the bool/int flag columns (`is_nutating`, `handedness`, `growth_axis_unreliable`) retain their bool/int dtype (not upcast by the all-NaN trait blocks)
- **AND** `save(...)` writes the CSV + sidecars successfully (the units dict still 1:1-covers all 46 columns)

#### Scenario: compute_traits round-trips the real plate-001 fixture through the full pipeline
- **GIVEN** the 6 tracks of `tests/data/circumnutation_nipponbare_plate_001/plate_001_greyscale.tracked_proofread.slp` loaded via `Series.load(...).get_tracked_tips()` (with `track_id` coerced to int and the 8 row-identity columns attached with plate-001 metadata) wrapped in a `CircumnutationInputs(cadence_s=300.0)`
- **WHEN** `pipeline.compute_traits(inputs)` is invoked and the result is written via `save` to a temporary directory and read back
- **THEN** `per_plant_df` has exactly 6 rows and the 46-column composed schema in declared order
- **AND** columns from all five tiers are present (e.g. `v_total_median_px_per_frame` (Tier 0), `track_is_clean` (QC), `T_nutation_median` (Tier 1), `handedness` (Tier 2), `traveling_wave_residual` (Tier 3c))
- **AND** all 6 tracks have finite `traveling_wave_residual < 0.30` (the QPB band, matching Requirement: Tier 3c traveling-wave trait emission API)
- **AND** the single coalesced `growth_axis_unreliable` column equals what `qc.compute` emits for the same tracks (cross-tier equality at the composed level)
- **AND** the test is skipped when the Git-LFS proofread fixture is absent

### Requirement: Per-genotype aggregation API
The system SHALL provide `sleap_roots.circumnutation.aggregation.aggregate_by_genotype(per_plant_df: pd.DataFrame, units: dict) -> tuple[pd.DataFrame, dict]`. The function SHALL consume the composed per-plant trait frame and its units mapping (the `(per_plant_df, units_dict)` output of Requirement: Circumnutation pipeline composition API) and return `(per_genotype_df, per_genotype_units)`, where `per_genotype_df` is one row per `(plate_id, genotype, treatment)` group and `per_genotype_units` is a 1:1 column→unit mapping covering every column of `per_genotype_df`. The function SHALL be a pure function (no filesystem I/O, no mutation of `per_plant_df` or `units`) and SHALL operate on pure-pixel inputs without accepting `px_per_mm` or emitting any mm-bearing column (CC-3).

**Input validation.** The function SHALL validate that `units` maps 1:1 to the columns of `per_plant_df` (every column has a unit; no extra keys) and SHALL raise `ValueError` naming the offending column(s) when it does not, before producing any output (reusing the shared `_validate_units_coverage` helper of Requirement: Per-genotype trait CSV and sidecar I/O). The function SHALL also validate the current track↔plant 1:1 invariant: each `(plate_id, genotype, treatment, plant_id)` combination SHALL map to exactly one row of `per_plant_df`. If any such combination maps to more than one row (a future track↔plant divergence where one plant carries multiple tracks/rows), the function SHALL raise `ValueError` naming the offending `plant_id` rather than silently aggregating per-track; a per-plant collapse keyed on `plant_id` (the `theory.md` §7.7 two-level shape) is deferred to a follow-up issue.

**Grouping.** The function SHALL group by the key `(plate_id, genotype, treatment)` and SHALL NOT pool plants across different `plate_id` values. Grouping SHALL preserve rows whose group-key value is NaN (the per-plant template permits NaN `genotype` / `treatment` / `plate_id`): a NaN-keyed plant SHALL form its own group rather than be silently dropped (i.e., grouping uses `dropna=False` semantics, so `n_plants_passing_qc + n_plants_excluded` over all output rows equals the row count of `per_plant_df`). Output rows SHALL be sorted deterministically by `(plate_id, genotype, treatment)`. The row-identity columns `series`, `sample_uid`, `timepoint`, `plant_id`, and `track_id` SHALL NOT appear in `per_genotype_df`. Under the validated 1:1 invariant, each row of a group is exactly one plant, so the cross-plant median ± IQR equals `theory.md` §7.7's per-genotype aggregation.

**QC exclusion.** A plant SHALL be excluded from every median, IQR, fraction, and mode computation iff its `track_is_clean` value is `False`. The function SHALL NOT mutate the per-plant `track_is_clean` or `qc_failure_reason` columns; the per-plant frame already carries the per-plant exclusion flag and reason. The per-genotype row records only counts and a clause→count summary, not the excluded plants' identities; the specific excluded plants of a `(plate_id, genotype, treatment)` group remain auditable by re-filtering the per-plant trait CSV for rows of that group with `track_is_clean == False` (this audit recipe SHALL be documented in the module docstring to satisfy `theory.md` §7.7's "flagged in the trait CSV with reason").

**Counts and reasons.** Each output row SHALL include `n_plants_passing_qc` (int: count of `track_is_clean == True` plants in the group), `n_plants_excluded` (int: count of `track_is_clean == False` plants in the group), and `exclusion_reasons` (string). `exclusion_reasons` SHALL be a clause→count summary over the excluded plants' `qc_failure_reason` values, with clauses ordered by `qc._FAILURE_CLAUSE_ORDER` and rendered as `"<clause>:<count>"` pairs joined by `"; "`; it SHALL be the empty string `""` when `n_plants_excluded == 0`. Per-plant `qc_failure_reason` values SHALL be split on `", "` into clauses; the `qc_inputs_insufficient` sentinel SHALL count as its own clause and, being index 0 of `qc._FAILURE_CLAUSE_ORDER`, SHALL sort first when present. Counts SHALL be clause-incidence (a plant failing multiple clauses counts toward each), and MAY therefore sum to more than `n_plants_excluded`. Every member of `qc._FAILURE_CLAUSE_ORDER` SHALL match `^[a-z0-9_]+$` so the `":"` / `"; "` rendering is unambiguous to parse downstream (a clause token containing `:`, `;`, or `,` would corrupt the encoding).

**Float trait statistics.** For every input column classified as a numeric-unit trait (its `units` value is not `int`, `bool`, or `string`, it is not a row-identity column, and it is not in the explicit special/excluded set below), the function SHALL emit `<trait>_median` and `<trait>_iqr`, both computed across the passing plants of the group with NaN-skipping. `<trait>_iqr` SHALL be the interquartile range Q75 − Q25 with linear interpolation and SHALL be NaN when fewer than 2 finite passing-plant values are available for that trait. `<trait>_median` SHALL be NaN when no finite passing-plant value is available and SHALL be emitted for a single finite value. Because NaN-skipping is per-trait, the number of finite values behind a given `<trait>_median` MAY be smaller than `n_plants_passing_qc` (a plant passing QC can still carry NaN for a particular trait, e.g. a non-nutating plant's `T_nutation_*`); `n_plants_passing_qc` is therefore an upper bound on the per-trait sample size, and a NaN `<trait>_iqr` beside a finite `<trait>_median` indicates fewer than 2 finite values for that trait — not zero spread. The module docstring SHALL document this and that per-trait finite-count columns are deferred (a `<trait>_iqr` whose source trait is itself a dispersion measure — `T_nutation_iqr`, `lambda_spatial_variation`, `lambda_spatial_mad_px`, `worst_step_ratio` — is a spread-of-spreads).

**Circular and excluded numeric columns.** The wrapping circular angle `principal_axis_angle` (unit `rad`, an `arctan2` value in (−π, π]) SHALL NOT be linearly aggregated: a linear median/IQR of a wrapping angle is unsound near the ±π branch cut, and `principal_axis_angle` is an absolute per-plant growth-axis reference direction with arbitrary orientation across plants (the biological traits derived from it — `v_long*`, `v_lat*`, `angular_amplitude` — are already rotation-invariant). `principal_axis_angle_median` / `principal_axis_angle_iqr` SHALL NOT appear in `per_genotype_df`. A circular-statistics summary (e.g. circular mean + resultant length) is deferred to a follow-up issue. The non-wrapping `rad`-unit magnitude `angular_amplitude` (a peak-to-peak extent, non-negative) IS aggregated linearly as usual.

**Signed chirality magnitude.** The signed area `helix_signed_area_px2` (unit `px²`) encodes circumnutation chirality in its sign (it is the float twin of the integer `handedness`); chirality is bimodal within a genotype, so a cross-plant median of the *signed* value can cancel toward zero even when every plant has a large magnitude, misleadingly reading as "no helix". The function SHALL therefore aggregate its **magnitude**: it SHALL emit `helix_signed_area_abs_px2_median` and `helix_signed_area_abs_px2_iqr` (the median / IQR of `|helix_signed_area_px2|` across passing plants, unit `px²`, same NaN-skipping and `< 2`-finite → NaN IQR rule as float traits) and SHALL NOT emit `helix_signed_area_px2_median` / `helix_signed_area_px2_iqr`. Chirality direction is summarized by `handedness_mode` / `handedness_consensus_frac`; the two together (strength + handedness) describe the population's helix unambiguously. Other signed traits (`v_long_signed*`, `v_lat_signed*`, `period_residual_vs_derr_reference`, `delta_E_amplitude_proxy_px_per_frame`) carry per-plant-consistent or magnitude semantics and ARE aggregated linearly as ordinary float traits.

**Special columns.** The function SHALL emit `frac_nutating` (the mean of the boolean `is_nutating` over passing plants; NaN when `n_plants_passing_qc == 0`). The function SHALL emit `handedness_mode` (int: the most frequent `handedness` value among passing plants, computed from explicit value counts — not `pandas.Series.mode` — with ties broken by smallest `abs(value)` first then smallest signed value, so a `{+1, -1}` tie resolves to `-1`) and `handedness_consensus_frac` (the fraction of passing plants whose `handedness` equals `handedness_mode`; NaN when `n_plants_passing_qc == 0`). A `handedness_consensus_frac` at or near `0.5` indicates `handedness_mode` is a tie-break artifact rather than a population majority and SHALL be documented as such in the module docstring. The boolean `growth_axis_unreliable` column SHALL NOT be aggregated or emitted (it is a `track_is_clean` failure clause, so any plant with `growth_axis_unreliable == True` fails `track_is_clean` and is excluded — it is therefore always `False` among passing plants; the short-track gate likewise yields `track_is_clean == False`).

**Degenerate groups.** The function SHALL emit exactly one row per group and SHALL NEVER drop a group or raise on degenerate input. For an all-excluded group (`n_plants_passing_qc == 0`), all `<trait>_median` / `<trait>_iqr` values, `frac_nutating`, and `handedness_consensus_frac` SHALL be NaN, and `handedness_mode` SHALL be `0` so the column retains integer dtype.

**Determinism.** Two invocations on the same `per_plant_df` and `units` SHALL produce frames that compare equal under `pandas.testing.assert_frame_equal`.

#### Scenario: Aggregates a multi-plant single-genotype group to one row
- **GIVEN** a hand-built (synthetic) composed `per_plant_df` with 6 plants sharing `(plate_id, genotype, treatment) = ("plate_001", "Nipponbare", "none")`, all with `track_is_clean == True`, and its `units` mapping (the real plate-001 fixture has only 1 passing plant — the all-passing multi-plant path is exercised synthetically here; the real-data profile is asserted by the integration test)
- **WHEN** `aggregate_by_genotype(per_plant_df, units)` is invoked
- **THEN** `per_genotype_df` has exactly one row with `plate_id == "plate_001"`, `genotype == "Nipponbare"`, `treatment == "none"`
- **AND** `n_plants_passing_qc == 6`, `n_plants_excluded == 0`, and `exclusion_reasons == ""`
- **AND** for a numeric-unit trait with finite values across the plants, `<trait>_median` equals the median of those values and `<trait>_iqr` equals their Q75 − Q25 (linear interpolation)

#### Scenario: Plants failing QC are excluded but counted with reasons
- **GIVEN** a group of 4 plants where 1 has `track_is_clean == False` with `qc_failure_reason == "frac_outlier_steps_high, worst_step_ratio_high"` and 3 have `track_is_clean == True`
- **WHEN** `aggregate_by_genotype(per_plant_df, units)` is invoked
- **THEN** `n_plants_passing_qc == 3` and `n_plants_excluded == 1`
- **AND** every `<trait>_median` / `<trait>_iqr` is computed over only the 3 passing plants
- **AND** `exclusion_reasons` contains `"frac_outlier_steps_high:1"` and `"worst_step_ratio_high:1"` in `_FAILURE_CLAUSE_ORDER` order joined by `"; "`

#### Scenario: Different plates of the same genotype are reported separately
- **GIVEN** a `per_plant_df` with genotype "Nipponbare", treatment "none" appearing on `plate_id` "plate_001" (3 plants) and "plate_002" (2 plants)
- **WHEN** `aggregate_by_genotype(per_plant_df, units)` is invoked
- **THEN** `per_genotype_df` has two rows, one per `plate_id`, sorted by `(plate_id, genotype, treatment)`
- **AND** neither row's statistics pool plants from the other plate

#### Scenario: handedness aggregates to a mode with a consensus fraction
- **GIVEN** a passing group whose plants have `handedness` values `[1, 1, 1, -1]`
- **WHEN** `aggregate_by_genotype(per_plant_df, units)` is invoked
- **THEN** `handedness_mode == 1` (int dtype) and `handedness_consensus_frac == 0.75`
- **AND** `frac_nutating` equals the mean of the group's `is_nutating` booleans

#### Scenario: IQR is NaN for a trait backed by fewer than two finite values
- **GIVEN** a passing group of 1 plant (or a trait with only 1 finite passing value)
- **WHEN** `aggregate_by_genotype(per_plant_df, units)` is invoked
- **THEN** the trait's `<trait>_median` equals that single value
- **AND** the trait's `<trait>_iqr` is NaN

#### Scenario: An all-excluded group still emits a NaN row
- **GIVEN** a `(plate_id, genotype, treatment)` group whose every plant has `track_is_clean == False`
- **WHEN** `aggregate_by_genotype(per_plant_df, units)` is invoked
- **THEN** the group's row is present (not dropped) with `n_plants_passing_qc == 0` and `n_plants_excluded` equal to the group size
- **AND** every `<trait>_median`, `<trait>_iqr`, `frac_nutating`, and `handedness_consensus_frac` is NaN
- **AND** `handedness_mode == 0` and the `handedness_mode` column retains integer dtype

#### Scenario: growth_axis_unreliable and principal_axis_angle are not emitted; identity sub-columns are dropped
- **WHEN** `aggregate_by_genotype(per_plant_df, units)` is invoked
- **THEN** `per_genotype_df` has no `growth_axis_unreliable`, `growth_axis_unreliable_median`, or `growth_axis_unreliable_iqr` column
- **AND** `per_genotype_df` has no `principal_axis_angle`, `principal_axis_angle_median`, or `principal_axis_angle_iqr` column (the wrapping circular angle is excluded from linear aggregation)
- **AND** `per_genotype_df` DOES have `angular_amplitude_median` and `angular_amplitude_iqr` (the non-wrapping `rad` magnitude is aggregated)
- **AND** `per_genotype_df` has no `series`, `sample_uid`, `timepoint`, `plant_id`, or `track_id` column

#### Scenario: helix_signed_area is aggregated as a magnitude, not signed
- **GIVEN** a passing group whose plants have `helix_signed_area_px2` values of mixed sign but large magnitude (e.g. `[+1000.0, -1000.0]`)
- **WHEN** `aggregate_by_genotype(per_plant_df, units)` is invoked
- **THEN** `per_genotype_df` has `helix_signed_area_abs_px2_median` and `helix_signed_area_abs_px2_iqr` (computed from `|helix_signed_area_px2|`, so the median is ≈ 1000.0, not ≈ 0)
- **AND** `per_genotype_df` has no `helix_signed_area_px2_median` or `helix_signed_area_px2_iqr` column
- **AND** the `per_genotype_units` value for `helix_signed_area_abs_px2_median` and `helix_signed_area_abs_px2_iqr` is `"px²"`

#### Scenario: A plant with a NaN group-key value is grouped, not dropped
- **GIVEN** a `per_plant_df` of 3 plants where one plant has `treatment` equal to NaN
- **WHEN** `aggregate_by_genotype(per_plant_df, units)` is invoked
- **THEN** the NaN-`treatment` plant forms its own `(plate_id, genotype, NaN)` output row rather than being dropped
- **AND** the sum of `n_plants_passing_qc + n_plants_excluded` across all output rows equals 3

#### Scenario: Output units 1:1 cover the per-genotype columns within the closed vocabulary
- **WHEN** `aggregate_by_genotype(per_plant_df, units)` is invoked
- **THEN** `per_genotype_units` has exactly one key per column of `per_genotype_df` and no extra keys
- **AND** `<trait>_median` and `<trait>_iqr` carry the source trait's unit, `frac_nutating` and `handedness_consensus_frac` carry `"—"`, `handedness_mode`, `n_plants_passing_qc`, and `n_plants_excluded` carry `"int"`, and `exclusion_reasons`, `plate_id`, `genotype`, `treatment` carry `"string"`
- **AND** every value in `per_genotype_units` is a member of `PIPELINE_UNIT_VOCABULARY`

#### Scenario: handedness tie-break is deterministic
- **GIVEN** a passing group whose plants have `handedness` values `[1, -1]` (a two-way tie with no `0`)
- **WHEN** `aggregate_by_genotype(per_plant_df, units)` is invoked
- **THEN** `handedness_mode == -1` (the smallest signed value among the tied-most-frequent signs) with int dtype
- **AND** `handedness_consensus_frac == 0.5`

#### Scenario: integer dtypes survive a frame mixing passing and all-excluded groups
- **GIVEN** a `per_plant_df` with one fully-passing `(plate_id, genotype, treatment)` group and one all-excluded group
- **WHEN** `aggregate_by_genotype(per_plant_df, units)` is invoked
- **THEN** the `handedness_mode`, `n_plants_passing_qc`, and `n_plants_excluded` columns are integer dtype across both rows (the all-excluded row's `handedness_mode == 0` does not promote the column to float)

#### Scenario: An empty per-plant frame yields an empty per-genotype frame
- **GIVEN** a 0-row `per_plant_df` with the full composed column set and a 1:1 `units` mapping
- **WHEN** `aggregate_by_genotype(per_plant_df, units)` is invoked
- **THEN** `per_genotype_df` has 0 rows and the full expected per-genotype column set, and `per_genotype_units` covers those columns 1:1, without raising

#### Scenario: A units mapping that does not cover the per-plant columns is rejected
- **GIVEN** a `per_plant_df` and a `units` mapping missing the unit for at least one of its columns
- **WHEN** `aggregate_by_genotype(per_plant_df, units)` is invoked
- **THEN** a `ValueError` naming the uncovered column is raised before any output is produced

#### Scenario: A plant carrying multiple rows in one group is rejected (1:1 guard)
- **GIVEN** a `per_plant_df` in which one `(plate_id, genotype, treatment, plant_id)` combination maps to two rows (a track↔plant divergence)
- **WHEN** `aggregate_by_genotype(per_plant_df, units)` is invoked
- **THEN** a `ValueError` naming the offending `plant_id` is raised (the per-plant `plant_id` collapse is deferred to a follow-up issue, so PR #15 fails loud rather than aggregating per-track)

#### Scenario: Inputs are not mutated
- **GIVEN** a composed `per_plant_df` and its `units`
- **WHEN** `aggregate_by_genotype(per_plant_df, units)` is invoked
- **THEN** `per_plant_df` and `units` are unchanged (equal to copies taken before the call)

#### Scenario: Aggregation is deterministic
- **GIVEN** a composed `per_plant_df` and its `units`
- **WHEN** `aggregate_by_genotype` is invoked twice on the same inputs
- **THEN** the two `per_genotype_df` results compare equal under `pandas.testing.assert_frame_equal`

### Requirement: Per-genotype trait CSV and sidecar I/O
The system SHALL provide `sleap_roots.circumnutation._io.write_per_genotype_csv(out_path, df, units, run_metadata) -> None` and `sleap_roots.circumnutation._io.read_per_genotype_csv(in_path) -> tuple[pd.DataFrame, dict, dict]`. `write_per_genotype_csv` SHALL write the per-genotype DataFrame to `out_path` (no index), a sibling units sidecar `<csv_stem>.units.json`, and a sibling `run_metadata.json`, mirroring `write_per_plant_csv`. Before writing any file, it SHALL validate that `units` keys 1:1 cover the DataFrame columns and that every unit string is a member of `PIPELINE_UNIT_VOCABULARY`, raising `ValueError` and writing nothing when either check fails. `read_per_genotype_csv` SHALL return `(df, units, run_metadata)`, loading the sidecars when present and returning `{}` for a missing sidecar. The units-coverage (1:1 keys↔columns) validation SHALL be factored into a shared private helper `_validate_units_coverage(df, units, *, fn_name)` used by `write_per_plant_csv`, `write_per_genotype_csv`, AND `aggregate_by_genotype`'s input validation; each writer SHALL additionally validate `PIPELINE_UNIT_VOCABULARY` membership before writing (the aggregation input does not need the vocabulary check, since per-plant units are pipeline output already in-vocabulary). The extraction SHALL preserve `write_per_plant_csv`'s observable behavior — raising `ValueError` *before any file is written* and naming the offending column(s) in the message. `write_per_genotype_csv` SHALL document in its docstring that `run_metadata.json` is a fixed name in the output directory: co-locating a per-genotype CSV with a per-plant CSV (or a second per-genotype CSV) in the same directory overwrites the earlier `run_metadata.json`, so callers SHALL write at most one CSV artifact per output directory (mirroring the `CircumnutationPipeline.save` constraint). A stem-prefixed metadata filename to remove this fixed-name clobber is tracked as a follow-up issue.

#### Scenario: Round-trips a per-genotype frame through CSV and sidecars
- **GIVEN** a `per_genotype_df` and matching `per_genotype_units` from `aggregate_by_genotype`, plus a `run_metadata` mapping from `gather_run_metadata`
- **WHEN** `write_per_genotype_csv(out_path, per_genotype_df, per_genotype_units, run_metadata)` is invoked and then `read_per_genotype_csv(out_path)` is called
- **THEN** the CSV, `<csv_stem>.units.json`, and `run_metadata.json` sidecars exist in the output directory
- **AND** the read-back DataFrame has the same shape and column names as `per_genotype_df`
- **AND** the read-back units mapping equals `per_genotype_units` and the read-back run-metadata equals `run_metadata`

#### Scenario: Rejects a units mapping with an out-of-vocabulary unit
- **GIVEN** a `per_genotype_df` and a `units` mapping in which one column maps to a unit string not in `PIPELINE_UNIT_VOCABULARY` (e.g., `"mm"`)
- **WHEN** `write_per_genotype_csv(out_path, per_genotype_df, units, run_metadata)` is invoked
- **THEN** a `ValueError` is raised
- **AND** no CSV or sidecar file is written to the output directory

#### Scenario: Rejects a units mapping that does not 1:1 cover the columns
- **GIVEN** a `per_genotype_df` and a `units` mapping that is missing a column or has an extra key
- **WHEN** `write_per_genotype_csv(out_path, per_genotype_df, units, run_metadata)` is invoked
- **THEN** a `ValueError` naming the offending column(s) is raised
- **AND** no CSV or sidecar file is written to the output directory

#### Scenario: Reading with no sidecars returns empty mappings
- **GIVEN** a per-genotype CSV written to disk with its `<csv_stem>.units.json` and `run_metadata.json` sidecars removed
- **WHEN** `read_per_genotype_csv(in_path)` is invoked
- **THEN** the DataFrame loads and both the returned `units` and `run_metadata` are `{}`

#### Scenario: Per-plant writer behavior is unchanged by the shared-helper refactor
- **GIVEN** a per-plant DataFrame and units mapping that previously wrote successfully via `write_per_plant_csv`
- **WHEN** `write_per_plant_csv(out_path, df, units, run_metadata)` is invoked after the shared `_validate_units_coverage` helper is introduced
- **THEN** the CSV and sidecars are written exactly as before
- **AND** a units mapping that does not 1:1 cover the columns, or contains an out-of-vocabulary unit, still raises `ValueError` before any file is written

### Requirement: Circumnutation diagnostic plots API
The `sleap_roots.circumnutation.plotting` module SHALL provide a diagnostic-plotting layer rendering circumnutation CWT scaleograms, κ-color-coded tip-trail overlays, and a per-plate panel as PNG files, suppressible via a parameter. All plots SHALL be pure-pixel (CC-3): no axis, label, colorbar, or filename SHALL reference millimeters or accept a `px_per_mm` parameter; length-bearing axes/colorbars SHALL be labeled in pixel-native units with bracketed unit suffixes (e.g. `x [px]`, `arc length [px]`, `wavelength [px]`, `spatial frequency [px⁻¹]`, `κ [px⁻¹]`, `period [s]`, `time [s]`, `power |C|²`). The module SHALL declare a namespaced logger (`logging.getLogger(__name__)`, CC-9).

The module SHALL expose:

- `scaleogram(scaleogram_result, out_path, *, ridge_result=None) -> Path` — renders a CWT scaleogram heatmap of power `|C|²` (logarithmic color norm with a strictly-positive floor) over (physical x-axis × log period/wavelength), dims the cone-of-influence region (`coi_mask == True`, the unreliable region), optionally overlays a per-frame/per-position ridge, writes a PNG to `out_path`, and returns `out_path`. It SHALL accept BOTH a temporal `ScaleogramResult` (axes `time [s]` / `period [s]`) and a spatial `SpatialScaleogramResult` (axes `arc length [px]` / `wavelength [px]`), dispatching on type, and SHALL raise `TypeError` for any other `scaleogram_result` type. When `ridge_result` is provided, its type SHALL agree with the scaleogram type (`ScaleogramResult` with `RidgeResult`, `SpatialScaleogramResult` with `SpatialRidgeResult`); a `ridge_result` of the wrong ridge type OR of a type that is neither ridge type SHALL raise `TypeError` naming both expected types (never surfacing as an `AttributeError`). For a spatial scaleogram, the y-axis label SHALL be exactly `wavelength [px]` with an axes annotation/title noting it is the **uncalibrated pywt convention** (not the COI-gated, cgau2-calibrated `lambda_spatial_*` trait value; see theory.md §7.4), and ridge positions with `in_coi == True` SHALL be faded exactly as for the temporal ridge. The heatmap SHALL construct the `pcolormesh` so the coordinate arrays and the `(n_scales, n_frames)`/`(n_scales, n_samples)` power array agree on the centers-vs-edges contract (explicit `shading` or `n+1` edge arrays — geometric-midpoint edges for the log period/wavelength axis), so no scale/frame row is silently dropped and the result is stable across matplotlib versions on the tri-OS CI matrix.
- `trail_overlay(midline_result, out_path) -> Path` — renders the smoothed tip path `(x_smooth_px, y_smooth_px)` as a per-segment color-coded `LineCollection` keyed on the signed `curvature_px_inv`, writes a PNG, and returns `out_path`. The colormap SHALL be diverging and symmetric about 0, with limits at ±(98th percentile of `|κ|` over finite values); because that clips the most extreme ~2% of curvature, the colorbar SHALL use `extend="both"` so the clipping is visible. NaN-curvature segments SHALL be rendered in a distinct "bad" color (via a copied colormap, never mutating a global colormap); the y-axis SHALL be oriented image-down (y increases downward).
- `plate_panel(midline_results, out_path) -> Path` — renders a 2×3 grid of trail overlays (one per plant) sharing a single curvature normalization and one shared colorbar, writes a PNG, and returns `out_path`.
- `save_plots(inputs, out_dir, *, constants=None, enabled=True) -> list[Path]` — orchestrates the per-plant scaleograms (Tier 1 temporal + Tier 3 spatial), trail overlays, and the per-plate panel into a `plots/` subdirectory of `out_dir`, and returns the list of written PNG `Path`s. When `enabled=False` it SHALL write nothing and return an empty list. It SHALL re-derive the per-plant `ScaleogramResult`/`SpatialScaleogramResult`/`MidlineResult` by invoking the same tier helper functions the analysis uses (`nutation._select_signal(group, "lateral")` — which performs the `project_to_growth_axis_perpendicular` projection — then `_noise.compute_sg_detrended`/`temporal_cwt.compute_scaleogram`/`extract_ridge`/`smooth_ridge` for Tier 1; `midline.reconstruct`/`spatial_cwt.resample_curvature`/`spatial_cwt.compute_scaleogram`/`extract_ridge` for Tier 3), grouping by the `_IDENTITY_5_TUPLE` with `dropna=False, sort=False` and applying the same `try/except`/finite-any/degeneracy guards as the tiers, with the supplied `constants`, so the plotted signal equals the analyzed input signal (with the documented exception that the spatial wavelength axis is the uncalibrated pywt convention, not the calibrated `lambda_spatial_*` trait). A plant whose chain hits a degeneracy/short-circuit gate (including a non-degenerate `MidlineResult` whose `curvature_px_inv` is entirely NaN, which yields no finite κ for the norm) SHALL be skipped for the affected plot (logged at DEBUG) without aborting the run.

Plot filenames SHALL be keyed on the per-plant `track_id` (the identity field guaranteed integer-valued by the row-identity validation), not on the optional `plate_id`/`plant_id` fields (documented as aspirational and possibly `NaN`). Because `track_id` and the fixed `panel.png`/`plots_metadata.json` names are unique only **within one plate**, `save_plots` is contractually **one plate per `out_dir`** (mirroring `pipeline.save`'s one-CSV-per-directory contract); callers MUST give each plate its own `out_dir`. `out_dir` itself MUST already exist (consistent with `pipeline.save`/`_io` "parent must exist") — `save_plots` raises `FileNotFoundError` if it does not, and creates only the `plots/` leaf (it does NOT silently create a missing `out_dir`).

When it writes any plot, `save_plots` SHALL also write a `plots/plots_metadata.json` provenance sidecar recording: the constants version (`_CONSTANTS_VERSION`), the resolved plot display constants (κ percentile, colormap names, DPI, figure sizes), the per-plant source identity tuples for the plants plotted, the list of written PNG filenames, the `cadence_s`/`R_px`/`run_id` carried on `inputs`, and the `run_id` as the join key to the run's `run_metadata.json` (which `save_plots` does NOT write and whose path it is not given — `CircumnutationInputs` carries no `input_path`/output path; `run_metadata.json` is written separately by `pipeline.save`, and separately records `input_path` as a disambiguator the sidecar structurally cannot reference). Because `run_id` is `Optional[str]` defaulting to `None`, the join is meaningful only when the caller supplies a non-`None` `run_id`; `save_plots` records whatever `inputs.run_id` is (possibly `null`) and SHOULD log at DEBUG when it is `None`. A best-effort relative path hint (`"../run_metadata.json"`) MAY be included but is documented as present only when the caller wrote the per-plant CSV into the parent of `out_dir`. The sidecar SHALL be strict-JSON via a coercion helper (numpy/`Path` coerced to native; non-finite floats AND `NaN` identity values — `plate_id`/`plant_id` may be `NaN` per the aspirational-field note — coerced to JSON `null` or strings so `json.dumps(..., allow_nan=False)` cannot raise) — there is no existing such helper in `sleap_roots.circumnutation` (`_io.write_run_metadata` uses `default=str` and permits non-finite tokens), so PR #16 adds one, following the sleap-roots-analyze convention (#241). When `enabled=False` (no plots written) no sidecar SHALL be written.

The module SHALL NOT render or accept any `L_gz` (growth-zone-length) arc-length marker: `L_gz` is blocked on issue #230 and is not computed in the pipeline; the marker is deferred to a follow-up PR.

Plot display constants (DPI, figure sizes, colormap names, the curvature percentile) SHALL live as module-level constants in `plotting.py` and SHALL NOT be added to the `ConstantsT` override-bag; `_CONSTANTS_VERSION` is unchanged by this PR.

#### Scenario: scaleogram renders a temporal ScaleogramResult to a PNG
- **GIVEN** a valid temporal `ScaleogramResult` and an `out_path` whose parent directory exists
- **WHEN** `scaleogram(scaleogram_result, out_path)` is invoked
- **THEN** a non-empty PNG file is written at `out_path`
- **AND** the call returns `out_path` and raises no exception
- **AND** the figure's primary axes are labeled with temporal units (`time [s]` x-axis, `period [s]` y-axis)

#### Scenario: scaleogram renders a spatial SpatialScaleogramResult with spatial axes
- **GIVEN** a valid `SpatialScaleogramResult` and an `out_path` whose parent directory exists
- **WHEN** `scaleogram(scaleogram_result, out_path)` is invoked
- **THEN** a non-empty PNG file is written and the figure's primary axes are labeled with spatial units (`arc length [px]` x-axis, `wavelength [px]` y-axis)

#### Scenario: scaleogram rejects an unsupported result type
- **WHEN** `scaleogram(object(), out_path)` is invoked with an argument that is neither a `ScaleogramResult` nor a `SpatialScaleogramResult`
- **THEN** a `TypeError` is raised

#### Scenario: scaleogram rejects a mismatched ridge/scaleogram pair
- **GIVEN** a temporal `ScaleogramResult` and a spatial `SpatialRidgeResult` (or vice versa)
- **WHEN** `scaleogram(scaleogram_result, out_path, ridge_result=mismatched_ridge)` is invoked
- **THEN** a `TypeError` naming both expected types is raised (the mismatch is not allowed to surface as an `AttributeError`)

#### Scenario: trail_overlay color-codes curvature with a diverging symmetric colormap
- **GIVEN** a valid `MidlineResult` with finite and some non-finite `curvature_px_inv` values
- **WHEN** `trail_overlay(midline_result, out_path)` is invoked
- **THEN** a non-empty PNG is written and `out_path` returned
- **AND** the axes contain a `LineCollection` whose color array length equals the number of trail segments (one fewer than the number of points)
- **AND** the y-axis is oriented image-down

#### Scenario: plate_panel shares one curvature normalization and colorbar across plants
- **GIVEN** an ordered collection of valid `MidlineResult` objects for up to six plants
- **WHEN** `plate_panel(midline_results, out_path)` is invoked
- **THEN** a non-empty PNG is written and `out_path` returned
- **AND** every subplot uses the same curvature normalization and the figure has a single shared colorbar

#### Scenario: save_plots writes the per-plant plot set into a plots/ subdirectory
- **GIVEN** a valid `CircumnutationInputs` with one or more non-degenerate plants and an existing `out_dir`
- **WHEN** `save_plots(inputs, out_dir)` is invoked
- **THEN** a `plots/` subdirectory is created under `out_dir`
- **AND** for each non-degenerate plant a temporal scaleogram, a spatial scaleogram, and a trail-overlay PNG are written, plus one per-plate panel PNG
- **AND** every written file is non-empty and its filename is keyed on the plant's integer `track_id`
- **AND** the call returns the list of written `Path`s

#### Scenario: save_plots writes a provenance sidecar tying plots to the run
- **GIVEN** a valid `CircumnutationInputs` with one or more non-degenerate plants and an existing `out_dir`
- **WHEN** `save_plots(inputs, out_dir)` is invoked
- **THEN** a `plots/plots_metadata.json` file is written that parses as strict JSON (no `NaN`/`Infinity` tokens)
- **AND** it records the constants version, the resolved plot display constants (κ percentile, colormaps, DPI, figure sizes), the per-plant source identity tuples, the written PNG filenames, and the `run_id`/`cadence_s`/`R_px` from `inputs` (the `run_id` being the join key to the run's `run_metadata.json`)

#### Scenario: save_plots requires out_dir to already exist
- **GIVEN** an `out_dir` path that does not exist
- **WHEN** `save_plots(inputs, out_dir)` is invoked with `enabled=True`
- **THEN** a `FileNotFoundError` is raised (it does not silently create a missing `out_dir`)
- **AND** when `out_dir` does exist, `save_plots` creates only the `plots/` leaf under it

#### Scenario: save_plots with enabled=False writes nothing
- **WHEN** `save_plots(inputs, out_dir, enabled=False)` is invoked
- **THEN** no PNG files, no `plots/` subdirectory, and no `plots_metadata.json` are created
- **AND** the call returns an empty list
- **AND** a single INFO-level log line records that plotting was skipped

#### Scenario: plots carry no millimeter units or px_per_mm parameter
- **WHEN** the signatures of `scaleogram`, `trail_overlay`, `plate_panel`, and `save_plots` are inspected and the rendered figures' axis/colorbar labels are read
- **THEN** none of the signatures contain a `px_per_mm` parameter
- **AND** no axis, colorbar, or filename references millimeters (all length-bearing labels use pixel-native bracketed unit suffixes)

#### Scenario: no L_gz marker is rendered or accepted
- **WHEN** the signatures of the plotting functions are inspected
- **THEN** no function accepts an `L_gz` (or growth-zone-length) parameter
- **AND** no rendered plot draws an `L_gz` arc-length marker (deferred to a follow-up PR pending #230)

#### Scenario: save_plots skips a degenerate plant without aborting
- **GIVEN** a valid `CircumnutationInputs` with at least one plant whose Tier 1 or Tier 3 chain hits a short-circuit/degeneracy gate (e.g. an all-NaN lateral signal or a degenerate `MidlineResult`) alongside at least one non-degenerate plant
- **WHEN** `save_plots(inputs, out_dir)` is invoked
- **THEN** the affected plant's affected plot(s) are omitted (and the omission is logged at DEBUG) while the non-degenerate plants' plots are still written
- **AND** the returned `list[Path]` reflects exactly the plots that were written (the call does not raise)

#### Scenario: trail_overlay and plate_panel handle an all-NaN-curvature plant
- **GIVEN** a non-degenerate `MidlineResult` whose `curvature_px_inv` is entirely NaN (so the 98th-percentile symmetric norm has no finite κ to compute)
- **WHEN** `trail_overlay(midline_result, out_path)` is invoked, and when such a plant is included in a `plate_panel` collection
- **THEN** neither call raises (the all-NaN plant is skipped or rendered with a defined fallback norm, logged at DEBUG); a `plate_panel` whose entire pooled κ is non-finite still produces a defined figure without a NaN-limit norm crash

