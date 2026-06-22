## MODIFIED Requirements
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

**Scope note on PR #16 stub-to-implementation transition.** The `plotting` module WAS a stub in PR #1–#15 (it appeared in the stub-callable table with canonical callable `scaleogram(scaleogram_result, out_path)`, PR #16). PR #16 graduates it to an implementation module: the implementation-module count grows from 11 to **12** AND the stub-module count shrinks from 2 to **1** (the same stub→impl shape as PR #7/#8/#9/#14 — only `parametric`/PR #11 remains a stub). The canonical callable KEEPS its name `scaleogram` (no rename); the implementation signature gains a keyword-only `ridge_result=None` and now returns the written `Path` (`scaleogram(scaleogram_result, out_path, *, ridge_result=None) -> Path`), and the module ADDS three further public symbols — `trail_overlay`, `plate_panel`, and a `save_plots` orchestrator — whose callability contracts are locked here for symmetry, mirroring how PR #5/#9's `extract_ridge` and PR #6's `smooth_ridge` were locked. `plotting` is therefore removed from the stub-callable table and from the "remaining stub" enumeration. The stub had **no** forward-compatible `constants=None` parameter, so no prior `constants=` scenario is affected. **Deviation note:** the `plotting` stub docstring promised "tip-trail overlays with κ-color-coding and an `L_gz` arc-length marker". `L_gz` (growth-zone length) is blocked on #230 and is not computed anywhere in the pipeline, so PR #16 OMITS the `L_gz` marker entirely (the graduated docstrings drop the claim; no `L_gz` parameter is added to any function); a follow-up PR adds it once #230 lands. See Requirement: Circumnutation diagnostic plots API.

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

## ADDED Requirements
### Requirement: Circumnutation diagnostic plots API
The `sleap_roots.circumnutation.plotting` module SHALL provide a diagnostic-plotting layer rendering circumnutation CWT scaleograms, κ-color-coded tip-trail overlays, and a per-plate panel as PNG files, suppressible via a parameter. All plots SHALL be pure-pixel (CC-3): no axis, label, colorbar, or filename SHALL reference millimeters or accept a `px_per_mm` parameter; length-bearing axes/colorbars SHALL be labeled in pixel-native units with bracketed unit suffixes (e.g. `x [px]`, `arc length [px]`, `wavelength [px]`, `spatial frequency [px⁻¹]`, `κ [px⁻¹]`, `period [s]`, `time [s]`, `power |C|²`). The module SHALL declare a namespaced logger (`logging.getLogger(__name__)`, CC-9).

The module SHALL expose:

- `scaleogram(scaleogram_result, out_path, *, ridge_result=None) -> Path` — renders a CWT scaleogram heatmap of power `|C|²` (logarithmic color norm with a strictly-positive floor) over (physical x-axis × log period/wavelength), dims the cone-of-influence region (`coi_mask == True`, the unreliable region), optionally overlays a per-frame/per-position ridge, writes a PNG to `out_path`, and returns `out_path`. It SHALL accept BOTH a temporal `ScaleogramResult` (axes `time [s]` / `period [s]`) and a spatial `SpatialScaleogramResult` (axes `arc length [px]` / `wavelength [px]`), dispatching on type, and SHALL raise `TypeError` for any other `scaleogram_result` type. When `ridge_result` is provided, its type SHALL agree with the scaleogram type (`ScaleogramResult` with `RidgeResult`, `SpatialScaleogramResult` with `SpatialRidgeResult`); a mismatched pair SHALL raise `TypeError` naming both expected types.
- `trail_overlay(midline_result, out_path) -> Path` — renders the smoothed tip path `(x_smooth_px, y_smooth_px)` as a per-segment color-coded `LineCollection` keyed on the signed `curvature_px_inv`, writes a PNG, and returns `out_path`. The colormap SHALL be diverging and symmetric about 0; NaN-curvature segments SHALL be rendered in a distinct "bad" color; the y-axis SHALL be oriented image-down (y increases downward).
- `plate_panel(midline_results, out_path) -> Path` — renders a 2×3 grid of trail overlays (one per plant) sharing a single curvature normalization and one shared colorbar, writes a PNG, and returns `out_path`.
- `save_plots(inputs, out_dir, *, constants=None, enabled=True) -> list[Path]` — orchestrates the per-plant scaleograms (Tier 1 temporal + Tier 3 spatial), trail overlays, and the per-plate panel into a `plots/` subdirectory of `out_dir`, and returns the list of written PNG `Path`s. When `enabled=False` it SHALL write nothing and return an empty list. It SHALL re-derive the per-plant `ScaleogramResult`/`SpatialScaleogramResult`/`MidlineResult` by invoking the same tier helper functions the analysis uses (`_geometry.project_to_growth_axis_perpendicular`/`_noise.compute_sg_detrended`/`temporal_cwt.compute_scaleogram`/`extract_ridge`/`smooth_ridge` for Tier 1; `midline.reconstruct`/`spatial_cwt.resample_curvature`/`compute_scaleogram`/`extract_ridge` for Tier 3) with the supplied `constants`, so the plotted signal equals the analyzed signal. A plant whose chain hits a degeneracy/short-circuit gate SHALL be skipped for the affected plot (logged at DEBUG) without aborting the run.

Plot filenames SHALL be keyed on the per-plant `track_id` (the identity field guaranteed integer-valued by the row-identity validation), not on the optional `plate_id`/`plant_id` fields (documented as aspirational and possibly `NaN`). The `plots/` subdirectory SHALL be created if absent.

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

#### Scenario: save_plots with enabled=False writes nothing
- **WHEN** `save_plots(inputs, out_dir, enabled=False)` is invoked
- **THEN** no PNG files and no `plots/` subdirectory are created
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
