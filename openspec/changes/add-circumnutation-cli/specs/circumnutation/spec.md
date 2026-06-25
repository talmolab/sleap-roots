## MODIFIED Requirements
### Requirement: Package layout
The system SHALL provide a `sleap_roots.circumnutation` sub-package whose import-tree is complete from this PR onward — every module name referenced anywhere in `docs/circumnutation/roadmap.md` exists as an importable Python module from PR #1 forward, even if its functions raise `NotImplementedError` until later PRs land. The package SHALL contain:

- 7 contract modules: `__init__.py`, `_constants.py`, `_types.py`, `_io.py`, `units.py`, `_noise.py`, `_geometry.py`
- 14 implementation modules: `kinematics` (implemented from PR #2 onward; see Requirement: Tier 0 raw kinematic traits), `qc` (implemented from PR #3 onward; see Requirement: QC tier per-track quality traits), `synthetic` (implemented from PR #4 onward; see Requirement: Synthetic trajectory generator), `temporal_cwt` (implemented from PR #5 onward; see Requirement: Temporal CWT scaleogram API and Requirement: Temporal CWT ridge API), `nutation` (implemented from PR #6 onward; see Requirement: Tier 1 nutation trait emission API), `psi_g` (implemented from PR #7 onward; see Requirement: Tier 2 ψ_g trait emission API), `midline` (implemented from PR #8 onward; see Requirement: Tier 3a midline reconstruction API), `spatial_cwt` (implemented from PR #9 onward; see Requirement: Tier 3b spatial curvature resample API, Requirement: Tier 3b spatial CWT scaleogram API, and Requirement: Tier 3b spatial CWT ridge API), `traveling_wave` (implemented from PR #10 onward; see Requirement: Tier 3c traveling-wave trait emission API), `pipeline` (implemented from PR #14 onward; see Requirement: Circumnutation pipeline composition API), `aggregation` (implemented from PR #15 onward; see Requirement: Per-genotype aggregation API), `plotting` (implemented from PR #16 onward; see Requirement: Circumnutation diagnostic plots API), `adapters` (implemented from PR #17 onward; see Requirement: Series-to-CircumnutationInputs adapter), and `cli` (implemented from PR #17 onward; see Requirement: Circumnutation analyze CLI)
- 1 stub module: `parametric`

Each stub module SHALL define exactly one canonical callable per the table below, raising `NotImplementedError(f"PR #{N} — see docs/circumnutation/roadmap.md")` with the documented PR number. Each stub callable SHALL also have a complete Google-style docstring (Args, Returns, Raises) so `pydocstyle --convention=google` and the mkdocstrings auto-generated reference pages remain informative.

| Stub module | Canonical callable | PR # |
|---|---|---|
| `parametric` | `compute(tier3_df, R_px, omega, Delta_phi)` | 11 |

The `kinematics` module SHALL be importable on the same terms as the other modules (clean import, namespaced logger) and SHALL expose `compute(trajectory_df, constants=None)` per Requirement: Tier 0 raw kinematic traits. The `qc` module SHALL be importable on the same terms and SHALL expose `compute(trajectory_df, constants=None)` per Requirement: QC tier per-track quality traits. The `synthetic` module SHALL be importable on the same terms and SHALL expose `generate_trajectory(...)` per Requirement: Synthetic trajectory generator. The `temporal_cwt` module SHALL be importable on the same terms and SHALL expose `compute_scaleogram(x, cadence_s, constants=None) -> ScaleogramResult` and `extract_ridge(scaleogram_result, constants=None) -> RidgeResult` per Requirement: Temporal CWT scaleogram API and Requirement: Temporal CWT ridge API, AND SHALL ALSO expose `smooth_ridge(ridge_result, window=None, constants=None) -> RidgeResult` per Requirement: Temporal CWT ridge-continuity smoothing API. The `nutation` module SHALL be importable on the same terms and SHALL expose `compute(trajectory_df, cadence_s, coordinate="lateral", constants=None) -> pd.DataFrame` per Requirement: Tier 1 nutation trait emission API. The `psi_g` module SHALL be importable on the same terms and SHALL expose `compute(trajectory_df, cadence_s, constants=None) -> pd.DataFrame` per Requirement: Tier 2 ψ_g trait emission API. The `midline` module SHALL be importable on the same terms and SHALL expose `reconstruct(x, y, cadence_s, sg_window=None, constants=None) -> MidlineResult` per Requirement: Tier 3a midline reconstruction API. The `spatial_cwt` module SHALL be importable on the same terms and SHALL expose `compute_scaleogram(kappa, ds, constants=None) -> SpatialScaleogramResult` and `extract_ridge(scaleogram_result, constants=None) -> SpatialRidgeResult` per Requirement: Tier 3b spatial CWT scaleogram API and Requirement: Tier 3b spatial CWT ridge API, AND SHALL ALSO expose `resample_curvature(curvature_px_inv, arc_length_px, velocity_sub_noise_mask=None, constants=None) -> ResampleResult` per Requirement: Tier 3b spatial curvature resample API. The `traveling_wave` module SHALL be importable on the same terms and SHALL expose `compute(trajectory_df, cadence_s, constants=None, *, tier0_df=None, tier1_df=None) -> pd.DataFrame` per Requirement: Tier 3c traveling-wave trait emission API. The `pipeline` module SHALL be importable on the same terms and SHALL expose `compute_traits(inputs, constants=None) -> tuple[pd.DataFrame, pd.DataFrame, dict]` and the `CircumnutationPipeline` class per Requirement: Circumnutation pipeline composition API. The `aggregation` module SHALL be importable on the same terms and SHALL expose `aggregate_by_genotype(per_plant_df, units) -> tuple[pd.DataFrame, dict]` per Requirement: Per-genotype aggregation API. The `plotting` module SHALL be importable on the same terms and SHALL expose `scaleogram(scaleogram_result, out_path, *, ridge_result=None) -> Path`, `trail_overlay(midline_result, out_path) -> Path`, `plate_panel(midline_results, out_path) -> Path`, and `save_plots(inputs, out_dir, *, constants=None, enabled=True) -> list[Path]` per Requirement: Circumnutation diagnostic plots API. The `adapters` module SHALL be importable on the same terms and SHALL expose `series_to_inputs(series, *, cadence_s, sample_uid, series_name=None, timepoint=None, plate_id=None, genotype=None, treatment=None, r_px=None, run_id=None) -> tuple[CircumnutationInputs, dict]` (the `CircumnutationInputs` plus an identity-provenance dict) per Requirement: Series-to-CircumnutationInputs adapter. The `cli` module SHALL be importable on the same terms and SHALL expose a `circumnutation` `click` group containing an `analyze` command (registered on the root `sleap_roots.cli:main` group via `main.add_command(circumnutation)`) per Requirement: Circumnutation analyze CLI. Unlike the stub modules, calling `kinematics.compute`, `qc.compute`, `synthetic.generate_trajectory`, `temporal_cwt.compute_scaleogram`, `nutation.compute`, `psi_g.compute`, `midline.reconstruct`, `spatial_cwt.resample_curvature`, `spatial_cwt.compute_scaleogram`, `spatial_cwt.extract_ridge`, `traveling_wave.compute`, `pipeline.compute_traits`, `aggregation.aggregate_by_genotype`, `plotting.scaleogram`, `plotting.trail_overlay`, `plotting.plate_panel`, `plotting.save_plots`, or `adapters.series_to_inputs` with a valid input SHALL NOT raise `NotImplementedError`.

The `_noise` and `_geometry` modules are private (underscore-prefixed) shared internals; their canonical callables are documented under Requirement: Tier 0 helper modules (and, for `_geometry.compute_signed_area`, under Requirement: Tier 2 ψ_g trait emission API; and, for `_noise.compute_sg_derivative` / `_geometry.compute_path_curvature`, under Requirement: Tier 3a midline reconstruction API).

**Scope note on PR #6 addition-vs-transition.** The `nutation` module is NEWLY created in PR #6 — it was never a stub module in PR #1–#5, and therefore does not appear in the stub-callable table. The implementation-module count grows from 4 (PR #5 baseline: kinematics, qc, synthetic, temporal_cwt) to 5 by ADDITION of `nutation`, not by transition from a prior stub. This was the first PR in the program to grow the implementation set without shrinking the stub set.

**Scope note on PR #7 stub-to-implementation transition.** The `psi_g` module was a stub in PR #1–#6 (it appeared in the stub-callable table with canonical callable `compute_psi_g(x, y, constants=None)`). PR #7 graduated it to an implementation module: the implementation-module count grew from 5 to 6 AND the stub-module count shrank from 6 to 5. The canonical callable was RENAMED `compute_psi_g` → `compute`.

**Scope note on PR #8 stub-to-implementation transition.** The `midline` module was a stub in PR #1–#7 (it appeared in the stub-callable table with canonical callable `reconstruct(x, y, cadence_s, constants=None)`). PR #8 graduated it to an implementation module: the implementation-module count grew from 6 to 7 AND the stub-module count shrank from 5 to 4 (the same stub→impl shape as PR #7). The canonical callable KEPT its name `reconstruct` (no rename); the implementation signature ADDED a `sg_window=None` parameter (`reconstruct(x, y, cadence_s, constants=None)` → `reconstruct(x, y, cadence_s, sg_window=None, constants=None)`), locked by Requirement: Tier 3a midline reconstruction API.

**Scope note on PR #9 stub-to-implementation transition.** The `spatial_cwt` module IS a stub in PR #1–#8 (it appeared in the stub-callable table with canonical callable `compute_scaleogram(kappa, ds, constants=None)`, PR #9). PR #9 graduates it to an implementation module: the implementation-module count grows from 7 to **8** AND the stub-module count shrinks from 4 to **3** (the same stub→impl shape as PR #7/#8). The canonical callable KEEPS its name `compute_scaleogram` (no rename); the implementation signature is EXACTLY the stub-table signature `compute_scaleogram(kappa, ds, constants=None)` — the speculative `wavelet=`/`scale_range=` keyword parameters present in the PR #1 stub file are DROPPED (the wavelet and scale range are derived from `constants`, mirroring `temporal_cwt.compute_scaleogram`'s `(x, cadence_s, constants=None)` precedent). `spatial_cwt` is therefore removed from the stub-callable table and from the "remaining stub" enumeration, and gains callability scenarios below. PR #9 ADDS two further public symbols not previously in the stub table — `resample_curvature` (the non-uniform→uniform κ(s) resample entry helper) and `extract_ridge` (the spatial ridge) — whose callability contracts are locked here for symmetry with `compute_scaleogram`, mirroring how PR #5's `extract_ridge` and PR #6's `smooth_ridge` were locked. **PR #9 descopes `L_gz`/`L_c` growth-zone-structure detection** (the §7.4 |κ|-envelope-peak premise does not transfer to top-view tip-trail κ(s); see the Tier 3b requirements and follow-up issue #230); `spatial_cwt` therefore exposes no `detect_growth_zone` symbol.

**Scope note on PR #10 addition-vs-transition.** The `traveling_wave` module is NEWLY created in PR #10 — it was never a stub module in PR #1–#9 (only `parametric`, `plotting`, `pipeline` remain stubs at PR #10), and therefore does not appear in the stub-callable table. The implementation-module count grows from 8 to **9** by ADDITION of `traveling_wave`, not by transition from a prior stub (the same addition shape as PR #6's `nutation`); the stub-module count is UNCHANGED at 3. The canonical callable is `compute(trajectory_df, cadence_s, constants=None) -> pd.DataFrame` (the Tier 1/Tier 2 trait-emission signature; `cadence_s` is an explicit positional parameter). **PR #10 ships reduced scope** per the PR #9 `L_gz`/`L_c` descope (#230): it emits only the 3 λ-based traits + diagnostics on the full reconstructed trail (no growth-zone mask); the 5 `L_gz`/`L_c`-dependent traits and the mask remain blocked on #230 and are OMITTED from the output schema (not reserved as NaN columns). `traveling_wave` gains a callability scenario below.

**Scope note on PR #14 stub-to-implementation transition.** The `pipeline` module WAS a stub in PR #1–#10 (it appeared in the stub-callable table with canonical callable `compute_traits(inputs, constants=None)`, PR #14). PR #14 graduates it to an implementation module: the implementation-module count grows from 9 to **10** AND the stub-module count shrinks from 3 to **2** (the same stub→impl shape as PR #7/#8/#9). The canonical callable KEEPS its name `compute_traits` (no rename); the implementation adds keyword-only file-writing via a separate `CircumnutationPipeline.save(...)` method and a picklable `CircumnutationPipeline` class, locked by Requirement: Circumnutation pipeline composition API. `pipeline` is therefore removed from the stub-callable table and from the "remaining stub" enumeration, and gains a callability scenario below. **`pipeline` was the last stub whose canonical callable carried the forward-compatible `constants=None` parameter** — the 2 remaining stubs (`parametric`, `plotting`) do not, so the prior "Stubs accept `constants=None` where the table prescribes it" scenario is REMOVED (no remaining subject). **Deviation note:** the `pipeline` stub docstring and the roadmap row described PR #14 as building "a TraitDef DAG matching the `Pipeline` base class pattern". PR #14 instead implements a **sequential merge-orchestrator** (the circumnutation tiers are per-track `DataFrame → DataFrame` functions, not the per-frame networkx `TraitDef` nodes of `sleap_roots.trait_pipelines.Pipeline`); the documented tier-order dependency structure is preserved without the per-frame node model. The stub docstring + roadmap row are corrected to match.

**Scope note on PR #15 addition-vs-transition.** The `aggregation` module is NEWLY created in PR #15 — it was never a stub module in PR #1–#14 (only `parametric` and `plotting` remain stubs at PR #15), and therefore does not appear in the stub-callable table. The implementation-module count grows from 10 to **11** by ADDITION of `aggregation`, not by transition from a prior stub (the same addition shape as PR #6's `nutation` and PR #10's `traveling_wave`); the stub-module count is UNCHANGED at 2. The canonical callable is `aggregate_by_genotype(per_plant_df, units) -> tuple[pd.DataFrame, dict]` (a post-pipeline aggregation that consumes the PR #14 composed per-plant frame). `aggregation` gains a callability scenario below.

**Scope note on PR #16 stub-to-implementation transition.** The `plotting` module WAS a stub in PR #1–#15 (it appeared in the stub-callable table with canonical callable `scaleogram(scaleogram_result, out_path)`, PR #16). PR #16 graduates it to an implementation module: the implementation-module count grows from 11 to **12** AND the stub-module count shrinks from 2 to **1** (the same stub→impl shape as PR #7/#8/#9/#14 — only `parametric`/PR #11 remains a stub). The canonical callable KEEPS its name `scaleogram` (no rename); the implementation signature gains a keyword-only `ridge_result=None` and now returns the written `Path` (`scaleogram(scaleogram_result, out_path, *, ridge_result=None) -> Path`), and the module ADDS three further public symbols — `trail_overlay`, `plate_panel`, and a `save_plots` orchestrator — whose callability contracts are locked here for symmetry, mirroring how PR #5/#9's `extract_ridge` and PR #6's `smooth_ridge` were locked. `plotting` is therefore removed from the stub-callable table and from the "remaining stub" enumeration. The stub had **no** forward-compatible `constants=None` parameter, so no prior `constants=` scenario is affected. The intro paragraph's sentence requiring stubs to carry a forward-compatible `constants=None` (preserved through PR #15) is now DROPPED here: the sole remaining stub (`parametric`) takes no `constants=` parameter, so that normative clause has no remaining subject. **Deviation note:** the `plotting` stub docstring promised "tip-trail overlays with κ-color-coding and an `L_gz` arc-length marker". `L_gz` (growth-zone length) is blocked on #230 and is not computed anywhere in the pipeline, so PR #16 OMITS the `L_gz` marker entirely (the graduated docstrings drop the claim; no `L_gz` parameter is added to any function); a follow-up PR adds it once #230 lands. See Requirement: Circumnutation diagnostic plots API.

**Scope note on PR #17 addition-vs-transition.** The `adapters` and `cli` modules are NEWLY created in PR #17 — neither was ever a stub module in PR #1–#16 (only `parametric`/PR #11 remains a stub at PR #17), so neither appears in the stub-callable table. The implementation-module count grows from 12 to **14** by ADDITION of BOTH `adapters` and `cli` (the same addition shape as PR #6's `nutation`, PR #10's `traveling_wave`, and PR #15's `aggregation` — but two modules in one PR, a first for the program); the stub-module count is UNCHANGED at 1. Neither module is referenced by name in `docs/circumnutation/roadmap.md` (which describes them only functionally — roadmap row 17 "`sleap-roots circumnutation analyze`" and the conceptual "adapter / cleanup step" of CC-4); they are net-new modules introduced at PR #17, not part of the PR #1-seeded roadmap import tree. `adapters` exposes `series_to_inputs(...)` (the `Series → CircumnutationInputs` bridge formalizing the `_load_plate001_inputs` test blueprint); `cli` exposes the `circumnutation` `click` group + `analyze` command. Both gain callability scenarios below.

#### Scenario: All stub modules import cleanly
- **WHEN** a user runs `import sleap_roots.circumnutation as c; import sleap_roots.circumnutation.kinematics, sleap_roots.circumnutation.qc, sleap_roots.circumnutation.synthetic, sleap_roots.circumnutation.temporal_cwt, sleap_roots.circumnutation.nutation, sleap_roots.circumnutation.psi_g, sleap_roots.circumnutation.midline, sleap_roots.circumnutation.spatial_cwt, sleap_roots.circumnutation.traveling_wave, sleap_roots.circumnutation.parametric, sleap_roots.circumnutation.plotting, sleap_roots.circumnutation.pipeline, sleap_roots.circumnutation.aggregation, sleap_roots.circumnutation.adapters, sleap_roots.circumnutation.cli`
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

#### Scenario: `adapters.series_to_inputs` is callable on a valid Series without raising
- **GIVEN** a valid tracked `Series` (loaded from a `.slp` whose tracks yield `get_tracked_tips()` rows) and a positive finite `cadence_s` plus a `sample_uid`
- **WHEN** `sleap_roots.circumnutation.adapters.series_to_inputs(series, cadence_s=300.0, sample_uid="plate_001", genotype="Nipponbare")` is invoked
- **THEN** the call returns a 2-tuple `(CircumnutationInputs, identity_provenance_dict)` without raising `NotImplementedError`
- **AND** since `adapters` is a NEW module introduced by PR #17 (not a transition from a prior stub), it does not appear in the stub-callable table — its callability contract is locked here in the MODIFIED Package layout requirement

#### Scenario: `cli` exposes the `circumnutation analyze` command registered on the root CLI
- **GIVEN** the package imported as `import sleap_roots.cli` and `import sleap_roots.circumnutation.cli`
- **WHEN** the root `sleap_roots.cli:main` group's registered commands are inspected
- **THEN** a `circumnutation` group is present, containing an `analyze` command
- **AND** invoking `analyze --help` via `click.testing.CliRunner` exits 0 without raising `NotImplementedError`
- **AND** since `cli` is a NEW module introduced by PR #17 (not a transition from a prior stub), it does not appear in the stub-callable table — its contract is locked here in the MODIFIED Package layout requirement and in Requirement: Circumnutation analyze CLI

#### Scenario: `synthetic.generate_trajectory` has no `px_per_mm` parameter
- **WHEN** `inspect.signature(sleap_roots.circumnutation.synthetic.generate_trajectory)` is inspected
- **THEN** the parameter list does not contain `px_per_mm`
- **AND** the docstring confirms the generator emits pure-pixel trajectories (callers compose `convert_to_mm()` if they want mm output)

#### Scenario: `import sleap_roots` succeeds without raising
- **WHEN** a user runs `import sleap_roots`
- **THEN** no exception is raised
- **AND** `sleap_roots.CircumnutationInputs` is accessible
- **AND** `sleap_roots.convert_to_mm` is accessible

### Requirement: Run-metadata sidecar
For every per-plant CSV the system SHALL write a sibling `run_metadata.json` capturing: `input_path`, `sleap_roots_git_sha`, `sleap_roots_version`, `sleap_io_version`, `numpy_version`, `scipy_version`, `pandas_version`, `python_version`, `platform`, `timestamp` (ISO 8601 UTC), `run_id`, `cadence_s`, `R_px`, `metadata_csv_path`, `metadata_csv_sha256`, `identity_source`, `_schema_version`, `_constants_version`, `_constants_snapshot`. The `cadence_s` and `R_px` fields (added in PR #14) capture the per-run acquisition inputs from `CircumnutationInputs` so a run is reproducible from the sidecars alone — `cadence_s` determines every period trait (`*_s`) and the traveling-wave residual, and is otherwise unrecoverable from the `.slp` at `input_path`. The `metadata_csv_path`, `metadata_csv_sha256`, and `identity_source` fields (added in PR #17) capture identity provenance so a reader of the sidecar can determine, without re-running, **where each row-identity field came from**:
- `metadata_csv_path` — the resolved-absolute path of the `--metadata-csv` (or `null` when no metadata CSV was used).
- `metadata_csv_sha256` — the SHA-256 hex digest of the metadata-CSV bytes (or `null`). Because the CSV is an external **mutable** input recorded by reference (its *contents* are not snapshotted by value, unlike `cadence_s`/`R_px`), the hash lets a reader verify the join is reproducible and detect post-run CSV drift.
- `identity_source` — a JSON-serializable mapping that is **total over all six adapter-populated row-identity fields** (`series`, `sample_uid`, `timepoint`, `plate_id`, `genotype`, `treatment`); every one of the six SHALL be present (a missing key signals a bug, not absence). Each maps to the literal source that won it: `"flag"` (an explicit CLI option / adapter argument), `"metadata_csv"` (a non-null cell resolved via `Series.get_metadata`), `"default"` (a real derived default — used for the filename-stem default of `series`), or `"absent"` (no value — the `NaN` fallback for `genotype`/`treatment`/`timepoint`/`plate_id` when neither flag nor CSV supplied one). `"default"` and `"absent"` are distinct: `"default"` means a value is present (derived), `"absent"` means the field is `NaN`. `plant_id`/`track_id` are intentionally **excluded** from `identity_source` because they are mechanically derived from the `.slp` track names (see Requirement: Series-to-CircumnutationInputs adapter), not sourced from a flag or CSV — their provenance is the `input_path` itself.

`gather_run_metadata` SHALL accept `cadence_s`, `R_px`, `metadata_csv_path`, `metadata_csv_sha256`, and `identity_source` as optional keyword parameters appended after the existing parameters (default `None`), so existing callers (which pass nothing past `input_path` positionally) are unaffected and write `null`; the PR #14 pipeline `save` SHALL pass `inputs.cadence_s` and `inputs.R_px`, and the PR #17 CLI SHALL additionally pass `metadata_csv_path`, `metadata_csv_sha256`, and `identity_source` (the adapter-computed identity provenance) to the top-level, per-plant, and per-genotype writes so all three sidecars carry identical identity provenance. The `_constants_snapshot` SHALL be a JSON-serializable mapping from every name in `_constants.py` to its value at write time. The numpy / scipy / pandas / platform fields support numerical reproducibility (IEEE float rounding can differ between numpy versions and across BLAS implementations).

#### Scenario: Run-metadata sidecar contains required fields
- **WHEN** the foundation writes a CSV
- **THEN** a sibling `run_metadata.json` exists in the same directory
- **AND** every key listed above is present and non-null (except `run_id`, `cadence_s`, `R_px`, `metadata_csv_path`, `metadata_csv_sha256`, and `identity_source`, which may be null when not supplied)
- **AND** `_schema_version` and `_constants_version` are integers
- **AND** `_constants_snapshot` contains every name in `_constants.py` with its current value
- **AND** `numpy_version`, `scipy_version`, `pandas_version` are present as version strings (e.g. `"2.3.4"`, `"1.16.3"`, `"2.2.0"`) or `"unknown"` if the dependency could not be imported
- **AND** `platform` is a non-empty string (the value of `platform.platform()`)

#### Scenario: Run-metadata captures cadence_s and R_px from the pipeline
- **GIVEN** the PR #14 pipeline `save` invoked with `inputs.cadence_s == 300.0` and `inputs.R_px == 12.0`
- **WHEN** `gather_run_metadata(input_path, run_id=inputs.run_id, constants=..., cadence_s=300.0, R_px=12.0)` writes the sidecar
- **THEN** `run_metadata["cadence_s"] == 300.0` and `run_metadata["R_px"] == 12.0`
- **AND** WHEN `cadence_s` / `R_px` are omitted (a non-pipeline caller), both keys are present with value `null`

#### Scenario: Run-metadata captures identity provenance from the CLI
- **GIVEN** the PR #17 CLI run with `--metadata-csv <csv>` (resolving `genotype` from the CSV) and an explicit `--treatment MOCK` flag, no `--plate-id`
- **WHEN** `gather_run_metadata(..., metadata_csv_path=<resolved-abs csv>, metadata_csv_sha256=<digest>, identity_source={"genotype": "metadata_csv", "treatment": "flag", "plate_id": "absent", ...})` writes the sidecar
- **THEN** `run_metadata["metadata_csv_path"]` is the resolved-absolute CSV path and `run_metadata["metadata_csv_sha256"]` is the SHA-256 of the CSV bytes
- **AND** `identity_source["genotype"] == "metadata_csv"`, `identity_source["treatment"] == "flag"`, and `identity_source["plate_id"] == "absent"`
- **AND** `identity_source` contains all six keys (`series`, `sample_uid`, `timepoint`, `plate_id`, `genotype`, `treatment`) and no `plant_id`/`track_id`
- **AND** a run with no `--metadata-csv` writes `metadata_csv_path == null`, `metadata_csv_sha256 == null`, and an `identity_source` with no `"metadata_csv"` values
- **AND** the top-level, per-plant, and per-genotype `run_metadata.json` carry identical `metadata_csv_path`, `metadata_csv_sha256`, and `identity_source`

#### Scenario: Constants snapshot reflects override
- **GIVEN** a custom `ConstantsT` override passed to the writer with `BAND_POWER_NOISE_RATIO = 4`
- **WHEN** the foundation writes the metadata
- **THEN** `_constants_snapshot["BAND_POWER_NOISE_RATIO"] == 4`
- **AND** the default-valued constants reflect the unmodified defaults

## ADDED Requirements

### Requirement: Series-to-CircumnutationInputs adapter
The system SHALL provide `sleap_roots.circumnutation.adapters.series_to_inputs(series, *, cadence_s, sample_uid, series_name=None, timepoint=None, plate_id=None, genotype=None, treatment=None, r_px=None, run_id=None) -> tuple[CircumnutationInputs, dict]`. It returns the constructed `CircumnutationInputs` AND an **identity-provenance dict** `{"metadata_csv_path": <resolved-absolute str | None>, "metadata_csv_sha256": <hex digest str | None>, "identity_source": {<field>: "flag" | "metadata_csv" | "default" | "absent"}}` so the caller can persist which source populated each row-identity field and verify the metadata join (consumed by the CLI's provenance, per Requirement: Run-metadata sidecar). This is the single bridge from `sleap_roots.series.Series` into the pure circumnutation core; it carries NO `click` dependency and SHALL be independently testable. It formalizes the mechanical transform of the `_load_plate001_inputs()` test blueprint (`tests/test_circumnutation_pipeline.py`), replacing that blueprint's hardcoded test literals (`genotype="Nipponbare"`, `treatment="none"`, `timepoint="T0"`) with the CSV/flag/NaN sourcing below.

The adapter SHALL:

1. Call `series.get_tracked_tips()` to obtain the per-frame `["track_id", "frame", "tip_x", "tip_y"]` long-format DataFrame.
2. Derive an integer `track_id` by stripping a **prefix-anchored** `"track_"` (e.g. `removeprefix("track_")` / a `^track_` match — NOT a global `str.replace`, which would corrupt an interior occurrence) and coercing to `int`. If any track name does not yield an integer after the prefix strip, it SHALL raise `ValueError` naming the offending track name(s) rather than allow a cryptic `astype(int)` error.
3. Set `plant_id = track_id` (the current track↔plant 1:1 convention).
4. Populate the 8 `ROW_IDENTITY_COLUMNS` (`series`, `sample_uid`, `timepoint`, `plate_id`, `plant_id`, `track_id`, `genotype`, `treatment`): `sample_uid` from the required argument; `series` from `series_name` (defaulting to `series.series_name`); and `timepoint`/`plate_id`/`genotype`/`treatment` via the metadata precedence below.
5. Construct `CircumnutationInputs(trajectory_df=df, cadence_s=cadence_s, R_px=r_px, run_id=run_id)`, letting that data class's validators raise `ValueError` for an empty `trajectory_df`, non-positive/non-finite `cadence_s`, etc., and return it together with the identity-provenance dict.

**Identity provenance.** The adapter SHALL record, for each of the six fields `series`/`sample_uid`/`timepoint`/`plate_id`/`genotype`/`treatment` (the map is total — all six always present), the literal source that won the value: `"flag"` when the explicit argument was non-`None`, `"metadata_csv"` when a non-null cell was resolved from the `--metadata-csv` via `Series.get_metadata`, `"default"` for the filename-stem default of `series` (a real derived value), or `"absent"` for the `NaN` fallback of `genotype`/`treatment`/`timepoint`/`plate_id`. `plant_id`/`track_id` are excluded (mechanically derived from track names, not flag/CSV-sourced). It SHALL also surface the resolved-absolute metadata-CSV path (or `None`) and the SHA-256 of the CSV bytes (or `None`). The CLI threads this dict into `gather_run_metadata` for the top-level, per-plant, and per-genotype writes (Requirement: Run-metadata sidecar).

**Metadata precedence — CSV-as-source, flags-as-override.** For each of `genotype`/`treatment`/`timepoint`, the adapter SHALL resolve a single value with a `pd.notna`-keyed rule, because `Series.get_metadata` returns `np.nan` indistinguishably for "no CSV", "missing column", and "empty cell": the per-field CSV value is `series.get_metadata(field)` when the `Series` carries a `csv_path` (the raw cell, NOT a coercing property such as `Series.timepoint`); the resolved value is the explicit flag argument when it is not `None`, otherwise the CSV value; and an INFO-level override log SHALL be emitted only when the flag is not `None` AND the CSV value is non-null (`pd.notna`) AND `str(flag) != str(csv_value)`. With neither flag nor a non-null CSV value, the field is `NaN`.

**`timepoint` dtype.** `timepoint` SHALL be an object/string label: the adapter reads the raw CSV cell via `series.get_metadata("timepoint")` (NOT the `Series.timepoint` property, which coerces to `float` and raises on a non-numeric value) and `str()`-normalizes both the flag and the CSV value so the column dtype is uniform.

**Malformed metadata CSV.** A metadata-CSV read failure (e.g. a non-parseable file surfacing a pandas error) SHALL be caught inside the adapter and re-raised as a clear `ValueError` rather than leaking a pandas traceback.

#### Scenario: Adapter builds CircumnutationInputs with the 8 identity columns
- **GIVEN** a tracked `Series` whose `get_tracked_tips()` yields ≥ 2 tracks named `"track_0"`, `"track_1"` with ≥ 64 frames each
- **WHEN** `series_to_inputs(series, cadence_s=300.0, sample_uid="plate_001", genotype="Nipponbare")` is invoked
- **THEN** it returns a 2-tuple `(inputs, identity_provenance)` where `inputs.trajectory_df` carries all 8 `ROW_IDENTITY_COLUMNS`
- **AND** `track_id` and `plant_id` are integer-valued (`0`, `1`) with `plant_id == track_id`
- **AND** `sample_uid == "plate_001"`, `series` defaults to `series.series_name`, and `genotype == "Nipponbare"`

#### Scenario: Adapter reports identity provenance per field
- **GIVEN** a `Series` with `csv_path` whose row has `genotype="Nipponbare"`, `treatment="MOCK"`, and an empty `plate_id`
- **WHEN** `series_to_inputs(series, cadence_s=300.0, sample_uid="plate_001", treatment="DROUGHT")` is invoked (genotype from CSV, treatment overridden by flag, plate_id absent)
- **THEN** `identity_provenance["metadata_csv_path"]` is the resolved-absolute CSV path and `identity_provenance["metadata_csv_sha256"]` is the SHA-256 of the CSV bytes
- **AND** `identity_provenance["identity_source"]` is total over the six fields and records `{"series": "default", "sample_uid": "flag", "timepoint": "absent", "plate_id": "absent", "genotype": "metadata_csv", "treatment": "flag"}` (and contains no `plant_id`/`track_id`)

#### Scenario: track_id prefix strip is anchored, not global
- **GIVEN** a `Series` with a track named `"track_track_1"` (an interior `"track_"` occurrence)
- **WHEN** `series_to_inputs(...)` derives the integer `track_id`
- **THEN** it raises `ValueError` naming the offending name — the prefix-anchored strip removes only the leading `"track_"`, leaving `"track_1"`, which is not a pure integer, so it fails loud
- **AND** a global `str.replace("track_", "")` (which would silently corrupt `"track_track_1"` to `1`) is NOT used — the anchored strip refuses to silently mangle a non-`track_<int>` name

#### Scenario: Non-integer track name raises a clear ValueError
- **GIVEN** a `Series` with a track named `"track_2a"` (or `"primary"`) that does not yield an integer after the prefix strip
- **WHEN** `series_to_inputs(...)` is invoked
- **THEN** it raises `ValueError` whose message names the offending track name(s)
- **AND** it does NOT surface a cryptic pandas `astype(int)` error

#### Scenario: Metadata precedence — flag overrides a real CSV value and logs at INFO
- **GIVEN** a `Series` loaded with `csv_path` pointing at a metadata CSV whose row has `genotype="Nipponbare"`, joined on `plant_qr_code == sample_uid`
- **WHEN** `series_to_inputs(series, cadence_s=300.0, sample_uid="plate_001", genotype="KitaakeX")` is invoked
- **THEN** the resolved `genotype` is `"KitaakeX"` (the flag wins)
- **AND** an INFO log records that `--genotype` overrode the metadata-csv value `"Nipponbare"`

#### Scenario: Metadata precedence — CSV value used when no flag; blank cell logs no spurious override
- **GIVEN** a `Series` with a metadata CSV whose `genotype` cell is populated but whose `treatment` cell is empty (NaN)
- **WHEN** `series_to_inputs(series, cadence_s=300.0, sample_uid="plate_001")` is invoked with no `genotype`/`treatment` flags
- **THEN** the resolved `genotype` is the CSV value and the resolved `treatment` is `NaN`
- **AND** no override INFO log is emitted for either field (no flag shadowed a real value)

#### Scenario: Neither CSV nor flag yields NaN
- **GIVEN** a `Series` with no `csv_path`
- **WHEN** `series_to_inputs(series, cadence_s=300.0, sample_uid="plate_001")` is invoked with no identity flags
- **THEN** `genotype`, `treatment`, `timepoint`, and `plate_id` are `NaN` in the resulting `trajectory_df`

#### Scenario: timepoint is string-normalized from a numeric CSV cell
- **GIVEN** a `Series` with a metadata CSV whose `timepoint` cell is the integer `0`
- **WHEN** `series_to_inputs(...)` resolves `timepoint` from the CSV
- **THEN** the resulting `timepoint` value is the string `"0"` (object dtype), read via the raw `get_metadata("timepoint")` rather than the coercing `Series.timepoint` property

#### Scenario: Malformed metadata CSV raises ValueError, not a pandas traceback
- **GIVEN** a `Series` whose `csv_path` points at a non-parseable file
- **WHEN** `series_to_inputs(...)` attempts to resolve identity from the CSV
- **THEN** it raises a clear `ValueError`
- **AND** it does NOT leak a raw pandas parser exception

### Requirement: Circumnutation analyze CLI
The system SHALL provide `sleap_roots.circumnutation.cli`, defining a `circumnutation` `click` group containing an `analyze` command, registered on the root `sleap_roots.cli:main` group via `main.add_command(circumnutation)` (mirroring the existing `main.add_command(viewer)`). `analyze` SHALL compose the full pipeline on one `.slp`: `Series.load` → `series_to_inputs` → `pipeline.compute_traits` → (assemble provenance via `gather_run_metadata`) → `write_per_plant_csv` (per-plant CSV) → (optionally) `aggregate_by_genotype` → `write_per_genotype_csv` → (optionally) `save_plots`.

**Options.** `analyze` SHALL accept: a positional `SLP_PATH` (`click.Path(exists=True)`); a **required** `--cadence-s` (`float`); a **required** `--sample-uid` (`str`); `--output-dir`/`-o` (default `./<series_name>_circumnutation/`); `--series-name` (default the `.slp` filename stem); `--metadata-csv` (`click.Path(exists=True)`, optional); `--timepoint`, `--plate-id`, `--genotype`, `--treatment` (`str`, optional identity overrides); `--r-px` (`float`, optional); `--run-id` (`str`, optional); `--no-plots` (flag); `--no-aggregate` (flag); and `-v`/`--verbose` (count). It SHALL NOT expose a `--px-per-mm` or any calibration option (CC-3).

**Output tree.** `analyze` SHALL write into distinct subdirectories of `--output-dir` so each `_io` writer's fixed-name `run_metadata.json` lives in its own directory (this sidesteps issue #238 without modifying the foundation `_io` contract; the CLI references #238 but does NOT close it):

```
<output-dir>/
├── run_metadata.json      # canonical run provenance (the plots sidecar's ../run_metadata.json target)
├── per_plant/             # per-plant CSV + .units.json + run_metadata.json (byte-identical copy)
├── per_genotype/          # (only when aggregation runs) per-genotype CSV + .units.json + run_metadata.json
└── plots/                 # (only when plotting runs) *.png + plots_metadata.json
```

The CLI SHALL also write the shared run-metadata to a **top-level `<output-dir>/run_metadata.json`** — `save_plots` writes `plots/plots_metadata.json` with a hardcoded `"run_metadata_ref": "../run_metadata.json"` (plotting.py:521) that resolves to `<output-dir>/run_metadata.json`; without this top-level file that pointer would dangle and plots would be unprovenanced. The top-level file is byte-identical to the `per_plant/`/`per_genotype/` copies (same shared dict). The CLI SHALL `mkdir(parents=True, exist_ok=True)` `--output-dir` and the `per_plant/` leaf always, and the `per_genotype/` leaf only on the aggregation path, BEFORE the writes (the `_io` writers require their parent dir to exist). Re-running on the same `.slp` overwrites prior outputs (documented behavior; the help text notes that distinct `--cadence-s`/`--r-px` runs should use distinct `-o` so provenance is not clobbered).

**Gated per-genotype aggregation.** Per-genotype aggregation SHALL be gated by `--no-aggregate`. When aggregation is on (default) and any plant's resolved `genotype` is `NaN`, `analyze` SHALL raise a `click.ClickException` (exit code 1) BEFORE writing any output, naming `--genotype` / `--metadata-csv` / `--no-aggregate`. When `--no-aggregate` is passed, `analyze` SHALL skip aggregation and the `per_genotype/` directory entirely and SHALL NOT require `genotype`.

**Headless plotting.** When plotting is enabled, `analyze` SHALL call `matplotlib.use("Agg", force=True)` BEFORE importing `sleap_roots.circumnutation.plotting`, then call `save_plots(inputs, out_dir=<output-dir>, enabled=True)`. When `--no-plots` is passed, `analyze` SHALL skip the matplotlib backend selection and the `plotting` import entirely, and the `plots/` directory SHALL NOT be created.

**Provenance.** `analyze` SHALL assemble the run-metadata **once** via a single `gather_run_metadata(input_path=Path(SLP_PATH).resolve(), run_id=inputs.run_id, constants=None, cadence_s=inputs.cadence_s, R_px=inputs.R_px, metadata_csv_path=<resolved-abs --metadata-csv or None>, metadata_csv_sha256=<sha256 of the CSV bytes or None>, identity_source=<adapter identity_source>)` and write that SAME dict to (1) a top-level `<output-dir>/run_metadata.json`, (2) `per_plant/run_metadata.json` (via `write_per_plant_csv`), and (3) on the aggregation path `per_genotype/run_metadata.json` (via `write_per_genotype_csv`) — all three byte-identical (same `timestamp`, `input_path`, `metadata_csv_path`, `metadata_csv_sha256`, `identity_source`). The CLI does NOT route the per-plant write through `CircumnutationPipeline.save()` (which would gather provenance internally and could not carry the CLI's identity provenance); it calls `write_per_plant_csv` directly with the shared run-metadata. (This supersedes the earlier "keep `save()`" design note: a single shared snapshot is what makes the identity provenance complete and the sidecars identical.) The resolved-absolute `.slp` path is recorded as `input_path` (a relative path is meaningless in `run_metadata.json`). The `metadata_csv_sha256` records the SHA-256 of the metadata-CSV bytes so the identity join is verifiable/reproducibility-checkable even though the CSV is an external mutable input recorded by reference (unlike `cadence_s`/`R_px`, the CSV *contents* are not snapshotted by value; the hash detects post-run drift). **Plots provenance:** `plots/plots_metadata.json` carries its own pixel-native plot constants, the per-plant `_IDENTITY_5_TUPLE` (`series`/`sample_uid`/`plate_id`/`plant_id`/`track_id`), and a `run_metadata_ref` pointer to the top-level `run_metadata.json` (resolved by writing that file); it does NOT yet embed `input_path`/`identity_source`/`genotype`/`treatment` directly — a dedicated follow-up issue (NOT #241) will thread those into `save_plots`.

**Error contract.** `click`'s built-in validation SHALL surface missing required options and a nonexistent `SLP_PATH` as exit code 2. `analyze` SHALL wrap the pipeline body in a single `try/except (ValueError, FileNotFoundError)` that re-raises as `click.ClickException(str(e))` (exit code 1, clean one-line message, no traceback); the adapter normalizes its own raisers (non-integer track name, malformed metadata CSV) to `ValueError`. `analyze` SHALL NOT use a broad `except Exception` — genuinely unanticipated errors surface as tracebacks.

**Logging (CC-9).** `-v`/`--verbose` SHALL set the log level: `0 → WARNING`, `1 → INFO`, `≥2 → DEBUG`, configured to write to stderr. User-facing result summaries SHALL go to stdout via `click.echo`.

**Help-text documentation.** The `analyze` `--help`/docstring SHALL document, beyond the CC-3 pure-pixel note: (a) the image pixel coordinate convention (origin top-left, y increases downward — the SLEAP convention `get_tracked_tips` emits), since the sign of direction-bearing traits depends on it; and (b) that the `timepoint` column is a free-form string label (so a downstream join against a numeric metadata table must cast).

#### Scenario: analyze runs the full pipeline and writes the output tree
- **GIVEN** a tracked `.slp` (≥ 2 tracks, ≥ 64 frames) and an empty `tmp_path`
- **WHEN** `analyze <slp> --cadence-s 300 --sample-uid plate_001 --genotype WT -o <tmp_path>` is invoked via `click.testing.CliRunner`
- **THEN** the command exits 0
- **AND** a top-level `<tmp_path>/run_metadata.json` exists (the plots-sidecar `run_metadata_ref` target)
- **AND** `<tmp_path>/per_plant/` contains the per-plant CSV, `.units.json`, and `run_metadata.json`
- **AND** `<tmp_path>/per_genotype/` contains the per-genotype CSV, `.units.json`, and `run_metadata.json`
- **AND** `<tmp_path>/plots/` contains ≥ 1 PNG, and `plots/plots_metadata.json`'s `run_metadata_ref` (`../run_metadata.json`) resolves to the top-level file
- **AND** the per-plant CSV row count equals the number of tracks

#### Scenario: missing required options exit 2
- **WHEN** `analyze <slp>` is invoked without `--cadence-s` (or without `--sample-uid`) via `CliRunner`
- **THEN** the command exits with code 2
- **AND** the output names the missing option

#### Scenario: nonexistent SLP_PATH exits 2
- **WHEN** `analyze does_not_exist.slp --cadence-s 300 --sample-uid x` is invoked via `CliRunner`
- **THEN** the command exits with code 2 (click `Path(exists=True)` validation)

#### Scenario: genotype unresolved with aggregation on is a hard error
- **GIVEN** a tracked `.slp`
- **WHEN** `analyze <slp> --cadence-s 300 --sample-uid plate_001 -o <tmp_path>` is invoked (no `--genotype`, no `--metadata-csv`, no `--no-aggregate`)
- **THEN** the command exits with code 1 via `click.ClickException`
- **AND** the message names `--genotype` / `--metadata-csv` / `--no-aggregate`
- **AND** no output tree is written (the error is raised before any file is created)

#### Scenario: --no-aggregate runs per-plant + plots without genotype
- **GIVEN** a tracked `.slp`
- **WHEN** `analyze <slp> --cadence-s 300 --sample-uid plate_001 --no-aggregate -o <tmp_path>` is invoked (no genotype)
- **THEN** the command exits 0
- **AND** `<tmp_path>/per_plant/` and `<tmp_path>/plots/` exist
- **AND** `<tmp_path>/per_genotype/` does NOT exist

#### Scenario: --no-plots omits the plots directory
- **GIVEN** a tracked `.slp`
- **WHEN** `analyze <slp> --cadence-s 300 --sample-uid plate_001 --genotype WT --no-plots -o <tmp_path>` is invoked
- **THEN** the command exits 0
- **AND** `<tmp_path>/plots/` does NOT exist

#### Scenario: non-positive cadence is a clean ClickException (exit 1)
- **WHEN** `analyze <slp> --cadence-s 0 --sample-uid plate_001 --genotype WT` (or a negative cadence) is invoked via `CliRunner`
- **THEN** the command exits with code 1 (the `CircumnutationInputs` `ValueError` re-raised as `ClickException`)
- **AND** a clean one-line message is printed, no Python traceback

#### Scenario: non-numeric cadence is a click usage error (exit 2)
- **WHEN** `analyze <slp> --cadence-s abc --sample-uid plate_001 --genotype WT` is invoked via `CliRunner`
- **THEN** the command exits with code 2 (click `float` type parse failure, before the pipeline body)
- **AND** no Python traceback is printed

#### Scenario: metadata CSV populates identity; flag override is logged
- **GIVEN** a tracked `.slp` and a metadata CSV with `plant_qr_code=plate_001, genotype=Nipponbare, treatment=MOCK`
- **WHEN** `analyze <slp> --cadence-s 300 --sample-uid plate_001 --metadata-csv <csv> -o <tmp_path>` is invoked
- **THEN** the per-plant CSV rows carry `genotype=Nipponbare`, `treatment=MOCK`
- **AND** adding `--genotype KitaakeX` instead yields `genotype=KitaakeX` in the output with an INFO log recording the override

#### Scenario: run_metadata records resolved-absolute input path and identity provenance
- **GIVEN** a successful default (aggregating) run with `--metadata-csv <csv>` (no `--r-px`)
- **WHEN** the top-level `run_metadata.json`, `per_plant/run_metadata.json`, and `per_genotype/run_metadata.json` are read
- **THEN** all three record `input_path` equal to the resolved-absolute `.slp` path
- **AND** all three record `metadata_csv_path` (resolved-absolute CSV path), `metadata_csv_sha256` (SHA-256 of the CSV bytes), and the same `identity_source` mapping
- **AND** all three are byte-identical on `cadence_s`, `run_id`, `timestamp`, and `R_px` (which is `null` when `--r-px` is omitted — identical `null`, not a `None`-vs-`NaN` mismatch)

#### Scenario: run without metadata CSV records null path/hash and no metadata_csv source
- **GIVEN** a successful run with `--genotype WT` and NO `--metadata-csv`
- **WHEN** the `run_metadata.json` files are read
- **THEN** `metadata_csv_path == null` and `metadata_csv_sha256 == null`
- **AND** `identity_source` is total over the six fields with no `"metadata_csv"` values (`genotype == "flag"`, the unsupplied ones `"absent"`, `series` `"default"` or `"flag"`)

#### Scenario: re-running overwrites prior outputs
- **GIVEN** a completed `analyze` run that populated `<tmp_path>`
- **WHEN** `analyze` is invoked a second time on the same `.slp` with the same `-o <tmp_path>`
- **THEN** the second command exits 0
- **AND** the outputs reflect the second run (overwritten in place, no `FileExistsError`, no appended rows)

#### Scenario: default output-dir derives from series_name
- **GIVEN** a tracked `.slp` named `plate_xyz.slp` and no `-o`
- **WHEN** `analyze plate_xyz.slp --cadence-s 300 --sample-uid plate_001 --genotype WT --no-plots` is invoked from an empty working directory
- **THEN** the command exits 0
- **AND** the outputs are written under `./plate_xyz_circumnutation/`

#### Scenario: verbose flag sets log level on stderr; results on stdout
- **GIVEN** a successful run invoked with `click.testing.CliRunner(mix_stderr=False)`
- **WHEN** `analyze ... -v` (INFO) versus no `-v` (WARNING) versus `-vv` (DEBUG) is invoked
- **THEN** INFO/DEBUG log lines appear on stderr only at the corresponding verbosity
- **AND** the final result summary appears on stdout (not stderr)

#### Scenario: no calibration option (CC-3)
- **WHEN** `analyze --help` is inspected via `CliRunner`
- **THEN** there is no `--px-per-mm` (or any calibration) option
- **AND** the help text states outputs are pixel-native and points to `sleap_roots.circumnutation.units.convert_to_mm` for mm conversion
- **AND** the help text documents the image pixel coordinate convention (origin top-left, y-down) and that `timepoint` is a string label
