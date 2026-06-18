## MODIFIED Requirements

### Requirement: Package layout
The system SHALL provide a `sleap_roots.circumnutation` sub-package whose import-tree is complete from this PR onward — every module name referenced anywhere in `docs/circumnutation/roadmap.md` exists as an importable Python module from PR #1 forward, even if its functions raise `NotImplementedError` until later PRs land. The package SHALL contain:

- 7 contract modules: `__init__.py`, `_constants.py`, `_types.py`, `_io.py`, `units.py`, `_noise.py`, `_geometry.py`
- 11 implementation modules: `kinematics` (implemented from PR #2 onward; see Requirement: Tier 0 raw kinematic traits), `qc` (implemented from PR #3 onward; see Requirement: QC tier per-track quality traits), `synthetic` (implemented from PR #4 onward; see Requirement: Synthetic trajectory generator), `temporal_cwt` (implemented from PR #5 onward; see Requirement: Temporal CWT scaleogram API and Requirement: Temporal CWT ridge API), `nutation` (implemented from PR #6 onward; see Requirement: Tier 1 nutation trait emission API), `psi_g` (implemented from PR #7 onward; see Requirement: Tier 2 ψ_g trait emission API), `midline` (implemented from PR #8 onward; see Requirement: Tier 3a midline reconstruction API), `spatial_cwt` (implemented from PR #9 onward; see Requirement: Tier 3b spatial curvature resample API, Requirement: Tier 3b spatial CWT scaleogram API, and Requirement: Tier 3b spatial CWT ridge API), `traveling_wave` (implemented from PR #10 onward; see Requirement: Tier 3c traveling-wave trait emission API), `pipeline` (implemented from PR #14 onward; see Requirement: Circumnutation pipeline composition API), and `aggregation` (implemented from PR #15 onward; see Requirement: Per-genotype aggregation API)
- 2 stub modules: `parametric`, `plotting`

Each stub module SHALL define exactly one canonical callable per the table below, raising `NotImplementedError(f"PR #{N} — see docs/circumnutation/roadmap.md")` with the documented PR number. Each stub callable SHALL also have a complete Google-style docstring (Args, Returns, Raises) so `pydocstyle --convention=google` and the mkdocstrings auto-generated reference pages remain informative. Stubs whose tier PR will compose with the typed `ConstantsT` override-bag SHALL include `constants=None` as a forward-compatible keyword parameter so callers do not get `TypeError` before `NotImplementedError`.

| Stub module | Canonical callable | PR # |
|---|---|---|
| `parametric` | `compute(tier3_df, R_px, omega, Delta_phi)` | 11 |
| `plotting` | `scaleogram(scaleogram_result, out_path)` | 16 |

The `kinematics` module SHALL be importable on the same terms as the other modules (clean import, namespaced logger) and SHALL expose `compute(trajectory_df, constants=None)` per Requirement: Tier 0 raw kinematic traits. The `qc` module SHALL be importable on the same terms and SHALL expose `compute(trajectory_df, constants=None)` per Requirement: QC tier per-track quality traits. The `synthetic` module SHALL be importable on the same terms and SHALL expose `generate_trajectory(...)` per Requirement: Synthetic trajectory generator. The `temporal_cwt` module SHALL be importable on the same terms and SHALL expose `compute_scaleogram(x, cadence_s, constants=None) -> ScaleogramResult` and `extract_ridge(scaleogram_result, constants=None) -> RidgeResult` per Requirement: Temporal CWT scaleogram API and Requirement: Temporal CWT ridge API, AND SHALL ALSO expose `smooth_ridge(ridge_result, window=None, constants=None) -> RidgeResult` per Requirement: Temporal CWT ridge-continuity smoothing API. The `nutation` module SHALL be importable on the same terms and SHALL expose `compute(trajectory_df, cadence_s, coordinate="lateral", constants=None) -> pd.DataFrame` per Requirement: Tier 1 nutation trait emission API. The `psi_g` module SHALL be importable on the same terms and SHALL expose `compute(trajectory_df, cadence_s, constants=None) -> pd.DataFrame` per Requirement: Tier 2 ψ_g trait emission API. The `midline` module SHALL be importable on the same terms and SHALL expose `reconstruct(x, y, cadence_s, sg_window=None, constants=None) -> MidlineResult` per Requirement: Tier 3a midline reconstruction API. The `spatial_cwt` module SHALL be importable on the same terms and SHALL expose `compute_scaleogram(kappa, ds, constants=None) -> SpatialScaleogramResult` and `extract_ridge(scaleogram_result, constants=None) -> SpatialRidgeResult` per Requirement: Tier 3b spatial CWT scaleogram API and Requirement: Tier 3b spatial CWT ridge API, AND SHALL ALSO expose `resample_curvature(curvature_px_inv, arc_length_px, velocity_sub_noise_mask=None, constants=None) -> ResampleResult` per Requirement: Tier 3b spatial curvature resample API. The `traveling_wave` module SHALL be importable on the same terms and SHALL expose `compute(trajectory_df, cadence_s, constants=None, *, tier0_df=None, tier1_df=None) -> pd.DataFrame` per Requirement: Tier 3c traveling-wave trait emission API. The `pipeline` module SHALL be importable on the same terms and SHALL expose `compute_traits(inputs, constants=None) -> tuple[pd.DataFrame, pd.DataFrame, dict]` and the `CircumnutationPipeline` class per Requirement: Circumnutation pipeline composition API. The `aggregation` module SHALL be importable on the same terms and SHALL expose `aggregate_by_genotype(per_plant_df, units) -> tuple[pd.DataFrame, dict]` per Requirement: Per-genotype aggregation API. Unlike the stub modules, calling `kinematics.compute`, `qc.compute`, `synthetic.generate_trajectory`, `temporal_cwt.compute_scaleogram`, `nutation.compute`, `psi_g.compute`, `midline.reconstruct`, `spatial_cwt.resample_curvature`, `spatial_cwt.compute_scaleogram`, `spatial_cwt.extract_ridge`, `traveling_wave.compute`, `pipeline.compute_traits`, or `aggregation.aggregate_by_genotype` with a valid input SHALL NOT raise `NotImplementedError`.

The `_noise` and `_geometry` modules are private (underscore-prefixed) shared internals; their canonical callables are documented under Requirement: Tier 0 helper modules (and, for `_geometry.compute_signed_area`, under Requirement: Tier 2 ψ_g trait emission API; and, for `_noise.compute_sg_derivative` / `_geometry.compute_path_curvature`, under Requirement: Tier 3a midline reconstruction API).

**Scope note on PR #6 addition-vs-transition.** The `nutation` module is NEWLY created in PR #6 — it was never a stub module in PR #1–#5, and therefore does not appear in the stub-callable table. The implementation-module count grows from 4 (PR #5 baseline: kinematics, qc, synthetic, temporal_cwt) to 5 by ADDITION of `nutation`, not by transition from a prior stub. This was the first PR in the program to grow the implementation set without shrinking the stub set.

**Scope note on PR #7 stub-to-implementation transition.** The `psi_g` module was a stub in PR #1–#6 (it appeared in the stub-callable table with canonical callable `compute_psi_g(x, y, constants=None)`). PR #7 graduated it to an implementation module: the implementation-module count grew from 5 to 6 AND the stub-module count shrank from 6 to 5. The canonical callable was RENAMED `compute_psi_g` → `compute`.

**Scope note on PR #8 stub-to-implementation transition.** The `midline` module was a stub in PR #1–#7 (it appeared in the stub-callable table with canonical callable `reconstruct(x, y, cadence_s, constants=None)`). PR #8 graduated it to an implementation module: the implementation-module count grew from 6 to 7 AND the stub-module count shrank from 5 to 4 (the same stub→impl shape as PR #7). The canonical callable KEPT its name `reconstruct` (no rename); the implementation signature ADDED a `sg_window=None` parameter (`reconstruct(x, y, cadence_s, constants=None)` → `reconstruct(x, y, cadence_s, sg_window=None, constants=None)`), locked by Requirement: Tier 3a midline reconstruction API.

**Scope note on PR #9 stub-to-implementation transition.** The `spatial_cwt` module IS a stub in PR #1–#8 (it appeared in the stub-callable table with canonical callable `compute_scaleogram(kappa, ds, constants=None)`, PR #9). PR #9 graduates it to an implementation module: the implementation-module count grows from 7 to **8** AND the stub-module count shrinks from 4 to **3** (the same stub→impl shape as PR #7/#8). The canonical callable KEEPS its name `compute_scaleogram` (no rename); the implementation signature is EXACTLY the stub-table signature `compute_scaleogram(kappa, ds, constants=None)` — the speculative `wavelet=`/`scale_range=` keyword parameters present in the PR #1 stub file are DROPPED (the wavelet and scale range are derived from `constants`, mirroring `temporal_cwt.compute_scaleogram`'s `(x, cadence_s, constants=None)` precedent). `spatial_cwt` is therefore removed from the stub-callable table and from the "remaining stub" enumeration, and gains callability scenarios below. PR #9 ADDS two further public symbols not previously in the stub table — `resample_curvature` (the non-uniform→uniform κ(s) resample entry helper) and `extract_ridge` (the spatial ridge) — whose callability contracts are locked here for symmetry with `compute_scaleogram`, mirroring how PR #5's `extract_ridge` and PR #6's `smooth_ridge` were locked. **PR #9 descopes `L_gz`/`L_c` growth-zone-structure detection** (the §7.4 |κ|-envelope-peak premise does not transfer to top-view tip-trail κ(s); see the Tier 3b requirements and follow-up issue #230); `spatial_cwt` therefore exposes no `detect_growth_zone` symbol.

**Scope note on PR #10 addition-vs-transition.** The `traveling_wave` module is NEWLY created in PR #10 — it was never a stub module in PR #1–#9 (only `parametric`, `plotting`, `pipeline` remain stubs at PR #10), and therefore does not appear in the stub-callable table. The implementation-module count grows from 8 to **9** by ADDITION of `traveling_wave`, not by transition from a prior stub (the same addition shape as PR #6's `nutation`); the stub-module count is UNCHANGED at 3. The canonical callable is `compute(trajectory_df, cadence_s, constants=None) -> pd.DataFrame` (the Tier 1/Tier 2 trait-emission signature; `cadence_s` is an explicit positional parameter). **PR #10 ships reduced scope** per the PR #9 `L_gz`/`L_c` descope (#230): it emits only the 3 λ-based traits + diagnostics on the full reconstructed trail (no growth-zone mask); the 5 `L_gz`/`L_c`-dependent traits and the mask remain blocked on #230 and are OMITTED from the output schema (not reserved as NaN columns). `traveling_wave` gains a callability scenario below.

**Scope note on PR #14 stub-to-implementation transition.** The `pipeline` module WAS a stub in PR #1–#10 (it appeared in the stub-callable table with canonical callable `compute_traits(inputs, constants=None)`, PR #14). PR #14 graduates it to an implementation module: the implementation-module count grows from 9 to **10** AND the stub-module count shrinks from 3 to **2** (the same stub→impl shape as PR #7/#8/#9). The canonical callable KEEPS its name `compute_traits` (no rename); the implementation adds keyword-only file-writing via a separate `CircumnutationPipeline.save(...)` method and a picklable `CircumnutationPipeline` class, locked by Requirement: Circumnutation pipeline composition API. `pipeline` is therefore removed from the stub-callable table and from the "remaining stub" enumeration, and gains a callability scenario below. **`pipeline` was the last stub whose canonical callable carried the forward-compatible `constants=None` parameter** — the 2 remaining stubs (`parametric`, `plotting`) do not, so the prior "Stubs accept `constants=None` where the table prescribes it" scenario is REMOVED (no remaining subject). **Deviation note:** the `pipeline` stub docstring and the roadmap row described PR #14 as building "a TraitDef DAG matching the `Pipeline` base class pattern". PR #14 instead implements a **sequential merge-orchestrator** (the circumnutation tiers are per-track `DataFrame → DataFrame` functions, not the per-frame networkx `TraitDef` nodes of `sleap_roots.trait_pipelines.Pipeline`); the documented tier-order dependency structure is preserved without the per-frame node model. The stub docstring + roadmap row are corrected to match.

**Scope note on PR #15 addition-vs-transition.** The `aggregation` module is NEWLY created in PR #15 — it was never a stub module in PR #1–#14 (only `parametric` and `plotting` remain stubs at PR #15), and therefore does not appear in the stub-callable table. The implementation-module count grows from 10 to **11** by ADDITION of `aggregation`, not by transition from a prior stub (the same addition shape as PR #6's `nutation` and PR #10's `traveling_wave`); the stub-module count is UNCHANGED at 2. The canonical callable is `aggregate_by_genotype(per_plant_df, units) -> tuple[pd.DataFrame, dict]` (a post-pipeline aggregation that consumes the PR #14 composed per-plant frame). `aggregation` gains a callability scenario below.

#### Scenario: All stub modules import cleanly
- **WHEN** a user runs `import sleap_roots.circumnutation as c; import sleap_roots.circumnutation.kinematics, sleap_roots.circumnutation.qc, sleap_roots.circumnutation.synthetic, sleap_roots.circumnutation.temporal_cwt, sleap_roots.circumnutation.nutation, sleap_roots.circumnutation.psi_g, sleap_roots.circumnutation.midline, sleap_roots.circumnutation.spatial_cwt, sleap_roots.circumnutation.traveling_wave, sleap_roots.circumnutation.parametric, sleap_roots.circumnutation.plotting, sleap_roots.circumnutation.pipeline, sleap_roots.circumnutation.aggregation`
- **THEN** every import succeeds
- **AND** no exception is raised at module-import time

#### Scenario: Calling each remaining stub raises NotImplementedError with the correct PR number
- **GIVEN** the package imported as in the previous scenario
- **WHEN** the canonical callable in each of the 2 remaining stub modules (`parametric`, `plotting`) is invoked (parameters per the table above; `NotImplementedError` fires before any argument check)
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
