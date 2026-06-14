## MODIFIED Requirements

### Requirement: Package layout
The system SHALL provide a `sleap_roots.circumnutation` sub-package whose import-tree is complete from this PR onward — every module name referenced anywhere in `docs/circumnutation/roadmap.md` exists as an importable Python module from PR #1 forward, even if its functions raise `NotImplementedError` until later PRs land. The package SHALL contain:

- 7 contract modules: `__init__.py`, `_constants.py`, `_types.py`, `_io.py`, `units.py`, `_noise.py`, `_geometry.py`
- 10 implementation modules: `kinematics` (implemented from PR #2 onward; see Requirement: Tier 0 raw kinematic traits), `qc` (implemented from PR #3 onward; see Requirement: QC tier per-track quality traits), `synthetic` (implemented from PR #4 onward; see Requirement: Synthetic trajectory generator), `temporal_cwt` (implemented from PR #5 onward; see Requirement: Temporal CWT scaleogram API and Requirement: Temporal CWT ridge API), `nutation` (implemented from PR #6 onward; see Requirement: Tier 1 nutation trait emission API), `psi_g` (implemented from PR #7 onward; see Requirement: Tier 2 ψ_g trait emission API), `midline` (implemented from PR #8 onward; see Requirement: Tier 3a midline reconstruction API), `spatial_cwt` (implemented from PR #9 onward; see Requirement: Tier 3b spatial curvature resample API, Requirement: Tier 3b spatial CWT scaleogram API, and Requirement: Tier 3b spatial CWT ridge API), `traveling_wave` (implemented from PR #10 onward; see Requirement: Tier 3c traveling-wave trait emission API), and `pipeline` (implemented from PR #14 onward; see Requirement: Circumnutation pipeline composition API)
- 2 stub modules: `parametric`, `plotting`

Each stub module SHALL define exactly one canonical callable per the table below, raising `NotImplementedError(f"PR #{N} — see docs/circumnutation/roadmap.md")` with the documented PR number. Each stub callable SHALL also have a complete Google-style docstring (Args, Returns, Raises) so `pydocstyle --convention=google` and the mkdocstrings auto-generated reference pages remain informative. Stubs whose tier PR will compose with the typed `ConstantsT` override-bag SHALL include `constants=None` as a forward-compatible keyword parameter so callers do not get `TypeError` before `NotImplementedError`.

| Stub module | Canonical callable | PR # |
|---|---|---|
| `parametric` | `compute(tier3_df, R_px, omega, Delta_phi)` | 11 |
| `plotting` | `scaleogram(scaleogram_result, out_path)` | 16 |

The `kinematics` module SHALL be importable on the same terms as the other modules (clean import, namespaced logger) and SHALL expose `compute(trajectory_df, constants=None)` per Requirement: Tier 0 raw kinematic traits. The `qc` module SHALL be importable on the same terms and SHALL expose `compute(trajectory_df, constants=None)` per Requirement: QC tier per-track quality traits. The `synthetic` module SHALL be importable on the same terms and SHALL expose `generate_trajectory(...)` per Requirement: Synthetic trajectory generator. The `temporal_cwt` module SHALL be importable on the same terms and SHALL expose `compute_scaleogram(x, cadence_s, constants=None) -> ScaleogramResult` and `extract_ridge(scaleogram_result, constants=None) -> RidgeResult` per Requirement: Temporal CWT scaleogram API and Requirement: Temporal CWT ridge API, AND SHALL ALSO expose `smooth_ridge(ridge_result, window=None, constants=None) -> RidgeResult` per Requirement: Temporal CWT ridge-continuity smoothing API. The `nutation` module SHALL be importable on the same terms and SHALL expose `compute(trajectory_df, cadence_s, coordinate="lateral", constants=None) -> pd.DataFrame` per Requirement: Tier 1 nutation trait emission API. The `psi_g` module SHALL be importable on the same terms and SHALL expose `compute(trajectory_df, cadence_s, constants=None) -> pd.DataFrame` per Requirement: Tier 2 ψ_g trait emission API. The `midline` module SHALL be importable on the same terms and SHALL expose `reconstruct(x, y, cadence_s, sg_window=None, constants=None) -> MidlineResult` per Requirement: Tier 3a midline reconstruction API. The `spatial_cwt` module SHALL be importable on the same terms and SHALL expose `compute_scaleogram(kappa, ds, constants=None) -> SpatialScaleogramResult` and `extract_ridge(scaleogram_result, constants=None) -> SpatialRidgeResult` per Requirement: Tier 3b spatial CWT scaleogram API and Requirement: Tier 3b spatial CWT ridge API, AND SHALL ALSO expose `resample_curvature(curvature_px_inv, arc_length_px, velocity_sub_noise_mask=None, constants=None) -> ResampleResult` per Requirement: Tier 3b spatial curvature resample API. The `traveling_wave` module SHALL be importable on the same terms and SHALL expose `compute(trajectory_df, cadence_s, constants=None, *, tier0_df=None, tier1_df=None) -> pd.DataFrame` per Requirement: Tier 3c traveling-wave trait emission API. The `pipeline` module SHALL be importable on the same terms and SHALL expose `compute_traits(inputs, constants=None) -> tuple[pd.DataFrame, pd.DataFrame, dict]` and the `CircumnutationPipeline` class per Requirement: Circumnutation pipeline composition API. Unlike the stub modules, calling `kinematics.compute`, `qc.compute`, `synthetic.generate_trajectory`, `temporal_cwt.compute_scaleogram`, `nutation.compute`, `psi_g.compute`, `midline.reconstruct`, `spatial_cwt.resample_curvature`, `spatial_cwt.compute_scaleogram`, `spatial_cwt.extract_ridge`, `traveling_wave.compute`, or `pipeline.compute_traits` with a valid input SHALL NOT raise `NotImplementedError`.

The `_noise` and `_geometry` modules are private (underscore-prefixed) shared internals; their canonical callables are documented under Requirement: Tier 0 helper modules (and, for `_geometry.compute_signed_area`, under Requirement: Tier 2 ψ_g trait emission API; and, for `_noise.compute_sg_derivative` / `_geometry.compute_path_curvature`, under Requirement: Tier 3a midline reconstruction API).

**Scope note on PR #6 addition-vs-transition.** The `nutation` module is NEWLY created in PR #6 — it was never a stub module in PR #1–#5, and therefore does not appear in the stub-callable table. The implementation-module count grows from 4 (PR #5 baseline: kinematics, qc, synthetic, temporal_cwt) to 5 by ADDITION of `nutation`, not by transition from a prior stub. This was the first PR in the program to grow the implementation set without shrinking the stub set.

**Scope note on PR #7 stub-to-implementation transition.** The `psi_g` module was a stub in PR #1–#6 (it appeared in the stub-callable table with canonical callable `compute_psi_g(x, y, constants=None)`). PR #7 graduated it to an implementation module: the implementation-module count grew from 5 to 6 AND the stub-module count shrank from 6 to 5. The canonical callable was RENAMED `compute_psi_g` → `compute`.

**Scope note on PR #8 stub-to-implementation transition.** The `midline` module was a stub in PR #1–#7 (it appeared in the stub-callable table with canonical callable `reconstruct(x, y, cadence_s, constants=None)`). PR #8 graduated it to an implementation module: the implementation-module count grew from 6 to 7 AND the stub-module count shrank from 5 to 4 (the same stub→impl shape as PR #7). The canonical callable KEPT its name `reconstruct` (no rename); the implementation signature ADDED a `sg_window=None` parameter (`reconstruct(x, y, cadence_s, constants=None)` → `reconstruct(x, y, cadence_s, sg_window=None, constants=None)`), locked by Requirement: Tier 3a midline reconstruction API.

**Scope note on PR #9 stub-to-implementation transition.** The `spatial_cwt` module IS a stub in PR #1–#8 (it appeared in the stub-callable table with canonical callable `compute_scaleogram(kappa, ds, constants=None)`, PR #9). PR #9 graduates it to an implementation module: the implementation-module count grows from 7 to **8** AND the stub-module count shrinks from 4 to **3** (the same stub→impl shape as PR #7/#8). The canonical callable KEEPS its name `compute_scaleogram` (no rename); the implementation signature is EXACTLY the stub-table signature `compute_scaleogram(kappa, ds, constants=None)` — the speculative `wavelet=`/`scale_range=` keyword parameters present in the PR #1 stub file are DROPPED (the wavelet and scale range are derived from `constants`, mirroring `temporal_cwt.compute_scaleogram`'s `(x, cadence_s, constants=None)` precedent). `spatial_cwt` is therefore removed from the stub-callable table and from the "remaining stub" enumeration, and gains callability scenarios below. PR #9 ADDS two further public symbols not previously in the stub table — `resample_curvature` (the non-uniform→uniform κ(s) resample entry helper) and `extract_ridge` (the spatial ridge) — whose callability contracts are locked here for symmetry with `compute_scaleogram`, mirroring how PR #5's `extract_ridge` and PR #6's `smooth_ridge` were locked. **PR #9 descopes `L_gz`/`L_c` growth-zone-structure detection** (the §7.4 |κ|-envelope-peak premise does not transfer to top-view tip-trail κ(s); see the Tier 3b requirements and follow-up issue #230); `spatial_cwt` therefore exposes no `detect_growth_zone` symbol.

**Scope note on PR #10 addition-vs-transition.** The `traveling_wave` module is NEWLY created in PR #10 — it was never a stub module in PR #1–#9 (only `parametric`, `plotting`, `pipeline` remain stubs at PR #10), and therefore does not appear in the stub-callable table. The implementation-module count grows from 8 to **9** by ADDITION of `traveling_wave`, not by transition from a prior stub (the same addition shape as PR #6's `nutation`); the stub-module count is UNCHANGED at 3. The canonical callable is `compute(trajectory_df, cadence_s, constants=None) -> pd.DataFrame` (the Tier 1/Tier 2 trait-emission signature; `cadence_s` is an explicit positional parameter). **PR #10 ships reduced scope** per the PR #9 `L_gz`/`L_c` descope (#230): it emits only the 3 λ-based traits + diagnostics on the full reconstructed trail (no growth-zone mask); the 5 `L_gz`/`L_c`-dependent traits and the mask remain blocked on #230 and are OMITTED from the output schema (not reserved as NaN columns). `traveling_wave` gains a callability scenario below.

**Scope note on PR #14 stub-to-implementation transition.** The `pipeline` module WAS a stub in PR #1–#10 (it appeared in the stub-callable table with canonical callable `compute_traits(inputs, constants=None)`, PR #14). PR #14 graduates it to an implementation module: the implementation-module count grows from 9 to **10** AND the stub-module count shrinks from 3 to **2** (the same stub→impl shape as PR #7/#8/#9). The canonical callable KEEPS its name `compute_traits` (no rename); the implementation adds keyword-only file-writing via a separate `CircumnutationPipeline.save(...)` method and a picklable `CircumnutationPipeline` class, locked by Requirement: Circumnutation pipeline composition API. `pipeline` is therefore removed from the stub-callable table and from the "remaining stub" enumeration, and gains a callability scenario below. **`pipeline` was the last stub whose canonical callable carried the forward-compatible `constants=None` parameter** — the 2 remaining stubs (`parametric`, `plotting`) do not, so the prior "Stubs accept `constants=None` where the table prescribes it" scenario is REMOVED (no remaining subject). **Deviation note:** the `pipeline` stub docstring and the roadmap row described PR #14 as building "a TraitDef DAG matching the `Pipeline` base class pattern". PR #14 instead implements a **sequential merge-orchestrator** (the circumnutation tiers are per-track `DataFrame → DataFrame` functions, not the per-frame networkx `TraitDef` nodes of `sleap_roots.trait_pipelines.Pipeline`); the documented tier-order dependency structure is preserved without the per-frame node model. The stub docstring + roadmap row are corrected to match.

#### Scenario: All stub modules import cleanly
- **WHEN** a user runs `import sleap_roots.circumnutation as c; import sleap_roots.circumnutation.kinematics, sleap_roots.circumnutation.qc, sleap_roots.circumnutation.synthetic, sleap_roots.circumnutation.temporal_cwt, sleap_roots.circumnutation.nutation, sleap_roots.circumnutation.psi_g, sleap_roots.circumnutation.midline, sleap_roots.circumnutation.spatial_cwt, sleap_roots.circumnutation.traveling_wave, sleap_roots.circumnutation.parametric, sleap_roots.circumnutation.plotting, sleap_roots.circumnutation.pipeline`
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

#### Scenario: `synthetic.generate_trajectory` has no `px_per_mm` parameter
- **WHEN** `inspect.signature(sleap_roots.circumnutation.synthetic.generate_trajectory)` is inspected
- **THEN** the parameter list does not contain `px_per_mm`
- **AND** the docstring confirms the generator emits pure-pixel trajectories (callers compose `convert_to_mm()` if they want mm output)

#### Scenario: `import sleap_roots` succeeds without raising
- **WHEN** a user runs `import sleap_roots`
- **THEN** no exception is raised
- **AND** `sleap_roots.CircumnutationInputs` is accessible
- **AND** `sleap_roots.convert_to_mm` is accessible

### Requirement: Tier 3c traveling-wave trait emission API
The system SHALL provide `sleap_roots.circumnutation.traveling_wave.compute(trajectory_df: pd.DataFrame, cadence_s: float, constants: Optional[ConstantsT] = None, *, tier0_df: Optional[pd.DataFrame] = None, tier1_df: Optional[pd.DataFrame] = None) -> pd.DataFrame`. The function SHALL accept the documented signature with `cadence_s` as an explicit positional parameter (mirroring `nutation.compute` / `psi_g.compute`) and `tier0_df` / `tier1_df` as keyword-only optional precomputed-frames parameters (the PR #14 dedup fast path; default `None`). It SHALL be the first consumer of the PR #9 spatial-CWT machinery and SHALL compute the Tier 3c traveling-wave validation traits that test the QPB steady-traveling-wave hypothesis `λ_spatial = v · T_nutation` (theory.md §4.7). The `coordinate` projection is NOT exposed as a parameter: the function SHALL internally use `coordinate="lateral"` for its Tier 1 recompute (the QPB residual is only defined against the lateral nutation period, CC-7).

The function SHALL emit a per-track DataFrame whose rows correspond to the unique 5-tuple `(series, sample_uid, plate_id, plant_id, track_id)` derived from `trajectory_df` via `groupby(_IDENTITY_5_TUPLE, dropna=False, sort=False)`. The returned DataFrame SHALL contain the 8 row-identity columns (per Requirement: Trait CSV row-identity schema) followed by exactly 6 trait columns, all `float64`, in this declared order: `lambda_spatial_median_px`, `lambda_spatial_variation`, `traveling_wave_residual`, `lambda_expected_px`, `lambda_spatial_mad_px`, `coi_valid_fraction`.

The function SHALL be self-contained by default: when `tier0_df` and `tier1_df` are both `None` it SHALL recompute Tier 0 via `kinematics.compute(trajectory_df, constants=constants)` and Tier 1 via `nutation.compute(trajectory_df, cadence_s, coordinate="lateral", constants=constants)` (passing the resolved `constants` into both), and SHALL join their `v_total_median_px_per_frame` and `T_nutation_median` operands onto the per-track results **by merging on the full `_IDENTITY_5_TUPLE`** (NOT on `track_id` alone — `track_id` is not unique across plates/samples). The Tier 1 `is_nutating` flag is consumed **implicitly** via `T_nutation_median` — Tier 1 NaN-gates `T_nutation_median` to NaN when `is_nutating == False`, so a non-nutating track NaNs `lambda_expected_px`/`traveling_wave_residual` without a separate `is_nutating` join. Because `kinematics.compute`/`nutation.compute` return their identity keys coerced to int64 while the raw `trajectory_df` keys may be float64, the function SHALL coerce the per-track frame's `track_id`/`plant_id` to int64 before the operand merge (mirroring the per-plant template-merge guard); a key-dtype mismatch would otherwise produce a SILENT all-NaN operand join (not a `KeyError`). This default recompute is redundant-by-design when called inside a full pipeline (Tier 0/Tier 1 are also computed as emitting tiers).

**PR #14 dedup fast path.** When `tier0_df` and `tier1_df` are BOTH provided, the function SHALL use them as the Tier 0 / Tier 1 operand source and SHALL NOT recompute `kinematics.compute` / `nutation.compute`. The function SHALL validate that each provided frame carries the `_IDENTITY_5_TUPLE` columns plus the single operand column it supplies (`v_total_median_px_per_frame` for `tier0_df`, `T_nutation_median` for `tier1_df`) and SHALL raise `ValueError` naming the missing column otherwise (cheaper than a recompute; prevents a silent all-NaN operand join). Supplying exactly ONE of `tier0_df` / `tier1_df` (the other `None`) SHALL raise `ValueError` (both-or-neither — a half-deduped call would silently recompute the other operand). The fast path SHALL produce Tier 3c trait columns IDENTICAL (`atol=0`) to the default recompute path for the same `(trajectory_df, cadence_s, constants)` — deduplication changes performance, not results — provided the supplied frames are the Tier 0 / Tier 1 (`coordinate="lateral"`) outputs for that same `trajectory_df`. When `tier0_df` and `tier1_df` are omitted (the standalone default), the function's behavior SHALL be byte-identical to its pre-PR-#14 recompute behavior.

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

## ADDED Requirements

### Requirement: Circumnutation pipeline composition API
The system SHALL provide `sleap_roots.circumnutation.pipeline.compute_traits(inputs: CircumnutationInputs, constants: Optional[ConstantsT] = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]` and a picklable `CircumnutationPipeline` class that composes the five trait-emitting tiers (Tier 0 `kinematics`, QC `qc`, Tier 1 `nutation`, Tier 2 `psi_g`, Tier 3c `traveling_wave`) into one per-plant trait table, units mapping, and provenance-bearing CSV. The composition SHALL be a **sequential merge-orchestrator**, not the per-frame networkx `TraitDef` DAG of `sleap_roots.trait_pipelines.Pipeline`: the circumnutation tiers are per-track `DataFrame → DataFrame` functions, so the pipeline calls each tier's `compute()` once and merges their per-plant outputs on `_IDENTITY_5_TUPLE`. The documented tier-dependency structure (Tier 0 / QC / Tier 1 / Tier 2 independent; Tier 3c depends on Tier 0 + Tier 1) SHALL be honored as a fixed call order; the pipeline introduces no new physical constant and `_CONSTANTS_VERSION` SHALL remain 6.

`CircumnutationPipeline` SHALL be an `attrs` class whose only field is `constants` (an `Optional[ConstantsT]`, default `None`), making instances picklable. It SHALL expose `compute_traits(self, inputs) -> tuple[pd.DataFrame, pd.DataFrame, dict]` (the pure composition) and `save(self, out_path, per_plant_df, units, *, input_path, run_id=None)` (the I/O step). The module-level `compute_traits(inputs, constants=None)` SHALL be a thin wrapper equal to `CircumnutationPipeline(constants=constants).compute_traits(inputs)`.

`compute_traits` SHALL be **pure** (no filesystem I/O): it SHALL read `inputs.trajectory_df` and `inputs.cadence_s`, call `kinematics.compute(df, constants)`, `qc.compute(df, constants)`, `nutation.compute(df, cadence_s, constants=constants)` (relying on `nutation.compute`'s default `coordinate="lateral"` — the pipeline SHALL NOT override `coordinate`, because the dedup `atol=0` equivalence requires the Tier 1 frame passed to `traveling_wave` to be the lateral one `traveling_wave` would otherwise recompute), `psi_g.compute(df, cadence_s, constants=constants)`, and `traveling_wave.compute(df, cadence_s, constants, tier0_df=<tier0 result>, tier1_df=<tier1 result>)` (the PR #14 dedup fast path — Tier 0 and Tier 1 are computed once and reused, NOT recomputed inside `traveling_wave`). The SAME `constants` object SHALL be passed to all five tiers AND to the `traveling_wave` fast path (a divergent `constants` would silently break the `atol=0` dedup equivalence). `cadence_s` SHALL be threaded only to the three cadence-consuming tiers (nutation, psi_g, traveling_wave); Tier 0 and QC take none. `inputs.R_px` SHALL NOT be passed to any of the five tiers (they are pure-pixel, CC-3); it is provenance-only. The pipeline SHALL add no redundant input validation of its own (the `CircumnutationInputs` `attrs` validators and each tier's internal validation already cover it).

`compute_traits` SHALL return the 3-tuple `(per_plant_df, trajectory_df, units_dict)` where `trajectory_df` is the input frame (echoed for provenance) and:

- `per_plant_df` SHALL have one row per unique `_IDENTITY_5_TUPLE` and exactly **46 columns** in fixed tier order: the 8 `ROW_IDENTITY_COLUMNS`, then the 10 Tier 0 trait columns (`_TIER0_TRAIT_COLUMNS`), then the QC trait columns EXCLUDING `growth_axis_unreliable` (10 columns), then the 8 Tier 1 columns (`_NUTATION_TRAIT_COLUMNS`), then the 4 Tier 2 columns (`_PSIG_TRAIT_COLUMNS`), then the 6 Tier 3c columns (`_TRAVELING_WAVE_TRAIT_COLUMNS`). Each tier's emitted dtypes SHALL be preserved through the merge (e.g. bool `is_nutating`, string `qc_failure_reason`); no global float coercion is applied. The merge SHALL be `how="left"` onto the shared per-plant template (`_io._build_per_plant_template_from_df`) with the int64 coercion-with-raise guard on `track_id`/`plant_id` (a key-dtype mismatch raises rather than producing a silent all-NaN row).
- `growth_axis_unreliable` is emitted by BOTH Tier 0 and QC (QC recomputes it with the same formula/inputs; an existing cross-tier equality test guarantees element-wise equality). Both are `bool` dtype. The pipeline SHALL keep exactly ONE `growth_axis_unreliable` column — Tier 0's, in the Tier 0 block — and SHALL drop QC's copy before merging (so no `_x`/`_y` suffix collision occurs). The pipeline SHALL assert the two source columns are equal via `Series.equals` (dtype + value, so a future nullable-boolean drift cannot pass a naive `==` check that treats `NaN != NaN`) before dropping QC's copy, raising `ValueError` naming the divergence if they differ.

- The composed schema deliberately OMITS the 5 `L_gz`/`L_c`-dependent Tier 3c traits (`L_gz_estimate`, `L_c_estimate`, `B_balance_number`, `L_gz_steady_state_residual`, `L_gz_resolvable`) and the growth-zone mask — they are blocked on #230 and were never emitted by `traveling_wave` (PR #10 reduced scope), so they are absent from the 46-column frame, not reserved as NaN.
- `units_dict` SHALL map every one of the 46 columns to a unit string in `PIPELINE_UNIT_VOCABULARY`, assembled as `ROW_IDENTITY_UNITS` (restricted to the identity columns) merged with the five per-tier `_*_TRAIT_UNITS` maps. `growth_axis_unreliable` SHALL appear exactly once.

To assemble `units_dict`, the `nutation` and `psi_g` modules SHALL each declare a `_*_TRAIT_UNITS` mapping (mirroring `_TIER0_TRAIT_UNITS` / `_QC_TRAIT_UNITS` / `_TRAVELING_WAVE_TRAIT_UNITS`): `nutation._NUTATION_TRAIT_UNITS` SHALL have one entry per `_NUTATION_TRAIT_COLUMNS` (8) and `psi_g._PSIG_TRAIT_UNITS` one entry per `_PSIG_TRAIT_COLUMNS` (4); every value SHALL be a member of `PIPELINE_UNIT_VOCABULARY`. These additions are additive module constants (no behavior change to the tiers). The unit strings are pinned below (semantic-correctness matters: the sidecar writer validates vocabulary membership but NOT semantic correctness, so an in-vocabulary-but-wrong string would silently mislabel a published column):

- `_NUTATION_TRAIT_UNITS`: `T_nutation_median` → `"s"`, `T_nutation_iqr` → `"s"`, `A_nutation_envelope_max_px` → `"px"`, `band_power_ratio` → `"—"`, `noise_floor_estimate` → `"px"` (it is a median FFT **amplitude** of the lateral px signal — NOT a dimensionless ratio), `is_nutating` → `"bool"`, `period_residual_vs_derr_reference` → `"—"`, `cadence_nyquist_ratio` → `"—"`.
- `_PSIG_TRAIT_UNITS`: `T_psig_median_s` → `"s"`, `delta_E_amplitude_proxy_px_per_frame` → `"px/frame"`, `handedness` → `"int"` (an integer sign in {−1, 0, +1}; matches the program's type-token convention for non-float columns, e.g. `track_id` → `"int"`, `is_nutating` → `"bool"`), `helix_signed_area_px2` → `"px²"` (the superscript-² glyph, the `PIPELINE_UNIT_VOCABULARY` token — a literal ASCII `"px2"` is NOT in vocabulary and would fail the writer).

The `"s"` (seconds) period units (`T_nutation_median`, `T_nutation_iqr`, `T_psig_median_s`) arrive **pre-converted** from the tiers: the frames→seconds conversion is performed once inside the temporal CWT machinery (`temporal_cwt._log_spaced_scales` derives `periods_s` via `pywt.scale2frequency(...) / cadence_s`), so the period traits are already in seconds when the pipeline receives them. The pipeline performs **no unit conversion** of its own — it threads `cadence_s` to the cadence-consuming tiers and labels each column with the unit the tier emits. Note the deliberate program convention asymmetry: **periods are in seconds** while **rates are per-frame** (`v_*_px_per_frame`, `delta_E_amplitude_proxy_px_per_frame` → `"px/frame"`, NOT px/s); the two are reconciled only inside `traveling_wave` (`T_frames = T_nutation_median / cadence_s`, then `lambda_expected_px = v · T_frames`, keeping λ pure-pixel).

**Scope note — GitHub issue #222.** PR #14 adds ONLY the two units maps the pipeline needs; it does NOT perform the broader #222 program-wide suffix-convention work (renaming `T_nutation_median` → `T_nutation_median_s` / `T_nutation_iqr` → `T_nutation_iqr_s`, documenting the rule, and the foundation suffix-gate), which remains #222's own job. The two maps are therefore keyed on the CURRENT (unsuffixed) `nutation` column names; when the #222 rename lands, the `_NUTATION_TRAIT_UNITS` keys for those two columns will be re-keyed accordingly.

`save` SHALL write the per-plant CSV plus its two sibling sidecars by delegating to `_io.gather_run_metadata(input_path, run_id, constants)` and `_io.write_per_plant_csv(out_path, per_plant_df, units, run_metadata)` (which writes the CSV, the `<stem>.units.json` units sidecar, and the `run_metadata.json` provenance sidecar per Requirement: Units sidecar JSON and Requirement: Run-metadata sidecar). The `input_path` (source `.slp` path) SHALL be supplied to `save` by the caller, since `CircumnutationInputs` does not carry it.

`compute_traits` SHALL be deterministic per CC-6: same `inputs` → bit-identical float trait columns across calls in the same process (`atol=0`), deferring to the per-tier determinism contracts (no separate composed-output canary is introduced). `CircumnutationPipeline` instances SHALL be picklable (`pickle.loads(pickle.dumps(pipeline))` round-trips) and the unpickled instance SHALL compute an identical `per_plant_df`.

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

#### Scenario: save writes the CSV and two sidecars and round-trips
- **GIVEN** a `(per_plant_df, trajectory_df, units_dict)` from `compute_traits(inputs)`, a temporary `out_path`, and an `input_path` (a `.slp` path string)
- **WHEN** `CircumnutationPipeline().save(out_path, per_plant_df, units_dict, input_path=input_path, run_id="r1")` is invoked
- **THEN** the CSV at `out_path`, a sibling `<stem>.units.json`, and a sibling `run_metadata.json` all exist
- **AND** `_io.read_per_plant_csv(out_path)` recovers the DataFrame, the units mapping, and the run-metadata
- **AND** the run-metadata contains `input_path`, the git SHA, the sleap_roots/sleap_io/numpy/scipy/pandas/python version fields, `platform`, an ISO-8601 `timestamp`, `_schema_version`, `_constants_version` (equal to 6), and `_constants_snapshot`

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
- **THEN** the float trait columns of the two `per_plant_df` results are bit-identical at `atol=0`

#### Scenario: compute_traits round-trips the real plate-001 fixture through the full pipeline
- **GIVEN** the 6 tracks of `tests/data/circumnutation_nipponbare_plate_001/plate_001_greyscale.tracked_proofread.slp` loaded via `Series.load(...).get_tracked_tips()` (with `track_id` coerced to int and the 8 row-identity columns attached with plate-001 metadata) wrapped in a `CircumnutationInputs(cadence_s=300.0)`
- **WHEN** `pipeline.compute_traits(inputs)` is invoked and the result is written via `save` to a temporary directory and read back
- **THEN** `per_plant_df` has exactly 6 rows and the 46-column composed schema in declared order
- **AND** columns from all five tiers are present (e.g. `v_total_median_px_per_frame` (Tier 0), `track_is_clean` (QC), `T_nutation_median` (Tier 1), `handedness` (Tier 2), `traveling_wave_residual` (Tier 3c))
- **AND** all 6 tracks have finite `traveling_wave_residual < 0.30` (the QPB band, matching Requirement: Tier 3c traveling-wave trait emission API)
- **AND** the single coalesced `growth_axis_unreliable` column equals what `qc.compute` emits for the same tracks (cross-tier equality at the composed level)
- **AND** the test is skipped when the Git-LFS proofread fixture is absent
