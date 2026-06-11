## MODIFIED Requirements

### Requirement: Package layout
The system SHALL provide a `sleap_roots.circumnutation` sub-package whose import-tree is complete from this PR onward — every module name referenced anywhere in `docs/circumnutation/roadmap.md` exists as an importable Python module from PR #1 forward, even if its functions raise `NotImplementedError` until later PRs land. The package SHALL contain:

- 7 contract modules: `__init__.py`, `_constants.py`, `_types.py`, `_io.py`, `units.py`, `_noise.py`, `_geometry.py`
- 9 implementation modules: `kinematics` (implemented from PR #2 onward; see Requirement: Tier 0 raw kinematic traits), `qc` (implemented from PR #3 onward; see Requirement: QC tier per-track quality traits), `synthetic` (implemented from PR #4 onward; see Requirement: Synthetic trajectory generator), `temporal_cwt` (implemented from PR #5 onward; see Requirement: Temporal CWT scaleogram API and Requirement: Temporal CWT ridge API), `nutation` (implemented from PR #6 onward; see Requirement: Tier 1 nutation trait emission API), `psi_g` (implemented from PR #7 onward; see Requirement: Tier 2 ψ_g trait emission API), `midline` (implemented from PR #8 onward; see Requirement: Tier 3a midline reconstruction API), `spatial_cwt` (implemented from PR #9 onward; see Requirement: Tier 3b spatial curvature resample API, Requirement: Tier 3b spatial CWT scaleogram API, and Requirement: Tier 3b spatial CWT ridge API), and `traveling_wave` (implemented from PR #10 onward; see Requirement: Tier 3c traveling-wave trait emission API)
- 3 stub modules: `parametric`, `plotting`, `pipeline`

Each stub module SHALL define exactly one canonical callable per the table below, raising `NotImplementedError(f"PR #{N} — see docs/circumnutation/roadmap.md")` with the documented PR number. Each stub callable SHALL also have a complete Google-style docstring (Args, Returns, Raises) so `pydocstyle --convention=google` and the mkdocstrings auto-generated reference pages remain informative. Stubs whose tier PR will compose with the typed `ConstantsT` override-bag SHALL include `constants=None` as a forward-compatible keyword parameter so callers do not get `TypeError` before `NotImplementedError`.

| Stub module | Canonical callable | PR # |
|---|---|---|
| `parametric` | `compute(tier3_df, R_px, omega, Delta_phi)` | 11 |
| `plotting` | `scaleogram(scaleogram_result, out_path)` | 16 |
| `pipeline` | `compute_traits(inputs, constants=None)` | 14 |

The `kinematics` module SHALL be importable on the same terms as the other modules (clean import, namespaced logger) and SHALL expose `compute(trajectory_df, constants=None)` per Requirement: Tier 0 raw kinematic traits. The `qc` module SHALL be importable on the same terms and SHALL expose `compute(trajectory_df, constants=None)` per Requirement: QC tier per-track quality traits. The `synthetic` module SHALL be importable on the same terms and SHALL expose `generate_trajectory(...)` per Requirement: Synthetic trajectory generator. The `temporal_cwt` module SHALL be importable on the same terms and SHALL expose `compute_scaleogram(x, cadence_s, constants=None) -> ScaleogramResult` and `extract_ridge(scaleogram_result, constants=None) -> RidgeResult` per Requirement: Temporal CWT scaleogram API and Requirement: Temporal CWT ridge API, AND SHALL ALSO expose `smooth_ridge(ridge_result, window=None, constants=None) -> RidgeResult` per Requirement: Temporal CWT ridge-continuity smoothing API. The `nutation` module SHALL be importable on the same terms and SHALL expose `compute(trajectory_df, cadence_s, coordinate="lateral", constants=None) -> pd.DataFrame` per Requirement: Tier 1 nutation trait emission API. The `psi_g` module SHALL be importable on the same terms and SHALL expose `compute(trajectory_df, cadence_s, constants=None) -> pd.DataFrame` per Requirement: Tier 2 ψ_g trait emission API. The `midline` module SHALL be importable on the same terms and SHALL expose `reconstruct(x, y, cadence_s, sg_window=None, constants=None) -> MidlineResult` per Requirement: Tier 3a midline reconstruction API. The `spatial_cwt` module SHALL be importable on the same terms and SHALL expose `compute_scaleogram(kappa, ds, constants=None) -> SpatialScaleogramResult` and `extract_ridge(scaleogram_result, constants=None) -> SpatialRidgeResult` per Requirement: Tier 3b spatial CWT scaleogram API and Requirement: Tier 3b spatial CWT ridge API, AND SHALL ALSO expose `resample_curvature(curvature_px_inv, arc_length_px, velocity_sub_noise_mask=None, constants=None) -> ResampleResult` per Requirement: Tier 3b spatial curvature resample API. The `traveling_wave` module SHALL be importable on the same terms and SHALL expose `compute(trajectory_df, cadence_s, constants=None) -> pd.DataFrame` per Requirement: Tier 3c traveling-wave trait emission API. Unlike the stub modules, calling `kinematics.compute`, `qc.compute`, `synthetic.generate_trajectory`, `temporal_cwt.compute_scaleogram`, `nutation.compute`, `psi_g.compute`, `midline.reconstruct`, `spatial_cwt.resample_curvature`, `spatial_cwt.compute_scaleogram`, `spatial_cwt.extract_ridge`, or `traveling_wave.compute` with a valid input SHALL NOT raise `NotImplementedError`.

The `_noise` and `_geometry` modules are private (underscore-prefixed) shared internals; their canonical callables are documented under Requirement: Tier 0 helper modules (and, for `_geometry.compute_signed_area`, under Requirement: Tier 2 ψ_g trait emission API; and, for `_noise.compute_sg_derivative` / `_geometry.compute_path_curvature`, under Requirement: Tier 3a midline reconstruction API).

**Scope note on PR #6 addition-vs-transition.** The `nutation` module is NEWLY created in PR #6 — it was never a stub module in PR #1–#5, and therefore does not appear in the stub-callable table. The implementation-module count grows from 4 (PR #5 baseline: kinematics, qc, synthetic, temporal_cwt) to 5 by ADDITION of `nutation`, not by transition from a prior stub. This was the first PR in the program to grow the implementation set without shrinking the stub set.

**Scope note on PR #7 stub-to-implementation transition.** The `psi_g` module was a stub in PR #1–#6 (it appeared in the stub-callable table with canonical callable `compute_psi_g(x, y, constants=None)`). PR #7 graduated it to an implementation module: the implementation-module count grew from 5 to 6 AND the stub-module count shrank from 6 to 5. The canonical callable was RENAMED `compute_psi_g` → `compute`.

**Scope note on PR #8 stub-to-implementation transition.** The `midline` module was a stub in PR #1–#7 (it appeared in the stub-callable table with canonical callable `reconstruct(x, y, cadence_s, constants=None)`). PR #8 graduated it to an implementation module: the implementation-module count grew from 6 to 7 AND the stub-module count shrank from 5 to 4 (the same stub→impl shape as PR #7). The canonical callable KEPT its name `reconstruct` (no rename); the implementation signature ADDED a `sg_window=None` parameter (`reconstruct(x, y, cadence_s, constants=None)` → `reconstruct(x, y, cadence_s, sg_window=None, constants=None)`), locked by Requirement: Tier 3a midline reconstruction API.

**Scope note on PR #9 stub-to-implementation transition.** The `spatial_cwt` module IS a stub in PR #1–#8 (it appeared in the stub-callable table with canonical callable `compute_scaleogram(kappa, ds, constants=None)`, PR #9). PR #9 graduates it to an implementation module: the implementation-module count grows from 7 to **8** AND the stub-module count shrinks from 4 to **3** (the same stub→impl shape as PR #7/#8). The canonical callable KEEPS its name `compute_scaleogram` (no rename); the implementation signature is EXACTLY the stub-table signature `compute_scaleogram(kappa, ds, constants=None)` — the speculative `wavelet=`/`scale_range=` keyword parameters present in the PR #1 stub file are DROPPED (the wavelet and scale range are derived from `constants`, mirroring `temporal_cwt.compute_scaleogram`'s `(x, cadence_s, constants=None)` precedent). `spatial_cwt` is therefore removed from the stub-callable table and from the "remaining stub" enumeration, and gains callability scenarios below. PR #9 ADDS two further public symbols not previously in the stub table — `resample_curvature` (the non-uniform→uniform κ(s) resample entry helper) and `extract_ridge` (the spatial ridge) — whose callability contracts are locked here for symmetry with `compute_scaleogram`, mirroring how PR #5's `extract_ridge` and PR #6's `smooth_ridge` were locked. **PR #9 descopes `L_gz`/`L_c` growth-zone-structure detection** (the §7.4 |κ|-envelope-peak premise does not transfer to top-view tip-trail κ(s); see the Tier 3b requirements and follow-up issue #230); `spatial_cwt` therefore exposes no `detect_growth_zone` symbol.

**Scope note on PR #10 addition-vs-transition.** The `traveling_wave` module is NEWLY created in PR #10 — it was never a stub module in PR #1–#9 (only `parametric`, `plotting`, `pipeline` remain stubs), and therefore does not appear in the stub-callable table. The implementation-module count grows from 8 to **9** by ADDITION of `traveling_wave`, not by transition from a prior stub (the same addition shape as PR #6's `nutation`); the stub-module count is UNCHANGED at 3. The canonical callable is `compute(trajectory_df, cadence_s, constants=None) -> pd.DataFrame` (the Tier 1/Tier 2 trait-emission signature; `cadence_s` is an explicit positional parameter). **PR #10 ships reduced scope** per the PR #9 `L_gz`/`L_c` descope (#230): it emits only the 3 λ-based traits + diagnostics on the full reconstructed trail (no growth-zone mask); the 5 `L_gz`/`L_c`-dependent traits and the mask remain blocked on #230 and are OMITTED from the output schema (not reserved as NaN columns). `traveling_wave` gains a callability scenario below.

#### Scenario: All stub modules import cleanly
- **WHEN** a user runs `import sleap_roots.circumnutation as c; import sleap_roots.circumnutation.kinematics, sleap_roots.circumnutation.qc, sleap_roots.circumnutation.synthetic, sleap_roots.circumnutation.temporal_cwt, sleap_roots.circumnutation.nutation, sleap_roots.circumnutation.psi_g, sleap_roots.circumnutation.midline, sleap_roots.circumnutation.spatial_cwt, sleap_roots.circumnutation.traveling_wave, sleap_roots.circumnutation.parametric, sleap_roots.circumnutation.plotting, sleap_roots.circumnutation.pipeline`
- **THEN** every import succeeds
- **AND** no exception is raised at module-import time

#### Scenario: Calling each remaining stub raises NotImplementedError with the correct PR number
- **GIVEN** the package imported as in the previous scenario
- **WHEN** the canonical callable in each of the 3 remaining stub modules (`parametric`, `plotting`, `pipeline`) is invoked (parameters per the table above; `NotImplementedError` fires before any argument check)
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

#### Scenario: Stubs accept `constants=None` where the table prescribes it
- **GIVEN** the stubs listed in the table above whose canonical callable includes `constants=None`
- **WHEN** a caller invokes that stub with `constants=...` keyword argument (any value)
- **THEN** `NotImplementedError` is raised (not `TypeError`)

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

### Requirement: Tier 3c traveling-wave trait emission API
The system SHALL provide `sleap_roots.circumnutation.traveling_wave.compute(trajectory_df: pd.DataFrame, cadence_s: float, constants: Optional[ConstantsT] = None) -> pd.DataFrame`. The function SHALL accept the documented signature with `cadence_s` as an explicit positional parameter (mirroring `nutation.compute` / `psi_g.compute`). It SHALL be the first consumer of the PR #9 spatial-CWT machinery and SHALL compute the Tier 3c traveling-wave validation traits that test the QPB steady-traveling-wave hypothesis `λ_spatial = v · T_nutation` (theory.md §4.7). The `coordinate` projection is NOT exposed as a parameter: the function SHALL internally use `coordinate="lateral"` for its Tier 1 recompute (the QPB residual is only defined against the lateral nutation period, CC-7).

The function SHALL emit a per-track DataFrame whose rows correspond to the unique 5-tuple `(series, sample_uid, plate_id, plant_id, track_id)` derived from `trajectory_df` via `groupby(_IDENTITY_5_TUPLE, dropna=False, sort=False)`. The returned DataFrame SHALL contain the 8 row-identity columns (per Requirement: Trait CSV row-identity schema) followed by exactly 6 trait columns, all `float64`, in this declared order: `lambda_spatial_median_px`, `lambda_spatial_variation`, `traveling_wave_residual`, `lambda_expected_px`, `lambda_spatial_mad_px`, `coi_valid_fraction`.

The function SHALL be self-contained: it SHALL recompute Tier 0 via `kinematics.compute(trajectory_df, constants=constants)` and Tier 1 via `nutation.compute(trajectory_df, cadence_s, coordinate="lateral", constants=constants)` (passing the resolved `constants` into both), and SHALL join their `v_total_median_px_per_frame`, `T_nutation_median`, and `is_nutating` operands onto the per-track results **by merging on the full `_IDENTITY_5_TUPLE`** (NOT on `track_id` alone — `track_id` is not unique across plates/samples), applying the same int64 coercion the per-plant template merge uses. This recompute is redundant-by-design (the PR #14 pipeline DAG SHALL dedup Tier 0/Tier 1 rather than calling this function naively).

The function SHALL compose the per-track spatial chain `midline.reconstruct` → `spatial_cwt.resample_curvature` → `spatial_cwt.compute_scaleogram` → `spatial_cwt.extract_ridge`, then gate and calibrate the ridge wavelengths:

- **Spatial-availability / COI gate.** The 5 spatial traits (`lambda_spatial_median_px`, `lambda_spatial_variation`, `traveling_wave_residual`, `lambda_expected_px`, `lambda_spatial_mad_px`) SHALL be NaN for a track when the spatial chain is unavailable: `MidlineResult.is_degenerate`, OR `ResampleResult.is_degenerate`, OR `compute_scaleogram`/`extract_ridge` raised `ValueError` (the function SHALL catch it and emit a NaN row, NOT crash), OR the cone-of-influence fraction exceeds the reuse of the existing `COI_FRACTION_MAX` constant — i.e. `coi_valid_fraction < (1 − COI_FRACTION_MAX)`, where `coi_valid_fraction = (~in_coi).sum() / in_coi.size`. `coi_valid_fraction` SHALL be finite whenever a ridge formed (including when the low-COI gate fires) and NaN only when no ridge formed (degenerate midline/resample or caught CWT raise).
- **cgau2 calibration.** `lambda_spatial_median_px`, `lambda_spatial_variation`, and `lambda_spatial_mad_px` SHALL be computed from a single calibrated wavelength array in true pixels, obtained by dividing each COI-valid ridge `wavelengths_px` by the cgau2 over-report ratio interpolated from `tests/data/circumnutation_spatial_cwt_calibration.json` (the strictly-increasing `λ_reported` axis of the `n=400` slice → well-posed `np.interp`). The honest `traveling_wave_residual` SHALL use this calibrated λ (true px on both sides); the function SHALL NOT emit the raw mixed-domain residual.
- **Trait definitions.** `lambda_spatial_median_px = median(λ_cal[interior])`; `lambda_spatial_mad_px = median(|λ_cal[interior] − median|)`; `lambda_spatial_variation = lambda_spatial_mad_px / lambda_spatial_median_px` (an orientation-invariant robust spread, 0 = uniform); `lambda_expected_px = v_total_median_px_per_frame · (T_nutation_median / cadence_s)`; `traveling_wave_residual = |lambda_spatial_median_px − lambda_expected_px| / lambda_expected_px`.

The function SHALL gate `traveling_wave_residual` and `lambda_expected_px` to NaN when the temporal/velocity operands are undefined: `T_nutation_median` is NaN (which occurs when `is_nutating == False`, per Requirement: Tier 1 nutation trait emission API), OR `v_total_median_px_per_frame` is non-finite, OR `lambda_expected_px ≤ 0`. The division SHALL be guarded so no `inf`/`np.RuntimeWarning` is produced. The pure-spatial traits `lambda_spatial_median_px` and `lambda_spatial_variation` SHALL remain valid (not gated by `is_nutating`) whenever the spatial chain succeeded.

The function SHALL emit only these 6 columns; the `L_gz`/`L_c`-dependent traits (`L_gz_estimate`, `L_c_estimate`, `B_balance_number`, `L_gz_steady_state_residual`, `L_gz_resolvable`) and the growth-zone mask SHALL NOT be emitted (blocked on #230; omitted, not reserved as NaN columns). The pipeline SHALL remain pure-pixel (CC-3): the `_px` columns are pixels and the unit-bearing columns SHALL use only `PIPELINE_UNIT_VOCABULARY` entries (`px`, `—`); no new constant is introduced and `_CONSTANTS_VERSION` SHALL remain 6.

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
- **THEN** the call returns one row per unique 5-tuple without raising (no `KeyError` from a dtype mismatch)
- **AND** each row's `lambda_expected_px` is computed from the `v_total_median_px_per_frame` and `T_nutation_median` of THAT row's own 5-tuple (not another plate's track with the same `track_id`)

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
- **AND** a captured canary on a fixed synthetic input matches the hardcoded expected values to within the measured `atol` (target `1e-6`) across Ubuntu / Windows / macOS CI runners AT THE TIME OF PR-MERGE; canary values are regression-detection sentinels and MAY be re-captured (in a follow-up commit cross-referencing this scenario) if upstream BLAS / scipy / pywt / numpy semantics legitimately shift after merge

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
- **GIVEN** a synthetic trajectory built so the spatial wavelength is known a priori — small lateral amplitude relative to growth drift (`amplitude_px ≪ growth_rate_px_per_frame · (T_nutation_s / cadence_s)`) so the trail arc-length-per-period ≈ `growth_rate_px_per_frame · (T_nutation_s / cadence_s)`
- **WHEN** `traveling_wave.compute(trajectory_df, cadence_s=...)` is invoked
- **THEN** `lambda_spatial_median_px` recovers the a-priori wavelength within the cgau2 calibration tolerance
- **AND** because the synthetic's spatial wavelength equals `v · T_frames` by construction, `traveling_wave_residual` is small (the synthetic cannot produce a large/QPB-violating residual; the plate-001 scenario covers the non-trivial-residual case)
