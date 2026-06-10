## MODIFIED Requirements

### Requirement: Package layout
The system SHALL provide a `sleap_roots.circumnutation` sub-package whose import-tree is complete from this PR onward — every module name referenced anywhere in `docs/circumnutation/roadmap.md` exists as an importable Python module from PR #1 forward, even if its functions raise `NotImplementedError` until later PRs land. The package SHALL contain:

- 7 contract modules: `__init__.py`, `_constants.py`, `_types.py`, `_io.py`, `units.py`, `_noise.py`, `_geometry.py`
- 8 implementation modules: `kinematics` (implemented from PR #2 onward; see Requirement: Tier 0 raw kinematic traits), `qc` (implemented from PR #3 onward; see Requirement: QC tier per-track quality traits), `synthetic` (implemented from PR #4 onward; see Requirement: Synthetic trajectory generator), `temporal_cwt` (implemented from PR #5 onward; see Requirement: Temporal CWT scaleogram API and Requirement: Temporal CWT ridge API), `nutation` (implemented from PR #6 onward; see Requirement: Tier 1 nutation trait emission API), `psi_g` (implemented from PR #7 onward; see Requirement: Tier 2 ψ_g trait emission API), `midline` (implemented from PR #8 onward; see Requirement: Tier 3a midline reconstruction API), and `spatial_cwt` (implemented from PR #9 onward; see Requirement: Tier 3b spatial curvature resample API, Requirement: Tier 3b spatial CWT scaleogram API, and Requirement: Tier 3b spatial CWT ridge API)
- 3 stub modules: `parametric`, `plotting`, `pipeline`

Each stub module SHALL define exactly one canonical callable per the table below, raising `NotImplementedError(f"PR #{N} — see docs/circumnutation/roadmap.md")` with the documented PR number. Each stub callable SHALL also have a complete Google-style docstring (Args, Returns, Raises) so `pydocstyle --convention=google` and the mkdocstrings auto-generated reference pages remain informative. Stubs whose tier PR will compose with the typed `ConstantsT` override-bag SHALL include `constants=None` as a forward-compatible keyword parameter so callers do not get `TypeError` before `NotImplementedError`.

| Stub module | Canonical callable | PR # |
|---|---|---|
| `parametric` | `compute(tier3_df, R_px, omega, Delta_phi)` | 11 |
| `plotting` | `scaleogram(scaleogram_result, out_path)` | 16 |
| `pipeline` | `compute_traits(inputs, constants=None)` | 14 |

The `kinematics` module SHALL be importable on the same terms as the other modules (clean import, namespaced logger) and SHALL expose `compute(trajectory_df, constants=None)` per Requirement: Tier 0 raw kinematic traits. The `qc` module SHALL be importable on the same terms and SHALL expose `compute(trajectory_df, constants=None)` per Requirement: QC tier per-track quality traits. The `synthetic` module SHALL be importable on the same terms and SHALL expose `generate_trajectory(...)` per Requirement: Synthetic trajectory generator. The `temporal_cwt` module SHALL be importable on the same terms and SHALL expose `compute_scaleogram(x, cadence_s, constants=None) -> ScaleogramResult` and `extract_ridge(scaleogram_result, constants=None) -> RidgeResult` per Requirement: Temporal CWT scaleogram API and Requirement: Temporal CWT ridge API, AND SHALL ALSO expose `smooth_ridge(ridge_result, window=None, constants=None) -> RidgeResult` per Requirement: Temporal CWT ridge-continuity smoothing API. The `nutation` module SHALL be importable on the same terms and SHALL expose `compute(trajectory_df, cadence_s, coordinate="lateral", constants=None) -> pd.DataFrame` per Requirement: Tier 1 nutation trait emission API. The `psi_g` module SHALL be importable on the same terms and SHALL expose `compute(trajectory_df, cadence_s, constants=None) -> pd.DataFrame` per Requirement: Tier 2 ψ_g trait emission API. The `midline` module SHALL be importable on the same terms and SHALL expose `reconstruct(x, y, cadence_s, sg_window=None, constants=None) -> MidlineResult` per Requirement: Tier 3a midline reconstruction API. The `spatial_cwt` module SHALL be importable on the same terms and SHALL expose `compute_scaleogram(kappa, ds, constants=None) -> SpatialScaleogramResult` and `extract_ridge(scaleogram_result, constants=None) -> SpatialRidgeResult` per Requirement: Tier 3b spatial CWT scaleogram API and Requirement: Tier 3b spatial CWT ridge API, AND SHALL ALSO expose `resample_curvature(curvature_px_inv, arc_length_px, velocity_sub_noise_mask=None, constants=None) -> ResampleResult` per Requirement: Tier 3b spatial curvature resample API. Unlike the stub modules, calling `kinematics.compute`, `qc.compute`, `synthetic.generate_trajectory`, `temporal_cwt.compute_scaleogram`, `nutation.compute`, `psi_g.compute`, `midline.reconstruct`, `spatial_cwt.resample_curvature`, `spatial_cwt.compute_scaleogram`, or `spatial_cwt.extract_ridge` with a valid input SHALL NOT raise `NotImplementedError`.

The `_noise` and `_geometry` modules are private (underscore-prefixed) shared internals; their canonical callables are documented under Requirement: Tier 0 helper modules (and, for `_geometry.compute_signed_area`, under Requirement: Tier 2 ψ_g trait emission API; and, for `_noise.compute_sg_derivative` / `_geometry.compute_path_curvature`, under Requirement: Tier 3a midline reconstruction API).

**Scope note on PR #6 addition-vs-transition.** The `nutation` module is NEWLY created in PR #6 — it was never a stub module in PR #1–#5, and therefore does not appear in the stub-callable table. The implementation-module count grows from 4 (PR #5 baseline: kinematics, qc, synthetic, temporal_cwt) to 5 by ADDITION of `nutation`, not by transition from a prior stub. This was the first PR in the program to grow the implementation set without shrinking the stub set.

**Scope note on PR #7 stub-to-implementation transition.** The `psi_g` module was a stub in PR #1–#6 (it appeared in the stub-callable table with canonical callable `compute_psi_g(x, y, constants=None)`). PR #7 graduated it to an implementation module: the implementation-module count grew from 5 to 6 AND the stub-module count shrank from 6 to 5. The canonical callable was RENAMED `compute_psi_g` → `compute`.

**Scope note on PR #8 stub-to-implementation transition.** The `midline` module was a stub in PR #1–#7 (it appeared in the stub-callable table with canonical callable `reconstruct(x, y, cadence_s, constants=None)`). PR #8 graduated it to an implementation module: the implementation-module count grew from 6 to 7 AND the stub-module count shrank from 5 to 4 (the same stub→impl shape as PR #7). The canonical callable KEPT its name `reconstruct` (no rename); the implementation signature ADDED a `sg_window=None` parameter (`reconstruct(x, y, cadence_s, constants=None)` → `reconstruct(x, y, cadence_s, sg_window=None, constants=None)`), locked by Requirement: Tier 3a midline reconstruction API.

**Scope note on PR #9 stub-to-implementation transition.** The `spatial_cwt` module IS a stub in PR #1–#8 (it appeared in the stub-callable table with canonical callable `compute_scaleogram(kappa, ds, constants=None)`, PR #9). PR #9 graduates it to an implementation module: the implementation-module count grows from 7 to **8** AND the stub-module count shrinks from 4 to **3** (the same stub→impl shape as PR #7/#8). The canonical callable KEEPS its name `compute_scaleogram` (no rename); the implementation signature is EXACTLY the stub-table signature `compute_scaleogram(kappa, ds, constants=None)` — the speculative `wavelet=`/`scale_range=` keyword parameters present in the PR #1 stub file are DROPPED (the wavelet and scale range are derived from `constants`, mirroring `temporal_cwt.compute_scaleogram`'s `(x, cadence_s, constants=None)` precedent). `spatial_cwt` is therefore removed from the stub-callable table and from the "remaining stub" enumeration, and gains callability scenarios below. PR #9 ADDS two further public symbols not previously in the stub table — `resample_curvature` (the non-uniform→uniform κ(s) resample entry helper) and `extract_ridge` (the spatial ridge) — whose callability contracts are locked here for symmetry with `compute_scaleogram`, mirroring how PR #5's `extract_ridge` and PR #6's `smooth_ridge` were locked. **PR #9 descopes `L_gz`/`L_c` growth-zone-structure detection** (the §7.4 |κ|-envelope-peak premise does not transfer to top-view tip-trail κ(s); see the Tier 3b requirements and follow-up issue #230); `spatial_cwt` therefore exposes no `detect_growth_zone` symbol.

#### Scenario: All stub modules import cleanly
- **WHEN** a user runs `import sleap_roots.circumnutation as c; import sleap_roots.circumnutation.kinematics, sleap_roots.circumnutation.qc, sleap_roots.circumnutation.synthetic, sleap_roots.circumnutation.temporal_cwt, sleap_roots.circumnutation.nutation, sleap_roots.circumnutation.psi_g, sleap_roots.circumnutation.midline, sleap_roots.circumnutation.spatial_cwt, sleap_roots.circumnutation.parametric, sleap_roots.circumnutation.plotting, sleap_roots.circumnutation.pipeline`
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

### Requirement: Module-level constants
The system SHALL expose all overridable defaults as module-level named constants in `sleap_roots/circumnutation/_constants.py`. The set SHALL include at minimum: `NOISE_MASK_K`, `LGZ_STEADY_STATE_RESIDUAL_MAX`, `NYQUIST_RATIO_MAX`, `SG_D2_AGREEMENT_MAX`, `SG_MSD_AGREEMENT_MAX`, `D2_MSD_AGREEMENT_MAX`, `LGZ_NMIN_RESOLVABLE`, `COI_FRACTION_MAX`, `BAND_POWER_NOISE_RATIO`, `WAVELET_DEFAULT_TEMPORAL`, `WAVELET_DEFAULT_SPATIAL`, `SG_WINDOW_SHORT`, `SG_DEGREE`, `SG_WINDOW_DETREND`, `OUTLIER_STEP_RATIO`, `FRAC_OUTLIER_STEPS_MAX`, `WORST_STEP_RATIO_MAX`, `GROWTH_AXIS_RELIABILITY_K`, `SYNTHETIC_T_NUTATION_S`, `SYNTHETIC_AMPLITUDE_PX`, `SYNTHETIC_GROWTH_RATE_PX_PER_FRAME`, `SYNTHETIC_NOISE_SIGMA_PX`, `SYNTHETIC_CADENCE_S`, `SYNTHETIC_N_FRAMES`, `SYNTHETIC_GROWTH_AXIS_ANGLE_RAD`, `COI_EFOLDING_FACTOR`, `CWT_SCALE_COUNT_DEFAULT`, `CWT_PERIOD_MIN_NYQUIST_FACTOR`, `CWT_PERIOD_MAX_SIGNAL_FRACTION`, `RIDGE_CONTINUITY_FILTER_WINDOW`, `NOISE_FLOOR_OUT_OF_BAND_FACTOR`, `BAND_POWER_BAND_LOW_FACTOR`, `BAND_POWER_BAND_HIGH_FACTOR`, `DERR_EXPECTED_PERIOD_S`, `TEMPORAL_NYQUIST_RATIO_MAX`, `SPATIAL_COI_EFOLDING_FACTOR`, `CWT_WAVELENGTH_MIN_NYQUIST_FACTOR`, `CWT_WAVELENGTH_MAX_SIGNAL_FRACTION`, `_SCHEMA_VERSION`, `_CONSTANTS_VERSION`. The values SHALL match the defaults in `docs/circumnutation/roadmap.md` cross-cutting concern CC-2 and `docs/circumnutation/theory.md` §7.6 (for the QC-tier-introduced thresholds: `FRAC_OUTLIER_STEPS_MAX = 0.05`, `WORST_STEP_RATIO_MAX = 5`, `SG_MSD_AGREEMENT_MAX = 1.5`, `D2_MSD_AGREEMENT_MAX = 1.5`) and `docs/circumnutation/preliminary_results_2026-05-07.md` §1, §3.4, §4.1, §4.3 (for the synthetic-generator defaults: `SYNTHETIC_T_NUTATION_S = 3333.0`, `SYNTHETIC_AMPLITUDE_PX = 10.0`, `SYNTHETIC_GROWTH_RATE_PX_PER_FRAME = 4.29`, `SYNTHETIC_NOISE_SIGMA_PX = 2.0`, `SYNTHETIC_CADENCE_S = 300.0`, `SYNTHETIC_N_FRAMES = 575`, `SYNTHETIC_GROWTH_AXIS_ANGLE_RAD = math.pi / 2`) and the wavelet-aware step-response derivation in PR #5's archived `design.md` D3 (for the CWT-machinery defaults: `COI_EFOLDING_FACTOR = math.sqrt(1.5)` calibrated for `cmor1.5-1.0` per `√B` envelope e-folding; `CWT_SCALE_COUNT_DEFAULT = 64`; `CWT_PERIOD_MIN_NYQUIST_FACTOR = 2.0`; `CWT_PERIOD_MAX_SIGNAL_FRACTION = 0.25`) and PR #6's `design.md` D4/D7/D8/D9 (for the Tier 1 / threshold defaults: `RIDGE_CONTINUITY_FILTER_WINDOW = 5`; `NOISE_FLOOR_OUT_OF_BAND_FACTOR = 5.0`; `BAND_POWER_BAND_LOW_FACTOR = 0.5` and `BAND_POWER_BAND_HIGH_FACTOR = 2.0`; `DERR_EXPECTED_PERIOD_S = 3333.0`; `TEMPORAL_NYQUIST_RATIO_MAX = 0.25`) and PR #9's `design.md` (for the spatial-CWT-machinery defaults: `SPATIAL_COI_EFOLDING_FACTOR` set to the empirically-measured `cgau2` e-folding factor from a step-response capture script — the analog of `COI_EFOLDING_FACTOR=√1.5` for `cmor1.5-1.0`, whose docstring explicitly defers the `cgau2` factor to PR #9; `CWT_WAVELENGTH_MIN_NYQUIST_FACTOR = 2.0` and `CWT_WAVELENGTH_MAX_SIGNAL_FRACTION = 0.25`, the spatial-domain siblings of `CWT_PERIOD_MIN_NYQUIST_FACTOR` / `CWT_PERIOD_MAX_SIGNAL_FRACTION` — same numeric defaults, dimensional separation in NAMES + docstrings). `_SCHEMA_VERSION` SHALL be `1` (unchanged from PR #1) and `_CONSTANTS_VERSION` SHALL be `6` (bumped from `5` in this PR per the version-sentinel contract — the constants set grew by 3). The module SHALL also expose `PIPELINE_UNIT_VOCABULARY` (px-based + calibration-independent units, the closed sidecar vocabulary), `CONVERTED_UNIT_VOCABULARY` (mm-based units produced by `convert_to_mm`), and `VALID_UNIT_VOCABULARY` (their union), plus `ROW_IDENTITY_UNITS` (the canonical units dict for the eight row-identity columns).

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

## ADDED Requirements

### Requirement: Tier 3b spatial curvature resample API
The system SHALL provide `sleap_roots.circumnutation.spatial_cwt.resample_curvature(curvature_px_inv: np.ndarray, arc_length_px: np.ndarray, velocity_sub_noise_mask: Optional[np.ndarray] = None, constants: Optional[ConstantsT] = None) -> ResampleResult`. The function SHALL resample the PR #8 midline curvature κ(s) — emitted on the native NON-uniform `arc_length_px` grid — onto a uniform-spacing grid suitable for the spatial CWT (the locked PR #8→#9 handoff). It SHALL take RAW arrays (NOT a `MidlineResult`), so it is unit-testable and carries no `spatial_cwt → midline` import edge.

The function SHALL: (1) drop frames flagged by `velocity_sub_noise_mask` (where provided; `True` = sub-noise = excluded, matching the PR #8 mask polarity) AND any non-finite `(curvature, arc_length)` pairs — sub-noise frames carry noise-amplified curvature since κ = (ẋÿ−ẏẍ)/|v|³ blows up as |v|→0; (2) reparameterize the surviving samples onto an apex-origin axis `s_a = max(arc_length_px) − arc_length_px` (apex at `s_a = 0`, per the theory §6.5 fossil-marker convention: the latest/largest-arc tip position is the current apex); (3) choose `ds` as the median of the positive differences of the sorted surviving `s_a` (the typical per-frame arc step, robust to large gaps); (4) build a uniform grid from `0` to the surviving arc-span at spacing `ds` and interpolate κ via `np.interp`. λ itself is orientation-invariant; the apex-origin axis exists so downstream consumers (PR #10) can do apex-vs-basal comparisons.

The function SHALL be deterministic per CC-6: same input → bit-identical `kappa_uniform` across calls in the same process (`atol=0`) AND identical to within `atol=1e-9, rtol=0` across Ubuntu / Windows / macOS CI runners. The function SHALL log exactly one `logger.debug` message at the start (after input validation) whose text begins with `"resample_curvature("` and contains the tokens `n_input=`, `n_unmasked=`, `ds=`, `arc_span_px=`. No INFO/WARNING/ERROR records SHALL be emitted on the happy path.

The function SHALL validate inputs strictly (field-named, runs FIRST and unconditionally): `curvature_px_inv` and `arc_length_px` SHALL be 1-D `np.ndarray` (coercible from integer/float; rejecting `complex`/`object`) of EQUAL length (else `ValueError`); non-ndarray raises `TypeError`; `velocity_sub_noise_mask`, where provided, SHALL be a bool-coercible 1-D array of the same length (else `ValueError`/`TypeError`); `constants` SHALL be `None` or a `ConstantsT` instance (else `TypeError`). Validation always wins over the degenerate path.

After validation, a degenerate gate runs ONLY on valid inputs and SHALL return a graceful all-NaN `ResampleResult` (`is_degenerate=True`) — never raising, never emitting `np.RuntimeWarning` (computed under `np.errstate` with a `~np.isfinite` sweep) — when, after dropping masked + non-finite frames, fewer than a documented minimum number of samples survive OR the surviving arc-span is non-positive.

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

#### Scenario: resample_curvature uses the apex-origin convention (apex at s_a = 0)
- **GIVEN** finite arrays with strictly increasing `arc_length_px` (no mask)
- **WHEN** `resample_curvature(curvature_px_inv, arc_length_px)` is invoked
- **THEN** `result.s_a_uniform_px[0] == 0.0` corresponds to the apex (the largest `arc_length_px`, i.e. the latest tip position per theory §6.5)
- **AND** `result.arc_span_px == max(arc_length_px) − min(arc_length_px)` to within numerical precision over the unmasked samples

#### Scenario: resample_curvature rejects malformed or mismatched inputs with TypeError or ValueError naming the field
- **WHEN** `resample_curvature(curvature_px_inv, arc_length_px)` is invoked with a non-ndarray argument, OR `curvature_px_inv.ndim != 1`, OR `complex`/`object` dtype, OR `len(curvature_px_inv) != len(arc_length_px)`, OR a `velocity_sub_noise_mask` of wrong length
- **THEN** `TypeError` (non-ndarray / bad mask type) or `ValueError` (shape / dtype / length) is raised
- **AND** the exception message names the offending field

#### Scenario: resample_curvature returns a graceful all-NaN ResampleResult on degenerate input
- **GIVEN** valid inputs that, after dropping masked + non-finite frames, leave fewer than the documented minimum surviving samples OR a non-positive surviving arc-span (e.g., an all-masked input, or all `arc_length_px` equal)
- **WHEN** `resample_curvature(...)` is invoked
- **THEN** the call does NOT raise and emits no `RuntimeWarning`
- **AND** the returned `ResampleResult` has `is_degenerate == True` and `kappa_uniform` all-NaN

#### Scenario: resample_curvature is deterministic and emits exactly one DEBUG record
- **GIVEN** a valid input and `caplog.set_level(logging.DEBUG)`
- **WHEN** `resample_curvature(...)` is invoked twice in the same process
- **THEN** `np.array_equal(result1.kappa_uniform, result2.kappa_uniform)` is True at `atol=0`
- **AND** exactly one `DEBUG` record is emitted by `sleap_roots.circumnutation.spatial_cwt`, its message starting with `"resample_curvature("` and containing the tokens `"n_input="`, `"n_unmasked="`, `"ds="`, `"arc_span_px="`
- **AND** no `INFO`/`WARNING`/`ERROR`/`CRITICAL` records are emitted

### Requirement: Tier 3b spatial CWT scaleogram API
The system SHALL provide `sleap_roots.circumnutation.spatial_cwt.compute_scaleogram(kappa: np.ndarray, ds: float, constants: Optional[ConstantsT] = None) -> SpatialScaleogramResult`. The function SHALL accept the canonical `(kappa, ds, constants=None)` signature locked by the foundation's Package layout requirement. The function SHALL compute a spatial Continuous Wavelet Transform of the uniform-grid curvature `kappa` using the `cgau2` mother wavelet by default (overridable via `constants.WAVELET_DEFAULT_SPATIAL`), at log-spaced scales over an auto-derived spatial-wavelength range `[constants.CWT_WAVELENGTH_MIN_NYQUIST_FACTOR * ds, constants.CWT_WAVELENGTH_MAX_SIGNAL_FRACTION * len(kappa) * ds]`, returning a frozen `SpatialScaleogramResult` containing the complex-valued scaleogram, scales axis, spatial-wavelength axis (`wavelengths_px`, derived via `pywt.scale2frequency` round-trip for wavelet-agnostic correctness), spatial-frequency axis (`spatial_freqs_px_inv = 1.0 / wavelengths_px`), cone-of-influence boolean mask (computed via wavelet-aware `constants.SPATIAL_COI_EFOLDING_FACTOR * scale`), the resolved `ds`, and the resolved wavelet name.

The function SHALL emit NO trait values. The function SHALL be deterministic per CC-6: same input → bit-identical scaleogram across calls in the same process AND identical to within `atol=1e-9, rtol=0` across Ubuntu / Windows / macOS CI runners. The function SHALL log exactly one `logger.debug` message at the start (after input validation) whose text begins with `"compute_scaleogram("` and contains the named tokens `n_samples=`, `ds=`, `n_scales=`, `wavelength_min_px=`, `wavelength_max_px=`, `wavelet=`. No INFO, WARNING, or ERROR log records SHALL be emitted on the happy path.

The function SHALL validate inputs strictly: `kappa` SHALL be a 1-D `np.ndarray` (coercible from integer/float dtypes; rejecting `complex` and `object`); `kappa` SHALL be all-finite (no NaN, no ±inf); `len(kappa)` SHALL be ≥ `MIN_SAMPLES_REQUIRED` derived at call time as `int(math.floor(constants.CWT_WAVELENGTH_MIN_NYQUIST_FACTOR / constants.CWT_WAVELENGTH_MAX_SIGNAL_FRACTION)) + 1` (= `9` at defaults); positive-finite guards SHALL fire on `constants.CWT_WAVELENGTH_MAX_SIGNAL_FRACTION` and `constants.CWT_WAVELENGTH_MIN_NYQUIST_FACTOR`; `ds` SHALL be a Python `int` or `float` or numpy scalar (rejecting `bool` subtype and `str`); `ds > 0` and `math.isfinite(ds)`; `constants` SHALL be `None` or a `ConstantsT` instance. Invalid inputs SHALL raise `ValueError` or `TypeError` with the offending field name embedded in the message.

The `SpatialScaleogramResult` class SHALL be an `@attrs.define(frozen=True, slots=False, kw_only=True, eq=False)` container with exactly the following fields (in this order): `scaleogram: np.ndarray` (shape `(n_scales, n_samples)`, dtype `complex128`); `scales: np.ndarray` (shape `(n_scales,)`, dtype `float64`, monotonically increasing); `wavelengths_px: np.ndarray` (shape `(n_scales,)`, dtype `float64`); `spatial_freqs_px_inv: np.ndarray` (shape `(n_scales,)`, dtype `float64`, equal to `1.0 / wavelengths_px` to within numerical precision); `coi_mask: np.ndarray` (shape `(n_scales, n_samples)`, dtype `bool`; `True` indicates inside-COI = unreliable); `ds: float`; `wavelet: str`. (`eq=False` because ndarray `__eq__` is ambiguous.)

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
- **AND** the dominant spatial wavelength from `extract_ridge` (the COI-valid median of `wavelengths_px`) is finite and physically plausible (recorded for auditability)

### Requirement: Tier 3b spatial CWT ridge API
The system SHALL provide `sleap_roots.circumnutation.spatial_cwt.extract_ridge(scaleogram_result: SpatialScaleogramResult, constants: Optional[ConstantsT] = None) -> SpatialRidgeResult`. The function SHALL extract a per-position ridge from the input spatial scaleogram via deterministic per-position argmax of `|scaleogram|` along the scale axis (numpy's documented tie-breaking returns the smallest index on equal values), yielding the dominant spatial wavelength λ(s_a) at each position — the steady-traveling-wave quantity PR #10 consumes.

The function SHALL emit NO trait values. The function SHALL be deterministic: same `SpatialScaleogramResult` input → identical `SpatialRidgeResult` output at `atol=0`. The function SHALL log exactly one `logger.debug` message at the start (after input validation) whose text begins with `"extract_ridge("` and contains tokens `n_scales=` and `n_samples=`. No INFO, WARNING, or ERROR log records SHALL be emitted on the happy path.

The function SHALL validate inputs strictly: `scaleogram_result` SHALL be a `SpatialScaleogramResult` instance (anything else raises `TypeError`); empty `SpatialScaleogramResult` (with `n_scales == 0` OR `n_samples == 0`) SHALL raise `ValueError`; `constants` SHALL be `None` or a `ConstantsT` instance.

The `SpatialRidgeResult` class SHALL be an `@attrs.define(frozen=True, slots=False, kw_only=True, eq=False)` container with exactly the following fields (in this order): `position_indices: np.ndarray` (shape `(n_samples,)`, dtype `int64`, equal to `np.arange(n_samples, dtype=np.int64)`); `wavelengths_px: np.ndarray` (shape `(n_samples,)`, dtype `float64`; indexed by position, NOT by scale — value at index `i` is the spatial wavelength AT THE RIDGE for position `i`); `amplitudes: np.ndarray` (shape `(n_samples,)`, dtype `float64`, equal to `|C|` at the ridge cell, all ≥ 0); `powers: np.ndarray` (shape `(n_samples,)`, dtype `float64`, equal to `amplitudes ** 2`, redundant by construction and intentionally preserved for symmetry with the temporal `RidgeResult`); `in_coi: np.ndarray` (shape `(n_samples,)`, dtype `bool`; `True` iff `scaleogram_result.coi_mask[ridge_scale_idx, position_idx]`). The ridge SHALL NOT be pre-COI-masked; PR #10's trait emission applies the COI mask.

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
- **THEN** the median of `result.wavelengths_px` over the COI-interior positions (`~result.in_coi`) equals `lambda_true` to within a documented tolerance (the spatial-CWT scale-grid resolution)

#### Scenario: extract_ridge rejects non-SpatialScaleogramResult input with TypeError
- **WHEN** `extract_ridge(x)` is invoked with `x` equal to `None` OR `{}` OR `(1, 2, 3)` OR `np.zeros((10, 10))`
- **THEN** `TypeError` is raised
- **AND** the exception message references the expected type `SpatialScaleogramResult`

#### Scenario: extract_ridge rejects empty SpatialScaleogramResult with ValueError
- **GIVEN** a `SpatialScaleogramResult` constructed with `n_scales == 0` (scaleogram shape `(0, n_samples)`) OR `n_samples == 0` (scaleogram shape `(n_scales, 0)`)
- **WHEN** `extract_ridge(scaleogram_result)` is invoked
- **THEN** `ValueError` is raised
- **AND** the exception message references the empty-axis condition (e.g., `"n_scales == 0"` or `"n_samples == 0"`)

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
