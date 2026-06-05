## MODIFIED Requirements

### Requirement: Package layout
The system SHALL provide a `sleap_roots.circumnutation` sub-package whose import-tree is complete from this PR onward — every module name referenced anywhere in `docs/circumnutation/roadmap.md` exists as an importable Python module from PR #1 forward, even if its functions raise `NotImplementedError` until later PRs land. The package SHALL contain:

- 7 contract modules: `__init__.py`, `_constants.py`, `_types.py`, `_io.py`, `units.py`, `_noise.py`, `_geometry.py`
- 6 implementation modules: `kinematics` (implemented from PR #2 onward; see Requirement: Tier 0 raw kinematic traits), `qc` (implemented from PR #3 onward; see Requirement: QC tier per-track quality traits), `synthetic` (implemented from PR #4 onward; see Requirement: Synthetic trajectory generator), `temporal_cwt` (implemented from PR #5 onward; see Requirement: Temporal CWT scaleogram API and Requirement: Temporal CWT ridge API), `nutation` (implemented from PR #6 onward; see Requirement: Tier 1 nutation trait emission API), and `psi_g` (implemented from PR #7 onward; see Requirement: Tier 2 ψ_g trait emission API)
- 5 stub modules: `midline`, `spatial_cwt`, `parametric`, `plotting`, `pipeline`

Each stub module SHALL define exactly one canonical callable per the table below, raising `NotImplementedError(f"PR #{N} — see docs/circumnutation/roadmap.md")` with the documented PR number. Each stub callable SHALL also have a complete Google-style docstring (Args, Returns, Raises) so `pydocstyle --convention=google` and the mkdocstrings auto-generated reference pages remain informative. Stubs whose tier PR will compose with the typed `ConstantsT` override-bag SHALL include `constants=None` as a forward-compatible keyword parameter so callers do not get `TypeError` before `NotImplementedError`.

| Stub module | Canonical callable | PR # |
|---|---|---|
| `midline` | `reconstruct(x, y, cadence_s, constants=None)` | 8 |
| `spatial_cwt` | `compute_scaleogram(kappa, ds, constants=None)` | 9 |
| `parametric` | `compute(tier3_df, R_px, omega, Delta_phi)` | 11 |
| `plotting` | `scaleogram(scaleogram_result, out_path)` | 16 |
| `pipeline` | `compute_traits(inputs, constants=None)` | 14 |

The `kinematics` module SHALL be importable on the same terms as the other modules (clean import, namespaced logger) and SHALL expose `compute(trajectory_df, constants=None)` per Requirement: Tier 0 raw kinematic traits. The `qc` module SHALL be importable on the same terms and SHALL expose `compute(trajectory_df, constants=None)` per Requirement: QC tier per-track quality traits. The `synthetic` module SHALL be importable on the same terms and SHALL expose `generate_trajectory(...)` per Requirement: Synthetic trajectory generator. The `temporal_cwt` module SHALL be importable on the same terms and SHALL expose `compute_scaleogram(x, cadence_s, constants=None) -> ScaleogramResult` and `extract_ridge(scaleogram_result, constants=None) -> RidgeResult` per Requirement: Temporal CWT scaleogram API and Requirement: Temporal CWT ridge API, AND SHALL ALSO expose `smooth_ridge(ridge_result, window=None, constants=None) -> RidgeResult` per Requirement: Temporal CWT ridge-continuity smoothing API. The `nutation` module SHALL be importable on the same terms and SHALL expose `compute(trajectory_df, cadence_s, coordinate="lateral", constants=None) -> pd.DataFrame` per Requirement: Tier 1 nutation trait emission API. The `psi_g` module SHALL be importable on the same terms and SHALL expose `compute(trajectory_df, cadence_s, constants=None) -> pd.DataFrame` per Requirement: Tier 2 ψ_g trait emission API. Unlike the stub modules, calling `kinematics.compute`, `qc.compute`, `synthetic.generate_trajectory`, `temporal_cwt.compute_scaleogram`, `nutation.compute`, or `psi_g.compute` with a valid input SHALL NOT raise `NotImplementedError`.

The `_noise` and `_geometry` modules are private (underscore-prefixed) shared internals; their canonical callables are documented under Requirement: Tier 0 helper modules (and, for `_geometry.compute_signed_area`, under Requirement: Tier 2 ψ_g trait emission API).

**Scope note on PR #6 addition-vs-transition.** The `nutation` module is NEWLY created in PR #6 — it was never a stub module in PR #1–#5, and therefore does not appear in the stub-callable table. The implementation-module count grows from 4 (PR #5 baseline: kinematics, qc, synthetic, temporal_cwt) to 5 by ADDITION of `nutation`, not by transition from a prior stub. The stub-module count stays at 6 (psi_g, midline, spatial_cwt, parametric, plotting, pipeline). This is the first PR in the program to grow the implementation set without shrinking the stub set.

**Scope note on PR #7 stub-to-implementation transition.** The `psi_g` module IS a stub in PR #1–#6 (it appeared in the stub-callable table with canonical callable `compute_psi_g(x, y, constants=None)`, PR #7). PR #7 graduates it to an implementation module: the implementation-module count grows from 5 to **6** AND the stub-module count shrinks from 6 to **5** — the inverse of PR #6's addition-only transition. The canonical callable is RENAMED `compute_psi_g` → `compute` to match the `kinematics.compute` / `qc.compute` / `nutation.compute` convention (the stub's `compute_psi_g` name collided conceptually with the locked `_geometry.compute_psi_g` helper, which is a DIFFERENT, retained symbol). `psi_g` is therefore removed from the stub-callable table and from the "remaining stub" enumeration, and gains a callability scenario below.

#### Scenario: All stub modules import cleanly
- **WHEN** a user runs `import sleap_roots.circumnutation as c; import sleap_roots.circumnutation.kinematics, sleap_roots.circumnutation.qc, sleap_roots.circumnutation.synthetic, sleap_roots.circumnutation.temporal_cwt, sleap_roots.circumnutation.nutation, sleap_roots.circumnutation.psi_g, sleap_roots.circumnutation.midline, sleap_roots.circumnutation.spatial_cwt, sleap_roots.circumnutation.parametric, sleap_roots.circumnutation.plotting, sleap_roots.circumnutation.pipeline`
- **THEN** every import succeeds
- **AND** no exception is raised at module-import time

#### Scenario: Calling each remaining stub raises NotImplementedError with the correct PR number
- **GIVEN** the package imported as in the previous scenario
- **WHEN** the canonical callable in each of the 5 remaining stub modules (`midline`, `spatial_cwt`, `parametric`, `plotting`, `pipeline`) is invoked (parameters per the table above; `NotImplementedError` fires before any argument check)
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
- **AND** since `extract_ridge` is a NEW public symbol introduced by PR #5 (not a transition from a prior stub), it does not appear in the stub-callable table — its callability contract is locked here in the MODIFIED Package layout requirement for symmetry with `compute_scaleogram` (per /openspec-review round-2 reviewer N-I1)

#### Scenario: `nutation.compute` is callable on a valid trajectory_df without raising
- **GIVEN** a valid `trajectory_df` (8 row-identity columns + `frame`, `tip_x`, `tip_y`; ≥ 9 rows for at least one track) and a positive finite `cadence_s` (e.g., 300.0)
- **WHEN** `sleap_roots.circumnutation.nutation.compute(trajectory_df, cadence_s=300.0)` is invoked
- **THEN** the call returns a `pandas.DataFrame` (the Tier 1 per-plant nutation trait output) without raising `NotImplementedError`
- **AND** since `nutation` is a NEW module introduced by PR #6 (not a transition from a prior stub), it does not appear in the stub-callable table — its callability contract is locked here in the MODIFIED Package layout requirement for symmetry with `kinematics.compute`, `qc.compute`, `synthetic.generate_trajectory`, and `temporal_cwt.compute_scaleogram`

#### Scenario: `temporal_cwt.smooth_ridge` is callable on a valid RidgeResult without raising
- **GIVEN** a valid `RidgeResult` produced by `extract_ridge(compute_scaleogram(x, 300.0))` on a length-≥9 finite array
- **WHEN** `sleap_roots.circumnutation.temporal_cwt.smooth_ridge(ridge_result)` is invoked
- **THEN** the call returns a `RidgeResult` (with smoothed `periods_s`) without raising any exception
- **AND** since `smooth_ridge` is a NEW public symbol introduced by PR #6 (closing GitHub issue #214 — ridge-tracking continuity post-filter; not a transition from a prior stub), it does not appear in the stub-callable table — its callability contract is locked here in the MODIFIED Package layout requirement for symmetry with `compute_scaleogram` and `extract_ridge`

#### Scenario: `psi_g.compute` is callable on a valid trajectory_df without raising
- **GIVEN** a valid `trajectory_df` (8 row-identity columns + `frame`, `tip_x`, `tip_y`; ≥ 24 rows for at least one track) and a positive finite `cadence_s` (e.g., 300.0)
- **WHEN** `sleap_roots.circumnutation.psi_g.compute(trajectory_df, cadence_s=300.0)` is invoked
- **THEN** the call returns a `pandas.DataFrame` (the Tier 2 per-plant ψ_g trait output) without raising `NotImplementedError`
- **AND** `psi_g` transitions FROM a stub (it WAS in the stub-callable table in PR #1–#6 with callable `compute_psi_g`); PR #7 removes it from that table and RENAMES the callable to `compute`, so the stub-module count shrinks 6 → 5 and the implementation-module count grows 5 → 6

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

### Requirement: Tier 2 ψ_g trait emission API
The system SHALL provide `sleap_roots.circumnutation.psi_g.compute(trajectory_df: pd.DataFrame, cadence_s: float, constants: Optional[ConstantsT] = None) -> pd.DataFrame`. Unlike `nutation.compute`, `psi_g.compute` SHALL NOT take a `coordinate` parameter: ψ_g is computed from the raw 2-D tip trajectory `(tip_x, tip_y)` via the locked `_geometry.compute_psi_g` helper (`atan2(dx, dy)`, unwrapped — Requirement: Tier 0 helper modules), so there is no 1-D-projection choice to make.

The function SHALL emit a per-track DataFrame whose rows correspond to the unique 5-tuple `(series, sample_uid, plate_id, plant_id, track_id)` derived via `groupby(_IDENTITY_5_TUPLE, dropna=False, sort=False)`. The returned DataFrame SHALL contain the 8 row-identity columns (Requirement: Trait CSV row-identity schema) followed by exactly 4 trait columns in this declared order: `T_psig_median_s: float64`, `delta_E_amplitude_proxy_px_per_frame: float64`, `handedness: int64`, `helix_signed_area_px2: float64`. All 4 trait units (`s`, `px/frame`, `int`, `px²`) SHALL be members of `PIPELINE_UNIT_VOCABULARY` (no vocabulary change is required).

The four traits SHALL be defined as follows, on the finite-masked tip coordinates (rows with non-finite `tip_x`/`tip_y` dropped first; let `N` = count of finite frames; ψ_g = `compute_psi_g(tip_x, tip_y)` has length `N−1`):

- `T_psig_median_s`: `np.nanmedian` of the COI-interior smoothed-ridge periods, computed by composing `temporal_cwt.compute_scaleogram → extract_ridge → smooth_ridge` on the **SG-detrended** ψ_g (`_noise.compute_sg_detrended(psi_g, window=SG_WINDOW_DETREND=23, polynomial_order=SG_DEGREE=3)`), masked by `~smooth_ridge.in_coi` (the per-frame boolean, NOT the 2-D `ScaleogramResult.coi_mask`). It is the ONLY trait that uses the conditioned signal and the CWT.
- `delta_E_amplitude_proxy_px_per_frame`: `np.median(√(dx² + dy²))` over all finite velocity samples (`dx, dy = np.diff` of finite tip coords); no COI mask, no `/cadence_s` (px/frame, matching Tier 0's velocity convention). Corresponds to `(L/2R)·ΔĖ` (Eq. 21).
- `handedness`: `int(np.sign(psi_g[-1] − psi_g[0]))` — the sign of the **net unwrapped ψ_g rotation over all finite frames**, with a determinism zero-guard `|psi_g[-1] − psi_g[0]| < 1e-9 rad → 0`. **Sign convention (anchored to avoid the "counterclockwise" ambiguity):** `+1` ⇔ ψ_g increasing ⇔ **positive mean `dψ_g/dt`** — identical to `synthetic.generate_trajectory(handedness=+1)`'s locked convention (Requirement: Synthetic trajectory generator, scenario "handedness=+1 yields positive mean dψ_g/dt"). In physical terms this is clockwise in standard (y-up) math axes and counterclockwise as displayed in the y-down image frame; the program anchors on the `dψ_g/dt` sign, NOT the word "counterclockwise". `0` = no net rotation / degenerate. This is COI-FREE (a deliberate deviation from theory.md §7.3's literal "sign of mean dψ_g/dt over COI-masked range" — see deviation note below).
- `helix_signed_area_px2`: `_geometry.compute_signed_area(tip_x, tip_y)` — the y-down-corrected Shoelace area `0.5·Σ(x_{i+1}·y_i − x_i·y_{i+1})` (the negation of the standard formula), so that `sign(helix_signed_area_px2) == handedness` (independent confirmation of handedness).

**Deviations from theory.md §7.3 / §6.3 (all recorded; theory.md is patched in this PR to match, preserving the original wording in an Appendix B correction note rather than silently overwriting):**

1. **`handedness` is COI-free** — theory.md §7.3 specifies "sign of mean dψ_g/dt over COI-masked range". PR #7 emits it COI-free because (a) the COI mask is a function of the SG-detrended CWT ridge, so COI-masking would couple a raw kinematic sign to the conditioned signal and to the CWT min-length floor; (b) the per-frame `ridge.in_coi` interior is not contiguous (per-frame argmax scale selection creates COI gaps), so an endpoint difference across a masked gap can report the wrong sign; and (c) COI is a CWT-edge-reliability concept that does not apply to a raw angular displacement (`atan2` of velocity has no edge contamination) — §7.3 itself omits COI for the sibling `delta_E` kinematic median, and the COI-free `angular_amplitude` (§7.1) is precedent.
2. **`delta_E` is px/frame, not px·hr⁻¹** — §7.3 specifies "Median of √(ẋ²+ẏ²) × (frames/hr)" in px·hr⁻¹. PR #7 emits `median(√(dx²+dy²))` in **px/frame** (drops the `(frames/hr)` cadence factor), because `px/s`/`px·hr⁻¹` is not the pipeline's velocity convention (Tier 0 emits all velocities in px/frame; `px/s` ∉ `PIPELINE_UNIT_VOCABULARY`). As a per-track *amplitude proxy* the prefactor-free, cadence-free magnitude preserves the shape and relative magnitude of `Δ·Ė` (R, L, cadence are per-track constants), which is all the proxy needs; the "`= (L/2R)·ΔĖ`" proportionality (Eq. 21 solved for the measured speed) still holds.
3. **ψ_g conditioning is SG-detrend, not §6.3's literal "smooth"** — `T_psig_median_s` feeds the CWT the SG **residual** (`compute_sg_detrended` = raw − SG-smooth), not the SG-smoothed signal. §6.3 says "pre-smoothed via Savitzky-Golay"; the residual is the oscillation component and is what a period-extracting CWT needs (smoothing-only would retain gravitropic drift biasing `T_psig`), and reuses the exact primitive Tier 1 uses. No public SG-smoothing primitive exists.

The function SHALL NOT consume Tier 1 output (no `nutation_df` / `is_nutating` input). Traits are emitted ungated; downstream consumers compose a `LEFT JOIN` of Tier 1's `is_nutating` on the shared 5-tuple to mask non-nutating tracks.

The function SHALL handle short and degenerate tracks gracefully (never raising, never emitting `np.RuntimeWarning`):

- `N < 3` finite frames (including an all-non-finite track): return `T_psig_median_s=NaN`, `delta_E_amplitude_proxy_px_per_frame=NaN`, `handedness=0`, `helix_signed_area_px2=NaN`.
- `3 ≤ N < 24` (ψ_g shorter than `SG_WINDOW_DETREND`): `T_psig_median_s=NaN` (the CWT path is skipped — `compute_sg_detrended` returns all-NaN for `len < window`, which would otherwise make `compute_scaleogram` raise); `handedness`, `delta_E_amplitude_proxy_px_per_frame`, `helix_signed_area_px2` are fully defined (they are CWT-free).
- Stationary tip or perfectly straight growth (zero detrended energy) with `N ≥ 24`: `T_psig_median_s=NaN` via a zero-energy guard (`np.allclose(psi_g_detrended, 0.0)` → skip the CWT), because `compute_scaleogram` accepts an all-zero signal without raising and `argmax`-over-zeros would otherwise yield a spurious shortest-period ridge.

The function SHALL be deterministic per CC-6: the 3 float trait columns SHALL be bit-identical at `atol=0` across calls in the same process AND identical to within `atol=1e-6` across Ubuntu / Windows / macOS CI runners (inheriting Tier 1's loosened float floor for the SG-detrend→scipy-CWT stack). The integer `handedness` column SHALL be exactly equal across OSes; its `1e-9 rad` numerical-zero guard is safe because the net-rotation endpoint difference is a raw `atan2` quantity (~`1e-12` reproducible), not a CWT output.

The function SHALL validate inputs strictly: `trajectory_df` validation delegates to `_validate_trajectory_df` (Requirement: Tier 0 input-validation boundary); `cadence_s` validation reuses `temporal_cwt._validate_cadence_s` (positive finite; explicit `bool`/`np.bool_` rejection); `constants` SHALL be None or `ConstantsT` else TypeError, and an invalid SG override (e.g. even `SG_WINDOW_DETREND`, or `SG_WINDOW_DETREND ≤ SG_DEGREE`) SHALL raise `ValueError` naming the field. The function SHALL log exactly one `logger.debug` record at the start (after validation) whose text begins with `"psi_g.compute("` and contains the tokens `n_tracks=` and `cadence_s=` (no `coordinate=` token); no INFO/WARNING/ERROR records SHALL be emitted on the happy path.

The system SHALL ALSO provide `_geometry.compute_signed_area(x: np.ndarray, y: np.ndarray) -> float`: the y-down-corrected Shoelace signed area `0.5·Σ_i (x_{i+1}·y_i − x_i·y_{i+1})` (cyclic). It SHALL return `0.0` for inputs of fewer than 3 points (degenerate polygon) and SHALL propagate NaN for non-finite coordinates. Its sign convention SHALL be documented as load-bearing: positive area ↔ counterclockwise ↔ `handedness=+1` under the same image-y-down `atan2(dx, dy)` convention `compute_psi_g` encodes.

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

#### Scenario: handedness tracks the generator's planted handedness and agrees with helix_signed_area sign
- **GIVEN** two synthetic orbits via `synthetic.generate_trajectory(handedness=+1, amplitude_px=10, noise_sigma_px=0.0)` and the same with `handedness=-1` (the generator's `handedness` is the locked `dψ_g/dt`-sign convention, NOT a screen-orientation label)
- **WHEN** `psi_g.compute(...)` is invoked for each
- **THEN** the `handedness=+1` track yields output `handedness == +1`; the `handedness=-1` track yields output `handedness == -1` (the emitted `handedness` equals the planted sign)
- **AND** in both cases the sign-agreement invariant holds: `int(np.sign(helix_signed_area_px2)) == handedness` (so the `+1` track has `helix_signed_area_px2 > 0` and the `-1` track has `helix_signed_area_px2 < 0`)

#### Scenario: compute_signed_area sign is pinned to an absolute hand-built orbit
- **WHEN** `_geometry.compute_signed_area(np.array([0.,1.,1.,0.]), np.array([0.,0.,1.,1.]))` is invoked
- **THEN** the returned value equals `-1.0` exactly (the y-down negation of the standard Shoelace `+1.0`)
- **AND** `_geometry.compute_psi_g(np.array([0.,1.,1.,0.]), np.array([0.,0.,1.,1.]))` has net change `psi_g[-1] − psi_g[0] = −π` so `int(np.sign(net)) == -1`, matching `int(np.sign(-1.0))`
- **AND** `compute_signed_area` returns `0.0` for an input of fewer than 3 points

#### Scenario: conditioning affects only T_psig_median_s
- **GIVEN** the same synthetic track conditioned identically
- **WHEN** the pipeline is traced
- **THEN** `handedness`, `delta_E_amplitude_proxy_px_per_frame`, and `helix_signed_area_px2` are computed from raw inputs (raw unwrapped ψ_g, raw velocity samples, raw coordinates respectively) and do NOT depend on `compute_sg_detrended` or the CWT
- **AND** only `T_psig_median_s` uses the SG-detrended ψ_g and the `compute_scaleogram → extract_ridge → smooth_ridge` chain

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
