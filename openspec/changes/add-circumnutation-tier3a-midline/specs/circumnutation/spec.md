## MODIFIED Requirements

### Requirement: Package layout
The system SHALL provide a `sleap_roots.circumnutation` sub-package whose import-tree is complete from this PR onward — every module name referenced anywhere in `docs/circumnutation/roadmap.md` exists as an importable Python module from PR #1 forward, even if its functions raise `NotImplementedError` until later PRs land. The package SHALL contain:

- 7 contract modules: `__init__.py`, `_constants.py`, `_types.py`, `_io.py`, `units.py`, `_noise.py`, `_geometry.py`
- 7 implementation modules: `kinematics` (implemented from PR #2 onward; see Requirement: Tier 0 raw kinematic traits), `qc` (implemented from PR #3 onward; see Requirement: QC tier per-track quality traits), `synthetic` (implemented from PR #4 onward; see Requirement: Synthetic trajectory generator), `temporal_cwt` (implemented from PR #5 onward; see Requirement: Temporal CWT scaleogram API and Requirement: Temporal CWT ridge API), `nutation` (implemented from PR #6 onward; see Requirement: Tier 1 nutation trait emission API), `psi_g` (implemented from PR #7 onward; see Requirement: Tier 2 ψ_g trait emission API), and `midline` (implemented from PR #8 onward; see Requirement: Tier 3a midline reconstruction API)
- 4 stub modules: `spatial_cwt`, `parametric`, `plotting`, `pipeline`

Each stub module SHALL define exactly one canonical callable per the table below, raising `NotImplementedError(f"PR #{N} — see docs/circumnutation/roadmap.md")` with the documented PR number. Each stub callable SHALL also have a complete Google-style docstring (Args, Returns, Raises) so `pydocstyle --convention=google` and the mkdocstrings auto-generated reference pages remain informative. Stubs whose tier PR will compose with the typed `ConstantsT` override-bag SHALL include `constants=None` as a forward-compatible keyword parameter so callers do not get `TypeError` before `NotImplementedError`.

| Stub module | Canonical callable | PR # |
|---|---|---|
| `spatial_cwt` | `compute_scaleogram(kappa, ds, constants=None)` | 9 |
| `parametric` | `compute(tier3_df, R_px, omega, Delta_phi)` | 11 |
| `plotting` | `scaleogram(scaleogram_result, out_path)` | 16 |
| `pipeline` | `compute_traits(inputs, constants=None)` | 14 |

The `kinematics` module SHALL be importable on the same terms as the other modules (clean import, namespaced logger) and SHALL expose `compute(trajectory_df, constants=None)` per Requirement: Tier 0 raw kinematic traits. The `qc` module SHALL be importable on the same terms and SHALL expose `compute(trajectory_df, constants=None)` per Requirement: QC tier per-track quality traits. The `synthetic` module SHALL be importable on the same terms and SHALL expose `generate_trajectory(...)` per Requirement: Synthetic trajectory generator. The `temporal_cwt` module SHALL be importable on the same terms and SHALL expose `compute_scaleogram(x, cadence_s, constants=None) -> ScaleogramResult` and `extract_ridge(scaleogram_result, constants=None) -> RidgeResult` per Requirement: Temporal CWT scaleogram API and Requirement: Temporal CWT ridge API, AND SHALL ALSO expose `smooth_ridge(ridge_result, window=None, constants=None) -> RidgeResult` per Requirement: Temporal CWT ridge-continuity smoothing API. The `nutation` module SHALL be importable on the same terms and SHALL expose `compute(trajectory_df, cadence_s, coordinate="lateral", constants=None) -> pd.DataFrame` per Requirement: Tier 1 nutation trait emission API. The `psi_g` module SHALL be importable on the same terms and SHALL expose `compute(trajectory_df, cadence_s, constants=None) -> pd.DataFrame` per Requirement: Tier 2 ψ_g trait emission API. The `midline` module SHALL be importable on the same terms and SHALL expose `reconstruct(x, y, cadence_s, sg_window=None, constants=None) -> MidlineResult` per Requirement: Tier 3a midline reconstruction API. Unlike the stub modules, calling `kinematics.compute`, `qc.compute`, `synthetic.generate_trajectory`, `temporal_cwt.compute_scaleogram`, `nutation.compute`, `psi_g.compute`, or `midline.reconstruct` with a valid input SHALL NOT raise `NotImplementedError`.

The `_noise` and `_geometry` modules are private (underscore-prefixed) shared internals; their canonical callables are documented under Requirement: Tier 0 helper modules (and, for `_geometry.compute_signed_area`, under Requirement: Tier 2 ψ_g trait emission API; and, for `_noise.compute_sg_derivative` / `_geometry.compute_path_curvature`, under Requirement: Tier 3a midline reconstruction API).

**Scope note on PR #6 addition-vs-transition.** The `nutation` module is NEWLY created in PR #6 — it was never a stub module in PR #1–#5, and therefore does not appear in the stub-callable table. The implementation-module count grows from 4 (PR #5 baseline: kinematics, qc, synthetic, temporal_cwt) to 5 by ADDITION of `nutation`, not by transition from a prior stub. This was the first PR in the program to grow the implementation set without shrinking the stub set.

**Scope note on PR #7 stub-to-implementation transition.** The `psi_g` module was a stub in PR #1–#6 (it appeared in the stub-callable table with canonical callable `compute_psi_g(x, y, constants=None)`). PR #7 graduated it to an implementation module: the implementation-module count grew from 5 to 6 AND the stub-module count shrank from 6 to 5. The canonical callable was RENAMED `compute_psi_g` → `compute`.

**Scope note on PR #8 stub-to-implementation transition.** The `midline` module IS a stub in PR #1–#7 (it appeared in the stub-callable table with canonical callable `reconstruct(x, y, cadence_s, constants=None)`, PR #8). PR #8 graduates it to an implementation module: the implementation-module count grows from 6 to **7** AND the stub-module count shrinks from 5 to **4** (the same stub→impl shape as PR #7). The canonical callable KEEPS its name `reconstruct` (no rename — unlike PR #7's `compute_psi_g → compute`); the implementation signature ADDS a `sg_window=None` parameter to the stub-table signature (`reconstruct(x, y, cadence_s, constants=None)` → `reconstruct(x, y, cadence_s, sg_window=None, constants=None)`), locked by Requirement: Tier 3a midline reconstruction API. `midline` is therefore removed from the stub-callable table and from the "remaining stub" enumeration, and gains a callability scenario below.

#### Scenario: All stub modules import cleanly
- **WHEN** a user runs `import sleap_roots.circumnutation as c; import sleap_roots.circumnutation.kinematics, sleap_roots.circumnutation.qc, sleap_roots.circumnutation.synthetic, sleap_roots.circumnutation.temporal_cwt, sleap_roots.circumnutation.nutation, sleap_roots.circumnutation.psi_g, sleap_roots.circumnutation.midline, sleap_roots.circumnutation.spatial_cwt, sleap_roots.circumnutation.parametric, sleap_roots.circumnutation.plotting, sleap_roots.circumnutation.pipeline`
- **THEN** every import succeeds
- **AND** no exception is raised at module-import time

#### Scenario: Calling each remaining stub raises NotImplementedError with the correct PR number
- **GIVEN** the package imported as in the previous scenario
- **WHEN** the canonical callable in each of the 4 remaining stub modules (`spatial_cwt`, `parametric`, `plotting`, `pipeline`) is invoked (parameters per the table above; `NotImplementedError` fires before any argument check)
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
- **AND** `midline` transitions FROM a stub (it WAS in the stub-callable table in PR #1–#7 with callable `reconstruct`); PR #8 removes it from that table, so the stub-module count shrinks 5 → 4 and the implementation-module count grows 6 → 7 (the callable name `reconstruct` is unchanged; the signature gains a `sg_window=None` parameter)

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

### Requirement: Tier 3a midline reconstruction API
The system SHALL provide `sleap_roots.circumnutation.midline.reconstruct(x: np.ndarray, y: np.ndarray, cadence_s: float, sg_window: Optional[int] = None, constants: Optional[ConstantsT] = None) -> MidlineResult`. It reconstructs the tip-trail-as-midline (theory.md §6.1): the curve of past tip positions parameterized by arc length `s(τ) = ∫|v|dσ`, with per-frame trajectory curvature `κ = (ẋÿ − ẏẍ)/(ẋ²+ẏ²)^{3/2}` (theory.md §6.2) and tip speed. It is **machinery**: it emits NO trait columns, ingests NO `trajectory_df`, performs NO 5-tuple groupby, builds NO `L_gz` growth-zone mask, and performs NO uniform-arc-length resampling.

`x`, `y`, and `cadence_s` are required positional parameters. `sg_window` defaults to `constants.SG_WINDOW_SHORT` (5); the Savitzky-Golay polynomial degree is `constants.SG_DEGREE` (3) and is NOT a parameter. `cadence_s` is validated (via the imported `temporal_cwt._validate_cadence_s`) and stored as provenance, but the reconstruction is **cadence-independent**: the per-frame outputs are frame-parameterized (velocity in px/frame, arc length integrated over the frame index), so `cadence_s` does not affect `arc_length_px`, `curvature_px_inv`, or `velocity_sub_noise_mask`. `constants=None` resolves to `ConstantsT()`.

**Differentiation = Savitzky-Golay analytic derivatives.** The smoothed coordinates and their first/second derivatives come from ONE `savgol_filter` polynomial per coordinate (deriv=0 → `x_smooth_px`/`y_smooth_px`; deriv=1 → ẋ/ẏ; deriv=2 → ẍ/ÿ), via the new shared helper `_noise.compute_sg_derivative` (below). This realizes theory §6.2's "SG smoothing BEFORE second-derivative operations" self-consistently.

**The function SHALL return a frozen `MidlineResult`** (`@attrs.define(frozen=True, slots=False, kw_only=True)`, mirroring `ScaleogramResult`/`RidgeResult`) with these per-frame arrays, each length `n = len(x)`, index `i` ↔ frame `i`:

- `frame_indices: int64` — `np.arange(n)`.
- `x_smooth_px: float64`, `y_smooth_px: float64` — SG deriv=0 of `x`, `y` (px).
- `speed_px_per_frame: float64` — `√(ẋ² + ẏ²)` from SG deriv=1 (px/frame, the program's cadence-independent velocity convention; `px/s` is NOT used per theory §10 Appendix B).
- `arc_length_px: float64` — `scipy.integrate.cumulative_trapezoid(speed_px_per_frame, dx=1.0, initial=0)` (px); `arc_length_px[0] == 0.0`, monotonic non-decreasing.
- `curvature_px_inv: float64` — `κ` (px⁻¹) via `_geometry.compute_path_curvature(ẋ, ẏ, ẍ, ÿ)`; non-finite entries (the underflow/overflow corner) swept to NaN. This is a SINGLE array: curvature is parameterization-invariant, so the time-domain `κ_path(τ)` and arc-length `κ(s(τ))` are bit-identical (theory §6.1) — pair it with `frame_indices` for the time view or with `arc_length_px` (NON-uniformly sampled) for the arc view.
- `velocity_sub_noise_mask: bool` — `True ⇔ speed_px_per_frame ≤ NOISE_MASK_K · σ_v` (theory §6.2's sub-noise guard; `σ_v = np.std(speed_px_per_frame, ddof=0)`). `True` flags a sub-noise frame to EXCLUDE before curvature use (consumers use `curvature_px_inv[~velocity_sub_noise_mask]`), matching `ScaleogramResult.coi_mask`'s True=unreliable polarity. **This is a per-FRAME, time-domain mask and is NOT the `L_gz` growth-zone mask** (per-arc-length apical region, built in PR #10) — the two share `σ_v`/`NOISE_MASK_K` vocabulary but are different objects in different domains.

and these provenance scalars: `cadence_s: float`, `sg_window: int`, `sg_degree: int`, `sigma_v_px_per_frame: float` (the `np.std` used; NaN on degenerate), `noise_mask_k: float`, `is_degenerate: bool`. The `is_degenerate` flag + provenance scalars are a deliberate divergence from `ScaleogramResult`/`RidgeResult` (which carry only data): an all-NaN `MidlineResult` + explicit flag is the correct degenerate output for a per-track reconstruction primitive (an exception would force every per-track caller to wrap try/except).

**The function SHALL handle invalid vs. degenerate-but-valid inputs with a split policy:**

- RAISE (field-named, CC-1): `x`/`y` not an ndarray (`TypeError`); not 1-D, complex/object/non-numeric dtype, `len(x) != len(y)`, or any non-finite (NaN/±inf) value (`ValueError`) — non-finite is REJECTED, not dropped, because SG and `cumulative_trapezoid` assume uniform frame spacing; invalid `cadence_s` (value→`ValueError`, type incl. `bool`/`np.bool_`/str→`TypeError`); invalid `sg_window` (even / `≤ SG_DEGREE` / non-int). The non-finite check PRECEDES the degenerate gate, so a short all-NaN track RAISES (does not return graceful-NaN).
- GRACEFUL all-NaN `MidlineResult` (`is_degenerate=True`, NEVER raising, NEVER emitting `np.RuntimeWarning`): `n == 0` (length-0 arrays); `n < sg_window` (SG cannot apply); raw-stationary (`np.ptp(x) == 0 and np.ptp(y) == 0` on the RAW input — post-SG speed is floating-point dust, never exactly 0, so stationarity is detected pre-SG). The degenerate gate returns BEFORE any `np.std`/`np.hypot`/`cumulative_trapezoid` call (`np.std([])` warns; `cumulative_trapezoid([])` raises), with `n == 0` as the first short-circuit disjunct (`np.ptp([])` raises). Curvature is computed under `np.errstate(divide="ignore", invalid="ignore", over="ignore")` followed by a `κ[~np.isfinite(κ)] = np.nan` sweep, so no `inf`/`-inf` leaks and no RuntimeWarning is emitted on any path (including a deliberately huge-magnitude curvature input).

**The function SHALL be deterministic per CC-6:** all float arrays SHALL be bit-identical at `atol=0` across calls in the same process; `frame_indices` (int) and `is_degenerate` (bool) SHALL be exactly equal. Across Ubuntu / Windows / macOS CI runners the float arrays SHALL match to within `atol=1e-9, rtol=0` (the measured full-pipeline ULP-propagation floor for the well-conditioned savgol-lstsq + `cumulative_trapezoid` + `|v|³`-division stack is ≈1e-14; PR #6/#7's looser 1e-6 was a coverage argument for a 4-path scipy trait stack and does not transfer). A determinism canary script `scripts/circumnutation/capture_midline_canary.py` SHALL capture canary values from BOTH a closed-form analytic input (a pure circle radius `R`, where `κ ≡ 1/R` exactly — a self-evident oracle) AND `synthetic.generate_trajectory(random_state=0, n_frames=128, …)` (a drift detector), asserted within the cross-OS `atol=1e-9`.

**The function SHALL log exactly one `logger.debug` record** at the start (after validation) whose text begins with `"midline.reconstruct("` and contains the tokens `n_frames=` and `sg_window=`; no INFO/WARNING/ERROR records SHALL be emitted on the happy path (CC-9).

**Handoff to PR #9 (recorded so the seam is explicit).** PR #9's locked stub `spatial_cwt.compute_scaleogram(kappa, ds, constants=None)` takes a scalar grid spacing `ds`, presuming curvature on a uniform-`ds` grid. PR #8 deliberately emits `curvature_px_inv` on the native NON-uniform `arc_length_px` grid; PR #9 OWNS the resample `(MidlineResult.curvature_px_inv, MidlineResult.arc_length_px) → (kappa_uniform, ds)` (where the `ds`/spatial-Nyquist `NYQUIST_RATIO_MAX` decision lives). `arc_length_px` + `curvature_px_inv` is the complete, sufficient input for that resample.

**The system SHALL ALSO provide the new shared helper `_noise.compute_sg_derivative(x: np.ndarray, window: int, polynomial_order: int, deriv: int, delta: float = 1.0, mode: str = "interp") -> np.ndarray`:** a `scipy.signal.savgol_filter(x, window, polynomial_order, deriv=deriv, delta=delta, mode=mode)` wrapper reusing `compute_sg_detrended`'s window/`polynomial_order` boundary validation, AND additionally validating `0 ≤ deriv ≤ polynomial_order` (because `scipy` SILENTLY returns all-zeros for `deriv > polynomial_order` — a silent-wrong-answer hazard — and raises an opaque `factorial()` error for `deriv < 0`; the helper converts both into field-named errors). It SHALL return `np.full(len(x), np.nan)` (length-preserving) when `len(x) < window`. As part of this PR, `_noise.py` SHALL use module-qualified `scipy.signal.savgol_filter` (the existing bare `from scipy.signal import savgol_filter` import is converted) per the program's scipy-import discipline.

**The system SHALL ALSO provide the new shared helper `_geometry.compute_path_curvature(x_dot: np.ndarray, y_dot: np.ndarray, x_ddot: np.ndarray, y_ddot: np.ndarray) -> np.ndarray`:** the trajectory curvature `κ = (ẋ·ÿ − ẏ·ẍ) / (ẋ² + ẏ²)^{3/2}` (px⁻¹; theory §6.2), sibling to `compute_psi_g`/`compute_signed_area`. It SHALL raise `ValueError` on length-mismatched inputs and SHALL set `κ = NaN` where `|v| = 0` (guarded — never `inf`, never `RuntimeWarning`). Its sign convention is **load-bearing** and SHALL be documented anchored on the FORMULA sign (NOT the frame-ambiguous word "left turn"): the formula is the standard y-up math curvature formula, so `+κ` is a left turn in y-up math axes and a clockwise/visual-right turn as displayed in the y-down image frame — the same anchor-on-the-sign discipline `_geometry.compute_signed_area` uses. **Cross-helper sign relationship (load-bearing — must be in the docstring to prevent a publication-trait inversion in PR #9/#10):** because the ψ_g family (`compute_psi_g`/`compute_signed_area`/`handedness`) uses the deliberately swapped `atan2(dx, dy)` argument order while `compute_path_curvature` uses the standard `(ẋÿ − ẏẍ)` formula, for the SAME physical loop `sign(κ) == −handedness` (e.g. a y-up-math-CCW circle gives `κ = +1/R` but `handedness = −1`). A consumer composing curvature chirality with `handedness` MUST account for this opposite polarity. The convention SHALL be pinned by an absolute hand-built-input test AND a cross-helper sign-consistency test (below), and theory.md §6.2 is patched in this PR with the y-down clarification + the `sign(κ) == −handedness` note (original "positive = left turn" wording preserved in an Appendix B correction note).

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
- **AND** the non-finite check precedes the length gate, so a `3 ≤ n < sg_window` all-NaN track RAISES (does not return graceful-NaN)

#### Scenario: midline.reconstruct degrades gracefully on degenerate-but-valid inputs without raising or warning
- **GIVEN** the following degenerate-but-valid inputs, each invoked under `warnings.simplefilter("error")`
- **WHEN** `midline.reconstruct(x, y, cadence_s=300.0)` is invoked with: (a) `n = 0` empty arrays; (b) `0 < n < sg_window` finite arrays; (c) a raw-stationary track (`x` and `y` all-constant, `n ≥ sg_window`)
- **THEN** each returns a `MidlineResult` with `is_degenerate == True`, per-frame arrays of length `n` filled with `NaN` (and `frame_indices = np.arange(n)`), and `sigma_v_px_per_frame` is `NaN`
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
- **AND** for a pure circle of radius `R` (`n = 128`) the interior `curvature_px_inv` recovers `1/R` to a LOOSE physical-accuracy tolerance (`atol ≈ 1e-3`; the SG-polynomial discretization of a sampled circle limits this to ~1e-4, NOT the reproducibility floor) — this is the closed-form ORACLE check
- **AND** SEPARATELY, the captured circle-canary and `synthetic.generate_trajectory(random_state=0, n_frames=128, …)` canary values match across Ubuntu / Windows / macOS to within the cross-OS REPRODUCIBILITY floor `atol=1e-9, rtol=0` (a different assertion from physical accuracy: runtime-value vs. captured-value, not vs. `1/R`)

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
