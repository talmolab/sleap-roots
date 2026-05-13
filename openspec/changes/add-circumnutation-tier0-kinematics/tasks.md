# Tasks for add-circumnutation-tier0-kinematics

TDD-ordered. Tests precede implementation per `superpowers:test-driven-development`. **Do not push commits from section 2 (red phase — tests only) without sections 3 and 4 (implementations) in the same push** — the new test file imports `_noise.py` and `_geometry.py` which sections 3.1 / 3.2 create, and asserts `kinematics.compute` returns a DataFrame which section 4.2 implements. The suite is expected to be red between 2.x and 4.x and the PR is one logical unit.

## 1. Fixture commit (no code dependency; can land first)

- [x] 1.1 Create directory `tests/data/circumnutation_nipponbare_plate_001/`.
- [x] 1.2 Copy `\\multilab-na.ad.salk.edu\hpi_dev\users\eberrigan\circumnutation\20250917_Suyash_Patil_CMTN_Nipponbare_0.8PG_GA4vsTZT\runs\run_20250917_201037\plate_001_greyscale.tracked_proofread.slp` to `tests/data/circumnutation_nipponbare_plate_001/plate_001_greyscale.tracked_proofread.slp`.
- [x] 1.3 Verify the file is staged via Git LFS — `git check-attr filter <path>` returns `lfs`. Confirm `.gitattributes` already covers `*.slp` (precedent: KitaakeX fixture).
- [x] 1.4 Synthesize `tests/data/circumnutation_nipponbare_plate_001/fixture_metadata.csv` — single row sourced from the NetApp `Plate001_META.csv`. Columns to populate (mirroring the KitaakeX fixture's CSV): `plant_qr_code="plate_001"` (legacy column name), `genotype="Nipponbare"` (renamed from `accesion`), `treatment="MOCK"`, `number_of_plants_cylinder=6` (legacy column name; reuses the existing convention even though this is a plate, not a cylinder), `timepoint=0`. Drop source columns not consumed by `Series.get_metadata` lookups.
- [x] 1.5 Write `tests/data/circumnutation_nipponbare_plate_001/README.md` mirroring `tests/data/circumnutation_plate/README.md` (precedent for the KitaakeX fixture):
  - Purpose section (reference prelim §1, §3.1, §4.1 as the analysis source)
  - Imaging geometry section (575 frames @ 5-min cadence, 47.9 h imaging window, 6 tracks, single-node skeleton — `r0` per foundation convention)
  - Acquisition context section (Experiment ID `20250917_Suyash_Patil_CMTN_Nipponbare_0.8PG_GA4vsTZT`; researcher; genotype `Nipponbare`; treatment `MOCK`; substrate `1/2 MS, 0.8% phytagel`)
  - Contents table with file sizes and LFS status (`plate_001_greyscale.tracked_proofread.slp` ~362 KB via Git LFS; `fixture_metadata.csv` <1 KB plain text; `README.md` plain text)
  - Conversion provenance section — include the literal source row from `Plate001_META.csv` as a code block (mirror the KitaakeX precedent at `tests/data/circumnutation_plate/README.md`), followed by the column-rename table mapping source columns to `fixture_metadata.csv` columns (legacy column-name caveat about `plant_qr_code` and `number_of_plants_cylinder` reused from KitaakeX README)
  - Known limitations section (per-frame metadata deferred to #186; HDF5 video file not shipped — same as KitaakeX)
  - Related issues section (epic #197; foundation #198 / PR #200; #163 column-renaming follow-up; #186 per-frame metadata)
- [x] 1.6 Verify `uv run python -c "import sleap_io as sio; labels = sio.load_slp('tests/data/circumnutation_nipponbare_plate_001/plate_001_greyscale.tracked_proofread.slp'); print(len(labels), labels.tracks)"` returns 575 frames and 6 tracks (sanity check).

## 2. Tests (write before implementation — TDD red phase)

- [x] 2.1 Create `tests/test_circumnutation_kinematics.py` with module docstring referencing the spec deltas and theory.md §7.1.

### 2.A — Trait set schema and structural tests (spec Requirement: Tier 0 raw kinematic traits)

- [x] 2.A.1 Test `compute_returns_per_plant_dataframe`: build a 6-track trajectory_df via the existing `valid_trajectory_df` pattern (from `test_circumnutation_foundation.py`), call `kinematics.compute(df)`, assert return is `pd.DataFrame` with exactly 6 rows.
- [x] 2.A.2 Test `output_columns_match_spec`: assert columns = the 8 row-identity columns (first, in declared order) + the 10 new columns (9 traits + `growth_axis_unreliable`) in the documented order. Hard-code the expected column order.
- [x] 2.A.3 Test `unit_columns_match_vocabulary`: assert every emitted trait column's unit string (derived from a sibling `default_units_for_template`-extended map) is in `PIPELINE_UNIT_VOCABULARY`. Cover: `px/frame` for the 5 velocity columns, `—` for the 2 ratios, `rad` for `angular_amplitude` and `principal_axis_angle`, `bool` for `growth_axis_unreliable`.
- [x] 2.A.4 Test `track_id_is_integer_and_plant_id_equals_track_id`: same invariants as the foundation tests, just on the Tier-0 output DataFrame.
- [x] 2.A.5 Test `output_sort_order_is_numeric`: identity columns sorted by the 5-tuple, `track_id` numeric (not lexicographic).
- [x] 2.A.6 Test `invalid_trajectory_df_raises_valueerror`: parametrize over non-DataFrame types (`None`, `[1, 2, 3]`, `{"frame": []}`, `np.array([1.0])`) — each MUST raise `ValueError` whose message mentions "DataFrame". Also assert `kinematics.compute(df_missing_tip_x)` raises `ValueError` whose message names `tip_x` explicitly (per spec scenario "Invalid trajectory_df raises ValueError"). Maps to the Requirement "Tier 0 input-validation boundary" and the spec scenario.

### 2.B — Synthetic exact-value tests (spec Requirement: Tier 0 raw kinematic traits / scenarios)

- [x] 2.B.1 Test `straight_line_track`: 100-frame track with `tip_x = frame`, `tip_y = 0` (constant unit horizontal velocity). Expected (within IEEE float tolerance):
  - `v_total_median_px_per_frame == 1.0`
  - `v_long_signed_median_px_per_frame == 1.0`
  - `v_long_abs_median_px_per_frame == 1.0`
  - `v_lat_signed_median_px_per_frame == 0.0`
  - `v_lat_abs_median_px_per_frame == 0.0`
  - `long_lat_ratio == np.nan` (denominator zero)
  - `path_displacement_ratio == 1.0` (exact — L = D = 99)
  - `angular_amplitude == 0.0` (psi_g is constant)
  - `principal_axis_angle == 0.0`
  - `growth_axis_unreliable == False`
- [x] 2.B.2 Test `straight_line_track_image_y_down`: 100-frame track with `tip_x = 0`, `tip_y = frame` (constant unit downward velocity, image-y-downward convention). Expected `principal_axis_angle == np.pi / 2` (matches the "roots grow downward = +π/2" docstring claim).
- [x] 2.B.3 Test `pure_noise_track`: 100-frame track with `tip_x, tip_y` drawn via `np.random.default_rng(0).normal(0, 1, size=(2, 100))` (explicit seed for cross-platform determinism — matches the spec scenario "Pure-noise track triggers gate" verbatim). Expected:
  - `growth_axis_unreliable == True` (D is small, ~few px, vs SG residual ~1 px → ratio < 10)
  - `v_long_signed_median_px_per_frame == np.nan` (rotation-dependent, NaN'd)
  - `v_long_abs_median_px_per_frame == np.nan`
  - `v_lat_signed_median_px_per_frame == np.nan`
  - `v_lat_abs_median_px_per_frame == np.nan`
  - `long_lat_ratio == np.nan`
  - `principal_axis_angle == np.nan`
  - `v_total_median_px_per_frame > 0` (NOT NaN — magnitude is rotation-invariant)
  - `path_displacement_ratio` is a finite number > 1 (NOT NaN unless D is exactly 0)
  - `angular_amplitude` is a finite number (NOT NaN — rotation-invariant)
- [x] 2.B.4 Test `circular_trajectory`: 100-frame track with `tip_x = cos(2π·t/100)`, `tip_y = sin(2π·t/100)` + a small `+x` drift so growth axis is well-defined. Expected:
  - `angular_amplitude ≈ 2π` within tolerance (psi_g sweeps through a full rotation)
  - `growth_axis_unreliable == False` (the drift gives D ≫ noise)
  - `v_total_median_px_per_frame ≈ 2π/100` (one revolution per 100 frames)
- [x] 2.B.5 Test `nan_rows_dropped`: 100-frame straight-line track (per 2.B.1 construction: `tip_x = frame * 1.0`, `tip_y = 0.0`) with 10 rows (selected via `np.random.default_rng(0).choice(100, 10, replace=False)`) having `tip_x = NaN`. Expected:
  - All trait values match the no-NaN-row case (2.B.1) within float tolerance — drop semantics validated.
  - **Explicit named assertion**: `path_displacement_ratio == 1.0` exactly (the load-bearing NaN-ordering invariant from spec scenario "NaN rows are dropped BEFORE diff (ordering is load-bearing)"; this proves that the impl dropna's BEFORE diffing and that `np.sum` of remaining step-magnitudes returns a finite value).
  - **Explicit named assertion**: `v_total_median_px_per_frame == 1.0` exactly (gap-aware diff normalizes the bigger jumps across dropped rows: `Δxy / Δframe = N / N = 1.0` for each remaining diff).
- [x] 2.B.6 Test `frame_gaps_handled`: straight-line track at velocity 1 px/frame, but frames `[40..50)` removed (10-frame gap). Expected `v_total_median_px_per_frame == 1.0` exactly — the gap-aware diff normalizes the big jump across frames 39 → 50 (Δframe = 11, Δx = 11) to a per-frame velocity of 1.0.
- [x] 2.B.7 Test `insufficient_frames_yields_nan`: track with only 1 frame. Expected: all 9 trait columns NaN, `growth_axis_unreliable == False`.
- [x] 2.B.8 Test `zero_displacement_yields_nan_for_ratio`: track that ends exactly where it started (closed loop). Expected: `path_displacement_ratio == np.nan`, `growth_axis_unreliable == True` (D = 0 < any positive multiple of SG residual).
- [x] 2.B.9 Test `signed_lateral_is_near_zero_for_circular`: same circular trajectory as 2.B.4. Expected `v_lat_signed_median_px_per_frame` is finite and `< 0.1 × v_lat_abs_median_px_per_frame` (the signed-lateral-≈-0 sanity check).

### 2.C — Growth-axis reliability gate tests (spec Requirement: Growth-axis reliability gate)

- [x] 2.C.1 Test `gate_fires_when_D_below_threshold`: trajectory with D = 5 px, simulated noise SG residual = 1.0 px → ratio 5 < 10 → `growth_axis_unreliable == True`.
- [x] 2.C.2 Test `gate_does_not_fire_when_D_above_threshold`: trajectory with D = 100 px, SG residual = 1.0 px → ratio 100 > 10 → `growth_axis_unreliable == False`.
- [x] 2.C.3 Test `gate_threshold_overridable_via_constants`: pass `ConstantsT(GROWTH_AXIS_RELIABILITY_K=5)`, trajectory with D = 5 px and SG residual = 1.0 → ratio 5 = 5*1 (boundary case is False — strict less-than per spec).
- [x] 2.C.4 Test `gate_uses_constants_dot_GROWTH_AXIS_RELIABILITY_K_default`: no constants override, default K=10, trajectory at the boundary D = 10 * residual confirms the default is applied.
- [x] 2.C.5 Test `rotation_invariant_traits_survive_gate`: trajectory where gate fires; assert `v_total_median_px_per_frame`, `path_displacement_ratio`, `angular_amplitude` are NOT NaN (unless their own divide-by-zero condition fires).
- [x] 2.C.6 Test `rotation_dependent_traits_NaN_when_gate_fires`: assert all 6 rotation-dependent columns are NaN when `growth_axis_unreliable=True`.

### 2.D — Helper module tests (spec Requirement: Tier 0 helper modules)

- [x] 2.D.1 Test `_noise.compute_sg_residual_xy` returns 0 for a perfect polynomial of degree ≤ SG_DEGREE.
- [x] 2.D.2 Test `_noise.compute_sg_residual_xy` recovers approximate σ on noisy data. **Per spec scenario "compute_sg_residual_xy recovers approximate σ on noisy data"**: use `x_smooth = np.linspace(0, 100, 1000)`, `y_smooth = np.zeros(1000)`, noise `np.random.default_rng(0).normal(0, 1.0, size=(2, 1000))`. With unit-σ on both x and y, the quadrature-sum target is `sqrt(σ_x² + σ_y²) = sqrt(2) ≈ 1.414`. Assert return is within `[1.0, 1.6]` (SG slightly under-estimates σ). **Tolerance aligned to spec scenario, not the looser ±20%.**
- [x] 2.D.3 Test `_noise.compute_sg_residual_xy` raises or returns NaN when `len(x) < window` (document and assert the chosen contract).
- [x] 2.D.4 Test `_geometry.compute_psi_g` returns the correct constant value for a straight-line track using the BM-Eq.-20 convention `atan2(dx, dy)`:
  - `x = np.arange(100, dtype=float)`, `y = np.zeros(100)` → velocity in +x direction; `atan2(dx=1, dy=0) = π/2`; assert all elements equal `math.pi / 2`.
  - `x = np.zeros(100)`, `y = np.arange(100, dtype=float)` → velocity in +y direction (image-down); `atan2(dx=0, dy=1) = 0`; assert all elements equal `0.0`.
- [x] 2.D.5 Test `_geometry.compute_psi_g` returns strictly monotonic unwrapped angles spanning ≈2π for a closed parametric circle. Use `t = np.linspace(0, 2π, 100)`, `x = cos(t)`, `y = sin(t)`. Under the BM convention `atan2(dx, dy)`, ψ_g for this parametric circle decreases monotonically (because the velocity-direction angle rotates by −2π over the revolution). Assertions: (a) sequence is strictly monotonic in some direction (don't bake in the direction; check via `np.all(np.diff(psi) < 0) or np.all(np.diff(psi) > 0)`); (b) total absolute span `abs(psi[-1] - psi[0])` is approximately `2π` within ±0.1.
- [x] 2.D.6 Test `_geometry.compute_psi_g` handles a phase-wrapping case correctly — input that crosses the ±π branch cut produces a continuous unwrapped output.
- [x] 2.D.7 Test `_geometry.compute_psi_g` returns an empty array when `len(x) < 2`.

### 2.E — `_io.py` template helper tests (spec Requirement: Per-plant template helper)

- [x] 2.E.1 Test `_build_per_plant_template_from_df` accepts a raw trajectory DataFrame (no `CircumnutationInputs` wrapper) and returns the same template as `build_per_plant_template(inputs)` for the same data — column-for-column equality.
- [x] 2.E.2 Test `_build_per_plant_template_from_df` enforces the same `track_id` integer constraint and the same 5-tuple-conflict constraint as the public wrapper — same `ValueError` messages.
- [x] 2.E.3 Test that `build_per_plant_template(inputs)` (the public foundation API) still returns the same output as before this PR's refactor — regression test on the foundation contract.

### 2.F — KitaakeX integration smoke test

- [x] 2.F.1 Test `kitaakex_smoke`: load `tests/data/circumnutation_plate/plate_001_greyscale.tracked.slp` via `TrackedTipPipeline`; enrich the resulting trajectory_df with the 4 missing identity columns (`plate_id="plate_001"`, `plant_id = track_id`, `genotype="KitaakeX"`, `treatment="MOCK"`); construct `CircumnutationInputs(trajectory_df=enriched_df, cadence_s=600.0)`; call `kinematics.compute(enriched_df)`. Assert:
  - return is a DataFrame with exactly 6 rows
  - columns are the 8 row-identity + 10 new
  - all rotation-invariant traits (`v_total_median`, `path_displacement_ratio`, `angular_amplitude`) are finite (no NaN)
  - NaN pattern in the 6 rotation-dependent columns matches `growth_axis_unreliable` exactly (for each row, all 6 are NaN iff the flag is True)
  - `v_total_median_px_per_frame` per-track values are all positive
  - **explicit units sidecar validation**: construct the per-column unit map (combine `default_units_for_template(template)` with the Tier-0 trait units; the test imports `_TIER0_TRAIT_UNITS` from `kinematics.py`) and assert `all(u in PIPELINE_UNIT_VOCABULARY for u in units.values())`. Also call `_io.write_per_plant_csv` to a `tmp_path` and confirm the sibling units.json validates round-trip.

### 2.G — Nipponbare reference-value sanity test

- [x] 2.G.1 Test `nipponbare_reference_values`: load `tests/data/circumnutation_nipponbare_plate_001/plate_001_greyscale.tracked_proofread.slp` via `TrackedTipPipeline`; enrich similarly (`plate_id="plate_001"`, `plant_id = track_id`, `genotype="Nipponbare"`, `treatment="MOCK"`); construct `CircumnutationInputs(trajectory_df=enriched_df, cadence_s=300.0)`; call `kinematics.compute(enriched_df)`. Assert per-track median values fall within tolerance ranges locked during impl (section 4.4 below):

  **Important: the prelim §4.1 numbers cited below are MEANS (`⟨s⟩=5.83`, `⟨Δ^g⟩=4.29`, `⟨|Δ^ℓ|⟩=2.75`, `⟨Δ^g⟩/⟨|Δ^ℓ|⟩=1.56`, `L/D=1.36`), NOT medians. Tier 0 emits MEDIANS. For step magnitude, prelim §4.1 actually tabulates median=6.93 vs mean=5.83 — a ~19% difference. So the prelim numbers are order-of-magnitude anchors, NOT direct tolerance bounds. The test tolerances below are anchored at the actual median values captured by running my impl on this fixture (section 4.4), with the prelim means used only as a sanity floor.**

  - `v_total_median_px_per_frame` ∈ [TODO_LOWER, TODO_UPPER] — value to be captured in section 4.4 (expected order-of-magnitude: prelim median=6.93 px); ±10% tolerance once locked
  - `v_long_abs_median_px_per_frame` ∈ [TODO_LOWER, TODO_UPPER] — value to be captured in section 4.4 (expected order-of-magnitude: prelim mean=4.29 px); ±10% tolerance once locked
  - `v_lat_abs_median_px_per_frame` ∈ [TODO_LOWER, TODO_UPPER] — value to be captured in section 4.4 (expected order-of-magnitude: prelim mean=2.75 px); ±10% tolerance once locked
  - `long_lat_ratio` ∈ [TODO_LOWER, TODO_UPPER] — value to be captured in section 4.4 (expected order-of-magnitude: prelim ratio-of-means=1.56); ±15% tolerance once locked
  - `path_displacement_ratio` ∈ [TODO_LOWER, TODO_UPPER] — value to be captured in section 4.4 (expected order-of-magnitude: prelim L/D=1.36); ±15% tolerance once locked
  - `growth_axis_unreliable == False` for all 6 tracks (healthy plate, well-conditioned growth axis)
  - `angular_amplitude > 0` for all 6 tracks (real nutation present)

  *Tolerances are locked during impl* (section 4.4 below) after one calibration run with a sanity-floor cross-check against the prelim means; the `TODO_*` placeholders are replaced with concrete bounds before the PR commits.

### 2.H — Run-the-suite

- [x] 2.H.1 Run `uv run pytest tests/test_circumnutation_kinematics.py` — expect tests to FAIL (ImportError on `_noise.py`, `_geometry.py`, or `NotImplementedError` from `kinematics.compute`). **TDD red phase confirmed.** Per project convention, do not push this commit alone — the implementation in section 4 must land in the same push.

## 3. Implementation — helper modules (TDD green phase, part 1)

- [x] 3.1 Create `sleap_roots/circumnutation/_noise.py`:
  - module-level `logger = logging.getLogger(__name__)`
  - public function `compute_sg_residual_xy(x: np.ndarray, y: np.ndarray, window: int, degree: int) -> float`
  - body: apply `scipy.signal.savgol_filter` to `x` and `y` separately; compute residuals; std of `(x - x_smooth)` and `(y - y_smooth)`; return `np.sqrt(std_x**2 + std_y**2)`
  - handle `len(x) < window` case per the test 2.D.3 contract — recommend: return `np.nan` and log a `DEBUG` message naming the case
  - Google-style docstring naming the formula source (Numerical Recipes / `theory.md` §7.6)
- [x] 3.2 Create `sleap_roots/circumnutation/_geometry.py`:
  - module-level `logger = logging.getLogger(__name__)`
  - public function `compute_psi_g(x: np.ndarray, y: np.ndarray) -> np.ndarray`
  - body: `dx = np.diff(x)`, `dy = np.diff(y)`; `psi = np.arctan2(dx, dy)` (**note argument order: `dx` first, then `dy` — matches Bastien-Meroz 2016 Eq. 20 / theory.md §3.5 verbatim; the reversed order `atan2(dy, dx)` would offset by π/2 AND flip the sign of `mean dψ_g/dt`, which would silently invert PR #7's `handedness` trait. This is the canonical convention.**); `np.unwrap(psi)`
  - returns shape `(n-1,)` (empty array when `n < 2`)
  - Google-style docstring referencing Bastien-Meroz 2016 Eq. 20 and theory.md §3.5; docstring SHALL explicitly state the argument order convention and warn against using `atan2(dy, dx)`

## 4. Implementation — `_io.py` refactor and `kinematics.py` (TDD green phase, part 2)

- [x] 4.1 In `sleap_roots/circumnutation/_io.py`, add private function `_build_per_plant_template_from_df(df: pd.DataFrame) -> pd.DataFrame` containing the body currently in `build_per_plant_template`. Change `build_per_plant_template(inputs)` to a thin wrapper: `return _build_per_plant_template_from_df(inputs.trajectory_df)`. Run the existing foundation tests to confirm no regression.

- [x] 4.1.5 Migrate the existing foundation tests `tests/test_circumnutation_foundation.py` to reflect `kinematics`'s reclassification from stub → implementation module (mandated by the MODIFIED Package layout requirement in this PR's spec delta):
  - Remove the `("kinematics", "compute", 2)` tuple from `STUB_MODULES` (line 27); the parametrize-id count for `test_stub_module_imports_cleanly` (line 112) and `test_stub_callable_raises_with_correct_pr` (line 119) drops from 10 → 9.
  - Remove the `("kinematics", "compute")` tuple from `STUBS_WITH_CONSTANTS_KWARG` (line 791); the parametrize-id count for `test_stub_accepts_constants_kwarg` (line 801) drops from N → N-1.
  - Extend the contract-module list in `test_module_logger_is_namespaced` (line 726) to include `kinematics`, `_noise`, `_geometry` so the logger-namespace assertion still covers the kinematics module (now an implementation module, not a stub) and the two new helpers.
  - Run `uv run pytest tests/test_circumnutation_foundation.py -v` — confirm all foundation tests still pass.
- [x] 4.2 Implement `sleap_roots/circumnutation/kinematics.py`:
  - imports: `logging`, `numpy`, `pandas`, `_constants`, `_geometry`, `_io._build_per_plant_template_from_df`, `_noise`
  - module-level `logger = logging.getLogger(__name__)` (already exists from the stub; verify)
  - module-level `_TIER0_TRAIT_COLUMNS: tuple[str, ...] = (...)` — explicit declared order of the 10 new columns
  - module-level `_TIER0_TRAIT_UNITS: dict[str, str] = {...}` — units sidecar mapping for the new columns
  - helper function `_emit_nan_row()` returning a dict of NaN/False values for the 10 columns (used when `len(subset) < 2`)
  - helper function `_compute_one_track(track_df, constants) -> dict[str, float]` implementing steps 1–9 from design.md D5
  - public function `compute(trajectory_df: pd.DataFrame, constants: Optional[ConstantsT] = None) -> pd.DataFrame`:
    0. **input validation (FIRST step)**: validate `trajectory_df` is a `pd.DataFrame` (else raise `ValueError` with message mentioning "DataFrame"); validate it contains the 8 row-identity columns + `frame`, `tip_x`, `tip_y` (else raise `ValueError` naming the missing column). Recommended impl: call `_types._validate_trajectory_df(self=None, attribute=None, value=trajectory_df)` to reuse the foundation's validator and inherit its message format. Maps to spec Requirement "Tier 0 input-validation boundary" and scenario "Invalid trajectory_df raises ValueError".
    1. resolve `constants` to module defaults if `None` (use `_constants.ConstantsT()`)
    2. `template = _build_per_plant_template_from_df(trajectory_df)`
    3. group `trajectory_df` by the 5-tuple; for each group, call `_compute_one_track(group, constants)`; collect a list of dicts
    4. construct a per-track DataFrame from the dicts, align to `template`'s index via the 5-tuple key; concatenate the 10 new columns onto `template`
    5. enforce column order: 8 row-identity columns first, then 10 trait columns in declared order
    6. return the per-plant DataFrame
  - replace the existing `NotImplementedError("PR #2 — see docs/circumnutation/roadmap.md")` body with the working implementation
  - Google-style docstring per pydocstyle, naming all 10 emitted columns and their units. **The docstring MUST include:**
    - An explicit "Coordinate convention" subsection naming the image-y-down convention from `theory.md` §2.1: *"This module uses image-space coordinates where y increases downward. As a consequence, `principal_axis_angle` for a root growing image-down reads as `+π/2` (positive y direction)."*
    - A "Sign conventions" subsection for `v_lat_signed`: *"`v_lat_signed > 0` indicates motion in the direction `û_lat = (−û_g[1], û_g[0])`, which is the 90° rotation of the growth axis in screen-axis (math) orientation. Under image-y-down, screen viewer perception may differ — refer to `theory.md` §2.1."*
    - A "Handedness" subsection cross-referencing the BM-Eq.-20 convention for ψ_g: *"`angular_amplitude` is invariant under the atan2 argument order, but `_geometry.compute_psi_g` (its underlying ψ_g time series) uses `atan2(dx, dy)` per Bastien-Meroz 2016 Eq. 20 / theory.md §3.5. PR #7's `handedness` trait depends on this convention."*

- [x] 4.3 Run `uv run pytest tests/test_circumnutation_kinematics.py -v` — all green except the Nipponbare reference test 2.G.1 (placeholders still `TODO_*`).
- [x] 4.4 Calibrate the Nipponbare tolerances:
  - Run `uv run python -c "..."` to invoke `kinematics.compute` on the Nipponbare fixture; capture the median-across-6-tracks for the 5 reference traits (`v_total_median_px_per_frame`, `v_long_abs_median_px_per_frame`, `v_lat_abs_median_px_per_frame`, `long_lat_ratio`, `path_displacement_ratio`).
  - **Sanity floor (BEFORE locking tight tolerances)**: assert each captured value falls within ±50% of the prelim §4.1 anchor (median=6.93 for total step; means 4.29, 2.75, 1.56, 1.36 for the others). If any captured value differs by more than ±50% from the prelim anchor, STOP and investigate before locking — this would indicate the impl disagrees with the published characterization.
  - **Once sanity-floor passes**, write the captured values as `value ± 10%` (medians) and `value ± 15%` (ratios) into test 2.G.1. Replace `TODO_*` placeholders with concrete numeric bounds.
  - Re-run test 2.G.1 — green.
- [x] 4.5 Run `uv run pytest tests/test_circumnutation_kinematics.py -v` — all green.
- [x] 4.6 Run `uv run pytest tests/ -x` — full suite green (no regression in foundation tests, tracked-tip-pipeline tests, etc.).

## 5. Docs and changelog

- [x] 5.1 Update `docs/circumnutation/theory.md` §7.1: replace the 7-row trait table with a 9-row table. Each row SHALL include:
  - Symbol column: the trait name as emitted in CSV (e.g., `v_total_median_px_per_frame`)
  - Units column: the pipeline-emitted unit (e.g., `px/frame`, `—`, `rad`) with a footnote: *"Pipeline emits `px/frame` per the pure-pixel + cadence-independent contract (CC-3). The previous `mm/hr` annotation in this column was the post-conversion form, applied downstream via `sleap_roots.circumnutation.units.convert_to_mm()` and a future `convert_to_per_hour()` utility."*
  - Description column with explicit sign convention (signed vs absolute median; `v_lat_signed ≈ 0` is a sanity check) and:
    - `angular_amplitude`: annotate as **"peak-to-peak `max(ψ_g) − min(ψ_g)` where `ψ_g` is the unwrapped velocity-direction time series per BM-Eq.-20 / §3.5; rotation-invariant under offset and sign-flip"**
    - `principal_axis_angle`: annotate as **"standard image-frame `atan2(y_N − y_1, x_N − x_1)` of the growth-axis vector; NOT the same as `ψ_g` from §3.5 (different formula, different quantity)"**
  - Source/anchor citation column: preserve existing Rivière 2022 / Bastien-Meroz 2016 / This-doc citations
- [x] 5.2 Add `growth_axis_unreliable` flag row to theory.md §7.1 (DECISION: place in §7.1 — Tier 0 owns the column emission per design D2; do NOT add to §7.6 table). Row should read: symbol `growth_axis_unreliable`, units `bool`, description *"True iff `D < GROWTH_AXIS_RELIABILITY_K × sg_residual_xy_local`; when True, Tier 0 NaN's the 6 rotation-dependent traits. Emitted by Tier 0 (PR #2), composed but not re-emitted by QC tier (PR #3)."*, source/anchor `roadmap.md` CC-5 + this PR's design D2. Separately, add a brief sentence to §7.6's preamble paragraph noting: *"`growth_axis_unreliable` is emitted by Tier 0 (§7.1); QC tier composes with it (e.g., as a clause in `track_is_clean`) but does not re-emit a duplicate column. See `openspec/changes/.../design.md` D2 for the rationale."*
- [x] 5.3 Update `docs/circumnutation/roadmap.md` CC-5 (Growth-axis edge case) to reflect the design D2 decision:
  - Change step 2 to NaN 6 traits (the original 4 plus the new `_signed` variants): `v_long_signed_median, v_long_abs_median, v_lat_signed_median, v_lat_abs_median, long_lat_ratio, principal_axis_angle`.
  - Change step 3 from *"The QC tier emits `growth_axis_unreliable` as a bool flag"* to *"Tier 0 (PR #2) emits `growth_axis_unreliable` as a bool flag on the per-plant trait DataFrame; QC tier (PR #3) composes with it (e.g., in `track_is_clean`) but does NOT re-emit a duplicate column. Rationale: avoids circular dependency where Tier 0's gate needs PR #3's `sg_residual_xy` value, and ensures each trait is emitted by exactly one tier. See `openspec/changes/add-circumnutation-tier0-kinematics/design.md` D2."*
- [x] 5.4 Add `docs/changelog.md` entry under "Added" (or equivalent — match the foundation's PR #200 entry style):
  - "Tier 0 raw kinematic traits emission (`sleap_roots.circumnutation.kinematics.compute`): 9 traits + `growth_axis_unreliable` flag; pure-pixel + cadence-independent emission; new `_noise.py` and `_geometry.py` helpers; new Nipponbare plate 001 test fixture. See `docs/circumnutation/theory.md` §7.1."

## 6. Verify (pre-PR-open gates)

- [x] 6.1 Run `uv run pytest tests/test_circumnutation_kinematics.py --cov=sleap_roots.circumnutation.kinematics --cov=sleap_roots.circumnutation._noise --cov=sleap_roots.circumnutation._geometry --cov-report=term-missing`. Target: 100% coverage on the three new/changed modules. **Explicit list of acceptable uncovered branches** (else exception list grows opaque over time): (a) DEBUG-level `logger.debug` calls when `len(subset) < 2` in `_compute_one_track` and when `len(x) < window` in `_noise.compute_sg_residual_xy` — both ARE testable via `caplog.at_level(logging.DEBUG)`, so prefer testing over excluding; (b) defensive `else: raise` guards that are unreachable by construction (none currently in spec; if any are added during impl, document inline and add `# pragma: no cover`). Any uncovered branch outside this enumerated list fails the gate.
- [x] 6.2 Run `uv run pytest tests/ -x` — full suite green.
- [x] 6.2.5 Run `uv run pytest tests/ --cov=sleap_roots --cov-fail-under=84` — confirm project-wide coverage does not regress below the 84% baseline established by the foundation PR. (Codecov tracks this; the local gate catches any regression introduced by the `_io.py` refactor or new helper modules pulling untested branches into the coverage denominator.)
- [x] 6.2.6 Parametrize-id sanity check: confirm `test_stub_callable_raises_with_correct_pr` now collects 9 parametrize cases (was 10), `test_stub_module_imports_cleanly` collects 9 (was 10), and `test_stub_accepts_constants_kwarg` collects N-1 (was N). Mention the expected pytest collection delta in the PR description so reviewers don't mistake the lower count for a bug.
- [x] 6.3 Run `uv lock --check` — confirm no dependency change snuck in (this PR adds no new dependencies; `scipy` is already in `[project.dependencies]` from PR #1).
- [x] 6.4 Run `uv run black --check sleap_roots tests` — passes.
- [x] 6.5 Run `uv run pydocstyle --convention=google sleap_roots/` — passes on the full sleap_roots scope (catches `__init__.py` drift).
- [x] 6.6 Run `uv run mkdocs build` — passes; `kinematics.compute` doc page renders with the full 10-column trait list visible in the docstring.
- [x] 6.7 Run `openspec validate add-circumnutation-tier0-kinematics --strict` — valid.

## 6.5 Scope expansion — `Series.get_tracked_tips` proofread dedup (`tracked-tip-pipeline` capability)

Pulled into PR #2 after Nipponbare calibration revealed that `Series.get_tracked_tips()` (PR #190) returns both `PredictedInstance` and user-corrected `Instance` for every proofread frame, violating its "one row per (track_id, frame)" docstring and propagating ±inf / NaN into every velocity-bearing Tier 0 trait. Per `docs/circumnutation/preliminary_results_2026-05-07.md` §3.1, user-corrected takes precedence. Spec delta lives at `openspec/changes/add-circumnutation-tier0-kinematics/specs/tracked-tip-pipeline/spec.md`.

- [x] 6.5.1 Add §13 to `tests/test_tracked_tip_pipeline.py` with three TDD-red tests against the Nipponbare proofread fixture:
  - §13.1 `test_get_tracked_tips_dedup_proofread_one_row_per_track_frame` — fixture sanity check (≥ 1 frame with both `PredictedInstance` AND user-corrected `Instance` for the same track), then `len(df) == 3450` (6×575) and `df.duplicated(subset=["track_id","frame"]).sum() == 0`.
  - §13.2 `test_get_tracked_tips_dedup_proofread_user_corrected_takes_precedence` — pick a (frame, track) where both types coexist and the xy differs by > 10 px; assert returned xy matches user-corrected, NOT predicted.
  - §13.3 `test_get_tracked_tips_kitaakex_non_proofread_no_change` — guards against the dedup logic accidentally dropping rows on non-proofread fixtures.
- [x] 6.5.2 Confirm RED: `uv run pytest tests/test_tracked_tip_pipeline.py::test_get_tracked_tips_dedup_proofread_one_row_per_track_frame tests/test_tracked_tip_pipeline.py::test_get_tracked_tips_dedup_proofread_user_corrected_takes_precedence -v` fails for the expected reason (3906 vs 3450, 2 rows vs 1).
- [x] 6.5.3 GREEN: refactor `sleap_roots/series.py::Series.get_tracked_tips` to two-pass — first pass collects every tracked instance + the set of (frame, track) keys with a user-corrected instance; second pass emits one row per instance unless it's a `PredictedInstance` shadowed by a user-corrected `Instance` at the same (frame, track). Preserves the existing same-type-duplicates contract (§6.14 in `tests/test_tracked_tip_pipeline.py`).
- [x] 6.5.4 Confirm GREEN: all three §13 tests pass; the existing `test_tracking_coverage_bounded_when_track_has_duplicate_frame` test (which exercises Instance+Instance dups) still passes.
- [x] 6.5.5 Remove the local workaround from `tests/test_circumnutation_kinematics.py::_load_and_enrich` — the helper now calls `Series.get_tracked_tips()` directly with no need for direct `sleap_io` loading. Test 2.G.1 still passes with the locked tolerances (transparent at trait-value level — the workaround had already produced the corrected median values that the test asserts).
- [x] 6.5.6 Write the spec delta at `openspec/changes/add-circumnutation-tier0-kinematics/specs/tracked-tip-pipeline/spec.md`. MODIFIED Requirement: paste the full original `Series.get_tracked_tips` requirement text from `openspec/specs/tracked-tip-pipeline/spec.md` lines 22-87; add two new bullets specifying the proofread dedup behavior + same-type-duplicate preservation; add three new scenarios (proofread dedup; non-proofread no-op; same-type duplicates preserved x2).
- [x] 6.5.7 Re-validate: `openspec validate add-circumnutation-tier0-kinematics --strict` passes.

## 7. PR open

- [ ] 7.1 Draft GitHub issue body to `c:\vaults\sleap-roots\circumnutation\github_issues\issue_add-circumnutation-tier0-kinematics.md` referencing epic #197, OpenSpec change-id, theory.md §7.1, prelim §3.2 / §3.5 / §4.1, CC-5. Show Elizabeth before posting (do NOT post unilaterally).
- [ ] 7.1.5 Draft a follow-up GitHub issue body to `c:\vaults\sleap-roots\circumnutation\github_issues\issue_growth_axis_k_sensitivity.md` proposing a sensitivity-sweep analysis of `GROWTH_AXIS_RELIABILITY_K` on real low-growth-mutant data (currently defaulted to `10` as a heuristic safety factor per design.md D2; no empirical anchor in prelim). The issue body should: (a) reference epic #197, (b) reference this PR's design.md D2 and CC-5, (c) note the value is overridable via `ConstantsT.GROWTH_AXIS_RELIABILITY_K`, (d) describe the sweep (test K ∈ {3, 5, 10, 15, 30} on the Nipponbare fixture and a future low-growth-mutant fixture). Show Elizabeth before posting.
- [ ] 7.2 Open PR with title "feat(circumnutation): Tier 0 raw kinematics (#<sub-issue>)" and a body cross-linking the epic, sub-issue, and OpenSpec change-id. Labels: `enhancement`, `circumnutation`, `multi-pr`. Body matches `.claude/commands/pr-description.md` template.
- [ ] 7.3 Push the branch. Confirm CI is green on Ubuntu / Windows / macOS at Python 3.11.

## 8. Post-merge

- [ ] 8.1 Run `/cleanup-merged` which delegates to `/openspec:archive`. The archive folds the ADDED requirements and the MODIFIED Package-layout requirement into `openspec/specs/circumnutation/spec.md` and moves this change folder to `openspec/changes/archive/YYYY-MM-DD-add-circumnutation-tier0-kinematics/`.
- [ ] 8.2 Update `docs/circumnutation/roadmap.md` row PR #2 → ✅ with the GitHub issue and PR numbers.
- [ ] 8.3 When PR #3's tracking sub-issue is later filed (sub-issue of #197), leave a comment cross-referencing this PR's `design.md` D2 so the PR #3 author honors the "Tier 0 owns `growth_axis_unreliable`; QC does not re-emit" convention. The roadmap.md CC-5 update from task 5.3 is sufficient mechanically, but a direct comment on the future tracking issue reduces the risk of the convention being missed during PR #3 scoping.
