# Design: add-circumnutation-tier0-kinematics

## Context

The Tier 0 traits are the entry-point of the circumnutation pipeline's trait-emission chain. They are *not* spectral (no CWT, no Fourier) and *not* spatial (no midline). They are per-track scalar summaries of tip kinematics — what every downstream tier composes from. This design doc covers the choices that shape downstream PRs: the trait-set asymmetry between signed and absolute components, the growth-axis reliability gate, the cadence-independent emission convention, the shared-helper-module pattern, and the test-data strategy.

Detailed theory: `docs/circumnutation/theory.md` §7.1. Empirical anchor: `docs/circumnutation/preliminary_results_2026-05-07.md` §3.2 / §3.5 / §4.1.

## Goals / Non-Goals

**Goals:**

- Emit per-track scalar kinematic traits matching `theory.md` §7.1 — expanded from the doc's 7 listed traits to **9 + 1 flag** as explained in D1 below.
- Maintain the foundation's canonical signature `compute(trajectory_df, constants=None)` — no breaking-API change.
- Honor the pure-pixel + cadence-independent emission convention (CC-3): pipeline emits `px/frame`, downstream user composes `convert_to_mm()` and (eventually) a separate time-axis utility for `mm/hr` output.
- Establish DRY helper modules (`_noise.py`, `_geometry.py`) that PR #3 (QC) and PR #7 (`psi_g`) will compose with, so the SG-residual and `ψ_g`-computation formulas live in exactly one place each across the program.
- Provide a real-data fixture (Nipponbare plate 001 proofread `.slp`) that subsequent tier PRs (#5 Tier 1, #7 Tier 2, etc.) will reuse. PR #2 is the first tier to need this fixture; investing the ~362 KB Git-LFS commit now pays off downstream.

**Non-Goals:**

- No spectral analysis. `angular_amplitude` is computed as the peak-to-peak extent of unwrapped `ψ_g(t)` — elementary, no CWT.
- No midline reconstruction. The growth axis is defined per the prelim §3.2 net-displacement convention; we do not fit a curve to the trail.
- No mm-bearing output. Velocity columns are emitted as `_px_per_frame` only.
- No per-hour conversion utility (deferred to a later PR when a consumer needs it).
- No `msd_noise_xy` — third noise estimator stays in PR #3.
- No manual researcher-veto override of `growth_axis_unreliable` — the auto-gate is the only path in this PR.

## Decisions

### D1. Expand the trait set from 7 to 9 + 1 flag (signed AND absolute medians)

`theory.md` §7.1 literally enumerates 7 traits: `v_total_median`, `v_long_median`, `v_lat_median`, `long_lat_ratio`, `path_displacement_ratio`, `angular_amplitude`, `principal_axis_angle`. The convention for "v_long" and "v_lat" is implicit ("Same" referring to `v_total`, which is a magnitude). The prelim §4.1 reports mean longitudinal step as *signed* (`⟨Δ^g⟩ = 4.29`) and mean lateral step as *absolute* (`⟨|Δ^ℓ|⟩ = 2.75`) — an asymmetry that is biologically motivated (signed lateral averages to ≈ 0 by symmetry around the growth axis) but is the kind of subtle convention that bites users downstream.

**Decision:** emit BOTH signed and absolute medians for `v_long` and `v_lat`, with column names that make the convention explicit:

| Original §7.1 name | Emitted columns |
|---|---|
| `v_total_median` | `v_total_median_px_per_frame` (magnitude — only one reading makes sense) |
| `v_long_median` | `v_long_signed_median_px_per_frame`, `v_long_abs_median_px_per_frame` |
| `v_lat_median` | `v_lat_signed_median_px_per_frame`, `v_lat_abs_median_px_per_frame` |
| `long_lat_ratio` | `long_lat_ratio` (= `v_long_abs / v_lat_abs`) |
| `path_displacement_ratio` | `path_displacement_ratio` |
| `angular_amplitude` | `angular_amplitude` |
| `principal_axis_angle` | `principal_axis_angle` |

Plus the boolean flag `growth_axis_unreliable`.

The `long_lat_ratio` uses `abs / abs` because the `signed / signed` form is dominated by the near-zero signed-lateral denominator — it amplifies noise and produces ±huge or NaN almost always. The `abs / abs` form is what the prelim's 1.56 represents conceptually and what scientists actually want ("how directional is the growth"). If a downstream user wants any other ratio, they can compute it in one line from the underlying medians, which are all emitted.

**Audit of downstream consumers** (Tier 1 / Tier 2 / Tier 3 / Tier 4 / QC / aggregation): no downstream Tier or QC trait consumes `long_lat_ratio` (signed or unsigned) as an input. It is a Tier-0 reporting / diagnostic trait, not upstream-of-anything. Emitting only the abs version is safe.

**`theory.md` §7.1 will be updated** as part of this PR to enumerate all 9 traits explicitly with their sign conventions, keeping the doc as single source of truth.

### D2. Growth-axis reliability gate — owned by Tier 0, uses a local SG residual

Roadmap CC-5 specifies the gate: *"If `D < GROWTH_AXIS_RELIABILITY_K * sg_residual_xy` (recommend `K = 10`), set `growth_axis_unreliable = True` and set the rotation-dependent traits to NaN."* The literal reading is that PR #2 (this PR) does the NaN'ing, but `sg_residual_xy` is a trait emitted by PR #3 (QC) — circular dependency.

**Resolution:** Tier 0 computes its own SG residual *locally* for the gate, using the formula that PR #3 will use to emit the canonical trait. The shared formula lives in a new private helper module (D3 below), so both tiers compute identical values from identical constants. This keeps Tier 0 self-contained (consumable standalone, no runtime dependency on PR #3) and avoids the API gymnastics of passing `sg_residual_xy` as a parameter (which would break the foundation's locked signature `compute(trajectory_df, constants=None)`).

**Alternatives considered:**

- *Tier 0 takes `sg_residual_xy` as a parameter.* Rejected: breaks the canonical signature and forces every caller to either run PR #3 first or pass a sentinel. Anti-user during the PR #2 → PR #3 interim.
- *Tier 0 emits rotation-dependent traits unconditionally; PR #3 post-processes to NaN.* Rejected: leaves the trait CSV with silently-meaningless numbers between PR #2 and PR #3 merges. The "trait CSV is internally coherent and consumable standalone" invariant matters more than algorithmic elegance.

**Gate emission:** Tier 0 emits `growth_axis_unreliable` as a column on the per-plant DataFrame, so a downstream consumer of the trait CSV can tell *why* the rotation-dependent traits are NaN without running anything else. Roadmap CC-5 step 3 said "the QC tier emits growth_axis_unreliable as a bool flag" — we re-interpret this as "Tier 0 emits the flag; QC may compose with it but does not re-emit." This is the cleanest split: each trait is emitted by exactly one tier.

**`K = 10` provenance:** the roadmap recommends `K = 10` with no specific empirical anchor. It is a heuristic safety factor — at the prelim's typical noise floor of ~2 px, `K = 10` means net displacement must exceed ~20 px before rotation-dependent traits are emitted, which is roughly 4 minutes of healthy growth on the Nipponbare plate. The angle estimate has ~`1/SNR` variance; `K = 10` gives ~20-px headroom. Smaller `K` (e.g., 3) would fire only at SNR ≈ 1 — too lenient. Larger `K` (e.g., 30) would NaN legitimately low-growth plants. The value is documented as overridable via `ConstantsT.GROWTH_AXIS_RELIABILITY_K`; a follow-up sensitivity analysis on mutant data is reserved for a future issue.

### D3. New helper modules — `_noise.py` and `_geometry.py` for DRY across tier PRs

Two formulas in this PR will be reused by future tier PRs:

- **SG residual** — `_noise.compute_sg_residual_xy(x, y, window, degree) -> float`. Std of `(x_raw − x_smoothed)` and `(y_raw − y_smoothed)` summed in quadrature. Used by Tier 0 (gate value) and reserved for PR #3 (emitted as the canonical `sg_residual_xy` QC trait).
- **`ψ_g(t)`** — `_geometry.compute_psi_g(x, y) -> np.ndarray`. Per Bastien-Meroz 2016 Eq. 20 and `docs/circumnutation/theory.md` §3.5: `np.unwrap(np.arctan2(Δx, Δy))` over consecutive frames. **Argument order is `Δx` first, then `Δy`** — convention-critical. The reversed order `atan2(Δy, Δx)` would offset ψ_g by π/2 AND flip the sign of `mean dψ_g/dt`, silently inverting PR #7's `handedness` trait (`theory.md` §7.3 defines `+1 = counterclockwise`). Tier 0 emits `angular_amplitude = max(ψ_g) − min(ψ_g)`, which is invariant under offset/negation so its value is the same regardless of argument order — but the underlying ψ_g array reserved for PR #7 must use the canonical BM convention. The helper enforces this single source of truth.

Both helpers are private (underscore-prefixed) per the foundation's convention. They are not re-exported from `__init__.py`; only their callers (`kinematics.py`, future `qc.py`, future `psi_g.py`) import them directly. Foundation-style tests exercise the helpers' contract independently via direct import.

The foundation spec's "Package layout" Requirement enumerates 5 contract modules; this PR's spec delta MODIFIES that to 7 (adding `_noise.py` and `_geometry.py`). The "10 stub modules" count drops to 9 — `kinematics` is no longer a stub.

**Alternatives considered:**

- *Inline the formulas in `kinematics.py` and re-implement them in PR #3 / PR #7.* Rejected: textbook DRY violation. Two implementations of the same formula will drift.
- *Add the helpers to `_constants.py` adjacent to the constants they use.* Rejected: mixes constants with computation; the constants module is meant to be a passive data module.
- *Add the helpers to `_io.py`.* Rejected: `_io.py` is for I/O (CSV writing, sidecar JSON, run metadata gathering), not signal processing.

### D4. `_io.py` refactor — `_build_per_plant_template_from_df` private helper

The foundation provides `build_per_plant_template(inputs: CircumnutationInputs)` — drops duplicates on the 5-tuple, sorts, coerces dtypes. But the canonical signature for `kinematics.compute` is `(trajectory_df, constants=None)` — a raw DataFrame, not a `CircumnutationInputs` wrapper. Three resolution paths:

1. **Wrap inside compute** — construct a `CircumnutationInputs(trajectory_df=df, cadence_s=???)` purely to satisfy the API. Forces Tier 0 to take or invent a `cadence_s` it does not need. Rejected.
2. **Duplicate the row-identity logic in `kinematics.py`** — same fix will be needed in PR #3 (`qc.compute(trajectory_df, constants=None)`) and PR #11 (`parametric.compute(...)`). Rejected.
3. **Factor a private helper.** Inside `_io.py`:

   ```python
   def _build_per_plant_template_from_df(df: pd.DataFrame) -> pd.DataFrame:
       """Same drop-duplicates + sort + dtype-coerce logic, on a raw DataFrame."""
       # ... existing body from build_per_plant_template ...

   def build_per_plant_template(inputs: CircumnutationInputs) -> pd.DataFrame:
       """Public API unchanged."""
       return _build_per_plant_template_from_df(inputs.trajectory_df)
   ```

   Tier modules import `_build_per_plant_template_from_df` directly. Foundation contract unchanged.

We pick option 3. The spec ADDs a Requirement for the helper's existence and contract.

### D5. Algorithm details

**Per-track scalar pipeline** (one execution of this loop per unique `(series, sample_uid, plate_id, plant_id, track_id)` 5-tuple):

1. **Subset and clean (NaN-then-sort ORDERING IS LOAD-BEARING).** `subset = track_df.dropna(subset=["tip_x", "tip_y"]).sort_values("frame").reset_index(drop=True)`. **The NaN drop MUST happen BEFORE any `np.diff` or arithmetic** — otherwise `np.diff` would propagate NaN to two adjacent diffs per dropped row, polluting downstream `np.sum` calls (e.g., path-length `L = sum(|Δxy_i|)` uses `np.sum`, not `np.nansum`, after this step; the dropna preconditions guarantee no NaN reaches the `np.sum`). If `len(subset) < 2` post-drop, emit NaN for all 9 trait columns and `False` for `growth_axis_unreliable` (we cannot judge reliability without ≥ 2 frames), log at `DEBUG` naming the offending `(plant_id, track_id)`. Skip the remaining steps.

2. **Build velocity arrays — gap-aware.** Let `xy = subset[["tip_x", "tip_y"]].to_numpy()`, `frame = subset["frame"].to_numpy().astype(float)`. Compute:

   - `delta_xy = np.diff(xy, axis=0)` — shape `(n−1, 2)`
   - `delta_frame = np.diff(frame)` — shape `(n−1,)`
   - `velocity = delta_xy / delta_frame[:, None]` — shape `(n−1, 2)`, units `px/frame`

   This handles non-contiguous frame indices (track gaps) correctly: a 2-frame gap halves the velocity vs a contiguous step of the same magnitude, so the median is interpretable.

3. **Per-frame step magnitude** = `np.linalg.norm(velocity, axis=1)`. `v_total_median_px_per_frame = np.nanmedian(steps)`.

4. **Growth axis.** `D = np.linalg.norm(xy[-1] - xy[0])`. If `D == 0` exactly: skip the rotation step (all 6 rotation-dependent traits get NaN), set `principal_axis_angle = NaN`. Otherwise:
   - `u_g = (xy[-1] - xy[0]) / D` — unit vector along growth axis
   - `u_lat = np.array([-u_g[1], u_g[0]])` — perpendicular (90° CCW rotation in standard math axes; under the image-y-down convention from `theory.md` §2.1, this corresponds to a 90° clockwise rotation as seen on a screen). **Sign convention:** `v_lat_signed > 0` indicates motion in the `u_lat` direction. The choice of perpendicular orientation (CCW vs CW math) is symmetric for the abs and ratio traits; the signed lateral trait inherits the math-CCW convention.
   - `delta_long_per_frame = velocity @ u_g` — signed scalar per frame
   - `delta_lat_per_frame = velocity @ u_lat` — signed scalar per frame
   - `principal_axis_angle = atan2(u_g[1], u_g[0])` — note this is a STANDARD math `atan2(y, x)` of the growth-axis components in the image-frame; unrelated to ψ_g's BM-Eq.-20 convention from step 7 (different quantity, different formula).

5. **Local SG residual** via `_noise.compute_sg_residual_xy(xy[:, 0], xy[:, 1], window=SG_WINDOW_SHORT, degree=SG_DEGREE)`. If `len(subset) < SG_WINDOW_SHORT`, the formula degrades gracefully (the helper documents this behavior — see Requirement scenarios). Result is a single positive float in px.

6. **Reliability gate.** `growth_axis_unreliable = (D < GROWTH_AXIS_RELIABILITY_K * sg_residual_xy_local)`. When `True`: NaN the 6 rotation-dependent trait values computed above and overwrite `principal_axis_angle = NaN`.

7. **`ψ_g(t)` and `angular_amplitude`.** `psi_g = _geometry.compute_psi_g(xy[:, 0], xy[:, 1])` returns an `(n−1,)` array of unwrapped angles using the BM-Eq.-20 convention `unwrap(atan2(Δx, Δy))` (argument order `Δx` first; see D3). `angular_amplitude = np.nanmax(psi_g) - np.nanmin(psi_g)`. Rotation-invariant under offset and sign-flip — survives the growth-axis-reliability gate.

8. **`long_lat_ratio` and `path_displacement_ratio`.**
   - `v_long_signed = np.nanmedian(delta_long_per_frame)`
   - `v_long_abs = np.nanmedian(np.abs(delta_long_per_frame))`
   - `v_lat_signed = np.nanmedian(delta_lat_per_frame)`
   - `v_lat_abs = np.nanmedian(np.abs(delta_lat_per_frame))`
   - `long_lat_ratio = v_long_abs / v_lat_abs if v_lat_abs > 0 else np.nan`
   - `L = float(np.sum(np.linalg.norm(delta_xy, axis=1)))` — present-frame path length. `np.sum` (not `np.nansum`) is correct here because step 1's `dropna` precondition guarantees no NaN reaches this sum. Slightly under-estimates true L if gaps span real motion (documented limitation).
   - `path_displacement_ratio = L / D if D > 0 else np.nan`

9. **Emit row.** Append the 10 new columns to the per-plant template row for this track.

**Multi-track composition.** Group `trajectory_df` by the 5-tuple via `_io._build_per_plant_template_from_df`; for each track, run steps 1–9; merge results into the template. Output is a DataFrame with rows ordered by the foundation's stable sort (`series`, `sample_uid`, `plate_id`, `plant_id`, `track_id`).

### D6. Cadence-independent emission — defer time conversion to a downstream utility

`theory.md` §7.1 lists velocity traits in `mm/hr`. Under the pure-pixel contract, that becomes `px/hr` — which requires `cadence_s` to convert from per-frame to per-hour. The foundation's canonical signature `compute(trajectory_df, constants=None)` does NOT include `cadence_s`. Resolution paths:

- **Modify the canonical signature** to add `cadence_s`. Spec amendment with cascading consequences for PR #3, #11. Rejected for breaking-change cost.
- **Emit BOTH per-frame and per-hour columns**, doubling the velocity column count. Rejected for column bloat with no information gain.
- **(Chosen) Emit `px/frame` only; defer time conversion to a downstream utility.** Symmetric with the pure-pixel/`convert_to_mm` decision: keep the pipeline cadence-and-calibration-independent at emission time, compose conversions downstream. The eventual `convert_to_per_hour` utility will be a peer of `convert_to_mm` in `units.py` (or a sibling module), added when the first downstream consumer (likely PR #14 or #15) needs per-hour output.

**Codebase audit confirms zero conflicts** with downstream tiers:

- `PIPELINE_UNIT_VOCABULARY` already includes `px/frame`, `px/hr`, `px·hr⁻¹` — `px/frame` is a first-class member.
- `convert_to_mm` handles `_px_per_frame` → `_mm_per_frame` correctly without modification (the function operates on column suffixes, not specific time-unit values).
- `_io.write_per_plant_csv`'s unit-vocabulary validation accepts `px/frame`.
- Tier 4's `δ̇₀ = ω·R_px·Δφ / (2·ΔL)` formula cancels lengths via `R_px/ΔL_px` — unit-agnostic for `ω` (per-frame or per-hour both fine, as long as `ω` and `T` are consistent in the cross-tier consistency check).
- Layer 3 cross-tier consistency `λ_spatial ≈ v · T_nutation` is unit-consistent either way (`v` in `px/frame` × `T` in frame-count = `λ` in px; or `v` in `px/hr` × `T` in `hr` = `λ` in px).
- Derr's regression target (`T = 3333 s`) lives in `T_nutation_median` (`hr`) emitted by PR #6, independent of Tier 0's velocity-unit choice.

`theory.md` §7.1 will be annotated to clarify that listed `mm/hr` units are *post-conversion*; the pipeline emits `px/frame` and the user composes `convert_to_mm()` and (eventually) `convert_to_per_hour()` for human-readable output.

### D7. Test data strategy — synthetic + KitaakeX smoke + Nipponbare reference

Three layers, each catching a different failure mode:

- **Synthetic exact-value tests** (deterministic correctness). Hand-construct trajectories where every trait value is analytically known: straight-line constant-speed, pure-noise around origin, circular trajectory, NaN-row injection, frame-gap injection. These bit-exact-value tests are the foundation of correctness — they catch algorithmic bugs that real data cannot.
- **KitaakeX smoke test** (integration regression). Use the existing `tests/data/circumnutation_plate/plate_001_greyscale.tracked.slp` (Aug-2025 KitaakeX MOCK, 311 frames @ 10-min cadence). Load via `TrackedTipPipeline`, enrich the trajectory_df with the 4 row-identity columns it does not emit (`plate_id`, `plant_id`, `genotype`, `treatment`), construct `CircumnutationInputs`, run `kinematics.compute`. Assert structural correctness: row count = 6, column set = 8 row-identity + 9 traits + 1 flag, units sidecar covers every column with vocabulary values, no NaN in rotation-invariant traits, NaN-pattern in rotation-dependent traits matches `growth_axis_unreliable` exactly. No value-equality assertions — this is "does it run on real data?" coverage.
- **Nipponbare reference-value sanity test** (empirical anchor). Use the new `tests/data/circumnutation_nipponbare_plate_001/plate_001_greyscale.tracked_proofread.slp` (Sept-2025 Nipponbare MOCK, 575 frames @ 5-min cadence, prelim §4.1 source). Run my Tier-0 impl on it once during development; capture the 9 trait values per track; bake them into the spec scenario as expected median-of-6-tracks ± tolerance (recommend ±10% on median speeds, ±15% on ratios). The prelim's empirical means (`5.83`, `4.29`, `2.75`, `1.56`, `1.36`) provide order-of-magnitude validation but are not directly the tolerances — those come from the actual median-based computation.

The Nipponbare fixture is a one-time data investment with high return: PRs #5, #6, #7 will all want the same Nipponbare anchor (especially PR #6's Derr regression at `T = 3333 s`).

### D8. Out of scope

Listed in `proposal.md` "What this change does NOT do". Of particular note: **no `convert_to_per_hour` utility**. The orthogonal time-axis conversion ships when the first downstream consumer demands per-hour output, not eagerly. This minimizes PR #2's scope and follows the established pattern where conversion utilities accumulate in `units.py` as needed.

## Risks / Trade-offs

- **Input validation is intentionally narrow.** Tier 0 validates `trajectory_df` is a `pd.DataFrame` with the required columns, but does NOT validate per-row finiteness of `tip_x`/`tip_y` (`±inf` is permitted to propagate through trait computations) or detect duplicate `(track_id, frame)` rows. Rationale: SLEAP predictions never emit `±inf`, and `TrackedTipPipeline` guarantees unique `(track_id, frame)` by construction — the upstream contract is tight. PR #3 QC's `frac_outlier_steps` trait is the right place to detect data-corruption cases. This narrow validation matches the foundation's permissive style (`_types._validate_trajectory_df` documents that finiteness is a tier-PR concern). Documented as the spec Requirement "Tier 0 input-validation boundary."
- **Trait CSV column count doubles for velocities.** Five velocity columns instead of three (per-frame). Acceptable — explicit names eliminate sign-convention ambiguity, and column count remains far under any practical CSV-width limit.
- **`K = 10` is a heuristic without specific empirical justification.** Documented as such; overridable per-call via `ConstantsT`. Follow-up: real-data sensitivity sweep on mutant / low-growth plants once such data is available.
- **`path_displacement_ratio = L / D` under-estimates true L by missing wiggle inside frame gaps.** Documented as a known limitation in the trait docstring. The bias is small for typical SLEAP tracks (gaps are rare in proofread data; even 1–2-frame gaps contribute negligibly to the integrated path length).
- **`angular_amplitude` is a per-track peak-to-peak extent of `ψ_g(t)` — not the more careful spectral construction PR #7 will build.** Documented as the literal §7.1 reading: "peak-to-peak angular extent of `ψ_g(t)`." Tier 0's value is sensitive to outlier velocity vectors (one bad frame creates a large `ψ_g` swing); the cross-tier consistency check (PR #7's `T_psig_median` vs Tier 0's `angular_amplitude`) may flag noisy plants. This is by design — Tier 0 is intentionally minimalist.
- **Helper modules add 2 new `_*.py` files to the package.** Acceptable — both are private, both are scientifically central, and adding them now is materially cheaper than retrofitting after PR #3 and PR #7 land.

## Migration Plan

No migration needed. The foundation's stubs (`kinematics.compute(...) → NotImplementedError`) are replaced by working implementations; no caller has run them yet. The `_io.py` refactor is additive (private helper added; public function unchanged).

## Open Questions

None blocking. The `K = 10` value is open in the sense of "could be tuned with future data" but is documented as a heuristic safety factor and is overridable.

## Cross-references

- `docs/circumnutation/theory.md` — single source of truth for trait list and definitions (§7.1 updated as part of this PR).
- `docs/circumnutation/roadmap.md` — umbrella plan; CC-2, CC-3, CC-4, CC-5, CC-9 apply.
- `docs/circumnutation/preliminary_results_2026-05-07.md` — empirical reference for Nipponbare plate 001 (§3.2 conventions, §4.1 numbers).
- `openspec/specs/circumnutation/spec.md` — canonical capability spec post-foundation archive.
- `sleap_roots/circumnutation/_constants.py` — overridable defaults including `SG_WINDOW_SHORT`, `SG_DEGREE`, `GROWTH_AXIS_RELIABILITY_K`.
- `sleap_roots/tracked_tip_pipeline.py` — analogous per-track trait pipeline; convention precedent for unit declarations and trait DataFrames.
- `tests/data/circumnutation_plate/README.md` — fixture-provenance template for the new Nipponbare fixture.
