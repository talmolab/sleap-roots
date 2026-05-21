# Design: add-circumnutation-synthetic-generator

## Context

This is PR #4 in the circumnutation analysis program tracked by `docs/circumnutation/roadmap.md`. Foundation (PR #1) shipped contracts; Tier 0 (PR #2) shipped raw kinematic traits + the `_noise.compute_sg_residual_xy` helper; QC tier (PR #3) shipped track-level signal-quality traits + the `_noise.compute_d2_residual_xy` / `_noise.compute_msd_residual_xy` siblings. This PR replaces the `synthetic.generate_trajectory` NotImplementedError stub with a working implementation that produces tip trajectories with known parameters, enabling round-trip validation of Tier 0 + QC outputs and laying the foundation for the broader Layer-1 validation suite that PR #12 will assemble.

Theory anchors:

- `docs/circumnutation/theory.md` §3.5 (BM2016 Eq. 20 `atan2(dx, dy)` handedness convention)
- `docs/circumnutation/theory.md` §4 (Rivière 2022 Eqs. 1/3/4/5)
- `docs/circumnutation/theory.md` §4.4 (closed-form `Δφ = 2·ΔL·δ̇₀ / (ωR)`)
- `docs/circumnutation/theory.md` §4.7 (steady traveling-wave hypothesis)
- `docs/circumnutation/theory.md` §8 Layer 1 validation strategy (±5% for T, ±15% for spatial)
- `docs/circumnutation/preliminary_results_2026-05-07.md` §1, §3.4, §4.1, §4.3 (empirical anchors from plate 001 Nipponbare)

Cross-cutting concerns: **CC-6 (determinism, including across OSs)**, CC-2 (constants), CC-3 (pure-pixel emission), CC-9 (logging).

## Goals / Non-Goals

**Goals:**

- Implement `sleap_roots.circumnutation.synthetic.generate_trajectory(...)` matching the canonical signature locked by the foundation's Package layout requirement: `generate_trajectory(...)` with NO `px_per_mm`.
- Realize Rivière 2022 Eq. 4 in **parametric closed form** (D1) using user-facing aggregate parameters `amplitude_px`, `T_nutation_s`, `growth_rate_px_per_frame`, `noise_sigma_px` (D2). The Rivière 6-tuple correspondence is **documented** but the API does not expose individually-degenerate parameters.
- Return a `pd.DataFrame` matching the `CircumnutationInputs.trajectory_df` schema (D3): 8 row-identity columns + `frame` + `tip_x` + `tip_y`, directly consumable by `kinematics.compute` / `qc.compute`. Dtype contract locked explicitly.
- Honor the determinism contract (CC-6 / D5): `random_state` accepts `int | np.random.Generator | None`; same int seed → bit-identical `tip_x` / `tip_y` across runs AND across OSs on 64-bit platforms.
- Add 7 empirically-anchored default constants to `_constants.py` + `ConstantsT` (D4); bump `_CONSTANTS_VERSION` 2 → 3. Constants follow the existing UPPER_SNAKE convention (no `_DEFAULT_` infix).
- Land an 8-section test file `tests/test_circumnutation_synthetic.py` (D12) including round-trip validation through Tier 0 + QC using the shared `_noise` and `_geometry` helpers and a non-tautological reference-fixture agreement test that compares `generate_trajectory()` defaults against `kinematics.compute()` recomputation on the plate 001 `.slp` fixture.

### Layer-1 recovery scope (per theory.md §8 targets vs PR ownership)

theory.md §8 enumerates 5 Layer-1 recovery targets. PR #4 implements ground-truth synthesis only; trait recovery is staged across multiple tier PRs.

| §8 target | Synthesis (PR #4) | Recovery PR | Recovered trait |
|---|---|---|---|
| `T_nutation` | ✓ via `T_nutation_s` | PR #6 (Tier 1) | `T_nutation_median` |
| `L_gz` | NOT synthesizable at tip level (D2 degeneracy) | PR #9 (Tier 3b spatial CWT) | `L_gz_estimate` |
| `L_c` (= `ΔL` linear regime) | NOT synthesizable at tip level (D2 degeneracy) | PR #9 (Tier 3b spatial CWT) | `L_c_estimate` |
| handedness | ✓ via `handedness` kwarg | PR #7 (Tier 2 ψ_g) | `handedness` |
| `δ̇₀` | NOT synthesizable at tip level (D2 degeneracy) | PR #11 (Tier 4 parametric) | `delta_dot_0_estimate` |
| `λ_spatial = v_growth · T_nutation` (theory.md §4.7 traveling-wave) | ✓ automatic via closed-form (single-frequency lateral oscillation propagating at `v_growth`) — for plate-001 defaults: `λ ≈ 4.29/300 · 3333 ≈ 47.7 px` | PR #9 (Tier 3b spatial CWT) + PR #10 (Tier 3c `traveling_wave_residual`) | `λ_spatial`, `traveling_wave_residual` |

PR #4 covers 2 of 5 §8 targets at the synthesis end (T_nutation, handedness) and additionally satisfies §4.7's `λ_spatial = v · T` traveling-wave relation tautologically (the closed-form lateral oscillation is exactly a single spatial frequency travelling at `v_growth`); recovery is gated on Tier 1 / Tier 2 / Tier 3 / Tier 4 landing. PR #12 (`add-circumnutation-layer1-validation`) is the umbrella PR that exercises the full pipeline end-to-end once all tier PRs ship — including the `traveling_wave_residual` cross-tier check. PR #4's tests validate the trait paths that ARE available (Tier 0 + QC) and document the deferred targets in the design.

**Non-Goals:**

- **No literal Eq. 5 ODE forward integration.** The closed-form derivation in §4.4 already gives the apex angular sweep analytically; numerical integration would couple integration error into the tolerances we're trying to validate against. Deferred to a future PR (likely Phase 2) if cross-tier validation demands it.
- **No Rivière-named API (`L_gz_px`, `Delta_L_px`, `delta_dot_0_per_s`, `epsilon_dot_0_per_s`, `omega_rad_per_s`, `R_px`).** See D2 for the tip-level degeneracy argument. PR #12 will wrap the user-facing API with a Rivière-named translation layer when PR #9 / PR #11 land the spatial-CWT traits that DO recover `L_gz` and `ΔL` individually.
- **No multi-track / `n_tracks` batch parameter.** Each call produces one track. Callers wanting a plate of N tracks loop `pd.concat([generate_trajectory(track_id=i, random_state=seed_seq.spawn(N)[i]) for i in range(N)])`. Single-track semantics keep the determinism contract clear.
- **No spatial-CWT-validation scope.** L_gz / L_c recovery via `κ(s)` is PR #9 / PR #11's responsibility.
- **No closed-form-trajectory + small-numerical-κ-perturbation hybrid** (the §4.6 local-contraction wavelet-doubling signature). Out of scope.
- **No new test fixtures.** The reference-fixture agreement test (D12 §2.H, revised per scientific-rigor reviewer B3) uses the EXISTING plate 001 `.slp` fixture loaded by PR #2 / PR #3 reference-value tests. No new fixtures are added.

## Decisions

### D1. Parametric closed-form realization (not literal Eq. 5 ODE integration)

`docs/circumnutation/theory.md` §8 says *"Generate a tip trajectory by integrating Eq. 4.3 forward"*. The word "integrating" is loose: §4.4 already derives the closed-form `Δφ = 2·ΔL·δ̇₀ / (ωR)` by analytically integrating Eq. 5 (`∂κ⊥/∂t ≈ δ̇/R`) around the apex angular deflection. PR #4 uses that analytical result rather than performing the integration numerically.

Concrete closed-form (using image-y-down axes per theory.md §2.1; revised per scientific-rigor review B1 to interpret `noise_sigma_px` as the xy-quadrature sum target — see D2 / D10 E3):

```
t_i = i * cadence_s                                # for i in [0, n_frames)
omega = 2 * math.pi / T_nutation_s
phase_i = omega * t_i + initial_phase_rad
v_growth_per_s = growth_rate_px_per_frame / cadence_s
A_lat = amplitude_px / 2                            # half peak-to-peak transverse

# growth-axis unit vector and the standard 90° CCW rotation
u_g  = (cos(growth_axis_angle_rad), sin(growth_axis_angle_rad))
u_lat = (-u_g[1], u_g[0])

# closed-form trajectory (no noise)
tip_x[i] = x0_px + v_growth_per_s * t_i * u_g[0]
                  + A_lat * sin(handedness * phase_i) * u_lat[0]
tip_y[i] = y0_px + v_growth_per_s * t_i * u_g[1]
                  + A_lat * sin(handedness * phase_i) * u_lat[1]

# additive iid Gaussian noise — per-axis sigma = noise_sigma_px / sqrt(2)
# so that the QC tier's xy-quadrature noise estimators recover noise_sigma_px directly
if noise_sigma_px > 0:
    rng = np.random.default_rng(random_state)
    sigma_per_axis = noise_sigma_px / math.sqrt(2.0)
    tip_x += rng.normal(0.0, sigma_per_axis, n_frames)
    tip_y += rng.normal(0.0, sigma_per_axis, n_frames)
```

**Why per-axis σ = `noise_sigma_px / √2`** (scientific-rigor reviewer B1):

The QC tier's noise estimators (`_noise.compute_sg_residual_xy`, `compute_d2_residual_xy`, `compute_msd_residual_xy`) return the **xy-quadrature sum** `√(σx² + σy²)`, not per-axis σ. For iid Gaussian noise with per-axis σ on both x and y, the expected return is `√(σ² + σ²) = σ·√2`. The empirical anchor in preliminary_results §4.2 (`sg_residual_xy ≈ 1.83 px` on plate 001) is itself an xy-quadrature value. To make `noise_sigma_px` directly comparable to `sg_residual_xy` (i.e., to make `generate_trajectory(noise_sigma_px=1.83)` produce a trajectory whose QC `sg_residual_xy` recovers ~1.83), the per-axis draws must be scaled by `1/√2`. The parameter name `noise_sigma_px` then refers to the xy-quadrature target, consistent with the QC trait naming.

This is the round-trip-friendly interpretation. The alternative (`noise_sigma_px` = per-axis σ) would require all round-trip tests to compare against `noise_sigma_px · √2` — strictly correct but confusing because the parameter no longer maps 1:1 to the QC trait it's validated against.

**Alternatives considered:**

- *Literal Eq. 5 ODE forward integration.* Rejected. (a) Integration error couples into the tolerances we're trying to validate against (±5% T, ±15% spatial). (b) Computational overhead is gratuitous when §4.4 already gives a closed form. (c) The literal ODE pulls `L_gz` / `ΔL` / `δ̇₀` / `ε̇₀` / `R` into the API surface, but tip-only observation can't disambiguate them (D2). Future PR can add ODE integration if a spatial-CWT-validation use case demands it.
- *Closed-form base + small numerical κ(s,t) perturbation.* Rejected as scope creep. The local-contraction wavelet-doubling signature (§4.6) is a feature of the SIDE-VIEW projection in the Rivière paper; our pipeline's TOP-VIEW root-tip data has no equivalent projection.
- *Per-axis σ = `noise_sigma_px` directly.* Rejected per scientific-rigor B1: would require all round-trip tests to compare against `√2 · noise_sigma_px`, breaking the empirical-anchor intent (`noise_sigma_px = 2.0` should match plate 001's `sg ≈ 1.83`, NOT match `1.83 / √2 ≈ 1.29`).

### D2. User-facing aggregate parameters (not Rivière 6-tuple)

The closed-form derivation in §4.4 collapses 6 Rivière parameters (`L_gz`, `Delta_L`, `delta_dot_0`, `epsilon_dot_0`, `omega`, `R`) into **3 aggregate observables** at the tip-trajectory level:

| Tip observable | Rivière combination | Notes |
|---|---|---|
| `v_growth_per_s` (longitudinal apex speed) | `ε̇₀ · R` | Direct |
| `A_lat = amplitude_px / 2` (half peak-to-peak transverse) | `ΔL · δ̇₀ / ω` | Equal to `R · Δφ / 2` by Eq. 1 (small-angle) |
| `ω = 2π / T_nutation_s` | `ω` | Direct |

Infinite (`L_gz`, `ΔL`, `δ̇₀`, `ε̇₀`, `ω`, `R`) tuples produce the **exact same** tip trajectory. `L_gz` does not enter the trajectory at all (it's recoverable only from spatial `κ(s)`, which PR #9 builds). Exposing the Rivière 6-tuple would therefore mislead callers into thinking inputs that have no observable effect at the tip level still matter.

**Decision: expose user-facing aggregates.** `amplitude_px`, `T_nutation_s`, `growth_rate_px_per_frame`, `noise_sigma_px`. These map 1:1 to recoverable Tier 0 + QC traits (D12 §2.C, §2.D) so round-trip tests are mechanical:

| API parameter | Recovered by |
|---|---|
| `growth_rate_px_per_frame` | Tier 0 `v_long_signed_median_px_per_frame` (after rotation to growth axis) |
| `amplitude_px` | Tier 0 `angular_amplitude` via the **exact** relation `angular_amplitude_peak_to_peak = 2 · arctan(amplitude_px · ω / (2 · v_growth_per_s))` (NOT the small-angle approximation; see D10 E2 and scientific-rigor I1) |
| `T_nutation_s` | PR #6 `T_nutation_median` (NOT in PR #4 scope) |
| `noise_sigma_px` | QC `sg_residual_xy`, `d2_noise_xy`, `msd_noise_xy` directly (per-axis σ = `noise_sigma_px / √2` per D1; xy-quadrature recovers `noise_sigma_px`) |
| `handedness` | PR #7 `handedness` trait (NOT in PR #4 scope); validated indirectly in PR #4 via `_geometry.compute_psi_g` sign-of-mean-dψ_g/dt + cross-checked via curl-sign (D6) |

**Rivière correspondence (documented in design.md + module docstring):**

```
v_growth_per_s = epsilon_dot_0 * R
A_lat (length, px) = Delta_L * delta_dot_0 / omega = R * Delta_phi / 2    # Eq. 1 / §4.4
amplitude_px = 2 * A_lat = 2 * Delta_L * delta_dot_0 / omega
omega = 2 * pi / T_nutation_s
```

PR #12 (`add-circumnutation-layer1-validation`) wraps `generate_trajectory(amplitude_px=..., ...)` with a `from_riviere_params(L_gz, Delta_L, delta_dot_0, epsilon_dot_0, omega, R, cadence_s, ...)` helper. PR #4's job is only to provide the building block.

### D3. Output type: `pd.DataFrame` matching `CircumnutationInputs.trajectory_df` schema, with locked dtype contract

Returns a DataFrame with 11 columns. **Dtype contract locked explicitly** (architecture reviewer B1):

| # | Column | Dtype | Notes |
|---|---|---|---|
| 1 | `series` | `object` | populated from kwarg, never NaN in output |
| 2 | `sample_uid` | `object` | populated from kwarg, never NaN |
| 3 | `timepoint` | `object` | populated from kwarg, never NaN |
| 4 | `plate_id` | `object` | populated from kwarg, never NaN |
| 5 | `plant_id` | `int64` | populated from kwarg, must be non-bool int |
| 6 | `track_id` | `int64` | populated from kwarg, must be non-bool int |
| 7 | `genotype` | `object` | `np.nan` when kwarg is `None`; `object` dtype forced via `pd.Series([...], dtype=object)` so `df["genotype"].isna()` returns True (not `== "None"` literal string) |
| 8 | `treatment` | `object` | same convention as `genotype` |
| 9 | `frame` | `int64` | values `[0, n_frames)` strictly monotonic ascending; `frame.iloc[0] == 0` is invariant |
| 10 | `tip_x` | `float64` | pure-pixel per CC-3 |
| 11 | `tip_y` | `float64` | pure-pixel per CC-3 |

**Implementation note:** the output is constructed directly inside `synthetic.py`; it does NOT pass through `_io._build_per_plant_template_from_df` (that function builds a per-PLANT template, not per-FRAME data). The dtype contract above is enforced via explicit `pd.Series(..., dtype=...)` constructions at DataFrame assembly time.

**No new column constant.** The output column tuple is `ROW_IDENTITY_COLUMNS + REQUIRED_PER_FRAME_COLUMNS` (both imported from `_types`). No `_SYNTHETIC_OUTPUT_COLUMNS` is added (architecture reviewer N1).

**Rationale for direct DataFrame return.** Round-trip with the rest of the package:

```python
df = generate_trajectory(amplitude_px=10.0, ...)
tier0 = kinematics.compute(df)         # works directly
qc = qc.compute(df)                     # works directly
```

### D4. Seven new defaults in `_constants.py` + `ConstantsT`; `_CONSTANTS_VERSION` 2 → 3

**Naming convention** (architecture reviewer I5): drop the `_DEFAULT_` infix used in the original draft. The module-level constants ARE the defaults; the infix is redundant given the existing convention (`SG_WINDOW_SHORT`, `WORST_STEP_RATIO_MAX` carry no `_DEFAULT_`). Keep the `SYNTHETIC_` prefix for scope clarity (the existing constants are trait-named; these are tier-named; the new prefix is intentional and disambiguates `SYNTHETIC_CADENCE_S` from a hypothetical future "default cadence for production CSV loading").

Per CC-2 (*"every constant is overridable via the pipeline class init or per-call kwarg"*), this PR adds:

| Constant | Default | Source |
|---|---|---|
| `SYNTHETIC_T_NUTATION_S` | `3333.0` | Derr Sept-2025 pilot; preliminary_results §3.4 |
| `SYNTHETIC_AMPLITUDE_PX` | `10.0` | plate 001 detrended peak-to-peak lateral amplitude; preliminary_results §1, §4.3 |
| `SYNTHETIC_GROWTH_RATE_PX_PER_FRAME` | `4.29` | plate 001 mean longitudinal step (from `preliminary_results §4.1`); see derivation below |
| `SYNTHETIC_NOISE_SIGMA_PX` | `2.0` | theory.md §8 Layer 1 noise level; plate 001 SG-residual ≈ 1.83 |
| `SYNTHETIC_CADENCE_S` | `300.0` | plate 001 cadence (5 min) |
| `SYNTHETIC_N_FRAMES` | `575` | plate 001 frame count (47.9 hr) |
| `SYNTHETIC_GROWTH_AXIS_ANGLE_RAD` | `math.pi / 2` | image-y-down convention; root growing downward |

**Derivation of `SYNTHETIC_GROWTH_RATE_PX_PER_FRAME = 4.29` (revised per scientific-rigor reviewer B2):**

The original draft mixed "median total step" (6.93 px) with "mean longitudinal/lateral ratio" (1.56) and produced `5.83` via algebraic decomposition. That was methodologically inconsistent (median ↔ mean mixing) AND tautologically equal to the mean total step.

Corrected derivation: preliminary_results §4.1 reports `Mean longitudinal step ⟨Δᵍ⟩ = 4.29 px/frame` **directly**. Use the empirical anchor verbatim. No decomposition.

**2.H consistency check.** With this default, the synthesized trajectory's mean per-frame total step is `√(4.29² + 2.83² + 2.0²) ≈ √(18.4 + 8.0 + 4.0) = √30.4 ≈ 5.51 px`, where:

- `long = 4.29 px/frame` (the new default)
- `lat per frame ≈ A_lat · ω · cadence_s = 5 · 0.001886 · 300 ≈ 2.83 px/frame` (matches §4.1 mean lateral 2.75 within 3%)
- `noise contribution per frame ≈ noise_sigma_px = 2.0` (xy-quadrature target)

The empirical mean total step is `5.83 px/frame` (§4.1). Synthesized 5.51 is within ±10% of empirical 5.83 (`|5.51 - 5.83| / 5.83 = 5.5%`). The synthesized median total step will differ from the empirical median (6.93 px/frame) by ~20% because median > mean for Rayleigh-like step-magnitude distributions and our closed-form noise model is Gaussian, NOT the real-data step distribution. **2.H asserts agreement on mean total step, not median**, and uses ±15% tolerance to absorb the residual.

**Aspect-ratio cross-check** (round-1 scientific-rigor reviewer N4 + round-2 scientific-rigor reviewer I-R2.1 wording polish): the synth longitudinal:lateral ratio is `4.29 / 2.83 = 1.516`, within 3% of the empirical mean ratio `1.56` from prelim §4.1. The 3% agreement is non-trivial cross-validation rather than a tautological match: `mean_long` is set DIRECTLY by `SYNTHETIC_GROWTH_RATE_PX_PER_FRAME` (0% match by construction), but `mean_lat = A_lat · ω · cadence_s = 5 · (2π/3333) · 300 ≈ 2.83` derives jointly from `SYNTHETIC_AMPLITUDE_PX` (calibrated from prelim §4.3's detrended **peak-to-peak** lateral span) AND `SYNTHETIC_T_NUTATION_S` (calibrated from prelim §3.4's Derr-pilot CWT peak). The time-averaged absolute lateral step (`mean_lat = 2.83 px`) thus emerges from two parameters fit against ENTIRELY DIFFERENT statistics (peak-to-peak amplitude vs spectral period), and the 3% match against prelim §4.1's time-averaged `mean_lat = 2.75` confirms the calibration is internally consistent across estimators. Net: 2 independent calibrations (`amplitude_px`, `T_nutation_s`) reproduce a 3rd independently-measured statistic (mean lateral step) within 3%.

All 7 constants added to:

- `_constants.py` module-level UPPER_SNAKE constants
- `ConstantsT` typed override-bag (7 new fields)
- `_default_constants_snapshot()` (so `run_metadata.json` records the values in effect)
- The foundation spec Requirement: Module-level constants — extended with the 7 names

`_CONSTANTS_VERSION` bumps `2 → 3` per the version-sentinel contract.

**Alternatives considered:**

- *Mandatory parameters with no defaults.* Rejected. Makes every test verbose; loses the "calibrated to plate 001 by construction" affordance.
- *Function-signature literals (no `_constants.py` entry).* Rejected. Defaults wouldn't be snapshot'd into `run_metadata.json` provenance; wouldn't be `ConstantsT`-overridable; wouldn't be inspectable.
- *Keep `_DEFAULT_` infix.* Rejected per architecture I5: violates the existing UPPER_SNAKE convention.

### D5. `random_state` typing: `int | np.random.Generator | None`; cross-OS bit-identical determinism on 64-bit platforms (CC-6)

Idiomatic numpy: a single `np.random.default_rng(random_state)` call handles all three cases:

- `random_state: int` → seeded `Generator` (deterministic, bit-identical across runs and 64-bit OSs at numpy 1.17+ per the numpy compatibility guarantee — see citation below)
- `random_state: np.random.Generator` → used as-is (caller-managed seed propagation, useful for test fixtures)
- `random_state: None` → fresh non-deterministic `Generator` (default; caller opts into determinism by passing a seed)

**Determinism contract** (locked by spec; tested in 2.B):

- Same `int` seed → bit-identical `tip_x` / `tip_y` arrays across two calls (test 2.B.1).
- Same `Generator` object passed twice → second call produces different output (Generator advances state; documented behavior; test 2.B.2).
- `int` seed and `np.random.default_rng(seed)` produce identical output for the same `seed` (test 2.B.3).
- Different seeds produce different output (test 2.B.4 — minimum-of-many-frames assertion).
- `noise_sigma_px = 0` short-circuits the RNG draw entirely; output is deterministic regardless of `random_state`; the passed-in `Generator` state is UNCHANGED after the call (D11; test 2.B.5 + 2.B.5b — TDD reviewer I2).
- Cross-OS determinism: CI matrix (Ubuntu / Windows / macOS at Python 3.11) asserts identical output for a known seed.

**Numpy stability citation** (tightened per scientific-rigor I4): numpy commits to backward-compatible streams from a given BitGenerator under [NEP 19](https://numpy.org/neps/nep-0019-rng-policy.html) ("RNG policy"). PCG64 — the default BitGenerator for `np.random.default_rng()` since numpy 1.17 — is a pure-integer algorithm with no platform-dependent floating-point reduction; its output stream is bit-identical across 64-bit platforms. sleap-roots' supported CI matrix (Ubuntu / Windows / macOS x86_64 + Apple Silicon arm64) is 100% 64-bit; 32-bit platforms are not supported. If a future numpy 2.x release breaks PCG64 stability, test 2.B.1 (known-seed → known-first-3-values assertion) fails and we pin the numpy version in `pyproject.toml`.

### D6. Handedness: explicit `Literal[+1, -1]` parameter

`handedness: int` parameter; only `+1` or `-1` are valid. Default `+1` = counterclockwise per BM2016 §"Constant principal direction of growth" + `theory.md` §3.5 / §7.3.

**Why `handedness` is intentionally NOT a `ConstantsT` field** (issue-alignment reviewer I2): CC-2 says "every constant is overridable via the pipeline class init or per-call kwarg", but `handedness` is a discrete categorical (`{+1, -1}`) representing a science convention (BM2016 Eq. 20 + theory.md §3.5: "+1 = counterclockwise (left-handed in image frame)"), NOT a tunable calibration. Promoting `+1` to a `ConstantsT.SYNTHETIC_HANDEDNESS` field would invite incorrect overrides (e.g., a user setting `SYNTHETIC_HANDEDNESS=+1` globally could mask a synthesis-bug where the trajectory actually rotates CW). Per-call override via the explicit `handedness=` kwarg gives the same flexibility without the global-default footgun. The other ConstantsT-overridable parameters are continuous-valued calibrations (period, amplitude, growth rate, noise σ, cadence, n_frames, growth-axis angle) where global defaults make sense.

Implementation: the phase argument is multiplied by `handedness` (`sin(handedness * phase)`); flipping handedness flips the sign of the lateral offset's time derivative, which inverts the rotation direction observed in `_geometry.compute_psi_g(tip_x, tip_y)`.

**Convention chain (load-bearing for PR #7's `handedness` trait):**

- `handedness = +1` → `sin(phase)` increases at `t = 0+` → tip moves in `+u_lat` direction initially → in standard math (y-up) axes, this is a counterclockwise sweep around the apex propagation line
- `_geometry.compute_psi_g` uses `atan2(dx, dy)` per BM2016 Eq. 20 → for the CCW sweep, the unwrapped `ψ_g(t)` has positive mean derivative → `sign(mean(diff(psi_g))) = +1` (PR #7's handedness trait)
- Under image-y-down convention (theory.md §2.1), the CCW math sweep displays as clockwise on screen — this is a visual rendering detail, not a sign-convention bug
- **Cross-check** (scientific-rigor N1): test 2.E.2 ALSO computes a `_geometry.compute_psi_g`-independent curl sign `mean(dx[i]·d²y[i] - dy[i]·d²x[i])` over the noise-free trajectory and asserts the sign matches `handedness`. This guards against a future renaming of `_geometry.compute_psi_g`'s `atan2(dx, dy)` argument order silently inverting the convention.

**Alternatives considered:**

- *Encoded via sign of `omega` or `T_nutation_s`.* Rejected. T_nutation is naturally positive; mixing magnitude with rotation direction in one parameter is a footgun.

### D7. Single-track per call (not batch via `n_tracks`)

Each call to `generate_trajectory` produces ONE track. The output DataFrame has exactly one unique `(series, sample_uid, plate_id, plant_id, track_id)` 5-tuple. Multi-track usage is via caller-side composition. **Recommended idiom** (scientific-rigor N4 — `seed + i` produces correlated streams; spawn is better):

```python
seed_seq = np.random.SeedSequence(42)
child_seeds = seed_seq.spawn(6)
df_plate = pd.concat([
    generate_trajectory(track_id=i, plant_id=i, random_state=child_seeds[i], ...)
    for i in range(6)
], ignore_index=True)
```

Documented in the module docstring. The simpler `random_state=seed + i` pattern is also acceptable when statistical independence between tracks is not load-bearing (e.g., schema tests where the exact noise realization is irrelevant).

### D8. Strict input validation at the boundary (matching `CircumnutationInputs` rigor)

Every numeric input is validated for type, finiteness, sign, and bool-rejection. **Acceptance rule** (scientific-rigor N2): `isinstance(value, (int, np.integer))` for integer parameters; `isinstance(value, (float, int, np.floating, np.integer))` for float parameters (with subsequent `bool` rejection because `True/False` are `int` subclasses in Python).

| Parameter | Validation |
|---|---|
| `n_frames` | `int` or `np.integer`, ≥ 1, not bool |
| `cadence_s` | positive finite float-like, not bool; explicit `float`/`int`/`np.floating`/`np.integer` only — NO string coercion (see "documented divergence" below) |
| `T_nutation_s` | positive finite float-like, not bool |
| `amplitude_px` | non-negative finite float-like, not bool (0 is allowed — degenerate zero-amplitude trajectory) |
| `growth_rate_px_per_frame` | finite float-like, not bool (negative is allowed — apex moving backward along `-u_g`) |
| `noise_sigma_px` | non-negative finite float-like, not bool (0 short-circuits RNG draw per D11) |
| `handedness` | exactly `+1` or `-1` integer (`int`/`np.integer`; reject bool, reject other ints, reject floats) |
| `growth_axis_angle_rad` | finite float-like, not bool (any finite value accepted; rotation well-defined mod 2π) |
| `x0_px`, `y0_px` | finite float-like, not bool |
| `initial_phase_rad` | finite float-like, not bool |
| `random_state` | `int`/`np.integer`, `np.random.Generator`, or `None`. **Reject `np.random.RandomState`** (legacy API; modern Generator API only per D5) |
| `constants` | `None` or `ConstantsT` instance |
| `series`, `sample_uid`, `timepoint`, `plate_id`, `genotype`, `treatment` | `str` or `None` (`None` allowed; coerced to `np.nan` in output for genotype/treatment; coerced to literal-string-`"None"` rejected; other 4 raise if `None`) |
| `plant_id`, `track_id` | `int`/`np.integer`, not bool |

**Documented divergence from `CircumnutationInputs.cadence_s` coercion** (architecture reviewer I2): Unlike `CircumnutationInputs` which has an attrs converter for `cadence_s = "300"` (string → float), the synthetic generator does NOT coerce string inputs. A string `cadence_s` raises `TypeError` cleanly. Rationale: `CircumnutationInputs` is a downstream-data dataclass that may receive YAML/JSON-parsed inputs; `generate_trajectory` is a programmatic test fixture that should reject ambiguous types at the call site.

**This divergence is explicitly documented in the module docstring** ("Note: unlike `CircumnutationInputs.cadence_s`, `generate_trajectory`'s `cadence_s` does not accept string inputs. Pass `float(...)` explicitly if your inputs come from a config file.").

Every ValueError / TypeError names the offending field in its message. Test section 2.F asserts every failure mode via a parametrized table — see D12 §2.F structure.

### D9. Identity columns as kw-only defaulted per-column parameters (with explicit fixture-affordance trade-off note)

Each of the 8 row-identity columns is a kw-only parameter on `generate_trajectory`:

```python
series: str = "synthetic",
sample_uid: str = "synthetic_001",
timepoint: str = "synthetic",
plate_id: str = "synthetic",
plant_id: int = 0,
track_id: int = 0,
genotype: Optional[str] = None,         # → np.nan in DataFrame
treatment: Optional[str] = None,        # → np.nan in DataFrame
```

The output DataFrame's identity columns are constant-valued (one unique row-identity 5-tuple per call per D7). Callers override individual fields when assembling multi-track plates.

**Signature size rationale** (architecture reviewer I1): the resulting full signature has ~18 kw-only parameters. This is heavier than `kinematics.compute(trajectory_df, constants=None)` or `qc.compute(trajectory_df, constants=None)`, AND that is intentional: a fixture-builder API is structurally different from a tier-compute API. Tier compute reduces a many-frame DataFrame to a single per-plant row; the synthetic generator INVERTS that — it expands ground-truth physics parameters into a many-frame DataFrame. The expansion needs a knob per ground-truth dimension. Grouping `x0_px` + `y0_px` into a tuple, or the 8 identity columns into a Mapping, was considered (architecture I1 alternatives) and rejected: tuple/Mapping inputs push validation errors from the call boundary into DataFrame construction time, harder to debug. Per-kwarg-with-validation is the clearer affordance even at the cost of a longer signature. This trade-off is documented in the module docstring and design.md and is NOT the model for tier compute signatures.

### D10. Closed-form trajectory math equivalences (the implementation contract)

This section locks the **mathematical equivalences** that tests 2.C / 2.E will assert.

**E1. Longitudinal velocity recovery (revised per TDD reviewer B1).** The Tier-0 trait `v_long_signed_median_px_per_frame` is computed by `kinematics.compute` after PROJECTING the per-frame velocity vector onto the growth axis. `kinematics` INFERS the growth axis from the data via `(x_N - x_1) / |x_N - x_1|` (the net-displacement axis, NOT a PCA computation). For a synthetic trajectory with `amplitude_px = 0` (pure linear, no nutation), the inferred growth axis matches `growth_axis_angle_rad` EXACTLY (because the trajectory is a straight line) and `v_long_signed_median_px_per_frame` recovers `growth_rate_px_per_frame` to floating-point precision.

For a synthetic trajectory WITH nutation (`amplitude_px > 0`), the trajectory endpoints' net-displacement direction may differ from `u_g` slightly because the endpoints sample arbitrary phases of the nutation. Magnitude of the deviation: `arctan(A_lat · sin(phase_N) / (n_frames · growth_rate_per_frame))` — at plate 001 defaults this is `arctan(5 · 1 / (575 · 4.29)) ≈ arctan(0.002) ≈ 0.002 rad ≈ 0.13°`, negligibly small. So the recovered `v_long_signed_median_px_per_frame` ≈ `growth_rate_px_per_frame` within 0.5% even WITH nutation, but the test makes the contract explicit by using `amplitude_px = 0` for the exact-equality assertion (2.C.1) and `amplitude_px = 10` for the approximate test (2.C.2).

**E2. Angular amplitude recovery (revised per scientific-rigor I1 + TDD reviewer B2 — use exact formula).** When `noise_sigma_px = 0`, the trajectory's per-frame velocity direction satisfies:

```
ψ_g(t) = atan2(dx/dt, dy/dt)
```

For the closed-form trajectory:

```
dx/dt = v_growth_per_s · u_g[0] + A_lat · omega · cos(handedness · phase) · handedness · u_lat[0]
dy/dt = v_growth_per_s · u_g[1] + A_lat · omega · cos(handedness · phase) · handedness · u_lat[1]
```

Decomposing in the (u_g, u_lat) frame: longitudinal component = `v_growth_per_s` (constant); lateral component = `A_lat · omega · cos(...) · handedness` (oscillates between `±A_lat · omega`). The instantaneous angle of the velocity vector relative to `u_g` is `arctan(lateral / longitudinal)`, oscillating between `±arctan(A_lat · omega / v_growth_per_s)`.

**Peak-to-peak angular amplitude (exact, not small-angle):**

```
angular_amplitude_peak_to_peak = 2 * arctan(A_lat * omega / v_growth_per_s)
                                = 2 * arctan(amplitude_px * omega / (2 * v_growth_per_s))
```

For plate-001-matching defaults (using the corrected `growth_rate = 4.29`, `cadence_s = 300`, `amplitude_px = 10`, `T_nutation_s = 3333`):

```
omega = 2π / 3333 ≈ 0.001886 rad/s
v_growth_per_s = 4.29 / 300 ≈ 0.0143 px/s
A_lat * omega / v_growth_per_s = 5 · 0.001886 / 0.0143 ≈ 0.659
angular_amplitude_peak_to_peak = 2 · arctan(0.659) = 2 · 0.583 = 1.166 rad ≈ 66.8°
```

The small-angle approximation `amplitude_px · omega / v_growth_per_s ≈ 1.32 rad ≈ 75.6°` over-estimates by ~13%. The test must use the EXACT relation; ±5% tolerance is sufficient because both sides are deterministic in the noise-free case.

For a "genuinely small-angle" sanity test (2.C.2), use `amplitude_px = 1.0` (other defaults unchanged): `A_lat · omega / v_growth_per_s = 0.5 · 0.001886 / 0.0143 ≈ 0.066`; exact = `2 · arctan(0.066) ≈ 0.131 rad ≈ 7.5°`; small-angle = `0.132 rad`. The two agree to 0.7%, so a small-angle-formula test is robustly tight there.

**E3. Noise recovery (revised per scientific-rigor B1).** When `noise_sigma_px > 0` and per-axis σ = `noise_sigma_px / √2` is used (per D1), the SG estimator returns approximately `noise_sigma_px` (modulo the documented SG under-bias of ~5-10%). Test 2.D.1 asserts QC `sg_residual_xy ≈ noise_sigma_px` within ±15% (matches theory.md §8 spatial tolerance AND absorbs SG under-bias).

The other two estimators have different bias profiles:

- `d2_noise_xy`: theoretically unbiased on iid noise (no SG-detrend); ±10% tolerance suffices but ±15% kept for consistency.
- `msd_noise_xy`: applies SG-detrend AND computes MSD; PR #3's reference-value test (`tests/test_circumnutation_qc.py` §2C6) shows recovery of σ=2 maps to msd ≈ 1.9 (within ~5%). ±15% tolerance is comfortable.

Test 2.D uses ±15% across all three estimators.

**E4. Handedness convention.** When `handedness = +1` AND `noise_sigma_px = 0`, `_geometry.compute_psi_g(tip_x, tip_y)` returns an unwrapped `ψ_g(t)` with `mean(np.diff(psi_g)) > 0`. Test 2.E.1 uses `noise_sigma_px = 0` (TDD reviewer I4) for unambiguous sign. Test 2.E.2 asserts the same via an independent curl-sign calc (scientific-rigor N1).

**E5. Growth-axis reliability under defaults** (architecture reviewer B2). For plate-001-matching defaults:

```
D ≈ growth_rate_px_per_frame · n_frames = 4.29 · 575 ≈ 2467 px
sg_residual_xy ≈ noise_sigma_px = 2.0 px
K · sg_residual_xy ≈ 10 · 2.0 = 20 px
D >> K · sg_residual_xy  → gate does NOT fire → growth_axis_unreliable = False
```

The margin is 2467 / 20 = ~123×, well above the threshold. Even at `noise_sigma_px = 4.0` (one of the 2.D adversarial values), the margin is ~62×. The gate fires only when `D < 10 · sg_residual_xy`, which for plate-001 cadence and frame count would require `growth_rate_px_per_frame < 0.035` — outside the realistic empirical regime. Test 2.C.3 asserts `growth_axis_unreliable = False` for plate-001 defaults.

### D11. `noise_sigma_px = 0` short-circuits the RNG draw (Layer-1 caveat added)

When `noise_sigma_px == 0.0` (exact equality), the implementation SHALL NOT call `np.random.default_rng(random_state)` or `rng.normal(...)`. The output is purely the closed-form trajectory.

**Rationale and observable contracts:**

1. **Determinism is decoupled from `random_state` in noise-free mode.** Test 2.B.5 asserts `generate_trajectory(noise_sigma_px=0, random_state=None)` and `generate_trajectory(noise_sigma_px=0, random_state=42)` produce IDENTICAL output.
2. **Generator state preservation.** A caller passing `random_state=my_rng` and `noise_sigma_px=0` MUST get `my_rng` back in unchanged state. Test 2.B.5b explicitly captures `my_rng.bit_generator.state` before and after and asserts equality (TDD reviewer I2). Without this test, an implementation that does `rng = np.random.default_rng(random_state); if noise_sigma_px > 0: rng.normal(...)` passes 2.B.5 trivially (no draws consumed) but a buggier variant `_ = np.random.default_rng(random_state).standard_normal()` then guarding inside would fail 2.B.5b.
3. **Edge case.** `noise_sigma_px = 0.0` exact triggers the short-circuit; `noise_sigma_px = 1e-10` does NOT (continues to the RNG path).

**Layer-1-validation caveat** (scientific-rigor I3): theory.md §8 explicitly says *"Apply Gaussian noise at σ = 2 px to the resulting (x, y)"* — Layer-1 contract is pipeline-under-realistic-noise validation. The `noise_sigma_px = 0` mode is for **closed-form correctness tests only** (testing the trajectory math itself, not the pipeline's noise robustness). PR #12's Layer-1 parameterized suite SHALL use `noise_sigma_px > 0`. The module docstring includes this caveat: *"`noise_sigma_px = 0` produces noise-free trajectories useful for testing the closed-form math; Layer-1 pipeline validation tests SHOULD use `noise_sigma_px > 0` per theory.md §8."*

### D12. Test taxonomy: 8 sections mirroring PR #3's structure (revised per all 3 reviewers)

`tests/test_circumnutation_synthetic.py` is organized into 8 sections.

#### §2.A schema/structural (TDD reviewer I1 + arch N3 additions)

Tests:

- 2.A.1: column order matches `ROW_IDENTITY_COLUMNS + REQUIRED_PER_FRAME_COLUMNS`
- 2.A.2: dtypes match D3 contract: `frame.dtype == int64`, `tip_x.dtype == tip_y.dtype == float64`, `plant_id.dtype == track_id.dtype == int64`, string identity columns have `object` dtype
- 2.A.3: `df["frame"].iloc[0] == 0` AND `df["frame"].iloc[-1] == n_frames - 1` (frame indexing convention)
- 2.A.4: `df["plant_id"].equals(df["track_id"])` when kwargs default (plant_id and track_id both default to 0 — populated identically per foundation convention)
- 2.A.5: when `genotype=None`, `df["genotype"].isna().all() == True` AND `df["genotype"].dtype == object` AND no literal `"None"` strings appear
- 2.A.6: kw-only signature — positional call raises `TypeError`
- 2.A.7: `inspect.signature(generate_trajectory).parameters` does NOT contain `px_per_mm` (re-asserts foundation spec scenario)
- 2.A.8: row count equals `n_frames`
- 2.A.9 (edge — arch N3): `n_frames=1` produces a 1-row DataFrame; `kinematics.compute` on it emits a NaN trait row per its existing single-frame contract (no exception)
- 2.A.10 (edge — arch N3): `growth_axis_angle_rad = 5*π` (outside [-π, π]) — round-trips without error (rotation is well-defined mod 2π)

#### §2.B determinism (CC-6) (TDD reviewer I2 — explicit RNG-state preservation)

Tests:

- 2.B.1: same int seed → bit-identical output; known seed (e.g., 0) → known first-3-elements expected values (canary for numpy PCG64 stability — per D5 / R4)
- 2.B.2: same `Generator` passed twice in sequence produces DIFFERENT output (Generator advances state — documents behavior)
- 2.B.3: `int` seed and `np.random.default_rng(seed)` produce identical output
- 2.B.4: different seeds produce different output (assert `not np.allclose(tip_x_a, tip_x_b)`)
- 2.B.5: `noise_sigma_px=0` short-circuits — `generate_trajectory(noise_sigma_px=0, random_state=None)` and `(noise_sigma_px=0, random_state=42)` produce identical output
- **2.B.5b** (new, TDD I2): caller-passed `Generator` state UNCHANGED after `noise_sigma_px=0` call:
  ```python
  rng = np.random.default_rng(42)
  state_before = rng.bit_generator.state
  generate_trajectory(noise_sigma_px=0, random_state=rng, ...)
  state_after = rng.bit_generator.state
  assert state_before == state_after  # deep dict comparison
  ```

#### §2.C parameter recovery via Tier 0 (TDD reviewer B1 + B2 split into separate tests)

Tests:

- **2.C.1** (exact-equality variant — `amplitude_px = 0`): `generate_trajectory(amplitude_px=0, growth_rate_px_per_frame=4.29, noise_sigma_px=0, growth_axis_angle_rad=π/2)` → `kinematics.compute(df)["v_long_signed_median_px_per_frame"]` equals `4.29` exactly (or within `1e-10`). Pure-linear trajectory; growth-axis inference is exact.
- **2.C.2** (small-angle analytical — `amplitude_px = 1.0`): `generate_trajectory(amplitude_px=1.0, ...)` → assert recovered `angular_amplitude` equals `2 * arctan(0.5 * omega / v_growth_per_s) ≈ 0.132 rad` within ±5%. Small-angle and exact formula agree to <1% here.
- **2.C.3** (plate-001 sanity — `amplitude_px = 10.0`): `generate_trajectory()` with all defaults → assert recovered `angular_amplitude` equals `2 * arctan(5 * omega / v_growth_per_s) ≈ 1.17 rad` (exact formula, NOT small-angle) within ±15% (theory.md §8 spatial tolerance). Also assert `growth_axis_unreliable == False` (per E5 derivation).

Without the 2.C.1 / 2.C.2 / 2.C.3 split, the single assertion would conflate two regimes and fail CI as TDD reviewer B2 warned.

#### §2.D parameter recovery via QC (round-trip noise sanity) (scientific-rigor B1 fix)

With per-axis σ = `noise_sigma_px / √2`, the QC tier's xy-quadrature noise estimators recover `noise_sigma_px` directly.

Tests:

- 2.D.1: `generate_trajectory(noise_sigma_px=2.0, random_state=42)` → `qc.compute(df)["sg_residual_xy"]` ≈ 2.0 within ±15%
- 2.D.2: same for `d2_noise_xy` ≈ 2.0 within ±15%
- 2.D.3: same for `msd_noise_xy` ≈ 2.0 within ±15%
- 2.D.4: parametrize over `noise_sigma_px ∈ {1.0, 2.0, 4.0}` for all 3 estimators (9 ids); each within ±15%
- 2.D.5: `qc.compute(df)["track_is_clean"]` == True for `noise_sigma_px=2.0` defaults (clean synthetic with proper noise estimator agreement)

#### §2.E handedness sign convention (TDD reviewer I4 — use `noise_sigma_px=0` + scientific-rigor N1 cross-check)

Tests (both use `noise_sigma_px = 0` for unambiguous determinism):

- 2.E.1: `handedness=+1, noise_sigma_px=0` → `mean(np.diff(_geometry.compute_psi_g(tip_x, tip_y))) > 0`
- 2.E.2: `handedness=-1, noise_sigma_px=0` → `mean(np.diff(_geometry.compute_psi_g(...))) < 0`
- 2.E.3 (cross-check per N1): `handedness=+1, noise_sigma_px=0` → compute curl-sign `np.mean(np.diff(tip_x)[1:] * np.diff(np.diff(tip_y)) - np.diff(tip_y)[1:] * np.diff(np.diff(tip_x)))` and assert sign is consistent with `_geometry.compute_psi_g`'s sign. Locks the BM2016 Eq. 20 atan2 argument-order convention against future inversion.
- 2.E.4 (noise-robust variant): `handedness=+1, noise_sigma_px=2.0, n_frames=575` → sign of mean dψ_g is still +1 (systematic dominates noise at this n; documents noise robustness without making it a load-bearing assertion in 2.E.1).

#### §2.F validation / error path (TDD reviewer I5 — parametrize table)

Single parametrize over `(param_name, invalid_value, exception_type, match_pattern)` covering:

| param | invalid values |
|---|---|
| `n_frames` | `0`, `-1`, `True`, `1.5`, `"100"`, `np.nan`, `np.inf` |
| `cadence_s` | `0.0`, `-1.0`, `np.nan`, `np.inf`, `-np.inf`, `True`, `"300"` |
| `T_nutation_s` | same as cadence_s |
| `amplitude_px` | `-1.0`, `np.nan`, `np.inf`, `True`, `"10"` |
| `growth_rate_px_per_frame` | `np.nan`, `np.inf`, `True`, `"5"` |
| `noise_sigma_px` | `-1.0`, `np.nan`, `np.inf`, `True`, `"2"` |
| `handedness` | `0`, `2`, `-2`, `1.0`, `True`, `"+1"`, `None` |
| `growth_axis_angle_rad` | `np.nan`, `np.inf`, `True`, `"π/2"` |
| `x0_px`, `y0_px`, `initial_phase_rad` | `np.nan`, `np.inf`, `True`, `"0"` |
| `random_state` | `1.5`, `"42"`, `np.random.RandomState(0)` (legacy API) |
| `constants` | `dict()`, `"constants"`, `42` |
| `plant_id`, `track_id` | `1.5`, `True`, `"0"`, `np.nan` |
| `series`, `sample_uid`, `timepoint`, `plate_id` | `0`, `1.5`, `np.nan` (`None` accepted, but for these 4 we require str) |

Approximately 60 parametrize ids. Each asserts the correct exception type AND that the exception message names the offending field. Test 2.F.0 (separate, not in parametrize) asserts the kw-only signature reject by attempting `generate_trajectory(575)` and matching `TypeError`.

#### §2.G constants snapshot + ConstantsT override (TDD reviewer B3 — lock resolution-order)

Tests:

- 2.G.1: `_default_constants_snapshot()` contains the 7 new keys (`SYNTHETIC_T_NUTATION_S`, `SYNTHETIC_AMPLITUDE_PX`, `SYNTHETIC_GROWTH_RATE_PX_PER_FRAME`, `SYNTHETIC_NOISE_SIGMA_PX`, `SYNTHETIC_CADENCE_S`, `SYNTHETIC_N_FRAMES`, `SYNTHETIC_GROWTH_AXIS_ANGLE_RAD`) with the defaults from D4
- 2.G.2: `_CONSTANTS_VERSION == 3`
- 2.G.3: `ConstantsT(SYNTHETIC_AMPLITUDE_PX=20.0)` is constructible and `inst.SYNTHETIC_AMPLITUDE_PX == 20.0`; other fields default to module-level
- **2.G.4** (resolution-order — D13): explicit kwarg overrides constants override:
  ```python
  # Constants overrides default
  df1 = generate_trajectory(noise_sigma_px=0, constants=ConstantsT(SYNTHETIC_AMPLITUDE_PX=20.0))
  # explicit kwarg overrides everything
  df2 = generate_trajectory(amplitude_px=15.0, noise_sigma_px=0,
                            constants=ConstantsT(SYNTHETIC_AMPLITUDE_PX=20.0))
  # df1 has amplitude 20; df2 has amplitude 15
  ```
  Recover amplitudes via tier 0 angular_amplitude formula and assert.

#### §2.H reference-fixture agreement (revised per scientific-rigor B3 — agreement vs real plate, NOT hardcoded numbers)

Reuse the existing plate-001 `.slp` fixture loaded by PR #2 / PR #3 reference-value tests. Compute Tier 0 + QC traits on BOTH the real-data trajectory AND `generate_trajectory()` with defaults; assert agreement within ±15%:

```python
real_traj = load_plate_001_fixture()                   # existing helper from PR #2 tests
synth = generate_trajectory()                          # defaults

real_qc = qc.compute(real_traj)
synth_qc = qc.compute(synth)
assert abs(real_qc["sg_residual_xy"].median() - synth_qc["sg_residual_xy"].iloc[0]) \
       / real_qc["sg_residual_xy"].median() < 0.15

real_t0 = kinematics.compute(real_traj)
synth_t0 = kinematics.compute(synth)
# Mean total step ≈ √(long² + lat² + noise²) recovers correctly
# (median is NOT within ±15% — see D4 calibration note — so we anchor on mean)
real_mean_step = compute_mean_step(real_traj)
synth_mean_step = compute_mean_step(synth)
assert abs(real_mean_step - synth_mean_step) / real_mean_step < 0.15
```

This makes 2.H **non-circular**: the defaults are calibrated to plate 001's published statistics, AND the test independently recomputes those statistics on the real plate, AND asserts the synthetic matches the recomputation. A bug in either the empirical anchors or the closed-form math breaks the test.

Coverage target: 100% on `synthetic.py`. The TDD ordering (TDD reviewer N3) is:

1. Step 0: foundation-test migration (`STUB_MODULES` → 7, `STUBS_WITH_CONSTANTS_KWARG` split per arch I3, `_CONSTANTS_VERSION → 3`)
2. 2.A schema (RED for column existence)
3. 2.F validation (RED for ValueError raises)
4. 2.B determinism (RED for closed-form math)
5. 2.G ConstantsT override (RED for resolution-order)
6. 2.E handedness (RED for sign-of-mean-dψ_g)
7. 2.D noise round-trip (RED for QC integration)
8. 2.C parameter recovery (RED for kinematics integration)
9. 2.H reference fixture (RED for end-to-end calibration)

### D13. ConstantsT resolution-order contract (NEW, per TDD reviewer B3)

The signature in D8 / D9 shows `amplitude_px: float = ...` with module-level constant as default. But D4 says ConstantsT can override the default. To make ConstantsT-override propagate, the resolution-order MUST be:

1. **Call-site kwarg wins.** If `amplitude_px` is explicitly passed (not `None`), use that value.
2. **`constants` parameter overrides module-level default.** If `amplitude_px` is `None` AND `constants` is provided, use `constants.SYNTHETIC_AMPLITUDE_PX`.
3. **Module-level default.** If both above are `None` / absent, use `_constants.SYNTHETIC_AMPLITUDE_PX` directly (which is the same value `ConstantsT()` would return — `ConstantsT` field defaults to the module-level constant).

**Implementation pattern:**

```python
def generate_trajectory(
    *,
    amplitude_px: Optional[float] = None,        # None = "use constants or module default"
    T_nutation_s: Optional[float] = None,
    # ... etc for all 7 calibrated parameters
    constants: Optional[ConstantsT] = None,
    # ... fixed-default parameters (handedness, x0_px, y0_px, etc.) keep their literals
) -> pd.DataFrame:
    _c = constants if constants is not None else ConstantsT()
    amplitude_px = amplitude_px if amplitude_px is not None else _c.SYNTHETIC_AMPLITUDE_PX
    T_nutation_s = T_nutation_s if T_nutation_s is not None else _c.SYNTHETIC_T_NUTATION_S
    # ... etc
    # then proceed with the closed-form math
```

The 7 calibrated parameters use `Optional[float] = None` in the signature; the 8 identity columns and the 4 deterministic/geometric parameters (`handedness`, `x0_px`, `y0_px`, `initial_phase_rad`) keep their direct literal defaults because they're NOT ConstantsT-overridable. `growth_axis_angle_rad` IS ConstantsT-overridable (default in `_constants.py`), so it uses the `Optional[float] = None` pattern too.

**Sentinel discipline.** `None` is the resolution-order sentinel; do NOT extend any of these parameters to accept `None` as a "use no-value" semantic. The validation in D8 still applies after resolution.

**Documented in module docstring + design.md.** Test 2.G.4 explicitly exercises the kwarg-wins-over-constants path.

### D14. Growth-axis-unreliable safety margin (NEW, per architecture reviewer B2)

Documented in D10 E5. Test 2.C.3 asserts the contract.

## Risks / Trade-offs

**R1. Rivière 6-tuple is degenerate at the tip level.** Documented extensively in D2 + module docstring. **Mitigation:** PR #12 wraps with a Rivière-named translation helper once PR #9 / PR #11 land the spatial-CWT recovery of `L_gz` / `ΔL` individually.

**R2. Small-angle approximation is NOT used for the synthesis trajectory itself — only for the Rivière correspondence (D2).** The closed-form trajectory in D1 is exact at any amplitude. The **angular-amplitude prediction** in D10 E2 uses the EXACT relation `2·arctan(amplitude_px·ω/(2·v_growth_per_s))`. **Mitigation:** test 2.C.3 uses the exact relation; the small-angle approximation appears only in 2.C.2 (`amplitude_px=1.0`, deep in the small-angle regime).

**R3. Default `growth_rate_px_per_frame = 4.29` is anchored directly to preliminary_results §4.1 mean longitudinal step.** No mixed-statistic decomposition. **Mitigation:** the 2.H test anchors on mean total step (5.83 px) which is within ±10% of the synthesized 5.51 px. Median total step (6.93 px) is NOT used as a tolerance anchor because the synthesis's Gaussian-noise model produces a different step-magnitude distribution shape than the real data (which has tail-events).

**R4. Cross-OS bit-identical determinism depends on numpy's PCG64 stability guarantee on 64-bit platforms.** Sleap-roots CI matrix is 100% 64-bit; the guarantee holds. **Mitigation:** test 2.B.1 (known-seed → known-first-3-values canary) catches any future numpy regression. If PCG64 changes in a major release, we pin the numpy version in `pyproject.toml`.

**R5. SG estimator under-bias on synthetic data is ~5-10% per the `_noise.compute_sg_residual_xy` docstring.** Test 2.D uses ±15% tolerance to accommodate. **Mitigation:** documented in D10 E3.

**R6. 2.H reference-fixture agreement test depends on the plate-001 `.slp` fixture remaining available** (it's not yet in `tests/data/circumnutation_nipponbare_plate_001/` for PR #2/#3 — Elizabeth confirmed it IS there). **Mitigation:** explicit `if not fixture_path.exists(): pytest.skip(...)` guard so PR #4 doesn't hard-couple to a fixture that could be moved or deleted independently.

**R7. `mean(np.diff(ψ_g)) > 0` discriminator for `handedness=+1` may be sign-unstable for bounded oscillations** (round-2 scientific-rigor reviewer N-R2.3): the closed-form trajectory produces a velocity-direction angle `ψ_g(t) = atan2(vx, vy)` that oscillates between `±arctan(A_lat·ω/v_growth) ≈ ±0.58 rad` for plate-001 defaults — bounded, NOT winding. For an integer number of nutation periods over the 575-frame window, `mean(np.diff(ψ_g))` integrates to approximately zero by the fundamental theorem of calculus (`ψ_g[N-1] - ψ_g[0]` divided by `N-1`). For 575 frames at `cadence=300s, T_nutation=3333s`, the window contains `575·300/3333 ≈ 51.75` periods — non-integer, so the mean is small but nonzero. **Mitigation:** §3.7 canary-capture step SHALL empirically verify that `np.mean(np.diff(_geometry.compute_psi_g(default_synth_xy))) > 0` for `handedness=+1` AND `< 0` for `handedness=-1`. If either sign comes out near-zero (`|mean(diff)| < 1e-3`) or sign-unstable across random seeds, swap the discriminator in tests 2.E.1 / 2.E.2 to a robuster alternative: (a) `helix_signed_area = 0.5 * sum(tip_x[i] * tip_y[i+1] - tip_x[i+1] * tip_y[i])` (signed Shoelace area; positive for CCW); OR (b) `psi_g[1] - psi_g[0]` evaluated at small phase (initial sign matches first quarter-period of the nutation). The 2.E.3 curl-sign cross-check would also catch the issue and remains the canonical handedness check. PR #7's `handedness` trait will own the definitive sign discriminator; PR #4 just needs to produce trajectories with the right rotation direction. This risk is pre-existing in design.md D6/D10 E4 and not introduced by any of the OpenSpec-review reconciliation edits.

## Migration Plan

This PR is additive — no breaking changes to existing capabilities. Foundation, Tier 0, QC outputs unchanged.

**Foundation test migration (`tests/test_circumnutation_foundation.py`)** (architecture reviewer I3 — spell out the split-table refactor):

- Remove `("synthetic", "generate_trajectory", 4)` from `STUB_MODULES`. Parametrize-id count for `test_stub_module_imports_cleanly` (unchanged — every module imports cleanly regardless of stub status) is fine. Parametrize-id count for `test_stub_callable_raises_with_correct_pr` drops **8 → 7**.
- **Split `STUBS_WITH_CONSTANTS_KWARG` into two tables**:
  ```python
  STUBS_WITH_CONSTANTS_KWARG: list[tuple[str, str]] = [
      ("temporal_cwt", "compute_scaleogram"),
      ("psi_g", "compute_psi_g"),
      ("midline", "reconstruct"),
      ("spatial_cwt", "compute_scaleogram"),
      ("pipeline", "compute_traits"),
  ]
  # NEW: implementations that also accept constants= but don't raise NotImplementedError
  IMPLEMENTATIONS_WITH_CONSTANTS_KWARG: list[tuple[str, str]] = [
      ("kinematics", "compute"),
      ("qc", "compute"),
      ("synthetic", "generate_trajectory"),     # ← added by PR #4
  ]
  ```
  Existing `test_stub_accepts_constants_kwarg` keeps its `NotImplementedError`-asserting body but now parametrizes only over `STUBS_WITH_CONSTANTS_KWARG`. A NEW test `test_implementation_accepts_constants_kwarg` is added in PR #4 that parametrizes over `IMPLEMENTATIONS_WITH_CONSTANTS_KWARG` and asserts NO exception is raised when calling with valid args + `constants=ConstantsT()`.
- Add positive-call test for synthetic mirroring `kinematics.compute` and `qc.compute` "no longer raises NotImplementedError" scenarios.
- Extend `test_module_logger_is_namespaced` parametrize to include `synthetic` (already module, no actual code change — verify only).
- **Update `test_constants_version_is_2`** (current name after PR #3) to assert `_CONSTANTS_VERSION == 3`. Rename to `test_constants_version_is_3` for clarity.
- The existing scenario test that `inspect.signature(synthetic.generate_trajectory).parameters` does NOT contain `px_per_mm` (already passes; verify the new signature also omits `px_per_mm`).

**`_constants.py` changes:**

- Add 7 new UPPER_SNAKE module-level constants per D4 (no `_DEFAULT_` infix).
- Extend `ConstantsT` with 7 new fields, defaults sourced from the module-level constants.
- Extend `_default_constants_snapshot()` to include the 7 new keys.
- Bump `_CONSTANTS_VERSION` from `2` to `3`. Update the docstring on the constant to note PR #4's contribution.

**`docs/circumnutation/roadmap.md` updates** (after merge, in cleanup-merged step):

- Row PR #4: status checkbox `⬜` → `✅`. Add issue / PR cross-links.

**`docs/changelog.md` updates** (under `[Unreleased] / ### Added`):

- "circumnutation: synthetic trajectory generator (`sleap_roots.circumnutation.synthetic.generate_trajectory`) for Layer-1 validation; closed-form realization of Rivière 2022 Eq. 4; deterministic via `random_state` (CC-6); emits pure-pixel `tip_x`/`tip_y` matching `CircumnutationInputs` schema. 7 new defaults in `_constants.py` + `ConstantsT`; `_CONSTANTS_VERSION` 2 → 3."

**`sleap_roots/circumnutation/__init__.py`:**

- **No new re-exports.** `synthetic.generate_trajectory` follows the same convention as `kinematics.compute` / `qc.compute` — accessed via `from sleap_roots.circumnutation import synthetic`. Top-level `sleap_roots` re-exports stay limited to `CircumnutationInputs` and `convert_to_mm`.
- **Re-export decision documented but reconsidered at PR #12** (architecture I4): if downstream test code accumulates significant `from sleap_roots.circumnutation import synthetic` boilerplate, revisit the re-export decision in PR #12's scope.

## Follow-up Issues

None new from this PR. PR #12 (`add-circumnutation-layer1-validation`) is the natural follow-up; it already exists as roadmap row 12. If during impl the empirical anchors turn out to require recalibration outside the ±15% tolerance (R3), file an issue against PR #4's impl rather than blocking the merge.

**Cross-reference to open follow-ups** (issue-alignment reviewer N1):

- **#202 (K=10 sensitivity sweep)** — the synthetic generator's `growth_rate_px_per_frame` and `noise_sigma_px` parameters give the K-sensitivity sweep a controlled lever (vary growth rate and noise σ to span the `D / (K · σ)` regime). When #202 lands, it can directly compose `generate_trajectory(growth_rate_px_per_frame=...)` instead of fabricating synthetic tracks inline. PR #4's existence makes #202 substantially easier.
- **#205 / #206 / #207 / #208 (α / β / γ / δ — PR #3 follow-ups on QC thresholds and ±inf detection)** — PR #4 does not change the QC thresholds; the synthetic generator's noise round-trip tests (§2.D) use empirical ±15% tolerance, decoupled from the α/β/γ threshold-validation work.
- **#199 (Python 3.11 + uv modernization)** — independent of PR #4; no interaction.

**Stub-signature replacement note** (architecture reviewer I3): the current `synthetic.py:18-29` stub has a positional signature `(L_gz=None, Delta_L=None, delta_dot_0=None, ...)` that was forward-looking documentation only (the body raises `NotImplementedError` immediately; no caller exercises the parameters). PR #4 replaces this with a kw-only signature (`*,`) and a different parameter set (the user-facing aggregates from D2). This is NOT an API break because (a) the stub was non-functional (always raised), (b) the foundation tests only invoke `generate_trajectory()` with zero args (`tests/test_circumnutation_foundation.py:130-131`), and (c) the foundation spec scenario "synthetic.generate_trajectory has no px_per_mm parameter" survives unchanged because neither signature contains `px_per_mm`. The replacement is deliberate per the brainstorming D2 decision; no migration step is needed for downstream callers.

## Open Questions

None blocking. The biggest design-time uncertainties (Eq. 4 realization style + parameter style) were resolved by D1 + D2 during brainstorming; subsequent review-driven revisions (B1 noise interpretation, B2 calibration, B3 reference-fixture restructure, TDD B1-B3 test fixes) tightened the design without changing the architectural shape.

## Appendix: Critical-Review Reconciliation

This design.md incorporates findings from three parallel critical-review agents (scientific rigor, architecture, TDD-testability). Reconciliation entries below quote each BLOCKING and IMPORTANT finding and identify the design location addressing it.

### Scientific-rigor reviewer

- **B1 (BLOCKING)**: "`_noise` helpers return σ_xy (quadrature sum), not per-axis σ — round-trip claim in 2.D is mis-specified." → Addressed in D1 (per-axis σ = `noise_sigma_px / √2`), D10 E3, D12 §2.D.
- **B2 (BLOCKING)**: "D4's `5.83 px/frame` default is empirically miscalibrated; mixing 'median total' with 'mean ratio' is methodologically inconsistent. §4.1 reports mean longitudinal step = 4.29 px directly." → Addressed in D4 (default changed to 4.29; derivation rewritten with single empirical anchor; 2.H tolerance anchor changed to mean total step 5.83 not median 6.93).
- **B3 (BLOCKING)**: "D12 §2.H is a tautological regression." → Addressed in D12 §2.H (restated as agreement test vs `kinematics.compute(plate_001_real_data)` recomputation; no hardcoded number comparison).
- **I1 (IMPORTANT)**: "D10 E2 angular-amplitude prediction relies on broken small-angle relation." → Addressed in D10 E2 (replaced with exact `2·arctan(amplitude_px·ω/(2·v_growth_per_s))`), D12 §2.C (split into small-angle 2.C.2 + exact 2.C.3).
- **I2 (IMPORTANT)**: "D2 amplitude correspondence drops a factor." → Addressed in D2 (table row corrected; `A_lat = Delta_L · delta_dot_0 / omega` without the spurious `R`).
- **I3 (IMPORTANT)**: "D11 `noise_sigma_px = 0` short-circuit is scientifically problematic for Layer-1 validation." → Addressed in D11 (Layer-1 caveat added to docstring + design: PR #12 SHALL use `noise_sigma_px > 0`).
- **I4 (IMPORTANT)**: "D5's NumPy stability citation needs sharpening." → Addressed in D5 (NEP 19 cited; 64-bit-only scope explicit).
- **I5 (IMPORTANT)**: "Partial Layer-1 status implied but never stated." → Addressed in Goals section (new "Layer-1 recovery scope" table mapping each §8 target to its PR).
- **N1 (NIT)**: "Visual-CCW caveat needs cross-check." → Addressed in D6 + D12 §2.E.3 (independent curl-sign cross-check).
- **N2 (NIT)**: "isinstance(int, np.integer) not duck-typing." → Addressed in D8 (explicit acceptance rule).
- **N3 (NIT)**: "Numpy pin check." → R4 references `pyproject.toml`'s `numpy>=2.0,<3.0` pin.
- **N4 (NIT)**: "pd.concat seed correlation; recommend SeedSequence.spawn." → Addressed in D7 (canonical idiom uses `seed_seq.spawn(N)`).

### Architecture reviewer

- **B1 (BLOCKING)**: "Output schema dtypes under-specified." → Addressed in D3 (explicit dtype contract table).
- **B2 (BLOCKING)**: "`growth_axis_unreliable=False` claim needs explicit math." → Addressed in D10 E5 (derivation showing 123× safety margin).
- **I1 (IMPORTANT)**: "18+ kwargs too many; group or document trade-off." → Addressed in D9 (fixture-affordance trade-off documented; tuple/Mapping alternatives explicitly rejected with rationale).
- **I2 (IMPORTANT)**: "cadence_s coercion divergence is a footgun." → Addressed in D8 (module-docstring callout).
- **I3 (IMPORTANT)**: "STUBS_WITH_CONSTANTS_KWARG migration hand-wavy." → Addressed in Migration Plan (explicit split-table refactor with code block).
- **I4 (IMPORTANT)**: "__init__.py re-export rationale incomplete." → Addressed in Migration Plan (rationale spelled out; reconsidered-at-PR-12 note added).
- **I5 (IMPORTANT)**: "SYNTHETIC_*_DEFAULT_* naming violates convention." → Addressed in D4 (renamed without `_DEFAULT_` infix).
- **N1 (NIT)**: "No `_SYNTHETIC_OUTPUT_COLUMNS` constant — clarify." → Addressed in D3 ("No new column constant; output order is `ROW_IDENTITY_COLUMNS + REQUIRED_PER_FRAME_COLUMNS` (both imported from `_types`).").
- **N2 (NIT)**: "Growth-axis convention tension." → Addressed in D10 E1 (synthesis vs inference explicitly noted).
- **N3 (NIT)**: "Missing edge test cases." → Addressed in D12 §2.A.9 (n_frames=1) and §2.A.10 (growth_axis_angle outside [-π, π]).
- **N4 (NIT)**: "R3 mitigation reads weird." → R3 rewritten to use direct §4.1 anchor; no "if calibration is off" mitigation language remains.

### TDD-testability reviewer

- **B1 (BLOCKING)**: "2.C parameter recovery mathematically unsafe (PCA-inferred growth axis)." → Addressed in D10 E1 (kinematics infers via NET-DISPLACEMENT, not PCA; deviation is `arctan(0.002) ≈ 0.13°` for plate-001 defaults — negligible) AND D12 §2.C.1 (uses `amplitude_px=0` for exact-equality test, separating it from the nutating 2.C.3).
- **B2 (BLOCKING)**: "2.C angular_amplitude tautological / non-small-angle." → Addressed in D10 E2 (exact formula) + D12 §2.C (split into small-amplitude analytical 2.C.2 + plate-001 sanity 2.C.3 at ±15% tolerance).
- **B3 (BLOCKING)**: "2.G ConstantsT override semantics ambiguous." → Addressed in D13 (NEW decision section locking resolution-order: kwarg > constants > module-default; implementation uses `Optional[float] = None` sentinel for the 7 ConstantsT-overridable parameters).
- **I1 (IMPORTANT)**: "2.A schema assertions under-specified." → Addressed in D12 §2.A.3 (frame[0]==0), §2.A.4 (plant_id.equals(track_id)), §2.A.5 (genotype.isna() not literal "None").
- **I2 (IMPORTANT)**: "2.B.4 RNG-state-unchanged assertion missing." → Addressed in D12 §2.B.5b (explicit `bit_generator.state` before/after equality assertion).
- **I3 (IMPORTANT)**: "2.D per-estimator tolerance bands." → Addressed in D10 E3 + D12 §2.D (±15% across all 3 estimators; documented bias profile per estimator).
- **I4 (IMPORTANT)**: "2.E handedness needs `noise_sigma_px=0`." → Addressed in D12 §2.E (all 4 sub-tests explicit about `noise_sigma_px = 0`; noise-robust variant 2.E.4 separately).
- **I5 (IMPORTANT)**: "2.F validation needs parametrize structure." → Addressed in D12 §2.F (table of ~60 ids enumerated).
- **N1 (NIT)**: "2.H cross-OS fragility." → Not a real concern; design.md acknowledges (cross-OS BLAS variation is O(1e-6), well inside ±15%).
- **N2 (NIT)**: "Coverage tractable in ~30-40 tests." → Acknowledged.
- **N3 (NIT)**: "TDD ordering." → Addressed in D12 (explicit ordering 0 → 1 → ... → 8 documented).

---

## Appendix: OpenSpec /openspec-review Reconciliation (5 subagents)

After scaffolding proposal.md / tasks.md / specs.md, the 5-subagent /openspec-review pass surfaced additional findings. Reconciliation entries below quote each BLOCKING and IMPORTANT finding and identify the location addressing it.

### Spec quality & OpenSpec best practices reviewer

- **I6 (IMPORTANT)**: "Curl-sign cross-check (tasks 2.E.3) needs a spec scenario, not just a task." → Addressed by ADDING a new spec scenario "handedness sign agrees with independent curl-sign cross-check" to the "Synthetic trajectory generator" requirement.
- **I7 (IMPORTANT)**: "2.C.3 ±15% tolerance is loose enough that a small-angle-approximation bug (which is ~13% off) would pass." → Addressed by leaving 2.C.3 tolerance at ±15% per theory.md §8 spatial tolerance, AND keeping 2.C.2 as the small-angle (`amplitude_px=1.0`) regime where exact-and-approximation agree to <1% (so a small-angle bug shows up there at the tighter ±5% tolerance). The two-test split makes the contract discriminating.

### Code & architecture feasibility reviewer

- **I1 (IMPORTANT)**: "tasks.md §2.I.1 claim that `test_stub_module_imports_cleanly` parametrize is unchanged is wrong." → Addressed by rewriting §2.I.1 to acknowledge the parametrize-id drop in BOTH `test_stub_module_imports_cleanly` AND `test_stub_callable_raises_with_correct_pr`, and noting that synthetic's import coverage migrates to the new `test_implementation_accepts_constants_kwarg`.
- **I2 (IMPORTANT)**: "§2.I.2 STUBS_WITH_CONSTANTS_KWARG arithmetic is confusingly worded — synthetic was never in this list." → Addressed by rewriting §2.I.2 to clarify that `STUBS_WITH_CONSTANTS_KWARG` is unchanged; PR #4 only ADDS a new `IMPLEMENTATIONS_WITH_CONSTANTS_KWARG` table.
- **I3 (IMPORTANT)**: "Backward compatibility — existing stub has positional signature." → Addressed in this Reconciliation Appendix's "Stub-signature replacement note" under Follow-up Issues.
- **N1 (NIT)**: "tasks.md §2.G.4/§2.G.5 reference free variables `omega` and `v_growth_per_s` without defining them." → Addressed by adding explicit setup lines to §2.G.4 (with the formula `omega = 2π/T_nutation_s; v_growth_per_s = growth_rate/cadence_s`).
- **N2 (NIT)**: "random_state validation isinstance check." → Implementation guidance — addressed in tasks.md §3.5 import list note.
- **N5 (NIT)**: "§2.I.3 may be redundant." → Addressed by changing §2.I.3 to "verify only" (no edit required).

### Issue alignment reviewer

- **I1 (IMPORTANT)**: "§4.7 traveling-wave (λ = v·T) not mentioned." → Addressed by extending the Layer-1 recovery scope table with a `λ_spatial = v_growth · T_nutation` row showing the closed-form satisfies it tautologically for plate-001 defaults (λ ≈ 47.7 px), with recovery deferred to PR #9 / PR #10.
- **I2 (IMPORTANT)**: "SYNTHETIC_HANDEDNESS constant missing from D4 set — design.md should explicitly note why." → Addressed by adding a "Why `handedness` is intentionally NOT a `ConstantsT` field" paragraph to D6 (discrete categorical, science convention not calibration; per-call kwarg gives flexibility without global-default footgun).
- **N1 (NIT)**: "Cross-link #202 in design.md Follow-up Issues." → Addressed in the new "Cross-reference to open follow-ups" subsection under Follow-up Issues.
- **N3 (NIT)**: "§3.5 typo — `_SYNTHETIC_*` (leading underscore) vs public `SYNTHETIC_*` names." → Addressed by editing tasks.md §3.5 import list to use the correct public UPPER_SNAKE names without leading underscore.

### TDD & testing strategy reviewer

- **B1 (BLOCKING)**: "Tasks 2.I.1 / 2.I.2 misstate the current foundation-test state." → Addressed in the rewrite of §2.I.1 + §2.I.2 (see "Code & architecture" reviewer I1/I2 above). The HEAD-state inspection (STUB_MODULES = 8 entries with synthetic; STUBS_WITH_CONSTANTS_KWARG = 5 entries without synthetic; test_schema_version assertion at == 2 with PR #3 docstring) is now documented inline at the top of §2.I.
- **B2 (BLOCKING)**: "Test 2.I.4 docstring update missing." → Addressed by extending §2.I.4 to include the docstring update ("bumped in PR #3" → "bumped in PR #4").
- **B3 (BLOCKING)**: "1e-10 tolerance in test 2.C.1 might be tight." → Addressed by loosening §2.C.1 to `1e-9` with rationale (cross-platform BLAS rounding through the `(4.29/300)·(i·300)` reintroduction).
- **I1 (IMPORTANT)**: "2.B.1 canary first-3-values are deferred — RED phase test cannot exist with placeholder values." → Addressed by adding a "Canary purpose (not an oracle)" note to §2.B.1 explicitly distinguishing the regression-detector role from a correctness oracle, with cross-reference to design.md R4.
- **I2 (IMPORTANT)**: "Spec scenario 'Reference-fixture agreement' missing pytest.skip semantics." → Addressed by adding the skip semantics to the spec scenario.
- **I3 (IMPORTANT)**: "§2.G.1 should also re-assert PR #3 QC constants stay in snapshot (regression guard)." → Addressed by renaming §2.G.1 to `constants_snapshot_contains_7_new_keys_and_preserves_pr3_qc_keys` and adding the 4 QC-constant assertions.
- **I4 (IMPORTANT)**: "§2.B.5b should use `numpy.testing.assert_equal` instead of `==`." → Addressed by editing §2.B.5b to use `npt.assert_equal` with a `copy.deepcopy` of the state-before snapshot.
- **I5 (IMPORTANT)**: "2.C.2 magic-number walkthrough" → Reviewer self-corrected on re-check; tasks.md 2.C.2 prediction is correct (`2·arctan(amplitude_px·ω/(2·v_growth)) ≈ 0.131 rad`). No change needed.
- **I6 (IMPORTANT)**: "Tasks 2.F.4 missing `-np.inf`." → Addressed by adding `-np.inf` to the §2.F.4 invalid set.
- **I7 (IMPORTANT)**: "Coverage of `logger.debug` branches not enumerated." → Addressed by adding §2.A.11 (caplog-based test verifying no DEBUG emissions on the happy path and explicit coverage of any DEBUG branches the impl introduces).

### Scientific rigor & data integrity reviewer

- **N1 (NIT)**: "growth_axis_angle_rad = π/2 → +y screen direction not explicit in spec." → Addressed by adding the explicit `u_g = (0, 1)` ⇒ +y screen direction note to the spec scenario "Default call returns 575-row DataFrame".
- **N2 (NIT)**: "long_lat_ratio NaN behavior in test 2.C.1 not asserted." → Addressed by adding `pd.isna(... long_lat_ratio ...)` assertion to §2.C.1.
- **N3 (NIT)**: "noise_sigma_px semantic naming could be clearer." → Addressed in the §3.5 docstring requirement ("Pure-pixel + per-axis noise (σ = noise_sigma_px / √2)" subsection); will verify at implementation time.
- **N4 (NIT)**: "Aspect-ratio sanity not in D4." → Addressed by adding an "Aspect-ratio cross-check" paragraph to D4 noting the synth long/lat ratio (1.516) matches empirical (1.56) within 3%.
- **B1/B2/B3, I1-I5**: not raised at this stage (the earlier scientific-rigor reviewer raised these and they were addressed in the pre-scaffold design.md revision; this stage's reviewer confirmed they're correctly reconciled).

---

## Appendix: OpenSpec /openspec-review Reconciliation — Round 2

A second 5-subagent pass on the round-1-reconciled artifacts surfaced one BLOCKING and one IMPORTANT finding, plus several smaller alignment items. Reconciliation entries below.

### Spec quality reviewer (R2)

- **B1 (BLOCKING)**: "Tolerance inconsistency between spec scenario (`1e-10`) and tasks.md §2.C.1 (`1e-9`); edit was applied in one place but not the other." → Addressed by aligning the spec scenario "Recovered longitudinal velocity equals input growth rate in pure-linear case" to `< 1e-9` with the same rationale cross-reference, AND adding the `long_lat_ratio == NaN` assertion to the spec scenario (matching the earlier addition to tasks.md §2.C.1).
- **I1 (IMPORTANT)**: "§2.B.5b uses `copy.deepcopy` but no `import copy` in test-file imports list." → Addressed by adding tasks.md §2.1 "Test-file imports" subsection enumerating `copy`, `math`, `numpy`, `numpy.testing as npt`, `pandas`, `pytest`, plus the public `SYNTHETIC_*` constants.
- **I2 (IMPORTANT)**: "§2.G.4/§2.G.5 setup defines `omega` and `v_growth_per_s` locally; pytest tests don't share locals." → Addressed by adding tasks.md §2.1b `synthetic_setup` pytest fixture that returns a dict of derived constants; §2.G.4 / §2.G.5 take the fixture as a parameter.
- **I3 (IMPORTANT)**: "Curl-sign scenario combines `handedness=+1` AND `-1` in one scenario via `**AND**`; should split per OpenSpec convention." → Addressed by splitting into two scenarios: "handedness=+1 curl-sign agrees with ψ_g sign" and "handedness=-1 curl-sign agrees with ψ_g sign", each with self-contained GIVEN.
- **I4 (IMPORTANT)**: "Three handedness scenarios — relationship needs clarification." → Acknowledged; the three scenarios (positive sign / negative sign / curl-sign cross-check) cover non-overlapping ground (existence of the +/- sign vs the cross-check between two independent computations) — left as-is. The §2.E.3 parametrize over both handedness signs now makes the test-to-spec mapping 1:1.

### Code & architecture reviewer (R2)

- **All HEAD-state claims verified.** No BLOCKING or IMPORTANT findings.
- **N1 (NIT)**: "§5.9 wording '±0 parametrize-id loss' contradicts the +1 net." → Addressed in the §5.9 rewrite that now explicitly counts -2 + 3 = +1 net id.
- **N2 (NIT)**: "noise_sigma_px per-arg docstring text not explicitly enumerated in §3.5." → Addressed in the §3.5 docstring requirements list; implementation-time review will verify the exact wording.

### Issue alignment reviewer (R2)

- **All checks pass.** No BLOCKING or IMPORTANT findings. Roadmap row PR #4 wording matches; theory.md / prelim citations verified; all 6 referenced GitHub issues (#199, #202, #205, #206, #207, #208) confirmed open; arithmetic (`4.29 / 2.83 = 1.516`; `(1.56 − 1.516) / 1.56 = 2.8%`; `λ = 47.7 px`) re-derived correctly; reviewer-appendix has the right 5 sections.

### TDD reviewer (R2)

- **IMP-1 (IMPORTANT)**: "§2.A.11 caplog test conflates 'MUST assert' with 'MAY assert if emissions exist'; contradicts §5.1's `logger.debug` carve-out." → Addressed by splitting §2.A.11 into §2.A.11a (unconditional WARNING/ERROR-free) + §2.A.11b (conditional branch-cover any `logger.debug` calls introduced by §3.5), AND REMOVING the `logger.debug` carve-out from §5.1.
- **NIT-1 (NIT)**: "§3.5 import list over-constrained vs tier precedent." → Addressed by simplifying §3.5 to import only `ConstantsT` (matching `kinematics.py:71` / `qc.py:66`); SYNTHETIC_* constants are imported in the test file only (where they ARE load-bearing for analytical predictions in §2.G.4 / §2.G.5).
- **NIT-3 (NIT)**: "§2.E.3 only tests `handedness=+1` but spec asserts both signs." → Addressed by parametrizing §2.E.3 over `handedness ∈ {+1, -1}`; AND splitting the curl-sign spec scenario into two scenarios (one per sign).
- **NIT-2 (NIT)**: "§2.G.1 partial overlap with PR #3 archived snapshot test." → Acknowledged as additive value (values-check is new; PR #3's archive only asserts presence). No change.

### Scientific rigor reviewer (R2)

- **All round-1 N1/N2 items landed correctly.** No new BLOCKING/IMPORTANT findings.
- **I-R2.1 (IMPORTANT)**: "D4 'three independent statistics' overstates the constraint." → Addressed by rewording D4's aspect-ratio paragraph to clarify that 2 independent calibrations (`amplitude_px` from peak-to-peak, `T_nutation_s` from spectral peak) reproduce a 3rd independently-measured statistic (mean lateral step from time-averaged absolute steps) within 3% — explicit credit/discredit for what's tautological vs cross-validated.
- **N-R2.3 (NIT)**: "ψ_g sign discriminator may be sign-unstable for bounded oscillations." → Pre-existing subtle concern (not introduced by reconciliation). Addressed by adding **R7** to the Risks section AND extending §3.7 canary-capture to empirically verify the sign and swap to `helix_signed_area` or `psi_g[1] - psi_g[0]` if unstable. Existing §2.E.3 curl-sign cross-check is an independent verifier; PR #7 owns the definitive handedness trait. PR #4 just needs to produce trajectories with the right rotation direction.
