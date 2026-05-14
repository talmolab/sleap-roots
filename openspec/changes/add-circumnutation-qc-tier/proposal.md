## Why

This is PR #3 in the circumnutation analysis program tracked by `docs/circumnutation/roadmap.md`. PR #1 (foundation) shipped contracts; PR #2 (Tier 0) shipped raw kinematic traits plus the shared `_noise.compute_sg_residual_xy` helper. This PR fills in the **QC tier** — per-track signal-quality flags defined by `docs/circumnutation/theory.md` §7.6 — so downstream tier and aggregation PRs (#6 Tier 1 onwards) can filter low-quality tracks before composition.

Theoretical foundation: `theory.md` §7.6 (QC trait table; methodological note on the two-estimator agreement and optional MSD-extrapolation). Empirical anchor: `docs/circumnutation/preliminary_results_2026-05-07.md` §3.3 (the two noise-estimator formulas) and §4.2 (plate 001 reference values: SG = 1.83 px median, d2 = 2.67 px median, `sg_d2_agreement ≈ 1.46`). Cross-cutting concerns addressed: CC-2 (constants — adds 4 new thresholds + bumps `_CONSTANTS_VERSION`), CC-3 (pure-pixel + cadence-independent — `qc.compute(trajectory_df, constants=None)` emits `px` / `—` / `bool` / `string` units only), CC-5 (composition with Tier 0's `growth_axis_unreliable` — see design.md D5 for the reversal of the original "no re-emission" rule), CC-9 (logging), **CC-10 (all three independent noise estimators in Phase 1)**. Full architectural reasoning lives in `design.md` (committed alongside this proposal).

## What Changes

- **NEW QC trait emission** — `sleap_roots.circumnutation.qc.compute(trajectory_df, constants=None)` returns a per-plant DataFrame with the 8 row-identity columns plus **11 new columns**: 3 noise estimators + 3 pairwise agreements + 2 outlier-step traits + `growth_axis_unreliable` + `track_is_clean` composite + `qc_failure_reason` diagnostic. Canonical signature preserved exactly per the foundation's Package layout table.
- **NEW trait set (11 columns)** — full table + units + formulas in spec delta Requirement "QC tier per-track quality traits":
  - 3 SLEAP-localization-noise estimators (CC-10): `sg_residual_xy`, `d2_noise_xy`, `msd_noise_xy` (`px`)
  - 3 pairwise agreements: `sg_d2_agreement`, `sg_msd_agreement`, `d2_msd_agreement` (`—`)
  - 2 outlier-step diagnostics: `frac_outlier_steps`, `worst_step_ratio` (`—`)
  - `growth_axis_unreliable` (`bool`) — recomputed locally via the shared `_noise.compute_sg_residual_xy` helper; numerically identical to Tier 0's column by construction
  - `track_is_clean` (`bool`) — composite AND of 6 clauses (see spec delta Requirement "QC tier track_is_clean and qc_failure_reason composition")
  - `qc_failure_reason` (`string`) — stable-ordered comma-separated failure clauses; sentinel `"qc_inputs_insufficient"` for short-track gate
- **NEW `_noise.py` helpers (single source of truth)** — sibling functions `compute_d2_residual_xy(x, y)` and `compute_msd_residual_xy(x, y, window, degree, lag=1)` added alongside the existing `compute_sg_residual_xy`. Same DRY discipline as PR #2's `_noise.py`/`_geometry.py` split. Formulas anchored in spec delta + design.md D8 (MSD factor-of-4 is load-bearing).
- **NEW `_constants.py` additions** — `FRAC_OUTLIER_STEPS_MAX = 0.05`, `WORST_STEP_RATIO_MAX = 5`, `SG_MSD_AGREEMENT_MAX = 1.5`, `D2_MSD_AGREEMENT_MAX = 1.5`. `_CONSTANTS_VERSION` bumps `1 → 2` per the version-sentinel contract. `ConstantsT` gains 4 new fields. `_default_constants_snapshot()` extended to emit all 4 into the run-metadata sidecar.
- **MODIFIED Package layout requirement** — module counts: 7 contract modules (unchanged), **2 implementation modules** (`kinematics` + `qc`, was 1), **8 stub modules** (was 9 — strikes `qc` from the stub table). Mirrors the structural pattern PR #2 used for `kinematics`.
- **MODIFIED Module-level constants requirement** — adds 4 new constants, bumps `_CONSTANTS_VERSION` to `2`.
- **MODIFIED Tier 0 helper modules requirement** — `_noise.py` now exposes 3 helpers (existing `compute_sg_residual_xy` unchanged + 2 new). `_geometry.py` unchanged.
- **NEW spec requirements (4 ADDED)** — QC tier per-track quality traits; QC tier `track_is_clean` and `qc_failure_reason` composition; QC tier `growth_axis_unreliable` equality with Tier 0; QC tier input-validation boundary.
- **NEW tests** — `tests/test_circumnutation_qc.py` (synthetic exact-value tests + Nipponbare reference-value sanity test + KitaakeX smoke + equality contract regression + ConstantsT-override parametrize + stationary-track edge case). Coverage target: 100% on the new `qc.py` module and the two new `_noise.py` functions.
- **FOUNDATION TEST MIGRATION** — `tests/test_circumnutation_foundation.py` similar to PR #2's pattern: remove `("qc", "compute", 3)` from `STUB_MODULES` (9 → 8 parametrize ids), remove `("qc", "compute")` from `STUBS_WITH_CONSTANTS_KWARG` (6 → 5), extend `test_module_logger_is_namespaced` to include `qc`, and **update `test_schema_and_constants_versions_are_integers_equal_to_one`** to assert `_CONSTANTS_VERSION == 2` (was `== 1`).
- **DOCS update** `docs/circumnutation/theory.md` §7.6 — edit the cross-tier ownership note (line ~504) to reflect the D5 reversal: replace "QC SHALL NOT re-emit a duplicate `growth_axis_unreliable` column" with the equality-by-construction wording. Add a footnote on `track_is_clean` pointing to the canonical OpenSpec formula.
- **DOCS update** `docs/circumnutation/roadmap.md` — row PR #3 description loses `cadence_nyquist_ratio` (deferred to PR #6) and duplicate-mention of `growth_axis_unreliable`; row PR #6 gains `cadence_nyquist_ratio`; CC-5 step 3 updated for both-tiers-emit-with-equality; CC-10 unchanged.
- **CHANGELOG entry** `docs/changelog.md` (lowercase per repo convention).

## Impact

- **Affected specs:** capability `circumnutation` — 3 MODIFIED Requirements (Package layout, Module-level constants, Tier 0 helper modules) + 4 ADDED Requirements (QC tier per-track quality traits; track_is_clean/qc_failure_reason composition; growth_axis_unreliable equality with Tier 0; QC tier input-validation boundary). No other capabilities touched.
- **Affected code:**
  - extended modules: `sleap_roots/circumnutation/_noise.py` (adds 2 helpers), `sleap_roots/circumnutation/_constants.py` (adds 4 constants + 4 `ConstantsT` fields + version bump + snapshot extension)
  - implementation: `sleap_roots/circumnutation/qc.py` (replaces stub)
  - foundation test migration: `tests/test_circumnutation_foundation.py` (4 surgical edits per PR #2 pattern)
  - tests: `tests/test_circumnutation_qc.py` (new)
  - docs: `docs/circumnutation/theory.md` §7.6 edit; `docs/circumnutation/roadmap.md` row #3 / row #6 / CC-5 step 3 edits; `docs/changelog.md` entry
  - no new dependencies; no new test fixtures (reuses Nipponbare plate 001 + KitaakeX)
- **What this change does NOT do:**
  - No `coi_fraction_t1`, `coi_fraction_t3`, `is_nutating`, `noise_floor_estimate` — these depend on Tier 1 (PR #6) or Tier 3 (PR #9-10) outputs that don't exist yet. Emitted by their owning tiers when those tiers land.
  - No `cadence_nyquist_ratio` — both §6.5 definitions depend on Tier 1/3 quantities AND the temporal flavor would break the cadence-independent signature contract. Deferred to PR #6 (which has `cadence_s` via `CircumnutationInputs`).
  - No pipeline composition of QC + Tier 0 + Tier 1+ outputs — that's PR #14.
  - No per-genotype aggregation of QC traits — PR #15.
  - No new test fixtures — reuses PR #2's Nipponbare and the existing KitaakeX.
  - No empirical-anchor validation work for the new constants — tracked as 3 follow-up issues α/β/γ (see design.md "Follow-up Issues"); ship PR #3 now with documented best-guess defaults + `ConstantsT` override path.
- **Open scientific blockers:** **none gating this PR.** Pure-pixel emission keeps DPI ambiguity (PR #19) decoupled. Threshold defaults inherit from §7.6 / prelim §4.2 single-plate observations; weakness documented in design.md R1 with three focused follow-up issues queued.
- **Cross-link:** GitHub epic [#197](https://github.com/talmolab/sleap-roots/issues/197); foundation PR [#200](https://github.com/talmolab/sleap-roots/pull/200) / sub-issue [#198](https://github.com/talmolab/sleap-roots/issues/198) (merged 2026-05-11); Tier 0 PR [#203](https://github.com/talmolab/sleap-roots/pull/203) / sub-issue [#201](https://github.com/talmolab/sleap-roots/issues/201) (merged 2026-05-14); follow-up [#202](https://github.com/talmolab/sleap-roots/issues/202) (K=10 sensitivity sweep, deferred).
