# Tasks: add-circumnutation-tier3c-traits

TDD discipline: one commit pair per unit — failing test (`test: … (TDD red)`) → implementation
(`feat:/fix: … (TDD green)`). Watch each RED fail before GREEN. Quick gate per pair: pytest the
touched test file + `black --check` + `pydocstyle --convention=google`; full suite for
shared-module edits (foundation, calibration script). Design + evidence:
`docs/superpowers/specs/2026-06-10-add-circumnutation-tier3c-traits-design.md` (CR-1..CR-13,
CR2-1..CR2-9); `docs/circumnutation/investigations/2026-06-10-tier3c-traveling-wave/report.md`.

## 0. PR #10 GitHub tracking issue (vault-draft → user OK → post) (OR: issue-hygiene)
- [x] 0.1 Draft the PR #10 issue body to the vault, post after user OK, backfill roadmap line 146.
  DONE: issue [#232](https://github.com/talmolab/sleap-roots/issues/232) posted (parent #197; labels
  `enhancement`, `circumnutation`, `multi-pr`); roadmap line 146 updated (rename + descope cleanup +
  `#232`, Issue-draft column → ✅). Use `Closes #232` in the PR body.

## 1. Module scaffold + foundation-test migrations (CR-10, CR2-2)
- [ ] 1.1 RED: extend `tests/test_circumnutation_foundation.py` — add `"traveling_wave"` (with a
  `# Added in PR #10: traveling_wave …` comment mirroring the PR #9 spatial_cwt block, for
  Copilot-regression symmetry) to the `test_module_logger_is_namespaced` parametrize list (after the
  `spatial_cwt` entry, ~line 801), to `IMPLEMENTATIONS_WITH_CONSTANTS_KWARG` (~line 891), and add a
  dedicated `elif module_name == "traveling_wave":` branch in
  `test_implementation_accepts_constants_kwarg` (mirror the `nutation` branch: ≥64-frame single-track
  df, `fn(df, 300.0, constants=ConstantsT())`, assert DataFrame). Confirm these fail (module absent).
  Verify `traveling_wave` is NOT added to `STUB_MODULES`/`STUBS_WITH_CONSTANTS_KWARG` (never a stub).
- [ ] 1.2 GREEN: create `sleap_roots/circumnutation/traveling_wave.py` with module docstring
  (sibling shape + Anchors to the design, the investigation report, theory.md §4.7/§6.4/§7.4),
  `logger = logging.getLogger(__name__)`, and a minimal `compute(trajectory_df, cadence_s,
  constants=None)` that validates inputs and returns the empty-shape DataFrame. Foundation tests pass.
  **Commit 1.1 + 1.2 TOGETHER** (the foundation suite imports `traveling_wave`; committing 1.1 alone
  leaves the shared suite RED across the commit boundary).

## 2. Input-validation boundary + emission skeleton (CR-5, CR-6)
- [ ] 2.1 RED: schema test — `compute` returns 8 `ROW_IDENTITY_COLUMNS` + 6 trait columns in the
  declared order with all-float64 trait dtypes; 5-tuple uniqueness holds; none of the omitted L_gz
  columns present. Validation RED tests (explicit, not delegation notes): bad `trajectory_df`
  (delegates to `_validate_trajectory_df`); `cadence_s` VALUE `∈ {0, -1.0, nan, inf, -inf}` →
  ValueError naming `cadence_s`; `cadence_s` TYPE `∈ {True, np.bool_(True), "300", [300.0]}` →
  TypeError naming `cadence_s`; `constants=object()` → TypeError; `ConstantsT(COI_FRACTION_MAX=0.0/1.5)`
  → ValueError naming `COI_FRACTION_MAX`.
- [ ] 2.2 GREEN: implement the emission tail verbatim from siblings — `groupby(list(_IDENTITY_5_TUPLE),
  dropna=False, sort=False)`; `_build_per_plant_template_from_df`; pre-merge int64 coercion guard
  (raise on cast failure); left-merge on the 5-tuple; final column order = 8 identity + 6 traits;
  single `astype(np.float64)` loop (no bool/int special case). Add `_all_nan_spatial_traits()` helper.
- [ ] 2.3 RED: units-vocabulary test (mirror `test_circumnutation_psi_g.py` §2) — assert
  `_TRAVELING_WAVE_TRAIT_UNITS` has all 6 columns mapped (`_px`→`"px"`, dimensionless→`"—"`) and every
  value ∈ `PIPELINE_UNIT_VOCABULARY`. GREEN: declare the mapping.
- [ ] 2.4 RED: logging test — `caplog.set_level(logging.DEBUG, logger="sleap_roots.circumnutation.traveling_wave")`;
  assert exactly one DEBUG record from that logger, `msg.startswith("traveling_wave.compute(")`,
  contains `"n_tracks="` and `"cadence_s="`, and zero records at INFO+ on the happy path.

## 3. Per-track spatial chain + error handling (CR-2, CR-3, CR2-5)
- [ ] 3.1 RED: per-track tests — (a) a healthy single track yields finite λ traits; (b) a degenerate
  /stationary track yields all-NaN spatial traits (#1–#5 NaN, #6 NaN) and does NOT raise nor emit
  `np.RuntimeWarning`; (c) a short/non-finite track does NOT crash `compute()` (other tracks survive);
  (d) an all-NaN-tip track AND a single-frame track each yield a well-formed all-NaN row (one row per
  5-tuple, no drop/dup); (e) a forced low-COI ridge gates #1–#5 to NaN while `coi_valid_fraction` (#6)
  stays finite.
- [ ] 3.2 GREEN: implement `_compute_one_track`: sort by `frame` + drop NON-finite tips
  (`np.isfinite` mask) before `reconstruct`; `is_degenerate` guards after `reconstruct` and
  `resample_curvature`; guard-before-call + `try/except ValueError → _all_nan_spatial_traits()`
  around `compute_scaleogram`/`extract_ridge`. Pin the CR-3 `coi_valid_fraction` rule (finite iff a
  ridge formed). Invariant: always returns a full 6-key dict, never raises.

## 4. COI gate + calibration consumer (CR-7, CR-11, OR: packaging / n-averaging)
> ORDERING: tasks 4.2/4.3 (the literal) depend on §6 having regenerated the extended JSON first —
> do §6 before §4.2/4.3 (or author 4.2 RED, then unblock its GREEN after §6.2).
- [ ] 4.1 RED: COI gate boundary test — construct (synthetic OR monkeypatched) `SpatialRidgeResult`s
  with `in_coi` fraction EXACTLY 0.5 and > 0.5; assert the exactly-0.5 case does NOT gate (strict
  inequality) and the >0.5 case gates #1–#5 while `coi_valid_fraction` stays finite in both.
  (Hitting exactly 0.5 deterministically requires a constructed/monkeypatched ridge, not real data.)
- [ ] 4.2 RED: in-package calibration-literal sync test — assert the module-level
  `_CGAU2_LAMBDA_CALIBRATION` literal equals the **n-averaged** computation from the authoritative
  JSON (for each `λ_true`, mean `ratio` across `n ∈ {200,400,600}` + `λ_reported_mean`, sorted by
  `λ_reported_mean`, `atol=0`), its axis is strictly increasing, and it covers `λ_true ≥ 140 px`.
  The literal SHALL be generated from full-precision JSON tokens (never hand-rounded). Calibration
  test — `λ_cal = λ_obs / np.interp(λ_obs, axis, ratio_mean)` recovers a known point. (The module
  reads the LITERAL, never `tests/data` at runtime — wheel-safe.)
- [ ] 4.3 GREEN: implement `~in_coi` interior selection, the COI fraction gate
  (`coi_valid_fraction < 1 − COI_FRACTION_MAX`), the n-averaged `_CGAU2_LAMBDA_CALIBRATION` literal,
  and the calibrated-λ helper (one calibrated array used for all three λ-traits). Declare
  `_TRAVELING_WAVE_TRAIT_UNITS` (done in 2.3).

## 5. Composition (Tier 0/1 recompute + 5-tuple join) + traits + gating (CR-1, CR2-1, CR-4)
- [ ] 5.1 RED: **multi-plate test with float64 `track_id`** (≥2 plates, overlapping track_ids) —
  asserts (i) one row per 5-tuple, no raise; (ii) for healthy nutating tracks the merged operands are
  FINITE, not all-NaN (the int64-vs-float64 silent-NaN guard — the failure mode is NaN, NOT KeyError);
  (iii) each row's `lambda_expected_px` derives from THAT 5-tuple's own `v`/`T` (correct plate).
  Plus: `is_nutating==False` → `traveling_wave_residual` NaN, λ traits valid; stationary `v≈0` →
  `lambda_expected_px`/`traveling_wave_residual` NaN with NO `np.RuntimeWarning` (assert via
  `warnings.catch_warnings`).
- [ ] 5.2 GREEN: recompute `kinematics.compute(df, constants=resolved)` and
  `nutation.compute(df, cadence_s, coordinate="lateral", constants=resolved)`; **merge** operands
  onto `trait_df` on the full `_IDENTITY_5_TUPLE` with int64 coercion (NOT `.at[key]`). Compute
  `T_frames = T/cadence_s`, `lambda_expected_px = v·T_frames`, `traveling_wave_residual`,
  `lambda_spatial_variation = MAD/median`, `lambda_spatial_mad_px`. Division guard: NaN #3/#4 when
  `v` non-finite or `lambda_expected_px ≤ 0`.

## 6. Calibration-table extension (append-only, load-and-passthrough) (CR-8, CR2-4) — DO BEFORE §4.2/4.3
- [ ] 6.1 RED: regression test asserting the existing 18 `(n, λ_true)` rows + the entire `provenance`
  block are unchanged after the extension — compared **by `(n, λ_true)` key lookup** (NOT list
  position, so an end-append can't trip it); the n-averaged axis (task 4.2) is strictly increasing;
  and the extension covers `λ_true ≥ 140 px` (→ `λ_reported ≈ 157 px ≥` the observed real λ ≈ 142.5,
  so no clamped extrapolation). Headroom is pinned on `λ_true` (≥140), NOT `λ_reported`.
- [ ] 6.2 GREEN: add an append-only / merge mode to `capture_spatial_coi_factor.py` that **reads the
  existing committed JSON, copies its `provenance` block and existing `wavelength_calibration` rows
  through verbatim (NO re-measure, do NOT call `_provenance()`), and measures ONLY the new `λ_true`
  knots (e.g. 100, 120, 140, 150) for ALL `n ∈ {200, 400, 600}`** (all three n are needed so the
  consumer's n-average is defined at the new knots), **appending them at the END** of the
  `wavelength_calibration` list (existing rows stay a contiguous prefix). This makes the freeze
  mechanical (not environment-dependent). Regenerate the committed JSON via that mode; derive the
  n-averaged `_CGAU2_LAMBDA_CALIBRATION` literal (task 4.2/4.3) from the regenerated table.
- [ ] 6.3 Update the stale `# … ratio [1.044, 1.156]` comment in `tests/test_circumnutation_spatial_cwt.py`
  to reflect the extended range (the hardcoded `[1.00, 1.25]` band still passes; only the comment is stale).

## 7. Determinism + real-data validation (CR2-8, CR-9, D6, D7)
- [ ] 7.1 RED: determinism — (a) two runs of the **full composed chain (incl. the np.interp
  calibration)** are bit-identical IN-PROCESS at `atol=0`; (b) capture and assert the ridge
  `wavelengths_px[interior]` array (PUBLIC `SpatialRidgeResult` field, 1:1 with the argmax scale index)
  EXACTLY equal cross-OS — an argmax tie-flip is a discrete scale-step jump, not atol-bounded, so this
  exact-equality (not a float tol) is the real spatial-λ cross-OS contract; (c) a captured float canary
  tuple for the v·T-derived columns (`lambda_expected_px`, `traveling_wave_residual`) matches hardcoded
  values to a measured `atol` (target 1e-6, inherited from Tier 1 — re-measure; looser only if a
  tie-flip shifts the median). State the canary columns/positions explicitly.
- [ ] 7.2 RED: real-data plate-001 test (all 6 tracks), **gated with
  `@pytest.mark.skipif(not _PROOFREAD_FIXTURE.exists(), …)`** exactly like `test_circumnutation_spatial_cwt.py`
  §7 — `traveling_wave_residual` finite and `< 0.30` (generous band; do NOT pin [0.087, 0.177] — both
  endpoints were extrapolation artifacts pre-extension); `lambda_spatial_variation` finite ~0.10–0.45;
  all gates pass (`coi_valid_fraction ≥ 1 − COI_FRACTION_MAX`); 6/6 nutating.
- [ ] 7.3 RED: synthetic known-λ recovery — a **small-amplitude, low-noise** trajectory
  (e.g. `generate_trajectory(amplitude_px=2.0, growth_rate_px_per_frame=4.29, T_nutation_s=3333,
  cadence_s=300, n_frames=575, noise_sigma_px=0.0, random_state=0)` → λ_apriori ≈ 47.7 px) so
  λ_spatial ≈ `growth_rate·T_frames` a priori; assert `abs(lambda_spatial_median_px − λ_apriori)/λ_apriori < 0.25`.
  On this NOISE-FREE uniform-λ trail ALSO assert `lambda_spatial_variation ≈ 0` (e.g. `< 0.05`) —
  confirming there is NO spurious argmax-quantization floor and the trait correctly reads "uniform"
  when λ is uniform. Document (test + theory) that real-data `lambda_spatial_variation` (~0.13–0.37)
  is genuine ridge-localization scatter that grows with noise, interpreted relative to the noise level
  — NOT a quantization floor and NOT pure biological λ variation.
- [ ] 7.4 GREEN: reconcile any numeric drift; lock the measured atol + canary values.

## 8. Spec deltas + docs deviations (CR-12, CR-13, CR2-6)
- [ ] 8.1 Author spec deltas: MODIFY "Package layout" (add `traveling_wave`, 8→9 impl, PR #10
  addition scope note, callability scenario, import scenario line 34); ADD "Tier 3c traveling-wave
  trait emission API". `npx openspec validate add-circumnutation-tier3c-traits --strict` passes.
- [ ] 8.2 theory.md edits: §7.4 rows for the 3 λ-traits (lines 504/505/507 — rename
  `lambda_spatial_median`→`_px`, `mm`→`px`, strike "basal of L_gz"; residual `T_frames`; rename
  `apex_basal_period_consistency`→`lambda_spatial_variation` = MAD/median); correct handoff notes 2
  (line 512) + 4 (line 514) in place; update the PR #9 scope-note dead name (lines 485–495); add
  **Appendix B(6)** preserving the ORIGINAL "bias cancels" + `apex_basal_period_consistency`
  wording, then the corrections (cite the design + investigation report). Also document in §7.4 /
  Appendix B(6): the cgau2 calibration uses an **n-averaged** `ratio(λ)` curve with an irreducible
  **~±5% systematic** (residual n-scatter), and `lambda_spatial_variation` is a **noise-sensitive
  spread** (≈0 on a noise-free uniform-λ trail — no spurious quantization floor; grows with
  curvature-localization noise, task 7.3) interpreted relative to the noise level, not as pure
  biological signal; the D7 plate-001 numbers are provisional pending the post-extension re-measurement.
- [ ] 8.3 roadmap.md line 146: rename both `apex_basal_period_consistency` → `lambda_spatial_variation`;
  remove the descoped `B_balance_number`/`L_gz_*` traits + "Applies the L_gz mask" claim (D1); add
  `traveling_wave` to the module enumeration. `docs/changelog.md`: PR #10 entry announces the rename
  (leave the historical PR #9 entry). GitHub-side rename propagation: the PR #10 issue body (task 0.1)
  states the rename; leave a comment on #230 (and note for the next #197 epic edit) recording
  `apex_basal_period_consistency` → `lambda_spatial_variation` so the open issues don't reference a
  dead trait name (post only after user OK, per the issue-hygiene rule).

## 9. Verification gates (pre-merge)
- [ ] 9.1 `uv run pytest tests/ -q` green; `black --check`; `pydocstyle --convention=google
  sleap_roots/circumnutation/`; `uv lock --check`; `uv run mkdocs build`;
  `npx openspec validate add-circumnutation-tier3c-traits --strict`.
- [ ] 9.2 Coverage reasoning (CI-enforced ≥84% project, aim ≥90% on the new module); confirm via
  Codecov check (local `--cov` hits the Windows numpy-reload env bug — don't fight it locally).
