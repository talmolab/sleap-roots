# Tasks: add-circumnutation-tier3c-traits

TDD discipline: one commit pair per unit — failing test (`test: … (TDD red)`) → implementation
(`feat:/fix: … (TDD green)`). Watch each RED fail before GREEN. Quick gate per pair: pytest the
touched test file + `black --check` + `pydocstyle --convention=google`; full suite for
shared-module edits (foundation, calibration script). Design + evidence:
`docs/superpowers/specs/2026-06-10-add-circumnutation-tier3c-traits-design.md` (CR-1..CR-13,
CR2-1..CR2-9); `docs/circumnutation/investigations/2026-06-10-tier3c-traveling-wave/report.md`.

## 1. Module scaffold + foundation-test migrations (CR-10, CR2-2)
- [ ] 1.1 RED: extend `tests/test_circumnutation_foundation.py` — add `"traveling_wave"` to the
  `test_module_logger_is_namespaced` parametrize list (after the `spatial_cwt` entry, ~line 801),
  to `IMPLEMENTATIONS_WITH_CONSTANTS_KWARG` (~line 891), and add a dedicated
  `elif module_name == "traveling_wave":` branch in `test_implementation_accepts_constants_kwarg`
  (mirror the `nutation` branch: ≥64-frame single-track df, `fn(df, 300.0, constants=ConstantsT())`,
  assert DataFrame). Confirm these fail (module absent). Verify `traveling_wave` is NOT added to
  `STUB_MODULES`/`STUBS_WITH_CONSTANTS_KWARG` (it was never a stub).
- [ ] 1.2 GREEN: create `sleap_roots/circumnutation/traveling_wave.py` with module docstring
  (sibling shape + Anchors to the design, the investigation report, theory.md §4.7/§6.4/§7.4),
  `logger = logging.getLogger(__name__)`, and a minimal `compute(trajectory_df, cadence_s,
  constants=None)` that validates inputs and returns the empty-shape DataFrame. Foundation tests pass.

## 2. Input-validation boundary + emission skeleton (CR-5, CR-6)
- [ ] 2.1 RED: schema test — `compute` returns 8 `ROW_IDENTITY_COLUMNS` + 6 trait columns in the
  declared order with all-float64 trait dtypes; 5-tuple uniqueness holds; validation rejects bad
  `trajectory_df` (delegates to `_validate_trajectory_df`) and bad `cadence_s` (reuse
  `temporal_cwt._validate_cadence_s`) and bad `constants` (None or ConstantsT). Local
  `COI_FRACTION_MAX` validated in (0, 1].
- [ ] 2.2 GREEN: implement the emission tail verbatim from siblings — `groupby(list(_IDENTITY_5_TUPLE),
  dropna=False, sort=False)`; `_build_per_plant_template_from_df`; pre-merge int64 coercion guard
  (raise on cast failure); left-merge on the 5-tuple; final column order = 8 identity + 6 traits;
  single `astype(np.float64)` loop (no bool/int special case). Add `_all_nan_spatial_traits()` helper.

## 3. Per-track spatial chain + error handling (CR-2, CR-3, CR2-5)
- [ ] 3.1 RED: per-track tests — (a) a healthy single track yields finite λ traits; (b) a degenerate
  /stationary track yields all-NaN spatial traits (#1–#5 NaN, #6 NaN) and does NOT raise; (c) a
  short/non-finite track does NOT crash `compute()` (other tracks survive); (d) a forced low-COI
  ridge gates #1–#5 to NaN while `coi_valid_fraction` (#6) stays finite.
- [ ] 3.2 GREEN: implement `_compute_one_track`: sort by `frame` + drop NON-finite tips
  (`np.isfinite` mask) before `reconstruct`; `is_degenerate` guards after `reconstruct` and
  `resample_curvature`; guard-before-call + `try/except ValueError → _all_nan_spatial_traits()`
  around `compute_scaleogram`/`extract_ridge`. Pin the CR-3 `coi_valid_fraction` rule (finite iff a
  ridge formed). Invariant: always returns a full 6-key dict, never raises.

## 4. COI gate + calibration consumer (CR-7, CR-11)
- [ ] 4.1 RED: COI gate test — gate NaN ⇔ `coi_valid_fraction < (1 − COI_FRACTION_MAX)`; pin the
  boundary (strict `>` on in-COI fraction; exactly-50%-in-COI does NOT gate). Calibration test —
  `λ_cal = λ_reported / ratio_interp(λ_reported)` using the **n=400** slice (strictly-increasing
  `λ_reported` → well-posed `np.interp`); recovers a known calibration knot.
- [ ] 4.2 GREEN: implement `~in_coi` interior selection, the COI fraction gate, and the calibrated-λ
  helper (one calibrated array used for all three λ-traits). Units mapping: declare the 6
  column→unit pairs (px/—; all already in `PIPELINE_UNIT_VOCABULARY` — no `_constants` change).

## 5. Composition (Tier 0/1 recompute + 5-tuple join) + traits + gating (CR-1, CR2-1, CR-4)
- [ ] 5.1 RED: **multi-plate test with float64 `track_id`** (≥2 plates, overlapping track_ids) —
  asserts each row's `v`/`T` operands come from the correct plate (the CR-1/CR2-1 regression).
  Plus: `is_nutating==False` → `traveling_wave_residual` NaN, λ traits valid; stationary `v≈0` →
  `lambda_expected_px`/`traveling_wave_residual` NaN (no inf/RuntimeWarning).
- [ ] 5.2 GREEN: recompute `kinematics.compute(df, constants=resolved)` and
  `nutation.compute(df, cadence_s, coordinate="lateral", constants=resolved)`; **merge** operands
  onto `trait_df` on the full `_IDENTITY_5_TUPLE` with int64 coercion (NOT `.at[key]`). Compute
  `T_frames = T/cadence_s`, `lambda_expected_px = v·T_frames`, `traveling_wave_residual`,
  `lambda_spatial_variation = MAD/median`, `lambda_spatial_mad_px`. Division guard: NaN #3/#4 when
  `v` non-finite or `lambda_expected_px ≤ 0`.

## 6. Calibration-table extension (append-only) (CR-8, CR2-4)
- [ ] 6.1 RED: regression test asserting the existing 18 `(n, λ_true)` rows + the `provenance` block
  are byte-for-byte unchanged after the extension; and that the extended n=400 `λ_reported` range
  covers the observed real λ (≥ ~150 px).
- [ ] 6.2 GREEN: add an append-only / merge mode to `capture_spatial_coi_factor.py` that generates
  only the new `λ_true` rows (n=400) up to ~150 px and preserves the existing rows + provenance
  verbatim (do NOT call `_provenance()` on append). Regenerate the committed JSON via that mode.

## 7. Determinism + real-data validation (CR2-8, CR-9, D6, D7)
- [ ] 7.1 RED: determinism canary — two runs of the **full composed chain (incl. np.interp
  calibration on the extended table)** are bit-identical in-process; a captured canary matches to a
  measured `atol` (target 1e-6 — re-measure, don't cargo-cult) cross-OS.
- [ ] 7.2 RED: real-data plate-001 test (all 6 tracks) — `traveling_wave_residual` finite and
  `< 0.30` (generous band; do NOT pin [0.087, 0.177] — both endpoints were extrapolation artifacts
  pre-extension); `lambda_spatial_variation` finite ~0.13–0.37; all gates pass; 6/6 nutating.
- [ ] 7.3 RED: synthetic known-λ recovery — a **small-amplitude** trajectory
  (`amplitude_px ≪ growth_rate·T_frames`) so λ_spatial ≈ `growth_rate·T_frames` a priori;
  `lambda_spatial_median_px` recovers it within calibration tolerance.
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
  wording, then the corrections (cite the design + investigation report).
- [ ] 8.3 roadmap.md line 146: rename both `apex_basal_period_consistency` → `lambda_spatial_variation`;
  remove the descoped `B_balance_number`/`L_gz_*` traits + "Applies the L_gz mask" claim (D1); add
  `traveling_wave` to the module enumeration. `docs/changelog.md`: PR #10 entry announces the rename
  (leave the historical PR #9 entry).

## 9. Verification gates (pre-merge)
- [ ] 9.1 `uv run pytest tests/ -q` green; `black --check`; `pydocstyle --convention=google
  sleap_roots/circumnutation/`; `uv lock --check`; `uv run mkdocs build`;
  `npx openspec validate add-circumnutation-tier3c-traits --strict`.
- [ ] 9.2 Coverage reasoning (CI-enforced ≥84% project, aim ≥90% on the new module); confirm via
  Codecov check (local `--cov` hits the Windows numpy-reload env bug — don't fight it locally).
