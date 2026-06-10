# Tasks — add-circumnutation-tier3b-spatial-cwt (PR #9)

Strict TDD: one commit pair per scope unit — failing test (`test: … (TDD red)`) →
implementation (`feat:`/`fix: … (TDD green)`). Watch each RED actually fail before
GREEN. Run the per-pair quick gate (pytest the touched test files + `black --check`
+ `pydocstyle`) after every pair; run the FULL suite for any `_constants` edit.

**Commit-atomicity contract (review BLOCKING reconciliation).** `spatial_cwt` is
foundation-locked to RAISE `NotImplementedError`, and the foundation suite +
`tests/test_circumnutation_temporal_cwt.py::test_2G4` both assert
`_CONSTANTS_VERSION == 5`. Therefore the version bump 5→6, the 3-constant
addition, the first non-raising `compute_scaleogram` body, the foundation-table
migration, AND the version-assertion updates in BOTH test files MUST land in ONE
commit (Section 5). Build everything else first as NEW (non-foundation-locked)
symbols, keeping `compute_scaleogram` raising until that single atomic commit.
`resample_curvature` and `extract_ridge` are testable beforehand by constructing
`SpatialScaleogramResult` directly; the private scale/COI helpers take the COI
factor as a parameter (like temporal `_coi_boundary_samples`) so they need no
constant before Section 5.

## 1. Measure the cgau2 COI e-folding factor (BEFORE constants — feeds the default)

- [ ] 1.1 `scripts/circumnutation/capture_spatial_coi_factor.py` — step-response measurement of the cgau2 e-folding factor across scales (mirroring PR #5's cmor calibration); record the measured value + provenance (platform, pywt/numpy versions). Document the derivation in `design.md`. The measured number becomes the `SPATIAL_COI_EFOLDING_FACTOR` default added in Section 5. (No package change; resolves the backwards dependency the review flagged.)

## 2. ResampleResult + resample_curvature (independent — no new constants)

- [ ] 2.1 RED: `ResampleResult` is a frozen `attrs` class (`eq=False`) with fields `(kappa_uniform, s_a_uniform_px, ds, n_unmasked, arc_span_px, is_degenerate)`; frozen-instance write raises.
- [ ] 2.2 GREEN: define `ResampleResult`.
- [ ] 2.3 RED: happy path — drops masked frames, apex-origin `s_a = arc_max − s`, `ds = median(positive Δs_a)`, `np.interp`; shapes/dtypes correct; `s_a_uniform_px[0] == 0`; **`result.n_unmasked == int((~mask).sum())`**; **`result.arc_span_px == max(arc)−min(arc)` over unmasked**; masked frames contribute no interpolation knots.
- [ ] 2.4 GREEN: implement the happy-path resample (drop masked, reparameterize, dedup duplicate-`s_a` by averaging κ, choose `ds`, interpolate).
- [ ] 2.5 RED: validation (structural only) — non-ndarray (`TypeError`); non-1-D / complex / object dtype / length-mismatch / wrong-length mask (`ValueError`/`TypeError` naming the field); invalid `constants` (`TypeError`). NOTE: non-finite VALUES are NOT a validation error here (see 2.7) — only structural malformation raises.
- [ ] 2.6 GREEN: implement field-named validation (runs first, unconditionally).
- [ ] 2.7 RED: non-finite + degenerate behavior — non-finite `(curvature, arc_length)` pairs are DROPPED (not rejected), no `RuntimeWarning`; degenerate gate (too-few survivors / non-positive span / all-equal arc_length) → graceful all-NaN `ResampleResult` (`is_degenerate=True`). The gate runs BEFORE `np.median(positive Δs_a)` so an all-equal arc_length cannot emit `RuntimeWarning` from `np.median([])`.
- [ ] 2.8 GREEN: implement drop-non-finite + the gate-before-median ordering (`np.errstate` + `~isfinite` sweep).
- [ ] 2.9 RED: duplicate-`s_a` determinism — inputs with duplicate `arc_length_px` → finite, deterministic `kappa_uniform` (averaged knots), `np.array_equal` across two runs at `atol=0`.
- [ ] 2.10 GREEN: confirm the dedup-by-averaging makes knots strictly increasing (any fix lands here).
- [ ] 2.11 RED: apex VALUE-pin — an asymmetric κ feature placed at the largest `arc_length_px` lands at SMALL `s_a` (near 0), guarding the `s_a = max−arc` orientation by value (PR #8 sign-anchoring lesson).
- [ ] 2.12 GREEN: confirm orientation (fix if flipped).
- [ ] 2.13 RED: exactly one DEBUG record `"resample_curvature("` with tokens `n_input=`, `n_unmasked=`, `ds=`, `arc_span_px=`; no INFO/WARN/ERROR.
- [ ] 2.14 GREEN: add the namespaced `logger.debug`.

## 3. SpatialScaleogramResult + SpatialRidgeResult classes + extract_ridge (construct Result directly)

- [ ] 3.1 RED: `SpatialScaleogramResult` frozen `attrs` (`eq=False`) fields `(scaleogram, scales, wavelengths_px, spatial_freqs_px_inv, coi_mask, ds, wavelet)`; frozen write raises.
- [ ] 3.2 GREEN: define `SpatialScaleogramResult`.
- [ ] 3.3 RED: `SpatialRidgeResult` frozen `attrs` (`eq=False`) fields `(position_indices, wavelengths_px, amplitudes, powers, in_coi)`; `powers == amplitudes**2`; frozen write raises.
- [ ] 3.4 GREEN: define `SpatialRidgeResult`.
- [ ] 3.5 RED: `extract_ridge(scaleogram_result, constants=None)` on a MANUALLY-constructed `SpatialScaleogramResult` — per-position argmax over scales → λ at ridge; amplitudes = |W| ≥ 0; `in_coi` from the COI mask; `position_indices == arange`; a constructed scale-tie returns the smallest index (deterministic tie-break).
- [ ] 3.6 GREEN: implement `extract_ridge` (deterministic argmax, no random tie-break).
- [ ] 3.7 RED: validation — non-`SpatialScaleogramResult` (`TypeError`); empty (`n_scales==0` or `n_samples==0`) (`ValueError`); invalid `constants` (`TypeError`).
- [ ] 3.8 GREEN: implement validation.
- [ ] 3.9 RED: exactly one DEBUG record `"extract_ridge("` with tokens `n_scales=`, `n_samples=`; no INFO/WARN/ERROR.
- [ ] 3.10 GREEN: add the `logger.debug`.

## 4. Private scale-axis + COI helpers (parameterized — no constant dependency yet)

- [ ] 4.1 RED: `_coi_boundary_samples(scale, coi_factor)` and `_make_coi_mask(scales, n_samples, coi_factor)` spatial siblings (mirror temporal) — boundary = `ceil(coi_factor * scale)`, mask shape/edges correct.
- [ ] 4.2 GREEN: implement the COI helpers (factor passed as a parameter).
- [ ] 4.3 RED: `_log_spaced_scales(...)`-style spatial scale axis — λ_min = `factor_min·ds`, λ_max = `fraction_max·n·ds`, `n_scales` log-spaced via `pywt.scale2frequency("cgau2", …)`; reciprocal `wavelengths_px`/`spatial_freqs_px_inv`. (Take the factors/wavelet/scale-count as params so this is testable pre-Section-5.)
- [ ] 4.4 GREEN: implement the scale axis.

## 5. ATOMIC graduation commit (version bump + constants + compute_scaleogram + foundation migration + both assertion updates)

All of 5.x land in a SINGLE commit so no committed suite is ever RED.

- [ ] 5.1 RED: write the `compute_scaleogram(kappa, ds, constants=None)` tests (shapes/dtypes `(CWT_SCALE_COUNT_DEFAULT, n)` complex128, monotonic `scales`, reciprocal wavelength/freq, `coi_mask`, `wavelet=="cgau2"`, `ds` echoed); validation (malformed/non-finite/too-short `kappa`; `ds` ≤0/NaN/inf→`ValueError`, bool/`np.bool_`/str/list→`TypeError`; accept-at MIN floor + reject-below); one DEBUG `"compute_scaleogram("` with tokens `n_samples=`,`ds=`,`n_scales=`,`wavelength_min_px=`,`wavelength_max_px=`,`wavelet=`. Also write: `_CONSTANTS_VERSION == 6`; 3 new constants exist with correct types + as `ConstantsT` fields + the override ROUND-TRIP (`ConstantsT(SPATIAL_COI_EFOLDING_FACTOR=…, CWT_WAVELENGTH_MIN_NYQUIST_FACTOR=…, CWT_WAVELENGTH_MAX_SIGNAL_FRACTION=…)` reflects values; unoverridden keep defaults); snapshot contains all 3 and `len == 38`. (These RED-fail because the stub still raises / version is still 5.)
- [ ] 5.2 GREEN (atomic, one commit): (a) add the 3 constants to `_constants.py` with docstrings (read 2–3 existing docstrings first; `CWT_WAVELENGTH_*` cross-reference the temporal `CWT_PERIOD_*`; `SPATIAL_COI_EFOLDING_FACTOR` = the Section-1 measured value), add to `ConstantsT` + `_default_constants_snapshot()`, bump `_CONSTANTS_VERSION` 5→6; (b) implement non-raising `compute_scaleogram` wiring the Section-4 helpers + constants + `pywt.cwt(kappa, scales, "cgau2")` → complex128; (c) MIGRATE foundation: remove `("spatial_cwt","compute_scaleogram",9)` from `STUB_MODULES` + `STUBS_WITH_CONSTANTS_KWARG`; add `("spatial_cwt","compute_scaleogram")` to `IMPLEMENTATIONS_WITH_CONSTANTS_KWARG` with an `elif module_name == "spatial_cwt":` branch (import `SpatialScaleogramResult`, build a length-≥9 `kappa` + `ds`, assert isinstance); add `"spatial_cwt"` to the explicit `test_module_logger_is_namespaced` list with a PR #9 comment block; update the STUB_MODULES block-comment (impl 7→8, stub 4→3); (d) update the foundation `_CONSTANTS_VERSION` assertion 5→6; (e) update `tests/test_circumnutation_temporal_cwt.py::test_2G4` `_CONSTANTS_VERSION` assertion 5→6.
- [ ] 5.3 GATE: run the FULL suite (foundation + temporal_cwt + spatial_cwt) — confirm GREEN before proceeding. (The single most important checkpoint.)

## 6. Analytic oracles + determinism canary (full chain)

- [ ] 6.1 RED: pure-sinusoid-of-known-λ oracle wired through `resample_curvature → compute_scaleogram → extract_ridge` (so the RED can fail on a wrong apex/`s_a` axis, not green-on-arrival) — COI-interior (`~in_coi`) median `wavelengths_px` recovers `lambda_true` within a LITERAL ±~5% (one log-scale step) tolerance.
- [ ] 6.2 GREEN: ensure recovery + convention hold (any fix lands here).
- [ ] 6.3 RED: harmonic robustness — fundamental + 30–40% second harmonic → ridge median tracks the fundamental λ, not λ/2.
- [ ] 6.4 GREEN: confirm (document any scale-band tuning).
- [ ] 6.5 RED: 50%-sparsity robustness — sinusoid on a non-uniform grid, ~50% randomly dropped → full chain recovers `lambda_true` within tolerance; record any interp-gap bias.
- [ ] 6.6 GREEN: confirm.
- [ ] 6.7 RED: determinism — `compute_scaleogram` + `extract_ridge` same input twice → `np.array_equal` at `atol=0`; hardcoded canary at interior COI-dodging `[scale_idx, positions]` matches at `atol=1e-9, rtol=0`.
- [ ] 6.8 GREEN: `scripts/circumnutation/capture_spatial_cwt_canary.py` (interior, COI-dodging) → paste literals into the test; ISOLATE the canary in its own test and pre-document (design.md) the per-canary loosen-on-legitimate-drift policy in case cgau2 (unproven cross-OS here) needs a single looser canary atol.

## 7. Real plate-001 validation (PR #7/#8 lesson — not synthetic-only)

- [ ] 7.1 RED→GREEN: on the 6 Nipponbare proofread tracks (reuse the midline test's `_load_proofread_enriched()` / `_track_xy()` helpers + `@pytest.mark.skipif` on the LFS fixture), run `midline.reconstruct` → `resample_curvature(mr.curvature_px_inv, mr.arc_length_px, mr.velocity_sub_noise_mask)` → (skip if `is_degenerate`) → `compute_scaleogram` → `extract_ridge`; assert no raise, shape-correct output, λ_spatial median over `~in_coi` finite + physically plausible, and the post-mask Nyquist sanity `2*ds < NYQUIST_RATIO_MAX*lambda_observed` (recorded for auditability).

## 8. Theory + program-doc deviation discipline (review BLOCKING reconciliation: roadmap/CC-1/#197)

- [ ] 8.1 Patch `docs/circumnutation/theory.md`: §7.4/§6.3 note that `L_gz`/`L_c` are not measurable from top-view tip-trail κ(s) as written (empirical, this PR); AND the §6.5 trait-status table rows for `L_gz`/`L_c` (currently "✓ Measurable (Tier 3)") → "✗ Not measurable from tip-trail as written — see #230" (resolve the §6.5-vs-§7.4 internal contradiction). λ_spatial is the measurable spatial quantity. Preserve the original §7.4 text in Appendix B with a "Why" note referencing #230.
- [ ] 8.2 Patch `docs/circumnutation/roadmap.md`: rewrite row #9 scope → "resample + cgau2 scaleogram + λ_spatial ridge; L_gz/L_c descoped to #230"; fill row #9 issue/status cells with #229; annotate CC-1 step 2 (now factually wrong — it says PR #9 peak-finds L_gz/fits L_c) with the descope + #230 pointer; annotate row #10 that its L_gz-dependent traits + mask are blocked on #230 while `traveling_wave_residual` + `apex_basal_period_consistency` remain deliverable from PR #9's λ.
- [ ] 8.3 Update epic #197 body: fill in #229/#230 for PR #9 and correct the stale "L_gz peak-find + L_c decay fit" scope line (tracking-hygiene; draft note in vault, post per the authorize-each-post rule).
- [ ] 8.4 Update `docs/circumnutation/__init__.py`-area docstrings if stale: `sleap_roots/circumnutation/__init__.py` "remaining stubs" docstring (currently lists `spatial_cwt`) → remove `spatial_cwt`.
- [ ] 8.5 Update `docs/changelog.md` (lowercase path) with the PR #9 entry + the deviation "Why" note.

## 9. Verification gates (before merge)

- [ ] 9.1 `uv run pytest tests/ -q` green.
- [ ] 9.2 `uv run black --check sleap_roots tests` clean.
- [ ] 9.3 `uv run pydocstyle --convention=google sleap_roots/circumnutation/` clean.
- [ ] 9.4 `uv lock --check` clean; `uv run mkdocs build` builds.
- [ ] 9.5 `npx openspec validate add-circumnutation-tier3b-spatial-cwt --strict` valid.
- [ ] 9.6 Coverage ≥84% project-wide (CI-enforced); aim ≥90% on `spatial_cwt.py`.
- [ ] 9.7 CI matrix Ubuntu/Windows/macOS green on the final commit (cross-OS determinism at `atol=1e-9`).
- [ ] 9.8 `/pre-merge` (Phase 3.5 `/review-pr`); after PR open + CI green: `/copilot-review` then post-PR `/review-pr`; reconcile all findings (each substantive code change ships a test).
