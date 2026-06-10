# Tasks — add-circumnutation-tier3b-spatial-cwt (PR #9)

Strict TDD: one commit pair per scope unit — failing test (`test: … (TDD red)`) →
implementation (`feat:`/`fix: … (TDD green)`). Watch each RED actually fail before
GREEN. Run the per-pair quick gate (pytest the touched test files + `black --check`
+ `pydocstyle`) after every pair; run the full suite for `_constants` edits.

## 1. Constants + version bump (5 → 6)

- [ ] 1.1 RED: assert `_constants.SPATIAL_COI_EFOLDING_FACTOR`, `CWT_WAVELENGTH_MIN_NYQUIST_FACTOR`, `CWT_WAVELENGTH_MAX_SIGNAL_FRACTION` exist with correct types; `_CONSTANTS_VERSION == 6`; `len(_default_constants_snapshot()) == 38`; the 3 new constants present in the snapshot and as `ConstantsT` fields.
- [ ] 1.2 GREEN: add the 3 constants to `_constants.py` (with docstrings — read 2–3 existing docstrings first per the convention), add them to `ConstantsT` + `_default_constants_snapshot()`, bump `_CONSTANTS_VERSION` 5→6. `CWT_WAVELENGTH_*` get spatial-sibling docstrings cross-referencing the temporal `CWT_PERIOD_*`. `SPATIAL_COI_EFOLDING_FACTOR` default is the measured cgau2 value from task 6.1.
- [ ] 1.3 Run full suite (shared `_constants` edit) — confirm no foundation regression yet (foundation `_CONSTANTS_VERSION` assertion updated atomically in task 7).

## 2. ResampleResult + resample_curvature

- [ ] 2.1 RED: `ResampleResult` is a frozen `attrs` class (`eq=False`) with fields `(kappa_uniform, s_a_uniform_px, ds, n_unmasked, arc_span_px, is_degenerate)`; frozen-instance write raises.
- [ ] 2.2 GREEN: define `ResampleResult`.
- [ ] 2.3 RED: `resample_curvature(curvature_px_inv, arc_length_px, velocity_sub_noise_mask=None, constants=None)` drops masked + non-finite frames, builds apex-origin `s_a = arc_max − s`, `ds = median(positive Δs_a)`, `np.interp` onto a uniform grid; shapes/dtypes correct; `s_a_uniform_px[0] == 0`.
- [ ] 2.4 GREEN: implement the happy-path resample.
- [ ] 2.5 RED: validation — non-ndarray (`TypeError`), non-1-D / complex / object / length-mismatch / non-finite-after-unmask (`ValueError` naming the field); invalid `constants` (`TypeError`).
- [ ] 2.6 GREEN: implement field-named validation (runs first, unconditionally).
- [ ] 2.7 RED: degenerate gate — too-few unmasked (`< min`), too-short arc-span, all-masked → graceful all-NaN `ResampleResult` (`is_degenerate=True`), never raises, never `RuntimeWarning`.
- [ ] 2.8 GREEN: implement the degenerate gate (errstate + `~isfinite` sweep).

## 3. SpatialScaleogramResult + compute_scaleogram (cgau2)

- [ ] 3.1 RED: `SpatialScaleogramResult` frozen `attrs` (`eq=False`) fields `(scaleogram, scales, wavelengths_px, spatial_freqs_px_inv, coi_mask, ds, wavelet)`; frozen write raises.
- [ ] 3.2 GREEN: define `SpatialScaleogramResult`.
- [ ] 3.3 RED: `compute_scaleogram(kappa, ds, constants=None)` returns shapes/dtypes — `scaleogram (CWT_SCALE_COUNT_DEFAULT, n)` complex128; `scales` float64 monotonic; `wavelengths_px`/`spatial_freqs_px_inv` reciprocal; `coi_mask` bool same shape; `wavelet == "cgau2"`; `ds` echoed.
- [ ] 3.4 GREEN: implement scale axis (log-spaced via `pywt.scale2frequency("cgau2", …)`, λ range from `CWT_WAVELENGTH_MIN_NYQUIST_FACTOR·ds` / `CWT_WAVELENGTH_MAX_SIGNAL_FRACTION·n·ds`), `pywt.cwt(kappa, scales, "cgau2")` → complex128, COI via `SPATIAL_COI_EFOLDING_FACTOR` (private `_make_coi_mask`/`_coi_boundary_samples` spatial siblings).
- [ ] 3.5 RED: validation — `kappa` malformed/non-finite/too-short (`ValueError`/`TypeError` naming `kappa`); `ds` ≤0/NaN/inf (`ValueError`), bool/str (`TypeError`); positive boundary at MIN length; invalid `constants`.
- [ ] 3.6 GREEN: implement validation.
- [ ] 3.7 RED: exactly one DEBUG log `"compute_scaleogram("` with tokens `n_samples=`, `ds=`, `n_scales=`, `wavelength_min_px=`, `wavelength_max_px=`, `wavelet=`; no INFO/WARN/ERROR.
- [ ] 3.8 GREEN: add the namespaced logger.debug.

## 4. SpatialRidgeResult + extract_ridge

- [ ] 4.1 RED: `SpatialRidgeResult` frozen `attrs` (`eq=False`) fields `(position_indices, wavelengths_px, amplitudes, powers, in_coi)`; `powers == amplitudes**2`; frozen write raises.
- [ ] 4.2 GREEN: define `SpatialRidgeResult`.
- [ ] 4.3 RED: `extract_ridge(scaleogram_result, constants=None)` — per-position argmax over scales → λ at ridge; amplitudes = |W| ≥ 0; `in_coi` from the scaleogram COI mask; `position_indices == arange`.
- [ ] 4.4 GREEN: implement extract_ridge (deterministic argmax, no random tie-break).
- [ ] 4.5 RED: validation — non-`SpatialScaleogramResult` (`TypeError`); empty (`n_scales==0` or `n_samples==0`) (`ValueError`); invalid `constants` (`TypeError`).
- [ ] 4.6 GREEN: implement validation.
- [ ] 4.7 RED: exactly one DEBUG log `"extract_ridge("` with tokens `n_scales=`, `n_samples=`; no INFO/WARN/ERROR.
- [ ] 4.8 GREEN: add the logger.debug.

## 5. Analytic oracle + determinism canary (CC-6)

- [ ] 5.1 RED: pure-sinusoid-of-known-λ oracle — `extract_ridge` recovers the planted λ at interior COI-dodging positions within a documented tolerance; pin the apex-origin convention.
- [ ] 5.2 GREEN: ensure the recovery holds (any convention fix lands here).
- [ ] 5.3 RED: determinism — same input twice → `np.array_equal` at `atol=0`; hardcoded canary at interior `[scale_idx, positions]` matches at `atol=1e-9, rtol=0`.
- [ ] 5.4 GREEN: `scripts/circumnutation/capture_spatial_cwt_canary.py` (interior indices, COI-dodging) → paste literals into the test.

## 6. cgau2 COI factor measurement

- [ ] 6.1 `scripts/circumnutation/capture_spatial_coi_factor.py` — step-response measurement of the cgau2 e-folding factor across scales (mirroring PR #5's cmor calibration); record the measured value + provenance. Feeds `SPATIAL_COI_EFOLDING_FACTOR` default in task 1.2. Document the derivation in `design.md`.

## 7. Foundation-test migration (ATOMIC with the first non-raising commit)

- [ ] 7.1 RED→GREEN (atomic): remove `("spatial_cwt","compute_scaleogram",9)` from `STUB_MODULES` + `STUBS_WITH_CONSTANTS_KWARG`; add `("spatial_cwt","compute_scaleogram")` to `IMPLEMENTATIONS_WITH_CONSTANTS_KWARG` (array-typed branch building `kappa, ds`); add `"spatial_cwt"` to the explicit `test_module_logger_is_namespaced` list with a PR #9 comment block; add the `elif module_name == "spatial_cwt":` branch; bump the `_CONSTANTS_VERSION` assertion 5→6; update the STUB_MODULES block-comment (impl 7→8, stub 4→3). Full suite green.

## 8. Real plate-001 validation (PR #7/#8 lesson — not synthetic-only)

- [ ] 8.1 RED→GREEN: on the 6 Nipponbare proofread tracks (`Series.load(...).get_tracked_tips()` → `midline.reconstruct` → `resample_curvature` → `compute_scaleogram` → `extract_ridge`), assert no raise, shape-correct output, and λ_spatial median is physically plausible + ~uniform along the trail (recorded for auditability). Mark `@pytest.mark.skipif` on the LFS fixture like the midline test.

## 9. Theory + docs deviation discipline

- [ ] 9.1 Patch `docs/circumnutation/theory.md` §7.4/§6.3: note `L_gz`/`L_c` not measurable from top-view tip-trail κ(s) as written (empirical, this PR); λ_spatial is the measurable spatial quantity. Preserve original §7.4 text in Appendix B with a "Why" note referencing #230.
- [ ] 9.2 Update `docs/changelog.md` (lowercase path) with the PR #9 entry + the deviation "Why" note.
- [ ] 9.3 Update `docs/circumnutation/roadmap.md` CC-2 constants table with the 3 new constants (if it enumerates them).

## 10. Verification gates (before merge)

- [ ] 10.1 `uv run pytest tests/ -q` green.
- [ ] 10.2 `uv run black --check sleap_roots tests` clean.
- [ ] 10.3 `uv run pydocstyle --convention=google sleap_roots/circumnutation/` clean.
- [ ] 10.4 `uv lock --check` clean; `uv run mkdocs build` builds.
- [ ] 10.5 `npx openspec validate add-circumnutation-tier3b-spatial-cwt --strict` valid.
- [ ] 10.6 Coverage ≥84% project-wide (CI-enforced); aim ≥90% on `spatial_cwt.py`.
- [ ] 10.7 CI matrix Ubuntu/Windows/macOS green on the final commit (cross-OS determinism at `atol=1e-9`).
- [ ] 10.8 `/pre-merge` (Phase 3.5 `/review-pr`); after PR open + CI green: `/copilot-review` then post-PR `/review-pr`; reconcile all findings (each substantive code change ships a test).
