# Tasks — add-circumnutation-tier2-psi-g (Tier 2 ψ_g)

Strict TDD: each scope unit ships a **failing test commit** (`test: … (TDD red)`)
then an **implementation commit** (`feat:/fix: … (TDD green)`). No bundling impl +
large test file into one commit (the PR #6 3b7b03e retrospective, #222). Fixup
commits for review findings ship a test alongside any substantive code change.

## 0. Pre-flight
- [ ] 0.1 Confirm branch `add-circumnutation-tier2-psi-g`; `uv sync`; baseline `uv run pytest tests/ -q` green.
- [ ] 0.2 Draft the `psig_long_consistency` follow-up issue in the vault (`c:\vaults\sleap-roots\circumnutation\`); show the user; do NOT post until OK.

## 1. `_geometry.compute_signed_area` (commit pair)
- [ ] 1.1 RED: test the **absolute anchor** — `compute_signed_area([0,1,1,0],[0,0,1,1]) == -1.0` exactly (y-down negation of standard Shoelace `+1.0`); `< 3` points → `0.0`; non-finite → NaN.
- [ ] 1.2 GREEN: implement `compute_signed_area(x, y)` in `_geometry.py` next to `compute_psi_g`; Google docstring documenting the load-bearing sign convention (`+area ↔ handedness +1`).

## 2. `psi_g.compute` schema/structure (commit pair)
- [ ] 2.1 RED: returns `pd.DataFrame`; 8 identity + 4 trait columns in declared order; dtypes 3×float64 + 1×int64; 5-tuple uniqueness. Stub emits the degenerate-row defaults.
- [ ] 2.2 GREEN: implement the `compute` shell — `groupby` 5-tuple → `_compute_one_track` → per-plant template merge → declared column order → dtype enforcement loop (`handedness → fillna(0).astype(int64)`, others `float64`).
- [ ] 2.3 RED+GREEN: units-vocabulary guard test (all 4 trait units ∈ `PIPELINE_UNIT_VOCABULARY`).

## 3. Input-validation boundary (commit pair)
- [ ] 3.1 RED: non-DataFrame `trajectory_df`; invalid `trajectory_df` (delegates to `_validate_trajectory_df`); bad `cadence_s` value→ValueError / type→TypeError (reuse `temporal_cwt._validate_cadence_s`); bad `constants` type→TypeError; invalid SG override (even `SG_WINDOW_DETREND`)→ValueError naming the field.
- [ ] 3.2 GREEN: implement `_check_constants` (SG fields only; defer CWT-field validation to `compute_scaleogram`) + wire validators.

## 4. Raw, CWT-free traits (commit pair)
- [ ] 4.1 RED: `handedness` (planted CW/CCW via `generate_trajectory(handedness=±1)`) ∈ {−1,0,+1} with the `1e-9` zero-guard; `delta_E_amplitude_proxy_px_per_frame` on a known constant-speed synthetic; handedness↔area agreement (CCW→+/+, CW→−/−).
- [ ] 4.2 GREEN: implement step-4 of `_compute_one_track` (finite mask → ψ_g endpoints → `handedness`; `np.median` speed → `delta_E`; `compute_signed_area` → `helix`). These pass for `N ≥ 3`, independent of the CWT path.

## 5. `T_psig_median_s` CWT path (commit pair)
- [ ] 5.1 RED: known-period synthetic (`T ∈ {3333, 4500}`, `n_frames ≥ 575`, `noise_sigma_px=0`) recovers `T_psig_median_s` within ±10% (cite nutation `test_2C2`); assert **no** `RuntimeWarning`.
- [ ] 5.2 GREEN: implement step-5 — length guard (`len(ψ_g) < SG_WINDOW_DETREND` → NaN); `compute_sg_detrended`; **zero-energy guard** (`np.allclose(detrended, 0.0)` → NaN, skip CWT); `try: compute_scaleogram → extract_ridge → smooth_ridge`, COI-interior `~in_coi` nanmedian with empty/all-NaN guard (no `RuntimeWarning`); `except ValueError` + post-detrend finite-check as `# pragma: no cover` defensive mirrors.

## 6. Degenerate / edge cases (commit pair)
- [ ] 6.1 RED: `N < 3` (2-row direct DataFrame) → all-degenerate row; `3 ≤ N < 24` (15-frame) → `T_psig=NaN`, raw traits defined, no exception; **stationary `N≥24`** (`generate_trajectory(amplitude_px=0, growth_rate_px_per_frame=0, noise_sigma_px=0)`) → `T_psig=NaN` (zero-energy guard) AND no `RuntimeWarning`; NaN-injection (à la `test_2F7b`) handled gracefully.
- [ ] 6.2 GREEN: implement step-1/2 finite-mask + `N<3` short-circuit; confirm the zero-energy and length guards cover the table.

## 7. Cross-tier consistency (commit pair)
- [ ] 7.1 RED: synthetic convention-lock — angle-identity fixture (`amplitude_px=0`, θ ∈ {0.3, −2.0}) asserts `abs(wrap_to_pi(circular_mean(ψ_g) − (π/2 − θ))) < 1e-6` and `handedness == 0`; handedness fixture (`amplitude_px>0`, `handedness=±1`) asserts the planted sign.
- [ ] 7.2 GREEN: implement any helper needed (`wrap_to_pi`, `circular_mean`) in the test or a shared spot; ensure RED passes.
- [ ] 7.3 GREEN-phase: plate-001 reconciliation test over the 6 proofread tracks (mirror the test-local `_load_proofread_track_df` loader) — skip NaN `principal_axis_angle` tracks; assert `≥N/6` within `_PSIG_AXIS_RECONCILE_TOL_RAD`; capture `N` + tolerance from a real run; "GREEN-phase Reconciliation" docstring.

## 8. Foundation-test migration (commit pair)
- [ ] 8.1 RED: update `tests/test_circumnutation_foundation.py` — remove `psi_g` from `STUB_MODULES` + `STUBS_WITH_CONSTANTS_KWARG`; add `("psi_g","compute")` to `IMPLEMENTATIONS_WITH_CONSTANTS_KWARG` + the matching `elif module_name == "psi_g":` branch in `test_implementation_accepts_constants_kwarg`; add `psi_g` to the explicit `test_module_logger_is_namespaced` list; the "Calling each remaining stub" test now enumerates 5 stubs.
- [ ] 8.2 GREEN: ensure the renamed `psi_g.compute` satisfies all foundation tests; multi-track integration test.

## 9. Docs + theory reconciliation
- [ ] 9.1 Patch `docs/circumnutation/theory.md` §7.3 `handedness` row → "net unwrapped ψ_g rotation over all finite frames (COI-free)"; add an Appendix B (Corrections) line citing PR #7 + the `angular_amplitude` COI-free precedent.
- [ ] 9.2 Fix the stale `atan2(dy/dt, dx/dt)` docstring carried by the old `psi_g.py` stub (the impl reuses `_geometry.compute_psi_g`'s locked `atan2(dx, dy)`).
- [ ] 9.3 `docs/changelog.md` entry; confirm mkdocstrings reference page renders `psi_g.compute` + `_geometry.compute_signed_area`.

## 10. Verification gates (must pass before requesting merge)
- [ ] 10.1 `uv run pytest tests/ -q` green; ≥ 84% project coverage; aim ≥ 90% on `psi_g.py` + the new `_geometry` helper (defensive branches `# pragma: no cover` with invariant comments).
- [ ] 10.2 `uv run black --check sleap_roots tests` clean.
- [ ] 10.3 `uv run pydocstyle --convention=google sleap_roots/circumnutation/` clean.
- [ ] 10.4 `uv lock --check` clean; `uv run mkdocs build` builds.
- [ ] 10.5 `npx openspec validate add-circumnutation-tier2-psi-g --strict` valid.
- [ ] 10.6 CI matrix Ubuntu/Windows/macOS green on the final commit.
- [ ] 10.7 `/copilot-review` + `/review-pr` findings reconciled or deferred to tracked issues.
