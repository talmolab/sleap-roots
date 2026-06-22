# Tasks — add-circumnutation-plots (PR #16)

TDD throughout: write the RED test first, watch it fail for the right reason, implement to GREEN, refactor. The first commit that makes any `plotting` callable non-raising MUST be **atomic** with the foundation migration (Task 2) — otherwise `test_stub_callable_raises_with_correct_pr` (parametrized over `STUB_MODULES`) and the logger-namespace test go red.

## 1. Test scaffolding + Agg backend

- [ ] 1.1 Create `tests/test_circumnutation_plotting.py`. Add an autouse fixture forcing the headless backend: `matplotlib.use("Agg", force=True)` (RED first: assert `matplotlib.get_backend().lower() == "agg"`). Add small builders for valid `ScaleogramResult`/`SpatialScaleogramResult`/`RidgeResult`/`SpatialRidgeResult`/`MidlineResult` fixtures (reuse the patterns in `test_circumnutation_spatial_cwt.py` / the tier tests; do NOT hand-roll frozen-dataclass fields that drift from source).
- [ ] 1.2 Add a teardown/assert that no figures leak: each test that builds a figure calls `plt.close(fig)`; add a final `assert plt.get_fignums() == []` helper used by the public-fn smoke tests.

## 2. Foundation migration (ATOMIC with the first non-raising callable)

- [ ] 2.1 RED: write the new plotting callability tests in `test_circumnutation_foundation.py` (mirror the PR #15 `aggregation` callability scenario) — `plotting.scaleogram(valid_result, tmp_path/"s.png")` returns a `Path` and does not raise; `trail_overlay`/`plate_panel`/`save_plots` likewise. These fail while the stub still raises.
- [ ] 2.2 Remove `("plotting", "scaleogram", 16)` from `STUB_MODULES`; update the "Calling each remaining stub" test/comment to the 1 remaining stub (`parametric`). Add `plotting` to the explicit `test_module_logger_is_namespaced` list. Confirm `scaleogram` is NOT added to `IMPLEMENTATIONS_WITH_CONSTANTS_KWARG` (no `constants=` kwarg).
- [ ] 2.3 GREEN: land 2.1–2.2 together with Task 3's first non-raising `scaleogram` so the foundation suite never goes red. Run `uv run pytest tests/test_circumnutation_foundation.py -q`.

## 3. `scaleogram` (temporal + spatial, polymorphic)

- [ ] 3.1 RED (structural): `_build_scaleogram_figure(result)` returns a `matplotlib.figure.Figure`; for a temporal `ScaleogramResult` assert 2 axes (heatmap + colorbar), a `QuadMesh` artist, x-label `time [s]`, y-label `period [s]`, log y-scale, colorbar label `power |C|²`. For a `SpatialScaleogramResult` assert `arc length [px]` / `wavelength [px]`.
- [ ] 3.2 RED: `LogNorm` floor — a result with exact-zero power cells must not raise (assert no `ValueError`/warning); the norm `vmin` is the smallest positive power.
- [ ] 3.3 RED: COI dimming — assert the `coi_mask==True` region is visually de-emphasized (an overlay/alpha artist present); confirm polarity (True == dimmed == unreliable).
- [ ] 3.4 RED: ridge overlay — passing a matching `RidgeResult` adds a ridge line; segments inside COI (`in_coi==True`) are faded. Passing a mismatched ridge type raises `TypeError` naming both expected types. Passing a non-result `scaleogram_result` raises `TypeError`.
- [ ] 3.5 RED (smoke): public `scaleogram(result, out_path)` writes a non-empty PNG, returns `out_path`, closes the figure (`plt.get_fignums()==[]`).
- [ ] 3.6 GREEN: implement `_build_scaleogram_figure` + `scaleogram` (savefig + `plt.close` in `finally`; return `out_path`). Module-level `_DPI`, `_FIGSIZE_SCALEOGRAM`, `_CMAP_SCALEOGRAM`.

## 4. `trail_overlay` (κ-color-coded, y-down)

- [ ] 4.1 RED (structural): `_build_trail_figure(midline_result)` returns a Figure with a `LineCollection` whose `get_array()` length == (n_points − 1) (per-segment, midpoint-averaged κ), a diverging colormap, a symmetric norm centered at 0 with limits ±(98th-pct |κ| over finite), y-axis inverted (image-down), equal aspect, colorbar `κ [px⁻¹]`.
- [ ] 4.2 RED: NaN-κ handling — a `MidlineResult` with NaN curvature segments renders them via a copied colormap's `set_bad` (assert the global colormap is unmodified — `cmap.copy()` used).
- [ ] 4.3 RED (smoke): `trail_overlay(midline_result, out_path)` writes a non-empty PNG, returns `out_path`, closes the figure.
- [ ] 4.4 GREEN: implement `_build_trail_figure` + `trail_overlay`. Module-level `_FIGSIZE_TRAIL`, `_CMAP_KAPPA`, `_KAPPA_PCT = 98.0`. Factor the segment+norm helper so `plate_panel` reuses it.

## 5. `plate_panel` (2×3, shared norm + colorbar)

- [ ] 5.1 RED (structural): `_build_panel_figure(midline_results)` returns a Figure with 6 axes; every subplot uses the SAME κ norm (one norm computed across all plants' finite κ); exactly one shared colorbar (via an explicit `ScalarMappable(norm, cmap)`); empty cells hidden when < 6 results; deterministic cell→plant order.
- [ ] 5.2 RED (smoke): `plate_panel(midline_results, out_path)` writes a non-empty PNG, returns `out_path`, closes the figure.
- [ ] 5.3 GREEN: implement `_build_panel_figure` + `plate_panel`. Module-level `_FIGSIZE_PANEL`.

## 6. `save_plots` orchestrator (re-derivation + plots/ + enabled + filenames)

- [ ] 6.1 RED: `save_plots(inputs, out_dir, enabled=False)` writes nothing, creates no `plots/`, returns `[]`, logs one INFO line.
- [ ] 6.2 RED: re-derivation fidelity — `save_plots` groups `inputs.trajectory_df` by `_IDENTITY_5_TUPLE` (`dropna=False, sort=False`, matching the tiers) and re-derives via the EXACT tier helpers: Tier 1 `_select_signal(group,"lateral")` (in `try/except ValueError`) → finite-any short-circuit → `_noise.compute_sg_detrended(raw, window=int(SG_WINDOW_DETREND), polynomial_order=int(SG_DEGREE))` → finite-any short-circuit → `try: temporal_cwt.compute_scaleogram(signal, cadence_s, constants) except ValueError` → `extract_ridge` → `smooth_ridge` (overlaid); Tier 3 finite-tip mask → `midline.reconstruct` → degeneracy gate → `spatial_cwt.resample_curvature` → degeneracy gate → `compute_scaleogram` → `extract_ridge`. Assert (e.g. via monkeypatched spies) the same helpers/kwargs are called.
- [ ] 6.3 RED: filenames keyed on integer `track_id` (`plant{track_id}_scaleogram_temporal.png`, `..._scaleogram_spatial.png`, `..._trail.png`, `panel.png`); NOT on `plate_id`/`plant_id`. Assert a row with `NaN` plate_id/plant_id still yields unique `track_id`-keyed names.
- [ ] 6.4 RED: degenerate-plant skip — a plant whose chain hits a gate omits only its affected plot(s) (DEBUG-logged), others still render; return list reflects the subset.
- [ ] 6.5 GREEN: implement `save_plots` (`plots/` via `mkdir(parents=True, exist_ok=True)`; INFO per-plate count, DEBUG per-plant). Reuse Tasks 3–5 renderers.

## 7. Real plate-001 integration test (the validation)

- [ ] 7.1 RED→GREEN (skipif-guarded, reuse `_load_plate001_inputs()`): `save_plots(inputs, out/"out")` writes its PNG set into `out/"out"/plots/`. Assert the returned `list[Path]` count equals the self-consistent derived count `(#finite T_nutation_median) + (#finite traveling_wave_residual) + (#non-degenerate trails) + 1`, every file non-empty, no exception. Separately assert `enabled=False` produces no files and `[]`.
- [ ] 7.2 Confirm the temporal chain succeeds for the plate-001 plants (record the actual per-plant pass counts in a comment); if any plant is degenerate, the derived count handles it — do NOT hard-code 19. If reality diverges from the design's "all 6 non-degenerate" expectation, update design.md with a `### Why N instead of 6?` note.

## 8. Docstrings, deviation, and verification gates

- [ ] 8.1 Rewrite the `plotting.py` module + function docstrings (Google-style; read 2–3 existing circumnutation docstrings first for convention). Drop the `L_gz` arc-length-marker claim entirely (deviation; #230). No `L_gz` parameter anywhere.
- [ ] 8.2 Run the full suite + gates: `uv run pytest tests/ -q` (green); `black --check`; `pydocstyle --convention=google sleap_roots/circumnutation/`; `uv lock --check`; `uv run mkdocs build`; `npx openspec validate add-circumnutation-plots --strict`. Aim ≥90% coverage on `plotting.py`; project ≥84% (CI-enforced).
- [ ] 8.3 Reconcile implementation vs proposal/spec/design (every named helper/kwarg/return type matches reality); if any deviation was forced, update proposal.md/spec.md/design.md with a `### Why …?` note and file an issue if a bug/workaround was involved.
