# Design — add-circumnutation-plots (PR #16)

## Context

PR #16 graduates the `plotting` module from a stub to an implementation, the
diagnostic-plotting layer of the circumnutation program. After this change the
package has **12 implementation modules and 1 stub** (`parametric`, PR #11
remains deferred on #230 + a Phase-2 gravitropism experiment).

Roadmap row 16 scope: *"Scaleograms (Tier 1 + Tier 3); trail overlay
(κ-color-coded); 6-up plate panel; `--no-plots` flag."* PNGs are written to a
`plots/` subdirectory of the output directory.

`matplotlib` is **already** a dependency — no new dependency. The renderers are
**matplotlib-only**; `seaborn` (also already present) is not used.

### Substrate (verified against current code)

- `temporal_cwt.ScaleogramResult` — `scaleogram` (complex `(n_scales, n_frames)`),
  `scales`, `periods_s`, `frequencies_hz`, `coi_mask` (bool, `True`=inside-COI=
  unreliable), `cadence_s`, `wavelet`. `RidgeResult` — `frame_indices`,
  `periods_s`, `amplitudes`, `powers`, `in_coi`.
- `spatial_cwt.SpatialScaleogramResult` — `scaleogram` (complex
  `(n_scales, n_samples)`), `scales`, `wavelengths_px`, `spatial_freqs_px_inv`,
  `coi_mask`, `ds`, `wavelet`. `SpatialRidgeResult` — `position_indices`,
  `wavelengths_px`, `amplitudes`, `powers`, `in_coi`.
- `midline.MidlineResult` — `frame_indices`, `x_smooth_px`, `y_smooth_px`
  (**y-down image coordinates**), `speed_px_per_frame`, `arc_length_px`,
  `curvature_px_inv` (**signed** κ, px⁻¹; non-finite blow-ups swept to NaN),
  `velocity_sub_noise_mask`, `cadence_s`, `sg_window`, `sg_degree`,
  `sigma_v_px_per_frame`, `noise_mask_k`, `is_degenerate`.
- `curvature_px_inv` is the **signed** differential-geometry curvature
  (`_geometry.compute_path_curvature:264`): `+κ` = clockwise / visual-right turn
  in the y-down image frame → a **diverging** colormap centered at 0 is correct.
- `pipeline.CircumnutationPipeline.compute_traits` calls each tier's `compute()`
  once and keeps only the trait columns; the intermediate `ScaleogramResult` /
  `SpatialScaleogramResult` / `MidlineResult` objects are computed **inside** the
  tiers (`pipeline.py:149-155`) and discarded.
- Stub registry: `STUB_MODULES` in `tests/test_circumnutation_foundation.py:73-76`.

## Goals / Non-Goals

**Goals**
- Implement `scaleogram`, `trail_overlay`, `plate_panel`, and a `save_plots`
  orchestrator in `sleap_roots/circumnutation/plotting.py`.
- Deterministic, OS-portable tests (no pixel baselines).
- Validate on real plate-001.

**Non-Goals**
- The `--no-plots` **CLI flag** and the `Series → CircumnutationInputs` adapter
  (PR #17). PR #16 supplies the `enabled` parameter the flag will set.
- The `L_gz` arc-length marker (blocked on #230 — see Deviation below).
- The Tier 2 (ψ_g) scaleogram — out of roadmap-row-16 scope (Tier 1 + Tier 3 only).
- mm/calibrated axes (CC-3 pure-pixel; mm is a downstream `convert_to_mm` concern).

## Decisions

### D1 — Public API (3 renderers + orchestrator)
- `scaleogram(scaleogram_result, out_path, *, ridge_result=None)` — **keeps its
  canonical name** (stub contract). Polymorphic via `isinstance` dispatch:
  `ScaleogramResult` → temporal axes (`time [s]` / `period [s]`),
  `SpatialScaleogramResult` → spatial axes (`arc length [px]` /
  `wavelength [px]`); `TypeError` on any other scaleogram type. `ridge_result`
  is optional, but **its type must agree with the scaleogram type**
  (`ScaleogramResult`↔`RidgeResult`, `SpatialScaleogramResult`↔
  `SpatialRidgeResult`) — the two ridge types expose different fields
  (`frame_indices`/`periods_s` vs `position_indices`/`wavelengths_px`), so a
  mismatched pair is rejected with a `TypeError` naming both expected types
  rather than left to surface as an `AttributeError`. Tested (R3).
- `trail_overlay(midline_result, out_path)` — consumes a `MidlineResult`.
- `plate_panel(midline_results, out_path)` — consumes an ordered collection of
  `MidlineResult` (one per plant).
- `save_plots(inputs, out_dir, *, constants=None, enabled=True) -> list[Path]` —
  orchestrator, signature symmetric with `compute_traits(inputs)`; returns the
  list of written PNG `Path`s (empty list when `enabled=False`).

**Return types (file-path-reference convention, mirrors sleap-roots-analyze
`files_generated`):** each renderer returns its `out_path` (`Path`); `save_plots`
returns the aggregated `list[Path]`. This surfaces the generated artifacts to
callers (PR #17 CLI, a future MCP wrapper) as path references rather than forcing
a directory re-scan. The graduated `scaleogram` docstring therefore documents
`Returns: the out_path` (superseding the stub's `Returns: None`).

### D1b — `--no-plots` threading
`save_plots(..., *, enabled=True)`. When `enabled=False` it returns immediately,
writes no PNGs, and logs one INFO line (CC-9). PR #17 maps `--no-plots →
enabled=False`. Positive boolean keeps the body free of double-negative logic and
the orchestrator owns the guard + log.

### D2 — Test strategy (Agg + structural/smoke; no pixel baselines)
- Force the `Agg` backend via an autouse fixture (`matplotlib.use("Agg",
  force=True)`); backend/rcParams are process-global.
- Each renderer splits into a pure `_build_*_figure(...) -> matplotlib.figure.Figure`
  (returns the Figure for introspection) and the public function, which calls the
  builder, `savefig`s to `out_path`, `plt.close(fig)` in a `finally` (closes
  even on exception), and returns `out_path`.
- **Structural** assertions on the builder (axis count, axis labels/units,
  expected artist types — `QuadMesh` for the heatmap, `LineCollection` for the
  κ-trail, 6 axes for the panel, colorbar present). Tests close the figures they
  introspect.
- **Smoke** assertions on the public fn (returns `out_path`, PNG exists, size >
  0, no exception).
- No pixel comparison — plots are not bit-reproducible across the tri-OS CI matrix.

### D3 — What each plot renders
- **scaleogram:** `pcolormesh` of power `|C|² = |scaleogram|²` with a `LogNorm`
  color norm; y-axis log-scaled (`period [s]` / `wavelength [px]`); x-axis in
  physical units (`time [s] = frame_index·cadence_s` / `arc length [px] =
  sample_index·ds`); the COI region (`coi_mask`) dimmed as unreliable; the
  optional ridge overlaid as a line (solid where `in_coi==False`, faded inside
  COI); labeled colorbar `power |C|²`. **`LogNorm` floor (R7):** power has exact
  zeros at degenerate cells, and `LogNorm` rejects `vmin<=0`; set `vmin` to the
  smallest *positive* power (or a low positive percentile) so the norm never sees
  a non-positive bound. **Which ridge is overlaid (R2b):** `save_plots` passes the
  ridge that the tier *analyzed* — for **temporal**, that is the
  `smooth_ridge(extract_ridge(...))` output (its median-filtered period is what
  produces `T_nutation_median`), so the overlay reflects the measurement (it
  tracks the argmax closely, a light median filter); for **spatial**, there is no
  smoothing stage, so the raw `extract_ridge` *is* the analyzed ridge. (This
  supersedes an earlier draft that overlaid the raw temporal ridge and over-claimed
  "plotted == analyzed" — see Review reconciliation R2b.) **Spatial faithfulness
  caveat (R-S1):** the spatial `wavelengths_px` are raw pywt-convention values and
  the overlaid spatial ridge is **ungated**, whereas the emitted `lambda_spatial_*`
  traits are COI-gated and cgau2-**calibrated** (theory.md §7.4 notes 2–3). So the
  spatial wavelength axis is labeled *uncalibrated convention px* and spatial ridge
  positions with `in_coi==True` are faded exactly as the temporal ridge — a reader
  must not mistake the plotted wavelength for the authoritative
  `lambda_spatial_median_px`.
- **trail_overlay:** the tip path `(x_smooth_px, y_smooth_px)` as a
  `LineCollection` of **N−1 segments** from consecutive point pairs; per-segment
  color = the **midpoint average** of the two endpoints' `curvature_px_inv`, so
  `set_array` receives a length-(N−1) array (NOT length N) — a segment adjacent to
  a NaN-κ frame averages to NaN (R7). Diverging colormap (`RdBu_r`) with a
  **symmetric** norm centered at 0, limits `±q` where `q = 98th percentile of |κ|
  over finite values` (robust to the few near-blow-up frames that survive the NaN
  sweep); NaN segments rendered via a **copied** colormap's `set_bad`
  (`cmap.copy()` first — never mutate a global colormap, same process-global hazard
  as the Agg backend); y-axis inverted (image y-down); equal aspect; colorbar
  `κ [px⁻¹]` with **`extend="both"`** so the ~2% clipped at the 98th-pct limits is
  visible (R-S2). **All-NaN-κ edge (R-S6):** a non-degenerate `MidlineResult` whose
  `curvature_px_inv` is entirely NaN gives `np.percentile([], 98)` no finite input;
  such a plant is skipped (DEBUG-logged) or rendered with a default symmetric limit,
  never crashing the norm — and the `plate_panel` pooled-κ norm guards the
  empty-pool case the same way.
- **plate_panel:** a 2×3 grid of trail overlays, one per plant; a **single**
  symmetric κ norm computed across all plants on the plate and applied to every
  subplot, with **one shared colorbar** driven by an explicit
  `matplotlib.cm.ScalarMappable(norm, cmap)` passed to `fig.colorbar(sm,
  ax=axes)` (R7); empty cells hidden when fewer than 6 plants. Cell→plant mapping
  follows the same deterministic groupby order as the re-derivation (R-groupby),
  so the structural test can assert which plant is in which cell.

### D5 — `save_plots` re-derives Results from `inputs` (reusing the exact tier helpers)
`compute_traits` discards the intermediate Results, so `save_plots` re-derives
them per plant by calling the **identical** helper functions the analysis used
(not a reimplementation), with the **same** `constants`. Plotted == analyzed by
construction — with one documented temporal-ridge nuance (R2b): the period trait
`T_nutation_median` derives from the *smoothed* ridge, so `save_plots` overlays
the smoothed ridge (see D3) to keep that equality honest.

**Grouping contract (R-groupby).** `save_plots` iterates plants by grouping
`inputs.trajectory_df` on the `_IDENTITY_5_TUPLE` (`series, sample_uid, plate_id,
plant_id, track_id`) with `dropna=False, sort=False` — the **exact** pattern
`nutation.compute` and `traveling_wave.compute` use. Identical grouping is what
makes plotted == analyzed and fixes a deterministic plant order (drives panel
cell assignment and the order of the returned `list[Path]`, which the integration
test asserts).

- **Tier 1 (temporal)** — mirrors `nutation._compute_one_track`
  (`nutation.py:418-450`) guard-for-guard, reusing its helpers:
  `try: raw = _select_signal(group, "lateral") except ValueError: skip` (the
  `lateral` branch is `project_to_growth_axis_perpendicular`, which **raises**
  `ValueError` on a NaN/inf tip column — real `.slp` tracks carry NaN frames, so
  this guard is load-bearing, not the `isfinite(...).any()` check) → if
  `not np.isfinite(raw).any()` skip → `signal = _noise.compute_sg_detrended(raw,
  window=int(constants.SG_WINDOW_DETREND), polynomial_order=int(constants.SG_DEGREE))`
  → if `not np.isfinite(signal).any()` skip → `try: sg =
  temporal_cwt.compute_scaleogram(signal, cadence_s, constants=constants) except
  ValueError: skip` (the "signal too short for the scale grid" case) →
  `raw_ridge = temporal_cwt.extract_ridge(sg, constants=constants)` →
  `ridge = temporal_cwt.smooth_ridge(raw_ridge, constants=constants)` (the overlaid
  ridge, per D3/R2b). `cadence_s` is passed as a keyword (`cadence_s=cadence_s`),
  matching the `nutation.py:444` call site.
- **Tier 3 (spatial)** — mirrors `traveling_wave._compute_one_track`
  (`traveling_wave.py:241-273`): first **drop non-finite tips**
  (`finite = np.isfinite(x) & np.isfinite(y); x, y = x[finite], y[finite]`)
  because `midline.reconstruct` *rejects* non-finite input (R2c) →
  `mr = midline.reconstruct(x, y, cadence_s=cadence_s, constants=constants)` (this
  `MidlineResult` also feeds the trail/panel) → if `mr.is_degenerate` skip →
  `rs = spatial_cwt.resample_curvature(mr.curvature_px_inv, mr.arc_length_px,
  mr.velocity_sub_noise_mask, constants=constants)` → if `rs.is_degenerate` skip →
  `sg = spatial_cwt.compute_scaleogram(rs.kappa_uniform, rs.ds, constants=constants)`
  → `ridge = spatial_cwt.extract_ridge(sg, constants=constants)` (no smoothing
  stage exists spatially → the raw ridge IS analyzed). `midline.reconstruct` is
  called **once** per plant and reused for the trail, the panel, and the spatial
  scaleogram chain.
- A plant whose Tier 1 (or Tier 3) chain hits a skip/degeneracy gate omits only
  its affected plot, logged at DEBUG; the rest still render. The panel includes a
  plant's trail whenever its `MidlineResult` is non-degenerate.

### D5 — output location & naming
- `plots/` subdirectory of `out_dir`, created with `mkdir(parents=True,
  exist_ok=True)`. (Unlike `pipeline.save()`, which does not create directories,
  `plots/` is a fresh subdir `save_plots` owns. The `plots/` subdir also sidesteps
  the `run_metadata` stem issue #238.)
- **Provenance sidecar (R-S4):** when any plot is written, `save_plots` also writes
  `plots/plots_metadata.json` — strict-JSON (finite floats; numpy/`Path` coerced to
  native, the sleap-roots-analyze #241 convention) — recording `_CONSTANTS_VERSION`,
  the resolved display constants (`_KAPPA_PCT`, colormaps, `_DPI`, figsizes), the
  per-plant `_IDENTITY_5_TUPLE` tuples plotted, the written PNG filenames, and a
  relative back-reference to the run's `run_metadata.json`. This ties each PNG to
  the run + constants that produced it (the CSV's `run_metadata.json` otherwise has
  no link to the plots). No sidecar is written when `enabled=False`.
- Filenames keyed on **`track_id`** — the one identity field
  `_validate_integer_identity` guarantees is integer-valued and finite
  (`_types.py`). `plate_id`/`plant_id` are documented as *aspirational* (no
  upstream produces them; they may populate as `NaN`), so keying on them would
  collapse every plant to `nan_plantnan_*.png` on un-decorated `.slp` data — the
  very #237 collision we mean to guard (R-filenames). Scheme:
  `plant{track_id}_scaleogram_temporal.png`, `plant{track_id}_scaleogram_spatial.png`,
  `plant{track_id}_trail.png`, and one `panel.png` per `save_plots` call (one plate
  per output dir, matching the one-CSV-per-dir contract). When `plate_id` *is*
  populated it may be prefixed (`{plate_id}_…`), but `track_id` alone carries the
  uniqueness guarantee.

### D6 — constants & figure-leak hygiene
- Display knobs are **structural**, not analysis parameters, so they live as
  module-level constants in `plotting.py` (`_DPI`, `_FIGSIZE_SCALEOGRAM`,
  `_FIGSIZE_TRAIL`, `_FIGSIZE_PANEL`, `_CMAP_SCALEOGRAM`, `_CMAP_KAPPA`,
  `_KAPPA_PCT = 98.0`). They are **not** added to the `ConstantsT` override-bag;
  **`_CONSTANTS_VERSION` stays 6** (no provenance churn).
- Every public renderer `plt.close(fig)` in a `finally`. `matplotlib` figure state
  is process-global; no figure is left open.

### D8 — axes (CC-3) & logging (CC-9)
- CC-3 pure-pixel: every axis/colorbar in pixel-native units, never mm —
  `x [px]`/`y [px]` (trail), `arc length [px]`/`wavelength [px]`/`spatial
  frequency [px⁻¹]` (spatial scaleogram), `time [s]`/`period [s]` (temporal
  scaleogram), `κ [px⁻¹]` (trail colorbar), `power |C|²` (heatmap colorbar).
  Bracketed unit suffixes align with #222. No `px_per_mm` anywhere.
- CC-9 logging: `save_plots` logs per-**plate** at INFO ("wrote N plots to
  …/plots/"), per-**plant** plot writes at DEBUG, the `enabled=False` skip at
  INFO. `plotting.py` already declares `logger = logging.getLogger(__name__)`.

### Atomic foundation migration
The first non-raising `plotting` commit MUST be atomic with removing
`("plotting", "scaleogram", 16)` from `STUB_MODULES`
(`tests/test_circumnutation_foundation.py:73-76`) so the foundation suite never
goes red. `scaleogram` has **no** `constants=` kwarg, so it is **not** added to
`IMPLEMENTATIONS_WITH_CONSTANTS_KWARG`; instead add explicit callability tests
(the lines 782–822 pattern for non-`STUB_MODULES` impl callables). **Also (R5):**
`plotting` is currently covered by the logger-namespace test only via its
`STUB_MODULES` membership; removing that row drops it from that coverage, so the
same commit MUST add `plotting` to the explicit list in
`test_module_logger_is_namespaced` (`test_circumnutation_foundation.py`).

## Alternatives Considered

- **API shape:** 3 renderers without an orchestrator (composition logic leaks to
  the caller, harder to unit-test here); a single `save_plots` with private
  renderers (drops the canonical public `scaleogram` name — conflicts with the
  stub contract). Chose 3 renderers + orchestrator.
- **Test strategy:** pixel baselines via `pytest-mpl` (flaky across
  matplotlib/freetype/font versions and the tri-OS CI; new dependency);
  pure smoke (won't catch wrong axis count, missing colorbar, mislabeled units).
  Chose Agg + structural/smoke.
- **Obtaining Results (D5):** expose the per-plant Results from the pipeline so
  plotting consumes them directly. Note there is **codebase precedent** for a
  pipeline surfacing intermediates: `trait_pipelines.compute_frame_traits`
  returns a full `Dict[str, Any]` of every DAG node
  (`trait_pipelines.py:292`), and `compute_plant_traits(..., return_non_scalar=
  False)` exposes the full non-scalar table behind an **opt-in flag**, slicing to
  the summary CSV columns by default (`trait_pipelines.py:340,395-398`). So the
  exposure could be a non-breaking opt-in flag (e.g. `compute_traits(...,
  return_results=False)`) preserving the current 3-tuple default — less invasive
  than a 4th positional return value. Still deferred from this PR (each tier must
  surface its Result objects); chose re-derive in `save_plots` for lowest blast
  radius — see the tradeoff note below. A separate
  `pipeline.compute_plot_artifacts()` method is the other candidate.

## Risks / Trade-offs

### Why re-derive instead of exposing Results? (recompute tradeoff → follow-up issue)
`save_plots` re-runs the per-plant CWT/midline chains that `compute_traits`
already ran inside `nutation.compute` (`nutation.py:443-449`) /
`traveling_wave.compute` (`traveling_wave.py:248-273`) and discarded — so a full
analysis-plus-plots run computes those scaleograms **twice**. This is a
deliberate tradeoff: re-derivation has the **lowest blast radius** (the five
tiers stay pure `DataFrame -> DataFrame`, `compute_traits`/`save` are untouched),
the recompute is **plate-scale and cheap**, and plots are optional
(`enabled`/`--no-plots`). The cleaner architecture — exposing the per-plant
Result artifacts from the pipeline (a 4th `compute_traits` return value, or a
`compute_plot_artifacts()` method) so plotting (and future consumers) need not
recompute — is tracked as **follow-up issue #241** (parent #197). Drift between the
plotted and analyzed signal is prevented by calling the **identical** helpers,
not by re-implementing the prep.

That follow-up should align with the sibling repo **sleap-roots-analyze**'s
serializable-result-types epic (#130; commits `a6ac418`/`c0d613a`/`f1167fe`):
frozen JSON-native result views with `to_dict()` + `to_json(allow_nan=False)`
(non-finite floats raise rather than emit `NaN`/`Infinity` tokens an MCP consumer
rejects) and **non-breaking `from_*_dict()` adapter** classmethods. Exposing the
circumnutation per-plant Results that way makes them **MCP-compatible** (the
sibling convention) *and* removes the plotting recompute in one move. Plots
themselves are surfaced as **file-path references** (sleap-roots-analyze's
`StepSummary.files_generated`), not embedded images — which motivates
`save_plots` returning the written PNG paths (adopted — see D1 "Return types").

### "Exact 19 PNGs" assumes all 6 plate-001 plants are non-degenerate
The D7 file-count assertion (6 plants × {temporal scaleogram, spatial scaleogram,
trail} + 1 panel = 19 PNGs) assumes no plant is degenerate. The existing
`traveling_wave_residual < 0.30` finite-for-all-6 invariant
(`test_circumnutation_pipeline.py:498-500`) confirms the **spatial** chain
succeeds for all 6; the **temporal** chain is very likely fine too but is not
separately asserted today. → Mitigation (R4): the integration test does **not**
hard-code 19. It derives the expected count from the per-plant DataFrame —
`expected = (#plants with finite T_nutation_median) + (#plants with finite
traveling_wave_residual) + (#plants with a non-degenerate trail) + 1 panel` — and
asserts the `list[Path]` returned by `save_plots` matches that **self-consistent**
count. On the current plate-001 that evaluates to 19, but a future degenerate
plant yields a correct lower count rather than a red magic-number assertion. The
count is never silently weakened.

## Deviation — no `L_gz` arc-length marker (D4)

The stub docstring promised *"tip-trail overlays with κ-color-coding and an
`L_gz` arc-length marker"* (`plotting.py:3-5`). `L_gz` (growth-zone length) is
blocked on #230 and is **not** computed anywhere in the pipeline. PR #16 **omits**
the marker entirely: the graduated module/function docstrings drop the claim, no
`L_gz` parameter is added to any function (nothing speculative or untested ships),
and a follow-up PR adds the marker + its API once #230 lands. (Mirrors PR #14's
TraitDef-DAG deviation discipline.)

## Migration Plan

1. Add `_build_*_figure` builders + public renderers + `save_plots` to
   `plotting.py` (TDD; structural tests first).
2. In the **same** commit as the first non-raising `plotting` callable, remove
   `("plotting", "scaleogram", 16)` from `STUB_MODULES`, add the explicit
   plotting callability tests, AND add `plotting` to the explicit
   `test_module_logger_is_namespaced` list (R5).
3. Update the spec: MODIFY "Package layout" (impl 11→12, stub 2→1; drop the
   plotting stub-callable row); ADD a "Circumnutation diagnostic plots API"
   requirement with callability scenarios for the new public symbols.
4. Real plate-001 integration test (D7).
5. Rollback: revert the `plotting.py` impl + restore the `STUB_MODULES` row
   (atomic, so the suite stays green either way).

## Tracking

- Tracking issue: **#242** (PR #16 body will `Closes #242`).
- Follow-up (MCP-compatible serializable Result views; removes recompute): **#241**.
- Parent epic: **#197**. L_gz marker blocked on **#230**.

## Open Questions

- None outstanding. **R-S4 resolved:** `save_plots` writes a
  `plots/plots_metadata.json` provenance sidecar (decided with the user; see D5
  output section). **#242 reconciliation:** the user chose to edit the tracking
  issue body to record the `track_id`-filename + derived-count supersessions; the
  edit is drafted to the vault for per-item OK before posting.

## Review reconciliation (adversarial critical-review of design.md, round 1)

Each BLOCKING/IMPORTANT finding and where it is now addressed:

- **R2a (BLOCKING) — Tier 1 re-derivation used the wrong guard mechanism.**
  `project_to_growth_axis_perpendicular` *raises* `ValueError` on NaN/inf tips
  (not all-NaN-return), so the draft's `isfinite(...).any()` check would never run
  — real `.slp` tracks would crash. **Fixed:** D5 Tier 1 now mirrors
  `nutation._compute_one_track` (`nutation.py:418-450`) guard-for-guard — reuses
  `_select_signal(group, "lateral")` inside `try/except ValueError`, both
  finite-any short-circuits, and the `try/except ValueError` around
  `compute_scaleogram`.
- **R2b (BLOCKING) — "plotted == analyzed" was false for the temporal ridge.**
  `T_nutation_median` derives from `smooth_ridge`, but the draft overlaid the raw
  ridge. **Fixed:** D5/D3 now overlay the **smoothed** ridge for temporal
  (`smooth_ridge(extract_ridge(...))`, returns `RidgeResult`) and the raw ridge for
  spatial (no smoothing stage exists there); the over-claim is corrected.
- **R-filenames (BLOCKING) — `{plate_id}_plant{plant_id}` collapses to
  `nan_plantnan`.** `plate_id`/`plant_id` are documented aspirational (may be NaN);
  only `track_id` is guaranteed integer by `_validate_integer_identity`. **Fixed:**
  D5 naming now keys on `track_id` (`plant{track_id}_*.png`), optional `plate_id`
  prefix only when populated.
- **R2c (IMPORTANT) — Tier 3 chain omitted the finite-tip pre-filter.**
  `midline.reconstruct` rejects non-finite input. **Fixed:** D5 Tier 3 now shows
  `finite = isfinite(x)&isfinite(y); x,y = x[finite],y[finite]` before `reconstruct`.
- **R3 (IMPORTANT) — cross-type ridge/scaleogram mismatch unguarded.** **Fixed:**
  D1 adds a ridge-pairing `TypeError` guard (temporal↔`RidgeResult`,
  spatial↔`SpatialRidgeResult`), with a test.
- **R5 (IMPORTANT) — logger-namespace coverage drops `plotting` on `STUB_MODULES`
  removal.** **Fixed:** Atomic-migration note + Migration Plan step 2 now add
  `plotting` to the explicit `test_module_logger_is_namespaced` list.
- **R7 (IMPORTANT) — matplotlib mechanics hand-waved.** **Fixed in D3:** `LogNorm`
  `vmin` set to smallest positive power (no `vmin<=0`); `LineCollection` uses N−1
  segments with midpoint-averaged κ for `set_array`; NaN via a **copied** colormap's
  `set_bad`; shared panel colorbar via an explicit `ScalarMappable(norm, cmap)`.
- **R4 (IMPORTANT) — "exact 19 PNGs" magic number.** **Fixed:** D7 risk note now
  derives the expected count from finite per-plant trait counts and asserts the
  returned `list[Path]` against that self-consistent value.
- **R-groupby (MINOR) — grouping/cell-order unspecified.** **Fixed:** D5 pins the
  `_IDENTITY_5_TUPLE` `dropna=False, sort=False` groupby (matching the tiers) and
  ties panel cell→plant order to it.
- **R-seaborn (MINOR).** **Fixed:** Context now states matplotlib-only; seaborn unused.

Confirmed correct as written (no change): COI mask polarity (`True`=inside=
unreliable), signed-κ diverging colormap, `isinstance` dispatch validity (classes
are distinct), the no-`constants=`/explicit-callability migration logic, and
`CircumnutationInputs` field coverage.

## Review reconciliation (openspec-review 5-subagent panel, round 1)

- **B1 (BLOCKING) — spec named the wrong Tier-1 helper.** The ADDED requirement
  listed `_geometry.project_to_growth_axis_perpendicular` for the re-derivation,
  but the analyzed path calls `nutation._select_signal(group, "lateral")` (which
  performs that projection after extracting the tip columns). **Fixed:** the spec
  ADDED requirement now names `nutation._select_signal`; tasks 6.2 require
  module-qualified helper calls. (design.md D5 was already correct.)
- **R-S1 (IMPORTANT, scientific) — spatial ridge faithfulness.** The plotted
  spatial `wavelengths_px` are raw/uncalibrated and the ridge is ungated, but the
  emitted `lambda_spatial_*` traits are COI-gated + cgau2-calibrated (theory §7.4).
  **Fixed:** D3 + spec label the spatial wavelength axis "uncalibrated convention
  px" and fade `in_coi==True` spatial ridge positions.
- **R-S2 (IMPORTANT) — κ clipping invisible.** The ±98th-pct symmetric norm clips
  ~2% of |κ|. **Fixed:** D3 + spec require colorbar `extend="both"`.
- **R-S6 (IMPORTANT) — all-NaN-κ non-degenerate plant.** `np.percentile([], 98)`
  is undefined. **Fixed:** D3 + spec skip/default-limit such a plant and guard the
  empty-pool panel; tasks 4.2b/5.1b test it.
- **R-spec-degenerate (IMPORTANT) — normative SHALL lacked a scenario.** **Fixed:**
  added "save_plots skips a degenerate plant" scenario to the ADDED requirement.
- **R-spec-constants (IMPORTANT) — silent normative removal.** The intro sentence
  requiring stubs to carry `constants=None` was dropped (parametric has none).
  **Fixed:** the PR #16 scope note now records the removal for auditability.
- **R-tests (IMPORTANT) — test gaps.** **Fixed in tasks.md:** G1 (no-mm/`px_per_mm`
  signature+label test), G2 (no-`L_gz` signature test), G4 (junk `ridge_result` →
  `TypeError`), G6 (spy at the `plotting.<helper>` binding; module-qualified calls),
  G7 (`plate_panel` >6-plants behavior decided+tested), autouse `plt.close("all")`
  teardown (no leak cascade), colormap-global-unchanged assertion, the atomic
  commit boundary made explicit (2.3) incl. the module-docstring `L_gz` drop, a
  named degenerate fixture for skip-branch coverage, and the integration test's
  file/skipif placement.
- **R-align (IMPORTANT) — tracking issue #242 drift.** PR #16 deliberately
  supersedes #242's `{plate_id}_plant{plant_id}` filename scheme (→ `track_id`) and
  its "exact 19-PNG" assertion (→ derived count). **Action pending user OK:** edit
  #242 (or comment) to record the supersession so issue and PR don't contradict.
- **R-S4 (IMPORTANT, provenance) — plots not traceable to the run.** Open decision
  surfaced to the user (sidecar vs CC-9 log); see Open Questions.
- **MINOR (design labels):** duplicate `### D5` headers and dangling `D4`/`D7`
  references noted by the spec reviewer; left as-is (the deviation/integration
  sections are clearly titled) — a renumber is deferred to avoid churn.
