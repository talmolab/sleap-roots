# Design — add-circumnutation-plots (PR #16)

## Context

PR #16 graduates the `plotting` module from a stub to an implementation, the
diagnostic-plotting layer of the circumnutation program. After this change the
package has **12 implementation modules and 1 stub** (`parametric`, PR #11
remains deferred on #230 + a Phase-2 gravitropism experiment).

Roadmap row 16 scope: *"Scaleograms (Tier 1 + Tier 3); trail overlay
(κ-color-coded); 6-up plate panel; `--no-plots` flag."* PNGs are written to a
`plots/` subdirectory of the output directory.

`matplotlib` + `seaborn` are **already** dependencies — no new dependency.

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
  `wavelength [px]`); `TypeError` on any other type. `ridge_result` is optional.
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
  COI); labeled colorbar `power |C|²`. The overlaid ridge is the **raw**
  `extract_ridge` output (the scaleogram's own argmax, so the line matches the
  heatmap), not the post-filtered `smooth_ridge`.
- **trail_overlay:** the tip path `(x_smooth_px, y_smooth_px)` as a
  `LineCollection`, segments colored by `curvature_px_inv` on a diverging
  colormap (`RdBu_r`) with a **symmetric** norm centered at 0, limits
  `±q` where `q = 98th percentile of |κ| over finite values` (robust to the few
  near-blow-up frames that survive the NaN sweep); NaN-κ segments rendered in a
  neutral `set_bad` color; y-axis inverted (image y-down); equal aspect; colorbar
  `κ [px⁻¹]`.
- **plate_panel:** a 2×3 grid of trail overlays, one per plant; a **single**
  symmetric κ norm computed across all plants on the plate and applied to every
  subplot, with **one shared colorbar** (so equal color == equal curvature across
  plants); empty cells hidden when fewer than 6 plants.

### D5 — `save_plots` re-derives Results from `inputs` (reusing the exact tier helpers)
`compute_traits` discards the intermediate Results, so `save_plots` re-derives
them per plant by calling the **identical** helper functions the analysis used
(not a reimplementation), with the **same** `constants`. Plotted == analyzed by
construction.

- **Tier 1 (temporal):**
  `_geometry.project_to_growth_axis_perpendicular(tip_x, tip_y)` → if
  `not np.isfinite(raw_signal).any()` skip → `_noise.compute_sg_detrended(raw,
  window=int(constants.SG_WINDOW_DETREND), polynomial_order=int(constants.SG_DEGREE))`
  → if not finite-any skip → `temporal_cwt.compute_scaleogram(signal,
  cadence_s=cadence_s, constants=constants)` → `temporal_cwt.extract_ridge(...,
  constants=constants)`. (Mirrors `nutation.py:423-449`.)
- **Tier 3 (spatial):**
  `midline.reconstruct(x, y, cadence_s=cadence_s, constants=constants)` (this
  `MidlineResult` also feeds the trail/panel) → if `mr.is_degenerate` skip →
  `spatial_cwt.resample_curvature(mr.curvature_px_inv, mr.arc_length_px,
  mr.velocity_sub_noise_mask, constants=constants)` → if `rs.is_degenerate` skip →
  `spatial_cwt.compute_scaleogram(rs.kappa_uniform, rs.ds, constants=constants)`
  → `spatial_cwt.extract_ridge(..., constants=constants)`. (Mirrors
  `traveling_wave.py:248-273`.) `midline.reconstruct` is called **once** per
  plant and reused for the trail, the panel, and the spatial scaleogram chain.
- A plant whose Tier 1 (or Tier 3) chain is degenerate skips only its affected
  plot, logged at DEBUG; the rest still render.

### D5 — output location & naming
- `plots/` subdirectory of `out_dir`, created with `mkdir(parents=True,
  exist_ok=True)`. (Unlike `pipeline.save()`, which does not create directories,
  `plots/` is a fresh subdir `save_plots` owns. The `plots/` subdir also sidesteps
  the `run_metadata` stem issue #238.)
- Filenames keyed on per-plant identity (`{plate_id}`, `{plant_id}`) to guarantee
  uniqueness (#237 guard): `{plate_id}_plant{plant_id}_scaleogram_temporal.png`,
  `{plate_id}_plant{plant_id}_scaleogram_spatial.png`,
  `{plate_id}_plant{plant_id}_trail.png`, and one `{plate_id}_panel.png` per plate.

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
(the lines 782–822 pattern for non-`STUB_MODULES` impl callables).

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
separately asserted today. → Mitigation: if TDD reveals a degenerate plant, the
count drops; the test will assert the produced subset (via the `list[Path]`
returned by `save_plots`) — or this is recorded as a deviation — the assertion
will not be silently weakened.

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
   `("plotting", "scaleogram", 16)` from `STUB_MODULES` and add the explicit
   plotting callability tests.
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

- None outstanding. (The `save_plots` return-paths question was resolved — see D1
  "Return types"; the follow-up dedup/MCP issue was filed as #241.)
