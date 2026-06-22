## Why

PR #16 of the multi-PR circumnutation program (parent epic #197, roadmap row 16) adds the **diagnostic-plotting layer**. The pipeline (PR #14) and per-genotype aggregation (PR #15) emit numeric traits, but there is no way to *see* the CWT scaleograms, the curvature-coded tip trail, or a plate at a glance — the diagnostics that let a scientist sanity-check a run. This PR graduates the `plotting` module from a stub to an implementation, leaving `parametric` (PR #11, deferred on #230) as the lone stub.

Closes #242. Follow-up: #241 (expose per-plant Result artifacts as MCP-compatible serializable views, removing the recompute this PR accepts).

## What Changes

- Implement `sleap_roots.circumnutation.plotting` with four public callables (see `design.md` D1):
  - `scaleogram(scaleogram_result, out_path, *, ridge_result=None) -> Path` — **keeps its canonical stub name**; polymorphic over temporal `ScaleogramResult` and spatial `SpatialScaleogramResult`; power `|C|²` heatmap (`LogNorm`), COI dimming, optional ridge overlay; ridge/scaleogram type-pairing guard.
  - `trail_overlay(midline_result, out_path) -> Path` — κ-color-coded (signed, diverging, symmetric-norm) tip trail, image-y-down.
  - `plate_panel(midline_results, out_path) -> Path` — 2×3 panel with one shared κ norm + colorbar.
  - `save_plots(inputs, out_dir, *, constants=None, enabled=True) -> list[Path]` — orchestrator writing the per-plant set into a `plots/` subdir; `enabled=False` is a no-op; re-derives Results by calling the **same tier helpers** the analysis uses; returns the written paths (file-path-reference convention, mirroring sleap-roots-analyze `files_generated`).
- **Stub→impl foundation migration (atomic):** remove `("plotting", "scaleogram", 16)` from `STUB_MODULES`, add explicit plotting callability tests, and add `plotting` to the explicit `test_module_logger_is_namespaced` list — in the same commit as the first non-raising callable.
- **Spec:** MODIFY "Package layout" (impl 11→12, stub 2→1; drop the plotting stub-callable row; add plotting callability scenarios + a PR #16 scope note) and ADD "Circumnutation diagnostic plots API".
- **Deviation:** the stub-promised `L_gz` arc-length marker is **omitted** (blocked on #230, not computed in the pipeline); graduated docstrings drop the claim. No `L_gz` parameter ships.
- **No new dependency** (matplotlib already present; seaborn unused). **No new constants** in `ConstantsT`; display knobs are module-level; `_CONSTANTS_VERSION` stays 6.

## Impact

- Affected specs: `circumnutation` (MODIFIED: Package layout; ADDED: Circumnutation diagnostic plots API).
- Affected code: `sleap_roots/circumnutation/plotting.py` (stub→impl); `tests/test_circumnutation_foundation.py` (`STUB_MODULES`, logger-namespace list, callability tests); new `tests/test_circumnutation_plotting.py`.
- Consumers: PR #17 (CLI) wires `--no-plots → enabled=False` and the `Series → CircumnutationInputs` adapter; no behavior change to `compute_traits`/`save`/`aggregate_by_genotype`.
- Cross-cutting: CC-3 (pure-pixel axes; unit suffixes per #222), CC-9 (logging). #237 (filename uniqueness via `track_id`), #238 (sidestepped by the `plots/` subdir).
