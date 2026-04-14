# Proposal: Add Pixel Unit Regression Test

**Change ID:** `add-pixel-unit-regression-test`
**Status:** PROPOSED
**Issue:** https://github.com/talmolab/sleap-roots/issues/146
**Related:** https://github.com/talmolab/sleap-roots/issues/150 (get_node_ind 2-node crash)

## Why

If a dependency (sleap-io, PIL, h5py) ever auto-converts pixel coordinates based on image DPI metadata, every trait calculation would silently produce wrong values — a ~47x error at 1200 DPI that could go undetected. This defensive regression test guards against that at multiple layers: **file I/O (TIFF backend open, .slp serialization)**, data loading (`sio.load_slp`, `Series.load`), pipeline orchestration, and trait computation.

## What Changes

Add a new test module `tests/test_pixel_units.py` that:

1. Creates a synthetic TIFF image (200x400, 1200 DPI metadata) using Pillow
2. Opens the TIFF via `sio.Video.from_filename()` (forcing the backend to read file metadata)
3. Creates synthetic sleap-io objects (Skeleton, Instance, LabeledFrame, Labels) with a 6-node primary root spanning 100px vertically
4. Serializes the Labels to a `.slp` file via `sio.save_slp()`
5. Reloads the `.slp` via `Series.load(primary_path=...)` — exercising the full I/O layer
6. Runs through `PrimaryRootPipeline` end-to-end
7. Asserts `primary_length` = 100.0 (pixels), not ~2.12 (mm)
8. Also tests `get_root_lengths()` and `get_base_tip_dist()` directly as contract tests

### Why 6 nodes instead of 2?

Issue #146 describes the minimal case as "two nodes 100px apart", but `PrimaryRootPipeline` computes angle traits (`primary_angle_proximal`, `primary_angle_distal`) that call `get_node_ind()`, which crashes on roots with fewer than 3 nodes (see [#150](https://github.com/talmolab/sleap-roots/issues/150)). To exercise the full pipeline end-to-end, this test uses a 6-node primary root with nodes evenly spaced 20px apart (total length 100px). The unit-level contract tests for `get_root_lengths()` and `get_base_tip_dist()` still use the minimal 2-node case.

## Impact

- Affected specs: pixel-unit-invariance (new capability)
- Affected code: `tests/test_pixel_units.py` (new), exercises `sleap_roots/lengths.py`, `sleap_roots/bases.py`, `sleap_roots/trait_pipelines.py` (PrimaryRootPipeline), `sleap_roots/series.py`, and the `sio.save_slp` / `sio.load_slp` round-trip
- Risk: None — test-only change with no production code modifications
- Backward compatibility: No impact
- Reproducibility: Strengthens reproducibility by preventing silent unit drift at every layer from file I/O through trait output