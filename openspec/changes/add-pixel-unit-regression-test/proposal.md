# Proposal: Add Pixel Unit Regression Test

**Change ID:** `add-pixel-unit-regression-test`
**Status:** PROPOSED
**Issue:** https://github.com/talmolab/sleap-roots/issues/146

## Why

If a dependency (sleap-io, PIL, h5py) ever auto-converts pixel coordinates based on image DPI metadata, every trait calculation would silently produce wrong values — a ~47x error at 1200 DPI that could go undetected. This defensive regression test guards against that at multiple layers: data loading, pipeline orchestration, and trait computation.

## What Changes

Add a new test module `tests/test_pixel_units.py` that:

1. Creates a synthetic TIFF image (200x400, 1200 DPI metadata) using Pillow
2. Creates synthetic sleap-io objects (Skeleton, Instance, LabeledFrame, Labels) with two nodes 100px apart
3. Constructs a `Series` object with these synthetic predictions
4. Runs through `PrimaryRootPipeline` end-to-end
5. Asserts `primary_length` = 100.0 (pixels), not ~2.12 (mm)
6. Also tests `get_root_lengths()` and `get_base_tip_dist()` directly as contract tests

## Impact

- Affected specs: pixel-unit-invariance (new capability)
- Affected code: `tests/test_pixel_units.py` (new), exercises `sleap_roots/lengths.py`, `sleap_roots/bases.py`, `sleap_roots/trait_pipelines.py` (PrimaryRootPipeline), `sleap_roots/series.py`
- Risk: None — test-only change with no production code modifications
- Backward compatibility: No impact
- Reproducibility: Strengthens reproducibility by preventing silent unit drift
