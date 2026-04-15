# Tasks: Add Pixel Unit Regression Test

## 1. Create test module with pixel unit regression tests

- [x] 1.1 Create `tests/test_pixel_units.py` with integration test: synthetic TIFF (200x400, 1200 DPI) opened via `sio.Video.from_filename()` + 6-node sleap-io Labels + `sio.save_slp()` + `Series.load()` + `PrimaryRootPipeline` → assert `primary_length` == 100.0 pixels
- [x] 1.2 Add unit-level contract tests: `test_root_length_straight`, `test_root_length_polyline`, `test_base_tip_dist_in_pixels`, `test_multi_instance_lengths`
- [x] 1.3 Run `uv run pytest tests/test_pixel_units.py -v` and verify all pass

## 2. Verify no regressions

- [x] 2.1 Run `uv run pytest tests/ -x` and verify full suite passes
- [x] 2.2 Run `uv run black --check sleap_roots/ tests/` and verify formatting

## Dependencies

- Task 2 depends on Task 1
- No parallelizable work (single test file)

## Notes

- The integration test uses a 6-node primary root (not 2 nodes as in the original issue description) because `PrimaryRootPipeline` computes angle traits via `get_node_ind()`, which crashes on roots with fewer than 3 nodes. This is tracked as a separate bug in issue #150.