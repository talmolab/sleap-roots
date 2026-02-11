# Tasks: Add Viewer Frame Sampling

## 1. Tests (TDD)

- [x] 1.1 Add test for `select_frame_indices()` function
- [x] 1.2 Add test for frame sampling in `generate()` method
- [x] 1.3 Add test for warning when >100 frames
- [x] 1.4 Add test for error when >1000 frames without `--no-limit`
- [x] 1.5 Add test for `--no-limit` override

## 2. Implementation

- [x] 2.1 Add `select_frame_indices()` function to generator.py
- [x] 2.2 Add `max_frames` parameter to `generate()` method
- [x] 2.3 Add frame limit validation with warning/error
- [x] 2.4 Update CLI to add `--max-frames` and `--no-limit` options

## 3. Validation

- [x] 3.1 Run all viewer tests (49 passed)
- [x] 3.2 Manual test with CLI help

## Files Changed

- `sleap_roots/viewer/generator.py` - Added `select_frame_indices()`, `FrameLimitExceededError`, modified `generate()`
- `sleap_roots/viewer/cli.py` - Added `--max-frames` and `--no-limit` options
- `tests/test_viewer.py` - Added `TestFrameSampling` and `TestFrameLimits` classes