# Tasks: Add Viewer Progress Indicator

## 1. Tests (TDD)

- [x] 1.1 Add test for progress_callback parameter
- [x] 1.2 Add test that callback is called with correct arguments
- [x] 1.3 Add test that callback is called for each frame

## 2. Implementation

- [x] 2.1 Add progress_callback parameter to generate()
- [x] 2.2 Call callback during frame rendering loop
- [x] 2.3 Update CLI to display progress with scan name and frame count

## 3. Validation

- [x] 3.1 Run all viewer tests (52 passed)

## Files Changed

- `sleap_roots/viewer/generator.py` - Added `progress_callback` parameter to `generate()`
- `sleap_roots/viewer/cli.py` - Added progress display during generation
- `tests/test_viewer.py` - Added `TestProgressCallback` class with 3 tests