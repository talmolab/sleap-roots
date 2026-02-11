# Tasks: Fix Pipeline Video Loading

## Implementation

- [x] Add `_find_and_remap_video()` method to `ViewerGenerator`
- [x] Modify `generate()` to call `_find_and_remap_video()` before checking `series.video is None`
- [x] Add test `test_generate_pipeline_output_includes_images`

## Testing

- [x] Run existing viewer tests (38 passed)
- [x] Run pipeline output tests (4 passed)
- [x] Verify HTML includes base64 images from pipeline output

## Files Changed

- `sleap_roots/viewer/generator.py` - Added `_find_and_remap_video()` method (lines 184-282)
- `tests/test_viewer.py` - Added `TestPipelineOutputDiscovery` test class