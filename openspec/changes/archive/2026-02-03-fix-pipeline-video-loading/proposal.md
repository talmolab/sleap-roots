# Fix Pipeline Video Loading for HTML Viewer

## Problem

The HTML prediction viewer fails to display scans from pipeline output because:

1. Pipeline output .slp files use `ImageVideo` backend with embedded absolute paths
2. These paths (e.g., `/workspace/images_input/...`) don't exist on the local machine
3. `Series.video` returns `None` when video paths are invalid
4. The `ViewerGenerator.generate()` method skips scans where `series.video is None`

## Investigation Findings

See `_scratch/2026-02-03-sleap-io-video-loading/` for full investigation.

Key findings:
- sleap-io's `Video.replace_filename()` can remap video paths
- The image directory name can be extracted from embedded paths
- After remapping, video frames are readable

## Solution

Add a `_find_and_remap_video()` method to `ViewerGenerator` that:

1. Checks if `series.video is None`
2. Gets the video from the labels (crown_labels, primary_labels, or lateral_labels)
3. Extracts the image directory name from embedded paths
4. Searches for the directory in `predictions_dir` or `images_dir`
5. Remaps video paths using `replace_filename()`
6. Sets `series.video` to the remapped video

## Changes

### sleap_roots/viewer/generator.py

Add method:
```python
def _find_and_remap_video(self, series: Series) -> bool:
    """Find local image directory and remap video paths.

    Returns True if video was successfully remapped.
    """
```

Modify `generate()` to call `_find_and_remap_video()` before checking `series.video is None`.

### tests/test_viewer.py

Add test:
- `test_generate_pipeline_output_includes_images` - Verify rendered images in HTML

## Impact

- Affected specs: `html-prediction-viewer` (ADDED requirement for pipeline output support)
- Affected code: `sleap_roots/viewer/generator.py`, `tests/test_viewer.py`

## Testing

- Run existing viewer tests
- Run new pipeline output test
- Manual test with user's Amaranth pipeline output