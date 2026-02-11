## 1. Tests (TDD) - Prediction Serialization

- [x] 1.1 Test `serialize_instance()` extracts points as [[x, y], ...] list
- [x] 1.2 Test `serialize_instance()` extracts edges as [[i, j], ...] pairs from skeleton
- [x] 1.3 Test `serialize_instance()` includes instance score
- [x] 1.4 Test `serialize_instance()` includes root_type (primary/lateral/crown)
- [x] 1.5 Test `serialize_frame_predictions()` returns all instances for a frame
- [x] 1.6 Test `serialize_frame_predictions()` includes image_path relative to HTML
- [x] 1.7 Test `serialize_scan_predictions()` returns predictions for all sampled frames
- [x] 1.8 Test serialization handles missing/None scores gracefully

## 2. Tests (TDD) - Client-Render Mode (Default)

- [x] 2.1 Test `generate()` default mode embeds predictions as JSON in HTML
- [x] 2.2 Test `generate()` default mode does NOT call matplotlib render functions
- [x] 2.3 Test `generate()` default mode HTML contains relative image paths
- [x] 2.4 Test `generate()` default mode completes fast (36s vs 315s with rendering)
- [x] 2.5 Test HTML contains Canvas drawing JavaScript code
- [x] 2.6 Test HTML includes viridis colormap implementation in JS
- [x] 2.7 Test HTML includes root type color constants in JS
- [x] 2.8 Test generated HTML includes overlay toggle checkbox
- [x] 2.9 Test JS drawPredictions normalizes raw scores before viridis colormap

## 3. Tests (TDD) - H5 Frame Extraction

- [x] 3.1 Test `frame_to_base64()` returns valid data URI
- [x] 3.2 Test `frame_to_base64()` respects quality parameter
- [x] 3.3 Test `frame_to_base64()` respects format (JPEG/PNG)
- [x] 3.4 Test `is_h5_video()` returns True for H5-backed video
- [x] 3.5 Test client-render with H5 source embeds base64 images
- [x] 3.6 Test client-render with H5 still includes predictions for canvas

## 4. Tests (TDD) - Pre-Rendered Mode

- [x] 4.1 Test `generate(render=True)` calls matplotlib render functions
- [x] 4.2 Test `generate(render=True)` creates viewer_images directory
- [x] 4.3 Test `generate(render=True)` saves root_type and confidence images
- [x] 4.4 Test `generate(render=True)` uses JPEG format by default
- [x] 4.5 Test `generate(render=True, image_format='png')` saves PNG images
- [x] 4.6 Test `generate(render=True)` HTML references rendered images
- [x] 4.7 Test `figure_to_file()` saves figure to disk with correct format

## 5. Tests (TDD) - Embedded Mode (Backwards Compatibility)

- [x] 5.1 Test `generate(embed=True)` produces base64-embedded HTML
- [x] 5.2 Test `generate(embed=True)` matches current output format
- [x] 5.3 Test `generate(embed=True)` does not create images directory
- [x] 5.4 Test `generate(render=True, embed=True)` raises error (mutually exclusive)

## 6. Tests (TDD) - ZIP Archive

- [x] 6.1 Test `generate(create_zip=True)` creates zip file
- [x] 6.2 Test client-render zip copies source images into archive
- [x] 6.3 Test client-render zip rewrites paths in HTML
- [x] 6.4 Test pre-rendered zip contains HTML + images directory
- [x] 6.5 Test embedded zip contains only HTML file
- [x] 6.6 Test zip file is named `{output_stem}.zip`

## 7. Implementation - Serialization Module

- [x] 7.1 Create `sleap_roots/viewer/serializer.py`
- [x] 7.2 Implement `serialize_instance(instance, skeleton, root_type) -> dict`
- [x] 7.3 Implement `serialize_frame_predictions(series, frame_idx, html_path) -> dict`
- [x] 7.4 Implement `serialize_scan_predictions(series, frame_indices, html_path) -> list`
- [x] 7.5 Implement `get_skeleton_edges(skeleton) -> list` helper

## 8. Implementation - Generator Updates

- [x] 8.1 Add `render`, `embed`, `image_format`, `image_quality`, `create_zip` params
- [x] 8.2 Implement mode validation (render and embed mutually exclusive)
- [x] 8.3 Implement client-render mode logic (serialize predictions, skip matplotlib)
- [x] 8.4 Implement pre-rendered mode logic (matplotlib to files)
- [x] 8.5 Implement h5 frame extraction for client-render mode
- [x] 8.6 Implement zip archive creation

## 9. Implementation - Serializer Updates

- [x] 9.1 Add `figure_to_file()` function for saving to disk (in generator as _save_figure_to_file)
- [x] 9.2 Add `frame_to_base64()` function for h5 frame extraction (in serializer)
- [x] 9.3 Add `is_h5_video()` helper function (in serializer)

## 10. Implementation - Template Updates

- [x] 10.1 Add Canvas element for overlay rendering
- [x] 10.2 Implement `drawPredictions(ctx, instances, mode)` JS function
- [x] 10.3 Implement `viridisColor(score)` JS function (colormap)
- [x] 10.4 Add `ROOT_TYPE_COLORS` constant object
- [x] 10.5 Add overlay toggle checkbox and handler
- [x] 10.6 Update image loading to work with all three modes
- [x] 10.7 Handle mode switching (show canvas overlay or pre-rendered image)

## 11. Implementation - CLI Updates

- [x] 11.1 Add `--render` flag
- [x] 11.2 Add `--embed` flag
- [x] 11.3 Add `--format` option (jpeg/png)
- [x] 11.4 Add `--quality` option (1-100)
- [x] 11.5 Add `--zip` flag
- [x] 11.6 Update help text to explain modes

## 12. Verification

- [x] 12.1 Run all tests and verify they pass (105 passed)
- [x] 12.2 Run black formatting check
- [x] 12.3 Run pydocstyle check
- [x] 12.4 Manual test client-render mode with pipeline output (image dirs)
- [x] 12.5 Manual test client-render mode with h5 source
- [x] 12.6 Manual test `--render` mode and verify overlay quality
- [x] 12.7 Manual test `--embed` mode (backwards compatibility)
- [x] 12.8 Manual test `--zip` with each mode
- [x] 12.9 Manual test overlay toggle (show/hide predictions)
- [x] 12.10 Verify generation time: client-render ~36s vs embed ~315s