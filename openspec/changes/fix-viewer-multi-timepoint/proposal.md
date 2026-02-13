## Why

The HTML viewer fails to correctly match image directories in multi-timepoint experiments where the same plant (e.g., `Fado_1`) is scanned across multiple days (Day0, Day3, Day5, etc.). The current video remapping logic only uses the leaf directory name, causing predictions from different timepoints to overlay on wrong images.

## What Changes

- **Fix video path remapping** to preserve enough path components to uniquely identify image directories across timepoints
- **Add timepoint organization** for gallery view (group scans by timepoint/day)
- **Add optional timepoint filtering** via `--timepoint` argument

## Impact

- Affected specs: `html-prediction-viewer`
- Affected code: `sleap_roots/viewer/generator.py` (video remapping logic)

## Background

When processing `.slp` files from pipeline output, embedded image paths have the structure:
```
/workspace/images_input/images/Wave1/Day0_2025-11-27/Fado_1/1.jpg
```

The current code extracts only `Fado_1` (leaf directory) and searches with `**/{Fado_1}`, finding the first match regardless of which day folder it belongs to. For multi-timepoint datasets, this causes:
1. Wrong images displayed for timepoints
2. Predictions from one day overlaid on another day's images
3. Confusing gallery where only some timepoints appear to work

## Design Decision Points

1. **Path matching strategy**: Use multiple path components (e.g., `Wave1/Day0/Fado_1`) for unique matching
2. **Gallery organization**: Group scans by timepoint in the gallery view for easier browsing
3. **Timepoint filtering**: Allow users to generate viewer for specific timepoints only