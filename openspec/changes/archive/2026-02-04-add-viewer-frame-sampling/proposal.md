# Add Viewer Frame Sampling

## Why

The HTML prediction viewer embeds ALL frames for ALL scans, which creates:
- Very large HTML files (potentially gigabytes) for datasets with many scans/frames
- Long generation times (10+ minutes for large datasets)
- Browser crashes when loading large files (>500MB)

The viewer is a QC tool - users need to spot-check predictions, not view every frame.

## What Changes

- Add `--max-frames` CLI option (default: 10) to sample frames per scan
- Sample frames evenly distributed across scan (always include first and last)
- Add warning when total frames exceed 100
- Add hard limit of 1000 total frames (overridable with `--no-limit`)
- Update spec to document frame sampling behavior

## Impact

- Affected specs: `html-prediction-viewer` (MODIFIED requirement for frame sampling)
- Affected code: `sleap_roots/viewer/generator.py`, `sleap_roots/viewer/cli.py`, `tests/test_viewer.py`

## Design

### Frame Selection Algorithm

For `max_frames=N` from a scan with `total_frames=T`:
1. If `T <= N`: use all frames
2. If `T > N`: select N evenly-spaced frames including first (0) and last (T-1)

```python
def select_frame_indices(total_frames: int, max_frames: int) -> List[int]:
    if max_frames <= 0 or total_frames <= max_frames:
        return list(range(total_frames))

    # Always include first and last, distribute rest evenly
    indices = [0]
    if max_frames > 1:
        step = (total_frames - 1) / (max_frames - 1)
        for i in range(1, max_frames - 1):
            indices.append(round(i * step))
        indices.append(total_frames - 1)
    return indices
```

### CLI Options

```
--max-frames N    Sample N frames per scan (default: 10, 0 for all)
--no-limit        Disable 1000 total frame limit (for advanced users)
```

### Limits

| Threshold | Action |
|-----------|--------|
| >100 total frames | Warning message |
| >1000 total frames | Error unless `--no-limit` |