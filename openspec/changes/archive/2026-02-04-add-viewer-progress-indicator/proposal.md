# Add Viewer Progress Indicator

## Why

When generating the HTML viewer, users see no feedback during the (potentially long) rendering process. The only output is a warning message, then silence until completion. Users cannot tell if the process is working or hung.

## What Changes

- Add `progress_callback` parameter to `ViewerGenerator.generate()`
- Update CLI to display a progress bar showing scan and frame progress
- Show estimated time remaining based on rendering speed

## Impact

- Affected specs: `html-prediction-viewer` (ADDED requirement for progress feedback)
- Affected code: `sleap_roots/viewer/generator.py`, `sleap_roots/viewer/cli.py`

## Design

### Progress Callback

```python
def generate(
    self,
    output_path: Path,
    max_frames: int = DEFAULT_MAX_FRAMES,
    no_limit: bool = False,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> None:
    """
    Args:
        progress_callback: Called with (scan_name, frames_done, total_frames)
    """
```

### CLI Progress Display

Use click's progress bar:
```
Rendering: [████████░░░░░░░░] 45/100 frames (scan: ABC123)
```