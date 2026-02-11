# Add Client-Side Rendering Mode

## Why

The current viewer embeds all rendered frames as base64 PNGs inside a single HTML file. For datasets with many scans (36+ scans x 10 frames), this produces files exceeding 800MB that crash browsers. The matplotlib rendering step takes minutes, and the browser must hold the entire DOM in memory.

Scientists need to:
1. Review large datasets without hitting memory limits
2. Generate viewers quickly (seconds, not minutes)
3. Share results with collaborators
4. Toggle overlays on/off for comparison

## What Changes

- **Default mode (client-render)**: Embed prediction data as JSON, reference source images in-place, JavaScript draws overlays on Canvas. Generation in seconds, no size limits.
- **Pre-rendered mode (`--render`)**: Matplotlib renders overlays to disk as JPEG. For zip-and-share or archiving.
- **Embedded mode (`--embed`)**: Current base64 behavior for small datasets or quick single-file sharing.
- Add `--zip` flag to package viewer for sharing (works with all modes).
- Add overlay toggle (show/hide predictions) - free with client-side rendering.

## Impact

- Affected specs: `html-prediction-viewer` (MODIFIED requirements for image handling)
- Affected code: `sleap_roots/viewer/generator.py`, `sleap_roots/viewer/templates/viewer.html`, `sleap_roots/viewer/cli.py`
- New code: `sleap_roots/viewer/serializer.py` (prediction data serialization)

## Design

### Three Output Modes

**1. Client-render (default)**
Source images stay on filesystem. Prediction data (skeleton points, edges, scores, root types) serialized as JSON. JavaScript Canvas draws overlays when viewing each frame.

```
viewer.html                     # ~100-200KB (includes JS + prediction JSON)
```

- Generation: seconds (no matplotlib rendering)
- Size: tiny HTML regardless of dataset size
- Overlays: drawn on-demand, can toggle on/off
- Constraint: source images must remain accessible; for h5 sources, frames extracted to `viewer_images/`

**2. Pre-rendered (`--render`)**
Matplotlib renders prediction overlays onto images, saves as JPEG files. HTML references them.

```
viewer.html                     # ~50KB
viewer_images/                  # Rendered frames with overlays
  scan_ABC123/
    frame_0_root_type.jpeg
    frame_0_confidence.jpeg
    ...
```

- Generation: minutes (matplotlib rendering)
- Size: depends on frame count
- Overlays: baked into images, static
- Use case: zip-and-share, archiving, printing

**3. Embedded (`--embed`)**
Current behavior. Base64 images in single HTML file.

```
viewer.html                     # Large single file
```

- Generation: minutes
- Size: large (base64 overhead)
- Portability: single file, no dependencies
- Use case: small datasets, quick sharing

### Prediction Data Serialization

New module `serializer.py` extracts prediction data from sleap-io objects:

```python
def serialize_frame_predictions(series: Series, frame_idx: int) -> dict:
    """Extract skeleton points, edges, and scores for JS rendering.

    Returns:
        {
            "instances": [
                {
                    "root_type": "primary",  # or "lateral", "crown"
                    "points": [[x1, y1], [x2, y2], ...],  # node coordinates
                    "edges": [[0, 1], [1, 2], ...],  # node index pairs
                    "score": 0.95,  # instance confidence
                },
                ...
            ],
            "image_path": "relative/path/to/source/image.jpg",
        }
    ```
```

### JavaScript Canvas Rendering

Template includes ~100 lines of JS to draw predictions:

```javascript
function drawPredictions(ctx, instances, mode) {
    // mode: 'root_type' or 'confidence'
    for (const inst of instances) {
        const color = mode === 'root_type'
            ? ROOT_TYPE_COLORS[inst.root_type]
            : viridisColor(inst.score);

        // Draw edges (lines between nodes)
        for (const [i, j] of inst.edges) {
            ctx.beginPath();
            ctx.moveTo(inst.points[i][0], inst.points[i][1]);
            ctx.lineTo(inst.points[j][0], inst.points[j][1]);
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.stroke();
        }

        // Draw nodes (circles at each point)
        for (const [x, y] of inst.points) {
            ctx.beginPath();
            ctx.arc(x, y, 3, 0, 2 * Math.PI);
            ctx.fillStyle = color;
            ctx.fill();
        }
    }
}
```

### Overlay Toggle

Client-render mode enables real-time overlay control:
- Checkbox to show/hide predictions
- Toggle between root-type and confidence coloring without reloading
- Useful for comparing raw image vs predictions

### ZIP Archive (`--zip`)

Works with all modes:
- **Client-render + zip**: Copies source images into archive, rewrites relative paths
- **Pre-rendered + zip**: Packages HTML + rendered images
- **Embedded + zip**: Zips the single HTML file

### CLI Changes

```
sleap-roots viewer <predictions_dir> -o output.html [options]

Options:
  --render         Pre-render overlays with matplotlib (slower, for sharing)
  --embed          Embed images as base64 in HTML (single file)
  --format TEXT    Image format for --render: jpeg (default) or png
  --quality INT    JPEG quality 1-100 (default: 85)
  --zip            Create a zip archive for sharing
```

### h5 Video Source Handling

For h5 files, browsers cannot read them directly. In all modes:
- Frames are extracted as JPEG to `viewer_images/{scan_name}/`
- HTML references extracted images
- This extraction is fast (no matplotlib overlay rendering)

### Generator Changes

```python
def generate(
    self,
    output_path: Path,
    max_frames: int = DEFAULT_MAX_FRAMES,
    no_limit: bool = False,
    progress_callback: Optional[Callable] = None,
    render: bool = False,         # NEW: pre-render with matplotlib
    embed: bool = False,          # NEW: embed as base64
    image_format: str = "jpeg",   # NEW: jpeg or png
    image_quality: int = 85,      # NEW: JPEG quality
    create_zip: bool = False,     # NEW: create zip archive
) -> None:
```

Mode resolution:
- `--render` and `--embed` are mutually exclusive
- Neither flag: client-render mode (default)
- `--render`: pre-render overlays to disk
- `--embed`: base64 in single HTML (current behavior)