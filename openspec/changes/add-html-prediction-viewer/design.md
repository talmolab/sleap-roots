## Context

Scientists processing root experiments with new species need to validate that SLEAP models generalize correctly before using extracted traits. The current workflow requires opening each .slp file in SLEAP GUI, which is slow for batch review. This feature adds a lightweight HTML-based viewer that works offline and can be shared with collaborators.

**Stakeholders**: Plant phenotyping researchers at Salk Institute (Talmo Lab, Busch Lab)

**Constraints**:
- Must work offline (no runtime server)
- Must be shareable as a single file
- Must reuse existing `Series.plot()` rendering
- Must support all root types (primary, lateral, crown)

## Goals / Non-Goals

**Goals**:
- Generate static HTML report from pipeline output
- Enable quick scan overview and frame-level drill-down
- Display prediction quality indicators (confidence scores)
- Support keyboard navigation for efficient review
- Zero runtime dependencies for viewing

**Non-Goals**:
- Real-time SLEAP model inference
- Editing or correcting predictions
- Interactive annotation
- Server-based viewer (Option B from investigation)
- Browser-based Python execution (Option C from investigation)

## Decisions

### Decision 1: Static HTML Generator (not server-based)
**What**: Generate self-contained HTML file with embedded images (base64)
**Why**:
- Scientists can share single file with collaborators
- Works offline without Python environment
- Simple implementation using existing `Series.plot()`
- Can run automatically at end of pipeline

**Alternatives considered**:
- Python local server: More complex, requires Python running
- PyScript/Pyodide: Slow initial load, large bundle, complex setup

### Decision 2: Two-level navigation
**What**: Scan overview (thumbnail grid) + Frame drill-down (all frames per scan)
**Why**:
- Matches scientist workflow: identify problematic scans, then review frames
- Thumbnail grid provides quick batch overview
- Frame drill-down enables detailed inspection
- Frame count is dynamic per scan (discovered at generation time)

### Decision 3: Reuse Series.plot() for composite rendering
**What**: Use existing matplotlib-based rendering from Series class
**Why**:
- Already renders all available root types (primary, lateral, crown) as composite overlay
- Each root type displayed with distinct colors on single image
- Consistent visualization with existing tools
- Well-tested, handles edge cases
- No additional dependencies

### Decision 4: Jinja2 for HTML templating
**What**: Use Jinja2 (already a Click dependency) for HTML generation
**Why**:
- No new dependencies
- Cleaner separation of template and logic
- Standard Python templating approach

### Decision 5: Base64-encoded images
**What**: Embed images as base64 data URIs in HTML
**Why**:
- Single-file output, easily shareable
- Works offline
- Trade-off: Larger file size (~33% overhead)

**Alternative**: Separate image folder
- Pro: Smaller HTML file
- Con: Must share folder + HTML, more complex

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| Large HTML file size (many frames) | Lazy loading with JavaScript, thumbnail-first approach |
| Memory usage during generation | Process frames sequentially, don't hold all in memory |
| Slow generation for large experiments | Progress bar, parallel image rendering option |
| Base64 overhead increases file size | Acceptable for validation use case; future: optional external images |

## Architecture

```
sleap_roots/
  viewer/
    __init__.py          # Public API exports
    cli.py               # Click CLI: `sleap-roots viewer`
    generator.py         # ViewerGenerator class
    renderer.py          # Frame rendering using Series.plot()
    templates/
      viewer.html        # Jinja2 template with embedded JS
```

### ViewerGenerator Class
```python
class ViewerGenerator:
    def __init__(self, predictions_dir: Path, images_dir: Optional[Path] = None):
        """Initialize with paths to predictions and optional images."""

    def discover_scans(self) -> List[ScanInfo]:
        """Find all scans in predictions directory."""

    def render_frame(self, scan: ScanInfo, frame_idx: int) -> str:
        """Render frame with overlay, return base64 PNG."""

    def generate(self, output_path: Path) -> None:
        """Generate complete HTML viewer."""
```

### HTML Template Features
- CSS Grid for scan thumbnail layout
- JavaScript for keyboard navigation (Left/Right arrows, Enter to drill-down, Esc to overview)
- Confidence score badges on thumbnails
- Frame counter and progress indicator
- Responsive design for various screen sizes

## Migration Plan

No migration needed - this is a new capability with no breaking changes.

### Decision 6: Toggle between root type and confidence views
**What**: Two visualization modes switchable via UI toggle
- **Root type view**: Colors represent root types (primary, lateral, crown) with distinct colors
- **Confidence view**: Colors represent confidence scores using a continuous colormap (e.g., viridis) mapped to the score range

**Why**:
- Avoids color conflict between root type identification and confidence display
- Continuous colormap provides more granularity than discrete thresholds
- User can focus on root identification or quality assessment as needed
- Confidence scores come from SLEAP predictions (`inst.score`, `point.score`)

## Open Questions

1. **Thumbnail resolution**: What size for scan thumbnails? (Proposed: 200x200px)
2. **Colormap choice**: Which matplotlib colormap for confidence view? (Proposed: viridis - perceptually uniform, colorblind-friendly)