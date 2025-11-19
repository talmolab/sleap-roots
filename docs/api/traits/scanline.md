# Scanline

## Overview

Count root intersections with horizontal scan lines for distribution analysis.

**Key functions**:
- `count_scanline_intersections` - Count intersections at given y-coordinates
- `get_scanline_first_ind` - First intersection index
- `get_scanline_last_ind` - Last intersection index

## Quick Example

```python
import sleap_roots as sr
import numpy as np

series = sr.Series.load("plant", primary_path="primary.slp")
pts = series.get_primary_points()

# Count intersections at y=100
intersections = sr.count_scanline_intersections(pts, y=100)
print(f"Intersections at y=100: {intersections}")
```

## API Reference

### count_scanline_intersections

::: sleap_roots.scanline.count_scanline_intersections
    options:
      show_source: true

---

### get_scanline_first_ind

::: sleap_roots.scanline.get_scanline_first_ind
    options:
      show_source: true

---

### get_scanline_last_ind

::: sleap_roots.scanline.get_scanline_last_ind
    options:
      show_source: true

---

## Related Modules

- **[Network Length](networklength.md)** - Network metrics
