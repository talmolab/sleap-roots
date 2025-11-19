# Ellipse

## Overview

Fit ellipses to root point distributions for compact spatial representation.

**Key functions**:
- `get_ellipse` - Fit ellipse to root points
- `fit_ellipse` - Core ellipse fitting algorithm

## Quick Example

```python
import sleap_roots as sr
import numpy as np

series = sr.Series.load("plant", primary_path="primary.slp")
pts = series.get_primary_points()

# Flatten to 2D points
pts_2d = pts[0, 0]  # (nodes, 2)

# Fit ellipse
ellipse_params = sr.get_ellipse(pts_2d)
print(f"Ellipse parameters: {ellipse_params}")
```

## API Reference

### get_ellipse

::: sleap_roots.get_ellipse
    options:
      show_source: true

---

### fit_ellipse

::: sleap_roots.fit_ellipse
    options:
      show_source: true

---

## Related Modules

- **[Convex Hull](convhull.md)** - Alternative spatial analysis
