# Ellipse

## Overview

Fit ellipses to root point distributions for compact spatial representation.

**Key functions**:
- `fit_ellipse` - Fit ellipse to root points
- `get_ellipse_a` - Get ellipse major axis
- `get_ellipse_b` - Get ellipse minor axis
- `get_ellipse_ratio` - Get ellipse aspect ratio

## Quick Example

```python
import sleap_roots as sr
import numpy as np

series = sr.Series.load("plant", primary_path="primary.slp")
pts = series.get_primary_points()

# Flatten to 2D points
pts_2d = pts[0, 0]  # (nodes, 2)

# Fit ellipse
ellipse = sr.ellipse.fit_ellipse(pts_2d)
a = sr.ellipse.get_ellipse_a(ellipse)
b = sr.ellipse.get_ellipse_b(ellipse)
ratio = sr.ellipse.get_ellipse_ratio(ellipse)

print(f"Major axis (a): {a:.2f} px")
print(f"Minor axis (b): {b:.2f} px")
print(f"Aspect ratio: {ratio:.2f}")
```

## API Reference

### fit_ellipse

::: sleap_roots.ellipse.fit_ellipse
    options:
      show_source: true

---

### get_ellipse_a

::: sleap_roots.ellipse.get_ellipse_a
    options:
      show_source: true

---

### get_ellipse_b

::: sleap_roots.ellipse.get_ellipse_b
    options:
      show_source: true

---

### get_ellipse_ratio

::: sleap_roots.ellipse.get_ellipse_ratio
    options:
      show_source: true

---

## Related Modules

- **[Convex Hull](convhull.md)** - Alternative spatial analysis
