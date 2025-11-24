# Convex Hull

## Overview

Compute convex hull features for spatial root system analysis. Useful for measuring root system spread and spatial distribution.

**Key functions**:
- `get_convhull` - Compute convex hull polygon
- `get_chull_area` - Hull area
- `get_chull_perimeter` - Hull perimeter

## Quick Example

```python
import sleap_roots as sr

series = sr.Series.load("plant", primary_path="primary.slp", lateral_path="lateral.slp")
primary_pts = series.get_primary_points()
lateral_pts_list = series.get_lateral_points()

# Get all points
from sleap_roots import join_pts
all_pts = join_pts([primary_pts] + lateral_pts_list)

# Compute convex hull
hull = sr.convhull.get_convhull(all_pts)

# Compute hull traits
area = sr.convhull.get_chull_area(all_pts)
perimeter = sr.convhull.get_chull_perimeter(all_pts)

print(f"Hull area: {area:.2f} px²")
print(f"Hull perimeter: {perimeter:.2f} px")
```

## API Reference

### get_convhull

::: sleap_roots.convhull.get_convhull
    options:
      show_source: true

---

### get_chull_area

::: sleap_roots.convhull.get_chull_area
    options:
      show_source: true

---

### get_chull_perimeter

::: sleap_roots.convhull.get_chull_perimeter
    options:
      show_source: true

---

## Related Modules

- **[Network Length](networklength.md)** - Network-level spatial metrics
- **[Ellipse](ellipse.md)** - Alternative spatial representation
