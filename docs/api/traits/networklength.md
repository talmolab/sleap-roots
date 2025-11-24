# Network Length

## Overview

Compute whole-plant network-level metrics including bounding box, width-depth ratio, and distribution statistics.

**Key functions**:
- `get_network_length` - Total root system length
- `get_network_width_depth_ratio` - W:D ratio
- `get_network_distribution` - Spatial distribution metrics
- `get_bbox` - Bounding box coordinates

## Quick Example

```python
import sleap_roots as sr

series = sr.Series.load("plant", primary_path="primary.slp", lateral_path="lateral.slp")
primary_pts = series.get_primary_points()
lateral_pts_list = series.get_lateral_points()

# Network metrics
from sleap_roots import join_pts
all_pts = join_pts([primary_pts] + lateral_pts_list)

network_len = sr.get_network_length(all_pts)
wd_ratio = sr.get_network_width_depth_ratio(all_pts)

print(f"Total network length: {network_len:.2f} px")
print(f"Width:Depth ratio: {wd_ratio:.2f}")
```

## API Reference

### get_network_length

::: sleap_roots.networklength.get_network_length
    options:
      show_source: true

---

### get_network_width_depth_ratio

::: sleap_roots.networklength.get_network_width_depth_ratio
    options:
      show_source: true

---

### get_network_distribution

::: sleap_roots.networklength.get_network_distribution
    options:
      show_source: true

---

### get_bbox

::: sleap_roots.networklength.get_bbox
    options:
      show_source: true

---

## Related Modules

- **[Convex Hull](convhull.md)** - Spatial analysis
- **[Lengths](lengths.md)** - Individual root lengths
