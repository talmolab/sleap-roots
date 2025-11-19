# Points

## Overview

Utility functions for manipulating, filtering, and transforming root point arrays.

**Key functions**:
- `join_pts` - Combine multiple point arrays
- `get_all_pts_array` - Flatten to 2D array
- `associate_lateral_to_primary` - Map laterals to primary
- `filter_roots_with_nans` - Remove invalid roots

## Quick Example

```python
import sleap_roots as sr

series = sr.Series.load("plant", primary_path="primary.slp", lateral_path="lateral.slp")
primary_pts = series.get_primary_points()
lateral_pts_list = series.get_lateral_points()

# Combine all points
all_pts = sr.join_pts([primary_pts] + lateral_pts_list)
print(f"Combined points shape: {all_pts.shape}")

# Filter out roots with NaN
valid_laterals = sr.filter_roots_with_nans(lateral_pts_list)
print(f"Valid laterals: {len(valid_laterals)}/{len(lateral_pts_list)}")
```

## API Reference

### join_pts

::: sleap_roots.join_pts
    options:
      show_source: true

---

### get_all_pts_array

::: sleap_roots.get_all_pts_array
    options:
      show_source: true

---

### associate_lateral_to_primary

::: sleap_roots.associate_lateral_to_primary
    options:
      show_source: true

---

### filter_roots_with_nans

::: sleap_roots.filter_roots_with_nans
    options:
      show_source: true

---

## Related Modules

- **[Series](../core/series.md)** - Loading point data
- All trait modules use point utilities
