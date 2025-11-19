# Tips

## Overview

Detect and extract root tip coordinates. Used for tracking root growth direction and computing tip-based traits.

**Key functions**:
- `get_tips` - Extract tip points from root arrays
- `get_tip_xs` - Get x-coordinates of tips
- `get_tip_ys` - Get y-coordinates of tips

## Quick Example

```python
import sleap_roots as sr

series = sr.Series.load("plant", lateral_path="lateral.slp")
lateral_pts_list = series.get_lateral_points()

# Get tip points
tips = sr.get_tips(lateral_pts_list)
print(f"Tip shape: {tips.shape}")  # (n_laterals, 2)
print(f"First tip: x={tips[0, 0]:.1f}, y={tips[0, 1]:.1f}")
```

## API Reference

### get_tips

::: sleap_roots.get_tips
    options:
      show_source: true

**Example**:
```python
import sleap_roots as sr
import numpy as np

series = sr.Series.load("plant", lateral_path="lateral.slp")
lateral_pts_list = series.get_lateral_points()

# Extract all lateral tips
tips = sr.get_tips(lateral_pts_list)

# Handle NaN (missing tips)
valid_tips = tips[~np.isnan(tips).any(axis=1)]
print(f"Valid tips: {len(valid_tips)}/{len(tips)}")
```

---

### get_tip_xs

::: sleap_roots.get_tip_xs
    options:
      show_source: true

---

### get_tip_ys

::: sleap_roots.get_tip_ys
    options:
      show_source: true

**Example**:
```python
import sleap_roots as sr

lateral_pts_list = sr.Series.load("plant", lateral_path="lateral.slp").get_lateral_points()

# Get tip coordinates separately
tip_xs = sr.get_tip_xs(lateral_pts_list)
tip_ys = sr.get_tip_ys(lateral_pts_list)

print(f"Tip x-coordinates: {tip_xs}")
print(f"Tip y-coordinates: {tip_ys}")
```

---

## Related Modules

- **[Bases](bases.md)** - Base point detection (opposite end of root)
- **[Lengths](lengths.md)** - Tip-to-base distances

## See Also

- [DicotPipeline](../core/pipelines.md#dicotpipeline)
