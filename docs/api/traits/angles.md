# Angles

## Overview

Compute root angle measurements relative to gravity (vertical axis). Essential for gravitropism analysis and root growth direction studies.

**Key functions**:
- `get_root_angle` - Calculate root angle from proximal points
- `get_vector_angles_from_gravity` - Core angle calculation from vectors
- `get_node_ind` - Select points for angle computation

**Angle Convention**: 0° = straight down (with gravity), 90° = horizontal, 180° = straight up

## Quick Example

```python
import sleap_roots as sr

series = sr.Series.load("plant", primary_path="primary.slp")
pts = series.get_primary_points()

# Compute root angle
angles = sr.get_root_angle(pts, n_points=5)
print(f"Primary root angle: {angles[0, 0]:.1f}°")
```

## API Reference

### get_root_angle

::: sleap_roots.angle.get_root_angle
    options:
      show_source: true

**Example**:
```python
import sleap_roots as sr
import numpy as np

series = sr.Series.load("plant", primary_path="primary.slp")
pts = series.get_primary_points()

# Default: use first 5 points
angles = sr.get_root_angle(pts)
print(f"Angle (5 pts): {angles[0, 0]:.1f}°")

# Use more points for smoother estimate
angles_smooth = sr.get_root_angle(pts, n_points=10)
print(f"Angle (10 pts): {angles_smooth[0, 0]:.1f}°")

# Interpret results
angle = angles[0, 0]
if angle < 30:
    print("Root growing downward (positive gravitropism)")
elif angle > 150:
    print("Root growing upward (negative gravitropism)")
else:
    print("Root growing at an angle")
```

**See Also**: [DicotPipeline](../core/pipelines.md#dicotpipeline)

---

### get_vector_angles_from_gravity

::: sleap_roots.angle.get_vector_angles_from_gravity
    options:
      show_source: true

---

### get_node_ind

::: sleap_roots.angle.get_node_ind
    options:
      show_source: true

**Example**:
```python
import sleap_roots as sr

pts = sr.Series.load("plant", primary_path="primary.slp").get_primary_points()

# Get indices for proximal 5 points
indices = sr.get_node_ind(pts, n_points=5)
print(f"Selected nodes: {indices[0, 0]}")  # Array of node indices
```

---

## Related Modules

- **[Lengths](lengths.md)** - Root length measurements
- **[Tips](tips.md)** - Tip point detection

## See Also

- [Gravitropism Analysis Guide](../../guides/custom-pipelines.md)
