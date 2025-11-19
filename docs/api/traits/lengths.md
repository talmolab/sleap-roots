# Lengths

## Overview

The `lengths` module provides functions for computing root length measurements from point coordinates. These are fundamental traits used across all pipelines.

**Key functions**:
- `get_root_lengths` - Calculate total root lengths
- `get_curve_index` - Measure root curvature
- `get_max_length_pts` - Find longest root path

**When to use**:
- Computing primary or lateral root lengths
- Measuring root curvature and tortuosity
- Comparing root lengths across treatments

## Quick Example

```python
import sleap_roots as sr
import numpy as np

# Get root points
series = sr.Series.load("plant", primary_path="primary.slp")
pts = series.get_primary_points()  # (plants, frames, nodes, 2)

# Compute lengths
lengths = sr.get_root_lengths(pts)
print(f"Primary root length: {lengths[0, 0]:.2f} pixels")

# Compute curvature
curve_idx = sr.get_curve_index(pts)
print(f"Curvature index: {curve_idx[0, 0]:.3f}")
```

## API Reference

### get_root_lengths

::: sleap_roots.get_root_lengths
    options:
      show_source: true

**Example**:
```python
import sleap_roots as sr
import numpy as np

# Load and extract points
series = sr.Series.load("plant", primary_path="primary.slp")
primary_pts = series.get_primary_points()

# Calculate lengths
lengths = sr.get_root_lengths(primary_pts)

# Results shape: (plants, frames)
print(f"Shape: {lengths.shape}")
print(f"Frame 0 length: {lengths[0, 0]:.2f} px")

# Handle NaN (missing data)
valid_lengths = lengths[~np.isnan(lengths)]
print(f"Mean length: {np.mean(valid_lengths):.2f} px")
```

**See Also**:
- [get_curve_index](#get_curve_index) - Root curvature
- [DicotPipeline](../core/pipelines.md#dicotpipeline) - Uses this function

---

### get_curve_index

::: sleap_roots.get_curve_index
    options:
      show_source: true

**Example**:
```python
import sleap_roots as sr

series = sr.Series.load("plant", primary_path="primary.slp")
pts = series.get_primary_points()

# Compute curvature index
curve_idx = sr.get_curve_index(pts)

# Lower values = straighter root
# Higher values = more curved/tortuous root
print(f"Curvature index: {curve_idx[0, 0]:.3f}")
```

---

### get_max_length_pts

::: sleap_roots.get_max_length_pts
    options:
      show_source: true

**Example**:
```python
import sleap_roots as sr

series = sr.Series.load("plant", lateral_path="lateral.slp")
lateral_pts_list = series.get_lateral_points()

# Find longest lateral root
max_pts = sr.get_max_length_pts(lateral_pts_list)
max_length = sr.get_root_lengths(max_pts[np.newaxis, ...])[0, 0]
print(f"Longest lateral: {max_length:.2f} px")
```

---

## Related Modules

- **[Angles](angles.md)** - Root angle measurements
- **[Network Length](networklength.md)** - Whole-plant length metrics
- **[Pipelines](../core/pipelines.md)** - Use length functions

---

## See Also

- [Pipeline Tutorials](../../tutorials/index.md)
- [Batch Processing Guide](../../guides/batch-processing.md)
