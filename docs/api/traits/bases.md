# Bases

## Overview

Detect lateral root base points and compute base-related traits. Critical for understanding lateral root emergence patterns and density.

**Key functions**:
- `get_bases` - Detect base points where laterals emerge from primary
- `get_base_length` - Length of lateral root zone
- `get_base_ct_density` - Lateral root density
- `get_root_widths` - Root width measurements

## Quick Example

```python
import sleap_roots as sr

series = sr.Series.load("plant", primary_path="primary.slp", lateral_path="lateral.slp")
primary_pts = series.get_primary_points()
lateral_pts_list = series.get_lateral_points()

# Detect base points
bases = sr.get_bases(primary_pts, lateral_pts_list)
print(f"Base points shape: {bases.shape}")

# Compute base zone length
base_length = sr.get_base_length(bases)
print(f"Lateral root zone length: {base_length:.2f} px")
```

## API Reference

### get_bases

::: sleap_roots.get_bases
    options:
      show_source: true

---

### get_base_length

::: sleap_roots.get_base_length
    options:
      show_source: true

---

### get_base_ct_density

::: sleap_roots.get_base_ct_density
    options:
      show_source: true

**Example**:
```python
import sleap_roots as sr

series = sr.Series.load("dicot", primary_path="primary.slp", lateral_path="lateral.slp")
primary_pts = series.get_primary_points()
lateral_pts_list = series.get_lateral_points()

bases = sr.get_bases(primary_pts, lateral_pts_list)
density = sr.get_base_ct_density(bases, primary_pts)

print(f"Lateral root density: {density:.3f} roots/pixel")
```

---

### get_root_widths

::: sleap_roots.get_root_widths
    options:
      show_source: true

---

## Related Modules

- **[Tips](tips.md)** - Tip point detection
- **[Lengths](lengths.md)** - Base-to-tip distances
- **[DicotPipeline](../core/pipelines.md#dicotpipeline)** - Uses base functions
