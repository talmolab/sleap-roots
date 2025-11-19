# Series

## Overview

The `Series` class is the primary data structure in sleap-roots for working with SLEAP predictions. It provides a unified interface for loading prediction files, accessing root point data, and managing plant metadata.

**Key features**:
- Load SLEAP `.slp` prediction files and `.h5` image series
- Access root points for primary, lateral, and crown roots
- Quality control and filtering capabilities
- Integration with pipeline classes for trait computation
- Plotting and visualization

**When to use**:
- You have SLEAP predictions and want to extract root point data
- You need to load multiple plants for batch processing
- You want to filter plants based on root count or quality
- You're building custom trait computation workflows

## Quick Example

```python
import sleap_roots as sr

# Load a single plant
series = sr.Series.load(
    series_name="canola_plant1",
    h5_path="predictions.h5",
    primary_path="primary_roots.slp",
    lateral_path="lateral_roots.slp"
)

# Access root points
primary_pts = series.get_primary_points()  # (plants, frames, nodes, 2)
lateral_pts_list = series.get_lateral_points()  # List of arrays

print(f"Loaded {series.series_name}")
print(f"Primary shape: {primary_pts.shape}")
print(f"Lateral roots: {len(lateral_pts_list)}")
```

## API Reference

### Series Class

::: sleap_roots.Series
    options:
      show_source: true
      members:
        - __init__
      heading_level: 3

---

### Loading Data

#### load

::: sleap_roots.Series.load
    options:
      show_source: true
      heading_level: 4

**Example**:
```python
import sleap_roots as sr

# Load plant with primary and lateral roots
series = sr.Series.load(
    series_name="my_plant",
    h5_path="data/predictions.h5",
    primary_path="data/primary.slp",
    lateral_path="data/lateral.slp"
)

# Load plant with all root types
series = sr.Series.load(
    series_name="complete_plant",
    h5_path="data/predictions.h5",
    primary_path="data/primary.slp",
    lateral_path="data/lateral.slp",
    crown_path="data/crown.slp"
)

# Load without video (points only)
series = sr.Series.load(
    series_name="points_only",
    primary_path="data/primary.slp",
    lateral_path="data/lateral.slp"
)
```

**Common Use Cases**:

```python
# Dicot plant (primary + lateral)
series = sr.Series.load(
    "dicot1",
    h5_path="dicot.h5",
    primary_path="primary.slp",
    lateral_path="lateral.slp"
)

# Monocot plant (crown roots only)
series = sr.Series.load(
    "monocot1",
    h5_path="monocot.h5",
    crown_path="crown.slp"
)

# Multiple primary roots
series = sr.Series.load(
    "multiple_primary",
    h5_path="data.h5",
    primary_path="primary.slp"
)
```

**See Also**:
- [find_all_h5_paths](#utility-functions) - Find all H5 files in directory
- [find_all_slp_paths](#utility-functions) - Find all SLP files in directory
- [load_series_from_h5s](#utility-functions) - Batch load from H5 files
- [load_series_from_slps](#utility-functions) - Batch load from SLP files

---

### Accessing Root Points

#### get_primary_points

::: sleap_roots.Series.get_primary_points
    options:
      show_source: true
      heading_level: 4

**Example**:
```python
import sleap_roots as sr

series = sr.Series.load(
    "plant1",
    h5_path="data.h5",
    primary_path="primary.slp"
)

# Get all primary root points
pts = series.get_primary_points()
print(pts.shape)  # (plants, frames, nodes, 2)

# Access specific frame
frame_pts = pts[0, 0]  # First plant, first frame
print(frame_pts.shape)  # (nodes, 2) - x,y coordinates

# Get x and y coordinates
x_coords = pts[..., 0]  # All x coordinates
y_coords = pts[..., 1]  # All y coordinates
```

**See Also**:
- [get_lateral_points](#get_lateral_points)
- [get_crown_points](#get_crown_points)

---

#### get_lateral_points

::: sleap_roots.Series.get_lateral_points
    options:
      show_source: true
      heading_level: 4

**Example**:
```python
import sleap_roots as sr

series = sr.Series.load(
    "plant1",
    primary_path="primary.slp",
    lateral_path="lateral.slp"
)

# Get lateral root points (returns list)
lateral_pts_list = series.get_lateral_points()

# Each element is an array for one lateral root
for i, root_pts in enumerate(lateral_pts_list):
    print(f"Lateral root {i}: {root_pts.shape}")
    # Shape: (plants, frames, nodes, 2)

# Access specific lateral root
first_lateral = lateral_pts_list[0]
first_frame = first_lateral[0, 0]  # (nodes, 2)
```

**Working with lateral roots**:
```python
# Count lateral roots
n_lateral = len(lateral_pts_list)

# Filter lateral roots with valid points
import numpy as np
valid_laterals = [
    pts for pts in lateral_pts_list
    if not np.all(np.isnan(pts))
]

# Get all lateral points as single array
from sleap_roots.points import join_pts
all_lateral = join_pts(lateral_pts_list)
```

**See Also**:
- [get_primary_points](#get_primary_points)
- [sleap_roots.points.join_pts](../traits/points.md#join_pts) - Combine point arrays

---

#### get_crown_points

::: sleap_roots.Series.get_crown_points
    options:
      show_source: true
      heading_level: 4

**Example**:
```python
import sleap_roots as sr

series = sr.Series.load(
    "monocot1",
    h5_path="data.h5",
    crown_path="crown.slp"
)

# Get crown root points (returns list)
crown_pts_list = series.get_crown_points()

# Process each crown root
for i, root_pts in enumerate(crown_pts_list):
    print(f"Crown root {i}: {root_pts.shape}")
    # Shape: (plants, frames, nodes, 2)
```

**See Also**:
- [get_primary_points](#get_primary_points)
- [get_lateral_points](#get_lateral_points)

---

### Quality Control

#### expected_count

::: sleap_roots.Series.expected_count
    options:
      show_source: true
      heading_level: 4

**Example**:
```python
import sleap_roots as sr

series = sr.Series.load(
    "plant1",
    primary_path="primary.slp",
    lateral_path="lateral.slp"
)

# Set expected root counts
series.expected_count(primary=1, lateral=5)

# Check if counts match
if not series.qc_fail():
    print("✓ Root counts match expected")
    # Proceed with analysis
else:
    print("✗ Unexpected root counts")
    # Handle QC failure
```

**Batch filtering**:
```python
# Load multiple plants
plants = sr.load_series_from_slps(
    "data/",
    primary_pattern="*primary*.slp",
    lateral_pattern="*lateral*.slp"
)

# Set expected counts for all
for s in plants:
    s.expected_count(primary=1, lateral=5)

# Filter plants that pass QC
valid_plants = [s for s in plants if not s.qc_fail()]
print(f"Valid plants: {len(valid_plants)}/{len(plants)}")
```

**See Also**:
- [qc_fail](#qc_fail)

---

#### qc_fail

::: sleap_roots.Series.qc_fail
    options:
      show_source: true
      heading_level: 4

**Example**:
```python
import sleap_roots as sr

series = sr.Series.load(
    "plant1",
    primary_path="primary.slp",
    lateral_path="lateral.slp"
)

# Set expected counts
series.expected_count(primary=1, lateral=5)

# Check QC status
if series.qc_fail():
    print(f"QC failed for {series.series_name}")
    print(f"Expected: 1 primary, 5 lateral")
    print(f"Found: {len(series.get_primary_points())}, {len(series.get_lateral_points())}")
else:
    print("QC passed - proceeding with analysis")
```

**See Also**:
- [expected_count](#expected_count)

---

### Visualization

#### get_frame

::: sleap_roots.Series.get_frame
    options:
      show_source: true
      heading_level: 4

**Example**:
```python
import sleap_roots as sr
import matplotlib.pyplot as plt

series = sr.Series.load(
    "plant1",
    h5_path="data.h5",
    primary_path="primary.slp"
)

# Get frame as image array
frame_img = series.get_frame(
    frame_idx=0,
    video_idx=0,
    return_rgb=True
)

# Display with matplotlib
plt.imshow(frame_img)
plt.title(f"{series.series_name} - Frame 0")
plt.axis('off')
plt.show()
```

**See Also**:
- [plot](#plot) - High-level plotting with predictions overlaid

---

#### plot

::: sleap_roots.Series.plot
    options:
      show_source: true
      heading_level: 4

**Example**:
```python
import sleap_roots as sr
import matplotlib.pyplot as plt

series = sr.Series.load(
    "plant1",
    h5_path="data.h5",
    primary_path="primary.slp",
    lateral_path="lateral.slp"
)

# Plot with predictions overlaid
fig = series.plot(
    frame_idx=0,
    primary=True,
    lateral=True,
    plot_scale=1.0
)
plt.show()

# Customize plot
fig = series.plot(
    frame_idx=5,
    primary=True,
    lateral=True,
    plot_scale=0.5  # Smaller display
)
plt.title(f"{series.series_name} - Frame 5")
plt.show()
```

**See Also**:
- [get_frame](#get_frame)

---

### Properties

The `Series` class has the following attributes:

- **series_name**: `str` - Name identifier for the plant
- **h5_path**: `str | None` - Path to H5 image file
- **primary_path**: `str | None` - Path to primary root SLP file
- **lateral_path**: `str | None` - Path to lateral root SLP file
- **crown_path**: `str | None` - Path to crown root SLP file
- **primary_labels**: `sleap_io.Labels | None` - Loaded primary root labels
- **lateral_labels**: `sleap_io.Labels | None` - Loaded lateral root labels
- **crown_labels**: `sleap_io.Labels | None` - Loaded crown root labels
- **video**: `sleap_io.Video | None` - Loaded video object
- **csv_path**: `str | None` - Path for CSV export

**Example**:
```python
import sleap_roots as sr

series = sr.Series.load(
    "plant1",
    h5_path="data.h5",
    primary_path="primary.slp"
)

# Access properties
print(f"Plant name: {series.series_name}")
print(f"H5 path: {series.h5_path}")
print(f"Primary predictions: {series.primary_path}")
print(f"Has video: {series.video is not None}")

# Check what was loaded
has_primary = series.primary_labels is not None
has_lateral = series.lateral_labels is not None
has_crown = series.crown_labels is not None
print(f"Loaded: primary={has_primary}, lateral={has_lateral}, crown={has_crown}")
```

---

### Utility Functions

These module-level functions help with batch loading:

#### find_all_h5_paths

::: sleap_roots.find_all_h5_paths
    options:
      show_source: true
      heading_level: 4

**Example**:
```python
from sleap_roots import find_all_h5_paths

# Find all H5 files in directory
h5_files = find_all_h5_paths("data/predictions/")
print(f"Found {len(h5_files)} H5 files")

for h5_path in h5_files:
    print(h5_path)
```

---

#### find_all_slp_paths

::: sleap_roots.find_all_slp_paths
    options:
      show_source: true
      heading_level: 4

**Example**:
```python
from sleap_roots import find_all_slp_paths

# Find all SLP files in directory
slp_files = find_all_slp_paths("data/predictions/")
print(f"Found {len(slp_files)} SLP files")

# Filter by pattern
primary_files = [f for f in slp_files if 'primary' in f.name]
lateral_files = [f for f in slp_files if 'lateral' in f.name]
```

---

#### load_series_from_h5s

::: sleap_roots.load_series_from_h5s
    options:
      show_source: true
      heading_level: 4

**Example**:
```python
from sleap_roots import load_series_from_h5s

# Load all plants from H5 directory
plants = load_series_from_h5s(
    h5_dir="data/h5_files/",
    primary_pattern="*primary*.slp",
    lateral_pattern="*lateral*.slp"
)

print(f"Loaded {len(plants)} plants")

# Process each plant
for series in plants:
    primary_pts = series.get_primary_points()
    print(f"{series.series_name}: {primary_pts.shape}")
```

---

#### load_series_from_slps

::: sleap_roots.load_series_from_slps
    options:
      show_source: true
      heading_level: 4

**Example**:
```python
from sleap_roots import load_series_from_slps

# Load all plants from SLP directory
plants = load_series_from_slps(
    slp_dir="data/predictions/",
    primary_pattern="*primary*.slp",
    lateral_pattern="*lateral*.slp",
    crown_pattern="*crown*.slp"
)

print(f"Loaded {len(plants)} plants")

# Batch processing
for series in plants:
    # Set QC expectations
    series.expected_count(primary=1, lateral=5)

    # Filter by QC
    if not series.qc_fail():
        # Run analysis
        primary_pts = series.get_primary_points()
        lateral_pts_list = series.get_lateral_points()
```

---

## Complete Workflow Example

```python
import sleap_roots as sr
import numpy as np

# 1. Load plant data
series = sr.Series.load(
    series_name="arabidopsis_1",
    h5_path="data/arabidopsis.h5",
    primary_path="data/arabidopsis_primary.slp",
    lateral_path="data/arabidopsis_lateral.slp"
)

# 2. Quality control
series.expected_count(primary=1, lateral=8)
if series.qc_fail():
    print(f"Warning: {series.series_name} failed QC")

# 3. Extract root points
primary_pts = series.get_primary_points()
lateral_pts_list = series.get_lateral_points()

print(f"Primary roots: {primary_pts.shape}")
print(f"Lateral roots: {len(lateral_pts_list)}")

# 4. Compute custom trait
from sleap_roots import get_root_lengths

primary_lengths = get_root_lengths(primary_pts)
lateral_lengths = [get_root_lengths(pts) for pts in lateral_pts_list]

print(f"Primary root length: {np.nanmean(primary_lengths):.2f} pixels")
print(f"Mean lateral length: {np.nanmean([np.nanmean(l) for l in lateral_lengths]):.2f} pixels")

# 5. Visualize
import matplotlib.pyplot as plt
fig = series.plot(frame_idx=0, primary=True, lateral=True)
plt.show()

# 6. Or use a pipeline for full trait extraction
pipeline = sr.DicotPipeline()
traits = pipeline.fit_series(series)
print(f"Computed {len(traits)} traits")
```

---

## Advanced Usage

### Working with Multiple Frames

```python
import sleap_roots as sr

series = sr.Series.load("plant1", primary_path="primary.slp")
primary_pts = series.get_primary_points()

# Shape: (plants, frames, nodes, 2)
n_plants, n_frames, n_nodes, _ = primary_pts.shape

# Analyze each frame
for frame_idx in range(n_frames):
    frame_pts = primary_pts[0, frame_idx]  # (nodes, 2)

    # Compute per-frame traits
    from sleap_roots import get_root_angle
    angle = get_root_angle(frame_pts[np.newaxis, :, :])
    print(f"Frame {frame_idx}: angle = {angle[0]:.1f}°")
```

### Batch Processing

```python
import sleap_roots as sr
from pathlib import Path

# Load all plants
plants = sr.load_series_from_slps(
    slp_dir="data/experiment1/",
    primary_pattern="*primary*.slp",
    lateral_pattern="*lateral*.slp"
)

# Process with QC
results = []
for series in plants:
    series.expected_count(primary=1, lateral=5)

    if not series.qc_fail():
        # Run pipeline
        pipeline = sr.DicotPipeline()
        traits = pipeline.fit_series(series)
        results.append({
            'plant': series.series_name,
            **traits
        })

# Export results
import pandas as pd
df = pd.DataFrame(results)
df.to_csv("experiment1_results.csv", index=False)
print(f"Processed {len(results)} valid plants")
```

---

## Related Modules

- **[Pipelines](pipelines.md)** - Pre-built trait computation workflows that use Series
- **[Trait Modules](../traits/lengths.md)** - Individual trait computation functions
- **[Points Utilities](../traits/points.md)** - Helper functions for manipulating point arrays

---

## See Also

- **[Installation Guide](../../getting-started/installation.md)** - Setting up sleap-roots
- **[Quick Start Tutorial](../../getting-started/quickstart.md)** - First analysis walkthrough
- **[Data Formats Guide](../../guides/data-formats/sleap-files.md)** - Understanding SLP and H5 files
- **[DicotPipeline Tutorial](../../tutorials/dicot-pipeline.md)** - Complete pipeline example