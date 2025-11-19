# Common Workflows

Complete end-to-end examples for typical sleap-roots analysis tasks.

## Workflow 1: Quick Pipeline Analysis

The fastest way to extract traits from SLEAP predictions.

```python
import sleap_roots as sr

# Load data
series = sr.Series.load(
    series_name="rice_plant1",
    h5_path="predictions.h5",
    primary_path="primary_roots.slp",
    lateral_path="lateral_roots.slp"
)

# Choose appropriate pipeline
pipeline = sr.YoungerMonocotPipeline()

# Extract all traits
traits_dict = pipeline.compute_plant_traits(series)

# Access specific traits
print(f"Primary root count: {traits_dict['primary_count']}")
print(f"Primary max length: {traits_dict['primary_max_length']:.2f} px")
print(f"Lateral count: {traits_dict['lateral_count']}")
print(f"Network length: {traits_dict['network_length_lower']:.2f} px")
```

**When to use**: You need standard traits quickly and don't need custom computations.

**See also**: [Pipelines documentation](../core/pipelines.md)

---

## Workflow 2: Custom Trait Computation

Compute specific traits with fine-grained control.

```python
import sleap_roots as sr
import numpy as np

# Load data
series = sr.Series.load("canola_plant1", primary_path="primary.slp")

# Get point arrays
primary_pts = series.get_primary_points()  # (plants, frames, nodes, 2)

# Compute multiple traits
lengths = sr.get_root_lengths(primary_pts)
angles = sr.get_root_angle(primary_pts, n_points=5)
curve_indices = sr.get_curve_index(primary_pts)

# Get first frame, first plant
frame_0 = 0
plant_0 = 0

print(f"Primary root length: {lengths[plant_0, frame_0]:.2f} px")
print(f"Growth angle: {angles[plant_0, frame_0]:.2f}°")
print(f"Curvature index: {curve_indices[plant_0, frame_0]:.3f}")

# Compute summary statistics across time
mean_length = np.nanmean(lengths[plant_0, :])
std_angle = np.nanstd(angles[plant_0, :])

print(f"\nAcross {series.frame_count} frames:")
print(f"Mean length: {mean_length:.2f} px")
print(f"Angle variability: {std_angle:.2f}°")
```

**When to use**: You need specific traits not available in pipelines, or want custom processing.

**See also**: [Lengths](../traits/lengths.md), [Angles](../traits/angles.md)

---

## Workflow 3: Lateral Root Analysis

Analyze lateral root emergence and distribution patterns.

```python
import sleap_roots as sr
import numpy as np

# Load with laterals
series = sr.Series.load(
    "wheat_plant1",
    primary_path="primary.slp",
    lateral_path="lateral.slp"
)

# Get lateral points (list of arrays)
lateral_pts_list = series.get_lateral_points()
primary_pts = series.get_primary_points()

# Analyze each lateral root
lateral_count = len(lateral_pts_list)
print(f"Detected {lateral_count} lateral roots\n")

for i, lateral_pts in enumerate(lateral_pts_list):
    # Compute traits for this lateral
    length = sr.get_root_lengths(lateral_pts)[0, 0]
    angle = sr.get_root_angle(lateral_pts)[0, 0]

    print(f"Lateral {i+1}:")
    print(f"  Length: {length:.2f} px")
    print(f"  Angle: {angle:.2f}°")

# Get lateral base positions along primary
bases = sr.get_bases(primary_pts, lateral_pts_list)
base_ys = bases[:, 1]  # y-coordinates

# Compute emergence density
primary_length = sr.get_root_lengths(primary_pts)[0, 0]
base_density = sr.get_base_ct_density(primary_pts, lateral_pts_list)[0, 0]

print(f"\nEmergence pattern:")
print(f"Primary length: {primary_length:.2f} px")
print(f"Lateral density: {base_density:.3f} laterals/px")
print(f"Base positions (y): {base_ys}")
```

**When to use**: Studying lateral root development, emergence patterns, or architecture.

**See also**: [Bases](../traits/bases.md), [Lateral Pipeline](../core/pipelines.md#lateral-root-pipeline)

---

## Workflow 4: Temporal Growth Analysis

Track root growth over time using multiple frames.

```python
import sleap_roots as sr
import numpy as np
import matplotlib.pyplot as plt

# Load time series data
series = sr.Series.load("timeseries_plant", primary_path="primary.slp")
primary_pts = series.get_primary_points()  # (1, frames, nodes, 2)

frames = series.frame_count
plant_idx = 0

# Compute traits over time
lengths_over_time = []
angles_over_time = []
tip_ys_over_time = []

for frame_idx in range(frames):
    frame_pts = primary_pts[plant_idx:plant_idx+1, frame_idx:frame_idx+1]

    length = sr.get_root_lengths(frame_pts)[0, 0]
    angle = sr.get_root_angle(frame_pts)[0, 0]
    tip_y = sr.get_tip_ys(frame_pts)[0, 0]

    lengths_over_time.append(length)
    angles_over_time.append(angle)
    tip_ys_over_time.append(tip_y)

# Analyze growth rate
growth_rates = np.diff(lengths_over_time)
mean_growth_rate = np.mean(growth_rates)

print(f"Analysis across {frames} frames:")
print(f"Initial length: {lengths_over_time[0]:.2f} px")
print(f"Final length: {lengths_over_time[-1]:.2f} px")
print(f"Total growth: {lengths_over_time[-1] - lengths_over_time[0]:.2f} px")
print(f"Mean growth rate: {mean_growth_rate:.2f} px/frame")

# Plot growth trajectory
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(lengths_over_time)
axes[0].set_title('Root Length Over Time')
axes[0].set_xlabel('Frame')
axes[0].set_ylabel('Length (px)')

axes[1].plot(angles_over_time)
axes[1].set_title('Growth Angle Over Time')
axes[1].set_xlabel('Frame')
axes[1].set_ylabel('Angle (°)')

axes[2].plot(tip_ys_over_time)
axes[2].set_title('Tip Position Over Time')
axes[2].set_xlabel('Frame')
axes[2].set_ylabel('Y Position (px)')

plt.tight_layout()
plt.savefig('growth_analysis.png', dpi=150)
print("\nSaved growth_analysis.png")
```

**When to use**: Time-lapse experiments, growth rate analysis, gravitropism studies.

**See also**: [Series documentation](../core/series.md), [Tips](../traits/tips.md)

---

## Workflow 5: Network-Level Spatial Analysis

Analyze whole root system architecture and spatial distribution.

```python
import sleap_roots as sr
from sleap_roots import join_pts

# Load complete root system
series = sr.Series.load(
    "mature_plant",
    primary_path="primary.slp",
    lateral_path="lateral.slp"
)

# Get all roots
primary_pts = series.get_primary_points()
lateral_pts_list = series.get_lateral_points()

# Combine into single point array
all_pts = join_pts([primary_pts] + lateral_pts_list)

# Network-level metrics
network_length = sr.get_network_length(all_pts)
wd_ratio = sr.get_network_width_depth_ratio(all_pts)
bbox = sr.get_bbox(all_pts)

print("Network Metrics:")
print(f"Total network length: {network_length:.2f} px")
print(f"Width:Depth ratio: {wd_ratio:.2f}")
print(f"Bounding box: {bbox}")

# Spatial coverage
hull_area = sr.convhull.get_chull_area(all_pts)
hull_perimeter = sr.convhull.get_chull_perimeter(all_pts)
print(f"\nSpatial Coverage:")
print(f"Convex hull area: {hull_area:.2f} px²")
print(f"Convex hull perimeter: {hull_perimeter:.2f} px")

# Ellipse fit for compact representation
pts_2d = sr.get_all_pts_array(all_pts)  # Flatten to (n_points, 2)
ellipse = sr.get_ellipse(pts_2d)
print(f"\nEllipse parameters: {ellipse}")

# Distribution analysis using scanlines
y_positions = [100, 200, 300, 400, 500]
for y in y_positions:
    intersections = sr.count_scanline_intersections(all_pts, y=y)
    print(f"Intersections at y={y}: {intersections}")
```

**When to use**: Characterizing root system architecture, comparing genotypes, spatial phenotyping.

**See also**: [Network Length](../traits/networklength.md), [Convex Hull](../traits/convhull.md), [Scanline](../traits/scanline.md)

---

## Workflow 6: Batch Processing Multiple Plants

Process multiple plant series and aggregate results.

```python
import sleap_roots as sr
import pandas as pd
from pathlib import Path

# Define plant data
plants = [
    {"name": "plant1", "h5": "plant1.h5", "primary": "plant1_primary.slp"},
    {"name": "plant2", "h5": "plant2.h5", "primary": "plant2_primary.slp"},
    {"name": "plant3", "h5": "plant3.h5", "primary": "plant3_primary.slp"},
]

# Load all series
all_series = []
for plant_info in plants:
    series = sr.Series.load(
        plant_info["name"],
        h5_path=plant_info["h5"],
        primary_path=plant_info["primary"]
    )
    all_series.append(series)

# Run batch pipeline processing
pipeline = sr.DicotPipeline()
batch_traits = pipeline.compute_batch_traits(all_series)

# Convert to DataFrame
df = pd.DataFrame(batch_traits)
df["plant_name"] = [p["name"] for p in plants]

print("Results:")
print(df[["plant_name", "primary_length", "primary_angle", "lateral_count"]])

# Save to CSV
df.to_csv("batch_analysis_results.csv", index=False)
print("\nSaved batch_analysis_results.csv")

# Compute summary statistics
print("\nSummary Statistics:")
for col in ["primary_length", "primary_angle", "lateral_count"]:
    stats = sr.get_summary(df[col].values, prefix=f"{col}_")
    print(f"\n{col}:")
    print(f"  Mean: {stats[f'{col}_mean']:.2f}")
    print(f"  Std: {stats[f'{col}_std']:.2f}")
    print(f"  Range: {stats[f'{col}_min']:.2f} - {stats[f'{col}_max']:.2f}")
```

**When to use**: High-throughput phenotyping, genotype comparisons, experimental studies.

**See also**: [Pipelines](../core/pipelines.md), [Summary Statistics](../utilities/summary.md)

---

## Workflow 7: Quality Control and Filtering

Filter out invalid roots and handle edge cases.

```python
import sleap_roots as sr
import numpy as np

# Load data
series = sr.Series.load(
    "experimental_data",
    primary_path="primary.slp",
    lateral_path="lateral.slp"
)

lateral_pts_list = series.get_lateral_points()

print(f"Initial lateral count: {len(lateral_pts_list)}")

# Filter out roots with NaN values (incomplete tracking)
valid_laterals = sr.filter_roots_with_nans(lateral_pts_list)
print(f"Valid laterals after NaN filtering: {len(valid_laterals)}")

# Additional quality filtering: remove very short roots
min_length_threshold = 10.0  # pixels

filtered_laterals = []
for lateral_pts in valid_laterals:
    length = sr.get_root_lengths(lateral_pts)[0, 0]
    if length >= min_length_threshold:
        filtered_laterals.append(lateral_pts)

print(f"Valid laterals after length filtering: {len(filtered_laterals)}")

# Compute traits only on valid roots
if len(filtered_laterals) > 0:
    lengths = [sr.get_root_lengths(pts)[0, 0] for pts in filtered_laterals]
    angles = [sr.get_root_angle(pts)[0, 0] for pts in filtered_laterals]

    print(f"\nFiltered Lateral Statistics:")
    print(f"Mean length: {np.mean(lengths):.2f} px")
    print(f"Mean angle: {np.mean(angles):.2f}°")
    print(f"Length range: {np.min(lengths):.2f} - {np.max(lengths):.2f} px")
else:
    print("No valid lateral roots after filtering")

# Check primary root quality
primary_pts = series.get_primary_points()
has_nans = np.any(np.isnan(primary_pts))

if has_nans:
    print("\nWarning: Primary root contains NaN values")
    # Handle by using only valid frames
    valid_frames = ~np.any(np.isnan(primary_pts), axis=(0, 2, 3))
    clean_pts = primary_pts[:, valid_frames, :, :]
    print(f"Using {np.sum(valid_frames)}/{len(valid_frames)} valid frames")
else:
    print("\nPrimary root tracking is complete")
```

**When to use**: Working with incomplete tracking data, ensuring data quality, debugging.

**See also**: [Points utilities](../traits/points.md), [Series](../core/series.md)

---

## Workflow 8: Multiple Dicot Plants

Process multiple dicot plants simultaneously with specialized pipeline.

```python
import sleap_roots as sr

# Load series with multiple plants
series = sr.Series.load(
    "arabidopsis_tray",
    h5_path="predictions.h5",
    primary_path="primary_roots.slp",
    lateral_path="lateral_roots.slp"
)

# Use multiple dicot pipeline
pipeline = sr.MultipleDicotPipeline()
traits_dict = pipeline.compute_multiple_dicots_traits(series)

# Access per-plant traits
print(f"Plants detected: {len(traits_dict['primary_length'])}")

for i, (length, angle, lat_count) in enumerate(zip(
    traits_dict['primary_length'],
    traits_dict['primary_angle'],
    traits_dict['lateral_count']
)):
    print(f"\nPlant {i+1}:")
    print(f"  Primary length: {length:.2f} px")
    print(f"  Primary angle: {angle:.2f}°")
    print(f"  Lateral count: {lat_count}")
```

**When to use**: High-throughput imaging with multiple plants per image.

**See also**: [MultipleDicotPipeline](../core/pipelines.md#multiple-dicot-pipeline)

---

## Next Steps

- **[API Reference](../index.md)** - Explore all available functions
- **[Pipelines Guide](../core/pipelines.md)** - Choose the right pipeline
- **[Tutorial Notebooks](../../notebooks/index.md)** - Interactive learning
