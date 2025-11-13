# Primary Root Pipeline Tutorial

This tutorial covers analysis of single primary roots using the `PrimaryRootPipeline`. This specialized pipeline focuses exclusively on primary root traits, providing a streamlined workflow for simple root architectures.

## What You'll Learn

- Analyze primary roots in isolation
- Compute focused primary root metrics
- Track primary root development over time
- Export simplified trait datasets

## Pipeline Overview

The `PrimaryRootPipeline` computes:

- **Length metrics**: Total length, smoothed length
- **Geometric properties**: Tip angle, curvature, straightness
- **Growth dynamics**: Elongation rates, temporal patterns
- **Spatial features**: Tip position, root trajectory

## Interactive Tutorial

{{ '../../notebooks/PrimaryRootPipeline.ipynb' }}

## When to Use This Pipeline

### Ideal Scenarios

- **Early development**: Seedlings before lateral emergence
- **Primary root focus**: Research targeting primary root only
- **Simplified phenotyping**: Quick screening of primary growth
- **Gravitropism studies**: Tracking primary root orientation
- **Subset analysis**: Extracting primary traits from larger pipelines

### Versus Other Pipelines

| Pipeline | Roots Analyzed | Use When |
|----------|----------------|----------|
| PrimaryRootPipeline | Primary only | No laterals/crown roots |
| DicotPipeline | Primary + laterals | Full dicot system |
| YoungerMonocotPipeline | Primary + crown | Early monocot |
| MultiplePrimaryRootPipeline | Multiple primary roots | Multi-plant primary-only |

## Root Architecture

### Expected Structure

```
Base
  |
  |  Primary Root
  |  (tracked continuously)
  |
  ↓
Tip
```

### SLEAP Requirements

- **Single tracked root**: One root from base to tip
- **Node consistency**: Consistent node naming
- **Complete tracking**: No gaps from base to tip

## Usage Examples

### Basic Analysis

```python
import sleap_roots as sr

# Load primary root only
series = sr.Series.load(
    "plant_primary",
    h5_path="predictions.h5",
    primary_path="primary.slp"
    # No lateral_path needed
)

# Compute primary traits
pipeline = sr.PrimaryRootPipeline()
traits = pipeline.compute_plant_traits(series, write_csv=True)

print(f"Primary root length: {traits['primary_length'].iloc[0]:.2f} px")
print(f"Tip angle: {traits['primary_tip_angle'].iloc[0]:.2f}°")
```

### Temporal Growth Tracking

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load time series
series = sr.Series.load(
    "plant_timeseries",
    h5_path="timeseries.h5",
    primary_path="primary.slp"
)

traits = pipeline.compute_plant_traits(series)

# Plot growth curve
plt.figure(figsize=(10, 6))
plt.plot(traits['frame'], traits['primary_length'])
plt.xlabel('Frame')
plt.ylabel('Primary Root Length (pixels)')
plt.title('Primary Root Growth Over Time')
plt.grid(True)
plt.show()

# Calculate growth rate
traits['growth_rate'] = traits['primary_length'].diff()
avg_growth_rate = traits['growth_rate'].mean()
print(f"Average growth rate: {avg_growth_rate:.2f} pixels/frame")
```

### Gravitropism Analysis

```python
# Analyze root angle changes over time
traits = pipeline.compute_plant_traits(series)

plt.figure(figsize=(10, 6))
plt.plot(traits['frame'], traits['primary_tip_angle'])
plt.axhline(y=0, color='r', linestyle='--', label='Vertical')
plt.xlabel('Frame')
plt.ylabel('Tip Angle (degrees)')
plt.title('Primary Root Gravitropic Response')
plt.legend()
plt.grid(True)
plt.show()

# Detect gravity response
initial_angle = traits['primary_tip_angle'].iloc[0]
final_angle = traits['primary_tip_angle'].iloc[-1]
angle_change = final_angle - initial_angle
print(f"Angle change: {angle_change:.2f}° (negative = downward)")
```

## Key Traits

### Length Measurements

| Trait | Description | Typical Use |
|-------|-------------|-------------|
| `primary_length` | Total path length from base to tip | Overall growth |
| `primary_length_smooth` | Smoothed length (noise-reduced) | Robust measurement |
| `primary_euclidean_length` | Straight-line base to tip | Straightness comparison |

### Geometric Properties

| Trait | Description | Range |
|-------|-------------|-------|
| `primary_tip_angle` | Angle of tip segment relative to vertical | -180° to 180° |
| `primary_curvature` | Overall root curvature | >0 (lower = straighter) |
| `primary_straightness` | Euclidean / path length ratio | 0-1 (1 = perfectly straight) |

### Spatial Features

| Trait | Description | Units |
|-------|-------------|-------|
| `primary_tip_x` | Horizontal tip position | pixels |
| `primary_tip_y` | Vertical tip position | pixels |
| `primary_depth` | Vertical extent from base | pixels |

## Advanced Analysis

### Root Curvature Profiles

```python
import numpy as np

# Compute curvature along root length
def compute_curvature_profile(pts):
    # pts: Nx2 array of root coordinates
    # Returns curvature at each point
    dx = np.gradient(pts[:, 0])
    dy = np.gradient(pts[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
    return curvature

curvature = compute_curvature_profile(series.primary_pts[0])

plt.plot(curvature)
plt.xlabel('Position along root')
plt.ylabel('Curvature')
plt.title('Root Curvature Profile')
plt.show()
```

### Growth Rate Analysis

```python
# Compute instantaneous and average growth rates
traits['instantaneous_growth'] = traits['primary_length'].diff()
traits['cumulative_growth'] = traits['primary_length'] - traits['primary_length'].iloc[0]

# Fit growth model
from scipy.optimize import curve_fit

def exponential_growth(t, L0, k):
    return L0 * np.exp(k * t)

params, _ = curve_fit(
    exponential_growth,
    traits['frame'],
    traits['primary_length']
)

print(f"Initial length: {params[0]:.2f} px")
print(f"Growth rate constant: {params[1]:.4f}")
```

### Tip Trajectory Visualization

```python
# Extract and plot tip trajectory over time
tip_trajectory = np.array([
    series.primary_pts[i][-1]  # Last point = tip
    for i in range(len(series.primary_pts))
])

plt.figure(figsize=(8, 10))
plt.plot(tip_trajectory[:, 0], tip_trajectory[:, 1], 'o-')
plt.scatter(tip_trajectory[0, 0], tip_trajectory[0, 1],
           color='green', s=100, label='Start', zorder=5)
plt.scatter(tip_trajectory[-1, 0], tip_trajectory[-1, 1],
           color='red', s=100, label='End', zorder=5)
plt.xlabel('X position (pixels)')
plt.ylabel('Y position (pixels)')
plt.title('Primary Root Tip Trajectory')
plt.legend()
plt.gca().invert_yaxis()  # Image coordinates
plt.axis('equal')
plt.show()
```

## Comparison Studies

### Genotype Comparison

```python
# Compare primary root growth across genotypes
genotypes = ['WT', 'mutant_A', 'mutant_B']
results = []

for genotype in genotypes:
    series = sr.Series.load(
        genotype,
        h5_path=f"{genotype}.h5",
        primary_path=f"{genotype}_primary.slp"
    )
    traits = pipeline.compute_plant_traits(series)
    traits['genotype'] = genotype
    results.append(traits)

df = pd.concat(results)

# Statistical comparison
import seaborn as sns
sns.boxplot(data=df, x='genotype', y='primary_length')
plt.title('Primary Root Length by Genotype')
plt.show()
```

### Treatment Effects

```python
# Compare control vs. treated plants
control_series = sr.Series.load(
    "control", h5_path="control.h5", primary_path="control_primary.slp"
)
treated_series = sr.Series.load(
    "treated", h5_path="treated.h5", primary_path="treated_primary.slp"
)

control_traits = pipeline.compute_plant_traits(control_series)
treated_traits = pipeline.compute_plant_traits(treated_series)

# Compare final lengths
control_length = control_traits['primary_length'].iloc[-1]
treated_length = treated_traits['primary_length'].iloc[-1]
percent_change = (treated_length - control_length) / control_length * 100

print(f"Control: {control_length:.2f} px")
print(f"Treated: {treated_length:.2f} px")
print(f"Change: {percent_change:.1f}%")
```

## Troubleshooting

**Tracking discontinuities:**
- Review SLEAP predictions for gaps
- Increase tracking confidence threshold
- Manually correct breaks in SLEAP GUI

**Noisy length measurements:**
- Use `primary_length_smooth` instead of `primary_length`
- Apply temporal smoothing to trait time series
- Check image quality and contrast

**Incorrect tip angle:**
- Verify tip is correctly identified (last tracked point)
- Check for tracking artifacts near tip
- Ensure sufficient points near tip for angle calculation

## Integration with Larger Pipelines

Extract primary-only data from full pipeline results:

```python
# Run full dicot pipeline
full_pipeline = sr.DicotPipeline()
full_traits = full_pipeline.compute_plant_traits(series)

# Extract primary traits (equivalent to PrimaryRootPipeline)
primary_traits = full_traits[[
    'primary_length',
    'primary_tip_angle',
    'primary_length_smooth',
    # ... other primary traits
]]
```

## Next Steps

- See [Dicot Pipeline](dicot-pipeline.md) for primary + lateral analysis
- Try [Multiple Primary Root Pipeline](multiple-primary-root-pipeline.md) for multi-plant setups
- Read [Trait Reference](../guides/trait-reference.md) for detailed trait definitions
- Explore [Custom Traits Cookbook](../cookbook/custom-traits.md) for advanced metrics