# Lateral Root Pipeline Tutorial

This tutorial demonstrates specialized analysis of lateral root systems using the `LateralRootPipeline`. This pipeline focuses exclusively on lateral roots, providing detailed metrics for lateral root architecture and development.

## What You'll Learn

- Analyze lateral root systems independently
- Compute lateral-specific traits and topology
- Study lateral root emergence patterns
- Track lateral root development dynamics

## Pipeline Overview

The `LateralRootPipeline` computes:

- **Lateral root counts**: Total number and emergence timing
- **Individual lateral lengths**: Per-root measurements
- **Emergence patterns**: Angles, positions, spacing
- **Network properties**: Aggregate lateral root metrics
- **Developmental dynamics**: Growth rates, initiation patterns

## Interactive Tutorial

{{ '../../notebooks/LateralRootPipeline.ipynb' }}

## When to Use This Pipeline

### Research Focus

Ideal for studies examining:

- **Lateral root initiation**: Timing and frequency of lateral emergence
- **Branching architecture**: Spatial distribution patterns
- **Lateral elongation**: Growth dynamics of individual laterals
- **Environmental responses**: How conditions affect lateral development
- **Comparative anatomy**: Lateral root differences across genotypes

### Versus Full Pipelines

| Feature | LateralRootPipeline | DicotPipeline |
|---------|---------------------|---------------|
| Primary root | Not analyzed | Analyzed |
| Lateral roots | Detailed analysis | Standard analysis |
| Use case | Lateral-focused research | Complete root system |
| Traits | Lateral-specific | Primary + lateral + network |

## Root Architecture

### Expected Structure

```
        Primary Root (not tracked)
              |
    ----------|----------
   /     |    |    |     \
  L1    L2   L3   L4    L5  (Lateral roots tracked)
```

### SLEAP Requirements

- **Lateral root file**: Contains all lateral roots
- **Base nodes**: Each lateral starts at emergence point
- **Tip tracking**: Each lateral tracked to tip
- **No primary needed**: Pipeline focuses on laterals only

## Usage Examples

### Basic Analysis

```python
import sleap_roots as sr

# Load lateral roots only
series = sr.Series.load(
    "plant_laterals",
    h5_path="predictions.h5",
    lateral_path="lateral.slp"
    # No primary_path needed
)

# Compute lateral traits
pipeline = sr.LateralRootPipeline()
traits = pipeline.compute_plant_traits(series, write_csv=True)

print(f"Lateral root count: {traits['lateral_count'].iloc[0]}")
print(f"Total lateral length: {traits['total_lateral_length'].iloc[0]:.2f} px")
print(f"Mean lateral length: {traits['mean_lateral_length'].iloc[0]:.2f} px")
```

### Emergence Pattern Analysis

```python
import numpy as np
import matplotlib.pyplot as plt

# Analyze lateral emergence along primary axis
traits = pipeline.compute_plant_traits(series)

# Get emergence positions (y-coordinates of base nodes)
emergence_positions = traits['lateral_emergence_positions'].iloc[0]

# Plot emergence pattern
plt.figure(figsize=(8, 6))
plt.scatter(emergence_positions, range(len(emergence_positions)))
plt.xlabel('Position along primary root (pixels)')
plt.ylabel('Lateral root index')
plt.title('Lateral Root Emergence Pattern')
plt.grid(True)
plt.show()

# Compute spacing statistics
spacings = np.diff(sorted(emergence_positions))
print(f"Mean spacing: {np.mean(spacings):.2f} px")
print(f"Spacing variability (std): {np.std(spacings):.2f} px")
```

### Lateral Length Distribution

```python
# Analyze distribution of lateral lengths
lateral_lengths = traits['lateral_root_lengths'].iloc[0]

plt.figure(figsize=(10, 6))
plt.hist(lateral_lengths, bins=20, edgecolor='black')
plt.xlabel('Lateral Root Length (pixels)')
plt.ylabel('Count')
plt.title('Distribution of Lateral Root Lengths')
plt.axvline(np.mean(lateral_lengths), color='r',
           linestyle='--', label=f'Mean: {np.mean(lateral_lengths):.1f} px')
plt.legend()
plt.show()

# Identify longest and shortest
print(f"Longest lateral: {max(lateral_lengths):.2f} px")
print(f"Shortest lateral: {min(lateral_lengths):.2f} px")
print(f"Length range: {max(lateral_lengths) - min(lateral_lengths):.2f} px")
```

## Key Traits

### Count Metrics

| Trait | Description | Units |
|-------|-------------|-------|
| `lateral_count` | Total number of lateral roots | count |
| `lateral_count_left` | Laterals on left side | count |
| `lateral_count_right` | Laterals on right side | count |

### Length Measurements

| Trait | Description | Units |
|-------|-------------|-------|
| `total_lateral_length` | Sum of all lateral lengths | pixels |
| `mean_lateral_length` | Average lateral length | pixels |
| `median_lateral_length` | Median lateral length | pixels |
| `lateral_root_lengths` | Individual lengths (array) | pixels |

### Spatial Distribution

| Trait | Description | Units |
|-------|-------------|-------|
| `lateral_emergence_positions` | Position of each lateral base | pixels (array) |
| `lateral_emergence_spacing` | Distance between adjacent laterals | pixels |
| `lateral_density` | Laterals per unit primary length | count/pixel |

### Angular Properties

| Trait | Description | Units |
|-------|-------------|-------|
| `lateral_emergence_angles` | Angle of each lateral at base | degrees (array) |
| `mean_emergence_angle` | Average emergence angle | degrees |
| `angular_variance` | Variability in emergence angles | degrees² |

## Advanced Analysis

### Temporal Development

```python
# Track lateral root development over time
series = sr.Series.load(
    "timeseries",
    h5_path="timeseries.h5",
    lateral_path="lateral.slp"
)

traits = pipeline.compute_plant_traits(series)

# Plot lateral count over time
plt.figure(figsize=(10, 6))
plt.plot(traits['frame'], traits['lateral_count'], 'o-')
plt.xlabel('Frame')
plt.ylabel('Lateral Root Count')
plt.title('Lateral Root Initiation Over Time')
plt.grid(True)
plt.show()

# Calculate initiation rate
frames_with_new_laterals = traits[traits['lateral_count'].diff() > 0]
mean_initiation_interval = frames_with_new_laterals['frame'].diff().mean()
print(f"Average frames between new laterals: {mean_initiation_interval:.1f}")
```

### Lateral Growth Dynamics

```python
# Track individual lateral growth (requires consistent tracking)
# Assuming laterals maintain consistent IDs across frames

import pandas as pd

growth_data = []
for frame_idx in range(len(series.lateral_pts)):
    frame_laterals = series.lateral_pts[frame_idx]

    for lateral_id, lateral_pts in enumerate(frame_laterals):
        if len(lateral_pts) > 0:
            length = sr.lengths.get_root_lengths([lateral_pts])[0]
            growth_data.append({
                'frame': frame_idx,
                'lateral_id': lateral_id,
                'length': length
            })

df = pd.DataFrame(growth_data)

# Plot growth curves for each lateral
plt.figure(figsize=(12, 6))
for lateral_id in df['lateral_id'].unique():
    lateral_data = df[df['lateral_id'] == lateral_id]
    plt.plot(lateral_data['frame'], lateral_data['length'],
            label=f'Lateral {lateral_id}')

plt.xlabel('Frame')
plt.ylabel('Lateral Length (pixels)')
plt.title('Individual Lateral Root Growth Curves')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
```

### Emergence Angle Analysis

```python
# Analyze lateral emergence angles
emergence_angles = traits['lateral_emergence_angles'].iloc[0]

plt.figure(figsize=(10, 6))
plt.hist(emergence_angles, bins=30, edgecolor='black')
plt.xlabel('Emergence Angle (degrees)')
plt.ylabel('Count')
plt.title('Distribution of Lateral Root Emergence Angles')
plt.axvline(np.mean(emergence_angles), color='r',
           linestyle='--', label=f'Mean: {np.mean(emergence_angles):.1f}°')
plt.legend()
plt.show()

# Check for left-right asymmetry
left_angles = emergence_angles[emergence_angles < 0]
right_angles = emergence_angles[emergence_angles > 0]

print(f"Left-side mean angle: {np.mean(left_angles):.2f}°")
print(f"Right-side mean angle: {np.mean(right_angles):.2f}°")
print(f"Asymmetry: {abs(np.mean(left_angles)) - abs(np.mean(right_angles)):.2f}°")
```

## Comparison Studies

### Genotype Comparison

```python
# Compare lateral root architecture across genotypes
genotypes = ['WT', 'mutant_high_lateral', 'mutant_low_lateral']
results = []

for genotype in genotypes:
    series = sr.Series.load(
        genotype,
        h5_path=f"{genotype}.h5",
        lateral_path=f"{genotype}_lateral.slp"
    )
    traits = pipeline.compute_plant_traits(series)
    traits['genotype'] = genotype
    results.append(traits)

df = pd.concat(results)

# Compare lateral counts
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.boxplot(data=df, x='genotype', y='lateral_count', ax=axes[0])
axes[0].set_title('Lateral Root Count by Genotype')

sns.boxplot(data=df, x='genotype', y='mean_lateral_length', ax=axes[1])
axes[1].set_title('Mean Lateral Length by Genotype')

plt.tight_layout()
plt.show()
```

### Treatment Effects

```python
# Assess treatment impact on lateral development
treatments = ['control', 'nitrogen', 'drought']
lateral_counts = []
lateral_lengths = []

for treatment in treatments:
    series = sr.Series.load(
        treatment,
        h5_path=f"{treatment}.h5",
        lateral_path=f"{treatment}_lateral.slp"
    )
    traits = pipeline.compute_plant_traits(series)

    lateral_counts.append(traits['lateral_count'].iloc[0])
    lateral_lengths.append(traits['mean_lateral_length'].iloc[0])

# Visualize treatment effects
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].bar(treatments, lateral_counts)
axes[0].set_ylabel('Lateral Count')
axes[0].set_title('Lateral Initiation by Treatment')

axes[1].bar(treatments, lateral_lengths)
axes[1].set_ylabel('Mean Length (pixels)')
axes[1].set_title('Lateral Elongation by Treatment')

plt.tight_layout()
plt.show()
```

## Lateral Root Topology

### Branching Patterns

```python
# Analyze branching pattern characteristics
traits = pipeline.compute_plant_traits(series)

# Compute branching index (laterals per unit length)
if 'primary_length' in traits.columns:
    branching_index = traits['lateral_count'] / traits['primary_length']
    print(f"Branching index: {branching_index.iloc[0]:.4f} laterals/pixel")

# Analyze spatial clustering
emergence_pos = traits['lateral_emergence_positions'].iloc[0]
sorted_pos = sorted(emergence_pos)

# Detect clusters (laterals closer than threshold)
cluster_threshold = 50  # pixels
clusters = []
current_cluster = [sorted_pos[0]]

for pos in sorted_pos[1:]:
    if pos - current_cluster[-1] < cluster_threshold:
        current_cluster.append(pos)
    else:
        clusters.append(current_cluster)
        current_cluster = [pos]
clusters.append(current_cluster)

print(f"Detected {len(clusters)} lateral root clusters")
for i, cluster in enumerate(clusters):
    print(f"Cluster {i+1}: {len(cluster)} laterals")
```

## Troubleshooting

**Low lateral count detected:**
- Verify SLEAP model detects all lateral roots
- Check tracking quality and confidence thresholds
- Ensure laterals aren't merged into single track

**Inconsistent lengths:**
- Confirm lateral roots tracked from base to tip
- Check for tracking discontinuities
- Verify consistent frame-to-frame tracking

**Incorrect emergence positions:**
- Ensure lateral base nodes correctly positioned
- Verify primary root alignment in SLEAP skeleton
- Check for flipped coordinates

**Missing temporal data:**
- Confirm H5 file includes all frames
- Verify frame indices align across timepoints
- Check for tracking gaps

## Integration with Full Pipelines

Compare lateral-only results with full pipeline:

```python
# Run both pipelines
lateral_pipeline = sr.LateralRootPipeline()
dicot_pipeline = sr.DicotPipeline()

# Load with both primary and lateral
full_series = sr.Series.load(
    "plant",
    h5_path="pred.h5",
    primary_path="primary.slp",
    lateral_path="lateral.slp"
)

# Load lateral-only
lateral_series = sr.Series.load(
    "plant",
    h5_path="pred.h5",
    lateral_path="lateral.slp"
)

lateral_traits = lateral_pipeline.compute_plant_traits(lateral_series)
full_traits = dicot_pipeline.compute_plant_traits(full_series)

# Compare lateral metrics
print("Lateral-only pipeline:", lateral_traits['lateral_count'].iloc[0])
print("Full dicot pipeline:", full_traits['lateral_count'].iloc[0])
```

## Next Steps

- See [Dicot Pipeline](dicot-pipeline.md) for combined primary + lateral analysis
- Try [Multiple Dicot Pipeline](multiple-dicot-pipeline.md) for multi-plant laterals
- Read [Trait Reference](../guides/trait-reference.md) for all lateral traits
- Explore [Custom Traits](../cookbook/custom-traits.md) for specialized lateral metrics