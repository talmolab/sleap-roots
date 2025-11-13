# Multiple Primary Root Pipeline Tutorial

This tutorial demonstrates analysis of multiple plants with primary roots using the `MultiplePrimaryRootPipeline`. This pipeline is designed for multi-plant setups where each plant has a single primary root without lateral branches.

## What You'll Learn

- Process multiple primary-root plants simultaneously
- Handle simplified multi-plant architectures
- Compute primary root traits in batch
- Compare growth across multiple individuals

## Pipeline Overview

The `MultiplePrimaryRootPipeline` provides:

- **Individual plant identification** via SLEAP track IDs
- **Primary root trait computation** for each plant
- **Comparative metrics** across plants
- **Streamlined output** for primary-root-only systems

## Interactive Tutorial

{{ '../../notebooks/MultiplePrimaryRootPipeline.ipynb' }}

## Use Cases

### Ideal Applications

- **Early-stage phenotyping**: Young seedlings before lateral emergence
- **Simple architectures**: Species with minimal branching
- **Root length screens**: Focused on primary root growth
- **High-throughput imaging**: Maximum plants per frame

### Experimental Examples

**Germination assays:**
- Screen 10-20 seedlings per image
- Track primary root emergence and elongation
- Compare germination vigor across genotypes

**Stress response:**
- Monitor primary root growth under drought, salinity
- Compare control vs. treatment in same frame
- Minimize imaging time and variability

**Gravitropism studies:**
- Track primary root angle and curvature
- Multiple plants per treatment condition
- Temporal analysis of root orientation

## Root System Architecture

### Expected Structure

```
Plant 1: Base → Primary Root → Tip
Plant 2: Base → Primary Root → Tip
Plant 3: Base → Primary Root → Tip
   ...
Plant N: Base → Primary Root → Tip
```

### Key Features

- **Single root per plant**: No laterals or crown roots
- **Track-based separation**: Each plant has unique track ID
- **Minimal complexity**: Faster processing than full dicot/monocot pipelines

## Usage Examples

### Basic Analysis

```python
import sleap_roots as sr

# Load all plants
series_list = sr.Series.load_multi(
    h5_path="primary_roots.h5",
    primary_path="primary.slp"
    # Note: No lateral_path needed
)

# Process all plants
pipeline = sr.MultiplePrimaryRootPipeline()
traits = pipeline.compute_multi_plant_traits(
    series_list,
    write_csv=True,
    csv_path="primary_traits.csv"
)

print(f"Analyzed {len(series_list)} plants")
```

### Germination Screen

```python
# Screen many seedlings
series_list = sr.Series.load_multi(
    h5_path="germination_plate.h5",
    primary_path="primary.slp"
)

traits = pipeline.compute_multi_plant_traits(series_list)

# Filter by germination success
germinated = traits[traits['primary_length'] > 50]  # >50 pixels
print(f"Germination rate: {len(germinated)/len(traits)*100:.1f}%")

# Find fastest growing
fastest = traits.loc[traits['primary_length'].idxmax()]
print(f"Fastest growth: {fastest['primary_length']:.1f} pixels")
```

### Temporal Growth Analysis

```python
import pandas as pd
import matplotlib.pyplot as plt

# Collect traits across multiple timepoints
all_data = []
for hour in range(0, 72, 6):  # Every 6 hours for 3 days
    series_list = sr.Series.load_multi(
        h5_path=f"hour_{hour}.h5",
        primary_path=f"primary_{hour}.slp"
    )

    traits = pipeline.compute_multi_plant_traits(series_list)
    traits['hour'] = hour
    all_data.append(traits)

df = pd.concat(all_data)

# Plot growth curves
for plant_id in df['plant_id'].unique():
    plant_data = df[df['plant_id'] == plant_id]
    plt.plot(plant_data['hour'], plant_data['primary_length'],
             label=f'Plant {plant_id}')

plt.xlabel('Time (hours)')
plt.ylabel('Primary Root Length (pixels)')
plt.legend()
plt.title('Primary Root Growth Over Time')
plt.show()
```

## Key Traits

### Length Metrics

| Trait | Description | Units |
|-------|-------------|-------|
| `primary_length` | Total root length from base to tip | pixels |
| `primary_length_smooth` | Smoothed length (reduces noise) | pixels |

### Geometric Traits

| Trait | Description | Units |
|-------|-------------|-------|
| `primary_tip_angle` | Angle of tip relative to vertical | degrees |
| `primary_curvature` | Overall root curvature | 1/pixels |
| `primary_straightness` | Euclidean distance / path length | 0-1 |

### Growth Metrics

| Trait | Description | Units |
|-------|-------------|-------|
| `growth_rate` | Length change per frame | pixels/frame |
| `relative_growth_rate` | Proportional growth rate | %/frame |

## Multi-Plant Comparisons

### Statistical Summaries

```python
# Summary statistics across all plants
summary = traits.groupby('plant_id')['primary_length'].agg([
    'mean', 'std', 'min', 'max'
])
print(summary)
```

### Genotype Comparisons

```python
# Add genotype information
genotype_map = {0: 'WT', 1: 'WT', 2: 'mutant', 3: 'mutant'}
traits['genotype'] = traits['plant_id'].map(genotype_map)

# Compare
import seaborn as sns
sns.boxplot(data=traits, x='genotype', y='primary_length')
plt.title('Primary Root Length by Genotype')
plt.show()

# Statistical test
from scipy import stats
wt = traits[traits['genotype'] == 'WT']['primary_length']
mut = traits[traits['genotype'] == 'mutant']['primary_length']
t_stat, p_value = stats.ttest_ind(wt, mut)
print(f"P-value: {p_value:.4f}")
```

### Ranking and Selection

```python
# Identify top performers
traits_sorted = traits.sort_values('primary_length', ascending=False)
top_5 = traits_sorted.head(5)
print("Top 5 longest roots:")
print(top_5[['plant_id', 'primary_length']])

# Select plants for next generation
selection_threshold = traits['primary_length'].quantile(0.75)
selected = traits[traits['primary_length'] > selection_threshold]
print(f"Selected {len(selected)} plants (top 25%)")
```

## Data Quality Checks

### Detecting Tracking Issues

```python
# Check for abnormally short roots (tracking failures)
min_length = 20  # pixels
failed_tracks = traits[traits['primary_length'] < min_length]
print(f"Potential tracking failures: {len(failed_tracks)} plants")

# Remove problematic plants
clean_traits = traits[traits['primary_length'] >= min_length]
```

### Validating Plant Count

```python
expected_count = 12  # 12 plants in setup
actual_count = len(series_list)

if actual_count != expected_count:
    print(f"Warning: Expected {expected_count}, found {actual_count}")
    # Investigate missing or extra tracks
```

## Performance Optimization

### Parallel Processing

```python
from concurrent.futures import ProcessPoolExecutor

def process_single(series):
    pipeline = sr.MultiplePrimaryRootPipeline()
    return pipeline.compute_plant_traits(series)

# Process plants in parallel
with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_single, series_list))

df = pd.concat(results, ignore_index=True)
```

### Batch vs. Individual

```python
# Batch processing (recommended for <20 plants)
traits = pipeline.compute_multi_plant_traits(series_list)

# Individual processing (for custom logic per plant)
traits = pd.concat([
    pipeline.compute_plant_traits(series)
    for series in series_list
])
```

## Troubleshooting

**Plant count mismatch:**
- Verify SLEAP tracking parameters
- Check for merged or split tracks
- Ensure one track per plant

**Inconsistent lengths:**
- Confirm all plants imaged at same scale
- Verify pixel-to-mm calibration
- Check for tracking discontinuities

**Missing frames:**
- Ensure H5 file includes all expected frames
- Check for tracking gaps in SLEAP
- Verify frame alignment across timepoints

## Next Steps

- Compare with [Multiple Dicot Pipeline](multiple-dicot-pipeline.md) for plants with laterals
- See [Primary Root Pipeline](primary-root-pipeline.md) for single-plant analysis
- Read [Batch Processing](../guides/batch-processing.md) for large experiments
- Explore [Statistical Cookbook](../cookbook/batch-optimization.md) for analysis recipes