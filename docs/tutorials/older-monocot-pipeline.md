# Older Monocot Pipeline Tutorial

This tutorial covers analysis of mature monocot root systems using the `OlderMonocotPipeline`. This pipeline is designed for plants where the primary root has senesced or is no longer the dominant root, and the root system consists primarily of crown roots.

## What You'll Learn

- Analyze crown root-dominated monocot systems
- Handle root systems without primary roots
- Compute traits for mature rice and maize plants
- Measure root system spread and architecture

## Pipeline Overview

The `OlderMonocotPipeline` focuses exclusively on crown roots:

- **Crown root metrics**: Individual and aggregate lengths, angles
- **Root system architecture**: Spread, density, symmetry
- **Network properties**: Total system characteristics
- **Temporal dynamics**: Growth rates across timepoints

## Interactive Tutorial

{{ '../../notebooks/OlderMonocotPipeline.ipynb' }}

## When to Use This Pipeline

### Developmental Indicators

Use `OlderMonocotPipeline` when:

- Primary root is no longer visible or tracked
- Crown roots are the dominant root type
- Plant is beyond early seedling stage (typically >2 weeks)
- Root system is primarily adventitious

### Versus YoungerMonocotPipeline

| Feature | Younger Pipeline | Older Pipeline |
|---------|------------------|----------------|
| Primary root | Required | Not present |
| Crown roots | Emerging | Dominant |
| Typical age | Days 3-12 | Days 12+ |
| SLEAP files | 2 files (primary + crown) | 1 file (crown only) |

## Root System Architecture

### Crown Root Structure

Mature monocot root systems feature:

```
      Stem Base
     /  |  |  \
    /   |  |   \
   CR1 CR2 CR3 ... CRn
```

Where CR = Crown Root

### Typical Characteristics

- **Root count**: 6-15 crown roots (species-dependent)
- **Angular distribution**: Radially distributed around stem
- **Length variation**: Central roots often longer
- **Growth dynamics**: Continuous root initiation

## Common Workflows

### Basic Analysis

```python
import sleap_roots as sr

series = sr.Series.load(
    "mature_rice",
    h5_path="rice_day14.h5",
    crown_path="crown.slp"  # Only crown roots needed
)

pipeline = sr.OlderMonocotPipeline()
traits = pipeline.compute_plant_traits(series, write_csv=True)

print(f"Crown root count: {traits['crown_count'].iloc[0]}")
print(f"Total root length: {traits['total_crown_length'].iloc[0]:.2f}")
```

### Time Series Analysis

```python
import pandas as pd

results = []
for day in range(12, 21):  # Days 12-20
    series = sr.Series.load(
        f"plant_day{day}",
        h5_path=f"day{day}.h5",
        crown_path=f"crown_day{day}.slp"
    )
    traits = pipeline.compute_plant_traits(series)
    traits['day'] = day
    results.append(traits)

df = pd.concat(results)
# Analyze root system development over time
```

## Key Traits

### Root Count and Length

| Trait | Description | Units |
|-------|-------------|-------|
| `crown_count` | Number of crown roots | count |
| `crown_root_lengths` | Individual crown root lengths | pixels (array) |
| `total_crown_length` | Sum of all crown root lengths | pixels |
| `mean_crown_length` | Average crown root length | pixels |

### Spatial Distribution

| Trait | Description | Units |
|-------|-------------|-------|
| `root_system_width` | Maximum lateral spread | pixels |
| `root_system_depth` | Maximum vertical extent | pixels |
| `convex_hull_area` | Area encompassing all roots | pixels² |

### Angular Properties

| Trait | Description | Units |
|-------|-------------|-------|
| `crown_emergence_angles` | Angles of each crown root | degrees (array) |
| `angular_spread` | Standard deviation of angles | degrees |
| `root_symmetry_index` | Measure of radial symmetry | 0-1 |

## Biological Insights

### Root System Strategies

**Shallow vs. Deep:**
- Wide angular spread → Resource acquisition from broad area
- Narrow angular spread → Deep penetration strategy

**Root Count Variation:**
- More roots → Greater resource capture potential
- Fewer, longer roots → Efficient nutrient uptake

### Growth Patterns

Monitor these dynamics:
1. **Root initiation rate**: New crown roots per day
2. **Elongation rate**: Length increase per root
3. **System expansion**: Increase in convex hull area
4. **Architectural changes**: Shifts in angular distribution

## Troubleshooting

**Low root count detected:**
- Verify SLEAP tracking caught all crown roots
- Check for occlusion or poor image quality
- Ensure contrast is sufficient at root base region

**Irregular trait values:**
- Confirm all roots start from stem base (not mis-tracked laterals)
- Verify pixel-to-mm scaling if values seem off
- Check for tracking errors in SLEAP predictions

**Missing temporal data:**
- Ensure frame numbers align across timepoints
- Verify H5 file contains all expected frames
- Check for gaps in SLEAP tracking

## Next Steps

- Compare with [Younger Monocot Pipeline](younger-monocot-pipeline.md) for developmental transitions
- See [Multiple Primary Root Pipeline](multiple-primary-root-pipeline.md) for multi-plant setups
- Read [Batch Processing](../guides/batch-processing.md) for high-throughput analysis
- Explore [Trait Reference](../guides/trait-reference.md) for detailed trait definitions