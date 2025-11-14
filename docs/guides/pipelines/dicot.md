# DicotPipeline

Pipeline for analyzing dicot plants with primary and lateral roots.

## Overview

**DicotPipeline** is designed for dicot species (e.g., soy, canola, arabidopsis) that have a single primary root with branching lateral roots.

**Root types required:**

- Primary root (1 instance)
- Lateral roots (multiple instances)

**Computed traits:** 40+ morphological measurements

## Quick Start

```python
import sleap_roots as sr

# Load SLEAP predictions
series = sr.Series.load(
    series_name="919QDUH",
    h5_path="919QDUH.h5",
    primary_path="919QDUH.primary.slp",
    lateral_path="919QDUH.lateral.slp"
)

# Create pipeline and compute traits
pipeline = sr.DicotPipeline()
traits = pipeline.compute_plant_traits(series, write_csv=True)

print(traits[['frame_idx', 'primary_length', 'lateral_count']].head())
```

## Trait Categories

### Length Traits

- `primary_length` – Total length of primary root
- `lateral_length_total` – Sum of all lateral root lengths
- `lateral_length_avg` – Average lateral root length
- `lateral_length_max` – Maximum lateral root length
- `lateral_length_min` – Minimum lateral root length

### Count Traits

- `lateral_count` – Number of lateral roots detected
- `primary_tip_count` – Number of primary root tips

### Angle Traits

- `primary_angle` – Angle of primary root from vertical
- `lateral_angle_avg` – Average emergence angle of laterals
- `lateral_angle_min` – Minimum lateral emergence angle
- `lateral_angle_max` – Maximum lateral emergence angle

### Topology Traits

- `convex_hull_area` – Area of convex hull around all roots
- `network_distribution_ratio` – Spatial distribution metric
- `network_solidity` – Compactness of root network

## Data Requirements

### SLEAP Skeleton

Primary root skeleton:
```
base → node1 → node2 → ... → tip
```

Lateral root skeleton (multiple instances):
```
base → node1 → node2 → ... → tip
```

### File Organization

Single time-point:
```
plant_name.primary.slp  # Primary root predictions
plant_name.lateral.slp  # Lateral root predictions
plant_name.h5           # Optional: video file
```

## Example Datasets

The test data includes dicot examples:

- **Canola (7 days old):** `tests/data/canola_7do/`
- **Soy (6 days old):** `tests/data/soy_6do/`

## Best Practices

- Ensure lateral roots are tracked separately from primary
- Label lateral root bases near primary root
- Use consistent skeleton structure across plants
- Verify SLEAP predictions before batch processing

## See Also

- [MultipleDicotPipeline](multiple-dicot.md) – For multi-plant setups
- [Trait Reference](../trait-reference.md) – Full trait descriptions
- [API Reference](../../reference/sleap_roots/trait_pipelines.md) – Detailed class documentation