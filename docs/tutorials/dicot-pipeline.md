# Dicot Pipeline Tutorial

This tutorial walks through a complete analysis of dicot root systems using the `DicotPipeline`. This pipeline is designed for plants with a primary root and lateral roots, such as soybean, canola, and arabidopsis.

## What You'll Learn

- Load SLEAP predictions for primary and lateral roots
- Initialize and configure the DicotPipeline
- Compute 50+ morphological traits
- Visualize root networks and convex hulls
- Export results to CSV

## Pipeline Overview

The `DicotPipeline` computes traits in several categories:

- **Primary root traits**: Length, growth, tip angle
- **Lateral root traits**: Count, lengths, emergence angles
- **Network traits**: Total lengths, medoid calculations
- **Topological traits**: Root base counts, convex hull area
- **Temporal traits**: Growth rates over time

## Interactive Tutorial

{{ '../../notebooks/DicotPipeline.ipynb' }}

## Key Concepts

### Root Network Structure

The pipeline expects SLEAP predictions with two files:

1. **Primary root** (`.slp`): Single tracked root with nodes from base to tip
2. **Lateral roots** (`.slp`): Multiple roots, each tracked from primary root junction to tip

### Trait Computation Flow

```mermaid
graph LR
    A[Load SLEAP Files] --> B[Series Object]
    B --> C[Get Root Landmarks]
    C --> D[Compute Primary Traits]
    C --> E[Compute Lateral Traits]
    D --> F[Compute Network Traits]
    E --> F
    F --> G[Export to CSV]
```

### Important Parameters

- **primary_name**: Node name for primary root base (default: `"base"`)
- **lateral_name**: Node name for lateral root bases (default: `"lateral"`)
- **csv_path**: Output path for trait CSV
- **plot**: Whether to generate visualizations

## Common Patterns

### Basic Usage

```python
import sleap_roots as sr

# Load data
series = sr.Series.load("plant1", h5_path="pred.h5",
                        primary_path="primary.slp",
                        lateral_path="lateral.slp")

# Compute traits
pipeline = sr.DicotPipeline()
traits = pipeline.compute_plant_traits(series, write_csv=True)
```

### Batch Processing

```python
from pathlib import Path

for h5_file in Path("data/").glob("*.h5"):
    series = sr.Series.load(
        series_name=h5_file.stem,
        h5_path=h5_file,
        primary_path=h5_file.with_suffix(".primary.slp"),
        lateral_path=h5_file.with_suffix(".lateral.slp")
    )
    traits = pipeline.compute_plant_traits(series, write_csv=True)
```

## Trait Highlights

Key traits computed by this pipeline:

| Trait | Description | Units |
|-------|-------------|-------|
| `primary_length` | Length of primary root from base to tip | pixels |
| `lateral_count` | Number of lateral roots detected | count |
| `total_length` | Sum of all root lengths | pixels |
| `primary_tip_angle` | Angle of primary root tip relative to vertical | degrees |
| `convex_hull_area` | Area of convex hull around all roots | pixels² |

See the [Trait Reference](../guides/trait-reference.md) for all 50+ traits.

## Next Steps

- Try the [Batch Processing](../guides/batch-processing.md) guide
- Learn about [Custom Pipelines](../guides/custom-pipelines.md)
- Explore other tutorials:
    - [Younger Monocot Pipeline](younger-monocot-pipeline.md)
    - [Multiple Dicot Pipeline](multiple-dicot-pipeline.md)