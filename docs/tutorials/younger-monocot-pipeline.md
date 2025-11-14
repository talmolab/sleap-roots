# Younger Monocot Pipeline Tutorial

This tutorial demonstrates analysis of early-stage monocot root systems using the `YoungerMonocotPipeline`. This pipeline handles monocots with both primary and crown roots, typically seen in younger rice and maize plants.

## What You'll Learn

- Load SLEAP predictions for primary and crown roots
- Configure the YoungerMonocotPipeline
- Compute monocot-specific traits
- Visualize crown root emergence patterns
- Compare primary vs. crown root development

## Pipeline Overview

The `YoungerMonocotPipeline` computes traits for:

- **Primary root**: Single seminal root present at germination
- **Crown roots**: Adventitious roots emerging from stem base
- **Combined metrics**: Total root system characteristics

## Interactive Tutorial

{{ '../../notebooks/YoungerMonocotPipeline.ipynb' }}

## Key Differences from Dicot Pipeline

### Root System Architecture

Unlike dicots with lateral branching, younger monocots have:

- One primary (seminal) root
- Multiple crown roots emerging from the stem base
- No lateral roots (or minimal branching)

### Node Structure

Expected SLEAP skeleton:

```
Primary root: base → nodes → tip
Crown roots: base → nodes → tip (multiple instances)
```

### Trait Categories

Specific to monocots:

- Crown root count and emergence timing
- Primary vs. crown root length comparison
- Root system symmetry measures
- Crown root angle distribution

## Common Use Cases

### Rice Phenotyping

```python
import sleap_roots as sr

series = sr.Series.load(
    "rice_plant_day7",
    h5_path="rice.h5",
    primary_path="primary.slp",
    crown_path="crown.slp"  # Note: crown instead of lateral
)

pipeline = sr.YoungerMonocotPipeline()
traits = pipeline.compute_plant_traits(series, write_csv=True)

print(f"Crown root count: {traits['crown_count'].iloc[0]}")
print(f"Primary length: {traits['primary_length'].iloc[0]:.2f}")
```

### Maize Early Development

```python
# Track development over multiple timepoints
for day in range(3, 10):
    series = sr.Series.load(
        f"maize_plant_day{day}",
        h5_path=f"maize_day{day}.h5",
        primary_path=f"primary_day{day}.slp",
        crown_path=f"crown_day{day}.slp"
    )
    traits = pipeline.compute_plant_traits(series)
    # Analyze temporal patterns
```

## Key Traits

| Trait | Description | Typical Values |
|-------|-------------|----------------|
| `crown_count` | Number of crown roots | 3-8 (early stage) |
| `primary_length` | Primary root length | Longer than crown initially |
| `crown_root_lengths` | Individual crown lengths | Variable, often shorter than primary |
| `root_system_width` | Lateral spread of crown roots | Increases with age |

## Developmental Stages

### Early Stage (Days 3-5)
- Primary root dominant
- 2-4 crown roots emerging
- Crown roots shorter than primary

### Mid Stage (Days 6-8)
- Crown roots elongating rapidly
- 4-6 crown roots present
- Crown roots approaching primary length

### Late Young Stage (Days 9-12)
- Crown roots may exceed primary length
- 6-10 crown roots typical
- Consider switching to `OlderMonocotPipeline` if primary root degrades

## Troubleshooting

**Crown roots not detected:**
- Verify SLEAP model detects all root bases
- Check tracking quality in crown root file
- Ensure sufficient image contrast at stem base

**Primary root confused with crown:**
- Use consistent node naming in SLEAP skeleton
- Primary should be tracked from seed location
- Crown roots should originate from stem base (above seed)

## Next Steps

- See [Older Monocot Pipeline](older-monocot-pipeline.md) for mature plants
- Compare with [Dicot Pipeline](dicot-pipeline.md) architecture differences
- Read [Trait Reference](../guides/trait-reference.md) for monocot-specific traits