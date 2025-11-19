# Summary Statistics

## Overview

Compute comprehensive summary statistics for trait vectors. Essential for analyzing root trait distributions across multiple plants or time points.

**Key function**: `get_summary` - Compute 9 summary statistics (min, max, mean, median, std, percentiles)

## Quick Example

```python
import sleap_roots as sr
import numpy as np

series = sr.Series.load("plant", primary_path="primary.slp", lateral_path="lateral.slp")
primary_pts = series.get_primary_points()

# Compute lengths for all plants/frames
lengths = sr.get_root_lengths(primary_pts)  # (plants, frames)

# Flatten to get all length values
all_lengths = lengths.flatten()

# Get summary statistics
stats = sr.get_summary(all_lengths)
print(f"Mean length: {stats['mean']:.2f} px")
print(f"Median length: {stats['median']:.2f} px")
print(f"Std deviation: {stats['std']:.2f} px")
print(f"5th percentile: {stats['p5']:.2f} px")
print(f"95th percentile: {stats['p95']:.2f} px")
```

**Output**:
```
Mean length: 245.67 px
Median length: 238.50 px
Std deviation: 45.32 px
5th percentile: 178.23 px
95th percentile: 325.89 px
```

## Using Prefixes for Multiple Traits

```python
# Compute multiple traits
lengths = sr.get_root_lengths(primary_pts).flatten()
angles = sr.get_root_angle(primary_pts).flatten()

# Get summaries with prefixes
length_stats = sr.get_summary(lengths, prefix="length_")
angle_stats = sr.get_summary(angles, prefix="angle_")

# Combine into single dictionary
all_stats = {**length_stats, **angle_stats}

print(f"Length mean: {all_stats['length_mean']:.2f} px")
print(f"Angle mean: {all_stats['angle_mean']:.2f}°")
```

## Handling Edge Cases

```python
# Empty arrays
empty_stats = sr.get_summary(np.array([]))
print(empty_stats['mean'])  # nan

# Arrays with NaN values
with_nans = np.array([10.0, 20.0, np.nan, 30.0])
stats = sr.get_summary(with_nans)
print(f"Mean (ignoring NaN): {stats['mean']:.2f}")  # 20.00
```

## Pipeline Integration

```python
# Use with pipeline output
pipeline = sr.DicotPipeline()
traits = pipeline.fit_series(series)

# Get summary of lateral lengths
lateral_lengths = traits['lateral_length']  # Array of all lateral lengths
lateral_stats = sr.get_summary(lateral_lengths, prefix="lateral_length_")

print(f"Lateral length mean: {lateral_stats['lateral_length_mean']:.2f} px")
print(f"Lateral length range: {lateral_stats['lateral_length_min']:.2f} - {lateral_stats['lateral_length_max']:.2f} px")
```

## API Reference

### get_summary

::: sleap_roots.get_summary
    options:
      show_source: true

**Returns**:

A dictionary with the following keys (prefixed if `prefix` is specified):

| Key | Description |
|-----|-------------|
| `min` | Minimum value |
| `max` | Maximum value |
| `mean` | Mean (average) |
| `median` | Median (50th percentile) |
| `std` | Standard deviation |
| `p5` | 5th percentile |
| `p25` | 25th percentile (Q1) |
| `p75` | 75th percentile (Q3) |
| `p95` | 95th percentile |

**Notes**:
- Uses `np.nan*` functions to ignore NaN values
- Returns all NaN values if input is empty or all NaN
- Handles non-numeric arrays gracefully

---

## Related Modules

- **[Pipelines](../core/pipelines.md)** - Generate trait vectors for summarization
- **[All trait modules](../traits/lengths.md)** - Compute traits to summarize

## See Also

- **[Common Workflows](../examples/common-workflows.md)** - End-to-end analysis examples