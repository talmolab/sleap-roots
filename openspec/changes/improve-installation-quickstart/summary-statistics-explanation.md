# Understanding Summary Statistics in sleap-roots

## Two Levels of Summarization

sleap-roots applies summary statistics **twice** for different purposes:

### Level 1: Per-Frame Summarization (for non-scalar traits)
When a trait has multiple values **within a single frame** (e.g., multiple lateral roots), summary statistics are computed across those instances **within that frame**.

**Example: `lateral_lengths`**
- Frame 0 has 5 lateral roots with lengths: [3.9, 34.9, 62.8, 76.8, 80.3] pixels
- Per-frame statistics:
  - `lateral_lengths_min` = 3.9
  - `lateral_lengths_max` = 80.3
  - `lateral_lengths_mean` = 51.7
  - `lateral_lengths_median` = 62.8
  - `lateral_lengths_std` = 28.7
  - etc.

### Level 2: Per-Plant Summarization (across frames)
When using `compute_batch_traits()`, statistics are computed **across all frames** for each plant.

**Example: `lateral_lengths_mean` (already a per-frame statistic)**
- Plant has 72 frames, each with a `lateral_lengths_mean` value
- Per-plant statistics:
  - `lateral_lengths_mean_min` = minimum of all frame means
  - `lateral_lengths_mean_max` = maximum of all frame means
  - `lateral_lengths_mean_mean` = average of all frame means
  - `lateral_lengths_mean_median` = median of all frame means
  - etc.

This creates **double-suffixed** names like `lateral_lengths_mean_median`:
- **First suffix** (`_mean`): statistic across lateral roots within each frame
- **Second suffix** (`_median`): statistic across frames for the plant

## Which Traits Get Single vs Double Statistics?

### Scalar Traits (One Value Per Frame)
These have **one level** of statistics in batch processing:

- `primary_length` → `primary_length_mean`, `primary_length_std`, etc.
- `primary_angle_proximal` → `primary_angle_proximal_mean`, etc.
- `lateral_count` → `lateral_count_mean`, `lateral_count_median`, etc.
- `network_solidity` → `network_solidity_mean`, etc.
- `chull_area` → `chull_area_mean`, etc.

**Naming pattern:** `{trait_name}_{statistic}`

### Non-Scalar Traits (Multiple Values Per Frame)
These have **two levels** of statistics in batch processing:

- `lateral_lengths` → `lateral_lengths_mean_median`, `lateral_lengths_min_max`, etc.
- `lateral_angles_distal` → `lateral_angles_distal_mean_std`, etc.
- `root_widths` → `root_widths_median_mean`, etc.
- `lateral_base_xs` → `lateral_base_xs_min_p95`, etc.

**Naming pattern:** `{trait_name}_{frame_stat}_{plant_stat}`

## Complete List of Summary Statistics

Both levels compute these 9 statistics:
- `min` - Minimum value
- `max` - Maximum value
- `mean` - Average (arithmetic mean)
- `median` - Middle value (50th percentile)
- `std` - Standard deviation
- `p5` - 5th percentile
- `p25` - 25th percentile (Q1)
- `p75` - 75th percentile (Q3)
- `p95` - 95th percentile

## When to Use Which Statistics

### For Time-Series Analysis (Frame-Level Output)
Use `compute_plant_traits()` when you want to:
- Track how traits change over time
- Plot growth curves
- Analyze temporal dynamics
- Identify specific time points

**Best for:** Understanding development, growth rates, responses to treatments over time

### For Plant-Level Comparisons (Batch Summary Output)
Use `compute_batch_traits()` when you want to:
- Compare different plants/genotypes/treatments
- Statistical tests between groups
- High-level phenotyping summaries
- Reduce temporal variation to plant-level metrics

**Best for:** Screening experiments, GWAS, comparing varieties

### Which Summary Statistic to Use?

**Mean** - Use when:
- Data is normally distributed
- You want average behavior
- Sensitive to all values including outliers

**Median** - Use when:
- Data may have outliers
- You want typical/middle value
- More robust to extreme values

**Min/Max** - Use when:
- Extreme values are biologically meaningful
- Looking for maximum growth potential
- Identifying stress responses

**Std** - Use when:
- Variability is important
- Assessing consistency/stability
- Understanding phenotypic plasticity

**Percentiles (p5, p25, p75, p95)** - Use when:
- Want robust estimates that ignore outliers
- Describing distribution shape
- p25/p75 define interquartile range
- p5/p95 capture 90% of data range

### Common Use Cases

**Comparing growth between treatments:**
- Use `primary_length_mean` (average length across frames)
- Or `primary_length_median` (typical length, robust to outliers)

**Identifying most vigorous plants:**
- Use `primary_length_max` (maximum length achieved)
- Or `lateral_count_max` (peak lateral root count)

**Measuring lateral root angle variability:**
- Use `lateral_angles_proximal_std_mean` (average variability across frames)
- Higher = more variable emergence angles

**Finding consistent phenotypes:**
- Use `*_std` columns - lower std = more consistent across frames
- Example: `primary_angle_proximal_std` - low = straight growth

## Real Output Examples

### Frame-Level (compute_plant_traits)
72 frames × 117 columns:
```
plant_name  frame_idx  lateral_count  lateral_lengths_min  lateral_lengths_max  lateral_lengths_mean  ...
919QDUH     0          5              3.90                 80.27                40.40                 ...
919QDUH     1          3              36.70                62.28                48.36                 ...
919QDUH     2          4              39.55                82.77                66.39                 ...
```

### Plant-Level (compute_batch_traits)
1 plant × 1036 columns:
```
plant_name  lateral_count_mean  lateral_lengths_mean_mean  lateral_lengths_mean_median  ...
6PR6AA22JK  4.2                 45.3                       44.1                         ...
```

Notice:
- `lateral_count_mean` = single statistic (scalar trait)
- `lateral_lengths_mean_mean` = double statistic (non-scalar trait)
- `lateral_lengths_mean_median` = mean across roots, then median across frames