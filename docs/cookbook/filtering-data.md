# Filtering and Cleaning Data

This recipe shows how to filter and clean root tracking data before trait computation, ensuring high-quality results.

## Problem

SLEAP predictions may contain:
- Low-confidence tracks
- Incomplete root segments
- Tracking artifacts
- Outlier points

These issues can lead to incorrect trait values. Filtering data before analysis improves accuracy.

## Solution Overview

Apply filtering at multiple levels:
1. **Confidence thresholding**: Remove low-confidence predictions
2. **Point count filtering**: Exclude roots with too few points
3. **Outlier detection**: Remove anomalous measurements
4. **Temporal smoothing**: Reduce noise in time series

## Basic Confidence Filtering

### Filter During Loading

```python
import sleap_roots as sr

# Set confidence threshold when loading
series = sr.Series.load(
    "plant",
    h5_path="predictions.h5",
    primary_path="primary.slp",
    lateral_path="lateral.slp",
    confidence_threshold=0.3  # Default: 0.0 (keep all)
)

print(f"Loaded {len(series.primary_pts[0])} primary root points")
```

**Recommended thresholds**:
- `0.1-0.2`: Lenient (keep most tracks, some noise)
- `0.3-0.5`: Moderate (good balance)
- `0.5-0.7`: Strict (high quality, may lose data)

### Post-Loading Filtering

```python
import numpy as np

def filter_by_confidence(pts, confidence, threshold=0.3):
    """
    Filter points below confidence threshold.

    Args:
        pts: Root coordinates (n, 2)
        confidence: Confidence scores (n,)
        threshold: Minimum confidence

    Returns:
        Filtered points array
    """
    mask = confidence >= threshold
    return pts[mask]

# Apply to loaded data
confidence_scores = series.get_confidence_scores()
filtered_pts = filter_by_confidence(
    series.primary_pts[0],
    confidence_scores,
    threshold=0.4
)
```

## Point Count Filtering

### Filter Incomplete Roots

```python
def filter_short_roots(pts_list, min_points=10):
    """
    Remove roots with too few points.

    Args:
        pts_list: List of root coordinate arrays
        min_points: Minimum number of points required

    Returns:
        List of roots meeting criteria
    """
    return [pts for pts in pts_list if len(pts) >= min_points]

# Filter lateral roots
lateral_pts_list = series.lateral_pts[0]
valid_laterals = filter_short_roots(lateral_pts_list, min_points=15)

print(f"Kept {len(valid_laterals)}/{len(lateral_pts_list)} lateral roots")
```

### Frame-Level Filtering

```python
def filter_frames_by_point_count(series, min_points=20):
    """
    Keep only frames with sufficient tracking points.

    Args:
        series: Series object
        min_points: Minimum points for primary root

    Returns:
        List of valid frame indices
    """
    valid_frames = []
    for i, pts in enumerate(series.primary_pts):
        if len(pts) >= min_points:
            valid_frames.append(i)
    return valid_frames

# Get valid frames
valid_frames = filter_frames_by_point_count(series, min_points=30)

# Process only valid frames
for frame_idx in valid_frames:
    pts = series.primary_pts[frame_idx]
    # Compute traits for this frame
```

## Outlier Detection

### Length-Based Outliers

```python
from sleap_roots import lengths

def detect_length_outliers(lateral_pts_list, std_threshold=3.0):
    """
    Detect lateral roots with outlier lengths.

    Args:
        lateral_pts_list: List of lateral root arrays
        std_threshold: Number of standard deviations for outlier

    Returns:
        Boolean mask (True = keep, False = outlier)
    """
    root_lengths = lengths.get_root_lengths(lateral_pts_list)

    mean_length = np.mean(root_lengths)
    std_length = np.std(root_lengths)

    # Mark outliers (too short or too long)
    lower_bound = mean_length - std_threshold * std_length
    upper_bound = mean_length + std_threshold * std_length

    mask = (root_lengths >= lower_bound) & (root_lengths <= upper_bound)

    return mask

# Apply outlier detection
lateral_pts_list = series.lateral_pts[0]
outlier_mask = detect_length_outliers(lateral_pts_list, std_threshold=2.5)
clean_laterals = [pts for pts, keep in zip(lateral_pts_list, outlier_mask) if keep]

print(f"Removed {(~outlier_mask).sum()} outliers")
```

### Angle-Based Outliers

```python
from sleap_roots import angles

def detect_angle_outliers(lateral_pts_list, primary_pts, angle_range=(-90, 90)):
    """
    Detect lateral roots with unrealistic emergence angles.

    Args:
        lateral_pts_list: Lateral root arrays
        primary_pts: Primary root array
        angle_range: Valid angle range (min, max) in degrees

    Returns:
        Boolean mask for valid roots
    """
    mask = []

    for lateral_pts in lateral_pts_list:
        if len(lateral_pts) < 2:
            mask.append(False)
            continue

        angle = angles.get_lateral_root_angle(primary_pts, lateral_pts)

        # Check if angle is in valid range
        valid = angle_range[0] <= angle <= angle_range[1]
        mask.append(valid)

    return np.array(mask)

# Filter by angle
angle_mask = detect_angle_outliers(
    lateral_pts_list,
    series.primary_pts[0],
    angle_range=(-80, 80)  # Most laterals within ±80°
)
valid_laterals = [pts for pts, keep in zip(lateral_pts_list, angle_mask) if keep]
```

## Temporal Smoothing

### Smooth Time Series

```python
from scipy.ndimage import gaussian_filter1d
import pandas as pd

def smooth_trait_timeseries(trait_values, sigma=2.0):
    """
    Apply Gaussian smoothing to trait time series.

    Args:
        trait_values: Array of trait values over time
        sigma: Smoothing parameter (higher = more smoothing)

    Returns:
        Smoothed trait values
    """
    return gaussian_filter1d(trait_values, sigma=sigma)

# Example: smooth primary root length over time
pipeline = sr.DicotPipeline()
traits = pipeline.compute_plant_traits(series)

# Smooth lengths
traits['primary_length_smooth'] = smooth_trait_timeseries(
    traits['primary_length'].values,
    sigma=3.0
)

# Plot comparison
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(traits['frame'], traits['primary_length'], 'o-', alpha=0.5, label='Raw')
plt.plot(traits['frame'], traits['primary_length_smooth'], '-', linewidth=2, label='Smoothed')
plt.xlabel('Frame')
plt.ylabel('Primary Root Length (pixels)')
plt.legend()
plt.title('Effect of Temporal Smoothing')
plt.show()
```

### Median Filtering

```python
from scipy.signal import medfilt

def median_filter_traits(trait_values, kernel_size=5):
    """
    Apply median filter to remove spikes.

    Args:
        trait_values: Trait time series
        kernel_size: Window size (must be odd)

    Returns:
        Filtered values
    """
    return medfilt(trait_values, kernel_size=kernel_size)

# Remove spikes from lateral count
traits['lateral_count_filtered'] = median_filter_traits(
    traits['lateral_count'].values,
    kernel_size=3
)
```

## Complete Filtering Pipeline

### Comprehensive Example

```python
import sleap_roots as sr
import numpy as np
from scipy.ndimage import gaussian_filter1d

class FilteredPipeline(sr.DicotPipeline):
    """DicotPipeline with integrated data filtering."""

    def __init__(
        self,
        confidence_threshold=0.3,
        min_primary_points=20,
        min_lateral_points=5,
        lateral_length_std=3.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.confidence_threshold = confidence_threshold
        self.min_primary_points = min_primary_points
        self.min_lateral_points = min_lateral_points
        self.lateral_length_std = lateral_length_std

    def compute_plant_traits(self, series, write_csv=False, csv_path=None):
        """Compute traits with filtering."""

        # Filter primary root
        primary_pts = series.get_primary_root_points()
        if len(primary_pts) < self.min_primary_points:
            print(f"Warning: Primary root has only {len(primary_pts)} points")
            # Handle insufficient data
            return pd.DataFrame()  # Return empty or NaN traits

        # Filter lateral roots
        lateral_pts_list = series.get_lateral_root_points()
        lateral_pts_list = self._filter_laterals(lateral_pts_list)

        # Create filtered series (optional: wrap in new Series object)
        # For now, modify in place

        # Compute traits normally
        traits = super().compute_plant_traits(series, write_csv=False)

        # Post-process: smooth temporal data if multiple frames
        if len(traits) > 1:
            traits = self._smooth_traits(traits)

        # Write to CSV if requested
        if write_csv:
            output_path = csv_path or f"{series.series_name}_filtered_traits.csv"
            traits.to_csv(output_path, index=False)

        return traits

    def _filter_laterals(self, lateral_pts_list):
        """Apply lateral root filters."""

        # 1. Point count filter
        valid_laterals = [
            pts for pts in lateral_pts_list
            if len(pts) >= self.min_lateral_points
        ]

        if len(valid_laterals) == 0:
            return []

        # 2. Length outlier filter
        lengths_array = sr.lengths.get_root_lengths(valid_laterals)
        mean_len = np.mean(lengths_array)
        std_len = np.std(lengths_array)

        outlier_mask = np.abs(lengths_array - mean_len) < (self.lateral_length_std * std_len)
        filtered_laterals = [
            pts for pts, keep in zip(valid_laterals, outlier_mask)
            if keep
        ]

        return filtered_laterals

    def _smooth_traits(self, traits):
        """Smooth traits across frames."""

        smoothable_traits = [
            'primary_length',
            'lateral_count',
            'total_length'
        ]

        for trait in smoothable_traits:
            if trait in traits.columns:
                traits[f'{trait}_smooth'] = gaussian_filter1d(
                    traits[trait].values,
                    sigma=2.0
                )

        return traits

# Usage
series = sr.Series.load(
    "plant",
    h5_path="predictions.h5",
    primary_path="primary.slp",
    lateral_path="lateral.slp",
    confidence_threshold=0.3  # Filter during load
)

filtered_pipeline = FilteredPipeline(
    confidence_threshold=0.3,
    min_primary_points=25,
    min_lateral_points=8,
    lateral_length_std=2.5
)

traits = filtered_pipeline.compute_plant_traits(series, write_csv=True)
print(f"Computed {len(traits)} frames of traits")
```

## Quality Metrics

### Assess Data Quality

```python
def compute_quality_metrics(series):
    """
    Compute metrics indicating data quality.

    Returns:
        Dictionary of quality metrics
    """
    metrics = {}

    # Primary root metrics
    primary_pts = series.primary_pts[0]
    metrics['primary_point_count'] = len(primary_pts)
    metrics['primary_completeness'] = len(primary_pts) / 200  # Assume 200 is ideal

    # Lateral root metrics
    lateral_pts_list = series.lateral_pts[0]
    metrics['lateral_count'] = len(lateral_pts_list)
    metrics['lateral_mean_points'] = np.mean([len(pts) for pts in lateral_pts_list])

    # Tracking continuity
    gaps = []
    for i in range(len(primary_pts) - 1):
        dist = np.linalg.norm(primary_pts[i+1] - primary_pts[i])
        gaps.append(dist)

    metrics['mean_gap'] = np.mean(gaps)
    metrics['max_gap'] = np.max(gaps)
    metrics['gap_std'] = np.std(gaps)

    # Quality score (0-1)
    quality_score = (
        min(metrics['primary_point_count'] / 200, 1.0) * 0.5 +
        min(metrics['lateral_count'] / 10, 1.0) * 0.3 +
        (1.0 - min(metrics['gap_std'] / 10, 1.0)) * 0.2
    )
    metrics['quality_score'] = quality_score

    return metrics

# Assess quality
quality = compute_quality_metrics(series)
print(f"Data quality score: {quality['quality_score']:.2f}")

if quality['quality_score'] < 0.5:
    print("Warning: Low data quality, consider re-tracking in SLEAP")
```

## Best Practices

### 1. Filter Early

Apply confidence thresholds during loading:
```python
# Good: filter during load
series = sr.Series.load(..., confidence_threshold=0.3)

# Less efficient: filter after loading
# (already loaded low-confidence points)
```

### 2. Document Filtering

Keep track of filtering parameters:
```python
metadata = {
    'confidence_threshold': 0.3,
    'min_primary_points': 20,
    'min_lateral_points': 5,
    'smoothing_sigma': 2.0,
    'date_filtered': '2024-01-15'
}

# Save with traits
traits['metadata'] = str(metadata)
traits.to_csv('traits_with_metadata.csv')
```

### 3. Validate Results

Check filtered vs. unfiltered:
```python
# Compare filtered vs. unfiltered
traits_raw = pipeline.compute_plant_traits(series_unfiltered)
traits_filtered = filtered_pipeline.compute_plant_traits(series)

comparison = pd.DataFrame({
    'trait': traits_raw.columns,
    'raw_mean': traits_raw.mean(),
    'filtered_mean': traits_filtered.mean(),
    'difference': traits_raw.mean() - traits_filtered.mean()
})

print(comparison)
```

## Troubleshooting

**Too much data removed**:
- Lower confidence threshold
- Reduce min_points requirements
- Relax outlier detection (higher std_threshold)

**Still have noisy data**:
- Increase confidence threshold
- Apply more aggressive outlier detection
- Use temporal smoothing
- Consider re-tracking in SLEAP with better model

**Temporal smoothing causes lag**:
- Reduce sigma parameter
- Use median filter instead of Gaussian
- Apply smoothing only to specific traits

## Next Steps

- See [Batch Processing](../guides/batch-processing.md) for filtering large datasets
- Read [Troubleshooting](../guides/troubleshooting.md) for common data issues
- Check [Custom Traits](custom-traits.md) for filtered trait computations