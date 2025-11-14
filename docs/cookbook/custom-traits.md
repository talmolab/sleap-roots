# Creating Custom Traits

This recipe demonstrates how to implement custom trait computations for specialized research questions.

## Problem

Pre-built pipelines may not include the specific traits you need for your research. You want to:

- Compute novel morphological metrics
- Combine existing traits in custom ways
- Implement domain-specific calculations
- Add experimental trait definitions

## Solution Overview

Create custom trait functions and integrate them into your analysis workflow.

## Simple Custom Trait

### Root Tortuosity Example

Let's implement a root tortuosity metric:

```python
import numpy as np
from sleap_roots import lengths

def compute_tortuosity(pts):
    """
    Compute root tortuosity (path length / Euclidean distance).

    Higher values indicate more curved/winding roots.

    Args:
        pts: Root coordinates (n, 2)

    Returns:
        Tortuosity value (≥ 1.0)
    """
    if len(pts) < 2:
        return np.nan

    # Path length
    path_length = lengths.get_root_lengths([pts])[0]

    # Euclidean distance
    euclidean_dist = np.linalg.norm(pts[-1] - pts[0])

    if euclidean_dist == 0:
        return np.inf

    return path_length / euclidean_dist

# Use it
import sleap_roots as sr

series = sr.Series.load(...)
primary_pts = series.primary_pts[0]

tortuosity = compute_tortuosity(primary_pts)
print(f"Root tortuosity: {tortuosity:.2f}")
```

### Add to Pipeline

```python
class ExtendedDicotPipeline(sr.DicotPipeline):
    """DicotPipeline with custom tortuosity trait."""

    def compute_plant_traits(self, series, write_csv=False, csv_path=None):
        """Compute standard + custom traits."""

        # Get standard traits
        traits = super().compute_plant_traits(series, write_csv=False)

        # Add custom trait
        primary_pts = series.get_primary_root_points()
        traits['primary_tortuosity'] = compute_tortuosity(primary_pts)

        # Add for laterals too
        lateral_pts_list = series.get_lateral_root_points()
        if len(lateral_pts_list) > 0:
            lateral_tortuosities = [compute_tortuosity(pts) for pts in lateral_pts_list]
            traits['mean_lateral_tortuosity'] = np.mean(lateral_tortuosities)

        # Write to CSV if requested
        if write_csv:
            output_path = csv_path or f"{series.series_name}_extended_traits.csv"
            traits.to_csv(output_path, index=False)

        return traits
```

## Complex Custom Trait

### Root Branching Density

Compute lateral root density along primary root:

```python
import numpy as np
from scipy.ndimage import gaussian_filter1d

def compute_branching_density(primary_pts, lateral_pts_list, window_size=50):
    """
    Compute spatial density of lateral root emergence along primary.

    Args:
        primary_pts: Primary root coordinates (n, 2)
        lateral_pts_list: List of lateral root arrays
        window_size: Spatial window for density calculation (pixels)

    Returns:
        Dictionary with density metrics
    """
    if len(lateral_pts_list) == 0:
        return {
            'branching_density': 0.0,
            'density_profile': np.array([]),
            'peak_density': 0.0,
            'peak_position': np.nan
        }

    # Get lateral base positions along primary axis
    # (Assuming first point of each lateral is base)
    lateral_bases = np.array([pts[0] for pts in lateral_pts_list])

    # Project lateral bases onto primary root axis
    # Simple approach: use distance from primary base
    primary_base = primary_pts[0]
    distances_from_base = np.linalg.norm(
        lateral_bases - primary_base,
        axis=1
    )

    # Create density histogram
    primary_length = lengths.get_root_lengths([primary_pts])[0]
    bins = np.arange(0, primary_length, window_size)

    hist, bin_edges = np.histogram(distances_from_base, bins=bins)

    # Smooth density profile
    density_profile = gaussian_filter1d(hist.astype(float), sigma=2.0)

    # Overall density
    overall_density = len(lateral_pts_list) / primary_length

    # Peak density and position
    peak_density = np.max(density_profile)
    peak_bin = np.argmax(density_profile)
    peak_position = bin_edges[peak_bin]

    return {
        'branching_density': overall_density,
        'density_profile': density_profile,
        'peak_density': peak_density,
        'peak_position': peak_position,
        'bin_edges': bin_edges[:-1]  # For plotting
    }

# Example usage
primary_pts = series.primary_pts[0]
lateral_pts_list = series.lateral_pts[0]

density_metrics = compute_branching_density(
    primary_pts,
    lateral_pts_list,
    window_size=30
)

print(f"Overall branching density: {density_metrics['branching_density']:.4f} laterals/pixel")
print(f"Peak density at: {density_metrics['peak_position']:.1f} pixels from base")

# Visualize
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.bar(
    density_metrics['bin_edges'],
    density_metrics['density_profile'],
    width=30,
    edgecolor='black'
)
plt.xlabel('Distance from Primary Base (pixels)')
plt.ylabel('Lateral Density')
plt.title('Lateral Root Branching Density Profile')
plt.show()
```

## Multi-Root Custom Trait

### Root System Asymmetry

Measure left-right asymmetry of lateral roots:

```python
from sleap_roots import angles

def compute_lateral_asymmetry(primary_pts, lateral_pts_list):
    """
    Compute left-right asymmetry of lateral root distribution.

    Args:
        primary_pts: Primary root coordinates
        lateral_pts_list: Lateral root arrays

    Returns:
        Dictionary with asymmetry metrics
    """
    if len(lateral_pts_list) == 0:
        return {
            'asymmetry_index': 0.0,
            'left_count': 0,
            'right_count': 0,
            'left_total_length': 0.0,
            'right_total_length': 0.0
        }

    # Get emergence angles
    emergence_angles = []
    for lateral_pts in lateral_pts_list:
        angle = angles.get_lateral_emergence_angle(primary_pts, lateral_pts)
        emergence_angles.append(angle)

    emergence_angles = np.array(emergence_angles)

    # Classify left vs. right (negative = left, positive = right)
    left_mask = emergence_angles < 0
    right_mask = emergence_angles > 0

    left_count = left_mask.sum()
    right_count = right_mask.sum()

    # Compute lengths
    all_lengths = lengths.get_root_lengths(lateral_pts_list)
    left_total_length = all_lengths[left_mask].sum() if left_count > 0 else 0.0
    right_total_length = all_lengths[right_mask].sum() if right_count > 0 else 0.0

    # Asymmetry indices
    count_asymmetry = abs(left_count - right_count) / (left_count + right_count)
    total_length = left_total_length + right_total_length
    length_asymmetry = abs(left_total_length - right_total_length) / total_length if total_length > 0 else 0.0

    # Combined asymmetry (0 = symmetric, 1 = fully asymmetric)
    asymmetry_index = (count_asymmetry + length_asymmetry) / 2

    return {
        'asymmetry_index': asymmetry_index,
        'count_asymmetry': count_asymmetry,
        'length_asymmetry': length_asymmetry,
        'left_count': int(left_count),
        'right_count': int(right_count),
        'left_total_length': float(left_total_length),
        'right_total_length': float(right_total_length)
    }

# Usage
asymmetry = compute_lateral_asymmetry(primary_pts, lateral_pts_list)
print(f"Asymmetry index: {asymmetry['asymmetry_index']:.2f}")
print(f"Left: {asymmetry['left_count']} roots, {asymmetry['left_total_length']:.1f} px")
print(f"Right: {asymmetry['right_count']} roots, {asymmetry['right_total_length']:.1f} px")
```

## Temporal Custom Trait

### Growth Rate Analysis

```python
def compute_growth_rates(trait_series, time_units='frame'):
    """
    Compute instantaneous and cumulative growth rates.

    Args:
        trait_series: Pandas Series of trait values over time
        time_units: Time unit for rates ('frame', 'hour', etc.)

    Returns:
        DataFrame with growth metrics
    """
    import pandas as pd

    results = pd.DataFrame({
        'value': trait_series.values,
        'instantaneous_rate': trait_series.diff(),
        'cumulative_growth': trait_series - trait_series.iloc[0],
        'relative_growth': trait_series.pct_change()
    })

    # Moving average growth rate
    results['avg_rate_5frame'] = results['instantaneous_rate'].rolling(5, center=True).mean()

    return results

# Example with temporal data
series = sr.Series.load(...)  # Multi-frame series
pipeline = sr.DicotPipeline()
traits = pipeline.compute_plant_traits(series)

# Compute growth rates for primary length
growth_rates = compute_growth_rates(traits['primary_length'])

# Visualize
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(growth_rates['value'])
plt.title('Primary Root Length')
plt.xlabel('Frame')

plt.subplot(1, 3, 2)
plt.plot(growth_rates['instantaneous_rate'], 'o-', alpha=0.5, label='Instantaneous')
plt.plot(growth_rates['avg_rate_5frame'], linewidth=2, label='5-frame avg')
plt.title('Growth Rate')
plt.xlabel('Frame')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(growth_rates['cumulative_growth'])
plt.title('Cumulative Growth')
plt.xlabel('Frame')

plt.tight_layout()
plt.show()
```

## Statistical Custom Traits

### Root Length Distribution Metrics

```python
from scipy import stats

def compute_length_distribution_metrics(lateral_pts_list):
    """
    Compute statistical metrics of lateral root length distribution.

    Args:
        lateral_pts_list: List of lateral root arrays

    Returns:
        Dictionary of distribution metrics
    """
    if len(lateral_pts_list) == 0:
        return {
            'mean': np.nan,
            'median': np.nan,
            'std': np.nan,
            'cv': np.nan,
            'skewness': np.nan,
            'kurtosis': np.nan,
            'iqr': np.nan
        }

    root_lengths = lengths.get_root_lengths(lateral_pts_list)

    metrics = {
        'mean': np.mean(root_lengths),
        'median': np.median(root_lengths),
        'std': np.std(root_lengths),
        'cv': np.std(root_lengths) / np.mean(root_lengths),  # Coefficient of variation
        'skewness': stats.skew(root_lengths),
        'kurtosis': stats.kurtosis(root_lengths),
        'iqr': stats.iqr(root_lengths),
        'range': np.ptp(root_lengths),  # Peak-to-peak (max - min)
        'q25': np.percentile(root_lengths, 25),
        'q75': np.percentile(root_lengths, 75)
    }

    return metrics

# Usage
dist_metrics = compute_length_distribution_metrics(lateral_pts_list)
print(f"Mean ± SD: {dist_metrics['mean']:.1f} ± {dist_metrics['std']:.1f}")
print(f"Coefficient of variation: {dist_metrics['cv']:.2f}")
print(f"Skewness: {dist_metrics['skewness']:.2f}")
```

## Geometry-Based Custom Traits

### Root Angle Distribution

```python
def compute_angle_distribution(primary_pts, window_size=10):
    """
    Compute distribution of local angles along root.

    Args:
        primary_pts: Root coordinates
        window_size: Window for local angle calculation

    Returns:
        Dictionary with angle distribution metrics
    """
    local_angles = []

    for i in range(window_size, len(primary_pts) - window_size):
        # Local tangent vector
        before = primary_pts[i] - primary_pts[i-window_size]
        after = primary_pts[i+window_size] - primary_pts[i]

        # Angle between segments
        angle = np.arctan2(
            before[0]*after[1] - before[1]*after[0],
            before[0]*after[0] + before[1]*after[1]
        )
        local_angles.append(np.degrees(angle))

    local_angles = np.array(local_angles)

    return {
        'mean_curvature': np.mean(np.abs(local_angles)),
        'max_curvature': np.max(np.abs(local_angles)),
        'curvature_variance': np.var(local_angles),
        'direction_changes': np.sum(np.abs(np.diff(np.sign(local_angles))) > 0)
    }

# Usage
angle_metrics = compute_angle_distribution(primary_pts, window_size=5)
print(f"Mean curvature: {angle_metrics['mean_curvature']:.2f}°")
print(f"Direction changes: {angle_metrics['direction_changes']}")
```

## Complete Custom Pipeline Example

### Research-Specific Pipeline

```python
import sleap_roots as sr
import pandas as pd
import numpy as np
from scipy import stats

class ResearchPipeline:
    """
    Custom pipeline with domain-specific traits.

    Example: Rice root system architecture study.
    """

    def __init__(self):
        self.name = "Rice RSA Pipeline"

    def compute_plant_traits(self, series, write_csv=False, csv_path=None):
        """Compute custom research traits."""

        traits = {}

        # Get root data
        primary_pts = series.get_primary_root_points()
        crown_pts_list = series.get_crown_root_points()  # For monocots

        # === Primary root traits ===
        traits.update(self._compute_primary_traits(primary_pts))

        # === Crown root traits ===
        traits.update(self._compute_crown_traits(crown_pts_list))

        # === System-level traits ===
        traits.update(self._compute_system_traits(primary_pts, crown_pts_list))

        # === Custom research traits ===
        traits.update(self._compute_research_specific_traits(primary_pts, crown_pts_list))

        df = pd.DataFrame([traits])

        if write_csv:
            output_path = csv_path or f"{series.series_name}_research_traits.csv"
            df.to_csv(output_path, index=False)

        return df

    def _compute_primary_traits(self, pts):
        """Basic primary root traits."""
        return {
            'primary_length': lengths.get_root_lengths([pts])[0],
            'primary_tortuosity': compute_tortuosity(pts)
        }

    def _compute_crown_traits(self, crown_pts_list):
        """Crown root traits."""
        if len(crown_pts_list) == 0:
            return {
                'crown_count': 0,
                'crown_total_length': 0.0,
                'crown_length_cv': np.nan
            }

        crown_lengths = lengths.get_root_lengths(crown_pts_list)

        return {
            'crown_count': len(crown_pts_list),
            'crown_total_length': np.sum(crown_lengths),
            'crown_mean_length': np.mean(crown_lengths),
            'crown_length_cv': np.std(crown_lengths) / np.mean(crown_lengths)
        }

    def _compute_system_traits(self, primary_pts, crown_pts_list):
        """Whole system traits."""
        all_roots = [primary_pts] + crown_pts_list
        all_lengths = lengths.get_root_lengths(all_roots)

        return {
            'total_root_length': np.sum(all_lengths),
            'total_root_count': len(all_roots),
            'root_length_gini': self._compute_gini_coefficient(all_lengths)
        }

    def _compute_research_specific_traits(self, primary_pts, crown_pts_list):
        """Domain-specific trait calculations."""

        # Example: Rice-specific traits
        traits = {}

        # Crown-to-primary ratio
        primary_len = lengths.get_root_lengths([primary_pts])[0]
        crown_total = np.sum(lengths.get_root_lengths(crown_pts_list)) if crown_pts_list else 0
        traits['crown_primary_ratio'] = crown_total / primary_len if primary_len > 0 else 0

        # Root depth efficiency (length per vertical depth)
        depth = np.max(primary_pts[:, 1]) - np.min(primary_pts[:, 1])
        traits['depth_efficiency'] = primary_len / depth if depth > 0 else 0

        # Shallow root fraction (crown roots in top 30% of depth)
        if len(crown_pts_list) > 0:
            shallow_count = 0
            depth_threshold = 0.3 * depth

            for crown_pts in crown_pts_list:
                crown_depth = crown_pts[:, 1].max() - crown_pts[:, 1].min()
                if crown_depth < depth_threshold:
                    shallow_count += 1

            traits['shallow_root_fraction'] = shallow_count / len(crown_pts_list)
        else:
            traits['shallow_root_fraction'] = 0.0

        return traits

    def _compute_gini_coefficient(self, values):
        """Compute Gini coefficient of length distribution."""
        values = np.array(values)
        if len(values) == 0:
            return np.nan

        sorted_values = np.sort(values)
        n = len(values)
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * sorted_values)) / (n * np.sum(sorted_values)) - (n + 1) / n


# Usage
research_pipeline = ResearchPipeline()
traits = research_pipeline.compute_plant_traits(series, write_csv=True)
print(traits)
```

## Testing Custom Traits

### Unit Tests

```python
import pytest

def test_tortuosity_straight_line():
    """Straight line should have tortuosity of 1.0."""
    pts = np.array([[0, 0], [1, 0], [2, 0]])
    tortuosity = compute_tortuosity(pts)
    assert np.isclose(tortuosity, 1.0)

def test_tortuosity_curved_path():
    """Curved path should have tortuosity > 1.0."""
    pts = np.array([[0, 0], [1, 1], [2, 0]])
    tortuosity = compute_tortuosity(pts)
    assert tortuosity > 1.0

def test_tortuosity_invalid_input():
    """Should handle edge cases."""
    # Single point
    pts = np.array([[0, 0]])
    assert np.isnan(compute_tortuosity(pts))

    # Coincident endpoints
    pts = np.array([[0, 0], [1, 1], [0, 0]])
    assert np.isinf(compute_tortuosity(pts))
```

## Best Practices

### 1. Validate Trait

- Test with known inputs
- Check edge cases
- Verify biological relevance

### 2. Document Thoroughly

```python
def my_custom_trait(pts):
    """
    One-line description.

    Detailed explanation of what the trait measures,
    biological interpretation, and any caveats.

    Args:
        pts: Root coordinates (n, 2)

    Returns:
        Trait value with units and expected range

    Example:
        >>> pts = np.array([[0, 0], [1, 1]])
        >>> my_custom_trait(pts)
        1.414

    References:
        Smith et al. (2020). Journal of Plant Science.
    """
    pass
```

### 3. Make Reusable

Package custom traits as a module:

```python
# my_custom_traits.py
import numpy as np

def tortuosity(pts):
    """..."""
    pass

def branching_density(primary, laterals, window=50):
    """..."""
    pass

# Use across projects
from my_custom_traits import tortuosity, branching_density
```

## Next Steps

- See [Creating Pipelines](../guides/custom-pipelines.md) for full pipeline development
- Read [Adding Traits](../dev/adding-traits.md) to contribute traits to sleap-roots
- Check [API Reference](../api/index.md) for available utility functions