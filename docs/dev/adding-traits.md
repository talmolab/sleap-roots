# Adding Traits

This guide explains how to add new trait computations to sleap-roots, from initial implementation to testing and documentation.

## Overview

Adding a new trait involves:

1. Implementing the trait computation function
2. Adding it to the appropriate module
3. Writing comprehensive tests
4. Documenting the trait
5. Integrating it into pipelines
6. Adding to trait reference

## Trait Computation Principles

### Design Philosophy

**Good traits are**:
- **Biologically meaningful**: Captures relevant morphological feature
- **Reproducible**: Same inputs always give same outputs
- **Well-documented**: Clear description, units, and formula
- **Tested**: Comprehensive unit and integration tests
- **Efficient**: Vectorized when possible, reasonable performance

**Avoid**:
- Overly complex computations without clear biological interpretation
- Traits that are just combinations of existing traits (do this at analysis time)
- Platform-specific or non-portable code

### Trait Categories

sleap-roots organizes traits into modules:

- `lengths.py`: Length-based measurements
- `angles.py`: Angular measurements
- `tips.py`: Root tip-related traits
- `bases.py`: Root base-related traits
- `convhull.py`: Convex hull and area metrics
- `networklength.py`: Network-level traits
- `monocots.py`: Monocot-specific traits
- `trait_pipelines.py`: Complete pipeline implementations

## Step-by-Step: Adding a New Trait

### Step 1: Implement the Function

Create your trait computation function in the appropriate module.

**Example**: Adding a "root sinuosity" trait to `lengths.py`

```python
# sleap_roots/lengths.py

import numpy as np
from typing import Union

def get_root_sinuosity(pts: np.ndarray) -> float:
    """
    Compute root sinuosity (tortuosity index).

    Sinuosity measures how much a root deviates from a straight line,
    defined as the ratio of path length to Euclidean distance.

    Args:
        pts: Array of shape (n, 2) containing root coordinates from
            base to tip.

    Returns:
        Sinuosity value. 1.0 indicates perfectly straight root,
        higher values indicate more curved/tortuous roots.

    Raises:
        ValueError: If pts has fewer than 2 points.

    Example:
        >>> pts = np.array([[0, 0], [1, 0], [2, 0]])  # Straight line
        >>> get_root_sinuosity(pts)
        1.0

        >>> pts = np.array([[0, 0], [1, 1], [2, 0]])  # Curved
        >>> get_root_sinuosity(pts)
        1.414...

    Note:
        - Value of 1.0 = perfectly straight
        - Higher values = more tortuous/sinuous
        - Sensitive to tracking noise; consider smoothing pts first
    """
    if len(pts) < 2:
        raise ValueError("Need at least 2 points to compute sinuosity")

    # Path length: sum of segment lengths
    segment_lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    path_length = np.sum(segment_lengths)

    # Euclidean distance: straight line from base to tip
    euclidean_distance = np.linalg.norm(pts[-1] - pts[0])

    # Handle edge case: coincident points
    if euclidean_distance == 0:
        return np.inf if path_length > 0 else 1.0

    return path_length / euclidean_distance
```

### Step 2: Write Tests

Create comprehensive tests in the corresponding test file.

```python
# tests/test_lengths.py

import numpy as np
import pytest
from sleap_roots import lengths

class TestRootSinuosity:
    """Tests for get_root_sinuosity function."""

    def test_straight_line(self):
        """Straight line should have sinuosity of 1.0."""
        pts = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
        sinuosity = lengths.get_root_sinuosity(pts)
        assert np.isclose(sinuosity, 1.0)

    def test_curved_path(self):
        """Curved path should have sinuosity > 1.0."""
        pts = np.array([[0, 0], [1, 1], [2, 0]])
        sinuosity = lengths.get_root_sinuosity(pts)
        assert sinuosity > 1.0

    def test_circular_path(self):
        """Circular/highly curved path should have high sinuosity."""
        # Create semicircle
        theta = np.linspace(0, np.pi, 50)
        pts = np.column_stack([np.cos(theta), np.sin(theta)])

        sinuosity = lengths.get_root_sinuosity(pts)
        expected = np.pi / 2  # Semicircle path length / diameter
        assert np.isclose(sinuosity, expected, rtol=0.1)

    def test_minimum_points(self):
        """Should work with exactly 2 points."""
        pts = np.array([[0, 0], [1, 1]])
        sinuosity = lengths.get_root_sinuosity(pts)
        assert np.isclose(sinuosity, 1.0)

    def test_insufficient_points(self):
        """Should raise error with < 2 points."""
        pts = np.array([[0, 0]])
        with pytest.raises(ValueError, match="at least 2 points"):
            lengths.get_root_sinuosity(pts)

    def test_coincident_points(self):
        """Should handle case where base == tip."""
        pts = np.array([[1, 1], [1.1, 1.05], [1, 1]])  # Returns to start
        sinuosity = lengths.get_root_sinuosity(pts)
        assert sinuosity == np.inf  # Path length > 0 but Euclidean = 0

    def test_single_point_repeated(self):
        """All same point should return 1.0."""
        pts = np.array([[5, 5], [5, 5], [5, 5]])
        sinuosity = lengths.get_root_sinuosity(pts)
        assert sinuosity == 1.0

    def test_numerical_precision(self):
        """Test with realistic noisy data."""
        # Create slightly noisy straight line
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y = np.random.normal(0, 0.1, 100)  # Small noise
        pts = np.column_stack([x, y])

        sinuosity = lengths.get_root_sinuosity(pts)
        assert 1.0 <= sinuosity <= 1.1  # Should be close to 1.0

    def test_different_scales(self):
        """Sinuosity should be scale-invariant."""
        pts_small = np.array([[0, 0], [1, 1], [2, 0]])
        pts_large = pts_small * 100

        sin_small = lengths.get_root_sinuosity(pts_small)
        sin_large = lengths.get_root_sinuosity(pts_large)

        assert np.isclose(sin_small, sin_large)

    def test_real_world_data(self, test_data_fixture):
        """Test with actual SLEAP tracking data."""
        series = test_data_fixture
        pts = series.primary_pts[0]

        sinuosity = lengths.get_root_sinuosity(pts)

        # Sanity checks for real data
        assert 1.0 <= sinuosity <= 5.0  # Reasonable range
        assert not np.isnan(sinuosity)
        assert not np.isinf(sinuosity)
```

### Step 3: Document the Trait

Add the trait to the trait reference documentation.

```markdown
<!-- docs/guides/trait-reference.md -->

#### Root Sinuosity

**Module**: `sleap_roots.lengths.get_root_sinuosity`

**Description**: Measures root tortuosity as the ratio of actual path length
to straight-line distance from base to tip.

**Formula**:
$$
\\text{sinuosity} = \\frac{\\text{path length}}{\\text{Euclidean distance}}
$$

**Units**: Dimensionless ratio

**Range**: [1.0, ∞)
- 1.0 = perfectly straight root
- 1.5 = moderately curved
- 2.0+ = highly tortuous

**Biological Relevance**: Sinuosity reflects root growth patterns and
environmental responses. Higher sinuosity may indicate:
- Obstacle avoidance behavior
- Resource seeking (e.g., following nutrient patches)
- Response to soil compaction or impedance
- Gravitropic or thigmotropic responses

**Interpretation**:
- **Low sinuosity (1.0-1.2)**: Direct, efficient growth path
- **Medium sinuosity (1.2-1.8)**: Moderate curvature, normal exploration
- **High sinuosity (>1.8)**: Highly curved, possibly stress response

**Limitations**:
- Sensitive to tracking noise (consider smoothing)
- May be affected by frame rate/sampling density
- Less meaningful for very short roots

**Related Traits**:
- `primary_length`: Path length component
- `primary_euclidean_length`: Straight-line distance component

**Example**:
```python
import sleap_roots as sr

series = sr.Series.load(...)
pts = series.primary_pts[0]
sinuosity = sr.lengths.get_root_sinuosity(pts)
print(f"Root sinuosity: {sinuosity:.2f}")
```
```

### Step 4: Integrate into Pipelines

Add the new trait to relevant pipelines.

```python
# sleap_roots/trait_pipelines.py

class DicotPipeline:
    """Pipeline for dicot root systems."""

    def compute_plant_traits(self, series, write_csv=False, csv_path=None):
        """Compute all dicot traits including new sinuosity trait."""

        traits = {}

        # Get root points
        primary_pts = series.get_primary_root_points()

        # Existing traits
        traits['primary_length'] = lengths.get_root_lengths([primary_pts])[0]
        # ... other traits ...

        # NEW TRAIT: Add sinuosity
        traits['primary_sinuosity'] = lengths.get_root_sinuosity(primary_pts)

        # ... rest of pipeline ...

        return pd.DataFrame([traits])
```

### Step 5: Add to Trait Definitions

Update trait metadata for documentation generation.

```python
# sleap_roots/trait_definitions.py (if exists)

TRAIT_DEFINITIONS = {
    'primary_sinuosity': {
        'name': 'Primary Root Sinuosity',
        'description': 'Ratio of root path length to Euclidean distance',
        'units': 'dimensionless',
        'range': '[1.0, inf)',
        'formula': 'path_length / euclidean_distance',
        'module': 'sleap_roots.lengths',
        'function': 'get_root_sinuosity',
        'category': 'morphology',
        'biological_relevance': 'Indicates root tortuosity and growth pattern'
    }
}
```

## Complex Example: Multi-Value Trait

Some traits return arrays or multiple values. Here's how to handle them:

```python
# sleap_roots/angles.py

def get_lateral_emergence_angles(
    primary_pts: np.ndarray,
    lateral_pts_list: list[np.ndarray]
) -> tuple[np.ndarray, float, float]:
    """
    Compute emergence angles for all lateral roots.

    Args:
        primary_pts: Primary root coordinates (n, 2)
        lateral_pts_list: List of lateral root coordinate arrays

    Returns:
        Tuple of:
        - angles: Array of emergence angles for each lateral (degrees)
        - mean_angle: Mean emergence angle (degrees)
        - angle_std: Standard deviation of angles (degrees)

    Example:
        >>> angles, mean, std = get_lateral_emergence_angles(primary, laterals)
        >>> print(f"Mean emergence angle: {mean:.1f}° ± {std:.1f}°")
    """
    angles = []

    for lateral_pts in lateral_pts_list:
        if len(lateral_pts) < 2:
            continue

        # Get base and direction points
        base_pt = lateral_pts[0]
        direction_pt = lateral_pts[min(5, len(lateral_pts)-1)]

        # Compute angle relative to primary axis
        lateral_vector = direction_pt - base_pt
        primary_vector = primary_pts[-1] - primary_pts[0]

        angle = compute_angle_between_vectors(lateral_vector, primary_vector)
        angles.append(angle)

    angles = np.array(angles)
    mean_angle = np.mean(angles) if len(angles) > 0 else np.nan
    angle_std = np.std(angles) if len(angles) > 0 else np.nan

    return angles, mean_angle, angle_std
```

**Using in pipeline**:

```python
# In pipeline
angles, mean_angle, angle_std = angles.get_lateral_emergence_angles(
    primary_pts, lateral_pts_list
)

traits['lateral_emergence_angles'] = angles  # Store array
traits['mean_emergence_angle'] = mean_angle  # Store scalar
traits['emergence_angle_std'] = angle_std    # Store scalar
```

## Testing Strategy

### Unit Tests

Test individual functions in isolation:

```python
def test_basic_functionality():
    """Test with simple, known input."""
    pts = np.array([[0, 0], [1, 0]])
    result = my_trait_function(pts)
    assert result == expected_value

def test_edge_cases():
    """Test edge cases."""
    # Empty input
    # Single point
    # Coincident points
    # Very large/small values

def test_error_handling():
    """Test that errors are raised appropriately."""
    with pytest.raises(ValueError):
        my_trait_function(invalid_input)
```

### Integration Tests

Test traits within pipelines:

```python
def test_trait_in_pipeline(test_series):
    """Test new trait integrates correctly in pipeline."""
    pipeline = sr.DicotPipeline()
    traits = pipeline.compute_plant_traits(test_series)

    # Verify trait is computed
    assert 'my_new_trait' in traits.columns

    # Verify trait value is reasonable
    value = traits['my_new_trait'].iloc[0]
    assert not np.isnan(value)
    assert 0 <= value <= 100  # Expected range
```

### Performance Tests

For computationally intensive traits:

```python
import time

def test_performance():
    """Ensure trait computation is reasonably fast."""
    # Large but realistic input
    pts = np.random.rand(1000, 2)

    start = time.time()
    result = my_trait_function(pts)
    elapsed = time.time() - start

    assert elapsed < 0.1  # Should complete in <100ms
```

## Code Style Guidelines

### Function Signatures

```python
def my_trait_function(
    pts: np.ndarray,
    param1: float = 1.0,
    param2: Optional[str] = None
) -> Union[float, np.ndarray]:
    """
    One-line summary.

    Detailed description explaining what the trait measures,
    biological relevance, and any important caveats.

    Args:
        pts: Root coordinates (n, 2) from base to tip.
        param1: Description of parameter 1 (default: 1.0).
        param2: Optional description (default: None).

    Returns:
        Trait value(s) with units and expected range.

    Raises:
        ValueError: When and why this error occurs.

    Example:
        >>> pts = np.array([[0, 0], [1, 1]])
        >>> result = my_trait_function(pts)
        >>> print(result)
        1.414

    Note:
        Any important implementation details or limitations.
    """
    # Implementation
    pass
```

### Type Hints

Always use type hints:

```python
from typing import Union, Optional, Tuple, List
import numpy as np

def process_roots(
    primary_pts: np.ndarray,
    lateral_pts_list: List[np.ndarray],
    threshold: float = 0.5
) -> Tuple[float, int]:
    """Type hints make code self-documenting and enable static checking."""
    pass
```

### Docstring Style

Follow Google style docstrings:

```python
def example_function(arg1: int, arg2: str = "default") -> bool:
    """
    Short one-line summary ending with period.

    Longer description providing context, explaining what the function
    does, and any important details.

    Args:
        arg1: Description of arg1.
        arg2: Description of arg2 (default: "default").

    Returns:
        Description of return value.

    Raises:
        ValueError: When arg1 is negative.
        TypeError: When arg2 is not a string.

    Example:
        >>> result = example_function(5, "test")
        >>> print(result)
        True
    """
    pass
```

## Vectorization Tips

Vectorize computations for better performance:

### Before (Slow)

```python
def compute_lengths_slow(pts_list):
    """Iterative computation."""
    lengths = []
    for pts in pts_list:
        length = 0
        for i in range(len(pts) - 1):
            segment = pts[i+1] - pts[i]
            length += np.linalg.norm(segment)
        lengths.append(length)
    return np.array(lengths)
```

### After (Fast)

```python
def compute_lengths_fast(pts_list):
    """Vectorized computation."""
    lengths = []
    for pts in pts_list:
        # Vectorized: compute all segments at once
        segments = np.diff(pts, axis=0)
        length = np.sum(np.linalg.norm(segments, axis=1))
        lengths.append(length)
    return np.array(lengths)
```

## Contributing Your Trait

To contribute your new trait to sleap-roots:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b add-sinuosity-trait`
3. **Implement trait** with tests and documentation
4. **Run all tests**: `pytest tests/ -v`
5. **Run code quality checks**:
   ```bash
   black sleap_roots tests
   pydocstyle sleap_roots/
   ```
6. **Create pull request** with:
   - Clear description of the new trait
   - Biological motivation
   - Test results
   - Documentation updates

## Next Steps

- Review [Testing Guide](testing.md) for comprehensive testing strategies
- See [Code Style](code-style.md) for style guidelines
- Read [Architecture](architecture.md) for module organization
- Check [Contributing](contributing.md) for PR process
- Explore [Trait Reference](../guides/trait-reference.md) for existing traits