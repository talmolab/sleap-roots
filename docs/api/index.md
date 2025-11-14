# API Reference

Welcome to the sleap-roots API reference documentation. This section provides detailed information about all public classes, functions, and modules.

## Overview

The sleap-roots API is organized into functional modules:

- **[Core Classes](#core-classes)**: `Series` and data structures
- **[Pipelines](#pipelines)**: Pre-built trait computation pipelines
- **[Trait Computation Modules](#trait-computation-modules)**: Individual trait calculations
- **[Utilities](#utilities)**: Helper functions and tools

## Core Classes

### Series

The `Series` class is the primary data structure for working with SLEAP predictions.

::: sleap_roots.Series
    options:
      members:
        - __init__
        - load
        - load_multi
        - get_primary_root_points
        - get_lateral_root_points
        - get_crown_root_points
      show_source: true

**Quick Example**:
```python
import sleap_roots as sr

# Load single plant
series = sr.Series.load(
    "my_plant",
    h5_path="predictions.h5",
    primary_path="primary.slp",
    lateral_path="lateral.slp"
)

# Access root points
primary_pts = series.get_primary_root_points()
lateral_pts_list = series.get_lateral_root_points()
```

**See also**: [Data Formats Guide](../guides/data-formats/sleap-files.md)

---

## Pipelines

Pre-built pipelines for common root system architectures.

### DicotPipeline

Pipeline for dicot plants with primary and lateral roots.

::: sleap_roots.DicotPipeline
    options:
      members:
        - __init__
        - compute_plant_traits
        - get_trait_definitions
      show_source: true

**Example**:
```python
pipeline = sr.DicotPipeline()
traits = pipeline.compute_plant_traits(series, write_csv=True)
```

**See also**: [Dicot Pipeline Tutorial](../tutorials/dicot-pipeline.md)

---

### YoungerMonocotPipeline

Pipeline for younger monocot plants with primary and crown roots.

::: sleap_roots.YoungerMonocotPipeline
    options:
      members:
        - __init__
        - compute_plant_traits
      show_source: true

**See also**: [Younger Monocot Tutorial](../tutorials/younger-monocot-pipeline.md)

---

### OlderMonocotPipeline

Pipeline for mature monocots with crown roots only.

::: sleap_roots.OlderMonocotPipeline
    options:
      members:
        - __init__
        - compute_plant_traits
      show_source: true

**See also**: [Older Monocot Tutorial](../tutorials/older-monocot-pipeline.md)

---

### MultipleDicotPipeline

Pipeline for multiple dicot plants in one image.

::: sleap_roots.MultipleDicotPipeline
    options:
      members:
        - __init__
        - compute_plant_traits
        - compute_multi_plant_traits
      show_source: true

**See also**: [Multiple Dicot Tutorial](../tutorials/multiple-dicot-pipeline.md)

---

### MultiplePrimaryRootPipeline

Pipeline for multiple plants with primary roots only.

::: sleap_roots.MultiplePrimaryRootPipeline
    options:
      members:
        - __init__
        - compute_plant_traits
        - compute_multi_plant_traits
      show_source: true

**See also**: [Multiple Primary Tutorial](../tutorials/multiple-primary-root-pipeline.md)

---

### PrimaryRootPipeline

Specialized pipeline for primary root analysis only.

::: sleap_roots.PrimaryRootPipeline
    options:
      members:
        - __init__
        - compute_plant_traits
      show_source: true

**See also**: [Primary Root Tutorial](../tutorials/primary-root-pipeline.md)

---

### LateralRootPipeline

Specialized pipeline for lateral root analysis.

::: sleap_roots.LateralRootPipeline
    options:
      members:
        - __init__
        - compute_plant_traits
      show_source: true

**See also**: [Lateral Root Tutorial](../tutorials/lateral-root-pipeline.md)

---

## Trait Computation Modules

Individual modules for computing specific trait types.

### Length Calculations

::: sleap_roots.lengths
    options:
      members:
        - get_root_lengths
        - get_max_length_pts
      show_source: false

**Module Functions**:
- `get_root_lengths()`: Compute path lengths for roots
- `get_max_length_pts()`: Get maximum length path through skeleton

**Example**:
```python
from sleap_roots import lengths

root_lengths = lengths.get_root_lengths([primary_pts, *lateral_pts_list])
print(f"Primary length: {root_lengths[0]:.2f} pixels")
```

**See also**: [Full lengths module reference](lengths.md)

---

### Angle Measurements

::: sleap_roots.angle
    options:
      members:
        - get_root_angle
        - get_lateral_emergence_angle
      show_source: false

**Module Functions**:
- `get_root_angle()`: Compute root angle relative to gravity
- `get_lateral_emergence_angle()`: Compute lateral root emergence angle

**Example**:
```python
from sleap_roots import angle

tip_angle = angle.get_root_angle(primary_pts, gravity_vector=(0, 1))
print(f"Primary root tip angle: {tip_angle:.1f}°")
```

**See also**: [Full angle module reference](angle.md)

---

### Tip Detection

::: sleap_roots.tips
    options:
      members:
        - get_root_tips
        - get_tip_angle
      show_source: false

**Module Functions**:
- `get_root_tips()`: Extract tip coordinates from roots
- `get_tip_angle()`: Compute angle at root tip

**See also**: [Full tips module reference](tips.md)

---

### Base Detection

::: sleap_roots.bases
    options:
      members:
        - count_root_bases
        - get_base_positions
      show_source: false

**Module Functions**:
- `count_root_bases()`: Count number of root bases
- `get_base_positions()`: Extract base coordinates

**See also**: [Full bases module reference](bases.md)

---

### Convex Hull

::: sleap_roots.convhull
    options:
      members:
        - get_convhull_area
        - get_convhull_perimeter
      show_source: false

**Module Functions**:
- `get_convhull_area()`: Compute convex hull area
- `get_convhull_perimeter()`: Compute convex hull perimeter

**Example**:
```python
from sleap_roots import convhull

all_roots = [primary_pts] + lateral_pts_list
hull_area = convhull.get_convhull_area(all_roots)
print(f"Root system convex hull area: {hull_area:.2f} pixels²")
```

**See also**: [Full convhull module reference](convhull.md)

---

### Network Analysis

::: sleap_roots.networklength
    options:
      members:
        - get_network_length
        - get_network_distribution
      show_source: false

**Module Functions**:
- `get_network_length()`: Total network length
- `get_network_distribution()`: Network length distribution metrics

**See also**: [Full networklength module reference](networklength.md)

---

## Utilities

### Visualization

Helper functions for plotting and visualization:

- `plot_root_network()`: Plot root network with nodes and edges
- `plot_trait_timeseries()`: Plot trait values over time
- `plot_convex_hull()`: Visualize convex hull around roots

### Data Export

Functions for exporting results:

- `export_to_csv()`: Export trait DataFrame to CSV
- `export_to_hdf5()`: Save results to HDF5 format

### Batch Processing

Tools for processing multiple plants:

- `process_directory()`: Process all files in a directory
- `parallel_process()`: Parallel processing with multiprocessing

---

## Module Organization

```
sleap_roots/
├── __init__.py           # Package initialization
├── series.py             # Series class
├── trait_pipelines.py    # Pipeline implementations
├── lengths.py            # Length calculations
├── angle.py              # Angle measurements
├── tips.py               # Tip-related traits
├── bases.py              # Base-related traits
├── convhull.py           # Convex hull computations
├── networklength.py      # Network-level metrics
├── monocots.py           # Monocot-specific traits
└── utils.py              # Utility functions
```

## Type Hints

sleap-roots uses comprehensive type hints for better IDE support and static checking:

```python
from typing import Union, Optional, Tuple, List
import numpy as np

def compute_trait(
    pts: np.ndarray,
    threshold: float = 0.5
) -> Union[float, np.ndarray]:
    """Function with clear type hints."""
    pass
```

## Error Handling

Most functions raise informative errors for invalid inputs:

```python
try:
    traits = pipeline.compute_plant_traits(series)
except ValueError as e:
    print(f"Invalid input: {e}")
except FileNotFoundError as e:
    print(f"File not found: {e}")
```

## Performance Considerations

### Vectorization

Most trait computations use vectorized NumPy operations for efficiency:

```python
# Fast: vectorized
lengths = np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))

# Slow: iterative
# lengths = sum(np.linalg.norm(pts[i+1] - pts[i]) for i in range(len(pts)-1))
```

### Batch Processing

Process multiple plants efficiently:

```python
# Good: batch processing
traits = pipeline.compute_multi_plant_traits(series_list)

# Less efficient: sequential
# traits = [pipeline.compute_plant_traits(s) for s in series_list]
```

## API Stability

### Stable API (v0.2+)

These components have stable APIs and can be relied upon:

- `Series` class and methods
- All pipeline classes
- Core trait computation functions in `lengths`, `angles`, `tips`, `bases`

### Experimental API

Components marked as experimental may change in future versions:

- Advanced network metrics
- Some visualization functions

Check function docstrings for stability notes.

## Version Compatibility

### Python Version

- **Minimum**: Python 3.7
- **Recommended**: Python 3.11+
- **Tested**: 3.7, 3.8, 3.9, 3.10, 3.11

### Dependency Requirements

Key dependencies:

- `numpy`: Array operations
- `pandas`: DataFrame handling
- `sleap-io >= 0.0.11`: Loading SLEAP files
- `h5py`: HDF5 file handling
- `scipy`: Scientific computations
- `scikit-image`: Image processing
- `shapely`: Geometry operations

See [Installation Guide](../getting-started/installation.md) for details.

## Contributing to the API

Interested in adding new traits or pipelines? See:

- [Adding Traits](../dev/adding-traits.md)
- [Creating Pipelines](../dev/creating-pipelines.md)
- [Contributing Guide](../dev/contributing.md)

## Complete API Index

For the complete, auto-generated API reference for all modules, see the [reference](../reference/) section (generated from docstrings).

## Next Steps

- **New users**: Start with [Quick Start](../getting-started/quickstart.md)
- **Pipeline users**: See [Tutorial](../tutorials/index.md)
- **Developers**: Read [Architecture](../dev/architecture.md)
- **Trait details**: Check [Trait Reference](../guides/trait-reference.md)