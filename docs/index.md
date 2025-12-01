# sleap-roots

Analysis tools for [SLEAP](https://sleap.ai)-based plant root phenotyping.

<div class="grid cards" markdown>

-   :seedling:{ .lg .middle } __Fast Root Phenotyping__

    ---

    Extract 50+ morphological traits from plant root images using pose estimation with SLEAP.

    [:octicons-arrow-right-24: Quick Start](getting-started/quickstart.md)

-   :bar_chart:{ .lg .middle } __Pipeline-Based Analysis__

    ---

    Pre-built pipelines for dicots, monocots, and custom root system architectures.

    [:octicons-arrow-right-24: Pipeline Guide](guides/index.md)

-   :microscope:{ .lg .middle } __Scientifically Validated__

    ---

    Published methods with reproducible trait computations and batch processing.

    [:octicons-arrow-right-24: Read the Paper](https://doi.org/10.34133/plantphenomics.0175)

-   :material-code-braces:{ .lg .middle } __Developer Friendly__

    ---

    Clean Python API with comprehensive docs, type hints, and [![codecov](https://codecov.io/gh/talmolab/sleap-roots/branch/main/graph/badge.svg)](https://codecov.io/gh/talmolab/sleap-roots) test coverage.

    [:octicons-arrow-right-24: API Reference](reference/)

</div>

## What is sleap-roots?

**sleap-roots** is a Python package for extracting morphological traits from plant root images analyzed with [SLEAP](https://sleap.ai) (Social LEAP Estimates Animal Poses). While SLEAP is designed for animal pose estimation, it works exceptionally well for tracking plant root landmarks over time.

This package provides:

- **Trait pipelines** for different root system architectures (dicots, monocots)
- **50+ morphological traits** including lengths, angles, counts, and topology
- **Batch processing** for high-throughput phenotyping experiments
- **Modular utilities** for custom trait development

## Key Features

### :material-checkbox-multiple-outline: Multiple Pipeline Support

Choose from pre-built pipelines optimized for different plant types:

- **DicotPipeline** – Primary + lateral roots (soy, canola, arabidopsis)
- **YoungerMonocotPipeline** – Primary + crown roots (early-stage rice, maize)
- **OlderMonocotPipeline** – Crown roots only (mature rice, maize)
- **MultipleDicotPipeline** – Multi-plant dicot setups
- **PrimaryRootPipeline** – Primary root only
- **LateralRootPipeline** – Lateral roots only

Or create your own custom pipeline!

### :material-speedometer: High Performance

- Process hundreds of plants in minutes
- Vectorized NumPy operations for fast computation
- Efficient batch processing with parallelization support

### :material-flask: Research Ready

- Published validation in [*Plant Phenomics*](https://doi.org/10.34133/plantphenomics.0175)
- Reproducible trait computations with clear documentation
- CSV export compatible with statistical analysis tools

### :material-tools: Extensible

- Modular design for custom trait development
- Clean Python API with comprehensive type hints
- Well-tested codebase [![codecov](https://codecov.io/gh/talmolab/sleap-roots/branch/main/graph/badge.svg)](https://codecov.io/gh/talmolab/sleap-roots)

## Quick Example

```python
import sleap_roots as sr

# Load SLEAP predictions
series = sr.Series.load(
    series_name="my_plant",
    h5_path="predictions.h5",
    primary_path="primary_roots.slp",
    lateral_path="lateral_roots.slp"
)

# Compute traits using dicot pipeline
pipeline = sr.DicotPipeline()
traits = pipeline.compute_plant_traits(series, write_csv=True)

# Access individual traits
print(f"Primary root length: {traits['primary_length'].iloc[0]:.2f} pixels")
print(f"Lateral root count: {traits['lateral_count'].iloc[0]}")
```

## Citation

If you use sleap-roots in your research, please cite:

E.M. Berrigan et al., *"Fast and Efficient Root Phenotyping via Pose Estimation"*, Plant Phenomics.
DOI: [10.34133/plantphenomics.0175](https://doi.org/10.34133/plantphenomics.0175)

## Acknowledgments

Created by the [Talmo Lab](https://talmolab.org) and [Busch Lab](https://busch.salk.edu) at the Salk Institute, as part of the [Harnessing Plants Initiative](https://www.salk.edu/harnessing-plants-initiative/).

### Contributors

- Elizabeth Berrigan
- Lin Wang
- Andrew O'Connor
- Talmo Pereira