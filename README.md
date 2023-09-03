# sleap-roots

[![CI](https://github.com/talmolab/sleap-roots/actions/workflows/ci.yml/badge.svg)](https://github.com/talmolab/sleap-roots/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/talmolab/sleap-roots/branch/main/graph/badge.svg)](https://codecov.io/gh/talmolab/sleap-roots)
[![Release](https://img.shields.io/github/v/release/talmolab/sleap-roots?label=Latest)](https://github.com/talmolab/sleap-roots/releases/)
[![PyPI](https://img.shields.io/pypi/v/sleap-roots?label=PyPI)](https://pypi.org/project/sleap-roots)

Analysis tools for [SLEAP](https://sleap.ai)-based plant root phenotyping.

## Installation
```
pip install sleap-roots
```

If you are using conda (recommended):
```
conda create -n sleap-roots python=3.9
conda activate sleap-roots
pip install sleap-roots
```

## Usage

**1. Computing traits for a single plant:**

```py
import sleap_roots as sr

plant = sr.Series.load(
    "tests/data/canola_7do/919QDUH.h5",
    primary_name="primary_multi_day",
    lateral_name="lateral_3_nodes"
)
pipeline = sr.DicotPipeline()
traits = pipeline.compute_plant_traits(plant, write_csv=True)
```

**2. Computing traits for a batch of plants:**

```py
import sleap_roots as sr

plant_paths = sr.find_all_series("tests/data/soy_6do")
plants = [
    sr.Series.load(
        plant_path,
        primary_name="primary_multi_day",
        lateral_name="lateral__nodes",
    ) for plant_path in plant_paths]

pipeline = sr.DicotPipeline()
all_traits = pipeline.compute_batch_traits(plants, write_csv=True)
```

**3. Computing individual traits:**

```py
import sleap_roots as sr
import numpy as np

plant = sr.Series.load(
    "tests/data/canola_7do/919QDUH.h5",
    primary_name="primary_multi_day",
    lateral_name="lateral_3_nodes"
)

primary, lateral = plant[10]
pts = np.concatenate([primary.numpy(), lateral.numpy()], axis=0).reshape(-1, 2)
convex_hull = sr.convhull.get_convhull(pts)
```

## Development
For development, first clone the repository:
```
git clone https://github.com/talmolab/sleap-roots && cd sleap-roots
```

Then, to create a **new conda environment** and install the package in editable mode:
```
conda env create -f environment.yml
```
This will create a conda environment called `sleap-roots`.

If you have an **existing conda environment** (such as where you installed SLEAP), you
can just install in editable mode directly. First, activate your environment and then:
```
pip install -e ".[dev]"
```
*Note:* The `[dev]` makes sure that the development-only dependencies are also
installed.

To **start fresh**, just delete the environment:
```
conda env remove -n sleap-roots
```

To **run tests**, first activate the environment:
```
conda activate sleap-roots
```
Then run `pytest` with:
```
pytest tests
```

## Acknowledgments

This repository was created by the [Talmo Lab](https://talmolab.org) and [Busch Lab](https://busch.salk.edu) at the Salk Institute for Biological Studies as part of the [Harnessing Plants Initiative](https://www.salk.edu/harnessing-plants-initiative/).

### Contributors

- Elizabeth Berrigan
- Lin Wang
- Talmo Pereira

### Citation

*Coming soon.*