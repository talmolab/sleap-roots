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

### `DicotPipeline`

**1. Computing traits for a single plant:**

```py
import sleap_roots as sr

plant = sr.Series.load(
    "tests/data/canola_7do/919QDUH.h5",
    # Specify the names of the primary and lateral roots for trait calculation
    primary_name="primary",
    lateral_name="lateral"
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
        # Specify the names of the primary and lateral roots for trait calculation
        primary_name="primary",
        lateral_name="lateral",
    ) for plant_path in plant_paths]

pipeline = sr.DicotPipeline()
all_traits = pipeline.compute_batch_traits(plants, write_csv=True)
```

**3. Computing individual traits:**

```py
import sleap_roots as sr
import numpy as np
import numpy as np
# Import utility for combining primary and lateral root points
from sleap_roots.points import get_all_pts_array

plant = sr.Series.load(
    "tests/data/canola_7do/919QDUH.h5",
    primary_name="primary",
    lateral_name="lateral"
)

frame_index = 0
primary_pts = plant.get_primary_points(frame_index)
lateral_pts = plant.get_lateral_points(frame_index)
pts = get_all_pts_array(primary_pts, lateral_pts)
convex_hull = sr.convhull.get_convhull(pts)
```

### `YoungerMonocotPipeline`

**1. Computing traits for a single plant:**

```py
import sleap_roots as sr

plant = sr.Series.load(
    "tests/data/rice_3do/0K9E8BI.h5",
    primary_name="primary",
    lateral_name="crown"
)
pipeline = sr.YoungerMonocotPipeline()
traits = pipeline.compute_plant_traits(plant, write_csv=True)
```

**2. Computing traits for a batch of plants:**

```py
import sleap_roots as sr

plant_paths = sr.find_all_series("tests/data/rice_3do")
plants = [
    sr.Series.load(
        plant_path,
        primary_name="primary",
        lateral_name="crown"
    ) for plant_path in plant_paths]

pipeline = sr.YoungerMonocotPipeline()
all_traits = pipeline.compute_batch_traits(plants, write_csv=True)
```

**3. Computing individual traits:**

```py
import sleap_roots as sr
import numpy as np
from sleap_roots.points import get_all_pts_array

plant = sr.Series.load(
    "tests/data/rice_3do/0K9E8BI.h5",
    primary_name="primary",
    lateral_name="crown"
)

frame_index = 0
primary_pts = plant.get_primary_points(frame_index)
lateral_pts = plant.get_lateral_points(frame_index)
pts = get_all_pts_array(primary_pts, lateral_pts)
convex_hull = sr.convhull.get_convhull(pts)
```

## Tutorials
Jupyter notebooks are located in this repo at `sleap-roots/notebooks`.

To use them, add Jupyter Lab to your conda environment (recommended):

```
conda activate sleap-roots
pip install jupyterlab
```
Clone this repository if you haven't already:

```
git clone https://github.com/talmolab/sleap-roots.git && cd sleap-roots
```

Then you can change directories to the location of the notebooks, and open Jupyter Lab:

```
cd notebooks
jupyter lab
```

Go through the commands in the notebooks to learn about each pipeline. 
You can use the test data located at `tests/data` or copy the notebooks elsewhere for use with your own data!


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

E.M. Berrigan, L. Wang, H. Carrillo, K. Echegoyen, M. Kappes, J. Torres, A. Ai-Perreira, E. McCoy, E. Shane, C.D. Copeland, L. Ragel, 
C. Georgousakis, S. Lee, D. Reynolds, A. Talgo, J. Gonzalez, L. Zhang, A.B. Rajurkar, M. Ruiz, E. Daniels, L. Maree, S. Pariyar, W. Busch, T.D. Pereira.  
"Fast and Efficient Root Phenotyping via Pose Estimation", *Plant Phenomics* 0: DOI:[10.34133/plantphenomics.0175](https://doi.org/10.34133/plantphenomics.0175).
