# sleap-roots

[![CI](https://github.com/talmolab/sleap-roots/actions/workflows/ci.yml/badge.svg)](https://github.com/talmolab/sleap-roots/actions/workflows/ci.yml)
[![Lint](https://github.com/talmolab/sleap-roots/actions/workflows/lint.yml/badge.svg)](https://github.com/talmolab/sleap-roots/actions/workflows/lint.yml)
[![codecov](https://codecov.io/gh/talmolab/sleap-roots/branch/main/graph/badge.svg)](https://codecov.io/gh/talmolab/sleap-roots)
[![Release](https://img.shields.io/github/v/release/talmolab/sleap-roots?label=Latest)](https://github.com/talmolab/sleap-roots/releases/)
[![PyPI](https://img.shields.io/pypi/v/sleap-roots?label=PyPI)](https://pypi.org/project/sleap-roots)

Analysis tools for [SLEAP](https://sleap.ai)-based plant root phenotyping.

## Installation
```
pip install git+https://github.com/talmolab/sleap-roots.git@main
```

If you are using conda:
```
conda create -n sleap-roots python=3.8
conda activate sleap-roots
pip install git+https://github.com/talmolab/sleap-roots.git@main
```

### Development
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

To **develop on M1 Macs**, you'll need to manually install SLEAP first like this:
```
git clone https://github.com/talmolab/sleap && cd sleap
conda env create -f environment_m1.yml -n sleap-roots
```
Then, install this package in editable mode:
```
cd .. && git clone https://github.com/talmolab/sleap-roots
pip install -e ".[dev]"
```