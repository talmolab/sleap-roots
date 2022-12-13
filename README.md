# sleap-roots

[![CI](https://github.com/talmolab/sleap-roots/actions/workflows/ci.yml/badge.svg)](https://github.com/talmolab/sleap-roots/actions/workflows/ci.yml)
[![Lint](https://github.com/talmolab/sleap-roots/actions/workflows/lint.yml/badge.svg)](https://github.com/talmolab/sleap-roots/actions/workflows/lint.yml)
[![codecov](https://codecov.io/gh/talmolab/sleap-roots/branch/main/graph/badge.svg)](https://codecov.io/gh/talmolab/sleap-roots)
[![Release](https://img.shields.io/github/v/release/talmolab/sleap-roots?label=Latest)](https://github.com/talmolab/sleap-roots/releases/)
[![PyPI](https://img.shields.io/pypi/v/sleap-roots?label=PyPI)](https://pypi.org/project/sleap-roots)

Analysis tools for [SLEAP](https://sleap.ai)-based plant root phenotyping.

## Installation
```
pip install sleap-roots
```

If you are using conda:
```
conda create -n sleap-roots python=3.8
conda activate sleap-roots
pip install sleap-roots
```

### Development
For development, use the following syntax to install in editable mode:
```
conda env create -f environment.yml
```
This will create a conda environment called `sleap-roots`.

To run tests, first activate the environment:
```
conda activate sleap-roots
```
Then run `pytest` with:
```
pytest tests
```
To start fresh, just delete the environment:
```
conda env remove -n sleap-roots
```
