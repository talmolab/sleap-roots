# Installation

## Quick Install

The simplest way to install sleap-roots is via pip:

```bash
pip install sleap-roots
```

## Recommended: Conda Environment

We recommend using conda to manage dependencies and avoid conflicts:

```bash
# Create a new environment with Python 3.11
conda create -n sleap-roots python=3.11

# Activate the environment
conda activate sleap-roots

# Install sleap-roots
pip install sleap-roots
```

!!! tip "Why Python 3.11?"
    While sleap-roots supports Python 3.7+, we recommend 3.11 for the best performance and compatibility with recent NumPy/Pandas versions.

## Development Installation

If you want to contribute to sleap-roots or run the latest development version, we recommend using **uv** for the fastest and most modern workflow.

### Modern Approach: Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast, modern Python package manager that handles dependency management with lockfiles for reproducibility.

#### 1. Install uv

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip
pip install uv
```

#### 2. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/talmolab/sleap-roots.git
cd sleap-roots

# Install Git LFS for test data
git lfs install
git lfs pull

# Create environment and install all dependencies
uv sync

# This automatically:
# - Creates a virtual environment (.venv)
# - Installs runtime dependencies
# - Installs dev dependencies from [dependency-groups]
# - Installs sleap-roots in editable mode
# - Generates/updates uv.lock for reproducibility
```

#### 3. Verify Installation

```bash
# Run tests
uv run pytest tests/

# Check formatting
uv run black --check sleap_roots tests

# Check docstrings
uv run pydocstyle sleap_roots/

# Import the package
uv run python -c "import sleap_roots; print(sleap_roots.__version__)"
```

!!! tip "Why uv?"
    - **10-100x faster** than pip/conda
    - **Automatic dependency locking** with uv.lock
    - **No separate environment activation** needed (use `uv run`)
    - **PEP 735 dependency groups** for clean dev/prod separation
    - **Built-in tool management** (no need for separate virtualenv)

### Alternative: Using Conda

If you prefer conda or need conda-specific packages:

#### 1. Clone the Repository

```bash
git clone https://github.com/talmolab/sleap-roots.git
cd sleap-roots
```

#### 2. Create Environment from File

```bash
conda env create -f environment.yml
conda activate sleap-roots
```

This installs:

- All runtime dependencies (numpy, pandas, sleap-io, etc.)
- Dev dependencies (pytest, black, pydocstyle, mkdocs)
- The package in editable mode

#### 3. Verify Installation

```bash
# Run tests
pytest tests/

# Check formatting
black --check sleap_roots tests

# Import the package
python -c "import sleap_roots; print(sleap_roots.__version__)"
```

## Platform Support

sleap-roots is tested on:

| Platform | Python Versions | Status |
|----------|----------------|--------|
| **Ubuntu 22.04** | 3.7, 3.8, 3.9, 3.10, 3.11 | :white_check_mark: Fully supported |
| **macOS** | 3.11 | :white_check_mark: Fully supported |
| **Windows** | 3.11 | :white_check_mark: Fully supported |

## Dependencies

sleap-roots has the following core dependencies:

- **numpy** – Numerical operations
- **pandas** – Data frames for trait output
- **h5py** – Reading HDF5 video files
- **sleap-io** – Loading SLEAP prediction files (.slp)
- **scikit-image** – Image processing utilities
- **shapely** – Geometric operations
- **matplotlib** – Visualization (optional)
- **seaborn** – Statistical plots (optional)

All dependencies are automatically installed with pip.

## Git LFS for Test Data

If you're developing or running tests, you'll need Git LFS to download test data:

```bash
# Install Git LFS
# macOS
brew install git-lfs

# Ubuntu
sudo apt-get install git-lfs

# Initialize Git LFS
git lfs install

# Pull test data (898 MB)
git lfs pull
```

Without Git LFS, test data files will be pointer files (~130 bytes) instead of the actual data, and tests will fail.

## Troubleshooting

### "ModuleNotFoundError: No module named 'sleap_roots'"

Make sure the conda environment is activated:

```bash
conda activate sleap-roots
```

### "FileNotFoundError" in Tests

Pull Git LFS data:

```bash
git lfs install
git lfs pull
```

### Conda Environment Creation is Slow

Use mamba (much faster than conda):

```bash
conda install -n base -c conda-forge mamba
mamba env create -f environment.yml
```

### Import Errors with sleap-io

Upgrade to the latest version:

```bash
pip install --upgrade sleap-io
```

## Verifying Your Installation

Run the environment validation command to check everything is set up correctly:

```bash
# If using Claude commands
/validate-env
```

Or manually check:

```python
import sleap_roots as sr
from sleap_roots import DicotPipeline

# Check version
print(f"sleap-roots version: {sr.__version__}")

# Instantiate a pipeline
pipeline = DicotPipeline()
print("✅ Installation successful!")
```

## Next Steps

- [Quick Start Tutorial](quickstart.md) – Learn the basics
- [What is SLEAP?](what-is-sleap.md) – Understand the underlying technology
- [Pipeline Guide](../guides/index.md) – Choose a pipeline for your data