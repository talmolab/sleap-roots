# Installation

sleap-roots is a Python package for extracting morphological traits from plant root images analyzed with [SLEAP](https://sleap.ai).

## Getting Started with uv (Recommended)

For starting a new project, [uv](https://github.com/astral-sh/uv) provides a fast, modern Python workflow that ensures clean dependency management:

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh  # macOS/Linux
# Or: pip install uv

# Create a new project
uv init my-root-analysis
cd my-root-analysis

# Add sleap-roots to your project
uv add sleap-roots

# Run your analysis script
uv run python my_analysis.py
```

!!! tip "Why uv?"
    uv is 10-100x faster than pip and provides automatic dependency locking for reproducibility.

!!! info "Python Version"
    We recommend Python 3.11 for best performance and compatibility with recent NumPy/Pandas versions.

## Verify Installation

Test that sleap-roots is installed correctly:

```python
import sleap_roots as sr

# Check version
print(f"sleap-roots version: {sr.__version__}")

# Instantiate a pipeline
pipeline = sr.DicotPipeline()
print("✅ Installation successful!")
```

Or run this one-liner:

```bash
uv run python -c "import sleap_roots; print(f'✅ sleap-roots {sleap_roots.__version__} installed successfully!')"
```

!!! note "Using uv run"
    The `uv run` prefix ensures the command runs in your project's isolated environment. If you installed with conda or pip instead, use `python -c "..."` directly.

## Alternative: Using Conda

If you prefer conda for package management:

```bash
# Create a new environment with Python 3.11
conda create -n sleap-roots python=3.11

# Activate the environment
conda activate sleap-roots

# Install sleap-roots
pip install sleap-roots
```

## Alternative: Using pip

If you just want to quickly install sleap-roots into an existing Python environment:

```bash
pip install sleap-roots
```

!!! warning "Environment Isolation"
    Installing with pip directly may install packages into your base Python environment, which can cause dependency conflicts with other projects. We recommend using uv (above) or conda for better isolation, especially if you're new to Python.

## Platform Support

sleap-roots is tested on:

| Platform | Python Version | CI Status |
|----------|----------------|-----------|
| **Ubuntu 22.04** | 3.11 | ✅ Tested in CI |
| **macOS** | 3.11 | ✅ Tested in CI |
| **Windows** | 3.11 | ✅ Tested in CI |
| **All platforms** | 3.7-3.12 | ⚠️ Should work (not CI tested) |

!!! info "Python Version Support"
    We officially test and support **Python 3.11** on all platforms through continuous integration. Other versions (3.7-3.12) should work based on our dependencies, but are not continuously tested. We recommend 3.11 for the best experience.

## Troubleshooting

!!! warning "Import Errors?"
    If you encounter `ModuleNotFoundError: No module named 'sleap_roots'`, see the [Troubleshooting Guide](../guides/troubleshooting.md#import-errors) for solutions.

## Next Steps

- **[Quick Start Tutorial](quickstart.md)** – Learn the basics in 5 minutes
- **[What is SLEAP?](what-is-sleap.md)** – Understand the underlying technology
- **[Pipeline Guide](../guides/index.md)** – Choose a pipeline for your data

!!! info "For Contributors"
    If you want to contribute to sleap-roots or run tests, see the **[Development Setup Guide](../dev/setup.md)** for instructions on cloning the repository and setting up a development environment.