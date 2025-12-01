# Installation

sleap-roots is a Python package for extracting morphological traits from plant root images analyzed with [SLEAP](https://sleap.ai).

## Quick Install

The simplest way to install sleap-roots:

```bash
pip install sleap-roots
```

That's it! You're ready to use sleap-roots.

## Recommended: Conda Environment

To avoid dependency conflicts, we recommend using conda:

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

## Modern Workflow: Using uv

For starting a new project, [uv](https://github.com/astral-sh/uv) provides a fast, modern Python workflow:

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
    uv is 10-100x faster than pip/conda and provides automatic dependency locking for reproducibility.

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
python -c "import sleap_roots; print(f'✅ sleap-roots {sleap_roots.__version__} installed successfully!')"
```

## Platform Support

sleap-roots is tested on:

| Platform | Python Versions | Status |
|----------|----------------|--------|
| **Ubuntu 22.04** | 3.7, 3.8, 3.9, 3.10, 3.11 | ✅ Fully supported |
| **macOS** | 3.11 | ✅ Fully supported |
| **Windows** | 3.11 | ✅ Fully supported |

## Troubleshooting

### "ModuleNotFoundError: No module named 'sleap_roots'"

Make sure your environment is activated:

```bash
conda activate sleap-roots
```

### Import Errors with sleap-io

Upgrade to the latest version:

```bash
pip install --upgrade sleap-io
```

For more help, see the [Troubleshooting Guide](../guides/troubleshooting.md).

## Next Steps

- **[Quick Start Tutorial](quickstart.md)** – Learn the basics in 5 minutes
- **[What is SLEAP?](what-is-sleap.md)** – Understand the underlying technology
- **[Pipeline Guide](../guides/index.md)** – Choose a pipeline for your data

!!! info "For Contributors"
    If you want to contribute to sleap-roots or run tests, see the **[Development Setup Guide](../dev/setup.md)** for instructions on cloning the repository and setting up a development environment.