# Development Setup

This guide will help you set up a development environment for contributing to sleap-roots.

!!! info "For End Users"
    If you just want to **use** sleap-roots (not contribute code), see the **[Installation Guide](../getting-started/installation.md)** instead. This page is for developers who want to contribute to the project or run tests.

## Prerequisites

Before you begin, ensure you have:

- **Git** (2.0+) – Version control
- **Git LFS** – For test data (898 MB)
- **Python** (3.7-3.11) – We recommend 3.11
- **uv** (recommended) or **conda** – Package management

## Quick Setup with uv (Recommended)

The fastest way to get started is with [uv](https://github.com/astral-sh/uv), a modern Python package manager.

### 1. Install uv

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip/pipx
pip install uv
```

### 2. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/talmolab/sleap-roots.git
cd sleap-roots

# Install Git LFS and pull test data
git lfs install
git lfs pull

# Setup environment (creates .venv, installs all dependencies)
uv sync
```

That's it! You're ready to develop.

### 3. Verify Setup

```bash
# Run tests
uv run pytest tests/

# Check code formatting
uv run black --check sleap_roots tests

# Check docstrings
uv run pydocstyle sleap_roots/

# Build documentation
uv run mkdocs serve
```

### 4. Development Workflow

```bash
# Run tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_pipelines.py

# Run with coverage
uv run pytest --cov=sleap_roots --cov-report=term-missing tests/

# Format code
uv run black sleap_roots tests

# Check types (if mypy added)
uv run mypy sleap_roots

# Build docs locally
uv run mkdocs serve
# Visit http://127.0.0.1:8000

# Add new dependency
uv add package-name

# Add dev dependency
uv add --dev package-name
```

!!! tip "No activation needed"
    With uv, you don't need to activate a virtual environment. Just use `uv run` before commands, and uv handles everything automatically.

### 5. Update Dependencies

```bash
# Update all dependencies to latest compatible versions
uv lock --upgrade

# Sync to updated lockfile
uv sync

# Check for outdated packages
uv tree
```

## Alternative: Setup with Conda

If you prefer conda or need conda-specific packages:

### 1. Install Conda/Mamba

```bash
# Install mamba (faster than conda)
conda install -n base -c conda-forge mamba
```

### 2. Clone and Setup

```bash
# Clone repository
git clone https://github.com/talmolab/sleap-roots.git
cd sleap-roots

# Install Git LFS
git lfs install
git lfs pull

# Create environment from file
conda env create -f environment.yml

# Activate environment
conda activate sleap-roots
```

### 3. Development Workflow

```bash
# Activate environment
conda activate sleap-roots

# Run tests
pytest tests/ -v

# Format code
black sleap_roots tests

# Check docstrings
pydocstyle sleap_roots/

# Build docs
mkdocs serve
```

## Git LFS Setup

sleap-roots uses Git LFS for test data (~898 MB). This is **required** for running tests.

### Install Git LFS

```bash
# macOS
brew install git-lfs

# Ubuntu/Debian
sudo apt-get install git-lfs

# Windows
# Download from https://git-lfs.github.com/

# Initialize
git lfs install
```

### Pull Test Data

```bash
# Pull all LFS files
git lfs pull

# Verify (should show binary data, not text pointers)
file tests/data/*.h5
# Output: tests/data/example.h5: Hierarchical Data Format (version 5) data
```

## IDE Configuration

### VS Code

Recommended extensions:
- Python (Microsoft)
- Pylance
- Black Formatter
- autoDocstring

Settings (`.vscode/settings.json`):
```json
{
  "python.defaultInterpreterPath": ".venv/bin/python",
  "python.formatting.provider": "black",
  "python.linting.enabled": true,
  "python.linting.pydocstyleEnabled": true,
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests"],
  "editor.formatOnSave": true
}
```

### PyCharm

1. Open project in PyCharm
2. File → Settings → Project → Python Interpreter
3. Add interpreter: `.venv/bin/python`
4. Enable Black formatter:
   - File → Settings → Tools → Black
   - Check "On code reformat" and "On save"
5. Configure pytest:
   - Run → Edit Configurations → Add pytest
   - Target: `tests/`

## Running Tests

### Full Test Suite

```bash
# Run all tests
uv run pytest tests/

# With verbose output
uv run pytest tests/ -v

# With coverage
uv run pytest --cov=sleap_roots --cov-report=html tests/
# Open htmlcov/index.html to view coverage report
```

### Specific Tests

```bash
# Run specific test file
uv run pytest tests/test_series.py

# Run specific test function
uv run pytest tests/test_series.py::test_load_series

# Run tests matching pattern
uv run pytest tests/ -k "dicot"
```

### Test Markers

```bash
# Run fast tests only
uv run pytest -m "not slow"

# Run integration tests
uv run pytest -m integration
```

## Code Quality Checks

### Formatting

```bash
# Check formatting (no changes)
uv run black --check sleap_roots tests

# Format code
uv run black sleap_roots tests
```

### Linting

```bash
# Check docstrings
uv run pydocstyle sleap_roots/

# Check for common issues (if ruff/flake8 added)
uv run ruff check sleap_roots/
```

### Pre-commit Hooks (Optional)

```bash
# Install pre-commit
uv add --dev pre-commit

# Install hooks
uv run pre-commit install

# Run on all files
uv run pre-commit run --all-files
```

## Building Documentation

### Local Preview

```bash
# Serve docs with live reload
uv run mkdocs serve

# Open http://127.0.0.1:8000
```

### Build Static Site

```bash
# Build production site
uv run mkdocs build

# Output in site/
# Test by opening site/index.html
```

### Documentation Structure

```
docs/
├── index.md              # Home page
├── getting-started/      # User installation guides
├── tutorials/            # Pipeline tutorials
├── guides/               # User guides
├── dev/                  # Developer documentation
├── api/                  # API reference
├── cookbook/             # Code recipes
└── changelog.md          # Release history
```

## Common Tasks

### Add a New Pipeline

1. Create pipeline class in `sleap_roots/trait_pipelines.py`
2. Add tests in `tests/test_pipelines.py`
3. Export in `sleap_roots/__init__.py`
4. Add documentation page in `docs/guides/pipelines/`
5. Add tutorial in `docs/tutorials/`
6. Update `mkdocs.yml` navigation

### Add a New Trait

1. Create trait function in appropriate module (e.g., `sleap_roots/lengths.py`)
2. Add to `TraitDef` in pipeline
3. Add tests in `tests/test_<module>.py`
4. Document in `docs/guides/trait-reference.md`
5. Add example in `docs/cookbook/custom-traits.md`

See [Adding Traits](adding-traits.md) for detailed guide.

### Release Process

1. Update version in `sleap_roots/__init__.py`
2. Update `docs/changelog.md`
3. Create release branch: `git checkout -b release/v0.x.x`
4. Run full test suite
5. Create PR to main
6. After merge, tag release: `git tag -a v0.x.x`
7. Push tag: `git push origin v0.x.x`
8. GitHub Actions builds and publishes to PyPI

See [Release Process](release-process.md) for complete workflow.

## Troubleshooting

### Tests Fail with "FileNotFoundError"

**Cause**: Git LFS data not pulled

**Solution**:
```bash
git lfs install
git lfs pull
```

### Import Errors in Tests

**Cause**: Package not installed in editable mode

**Solution**:
```bash
# With uv
uv sync

# With conda
pip install -e .
```

### Black Formatting Conflicts

**Cause**: Different Black versions

**Solution**:
```bash
# Use project-pinned version
uv run black sleap_roots tests

# Or update Black
uv add --dev black
```

### Documentation Build Fails

**Cause**: Missing mkdocs dependencies

**Solution**:
```bash
# Install doc dependencies
uv sync

# Or specifically
uv add --dev mkdocs mkdocs-material mkdocstrings
```

### uv Commands Fail

**Cause**: uv not in PATH or outdated

**Solution**:
```bash
# Reinstall uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Update uv
uv self update

# Check version
uv --version
```

## Getting Help

- **Documentation**: https://talmolab.github.io/sleap-roots/
- **Issues**: https://github.com/talmolab/sleap-roots/issues
- **Discussions**: https://github.com/talmolab/sleap-roots/discussions
- **SLEAP Community**: https://sleap.ai/community

## Next Steps

- Read [Contributing Guide](contributing.md) for workflow
- Review [Code Style](code-style.md) for conventions
- See [Creating Pipelines](creating-pipelines.md) for pipeline development
- Check [Testing Guide](testing.md) for test best practices