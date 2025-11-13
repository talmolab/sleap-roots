# Lint & Format Code

Run linting and formatting checks to ensure code quality and consistent style.

## Commands

```bash
# Format code with Black
uv run black sleap_roots tests

# Check formatting without making changes
uv run black --check sleap_roots tests

# Run pydocstyle for Google-style docstring checks
uv run pydocstyle --convention=google sleap_roots/
```

## What to do after running

1. **Review formatting changes** - Check the diff to ensure Black changes look correct
2. **Fix docstring issues** - Address any pydocstyle errors for Google-style docstrings
3. **Commit changes** - If formatting changed files, commit them separately from logic changes

## Project Context

### Black Formatting
- **Line length**: 88 characters
- **Style**: PEP 8 compliant with Black defaults
- **Configuration**: Defined in `pyproject.toml`

### Pydocstyle
- **Convention**: Google-style docstrings (enforced)
- **Scope**: All modules in `sleap_roots/`
- **Configuration**: Defined in `pyproject.toml`

### CI Requirements
Both checks run in CI (`.github/workflows/ci.yml`):
- Black check must pass before merge
- Pydocstyle check must pass before merge

Run these commands locally to catch issues before pushing!

## Common Issues & Fixes

### Black Formatting Issues
```bash
# Fix all formatting automatically
black sleap_roots tests

# Check specific file
black --check sleap_roots/lengths.py
```

### Pydocstyle Issues

**Missing docstrings**:
```python
# Bad
def calculate_length(points):
    return np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))

# Good
def calculate_length(points):
    """Calculate total length of a polyline.

    Args:
        points: Array of shape (n, 2) containing x,y coordinates.

    Returns:
        Total length as a float.
    """
    return np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))
```

**Incorrect docstring format**:
```python
# Bad (ReStructuredText style)
"""
:param points: Array of points
:return: Length
"""

# Good (Google style)
"""Calculate length.

Args:
    points: Array of points.

Returns:
    Length as float.
"""
```

## Quick Reference

### Google-style Docstring Template
```python
def function_name(arg1, arg2, arg3=None):
    """Brief one-line summary.

    Optional longer description with more details about what the function
    does and why it exists.

    Args:
        arg1: Description of arg1.
        arg2: Description of arg2.
        arg3: Optional description with default noted.

    Returns:
        Description of return value.

    Raises:
        ValueError: When invalid input is provided.

    Examples:
        >>> function_name(1, 2)
        3
    """
    pass
```

### Class Docstrings
```python
@attrs.define
class Series:
    """Data and predictions for a single image series.

    Longer description of the class purpose.

    Attributes:
        series_name: Unique identifier for the series.
        h5_path: Optional path to the HDF5 image file.
        primary_labels: Optional Labels for primary root predictions.

    Methods:
        load: Load predictions for this series.
        plot: Plot predictions on top of images.
    """
    series_name: str
    h5_path: Optional[str] = None
```

## Integration with Development Workflow

```bash
# Before committing
uv run black sleap_roots tests
uv run pydocstyle --convention=google sleap_roots/

# Before creating PR
uv run black --check sleap_roots tests && uv run pydocstyle --convention=google sleap_roots/

# If checks fail, fix and commit
uv run black sleap_roots tests
git add -u
git commit -m "style: apply Black formatting"
```