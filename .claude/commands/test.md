# Run Tests

Execute the pytest test suite with various options and filters.

## Commands

```bash
# Run all tests
pytest tests/

# Run all tests with verbose output
pytest -v tests/

# Run tests for a specific module
pytest tests/test_lengths.py

# Run a specific test function
pytest tests/test_lengths.py::test_get_root_lengths

# Run tests matching a pattern
pytest -k "primary_root" tests/

# Run tests with output (print statements visible)
pytest -s tests/

# Run tests and stop on first failure
pytest -x tests/

# Run previously failed tests
pytest --lf tests/

# Run tests in parallel (if pytest-xdist installed)
pytest -n auto tests/
```

## Understanding Test Output

### Successful test run:
```
================================ test session starts =================================
platform darwin -- Python 3.11.0, pytest-7.4.0, pluggy-1.0.0
rootdir: /Users/elizabethberrigan/repos/sleap-roots
collected 87 items

tests/test_lengths.py ........                                              [  9%]
tests/test_angles.py .......                                                [ 17%]
tests/test_tips.py .....                                                    [ 23%]
tests/test_bases.py ....                                                    [ 27%]
...

================================ 87 passed in 12.34s =================================
```

### Failed test:
```
================================== FAILURES ==========================================
____________________________ test_get_root_lengths __________________________________

    def test_get_root_lengths():
        points = np.array([[0, 0], [3, 4]])
>       assert get_root_lengths(points) == 5.0
E       AssertionError: assert 4.999999 == 5.0

tests/test_lengths.py:42: AssertionError
========================== short test summary info ===================================
FAILED tests/test_lengths.py::test_get_root_lengths - AssertionError: assert 4.9...
========================== 1 failed, 86 passed in 12.34s =============================
```

## Test Organization

```
tests/
├── conftest.py              # Pytest fixtures and configuration
├── fixtures/                # Reusable test data
├── data/                    # Real SLEAP data (Git LFS)
│   ├── canola_7do/         # Dicot test data
│   ├── soy_6do/            # Dicot test data
│   ├── rice_3do/           # Monocot test data
│   ├── rice_10do/          # Monocot test data
│   └── multiple_arabidopsis_11do/  # Multi-plant test data
├── test_lengths.py          # Tests for sleap_roots/lengths.py
├── test_angles.py           # Tests for sleap_roots/angles.py
├── test_tips.py             # Tests for sleap_roots/tips.py
├── test_bases.py            # Tests for sleap_roots/bases.py
├── test_convhull.py         # Tests for sleap_roots/convhull.py
├── test_ellipse.py          # Tests for sleap_roots/ellipse.py
├── test_networklength.py    # Tests for sleap_roots/networklength.py
├── test_scanline.py         # Tests for sleap_roots/scanline.py
├── test_series.py           # Tests for sleap_roots/series.py
├── test_points.py           # Tests for sleap_roots/points.py
├── test_summary.py          # Tests for sleap_roots/summary.py
└── test_trait_pipelines.py  # Tests for sleap_roots/trait_pipelines.py
```

## Cross-Platform Testing

Tests run on multiple platforms in CI:
- **Ubuntu 22.04** (primary coverage platform)
- **Windows 2022**
- **macOS latest**

Ensure tests pass locally before pushing, especially if modifying:
- File path handling
- Line endings
- Platform-specific code

## Test Data (Git LFS)

Test data is stored with Git LFS (Large File Storage):

```bash
# Ensure LFS is installed and data is pulled
git lfs pull

# Check LFS status
git lfs ls-files
```

Test data includes:
- `.slp` files (SLEAP predictions)
- `.h5` files (HDF5 image series)
- `.csv` files (expected trait outputs)

## Common Testing Workflows

### 1. Running tests during development
```bash
# Run tests for the module you're working on
pytest tests/test_lengths.py -v

# Re-run on file change (requires pytest-watch)
ptw tests/test_lengths.py
```

### 2. Testing a new feature
```bash
# Run tests with verbose output to see details
pytest -v tests/test_new_feature.py

# Run with coverage to ensure new code is tested
pytest --cov=sleap_roots.new_module tests/test_new_feature.py
```

### 3. Debugging a failing test
```bash
# Run with output visible (print statements)
pytest -s tests/test_failing.py

# Drop into debugger on failure (requires pdb or ipdb)
pytest --pdb tests/test_failing.py

# Run only the failing test
pytest tests/test_failing.py::test_specific_function
```

### 4. Before creating a PR
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=sleap_roots --cov-report=term-missing tests/

# Ensure linting passes
black --check sleap_roots tests
pydocstyle --convention=google sleap_roots/
```

## Pytest Fixtures

Common fixtures defined in `tests/conftest.py`:

```python
@pytest.fixture
def sample_primary_points():
    """Fixture providing sample primary root points."""
    return np.array([[0, 0], [10, 0], [20, 5], [30, 10]])

@pytest.fixture
def sample_series():
    """Fixture providing a sample Series object."""
    return Series.load(
        series_name="test_series",
        primary_path="tests/data/canola_7do/919QDUH.primary.slp",
        h5_path="tests/data/canola_7do/919QDUH.h5"
    )
```

Use fixtures in your tests:
```python
def test_with_fixture(sample_primary_points):
    length = get_root_lengths(sample_primary_points)
    assert length > 0
```

## Writing Good Tests

### 1. Test naming convention
```python
# Good: Descriptive test names
def test_get_root_lengths_with_empty_array():
    pass

def test_get_root_lengths_with_single_point():
    pass

# Bad: Vague test names
def test_lengths():
    pass

def test_case1():
    pass
```

### 2. Test structure (Arrange, Act, Assert)
```python
def test_get_root_lengths():
    # Arrange: Set up test data
    points = np.array([[0, 0], [3, 4], [6, 8]])

    # Act: Execute the function
    result = get_root_lengths(points)

    # Assert: Verify the result
    assert np.isclose(result, 10.0)
```

### 3. Test edge cases
```python
def test_get_root_lengths_empty_array():
    """Test with empty input."""
    points = np.array([])
    assert get_root_lengths(points) == 0.0

def test_get_root_lengths_single_point():
    """Test with single point."""
    points = np.array([[0, 0]])
    assert get_root_lengths(points) == 0.0

def test_get_root_lengths_collinear_points():
    """Test with collinear points."""
    points = np.array([[0, 0], [1, 0], [2, 0]])
    assert np.isclose(get_root_lengths(points), 2.0)
```

## CI Testing

GitHub Actions runs tests on:
- Multiple platforms (Ubuntu, Windows, macOS)
- Python 3.11 (primary version)
- With coverage reporting (Ubuntu only)

CI configuration: `.github/workflows/ci.yml`

```yaml
- name: Test with pytest (with coverage)
  run: |
    pytest --cov=sleap_roots --cov-report=xml tests/
```

## Tips

1. **Run tests frequently**: Don't wait until the end to test
2. **Test one thing at a time**: Each test should verify one specific behavior
3. **Use descriptive assertions**: Include messages to help debug failures
4. **Mock external dependencies**: Don't rely on network, filesystem, etc.
5. **Keep tests fast**: Slow tests won't get run as often
6. **Test error cases**: Not just the happy path
7. **Use parametrize for similar tests**: Reduce code duplication

## Pytest Configuration

Configuration in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]
```