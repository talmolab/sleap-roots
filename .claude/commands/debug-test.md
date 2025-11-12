# Debug Test

Run a specific test with debugging tools and verbose output to troubleshoot failures.

## Quick Start

```bash
# Debug a specific test
```

This will:
1. Run the test with maximum verbosity
2. Show full tracebacks and print statements
3. Drop into debugger (pdb) on failure
4. Display fixture values being used
5. Show test data paths and file contents

## Usage Patterns

### Debug Single Test Function

```bash
# Debug specific test
uv run pytest tests/test_lengths.py::test_get_root_lengths -vv -s --pdb
```

### Debug All Tests in Module

```bash
# Debug all tests in file
uv run pytest tests/test_lengths.py -vv -s --pdb
```

### Debug Tests Matching Pattern

```bash
# Debug tests matching keyword
uv run pytest -k "primary_root" -vv -s --pdb
```

## Command Flags Explained

### `-vv` (Very Verbose)
Shows detailed test information:
- Full test names
- Full assertion diffs
- Fixture setup/teardown

```
tests/test_lengths.py::test_get_root_lengths PASSED [100%]
    Points: [[0, 0], [3, 4], [6, 8]]
    Expected: 10.0
    Got: 10.0
    ✓ Test passed
```

### `-s` (No Capture)
Shows `print()` statements and stdout:

```python
def test_example():
    print("Debug: points shape:", points.shape)  # This will be visible
    assert True
```

### `--pdb` (Drop to Debugger)
Starts Python debugger on test failure:

```python
def test_failing():
    result = calculate_length(bad_data)
    assert result == 5.0  # Fails here

# Drops into pdb:
> /path/to/test.py(42)test_failing()
-> assert result == 5.0
(Pdb)
```

### `--pdbcls=IPython.terminal.debugger:TerminalPdb` (IPython Debugger)
Uses IPython's enhanced debugger (if installed):
- Syntax highlighting
- Tab completion
- Better history

## Debugger Commands

When test drops into pdb:

```
(Pdb) p result              # Print variable
10.5

(Pdb) pp locals()           # Pretty-print all local variables
{'points': array([[0, 0], [3, 4]]),
 'result': 10.5,
 'expected': 5.0}

(Pdb) l                     # List code around current line
 39         def test_failing():
 40             result = calculate_length(bad_data)
 41  ->         assert result == 5.0
 42

(Pdb) c                     # Continue execution

(Pdb) q                     # Quit debugger
```

## Advanced Debugging Options

### Show Fixture Values

```bash
# Show which fixtures are being used
uv run pytest tests/test_lengths.py::test_example --fixtures-per-test
```

Output:
```
test_example uses fixtures:
  - sample_points (from conftest.py)
  - tmp_path (pytest builtin)
```

### Show Setup/Teardown

```bash
# Show fixture setup and teardown
uv run pytest tests/test_lengths.py -vv --setup-show
```

Output:
```
SETUP    F sample_points
tests/test_lengths.py::test_example PASSED
TEARDOWN F sample_points
```

### Capture Logs

```bash
# Show log output during test
uv run pytest tests/test_lengths.py -vv --log-cli-level=DEBUG
```

### Stop on First Failure

```bash
# Stop immediately on first failure
uv run pytest tests/ -x --pdb
```

### Last Failed Test

```bash
# Re-run only the test that failed last time
uv run pytest --lf -vv -s --pdb
```

## Common Debugging Scenarios

### Scenario 1: Test Fails with Assertion Error

```python
def test_length_calculation():
    points = np.array([[0, 0], [3, 4]])
    result = get_root_lengths(points)
    assert result == 5.0  # Fails: AssertionError: assert 5.0000001 == 5.0
```

**Debug:**
```bash
uv run pytest tests/test_lengths.py::test_length_calculation -vv --pdb
```

**In debugger:**
```
(Pdb) p result
5.0000001

(Pdb) p points
array([[0, 0], [3, 4]])

(Pdb) # Ah, floating point precision issue!
(Pdb) import numpy as np
(Pdb) np.isclose(result, 5.0)
True
```

**Fix:** Use `np.isclose()` instead of `==`

### Scenario 2: Test Fails with Unexpected Data

```python
def test_lateral_count():
    series = load_test_series()
    pipeline = DicotPipeline()
    traits = pipeline.compute_traits(series, frame_index=0)
    assert traits['lateral_count'] == 12  # Fails: got 0
```

**Debug:**
```bash
uv run pytest tests/test_trait_pipelines.py::test_lateral_count -vv -s --pdb
```

**In debugger:**
```
(Pdb) p traits
{'lateral_count': 0, 'primary_length': 145.2, ...}

(Pdb) p series.lateral_labels
None  # Ah! Lateral data not loaded

(Pdb) p series.lateral_path
None  # Path not provided
```

**Fix:** Provide `lateral_path` when loading series

### Scenario 3: Test Passes Locally, Fails in CI

```bash
# Run with same Python version as CI
uv run pytest tests/test_angles.py -vv --pdb

# Check test data
uv run pytest tests/test_angles.py -vv -s
# Add print statements to check file paths
```

**Common causes:**
- Git LFS data not pulled
- Platform-specific path issues
- Floating point differences across platforms

### Scenario 4: Fixture Not Working

```bash
# Show fixture values
uv run pytest tests/test_series.py::test_load -vv --fixtures-per-test

# See fixture source
uv run pytest --fixtures | grep sample_series
```

## Inspecting Test Data

### Show Test Data Paths

Add to test:
```python
def test_example(sample_series):
    print(f"Primary path: {sample_series.primary_path}")
    print(f"H5 path: {sample_series.h5_path}")
    # ... rest of test
```

Run with `-s`:
```bash
uv run pytest tests/test_series.py::test_example -s
```

### Verify Test Data Loaded

```python
def test_data_loading():
    series = Series.load(
        series_name="test",
        primary_path="tests/data/canola_7do/919QDUH.primary.slp"
    )

    # Debug output
    print(f"Labels: {series.primary_labels}")
    print(f"Frame count: {len(series)}")
    print(f"First frame points: {series.get_primary_points(0)}")

    assert series.primary_labels is not None
```

## Debugging Tips

### 1. Add Strategic Prints

```python
def test_complex_computation():
    points = get_points()
    print(f"Points shape: {points.shape}")

    filtered = filter_points(points)
    print(f"After filtering: {filtered.shape}")

    result = compute_trait(filtered)
    print(f"Result: {result}")

    assert result > 0
```

Run with `-s` to see all prints.

### 2. Use pytest.set_trace()

```python
import pytest

def test_with_breakpoint():
    points = get_points()

    # Drop into debugger here
    pytest.set_trace()

    result = compute_trait(points)
    assert result > 0
```

### 3. Parametrize for Debugging

```python
@pytest.mark.parametrize("frame_idx", [0, 50, 99])
def test_all_frames(frame_idx):
    series = load_series()
    traits = compute_traits(series, frame_idx)
    assert traits is not None
```

Find which frame fails:
```bash
uv run pytest tests/test_example.py::test_all_frames -vv
```

### 4. Mark Tests for Debugging

```python
@pytest.mark.debug
def test_problematic():
    # Test that needs investigation
    pass
```

Run only debug tests:
```bash
uv run pytest -m debug -vv --pdb
```

## Test Data Debugging

### Check Git LFS Status

```bash
# Verify test data downloaded
git lfs ls-files tests/data/

# Should show actual files, not pointers
ls -lh tests/data/canola_7do/919QDUH.h5
# Should be ~45 MB, not ~130 bytes
```

### Validate Test Files Can Load

```python
def test_data_loads():
    \"\"\"Verify test data files are valid.\"\"\"
    import h5py
    import sleap_io as sio

    # Check H5
    with h5py.File("tests/data/canola_7do/919QDUH.h5") as f:
        assert 'video' in f
        print(f"Video shape: {f['video'].shape}")

    # Check SLP
    labels = sio.load_slp("tests/data/canola_7do/919QDUH.primary.slp")
    assert len(labels) > 0
    print(f"Loaded {len(labels)} frames")
```

## Performance Debugging

### Profile Slow Tests

```bash
# Install pytest-profiling
uv add --dev pytest-profiling

# Profile test
uv run pytest tests/test_slow.py --profile
```

### Time Individual Tests

```bash
# Show test durations
uv run pytest tests/ --durations=10
```

Output:
```
slowest 10 test durations:
4.23s    test_batch_processing
2.15s    test_large_dataset
1.87s    test_convex_hull
...
```

## Configuration

### pytest.ini or pyproject.toml

```toml
[tool.pytest.ini_options]
# Always use verbose mode for debugging
addopts = "-vv"

# Show local variables in tracebacks
showlocals = true

# Use IPython debugger if available
addopts = "--pdbcls=IPython.terminal.debugger:TerminalPdb"
```

## Debugging Test Fixtures

### Show Available Fixtures

```bash
uv run pytest --fixtures
```

### Debug Fixture Itself

```python
# In conftest.py
@pytest.fixture
def sample_series():
    print("Setting up sample_series fixture")
    series = Series.load(...)
    print(f"Loaded series: {series.series_name}")
    yield series
    print("Tearing down sample_series fixture")
```

Run with `-s --setup-show` to see fixture lifecycle.

## Common Issues

### Issue: "No module named 'sleap_roots'"

Package not installed:
```bash
uv sync
```

### Issue: "FileNotFoundError: test data not found"

Git LFS data not pulled:
```bash
git lfs pull
```

### Issue: "Debugger doesn't start"

Missing pdb or IPython:
```bash
uv add --dev ipython ipdb
```

### Issue: "Too much output"

Limit verbosity:
```bash
# Just run one test
uv run pytest tests/test_file.py::test_specific -vv --pdb

# Stop on first failure
uv run pytest -x --pdb
```

## Related Commands

- `/test` - Run tests normally
- `/coverage` - Run tests with coverage
- `/run-ci-locally` - Run all CI checks
- `/validate-env` - Check environment setup

## Tips

1. **Start with `-vv -s`**: See everything first, then add `--pdb` if needed
2. **Use `--lf`**: Re-run only failed tests for faster iteration
3. **Print liberally**: Add debug prints, they're easy to remove later
4. **Check test data**: Many failures are due to missing/corrupted LFS data
5. **Read tracebacks carefully**: Full traceback often reveals the issue
6. **Use breakpoints**: `pytest.set_trace()` is your friend
7. **Isolate tests**: Debug one test at a time, not the whole suite