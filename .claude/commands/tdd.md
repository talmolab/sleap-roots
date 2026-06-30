---
description: Test-driven development workflow for scientific software
---

# Test-Driven Development (TDD)

Structured TDD workflow for implementing features with tests first, ensuring scientific correctness and code quality.

## Purpose

TDD is critical for scientific software where correctness matters. This workflow ensures:

1. Requirements are captured as executable tests before implementation
2. Edge cases are considered upfront (NaN, empty arrays, single points, collinear nodes)
3. Trait calculations have known-answer test fixtures
4. Regressions in published trait values are caught immediately

## TDD Cycle

### Phase 1: Red (Write Failing Tests)

Write tests that define the expected behavior of the new feature:

```python
# tests/test_<module>.py

import pytest
import numpy as np


class TestNewTrait:
    """Tests for <trait/feature description>."""

    def test_basic_functionality(self, sample_pts):
        """Test that the trait computes for normal input."""
        result = compute_new_trait(sample_pts)
        assert result is not None
        # Assert specific expected values

    def test_edge_case_empty(self):
        """Test behavior with empty input."""
        result = compute_new_trait(np.empty((0, 2)))
        # Assert returns NaN / empty as documented (don't crash)

    def test_edge_case_nan_values(self, sample_pts_with_nans):
        """Test NaN handling for missing landmarks."""
        result = compute_new_trait(sample_pts_with_nans)
        # Assert NaN landmarks are handled correctly

    def test_known_answer(self):
        """Test with a hand-calculated fixture for correctness."""
        pts = np.array([[0.0, 0.0], [3.0, 4.0]])  # length 5 by construction
        result = compute_new_trait(pts)
        np.testing.assert_allclose(result, 5.0, rtol=1e-6)
```

### Phase 2: Confirm Red

Run the tests to confirm they fail as expected:

```bash
uv run pytest tests/test_<module>.py -v
```

All new tests should fail with `ImportError`, `AttributeError`, or `AssertionError` - not with unexpected errors. If tests fail for wrong reasons, fix the test setup first.

### Phase 3: Green (Implement the Feature)

Write the minimum code to make all tests pass:

```python
# sleap_roots/<module>.py

def compute_new_trait(pts):
    """Implement the trait calculation."""
    # Write implementation that satisfies the tests
    ...
```

Run tests again:

```bash
uv run pytest tests/test_<module>.py -v
```

All tests should pass. If not, fix the implementation (not the tests, unless the test itself was wrong).

### Phase 4: Refactor

Improve the implementation while keeping tests green:

1. Clean up code structure
2. Add type hints
3. Improve variable names
4. Extract helper functions if needed

Run tests after each refactor step:

```bash
uv run pytest tests/test_<module>.py -v
```

### Phase 5: Verify Quality

Run the full quality check suite:

```bash
# Formatting + docstring style (matches CI)
uv run black --check sleap_roots tests
uv run pydocstyle --convention=google sleap_roots/

# Full test suite (not just new tests)
uv run pytest tests/

# Coverage for the new module
uv run pytest --cov=sleap_roots --cov-report=term-missing tests/
```

### Phase 6: Commit

Commit with a descriptive message linking the test and implementation:

```bash
git add sleap_roots/<module>.py tests/test_<module>.py
git commit -m "feat: Add <feature description>

- Tests define expected behavior including edge cases
- Implementation satisfies all test cases
- Known-answer fixtures verify trait correctness"
```

## Scientific Testing Patterns

### Known-Answer Tests

For trait calculations, use hand-calculated or reference values:

```python
def test_root_length_known_answer(self):
    """Verify length with hand-calculated value."""
    pts = np.array([[0.0, 0.0], [0.0, 3.0], [4.0, 3.0]])  # 3 + 4 = 7
    result = get_root_length(pts)
    np.testing.assert_allclose(result, 7.0, atol=1e-10)
```

### Boundary Condition Tests

Test at the edges of valid input:

```python
def test_single_point(self):
    """A single landmark has no length."""
    pts = np.array([[1.0, 2.0]])
    result = get_root_length(pts)
    assert result == 0.0 or np.isnan(result)

def test_collinear_points(self):
    """Angles for collinear points must not return NaN (#142)."""
    pts = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
    result = get_root_angle(pts)
    assert not np.isnan(result)
```

### Numerical Stability Tests

Verify calculations are stable with extreme values:

```python
def test_large_coordinates(self):
    """Test numerical stability with large pixel coordinates."""
    pts = np.array([[1e6, 1e6], [1e6, 1e6 + 3.0]])
    result = get_root_length(pts)
    np.testing.assert_allclose(result, 3.0, atol=1e-3)
```

## Fixture Patterns

### Parametrized Tests

```python
@pytest.mark.parametrize("pipeline_cls", [
    DicotPipeline,
    YoungerMonocotPipeline,
    OlderMonocotPipeline,
])
def test_pipeline_runs(self, sample_series, pipeline_cls):
    """Each pipeline produces a traits frame without crashing."""
    traits = pipeline_cls().compute_plant_traits(sample_series)
    assert traits is not None
```

### Shared Fixtures

Reuse the fixtures and test data already wired in `tests/fixtures/` and `tests/data/`
(loaded via Git LFS). Prefer round-tripping through a real `.slp` via `sio` over
constructing `Labels` in memory when a test is meant to regression-test the loading path.

## Integration

- Run `/lint` during Phase 5 to check formatting and docstring style
- Run `/coverage` to verify test coverage
- Run `/run-ci-locally` before committing to ensure full CI passes
- Use `/test` and `/debug-test` to run and diagnose failing tests
