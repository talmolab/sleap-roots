# Test Coverage Analysis

Run tests with coverage analysis to identify untested code and ensure quality.

## Commands

```bash
# Run all tests with coverage
uv run pytest --cov=sleap_roots --cov-report=xml tests/

# Run tests with coverage and HTML report
uv run pytest --cov=sleap_roots --cov-report=html tests/

# Run tests for specific module with coverage
uv run pytest --cov=sleap_roots.lengths tests/test_lengths.py

# Run all tests with verbose coverage output
uv run pytest --cov=sleap_roots --cov-report=term-missing tests/
```

## Understanding Coverage Results

After running coverage, you'll see a table like:

```
Name                          Stmts   Miss  Cover   Missing
-----------------------------------------------------------
sleap_roots/lengths.py           45      0   100%
sleap_roots/angles.py            38      2    95%   67-68
sleap_roots/trait_pipelines.py  156      8    95%   234-241
-----------------------------------------------------------
TOTAL                           892     12    99%
```

### What the columns mean:
- **Stmts**: Total number of executable statements
- **Miss**: Number of statements not executed by tests
- **Cover**: Percentage of statements covered
- **Missing**: Line numbers of uncovered statements

## Coverage Goals

- **Target**: Full coverage on core trait computation modules
- **Current**: ~84% overall, tracked via Codecov badge on each PR
- **CI Requirement**: Coverage must be maintained (not decrease)
- **Badge**: Auto-updates in README.md and docs/index.md

### Priority Areas
1. **High priority**: Core trait computations (`lengths.py`, `angles.py`, `tips.py`, etc.)
2. **Medium priority**: Pipeline classes (`trait_pipelines.py`)
3. **Lower priority**: Utility functions and visualization code

## Viewing HTML Coverage Report

```bash
# Generate HTML report
uv run pytest --cov=sleap_roots --cov-report=html tests/

# Open in browser (macOS)
open htmlcov/index.html

# Open in browser (Linux)
xdg-open htmlcov/index.html

# Open in browser (Windows)
start htmlcov/index.html
```

The HTML report shows:
- Line-by-line coverage highlighting
- Which branches were/weren't taken
- Easy navigation to untested code

## Identifying Untested Code

Use the `--cov-report=term-missing` flag to see exactly which lines need tests:

```bash
uv run pytest --cov=sleap_roots --cov-report=term-missing tests/test_lengths.py

# Output shows missing lines:
sleap_roots/lengths.py    95%   67-68, 102-105
```

Then add tests for those specific lines.

## Coverage in CI

CI runs coverage on Ubuntu:

```yaml
uv run pytest --cov=sleap_roots --cov-report=xml tests/
```

Results are uploaded to Codecov for tracking over time.

## Common Scenarios

### 1. Adding a new module
```bash
# Run coverage for just your new module
uv run pytest --cov=sleap_roots.new_module tests/test_new_module.py

# Aim for 100% coverage on new code
```

### 2. Fixing a bug
```bash
# Check coverage before fix
uv run pytest --cov=sleap_roots.lengths tests/test_lengths.py

# Add regression test
# Re-run to ensure new test increases coverage
uv run pytest --cov=sleap_roots.lengths tests/test_lengths.py
```

### 3. Refactoring
```bash
# Ensure coverage doesn't decrease
uv run pytest --cov=sleap_roots --cov-report=term-missing tests/

# Coverage should stay the same or increase
```

## Test Data Location

Tests use real SLEAP prediction files and HDF5 images stored in `tests/data/`:

- `tests/data/canola_7do/` - Canola dicot samples
- `tests/data/soy_6do/` - Soybean samples
- `tests/data/rice_3do/`, `tests/data/rice_10do/` - Rice monocot samples
- `tests/data/multiple_arabidopsis_11do/` - Multi-plant samples

**Note**: Test data is stored with Git LFS (Large File Storage).

## Tips for Writing Tests

1. **Test edge cases**: Empty arrays, single points, extreme values
2. **Test error conditions**: Invalid inputs, missing data
3. **Test biological scenarios**: Different root types, developmental stages
4. **Use fixtures**: Defined in `tests/conftest.py` for reusable setup
5. **Test cross-platform**: Ensure tests pass on Ubuntu, Windows, macOS

## Coverage Configuration

Coverage settings in `pyproject.toml`:

```toml
[tool.coverage.run]
source = ["sleap_roots"]
omit = ["tests/*"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise NotImplementedError",
]
```

## Quick Workflow

```bash
# 1. Run tests with coverage
uv run pytest --cov=sleap_roots --cov-report=html tests/

# 2. Open HTML report
open htmlcov/index.html

# 3. Identify untested lines (red highlighting)

# 4. Write tests for those lines

# 5. Re-run coverage to verify
uv run pytest --cov=sleap_roots --cov-report=html tests/

# 6. Commit when coverage is satisfactory
git add tests/
git commit -m "test: add coverage for edge cases"
```