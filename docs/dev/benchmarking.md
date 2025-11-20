# Performance Benchmarking Guide

This guide explains how to run, interpret, and add performance benchmarks for sleap-roots trait extraction pipelines.

## Overview

sleap-roots uses [pytest-benchmark](https://pytest-benchmark.readthedocs.io/) to measure trait extraction performance. Benchmarks provide:

- **Statistical rigor**: Multiple iterations with mean, stddev, min, max, percentiles
- **Regression detection**: Track performance changes over time
- **Optimization guidance**: Identify bottlenecks for targeted improvements
- **User expectations**: Document actual performance on standard hardware

## Running Benchmarks Locally

### Run all benchmarks

```bash
uv run pytest tests/benchmarks/ --benchmark-only
```

### Run specific benchmark

```bash
uv run pytest tests/benchmarks/test_pipeline_performance.py::TestSinglePlantPipelines::test_dicot_pipeline_performance --benchmark-only
```

### Compare with previous run

```bash
# First run - save baseline
uv run pytest tests/benchmarks/ --benchmark-only --benchmark-save=baseline

# Make code changes...

# Compare against baseline
uv run pytest tests/benchmarks/ --benchmark-only --benchmark-compare=baseline
```

### Generate histogram

```bash
uv run pytest tests/benchmarks/ --benchmark-only --benchmark-histogram=benchmark_hist
```

This creates `benchmark_hist.svg` showing performance distribution.

## Interpreting Results

### Understanding the output

```
----------------------------------------- benchmark: 7 tests ----------------------------------------
Name (time in ms)                                    Min       Max      Mean    StdDev    Median
----------------------------------------------------------------------------------------------------
test_dicot_pipeline_performance                  125.32    156.78    138.45      8.23    136.92
test_younger_monocot_pipeline_performance        102.45    128.34    115.67      7.12    114.23
test_multiple_dicot_pipeline_performance         456.78    523.12    489.34     18.45    487.56
----------------------------------------------------------------------------------------------------
```

**Key metrics:**
- **Mean**: Average execution time (most important for typical performance)
- **StdDev**: Standard deviation (lower is more consistent)
- **Min/Max**: Best and worst case times
- **Median**: Middle value (less affected by outliers)

### What's good performance?

Based on the published paper (Berrigan et al., 2024):

| Pipeline Type | Expected Time | Hardware |
|---------------|---------------|----------|
| Single plant pipelines | 0.1-0.5s per plant | GitHub Actions Ubuntu 22.04 |
| Multiple plant pipelines | 0.5-2s (varies with plant count) | GitHub Actions Ubuntu 22.04 |

**Note**: Local performance may vary based on your CPU, available memory, and system load.

### Performance vs. Profiling

- **Benchmarks** (what we have): Measure end-to-end execution time with statistical rigor
- **Profiling** (different tools): Identify *which* functions are slow within a benchmark

If a benchmark shows poor performance, use profiling tools to investigate:

```bash
# Profile a slow pipeline
uv run python -m cProfile -o profile.stats -c "
import sleap_roots as sr
series = sr.Series.load('plant', primary_path='primary.slp', lateral_path='lateral.slp')
pipeline = sr.DicotPipeline()
pipeline.fit_series(series)
"

# Analyze profile
uv run python -m pstats profile.stats
```

## Adding New Benchmarks

### 1. Identify what to benchmark

Good candidates:
- New pipeline classes
- Alternative implementations of existing algorithms
- Performance-critical trait computations

**Don't benchmark:**
- Individual helper functions (too granular)
- I/O operations (data loading - too variable)
- Plotting/visualization (not core workflow)

### 2. Write the benchmark

Add to `tests/benchmarks/test_pipeline_performance.py`:

```python
def test_my_new_pipeline_performance(
    self,
    benchmark,
    test_data_fixture,  # Reuse existing fixtures
):
    """Benchmark MyNewPipeline.fit_series().

    Dataset: Description of test data
    Expected: ~X.X-X.Xs per plant on GitHub Actions runners
    """
    series = sr.Series.load(
        "my_test_plant",
        primary_path=test_data_fixture,
    )
    pipeline = sr.MyNewPipeline()

    # benchmark() calls the function multiple times
    result = benchmark(pipeline.fit_series, series)

    # Sanity check: ensure it worked
    assert result is not None
    assert "my_expected_trait" in result
```

### 3. Run and document

```bash
# Run your new benchmark
uv run pytest tests/benchmarks/test_pipeline_performance.py::test_my_new_pipeline_performance --benchmark-only

# Document expected performance in the docstring
# Update docs/api/core/pipelines.md if it's a new pipeline
```

## CI Integration

Benchmarks run automatically on pushes to `main` branch:

1. **When**: Only on `main` (not PRs) to avoid noise
2. **Where**: Ubuntu 22.04 runners (standardized environment)
3. **Output**: JSON results uploaded as artifact (30-day retention)
4. **Purpose**: Track performance trends over time

### Viewing CI benchmark results

1. Go to GitHub Actions for the commit
2. Find the "Performance Benchmarks" job
3. View the "Display benchmark results" step for summary
4. Download `benchmark-results.json` artifact for detailed analysis

## Best Practices

### DO:
- ✅ Use real test data (from `tests/data/`)
- ✅ Document expected performance in docstrings
- ✅ Include hardware context (GitHub Actions vs. local)
- ✅ Focus on end-user-facing operations (`fit_series()`)
- ✅ Run multiple iterations (pytest-benchmark does this automatically)

### DON'T:
- ❌ Optimize for benchmarks at the expense of correctness
- ❌ Benchmark unrealistic inputs (edge cases)
- ❌ Compare benchmarks across different hardware
- ❌ Expect identical results every run (some variance is normal)
- ❌ Fail CI on small regressions (runner variance can cause false positives)

## Troubleshooting

### Benchmarks are slow locally

```bash
# Reduce iterations for faster feedback
uv run pytest tests/benchmarks/ --benchmark-only --benchmark-min-rounds=3
```

### High variance in results

Possible causes:
- Background processes consuming CPU
- Thermal throttling on laptop
- Small dataset (noise dominates signal)

Solutions:
- Close unnecessary applications
- Use `--benchmark-warmup=on` to stabilize
- Use larger/more representative test data

### Benchmark fails but test passes

Benchmarks call the same code as regular tests. If a benchmark fails:

1. Run the regular test: `uv run pytest tests/test_trait_pipelines.py::test_dicot_pipeline`
2. If regular test passes, the benchmark setup might be wrong (fixture issue)
3. Check that benchmark is using correct test data

## References

- [pytest-benchmark documentation](https://pytest-benchmark.readthedocs.io/)
- [Performance testing best practices](https://pythonspeed.com/articles/consistent-benchmarking/)
- [Berrigan et al. 2024 - Original performance metrics](https://doi.org/10.34133/plantphenomics.0175)