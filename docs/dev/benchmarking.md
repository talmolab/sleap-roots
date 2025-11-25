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

Benchmarks run automatically in two contexts:

### 1. Main Branch (Baseline Storage)

On pushes to `main` branch:

1. **When**: After merging PRs to main
2. **Where**: Ubuntu 22.04 runners (standardized environment)
3. **Output**: Baseline stored in `.benchmarks/baselines/main.json` (committed to repo)
4. **Purpose**: Create baseline for comparing future PRs
5. **History**: Results also saved to `.benchmarks/history/<date>.json` for trends

### 2. Pull Requests (Regression Detection)

On all pull requests:

1. **When**: Automatically on every PR
2. **What**: Compares PR benchmarks against main branch baseline
3. **Comment**: Posts/updates PR comment with comparison table
4. **Threshold**: Fails CI if any benchmark regresses >15%
5. **Artifacts**: Uploads `benchmark-results.json` and `benchmark-comparison.md` (30-day retention)

### Viewing CI benchmark results

**For main branch:**
1. Go to GitHub Actions for the commit
2. Find the "Performance Benchmarks" job
3. View the "Display benchmark results" step for summary
4. Check `.benchmarks/baselines/main.json` in repo for baseline

**For pull requests:**
1. Check the automated PR comment with benchmark comparison table
2. Review the "PR Benchmark Comparison" job status
3. Download artifacts if detailed analysis needed:
   ```bash
   gh run list --limit 5
   gh run download <run-id> --name pr-benchmark-results
   ```

## PR Workflow: Benchmark Regression Detection

When you open a PR, benchmarks run automatically and post a comparison comment.

### Understanding the PR Comment

The benchmark bot posts a comment like this:

```markdown
## 📊 Benchmark Results

| Benchmark | Main | PR | Change | Status |
|-----------|------|-----|--------|--------|
| test_dicot_pipeline_performance | 138.5ms | 142.1ms | +2.6% | ✅ |
| test_younger_monocot_pipeline_performance | 115.7ms | 110.3ms | -4.7% | ✅ |
| test_lateral_root_pipeline_performance | 202.3ms | 235.8ms | +16.5% | ⚠️ |
```

**Status indicators:**
- ✅ **OK**: Change within acceptable range (<15% regression)
- 🚀 **Improvement**: >5% faster than baseline
- ⚠️ **Regression**: >15% slower (fails CI)
- 🆕 **New**: Benchmark doesn't exist in baseline

### Regression Threshold

Default: **15%** (configurable via `BENCHMARK_MAX_REGRESSION` env var)

Why 15%?
- Accounts for CI runner variance (~5-10%)
- Catches meaningful regressions
- Avoids false positives from noise

### What to do about regressions

#### ⚠️ If your PR has regressions >15%:

1. **Understand why**:
   ```bash
   # Checkout your PR branch
   gh pr checkout <number>

   # Profile the slow benchmark
   uv run pytest tests/benchmarks/test_pipeline_performance.py::test_lateral_root_pipeline_performance --benchmark-only -v
   ```

2. **Determine if justified**:
   - Is this expected from your algorithm change?
   - Does accuracy improvement outweigh performance cost?
   - Is this a temporary regression (refactoring in progress)?

3. **Options**:
   - **Fix it**: Optimize the code to reduce regression
   - **Justify it**: Document why regression is acceptable in PR description
   - **Split it**: Defer optimization to follow-up PR if algorithm change is necessary

#### ✅ If regressions are acceptable:

Add explanation to PR description:
```markdown
## Performance Note

The 16.5% regression in `test_lateral_root_pipeline_performance` is expected because:
- New validation step adds safety checks for edge cases
- Accuracy improved from 92% to 98% on test dataset
- Performance still within acceptable range (235ms vs. 202ms on CI)
- Optimization tracked in issue #XXX
```

#### 🚀 If you improved performance:

Great! Note it in the PR description and the bot will highlight it automatically.

### For Reviewers

When reviewing PRs with benchmark results:

1. **Check the automated comment** - It appears shortly after benchmarks run
2. **Evaluate regressions** - Are they justified or concerning?
3. **Review artifacts** - Download for detailed analysis if needed:
   ```bash
   gh run download <run-id> --name pr-benchmark-results
   cat benchmark-comparison.md
   ```
4. **Request optimization** - If large unexplained regressions exist

See `.claude/commands/review-pr.md` for detailed review guidance.

### Baseline Management

Baselines are automatically managed:

- **Created**: When benchmarks run on main branch after PR merge
- **Updated**: On every push to main (with `[skip ci]` to avoid loops)
- **Stored**: In `.benchmarks/baselines/main.json` (committed to repo)
- **Cleaned**: Old commit-specific baselines removed after 90 days

**Note**: The first PR with benchmarks will show "No baseline" - this is expected. The baseline is created after merging to main.

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

### False positives: CI fails but regression seems small

Sometimes CI variance causes borderline failures:

**Symptoms:**
- Regression is 15-17% (just over threshold)
- Re-running the workflow gives different results
- Local benchmarks don't show regression

**Solutions:**

1. **Re-run the workflow**: Click "Re-run jobs" in GitHub Actions
   - If it passes on second run, it was likely CI variance

2. **Check recent main branch runs**: Compare against several recent baselines
   ```bash
   # View recent benchmark results on main
   git log main --all --grep="chore: update benchmark baselines"
   git show <commit-sha>:.benchmarks/baselines/main.json
   ```

3. **Run locally multiple times**: Check consistency
   ```bash
   # Run 5 times and compare
   for i in {1..5}; do
     uv run pytest tests/benchmarks/ --benchmark-only --benchmark-json=run_$i.json
   done
   ```

4. **Request threshold adjustment**: If a specific benchmark is consistently noisy
   - Document the variance in an issue
   - Consider per-benchmark thresholds (future feature)

### CI baseline is missing or outdated

**Symptom**: PR shows "No baseline" even though benchmarks exist on main

**Causes:**
- First time benchmarks are running (expected)
- Baseline commit failed to push
- `.benchmarks/` directory not in repo

**Solution:**
1. Check if `.benchmarks/baselines/main.json` exists on main branch
2. If missing, merge any PR to trigger baseline creation
3. The next PR will have a baseline to compare against

## Design Rationale

For the full design rationale behind the benchmark regression detection system, see the [OpenSpec proposal](https://github.com/talmolab/sleap-roots/tree/main/openspec/changes/benchmark-regression-detection).

Key design decisions:
- **15% threshold**: Balances catching real regressions vs. CI variance false positives
- **Automatic PR comments**: Provides immediate visibility without manual artifact downloads
- **Baseline storage in repo**: Ensures deterministic comparisons across CI runs
- **History tracking**: Enables future performance trend analysis and charts

## References

- [pytest-benchmark documentation](https://pytest-benchmark.readthedocs.io/)
- [Performance testing best practices](https://pythonspeed.com/articles/consistent-benchmarking/)
- [Berrigan et al. 2024 - Original performance metrics](https://doi.org/10.34133/plantphenomics.0175)
- [OpenSpec: Benchmark Regression Detection](https://github.com/talmolab/sleap-roots/tree/main/openspec/changes/benchmark-regression-detection)