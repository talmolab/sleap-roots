# Design: Performance Benchmarking System

## Context

The sleap-roots paper reports specific performance metrics for trait extraction pipelines, but these are not systematically validated. As the codebase evolves, there's risk of performance regressions going unnoticed. We need automated benchmarking to:

1. Validate published performance claims
2. Detect regressions in future development
3. Provide users with accurate performance expectations
4. Enable performance-driven optimization decisions

**Constraints:**
- Must not slow down standard test runs (benchmarks should be opt-in)
- Must work in CI environment (GitHub Actions runners)
- Must provide statistically meaningful results (not single measurements)
- Should reuse existing test data to minimize maintenance

**Stakeholders:**
- Researchers using sleap-roots for high-throughput phenotyping
- Contributors optimizing trait computation algorithms
- Users evaluating whether sleap-roots meets their performance needs

## Goals / Non-Goals

**Goals:**
- Measure end-to-end trait extraction time for all 7 pipeline classes
- Provide statistical measures (mean, stddev, outliers) for reliability
- Store historical benchmark data for regression detection
- Document expected performance on standard hardware
- Enable local benchmarking for optimization work

**Non-Goals:**
- Fine-grained profiling of individual functions (use separate profiling tools for that)
- Performance optimization in this change (only measurement infrastructure)
- Automatic performance regression blocking (CI will warn but not fail)
- Benchmarking SLEAP inference (out of scope - this project only does trait extraction)

## Decisions

### Decision 1: Use pytest-benchmark

**Rationale:**
- Native pytest integration (we already use pytest)
- Provides statistical analysis (mean, stddev, min, max, percentiles)
- Handles warmup rounds and multiple iterations automatically
- Can save/compare historical results
- Widely used in scientific Python ecosystem

**Alternatives considered:**
- `timeit`: Too low-level, no statistical analysis, manual iteration handling
- `perfplot`: Designed for scaling analysis, overkill for our needs
- `py-spy` or `cProfile`: Profilers, not benchmarking tools (different use case)
- Custom timing code: Reinventing the wheel, error-prone

**Trade-offs:**
- ✅ Pro: Statistical rigor, automatic outlier detection
- ✅ Pro: JSON export for CI artifact storage
- ✅ Pro: Built-in comparison mode for regression detection
- ⚠️  Con: Adds dependency (~100KB package)
- ⚠️  Con: Learning curve for interpreting statistics

### Decision 2: Separate Benchmark Directory

**Structure:**
```
tests/
├── benchmarks/          # NEW: Performance benchmarks
│   ├── __init__.py
│   ├── conftest.py     # Benchmark-specific fixtures
│   └── test_pipeline_performance.py
├── fixtures/            # Existing: Shared test data
├── data/                # Existing: Test datasets (Git LFS)
└── test_*.py            # Existing: Unit/integration tests
```

**Rationale:**
- Keeps benchmarks separate from regular tests (different purpose)
- Allows running benchmarks independently: `pytest tests/benchmarks/`
- Prevents accidental execution during normal test runs
- Makes it obvious where to add new performance tests

**Alternatives considered:**
- Same directory as unit tests: Would require markers to exclude from normal runs, messy
- Top-level `benchmarks/` directory: Separates from test infrastructure (fixtures, conftest)

### Decision 3: CI Benchmark Job Configuration

**When to run:**
- Only on pushes to `main` branch (not PRs)
- Scheduled weekly run (optional, for tracking trends)

**Why not on PRs:**
- Adds ~2-3 minutes to CI time
- Results can vary due to runner load
- Manual benchmark comparison is better for optimization PRs
- Reduces CI cost/resource usage

**Output:**
- Upload `benchmark-results.json` as artifact
- Print summary table in CI logs
- Store in repository for historical tracking (future enhancement)

**Configuration:**
```yaml
benchmark:
  name: Performance Benchmarks
  runs-on: ubuntu-22.04  # Standardize hardware
  if: github.ref == 'refs/heads/main'  # Only main branch
  steps:
    - name: Run benchmarks
      run: uv run pytest tests/benchmarks/ --benchmark-only --benchmark-json=benchmark-results.json
    - name: Upload results
      uses: actions/upload-artifact@v4
      with:
        name: benchmark-results
        path: benchmark-results.json
```

**Alternatives considered:**
- Run on every PR: Too noisy, slows down development
- Fail CI on regression: False positives due to runner variance
- Compare against baseline automatically: Complex, requires storage solution

### Decision 4: Benchmark Scope

**What to measure:**
- **Entry point:** `Pipeline.fit_series()` method (end-to-end trait extraction)
- **Input:** Real test data from `tests/data/` (canola, soy, rice)
- **Iterations:** Let pytest-benchmark decide (typically 5-10 rounds)
- **Warmup:** 1-2 rounds to load data into memory

**What NOT to measure:**
- Individual trait functions (too granular, would be 100+ benchmarks)
- Data loading time (I/O dependent, not trait computation)
- Visualization/plotting (not part of core workflow)

**Example benchmark:**
```python
def test_dicot_pipeline_performance(benchmark, canola_series):
    """Benchmark DicotPipeline.fit_series() on single canola plant."""
    pipeline = sr.DicotPipeline()
    result = benchmark(pipeline.fit_series, canola_series)
    assert result is not None  # Sanity check
```

**Rationale:**
- Matches paper's reported metrics (per-plant processing time)
- Meaningful to end users (what they care about is total time)
- Stable across code changes (high-level API unlikely to change)

## Risks / Trade-offs

### Risk 1: Runner Variance

**Issue:** GitHub Actions runners have variable performance due to:
- Shared infrastructure (noisy neighbors)
- Different CPU/memory configurations over time
- Thermal throttling, background processes

**Mitigation:**
- Run multiple iterations (pytest-benchmark does this)
- Focus on mean ± stddev, not single measurements
- Document that benchmarks are for trend detection, not absolute guarantees
- Use `--benchmark-disable-gc` for more stable results

**Acceptance:** Some variance is acceptable; we care about order-of-magnitude and trends.

### Risk 2: Test Data Representativeness

**Issue:** Benchmark results depend on input data complexity:
- Number of roots
- Image resolution
- Plant developmental stage

**Mitigation:**
- Use same test data as in paper experiments (canola_7do, soy_6do, rice_3do/10do)
- Document dataset characteristics in benchmark docstrings
- Benchmarks represent "typical" use case, not worst-case

**Acceptance:** Benchmarks are illustrative, not comprehensive across all possible inputs.

### Risk 3: Optimization Pressure

**Issue:** Developers might optimize for benchmarks at expense of correctness/readability.

**Mitigation:**
- Emphasize that correctness comes first
- Require tests to pass before optimizing
- Review performance changes for maintainability
- Document that benchmarks are for regression detection, not competition

**Acceptance:** Code review will catch premature optimization.

## Migration Plan

**Phase 1: Add Infrastructure (Week 1)**
1. Add pytest-benchmark dependency
2. Create benchmark test suite
3. Run locally to validate

**Phase 2: Integrate CI (Week 2)**
1. Add CI job for benchmarks
2. Test on feature branch
3. Merge to main

**Phase 3: Documentation (Week 3)**
1. Update pipeline docs with real metrics
2. Create benchmarking guide
3. Add performance section to README

**Rollback Plan:**
- Benchmarks are additive (no code changes to source)
- Can disable CI job by commenting out in workflow file
- Can remove pytest-benchmark from dependencies if issues arise

**No data migration needed:** All uses existing test data.

## Open Questions

1. **Storage:** Should we commit benchmark history to repo or use external service?
   - **Answer:** Start with CI artifacts (30-day retention). If we want long-term tracking, revisit with GitHub Pages or dedicated service.

2. **Thresholds:** Should we set acceptable performance ranges?
   - **Answer:** Not initially. Establish baseline first, then consider thresholds in future.

3. **Matrix:** Should we benchmark across multiple Python versions or OS?
   - **Answer:** No. Standardize on Ubuntu 22.04 + Python 3.11 to reduce variance. Performance should be similar across platforms.

4. **Comparison:** Should benchmarks auto-compare against previous runs?
   - **Answer:** Future enhancement. For now, manual comparison is sufficient.