# Spec: Performance Regression Testing

## MODIFIED Requirements

### Requirement: Benchmark comparison against baseline

The benchmark suite SHALL compare current performance against stored baseline from main branch.

#### Scenario: PR benchmark compares against main baseline

- **Given** a PR modifies pipeline code
- **And** main branch has stored benchmark baseline at `.benchmarks/baselines/main.json`
- **When** CI runs benchmarks on the PR
- **Then** each benchmark result is compared to corresponding baseline measurement
- **And** percentage change (regression or improvement) is calculated
- **And** comparison results are stored in `benchmark-results.json`

#### Scenario: No baseline available for new benchmarks

- **Given** a PR adds a new benchmark test
- **And** no baseline exists for this test name
- **When** CI runs benchmarks on the PR
- **Then** the new benchmark runs without comparison
- **And** the result is marked as "no baseline" in output
- **And** CI does not fail due to missing baseline

### Requirement: Regression detection threshold enforcement

Benchmarks MUST fail CI if performance regresses beyond configurable threshold.

#### Scenario: Performance regression exceeds threshold

- **Given** baseline shows `test_dicot_pipeline` mean of 150ms
- **And** regression threshold is 15%
- **When** PR benchmark measures 180ms mean (20% slower)
- **Then** the benchmark comparison fails
- **And** CI job fails with exit code 1
- **And** regression details are written to `benchmark-regressions.json`
- **And** error message indicates which benchmarks regressed

#### Scenario: Performance within acceptable threshold

- **Given** baseline shows `test_dicot_pipeline` mean of 150ms
- **And** regression threshold is 15%
- **When** PR benchmark measures 165ms mean (10% slower)
- **Then** the benchmark comparison passes
- **And** CI job succeeds
- **And** result is logged but not treated as failure

#### Scenario: Performance improvement

- **Given** baseline shows `test_dicot_pipeline` mean of 150ms
- **When** PR benchmark measures 130ms mean (13% faster)
- **Then** the benchmark comparison passes
- **And** improvement is noted in results
- **And** CI job succeeds

### Requirement: Configurable regression thresholds

Regression tolerance MUST be configurable globally and per-benchmark.

#### Scenario: Global threshold via environment variable

- **Given** CI environment sets `BENCHMARK_MAX_REGRESSION=0.10`
- **When** any benchmark regresses by more than 10%
- **Then** the benchmark comparison fails
- **And** CI uses 10% threshold for all benchmarks

#### Scenario: Per-benchmark threshold override

- **Given** test is marked with `@pytest.mark.benchmark(max_regression=0.05)`
- **And** global threshold is 15%
- **When** this benchmark regresses by 8%
- **Then** the benchmark comparison fails (exceeds 5% test-specific threshold)
- **And** other benchmarks still use 15% global threshold

## ADDED Requirements

### Requirement: Statistical confidence in regression detection

Regression detection MUST account for measurement variance to avoid false positives.

#### Scenario: Multiple rounds for statistical confidence

- **Given** PR benchmark configuration sets `--benchmark-min-rounds=5`
- **When** CI runs benchmarks
- **Then** each benchmark executes at least 5 iterations
- **And** mean, standard deviation, and median are calculated
- **And** regression is based on mean comparison
- **And** high variance is noted in output

#### Scenario: Outlier handling

- **Given** one benchmark iteration has anomalous performance (e.g., 500ms vs typical 150ms)
- **When** pytest-benchmark calculates statistics
- **Then** outlier detection removes extreme values
- **And** mean is calculated from remaining iterations
- **And** comparison is based on robust statistics

### Requirement: Regression artifact storage

Regression details MUST be stored as CI artifacts for investigation.

#### Scenario: Regression details available for review

- **Given** PR benchmarks detect 2 regressions
- **When** CI job completes
- **Then** `benchmark-regressions.json` contains details for both regressions
- **And** file includes: benchmark name, baseline mean, current mean, regression %
- **And** file is uploaded as GitHub Actions artifact
- **And** artifact is retained for 30 days

#### Scenario: Comparison data for all benchmarks

- **Given** PR runs 7 benchmark tests
- **When** CI job completes (regardless of pass/fail)
- **Then** `benchmark-results.json` contains all 7 results
- **And** file includes comparison to baseline where available
- **And** file is uploaded as artifact
- **And** artifact can be downloaded via `gh run download`