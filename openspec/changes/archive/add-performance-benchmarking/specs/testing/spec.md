# Testing Capability

## ADDED Requirements

### Requirement: Performance Benchmarking

The system SHALL provide automated performance benchmarks for all trait extraction pipelines to measure processing speed and detect regressions.

#### Scenario: Benchmark single plant pipeline

- **GIVEN** a test dataset with real SLEAP predictions (canola, soy, or rice)
- **WHEN** a benchmark test runs `DicotPipeline.fit_series()` on the dataset
- **THEN** pytest-benchmark SHALL measure execution time across multiple iterations
- **AND** report statistical metrics (mean, standard deviation, min, max, percentiles)
- **AND** execution time SHALL be within expected range (~0.1-0.5 seconds per plant on GitHub Actions runners)

#### Scenario: Benchmark multiple plant pipeline

- **GIVEN** a test dataset with multiple plants
- **WHEN** a benchmark test runs `MultipleDicotPipeline.fit_series()` on the dataset
- **THEN** pytest-benchmark SHALL measure execution time
- **AND** execution time SHALL scale with plant count (~0.5-2 seconds on GitHub Actions runners)

#### Scenario: Run benchmarks in CI

- **GIVEN** code is pushed to the main branch
- **WHEN** the CI benchmark job executes
- **THEN** all pipeline benchmarks SHALL run using `pytest tests/benchmarks/ --benchmark-only`
- **AND** benchmark results SHALL be exported to JSON format
- **AND** JSON results SHALL be uploaded as a CI artifact
- **AND** benchmark statistics SHALL be printed in CI logs

#### Scenario: Run benchmarks locally

- **GIVEN** a developer has pytest-benchmark installed
- **WHEN** they run `uv run pytest tests/benchmarks/ --benchmark-only`
- **THEN** benchmarks SHALL execute with statistical analysis
- **AND** results SHALL display in terminal with formatted table
- **AND** benchmarks SHALL complete within 5 minutes for full suite

#### Scenario: Exclude benchmarks from normal test runs

- **GIVEN** benchmarks exist in `tests/benchmarks/` directory
- **WHEN** a developer runs `uv run pytest tests/` (normal test command)
- **THEN** benchmark tests SHALL NOT execute automatically
- **AND** only when `--benchmark-only` flag is used SHALL benchmarks run

### Requirement: Benchmark Test Coverage

The system SHALL include benchmark tests for all trait extraction pipeline classes to ensure comprehensive performance monitoring.

#### Scenario: Cover all pipeline types

- **GIVEN** the sleap-roots codebase has 7 pipeline classes
- **WHEN** benchmark tests are written
- **THEN** the following pipelines SHALL have dedicated benchmark tests:
  - `DicotPipeline` - primary + lateral roots
  - `YoungerMonocotPipeline` - crown + primary roots (3 DAG)
  - `OlderMonocotPipeline` - crown roots only (10 DAG)
  - `MultipleDicotPipeline` - batch dicot processing
  - `PrimaryRootPipeline` - primary root only
  - `MultiplePrimaryRootPipeline` - batch primary roots
  - `LateralRootPipeline` - lateral roots only
- **AND** each benchmark SHALL use representative test data from `tests/data/`

#### Scenario: Benchmark real-world scenarios

- **GIVEN** benchmarks measure pipeline performance
- **WHEN** selecting test data for benchmarks
- **THEN** test data SHALL be real SLEAP predictions from paper experiments
- **AND** data SHALL represent typical use cases (not edge cases)
- **AND** data characteristics SHALL be documented in benchmark docstrings

### Requirement: Performance Documentation

The system SHALL document expected performance characteristics based on automated benchmarks to set accurate user expectations.

#### Scenario: Document pipeline performance metrics

- **GIVEN** benchmarks have been run on standard hardware
- **WHEN** updating pipeline documentation
- **THEN** `docs/api/core/pipelines.md` SHALL include actual benchmark results
- **AND** metrics SHALL specify hardware configuration (GitHub Actions Ubuntu 22.04)
- **AND** metrics SHALL include mean execution time and standard deviation
- **AND** metrics SHALL link to benchmark test suite for reproducibility

#### Scenario: Provide benchmarking guide

- **GIVEN** developers may want to run or add benchmarks
- **WHEN** creating developer documentation
- **THEN** `docs/dev/benchmarking.md` SHALL be created with:
  - Instructions for running benchmarks locally
  - Explanation of pytest-benchmark output interpretation
  - Guide for adding new benchmark tests
  - Best practices for performance testing
- **AND** guide SHALL explain difference between benchmarking and profiling

### Requirement: Statistical Rigor

The system SHALL ensure benchmark measurements are statistically meaningful and account for variance in execution environment.

#### Scenario: Multiple iterations for reliability

- **GIVEN** a benchmark test executes
- **WHEN** pytest-benchmark runs the test
- **THEN** the pipeline function SHALL be executed multiple times (minimum 5 rounds)
- **AND** warmup rounds SHALL be performed to stabilize execution
- **AND** statistical outliers SHALL be detected and handled appropriately

#### Scenario: Report confidence metrics

- **GIVEN** benchmark results are generated
- **WHEN** results are displayed or exported
- **THEN** output SHALL include:
  - Mean execution time
  - Standard deviation
  - Minimum time
  - Maximum time
  - Median time
  - Percentiles (25th, 75th)
- **AND** results SHALL be exported in machine-readable JSON format

### Requirement: CI Integration

The system SHALL integrate performance benchmarks into CI workflow to enable continuous performance monitoring without impacting development velocity.

#### Scenario: Benchmark on main branch only

- **GIVEN** benchmarks add ~2-3 minutes to CI time
- **WHEN** CI workflow is configured
- **THEN** benchmark job SHALL only run on pushes to main branch
- **AND** benchmark job SHALL NOT run on pull requests
- **AND** this prevents noise and reduces CI resource usage during development

#### Scenario: Store benchmark results

- **GIVEN** benchmarks run in CI
- **WHEN** benchmark execution completes
- **THEN** results SHALL be saved as `benchmark-results.json`
- **AND** JSON file SHALL be uploaded as GitHub Actions artifact
- **AND** artifact SHALL be retained for 30 days for historical comparison
- **AND** artifact SHALL be downloadable for local analysis

#### Scenario: Standardized benchmark environment

- **GIVEN** benchmark results should be comparable across runs
- **WHEN** CI benchmark job is configured
- **THEN** job SHALL run on Ubuntu 22.04 runner (standardized)
- **AND** job SHALL use Python 3.11 from `.python-version`
- **AND** job SHALL use frozen lockfile via `uv sync --frozen`
- **AND** this ensures consistent dependency versions across benchmark runs