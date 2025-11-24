# Spec: Benchmark Documentation and Visibility

## ADDED Requirements

### Requirement: Historical benchmark results page

Documentation site MUST display benchmark performance trends over time.

#### Scenario: Benchmark history page exists

- **Given** documentation site is deployed
- **When** user navigates to `/benchmarks/`
- **Then** page displays "Performance Benchmarks" heading
- **And** page shows introduction explaining benchmark methodology
- **And** page links to benchmarking guide at `/dev/benchmarking/`
- **And** page is listed in mkdocs.yml navigation

#### Scenario: Performance charts show trends

- **Given** `.benchmarks/history/` contains 30 days of results
- **When** benchmark history page renders
- **Then** page displays line chart for each pipeline benchmark
- **And** X-axis shows dates over last 30 days
- **And** Y-axis shows mean execution time in milliseconds
- **And** chart highlights version releases as annotations
- **And** chart shows ±1 stddev bands around mean

#### Scenario: Current vs historical comparison

- **Given** benchmark history page displays
- **When** user views summary table
- **Then** table shows latest benchmark results
- **And** table compares to 7-day moving average
- **And** table compares to 30-day moving average
- **And** trends (improving/stable/degrading) are indicated
- **And** table links to raw JSON data

### Requirement: Benchmark data transparency

Raw benchmark data MUST be accessible and explorable.

#### Scenario: Download historical data

- **Given** user views benchmark history page
- **When** user clicks "Download Data" link
- **Then** browser downloads `.benchmarks/history/` directory as ZIP
- **And** ZIP contains JSON files for all available dates
- **And** README.txt explains JSON schema

#### Scenario: Link to specific benchmark run

- **Given** user views chart showing regression on 2024-01-15
- **When** user clicks that data point
- **Then** page scrolls to details for 2024-01-15
- **And** details show all benchmark results for that date
- **And** details link to GitHub commit for that day
- **And** details link to CI run that produced the data

### Requirement: Review command benchmark integration

The `/review-pr` command MUST fetch and display benchmark results.

#### Scenario: Fetch benchmark artifact in review workflow

- **Given** reviewer runs `/review-pr 133`
- **And** PR 133 has completed benchmark checks
- **When** command executes benchmark fetch step
- **Then** command runs `gh run download` for PR's head commit
- **And** command downloads "benchmark-pr-results" artifact
- **And** `benchmark-results.json` is extracted to temp directory

#### Scenario: Format benchmark comparison for review

- **Given** benchmark artifact is downloaded
- **And** baseline data exists
- **When** review command formats output
- **Then** output includes "📊 Benchmark Comparison" section
- **And** section contains formatted table comparing main vs PR
- **And** regressions are highlighted with ⚠️
- **And** improvements are noted
- **And** statistical significance is noted for borderline cases

#### Scenario: Include benchmark summary in review comment

- **Given** reviewer approves PR with `/review-pr 133 --approve`
- **And** benchmark data is available
- **When** review comment is posted
- **Then** comment includes benchmark comparison table
- **And** comment notes if performance is acceptable
- **And** comment links to full artifact for details
- **And** reviewer can approve even if benchmarks show minor regression

#### Scenario: Handle missing benchmark data gracefully

- **Given** reviewer runs `/review-pr 133`
- **And** PR 133 has no benchmark results (not yet run or skipped)
- **When** command attempts to fetch artifact
- **Then** command logs "No benchmark data available for this PR"
- **And** review proceeds without benchmark section
- **And** command does not fail

## MODIFIED Requirements

### Requirement: Benchmarking guide documentation

Existing benchmarking guide MUST explain regression detection workflow.

#### Scenario: Guide explains PR benchmark workflow

- **Given** user reads `docs/dev/benchmarking.md`
- **When** user navigates to "CI Integration" section
- **Then** section explains PR benchmark jobs
- **And** section documents regression threshold (15%)
- **And** section shows example of PR comment with results
- **And** section links to regression detection OpenSpec proposal

#### Scenario: Guide explains baseline management

- **Given** developer wants to update baseline
- **When** developer reads benchmarking guide
- **Then** guide explains baselines are auto-updated on main branch
- **And** guide shows how to manually download and inspect baselines
- **And** guide documents baseline storage location (`.benchmarks/baselines/`)
- **And** guide explains 90-day cleanup policy

#### Scenario: Guide documents threshold configuration

- **Given** developer needs stricter threshold for critical pipeline
- **When** developer reads benchmarking guide
- **Then** guide shows `@pytest.mark.benchmark(max_regression=0.05)` syntax
- **And** guide explains global `BENCHMARK_MAX_REGRESSION` env var
- **And** guide provides examples of when to adjust thresholds
- **And** guide warns against over-tuning to avoid false negatives