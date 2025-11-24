# Spec: CI/CD Benchmark Integration

## ADDED Requirements

### Requirement: PR benchmark workflow

Pull requests MUST run benchmarks with baseline comparison before merge.

#### Scenario: PR triggers benchmark job

- **Given** a PR is opened targeting main branch
- **When** CI workflows execute
- **Then** a "Benchmark (PR)" job runs on ubuntu-22.04
- **And** job checks out PR branch code
- **And** job fetches baseline from `.benchmarks/baselines/main.json`
- **And** job runs pytest benchmarks with comparison
- **And** job uploads results as artifacts

#### Scenario: PR benchmark uses main branch baseline

- **Given** main branch has baseline at commit SHA `abc123`
- **And** `.benchmarks/baselines/main.json` exists in that commit
- **When** PR benchmark job runs
- **Then** job executes `git fetch origin main`
- **And** job checks out `.benchmarks/baselines/main.json` from origin/main
- **And** comparison uses this baseline file
- **And** missing baseline does not fail the job

#### Scenario: Benchmark failure blocks PR merge

- **Given** PR benchmarks detect regression beyond threshold
- **When** CI completes
- **Then** "Benchmark (PR)" check shows failed status
- **And** PR cannot be merged (if branch protection enabled)
- **And** PR checks page shows "Benchmark regression detected"

### Requirement: Main branch baseline storage

Main branch builds MUST store benchmark results as new baseline.

#### Scenario: Main branch push updates baseline

- **Given** a commit is pushed to main branch
- **When** "Benchmark (Main)" job completes successfully
- **Then** `benchmark-results.json` is copied to `.benchmarks/baselines/main.json`
- **And** `benchmark-results.json` is copied to `.benchmarks/baselines/<commit-sha>.json`
- **And** files are committed to repo with message "chore: update benchmark baselines [skip ci]"
- **And** commit is pushed to main

#### Scenario: Daily history aggregation

- **Given** main branch benchmark completes on date 2024-01-15
- **When** baseline storage step runs
- **Then** `benchmark-results.json` is copied to `.benchmarks/history/2024-01-15.json`
- **And** file overwrites any existing entry for that date
- **And** file is committed to repo along with baseline

#### Scenario: Baseline cleanup

- **Given** `.benchmarks/baselines/` contains files older than 90 days
- **When** main branch benchmark cleanup step runs
- **Then** files with mtime > 90 days are deleted
- **And** only recent baselines are committed
- **And** `.benchmarks/baselines/main.json` is never deleted

### Requirement: Benchmark artifact management

Benchmark results MUST be uploaded as CI artifacts for external access.

#### Scenario: PR artifact upload

- **Given** PR benchmark job completes
- **When** upload step executes
- **Then** `benchmark-results.json` is uploaded with name "benchmark-pr-results"
- **And** `benchmark-regressions.json` is uploaded if it exists
- **And** artifacts have 30-day retention
- **And** artifacts are downloadable via GitHub UI or `gh` CLI

#### Scenario: Main branch artifact upload

- **Given** main branch benchmark job completes
- **When** upload step executes
- **Then** `benchmark-results.json` is uploaded with name "benchmark-results"
- **And** artifact has 30-day retention
- **And** artifact is tagged with commit SHA

### Requirement: PR comment with benchmark results

Pull requests MUST receive automated comment with benchmark comparison.

#### Scenario: Post comparison table to PR

- **Given** PR benchmark job completes
- **And** baseline comparison data is available
- **When** GitHub Actions script executes
- **Then** a comment is posted to the PR
- **And** comment contains markdown table with columns: Test, Main, PR, Change, Status
- **And** each benchmark row shows baseline vs PR performance
- **And** regressions are marked with ⚠️ emoji
- **And** acceptable results are marked with ✅ emoji

#### Scenario: Comment includes regression warning

- **Given** PR benchmarks detect 2 regressions
- **When** PR comment is posted
- **Then** comment includes "⚠️ Performance regression detected in 2 benchmark(s)"
- **And** comment lists which tests regressed and by how much
- **And** comment links to uploaded artifacts for investigation

#### Scenario: No comment on baseline comparison failure

- **Given** PR runs benchmarks
- **And** no baseline file exists for comparison
- **When** comment script executes
- **Then** comment indicates "No baseline available for comparison"
- **And** comment shows PR results without comparison
- **And** comment notes this is expected for new benchmarks