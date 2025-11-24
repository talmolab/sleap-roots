# Implementation Tasks

## 1. Add Dependencies
- [ ] 1.1 Add `pytest-benchmark` to `[dependency-groups].dev` in pyproject.toml
- [ ] 1.2 Add `pytest-benchmark` to `[project.optional-dependencies].dev` for pip compatibility
- [ ] 1.3 Run `uv lock` to update lockfile
- [ ] 1.4 Verify installation with `uv sync && uv run pytest --co -q tests/` (no errors)

## 2. Create Benchmark Test Suite
- [ ] 2.1 Create `tests/benchmarks/` directory
- [ ] 2.2 Create `tests/benchmarks/__init__.py`
- [ ] 2.3 Create `tests/benchmarks/conftest.py` with benchmark fixtures (reuse existing test data)
- [ ] 2.4 Create `tests/benchmarks/test_pipeline_performance.py` with benchmarks for:
  - [ ] 2.4.1 `DicotPipeline.fit_series()` - single canola plant
  - [ ] 2.4.2 `YoungerMonocotPipeline.fit_series()` - single rice plant (3 DAG)
  - [ ] 2.4.3 `OlderMonocotPipeline.fit_series()` - single rice plant (10 DAG)
  - [ ] 2.4.4 `MultipleDicotPipeline.fit_series()` - batch of plants
  - [ ] 2.4.5 `PrimaryRootPipeline.fit_series()` - primary root only
  - [ ] 2.4.6 `MultiplePrimaryRootPipeline.fit_series()` - multiple primary roots
  - [ ] 2.4.7 `LateralRootPipeline.fit_series()` - lateral roots only
- [ ] 2.5 Run benchmarks locally: `uv run pytest tests/benchmarks/ --benchmark-only`
- [ ] 2.6 Verify benchmarks produce statistical output (mean, stddev, min, max)

## 3. Integrate into CI
- [ ] 3.1 Add benchmark job to `.github/workflows/ci.yml` that runs on Ubuntu 22.04
- [ ] 3.2 Configure job to run `uv run pytest tests/benchmarks/ --benchmark-only --benchmark-json=benchmark-results.json`
- [ ] 3.3 Add benchmark results artifact upload (JSON file)
- [ ] 3.4 Set benchmark job to run only on pushes to main (not PRs) to avoid noise
- [ ] 3.5 Test CI workflow on feature branch

## 4. Documentation
- [ ] 4.1 Update `docs/api/core/pipelines.md` Performance Considerations section with:
  - [ ] 4.1.1 Actual benchmarked timings from test run
  - [ ] 4.1.2 Hardware specs (GitHub Actions Ubuntu 22.04 runner)
  - [ ] 4.1.3 Link to benchmark test suite
- [ ] 4.2 Add `docs/dev/benchmarking.md` guide explaining:
  - [ ] 4.2.1 How to run benchmarks locally
  - [ ] 4.2.2 How to interpret results
  - [ ] 4.2.3 How to add new benchmarks
  - [ ] 4.2.4 Performance regression detection process
- [ ] 4.3 Update main README.md with performance metrics section

## 5. Validation
- [ ] 5.1 Run full test suite to ensure benchmarks don't interfere: `uv run pytest tests/`
- [ ] 5.2 Run benchmarks in isolation: `uv run pytest tests/benchmarks/ --benchmark-only`
- [ ] 5.3 Verify CI benchmark job completes successfully
- [ ] 5.4 Download and inspect benchmark JSON artifact
- [ ] 5.5 Compare benchmark results against paper-reported metrics (should be similar order of magnitude)