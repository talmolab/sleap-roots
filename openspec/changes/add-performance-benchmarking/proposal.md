# Proposal: Add Performance Benchmarking to CI

## Why

The paper reports trait extraction performance metrics (~0.1-0.5s per plant for single plant pipelines, ~0.5-2s for multiple plant pipelines), but these numbers are not systematically tracked or validated in CI. Without automated benchmarking, performance regressions can silently creep in, and claims about processing speed cannot be verified against the published metrics.

## What Changes

- Add `pytest-benchmark` to dev dependencies for statistical performance measurement
- Create benchmark tests for all 7 pipeline classes to measure trait extraction speed
- Integrate benchmarks into CI workflow to track performance over time
- Store benchmark results as CI artifacts for historical comparison
- Document expected performance ranges in pipeline documentation

## Impact

**Affected specs:**
- `testing` (new capability) - Performance benchmarking requirements

**Affected code:**
- `pyproject.toml` - Add pytest-benchmark dependency
- `tests/benchmarks/` - New benchmark test suite
- `.github/workflows/ci.yml` - Add benchmark job
- `docs/api/core/pipelines.md` - Update performance metrics with benchmarked data

**Non-breaking:** This change only adds new tests and CI jobs; existing functionality is unchanged.