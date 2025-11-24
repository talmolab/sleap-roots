# Design: Benchmark Regression Detection and Historical Tracking

## Architecture Overview

This design implements a two-tier benchmarking system:
1. **PR-level regression detection** - Catch performance issues before merge
2. **Main-level historical tracking** - Track trends and publish results

## Key Design Decisions

### 1. Baseline Storage Strategy

**Decision:** Store baseline benchmarks in Git as JSON files

**Rationale:**
- ✅ Version controlled - Baselines tied to specific commits
- ✅ No external dependencies - Works without databases or cloud services
- ✅ Fast comparison - Local file access in CI
- ✅ Transparent - Anyone can inspect baseline data
- ❌ Repo size impact - Mitigated by storing only summary stats (not full data)

**Alternatives considered:**
- **GitHub Artifacts API**: 90-day retention limit, complex API calls, rate limits
- **External database**: Requires secrets, adds infrastructure dependency
- **GitHub Releases**: Would clutter releases with non-user-facing data

**Implementation:**
```
.benchmarks/
  baselines/
    main.json          # Latest main branch baseline
    <commit-sha>.json  # Historical baselines (pruned to 90 days)
  history/
    YYYY-MM-DD.json    # Daily aggregated results for charting
```

### 2. Regression Detection Threshold

**Decision:** 15% performance regression threshold (configurable)

**Rationale:**
- Accounts for CI runner variance (typically 5-10%)
- Catches significant regressions while avoiding false positives
- Published benchmarks show 2-3% stddev, so 15% is ~5σ
- Can be overridden per-benchmark via pytest marks

**Configuration:**
```yaml
env:
  BENCHMARK_MAX_REGRESSION: "0.15"  # 15% allowed regression
  BENCHMARK_MIN_ROUNDS: "5"         # Statistical confidence
```

**Per-benchmark override:**
```python
@pytest.mark.benchmark(max_regression=0.10)  # Stricter for critical pipelines
def test_dicot_pipeline_performance(...):
    ...
```

### 3. PR Review Integration

**Decision:** Extend `/review-pr` command to fetch and format benchmark data

**Rationale:**
- ✅ Consistent with existing review workflow
- ✅ No manual artifact download needed
- ✅ Formatted comparison in review comment
- ✅ Leverages GitHub CLI (`gh`)

**Workflow:**
1. PR author pushes changes
2. CI runs benchmarks and compares to baseline
3. Reviewer runs `/review-pr <number>`
4. Command fetches benchmark artifact via `gh run download`
5. Parses JSON and formats comparison table
6. Posts review with performance summary

**Example output in PR comment:**
```markdown
## 📊 Benchmark Results

| Test | Main | PR | Change | Status |
|------|------|----|---------:--------|
| test_dicot_pipeline | 155.2ms | 158.3ms | +2.0% | ✅ |
| test_younger_monocot | 124.0ms | 145.8ms | +17.6% | ⚠️ REGRESSION |

⚠️ **Performance regression detected** in 1 benchmark(s).
The `test_younger_monocot` pipeline is 17.6% slower than baseline (threshold: 15%).
```

### 4. Historical Tracking and Visualization

**Decision:** Publish benchmark history to GitHub Pages as part of docs site

**Rationale:**
- ✅ Integrated with existing docs infrastructure
- ✅ No separate hosting needed
- ✅ Versioned with code
- ✅ Searchable and linkable

**Implementation approach:**
- Store daily aggregated results in `.benchmarks/history/`
- Generate static charts using matplotlib or plotly
- Render as markdown page at `docs/benchmarks/index.md`
- Auto-update via GitHub Action on main branch pushes

**Chart design:**
```python
# Line chart: Performance over time per pipeline
# X-axis: Date
# Y-axis: Mean execution time (ms)
# Series: One line per benchmark test
# Annotations: Version releases, major changes
```

### 5. Artifact Storage and Cleanup

**Decision:** 90-day rolling window for benchmark history

**Rationale:**
- Sufficient for detecting gradual performance degradation
- Keeps repo size manageable (<10MB for 90 days of daily results)
- Aligns with quarterly release cadence
- Old data can be archived to GitHub Releases if needed

**Cleanup strategy:**
```bash
# In CI, after storing new baseline:
find .benchmarks/baselines -name "*.json" -mtime +90 -delete
find .benchmarks/history -name "*.json" -mtime +90 -delete
```

## Component Design

### 1. Baseline Comparison Utility

**File:** `tests/benchmarks/conftest.py`

```python
import json
import pytest
from pathlib import Path

def pytest_benchmark_compare_machine_info(config, machine_info):
    """Hook to load baseline for comparison."""
    baseline_path = Path(".benchmarks/baselines/main.json")
    if baseline_path.exists():
        with open(baseline_path) as f:
            return json.load(f)
    return None

def pytest_benchmark_update_json(config, benchmarks, output_json):
    """Hook to check for regressions after benchmark run."""
    baseline = pytest_benchmark_compare_machine_info(config, None)
    if not baseline:
        return  # No baseline to compare against

    threshold = float(os.getenv("BENCHMARK_MAX_REGRESSION", "0.15"))
    regressions = []

    for bench in benchmarks:
        baseline_bench = next(
            (b for b in baseline["benchmarks"] if b["name"] == bench.name),
            None
        )
        if not baseline_bench:
            continue

        baseline_mean = baseline_bench["stats"]["mean"]
        current_mean = bench.stats["mean"]
        regression = (current_mean - baseline_mean) / baseline_mean

        if regression > threshold:
            regressions.append({
                "name": bench.name,
                "baseline": baseline_mean,
                "current": current_mean,
                "regression": regression,
            })

    if regressions:
        # Store regression details in artifact
        with open("benchmark-regressions.json", "w") as f:
            json.dump(regressions, f, indent=2)

        # Fail the benchmark run
        pytest.fail(
            f"Performance regression detected in {len(regressions)} benchmark(s). "
            f"See benchmark-regressions.json for details."
        )
```

### 2. CI Workflow Changes

**File:** `.github/workflows/ci.yml`

Add new benchmark job for PRs:

```yaml
benchmark-pr:
  name: Benchmark (PR)
  runs-on: ubuntu-22.04
  if: github.event_name == 'pull_request'

  steps:
    - name: Checkout PR branch
      uses: actions/checkout@v4
      with:
        lfs: true

    - name: Checkout main branch baseline
      run: |
        git fetch origin main
        git checkout origin/main -- .benchmarks/baselines/main.json || echo "No baseline found"

    - name: Set up uv
      uses: astral-sh/setup-uv@v5
      with:
        enable-cache: true
        github-token: ${{ secrets.GITHUB_TOKEN }}
        cache-dependency-glob: "**/uv.lock"

    - name: Install Python
      run: uv python install

    - name: Sync deps
      run: uv sync --frozen

    - name: Run benchmarks with comparison
      run: |
        uv run pytest tests/benchmarks/ \
          --benchmark-only \
          --benchmark-json=benchmark-results.json \
          --benchmark-compare=.benchmarks/baselines/main.json \
          --benchmark-compare-fail=mean:15% \
          --benchmark-columns=min,max,mean,stddev,median \
          --benchmark-sort=name

    - name: Upload benchmark results
      uses: actions/upload-artifact@v4
      if: always()
      with:
        name: benchmark-pr-results
        path: |
          benchmark-results.json
          benchmark-regressions.json
        retention-days: 30

    - name: Comment on PR with results
      if: always() && github.event_name == 'pull_request'
      uses: actions/github-script@v7
      with:
        script: |
          const fs = require('fs');

          // Load results
          const results = JSON.parse(fs.readFileSync('benchmark-results.json', 'utf8'));
          const baseline = JSON.parse(fs.readFileSync('.benchmarks/baselines/main.json', 'utf8'));

          // Format table
          let table = '| Test | Main | PR | Change | Status |\n';
          table += '|------|------|----|---------:--------|\n';

          for (const bench of results.benchmarks) {
            const baselineBench = baseline.benchmarks.find(b => b.name === bench.name);
            if (!baselineBench) continue;

            const baselineMean = (baselineBench.stats.mean * 1000).toFixed(2);
            const currentMean = (bench.stats.mean * 1000).toFixed(2);
            const change = ((bench.stats.mean - baselineBench.stats.mean) / baselineBench.stats.mean * 100).toFixed(1);
            const status = Math.abs(change) > 15 ? '⚠️ REGRESSION' : '✅';

            table += `| ${bench.name} | ${baselineMean}ms | ${currentMean}ms | ${change > 0 ? '+' : ''}${change}% | ${status} |\n`;
          }

          // Post comment
          await github.rest.issues.createComment({
            owner: context.repo.owner,
            repo: context.repo.repo,
            issue_number: context.issue.number,
            body: `## 📊 Benchmark Results\n\n${table}`
          });
```

Update existing benchmark job for main:

```yaml
benchmark:
  name: Benchmark (Main)
  runs-on: ubuntu-22.04
  if: github.ref == 'refs/heads/main' && github.event_name == 'push'

  steps:
    # ... existing steps ...

    - name: Store baseline
      run: |
        mkdir -p .benchmarks/baselines .benchmarks/history
        cp benchmark-results.json .benchmarks/baselines/main.json
        cp benchmark-results.json .benchmarks/baselines/${{ github.sha }}.json

        # Daily history aggregate
        DATE=$(date +%Y-%m-%d)
        cp benchmark-results.json .benchmarks/history/${DATE}.json

    - name: Commit baseline to repo
      run: |
        git config user.name "github-actions[bot]"
        git config user.email "github-actions[bot]@users.noreply.github.com"
        git add .benchmarks/
        git commit -m "chore: update benchmark baselines [skip ci]" || echo "No changes"
        git push
```

### 3. Review Command Enhancement

**File:** `.claude/commands/review-pr.md`

Add new section after "Review Checklist":

```markdown
### 8. Performance

- [ ] No benchmark regressions beyond acceptable threshold (15%)
- [ ] Performance-sensitive code paths are optimized
- [ ] Large dataset handling is memory-efficient
- [ ] Changes that affect performance are documented

## Benchmark Summary (Automated)

When reviewing a PR, fetch and display benchmark results:

```bash
# Fetch benchmark artifact
gh run download $(gh pr view <number> --json headSha -q .headSha | xargs -I {} gh run list --commit {} --json databaseId -q '.[0].databaseId') --name benchmark-pr-results

# Display comparison
python3 << 'EOF'
import json
with open('benchmark-results.json') as f:
    results = json.load(f)
with open('.benchmarks/baselines/main.json') as f:
    baseline = json.load(f)

print("\n📊 Benchmark Comparison\n")
print(f"{'Test':<45} {'Main':<12} {'PR':<12} {'Change':<10} Status")
print("=" * 90)

for bench in results['benchmarks']:
    base = next((b for b in baseline['benchmarks'] if b['name'] == bench['name']), None)
    if not base:
        continue

    base_mean = base['stats']['mean'] * 1000
    curr_mean = bench['stats']['mean'] * 1000
    change = (curr_mean - base_mean) / base_mean * 100
    status = '⚠️ REGRESSION' if abs(change) > 15 else '✅ OK'

    print(f"{bench['name']:<45} {base_mean:>10.2f}ms {curr_mean:>10.2f}ms {change:>+8.1f}% {status}")

EOF
```

Include this output in your review comment.
```

## Testing Strategy

### Unit Tests
- `tests/benchmarks/test_baseline_comparison.py` - Test baseline loading and comparison logic
- `tests/benchmarks/test_regression_detection.py` - Test threshold enforcement

### Integration Tests
- Manual PR with intentional performance regression
- Verify CI fails and posts comment
- Verify baseline updates after merge to main

### Documentation Tests
- Verify benchmark history page renders correctly
- Check that charts load and display data
- Validate links to specific benchmark runs

## Rollout Plan

### Phase 1: Baseline Infrastructure (Week 1)
1. Add baseline storage directory structure
2. Update main branch benchmark job to store baselines
3. Collect initial baseline data (run on main)

### Phase 2: PR Regression Detection (Week 2)
4. Add PR benchmark job with comparison
5. Implement regression detection logic
6. Add artifact upload for investigation

### Phase 3: Review Integration (Week 3)
7. Update `/review-pr` command with benchmark fetching
8. Test on dogfooding PR
9. Document workflow for reviewers

### Phase 4: Historical Tracking (Week 4)
10. Create benchmark history page template
11. Add chart generation script
12. Set up GitHub Pages publishing workflow
13. Backfill historical data from artifacts

## Monitoring and Maintenance

### Metrics to Track
- False positive rate (regressions that were actually CI variance)
- True positive rate (caught real performance issues)
- Baseline storage size over time
- Chart generation time

### Maintenance Tasks
- Monthly: Review and adjust regression threshold based on false positive rate
- Quarterly: Archive old benchmark data to releases
- Annual: Evaluate alternative visualization libraries

## Future Enhancements

- **Per-benchmark thresholds**: Allow different thresholds for different pipelines
- **Benchmark tags**: Tag benchmarks as "critical" vs "informational"
- **Automatic bisection**: If regression detected, auto-bisect to find culprit commit
- **Slack notifications**: Alert team channel on main branch regressions
- **Performance budgets**: Set absolute time limits (not just % regression)