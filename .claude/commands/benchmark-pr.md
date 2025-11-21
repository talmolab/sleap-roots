# Benchmark Pull Request

Run benchmarks on a PR and post a comparison summary as a comment.

## What This Command Does

This command:
1. Runs the benchmark suite on the current branch
2. Compares results against the main branch baseline (if available)
3. Formats a comparison table showing performance changes
4. Posts the results as a comment on the specified PR

## Usage

```bash
# Run benchmarks and post to current PR
/benchmark-pr

# Run benchmarks and post to specific PR
/benchmark-pr <PR_NUMBER>
```

## Prerequisites

- PR must exist on GitHub
- Local branch should be up to date with PR
- Main branch baseline should exist at `.benchmarks/baselines/main.json` (created after first main branch build)

## Workflow

### Step 1: Get PR Number

```bash
# Get current PR number
gh pr status

# Or list all open PRs
gh pr list
```

### Step 2: Run Benchmarks

```bash
# Run benchmark suite with JSON output
uv run pytest tests/benchmarks/ --benchmark-only --benchmark-json=benchmark-results.json

# View results summary
cat benchmark-results.json | jq '.benchmarks[] | {name: .name, mean: .stats.mean, stddev: .stats.stddev}'
```

### Step 3: Compare Against Baseline

```bash
# Check if baseline exists
if [ -f .benchmarks/baselines/main.json ]; then
  echo "Baseline found, will compare results"
else
  echo "No baseline found - this is expected for new benchmarks"
fi
```

### Step 4: Format Comparison Table

Create a markdown table comparing PR results to baseline:

```markdown
## 📊 Benchmark Results

| Benchmark | Main | PR | Change | Status |
|-----------|------|-----|--------|--------|
| test_dicot_pipeline_performance | 155.2ms | 156.1ms | +0.6% | ✅ |
| test_younger_monocot_pipeline_performance | 124.0ms | 125.5ms | +1.2% | ✅ |
| test_older_monocot_pipeline_performance | 172.6ms | 173.7ms | +0.6% | ✅ |
| test_primary_root_pipeline_performance | 5.4ms | 5.4ms | 0.0% | ✅ |
| test_lateral_root_pipeline_performance | 65.1ms | 64.9ms | -0.3% | ✅ |
| test_multiple_dicot_pipeline_performance | 66.4ms | 66.7ms | +0.5% | ✅ |
| test_multiple_primary_root_pipeline_performance | 7.5ms | 7.5ms | 0.0% | ✅ |

**Summary:**
- All benchmarks within acceptable range (< 15% regression)
- No performance regressions detected

**Notes:**
- Benchmarks run on local machine (results may vary from CI)
- Regression threshold: 15% (configurable via `BENCHMARK_MAX_REGRESSION`)
```

### Step 5: Post Comment to PR

```bash
# Post comment using gh CLI
gh pr comment <PR_NUMBER> --body "$(cat benchmark-comment.md)"
```

## Python Script for Comparison

Here's a Python script to automate the comparison:

```python
#!/usr/bin/env python3
"""Compare benchmark results and generate markdown table."""
import json
import sys
from pathlib import Path

def load_json(path):
    """Load JSON file."""
    with open(path) as f:
        return json.load(f)

def compare_benchmarks(baseline_path, current_path, threshold=0.15):
    """Compare benchmark results and generate markdown table.

    Args:
        baseline_path: Path to baseline results JSON
        current_path: Path to current results JSON
        threshold: Maximum acceptable regression (default 15%)

    Returns:
        str: Markdown formatted comparison table
    """
    baseline = load_json(baseline_path)
    current = load_json(current_path)

    # Create lookup for baseline results
    baseline_lookup = {
        b['name']: b['stats']['mean']
        for b in baseline['benchmarks']
    }

    # Build comparison table
    rows = []
    regressions = []

    for bench in current['benchmarks']:
        name = bench['name']
        current_mean = bench['stats']['mean']

        if name in baseline_lookup:
            baseline_mean = baseline_lookup[name]
            change_pct = ((current_mean - baseline_mean) / baseline_mean) * 100

            # Format values
            baseline_str = format_time(baseline_mean)
            current_str = format_time(current_mean)
            change_str = f"{change_pct:+.1f}%"

            # Status emoji
            if change_pct > threshold * 100:
                status = "⚠️"
                regressions.append((name, change_pct))
            else:
                status = "✅"

            rows.append(f"| {name} | {baseline_str} | {current_str} | {change_str} | {status} |")
        else:
            # New benchmark
            current_str = format_time(current_mean)
            rows.append(f"| {name} | N/A | {current_str} | NEW | 🆕 |")

    # Build markdown
    lines = [
        "## 📊 Benchmark Results",
        "",
        "| Benchmark | Main | PR | Change | Status |",
        "|-----------|------|-----|--------|--------|",
    ]
    lines.extend(rows)
    lines.append("")

    # Summary
    if regressions:
        lines.append("**⚠️ Performance Regressions Detected:**")
        for name, pct in regressions:
            lines.append(f"- `{name}`: {pct:+.1f}% (exceeds {threshold*100:.0f}% threshold)")
        lines.append("")
    else:
        lines.append(f"**Summary:** All benchmarks within acceptable range (< {threshold*100:.0f}% regression)")
        lines.append("")

    # Notes
    lines.extend([
        "**Notes:**",
        "- Benchmarks run on local machine (results may vary from CI)",
        f"- Regression threshold: {threshold*100:.0f}% (configurable via `BENCHMARK_MAX_REGRESSION`)",
        "- For details, see [benchmarking guide](https://talmolab.github.io/sleap-roots/dev/benchmarking/)",
    ])

    return "\n".join(lines)

def format_time(seconds):
    """Format time in appropriate units."""
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.1f}µs"
    elif seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    else:
        return f"{seconds:.2f}s"

if __name__ == "__main__":
    baseline = Path(".benchmarks/baselines/main.json")
    current = Path("benchmark-results.json")

    if not current.exists():
        print("Error: benchmark-results.json not found. Run benchmarks first.")
        sys.exit(1)

    if not baseline.exists():
        print("No baseline found. Showing current results only.")
        # Generate table without comparison
        data = load_json(current)
        print("\n## 📊 Benchmark Results (No Baseline)")
        print("\n| Benchmark | Time |")
        print("|-----------|------|")
        for bench in data['benchmarks']:
            name = bench['name']
            mean = bench['stats']['mean']
            print(f"| {name} | {format_time(mean)} |")
        print("\n**Note:** No baseline available for comparison. These are the first benchmark results.")
    else:
        markdown = compare_benchmarks(baseline, current)
        print(markdown)
```

Save this as `.claude/scripts/compare-benchmarks.py` and make it executable:

```bash
chmod +x .claude/scripts/compare-benchmarks.py
```

## Complete Workflow Example

### Scenario: Running benchmarks on PR #133

```bash
# 1. Ensure you're on the PR branch
gh pr checkout 133

# 2. Run benchmarks
uv run pytest tests/benchmarks/ --benchmark-only --benchmark-json=benchmark-results.json

# 3. Generate comparison markdown
python .claude/scripts/compare-benchmarks.py > benchmark-comment.md

# 4. Review the comment
cat benchmark-comment.md

# 5. Post to PR
gh pr comment 133 --body "$(cat benchmark-comment.md)"

# 6. Cleanup
rm benchmark-results.json benchmark-comment.md
```

## Integration with CI

Once Phase 2 of the benchmark regression detection OpenSpec is implemented, this manual process will be automated:

- PR benchmarks will run automatically in CI
- Comparison will happen in GitHub Actions
- Comments will be posted automatically
- Regressions beyond threshold will fail the CI check

For now, use this command to manually verify performance before requesting review.

## Interpreting Results

### Performance Change Indicators

- **✅ Green check**: Performance within acceptable range
- **⚠️ Warning**: Performance regression exceeds threshold (15% by default)
- **🆕 New**: New benchmark without baseline for comparison

### What Counts as a Regression?

- **< 5% change**: Normal variance, no concern
- **5-15% slower**: Noticeable but acceptable
- **> 15% slower**: Regression, investigate and fix
- **Faster**: Improvement, celebrate! 🎉

### Common Causes of Variance

- **Machine load**: Other processes consuming CPU/memory
- **Data cache**: First run vs. subsequent runs
- **Thermal throttling**: CPU temperature affecting performance
- **Different hardware**: CI runners vs. local machine

### Troubleshooting

**Benchmark results seem inconsistent:**
- Close other applications
- Run benchmarks multiple times to establish baseline
- Check CPU usage during benchmark run

**Large regression detected:**
- Review code changes in the PR
- Profile the specific pipeline to identify bottleneck
- Consider if the change is worth the performance cost
- Document intentional regressions with rationale

**Baseline doesn't exist:**
- Normal for first benchmark run on main branch
- Baseline will be created after PR is merged and main CI runs
- Future PRs will have baseline for comparison

## Advanced Usage

### Custom Regression Threshold

```bash
# Set stricter threshold (10% instead of 15%)
BENCHMARK_MAX_REGRESSION=0.10 python .claude/scripts/compare-benchmarks.py
```

### Compare Against Specific Baseline

```bash
# Compare against specific commit's baseline
python .claude/scripts/compare-benchmarks.py \
  --baseline .benchmarks/baselines/abc123def.json \
  --current benchmark-results.json
```

### Run Specific Benchmarks

```bash
# Only run dicot pipeline benchmarks
uv run pytest tests/benchmarks/ -k "dicot" --benchmark-only --benchmark-json=benchmark-results.json
```

## Related Commands

- `/test` - Run all tests including benchmarks
- `/run-ci-locally` - Run full CI suite locally
- `/pre-merge` - Complete pre-merge checklist
- `/review-pr` - Comprehensive PR review workflow

## Future Enhancements

When Phase 2-3 of the OpenSpec is complete:

1. **Automatic PR comments**: CI will post comparison automatically
2. **Regression detection**: CI will fail if threshold exceeded
3. **Historical tracking**: View trends over time on docs site
4. **Review integration**: `/review-pr` will include benchmark summary

## Tips

1. **Run benchmarks early**: Check performance impact before requesting review
2. **Document regressions**: If intentional, explain why in PR description
3. **Compare locally first**: Catch issues before CI runs
4. **Watch for trends**: Small regressions can accumulate over time
5. **Profile when needed**: Use `pytest-profiling` for detailed analysis

---

**See also:**
- [Benchmarking Guide](../docs/dev/benchmarking.md)
- [OpenSpec Proposal](../openspec/changes/benchmark-regression-detection/proposal.md)
- [Pre-merge Checklist](.claude/commands/pre-merge.md)