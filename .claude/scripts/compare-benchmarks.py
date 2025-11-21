#!/usr/bin/env python3
"""Compare benchmark results and generate markdown table.

This script compares current benchmark results against a baseline and generates
a markdown-formatted comparison table for posting as a PR comment.

Usage:
    python compare-benchmarks.py [--baseline PATH] [--current PATH] [--threshold FLOAT]

Examples:
    # Compare with default paths
    python compare-benchmarks.py

    # Use custom baseline
    python compare-benchmarks.py --baseline .benchmarks/baselines/abc123.json

    # Set custom threshold (10% instead of 15%)
    python compare-benchmarks.py --threshold 0.10
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def load_json(path: Path) -> Dict:
    """Load JSON file.

    Args:
        path: Path to JSON file

    Returns:
        Parsed JSON data as dictionary

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If JSON is invalid
    """
    with open(path) as f:
        return json.load(f)


def format_time(seconds: float) -> str:
    """Format time in appropriate units.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted string with appropriate unit (µs, ms, or s)
    """
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.1f}µs"
    elif seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    else:
        return f"{seconds:.2f}s"


def compare_benchmarks(
    baseline_path: Path, current_path: Path, threshold: float = 0.15
) -> str:
    """Compare benchmark results and generate markdown table.

    Args:
        baseline_path: Path to baseline results JSON
        current_path: Path to current results JSON
        threshold: Maximum acceptable regression as decimal (default 0.15 = 15%)

    Returns:
        Markdown formatted comparison table with summary

    Raises:
        FileNotFoundError: If either file doesn't exist
        KeyError: If JSON structure is unexpected
    """
    baseline = load_json(baseline_path)
    current = load_json(current_path)

    # Create lookup for baseline results
    baseline_lookup = {b["name"]: b["stats"]["mean"] for b in baseline["benchmarks"]}

    # Build comparison table
    rows = []
    regressions = []
    improvements = []
    new_benchmarks = []

    for bench in current["benchmarks"]:
        name = bench["name"]
        current_mean = bench["stats"]["mean"]
        current_stddev = bench["stats"]["stddev"]

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
            elif change_pct < -5:  # Improvement > 5%
                status = "🚀"
                improvements.append((name, abs(change_pct)))
            else:
                status = "✅"

            rows.append(
                f"| {name} | {baseline_str} | {current_str} | {change_str} | {status} |"
            )
        else:
            # New benchmark
            current_str = format_time(current_mean)
            rows.append(f"| {name} | N/A | {current_str} | NEW | 🆕 |")
            new_benchmarks.append(name)

    # Build markdown
    lines = [
        "## 📊 Benchmark Results",
        "",
        "| Benchmark | Main | PR | Change | Status |",
        "|-----------|------|-----|--------|--------|",
    ]
    lines.extend(rows)
    lines.append("")

    # Summary sections
    if regressions:
        lines.append("### ⚠️ Performance Regressions Detected")
        lines.append("")
        for name, pct in regressions:
            lines.append(
                f"- `{name}`: **{pct:+.1f}%** (exceeds {threshold*100:.0f}% threshold)"
            )
        lines.append("")

    if improvements:
        lines.append("### 🚀 Performance Improvements")
        lines.append("")
        for name, pct in improvements:
            lines.append(f"- `{name}`: **-{pct:.1f}%** faster")
        lines.append("")

    if new_benchmarks:
        lines.append("### 🆕 New Benchmarks")
        lines.append("")
        for name in new_benchmarks:
            lines.append(f"- `{name}`")
        lines.append("")

    if not regressions and not new_benchmarks:
        lines.append(
            f"**Summary:** All benchmarks within acceptable range (< {threshold*100:.0f}% regression)"
        )
        lines.append("")

    # Notes
    lines.extend(
        [
            "---",
            "",
            "**Notes:**",
            "- Benchmarks run on local machine (results may vary from CI)",
            f"- Regression threshold: {threshold*100:.0f}% (configurable via `BENCHMARK_MAX_REGRESSION`)",
            "- For details, see [benchmarking guide](https://talmolab.github.io/sleap-roots/dev/benchmarking/)",
        ]
    )

    return "\n".join(lines)


def show_results_without_baseline(current_path: Path) -> str:
    """Generate markdown table for current results without baseline comparison.

    Args:
        current_path: Path to current results JSON

    Returns:
        Markdown formatted table showing current benchmark results
    """
    data = load_json(current_path)

    lines = [
        "## 📊 Benchmark Results (No Baseline)",
        "",
        "| Benchmark | Mean | StdDev |",
        "|-----------|------|--------|",
    ]

    for bench in data["benchmarks"]:
        name = bench["name"]
        mean = bench["stats"]["mean"]
        stddev = bench["stats"]["stddev"]
        lines.append(
            f"| {name} | {format_time(mean)} | ±{format_time(stddev)} |"
        )

    lines.extend(
        [
            "",
            "**Note:** No baseline available for comparison. These are the first benchmark results.",
            "",
            "A baseline will be created after this PR is merged and benchmarks run on the main branch.",
        ]
    )

    return "\n".join(lines)


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Compare benchmark results and generate markdown table",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare with default paths
  python compare-benchmarks.py

  # Use custom baseline
  python compare-benchmarks.py --baseline .benchmarks/baselines/abc123.json

  # Set custom threshold (10% instead of 15%)
  python compare-benchmarks.py --threshold 0.10
        """,
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path(".benchmarks/baselines/main.json"),
        help="Path to baseline results JSON (default: .benchmarks/baselines/main.json)",
    )
    parser.add_argument(
        "--current",
        type=Path,
        default=Path("benchmark-results.json"),
        help="Path to current results JSON (default: benchmark-results.json)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.15,
        help="Maximum acceptable regression as decimal (default: 0.15 = 15%%)",
    )

    args = parser.parse_args()

    # Check if current results exist
    if not args.current.exists():
        print(
            f"Error: {args.current} not found. Run benchmarks first:",
            file=sys.stderr,
        )
        print(
            "  uv run pytest tests/benchmarks/ --benchmark-only --benchmark-json=benchmark-results.json",
            file=sys.stderr,
        )
        sys.exit(1)

    # Compare or show results
    try:
        if not args.baseline.exists():
            print(
                f"No baseline found at {args.baseline}. Showing current results only.",
                file=sys.stderr,
            )
            print(file=sys.stderr)
            markdown = show_results_without_baseline(args.current)
        else:
            markdown = compare_benchmarks(args.baseline, args.current, args.threshold)

        print(markdown)

    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error parsing JSON: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()