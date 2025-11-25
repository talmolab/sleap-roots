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
from typing import Dict, List, Optional, Tuple

# Benchmark sample information - maps test names to dataset details from test fixtures
# See tests/fixtures/data.py and tests/benchmarks/test_pipeline_performance.py
BENCHMARK_SAMPLES = {
    "test_dicot_pipeline_performance": {
        "dataset": "canola_7do",
        "sample_id": "919QDUH",
        "data_path": "tests/data/canola_7do/",
        "files": [
            "919QDUH.h5",
            "919QDUH.primary.predictions.slp",
            "919QDUH.lateral.predictions.slp",
        ],
        "root_types": "Primary + Lateral",
        "description": "7 day old canola with primary and lateral root predictions",
    },
    "test_younger_monocot_pipeline_performance": {
        "dataset": "rice_3do",
        "sample_id": "YR39SJX",
        "data_path": "tests/data/rice_3do/",
        "files": [
            "YR39SJX.h5",
            "YR39SJX.primary.predictions.slp",
            "YR39SJX.crown.predictions.slp",
        ],
        "root_types": "Primary + Crown",
        "description": "3 day old rice with primary (longest) and crown root predictions",
    },
    "test_older_monocot_pipeline_performance": {
        "dataset": "rice_10do",
        "sample_id": "0K9E8BI",
        "data_path": "tests/data/rice_10do/",
        "files": ["0K9E8BI.h5", "0K9E8BI.crown.predictions.slp"],
        "root_types": "Crown only",
        "description": "10 day old rice with crown root predictions only",
    },
    "test_primary_root_pipeline_performance": {
        "dataset": "canola_7do",
        "sample_id": "919QDUH",
        "data_path": "tests/data/canola_7do/",
        "files": ["919QDUH.h5", "919QDUH.primary.predictions.slp"],
        "root_types": "Primary only",
        "description": "7 day old canola with primary root predictions only",
    },
    "test_lateral_root_pipeline_performance": {
        "dataset": "canola_7do",
        "sample_id": "919QDUH",
        "data_path": "tests/data/canola_7do/",
        "files": ["919QDUH.h5", "919QDUH.lateral.predictions.slp"],
        "root_types": "Lateral only",
        "description": "7 day old canola with lateral root predictions only",
    },
    "test_multiple_dicot_pipeline_performance": {
        "dataset": "multiple_arabidopsis_11do",
        "sample_id": "997_1",
        "data_path": "tests/data/multiple_arabidopsis_11do/",
        "files": [
            "997_1.h5",
            "997_1.primary.predictions.slp",
            "997_1.lateral.predictions.slp",
        ],
        "root_types": "Primary + Lateral (3 plants)",
        "description": "11 day old Arabidopsis with 3 plants per image",
    },
    "test_multiple_primary_root_pipeline_performance": {
        "dataset": "multiple_arabidopsis_11do",
        "sample_id": "997_1",
        "data_path": "tests/data/multiple_arabidopsis_11do/",
        "files": ["997_1.h5", "997_1.primary.predictions.slp"],
        "root_types": "Primary only (3 plants)",
        "description": "11 day old Arabidopsis with 3 plants per image, primary roots only",
    },
}


def get_sample_info(benchmark_name: str) -> Optional[Dict]:
    """Get sample information for a benchmark test.

    Args:
        benchmark_name: Name of the benchmark test

    Returns:
        Dictionary with sample details or None if not found
    """
    return BENCHMARK_SAMPLES.get(benchmark_name)


def format_sample_details() -> str:
    """Generate markdown section with benchmark sample details.

    Returns:
        Markdown formatted sample information section
    """
    lines = [
        "",
        "### 📁 Test Data Details",
        "",
        "| Benchmark | Dataset | Sample ID | Root Types |",
        "|-----------|---------|-----------|------------|",
    ]

    for name, info in BENCHMARK_SAMPLES.items():
        # Shorten benchmark name for display
        short_name = name.replace("_performance", "").replace("test_", "")
        lines.append(
            f"| {short_name} | `{info['dataset']}` | `{info['sample_id']}` | {info['root_types']} |"
        )

    lines.extend(
        [
            "",
            "<details>",
            "<summary>Full sample information</summary>",
            "",
        ]
    )

    for name, info in BENCHMARK_SAMPLES.items():
        short_name = name.replace("_performance", "").replace("test_", "")
        lines.append(f"**{short_name}**")
        lines.append(f"- Dataset: `{info['dataset']}`")
        lines.append(f"- Sample ID: `{info['sample_id']}`")
        lines.append(f"- Path: `{info['data_path']}`")
        lines.append(f"- Files: {', '.join(f'`{f}`' for f in info['files'])}")
        lines.append(f"- Description: {info['description']}")
        lines.append("")

    lines.append("</details>")

    return "\n".join(lines)


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

    # Add sample details section
    lines.append(format_sample_details())

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
        lines.append(f"| {name} | {format_time(mean)} | ±{format_time(stddev)} |")

    lines.extend(
        [
            "",
            "**Note:** No baseline available for comparison. These are the first benchmark results.",
            "",
            "A baseline will be created after this PR is merged and benchmarks run on the main branch.",
        ]
    )

    # Add sample details section
    lines.append(format_sample_details())

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
        # Check if baseline exists and is valid JSON
        baseline_valid = False
        if args.baseline.exists():
            try:
                with open(args.baseline) as f:
                    content = f.read().strip()
                    if content:  # Not empty
                        json.loads(content)
                        baseline_valid = True
            except (json.JSONDecodeError, ValueError):
                pass

        if not baseline_valid:
            print(
                f"No valid baseline found at {args.baseline}. Showing current results only.",
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
