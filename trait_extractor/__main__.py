"""CLI entry point: ``python -m trait_extractor <input_dir> <output_dir>``.

The legacy ``main(input_dir, output_dir)`` analog and the container's entry command.
Discovers each ``{scan_key}.predictions.json`` under ``input_dir`` (scoped to a
``run_manifest.json``'s ``scan_keys`` when present), emits one
``{scan_key}.result.json`` per scan to ``output_dir``, and exits non-zero if any scan
failed (successful and skipped envelopes are still counted/reported).
"""

import argparse
import sys
from typing import List, Optional

from trait_extractor.extractor import extract_batch


def main(argv: Optional[List[str]] = None) -> int:
    """Run the batch extractor over a directory.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code: 0 if every scan succeeded, 1 otherwise.
    """
    parser = argparse.ArgumentParser(prog="trait_extractor")
    parser.add_argument("input_dir", help="Directory of predict per-scan outputs.")
    parser.add_argument("output_dir", help="Directory to write result envelopes into.")
    args = parser.parse_args(argv)

    result = extract_batch(args.input_dir, args.output_dir)
    for scan_key in result.succeeded:
        print(f"ok    {scan_key}")
    for scan_key in result.skipped:
        print(f"skip  {scan_key}")
    for scan_key, error in result.failed:
        print(f"FAIL  {scan_key}: {error}", file=sys.stderr)
    print(
        f"{len(result.succeeded)} succeeded, {len(result.skipped)} skipped, "
        f"{len(result.failed)} failed",
        file=sys.stderr,
    )
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
