"""CLI entry point: ``python -m trait_extractor <input_dir> <output_dir>``.

The legacy ``main(input_dir, output_dir)`` analog and the container's entry command.
Discovers each ``{scan_key}.predictions.json`` under ``input_dir`` (scoped to a
``run_manifest.json``'s ``scan_keys`` when present), emits one
``{scan_key}.result.json`` per scan to ``output_dir``, and exits with one of three
driver-owned codes: 0 (full success), 3 (partial -- isolated per-scan failures), or 1
(crash -- an exception escaped ``extract_batch`` entirely). Exit code 2 is reserved by
``argparse`` for CLI usage errors, not by this driver.
"""

import argparse
import logging
import signal
import sys
from typing import List, Optional

import pydantic
import yaml

from trait_extractor.extractor import extract_batch

logger = logging.getLogger(__name__)


def _handle_sigterm(signum, frame) -> None:
    """Log and exit promptly on SIGTERM (Argo preemption/cancellation).

    Per-scan writes are already atomic (temp->rename) and the batch is idempotent
    on retry, so an immediate exit here can only ever abandon an in-flight temp
    file, never corrupt a completed ``{scan_key}.result.json``.
    """
    logger.error("Received SIGTERM, exiting")
    sys.exit(143)


def main(argv: Optional[List[str]] = None) -> int:
    """Run the batch extractor over a directory.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code: 0 if every scan succeeded or was skipped, 3 if one or
        more scans isolated-failed but the batch ran to completion (matches
        ``BatchResult.ok``), or Python's default 1 if an exception escaped
        ``extract_batch`` entirely (e.g. an invalid ``run_manifest.json`` or an
        empty input directory). Exit code 2 is reserved by ``argparse`` for CLI
        usage errors and is never returned by this function.
    """
    signal.signal(signal.SIGTERM, _handle_sigterm)

    parser = argparse.ArgumentParser(prog="trait_extractor")
    parser.add_argument("input_dir", help="Directory of predict per-scan outputs.")
    parser.add_argument("output_dir", help="Directory to write result envelopes into.")
    args = parser.parse_args(argv)

    try:
        result = extract_batch(args.input_dir, args.output_dir)
    except (
        RuntimeError,
        pydantic.ValidationError,
        OSError,
        UnicodeDecodeError,
        yaml.YAMLError,
    ) as exc:
        logger.error("Batch aborted: %s", exc)
        raise

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
    return 0 if result.ok else 3


if __name__ == "__main__":
    sys.exit(main())
