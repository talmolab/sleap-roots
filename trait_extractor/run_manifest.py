"""Load + copy-forward the run-scoping ``RunManifest`` (bloomctl's cross-repo scoping shape).

This is a *different* concept from ``manifest.py``'s ``PredictionManifest``/``ScanMetadata``:
``RunManifest`` scopes an entire batch run to a set of ``scan_keys`` (contamination prevention
across pipeline runs), while ``PredictionManifest`` describes one scan's predict output. Kept
in a separate module so "manifest" never means two different things in one place (see
talmolab/sleap-roots-pipeline#37).
"""

import shutil
from pathlib import Path
from typing import Optional, Union

from sleap_roots_contracts import RUN_MANIFEST_FILENAME, RunManifest


def load_run_manifest(input_dir: Union[str, Path]) -> Optional[RunManifest]:
    """Load and validate ``run_manifest.json`` from the top level of ``input_dir``.

    Args:
        input_dir: Directory to look for ``RUN_MANIFEST_FILENAME`` in (not searched
            recursively; the manifest is written once at the top level of the shared
            staging directory by ``bloomctl``).

    Returns:
        The validated ``RunManifest``, or ``None`` if no manifest file is present.

    Raises:
        pydantic.ValidationError: If the file is not valid JSON, or is valid JSON that
            fails ``RunManifest``'s validation (e.g. empty ``scan_keys``) --
            ``model_validate_json`` parses and validates in one step, so both cases raise
            this same exception type. This is a top-level, once-per-batch file, so an
            invalid manifest raises rather than being treated as a per-scan best-effort
            read.
    """
    path = Path(input_dir) / RUN_MANIFEST_FILENAME
    if not path.is_file():
        return None
    return RunManifest.model_validate_json(path.read_text(encoding="utf-8"))


def copy_run_manifest_forward(
    input_dir: Union[str, Path], output_dir: Union[str, Path]
) -> None:
    """Copy ``run_manifest.json`` from ``input_dir`` into ``output_dir``, if present.

    A raw file copy (not a re-serialization through the frozen ``RunManifest`` model), so
    the forwarded file is byte-identical to what ``bloomctl``/the prior stage wrote —
    matching predict's established "copy the sidecar forward" convention so the next
    pipeline stage (``write-back``) can see the manifest without a new Argo mount.

    Args:
        input_dir: Directory to look for ``RUN_MANIFEST_FILENAME`` in.
        output_dir: Directory to copy the manifest into (created if missing).

    Returns:
        None. No-op if no manifest is present under ``input_dir``, or if ``input_dir``
        and ``output_dir`` already resolve to the same file (e.g. a caller invoking with
        ``input_dir == output_dir``) -- copying forward is trivially already satisfied.
    """
    source = Path(input_dir) / RUN_MANIFEST_FILENAME
    if not source.is_file():
        return
    destination_dir = Path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / RUN_MANIFEST_FILENAME
    if source.resolve() == destination.resolve():
        return
    shutil.copyfile(source, destination)
