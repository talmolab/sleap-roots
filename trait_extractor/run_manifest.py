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
        OSError: If the file exists but can't be read (e.g. a permissions issue) --
            not caught here for the same reason: this file's presence/validity is a
            once-per-batch precondition, not a per-scan best-effort check (contrast with
            ``envelope.read_existing_identity``, which deliberately treats the analogous
            per-scan case as "not done" rather than raising).
        UnicodeDecodeError: If the file's bytes aren't valid UTF-8.
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

    Raises:
        OSError: If the copy fails (e.g. a disk or permission error). The caller
            (``extract_batch``) treats this as best-effort infrastructure and catches
            it rather than letting it abort the batch.
    """
    source = Path(input_dir) / RUN_MANIFEST_FILENAME
    if not source.is_file():
        return
    destination_dir = Path(output_dir)
    destination = destination_dir / RUN_MANIFEST_FILENAME
    # Path-identity fast path: covers the common case (a caller invoking with
    # input_dir == output_dir, or an equivalent-but-differently-spelled path) without
    # touching the filesystem beyond the two resolve() calls. This is a path-string
    # comparison, not a device/inode identity check, so it can't catch every possible
    # same-file aliasing (e.g. two hardlinks) -- shutil.copyfile's own os.path.samefile
    # check is the backstop for that, surfacing as shutil.SameFileError, which the
    # caller (extract_batch) already catches as an OSError subclass.
    if source.resolve() == destination.resolve():
        return
    destination_dir.mkdir(parents=True, exist_ok=True)
    # tmp file + replace, matching envelope.write_envelope's atomicity convention, so a
    # crash/kill mid-copy never leaves a truncated run_manifest.json visible to a
    # downstream reader (write-back) -- the previous plain shutil.copyfile(..., dest)
    # wrote directly to the final path.
    tmp = destination.with_name(destination.name + ".tmp")
    shutil.copyfile(source, tmp)
    tmp.replace(destination)
