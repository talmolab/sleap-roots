"""Copy the run-scoping ``RunManifest`` forward (bloomctl's cross-repo scoping shape).

This is a *different* concept from ``manifest.py``'s ``PredictionManifest``/``ScanMetadata``:
``RunManifest`` scopes an entire batch run to a set of ``scan_keys`` (contamination prevention
across pipeline runs), while ``PredictionManifest`` describes one scan's predict output. Kept
in a separate module so "manifest" never means two different things in one place (see
talmolab/sleap-roots-pipeline#37).

Loading is not done here: ``extract_batch`` loads the manifest once with contracts'
``load_run_manifest`` and hands the resulting read to :func:`copy_run_manifest_forward`, so
the bytes forwarded are the bytes scoped against. Two deliberate differences from
``sleap-roots-predict``'s forwarder: a copy failure is best-effort here (the caller logs it
rather than failing the batch), and cleanup also covers ``BaseException`` so a SIGTERM's
``SystemExit`` mid-forward leaves no temp file.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Union

from sleap_roots_contracts import RunManifestRead

logger = logging.getLogger(__name__)


def copy_run_manifest_forward(
    read: RunManifestRead, input_dir: Union[str, Path], output_dir: Union[str, Path]
) -> None:
    """Republish an already-read run manifest into ``output_dir`` under the name it was read.

    Writes ``read.data`` -- the exact bytes ``extract_batch`` scoped against, never a
    re-serialization through the frozen ``RunManifest`` model and never a second read of
    the source -- to ``output_dir / read.filename`` (the per-run
    ``run_manifest.<pipeline_run_id>.json`` or the legacy ``run_manifest.json``), so the
    next pipeline stage (``write-back``) can see it without a new Argo mount.

    The publish is atomic: the bytes go to a uniquely named, dot-prefixed temporary file
    inside ``output_dir`` (same filesystem, so the replace is never cross-device on NFS;
    unique, so concurrent writers never share a temp path; dot-prefixed, so an orphan
    left by a SIGKILL is hidden from a ``run_manifest*`` glob), its mode is set to
    ``read.mode`` (``mkstemp`` creates at ``0600``, and write-back runs as a different uid
    on the shared mount), and only then is it moved into place with ``os.replace``.

    Args:
        read: The read ``extract_batch`` loaded (``LoadedRunManifest.read``).
        input_dir: The directory ``read`` came from.
        output_dir: Directory to publish the manifest into (created if missing).

    Returns:
        None. A no-op when ``input_dir / read.filename`` and ``output_dir /
        read.filename`` resolve to the same path (e.g. ``input_dir == output_dir``) --
        forwarding is already satisfied.

    Raises:
        OSError: If publishing fails (e.g. a disk or permission error, or a temp name
            over ``NAME_MAX`` for a very long run id). Removal of the temporary file is
            attempted first; a removal failure is logged, not raised. The caller
            (``extract_batch``) treats this as best-effort infrastructure and logs it
            rather than aborting the batch. Any other exception raised mid-publish --
            including the SIGTERM handler's ``SystemExit`` -- gets the same cleanup and
            propagates unchanged.
    """
    destination_dir = Path(output_dir)
    destination = destination_dir / read.filename
    # Path identity after symlink resolution, not inode identity: two hardlinked paths
    # would get a replace of identical bytes, which breaks the link but loses no data.
    # os.path.realpath rather than Path.resolve(): on Python 3.12 (the image's) resolve()
    # raises RuntimeError on a symlink loop, which would escape the caller's best-effort
    # `except OSError` after every envelope is already written.
    source_path = os.path.realpath(Path(input_dir) / read.filename)
    if source_path == os.path.realpath(destination):
        return
    destination_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=destination_dir, prefix=f".{read.filename}.", suffix=".tmp"
    )
    tmp = Path(tmp_name)
    try:
        # Write through mkstemp's fd and close it before anything else: on Windows an
        # open handle makes both os.replace and the cleanup unlink fail (WinError 32).
        try:
            handle = os.fdopen(fd, "wb")
        except BaseException:
            # fdopen may not have taken ownership, so `with` can't close it. If it did
            # (and already closed the fd on its own error path), this is EBADF: ignore.
            try:
                os.close(fd)
            except OSError:
                pass
            raise
        with handle:
            handle.write(read.data)
        # Before the replace, never after: after would briefly publish at 0600.
        os.chmod(tmp, read.mode)
        os.replace(tmp, destination)
    except BaseException:
        # BaseException, not Exception: the SIGTERM handler's SystemExit(143) must not
        # orphan the temp file either. Re-raised unchanged below.
        try:
            tmp.unlink(missing_ok=True)
        except OSError as cleanup_exc:
            logger.warning(
                "could not remove temporary file %s: %s", tmp.as_posix(), cleanup_exc
            )
        raise
