"""Assemble Provenance + ResultEnvelope and write it atomically to disk."""

import functools
import importlib.metadata
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pydantic
import sleap_roots
from sleap_roots_contracts import Provenance, ResolvedParams, ResultEnvelope, TraitValue

from trait_extractor.manifest import PredictionManifest, ScanMetadata

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def contract_version() -> str:
    """Return the pinned bare ``sleap-roots-contracts`` package version.

    Cached: the installed version cannot change mid-process, so avoid re-scanning
    dist-info once per scan in a batch.
    """
    return importlib.metadata.version("sleap-roots-contracts")


def _resolve_identity(explicit: str, env_var: str) -> str:
    """Resolve a build-identity field fail-soft: explicit arg, then env, then ''."""
    if explicit:
        return explicit
    return os.environ.get(env_var, "")


def build_provenance(
    manifest: PredictionManifest,
    sidecar: ScanMetadata,
    params: ResolvedParams,
    *,
    traits_code_sha: str = "",
    traits_container_digest: str = "",
) -> Provenance:
    """Build the ``Provenance`` (with a deterministic idempotency key).

    predict-stage fields come from the manifest; ``inputs`` from the sidecar; ``params``
    is the single canonical ``ResolvedParams`` the caller also feeds to pipeline
    selection; ``predict_models`` is derived from the artifacts. ``produced_at`` and the
    orchestration fields are left ``None`` for byte-stable re-emission. ``traits_*``
    build identity resolves fail-soft (arg -> env -> "").

    Args:
        manifest: The consumed prediction manifest.
        sidecar: The scan-metadata sidecar.
        params: The canonical ``ResolvedParams`` (shared with pipeline selection).
        traits_code_sha: Optional traits build code sha.
        traits_container_digest: Optional traits container digest.

    Returns:
        A validated ``Provenance`` whose ``idempotency_key`` is auto-derived.
    """
    return Provenance(
        contract_version=contract_version(),
        scan_key=manifest.scan_key,
        inputs=sidecar.to_input_ref(),
        predict_models=[artifact.model for artifact in manifest.artifacts],
        predict_container_digest=manifest.predict_container_digest,
        predict_code_sha=manifest.predict_code_sha,
        predict_inference_config=manifest.predict_inference_config,
        predict_output_params=manifest.predict_output_params,
        traits_sleap_roots_version=sleap_roots.__version__,
        traits_container_digest=_resolve_identity(
            traits_container_digest, "SRT_TRAITS_CONTAINER_DIGEST"
        ),
        traits_code_sha=_resolve_identity(traits_code_sha, "SRT_TRAITS_CODE_SHA"),
        params=params,
    )


def build_envelope(provenance: Provenance, traits: List[TraitValue]) -> ResultEnvelope:
    """Assemble the per-scan ``ResultEnvelope`` (no blobs this slice)."""
    return ResultEnvelope(provenance=provenance, traits=traits, blobs=[])


def write_envelope(envelope: ResultEnvelope, output_dir: Union[str, Path]) -> Path:
    """Write ``{scan_key}.result.json`` atomically under ``output_dir``.

    Creates ``output_dir`` if missing and writes via a temp file + ``replace`` so a
    mid-write crash never leaves a partial file. ``produced_at`` is ``None``, so
    re-emitting over identical inputs yields byte-identical output.

    Args:
        envelope: The envelope to serialize.
        output_dir: The destination directory (created if missing).

    Returns:
        The path written.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{envelope.provenance.scan_key}.result.json"
    data = envelope.model_dump_json(indent=2)
    tmp = path.with_name(path.name + ".tmp")
    # newline="\n" so the file is byte-identical across OSes (default translation would
    # write CRLF on Windows, breaking the byte-stable re-emission guarantee cross-platform).
    tmp.write_text(data, encoding="utf-8", newline="\n")
    tmp.replace(path)
    return path


def read_existing_identity(
    output_dir: Union[str, Path], scan_key: str
) -> Optional[Tuple[str, str]]:
    """Read a pre-existing ``{scan_key}.result.json``'s identity, for skip-if-done.

    A best-effort read, not a hard dependency: a missing, corrupt, or schema-invalid
    file is treated as "not done" (the scan will be recomputed) rather than raising. A
    plain missing file is the ordinary first-run case and is not logged; a *present but
    unreadable/invalid* file is logged as a warning, since that's an anomaly a researcher
    auditing results later would want a trace of (a prior crashed run, a corrupted file,
    someone else's process writing into ``output_dir``).

    Args:
        output_dir: Directory to look for ``{scan_key}.result.json`` in.
        scan_key: The scan's identifier (also the output filename stem).

    Returns:
        ``(idempotency_key, contract_version)`` from the pre-existing envelope, or
        ``None`` if no valid pre-existing envelope exists.
    """
    path = Path(output_dir) / f"{scan_key}.result.json"
    try:
        # is_file() lives inside this try too: it already swallows ENOENT/ENOTDIR-style
        # "doesn't exist" errors internally and returns False for those (the ordinary,
        # unlogged missing-file case below), but a PermissionError from the underlying
        # stat() call is NOT one of the errors it swallows -- it propagates. Catching it
        # here, alongside the read/parse errors, keeps the whole "does a usable
        # pre-existing envelope exist" check inside one best-effort boundary.
        if not path.is_file():
            return None
        envelope = ResultEnvelope.model_validate_json(path.read_text(encoding="utf-8"))
    except (pydantic.ValidationError, UnicodeDecodeError, OSError) as exc:
        # model_validate_json parses + validates in one step, so pydantic.ValidationError
        # (re-exported from pydantic_core) covers both malformed JSON and schema-invalid
        # JSON -- there is no separate json.JSONDecodeError case to catch here. OSError
        # covers a read-time anomaly on a file that exists (e.g. a permissions issue, or
        # a TOCTOU race where it's deleted in between) -- without this, such an error
        # would misclassify the scan as FAILED instead of "not done", contradicting the
        # documented "missing, unreadable, or invalid -> not done" contract.
        logger.warning(
            "existing %s is unreadable or fails ResultEnvelope validation; "
            "treating as not done and recomputing: %s",
            path.as_posix(),
            exc,
        )
        return None
    return envelope.provenance.idempotency_key, envelope.provenance.contract_version
