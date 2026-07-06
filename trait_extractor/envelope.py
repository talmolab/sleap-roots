"""Assemble Provenance + ResultEnvelope and write it atomically to disk."""

import importlib.metadata
import os
from pathlib import Path
from typing import List, Union

import sleap_roots
from sleap_roots_contracts import Provenance, ResultEnvelope, TraitValue

from trait_extractor.manifest import PredictionManifest, ScanMetadata


def contract_version() -> str:
    """Return the pinned bare ``sleap-roots-contracts`` package version."""
    return importlib.metadata.version("sleap-roots-contracts")


def _resolve_identity(explicit: str, env_var: str) -> str:
    """Resolve a build-identity field fail-soft: explicit arg, then env, then ''."""
    if explicit:
        return explicit
    return os.environ.get(env_var, "")


def build_provenance(
    manifest: PredictionManifest,
    sidecar: ScanMetadata,
    *,
    traits_code_sha: str = "",
    traits_container_digest: str = "",
) -> Provenance:
    """Build the ``Provenance`` (with a deterministic idempotency key).

    predict-stage fields come from the manifest; ``inputs``/``params`` from the
    sidecar; ``predict_models`` is derived from the artifacts. ``produced_at`` and the
    orchestration fields are left ``None`` for byte-stable re-emission. ``traits_*``
    build identity resolves fail-soft (arg -> env -> "").

    Args:
        manifest: The consumed prediction manifest.
        sidecar: The scan-metadata sidecar.
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
        params=sidecar.to_resolved_params(),
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
    tmp.write_text(data, encoding="utf-8")
    tmp.replace(path)
    return path
