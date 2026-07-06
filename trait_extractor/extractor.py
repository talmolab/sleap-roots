"""Orchestrate a per-scan extraction: manifest + sidecar -> ResultEnvelope JSON."""

from pathlib import Path
from typing import List, Optional, Union

from sleap_roots_contracts import ResultEnvelope

from trait_extractor.compatibility import check_pipeline_compatible
from trait_extractor.envelope import build_envelope, build_provenance, write_envelope
from trait_extractor.loading import load_series
from trait_extractor.manifest import load_manifest, load_scan_metadata
from trait_extractor.pipeline_chooser import (
    PipelineCard,
    choose_pipeline,
    load_pipeline_cards,
)
from trait_extractor.traits import compute_scan_traits, scan_trait_values


def extract_scan(
    manifest_path: Union[str, Path],
    scan_metadata_path: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    traits_code_sha: str = "",
    traits_container_digest: str = "",
    cards: Optional[List[PipelineCard]] = None,
) -> ResultEnvelope:
    """Extract one scan: load, select, compute, and emit a ``ResultEnvelope``.

    Args:
        manifest_path: Path to ``{scan_key}.predictions.json``.
        scan_metadata_path: Path to the co-located ``{scan_key}.scan_metadata.json``.
        output_dir: Directory to write ``{scan_key}.result.json`` into.
        traits_code_sha: Optional traits build code sha (else env, else "").
        traits_container_digest: Optional traits container digest (else env, else "").
        cards: Optional pipeline selection cards (defaults to the packaged YAML).

    Returns:
        The emitted ``ResultEnvelope`` (also written to ``output_dir``).

    Raises:
        ValueError: On empty artifacts, scan_key mismatch, no/ambiguous pipeline
            match, or an incompatible/unsupported pipeline.
        FileNotFoundError: If a referenced ``.slp`` is missing.
    """
    manifest_path = Path(manifest_path)
    manifest = load_manifest(manifest_path)
    sidecar = load_scan_metadata(scan_metadata_path, manifest.scan_key)
    params = sidecar.to_resolved_params()

    series = load_series(manifest, manifest_path.parent)
    pipeline_cls = choose_pipeline(params, cards or load_pipeline_cards())
    check_pipeline_compatible(series, pipeline_cls)

    traits = compute_scan_traits(series, pipeline_cls)
    values = scan_trait_values(traits, manifest.scan_key)
    provenance = build_provenance(
        manifest,
        sidecar,
        traits_code_sha=traits_code_sha,
        traits_container_digest=traits_container_digest,
    )
    envelope = build_envelope(provenance, values)
    write_envelope(envelope, output_dir)
    return envelope
