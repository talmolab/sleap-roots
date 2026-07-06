"""Orchestrate a per-scan extraction: manifest + sidecar -> ResultEnvelope JSON."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

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

_MANIFEST_SUFFIX = ".predictions.json"
_MANIFEST_GLOB = "*" + _MANIFEST_SUFFIX
_SIDECAR_SUFFIX = ".scan_metadata.json"


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
        params,
        traits_code_sha=traits_code_sha,
        traits_container_digest=traits_container_digest,
    )
    envelope = build_envelope(provenance, values)
    write_envelope(envelope, output_dir)
    return envelope


@dataclass
class BatchResult:
    """Outcome of a batch run: succeeded scan keys and per-scan failures."""

    succeeded: List[str] = field(default_factory=list)
    failed: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when every discovered scan succeeded."""
        return not self.failed


def extract_batch(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    traits_code_sha: str = "",
    traits_container_digest: str = "",
    cards: Optional[List[PipelineCard]] = None,
) -> BatchResult:
    """Extract every scan discovered under ``input_dir`` with per-scan isolation.

    Recursively discovers ``{scan_key}.predictions.json`` (matching predict's per-scan
    ``out_dir/{scan_key}/`` batch layout), pairs each with its co-located sidecar, and
    writes one ``{scan_key}.result.json`` per scan to ``output_dir``. One scan's failure
    (bad manifest, missing sidecar, stem/scan_key disagreement, incompatible pipeline,
    ...) is recorded and does NOT discard the other scans' envelopes — the broad
    ``except`` lives ONLY here.

    Args:
        input_dir: Root to search for manifests.
        output_dir: Where to write result envelopes.
        traits_code_sha: Optional traits build code sha.
        traits_container_digest: Optional traits container digest.
        cards: Optional pipeline selection cards (defaults to the packaged YAML).

    Returns:
        A ``BatchResult`` summarizing successes and failures.
    """
    cards = cards or load_pipeline_cards()
    result = BatchResult()
    for manifest_path in sorted(Path(input_dir).rglob(_MANIFEST_GLOB)):
        stem = manifest_path.name.removesuffix(_MANIFEST_SUFFIX)
        try:
            sidecar_path = manifest_path.parent / f"{stem}{_SIDECAR_SUFFIX}"
            if not sidecar_path.exists():
                raise FileNotFoundError(
                    f"no scan-metadata sidecar for {stem!r}: {sidecar_path.as_posix()}"
                )
            manifest = load_manifest(manifest_path)
            if manifest.scan_key != stem:
                raise ValueError(
                    f"manifest filename stem {stem!r} != scan_key "
                    f"{manifest.scan_key!r}"
                )
            extract_scan(
                manifest_path,
                sidecar_path,
                output_dir,
                traits_code_sha=traits_code_sha,
                traits_container_digest=traits_container_digest,
                cards=cards,
            )
            result.succeeded.append(stem)
        except Exception as exc:  # noqa: BLE001 - isolation boundary (batch only)
            result.failed.append((stem, str(exc)))
    return result
