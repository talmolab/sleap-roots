"""Orchestrate a per-scan extraction: manifest + sidecar -> ResultEnvelope JSON."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from sleap_roots_contracts import ResultEnvelope

from trait_extractor.compatibility import check_pipeline_compatible
from trait_extractor.envelope import (
    build_envelope,
    build_provenance,
    read_existing_identity,
    write_envelope,
)
from trait_extractor.loading import load_series
from trait_extractor.manifest import load_manifest, load_scan_metadata
from trait_extractor.pipeline_chooser import (
    PipelineCard,
    choose_pipeline,
    load_pipeline_cards,
)
from trait_extractor.run_manifest import copy_run_manifest_forward, load_run_manifest
from trait_extractor.traits import compute_scan_traits, scan_trait_values

logger = logging.getLogger(__name__)

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
) -> Optional[ResultEnvelope]:
    """Extract one scan: load, select, compute, and emit a ``ResultEnvelope``.

    Builds the scan's ``Provenance`` (and its ``idempotency_key``) immediately after
    resolving ``params`` -- before ``Series`` loading, pipeline selection/compatibility,
    or trait computation -- and compares it against any pre-existing
    ``output_dir/{scan_key}.result.json``. If both ``idempotency_key`` and
    ``contract_version`` already match, the scan is skipped: none of the expensive steps
    run and ``None`` is returned. Otherwise the scan is computed and written/overwritten
    exactly as if skip-if-done did not exist, reusing the already-built ``provenance``
    for the final envelope.

    Args:
        manifest_path: Path to ``{scan_key}.predictions.json``.
        scan_metadata_path: Path to the co-located ``{scan_key}.scan_metadata.json``.
        output_dir: Directory to write ``{scan_key}.result.json`` into.
        traits_code_sha: Optional traits build code sha (else env, else "").
        traits_container_digest: Optional traits container digest (else env, else "").
        cards: Optional pipeline selection cards (defaults to the packaged YAML).

    Returns:
        The emitted ``ResultEnvelope`` (also written to ``output_dir``), or ``None`` if
        the scan was skipped because its output already matches.

    Raises:
        ValueError: On empty artifacts, scan_key mismatch, no/ambiguous pipeline
            match, or an incompatible/unsupported pipeline.
        FileNotFoundError: If a referenced ``.slp`` is missing.
        json.JSONDecodeError: If the manifest or sidecar is not valid JSON.
        pydantic.ValidationError: If the manifest or sidecar fails schema validation.
    """
    manifest_path = Path(manifest_path)
    manifest = load_manifest(manifest_path)
    sidecar = load_scan_metadata(scan_metadata_path, manifest.scan_key)
    params = sidecar.to_resolved_params()

    provenance = build_provenance(
        manifest,
        sidecar,
        params,
        traits_code_sha=traits_code_sha,
        traits_container_digest=traits_container_digest,
    )
    existing = read_existing_identity(output_dir, manifest.scan_key)
    if existing == (provenance.idempotency_key, provenance.contract_version):
        return None

    series = load_series(manifest, manifest_path.parent)
    pipeline_cls = choose_pipeline(params, cards or load_pipeline_cards())
    check_pipeline_compatible(series, pipeline_cls)

    traits = compute_scan_traits(series, pipeline_cls)
    values = scan_trait_values(traits, manifest.scan_key)
    envelope = build_envelope(provenance, values)
    write_envelope(envelope, output_dir)
    return envelope


@dataclass
class BatchResult:
    """Outcome of a batch run: succeeded/skipped scan keys and per-scan failures.

    The three lists are NOT guaranteed mutually exclusive: a duplicate-scan_key
    collision (two candidate files resolving to the same scan_key) records the
    *first*-discovered candidate's own outcome -- ``succeeded`` or ``skipped`` -- in
    the corresponding list, then ALSO records the collision itself as a ``failed``
    entry for that same scan_key. Always check ``ok`` (or scan ``failed`` directly) to
    detect this, rather than assuming a scan_key present in ``succeeded``/``skipped``
    is free of any associated failure.
    """

    succeeded: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    failed: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when every discovered scan succeeded or was skipped."""
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

    If a ``run_manifest.json`` (``RunManifest``) is present at the top level of
    ``input_dir``, discovery is scoped to exactly its ``scan_keys`` -- a
    ``{scan_key}.predictions.json`` present but not in ``scan_keys`` is silently
    excluded (contamination prevention), and a manifest-declared ``scan_key`` with no
    matching file is recorded as a per-scan failure. The manifest is copied forward
    into ``output_dir`` so ``write-back`` can see it. If no manifest is present,
    discovery falls back to **recursively** discovering every
    ``{scan_key}.predictions.json`` under ``input_dir`` (matching predict's per-scan
    ``out_dir/{scan_key}/`` batch layout), unchanged from before manifest support.

    In both cases, each manifest is paired with its co-located sidecar and one
    ``{scan_key}.result.json`` is written per scan to ``output_dir``. A scan whose
    output already matches (see ``extract_scan``'s skip-if-done check) is skipped, not
    recomputed. One scan's failure (bad manifest, missing sidecar, stem/scan_key
    disagreement, incompatible pipeline, ...) is recorded and does NOT discard the
    other scans' envelopes — the broad ``except`` lives ONLY here.

    Args:
        input_dir: Root to search for manifests.
        output_dir: Where to write result envelopes.
        traits_code_sha: Optional traits build code sha.
        traits_container_digest: Optional traits container digest.
        cards: Optional pipeline selection cards (defaults to the packaged YAML).

    Returns:
        A ``BatchResult`` summarizing successes, skips, and failures.

    Raises:
        pydantic.ValidationError: If ``run_manifest.json`` is present but fails
            ``RunManifest``'s own validation (e.g. empty ``scan_keys``) -- a
            once-per-batch, top-level file, not a per-scan best-effort read.
        OSError: If ``run_manifest.json`` is present but can't be read (see
            ``run_manifest.load_run_manifest``), for the same reason.
        UnicodeDecodeError: If ``run_manifest.json``'s bytes aren't valid UTF-8.
    """
    cards = cards or load_pipeline_cards()
    run_manifest = load_run_manifest(input_dir)
    scope = set(run_manifest.scan_keys) if run_manifest is not None else None
    result = BatchResult()
    seen: Dict[str, Path] = {}
    seen_casefold: Dict[str, str] = {}  # casefolded stem -> the original stem
    manifest_paths = sorted(Path(input_dir).rglob(_MANIFEST_GLOB))
    if scope is None and not manifest_paths:
        raise RuntimeError(
            f"no {_MANIFEST_SUFFIX} files found under {Path(input_dir).as_posix()}"
        )
    for manifest_path in manifest_paths:
        stem = manifest_path.name.removesuffix(_MANIFEST_SUFFIX)
        if scope is not None and stem not in scope:
            # Out of scope for this run: exactly the contamination this manifest
            # exists to prevent. Not processed, not reported anywhere.
            continue
        try:
            # Two manifests declaring the same scan_key would write the same
            # {scan_key}.result.json; refuse the collision rather than silently clobber
            # one envelope while reporting success (predict's producer guards this too).
            # This also protects the scoped path: a stale leftover directory alongside
            # the correct one for an in-scope scan_key is refused, not silently picked.
            # Note: the first-seen candidate is processed (and its envelope durably
            # written) before a later duplicate is even discovered, so a duplicate
            # collision leaves the first candidate's output on disk even though the
            # batch is reported as failed for that scan_key -- BatchResult.ok correctly
            # reports False, but a consumer checking only the output file's existence,
            # not BatchResult, would not see the collision.
            if stem in seen:
                raise ValueError(
                    f"duplicate scan_key {stem!r}: {manifest_path.as_posix()} collides "
                    f"with {seen[stem].as_posix()}"
                )
            # Two DIFFERENT scan_keys that only differ by case would still write to the
            # same output filename on a case-insensitive filesystem (default on Windows
            # and macOS -- the repo's own dev platform), silently overwriting one
            # envelope's output.json with the other's despite `seen`'s exact-string keys
            # never detecting a collision. RunManifest's own scan_keys validation is
            # exact-match only (documented as not even whitespace-normalized), so this
            # is not caught upstream either. Refuse it the same way an exact duplicate
            # is refused, rather than let it silently corrupt output_dir.
            folded = stem.casefold()
            if folded in seen_casefold and seen_casefold[folded] != stem:
                raise ValueError(
                    f"scan_key {stem!r} collides case-insensitively with "
                    f"{seen_casefold[folded]!r} ({seen[seen_casefold[folded]].as_posix()}) "
                    "-- their output files would collide on a case-insensitive filesystem"
                )
            seen[stem] = manifest_path
            seen_casefold[folded] = stem
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
            envelope = extract_scan(
                manifest_path,
                sidecar_path,
                output_dir,
                traits_code_sha=traits_code_sha,
                traits_container_digest=traits_container_digest,
                cards=cards,
            )
            if envelope is None:
                result.skipped.append(stem)
            else:
                result.succeeded.append(stem)
        except Exception as exc:  # noqa: BLE001 - isolation boundary (batch only)
            result.failed.append((stem, str(exc)))

    if scope is not None:
        for missing_scan_key in sorted(scope - seen.keys()):
            result.failed.append(
                (
                    missing_scan_key,
                    f"no {missing_scan_key}{_MANIFEST_SUFFIX} found under "
                    f"{Path(input_dir).as_posix()} for manifest-declared scan_key",
                )
            )
        # A scan_key that was in scope in a PRIOR run over this same output_dir but has
        # since dropped out of scope (a shrinking manifest) leaves that prior run's
        # {scan_key}.result.json sitting untouched -- not reprocessed, not reported in
        # any BatchResult bucket, not cleaned up. Never treated as this run's problem to
        # fix (this run only owns what's in ITS scope), but silent orphaning of files a
        # downstream consumer might still glob over (see bloom#678) is worth a trace.
        orphaned = sorted(
            path.name.removesuffix(".result.json")
            for path in Path(output_dir).glob("*.result.json")
            if path.name.removesuffix(".result.json") not in scope
        )
        if orphaned:
            logger.warning(
                "%d pre-existing result file(s) in %s are outside this run's scope "
                "(from a prior run's wider manifest): %s",
                len(orphaned),
                Path(output_dir).as_posix(),
                ", ".join(orphaned),
            )
        try:
            copy_run_manifest_forward(input_dir, output_dir)
        except OSError as exc:
            # Copy-forward is best-effort infrastructure for the next pipeline stage,
            # not part of this batch's own computed result -- a disk/permission error
            # here must not discard the already-computed (and already durably written)
            # succeeded/skipped/failed results above. Name both directories explicitly
            # rather than relying on the exception's own str() (not every OSError, e.g.
            # a disk-full error from a raw write, carries a filename attribute).
            logger.warning(
                "failed to copy run_manifest.json from %s to %s: %s",
                Path(input_dir).as_posix(),
                Path(output_dir).as_posix(),
                exc,
            )

    return result
