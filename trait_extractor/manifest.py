"""Consumer-side models for predict's output contract + the scan-metadata sidecar.

These re-declare a *consumer's view* of ``sleap-roots-predict``'s ``PredictionManifest``
(``schema_version: "1"``) so the trait-extractor never takes a runtime dependency on
``sleap-roots-predict`` (which pulls the GPU/torch/sleap-nn stack). Only the lightweight
``ModelRef`` / ``RootType`` types are imported from ``sleap-roots-contracts``. A
skip-if-unimportable cross-check test guards against schema drift.

The ``ScanMetadata`` sidecar is a NEW contract this service defines: it supplies the
idempotency inputs predict defers (``image_ids`` / ``images_checksum`` and the resolved
``params``). To keep the idempotency key reproducible, ``to_resolved_params`` builds
``ResolvedParams.values`` as the closed set ``{species, mode, age}`` with ``age``
canonicalized to ``int`` — so ``age`` encoded as ``3`` / ``3.0`` / ``"3"`` and any extra
sidecar keys never change the key.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Union

from pydantic import BaseModel, ConfigDict
from sleap_roots_contracts import InputRef, ModelRef, ResolvedParams, RootType


class PredictionArtifact(BaseModel):
    """One predicted root type: its model identity and the per-root ``.slp``."""

    model_config = ConfigDict(protected_namespaces=())

    root_type: RootType
    model_id: str
    model: ModelRef
    slp_path: str
    checksum: str
    file_size: int


class PredictionManifest(BaseModel):
    """Consumer view of predict's per-scan ``{scan_key}.predictions.json``."""

    model_config = ConfigDict(protected_namespaces=())

    schema_version: Literal["1"]
    scan_key: str
    plant_qr_code: str = ""
    artifacts: List[PredictionArtifact]
    predict_inference_config: Union[Dict[str, Any], None] = None
    predict_output_params: Union[Dict[str, Any], None] = None
    predict_code_sha: str = ""
    predict_container_digest: str = ""


def _canonical_age(raw: Any) -> int:
    """Coerce a sidecar ``age`` to a whole-number ``int``.

    Rejects ``bool`` and any non-whole value so that ``age`` encoded as ``3``,
    ``3.0``, or ``"3"`` all canonicalize to the same ``int`` (keeping the idempotency
    key stable), while ``3.5`` / ``"abc"`` raise.

    Args:
        raw: The raw ``age`` value from the sidecar ``params``.

    Returns:
        The canonical integer age.

    Raises:
        ValueError: If ``raw`` is a bool or is not a whole number.
    """
    if isinstance(raw, bool):
        raise ValueError(f"age must be an integer, got bool {raw!r}")
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        if raw.is_integer():
            return int(raw)
        raise ValueError(f"age must be a whole number, got {raw!r}")
    if isinstance(raw, str):
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(f"age must be a whole number, got {raw!r}") from exc
        return value
    raise ValueError(f"age must be a whole number, got {raw!r}")


class ScanMetadata(BaseModel):
    """Per-scan sidecar supplying the idempotency inputs predict defers."""

    scan_key: str
    image_ids: List[str]
    images_checksum: str
    params: Dict[str, Any]

    def to_input_ref(self) -> InputRef:
        """Build the ``InputRef`` (image ids + checksum) for provenance."""
        return InputRef(
            image_ids=list(self.image_ids), images_checksum=self.images_checksum
        )

    def to_resolved_params(self) -> ResolvedParams:
        """Build a canonical ``ResolvedParams`` = closed ``{species, mode, age:int}``.

        Extra sidecar ``params`` keys are excluded so they never enter the
        ``param_hash``; ``age`` is canonicalized to ``int``.

        Returns:
            A ``ResolvedParams`` whose ``values`` is exactly ``{species, mode, age}``.

        Raises:
            ValueError: If a required key is missing or ``age`` is not a whole number.
        """
        try:
            species = self.params["species"]
            mode = self.params["mode"]
            age = self.params["age"]
        except KeyError as exc:
            raise ValueError(f"sidecar params missing required key: {exc}") from exc
        values = {
            "species": str(species),
            "mode": str(mode),
            "age": _canonical_age(age),
        }
        return ResolvedParams(values=values)


def load_manifest(path: Union[str, Path]) -> PredictionManifest:
    """Load and validate a ``{scan_key}.predictions.json`` manifest.

    Args:
        path: Path to the manifest JSON.

    Returns:
        The validated ``PredictionManifest``.

    Raises:
        json.JSONDecodeError: If the file is not valid JSON.
        pydantic.ValidationError: If the JSON does not satisfy the manifest schema.
    """
    text = Path(path).read_text(encoding="utf-8")
    return PredictionManifest.model_validate(json.loads(text))


def resolve_artifact_paths(
    manifest: PredictionManifest, manifest_dir: Union[str, Path]
) -> Dict[str, Path]:
    """Resolve each artifact's ``slp_path`` (basename) relative to ``manifest_dir``.

    Args:
        manifest: The parsed manifest.
        manifest_dir: The directory the manifest lives in.

    Returns:
        A mapping of ``root_type`` to the resolved ``.slp`` path.

    Raises:
        FileNotFoundError: If any artifact's ``.slp`` is absent from ``manifest_dir``.
    """
    base = Path(manifest_dir)
    resolved: Dict[str, Path] = {}
    for artifact in manifest.artifacts:
        path = base / artifact.slp_path
        if not path.exists():
            raise FileNotFoundError(
                f"artifact .slp not found for root_type {artifact.root_type!r}: "
                f"{path.as_posix()}"
            )
        resolved[artifact.root_type] = path
    return resolved


def load_scan_metadata(path: Union[str, Path], expected_scan_key: str) -> ScanMetadata:
    """Load a sidecar and enforce ``scan_key`` agreement with the manifest.

    Args:
        path: Path to the ``{scan_key}.scan_metadata.json`` sidecar.
        expected_scan_key: The manifest's ``scan_key`` the sidecar must match.

    Returns:
        The validated ``ScanMetadata``.

    Raises:
        ValueError: If the sidecar's ``scan_key`` differs from ``expected_scan_key``.
    """
    text = Path(path).read_text(encoding="utf-8")
    meta = ScanMetadata.model_validate(json.loads(text))
    if meta.scan_key != expected_scan_key:
        raise ValueError(
            f"scan_key mismatch: sidecar {meta.scan_key!r} != "
            f"manifest {expected_scan_key!r}"
        )
    return meta
