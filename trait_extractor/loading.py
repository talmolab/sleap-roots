"""Build a ``sleap_roots.Series`` from a consumed prediction manifest."""

from pathlib import Path
from typing import Union

from sleap_roots import Series

from trait_extractor.manifest import PredictionManifest, resolve_artifact_paths

# Maps a manifest artifact ``root_type`` to the ``Series.load`` keyword argument.
_ROOT_TYPE_TO_KWARG = {
    "primary": "primary_path",
    "lateral": "lateral_path",
    "crown": "crown_path",
}


def load_series(manifest: PredictionManifest, manifest_dir: Union[str, Path]) -> Series:
    """Load a ``Series`` from a manifest's per-root artifacts.

    ``Series.load`` tolerates missing paths silently (it prints and leaves labels
    ``None`` rather than raising), so an empty ``artifacts`` list is rejected here with
    an explicit guard **before** ``Series.load`` is called — otherwise an envelope with
    no traits would be emitted for a scan predict resolved zero roots for.

    Args:
        manifest: The parsed prediction manifest.
        manifest_dir: The directory the manifest (and its ``.slp``) live in.

    Returns:
        A ``Series`` named by ``manifest.scan_key`` with the artifacts' labels loaded.

    Raises:
        ValueError: If the manifest has no artifacts.
        FileNotFoundError: If a referenced ``.slp`` is missing (via path resolution).
    """
    if not manifest.artifacts:
        raise ValueError(
            f"manifest for {manifest.scan_key!r} has no artifacts "
            "(predict resolved zero roots); refusing to load an empty Series"
        )
    paths = resolve_artifact_paths(manifest, manifest_dir)
    kwargs = {
        _ROOT_TYPE_TO_KWARG[root_type]: path.as_posix()
        for root_type, path in paths.items()
    }
    return Series.load(series_name=manifest.scan_key, **kwargs)
