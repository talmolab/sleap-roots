"""I/O helpers for the circumnutation pipeline's per-plant trait CSVs.

This module owns the foundation contracts:

- :func:`build_per_plant_template` — derive a per-plant trait
  DataFrame from a :class:`~sleap_roots.circumnutation._types.CircumnutationInputs`,
  populated with the eight row-identity columns and sorted numerically.
- :func:`default_units_for_template` — return the unit-string mapping
  for the row-identity columns (no trait columns yet).
- :func:`write_per_plant_csv` and :func:`read_per_plant_csv` —
  round-trippable CSV + sidecar writer/reader.
- :func:`write_units_sidecar` / :func:`read_units_sidecar` —
  UTF-8 JSON helpers.
- :func:`write_run_metadata` / :func:`read_run_metadata` —
  same, for the provenance sidecar.
- :func:`gather_run_metadata` — assemble the provenance dict with
  git SHA, package versions, schema and constants versions, and a
  snapshot of every constant in
  :mod:`sleap_roots.circumnutation._constants`.
"""

import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple, Union

import pandas as pd

from sleap_roots.circumnutation import _constants
from sleap_roots.circumnutation._constants import (
    _CONSTANTS_VERSION,
    _SCHEMA_VERSION,
    ConstantsT,
)
from sleap_roots.circumnutation._types import (
    ROW_IDENTITY_COLUMNS,
    CircumnutationInputs,
)


logger = logging.getLogger(__name__)


PathLike = Union[str, os.PathLike]


# ---------------------------------------------------------------------------
# Per-plant template builder
# ---------------------------------------------------------------------------


def build_per_plant_template(inputs: CircumnutationInputs) -> pd.DataFrame:
    """Derive a per-plant trait DataFrame template from per-frame trajectory data.

    Returns one row per unique track in ``inputs.trajectory_df``. The
    eight row-identity columns are populated; no trait columns are
    added (tier modules do that). Rows are sorted via
    ``pandas.DataFrame.sort_values`` over
    ``(series, sample_uid, plate_id, plant_id, track_id)`` — string
    columns sort lexicographically and integer columns sort numerically.

    Args:
        inputs: Validated :class:`CircumnutationInputs`.

    Returns:
        A per-plant DataFrame with the eight :data:`ROW_IDENTITY_COLUMNS`
        columns (in declared order) and one row per unique track.
        ``track_id`` carries integer dtype; ``plant_id`` is column-wise
        equal to ``track_id``.
    """
    df = inputs.trajectory_df

    # Drop duplicates on the row-identity columns; keep first occurrence.
    template = df[list(ROW_IDENTITY_COLUMNS)].drop_duplicates().reset_index(drop=True)

    # Coerce track_id and plant_id to integer dtype where possible.
    for col in ("track_id", "plant_id"):
        template[col] = template[col].astype("int64")

    # Numeric / lexicographic sort across the 5-key tuple.
    template = template.sort_values(
        by=["series", "sample_uid", "plate_id", "plant_id", "track_id"],
        kind="stable",
    ).reset_index(drop=True)

    # Ensure the eight columns are at the front, in declared order.
    template = template[list(ROW_IDENTITY_COLUMNS)]

    return template


# ---------------------------------------------------------------------------
# Default units mapping for the row-identity columns
# ---------------------------------------------------------------------------


_ROW_IDENTITY_UNITS: dict = {
    "series": "string",
    "sample_uid": "string",
    "timepoint": "string",
    "plate_id": "string",
    "plant_id": "int",
    "track_id": "int",
    "genotype": "string",
    "treatment": "string",
}


def default_units_for_template(template_df: pd.DataFrame) -> dict:
    """Return the per-column units dict for a per-plant template DataFrame.

    Today only the eight row-identity columns are present; tier PRs
    extend this mapping with their trait-specific unit strings.

    Args:
        template_df: Output of :func:`build_per_plant_template`.

    Returns:
        Mapping of column-name → unit-string in
        :data:`sleap_roots.circumnutation._constants.VALID_UNIT_VOCABULARY`.
    """
    return {col: _ROW_IDENTITY_UNITS[col] for col in template_df.columns}


# ---------------------------------------------------------------------------
# Sidecar JSON writers / readers
# ---------------------------------------------------------------------------


def write_units_sidecar(out_path: PathLike, units: dict) -> None:
    """Write the units mapping as UTF-8 JSON.

    Args:
        out_path: Destination path; parent directories must exist.
        units: Column-name → unit-string mapping.
    """
    Path(out_path).write_text(
        json.dumps(units, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def read_units_sidecar(in_path: PathLike) -> dict:
    """Load a units sidecar JSON (UTF-8) and return its mapping."""
    return json.loads(Path(in_path).read_text(encoding="utf-8"))


def write_run_metadata(out_path: PathLike, metadata: dict) -> None:
    """Write a run-metadata mapping as UTF-8 JSON.

    Args:
        out_path: Destination path; parent directories must exist.
        metadata: Mapping returned by :func:`gather_run_metadata`.
    """
    Path(out_path).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def read_run_metadata(in_path: PathLike) -> dict:
    """Load a run-metadata sidecar JSON (UTF-8) and return its mapping."""
    return json.loads(Path(in_path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Per-plant CSV + sidecar bundle
# ---------------------------------------------------------------------------


def write_per_plant_csv(
    out_path: PathLike,
    df: pd.DataFrame,
    units: dict,
    run_metadata: dict,
) -> None:
    """Write a per-plant trait CSV with sibling units and run-metadata sidecars.

    The CSV is written via ``DataFrame.to_csv`` with no index. The
    sibling sidecars are placed in the same directory as the CSV:

    - ``<csv_stem>.units.json`` — column → unit-string
    - ``run_metadata.json`` — provenance bundle

    Args:
        out_path: CSV path; parent directory must exist.
        df: Per-plant DataFrame (e.g., output of
            :func:`build_per_plant_template`, possibly augmented with
            tier-emitted trait columns).
        units: Column-name → unit-string mapping (every column in
            ``df`` should be present).
        run_metadata: Mapping returned by :func:`gather_run_metadata`.
    """
    csv_path = Path(out_path)
    df.to_csv(csv_path.as_posix(), index=False, encoding="utf-8")

    units_path = csv_path.with_suffix("").with_suffix(".units.json")
    write_units_sidecar(units_path, units)

    metadata_path = csv_path.parent / "run_metadata.json"
    write_run_metadata(metadata_path, run_metadata)


def read_per_plant_csv(
    in_path: PathLike,
) -> Tuple[pd.DataFrame, dict, dict]:
    """Read a per-plant CSV and its sibling sidecars.

    Args:
        in_path: CSV path.

    Returns:
        ``(df, units, run_metadata)`` triple. ``df`` is the loaded
        DataFrame; ``units`` is the loaded units sidecar; ``run_metadata``
        is the loaded run-metadata sidecar (or ``{}`` if missing).
    """
    csv_path = Path(in_path)
    df = pd.read_csv(csv_path.as_posix())

    units_path = csv_path.with_suffix("").with_suffix(".units.json")
    units = read_units_sidecar(units_path) if units_path.exists() else {}

    metadata_path = csv_path.parent / "run_metadata.json"
    metadata = read_run_metadata(metadata_path) if metadata_path.exists() else {}

    return df, units, metadata


# ---------------------------------------------------------------------------
# Run-metadata gatherer
# ---------------------------------------------------------------------------


def gather_run_metadata(
    input_path: PathLike,
    run_id: Optional[str] = None,
    constants: Optional[ConstantsT] = None,
) -> dict:
    """Assemble the provenance bundle for one pipeline run.

    Args:
        input_path: Source data path (.slp, .csv, etc.). Stored as a
            string for portability across machines.
        run_id: Optional human-readable identifier.
        constants: Optional :class:`ConstantsT` override. When ``None``,
            the snapshot reflects the module-level defaults.

    Returns:
        A JSON-serializable dict containing every required provenance
        field per the foundation spec.
    """
    snapshot = _build_constants_snapshot(constants)

    return {
        "input_path": str(input_path),
        "sleap_roots_git_sha": _get_git_sha(),
        "sleap_roots_version": _get_sleap_roots_version(),
        "sleap_io_version": _get_sleap_io_version(),
        "python_version": sys.version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "_schema_version": _SCHEMA_VERSION,
        "_constants_version": _CONSTANTS_VERSION,
        "_constants_snapshot": snapshot,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_constants_snapshot(constants: Optional[ConstantsT]) -> dict:
    """Return a JSON-serializable snapshot of either the override-bag or the defaults."""
    if constants is None:
        return _constants._default_constants_snapshot()
    # attrs.frozen instance — extract every field by name.
    snapshot = {}
    for attr in constants.__attrs_attrs__:
        snapshot[attr.name] = getattr(constants, attr.name)
    return snapshot


def _get_git_sha() -> str:
    """Return the current sleap-roots git SHA, or ``"unknown"`` if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            cwd=str(Path(__file__).resolve().parent),
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, OSError):
        pass
    return "unknown"


def _get_sleap_roots_version() -> str:
    """Return ``sleap_roots.__version__`` or ``"unknown"``."""
    try:
        import sleap_roots

        return getattr(sleap_roots, "__version__", "unknown")
    except Exception:  # noqa: BLE001 — provenance must never crash a run
        return "unknown"


def _get_sleap_io_version() -> str:
    """Return ``sleap_io.__version__`` or ``"unknown"``."""
    try:
        import sleap_io as sio

        return getattr(sio, "__version__", "unknown")
    except Exception:  # noqa: BLE001
        return "unknown"
