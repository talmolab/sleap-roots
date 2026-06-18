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

import importlib
import json
import logging
import os
import platform
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
    PIPELINE_UNIT_VOCABULARY,
    ROW_IDENTITY_UNITS,
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


_IDENTITY_5_TUPLE: tuple = ("series", "sample_uid", "plate_id", "plant_id", "track_id")
"""The 5-tuple that uniquely identifies a physical plant track (used as the duplicate-detection key)."""

_CONFLICT_CHECK_COLUMNS: tuple = ("timepoint", "genotype", "treatment")
"""Columns expected to be constant per :data:`_IDENTITY_5_TUPLE`; conflicts indicate upstream join error."""


def build_per_plant_template(inputs: CircumnutationInputs) -> pd.DataFrame:
    """Derive a per-plant trait DataFrame template from per-frame trajectory data.

    Returns one row per unique 5-tuple ``(series, sample_uid, plate_id,
    plant_id, track_id)`` in ``inputs.trajectory_df``. The eight
    row-identity columns are populated; no trait columns are added
    (tier modules do that). Rows are sorted via
    ``pandas.DataFrame.sort_values`` over the 5-tuple — string columns
    sort lexicographically and integer columns sort numerically.

    This is a thin wrapper over :func:`_build_per_plant_template_from_df`;
    tier modules whose canonical signature takes a raw ``trajectory_df``
    (e.g., ``kinematics.compute`` per PR #2) should import that private
    helper directly to avoid wrapping the DataFrame in an unused
    :class:`CircumnutationInputs` instance.

    Args:
        inputs: Validated :class:`CircumnutationInputs`.

    Returns:
        A per-plant DataFrame with the eight :data:`ROW_IDENTITY_COLUMNS`
        columns (in declared order) and one row per unique track.
        ``track_id`` carries integer dtype; ``plant_id`` is column-wise
        equal to ``track_id``.

    Raises:
        ValueError: If ``track_id`` contains ``NaN`` (cannot be coerced
            to integer); or if the same 5-tuple plant has conflicting
            values in ``timepoint`` / ``genotype`` / ``treatment``
            across frames (sign of upstream join error).
    """
    return _build_per_plant_template_from_df(inputs.trajectory_df)


def _build_per_plant_template_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """Apply drop-duplicates + sort + dtype-coerce on a raw trajectory DataFrame.

    Tier modules whose canonical signature takes a raw ``trajectory_df``
    (``kinematics.compute`` today; ``qc.compute``, ``parametric.compute``
    in later PRs) import this helper directly rather than wrapping their
    DataFrame in a :class:`CircumnutationInputs` purely to satisfy the
    public API. The public :func:`build_per_plant_template` wraps this
    helper for the foundation-style caller path.

    Behaviorally identical to the public function: same validations,
    same error messages, same column order, same dtype coercions.

    Args:
        df: Raw trajectory DataFrame containing at minimum the eight
            row-identity columns. Per-frame columns (``frame``, ``tip_x``,
            ``tip_y``) need not be present for this helper, since it only
            inspects the row-identity columns.

    Returns:
        Per-plant DataFrame as documented on :func:`build_per_plant_template`.

    Raises:
        ValueError: Same conditions as the public wrapper.
    """
    # B3: clear ValueError for NaN track_id rather than pandas' IntCastingNaNError.
    if df["track_id"].isna().any():
        bad_rows = df[df["track_id"].isna()].index.tolist()[:5]
        raise ValueError(
            f"trajectory_df has NaN in track_id (rows {bad_rows}); "
            f"track_id must be non-NaN integer-coercible"
        )

    # B3 + F1: detect conflicting per-frame metadata (different values for
    # the same 5-tuple) BEFORE drop_duplicates collapses them. A future tier
    # PR merging trait DataFrames onto the template would silently duplicate
    # rows; raise here instead.
    #
    # Use dropna=False on BOTH groupby and nunique so NaN-vs-concrete
    # mismatches are flagged. The prior dropna=True (nunique) form silently
    # accepted a plant with genotype=NaN on some frames and genotype="WT"
    # on others (nunique saw only one non-NaN value), leaving drop_duplicates
    # to emit two rows for the same 5-tuple plant.
    for conflict_col in _CONFLICT_CHECK_COLUMNS:
        grouped = df.groupby(list(_IDENTITY_5_TUPLE), dropna=False)[
            conflict_col
        ].nunique(dropna=False)
        offenders = grouped[grouped > 1]
        if not offenders.empty:
            first_offender = offenders.index[0]
            raise ValueError(
                f"build_per_plant_template: conflicting {conflict_col!r} values "
                f"for plant {dict(zip(_IDENTITY_5_TUPLE, first_offender))} — "
                f"every frame of a single plant must share the same "
                f"{conflict_col!r} (including all-NaN vs concrete-value "
                f"mismatch). Fix upstream metadata join before constructing "
                f"CircumnutationInputs."
            )

    # Drop duplicates on the row-identity columns; keep first occurrence.
    template = df[list(ROW_IDENTITY_COLUMNS)].drop_duplicates().reset_index(drop=True)

    # Coerce track_id and plant_id to integer dtype where possible.
    for col in ("track_id", "plant_id"):
        template[col] = template[col].astype("int64")

    # I8: enforce object dtype for the string identity columns. An all-NaN
    # genotype/treatment column gets inferred as float64 otherwise, which
    # violates the spec requirement that these columns hold strings + NaN.
    for col in (
        "series",
        "sample_uid",
        "timepoint",
        "plate_id",
        "genotype",
        "treatment",
    ):
        template[col] = template[col].astype(object)

    # Numeric / lexicographic sort across the 5-key tuple.
    template = template.sort_values(
        by=list(_IDENTITY_5_TUPLE),
        kind="stable",
    ).reset_index(drop=True)

    # Ensure the eight columns are at the front, in declared order.
    template = template[list(ROW_IDENTITY_COLUMNS)]

    return template


# ---------------------------------------------------------------------------
# Default units mapping for the row-identity columns
# ---------------------------------------------------------------------------


def default_units_for_template(template_df: pd.DataFrame) -> dict:
    """Return the per-column units dict for a per-plant template DataFrame.

    Today only the eight row-identity columns are present; tier PRs
    extend this mapping with their trait-specific unit strings.

    Args:
        template_df: Output of :func:`build_per_plant_template`.

    Returns:
        Mapping of column-name → unit-string in
        :data:`sleap_roots.circumnutation._constants.PIPELINE_UNIT_VOCABULARY`.
        Sourced from
        :data:`sleap_roots.circumnutation._constants.ROW_IDENTITY_UNITS`.
    """
    return {col: ROW_IDENTITY_UNITS[col] for col in template_df.columns}


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


def _validate_units_coverage(df: pd.DataFrame, units: dict, *, fn_name: str) -> None:
    """Raise ``ValueError`` unless ``units`` keys 1:1 cover ``df`` columns.

    Shared by :func:`write_per_plant_csv`, :func:`write_per_genotype_csv`, and
    :func:`sleap_roots.circumnutation.aggregation.aggregate_by_genotype`'s input
    validation. Coverage only — the writers run the separate
    :data:`PIPELINE_UNIT_VOCABULARY` membership check inline (the aggregation
    input does not need it; per-plant units are pipeline output already in the
    vocabulary).

    Args:
        df: The DataFrame whose columns must each have a unit.
        units: Column-name → unit-string mapping.
        fn_name: Caller name, prefixed onto the error message.

    Raises:
        ValueError: If any column lacks a unit, or ``units`` has an extra key,
            naming the offending column(s). Raised before any side effect.
    """
    df_cols = set(df.columns)
    units_keys = set(units.keys())
    missing_in_units = df_cols - units_keys
    extra_in_units = units_keys - df_cols
    if missing_in_units or extra_in_units:
        parts = []
        if missing_in_units:
            parts.append(
                f"DataFrame columns missing from units dict: "
                f"{sorted(missing_in_units)}"
            )
        if extra_in_units:
            parts.append(
                f"units keys not present in DataFrame: " f"{sorted(extra_in_units)}"
            )
        raise ValueError(
            f"{fn_name}: units dict does not 1:1 cover DataFrame "
            "columns. " + "; ".join(parts) + ". Fix the units mapping before "
            "writing the sidecar."
        )


def _validate_units_vocabulary(units: dict, *, fn_name: str) -> None:
    """Raise ``ValueError`` unless every unit string is in the closed vocabulary.

    The split vocabularies are the structural defense of the pure-pixel
    contract; without writer enforcement that defense is documentation-only.
    Fails loud and atomic — no files written when validation fails.

    Args:
        units: Column-name → unit-string mapping.
        fn_name: Caller name, prefixed onto the error message.

    Raises:
        ValueError: If any unit string is outside
            :data:`PIPELINE_UNIT_VOCABULARY`, naming the offending pairs.
    """
    invalid = [
        (col, u) for col, u in units.items() if u not in PIPELINE_UNIT_VOCABULARY
    ]
    if invalid:
        offending = ", ".join(f"{col!r}: {u!r}" for col, u in invalid)
        raise ValueError(
            f"{fn_name}: units dict contains values outside "
            f"PIPELINE_UNIT_VOCABULARY (the closed sidecar vocabulary). "
            f"Offending column/unit pairs: {offending}. The pure-pixel "
            f"contract requires sidecar values to use only px-based or "
            f"calibration-independent units; mm-based values must come "
            f"from convert_to_mm() output, not pipeline emission."
        )


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
    # F2/B4: enforce 1:1 units coverage + closed-vocabulary membership BEFORE
    # writing any files. Coverage is the shared helper (also used by
    # write_per_genotype_csv and aggregate_by_genotype); vocabulary membership
    # stays here. Fail loud and atomic — no CSV / sidecar files written on error.
    _validate_units_coverage(df, units, fn_name="write_per_plant_csv")
    _validate_units_vocabulary(units, fn_name="write_per_plant_csv")

    csv_path = Path(out_path)
    df.to_csv(csv_path.as_posix(), index=False, encoding="utf-8")

    # Only strip the final `.csv` extension — never intermediate dots.
    # `csv_path.stem` preserves dotted CSV names like `traits.per.plant.csv`.
    units_path = csv_path.parent / f"{csv_path.stem}.units.json"
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

    # Mirror the writer: strip only the final `.csv` extension.
    units_path = csv_path.parent / f"{csv_path.stem}.units.json"
    units = read_units_sidecar(units_path) if units_path.exists() else {}

    metadata_path = csv_path.parent / "run_metadata.json"
    metadata = read_run_metadata(metadata_path) if metadata_path.exists() else {}

    return df, units, metadata


# ---------------------------------------------------------------------------
# Per-genotype CSV + sidecar bundle
# ---------------------------------------------------------------------------


def write_per_genotype_csv(
    out_path: PathLike,
    df: pd.DataFrame,
    units: dict,
    run_metadata: dict,
) -> None:
    """Write a per-genotype aggregation CSV with sibling units/run-metadata sidecars.

    Mirrors :func:`write_per_plant_csv` for the per-genotype output of
    :func:`sleap_roots.circumnutation.aggregation.aggregate_by_genotype`. The
    CSV is written via ``DataFrame.to_csv`` with no index; siblings are placed
    in the same directory:

    - ``<csv_stem>.units.json`` — column → unit-string
    - ``run_metadata.json`` — provenance bundle

    .. note::
        ``run_metadata.json`` has a **fixed name** in the output directory. Writing
        a per-genotype CSV alongside a per-plant CSV (or a second per-genotype CSV)
        in the same directory overwrites the earlier ``run_metadata.json``. Write at
        most one CSV artifact per output directory, mirroring
        :meth:`CircumnutationPipeline.save`'s constraint. A stem-prefixed metadata
        filename to remove this fixed-name clobber is tracked as a follow-up issue
        ("stem-prefixed run_metadata.json name to remove the _io fixed-name
        clobber").

    Args:
        out_path: CSV path; parent directory must exist.
        df: Per-genotype DataFrame (output of ``aggregate_by_genotype``).
        units: Column-name → unit-string mapping (every column present, 1:1).
        run_metadata: Mapping returned by :func:`gather_run_metadata`.

    Raises:
        ValueError: If ``units`` does not 1:1 cover the columns, or contains a
            unit outside :data:`PIPELINE_UNIT_VOCABULARY`. Raised before any
            file is written.
    """
    _validate_units_coverage(df, units, fn_name="write_per_genotype_csv")
    _validate_units_vocabulary(units, fn_name="write_per_genotype_csv")

    csv_path = Path(out_path)
    df.to_csv(csv_path.as_posix(), index=False, encoding="utf-8")

    # Only strip the final `.csv` extension — never intermediate dots.
    units_path = csv_path.parent / f"{csv_path.stem}.units.json"
    write_units_sidecar(units_path, units)

    metadata_path = csv_path.parent / "run_metadata.json"
    write_run_metadata(metadata_path, run_metadata)


def read_per_genotype_csv(
    in_path: PathLike,
) -> Tuple[pd.DataFrame, dict, dict]:
    """Read a per-genotype CSV and its sibling sidecars.

    Args:
        in_path: CSV path.

    Returns:
        ``(df, units, run_metadata)`` triple. ``df`` is the loaded DataFrame;
        ``units`` is the loaded units sidecar (or ``{}`` if missing);
        ``run_metadata`` is the loaded run-metadata sidecar (or ``{}`` if
        missing).
    """
    csv_path = Path(in_path)
    df = pd.read_csv(csv_path.as_posix())

    units_path = csv_path.parent / f"{csv_path.stem}.units.json"
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
    cadence_s: Optional[float] = None,
    R_px: Optional[float] = None,
) -> dict:
    """Assemble the provenance bundle for one pipeline run.

    Args:
        input_path: Source data path (.slp, .csv, etc.). Stored as a
            string for portability across machines.
        run_id: Optional human-readable identifier.
        constants: Optional :class:`ConstantsT` override. When ``None``,
            the snapshot reflects the module-level defaults.
        cadence_s: Optional per-run frame cadence in seconds (added in PR #14).
            ``cadence_s`` determines every period trait and the traveling-wave
            residual, and is otherwise unrecoverable from the ``.slp`` at
            ``input_path``; recording it makes a run reproducible from the
            sidecars alone. ``None`` (non-pipeline callers) writes ``null``.
        R_px: Optional root cross-section radius in pixels (added in PR #14;
            ``None`` writes ``null``).

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
        "numpy_version": _get_package_version("numpy"),
        "scipy_version": _get_package_version("scipy"),
        "pandas_version": _get_package_version("pandas"),
        "python_version": sys.version,
        "platform": platform.platform(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "cadence_s": cadence_s,
        "R_px": R_px,
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


def _get_package_version(package_name: str) -> str:
    """Return ``<package>.__version__`` or ``"unknown"`` if not importable."""
    try:
        mod = importlib.import_module(package_name)
        return getattr(mod, "__version__", "unknown")
    except Exception:  # noqa: BLE001 — provenance must never crash a run
        return "unknown"
