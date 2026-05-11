"""Foundational data types for the circumnutation pipeline.

Defines :data:`ROW_IDENTITY_COLUMNS` (the eight-column trait-CSV
row-identity schema, see roadmap.md CC-4) and :class:`CircumnutationInputs`
(the per-call container for trajectory data, cadence, root cross-section
radius, and run id). The pipeline's contract for calibration is
pure-pixel: there is no ``px_per_mm`` field on this class — calibration
is a downstream concern handled by
:func:`sleap_roots.circumnutation.units.convert_to_mm`.
"""

import logging
import math
from typing import Optional

import attrs
import pandas as pd


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Row-identity schema (roadmap.md CC-4)
# ---------------------------------------------------------------------------

ROW_IDENTITY_COLUMNS: tuple = (
    "series",
    "sample_uid",
    "timepoint",
    "plate_id",
    "plant_id",
    "track_id",
    "genotype",
    "treatment",
)
"""Eight-column row-identity schema for the per-plant trait CSV.

Today ``plant_id`` and ``track_id`` are populated identically; both
columns exist so a future divergence (multi-track-per-plant) is a
non-breaking schema extension. ``plate_id`` and ``treatment`` are
aspirational — no upstream produces them today; they populate as
``NaN`` and are reserved for future upstream metadata work.
"""


# ---------------------------------------------------------------------------
# CircumnutationInputs data class
# ---------------------------------------------------------------------------


def _validate_trajectory_df(instance, attribute, value: pd.DataFrame) -> None:
    """Validate that ``trajectory_df`` is non-empty and carries every row-identity column."""
    if not isinstance(value, pd.DataFrame):
        raise ValueError(
            f"trajectory_df must be a pandas DataFrame, got {type(value).__name__}"
        )
    if len(value) == 0:
        raise ValueError("trajectory_df is empty (zero rows); cannot analyze")
    missing = [col for col in ROW_IDENTITY_COLUMNS if col not in value.columns]
    if missing:
        raise ValueError(
            f"trajectory_df is missing required row-identity column(s): "
            f"{', '.join(missing)}"
        )


def _validate_cadence_s(instance, attribute, value: float) -> None:
    """Validate that ``cadence_s`` is a positive finite float.

    Mirrors the try/except pattern in :func:`_validate_R_px` so that
    non-numeric inputs (e.g. strings) surface as ``ValueError`` naming
    the ``cadence_s`` field, matching the class docstring contract.
    """
    if value is None:
        raise ValueError(f"cadence_s must be a positive finite float, got {value!r}")
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"cadence_s must be a positive finite float, got {value!r}")
    if math.isnan(as_float) or as_float <= 0:
        raise ValueError(f"cadence_s must be a positive finite float, got {value!r}")


def _validate_R_px(instance, attribute, value: Optional[float]) -> None:
    """Validate that ``R_px`` is None or a positive finite float."""
    if value is None:
        return
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"R_px must be None or a positive finite float, got {value!r}")
    if math.isnan(as_float) or as_float <= 0:
        raise ValueError(f"R_px must be None or a positive finite float, got {value!r}")


@attrs.define(slots=False, kw_only=True)
class CircumnutationInputs:
    """Per-call inputs to the circumnutation pipeline.

    The pipeline never sees a calibration value (``px_per_mm``) — every
    length-bearing trait the pipeline emits is in pixels. The
    :func:`sleap_roots.circumnutation.units.convert_to_mm` utility
    handles downstream conversion when a user wants mm output.

    Args:
        trajectory_df: Per-track tip-trajectory DataFrame containing the
            eight :data:`ROW_IDENTITY_COLUMNS` plus per-frame ``frame``,
            ``tip_x``, ``tip_y`` columns. Must be non-empty.
        cadence_s: Frame cadence in seconds. Must be positive and finite.
        R_px: Optional root cross-section radius in pixels (Bastien-Meroz
            2016 use). Must be ``None`` or a positive finite float. The
            user converts their physical-unit radius measurement to
            pixels using whatever calibration they trust; the pipeline's
            Tier 4 formulas cancel dimensions.
        run_id: Optional human-readable identifier for the run; included
            in ``run_metadata.json``.

    Raises:
        ValueError: If any field fails validation. The message names
            the offending field.
    """

    trajectory_df: pd.DataFrame = attrs.field(validator=_validate_trajectory_df)
    cadence_s: float = attrs.field(validator=_validate_cadence_s)
    R_px: Optional[float] = attrs.field(default=None, validator=_validate_R_px)
    run_id: Optional[str] = attrs.field(default=None)
