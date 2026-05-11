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


REQUIRED_PER_FRAME_COLUMNS: tuple = ("frame", "tip_x", "tip_y")
"""Per-frame columns required on every trajectory_df in addition to ROW_IDENTITY_COLUMNS."""


def _validate_trajectory_df(instance, attribute, value: pd.DataFrame) -> None:
    """Validate that ``trajectory_df`` is non-empty and carries every required column."""
    if not isinstance(value, pd.DataFrame):
        raise ValueError(
            f"trajectory_df must be a pandas DataFrame, got {type(value).__name__}"
        )
    if len(value) == 0:
        raise ValueError("trajectory_df is empty (zero rows); cannot analyze")
    missing_identity = [col for col in ROW_IDENTITY_COLUMNS if col not in value.columns]
    if missing_identity:
        raise ValueError(
            f"trajectory_df is missing required row-identity column(s): "
            f"{', '.join(missing_identity)}"
        )
    missing_per_frame = [
        col for col in REQUIRED_PER_FRAME_COLUMNS if col not in value.columns
    ]
    if missing_per_frame:
        raise ValueError(
            f"trajectory_df is missing required per-frame column(s): "
            f"{', '.join(missing_per_frame)}"
        )


def _coerce_cadence_s(value):
    """Coerce ``cadence_s`` to ``float``; raise ``ValueError`` naming the field on failure.

    Used as an ``attrs`` converter so that string-convertible numeric
    inputs (e.g. ``cadence_s="300"``) are stored as ``float`` and so
    that non-numeric inputs raise a ``ValueError`` whose message names
    ``cadence_s``. Booleans are rejected explicitly (Python ``bool``
    subclasses ``int`` and would otherwise pass as ``1.0``). The
    validator runs on the post-coercion value and catches NaN / ±inf
    / non-positive.
    """
    if value is None:
        return value  # validator rejects None with a field-naming message
    if isinstance(value, bool):
        raise ValueError(f"cadence_s must be a positive finite float, got {value!r}")
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"cadence_s must be a positive finite float, got {value!r}")


def _coerce_R_px(value):
    """Coerce ``R_px`` to ``float`` (or ``None``); raise ``ValueError`` naming the field on failure.

    Booleans are rejected explicitly (Python ``bool`` subclasses ``int``
    and would otherwise pass as ``1.0``).
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"R_px must be None or a positive finite float, got {value!r}")
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"R_px must be None or a positive finite float, got {value!r}")


def _validate_cadence_s(instance, attribute, value) -> None:
    """Validate that ``cadence_s`` (post-coercion) is a positive finite float.

    ``math.isfinite`` rejects NaN AND ±inf in one check (the prior
    ``math.isnan(value) or value <= 0`` form silently accepted ``+inf``).
    """
    if value is None or not math.isfinite(value) or value <= 0:
        raise ValueError(f"cadence_s must be a positive finite float, got {value!r}")


def _validate_R_px(instance, attribute, value: Optional[float]) -> None:
    """Validate that ``R_px`` (post-coercion) is None or a positive finite float."""
    if value is None:
        return
    if not math.isfinite(value) or value <= 0:
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
    cadence_s: float = attrs.field(
        converter=_coerce_cadence_s, validator=_validate_cadence_s
    )
    R_px: Optional[float] = attrs.field(
        default=None, converter=_coerce_R_px, validator=_validate_R_px
    )
    run_id: Optional[str] = attrs.field(default=None)
