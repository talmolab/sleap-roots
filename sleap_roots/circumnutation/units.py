"""Pixel ↔ millimeter conversion utility for circumnutation trait DataFrames.

The circumnutation pipeline is pure-pixel: every length-bearing trait
the pipeline emits is in pixels. Users who want millimeter output
compose :func:`convert_to_mm` on a per-plant trait DataFrame. This
function is pure (does not mutate inputs) and lives here so that the
foundation contract — "pipeline runs in px, conversion is downstream"
— is testable from PR #1 forward.

The conversion logic matches column-name suffix (``_px``,
``_px_per_frame``, ``_px_per_hr``) to the corresponding ``_mm`` form,
scaling values by ``1 / px_per_mm`` for length and ``1 / px_per_mm``
for length-per-time. Other columns (``hr``, ``rad``, dimensionless,
boolean, etc.) pass through unchanged.
"""

import logging
import math
from typing import Tuple

import pandas as pd


logger = logging.getLogger(__name__)


def convert_to_mm(
    traits_df: pd.DataFrame,
    units: dict,
    px_per_mm: float,
) -> Tuple[pd.DataFrame, dict]:
    """Convert pixel-unit columns of a per-plant trait DataFrame to mm.

    Pure function: original ``traits_df`` and ``units`` are not mutated.
    For every column whose unit string is one of ``"px"``, ``"px²"``,
    ``"px/frame"``, ``"px/hr"``, or ``"px·hr⁻¹"``, the column values are
    scaled by ``1 / px_per_mm`` (or ``(1 / px_per_mm) ** 2`` for ``px²``)
    and the column is renamed to swap its ``_px``-suffix for the
    corresponding ``_mm`` form. Columns with non-px units pass through
    unchanged.

    Args:
        traits_df: Per-plant trait DataFrame with ``_px``-suffixed columns.
        units: Mapping of column-name → unit-string (must align with
            ``traits_df.columns`` for any column intended to be converted).
        px_per_mm: Calibration factor; must be a positive finite float.

    Returns:
        Tuple of ``(mm_traits_df, mm_units)`` — copies with calibration
        applied and ``_px`` columns renamed.

    Raises:
        ValueError: If ``px_per_mm`` is not a positive finite float.
    """
    try:
        scale = float(px_per_mm)
    except (TypeError, ValueError):
        raise ValueError(
            f"px_per_mm must be a positive finite float, got {px_per_mm!r}"
        )
    if math.isnan(scale) or scale <= 0:
        raise ValueError(
            f"px_per_mm must be a positive finite float, got {px_per_mm!r}"
        )

    out_df = traits_df.copy()
    out_units = dict(units)

    rename_map: dict = {}
    new_unit_for: dict = {}

    for col, unit in units.items():
        if col not in out_df.columns:
            continue

        # Handle pixel-area first (its own suffix scheme)
        if unit == "px²":
            new_col = _rename_px_squared(col)
            out_df[col] = out_df[col] / (scale**2)
            rename_map[col] = new_col
            new_unit_for[new_col] = "mm²"
            continue

        # px·hr⁻¹ is just an alternate notation for px/hr
        if unit in ("px/hr", "px·hr⁻¹"):
            new_col = _rename_px_per_x(col, "_per_hr")
            out_df[col] = out_df[col] / scale
            rename_map[col] = new_col
            new_unit_for[new_col] = "mm/hr"
            continue

        if unit == "px/frame":
            new_col = _rename_px_per_x(col, "_per_frame")
            out_df[col] = out_df[col] / scale
            rename_map[col] = new_col
            new_unit_for[new_col] = "mm/frame"
            continue

        if unit == "px":
            # Replace the FIRST trailing _px (not earlier occurrences).
            new_col = _rename_trailing_px(col)
            out_df[col] = out_df[col] / scale
            rename_map[col] = new_col
            new_unit_for[new_col] = "mm"
            continue

        # Non-px columns pass through; nothing to do.

    # Apply renames in one go
    out_df = out_df.rename(columns=rename_map)

    # Build the new units dict: drop renamed olds, add renamed news
    for old_col in rename_map:
        out_units.pop(old_col, None)
    out_units.update(new_unit_for)

    return out_df, out_units


def _rename_trailing_px(col: str) -> str:
    """Replace a trailing ``_px`` with ``_mm``, or append ``_mm`` if absent."""
    if col.endswith("_px"):
        return col[: -len("_px")] + "_mm"
    return col + "_mm"


def _rename_px_per_x(col: str, suffix: str) -> str:
    """Replace ``_px<suffix>`` with ``_mm<suffix>``; e.g. ``v_long_px_per_hr`` → ``v_long_mm_per_hr``."""
    target = "_px" + suffix
    replacement = "_mm" + suffix
    if col.endswith(target):
        return col[: -len(target)] + replacement
    return col + replacement


def _rename_px_squared(col: str) -> str:
    """Replace a trailing ``_px²`` / ``_px2`` with ``_mm²`` / ``_mm2`` accordingly."""
    if col.endswith("_px²"):
        return col[: -len("_px²")] + "_mm²"
    if col.endswith("_px2"):
        return col[: -len("_px2")] + "_mm2"
    if col.endswith("_px"):
        return col[: -len("_px")] + "_mm²"
    return col + "_mm²"
