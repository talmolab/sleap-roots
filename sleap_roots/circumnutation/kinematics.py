"""Tier 0 — raw tip kinematics (per ``docs/circumnutation/theory.md`` §7.1).

Computes per-track scalar kinematic traits directly from tip trajectories,
with no spectral analysis and no midline reconstruction. Emits 9 trait
columns plus the ``growth_axis_unreliable`` boolean flag onto the per-plant
DataFrame produced by :func:`sleap_roots.circumnutation._io._build_per_plant_template_from_df`.

Coordinate convention
---------------------
This module uses image-space coordinates where y increases downward
(theory.md §2.1). As a consequence, ``principal_axis_angle`` for a root
growing image-down (positive y direction) reads as ``+π/2`` because the
standard math ``atan2(y, x)`` returns ``+π/2`` for ``(y > 0, x = 0)``.

Sign conventions
----------------
``v_lat_signed_median_px_per_frame > 0`` indicates motion in the direction
``û_lat = (−û_g[1], û_g[0])`` — a 90° rotation of the growth-axis unit
vector in standard math (counter-clockwise) axes. Under the image-y-down
convention, this corresponds to a 90° clockwise rotation as viewed on
screen. ``v_long_signed_median_px_per_frame`` is positive when the per-frame
velocity projects onto the +growth-axis direction (i.e., the plant is
growing rather than retracting).

Handedness
----------
``angular_amplitude`` is the peak-to-peak extent of the unwrapped velocity-
direction time series ``ψ_g(t)`` and is INVARIANT under the ``atan2``
argument order (max − min is preserved under offset and sign-flip). But
the underlying ``ψ_g(t)`` time series itself uses ``atan2(dx, dy)`` per
Bastien-Meroz 2016 Eq. 20 / theory.md §3.5; this convention is enforced
by :func:`sleap_roots.circumnutation._geometry.compute_psi_g`. PR #7's
``handedness`` trait (sign of ``mean dψ_g/dt``) depends on this
convention; the reversed order ``atan2(dy, dx)`` would silently invert
the handedness sign.

Pure-pixel + cadence-independent emission
-----------------------------------------
All velocity columns are emitted in ``px/frame`` (gap-aware diff:
``Δxy / Δframe``). The module does NOT accept ``cadence_s`` and does
NOT emit ``px/hr`` or ``mm`` columns. Downstream users compose
:func:`sleap_roots.circumnutation.units.convert_to_mm` (and a future
``convert_to_per_hour`` utility) for human-readable units. See roadmap.md
CC-3 for the pure-pixel pipeline convention rationale.

Growth-axis reliability gate
----------------------------
``growth_axis_unreliable = (D < K * sg_residual_xy_local)`` where ``D`` is
the per-track net displacement, ``K = constants.GROWTH_AXIS_RELIABILITY_K``
(default 10), and ``sg_residual_xy_local`` is computed via
:func:`sleap_roots.circumnutation._noise.compute_sg_residual_xy`. When
True, the six rotation-dependent traits (``v_long_signed_median_px_per_frame``,
``v_long_abs_median_px_per_frame``, ``v_lat_signed_median_px_per_frame``,
``v_lat_abs_median_px_per_frame``, ``long_lat_ratio``,
``principal_axis_angle``) are NaN'd. The three rotation-invariant traits
(``v_total_median_px_per_frame``, ``path_displacement_ratio``,
``angular_amplitude``) survive the gate.

Tier 0 owns the ``growth_axis_unreliable`` column emission. PR #3 QC
composes with the flag (e.g., as a clause in ``track_is_clean``) but does
NOT re-emit a duplicate column.
"""

import logging
import math
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from sleap_roots.circumnutation import _geometry, _noise
from sleap_roots.circumnutation._constants import ConstantsT
from sleap_roots.circumnutation._io import (
    _IDENTITY_5_TUPLE,
    _build_per_plant_template_from_df,
)
from sleap_roots.circumnutation._types import (
    ROW_IDENTITY_COLUMNS,
    _validate_trajectory_df,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tier 0 output schema (locked by spec scenario "Output DataFrame columns
# are in the specified order")
# ---------------------------------------------------------------------------

_TIER0_TRAIT_COLUMNS: tuple = (
    "v_total_median_px_per_frame",
    "v_long_signed_median_px_per_frame",
    "v_long_abs_median_px_per_frame",
    "v_lat_signed_median_px_per_frame",
    "v_lat_abs_median_px_per_frame",
    "long_lat_ratio",
    "path_displacement_ratio",
    "angular_amplitude",
    "principal_axis_angle",
    "growth_axis_unreliable",
)
"""Declared order of the 10 trait/flag columns appended by Tier 0."""


_TIER0_TRAIT_UNITS: Dict[str, str] = {
    "v_total_median_px_per_frame": "px/frame",
    "v_long_signed_median_px_per_frame": "px/frame",
    "v_long_abs_median_px_per_frame": "px/frame",
    "v_lat_signed_median_px_per_frame": "px/frame",
    "v_lat_abs_median_px_per_frame": "px/frame",
    "long_lat_ratio": "—",
    "path_displacement_ratio": "—",
    "angular_amplitude": "rad",
    "principal_axis_angle": "rad",
    "growth_axis_unreliable": "bool",
}
"""Unit-string mapping for the 10 Tier-0 trait/flag columns.

Every value is in :data:`sleap_roots.circumnutation._constants.PIPELINE_UNIT_VOCABULARY`.
Consumed by the units-sidecar JSON writer (
:func:`sleap_roots.circumnutation._io.write_per_plant_csv`).
"""


# Six rotation-dependent traits that get NaN'd when the gate fires.
_ROTATION_DEPENDENT_TRAITS: tuple = (
    "v_long_signed_median_px_per_frame",
    "v_long_abs_median_px_per_frame",
    "v_lat_signed_median_px_per_frame",
    "v_lat_abs_median_px_per_frame",
    "long_lat_ratio",
    "principal_axis_angle",
)


def _emit_nan_row() -> Dict[str, Any]:
    """Return a trait-dict with NaN for all 9 numeric columns and False for the flag.

    Used when a track has fewer than 2 valid frames after NaN-drop and
    cannot be analyzed. The flag is False (not True) because reliability
    is undefined with insufficient data — we cannot judge unreliability
    without at least two consecutive frames to estimate noise from.
    """
    row: Dict[str, Any] = {col: float("nan") for col in _TIER0_TRAIT_COLUMNS[:-1]}
    row["growth_axis_unreliable"] = False
    return row


def _compute_one_track(group: pd.DataFrame, constants: ConstantsT) -> Dict[str, Any]:
    """Compute the 10 Tier-0 trait values for a single track.

    Implements the algorithm in `design.md` D5:

    1. Drop NaN rows on tip_x / tip_y and sort by frame.
    2. If fewer than 2 valid frames remain, emit NaN traits (flag False).
    3. Build gap-aware per-frame velocities ``velocity = Δxy / Δframe``.
    4. Compute step magnitudes; ``v_total_median``.
    5. Compute net displacement ``D``; if ``D == 0``, all rotation-dep
       traits NaN and flag True (closed-loop track).
    6. Otherwise build growth-axis unit vector ``û_g``, lateral
       perpendicular ``û_lat``, signed and absolute long/lat medians,
       ``principal_axis_angle``.
    7. Compute local SG residual via :func:`_noise.compute_sg_residual_xy`
       and apply the reliability gate (strict less-than `D < K * residual`).
    8. Compute ``angular_amplitude = max(ψ_g) − min(ψ_g)`` via
       :func:`_geometry.compute_psi_g` (rotation-invariant, survives gate).
    9. Compute ``path_displacement_ratio = L / D`` (NaN if `D == 0`).
    """
    # Step 1 — NaN-then-sort ordering is load-bearing
    subset = group.dropna(subset=["tip_x", "tip_y"]).sort_values("frame")
    n = len(subset)
    if n < 2:
        logger.debug(
            "Track has fewer than 2 valid frames after NaN-drop "
            "(group keys: %s); emitting NaN traits.",
            tuple(group[list(_IDENTITY_5_TUPLE)].iloc[0]) if len(group) > 0 else None,
        )
        return _emit_nan_row()

    xy = subset[["tip_x", "tip_y"]].to_numpy(dtype=float)
    frame = subset["frame"].to_numpy(dtype=float)

    # Step 2 — gap-aware diffs (Δframe normalizes for non-contiguous frame indices).
    # Suppress numpy divide-warnings: per spec Requirement "Tier 0 input-validation
    # boundary", duplicate (track_id, frame) rows produce Δframe = 0 and a
    # documented non-finite trait value. The warnings would otherwise spam stderr
    # on real fixtures that contain such duplicates (e.g., Nipponbare plate 001).
    delta_xy = np.diff(xy, axis=0)  # shape (n-1, 2)
    delta_frame = np.diff(frame)  # shape (n-1,)
    with np.errstate(divide="ignore", invalid="ignore"):
        velocity = delta_xy / delta_frame[:, None]  # shape (n-1, 2), units px/frame

    # Step 3 — step magnitudes and v_total_median
    steps = np.linalg.norm(velocity, axis=1)
    v_total_median = float(np.nanmedian(steps))

    # Step 4 — net displacement and growth axis
    displacement_vec = xy[-1] - xy[0]
    D = float(np.linalg.norm(displacement_vec))

    # Step 5 — local SG residual + reliability gate (single point of truth).
    # The gate has three contributing conditions; collapsing them into one
    # boolean expression removes the dual-write footgun that the prior
    # implementation had (gate set in step 5, then potentially overwritten
    # by step 6's D==0 short-circuit). Equivalence preserved across all
    # branches; tests 2.C.1–2.C.6 + 2.B.3 + 2.B.7 + 2.B.8 cover every case.
    sg_residual_local = _noise.compute_sg_residual_xy(
        xy[:, 0],
        xy[:, 1],
        window=constants.SG_WINDOW_SHORT,
        degree=constants.SG_DEGREE,
    )
    # Gate fires if: (a) D == 0 (closed loop — D < any positive K·residual),
    # OR (b) D < K · sg_residual when the residual is well-defined.
    # Strict less-than: at D == K * residual the axis is judged reliable.
    # NaN residual (track too short for SG) is treated as gate-does-not-fire.
    growth_axis_unreliable = D == 0.0 or (
        not math.isnan(sg_residual_local)
        and D < constants.GROWTH_AXIS_RELIABILITY_K * sg_residual_local
    )

    # Step 6 — D=0 closed-loop short-circuit: rotation-dep traits are NaN.
    # The growth_axis_unreliable flag is already True from step 5 (D==0 case
    # is one of its disjuncts).
    if D == 0.0:
        # ψ_g (rotation-invariant — survives) and v_total (already computed)
        psi_g = _geometry.compute_psi_g(xy[:, 0], xy[:, 1])
        angular_amplitude = (
            float(np.nanmax(psi_g) - np.nanmin(psi_g))
            if len(psi_g) > 0
            else float("nan")
        )
        return {
            "v_total_median_px_per_frame": v_total_median,
            "v_long_signed_median_px_per_frame": float("nan"),
            "v_long_abs_median_px_per_frame": float("nan"),
            "v_lat_signed_median_px_per_frame": float("nan"),
            "v_lat_abs_median_px_per_frame": float("nan"),
            "long_lat_ratio": float("nan"),
            "path_displacement_ratio": float("nan"),  # D == 0
            "angular_amplitude": angular_amplitude,
            "principal_axis_angle": float("nan"),
            "growth_axis_unreliable": growth_axis_unreliable,
        }

    # Normal case: D > 0, construct growth axis and project
    u_g = displacement_vec / D
    # Lateral perpendicular (90° rotation in standard math axes; under image-y-
    # down convention this is 90° clockwise as viewed on screen).
    u_lat = np.array([-u_g[1], u_g[0]])

    # Same errstate suppression for the projections: if velocity contains NaN
    # from a Δframe=0 row, the matmul propagates NaN — np.nanmedian then skips it.
    with np.errstate(invalid="ignore"):
        delta_long_per_frame = velocity @ u_g  # signed scalar per frame
        delta_lat_per_frame = velocity @ u_lat  # signed scalar per frame

    v_long_signed = float(np.nanmedian(delta_long_per_frame))
    v_long_abs = float(np.nanmedian(np.abs(delta_long_per_frame)))
    v_lat_signed = float(np.nanmedian(delta_lat_per_frame))
    v_lat_abs = float(np.nanmedian(np.abs(delta_lat_per_frame)))

    # Divide-by-zero → NaN (spec convention: NaN, not inf, not raise)
    long_lat_ratio = v_long_abs / v_lat_abs if v_lat_abs > 0 else float("nan")

    # principal_axis_angle: STANDARD math atan2(y_component, x_component) of the
    # growth-axis vector; NOT the BM-Eq.-20 ψ_g convention.
    principal_axis_angle = float(np.arctan2(u_g[1], u_g[0]))

    # Step 7 — ψ_g and angular_amplitude (rotation-invariant — survives gate)
    psi_g = _geometry.compute_psi_g(xy[:, 0], xy[:, 1])
    angular_amplitude = (
        float(np.nanmax(psi_g) - np.nanmin(psi_g)) if len(psi_g) > 0 else float("nan")
    )

    # Step 8 — path-length and path_displacement_ratio.
    # NaN-then-diff ordering (step 1) guarantees no NaN reached delta_xy, so
    # np.sum (not np.nansum) is correct.
    L = float(np.sum(np.linalg.norm(delta_xy, axis=1)))
    path_displacement_ratio = L / D  # D > 0 guaranteed here

    # Step 9 — apply gate: NaN the 6 rotation-dependent traits if firing.
    if growth_axis_unreliable:
        v_long_signed = float("nan")
        v_long_abs = float("nan")
        v_lat_signed = float("nan")
        v_lat_abs = float("nan")
        long_lat_ratio = float("nan")
        principal_axis_angle = float("nan")

    return {
        "v_total_median_px_per_frame": v_total_median,
        "v_long_signed_median_px_per_frame": v_long_signed,
        "v_long_abs_median_px_per_frame": v_long_abs,
        "v_lat_signed_median_px_per_frame": v_lat_signed,
        "v_lat_abs_median_px_per_frame": v_lat_abs,
        "long_lat_ratio": long_lat_ratio,
        "path_displacement_ratio": path_displacement_ratio,
        "angular_amplitude": angular_amplitude,
        "principal_axis_angle": principal_axis_angle,
        "growth_axis_unreliable": growth_axis_unreliable,
    }


def compute(
    trajectory_df: pd.DataFrame, constants: Optional[ConstantsT] = None
) -> pd.DataFrame:
    """Compute Tier 0 raw kinematic traits for each track in ``trajectory_df``.

    Returns a per-plant DataFrame with 18 columns: the 8 row-identity
    columns (``series``, ``sample_uid``, ``timepoint``, ``plate_id``,
    ``plant_id``, ``track_id``, ``genotype``, ``treatment``) followed by
    the 10 Tier-0 trait/flag columns in the declared order (see module
    docstring for the column-by-column contract).

    The canonical signature is `(trajectory_df, constants=None)` per the
    foundation's Package layout requirement. No `cadence_s` parameter:
    velocity columns are emitted in `px/frame` (cadence-independent);
    downstream users compose `convert_to_per_hour()` and `convert_to_mm()`
    utilities for `mm/hr` output.

    Args:
        trajectory_df: Per-frame tip-trajectory DataFrame containing the
            eight row-identity columns plus ``frame``, ``tip_x``, ``tip_y``.
            ``track_id`` must be non-NaN and integer-coercible. Per-row
            finiteness of ``tip_x`` / ``tip_y`` is NOT validated —
            ``NaN`` rows are dropped before computation and ``±inf``
            propagates through. Duplicate ``(track_id, frame)`` rows
            are NOT detected (would produce non-finite velocity from
            ``Δframe = 0``); the caller is responsible for upstream
            data integrity. See the spec Requirement "Tier 0
            input-validation boundary".
        constants: Optional :class:`ConstantsT` override-bag. When ``None``
            (default), module-level defaults from
            :mod:`sleap_roots.circumnutation._constants` are used.
            Consumed: ``SG_WINDOW_SHORT``, ``SG_DEGREE`` (for the local
            SG-residual computation), ``GROWTH_AXIS_RELIABILITY_K`` (the
            gate threshold multiplier).

    Returns:
        Per-plant DataFrame with one row per unique
        ``(series, sample_uid, plate_id, plant_id, track_id)`` 5-tuple
        in the input. Columns: 8 row-identity columns + 10 trait/flag
        columns in declared order. Sorted by the 5-tuple via stable sort.

    Raises:
        ValueError: If ``trajectory_df`` is not a ``pandas.DataFrame``;
            or if it is missing any of the eight row-identity columns
            or the three per-frame columns (``frame``, ``tip_x``,
            ``tip_y``); or if ``track_id`` contains NaN; or if the
            same 5-tuple has conflicting ``timepoint`` / ``genotype`` /
            ``treatment`` values across frames.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     "series": ["plate_001"] * 10,
        ...     "sample_uid": ["plate_001"] * 10,
        ...     "timepoint": ["T0"] * 10,
        ...     "plate_id": ["plate_001"] * 10,
        ...     "plant_id": [0] * 10,
        ...     "track_id": [0] * 10,
        ...     "genotype": ["Nipponbare"] * 10,
        ...     "treatment": ["MOCK"] * 10,
        ...     "frame": list(range(10)),
        ...     "tip_x": [float(i) for i in range(10)],
        ...     "tip_y": [0.0] * 10,
        ... })
        >>> result = compute(df)
        >>> float(result["v_total_median_px_per_frame"].iloc[0])
        1.0
    """
    # Input validation per spec Requirement "Tier 0 input-validation boundary".
    # Delegate to the foundation's validator to inherit its message format
    # (and any future tightening). The `instance, attribute` parameters are
    # attrs-validator-style; pass None since we're invoking outside an attrs
    # class context.
    if not isinstance(trajectory_df, pd.DataFrame):
        raise ValueError(
            f"trajectory_df must be a pandas DataFrame, got {type(trajectory_df).__name__}"
        )
    _validate_trajectory_df(None, None, trajectory_df)

    # Resolve constants to module defaults if None.
    if constants is None:
        constants = ConstantsT()

    # Per-plant template (8 row-identity columns, one row per unique 5-tuple).
    template = _build_per_plant_template_from_df(trajectory_df)

    # Compute per-track trait values. Use groupby with sort=False to preserve
    # input order; downstream merge re-aligns via the 5-tuple key.
    trait_rows = []
    for key, group in trajectory_df.groupby(
        list(_IDENTITY_5_TUPLE), dropna=False, sort=False
    ):
        traits = _compute_one_track(group, constants)
        # Re-key by 5-tuple for the merge step.
        identity = dict(zip(_IDENTITY_5_TUPLE, key))
        trait_rows.append({**identity, **traits})

    trait_df = pd.DataFrame(
        trait_rows,
        columns=list(_IDENTITY_5_TUPLE) + list(_TIER0_TRAIT_COLUMNS),
    )

    # Coerce identity dtypes on the trait_df side to match the template
    # (template has int64 track_id/plant_id; the trait_df might have ints from
    # the groupby key but we still cast explicitly for safety).
    for col in ("track_id", "plant_id"):
        if col in trait_df.columns and not trait_df[col].empty:
            trait_df[col] = trait_df[col].astype("int64")

    # Merge trait_df onto template via the 5-tuple key. Left-merge preserves
    # template's sort order (foundation contract).
    result = template.merge(trait_df, on=list(_IDENTITY_5_TUPLE), how="left")

    # Enforce declared column order: 8 row-identity columns + 10 trait/flag
    # columns. Spec scenario "Output DataFrame columns are in the specified
    # order" asserts this exact ordering.
    result = result[list(ROW_IDENTITY_COLUMNS) + list(_TIER0_TRAIT_COLUMNS)]

    return result
