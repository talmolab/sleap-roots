"""QC tier — per-track signal-quality flags and noise estimators (PR #3).

Defined by ``docs/circumnutation/theory.md`` §7.6: emits 11 trait columns
per track summarizing track-level data quality:

- 3 independent SLEAP-localization-noise estimators (CC-10):
  ``sg_residual_xy``, ``d2_noise_xy``, ``msd_noise_xy``.
- 3 pairwise agreements: ``sg_d2_agreement``, ``sg_msd_agreement``,
  ``d2_msd_agreement``. Each is ``max(a, b) / min(a, b)``; NaN if either
  operand is NaN.
- 2 outlier-step diagnostics: ``frac_outlier_steps``, ``worst_step_ratio``.
- ``growth_axis_unreliable`` — bool flag, element-wise equal to Tier 0's
  emission of the same column by construction (both tiers recompute via
  the shared :func:`sleap_roots.circumnutation._noise.compute_sg_residual_xy`
  helper). See design.md D5 and spec Requirement "QC tier
  growth_axis_unreliable equality with Tier 0".
- ``track_is_clean`` — composite bool of 6 clauses (see
  :func:`_compose_track_is_clean_and_reason`).
- ``qc_failure_reason`` — comma-separated string of failure clauses in
  :data:`_FAILURE_CLAUSE_ORDER`; empty when ``track_is_clean == True``;
  literal sentinel ``"qc_inputs_insufficient"`` when the short-track gate
  fires (overrides per-clause concatenation).

Coordinate convention
---------------------
This module uses image-space coordinates where y increases downward
(theory.md §2.1). Inherited from Tier 0; no QC-specific consequence.

Pure-pixel + cadence-independent emission
-----------------------------------------
All trait columns use units in ``PIPELINE_UNIT_VOCABULARY``: ``px`` for
the 3 noise estimators, ``—`` for the 5 dimensionless ratios/fractions,
``bool`` for the 2 boolean flags, ``string`` for the failure-reason
column. NO ``cadence_s`` input, NO ``mm``-bearing columns. Downstream
users compose :func:`sleap_roots.circumnutation.units.convert_to_mm` for
human-readable units if they want them.

Composes with Tier 0
--------------------
QC's ``growth_axis_unreliable`` column is recomputed locally using the
same ``_noise.compute_sg_residual_xy`` helper and the same gate formula
as Tier 0's ``kinematics.compute``. Values are element-wise equal as
``bool`` dtype by construction. PR #14 pipeline composition may coalesce
or drop one of the two emissions (either choice is safe). See
``openspec/changes/.../design.md`` D5 for the rationale.

Caveat: noiseless input
-----------------------
Noiseless synthetic data (perfectly smooth trajectories like
``tip_x = np.arange(N)``, ``tip_y = 0``) makes all three noise estimators
return ``0.0``, which yields ``0/0 = NaN`` for all three pairwise
agreements. Under NaN-comparison semantics this fires the three
``*_agreement_high`` clauses and ``track_is_clean = False``. Real
SLEAP-tracked data always carries ≳ 0.1 px quantization noise so this
is benign in practice, but synthetic smoke tests should expect this
behavior.
"""

import logging
import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from sleap_roots.circumnutation import _noise
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
# QC output schema (locked by spec scenario "Output DataFrame columns
# are in the specified order")
# ---------------------------------------------------------------------------

_QC_TRAIT_COLUMNS: tuple = (
    "sg_residual_xy",
    "d2_noise_xy",
    "msd_noise_xy",
    "sg_d2_agreement",
    "sg_msd_agreement",
    "d2_msd_agreement",
    "frac_outlier_steps",
    "worst_step_ratio",
    "growth_axis_unreliable",
    "track_is_clean",
    "qc_failure_reason",
)
"""Declared order of the 11 QC trait/flag columns appended by qc.compute."""


_QC_TRAIT_UNITS: Dict[str, str] = {
    "sg_residual_xy": "px",
    "d2_noise_xy": "px",
    "msd_noise_xy": "px",
    "sg_d2_agreement": "—",
    "sg_msd_agreement": "—",
    "d2_msd_agreement": "—",
    "frac_outlier_steps": "—",
    "worst_step_ratio": "—",
    "growth_axis_unreliable": "bool",
    "track_is_clean": "bool",
    "qc_failure_reason": "string",
}
"""Unit-string mapping for the 11 QC trait/flag columns.

Every value is in :data:`sleap_roots.circumnutation._constants.PIPELINE_UNIT_VOCABULARY`.
Consumed by the units-sidecar JSON writer (:func:`sleap_roots.circumnutation._io.write_per_plant_csv`).
"""


_FAILURE_CLAUSE_ORDER: tuple = (
    "qc_inputs_insufficient",  # sentinel — overrides all other clauses
    "growth_axis_unreliable",
    "sg_d2_agreement_high",
    "sg_msd_agreement_high",
    "d2_msd_agreement_high",
    "frac_outlier_steps_high",
    "worst_step_ratio_high",
)
"""Canonical stable order for ``qc_failure_reason`` clause concatenation.

The first entry ``qc_inputs_insufficient`` is a sentinel: when the
short-track gate fires, ``qc_failure_reason`` is LITERALLY this single
string with no other clauses appended. The remaining 6 clauses can
co-occur and SHALL be concatenated via ``", ".join(...)`` in declared
tuple order.
"""


def _emit_short_track_row(growth_axis_unreliable: bool) -> Dict[str, Any]:
    """Return the all-NaN row used when the short-track gate fires.

    The 8 numeric traits and 3 pairwise agreements are NaN. The
    ``growth_axis_unreliable`` value is computed by the caller via the
    same Tier 0 logic so the equality contract is preserved even when
    the rest of QC's algorithm cannot run.
    """
    row: Dict[str, Any] = {
        "sg_residual_xy": float("nan"),
        "d2_noise_xy": float("nan"),
        "msd_noise_xy": float("nan"),
        "sg_d2_agreement": float("nan"),
        "sg_msd_agreement": float("nan"),
        "d2_msd_agreement": float("nan"),
        "frac_outlier_steps": float("nan"),
        "worst_step_ratio": float("nan"),
        "growth_axis_unreliable": growth_axis_unreliable,
        "track_is_clean": False,
        "qc_failure_reason": "qc_inputs_insufficient",
    }
    return row


def _pairwise_agreement(a: float, b: float) -> float:
    """Return ``max(a, b) / min(a, b)``; NaN if either operand is NaN or non-positive.

    Non-positive operands also yield NaN (the agreement is undefined for a
    zero noise estimate; in practice a smooth signal yields ``a == b == 0``
    and the 0/0 case is documented as "fires the *_high clause" per the
    caveat in this module's docstring).
    """
    if math.isnan(a) or math.isnan(b):
        return float("nan")
    if a <= 0.0 or b <= 0.0:
        return float("nan")
    return float(max(a, b) / min(a, b))


def _compose_track_is_clean_and_reason(
    traits: Dict[str, Any], constants: ConstantsT
) -> Tuple[bool, str]:
    """Compose ``track_is_clean`` and ``qc_failure_reason`` from the per-track traits.

    Implements the formula from spec Requirement "QC tier track_is_clean
    and qc_failure_reason composition". Returns (track_is_clean, reason).
    """
    clauses = []
    if bool(traits["growth_axis_unreliable"]):
        clauses.append("growth_axis_unreliable")
    if not (traits["sg_d2_agreement"] < constants.SG_D2_AGREEMENT_MAX):
        clauses.append("sg_d2_agreement_high")
    if not (traits["sg_msd_agreement"] < constants.SG_MSD_AGREEMENT_MAX):
        clauses.append("sg_msd_agreement_high")
    if not (traits["d2_msd_agreement"] < constants.D2_MSD_AGREEMENT_MAX):
        clauses.append("d2_msd_agreement_high")
    if not (traits["frac_outlier_steps"] < constants.FRAC_OUTLIER_STEPS_MAX):
        clauses.append("frac_outlier_steps_high")
    if not (traits["worst_step_ratio"] < constants.WORST_STEP_RATIO_MAX):
        clauses.append("worst_step_ratio_high")

    track_is_clean = len(clauses) == 0
    reason = "" if track_is_clean else ", ".join(clauses)
    return track_is_clean, reason


def _compute_growth_axis_unreliable_local(
    xy: np.ndarray, constants: ConstantsT
) -> Tuple[float, float, bool]:
    """Recompute growth_axis_unreliable using the same formula and inputs as Tier 0.

    Returns ``(D, sg_residual, growth_axis_unreliable)``. Mirrors
    ``kinematics.py:198-221`` exactly so the equality contract holds.

    Args:
        xy: ``(n, 2)`` array of (tip_x, tip_y) after dropna + sort, cast
            to ``dtype=float``.
        constants: Resolved ``ConstantsT`` instance.
    """
    D = float(np.linalg.norm(xy[-1] - xy[0]))
    sg_residual = _noise.compute_sg_residual_xy(
        xy[:, 0],
        xy[:, 1],
        window=constants.SG_WINDOW_SHORT,
        degree=constants.SG_DEGREE,
    )
    growth_axis_unreliable = D == 0.0 or (
        not math.isnan(sg_residual)
        and D < constants.GROWTH_AXIS_RELIABILITY_K * sg_residual
    )
    return D, sg_residual, growth_axis_unreliable


def _compute_one_track(group: pd.DataFrame, constants: ConstantsT) -> Dict[str, Any]:
    """Compute the 11 QC trait values for a single track.

    Implements the algorithm in design.md D7:

    1. Drop NaN rows on tip_x/tip_y; sort by frame.
    2. ALWAYS compute ``growth_axis_unreliable`` using the same Tier 0
       logic (preserves equality contract).
    3. Short-track gate: if len < SG_WINDOW_SHORT, emit NaN traits +
       ``qc_inputs_insufficient`` sentinel.
    4. Otherwise compute the 3 noise estimators, 3 pairwise agreements,
       2 outlier-step diagnostics.
    5. Compose ``track_is_clean`` and ``qc_failure_reason``.
    """
    # Step 1 — NaN-then-sort ordering (matches Tier 0)
    subset = group.dropna(subset=["tip_x", "tip_y"]).sort_values("frame")
    n = len(subset)

    # Step 2 — ALWAYS compute growth_axis_unreliable (preserve equality
    # contract with Tier 0 in all paths)
    if n < 2:
        growth_axis_unreliable = False
        # No xy to work with; sg_residual is NaN by convention.
    else:
        xy = subset[["tip_x", "tip_y"]].to_numpy(dtype=float)
        _, _, growth_axis_unreliable = _compute_growth_axis_unreliable_local(
            xy, constants
        )

    # Step 3 — Short-track gate
    if n < constants.SG_WINDOW_SHORT:
        logger.debug(
            "qc.compute: short-track gate fires for len=%d < SG_WINDOW_SHORT=%d",
            n,
            constants.SG_WINDOW_SHORT,
        )
        return _emit_short_track_row(growth_axis_unreliable)

    # Step 4 — three noise estimators (sg from step 2 if it were stashed;
    # recompute for clarity since the cost is small and re-cast of xy is
    # already done)
    xy = subset[["tip_x", "tip_y"]].to_numpy(dtype=float)
    frame = subset["frame"].to_numpy(dtype=float)
    sg = _noise.compute_sg_residual_xy(
        xy[:, 0],
        xy[:, 1],
        window=constants.SG_WINDOW_SHORT,
        degree=constants.SG_DEGREE,
    )
    d2 = _noise.compute_d2_residual_xy(xy[:, 0], xy[:, 1])
    msd = _noise.compute_msd_residual_xy(
        xy[:, 0],
        xy[:, 1],
        window=constants.SG_WINDOW_SHORT,
        degree=constants.SG_DEGREE,
        lag=1,
    )

    # Step 5 — three pairwise agreements
    sg_d2 = _pairwise_agreement(sg, d2)
    sg_msd = _pairwise_agreement(sg, msd)
    d2_msd = _pairwise_agreement(d2, msd)

    # Step 6 — step magnitudes (gap-aware diff)
    delta_xy = np.diff(xy, axis=0)
    delta_frame = np.diff(frame)
    with np.errstate(divide="ignore", invalid="ignore"):
        steps = np.linalg.norm(delta_xy / delta_frame[:, None], axis=1)
    median = float(np.nanmedian(steps))
    if median == 0.0 or math.isnan(median):
        frac_outlier_steps = float("nan")
        worst_step_ratio = float("nan")
    else:
        frac_outlier_steps = float(
            (steps > constants.OUTLIER_STEP_RATIO * median).mean()
        )
        worst_step_ratio = float(np.nanmax(steps) / median)

    # Step 7 — assemble per-trait dict
    traits: Dict[str, Any] = {
        "sg_residual_xy": sg,
        "d2_noise_xy": d2,
        "msd_noise_xy": msd,
        "sg_d2_agreement": sg_d2,
        "sg_msd_agreement": sg_msd,
        "d2_msd_agreement": d2_msd,
        "frac_outlier_steps": frac_outlier_steps,
        "worst_step_ratio": worst_step_ratio,
        "growth_axis_unreliable": growth_axis_unreliable,
    }

    # Step 8 — compose track_is_clean and qc_failure_reason
    track_is_clean, reason = _compose_track_is_clean_and_reason(traits, constants)
    traits["track_is_clean"] = track_is_clean
    traits["qc_failure_reason"] = reason
    return traits


def compute(
    trajectory_df: pd.DataFrame, constants: Optional[ConstantsT] = None
) -> pd.DataFrame:
    """Compute QC tier per-track quality traits for each track in ``trajectory_df``.

    Returns a per-plant DataFrame with 19 columns: the 8 row-identity
    columns (``series``, ``sample_uid``, ``timepoint``, ``plate_id``,
    ``plant_id``, ``track_id``, ``genotype``, ``treatment``) followed by
    the 11 QC trait/flag columns in the declared order (see module
    docstring for the column-by-column contract).

    The canonical signature is ``(trajectory_df, constants=None)`` per
    the foundation's Package layout requirement. No ``cadence_s``
    parameter — QC traits are cadence-independent.

    Args:
        trajectory_df: Per-frame tip-trajectory DataFrame containing the
            eight row-identity columns plus ``frame``, ``tip_x``,
            ``tip_y``. Per-row finiteness of ``tip_x``/``tip_y`` is NOT
            validated — ``NaN`` rows are dropped before computation and
            ``±inf`` propagates through (see spec Requirement "QC tier
            input-validation boundary").
        constants: Optional :class:`ConstantsT` override-bag. When
            ``None`` (default), module-level defaults from
            :mod:`sleap_roots.circumnutation._constants` are used.
            Consumed: ``SG_WINDOW_SHORT``, ``SG_DEGREE``,
            ``OUTLIER_STEP_RATIO``, ``GROWTH_AXIS_RELIABILITY_K``,
            ``SG_D2_AGREEMENT_MAX``, ``SG_MSD_AGREEMENT_MAX``,
            ``D2_MSD_AGREEMENT_MAX``, ``FRAC_OUTLIER_STEPS_MAX``,
            ``WORST_STEP_RATIO_MAX``.

    Returns:
        Per-plant DataFrame with one row per unique
        ``(series, sample_uid, plate_id, plant_id, track_id)`` 5-tuple
        in the input. Columns: 8 row-identity columns + 11 trait/flag
        columns in declared order. Sorted by the 5-tuple via stable sort.

    Raises:
        ValueError: If ``trajectory_df`` is not a ``pandas.DataFrame``;
            or if it is missing any of the eight row-identity columns
            or the three per-frame columns (``frame``, ``tip_x``,
            ``tip_y``); or if ``track_id`` contains NaN; or if the
            same 5-tuple has conflicting ``timepoint`` / ``genotype`` /
            ``treatment`` values across frames.
    """
    if not isinstance(trajectory_df, pd.DataFrame):
        raise ValueError(
            f"trajectory_df must be a pandas DataFrame, got {type(trajectory_df).__name__}"
        )
    _validate_trajectory_df(None, None, trajectory_df)

    if constants is None:
        constants = ConstantsT()

    template = _build_per_plant_template_from_df(trajectory_df)

    trait_rows = []
    for key, group in trajectory_df.groupby(
        list(_IDENTITY_5_TUPLE), dropna=False, sort=False
    ):
        traits = _compute_one_track(group, constants)
        identity = dict(zip(_IDENTITY_5_TUPLE, key))
        trait_rows.append({**identity, **traits})

    trait_df = pd.DataFrame(
        trait_rows,
        columns=list(_IDENTITY_5_TUPLE) + list(_QC_TRAIT_COLUMNS),
    )

    # Coerce identity dtypes
    for col in ("track_id", "plant_id"):
        if col in trait_df.columns and not trait_df[col].empty:
            trait_df[col] = trait_df[col].astype("int64")

    # Coerce growth_axis_unreliable and track_is_clean to bool dtype
    # (load-bearing for the equality contract per spec Requirement "QC
    # tier growth_axis_unreliable equality with Tier 0")
    if not trait_df.empty:
        trait_df["growth_axis_unreliable"] = trait_df["growth_axis_unreliable"].astype(
            "bool"
        )
        trait_df["track_is_clean"] = trait_df["track_is_clean"].astype("bool")

    result = template.merge(trait_df, on=list(_IDENTITY_5_TUPLE), how="left")

    # Final re-selection: enforce the declared 19-column order. Load-bearing
    # because the groupby key uses _IDENTITY_5_TUPLE (omits `timepoint`),
    # but the template carries all 8 row-identity columns. Without this
    # re-selection `timepoint` could drop or shift.
    result = result[list(ROW_IDENTITY_COLUMNS) + list(_QC_TRAIT_COLUMNS)]

    return result
