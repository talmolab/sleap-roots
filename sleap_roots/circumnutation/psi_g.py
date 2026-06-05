"""Tier 2 — Bastien-Meroz ψ_g trait emission (PR #7, ``add-circumnutation-tier2-psi-g``).

Public callable: :func:`compute` — emits 4 self-contained Tier 2 trait
columns per track from the unwrapped velocity-direction angle ``ψ_g(t)``
(``_geometry.compute_psi_g``, ``atan2(dx, dy)`` per BM2016 Eq. 20 /
``docs/circumnutation/theory.md`` §3.5):

- ``T_psig_median_s`` — COI-masked median of the smoothed-ridge periods of
  the SG-detrended ψ_g (composes ``temporal_cwt.compute_scaleogram`` →
  ``extract_ridge`` → ``smooth_ridge``). The ONLY trait that uses the
  conditioned signal + CWT.
- ``delta_E_amplitude_proxy_px_per_frame`` — ``median(√(dx²+dy²))`` over raw
  finite velocity samples (Eq. 21 amplitude proxy; px/frame, no COI).
- ``handedness`` — ``int(np.sign(ψ_g[-1] − ψ_g[0]))`` over all finite frames
  (net unwrapped rotation; COI-free). ``+1`` ⇔ positive ``dψ_g/dt``.
- ``helix_signed_area_px2`` — ``_geometry.compute_signed_area`` (y-down
  Shoelace); ``sign(area) == handedness``.

Tier 2 is self-contained: it does NOT consume Tier 1 output. The 5th §7.3
trait ``psig_long_consistency`` (cross-tier ``T_psig ↔ T_nutation``) is
deferred to the roadmapped PR #13 Layer-3 work.

Anchors: spec at
``openspec/changes/add-circumnutation-tier2-psi-g/specs/circumnutation/spec.md``;
design at ``openspec/changes/add-circumnutation-tier2-psi-g/design.md`` +
``docs/superpowers/specs/2026-06-05-add-circumnutation-tier2-psi-g-design.md``
§13; ``docs/circumnutation/theory.md`` §3.5 (ψ_g) + §7.3 (Tier 2 trait table).
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

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
# Module-level contracts
# ---------------------------------------------------------------------------

# 4 trait columns in declared order (per spec ADDED requirement).
_PSIG_TRAIT_COLUMNS: tuple = (
    "T_psig_median_s",
    "delta_E_amplitude_proxy_px_per_frame",
    "handedness",
    "helix_signed_area_px2",
)

# Per-frame tip columns required on trajectory_df (alongside
# ROW_IDENTITY_COLUMNS).
_TIP_X_COLUMN: str = "tip_x"
_TIP_Y_COLUMN: str = "tip_y"


def _all_degenerate_traits() -> Dict[str, Any]:
    """Return the all-degenerate trait row (too-short / undefined track).

    ``handedness`` is the neutral ``0`` (no net rotation); the three float
    traits are NaN (undefined). Used by the ``N < 3`` short-circuit.
    """
    return {
        "T_psig_median_s": float("nan"),
        "delta_E_amplitude_proxy_px_per_frame": float("nan"),
        "handedness": 0,
        "helix_signed_area_px2": float("nan"),
    }


def _compute_one_track(
    group: pd.DataFrame,
    cadence_s: float,
    constants: ConstantsT,
) -> Dict[str, Any]:
    """Compute the 4 Tier 2 ψ_g traits for one track (per design §3.1).

    Placeholder pending the §4/§5 RED tests: returns the all-degenerate row.
    """
    return _all_degenerate_traits()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute(
    trajectory_df: pd.DataFrame,
    cadence_s: float,
    constants: Optional[ConstantsT] = None,
) -> pd.DataFrame:
    """Emit Tier 2 ψ_g traits per track (PR #7, theory.md §3.5 + §7.3).

    The 4 trait columns are emitted in declared order: ``T_psig_median_s``
    (s), ``delta_E_amplitude_proxy_px_per_frame`` (px/frame), ``handedness``
    (int64 {−1, 0, +1}), ``helix_signed_area_px2`` (px²). See the module
    docstring for each trait's definition.

    Unlike ``nutation.compute``, there is no ``coordinate`` parameter: ψ_g is
    computed from the raw 2-D tip trajectory ``(tip_x, tip_y)`` via
    ``_geometry.compute_psi_g`` (``atan2(dx, dy)``), so there is no
    1-D-projection choice.

    Args:
        trajectory_df: Per-frame tip-trajectory DataFrame with the eight
            row-identity columns + ``frame``, ``tip_x``, ``tip_y``.
        cadence_s: Frame cadence in seconds. Positive finite float.
        constants: Optional :class:`ConstantsT` override-bag. ``None``
            (default) resolves to module-level defaults.

    Returns:
        Per-plant DataFrame with 1 row per unique
        ``(series, sample_uid, plate_id, plant_id, track_id)`` 5-tuple.
        Columns: 8 row-identity columns + 4 trait columns in declared order
        (3 ``float64`` + 1 ``int64``).
    """
    resolved_constants = constants if constants is not None else ConstantsT()

    # Per-track loop (mirrors kinematics.py / qc.py / nutation.py precedent).
    trait_rows: list = []
    for key, group in trajectory_df.groupby(
        list(_IDENTITY_5_TUPLE), dropna=False, sort=False
    ):
        traits = _compute_one_track(
            group, cadence_s=cadence_s, constants=resolved_constants
        )
        identity = dict(zip(_IDENTITY_5_TUPLE, key))
        trait_rows.append({**identity, **traits})

    trait_df = pd.DataFrame(
        trait_rows,
        columns=list(_IDENTITY_5_TUPLE) + list(_PSIG_TRAIT_COLUMNS),
    )

    # Per-plant template via the shared `_io._build_per_plant_template_from_df`
    # helper (mirrors kinematics.compute / nutation.compute). Coerce trait_df's
    # identity dtypes to match the template's int64 keys BEFORE merge so merges
    # don't silently fall through to all-NaN on numeric-string IDs.
    template = _build_per_plant_template_from_df(trajectory_df)
    for col in ("track_id", "plant_id"):
        if col in trait_df.columns and col in template.columns:
            try:
                trait_df[col] = trait_df[col].astype(template[col].dtype)
            except (TypeError, ValueError) as exc:  # pragma: no cover
                raise ValueError(
                    f"trait_df[{col!r}] cannot be cast to template dtype "
                    f"{template[col].dtype!r}; this would silently break the "
                    f"per-plant merge and produce all-NaN traits."
                ) from exc

    result = template.merge(trait_df, on=list(_IDENTITY_5_TUPLE), how="left")

    # Enforce trait dtypes: 3 float64 + 1 int64. handedness uses fillna(0)
    # before astype(int64) (mirrors nutation's is_nutating fillna(False)
    # defensive pattern) — 0 is the documented neutral value, safe only under
    # the 5-tuple merge invariant (template derives from the same
    # trajectory_df → no unmatched rows in practice).
    for col in _PSIG_TRAIT_COLUMNS:
        if col == "handedness":
            result[col] = result[col].fillna(0).astype(np.int64)
        else:
            result[col] = result[col].astype(np.float64)

    # Enforce declared column order: 8 row-identity + 4 trait.
    result = result[list(ROW_IDENTITY_COLUMNS) + list(_PSIG_TRAIT_COLUMNS)]

    return result
