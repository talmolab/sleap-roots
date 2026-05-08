"""Tier 0 — raw tip kinematics (PR #2 will implement).

Defined by ``docs/circumnutation/theory.md`` §7.1: emits
``v_total_median``, ``v_long_median``, ``v_lat_median``,
``long_lat_ratio``, ``path_displacement_ratio``, ``angular_amplitude``,
``principal_axis_angle``. Computation is purely kinematic — no
spectral analysis. The growth axis is defined per-track as the unit
vector from ``(x_1, y_1)`` to ``(x_N, y_N)`` (net displacement); when
net displacement falls below ``GROWTH_AXIS_RELIABILITY_K * sg_residual``,
the QC tier flags the track as ``growth_axis_unreliable`` and the
rotation-dependent traits are NaN'd.
"""

import logging


logger = logging.getLogger(__name__)


def compute(*args, **kwargs):
    """Compute Tier 0 raw kinematic traits (PR #2 will implement).

    Args (when implemented):
        trajectory_df: Per-track tip-trajectory DataFrame from
            :class:`~sleap_roots.circumnutation.CircumnutationInputs`.

    Returns (when implemented):
        Per-plant DataFrame with the seven Tier-0 trait columns plus
        the eight row-identity columns.

    Raises:
        NotImplementedError: Always; this is a stub for PR #2.
    """
    raise NotImplementedError("PR #2 — see docs/circumnutation/roadmap.md")
