"""Tier 3c — traveling-wave validation traits (PR #10, ``add-circumnutation-tier3c-traits``).

Public callable: :func:`compute` — emits the Tier 3c traveling-wave validation
traits per track, the first consumer of PR #9's spatial-CWT machinery. Its
headline output is ``traveling_wave_residual``, the program's central
falsifiable test of the QPB steady-traveling-wave hypothesis
``lambda_spatial = v * T_nutation`` (``docs/circumnutation/theory.md`` §4.7).

This is a trait-emission module mirroring :mod:`sleap_roots.circumnutation.nutation`
(Tier 1) and :mod:`sleap_roots.circumnutation.psi_g` (Tier 2): the canonical
signature is ``compute(trajectory_df, cadence_s, constants=None) -> pd.DataFrame``
with a per-track 5-tuple groupby and the 8 row-identity columns. It emits 6
float64 trait columns (declared order): ``lambda_spatial_median_px``,
``lambda_spatial_variation``, ``traveling_wave_residual``, ``lambda_expected_px``,
``lambda_spatial_mad_px``, ``coi_valid_fraction``.

Reduced scope (PR #9 descope → GitHub issue #230): the ``L_gz``/``L_c``-dependent
traits and the growth-zone mask are NOT emitted (blocked on #230).

Anchors: spec at
``openspec/changes/add-circumnutation-tier3c-traits/specs/circumnutation/spec.md``;
design at ``openspec/changes/add-circumnutation-tier3c-traits/design.md`` +
``docs/superpowers/specs/2026-06-10-add-circumnutation-tier3c-traits-design.md``;
investigation at
``docs/circumnutation/investigations/2026-06-10-tier3c-traveling-wave/report.md``;
``docs/circumnutation/theory.md`` §4.7 (lambda = v*T), §6.4 (frame-domain unit
reconciliation), §7.4 (Tier 3 trait table + handoff) + Appendix B(6).
"""

import logging
from typing import Optional

import pandas as pd

from sleap_roots.circumnutation._constants import ConstantsT

logger = logging.getLogger(__name__)


def compute(
    trajectory_df: pd.DataFrame,
    cadence_s: float,
    constants: Optional[ConstantsT] = None,
) -> pd.DataFrame:
    """Compute Tier 3c traveling-wave validation traits for each track.

    Args:
        trajectory_df: Per-frame tip-trajectory DataFrame containing the eight
            row-identity columns plus ``frame``, ``tip_x``, ``tip_y``.
        cadence_s: Sampling cadence in seconds per frame (positive finite).
        constants: Optional :class:`ConstantsT` override-bag. When ``None``
            (default), module-level defaults are used.

    Returns:
        Per-plant DataFrame with one row per unique
        ``(series, sample_uid, plate_id, plant_id, track_id)`` 5-tuple: the 8
        row-identity columns followed by the 6 Tier 3c trait columns in the
        declared order.
    """
    # Minimal scaffold (PR #10 TDD step 1.2): the per-plant emission, validation,
    # spatial chain, calibration, and gating are implemented in subsequent TDD
    # steps (tasks 2-7). Returns an empty DataFrame until then.
    return pd.DataFrame()
