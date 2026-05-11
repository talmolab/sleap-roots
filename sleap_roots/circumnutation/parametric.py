"""Tier 4 — partial parametric Bastien-Meroz fit (PR #11 will implement).

Phase 1 emits ``B_balance_number = L_gz / L_c`` (theory.md Eq. M-7),
``gamma_over_beta = L_c`` (alias, Meroz 2026 Eq. 6), and
``delta_dot_0_estimate = ω·R·Δφ / (2·ΔL)`` (Rivière 2022 Eq. 1).
The full ``(β, γ, θ_p)`` identification is Phase 2 — it requires a
gravitropism stimulus experiment to disentangle β from γ. Phase-2
columns are reserved as NaN to keep the multi-phase CSV stable.
"""

import logging


logger = logging.getLogger(__name__)


def compute(tier3_df=None, R_px=None, omega=None, Delta_phi=None):
    """Compute the Phase-1 partial parametric fit (PR #11 will implement).

    Args (when implemented):
        tier3_df: Per-plant DataFrame with Tier-3 traits already populated.
        R_px: Root cross-section radius in pixels (from
            :class:`~sleap_roots.circumnutation.CircumnutationInputs`).
        omega: Angular frequency from Tier-1 ``T_nutation_median``.
        Delta_phi: Angular amplitude from Tier-0 ``angular_amplitude``.

    Returns (when implemented):
        Per-plant DataFrame with Tier-4 trait columns plus the eight
        row-identity columns. Phase-2 columns (``beta``, ``gamma``,
        ``theta_p``) are present with NaN values.

    Raises:
        NotImplementedError: Always; this is a stub for PR #11.
    """
    raise NotImplementedError("PR #11 — see docs/circumnutation/roadmap.md")
