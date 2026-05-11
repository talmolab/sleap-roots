"""Tier 2 — Bastien-Meroz ``ψ_g(t)`` tip tangent angle (PR #7 will implement).

Computes ``ψ_g(t) = atan2(dy/dt, dx/dt)``, unwrapped, with
Savitzky-Golay smoothing of ``(x, y)`` before differentiation
(per ``docs/circumnutation/theory.md`` §3.5). The wavelet of
``ψ_g(t)`` provides the temporal-domain Bastien-Meroz tangent-angle
oscillator; cross-tier consistency between ``T_psig_median`` and the
Tier-1 ``T_nutation_median`` flags violations of the H1 spatial
homogeneity assumption (theory.md §3.4 / §3.7).
"""

import logging


logger = logging.getLogger(__name__)


def compute_psi_g(x=None, y=None, sg_window=None, sg_degree=None, constants=None):
    """Compute the unwrapped tip tangent angle ``ψ_g(t)`` (PR #7 will implement).

    Args (when implemented):
        x: 1-D numpy array of tip x-coordinates.
        y: 1-D numpy array of tip y-coordinates.
        sg_window: Savitzky-Golay window length (frames; default
            :data:`~sleap_roots.circumnutation._constants.SG_WINDOW_SHORT`).
        sg_degree: Polynomial degree (default
            :data:`~sleap_roots.circumnutation._constants.SG_DEGREE`).

    Returns (when implemented):
        1-D numpy array of unwrapped angles (radians).

    Raises:
        NotImplementedError: Always; this is a stub for PR #7.
    """
    raise NotImplementedError("PR #7 — see docs/circumnutation/roadmap.md")
