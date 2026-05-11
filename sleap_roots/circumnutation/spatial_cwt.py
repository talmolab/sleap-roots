"""Tier 3b — spatial CWT machinery + ``L_gz`` peak-find + ``L_c`` decay fit (PR #9 will implement).

Computes a spatial CWT of ``κ(s)`` using the second-derivative complex
Gaussian ``cgau2`` mother wavelet by default
(:data:`~sleap_roots.circumnutation._constants.WAVELET_DEFAULT_SPATIAL`),
matching Rivière 2022 §"Kinematics: fine elongation measurements".
Peak-finds ``L_gz_estimate`` from the ``|κ(s)|`` envelope (theory.md
§6.4) and fits an exponential decay basal of the peak for
``L_c_estimate``. The trait-emission step (Tier 3c, PR #10) applies
the growth-zone mask once ``L_gz`` is known.
"""

import logging


logger = logging.getLogger(__name__)


def compute_scaleogram(
    kappa=None, ds=None, wavelet=None, scale_range=None, constants=None
):
    """Compute a spatial CWT scaleogram of ``κ(s)`` (PR #9 will implement).

    Args (when implemented):
        kappa: 1-D numpy array of midline curvature values along arc length.
        ds: Arc-length spacing between samples (px).
        wavelet: Mother wavelet name (default
            :data:`~sleap_roots.circumnutation._constants.WAVELET_DEFAULT_SPATIAL`).
        scale_range: Optional scale range; auto-derived if omitted.

    Returns (when implemented):
        A ``ScaleogramResult`` containing the CWT coefficients, scale-
        to-spatial-wavelength mapping, and cone-of-influence mask.

    Raises:
        NotImplementedError: Always; this is a stub for PR #9.
    """
    raise NotImplementedError("PR #9 — see docs/circumnutation/roadmap.md")
