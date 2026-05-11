"""Tier 3a — tip-trail-as-midline reconstruction (PR #8 will implement).

Implements the tip-trail-as-midline identity (theory.md §6.1): for an
apically-growing organ where tissue past the elongation zone does not
reshape, the curve formed by past tip positions IS the organ midline.
This module returns the reconstructed midline parameterized by arc
length ``s(τ) = ∫|v|dσ``, plus per-frame trajectory curvature
``κ_path(t)``, midline curvature ``κ(s)``, and tip speed ``v(t)``.
SG smoothing is applied to ``(x, y)`` before second-derivative
operations (which would amplify 1-px tip noise into curvature spikes;
theory.md §6.2). The ``L_gz`` mask is **not** applied here — it lands
in PR #10 (Tier 3c) after PR #9's spatial CWT detects the peak.
"""

import logging


logger = logging.getLogger(__name__)


def reconstruct(x=None, y=None, cadence_s=None, sg_window=None):
    """Reconstruct the midline from a tip trail (PR #8 will implement).

    Args (when implemented):
        x: 1-D numpy array of tip x-coordinates.
        y: 1-D numpy array of tip y-coordinates.
        cadence_s: Sampling cadence in seconds.
        sg_window: Savitzky-Golay window length (default
            :data:`~sleap_roots.circumnutation._constants.SG_WINDOW_SHORT`).

    Returns (when implemented):
        A ``MidlineResult`` containing arc length ``s``, smoothed
        ``(x, y)``, ``κ_path`` (time-domain), ``κ_arc`` (arc-length
        domain), per-frame velocity magnitude, and a velocity-bandpass
        mask (``|v| > NOISE_MASK_K · σ_v``) for sub-noise frames.

    Raises:
        NotImplementedError: Always; this is a stub for PR #8.
    """
    raise NotImplementedError("PR #8 — see docs/circumnutation/roadmap.md")
