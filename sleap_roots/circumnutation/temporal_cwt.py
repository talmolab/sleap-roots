"""Tier 1 — temporal continuous wavelet transform machinery (PR #5 will implement).

Provides the generic CWT scaleogram + ridge-extraction helpers that
Tier 1 (PR #6, Derr-faithful trait emission) and Tier 2 (PR #7,
Bastien-Meroz ψ_g) compose. Uses the analytic complex Morlet
``cmor1.5-1.0`` mother wavelet by default
(:data:`~sleap_roots.circumnutation._constants.WAVELET_DEFAULT_TEMPORAL`),
chosen as a forensic match to Derr's Sept-2025 oracle scaleogram on
the same Suyash CMTN plate. The returned scaleogram is complex-valued;
ridge extraction is deterministic (no random tie-breaking) per
roadmap CC-6.
"""

import logging


logger = logging.getLogger(__name__)


def compute_scaleogram(
    x=None, cadence_s=None, wavelet=None, scale_range=None, constants=None
):
    """Compute a temporal CWT scaleogram (PR #5 will implement).

    Args (when implemented):
        x: 1-D numpy array of the input signal (e.g., one tip coordinate).
        cadence_s: Sampling cadence in seconds.
        wavelet: Mother wavelet name (default
            :data:`~sleap_roots.circumnutation._constants.WAVELET_DEFAULT_TEMPORAL`).
        scale_range: Optional ``(low, high)`` scale range; auto-derived
            from cadence and signal length if omitted.

    Returns (when implemented):
        A ``ScaleogramResult`` containing the complex CWT coefficients,
        scale-to-period mapping, and cone-of-influence mask.

    Raises:
        NotImplementedError: Always; this is a stub for PR #5.
    """
    raise NotImplementedError("PR #5 — see docs/circumnutation/roadmap.md")
