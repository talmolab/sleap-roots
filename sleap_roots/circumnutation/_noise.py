"""Shared noise-estimation helpers for the circumnutation pipeline.

Single source of truth for the Savitzky-Golay residual formula used as a
local tracking-noise estimator. Consumed by:

- Tier 0 (``kinematics.py``, PR #2) — uses the residual as the
  growth-axis reliability gate value (``D < K * sg_residual_xy_local``).
- QC tier (``qc.py``, PR #3 — not yet implemented) — will emit
  ``sg_residual_xy`` as a canonical per-track trait using this same
  formula.

Keeping the formula here (and not duplicated across tier modules)
guarantees the Tier 0 gate value and the canonical QC trait are
numerically identical for the same inputs.

Theory references:

- Press et al. *Numerical Recipes* (any edition) — Savitzky-Golay
  residual estimation is a standard signal-processing technique.
- ``docs/circumnutation/theory.md`` §7.6 — QC tier methodological note
  on the two-estimator agreement (`sg_residual_xy` vs `d2_noise_xy`).
"""

import logging

import numpy as np
from scipy.signal import savgol_filter


logger = logging.getLogger(__name__)


def compute_sg_residual_xy(
    x: np.ndarray, y: np.ndarray, window: int, degree: int
) -> float:
    r"""Estimate localization noise σ via SG-residual on ``(x, y)``.

    Applies a Savitzky-Golay filter of the given ``window`` and polynomial
    ``degree`` independently to ``x`` and ``y``, computes the standard
    deviation of residuals ``(raw − smoothed)`` for each, and returns
    their quadrature sum:

    .. math:: \sigma_{SG} = \sqrt{\mathrm{std}(x - x_{smooth})^2 + \mathrm{std}(y - y_{smooth})^2}

    For a smooth signal of polynomial degree ≤ ``degree`` the residual
    is identically zero. For a signal of the form
    ``s(t) + i.i.d. N(0, σ²)``, the function returns approximately
    ``√(σ_x² + σ_y²)`` (the SG filter slightly under-estimates σ).

    Args:
        x: 1-D array of x-coordinates (length ≥ ``window``).
        y: 1-D array of y-coordinates (length ≥ ``window``; same length as ``x``).
        window: SG window length (must be odd and ≤ ``len(x)``).
        degree: SG polynomial degree (must be < ``window``).

    Returns:
        Float quadrature-sum residual in pixels. Returns ``np.nan`` and
        logs a ``DEBUG`` record when ``len(x) < window`` (the SG filter
        cannot be applied) — the caller is expected to handle this case
        (e.g., Tier 0 treats it as "gate inputs unavailable" and skips
        the reliability check for that track).

    Notes:
        Used by Tier 0 (PR #2) for the growth-axis-reliability gate value
        and reserved for PR #3 QC's canonical ``sg_residual_xy`` trait
        emission. Both tiers SHALL call this function rather than
        re-implementing the formula; the two values are guaranteed
        identical for identical inputs.
    """
    if len(x) < window:
        logger.debug(
            "compute_sg_residual_xy: len(x)=%d < window=%d; returning NaN",
            len(x),
            window,
        )
        return float("nan")
    x_smooth = savgol_filter(x, window_length=window, polyorder=degree)
    y_smooth = savgol_filter(y, window_length=window, polyorder=degree)
    std_x = float(np.std(x - x_smooth))
    std_y = float(np.std(y - y_smooth))
    return float(np.sqrt(std_x**2 + std_y**2))
