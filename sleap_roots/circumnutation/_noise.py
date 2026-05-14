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


def compute_d2_residual_xy(x: np.ndarray, y: np.ndarray) -> float:
    r"""Estimate localization noise σ via second-difference variance on ``(x, y)``.

    Computes the second-difference array
    ``Δ²x[t] = x[t+1] - 2·x[t] + x[t-1]`` (and similarly for ``y``), then
    returns the quadrature sum of standard deviations divided by ``√6``:

    .. math:: \sigma_{d2} = \frac{1}{\sqrt{6}} \sqrt{\mathrm{std}(\Delta^2 x)^2 + \mathrm{std}(\Delta^2 y)^2}

    For an i.i.d. noise process ``ε ~ (0, σ²)`` added to a smooth signal,
    the white-noise propagation rule ``Var(Δ²ε) = 6σ²`` gives
    ``std(Δ²x) = √6·σ_x``, so the quadrature sum recovers ``√(σ_x² + σ_y²)``.

    Args:
        x: 1-D array of x-coordinates (length ≥ 3).
        y: 1-D array of y-coordinates (same length as ``x``).

    Returns:
        Float quadrature-sum d2-noise estimate in pixels. Returns ``np.nan``
        and logs a ``DEBUG`` record when ``len(x) < 3`` (the second-difference
        operator requires at least 3 samples).

    Notes:
        Independent estimator from :func:`compute_sg_residual_xy`. The two
        estimators make different smoothness assumptions; pairwise agreement
        (``max/min ≤ 1.5`` is the QC clean-track threshold) is the
        cross-check. Theory: ``docs/circumnutation/theory.md`` §7.6;
        ``docs/circumnutation/preliminary_results_2026-05-07.md`` §3.3.
    """
    if len(x) < 3:
        logger.debug(
            "compute_d2_residual_xy: len(x)=%d < 3; returning NaN",
            len(x),
        )
        return float("nan")
    delta2_x = x[2:] - 2.0 * x[1:-1] + x[:-2]
    delta2_y = y[2:] - 2.0 * y[1:-1] + y[:-2]
    std_x = float(np.std(delta2_x))
    std_y = float(np.std(delta2_y))
    return float(np.sqrt(std_x**2 + std_y**2) / np.sqrt(6.0))


def compute_msd_residual_xy(
    x: np.ndarray,
    y: np.ndarray,
    window: int,
    degree: int,
    lag: int = 1,
) -> float:
    r"""Estimate localization noise σ via MSD extrapolation on SG-detrended ``(x, y)``.

    Applies a Savitzky-Golay filter of the given ``window`` and polynomial
    ``degree`` to detrend ``x`` and ``y`` independently, computes residuals
    ``x_res = x - x_smooth`` (similarly ``y``), then computes the 2D MSD at
    lag ``τ = lag``:

    .. math:: \mathrm{MSD}(\tau) = \langle (x_\mathrm{res}(t+\tau) - x_\mathrm{res}(t))^2 + (y_\mathrm{res}(t+\tau) - y_\mathrm{res}(t))^2 \rangle

    For an i.i.d. noise process ``ε ~ (0, σ²)`` added to a smooth signal,
    ``MSD(τ → 0) = 4σ²`` in 2D (= 2σ² per dimension × 2 dimensions). Inverting:

    .. math:: \sigma_{MSD} = \sqrt{\mathrm{MSD}(\tau)/4}

    **The factor of 4 (NOT 2) is load-bearing**: the 1D MSD is ``2σ²``, the 2D MSD
    aggregates both dimensions and is ``4σ²``. Confusing the two would
    over-estimate σ by ``√2``. Theory: ``docs/circumnutation/theory.md``
    §7.6; Michalet 2010 *Phys. Rev. E* 82:041914.

    Args:
        x: 1-D array of x-coordinates (length ≥ ``window + lag``).
        y: 1-D array of y-coordinates (same length as ``x``).
        window: SG window length for detrending (must be odd and ≤ ``len(x)``).
        degree: SG polynomial degree for detrending (must be < ``window``).
        lag: Lag in frames at which MSD is evaluated. Default 1 (theory.md §7.6).

    Returns:
        Float MSD-extrapolation noise estimate in pixels. Returns ``np.nan``
        and logs a ``DEBUG`` record when ``len(x) < window + lag`` (the SG
        detrend produces ``len`` residuals, of which ``len - lag`` pairs are
        needed for the MSD aggregation).

    Notes:
        Third independent estimator alongside :func:`compute_sg_residual_xy`
        and :func:`compute_d2_residual_xy`. Single-particle-tracking-derived
        methodology; CC-10 of ``docs/circumnutation/roadmap.md`` requires all
        three in Phase 1. The MSD method may differ from SG and d2 because
        it samples a longer time scale (``2·lag`` frames vs ``window``);
        empirical comparison is tracked in GitHub Issue α (PR #3 follow-up).
    """
    if len(x) < window + lag:
        logger.debug(
            "compute_msd_residual_xy: len(x)=%d < window+lag=%d; returning NaN",
            len(x),
            window + lag,
        )
        return float("nan")
    x_smooth = savgol_filter(x, window_length=window, polyorder=degree)
    y_smooth = savgol_filter(y, window_length=window, polyorder=degree)
    x_res = x - x_smooth
    y_res = y - y_smooth
    diffs_sq = (x_res[lag:] - x_res[:-lag]) ** 2 + (y_res[lag:] - y_res[:-lag]) ** 2
    msd = float(np.mean(diffs_sq))
    return float(np.sqrt(msd / 4.0))
