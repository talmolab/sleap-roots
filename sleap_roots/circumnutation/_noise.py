"""Shared SLEAP-localization-noise-estimation helpers for the circumnutation pipeline.

Single source of truth for the three independent noise-estimator formulas
emitted by the QC tier (PR #3 per CC-10) and shared with Tier 0 (PR #2)
for the growth-axis reliability gate. Three deterministic helpers:

- :func:`compute_sg_residual_xy` — Savitzky-Golay residual quadrature
  sum. Consumed by Tier 0 (``kinematics.py``, PR #2) for the growth-axis
  reliability gate value (``D < K * sg_residual_xy_local``) AND by the
  QC tier (``qc.py``, PR #3) as the canonical ``sg_residual_xy`` trait
  emission. Keeping the formula here (not duplicated across tier modules)
  guarantees the Tier 0 gate value and the canonical QC trait are
  numerically identical for the same inputs.
- :func:`compute_d2_residual_xy` — second-difference variance estimator
  with ``/sqrt(6)`` white-noise-propagation normalization. Consumed by
  QC tier as the canonical ``d2_noise_xy`` trait.
- :func:`compute_msd_residual_xy` — MSD-extrapolation estimator (2D
  MSD = 4σ²; the factor of 4 is load-bearing). Consumed by QC tier as
  the canonical ``msd_noise_xy`` trait per CC-10.

All three helpers return ``float`` and emit ``logger.debug`` + ``np.nan``
when the input is too short for their respective formulas.

Theory references:

- Press et al. *Numerical Recipes* (any edition) — Savitzky-Golay
  residual estimation is a standard signal-processing technique.
- ``docs/circumnutation/preliminary_results_2026-05-07.md`` §3.3 — the
  SG-residual and d2 formulas anchored to plate 001 reference values.
- ``docs/circumnutation/theory.md`` §7.6 — QC tier methodological note
  on the three-estimator agreement (sg / d2 / MSD).
- Michalet 2010 *Phys. Rev. E* 82:041914 — MSD-extrapolation method
  from single-particle-tracking literature.
"""

import logging

import numpy as np
import scipy.fft
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


def compute_sg_detrended(
    x: np.ndarray, window: int, polynomial_order: int
) -> np.ndarray:
    """Return the 1D Savitzky-Golay residual ``x - savgol(x)`` (PR #6, S1 round-1).

    Implements the SG-detrending prescription from ``preliminary_results
    _2026-05-07.md`` §3.4: "Detrend the lateral coordinate ℓ(t) with a
    long-window Savitzky-Golay filter (window = 23 frames ≈ 2 nutation
    periods, polynomial order 3). This suppresses the oscillation and
    retains slow centerline drift" — i.e., the SG-smoothed output captures
    the SLOW drift, and the RESIDUAL (raw - smoothed) is the OSCILLATION
    component that downstream CWT/FFT acts on.

    PR #6's ``nutation.compute`` applies this helper to the lateral
    signal AFTER ``_geometry.project_to_growth_axis_perpendicular`` and
    BEFORE the temporal CWT + Fourier noise floor + band-power
    computations. Without this detrending, low-frequency centerline
    drift contaminates the spectral analysis (PR #5's GREEN-phase
    observation: ~70-170 px drift on plate-001).

    Args:
        x: 1-D float array. Must be longer than ``window``.
        window: Window length in frames (must be odd, > 0, and >
            ``polynomial_order``). The PR #6 default sourced via
            ``constants.SG_WINDOW_DETREND = 23``.
        polynomial_order: Polynomial degree for the SG filter (must be
            < ``window``). The PR #6 default sourced via
            ``constants.SG_DEGREE = 3``.

    Returns:
        Length-``len(x)`` 1-D ``float64`` residual array (raw minus
        SG-smoothed-low-pass = oscillation component). Returns
        ``np.full(len(x), np.nan, dtype=np.float64)`` when ``len(x) <
        window`` (boundary conditions undefined for short inputs).

    Notes:
        Uses ``mode='nearest'`` for boundary handling: the filter
        replicates the nearest valid value at the array edges. This is
        the recommended scipy boundary policy for non-periodic signals.
    """
    x = np.asarray(x, dtype=np.float64)
    if len(x) < window:
        logger.debug(
            "compute_sg_detrended: len(x)=%d < window=%d, returning all-NaN",
            len(x),
            window,
        )
        return np.full(len(x), np.nan, dtype=np.float64)
    smoothed = savgol_filter(
        x,
        window_length=window,
        polyorder=polynomial_order,
        mode="nearest",
    )
    residual = x - smoothed
    return residual.astype(np.float64, copy=False)


def compute_fourier_noise_floor(
    x: np.ndarray,
    cadence_s: float,
    t_nutation_median_s: float,
    factor: float,
) -> float:
    """Median Fourier amplitude over the out-of-band region (PR #6, CC-8).

    Per ``docs/circumnutation/roadmap.md`` CC-8 verbatim: the noise floor
    estimate is the median of ``|scipy.fft.rfft(x)|`` over frequencies
    ``f > factor / t_nutation_median_s`` — i.e., the "well above the
    candidate-nutation-frequency" region of the spectrum. This is
    PR #6's noise-floor estimator consumed by ``nutation.compute`` for
    the ``noise_floor_estimate`` trait and the ``is_nutating`` gate.

    Args:
        x: 1-D float array (typically the SG-detrended lateral signal).
            Must have ``len(x) >= 2`` for the rfft to be defined.
        cadence_s: Frame cadence in seconds. Positive finite.
        t_nutation_median_s: Candidate nutation period in seconds. The
            out-of-band frequency cutoff is computed as
            ``factor / t_nutation_median_s``.
        factor: Out-of-band cutoff factor. The PR #6 default sourced
            via ``constants.NOISE_FLOOR_OUT_OF_BAND_FACTOR = 5.0``.

    Returns:
        Median amplitude (a single ``float``) over the out-of-band
        region of the Fourier spectrum. Returns ``np.nan`` when the
        input is too short (``len(x) < 2``) or when the out-of-band
        region is empty (cutoff exceeds the Nyquist frequency).
    """
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 2:
        logger.debug(
            "compute_fourier_noise_floor: len(x)=%d < 2, returning NaN",
            len(x),
        )
        return float("nan")
    spectrum = np.abs(scipy.fft.rfft(x))
    freqs = scipy.fft.rfftfreq(len(x), d=cadence_s)
    if not np.isfinite(t_nutation_median_s) or t_nutation_median_s <= 0:
        logger.debug(
            "compute_fourier_noise_floor: invalid t_nutation_median_s=%r, returning NaN",
            t_nutation_median_s,
        )
        return float("nan")
    f_cut = factor / t_nutation_median_s
    band_mask = freqs > f_cut
    if not band_mask.any():
        logger.debug(
            "compute_fourier_noise_floor: empty out-of-band region "
            "(f_cut=%.6f Hz > nyquist=%.6f Hz), returning NaN",
            f_cut,
            freqs[-1] if len(freqs) > 0 else 0.0,
        )
        return float("nan")
    return float(np.median(spectrum[band_mask]))
