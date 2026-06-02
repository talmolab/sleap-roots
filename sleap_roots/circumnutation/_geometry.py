"""Shared geometry helpers for the circumnutation pipeline.

Single source of truth for the per-frame velocity-direction angle
``ψ_g(t)`` defined by Bastien & Meroz 2016 Eq. 20 and pinned in
``docs/circumnutation/theory.md`` §3.5. Consumed by:

- Tier 0 (``kinematics.py``, PR #2) — uses the peak-to-peak extent of
  ``ψ_g(t)`` as ``angular_amplitude``.
- Tier 2 (``psi_g.py``, PR #7 — not yet implemented) — will apply CWT
  to ``ψ_g(t)`` to extract ``T_psig_median`` and compose the
  ``handedness`` trait.

The argument order in ``np.arctan2(dx, dy)`` (``dx`` first, then ``dy``)
is convention-critical. The reversed order ``atan2(dy, dx)`` would
offset every ``ψ_g`` value by ``π/2`` AND flip the sign of
``mean dψ_g/dt``, silently inverting PR #7's ``handedness`` trait
(which by Bastien-Meroz 2016 §"Constant principal direction of growth"
assigns ``+1 = counterclockwise``). **Always use** ``arctan2(dx, dy)``
in this module; the docstring on :func:`compute_psi_g` reiterates this.

Theory reference:

- Bastien R., Meroz Y. (2016). PLoS Comp. Biol. 12(12):e1005238.
  arXiv:1603.00459. Equation 20 (tip-only ψ_g extraction).
- ``docs/circumnutation/theory.md`` §3.5 — pipeline convention note
  *"The pipeline must use `atan2(dx/dt, dy/dt)` and unwrap the result"*.
"""

import logging
import math

import numpy as np


logger = logging.getLogger(__name__)


def compute_psi_g(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute the unwrapped velocity-direction angle ψ_g(t) per BM-Eq.-20.

    Computes ``ψ_g[i] = atan2(dx[i], dy[i])`` where ``dx = np.diff(x)``
    and ``dy = np.diff(y)``, then unwraps the result so the time series
    is continuous (no ``±2π`` jumps at the atan2 branch cut).

    **Argument order is convention-critical**: this function uses
    ``np.arctan2(dx, dy)`` (``dx`` first, then ``dy``) per Bastien-Meroz
    2016 Eq. 20 / theory.md §3.5. **Do not change to** ``arctan2(dy, dx)``
    — the reversed order would silently invert PR #7's ``handedness``
    trait sign convention.

    Args:
        x: 1-D array of x-coordinates.
        y: 1-D array of y-coordinates (same length as ``x``).

    Returns:
        Length-``(len(x) - 1)`` 1-D array of unwrapped ``ψ_g`` values in
        radians. Returns an empty array ``np.array([])`` when ``len(x) < 2``
        (no consecutive frames to compute velocity from).

    Notes:
        Used by Tier 0 (PR #2) to compute ``angular_amplitude =
        max(ψ_g) − min(ψ_g)`` (peak-to-peak extent; rotation-invariant
        under offset and sign-flip). Used by PR #7 (``psi_g.py``) to
        compute ``T_psig_median`` via CWT.

        The unwrap uses ``np.unwrap`` default ``discont=π`` — appropriate
        for the atan2 branch cut where the natural jump is ``±2π``.
    """
    if len(x) < 2:
        return np.array([], dtype=float)
    dx = np.diff(x)
    dy = np.diff(y)
    # NB: argument order — dx first, dy second — per BM 2016 Eq. 20 / theory.md §3.5.
    psi = np.arctan2(dx, dy)
    return np.unwrap(psi)


def project_to_growth_axis_perpendicular(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Project (x, y) tip positions onto the growth-axis perpendicular (PR #6, CC-7).

    Estimates the growth-axis unit vector from net displacement
    ``(x[-1]-x[0], y[-1]-y[0])``, constructs the perpendicular unit
    vector via 90° rotation ``(-u_g[1], u_g[0])``, centers (x, y) at
    their mean, and projects onto the perpendicular axis. Returns a 1D
    float64 lateral position time series of length ``len(x)``.

    This is the load-bearing CC-7 lateral-coordinate preprocessing for
    PR #6's ``nutation.compute``. Per PR #5's GREEN-phase observation
    (archived design.md Reconciliation Appendix), raw ``tip_x`` on
    plate-001 carries ~70-170 px of growth-axis drift that dominates
    the ~10 px nutation signal. Projecting onto the perpendicular
    isolates the nutation component.

    Args:
        x: 1-D array of x-coordinates. Must be finite (no NaN/±inf).
            Must match ``len(y)``.
        y: 1-D array of y-coordinates (same length as ``x``). Must be
            finite.

    Returns:
        Length-``len(x)`` 1-D float64 array of lateral positions
        (perpendicular component, centered at zero). On a stationary
        track (zero net displacement, where the growth axis is
        undefined), returns ``np.full(len(x), np.nan, dtype=np.float64)``
        — the graceful-NaN policy per design.md D2 (mirrors kinematics'
        precedent and avoids forcing downstream callers to handle
        ValueError exceptions).

    Raises:
        ValueError: If ``x`` and ``y`` have different lengths.
        ValueError: If ``x`` or ``y`` contain non-finite values
            (NaN or ±inf).

    Notes:
        Image-y-downward convention (SLEAP standard): the 90° rotation
        ``u_perp = (-u_g[1], u_g[0])`` gives the perpendicular pointing
        in a specific handedness, but downstream traits (CWT period,
        FFT amplitude, band-power ratio) are sign-invariant — flipping
        the perpendicular direction leaves the magnitude spectrum
        unchanged. So this sign convention is not externally observable.

        Centering before projection prevents the lateral signal from
        carrying a DC offset that would otherwise pollute the
        downstream FFT-based noise floor + band-power computations.

    Examples:
        >>> import numpy as np
        >>> # Track moving along +y with small lateral oscillation in x.
        >>> y = np.linspace(0.0, 100.0, 101)
        >>> x = 5.0 * np.sin(2 * np.pi * y / 25.0)
        >>> lateral = project_to_growth_axis_perpendicular(x, y)
        >>> lateral.shape == (101,)
        True
        >>> # Stationary track returns all-NaN.
        >>> stat_x = np.array([1.0, 1.5, 2.0, 1.5, 1.0])
        >>> stat_y = np.array([0.0, 0.5, 1.0, 0.5, 0.0])
        >>> result = project_to_growth_axis_perpendicular(stat_x, stat_y)
        >>> np.all(np.isnan(result))
        True
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) != len(y):
        raise ValueError(
            f"x and y must have the same length; got len(x)={len(x)} "
            f"len(y)={len(y)}"
        )
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("x and y must contain only finite values (no NaN, no ±inf)")
    n = len(x)
    if n < 2:
        # Single-point or empty: no displacement to define a growth axis.
        return np.full(n, np.nan, dtype=np.float64)
    dx_net = float(x[-1] - x[0])
    dy_net = float(y[-1] - y[0])
    net_length = math.hypot(dx_net, dy_net)
    if net_length == 0.0:
        logger.debug(
            "project_to_growth_axis_perpendicular: zero net displacement, "
            "returning all-NaN (n=%d)",
            n,
        )
        return np.full(n, np.nan, dtype=np.float64)
    u_g_x = dx_net / net_length
    u_g_y = dy_net / net_length
    # 90° rotation: (-u_g[1], u_g[0]).
    u_perp_x = -u_g_y
    u_perp_y = u_g_x
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    lateral = x_centered * u_perp_x + y_centered * u_perp_y
    return lateral.astype(np.float64, copy=False)
