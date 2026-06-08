"""Shared geometry helpers for the circumnutation pipeline.

Single source of truth for the per-frame velocity-direction angle
``ψ_g(t)`` defined by Bastien & Meroz 2016 Eq. 20 and pinned in
``docs/circumnutation/theory.md`` §3.5. Consumed by:

- Tier 0 (``kinematics.py``, PR #2) — uses the peak-to-peak extent of
  ``ψ_g(t)`` as ``angular_amplitude``.
- Tier 2 (``psi_g.py``, PR #7) — applies CWT to ``ψ_g(t)`` to extract
  ``T_psig_median_s`` and composes the ``handedness`` /
  ``helix_signed_area_px2`` traits (the latter via
  :func:`compute_signed_area`).

The argument order in ``np.arctan2(dx, dy)`` (``dx`` first, then ``dy``)
is convention-critical. The reversed order ``atan2(dy, dx)`` would
offset every ``ψ_g`` value by ``π/2`` AND flip the sign of
``mean dψ_g/dt``, silently inverting PR #7's ``handedness`` trait.
**Sign convention (anchored on the sign, not the word):**
``handedness = +1`` ⇔ ``ψ_g`` increasing ⇔ **positive** ``mean dψ_g/dt``;
:func:`compute_signed_area` is negated so ``sign(area) == handedness``.
In physical terms ``+1`` is clockwise in standard (y-up) math axes and
counterclockwise as displayed in the y-down image frame — the program
anchors on the ``dψ_g/dt`` sign, NOT the ambiguous word
"counterclockwise". **Always use** ``arctan2(dx, dy)`` in this module.

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
        >>> # Zero-net-displacement track (returns to start) yields all-NaN.
        >>> # The gate is on net displacement, NOT stationarity — a closed-
        >>> # loop trajectory like this one fails it even though it moves.
        >>> closed_x = np.array([1.0, 1.5, 2.0, 1.5, 1.0])
        >>> closed_y = np.array([0.0, 0.5, 1.0, 0.5, 0.0])
        >>> result = project_to_growth_axis_perpendicular(closed_x, closed_y)
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


def compute_signed_area(x: np.ndarray, y: np.ndarray) -> float:
    """Signed area enclosed by the tip trajectory (y-down Shoelace; PR #7).

    Computes the **y-down-corrected** Shoelace signed area

    ``A = 0.5 * Σ_i (x_{i+1}·y_i − x_i·y_{i+1})``  (cyclic)

    which is the **negation** of the standard
    ``0.5 * Σ_i (x_i·y_{i+1} − x_{i+1}·y_i)``. The negation makes the sign
    agree with PR #7's ``handedness`` trait under the same image-y-down
    ``atan2(dx, dy)`` convention :func:`compute_psi_g` encodes:

    ``sign(compute_signed_area(x, y)) == handedness ==
    int(np.sign(ψ_g[-1] − ψ_g[0]))``.

    **Sign is load-bearing** and anchored on the ``dψ_g/dt`` sign, NOT the
    ambiguous word "counterclockwise": a positive area corresponds to
    ``handedness = +1`` (``ψ_g`` increasing). Verified by the absolute
    hand-built anchor ``x=[0,1,1,0], y=[0,0,1,1] → −1.0`` (standard Shoelace
    ``+1.0``), whose net ``ψ_g`` change is ``−π`` (``handedness = −1``).

    Used by ``psi_g.compute`` (PR #7) to emit ``helix_signed_area_px2`` — an
    independent confirmation of ``handedness``.

    Args:
        x: 1-D array of x-coordinates.
        y: 1-D array of y-coordinates (same length as ``x``).

    Returns:
        The signed area as a Python ``float`` (px² when ``x``/``y`` are in
        pixels). Returns ``0.0`` for inputs of fewer than 3 points (a
        degenerate polygon has no area). Returns ``NaN`` when any coordinate
        is non-finite (an explicit guard — the ``psi_g.compute`` caller
        short-circuits non-finite tracks before calling, so the trait reports
        NaN rather than ``0.0`` for too-short tracks).

    Raises:
        ValueError: If ``x`` and ``y`` have different lengths.

    Examples:
        >>> import numpy as np
        >>> # Unit square traversed [0,1,1,0]/[0,0,1,1]: y-down area = -1.0.
        >>> float(compute_signed_area(np.array([0.0, 1.0, 1.0, 0.0]),
        ...                           np.array([0.0, 0.0, 1.0, 1.0])))
        -1.0
        >>> # Fewer than 3 points → 0.0 (degenerate polygon).
        >>> compute_signed_area(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
        0.0
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) != len(y):
        raise ValueError(
            f"x and y must have the same length; got len(x)={len(x)} "
            f"len(y)={len(y)}"
        )
    if len(x) < 3:
        return 0.0
    # Explicit non-finite guard so the NaN contract is deterministic regardless
    # of float-arithmetic happenstance (an unguarded Shoelace sum over ±inf can
    # evaluate to ±inf rather than NaN). Mirrors the explicit finite validation
    # in project_to_growth_axis_perpendicular.
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        return float("nan")
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    # y-down-corrected (negated) Shoelace — see docstring; positive area ↔
    # handedness +1 (ψ_g increasing) under the atan2(dx, dy) image-y-down
    # convention compute_psi_g encodes.
    return 0.5 * float(np.sum(x_next * y - x * y_next))


def compute_path_curvature(
    x_dot: np.ndarray,
    y_dot: np.ndarray,
    x_ddot: np.ndarray,
    y_ddot: np.ndarray,
) -> np.ndarray:
    r"""Per-frame trajectory curvature ``κ`` from velocity/acceleration (PR #8, Tier 3a).

    Computes the standard differential-geometry curvature (theory.md §6.2):

    .. math:: \kappa = \frac{\dot{x}\,\ddot{y} - \dot{y}\,\ddot{x}}{(\dot{x}^2 + \dot{y}^2)^{3/2}}

    in inverse pixels (px⁻¹). Used by ``midline.reconstruct`` (Tier 3a), which
    feeds it the Savitzky-Golay analytic derivatives of the smoothed tip
    coordinates (``_noise.compute_sg_derivative`` with ``deriv=1`` and
    ``deriv=2``).

    **Sign convention (load-bearing — anchored on the FORMULA sign, NOT the
    frame-ambiguous word "left turn").** This is the literal standard y-up math
    curvature formula. Anchored (like :func:`compute_signed_area`'s
    ``[0,1,1,0]/[0,0,1,1] → −1.0`` anchor) by the absolute hand-built input
    ``compute_path_curvature([1],[0],[0],[1]) == +1.0`` (unit velocity ``+x``,
    unit acceleration ``+y``). A counterclockwise (y-up math) circle of radius
    ``R`` gives ``κ = +1/R``; a clockwise circle gives ``−1/R``. theory.md §6.2
    labels ``κ > 0`` a "left turn" — that is the standard y-up math convention;
    in the **y-down image frame** the pipeline runs in, ``+κ`` is a
    clockwise / visual-right turn as displayed (so we anchor on the sign, not
    the word — the same discipline :func:`compute_signed_area` uses).

    **Cross-helper sign relationship (publication-trait-inversion guard).**
    Because the ψ_g family (:func:`compute_psi_g`, :func:`compute_signed_area`,
    the ``handedness`` trait) uses the deliberately swapped ``atan2(dx, dy)``
    argument order, the exact per-frame identity is ``dψ_g/dt = −κ·|v|``, so
    ``sign(dψ_g/dt) = −sign(κ)`` frame-by-frame wherever ``|v| > 0``. For a loop
    traversed with a SINGLE sense of rotation (single-signed ``κ`` — a
    circle/ellipse/arc) this collapses to the scalar ``sign(κ) == −handedness``
    (e.g. a y-up-math-CCW circle gives ``κ = +1/R`` but ``handedness = −1``).
    A consumer composing curvature chirality with ``handedness`` (PR #9/#10)
    MUST account for this opposite polarity.

    Args:
        x_dot: 1-D array of ẋ (first derivative of x).
        y_dot: 1-D array of ẏ (same length as ``x_dot``).
        x_ddot: 1-D array of ẍ (second derivative of x; same length).
        y_ddot: 1-D array of ÿ (same length).

    Returns:
        Length-``len(x_dot)`` 1-D ``float64`` array of curvature in px⁻¹.
        Frames where ``|v| = √(ẋ² + ẏ²) = 0`` (the curvature denominator
        vanishes) are set to ``NaN``. The computation is guarded by
        ``np.errstate(divide, invalid, over)`` so no ``np.RuntimeWarning`` is
        emitted and no ``±inf`` is returned at the exact-zero denominator; the
        caller (``midline.reconstruct``) additionally sweeps any non-finite
        curvature at the near-zero-denominator corner to ``NaN``.

    Raises:
        ValueError: If the four input arrays do not all have the same length.
    """
    x_dot = np.asarray(x_dot, dtype=np.float64)
    y_dot = np.asarray(y_dot, dtype=np.float64)
    x_ddot = np.asarray(x_ddot, dtype=np.float64)
    y_ddot = np.asarray(y_ddot, dtype=np.float64)
    if not (len(x_dot) == len(y_dot) == len(x_ddot) == len(y_ddot)):
        raise ValueError(
            f"x_dot, y_dot, x_ddot, y_ddot must have the same length; got "
            f"{len(x_dot)}, {len(y_dot)}, {len(x_ddot)}, {len(y_ddot)}"
        )
    denom = (x_dot**2 + y_dot**2) ** 1.5
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        kappa = (x_dot * y_ddot - y_dot * x_ddot) / denom
    kappa = np.asarray(kappa, dtype=np.float64)
    # Exact zero-velocity frames (vanishing denominator) → NaN, never ±inf.
    kappa[denom == 0.0] = np.nan
    return kappa
