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
