"""Tier 3a — tip-trail-as-midline reconstruction (PR #8).

Implements the tip-trail-as-midline identity (theory.md §6.1): for an
apically-growing organ where tissue past the elongation zone does not
reshape, the curve formed by past tip positions IS the organ midline.
:func:`reconstruct` returns that midline parameterized by arc length
``s(τ) = ∫|v|dσ``, plus per-frame trajectory curvature ``κ`` (theory.md
§6.2), tip speed, and a velocity-bandpass mask flagging sub-noise frames.

Machinery PR mirroring PR #5 ``temporal_cwt``: input-validating, a frozen
``attrs`` Result class, a determinism contract — and **no trait emission**.
Savitzky-Golay smoothing/differentiation come from one fitted polynomial
per coordinate (``_noise.compute_sg_derivative``) so smoothing happens
BEFORE the second-derivative operations the curvature needs (§6.2). The
reconstruction is **cadence-independent** (frame-parameterized; velocity in
px/frame, the program convention — ``px/s`` is deliberately not used).

Scope boundaries: the ``L_gz`` growth-zone mask is **not** built here (CC-1;
PR #10), and ``κ(s)`` is emitted on the native NON-uniform ``arc_length_px``
grid — PR #9 owns the resample to the uniform ``(kappa, ds)`` grid its
spatial CWT needs. ``velocity_sub_noise_mask`` is a per-FRAME time-domain
mask and is NOT the per-arc-length ``L_gz`` mask.
"""

import logging
from typing import Optional

import attrs
import numpy as np
import scipy.integrate

from sleap_roots.circumnutation._constants import ConstantsT
from sleap_roots.circumnutation._geometry import compute_path_curvature
from sleap_roots.circumnutation._noise import compute_sg_derivative


logger = logging.getLogger(__name__)


@attrs.define(frozen=True, slots=False, kw_only=True, eq=False)
class MidlineResult:
    """Output of :func:`reconstruct`. Frozen container of the reconstructed midline.

    Frozen (``frozen=True`` prevents rebinding the attribute but does NOT
    deep-freeze the underlying numpy arrays), mirroring the foundation's
    :class:`~sleap_roots.circumnutation.temporal_cwt.ScaleogramResult` /
    :class:`~sleap_roots.circumnutation.temporal_cwt.RidgeResult` precedent.
    Unlike those, ``MidlineResult`` adds ``eq=False`` (with ndarray fields the
    generated ``__eq__`` on ``r1 == r2`` is ambiguous and can raise
    ``ValueError: ambiguous truth value``; ``eq=False`` makes ``==`` identity
    and forces field-by-field ``np.array_equal`` comparison).

    All per-frame arrays have length ``n = len(x)`` and are indexed by frame
    (index ``i`` ↔ frame ``i``). The reconstruction is **cadence-independent**:
    velocity is in px/frame and arc length integrates over the frame index, so
    ``cadence_s`` does not affect ``arc_length_px`` / ``curvature_px_inv`` /
    ``velocity_sub_noise_mask`` (it is stored as provenance only).

    Attributes:
        frame_indices: ``np.arange(n)``, dtype ``int64``.
        x_smooth_px: SG deriv=0 of ``x`` (px), dtype ``float64``.
        y_smooth_px: SG deriv=0 of ``y`` (px), dtype ``float64``.
        speed_px_per_frame: ``√(ẋ² + ẏ²)`` from SG deriv=1 (px/frame), ``float64``.
        arc_length_px: ``cumulative_trapezoid(speed, dx=1, initial=0)`` (px),
            ``float64``; ``arc_length_px[0] == 0.0``, monotonic non-decreasing.
        curvature_px_inv: ``κ`` (px⁻¹) via
            :func:`~sleap_roots.circumnutation._geometry.compute_path_curvature`;
            ``float64``. ONE array — curvature is parameterization-invariant, so
            the time-domain ``κ_path(τ)`` and arc-length ``κ(s(τ))`` are
            bit-identical (theory.md §6.1): pair with ``frame_indices`` for the
            time view or with ``arc_length_px`` (non-uniformly sampled) for the
            arc view. Non-finite entries at the curvature blow-up corner are
            swept to ``NaN``.
        velocity_sub_noise_mask: ``True ⇔ speed_px_per_frame ≤ NOISE_MASK_K · σ_v``
            (theory.md §6.2 sub-noise guard), dtype ``bool``. ``True`` flags a
            sub-noise frame to EXCLUDE before curvature use (consumers use
            ``curvature_px_inv[~velocity_sub_noise_mask]``), matching the
            ``coi_mask`` True=unreliable polarity. This is a per-FRAME mask and
            is NOT the ``L_gz`` growth-zone mask (PR #10).
        cadence_s: Resolved cadence in seconds (provenance; not used in the
            cadence-independent core).
        sg_window: Resolved Savitzky-Golay window length actually used.
        sg_degree: Resolved Savitzky-Golay polynomial degree (``SG_DEGREE``).
        sigma_v_px_per_frame: ``np.std(speed_px_per_frame, ddof=0)`` used for the
            mask (``NaN`` on the degenerate path).
        noise_mask_k: ``NOISE_MASK_K`` used for the mask.
        is_degenerate: ``True`` when the reconstruction is void (SG could not be
            applied: ``n < sg_window`` / ``n == 0``, or a raw-stationary track)
            and the per-frame float arrays are all ``NaN``.
    """

    frame_indices: np.ndarray
    x_smooth_px: np.ndarray
    y_smooth_px: np.ndarray
    speed_px_per_frame: np.ndarray
    arc_length_px: np.ndarray
    curvature_px_inv: np.ndarray
    velocity_sub_noise_mask: np.ndarray
    cadence_s: float
    sg_window: int
    sg_degree: int
    sigma_v_px_per_frame: float
    noise_mask_k: float
    is_degenerate: bool


def reconstruct(
    x: np.ndarray,
    y: np.ndarray,
    cadence_s: float,
    sg_window: Optional[int] = None,
    constants: Optional[ConstantsT] = None,
) -> MidlineResult:
    """Reconstruct the tip-trail midline (arc length, curvature, speed, mask).

    Args:
        x: 1-D ``np.ndarray`` of tip x-coordinates (finite real values).
        y: 1-D ``np.ndarray`` of tip y-coordinates (same length as ``x``).
        cadence_s: Sampling cadence in seconds (validated and stored as
            provenance; the reconstruction itself is cadence-independent).
        sg_window: Savitzky-Golay window length. ``None`` →
            ``constants.SG_WINDOW_SHORT`` (default 5). The SG polynomial degree
            is ``constants.SG_DEGREE`` (default 3) and is not a parameter.
        constants: Optional :class:`ConstantsT` override-bag. ``None`` →
            ``ConstantsT()``.

    Returns:
        A frozen :class:`MidlineResult`.
    """
    _c = constants if constants is not None else ConstantsT()
    window = int(sg_window) if sg_window is not None else int(_c.SG_WINDOW_SHORT)
    degree = int(_c.SG_DEGREE)

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)

    x_smooth = compute_sg_derivative(x, window, degree, deriv=0)
    y_smooth = compute_sg_derivative(y, window, degree, deriv=0)
    x_dot = compute_sg_derivative(x, window, degree, deriv=1, delta=1.0)
    y_dot = compute_sg_derivative(y, window, degree, deriv=1, delta=1.0)
    x_ddot = compute_sg_derivative(x, window, degree, deriv=2, delta=1.0)
    y_ddot = compute_sg_derivative(y, window, degree, deriv=2, delta=1.0)

    speed = np.hypot(x_dot, y_dot)
    arc_length = scipy.integrate.cumulative_trapezoid(speed, dx=1.0, initial=0.0)

    curvature = compute_path_curvature(x_dot, y_dot, x_ddot, y_ddot)
    curvature[~np.isfinite(curvature)] = np.nan

    sigma_v = float(np.std(speed, ddof=0))
    noise_mask_k = float(_c.NOISE_MASK_K)
    mask = speed <= noise_mask_k * sigma_v

    logger.debug(
        "midline.reconstruct(n_frames=%d, sg_window=%d, sg_degree=%d, cadence_s=%.6f)",
        n,
        window,
        degree,
        float(cadence_s),
    )

    return MidlineResult(
        frame_indices=np.arange(n, dtype=np.int64),
        x_smooth_px=x_smooth.astype(np.float64, copy=False),
        y_smooth_px=y_smooth.astype(np.float64, copy=False),
        speed_px_per_frame=speed.astype(np.float64, copy=False),
        arc_length_px=arc_length.astype(np.float64, copy=False),
        curvature_px_inv=curvature.astype(np.float64, copy=False),
        velocity_sub_noise_mask=mask.astype(bool, copy=False),
        cadence_s=float(cadence_s),
        sg_window=window,
        sg_degree=degree,
        sigma_v_px_per_frame=sigma_v,
        noise_mask_k=noise_mask_k,
        is_degenerate=False,
    )
