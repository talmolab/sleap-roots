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
import math
from typing import Any, Optional

import attrs
import numpy as np
import scipy.integrate

from sleap_roots.circumnutation._constants import ConstantsT
from sleap_roots.circumnutation._geometry import compute_path_curvature
from sleap_roots.circumnutation._noise import (
    _validate_sg_window_polyorder,
    compute_sg_derivative,
)
from sleap_roots.circumnutation.temporal_cwt import _validate_cadence_s


logger = logging.getLogger(__name__)


def _check_constants(value: Any) -> Optional[ConstantsT]:
    """Validate ``constants`` is ``None`` or a :class:`ConstantsT` instance."""
    if value is None:
        return None
    if not isinstance(value, ConstantsT):
        raise TypeError(
            f"constants must be None or a ConstantsT instance, "
            f"got {type(value).__name__}: {value!r}"
        )
    return value


def _validate_midline_constants(constants: ConstantsT) -> None:
    """Validate the ``ConstantsT`` fields ``reconstruct`` consumes (field-named).

    Mirrors the sibling tier modules (``nutation`` / ``psi_g`` / ``temporal_cwt``)
    which validate the specific constants they consume at the boundary so an
    invalid override fails fast with a field-named ``ValueError`` rather than a
    silent coercion (``int()`` truncation) or a confusing downstream SciPy error.
    ``attrs`` does not type-check at construction, so this guards explicit
    overrides like ``ConstantsT(NOISE_MASK_K=-1)``.

    ``SG_WINDOW_SHORT`` is NOT validated here: it is consumed only when
    ``sg_window is None`` and is then validated by :func:`_resolve_sg_window`.

    Raises:
        ValueError: If ``NOISE_MASK_K`` is not a finite non-negative number, or
            ``SG_DEGREE`` is not a non-negative int.
    """
    k = constants.NOISE_MASK_K
    if isinstance(k, (bool, np.bool_)) or not isinstance(
        k, (int, float, np.integer, np.floating)
    ):
        raise ValueError(
            f"constants.NOISE_MASK_K must be a finite non-negative number, "
            f"got {type(k).__name__}: {k!r}"
        )
    if not math.isfinite(float(k)) or float(k) < 0:
        raise ValueError(f"constants.NOISE_MASK_K must be finite and >= 0, got {k!r}")
    degree = constants.SG_DEGREE
    if isinstance(degree, (bool, np.bool_)) or not isinstance(
        degree, (int, np.integer)
    ):
        raise ValueError(
            f"constants.SG_DEGREE must be a non-negative int (no float coercion), "
            f"got {type(degree).__name__}: {degree!r}"
        )
    if int(degree) < 0:
        raise ValueError(f"constants.SG_DEGREE must be >= 0, got {int(degree)}")


def _validate_xy(x: Any, y: Any) -> tuple:
    """Validate ``x``/``y`` are finite 1-D numeric ndarrays of equal length.

    Mirrors :func:`~sleap_roots.circumnutation.temporal_cwt._validate_x`: rejects
    non-ndarray (``TypeError``), non-1-D / complex / object / non-numeric dtype /
    non-finite (``ValueError``). Non-finite is REJECTED, not dropped, because the
    Savitzky-Golay differentiation and ``cumulative_trapezoid`` integration both
    assume uniform frame spacing (dropping NaN frames would silently break it).
    Returns the coerced ``(x_float64, y_float64)``.
    """
    for name, arr in (("x", x), ("y", y)):
        if not isinstance(arr, np.ndarray):
            raise TypeError(
                f"{name} must be a numpy ndarray, got {type(arr).__name__}: {arr!r}"
            )
        if arr.ndim != 1:
            raise ValueError(f"{name} must be a 1-D ndarray, got shape {arr.shape}")
        if np.issubdtype(arr.dtype, np.complexfloating):
            raise ValueError(
                f"{name} must have a real numeric dtype, got complex dtype: {arr.dtype}"
            )
        if arr.dtype == object or not np.issubdtype(arr.dtype, np.number):
            raise ValueError(f"{name} must have a numeric dtype, got dtype {arr.dtype}")
    x_float = x.astype(np.float64, copy=False)
    y_float = y.astype(np.float64, copy=False)
    # Per-array, count-reporting non-finite message (mirrors
    # temporal_cwt._validate_x) so a bad input is easy to diagnose: which array
    # (x vs y) and how many NaN / ±inf. Non-finite is REJECTED, not dropped,
    # because SG differentiation + cumulative_trapezoid assume uniform spacing.
    for name, arr in (("x", x_float), ("y", y_float)):
        if not np.isfinite(arr).all():
            n_nan = int(np.isnan(arr).sum())
            n_inf = int(np.isinf(arr).sum())
            raise ValueError(
                f"{name} must contain only finite values; found {n_nan} NaN(s) "
                f"and {n_inf} ±inf value(s). Non-finite frames are rejected, not "
                f"dropped (SG + arc-length integration assume uniform frame spacing)"
            )
    if len(x_float) != len(y_float):
        raise ValueError(
            f"x and y must have the same length; got len(x)={len(x_float)} "
            f"len(y)={len(y_float)}"
        )
    return x_float, y_float


def _resolve_sg_window(sg_window: Any, constants: ConstantsT) -> int:
    """Resolve + validate ``sg_window`` (positive odd int, ``> SG_DEGREE``).

    ``None`` → ``constants.SG_WINDOW_SHORT``. Validation reuses
    :func:`~sleap_roots.circumnutation._noise._validate_sg_window_polyorder`
    against ``constants.SG_DEGREE`` so an even / ``<= SG_DEGREE`` / non-int
    ``sg_window`` raises before any reconstruction work.
    """
    degree = int(constants.SG_DEGREE)
    if sg_window is None:
        sg_window = int(constants.SG_WINDOW_SHORT)
    if isinstance(sg_window, bool) or not isinstance(sg_window, (int, np.integer)):
        raise TypeError(
            f"sg_window must be a positive odd int (> SG_DEGREE), "
            f"got {type(sg_window).__name__}: {sg_window!r}"
        )
    # Reuse the shared SG window/order validator, but re-raise naming the PUBLIC
    # parameter `sg_window` so the midline API's field-named error contract holds
    # (the shared validator's message names "window"/"polynomial_order").
    try:
        window_int, _ = _validate_sg_window_polyorder(int(sg_window), degree)
    except ValueError as exc:
        raise ValueError(
            f"sg_window must be a positive odd int > SG_DEGREE ({degree}), "
            f"got sg_window={sg_window!r}: {exc}"
        ) from exc
    return window_int


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
    # ALL field-named validation runs first and unconditionally; the degenerate
    # gate (below) runs only on fully-valid inputs (so n==0-with-bad-cadence
    # RAISES rather than returning a graceful MidlineResult).
    _c = _check_constants(constants) or ConstantsT()
    _validate_midline_constants(_c)
    window = _resolve_sg_window(sg_window, _c)
    degree = int(_c.SG_DEGREE)
    cadence_s = _validate_cadence_s(cadence_s)
    x, y = _validate_xy(x, y)
    n = len(x)
    noise_mask_k = float(_c.NOISE_MASK_K)

    # Degenerate gate (graceful, no raise, no RuntimeWarning). Returns BEFORE any
    # np.std / np.hypot / cumulative_trapezoid call (np.std([]) warns;
    # cumulative_trapezoid([]) raises). `n == 0` MUST be the first short-circuit
    # disjunct — np.ptp([]) raises. Stationarity is detected on the RAW input
    # (post-SG speed is float dust, never exactly 0).
    if n == 0 or n < window or (np.ptp(x) == 0.0 and np.ptp(y) == 0.0):
        logger.debug(
            "midline.reconstruct(n_frames=%d, sg_window=%d): degenerate input, "
            "returning all-NaN MidlineResult",
            n,
            window,
        )
        nan_arr = np.full(n, np.nan, dtype=np.float64)
        return MidlineResult(
            frame_indices=np.arange(n, dtype=np.int64),
            x_smooth_px=nan_arr.copy(),
            y_smooth_px=nan_arr.copy(),
            speed_px_per_frame=nan_arr.copy(),
            arc_length_px=nan_arr.copy(),
            curvature_px_inv=nan_arr.copy(),
            velocity_sub_noise_mask=np.zeros(n, dtype=bool),
            cadence_s=float(cadence_s),
            sg_window=window,
            sg_degree=degree,
            sigma_v_px_per_frame=float("nan"),
            noise_mask_k=noise_mask_k,
            is_degenerate=True,
        )

    x_smooth = compute_sg_derivative(x, window, degree, deriv=0)
    y_smooth = compute_sg_derivative(y, window, degree, deriv=0)
    x_dot = compute_sg_derivative(x, window, degree, deriv=1, delta=1.0)
    y_dot = compute_sg_derivative(y, window, degree, deriv=1, delta=1.0)
    x_ddot = compute_sg_derivative(x, window, degree, deriv=2, delta=1.0)
    y_ddot = compute_sg_derivative(y, window, degree, deriv=2, delta=1.0)

    speed = np.hypot(x_dot, y_dot)
    arc_length = scipy.integrate.cumulative_trapezoid(speed, dx=1.0, initial=0.0)

    curvature = compute_path_curvature(x_dot, y_dot, x_ddot, y_ddot)
    # Defensive depth (intentionally redundant): compute_path_curvature already
    # sweeps non-finite → NaN internally, so this is a no-op today. Kept so a
    # future change to the helper's internal sweep cannot reintroduce ±inf into
    # MidlineResult.curvature_px_inv. Do NOT "DRY-delete" the HELPER's sweep —
    # direct callers (PR #9/#10 on a resampled κ(s) grid) rely on it.
    curvature[~np.isfinite(curvature)] = np.nan

    sigma_v = float(np.std(speed, ddof=0))
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
