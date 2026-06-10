"""Tier 3b — spatial CWT machinery + ``λ_spatial`` ridge (PR #9).

Resamples the PR #8 midline curvature ``κ(s)`` (emitted on the native
NON-uniform ``arc_length_px`` grid) onto a uniform-spacing grid
(:func:`resample_curvature`), computes a spatial CWT of ``κ(s)`` using the
second-derivative complex Gaussian ``cgau2`` mother wavelet by default
(:func:`compute_scaleogram`), and extracts the per-position dominant spatial
wavelength ``λ(s_a)`` ridge (:func:`extract_ridge`).

This is a MACHINERY PR mirroring PR #5 ``temporal_cwt`` (input-validating, frozen
``attrs`` Results, a determinism contract — and NO trait emission). The
``L_gz``/``L_c`` growth-zone-structure traits of theory §7.4 are DESCOPED (they do
not transfer to top-view tip-trail ``κ(s)``; see GitHub issue #230); PR #9 ships the
spatial scaleogram + ``λ_spatial``, which is what the steady-traveling-wave data
supports and what PR #10's ``traveling_wave_residual`` consumes.
"""

import logging
import math
from typing import Any, Optional

import attrs
import numpy as np

from sleap_roots.circumnutation._constants import ConstantsT


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# §1 — Frozen result containers
# ---------------------------------------------------------------------------


@attrs.define(frozen=True, slots=False, kw_only=True, eq=False)
class ResampleResult:
    """Uniform-grid resample of midline curvature κ(s) (PR #9).

    ``eq=False`` because ndarray fields make the auto-generated ``__eq__``
    ambiguous (``arr1 == arr2`` is an array, not a bool).

    Attributes:
        kappa_uniform: float64 curvature (px⁻¹) on the uniform apex-origin grid.
        s_a_uniform_px: float64 apex-origin arc length (px); ``s_a_uniform_px[0]``
            is ``0.0`` at the apex (largest surviving ``arc_length_px``).
        ds: Uniform grid spacing (px), the median of positive surviving Δs_a.
        n_unmasked: Number of finite, unmasked surviving samples.
        arc_span_px: Surviving arc-length span (px), ``max − min`` over survivors.
        is_degenerate: ``True`` when the resample is void (all-NaN output).
    """

    kappa_uniform: np.ndarray
    s_a_uniform_px: np.ndarray
    ds: float
    n_unmasked: int
    arc_span_px: float
    is_degenerate: bool


# ---------------------------------------------------------------------------
# §2 — Validators (field-named; run first, unconditionally)
# ---------------------------------------------------------------------------


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


def _validate_1d_float(value: Any, name: str) -> np.ndarray:
    """Validate ``value`` is a 1-D numeric ndarray; return a float64 view/copy.

    Rejects non-ndarray (TypeError) and non-1-D / complex / object dtype
    (ValueError). Non-finite VALUES are NOT rejected here — they are dropped
    downstream (curvature legitimately carries NaN from a degenerate midline).
    """
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy ndarray, got {type(value).__name__}")
    if value.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got ndim={value.ndim}")
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real-valued, got complex dtype {value.dtype}")
    if value.dtype == object:
        raise ValueError(f"{name} must be numeric, got object dtype")
    try:
        return np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError(f"{name} must be float64-coercible: {exc}") from exc


def _min_samples_required(constants: ConstantsT) -> int:
    """Spatial MIN floor: floor(WL_MIN_NYQUIST / WL_MAX_FRACTION) + 1 (= 9 default)."""
    return (
        int(
            math.floor(
                constants.CWT_WAVELENGTH_MIN_NYQUIST_FACTOR
                / constants.CWT_WAVELENGTH_MAX_SIGNAL_FRACTION
            )
        )
        + 1
    )


def _degenerate_resample(n_unmasked: int) -> ResampleResult:
    """Graceful all-NaN ResampleResult (never raises, never RuntimeWarning)."""
    empty = np.empty(0, dtype=np.float64)
    return ResampleResult(
        kappa_uniform=empty.copy(),
        s_a_uniform_px=empty.copy(),
        ds=float("nan"),
        n_unmasked=int(n_unmasked),
        arc_span_px=float("nan"),
        is_degenerate=True,
    )


# ---------------------------------------------------------------------------
# §3 — Public: resample_curvature
# ---------------------------------------------------------------------------


def resample_curvature(
    curvature_px_inv: np.ndarray,
    arc_length_px: np.ndarray,
    velocity_sub_noise_mask: Optional[np.ndarray] = None,
    constants: Optional[ConstantsT] = None,
) -> ResampleResult:
    """Resample non-uniform midline κ(s) onto a uniform apex-origin grid.

    Drops sub-noise-masked and non-finite frames, reparameterizes to the
    apex-origin axis ``s_a = max(surviving arc) − arc`` (apex at ``s_a = 0`` per
    theory §6.5), chooses ``ds`` = median of positive surviving Δs_a, deduplicates
    duplicate-``s_a`` knots by averaging κ (so ``np.interp`` sees a strictly
    increasing xp), and interpolates onto a uniform grid of exactly
    ``floor(arc_span/ds) + 1`` points.

    Args:
        curvature_px_inv: 1-D float64 curvature (px⁻¹); may contain NaN.
        arc_length_px: 1-D float64 native arc length (px), monotonic non-decreasing.
        velocity_sub_noise_mask: Optional 1-D bool-coercible mask; ``True`` frames
            are excluded (the PR #8 sub-noise polarity).
        constants: Optional :class:`ConstantsT` override-bag.

    Returns:
        A :class:`ResampleResult`. Degenerate inputs (too few survivors, or an
        output grid shorter than the MIN floor) yield a graceful all-NaN result
        with ``is_degenerate=True`` — never raising, never ``RuntimeWarning``.

    Raises:
        TypeError: ``curvature_px_inv``/``arc_length_px`` not ndarrays, bad
            ``velocity_sub_noise_mask`` type, or bad ``constants`` type.
        ValueError: non-1-D / complex / object dtype, length mismatch, or a
            ``velocity_sub_noise_mask`` of wrong length.
    """
    _c = _check_constants(constants) or ConstantsT()
    curvature = _validate_1d_float(curvature_px_inv, "curvature_px_inv")
    arc = _validate_1d_float(arc_length_px, "arc_length_px")
    if curvature.shape[0] != arc.shape[0]:
        raise ValueError(
            f"curvature_px_inv and arc_length_px must have equal length, "
            f"got {curvature.shape[0]} and {arc.shape[0]}"
        )

    keep = np.isfinite(curvature) & np.isfinite(arc)
    if velocity_sub_noise_mask is not None:
        if not isinstance(velocity_sub_noise_mask, np.ndarray):
            raise TypeError(
                "velocity_sub_noise_mask must be a numpy ndarray, got "
                f"{type(velocity_sub_noise_mask).__name__}"
            )
        if velocity_sub_noise_mask.shape[0] != arc.shape[0]:
            raise ValueError(
                "velocity_sub_noise_mask must match the input length, got "
                f"{velocity_sub_noise_mask.shape[0]} vs {arc.shape[0]}"
            )
        keep &= ~velocity_sub_noise_mask.astype(bool)

    n_unmasked = int(keep.sum())
    min_samples = _min_samples_required(_c)

    arc_s = arc[keep]
    kappa_s = curvature[keep]

    n_input = curvature.shape[0]

    # Stage 1 gate (count + span ONLY; before max/min/median).
    if n_unmasked < min_samples:
        logger.debug(
            "resample_curvature(n_input=%d, n_unmasked=%d, ds=nan, arc_span_px=nan): "
            "degenerate (n_unmasked < min_samples=%d)",
            n_input,
            n_unmasked,
            min_samples,
        )
        return _degenerate_resample(n_unmasked)
    arc_span = float(arc_s.max() - arc_s.min())
    if not (arc_span > 0.0):
        logger.debug(
            "resample_curvature(n_input=%d, n_unmasked=%d, ds=nan, arc_span_px=%.6f): "
            "degenerate (non-positive arc-span)",
            n_input,
            n_unmasked,
            arc_span,
        )
        return _degenerate_resample(n_unmasked)

    # Apex-origin axis (apex at s_a = 0 = largest surviving arc).
    s_a = arc_s.max() - arc_s
    # Deduplicate duplicate-s_a knots by averaging κ -> strictly-increasing xp.
    s_a_unique, inverse = np.unique(s_a, return_inverse=True)
    counts = np.bincount(inverse)
    kappa_unique = np.bincount(inverse, weights=kappa_s) / counts

    ds = float(np.median(np.diff(s_a_unique)))

    # Stage 2 gate (output grid length).
    grid_len = int(math.floor(arc_span / ds)) + 1
    if grid_len < min_samples:
        logger.debug(
            "resample_curvature(n_input=%d, n_unmasked=%d, ds=%.6f, arc_span_px=%.6f): "
            "degenerate (grid_len=%d < min_samples=%d)",
            n_input,
            n_unmasked,
            ds,
            arc_span,
            grid_len,
            min_samples,
        )
        return _degenerate_resample(n_unmasked)

    grid = np.arange(grid_len, dtype=np.float64) * ds
    kappa_uniform = np.interp(grid, s_a_unique, kappa_unique)

    logger.debug(
        "resample_curvature(n_input=%d, n_unmasked=%d, ds=%.6f, arc_span_px=%.6f)",
        n_input,
        n_unmasked,
        ds,
        arc_span,
    )
    return ResampleResult(
        kappa_uniform=kappa_uniform,
        s_a_uniform_px=grid,
        ds=ds,
        n_unmasked=n_unmasked,
        arc_span_px=arc_span,
        is_degenerate=False,
    )


# ---------------------------------------------------------------------------
# Stub (graduated in PR #9 §5 atomic commit)
# ---------------------------------------------------------------------------


def compute_scaleogram(kappa=None, ds=None, constants=None):
    """Compute a spatial CWT scaleogram of ``κ(s)`` (PR #9 §5 will implement).

    Args:
        kappa: 1-D numpy array of uniform-grid curvature values.
        ds: Arc-length spacing between samples (px).
        constants: Optional :class:`ConstantsT` override-bag.

    Returns:
        A ``SpatialScaleogramResult`` (when implemented).

    Raises:
        NotImplementedError: Until the PR #9 §5 atomic graduation commit.
    """
    raise NotImplementedError("PR #9 — see docs/circumnutation/roadmap.md")
