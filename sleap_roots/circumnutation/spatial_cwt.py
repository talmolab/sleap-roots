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
import pywt

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


@attrs.define(frozen=True, slots=False, kw_only=True, eq=False)
class SpatialScaleogramResult:
    """cgau2 spatial CWT scaleogram of κ(s) (PR #9).

    ``eq=False`` (ndarray fields make auto ``__eq__`` ambiguous).

    Attributes:
        scaleogram: complex128 ``(n_scales, n_samples)`` CWT coefficients.
        scales: float64 ``(n_scales,)`` log-spaced dimensionless pywt scales.
        wavelengths_px: float64 ``(n_scales,)`` spatial wavelength axis (px), the
            honest ``pywt.scale2frequency`` convention value (NOT bias-corrected).
        spatial_freqs_px_inv: float64 ``(n_scales,)``, ``1.0 / wavelengths_px``.
        coi_mask: bool ``(n_scales, n_samples)``; ``True`` = inside-COI = unreliable.
        ds: Resolved uniform grid spacing (px).
        wavelet: Resolved wavelet name.
    """

    scaleogram: np.ndarray
    scales: np.ndarray
    wavelengths_px: np.ndarray
    spatial_freqs_px_inv: np.ndarray
    coi_mask: np.ndarray
    ds: float
    wavelet: str


@attrs.define(frozen=True, slots=False, kw_only=True, eq=False)
class SpatialRidgeResult:
    """Per-position dominant spatial wavelength λ(s_a) ridge (PR #9).

    Spatial sibling of ``temporal_cwt.RidgeResult`` (frame→position,
    period_s→wavelength_px). ``eq=False`` (ndarray fields).

    Attributes:
        position_indices: int64 ``(n_samples,)``, ``np.arange(n_samples)``.
        wavelengths_px: float64 ``(n_samples,)`` spatial wavelength AT the ridge.
        amplitudes: float64 ``(n_samples,)`` ``|W|`` at the ridge cell (≥ 0).
        powers: float64 ``(n_samples,)`` ``amplitudes ** 2`` (redundant by design).
        in_coi: bool ``(n_samples,)``; ``True`` iff the ridge cell is inside-COI.
    """

    position_indices: np.ndarray
    wavelengths_px: np.ndarray
    amplitudes: np.ndarray
    powers: np.ndarray
    in_coi: np.ndarray


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


def _validate_ds(ds: Any) -> float:
    """Validate ``ds`` is a positive finite float-like (not bool/str/list)."""
    if isinstance(ds, (bool, np.bool_)):
        raise TypeError(f"ds must be a real number, got bool {ds!r}")
    if not isinstance(ds, (int, float, np.integer, np.floating)):
        raise TypeError(f"ds must be int or float, got {type(ds).__name__}: {ds!r}")
    ds_f = float(ds)
    if not math.isfinite(ds_f) or ds_f <= 0.0:
        raise ValueError(f"ds must be positive and finite, got {ds!r}")
    return ds_f


def _validate_positive_finite(value: Any, name: str) -> None:
    """Raise ValueError (field-named) unless ``value`` is a positive finite number."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(
            f"constants.{name} must be a positive finite number, got bool {value!r}"
        )
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(
            f"constants.{name} must be a positive finite number, "
            f"got {type(value).__name__}: {value!r}"
        )
    v = float(value)
    if not math.isfinite(v) or v <= 0.0:
        raise ValueError(
            f"constants.{name} must be a positive finite number, got {value!r}"
        )


def _validate_wavelength_factors(constants: ConstantsT) -> None:
    """Validate the two spatial wavelength-range factors (consumed by the MIN floor).

    Field-named guards (CC-1) so a bad ``ConstantsT`` override surfaces a clear
    ``ValueError`` naming the field rather than a cryptic ``ZeroDivisionError``
    from :func:`_min_samples_required`.
    """
    _validate_positive_finite(
        constants.CWT_WAVELENGTH_MIN_NYQUIST_FACTOR, "CWT_WAVELENGTH_MIN_NYQUIST_FACTOR"
    )
    _validate_positive_finite(
        constants.CWT_WAVELENGTH_MAX_SIGNAL_FRACTION,
        "CWT_WAVELENGTH_MAX_SIGNAL_FRACTION",
    )


def _validate_spatial_cwt_constants(constants: ConstantsT) -> None:
    """Validate every spatial-CWT constant ``compute_scaleogram`` consumes.

    Spatial sibling of ``temporal_cwt._validate_cwt_constants``; field-named
    ``ValueError`` per CC-1. Covers the two wavelength factors, the COI factor,
    ``CWT_SCALE_COUNT_DEFAULT`` (positive int), and ``WAVELET_DEFAULT_SPATIAL``
    (a non-empty string pywt recognizes).
    """
    _validate_wavelength_factors(constants)
    _validate_positive_finite(
        constants.SPATIAL_COI_EFOLDING_FACTOR, "SPATIAL_COI_EFOLDING_FACTOR"
    )
    sc = constants.CWT_SCALE_COUNT_DEFAULT
    if (
        isinstance(sc, (bool, np.bool_))
        or not isinstance(sc, (int, np.integer))
        or int(sc) < 1
    ):
        raise ValueError(
            f"constants.CWT_SCALE_COUNT_DEFAULT must be an int >= 1, got {sc!r}"
        )
    wavelet = constants.WAVELET_DEFAULT_SPATIAL
    if not isinstance(wavelet, str) or not wavelet:
        raise ValueError(
            f"constants.WAVELET_DEFAULT_SPATIAL must be a non-empty string, got {wavelet!r}"
        )
    try:
        pywt.scale2frequency(wavelet, 1.0)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"constants.WAVELET_DEFAULT_SPATIAL is not a recognized pywt wavelet: "
            f"{wavelet!r} ({exc})"
        ) from exc


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
    _validate_wavelength_factors(_c)
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
        if velocity_sub_noise_mask.ndim != 1:
            raise ValueError(
                "velocity_sub_noise_mask must be 1-D, got "
                f"ndim={velocity_sub_noise_mask.ndim}"
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

    # Numeric body under errstate as a belt-and-suspenders guarantee of the
    # never-RuntimeWarning contract (the two-stage gate above already makes every
    # op below well-defined: arc_span > 0 => >= 2 unique knots => non-empty
    # positive-diff median; counts >= 1 per unique value => no divide-by-zero).
    with np.errstate(invalid="ignore", divide="ignore"):
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
# §4 — Public: extract_ridge
# ---------------------------------------------------------------------------


def extract_ridge(
    scaleogram_result: SpatialScaleogramResult,
    constants: Optional[ConstantsT] = None,
) -> SpatialRidgeResult:
    """Extract the per-position dominant spatial wavelength ridge λ(s_a).

    Deterministic per-position argmax of ``|scaleogram|`` along the scale axis
    (numpy's documented smallest-index tie-break). Ships the raw ridge — NOT
    COI-masked (PR #10 applies the COI reliability gate).

    Args:
        scaleogram_result: A :class:`SpatialScaleogramResult`.
        constants: Optional :class:`ConstantsT` override-bag.

    Returns:
        A :class:`SpatialRidgeResult`.

    Raises:
        TypeError: ``scaleogram_result`` not a :class:`SpatialScaleogramResult`,
            or bad ``constants`` type.
        ValueError: empty scaleogram (``n_scales == 0`` or ``n_samples == 0``).
    """
    _check_constants(constants)
    if not isinstance(scaleogram_result, SpatialScaleogramResult):
        raise TypeError(
            "scaleogram_result must be a SpatialScaleogramResult, got "
            f"{type(scaleogram_result).__name__}"
        )
    scaleogram = scaleogram_result.scaleogram
    n_scales, n_samples = scaleogram.shape
    if n_scales == 0 or n_samples == 0:
        raise ValueError(
            f"scaleogram must be non-empty, got n_scales={n_scales}, "
            f"n_samples={n_samples}"
        )

    logger.debug("extract_ridge(n_scales=%d, n_samples=%d)", n_scales, n_samples)

    mag = np.abs(scaleogram)
    ridge_scale_idx = np.argmax(mag, axis=0).astype(np.int64)
    positions = np.arange(n_samples)
    amplitudes = mag[ridge_scale_idx, positions].astype(np.float64)
    wavelengths_px = scaleogram_result.wavelengths_px[ridge_scale_idx].astype(
        np.float64
    )
    in_coi = scaleogram_result.coi_mask[ridge_scale_idx, positions].astype(bool)

    return SpatialRidgeResult(
        position_indices=positions.astype(np.int64),
        wavelengths_px=wavelengths_px,
        amplitudes=amplitudes,
        powers=amplitudes**2,
        in_coi=in_coi,
    )


# ---------------------------------------------------------------------------
# §5 — private scale-axis + COI helpers (spatial siblings of temporal_cwt)
# ---------------------------------------------------------------------------


def _coi_boundary_samples(scale: float, coi_factor: float) -> int:
    """Return the COI boundary in integer samples: ``ceil(coi_factor * scale)``."""
    return int(math.ceil(coi_factor * scale))


def _make_coi_mask(scales: np.ndarray, n_samples: int, coi_factor: float) -> np.ndarray:
    """Build the boolean COI mask of shape ``(n_scales, n_samples)``.

    ``True`` flags inside-COI = unreliable cells near each edge (same polarity as
    ``temporal_cwt._make_coi_mask``).
    """
    coi_mask = np.zeros((len(scales), n_samples), dtype=bool)
    for i_scale, s in enumerate(scales):
        boundary = _coi_boundary_samples(float(s), coi_factor)
        left = min(boundary, n_samples)
        right_start = max(0, n_samples - boundary)
        coi_mask[i_scale, :left] = True
        coi_mask[i_scale, right_start:] = True
    return coi_mask


def _spatial_scale_axis(
    n_samples: int,
    ds: float,
    wavelet: str,
    scale_count: int,
    wl_min_factor: float,
    wl_max_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derive log-spaced scales + spatial-wavelength/frequency axes.

    Spatial sibling of ``temporal_cwt._log_spaced_scales`` (period→wavelength).
    The wavelength axis is the honest ``pywt.scale2frequency`` round-trip
    convention value (wavelet-agnostic; NOT bias-corrected — see the cgau2
    calibration note on :class:`SpatialScaleogramResult`). Returns
    ``(scales, wavelengths_px, spatial_freqs_px_inv)``.
    """
    center_freq = float(pywt.scale2frequency(wavelet, 1.0))
    wl_min_samples = wl_min_factor  # wavelength_min / ds
    wl_max_samples = wl_max_fraction * n_samples
    scale_min = wl_min_samples * center_freq
    scale_max = wl_max_samples * center_freq
    scales = np.logspace(
        math.log10(scale_min), math.log10(scale_max), num=scale_count
    ).astype(np.float64)
    freqs_normalized = np.asarray(
        pywt.scale2frequency(wavelet, scales), dtype=np.float64
    )
    spatial_freqs_px_inv = freqs_normalized / ds
    wavelengths_px = 1.0 / spatial_freqs_px_inv
    return scales, wavelengths_px, spatial_freqs_px_inv


# ---------------------------------------------------------------------------
# §6 — Public: compute_scaleogram (cgau2 spatial CWT)
# ---------------------------------------------------------------------------


def compute_scaleogram(
    kappa: np.ndarray,
    ds: float,
    constants: Optional[ConstantsT] = None,
) -> SpatialScaleogramResult:
    """Compute a cgau2 spatial CWT scaleogram of uniform-grid curvature ``κ(s)``.

    Spatial sibling of ``temporal_cwt.compute_scaleogram`` (``cgau2`` default,
    ``ds`` instead of ``cadence_s``). Unlike :func:`resample_curvature`, this
    REJECTS non-finite ``kappa`` (it expects the clean uniform grid the resample
    produced).

    Args:
        kappa: 1-D float64 uniform-grid curvature (px⁻¹), all-finite, length ≥
            ``MIN_SAMPLES_REQUIRED`` (= 9 at defaults).
        ds: Uniform grid spacing (px); positive finite.
        constants: Optional :class:`ConstantsT` override-bag.

    Returns:
        A :class:`SpatialScaleogramResult`.

    Raises:
        TypeError: ``kappa`` not an ndarray, bad ``ds`` type, or bad ``constants``.
        ValueError: ``kappa`` non-1-D / complex / object / non-finite / too short,
            or ``ds`` non-positive / non-finite.
    """
    _c = _check_constants(constants) or ConstantsT()
    _validate_spatial_cwt_constants(_c)
    kappa_v = _validate_1d_float(kappa, "kappa")
    if not np.isfinite(kappa_v).all():
        n_bad = int((~np.isfinite(kappa_v)).sum())
        raise ValueError(f"kappa must be all-finite, got {n_bad} non-finite value(s)")
    ds_v = _validate_ds(ds)
    min_samples = _min_samples_required(_c)
    if kappa_v.shape[0] < min_samples:
        raise ValueError(
            f"kappa too short: len={kappa_v.shape[0]} < MIN_SAMPLES_REQUIRED="
            f"{min_samples}"
        )

    wavelet = _c.WAVELET_DEFAULT_SPATIAL
    n_samples = kappa_v.shape[0]
    scales, wavelengths_px, spatial_freqs_px_inv = _spatial_scale_axis(
        n_samples,
        ds_v,
        wavelet,
        _c.CWT_SCALE_COUNT_DEFAULT,
        _c.CWT_WAVELENGTH_MIN_NYQUIST_FACTOR,
        _c.CWT_WAVELENGTH_MAX_SIGNAL_FRACTION,
    )

    logger.debug(
        "compute_scaleogram(n_samples=%d, ds=%.6f, n_scales=%d, "
        "wavelength_min_px=%.6f, wavelength_max_px=%.6f, wavelet=%r)",
        n_samples,
        ds_v,
        len(scales),
        float(wavelengths_px.min()),
        float(wavelengths_px.max()),
        wavelet,
    )

    coefs, _ = pywt.cwt(kappa_v, scales, wavelet)
    scaleogram = np.asarray(coefs, dtype=np.complex128)
    coi_mask = _make_coi_mask(scales, n_samples, _c.SPATIAL_COI_EFOLDING_FACTOR)

    return SpatialScaleogramResult(
        scaleogram=scaleogram,
        scales=scales,
        wavelengths_px=wavelengths_px,
        spatial_freqs_px_inv=spatial_freqs_px_inv,
        coi_mask=coi_mask,
        ds=ds_v,
        wavelet=wavelet,
    )
