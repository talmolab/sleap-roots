"""Tier 1 — temporal continuous wavelet transform machinery (PR #5).

Provides the generic CWT scaleogram + ridge-extraction primitives that
Tier 1 (PR #6, Derr-faithful trait emission), Tier 2 (PR #7, Bastien-Meroz
ψ_g), and the QC tier's ``coi_fraction_t1`` reliability gate compose on top
of. PR #5 emits **NO TRAITS** (scope discipline) — trait emission is PR #6's
responsibility per ``docs/circumnutation/theory.md`` §7.2.

Public API
==========

- :func:`compute_scaleogram` — full CWT scaleogram + COI mask
- :func:`extract_ridge` — per-frame ridge of period + amplitude + power
- :class:`ScaleogramResult` — frozen attrs container for ``compute_scaleogram`` output
- :class:`RidgeResult` — frozen attrs container for ``extract_ridge`` output

Wavelet, scale axis, COI formula
================================

Mother wavelet: ``cmor1.5-1.0`` (default per
:data:`~sleap_roots.circumnutation._constants.WAVELET_DEFAULT_TEMPORAL`) —
forensic match to Derr's Sept-2025 oracle scaleogram on the same Suyash
CMTN plate. Scales are log-spaced over an auto-derived period range
``[CWT_PERIOD_MIN_NYQUIST_FACTOR · cadence_s, CWT_PERIOD_MAX_SIGNAL_FRACTION
· n_frames · cadence_s]``; period-axis derivation uses
``pywt.scale2frequency`` for wavelet-agnostic correctness (the shortcut
``period = scale · cadence_s`` only holds at ``center_freq = 1.0``).

The cone-of-influence (COI) mask flags frames where the wavelet support
extends past a signal boundary, making the scaleogram unreliable. For
cmor1.5-1.0 the COI half-width is ``√1.5 · scale ≈ 1.225 · scale`` samples,
empirically verified across cmor variants (each gives factor ``√B``). See
``openspec/changes/add-circumnutation-temporal-cwt-machinery/design.md`` D3
for the full derivation.

Determinism contract (CC-6)
===========================

Same input → bit-identical scaleogram in the same Python process AND within
``atol=1e-9`` across Ubuntu / Windows / macOS CI runners (matching PR #4's
synthetic generator baseline). pywt.cwt is documented deterministic; tests
in §2.B of ``tests/test_circumnutation_temporal_cwt.py`` validate same-
process bit-identical (atol=0) and lock a 3-value canary at the resonant
scale against ``synthetic.generate_trajectory(random_state=0, n_frames=128,
T_nutation_s=3333, cadence_s=300, noise_sigma_px=0)`` input.

Scope discipline
================

No trait emission, no Layer-2 Derr forensic match (deferred to PR #6's
``derr_match_residual`` trait), no parabolic refinement of the ridge
(deferred to PR #6 if its ``T_nutation_iqr`` accuracy spec demands it), no
ridge-tracking continuity post-filter (planned follow-up GitHub issue —
per-frame argmax can hop discontinuously at multi-harmonic frames per
Mallat 1999 *A Wavelet Tour of Signal Processing* §4.4.2).
"""

import logging
import math
from typing import Any, Optional

import attrs
import numpy as np
import pywt

from sleap_roots.circumnutation._constants import (
    COI_EFOLDING_FACTOR,
    CWT_PERIOD_MAX_SIGNAL_FRACTION,
    CWT_PERIOD_MIN_NYQUIST_FACTOR,
    CWT_SCALE_COUNT_DEFAULT,
    ConstantsT,
    WAVELET_DEFAULT_TEMPORAL,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output dataclasses (design.md D1 + D2)
# ---------------------------------------------------------------------------


@attrs.define(frozen=True, slots=False, kw_only=True)
class ScaleogramResult:
    """Output of :func:`compute_scaleogram`. Frozen container of CWT outputs.

    All ndarray fields are immutable-by-convention (``frozen=True`` prevents
    rebinding the attribute but does NOT deep-freeze the underlying numpy
    arrays). The pattern mirrors the foundation's
    :class:`~sleap_roots.circumnutation._types.CircumnutationInputs` /
    :class:`~sleap_roots.circumnutation._constants.ConstantsT` precedent.

    Attributes:
        scaleogram: Complex-valued CWT coefficients of shape
            ``(n_scales, n_frames)``, dtype ``complex128``. ``|scaleogram|``
            is the magnitude envelope; ``np.angle(scaleogram)`` is the phase.
        scales: Pywt scale axis of shape ``(n_scales,)``, dtype ``float64``,
            strictly monotonically increasing (log-spaced by default).
        periods_s: Period axis in seconds of shape ``(n_scales,)``, dtype
            ``float64``. Derived via ``pywt.scale2frequency`` round-trip for
            wavelet-agnostic correctness.
        frequencies_hz: Frequency axis in Hz of shape ``(n_scales,)``, dtype
            ``float64``. Equal to ``1.0 / periods_s`` to within numerical
            precision.
        coi_mask: Cone-of-influence boolean mask of shape
            ``(n_scales, n_frames)``. ``True`` indicates inside-COI =
            unreliable (within ``COI_EFOLDING_FACTOR · scale`` samples of a
            signal boundary).
        cadence_s: The ``cadence_s`` value passed to ``compute_scaleogram``.
        wavelet: The resolved wavelet name (default
            :data:`WAVELET_DEFAULT_TEMPORAL` = ``"cmor1.5-1.0"``).
    """

    scaleogram: np.ndarray
    scales: np.ndarray
    periods_s: np.ndarray
    frequencies_hz: np.ndarray
    coi_mask: np.ndarray
    cadence_s: float
    wavelet: str


@attrs.define(frozen=True, slots=False, kw_only=True)
class RidgeResult:
    """Output of :func:`extract_ridge`. Per-frame ridge of the scaleogram.

    The ridge is NOT pre-COI-masked — PR #6's trait emission applies the
    mask per ``theory.md`` §7.2 "COI-masked" language. ``in_coi`` exposes
    per-frame COI status so downstream callers can filter as they wish.

    Note on field redundancy: ``powers = amplitudes ** 2`` by construction.
    Both are exposed so downstream consumers can use either ``|C|`` or
    ``|C|²`` without recomputing. The redundancy is intentional and locked
    by the spec's "Temporal CWT ridge API" requirement scenario.

    Attributes:
        frame_indices: Frame index axis of shape ``(n_frames,)``, dtype
            ``int64``. Always equal to ``np.arange(n_frames, dtype=np.int64)``.
        periods_s: Period at the argmax(|scaleogram|) cell for each frame.
            Shape ``(n_frames,)``, dtype ``float64``. **Indexed by frame, NOT
            by scale** — value at index ``i`` is the period at the ridge for
            frame ``i``.
        amplitudes: ``|C|`` at the ridge cell for each frame. Shape
            ``(n_frames,)``, dtype ``float64``, all values ≥ 0.
        powers: ``|C|² = amplitudes ** 2`` (redundant by construction).
            Shape ``(n_frames,)``, dtype ``float64``.
        in_coi: COI status at the ridge cell for each frame. Shape
            ``(n_frames,)``, dtype ``bool``. ``True`` means the ridge fell
            into the COI band at that frame.
    """

    frame_indices: np.ndarray
    periods_s: np.ndarray
    amplitudes: np.ndarray
    powers: np.ndarray
    in_coi: np.ndarray


# ---------------------------------------------------------------------------
# Validation helpers (design.md D8)
# ---------------------------------------------------------------------------


def _check_constants(value: Any) -> Optional[ConstantsT]:
    """Validate ``constants`` is ``None`` or a :class:`ConstantsT` instance.

    Named ``_check_constants`` for DRY consistency with
    :func:`sleap_roots.circumnutation.synthetic._check_constants` (PR #4
    precedent, /openspec-review round-1 Code-I2).
    """
    if value is None:
        return None
    if not isinstance(value, ConstantsT):
        raise TypeError(
            f"constants must be None or a ConstantsT instance, "
            f"got {type(value).__name__}: {value!r}"
        )
    return value


def _validate_cadence_s(cadence_s: Any) -> float:
    """Validate ``cadence_s`` is a positive finite float-like (not bool/str).

    Mirrors :func:`~sleap_roots.circumnutation.synthetic._check_float_finite`
    semantics: accept Python ``int``/``float``, accept numpy
    ``np.integer``/``np.floating`` scalars, explicitly reject ``bool``
    (Python) AND ``np.bool_`` (numpy scalar) by an isinstance check that
    PRECEDES the int/float check, reject ``str``/``list``/``complex``/tuple,
    require positive finite. Returns coerced ``float``.

    The ``np.bool_`` guard is load-bearing per /openspec-review round-1
    Code-I1 — ``np.bool_`` is a numpy scalar subclass of ``int`` on some
    numpy versions, so an int-check-only path would accept it silently.
    """
    if isinstance(cadence_s, (bool, np.bool_)):
        raise TypeError(f"cadence_s must be a finite float, got bool: {cadence_s!r}")
    if not isinstance(cadence_s, (int, float, np.floating, np.integer)):
        raise TypeError(
            f"cadence_s must be a float, got {type(cadence_s).__name__}: "
            f"{cadence_s!r}"
        )
    float_value = float(cadence_s)
    if not math.isfinite(float_value):
        raise ValueError(
            f"cadence_s must be a finite float (not NaN, +inf, or -inf), "
            f"got {float_value!r}"
        )
    if float_value <= 0:
        raise ValueError(f"cadence_s must be positive, got {float_value!r}")
    return float_value


def _derive_min_frames_required(constants: ConstantsT) -> int:
    """Derive the minimum ``len(x)`` required by ``compute_scaleogram``.

    Non-empty scale range requires ``period_max_s > period_min_s`` strictly,
    i.e., ``n > NYQUIST_FACTOR / SIGNAL_FRACTION``. The integer floor is
    ``int(floor(NYQUIST_FACTOR / SIGNAL_FRACTION)) + 1``. At defaults
    (``2.0 / 0.25 = 8.0``) this gives 9 frames.

    Positive-finite guards on both constants close the ``SIGNAL_FRACTION=0``
    ZeroDivisionError edge case (round-2 reviewer R2-N2). Raises
    ``ValueError`` with the named offending field per the CC-1 validation
    convention.
    """
    if (
        not math.isfinite(constants.CWT_PERIOD_MAX_SIGNAL_FRACTION)
        or constants.CWT_PERIOD_MAX_SIGNAL_FRACTION <= 0
    ):
        raise ValueError(
            f"constants.CWT_PERIOD_MAX_SIGNAL_FRACTION must be a positive finite "
            f"float, got {constants.CWT_PERIOD_MAX_SIGNAL_FRACTION!r}"
        )
    if (
        not math.isfinite(constants.CWT_PERIOD_MIN_NYQUIST_FACTOR)
        or constants.CWT_PERIOD_MIN_NYQUIST_FACTOR <= 0
    ):
        raise ValueError(
            f"constants.CWT_PERIOD_MIN_NYQUIST_FACTOR must be a positive finite "
            f"float, got {constants.CWT_PERIOD_MIN_NYQUIST_FACTOR!r}"
        )
    return (
        int(
            math.floor(
                constants.CWT_PERIOD_MIN_NYQUIST_FACTOR
                / constants.CWT_PERIOD_MAX_SIGNAL_FRACTION
            )
        )
        + 1
    )


def _validate_x(x: Any, min_frames_required: int) -> np.ndarray:
    """Validate ``x`` is a finite 1-D float64-coercible ndarray of length ≥ ``min_frames_required``.

    Coerces to ``np.float64`` for downstream pywt.cwt. Rejects complex
    dtype, object dtype, and any non-finite values (NaN, ±inf) per the
    strict design.md D8 contract.
    """
    if not isinstance(x, np.ndarray):
        raise TypeError(f"x must be a numpy ndarray, got {type(x).__name__}: {x!r}")
    if x.ndim != 1:
        raise ValueError(f"x must be a 1-D ndarray, got shape {x.shape}")
    if np.issubdtype(x.dtype, np.complexfloating):
        raise ValueError(
            f"x must have a real numeric dtype, got complex dtype: {x.dtype}"
        )
    if x.dtype == object:
        raise ValueError(f"x must have a numeric dtype, got object dtype")
    if not np.issubdtype(x.dtype, np.number):
        raise ValueError(f"x must have a numeric dtype, got dtype {x.dtype}")
    x_float = x.astype(np.float64, copy=False)
    if not np.isfinite(x_float).all():
        n_nan = int(np.isnan(x_float).sum())
        n_inf = int(np.isinf(x_float).sum())
        raise ValueError(
            f"x must contain only finite values; found {n_nan} NaN(s) and "
            f"{n_inf} ±inf value(s)"
        )
    if len(x_float) < min_frames_required:
        raise ValueError(
            f"x must have length >= MIN_FRAMES_REQUIRED = {min_frames_required} "
            f"(derived from constants.CWT_PERIOD_MIN_NYQUIST_FACTOR / "
            f"constants.CWT_PERIOD_MAX_SIGNAL_FRACTION + 1); got len(x) = {len(x_float)}"
        )
    return x_float


# ---------------------------------------------------------------------------
# COI mask helper (private but test-importable per design.md D3)
# ---------------------------------------------------------------------------


def _coi_boundary_samples(scale: float, coi_factor: float) -> int:
    """Return the COI boundary in integer samples for the given scale.

    COI half-width = ``ceil(coi_factor * scale)`` samples. Test and impl
    both import this helper so the integer expression is shared bit-exactly,
    eliminating floating-point-rounding-path ambiguity at the test/impl
    boundary (per TDD round-2 reviewer R2-I3).
    """
    return int(math.ceil(coi_factor * scale))


def _make_coi_mask(scales: np.ndarray, n_frames: int, coi_factor: float) -> np.ndarray:
    """Build the boolean COI mask of shape ``(n_scales, n_frames)``."""
    coi_mask = np.zeros((len(scales), n_frames), dtype=bool)
    for i_scale, s in enumerate(scales):
        boundary = _coi_boundary_samples(float(s), coi_factor)
        left = min(boundary, n_frames)
        right_start = max(0, n_frames - boundary)
        coi_mask[i_scale, :left] = True
        coi_mask[i_scale, right_start:] = True
    return coi_mask


# ---------------------------------------------------------------------------
# Scale-axis derivation (design.md D4)
# ---------------------------------------------------------------------------


def _log_spaced_scales(
    n_frames: int, cadence_s: float, constants: ConstantsT
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Derive log-spaced scales + period/frequency axes for the CWT.

    Returns ``(scales, periods_s, frequencies_hz, wavelet_name)``. Uses
    ``pywt.scale2frequency`` round-trip for wavelet-agnostic correctness
    (works under arbitrary ``WAVELET_DEFAULT_TEMPORAL`` overrides where
    ``center_freq != 1.0``).
    """
    wavelet_name = constants.WAVELET_DEFAULT_TEMPORAL
    period_min_s = constants.CWT_PERIOD_MIN_NYQUIST_FACTOR * cadence_s
    period_max_s = constants.CWT_PERIOD_MAX_SIGNAL_FRACTION * n_frames * cadence_s
    period_min_samples = period_min_s / cadence_s
    period_max_samples = period_max_s / cadence_s
    # pywt.scale2frequency(wavelet, scale) returns normalized frequency
    # (cycles per sample) = center_freq(wavelet) / scale. So
    # scale(period_samples) = period_samples · center_freq(wavelet).
    center_freq = float(pywt.scale2frequency(wavelet_name, 1.0))
    scale_min = period_min_samples * center_freq
    scale_max = period_max_samples * center_freq
    scales = np.logspace(
        math.log10(scale_min),
        math.log10(scale_max),
        num=constants.CWT_SCALE_COUNT_DEFAULT,
    ).astype(np.float64)
    freqs_normalized = pywt.scale2frequency(wavelet_name, scales)
    frequencies_hz = np.asarray(freqs_normalized, dtype=np.float64) / cadence_s
    periods_s = 1.0 / frequencies_hz
    return scales, periods_s, frequencies_hz, wavelet_name


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_scaleogram(
    x: np.ndarray,
    cadence_s: float,
    constants: Optional[ConstantsT] = None,
) -> ScaleogramResult:
    """Compute a temporal CWT scaleogram on a 1-D signal.

    Args:
        x: 1-D ``np.ndarray`` of finite real values (NaN / ±inf rejected).
            Must have length ≥ ``MIN_FRAMES_REQUIRED`` derived at call time
            from the resolved ``constants`` (= 9 at defaults). Coerced to
            ``np.float64`` for downstream pywt.cwt.
        cadence_s: Sampling cadence in seconds (positive finite float; bool
            and string types are explicitly rejected).
        constants: Optional :class:`ConstantsT` override-bag. 2-tier
            resolution per design.md D7: ``constants or ConstantsT()``.

    Returns:
        A frozen :class:`ScaleogramResult` containing the complex CWT
        coefficients, scale axis, period/frequency axes, boolean COI mask,
        and the resolved ``cadence_s`` + ``wavelet`` strings.

    Raises:
        ValueError: If ``x`` is non-finite / wrong shape / wrong dtype /
            too-short; if ``cadence_s`` has an invalid value (0, negative,
            NaN, ±inf); if a ``constants`` field violates its positive-finite
            invariant.
        TypeError: If ``x`` is not an ndarray; if ``cadence_s`` has an
            invalid type (bool, str, list); if ``constants`` is not ``None``
            and not a ``ConstantsT`` instance.

    Examples:
        Default cmor1.5-1.0 scaleogram on a 575-frame trajectory::

            >>> import numpy as np
            >>> from sleap_roots.circumnutation.temporal_cwt import compute_scaleogram
            >>> x = np.linspace(0.0, 100.0, 575, dtype=np.float64)
            >>> result = compute_scaleogram(x, 300.0)
            >>> result.scaleogram.shape
            (64, 575)
            >>> result.scaleogram.dtype
            dtype('complex128')
    """
    constants_resolved = _check_constants(constants)
    _c = constants_resolved if constants_resolved is not None else ConstantsT()
    cadence_s_v = _validate_cadence_s(cadence_s)
    min_frames_required = _derive_min_frames_required(_c)
    x_v = _validate_x(x, min_frames_required)
    n_frames = len(x_v)

    scales, periods_s, frequencies_hz, wavelet_name = _log_spaced_scales(
        n_frames, cadence_s_v, _c
    )

    logger.debug(
        f"compute_scaleogram(n_frames={n_frames}, cadence_s={cadence_s_v:.6f}, "
        f"n_scales={len(scales)}, period_min_s={float(periods_s.min()):.6f}, "
        f"period_max_s={float(periods_s.max()):.6f}, wavelet={wavelet_name!r})"
    )

    coefs, _ = pywt.cwt(x_v, scales, wavelet_name)
    scaleogram = np.asarray(coefs, dtype=np.complex128)

    coi_mask = _make_coi_mask(scales, n_frames, _c.COI_EFOLDING_FACTOR)

    return ScaleogramResult(
        scaleogram=scaleogram,
        scales=scales,
        periods_s=periods_s,
        frequencies_hz=frequencies_hz,
        coi_mask=coi_mask,
        cadence_s=cadence_s_v,
        wavelet=wavelet_name,
    )


def extract_ridge(
    scaleogram_result: ScaleogramResult,
    constants: Optional[ConstantsT] = None,
) -> RidgeResult:
    """Extract the per-frame ridge from a :class:`ScaleogramResult`.

    For each frame, the ridge is the scale index where ``|scaleogram|`` is
    maximized. Per-frame ``np.argmax`` is deterministic via numpy's
    documented tie-breaking (smallest index on equal values).

    Known limitation (Mallat 1999 §4.4.2): per-frame independence can
    produce spurious jitter at frames where two harmonics have similar
    amplitude. PR #6 may add a continuity post-filter (median-window or
    ridge-following) if its ``T_nutation_iqr`` accuracy spec demands it.

    Args:
        scaleogram_result: A :class:`ScaleogramResult` instance.
        constants: Optional :class:`ConstantsT` override-bag. Accepted for
            forward-compatibility (e.g., future parabolic-refinement
            threshold); currently unused but type-validated.

    Returns:
        A frozen :class:`RidgeResult` with per-frame ridge fields.

    Raises:
        TypeError: If ``scaleogram_result`` is not a ``ScaleogramResult``;
            if ``constants`` is not ``None`` and not a ``ConstantsT`` instance.
        ValueError: If ``scaleogram_result`` has ``n_scales == 0`` or
            ``n_frames == 0`` (empty scaleogram).
    """
    if not isinstance(scaleogram_result, ScaleogramResult):
        raise TypeError(
            f"extract_ridge requires a ScaleogramResult, got "
            f"{type(scaleogram_result).__name__}: {scaleogram_result!r}"
        )
    _check_constants(constants)

    n_scales, n_frames = scaleogram_result.scaleogram.shape
    if n_scales == 0:
        raise ValueError(
            "extract_ridge received an empty ScaleogramResult (n_scales == 0); "
            "cannot derive a ridge from an empty scale axis"
        )
    if n_frames == 0:
        raise ValueError(
            "extract_ridge received an empty ScaleogramResult (n_frames == 0); "
            "cannot derive a ridge from an empty frame axis"
        )

    logger.debug(f"extract_ridge(n_scales={n_scales}, n_frames={n_frames})")

    abs_scaleogram = np.abs(scaleogram_result.scaleogram)
    ridge_scale_idx = np.argmax(abs_scaleogram, axis=0).astype(np.int64)
    frame_arange = np.arange(n_frames, dtype=np.int64)

    periods_s = scaleogram_result.periods_s[ridge_scale_idx].astype(np.float64)
    amplitudes = abs_scaleogram[ridge_scale_idx, frame_arange].astype(np.float64)
    powers = amplitudes**2
    in_coi = scaleogram_result.coi_mask[ridge_scale_idx, frame_arange].astype(bool)

    return RidgeResult(
        frame_indices=frame_arange,
        periods_s=periods_s,
        amplitudes=amplitudes,
        powers=powers,
        in_coi=in_coi,
    )
