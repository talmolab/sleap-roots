"""Tests for Tier 3a midline reconstruction (PR #8, add-circumnutation-tier3a-midline).

Covers the two new shared helpers (`_noise.compute_sg_derivative`,
`_geometry.compute_path_curvature`) and the public
`midline.reconstruct(...) -> MidlineResult` machinery, mirroring the PR #5
`temporal_cwt` machinery-test shape. See
`openspec/changes/add-circumnutation-tier3a-midline/specs/circumnutation/spec.md`
Requirement: Tier 3a midline reconstruction API.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# §1 — _noise.compute_sg_derivative (SG analytic derivatives, deriv 0/1/2)
# ---------------------------------------------------------------------------


def test_compute_sg_derivative_polynomial_exactness():
    """deriv 0/1/2 of a degree-≤order polynomial are recovered to machine precision.

    For x = 2t² + 3t + 1 with window=5, polynomial_order=3:
    deriv0 recovers x itself; deriv1 = 4t + 3; deriv2 = 4 (constant).
    """
    from sleap_roots.circumnutation._noise import compute_sg_derivative

    t = np.arange(11.0)
    x = 2.0 * t**2 + 3.0 * t + 1.0

    d0 = compute_sg_derivative(x, window=5, polynomial_order=3, deriv=0)
    d1 = compute_sg_derivative(x, window=5, polynomial_order=3, deriv=1, delta=1.0)
    d2 = compute_sg_derivative(x, window=5, polynomial_order=3, deriv=2, delta=1.0)

    np.testing.assert_allclose(d0, x, atol=1e-9)
    np.testing.assert_allclose(d1, 4.0 * t + 3.0, atol=1e-9)
    np.testing.assert_allclose(d2, np.full_like(t, 4.0), atol=1e-9)


def test_compute_sg_derivative_delta_scaling():
    """deriv=1 scales by 1/delta (the analytic derivative wrt the sample spacing)."""
    from sleap_roots.circumnutation._noise import compute_sg_derivative

    t = np.arange(11.0)
    x = 3.0 * t + 1.0  # slope 3 in sample-index units
    d1_unit = compute_sg_derivative(x, window=5, polynomial_order=2, deriv=1, delta=1.0)
    d1_half = compute_sg_derivative(x, window=5, polynomial_order=2, deriv=1, delta=2.0)

    np.testing.assert_allclose(d1_unit, np.full_like(t, 3.0), atol=1e-9)
    # delta=2 halves the per-unit derivative
    np.testing.assert_allclose(d1_half, np.full_like(t, 1.5), atol=1e-9)


def test_compute_sg_derivative_short_input_returns_all_nan():
    """len(x) < window returns a length-preserving all-NaN array (no raise/warning)."""
    from sleap_roots.circumnutation._noise import compute_sg_derivative

    x = np.arange(3.0)  # len 3 < window 5
    out = compute_sg_derivative(x, window=5, polynomial_order=3, deriv=1)
    assert out.shape == (3,)
    assert np.isnan(out).all()


@pytest.mark.parametrize(
    "window,polynomial_order,error_type,token",
    [
        (4, 3, ValueError, "window"),  # even window
        (3, 3, ValueError, "polynomial_order"),  # order >= window
        (5.0, 3, TypeError, "window"),  # non-int window
        (5, 2.0, TypeError, "polynomial_order"),  # non-int order
        (-5, 3, ValueError, "window"),  # non-positive window
    ],
)
def test_compute_sg_derivative_window_order_validation(
    window, polynomial_order, error_type, token
):
    """Reuses compute_sg_detrended's window/polynomial_order boundary validation."""
    from sleap_roots.circumnutation._noise import compute_sg_derivative

    x = np.zeros(20, dtype=np.float64)
    with pytest.raises(error_type, match=token):
        compute_sg_derivative(
            x, window=window, polynomial_order=polynomial_order, deriv=1
        )


def test_compute_sg_derivative_deriv_equal_order_is_accepted():
    """deriv == polynomial_order is the inclusive upper bound and is accepted."""
    from sleap_roots.circumnutation._noise import compute_sg_derivative

    t = np.arange(11.0)
    x = t**3
    # deriv=3 == order=3 → constant 6 (d³/dt³ of t³)
    d3 = compute_sg_derivative(x, window=5, polynomial_order=3, deriv=3, delta=1.0)
    np.testing.assert_allclose(d3, np.full_like(t, 6.0), atol=1e-9)


def test_compute_sg_derivative_deriv_above_order_raises_naming_deriv():
    """deriv > polynomial_order raises ValueError naming deriv (NOT scipy's silent zeros)."""
    from sleap_roots.circumnutation._noise import compute_sg_derivative

    x = np.zeros(20, dtype=np.float64)
    with pytest.raises(ValueError, match="deriv"):
        compute_sg_derivative(x, window=5, polynomial_order=3, deriv=4)


def test_compute_sg_derivative_negative_deriv_raises_naming_deriv():
    """deriv < 0 raises ValueError naming deriv (NOT scipy's opaque factorial error)."""
    from sleap_roots.circumnutation._noise import compute_sg_derivative

    x = np.zeros(20, dtype=np.float64)
    with pytest.raises(ValueError, match="deriv"):
        compute_sg_derivative(x, window=5, polynomial_order=3, deriv=-1)


# ---------------------------------------------------------------------------
# §2 — _geometry.compute_path_curvature (κ = (ẋÿ − ẏẍ)/|v|³, sign-anchored)
# ---------------------------------------------------------------------------


def _circle_derivs(radius, n, clockwise=False):
    """Analytic 1st/2nd derivatives of a circle of the given radius (n samples)."""
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    sign = -1.0 if clockwise else 1.0
    x_dot = -radius * np.sin(t)
    y_dot = sign * radius * np.cos(t)
    x_ddot = -radius * np.cos(t)
    y_ddot = -sign * radius * np.sin(t)
    return x_dot, y_dot, x_ddot, y_ddot


def test_compute_path_curvature_absolute_sign_anchor():
    """Hand-built anchor: unit velocity +x, unit acceleration +y → κ = +1.0 exactly.

    This pins the FORMULA sign (the standard y-up math curvature formula), not
    the frame-ambiguous word "left turn" — mirroring compute_signed_area's
    absolute anchor.
    """
    from sleap_roots.circumnutation._geometry import compute_path_curvature

    kappa = compute_path_curvature(
        np.array([1.0]), np.array([0.0]), np.array([0.0]), np.array([1.0])
    )
    assert kappa.shape == (1,)
    assert kappa[0] == 1.0


def test_compute_path_curvature_circle_recovers_inverse_radius():
    """CCW circle radius R → κ = +1/R; CW circle → −1/R."""
    from sleap_roots.circumnutation._geometry import compute_path_curvature

    R = 50.0
    kappa_ccw = compute_path_curvature(*_circle_derivs(R, 128, clockwise=False))
    kappa_cw = compute_path_curvature(*_circle_derivs(R, 128, clockwise=True))
    np.testing.assert_allclose(kappa_ccw, np.full(128, 1.0 / R), atol=1e-12)
    np.testing.assert_allclose(kappa_cw, np.full(128, -1.0 / R), atol=1e-12)


def test_compute_path_curvature_straight_line_is_zero():
    """Zero acceleration (straight line) → κ ≈ 0."""
    from sleap_roots.circumnutation._geometry import compute_path_curvature

    n = 10
    x_dot = np.full(n, 3.0)
    y_dot = np.full(n, 4.0)
    x_ddot = np.zeros(n)
    y_ddot = np.zeros(n)
    kappa = compute_path_curvature(x_dot, y_dot, x_ddot, y_ddot)
    np.testing.assert_allclose(kappa, np.zeros(n), atol=1e-12)


def test_compute_path_curvature_zero_velocity_is_nan_no_warning():
    """A constructed exact |v|=0 frame → NaN, with no RuntimeWarning."""
    import warnings

    from sleap_roots.circumnutation._geometry import compute_path_curvature

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        kappa = compute_path_curvature(
            np.array([0.0]), np.array([0.0]), np.array([1.0]), np.array([1.0])
        )
    assert np.isnan(kappa[0])


def test_compute_path_curvature_length_mismatch_raises():
    """Mismatched input lengths raise ValueError (sibling-helper guard)."""
    from sleap_roots.circumnutation._geometry import compute_path_curvature

    with pytest.raises(ValueError):
        compute_path_curvature(
            np.array([1.0, 2.0]), np.array([0.0]), np.array([0.0]), np.array([1.0])
        )


def test_compute_path_curvature_sign_is_opposite_handedness():
    """Cross-helper anchor (publication-trait-inversion guard): sign(κ) == −handedness.

    On a y-up-math CCW circle, the standard curvature formula gives κ = +1/R,
    while compute_psi_g (which uses the swapped atan2(dx, dy)) gives a NEGATIVE
    net rotation → handedness = −1. So sign(κ) == −handedness.
    """
    from sleap_roots.circumnutation._geometry import (
        compute_path_curvature,
        compute_psi_g,
    )

    n = 256
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x = np.cos(t)
    y = np.sin(t)
    kappa = compute_path_curvature(*_circle_derivs(1.0, n, clockwise=False))
    psi_g = compute_psi_g(x, y)
    handedness = int(np.sign(psi_g[-1] - psi_g[0]))

    assert int(np.sign(kappa[0])) == 1
    assert handedness == -1
    assert int(np.sign(kappa[0])) == -handedness


# ---------------------------------------------------------------------------
# §3 — MidlineResult + reconstruct (fields / dtypes / arc-length contract)
# ---------------------------------------------------------------------------


def _wobble_track(n=32, growth=1.0, amp=3.0, period=8.0):
    """A non-degenerate growth+lateral-wobble track (x growth, y sinusoid)."""
    t = np.arange(n, dtype=np.float64)
    x = growth * t
    y = amp * np.sin(2.0 * np.pi * t / period)
    return x, y


def test_reconstruct_returns_midline_result_with_fields_and_dtypes():
    """reconstruct returns a MidlineResult with the documented arrays + dtypes."""
    from sleap_roots.circumnutation.midline import MidlineResult, reconstruct

    x, y = _wobble_track(n=32)
    result = reconstruct(x, y, cadence_s=300.0)
    assert isinstance(result, MidlineResult)

    n = 32
    assert result.frame_indices.shape == (n,)
    assert result.frame_indices.dtype == np.int64
    np.testing.assert_array_equal(result.frame_indices, np.arange(n))
    for name in (
        "x_smooth_px",
        "y_smooth_px",
        "speed_px_per_frame",
        "arc_length_px",
        "curvature_px_inv",
    ):
        arr = getattr(result, name)
        assert arr.shape == (n,), name
        assert arr.dtype == np.float64, name
    assert result.velocity_sub_noise_mask.shape == (n,)
    assert result.velocity_sub_noise_mask.dtype == np.bool_


def test_reconstruct_provenance_scalars_take_default_values():
    """The resolved provenance scalars carry the documented defaults."""
    from sleap_roots.circumnutation.midline import reconstruct

    x, y = _wobble_track(n=32)
    result = reconstruct(x, y, cadence_s=300.0)
    assert result.cadence_s == 300.0
    assert result.sg_window == 5
    assert result.sg_degree == 3
    assert result.noise_mask_k == 2
    assert result.is_degenerate is False
    assert isinstance(result.sigma_v_px_per_frame, float)


def test_reconstruct_arc_length_starts_at_zero_and_is_monotonic():
    """arc_length_px[0] == 0 and the arc length is monotonic non-decreasing."""
    from sleap_roots.circumnutation.midline import reconstruct

    x, y = _wobble_track(n=40)
    result = reconstruct(x, y, cadence_s=300.0)
    assert result.arc_length_px[0] == 0.0
    assert np.all(np.diff(result.arc_length_px) >= -1e-12)


# ---------------------------------------------------------------------------
# §4 — input-validation boundary (raise, field-named; non-finite rejected)
# ---------------------------------------------------------------------------


def test_reconstruct_non_ndarray_x_raises_type_error():
    """x not an ndarray → TypeError."""
    from sleap_roots.circumnutation.midline import reconstruct

    with pytest.raises(TypeError):
        reconstruct([0.0, 1.0, 2.0, 3.0, 4.0], np.arange(5.0), cadence_s=300.0)


@pytest.mark.parametrize(
    "bad_x",
    [
        np.zeros((5, 2), dtype=np.float64),  # not 1-D
        np.arange(5).astype(np.complex128),  # complex dtype
        np.array(["a", "b", "c", "d", "e"], dtype=object),  # object dtype
    ],
)
def test_reconstruct_bad_x_array_raises_value_error(bad_x):
    """Non-1-D / complex / object-dtype x → ValueError."""
    from sleap_roots.circumnutation.midline import reconstruct

    y = np.arange(len(bad_x) if bad_x.ndim == 1 else bad_x.shape[0], dtype=np.float64)
    with pytest.raises(ValueError):
        reconstruct(bad_x, y, cadence_s=300.0)


def test_reconstruct_length_mismatch_raises_value_error():
    """len(x) != len(y) → ValueError naming the lengths."""
    from sleap_roots.circumnutation.midline import reconstruct

    with pytest.raises(ValueError, match="length"):
        reconstruct(np.arange(8.0), np.arange(7.0), cadence_s=300.0)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_reconstruct_non_finite_is_rejected_not_dropped(bad):
    """A NaN/±inf in x or y RAISES ValueError (rejected, not dropped)."""
    from sleap_roots.circumnutation.midline import reconstruct

    x = np.arange(8.0)
    x[3] = bad
    with pytest.raises(ValueError):
        reconstruct(x, np.linspace(0.0, 3.0, 8), cadence_s=300.0)


def test_reconstruct_non_finite_precedes_length_gate():
    """A 3 ≤ n < sg_window all-NaN track RAISES (validation before the degenerate gate)."""
    from sleap_roots.circumnutation.midline import reconstruct

    x = np.full(3, np.nan)  # n=3 < sg_window=5, AND non-finite
    with pytest.raises(ValueError):
        reconstruct(x, np.full(3, np.nan), cadence_s=300.0)


@pytest.mark.parametrize("bad", [0, -1.0, np.nan, np.inf, -np.inf])
def test_reconstruct_invalid_cadence_value_raises_value_error(bad):
    """Invalid cadence_s value → ValueError naming cadence_s."""
    from sleap_roots.circumnutation.midline import reconstruct

    x, y = _wobble_track(n=8)
    with pytest.raises(ValueError, match="cadence_s"):
        reconstruct(x, y, cadence_s=bad)


@pytest.mark.parametrize("bad", [True, np.bool_(True), "300", [300.0]])
def test_reconstruct_invalid_cadence_type_raises_type_error(bad):
    """Invalid cadence_s type (bool/np.bool_/str/list) → TypeError naming cadence_s."""
    from sleap_roots.circumnutation.midline import reconstruct

    x, y = _wobble_track(n=8)
    with pytest.raises(TypeError, match="cadence_s"):
        reconstruct(x, y, cadence_s=bad)


def test_reconstruct_validation_precedes_degenerate_gate():
    """n == 0 with an invalid cadence_s RAISES (validation wins over graceful path)."""
    from sleap_roots.circumnutation.midline import reconstruct

    with pytest.raises((ValueError, TypeError)):
        reconstruct(np.array([]), np.array([]), cadence_s=-1.0)


@pytest.mark.parametrize("bad_window", [4, 3, 5.0])  # even, <=SG_DEGREE, non-int
def test_reconstruct_invalid_sg_window_raises_naming_field(bad_window):
    """Even / ≤ SG_DEGREE / non-int sg_window → ValueError|TypeError naming the field."""
    from sleap_roots.circumnutation.midline import reconstruct

    x, y = _wobble_track(n=16)
    with pytest.raises((ValueError, TypeError)):
        reconstruct(x, y, cadence_s=300.0, sg_window=bad_window)


def test_reconstruct_non_constantst_constants_raises_type_error():
    """constants not None and not a ConstantsT → TypeError naming constants."""
    from sleap_roots.circumnutation.midline import reconstruct

    x, y = _wobble_track(n=8)
    with pytest.raises(TypeError, match="constants"):
        reconstruct(x, y, cadence_s=300.0, constants=object())


# ---------------------------------------------------------------------------
# §5 — arc / speed / curvature numerics + cadence-independence
# (contract-locking: the happy-path orchestration was built in §3)
# ---------------------------------------------------------------------------


def _circle_xy(radius, n):
    """A densely-sampled circle of the given radius (closed loop, endpoint=False)."""
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return radius * np.cos(t), radius * np.sin(t)


def test_reconstruct_circle_numerics():
    """On a dense circle: speed ≈ constant, curvature ≈ +1/R, arc length grows."""
    from sleap_roots.circumnutation.midline import reconstruct

    R, n = 50.0, 200
    x, y = _circle_xy(R, n)
    result = reconstruct(x, y, cadence_s=300.0)

    # Per-frame arc step of a circle sampled at n points ≈ 2πR/n (constant speed).
    expected_speed = 2.0 * np.pi * R / n
    interior = slice(5, -5)  # avoid SG edge frames
    np.testing.assert_allclose(
        result.speed_px_per_frame[interior], expected_speed, rtol=1e-3
    )
    # Interior curvature ≈ +1/R (loose physical tolerance — SG discretization).
    np.testing.assert_allclose(result.curvature_px_inv[interior], 1.0 / R, atol=1e-3)
    # Arc length is monotonic, starts at 0, and ≈ the cumulative speed sum.
    assert result.arc_length_px[0] == 0.0
    assert result.arc_length_px[-1] > 0.0


def test_reconstruct_mask_polarity_and_sigma_v_definition():
    """velocity_sub_noise_mask is exactly (speed ≤ noise_mask_k·σ_v); σ_v = std(speed, ddof=0)."""
    from sleap_roots.circumnutation.midline import reconstruct

    x, y = _wobble_track(n=40, growth=1.0, amp=6.0, period=8.0)
    result = reconstruct(x, y, cadence_s=300.0)

    expected_sigma_v = float(np.std(result.speed_px_per_frame, ddof=0))
    assert result.sigma_v_px_per_frame == expected_sigma_v
    expected_mask = result.speed_px_per_frame <= (
        result.noise_mask_k * result.sigma_v_px_per_frame
    )
    np.testing.assert_array_equal(result.velocity_sub_noise_mask, expected_mask)


def test_reconstruct_is_cadence_independent():
    """arc / curvature / speed / mask are bit-identical for cadence_s=300 vs 1."""
    from sleap_roots.circumnutation.midline import reconstruct

    x, y = _wobble_track(n=48, amp=4.0)
    r300 = reconstruct(x, y, cadence_s=300.0)
    r1 = reconstruct(x, y, cadence_s=1.0)

    np.testing.assert_array_equal(r300.arc_length_px, r1.arc_length_px)
    np.testing.assert_array_equal(r300.curvature_px_inv, r1.curvature_px_inv)
    np.testing.assert_array_equal(r300.speed_px_per_frame, r1.speed_px_per_frame)
    np.testing.assert_array_equal(
        r300.velocity_sub_noise_mask, r1.velocity_sub_noise_mask
    )
    # Only the provenance scalar differs.
    assert r300.cadence_s == 300.0 and r1.cadence_s == 1.0


# ---------------------------------------------------------------------------
# §6 — degenerate / edge cases (graceful all-NaN; no raise, no RuntimeWarning)
# ---------------------------------------------------------------------------


def _assert_degenerate_result(result, n):
    """A degenerate MidlineResult: float arrays all-NaN, mask all-False bool, etc."""
    assert result.is_degenerate is True
    assert result.frame_indices.shape == (n,)
    np.testing.assert_array_equal(result.frame_indices, np.arange(n))
    for name in (
        "x_smooth_px",
        "y_smooth_px",
        "speed_px_per_frame",
        "arc_length_px",
        "curvature_px_inv",
    ):
        arr = getattr(result, name)
        assert arr.shape == (n,), name
        if n > 0:
            assert np.isnan(arr).all(), name
    # A bool array cannot hold NaN; the degenerate mask is all-False.
    assert result.velocity_sub_noise_mask.dtype == np.bool_
    assert result.velocity_sub_noise_mask.shape == (n,)
    assert not result.velocity_sub_noise_mask.any()
    assert np.isnan(result.sigma_v_px_per_frame)


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4])
def test_reconstruct_degenerate_below_window(n):
    """n < sg_window (incl n=0) → graceful all-NaN, is_degenerate=True, no warning."""
    import warnings

    from sleap_roots.circumnutation.midline import reconstruct

    x = np.arange(n, dtype=np.float64)
    y = np.linspace(0.0, 1.0, n) if n > 0 else np.array([])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = reconstruct(x, y, cadence_s=300.0)
    _assert_degenerate_result(result, n)


def test_reconstruct_n_equals_window_is_not_degenerate():
    """n == sg_window (non-stationary) → a real reconstruction (is_degenerate=False)."""
    from sleap_roots.circumnutation.midline import reconstruct

    x = np.arange(5, dtype=np.float64)
    y = np.array([0.0, 1.0, 0.5, 1.5, 0.25])
    result = reconstruct(x, y, cadence_s=300.0)
    assert result.is_degenerate is False
    assert np.isfinite(result.arc_length_px).all()


def test_reconstruct_raw_stationary_is_degenerate():
    """A raw-stationary track (x, y all-constant, n ≥ sg_window) → graceful all-NaN.

    Stationarity is detected on the RAW input (ptp(x)==0 and ptp(y)==0) because
    post-SG speed is floating-point dust, never exactly 0.
    """
    import warnings

    from sleap_roots.circumnutation.midline import reconstruct

    n = 8
    x = np.full(n, 3.0)
    y = np.full(n, -7.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = reconstruct(x, y, cadence_s=300.0)
    _assert_degenerate_result(result, n)


def test_reconstruct_no_inf_curvature_near_stall_no_warning():
    """A non-degenerate track with a near-stall yields finite-or-NaN curvature, no inf."""
    import warnings

    from sleap_roots.circumnutation.midline import reconstruct

    # Mostly-growing track with a brief near-stall (repeated-ish points).
    x = np.array([0.0, 1.0, 2.0, 2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    y = np.array([0.0, 0.2, 0.1, 0.1, 0.1, 0.3, 0.2, 0.4, 0.1, 0.5])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = reconstruct(x, y, cadence_s=300.0)
    assert not np.isinf(result.curvature_px_inv).any()


# ---------------------------------------------------------------------------
# §7 — determinism + canary + logging (contract-locking)
# ---------------------------------------------------------------------------

# Captured by scripts/circumnutation/capture_midline_canary.py (Windows, numpy
# 2.3.4 / scipy 1.16.3). Cross-OS reproducibility floor: atol=1e-9, rtol=0.
_CANARY_FRAME_INDICES = [20, 64, 100]
_MIDLINE_CIRCLE_CANARY_KAPPA = np.array(
    [0.019982228651719475, 0.019982228651719041, 0.019982228651719825],
    dtype=np.float64,
)
_MIDLINE_CIRCLE_CANARY_ARC = np.array(
    [49.087381306541396, 157.07960787946203, 245.43688416639691], dtype=np.float64
)
_MIDLINE_SYNTHETIC_CANARY_KAPPA = np.array(
    [0.070983916763426955, 0.052506569868884842, 0.0038495167641695101],
    dtype=np.float64,
)
_MIDLINE_SYNTHETIC_CANARY_ARC = np.array(
    [94.860183661027691, 303.29354513251394, 473.48121320074745], dtype=np.float64
)


def test_reconstruct_is_deterministic_same_process():
    """Two same-process calls are bit-identical (field-by-field; never r1 == r2)."""
    from sleap_roots.circumnutation.midline import reconstruct

    x, y = _wobble_track(n=64, amp=5.0)
    r1 = reconstruct(x, y, cadence_s=300.0)
    r2 = reconstruct(x, y, cadence_s=300.0)
    for name in (
        "x_smooth_px",
        "y_smooth_px",
        "speed_px_per_frame",
        "arc_length_px",
        "curvature_px_inv",
        "frame_indices",
        "velocity_sub_noise_mask",
    ):
        np.testing.assert_array_equal(getattr(r1, name), getattr(r2, name))
    assert r1.is_degenerate == r2.is_degenerate


def test_reconstruct_circle_canary_oracle_and_reproducibility():
    """Circle: κ ≈ 1/R (loose oracle) AND matches the captured canary (atol=1e-9)."""
    from sleap_roots.circumnutation.midline import reconstruct

    R, n = 50.0, 128
    x, y = _circle_xy(R, n)
    result = reconstruct(x, y, cadence_s=300.0)

    # ORACLE: physical accuracy to 1/R at a LOOSE tolerance (SG discretization).
    np.testing.assert_allclose(
        result.curvature_px_inv[_CANARY_FRAME_INDICES], 1.0 / R, atol=1e-3
    )
    # REPRODUCIBILITY: runtime vs captured-literal at the cross-OS floor.
    np.testing.assert_allclose(
        result.curvature_px_inv[_CANARY_FRAME_INDICES],
        _MIDLINE_CIRCLE_CANARY_KAPPA,
        atol=1e-9,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        result.arc_length_px[_CANARY_FRAME_INDICES],
        _MIDLINE_CIRCLE_CANARY_ARC,
        atol=1e-9,
        rtol=0.0,
    )


def test_reconstruct_synthetic_canary_reproducibility():
    """Synthetic generator: matches the captured canary across OSs (atol=1e-9)."""
    from sleap_roots.circumnutation import synthetic
    from sleap_roots.circumnutation.midline import reconstruct

    df = synthetic.generate_trajectory(
        random_state=0,
        n_frames=128,
        T_nutation_s=3333,
        cadence_s=300,
        noise_sigma_px=0.5,
    )
    x = df["tip_x"].to_numpy(dtype=np.float64)
    y = df["tip_y"].to_numpy(dtype=np.float64)
    result = reconstruct(x, y, cadence_s=300.0)

    np.testing.assert_allclose(
        result.curvature_px_inv[_CANARY_FRAME_INDICES],
        _MIDLINE_SYNTHETIC_CANARY_KAPPA,
        atol=1e-9,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        result.arc_length_px[_CANARY_FRAME_INDICES],
        _MIDLINE_SYNTHETIC_CANARY_ARC,
        atol=1e-9,
        rtol=0.0,
    )


def test_reconstruct_emits_one_debug_record(caplog):
    """Exactly one DEBUG record from the midline logger; no INFO/WARNING/ERROR."""
    import logging

    from sleap_roots.circumnutation.midline import reconstruct

    x, y = _wobble_track(n=32)
    with caplog.at_level(logging.DEBUG, logger="sleap_roots.circumnutation.midline"):
        reconstruct(x, y, cadence_s=300.0)

    records = [
        r for r in caplog.records if r.name == "sleap_roots.circumnutation.midline"
    ]
    debug_records = [r for r in records if r.levelno == logging.DEBUG]
    assert len(debug_records) == 1
    msg = debug_records[0].getMessage()
    assert msg.startswith("midline.reconstruct(")
    assert "n_frames=" in msg
    assert "sg_window=" in msg
    assert not any(r.levelno >= logging.INFO for r in records)


# ---------------------------------------------------------------------------
# §8 — real plate-001 Nipponbare validation + cross-tier (GREEN-phase)
# ---------------------------------------------------------------------------

_PROOFREAD_FIXTURE = Path(
    "tests/data/circumnutation_nipponbare_plate_001/"
    "plate_001_greyscale.tracked_proofread.slp"
)


def _load_proofread_enriched():
    """Load the 6-track Nipponbare proofread fixture, enriched + int track_id."""
    from sleap_roots.series import Series

    series = Series.load(series_name="plate_001", primary_path=str(_PROOFREAD_FIXTURE))
    df = series.get_tracked_tips()
    df["track_id"] = df["track_id"].str.replace("track_", "", regex=False).astype(int)
    df["series"] = "plate_001"
    df["sample_uid"] = "plate_001"
    df["timepoint"] = "T0"
    df["plate_id"] = "plate_001"
    df["plant_id"] = df["track_id"]
    df["genotype"] = np.nan
    df["treatment"] = np.nan
    return df


def _track_xy(df, track_id):
    """Sorted, finite tip (x, y) float64 arrays for one track."""
    sub = df[df.track_id == track_id].dropna(subset=["tip_x", "tip_y"])
    sub = sub.sort_values("frame")
    return (
        sub.tip_x.to_numpy(dtype=np.float64),
        sub.tip_y.to_numpy(dtype=np.float64),
    )


@pytest.mark.skipif(
    not _PROOFREAD_FIXTURE.exists(),
    reason=f"Git-LFS proofread fixture not present: {_PROOFREAD_FIXTURE}",
)
def test_reconstruct_real_plate001_is_physically_plausible():
    """Per real track: arc monotonic, curvature finite + bounded on unmasked frames.

    GREEN-phase observed (recorded for auditability): max|κ[~mask]| ≈ 0.09–0.17
    px⁻¹ (the unmasked array is the plausible one — the full raw array has
    near-stall blow-up); mask fraction ≈ 0.38–0.61 (the rice tip moves only ~3×
    the localization-noise floor, so ~half the frames are genuinely sub-noise —
    NOT "flags ~nothing"; the fraction is data-dependent).
    """
    from sleap_roots.circumnutation.midline import reconstruct

    df = _load_proofread_enriched()
    observed = {}
    for track_id in sorted(df.track_id.unique()):
        x, y = _track_xy(df, track_id)
        result = reconstruct(x, y, cadence_s=300.0)
        assert not result.is_degenerate, track_id

        assert result.arc_length_px[0] == 0.0
        assert np.all(np.diff(result.arc_length_px) >= -1e-9), track_id

        unmasked = ~result.velocity_sub_noise_mask
        kappa_unmasked = result.curvature_px_inv[unmasked]
        assert np.isfinite(kappa_unmasked).all(), track_id
        max_abs_kappa = float(np.max(np.abs(kappa_unmasked)))
        mask_frac = float(result.velocity_sub_noise_mask.mean())

        # The unmasked curvature is physically plausible (< 1 px⁻¹; ~6x headroom).
        assert max_abs_kappa < 1.0, (track_id, max_abs_kappa)
        # The mask fraction sits in a sane, data-dependent band (NOT ~0).
        assert 0.1 < mask_frac < 0.85, (track_id, mask_frac)
        observed[int(track_id)] = (round(max_abs_kappa, 4), round(mask_frac, 3))

    # Record the observed per-track (max|κ[~mask]|, mask_frac) for auditability.
    print(f"\nplate-001 midline observed (max_abs_kappa, mask_frac): {observed}")


@pytest.mark.skipif(
    not _PROOFREAD_FIXTURE.exists(),
    reason=f"Git-LFS proofread fixture not present: {_PROOFREAD_FIXTURE}",
)
def test_reconstruct_real_plate001_arc_length_agrees_with_tier0():
    """Cross-tier: midline arc_length[-1] vs Tier 0 path length L = ratio·D.

    Primary ROBUST invariant: arc_length_px[-1] ≤ L (SG smoothing never
    lengthens the path; SNR-independent). Secondary magnitude tolerance ≤ ~5%
    (GREEN-phase observed 2.1–3.1%, data-SNR-dependent; σ_pos recorded). Tier 0
    NaN-drops gap frames before summing, so the agreement holds only on the
    gap-free plate-001 tracks.
    """
    from sleap_roots.circumnutation import kinematics
    from sleap_roots.circumnutation._noise import compute_sg_residual_xy
    from sleap_roots.circumnutation.midline import reconstruct

    df = _load_proofread_enriched()
    tier0 = kinematics.compute(df)

    observed = {}
    for track_id in sorted(df.track_id.unique()):
        x, y = _track_xy(df, track_id)
        # Tier 0 path length L = path_displacement_ratio * D (D = net displacement).
        ratio = float(
            tier0.loc[tier0.track_id == track_id, "path_displacement_ratio"].iloc[0]
        )
        net_disp = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))
        L = ratio * net_disp

        arc_last = float(reconstruct(x, y, cadence_s=300.0).arc_length_px[-1])
        sigma_pos = compute_sg_residual_xy(x, y, 5, 3)

        # Robust SNR-independent invariant: smoothed arc never exceeds the raw sum.
        assert arc_last <= L + 1e-6, (track_id, arc_last, L)
        # Secondary magnitude tolerance (data-SNR-dependent).
        rel = abs(arc_last - L) / L
        assert rel <= 0.05, (track_id, rel)
        observed[int(track_id)] = (round(rel, 4), round(float(sigma_pos), 3))

    print(f"\nplate-001 cross-tier observed (rel_err, sigma_pos): {observed}")
