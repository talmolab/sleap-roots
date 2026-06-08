"""Tests for Tier 3a midline reconstruction (PR #8, add-circumnutation-tier3a-midline).

Covers the two new shared helpers (`_noise.compute_sg_derivative`,
`_geometry.compute_path_curvature`) and the public
`midline.reconstruct(...) -> MidlineResult` machinery, mirroring the PR #5
`temporal_cwt` machinery-test shape. See
`openspec/changes/add-circumnutation-tier3a-midline/specs/circumnutation/spec.md`
Requirement: Tier 3a midline reconstruction API.
"""

import numpy as np
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
