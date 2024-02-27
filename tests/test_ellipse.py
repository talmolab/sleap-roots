from sleap_roots import Series
import numpy as np
from sleap_roots.ellipse import (
    fit_ellipse,
    get_ellipse_a,
    get_ellipse_b,
    get_ellipse_ratio,
)
from sleap_roots.lengths import get_max_length_pts
from sleap_roots.points import get_all_pts_array
from typing import Literal


def test_get_ellipse(canola_h5: Literal["tests/data/canola_7do/919QDUH.h5"]):
    # Set the frame index = 0
    frame_index = 0
    # Load the canola dataset
    series = Series.load(canola_h5, primary_name="primary", lateral_name="lateral")
    primary_pts = series.get_primary_points(frame_index)
    # only test ellipse for primary root points
    a, b, ratio = fit_ellipse(primary_pts)
    np.testing.assert_almost_equal(a, 733.3038028507555, decimal=3)
    np.testing.assert_almost_equal(b, 146.47723651978848, decimal=3)
    np.testing.assert_almost_equal(ratio, 5.006264591916579, decimal=3)

    a, b, ratio = fit_ellipse(np.array([[1, 2], [np.nan, np.nan], [np.nan, np.nan]]))
    assert np.isnan(a)
    assert np.isnan(b)
    assert np.isnan(ratio)

    a, b, ratio = fit_ellipse(np.array([[np.nan, np.nan], [np.nan, np.nan]]))
    assert np.isnan(a)
    assert np.isnan(b)
    assert np.isnan(ratio)


def test_get_ellipse_all_points(canola_h5: Literal["tests/data/canola_7do/919QDUH.h5"]):
    # Set the frame index = 0
    frame_index = 0
    # Load the canola dataset
    series = Series.load(canola_h5, primary_name="primary", lateral_name="lateral")
    primary_pts = series.get_primary_points(frame_index)
    primary_max_length_pts = get_max_length_pts(primary_pts)
    lateral_pts = series.get_lateral_points(frame_index)
    pts_all_array = get_all_pts_array(primary_max_length_pts, lateral_pts)
    ellipse_a = get_ellipse_a(pts_all_array)
    ellipse_b = get_ellipse_b(pts_all_array)
    ellipse_ratio = get_ellipse_ratio(pts_all_array)
    np.testing.assert_almost_equal(ellipse_a, 398.1275346610801, decimal=3)
    np.testing.assert_almost_equal(ellipse_b, 115.03734180292595, decimal=3)
    np.testing.assert_almost_equal(ellipse_ratio, 3.460854783511295, decimal=3)


def test_fit_ellipse():
    # Test when pts is empty
    pts = np.array([])
    assert np.isnan(fit_ellipse(pts)).all()

    # Test when pts has less than 5 points
    pts = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    assert np.isnan(fit_ellipse(pts)).all()

    # Test when pts has NaNs only
    pts = np.array([[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]])
    assert np.isnan(fit_ellipse(pts)).all()

    # Test with collinear points
    pts = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]])
    a, b, ratio = fit_ellipse(pts)
    assert np.isnan(fit_ellipse(pts)).all()

    # Test when pts can actually fit an ellipse
    pts = np.array([[0, 1], [1, 0], [1, 1], [2, 1], [0.5, 0.5]])
    a, b, ratio = fit_ellipse(pts)
    np.testing.assert_almost_equal(a, 1.50246, decimal=3)
    np.testing.assert_almost_equal(b, 0.57389, decimal=3)
    np.testing.assert_almost_equal(ratio, 2.61803, decimal=3)

    # Test when pts has some NaNs but enough valid points to fit an ellipse
    pts = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [np.nan, np.nan]])
    a, b, ratio = fit_ellipse(pts)
    assert np.isnan(fit_ellipse(pts)).all()
