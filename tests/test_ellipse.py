from sleap_roots import Series
import numpy as np
from sleap_roots.ellipse import (
    fit_ellipse,
    get_ellipse_a,
    get_ellipse_b,
    get_ellipse_ratio,
)
from sleap_roots.points import get_all_pts_array


def test_get_ellipse(canola_h5):
    series = Series.load(
        canola_h5, primary_name="primary_multi_day", lateral_name="lateral_3_nodes"
    )
    primary, lateral = series[0]
    # only test ellispe for primary root points
    pts = primary.numpy()
    a, b, ratio = fit_ellipse(pts)
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


def test_get_ellipse_a(canola_h5):
    plant = Series.load(
        canola_h5, primary_name="primary_multi_day", lateral_name="lateral_3_nodes"
    )
    pts_all_array = get_all_pts_array(plant=plant, frame=0, monocots=False)
    a, b, ratio = fit_ellipse(pts_all_array)
    ellipse_a = get_ellipse_a(pts_all_array)
    np.testing.assert_almost_equal(ellipse_a, 398.1275346610801, decimal=3)


def test_get_ellipse_b(canola_h5):
    plant = Series.load(
        canola_h5, primary_name="primary_multi_day", lateral_name="lateral_3_nodes"
    )
    pts_all_array = get_all_pts_array(plant=plant, frame=0, monocots=False)
    ellipse_b = get_ellipse_b(pts_all_array)
    np.testing.assert_almost_equal(ellipse_b, 115.03734180292595, decimal=3)


def test_get_ellipse_ratio(canola_h5):
    plant = Series.load(
        canola_h5, primary_name="primary_multi_day", lateral_name="lateral_3_nodes"
    )
    pts_all_array = get_all_pts_array(plant=plant, frame=0, monocots=False)
    ellipse_ratio = get_ellipse_ratio(pts_all_array)
    np.testing.assert_almost_equal(ellipse_ratio, 3.460854783511295, decimal=3)
