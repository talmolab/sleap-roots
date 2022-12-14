from sleap_roots import Series
import numpy as np
from sleap_roots.ellipse import fit_ellipse


def test_get_ellipse(canola_h5):
    series = Series.load(
        canola_h5, primary_name="primary_multi_day", lateral_name="lateral_3_nodes"
    )
    primary, lateral = series[0]

    pts = primary.numpy()
    a, b, ratio = fit_ellipse(pts)
    np.testing.assert_almost_equal(a, 733.3038028507555, decimal=3)
    np.testing.assert_almost_equal(b, 146.47723651978848, decimal=3)
    np.testing.assert_almost_equal(ratio, 0.19974972985323525, decimal=3)

    a, b, ratio = fit_ellipse(np.array([[1, 2], [np.nan, np.nan], [np.nan, np.nan]]))
    assert np.isnan(a)
    assert np.isnan(b)
    assert np.isnan(ratio)

    a, b, ratio = fit_ellipse(np.array([[np.nan, np.nan], [np.nan, np.nan]]))
    assert np.isnan(a)
    assert np.isnan(b)
    assert np.isnan(ratio)
