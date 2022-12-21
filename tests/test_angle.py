from sleap_roots import Series
import numpy as np
from sleap_roots.angle import get_root_base_angle


def test_get_root_base_angle(canola_h5):
    series = Series.load(
        canola_h5, primary_name="primary_multi_day", lateral_name="lateral_3_nodes"
    )
    primary, lateral = series[0]

    pts = primary.numpy()
    angs = get_root_base_angle(pts)
    assert angs.shape == (1,)
    assert pts.shape == (1, 6, 2)
    np.testing.assert_almost_equal(angs, 50.1312956, decimal=3)

    # test the instance with second node is nan value
    pts = np.array(
        [
            [
                [852.17755127, 216.95648193],
                [np.nan, np.nan],
                [844.45300293, 472.83520508],
                [837.03405762, 588.5123291],
                [828.87963867, 692.72009277],
                [816.71142578, 808.12585449],
            ]
        ]
    )
    angs = get_root_base_angle(pts)
    assert angs.shape == (1,)
    assert pts.shape == (1, 6, 2)
    np.testing.assert_almost_equal(angs, 1.7291381, decimal=3)

    # test the instance with third node is nan value
    pts = np.array(
        [
            [
                [852.17755127, 216.95648193],
                [np.nan, np.nan],
                [np.nan, np.nan],
                [837.03405762, 588.5123291],
                [828.87963867, 692.72009277],
                [816.71142578, 808.12585449],
            ]
        ]
    )
    angs = get_root_base_angle(pts)
    assert angs.shape == (1,)
    assert pts.shape == (1, 6, 2)
    np.testing.assert_almost_equal(angs, np.nan, decimal=3)

    # test two instances
    pts = np.array(
        [
            [
                [852.17755127, 216.95648193],
                [np.nan, np.nan],
                [np.nan, np.nan],
                [837.03405762, 588.5123291],
                [828.87963867, 692.72009277],
                [816.71142578, 808.12585449],
            ],
            [
                [852.17755127, 216.95648193],
                [np.nan, np.nan],
                [844.45300293, 472.83520508],
                [837.03405762, 588.5123291],
                [828.87963867, 692.72009277],
                [816.71142578, 808.12585449],
            ],
        ]
    )
    angs = get_root_base_angle(pts)
    assert angs.shape == (2,)
    assert pts.shape == (2, 6, 2)
    np.testing.assert_almost_equal(angs, [np.nan, 1.7291381], decimal=3)
