import pytest
import numpy as np
from sleap_roots import Series
from sleap_roots.networklength import get_network_length, get_network_length_ratio


@pytest.fixture
def pts_nan3():
    return np.array(
        [
            [
                [852.17755127, 216.95648193],
                [np.nan, 472.83520508],
                [844.45300293, np.nan],
            ]
        ]
    )


def test_get_network_length(canola_h5):
    series = Series.load(
        canola_h5, primary_name="primary_multi_day", lateral_name="lateral_3_nodes"
    )
    primary, lateral = series[0]
    pts = primary.numpy()
    fraction = 2 / 3
    root_length = get_network_length(pts, fraction)
    np.testing.assert_almost_equal(root_length, 589.4322131363684, decimal=7)


def test_get_network_length_rice(rice_h5):
    series = Series.load(
        rice_h5, primary_name="main_3do_6nodes", lateral_name="longest_3do_6nodes"
    )
    primary, lateral = series[0]
    pts = primary.numpy()
    fraction = 2 / 3
    root_length = get_network_length(pts, fraction)
    np.testing.assert_almost_equal(root_length, 477.77168597561507, decimal=7)


def test_get_network_length_nan(pts_nan3):
    pts = pts_nan3
    fraction = 2 / 3
    root_length = get_network_length(pts, fraction)
    np.testing.assert_almost_equal(root_length, 0, decimal=7)


def test_get_network_length_ratio(canola_h5):
    series = Series.load(
        canola_h5, primary_name="primary_multi_day", lateral_name="lateral_3_nodes"
    )
    primary, lateral = series[0]
    pts = primary.numpy()
    fraction = 2 / 3
    ratio = get_network_length_ratio(pts, fraction)
    np.testing.assert_almost_equal(ratio, 0.6070047, decimal=7)


def test_get_network_length_ratio_rice(rice_h5):
    series = Series.load(
        rice_h5, primary_name="main_3do_6nodes", lateral_name="longest_3do_6nodes"
    )
    primary, lateral = series[0]
    pts = primary.numpy()
    fraction = 2 / 3
    ratio = get_network_length_ratio(pts, fraction)
    np.testing.assert_almost_equal(ratio, 0.5982820592421038, decimal=7)


def test_get_network_length_ratio_nan(pts_nan3):
    pts = pts_nan3
    fraction = 2 / 3
    ratio = get_network_length_ratio(pts, fraction)
    np.testing.assert_almost_equal(ratio, np.nan, decimal=7)
