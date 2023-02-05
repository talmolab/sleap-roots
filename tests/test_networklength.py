import pytest
import numpy as np
from sleap_roots import Series
from sleap_roots.networklength import get_bbox
from sleap_roots.networklength import get_network_distribution
from sleap_roots.networklength import get_network_distribution_ratio
from sleap_roots.networklength import get_network_length
from sleap_roots.networklength import get_network_solidity
from sleap_roots.networklength import get_network_width_depth_ratio


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


def test_get_bbox(canola_h5):
    series = Series.load(
        canola_h5, primary_name="primary_multi_day", lateral_name="lateral_3_nodes"
    )
    primary, lateral = series[0]
    pts = primary.numpy()
    bbox = get_bbox(pts)
    np.testing.assert_almost_equal(
        bbox, [1016.7844238, 144.4191589, 192.1080322, 876.5622253], decimal=7
    )


def test_get_bbox_rice(rice_h5):
    series = Series.load(
        rice_h5, primary_name="main_3do_6nodes", lateral_name="longest_3do_6nodes"
    )
    primary, lateral = series[0]
    pts = primary.numpy()
    bbox = get_bbox(pts)
    np.testing.assert_almost_equal(
        bbox, [796.2611694, 248.6078033, 64.3410645, 715.6949921], decimal=7
    )


def test_get_bbox_nan(pts_nan3):
    pts = pts_nan3
    bbox = get_bbox(pts)
    np.testing.assert_almost_equal(
        bbox, [852.1775513, 216.9564819, 0.0, 0.0], decimal=7
    )


def test_get_network_width_depth_ratio(canola_h5):
    series = Series.load(
        canola_h5, primary_name="primary_multi_day", lateral_name="lateral_3_nodes"
    )
    primary, lateral = series[0]
    pts = primary.numpy()
    ratio = get_network_width_depth_ratio(pts)
    np.testing.assert_almost_equal(ratio, 0.2191607471467916, decimal=7)


def test_get_network_width_depth_ratio_rice(rice_h5):
    series = Series.load(
        rice_h5, primary_name="main_3do_6nodes", lateral_name="longest_3do_6nodes"
    )
    primary, lateral = series[0]
    pts = primary.numpy()
    ratio = get_network_width_depth_ratio(pts)
    np.testing.assert_almost_equal(ratio, 0.0899001182996162, decimal=7)


def test_get_network_width_depth_ratio_nan(pts_nan3):
    pts = pts_nan3
    ratio = get_network_width_depth_ratio(pts)
    np.testing.assert_almost_equal(ratio, np.nan, decimal=7)


def test_get_network_solidity(canola_h5):
    series = Series.load(
        canola_h5, primary_name="primary_multi_day", lateral_name="lateral_3_nodes"
    )
    primary, lateral = series[0]
    pts = primary.numpy()
    ratio = get_network_solidity(pts)
    np.testing.assert_almost_equal(ratio, 2.0351966283618834, decimal=7)


def test_get_network_solidity_rice(rice_h5):
    series = Series.load(
        rice_h5, primary_name="main_3do_6nodes", lateral_name="longest_3do_6nodes"
    )
    primary, lateral = series[0]
    pts = primary.numpy()
    ratio = get_network_solidity(pts)
    np.testing.assert_almost_equal(ratio, 1.9411037610434734, decimal=7)


def test_get_network_solidity_nan(pts_nan3):
    pts = pts_nan3
    ratio = get_network_solidity(pts)
    np.testing.assert_almost_equal(ratio, np.nan, decimal=7)


def test_get_network_distribution(canola_h5):
    series = Series.load(
        canola_h5, primary_name="primary_multi_day", lateral_name="lateral_3_nodes"
    )
    primary, lateral = series[0]
    pts = primary.numpy()
    fraction = 2 / 3
    root_length = get_network_distribution(pts, fraction)
    np.testing.assert_almost_equal(root_length, 589.4322131363684, decimal=7)


def test_get_network_distribution_rice(rice_h5):
    series = Series.load(
        rice_h5, primary_name="main_3do_6nodes", lateral_name="longest_3do_6nodes"
    )
    primary, lateral = series[0]
    pts = primary.numpy()
    fraction = 2 / 3
    root_length = get_network_distribution(pts, fraction)
    np.testing.assert_almost_equal(root_length, 477.77168597561507, decimal=7)


def test_get_network_distribution_nan(pts_nan3):
    pts = pts_nan3
    fraction = 2 / 3
    root_length = get_network_distribution(pts, fraction)
    np.testing.assert_almost_equal(root_length, 0, decimal=7)


def test_get_network_length(canola_h5):
    series = Series.load(
        canola_h5, primary_name="primary_multi_day", lateral_name="lateral_3_nodes"
    )
    primary, lateral = series[0]
    pts = primary.numpy()
    length = get_network_length(pts)
    np.testing.assert_almost_equal(length, 971.0504174567843, decimal=7)


def test_get_network_length_rice(rice_h5):
    series = Series.load(
        rice_h5, primary_name="main_3do_6nodes", lateral_name="longest_3do_6nodes"
    )
    primary, lateral = series[0]
    pts = primary.numpy()
    length = get_network_length(pts)
    np.testing.assert_almost_equal(length, 798.5726441151357, decimal=7)


def test_get_network_length_nan(pts_nan3):
    pts = pts_nan3
    length = get_network_length(pts)
    np.testing.assert_almost_equal(length, 0, decimal=7)


def test_get_network_distribution_ratio(canola_h5):
    series = Series.load(
        canola_h5, primary_name="primary_multi_day", lateral_name="lateral_3_nodes"
    )
    primary, lateral = series[0]
    pts = primary.numpy()
    fraction = 2 / 3
    ratio = get_network_distribution_ratio(pts, fraction)
    np.testing.assert_almost_equal(ratio, 0.6070047, decimal=7)


def test_get_network_distribution_ratio_rice(rice_h5):
    series = Series.load(
        rice_h5, primary_name="main_3do_6nodes", lateral_name="longest_3do_6nodes"
    )
    primary, lateral = series[0]
    pts = primary.numpy()
    fraction = 2 / 3
    ratio = get_network_distribution_ratio(pts, fraction)
    np.testing.assert_almost_equal(ratio, 0.5982820592421038, decimal=7)


def test_get_network_distribution_ratio_nan(pts_nan3):
    pts = pts_nan3
    fraction = 2 / 3
    ratio = get_network_distribution_ratio(pts, fraction)
    np.testing.assert_almost_equal(ratio, np.nan, decimal=7)
