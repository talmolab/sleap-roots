import pytest
import numpy as np
from sleap_roots import Series
from sleap_roots.convhull import get_chull_area
from sleap_roots.networklength import get_bbox
from sleap_roots.networklength import get_network_distribution
from sleap_roots.networklength import get_network_distribution_ratio
from sleap_roots.networklength import get_network_length
from sleap_roots.networklength import get_network_solidity
from sleap_roots.networklength import get_network_width_depth_ratio
from sleap_roots.points import get_all_pts_array


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
    primary_pts = primary.numpy()
    lateral_pts = lateral.numpy()
    pts_all_array = get_all_pts_array(plant=series, frame=0, monocots=False)
    chull_area = None
    monocots = False
    ratio = get_network_solidity(
        primary_pts, lateral_pts, pts_all_array, chull_area, monocots
    )
    np.testing.assert_almost_equal(ratio, 0.012578941125511587, decimal=7)


def test_get_network_solidity_withchullarea(canola_h5):
    series = Series.load(
        canola_h5, primary_name="primary_multi_day", lateral_name="lateral_3_nodes"
    )
    primary, lateral = series[0]
    primary_pts = primary.numpy()
    lateral_pts = lateral.numpy()
    pts_all_array = get_all_pts_array(plant=series, frame=0, monocots=False)
    chull_area = get_chull_area(pts_all_array)
    monocots = False
    ratio = get_network_solidity(
        primary_pts, lateral_pts, pts_all_array, chull_area, monocots
    )
    np.testing.assert_almost_equal(ratio, 0.012578941125511587, decimal=7)


def test_get_network_solidity_rice(rice_h5):
    series = Series.load(
        rice_h5, primary_name="main_3do_6nodes", lateral_name="longest_3do_6nodes"
    )
    primary, lateral = series[0]
    primary_pts = primary.numpy()
    lateral_pts = lateral.numpy()
    pts_all_array = get_all_pts_array(plant=series, frame=0, monocots=True)
    chull_area = None
    monocots = True
    ratio = get_network_solidity(
        primary_pts, lateral_pts, pts_all_array, chull_area, monocots
    )
    np.testing.assert_almost_equal(ratio, 0.17930631242462894, decimal=7)


def test_get_network_distribution(canola_h5):
    series = Series.load(
        canola_h5, primary_name="primary_multi_day", lateral_name="lateral_3_nodes"
    )
    primary, lateral = series[0]
    primary_pts = primary.numpy()
    lateral_pts = lateral.numpy()
    pts_all_array = get_all_pts_array(plant=series, frame=0, monocots=False)
    bbox = None
    fraction = 2 / 3
    monocots = False
    root_length = get_network_distribution(
        primary_pts, lateral_pts, pts_all_array, bbox, fraction, monocots
    )
    np.testing.assert_almost_equal(root_length, 589.4322131363684, decimal=7)


def test_get_network_distribution_rice(rice_h5):
    series = Series.load(
        rice_h5, primary_name="main_3do_6nodes", lateral_name="longest_3do_6nodes"
    )
    primary, lateral = series[0]
    primary_pts = primary.numpy()
    lateral_pts = lateral.numpy()
    pts_all_array = get_all_pts_array(plant=series, frame=0, monocots=True)
    bbox = None
    fraction = 2 / 3
    monocots = True
    root_length = get_network_distribution(
        primary_pts, lateral_pts, pts_all_array, bbox, fraction, monocots
    )
    np.testing.assert_almost_equal(root_length, 475.89810040497025, decimal=7)


def test_get_network_length(canola_h5):
    series = Series.load(
        canola_h5, primary_name="primary_multi_day", lateral_name="lateral_3_nodes"
    )
    primary, lateral = series[0]
    primary_pts = primary.numpy()
    lateral_pts = lateral.numpy()
    monocots = False
    length = get_network_length(primary_pts, lateral_pts, monocots)
    np.testing.assert_almost_equal(length, 1173.0531992388217, decimal=7)


def test_get_network_length_rice(rice_h5):
    series = Series.load(
        rice_h5, primary_name="main_3do_6nodes", lateral_name="longest_3do_6nodes"
    )
    primary, lateral = series[0]
    primary_pts = primary.numpy()
    lateral_pts = lateral.numpy()
    monocots = True
    length = get_network_length(primary_pts, lateral_pts, monocots)
    np.testing.assert_almost_equal(length, 798.5726441151357, decimal=7)


def test_get_network_distribution_ratio(canola_h5):
    series = Series.load(
        canola_h5, primary_name="primary_multi_day", lateral_name="lateral_3_nodes"
    )
    primary, lateral = series[0]
    primary_pts = primary.numpy()
    lateral_pts = lateral.numpy()
    pts_all_array = get_all_pts_array(plant=series, frame=0, monocots=False)
    primary_length = None
    lateral_lengths = None
    bbox = None
    network_length_lower = None
    fraction = 2 / 3
    monocots = False
    ratio = get_network_distribution_ratio(
        primary_pts,
        lateral_pts,
        pts_all_array,
        primary_length,
        lateral_lengths,
        bbox,
        network_length_lower,
        fraction,
        monocots,
    )
    np.testing.assert_almost_equal(ratio, 0.5024769665338648, decimal=7)


def test_get_network_distribution_ratio_rice(rice_h5):
    series = Series.load(
        rice_h5, primary_name="main_3do_6nodes", lateral_name="longest_3do_6nodes"
    )
    primary, lateral = series[0]
    primary_pts = primary.numpy()
    lateral_pts = lateral.numpy()
    pts_all_array = get_all_pts_array(plant=series, frame=0, monocots=True)
    primary_length = None
    lateral_lengths = None
    bbox = None
    network_length_lower = None
    fraction = 2 / 3
    monocots = True
    ratio = get_network_distribution_ratio(
        primary_pts,
        lateral_pts,
        pts_all_array,
        primary_length,
        lateral_lengths,
        bbox,
        network_length_lower,
        fraction,
        monocots,
    )
    np.testing.assert_almost_equal(ratio, 0.5959358912579489, decimal=7)
