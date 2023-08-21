import pytest
import numpy as np
from sleap_roots import Series
from sleap_roots.convhull import get_chull_area, get_convhull
from sleap_roots.lengths import get_max_length_pts, get_root_lengths
from sleap_roots.networklength import get_bbox
from sleap_roots.networklength import get_network_distribution
from sleap_roots.networklength import get_network_distribution_ratio
from sleap_roots.networklength import get_network_length
from sleap_roots.networklength import get_network_solidity
from sleap_roots.networklength import get_network_width_depth_ratio
from sleap_roots.points import get_all_pts_array, join_pts


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


def test_get_network_distribution_basic_functionality():
    pts_list = [np.array([[0, 0], [4, 0]]), np.array([[0, 1], [4, 1]])]
    bounding_box = (0, 0, 4, 1)
    fraction = 2 / 3
    result = get_network_distribution(pts_list, bounding_box, fraction)
    assert (
        result == 4
    )  # Only the first line segment is in the lower 2/3 of the bounding box


def test_get_network_distribution_invalid_shape():
    with pytest.raises(ValueError):
        pts_list = [np.array([0, 1])]
        bounding_box = (0, 0, 4, 4)
        get_network_distribution(pts_list, bounding_box)


def test_get_network_distribution_invalid_bounding_box():
    with pytest.raises(ValueError):
        pts_list = [np.array([[0, 0], [4, 0]])]
        bounding_box = (0, 0, 4)
        get_network_distribution(pts_list, bounding_box)


def test_get_network_distribution_with_nan():
    # NaNs should be filtered out
    pts_list = [np.array([[0, 0], [4, 0]]), np.array([[0, 1], [4, np.nan]])]
    bounding_box = (0, 0, 4, 1)
    fraction = 2 / 3
    result = get_network_distribution(pts_list, bounding_box, fraction)
    assert (
        result == 0.0
    )  # Given (0,0) is the top-left, the line segment is in the upper 1/3


def test_get_network_distribution_with_nan_nonzero_length():
    # First line segment is at y = 2/3, which is in the lower 2/3 of the bounding box.
    # Second line segment has a NaN value and will be filtered out.
    pts_list = [np.array([[0, 2 / 3], [4, 2 / 3]]), np.array([[0, 1], [4, np.nan]])]
    bounding_box = (0, 0, 4, 1)
    fraction = 2 / 3
    result = get_network_distribution(pts_list, bounding_box, fraction)
    assert (
        result == 4.0
    )  # Only the first line segment is in the lower 2/3 and its length is 4.


def test_get_network_distribution_different_fraction():
    pts_list = [np.array([[0, 0], [4, 0]]), np.array([[0, 1], [4, 1]])]
    bounding_box = (0, 0, 4, 1)
    fraction = 1  # Cover the whole bounding box
    result = get_network_distribution(pts_list, bounding_box, fraction)
    assert result == 8  # Both line segments are in the lower part of the bounding box


def test_get_network_length(canola_h5):
    series = Series.load(
        canola_h5, primary_name="primary_multi_day", lateral_name="lateral_3_nodes"
    )
    primary, lateral = series[0]
    primary_pts = primary.numpy()
    # get primary length
    primary_max_length_pts = get_max_length_pts(primary_pts)
    primary_length = get_root_lengths(primary_max_length_pts)
    # get lateral_lengths
    lateral_pts = lateral.numpy()
    lateral_lengths = get_root_lengths(lateral_pts)
    monocots = False
    length = get_network_length(primary_length, lateral_lengths, monocots)
    np.testing.assert_almost_equal(length, 1173.0531992388217, decimal=7)


def test_get_network_length_rice(rice_h5):
    series = Series.load(
        rice_h5, primary_name="longest_3do_6nodes", lateral_name="main_3do_6nodes"
    )
    primary, lateral = series[0]
    primary_pts = primary.numpy()
    # get primary length
    primary_max_length_pts = get_max_length_pts(primary_pts)
    primary_length = get_root_lengths(primary_max_length_pts)
    # get lateral_lengths
    lateral_pts = lateral.numpy()
    lateral_lengths = get_root_lengths(lateral_pts)
    monocots = True
    length = get_network_length(primary_length, lateral_lengths, monocots)
    np.testing.assert_almost_equal(length, 798.5726441151357, decimal=7)


def test_get_network_solidity(canola_h5):
    series = Series.load(
        canola_h5, primary_name="primary_multi_day", lateral_name="lateral_3_nodes"
    )
    primary, lateral = series[0]
    primary_pts = primary.numpy()
    # get primary length
    primary_max_length_pts = get_max_length_pts(primary_pts)
    primary_length = get_root_lengths(primary_max_length_pts)
    # get lateral_lengths
    lateral_pts = lateral.numpy()
    lateral_lengths = get_root_lengths(lateral_pts)
    monocots = False
    network_length = get_network_length(primary_length, lateral_lengths, monocots)

    # get chull_area
    pts_all_array = get_all_pts_array(primary_max_length_pts, lateral_pts)
    convex_hull = get_convhull(pts_all_array)
    chull_area = get_chull_area(convex_hull)

    ratio = get_network_solidity(network_length, chull_area)
    np.testing.assert_almost_equal(ratio, 0.012578941125511587, decimal=7)


def test_get_network_solidity_rice(rice_h5):
    series = Series.load(
        rice_h5, primary_name="longest_3do_6nodes", lateral_name="main_3do_6nodes"
    )
    primary, lateral = series[0]
    primary_pts = primary.numpy()
    # get primary length
    primary_max_length_pts = get_max_length_pts(primary_pts)
    primary_length = get_root_lengths(primary_max_length_pts)
    # get lateral_lengths
    lateral_pts = lateral.numpy()
    lateral_lengths = get_root_lengths(lateral_pts)
    monocots = True
    network_length = get_network_length(primary_length, lateral_lengths, monocots)

    # get chull_area
    pts_all_array = get_all_pts_array(lateral_pts)
    convex_hull = get_convhull(pts_all_array)
    chull_area = get_chull_area(convex_hull)

    ratio = get_network_solidity(network_length, chull_area)
    np.testing.assert_almost_equal(ratio, 0.03366254601775008, decimal=7)


def test_get_network_distribution(canola_h5):
    series = Series.load(
        canola_h5, primary_name="primary_multi_day", lateral_name="lateral_3_nodes"
    )
    primary, lateral = series[0]
    primary_pts = primary.numpy()
    primary_max_length_pts = get_max_length_pts(primary_pts)
    lateral_pts = lateral.numpy()
    pts_all_array = get_all_pts_array(primary_max_length_pts, lateral_pts)
    bbox = get_bbox(pts_all_array)
    pts_all_list = join_pts(primary_max_length_pts, lateral_pts)
    fraction = 2 / 3
    root_length = get_network_distribution(pts_all_list, bbox, fraction)
    np.testing.assert_almost_equal(root_length, 589.4322131363684, decimal=7)


def test_get_network_distribution_rice(rice_h5):
    series = Series.load(
        rice_h5, primary_name="longest_3do_6nodes", lateral_name="main_3do_6nodes"
    )
    lateral = series[0][1]
    lateral_pts = lateral.numpy()
    pts_all_array = get_all_pts_array(lateral_pts)
    bbox = get_bbox(pts_all_array)
    fraction = 2 / 3
    pts_all_list = join_pts(lateral_pts)
    root_length = get_network_distribution(pts_all_list, bbox, fraction)
    np.testing.assert_almost_equal(root_length, 477.77168597561507, decimal=7)


def test_get_network_distribution_ratio(canola_h5):
    series = Series.load(
        canola_h5, primary_name="primary_multi_day", lateral_name="lateral_3_nodes"
    )
    monocots = False
    primary, lateral = series[0]
    primary_pts = primary.numpy()
    # get primary length
    primary_max_length_pts = get_max_length_pts(primary_pts)
    primary_length = get_root_lengths(primary_max_length_pts)
    # get lateral lengths
    lateral_pts = lateral.numpy()
    lateral_lengths = get_root_lengths(lateral_pts)
    # get pts_all_array
    pts_all_array = get_all_pts_array(primary_max_length_pts, lateral_pts)
    bbox = get_bbox(pts_all_array)
    pts_all_list = join_pts(primary_max_length_pts, lateral_pts)
    # get network_length_lower
    network_length_lower = get_network_distribution(pts_all_list, bbox)
    ratio = get_network_distribution_ratio(
        primary_length,
        lateral_lengths,
        network_length_lower,
        monocots,
    )
    np.testing.assert_almost_equal(ratio, 0.5024769665338648, decimal=7)


def test_get_network_distribution_ratio_rice(rice_h5):
    series = Series.load(
        rice_h5, primary_name="longest_3do_6nodes", lateral_name="main_3do_6nodes"
    )
    monocots = True
    fraction = 2 / 3
    primary, lateral = series[0]
    primary_pts = primary.numpy()
    # get primary length
    primary_max_length_pts = get_max_length_pts(primary_pts)
    primary_length = get_root_lengths(primary_max_length_pts)
    # get lateral lengths
    lateral_pts = lateral.numpy()
    lateral_lengths = get_root_lengths(lateral_pts)
    # get pts_all_array
    pts_all_array = get_all_pts_array(lateral_pts)
    bbox = get_bbox(pts_all_array)
    pts_all_list = join_pts(lateral_pts)
    # get network_length_lower
    network_length_lower = get_network_distribution(
        pts_all_list, bbox, fraction=fraction
    )
    ratio = get_network_distribution_ratio(
        primary_length,
        lateral_lengths,
        network_length_lower,
        monocots,
    )

    np.testing.assert_almost_equal(ratio, 0.5982820592421038, decimal=7)
