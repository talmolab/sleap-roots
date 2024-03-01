import pytest
import numpy as np
from shapely import LineString, Polygon
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
    # Set the frame index = 0
    frame_index = 0
    # Load the series from canola
    series = Series.load(canola_h5, primary_name="primary", lateral_name="lateral")
    primary_pts = series.get_primary_points(frame_index)
    lateral_pts = series.get_lateral_points(frame_index)
    pts_all_array = get_all_pts_array(primary_pts, lateral_pts)
    bbox = get_bbox(pts_all_array)
    np.testing.assert_almost_equal(
        bbox, [1016.7844238, 144.4191589, 211.2792969, 876.5622253], decimal=7
    )


def test_get_bbox_rice(rice_h5):
    # Set the frame index = 0
    frame_index = 0
    # Load the series from rice
    series = Series.load(rice_h5, crown_name="crown", primary_name="primary")
    crown_pts = series.get_crown_points(frame_index)
    bbox = get_bbox(crown_pts)
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
    # Set the frame index = 0
    frame_index = 0
    # Load the series from canola
    series = Series.load(canola_h5, primary_name="primary", lateral_name="lateral")
    primary_pts = series.get_primary_points(frame_index)
    ratio = get_network_width_depth_ratio(primary_pts)
    np.testing.assert_almost_equal(ratio, 0.2191607471467916, decimal=7)


def test_get_network_width_depth_ratio_rice(rice_h5):
    # Set the frame index = 0
    frame_index = 0
    # Load the series from rice
    series = Series.load(rice_h5, crown_name="crown", primary_name="primary")
    crown_pts = series.get_crown_points(frame_index)
    ratio = get_network_width_depth_ratio(crown_pts)
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
    # Set the frame index = 0
    frame_index = 0
    # Load the series from canola
    series = Series.load(canola_h5, primary_name="primary", lateral_name="lateral")
    primary_pts = series.get_primary_points(frame_index)
    # get primary length
    primary_max_length_pts = get_max_length_pts(primary_pts)
    primary_length = get_root_lengths(primary_max_length_pts)
    # get lateral_lengths
    lateral_pts = series.get_lateral_points(frame_index)
    lateral_lengths = get_root_lengths(lateral_pts)
    length = get_network_length(primary_length, lateral_lengths)
    np.testing.assert_almost_equal(length, 1173.0531992388217, decimal=7)


def test_get_network_length_rice(rice_h5):
    # Set the frame index = 0
    frame_index = 0
    series = Series.load(rice_h5, primary_name="primary", crown_name="crown")
    crown_pts = series.get_crown_points(frame_index)
    crown_lengths = get_root_lengths(crown_pts)
    length = get_network_length(crown_lengths)
    np.testing.assert_almost_equal(length, 798.5726441151357, decimal=7)


def test_get_network_solidity(canola_h5):
    # Set the frame index = 0
    frame_index = 0
    # Load the series from canola
    series = Series.load(canola_h5, primary_name="primary", lateral_name="lateral")
    primary_pts = series.get_primary_points(frame_index)
    # get primary length
    primary_max_length_pts = get_max_length_pts(primary_pts)
    primary_length = get_root_lengths(primary_max_length_pts)
    # get lateral_lengths
    lateral_pts = series.get_lateral_points(frame_index)
    lateral_lengths = get_root_lengths(lateral_pts)
    network_length = get_network_length(primary_length, lateral_lengths)

    # get chull_area
    pts_all_array = get_all_pts_array(primary_max_length_pts, lateral_pts)
    convex_hull = get_convhull(pts_all_array)
    chull_area = get_chull_area(convex_hull)

    ratio = get_network_solidity(network_length, chull_area)
    np.testing.assert_almost_equal(ratio, 0.012578941125511587, decimal=7)


def test_get_network_solidity_rice(rice_h5):
    # Set the frame index = 0
    frame_index = 0
    # Load the series from rice
    series = Series.load(rice_h5, primary_name="primary", crown_name="crown")
    crown_pts = series.get_crown_points(frame_index)
    crown_lengths = get_root_lengths(crown_pts)
    network_length = get_network_length(crown_lengths)
    convex_hull = get_convhull(crown_pts)
    chull_area = get_chull_area(convex_hull)
    ratio = get_network_solidity(network_length, chull_area)
    np.testing.assert_almost_equal(ratio, 0.03366254601775008, decimal=7)


def test_get_network_distribution_one_point():
    # Define inputs
    primary_pts = np.array([[[1, 1], [2, 2], [3, 3]]])
    lateral_pts = np.array(
        [[[4, 4], [5, 5]], [[6, 6], [np.nan, np.nan]]]
    )  # One of the roots has only one point
    bounding_box = (0, 0, 10, 10)
<<<<<<< HEAD
=======
    fraction = 2 / 3
    monocots = False

    # Call the function
    network_length = get_network_distribution(
        primary_pts, lateral_pts, bounding_box, fraction, monocots
    )

    # Define the expected result
    # Only the valid roots should be considered in the calculation
    lower_box = Polygon(
        [(0, 10 - 10 * (2 / 3)), (0, 10), (10, 10), (10, 10 - 10 * (2 / 3))]
    )
    expected_length = (
        LineString(primary_pts[0]).intersection(lower_box).length
        + LineString(lateral_pts[0]).intersection(lower_box).length
    )

    # Assert that the result is as expected
    assert network_length == pytest.approx(expected_length)


def test_get_network_distribution_empty_arrays():
    primary_pts = np.full((2, 2), np.nan)
    lateral_pts = np.full((2, 2, 2), np.nan)
    bounding_box = (0, 0, 10, 10)

    network_length = get_network_distribution(primary_pts, lateral_pts, bounding_box)
    assert network_length == 0


def test_get_network_distribution_with_nans():
    primary_pts = np.array([[1, 1], [2, 2], [np.nan, np.nan]])
    lateral_pts = np.array([[[4, 4], [5, 5], [np.nan, np.nan]]])
    bounding_box = (0, 0, 10, 10)

    network_length = get_network_distribution(primary_pts, lateral_pts, bounding_box)

    lower_box = Polygon(
        [(0, 10 - 10 * (2 / 3)), (0, 10), (10, 10), (10, 10 - 10 * (2 / 3))]
    )
    expected_length = (
        LineString(primary_pts[:-1]).intersection(lower_box).length
        + LineString(lateral_pts[0, :-1]).intersection(lower_box).length
    )

    assert network_length == pytest.approx(expected_length)


def test_get_network_distribution_monocots():
    primary_pts = np.array([[1, 1], [2, 2], [3, 3]])
    lateral_pts = np.array([[[4, 4], [5, 5]]])
    bounding_box = (0, 0, 10, 10)
    monocots = True

    network_length = get_network_distribution(
        primary_pts, lateral_pts, bounding_box, monocots=monocots
    )

    lower_box = Polygon(
        [(0, 10 - 10 * (2 / 3)), (0, 10), (10, 10), (10, 10 - 10 * (2 / 3))]
    )
    expected_length = (
        LineString(lateral_pts[0]).intersection(lower_box).length
    )  # Only lateral_pts are considered

    assert network_length == pytest.approx(expected_length)


def test_get_network_distribution_different_fraction():
    primary_pts = np.array([[1, 1], [2, 2], [3, 3]])
    lateral_pts = np.array([[[4, 4], [5, 5]]])
    bounding_box = (0, 0, 10, 10)
    fraction = 0.5

    network_length = get_network_distribution(
        primary_pts, lateral_pts, bounding_box, fraction=fraction
    )

    lower_box = Polygon(
        [(0, 10 - 10 * fraction), (0, 10), (10, 10), (10, 10 - 10 * fraction)]
    )
    expected_length = (
        LineString(primary_pts).intersection(lower_box).length
        + LineString(lateral_pts[0]).intersection(lower_box).length
    )

    assert network_length == pytest.approx(expected_length)


def test_get_network_distribution_one_point():
    # Define inputs
    primary_pts = np.array([[[1, 1], [2, 2], [3, 3]]])
    lateral_pts = np.array(
        [[[4, 4], [5, 5]], [[6, 6], [np.nan, np.nan]]]
    )  # One of the roots has only one point
    bounding_box = (0, 0, 10, 10)
    fraction = 2 / 3
    monocots = False

    # Call the function
    network_length = get_network_distribution(
        primary_pts, lateral_pts, bounding_box, fraction, monocots
    )

    # Define the expected result
    # Only the valid roots should be considered in the calculation
    lower_box = Polygon(
        [(0, 10 - 10 * (2 / 3)), (0, 10), (10, 10), (10, 10 - 10 * (2 / 3))]
    )
    expected_length = (
        LineString(primary_pts[0]).intersection(lower_box).length
        + LineString(lateral_pts[0]).intersection(lower_box).length
    )

    # Assert that the result is as expected
    assert network_length == pytest.approx(expected_length)


def test_get_network_distribution_empty_arrays():
    primary_pts = np.full((2, 2), np.nan)
    lateral_pts = np.full((2, 2, 2), np.nan)
    bounding_box = (0, 0, 10, 10)

    network_length = get_network_distribution(primary_pts, lateral_pts, bounding_box)
    assert network_length == 0


def test_get_network_distribution_with_nans():
    primary_pts = np.array([[1, 1], [2, 2], [np.nan, np.nan]])
    lateral_pts = np.array([[[4, 4], [5, 5], [np.nan, np.nan]]])
    bounding_box = (0, 0, 10, 10)

    network_length = get_network_distribution(primary_pts, lateral_pts, bounding_box)

    lower_box = Polygon(
        [(0, 10 - 10 * (2 / 3)), (0, 10), (10, 10), (10, 10 - 10 * (2 / 3))]
    )
    expected_length = (
        LineString(primary_pts[:-1]).intersection(lower_box).length
        + LineString(lateral_pts[0, :-1]).intersection(lower_box).length
    )

    assert network_length == pytest.approx(expected_length)


def test_get_network_distribution_monocots():
    primary_pts = np.array([[1, 1], [2, 2], [3, 3]])
    lateral_pts = np.array([[[4, 4], [5, 5]]])
    bounding_box = (0, 0, 10, 10)
    monocots = True

    network_length = get_network_distribution(
        primary_pts, lateral_pts, bounding_box, monocots=monocots
    )

    lower_box = Polygon(
        [(0, 10 - 10 * (2 / 3)), (0, 10), (10, 10), (10, 10 - 10 * (2 / 3))]
    )
    expected_length = (
        LineString(lateral_pts[0]).intersection(lower_box).length
    )  # Only lateral_pts are considered

    assert network_length == pytest.approx(expected_length)


def test_get_network_distribution_different_fraction():
    primary_pts = np.array([[1, 1], [2, 2], [3, 3]])
    lateral_pts = np.array([[[4, 4], [5, 5]]])
    bounding_box = (0, 0, 10, 10)
    fraction = 0.5

    network_length = get_network_distribution(
        primary_pts, lateral_pts, bounding_box, fraction=fraction
    )

    lower_box = Polygon(
        [(0, 10 - 10 * fraction), (0, 10), (10, 10), (10, 10 - 10 * fraction)]
    )
    expected_length = (
        LineString(primary_pts).intersection(lower_box).length
        + LineString(lateral_pts[0]).intersection(lower_box).length
    )

    assert network_length == pytest.approx(expected_length)


def test_get_network_distribution(canola_h5):
    series = Series.load(
        canola_h5, primary_name="primary_multi_day", lateral_name="lateral_3_nodes"
    )
    primary, lateral = series[0]
    primary_pts = primary.numpy()
    primary_max_length_pts = get_max_length_pts(primary_pts)
    lateral_pts = lateral.numpy()
    monocots = False
    pts_all_array = get_all_pts_array(primary_max_length_pts, lateral_pts, monocots)
    bbox = get_bbox(pts_all_array)
>>>>>>> main
    fraction = 2 / 3
    pts = join_pts(primary_pts, lateral_pts)
    # Call the function
    network_length = get_network_distribution(pts, bounding_box, fraction)

    # Define the expected result
    # Only the valid roots should be considered in the calculation
    lower_box = Polygon(
        [(0, 10 - 10 * (2 / 3)), (0, 10), (10, 10), (10, 10 - 10 * (2 / 3))]
    )
    expected_length = (
        LineString(primary_pts[0]).intersection(lower_box).length
        + LineString(lateral_pts[0]).intersection(lower_box).length
    )

    # Assert that the result is as expected
    assert network_length == pytest.approx(expected_length)


def test_get_network_distribution_empty_arrays():
    primary_pts = np.full((2, 2), np.nan)
    lateral_pts = np.full((2, 2, 2), np.nan)
    bounding_box = (0, 0, 10, 10)
    pts = join_pts(primary_pts, lateral_pts)
    network_length = get_network_distribution(pts, bounding_box)
    assert network_length == 0


def test_get_network_distribution_with_nans():
    primary_pts = np.array([[1, 1], [2, 2], [np.nan, np.nan]])
    lateral_pts = np.array([[[4, 4], [5, 5], [np.nan, np.nan]]])
    bounding_box = (0, 0, 10, 10)
    pts = join_pts(primary_pts, lateral_pts)
    network_length = get_network_distribution(pts, bounding_box)
    lower_box = Polygon(
        [(0, 10 - 10 * (2 / 3)), (0, 10), (10, 10), (10, 10 - 10 * (2 / 3))]
    )
    expected_length = (
        LineString(primary_pts[:-1]).intersection(lower_box).length
        + LineString(lateral_pts[0, :-1]).intersection(lower_box).length
    )
    assert network_length == pytest.approx(expected_length)


def test_get_network_distribution_basic_functionality():
    primary_pts = np.array([[1, 1], [2, 2], [3, 3]])
    lateral_pts = np.array([[[4, 4], [5, 5]]])
    bounding_box = (0, 0, 10, 10)
    pts = join_pts(primary_pts, lateral_pts)
    network_length = get_network_distribution(pts, bounding_box)

    lower_box = Polygon(
        [(0, 10 - 10 * (2 / 3)), (0, 10), (10, 10), (10, 10 - 10 * (2 / 3))]
    )
    expected_length = (
        LineString(lateral_pts[0]).intersection(lower_box).length
    )  # Only lateral_pts are considered

    assert network_length == pytest.approx(expected_length)


def test_get_network_distribution_different_fraction():
    primary_pts = np.array([[1, 1], [2, 2], [3, 3]])
    lateral_pts = np.array([[[4, 4], [5, 5]]])
    bounding_box = (0, 0, 10, 10)
    fraction = 0.5
    pts = join_pts(primary_pts, lateral_pts)
    network_length = get_network_distribution(pts, bounding_box, fraction)

    lower_box = Polygon(
        [(0, 10 - 10 * fraction), (0, 10), (10, 10), (10, 10 - 10 * fraction)]
    )
    expected_length = (
        LineString(primary_pts).intersection(lower_box).length
        + LineString(lateral_pts[0]).intersection(lower_box).length
    )

    assert network_length == pytest.approx(expected_length)


def test_get_network_distribution(canola_h5):
    # Set the frame index = 0
    frame_index = 0
    # Load the series from canola
    series = Series.load(canola_h5, primary_name="primary", lateral_name="lateral")
    primary_pts = series.get_primary_points(frame_index)
    primary_max_length_pts = get_max_length_pts(primary_pts)
    lateral_pts = series.get_lateral_points(frame_index)
    pts_all_array = get_all_pts_array(primary_max_length_pts, lateral_pts)
    bbox = get_bbox(pts_all_array)
    pts_all_list = join_pts(primary_max_length_pts, lateral_pts)
    fraction = 2 / 3
    root_length = get_network_distribution(pts_all_list, bbox, fraction)
    np.testing.assert_almost_equal(root_length, 589.4322131363684, decimal=7)


def test_get_network_distribution_rice(rice_h5):
    # Set the frame index = 0
    frame_index = 0
    # Load the series from rice
    series = Series.load(rice_h5, primary_name="primary", crown_name="crown")
    crown_pts = series.get_crown_points(frame_index)
    bbox = get_bbox(crown_pts)
    fraction = 2 / 3
    root_length = get_network_distribution(crown_pts, bbox, fraction)
    np.testing.assert_almost_equal(root_length, 477.77168597561507, decimal=7)


def test_get_network_distribution_ratio(canola_h5):
    # Set the frame index = 0
    frame_index = 0
    # Load the series from canola
    series = Series.load(canola_h5, primary_name="primary", lateral_name="lateral")
    primary_pts = series.get_primary_points(frame_index)
    # get primary length
    primary_max_length_pts = get_max_length_pts(primary_pts)
    primary_length = get_root_lengths(primary_max_length_pts)
    # get lateral lengths
    lateral_pts = series.get_lateral_points(frame_index)
    lateral_lengths = get_root_lengths(lateral_pts)
    # get pts_all_array
    pts_all_array = get_all_pts_array(primary_max_length_pts, lateral_pts)
    bbox = get_bbox(pts_all_array)
    pts_all_list = join_pts(primary_max_length_pts, lateral_pts)
    # get network_length_lower
    network_length_lower = get_network_distribution(pts_all_list, bbox)
    # get total network length
    network_length = get_network_length(primary_length, lateral_lengths)
    # get ratio of network length in lower 2/3 of bounding box to total network length
    ratio = get_network_distribution_ratio(
        network_length,
        network_length_lower,
    )
    np.testing.assert_almost_equal(ratio, 0.5024769665338648, decimal=7)


def test_get_network_distribution_ratio_rice(rice_h5):
    # Set the frame index = 0
    frame_index = 0
    # Load the series from rice
    series = Series.load(rice_h5, primary_name="primary", crown_name="crown")
    fraction = 2 / 3
    crown_pts = series.get_crown_points(frame_index)
    crown_lengths = get_root_lengths(crown_pts)
    bbox = get_bbox(crown_pts)
    # get network_length_lower
    network_length_lower = get_network_distribution(crown_pts, bbox, fraction=fraction)
    # get total network length
    network_length = get_network_length(crown_lengths)
    # get ratio of network length in lower 2/3 of bounding box to total network length
    ratio = get_network_distribution_ratio(
        network_length,
        network_length_lower,
    )
    np.testing.assert_almost_equal(ratio, 0.5982820592421038, decimal=7)
