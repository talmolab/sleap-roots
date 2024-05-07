from scipy.spatial import ConvexHull
from sleap_roots import Series
from sleap_roots.convhull import (
    get_convhull,
    get_chull_line_lengths,
    get_chull_area,
    get_chull_max_height,
    get_chull_max_width,
    get_chull_perimeter,
    get_chull_division_areas,
    get_chull_areas_via_intersection,
    get_chull_intersection_vectors,
)
from sleap_roots.lengths import get_max_length_pts
from sleap_roots.points import get_all_pts_array, get_nodes
import numpy as np
import pytest


@pytest.fixture
def valid_input():
    # Example points forming a convex hull with points above and below a line
    pts = np.array([[[0, 0], [2, 2], [4, 0], [2, -2], [0, -4], [4, -4]]])
    rn_pts = np.array([[0, 0], [4, 0]])  # Line from the leftmost to rightmost rn nodes
    hull = ConvexHull(pts.reshape(-1, 2))
    expected_area_above = 16.0
    expected_area_below = 4.0
    return rn_pts, pts, hull, (expected_area_above, expected_area_below)


@pytest.fixture
def invalid_pts_shape():
    rn_pts = np.array([[0, 0], [1, 1]])
    pts = np.array([1, 2])  # Incorrect shape
    return rn_pts, pts


@pytest.fixture
def nan_in_rn_pts():
    rn_pts = np.array([[np.nan, np.nan], [1, 1]])
    pts = np.array([[[0, 0], [1, 2], [2, 3]], [[3, 1], [4, 2], [5, 3]]])
    hull = ConvexHull(pts.reshape(-1, 2))
    return rn_pts, pts, hull


@pytest.fixture
def insufficient_unique_points_for_hull():
    rn_pts = np.array([[0, 0], [1, 1]])
    pts = np.array([[[0, 0], [0, 0], [0, 0]]])  # Only one unique point
    return rn_pts, pts


@pytest.fixture
def pts_shape_3_6_2():
    return np.array(
        [
            [[-1, 0], [-1, 1], [-1, 2], [-1, 3], [-2, 4], [-3, 5]],
            [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5]],
            [[1, 0], [1, 1], [2, 2], [3, 3], [4, 4], [4, 5]],
        ]
    )


@pytest.fixture
def pts_nan31_5node():
    return np.array(
        [
            [
                [852.17755127, 216.95648193],
                [np.nan, np.nan],
                [np.nan, np.nan],
                [np.nan, np.nan],
                [816.71142578, 808.12585449],
            ],
            [
                [852.17755127, 216.95648193],
                [np.nan, np.nan],
                [837.03405762, 588.5123291],
                [828.87963867, 692.72009277],
                [816.71142578, 808.12585449],
            ],
        ]
    )


@pytest.fixture
def pts_nan_5node():
    return np.array(
        [
            [
                [852.17755127, 216.95648193],
                [np.nan, np.nan],
                [np.nan, np.nan],
                [np.nan, np.nan],
                [816.71142578, 808.12585449],
            ],
        ]
    )


@pytest.fixture
def lateral_pts():
    return np.array(
        [
            [
                [852.17755127, 216.95648193],
                [np.nan, np.nan],
                [837.03405762, 588.5123291],
                [828.87963867, 692.72009277],
                [816.71142578, 808.12585449],
            ],
            [
                [852.17755127, 216.95648193],
                [844.45300293, 472.83520508],
                [837.03405762, 588.5123291],
                [828.87963867, 692.72009277],
                [816.71142578, 808.12585449],
            ],
        ]
    )


# test get_convhull function using canola
def test_get_convhull_canola(canola_h5):
    # Set frame index to 0
    frame_index = 0
    # Load the series from the canola dataset
    series = Series.load(canola_h5, primary_name="primary", lateral_name="lateral")
    # Get the primary and lateral root from the series
    primary_pts = series.get_primary_points(frame_index)
    lateral_pts = series.get_lateral_points(frame_index)
    # Get the maximum length points from the primary root
    primary_max_length_pts = get_max_length_pts(primary_pts)
    # Get all points from the primary and lateral roots
    pts = get_all_pts_array(primary_max_length_pts, lateral_pts)
    convex_hull = get_convhull(pts)
    assert type(convex_hull) == ConvexHull


# test canola model
def test_get_convhull_features_canola(canola_h5):
    # Set the frame index to 0
    frame_index = 0
    # Load the series from the canola dataset
    series = Series.load(canola_h5, primary_name="primary", lateral_name="lateral")
    # Get the primary and lateral root from the series
    primary_pts = series.get_primary_points(frame_index)
    lateral_pts = series.get_lateral_points(frame_index)
    primary_max_length_pts = get_max_length_pts(primary_pts)
    # Get all points from the primary and lateral roots
    pts = get_all_pts_array(primary_max_length_pts, lateral_pts)
    # Get the convex hull from the points
    convex_hull = get_convhull(pts)

    perimeter = get_chull_perimeter(convex_hull)
    area = get_chull_area(convex_hull)
    max_width = get_chull_max_width(convex_hull)
    max_height = get_chull_max_height(convex_hull)

    np.testing.assert_almost_equal(perimeter, 1910.0476127930017, decimal=3)
    np.testing.assert_almost_equal(area, 93255.32153574759, decimal=3)
    np.testing.assert_almost_equal(max_width, 211.279296875, decimal=3)
    np.testing.assert_almost_equal(max_height, 876.5622253417969, decimal=3)


# test rice model
def test_get_convhull_features_rice(rice_h5):
    # Set the frame index to 0
    frame_index = 0
    # Load the series from the rice dataset
    series = Series.load(rice_h5, primary_name="primary", crown_name="crown")
    # Get the crown root from the series
    crown_pts = series.get_crown_points(frame_index)
    # Get the convex hull from the points
    convex_hull = get_convhull(crown_pts)
    perimeter = get_chull_perimeter(convex_hull)
    area = get_chull_area(convex_hull)
    max_width = get_chull_max_width(convex_hull)
    max_height = get_chull_max_height(convex_hull)

    np.testing.assert_almost_equal(perimeter, 1450.6365795858003, decimal=3)
    np.testing.assert_almost_equal(area, 23722.883102604676, decimal=3)
    np.testing.assert_almost_equal(max_width, 64.341064453125, decimal=3)
    np.testing.assert_almost_equal(max_height, 715.6949920654297, decimal=3)


# test plant with 2 roots/instances with nan nodes
def test_get_convhull_features_nan(pts_nan31_5node):
    convex_hull = get_convhull(pts_nan31_5node)

    perimeter = get_chull_perimeter(convex_hull)
    area = get_chull_area(convex_hull)
    max_width = get_chull_max_width(convex_hull)
    max_height = get_chull_max_height(convex_hull)

    np.testing.assert_almost_equal(perimeter, 1184.6684128638494, decimal=3)
    np.testing.assert_almost_equal(area, 2276.1159928281368, decimal=3)
    np.testing.assert_almost_equal(max_width, 35.46612548999997, decimal=3)
    np.testing.assert_almost_equal(max_height, 591.16937256, decimal=3)


# test plant with 1 root/instance with only 2 non-nan nodes
def test_get_convhull_features_nanall(pts_nan_5node):
    convex_hull = get_convhull(pts_nan_5node)

    perimeter = get_chull_perimeter(convex_hull)
    area = get_chull_area(convex_hull)
    max_width = get_chull_max_width(convex_hull)
    max_height = get_chull_max_height(convex_hull)

    np.testing.assert_almost_equal(perimeter, np.nan, decimal=3)
    np.testing.assert_almost_equal(area, np.nan, decimal=3)
    np.testing.assert_almost_equal(max_width, np.nan, decimal=3)
    np.testing.assert_almost_equal(max_height, np.nan, decimal=3)


# test get_chull_perimeter with defined lateral_pts
def test_get_chull_perimeter(lateral_pts):
    perimeter = get_chull_perimeter(lateral_pts)
    np.testing.assert_almost_equal(perimeter, 1184.7141710619985, decimal=3)


# test get_chull_line_lengths with canola
def test_get_chull_line_lengths(canola_h5):
    # Set the frame index to 0
    frame_index = 0
    # Load the series from the canola dataset
    series = Series.load(canola_h5, primary_name="primary", lateral_name="lateral")
    # Get the primary and lateral root from the series
    primary_pts = series.get_primary_points(frame_index)
    lateral_pts = series.get_lateral_points(frame_index)
    # Get the maximum length points from the primary root
    primary_max_length_pts = get_max_length_pts(primary_pts)
    pts = get_all_pts_array(primary_max_length_pts, lateral_pts)
    convex_hull = get_convhull(pts)
    chull_line_lengths = get_chull_line_lengths(convex_hull)
    assert chull_line_lengths.shape[0] == 10
    np.testing.assert_almost_equal(chull_line_lengths[0], 227.553, decimal=3)


# test get_chull_line_lengths with none hull
def test_get_chull_line_lengths_nonehull(pts_nan_5node):
    chull_line_lengths = get_chull_line_lengths(pts_nan_5node)
    np.testing.assert_almost_equal(chull_line_lengths, np.nan, decimal=3)


def test_get_chull_division_areas(pts_shape_3_6_2):
    # Points arranged in a way that the line between the leftmost and rightmost
    # r1 nodes has and area above it and below it
    hull = get_convhull(pts_shape_3_6_2)
    r1_pts = get_nodes(pts_shape_3_6_2, 1)
    above, below = get_chull_division_areas(r1_pts, pts_shape_3_6_2, hull)
    np.testing.assert_almost_equal(above, 2.0, decimal=3)
    np.testing.assert_almost_equal(below, 16.0, decimal=3)


def test_get_chull_area_via_intersection_valid(valid_input):
    rn_pts, pts, hull, expected_areas = valid_input
    above, below = get_chull_areas_via_intersection(rn_pts, pts, hull)
    np.testing.assert_almost_equal(above, expected_areas[0], decimal=3)
    np.testing.assert_almost_equal(below, expected_areas[1], decimal=3)


def test_invalid_pts_shape_area_via_intersection(invalid_pts_shape):
    rn_pts, pts = invalid_pts_shape
    with pytest.raises(ValueError):
        _ = get_chull_areas_via_intersection(rn_pts, pts, None)


def test_nan_in_rn_pts_area_via_intersection(nan_in_rn_pts):
    rn_pts, pts, hull = nan_in_rn_pts
    area_above_line, area_below_line = get_chull_areas_via_intersection(
        rn_pts, pts, hull
    )
    assert np.isnan(area_above_line) and np.isnan(
        area_below_line
    ), "Expected NaN areas when rn_pts contains NaN values"


def test_insufficient_unique_points_for_hull_area_via_intersection(
    insufficient_unique_points_for_hull,
):
    rn_pts, pts = insufficient_unique_points_for_hull
    area_above_line, area_below_line = get_chull_areas_via_intersection(
        rn_pts, pts, None
    )
    assert np.isnan(area_above_line) and np.isnan(
        area_below_line
    ), "Expected NaN areas when there are insufficient unique points for a convex hull"


# Helper function to create a convex hull from points
def create_convex_hull_from_points(points):
    return ConvexHull(points)


# Basic functionality test
def test_basic_functionality(pts_shape_3_6_2):
    r0_pts = pts_shape_3_6_2[:, 0, :]
    r1_pts = pts_shape_3_6_2[:, 1, :]
    pts = pts_shape_3_6_2
    hull = create_convex_hull_from_points(pts.reshape(-1, 2))

    left_vector, right_vector = get_chull_intersection_vectors(
        r0_pts, r1_pts, pts, hull
    )

    # TODO: Add more specific tests as needed
    assert not np.isnan(left_vector).any(), "Left vector should not contain NaNs"
    assert not np.isnan(right_vector).any(), "Right vector should not contain NaNs"


# Test with invalid input shapes
@pytest.mark.parametrize(
    "invalid_input",
    [
        (np.array([[1, 2, 3]]), np.array([[3, 4]]), np.array([[[1, 2], [3, 4]]]), None),
        # Add more invalid inputs as needed
    ],
)
def test_invalid_input_shapes(invalid_input):
    r0_pts, rn_pts, pts, hull = invalid_input
    with pytest.raises(ValueError):
        get_chull_intersection_vectors(r0_pts, rn_pts, pts, hull)


# Test with no convex hull
def test_no_convex_hull():
    r0_pts = np.array([[1, 1], [2, 2]])
    rn_pts = np.array([[3, 3], [4, 4]])
    pts = np.array([[[1, 1], [2, 2], [3, 3], [4, 4]]])

    left_vector, right_vector = get_chull_intersection_vectors(
        r0_pts, rn_pts, pts, None
    )

    assert np.isnan(
        left_vector
    ).all(), "Expected NaN vector for left_vector when hull is None"
    assert np.isnan(
        right_vector
    ).all(), "Expected NaN vector for right_vector when hull is None"
