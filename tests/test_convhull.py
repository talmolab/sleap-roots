from scipy.spatial import ConvexHull
from sleap_roots import Series
from sleap_roots.convhull import (
    get_convhull,
    get_chull_line_lengths,
    get_chull_area,
    get_chull_max_height,
    get_chull_max_width,
    get_chull_perimeter,
)
from sleap_roots.lengths import get_max_length_pts
from sleap_roots.points import get_all_pts_array
import numpy as np
import pytest


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
    series = Series.load(
        canola_h5, primary_name="primary_multi_day", lateral_name="lateral_3_nodes"
    )
    primary, lateral = series[0]
    primary_pts = primary.numpy()
    lateral_pts = lateral.numpy()
    primary_max_length_pts = get_max_length_pts(primary_pts)
    pts = get_all_pts_array(primary_max_length_pts, lateral_pts)
    convex_hull = get_convhull(pts)
    assert type(convex_hull) == ConvexHull


# test canola model
def test_get_convhull_features_canola(canola_h5):
    series = Series.load(
        canola_h5, primary_name="primary_multi_day", lateral_name="lateral_3_nodes"
    )
    primary, lateral = series[0]
    primary_pts = primary.numpy()
    lateral_pts = lateral.numpy()
    primary_max_length_pts = get_max_length_pts(primary_pts)
    pts = get_all_pts_array(primary_max_length_pts, lateral_pts)
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
    series = Series.load(
        rice_h5, primary_name="longest_3do_6nodes", lateral_name="main_3do_6nodes"
    )
    primary, lateral = series[0]
    primary_pts = primary.numpy()
    lateral_pts = lateral.numpy()
    primary_max_length_pts = get_max_length_pts(primary_pts)
    pts = get_all_pts_array(primary_max_length_pts, lateral_pts)
    convex_hull = get_convhull(pts)

    perimeter = get_chull_perimeter(convex_hull)
    area = get_chull_area(convex_hull)
    max_width = get_chull_max_width(convex_hull)
    max_height = get_chull_max_height(convex_hull)

    np.testing.assert_almost_equal(perimeter, 1458.8585933576614, decimal=3)
    np.testing.assert_almost_equal(area, 23878.72090798154, decimal=3)
    np.testing.assert_almost_equal(max_width, 64.4229736328125, decimal=3)
    np.testing.assert_almost_equal(max_height, 720.0375061035156, decimal=3)


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
    series = Series.load(
        canola_h5, primary_name="primary_multi_day", lateral_name="lateral_3_nodes"
    )
    primary, lateral = series[0]
    primary_pts = primary.numpy()
    lateral_pts = lateral.numpy()
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
