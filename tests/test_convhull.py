from scipy.spatial import ConvexHull
from sleap_roots import Series
from sleap_roots.convhull import (
    get_convhull,
    get_convhull_features,
    get_chull_line_lengths,
    get_chull_area,
    get_chull_max_height,
    get_chull_max_width,
    get_chull_perimeter,
)
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

    primary_points = primary.numpy().reshape(-1, 2)
    lateral_points = lateral.numpy().reshape(-1, 2)
    convex_hull_points = np.concatenate((primary_points, lateral_points), axis=0)
    convex_hull = get_convhull(convex_hull_points)
    assert type(convex_hull) == ConvexHull


# test canola model
def test_get_convhull_features_canola(canola_h5):
    series = Series.load(
        canola_h5, primary_name="primary_multi_day", lateral_name="lateral_3_nodes"
    )
    primary, lateral = series[0]

    primary_points = primary.numpy().reshape(-1, 2)
    lateral_points = lateral.numpy().reshape(-1, 2)
    convex_hull_points = np.concatenate((primary_points, lateral_points), axis=0)

    (
        perimeters,
        areas,
        max_widths,
        max_heights,
    ) = get_convhull_features(convex_hull_points)

    np.testing.assert_almost_equal(perimeters, 1910.0476127930017, decimal=3)
    np.testing.assert_almost_equal(areas, 93255.32153574759, decimal=3)
    np.testing.assert_almost_equal(max_widths, 211.279296875, decimal=3)
    np.testing.assert_almost_equal(max_heights, 876.5622253417969, decimal=3)


# test rice model
def test_get_convhull_features_rice(rice_h5):
    series = Series.load(
        rice_h5, primary_name="main_3do_6nodes", lateral_name="longest_3do_6nodes"
    )
    primary, lateral = series[0]

    primary_points = primary.numpy().reshape(-1, 2)
    lateral_points = lateral.numpy().reshape(-1, 2)
    convex_hull_points = np.concatenate((primary_points, lateral_points), axis=0)

    (
        perimeters,
        areas,
        max_widths,
        max_heights,
    ) = get_convhull_features(convex_hull_points)

    np.testing.assert_almost_equal(perimeters, 1458.8585933576614, decimal=3)
    np.testing.assert_almost_equal(areas, 23878.72090798154, decimal=3)
    np.testing.assert_almost_equal(max_widths, 64.4229736328125, decimal=3)
    np.testing.assert_almost_equal(max_heights, 720.0375061035156, decimal=3)


# test plant with 2 roots/instances with nan nodes
def test_get_convhull_features_nan(pts_nan31_5node):
    (
        perimeters,
        areas,
        max_widths,
        max_heights,
    ) = get_convhull_features(pts_nan31_5node)

    np.testing.assert_almost_equal(perimeters, 1184.6684128638494, decimal=3)
    np.testing.assert_almost_equal(areas, 2276.1159928281368, decimal=3)
    np.testing.assert_almost_equal(max_widths, 35.46612548999997, decimal=3)
    np.testing.assert_almost_equal(max_heights, 591.16937256, decimal=3)


# test plant with 1 root/instance with only 2 non-nan nodes
def test_get_convhull_features_nanall(pts_nan_5node):
    (
        perimeters,
        areas,
        max_widths,
        max_heights,
    ) = get_convhull_features(pts_nan_5node)

    np.testing.assert_almost_equal(perimeters, np.nan, decimal=3)
    np.testing.assert_almost_equal(areas, np.nan, decimal=3)
    np.testing.assert_almost_equal(max_widths, np.nan, decimal=3)
    np.testing.assert_almost_equal(max_heights, np.nan, decimal=3)


# test get_chull_perimeter with defined lateral_pts
def test_get_chull_perimeter(lateral_pts):
    perimeter = get_chull_perimeter(lateral_pts)
    np.testing.assert_almost_equal(perimeter, 1184.7141710619985, decimal=3)


# test get_chull_perimeter with canola
def test_get_chull_perimeter_canola(canola_h5):
    plant = Series.load(
        canola_h5, primary_name="primary_multi_day", lateral_name="lateral_3_nodes"
    )
    pts = get_all_pts_array(plant=plant, frame=0, lateral_only=False)
    perimeter = get_chull_perimeter(pts)
    np.testing.assert_almost_equal(perimeter, 1910.0476127930017, decimal=3)


# test get_chull_area with canola
def test_get_chull_area_canola(canola_h5):
    plant = Series.load(
        canola_h5, primary_name="primary_multi_day", lateral_name="lateral_3_nodes"
    )
    pts = get_all_pts_array(plant=plant, frame=0, lateral_only=False)
    area = get_chull_area(pts)
    np.testing.assert_almost_equal(area, 93255.32153574759, decimal=3)


# test get_chull_max_width with canola
def test_get_chull_max_width(canola_h5):
    plant = Series.load(
        canola_h5, primary_name="primary_multi_day", lateral_name="lateral_3_nodes"
    )
    pts = get_all_pts_array(plant=plant, frame=0, lateral_only=False)
    max_width = get_chull_max_width(pts)
    np.testing.assert_almost_equal(max_width, 211.279296875, decimal=3)


def test_get_chull_max_height(canola_h5):
    plant = Series.load(
        canola_h5, primary_name="primary_multi_day", lateral_name="lateral_3_nodes"
    )
    pts = get_all_pts_array(plant=plant, frame=0, lateral_only=False)
    max_height = get_chull_max_height(pts)
    np.testing.assert_almost_equal(max_height, 876.5622253417969, decimal=3)


# test get_chull_line_lengths with canola
def test_get_chull_line_lengths(canola_h5):
    plant = Series.load(
        canola_h5, primary_name="primary_multi_day", lateral_name="lateral_3_nodes"
    )
    pts = get_all_pts_array(plant=plant, frame=0, lateral_only=False)
    chull_line_lengths = get_chull_line_lengths(pts)
    assert chull_line_lengths.shape[0] == 10
    np.testing.assert_almost_equal(chull_line_lengths[0], 227.553, decimal=3)


# test get_chull_line_lengths with none hull
def test_get_chull_line_lengths_nonehull(pts_nan_5node):
    chull_line_lengths = get_chull_line_lengths(pts_nan_5node)
    np.testing.assert_almost_equal(chull_line_lengths, np.nan, decimal=3)
