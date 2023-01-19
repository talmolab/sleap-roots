from scipy.spatial import ConvexHull
from sleap_roots import Series
from sleap_roots.convhull import get_convhull, get_convhull_features
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
        longest_dists,
        shortest_dists,
        median_dists,
        max_widths,
        max_heights,
    ) = get_convhull_features(convex_hull_points)

    np.testing.assert_almost_equal(perimeters, 1910.0476127930017, decimal=3)
    np.testing.assert_almost_equal(areas, 93255.32153574759, decimal=3)
    np.testing.assert_almost_equal(longest_dists, 884.6450178192455, decimal=3)
    np.testing.assert_almost_equal(shortest_dists, 185.15460001685398, decimal=3)
    np.testing.assert_almost_equal(median_dists, 404.77175083902506, decimal=3)
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
        longest_dists,
        shortest_dists,
        median_dists,
        max_widths,
        max_heights,
    ) = get_convhull_features(convex_hull_points)

    np.testing.assert_almost_equal(perimeters, 1458.8585933576614, decimal=3)
    np.testing.assert_almost_equal(areas, 23878.72090798154, decimal=3)
    np.testing.assert_almost_equal(longest_dists, 720.0494276295367, decimal=3)
    np.testing.assert_almost_equal(shortest_dists, 139.99063366108192, decimal=3)
    np.testing.assert_almost_equal(median_dists, 480.1270688506847, decimal=3)
    np.testing.assert_almost_equal(max_widths, 64.4229736328125, decimal=3)
    np.testing.assert_almost_equal(max_heights, 720.0375061035156, decimal=3)


# test plant with 2 roots/instances with nan nodes
def test_get_convhull_features_nan(pts_nan31_5node):
    (
        perimeters,
        areas,
        longest_dists,
        shortest_dists,
        median_dists,
        max_widths,
        max_heights,
    ) = get_convhull_features(pts_nan31_5node)

    np.testing.assert_almost_equal(perimeters, 1184.6684128638494, decimal=3)
    np.testing.assert_almost_equal(areas, 2276.1159928281368, decimal=3)
    np.testing.assert_almost_equal(longest_dists, 592.2322796929061, decimal=3)
    np.testing.assert_almost_equal(shortest_dists, 104.52632471064256, decimal=3)
    np.testing.assert_almost_equal(median_dists, 296.20807552792064, decimal=3)
    np.testing.assert_almost_equal(max_widths, 35.46612548999997, decimal=3)
    np.testing.assert_almost_equal(max_heights, 591.16937256, decimal=3)


# test plant with 1 root/instance with only 2 non-nan nodes
def test_get_convhull_features_nanall(pts_nan_5node):
    (
        perimeters,
        areas,
        longest_dists,
        shortest_dists,
        median_dists,
        max_widths,
        max_heights,
    ) = get_convhull_features(pts_nan_5node)

    np.testing.assert_almost_equal(perimeters, np.nan, decimal=3)
    np.testing.assert_almost_equal(areas, np.nan, decimal=3)
    np.testing.assert_almost_equal(longest_dists, np.nan, decimal=3)
    np.testing.assert_almost_equal(shortest_dists, np.nan, decimal=3)
    np.testing.assert_almost_equal(median_dists, np.nan, decimal=3)
    np.testing.assert_almost_equal(max_widths, np.nan, decimal=3)
    np.testing.assert_almost_equal(max_heights, np.nan, decimal=3)
