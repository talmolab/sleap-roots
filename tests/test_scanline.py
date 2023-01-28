import pytest
import numpy as np
from sleap_roots import Series
from sleap_roots.scanline import get_scanline_intersections, count_scaline_intersections


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


def test_get_scanline_intersections_canola(canola_h5):
    series = Series.load(
        canola_h5, primary_name="primary_multi_day", lateral_name="lateral_3_nodes"
    )
    primary, lateral = series[0]
    pts = primary.numpy()
    depth = 1080
    width = 2048
    n_line = 50
    intersection = get_scanline_intersections(pts, depth, width, n_line)
    assert len(intersection) == 50
    np.testing.assert_almost_equal(
        intersection[10], [[1146.7898883311389, 253.0]], decimal=7
    )


def test_get_scanline_intersections_rice(rice_h5):
    series = Series.load(
        rice_h5, primary_name="main_3do_6nodes", lateral_name="longest_3do_6nodes"
    )
    primary, lateral = series[0]
    pts = primary.numpy()

    depth = 1080
    width = 2048
    n_line = 50

    intersection = get_scanline_intersections(pts, depth, width, n_line)
    assert len(intersection) == 50
    np.testing.assert_almost_equal(
        intersection[14],
        [[811.6129907162684, 345.0], [850.4184814416584, 345.0]],
        decimal=7,
    )


def test_get_scanline_intersections_nan(pts_nan3):
    pts = pts_nan3
    depth = 1080
    width = 2048
    n_line = 50
    intersection = get_scanline_intersections(pts, depth, width, n_line)
    assert len(intersection) == 50
    np.testing.assert_almost_equal(intersection[1], [], decimal=7)


def test_count_scaline_intersections_canola(canola_h5):
    series = Series.load(
        canola_h5, primary_name="primary_multi_day", lateral_name="lateral_3_nodes"
    )
    primary, lateral = series[0]
    pts = primary.numpy()
    depth = 1080
    width = 2048
    n_line = 50
    n_inter = count_scaline_intersections(pts, depth, width, n_line)
    assert n_inter.shape == (50,)
    np.testing.assert_equal(n_inter[14], 1)


def test_count_scaline_intersections_rice(rice_h5):
    series = Series.load(
        rice_h5, primary_name="main_3do_6nodes", lateral_name="longest_3do_6nodes"
    )
    primary, lateral = series[0]
    pts = primary.numpy()

    depth = 1080
    width = 2048
    n_line = 50

    n_inter = count_scaline_intersections(pts, depth, width, n_line)
    assert n_inter.shape == (50,)
    np.testing.assert_equal(n_inter[14], 2)


def test_count_scaline_intersections_nan(pts_nan3):
    pts = pts_nan3
    depth = 1080
    width = 2048
    n_line = 50
    n_inter = count_scaline_intersections(pts, depth, width, n_line)
    assert len(n_inter) == 50
    np.testing.assert_equal(n_inter[14], 0)
