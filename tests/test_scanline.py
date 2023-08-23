import pytest
import numpy as np
from typing import List
from sleap_roots import Series
from sleap_roots.points import join_pts
from sleap_roots.lengths import get_max_length_pts
from sleap_roots.scanline import (
    get_scanline_first_ind,
    get_scanline_last_ind,
    count_scanline_intersections,
)


@pytest.fixture
def pts_3roots_with_nan():
    return np.array(
        [
            [
                [920.48862368, 267.57325711],
                [908.88587777, 285.13679716],
                [906.19049368, 308.02657426],
                [900.85484007, 332.35722827],
                [894.77714477, 353.2618793],
                [np.nan, np.nan],
            ],
            [
                [918.0094082, 248.52049295],
                [875.89084055, 312.34001093],
                [886.19983474, 408.7826485],
                [892.15722656, 492.16012042],
                [899.53514073, 576.43033348],
                [908.02496338, 668.82440186],
            ],
            [
                [939.49111908, 291.1798956],
                [938.01029766, 309.0299704],
                [938.39169586, 324.6796079],
                [939.44596587, 339.22885535],
                [939.13551705, 355.82854929],
                [938.88545817, 371.64891802],
            ],
        ]
    )


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


def test_count_scanline_intersections_canola(canola_h5):
    series = Series.load(
        canola_h5, primary_name="primary_multi_day", lateral_name="lateral_3_nodes"
    )
    primary, lateral = series[0]
    primary_pts = primary.numpy()
    lateral_pts = lateral.numpy()
    primary_pts = get_max_length_pts(primary_pts)
    pts_all_list = join_pts(primary_pts, lateral_pts)
    depth = 1080
    n_line = 50
    n_inter = count_scanline_intersections(pts_all_list, depth, n_line)
    assert n_inter.shape == (50,)
    np.testing.assert_equal(n_inter[14], 1)


def test_count_scanline_intersections_rice(rice_h5):
    series = Series.load(
        rice_h5, primary_name="longest_3do_6nodes", lateral_name="main_3do_6nodes"
    )
    lateral = series[0][1]
    lateral_pts = lateral.numpy()
    depth = 1080
    n_line = 50
    pts_all_list = join_pts(lateral_pts)
    n_inter = count_scanline_intersections(pts_all_list, depth, n_line)
    assert n_inter.shape == (50,)
    np.testing.assert_equal(n_inter[14], 2)


def test_count_scanline_intersections_basic():
    pts_list = [np.array([[0, 0], [4, 0]]), np.array([[0, 1], [4, 1]])]
    height = 2
    n_line = 3  # y-values: 0, 1, 2
    result = count_scanline_intersections(pts_list, height, n_line)
    assert np.all(result == np.array([1, 1, 0]))  # Intersections at y = 0 and y = 1


def test_count_scanline_intersections_invalid_shape():
    with pytest.raises(ValueError):
        pts_list = [np.array([0, 1])]
        count_scanline_intersections(pts_list)


def test_count_scanline_intersections_with_nan():
    pts_list = [np.array([[0, 0], [4, 0]]), np.array([[0, 1], [4, np.nan]])]
    height = 2
    n_line = 3  # y-values: 0, 1, 2
    result = count_scanline_intersections(pts_list, height, n_line)
    assert np.all(result == np.array([1, 0, 0]))  # Only one valid intersection at y = 0


def test_count_scanline_intersections_different_params():
    pts_list = [np.array([[0, 0], [4, 0]]), np.array([[0, 2], [4, 2]])]
    height = 4
    n_line = 5  # y-values: 0, 1, 2, 3, 4
    result = count_scanline_intersections(pts_list, height, n_line)
    assert np.all(
        result == np.array([1, 0, 1, 0, 0])
    )  # Intersections at y = 0 and y = 2


# test get_scanline_first_ind with canola
def test_get_scanline_first_ind(canola_h5):
    plant = Series.load(
        canola_h5, primary_name="primary_multi_day", lateral_name="lateral_3_nodes"
    )
    primary, lateral = plant[0]
    primary_pts = primary.numpy()
    lateral_pts = lateral.numpy()
    primary_pts = get_max_length_pts(primary_pts)
    depth = 1080
    n_line = 50
    pts_all_list = join_pts(primary_pts, lateral_pts)
    scanline_intersection_counts = count_scanline_intersections(
        pts_all_list,
        depth,
        n_line,
    )
    scanline_first_ind = get_scanline_first_ind(scanline_intersection_counts)
    np.testing.assert_equal(scanline_first_ind, 7)


# test get_scanline_last_ind with canola
def test_get_scanline_last_ind(canola_h5):
    plant = Series.load(
        canola_h5, primary_name="primary_multi_day", lateral_name="lateral_3_nodes"
    )
    primary, lateral = plant[0]
    primary_pts = primary.numpy()
    lateral_pts = lateral.numpy()
    primary_pts = get_max_length_pts(primary_pts)
    depth = 1080
    n_line = 50
    pts_all_list = join_pts(primary_pts, lateral_pts)
    scanline_intersection_counts = count_scanline_intersections(
        pts_all_list, depth, n_line
    )
    scanline_last_ind = get_scanline_last_ind(scanline_intersection_counts)
    np.testing.assert_equal(scanline_last_ind, 46)
