import pytest
import numpy as np
from sleap_roots import Series
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
    depth = 1080
    width = 2048
    n_line = 50
    monocots = False
    n_inter = count_scanline_intersections(
        primary_pts, lateral_pts, depth, width, n_line, monocots
    )
    assert n_inter.shape == (50,)
    np.testing.assert_equal(n_inter[14], 1)


def test_count_scanline_intersections_rice(rice_h5):
    series = Series.load(
        rice_h5, primary_name="main_3do_6nodes", lateral_name="longest_3do_6nodes"
    )
    primary, lateral = series[0]
    primary_pts = primary.numpy()
    lateral_pts = lateral.numpy()
    depth = 1080
    width = 2048
    n_line = 50
    monocots = True
    n_inter = count_scanline_intersections(
        primary_pts, lateral_pts, depth, width, n_line, monocots
    )
    assert n_inter.shape == (50,)
    np.testing.assert_equal(n_inter[14], 1)


# test get_scanline_first_ind with canola
def test_get_scanline_first_ind(canola_h5):
    plant = Series.load(
        canola_h5, primary_name="primary_multi_day", lateral_name="lateral_3_nodes"
    )
    primary, lateral = plant[0]
    primary_pts = primary.numpy()
    lateral_pts = lateral.numpy()
    depth = 1080
    width = 2048
    n_line = 50
    monocots = False
    scanline_first_ind = get_scanline_first_ind(
        primary_pts, lateral_pts, depth, width, n_line, monocots
    )
    np.testing.assert_equal(scanline_first_ind, 6)


# test get_scanline_last_ind with canola
def test_get_scanline_last_ind(canola_h5):
    plant = Series.load(
        canola_h5, primary_name="primary_multi_day", lateral_name="lateral_3_nodes"
    )
    primary, lateral = plant[0]
    primary_pts = primary.numpy()
    lateral_pts = lateral.numpy()
    depth = 1080
    width = 2048
    n_line = 50
    monocots = True
    scanline_last_ind = get_scanline_last_ind(
        primary_pts, lateral_pts, depth, width, n_line, monocots
    )
    np.testing.assert_equal(scanline_last_ind, 15)
