import numpy as np
import pytest
from sleap_roots import Series
from sleap_roots.points import (
    get_pt_ind,
    get_primary_pts,
    get_lateral_pts,
    get_all_pts,
    get_all_pts_array,
)


@pytest.fixture
def pts_nan2():
    return np.array(
        [
            [
                [852.17755127, 216.95648193],
                [np.nan, 472.83520508],
                [844.45300293, 472.83520508],
                [837.03405762, 588.5123291],
                [828.87963867, 692.72009277],
                [816.71142578, 808.12585449],
            ]
        ]
    )


@pytest.fixture
def pts_nan3():
    return np.array(
        [
            [
                [852.17755127, 216.95648193],
                [np.nan, np.nan],
                [np.nan, np.nan],
                [837.03405762, 588.5123291],
                [828.87963867, 692.72009277],
                [816.71142578, 808.12585449],
            ]
        ]
    )


@pytest.fixture
def pts_nan6():
    return np.array(
        [
            [
                [852.17755127, 216.95648193],
                [np.nan, np.nan],
                [np.nan, np.nan],
                [837.03405762, 588.5123291],
                [828.87963867, 692.72009277],
                [np.nan, np.nan],
            ]
        ]
    )


@pytest.fixture
def pts_nanall():
    return np.array(
        [
            [
                [np.nan, np.nan],
                [np.nan, np.nan],
                [np.nan, np.nan],
                [np.nan, np.nan],
                [np.nan, np.nan],
                [np.nan, np.nan],
            ]
        ]
    )


@pytest.fixture
def pts_nan32():
    return np.array(
        [
            [
                [852.17755127, 216.95648193],
                [np.nan, np.nan],
                [np.nan, np.nan],
                [837.03405762, 588.5123291],
                [828.87963867, 692.72009277],
                [816.71142578, 808.12585449],
            ],
            [
                [852.17755127, 216.95648193],
                [np.nan, np.nan],
                [844.45300293, 472.83520508],
                [837.03405762, 588.5123291],
                [828.87963867, 692.72009277],
                [816.71142578, 808.12585449],
            ],
        ]
    )


@pytest.fixture
def pts_nan32_5node():
    return np.array(
        [
            [
                [852.17755127, 216.95648193],
                [np.nan, np.nan],
                [np.nan, np.nan],
                [828.87963867, 692.72009277],
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


# test get_pt_ind function
def test_get_pt_ind(canola_h5):
    series = Series.load(
        canola_h5, primary_name="primary_multi_day", lateral_name="lateral_3_nodes"
    )
    primary, lateral = series[0]
    pts = primary.numpy()
    proximal = True
    node_ind = get_pt_ind(pts, proximal)
    np.testing.assert_array_equal(node_ind, [1])


# test get_pt_ind function using root that without second node
def test_get_pt_ind_nan2(pts_nan2):
    proximal = True
    node_ind = get_pt_ind(pts_nan2, proximal)
    np.testing.assert_array_equal(node_ind, [2])


# test get_pt_ind function using root that without second and third nodes
def test_get_pt_ind_nan3(pts_nan3):
    proximal = True
    node_ind = get_pt_ind(pts_nan3, proximal)
    np.testing.assert_array_equal(node_ind, [3])


# test get_pt_ind function using two roots/instances
def test_get_pt_ind_nan32(pts_nan32):
    proximal = True
    node_ind = get_pt_ind(pts_nan32, proximal)
    np.testing.assert_array_equal(node_ind, [3, 2])


# test get_pt_ind function using root that without last node
def test_get_pt_ind_nan6(pts_nan6):
    proximal = False
    node_ind = get_pt_ind(pts_nan6, proximal)
    np.testing.assert_array_equal(node_ind, [4])


# test get_pt_ind function using root with all nan node
def test_get_pt_ind_nanall(pts_nanall):
    proximal = False
    node_ind = get_pt_ind(pts_nanall, proximal)
    np.testing.assert_array_equal(node_ind, [0])


# test get_primary_pts function
def test_get_primary_pts(canola_h5):
    plant = Series.load(
        canola_h5, primary_name="primary_multi_day", lateral_name="lateral_3_nodes"
    )
    pts_pr = get_primary_pts(plant=plant, frame=0)
    assert pts_pr.shape == (1, 6, 2)


# test get_lateral_pts function
def test_get_lateral_pts(canola_h5):
    plant = Series.load(
        canola_h5, primary_name="primary_multi_day", lateral_name="lateral_3_nodes"
    )
    pts_lr = get_lateral_pts(plant=plant, frame=0)
    assert pts_lr.shape == (5, 3, 2)


# test get_all_pts function
def test_get_all_pts(canola_h5):
    plant = Series.load(
        canola_h5, primary_name="primary_multi_day", lateral_name="lateral_3_nodes"
    )
    pts_all = get_all_pts(plant=plant, frame=0, monocots=False)
    assert len(pts_all) == 6
    assert len(pts_all[0]) == 6
    assert len(pts_all[1]) == 3


# test get_all_pts_array function
def test_get_all_pts_array(canola_h5):
    plant = Series.load(
        canola_h5, primary_name="primary_multi_day", lateral_name="lateral_3_nodes"
    )
    pts_all_array = get_all_pts_array(plant=plant, frame=0, monocots=False)
    assert pts_all_array.shape[0] == 21
