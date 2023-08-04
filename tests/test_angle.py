import numpy as np
import pytest
from sleap_roots import Series
from sleap_roots.angle import get_node_ind, get_root_angle


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


# test get_node_ind function
def test_get_node_ind(canola_h5):
    series = Series.load(
        canola_h5, primary_name="primary_multi_day", lateral_name="lateral_3_nodes"
    )
    primary, lateral = series[0]
    pts = primary.numpy()
    proximal = True
    node_ind = get_node_ind(pts, proximal)
    np.testing.assert_array_equal(node_ind, 1)


# test get_node_ind function using root that without second node
def test_get_node_ind_nan2(pts_nan2):
    proximal = True
    node_ind = get_node_ind(pts_nan2, proximal)
    np.testing.assert_array_equal(node_ind, 2)


# test get_node_ind function using root that without second and third nodes
def test_get_node_ind_nan3(pts_nan3):
    proximal = True
    node_ind = get_node_ind(pts_nan3, proximal)
    np.testing.assert_array_equal(node_ind, 0)


# test get_node_ind function using two roots/instances
def test_get_node_ind_nan32(pts_nan32):
    proximal = True
    node_ind = get_node_ind(pts_nan32, proximal)
    np.testing.assert_array_equal(node_ind, [0, 2])


# test get_node_ind function using root that without last node
def test_get_node_ind_nan6(pts_nan6):
    proximal = False
    node_ind = get_node_ind(pts_nan6, proximal)
    np.testing.assert_array_equal(node_ind, 4)


# test get_node_ind function using root with all nan node
def test_get_node_ind_nanall(pts_nanall):
    proximal = False
    node_ind = get_node_ind(pts_nanall, proximal)
    np.testing.assert_array_equal(node_ind, np.nan)


# test get_node_ind function using root with pts_nan32_5node
def test_get_node_ind_5node(pts_nan32_5node):
    proximal = False
    node_ind = get_node_ind(pts_nan32_5node, proximal)
    np.testing.assert_array_equal(node_ind, [4, 4])


# test get_node_ind function (proximal) using root with pts_nan32_5node
def test_get_node_ind_5node_proximal(pts_nan32_5node):
    proximal = True
    node_ind = get_node_ind(pts_nan32_5node, proximal)
    np.testing.assert_array_equal(node_ind, [0, 2])


# test canola get_root_angle function (base node to distal node angle)
def test_get_root_angle_distal(canola_h5):
    series = Series.load(
        canola_h5, primary_name="primary_multi_day", lateral_name="lateral_3_nodes"
    )
    primary, lateral = series[0]
    pts = primary.numpy()
    proximal = False
    angs = get_root_angle(pts, proximal)
    assert angs.shape == (1,)
    assert pts.shape == (1, 6, 2)
    np.testing.assert_almost_equal(angs, 7.7511306, decimal=3)


# test rice get_root_angle function (base node to proximal node angle)
def test_get_root_angle_proximal_rice(rice_h5):
    series = Series.load(
        rice_h5, primary_name="main_3do_6nodes", lateral_name="longest_3do_6nodes"
    )
    primary, lateral = series[0]
    pts = primary.numpy()
    proximal = True
    angs = get_root_angle(pts, proximal)
    assert angs.shape == (2,)
    assert pts.shape == (2, 6, 2)
    np.testing.assert_almost_equal(angs, [17.3180819, 3.2692877], decimal=3)


# test get_root_angle function using two roots/instances (base node to proximal node angle)
def test_get_root_angle_proximal(pts_nan32):
    proximal = True
    angs = get_root_angle(pts_nan32, proximal)
    assert angs.shape == (2,)
    np.testing.assert_almost_equal(angs, [np.nan, 1.7291381], decimal=3)


# test get_root_angle function using two roots/instances (base node to proximal node angle)
def test_get_root_angle_proximal_5node(pts_nan32_5node):
    proximal = True
    angs = get_root_angle(pts_nan32_5node, proximal)
    assert angs.shape == (2,)
    np.testing.assert_almost_equal(angs, [np.nan, 2.3339111], decimal=3)


# test get_root_angle function using root/instance with all nan value
def test_get_root_angle_proximal_5node(pts_nanall):
    proximal = True
    angs = get_root_angle(pts_nanall, proximal)
    assert angs.shape == (1,)
    np.testing.assert_almost_equal(angs, np.nan, decimal=3)
