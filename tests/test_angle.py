import numpy as np
import pytest
from sleap_roots import Series
from sleap_roots.angle import (
    get_node_ind,
    get_root_angle,
    get_vector_angles_from_gravity,
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


@pytest.mark.parametrize(
    "vector, expected_angle",
    [
        (np.array([[0, 1]]), 0),  # Directly downwards (with gravity)
        (np.array([[0, -1]]), 180),  # Directly upwards (against gravity)
        (np.array([[1, 0]]), 90),  # Right, perpendicular to gravity
        (np.array([[-1, 0]]), 90),  # Left, perpendicular to gravity
        (np.array([[1, 1]]), 45),  # Diagonal right-down
        (np.array([[1, -1]]), 135),  # Diagonal right-up, against gravity
        (np.array([[-1, 1]]), 45),  # Diagonal left-down, aligned with gravity
        (np.array([[-1, -1]]), 135),  # Diagonal left-up, against gravity
    ],
)
def test_get_vector_angle_from_gravity(vector, expected_angle):
    """Test get_vector_angle_from_gravity function with vectors from various directions,
    considering a coordinate system where positive y-direction is downwards.
    """
    angle = get_vector_angles_from_gravity(vector)
    np.testing.assert_almost_equal(angle, expected_angle, decimal=3)


# test get_node_ind function
def test_get_node_ind(canola_h5, canola_primary_slp, canola_lateral_slp):
    # Set the frame index to 0
    frame_index = 0
    # Load the series from the canola dataset
    series = Series.load(
        series_name="canola_test",
        h5_path=canola_h5,
        primary_path=canola_primary_slp,
        lateral_path=canola_lateral_slp,
    )
    # Get the primary root points
    primary_points = series.get_primary_points(frame_index)
    # Set the proximal flag to True
    proximal = True
    # Get the node index
    node_ind = get_node_ind(primary_points, proximal)
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
    np.testing.assert_array_equal(node_ind, 0)


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
def test_get_root_angle_distal(canola_h5, canola_primary_slp, canola_lateral_slp):
    # Set the frame index to 0
    frame_index = 0
    # Load the series from the canola dataset
    series = Series.load(
        series_name="canola_test",
        h5_path=canola_h5,
        primary_path=canola_primary_slp,
        lateral_path=canola_lateral_slp,
    )
    # Get the primary root points
    primary_points = series.get_primary_points(frame_index)
    # Set the proximal flag to False
    proximal = False
    # Get the distal node index
    node_ind = get_node_ind(primary_points, proximal)
    angs = get_root_angle(primary_points, node_ind, proximal)
    assert primary_points.shape == (1, 6, 2)
    np.testing.assert_almost_equal(angs, 7.7511306, decimal=3)


# test rice get_root_angle function (base node to proximal node angle)
def test_get_root_angle_proximal_rice(rice_h5, rice_main_slp, rice_long_slp):
    # Set the frame index to 0
    frame_index = 0
    # Load the series from the rice dataset
    series = Series.load(
        series_name="rice_test",
        h5_path=rice_h5,
        primary_path=rice_long_slp,
        crown_path=rice_main_slp,
    )
    # Get the crown root points
    crown_points = series.get_crown_points(frame_index)
    # Set the proximal flag to True
    proximal = True
    # Get the proximal node index
    node_ind = get_node_ind(crown_points, proximal)
    angs = get_root_angle(crown_points, node_ind, proximal)
    assert crown_points.shape == (2, 6, 2)
    assert angs.shape == (2,)
    np.testing.assert_almost_equal(angs, [17.3180819, 3.2692877], decimal=3)


# test get_root_angle function using two roots/instances (base node to proximal node angle)
def test_get_root_angle_proximal(pts_nan32):
    proximal = True
    node_ind = get_node_ind(pts_nan32, proximal)
    angs = get_root_angle(pts_nan32, node_ind, proximal)
    assert angs.shape == (2,)
    np.testing.assert_almost_equal(angs, [np.nan, 1.7291381], decimal=3)


# test get_root_angle function using two roots/instances (base node to proximal node angle)
def test_get_root_angle_proximal_5node(pts_nan32_5node):
    proximal = True
    node_ind = get_node_ind(pts_nan32_5node, proximal)
    angs = get_root_angle(pts_nan32_5node, node_ind, proximal)
    assert angs.shape == (2,)
    np.testing.assert_almost_equal(angs, [np.nan, 2.3339111], decimal=3)


# test get_root_angle function using root/instance with all nan value
def test_get_root_angle_proximal_allnan(pts_nanall):
    proximal = True
    node_ind = get_node_ind(pts_nanall, proximal)
    angs = get_root_angle(pts_nanall, node_ind, proximal)
    np.testing.assert_almost_equal(angs, np.nan, decimal=3)


def test_get_root_angle_horizontal():
    # Root pointing right, should be 90 degrees from the downward gravity vector
    # Gravity vector is upwards in this coordinate system
    pts = np.array(
        [[[0, 0], [1, 0]]]
    )  # Two nodes: base and end node horizontally aligned
    node_ind = np.array([1])
    expected_angles = np.array([90])
    angles = get_root_angle(pts, node_ind)
    assert np.allclose(angles, expected_angles), "Angle for horizontal root incorrect."


def test_get_root_angle_vertical():
    # Root pointing down, should be 0 degrees from the gravity vector
    # Gravity vector is upwards in this coordinate system
    pts = np.array(
        [[[0, 0], [0, 1]]]
    )  # Two nodes: base and end node vertically aligned downwards
    node_ind = np.array([1])
    expected_angles = np.array([0])
    angles = get_root_angle(pts, node_ind)
    assert np.allclose(angles, expected_angles), "Angle for vertical root incorrect."


def test_get_root_angle_up_left():
    # Root pointing up and to the left: should be 45 degrees from the gravity vector
    # Gravity vector is upwards in this coordinate system
    pts = np.array(
        [[[0, 0], [-1, 1]]]
    )  # Two nodes: base and end node diagonally upwards to the left
    node_ind = np.array([1])
    expected_angles = np.array([45])
    angles = get_root_angle(pts, node_ind)
    assert np.allclose(angles, expected_angles), "Angle for vertical root incorrect."


def test_get_root_angle_up_right():
    # Root pointing up and to the right: should be 45 degrees from the gravity vector
    # Gravity vector is upwards in this coordinate system
    pts = np.array(
        [[[0, 0], [1, 1]]]
    )  # Two nodes: base and end node diagonally upwards to the right
    node_ind = np.array([1])
    expected_angles = np.array([45])
    angles = get_root_angle(pts, node_ind)
    assert np.allclose(
        angles, expected_angles
    ), "Angle for diagonally upwards root incorrect."


def test_get_root_angle_down_left():
    # Root pointing down and to the left: should be 135 degrees from the gravity vector
    # Gravity vector is upwards in this coordinate system
    pts = np.array(
        [[[0, 0], [-1, -1]]]
    )  # Two nodes: base and end node diagonally downwards to the left
    node_ind = np.array([1])
    expected_angles = np.array([135])
    angles = get_root_angle(pts, node_ind)
    assert np.allclose(angles, expected_angles), "Angle for vertical root incorrect."


def test_get_root_angle_down_right():
    # Root pointing down and to the right: should be 135 degrees from the gravity vector
    # Gravity vector is upwards in this coordinate system
    pts = np.array(
        [[[0, 0], [1, -1]]]
    )  # Two nodes: base and end node diagonally downwards to the right
    node_ind = np.array([1])
    expected_angles = np.array([135])
    angles = get_root_angle(pts, node_ind)
    assert np.allclose(
        angles, expected_angles
    ), "Angle for diagonally upwards root incorrect."
