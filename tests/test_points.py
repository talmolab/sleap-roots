import pytest
import numpy as np
from sleap_roots import Series
from sleap_roots.lengths import get_max_length_pts
from sleap_roots.points import get_count, join_pts
from sleap_roots.points import (
    get_all_pts_array,
)


# test get_count function with canola
def test_get_lateral_count(canola_h5):
    # Set frame index to 0
    frame_idx = 0
    # Load the series
    series = Series.load(canola_h5, primary_name="primary", lateral_name="lateral")
    # Get the lateral points
    lateral_points = series.get_lateral_points(frame_idx)
    # Get the count of lateral roots
    lateral_count = get_count(lateral_points)
    assert lateral_count == 5


def test_join_pts_basic_functionality():
    pts1 = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
    pts2 = np.array([[8, 9], [10, 11]])
    result = join_pts(pts1, pts2)

    expected = [
        np.array([[0, 1], [2, 3]]),
        np.array([[4, 5], [6, 7]]),
        np.array([[8, 9], [10, 11]]),
    ]
    for r, e in zip(result, expected):
        assert np.array_equal(r, e)
        assert r.shape == (2, 2)


def test_join_pts_single_array_input():
    pts = np.array([[[0, 1], [2, 3]]])
    result = join_pts(pts)

    expected = [np.array([[0, 1], [2, 3]])]
    for r, e in zip(result, expected):
        assert np.array_equal(r, e)
        assert r.shape == (2, 2)


def test_join_pts_none_input():
    pts1 = np.array([[[0, 1], [2, 3]]])
    pts2 = None
    result = join_pts(pts1, pts2)

    expected = [np.array([[0, 1], [2, 3]])]
    for r, e in zip(result, expected):
        assert np.array_equal(r, e)
        assert r.shape == (2, 2)


def test_join_pts_invalid_shape():
    # Test for array with last dimension not equal to 2
    with pytest.raises(ValueError):
        pts = np.array([[[0, 1, 2]]])
        join_pts(pts)

    # Test for array with more than 3 dimensions
    with pytest.raises(ValueError):
        pts = np.array([[[[0, 1]]]])
        join_pts(pts)

    # Test for array with fewer than 2 dimensions
    with pytest.raises(ValueError):
        pts = np.array([0, 1])
        join_pts(pts)


def test_join_pts_mixed_shapes():
    pts1 = np.array([[0, 1], [2, 3]])
    pts2 = np.array([[[4, 5], [6, 7]], [[8, 9], [10, 11]]])
    result = join_pts(pts1, pts2)

    expected = [
        np.array([[0, 1], [2, 3]]),
        np.array([[4, 5], [6, 7]]),
        np.array([[8, 9], [10, 11]]),
    ]
    for r, e in zip(result, expected):
        assert np.array_equal(r, e)
        assert r.shape == (2, 2)


# test get_all_pts_array function
def test_get_all_pts_array(canola_h5):
    # Set frame index to 0
    frame_idx = 0
    # Load the series
    plant = Series.load(canola_h5, primary_name="primary", lateral_name="lateral")
    # Get the primary points
    primary_pts = plant.get_primary_points(frame_idx)
    # Get primary length
    primary_max_length_pts = get_max_length_pts(primary_pts)
    # Get lateral points
    lateral_pts = plant.get_lateral_points(frame_idx)
    pts_all_array = get_all_pts_array(primary_max_length_pts, lateral_pts)
    assert pts_all_array.shape[0] == 21


# test get_all_pts_array function
def test_get_all_pts_array_rice(rice_h5):
    # Set frame index to 0
    frame_idx = 0
    # Load the series
    plant = Series.load(rice_h5, primary_name="primary", lateral_name="crown")
    # Get the lateral points
    lateral_pts = plant.get_lateral_points(frame_idx)
    # Get the flattened array with all of the points
    pts_all_array = get_all_pts_array(lateral_pts)
    assert pts_all_array.shape[0] == 12
