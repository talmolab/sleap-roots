import numpy as np
import pytest
from shapely.geometry import LineString
from sleap_roots import Series
from sleap_roots.lengths import get_max_length_pts, get_root_lengths
from sleap_roots.points import (
    get_all_pts_array,
    associate_lateral_to_primary,
    flatten_associated_points,
)


# test get_all_pts_array function
def test_get_all_pts_array(canola_h5):
    plant = Series.load(
        canola_h5, primary_name="primary_multi_day", lateral_name="lateral_3_nodes"
    )
    primary, lateral = plant[0]
    primary_pts = primary.numpy()
    # get primary length
    primary_max_length_pts = get_max_length_pts(primary_pts)
    # get lateral_lengths
    lateral_pts = lateral.numpy()
    monocots = False
    pts_all_array = get_all_pts_array(
        primary_max_length_pts, lateral_pts, monocots=monocots
    )
    assert pts_all_array.shape[0] == 21


# test get_all_pts_array function
def test_get_all_pts_array_rice(rice_h5):
    plant = Series.load(
        rice_h5, primary_name="longest_3do_6nodes", lateral_name="main_3do_6nodes"
    )
    primary, lateral = plant[0]
    primary_pts = primary.numpy()
    # get primary length
    primary_max_length_pts = get_max_length_pts(primary_pts)
    # get lateral_lengths
    lateral_pts = lateral.numpy()
    monocots = True
    pts_all_array = get_all_pts_array(
        primary_max_length_pts, lateral_pts, monocots=monocots
    )
    assert pts_all_array.shape[0] == 12


def test_associate_basic():
    # Tests basic association between one primary and one lateral root.
    primary_pts = np.array([[[0, 0], [0, 1]]])
    lateral_pts = np.array([[[0, 1], [0, 2]]])

    expected = {0: [np.array([[0, 1], [0, 2]])]}
    result = associate_lateral_to_primary(primary_pts, lateral_pts)

    # Loop through the result and the expected dictionary to compare the numpy arrays
    for key in expected:
        assert key in result
        np.testing.assert_array_equal(result[key], expected[key])


def test_associate_no_primary():
    # Tests that an empty dictionary is returned when there are no primary roots.
    primary_pts = np.array([])  # Empty array representing no primary roots
    lateral_pts = np.array([[[0, 1], [0, 2]]])  # Some lateral roots for the test

    expected = {}
    result = associate_lateral_to_primary(primary_pts, lateral_pts)

    assert result == expected


def test_associate_no_lateral():
    # Tests that correct association is made when there are no lateral roots.
    primary_pts = np.array([[[0, 0], [0, 1]]])
    lateral_pts = np.array([])  # No lateral roots

    expected = {0: []}  # Expected association: primary root index to an empty list
    result = associate_lateral_to_primary(primary_pts, lateral_pts)

    assert result == expected

    expected = {0: []}
    result = associate_lateral_to_primary(primary_pts, lateral_pts)
    # Loop through the result and the expected dictionary to compare the numpy arrays
    for key in expected:
        assert key in result
        np.testing.assert_array_equal(result[key], expected[key])


def test_associate_invalid_input_type():
    # Tests that the function raises a TypeError with invalid input types.
    primary_pts = [[[0, 0], [0, 1]]]
    lateral_pts = [[[0, 1], [0, 2]]]

    with pytest.raises(TypeError):
        associate_lateral_to_primary(primary_pts, lateral_pts)


def test_associate_incorrect_dimensions():
    # Tests the function raises a ValueError when input dimensions are incorrect.
    primary_pts = np.array([[0, 0], [0, 1]])  # Missing a dimension
    lateral_pts = np.array([[[0, 1], [0, 2]]])

    with pytest.raises(ValueError):
        associate_lateral_to_primary(primary_pts, lateral_pts)


def test_associate_incorrect_coordinate_dimensions():
    # Tests that the function handles incorrect coordinate dimensions.
    primary_pts = np.array([[[0, 0, 0], [0, 1, 1]]])
    lateral_pts = np.array([[[0, 1, 1], [0, 2, 2]]])

    with pytest.raises(ValueError):
        associate_lateral_to_primary(primary_pts, lateral_pts)


def test_flatten_associated_points_single_primary_no_lateral():
    # Given a single primary root with no lateral roots,
    # the function should return a dictionary with a flattened array of the primary points.
    associations = {0: []}
    primary_pts = np.array([[[1, 2], [3, 4]]])
    expected = {0: np.array([1, 2, 3, 4])}
    # When
    result = flatten_associated_points(associations, primary_pts)
    # Then
    np.testing.assert_array_equal(result[0], expected[0])


def test_flatten_associated_points_single_primary_single_lateral():
    # Given a single primary root with one lateral root,
    # the function should return a flattened array combining both primary and lateral points.
    associations = {0: [np.array([[5, 6], [7, 8]])]}
    primary_pts = np.array([[[1, 2], [3, 4]]])
    expected = {0: np.array([1, 2, 3, 4, 5, 6, 7, 8])}
    # When
    result = flatten_associated_points(associations, primary_pts)
    # Then
    np.testing.assert_array_equal(result[0], expected[0])


def test_flatten_associated_points_multiple_primaries_multiple_laterals():
    # Given multiple primary roots, each with one or more lateral roots,
    # the function should return a dictionary with keys as primary root indices
    # and values as flattened arrays of their associated primary and lateral points.
    associations = {
        0: [np.array([[5, 6], [7, 8]])],
        1: [np.array([[9, 10], [11, 12]]), np.array([[13, 14], [15, 16]])],
    }
    primary_pts = np.array([[[1, 2], [3, 4]], [[17, 18], [19, 20]]])
    expected = {
        0: np.array([1, 2, 3, 4, 5, 6, 7, 8]),
        1: np.array([17, 18, 19, 20, 9, 10, 11, 12, 13, 14, 15, 16]),
    }
    # When
    result = flatten_associated_points(associations, primary_pts)
    # Then
    for key in expected:
        np.testing.assert_array_equal(result[key], expected[key])


def test_flatten_associated_points_empty_input():
    # Given an empty dictionary for associations and an empty array for primary points,
    # the function should return an empty dictionary.
    associations = {}
    primary_pts = np.array([])
    expected = {}
    # When
    result = flatten_associated_points(associations, primary_pts)
    # Then
    assert result == expected


def test_flatten_associated_points_invalid_association_key():
    # Given an invalid association key that does not exist in the primary points,
    # the function should raise an IndexError.
    associations = {2: [np.array([[5, 6], [7, 8]])]}
    primary_pts = np.array([[[1, 2], [3, 4]]])
    # When / Then
    with pytest.raises(IndexError):
        _ = flatten_associated_points(associations, primary_pts)


@pytest.mark.parametrize(
    "associations, primary_pts, expected",
    [
        ({0: [np.array([[5, 6]])]}, np.array([[[1, 2]]]), {0: np.array([1, 2, 5, 6])}),
        ({}, np.array([]), {}),
    ],
)
def test_flatten_associated_points_parametrized(associations, primary_pts, expected):
    # This parametrized test checks the function with various combinations
    # of associations and primary points.
    # When
    result = flatten_associated_points(associations, primary_pts)
    # Then
    for key in expected:
        np.testing.assert_array_equal(result[key], expected[key])
