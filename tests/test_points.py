import numpy as np
import pytest
from shapely.geometry import LineString
from sleap_roots import Series
from sleap_roots.lengths import get_max_length_pts
from sleap_roots.points import get_count, join_pts
from sleap_roots.points import (
    get_all_pts_array,
    get_nodes,
    get_left_right_normalized_vectors,
    get_left_normalized_vector,
    get_right_normalized_vector,
    get_line_equation_from_points,
    associate_lateral_to_primary,
    flatten_associated_points,
    filter_roots_with_nans,
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


def test_single_instance():
    # Single instance with two nodes
    pts = np.array([[1, 2], [3, 4]])
    node_index = 1
    expected_output = np.array([3, 4])
    assert np.array_equal(get_nodes(pts, node_index), expected_output)


def test_multiple_instances():
    # Multiple instances, each with two nodes
    pts = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    node_index = 0
    expected_output = np.array([[1, 2], [5, 6]])
    assert np.array_equal(get_nodes(pts, node_index), expected_output)


def test_node_index_out_of_bounds():
    # Test with node_index out of bounds for the given points
    pts = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    node_index = 2  # Out of bounds
    with pytest.raises(ValueError):
        get_nodes(pts, node_index)


def test_invalid_shape():
    # Test with invalid shape (not 2 or 3 dimensions)
    pts = np.array([1, 2, 3])  # Invalid shape
    node_index = 0
    with pytest.raises(ValueError):
        get_nodes(pts, node_index)


def test_return_shape_single_instance():
    # Single instance input should return shape (2,)
    pts = np.array([[1, 2], [3, 4]])
    node_index = 0
    output = get_nodes(pts, node_index)
    assert output.shape == (2,)


def test_return_shape_multiple_instances():
    # Multiple instances input should return shape (instances, 2)
    pts = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    node_index = 0
    output = get_nodes(pts, node_index)
    assert output.shape == (2, 2)


def test_valid_input_vectors():
    """Test the get_left_right_normalized_vectors function with valid input arrays
    where normalization is straightforward.
    """
    r0_pts = np.array([[0, 1], [2, 3], [4, 5]])
    r1_pts = np.array([[1, 2], [3, 4], [5, 6]])
    expected_left_vector = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])
    expected_right_vector = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])

    norm_vector_left, norm_vector_right = get_left_right_normalized_vectors(
        r0_pts, r1_pts
    )

    assert np.allclose(
        norm_vector_left, expected_left_vector
    ), "Left vector normalization failed"
    assert np.allclose(
        norm_vector_right, expected_right_vector
    ), "Right vector normalization failed"


def test_zero_length_vector_vectors():
    """Test the get_left_right_normalized_vectors function with inputs that result in
    a zero-length vector, expecting vectors filled with NaNs.
    """
    r0_pts = np.array([[0, 0], [0, 0]])
    r1_pts = np.array([[0, 0], [0, 0]])

    norm_vector_left, norm_vector_right = get_left_right_normalized_vectors(
        r0_pts, r1_pts
    )

    assert np.isnan(
        norm_vector_left
    ).all(), "Left vector should be NaN for zero-length vector"
    assert np.isnan(
        norm_vector_right
    ).all(), "Right vector should be NaN for zero-length vector"


def test_invalid_input_shapes_vectors():
    """Test the get_left_right_normalized_vectors function with inputs that have
    mismatched shapes, expecting vectors filled with NaNs.
    """
    r0_pts = np.array([[0, 1]])
    r1_pts = np.array([[1, 2], [3, 4]])

    norm_vector_left, norm_vector_right = get_left_right_normalized_vectors(
        r0_pts, r1_pts
    )

    assert np.isnan(
        norm_vector_left
    ).all(), "Left vector should be NaN for invalid input shapes"
    assert np.isnan(
        norm_vector_right
    ).all(), "Right vector should be NaN for invalid input shapes"


def test_single_instance_input_vectors():
    """Test the get_left_right_normalized_vectors function with a single instance,
    which should return vectors filled with NaNs since the function requires
    more than one instance for comparison.
    """
    r0_pts = np.array([[0, 1]])
    r1_pts = np.array([[1, 2]])

    norm_vector_left, norm_vector_right = get_left_right_normalized_vectors(
        r0_pts, r1_pts
    )

    assert np.isnan(
        norm_vector_left
    ).all(), "Left vector should be NaN for single instance input"
    assert np.isnan(
        norm_vector_right
    ).all(), "Right vector should be NaN for single instance input"


def test_get_left_normalized_vector_with_valid_input():
    """
    Test get_left_normalized_vector with a valid pair of normalized vectors.
    """
    left_vector = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])
    right_vector = np.array([-1 / np.sqrt(2), 1 / np.sqrt(2)])
    normalized_vectors = (left_vector, right_vector)

    result = get_left_normalized_vector(normalized_vectors)
    assert np.allclose(
        result, left_vector
    ), "The left vector was not returned correctly."


def test_get_right_normalized_vector_with_valid_input():
    """
    Test get_right_normalized_vector with a valid pair of normalized vectors.
    """
    left_vector = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])
    right_vector = np.array([-1 / np.sqrt(2), 1 / np.sqrt(2)])
    normalized_vectors = (left_vector, right_vector)

    result = get_right_normalized_vector(normalized_vectors)
    assert np.allclose(
        result, right_vector
    ), "The right vector was not returned correctly."


def test_get_left_normalized_vector_with_nan():
    """
    Test get_left_normalized_vector when the left vector is filled with NaNs.
    """
    left_vector = np.array([np.nan, np.nan])
    right_vector = np.array([1, 0])
    normalized_vectors = (left_vector, right_vector)

    result = get_left_normalized_vector(normalized_vectors)
    assert np.isnan(result).all(), "Expected a vector of NaNs for the left side."


def test_get_right_normalized_vector_with_nan():
    """
    Test get_right_normalized_vector when the right vector is filled with NaNs.
    """
    left_vector = np.array([0, 1])
    right_vector = np.array([np.nan, np.nan])
    normalized_vectors = (left_vector, right_vector)

    result = get_right_normalized_vector(normalized_vectors)
    assert np.isnan(result).all(), "Expected a vector of NaNs for the right side."


def test_normalized_vectors_with_empty_arrays():
    """
    Test get_left_normalized_vector and get_right_normalized_vector with empty arrays.
    """
    left_vector = np.array([])
    right_vector = np.array([])
    normalized_vectors = (left_vector, right_vector)

    left_result = get_left_normalized_vector(normalized_vectors)
    right_result = get_right_normalized_vector(normalized_vectors)

    assert (
        left_result.size == 0 and right_result.size == 0
    ), "Expected empty arrays for both left and right vectors."


@pytest.mark.parametrize(
    "pts1, pts2, expected",
    [
        (np.array([0, 0]), np.array([1, 1]), (1, 0)),  # Diagonal line, positive slope
        (np.array([1, 1]), np.array([2, 2]), (1, 0)),  # Diagonal line, positive slope
        (np.array([0, 1]), np.array([1, 0]), (-1, 1)),  # Diagonal line, negative slope
        (np.array([1, 2]), np.array([3, 2]), (0, 2)),  # Horizontal line
        (
            np.array([2, 3]),
            np.array([2, 5]),
            (np.nan, np.nan),
        ),  # Vertical line should return NaNs
        (
            np.array([0, 0]),
            np.array([0, 0]),
            (np.nan, np.nan),
        ),  # Identical points should return NaNs
    ],
)
def test_get_line_equation_from_points(pts1, pts2, expected):
    m, b = get_line_equation_from_points(pts1, pts2)
    assert np.isclose(m, expected[0], equal_nan=True) and np.isclose(
        b, expected[1], equal_nan=True
    ), f"Expected slope {expected[0]} and intercept {expected[1]} but got slope {m} and intercept {b}"


@pytest.mark.parametrize(
    "pts1, pts2",
    [
        (np.array([1]), np.array([1, 2])),  # Incorrect shape
        (5, np.array([1, 2])),  # Not an array
        ("test", "test"),  # Incorrect type
    ],
)
def test_get_line_equation_input_errors(pts1, pts2):
    with pytest.raises(ValueError):
        get_line_equation_from_points(pts1, pts2)


import numpy as np

def test_associate_basic():
    # Tests basic association between one primary and one lateral root.
    primary_pts = np.array([[[0, 0], [0, 1]]])
    lateral_pts = np.array([[[0, 1], [0, 2]]])

    expected = {0: {'primary_points': primary_pts, 'lateral_points': lateral_pts}}
    result = associate_lateral_to_primary(primary_pts, lateral_pts)

    # Ensure the keys match
    assert set(result.keys()) == set(expected.keys())

    # Loop through the result and the expected dictionary to compare the numpy arrays within
    for key in expected:
        # Ensure both dictionaries have the same keys (e.g., 'primary_points', 'lateral_points')
        assert set(result[key].keys()) == set(expected[key].keys())
        
        # Now compare the NumPy arrays for each key within the dictionaries
        for sub_key in expected[key]:
            np.testing.assert_array_equal(result[key][sub_key], expected[key][sub_key])



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


def test_filter_roots_with_nans_no_nans():
    """Test with an array that contains no NaN values."""
    pts = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    expected = pts
    result = filter_roots_with_nans(pts)
    np.testing.assert_array_equal(result, expected)

def test_filter_roots_with_nans_nan_in_one_instance():
    """Test with an array where one instance contains NaN values."""
    pts = np.array([[[1, 2], [3, 4]], [[np.nan, 6], [7, 8]]])
    expected = np.array([[[1, 2], [3, 4]]])
    result = filter_roots_with_nans(pts)
    np.testing.assert_array_equal(result, expected)

def test_filter_roots_with_nans_all_nans_in_one_instance():
    """Test with an array where one instance is entirely NaN."""
    pts = np.array([[[np.nan, np.nan], [np.nan, np.nan]], [[5, 6], [7, 8]]])
    expected = np.array([[[5, 6], [7, 8]]])
    result = filter_roots_with_nans(pts)
    np.testing.assert_array_equal(result, expected)

def test_filter_roots_with_nans_nan_across_multiple_instances():
    """Test with NaN values scattered across multiple instances."""
    pts = np.array([[[1, np.nan], [3, 4]], [[5, 6], [np.nan, 8]], [[9, 10], [11, 12]]])
    expected = np.array([[[9, 10], [11, 12]]])
    result = filter_roots_with_nans(pts)
    np.testing.assert_array_equal(result, expected)

def test_filter_roots_with_nans_all_instances_contain_nans():
    """Test with an array where all instances contain at least one NaN value."""
    pts = np.array([[[np.nan, 2], [3, 4]], [[5, np.nan], [7, 8]], [[9, 10], [np.nan, 12]]])
    expected = np.empty((0, pts.shape[1], 2))
    result = filter_roots_with_nans(pts)
    np.testing.assert_array_equal(result, expected)

def test_filter_roots_with_nans_empty_array():
    """Test with an empty array."""
    pts = np.array([])
    expected = np.empty((0, 0, 2))
    result = filter_roots_with_nans(pts)
    np.testing.assert_array_equal(result, expected)

def test_filter_roots_with_nans_single_instance_with_nans():
    """Test with a single instance that contains NaN values."""
    pts = np.array([[[np.nan, np.nan], [np.nan, np.nan]]])
    expected = np.empty((0, pts.shape[1], 2))
    result = filter_roots_with_nans(pts)
    np.testing.assert_array_equal(result, expected)

def test_filter_roots_with_nans_single_instance_without_nans():
    """Test with a single instance that does not contain NaN values."""
    pts = np.array([[[1, 2], [3, 4]]])
    expected = pts
    result = filter_roots_with_nans(pts)
    np.testing.assert_array_equal(result, expected)
