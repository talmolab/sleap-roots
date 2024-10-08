from sleap_roots.lengths import (
    get_curve_index,
    get_root_lengths,
    get_max_length_pts,
    get_min_distance_line_to_line,
)
from sleap_roots.bases import get_base_tip_dist, get_bases
from sleap_roots.tips import get_tips
from sleap_roots import Series
from shapely.geometry import LineString
import numpy as np
import pytest


@pytest.fixture
def pts_standard():
    return np.array(
        [
            [
                [1, 2],
                [3, 4],
            ],
            [
                [5, 6],
                [7, 8],
            ],
        ]
    )


@pytest.fixture
def pts_no_bases():
    return np.array(
        [
            [
                [np.nan, np.nan],
                [3, 4],
            ],
            [
                [np.nan, np.nan],
                [7, 8],
            ],
        ]
    )


@pytest.fixture
def pts_one_base():
    return np.array(
        [
            [
                [1, 2],
                [3, 4],
            ],
            [
                [np.nan, np.nan],
                [7, 8],
            ],
        ]
    )


@pytest.fixture
def pts_no_roots():
    return np.array(
        [
            [
                [np.nan, np.nan],
                [np.nan, np.nan],
            ],
            [
                [np.nan, np.nan],
                [np.nan, np.nan],
            ],
        ]
    )


@pytest.fixture
def pts_not_contiguous():
    return np.array(
        [
            [
                [1, 2],
                [np.nan, np.nan],
                [3, 4],
            ],
            [
                [5, 6],
                [np.nan, np.nan],
                [7, 8],
            ],
        ]
    )


@pytest.fixture
def primary_pts():
    return np.array(
        [
            [
                [852.17755127, 216.95648193],
                [850.17755127, 472.83520508],
                [844.45300293, 472.83520508],
                [837.03405762, 588.5123291],
                [828.87963867, 692.72009277],
                [816.71142578, 808.12585449],
            ]
        ]
    )


@pytest.fixture
def lateral_pts():
    return np.array(
        [
            [
                [852.17755127, 216.95648193],
                [np.nan, np.nan],
                [837.03405762, 588.5123291],
                [828.87963867, 692.72009277],
                [816.71142578, 808.12585449],
            ],
            [
                [852.17755127, 216.95648193],
                [844.45300293, 472.83520508],
                [837.03405762, 588.5123291],
                [828.87963867, 692.72009277],
                [816.71142578, 808.12585449],
            ],
        ]
    )


@pytest.fixture
def lengths_normal():
    return np.array([145, 234, 329.4])


@pytest.fixture
def lengths_with_nan():
    return np.array([145, 234, 329.4, np.nan])


@pytest.fixture
def lengths_all_nan():
    return np.array([np.nan, np.nan, np.nan])


def test_min_distance_line_to_line():
    # Test with non-intersecting lines
    line1 = LineString([(0, 0), (1, 1)])
    line2 = LineString([(1, 0), (2, 0)])
    assert get_min_distance_line_to_line(line1, line2) == np.sqrt(2) / 2

    # Test with intersecting lines (expect 0 distance)
    line1 = LineString([(0, 0), (1, 1)])
    line2 = LineString([(0, 1), (1, 0)])
    assert get_min_distance_line_to_line(line1, line2) == 0

    # Test with parallel lines
    line1 = LineString([(0, 0), (1, 0)])
    line2 = LineString([(0, 1), (1, 1)])
    assert get_min_distance_line_to_line(line1, line2) == 1

    # Test with invalid input types
    with pytest.raises(TypeError):
        get_min_distance_line_to_line("not a linestring", LineString([(0, 0), (1, 1)]))
    with pytest.raises(TypeError):
        get_min_distance_line_to_line(LineString([(0, 0), (1, 1)]), "not a linestring")


# tests for get_curve_index function
def test_get_curve_index_canola(canola_h5, canola_primary_slp, canola_lateral_slp):
    # Set the frame index to 0
    frame_idx = 0
    # Load the series from the canola dataset
    series = Series.load(
        series_name="canola_test",
        h5_path=canola_h5,
        primary_path=canola_primary_slp,
        lateral_path=canola_lateral_slp,
    )
    # Get the primary points from the first frame
    primary_pts = series.get_primary_points(frame_idx)
    # Get the points from the root with maximum length
    primary_pts = get_max_length_pts(primary_pts)
    # Get the length of the root with maximum length
    primary_length = get_root_lengths(primary_pts)
    # Get the bases and tips of the root
    bases = get_bases(primary_pts)
    tips = get_tips(primary_pts)
    # Get the distance between the base and the tip of the root
    base_tip_dist = get_base_tip_dist(bases, tips)
    # Get the curvature index of the root
    curve_index = get_curve_index(primary_length, base_tip_dist)
    np.testing.assert_almost_equal(curve_index, 0.08898137324716636)


def test_get_curve_index():
    # Test 1: Scalar inputs where length > base_tip_dist
    # Curvature index should be (10 - 8) / 10 = 0.2
    assert get_curve_index(10, 8) == 0.2

    # Test 2: Scalar inputs where length and base_tip_dist are zero
    # Should return NaN as length is zero
    assert np.isnan(get_curve_index(0, 0))

    # Test 3: Scalar inputs where length < base_tip_dist
    # Should return NaN as it's an invalid case
    assert np.isnan(get_curve_index(5, 10))

    # Test 4: Array inputs covering various cases
    # Case 1: length > base_tip_dist, should return 0.2
    # Case 2: length = 0, should return NaN
    # Case 3: length < base_tip_dist, should return NaN
    # Case 4: length > base_tip_dist, should return 0.2
    lengths = np.array([10, 0, 5, 15])
    base_tip_dists = np.array([8, 0, 10, 12])
    expected = np.array([0.2, np.nan, np.nan, 0.2])
    result = get_curve_index(lengths, base_tip_dists)
    assert np.allclose(result, expected, equal_nan=True)

    # Test 5: Mismatched shapes between lengths and base_tip_dists
    # Should raise a ValueError
    with pytest.raises(ValueError):
        get_curve_index(np.array([10, 20]), np.array([8]))

    # Test 6: Array inputs with NaN values
    # Case 1: length > base_tip_dist, should return 0.2
    # Case 2 and 3: either length or base_tip_dist is NaN, should return NaN
    lengths = np.array([10, np.nan, np.nan])
    base_tip_dists = np.array([8, 8, np.nan])
    expected = np.array([0.2, np.nan, np.nan])
    result = get_curve_index(lengths, base_tip_dists)
    assert np.allclose(result, expected, equal_nan=True)


def test_get_curve_index_shape():
    # Check if scalar inputs result in scalar output
    result = get_curve_index(10, 8)
    assert isinstance(
        result, (int, float)
    ), f"Expected scalar output, got {type(result)}"

    # Check if array inputs result in array output
    lengths = np.array([10, 15])
    base_tip_dists = np.array([8, 12])
    result = get_curve_index(lengths, base_tip_dists)
    assert isinstance(
        result, np.ndarray
    ), f"Expected np.ndarray output, got {type(result)}"

    # Check the shape of output for array inputs
    # Should match the shape of the input arrays
    assert (
        result.shape == lengths.shape
    ), f"Output shape {result.shape} does not match input shape {lengths.shape}"

    # Check the shape of output for larger array inputs
    lengths = np.array([10, 15, 20, 25])
    base_tip_dists = np.array([8, 12, 18, 22])
    result = get_curve_index(lengths, base_tip_dists)
    assert (
        result.shape == lengths.shape
    ), f"Output shape {result.shape} does not match input shape {lengths.shape}"


def test_nan_values():
    lengths = np.array([10, np.nan, 30])
    base_tip_dists = np.array([8, 16, np.nan])
    np.testing.assert_array_equal(
        get_curve_index(lengths, base_tip_dists), np.array([0.2, np.nan, np.nan])
    )


def test_zero_lengths():
    lengths = np.array([0, 20, 30])
    base_tip_dists = np.array([0, 16, 24])
    np.testing.assert_array_equal(
        get_curve_index(lengths, base_tip_dists), np.array([np.nan, 0.2, 0.2])
    )


def test_invalid_scalar_values():
    assert np.isnan(get_curve_index(np.nan, 8))
    assert np.isnan(get_curve_index(10, np.nan))
    assert np.isnan(get_curve_index(0, 8))


def test_curve_index_float():
    assert get_curve_index(10.0, 5.0) == 0.5


def test_curve_index_float_invalid():
    assert np.isnan(get_curve_index(np.nan, 5.0))


def test_curve_index_array():
    lengths = np.array([10, 20, 30, 0, np.nan])
    base_tip_dists = np.array([5, 15, 25, 0, np.nan])
    expected = np.array([0.5, 0.25, 0.16666667, np.nan, np.nan])
    np.testing.assert_allclose(
        get_curve_index(lengths, base_tip_dists), expected, rtol=1e-6
    )


def test_curve_index_mixed_invalid():
    lengths = np.array([10, np.nan, 0])
    base_tip_dists = np.array([5, 5, 5])
    expected = np.array([0.5, np.nan, np.nan])
    np.testing.assert_allclose(
        get_curve_index(lengths, base_tip_dists), expected, rtol=1e-6
    )


def test_get_root_lengths(canola_h5, canola_primary_slp, canola_lateral_slp):
    # Set the frame index to 0
    frame_idx = 0
    # Load the series from the canola dataset
    series = Series.load(
        series_name="canola_test",
        h5_path=canola_h5,
        primary_path=canola_primary_slp,
        lateral_path=canola_lateral_slp,
    )
    # Get the primary points from the first frame
    primary_pts = series.get_primary_points(frame_idx)
    assert primary_pts.shape == (1, 6, 2)
    # Get the root lengths
    root_lengths = get_root_lengths(primary_pts)
    assert np.isscalar(root_lengths)
    np.testing.assert_array_almost_equal(root_lengths, [971.050417])
    # Get the lateral points from the first frame
    lateral_pts = series.get_lateral_points(frame_idx)
    assert lateral_pts.shape == (5, 3, 2)
    # Get the root lengths
    root_lengths = get_root_lengths(lateral_pts)
    assert root_lengths.shape == (5,)
    np.testing.assert_array_almost_equal(
        root_lengths, [20.129579, 62.782368, 80.268003, 34.925591, 3.89724]
    )


def test_get_root_lengths_no_roots(pts_no_bases):
    root_lengths = get_root_lengths(pts_no_bases)
    assert root_lengths.shape == (2,)
    np.testing.assert_array_almost_equal(root_lengths, np.array([np.nan, np.nan]))


def test_get_root_lengths_one_point(pts_one_base):
    root_lengths = get_root_lengths(pts_one_base)
    assert root_lengths.shape == (2,)
    np.testing.assert_array_almost_equal(
        root_lengths, np.array([2.82842712475, np.nan])
    )


def test_get_max_length_pts(canola_h5, canola_primary_slp, canola_lateral_slp):
    # Set the frame index to 0
    frame_idx = 0
    # Load the series from the canola dataset
    series = Series.load(
        series_name="canola_test",
        h5_path=canola_h5,
        primary_path=canola_primary_slp,
        lateral_path=canola_lateral_slp,
    )
    # Get the primary points from the first frame
    primary_pts = series.get_primary_points(frame_idx)
    # Get the points from the root with maximum length
    max_length_pts = get_max_length_pts(primary_pts)
    assert max_length_pts.shape == (6, 2)
    np.testing.assert_almost_equal(
        max_length_pts[0], np.array([1016.7844238, 144.4191589])
    )
