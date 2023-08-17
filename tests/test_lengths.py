from sleap_roots.lengths import (
    get_grav_index,
    get_root_lengths,
    get_root_lengths_max,
    get_max_length_pts,
)
from sleap_roots.bases import get_base_tip_dist, get_bases
from sleap_roots.tips import get_tips
from sleap_roots import Series
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


# test get_grav_index function
def test_get_grav_index(canola_h5):
    series = Series.load(
        canola_h5, primary_name="primary_multi_day", lateral_name="lateral_3_nodes"
    )
    primary = series[0][0]  # first frame, primary labels
    primary_pts = primary.numpy()  # primary points as numpy array
    primary_length = get_root_lengths_max(primary_pts)
    max_length_pts = get_max_length_pts(primary_pts)
    bases = get_bases(max_length_pts)
    tips = get_tips(max_length_pts)
    base_tip_dist = get_base_tip_dist(bases, tips)
    grav_index = get_grav_index(primary_length, base_tip_dist)
    np.testing.assert_almost_equal(grav_index, 0.08898137324716636)


def test_get_root_lengths(canola_h5):
    series = Series.load(
        canola_h5, primary_name="primary_multi_day", lateral_name="lateral_3_nodes"
    )
    primary, lateral = series[0]
    pts = primary.numpy()
    assert pts.shape == (1, 6, 2)

    root_lengths = get_root_lengths(pts)
    assert np.isscalar(root_lengths)
    np.testing.assert_array_almost_equal(root_lengths, [971.050417])

    pts = lateral.numpy()
    assert pts.shape == (5, 3, 2)

    root_lengths = get_root_lengths(pts)
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


# test get_root_lengths_max function with lengths_normal
def test_get_root_lengths_max_normal(lengths_normal):
    max_length = get_root_lengths_max(lengths_normal)
    np.testing.assert_array_almost_equal(max_length, 329.4)


# test get_root_lengths_max function with lengths_with_nan
def test_get_root_lengths_max_with_nan(lengths_with_nan):
    max_length = get_root_lengths_max(lengths_with_nan)
    np.testing.assert_array_almost_equal(max_length, 329.4)


# test get_root_lengths_max function with lengths_all_nan
def test_get_root_lengths_max_all_nan(lengths_all_nan):
    max_length = get_root_lengths_max(lengths_all_nan)
    np.testing.assert_array_almost_equal(max_length, np.nan)


def test_get_max_length_pts(canola_h5):
    series = Series.load(
        canola_h5, primary_name="primary_multi_day", lateral_name="lateral_3_nodes"
    )
    primary = series[0][0]  # first frame, primary labels
    primary_pts = primary.numpy()  # primary points as numpy array
    max_length_pts = get_max_length_pts(primary_pts)
    assert max_length_pts.shape == (6, 2)
    np.testing.assert_almost_equal(
        max_length_pts[0], np.array([1016.7844238, 144.4191589])
    )
