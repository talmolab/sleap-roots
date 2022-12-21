from sleap_roots.bases import get_bases, get_root_lengths
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


def test_bases_standard(pts_standard):
    bases = get_bases(pts_standard)
    assert bases.shape == (2, 2)
    np.testing.assert_array_equal(bases, [[1, 2], [5, 6]])


def test_bases_no_bases(pts_no_bases):
    bases = get_bases(pts_no_bases)
    assert bases.shape == (0, 2)
    np.testing.assert_array_equal(bases, np.empty((0, 2)))


def test_bases_one_base(pts_one_base):
    bases = get_bases(pts_one_base)
    assert bases.shape == (1, 2)
    np.testing.assert_array_equal(bases, [[1, 2]])


def test_bases_no_roots(pts_no_roots):
    bases = get_bases(pts_no_roots)
    assert bases.shape == (0, 2)
    np.testing.assert_array_equal(bases, np.empty((0, 2)))


def test_get_root_lengths(canola_h5):
    series = Series.load(
        canola_h5, primary_name="primary_multi_day", lateral_name="lateral_3_nodes"
    )
    primary, lateral = series[0]
    pts = primary.numpy()
    assert pts.shape == (1, 6, 2)

    root_lengths = get_root_lengths(pts)
    assert root_lengths.shape == (1,)
    np.testing.assert_array_almost_equal(root_lengths, [971.050417])

    pts = lateral.numpy()
    assert pts.shape == (5, 3, 2)

    root_lengths = get_root_lengths(pts)
    assert root_lengths.shape == (5,)
    np.testing.assert_array_almost_equal(
        root_lengths, [20.129579, 62.782368, 80.268003, 34.925591, 3.89724]
    )


def test_get_root_lengths_one_point(pts_no_bases):
    root_lengths = get_root_lengths(pts_no_bases)
    assert root_lengths.shape == (2,)
    np.testing.assert_array_almost_equal(root_lengths, np.array([np.nan, np.nan]))


def test_get_root_lengths_no_roots():
    pts = np.array(
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

    root_lengths = get_root_lengths(pts)
    assert root_lengths.shape == (2,)
    np.testing.assert_array_almost_equal(root_lengths, np.array([np.nan, np.nan]))
