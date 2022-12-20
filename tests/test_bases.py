from sleap_roots.bases import get_bases, get_root_lengths
from sleap_roots import Series
import numpy as np
import pytest

pts_standard = np.array(
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

pts_no_bases = np.array(
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


pts_one_base = np.array(
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


pts_no_roots = np.array(
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


@pytest.mark.parametrize(
    "test_input,expected_shape,expected_array",
    [
        (pts_standard, (2, 2), [[1, 2], [5, 6]]),
        (pts_no_bases, (0, 2), np.empty((0, 2))),
        (pts_one_base, (1, 2), [[1, 2]]),
        (pts_no_roots, (0, 2), np.empty((0, 2))),
    ],
)
def test_bases(test_input, expected):
    bases = get_bases(test_input)
    assert bases.shape == expected_shape
    np.testing.assert_array_equal(bases, expected_array)


def test_get_bases_standard():
    pts = np.array(
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

    bases = get_bases(pts)
    assert bases.shape == (2, 2)
    np.testing.assert_array_equal(bases, [[1, 2], [5, 6]])


def test_get_bases_no_bases():
    pts = np.array(
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

    bases = get_bases(pts)
    assert bases.shape == (0, 2)
    np.testing.assert_array_equal(bases, np.empty((0, 2)))


def test_get_bases_one_base():
    pts = np.array(
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

    bases = get_bases(pts)
    assert bases.shape == (1, 2)
    np.testing.assert_array_equal(bases, [[1, 2]])


def test_get_bases_no_roots():
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

    bases = get_bases(pts)
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

    assert pts.shape == (2, 2, 2)

    root_lengths = get_root_lengths(pts)
    assert root_lengths.shape == (2,)
    np.testing.assert_array_almost_equal(root_lengths, [0, 0])
