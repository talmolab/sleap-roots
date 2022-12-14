from sleap_roots.bases import get_bases, get_root_lengths
from sleap_roots import Series
import numpy as np


def test_get_bases():
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
