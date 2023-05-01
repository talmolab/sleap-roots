from sleap_roots.bases import (
    get_bases,
    get_base_tip_dist,
    get_grav_index,
    get_lateral_count,
    get_root_lengths,
    get_root_lengths_max,
    get_base_xs,
    get_base_ys,
    get_base_length,
    get_root_pair_widths_projections,
)
from sleap_roots.points import get_lateral_pts
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
def lengths_normal():
    return np.array([145, 234, 329.4])


@pytest.fixture
def lengths_with_nan():
    return np.array([145, 234, 329.4, np.nan])


@pytest.fixture
def lengths_all_nan():
    return np.array([np.nan, np.nan, np.nan])


def test_bases_standard(pts_standard):
    bases = get_bases(pts_standard)
    assert bases.shape == (2, 2)
    np.testing.assert_array_equal(bases, [[1, 2], [5, 6]])


def test_bases_no_bases(pts_no_bases):
    bases = get_bases(pts_no_bases)
    assert bases.shape == (2, 2)
    np.testing.assert_array_equal(bases, [[np.nan, np.nan], [np.nan, np.nan]])


def test_bases_one_base(pts_one_base):
    bases = get_bases(pts_one_base)
    assert bases.shape == (2, 2)
    np.testing.assert_array_equal(bases, [[1, 2], [np.nan, np.nan]])


def test_bases_no_roots(pts_no_roots):
    bases = get_bases(pts_no_roots)
    assert bases.shape == (2, 2)
    np.testing.assert_array_equal(bases, [[np.nan, np.nan], [np.nan, np.nan]])


# test get_base_tip_dist with standard points
def test_get_base_tip_dist_standard(pts_standard):
    distance = get_base_tip_dist(pts_standard)
    assert distance.shape == (2,)
    np.testing.assert_almost_equal(distance, [2.82842712, 2.82842712], decimal=7)


# test get_base_tip_dist with roots without bases
def test_get_base_tip_dist_no_bases(pts_no_bases):
    distance = get_base_tip_dist(pts_no_bases)
    assert distance.shape == (2,)
    np.testing.assert_almost_equal(distance, [np.nan, np.nan], decimal=7)


# test get_base_tip_dist with roots with one base
def test_get_base_tip_dist_one_base(pts_one_base):
    distance = get_base_tip_dist(pts_one_base)
    assert distance.shape == (2,)
    np.testing.assert_almost_equal(distance, [2.82842712, np.nan], decimal=7)


# test get_base_tip_dist with no roots
def test_get_base_tip_dist_no_roots(pts_no_roots):
    distance = get_base_tip_dist(pts_no_roots)
    assert distance.shape == (2,)
    np.testing.assert_almost_equal(distance, [np.nan, np.nan], decimal=7)


# test get_grav_index function
def test_get_grav_index(canola_h5):
    series = Series.load(
        canola_h5, primary_name="primary_multi_day", lateral_name="lateral_3_nodes"
    )
    primary, lateral = series[0]
    pts = primary.numpy()
    grav_index = get_grav_index(pts)
    np.testing.assert_almost_equal(grav_index, 0.08898137324716636)


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


# test get_lateral_count function with canola
def test_get_lateral_count(canola_h5):
    series = Series.load(
        canola_h5, primary_name="primary_multi_day", lateral_name="lateral_3_nodes"
    )
    primary, lateral = series[0]
    lateral_pts = lateral.numpy()
    lateral_count = get_lateral_count(lateral_pts)
    assert lateral_count == 5


# test get_base_xs with canola
def test_get_base_xs_canola(canola_h5):
    plant = Series.load(
        canola_h5, primary_name="primary_multi_day", lateral_name="lateral_3_nodes"
    )
    pts_lr = get_lateral_pts(plant=plant, frame=0)
    base_xs = get_base_xs(pts_lr)
    assert base_xs.shape[0] == 5
    np.testing.assert_almost_equal(base_xs[1], 1112.5506591796875, decimal=3)


# test get_base_xs with pts_standard
def test_get_base_xs_standard(pts_standard):
    base_xs = get_base_xs(pts_standard)
    assert base_xs.shape[0] == 2
    np.testing.assert_almost_equal(base_xs[0], 1, decimal=3)
    np.testing.assert_almost_equal(base_xs[1], 5, decimal=3)


# test get_base_xs with pts_no_roots
def test_get_base_xs_no_roots(pts_no_roots):
    base_xs = get_base_xs(pts_no_roots)
    assert base_xs.shape[0] == 2
    np.testing.assert_almost_equal(base_xs[0], np.nan, decimal=3)


# test get_base_ys with canola
def test_get_base_ys_canola(canola_h5):
    plant = Series.load(
        canola_h5, primary_name="primary_multi_day", lateral_name="lateral_3_nodes"
    )
    pts_lr = get_lateral_pts(plant=plant, frame=0)
    base_ys = get_base_ys(pts_lr)
    assert base_ys.shape[0] == 5
    np.testing.assert_almost_equal(base_ys[1], 228.0966796875, decimal=3)


# test get_base_ys with pts_standard
def test_get_base_ys_standard(pts_standard):
    base_ys = get_base_ys(pts_standard)
    assert base_ys.shape[0] == 2
    np.testing.assert_almost_equal(base_ys[0], 2, decimal=3)
    np.testing.assert_almost_equal(base_ys[1], 6, decimal=3)


# test get_base_ys with pts_no_roots
def test_get_base_ys_no_roots(pts_no_roots):
    base_ys = get_base_ys(pts_no_roots)
    assert base_ys.shape[0] == 2
    np.testing.assert_almost_equal(base_ys[0], np.nan, decimal=3)


# test get_base_length with canola
def test_get_base_length_canola(canola_h5):
    plant = Series.load(
        canola_h5, primary_name="primary_multi_day", lateral_name="lateral_3_nodes"
    )
    pts_lr = get_lateral_pts(plant=plant, frame=0)
    base_length = get_base_length(pts_lr)
    np.testing.assert_almost_equal(base_length, 83.69914245605469, decimal=3)


# test get_base_length with pts_standard
def test_get_base_length_standard(pts_standard):
    base_length = get_base_length(pts_standard)
    np.testing.assert_almost_equal(base_length, 4, decimal=3)


# test get_base_length with pts_no_roots
def test_get_base_length_no_roots(pts_no_roots):
    base_length = get_base_length(pts_no_roots)
    np.testing.assert_almost_equal(base_length, np.nan, decimal=3)


def test_stem_width(canola_h5):
    series = Series.load(
        canola_h5, primary_name="primary_multi_day", lateral_name="lateral_3_nodes"
    )
    primary, lateral = series[0]
    primary_pts = primary.numpy()
    lateral_pts = lateral.numpy()
    assert primary_pts.shape == (1, 6, 2)
    assert lateral_pts.shape == (5, 3, 2)

    stem_widths = get_root_pair_widths_projections(lateral_pts, primary_pts, 0.02)
    np.testing.assert_array_almost_equal(stem_widths, [[31.603239], [1], [0]])
