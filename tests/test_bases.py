from sleap_roots.bases import (
    get_bases,
    get_base_ct_density,
    get_base_tip_dist,
    get_base_xs,
    get_base_ys,
    get_base_length,
    get_base_length_ratio,
    get_root_pair_widths_projections,
)
from sleap_roots.lengths import get_max_length_pts, get_root_lengths_max
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
    primary_pts = pts_standard
    primary_base_pt = get_bases(primary_pts)
    primary_tip_pt = get_tips(primary_pts)
    distance = get_base_tip_dist(primary_base_pt, primary_tip_pt)
    assert distance.shape == (2,)
    np.testing.assert_almost_equal(distance, [2.82842712, 2.82842712], decimal=7)


# test get_base_tip_dist with roots without bases
def test_get_base_tip_dist_no_bases(pts_no_bases):
    primary_pts = pts_no_bases
    primary_base_pt = get_bases(primary_pts)
    primary_tip_pt = get_tips(primary_pts)
    distance = get_base_tip_dist(primary_base_pt, primary_tip_pt)
    assert distance.shape == (2,)
    np.testing.assert_almost_equal(distance, [np.nan, np.nan], decimal=7)


# test get_base_tip_dist with roots with one base
def test_get_base_tip_dist_one_base(pts_one_base):
    primary_pts = pts_one_base
    primary_base_pt = get_bases(primary_pts)
    primary_tip_pt = get_tips(primary_pts)
    distance = get_base_tip_dist(primary_base_pt, primary_tip_pt)
    assert distance.shape == (2,)
    np.testing.assert_almost_equal(distance, [2.82842712, np.nan], decimal=7)


# test get_base_tip_dist with no roots
def test_get_base_tip_dist_no_roots(pts_no_roots):
    primary_pts = pts_no_roots
    primary_base_pt = get_bases(primary_pts)
    primary_tip_pt = get_tips(primary_pts)
    distance = get_base_tip_dist(primary_base_pt, primary_tip_pt)
    assert distance.shape == (2,)
    np.testing.assert_almost_equal(distance, [np.nan, np.nan], decimal=7)


# test get_base_xs with canola
def test_get_base_xs_canola(canola_h5):
    monocots = False
    plant = Series.load(
        canola_h5, primary_name="primary_multi_day", lateral_name="lateral_3_nodes"
    )
    lateral = plant[0][1]  # first frame, lateral labels
    lateral_pts = lateral.numpy()  # lateral points as numpy array
    base_xs = get_base_xs(lateral_pts, monocots)
    assert base_xs.shape[0] == 5
    np.testing.assert_almost_equal(base_xs[1], 1112.5506591796875, decimal=3)


# test get_base_xs with rice
def test_get_base_xs_rice(rice_h5):
    monocots = True
    plant = Series.load(
        rice_h5, primary_name="longest_3do_6nodes", lateral_name="main_3do_6nodes"
    )
    lateral = plant[0][1]  # first frame, lateral labels
    lateral_pts = lateral.numpy()  # lateral points as numpy array
    base_xs = get_base_xs(lateral_pts, monocots)
    assert np.isnan(base_xs)


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
    monocots = False
    plant = Series.load(
        canola_h5, primary_name="primary_multi_day", lateral_name="lateral_3_nodes"
    )
    lateral = plant[0][1]  # first frame, lateral labels
    lateral_pts = lateral.numpy()  # lateral points as numpy array
    base_pts = get_bases(lateral_pts)  # get the bases of the lateral roots
    base_ys = get_base_ys(base_pts, monocots)
    assert base_ys.shape[0] == 5
    np.testing.assert_almost_equal(base_ys[1], 228.0966796875, decimal=3)


# test get_base_ys with rice
def test_get_base_ys_rice(rice_h5):
    monocots = True
    plant = Series.load(
        rice_h5, primary_name="longest_3do_6nodes", lateral_name="main_3do_6nodes"
    )
    lateral = plant[0][1]  # first frame, lateral labels
    lateral_pts = lateral.numpy()  # lateral points as numpy array
    base_pts = get_bases(lateral_pts, monocots)  # get the bases of the lateral roots
    base_ys = get_base_ys(base_pts, monocots)
    assert np.isnan(base_ys)


# test get_base_ys with pts_standard
def test_get_base_ys_standard(pts_standard):
    bases = get_bases(pts_standard)
    base_ys = get_base_ys(bases)
    assert base_ys.shape[0] == 2
    np.testing.assert_almost_equal(base_ys[0], 2, decimal=3)
    np.testing.assert_almost_equal(base_ys[1], 6, decimal=3)


# test get_base_ys with pts_no_roots
def test_get_base_ys_no_roots(pts_no_roots):
    bases = get_bases(pts_no_roots)
    base_ys = get_base_ys(bases)
    assert base_ys.shape[0] == 2
    np.testing.assert_almost_equal(base_ys[0], np.nan, decimal=3)


# test get_base_length with canola
def test_get_base_length_canola(canola_h5):
    plant = Series.load(
        canola_h5, primary_name="primary_multi_day", lateral_name="lateral_3_nodes"
    )
    lateral = plant[0][1]  # first frame, lateral labels
    lateral_pts = lateral.numpy()  # lateral points as numpy array
    bases = get_bases(lateral_pts)  # get bases of lateral roots
    base_ys = get_base_ys(bases)  # get y-coordinates of bases
    base_length = get_base_length(base_ys)
    np.testing.assert_almost_equal(base_length, 83.69914245605469, decimal=3)


# test get_base_length with rice
def test_get_base_length_rice(rice_h5):
    plant = Series.load(
        rice_h5, primary_name="longest_3do_6nodes", lateral_name="main_3do_6nodes"
    )
    lateral = plant[0][1]  # first frame, lateral labels
    lateral_pts = lateral.numpy()  # lateral points as numpy array
    base_length = get_base_length(lateral_pts, monocots=True)
    assert np.isnan(base_length)


# test get_base_length with pts_standard
def test_get_base_length_standard(pts_standard):
    bases = get_bases(pts_standard)  # get bases of lateral roots
    base_ys = get_base_ys(bases)  # get y-coordinates of bases
    base_length = get_base_length(base_ys)
    np.testing.assert_almost_equal(base_length, 4, decimal=3)


# test get_base_length with pts_no_roots
def test_get_base_length_no_roots(pts_no_roots):
    base_length = get_base_length(pts_no_roots)
    assert np.isnan(base_length)


# test get_base_ct_density function with defined primary and lateral points
def test_get_base_ct_density(primary_pts, lateral_pts):
    primary_length_max = get_root_lengths_max(primary_pts)
    lateral_base_pts = get_bases(lateral_pts)
    base_ct_density = get_base_ct_density(primary_length_max, lateral_base_pts)
    np.testing.assert_almost_equal(base_ct_density, 0.00334, decimal=5)


# test get_base_ct_density function with canola example
def test_get_base_ct_density_canola(canola_h5):
    monocots = False
    series = Series.load(
        canola_h5, primary_name="primary_multi_day", lateral_name="lateral_3_nodes"
    )
    primary, lateral = series[0]
    primary_pts = primary.numpy()
    lateral_pts = lateral.numpy()
    primary_length_max = get_root_lengths_max(primary_pts)
    lateral_base_pts = get_bases(lateral_pts)
    base_ct_density = get_base_ct_density(primary_length_max, lateral_base_pts)
    np.testing.assert_almost_equal(base_ct_density, 0.004119, decimal=5)


# test get_base_ct_density function with rice example
def test_get_base_ct_density_rice(rice_h5):
    monocots = True
    series = Series.load(
        rice_h5, primary_name="longest_3do_6nodes", lateral_name="main_3do_6nodes"
    )
    primary, lateral = series[0]
    primary_pts = primary.numpy()
    lateral_pts = lateral.numpy()
    base_ct_density = get_base_ct_density(primary_pts, lateral_pts, monocots)
    assert np.isnan(base_ct_density)


# test get_base_length_ratio with canola
def test_get_base_length_ratio(canola_h5):
    series = Series.load(
        canola_h5, primary_name="primary_multi_day", lateral_name="lateral_3_nodes"
    )
    primary, lateral = series[0]
    primary_pts = primary.numpy()
    lateral_pts = lateral.numpy()
    primary_length_max = get_root_lengths_max(primary_pts)
    bases = get_bases(lateral_pts)
    lateral_base_ys = get_base_ys(bases)
    base_length = get_base_length(lateral_base_ys)
    base_length_ratio = get_base_length_ratio(primary_length_max, base_length)
    np.testing.assert_almost_equal(base_length_ratio, 0.086, decimal=3)


def test_root_width(canola_h5):
    series = Series.load(
        canola_h5, primary_name="primary_multi_day", lateral_name="lateral_3_nodes"
    )
    primary, lateral = series[0]
    primary_pts = primary.numpy()
    primary_max_length_pts = get_max_length_pts(primary_pts)
    lateral_pts = lateral.numpy()
    assert primary_max_length_pts.shape == (6, 2)
    assert lateral_pts.shape == (5, 3, 2)

    root_widths = get_root_pair_widths_projections(
        primary_max_length_pts, lateral_pts, 0.02
    )
    np.testing.assert_almost_equal(root_widths, np.array([31.60323909]), decimal=7)
