from sleap_roots.tips import get_tips, get_tip_xs, get_tip_ys
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
def pts_no_tips():
    return np.array(
        [
            [
                [1, 2],
                [np.nan, np.nan],
            ],
            [
                [5, 6],
                [np.nan, np.nan],
            ],
        ]
    )


@pytest.fixture
def pts_one_tip():
    return np.array(
        [
            [
                [1, 2],
                [3, 4],
            ],
            [
                [5, 6],
                [np.nan, np.nan],
            ],
        ]
    )


@pytest.fixture
def pt_standard():
    return np.array([[[1, 2], [3, 4]]])


@pytest.fixture
def pt_nan_tip():
    return np.array([[[1, 2], [np.nan, np.nan]]])


def test_tips_standard(pts_standard):
    tips = get_tips(pts_standard)
    assert tips.shape == (2, 2)
    np.testing.assert_array_equal(tips, [[3, 4], [7, 8]])


def test_tips_no_tips(pts_no_tips):
    tips = get_tips(pts_no_tips)
    assert tips.shape == (2, 2)
    np.testing.assert_array_equal(tips, [[np.nan, np.nan], [np.nan, np.nan]])


def test_tips_one_tip(pts_one_tip):
    tips = get_tips(pts_one_tip)
    assert tips.shape == (2, 2)
    np.testing.assert_array_equal(tips, [[3, 4], [np.nan, np.nan]])


# test get_tip_xs with canola
def test_get_tip_xs_canola(canola_h5):
    # Set the frame index to 0
    frame_index = 0
    # Load the series with a canola dataset
    series = Series.load(canola_h5, primary_name="primary", lateral_name="lateral")
    # Get the lateral roots from the series
    lateral_pts = series.get_lateral_points(frame_index)
    # Get the tips from the lateral roots
    tips = get_tips(lateral_pts)
    # Get the tip x-coordinates
    tip_xs = get_tip_xs(tips)
    assert tip_xs.shape[0] == 5
    np.testing.assert_almost_equal(tip_xs[1], 1072.6610107421875, decimal=3)


# test get_tip_xs with standard points
def test_get_tip_xs_standard(pts_standard):
    # Get the tips from the standard points
    tips = get_tips(pts_standard)
    tip_xs = get_tip_xs(tips)
    assert tip_xs.shape[0] == 2
    np.testing.assert_almost_equal(tip_xs[0], 3, decimal=3)
    np.testing.assert_almost_equal(tip_xs[1], 7, decimal=3)


# test get_tip_xs with no tips
def test_get_tip_xs_no_tip(pts_no_tips):
    # Get the tips from the no tips points
    tips = get_tips(pts_no_tips)
    tip_xs = get_tip_xs(tips)
    assert tip_xs.shape[0] == 2
    np.testing.assert_almost_equal(tip_xs[1], np.nan, decimal=3)
    assert type(tip_xs) == np.ndarray

    tip_xs = get_tip_xs(tips[[0]])
    assert type(tip_xs) == np.ndarray


# test get_tip_ys with canola
def test_get_tip_ys_canola(canola_h5):
    # Set the frame index to 0
    frame_index = 0
    # Load the series with a canola dataset
    series = Series.load(canola_h5, primary_name="primary", lateral_name="lateral")
    # Get the lateral root points from the series
    lateral_pts = series.get_lateral_points(frame_index)
    # Get the tips from the lateral roots
    tips = get_tips(lateral_pts)
    # Get the tip y-coordinates
    tip_ys = get_tip_ys(tips)
    assert tip_ys.shape[0] == 5
    np.testing.assert_almost_equal(tip_ys[1], 276.51275634765625, decimal=3)


# test get_tip_ys with standard points
def test_get_tip_ys_standard(pts_standard):
    tips = get_tips(pts_standard)
    tip_ys = get_tip_ys(tips)
    assert tip_ys.shape[0] == 2
    np.testing.assert_almost_equal(tip_ys[0], 4, decimal=3)
    np.testing.assert_almost_equal(tip_ys[1], 8, decimal=3)


# test get_tip_ys with no tips
def test_get_tip_ys_no_tip(pts_no_tips):
    tips = get_tips(pts_no_tips)
    tip_ys = get_tip_ys(tips)
    assert tip_ys.shape[0] == 2
    np.testing.assert_almost_equal(tip_ys[1], np.nan, decimal=3)
    assert type(tip_ys) == np.ndarray

    tip_ys = get_tip_ys(tips[[0]])
    assert type(tip_ys) == np.ndarray
