from sleap_roots.tips import get_tips, get_tips_percentile
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


# test get_tips_percentile with standard points
def test_get_tips_percentile_standard(pts_standard):
    tip_pctl = get_tips_percentile(pts_standard, [25, 75])
    assert tip_pctl.shape == (2,)
    np.testing.assert_almost_equal(tip_pctl, [5, 7], decimal=7)


# test get_tips_percentile with roots without bases
def test_get_tips_percentile_no_tips(pts_no_tips):
    tip_pctl = get_tips_percentile(pts_no_tips, [25, 75])
    assert tip_pctl.shape == (2,)
    np.testing.assert_almost_equal(tip_pctl, [np.nan, np.nan], decimal=7)


# test get_tips_percentile with roots with one base
def test_get_tips_percentile_one_base(pts_one_tip):
    tip_pctl = get_tips_percentile(pts_one_tip, [25, 75])
    assert tip_pctl.shape == (2,)
    np.testing.assert_almost_equal(tip_pctl, [4, 4], decimal=7)
