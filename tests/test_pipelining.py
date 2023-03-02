import pytest
import numpy as np
from sleap_roots.pipelining import (
    get_statistics,
    get_traits_frame,
    get_traits_plant,
)
from sleap_roots.series import Series


@pytest.fixture
def traits_array_random():
    np.random.seed(0)
    return np.random.randint(100, size=(20))


def test_get_statistics(traits_array_random):
    trait_max, trait_min, trait_mean, trait_std, trait_median = get_statistics(
        traits_array_random
    )
    np.testing.assert_almost_equal(trait_max, [88], decimal=7)
    np.testing.assert_almost_equal(trait_min, [9], decimal=7)
    np.testing.assert_almost_equal(trait_mean, [58.3], decimal=7)
    np.testing.assert_almost_equal(trait_std, [25.088045], decimal=7)
    np.testing.assert_almost_equal(trait_median, [64.5], decimal=7)


def test_get_traits_frame(canola_h5):
    plant = Series.load(
        canola_h5, primary_name="primary_multi_day", lateral_name="lateral_3_nodes"
    )
    df = get_traits_frame(
        plant,
        rice=False,
        frame=0,
        tolerance=0.02,
        pctl_base=[25, 75],
        pctl_tip=[25, 75],
        fraction=2 / 3,
        depth=1080,
        width=2048,
        n_line=50,
    )
    assert len(df) == 1
    assert len(df.columns) == 79
    np.testing.assert_almost_equal(
        df.primary_angles_proximal[0], 50.13129559736394, decimal=7
    )
    np.testing.assert_almost_equal(df.grav_index[0], 0.08898137324716636, decimal=7)
    np.testing.assert_almost_equal(
        df.lateral_length_mean[0], 40.400556356407456, decimal=7
    )
    np.testing.assert_almost_equal(df.stem_widths_median[0], 1, decimal=7)
    np.testing.assert_almost_equal(df.conv_areas[0], 93255.32153574759, decimal=7)
    np.testing.assert_almost_equal(df.ellipse_ratio[0], 0.2889459577340097, decimal=7)
    np.testing.assert_almost_equal(
        df.network_solidity[0], 0.025467588561201876, decimal=7
    )
    np.testing.assert_almost_equal(df.scanline_start[0], 6, decimal=7)


def test_get_traits_plant(canola_h5):
    df = get_traits_plant(
        canola_h5,
        rice=False,
        tolerance=0.02,
        pctl_base=[25, 75],
        pctl_tip=[25, 75],
        fraction=2 / 3,
        depth=1080,
        width=2048,
        n_line=50,
        write_csv=False,
    )
    assert len(df) == 1
    assert len(df.columns) == 386

    np.testing.assert_almost_equal(
        df.primary_angles_proximal_max[0], 62.483951542407084, decimal=7
    )
    np.testing.assert_almost_equal(
        df.lateral_length_mean_min[0], 20.29789164406169, decimal=7
    )
    np.testing.assert_almost_equal(df.stem_widths_median_median[0], 1.25, decimal=7)
    np.testing.assert_almost_equal(df.conv_areas_max[0], 104594.94870982229, decimal=7)
    np.testing.assert_almost_equal(
        df.ellipse_ratio_min[0], 0.09748883191631912, decimal=7
    )
    np.testing.assert_almost_equal(
        df.network_solidity_std[0], 0.063421767709279, decimal=7
    )
    np.testing.assert_almost_equal(
        df.scanline_start_mean[0], 6.694444444444445, decimal=7
    )
