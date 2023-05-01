import numpy as np
import pytest
from sleap_roots.summary import get_summary


@pytest.fixture
def array_random():
    np.random.seed(0)
    return np.random.rand(100)


# test get_summary function with random array
def test_get_summary(array_random):
    [
        trait_min,
        trait_max,
        trait_mean,
        trait_median,
        trait_std,
        trait_prc5,
        trait_prc25,
        trait_prc75,
        trait_prc95,
    ] = get_summary(array_random)
    np.testing.assert_almost_equal(trait_min, 0.004695476192547066, decimal=3)
    np.testing.assert_almost_equal(trait_mean, 0.4727938395125177, decimal=3)
    np.testing.assert_almost_equal(trait_prc95, 0.9456186092221561, decimal=3)
