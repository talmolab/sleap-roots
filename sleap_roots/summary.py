"""Get summary of the traits."""

import numpy as np
from typing import Tuple


def get_summary(
    trait: np.ndarray,
) -> Tuple[float, float, float, float, float, float, float, float, float]:
    """Get summary of traits.

    Args:
        traits: Vector of trait values as a numpy array of shape (n,).

    Returns:
        A tuple of 9 scalar statistical summary measures:
            min, max, mean, median, standard deviation
            percentiles: 5, 25, 75, 95
    """
    trait_min = np.nanmin(trait)
    trait_max = np.nanmax(trait)
    trait_mean = np.nanmean(trait)
    trait_median = np.nanmedian(trait)
    trait_std = np.nanstd(trait)
    trait_prc5 = np.nanpercentile(trait, 5)
    trait_prc25 = np.nanpercentile(trait, 25)
    trait_prc75 = np.nanpercentile(trait, 75)
    trait_prc95 = np.nanpercentile(trait, 95)
    return (
        trait_min,
        trait_max,
        trait_mean,
        trait_median,
        trait_std,
        trait_prc5,
        trait_prc25,
        trait_prc75,
        trait_prc95,
    )
