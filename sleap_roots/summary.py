"""Get summary of the traits."""

import numpy as np
from typing import Dict, Optional

SUMMARY_SUFFIXES = ["min", "max", "mean", "median", "std", "p5", "p25", "p75", "p95"]


def get_summary(
    X: np.ndarray,
    prefix: Optional[str] = None,
) -> Dict[str, float]:
    """Get summary of a vector of observations.

    Args:
        X: Vector of values as a numpy array of shape `(n,)`.
        prefix: Prefix of the variable name. If not `None`, this string will be appended
            to the key names of the returned dictionary.

    Returns:
        A dictionary of summary statistics of the input vector with keys:
            "min", "max", "mean", "median", "std", "p5", "p25", "p75", "p95"

        If `prefix` was specified, the keys will be prefixed with the string.
    """
    if prefix is None:
        prefix = ""

    return {
        "{prefix}min": np.nanmin(X),
        "{prefix}max": np.nanmax(X),
        "{prefix}mean": np.nanmean(X),
        "{prefix}median": np.nanmedian(X),
        "{prefix}std": np.nanstd(X),
        "{prefix}p5": np.nanpercentile(X, 5),
        "{prefix}p25": np.nanpercentile(X, 25),
        "{prefix}p75": np.nanpercentile(X, 75),
        "{prefix}p95": np.nanpercentile(X, 95),
    }
