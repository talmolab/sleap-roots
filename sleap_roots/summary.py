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

    X = np.atleast_1d(X)

    if len(X) == 0 or np.all(np.isnan(X)):
        return {
            f"{prefix}min": np.nan,
            f"{prefix}max": np.nan,
            f"{prefix}mean": np.nan,
            f"{prefix}median": np.nan,
            f"{prefix}std": np.nan,
            f"{prefix}p5": np.nan,
            f"{prefix}p25": np.nan,
            f"{prefix}p75": np.nan,
            f"{prefix}p95": np.nan,
        }
    elif np.issubdtype(X.dtype, np.number):
        return {
            f"{prefix}min": np.nanmin(X),
            f"{prefix}max": np.nanmax(X),
            f"{prefix}mean": np.nanmean(X),
            f"{prefix}median": np.nanmedian(X),
            f"{prefix}std": np.nanstd(X),
            f"{prefix}p5": np.nanpercentile(X, 5),
            f"{prefix}p25": np.nanpercentile(X, 25),
            f"{prefix}p75": np.nanpercentile(X, 75),
            f"{prefix}p95": np.nanpercentile(X, 95),
        }
    else:
        print("X contains non-numeric values")
        return {
            f"{prefix}min": np.nan,
            f"{prefix}max": np.nan,
            f"{prefix}mean": np.nan,
            f"{prefix}median": np.nan,
            f"{prefix}std": np.nan,
            f"{prefix}p5": np.nan,
            f"{prefix}p25": np.nan,
            f"{prefix}p75": np.nan,
            f"{prefix}p95": np.nan,
        }
