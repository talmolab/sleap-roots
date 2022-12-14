"""Ellipse fitting and derived trait calculation."""

import numpy as np
from skimage.measure import EllipseModel
from typing import Tuple


def fit_ellipse(pts: np.ndarray) -> Tuple[float, float, float]:
    """Find a best fit ellipse for the points per frame.

    Args:
        pts: Root landmarks as array of shape (..., 2).

    Returns:
        A tuple of (a, b, ratio) containing the semi-major axis length, semi-minor axis
        length, and the ratio of the minor to major lengths.

        If the ellipse fitting fails, NaNs are returned.
    """
    # Filter out NaNs.
    pts = pts.reshape(-1, 2)
    pts = pts[~(np.isnan(pts).all(axis=-1))]

    if len(pts) == 0:
        return np.nan, np.nan, np.nan

    ell = EllipseModel()
    success = ell.estimate(pts)

    if success:
        xc, yc, a_f, b_f, theta = ell.params
        a_f, b_f = np.maximum(a_f, b_f), np.minimum(a_f, b_f)
        ratio_ba_f = b_f / a_f
        return a_f, b_f, ratio_ba_f
    else:
        return np.nan, np.nan, np.nan
