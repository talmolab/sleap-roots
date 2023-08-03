"""Ellipse fitting and derived trait calculation."""

import numpy as np
from skimage.measure import EllipseModel
from typing import Tuple, Union


def fit_ellipse(pts: np.ndarray) -> Tuple[float, float, float]:
    """Find a best fit ellipse for the points per frame.

    Args:
        pts: Root landmarks as array of shape (..., 2).

    Returns:
        A tuple of (a, b, ratio) containing the semi-major axis length, semi-minor axis
        length, and the ratio of the major to minor lengths.

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
        ratio_ba_f = a_f / b_f
        return a_f, b_f, ratio_ba_f
    else:
        return np.nan, np.nan, np.nan


def get_ellipse_a(pts_all_array: Union[np.ndarray, Tuple[float, float, float]]):
    """Get semi-major axis length of the fitted ellipse.

    Args:
        pts_all_array: landmark points or tuple of ellipse restults.

    Return:
        Scalar of semi-major axis length.
    """
    if type(pts_all_array) == tuple:
        ellipse_a = pts_all_array[0]
    else:
        ellipse_features = fit_ellipse(pts_all_array)
        ellipse_a = ellipse_features[0]
    return ellipse_a


def get_ellipse_b(pts_all_array: Union[np.ndarray, Tuple[float, float, float]]):
    """Get semi-minor axis length of the fitted ellipse.

    Args:
        pts_all_array: landmark points or tuple of ellipse restults.

    Return:
        Scalar of semi-minor axis length.
    """
    if type(pts_all_array) == tuple:
        ellipse_b = pts_all_array[1]
    else:
        ellipse_features = fit_ellipse(pts_all_array)
        ellipse_b = ellipse_features[1]
    return ellipse_b


def get_ellipse_ratio(pts_all_array: Union[np.ndarray, Tuple[float, float, float]]):
    """Get ratio of the minor to major lengths of the fitted ellipse.

    Args:
        pts_all_array: landmark points or tuple of ellipse restults.

    Return:
        Scalar of ratio of the minor to major lengths.
    """
    if type(pts_all_array) == tuple:
        ellipse_ratio = pts_all_array[2]
    else:
        ellipse_features = fit_ellipse(pts_all_array)
        ellipse_ratio = ellipse_features[2]
    return ellipse_ratio
