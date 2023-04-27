"""Trait calculations that rely on tips."""

import numpy as np


def get_tips(pts):
    """Return tips (last node) from each lateral root.

    Args:
        pts: Root landmarks as array of shape (instances, nodes, 2)

    Returns:
        Array of tips (instances, (x, y)).
        If there is no root, or the roots don't have tips an array of shape
        (instances, 2) of NaNs will be returned.
    """
    # Get the last point of each instance. Shape is (instances, 2)
    tip_pts = pts[:, -1]
    return tip_pts


def get_primary_depth(pts: np.ndarray) -> np.ndarray:
    """Get primary root tip depth.

    Args:
        pts: primary root landmarks as array of shape (1, point, 2)

    Returns:
        Primary root tip depth (location in y-axis).
    """
    # get the last point of primary root, if invisible, return nan
    if pts[:, -1].any() == np.nan:
        return np.nan
    else:
        return pts[:, -1, 1]


def get_tip_xs(pts: np.ndarray) -> np.ndarray:
    """Get x coordinations of tip points.

    Args:
        pts: root landmarks as array of shape (instance, point, 2)

    Return:
        An array of tips in x axis (instance,).
    """
    _tip_pts = get_tips(pts)
    tip_xs = _tip_pts[:, 0]
    return tip_xs


def get_tip_ys(pts: np.ndarray) -> np.ndarray:
    """Get y coordinations of tip points.

    Args:
        pts: root landmarks as array of shape (instance, point, 2)

    Return:
        An array of tips in y axis (instance,)
    """
    _tip_pts = get_tips(pts)
    tip_ys = _tip_pts[:, 1]
    return tip_ys
