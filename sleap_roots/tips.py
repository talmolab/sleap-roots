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
        pts: root landmarks as array of shape (instance, point, 2) or tips (instance, 2)

    Return:
        An array tip x-coordinates (instance,).
    """
    if pts.ndim not in (2, 3):
        raise ValueError(
            "Input array must be 2-dimensional (n_tips, 2) or "
            "3-dimensional (n_roots, n_nodes, 2)."
        )

    if pts.ndim == 3:
        _tip_pts = get_tips(
            pts
        )  # Assuming get_tips returns an array of shape (instance, 2)
    else:
        _tip_pts = pts

    if _tip_pts.ndim != 2 or _tip_pts.shape[1] != 2:
        raise ValueError(
            "Array of tip points must be 2-dimensional with shape (instance, 2)."
        )

    tip_xs = _tip_pts[:, 0]
    return tip_xs


def get_tip_ys(pts: np.ndarray) -> np.ndarray:
    """Get y coordinations of tip points.

    Args:
        pts: root landmarks as array of shape (instance, point, 2) or tips (instance, 2)

    Return:
        An array tip y-coordinates (instance,).
    """
    if pts.ndim not in (2, 3):
        raise ValueError(
            "Input array must be 2-dimensional (n_tips, 2) or "
            "3-dimensional (n_roots, n_nodes, 2)."
        )

    if pts.ndim == 3:
        _tip_pts = get_tips(
            pts
        )  # Assuming get_tips returns an array of shape (instance, 2)
    else:
        _tip_pts = pts

    if _tip_pts.ndim != 2 or _tip_pts.shape[1] != 2:
        raise ValueError(
            "Array of tip points must be 2-dimensional with shape (instance, 2)."
        )

    tip_ys = _tip_pts[:, 1]
    return tip_ys
