"""Trait calculations that rely on tips."""

import numpy as np


def get_tips(pts: np.ndarray) -> np.ndarray:
    """Return tips (last node) from each root.

    Args:
        pts: Root landmarks as array of shape `(instances, nodes, 2)` or `(nodes, 2)`.

    Returns:
        Array of tips. If the input is `(nodes, 2)`, an array of shape `(2,)` will be
        returned. If the input is `(instances, nodes, 2)`, an array of shape
        `(instances, 2)` will be returned. If there is no root, or the roots don't have
        tips, an array of shape `(instances, 2)` of NaNs will be returned.
    """
    # If the input has shape `(nodes, 2)`, reshape it for consistency
    if pts.ndim == 2:
        pts = pts[np.newaxis, ...]

    # Get the last point of each instance
    tip_pts = pts[:, -1]  # Shape is `(instances, 2)`

    # If the input was `(nodes, 2)`, return an array of shape `(2,)` instead of `(1, 2)`
    if tip_pts.shape[0] == 1:
        return tip_pts[0]

    return tip_pts


def get_tip_xs(tip_pts: np.ndarray) -> np.ndarray:
    """Get x coordinates of tip points.

    Args:
        tip_pts: Root tip points as array of shape `(instances, 2)` or `(2,)` when there
            is only one tip.

    Return:
        An array of tip x-coordinates (instances,) or (1,) when there is only one root.
    """
    if tip_pts.ndim not in (1, 2):
        raise ValueError(
            "Input array must be 2-dimensional (instances, 2) or 1-dimensional (2,)."
        )
    if tip_pts.shape[-1] != 2:
        raise ValueError("Last dimension must be (x, y).")

    tip_xs = tip_pts[..., 0]
    return tip_xs


def get_tip_ys(tip_pts: np.ndarray) -> np.ndarray:
    """Get y coordinates of tip points.

    Args:
        tip_pts: Root tip points as array of shape `(instances, 2)` or `(2,)` when there
            is only one tip.

    Return:
        An array of tip y-coordinates (instances,) or (1,) when there is only one root.
    """
    if tip_pts.ndim not in (1, 2):
        raise ValueError(
            "Input array must be 2-dimensional (instances, 2) or 1-dimensional (2,)."
        )
    if tip_pts.shape[-1] != 2:
        raise ValueError("Last dimension must be (x, y).")

    tip_ys = tip_pts[..., 1]
    return tip_ys
