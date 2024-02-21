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
        tip_pts: Root tips as array of shape `(instances, 2)` or `(2)`
            when there is only one tip.

    Return:
        An array of the y-coordinates of tips (instances,) or () if `flatten` is `True`.
    """
    # If the input is a single number (float or integer), raise an error
    if isinstance(tip_pts, (np.floating, float, np.integer, int)):
        raise ValueError("Input must be an array of shape `(instances, 2)` or `(2, )`.")

    # Check for the 2D shape of the input array
    if tip_pts.ndim == 1:
        # If shape is `(2,)`, then reshape it to `(1, 2)` for consistency
        tip_pts = tip_pts.reshape(1, 2)
    elif tip_pts.ndim != 2:
        raise ValueError("Input array must be of shape `(instances, 2)` or `(2, )`.")

    # At this point, `tip_pts` should be of shape `(instances, 2)`.
    tip_ys = tip_pts[:, 1]
    return tip_ys
