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


def get_tips_percentile(pts: np.ndarray, pctl: np.ndarray) -> np.ndarray:
    """Return y axis of all tip points based on given percentile.

    Args:
        pts: Root landmarks as array of shape (instances, nodes, 2)
        pctl: the percentile of all tip points to calculate y-axis.

    Returns:
        Array of y-axis value of given pctl value.
    """
    tip_pts = get_tips(pts)
    tip_pctl = np.nanpercentile(tip_pts[:, 1], pctl)
    return tip_pctl
