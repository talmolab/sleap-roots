"""Get angle of each root."""

import numpy as np


def get_root_base_angle(pts: np.ndarray, base_ind=0, tip_ind=1) -> np.ndarray:
    """Find angles for each root.

    Args:
        pts: Numpy array of points of shape (instances, nodes, 2).
        base_ind: Index of base node in the skeleton (default: 0).
        tip_ind: Index of distal node in the skeleton (default: 1).
            Use 1 to get the angle to the second node.

    Returns:
        An array of shape (instances,) of angles in degrees, modulo 360.
    """
    while np.isnan(pts[:, tip_ind]).all() and tip_ind < int(pts.shape[1] / 2 - 1):
        tip_ind += 1

    if tip_ind < int(pts.shape[1] / 2):
        xy = pts[:, tip_ind] - pts[:, base_ind]  # center on base node
        # calculate the angle and convert to the start with gravity ([0,-1] direction)
        ang = np.arctan2(-xy[..., 1], xy[..., 0]) * 180 / np.pi
        angs = abs(ang + 90) if ang.all() < 90 else abs(-(360 - 90 - ang))
    else:
        angs = np.nan
    return angs
