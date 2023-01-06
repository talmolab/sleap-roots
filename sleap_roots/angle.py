"""Get angle of each root."""

import numpy as np
import math


def get_node_ind(pts: np.ndarray, proximal=True) -> np.ndarray:
    """Find nproximal/distal node index.

    Args:
        pts: Numpy array of points of shape (instances, nodes, 2).
        proximal: Boolean value, where true is proximal (default), false is distal.

    Returns:
        An array of shape (instances,) of proximal or distal node index.
    """
    node_ind = []
    for i in range(pts.shape[0]):
        ind = 1 if proximal else pts.shape[1] - 1  # set initial proximal/distal node
        while np.isnan(pts[i, ind]).any():
            ind += 1 if proximal else -1
        node_ind.append(ind)
    return node_ind


def get_root_angle(pts: np.ndarray, proximal=True, base_ind=0) -> np.ndarray:
    """Find angles for each root.

    Args:
        pts: Numpy array of points of shape (instances, nodes, 2).
        proximal: Boolean value, where true is proximal (default), false is distal.
        base_ind: Index of base node in the skeleton (default: 0).

    Returns:
        An array of shape (instances,) of angles in degrees, modulo 360.
    """
    node_ind = get_node_ind(pts, proximal)  # get proximal or distal node index
    angs_root = []
    for i in range(len(node_ind)):
        # filter out the cases if all nan nodes in last/first half part
        # to calculate proximal/distal angle
        if (node_ind[i] < math.ceil(pts.shape[1] / 2) and proximal) or (
            node_ind[i] >= math.floor(pts.shape[1] / 2) and not (proximal)
        ):
            xy = pts[i, node_ind[i], :] - pts[i, base_ind, :]  # center on base node
            # calculate the angle and convert to the start with gravity direction
            ang = np.arctan2(-xy[1], xy[0]) * 180 / np.pi
            angs = abs(ang + 90) if ang < 90 else abs(-(360 - 90 - ang))
        else:
            angs = np.nan
        angs_root.append(angs)
    return np.array(angs_root)
