"""Get angle of each root."""

import numpy as np
import math


def get_node_ind(pts: np.ndarray, proximal: bool = True) -> np.ndarray:
    """Find proximal/distal node index.

    Args:
        pts: Numpy array of points of shape (instances, nodes, 2) or (nodes, 2).
        proximal: Boolean value, where true is proximal (default), false is distal.

    Returns:
        An array of shape (instances,) of proximal or distal node index.
    """
    # Check if pts is a numpy array
    if not isinstance(pts, np.ndarray):
        raise TypeError("Input pts should be a numpy array.")

    # Check if pts has 2 or 3 dimensions
    if pts.ndim not in [2, 3]:
        raise ValueError("Input pts should have 2 or 3 dimensions.")

    # Check if the last dimension of pts has size 2
    if pts.shape[-1] != 2:
        raise ValueError(
            "The last dimension of the input pts should have size 2,"
            "representing x and y coordinates."
        )

    # Check if pts is 2D, if so, reshape to 3D
    if pts.ndim == 2:
        pts = pts[np.newaxis, ...]

    # Identify where NaN values exist
    nan_mask = np.isnan(pts).any(axis=-1)

    # If only NaN values, return NaN
    if nan_mask.all():
        return np.nan

    if proximal:
        # For proximal, we want the first non-NaN node in the first half root
        # get the first half nan mask (exclude the base node)
        node_proximal = nan_mask[:, 1 : int((nan_mask.shape[1] + 1) / 2)]
        # get the nearest non-Nan node index
        node_ind = np.argmax(~node_proximal, axis=-1)
        # if there is no non-Nan node, set value of 99
        node_ind[node_proximal.all(axis=1)] = 99
        node_ind = node_ind + 1  # adjust indices by adding one (base node)
    else:
        # For distal, we want the last non-NaN node in the last half root
        # get the last half nan mask
        node_distal = nan_mask[:, int(nan_mask.shape[1] / 2) :]
        # get the farest non-Nan node
        node_ind = (node_distal[:, ::-1] == False).argmax(axis=1)
        node_ind[node_distal.all(axis=1)] = -95  # set value if no non-Nan node
        node_ind = pts.shape[1] - node_ind - 1  # adjust indices by reversing

    # reset indices of 0 (base node) if no non-Nan node
    node_ind[node_ind == 100] = 0

    # If pts was originally 2D, return a scalar instead of a single-element array
    if pts.shape[0] == 1:
        return node_ind[0]

    # If only one root, return a scalar instead of a single-element array
    if node_ind.shape[0] == 1:
        return node_ind[0]

    return node_ind


def get_root_angle(
    pts: np.ndarray, node_ind: np.ndarray, proximal: bool = True, base_ind=0
) -> np.ndarray:
    """Find angles for each root.

    Args:
        pts: Numpy array of points of shape (instances, nodes, 2).
        node_ind: Primary or lateral root node index.
        proximal: Boolean value, where true is proximal (default), false is distal.
        base_ind: Index of base node in the skeleton (default: 0).

    Returns:
        An array of shape (instances,) of angles in degrees, modulo 360.
    """
    angs_root = []
    # if node_ind is a single  int value, make it as array to keep consistent
    if not isinstance(node_ind, np.ndarray):
        node_ind = [node_ind]

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
