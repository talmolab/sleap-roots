"""Get angle of each root."""

import numpy as np


def get_node_ind(pts: np.ndarray, proximal: bool = True) -> np.ndarray:
    """Find proximal/distal node index.

    Args:
        pts: Numpy array of points of shape (instances, nodes, 2) or (nodes, 2).
        proximal: Boolean value, where true is proximal (default), false is distal.

    Returns:
        An array of shape (instances,) of proximal or distal node indices.

        The proximal node is the first non-NaN node in the first half of the root.

        The distal node is the last non-NaN node in the last half of the root.

        If all nodes (or all nodes in the half of the root) are NaN, then zero is
        returned.
    """
    # Check if pts is 2D, if so, reshape to 3D
    if pts.ndim == 2:
        pts = pts[np.newaxis, ...]

    n_instances, n_nodes, _ = pts.shape

    # Identify where NaN values exist
    is_nan = np.isnan(pts).any(axis=-1)  # (n_instances, n_nodes)

    # If only NaN values, return NaN
    if is_nan.all():
        return np.zeros((n_instances,))

    if proximal:
        # Proximal nodes are in the first half of the root.
        is_nan = is_nan[:, 1 : (n_nodes + 1) // 2]
        node_ind = np.argmax(~is_nan, axis=-1) + 1
    else:
        # Distal nodes are in the last half of the root.
        is_nan = is_nan[:, (n_nodes + 1) // 2 :]
        node_ind = np.argmax(~is_nan[:, ::-1], axis=-1)
        node_ind = n_nodes - node_ind - 1

    # If the selected index is missing originally, return 0.
    node_ind = np.where(is_nan.all(axis=-1), 0, node_ind)

    return node_ind


def get_root_angle(
    pts: np.ndarray, node_ind: np.ndarray, proximal: bool = True, base_ind: int = 0
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
    # if node_ind is a single  int value, make it as array to keep consistent
    if not isinstance(node_ind, np.ndarray):
        node_ind = [node_ind]

    if np.isnan(node_ind).all():
        return np.nan

    if pts.ndim == 2:
        pts = np.expand_dims(pts, axis=0)

    angs_root = []
    for i in range(len(node_ind)):
        # if the node_ind is 0, do NOT calculate angs
        if node_ind[i] == 0:
            angs = np.nan
        else:
            xy = pts[i, node_ind[i], :] - pts[i, base_ind, :]  # center on base node
            # calculate the angle and convert to the start with gravity direction
            ang = np.arctan2(-xy[1], xy[0]) * 180 / np.pi
            angs = abs(ang + 90) if ang < 90 else abs(-(360 - 90 - ang))
        angs_root.append(angs)
    angs_root = np.array(angs_root)

    # If only one root, return a scalar instead of a single-element array
    if angs_root.shape[0] == 1:
        return angs_root[0]
    return angs_root
