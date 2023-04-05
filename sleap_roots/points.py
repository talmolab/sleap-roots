"""Get points function."""

import numpy as np
from sleap_roots.bases import get_root_lengths


def get_pt_ind(pts: np.ndarray, proximal=True) -> np.ndarray:
    """Find proximal/distal point index.

    Args:
        pts: Numpy array of points of shape (instances, point, 2).
        proximal: Boolean value, where true is proximal (default), false is distal.

    Returns:
        An array of shape (instances,) of proximal or distal point index.
    """
    pt_ind = []
    for i in range(pts.shape[0]):
        ind = 1 if proximal else pts.shape[1] - 1  # set initial proximal/distal point
        while np.isnan(pts[i, ind]).any():
            ind += 1 if proximal else -1
            if (ind == pts.shape[1] and proximal) or (ind == 0 and not proximal):
                break
        pt_ind.append(ind)
    return pt_ind


def get_primary_pts(plant, frame):
    """Get primary root points.

    Args:
        plant: plant series name
        frame: frame index

    Return:
        An array of primary root points in shape (1, point, 2).
    """
    # get the primary root points, if >1 primary roots, return the longest primary root
    pts_pr = plant.get_primary_points(frame_idx=frame)
    max_length_idx = np.nanargmax(get_root_lengths(pts_pr))
    pts_pr = pts_pr[np.newaxis, max_length_idx]
    return pts_pr


def get_lateral_pts(plant, frame):
    """Get lateral root points.

    Args:
        plant: plant series name
        frame: frame index

    Return:
        An array of primary root points in shape (instance, point, 2)
    """
    pts_lr = plant.get_lateral_points(frame_idx=frame)
    return pts_lr


def get_all_pts(plant, frame, rice):
    """Get all points within a frame.

    Args:
        plant: plant series name
        frame: frame index
        rice: boolean value, where True is rice frame

    Return:
        an array of all points in shape (instance, point, 2)
    """
    # get primary and lateral root points
    pts_pr = get_primary_pts(plant, frame).tolist()
    pts_lr = get_lateral_pts(plant, frame).tolist()

    pts_all = pts_lr if rice else pts_pr + pts_lr

    return pts_all


def get_all_pts_array(plant, frame, rice):
    """Get all points within a frame.

    Args:
        plant: plant series name
        frame: frame index
        rice: boolean value, where True is rice frame

    Return:
        an array of all points in shape (instance, point, 2)
    """
    # get primary and lateral root points
    pts_pr = get_primary_pts(plant, frame)
    pts_lr = get_lateral_pts(plant, frame)

    pts_all_array = (
        pts_lr.reshape(-1, 2)
        if rice
        else np.concatenate((pts_pr.reshape(-1, 2), pts_lr.reshape(-1, 2)), axis=0)
    )

    return pts_all_array