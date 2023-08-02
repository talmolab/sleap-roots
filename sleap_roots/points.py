"""Get points function."""

import numpy as np
from sleap_roots.bases import get_root_lengths
from sleap_roots.series import Series
from typing import List


def get_pt_ind(pts: np.ndarray, proximal: bool = True) -> np.ndarray:
    """Find proximal/distal point index.

    Args:
        pts: Numpy array of points of shape (instances, point, 2).
        proximal: Boolean value, where True is proximal (default), False is distal.

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


def get_primary_pts(plant: Series, frame: int) -> np.ndarray:
    """Get primary root points.

    Args:
        plant: Series object representing a plant image series.
        frame: frame index

    Return:
        An array of primary root points of shape (1, n_points, 2).
        If more than one primary root is present, the longest will be used.
    """
    # get the primary root points, if >1 primary roots, return the longest primary root
    pts_pr = plant.get_primary_points(frame_idx=frame)
    if len(pts_pr) == 0:
        return pts_pr
    else:
        max_length_idx = np.nanargmax(get_root_lengths(pts_pr))
        pts_pr = pts_pr[np.newaxis, max_length_idx]
    return pts_pr


def get_lateral_pts(plant: Series, frame: int) -> np.ndarray:
    """Get lateral root points.

    Args:
        plant: Series object representing a plant image series.
        frame: frame index

    Return:
        An array of primary root points in shape (instance, point, 2)
    """
    pts_lr = plant.get_lateral_points(frame_idx=frame)
    return pts_lr


def get_all_pts(plant: Series, frame: int, monocots: bool = False) -> List[np.ndarray]:
    """Get all points within a frame.

    Args:
        plant: Series object representing a plant image series.
        frame: frame index
        rice: boolean value, where True is rice frame
        monocots: If False (the default), returns primary and lateral points
        combined. If True, only lateral root points will be returned. This is useful for
        monocot species such as rice.

    Return:
        A list of numpy arrays containing sets of points of shape
        (n_instances, n_points, 2).
    """
    # get primary and lateral root points
    pts_pr = get_primary_pts(plant, frame).tolist()
    pts_lr = get_lateral_pts(plant, frame).tolist()

    pts_all = pts_lr if monocots else pts_pr + pts_lr

    return pts_all


def get_all_pts_array(
    plant: pd.Series, frame: int, monocots: bool = False
) -> np.ndarray:
    """Get all landmark points within a given frame as a flat array of coordinates.

    Args:
        plant (pd.Series): A Pandas Series object representing a plant image series.
        frame (int): The index of the frame from which points are to be extracted.
        monocots (bool, optional): If False (default), returns a combined array of primary
            and lateral root points. If True, returns only lateral root points.

    Returns:
        np.ndarray: A 2D array of shape (n_points, 2), containing the coordinates of all
            extracted points.
    """
    # Get primary and lateral root points
    pts_pr = get_primary_pts(plant, frame)
    pts_lr = get_lateral_pts(plant, frame)

    # Convert to numpy arrays
    pts_pr = np.array(pts_pr).reshape(-1, 2)
    pts_lr = np.array(pts_lr).reshape(-1, 2)

    # Combine points
    if monocots:
        pts_all_array = pts_lr
    else:
        pts_all_array = np.concatenate((pts_pr, pts_lr), axis=0)

    return pts_all_array
