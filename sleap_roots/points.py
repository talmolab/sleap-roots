"""Get traits related to the points."""

import numpy as np


def get_all_pts_array(
    primary_max_length_pts: np.ndarray, lateral_pts: np.ndarray, monocots: bool = False
) -> np.ndarray:
    """Get all landmark points within a given frame as a flat array of coordinates.

    Args:
        primary_max_length_pts: Points of the primary root with maximum length of shape
            `(nodes, 2)`.
        lateral_pts: Lateral root points of shape `(instances, nodes, 2)`.
        monocots: If False (default), returns a combined array of primary and lateral
            root points. If True, returns only lateral root points.

    Returns:
        A 2D array of shape (n_points, 2), containing the coordinates of all extracted
        points.
    """
    # Check if the input arrays have the right number of dimensions
    if primary_max_length_pts.ndim != 2 or lateral_pts.ndim != 3:
        raise ValueError(
            "Input arrays should have the correct number of dimensions:"
            "primary_max_length_pts should be 2-dimensional and lateral_pts should be"
            "3-dimensional."
        )

    # Check if the last dimension of the input arrays has size 2
    # (representing x and y coordinates)
    if primary_max_length_pts.shape[-1] != 2 or lateral_pts.shape[-1] != 2:
        raise ValueError(
            "The last dimension of the input arrays should have size 2, representing x"
            "and y coordinates."
        )

    # Flatten the arrays to 2D
    primary_max_length_pts = primary_max_length_pts.reshape(-1, 2)
    lateral_pts = lateral_pts.reshape(-1, 2)

    # Combine points
    if monocots:
        pts_all_array = lateral_pts
    else:
        # Check if the data types of the arrays are compatible
        if primary_max_length_pts.dtype != lateral_pts.dtype:
            raise ValueError("Input arrays should have the same data type.")

        pts_all_array = np.concatenate((primary_max_length_pts, lateral_pts), axis=0)

    return pts_all_array
