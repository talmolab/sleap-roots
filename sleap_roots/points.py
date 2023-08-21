"""Get traits related to the points."""

import numpy as np
from typing import List, Optional


def get_count(pts: np.ndarray):
    """Get number of roots.

    Args:
        pts: Root landmarks as array of shape `(instance, node, 2)`.

    Return:
        Scalar of number of lateral roots.
    """
    # number of lateral roots is the number of instances
    count = pts.shape[0]
    return count


def join_pts(pts0: np.ndarray, *args: Optional[np.ndarray]) -> List[np.ndarray]:
    """Join an arbitrary number of points arrays and return them as a list.

    Args:
        pts0: The first array of points. Should have shape `(instances, nodes, 2)`
            or `(nodes, 2)`.
        *args: Additional optional arrays of points. Each should have shape
            `(instances, nodes, 2)` or `(nodes, 2)`.

    Returns:
        A list of arrays, each having shape `(nodes, 2)`.
    """
    # Initialize an empty list to store the points
    all_pts = []
    # Loop over the input arrays
    for pts in [pts0] + list(args):
        if pts is None:
            continue  # Skip None values

        # If an array has shape `(nodes, 2)`, expand dimensions to `(1, nodes, 2)`
        if pts.ndim == 2 and pts.shape[-1] == 2:
            pts = pts[np.newaxis, :, :]

        # Validate the shape of each array
        if pts.ndim != 3 or pts.shape[-1] != 2:
            raise ValueError(
                "Points should have a shape of `(instances, nodes, 2)` or `(nodes, 2)`."
            )

        # Add the points to the list
        all_pts.extend(list(pts))

    return all_pts


def get_all_pts_array(pts0: np.ndarray, *args: Optional[np.ndarray]) -> np.ndarray:
    """Get all landmark points within a given frame as a flat array of coordinates.

    Args:
        pts0: The first array of points. Should have shape `(instances, nodes, 2)`
            or `(nodes, 2)`.
        *args: Additional optional arrays of points. Each should have shape
            `(instances, nodes, 2)` or `(nodes, 2)`.

    Returns:
        A 2D array of shape (n_points, 2), containing the coordinates of all extracted
        points.
    """
    # Initialize an empty list to store the points
    concatenated_pts = []

    # Loop over the input arrays
    for pts in [pts0] + list(args):
        if pts is None:
            continue

        # Check if the array has the right number of dimensions
        if pts.ndim not in [2, 3]:
            raise ValueError("Each input array should be 2D or 3D.")

        # Check if the last dimension of the array has size 2
        # (representing x and y coordinates)
        if pts.shape[-1] != 2:
            raise ValueError(
                "The last dimension should have size 2, representing x and y coordinates."
            )

        # Flatten the array to 2D and append to list
        flat_pts = pts.reshape(-1, 2)
        concatenated_pts.append(flat_pts)

    # Concatenate all points into a single array
    return np.concatenate(concatenated_pts, axis=0)


# def get_all_pts_array(
#     primary_max_length_pts: np.ndarray, lateral_pts: np.ndarray, monocots: bool = False
# ) -> np.ndarray:
#     """Get all landmark points within a given frame as a flat array of coordinates.

#     Args:
#         primary_max_length_pts: Points of the primary root with maximum length of shape
#             `(nodes, 2)`.
#         lateral_pts: Lateral root points of shape `(instances, nodes, 2)`.
#         monocots: If False (default), returns a combined array of primary and lateral
#             root points. If True, returns only lateral root points.

#     Returns:
#         A 2D array of shape (n_points, 2), containing the coordinates of all extracted
#         points.
#     """
#     # Check if the input arrays have the right number of dimensions
#     if primary_max_length_pts.ndim != 2 or lateral_pts.ndim != 3:
#         raise ValueError(
#             "Input arrays should have the correct number of dimensions:"
#             "primary_max_length_pts should be 2-dimensional and lateral_pts should be"
#             "3-dimensional."
#         )

#     # Check if the last dimension of the input arrays has size 2
#     # (representing x and y coordinates)
#     if primary_max_length_pts.shape[-1] != 2 or lateral_pts.shape[-1] != 2:
#         raise ValueError(
#             "The last dimension of the input arrays should have size 2, representing x"
#             "and y coordinates."
#         )

#     # Flatten the arrays to 2D
#     primary_max_length_pts = primary_max_length_pts.reshape(-1, 2)
#     lateral_pts = lateral_pts.reshape(-1, 2)

#     # Combine points
#     if monocots:
#         pts_all_array = lateral_pts
#     else:
#         # Check if the data types of the arrays are compatible
#         if primary_max_length_pts.dtype != lateral_pts.dtype:
#             raise ValueError("Input arrays should have the same data type.")

#         pts_all_array = np.concatenate((primary_max_length_pts, lateral_pts), axis=0)

#     return pts_all_array
