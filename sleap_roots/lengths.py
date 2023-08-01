"""Get length-related traits"""
import numpy as np


def get_max_length_pts(pts: np.ndarray) -> np.ndarray:
    """
    Points of the root with maximum length (intended for primary root traits).

    Args:
        pts (np.ndarray): Root landmarks as array of shape `(instances, nodes, 2)`.

    Returns:
        np.ndarray: Array of points with shape `(nodes, 2)` from the root with maximum
        length.
    """
    # Return NaN points if the input array is empty
    if len(pts) == 0:
        return np.array([[np.nan, np.nan]])

    # Check if pts has the correct shape, raise error if it does not
    if pts.ndim != 3 or pts.shape[2] != 2:
        raise ValueError("Input array should have shape (instances, nodes, 2)")

    # Calculate the differences between consecutive points in each root
    segment_diffs = np.diff(pts, axis=1)

    # Calculate the length of each segment (the Euclidean distance between consecutive
    # points)
    segment_lengths = np.linalg.norm(segment_diffs, axis=-1)

    # Sum the lengths of the segments for each root
    total_lengths = np.nansum(segment_lengths, axis=-1)

    # Handle roots where all segment lengths are NaN, recording NaN in place of the
    # total length for these roots
    total_lengths[np.isnan(segment_lengths).all(axis=-1)] = np.nan

    # Return NaN points if all total lengths are NaN
    if np.isnan(total_lengths).all():
        return np.array([[np.nan, np.nan]])

    # Find the index of the root with the maximum total length
    max_length_idx = np.nanargmax(total_lengths)

    # Return the points of the root with this index
    return pts[max_length_idx]


def get_root_lengths(pts: np.ndarray) -> np.ndarray:
    """Return root lengths for all roots in a frame.

    Args:
        pts: Root landmarks as array of shape `(instances, nodes, 2)`.

    Returns:
        Array of root lengths of shape `(instances,)`. If there is no root, or the root
            is one point only (all of the rest of the points are NaNs), an array of NaNs
            with shape (len(pts),) is returned. This is also the case for non-contiguous
            points.
    """
    # Get the (x,y) differences of segments for each instance.
    segment_diffs = np.diff(pts, axis=1)
    # Get the lengths of each segment by taking the norm.
    segment_lengths = np.linalg.norm(segment_diffs, axis=-1)
    # Add the segments together to get the total length using nansum.
    total_lengths = np.nansum(segment_lengths, axis=-1)
    # Find the NaN segment lengths and record NaN in place of 0 when finding the total
    # length.
    total_lengths[np.isnan(segment_lengths).all(axis=-1)] = np.nan
    return total_lengths


def get_root_lengths_max(pts: np.ndarray) -> np.ndarray:
    """Return maximum root length for all roots in a frame.

    Args:
        pts: root landmarks as array of shape `(instance, nodes, 2)` or lengths
            `(instances)`.

    Returns:
        Scalar of the maximum root length.
    """
    # If the pts are NaNs, return NaN
    if np.isnan(pts).all():
        return np.nan

    if pts.ndim not in (1, 3):
        raise ValueError(
            "Input array must be 1-dimensional (n_lengths) or "
            "3-dimensional (n_roots, n_nodes, 2)."
        )

    # If the input array has 3 dimensions, calculate the root lengths,
    # otherwise, assume the input array already contains the root lengths
    if pts.ndim == 3:
        root_lengths = get_root_lengths(
            pts
        )  # Assuming get_root_lengths returns an array of shape (instances)
        max_length = np.nanmax(root_lengths)
    else:
        max_length = np.nanmax(pts)

    return max_length


def get_grav_index(
    primary_length: float = None,
    primary_base_tip_dist: float = None,
    pts: np.ndarray = None,
):
    """Get gravitropism index based on primary_length_max and primary_base_tip_dist.

    Args:
        primary_base_tip_dist: scalar of distance from base to tip of primary root
            (longest primary root prediction used if there is more than one). Not used
            if `pts` is specified.
        primary_length: scalar of length of primary root (longest primary root
            prediction used if there is more than one). Not used if `pts` is specified.
        pts: primary root landmarks as array of shape `(instances, nodes, 2)`.

    Returns:
        Scalar of primary root gravity index.
    """
    if primary_length is not None and primary_base_tip_dist is not None:
        # If primary_length and primary_base_tip_dist are provided, use them
        if np.isnan(primary_length) or np.isnan(primary_base_tip_dist):
            return np.nan
        pl_max = primary_length
        primary_base_tip_dist_max = primary_base_tip_dist
    elif pts is not None:
        # If pts is provided, calculate lengths and base-tip distances based on it
        if np.isnan(pts).all():
            return np.nan
        primary_length_max = get_root_lengths_max(pts=pts)
        primary_base_tip_dist = get_base_tip_dist(pts=pts)
        pl_max = np.nanmax(primary_length_max)
        primary_base_tip_dist_max = np.nanmax(primary_base_tip_dist)
    else:
        # If neither primary_length and primary_base_tip_dist nor pts is provided, raise an exception
        raise ValueError(
            "Either both primary_length and primary_base_tip_dist, or pts must be provided."
        )
    # Check if pl_max or primary_base_tip_dist_max is NaN, if so, return NaN
    if np.isnan(pl_max) or np.isnan(primary_base_tip_dist_max):
        return np.nan
    # calculate gravitropism index
    if pl_max == 0:
        return np.nan
    grav_index = (pl_max - primary_base_tip_dist_max) / pl_max
    return grav_index
