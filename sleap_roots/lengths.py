"""Get length-related traits"""
import numpy as np
from sleap_roots.bases import get_base_tip_dist
from typing import Optional


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
        return np.array([[(np.nan, np.nan), (np.nan, np.nan)]])

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
        pts: Root landmarks as array of shape `(instances, nodes, 2)` or `(nodes, 2)`.

    Returns:
        Array of root lengths of shape `(instances,)`. If there is no root, or the root
        is one point only (all of the rest of the points are NaNs), an array of NaNs
        with shape (len(pts),) is returned. This is also the case for non-contiguous
        points.
    """

    # If the input has shape `(nodes, 2)`, reshape it for consistency
    if pts.ndim == 2:
        pts = pts[np.newaxis, ...]

    # Get the (x,y) differences of segments for each instance
    segment_diffs = np.diff(pts, axis=1)
    # Get the lengths of each segment by taking the norm
    segment_lengths = np.linalg.norm(segment_diffs, axis=-1)
    # Add the segments together to get the total length using nansum
    total_lengths = np.nansum(segment_lengths, axis=-1)
    # Find the NaN segment lengths and record NaN in place of 0 when finding the total length
    total_lengths[np.isnan(segment_lengths).all(axis=-1)] = np.nan

    # If there is 1 instance, return a scalar instead of an array of length 1
    if len(total_lengths) == 1:
        return total_lengths[0]

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
    primary_length: Optional[float] = None,
    primary_base_tip_dist: Optional[float] = None,
    pts: Optional[np.ndarray] = None,
) -> float:
    """Calculate the gravitropism index of a primary root.

    The gravitropism index quantifies the curviness of the root's growth. A higher
    gravitropism index indicates a curvier root (less responsive to gravity), while a
    lower index indicates a straighter root (more responsive to gravity). The index is
    computed as the difference between the maximum primary root length and straight-line
    distance from the base to the tip of the primary root, normalized by the root length.

    Args:
        primary_length: Maximum length of the primary root. Used if `pts` is not
            provided.
        primary_base_tip_dist: The straight-line distance from the base to the tip of
        the primary root. Used if `pts` is not provided.
        pts: Landmarks of the primary root of shape `(instances, nodes, 2)`. If
        provided, `primary_length` and `primary_base_tip_dist` are ignored.

    Returns:
        float: Gravitropism index of the primary root, quantifying its curviness.
    """
    # Use provided scalar values if available
    if primary_length is not None and primary_base_tip_dist is not None:
        max_primary_length = primary_length
        max_base_tip_distance = primary_base_tip_dist

    # Use provided pts array to compute required values if available
    elif pts is not None:
        if np.isnan(pts).all():
            return np.nan
        primary_length_max = get_root_lengths_max(pts=pts)
        primary_base_tip_dist = get_base_tip_dist(pts=pts)
        max_primary_length = np.nanmax(primary_length_max)
        max_base_tip_distance = np.nanmax(primary_base_tip_dist)

    else:
        raise ValueError(
            "Either both primary_length and primary_base_tip_dist, or pts"
            "must be provided."
        )

    # Check for invalid values (NaN or zero lengths)
    if (
        np.isnan(max_primary_length)
        or np.isnan(max_base_tip_distance)
        or max_primary_length == 0
    ):
        return np.nan

    # Calculate and return gravitropism index
    grav_index = (max_primary_length - max_base_tip_distance) / max_primary_length
    return grav_index
