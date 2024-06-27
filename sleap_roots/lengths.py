"""Get length-related traits."""

import numpy as np
from typing import Union
from shapely.geometry import LineString


def get_max_length_pts(pts: np.ndarray) -> np.ndarray:
    """Points of the root with maximum length (intended for primary root traits).

    Args:
        pts: Root landmarks as array of shape `(instances, nodes, 2)` or `(nodes, 2)`.

    Returns:
        np.ndarray: Array of points with shape `(nodes, 2)` from the root with maximum
        length, or the input array unchanged if its shape is `(nodes, 2)`.
    """
    # Return the input array unchanged if its shape is (nodes, 2)
    if pts.ndim == 2 and pts.shape[1] == 2:
        return pts

    # Return NaN points if the input array is empty
    if len(pts) == 0:
        return np.array([[np.nan, np.nan]])

    # Check if pts has the correct shape for processing multiple instances
    if pts.ndim != 3 or pts.shape[2] != 2:
        raise ValueError(
            "Input array should have shape (instances, nodes, 2) for multiple instances"
        )

    # Calculate the differences between consecutive points in each root
    segment_diffs = np.diff(pts, axis=1)

    # Calculate the length of each segment
    segment_lengths = np.linalg.norm(segment_diffs, axis=-1)

    # Sum the lengths of the segments for each root
    total_lengths = np.nansum(segment_lengths, axis=-1)

    # Handle roots where all segment lengths are NaN,
    # recording NaN in place of the total length for these roots
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
    # Find the NaN segment lengths and record NaN in place of 0 when finding the total
    # length
    total_lengths[np.isnan(segment_lengths).all(axis=-1)] = np.nan

    # If there is 1 instance, return a scalar instead of an array of length 1
    if len(total_lengths) == 1:
        return total_lengths[0]

    return total_lengths


def get_curve_index(
    lengths: Union[float, np.ndarray], base_tip_dists: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """Calculate the curvature index of a root.

    The curvature index quantifies the curviness of the root's growth. A higher
    curvature index indicates a curvier root (less responsive to gravity), while a
    lower index indicates a straighter root (more responsive to gravity). The index is
    computed as the difference between the maximum root length and straight-line
    distance from the base to the tip of the root, normalized by the root length.

    Args:
        lengths: Maximum length of the root(s). Can be a scalar or a 1D numpy array
            of shape `(instances,)`.
        base_tip_dists: The straight-line distance from the base to the tip of the
            root(s). Can be a scalar or a 1D numpy array of shape `(instances,)`.

    Returns:
       Curvature index of the root(s), quantifying its/their curviness. Will be a
            scalar if input is scalar, or a 1D numpy array of shape `(instances,)`
            otherwise.
    """
    # Check if the input is scalar or array
    is_scalar_input = np.isscalar(lengths) and np.isscalar(base_tip_dists)

    # Convert scalars to numpy arrays for uniform handling
    lengths = np.atleast_1d(np.asarray(lengths, dtype=float))
    base_tip_dists = np.atleast_1d(np.asarray(base_tip_dists, dtype=float))

    # Check for shape mismatch
    if lengths.shape != base_tip_dists.shape:
        raise ValueError("The shapes of lengths and base_tip_dists must match.")

    # Calculate the curvature index where possible
    curve_index = np.where(
        (~np.isnan(lengths))
        & (~np.isnan(base_tip_dists))
        & (lengths > 0)
        & (lengths >= base_tip_dists),
        (lengths - base_tip_dists) / np.where(lengths != 0, lengths, np.nan),
        np.nan,
    )

    # Return scalar or array based on the input type
    if is_scalar_input:
        return curve_index.item()
    else:
        return curve_index


def get_min_distance_line_to_line(line1: LineString, line2: LineString) -> float:
    """Calculate the minimum distance between two LineString objects.

    This function computes the shortest distance between any two points on the first
    line segment and the second line segment. If the lines intersect, the minimum
    distance is zero. The distance is calculated in the same units as the coordinates
    of the LineStrings.

    Args:
    line1: The first LineString object representing a line segment.
    line2: The second LineString object representing a line segment.

    Returns:
    The minimum distance between the two line segments.
    """
    # Check if the inputs are LineString instances
    if not isinstance(line1, LineString):
        raise TypeError("The first argument must be a LineString object.")
    if not isinstance(line2, LineString):
        raise TypeError("The second argument must be a LineString object.")

    return line1.distance(line2)
