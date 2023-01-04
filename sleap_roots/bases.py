"""Trait calculations that rely on bases (i.e., dicot-only)."""

import numpy as np
import shapely
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points


def get_bases(pts: np.ndarray) -> np.ndarray:
    """Return bases (r1) from each lateral root.

    Args:
        pts: Root landmarks as array of shape (instances, nodes, 2)

    Returns:
        Array of bases (instances, (x, y)).
    """
    # Get the first point of each instance. Shape is (instances, 2)
    base_pts = pts[:, 0]
    return base_pts


def get_root_lengths(pts: np.ndarray) -> np.ndarray:
    """Return root lengths for all roots in a frame.

    Args:
        pts: Root landmarks as array of shape (instances, nodes, 2).

    Returns:
        Array of root lengths of shape (instances,).
        If there is no root, or the roots is one point only (all of the rest of the
        points are NaNs), an array of NaNs with shape (len(pts),) is returned.
        This is also the case for non-contiguous points at the moment.
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


# def get_root_pair_widths_projections(lateral_pts: np.ndarray, primary_pts: np.ndarray, tolerance: float) -> tuple(vector, vector, vector):
def get_root_pair_widths_projections(lateral_pts, primary_pts, tolerance):
    """Match pairs of roots on the left and right using their projection on the primary
    root and return their distance. Calls on functions "get_root_lengths", "get_bases".

    Args:
        lateral_pts: Lateral roots as arrays of shape (n, nodes, 2).
        primary_pts: Lateral roots as arrays of shape (n, nodes, 2).
        tolerance: difference in projection norm between the right and left side (~0.02).

    Returns:
        A tuple of (dists, left_inds, right_inds) where:

        - match_dists is the distance in pixels between the bases of matched
            roots as a vector of size (n_matches,).
        - left_inds is are the indices of the left roots that were matched as
            a vector of size (n_matches,).
        - right_inds is are the indices of the right roots that were matched as
            a vector of size (n_matches,).

        If all the lateral roots are on one side of the primary root, 3 empty arrays of
        shape 0 are returned.
    """
    # Must be done with instance of primary_pts with max root length in frame
    max_length_idx = np.nanargmax(get_root_lengths(primary_pts))

    # Make a line of the primary points
    primary_line = LineString(primary_pts[max_length_idx])

    # Filter by whether the base node is present.
    has_base = ~np.isnan(lateral_pts[:, 0, 0])
    valid_inds = np.argwhere(has_base).squeeze()
    lateral_pts = lateral_pts[has_base]

    # Find roots facing left based on whether the base x-coord
    # is larger than the tip x-coord.
    is_left = lateral_pts[:, 0, 0] > np.nanmin(lateral_pts[:, 1:, 0], axis=1)

    # Edge Case: Only found roots on one side.
    if is_left.all() or (~is_left).all():
        return np.array([]), np.array([]), np.array([])

    # Get left and right base points.
    left_bases, right_bases = lateral_pts[is_left, 0], lateral_pts[~is_left, 0]

    # Find the nearest point to each right lateral base on the primary root line
    nearest_primary_right = [
        nearest_points(primary_line, Point(right_base))[0] for right_base in right_bases
    ]

    # Find the nearest point to each left lateral base on the primary root line
    nearest_primary_left = [
        nearest_points(primary_line, Point(left_base))[0] for left_base in left_bases
    ]

    # Returns the distance along the primary line of point in nearest_primary_right, normalized to the length of the object.
    nearest_primary_norm_right = np.array(
        [primary_line.project(pt, normalized=True) for pt in nearest_primary_right]
    )
    # Returns the distance along the primary line of point in nearest_primary_left, normalized to the length of the object.
    nearest_primary_norm_left = np.array(
        [primary_line.project(pt, normalized=True) for pt in nearest_primary_left]
    )

    # get all possible differences in projections from all base pairs
    projection_diffs = np.abs(
        nearest_primary_norm_left.reshape(-1, 1)
        - nearest_primary_norm_right.reshape(1, -1)
    )

    # shape is [# of valid base pairs, 2 [left right]]
    indices = np.argwhere(projection_diffs <= tolerance)

    left_inds = indices[:, 0]
    right_inds = indices[:, 1]

    # Find pairwise distances. (shape is (# of left bases, # of right bases))
    dists = np.linalg.norm(
        np.expand_dims(left_bases, axis=1) - np.expand_dims(right_bases, axis=0),
        axis=-1,
    )

    # Pull out match distances.
    match_dists = np.array([dists[l, r] for l, r in zip(left_inds, right_inds)])

    # Convert matches to indices before splitting by side.
    left_inds = np.argwhere(is_left).reshape(-1)[left_inds]
    right_inds = np.argwhere(~is_left).reshape(-1)[right_inds]

    # Convert matches to indices before filtering.
    left_inds = valid_inds[left_inds]
    right_inds = valid_inds[right_inds]

    return match_dists, left_inds, right_inds
