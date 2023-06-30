"""Trait calculations that rely on bases (i.e., dicot-only)."""

import numpy as np
import shapely
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points


def get_bases(pts: np.ndarray, monocots: bool = False) -> np.ndarray:
    """Return bases (r1) from each lateral root.

    Args:
        pts: Root landmarks as array of shape (instances, nodes, 2)
        monocots: Boolean value, where false is dicot (default), true is rice.

    Returns:
        Array of bases (instances, (x, y)).
    """
    # Get the first point of each instance. Shape is (instances, 2)
    if monocots:
        return np.nan
    else:
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


def get_root_lengths_max(lengths: np.ndarray) -> np.ndarray:
    """Return maximum root length for all roots in a frame.

    Args:
        lengths: root lengths with shape of (instances,).

    Returns:
        Scalar of the maximum root length.
    """
    max_length = np.nanmax(lengths)
    return max_length


def get_base_tip_dist(pts: np.ndarray) -> np.ndarray:
    """Return distance from root base to tip.

    Args:
        pts: Root landmarks as array of shape (instances, nodes, 2)

    Returns:
        Array of distances from base to tip of shape (instances,).
    """
    base_pt = pts[:, 0]
    tip_pt = pts[:, -1]
    distance = np.linalg.norm(base_pt - tip_pt, axis=-1)
    return distance


def get_grav_index(pts: np.ndarray):
    """Get gravitropism index based on primary_length_max and primary_base_tip_dist.

    Args:
        pts: primary root landmarks as array of shape (1, node, 2)

    Returns:
        Scalar of primary root gravity index.
    """
    # get primary root length, if predicted >1 primary roots, use the longest one
    primary_length = get_root_lengths(pts)
    primary_length_max = get_root_lengths_max(primary_length)

    # get the distance between base and tip in y axis
    primary_base_tip_dist = get_base_tip_dist(pts)

    # calculate gravitropism index
    pl_max = np.nanmax(primary_length_max)
    if pl_max == 0:
        return np.nan
    grav_index = (pl_max - np.nanmax(primary_base_tip_dist)) / pl_max
    return grav_index


def get_lateral_count(pts: np.ndarray):
    """Get number of lateral roots.

    Args:
        pts: lateral root landmarks as array of shape (instance, node, 2)

    Return:
        Scalar of number of lateral roots.
    """
    lateral_count = pts.shape[0]
    return lateral_count


def get_base_xs(pts: np.ndarray, monocots: bool = False) -> np.ndarray:
    """Get x coordinations of base points.

    Args:
        pts: root landmarks as array of shape (instance, point, 2)
        monocots: Boolean value, where false is dicot (default), true is rice.

    Return:
        An array of bases in x axis (instance,).
    """
    _base_pts = get_bases(pts, monocots)
    if isinstance(_base_pts, (np.floating, float, np.integer, int)):
        return np.nan
    else:
        base_xs = _base_pts[:, 0]
        return base_xs


def get_base_ys(pts: np.ndarray, monocots: bool = False) -> np.ndarray:
    """Get y coordinations of base points.

    Args:
        pts: root landmarks as array of shape (instance, point, 2)
        monocots: Boolean value, where false is dicot (default), true is rice.

    Return:
        An array of bases in y axis (instance,).
    """
    _base_pts = get_bases(pts, monocots)
    if isinstance(_base_pts, (np.floating, float, np.integer, int)):
        return np.nan
    else:
        base_ys = _base_pts[:, 1]
        return base_ys


def get_base_length(pts: np.ndarray, monocots: bool = False):
    """Get lateral roots top and deepest bases distance in y axis.

    Args:
        pts: lateral root landmarks as array of shape (instance, point, 2)
        monocots: Boolean value, where false is dicot (default), true is rice.

    Return:
        Top and deepest bases distance y-axis.
    """
    base_ys = get_base_ys(pts, monocots)
    base_length = np.nanmax(base_ys) - np.nanmin(base_ys)
    return base_length


def get_base_ct_density(primary_pts, lateral_pts, monocots: bool = False):
    """Get number of base points to maximum primary root length.

    Args:
        primary_pts: primary root points
        lateral_pts: lateral root points
        monocots: Boolean value, where false is dicot (default), true is rice.

    Return:
        Scalar of base count density.
    """
    # get number of base points of lateral roots
    _base_pts = get_bases(lateral_pts, monocots)
    if isinstance(_base_pts, (np.floating, float, np.integer, int)):
        return np.nan
    else:
        base_ct = len(_base_pts[~np.isnan(_base_pts[:, 0])])
        # get primary root length
        lengths_primary = get_root_lengths(primary_pts)
        base_ct_density = base_ct / np.nanmax(lengths_primary)
        return base_ct_density


def get_primary_depth(primary_pts):
    """Get primary root tip depth.

    Args:
        primary_pts: primary root points.

    Return:
        Scalar of primary root tip depth.
    """
    primary_depth = np.nanmax(primary_pts[:, :, 1])
    return primary_depth


def get_base_length_ratio(primary_pts: np.ndarray, lateral_pts: np.ndarray):
    """Get ratio of top-deep base length to primary root length.

    Args:
        primary_pts: primary root points.
        lateral_pts: lateral root points.

    Return:
        Scalar of base length ratio.
    """
    base_length = get_base_length(lateral_pts)
    primary_length = get_root_lengths(primary_pts)
    primary_length_max = get_root_lengths_max(primary_length)
    if primary_length_max == 0:
        return np.nan
    else:
        base_length_ratio = base_length / primary_length_max
        return base_length_ratio


def get_base_median_ratio(
    primary_pts: np.ndarray, lateral_pts: np.ndarray, monocots: bool = False
):
    """Get ratio of median value in all base points to tip of primary root in y axis.

    Args:
        primary_pts: primary root points.
        lateral_pts: lateral root points.
        monocots: Boolean value, where false is dicot (default), true is rice.

    Return:
        Scalar of base median ratio.
    """
    _base_pts = get_bases(lateral_pts, monocots)
    pr_tip_depth = np.nanmax(primary_pts[:, :, 1])
    if np.isnan(_base_pts):
        return np.nan
    else:
        base_median_ratio = np.nanmedian(_base_pts[:, 1]) / pr_tip_depth
        return base_median_ratio


def get_root_pair_widths_projections(
    lateral_pts, primary_pts, tolerance, monocots: bool = False
):
    """Return estimation of stem width using bases of lateral roots.

    Args:
        lateral_pts: Lateral roots as arrays of shape (n, nodes, 2).
        primary_pts: longest primary root as arrays of shape (n, nodes, 2).
        tolerance: difference in projection norm between the right and left side (~0.02).
        monocots: Boolean value, where false is dicot (default), true is rice.

    Returns:
        A match_dists is the distance in pixels between the bases of matched
            roots as a vector of size (n_matches,).

    """
    if monocots:
        return np.nan
    else:
        if np.isnan(primary_pts).all():
            return np.nan
        else:
            primary_pts_filtered = primary_pts[~np.isnan(primary_pts).any(axis=2)]
            primary_line = LineString(primary_pts_filtered)

            # Make a line of the primary points
            primary_line = LineString(primary_pts_filtered)

            # Filter by whether the base node is present.
            has_base = ~np.isnan(lateral_pts[:, 0, 0])
            valid_inds = np.argwhere(has_base).squeeze()
            lateral_pts = lateral_pts[has_base]

            # Find roots facing left based on whether the base x-coord
            # is larger than the tip x-coord.
            is_left = lateral_pts[:, 0, 0] > np.nanmin(lateral_pts[:, 1:, 0], axis=1)

            # Edge Case: Only found roots on one side.
            if is_left.all() or (~is_left).all():
                return np.nan

            # Get left and right base points.
            left_bases, right_bases = lateral_pts[is_left, 0], lateral_pts[~is_left, 0]

            # Find the nearest point to each right lateral base on the primary root line
            nearest_primary_right = [
                nearest_points(primary_line, Point(right_base))[0]
                for right_base in right_bases
            ]

            # Find the nearest point to each left lateral base on the primary root line
            nearest_primary_left = [
                nearest_points(primary_line, Point(left_base))[0]
                for left_base in left_bases
            ]

            # Returns the distance along the primary line of point in nearest_primary_right, normalized to the length of the object.
            nearest_primary_norm_right = np.array(
                [
                    primary_line.project(pt, normalized=True)
                    for pt in nearest_primary_right
                ]
            )
            # Returns the distance along the primary line of point in nearest_primary_left, normalized to the length of the object.
            nearest_primary_norm_left = np.array(
                [
                    primary_line.project(pt, normalized=True)
                    for pt in nearest_primary_left
                ]
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
                np.expand_dims(left_bases, axis=1)
                - np.expand_dims(right_bases, axis=0),
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

        return match_dists
