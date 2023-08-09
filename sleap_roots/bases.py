"""Trait calculations that rely on bases (i.e., dicot-only)."""

import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points
from typing import Optional


def get_bases(pts: np.ndarray, monocots: bool = False) -> np.ndarray:
    """Return bases (r1) from each root.

    Args:
        pts: Root landmarks as array of shape `(instances, nodes, 2)` or `(nodes, 2)`.
        monocots: Boolean value, where false is dicot (default), true is rice.

    Returns:
        Array of bases `(instances, (x, y))`. If the input is `(nodes, 2)`, an array of
        shape `(2,)` will be returned.
    """

    if monocots:
        return np.nan

    # If the input has shape `(nodes, 2)`, reshape it for consistency
    if pts.ndim == 2:
        pts = pts[np.newaxis, ...]

    # Get the first point of each instance
    base_pts = pts[:, 0]  # Shape is `(instances, 2)`

    # If the input was `(nodes, 2)`, return an array of shape `(2,)` instead of `(1, 2)`
    if base_pts.shape[0] == 1:
        return base_pts[0]

    return base_pts


def get_base_tip_dist(primary_base_pt: tuple, primary_tip_pt: tuple) -> float:
    """
    Calculate the straight-line distance from the base to the tip of the primary root.

    Args:
        primary_base_pt: The x and y coordinates of the base point of the primary root.
        primary_tip_pt: The x and y coordinates of the tip point of the primary root.

    Returns:
        float: Distance from the base to the tip of the primary root.
    """

    # Convert the tuples to numpy arrays for vectorized computation
    base_pt_array = np.array(primary_base_pt)
    tip_pt_array = np.array(primary_tip_pt)

    # If either of the points is NaN, return NaN
    if np.isnan(base_pt_array).all() or np.isnan(tip_pt_array).all():
        return np.nan

    # Compute and return the Euclidean distance between the two points
    distance = np.linalg.norm(base_pt_array - tip_pt_array)
    return distance


def get_lateral_count(pts: np.ndarray):
    """Get number of lateral roots.

    Args:
        pts: lateral root landmarks as array of shape `(instance, node, 2)`.

    Return:
        Scalar of number of lateral roots.
    """
    lateral_count = pts.shape[0]
    return lateral_count


def get_base_xs(pts: np.ndarray, monocots: bool = False) -> np.ndarray:
    """Get x coordinates of the base of each lateral root.

    Args:
        pts: root landmarks as array of shape `(instances, point, 2)` or bases
            `(instances, 2)`.
        monocots: Boolean value, where false is dicot (default), true is rice.

    Return:
        An array of the x-coordinates of bases `(instance,)`.
    """
    # If the input is a single number (float or integer), return np.nan
    if isinstance(pts, (np.floating, float, np.integer, int)):
        return np.nan

    # If the input array doesn't have 2 or 3 dimensions, raise an error
    if pts.ndim not in (2, 3):
        raise ValueError(
            "Input array must be 2-dimensional (n_bases, 2) or "
            "3-dimensional (n_roots, n_nodes, 2)."
        )

    # If the input array has 3 dimensions, calculate the base points,
    # otherwise, assume the input array already contains the base points
    if pts.ndim == 3:
        _base_pts = get_bases(
            pts, monocots
        )  # Assuming get_bases returns an array of shape (instance, 2)
    else:
        _base_pts = pts

    # If _base_pts is a single number (float or integer), return np.nan
    if isinstance(_base_pts, (np.floating, float, np.integer, int)):
        return np.nan

    # If the base points array doesn't have exactly 2 dimensions or
    # the second dimension is not of size 2, raise an error
    if _base_pts.ndim != 2 or _base_pts.shape[1] != 2:
        raise ValueError(
            "Array of base points must be 2-dimensional with shape (instance, 2)."
        )

    # If everything is fine, extract and return the x-coordinates of the base points
    else:
        base_xs = _base_pts[:, 0]
        return base_xs


def get_base_ys(base_pts: np.ndarray, monocots: bool = False) -> np.ndarray:
    """Get y coordinates of the base of each root.

    Args:
        base_pts: root bases as array of shape `(instances, 2)` or `(2)`
            when there is only one root, as is the case for primary roots.
        monocots: Boolean value, where false is dicot (default), true is rice.

    Return:
        An array of the y-coordinates of bases (instances,).
    """
    # If the input is a single number (float or integer), return np.nan
    if isinstance(base_pts, (np.floating, float, np.integer, int)):
        return np.nan

    # Check for the 2D shape of the input array
    if base_pts.ndim == 1:
        # If shape is `(2,)`, then reshape it to `(1, 2)` for consistency
        base_pts = base_pts.reshape(1, 2)
    elif base_pts.ndim != 2:
        raise ValueError("Input array must be of shape `(instances, 2)` or `(2, )`.")

    # At this point, `base_pts` should be of shape `(instances, 2)`.
    base_ys = base_pts[:, 1]
    return base_ys


def get_base_length(pts: np.ndarray, monocots: bool = False):
    """Get the y-axis difference from the top lateral base to the bottom lateral base.

    Args:
        pts: root landmarks as array of shape `(instances, point, 2)` or base_ys
            `(instances)`.
        monocots: Boolean value, where false is dicot (default), true is rice.

    Return:
        The distance between the top base y-coordinate and the deepest
            base y-coordinate.
    """
    # If the input is a single number (float or integer), return np.nan
    if isinstance(pts, (np.floating, float, np.integer, int)):
        return np.nan

    if pts.ndim not in (1, 3):
        raise ValueError(
            "Input array must be 1-dimensional (n_base_ys) or "
            "3-dimensional (n_roots, n_nodes, 2)."
        )

    if pts.ndim == 3:
        base_ys = get_base_ys(
            pts, monocots
        )  # Assuming get_base_ys returns an array of shape (instances)
    else:
        base_ys = pts

    # If base_ys is a single number (float or integer), return np.nan
    if isinstance(base_ys, (np.floating, float, np.integer, int)):
        return np.nan

    if base_ys.ndim != 1:
        raise ValueError(
            "Array of base y-coordinates must be 1-dimensional with shape (instances)."
        )

    base_length = np.nanmax(base_ys) - np.nanmin(base_ys)
    return base_length


def get_base_ct_density(
    primary_length_max: float, lateral_base_pts: np.ndarray, monocots: bool = False
):
    """Get a ratio of the number of base points to maximum primary root length.

    Args:
        primary_length_max: Scalar of maximum primary root length.
        lateral_base_pts: Base points of lateral roots of shape (instances, 2).
        monocots: Boolean value, where false is dicot (default), true is rice.

    Return:
        Scalar of base count density.
    """

    # Check if the input is valid for lateral_base_pts or if monocots is True
    if (
        monocots
        or isinstance(lateral_base_pts, (np.floating, float, np.integer, int))
        or np.isnan(lateral_base_pts).all()
    ):
        return np.nan

    # Get the number of base points of lateral roots
    base_ct = len(lateral_base_pts[~np.isnan(lateral_base_pts[:, 0])])

    # Handle cases where maximum primary length is zero or NaN to avoid division by zero
    if primary_length_max == 0 or np.isnan(primary_length_max):
        return np.nan

    # Calculate base_ct_density
    base_ct_density = base_ct / primary_length_max

    return base_ct_density


def get_base_length_ratio(
    primary_length: float, base_length: float, monocots: bool = False
) -> float:
    """
    Calculate the ratio of the length of the bases along the primary root to the
    primary root length.

    Args:
        primary_length (float): Length of the primary root.
        base_length (float): Length of the bases along the primary root.
        monocots (bool): True if the roots are monocots, False if they are dicots.

    Returns:
        Ratio of the length of the bases along the primary root to the primary root
            length.
    """

    # If roots are monocots or either of the lengths are NaN, return NaN
    if monocots or np.isnan(primary_length) or np.isnan(base_length):
        return np.nan

    # Handle case where primary length is zero to avoid division by zero
    if primary_length == 0:
        return np.nan

    # Compute and return the base length ratio
    base_length_ratio = base_length / primary_length
    return base_length_ratio


def get_base_median_ratio(lateral_base_ys, primary_tip_pt_y, monocots: bool = False):
    """Get ratio of median value in all base points to tip of primary root in y axis.

    Args:
        lateral_base_ys: y-coordinates of the base points of lateral roots of shape
            `(instances,)`.
        primary_tip_pt_y: y-coordinate of the tip point of the primary root of shape
            `(1)`.
        monocots: Boolean value, where false is dicot (default), true is rice.

    Return:
        Scalar of base median ratio. If all y-coordinates of the lateral root bases are
        NaN, the function returns NaN.
    """

    # Check if the roots are monocots, if so return NaN
    if monocots:
        return np.nan

    # Check if all y-coordinates of lateral root bases are NaN, if so return NaN
    if np.isnan(lateral_base_ys).all():
        return np.nan

    # Calculate the median of all y-coordinates of lateral root bases
    median_base_y = np.nanmedian(lateral_base_ys)

    # If primary_tip_pt_y is an array of shape (1), extract the scalar value
    if isinstance(primary_tip_pt_y, np.ndarray) and primary_tip_pt_y.shape == (1,):
        primary_tip_pt_y = primary_tip_pt_y[0]

    # Compute the ratio of the median y-coordinate of lateral root bases to the
    # y-coordinate of the primary root tip
    base_median_ratio = median_base_y / primary_tip_pt_y

    return base_median_ratio


def get_root_pair_widths_projections(
    primary_max_length_pts: np.ndarray,
    lateral_pts: np.ndarray,
    tolerance: float,
    monocots: bool = False,
) -> float:
    """Return estimation of root width using bases of lateral roots.

    Args:
        primary_max_length_pts: Longest primary root as an array of shape (nodes, 2).
        lateral_pts: Lateral roots as an array of shape (n, nodes, 2).
        tolerance: Difference in projection norm between the right and left side
            (~0.02).
        monocots: Boolean value, where False is dicot (default), True is rice.

    Returns:
        float: The distance in pixels between the bases of matched roots, or NaN
            if no matches were found or all input points were NaN.

    Raises:
        ValueError: If the input arrays are of incorrect shape.
    """

    if primary_max_length_pts.ndim != 2 or lateral_pts.ndim != 3:
        raise ValueError("Input arrays should be 2-dimensional and 3-dimensional")

    if (
        monocots
        or np.isnan(primary_max_length_pts).all()
        or np.isnan(lateral_pts).all()
    ):
        return np.nan

    primary_pts_filtered = primary_max_length_pts[
        ~np.isnan(primary_max_length_pts).any(axis=-1)
    ]
    primary_line = LineString(primary_pts_filtered)

    has_base = ~np.isnan(lateral_pts[:, 0, 0])
    valid_inds = np.argwhere(has_base).squeeze()
    lateral_pts = lateral_pts[has_base]

    is_left = lateral_pts[:, 0, 0] > np.nanmin(lateral_pts[:, 1:, 0], axis=1)

    if is_left.all() or (~is_left).all():
        return np.nan

    left_bases, right_bases = lateral_pts[is_left, 0], lateral_pts[~is_left, 0]

    nearest_primary_right = [
        nearest_points(primary_line, Point(right_base))[0] for right_base in right_bases
    ]

    nearest_primary_left = [
        nearest_points(primary_line, Point(left_base))[0] for left_base in left_bases
    ]

    nearest_primary_norm_right = np.array(
        [primary_line.project(pt, normalized=True) for pt in nearest_primary_right]
    )

    nearest_primary_norm_left = np.array(
        [primary_line.project(pt, normalized=True) for pt in nearest_primary_left]
    )

    projection_diffs = np.abs(
        nearest_primary_norm_left.reshape(-1, 1)
        - nearest_primary_norm_right.reshape(1, -1)
    )

    indices = np.argwhere(projection_diffs <= tolerance)

    left_inds = indices[:, 0]
    right_inds = indices[:, 1]

    match_dists = np.linalg.norm(
        left_bases[left_inds] - right_bases[right_inds],
        axis=-1,
    )

    return match_dists
