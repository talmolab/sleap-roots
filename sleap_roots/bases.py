"""Trait calculations that rely on bases."""

import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points
from scipy.optimize import linear_sum_assignment
from typing import Union, Tuple


def get_bases(pts: np.ndarray) -> np.ndarray:
    """Return bases (r1) from each root.

    Args:
        pts: Root landmarks as array of shape `(instances, nodes, 2)` or `(nodes, 2)`.

    Returns:
        Array of bases `(instances, (x, y))`. If the input is `(nodes, 2)`, an array of
            shape `(2,)` will be returned.
    """
    # If the input has shape `(nodes, 2)`, reshape it for consistency
    if pts.ndim == 2:
        pts = pts[np.newaxis, ...]

    # Get the first point of each instance
    base_pts = pts[:, 0]  # Shape is `(instances, 2)`

    # If the input was `(nodes, 2)`, return an array of shape `(2,)` instead of `(1, 2)`
    if base_pts.shape[0] == 1:
        return base_pts[0]

    return base_pts


def get_base_tip_dist(
    base_pts: np.ndarray, tip_pts: np.ndarray
) -> Union[np.ndarray, float]:
    """Calculate the straight-line distance(s) from the base(s) to the tip(s).

    Args:
        base_pts: The x and y coordinates of the base point(s) of the root(s). Shape can
            be either `(2,)` for a single point or `(instances, 2)` for multiple
            instances.
        tip_pts: The x and y coordinates of the tip point(s) of the root(s). Shape
            should match that of `base_pts`.

    Returns:
        Distance(s) from the base(s) to the tip(s) of the root(s). If there's only one
            distance (i.e., shape is `(1,)`), a scalar is returned. Otherwise, an array
            matching the first dimension of the input arrays is returned.
    """
    # Check if the shapes of the two input arrays match
    if base_pts.shape != tip_pts.shape:
        raise ValueError("The shapes of base_pts and tip_pts must match.")

    # Compute the Euclidean distance(s) between the point(s)
    distances = np.linalg.norm(base_pts - tip_pts, axis=-1)

    # If distances is a scalar, check if either base_pts or tip_pts is NaN, and
    # return NaN if true
    if np.isscalar(distances):
        if np.isnan(base_pts).any() or np.isnan(tip_pts).any():
            return np.nan
        return distances

    # If distances is an array, create and apply the nan_mask
    nan_mask = np.isnan(base_pts).any(axis=-1) | np.isnan(tip_pts).any(axis=-1)
    distances[nan_mask] = np.nan

    return distances


def get_base_xs(base_pts: np.ndarray) -> np.ndarray:
    """Get x coordinates of the base of each lateral root.

    Args:
        base_pts: root bases as array of shape `(instances, 2)` or `(2)` when there is
            only one root, as is the case for primary roots.

    Return:
        An array of base x-coordinates (instances,) or (1,) when there is only one root.
    """
    if base_pts.ndim not in (1, 2):
        raise ValueError(
            "Input array must be 2-dimensional (instances, 2) or 1-dimensional (2,)."
        )
    if base_pts.shape[-1] != 2:
        raise ValueError("Last dimension must be (x, y).")

    base_xs = base_pts[..., 0]
    return base_xs


def get_base_ys(base_pts: np.ndarray) -> np.ndarray:
    """Get y coordinates of the base of each root.

    Args:
        base_pts: root bases as array of shape `(instances, 2)` or `(2)`
            when there is only one root, as is the case for primary roots.

    Return:
        An array of the y-coordinates of bases (instances,).
    """
    # Check for the 2D shape of the input array
    if base_pts.ndim == 1:
        # If shape is `(2,)`, then reshape it to `(1, 2)` for consistency
        base_pts = base_pts.reshape(1, 2)
    elif base_pts.ndim != 2:
        raise ValueError("Input array must be of shape `(instances, 2)` or `(2, )`.")

    # At this point, `base_pts` should be of shape `(instances, 2)`.
    base_ys = base_pts[:, 1]
    return base_ys


def get_base_length(lateral_base_ys: np.ndarray) -> float:
    """Get the y-axis difference from the top lateral base to the bottom lateral base.

    Args:
        lateral_base_ys: y-coordinates of the base points of lateral roots of shape
            `(instances,)`.

    Return:
        The distance between the top base y-coordinate and the deepest
        base y-coordinate.
    """
    # Compute the difference between the maximum and minimum y-coordinates
    base_length = np.nanmax(lateral_base_ys) - np.nanmin(lateral_base_ys)
    return base_length


def get_base_ct_density(
    primary_length_max: float, lateral_base_pts: np.ndarray
) -> float:
    """Get a ratio of the number of base points to maximum primary root length.

    Args:
        primary_length_max: Scalar of maximum primary root length.
        lateral_base_pts: Base points of lateral roots as returned by `get_bases`,
            shape `(instances, 2)` or `(2,)`.

    Return:
        Scalar of base count density.
    """
    # Check if the input is valid for lateral_base_pts
    if (
        isinstance(lateral_base_pts, (np.floating, float, np.integer, int))
        or np.isnan(lateral_base_pts).all()
    ):
        return np.nan

    # Handle the case where lateral_base_pts has shape `(2,)`
    if lateral_base_pts.ndim == 1 and lateral_base_pts.shape[0] == 2:
        base_ct = 1  # Only one base point in this case

    # Handle the case where lateral_base_pts has shape `(instances, 2)`
    else:
        base_ct = len(lateral_base_pts[~np.isnan(lateral_base_pts[:, 0])])

    # Handle cases where maximum primary length is zero or NaN to avoid division by zero
    if primary_length_max == 0 or np.isnan(primary_length_max):
        return np.nan

    # Calculate base_ct_density
    base_ct_density = base_ct / primary_length_max

    return base_ct_density


def get_base_length_ratio(primary_length: float, base_length: float) -> float:
    """Calculate the ratio of the length of the bases to the primary root length.

    Args:
        primary_length (float): Length of the primary root.
        base_length (float): Length of the bases along the primary root.

    Returns:
        Ratio of the length of the bases along the primary root to the primary root
            length.
    """
    # If either of the lengths are NaN, return NaN
    if np.isnan(primary_length) or np.isnan(base_length):
        return np.nan

    # Handle case where primary length is zero to avoid division by zero
    if primary_length == 0:
        return np.nan

    # Compute and return the base length ratio
    base_length_ratio = base_length / primary_length
    return base_length_ratio


def get_base_median_ratio(lateral_base_ys, primary_tip_pt_y):
    """Get ratio of median value in all base points to tip of primary root in y axis.

    Args:
        lateral_base_ys: Y-coordinates of the base points of lateral roots of shape
            `(instances,)`.
        primary_tip_pt_y: Y-coordinate of the tip point of the primary root of shape
            `(1)` or a scalar.

    Return:
        Scalar of base median ratio. If all y-coordinates of the lateral root bases are
            NaN, the function returns NaN.
    """
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


def get_root_widths(
    primary_max_length_pts: np.ndarray,
    lateral_pts: np.ndarray,
    tolerance: float = 0.02,
    return_inds: bool = False,
) -> Tuple[np.ndarray, list, np.ndarray, np.ndarray]:
    """Estimate root width using bases of lateral roots.

    Args:
        primary_max_length_pts: Longest primary root, represented
            as a 2D array of shape (nodes, 2).
        lateral_pts: Lateral roots, represented as a 3D array of
            shape (n, nodes, 2).
        tolerance: Tolerance level for the projection difference between matched roots.
            Defaults to 0.02.
        return_inds: Flag to indicate whether to return matched indices along with
            distances. Defaults to False.

    Returns:
        - If `return_inds` is False (default):
            Returns an array of distances between the bases of matched roots. If no
            matched indices are found, NaN is returned.

        - If `return_inds` is True:
            Returns a tuple containing the following four elements:
                - matched_dists: Distances between the bases of matched roots. If no
                    matched indices are found, NaN is returned.
                - matched_indices: List of tuples, each containing the indices
                    of matched roots on the left and right sides. A list containing a
                    tuple of NaNs is returned if no matched indices are found.
                - left_bases_final: (n, 2) array containing the (x, y)
                    coordinates of the left bases of the matched roots. An array of
                    NaNs is returned if no matched indices are found.
                - right_bases_final: (n, 2) array containing the (x, y)
                    coordinates of the right bases of the matched roots. An array of
                    NaNs is returned if no matched indices are found.
    """
    # Validate tolerance
    if tolerance <= 0:
        raise ValueError("Tolerance should be a positive number")

    # Check array dimensions
    if primary_max_length_pts.ndim != 2 or lateral_pts.ndim != 3:
        raise ValueError("Input arrays should be 2-dimensional and 3-dimensional")

    # Check the shape of the last dimensions
    if primary_max_length_pts.shape[1] != 2 or lateral_pts.shape[2] != 2:
        raise ValueError("The last dimension should contain x and y coordinates")

    # Initialize default return values
    default_dists = np.nan
    default_indices = [(np.nan, np.nan)]  # List of tuples with NaN values
    default_left_bases = np.full((1, 2), np.nan)  # 2D array filled with NaN values
    default_right_bases = np.full((1, 2), np.nan)  # 2D array filled with NaN values

    # Check for minimum length, or all NaNs in arrays
    if (
        len(primary_max_length_pts) < 2
        or len(lateral_pts) < 2
        or np.isnan(primary_max_length_pts).all()
        or np.isnan(lateral_pts).all()
    ):
        if return_inds:
            # Return the distances, matched indices, and the final left and right bases
            return (
                default_dists,
                default_indices,
                default_left_bases,
                default_right_bases,
            )
        else:
            # Default: Return the distances
            return default_dists

    # Filter out any NaN points from the primary root points
    primary_pts_filtered = primary_max_length_pts[
        ~np.isnan(primary_max_length_pts).any(axis=-1)
    ]
    # Create a LineString object for the primary root
    primary_line = LineString(primary_pts_filtered)

    # Identify lateral roots that have a defined base (not NaN)
    has_base = ~np.isnan(lateral_pts[:, 0, 0])
    # Filter the lateral roots based on the valid base points
    lateral_pts = lateral_pts[has_base]

    # Determine if the base of each lateral root is to the left or right of the rest of
    # the root
    is_left = lateral_pts[:, 0, 0] > np.nanmin(lateral_pts[:, 1:, 0], axis=1)

    # If all lateral roots are on the same side, return default values
    if is_left.all() or (~is_left).all():
        if return_inds:
            # Return the distances, matched indices, and the final left and right bases
            return (
                default_dists,
                default_indices,
                default_left_bases,
                default_right_bases,
            )
        else:
            # Default: Return the distances
            return default_dists

    # Split lateral roots into left and right bases
    left_bases, right_bases = lateral_pts[is_left, 0], lateral_pts[~is_left, 0]

    # Find the nearest points on the primary root for each right base
    nearest_primary_right = [
        nearest_points(primary_line, Point(right_base))[0] for right_base in right_bases
    ]

    # Find the nearest points on the primary root for each left base
    nearest_primary_left = [
        nearest_points(primary_line, Point(left_base))[0] for left_base in left_bases
    ]

    # Calculate the normalized projection of each nearest point on the primary root
    # (right side)
    nearest_primary_norm_right = np.array(
        [primary_line.project(pt, normalized=True) for pt in nearest_primary_right]
    )

    # Calculate the normalized projection of each nearest point on the primary root
    # (left side)
    nearest_primary_norm_left = np.array(
        [primary_line.project(pt, normalized=True) for pt in nearest_primary_left]
    )

    # Create a cost matrix based on the differences in projections between left and
    # right bases
    cost_matrix = np.abs(
        nearest_primary_norm_left.reshape(-1, 1)
        - nearest_primary_norm_right.reshape(1, -1)
    )

    # Use the Hungarian algorithm to find an optimal pairing that minimizes the sum of
    # projection differences
    left_inds, right_inds = linear_sum_assignment(cost_matrix)

    # Filter out pairs where the projection difference exceeds the given tolerance
    valid_pairs = cost_matrix[left_inds, right_inds] <= tolerance
    left_inds = left_inds[valid_pairs]
    right_inds = right_inds[valid_pairs]

    # If no valid pairs remain, return default values
    if len(left_inds) == 0 or len(right_inds) == 0:
        if return_inds:
            # Return the distances, matched indices, and the final left and right bases
            return (
                default_dists,
                default_indices,
                default_left_bases,
                default_right_bases,
            )
        else:
            # Default: Return the distances
            return default_dists

    # Filter out pairs that do not intersect the primary root
    is_intersecting = np.array(
        [
            primary_line.intersects(
                LineString([left_bases[left_ind], right_bases[right_ind]])
            )
            for left_ind, right_ind in zip(left_inds, right_inds)
        ]
    )
    left_inds = left_inds[is_intersecting]
    right_inds = right_inds[is_intersecting]

    # If no valid pairs remain, return default values
    if len(left_inds) == 0 or len(right_inds) == 0:
        if return_inds:
            # Return the distances, matched indices, and the final left and right bases
            return (
                default_dists,
                default_indices,
                default_left_bases,
                default_right_bases,
            )
        else:
            # Default: Return the distances
            return default_dists

    # Update the left and right bases of the final paired coordinates
    left_bases_final = left_bases[left_inds]
    right_bases_final = right_bases[right_inds]

    # Calculate the Euclidean distance between the bases of the valid pairs
    match_dists = np.linalg.norm(
        left_bases[left_inds] - right_bases[right_inds],
        axis=-1,
    )

    # Create a list of tuples representing the indices of the matched pairs
    matched_indices = list(zip(left_inds, right_inds))

    if return_inds:
        # Return the distances, matched indices, and the final left and right bases
        return match_dists, matched_indices, left_bases_final, right_bases_final
    else:
        # Default: Return the distances
        return match_dists
