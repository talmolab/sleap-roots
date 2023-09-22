"""Fraction of root network length in the lower fraction of the plant."""

import numpy as np
from shapely import LineString, Polygon
from sleap_roots.lengths import get_max_length_pts
from typing import Tuple, Union


def get_bbox(pts: np.ndarray) -> Tuple[float, float, float, float]:
    """Return the bounding box of all landmarks.

    Args:
        pts: Root landmarks as array of shape (..., 2).

    Returns:
        Tuple of four parameters in bounding box:
        left_x, the x axis value of left side
        top_y, the y axis value of top side
        width, the width of the bounding box
        height, the height of bounding box.
    """
    # reshape to (# instance, 2) and filter out NaNs.
    pts2 = pts.reshape(-1, 2)
    pts2 = pts2[~(np.isnan(pts2).any(axis=-1))]

    # get the bounding box
    if pts2.shape[0] == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    else:
        left_x, top_y = np.min(pts2[:, 0]), np.min(pts2[:, 1])
        width, height = np.max(pts2[:, 0]) - np.min(pts2[:, 0]), np.max(
            pts2[:, 1]
        ) - np.min(pts2[:, 1])
        bbox = (left_x, top_y, width, height)
    return bbox


def get_network_width_depth_ratio(
    pts: Union[np.ndarray, Tuple[float, float, float, float]]
) -> float:
    """Return width to depth ratio of bounding box for root network.

    Args:
        pts: Root landmarks as array of shape (..., 2) or boundary box.

    Returns:
        Float of bounding box width to depth ratio of root network.
    """
    # get the bounding box
    if type(pts) == tuple:
        bbox = pts
    else:
        bbox = get_bbox(pts)
    width, height = bbox[2], bbox[3]
    if width > 0 and height > 0:
        ratio = width / height
        return ratio
    else:
        return np.nan


def get_network_length(
    primary_length: float,
    lateral_lengths: Union[float, np.ndarray],
    monocots: bool = False,
) -> float:
    """Return the total root network length given primary and lateral root lengths.

    Args:
        primary_length: Primary root length.
        lateral_lengths: Either a float representing the length of a single lateral
          root or an array of lateral root lengths with shape `(instances,)`.
        monocots: A boolean value, where True is rice.

    Returns:
        Total length of root network.
    """
    # Ensure primary_length is a scalar
    if not isinstance(primary_length, (float, np.float64)):
        raise ValueError("Input primary_length must be a scalar value.")

    # Ensure lateral_lengths is either a scalar or has the correct shape
    if not (
        isinstance(lateral_lengths, (float, np.float64)) or lateral_lengths.ndim == 1
    ):
        raise ValueError(
            "Input lateral_lengths must be a scalar or have shape (instances,)."
        )

    # Calculate the total lateral root length using np.nansum
    total_lateral_length = np.nansum(lateral_lengths)

    if monocots:
        length = total_lateral_length
    else:
        # Calculate the total root network length using np.nansum so the total length
        # will not be NaN if one of primary or lateral lengths are NaN
        length = np.nansum([primary_length, total_lateral_length])

    return length


def get_network_solidity(
    network_length: float,
    chull_area: float,
) -> float:
    """Return the total network length divided by the network convex area.

    Args:
        network_length: Total root length of network.
        chull_area: Convex hull area.

    Returns:
        Float of the total network length divided by the network convex area.
    """
    if network_length > 0 and chull_area > 0:
        ratio = network_length / chull_area
        return ratio
    else:
        return np.nan


def get_network_distribution(
    primary_pts: np.ndarray,
    lateral_pts: np.ndarray,
    bounding_box: Tuple[float, float, float, float],
    fraction: float = 2 / 3,
    monocots: bool = False,
) -> float:
    """Return the root length in the lower fraction of the plant.

    Args:
        primary_pts: Array of primary root landmarks. Can have shape `(nodes, 2)` or
            `(1, nodes, 2)`.
        lateral_pts: Array of lateral root landmarks with shape `(instances, nodes, 2)`.
        bounding_box: Tuple in the form `(left_x, top_y, width, height)`.
        fraction: Lower fraction value. Defaults to 2/3.
        monocots: A boolean value, where True indicates rice. Defaults to False.

    Returns:
        Root network length in the lower fraction of the plant.
    """
    # Input validation
    if primary_pts.ndim not in [2, 3]:
        raise ValueError(
            "primary_pts should have a shape of `(nodes, 2)` or `(1, nodes, 2)`."
        )

    if primary_pts.ndim == 2 and primary_pts.shape[-1] != 2:
        raise ValueError("primary_pts should have a shape of `(nodes, 2)`.")

    if primary_pts.ndim == 3 and primary_pts.shape[-1] != 2:
        raise ValueError("primary_pts should have a shape of `(1, nodes, 2)`.")

    if lateral_pts.ndim != 3 or lateral_pts.shape[-1] != 2:
        raise ValueError("lateral_pts should have a shape of `(instances, nodes, 2)`.")

    if len(bounding_box) != 4:
        raise ValueError(
            "bounding_box should be in the form `(left_x, top_y, width, height)`."
        )

    # Make sure the longest primary root is used
    if primary_pts.ndim == 3:
        primary_pts = get_max_length_pts(primary_pts)  # shape is (nodes, 2)

    # Make primary_pts and lateral_pts have the same dimension of 3
    primary_pts = (
        primary_pts[np.newaxis, :, :] if primary_pts.ndim == 2 else primary_pts
    )

    # Filter out NaN values
    primary_pts = [root[~np.isnan(root).any(axis=1)] for root in primary_pts]
    lateral_pts = [root[~np.isnan(root).any(axis=1)] for root in lateral_pts]

    # Collate root points.
    all_roots = primary_pts + lateral_pts if not monocots else lateral_pts

    # Get the vertices of the bounding box
    left_x, top_y, width, height = bounding_box

    # Calculate the bounding box of the lower fraction
    lower_height = height * fraction
    if np.isnan(lower_height):
        return np.nan

    # Convert lower bounding box to polygon
    # Vertices are in counter-clockwise order
    lower_box = Polygon(
        [
            [left_x, top_y + (height - lower_height)],
            [left_x, top_y + height],
            [left_x + width, top_y + height],
            [left_x + width, top_y + (height - lower_height)],
        ]
    )

    # Calculate length of roots within the lower bounding box
    network_length = 0
    for root in all_roots:
        if len(root) > 1:  # Ensure that root has more than one point
            root_poly = LineString(root)
            lower_intersection = root_poly.intersection(lower_box)
            root_length = lower_intersection.length
            network_length += root_length if ~np.isnan(root_length) else 0

    return network_length


def get_network_distribution_ratio(
    primary_length: float,
    lateral_lengths: Union[float, np.ndarray],
    network_length_lower: float,
    fraction: float = 2 / 3,
    monocots: bool = False,
) -> float:
    """Return ratio of the root length in the lower fraction over all root length.

    Args:
        primary_length: Primary root length.
        lateral_lengths: Lateral root lengths. Can be a single float (for one root)
            or an array of floats (for multiple roots).
        network_length_lower: The root length in the lower network.
        fraction: The fraction of the network considered as 'lower'. Defaults to 2/3.
        monocots: A boolean value, where True indicates rice. Defaults to False.

    Returns:
        Float of ratio of the root network length in the lower fraction of the plant
        over all root length.
    """
    # Ensure primary_length is a scalar
    if not isinstance(primary_length, (float, np.float64)):
        raise ValueError("Input primary_length must be a scalar value.")

    # Ensure lateral_lengths is either a scalar or a 1-dimensional array
    if not isinstance(lateral_lengths, (float, np.float64, np.ndarray)):
        raise ValueError(
            "Input lateral_lengths must be a scalar or a 1-dimensional array."
        )

    # If lateral_lengths is an ndarray, it must be one-dimensional
    if isinstance(lateral_lengths, np.ndarray) and lateral_lengths.ndim != 1:
        raise ValueError("Input lateral_lengths array must have shape (instances,).")

    # Ensure network_length_lower is a scalar
    if not isinstance(network_length_lower, (float, np.float64)):
        raise ValueError("Input network_length_lower must be a scalar value.")

    # Calculate the total lateral root length
    total_lateral_length = np.nansum(lateral_lengths)

    # Determine total root length based on monocots flag
    if monocots:
        total_root_length = total_lateral_length
    else:
        total_root_length = np.nansum([primary_length, total_lateral_length])

    # Calculate the ratio
    ratio = network_length_lower / total_root_length
    return ratio
