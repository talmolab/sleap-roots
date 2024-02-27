"""Fraction of root network length in the lower fraction of the plant."""

import numpy as np
from shapely import LineString, Polygon
from sleap_roots.lengths import get_max_length_pts
from typing import Tuple, Union, List, Optional


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
    lengths0: Union[float, np.ndarray],
    *args: Optional[Union[float, np.ndarray]],
) -> float:
    """Return the total root network length given primary and lateral root lengths.

    Args:
        lengths0: Either a float representing the length of a single
            root or an array of root lengths with shape `(instances,)`.
        *args: Additional optional floats representing the lengths of single
            roots or arrays of root lengths with shape `(instances,)`.

    Returns:
        Total length of root network.
    """
    # Initialize an empty list to store the lengths
    all_lengths = []
    # Loop over the input arrays
    for length in [lengths0] + list(args):
        if length is None:
            continue  # Skip None values
        # Ensure length is either a scalar or has the correct shape
        if not (np.isscalar(length) or (hasattr(length, "ndim") and length.ndim == 1)):
            raise ValueError(
                "Input length must be a scalar or have shape (instances,)."
            )
        # Add the length to the list
        if np.isscalar(length):
            all_lengths.append(length)
        else:
            all_lengths.extend(list(length))

    # Calculate the total root network length using np.nansum so the total length
    # will not be NaN if one of primary or lateral lengths are NaN
    total_network_length = np.nansum(all_lengths)

    return total_network_length


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
    pts_list: List[np.ndarray],
    bounding_box: Tuple[float, float, float, float],
    fraction: float = 2 / 3,
) -> float:
    """Return the root length in the lower fraction of the plant.

    Args:
        pts_list: A list of arrays, each having shape `(nodes, 2)`.
        bounding_box: Tuple in the form `(left_x, top_y, width, height)`.
        fraction: Lower fraction value. Defaults to 2/3.

    Returns:
        Root network length in the lower fraction of the plant.
    """
    # Input validation for pts_list
    if any(pts.ndim != 2 or pts.shape[-1] != 2 for pts in pts_list):
        raise ValueError(
            "Each pts array in pts_list should have a shape of `(nodes, 2)`."
        )

    # Input validation for bounding_box
    if len(bounding_box) != 4:
        raise ValueError(
            "bounding_box must contain exactly 4 elements: `(left_x, top_y, width, height)`."
        )

    # Filter out NaN values
    pts_list = [pts[~np.isnan(pts).any(axis=-1)] for pts in pts_list]

    # Get the vertices of the bounding box
    left_x, top_y, width, height = bounding_box

    # Calculate the bounding box of the lower fraction
    lower_height = height * fraction
    if np.isnan(lower_height):
        return np.nan

    # Convert lower bounding box to polygon
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
    for root in pts_list:
        if len(root) > 1:  # Ensure that root has more than one point
            root_poly = LineString(root)
            lower_intersection = root_poly.intersection(lower_box)
            root_length = lower_intersection.length
            network_length += root_length if ~np.isnan(root_length) else 0

    return network_length


def get_network_distribution_ratio(
    network_length: float,
    network_length_lower: float,
) -> float:
    """Return ratio of the root length in the lower fraction to total root length.

    Args:
        network_length_lower: The root length in the lower network.
        network_length: Total root length of network.

    Returns:
        Float of ratio of the root network length in the lower fraction of the plant
            over the total root length.
    """
    # Ensure primary_length is a scalar
    if not isinstance(network_length, (float, np.float64)):
        raise ValueError("Input network_length must be a scalar value.")

    # Ensure network_length_lower is a scalar
    if not isinstance(network_length_lower, (float, np.float64)):
        raise ValueError("Input network_length_lower must be a scalar value.")

    # Calculate the ratio
    ratio = network_length_lower / network_length
    return ratio
