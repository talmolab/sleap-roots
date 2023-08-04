"""Fraction of root network length in the lower fraction of the plant."""

import numpy as np
from shapely import LineString, Polygon
from sleap_roots.lengths import get_root_lengths
from sleap_roots.convhull import get_convhull_features
from typing import Optional, Tuple, Union


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
        bbox: Optional, the bounding box of all root landmarks.

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


def get_network_solidity(
    network_length: float,
    chull_area: float,
) -> float:
    """Return the total network length divided by the network convex area.

    Args:
        network_length: all root lengths.
        chull_area: an optional argument of convex hull area.

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
    pts_all_array: Union[np.ndarray, Tuple[float, float, float, float]],
    fraction: float = 2 / 3,
    monocots: bool = False,
) -> float:
    """Return the root length in the lower fraction of the plant.

    Args:
        primary_pts: primary root landmarks as array of shape (..., 2).
        lateral_pts: lateral root landmarks as array of shape (..., 2).
        pts_all_array: primary and lateral root landmarks or the boundary box.
        fraction: the network length found in the lower fration value of the network.
        monocots: a boolean value, where True is rice.

    Returns:
        Float of the root network length in the lower fraction of the plant.
    """
    # get the bounding box
    if type(pts_all_array) == tuple:
        bbox = pts_all_array
    else:
        bbox = get_bbox(pts_all_array)
    left_x, top_y, width, height = bbox[0], bbox[1], bbox[2], bbox[3]

    # get the bounding box of the lower fraction
    lower_height = bbox[3] * fraction
    if np.isnan(lower_height):
        return np.nan
    lower_bbox = (bbox[0], bbox[1] + (bbox[3] - lower_height), bbox[2], lower_height)

    # convert lower bounding box to polygon
    polygon = Polygon(
        [
            [bbox[0], bbox[1] + (bbox[3] - lower_height)],
            [bbox[0], bbox[1] + height],
            [bbox[0] + width, bbox[1] + height],
            [bbox[0] + width, bbox[1] + (bbox[3] - lower_height)],
        ]
    )

    # filter out the nan nodes
    if monocots:
        points = list(lateral_pts)
    else:
        points = list(primary_pts) + list(lateral_pts)

    # get length of lines within the lower bounding box
    root_length = 0
    for j in range(len(points)):
        # filter out nan nodes
        pts_j = points[j][~np.isnan(points[j]).any(axis=1)]
        if pts_j.shape[0] > 1:
            linestring = LineString(pts_j)
            if linestring.intersection(polygon):
                intersection = linestring.intersection(polygon)
                root_length += (
                    intersection.length if ~np.isnan(intersection.length) else 0
                )

    return root_length


def get_network_length(
    primary_length: Union[float, np.ndarray],
    lateral_lengths: Union[float, np.ndarray],
    monocots: bool = False,
) -> float:
    """Return all primary or lateral root length one frame.

    Args:
        primary_length: primary root length or maximum length primary root landmarks as
            array of shape `(node, 2)`.
        lateral_lengths: lateral root length or lateral root landmarks as array of shape
             `(instance, node, 2)`.
        monocots: a boolean value, where True is rice.

    Returns:
        Float of all roots network length.
    """
    # check whether primary_length is the maximum length or maximum primary root
    if not (
        isinstance(primary_length, (float, np.float64)) or primary_length.ndim != 2
    ):
        raise ValueError(
            "Input primary_length should be the maximum primary root "
            "length or array have shape (nodes, 2)."
        )
    # get primary_root_length
    primary_root_length = (
        primary_length
        if not isinstance(primary_length, np.ndarray)
        else get_root_lengths(primary_length)
    )

    # check whether lateral_lengths is the lengths or lateral root nodex.
    if not (
        isinstance(lateral_lengths, (float, np.float64))  # length with only one root
        or lateral_lengths.ndim != 1  # lenthgs with more than one lateral roots
        or lateral_lengths.ndim != 3  # lateral root nodes
    ):
        raise ValueError(
            "Input lateral_lengths should be the lateral root lengths or array have "
            "shape (instance, nodes, 2)."
        )

    # get lateral_root_length
    if lateral_lengths.ndim != 3:  # lateral root nodes
        lateral_root_length = np.sum(get_root_lengths(lateral_lengths))
    elif lateral_lengths.ndim != 1:  # lenthgs with more than one lateral roots
        lateral_root_length = np.sum(lateral_lengths)
    else:  # length with only one lateral root
        lateral_root_length = lateral_lengths

    # return Nan if lengths less than 0
    if primary_root_length + lateral_root_length < 0:
        return np.nan

    if monocots:
        length = lateral_root_length
    else:
        length = primary_root_length + lateral_root_length

    return length


def get_network_distribution_ratio(
    primary_length: Union[float, np.ndarray],
    lateral_lengths: Union[float, np.ndarray],
    network_length_lower: Union[float, np.ndarray],
    fraction: float = 2 / 3,
    monocots: bool = False,
) -> float:
    """Return ratio of the root length in the lower fraction over all root length.

    Args:
        primary_length: primary root length or maximum length primary root landmarks as
            array of shape `(node, 2)`.
        lateral_lengths: lateral root length or lateral root landmarks as array of shape
             `(instance, node, 2)`.
        network_length_lower: the root length in lower network or primary and lateral
            root landmarks.
        fraction: the network length found in the lower fration value of the network.
        monocots: a boolean value, where True is rice.

    Returns:
        Float of ratio of the root network length in the lower fraction of the plant
        over all root length.
    """
    # check whether primary_length is the maximum length or maximum primary root
    if not (
        isinstance(primary_length, (float, np.float64)) or primary_length.ndim != 2
    ):
        raise ValueError(
            "Input primary_length should be the maximum primary root "
            "length or array have shape (nodes, 2)."
        )
    # get primary_root_length
    primary_root_length = (
        primary_length
        if not isinstance(primary_length, np.ndarray)
        else get_root_lengths(primary_length)
    )

    # check whether lateral_lengths is the lengths or lateral root nodex.
    if not (
        isinstance(lateral_lengths, (float, np.float64))  # length with only one root
        or lateral_lengths.ndim != 1  # lenthgs with more than one lateral roots
        or lateral_lengths.ndim != 3  # lateral root nodes
    ):
        raise ValueError(
            "Input lateral_lengths should be the lateral root lengths or array have "
            "shape (instance, nodes, 2)."
        )

    # get lateral_root_length
    if lateral_lengths.ndim != 3:  # lateral root nodes
        lateral_root_length = np.sum(get_root_lengths(lateral_lengths))
    elif lateral_lengths.ndim != 1:  # lenthgs with more than one lateral roots
        lateral_root_length = np.sum(lateral_lengths)
    else:  # length with only one lateral root
        lateral_root_length = lateral_lengths

    # get network_length_lower
    if isinstance(network_length_lower, (float, np.float64)):
        network_length_lower = network_length_lower
    elif (
        primary_length.ndim == 2
        and lateral_lengths.ndim == 3
        and network_length_lower.ndim == 3
    ):
        network_length_lower = get_network_distribution(
            primary_length, lateral_lengths, network_length_lower, fraction, monocots
        )
    else:
        raise ValueError(
            "Input network_length_lower should be a float value, otherwise "
            "primary_length is maximimum length primary root in shape  `(node, 2)` and "
            "primary_length is lateral root in shape `(instance, nodes, 2)`."
        )

    # return Nan if lengths less than 0
    if primary_root_length + lateral_root_length < 0:
        return np.nan

    if monocots:
        ratio = network_length_lower / primary_root_length
    else:
        ratio = network_length_lower / (primary_root_length + lateral_root_length)
    return ratio
