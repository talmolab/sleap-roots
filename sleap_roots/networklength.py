"""Fraction of root network length in the lower fraction of the plant."""

import numpy as np
from shapely import LineString, Polygon
from sleap_roots.bases import get_root_lengths
from typing import Tuple


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
    left_x, top_y = np.min(pts2[:, 0]), np.min(pts2[:, 1])
    width, height = np.max(pts2[:, 0]) - np.min(pts2[:, 0]), np.max(
        pts2[:, 1]
    ) - np.min(pts2[:, 1])
    bbox = (left_x, top_y, width, height)
    return bbox


def get_network_width_depth_ratio(pts: np.ndarray) -> float:
    """Return width to depth ratio of bounding box for root network.

    Args:
        pts: Root landmarks as array of shape (..., 2).

    Returns:
        float of bounding box width to depth ratio of root network.
    """
    # get the bounding box
    bbox = get_bbox(pts)
    width, height = bbox[2], bbox[3]
    if width > 0 and height > 0:
        ratio = width / height
        return ratio
    else:
        return np.nan


def get_network_distribution(pts: np.ndarray, fraction: float = 2 / 3) -> float:
    """Return the root length in the lower fraction of the plant.

    Args:
        pts: Root landmarks as array of shape (..., 2).
        fraction: the network length found in the lower fration value of the network.

    Returns:
        float of the root network length in the lower fraction of the plant.
    """
    # get the bounding box
    bbox = get_bbox(pts)
    left_x, top_y, width, height = bbox[0], bbox[1], bbox[2], bbox[3]

    # get the bounding box of the lower fraction
    lower_height = bbox[3] * fraction
    lower_bbox = (bbox[0], bbox[1] + (bbox[3] - lower_height), bbox[2], lower_height)

    # convert bounding box to polygon
    polygon = Polygon(
        [
            [bbox[0], bbox[1] + (bbox[3] - lower_height)],
            [bbox[0], bbox[1] + height],
            [bbox[0] + width, bbox[1] + height],
            [bbox[0] + width, bbox[1] + (bbox[3] - lower_height)],
        ]
    )

    # filter out the nan nodes
    points = list(pts)
    pts_nnan = []
    for j in range(len(points)):
        pts_j = points[j][~np.isnan(points[j]).any(axis=1)]
        pts_nnan.append(pts_j)

    # get length of lines within the lower bounding box
    root_length = 0
    for j in range(len(points)):
        # filter out nan nodes
        pts_j = points[j][~np.isnan(points[j]).any(axis=1)]
        linestring = LineString(points[j])
        if pts_j.shape[0] > 1:
            if linestring.intersection(polygon):
                intersection = linestring.intersection(polygon)
                root_length += (
                    intersection.length if ~np.isnan(intersection.length) else 0
                )

    return root_length


def get_network_length(pts: np.ndarray) -> float:
    """Return all primary or lateral root length one frame.

    Args:
        pts: Root landmarks as array of shape (..., 2).

    Returns:
        float of primary or lateral root network length.
    """
    if np.sum(get_root_lengths(pts)) > 0:
        length = np.sum(get_root_lengths(pts))
        return length
    else:
        return 0


def get_network_distribution_ratio(pts: np.ndarray, fraction: float = 2 / 3) -> float:
    """Return ratio of the root length in the lower fraction over all root length.

    Args:
        pts: Root landmarks as array of shape (..., 2).
        fraction: the network length found in the lower fration value of the network.

    Returns:
        float of ratio of the root network length in the lower fraction of the plant
        over all root length.
    """
    if np.sum(get_root_lengths(pts)) > 0:
        ratio = get_network_distribution(pts, fraction) / np.sum(get_root_lengths(pts))
        return ratio
    else:
        return np.nan
