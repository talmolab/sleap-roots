"""Fraction of root network length in the lower fraction of the plant."""

import numpy as np
from shapely import LineString, Polygon
from sleap_roots.bases import get_root_lengths
from sleap_roots.convhull import get_convhull_features
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
    if pts2.shape[0] == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    else:
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
        Float of bounding box width to depth ratio of root network.
    """
    # get the bounding box
    bbox = get_bbox(pts)
    width, height = bbox[2], bbox[3]
    if width > 0 and height > 0:
        ratio = width / height
        return ratio
    else:
        return np.nan


def get_network_solidity(
    primary_pts: np.ndarray,
    lateral_pts: np.ndarray,
    pts_all_array: np.ndarray,
    lateral_only: bool = False,
) -> float:
    """Return the total network length divided by the network convex area.

    Args:
        primary_pts: primary root landmarks as array of shape (..., 2).
        lateral_pts: lateral root landmarks as array of shape (..., 2).
        pts_all_array: primary and lateral root landmarks.
        lateral_only: a boolean value, where True is rice.

    Returns:
        Float of the total network length divided by the network convex area.
    """
    # get the total network length
    network_length = get_network_length(primary_pts, lateral_pts, lateral_only)

    # get the convex hull area
    convhull_features = get_convhull_features(pts_all_array)
    conv_area = convhull_features[1]

    if network_length > 0 and conv_area > 0:
        ratio = network_length / conv_area
        return ratio
    else:
        return np.nan


def get_network_distribution(
    primary_pts: np.ndarray,
    lateral_pts: np.ndarray,
    pts_all_array: np.ndarray,
    fraction: float = 2 / 3,
    lateral_only: bool = False,
) -> float:
    """Return the root length in the lower fraction of the plant.

    Args:
        primary_pts: primary root landmarks as array of shape (..., 2).
        lateral_pts: lateral root landmarks as array of shape (..., 2).
        pts_all_array: primary and lateral root landmarks.
        fraction: the network length found in the lower fration value of the network.
        lateral_only: a boolean value, where True is rice.

    Returns:
        Float of the root network length in the lower fraction of the plant.
    """
    # get the bounding box
    bbox = get_bbox(pts_all_array)
    left_x, top_y, width, height = bbox[0], bbox[1], bbox[2], bbox[3]

    # get the bounding box of the lower fraction
    lower_height = bbox[3] * fraction
    if np.isnan(lower_height):
        return np.nan
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
    if lateral_only:
        points = list(primary_pts)
    else:
        points = list(primary_pts) + list(lateral_pts)
    pts_nnan = []
    for j in range(len(points)):
        pts_j = points[j][~np.isnan(points[j]).any(axis=1)]
        pts_nnan.append(pts_j)

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
    primary_pts: np.ndarray,
    lateral_pts: np.ndarray,
    lateral_only: bool = False,
) -> float:
    """Return all primary or lateral root length one frame.

    Args:
        primary_pts: primary root landmarks as array of shape (..., 2).
        lateral_pts: lateral root landmarks as array of shape (..., 2).
        lateral_only: a boolean value, where True is rice.

    Returns:
        Float of primary or lateral root network length.
    """
    if (
        np.sum(get_root_lengths(primary_pts)) > 0
        or np.sum(get_root_lengths(lateral_pts)) > 0
    ):
        if lateral_only:
            length = np.nansum(get_root_lengths(primary_pts))
        else:
            length = np.nansum(get_root_lengths(primary_pts)) + np.nansum(
                get_root_lengths(lateral_pts)
            )
        return length
    else:
        return 0


def get_network_distribution_ratio(
    primary_pts: np.ndarray,
    lateral_pts: np.ndarray,
    pts_all_array: np.ndarray,
    fraction: float = 2 / 3,
    lateral_only: bool = False,
) -> float:
    """Return ratio of the root length in the lower fraction over all root length.

    Args:
        primary_pts: primary root landmarks as array of shape (..., 2).
        lateral_pts: lateral root landmarks as array of shape (..., 2).
        pts_all_array: primary and lateral root landmarks.
        fraction: the network length found in the lower fration value of the network.
        lateral_only: a boolean value, where True is rice.

    Returns:
        Float of ratio of the root network length in the lower fraction of the plant
        over all root length.
    """
    if (
        np.sum(get_root_lengths(primary_pts)) + np.sum(get_root_lengths(lateral_pts))
        > 0
    ):
        if lateral_only:
            ratio = get_network_distribution(
                primary_pts, lateral_pts, pts_all_array, fraction, lateral_only
            ) / (np.sum(get_root_lengths(primary_pts)))
        else:
            ratio = get_network_distribution(
                primary_pts, lateral_pts, pts_all_array, fraction
            ) / (
                np.sum(get_root_lengths(primary_pts))
                + np.sum(get_root_lengths(lateral_pts))
            )
        return ratio
    else:
        return np.nan
