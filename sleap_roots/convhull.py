"""Convex hull fitting and derived trait calculation."""

import numpy as np
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy.spatial.distance import pdist
from typing import Tuple, Optional, Union


def get_convhull(pts: np.ndarray) -> Optional[ConvexHull]:
    """Get the convex hull for the points per frame.

    Args:
        pts: Root landmarks as array of shape (..., 2).

    Returns:
        An object of convex hull.
    """
    pts = pts.reshape(-1, 2)
    pts = pts[~(np.isnan(pts).any(axis=-1))]

    if len(pts) <= 2:
        return None

    # Get convex hull
    hull = ConvexHull(pts)

    return hull


def get_convhull_features(
    pts: Union[np.ndarray, ConvexHull]
) -> Tuple[float, float, float, float, float, float, float]:
    """Get the convex hull features for the points per frame.

    Args:
        pts: Root landmarks as array of shape (..., 2).

    Returns:
        A tuple of 7 convex hull features
            perimeters, perimeter of the convex hull
            areas, area of the convex hull
            longest_dists, longest distance between vertices
            shortest_dists, smallest distance between vertices
            median_dists, median distance between vertices
            max_widths, maximum width of convex hull
            max_heights, maximum height of convex hull

        If the convex hull fitting fails, NaNs are returned.
    """
    hull = pts if type(pts) == ConvexHull else get_convhull(pts)

    if hull is None:
        return np.full((7,), np.nan)

    # perimeter
    perimeter = hull.area
    # area
    area = hull.volume
    # longest distance between vertices
    longest_dist = np.nanmax(pdist(hull.points[hull.vertices], "euclidean"))
    # smallest distance between vertices
    shortest_dist = np.nanmin(pdist(hull.points[hull.vertices], "euclidean"))
    # median distance between vertices
    median_dist = np.nanmedian(pdist(hull.points[hull.vertices], "euclidean"))

    pts = pts.reshape(-1, 2)
    pts = pts[~(np.isnan(pts).any(axis=-1))]

    # max 'width'
    max_width = np.nanmax(pts[:, 0]) - np.nanmin(pts[:, 0])
    # max 'height'
    max_height = np.nanmax(pts[:, 1]) - np.nanmin(pts[:, 1])

    return (
        perimeter,
        area,
        longest_dist,
        shortest_dist,
        median_dist,
        max_width,
        max_height,
    )
