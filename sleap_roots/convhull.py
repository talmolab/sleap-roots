"""Convex hull fitting and derived trait calculation."""

import numpy as np
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy.spatial.distance import pdist
from typing import Tuple


def get_convhull(
    pts: np.ndarray,
) -> Tuple[float, float, float, float, float, float, float]:
    """Get the convex hull for the points per frame.

    Args:
        pts: Root landmarks as array of shape (..., 2).

    Returns:
        A tuple of (perimeters, areas, longest_dists, shortest_dists, median_dists,
        max_widths, max_heights) containing perimeter, area, longest distance between
        vertices, smallest distance between vertices, median distance between vertices,
        maximum width and maximum height of the convex hull.

        If the convex hull fitting fails, NaNs are returned.
    """
    pts = pts.reshape(-1, 2)
    pts = pts[~(np.isnan(pts).any(axis=-1))]

    if len(pts) <= 2:
        return np.full((7,), np.nan)

    # Get Convex Hull
    hull = ConvexHull(pts)
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
