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
) -> Tuple[float, float, float, float]:
    """Get the convex hull features for the points per frame.

    Args:
        pts: Root landmarks as array of shape (..., 2).

    Returns:
        A tuple of 4 convex hull features
            perimeters, perimeter of the convex hull
            areas, area of the convex hull
            max_widths, maximum width of convex hull
            max_heights, maximum height of convex hull

        If the convex hull fitting fails, NaNs are returned.
    """
    hull = pts if type(pts) == ConvexHull else get_convhull(pts)

    if hull is None:
        return np.full((4,), np.nan)

    # perimeter
    perimeter = hull.area
    # area
    area = hull.volume

    pts = pts.reshape(-1, 2)
    pts = pts[~(np.isnan(pts).any(axis=-1))]

    # max 'width'
    max_width = np.nanmax(pts[:, 0]) - np.nanmin(pts[:, 0])
    # max 'height'
    max_height = np.nanmax(pts[:, 1]) - np.nanmin(pts[:, 1])

    return (
        perimeter,
        area,
        max_width,
        max_height,
    )


def get_chull_line_lengths(pts: Union[np.ndarray, ConvexHull]):
    """Get the convex hull line lengths per frame.

    Args:
        pts: Root landmarks as array of shape (..., 2) or ConvexHull object.

    Returns:
        Lengths of lines connecting any two vertices on the convex hull.
        If the convex hull fitting fails, NaNs are returned.
    """
    hull = pts if type(pts) == ConvexHull else get_convhull(pts)

    if hull is None:
        return np.nan

    # Lengths of lines connecting any two vertices on the convex hull
    chull_line_lengths = pdist(hull.points[hull.vertices], "euclidean")

    return chull_line_lengths
