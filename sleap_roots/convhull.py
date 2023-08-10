"""Convex hull fitting and derived trait calculation."""

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist
from typing import Tuple, Optional, Union


def get_convhull(pts: np.ndarray) -> Optional[ConvexHull]:
    """Compute the convex hull for the points per frame.

    Args:
        pts: Root landmarks as an array of shape (..., 2).

    Returns:
        An object representing the convex hull or None if a hull can't be formed.
    """

    # Ensure the input is an array of shape (n, 2)
    if pts.ndim < 2 or pts.shape[-1] != 2:
        raise ValueError("Input points should be of shape (..., 2).")

    # Reshape and filter out NaN values
    pts = pts.reshape(-1, 2)
    pts = pts[~np.isnan(pts).any(axis=-1)]

    # Check for NaNs or infinite values
    if np.isnan(pts).any() or np.isinf(pts).any():
        return None

    # Ensure there are at least 3 unique non-collinear points
    if len(np.unique(pts, axis=0)) < 3:
        return None

    # Compute and return the convex hull
    return ConvexHull(pts)


def get_chull_perimeter(hull: Union[np.ndarray, ConvexHull, None]) -> float:
    """Calculate the perimeter of the convex hull formed by the given points.

    Args:
        hull: Either an array of landmark points, a pre-computed convex hull, or None.

    Returns:
        Scalar value representing the perimeter of the convex hull. Returns NaN if
        unable to compute the convex hull or if the input is None.
    """

    # If the input hull is None, return NaN
    if hull is None:
        return np.nan

    # If the input is an array, compute its convex hull
    if isinstance(hull, np.ndarray):
        hull = get_convhull(hull)

    # If hull becomes None after attempting to compute the convex hull, return NaN
    if hull is None:
        return np.nan

    # Ensure that the hull is of type ConvexHull
    if not isinstance(hull, ConvexHull):
        raise TypeError("After processing, the input must be a ConvexHull object.")

    # Compute the perimeter of the convex hull
    return hull.area


def get_chull_area(hull: Union[np.ndarray, ConvexHull]) -> float:
    """
    Calculate the area of the convex hull formed by the given points.

    Args:
        hull: Either an array of landmark points or a pre-computed convex hull.

    Returns:
        Scalar value representing the area of the convex hull. Returns NaN if unable
        to compute the convex hull.
    """

    # If the input hull is None, return NaN
    if hull is None:
        return np.nan

    # If the input is an array, compute its convex hull
    if isinstance(hull, np.ndarray):
        hull = get_convhull(hull)

    # If hull becomes None after attempting to compute the convex hull, return NaN
    if hull is None:
        return np.nan

    # Ensure that the hull is of type ConvexHull
    if not isinstance(hull, ConvexHull):
        raise TypeError("After processing, the input must be a ConvexHull object.")

    # If hull couldn't be formed, return NaN
    if hull is None:
        return np.nan

    # Return the area of the convex hull
    return hull.volume


def get_chull_max_width(hull: Union[np.ndarray, ConvexHull]) -> float:
    """Calculate the maximum width (in the x-axis direction) of the convex hull formed
    by the given points.

    Args:
        hull: Either an array of landmark points or a pre-computed convex hull.

    Returns:
        Scalar value representing the maximum width of the convex hull. Returns NaN if
            unable to compute the convex hull.
    """

    # If hull is None, return NaN
    if hull is None:
        return np.nan

    # If the input is an array, compute its convex hull
    if isinstance(hull, np.ndarray):
        hull = get_convhull(hull)
        if hull is None:
            return np.nan
        # Extract the convex hull points
        hull_pts = hull.points[hull.vertices]
    elif isinstance(hull, ConvexHull):
        hull_pts = hull.points[hull.vertices]
    else:
        raise TypeError(
            "Input must be either an array of points or a ConvexHull object."
        )

    # Calculate the maximum width (difference in x-coordinates)
    max_width = np.nanmax(hull_pts[:, 0]) - np.nanmin(hull_pts[:, 0])

    return max_width


def get_chull_max_height(hull: Union[np.ndarray, ConvexHull]) -> float:
    """Get maximum height of convex hull.

    Args:
        hull: landmark points or a precomputed convex hull.

    Return:
        Scalar of convex hull maximum height. If the hull cannot be computed (e.g.,
        insufficient valid points), NaN is returned.
    """

    # If hull is None, return NaN
    if hull is None:
        return np.nan

    # If the input is a ConvexHull object, use it directly
    if isinstance(hull, ConvexHull):
        hull = hull
    else:
        # Otherwise, compute the convex hull
        hull = get_convhull(hull)

    # If no valid convex hull could be computed, return NaN
    if hull is None:
        return np.nan

    # Use the convex hull's vertices to compute the maximum height
    max_height = np.nanmax(hull.points[hull.vertices, 1]) - np.nanmin(
        hull.points[hull.vertices, 1]
    )

    return max_height


def get_chull_line_lengths(hull: Union[np.ndarray, ConvexHull]) -> np.ndarray:
    """Get the pairwise distances between all vertices of the convex hull.

    Args:
        hull: Root landmarks as array of shape (..., 2) or a ConvexHull object.

    Returns:
        An array containing the pairwise distances between all vertices of the convex
            hull. If the convex hull fitting fails, an empty array is returned.
    """
    # If hull is None, return NaN
    if hull is None:
        return np.nan

    # Ensure pts is a ConvexHull object, otherwise get the convex hull
    hull = hull if isinstance(hull, ConvexHull) else get_convhull(hull)

    if hull is None:
        return np.array([])

    # Compute the pairwise distances between all vertices of the convex hull
    chull_line_lengths = pdist(hull.points[hull.vertices], "euclidean")

    return chull_line_lengths
