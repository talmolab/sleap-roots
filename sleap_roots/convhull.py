"""Convex hull fitting and derived trait calculation."""

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist
from typing import Tuple, Optional, Union
from sleap_roots.points import (
    extract_points_from_geometry,
    get_line_equation_from_points,
)
from shapely import box, LineString, normalize, Polygon


def get_convhull(pts: np.ndarray) -> Optional[ConvexHull]:
    """Compute the convex hull for the points per frame.

    Args:
        pts: Root landmarks as an array of shape (..., 2).

    Returns:
        An object representing the convex hull or None if a hull can't be formed.
    """
    # Ensure the input is an array of shape (..., 2)
    if pts.ndim < 2 or pts.shape[-1] != 2:
        raise ValueError("Input points should be of shape (..., 2).")

    # Reshape and filter out NaN values
    pts = pts.reshape(-1, 2)
    pts = pts[~np.isnan(pts).any(axis=-1)]

    # Check for infinite values
    if np.isinf(pts).any():
        print("Cannot compute convex hull: input contains infinite values.")
        return None

    # Ensure there are at least 3 unique non-collinear points
    unique_pts = np.unique(pts, axis=0)
    if len(unique_pts) < 3:
        print("Cannot compute convex hull: not enough unique points.")
        return None

    try:
        # Compute and return the convex hull
        return ConvexHull(unique_pts)
    except Exception as e:
        print(f"Cannot compute convex hull: {e}")
        return None


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
    """Calculate the area of the convex hull formed by the given points.

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
    """Calculate the maximum width (in the x-axis direction) of the convex hull.

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


def get_chull_division_areas(
    rn_pts: np.ndarray, pts: np.ndarray, hull: ConvexHull
) -> Tuple[float, float]:
    """Get areas above and below the line formed by the leftmost and rightmost rn nodes.

    Args:
        rn_pts: The nth root nodes when indexing from 0. Shape is (instances, 2).
        pts: Numpy array of points with shape (instances, nodes, 2).
        hull: A ConvexHull object computed from pts.

    Returns:
        A tuple containing the areas of the convex hull of the points above and below
        the line, respectively, where the line is formed by the leftmost and rightmost
        rn nodes and the y-axis increases downward in image coordinates. Returns
        (np.nan, np.nan) if the area cannot be calculated.

    Raises:
        ValueError: If pts does not have the expected shape, or if hull is not a valid
        ConvexHull object.
    """
    if not isinstance(pts, np.ndarray) or pts.ndim != 3 or pts.shape[-1] != 2:
        raise ValueError("pts must be a numpy array of shape (instances, nodes, 2).")
    if not isinstance(hull, ConvexHull):
        raise ValueError("hull must be a ConvexHull object.")

    # There must be at least 3 unique non-collinear points to form a convex hull
    # Flatten pts to 2D array and check for at least 3 unique points
    flattened_pts = pts.reshape(-1, 2)
    unique_pts = np.unique(flattened_pts, axis=0)
    if len(unique_pts) < 3:
        return np.nan, np.nan

    # Attempt to get the line equation between the leftmost and rightmost r1 nodes
    try:
        leftmost_rn = rn_pts[np.argmin(rn_pts[:, 0])]
        rightmost_rn = rn_pts[np.argmax(rn_pts[:, 0])]
        m, b = get_line_equation_from_points(leftmost_rn, rightmost_rn)
    except Exception:
        # If line equation cannot be found, return NaNs
        return np.nan, np.nan

    # Initialize lists to hold points above/on and below the line
    above_or_on_line = []
    below_line = []
    # Classify each point as being above or below the line
    for point in flattened_pts:
        if (
            point[1] <= m * point[0] + b
        ):  # y <= mx + b (y increases downward in image coordinates)
            above_or_on_line.append(point)
        else:
            below_line.append(point)

    # Calculate areas using get_chull_area, return np.nan if no points satisfy the condition
    area_above_line = (
        get_chull_area(np.array(above_or_on_line)) if above_or_on_line else np.nan
    )
    area_below_line = get_chull_area(np.array(below_line)) if below_line else np.nan

    return area_above_line, area_below_line


def get_chull_division_areas_above(areas: Tuple[float, float]) -> float:
    """Get the chull area of the points above the line from `get_chull_division_areas`.

    Args:
        areas: Tuple containing two float objects:
            - The first is the area of the convex hull of the points above the line
            formed by the leftmost and rightmost rn nodes.
            - The second is the area of the convex hull of the points below the line
            formed by the leftmost and rightmost rn nodes.

    Returns:
        area_above_line: the area of the convex hull of the points above the line,
            formed by the leftmost and rightmost rn nodes.
    """
    return areas[0]


def get_chull_division_areas_below(areas: Tuple[float, float]) -> float:
    """Get the chull area of the points below the line from `get_chull_division_areas`.

    Args:
        areas: Tuple containing two float objects:
            - The first is the area of the convex hull of the points above the line
            formed by the leftmost and rightmost rn nodes.
            - The second is the area of the convex hull of the points below the line
            formed by the leftmost and rightmost rn nodes.

    Returns:
        area_below_line: the area of the convex hull of the points below the line,
            formed by the leftmost and rightmost rn nodes.
    """
    return areas[1]


def get_chull_areas_via_intersection(
    rn_pts: np.ndarray, pts: np.ndarray, hull: Optional[ConvexHull]
) -> Tuple[float, float]:
    """Get convex hull areas above and below the intersecting line.

    Args:
        rn_pts: The nth root nodes when indexing from 0. Shape is (instances, 2).
        pts: Numpy array of points with shape (instances, nodes, 2).
        hull: A ConvexHull object computed from pts, or None if a convex hull couldn't be formed.

    Returns:
        A tuple containing the areas of the convex hull above and below
        the line, respectively, where the line is formed by the leftmost and rightmost
        rn nodes and the y-axis increases downward in image coordinates. Returns
        (np.nan, np.nan) if the area cannot be calculated.

    Raises:
        ValueError: If pts does not have the expected shape.
    """
    # Check for valid pts input
    if not isinstance(pts, np.ndarray) or pts.ndim != 3 or pts.shape[-1] != 2:
        raise ValueError("pts must be a numpy array of shape (instances, nodes, 2).")

    # Flatten pts to 2D array and remove NaN values
    flattened_pts = pts.reshape(-1, 2)
    valid_pts = flattened_pts[~np.isnan(flattened_pts).any(axis=1)]
    # Get unique points
    unique_pts = np.unique(valid_pts, axis=0)

    # Check for a valid or existing convex hull
    if hull is None or len(unique_pts) < 3:
        return np.nan, np.nan

    # Ensure rn_pts does not contain NaN values
    rn_pts_valid = rn_pts[~np.isnan(rn_pts).any(axis=1)]
    # Need at least two points to define a line
    if len(rn_pts_valid) < 2:
        return np.nan, np.nan

    # Attempt to get the line equation between the leftmost and rightmost rn nodes
    try:
        leftmost_rn = rn_pts[np.argmin(rn_pts[:, 0])]
        rightmost_rn = rn_pts[np.argmax(rn_pts[:, 0])]
        m, b = get_line_equation_from_points(leftmost_rn, rightmost_rn)
    except Exception:
        # If line equation cannot be found, return NaNs
        return np.nan, np.nan

    # Initialize lists to hold points above/on and below the line
    above_line = []
    below_line = []
    # Classify each point as being above or below the line
    for point in unique_pts:
        if (
            point[1] <= m * point[0] + b
        ):  # y <= mx + b (y increases downward in image coordinates)
            above_line.append(point)
        if point[1] >= m * point[0] + b:
            below_line.append(point)

    # Find the leftmost and rightmost points
    leftmost_pt = np.nanmin(unique_pts[:, 0])
    rightmost_pt = np.nanmax(unique_pts[:, 0])

    # Define how far to extend the line in terms of x
    x_min_extended = leftmost_pt  # Far left point
    x_max_extended = rightmost_pt  # Far right point

    # Calculate the corresponding y-values using the line equation
    y_min_extended = m * x_min_extended + b
    y_max_extended = m * x_max_extended + b

    # Create the extended line
    extended_line = LineString(
        [(x_min_extended, y_min_extended), (x_max_extended, y_max_extended)]
    )

    # Create a LineString that represents the perimeter of the convex hull
    hull_perimeter = LineString(
        hull.points[hull.vertices].tolist() + [hull.points[hull.vertices[0]].tolist()]
    )

    # Find the intersection between the hull perimeter and the extended line
    intersection = extended_line.intersection(hull_perimeter)

    # Compute the intersection points and add to lists
    if not intersection.is_empty:
        intersect_points = extract_points_from_geometry(intersection)
        above_line.extend(intersect_points)
        below_line.extend(intersect_points)

    # Calculate areas using get_chull_area
    area_above_line = get_chull_area(np.array(above_line)) if above_line else 0.0
    area_below_line = get_chull_area(np.array(below_line)) if below_line else 0.0

    return area_above_line, area_below_line


def get_chull_area_via_intersection_above(areas: Tuple[float, float]) -> float:
    """Get the chull area above the line from `get_chull_area_via_intersection`.

    Args:
        areas: Tuple containing two float objects:
            - The first is the area of the convex hull above the line
            formed by the leftmost and rightmost rn nodes.
            - The second is the area of the convex hull below the line
            formed by the leftmost and rightmost rn nodes.

    Returns:
        area_above_line: the area of the convex hull above the line,
            formed by the leftmost and rightmost rn nodes.
    """
    return areas[0]


def get_chull_area_via_intersection_below(areas: Tuple[float, float]) -> float:
    """Get the chull area below the line from `get_chull_area_via_intersection`.

    Args:
        areas: Tuple containing two float objects:
            - The first is the area of the convex hull above the line
            formed by the leftmost and rightmost rn nodes.
            - The second is the area of the convex hull below the line
            formed by the leftmost and rightmost rn nodes.

    Returns:
        area_below_line: the area of the convex hull below the line,
            formed by the leftmost and rightmost rn nodes.
    """
    return areas[1]


def get_chull_intersection_vectors(
    r0_pts: np.ndarray, rn_pts: np.ndarray, pts: np.ndarray, hull: Optional[ConvexHull]
) -> Tuple[np.ndarray, np.ndarray]:
    """Get vectors from top left and top right to intersection on convex hull.

    Args:
        r0_pts: The 0th root nodes when indexing from 0. Shape is (instances, 2).
        rn_pts: The nth root nodes when indexing from 0. Shape is (instances, 2).
        pts: Numpy array of points with shape (instances, nodes, 2).
        hull: A ConvexHull object computed from pts, or None if a convex hull couldn't be formed.

    Returns:
        A tuple containing vectors from the top left point to the left intersection point, and from
        the top right point to the right intersection point with the convex hull. Returns two vectors
        of NaNs if the vectors can't be calculated. Vectors are of shape (1, 2).

    Raises:
        ValueError: If pts does not have the expected shape.
    """
    if r0_pts.ndim == 1 or rn_pts.ndim == 1 or pts.ndim == 2:
        print(
            "Not enough instances or incorrect format to compute convex hull intersections."
        )
        return (np.array([[np.nan, np.nan]]), np.array([[np.nan, np.nan]]))

    # Check for valid pts input
    if not isinstance(pts, np.ndarray) or pts.ndim != 3 or pts.shape[-1] != 2:
        raise ValueError("pts must be a numpy array of shape (instances, nodes, 2).")
    # Ensure rn_pts is a numpy array of shape (instances, 2)
    if not isinstance(rn_pts, np.ndarray) or rn_pts.ndim != 2 or rn_pts.shape[-1] != 2:
        raise ValueError("rn_pts must be a numpy array of shape (instances, 2).")
    # Ensure r0_pts is a numpy array of shape (instances, 2)
    if not isinstance(r0_pts, np.ndarray) or r0_pts.ndim != 2 or r0_pts.shape[-1] != 2:
        raise ValueError(f"r0_pts must be a numpy array of shape (instances, 2).")

    # Flatten pts to 2D array and remove NaN values
    flattened_pts = pts.reshape(-1, 2)
    valid_pts = flattened_pts[~np.isnan(flattened_pts).any(axis=1)]
    # Get unique points
    unique_pts = np.unique(valid_pts, axis=0)

    # Check for a valid or existing convex hull
    if hull is None or len(unique_pts) < 3:
        # Return two vectors of NaNs if not valid hull
        return (np.array([[np.nan, np.nan]]), np.array([[np.nan, np.nan]]))

    # Ensure rn_pts does not contain NaN values
    rn_pts_valid = rn_pts[~np.isnan(rn_pts).any(axis=1)]
    # Need at least two points to define a line
    if len(rn_pts_valid) < 2:
        return (np.array([[np.nan, np.nan]]), np.array([[np.nan, np.nan]]))

    # Ensuring r0_pts does not contain NaN values
    r0_pts_valid = r0_pts[~np.isnan(r0_pts).any(axis=1)]
    # Expect two vectors in the end
    if len(r0_pts_valid) < 2:
        return (np.array([[np.nan, np.nan]]), np.array([[np.nan, np.nan]]))

    # Get the vertices of the convex hull
    hull_vertices = hull.points[hull.vertices]

    # Find the leftmost and rightmost r0 point
    leftmost_r0 = r0_pts_valid[np.argmin(r0_pts_valid[:, 0])]
    rightmost_r0 = r0_pts_valid[np.argmax(r0_pts_valid[:, 0])]

    # Check if these points are on the convex hull
    is_leftmost_on_hull = any(
        np.array_equal(leftmost_r0, vertex) for vertex in hull_vertices
    )
    is_rightmost_on_hull = any(
        np.array_equal(rightmost_r0, vertex) for vertex in hull_vertices
    )

    # Initialize vectors
    leftmost_vector = np.array([[np.nan, np.nan]])
    rightmost_vector = np.array([[np.nan, np.nan]])
    if not is_leftmost_on_hull and not is_rightmost_on_hull:
        # If leftmost and rightmost r0 points are not on the convex hull return NaNs
        return leftmost_vector, rightmost_vector

    # Attempt to get the line equation between the leftmost and rightmost rn nodes
    try:
        leftmost_rn = rn_pts[np.argmin(rn_pts[:, 0])]
        rightmost_rn = rn_pts[np.argmax(rn_pts[:, 0])]
        m, b = get_line_equation_from_points(leftmost_rn, rightmost_rn)
    except Exception:
        # If line equation cannot be found, return NaNs
        return leftmost_vector, rightmost_vector

    # Find the leftmost and rightmost points
    leftmost_pt = np.nanmin(unique_pts[:, 0])
    rightmost_pt = np.nanmax(unique_pts[:, 0])

    # Define how far to extend the line in terms of x
    x_min_extended = leftmost_pt  # Far left point
    x_max_extended = rightmost_pt  # Far right point

    # Calculate the corresponding y-values using the line equation
    y_min_extended = m * x_min_extended + b
    y_max_extended = m * x_max_extended + b

    # Create the extended line
    extended_line = LineString(
        [(x_min_extended, y_min_extended), (x_max_extended, y_max_extended)]
    )

    # Create a LineString that represents the perimeter of the convex hull
    hull_perimeter = LineString(
        hull.points[hull.vertices].tolist() + [hull.points[hull.vertices[0]].tolist()]
    )

    # Find the intersection between the hull perimeter and the extended line
    intersection = extended_line.intersection(hull_perimeter)

    # Get the intersection points
    if not intersection.is_empty:
        intersect_points = extract_points_from_geometry(intersection)
    else:
        # Return two vectors of NaNs if there is no intersection
        return leftmost_vector, rightmost_vector

    # Convert the list of NumPy arrays to a 2D NumPy array
    intersection_points_array = np.vstack(intersect_points)

    # Find the leftmost and rightmost intersection points
    leftmost_intersect = intersection_points_array[
        np.argmin(intersection_points_array[:, 0])
    ]
    rightmost_intersect = intersection_points_array[
        np.argmax(intersection_points_array[:, 0])
    ]

    # Make a vector from the leftmost r0 point to the leftmost intersection point
    leftmost_vector = (leftmost_intersect - leftmost_r0).reshape(1, -1)

    # Make a vector from the rightmost r0 point to the rightmost intersection point
    rightmost_vector = (rightmost_intersect - rightmost_r0).reshape(1, -1)

    return leftmost_vector, rightmost_vector


def get_chull_intersection_vectors_left(
    vectors: Tuple[np.ndarray, np.ndarray]
) -> np.ndarray:
    """Get the vector from the top left point to the left intersection point.

    Args:
        vectors: Tuple containing two numpy arrays:
            - The first is the vector from the top left point to the left intersection point.
            - The second is the vector from the top right point to the right intersection point.

    Returns:
        leftmost_vector: the vector from the top left point to the left intersection point.
    """
    return vectors[0]


def get_chull_intersection_vectors_right(
    vectors: Tuple[np.ndarray, np.ndarray]
) -> np.ndarray:
    """Get the vector from the top right point to the right intersection point.

    Args:
        vectors: Tuple containing two numpy arrays:
            - The first is the vector from the top left point to the left intersection point.
            - The second is the vector from the top right point to the right intersection point.

    Returns:
        rightmost_vector: the vector from the top right point to the right intersection point.
    """
    return vectors[1]
