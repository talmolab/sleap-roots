"""Get traits related to the points."""

import numpy as np
from shapely.geometry import LineString
from sleap_roots.lengths import get_min_distance_line_to_line
from typing import List, Optional, Tuple


def get_count(pts: np.ndarray):
    """Get number of roots.

    Args:
        pts: Root landmarks as array of shape `(instances, nodes, 2)`.

    Return:
        Scalar of number of  roots.
    """
    # The number of roots is the number of instances
    count = pts.shape[0]
    return count


def join_pts(pts0: np.ndarray, *args: Optional[np.ndarray]) -> List[np.ndarray]:
    """Join an arbitrary number of points arrays and return them as a list.

    Args:
        pts0: The first array of points. Should have shape `(instances, nodes, 2)`
            or `(nodes, 2)`.
        *args: Additional optional arrays of points. Each should have shape
            `(instances, nodes, 2)` or `(nodes, 2)`.

    Returns:
        A list of arrays, each having shape `(nodes, 2)`.
    """
    # Initialize an empty list to store the points
    all_pts = []
    # Loop over the input arrays
    for pts in [pts0] + list(args):
        if pts is None:
            continue  # Skip None values

        # If an array has shape `(nodes, 2)`, expand dimensions to `(1, nodes, 2)`
        if pts.ndim == 2 and pts.shape[-1] == 2:
            pts = pts[np.newaxis, :, :]

        # Validate the shape of each array
        if pts.ndim != 3 or pts.shape[-1] != 2:
            raise ValueError(
                "Points should have a shape of `(instances, nodes, 2)` or `(nodes, 2)`."
            )

        # Add the points to the list
        all_pts.extend(list(pts))

    return all_pts


def get_all_pts_array(pts0: np.ndarray, *args: Optional[np.ndarray]) -> np.ndarray:
    """Get all landmark points within a given frame as a flat array of coordinates.

    Args:
        pts0: The first array of points. Should have shape `(instances, nodes, 2)`
            or `(nodes, 2)`.
        *args: Additional optional arrays of points. Each should have shape
            `(instances, nodes, 2)` or `(nodes, 2)`.

    Returns:
        A 2D array of shape (n_points, 2), containing the coordinates of all extracted
        points.
    """
    # Initialize an empty list to store the points
    concatenated_pts = []

    # Loop over the input arrays
    for pts in [pts0] + list(args):
        if pts is None:
            continue

        # Check if the array has the right number of dimensions
        if pts.ndim not in [2, 3]:
            raise ValueError("Each input array should be 2D or 3D.")

        # Check if the last dimension of the array has size 2
        # (representing x and y coordinates)
        if pts.shape[-1] != 2:
            raise ValueError(
                "The last dimension should have size 2, representing x and y coordinates."
            )

        # Flatten the array to 2D and append to list
        flat_pts = pts.reshape(-1, 2)
        concatenated_pts.append(flat_pts)

    # Concatenate all points into a single array
    return np.concatenate(concatenated_pts, axis=0)


def get_nodes(pts: np.ndarray, node_index: int) -> np.ndarray:
    """Extracts the (x, y) coordinates of a specified node.

    Args:
        pts: An array of points. For multiple instances, the shape should be
            (instances, nodes, 2). For a single instance,the shape should be (nodes, 2).
        node_index: The index of the node for which to extract the coordinates, based on
            the node's position in the sequence of connected nodes (0-based indexing).

    Returns:
        np.ndarray: An array of (x, y) coordinates for the specified node. For multiple
            instances, the shape will be (instances, 2). For a single instance, the
            shape will be (2,).

    Raises:
        ValueError: If node_index is out of bounds for the number of nodes.
    """
    # Adjust for a single instance with shape (nodes, 2)
    if pts.ndim == 2:
        if not 0 <= node_index < pts.shape[0]:
            raise ValueError("node_index is out of bounds for the number of nodes.")
        # Return a (2,) shape array for the node coordinates in a single instance
        return pts[node_index, :]

    # Handle multiple instances with shape (instances, nodes, 2)
    elif pts.ndim == 3:
        if not 0 <= node_index < pts.shape[1]:
            raise ValueError("node_index is out of bounds for the number of nodes.")
        # Return (instances, 2) shape array for the node coordinates across instances
        return pts[:, node_index, :]

    else:
        raise ValueError(
            "Input array should have shape (nodes, 2) for a single instance "
            "or (instances, nodes, 2) for multiple instances."
        )


def get_root_vectors(start_nodes: np.ndarray, end_nodes: np.ndarray) -> np.ndarray:
    """Calculate the vector from start to end for each instance in a set of points.

    Args:
        start_nodes: array of points with shape (instances, 2) or (2,) representing the
            start node in each instance.
        end_nodes: array of points with shape (instances, 2) or (2,) representing the
            end node in each instance.

    Returns:
        An array of vectors with shape (instances, 2), representing the vector from start
        to end for each instance.
    """
    # Ensure that the start and end nodes have the same shapes
    if start_nodes.shape != end_nodes.shape:
        raise ValueError("start_nodes and end_nodes should have the same shape.")
    # Handle single instances with shape (2,)
    if start_nodes.ndim == 1:
        start_nodes = start_nodes[np.newaxis, :]
    if end_nodes.ndim == 1:
        end_nodes = end_nodes[np.newaxis, :]
    # Calculate the vectors from start to end for each instance
    vectors = start_nodes - end_nodes
    return vectors


def get_left_right_normalized_vectors(
    r0_pts: np.ndarray, r1_pts: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Get the unit vectors formed from r0 to r1 on the left and right sides of a crown root system.

    Args:
        r0_pts: An array of points representing the r0 nodes, with shape (instances, 2),
                where instances are different observations of r0 points, and 2 represents
                the x and y coordinates.
        r1_pts: An array of points representing the r1 nodes, similar in structure to r0_pts.

    Returns:
        A tuple containing two np.ndarray objects:
        - The first is a normalized vector from r0 to r1 on the left side, or a vector
            of NaNs if normalization fails.
        - The second is a normalized vector from r0 to r1 on the right side, or a vector
            of NaNs if normalization fails.
    """
    # Validate input shapes and ensure there are multiple instances for comparison
    if (
        r0_pts.ndim == 2
        and r1_pts.ndim == 2
        and r0_pts.shape == r1_pts.shape
        and r0_pts.shape[0] > 1
    ):
        # Find indices of the leftmost and rightmost r0 and r1 points
        leftmost_r0_index = np.nanargmin(r0_pts[:, 0])
        rightmost_r0_index = np.nanargmax(r0_pts[:, 0])
        leftmost_r1_index = np.nanargmin(r1_pts[:, 0])
        rightmost_r1_index = np.nanargmax(r1_pts[:, 0])

        # Extract the corresponding r0 and r1 points for leftmost and rightmost nodes
        r0_left = r0_pts[leftmost_r0_index]
        r1_left = r1_pts[leftmost_r1_index]
        r0_right = r0_pts[rightmost_r0_index]
        r1_right = r1_pts[rightmost_r1_index]

        # Calculate the vectors from r0 to r1 for both the leftmost and rightmost points
        vector_left = r1_left - r0_left
        vector_right = r1_right - r0_right

        # Calculate norms of both vectors for normalization
        norm_left = np.linalg.norm(vector_left)
        norm_right = np.linalg.norm(vector_right)

        # Normalize the vectors if their norms are non-zero
        # otherwise, return vectors filled with NaNs
        norm_vector_left = (
            vector_left / norm_left if norm_left > 0 else np.array([np.nan, np.nan])
        )
        norm_vector_right = (
            vector_right / norm_right if norm_right > 0 else np.array([np.nan, np.nan])
        )

        return norm_vector_left, norm_vector_right
    else:
        # Return pairs of NaN vectors if inputs are invalid or do not meet the requirements
        return np.array([np.nan, np.nan]), np.array([np.nan, np.nan])


def get_left_normalized_vector(
    normalized_vectors: Tuple[np.ndarray, np.ndarray]
) -> np.ndarray:
    """Get the normalized vector from r0 to r1 on the left side of a crown root system.

    Args:
        normalized_vectors: A tuple containing two np.ndarray objects:
            - The first is a normalized vector from r0 to r1 on the left side, or a vector
                of NaNs if normalization fails.
            - The second is a normalized vector from r0 to r1 on the right side, or a vector
                of NaNs if normalization fails.

    Returns:
        np.ndarray: A normalized vector from r0 to r1 on the left side, or a vector of NaNs
            if normalization fails.
    """
    return normalized_vectors[0]


def get_right_normalized_vector(
    normalized_vectors: Tuple[np.ndarray, np.ndarray]
) -> np.ndarray:
    """Get the normalized vector from r0 to r1 on the right side of a crown root system.

    Args:
        normalized_vectors: A tuple containing two np.ndarray objects:
            - The first is a normalized vector from r0 to r1 on the left side, or a vector
                of NaNs if normalization fails.
            - The second is a normalized vector from r0 to r1 on the right side, or a vector
                of NaNs if normalization fails.

    Returns:
        np.ndarray: A normalized vector from r0 to r1 on the right side, or a vector of NaNs
            if normalization fails.
    """
    return normalized_vectors[1]


def get_line_equation_from_points(pts1: np.ndarray, pts2: np.ndarray):
    """Calculate the slope (m) and y-intercept (b) of the line connecting two points.

    Args:
        pts1: First point as (x, y). 1D array of shape (2,).
        pts2: Second point as (x, y). 1D array of shape (2,).

    Returns:
        A tuple (m, b) representing the slope and y-intercept of the line. If the line is
        vertical, NaNs are returned.
    """
    # Convert inputs to arrays if they're not already
    pts1 = np.asarray(pts1)
    pts2 = np.asarray(pts2)

    # Validate input shapes
    if pts1.ndim != 1 or pts1.shape[0] != 2 or pts2.ndim != 1 or pts2.shape[0] != 2:
        raise ValueError("Each input point must be a 1D array of shape (2,).")

    # If the line is vertical return NaNs
    if pts1[0] == pts2[0]:
        return np.nan, np.nan
    else:
        # Calculate the slope
        m = (pts2[1] - pts1[1]) / (pts2[0] - pts1[0])

    # Calculate the y-intercept
    b = pts1[1] - m * pts1[0]

    return m, b


def filter_roots_with_nans(pts: np.ndarray) -> np.ndarray:
    """Remove roots with NaN values from an array of root points.

    Args:
        pts: An array of points representing roots, with shape (instances, nodes, 2),
            where 'instances' is the number of roots, 'nodes' is the number of points in
            each root, and '2' corresponds to the x and y coordinates.

    Returns:
        np.ndarray: An array of shape (instances, nodes, 2) with NaN-containing roots
            removed. If all roots contain NaN values, an empty array of shape
            (0, nodes, 2) is returned.
    """
    if not isinstance(pts, np.ndarray):
        raise TypeError("Input must be a numpy array.")
    if pts.ndim != 3 or pts.shape[2] != 2:
        raise ValueError("Input array must have a shape of (instances, nodes, 2).")

    cleaned_pts = np.array([root for root in pts if not np.isnan(root).any()])

    if cleaned_pts.size == 0:
        return np.empty((0, pts.shape[1], 2))

    return cleaned_pts


def filter_plants_with_unexpected_ct(
    primary_pts: np.ndarray, lateral_pts: np.ndarray, expected_count: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Filter out primary and lateral roots with an unexpected number of plants.

    Args:
        primary_pts: A numpy array of primary root points with shape
            (instances, nodes, 2), where 'instances' is the number of primary roots,
            'nodes' is the number of points in each root, and '2' corresponds to the x and y
            coordinates.
        lateral_pts: A numpy array of lateral root points with a shape similar
            to primary_pts, representing the lateral roots.
        expected_count: The expected number of primary roots as a float or NaN. If NaN,
            no filtering is applied based on count. If a number, it will be rounded to
            the nearest integer for comparison.

    Returns:
        A tuple containing the filtered primary and lateral root points arrays. If the
        input types are incorrect, the function will raise a ValueError.

    Raises:
        ValueError: If input types are incorrect.
    """
    # Type checking
    if not isinstance(primary_pts, np.ndarray) or not isinstance(
        lateral_pts, np.ndarray
    ):
        raise ValueError("primary_pts and lateral_pts must be numpy arrays.")
    if not np.issubdtype(type(expected_count), np.number):
        raise ValueError("expected_count must be a numeric type.")


    # Handle NaN expected_count: Skip filtering if expected_count is NaN
    if not np.isnan(expected_count):
        # Rounding expected_count to the nearest integer for comparison
        expected_count_rounded = round(expected_count)

        if len(primary_pts) != expected_count_rounded:
            # Adjusting primary and lateral roots to empty arrays of the same shape
            primary_pts = np.empty((0, primary_pts.shape[1], 2))
            lateral_pts = np.empty((0, lateral_pts.shape[1], 2))

    return primary_pts, lateral_pts


def get_filtered_primary_pts(filtered_pts: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Get the filtered primary root points from a tuple of filtered primary and lateral roots.

    Args:
        filtered_pts: A tuple containing the filtered primary and lateral root points arrays.

    Returns:
        np.ndarray: The filtered primary root points array.
    """
    return filtered_pts[0]


def get_filtered_lateral_pts(filtered_pts: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Get the filtered lateral root points from a tuple of filtered primary and lateral roots.

    Args:
        filtered_pts: A tuple containing the filtered primary and lateral root points arrays.

    Returns:
        np.ndarray: The filtered lateral root points array.
    """
    return filtered_pts[1]


def is_line_valid(line: np.ndarray) -> bool:
    """Check if a line (numpy array of points) does not contain NaN values, indicating it is valid.

    Args:
        line: A numpy array representing a line with shape (nodes, 2), where 'nodes' is
            the number of points in the line.

    Returns:
        True if the line does not contain any NaN values, False otherwise.
    """
    return not np.isnan(line).any()


def clean_points(points):
    """Remove NaN points from root points.

    Args:
        points: An array of points representing a root, with shape (nodes, 2).

    Returns:
        np.ndarray: An array of the same points with NaN values removed.
    """
    # Filter out points with NaN values and return the cleaned array
    return np.array([pt for pt in points if not np.isnan(pt).any()])


def associate_lateral_to_primary(
    primary_pts: np.ndarray, lateral_pts: np.ndarray
) -> dict:
    """Associates each lateral root with the closest primary root.

    Args:
        primary_pts: A numpy array of primary root points with shape
            (instances, nodes, 2), where 'instances' is the number of primary roots,
            'nodes' is the number of points in each root, and '2' corresponds to the x and y
            coordinates. Points cannot have NaN values.
        lateral_pts: A numpy array of lateral root points with a shape similar
            to primary_pts, representing the lateral roots. Points cannot have NaN values.

    Returns:
        dict: A dictionary where each key is an index of a primary root (from the primary_pts
        array) and each value is a dictionary containing 'primary_points' as the points of
        the primary root (1, nodes, 2) and 'lateral_points' as an array of
        lateral root points that are closest to that primary root. The shape of
        'lateral_points' is (instances, nodes, 2), where instances is the number of
        lateral roots associated with the primary root.
    """
    # Basic input validation
    if not isinstance(primary_pts, np.ndarray) or not isinstance(
        lateral_pts, np.ndarray
    ):
        raise ValueError("Both primary_pts and lateral_pts must be numpy arrays.")
    if len(primary_pts.shape) != 3 or len(lateral_pts.shape) != 3:
        raise ValueError("Input arrays must have a shape of (instances, nodes, 2).")
    if primary_pts.shape[2] != 2 or lateral_pts.shape[2] != 2:
        raise ValueError(
            "The last dimension of input arrays must be 2, representing x and y coordinates."
        )

    plant_associations = {}

    # Initialize plant associations dictionary
    for i, primary_root in enumerate(primary_pts):
        if not is_line_valid(primary_root):
            continue  # Skip primary roots containing NaN values
        plant_associations[i] = {
            "primary_points": primary_root,
            "lateral_points": [],
        }

    # Associate each lateral root with the closest primary root
    for lateral_root in lateral_pts:
        if not is_line_valid(lateral_root):
            continue  # Skip lateral roots containing NaN values

        lateral_line = LineString(lateral_root)
        min_distance = float("inf")
        closest_primary_index = None

        for primary_index, primary_data in plant_associations.items():
            primary_root = primary_data["primary_points"]
            try:
                primary_line = LineString(primary_root)
                distance = primary_line.distance(lateral_line)
            except Exception as e:
                print(f"Error computing distance: {e}")
                continue

            if distance < min_distance:
                min_distance = distance
                closest_primary_index = primary_index

        if closest_primary_index is not None:
            plant_associations[closest_primary_index]["lateral_points"].append(
                lateral_root
            )

    # Convert lateral points lists into arrays
    for primary_index, data in plant_associations.items():
        lateral_points_list = data["lateral_points"]
        if lateral_points_list:  # Check if there are any lateral points to convert
            lateral_points_array = np.array(lateral_points_list)
            plant_associations[primary_index]["lateral_points"] = lateral_points_array
        else:
            plant_associations[primary_index]["lateral_points"] = np.empty(
                (0, lateral_pts.shape[1], 2)
            )

    return plant_associations


def flatten_associated_points(associations: dict, primary_pts: np.ndarray) -> dict:
    """
    Creates a dictionary of flattened arrays containing primary and lateral root points.

    Args:
        associations: A dictionary with primary root indices as keys and lists of lateral root
                      point arrays as values.
        primary_pts: A numpy array of primary root points with shape (instances, nodes, 2),
                     where 'instances' is the number of primary roots, 'nodes' is the number
                     of points in each root, and '2' corresponds to the x and y coordinates.

    Returns:
        A dictionary with the same keys as associations. Each key corresponds to a flattened
            array containing all the primary and lateral root points for that plant.
    """
    flattened_points = {}

    for key, laterals in associations.items():
        # Get the primary root points for the current key
        primary_root_points = primary_pts[key]

        # Initialize a list with the primary root points
        all_points = [primary_root_points]

        # Extend the list with lateral root points for the current primary root
        for lateral in laterals:
            all_points.append(lateral)

        # Concatenate all the points into a single array and flatten it
        all_points_array = np.vstack(all_points).flatten()

        # Add to the dictionary
        flattened_points[key] = all_points_array

    return flattened_points
