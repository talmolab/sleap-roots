"""Get traits related to the points."""

import numpy as np
from shapely.geometry import LineString
from sleap_roots.lengths import min_distance_line_to_line


def get_all_pts_array(
    primary_max_length_pts: np.ndarray, lateral_pts: np.ndarray, monocots: bool = False
) -> np.ndarray:
    """Get all landmark points within a given frame as a flat array of coordinates.

    Args:
        primary_max_length_pts: Points of the primary root with maximum length of shape
            `(nodes, 2)`.
        lateral_pts: Lateral root points of shape `(instances, nodes, 2)`.
        monocots: If False (default), returns a combined array of primary and lateral
            root points. If True, returns only lateral root points.

    Returns:
        A 2D array of shape (n_points, 2), containing the coordinates of all extracted
        points.
    """
    # Check if the input arrays have the right number of dimensions
    if primary_max_length_pts.ndim != 2 or lateral_pts.ndim != 3:
        raise ValueError(
            "Input arrays should have the correct number of dimensions:"
            "primary_max_length_pts should be 2-dimensional and lateral_pts should be"
            "3-dimensional."
        )

    # Check if the last dimension of the input arrays has size 2
    # (representing x and y coordinates)
    if primary_max_length_pts.shape[-1] != 2 or lateral_pts.shape[-1] != 2:
        raise ValueError(
            "The last dimension of the input arrays should have size 2, representing x"
            "and y coordinates."
        )

    # Flatten the arrays to 2D
    primary_max_length_pts = primary_max_length_pts.reshape(-1, 2)
    lateral_pts = lateral_pts.reshape(-1, 2)

    # Combine points
    if monocots:
        pts_all_array = lateral_pts
    else:
        # Check if the data types of the arrays are compatible
        if primary_max_length_pts.dtype != lateral_pts.dtype:
            raise ValueError("Input arrays should have the same data type.")

        pts_all_array = np.concatenate((primary_max_length_pts, lateral_pts), axis=0)

    return pts_all_array


def associate_lateral_to_primary(
    primary_pts: np.ndarray, lateral_pts: np.ndarray
) -> dict:
    """Associates each lateral root point with the closest primary root point.

    This function iterates through each lateral root represented as a LineString and
    calculates the minimum distance to each primary root, also represented as LineStrings.
    Each lateral root is then associated with the closest primary root, and this association
    is stored in a dictionary.

    Args:
        primary_pts: A numpy array of primary root points with shape
            (instances, nodes, 2), where 'instances' is the number of primary roots,
            'nodes' is the number of points in each root, and '2' corresponds to the x and y
            coordinates.

        lateral_pts: A numpy array of lateral root points with a shape similar
            to primary_pts, representing the lateral roots.

    Returns:
        A dictionary where each key is an index of a primary root (from the primary_pts
        array) and each value is a list of lateral root points (from the lateral_pts array)
        that are closest to that primary root.
    """
    # Check if inputs are numpy arrays
    if not (
        isinstance(primary_pts, np.ndarray) and isinstance(lateral_pts, np.ndarray)
    ):
        raise TypeError("Both primary_pts and lateral_pts should be NumPy arrays.")

    # If there are no primary points, return an empty dictionary immediately
    if primary_pts.size == 0:
        return {}
    # Return with empty associations if no lateral roots
    if lateral_pts.size == 0:
        return {i: [] for i in range(len(primary_pts))}

    if len(primary_pts.shape) != 3 or len(lateral_pts.shape) != 3:
        raise ValueError("Input arrays must be 3-dimensional.")

    if primary_pts.shape[-1] != 2 or lateral_pts.shape[-1] != 2:
        raise ValueError(
            "The last dimension of the input arrays must be 2 (for x and y coordinates)."
        )

    # Convert primary and lateral roots to LineString
    primary_line_strings = [LineString(root) for root in primary_pts]
    lateral_line_strings = [LineString(root) for root in lateral_pts]

    # Initialize a dictionary with an empty list for each primary root index
    plant_associations = {i: [] for i in range(len(primary_line_strings))}

    # Associate each lateral root with the closest primary root
    for i, lateral_line in enumerate(lateral_line_strings):
        min_dists = [
            min_distance_line_to_line(lateral_line, primary_line)
            for primary_line in primary_line_strings
        ]
        # Index of the primary root closest to this lateral root
        index = np.nanargmin(min_dists)

        # Append the lateral root points to the list of the associated primary root
        plant_associations[index].append(lateral_pts[i])

    # `plant_associations` contains lateral roots grouped by their closest primary roots
    # and ensures that all primary roots are included in the dictionary, even if they
    # do not have lateral roots associated with them
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
