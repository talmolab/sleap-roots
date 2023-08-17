"""Get intersections between roots and horizontal scan lines."""

import numpy as np
import math


def count_scanline_intersections(
    primary_pts: np.ndarray,
    lateral_pts: np.ndarray,
    height: int = 1080,
    width: int = 2048,
    n_line: int = 50,
    monocots: bool = False,
) -> np.ndarray:
    """Count intersections of roots with a series of horizontal scanlines.

    This function calculates the number of intersections between the provided
    primary and lateral root points and a set of horizontal scanlines. The scanlines
    are equally spaced across the specified height.

    Args:
        primary_pts: Array of primary root landmarks of shape `(nodes, 2)`.
            Will be reshaped internally to `(1, nodes, 2)`.
        lateral_pts: Array of lateral root landmarks with shape
            `(instances, nodes, 2)`.
        height: The height of the image or cylinder. Defaults to 1080.
        width: The width of the image or cylinder. Defaults to 2048.
        n_line: Number of scanlines to use. Defaults to 50.
        monocots: If `True`, only uses lateral roots (e.g., for rice).
            If `False`, uses both primary and lateral roots (e.g., for dicots).
            Defaults to `False`.

    Returns:
        An array with shape `(n_line,)` representing the number of intersections
            of roots with each scanline.
    """

    # Input validation
    if primary_pts.ndim != 2 or primary_pts.shape[-1] != 2:
        raise ValueError("primary_pts should have a shape of `(nodes, 2)`.")

    if lateral_pts.ndim != 3 or lateral_pts.shape[-1] != 2:
        raise ValueError("lateral_pts should have a shape of `(instances, nodes, 2)`.")

    # Reshape primary_pts to have three dimensions
    primary_pts = primary_pts[np.newaxis, :, :]

    # Collate root points.
    all_roots = list(primary_pts) + list(lateral_pts) if not monocots else lateral_pts

    # Calculate the interval between two scanlines
    interval = height / (n_line - 1)

    intersections = []

    # Iterate over scanlines
    for i in range(n_line):
        y_coord = interval * i
        line_intersections = 0

        for root_points in all_roots:
            # Remove NaN values
            valid_points = root_points[(~np.isnan(root_points)).any(axis=1)]

            if len(valid_points) > 1:
                for j in range(len(valid_points) - 1):
                    y1 = valid_points[j][1]
                    y2 = valid_points[j + 1][1]

                    if (y1 >= y_coord >= y2) or (y2 >= y_coord >= y1):
                        line_intersections += 1

        intersections.append(line_intersections)

    return np.array(intersections)


def get_scanline_first_ind(scanline_intersection_counts: np.ndarray):
    """Get the index of count_scanline_interaction for the first interaction.

    Args:
        scanline_intersection_counts: An array with shape of `(#Nline,)` of intersection
            numbers of each scan line.

    Return:
        Scalar of count_scanline_interaction index for the first interaction.
    """
    # get the first scanline index using scanline_intersection_counts
    if np.where((scanline_intersection_counts > 0))[0].shape[0] > 0:
        scanline_first_ind = np.where((scanline_intersection_counts > 0))[0][0]
        return scanline_first_ind
    else:
        return np.nan


def get_scanline_last_ind(scanline_intersection_counts: np.ndarray):
    """Get the index of count_scanline_interaction for the last interaction.

    Args:
        scanline_intersection_counts: An array with shape of `(#Nline,)` of intersection
            numbers of each scan line.

    Return:
        Scalar of count_scanline_interaction index for the last interaction.
    """
    # get the first scanline index using scanline_intersection_counts
    if np.where((scanline_intersection_counts > 0))[0].shape[0] > 0:
        scanline_last_ind = np.where((scanline_intersection_counts > 0))[0][-1]
        return scanline_last_ind
    else:
        return np.nan
