"""Get intersections between roots and horizontal scan lines."""

import numpy as np
from typing import List


def count_scanline_intersections(
    pts_list: List[np.ndarray],
    height: int = 1080,
    n_line: int = 50,
) -> np.ndarray:
    """Count intersections of roots with a series of horizontal scanlines.

    This function calculates the number of intersections between the provided
    primary and lateral root points and a set of horizontal scanlines. The scanlines
    are equally spaced across the specified height.

    Args:
        pts_list: A list of arrays, each having shape `(nodes, 2)`.
        height: The height of the image or cylinder. Defaults to 1080.
        n_line: Number of scanlines to use. Defaults to 50.

    Returns:
        An array with shape `(n_line,)` representing the number of intersections
            of roots with each scanline.
    """
    # Input validation for pts_list
    if any(pts.ndim != 2 or pts.shape[-1] != 2 for pts in pts_list):
        raise ValueError(
            "Each pts array in pts_list should have a shape of `(nodes, 2)`."
        )

    # Calculate the interval between two scanlines
    interval = height / (n_line - 1)

    intersections = []

    # Iterate over scanlines
    for i in range(n_line):
        y_coord = interval * i
        line_intersections = 0

        for root_points in pts_list:
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
    # get the last scanline index using scanline_intersection_counts
    if np.where((scanline_intersection_counts > 0))[0].shape[0] > 0:
        scanline_last_ind = np.where((scanline_intersection_counts > 0))[0][-1]
        return scanline_last_ind
    else:
        return np.nan
