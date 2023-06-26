"""Get intersections between roots and horizontal scan lines."""

import numpy as np
import math
from shapely import LineString, Point


def count_scanline_intersections(
    primary_pts: np.ndarray,
    lateral_pts: np.ndarray,
    depth: int = 1080,
    width: int = 2048,
    n_line: int = 50,
    lateral_only: bool = False,
) -> np.ndarray:
    """Get intersection points of roots and scan lines.

    Args:
        primary_pts: Numpy array of primary points of shape (instances, nodes, 2).
        lateral_pts: Numpy array of lateral points of shape (instances, nodes, 2).
        depth: the depth of cylinder, or number of rows of the image.
        width: the width of cylinder, or number of columns of the image.
        n_line: number of scan lines.
        lateral_only: whether True: only lateral roots (e.g., rice), or False: dicots

    Returns:
        An array with shape of (#Nline,) of intersection numbers of each scan line.
    """
    # connect the points to lines using shapely
    if lateral_only:
        points = list(primary_pts)
    else:
        points = list(primary_pts) + list(lateral_pts)

    # calculate interval between two scan lines
    n_interval = math.ceil(depth / (n_line - 1))

    intersection = []

    for i in range(n_line): 
        horizontal_line_y = n_interval * (i + 1)
        intersection_line = 0

        for j in range(len(points) - 1):
            intersection_counts_root = 0
            # filter out nan nodes
            pts_j = np.array(points[j])[~np.isnan(points[j]).any(axis=1)]
            current_root = 0
            if pts_j.shape[0] > 1:
                for k in range(len(pts_j) - 1):
                    x1, y1 = pts_j[k]
                    x2, y2 = pts_j[k + 1]
                    if (y1 >= horizontal_line_y and y2 < horizontal_line_y) or (
                        y1 < horizontal_line_y and y2 >= horizontal_line_y
                    ):
                        current_root += 1
                intersection_counts_root += current_root
            intersection_line += intersection_counts_root
        intersection.append(intersection_line)
    return np.array(intersection)


def get_scanline_first_ind(
    primary_pts: np.ndarray,
    lateral_pts: np.ndarray,
    depth: int = 1080,
    width: int = 2048,
    n_line: int = 50,
    lateral_only: bool = False,
):
    """Get the index of count_scanline_interaction for the first interaction.

    Args:
        primary_pts: Numpy array of primary points of shape (instances, nodes, 2).
        lateral_pts: Numpy array of lateral points of shape (instances, nodes, 2).
        depth: the depth of cylinder, or number of rows of the image.
        width: the width of cylinder, or number of columns of the image.
        n_line: number of scan lines, np.nan for no interaction.
        lateral_only: whether True: only lateral roots (e.g., rice), or False: dicots.

    Return:
        Scalar of count_scanline_interaction index for the first interaction.
    """
    count_scanline_interaction = count_scanline_intersections(
        primary_pts, lateral_pts, depth, width, n_line, lateral_only
    )
    if np.where((count_scanline_interaction > 0))[0].shape[0] > 0:
        scanline_first_ind = np.where((count_scanline_interaction > 0))[0][0]
        return scanline_first_ind
    else:
        return np.nan


def get_scanline_last_ind(
    primary_pts: np.ndarray,
    lateral_pts: np.ndarray,
    depth: int = 1080,
    width: int = 2048,
    n_line: int = 50,
    lateral_only: bool = False,
):
    """Get the index of count_scanline_interaction for the last interaction.

    Args:
        primary_pts: Numpy array of primary points of shape (instances, nodes, 2).
        lateral_pts: Numpy array of lateral points of shape (instances, nodes, 2).
        depth: the depth of cylinder, or number of rows of the image.
        width: the width of cylinder, or number of columns of the image.
        n_line: number of scan lines, np.nan for no interaction.
        lateral_only: whether True: only lateral roots (e.g., rice), or False: dicots.

    Return:
        Scalar of count_scanline_interaction index for the last interaction.
    """
    count_scanline_interaction = count_scanline_intersections(
        primary_pts, lateral_pts, depth, width, n_line, lateral_only
    )
    if np.where((count_scanline_interaction > 0))[0].shape[0] > 0:
        scanline_last_ind = np.where((count_scanline_interaction > 0))[0][-1]
        return scanline_last_ind
    else:
        return np.nan
