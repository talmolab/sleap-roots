"""Get intersections between roots and horizontal scan lines."""

import numpy as np
import math
from shapely import LineString, Point


def get_scanline_intersections(
    primary_pts: np.ndarray,
    lateral_pts: np.ndarray,
    depth: int = 1080,
    width: int = 2048,
    n_line: int = 50,
    lateral_only: bool = False,
) -> list:
    """Get intersection points of roots and scan lines.

    Args:
        primary_pts: Numpy array of primary points of shape (instances, nodes, 2).
        lateral_pts: Numpy array of lateral points of shape (instances, nodes, 2).
        depth: the depth of cylinder, or number of rows of the image.
        width: the width of cylinder, or number of columns of the image.
        n_line: number of scan lines.
        lateral_only: whether True: only lateral roots (e.g., rice), or False: dicots

    Returns:
        A list of intersection xy location, with length of Nline, each has shape
        (# intersection,2).
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
        y_loc = n_interval * (i + 1)
        line = LineString([(0, y_loc), (width, y_loc)])

        intersection_line = []
        for j in range(len(points)):
            # filter out nan nodes
            pts_j = np.array(points[j])[~np.isnan(points[j]).any(axis=1)]
            if pts_j.shape[0] > 1:
                if line.intersects(LineString(pts_j)):
                    intersection_root = line.intersection(LineString(pts_j))
                    # Get the coordinates of points within the MultiPoint object
                    if type(intersection_root) == Point:
                        intersection_line.append(
                            [intersection_root.x, intersection_root.y]
                        )
                    else:
                        for point in intersection_root.geoms:
                            intersection_line.append([point.x, point.y])
        intersection.append(intersection_line)
    return intersection


def count_scanline_intersections(
    primary_pts: np.ndarray,
    lateral_pts: np.ndarray,
    depth: int = 1080,
    width: int = 2048,
    n_line: int = 50,
    lateral_only: bool = False,
) -> np.ndarray:
    """Get number of intersection points of roots and scan lines.

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
    intersection = get_scanline_intersections(
        primary_pts, lateral_pts, depth, width, n_line, lateral_only
    )
    n_inter = []
    for i in range(len(intersection)):
        if len(intersection[i]) > 0:
            num_inter = len(intersection[i])
        else:
            num_inter = np.nan
        n_inter.append(num_inter)
    Ninter = np.array(n_inter)
    return Ninter


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
    scanline_first_ind = np.where((count_scanline_interaction > 0))[0][0]
    return scanline_first_ind


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
    scanline_last_ind = np.where((count_scanline_interaction > 0))[0][-1]
    return scanline_last_ind
