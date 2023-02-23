"""Get intersections between roots and horizontal scan lines."""

import numpy as np
import math
from shapely import LineString


def get_scanline_intersections(
    pts: np.ndarray, depth: int = 1080, width: int = 2048, n_line: int = 50
) -> list:
    """Get intersection points of roots and scan lines.

    Args:
        pts: Numpy array of points of shape (instances, nodes, 2).
        depth: the depth of cylinder, or number of rows of the image.
        width: the width of cylinder, or number of columns of the image.
        n_line: number of scan lines.

    Returns:
        A list of intersection xy location, with length of Nline, each has shape
        (# intersection,2).
    """
    # connect the points to lines using shapely
    points = list(pts)

    # calculate interval between two scan lines
    n_interval = math.ceil(depth / (n_line - 1))

    intersection = []
    for i in range(n_line):
        y_loc = n_interval * (i + 1)
        line = LineString([(0, y_loc), (width, y_loc)])

        intersection_line = []
        for j in range(len(points)):
            # filter out nan nodes
            pts_j = points[j][~np.isnan(points[j]).any(axis=1)]
            if pts_j.shape[0] > 1:
                if line.intersects(LineString(pts_j)):
                    intersection_root = line.intersection(LineString(pts_j))
                    # append intersection(s) location
                    intersection_line.append(
                        [intersection_root.x, intersection_root.y]
                    ) if intersection_root.geom_type == "Point" else [
                        intersection_line.append(
                            [
                                list(intersection_root.geoms)[k].x,
                                list(intersection_root.geoms)[k].y,
                            ]
                        )
                        for k in range(len(intersection_root.geoms))
                    ]
        intersection.append(intersection_line)
    return intersection


def count_scanline_intersections(
    pts: np.ndarray, depth: int = 1080, width: int = 2048, n_line: int = 50
) -> np.ndarray:
    """Get number of intersection points of roots and scan lines.

    Args:
        pts: Numpy array of points of shape (instances, nodes, 2).
        depth: the depth of cylinder, or number of rows of the image.
        width: the width of cylinder, or number of columns of the image.
        n_line: number of scan lines, np.nan for no interaction.

    Returns:
        An array with shape of (#Nline,) of intersection numbers of each scan line.
    """
    intersection = get_scanline_intersections(pts, depth, width, n_line)
    n_inter = []
    for i in range(len(intersection)):
        if len(intersection[i]) > 0:
            num_inter = len(intersection[i])
        else:
            num_inter = np.nan
        n_inter.append(num_inter)
    Ninter = np.array(n_inter)
    return Ninter
