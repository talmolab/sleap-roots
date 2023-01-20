"""Get intersections between roots and horizontal scan lines."""

import numpy as np
import math
from shapely import LineString


def get_pt_scanline(pts: np.ndarray, depth=1080, width=2048, Nline=50) -> list:
    """Get intersection points of roots and scan lines.
    
    Args:
        pts: Numpy array of points of shape (instances, nodes, 2).
        depth: the depth of cylinder, or number of rows of the image.
        width: the width of cylinder, or number of columns of the image.
        Nline: number of scan lines.
        
    Returns:
        A list of intersection xy location, with length of Nline, each has shape 
        (# intersection,2).
    """
    # connect the points to lines using shapely
    points = list(pts)
    
    # calculate interval between two scan lines
    Ninterval = math.ceil(depth/(Nline-1))
    
    intersection = []
    for i in range(Nline):
        y_loc = Ninterval * (i+1)
        line = LineString([(0,y_loc),(width,y_loc)])
        
        #intersection = np.zeros([5,2])
        intersection_line = []
        for j in range(len(points)):
            # filter out nan nodes
            pts_j = points[j][~np.isnan(points[j]).any(axis=1)]
            if pts_j.shape[0] > 1:
                if line.intersects(LineString(points[j])):
                    intersection_root = line.intersection(LineString(points[j]))
                    intersection_line.append([intersection_root.x,intersection_root.y])
    
        intersection.append(intersection_line)
    return intersection


def get_Npt_scanline(pts: np.ndarray, depth=1080, width=2048, Nline=50) -> np.ndarray:
    """Get number of intersection points of roots and scan lines.
    
    Args:
        pts: Numpy array of points of shape (instances, nodes, 2).
        depth: the depth of cylinder, or number of rows of the image.
        width: the width of cylinder, or number of columns of the image.
        Nline: number of scan lines.
        
    Returns:
        An array with shape of (#Nline,) of intersection numbers of each scan line.
    """
    intersection = get_pt_scanline(pts, depth, width, Nline)

    Ninter = []
    for i in range(len(intersection)):
        Num_inter = len(intersection[i])
        Ninter.append(Num_inter)
    Ninter = np.array(Ninter)
    return Ninter
