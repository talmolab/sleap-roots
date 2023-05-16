"""Extract traits based on the networkx graph."""

import numpy as np
from sleap_roots.traitsgraph import get_traits_graph
from sleap_roots.angle import get_root_angle
from sleap_roots.bases import (
    get_bases,
    get_base_ct_density,
    get_base_length,
    get_base_length_ratio,
    get_base_median_ratio,
    get_base_tip_dist,
    get_base_xs,
    get_base_ys,
    get_grav_index,
    get_lateral_count,
    get_primary_depth,
    get_root_lengths,
    get_root_pair_widths_projections,
)
from sleap_roots.convhull import (
    get_chull_area,
    get_chull_line_lengths,
    get_chull_max_width,
    get_chull_max_height,
    get_chull_perimeter,
    get_convhull_features,
)
from sleap_roots.ellipse import (
    fit_ellipse,
    get_ellipse_a,
    get_ellipse_b,
    get_ellipse_ratio,
)
from sleap_roots.networklength import (
    get_bbox,
    get_network_distribution_ratio,
    get_network_distribution,
    get_network_solidity,
    get_network_width_depth_ratio,
)
from sleap_roots.scanline import (
    count_scanline_intersections,
    get_scanline_first_ind,
    get_scanline_intersections,
    get_scanline_last_ind,
)
from sleap_roots.tips import get_tips, get_tip_xs, get_tip_ys


def get_traits_value_frame(
    primary_pts: np.ndarray,
    lateral_pts: np.ndarray,
    pts_all_array: np.ndarray,
    pts_all_list: list,
) -> dict:
    """Get SLEAP traits per frame based on graph.

    Args:
        primary_pts: primary points
        lateral_pts: lateral points
        pts_all_array: all points in array format
        pts_all_list: all points in list format

    Return:
        A dictionary with all traits.
    """
    trait_map = {
        # get_bases(pts: np.ndarray) -> np.ndarray
        "primary_base_pt": (get_bases, [primary_pts]),
        # get_root_angle(pts: np.ndarray, proximal=True, base_ind=0) -> np.ndarray
        "primary_angle_proximal": (get_root_angle, [primary_pts, True, 0]),
        "primary_angle_distal": (get_root_angle, [primary_pts, False, 0]),
        # get_root_lengths(pts: np.ndarray) -> np.ndarray
        "primary_length": (get_root_lengths, [primary_pts]),
        # get_tips(pts)
        "primary_tip_pt": (get_tips, [primary_pts]),
        # fit_ellipse(pts: np.ndarray) -> Tuple[float, float, float]
        "ellipse": (fit_ellipse, [pts_all_array]),
        # get_bbox(pts: np.ndarray) -> Tuple[float, float, float, float]
        "bounding_box": (get_bbox, [pts_all_array]),
        # get_root_pair_widths_projections(lateral_pts, primary_pts, tolerance)
        "stem_widths": (
            get_root_pair_widths_projections,
            [lateral_pts, primary_pts, 0.02],
        ),
        # get_convhull_features(pts: Union[np.ndarray, ConvexHull]) -> Tuple[float, float, float, float]
        "convex_hull": (get_convhull_features, [pts_all_array]),
        # get_scanline_intersections(pts: np.ndarray, depth: int = 1080, width: int = 2048, n_line: int = 50) -> list
        "scanline_intersections": (
            get_scanline_intersections,
            [pts_all_list, 1080, 2048, 50],
        ),
        # get_lateral_count(pts: np.ndarray)
        "lateral_count": (get_lateral_count, [lateral_pts]),
        # # get_root_angle(pts: np.ndarray, proximal=True, base_ind=0) -> np.ndarray
        "lateral_angles_proximal": (get_root_angle, [lateral_pts, True, 0]),
        "lateral_angles_distal": (get_root_angle, [lateral_pts, False, 0]),
        # get_root_lengths(pts: np.ndarray) -> np.ndarray
        "lateral_lengths": (get_root_lengths, [lateral_pts]),
        # get_bases(pts: np.ndarray) -> np.ndarray
        "lateral_base_pts": (get_bases, [lateral_pts]),
        # get_tips(pts)
        "lateral_tip_pts": (get_tips, [lateral_pts]),
        # get_base_ys(pts: np.ndarray) -> np.ndarray
        # or just based on primary_base_pt, but the primary_base_pt trait must generate before
        # "primary_base_pt_y": (get_pt_ys, [data["primary_base_pt"]]),
        "primary_base_pt_y": (get_base_ys, [primary_pts]),
        # get_base_ct_density(primary_pts, lateral_pts)
        "base_ct_density": (get_base_ct_density, [primary_pts, lateral_pts]),
        # get_network_solidity(pts: np.ndarray) -> float
        "network_solidity": (get_network_solidity, [pts_all_array]),
        # get_network_distribution_ratio(pts: np.ndarray, fraction: float = 2 / 3) -> float
        "network_distribution_ratio": (get_network_distribution_ratio, [pts_all_array]),
        # get_network_distribution(pts: np.ndarray, fraction: float = 2 / 3) -> float
        "network_length_lower": (get_network_distribution, [pts_all_array, 2 / 3]),
        # get_tip_ys(pts: np.ndarray) -> np.ndarray
        "primary_tip_pt_y": (get_tip_ys, [primary_pts]),
        # get_ellipse_a(pts_all_array: Union[np.ndarray, Tuple[float, float, float]])
        "ellipse_a": (get_ellipse_a, [pts_all_array]),
        # get_ellipse_b(pts_all_array: Union[np.ndarray, Tuple[float, float, float]])
        "ellipse_b": (get_ellipse_b, [pts_all_array]),
        # get_network_width_depth_ratio(pts: np.ndarray) -> float
        "network_width_depth_ratio": (get_network_width_depth_ratio, [pts_all_array]),
        # get_chull_perimeter(pts: Union[np.ndarray, ConvexHull, Tuple[float, float, float, float]])
        "chull_perimeter": (get_chull_perimeter, [pts_all_array]),
        # get_chull_area(pts: Union[np.ndarray, ConvexHull, Tuple[float, float, float, float]])
        "chull_area": (get_chull_area, [pts_all_array]),
        # get_chull_max_width(pts: Union[np.ndarray, ConvexHull, Tuple[float, float, float, float]])
        "chull_max_width": (get_chull_max_width, [pts_all_array]),
        # get_chull_max_height(pts: Union[np.ndarray, ConvexHull, Tuple[float, float, float, float]])
        "chull_max_height": (get_chull_max_height, [pts_all_array]),
        # get_chull_line_lengths(pts: Union[np.ndarray, ConvexHull]) -> np.ndarray
        "chull_line_lengths": (get_chull_line_lengths, [pts_all_array]),
        # count_scanline_intersections(pts: np.ndarray, depth: int = 1080, width: int = 2048, n_line: int = 50) -> np.ndarray
        "scanline_intersection_counts": (
            count_scanline_intersections,
            [pts_all_list, 1080, 2048, 50],
        ),
        # get_base_xs(pts: np.ndarray) -> np.ndarray
        "lateral_base_xs": (get_base_xs, [lateral_pts]),
        # get_base_ys(pts: np.ndarray) -> np.ndarray
        "lateral_base_ys": (get_base_ys, [lateral_pts]),
        # get_tip_xs(pts: np.ndarray) -> np.ndarray
        "lateral_tip_xs": (get_tip_xs, [lateral_pts]),
        # get_tip_ys(pts: np.ndarray) -> np.ndarray
        "lateral_tip_ys": (get_tip_ys, [lateral_pts]),
        # get_base_tip_dist(pts: np.ndarray) -> np.ndarray
        "primary_base_tip_dist": (get_base_tip_dist, [primary_pts]),
        # get_primary_depth(primary_pts)
        "primary_depth": (get_primary_depth, [primary_pts]),
        # get_base_median_ratio(primary_pts: np.ndarray, lateral_pts: np.ndarray)
        "base_median_ratio": (get_base_median_ratio, [primary_pts, lateral_pts]),
        # get_ellipse_ratio(pts_all_array: Union[np.ndarray, Tuple[float, float, float]])
        "ellipse_ratio": (get_ellipse_ratio, [pts_all_array]),
        # get_scanline_last_ind(pts: np.ndarray, depth: int = 1080, width: int = 2048, n_line: int = 50)
        "scanline_last_ind": (get_scanline_last_ind, [pts_all_list]),
        # get_scanline_first_ind(pts: np.ndarray, depth: int = 1080, width: int = 2048, n_line: int = 50)
        "scanline_first_ind": (get_scanline_first_ind, [pts_all_list]),
        # get_base_length(pts: np.ndarray)
        "base_length": (get_base_length, [lateral_pts]),
        # get_grav_index(pts: np.ndarray)
        "grav_index": (get_grav_index, [primary_pts]),
        # get_base_length_ratio(primary_pts: np.ndarray, lateral_pts: np.ndarray)
        "base_length_ratio": (get_base_length_ratio, [primary_pts, lateral_pts]),
    }

    dts = get_traits_graph()

    data = {}
    for trait_name in dts:
        print("trait name:", trait_name)
        outputs = (trait_name,)
        fn, inputs = trait_map[trait_name]
        fn_outputs = fn(*[input_trait for input_trait in inputs])
        data[trait_name] = fn_outputs
    return data
