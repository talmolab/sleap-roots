"""Extract traits based on the networkx graph."""

import numpy as np
from sleap_roots.traitsgraph import get_traits_graph
from sleap_roots.bases import get_bases, get_root_pair_widths_projections
from sleap_roots.bases import get_lateral_count
from sleap_roots.bases import get_bases
from sleap_roots.bases import get_base_ys
from sleap_roots.bases import get_base_xs
from sleap_roots.bases import get_base_tip_dist
from sleap_roots.angle import get_root_angle  # get_node_ind,
from sleap_roots.bases import get_root_lengths
from sleap_roots.bases import get_base_length
from sleap_roots.bases import get_grav_index
from sleap_roots.bases import get_root_lengths_max
from sleap_roots.tips import get_tips
from sleap_roots.tips import get_tip_ys
from sleap_roots.tips import get_tip_xs
from sleap_roots.ellipse import fit_ellipse
from sleap_roots.networklength import get_bbox
from sleap_roots.networklength import get_network_solidity
from sleap_roots.networklength import (
    get_network_distribution_ratio,
    get_network_distribution,
)
from sleap_roots.convhull import get_convhull_features
from sleap_roots.convhull import get_chull_line_lengths
from sleap_roots.scanline import (
    count_scanline_intersections,
    get_scanline_intersections,
)


def get_traits_value_frame(
    primary_pts: np.ndarray,
    lateral_pts: np.ndarray,
    pts_all_array: np.ndarray,
    pts_all_list: list,
) -> dict:
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
        # get_network_solidity(pts: np.ndarray) -> float
        "network_solidity": (get_network_solidity, [pts_all_array]),
        # get_network_distribution_ratio(pts: np.ndarray, fraction: float = 2 / 3) -> float
        "network_distribution_ratio": (get_network_distribution_ratio, [pts_all_array]),
        # get_network_distribution(pts: np.ndarray, fraction: float = 2 / 3) -> float
        "network_length_lower": (get_network_distribution, [pts_all_array, 2 / 3]),
        # get_tip_ys(pts: np.ndarray) -> np.ndarray
        "primary_tip_pt_y": (get_tip_ys, [primary_pts]),
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
        # get_base_length(pts: np.ndarray)
        "base_length": (get_base_length, [lateral_pts]),
        # get_grav_index(pts: np.ndarray)
        "grav_index": (get_grav_index, [primary_pts]),
    }

    dts = get_traits_graph()

    data = {}
    for trait_name in dts:
        print("trait name:", trait_name)
        # outputs = ("primary_base_pt",)
        outputs = (trait_name,)
        fn, inputs = trait_map[trait_name]
        fn_outputs = fn(*[input_trait for input_trait in inputs])
        # if type(fn_outputs) == tuple:
        #     fn_outputs = np.array(fn_outputs).reshape((1, -1))
        # if type(fn_outputs) == np.ndarray and fn_outputs.shape == (fn_outputs.size,):
        #     fn_outputs = fn_outputs.reshape((1,-1))
        # if type(fn_outputs) == int:
        #     fn_outputs = np.array(fn_outputs).reshape((1,-1))
        # fn_outputs = fn(*[data[input_trait] for input_trait in inputs]) works for only input of points
        # for k, v in zip(outputs, fn_outputs):
        #     data[k] = v
        data[trait_name] = fn_outputs
    return data
