"""Extract traits in a pipeline based on the trait graph."""

import numpy as np
import pandas as pd
import attrs
from typing import List, Dict, Tuple, Callable, Optional, Any
from fractions import Fraction
import networkx as nx
from pathlib import Path
from sleap_roots.angle import get_root_angle, get_node_ind
from sleap_roots.lengths import get_root_lengths, get_grav_index, get_max_length_pts
from sleap_roots.bases import (
    get_bases,
    get_base_ct_density,
    get_base_length,
    get_base_length_ratio,
    get_base_median_ratio,
    get_base_tip_dist,
    get_base_xs,
    get_base_ys,
    get_lateral_count,
    get_root_pair_widths_projections,
)
from sleap_roots.tips import get_tips, get_tip_xs, get_tip_ys
from sleap_roots.convhull import (
    get_convhull,
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
from sleap_roots.points import get_all_pts_array, get_all_pts
from sleap_roots.scanline import (
    count_scanline_intersections,
    get_scanline_first_ind,
    get_scanline_last_ind,
)
from sleap_roots.series import Series, find_all_series
from sleap_roots.summary import get_summary
import warnings


SCALAR_TRAITS = (
    "primary_angle_proximal",
    "primary_angle_distal",
    "primary_length",
    "primary_base_tip_dist",
    "lateral_count",
    "grav_index",
    "base_length",
    "base_length_ratio",
    "primary_tip_pt_y",
    "base_median_ratio",
    "base_ct_density",
    "chull_perimeter",
    "chull_area",
    "chull_max_width",
    "chull_max_height",
    "ellipse_a",
    "ellipse_b",
    "ellipse_ratio",
    "network_width_depth_ratio",
    "network_solidity",
    "network_length_lower",
    "network_distribution_ratio",
    "scanline_first_ind",
    "scanline_last_ind",
)

NON_SCALAR_TRAITS = (
    "lateral_angles_proximal",
    "lateral_angles_distal",
    "lateral_lengths",
    "root_widths",
    "lateral_base_xs",
    "lateral_base_ys",
    "lateral_tip_xs",
    "lateral_tip_ys",
    "chull_line_lengths",
    "scanline_intersection_counts",
)


warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in intersection",
    category=RuntimeWarning,
    module="shapely",
)
warnings.filterwarnings(
    "ignore", message="All-NaN slice encountered", category=RuntimeWarning
)
warnings.filterwarnings(
    "ignore", message="All-NaN axis encountered", category=RuntimeWarning
)
warnings.filterwarnings(
    "ignore",
    message="Degrees of freedom <= 0 for slice.",
    category=RuntimeWarning,
    module="numpy",
)
warnings.filterwarnings(
    "ignore", message="Mean of empty slice", category=RuntimeWarning
)
warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in sqrt",
    category=RuntimeWarning,
    module="skimage",
)
warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in double_scalars",
    category=RuntimeWarning,
)
warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in scalar divide",
    category=RuntimeWarning,
    module="ellipse",
)


@attrs.define
class TraitDef:
    """Definition of how to compute a trait.

    Attributes:
        name: Unique identifier for the trait.
        fn: Function used to compute the trait's value.
        input_traits: List of trait names that should be computed before the current
            trait and are expected as input positional arguments to `fn`.
        scalar: Indicates if the trait is scalar (has a dimension of 0 per frame). If
            `True`, the trait is also listed in `SCALAR_TRAITS`.
        include_in_csv: `True `indicates the trait should be included in downstream CSV
            files.
        kwargs: Additional keyword arguments to be passed to the `fn` function. These
            arguments are not reused from previously computed traits.
        description: String describing the trait for documentation purposes.

    Notes:
        The `fn` specified will be called with a pattern like:

        ```
        trait_def = TraitDef(
            name="my_trait",
            fn=compute_my_trait,
            input_traits=["input_trait_1", "input_trait_2"],
            scalar=True,
            include_in_csv=True,
            kwargs={"kwarg1": True}
        )
        traits[trait_def.name] = trait_def.fn(
            *[traits[input_trait] for input_trait in trait_def.input_traits],
            **trait_def.kwargs
        )
        ```

        For this example, the last line is equivalent to:

        ```
        traits["my_trait"] = trait_def.fn(
            traits["input_trait_1"], traits["input_trait_2"],
            kwarg1=True
        )
        ```
    """

    name: str
    fn: Callable
    input_traits: List[str]
    scalar: bool
    include_in_csv: bool
    kwargs: Dict[str, Any] = attrs.field(factory=dict)
    description: Optional[str] = None


def get_traits_value_frame(
    primary_pts: np.ndarray,
    lateral_pts: np.ndarray,
    root_width_tolerance: float = 0.02,
    n_line: int = 50,
    network_fraction: float = 2 / 3,
    monocots: bool = False,
) -> Dict:
    """Get SLEAP traits per frame based on graph.

    Args:
        primary_pts: primary points
        lateral_pts: lateral points
        root_width_tolerance: Difference in projection norm between right and left side.
        n_line: number of scan lines, np.nan for no interaction.
        network_fraction: length found in the lower fration value of the network.
        monocots: Boolean value, where false is dicot (default), true is rice.

    Return:
        A dictionary with all traits per frame.
    """
    # Define the trait computations.
    trait_definitions = [
        TraitDef(
            name="primary_max_length_pts",
            fn=get_max_length_pts,
            input_traits=["primary_pts"],
            scalar=False,
            include_in_csv=False,
            kwargs={},
            description="Points of the primary root with maximum length.",
        ),
        TraitDef(
            name="pts_all_array",
            fn=get_all_pts_array,
            input_traits=["primary_max_length_pts", "lateral_pts"],
            scalar=False,
            include_in_csv=False,
            kwargs={"monocots": monocots},
            description="Landmark points within a given frame as a flat array"
            "of coordinates.",
        ),
        TraitDef(
            name="root_widths",
            fn=get_root_pair_widths_projections,
            input_traits=["primary_max_length_pts", "lateral_pts"],
            scalar=False,
            include_in_csv=True,
            kwargs={"tolerance": root_width_tolerance, "monocots": monocots},
            description="Return estimation of root width using bases of lateral roots.",
        ),
        TraitDef(
            name="lateral_count",
            fn=get_lateral_count,
            input_traits=["lateral_pts"],
            scalar=True,
            include_in_csv=True,
            kwargs={},
            description="Get the number of lateral roots.",
        ),
        TraitDef(
            name="lateral_proximal_node_inds",
            fn=get_node_ind,
            input_traits=["lateral_pts"],
            scalar=False,
            include_in_csv=False,
            kwargs={"proximal": True},
            description="Get the indices of the proximal nodes of lateral roots.",
        ),
        TraitDef(
            name="lateral_distal_node_inds",
            fn=get_node_ind,
            input_traits=["lateral_pts"],
            scalar=False,
            include_in_csv=False,
            kwargs={"proximal": False},
            description="Get the indices of the distal nodes of lateral roots.",
        ),
        TraitDef(
            name="lateral_lengths",
            fn=get_root_lengths,
            input_traits=["lateral_pts"],
            scalar=False,
            include_in_csv=True,
            kwargs={},
            description="Array of root lengths of shape `(instances,)`.",
        ),
        TraitDef(
            name="lateral_base_pts",
            fn=get_bases,
            input_traits=["lateral_pts"],
            scalar=False,
            include_in_csv=False,
            kwargs={"monocots": monocots},
            description="Array of lateral bases `(instances, (x, y))`.",
        ),
        TraitDef(
            name="lateral_tip_pts",
            fn=get_tips,
            input_traits=["lateral_pts"],
            scalar=False,
            include_in_csv=False,
            kwargs={"monocots": monocots},
            description="Array of lateral tips `(instances, (x, y))`.",
        ),
        TraitDef(
            name="scanline_intersection_counts",
            fn=count_scanline_intersections,
            input_traits=["primary_max_length_pts", "lateral_pts"],
            scalar=False,
            include_in_csv=True,
            kwargs={
                "depth": 1080,
                "width": 2048,
                "n_line": n_line,
                "monocots": monocots,
            },
            description="Array of intersections of each scanline `(#Nline,)`.",
        ),
        TraitDef(
            name="lateral_angles_distal",
            fn=get_root_angle,
            input_traits=["lateral_pts", "lateral_distal_node_inds"],
            scalar=False,
            include_in_csv=True,
            kwargs={"proximal": False, "base_ind": 0},
            description="Array of lateral distal angles in degrees `(instances,)`.",
        ),
        TraitDef(
            name="lateral_angles_proximal",
            fn=get_root_angle,
            input_traits=["lateral_pts", "lateral_proximal_node_inds"],
            scalar=False,
            include_in_csv=True,
            kwargs={"proximal": True, "base_ind": 0},
            description="Array of lateral proximal angles in degrees `(instances,)`.",
        ),
        TraitDef(
            name="network_solidity",
            fn=get_network_solidity,
            input_traits=["primary_max_length_pts", "lateral_pts", "chull_area"],
            scalar=True,
            include_in_csv=True,
            kwargs={"monocots": monocots},
            description="Scalar of the total network length divided by the network convex area.",
        ),
        TraitDef(
            name="ellipse",
            fn=fit_ellipse,
            input_traits=["pts_all_array"],
            scalar=False,
            include_in_csv=False,
            kwargs={},
            description="Tuple of (a, b, ratio) containing the semi-major axis length, semi-minor axis length, and the ratio of the major to minor lengths.",
        ),
        TraitDef(
            name="bounding_box",
            fn=get_bbox,
            input_traits=["pts_all_array"],
            scalar=False,
            include_in_csv=False,
            kwargs={},
            description="Tuple of four parameters in bounding box.",
        ),
        TraitDef(
            name="convex_hull",
            fn=get_convhull,
            input_traits=["pts_all_array"],
            scalar=False,
            include_in_csv=False,
            kwargs={"monocots": monocots},
            description="Convex hull of the points.",
        ),
        TraitDef(
            name="primary_proximal_node_ind",
            fn=get_node_ind,
            input_traits=["primary_max_length_pts"],
            scalar=True,
            include_in_csv=False,
            kwargs={"proximal": True},
            description="Get the indices of the proximal nodes of primary roots.",
        ),
        TraitDef(
            name="primary_angle_proximal",
            fn=get_root_angle,
            input_traits=["primary_max_length_pts", "primary_proximal_node_ind"],
            scalar=True,
            include_in_csv=True,
            kwargs={"proximal": True, "base_ind": 0},
            description="Array of primary proximal angles in degrees `(instances,)`.",
        ),
        TraitDef(
            name="primary_distal_node_ind",
            fn=get_node_ind,
            input_traits=["primary_max_length_pts"],
            scalar=True,
            include_in_csv=False,
            kwargs={"proximal": False},
            description="Get the indices of the distal nodes of primary roots.",
        ),
        TraitDef(
            name="primary_angle_distal",
            fn=get_root_angle,
            input_traits=["primary_max_length_pts", "primary_distal_node_ind"],
            scalar=True,
            include_in_csv=True,
            kwargs={"proximal": False, "base_ind": 0},
            description="Array of primary distal angles in degrees `(instances,)`.",
        ),
        TraitDef(
            name="primary_length",
            fn=get_root_lengths,
            input_traits=["primary_max_length_pts"],
            scalar=True,
            include_in_csv=True,
            kwargs={},
            description="Scalar of primary root length.",
        ),
        TraitDef(
            name="primary_base_pt",
            fn=get_bases,
            input_traits=["primary_max_length_pts"],
            scalar=False,
            include_in_csv=False,
            kwargs={"monocots": monocots},
            description="Array of primary bases `(, (x, y))`.",
        ),
        TraitDef(
            name="primary_tip_pt",
            fn=get_tips,
            input_traits=["primary_max_length_pts"],
            scalar=False,
            include_in_csv=False,
            kwargs={"monocots": monocots},
            description="Array of primary tips `(, (x, y))`.",
        ),
        TraitDef(
            name="network_length_lower",
            fn=get_network_distribution,
            input_traits=["primary_max_length_pts", "lateral_pts", "bounding_box"],
            scalar=True,
            include_in_csv=True,
            kwargs={"fraction": network_fraction, "monocots": monocots},
            description="Scalar of the root network length in the lower fraction of the plant.",
        ),
        TraitDef(
            name="lateral_base_xs",
            fn=get_base_xs,
            input_traits=["lateral_base_pts"],
            scalar=False,
            include_in_csv=True,
            kwargs={"monocots": monocots},
            description="Array of the x-coordinates of lateral bases `(instance,)`.",
        ),
        TraitDef(
            name="lateral_base_ys",
            fn=get_base_ys,
            input_traits=["lateral_base_pts"],
            scalar=False,
            include_in_csv=True,
            kwargs={"monocots": monocots},
            description="Array of the y-coordinates of lateral bases `(instance,)`.",
        ),
        TraitDef(
            name="base_ct_density",
            fn=get_base_ct_density,
            input_traits=["primary_length", "lateral_base_pts"],
            scalar=True,
            include_in_csv=True,
            kwargs={"monocots": monocots},
            description="Scalar of base count density.",
        ),
        TraitDef(
            name="lateral_tip_xs",
            fn=get_tip_xs,
            input_traits=["lateral_tip_pts"],
            scalar=False,
            include_in_csv=True,
            kwargs={"monocots": monocots},
            description="Array of the x-coordinates of lateral tips `(instance,)`.",
        ),
        TraitDef(
            name="lateral_tip_ys",
            fn=get_tip_ys,
            input_traits=["lateral_tip_pts"],
            scalar=False,
            include_in_csv=True,
            kwargs={"monocots": monocots},
            description="Array of the y-coordinates of lateral tips `(instance,)`.",
        ),
        TraitDef(
            name="network_distribution_ratio",
            fn=get_network_distribution_ratio,
            input_traits=["primary_length", "lateral_lengths", "network_length_lower"],
            scalar=True,
            include_in_csv=True,
            kwargs={"fraction": network_fraction, "monocots": monocots},
            description="Scalar of ratio of the root network length in the lower fraction of the plant over all root length.",
        ),
        TraitDef(
            name="primary_base_pt_y",
            fn=get_base_ys,
            input_traits=["primary_base_pt"],
            scalar=True,
            include_in_csv=False,
            kwargs={"monocots": monocots},
            description="Scalar of the y-coordinates of primary base `(instance,)`.",
        ),
        TraitDef(
            name="primary_tip_pt_y",
            fn=get_tip_ys,
            input_traits=["primary_base_pt"],
            scalar=True,
            include_in_csv=False,
            kwargs={"monocots": monocots},
            description="Scalar of the y-coordinates of primary tip `(instance,)`.",
        ),
        TraitDef(
            name="ellipse_a",
            fn=get_ellipse_a,
            input_traits=["ellipse"],
            scalar=True,
            include_in_csv=True,
            kwargs={},
            description="Scalar of semi-major axis length.",
        ),
        TraitDef(
            name="ellipse_b",
            fn=get_ellipse_b,
            input_traits=["ellipse"],
            scalar=True,
            include_in_csv=True,
            kwargs={},
            description="Scalar of semi-minor axis length.",
        ),
        TraitDef(
            name="chull_perimeter",
            fn=get_chull_perimeter,
            input_traits=["convex_hull"],
            scalar=True,
            include_in_csv=True,
            kwargs={},
            description="Scalar of convex hull perimeter.",
        ),
        TraitDef(
            name="chull_area",
            fn=get_chull_area,
            input_traits=["convex_hull"],
            scalar=True,
            include_in_csv=True,
            kwargs={},
            description="Scalar of convex hull area.",
        ),
        TraitDef(
            name="chull_max_width",
            fn=get_chull_max_width,
            input_traits=["convex_hull"],
            scalar=True,
            include_in_csv=True,
            kwargs={},
            description="Scalar of convex hull maximum width.",
        ),
        TraitDef(
            name="chull_max_height",
            fn=get_chull_max_height,
            input_traits=["convex_hull"],
            scalar=True,
            include_in_csv=True,
            kwargs={},
            description="Scalar of convex hull maximum height.",
        ),
        TraitDef(
            name="chull_line_lengths",
            fn=get_chull_line_lengths,
            input_traits=["convex_hull"],
            scalar=False,
            include_in_csv=True,
            kwargs={},
            description="Array of line lengths connecting any two vertices on the convex hull.",
        ),
        TraitDef(
            name="base_length",
            fn=get_base_length,
            input_traits=["lateral_base_ys"],
            scalar=True,
            include_in_csv=True,
            kwargs={},
            description="Scalar of the distance between the top and deepest base y-coordinates.",
        ),
        TraitDef(
            name="base_median_ratio",
            fn=get_base_median_ratio,
            input_traits=["lateral_base_ys", "primary_tip_pt_y"],
            scalar=True,
            include_in_csv=True,
            kwargs={"monocots": monocots},
            description="Scalar of base median ratio.",
        ),
        TraitDef(
            name="grav_index",
            fn=get_grav_index,
            input_traits=["primary_length", "primary_base_tip_dist"],
            scalar=True,
            include_in_csv=True,
            kwargs={"monocots": monocots},
            description="Scalar of primary root gravity index.",
        ),
        TraitDef(
            name="base_length_ratio",
            fn=get_base_length_ratio,
            input_traits=["primary_length", "base_length"],
            scalar=True,
            include_in_csv=True,
            kwargs={"monocots": monocots},
            description="Scalar of base length ratio.",
        ),
        TraitDef(
            name="primary_base_tip_dist",
            fn=get_base_tip_dist,
            input_traits=["primary_base_pt_y", "primary_tip_pt_y"],
            scalar=True,
            include_in_csv=True,
            kwargs={},
            description="Scalar of distances from primary base to tip.",
        ),
        TraitDef(
            name="ellipse_ratio",
            fn=get_ellipse_ratio,
            input_traits=["ellipse"],
            scalar=True,
            include_in_csv=True,
            kwargs={},
            description="Scalar of ratio of the minor to major lengths.",
        ),
        TraitDef(
            name="scanline_last_ind",
            fn=get_scanline_last_ind,
            input_traits=["scanline_intersection_counts"],
            scalar=True,
            include_in_csv=True,
            kwargs={
                "depth": 1080,
                "width": 2048,
                "n_line": n_line,
                "monocots": monocots,
            },
            description="Scalar of count_scanline_interaction index for the last interaction.",
        ),
        TraitDef(
            name="scanline_first_ind",
            fn=get_scanline_first_ind,
            input_traits=["scanline_intersection_counts"],
            scalar=True,
            include_in_csv=True,
            kwargs={
                "depth": 1080,
                "width": 2048,
                "n_line": n_line,
                "monocots": monocots,
            },
            description="Scalar of count_scanline_interaction index for the first interaction.",
        ),
    ]

    # Map trait names to their definitions.
    trait_map = {trait_def.name: trait_def for trait_def in trait_definitions}

    # trait_map = {
    #     # get_bases(pts: np.ndarray,monocots) -> np.ndarray
    #     "primary_base_pt": (get_bases, ["primary_pts"], {"monocots": monocots}),
    #     # get_root_angle(pts: np.ndarray, proximal=True, base_ind=0) -> np.ndarray
    #     "primary_angle_proximal": (
    #         get_root_angle,
    #         ["primary_pts"],
    #         {"proximal": True, "base_ind": 0},
    #     ),
    #     "primary_angle_distal": (
    #         get_root_angle,
    #         ["primary_pts"],
    #         {"proximal": False, "base_ind": 0},
    #     ),
    #     # get_root_lengths(pts: np.ndarray) -> np.ndarray
    #     "primary_length": (get_root_lengths, ["primary_pts"], {}),
    #     # get_tips(pts)
    #     "primary_tip_pt": (get_tips, ["primary_pts"], {}),
    #     # fit_ellipse(pts: np.ndarray) -> Tuple[float, float, float]
    #     "ellipse": (fit_ellipse, ["pts_all_array"], {}),
    #     # get_bbox(pts: np.ndarray) -> Tuple[float, float, float, float]
    #     "bounding_box": (get_bbox, ["pts_all_array"], {}),
    #     # get_root_pair_widths_projections(lateral_pts, primary_pts, tolerance,monocots)
    #     "root_widths": (
    #         get_root_pair_widths_projections,
    #         ["primary_pts", "lateral_pts"],
    #         {"root_width_tolerance": root_width_tolerance, "monocots": monocots},
    #     ),
    #     # get_convhull_features(pts: Union[np.ndarray, ConvexHull]) -> Tuple[float, float, float, float]
    #     "convex_hull": (get_convhull_features, ["pts_all_array"], {}),
    #     # get_lateral_count(pts: np.ndarray)
    #     "lateral_count": (get_lateral_count, ["lateral_pts"], {}),
    #     # # get_root_angle(pts: np.ndarray, proximal=True, base_ind=0) -> np.ndarray
    #     "lateral_angles_proximal": (
    #         get_root_angle,
    #         ["lateral_pts"],
    #         {"proximal": True, "base_ind": 0},
    #     ),
    #     "lateral_angles_distal": (
    #         get_root_angle,
    #         ["lateral_pts"],
    #         {"proximal": False, "base_ind": 0},
    #     ),
    #     # get_root_lengths(pts: np.ndarray) -> np.ndarray
    #     "lateral_lengths": (get_root_lengths, ["lateral_pts"], {}),
    #     # get_bases(pts: np.ndarray,monocots) -> np.ndarray
    #     "lateral_base_pts": (get_bases, ["lateral_pts"], {"monocots": monocots}),
    #     # get_tips(pts)
    #     "lateral_tip_pts": (get_tips, ["lateral_pts"], {}),
    #     # get_base_ys(pts: np.ndarray) -> np.ndarray
    #     # or just based on primary_base_pt, but the primary_base_pt trait must generate before
    #     # "primary_base_pt_y": (get_pt_ys, [data["primary_base_pt"]]),
    #     "primary_base_pt_y": (get_base_ys, ["primary_pts"], {}),
    #     # get_base_ct_density(primary_pts, lateral_pts)
    #     "base_ct_density": (
    #         get_base_ct_density,
    #         [
    #             "primary_pts",
    #             "lateral_pts",
    #         ],
    #         {"monocots": monocots},
    #     ),
    #     # get_network_solidity(primary_pts: np.ndarray, lateral_pts: np.ndarray, pts_all_array: np.ndarray, monocots: bool = False,) -> float
    #     "network_solidity": (
    #         get_network_solidity,
    #         ["primary_pts", "lateral_pts", "chull_area"],
    #         {"monocots": monocots},
    #     ),
    #     # get_network_distribution_ratio(primary_pts: np.ndarray,lateral_pts: np.ndarray,pts_all_array: np.ndarray,fraction: float = 2 / 3, monocots: bool = False) -> float:
    #     "network_distribution_ratio": (
    #         get_network_distribution_ratio,
    #         [
    #             "primary_length",
    #             "lateral_lengths",
    #             "network_length_lower",
    #         ],
    #         {"network_fraction": network_fraction, "monocots": monocots},
    #     ),
    #     # get_network_distribution(primary_pts: np.ndarray,lateral_pts: np.ndarray,pts_all_array: np.ndarray,fraction: float = 2 / 3, monocots: bool = False) -> float:
    #     "network_length_lower": (
    #         get_network_distribution,
    #         ["primary_pts", "lateral_pts", "bounding_box"],
    #         {"network_fraction": network_fraction, "monocots": monocots},
    #     ),
    #     # get_network_width_depth_ratio(pts: np.ndarray) -> float
    #     "network_width_depth_ratio": (
    #         get_network_width_depth_ratio,
    #         ["bounding_box"],
    #         {},
    #     ),
    #     # get_tip_ys(pts: np.ndarray) -> np.ndarray
    #     "primary_tip_pt_y": (get_tip_ys, ["primary_tip_pt"], {}),
    #     # get_ellipse_a(pts_all_array: Union[np.ndarray, Tuple[float, float, float]])
    #     "ellipse_a": (get_ellipse_a, ["ellipse"], {}),
    #     # get_ellipse_b(pts_all_array: Union[np.ndarray, Tuple[float, float, float]])
    #     "ellipse_b": (get_ellipse_b, ["ellipse"], {}),
    #     # get_ellipse_ratio(pts_all_array: Union[np.ndarray, Tuple[float, float, float]])
    #     "ellipse_ratio": (get_ellipse_ratio, ["ellipse"], {}),
    #     # get_chull_perimeter(pts: Union[np.ndarray, ConvexHull, Tuple[float, float, float, float]])
    #     "chull_perimeter": (get_chull_perimeter, ["convex_hull"], {}),
    #     # get_chull_area(pts: Union[np.ndarray, ConvexHull, Tuple[float, float, float, float]])
    #     "chull_area": (get_chull_area, ["convex_hull"], {}),
    #     # get_chull_max_width(pts: Union[np.ndarray, ConvexHull, Tuple[float, float, float, float]])
    #     "chull_max_width": (get_chull_max_width, ["convex_hull"], {}),
    #     # get_chull_max_height(pts: Union[np.ndarray, ConvexHull, Tuple[float, float, float, float]])
    #     "chull_max_height": (get_chull_max_height, ["convex_hull"], {}),
    #     # get_chull_line_lengths(pts: Union[np.ndarray, ConvexHull]) -> np.ndarray
    #     "chull_line_lengths": (get_chull_line_lengths, ["convex_hull"], {}),
    #     # scanline_intersection_counts:
    #     "scanline_intersection_counts": (
    #         count_scanline_intersections,
    #         [primary_pts, lateral_pts],
    #         {"depth": 1080, "width": 2048, "n_line": 50, "monocots": monocots},
    #     ),
    #     # get_base_xs(pts: np.ndarray) -> np.ndarray
    #     "lateral_base_xs": (get_base_xs, ["lateral_base_pts"], {"monocots": monocots}),
    #     # get_base_ys(pts: np.ndarray) -> np.ndarray
    #     "lateral_base_ys": (get_base_ys, ["lateral_base_pts"], {"monocots": monocots}),
    #     # get_tip_xs(pts: np.ndarray) -> np.ndarray
    #     "lateral_tip_xs": (get_tip_xs, ["lateral_tip_pts"], {"monocots": monocots}),
    #     # get_tip_ys(pts: np.ndarray) -> np.ndarray
    #     "lateral_tip_ys": (get_tip_ys, ["lateral_tip_pts"], {"monocots": monocots}),
    #     # get_base_tip_dist(pts: np.ndarray) -> np.ndarray
    #     "primary_base_tip_dist": (
    #         get_base_tip_dist,
    #         {
    #             "base_pts": "primary_base_pt",
    #             "tip_pts": "primary_tip_pt",
    #             "pts": "primary_pts",
    #         },
    #     ),
    #     # get_base_median_ratio(primary_pts: np.ndarray, lateral_pts: np.ndarray)
    #     "base_median_ratio": (
    #         get_base_median_ratio,
    #         ["lateral_base_ys", "primary_tip_pt_y"],
    #         {"monocots": monocots},
    #     ),
    #     # get_scanline_last_ind(primary_pts: np.ndarray,lateral_pts: np.ndarray,depth: int = 1080, width: int = 2048, n_line: int = 50, monocots: bool = False)
    #     "scanline_last_ind": (
    #         get_scanline_last_ind,
    #         [primary_pts, lateral_pts, 1080, 2048, n_line, monocots],
    #     ),
    #     # get_scanline_first_ind(primary_pts: np.ndarray,lateral_pts: np.ndarray,depth: int = 1080, width: int = 2048, n_line: int = 50, monocots: bool = False)
    #     "scanline_first_ind": (
    #         get_scanline_first_ind,
    #         [primary_pts, lateral_pts, 1080, 2048, n_line, monocots],
    #     ),
    #     # get_base_length(pts: np.ndarray)
    #     "base_length": (get_base_length, [lateral_pts, monocots]),
    #     # get_grav_index(pts: np.ndarray)
    #     "grav_index": (get_grav_index, [primary_pts]),
    #     # get_base_length_ratio(primary_pts: np.ndarray, lateral_pts: np.ndarray)
    #     "base_length_ratio": (get_base_length_ratio, [primary_pts, lateral_pts]),
    # }

    # Initialize edges with precomputed top-level traits.
    edges = [("pts", "primary_pts"), ("pts", "lateral_pts")]

    # Infer edges from trait map.
    # for output_trait, (_, input_traits, _) in trait_map.items():
    #     for input_trait in input_traits:
    #         edges.append((input_trait, output_trait))
    for trait_def in trait_definitions:
        for input_trait in trait_def.input_traits:
            edges.append((input_trait, trait_def.name))

    # Compute breadth-first ordering.
    G = nx.DiGraph()
    G.add_edges_from(edges)
    trait_computation_order = [
        dst for (src, dst) in list(nx.bfs_tree(G, "pts").edges())[2:]
    ]

    # Initialize traits container with initial points.
    traits = {"primary_pts": primary_pts, "lateral_pts": lateral_pts}

    # Compute traits!
    for trait_name in trait_computation_order:
        # fn, input_traits, kwargs = trait_map[trait_name]
        trait_def = trait_map[trait_name]

        traits[trait_name] = trait_def.fn(
            *[traits[input_trait] for input_trait in trait_def.input_traits],
            **trait_def.kwargs,
        )

    return traits


def get_traits_value_plant(
    h5,
    monocots: bool = False,
    primary_name: str = "primary_multi_day",
    lateral_name: str = "lateral_3_nodes",
    root_width_tolerance: float = 0.02,
    n_line: int = 50,
    network_fraction: float = 2 / 3,
    write_csv: bool = False,
    csv_suffix: str = ".traits.csv",
) -> Tuple[Dict, pd.DataFrame, str]:
    """Get detailed SLEAP traits for every frame of a plant, based on the graph.

    Args:
        h5: The h5 file representing the plant image series.
        monocots: A boolean value indicating whether the plant is a monocot (True)
            or a dicot (False) (default).
        primary_name: Name of the primary root predictions. The predictions file is
            expected to be named `"{h5_path}.{primary_name}.predictions.slp"`.
        lateral_name: Name of the lateral root predictions. The predictions file is
            expected to be named `"{h5_path}.{lateral_name}.predictions.slp"`.
        root_width_tolerance: The difference in the projection norm between
            the right and left side of the root.
        n_line: The number of scan lines. Use np.nan for no interaction.
        network_fraction: The length found in the lower fraction value of the network.
        write_csv: A boolean value. If True, it writes per plant detailed
            CSVs with traits for every instance on every frame.
        csv_suffix: If write_csv=True, the CSV file will be saved with the
            h5 path + csv_suffix.

    Returns:
        A tuple containing a dictionary and a DataFrame with all traits per plant,
            and the plant name. The Dataframe has root traits per instance and frame
            where each row corresponds to a frame in the H5 file. The plant_name is
            given by the h5 file.
    """
    plant = Series.load(h5, primary_name=primary_name, lateral_name=lateral_name)
    plant_name = plant.series_name
    # get number of frames per plant
    n_frame = len(plant)

    data_plant = []
    # get traits for each frames in a row
    for frame in range(n_frame):
        primary, lateral = plant[frame]

        gt_instances_pr = primary.user_instances + primary.unused_predictions
        gt_instances_lr = lateral.user_instances + lateral.unused_predictions

        if len(gt_instances_lr) == 0:
            lateral_pts = np.array([[(np.nan, np.nan), (np.nan, np.nan)]])
        else:
            lateral_pts = np.stack([inst.numpy() for inst in gt_instances_lr], axis=0)

        if len(gt_instances_pr) == 0:
            primary_pts = np.array([[(np.nan, np.nan), (np.nan, np.nan)]])
        else:
            primary_pts = np.stack([inst.numpy() for inst in gt_instances_pr], axis=0)

        data = get_traits_value_frame(
            primary_pts,
            lateral_pts,
            root_width_tolerance,
            n_line,
            network_fraction,
            monocots,
        )

        data["plant_name"] = plant_name
        data["frame_idx"] = frame
        data_plant.append(data)
    data_plant_df = pd.DataFrame(data_plant)

    # reorganize the column position
    column_names = data_plant_df.columns.tolist()
    column_names = [column_names[-2]] + [column_names[-1]] + column_names[:-2]
    data_plant_df = data_plant_df[column_names]

    # convert the data in scalar column to the value without []
    columns_to_convert = data_plant_df.columns[
        data_plant_df.apply(
            lambda x: all(
                isinstance(val, np.ndarray) and val.shape == (1,) for val in x
            )
        )
    ]
    data_plant_df[columns_to_convert] = data_plant_df[columns_to_convert].apply(
        lambda x: x.apply(lambda val: val[0])
    )

    if write_csv:
        csv_name = Path(h5).with_suffix(f"{csv_suffix}")
        data_plant_df.to_csv(csv_name, index=False)
    return data_plant, data_plant_df, plant_name


def get_traits_value_plant_summary(
    h5,
    monocots: bool = False,
    primary_name: str = "longest_3do_6nodes",
    lateral_name: str = "main_3do_6nodes",
    root_width_tolerance: float = 0.02,
    n_line: int = 50,
    network_fraction: float = 2 / 3,
    write_csv: bool = False,
    csv_suffix: str = ".traits.csv",
    write_summary_csv: bool = False,
    summary_csv_suffix: str = ".summary_traits.csv",
) -> pd.DataFrame:
    """Get summary statistics of SLEAP traits per plant based on graph.

    Args:
        h5: The h5 file representing the plant image series.
        monocots: A boolean value indicating whether the plant is a monocot (True)
            or a dicot (False) (default).
        primary_name: Name of the primary root predictions. The predictions file is
            expected to be named `"{h5_path}.{primary_name}.predictions.slp"`.
        lateral_name: Name of the lateral root predictions. The predictions file is
            expected to be named `"{h5_path}.{lateral_name}.predictions.slp"`.
        root_width_tolerance: The difference in the projection norm between
            the right and left side of the root.
        n_line: The number of scan lines. Use np.nan for no interaction.
        network_fraction: The length found in the lower fraction value of the network.
        write_csv: A boolean value. If True, it writes per plant detailed
            CSVs with traits for every instance on every frame.
        csv_suffix: If write_csv=True, the CSV file will be saved with the name
            h5 path + csv_suffix.
        write_summary_csv: Boolean value, where true is write summarized csv file.
        summary_csv_suffix: If write_summary_csv=True, the CSV file with the summary
            statistics per plant will be saved with the name
            h5 path + summary_csv_suffix.

    Return:
        A DataFrame with summary statistics of all traits per plant.
    """
    data_plant, data_plant_df, plant_name = get_traits_value_plant(
        h5,
        monocots,
        primary_name,
        lateral_name,
        root_width_tolerance,
        n_line,
        network_fraction,
        write_csv,
        csv_suffix,
    )

    # get summarized non-scalar traits per frame
    data_plant_frame_summary = []
    data_plant_frame_summary_non_scalar = {}

    for i in range(len(NON_SCALAR_TRAITS)):
        trait = data_plant_df[NON_SCALAR_TRAITS[i]]

        if not trait.isna().all():
            data_plant_frame_summary_non_scalar[
                NON_SCALAR_TRAITS[i] + "_fmin"
            ] = trait.apply(
                lambda x: x
                if isinstance(x, (np.floating, float, np.integer, int))
                else (np.nanmin(x) if len(x) > 0 else np.nan)
            )
            data_plant_frame_summary_non_scalar[
                NON_SCALAR_TRAITS[i] + "_fmax"
            ] = trait.apply(
                lambda x: x
                if isinstance(x, (np.floating, float, np.integer, int))
                else (np.nanmax(x) if len(x) > 0 else np.nan)
            )
            data_plant_frame_summary_non_scalar[
                NON_SCALAR_TRAITS[i] + "_fmean"
            ] = trait.apply(
                lambda x: x
                if isinstance(x, (np.floating, float, np.integer, int))
                else (np.nanmean(x) if len(x) > 0 else np.nan)
            )
            data_plant_frame_summary_non_scalar[
                NON_SCALAR_TRAITS[i] + "_fmedian"
            ] = trait.apply(
                lambda x: x
                if isinstance(x, (np.floating, float, np.integer, int))
                else (np.nanmedian(x) if len(x) > 0 else np.nan)
            )
            data_plant_frame_summary_non_scalar[
                NON_SCALAR_TRAITS[i] + "_fstd"
            ] = trait.apply(
                lambda x: x
                if isinstance(x, (np.floating, float, np.integer, int))
                else (np.nanstd(x) if len(x) > 0 else np.nan)
            )
            data_plant_frame_summary_non_scalar[
                NON_SCALAR_TRAITS[i] + "_fprc5"
            ] = trait.apply(
                lambda x: x
                if isinstance(x, (np.floating, float, np.integer, int))
                else (np.nan if np.isnan(x).all() else np.percentile(x[~pd.isna(x)], 5))
            )
            data_plant_frame_summary_non_scalar[
                NON_SCALAR_TRAITS[i] + "_fprc25"
            ] = trait.apply(
                lambda x: x
                if isinstance(x, (np.floating, float, np.integer, int))
                else (
                    np.nan if np.isnan(x).all() else np.percentile(x[~pd.isna(x)], 25)
                )
            )
            data_plant_frame_summary_non_scalar[
                NON_SCALAR_TRAITS[i] + "_fprc75"
            ] = trait.apply(
                lambda x: x
                if isinstance(x, (np.floating, float, np.integer, int))
                else (
                    np.nan if np.isnan(x).all() else np.percentile(x[~pd.isna(x)], 75)
                )
            )
            data_plant_frame_summary_non_scalar[
                NON_SCALAR_TRAITS[i] + "_fprc95"
            ] = trait.apply(
                lambda x: x
                if isinstance(x, (np.floating, float, np.integer, int))
                else (
                    np.nan if np.isnan(x).all() else np.percentile(x[~pd.isna(x)], 95)
                )
            )
        else:
            data_plant_frame_summary_non_scalar[NON_SCALAR_TRAITS[i] + "_fmin"] = np.nan
            data_plant_frame_summary_non_scalar[NON_SCALAR_TRAITS[i] + "_fmax"] = np.nan
            data_plant_frame_summary_non_scalar[
                NON_SCALAR_TRAITS[i] + "_fmean"
            ] = np.nan
            data_plant_frame_summary_non_scalar[
                NON_SCALAR_TRAITS[i] + "_fmedian"
            ] = np.nan
            data_plant_frame_summary_non_scalar[NON_SCALAR_TRAITS[i] + "_fstd"] = np.nan
            data_plant_frame_summary_non_scalar[
                NON_SCALAR_TRAITS[i] + "_fprc5"
            ] = np.nan
            data_plant_frame_summary_non_scalar[
                NON_SCALAR_TRAITS[i] + "_fprc25"
            ] = np.nan
            data_plant_frame_summary_non_scalar[
                NON_SCALAR_TRAITS[i] + "_fprc75"
            ] = np.nan
            data_plant_frame_summary_non_scalar[
                NON_SCALAR_TRAITS[i] + "_fprc95"
            ] = np.nan

    # get summarized scalar traits per plant
    column_names = data_plant_df.columns.tolist()
    data_plant_frame_summary = {}
    for i in range(len(SCALAR_TRAITS)):
        if SCALAR_TRAITS[i] in column_names:
            trait = data_plant_df[SCALAR_TRAITS[i]]
            if trait.shape[0] > 0:
                if not (
                    isinstance(trait[0], (np.floating, float))
                    or isinstance(trait[0], (np.integer, int))
                ):
                    values = np.array([element[0] for element in trait])
                    trait = values
            trait = trait.astype(float)
            trait = np.reshape(trait, (len(trait), 1))
            (
                trait_min,
                trait_max,
                trait_mean,
                trait_median,
                trait_std,
                trait_prc5,
                trait_prc25,
                trait_prc75,
                trait_prc95,
            ) = get_summary(trait)

            data_plant_frame_summary[SCALAR_TRAITS[i] + "_min"] = trait_min
            data_plant_frame_summary[SCALAR_TRAITS[i] + "_max"] = trait_max
            data_plant_frame_summary[SCALAR_TRAITS[i] + "_mean"] = trait_mean
            data_plant_frame_summary[SCALAR_TRAITS[i] + "_median"] = trait_median
            data_plant_frame_summary[SCALAR_TRAITS[i] + "_std"] = trait_std
            data_plant_frame_summary[SCALAR_TRAITS[i] + "_prc5"] = trait_prc5
            data_plant_frame_summary[SCALAR_TRAITS[i] + "_prc25"] = trait_prc25
            data_plant_frame_summary[SCALAR_TRAITS[i] + "_prc75"] = trait_prc75
            data_plant_frame_summary[SCALAR_TRAITS[i] + "_prc95"] = trait_prc95

    # append the summarized non-scalar traits per plant
    data_plant_frame_summary_key = list(data_plant_frame_summary_non_scalar.keys())
    for j in range(len(data_plant_frame_summary_non_scalar)):
        trait = data_plant_frame_summary_non_scalar[data_plant_frame_summary_key[j]]
        (
            trait_min,
            trait_max,
            trait_mean,
            trait_median,
            trait_std,
            trait_prc5,
            trait_prc25,
            trait_prc75,
            trait_prc95,
        ) = get_summary(trait)

        data_plant_frame_summary[data_plant_frame_summary_key[j] + "_min"] = trait_min
        data_plant_frame_summary[data_plant_frame_summary_key[j] + "_max"] = trait_max
        data_plant_frame_summary[data_plant_frame_summary_key[j] + "_mean"] = trait_mean
        data_plant_frame_summary[
            data_plant_frame_summary_key[j] + "_median"
        ] = trait_median
        data_plant_frame_summary[data_plant_frame_summary_key[j] + "_std"] = trait_std
        data_plant_frame_summary[data_plant_frame_summary_key[j] + "_prc5"] = trait_prc5
        data_plant_frame_summary[
            data_plant_frame_summary_key[j] + "_prc25"
        ] = trait_prc25
        data_plant_frame_summary[
            data_plant_frame_summary_key[j] + "_prc75"
        ] = trait_prc75
        data_plant_frame_summary[
            data_plant_frame_summary_key[j] + "_prc95"
        ] = trait_prc95
    data_plant_frame_summary["plant_name"] = [plant_name]
    data_plant_frame_summary_df = pd.DataFrame(data_plant_frame_summary)

    # reorganize the column position
    column_names = data_plant_frame_summary_df.columns.tolist()
    column_names = [column_names[-1]] + column_names[:-1]
    data_plant_frame_summary_df = data_plant_frame_summary_df[column_names]

    if write_summary_csv:
        summary_csv_name = Path(h5).with_suffix(f"{summary_csv_suffix}")
        data_plant_frame_summary_df.to_csv(summary_csv_name, index=False)
    return data_plant_frame_summary_df


def get_all_plants_traits(
    data_folders: List[str],
    primary_name: str,
    lateral_name: str,
    root_width_tolerance: float = 0.02,
    n_line: int = 50,
    network_fraction: Fraction = Fraction(2, 3),
    write_per_plant_details: bool = False,
    per_plant_details_csv_suffix: str = ".traits.csv",
    write_per_plant_summary: bool = False,
    per_plant_summary_csv_suffix: str = ".summary_traits.csv",
    monocots: bool = False,
    all_plants_csv_name: str = "all_plants_traits.csv",
) -> pd.DataFrame:
    """Get a DataFrame with summary traits from all plants in the given data folders.

    Args:
        h5: The h5 file representing the plant image series.
        monocots: A boolean value indicating whether the plant is a monocot (True)
            or a dicot (False) (default).
        primary_name: Name of the primary root predictions. The predictions file is
            expected to be named `"{h5_path}.{primary_name}.predictions.slp"`.
        lateral_name: Name of the lateral root predictions. The predictions file is
            expected to be named `"{h5_path}.{lateral_name}.predictions.slp"`.
        root_width_tolerance: The difference in the projection norm between
            the right and left side of the root.
        n_line: The number of scan lines. Use np.nan for no interaction.
        network_fraction: The length found in the lower fraction value of the network.
        write_per_plant_details: A boolean value. If True, it writes per plant detailed
            CSVs with traits for every instance.
        per_plant_details_csv_suffix: If write_csv=True, the CSV file will be saved
            with the name h5 path + csv_suffix.
        write_per_plant_summary: A boolean value. If True, it writes per plant summary
            CSVs.
        per_plant_summary_csv_suffix: If write_summary_csv=True, the CSV file with the
            summary statistics per plant will be saved with the name
            h5 path + summary_csv_suffix.
        all_plants_csv_name: The name of the output CSV file containing all plants'
            summary traits.

    Returns:
        A pandas DataFrame with summary root traits for all plants in the data folders.
        Each row is a sample.
    """
    h5_series = find_all_series(data_folders)

    all_traits = []
    for h5 in h5_series:
        plant_traits = get_traits_value_plant_summary(
            h5,
            monocots=monocots,
            primary_name=primary_name,
            lateral_name=lateral_name,
            root_width_tolerance=root_width_tolerance,
            n_line=n_line,
            network_fraction=network_fraction,
            write_csv=write_per_plant_details,
            csv_suffix=per_plant_details_csv_suffix,
            write_summary_csv=write_per_plant_summary,
            summary_csv_suffix=per_plant_summary_csv_suffix,
        )
        plant_traits["path"] = h5
        all_traits.append(plant_traits)

    all_traits_df = pd.concat(all_traits, ignore_index=True)

    all_traits_df.to_csv(all_plants_csv_name, index=False)
    return all_traits_df
