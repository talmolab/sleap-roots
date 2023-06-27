"""Extract traits based on the networkx graph."""

import numpy as np
import pandas as pd
import os
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
from sleap_roots.points import get_all_pts_array, get_all_pts
from sleap_roots.scanline import (
    count_scanline_intersections,
    get_scanline_first_ind,
    get_scanline_last_ind,
)
from sleap_roots.series import Series
from sleap_roots.summary import get_summary
from sleap_roots.tips import get_tips, get_tip_xs, get_tip_ys
from typing import Dict, Tuple
import warnings


SCALAR_TRAITS = (
    "primary_angle_proximal",
    "primary_angle_distal",
    "primary_length",
    "primary_base_tip_dist",
    "primary_depth",
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
    "stem_widths",
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


def get_traits_value_frame(
    primary_pts: np.ndarray,
    lateral_pts: np.ndarray,
    pts_all_array: np.ndarray,
    pts_all_list: list,
    stem_width_tolerance: float = 0.02,
    n_line: int = 50,
    network_fraction: float = 2 / 3,
    monocots: bool = False,
) -> Dict:
    """Get SLEAP traits per frame based on graph.

    Args:
        primary_pts: primary points
        lateral_pts: lateral points
        pts_all_array: all points in array format
        pts_all_list: all points in list format
        stem_width_tolerance: difference in projection norm between right and left side.
        n_line: number of scan lines, np.nan for no interaction.
        network_fraction: length found in the lower fration value of the network.
        monocots: Boolean value, where false is dicot (default), true is rice.

    Return:
        A dictionary with all traits per frame.
    """
    trait_map = {
        # get_bases(pts: np.ndarray,monocots) -> np.ndarray
        "primary_base_pt": (get_bases, [primary_pts, monocots]),
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
        # get_root_pair_widths_projections(lateral_pts, primary_pts, tolerance,monocots)
        "stem_widths": (
            get_root_pair_widths_projections,
            [lateral_pts, primary_pts, stem_width_tolerance, monocots],
        ),
        # get_convhull_features(pts: Union[np.ndarray, ConvexHull]) -> Tuple[float, float, float, float]
        "convex_hull": (get_convhull_features, [pts_all_array]),
        # get_lateral_count(pts: np.ndarray)
        "lateral_count": (get_lateral_count, [lateral_pts]),
        # # get_root_angle(pts: np.ndarray, proximal=True, base_ind=0) -> np.ndarray
        "lateral_angles_proximal": (get_root_angle, [lateral_pts, True, 0]),
        "lateral_angles_distal": (get_root_angle, [lateral_pts, False, 0]),
        # get_root_lengths(pts: np.ndarray) -> np.ndarray
        "lateral_lengths": (get_root_lengths, [lateral_pts]),
        # get_bases(pts: np.ndarray,monocots) -> np.ndarray
        "lateral_base_pts": (get_bases, [lateral_pts, monocots]),
        # get_tips(pts)
        "lateral_tip_pts": (get_tips, [lateral_pts]),
        # get_base_ys(pts: np.ndarray) -> np.ndarray
        # or just based on primary_base_pt, but the primary_base_pt trait must generate before
        # "primary_base_pt_y": (get_pt_ys, [data["primary_base_pt"]]),
        "primary_base_pt_y": (get_base_ys, [primary_pts]),
        # get_base_ct_density(primary_pts, lateral_pts)
        "base_ct_density": (get_base_ct_density, [primary_pts, lateral_pts]),
        # get_network_solidity(primary_pts: np.ndarray, lateral_pts: np.ndarray, pts_all_array: np.ndarray, monocots: bool = False,) -> float
        "network_solidity": (
            get_network_solidity,
            [primary_pts, lateral_pts, pts_all_array, monocots],
        ),
        # get_network_distribution_ratio(primary_pts: np.ndarray,lateral_pts: np.ndarray,pts_all_array: np.ndarray,fraction: float = 2 / 3, monocots: bool = False) -> float:
        "network_distribution_ratio": (
            get_network_distribution_ratio,
            [primary_pts, lateral_pts, pts_all_array, network_fraction, monocots],
        ),
        # get_network_distribution(primary_pts: np.ndarray,lateral_pts: np.ndarray,pts_all_array: np.ndarray,fraction: float = 2 / 3, monocots: bool = False) -> float:
        "network_length_lower": (
            get_network_distribution,
            [primary_pts, lateral_pts, pts_all_array, network_fraction, monocots],
        ),
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
        # count_scanline_intersections(primary_pts: np.ndarray,lateral_pts: np.ndarray,depth: int = 1080,width: int = 2048,n_line: int = 50,monocots: bool = False,) -> np.ndarray
        "scanline_intersection_counts": (
            count_scanline_intersections,
            [primary_pts, lateral_pts, 1080, 2048, 50, monocots],
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
        # get_scanline_last_ind(primary_pts: np.ndarray,lateral_pts: np.ndarray,depth: int = 1080, width: int = 2048, n_line: int = 50, monocots: bool = False)
        "scanline_last_ind": (
            get_scanline_last_ind,
            [primary_pts, lateral_pts, 1080, 2048, n_line, monocots],
        ),
        # get_scanline_first_ind(primary_pts: np.ndarray,lateral_pts: np.ndarray,depth: int = 1080, width: int = 2048, n_line: int = 50, monocots: bool = False)
        "scanline_first_ind": (
            get_scanline_first_ind,
            [primary_pts, lateral_pts, 1080, 2048, n_line, monocots],
        ),
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
        fn, inputs = trait_map[trait_name]
        fn_outputs = fn(*[input_trait for input_trait in inputs])
        if type(fn_outputs) == tuple:
            fn_outputs = np.array(fn_outputs).reshape((1, -1))
        if isinstance(fn_outputs, (np.floating, float)) or isinstance(
            fn_outputs, (np.integer, int)
        ):
            fn_outputs = np.array(fn_outputs)[np.newaxis]
        data[trait_name] = fn_outputs
    return data


def get_traits_value_plant(
    h5,
    monocots: bool = False,
    primary_name: str = "primary_multi_day",
    lateral_name: str = "lateral_3_nodes",
    stem_width_tolerance: float = 0.02,
    n_line: int = 50,
    network_fraction: float = 2 / 3,
    write_csv: bool = False,
    csv_name: str = "plant_original_traits.csv",
) -> Tuple[Dict, pd.DataFrame]:
    """Get SLEAP traits per plant based on graph.

    Args:
        h5: h5 file, plant image series.
        monocots: Boolean value, where false is dicot (default), true is rice.
        primary_name: primary model name.
        lateral_name: lateral model name.
        stem_width_tolerance: difference in projection norm between right and left side.
        n_line: number of scan lines, np.nan for no interaction.
        network_fraction: length found in the lower fration value of the network.
        write_csv: Boolean value, where true is write csv file.
        csv_name: saved csv file name.

    Return:
        Tuple of a dictionary and a DataFrame with all traits per plant.
    """
    plant = Series.load(h5, primary_name=primary_name, lateral_name=lateral_name)
    plant_name = plant.series_name
    # get nymber of frames per plant
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

        pts_all_array = get_all_pts_array(plant=plant, frame=frame, monocots=False)
        if len(pts_all_array) == 0:
            pts_all_array = np.array([[(np.nan, np.nan), (np.nan, np.nan)]])
        pts_all_list = []

        if get_root_lengths(primary_pts).shape[0] > 0 and not len(gt_instances_pr) == 0:
            max_length_idx = np.nanargmax(get_root_lengths(primary_pts))
            long_primary_pts = primary_pts[max_length_idx]
            primary_pts = np.reshape(
                long_primary_pts,
                (1, long_primary_pts.shape[0], long_primary_pts.shape[1]),
            )
        else:
            # if no primary root, just give two nan points
            primary_pts = np.array([[(np.nan, np.nan), (np.nan, np.nan)]])

        data = get_traits_value_frame(
            primary_pts,
            lateral_pts,
            pts_all_array,
            pts_all_list,
            stem_width_tolerance,
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
        csv_name = "plant_original_traits_" + plant_name + ".csv"
        data_plant_df.to_csv(csv_name, index=False)
    return data_plant, data_plant_df


def get_traits_value_plant_summary(
    h5,
    monocots: bool = False,
    primary_name: str = "longest_3do_6nodes",
    lateral_name: str = "main_3do_6nodes",
    stem_width_tolerance: float = 0.02,
    n_line: int = 50,
    network_fraction: float = 2 / 3,
    write_csv: bool = False,
    csv_name: str = "plant_original_traits.csv",
    write_summary_csv: bool = False,
    summary_csv_name: str = "plant_summary_traits.csv",
) -> pd.DataFrame:
    """Get summarized SLEAP traits per plant based on graph.

    Args:
        h5: h5 file, plant image series.
        monocots: Boolean value, where false is dicot (default), true is rice.
        primary_name: primary model name.
        lateral_name: lateral model name.
        stem_width_tolerance: difference in projection norm between right and left side.
        n_line: number of scan lines, np.nan for no interaction.
        network_fraction: length found in the lower fration value of the network.
        write_csv: Boolean value, where true is write csv file.
        csv_name: saved csv file name.
        write_summary_csv: Boolean value, where true is write summarized csv file.
        summary_csv_name: saved summarized csv file name.

    Return:
        A DataFrame with all summarized traits per plant.
    """
    data_plant, data_plant_df = get_traits_value_plant(
        h5,
        monocots,
        primary_name,
        lateral_name,
        stem_width_tolerance,
        n_line,
        network_fraction,
        write_csv,
        csv_name,
    )

    # get summarized non-scalar traits per frame
    data_plant_frame_summary = []
    data_plant_frame_summary_non_scalar = {}

    for i in range(len(NON_SCALAR_TRAITS)):
        trait = data_plant_df[NON_SCALAR_TRAITS[i]]

        data_plant_frame_summary_non_scalar[
            NON_SCALAR_TRAITS[i] + "_fmin"
        ] = trait.apply(lambda x: np.nanmin(x) if (len(x) > 0) else np.nan)
        data_plant_frame_summary_non_scalar[
            NON_SCALAR_TRAITS[i] + "_fmax"
        ] = trait.apply(lambda x: np.nanmax(x) if len(x) > 0 else np.nan)
        data_plant_frame_summary_non_scalar[
            NON_SCALAR_TRAITS[i] + "_fmean"
        ] = trait.apply(lambda x: np.nanmean(x) if len(x) > 0 else np.nan)
        data_plant_frame_summary_non_scalar[
            NON_SCALAR_TRAITS[i] + "_fmedian"
        ] = trait.apply(lambda x: np.nanmedian(x) if len(x) > 0 else np.nan)
        data_plant_frame_summary_non_scalar[
            NON_SCALAR_TRAITS[i] + "_fstd"
        ] = trait.apply(lambda x: np.nanstd(x) if len(x) > 0 else np.nan)
        data_plant_frame_summary_non_scalar[
            NON_SCALAR_TRAITS[i] + "_fprc5"
        ] = trait.apply(
            lambda x: np.percentile(x[~pd.isna(x)], 5)
            if len(x[~pd.isna(x)]) > 0
            else np.nan
        )
        data_plant_frame_summary_non_scalar[
            NON_SCALAR_TRAITS[i] + "_fprc25"
        ] = trait.apply(
            lambda x: np.percentile(x[~pd.isna(x)], 25)
            if len(x[~pd.isna(x)]) > 0
            else np.nan
        )
        data_plant_frame_summary_non_scalar[
            NON_SCALAR_TRAITS[i] + "_fprc75"
        ] = trait.apply(
            lambda x: np.percentile(x[~pd.isna(x)], 75)
            if len(x[~pd.isna(x)]) > 0
            else np.nan
        )
        data_plant_frame_summary_non_scalar[
            NON_SCALAR_TRAITS[i] + "_fprc95"
        ] = trait.apply(
            lambda x: np.percentile(x[~pd.isna(x)], 95)
            if len(x[~pd.isna(x)]) > 0
            else np.nan
        )

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
    data_plant_frame_summary["plant_name"] = [os.path.splitext(h5)[0]]
    data_plant_frame_summary_df = pd.DataFrame(data_plant_frame_summary)

    # reorganize the column position
    column_names = data_plant_frame_summary_df.columns.tolist()
    column_names = [column_names[-1]] + column_names[:-1]
    data_plant_frame_summary_df = data_plant_frame_summary_df[column_names]

    if write_summary_csv:
        data_plant_frame_summary_df.to_csv(summary_csv_name, index=False)
    return data_plant_frame_summary_df
