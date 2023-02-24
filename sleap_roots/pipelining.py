"""Extract plant traits based on SLEAP prediction."""

import numpy as np
import os
import pandas as pd
from sleap_roots.series import Series
from sleap_roots.angle import get_root_angle
from sleap_roots.bases import (
    get_root_lengths,
    get_bases,
    get_bases_percentile,
    get_base_tip_dist,
    get_root_pair_widths_projections,
)
from sleap_roots.convhull import get_convhull_features
from sleap_roots.ellipse import fit_ellipse
from sleap_roots.networklength import get_network_distribution
from sleap_roots.networklength import get_network_distribution_ratio
from sleap_roots.networklength import get_network_solidity
from sleap_roots.networklength import get_network_width_depth_ratio
from sleap_roots.scanline import count_scanline_intersections
from sleap_roots.tips import get_tips, get_tips_percentile
from typing import Tuple


def get_statistics(
    traits_array: np.ndarray,
) -> Tuple[float, float, float, float, float]:
    """Get 5 statistics of the traits.

    Args:
        traits_array: the traits array that need to calculate 5 statistics.

    Returns:
        A Tuple of 5 statistics: max, min, mean, std, median.
    """
    trait_max = np.nanmax(traits_array)
    trait_min = np.nanmin(traits_array)
    trait_mean = np.nanmean(traits_array)
    trait_std = np.nanstd(traits_array)
    trait_median = np.nanmedian(traits_array)
    return trait_max, trait_min, trait_mean, trait_std, trait_median


def get_pts_pr_lr(h5, rice=False, frame=0, primaryroot=True):
    """Get all predicted points of primary/lateral root (# instance, # nodes, 2).

    Args:
        h5: h5 file, SLEAP prediction.
        rice: Boolean value, where true is rice (default), false is other species.
        frame: the frame index, default value is 0, i.e., first frame.
        primaryroot: Boolean value, where true is primary root, false is lateral root.

    Returns:
        An array of points location in shape of (# points, 2).
    """
    if rice:
        series = Series.load(
            h5, primary_name="longest_3do_6nodes", lateral_name="main_3do_6nodes"
        )
    else:
        series = Series.load(
            h5, primary_name="primary_multi_day", lateral_name="lateral_3_nodes"
        )
    primary, lateral = series[frame]
    if primaryroot:
        return primary.numpy()
    else:
        return lateral.numpy()


def get_pts_all(h5, rice=False, frame=0) -> np.ndarray:
    """Get all predicted points in shape of (# points, 2).

    Args:
        h5: h5 file, SLEAP prediction.
        rice: Boolean value, where true is rice (default), false is other species.
        frame: the frame index, default value is 0, i.e., first frame.

    Returns:
        An array of points location in shape of (# points, 2).
    """
    if rice:
        pts_lr = get_pts_pr_lr(h5, rice=True, frame=frame, primaryroot=False)
        pts_all = pts_lr.reshape(-1, 2)
    else:
        pts_pr = get_pts_pr_lr(h5, rice=False, frame=frame, primaryroot=True)
        pts_lr = get_pts_pr_lr(h5, rice=False, frame=frame, primaryroot=False)
        primary_points = pts_pr.reshape(-1, 2)
        lateral_points = pts_lr.reshape(-1, 2)
        pts_all = np.concatenate((primary_points, lateral_points), axis=0)
    return pts_all


def get_traits_frame(
    path,
    h5,
    rice=False,
    frame=0,
    tolerance=0.02,
    pctl_base=[25, 75],
    pctl_tip=[25, 75],
    fraction: float = 2 / 3,
    depth: int = 1080,
    width: int = 2048,
    n_line: int = 50,
) -> Tuple:
    """Extract frame-based traits based on SLEAP prediction *.h5 and *.slp files.

    Args:
        path: the work directory where store h5 and slp files.
        h5: h5 file, SLEAP prediction.
        rice: Boolean value, where true is rice (default), false is other species.
        frame: the frame index, default value is 0, i.e., first frame.
        tolerance: difference in projection norm between the right and left side.
        pctl_base: the percentile of all base points to calculate y-axis.
        pctl_tip: the percentile of all tip points to calculate y-axis.
        fraction: the network length found in the lower fration value of the network.
        depth: the depth of cylinder, or number of rows of the image.
        width: the width of cylinder, or number of columns of the image.
        n_line: number of scan lines.

    Returns:
        A DataFrame with a single row containing columns with traits:
            - plant_name: record of plant name
            - frame: frame index, 72 frames in total, 0 to 71
            - primary_angles_proximal: primary root angle with base node and proximal
            node
            - primary_angles_distal: primary root angle with base node and distal node
            - lateral_angles_proximal_max: maximum lateral root angle with proximal node
            among all lateral roots within one frame
            - lateral_angles_proximal_min: minimum lateral root angle with proximal node
            among all lateral roots within one frame
            - lateral_angles_proximal_mean: mean lateral root angle with proximal node
            among all lateral roots within one frame
            - lateral_angles_proximal_std: standard deviation of lateral root angle with
            proximal node among all lateral roots within one frame
            - lateral_angles_proximal_median: median lateral root angle with proximal
            node among all lateral roots within one frame
            - lateral_angles_distal_max: maximum lateral root angle with distal node
            among all lateral roots within one frame
            - lateral_angles_distal_min: minimum lateral root angle with distal node
            among all lateral roots within one frame
            - lateral_angles_distal_mean: mean lateral root angle with distal node
            among all lateral roots within one frame
            - lateral_angles_distal_std: standard deviation of lateral root angle with
            distal node among all lateral roots within one frame
            - lateral_angles_distal_median: median lateral root angle with distal
            node among all lateral roots within one frame
            - primary_length: length of primary root
            - primary_base_tip_dist: primary root base to tip distance in y axis
            - primary_depth: primary root tip depth
            - grav_index: primary root gravitropism index
            - lateral_length_total: total lateral root length
            - lateral_length_max: maximum length of lateral roots
            - lateral_length_min: minimum length of lateral roots
            - lateral_length_mean: mean length of lateral roots
            - lateral_length_std: standard deviation length of lateral roots
            - lateral_length_median: median length of lateral roots
            - stem_widths_max: maximum stem width based on paired lateral root bases
            - stem_widths_min: minimum stem width based on paired lateral root bases
            - stem_widths_mean: mean stem width based on paired lateral root bases
            - stem_widths_std: standard deviation stem width based on paired lateral
            root bases
            - stem_widths_median: median stem width based on paired lateral root bases
            - base_ct: number of lateral root bases
            - base_top: top base point in y-axis
            - base_deep: deepest base point in y-axis
            - base_length: top and deepest bases distance y-axis
            - base_length_ratio: ratio of top-deep length to primary root length
            - base_meanx: mean value of x axis of all base points
            - base_meany: mean value of y axis of all base points
            - base_medianx: median value of x axis of all base points
            - base_mediany: median value of y axis of all base points
            - base_stdx: standard deviation of x axis of all base points
            - base_stdy: standard deviation of y axis of all base points
            - pr_tip_depth: tip of primary root in y axis
            - base_median_ratio: ratio of median value in all base points to tip of primary root in y axis
            - base_ct_density: number of base points to maximum primary root length
            - tip_ct: number of lateral root tips
            - tip_top: top tip point in y-axis
            - tip_deep: deepest tip point in y-axis
            - tip_meanx: mean value of x axis of all tip points
            - tip_meany: mean value of y axis of all tip points
            - tip_medianx: median value of x axis of all tip points
            - tip_mediany: median value of y axis of all tip points
            - tip_stdx: standard deviation of x axis of all tip points
            - tip_stdy: standard deviation of y axis of all tip points
            - tip_sdx_sdys: ratio of tip_stdx to tip_stdy
            - tip_median_ratio: ratio of median value in all tip points to tip of primary root in y axis
            - conv_perimeters: perimeter of convex hull
            - conv_areas: area of convex hull
            - conv_longest_dists: longest distance of convex hull
            - conv_shortest_dists: shortest of convex hull
            - conv_median_dists: median distance of convex hull
            - conv_max_widths: maximum width of convex hull
            - conv_max_heights: maxinum height of convex hull
            - ellipse_a: semi-major axis length of the fitted ellipse
            - ellipse_b: semi-minor axis length of the fitted ellipse
            - ellipse_ratio: ratio of the minor to major lengths
            - network_width_depth_ratio: width to depth ratio of bounding box for root
            network
            - network_solidity: total network length divided by the network convex area
            - network_length_lower_network: root length in the lower fraction
            - network_distribution_ratio: ratio of the root length in the lower fraction
            over all root length
            - count_scanline_interaction_max: maximum interaction of scanline and roots
            - count_scanline_interaction_min: minimum interaction of scanline and roots
            - count_scanline_interaction_mean: mean interaction of scanline and roots
            - count_scanline_interaction_std: standard deviation interaction of scanline
            and roots
            - count_scanline_interaction_median: median interaction of scanline and
            roots
            - scanline_start: the scanline index where start interaction with roots
            - scanline_end: the scanline index where end interaction with roots
            - base_pctl: lateral root bases percentile in y-axis, # columns = # pctl
            - tip_pctl: lateral root tips percentile in y-axis, # columns = # pctl
    """
    # check whether 2 *.slp files exist
    plant_name = os.path.splitext(os.path.split(h5)[1])[0]
    slp_files = [
        file
        for file in os.listdir(path)
        if (file.endswith(".slp") and file.startswith(plant_name))
    ]

    if len(slp_files) < 2:
        return "Incomplete SLEAP prediction!"

    # get the primary and lateral roots points
    # choose the longest primary root if more than one
    pts_pr = get_pts_pr_lr(h5, rice=rice, frame=frame, primaryroot=True)
    max_length_idx = np.nanargmax(get_root_lengths(pts_pr))
    pts_pr = pts_pr[np.newaxis, max_length_idx]
    pts_lr = get_pts_pr_lr(h5, rice=rice, frame=frame, primaryroot=False)

    # calculate root angle related traits
    # priminary root angle with the proximal node and base node
    primary_angles_proximal = get_root_angle(pts_pr, proximal=True)
    # priminary root angle with the distal node and base node
    primary_angles_distal = get_root_angle(pts_pr, proximal=False)
    # lateral root angle with the proximal node and base node, 5 statistics
    lateral_angles_proximal = get_root_angle(pts_lr, proximal=True)
    (
        lateral_angles_proximal_max,
        lateral_angles_proximal_min,
        lateral_angles_proximal_mean,
        lateral_angles_proximal_std,
        lateral_angles_proximal_median,
    ) = get_statistics(lateral_angles_proximal)

    # lateral root angle with the distal node and base node, 5 statistics
    lateral_angles_distal = get_root_angle(pts_lr, proximal=False)
    (
        lateral_angles_distal_max,
        lateral_angles_distal_min,
        lateral_angles_distal_mean,
        lateral_angles_distal_std,
        lateral_angles_distal_median,
    ) = get_statistics(lateral_angles_distal)

    # calculate root length related traits
    # primary root length
    primary_length = get_root_lengths(pts_pr)
    # primary root base to tip distance in y axis
    primary_base_tip_dist = get_base_tip_dist(pts_pr)
    # primary root tip depth
    primary_depth = np.nanmax(pts_pr[:, :, 1])
    # primary root gravitropism index
    grav_index = (
        np.nanmax(primary_length) - np.nanmax(primary_base_tip_dist)
    ) / np.nanmax(primary_length)
    # lateral root length
    lateral_length = get_root_lengths(pts_lr)
    lateral_length_total = np.nansum(lateral_length)
    (
        lateral_length_max,
        lateral_length_min,
        lateral_length_mean,
        lateral_length_std,
        lateral_length_median,
    ) = get_statistics(lateral_length)

    # calculate root width using bases of lateral roots
    stem_widths = get_root_pair_widths_projections(pts_lr, pts_pr, tolerance=tolerance)
    if np.min([len(a) for a in stem_widths]) == 0:
        (
            stem_widths_max,
            stem_widths_min,
            stem_widths_mean,
            stem_widths_std,
            stem_widths_median,
        ) = (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )
    else:
        (
            stem_widths_max,
            stem_widths_min,
            stem_widths_mean,
            stem_widths_std,
            stem_widths_median,
        ) = get_statistics(stem_widths)

    # calculate base-based traits
    if len(pts_pr) == 0 or len(pts_lr) == 0 or np.isnan(get_bases(pts_lr)).all():
        base_ct = base_top = base_deep = base_length = base_length_ratio = np.nan
        base_meanx = (
            base_meany
        ) = base_medianx = base_mediany = base_stdx = base_stdy = np.nan
        pr_tip_depth = base_median_ratio = base_ct_density = base_pctl = np.nan
    else:
        lengths_primary = get_root_lengths(pts_pr)
        _base_pts = get_bases(pts_lr)
        valid_pts = _base_pts[~np.isnan(_base_pts[:, 0])]
        _base_sorted_inds = np.argsort(valid_pts[:, 1])
        _base_pts_sorted = valid_pts[_base_sorted_inds]

        # number of lateral root bases
        base_ct = len(_base_pts[~np.isnan(_base_pts[:, 0])])
        # Top and deepest base points y-axis
        base_top = _base_pts_sorted[0][1]
        base_deep = _base_pts_sorted[-1][1]
        # top and deepest bases distance y-axis
        base_length = np.linalg.norm(base_deep - base_top)
        # ratio of top-deep length to primary root length
        base_length_ratio = base_length / np.nanmax(lengths_primary)

        # mean value of x axis and y axis of all base points
        base_meanx = np.nanmean(_base_pts[:, 0])
        base_meany = np.nanmean(_base_pts[:, 1])
        # median value of x axis and y axis of all base points
        base_medianx = np.nanmedian(_base_pts[:, 0])
        base_mediany = np.nanmedian(_base_pts[:, 1])
        # standard deviation of x axis and y axis of all base points
        base_stdx = np.nanstd(_base_pts[:, 0])
        base_stdy = np.nanstd(_base_pts[:, 1])

        # tip of primary root in y axis
        pr_tip_depth = np.nanmax(pts_pr[:, :, 1])
        # ratio of median value in all base points to tip of primary root in y axis
        base_median_ratio = np.nanmedian(_base_pts[:, 1]) / pr_tip_depth
        # number of base points to maximum primary root length
        base_ct_density = base_ct / np.nanmax(lengths_primary)

        # lateral root bases percentile in y-axis
        base_pctl = get_bases_percentile(pts_lr, pctl_base)

    # calculate tip-based traits
    if len(pts_pr) == 0 or len(pts_lr) == 0 or np.isnan(get_tips(pts_lr)).all():
        tip_ct = tip_top = tip_deep = np.nan
        tip_meanx = tip_meany = tip_medianx = tip_mediany = tip_stdx = tip_stdy = np.nan
        tip_sdx_sdys = tip_median_ratio = tip_pctl = np.nan
    else:
        _tip_pts = get_tips(pts_lr)
        valid_pts = _tip_pts[~np.isnan(_tip_pts[:, 0])]
        _tip_sorted_inds = np.argsort(valid_pts[:, 1])
        _tip_pts_sorted = valid_pts[_tip_sorted_inds]

        # number of lateral root tips
        tip_ct = len(_tip_pts[~np.isnan(_tip_pts[:, 0])])
        # Top and deepest tip points y-axis
        tip_top = _tip_pts_sorted[0][1]
        tip_deep = _tip_pts_sorted[-1][1]

        # mean value of x axis and y axis of all tip points
        tip_meanx = np.nanmean(_tip_pts[:, 0])
        tip_meany = np.nanmean(_tip_pts[:, 1])
        # median value of x axis and y axis of all tip points
        tip_medianx = np.nanmedian(_tip_pts[:, 0])
        tip_mediany = np.nanmedian(_tip_pts[:, 1])
        # standard deviation of x axis and y axis of all tip points
        tip_stdx = np.nanstd(_tip_pts[:, 0])
        tip_stdy = np.nanstd(_tip_pts[:, 1])
        # ratio of tip_stdx to tip_stdy
        tip_sdx_sdys = tip_stdx / tip_stdy

        # tip of primary root in y axis
        pr_tip_depth = np.nanmax(pts_pr[:, :, 1])
        # ratio of median value in all tip points to tip of primary root in y axis
        tip_median_ratio = np.nanmedian(_tip_pts[:, 1]) / pr_tip_depth

        # lateral root tips percentile in y-axis
        tip_pctl = get_tips_percentile(pts_lr, pctl_tip)

    # calculate root prediction points convex hull related traits
    # get all points (lateral for rice; primary+lateral for other species)
    pts_all = get_pts_all(h5)

    (
        conv_perimeters,
        conv_areas,
        conv_longest_dists,
        conv_shortest_dists,
        conv_median_dists,
        conv_max_widths,
        conv_max_heights,
    ) = get_convhull_features(pts_all)

    # calculate best fitted ellipse related traits
    # get all points (lateral for rice; primary+lateral for other species)
    ellipse_a, ellipse_b, ellipse_ratio = fit_ellipse(pts_all)

    # calculate network related traits
    # width to depth ratio of bounding box for root network
    network_width_depth_ratio = get_network_width_depth_ratio(pts_all)

    # network_solidity: the total network length divided by the network convex area
    # network_length_lower_network:the root length in the lower fraction of the plant
    # network_distribution_ratio:
    # ratio of the root length in the lower fraction over all root length
    # calculate scan line related traits: count_scanline_interaction
    # return interaction points number of the scanline and predicted root
    if rice:
        network_solidity = get_network_solidity(pts_pr)
        network_length_lower_network = get_network_distribution(pts_pr, fraction)
        network_distribution_ratio = get_network_distribution_ratio(pts_pr)
        count_scanline_interaction = count_scanline_intersections(
            pts_pr, depth, width, n_line
        )
    else:
        network_solidity = get_network_solidity(pts_pr) + get_network_solidity(pts_lr)
        network_length_lower_network = get_network_distribution(
            pts_pr, fraction
        ) + get_network_distribution(pts_lr, fraction)
        network_distribution_ratio = get_network_distribution_ratio(
            pts_pr
        ) + get_network_distribution_ratio(pts_lr)
        count_scanline_interaction = np.nansum(
            np.dstack(
                (
                    count_scanline_intersections(pts_pr, depth, width, n_line),
                    count_scanline_intersections(pts_lr, depth, width, n_line),
                )
            ),
            2,
        )
        count_scanline_interaction[count_scanline_interaction == 0] = np.nan

    (
        count_scanline_interaction_max,
        count_scanline_interaction_min,
        count_scanline_interaction_mean,
        count_scanline_interaction_std,
        count_scanline_interaction_median,
    ) = get_statistics(count_scanline_interaction)
    # get the scan line number (or index of count_scanline_interaction) for the
    # first interaction
    scanline_start = np.where((count_scanline_interaction > 0))[1][0]
    # get the scan line number (or index of count_scanline_interaction) for the
    # last interaction
    scanline_end = np.where((count_scanline_interaction > 0))[1][-1]

    # save data as dataframe
    df = pd.DataFrame(
        {
            "plant_name": plant_name,
            "frame": frame,
            "primary_angles_proximal": primary_angles_proximal,
            "primary_angles_distal": primary_angles_distal,
            "lateral_angles_proximal_max": lateral_angles_proximal_max,
            "lateral_angles_proximal_min": lateral_angles_proximal_min,
            "lateral_angles_proximal_mean": lateral_angles_proximal_mean,
            "lateral_angles_proximal_std": lateral_angles_proximal_std,
            "lateral_angles_proximal_median": lateral_angles_proximal_median,
            "lateral_angles_distal_max": lateral_angles_distal_max,
            "lateral_angles_distal_min": lateral_angles_distal_min,
            "lateral_angles_distal_mean": lateral_angles_distal_mean,
            "lateral_angles_distal_std": lateral_angles_distal_std,
            "lateral_angles_distal_median": lateral_angles_distal_median,
            "primary_length": primary_length,
            "primary_base_tip_dist": primary_base_tip_dist,
            "primary_depth": primary_depth,
            "grav_index": grav_index,
            "lateral_length_total": lateral_length_total,
            "lateral_length_max": lateral_length_max,
            "lateral_length_min": lateral_length_min,
            "lateral_length_mean": lateral_length_mean,
            "lateral_length_std": lateral_length_std,
            "lateral_length_median": lateral_length_median,
            "stem_widths_max": stem_widths_max,
            "stem_widths_min": stem_widths_min,
            "stem_widths_mean": stem_widths_mean,
            "stem_widths_std": stem_widths_std,
            "stem_widths_median": stem_widths_median,
            "base_ct": base_ct,
            "base_top": base_top,
            "base_deep": base_deep,
            "base_length": base_length,
            "base_length_ratio": base_length_ratio,
            "base_meanx": base_meanx,
            "base_meany": base_meany,
            "base_medianx": base_medianx,
            "base_mediany": base_mediany,
            "base_stdx": base_stdx,
            "base_stdy": base_stdy,
            "pr_tip_depth": pr_tip_depth,
            "base_median_ratio": base_median_ratio,
            "base_ct_density": base_ct_density,
            "tip_ct": tip_ct,
            "tip_top": tip_top,
            "tip_deep": tip_deep,
            "tip_meanx": tip_meanx,
            "tip_meany": tip_meany,
            "tip_medianx": tip_medianx,
            "tip_mediany": tip_mediany,
            "tip_stdx": tip_stdx,
            "tip_stdy": tip_stdy,
            "tip_sdx_sdys": tip_sdx_sdys,
            "tip_median_ratio": tip_median_ratio,
            "conv_perimeters": conv_perimeters,
            "conv_areas": conv_areas,
            "conv_longest_dists": conv_longest_dists,
            "conv_shortest_dists": conv_shortest_dists,
            "conv_median_dists": conv_median_dists,
            "conv_max_widths": conv_max_widths,
            "conv_max_heights": conv_max_heights,
            "ellipse_a": ellipse_a,
            "ellipse_b": ellipse_b,
            "ellipse_ratio": ellipse_ratio,
            "network_width_depth_ratio": network_width_depth_ratio,
            "network_solidity": network_solidity,
            "network_length_lower_network": network_length_lower_network,
            "network_distribution_ratio": network_distribution_ratio,
            "count_scanline_interaction_max": count_scanline_interaction_max,
            "count_scanline_interaction_min": count_scanline_interaction_min,
            "count_scanline_interaction_mean": count_scanline_interaction_mean,
            "count_scanline_interaction_std": count_scanline_interaction_std,
            "count_scanline_interaction_median": count_scanline_interaction_median,
            "scanline_start": scanline_start,
            "scanline_end": scanline_end,
        }
    )

    # append base and tip percentile data
    for pctl in range(len(pctl_base)):
        df["base_pctl_" + str(pctl_base[pctl])] = base_pctl[pctl]
    for pctl in range(len(pctl_tip)):
        df["base_pctl_" + str(pctl_tip[pctl])] = tip_pctl[pctl]

    return df


def get_traits_plant(
    path,
    h5,
    rice=False,
    tolerance=0.02,
    pctl_base=[25, 75],
    pctl_tip=[25, 75],
    fraction: float = 2 / 3,
    depth: int = 1080,
    width: int = 2048,
    n_line: int = 50,
    write_csv=True,
) -> Tuple:
    """Extract plant-based traits based on SLEAP prediction *.h5 and *.slp files.

    Args:
        path: the work directory where store h5 and slp files.
        h5: h5 file, SLEAP prediction.
        rice: Boolean value, where true is rice (default), false is other species.
        tolerance: difference in projection norm between the right and left side.
        fraction: the network length found in the lower fration value of the network.
        depth: the depth of cylinder, or number of rows of the image.
        width: the width of cylinder, or number of columns of the image.
        n_line: number of scan lines.
        write_csv: Boolean value, where true is write csv file.

    Returns:
        A DataFrame with a single row containing columns with summarized (max, min,
        mean, std, median) traits from all frames per plant.
    """
    # load series
    if rice:
        series = Series.load(
            h5, primary_name="longest_3do_6nodes", lateral_name="main_3do_6nodes"
        )
    else:
        series = Series.load(
            h5, primary_name="primary_multi_day", lateral_name="lateral_3_nodes"
        )
    # get nymber of frames per plant
    n_frame = len(series)

    # get traits for each frames in a row
    for i in range(n_frame):
        df_frame = get_traits_frame(
            path,
            h5,
            rice,
            i,
            tolerance,
            pctl_base,
            pctl_tip,
            fraction,
            depth,
            width,
            n_line,
        )
        df_plant = df_frame if i == 0 else df_plant.append(df_frame, ignore_index=True)

    # generate summarized traits per plant
    df_plant_stat = pd.DataFrame([{"plant_name": df_plant["plant_name"][0]}])

    for j in range(len(df_plant.columns) - 2):
        trait_max, trait_min, trait_mean, trait_std, trait_median = get_statistics(
            df_plant.loc[:, df_plant.columns[j + 2]]
        )

        df_plant_stat[df_plant.columns[j + 2] + "_max"] = trait_max
        df_plant_stat[df_plant.columns[j + 2] + "_min"] = trait_min
        df_plant_stat[df_plant.columns[j + 2] + "_mean"] = trait_mean
        df_plant_stat[df_plant.columns[j + 2] + "_std"] = trait_std
        df_plant_stat[df_plant.columns[j + 2] + "_median"] = trait_median

    # write dataframe to csv file
    if write_csv:
        df_plant_stat.to_csv("traits_per_plant_SLEAP.csv", df_plant_stat)

    return df_plant_stat
