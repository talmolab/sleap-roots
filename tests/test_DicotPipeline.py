import numpy as np
import pandas as pd
import pytest

from sleap_roots.trait_pipelines import (
    DicotPipeline,
)
from sleap_roots.series import (
    Series,
)
from sleap_roots.lengths import get_max_length_pts, get_root_lengths, get_curve_index

from sleap_roots.points import get_count, join_pts, get_all_pts_array

from sleap_roots.bases import (
    get_root_widths,
    get_base_xs,
    get_base_ys,
    get_bases,
    get_base_ct_density,
    get_base_length,
    get_base_length_ratio,
    get_base_median_ratio,
    get_base_tip_dist,
)

from sleap_roots.tips import (
    get_tips,
    get_tip_xs,
    get_tip_ys,
)

from sleap_roots.scanline import (
    count_scanline_intersections,
    get_scanline_first_ind,
    get_scanline_last_ind,
)

from sleap_roots.angle import get_root_angle, get_node_ind

from sleap_roots.networklength import (
    get_network_solidity,
    get_network_length,
    get_network_distribution,
    get_network_width_depth_ratio,
    get_network_distribution_ratio,
    get_bbox,
)

from sleap_roots.convhull import (
    get_convhull,
    get_chull_area,
    get_chull_perimeter,
    get_chull_max_height,
    get_chull_max_width,
    get_chull_line_lengths,
)

from sleap_roots.ellipse import (
    fit_ellipse,
    get_ellipse_a,
    get_ellipse_b,
    get_ellipse_ratio,
)


def test_dicot_pipeline(
    canola_h5,
    soy_h5,
    canola_primary_slp,
    canola_lateral_slp,
    soy_primary_slp,
    soy_lateral_slp,
    canola_traits_csv,
    canola_batch_traits_csv,
    soy_traits_csv,
    soy_batch_traits_csv,
):
    canola = Series.load(
        series_name="canola",
        h5_path=canola_h5,
        primary_path=canola_primary_slp,
        lateral_path=canola_lateral_slp,
    )
    soy = Series.load(
        series_name="soy",
        h5_path=soy_h5,
        primary_path=soy_primary_slp,
        lateral_path=soy_lateral_slp,
    )

    pipeline = DicotPipeline()
    canola_traits = pipeline.compute_plant_traits(canola)
    soy_traits = pipeline.compute_plant_traits(soy)
    all_traits = pipeline.compute_batch_traits([canola, soy])

    canola_traits_csv = pd.read_csv(canola_traits_csv)
    soy_traits_csv = pd.read_csv(soy_traits_csv)

    assert canola_traits.shape == (72, 117)
    assert soy_traits.shape == (72, 117)
    assert all_traits.shape == (2, 1036)

    assert canola_traits_csv.shape == (72, 117)
    assert soy_traits_csv.shape == (72, 117)

    pd.testing.assert_frame_equal(
        canola_traits.iloc[:, 1:],
        canola_traits_csv.iloc[:, 1:],
        check_exact=False,
        atol=1e-7,
    )

    pd.testing.assert_frame_equal(
        soy_traits.iloc[:, 1:],
        soy_traits_csv.iloc[:, 1:],
        check_exact=False,
        atol=1e-7,
    )


def test_dicot_compute_plant_traits(
    canola_h5,
    soy_h5,
    canola_primary_slp,
    canola_lateral_slp,
    soy_primary_slp,
    soy_lateral_slp,
    canola_traits_csv,
    canola_batch_traits_csv,
    soy_traits_csv,
    soy_batch_traits_csv,
):

    canola = Series.load(
        series_name="canola",
        h5_path=canola_h5,
        primary_path=canola_primary_slp,
        lateral_path=canola_lateral_slp,
    )
    soy = Series.load(
        series_name="soy",
        h5_path=soy_h5,
        primary_path=soy_primary_slp,
        lateral_path=soy_lateral_slp,
    )

    pipeline = DicotPipeline()
    canola_traits = pipeline.compute_plant_traits(canola)
    soy_traits = pipeline.compute_plant_traits(soy)

    canola_traits_csv = pd.read_csv(canola_traits_csv)
    soy_traits_csv = pd.read_csv(soy_traits_csv)

    def check_summary_traits(traits_iterable, comparison_df, trait_name, frame_idx):
        """
        traits_iterable: The output of a trait calculation if trait is not a scalar
        comparison_df (pd.DataFrame): should be a pandas df of already computed pipeline results
        trait_name (str): The name of a trait
        frame_idx (int): The image frame index
        """

        if np.isnan(traits_iterable).all():
            assert np.isnan(comparison_df[f"{trait_name}_min"][frame_idx]).all()
            assert np.isnan(comparison_df[f"{trait_name}_max"][frame_idx]).all()
            assert np.isnan(comparison_df[f"{trait_name}_mean"][frame_idx]).all()
            assert np.isnan(comparison_df[f"{trait_name}_median"][frame_idx]).all()
            assert np.isnan(comparison_df[f"{trait_name}_std"][frame_idx]).all()
            assert np.isnan(comparison_df[f"{trait_name}_p5"][frame_idx]).all()
            assert np.isnan(comparison_df[f"{trait_name}_p25"][frame_idx]).all()
            assert np.isnan(comparison_df[f"{trait_name}_p75"][frame_idx]).all()
            assert np.isnan(comparison_df[f"{trait_name}_p95"][frame_idx]).all()
        else:
            np.testing.assert_almost_equal(
                np.nanmin(traits_iterable),
                comparison_df[f"{trait_name}_min"][frame_idx],
                decimal=7,
            )

            np.testing.assert_almost_equal(
                np.nanmax(traits_iterable),
                comparison_df[f"{trait_name}_max"][frame_idx],
                decimal=7,
            )

            np.testing.assert_almost_equal(
                np.nanmean(traits_iterable),
                comparison_df[f"{trait_name}_mean"][frame_idx],
                decimal=7,
            )

            np.testing.assert_almost_equal(
                np.nanmedian(traits_iterable),
                comparison_df[f"{trait_name}_median"][frame_idx],
                decimal=7,
            )

            np.testing.assert_almost_equal(
                np.nanstd(traits_iterable),
                comparison_df[f"{trait_name}_std"][frame_idx],
                decimal=7,
            )

            np.testing.assert_almost_equal(
                np.nanpercentile(traits_iterable, 5),
                comparison_df[f"{trait_name}_p5"][frame_idx],
                decimal=7,
            )

            np.testing.assert_almost_equal(
                np.nanpercentile(traits_iterable, 25),
                comparison_df[f"{trait_name}_p25"][frame_idx],
                decimal=7,
            )

            np.testing.assert_almost_equal(
                np.nanpercentile(traits_iterable, 75),
                comparison_df[f"{trait_name}_p75"][frame_idx],
                decimal=7,
            )

            np.testing.assert_almost_equal(
                np.nanpercentile(traits_iterable, 95),
                comparison_df[f"{trait_name}_p95"][frame_idx],
                decimal=7,
            )

    for frame_idx in range(72):

        canola_primary_points = canola.get_primary_points(frame_idx)
        canola_primary_max_length_pts = get_max_length_pts(canola_primary_points)
        canola_lateral_points = canola.get_lateral_points(frame_idx)

        ##### CHECK: canola_root_widths (summary) #####
        canola_root_widths = get_root_widths(
            primary_max_length_pts=canola_primary_max_length_pts,
            lateral_pts=canola_lateral_points,
        )

        check_summary_traits(
            traits_iterable=canola_root_widths,
            comparison_df=canola_traits_csv,
            trait_name="root_widths",
            frame_idx=frame_idx,
        )

        check_summary_traits(
            traits_iterable=canola_root_widths,
            comparison_df=canola_traits,
            trait_name="root_widths",
            frame_idx=frame_idx,
        )

        ##### CHECK: canola_lateral_count #####
        canola_lateral_count = get_count(canola_lateral_points)
        np.testing.assert_almost_equal(
            canola_lateral_count,
            canola_traits_csv["lateral_count"][frame_idx],
            decimal=7,
        )

        np.testing.assert_almost_equal(
            canola_lateral_count,
            canola_traits["lateral_count"][frame_idx],
            decimal=7,
        )

        ##### CHECK: canola_lateral_lengths (summary) #####
        canola_lateral_lengths = get_root_lengths(canola_lateral_points)

        check_summary_traits(
            traits_iterable=canola_lateral_lengths,
            comparison_df=canola_traits_csv,
            trait_name="lateral_lengths",
            frame_idx=frame_idx,
        )

        check_summary_traits(
            traits_iterable=canola_lateral_lengths,
            comparison_df=canola_traits,
            trait_name="lateral_lengths",
            frame_idx=frame_idx,
        )

        ##### CHECK: scanline_intersection_counts (summary) #####
        canola_pts_list = join_pts(canola_primary_max_length_pts, canola_lateral_points)
        canola_scanline_counts = count_scanline_intersections(
            pts_list=canola_pts_list,
            height=DicotPipeline().img_height,
            n_line=DicotPipeline().n_scanlines,
        )

        check_summary_traits(
            traits_iterable=canola_scanline_counts,
            comparison_df=canola_traits_csv,
            trait_name="scanline_intersection_counts",
            frame_idx=frame_idx,
        )

        check_summary_traits(
            traits_iterable=canola_scanline_counts,
            comparison_df=canola_traits,
            trait_name="scanline_intersection_counts",
            frame_idx=frame_idx,
        )

        ##### CHECK: lateral_angles_distal (summary) #####
        canola_lateral_distal_node_inds = get_node_ind(
            canola_lateral_points, proximal=False
        )
        canola_lateral_angles_distal = get_root_angle(
            pts=canola_lateral_points,
            node_ind=canola_lateral_distal_node_inds,
            proximal=False,
            base_ind=0,
        )

        check_summary_traits(
            traits_iterable=canola_lateral_angles_distal,
            comparison_df=canola_traits_csv,
            trait_name="lateral_angles_distal",
            frame_idx=frame_idx,
        )

        check_summary_traits(
            traits_iterable=canola_lateral_angles_distal,
            comparison_df=canola_traits,
            trait_name="lateral_angles_distal",
            frame_idx=frame_idx,
        )

        ##### CHECK: lateral_angles_proximal (summary) #####
        canola_lateral_distal_node_inds = get_node_ind(canola_lateral_points)
        canola_lateral_angles_proximal = get_root_angle(
            pts=canola_lateral_points,
            node_ind=canola_lateral_distal_node_inds,
            proximal=True,
            base_ind=0,
        )

        check_summary_traits(
            traits_iterable=canola_lateral_angles_proximal,
            comparison_df=canola_traits_csv,
            trait_name="lateral_angles_proximal",
            frame_idx=frame_idx,
        )

        check_summary_traits(
            traits_iterable=canola_lateral_angles_proximal,
            comparison_df=canola_traits,
            trait_name="lateral_angles_proximal",
            frame_idx=frame_idx,
        )

        ##### CHECK: network_solidity #####
        canola_all_pts_array = get_all_pts_array(
            canola_primary_max_length_pts, canola_lateral_points
        )
        canola_convex_hull = get_convhull(canola_all_pts_array)
        canola_chull_area = get_chull_area(hull=canola_convex_hull)

        canola_primary_length = get_root_lengths(canola_primary_max_length_pts)
        canola_lateral_lengths = get_root_lengths(canola_lateral_points)

        canola_network_length = get_network_length(
            canola_primary_length, canola_lateral_lengths
        )
        canola_network_solidity = get_network_solidity(
            network_length=canola_network_length, chull_area=canola_chull_area
        )

        assert np.isclose(
            canola_network_solidity, canola_traits_csv["network_solidity"][frame_idx]
        )

        assert np.isclose(
            canola_network_solidity, canola_traits["network_solidity"][frame_idx]
        )

        ##### CHECK: primary_angle_proximal #####
        canola_primary_proximal_node_inds = get_node_ind(
            canola_primary_max_length_pts, proximal=True
        )

        canola_primary_angle_proximal = get_root_angle(
            pts=canola_primary_max_length_pts,
            node_ind=canola_primary_proximal_node_inds,
            proximal=True,
            base_ind=0,
        )

        assert np.isclose(
            canola_primary_angle_proximal,
            canola_traits_csv["primary_angle_proximal"][frame_idx],
            equal_nan=True,
        )

        assert np.isclose(
            canola_primary_angle_proximal,
            canola_traits["primary_angle_proximal"][frame_idx],
            equal_nan=True,
        )

        ##### CHECK: primary_angle_distal #####
        canola_primary_distal_node_inds = get_node_ind(
            pts=canola_primary_max_length_pts, proximal=False
        )

        canola_primary_angle_distal = get_root_angle(
            pts=canola_primary_max_length_pts,
            node_ind=canola_primary_distal_node_inds,
            proximal=False,
            base_ind=0,
        )

        assert np.isclose(
            canola_primary_angle_distal,
            canola_traits_csv["primary_angle_distal"][frame_idx],
            equal_nan=True,
        )

        assert np.isclose(
            canola_primary_angle_distal,
            canola_traits["primary_angle_distal"][frame_idx],
            equal_nan=True,
        )

        ##### CHECK: primary_length #####
        canola_primary_length = get_root_lengths(canola_primary_max_length_pts)

        assert np.isclose(
            canola_primary_length,
            canola_traits_csv["primary_length"][frame_idx],
            equal_nan=True,
        )

        assert np.isclose(
            canola_primary_length,
            canola_traits["primary_length"][frame_idx],
            equal_nan=True,
        )

        ##### CHECK: network_length_lower #####
        canola_bounding_box = get_bbox(canola_all_pts_array)
        canola_network_length_lower = get_network_distribution(
            canola_pts_list, canola_bounding_box
        )

        assert np.isclose(
            canola_network_length_lower,
            canola_traits_csv["network_length_lower"][frame_idx],
            equal_nan=True,
        )

        assert np.isclose(
            canola_network_length_lower,
            canola_traits["network_length_lower"][frame_idx],
            equal_nan=True,
        )

        ##### CHECK: lateral_base_xs (summary) #####
        canola_lateral_base_pts = get_bases(canola_lateral_points)
        canola_lateral_base_xs = get_base_xs(base_pts=canola_lateral_base_pts)

        check_summary_traits(
            traits_iterable=canola_lateral_base_xs,
            comparison_df=canola_traits_csv,
            trait_name="lateral_base_xs",
            frame_idx=frame_idx,
        )

        ##### CHECK: lateral_base_ys (summary) #####
        canola_lateral_base_ys = get_base_ys(base_pts=canola_lateral_base_pts)
        check_summary_traits(
            traits_iterable=canola_lateral_base_ys,
            comparison_df=canola_traits_csv,
            trait_name="lateral_base_ys",
            frame_idx=frame_idx,
        )

        ##### CHECK: base_ct_density #####
        canola_base_ct_density = get_base_ct_density(
            primary_length_max=canola_primary_length,
            lateral_base_pts=canola_lateral_base_pts,
        )

        assert np.isclose(
            canola_base_ct_density,
            canola_traits_csv["base_ct_density"][frame_idx],
            equal_nan=True,
        )

        assert np.isclose(
            canola_base_ct_density,
            canola_traits["base_ct_density"][frame_idx],
            equal_nan=True,
        )

        ##### CHECK: lateral_tip_xs (summary) #####
        canola_lateral_tips = get_tips(pts=canola_lateral_points)
        canola_lateral_tip_xs = get_tip_xs(canola_lateral_tips)

        check_summary_traits(
            traits_iterable=canola_lateral_tip_xs,
            comparison_df=canola_traits_csv,
            trait_name="lateral_tip_xs",
            frame_idx=frame_idx,
        )

        ##### CHECK: lateral_tip_xs (summary) #####
        canola_lateral_tip_ys = get_tip_ys(canola_lateral_tips)

        check_summary_traits(
            traits_iterable=canola_lateral_tip_ys,
            comparison_df=canola_traits_csv,
            trait_name="lateral_tip_ys",
            frame_idx=frame_idx,
        )

        ##### CHECK: network_distribution_ratio #####
        canola_network_distribution_ratio = get_network_distribution_ratio(
            network_length=canola_network_length,
            network_length_lower=canola_network_length_lower,
        )
        assert np.isclose(
            canola_network_distribution_ratio,
            canola_traits_csv["network_distribution_ratio"][frame_idx],
            equal_nan=True,
        )

        assert np.isclose(
            canola_network_distribution_ratio,
            canola_traits["network_distribution_ratio"][frame_idx],
            equal_nan=True,
        )

        ##### CHECK: network_length #####
        assert np.isclose(
            canola_network_length,
            canola_traits_csv["network_length"][frame_idx],
            equal_nan=True,
        )

        assert np.isclose(
            canola_network_length,
            canola_traits["network_length"][frame_idx],
            equal_nan=True,
        )

        ##### CHECK: primary_tip_pt_y #####
        canola_primary_tip_pt = get_tips(canola_primary_max_length_pts)
        canola_primary_tip_pt_y = get_tip_ys(canola_primary_tip_pt)

        assert np.isclose(
            canola_primary_tip_pt_y,
            canola_traits_csv["primary_tip_pt_y"][frame_idx],
            equal_nan=True,
        )

        assert np.isclose(
            canola_primary_tip_pt_y,
            canola_traits["primary_tip_pt_y"][frame_idx],
            equal_nan=True,
        )

        ##### CHECK: ellipse_a #####
        canola_ellipse = fit_ellipse(pts=canola_all_pts_array)
        canola_ellipse_a = get_ellipse_a(canola_ellipse)

        assert np.isclose(
            canola_ellipse_a, canola_traits_csv["ellipse_a"][frame_idx], equal_nan=True
        )

        assert np.isclose(
            canola_ellipse_a, canola_traits["ellipse_a"][frame_idx], equal_nan=True
        )

        ##### CHECK: ellipse_b #####
        canola_ellipse_b = get_ellipse_b(canola_ellipse)

        assert np.isclose(
            canola_ellipse_b, canola_traits_csv["ellipse_b"][frame_idx], equal_nan=True
        )

        assert np.isclose(
            canola_ellipse_b, canola_traits["ellipse_b"][frame_idx], equal_nan=True
        )

        ##### CHECK: network_width_depth_ratio #####
        canola_width_depth_ratio = get_network_width_depth_ratio(
            pts=canola_bounding_box
        )
        assert np.isclose(
            canola_width_depth_ratio,
            canola_traits_csv["network_width_depth_ratio"][frame_idx],
            equal_nan=True,
        )

        assert np.isclose(
            canola_width_depth_ratio,
            canola_traits["network_width_depth_ratio"][frame_idx],
            equal_nan=True,
        )

        ##### CHECK: chull_perimeter #####
        canola_chull_perimeter = get_chull_perimeter(canola_convex_hull)
        assert np.isclose(
            canola_chull_perimeter,
            canola_traits_csv["chull_perimeter"][frame_idx],
            equal_nan=True,
        )

        assert np.isclose(
            canola_chull_perimeter,
            canola_traits["chull_perimeter"][frame_idx],
            equal_nan=True,
        )

        ##### CHECK: chull_area #####
        assert np.isclose(
            canola_chull_area,
            canola_traits_csv["chull_area"][frame_idx],
            equal_nan=True,
        )

        assert np.isclose(
            canola_chull_area,
            canola_traits["chull_area"][frame_idx],
            equal_nan=True,
        )

        ##### CHECK: chull_max_width #####
        canola_chull_max_width = get_chull_max_width(canola_convex_hull)
        assert np.isclose(
            canola_chull_max_width,
            canola_traits_csv["chull_max_width"][frame_idx],
            equal_nan=True,
        )

        canola_chull_max_width = get_chull_max_width(canola_convex_hull)
        assert np.isclose(
            canola_chull_max_width,
            canola_traits["chull_max_width"][frame_idx],
            equal_nan=True,
        )

        ##### CHECK: chull_max_height #####
        canola_chull_max_height = get_chull_max_height(canola_convex_hull)
        assert np.isclose(
            canola_chull_max_height,
            canola_traits_csv["chull_max_height"][frame_idx],
            equal_nan=True,
        )

        assert np.isclose(
            canola_chull_max_height,
            canola_traits["chull_max_height"][frame_idx],
            equal_nan=True,
        )

        ##### CHECK: chull_line_lengths (summary) #####
        canola_chull_line_lengths = get_chull_line_lengths(canola_convex_hull)
        check_summary_traits(
            traits_iterable=canola_chull_line_lengths,
            comparison_df=canola_traits_csv,
            trait_name="chull_line_lengths",
            frame_idx=frame_idx,
        )

        ##### CHECK: base_length #####
        canola_base_length = get_base_length(lateral_base_ys=canola_lateral_base_ys)

        assert np.isclose(
            canola_base_length,
            canola_traits_csv["base_length"][frame_idx],
            equal_nan=True,
        )

        assert np.isclose(
            canola_base_length,
            canola_traits["base_length"][frame_idx],
            equal_nan=True,
        )

        ##### CHECK: base_median_ratio #####
        canola_base_median_ratio = get_base_median_ratio(
            lateral_base_ys=canola_lateral_base_ys,
            primary_tip_pt_y=canola_primary_tip_pt_y,
        )

        assert np.isclose(
            canola_base_median_ratio,
            canola_traits_csv["base_median_ratio"][frame_idx],
            equal_nan=True,
        )

        assert np.isclose(
            canola_base_median_ratio,
            canola_traits["base_median_ratio"][frame_idx],
            equal_nan=True,
        )

        ##### CHECK: curve_index #####
        canola_primary_base_pt = get_bases(canola_primary_max_length_pts)
        canola_primary_tip_pt = get_tips(canola_primary_max_length_pts)

        canola_primary_base_tip_dist = get_base_tip_dist(
            base_pts=canola_primary_base_pt, tip_pts=canola_primary_tip_pt
        )

        canola_curve_index = get_curve_index(
            lengths=canola_primary_length, base_tip_dists=canola_primary_base_tip_dist
        )

        assert np.isclose(
            canola_curve_index,
            canola_traits_csv["curve_index"][frame_idx],
            equal_nan=True,
        )

        assert np.isclose(
            canola_curve_index,
            canola_traits["curve_index"][frame_idx],
            equal_nan=True,
        )

        ##### CHECK: base_length_ratio #####
        canola_base_length_ratio = get_base_length_ratio(
            primary_length=canola_primary_length,
            base_length=canola_base_length,
        )

        assert np.isclose(
            canola_base_length_ratio,
            canola_traits_csv["base_length_ratio"][frame_idx],
            equal_nan=True,
        )

        assert np.isclose(
            canola_base_length_ratio,
            canola_traits["base_length_ratio"][frame_idx],
            equal_nan=True,
        )

        ##### CHECK: primary_base_tip_dist #####
        assert np.isclose(
            canola_primary_base_tip_dist,
            canola_traits_csv["primary_base_tip_dist"][frame_idx],
            equal_nan=True,
        )

        assert np.isclose(
            canola_primary_base_tip_dist,
            canola_traits["primary_base_tip_dist"][frame_idx],
            equal_nan=True,
        )

        ##### CHECK: ellipse_ratio #####
        canola_ellipse_ratio = get_ellipse_ratio(canola_ellipse)

        assert np.isclose(
            canola_ellipse_ratio,
            canola_traits_csv["ellipse_ratio"][frame_idx],
            equal_nan=True,
        )

        assert np.isclose(
            canola_ellipse_ratio,
            canola_traits["ellipse_ratio"][frame_idx],
            equal_nan=True,
        )

        ##### CHECK: scanline_last_ind #####
        canola_scanline_intersection_counts = count_scanline_intersections(
            canola_pts_list
        )

        canola_scanline_last_ind = get_scanline_last_ind(
            canola_scanline_intersection_counts
        )

        assert np.isclose(
            canola_scanline_last_ind,
            canola_traits_csv["scanline_last_ind"][frame_idx],
            equal_nan=True,
        )

        assert np.isclose(
            canola_scanline_last_ind,
            canola_traits["scanline_last_ind"][frame_idx],
            equal_nan=True,
        )

        ##### CHECK: scanline_first_ind #####
        canola_scanline_first_ind = get_scanline_first_ind(
            canola_scanline_intersection_counts
        )

        assert np.isclose(
            canola_scanline_first_ind,
            canola_traits_csv["scanline_first_ind"][frame_idx],
            equal_nan=True,
        )

        assert np.isclose(
            canola_scanline_first_ind,
            canola_traits["scanline_first_ind"][frame_idx],
            equal_nan=True,
        )

        soy_primary_points = soy.get_primary_points(frame_idx)
        soy_primary_max_length_pts = get_max_length_pts(soy_primary_points)
        soy_lateral_points = soy.get_lateral_points(frame_idx)

        ##### CHECK: soy_root_widths (summary) #####
        soy_root_widths = get_root_widths(
            primary_max_length_pts=soy_primary_max_length_pts,
            lateral_pts=soy_lateral_points,
        )

        check_summary_traits(
            traits_iterable=soy_root_widths,
            comparison_df=soy_traits_csv,
            trait_name="root_widths",
            frame_idx=frame_idx,
        )

        check_summary_traits(
            traits_iterable=soy_root_widths,
            comparison_df=soy_traits,
            trait_name="root_widths",
            frame_idx=frame_idx,
        )

        ##### CHECK: soy_lateral_count #####
        soy_lateral_count = get_count(soy_lateral_points)
        np.testing.assert_almost_equal(
            soy_lateral_count,
            soy_traits_csv["lateral_count"][frame_idx],
            decimal=7,
        )

        np.testing.assert_almost_equal(
            soy_lateral_count,
            soy_traits["lateral_count"][frame_idx],
            decimal=7,
        )

        ##### CHECK: soy_lateral_lengths (summary) #####
        soy_lateral_lengths = get_root_lengths(soy_lateral_points)

        check_summary_traits(
            traits_iterable=soy_lateral_lengths,
            comparison_df=soy_traits_csv,
            trait_name="lateral_lengths",
            frame_idx=frame_idx,
        )

        check_summary_traits(
            traits_iterable=soy_lateral_lengths,
            comparison_df=soy_traits,
            trait_name="lateral_lengths",
            frame_idx=frame_idx,
        )

        ##### CHECK: scanline_intersection_counts (summary) #####
        soy_pts_list = join_pts(soy_primary_max_length_pts, soy_lateral_points)
        soy_scanline_counts = count_scanline_intersections(
            pts_list=soy_pts_list,
            height=DicotPipeline().img_height,
            n_line=DicotPipeline().n_scanlines,
        )

        check_summary_traits(
            traits_iterable=soy_scanline_counts,
            comparison_df=soy_traits_csv,
            trait_name="scanline_intersection_counts",
            frame_idx=frame_idx,
        )

        check_summary_traits(
            traits_iterable=soy_scanline_counts,
            comparison_df=soy_traits,
            trait_name="scanline_intersection_counts",
            frame_idx=frame_idx,
        )

        ##### CHECK: lateral_angles_distal (summary) #####
        soy_lateral_distal_node_inds = get_node_ind(soy_lateral_points, proximal=False)
        soy_lateral_angles_distal = get_root_angle(
            pts=soy_lateral_points,
            node_ind=soy_lateral_distal_node_inds,
            proximal=False,
            base_ind=0,
        )

        check_summary_traits(
            traits_iterable=soy_lateral_angles_distal,
            comparison_df=soy_traits_csv,
            trait_name="lateral_angles_distal",
            frame_idx=frame_idx,
        )

        check_summary_traits(
            traits_iterable=soy_lateral_angles_distal,
            comparison_df=soy_traits,
            trait_name="lateral_angles_distal",
            frame_idx=frame_idx,
        )

        ##### CHECK: lateral_angles_proximal (summary) #####
        soy_lateral_distal_node_inds = get_node_ind(soy_lateral_points)
        soy_lateral_angles_proximal = get_root_angle(
            pts=soy_lateral_points,
            node_ind=soy_lateral_distal_node_inds,
            proximal=True,
            base_ind=0,
        )

        check_summary_traits(
            traits_iterable=soy_lateral_angles_proximal,
            comparison_df=soy_traits_csv,
            trait_name="lateral_angles_proximal",
            frame_idx=frame_idx,
        )

        check_summary_traits(
            traits_iterable=soy_lateral_angles_proximal,
            comparison_df=soy_traits,
            trait_name="lateral_angles_proximal",
            frame_idx=frame_idx,
        )

        ##### CHECK: network_solidity #####
        soy_all_pts_array = get_all_pts_array(
            soy_primary_max_length_pts, soy_lateral_points
        )
        soy_convex_hull = get_convhull(soy_all_pts_array)
        soy_chull_area = get_chull_area(hull=soy_convex_hull)

        soy_primary_length = get_root_lengths(soy_primary_max_length_pts)
        soy_lateral_lengths = get_root_lengths(soy_lateral_points)

        soy_network_length = get_network_length(soy_primary_length, soy_lateral_lengths)
        soy_network_solidity = get_network_solidity(
            network_length=soy_network_length, chull_area=soy_chull_area
        )

        assert np.isclose(
            soy_network_solidity, soy_traits_csv["network_solidity"][frame_idx]
        )

        assert np.isclose(
            soy_network_solidity, soy_traits["network_solidity"][frame_idx]
        )

        ##### CHECK: primary_angle_proximal #####
        soy_primary_proximal_node_inds = get_node_ind(
            soy_primary_max_length_pts, proximal=True
        )

        soy_primary_angle_proximal = get_root_angle(
            pts=soy_primary_max_length_pts,
            node_ind=soy_primary_proximal_node_inds,
            proximal=True,
            base_ind=0,
        )

        assert np.isclose(
            soy_primary_angle_proximal,
            soy_traits_csv["primary_angle_proximal"][frame_idx],
            equal_nan=True,
        )

        assert np.isclose(
            soy_primary_angle_proximal,
            soy_traits["primary_angle_proximal"][frame_idx],
            equal_nan=True,
        )

        ##### CHECK: primary_angle_distal #####
        soy_primary_distal_node_inds = get_node_ind(
            pts=soy_primary_max_length_pts, proximal=False
        )

        soy_primary_angle_distal = get_root_angle(
            pts=soy_primary_max_length_pts,
            node_ind=soy_primary_distal_node_inds,
            proximal=False,
            base_ind=0,
        )

        assert np.isclose(
            soy_primary_angle_distal,
            soy_traits_csv["primary_angle_distal"][frame_idx],
            equal_nan=True,
        )

        assert np.isclose(
            soy_primary_angle_distal,
            soy_traits["primary_angle_distal"][frame_idx],
            equal_nan=True,
        )

        ##### CHECK: primary_length #####
        soy_primary_length = get_root_lengths(soy_primary_max_length_pts)

        assert np.isclose(
            soy_primary_length,
            soy_traits_csv["primary_length"][frame_idx],
            equal_nan=True,
        )

        assert np.isclose(
            soy_primary_length,
            soy_traits["primary_length"][frame_idx],
            equal_nan=True,
        )

        ##### CHECK: network_length_lower #####
        soy_bounding_box = get_bbox(soy_all_pts_array)
        soy_network_length_lower = get_network_distribution(
            soy_pts_list, soy_bounding_box
        )

        assert np.isclose(
            soy_network_length_lower,
            soy_traits_csv["network_length_lower"][frame_idx],
            equal_nan=True,
        )

        assert np.isclose(
            soy_network_length_lower,
            soy_traits["network_length_lower"][frame_idx],
            equal_nan=True,
        )

        ##### CHECK: lateral_base_xs (summary) #####
        soy_lateral_base_pts = get_bases(soy_lateral_points)
        soy_lateral_base_xs = get_base_xs(base_pts=soy_lateral_base_pts)

        check_summary_traits(
            traits_iterable=soy_lateral_base_xs,
            comparison_df=soy_traits_csv,
            trait_name="lateral_base_xs",
            frame_idx=frame_idx,
        )

        ##### CHECK: lateral_base_ys (summary) #####
        soy_lateral_base_ys = get_base_ys(base_pts=soy_lateral_base_pts)
        check_summary_traits(
            traits_iterable=soy_lateral_base_ys,
            comparison_df=soy_traits_csv,
            trait_name="lateral_base_ys",
            frame_idx=frame_idx,
        )

        ##### CHECK: base_ct_density #####
        soy_base_ct_density = get_base_ct_density(
            primary_length_max=soy_primary_length,
            lateral_base_pts=soy_lateral_base_pts,
        )

        assert np.isclose(
            soy_base_ct_density,
            soy_traits_csv["base_ct_density"][frame_idx],
            equal_nan=True,
        )

        assert np.isclose(
            soy_base_ct_density,
            soy_traits["base_ct_density"][frame_idx],
            equal_nan=True,
        )

        ##### CHECK: lateral_tip_xs (summary) #####
        soy_lateral_tips = get_tips(pts=soy_lateral_points)
        soy_lateral_tip_xs = get_tip_xs(soy_lateral_tips)

        check_summary_traits(
            traits_iterable=soy_lateral_tip_xs,
            comparison_df=soy_traits_csv,
            trait_name="lateral_tip_xs",
            frame_idx=frame_idx,
        )

        ##### CHECK: lateral_tip_xs (summary) #####
        soy_lateral_tip_ys = get_tip_ys(soy_lateral_tips)

        check_summary_traits(
            traits_iterable=soy_lateral_tip_ys,
            comparison_df=soy_traits_csv,
            trait_name="lateral_tip_ys",
            frame_idx=frame_idx,
        )

        ##### CHECK: network_distribution_ratio #####
        soy_network_distribution_ratio = get_network_distribution_ratio(
            network_length=soy_network_length,
            network_length_lower=soy_network_length_lower,
        )
        assert np.isclose(
            soy_network_distribution_ratio,
            soy_traits_csv["network_distribution_ratio"][frame_idx],
            equal_nan=True,
        )

        assert np.isclose(
            soy_network_distribution_ratio,
            soy_traits["network_distribution_ratio"][frame_idx],
            equal_nan=True,
        )

        ##### CHECK: network_length #####
        assert np.isclose(
            soy_network_length,
            soy_traits_csv["network_length"][frame_idx],
            equal_nan=True,
        )

        assert np.isclose(
            soy_network_length,
            soy_traits["network_length"][frame_idx],
            equal_nan=True,
        )

        ##### CHECK: primary_tip_pt_y #####
        soy_primary_tip_pt = get_tips(soy_primary_max_length_pts)
        soy_primary_tip_pt_y = get_tip_ys(soy_primary_tip_pt)

        assert np.isclose(
            soy_primary_tip_pt_y,
            soy_traits_csv["primary_tip_pt_y"][frame_idx],
            equal_nan=True,
        )

        assert np.isclose(
            soy_primary_tip_pt_y,
            soy_traits["primary_tip_pt_y"][frame_idx],
            equal_nan=True,
        )

        ##### CHECK: ellipse_a #####
        soy_ellipse = fit_ellipse(pts=soy_all_pts_array)
        soy_ellipse_a = get_ellipse_a(soy_ellipse)

        assert np.isclose(
            soy_ellipse_a, soy_traits_csv["ellipse_a"][frame_idx], equal_nan=True
        )

        assert np.isclose(
            soy_ellipse_a, soy_traits["ellipse_a"][frame_idx], equal_nan=True
        )

        ##### CHECK: ellipse_b #####
        soy_ellipse_b = get_ellipse_b(soy_ellipse)

        assert np.isclose(
            soy_ellipse_b, soy_traits_csv["ellipse_b"][frame_idx], equal_nan=True
        )

        assert np.isclose(
            soy_ellipse_b, soy_traits["ellipse_b"][frame_idx], equal_nan=True
        )

        ##### CHECK: network_width_depth_ratio #####
        soy_width_depth_ratio = get_network_width_depth_ratio(pts=soy_bounding_box)
        assert np.isclose(
            soy_width_depth_ratio,
            soy_traits_csv["network_width_depth_ratio"][frame_idx],
            equal_nan=True,
        )

        assert np.isclose(
            soy_width_depth_ratio,
            soy_traits["network_width_depth_ratio"][frame_idx],
            equal_nan=True,
        )

        ##### CHECK: chull_perimeter #####
        soy_chull_perimeter = get_chull_perimeter(soy_convex_hull)
        assert np.isclose(
            soy_chull_perimeter,
            soy_traits_csv["chull_perimeter"][frame_idx],
            equal_nan=True,
        )

        assert np.isclose(
            soy_chull_perimeter,
            soy_traits["chull_perimeter"][frame_idx],
            equal_nan=True,
        )

        ##### CHECK: chull_area #####
        assert np.isclose(
            soy_chull_area,
            soy_traits_csv["chull_area"][frame_idx],
            equal_nan=True,
        )

        assert np.isclose(
            soy_chull_area,
            soy_traits["chull_area"][frame_idx],
            equal_nan=True,
        )

        ##### CHECK: chull_max_width #####
        soy_chull_max_width = get_chull_max_width(soy_convex_hull)
        assert np.isclose(
            soy_chull_max_width,
            soy_traits_csv["chull_max_width"][frame_idx],
            equal_nan=True,
        )

        soy_chull_max_width = get_chull_max_width(soy_convex_hull)
        assert np.isclose(
            soy_chull_max_width,
            soy_traits["chull_max_width"][frame_idx],
            equal_nan=True,
        )

        ##### CHECK: chull_max_height #####
        soy_chull_max_height = get_chull_max_height(soy_convex_hull)
        assert np.isclose(
            soy_chull_max_height,
            soy_traits_csv["chull_max_height"][frame_idx],
            equal_nan=True,
        )

        assert np.isclose(
            soy_chull_max_height,
            soy_traits["chull_max_height"][frame_idx],
            equal_nan=True,
        )

        ##### CHECK: chull_line_lengths (summary) #####
        soy_chull_line_lengths = get_chull_line_lengths(soy_convex_hull)
        check_summary_traits(
            traits_iterable=soy_chull_line_lengths,
            comparison_df=soy_traits_csv,
            trait_name="chull_line_lengths",
            frame_idx=frame_idx,
        )

        ##### CHECK: base_length #####
        soy_base_length = get_base_length(lateral_base_ys=soy_lateral_base_ys)

        assert np.isclose(
            soy_base_length,
            soy_traits_csv["base_length"][frame_idx],
            equal_nan=True,
        )

        assert np.isclose(
            soy_base_length,
            soy_traits["base_length"][frame_idx],
            equal_nan=True,
        )

        ##### CHECK: base_median_ratio #####
        soy_base_median_ratio = get_base_median_ratio(
            lateral_base_ys=soy_lateral_base_ys,
            primary_tip_pt_y=soy_primary_tip_pt_y,
        )

        assert np.isclose(
            soy_base_median_ratio,
            soy_traits_csv["base_median_ratio"][frame_idx],
            equal_nan=True,
        )

        assert np.isclose(
            soy_base_median_ratio,
            soy_traits["base_median_ratio"][frame_idx],
            equal_nan=True,
        )

        ##### CHECK: curve_index #####
        soy_primary_base_pt = get_bases(soy_primary_max_length_pts)
        soy_primary_tip_pt = get_tips(soy_primary_max_length_pts)

        soy_primary_base_tip_dist = get_base_tip_dist(
            base_pts=soy_primary_base_pt, tip_pts=soy_primary_tip_pt
        )

        soy_curve_index = get_curve_index(
            lengths=soy_primary_length, base_tip_dists=soy_primary_base_tip_dist
        )

        assert np.isclose(
            soy_curve_index,
            soy_traits_csv["curve_index"][frame_idx],
            equal_nan=True,
        )

        assert np.isclose(
            soy_curve_index,
            soy_traits["curve_index"][frame_idx],
            equal_nan=True,
        )

        ##### CHECK: base_length_ratio #####
        soy_base_length_ratio = get_base_length_ratio(
            primary_length=soy_primary_length,
            base_length=soy_base_length,
        )

        assert np.isclose(
            soy_base_length_ratio,
            soy_traits_csv["base_length_ratio"][frame_idx],
            equal_nan=True,
        )

        assert np.isclose(
            soy_base_length_ratio,
            soy_traits["base_length_ratio"][frame_idx],
            equal_nan=True,
        )

        ##### CHECK: primary_base_tip_dist #####
        assert np.isclose(
            soy_primary_base_tip_dist,
            soy_traits_csv["primary_base_tip_dist"][frame_idx],
            equal_nan=True,
        )

        assert np.isclose(
            soy_primary_base_tip_dist,
            soy_traits["primary_base_tip_dist"][frame_idx],
            equal_nan=True,
        )

        ##### CHECK: ellipse_ratio #####
        soy_ellipse_ratio = get_ellipse_ratio(soy_ellipse)

        assert np.isclose(
            soy_ellipse_ratio,
            soy_traits_csv["ellipse_ratio"][frame_idx],
            equal_nan=True,
        )

        assert np.isclose(
            soy_ellipse_ratio,
            soy_traits["ellipse_ratio"][frame_idx],
            equal_nan=True,
        )

        ##### CHECK: scanline_last_ind #####
        soy_scanline_intersection_counts = count_scanline_intersections(soy_pts_list)

        soy_scanline_last_ind = get_scanline_last_ind(soy_scanline_intersection_counts)

        assert np.isclose(
            soy_scanline_last_ind,
            soy_traits_csv["scanline_last_ind"][frame_idx],
            equal_nan=True,
        )

        assert np.isclose(
            soy_scanline_last_ind,
            soy_traits["scanline_last_ind"][frame_idx],
            equal_nan=True,
        )

        ##### CHECK: scanline_first_ind #####
        soy_scanline_first_ind = get_scanline_first_ind(
            soy_scanline_intersection_counts
        )

        assert np.isclose(
            soy_scanline_first_ind,
            soy_traits_csv["scanline_first_ind"][frame_idx],
            equal_nan=True,
        )

        assert np.isclose(
            soy_scanline_first_ind,
            soy_traits["scanline_first_ind"][frame_idx],
            equal_nan=True,
        )
