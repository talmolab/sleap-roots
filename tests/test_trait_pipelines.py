import numpy as np
import pandas as pd
import json
import pytest

from sleap_roots.trait_pipelines import (
    DicotPipeline,
    YoungerMonocotPipeline,
    OlderMonocotPipeline,
    MultipleDicotPipeline,
    NumpyArrayEncoder,
    PrimaryRootPipeline,
    MultiplePrimaryRootPipeline,
)
from sleap_roots.series import (
    Series,
    find_all_h5_paths,
    find_all_slp_paths,
    load_series_from_h5s,
    load_series_from_slps,
)
from sleap_roots.lengths import get_max_length_pts, get_root_lengths, get_curve_index

from sleap_roots.points import (
    get_count,
    join_pts,
    get_all_pts_array,
    get_nodes,
    filter_primary_roots_with_unexpected_count,
)

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

from sleap_roots.angle import (
    get_root_angle,
    get_node_ind,
    get_vector_angles_from_gravity,
)

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
    get_chull_intersection_vectors,
    get_chull_intersection_vectors_left,
    get_chull_intersection_vectors_right,
    get_chull_areas_via_intersection,
    get_chull_area_via_intersection_above,
    get_chull_area_via_intersection_below,
)

from sleap_roots.ellipse import (
    fit_ellipse,
    get_ellipse_a,
    get_ellipse_b,
    get_ellipse_ratio,
)

from sleap_roots.summary import get_summary

import sleap_roots as sr


def test_numpy_array_serialization():
    array = np.array([1, 2, 3])
    expected = [1, 2, 3]
    json_str = json.dumps(array, cls=NumpyArrayEncoder)
    assert json.loads(json_str) == expected


def test_numpy_int64_serialization():
    int64_value = np.int64(42)
    expected = 42
    json_str = json.dumps(int64_value, cls=NumpyArrayEncoder)
    assert json.loads(json_str) == expected


def test_unsupported_type_serialization():
    class UnsupportedType:
        pass

    with pytest.raises(TypeError):
        json.dumps(UnsupportedType(), cls=NumpyArrayEncoder)


def test_mixed_data_serialization():
    data = {
        "array": np.array([1, 2, 3]),
        "int64": np.int64(42),
        "regular_int": 99,
        "list": [4, 5, 6],
        "dict": {"key": "value"},
    }
    expected = {
        "array": [1, 2, 3],
        "int64": 42,
        "regular_int": 99,
        "list": [4, 5, 6],
        "dict": {"key": "value"},
    }
    json_str = json.dumps(data, cls=NumpyArrayEncoder)
    assert json.loads(json_str) == expected


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
    """
    Tests the DicotPipeline to ensure frame-level and plant-level traits are calculated
    correctly. Manually calculated traits directly call the functions and use inputs
    from the trait definition.

    Compares:
        - Manually calculated frame-level traits with the result of
         `compute_plant_traits` and the fixture
        - Manually calculated plant-level (batch) traits with the result of
         `compute_batch_traits` and the fixture
    """

    # Load the data
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

    canola_computed_traits = pipeline.compute_plant_traits(canola)
    canola_traits_csv = pd.read_csv(canola_traits_csv)
    canola_batch_csv = pd.read_csv(canola_batch_traits_csv)

    soy_computed_traits = pipeline.compute_plant_traits(soy)
    soy_traits_csv = pd.read_csv(soy_traits_csv)
    soy_batch_csv = pd.read_csv(soy_batch_traits_csv)

    computed_batch_traits = pipeline.compute_batch_traits([canola, soy])
    batch_traits_csv = pd.concat([canola_batch_csv, soy_batch_csv], ignore_index=True)

    # Check shape of computed traits, computed batch traits, and dicot fixtures.
    assert canola_computed_traits.shape == (72, 117)
    assert canola_traits_csv.shape == (72, 117)

    assert soy_computed_traits.shape == (72, 117)
    assert soy_traits_csv.shape == (72, 117)

    assert computed_batch_traits.shape == (2, 1036)
    assert batch_traits_csv.shape == (2, 1036)

    # Compare computed traits dataframe and dicot fixtures.
    pd.testing.assert_frame_equal(
        canola_computed_traits.iloc[:, 1:],
        canola_traits_csv.iloc[:, 1:],
        check_exact=False,
        atol=1e-7,
    )
    pd.testing.assert_frame_equal(
        soy_computed_traits.iloc[:, 1:],
        soy_traits_csv.iloc[:, 1:],
        check_exact=False,
        atol=1e-7,
    )

    angle_traits = (
        "primary_angle_proximal",
        "primary_angle_distal",
        "lateral_angles_distal",
        "lateral_angles_proximal",
    )

    numeric_types = (int, np.integer, float, np.floating)
    summary_suffixes = [
        "_min",
        "_max",
        "_mean",
        "_median",
        "_std",
        "_p5",
        "_p25",
        "_p75",
        "_p95",
    ]

    dicots = {"canola": canola, "soy": soy}
    all_dicot_traits_df = []

    for dicot in dicots:
        traits_records = []
        for frame_idx in range(72):
            if dicot == "canola":
                computed_traits = canola_computed_traits
                traits_csv = canola_traits_csv
            else:
                computed_traits = soy_computed_traits
                traits_csv = soy_traits_csv

            # Construct a dictionary of manually calculated traits.
            trait_dict = {
                "plant_name": dicot,
                "frame_idx": frame_idx,
                "primary_pts": dicots[dicot].get_primary_points(frame_idx),
                "lateral_pts": dicots[dicot].get_lateral_points(frame_idx),
            }
            trait_dict["primary_max_length_pts"] = get_max_length_pts(
                trait_dict["primary_pts"]
            )
            trait_dict["lateral_count"] = get_count(trait_dict["lateral_pts"])
            trait_dict["lateral_proximal_node_inds"] = get_node_ind(
                trait_dict["lateral_pts"], proximal=True
            )
            trait_dict["lateral_distal_node_inds"] = get_node_ind(
                trait_dict["lateral_pts"], proximal=False
            )
            trait_dict["lateral_lengths"] = get_root_lengths(trait_dict["lateral_pts"])
            trait_dict["lateral_base_pts"] = get_bases(trait_dict["lateral_pts"])
            trait_dict["lateral_tip_pts"] = get_tips(trait_dict["lateral_pts"])
            trait_dict["pts_all_array"] = get_all_pts_array(
                trait_dict["primary_max_length_pts"], trait_dict["lateral_pts"]
            )
            trait_dict["pts_list"] = join_pts(
                trait_dict["primary_max_length_pts"], trait_dict["lateral_pts"]
            )
            trait_dict["root_widths"] = get_root_widths(
                trait_dict["primary_max_length_pts"],
                trait_dict["lateral_pts"],
            )
            trait_dict["primary_proximal_node_ind"] = get_node_ind(
                trait_dict["primary_max_length_pts"], proximal=True
            )
            trait_dict["primary_distal_node_ind"] = get_node_ind(
                trait_dict["primary_max_length_pts"], proximal=False
            )
            trait_dict["primary_length"] = get_root_lengths(
                trait_dict["primary_max_length_pts"]
            )
            trait_dict["primary_base_pt"] = get_bases(
                trait_dict["primary_max_length_pts"]
            )
            trait_dict["primary_tip_pt"] = get_tips(
                trait_dict["primary_max_length_pts"]
            )
            trait_dict["lateral_angles_proximal"] = get_root_angle(
                trait_dict["lateral_pts"],
                trait_dict["lateral_proximal_node_inds"],
                proximal=True,
                base_ind=0,
            )
            trait_dict["lateral_angles_distal"] = get_root_angle(
                trait_dict["lateral_pts"],
                trait_dict["lateral_distal_node_inds"],
                proximal=False,
                base_ind=0,
            )
            trait_dict["lateral_base_xs"] = get_base_xs(trait_dict["lateral_base_pts"])
            trait_dict["lateral_base_ys"] = get_base_ys(trait_dict["lateral_base_pts"])
            trait_dict["lateral_tip_xs"] = get_tip_xs(trait_dict["lateral_tip_pts"])
            trait_dict["lateral_tip_ys"] = get_tip_ys(trait_dict["lateral_tip_pts"])
            trait_dict["ellipse"] = fit_ellipse(trait_dict["pts_all_array"])
            trait_dict["bounding_box"] = get_bbox(trait_dict["pts_all_array"])
            trait_dict["convex_hull"] = get_convhull(trait_dict["pts_all_array"])
            trait_dict["scanline_intersection_counts"] = count_scanline_intersections(
                trait_dict["pts_list"]
            )
            trait_dict["primary_angle_proximal"] = get_root_angle(
                trait_dict["primary_max_length_pts"],
                trait_dict["primary_proximal_node_ind"],
                proximal=True,
                base_ind=0,
            )
            trait_dict["primary_angle_distal"] = get_root_angle(
                trait_dict["primary_max_length_pts"],
                trait_dict["primary_distal_node_ind"],
                proximal=False,
                base_ind=0,
            )
            trait_dict["base_ct_density"] = get_base_ct_density(
                trait_dict["primary_length"], trait_dict["lateral_base_pts"]
            )
            trait_dict["network_length"] = get_network_length(
                trait_dict["primary_length"], trait_dict["lateral_lengths"]
            )
            trait_dict["primary_base_pt_y"] = get_base_ys(trait_dict["primary_base_pt"])
            trait_dict["primary_tip_pt_y"] = get_tip_ys(trait_dict["primary_tip_pt"])
            trait_dict["primary_base_tip_dist"] = get_base_tip_dist(
                trait_dict["primary_base_pt"], trait_dict["primary_tip_pt"]
            )
            trait_dict["base_length"] = get_base_length(trait_dict["lateral_base_ys"])
            trait_dict["ellipse_a"] = get_ellipse_a(trait_dict["ellipse"])
            trait_dict["ellipse_b"] = get_ellipse_b(trait_dict["ellipse"])
            trait_dict["ellipse_ratio"] = get_ellipse_ratio(trait_dict["ellipse"])
            trait_dict["network_length_lower"] = get_network_distribution(
                trait_dict["pts_list"], trait_dict["bounding_box"]
            )
            trait_dict["network_width_depth_ratio"] = get_network_width_depth_ratio(
                trait_dict["bounding_box"]
            )
            trait_dict["chull_perimeter"] = get_chull_perimeter(
                trait_dict["convex_hull"]
            )
            trait_dict["chull_area"] = get_chull_area(trait_dict["convex_hull"])
            trait_dict["chull_max_width"] = get_chull_max_width(
                trait_dict["convex_hull"]
            )
            trait_dict["chull_max_height"] = get_chull_max_height(
                trait_dict["convex_hull"]
            )
            trait_dict["chull_line_lengths"] = get_chull_line_lengths(
                trait_dict["convex_hull"]
            )
            trait_dict["scanline_last_ind"] = get_scanline_last_ind(
                trait_dict["scanline_intersection_counts"]
            )
            trait_dict["scanline_first_ind"] = get_scanline_first_ind(
                trait_dict["scanline_intersection_counts"]
            )
            trait_dict["base_median_ratio"] = get_base_median_ratio(
                trait_dict["lateral_base_ys"], trait_dict["primary_tip_pt_y"]
            )
            trait_dict["curve_index"] = get_curve_index(
                trait_dict["primary_length"], trait_dict["primary_base_tip_dist"]
            )
            trait_dict["base_length_ratio"] = get_base_length_ratio(
                trait_dict["primary_length"], trait_dict["base_length"]
            )
            trait_dict["network_distribution_ratio"] = get_network_distribution_ratio(
                trait_dict["network_length"], trait_dict["network_length_lower"]
            )
            trait_dict["network_solidity"] = get_network_solidity(
                trait_dict["network_length"], trait_dict["chull_area"]
            )

            # Add summary traits to trait dict.
            for trait in pipeline.summary_traits:
                X = np.atleast_1d(trait_dict[trait])
                if len(X) == 0 or np.all(np.isnan(X)):
                    trait_summary_dict = {
                        f"{trait}_min": np.nan,
                        f"{trait}_max": np.nan,
                        f"{trait}_mean": np.nan,
                        f"{trait}_median": np.nan,
                        f"{trait}_std": np.nan,
                        f"{trait}_p5": np.nan,
                        f"{trait}_p25": np.nan,
                        f"{trait}_p75": np.nan,
                        f"{trait}_p95": np.nan,
                    }
                elif np.issubdtype(X.dtype, np.number):
                    trait_summary_dict = {
                        f"{trait}_min": np.nanmin(X),
                        f"{trait}_max": np.nanmax(X),
                        f"{trait}_mean": np.nanmean(X),
                        f"{trait}_median": np.nanmedian(X),
                        f"{trait}_std": np.nanstd(X),
                        f"{trait}_p5": np.nanpercentile(X, 5),
                        f"{trait}_p25": np.nanpercentile(X, 25),
                        f"{trait}_p75": np.nanpercentile(X, 75),
                        f"{trait}_p95": np.nanpercentile(X, 95),
                    }
                else:
                    trait_summary_dict = {
                        f"{trait}_min": np.nan,
                        f"{trait}_max": np.nan,
                        f"{trait}_mean": np.nan,
                        f"{trait}_median": np.nan,
                        f"{trait}_std": np.nan,
                        f"{trait}_p5": np.nan,
                        f"{trait}_p25": np.nan,
                        f"{trait}_p75": np.nan,
                        f"{trait}_p95": np.nan,
                    }

                trait_dict.update(trait_summary_dict)

            csv_traits_list = pipeline.csv_traits

            angle_traits = (
                "primary_angle_proximal",
                "primary_angle_distal",
                "lateral_angles_distal",
                "lateral_angles_proximal",
            )

            # Type and range check.
            for trait in trait_dict:
                if trait not in csv_traits_list:
                    continue

                trait_values = [
                    trait_dict[trait],
                    computed_traits[trait][frame_idx],
                    traits_csv[trait][frame_idx],
                ]
                # Angle traits are constrained between 0 and 180 (inclusive).
                if (
                    (trait in csv_traits_list)
                    and trait.startswith(angle_traits)
                    and not trait.endswith("_std")
                ):
                    for trait_value in trait_values:
                        assert np.isnan(trait_value) or (
                            (0 <= trait_value <= 180)
                            and isinstance(trait_dict[trait], numeric_types)
                        )
                # All other trait must be nonnegative and have type numeric (or np.nan).
                else:
                    for trait_value in trait_values:
                        assert np.isnan(trait_value) or (
                            (trait_value >= 0)
                            and isinstance(
                                trait_value,
                                numeric_types,
                            )
                        )

            # Compare manually calculated traits to computed traits and the fixtures.
            for trait_idx in range(len(pipeline.traits)):
                trait = pipeline.traits[trait_idx]
                if trait.include_in_csv and trait.scalar:
                    assert np.isclose(
                        trait_dict[trait.name],
                        computed_traits[trait.name][frame_idx],
                        equal_nan=True,
                    )
                    assert np.isclose(
                        trait_dict[trait.name],
                        traits_csv[trait.name][frame_idx],
                        equal_nan=True,
                    )
                elif trait.include_in_csv and not trait.scalar:
                    for suffix in summary_suffixes:
                        assert np.isclose(
                            trait_dict[trait.name + suffix],
                            computed_traits[trait.name + suffix][frame_idx],
                            equal_nan=True,
                        )
                        assert np.isclose(
                            trait_dict[trait.name + suffix],
                            traits_csv[trait.name + suffix][frame_idx],
                            equal_nan=True,
                        )
                else:
                    assert trait.name not in computed_traits.columns
                    assert trait.name not in traits_csv.columns

            # Construct traits dataframe row by row, with metadata.
            temp_dict = {
                "plant_name": dicots[dicot].series_name,
                "frame_idx": frame_idx,
            }
            for trait in pipeline.csv_traits:
                temp_dict[trait] = trait_dict[trait]

            traits_records.append(temp_dict)

        # Create a dataframe of traits for all frames.
        curr_dicot_traits_df = pd.DataFrame.from_records(
            traits_records, columns=computed_traits.columns
        )

        all_dicot_traits_df.append(curr_dicot_traits_df)

        # Compare manually calculated traits df with fixture and computed df.
        pd.testing.assert_frame_equal(
            curr_dicot_traits_df.iloc[:, 1:],
            computed_traits.iloc[:, 1:],
            check_exact=False,
            atol=1e-7,
        )
        pd.testing.assert_frame_equal(
            curr_dicot_traits_df.iloc[:, 1:],
            traits_csv.iloc[:, 1:],
            check_exact=False,
            atol=1e-7,
        )

    # Combine traits dataframes and aggregate to obtain batch traits.
    batch_df = pd.concat(all_dicot_traits_df, ignore_index=True).drop(
        columns={"frame_idx"}
    )

    agg_funcs = [
        lambda x: np.nanmin(x),
        lambda x: np.nanmax(x),
        lambda x: np.nanmean(x),
        lambda x: np.nanmedian(x),
        lambda x: np.nanstd(x),
        lambda x: np.nanpercentile(x, 5),
        lambda x: np.nanpercentile(x, 25),
        lambda x: np.nanpercentile(x, 75),
        lambda x: np.nanpercentile(x, 95),
    ]

    batch_df = batch_df.groupby("plant_name").agg(agg_funcs)
    batch_df.columns = ["_".join(map(str, col)).strip() for col in batch_df.columns]

    colname_update = {
        "<lambda_0>": "min",
        "<lambda_1>": "max",
        "<lambda_2>": "mean",
        "<lambda_3>": "median",
        "<lambda_4>": "std",
        "<lambda_5>": "p5",
        "<lambda_6>": "p25",
        "<lambda_7>": "p75",
        "<lambda_8>": "p95",
    }

    batch_df.columns = (
        batch_df.columns.to_series().replace(colname_update, regex=True).values
    )

    # Sort batch dataframes before comparing.
    batch_df = batch_df.reset_index().sort_values("plant_name").reset_index(drop=True)
    computed_batch_traits = computed_batch_traits.sort_values("plant_name")
    batch_traits_csv = batch_traits_csv.sort_values("plant_name")

    # Range check over the batch traits.
    for col in batch_df.columns:
        if col == "plant_name":
            continue
        # Angle traits are constrained between 0 and 180 (inclusive).
        elif col.startswith(angle_traits) and not col.endswith("_std"):
            assert (
                ((batch_df[col] >= 0) & (batch_df[col] <= 180)) | (batch_df[col].isna())
            ).all()
            assert (
                (
                    (computed_batch_traits[col] >= 0)
                    & (computed_batch_traits[col] <= 180)
                )
                | (batch_df[col].isna())
            ).all()
            assert (
                ((batch_traits_csv[col] >= 0) & (batch_traits_csv[col] <= 180))
                | (batch_df[col].isna())
            ).all()
        else:
            # All other traits must be nonnegative.
            assert ((batch_df[col] >= 0) | (batch_df[col].isna())).all()
            assert (
                (computed_batch_traits[col] >= 0) | (computed_batch_traits[col].isna())
            ).all()
            assert ((batch_traits_csv[col] >= 0) | (batch_traits_csv[col].isna())).all()

    pd.testing.assert_frame_equal(batch_df, computed_batch_traits, check_exact=False)
    pd.testing.assert_frame_equal(
        batch_df.iloc[:, 1:],
        batch_traits_csv.iloc[:, 1:],
        check_exact=False,
        check_like=True,
        atol=1e-7,
    )


def test_younger_monocot_pipeline(
    rice_folder,
    rice_3do_0K9E8B1_traits_csv,
    rice_3do_YR39SJX_traits_csv,
    rice_3do_batch_traits_csv,
):
    all_slp_paths = sr.find_all_slp_paths(rice_folder)

    # List of length 2 containing 2 Series objects.
    all_series = sr.load_series_from_slps(slp_paths=all_slp_paths, h5s=True)

    series_YR39SJX = [
        series for series in all_series if series.series_name == "YR39SJX"
    ][0]
    series_0K9E8BI = [
        series for series in all_series if series.series_name == "0K9E8BI"
    ][0]

    pipeline = YoungerMonocotPipeline()

    traits_computed_0K9E8BI = pipeline.compute_plant_traits(series_0K9E8BI)
    traits_computed_YR39SJX = pipeline.compute_plant_traits(series_YR39SJX)
    traits_csv_0K9E8BI = pd.read_csv(rice_3do_0K9E8B1_traits_csv)
    traits_csv_YR39SJX = pd.read_csv(rice_3do_YR39SJX_traits_csv)

    batch_traits_computed = pipeline.compute_batch_traits(
        [series_0K9E8BI, series_YR39SJX]
    )
    batch_traits_csv = pd.read_csv(rice_3do_batch_traits_csv)

    # Shape check.
    assert traits_computed_0K9E8BI.shape == (72, 104)
    assert traits_computed_YR39SJX.shape == (72, 104)
    assert batch_traits_computed.shape == (2, 919)

    expected_dtypes = (int, float, np.integer, np.floating)

    monocot_dfs = {}

    for series in all_series:
        traits_records = []
        for frame_idx in range(72):
            # Calculate traits.
            trait_dict = {
                "plant_name": series.series_name,
                "frame_idx": frame_idx,
                "primary_pts": series.get_primary_points(frame_idx),
                "crown_pts": series.get_crown_points(frame_idx),
            }

            trait_dict["primary_max_length_pts"] = get_max_length_pts(
                trait_dict["primary_pts"]
            )
            trait_dict["pts_all_array"] = get_all_pts_array(trait_dict["crown_pts"])
            trait_dict["crown_count"] = get_count(trait_dict["crown_pts"])
            trait_dict["crown_proximal_node_inds"] = get_node_ind(
                trait_dict["crown_pts"], proximal=True
            )
            trait_dict["crown_distal_node_inds"] = get_node_ind(
                trait_dict["crown_pts"], proximal=False
            )
            trait_dict["crown_lengths"] = get_root_lengths(trait_dict["crown_pts"])
            trait_dict["crown_base_pts"] = get_bases(trait_dict["crown_pts"])
            trait_dict["crown_tip_pts"] = get_tips(trait_dict["crown_pts"])
            trait_dict["scanline_intersection_counts"] = count_scanline_intersections(
                trait_dict["crown_pts"],
                height=pipeline.img_height,
                n_line=pipeline.n_scanlines,
            )
            trait_dict["crown_angles_distal"] = get_root_angle(
                trait_dict["crown_pts"],
                trait_dict["crown_distal_node_inds"],
                proximal=False,
                base_ind=0,
            )
            trait_dict["crown_angles_proximal"] = get_root_angle(
                trait_dict["crown_pts"],
                trait_dict["crown_proximal_node_inds"],
                proximal=True,
                base_ind=0,
            )
            trait_dict["bounding_box"] = get_bbox(trait_dict["pts_all_array"])
            trait_dict["network_length_lower"] = get_network_distribution(
                trait_dict["crown_pts"],
                trait_dict["bounding_box"],
                fraction=pipeline.network_fraction,
            )
            trait_dict["ellipse"] = fit_ellipse(trait_dict["pts_all_array"])
            trait_dict["convex_hull"] = get_convhull(trait_dict["pts_all_array"])
            trait_dict["primary_proximal_node_ind"] = get_node_ind(
                trait_dict["primary_max_length_pts"], proximal=True
            )
            trait_dict["primary_angle_proximal"] = get_root_angle(
                trait_dict["primary_max_length_pts"],
                trait_dict["primary_proximal_node_ind"],
                proximal=True,
                base_ind=0,
            )
            trait_dict["primary_distal_node_ind"] = get_node_ind(
                trait_dict["primary_max_length_pts"], proximal=False
            )
            trait_dict["primary_angle_distal"] = get_root_angle(
                trait_dict["primary_max_length_pts"],
                trait_dict["primary_distal_node_ind"],
                proximal=False,
                base_ind=0,
            )
            trait_dict["primary_length"] = get_root_lengths(
                trait_dict["primary_max_length_pts"]
            )
            trait_dict["primary_base_pt"] = get_bases(
                trait_dict["primary_max_length_pts"]
            )
            trait_dict["primary_tip_pt"] = get_tips(
                trait_dict["primary_max_length_pts"]
            )
            trait_dict["crown_tip_xs"] = get_tip_xs(trait_dict["crown_tip_pts"])
            trait_dict["crown_tip_ys"] = get_tip_ys(trait_dict["crown_tip_pts"])
            trait_dict["network_length"] = get_network_length(
                trait_dict["crown_lengths"]
            )
            trait_dict["network_distribution_ratio"] = get_network_distribution_ratio(
                trait_dict["network_length"], trait_dict["network_length_lower"]
            )
            trait_dict["crown_base_tip_dists"] = get_base_tip_dist(
                trait_dict["crown_base_pts"], trait_dict["crown_tip_pts"]
            )
            trait_dict["crown_curve_indices"] = get_curve_index(
                trait_dict["crown_lengths"], trait_dict["crown_base_tip_dists"]
            )
            trait_dict["primary_tip_pt_y"] = get_tip_ys(trait_dict["primary_tip_pt"])
            trait_dict["ellipse_a"] = get_ellipse_a(trait_dict["ellipse"])
            trait_dict["ellipse_b"] = get_ellipse_b(trait_dict["ellipse"])
            trait_dict["network_width_depth_ratio"] = get_network_width_depth_ratio(
                trait_dict["bounding_box"]
            )
            trait_dict["chull_perimeter"] = get_chull_perimeter(
                trait_dict["convex_hull"]
            )
            trait_dict["chull_area"] = get_chull_area(trait_dict["convex_hull"])
            trait_dict["chull_max_width"] = get_chull_max_width(
                trait_dict["convex_hull"]
            )
            trait_dict["chull_max_height"] = get_chull_max_height(
                trait_dict["convex_hull"]
            )
            trait_dict["chull_line_lengths"] = get_chull_line_lengths(
                trait_dict["convex_hull"]
            )
            trait_dict["primary_base_tip_dist"] = get_base_tip_dist(
                trait_dict["primary_base_pt"], trait_dict["primary_tip_pt"]
            )
            trait_dict["curve_index"] = get_curve_index(
                trait_dict["primary_length"], trait_dict["primary_base_tip_dist"]
            )
            trait_dict["ellipse_ratio"] = get_ellipse_ratio(trait_dict["ellipse"])
            trait_dict["scanline_last_ind"] = get_scanline_last_ind(
                trait_dict["scanline_intersection_counts"]
            )
            trait_dict["scanline_first_ind"] = get_scanline_first_ind(
                trait_dict["scanline_intersection_counts"]
            )
            trait_dict["network_solidity"] = get_network_solidity(
                trait_dict["network_length"], trait_dict["chull_area"]
            )

            # Add summary traits to traits dict.
            for trait in pipeline.summary_traits:
                X = np.atleast_1d(trait_dict[trait])
                if len(X) == 0 or np.all(np.isnan(X)):
                    trait_summary_dict = {
                        f"{trait}_min": np.nan,
                        f"{trait}_max": np.nan,
                        f"{trait}_mean": np.nan,
                        f"{trait}_median": np.nan,
                        f"{trait}_std": np.nan,
                        f"{trait}_p5": np.nan,
                        f"{trait}_p25": np.nan,
                        f"{trait}_p75": np.nan,
                        f"{trait}_p95": np.nan,
                    }
                elif np.issubdtype(X.dtype, np.number):
                    trait_summary_dict = {
                        f"{trait}_min": np.nanmin(X),
                        f"{trait}_max": np.nanmax(X),
                        f"{trait}_mean": np.nanmean(X),
                        f"{trait}_median": np.nanmedian(X),
                        f"{trait}_std": np.nanstd(X),
                        f"{trait}_p5": np.nanpercentile(X, 5),
                        f"{trait}_p25": np.nanpercentile(X, 25),
                        f"{trait}_p75": np.nanpercentile(X, 75),
                        f"{trait}_p95": np.nanpercentile(X, 95),
                    }
                else:
                    trait_summary_dict = {
                        f"{trait}_min": np.nan,
                        f"{trait}_max": np.nan,
                        f"{trait}_mean": np.nan,
                        f"{trait}_median": np.nan,
                        f"{trait}_std": np.nan,
                        f"{trait}_p5": np.nan,
                        f"{trait}_p25": np.nan,
                        f"{trait}_p75": np.nan,
                        f"{trait}_p95": np.nan,
                    }

                trait_dict.update(trait_summary_dict)

            angle_traits = (
                "primary_angle_proximal",
                "primary_angle_distal",
                "crown_angles_distal",
                "crown_angles_proximal",
            )

            ratio_traits = ("curve_index", "crown_curve_indices")

            # Type and range check for traits at the current frame.
            for trait in pipeline.csv_traits:

                if trait in {"plant_name", "frame_idx"}:
                    continue

                # Type check.
                assert isinstance(trait_dict[trait], expected_dtypes)

                # No range check for standard deviation.
                if trait.endswith("_std"):
                    continue

                # All traits must be nonnegative.
                assert (trait_dict[trait] >= 0) or np.isnan(trait_dict[trait])

                # Angle traits must be in range [0, 180].
                if trait.startswith(angle_traits):
                    assert (0 <= trait_dict[trait] <= 180) or np.isnan(
                        trait_dict[trait]
                    )

                # Ratio traits must be in range [0, 1].
                if trait.startswith(ratio_traits):
                    assert (0 <= trait_dict[trait] <= 1) or np.isnan(trait_dict[trait])

            # Construct traits dataframe row by row, with metadata.
            csv_traits_dict = {
                "plant_name": series.series_name,
                "frame_idx": frame_idx,
            }
            for trait in pipeline.csv_traits:
                csv_traits_dict[trait] = trait_dict[trait]

            traits_records.append(csv_traits_dict)

        curr_monocot_df = pd.DataFrame.from_records(traits_records)
        monocot_dfs[series.series_name] = curr_monocot_df

    # Sample 0K9E8BI: Manual calculation compared to computed pipeline output.
    pd.testing.assert_frame_equal(
        monocot_dfs["0K9E8BI"],
        traits_computed_0K9E8BI,
        check_exact=False,
        check_dtype=True,
    )

    # Sample 0K9E8BI: Manual calculation compared to csv fixture.
    pd.testing.assert_frame_equal(
        monocot_dfs["0K9E8BI"],
        traits_csv_0K9E8BI,
        check_exact=False,
        check_dtype=True,
    )

    # Sample 0K9E8BI: Computed pipeline output compared to csv fixture.
    pd.testing.assert_frame_equal(
        traits_computed_0K9E8BI,
        traits_csv_0K9E8BI,
        check_exact=False,
        check_dtype=True,
    )

    # Sample YR39SJX: Manual calculation compared to computed pipeline output.
    pd.testing.assert_frame_equal(
        monocot_dfs["YR39SJX"],
        traits_computed_YR39SJX,
        check_exact=False,
        check_dtype=True,
    )

    # Sample YR39SJX: Manual calculation compared to csv fixture.
    pd.testing.assert_frame_equal(
        monocot_dfs["YR39SJX"],
        traits_computed_YR39SJX,
        check_exact=False,
        check_dtype=True,
    )

    # Sample YR39SJX: Computed pipeline output compared to csv fixture.
    pd.testing.assert_frame_equal(
        traits_computed_YR39SJX,
        traits_csv_YR39SJX,
        check_exact=False,
        check_dtype=True,
    )

    # Combine traits dataframes and aggregate to obtain batch traits.
    batch_traits_manual = pd.concat(monocot_dfs.values(), ignore_index=True).drop(
        columns={"frame_idx"}
    )

    agg_funcs = [
        lambda x: np.nanmin(x),
        lambda x: np.nanmax(x),
        lambda x: np.nanmean(x),
        lambda x: np.nanmedian(x),
        lambda x: np.nanstd(x),
        lambda x: np.nanpercentile(x, 5),
        lambda x: np.nanpercentile(x, 25),
        lambda x: np.nanpercentile(x, 75),
        lambda x: np.nanpercentile(x, 95),
    ]

    batch_traits_manual = batch_traits_manual.groupby("plant_name").agg(agg_funcs)
    batch_traits_manual.columns = [
        "_".join(map(str, col)).strip() for col in batch_traits_manual.columns
    ]

    colname_update = {
        "<lambda_0>": "min",
        "<lambda_1>": "max",
        "<lambda_2>": "mean",
        "<lambda_3>": "median",
        "<lambda_4>": "std",
        "<lambda_5>": "p5",
        "<lambda_6>": "p25",
        "<lambda_7>": "p75",
        "<lambda_8>": "p95",
    }

    batch_traits_manual.columns = (
        batch_traits_manual.columns.to_series()
        .replace(colname_update, regex=True)
        .values
    )

    # Sort batch dataframes before comparing.
    batch_traits_manual = batch_traits_manual.reset_index().sort_values("plant_name")
    batch_traits_computed = batch_traits_computed.sort_values("plant_name")
    batch_traits_csv = batch_traits_csv.sort_values("plant_name")

    # Compare manual batch traits calculation and computed pipeline output.
    pd.testing.assert_frame_equal(
        batch_traits_computed,
        batch_traits_manual,
        check_exact=False,
        atol=1e-8,
        check_dtype=True,
    )

    # Compare computed pipeline output and csv fixture.
    pd.testing.assert_frame_equal(
        batch_traits_computed,
        batch_traits_csv,
        check_exact=False,
        atol=1e-8,
        check_dtype=True,
    )


def test_older_monocot_pipeline(
    rice_10do_folder, rice_10do_traits_csv, rice_10do_batch_traits_csv
):

    # Find slp paths in folder
    slp_paths = find_all_slp_paths(rice_10do_folder)
    assert len(slp_paths) == 1

    # Load series from slps
    all_series = load_series_from_slps(slp_paths=slp_paths, h5s=True, csv_path=None)
    assert len(all_series) == 1

    # Get first series
    rice_series = all_series[0]

    pipeline = OlderMonocotPipeline()
    computed_traits = pipeline.compute_plant_traits(rice_series)
    computed_batch_traits = pipeline.compute_batch_traits(all_series)

    traits_csv = pd.read_csv(rice_10do_traits_csv)
    batch_traits_csv = pd.read_csv(rice_10do_batch_traits_csv)

    # Dataframe shape assertions
    assert computed_traits.shape == (72, 102)
    assert computed_batch_traits.shape == (1, 901)

    assert traits_csv.shape == (72, 102)
    assert batch_traits_csv.shape == (1, 901)

    expected_dtypes = (int, float, np.integer, np.floating)

    monocot_dfs = {}

    for series in all_series:
        traits_records = []
        for frame_idx in range(72):
            # Calculate traits.
            trait_dict = {
                "plant_name": series.series_name,
                "frame_idx": frame_idx,
                "crown_pts": series.get_crown_points(frame_idx),
            }

            trait_dict["crown_count"] = get_count(trait_dict["crown_pts"])
            trait_dict["crown_proximal_node_inds"] = get_node_ind(
                trait_dict["crown_pts"], proximal=True
            )
            trait_dict["crown_distal_node_inds"] = get_node_ind(
                trait_dict["crown_pts"], proximal=False
            )
            trait_dict["crown_lengths"] = get_root_lengths(trait_dict["crown_pts"])
            trait_dict["crown_base_pts"] = get_bases(trait_dict["crown_pts"])
            trait_dict["crown_tip_pts"] = get_tips(trait_dict["crown_pts"])
            trait_dict["scanline_intersection_counts"] = count_scanline_intersections(
                trait_dict["crown_pts"],
                height=pipeline.img_height,
                n_line=pipeline.n_scanlines,
            )
            trait_dict["crown_angles_distal"] = get_root_angle(
                trait_dict["crown_pts"],
                trait_dict["crown_distal_node_inds"],
                proximal=False,
                base_ind=0,
            )
            trait_dict["crown_angles_proximal"] = get_root_angle(
                trait_dict["crown_pts"],
                trait_dict["crown_proximal_node_inds"],
                proximal=True,
                base_ind=0,
            )
            trait_dict["bounding_box"] = get_bbox(trait_dict["crown_pts"])
            trait_dict["network_length_lower"] = get_network_distribution(
                trait_dict["crown_pts"],
                trait_dict["bounding_box"],
                fraction=pipeline.network_fraction,
            )
            trait_dict["ellipse"] = fit_ellipse(trait_dict["crown_pts"])
            trait_dict["convex_hull"] = get_convhull(trait_dict["crown_pts"])

            trait_dict["crown_tip_xs"] = get_tip_xs(trait_dict["crown_tip_pts"])
            trait_dict["crown_tip_ys"] = get_tip_ys(trait_dict["crown_tip_pts"])
            trait_dict["network_length"] = get_network_length(
                trait_dict["crown_lengths"]
            )
            trait_dict["network_distribution_ratio"] = get_network_distribution_ratio(
                trait_dict["network_length"], trait_dict["network_length_lower"]
            )
            trait_dict["crown_base_tip_dists"] = get_base_tip_dist(
                trait_dict["crown_base_pts"], trait_dict["crown_tip_pts"]
            )
            trait_dict["crown_curve_indices"] = get_curve_index(
                trait_dict["crown_lengths"], trait_dict["crown_base_tip_dists"]
            )
            trait_dict["ellipse_a"] = get_ellipse_a(trait_dict["ellipse"])
            trait_dict["ellipse_b"] = get_ellipse_b(trait_dict["ellipse"])
            trait_dict["network_width_depth_ratio"] = get_network_width_depth_ratio(
                trait_dict["bounding_box"]
            )
            trait_dict["chull_perimeter"] = get_chull_perimeter(
                trait_dict["convex_hull"]
            )
            trait_dict["chull_area"] = get_chull_area(trait_dict["convex_hull"])
            trait_dict["chull_max_width"] = get_chull_max_width(
                trait_dict["convex_hull"]
            )
            trait_dict["chull_max_height"] = get_chull_max_height(
                trait_dict["convex_hull"]
            )
            trait_dict["chull_line_lengths"] = get_chull_line_lengths(
                trait_dict["convex_hull"]
            )

            trait_dict["ellipse_ratio"] = get_ellipse_ratio(trait_dict["ellipse"])
            trait_dict["scanline_last_ind"] = get_scanline_last_ind(
                trait_dict["scanline_intersection_counts"]
            )
            trait_dict["scanline_first_ind"] = get_scanline_first_ind(
                trait_dict["scanline_intersection_counts"]
            )

            trait_dict["crown_r1_pts"] = get_nodes(
                trait_dict["crown_pts"], node_index=1
            )

            trait_dict["chull_r1_intersection_vectors"] = (
                get_chull_intersection_vectors(
                    trait_dict["crown_base_pts"],
                    trait_dict["crown_r1_pts"],
                    trait_dict["crown_pts"],
                    trait_dict["convex_hull"],
                )
            )

            trait_dict["chull_r1_left_intersection_vector"] = (
                get_chull_intersection_vectors_left(
                    trait_dict["chull_r1_intersection_vectors"]
                )
            )

            trait_dict["chull_r1_right_intersection_vector"] = (
                get_chull_intersection_vectors_right(
                    trait_dict["chull_r1_intersection_vectors"]
                )
            )

            trait_dict["angle_chull_r1_left_intersection_vector"] = (
                get_vector_angles_from_gravity(
                    trait_dict["chull_r1_left_intersection_vector"]
                )
            )

            trait_dict["angle_chull_r1_right_intersection_vector"] = (
                get_vector_angles_from_gravity(
                    trait_dict["chull_r1_right_intersection_vector"]
                )
            )

            trait_dict["chull_areas_r1_intersection"] = (
                get_chull_areas_via_intersection(
                    trait_dict["crown_r1_pts"],
                    trait_dict["crown_pts"],
                    trait_dict["convex_hull"],
                )
            )

            trait_dict["chull_area_above_r1_intersection"] = (
                get_chull_area_via_intersection_above(
                    trait_dict["chull_areas_r1_intersection"]
                )
            )
            trait_dict["chull_area_below_r1_intersection"] = (
                get_chull_area_via_intersection_below(
                    trait_dict["chull_areas_r1_intersection"]
                )
            )

            trait_dict["network_solidity"] = get_network_solidity(
                trait_dict["network_length"], trait_dict["chull_area"]
            )

            # Add summary traits to traits dict.
            for trait in pipeline.summary_traits:
                X = np.atleast_1d(trait_dict[trait])
                if len(X) == 0 or np.all(np.isnan(X)):
                    trait_summary_dict = {
                        f"{trait}_min": np.nan,
                        f"{trait}_max": np.nan,
                        f"{trait}_mean": np.nan,
                        f"{trait}_median": np.nan,
                        f"{trait}_std": np.nan,
                        f"{trait}_p5": np.nan,
                        f"{trait}_p25": np.nan,
                        f"{trait}_p75": np.nan,
                        f"{trait}_p95": np.nan,
                    }
                elif np.issubdtype(X.dtype, np.number):
                    trait_summary_dict = {
                        f"{trait}_min": np.nanmin(X),
                        f"{trait}_max": np.nanmax(X),
                        f"{trait}_mean": np.nanmean(X),
                        f"{trait}_median": np.nanmedian(X),
                        f"{trait}_std": np.nanstd(X),
                        f"{trait}_p5": np.nanpercentile(X, 5),
                        f"{trait}_p25": np.nanpercentile(X, 25),
                        f"{trait}_p75": np.nanpercentile(X, 75),
                        f"{trait}_p95": np.nanpercentile(X, 95),
                    }
                else:
                    trait_summary_dict = {
                        f"{trait}_min": np.nan,
                        f"{trait}_max": np.nan,
                        f"{trait}_mean": np.nan,
                        f"{trait}_median": np.nan,
                        f"{trait}_std": np.nan,
                        f"{trait}_p5": np.nan,
                        f"{trait}_p25": np.nan,
                        f"{trait}_p75": np.nan,
                        f"{trait}_p95": np.nan,
                    }

                trait_dict.update(trait_summary_dict)

            ratio_traits = "crown_curve_indices_mean_median"
            angle_traits = (
                "crown_angles_proximal_p95",
                "crown_angles_proximal_median_p95",
                "angle_chull_r1_left_intersection_vector",
                "angle_chull_r1_left_intersection_vector_p95",
                "angle_chull_r1_right_intersection_vector",
                "angle_chull_r1_right_intersection_vector_p95",
            )

            # Type and range check for traits at the current frame.
            for trait in pipeline.csv_traits:

                if trait in {"plant_name", "frame_idx"}:
                    continue

                # Type check.
                assert isinstance(trait_dict[trait], expected_dtypes)

                # No range check for standard deviation.
                if trait.endswith("_std"):
                    continue

                # All traits must be nonnegative.
                assert (trait_dict[trait] >= 0) or np.isnan(trait_dict[trait])

                # Angle traits must be in range [0, 180].
                if trait.startswith(angle_traits):
                    assert (0 <= trait_dict[trait] <= 180) or np.isnan(
                        trait_dict[trait]
                    )

                # Ratio traits must be in range [0, 1].
                if trait.startswith(ratio_traits):
                    assert (0 <= trait_dict[trait] <= 1) or np.isnan(trait_dict[trait])

            # Construct traits dataframe row by row, with metadata.
            csv_traits_dict = {
                "plant_name": series.series_name,
                "frame_idx": frame_idx,
            }
            for trait in pipeline.csv_traits:
                csv_traits_dict[trait] = trait_dict[trait]

            traits_records.append(csv_traits_dict)

        curr_monocot_df = pd.DataFrame.from_records(traits_records)
        monocot_dfs[series.series_name] = curr_monocot_df

    # Sample 0K9E8BI: Manual calculation compared to computed pipeline output.
    pd.testing.assert_frame_equal(
        monocot_dfs["0K9E8BI"],
        computed_traits,
        check_exact=False,
        check_dtype=True,
    )

    # Sample 0K9E8BI: Manual calculation compared to csv fixture.
    pd.testing.assert_frame_equal(
        monocot_dfs["0K9E8BI"],
        traits_csv,
        check_exact=False,
        check_dtype=True,
    )

    # Sample 0K9E8BI: Computed pipeline output compared to csv fixture.
    pd.testing.assert_frame_equal(
        computed_traits,
        traits_csv,
        check_exact=False,
        check_dtype=True,
    )

    # Combine traits dataframes and aggregate to obtain batch traits.
    batch_traits_manual = pd.concat(monocot_dfs.values(), ignore_index=True).drop(
        columns={"frame_idx"}
    )

    agg_funcs = [
        lambda x: np.nanmin(x),
        lambda x: np.nanmax(x),
        lambda x: np.nanmean(x),
        lambda x: np.nanmedian(x),
        lambda x: np.nanstd(x),
        lambda x: np.nanpercentile(x, 5),
        lambda x: np.nanpercentile(x, 25),
        lambda x: np.nanpercentile(x, 75),
        lambda x: np.nanpercentile(x, 95),
    ]

    batch_traits_manual = batch_traits_manual.groupby("plant_name").agg(agg_funcs)
    batch_traits_manual.columns = [
        "_".join(map(str, col)).strip() for col in batch_traits_manual.columns
    ]

    colname_update = {
        "<lambda_0>": "min",
        "<lambda_1>": "max",
        "<lambda_2>": "mean",
        "<lambda_3>": "median",
        "<lambda_4>": "std",
        "<lambda_5>": "p5",
        "<lambda_6>": "p25",
        "<lambda_7>": "p75",
        "<lambda_8>": "p95",
    }

    batch_traits_manual.columns = (
        batch_traits_manual.columns.to_series()
        .replace(colname_update, regex=True)
        .values
    )

    # Sort batch dataframes before comparing.
    batch_traits_manual = batch_traits_manual.reset_index().sort_values("plant_name")
    computed_batch_traits = computed_batch_traits.sort_values("plant_name")
    batch_traits_csv = batch_traits_csv.sort_values("plant_name")

    # Compare manual batch traits calculation and computed pipeline output.
    pd.testing.assert_frame_equal(
        computed_batch_traits,
        batch_traits_manual,
        check_exact=False,
        atol=1e-8,
        check_dtype=True,
    )

    # Compare computed pipeline output and csv fixture.
    pd.testing.assert_frame_equal(
        computed_batch_traits,
        batch_traits_csv,
        check_exact=False,
        atol=1e-8,
        check_dtype=True,
    )


def test_multiple_dicot_pipeline(
    multiple_arabidopsis_11do_h5,
    multiple_arabidopsis_11do_folder,
    multiple_arabidopsis_11do_csv,
    multiple_arabidopsis_11do_primary_slp,
    multiple_arabidopsis_11do_lateral_slp,
):
    arabidopsis = Series.load(
        series_name="997_1",
        h5_path=multiple_arabidopsis_11do_h5,
        primary_path=multiple_arabidopsis_11do_primary_slp,
        lateral_path=multiple_arabidopsis_11do_lateral_slp,
        csv_path=multiple_arabidopsis_11do_csv,
    )
    arabidopsis_slp_paths = find_all_slp_paths(multiple_arabidopsis_11do_folder)
    arabidopsis_series_all = load_series_from_slps(
        slp_paths=arabidopsis_slp_paths,
        h5s=True,
        csv_path=multiple_arabidopsis_11do_csv,
    )

    pipeline = MultipleDicotPipeline()
    arabidopsis_traits = pipeline.compute_multiple_dicots_traits(arabidopsis)
    all_traits = pipeline.compute_batch_multiple_dicots_traits(arabidopsis_series_all)

    # Dataframe shape assertions
    assert pd.DataFrame([arabidopsis_traits["summary_stats"]]).shape == (1, 315)
    assert all_traits.shape == (4, 316)

    # Dataframe dtype assertions
    expected_all_traits_dtypes = {
        "lateral_count_min": "int64",
        "lateral_count_max": "int64",
    }

    for col, expected_dtype in expected_all_traits_dtypes.items():
        assert np.issubdtype(
            all_traits[col].dtype, np.integer
        ), f"Unexpected dtype for column {col} in all_traits. Expected integer, got {all_traits[col].dtype}"

    # Value range assertions for traits
    assert (
        all_traits["curve_index_median"] >= 0
    ).all(), "curve_index in all_traits contains negative values"

    # Check that series dictionary
    assert isinstance(arabidopsis_traits, dict)
    assert arabidopsis_traits["series"] == "997_1"
    assert arabidopsis_traits["group"] == "997"


def test_primary_root_pipeline(
    canola_folder,
    canola_traits_csv,
    canola_batch_traits_csv,
    soy_folder,
    soy_traits_csv,
    soy_batch_traits_csv,
    rice_folder,
    rice_3do_0K9E8B1_traits_csv,
    rice_3do_YR39SJX_traits_csv,
    rice_3do_batch_traits_csv,
):

    # Dicot data (canola_7do, soy_6do).
    canola_slps = sr.find_all_slp_paths(canola_folder)
    canola = sr.load_series_from_slps(canola_slps, h5s=True)[0]

    soy_slps = sr.find_all_slp_paths(soy_folder)
    soy = sr.load_series_from_slps(soy_slps, h5s=True)[0]

    canola_traits_fixture = pd.read_csv(canola_traits_csv)
    canola_batch_traits_fixture = pd.read_csv(canola_batch_traits_csv)

    soy_traits_fixture = pd.read_csv(soy_traits_csv)
    soy_batch_traits_fixture = pd.read_csv(soy_batch_traits_csv)

    # Younger monocot data (rice_3do).
    rice_slps = sr.find_all_slp_paths(rice_folder)
    rice_all_series = sr.load_series_from_slps(rice_slps)
    rice_YR39SJX = [
        series for series in rice_all_series if series.series_name == "YR39SJX"
    ][0]
    rice_0K9E8BI = [
        series for series in rice_all_series if series.series_name == "0K9E8BI"
    ][0]
    rice_YR39SJX_traits_fixture = pd.read_csv(rice_3do_YR39SJX_traits_csv)
    rice_0K9E8BI_traits_fixture = pd.read_csv(rice_3do_0K9E8B1_traits_csv)
    rice_batch_traits_fixture = pd.read_csv(rice_3do_batch_traits_csv)

    trait_cols = [
        "plant_name",
        "curve_index",
        "primary_angle_distal",
        "primary_angle_proximal",
        "primary_base_tip_dist",
        "primary_length",
        "primary_tip_pt_y",
    ]

    pipeline = PrimaryRootPipeline()

    # Compare computed traits to fixtures.
    canola_computed_traits = pipeline.compute_plant_traits(canola)
    soy_computed_traits = pipeline.compute_plant_traits(soy)
    rice_YR39SJX_computed_traits = pipeline.compute_plant_traits(rice_YR39SJX)
    rice_0K9E8BI_computed_traits = pipeline.compute_plant_traits(rice_0K9E8BI)

    pd.testing.assert_frame_equal(
        canola_computed_traits[trait_cols], canola_traits_fixture[trait_cols]
    )
    pd.testing.assert_frame_equal(
        soy_computed_traits[trait_cols], soy_traits_fixture[trait_cols]
    )
    pd.testing.assert_frame_equal(
        rice_YR39SJX_computed_traits[trait_cols],
        rice_YR39SJX_traits_fixture[trait_cols],
    )
    pd.testing.assert_frame_equal(
        rice_0K9E8BI_computed_traits[trait_cols],
        rice_0K9E8BI_traits_fixture[trait_cols],
    )

    # Compare computed batch traits to fixtures.
    summary_suffixes = [
        "min",
        "max",
        "median",
        "mean",
        "std",
        "p5",
        "p25",
        "p75",
        "p95",
    ]

    # Match each trait name with the summary statistic suffix in a list, execept plant_name.
    batch_trait_cols = [
        "plant_name" if trait == "plant_name" else f"{trait}_{suffix}"
        for trait in trait_cols
        for suffix in summary_suffixes
    ]

    canola_computed_batch_traits = pipeline.compute_batch_traits([canola])
    soy_computed_batch_traits = pipeline.compute_batch_traits([soy])
    rice_computed_batch_traits = pipeline.compute_batch_traits(
        [rice_YR39SJX, rice_0K9E8BI]
    )

    pd.testing.assert_frame_equal(
        canola_computed_batch_traits[batch_trait_cols],
        canola_batch_traits_fixture[batch_trait_cols],
    )
    pd.testing.assert_frame_equal(
        soy_computed_batch_traits[batch_trait_cols],
        soy_batch_traits_fixture[batch_trait_cols],
    )

    # Sort dataframe before comparing since there are 2 samples.
    rice_computed_batch_traits = rice_computed_batch_traits.sort_values(
        "plant_name"
    ).reset_index()
    rice_batch_traits_fixture = rice_batch_traits_fixture.sort_values(
        "plant_name"
    ).reset_index()

    pd.testing.assert_frame_equal(
        rice_computed_batch_traits[batch_trait_cols],
        rice_batch_traits_fixture[batch_trait_cols],
    )


def test_multiple_primary_root_pipeline(
    multiple_arabidopsis_11do_folder,
    multiple_arabidopsis_11do_csv,
    multiple_arabidopsis_11do_batch_traits_MultiplePrimaryRootPipeline,
    multiple_arabidopsis_11do_group_batch_traits_MultiplePrimaryRootPipeline,
):

    all_slps = sr.find_all_slp_paths(multiple_arabidopsis_11do_folder)

    # Load arabidopsis examples.
    all_multiple_dicot_series = sr.load_series_from_slps(
        slp_paths=all_slps, h5s=True, csv_path=multiple_arabidopsis_11do_csv
    )

    assert len(all_multiple_dicot_series) == 4

    multiple_primary_root_pipeline = MultiplePrimaryRootPipeline()

    # Extract each series out in a variable.
    series_997_1 = [
        series for series in all_multiple_dicot_series if series.series_name == "997_1"
    ][0]
    series_7327_2 = [
        series for series in all_multiple_dicot_series if series.series_name == "7327_2"
    ][0]
    series_6039_1 = [
        series for series in all_multiple_dicot_series if series.series_name == "6039_1"
    ][0]
    series_9535_1 = [
        series for series in all_multiple_dicot_series if series.series_name == "9535_1"
    ][0]

    assert series_997_1.qc_fail == 0
    assert series_7327_2.qc_fail == 0
    assert series_6039_1.qc_fail == 1
    assert series_9535_1.qc_fail == 0

    # Compute traits using compute_multiple_dicots_trait for each sample.
    computed_traits_997_1 = (
        multiple_primary_root_pipeline.compute_multiple_primary_roots_traits(
            series_997_1
        )
    )
    computed_traits_7327_2 = (
        multiple_primary_root_pipeline.compute_multiple_primary_roots_traits(
            series_7327_2
        )
    )
    computed_traits_6039_1 = (
        multiple_primary_root_pipeline.compute_multiple_primary_roots_traits(
            series_6039_1
        )
    )
    computed_traits_9535_1 = (
        multiple_primary_root_pipeline.compute_multiple_primary_roots_traits(
            series_9535_1
        )
    )

    # Compute traits per group.
    computed_grouped_traits = (
        multiple_primary_root_pipeline.compute_multiple_primary_roots_traits_for_groups(
            all_multiple_dicot_series
        )
    )

    assert (
        isinstance(computed_grouped_traits, list) and len(computed_grouped_traits) == 3
    )

    assert isinstance(computed_traits_997_1, dict)
    assert computed_traits_997_1["series"] == "997_1"
    assert computed_traits_997_1["group"] == "997"
    assert pd.DataFrame([computed_traits_997_1["traits"]]).shape == (1, 9)
    assert pd.DataFrame([computed_traits_997_1["summary_stats"]]).shape == (1, 81)

    assert isinstance(computed_traits_7327_2, dict)
    assert computed_traits_7327_2["series"] == "7327_2"
    assert computed_traits_7327_2["group"] == "7327"
    assert pd.DataFrame([computed_traits_7327_2["traits"]]).shape == (1, 9)
    assert pd.DataFrame([computed_traits_7327_2["summary_stats"]]).shape == (1, 81)

    assert isinstance(computed_traits_6039_1, dict)
    assert computed_traits_6039_1["series"] == "6039_1"
    assert computed_traits_6039_1["group"] == "6039"
    assert pd.DataFrame([computed_traits_6039_1["traits"]]).shape == (1, 9)
    assert pd.DataFrame([computed_traits_6039_1["summary_stats"]]).shape == (1, 81)

    assert isinstance(computed_traits_9535_1, dict)
    assert computed_traits_9535_1["series"] == "9535_1"
    assert computed_traits_9535_1["group"] == "9535"
    assert pd.DataFrame([computed_traits_6039_1["traits"]]).shape == (1, 9)
    assert pd.DataFrame([computed_traits_9535_1["summary_stats"]]).shape == (1, 81)

    angle_traits = (
        "lateral_angles_proximal",
        "lateral_angles_distal",
        "primary_angle_distal",
        "primary_angle_proximal",
    )

    expected_dtypes = (int, float, np.integer, np.floating)

    multiple_primary_root_pipeline = MultiplePrimaryRootPipeline()

    all_series_summaries = []

    for series in all_multiple_dicot_series:

        # Compute traits for the current series.
        computed_traits = (
            multiple_primary_root_pipeline.compute_multiple_primary_roots_traits(series)
        )

        # Manually create dictionary storing traits for all frames.
        result = {
            "series": str(series.series_name),
            "group": str(series.group),
            "qc_fail": series.qc_fail,
            "traits": {},
            "summary_stats": {},
        }

        aggregated_traits = {}

        for frame in range(len(series)):

            frame_traits = {
                "primary_pts": series.get_primary_points(frame),
                "expected_plant_ct": series.expected_count,
            }

            frame_traits["filtered_primary_pts_with_expected_ct"] = (
                filter_primary_roots_with_unexpected_count(
                    frame_traits["primary_pts"], frame_traits["expected_plant_ct"]
                )
            )

            primary_root_pipeline = PrimaryRootPipeline()

            # Retrive numpy ndarray of filtered_primary_pts (instances, nodes, 2)
            primary_root_instances = frame_traits[
                "filtered_primary_pts_with_expected_ct"
            ]

            for primary_root_inst in primary_root_instances:

                # Get the initial frame traits for this plant using the filtered primary points
                initial_frame_traits = {
                    "primary_pts": primary_root_inst,
                }

                # Use the primary root pipeline to compute the plant traits on this frame
                plant_traits = primary_root_pipeline.compute_frame_traits(
                    initial_frame_traits
                )

                # For each plant's traits in the frame
                for trait_name, trait_value in plant_traits.items():
                    # Not all traits are added to the aggregated traits dictionary
                    if trait_name in primary_root_pipeline.csv_traits_multiple_plants:
                        if trait_name not in aggregated_traits:
                            # Initialize the trait array if it's the first frame
                            aggregated_traits[trait_name] = [np.atleast_1d(trait_value)]
                        else:
                            # Append new trait values for subsequent frames
                            aggregated_traits[trait_name].append(
                                np.atleast_1d(trait_value)
                            )

        # After processing, update the result dictionary with computed traits
        for trait, arrays in aggregated_traits.items():
            aggregated_traits[trait] = np.concatenate(arrays, axis=0)
        result["traits"] = aggregated_traits

        # Compute summary statistics and update result
        summary_stats = {}
        for trait_name, trait_values in aggregated_traits.items():
            trait_stats = get_summary(trait_values, prefix=f"{trait_name}_")
            summary_stats.update(trait_stats)
        result["summary_stats"] = summary_stats

        # Assert manually calculated and computed traits have the same keys.
        result.keys() == computed_traits.keys()

        # Assert manually calculated and computed traits have the same trait names.
        result["traits"].keys() == computed_traits["traits"].keys()

        # Assert manually calculated and computed traits have the same summary trait names.
        result["summary_stats"].keys() == computed_traits["summary_stats"].keys()

        # Check that the trait values for manually calculated traits and computed traits are the same.
        for key in result["traits"].keys():
            curr_trait_val = result["traits"][key]
            if isinstance(curr_trait_val, np.ndarray):
                np.testing.assert_almost_equal(
                    curr_trait_val, computed_traits["traits"][key]
                )
            else:
                assert curr_trait_val == computed_traits[key]

        # Check that the summary trait values for manually calculated traits and computed traits are the same.
        for key in result["summary_stats"].keys():
            assert np.isclose(
                result["summary_stats"][key],
                computed_traits["summary_stats"][key],
                equal_nan=True,
            )

        # Append the current dictionary to the all_series_summaries list.
        all_series_summaries.append(result)

        # Type and Range Check over the traits.
        for key in computed_traits["traits"].keys():

            arr1 = computed_traits["traits"][key]
            arr2 = result["traits"][key]

            # Trait values should be stored as an array.
            assert isinstance(arr1, np.ndarray), "Trait value is not an array."
            assert isinstance(arr2, np.ndarray), "Trait value is not an array."

            # Type check.
            assert np.all(
                [isinstance(x, expected_dtypes) or np.isnan(x) for x in arr1.flat]
            ), "Array contains invalid types."
            assert np.all(
                [isinstance(x, expected_dtypes) or np.isnan(x) for x in arr2.flat]
            ), "Array contains invalid types."

            # Range check.
            if key in angle_traits:
                assert np.all(
                    ((arr1 >= 0) & (arr1 <= 180)) | np.isnan(arr1)
                ), "Angle trait is out of range."
                assert np.all(
                    ((arr2 >= 0) & (arr2 <= 180)) | np.isnan(arr2)
                ), "Angle trait is out of range."

            else:
                assert np.all(
                    (arr1 >= 0) | np.isnan(arr1)
                ), "Array contains negative values."
                assert np.all(
                    (arr2 >= 0) | np.isnan(arr2)
                ), "Array contains negative values."

        # Type and Range Check over the summary traits.
        for key in computed_traits["summary_stats"].keys():
            if key.endswith("_std"):
                continue
            elif key.startswith(angle_traits):
                assert (
                    (result["summary_stats"][key] >= 0)
                    and (result["summary_stats"][key] <= 180)
                ) or result["summary_stats"][key], "Angle trait is out of range."
                assert (
                    (computed_traits["summary_stats"][key] >= 0)
                    and (computed_traits["summary_stats"][key] <= 180)
                ) or computed_traits["summary_stats"][
                    key
                ], "Angle trait is out of range."
            else:
                assert (result["summary_stats"][key] >= 0) or result["summary_stats"][
                    key
                ], f"Trait {key} is a negative value."
                assert (computed_traits["summary_stats"][key] >= 0) or computed_traits[
                    "summary_stats"
                ][key], f"Trait {key} is a negative value."

    # Check batch calculations for all series.

    batch_df_fixture = pd.read_csv(
        multiple_arabidopsis_11do_batch_traits_MultiplePrimaryRootPipeline
    )

    batch_df_rows = []

    for series in all_series_summaries:
        series_summary = {"series_name": series["series"], **series["summary_stats"]}
        batch_df_rows.append(series_summary)

    batch_df = pd.DataFrame(batch_df_rows)

    computed_batch_traits = (
        multiple_primary_root_pipeline.compute_batch_multiple_primary_roots_traits(
            all_series=all_multiple_dicot_series
        )
    )
    assert batch_df.shape == (4, 82)
    assert computed_batch_traits.shape == (4, 82)
    assert batch_df_fixture.shape == (4, 82)

    # Ensure series_name column is of type string. Then, sort dataframes before comparing.
    batch_df["series_name"] = batch_df["series_name"].astype(str)
    computed_batch_traits["series_name"] = computed_batch_traits["series_name"].astype(
        str
    )
    batch_df_fixture["series_name"] = batch_df_fixture["series_name"].astype(str)

    batch_df = batch_df.sort_values(by="series_name").reset_index(drop=True)
    computed_batch_traits = computed_batch_traits.sort_values(
        by="series_name"
    ).reset_index(drop=True)
    batch_df_fixture = batch_df_fixture.sort_values(by="series_name").reset_index(
        drop=True
    )

    pd.testing.assert_frame_equal(
        batch_df,
        computed_batch_traits,
        check_exact=False,
    )
    pd.testing.assert_frame_equal(
        batch_df,
        batch_df_fixture,
        check_exact=False,
    )
    pd.testing.assert_frame_equal(
        computed_batch_traits,
        batch_df_fixture,
        check_exact=False,
    )

    # Check back calculations per group.
    group_batch_df_fixture = pd.read_csv(
        multiple_arabidopsis_11do_group_batch_traits_MultiplePrimaryRootPipeline
    )

    group_batch_df_rows = []

    for series in all_series_summaries:
        if series["qc_fail"] == 1:
            continue
        else:
            series_summary = {"genotype": series["group"], **series["summary_stats"]}
            group_batch_df_rows.append(series_summary)

    group_batch_df = pd.DataFrame(group_batch_df_rows)

    computed_group_batch_traits = multiple_primary_root_pipeline.compute_batch_multiple_primary_roots_traits_for_groups(
        all_multiple_dicot_series
    )
    assert computed_group_batch_traits.shape == (3, 82)
    assert group_batch_df.shape == (3, 82)
    assert group_batch_df_fixture.shape == (3, 82)

    # Ensure genotype column is of type string. Then, sort dataframes before comparing.
    group_batch_df["genotype"] = group_batch_df["genotype"].astype(str)
    computed_group_batch_traits["genotype"] = computed_group_batch_traits[
        "genotype"
    ].astype(str)
    group_batch_df_fixture["genotype"] = group_batch_df_fixture["genotype"].astype(str)

    group_batch_df = group_batch_df.sort_values(by="genotype").reset_index(drop=True)
    computed_group_batch_traits = computed_group_batch_traits.sort_values(
        by="genotype"
    ).reset_index(drop=True)
    group_batch_df_fixture = group_batch_df_fixture.sort_values(
        by="genotype"
    ).reset_index(drop=True)

    pd.testing.assert_frame_equal(group_batch_df, computed_group_batch_traits)
    pd.testing.assert_frame_equal(group_batch_df, group_batch_df_fixture)
    pd.testing.assert_frame_equal(computed_group_batch_traits, group_batch_df_fixture)
