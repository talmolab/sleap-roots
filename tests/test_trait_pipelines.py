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
)
from sleap_roots.series import (
    Series,
    find_all_h5_paths,
    find_all_slp_paths,
    load_series_from_h5s,
    load_series_from_slps,
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

from sleap_roots.summary import get_summary


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
    canola_traits = pipeline.compute_plant_traits(canola)
    soy_traits = pipeline.compute_plant_traits(soy)
    all_traits = pipeline.compute_batch_traits([canola, soy])

    canola_traits_csv = pd.read_csv(canola_traits_csv)
    soy_traits_csv = pd.read_csv(soy_traits_csv)

    canola_batch_traits_csv = pd.read_csv(canola_batch_traits_csv)
    soy_batch_traits_csv = pd.read_csv(soy_batch_traits_csv)

    batch_traits = pipeline.compute_batch_traits([canola, soy])
    batch_traits_csv = pd.concat(
        [canola_batch_traits_csv, soy_batch_traits_csv], ignore_index=True
    )

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

    pd.testing.assert_frame_equal(
        batch_traits, batch_traits_csv, check_exact=False, atol=1e-7
    )


def test_dicot_compute_plant_traits(
    canola_h5,
    soy_h5,
    canola_primary_slp,
    canola_lateral_slp,
    soy_primary_slp,
    soy_lateral_slp,
    canola_traits_csv,
    soy_traits_csv,
):
    def check_summary_traits(traits_iterable, comparison_df, trait_name, frame_idx):
        """
        Helper function for test_dicot_compute_plant_traits

        traits_iterable: The output of a trait calculation if trait is not a scalar
        comparison_df (pd.DataFrame): should be a pandas df of already computed pipeline results
        trait_name (str): The name of a trait
        frame_idx (int): The image frame index
        """
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
        summary_functions = [
            np.nanmin,
            np.nanmax,
            np.nanmean,
            np.nanmedian,
            np.nanstd,
            lambda x: np.nanpercentile(x, 5),
            lambda x: np.nanpercentile(x, 25),
            lambda x: np.nanpercentile(x, 75),
            lambda x: np.nanpercentile(x, 95),
        ]
        summary_dict = dict(zip(summary_suffixes, summary_functions))
        if np.isnan(traits_iterable).all():
            for suffix in summary_dict.keys():
                assert np.isnan(comparison_df[f"{trait_name}{suffix}"][frame_idx]).all()
        else:
            for suffix in summary_dict.keys():
                func = summary_dict[suffix]
                np.testing.assert_almost_equal(
                    func(traits_iterable),
                    comparison_df[f"{trait_name}{suffix}"][frame_idx],
                    decimal=7,
                )

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
    dicot_traits = pipeline.traits
    canola_traits = pipeline.compute_plant_traits(canola)
    soy_traits = pipeline.compute_plant_traits(soy)

    canola_traits_csv = pd.read_csv(canola_traits_csv)
    soy_traits_csv = pd.read_csv(soy_traits_csv)

    dicots = ["canola", "soy"]
    for dicot in dicots:
        for frame_idx in range(72):
            if dicot == "canola":
                lateral_pts = canola.get_lateral_points(frame_idx)
                primary_pts = canola.get_primary_points(frame_idx)
                traits_output = canola_traits
                traits_csv = canola_traits_csv
            else:
                lateral_pts = soy.get_lateral_points(frame_idx)
                primary_pts = soy.get_primary_points(frame_idx)
                traits_output = soy_traits
                traits_csv = soy_traits_csv

            trait_dict = {
                "primary_pts": primary_pts,
                "lateral_pts": lateral_pts,
                "primary_max_length_pts": get_max_length_pts(primary_pts),
                "distal_node_ind": get_node_ind(lateral_pts),
            }
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
            trait_dict["primary_tip_pt_y"] = get_tip_ys(trait_dict["primary_tip_pt"])
            trait_dict["primary_base_tip_dist"] = get_base_tip_dist(
                base_pts=trait_dict["primary_base_pt"],
                tip_pts=trait_dict["primary_tip_pt"],
            )
            trait_dict["root_widths"] = get_root_widths(
                trait_dict["primary_max_length_pts"],
                trait_dict["lateral_pts"],
            )
            trait_dict["pts_list"] = join_pts(
                trait_dict["primary_max_length_pts"], trait_dict["lateral_pts"]
            )
            trait_dict["lateral_base_pts"] = get_bases(trait_dict["lateral_pts"])
            trait_dict["lateral_tip_pts"] = get_tips(trait_dict["lateral_pts"])
            trait_dict["lateral_base_ys"] = get_base_ys(trait_dict["lateral_base_pts"])
            trait_dict["lateral_distal_node_inds"] = get_node_ind(
                trait_dict["lateral_pts"], proximal=False
            )
            trait_dict["base_length"] = get_base_length(trait_dict["lateral_base_ys"])
            trait_dict["lateral_lengths"] = get_root_lengths(trait_dict["lateral_pts"])
            trait_dict["lateral_proximal_node_inds"] = get_node_ind(
                trait_dict["lateral_pts"], proximal=True
            )
            trait_dict["lateral_angles_distal"] = get_root_angle(
                trait_dict["lateral_pts"],
                trait_dict["lateral_distal_node_inds"],
                proximal=False,
            )
            trait_dict["lateral_count"] = get_count(trait_dict["lateral_pts"])
            trait_dict["scanline_intersection_counts"] = count_scanline_intersections(
                trait_dict["pts_list"]
            )
            trait_dict["pts_all_array"] = get_all_pts_array(
                trait_dict["primary_max_length_pts"], trait_dict["lateral_pts"]
            )
            trait_dict["bounding_box"] = get_bbox(trait_dict["pts_all_array"])
            trait_dict["ellipse"] = fit_ellipse(trait_dict["pts_all_array"])
            trait_dict["convex_hull"] = get_convhull(trait_dict["pts_all_array"])
            trait_dict["network_length_lower"] = get_network_distribution(
                trait_dict["pts_list"], trait_dict["bounding_box"]
            )
            trait_dict["network_length"] = get_network_length(
                trait_dict["primary_length"], trait_dict["lateral_lengths"]
            )
            trait_dict["chull_area"] = get_chull_area(trait_dict["convex_hull"])

            ###### trait calculation

            for trait_idx in range(len(dicot_traits)):
                trait = dicot_traits[trait_idx]
                trait_positional_args = [trait_dict[arg] for arg in trait.input_traits]
                trait_value = trait.fn(*trait_positional_args, **trait.kwargs)

                if trait.include_in_csv and not trait.scalar:
                    check_summary_traits(
                        traits_iterable=trait_value,
                        comparison_df=traits_output,
                        trait_name=trait.name,
                        frame_idx=frame_idx,
                    )
                    check_summary_traits(
                        traits_iterable=trait_value,
                        comparison_df=traits_csv,
                        trait_name=trait.name,
                        frame_idx=frame_idx,
                    )

                elif trait.include_in_csv and trait.scalar:
                    assert np.isclose(
                        trait_value,
                        traits_csv[trait.name][frame_idx],
                        equal_nan=True,
                    )
                    assert np.isclose(
                        trait_value,
                        traits_output[trait.name][frame_idx],
                        equal_nan=True,
                    )
                else:
                    assert trait.name not in traits_csv.columns
                    assert trait.name not in traits_output.columns


def test_OlderMonocot_pipeline(rice_main_10do_h5, rice_main_10do_slp):
    rice = Series.load(
        series_name="rice_10do",
        h5_path=rice_main_10do_h5,
        crown_path=rice_main_10do_slp,
    )

    pipeline = OlderMonocotPipeline()
    rice_10dag_traits = pipeline.compute_plant_traits(rice)

    assert rice_10dag_traits.shape == (72, 102)


def test_younger_monocot_pipeline(rice_pipeline_output_folder):
    # Find slp paths in folder
    slp_paths = find_all_slp_paths(rice_pipeline_output_folder)
    assert len(slp_paths) == 4
    # Load series from slps
    rice_series_all = load_series_from_slps(
        slp_paths=slp_paths, h5s=False, csv_path=None
    )
    assert len(rice_series_all) == 2
    # Get first series
    rice_series = rice_series_all[0]
    # Initialize pipeline for younger monocot
    pipeline = YoungerMonocotPipeline()
    # Get traits for the first series using the pipeline
    rice_traits = pipeline.compute_plant_traits(rice_series)
    # Get all traits for all series using the pipeline
    all_traits = pipeline.compute_batch_traits(rice_series_all)

    # Dataframe shape assertions
    assert rice_traits.shape == (72, 104)
    assert all_traits.shape == (2, 919)

    # Dataframe dtype assertions
    expected_rice_traits_dtypes = {
        "frame_idx": "int64",
        "crown_count": "int64",
    }

    expected_all_traits_dtypes = {
        "crown_count_min": "int64",
        "crown_count_max": "int64",
    }

    for col, expected_dtype in expected_rice_traits_dtypes.items():
        assert (
            rice_traits[col].dtype == expected_dtype
        ), f"Unexpected dtype for column {col} in rice_traits"

    for col, expected_dtype in expected_all_traits_dtypes.items():
        assert (
            all_traits[col].dtype == expected_dtype
        ), f"Unexpected dtype for column {col} in all_traits"

    # Value range assertions for traits
    assert (
        rice_traits["curve_index"].fillna(0) >= 0
    ).all(), "curve_index in rice_traits contains negative values"
    assert (
        rice_traits["curve_index"].fillna(0).max() <= 1
    ), "curve_index in rice_traits contains values greater than 1"
    assert (
        all_traits["curve_index_median"] >= 0
    ).all(), "curve_index in all_traits contains negative values"
    assert (
        all_traits["curve_index_median"].max() <= 1
    ), "curve_index in all_traits contains values greater than 1"
    assert (
        all_traits["crown_curve_indices_mean_median"] >= 0
    ).all(), "crown_curve_indices_mean_median in all_traits contains negative values"
    assert (
        all_traits["crown_curve_indices_mean_median"] <= 1
    ).all(), (
        "crown_curve_indices_mean_median in all_traits contains values greater than 1"
    )
    assert (
        (0 <= rice_traits["crown_angles_proximal_p95"])
        & (rice_traits["crown_angles_proximal_p95"] <= 180)
    ).all(), "angle_column in rice_traits contains values out of range [0, 180]"
    assert (
        (0 <= all_traits["crown_angles_proximal_median_p95"])
        & (all_traits["crown_angles_proximal_median_p95"] <= 180)
    ).all(), "angle_column in all_traits contains values out of range [0, 180]"


def test_older_monocot_pipeline(rice_10do_pipeline_output_folder):
    # Find slp paths in folder
    slp_paths = find_all_slp_paths(rice_10do_pipeline_output_folder)
    assert len(slp_paths) == 2
    # Load series from slps
    rice_series_all = load_series_from_slps(
        slp_paths=slp_paths, h5s=False, csv_path=None
    )
    assert len(rice_series_all) == 2
    # Get first series
    rice_series = rice_series_all[0]

    pipeline = OlderMonocotPipeline()
    all_traits = pipeline.compute_batch_traits(rice_series_all)
    rice_traits = pipeline.compute_plant_traits(rice_series)

    # Dataframe shape assertions
    assert rice_traits.shape == (72, 102)
    assert all_traits.shape == (2, 901)

    # Dataframe dtype assertions
    expected_rice_traits_dtypes = {
        "frame_idx": "int64",
        "crown_count": "int64",
    }

    expected_all_traits_dtypes = {
        "crown_count_min": "int64",
        "crown_count_max": "int64",
    }

    for col, expected_dtype in expected_rice_traits_dtypes.items():
        assert (
            rice_traits[col].dtype == expected_dtype
        ), f"Unexpected dtype for column {col} in rice_traits"

    for col, expected_dtype in expected_all_traits_dtypes.items():
        assert (
            all_traits[col].dtype == expected_dtype
        ), f"Unexpected dtype for column {col} in all_traits"

    # Value range assertions for traits
    assert (
        all_traits["crown_curve_indices_mean_median"].dropna() >= 0
    ).all(), "crown_curve_indices_mean_median in all_traits contains negative values"
    assert (
        all_traits["crown_curve_indices_mean_median"].dropna() <= 1
    ).all(), (
        "crown_curve_indices_mean_median in all_traits contains values greater than 1"
    )
    assert (
        (0 <= rice_traits["crown_angles_proximal_p95"].dropna())
        & (rice_traits["crown_angles_proximal_p95"].dropna() <= 180)
    ).all(), "angle_column in rice_traits contains values out of range [0, 180]"
    assert (
        (0 <= all_traits["crown_angles_proximal_median_p95"].dropna())
        & (all_traits["crown_angles_proximal_median_p95"].dropna() <= 180)
    ).all(), "angle_column in all_traits contains values out of range [0, 180]"
    assert (
        (0 <= rice_traits["angle_chull_r1_left_intersection_vector"].dropna())
        & (rice_traits["angle_chull_r1_left_intersection_vector"].dropna() <= 180)
    ).all(), "angle column in rice_traits contains values out of range [0, 180]"
    assert (
        (0 <= all_traits["angle_chull_r1_left_intersection_vector_p95"].dropna())
        & (all_traits["angle_chull_r1_left_intersection_vector_p95"].dropna() <= 180)
    ).all(), "angle column in all_traits contains values out of range [0, 180]"
    assert (
        (0 <= rice_traits["angle_chull_r1_right_intersection_vector"].dropna())
        & (rice_traits["angle_chull_r1_right_intersection_vector"].dropna() <= 180)
    ).all(), "angle column in rice_traits contains values out of range [0, 180]"
    assert (
        (0 <= all_traits["angle_chull_r1_right_intersection_vector_p95"].dropna())
        & (all_traits["angle_chull_r1_right_intersection_vector_p95"].dropna() <= 180)
    ).all(), "angle column in all_traits contains values out of range [0, 180]"


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
