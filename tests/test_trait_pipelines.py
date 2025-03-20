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

from sleap_roots.points import (
    get_count,
    join_pts,
    get_all_pts_array,
    get_nodes,
    get_root_vectors,
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
)

from sleap_roots.ellipse import (
    fit_ellipse,
    get_ellipse_a,
    get_ellipse_b,
    get_ellipse_ratio,
)


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
            trait_dict["primary_proximal_node_pt"] = get_nodes(
                trait_dict["primary_max_length_pts"],
                trait_dict["primary_proximal_node_ind"],
            )
            trait_dict["primary_distal_node_pt"] = get_nodes(
                trait_dict["primary_max_length_pts"],
                trait_dict["primary_distal_node_ind"],
            )
            trait_dict["lateral_proximal_node_pts"] = get_nodes(
                trait_dict["lateral_pts"], trait_dict["lateral_proximal_node_inds"]
            )
            trait_dict["lateral_distal_node_pts"] = get_nodes(
                trait_dict["lateral_pts"], trait_dict["lateral_distal_node_inds"]
            )
            trait_dict["primary_proximal_root_vector"] = get_root_vectors(
                trait_dict["primary_proximal_node_pt"], trait_dict["primary_base_pt"]
            )
            trait_dict["primary_distal_root_vector"] = get_root_vectors(
                trait_dict["primary_distal_node_pt"], trait_dict["primary_base_pt"]
            )
            trait_dict["lateral_proximal_root_vectors"] = get_root_vectors(
                trait_dict["lateral_proximal_node_pts"], trait_dict["lateral_base_pts"]
            )
            trait_dict["lateral_distal_root_vectors"] = get_root_vectors(
                trait_dict["lateral_distal_node_pts"], trait_dict["lateral_base_pts"]
            )
            trait_dict["primary_angle_proximal"] = get_vector_angles_from_gravity(
                trait_dict["primary_proximal_root_vector"]
            )
            trait_dict["primary_angle_distal"] = get_vector_angles_from_gravity(
                trait_dict["primary_distal_root_vector"]
            )
            trait_dict["lateral_angles_proximal"] = get_vector_angles_from_gravity(
                trait_dict["lateral_proximal_root_vectors"]
            )
            trait_dict["lateral_angles_distal"] = get_vector_angles_from_gravity(
                trait_dict["lateral_distal_root_vectors"]
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
