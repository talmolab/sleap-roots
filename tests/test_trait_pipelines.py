import numpy as np
import pandas as pd
from sleap_roots.trait_pipelines import (
    DicotPipeline,
    YoungerMonocotPipeline,
    OlderMonocotPipeline,
    MultipleDicotPipeline,
)
from sleap_roots.series import (
    Series,
    find_all_h5_paths,
    find_all_slp_paths,
    load_series_from_h5s,
    load_series_from_slps,
)


def test_dicot_pipeline(
    canola_h5,
    soy_h5,
    canola_primary_slp,
    canola_lateral_slp,
    soy_primary_slp,
    soy_lateral_slp,
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

    assert canola_traits.shape == (72, 117)
    assert soy_traits.shape == (72, 117)
    assert all_traits.shape == (2, 1036)


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
        all_traits["curve_index_median"] >= 0
    ).all(), "curve_index in all_traits contains negative values"
    assert (
        all_traits["crown_curve_indices_mean_median"] >= 0
    ).all(), "crown_curve_indices_mean_median in all_traits contains negative values"
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
    assert len(slp_paths) == 1
    # Load series from slps
    rice_series_all = load_series_from_slps(
        slp_paths=slp_paths, h5s=False, csv_path=None
    )
    assert len(rice_series_all) == 1
    # Get first series
    rice_series = rice_series_all[0]
    assert rice_series.series_name == "scan0K9E8BI"

    pipeline = OlderMonocotPipeline()
    all_traits = pipeline.compute_batch_traits(rice_series_all)
    rice_traits = pipeline.compute_plant_traits(rice_series)

    # Dataframe shape assertions
    assert rice_traits.shape == (72, 102)
    assert all_traits.shape == (1, 901)

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
        all_traits["crown_curve_indices_mean_median"] >= 0
    ).all(), "crown_curve_indices_mean_median in all_traits contains negative values"
    assert (
        (0 <= rice_traits["crown_angles_proximal_p95"])
        & (rice_traits["crown_angles_proximal_p95"] <= 180)
    ).all(), "angle_column in rice_traits contains values out of range [0, 180]"
    assert (
        (0 <= all_traits["crown_angles_proximal_median_p95"])
        & (all_traits["crown_angles_proximal_median_p95"] <= 180)
    ).all(), "angle_column in all_traits contains values out of range [0, 180]"


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
