from sleap_roots.trait_pipelines import DicotPipeline, YoungerMonocotPipeline
from sleap_roots.series import Series, find_all_series


def test_dicot_pipeline(canola_h5, soy_h5):
    canola = Series.load(
        canola_h5, primary_name="primary_multi_day", lateral_name="lateral_3_nodes"
    )
    soy = Series.load(
        soy_h5, primary_name="primary_multi_day", lateral_name="lateral__nodes"
    )

    pipeline = DicotPipeline()
    canola_traits = pipeline.compute_plant_traits(canola)
    soy_traits = pipeline.compute_plant_traits(soy)
    all_traits = pipeline.compute_batch_traits([canola, soy])

    assert canola_traits.shape == (72, 117)
    assert soy_traits.shape == (72, 117)
    assert all_traits.shape == (2, 1036)


def test_younger_monocot_pipeline(rice_h5, rice_folder):
    rice = Series.load(
        rice_h5, primary_name="longest_3do_6nodes", lateral_name="main_3do_6nodes"
    )
    rice_series_all = find_all_series(rice_folder)
    series_all = [
        Series.load(
            series, primary_name="longest_3do_6nodes", lateral_name="main_3do_6nodes"
        )
        for series in rice_series_all
    ]

    pipeline = YoungerMonocotPipeline()
    rice_traits = pipeline.compute_plant_traits(rice)
    all_traits = pipeline.compute_batch_traits(series_all)

    # Dataframe shape assertions
    assert rice_traits.shape == (72, 104)
    assert all_traits.shape == (2, 919)

    # Dataframe dtype assertions
    expected_rice_traits_dtypes = {
        "frame_idx": "int64",
        "main_count": "int64",
    }

    expected_all_traits_dtypes = {
        "main_count_min": "int64",
        "main_count_max": "int64",
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
        all_traits["main_curve_indices_mean_median"] >= 0
    ).all(), "main_curve_indices_mean_median in all_traits contains negative values"
    assert (
        (0 <= rice_traits["main_angles_proximal_p95"])
        & (rice_traits["main_angles_proximal_p95"] <= 180)
    ).all(), "angle_column in rice_traits contains values out of range [0, 180]"
    assert (
        (0 <= all_traits["main_angles_proximal_median_p95"])
        & (all_traits["main_angles_proximal_median_p95"] <= 180)
    ).all(), "angle_column in all_traits contains values out of range [0, 180]"
