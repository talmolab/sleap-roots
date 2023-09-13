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

    assert rice_traits.shape == (72, 104)
    assert all_traits.shape == (2, 919)
