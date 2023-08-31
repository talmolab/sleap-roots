from sleap_roots.trait_pipelines import DicotPipeline, OlderMonocotPipeline
from sleap_roots.series import Series


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

    assert canola_traits.shape == (72, 115)
    assert soy_traits.shape == (72, 115)
    assert all_traits.shape == (2, 1018)


def test_OlderMonocot_pipeline(rice_main_10do_h5):
    rice = Series.load(rice_main_10do_h5, ["main_10do_6nodes"])

    pipeline = OlderMonocotPipeline()
    rice_10dag_traits = pipeline.compute_plant_traits(rice)

    assert rice_10dag_traits.shape == (72, 98)
