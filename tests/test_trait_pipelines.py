from sleap_roots.trait_pipelines import DicotPipeline
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

    assert canola_traits.shape == (72, 117)
    assert soy_traits.shape == (72, 117)
    assert all_traits.shape == (2, 1036)
