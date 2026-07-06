"""Cross-check the consumer manifest model against predict's real output.

Skips cleanly when ``sleap-roots-predict`` is not installed (it is not a dependency of
this repo). When it IS importable, we assert the consumer ``PredictionManifest`` accepts
predict's own ``model_dump_json`` output — guarding against schema drift.
"""

import json

import pytest

from trait_extractor.manifest import PredictionManifest


def test_consumer_model_accepts_predict_real_output():
    """Consumer PredictionManifest validates predict's model_dump_json (if importable)."""
    oc = pytest.importorskip("sleap_roots_predict.output_contract")
    from sleap_roots_contracts import ModelRef

    artifact = oc.PredictionArtifact(
        root_type="primary",
        model_id="reg-rice-primary-v1",
        model=ModelRef(
            registry_id="reg/rice-primary",
            version="v1",
            sleap_nn_version="0.3.0",
            root_type="primary",
        ),
        slp_path="scan0731.modelreg-rice-primary-v1.rootprimary.slp",
        checksum="9f2c",
        file_size=20481,
    )
    manifest = oc.PredictionManifest(scan_key="scan0731", artifacts=[artifact])

    consumed = PredictionManifest.model_validate(json.loads(manifest.model_dump_json()))
    assert consumed.scan_key == "scan0731"
    assert consumed.artifacts[0].model.registry_id == "reg/rice-primary"
