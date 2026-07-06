"""Tests for mapping scan-grain traits to TraitValues."""

import math
from pathlib import Path

from sleap_roots.trait_pipelines import YoungerMonocotPipeline

from trait_extractor.loading import load_series
from trait_extractor.manifest import load_manifest
from trait_extractor.traits import compute_scan_traits, scan_trait_values

_RICE_DIR = Path("tests/data/rice_3do_pipeline_output/scan0K9E8BI")


def _rice_series():
    manifest = load_manifest(_RICE_DIR / "scan0K9E8BI.predictions.json")
    return load_series(manifest, _RICE_DIR)


def test_batch_summary_becomes_scan_trait_values():
    """A one-row batch summary maps to grain='scan' TraitValues, plant_name dropped."""
    series = _rice_series()
    traits = compute_scan_traits(series, YoungerMonocotPipeline)
    values = scan_trait_values(traits, scan_key="scan0K9E8BI")

    names = {tv.name for tv in values}
    assert "plant_name" not in names
    assert "crown_count_mean" in names
    assert values, "expected at least one trait value"
    for tv in values:
        assert tv.grain == "scan"
        assert tv.scan_key == "scan0K9E8BI"
        assert tv.value is None or math.isfinite(tv.value)


def test_non_finite_traits_coerce_to_none():
    """NaN/inf trait values become None."""
    traits = {
        "plant_name": "scan0K9E8BI",
        "trait_nan": float("nan"),
        "trait_inf": float("inf"),
        "trait_ok": 1.5,
    }
    values = {tv.name: tv.value for tv in scan_trait_values(traits, "scan0K9E8BI")}
    assert values["trait_nan"] is None
    assert values["trait_inf"] is None
    assert values["trait_ok"] == 1.5
