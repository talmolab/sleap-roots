"""Tests for building a Series from manifest artifacts."""

from pathlib import Path

import pytest

from trait_extractor.loading import load_series
from trait_extractor.manifest import PredictionManifest, load_manifest

_FIXTURE_DIR = Path("tests/data/rice_3do_pipeline_output/scan0K9E8BI")
_MANIFEST = _FIXTURE_DIR / "scan0K9E8BI.predictions.json"


def test_load_series_loads_primary_and_crown():
    """A rice manifest loads a Series with primary + crown labels (real .slp)."""
    manifest = load_manifest(_MANIFEST)
    series = load_series(manifest, _FIXTURE_DIR)
    assert series.series_name == "scan0K9E8BI"
    assert series.primary_labels is not None
    assert series.crown_labels is not None


def test_empty_artifacts_rejected_before_load():
    """An empty artifacts list raises before Series.load, naming the scan."""
    manifest = PredictionManifest(
        schema_version="1", scan_key="scanEMPTY", artifacts=[]
    )
    with pytest.raises(ValueError, match="scanEMPTY"):
        load_series(manifest, _FIXTURE_DIR)
