"""Tests for the pipeline compatibility + scan-grain support guards."""

import ast
import inspect
import textwrap
from pathlib import Path

import pytest
from sleap_roots import Series
from sleap_roots.trait_pipelines import (
    DicotPipeline,
    MultipleDicotPlatePipeline,
    OlderMonocotPipeline,
    Pipeline,
    YoungerMonocotPipeline,
)
from sleap_roots_contracts import ResolvedParams

from trait_extractor.compatibility import (
    MULTI_PLANT_PIPELINES,
    PIPELINE_REQUIRED_ROOTS,
    check_pipeline_compatible,
    loaded_root_types,
)
from trait_extractor.loading import load_series
from trait_extractor.manifest import load_manifest
from trait_extractor.pipeline_chooser import (
    PIPELINE_CLASSES,
    choose_pipeline,
    load_pipeline_cards,
)

_RICE_DIR = Path("tests/data/rice_3do_pipeline_output/scan0K9E8BI")


def _rice_series():
    """Load the rice scan Series (primary + crown)."""
    manifest = load_manifest(_RICE_DIR / "scan0K9E8BI.predictions.json")
    return load_series(manifest, _RICE_DIR)


def _canola_series():
    """Load a canola Series (primary + lateral, no crown)."""
    return Series.load(
        series_name="919QDUH",
        primary_path="tests/data/canola_7do/919QDUH.primary.predictions.slp",
        lateral_path="tests/data/canola_7do/919QDUH.lateral.predictions.slp",
    )


def test_grain_guard_rejects_multiplant_first():
    """A multi-plant/plate pipeline raises the grain error, before any root check.

    `MultipleDicotPlatePipeline` is NOT in PIPELINE_REQUIRED_ROOTS, so if the grain
    guard did not short-circuit first it would hit the "not registered" branch. The
    match string ("not supported") is unique to the grain branch, so this pins ordering.
    """
    series = _rice_series()  # primary+crown; roots irrelevant to the grain guard
    with pytest.raises(ValueError, match="not supported for scan-grain emission"):
        check_pipeline_compatible(series, MultipleDicotPlatePipeline)


def test_crown_only_pipeline_accepts_superset():
    """OlderMonocotPipeline (crown-only) passes against a primary+crown scan."""
    series = _rice_series()
    check_pipeline_compatible(series, OlderMonocotPipeline)  # no raise


def _packaged_selection(species, age):
    """Select a cylinder scan's pipeline class from the packaged cards."""
    params = ResolvedParams(values={"species": species, "mode": "cylinder", "age": age})
    return choose_pipeline(params, load_pipeline_cards())


def test_wheat_selection_accepts_crown_only_series():
    """Wheat's packaged pipeline passes against a crown-only scan.

    The wheat model labels seminal roots as crown, so wheat scans load crown only. No
    wheat .slp is in tests/data; the guard reads only which root types loaded, so a
    crown-only rice series exercises the same path.
    """
    series = Series.load(
        series_name="0K9E8BI",
        crown_path="tests/data/rice_10do/0K9E8BI.crown.predictions.slp",
    )
    # Series.load leaves labels None for a missing path, so pin what actually loaded.
    assert loaded_root_types(series) == {"crown"}
    pipeline_cls = _packaged_selection("wheat", 10)
    assert pipeline_cls is OlderMonocotPipeline
    check_pipeline_compatible(series, pipeline_cls)  # no raise


def test_sorghum_selection_accepts_primary_lateral_series():
    """Sorghum's packaged pipeline passes against a primary + lateral scan.

    No sorghum .slp is in tests/data; a canola series loads the same root types.
    """
    series = _canola_series()
    assert loaded_root_types(series) == {"primary", "lateral"}
    pipeline_cls = _packaged_selection("sorghum", 8)
    assert pipeline_cls is DicotPipeline
    check_pipeline_compatible(series, pipeline_cls)  # no raise


def test_missing_required_root_raises():
    """A primary+crown pipeline vs a primary+lateral scan raises, naming the miss."""
    series = _canola_series()
    with pytest.raises(ValueError, match="YoungerMonocotPipeline.*crown"):
        check_pipeline_compatible(series, YoungerMonocotPipeline)


def test_unregistered_pipeline_raises_clear_error():
    """A class in neither the map nor the reject-list raises a clear error."""
    series = _rice_series()
    with pytest.raises(ValueError, match="not registered"):
        check_pipeline_compatible(series, Pipeline)


def _roots_called_by(cls):
    """Derive required roots from a pipeline's get_initial_frame_traits source (ast)."""
    getter_to_root = {
        "get_primary_points": "primary",
        "get_lateral_points": "lateral",
        "get_crown_points": "crown",
    }
    src = textwrap.dedent(inspect.getsource(cls.get_initial_frame_traits))
    tree = ast.parse(src)
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    return frozenset(getter_to_root[c] for c in called if c in getter_to_root)


def test_required_roots_map_matches_pipeline_source():
    """Each PIPELINE_REQUIRED_ROOTS entry matches the pipeline's actual getters."""
    for cls, declared in PIPELINE_REQUIRED_ROOTS.items():
        assert declared == _roots_called_by(cls), cls.__name__


def test_map_and_rejectlist_partition_all_selectable_classes():
    """The map and reject-list are disjoint and cover every selectable class."""
    mapped = set(PIPELINE_REQUIRED_ROOTS)
    rejected = set(MULTI_PLANT_PIPELINES)
    assert mapped.isdisjoint(rejected)
    assert mapped | rejected == set(PIPELINE_CLASSES.values())
