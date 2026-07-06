"""Tests for species/mode/age -> Pipeline selection."""

import pytest
from sleap_roots.trait_pipelines import (
    DicotPipeline,
    MultipleDicotPlatePipeline,
    OlderMonocotPipeline,
    YoungerMonocotPipeline,
)
from sleap_roots_contracts import ResolvedParams

from trait_extractor.pipeline_chooser import (
    PipelineCard,
    choose_pipeline,
    load_pipeline_cards,
)


def _params(species, mode, age):
    """Build a canonical ResolvedParams (age already an int)."""
    return ResolvedParams(values={"species": species, "mode": mode, "age": age})


def test_yaml_cards_select_expected():
    """The packaged cards resolve the expected pipeline classes."""
    cards = load_pipeline_cards()
    assert choose_pipeline(_params("rice", "cylinder", 3), cards) is YoungerMonocotPipeline
    assert choose_pipeline(_params("rice", "cylinder", 8), cards) is OlderMonocotPipeline
    assert choose_pipeline(_params("canola", "cylinder", 7), cards) is DicotPipeline
    assert (
        choose_pipeline(_params("arabidopsis", "plate", 10), cards)
        is MultipleDicotPlatePipeline
    )


def test_override_wins():
    """An explicit override bypasses species/mode/age matching."""
    assert (
        choose_pipeline(_params("rice", "cylinder", 3), [], override="DicotPipeline")
        is DicotPipeline
    )


def test_no_match_raises():
    """An age outside every window raises."""
    cards = load_pipeline_cards()
    with pytest.raises(ValueError):
        choose_pipeline(_params("rice", "cylinder", 99), cards)


def test_ambiguous_match_raises():
    """Two overlapping windows for the same species/mode raise."""
    cards = [
        PipelineCard(
            species="rice",
            mode="cylinder",
            age_min=2,
            age_max=5,
            pipeline_class="YoungerMonocotPipeline",
        ),
        PipelineCard(
            species="rice",
            mode="cylinder",
            age_min=3,
            age_max=6,
            pipeline_class="OlderMonocotPipeline",
        ),
    ]
    with pytest.raises(ValueError):
        choose_pipeline(_params("rice", "cylinder", 4), cards)


def test_unknown_class_raises():
    """An unknown pipeline_class name (matched or overridden) raises."""
    cards = [
        PipelineCard(
            species="x", mode="y", age_min=1, age_max=9, pipeline_class="NopePipeline"
        )
    ]
    with pytest.raises(ValueError):
        choose_pipeline(_params("x", "y", 3), cards)
    with pytest.raises(ValueError):
        choose_pipeline(_params("x", "y", 3), [], override="NopePipeline")


def test_choose_pipeline_does_not_mutate_params():
    """choose_pipeline must not mutate params.values."""
    cards = load_pipeline_cards()
    params = _params("rice", "cylinder", 3)
    before = dict(params.values)
    choose_pipeline(params, cards)
    assert params.values == before
