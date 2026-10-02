"""Tests for species/mode/age -> Pipeline selection."""

import logging
import re

import pytest
from sleap_roots.trait_pipelines import (
    DicotPipeline,
    MultipleDicotPipeline,
    MultipleDicotPlatePipeline,
    OlderMonocotPipeline,
    YoungerMonocotPipeline,
)
from sleap_roots_contracts import ResolvedParams, compute_param_hash

from trait_extractor import pipeline_chooser
from trait_extractor.compatibility import MULTI_PLANT_PIPELINES
from trait_extractor.pipeline_chooser import (
    PipelineCard,
    choose_pipeline,
    load_pipeline_cards,
)


def _params(species, mode, age):
    """Build a canonical ResolvedParams (age already an int)."""
    return ResolvedParams(values={"species": species, "mode": mode, "age": age})


def _card(species, mode, age_min, age_max, pipeline_class):
    """Build an injected PipelineCard."""
    return PipelineCard(
        species=species,
        mode=mode,
        age_min=age_min,
        age_max=age_max,
        pipeline_class=pipeline_class,
    )


def _case_id(species, mode, age):
    """Build a ``species-mode-age`` test id (spaces in the mode become ``_``)."""
    return f"{species}-{mode.replace(' ', '_')}-{age}"


def _shared(species, mode, age, expected, past_window_age):
    """One packaged-card row of the shared case list (tasks.md)."""
    return pytest.param(
        species,
        mode,
        age,
        expected,
        past_window_age,
        id=_case_id(species, mode, age),
    )


# The shared case list against the packaged cards, in tasks.md's order, so it can be
# diffed row for row with sleap-roots-predict's choose_models tests. Columns: species,
# mode, age, expected class (None: raises "No pipeline matches"), and past_window_age's
# return (None: no clamp).
SHARED_CASES = [
    _shared("soybean", "cylinder", 10, DicotPipeline, 8),
    _shared("canola", "cylinder", 14, DicotPipeline, 13),
    _shared("pennycress", "cylinder", 15, DicotPipeline, 14),
    _shared("pennycress", "cylinder", 20, DicotPipeline, 14),
    _shared("arabidopsis", "cylinder", 28, DicotPipeline, 14),
    _shared("rice", "cylinder", 18, OlderMonocotPipeline, 10),
    _shared("arabidopsis", "cylinder", 365, DicotPipeline, 14),
    _shared("rice", "cylinder", 99, OlderMonocotPipeline, 10),
    _shared("rice", "cylinder", 11, OlderMonocotPipeline, 10),
    _shared("soybean", "cylinder", 9, DicotPipeline, 8),
    _shared("arabidopsis", "multiplant cylinder", 28, MultipleDicotPipeline, 14),
    _shared("arabidopsis", "plate", 20, MultipleDicotPlatePipeline, 14),
    _shared("arabidopsis", "cylinder", 10, DicotPipeline, None),
    _shared("arabidopsis", "cylinder", 14, DicotPipeline, None),
    _shared("rice", "cylinder", 4, YoungerMonocotPipeline, None),
    _shared("rice", "cylinder", 8, OlderMonocotPipeline, None),
    _shared("canola", "cylinder", 13, DicotPipeline, None),
    _shared("soybean", "cylinder", 8, DicotPipeline, None),
    _shared("rice", "cylinder", 10, OlderMonocotPipeline, None),
    _shared("rice", "cylinder", 3, YoungerMonocotPipeline, None),
    _shared("canola", "cylinder", 0, None, None),
    _shared("arabidopsis", "cylinder", 1, None, None),
    _shared("rice", "cylinder", 1, None, None),
    _shared("arabidopsis", "plate", 3, None, None),
    _shared("sorghum", "cylinder", 30, None, None),
    _shared("canola", "multiplant cylinder", 20, None, None),
    _shared("soybean", "plate", 10, None, None),
]


def _select(cases, predicate):
    """Filter shared-case params by ``predicate(species, mode, age, expected, pwa)``."""
    return [case for case in cases if predicate(*case.values)]


PAST_WINDOW_CASES = _select(
    SHARED_CASES,
    lambda s, m, a, cls, pwa: pwa is not None and cls not in MULTI_PLANT_PIPELINES,
)
MULTI_PLANT_CASES = _select(
    SHARED_CASES,
    lambda s, m, a, cls, pwa: pwa is not None and cls in MULTI_PLANT_PIPELINES,
)
IN_WINDOW_CASES = _select(
    SHARED_CASES, lambda s, m, a, cls, pwa: pwa is None and cls is not None
)
UNMATCHED_CASES = _select(SHARED_CASES, lambda s, m, a, cls, pwa: cls is None)

_CANOLA_BY_MODE = [
    _card("canola", "cylinder", 2, 13, "DicotPipeline"),
    _card("canola", "multiplant cylinder", 2, 20, "MultipleDicotPipeline"),
]
_LOWER_WINDOW = [
    _card("x", "cylinder", 2, 10, "DicotPipeline"),
    _card("x", "cylinder", 2, 14, "OlderMonocotPipeline"),
]
_GAPPED = [
    _card("x", "cylinder", 2, 5, "YoungerMonocotPipeline"),
    _card("x", "cylinder", 8, 10, "OlderMonocotPipeline"),
]
_TIED = [
    _card("arabidopsis", "cylinder", 2, 14, "DicotPipeline"),
    _card("arabidopsis", "cylinder", 10, 14, "OlderMonocotPipeline"),
]
_RICE_YOUNGER_ONLY = [_card("rice", "cylinder", 2, 5, "YoungerMonocotPipeline")]
_PER_SPECIES_LOWEST = [
    _card("canola", "cylinder", 5, 13, "DicotPipeline"),
    _card("pennycress", "cylinder", 2, 14, "OlderMonocotPipeline"),
]
_UNKNOWN_CLASS = [_card("x", "cylinder", 2, 10, "NopePipeline")]


def _no_match_pattern(species, mode, age):
    """Anchored regex for the 'No pipeline matches' error with the real age."""
    return (
        rf"^No pipeline matches species={re.escape(repr(species))} "
        rf"mode={re.escape(repr(mode))} age={age}$"
    )


def _injected(cards, species, mode, age, expected, past_window_age, label):
    """One injected-card row of the shared case list (tasks.md)."""
    return pytest.param(cards, species, mode, age, expected, past_window_age, id=label)


# The shared case list's injected-card rows, in tasks.md's order. Columns: cards,
# species, mode, age, expected outcome (a Pipeline class, or an anchored regex the
# raised ValueError must match), and past_window_age's return.
INJECTED_CASES = [
    _injected(
        _CANOLA_BY_MODE,
        "canola",
        "cylinder",
        15,
        DicotPipeline,
        13,
        "per-mode-cylinder",
    ),
    _injected(
        _CANOLA_BY_MODE,
        "canola",
        "multiplant cylinder",
        15,
        MultipleDicotPipeline,
        None,
        "per-mode-multiplant",
    ),
    _injected(
        _LOWER_WINDOW, "x", "cylinder", 28, OlderMonocotPipeline, 14, "lower-window"
    ),
    _injected(
        _GAPPED,
        "x",
        "cylinder",
        6,
        _no_match_pattern("x", "cylinder", 6),
        None,
        "gap-6",
    ),
    _injected(_GAPPED, "x", "cylinder", 11, OlderMonocotPipeline, 10, "gap-11"),
    _injected(
        _TIED,
        "arabidopsis",
        "cylinder",
        28,
        r"^Ambiguous pipeline selection \(2 cards match\) for "
        r"species='arabidopsis' mode='cylinder' age=28 matched as age=14$",
        14,
        "tie",
    ),
    _injected(
        _RICE_YOUNGER_ONLY,
        "rice",
        "cylinder",
        9,
        YoungerMonocotPipeline,
        5,
        "rice-2-5-age-9",
    ),
    _injected(
        _PER_SPECIES_LOWEST,
        "canola",
        "cylinder",
        3,
        _no_match_pattern("canola", "cylinder", 3),
        None,
        "per-species-lowest",
    ),
    _injected(
        [],
        "rice",
        "cylinder",
        100,
        _no_match_pattern("rice", "cylinder", 100),
        None,
        "no-cards",
    ),
    _injected(
        _UNKNOWN_CLASS,
        "x",
        "cylinder",
        12,
        "^Unknown pipeline class",
        10,
        "unknown-class",
    ),
]


def test_yaml_cards_select_expected():
    """The packaged cards resolve the expected pipeline classes."""
    cards = load_pipeline_cards()
    assert (
        choose_pipeline(_params("rice", "cylinder", 3), cards) is YoungerMonocotPipeline
    )
    assert (
        choose_pipeline(_params("rice", "cylinder", 8), cards) is OlderMonocotPipeline
    )
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


def test_override_wins_for_past_window_age():
    """An override still wins when the scan's age is past every window."""
    cards = load_pipeline_cards()
    assert (
        choose_pipeline(
            _params("rice", "cylinder", 18), cards, override="DicotPipeline"
        )
        is DicotPipeline
    )


@pytest.mark.parametrize("species, mode, age, expected, past_window_age", SHARED_CASES)
def test_past_window_age_shared_cases(
    species, mode, age, expected, past_window_age, caplog
):
    """past_window_age returns the shared case list's matched-as age, or None."""
    cards = load_pipeline_cards()
    params = _params(species, mode, age)
    with caplog.at_level(logging.DEBUG):
        assert pipeline_chooser.past_window_age(params, cards) == past_window_age
    assert caplog.records == []


@pytest.mark.parametrize(
    "cards, species, mode, age, expected, past_window_age", INJECTED_CASES
)
def test_past_window_age_injected_cases(
    cards, species, mode, age, expected, past_window_age, caplog
):
    """past_window_age returns the injected rows' matched-as age, or None."""
    params = _params(species, mode, age)
    with caplog.at_level(logging.DEBUG):
        assert pipeline_chooser.past_window_age(params, cards) == past_window_age
    assert caplog.records == []


@pytest.mark.parametrize(
    "species, mode, age, expected, past_window_age", PAST_WINDOW_CASES
)
def test_past_window_age_selects_highest_window(
    species, mode, age, expected, past_window_age, caplog
):
    """A past-window age selects the highest window's class, keeping params intact."""
    cards = load_pipeline_cards()
    params = _params(species, mode, age)
    before = dict(params.values)
    with caplog.at_level(logging.DEBUG):
        selected = choose_pipeline(params, cards)
    assert selected is expected
    assert params.values == before
    assert params.param_hash == compute_param_hash(params.values)
    assert caplog.records == []


@pytest.mark.parametrize(
    "species, mode, age, expected, past_window_age", IN_WINDOW_CASES
)
def test_in_window_selection_unchanged(
    species, mode, age, expected, past_window_age, caplog
):
    """In-window ages, including each highest age_max, select as before."""
    cards = load_pipeline_cards()
    with caplog.at_level(logging.DEBUG):
        assert choose_pipeline(_params(species, mode, age), cards) is expected
    assert caplog.records == []


@pytest.mark.parametrize(
    "species, mode, age, expected, past_window_age", UNMATCHED_CASES
)
def test_unmatched_scans_still_raise(
    species, mode, age, expected, past_window_age, caplog
):
    """Younger-than-window and no-card scans raise with the real age."""
    cards = load_pipeline_cards()
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(ValueError, match=_no_match_pattern(species, mode, age)):
            choose_pipeline(_params(species, mode, age), cards)
    assert caplog.records == []


@pytest.mark.parametrize(
    "cards, species, mode, age, expected, past_window_age", INJECTED_CASES
)
def test_injected_cards_select_or_raise(
    cards, species, mode, age, expected, past_window_age, caplog
):
    """Injected cards: per species + mode maximum, gaps, ties and unknown classes."""
    params = _params(species, mode, age)
    with caplog.at_level(logging.DEBUG):
        if isinstance(expected, str):
            with pytest.raises(ValueError, match=expected):
                choose_pipeline(params, cards)
        else:
            assert choose_pipeline(params, cards) is expected
    assert caplog.records == []


@pytest.mark.parametrize(
    "species, mode, age, expected, past_window_age", MULTI_PLANT_CASES
)
def test_past_window_multi_plant_scans_clamp(
    species, mode, age, expected, past_window_age
):
    """Past-window multiplant/plate scans clamp to a class the grain guard rejects."""
    cards = load_pipeline_cards()
    selected = choose_pipeline(_params(species, mode, age), cards)
    assert selected is expected
    assert selected in MULTI_PLANT_PIPELINES


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
    with pytest.raises(
        ValueError,
        match=(
            r"^Ambiguous pipeline selection \(2 cards match\) for "
            r"species='rice' mode='cylinder' age=4$"
        ),
    ):
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
