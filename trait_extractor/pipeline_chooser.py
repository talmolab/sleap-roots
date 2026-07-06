"""Select a ``sleap_roots`` Pipeline by species/mode/age (ported chooser).

Modeled on ``sleap-roots-predict``'s ``choose_models`` / ``ModelCard`` (which are NOT
importable here — the matcher is authored in-tree): a ``PipelineCard`` carries a
contiguous inclusive ``[age_min, age_max]`` window, and ``choose_pipeline`` returns the
single matching Pipeline subclass. Whether the selected pipeline can be emitted at
scan grain is a separate concern (see ``compatibility.py``).
"""

from pathlib import Path
from typing import List, Optional, Type, Union

import yaml
from pydantic import BaseModel, model_validator
from sleap_roots.trait_pipelines import (
    DicotPipeline,
    LateralRootPipeline,
    MultipleDicotPipeline,
    MultipleDicotPlatePipeline,
    MultiplePrimaryRootPipeline,
    OlderMonocotPipeline,
    Pipeline,
    PrimaryRootPipeline,
    YoungerMonocotPipeline,
)
from sleap_roots_contracts import ResolvedParams

# Name -> Pipeline subclass. The single registry of pipeline-class names the chooser and
# the compatibility guard both resolve against.
PIPELINE_CLASSES: dict = {
    "DicotPipeline": DicotPipeline,
    "YoungerMonocotPipeline": YoungerMonocotPipeline,
    "OlderMonocotPipeline": OlderMonocotPipeline,
    "MultipleDicotPipeline": MultipleDicotPipeline,
    "MultipleDicotPlatePipeline": MultipleDicotPlatePipeline,
    "MultiplePrimaryRootPipeline": MultiplePrimaryRootPipeline,
    "PrimaryRootPipeline": PrimaryRootPipeline,
    "LateralRootPipeline": LateralRootPipeline,
}

_YAML_PATH = Path(__file__).parent / "pipeline_selection.yaml"


class PipelineCard(BaseModel):
    """A species/mode/age-window -> pipeline-class selection entry."""

    species: str
    mode: str
    age_min: int
    age_max: int
    pipeline_class: str

    @model_validator(mode="after")
    def _check_age_range(self) -> "PipelineCard":
        """Reject an inverted age window."""
        if self.age_min > self.age_max:
            raise ValueError(
                f"age_min ({self.age_min}) must be <= age_max ({self.age_max})"
            )
        return self


def load_pipeline_cards() -> List[PipelineCard]:
    """Load the packaged ``pipeline_selection.yaml`` selection cards."""
    raw = yaml.safe_load(_YAML_PATH.read_text(encoding="utf-8"))
    return [PipelineCard.model_validate(entry) for entry in raw]


def _resolve_class(name: str) -> Type[Pipeline]:
    """Resolve a pipeline-class name to its class, raising on an unknown name."""
    cls = PIPELINE_CLASSES.get(name)
    if cls is None:
        raise ValueError(f"Unknown pipeline class: {name!r}")
    return cls


def choose_pipeline(
    params: ResolvedParams,
    cards: List[PipelineCard],
    override: Optional[str] = None,
) -> Type[Pipeline]:
    """Select exactly one Pipeline subclass for a scan.

    Reads ``species``/``mode``/``age`` from the already-canonical ``params.values``
    (``age`` is an ``int``); does not mutate ``params``. An explicit ``override``
    class-name wins and bypasses matching.

    Args:
        params: Canonical resolved params (``values`` has ``species``, ``mode``,
            ``age: int``).
        cards: Candidate selection cards.
        override: Optional pipeline-class name that wins outright.

    Returns:
        The selected ``Pipeline`` subclass.

    Raises:
        ValueError: On zero matches, more than one match, or an unknown class name.
    """
    if override is not None:
        return _resolve_class(override)

    values = params.values
    species = values["species"]
    mode = values["mode"]
    age = values["age"]
    matches = [
        card
        for card in cards
        if card.species == species
        and card.mode == mode
        and card.age_min <= age <= card.age_max
    ]
    if not matches:
        raise ValueError(
            f"No pipeline matches species={species!r} mode={mode!r} age={age}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous pipeline selection ({len(matches)} cards match) for "
            f"species={species!r} mode={mode!r} age={age}"
        )
    return _resolve_class(matches[0].pipeline_class)
