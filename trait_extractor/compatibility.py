"""Pipeline compatibility + scan-grain support guards.

Before computing traits the orchestrator checks, in order:

1. **Scan-grain support** — multi-plant / plate pipelines route through the base
   ``compute_batch_traits`` single-plant path and yield *empty or count-only*
   scan-grain output (their per-plant traits are ``include_in_csv=False``; real
   per-plant expansion lives in ``compute_multiple_dicots_traits`` /
   ``compute_multiple_primary_roots_traits``, which the scan-grain path never calls),
   so they are rejected here. This guard runs FIRST so a multi-plant pipeline yields a
   clean error regardless of loaded roots.
2. **Root compatibility** — the pipeline's required root types must be a subset of the
   loaded root types (``required ⊆ loaded``), so a crown-only pipeline accepts a
   primary+crown scan. Required roots come from ``PIPELINE_REQUIRED_ROOTS`` (a class is
   in exactly one of the map or the reject-list — see the guard test).

``PIPELINE_REQUIRED_ROOTS`` is a hardcoded workaround for a missing public pipeline API
(talmolab/sleap-roots#251); a guard test pins each entry to the pipeline's real
``get_initial_frame_traits`` via ``ast``.
"""

from typing import Dict, FrozenSet, Type

from sleap_roots import Series
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

# Required root types per scan-grain-emittable pipeline class. Pinned to each pipeline's
# get_initial_frame_traits by the guard test in test_compatibility.py.
PIPELINE_REQUIRED_ROOTS: Dict[Type[Pipeline], FrozenSet[str]] = {
    DicotPipeline: frozenset({"primary", "lateral"}),
    YoungerMonocotPipeline: frozenset({"primary", "crown"}),
    OlderMonocotPipeline: frozenset({"crown"}),
    PrimaryRootPipeline: frozenset({"primary"}),
    LateralRootPipeline: frozenset({"lateral"}),
}

# Multi-plant / plate pipelines: selectable, but not valid for one-row scan-grain
# emission this slice (talmolab/sleap-roots#252).
MULTI_PLANT_PIPELINES: FrozenSet[Type[Pipeline]] = frozenset(
    {MultipleDicotPipeline, MultipleDicotPlatePipeline, MultiplePrimaryRootPipeline}
)


def loaded_root_types(series: Series) -> FrozenSet[str]:
    """Return the root types whose labels are loaded on a ``Series``."""
    loaded = set()
    if series.primary_labels is not None:
        loaded.add("primary")
    if series.lateral_labels is not None:
        loaded.add("lateral")
    if series.crown_labels is not None:
        loaded.add("crown")
    return frozenset(loaded)


def check_pipeline_compatible(series: Series, pipeline_cls: Type[Pipeline]) -> None:
    """Assert the selected pipeline can emit scan-grain traits for this scan.

    Runs the scan-grain-support guard first, then the root-subset check.

    Args:
        series: The loaded scan Series.
        pipeline_cls: The selected pipeline class.

    Raises:
        ValueError: If the pipeline is multi-plant (unsupported grain), is not
            registered for scan-grain emission, or requires a root type the scan did
            not load.
    """
    if pipeline_cls in MULTI_PLANT_PIPELINES:
        raise ValueError(
            f"{pipeline_cls.__name__} is not supported for scan-grain emission "
            "(multi-plant pipelines yield empty/count-only scan-grain output; see "
            "talmolab/sleap-roots#252)"
        )
    required = PIPELINE_REQUIRED_ROOTS.get(pipeline_cls)
    if required is None:
        raise ValueError(
            f"{pipeline_cls.__name__} is not registered for scan-grain emission"
        )
    loaded = loaded_root_types(series)
    missing = required - loaded
    if missing:
        raise ValueError(
            f"{pipeline_cls.__name__} requires root types {sorted(required)} but the "
            f"scan is missing {sorted(missing)} (loaded: {sorted(loaded)})"
        )
