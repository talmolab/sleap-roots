"""Compute scan-grain traits and map them to contract ``TraitValue``s."""

from typing import Any, Dict, List, Type

from sleap_roots import Series
from sleap_roots.trait_pipelines import Pipeline
from sleap_roots_contracts import TraitValue


def compute_scan_traits(series: Series, pipeline_cls: Type[Pipeline]) -> Dict[str, Any]:
    """Compute the one-row scan-grain trait summary for a Series.

    Args:
        series: The loaded scan Series.
        pipeline_cls: The selected (scan-grain-supported) pipeline class.

    Returns:
        A flat ``{trait_name: value}`` mapping (the single ``compute_batch_traits`` row,
        including the ``plant_name`` identity column).
    """
    df = pipeline_cls().compute_batch_traits([series])
    return df.iloc[0].to_dict()


def scan_trait_values(traits: Dict[str, Any], scan_key: str) -> List[TraitValue]:
    """Map a flat trait mapping to ``grain="scan"`` ``TraitValue``s.

    The ``plant_name`` identity column is dropped; non-finite values (``NaN``/``inf``)
    normalize to ``None`` (the contract model enforces this).

    Args:
        traits: A flat ``{trait_name: value}`` mapping.
        scan_key: The scan identity stamped on every ``TraitValue``.

    Returns:
        One ``TraitValue`` per trait column (excluding ``plant_name``).
    """
    values: List[TraitValue] = []
    for name, raw in traits.items():
        if name == "plant_name":
            continue
        value = None if raw is None else float(raw)
        values.append(
            TraitValue(name=name, value=value, grain="scan", scan_key=scan_key)
        )
    return values
