"""Circumnutation analysis pipeline for sleap-roots.

This package consumes per-track tip-trajectory DataFrames matching the
:data:`ROW_IDENTITY_COLUMNS` schema (8 columns: ``series``,
``sample_uid``, ``timepoint``, ``plate_id``, ``plant_id``, ``track_id``,
``genotype``, ``treatment``, plus per-frame ``frame``, ``tip_x``,
``tip_y``) and emits per-plant circumnutation traits — period,
amplitude, handedness, growth-zone length, balance number,
traveling-wave residual — plus per-plant time-series CSVs and
diagnostic plots.

**Note on input shape.** ``TrackedTipPipeline`` today emits only 7
columns (``series``, ``sample_uid``, ``timepoint``, ``track_id``,
``frame``, ``tip_x``, ``tip_y``); callers wanting to feed
``TrackedTipPipeline`` output directly into ``CircumnutationInputs``
must first enrich the DataFrame with ``plate_id`` (==``series`` is a
fine fallback), ``plant_id`` (==``track_id`` today), ``genotype``
(NaN if unavailable), and ``treatment`` (NaN if unavailable). A
future tier PR may provide a convenience adapter; the foundation
requires the full 8-column schema explicitly so the row-identity
contract is unambiguous.

The pipeline is organized as five computation tiers + a QC tier
(see ``docs/circumnutation/theory.md`` §6.3 for the full architecture
and ``docs/circumnutation/roadmap.md`` for the multi-PR rollout).

This foundation PR (PR #1) lands the contracts only — package layout,
:class:`CircumnutationInputs` data class, :data:`ROW_IDENTITY_COLUMNS`
schema, units sidecar, run-metadata sidecar, module-level constants,
:func:`~sleap_roots.circumnutation.units.convert_to_mm` downstream
utility, and per-module logging convention. The 10 stub modules
(``kinematics``, ``qc``, ``synthetic``, ``temporal_cwt``, ``psi_g``,
``midline``, ``spatial_cwt``, ``parametric``, ``plotting``,
``pipeline``) raise ``NotImplementedError`` until their tier PRs land.

Pure-pixel convention: the pipeline never accepts ``px_per_mm`` and
never emits ``[mm]`` columns. Calibration is always a downstream
concern handled by :func:`~sleap_roots.circumnutation.units.convert_to_mm`.
"""

import logging

from sleap_roots.circumnutation._types import (
    ROW_IDENTITY_COLUMNS,
    CircumnutationInputs,
)
from sleap_roots.circumnutation.units import convert_to_mm


logger = logging.getLogger(__name__)


__all__ = [
    "CircumnutationInputs",
    "ROW_IDENTITY_COLUMNS",
    "convert_to_mm",
]
