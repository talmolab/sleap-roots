"""``CircumnutationPipeline`` — sequential merge-orchestrator (PR #14).

Composes the five trait-emitting tiers (Tier 0 ``kinematics``, QC ``qc``, Tier 1
``nutation``, Tier 2 ``psi_g``, Tier 3c ``traveling_wave``) into one per-plant
trait DataFrame, units mapping, and provenance-bearing CSV.

This is a **sequential merge-orchestrator**, NOT the per-frame networkx
``TraitDef`` DAG of :class:`sleap_roots.trait_pipelines.Pipeline`: the
circumnutation tiers are per-track ``DataFrame -> DataFrame`` functions, so the
pipeline calls each tier's ``compute()`` once and merges their per-plant outputs
on the per-plant 5-tuple. The documented tier-dependency order (Tier 0 / QC /
Tier 1 / Tier 2 independent; Tier 3c depends on Tier 0 + Tier 1, fed via the
dedup fast path) is honored as a fixed call order. Pure and picklable for future
multiprocessing parallelization.
"""

import logging

import attrs
import numpy as np
import pandas as pd

from sleap_roots.circumnutation import (
    kinematics,
    nutation,
    psi_g,
    qc,
    traveling_wave,
)
from sleap_roots.circumnutation._constants import ROW_IDENTITY_UNITS, ConstantsT
from sleap_roots.circumnutation._io import (
    _IDENTITY_5_TUPLE,
    _build_per_plant_template_from_df,
)
from sleap_roots.circumnutation._types import (
    ROW_IDENTITY_COLUMNS,
    CircumnutationInputs,
)


logger = logging.getLogger(__name__)


# Composed per-plant column order: 8 row-identity + each tier's trait block in
# fixed tier order. QC drops growth_axis_unreliable (Tier 0 owns it).
_QC_COMPOSED_COLUMNS: tuple = tuple(
    c for c in qc._QC_TRAIT_COLUMNS if c != "growth_axis_unreliable"
)
_COMPOSED_COLUMN_ORDER: tuple = (
    tuple(ROW_IDENTITY_COLUMNS)
    + tuple(kinematics._TIER0_TRAIT_COLUMNS)
    + _QC_COMPOSED_COLUMNS
    + tuple(nutation._NUTATION_TRAIT_COLUMNS)
    + tuple(psi_g._PSIG_TRAIT_COLUMNS)
    + tuple(traveling_wave._TRAVELING_WAVE_TRAIT_COLUMNS)
)


def _coerce_identity_int64(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce ``track_id`` / ``plant_id`` to int64, raising on a bad cast.

    A silent int64-vs-float64 key mismatch would resolve a merge to all-NaN
    rather than raising; coercing both sides with a raising guard prevents that.
    """
    out = df.copy()
    for col in ("track_id", "plant_id"):
        try:
            out[col] = out[col].astype(np.int64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{col!r} cannot be cast to int64 for the per-plant merge; this "
                f"would silently break the merge and produce all-NaN traits."
            ) from exc
    return out


def _assemble_units() -> dict:
    """Compose the units dict: row-identity + the five tier ``_*_TRAIT_UNITS``.

    ``growth_axis_unreliable`` appears in both the Tier 0 and QC maps with the
    same ``"bool"`` value; the dict merge dedups it to one key (mapping to the
    Tier-0-owned column).
    """
    units: dict = {col: ROW_IDENTITY_UNITS[col] for col in ROW_IDENTITY_COLUMNS}
    units.update(kinematics._TIER0_TRAIT_UNITS)
    units.update(qc._QC_TRAIT_UNITS)
    units.update(nutation._NUTATION_TRAIT_UNITS)
    units.update(psi_g._PSIG_TRAIT_UNITS)
    units.update(traveling_wave._TRAVELING_WAVE_TRAIT_UNITS)
    return units


@attrs.define
class CircumnutationPipeline:
    """Picklable merge-orchestrator composing the five circumnutation tiers.

    Attributes:
        constants: Optional :class:`ConstantsT` override-bag threaded to every
            tier. ``None`` (default) uses module-level defaults.
    """

    constants: "ConstantsT | None" = attrs.field(default=None)

    def compute_traits(self, inputs: CircumnutationInputs):
        """Compose the five tiers into one per-plant trait table (pure, no I/O).

        Args:
            inputs: Validated :class:`CircumnutationInputs`.

        Returns:
            Tuple ``(per_plant_df, trajectory_df, units_dict)``: the composed
            46-column per-plant DataFrame, the (unmodified) input
            ``trajectory_df`` echoed for provenance, and the column-to-unit
            mapping covering every emitted column.
        """
        df = inputs.trajectory_df
        cadence_s = inputs.cadence_s
        constants = self.constants

        n_tracks = df[list(_IDENTITY_5_TUPLE)].drop_duplicates().shape[0]
        logger.debug(
            "CircumnutationPipeline.compute_traits(n_tracks=%d, cadence_s=%.6f)",
            n_tracks,
            cadence_s,
        )

        # Tier 0 / QC / Tier 1 / Tier 2 are independent; Tier 3c depends on Tier 0
        # + Tier 1, fed via the dedup fast path (computed once, not recomputed).
        tier0 = kinematics.compute(df, constants=constants)
        qc_df = qc.compute(df, constants=constants)
        tier1 = nutation.compute(df, cadence_s, constants=constants)
        tier2 = psi_g.compute(df, cadence_s, constants=constants)
        tier3c = traveling_wave.compute(
            df, cadence_s, constants, tier0_df=tier0, tier1_df=tier1
        )

        # growth_axis_unreliable: Tier 0 owns it; assert QC's copy is equal (dtype
        # + value via Series.equals) before dropping it, so the coalescing cannot
        # silently mask a future divergence.
        keys = list(_IDENTITY_5_TUPLE)
        gau_t0 = (
            tier0[keys + ["growth_axis_unreliable"]]
            .sort_values(keys)
            .reset_index(drop=True)
        )
        gau_qc = (
            qc_df[keys + ["growth_axis_unreliable"]]
            .sort_values(keys)
            .reset_index(drop=True)
        )
        if not gau_t0["growth_axis_unreliable"].equals(
            gau_qc["growth_axis_unreliable"]
        ):
            raise ValueError(
                "growth_axis_unreliable diverges between Tier 0 and QC; the "
                "cross-tier equal-by-construction invariant is violated."
            )

        # Build from the shared per-plant template (8 identity cols, int64 keys),
        # left-merging each tier projected to [*5-tuple, *its trait cols] so the
        # 3 non-key identity columns (timepoint/genotype/treatment) cannot collide.
        template = _coerce_identity_int64(_build_per_plant_template_from_df(df))
        blocks = [
            (tier0, list(kinematics._TIER0_TRAIT_COLUMNS)),
            (qc_df, list(_QC_COMPOSED_COLUMNS)),
            (tier1, list(nutation._NUTATION_TRAIT_COLUMNS)),
            (tier2, list(psi_g._PSIG_TRAIT_COLUMNS)),
            (tier3c, list(traveling_wave._TRAVELING_WAVE_TRAIT_COLUMNS)),
        ]
        result = template
        for tier_df, trait_cols in blocks:
            proj = _coerce_identity_int64(tier_df[keys + trait_cols])
            result = result.merge(proj, on=keys, how="left")

        result = result[list(_COMPOSED_COLUMN_ORDER)]
        units = _assemble_units()
        return result, df, units


def compute_traits(inputs: CircumnutationInputs, constants=None):
    """Run the full circumnutation pipeline on a :class:`CircumnutationInputs`.

    Thin module-level wrapper preserving the stub's signature; equal to
    ``CircumnutationPipeline(constants=constants).compute_traits(inputs)``.

    Args:
        inputs: Validated :class:`CircumnutationInputs`.
        constants: Optional :class:`ConstantsT` override-bag.

    Returns:
        Tuple ``(per_plant_df, trajectory_df, units_dict)``.
    """
    return CircumnutationPipeline(constants=constants).compute_traits(inputs)
