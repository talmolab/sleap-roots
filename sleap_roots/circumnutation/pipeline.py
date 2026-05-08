"""``CircumnutationPipeline`` DAG composition (PR #14 will implement).

Composes the per-tier compute functions (``kinematics``, ``qc``,
``temporal_cwt``, ``psi_g``, ``midline``, ``spatial_cwt``,
``parametric``) into one TraitDef DAG matching the existing
``Pipeline`` base class pattern in ``sleap_roots/trait_pipelines.py``.
Owns the integration test against plate 001 of the Sept-2025
Nipponbare 0.8PG MOCK control. Tier-level functions are pure and
picklable for future multiprocessing parallelization (per the
convention established by ``TrackedTipPipeline``).
"""

import logging


logger = logging.getLogger(__name__)


def compute_traits(*args, **kwargs):
    """Run the full circumnutation DAG on a :class:`CircumnutationInputs` (PR #14 will implement).

    Args (when implemented):
        inputs: :class:`~sleap_roots.circumnutation.CircumnutationInputs`.
        constants: Optional :class:`~sleap_roots.circumnutation._constants.ConstantsT`
            override-bag.

    Returns (when implemented):
        Tuple ``(per_plant_df, trajectory_df, units_dict)``.

    Raises:
        NotImplementedError: Always; this is a stub for PR #14.
    """
    raise NotImplementedError("PR #14 — see docs/circumnutation/roadmap.md")
