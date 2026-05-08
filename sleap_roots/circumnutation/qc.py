"""QC tier — track-quality flags and noise estimators (PR #3 will implement).

Defined by ``docs/circumnutation/theory.md`` §7.6: emits
``sg_residual_xy``, ``d2_noise_xy``, ``msd_noise_xy``,
``sg_d2_agreement``, ``sg_msd_agreement``, ``d2_msd_agreement``,
``frac_outlier_steps``, ``worst_step_ratio``, ``track_is_clean``,
``coi_fraction_t1``, ``coi_fraction_t3``, ``is_nutating``,
``cadence_nyquist_ratio``, ``growth_axis_unreliable``,
``qc_failure_reason``. Three independent SLEAP-localization-noise
estimators (sg / d2 / msd) provide pairwise agreement traits — see
roadmap.md cross-cutting concern CC-10.
"""

import logging


logger = logging.getLogger(__name__)


def compute(*args, **kwargs):
    """Compute the QC tier (PR #3 will implement).

    Args (when implemented):
        trajectory_df: Per-track tip-trajectory DataFrame from
            :class:`~sleap_roots.circumnutation.CircumnutationInputs`.
        constants: Optional :class:`~sleap_roots.circumnutation._constants.ConstantsT`
            override-bag.

    Returns (when implemented):
        Per-plant DataFrame with QC trait columns plus the eight
        row-identity columns.

    Raises:
        NotImplementedError: Always; this is a stub for PR #3.
    """
    raise NotImplementedError("PR #3 — see docs/circumnutation/roadmap.md")
