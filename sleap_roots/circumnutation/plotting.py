"""Diagnostic plots for circumnutation traits (PR #16 will implement).

Provides scaleograms (Tier 1 + Tier 3), tip-trail overlays with
κ-color-coding and an ``L_gz`` arc-length marker, and 6-up plate
panels. All plots saved as PNGs in a ``plots/`` subdirectory of the
pipeline output directory; suppressible via ``--no-plots``.
"""

import logging


logger = logging.getLogger(__name__)


def scaleogram(scaleogram_result=None, out_path=None):
    """Render a CWT scaleogram diagnostic plot (PR #16 will implement).

    Args (when implemented):
        scaleogram_result: ``ScaleogramResult`` from
            :func:`~sleap_roots.circumnutation.temporal_cwt.compute_scaleogram`
            or :func:`~sleap_roots.circumnutation.spatial_cwt.compute_scaleogram`.
        out_path: PNG output path; parent directory must exist.

    Returns (when implemented):
        ``None``.

    Raises:
        NotImplementedError: Always; this is a stub for PR #16.
    """
    raise NotImplementedError("PR #16 — see docs/circumnutation/roadmap.md")
