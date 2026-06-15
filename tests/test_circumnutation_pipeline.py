"""Tests for the circumnutation pipeline composition (PR #14).

Covers the additive per-tier units maps (`_NUTATION_TRAIT_UNITS` /
`_PSIG_TRAIT_UNITS`), the Tier 0/Tier 1 dedup fast path on
``traveling_wave.compute``, and the ``CircumnutationPipeline`` merge-orchestrator
(``compute_traits`` + ``save``).
"""

import numpy as np
import pandas as pd

from sleap_roots.circumnutation import nutation, psi_g
from sleap_roots.circumnutation._constants import PIPELINE_UNIT_VOCABULARY


# ---------------------------------------------------------------------------
# Task 2 — additive per-tier units maps (#222 units-map portion)
# ---------------------------------------------------------------------------


def test_nutation_trait_units_pinned_values():
    """`_NUTATION_TRAIT_UNITS` covers the 8 Tier 1 columns with the pinned units."""
    expected = {
        "T_nutation_median": "s",
        "T_nutation_iqr": "s",
        "A_nutation_envelope_max_px": "px",
        "band_power_ratio": "—",
        "noise_floor_estimate": "px",
        "is_nutating": "bool",
        "period_residual_vs_derr_reference": "—",
        "cadence_nyquist_ratio": "—",
    }
    assert nutation._NUTATION_TRAIT_UNITS == expected
    # one entry per declared column, every value in vocabulary
    assert set(nutation._NUTATION_TRAIT_UNITS) == set(nutation._NUTATION_TRAIT_COLUMNS)
    for col, unit in nutation._NUTATION_TRAIT_UNITS.items():
        assert unit in PIPELINE_UNIT_VOCABULARY, f"{col} unit {unit!r} not in vocab"


def test_psig_trait_units_pinned_values():
    """`_PSIG_TRAIT_UNITS` covers the 4 Tier 2 columns with the pinned units."""
    expected = {
        "T_psig_median_s": "s",
        "delta_E_amplitude_proxy_px_per_frame": "px/frame",
        "handedness": "int",
        "helix_signed_area_px2": "px²",
    }
    assert psi_g._PSIG_TRAIT_UNITS == expected
    assert set(psi_g._PSIG_TRAIT_UNITS) == set(psi_g._PSIG_TRAIT_COLUMNS)
    for col, unit in psi_g._PSIG_TRAIT_UNITS.items():
        assert unit in PIPELINE_UNIT_VOCABULARY, f"{col} unit {unit!r} not in vocab"
