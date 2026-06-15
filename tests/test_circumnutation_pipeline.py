"""Tests for the circumnutation pipeline composition (PR #14).

Covers the additive per-tier units maps (`_NUTATION_TRAIT_UNITS` /
`_PSIG_TRAIT_UNITS`), the Tier 0/Tier 1 dedup fast path on
``traveling_wave.compute``, and the ``CircumnutationPipeline`` merge-orchestrator
(``compute_traits`` + ``save``).
"""

import inspect

import numpy as np
import pandas as pd
import pytest

from sleap_roots.circumnutation import kinematics, nutation, psi_g, traveling_wave
from sleap_roots.circumnutation._constants import PIPELINE_UNIT_VOCABULARY

_TW_TRAIT_COLUMNS = traveling_wave._TRAVELING_WAVE_TRAIT_COLUMNS


def _track_rows(track_id, n_frames=64, plate_id="plate", amp=6.0, freq=0.08):
    """Per-frame rows for one curved track (amp>0 → a usable spatial trail)."""
    return [
        {
            "series": "test",
            "sample_uid": "test",
            "timepoint": "T0",
            "plate_id": plate_id,
            "plant_id": track_id,
            "track_id": track_id,
            "genotype": np.nan,
            "treatment": np.nan,
            "frame": frame,
            "tip_x": float(frame),
            "tip_y": amp * np.sin(freq * frame),
        }
        for frame in range(n_frames)
    ]


def _traj_df(n_tracks=2, n_frames=64, **kw):
    rows = []
    for t in range(n_tracks):
        rows.extend(_track_rows(t, n_frames=n_frames, **kw))
    return pd.DataFrame(rows)


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


# ---------------------------------------------------------------------------
# Task 3 — Tier 0/Tier 1 dedup fast path on traveling_wave.compute
# ---------------------------------------------------------------------------


def test_dedup_fast_path_matches_recompute():
    """Precomputed tier0_df/tier1_df produce identical Tier 3c columns (atol=0)."""
    df = _traj_df(n_tracks=2)
    tier0 = kinematics.compute(df)
    tier1 = nutation.compute(df, 300.0, coordinate="lateral")

    recompute = traveling_wave.compute(df, cadence_s=300.0)
    fast = traveling_wave.compute(df, cadence_s=300.0, tier0_df=tier0, tier1_df=tier1)

    for col in _TW_TRAIT_COLUMNS:
        np.testing.assert_array_equal(
            recompute[col].to_numpy(), fast[col].to_numpy(), err_msg=col
        )


def test_dedup_fast_path_requires_both_or_neither():
    """Supplying exactly one of tier0_df/tier1_df raises ValueError."""
    df = _traj_df(n_tracks=1)
    tier0 = kinematics.compute(df)
    tier1 = nutation.compute(df, 300.0, coordinate="lateral")
    with pytest.raises(ValueError, match="tier0_df|tier1_df|both"):
        traveling_wave.compute(df, cadence_s=300.0, tier0_df=tier0)
    with pytest.raises(ValueError, match="tier0_df|tier1_df|both"):
        traveling_wave.compute(df, cadence_s=300.0, tier1_df=tier1)


def test_dedup_fast_path_rejects_frame_missing_operand_column():
    """A precomputed frame missing its operand column raises ValueError naming it."""
    df = _traj_df(n_tracks=1)
    tier0 = kinematics.compute(df)
    tier1 = nutation.compute(df, 300.0, coordinate="lateral")
    bad_tier0 = tier0.drop(columns=["v_total_median_px_per_frame"])
    with pytest.raises(ValueError, match="v_total_median_px_per_frame"):
        traveling_wave.compute(df, 300.0, tier0_df=bad_tier0, tier1_df=tier1)
    bad_tier1 = tier1.drop(columns=["T_nutation_median"])
    with pytest.raises(ValueError, match="T_nutation_median"):
        traveling_wave.compute(df, 300.0, tier0_df=tier0, tier1_df=bad_tier1)


def test_dedup_fast_path_projects_to_operand_columns_only():
    """Full-column tier frames don't leak extra columns into the 14-col output."""
    df = _traj_df(n_tracks=2)
    tier0 = kinematics.compute(df)  # 8 identity + 10 traits
    tier1 = nutation.compute(df, 300.0, coordinate="lateral")  # 8 identity + 8 traits
    result = traveling_wave.compute(df, 300.0, tier0_df=tier0, tier1_df=tier1)
    # 8 row-identity + 6 Tier 3c traits = 14, nothing extra leaked in
    assert result.shape[1] == 14
    assert "growth_axis_unreliable" not in result.columns  # a Tier 0 extra
    assert "is_nutating" not in result.columns  # a Tier 1 extra


def test_nutation_compute_coordinate_default_is_lateral():
    """The dedup atol=0 equivalence depends on this default staying 'lateral'."""
    assert (
        inspect.signature(nutation.compute).parameters["coordinate"].default
        == "lateral"
    )
