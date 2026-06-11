"""Tests for Tier 3c traveling-wave trait emission (PR #10).

Mirrors tests/test_circumnutation_nutation.py / test_circumnutation_psi_g.py
structure. Spec: openspec/changes/add-circumnutation-tier3c-traits/specs/
circumnutation/spec.md (Requirement: Tier 3c traveling-wave trait emission API).
"""

import logging

import numpy as np
import pandas as pd
import pytest

from sleap_roots.circumnutation import traveling_wave
from sleap_roots.circumnutation._constants import ConstantsT
from sleap_roots.circumnutation._io import _IDENTITY_5_TUPLE
from sleap_roots.circumnutation._types import ROW_IDENTITY_COLUMNS

_TRAIT_COLUMNS = (
    "lambda_spatial_median_px",
    "lambda_spatial_variation",
    "traveling_wave_residual",
    "lambda_expected_px",
    "lambda_spatial_mad_px",
    "coi_valid_fraction",
)

_OMITTED_LGZ_COLUMNS = (
    "L_gz_estimate",
    "L_c_estimate",
    "B_balance_number",
    "L_gz_steady_state_residual",
    "L_gz_resolvable",
)


def _track_rows(track_id, n_frames=64, plate_id="plate", amp=0.0, freq=0.05):
    """Build per-frame rows for one track (a gently curved trail by default)."""
    rows = []
    for frame in range(n_frames):
        rows.append(
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
        )
    return rows


def _traj_df(n_tracks=1, n_frames=64, **kw):
    rows = []
    for t in range(n_tracks):
        rows.extend(_track_rows(t, n_frames=n_frames, **kw))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Task 2 — schema + validation + units + logging
# ---------------------------------------------------------------------------


def test_compute_returns_documented_columns_dtypes_and_uniqueness():
    """8 row-identity + 6 float64 trait columns, declared order, 5-tuple unique."""
    df = _traj_df(n_tracks=2)
    result = traveling_wave.compute(df, cadence_s=300.0)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == list(ROW_IDENTITY_COLUMNS) + list(_TRAIT_COLUMNS)
    for col in _TRAIT_COLUMNS:
        assert result[col].dtype == np.float64, col
    assert result[list(_IDENTITY_5_TUPLE)].duplicated().sum() == 0
    # The 5 L_gz/L_c traits are OMITTED (blocked on #230), not reserved as NaN.
    for col in _OMITTED_LGZ_COLUMNS:
        assert col not in result.columns


def test_compute_one_row_per_track():
    df = _traj_df(n_tracks=3)
    result = traveling_wave.compute(df, cadence_s=300.0)
    assert len(result) == 3


@pytest.mark.parametrize("bad", [0, -1.0, float("nan"), float("inf"), float("-inf")])
def test_compute_rejects_invalid_cadence_value(bad):
    df = _traj_df()
    with pytest.raises(ValueError, match="cadence_s"):
        traveling_wave.compute(df, cadence_s=bad)


@pytest.mark.parametrize("bad", [True, np.bool_(True), "300", [300.0]])
def test_compute_rejects_invalid_cadence_type(bad):
    df = _traj_df()
    with pytest.raises(TypeError, match="cadence_s"):
        traveling_wave.compute(df, cadence_s=bad)


def test_compute_rejects_invalid_constants_type():
    df = _traj_df()
    with pytest.raises(TypeError, match="constants"):
        traveling_wave.compute(df, cadence_s=300.0, constants=object())


@pytest.mark.parametrize("bad", [0.0, 1.5, -0.1])
def test_compute_rejects_coi_fraction_max_out_of_range(bad):
    df = _traj_df()
    with pytest.raises(ValueError, match="COI_FRACTION_MAX"):
        traveling_wave.compute(
            df, cadence_s=300.0, constants=ConstantsT(COI_FRACTION_MAX=bad)
        )


def test_compute_rejects_empty_trajectory_df():
    with pytest.raises(ValueError):
        traveling_wave.compute(pd.DataFrame(), cadence_s=300.0)


def test_trait_units_all_in_pipeline_vocabulary():
    """_TRAVELING_WAVE_TRAIT_UNITS maps all 6 columns; units in vocabulary."""
    from sleap_roots.circumnutation._constants import PIPELINE_UNIT_VOCABULARY

    units = traveling_wave._TRAVELING_WAVE_TRAIT_UNITS
    assert set(units) == set(_TRAIT_COLUMNS)
    for col in (
        "lambda_spatial_median_px",
        "lambda_expected_px",
        "lambda_spatial_mad_px",
    ):
        assert units[col] == "px"
    for col in (
        "lambda_spatial_variation",
        "traveling_wave_residual",
        "coi_valid_fraction",
    ):
        assert units[col] == "—"
    for value in units.values():
        assert value in PIPELINE_UNIT_VOCABULARY


def test_compute_emits_exactly_one_debug_record(caplog):
    df = _traj_df()
    with caplog.at_level(
        logging.DEBUG, logger="sleap_roots.circumnutation.traveling_wave"
    ):
        traveling_wave.compute(df, cadence_s=300.0)
    records = [
        r
        for r in caplog.records
        if r.name == "sleap_roots.circumnutation.traveling_wave"
    ]
    debug = [r for r in records if r.levelno == logging.DEBUG]
    assert len(debug) == 1
    assert debug[0].getMessage().startswith("traveling_wave.compute(")
    assert "n_tracks=" in debug[0].getMessage()
    assert "cadence_s=" in debug[0].getMessage()
    assert [r for r in records if r.levelno >= logging.INFO] == []
