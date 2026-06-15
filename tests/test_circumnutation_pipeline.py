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


# ---------------------------------------------------------------------------
# Task 4 — CircumnutationPipeline.compute_traits (merge-orchestrator)
# ---------------------------------------------------------------------------

from sleap_roots.circumnutation import pipeline, qc  # noqa: E402
from sleap_roots.circumnutation._io import _IDENTITY_5_TUPLE  # noqa: E402
from sleap_roots.circumnutation._types import (  # noqa: E402
    ROW_IDENTITY_COLUMNS,
    CircumnutationInputs,
)


def _expected_46_columns():
    qc_cols = [c for c in qc._QC_TRAIT_COLUMNS if c != "growth_axis_unreliable"]
    return (
        list(ROW_IDENTITY_COLUMNS)
        + list(kinematics._TIER0_TRAIT_COLUMNS)
        + qc_cols
        + list(nutation._NUTATION_TRAIT_COLUMNS)
        + list(psi_g._PSIG_TRAIT_COLUMNS)
        + list(traveling_wave._TRAVELING_WAVE_TRAIT_COLUMNS)
    )


def _inputs(n_tracks=2, n_plates=1, n_frames=64, float_track_id=True):
    """Build CircumnutationInputs; multi-plate with overlapping track_ids."""
    rows = []
    for p in range(n_plates):
        for t in range(n_tracks):
            rows.extend(_track_rows(t, n_frames=n_frames, plate_id=f"plate{p}"))
    df = pd.DataFrame(rows)
    if float_track_id:
        df["track_id"] = df["track_id"].astype(np.float64)
        df["plant_id"] = df["plant_id"].astype(np.float64)
    return CircumnutationInputs(trajectory_df=df, cadence_s=300.0)


def test_compute_traits_returns_46_column_schema():
    """3-tuple; per_plant_df has exactly the 46 columns in declared tier order."""
    inputs = _inputs(n_tracks=2, n_plates=2)
    out = pipeline.compute_traits(inputs)

    assert isinstance(out, tuple) and len(out) == 3
    per_plant_df, trajectory_df, units = out
    assert isinstance(per_plant_df, pd.DataFrame)
    assert isinstance(trajectory_df, pd.DataFrame)
    assert isinstance(units, dict)

    assert list(per_plant_df.columns) == _expected_46_columns()
    assert len(per_plant_df.columns) == 46
    # one row per 5-tuple (2 plates x 2 tracks = 4)
    assert per_plant_df[list(_IDENTITY_5_TUPLE)].duplicated().sum() == 0
    assert len(per_plant_df) == 4
    # flag dtypes preserved through the merge
    assert per_plant_df["is_nutating"].dtype == np.bool_
    assert per_plant_df["handedness"].dtype == np.int64


def test_compute_traits_coalesces_growth_axis_unreliable():
    """Exactly one growth_axis_unreliable column, Tier-0-owned, equals QC's."""
    inputs = _inputs(n_tracks=2)
    per_plant_df, _, _ = pipeline.compute_traits(inputs)
    cols = list(per_plant_df.columns)
    assert cols.count("growth_axis_unreliable") == 1
    assert "growth_axis_unreliable_x" not in cols
    # sits in the Tier 0 block (immediately after the 8 identity + 9 other Tier0)
    tier0_block = cols[8 : 8 + len(kinematics._TIER0_TRAIT_COLUMNS)]
    assert "growth_axis_unreliable" in tier0_block
    # equals what qc.compute emits for the same tracks
    qc_df = qc.compute(inputs.trajectory_df)
    keys = list(_IDENTITY_5_TUPLE)
    merged = per_plant_df[keys + ["growth_axis_unreliable"]].merge(
        qc_df[keys + ["growth_axis_unreliable"]], on=keys, suffixes=("_p", "_q")
    )
    assert (
        merged["growth_axis_unreliable_p"] == merged["growth_axis_unreliable_q"]
    ).all()


def test_compute_traits_units_dict_covers_all_columns_in_vocab(tmp_path):
    """units_dict 1:1-covers the 46 columns, all in vocab, and the writer accepts it."""
    from sleap_roots.circumnutation._io import write_per_plant_csv

    inputs = _inputs(n_tracks=2)
    per_plant_df, _, units = pipeline.compute_traits(inputs)
    assert set(units.keys()) == set(per_plant_df.columns)
    for col, unit in units.items():
        assert unit in PIPELINE_UNIT_VOCABULARY, f"{col}={unit!r}"
    assert list(units.keys()).count("growth_axis_unreliable") == 1
    # writer does not raise the coverage/vocabulary ValueError
    write_per_plant_csv(tmp_path / "traits.csv", per_plant_df, units, {})


def test_compute_traits_dedup_equivalence_with_standalone():
    """Pipeline Tier 3c columns equal a standalone traveling_wave.compute (atol=0)."""
    inputs = _inputs(n_tracks=2)
    per_plant_df, _, _ = pipeline.compute_traits(inputs)
    standalone = traveling_wave.compute(inputs.trajectory_df, inputs.cadence_s)
    keys = list(_IDENTITY_5_TUPLE)
    merged = per_plant_df[keys + list(_TW_TRAIT_COLUMNS)].merge(
        standalone[keys + list(_TW_TRAIT_COLUMNS)], on=keys, suffixes=("_p", "_s")
    )
    for col in _TW_TRAIT_COLUMNS:
        np.testing.assert_array_equal(
            merged[f"{col}_p"].to_numpy(), merged[f"{col}_s"].to_numpy(), err_msg=col
        )


def test_compute_traits_computes_tier0_and_tier1_once(monkeypatch):
    """Tier 0 / Tier 1 are computed exactly once (dedup: not twice via Tier 3c)."""
    calls = {"kinematics": 0, "nutation": 0}
    real_k, real_n = kinematics.compute, nutation.compute

    def spy_k(*a, **k):
        calls["kinematics"] += 1
        return real_k(*a, **k)

    def spy_n(*a, **k):
        calls["nutation"] += 1
        return real_n(*a, **k)

    monkeypatch.setattr(pipeline.kinematics, "compute", spy_k)
    monkeypatch.setattr(pipeline.nutation, "compute", spy_n)
    pipeline.compute_traits(_inputs(n_tracks=2))
    assert calls == {"kinematics": 1, "nutation": 1}


def test_compute_traits_performs_no_filesystem_io(tmp_path, monkeypatch):
    """compute_traits writes zero files (writing is save's job)."""
    monkeypatch.chdir(tmp_path)
    pipeline.compute_traits(_inputs(n_tracks=2))
    assert list(tmp_path.iterdir()) == []


def test_compute_traits_negative_units_coverage_raises(tmp_path, monkeypatch):
    """A tier units map missing a key → the writer raises the coverage error."""
    from sleap_roots.circumnutation._io import write_per_plant_csv

    broken = dict(nutation._NUTATION_TRAIT_UNITS)
    broken.pop("band_power_ratio")
    monkeypatch.setattr(nutation, "_NUTATION_TRAIT_UNITS", broken)
    per_plant_df, _, units = pipeline.compute_traits(_inputs(n_tracks=1))
    with pytest.raises(ValueError, match="band_power_ratio"):
        write_per_plant_csv(tmp_path / "t.csv", per_plant_df, units, {})


def test_compute_traits_all_degenerate_input():
    """Every track degenerate → full 46-col frame, no raise, flag dtypes intact."""
    rows = []
    for t in range(3):
        for frame in range(64):
            rows.append(
                {
                    "series": "test",
                    "sample_uid": "test",
                    "timepoint": "T0",
                    "plate_id": "plate",
                    "plant_id": t,
                    "track_id": t,
                    "genotype": np.nan,
                    "treatment": np.nan,
                    "frame": frame,
                    "tip_x": np.nan,
                    "tip_y": np.nan,
                }
            )
    inputs = CircumnutationInputs(trajectory_df=pd.DataFrame(rows), cadence_s=300.0)
    per_plant_df, _, _ = pipeline.compute_traits(inputs)
    assert list(per_plant_df.columns) == _expected_46_columns()
    assert len(per_plant_df) == 3
    assert per_plant_df["is_nutating"].dtype == np.bool_
    assert per_plant_df["handedness"].dtype == np.int64
    assert per_plant_df["growth_axis_unreliable"].dtype == np.bool_


def test_compute_traits_echoes_unmodified_trajectory_df():
    """The echoed trajectory_df is the unmodified input object."""
    inputs = _inputs(n_tracks=2)
    before = inputs.trajectory_df.copy()
    _, echoed, _ = pipeline.compute_traits(inputs)
    assert echoed is inputs.trajectory_df
    pd.testing.assert_frame_equal(echoed, before)
