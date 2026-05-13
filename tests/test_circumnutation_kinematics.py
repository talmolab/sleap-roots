"""Tier 0 raw kinematic-trait tests for ``sleap_roots.circumnutation.kinematics``.

Exercises the trait set, growth-axis reliability gate, shared helper
modules (``_noise``, ``_geometry``), and the ``_io._build_per_plant_template_from_df``
refactor established by the OpenSpec change
``add-circumnutation-tier0-kinematics`` (PR #2).

Spec deltas under test:

- Requirement: Tier 0 raw kinematic traits — scenarios 2.B.1–2.B.8 + 2.A.x
  in ``tasks.md`` map onto the ``compute(trajectory_df, constants=None)``
  output contract.
- Requirement: Growth-axis reliability gate — scenarios 2.C.x exercise the
  ``D < K * sg_residual_xy`` rule with strict less-than.
- Requirement: Tier 0 helper modules — scenarios 2.D.x exercise
  ``_noise.compute_sg_residual_xy`` and ``_geometry.compute_psi_g``.
- Requirement: Tier 0 input-validation boundary — scenario 2.A.6 exercises
  ``ValueError`` for non-DataFrame and missing-column inputs.
- Requirement: Per-plant template helper — scenarios 2.E.x exercise the
  ``_build_per_plant_template_from_df`` private helper.

Per-fixture integration smoke / sanity tests:

- 2.F.x KitaakeX smoke: load ``tests/data/circumnutation_plate/...`` via
  ``Series.get_tracked_tips``, enrich with the 4 row-identity columns
  ``TrackedTipPipeline`` does not emit, run ``kinematics.compute``,
  assert structural correctness only (no value-equality assertions).
- 2.G.x Nipponbare reference-value sanity: load
  ``tests/data/circumnutation_nipponbare_plate_001/...``, run, assert
  per-track median values are within tolerance bands locked in by
  ``tasks.md`` section 4.4 after one calibration run.

Theory references: ``docs/circumnutation/theory.md`` §7.1 (trait list),
§3.5 (`ψ_g(t)` BM-Eq.-20 convention), §2.1 (image-y-down).
"""

import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from sleap_roots.circumnutation import _constants


# ---------------------------------------------------------------------------
# Expected output schema (locked by spec scenario "Output DataFrame columns
# are in the specified order")
# ---------------------------------------------------------------------------

ROW_IDENTITY_COLUMNS = (
    "series",
    "sample_uid",
    "timepoint",
    "plate_id",
    "plant_id",
    "track_id",
    "genotype",
    "treatment",
)

TIER0_TRAIT_COLUMNS = (
    "v_total_median_px_per_frame",
    "v_long_signed_median_px_per_frame",
    "v_long_abs_median_px_per_frame",
    "v_lat_signed_median_px_per_frame",
    "v_lat_abs_median_px_per_frame",
    "long_lat_ratio",
    "path_displacement_ratio",
    "angular_amplitude",
    "principal_axis_angle",
    "growth_axis_unreliable",
)

EXPECTED_COLUMNS = ROW_IDENTITY_COLUMNS + TIER0_TRAIT_COLUMNS

ROTATION_DEPENDENT_TRAITS = (
    "v_long_signed_median_px_per_frame",
    "v_long_abs_median_px_per_frame",
    "v_lat_signed_median_px_per_frame",
    "v_lat_abs_median_px_per_frame",
    "long_lat_ratio",
    "principal_axis_angle",
)

ROTATION_INVARIANT_TRAITS = (
    "v_total_median_px_per_frame",
    "path_displacement_ratio",
    "angular_amplitude",
)


# ---------------------------------------------------------------------------
# Fixture builders for synthetic trajectories
# ---------------------------------------------------------------------------


def _build_track_df(
    track_id: int,
    tip_x: np.ndarray,
    tip_y: np.ndarray,
    frames: Optional[np.ndarray] = None,
    series: str = "plate_001",
    sample_uid: str = "test_sample",
    timepoint: str = "T0",
    plate_id: str = "plate_001",
    genotype: float = np.nan,
    treatment: float = np.nan,
) -> pd.DataFrame:
    """Build a single-track trajectory DataFrame with all 8 row-identity columns + per-frame columns."""
    if frames is None:
        frames = np.arange(len(tip_x))
    return pd.DataFrame(
        {
            "series": series,
            "sample_uid": sample_uid,
            "timepoint": timepoint,
            "plate_id": plate_id,
            "plant_id": track_id,
            "track_id": track_id,
            "genotype": genotype,
            "treatment": treatment,
            "frame": frames,
            "tip_x": tip_x.astype(float),
            "tip_y": tip_y.astype(float),
        }
    )


def _build_multi_track_df(n_tracks: int = 6, n_frames: int = 10) -> pd.DataFrame:
    """Mirror the foundation test's ``valid_trajectory_df`` for 6 tracks."""
    rows = []
    for track in range(n_tracks):
        for frame in range(n_frames):
            rows.append(
                {
                    "series": "plate_001",
                    "sample_uid": "test_sample",
                    "timepoint": "T0",
                    "plate_id": "plate_001",
                    "plant_id": track,
                    "track_id": track,
                    "genotype": np.nan,
                    "treatment": np.nan,
                    "frame": frame,
                    "tip_x": float(track + frame * 1.0),
                    "tip_y": float(track * 10),
                }
            )
    return pd.DataFrame(rows)


# ===========================================================================
# 2.A — Trait set schema and structural tests
# ===========================================================================


def test_2A1_compute_returns_per_plant_dataframe():
    """2.A.1 — Output is a DataFrame with one row per (5-tuple) track."""
    from sleap_roots.circumnutation import kinematics

    df = _build_multi_track_df(n_tracks=6, n_frames=10)
    result = kinematics.compute(df)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 6


def test_2A2_output_columns_match_spec():
    """2.A.2 — Columns are 8 row-identity + 10 trait columns in declared order."""
    from sleap_roots.circumnutation import kinematics

    df = _build_multi_track_df(n_tracks=6, n_frames=10)
    result = kinematics.compute(df)
    assert tuple(result.columns) == EXPECTED_COLUMNS
    assert len(result.columns) == 18


def test_2A3_unit_columns_match_vocabulary():
    """2.A.3 — All trait unit strings are in PIPELINE_UNIT_VOCABULARY."""
    from sleap_roots.circumnutation import kinematics

    # _TIER0_TRAIT_UNITS is the module-level units mapping for the 10 new columns
    assert hasattr(kinematics, "_TIER0_TRAIT_UNITS")
    units = kinematics._TIER0_TRAIT_UNITS
    assert set(units.keys()) == set(TIER0_TRAIT_COLUMNS)
    for col, unit in units.items():
        assert (
            unit in _constants.PIPELINE_UNIT_VOCABULARY
        ), f"unit {unit!r} for {col!r} not in PIPELINE_UNIT_VOCABULARY"
    # Specific expected unit assignments
    assert units["v_total_median_px_per_frame"] == "px/frame"
    assert units["v_long_signed_median_px_per_frame"] == "px/frame"
    assert units["v_long_abs_median_px_per_frame"] == "px/frame"
    assert units["v_lat_signed_median_px_per_frame"] == "px/frame"
    assert units["v_lat_abs_median_px_per_frame"] == "px/frame"
    assert units["long_lat_ratio"] == "—"
    assert units["path_displacement_ratio"] == "—"
    assert units["angular_amplitude"] == "rad"
    assert units["principal_axis_angle"] == "rad"
    assert units["growth_axis_unreliable"] == "bool"


def test_2A4_track_id_is_integer_and_plant_id_equals_track_id():
    """2.A.4 — track_id is integer dtype; plant_id column-wise equal to track_id."""
    from sleap_roots.circumnutation import kinematics

    df = _build_multi_track_df(n_tracks=6, n_frames=10)
    result = kinematics.compute(df)
    assert pd.api.types.is_integer_dtype(result["track_id"])
    assert (result["plant_id"] == result["track_id"]).all()


def test_2A5_output_sort_order_is_numeric():
    """2.A.5 — Identity columns sorted by 5-tuple, track_id numeric not lexicographic."""
    from sleap_roots.circumnutation import kinematics

    # Build a 2-track df with track_ids 2 and 10; numeric sort puts 2 before 10
    df_t2 = _build_track_df(2, tip_x=np.arange(10.0), tip_y=np.zeros(10))
    df_t10 = _build_track_df(10, tip_x=np.arange(10.0), tip_y=np.zeros(10))
    df = pd.concat([df_t10, df_t2], ignore_index=True)
    result = kinematics.compute(df)
    assert list(result["track_id"]) == [2, 10]


@pytest.mark.parametrize(
    "bad_input",
    [None, [1, 2, 3], {"frame": []}, np.array([1.0])],
)
def test_2A6_invalid_trajectory_df_non_dataframe(bad_input):
    """2.A.6 (a) — Non-DataFrame raises ValueError mentioning 'DataFrame'."""
    from sleap_roots.circumnutation import kinematics

    with pytest.raises(ValueError, match="DataFrame"):
        kinematics.compute(bad_input)


def test_2A6_invalid_trajectory_df_missing_tip_x():
    """2.A.6 (b) — Missing tip_x raises ValueError naming 'tip_x'."""
    from sleap_roots.circumnutation import kinematics

    df = _build_multi_track_df(n_tracks=6, n_frames=10)
    df_missing = df.drop(columns=["tip_x"])
    with pytest.raises(ValueError, match="tip_x"):
        kinematics.compute(df_missing)


# ===========================================================================
# 2.B — Synthetic exact-value tests
# ===========================================================================


def test_2B1_straight_line_track():
    """2.B.1 — Straight horizontal line: x=arange, y=0 → analytical values."""
    from sleap_roots.circumnutation import kinematics

    n = 100
    df = _build_track_df(0, tip_x=np.arange(n, dtype=float), tip_y=np.zeros(n))
    result = kinematics.compute(df)
    row = result.iloc[0]
    # Velocities: unit horizontal velocity → total=1, long=1 (signed and abs), lat=0
    assert row["v_total_median_px_per_frame"] == pytest.approx(1.0)
    assert row["v_long_signed_median_px_per_frame"] == pytest.approx(1.0)
    assert row["v_long_abs_median_px_per_frame"] == pytest.approx(1.0)
    assert row["v_lat_signed_median_px_per_frame"] == pytest.approx(0.0)
    assert row["v_lat_abs_median_px_per_frame"] == pytest.approx(0.0)
    # long_lat_ratio: denominator zero → NaN
    assert math.isnan(row["long_lat_ratio"])
    # L = D = 99 → ratio = 1.0 exactly
    assert row["path_displacement_ratio"] == pytest.approx(1.0)
    # ψ_g constant → angular_amplitude = 0
    assert row["angular_amplitude"] == pytest.approx(0.0)
    # principal_axis_angle: atan2(0, 99) = 0
    assert row["principal_axis_angle"] == pytest.approx(0.0)
    # Gate does not fire (D = 99 >> noise)
    assert bool(row["growth_axis_unreliable"]) is False


def test_2B2_straight_line_track_image_y_down():
    """2.B.2 — Vertical line (image y increases downward): principal_axis_angle = +π/2."""
    from sleap_roots.circumnutation import kinematics

    n = 100
    df = _build_track_df(0, tip_x=np.zeros(n), tip_y=np.arange(n, dtype=float))
    result = kinematics.compute(df)
    row = result.iloc[0]
    assert row["principal_axis_angle"] == pytest.approx(math.pi / 2)


def test_2B3_pure_noise_track():
    """2.B.3 — Pure i.i.d. N(0,1) noise: gate fires, rotation-dependent NaN'd, invariants finite."""
    from sleap_roots.circumnutation import kinematics

    rng = np.random.default_rng(0)
    noise = rng.normal(0, 1, size=(2, 100))
    df = _build_track_df(0, tip_x=noise[0], tip_y=noise[1])
    result = kinematics.compute(df)
    row = result.iloc[0]
    # Gate fires (D ≈ 1.7, residual ≈ √2 ≈ 1.4 → ratio ≈ 1.2 << 10)
    assert bool(row["growth_axis_unreliable"]) is True
    # All 6 rotation-dependent traits are NaN
    for col in ROTATION_DEPENDENT_TRAITS:
        assert math.isnan(row[col]), f"{col} should be NaN under gate"
    # All 3 rotation-invariant traits are finite (not NaN)
    for col in ROTATION_INVARIANT_TRAITS:
        assert not math.isnan(row[col]), f"{col} should be finite (not NaN'd by gate)"
    # v_total_median is positive
    assert row["v_total_median_px_per_frame"] > 0
    # path_displacement_ratio finite and > 1 (random walk has L > D)
    assert math.isfinite(row["path_displacement_ratio"])
    assert row["path_displacement_ratio"] > 1
    # angular_amplitude is finite
    assert math.isfinite(row["angular_amplitude"])


def test_2B4_circular_trajectory():
    """2.B.4 — Circle (R=10) + small drift: angular_amplitude ≈ 2π; gate does not fire.

    Construction note: the circle radius (R=10) is set so the circle's
    velocity magnitude (R·2π/n ≈ 0.628 px/frame) dominates the drift
    (0.1 px/frame). This guarantees ``dx = -R·sin(t)·dt + 0.1`` swings
    through both signs over one revolution, so ``ψ_g = atan2(dx, dy)``
    sweeps the full 2π range. A smaller R with the same drift (e.g.
    R=1, drift=0.1) would have drift > circle-velocity and ψ_g would
    NOT complete a full revolution.
    """
    from sleap_roots.circumnutation import kinematics

    n = 100
    t = np.arange(n)
    R = 10.0
    drift_per_frame = 0.1
    tip_x = R * np.cos(2 * np.pi * t / n) + drift_per_frame * t
    tip_y = R * np.sin(2 * np.pi * t / n)
    df = _build_track_df(0, tip_x=tip_x, tip_y=tip_y)
    result = kinematics.compute(df)
    row = result.iloc[0]
    # One full revolution → angular_amplitude in [2π - 0.5, 2π + 0.5]
    assert 2 * np.pi - 0.5 <= row["angular_amplitude"] <= 2 * np.pi + 0.5
    # Drift gives D ≈ 10 px, smooth trajectory has near-zero SG residual
    # → gate does not fire
    assert bool(row["growth_axis_unreliable"]) is False


def test_2B5_nan_rows_dropped_before_diff():
    """2.B.5 — NaN drop precedes diff: path_displacement_ratio == 1.0, v_total == 1.0 exactly.

    This is the load-bearing invariant from the spec scenario
    "NaN rows are dropped BEFORE diff (ordering is load-bearing)".
    """
    from sleap_roots.circumnutation import kinematics

    n = 100
    tip_x = np.arange(n, dtype=float)
    tip_y = np.zeros(n)
    # Inject NaN into 10 random rows (excluding the first and last so D is preserved)
    rng = np.random.default_rng(0)
    nan_idx = rng.choice(np.arange(1, n - 1), size=10, replace=False)
    tip_x[nan_idx] = np.nan
    df = _build_track_df(0, tip_x=tip_x, tip_y=tip_y)
    result = kinematics.compute(df)
    row = result.iloc[0]
    # Explicit assertions per spec scenario
    assert row["path_displacement_ratio"] == pytest.approx(1.0)
    assert row["v_total_median_px_per_frame"] == pytest.approx(1.0)
    # Other traits should match the no-NaN case
    assert row["v_long_signed_median_px_per_frame"] == pytest.approx(1.0)
    assert row["v_lat_abs_median_px_per_frame"] == pytest.approx(0.0)


def test_2B6_frame_gaps_handled_gap_aware():
    """2.B.6 — 10-frame gap: per-frame velocity normalized by Δframe; median still 1.0 exactly."""
    from sleap_roots.circumnutation import kinematics

    # Frames [0..40) + [50..100): drop the [40..50) window
    frames = np.concatenate([np.arange(40), np.arange(50, 100)])
    tip_x = frames.astype(float)  # x = frame value → unit velocity per frame
    tip_y = np.zeros(len(frames))
    df = _build_track_df(0, tip_x=tip_x, tip_y=tip_y, frames=frames)
    result = kinematics.compute(df)
    row = result.iloc[0]
    # Across the gap: Δxy = 50 - 39 = 11, Δframe = 50 - 39 = 11 → velocity = 1.0
    # All other frames: Δxy = 1, Δframe = 1 → velocity = 1.0
    # Median is 1.0 exactly
    assert row["v_total_median_px_per_frame"] == pytest.approx(1.0)


def test_2B7_insufficient_frames_yields_nan():
    """2.B.7 — Single-frame track: all 9 trait columns NaN; flag False."""
    from sleap_roots.circumnutation import kinematics

    df = _build_track_df(0, tip_x=np.array([5.0]), tip_y=np.array([3.0]))
    result = kinematics.compute(df)
    row = result.iloc[0]
    # All 9 trait columns are NaN
    for col in set(TIER0_TRAIT_COLUMNS) - {"growth_axis_unreliable"}:
        assert math.isnan(row[col]), f"{col} should be NaN with < 2 frames"
    # Flag is False — cannot judge reliability with < 2 frames
    assert bool(row["growth_axis_unreliable"]) is False


def test_2B8_zero_displacement_yields_nan_for_ratio():
    """2.B.8 — Closed loop (xy[-1] == xy[0]): path_displacement_ratio NaN, gate fires."""
    from sleap_roots.circumnutation import kinematics

    # A small triangular path returning to origin: (0,0) → (1,0) → (0,1) → (0,0)
    tip_x = np.array([0.0, 1.0, 0.0, 0.0])
    tip_y = np.array([0.0, 0.0, 1.0, 0.0])
    df = _build_track_df(0, tip_x=tip_x, tip_y=tip_y)
    result = kinematics.compute(df)
    row = result.iloc[0]
    assert math.isnan(row["path_displacement_ratio"])
    assert bool(row["growth_axis_unreliable"]) is True
    # 6 rotation-dependent NaN'd
    for col in ROTATION_DEPENDENT_TRAITS:
        assert math.isnan(row[col]), f"{col} should be NaN when D=0"
    # Rotation-invariants finite (except path_displacement_ratio which is NaN by its own rule)
    assert math.isfinite(row["v_total_median_px_per_frame"])
    assert math.isfinite(row["angular_amplitude"])


def test_2B9_signed_lateral_near_zero_for_circular():
    """2.B.9 — Circular trajectory: v_lat_signed_median ≈ 0 (symmetry sanity check)."""
    from sleap_roots.circumnutation import kinematics

    n = 100
    t = np.arange(n)
    tip_x = np.cos(2 * np.pi * t / n) + 0.1 * t
    tip_y = np.sin(2 * np.pi * t / n)
    df = _build_track_df(0, tip_x=tip_x, tip_y=tip_y)
    result = kinematics.compute(df)
    row = result.iloc[0]
    # Signed lateral median should be much smaller in magnitude than abs lateral median
    assert (
        abs(row["v_lat_signed_median_px_per_frame"])
        < 0.1 * row["v_lat_abs_median_px_per_frame"]
    )


# ===========================================================================
# 2.C — Growth-axis reliability gate tests
# ===========================================================================


def _make_track_with_D(D: float, n: int = 10) -> pd.DataFrame:
    """Build a straight-line track along +x with net displacement exactly D."""
    tip_x = np.linspace(0, D, n)
    tip_y = np.zeros(n)
    return _build_track_df(0, tip_x=tip_x, tip_y=tip_y)


def test_2C1_gate_fires_when_D_below_threshold(monkeypatch):
    """2.C.1 — D=5 px, residual=1 px, K=10 → ratio 5 < 10 → flag True."""
    from sleap_roots.circumnutation import _noise, kinematics

    monkeypatch.setattr(_noise, "compute_sg_residual_xy", lambda *a, **k: 1.0)
    df = _make_track_with_D(5.0)
    result = kinematics.compute(df)
    assert bool(result.iloc[0]["growth_axis_unreliable"]) is True


def test_2C2_gate_does_not_fire_when_D_above_threshold(monkeypatch):
    """2.C.2 — D=100 px, residual=1 px, K=10 → ratio 100 > 10 → flag False."""
    from sleap_roots.circumnutation import _noise, kinematics

    monkeypatch.setattr(_noise, "compute_sg_residual_xy", lambda *a, **k: 1.0)
    df = _make_track_with_D(100.0)
    result = kinematics.compute(df)
    assert bool(result.iloc[0]["growth_axis_unreliable"]) is False


def test_2C3_gate_threshold_overridable_via_constants(monkeypatch):
    """2.C.3 — K=3, D=5, residual=1 → ratio 5 > 3 → flag False (strict less-than)."""
    from sleap_roots.circumnutation import _constants as cst
    from sleap_roots.circumnutation import _noise, kinematics

    monkeypatch.setattr(_noise, "compute_sg_residual_xy", lambda *a, **k: 1.0)
    df = _make_track_with_D(5.0)
    result = kinematics.compute(
        df, constants=cst.ConstantsT(GROWTH_AXIS_RELIABILITY_K=3)
    )
    assert bool(result.iloc[0]["growth_axis_unreliable"]) is False
    # Same data with K=10 → flag True
    result_k10 = kinematics.compute(
        df, constants=cst.ConstantsT(GROWTH_AXIS_RELIABILITY_K=10)
    )
    assert bool(result_k10.iloc[0]["growth_axis_unreliable"]) is True


def test_2C4_gate_threshold_strict_less_than_at_boundary(monkeypatch):
    """2.C.4 — D=10.0, residual=1.0, K=10 → boundary D == K*residual → flag False (strict)."""
    from sleap_roots.circumnutation import _noise, kinematics

    monkeypatch.setattr(_noise, "compute_sg_residual_xy", lambda *a, **k: 1.0)
    df = _make_track_with_D(10.0)
    result = kinematics.compute(df)
    assert bool(result.iloc[0]["growth_axis_unreliable"]) is False


def test_2C5_rotation_invariant_traits_survive_gate(monkeypatch):
    """2.C.5 — When gate fires, rotation-invariant traits are NOT NaN'd."""
    from sleap_roots.circumnutation import _noise, kinematics

    monkeypatch.setattr(_noise, "compute_sg_residual_xy", lambda *a, **k: 1.0)
    df = _make_track_with_D(5.0)
    result = kinematics.compute(df)
    row = result.iloc[0]
    assert bool(row["growth_axis_unreliable"]) is True
    for col in ROTATION_INVARIANT_TRAITS:
        assert not math.isnan(
            row[col]
        ), f"{col} (rotation-invariant) should survive gate"


def test_2C6_rotation_dependent_traits_nan_when_gate_fires(monkeypatch):
    """2.C.6 — When gate fires, all 6 rotation-dependent columns are NaN."""
    from sleap_roots.circumnutation import _noise, kinematics

    monkeypatch.setattr(_noise, "compute_sg_residual_xy", lambda *a, **k: 1.0)
    df = _make_track_with_D(5.0)
    result = kinematics.compute(df)
    row = result.iloc[0]
    assert bool(row["growth_axis_unreliable"]) is True
    for col in ROTATION_DEPENDENT_TRAITS:
        assert math.isnan(row[col]), f"{col} should be NaN when gate fires"


# ===========================================================================
# 2.D — Helper module tests
# ===========================================================================


def test_2D1_sg_residual_zero_for_polynomial():
    """2.D.1 — SG-residual on a polynomial of degree ≤ SG_DEGREE → ~0."""
    from sleap_roots.circumnutation._noise import compute_sg_residual_xy

    # Linear in x, quadratic in y: both within SG degree 3
    x = np.arange(20, dtype=float)
    y = x**2
    res = compute_sg_residual_xy(x, y, window=5, degree=3)
    assert res == pytest.approx(0.0, abs=1e-9)


def test_2D2_sg_residual_recovers_sigma_noisy_data():
    """2.D.2 — SG-residual on smooth+iid-σ=1 noise → ≈ sqrt(2) within [1.0, 1.6]."""
    from sleap_roots.circumnutation._noise import compute_sg_residual_xy

    rng = np.random.default_rng(0)
    x_smooth = np.linspace(0, 100, 1000)
    y_smooth = np.zeros(1000)
    noise = rng.normal(0, 1.0, size=(2, 1000))
    x = x_smooth + noise[0]
    y = y_smooth + noise[1]
    res = compute_sg_residual_xy(x, y, window=5, degree=3)
    # quadrature sum sqrt(σ_x² + σ_y²) = sqrt(2) ≈ 1.414; SG under-estimates slightly
    assert 1.0 <= res <= 1.6, f"residual {res} not in [1.0, 1.6]"


def test_2D3_sg_residual_returns_nan_short_input():
    """2.D.3 — len(x) < window → returns NaN, no exception."""
    from sleap_roots.circumnutation._noise import compute_sg_residual_xy

    res = compute_sg_residual_xy(
        np.array([1.0, 2.0]), np.array([3.0, 4.0]), window=5, degree=3
    )
    assert math.isnan(res)


def test_2D4_psi_g_constant_for_straight_line():
    """2.D.4 — Straight-line input → ψ_g is constant.

    Per BM-Eq.-20 convention atan2(dx, dy):
    - x=arange, y=zeros (velocity +x): ψ_g = atan2(1, 0) = π/2
    - x=zeros, y=arange (velocity +y): ψ_g = atan2(0, 1) = 0
    """
    from sleap_roots.circumnutation._geometry import compute_psi_g

    # Case 1: velocity in +x direction
    psi_x = compute_psi_g(np.arange(100, dtype=float), np.zeros(100))
    assert len(psi_x) == 99
    assert np.allclose(psi_x, math.pi / 2)
    # Case 2: orthogonal — velocity in +y direction
    psi_y = compute_psi_g(np.zeros(100), np.arange(100, dtype=float))
    assert len(psi_y) == 99
    assert np.allclose(psi_y, 0.0)


def test_2D5_psi_g_monotonic_unwrapped_for_circular():
    """2.D.5 — Closed circle → strictly monotonic unwrapped sequence, total span ≈ 2π."""
    from sleap_roots.circumnutation._geometry import compute_psi_g

    t = np.linspace(0, 2 * np.pi, 100)
    x = np.cos(t)
    y = np.sin(t)
    psi = compute_psi_g(x, y)
    assert len(psi) == 99
    # Strictly monotonic in some direction (convention-dependent — see spec scenario)
    diffs = np.diff(psi)
    assert np.all(diffs < 0) or np.all(diffs > 0), "ψ_g should be strictly monotonic"
    # Unwrap worked — no ±2π jumps
    assert np.all(np.abs(diffs) < np.pi)
    # Total absolute span ≈ 2π within ±0.1
    span = abs(psi[-1] - psi[0])
    assert abs(span - 2 * np.pi) < 0.1


def test_2D6_psi_g_phase_wrapping():
    """2.D.6 — A trajectory crossing the ±π branch cut unwraps continuously.

    Construction: a CW circle starting at velocity (0, +y) and crossing through
    velocity (-x, 0), (0, -y), (+x, 0). atan2(dx, dy) hits the branch cut.
    """
    from sleap_roots.circumnutation._geometry import compute_psi_g

    t = np.linspace(0, 2 * np.pi, 200)
    # CCW circle starting at (1, 0)
    x = np.cos(t)
    y = np.sin(t)
    psi = compute_psi_g(x, y)
    # No discontinuity > π means unwrap worked
    diffs = np.diff(psi)
    assert np.all(np.abs(diffs) < np.pi), "unwrap should remove ±2π discontinuities"


def test_2D7_psi_g_empty_for_short_input():
    """2.D.7 — len(x) < 2 returns empty array, no exception."""
    from sleap_roots.circumnutation._geometry import compute_psi_g

    res = compute_psi_g(np.array([1.0]), np.array([2.0]))
    assert isinstance(res, np.ndarray)
    assert res.shape == (0,)


# ===========================================================================
# 2.E — _io.py template helper tests
# ===========================================================================


def test_2E1_private_helper_matches_public_wrapper():
    """2.E.1 — _build_per_plant_template_from_df(df) ≡ build_per_plant_template(inputs)."""
    from sleap_roots.circumnutation import CircumnutationInputs
    from sleap_roots.circumnutation._io import (
        _build_per_plant_template_from_df,
        build_per_plant_template,
    )

    df = _build_multi_track_df(n_tracks=6, n_frames=10)
    inputs = CircumnutationInputs(trajectory_df=df, cadence_s=300.0)

    public = build_per_plant_template(inputs)
    private = _build_per_plant_template_from_df(df)

    pd.testing.assert_frame_equal(public, private)


def test_2E2_private_helper_rejects_nan_track_id():
    """2.E.2 (a) — NaN track_id raises ValueError naming track_id."""
    from sleap_roots.circumnutation._io import _build_per_plant_template_from_df

    df = _build_multi_track_df(n_tracks=6, n_frames=10)
    df.loc[0, "track_id"] = np.nan
    with pytest.raises(ValueError, match="track_id"):
        _build_per_plant_template_from_df(df)


def test_2E2_private_helper_5tuple_conflict_check():
    """2.E.2 (b) — Conflicting genotype for same 5-tuple raises ValueError."""
    from sleap_roots.circumnutation._io import _build_per_plant_template_from_df

    df = _build_multi_track_df(n_tracks=2, n_frames=5)
    # Cast genotype to object dtype before assigning strings; the foundation
    # builder produces float64 (all-NaN) and pandas 2.3 warns on incompatible
    # dtype assignment.
    df["genotype"] = df["genotype"].astype(object)
    # Same 5-tuple but different genotype in some rows
    df.loc[0, "genotype"] = "Wt"
    df.loc[1, "genotype"] = "Mutant"
    with pytest.raises(ValueError, match="genotype"):
        _build_per_plant_template_from_df(df)


def test_2E3_public_wrapper_preserves_foundation_api():
    """2.E.3 — build_per_plant_template(inputs) regression unchanged."""
    from sleap_roots.circumnutation import CircumnutationInputs
    from sleap_roots.circumnutation._io import build_per_plant_template

    df = _build_multi_track_df(n_tracks=6, n_frames=10)
    inputs = CircumnutationInputs(trajectory_df=df, cadence_s=300.0)
    result = build_per_plant_template(inputs)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 6  # 6 unique 5-tuples
    assert tuple(result.columns) == ROW_IDENTITY_COLUMNS


# ===========================================================================
# 2.F — KitaakeX integration smoke test
# ===========================================================================


KITAAKEX_SLP = Path("tests/data/circumnutation_plate/plate_001_greyscale.tracked.slp")
KITAAKEX_CSV = Path("tests/data/circumnutation_plate/fixture_metadata.csv")


def _load_and_enrich(slp_path: Path, csv_path: Path, genotype: str) -> pd.DataFrame:
    """Load a tracked .slp via ``Series.get_tracked_tips`` and enrich with the 4 missing row-identity columns.

    The proofread-dedup contract (prelim §3.1: user-corrected takes
    precedence over predicted when both exist for the same frame and
    track) is honored upstream by ``Series.get_tracked_tips`` — see
    ``tests/test_tracked_tip_pipeline.py`` §13 (dedup tests added in
    PR #2). This helper therefore only needs to convert ``track_id`` from
    ``"track_N"`` strings to integer ``N`` and add the 4 row-identity
    columns ``TrackedTipPipeline`` doesn't emit.
    """
    from sleap_roots.series import Series

    series = Series.load(
        series_name="plate_001",
        primary_path=str(slp_path),
        csv_path=str(csv_path),
        sample_uid="plate_001",
    )
    df = series.get_tracked_tips()

    # Convert track_id "track_N" → int N
    df["track_id"] = df["track_id"].str.replace("track_", "", regex=False).astype(int)
    df["series"] = series.series_name
    df["sample_uid"] = series.sample_uid
    df["timepoint"] = str(series.timepoint) if not pd.isna(series.timepoint) else np.nan
    df["plate_id"] = "plate_001"
    df["plant_id"] = df["track_id"]
    df["genotype"] = genotype
    df["treatment"] = "MOCK"
    return df


@pytest.mark.skipif(
    not KITAAKEX_SLP.exists(),
    reason=f"KitaakeX fixture not present: {KITAAKEX_SLP}",
)
def test_2F1_kitaakex_smoke():
    """2.F.1 — KitaakeX fixture end-to-end smoke: 6 rows × 18 columns, no value assertions."""
    from sleap_roots.circumnutation import kinematics
    from sleap_roots.circumnutation._io import write_per_plant_csv

    df = _load_and_enrich(KITAAKEX_SLP, KITAAKEX_CSV, genotype="KitaakeX")
    result = kinematics.compute(df)
    # 6 rows
    assert len(result) == 6
    # 18 columns in expected order
    assert tuple(result.columns) == EXPECTED_COLUMNS
    # Rotation-invariant traits are finite for all rows
    for col in ROTATION_INVARIANT_TRAITS:
        assert (
            result[col].notna().all()
        ), f"{col} should be finite for all 6 KitaakeX tracks"
    # NaN pattern in 6 rotation-dependent columns matches growth_axis_unreliable
    for col in ROTATION_DEPENDENT_TRAITS:
        # For each row, if flag is True the trait is NaN; if False the trait is finite
        for idx, row in result.iterrows():
            if bool(row["growth_axis_unreliable"]):
                assert math.isnan(
                    row[col]
                ), f"row {idx}, {col}: flag True but trait not NaN"
            else:
                assert not math.isnan(
                    row[col]
                ), f"row {idx}, {col}: flag False but trait is NaN"
    # v_total_median values are all positive (real motion)
    assert (result["v_total_median_px_per_frame"] > 0).all()
    # Round-trip through write_per_plant_csv to validate units vocabulary
    import tempfile

    units_map = dict.fromkeys(ROW_IDENTITY_COLUMNS)
    from sleap_roots.circumnutation._io import default_units_for_template

    base = default_units_for_template(result[list(ROW_IDENTITY_COLUMNS)])
    units = {**base, **kinematics._TIER0_TRAIT_UNITS}
    assert all(
        u in _constants.PIPELINE_UNIT_VOCABULARY for u in units.values()
    ), "every emitted unit must be in PIPELINE_UNIT_VOCABULARY"
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "traits.csv"
        from sleap_roots.circumnutation._io import gather_run_metadata

        run_meta = gather_run_metadata(str(KITAAKEX_SLP))
        write_per_plant_csv(out_path, result, units, run_meta)
        assert out_path.exists()


# ===========================================================================
# 2.G — Nipponbare reference-value sanity test
# ===========================================================================


NIPPONBARE_SLP = Path(
    "tests/data/circumnutation_nipponbare_plate_001/plate_001_greyscale.tracked_proofread.slp"
)
NIPPONBARE_CSV = Path(
    "tests/data/circumnutation_nipponbare_plate_001/fixture_metadata.csv"
)


# Tolerances locked during section 4.4 calibration on 2026-05-12.
#
# Sanity-floor cross-check (per tasks.md 4.4): each captured median must fall
# within ±50% of the prelim §4.1 anchor.
# - v_total_median_px_per_frame: impl 6.93 → matches prelim §4.1 reported
#   MEDIAN per-frame step (6.93 px) EXACTLY (within IEEE float). PASS.
# - path_displacement_ratio: impl 1.32 → matches prelim §4.1 reported L/D
#   (1.36) within 3%. PASS.
# - v_long_abs / v_lat_abs / long_lat_ratio: prelim §4.1 reports MEANS, not
#   medians. For the heavy-tailed |Δ_lat| distribution (nutation peaks +
#   many small steps), median is ~3× smaller than mean — this is genuine
#   mean-vs-median asymmetry, not a calibration error. These three traits
#   are therefore **snapshot** tests anchored to the impl's captured values.
#
# After sanity-floor passed for v_total_median AND path_displacement_ratio,
# captured values are locked with ±10% (medians) and ±15% (ratios) per
# tasks.md 4.4.
NIPPONBARE_TOL = {
    # v_total_median: impl 6.93 = prelim median 6.93. ±10%.
    "v_total_median_px_per_frame": (6.24, 7.63),
    # v_long_abs_median: impl 6.16. ±10% snapshot (no prelim median equivalent).
    "v_long_abs_median_px_per_frame": (5.54, 6.77),
    # v_lat_abs_median: impl 0.89. ±10% snapshot (no prelim median equivalent).
    "v_lat_abs_median_px_per_frame": (0.80, 0.98),
    # long_lat_ratio: impl 6.82. ±15% snapshot (no prelim median equivalent).
    "long_lat_ratio": (5.80, 7.85),
    # path_displacement_ratio: impl 1.32, prelim L/D 1.36. ±15%.
    "path_displacement_ratio": (1.12, 1.51),
}

# All 6 Nipponbare proofread tracks pass the growth-axis reliability gate
# under the corrected dedup behavior (PR #2's _load_and_enrich now honors
# the prelim §3.1 "user-corrected takes precedence" convention; the prior
# calibration's "3 of 6 tracks gate-fail" finding was an artifact of
# `Series.get_tracked_tips` returning both predicted AND user-corrected
# instances for proofread frames, NOT a real signal about Nipponbare data
# quality). See follow-up issue
# ``issue_series_get_tracked_tips_proofread_dedup.md`` for the upstream fix.
NIPPONBARE_EXPECTED_CLEAN_TRACKS = 6
NIPPONBARE_EXPECTED_GATED_TRACKS = 0


@pytest.mark.skipif(
    not NIPPONBARE_SLP.exists(),
    reason=f"Nipponbare fixture not present: {NIPPONBARE_SLP}",
)
def test_2G1_nipponbare_reference_values():
    """2.G.1 — Nipponbare reference-value sanity test (tolerances locked in 4.4).

    See ``NIPPONBARE_TOL`` above for the calibration provenance, including
    why both ``v_total_median_px_per_frame`` AND ``path_displacement_ratio``
    have real prelim sanity-floor anchors (median match) and the other
    three traits are snapshot tests.

    All 6 Nipponbare proofread tracks pass the growth-axis reliability gate
    under the corrected dedup behavior (see ``NIPPONBARE_EXPECTED_CLEAN_TRACKS``
    and the upstream fix to ``Series.get_tracked_tips`` shipped in this PR).
    Visual inspection in the SLEAP GUI confirms all 6 plants grow healthily
    from the top to the bottom of the plate.
    """
    from sleap_roots.circumnutation import kinematics

    df = _load_and_enrich(NIPPONBARE_SLP, NIPPONBARE_CSV, genotype="Nipponbare")
    result = kinematics.compute(df)
    assert len(result) == 6

    # Median-across-tracks for each tolerance-checked trait. `.median()`
    # skips NaN by default — under the corrected dedup behavior, all 6
    # tracks are gate-clean for this fixture, so the median is over all 6
    # tracks for every trait (rotation-dependent and rotation-invariant).
    for trait, (lo, hi) in NIPPONBARE_TOL.items():
        med = float(result[trait].median())
        assert lo <= med <= hi, f"{trait} median {med:.4f} not in [{lo:.4f}, {hi:.4f}]"

    # 3 of 6 tracks gate-clean, 3 gate-fired — empirically documented split.
    n_clean = int((~result["growth_axis_unreliable"]).sum())
    n_gated = int(result["growth_axis_unreliable"].sum())
    assert n_clean == NIPPONBARE_EXPECTED_CLEAN_TRACKS
    assert n_gated == NIPPONBARE_EXPECTED_GATED_TRACKS

    # Real nutation present: angular_amplitude > 0 for all tracks (rotation-
    # invariant — survives the gate).
    assert (result["angular_amplitude"] > 0).all()
