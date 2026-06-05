"""Tests for ``sleap_roots.circumnutation.psi_g`` (PR #7, Tier 2 ψ_g).

Mirrors the PR #6 ``test_circumnutation_nutation.py`` taxonomy, adapted to
Tier 2's 4 self-contained ψ_g traits:

- §1 ``_geometry.compute_signed_area`` sign convention (y-down Shoelace),
  pinned by an absolute hand-built orbit + degenerate cases.
- §2 schema/structure (8 identity + 4 trait columns, dtypes, order).
- §3 input-validation boundary.
- §4 raw CWT-free traits (handedness, delta_E, helix) + conditioning isolation.
- §5 ``T_psig_median_s`` CWT path (±10% recovery; min-length + zero-energy guards).
- §6 degenerate / edge cases (the spec degenerate table).
- §7 determinism (CC-6) + the one-DEBUG-record logging contract.
- §8 cross-tier consistency vs Tier 0 ``principal_axis_angle``.

Anchors: spec delta at
``openspec/changes/add-circumnutation-tier2-psi-g/specs/circumnutation/spec.md``;
design.md D1–D9 + the §13 reconciliation log;
theory.md §3.5 (BM2016 Eq. 20 — ψ_g) + §7.3 (Tier 2 trait table).
"""

import numpy as np
import pandas as pd

from sleap_roots.circumnutation import psi_g, synthetic
from sleap_roots.circumnutation._constants import ConstantsT
from sleap_roots.circumnutation._geometry import compute_psi_g, compute_signed_area
from sleap_roots.circumnutation._types import ROW_IDENTITY_COLUMNS

_PSIG_TRAIT_COLUMNS = (
    "T_psig_median_s",
    "delta_E_amplitude_proxy_px_per_frame",
    "handedness",
    "helix_signed_area_px2",
)


def _make_track_df(
    *,
    n_frames: int = 64,
    track_id: int = 0,
    amplitude_px: float = 10.0,
    period_frames: float = 11.0,
    growth_px_per_frame: float = 2.0,
    x0: float = 100.0,
    y0: float = 100.0,
    handedness: int = 1,
) -> pd.DataFrame:
    """Build a single-track trajectory_df with a lateral wobble + linear growth.

    Image-y-down: growth advances ``tip_y``; the nutation wobble is a sinusoid
    in ``tip_x``. ``handedness`` flips the sweep direction. Enough frames
    (``n_frames``) for the SG-detrend/CWT path when ≥ 24.
    """
    frames = np.arange(n_frames, dtype=float)
    phase = handedness * 2.0 * np.pi * frames / period_frames
    x = x0 + amplitude_px * np.sin(phase)
    y = y0 + growth_px_per_frame * frames
    return pd.DataFrame(
        {
            "series": "s",
            "sample_uid": "u",
            "timepoint": "T0",
            "plate_id": "p",
            "plant_id": int(track_id),
            "track_id": int(track_id),
            "genotype": np.nan,
            "treatment": np.nan,
            "frame": frames.astype(int),
            "tip_x": x,
            "tip_y": y,
        }
    )


def _make_multi_track_df(n_tracks: int = 3, **kwargs) -> pd.DataFrame:
    """Concatenate ``n_tracks`` single-track DataFrames (distinct track_ids)."""
    return pd.concat(
        [_make_track_df(track_id=i, **kwargs) for i in range(n_tracks)],
        ignore_index=True,
    )


# ===========================================================================
# §1 — _geometry.compute_signed_area (y-down Shoelace sign convention)
# ===========================================================================


def test_1_signed_area_absolute_anchor_is_minus_one():
    """§1: hand-built orbit [0,1,1,0],[0,0,1,1] → exactly -1.0 (y-down negation).

    The standard Shoelace of this vertex order is +1.0; the y-down-corrected
    (negated) form returns -1.0. This is the absolute, machinery-free anchor
    that breaks the handedness↔area joint-flip degeneracy.
    """
    x = np.array([0.0, 1.0, 1.0, 0.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    assert compute_signed_area(x, y) == -1.0


def test_1_signed_area_sign_agrees_with_handedness_on_anchor():
    """§1: on the same anchor orbit, sign(area) == sign(net ψ_g) (== -1).

    Independent of the psi_g.compute machinery: net unwrapped ψ_g change is
    -π → handedness -1; the negated Shoelace area is -1.0 → sign -1. The two
    convention-critical helpers agree.
    """
    x = np.array([0.0, 1.0, 1.0, 0.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    psi = compute_psi_g(x, y)
    net = float(psi[-1] - psi[0])
    assert int(np.sign(net)) == int(np.sign(compute_signed_area(x, y))) == -1


def test_1_signed_area_fewer_than_three_points_is_zero():
    """§1: a degenerate polygon (< 3 points) has area 0.0 at the helper level."""
    assert compute_signed_area(np.array([0.0, 1.0]), np.array([0.0, 1.0])) == 0.0
    assert compute_signed_area(np.array([5.0]), np.array([7.0])) == 0.0
    assert compute_signed_area(np.array([]), np.array([])) == 0.0


def test_1_signed_area_non_finite_propagates_nan():
    """§1: a non-finite coordinate propagates to a NaN area (caller guards first)."""
    x = np.array([0.0, 1.0, np.nan, 0.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    assert np.isnan(compute_signed_area(x, y))


def test_1_signed_area_sign_flips_with_traversal_direction():
    """§1: reversing the vertex order flips the sign (orientation-sensitive)."""
    x = np.array([0.0, 1.0, 1.0, 0.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    forward = compute_signed_area(x, y)
    backward = compute_signed_area(x[::-1], y[::-1])
    assert forward == -backward
    assert forward == -1.0


# ===========================================================================
# §2 — schema / structure
# ===========================================================================


def test_2_compute_returns_dataframe():
    """§2: psi_g.compute returns a pandas DataFrame."""
    df = _make_track_df(n_frames=64)
    result = psi_g.compute(df, cadence_s=300.0)
    assert isinstance(result, pd.DataFrame)


def test_2_columns_in_declared_order():
    """§2: 8 row-identity columns then the 4 trait columns in declared order."""
    df = _make_multi_track_df(n_tracks=3, n_frames=64)
    result = psi_g.compute(df, cadence_s=300.0)
    expected = list(ROW_IDENTITY_COLUMNS) + list(_PSIG_TRAIT_COLUMNS)
    assert list(result.columns) == expected


def test_2_trait_dtypes_three_float64_one_int64():
    """§2: T_psig/delta_E/helix are float64; handedness is int64."""
    df = _make_multi_track_df(n_tracks=3, n_frames=64)
    result = psi_g.compute(df, cadence_s=300.0)
    assert result["T_psig_median_s"].dtype == np.float64
    assert result["delta_E_amplitude_proxy_px_per_frame"].dtype == np.float64
    assert result["helix_signed_area_px2"].dtype == np.float64
    assert result["handedness"].dtype == np.int64


def test_2_one_row_per_track_with_unique_5tuple():
    """§2: one row per unique (series, sample_uid, plate_id, plant_id, track_id)."""
    df = _make_multi_track_df(n_tracks=3, n_frames=64)
    result = psi_g.compute(df, cadence_s=300.0)
    assert len(result) == 3
    key = ["series", "sample_uid", "plate_id", "plant_id", "track_id"]
    assert result[key].duplicated().sum() == 0


def test_2_trait_units_all_in_pipeline_vocabulary():
    """§2: the 4 declared trait units are all members of PIPELINE_UNIT_VOCABULARY."""
    from sleap_roots.circumnutation._constants import PIPELINE_UNIT_VOCABULARY

    units = {
        "T_psig_median_s": "s",
        "delta_E_amplitude_proxy_px_per_frame": "px/frame",
        "handedness": "int",
        "helix_signed_area_px2": "px²",
    }
    for col, unit in units.items():
        assert unit in PIPELINE_UNIT_VOCABULARY, f"{col} unit {unit!r} not in vocab"


# ===========================================================================
# §3 — input-validation boundary
# ===========================================================================


def test_3_non_dataframe_trajectory_df_raises_valueerror():
    """§3: a non-DataFrame trajectory_df raises ValueError."""
    import pytest

    with pytest.raises(ValueError):
        psi_g.compute([1, 2, 3], cadence_s=300.0)


def test_3_invalid_trajectory_df_missing_tip_column_raises():
    """§3: a trajectory_df missing tip_x raises ValueError (via _validate_trajectory_df)."""
    import pytest

    df = _make_track_df(n_frames=32).drop(columns=["tip_x"])
    with pytest.raises(ValueError):
        psi_g.compute(df, cadence_s=300.0)


def test_3_empty_trajectory_df_raises():
    """§3: an empty trajectory_df raises ValueError."""
    import pytest

    df = _make_track_df(n_frames=32).iloc[0:0]
    with pytest.raises(ValueError):
        psi_g.compute(df, cadence_s=300.0)


import pytest  # noqa: E402


@pytest.mark.parametrize("bad", [0, -1.0, float("nan"), float("inf"), float("-inf")])
def test_3_invalid_cadence_s_value_raises_valueerror_naming_field(bad):
    """§3: non-positive / non-finite cadence_s raises ValueError naming cadence_s."""
    df = _make_track_df(n_frames=32)
    with pytest.raises(ValueError, match="cadence_s"):
        psi_g.compute(df, cadence_s=bad)


@pytest.mark.parametrize("bad", [True, np.bool_(True), "300", [300.0]])
def test_3_invalid_cadence_s_type_raises_typeerror_naming_field(bad):
    """§3: bool/str/list cadence_s raises TypeError naming cadence_s."""
    df = _make_track_df(n_frames=32)
    with pytest.raises(TypeError, match="cadence_s"):
        psi_g.compute(df, cadence_s=bad)


@pytest.mark.parametrize("bad", [42, "default", []])
def test_3_invalid_constants_type_raises_typeerror_naming_field(bad):
    """§3: a non-ConstantsT constants argument raises TypeError naming constants."""
    df = _make_track_df(n_frames=32)
    with pytest.raises(TypeError, match="constants"):
        psi_g.compute(df, cadence_s=300.0, constants=bad)


@pytest.mark.parametrize("bad_window", [24, 2])
def test_3_invalid_sg_window_override_raises_valueerror_naming_field(bad_window):
    """§3: an even / too-small SG_WINDOW_DETREND override raises ValueError naming it.

    NOTE: ``ConstantsT`` is imported at module level (binding the same class
    object ``psi_g.py`` holds). A function-local re-import would fetch a fresh
    class object after ``test_no_root_handlers_added_at_import`` reloads
    ``sleap_roots.circumnutation.*``, breaking ``isinstance`` in full-suite runs.
    """
    df = _make_track_df(n_frames=32)
    override = ConstantsT(SG_WINDOW_DETREND=bad_window)
    with pytest.raises(ValueError, match="SG_WINDOW_DETREND"):
        psi_g.compute(df, cadence_s=300.0, constants=override)


# ===========================================================================
# §4 — raw, CWT-free traits (handedness, delta_E, helix) + conditioning isolation
# ===========================================================================


@pytest.mark.parametrize("planted", [+1, -1])
def test_4_handedness_equals_planted_generator_handedness(planted):
    """§4: psi_g.compute handedness equals the generator's planted ±1 convention."""
    df = synthetic.generate_trajectory(
        handedness=planted, amplitude_px=10.0, noise_sigma_px=0.0, n_frames=575
    )
    result = psi_g.compute(df, cadence_s=300.0)
    assert int(result["handedness"].iloc[0]) == planted


@pytest.mark.parametrize("planted", [+1, -1])
def test_4_helix_sign_agrees_with_handedness(planted):
    """§4: sign(helix_signed_area_px2) == handedness (independent confirmation)."""
    df = synthetic.generate_trajectory(
        handedness=planted, amplitude_px=10.0, noise_sigma_px=0.0, n_frames=575
    )
    result = psi_g.compute(df, cadence_s=300.0)
    h = int(result["handedness"].iloc[0])
    area = float(result["helix_signed_area_px2"].iloc[0])
    assert int(np.sign(area)) == h == planted


def test_4_delta_E_recovers_constant_step_speed():
    """§4: delta_E = median(√(dx²+dy²)) recovers the constant per-frame speed.

    A pure-growth track (amplitude 0, no noise) advances a fixed step per
    frame along the growth axis, so the median velocity magnitude equals the
    growth rate exactly.
    """
    df = synthetic.generate_trajectory(
        amplitude_px=0.0,
        growth_rate_px_per_frame=2.0,
        noise_sigma_px=0.0,
        n_frames=64,
    )
    result = psi_g.compute(df, cadence_s=300.0)
    assert result["delta_E_amplitude_proxy_px_per_frame"].iloc[0] == pytest.approx(
        2.0, abs=1e-9
    )


def test_4_conditioning_isolation_raw_traits_invariant_to_sg_override():
    """§4.3: handedness/delta_E/helix are identical under an SG_WINDOW_DETREND override.

    Only T_psig_median_s uses the SG-detrended CWT; the 3 raw traits must be
    bit-identical whether the conditioning window is the default 23 or an
    override 31.
    """
    df = synthetic.generate_trajectory(
        handedness=+1, amplitude_px=10.0, noise_sigma_px=0.0, n_frames=575
    )
    r_default = psi_g.compute(df, cadence_s=300.0)
    r_override = psi_g.compute(
        df, cadence_s=300.0, constants=ConstantsT(SG_WINDOW_DETREND=31)
    )
    for col in (
        "handedness",
        "delta_E_amplitude_proxy_px_per_frame",
        "helix_signed_area_px2",
    ):
        assert r_default[col].iloc[0] == r_override[col].iloc[0]


# ===========================================================================
# §5 — T_psig_median_s CWT path (±10% recovery; min-length + zero-energy guards)
# ===========================================================================

import warnings  # noqa: E402


@pytest.mark.parametrize("T_s", [3333.0, 4500.0])
def test_5_T_psig_recovers_planted_period_within_10pct(T_s):
    """§5: T_psig_median_s recovers a planted nutation period within ±10%.

    SG-detrend distorts noise-free recovery to ~5% (nutation test_2C2 uses the
    identical ±10% bar on the same SG-detrend→CWT path); periods are chosen
    from the in-band set {3333, 4500}.
    """
    df = synthetic.generate_trajectory(
        T_nutation_s=T_s,
        n_frames=575,
        cadence_s=300.0,
        amplitude_px=10.0,
        noise_sigma_px=0.0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = psi_g.compute(df, cadence_s=300.0)
    t_psig = float(result["T_psig_median_s"].iloc[0])
    assert abs(t_psig - T_s) / T_s < 0.10


def test_5_T_psig_no_runtimewarning_on_clean_track():
    """§5: a clean nutating track produces T_psig with no numpy RuntimeWarning."""
    df = synthetic.generate_trajectory(
        T_nutation_s=3333.0, n_frames=575, cadence_s=300.0, noise_sigma_px=0.0
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = psi_g.compute(df, cadence_s=300.0)
    assert np.isfinite(result["T_psig_median_s"].iloc[0])


# ===========================================================================
# §6 — degenerate / edge cases (the spec degenerate table)
# ===========================================================================


def _single_row_result(df):
    """Run psi_g.compute under a RuntimeWarning-as-error guard; return row 0."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        return psi_g.compute(df, cadence_s=300.0).iloc[0]


def test_6_degenerate_two_frame_track_all_degenerate_row():
    """§6: N<3 (2 frames) → T_psig/delta_E/helix NaN, handedness 0, no raise."""
    df = _make_track_df(n_frames=2)
    row = _single_row_result(df)
    assert np.isnan(row["T_psig_median_s"])
    assert np.isnan(row["delta_E_amplitude_proxy_px_per_frame"])
    assert np.isnan(row["helix_signed_area_px2"])
    assert int(row["handedness"]) == 0


def test_6_short_track_3_to_23_frames_T_psig_nan_raw_defined():
    """§6: 3≤N<24 → T_psig NaN (CWT skipped) while raw traits are finite/defined."""
    df = _make_track_df(n_frames=15)
    row = _single_row_result(df)
    assert np.isnan(row["T_psig_median_s"])
    assert np.isfinite(row["delta_E_amplitude_proxy_px_per_frame"])
    assert np.isfinite(row["helix_signed_area_px2"])
    assert int(row["handedness"]) in (-1, 0, 1)


def test_6_stationary_track_N_ge_24_zero_energy_guard():
    """§6: stationary N≥24 → T_psig NaN (zero-energy guard), delta_E/helix 0.0, handedness 0.

    No spurious 2·cadence period; no RuntimeWarning.
    """
    df = synthetic.generate_trajectory(
        amplitude_px=0.0,
        growth_rate_px_per_frame=0.0,
        noise_sigma_px=0.0,
        n_frames=64,
    )
    row = _single_row_result(df)
    assert np.isnan(row["T_psig_median_s"])
    assert row["delta_E_amplitude_proxy_px_per_frame"] == 0.0
    assert row["helix_signed_area_px2"] == 0.0
    assert int(row["handedness"]) == 0


def test_6_nan_injected_rows_dropped_before_diff():
    """§6: non-finite tip rows are dropped before ψ_g; the track still computes."""
    df = _make_track_df(n_frames=64).copy()
    df.loc[5, "tip_x"] = float("nan")
    df.loc[20, "tip_y"] = float("inf")
    row = _single_row_result(df)
    # 62 finite frames remain (>= 24) → all four traits defined.
    assert np.isfinite(row["T_psig_median_s"])
    assert np.isfinite(row["delta_E_amplitude_proxy_px_per_frame"])
    assert int(row["handedness"]) in (-1, 0, 1)


def test_6_all_nonfinite_track_is_degenerate():
    """§6: a track whose every tip is non-finite → the all-degenerate row."""
    df = _make_track_df(n_frames=64).copy()
    df["tip_x"] = float("nan")
    row = _single_row_result(df)
    assert np.isnan(row["T_psig_median_s"])
    assert int(row["handedness"]) == 0


# ===========================================================================
# §7 — determinism (CC-6) + the one-DEBUG-record logging contract
# ===========================================================================

import logging  # noqa: E402


def test_7_same_process_determinism_bit_identical():
    """§7: same input → 3 float columns bit-identical (atol=0), handedness equal."""
    df = synthetic.generate_trajectory(
        random_state=0,
        noise_sigma_px=0.5,
        n_frames=575,
        T_nutation_s=3333,
        cadence_s=300,
    )
    a = psi_g.compute(df, cadence_s=300.0)
    b = psi_g.compute(df, cadence_s=300.0)
    for col in (
        "T_psig_median_s",
        "delta_E_amplitude_proxy_px_per_frame",
        "helix_signed_area_px2",
    ):
        np.testing.assert_array_equal(a[col].to_numpy(), b[col].to_numpy())
    np.testing.assert_array_equal(
        a["handedness"].to_numpy(), b["handedness"].to_numpy()
    )


def test_7_cross_os_canary_at_atol_1e_6():
    """§7: 3-value canary matches the captured sentinel at atol=1e-6 (CC-6).

    Captured on Windows 11 + Python 3.11.13; Ubuntu/macOS verified in CI. The
    canary is a regression-detection sentinel and MAY be re-captured (with a
    cross-reference to this test) if upstream BLAS/scipy/pywt/numpy semantics
    legitimately shift after merge. handedness is integer → exact equality.
    """
    df = synthetic.generate_trajectory(
        random_state=0,
        noise_sigma_px=0.5,
        n_frames=575,
        T_nutation_s=3333,
        cadence_s=300,
    )
    row = psi_g.compute(df, cadence_s=300.0).iloc[0]
    expected = np.array([3499.82238829379, 4.731402527735528, 5051.7188736809085])
    got = np.array(
        [
            float(row["T_psig_median_s"]),
            float(row["delta_E_amplitude_proxy_px_per_frame"]),
            float(row["helix_signed_area_px2"]),
        ]
    )
    np.testing.assert_allclose(got, expected, atol=1e-6, rtol=0)
    assert int(row["handedness"]) == 1


def test_7_emits_exactly_one_debug_record(caplog):
    """§7: exactly one DEBUG record, prefix 'psi_g.compute(', tokens, no coordinate=."""
    df = _make_multi_track_df(n_tracks=3, n_frames=64)
    with caplog.at_level(logging.DEBUG, logger="sleap_roots.circumnutation.psi_g"):
        psi_g.compute(df, cadence_s=300.0)
    records = [
        r
        for r in caplog.records
        if r.name == "sleap_roots.circumnutation.psi_g" and r.levelno == logging.DEBUG
    ]
    assert len(records) == 1
    msg = records[0].getMessage()
    assert msg.startswith("psi_g.compute(")
    assert "n_tracks=" in msg
    assert "cadence_s=" in msg
    assert "coordinate=" not in msg
    assert not any(r.levelno >= logging.INFO for r in caplog.records)


# ===========================================================================
# §8 — cross-tier consistency vs Tier 0 principal_axis_angle
# ===========================================================================

import math  # noqa: E402


def _wrap_to_pi(d: float) -> float:
    """Wrap an angle difference to (-π, π] for branch-cut-safe comparison."""
    return (d + math.pi) % (2.0 * math.pi) - math.pi


def _circular_mean(angles: np.ndarray) -> float:
    """Circular mean atan2(mean(sin), mean(cos)) — robust at the ±π branch cut."""
    return float(np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles))))


@pytest.mark.parametrize("theta", [0.3, -2.0])
def test_8_convention_lock_angle_identity_amplitude_zero(theta):
    """§8.1a: circular_mean(ψ_g) ≈ π/2 − θ for a non-oscillating planted axis.

    amplitude_px=0 (pure straight growth → ψ_g constant) is the only regime
    where the angle identity holds at 1e-6 (oscillation biases circular_mean).
    θ=−2.0 places π/2−θ outside (−π, π], exercising the branch-cut wrap.
    handedness is 0 for a non-rotating track (assert, do NOT assert ±1 here).
    """
    df = synthetic.generate_trajectory(
        amplitude_px=0.0, growth_axis_angle_rad=theta, noise_sigma_px=0.0, n_frames=64
    )
    x = df["tip_x"].to_numpy()
    y = df["tip_y"].to_numpy()
    psi = compute_psi_g(x, y)
    cm = _circular_mean(psi)
    assert abs(_wrap_to_pi(cm - (math.pi / 2.0 - theta))) < 1e-6
    result = psi_g.compute(df, cadence_s=300.0)
    assert int(result["handedness"].iloc[0]) == 0


@pytest.mark.parametrize("planted", [+1, -1])
def test_8_convention_lock_handedness_amplitude_positive(planted):
    """§8.1b: an oscillating planted-handedness track locks handedness == planted."""
    df = synthetic.generate_trajectory(
        handedness=planted, amplitude_px=10.0, noise_sigma_px=0.0, n_frames=575
    )
    result = psi_g.compute(df, cadence_s=300.0)
    assert int(result["handedness"].iloc[0]) == planted


# §8.3 — plate-001 GREEN-phase reconciliation -------------------------------

from pathlib import Path  # noqa: E402

_PROOFREAD_FIXTURE_PATH = Path(
    "tests/data/circumnutation_nipponbare_plate_001/"
    "plate_001_greyscale.tracked_proofread.slp"
)

# GREEN-phase reconciliation values captured from a real run (Windows 11 +
# Python 3.11.13). Observed per-track |circmean(ψ_g) − (π/2 − principal_axis_angle)|
# deviations on the 6 Nipponbare proofread tracks: max 0.0311 rad (1.8°), all 6
# within 0.0311 rad — far inside the pre-committed floor (N≥2, tol≤0.35 rad). The
# 0.10 rad tolerance keeps ~3× headroom; ≥5/6 allows one-track robustness to a
# future fixture re-export. NOT a pre-known RED threshold.
_PSIG_AXIS_RECONCILE_TOL_RAD = 0.10
_PSIG_AXIS_RECONCILE_MIN_TRACKS = 5


def _load_proofread_enriched() -> pd.DataFrame:
    """Load the 6-track Nipponbare proofread fixture, enriched with identity columns."""
    from sleap_roots.series import Series

    s = Series.load(series_name="plate_001", primary_path=str(_PROOFREAD_FIXTURE_PATH))
    df = s.get_tracked_tips()
    df["track_id"] = df["track_id"].str.replace("track_", "", regex=False).astype(int)
    df["series"] = "plate_001"
    df["sample_uid"] = "plate_001"
    df["timepoint"] = "T0"
    df["plate_id"] = "plate_001"
    df["plant_id"] = df["track_id"]
    df["genotype"] = np.nan
    df["treatment"] = np.nan
    return df


@pytest.mark.skipif(
    not _PROOFREAD_FIXTURE_PATH.exists(),
    reason=f"proofread fixture not present: {_PROOFREAD_FIXTURE_PATH}",
)
def test_8_cross_tier_plate001_reconciliation_green_phase():
    """§8.3: circmean(ψ_g) ≈ π/2 − principal_axis_angle on plate-001 (GREEN-phase).

    Skips tracks whose Tier 0 principal_axis_angle is NaN (growth_axis_unreliable
    gate). Asserts ≥5 of 6 surviving tracks within _PSIG_AXIS_RECONCILE_TOL_RAD.
    Observed: all 6 within 0.0311 rad (1.8°) — see the constants' docstring.
    """
    from sleap_roots.circumnutation import kinematics

    df = _load_proofread_enriched()
    k = kinematics.compute(df)
    deviations = []
    for tid in sorted(df["track_id"].unique()):
        paa = float(k[k["track_id"] == tid]["principal_axis_angle"].iloc[0])
        if np.isnan(paa):
            continue  # growth_axis_unreliable gate fired → skip
        sub = df[df["track_id"] == tid].sort_values("frame")
        x = sub["tip_x"].to_numpy(float)
        y = sub["tip_y"].to_numpy(float)
        finite = np.isfinite(x) & np.isfinite(y)
        psi = compute_psi_g(x[finite], y[finite])
        dev = abs(_wrap_to_pi(_circular_mean(psi) - (math.pi / 2.0 - paa)))
        deviations.append(dev)
    deviations = np.array(deviations)
    assert deviations.size >= 2, "too few surviving tracks for a meaningful check"
    n_within = int((deviations < _PSIG_AXIS_RECONCILE_TOL_RAD).sum())
    assert n_within >= _PSIG_AXIS_RECONCILE_MIN_TRACKS, (
        f"only {n_within} tracks within {_PSIG_AXIS_RECONCILE_TOL_RAD} rad; "
        f"deviations (rad) = {np.round(deviations, 4).tolist()} "
        f"(max {deviations.max():.4f})"
    )


# ===========================================================================
# §9 — multi-track integration (heterogeneous track lengths)
# ===========================================================================


def test_9_multi_track_mixed_lengths_no_cross_contamination():
    """§9: a df mixing a <24-frame track and ≥24-frame tracks yields correct rows.

    The short track gets T_psig=NaN (CWT skipped) while its raw traits and the
    long tracks' full traits are computed independently — no cross-track bleed.
    The sign-asserted tracks use ``synthetic.generate_trajectory`` (which plants
    a genuine net rotation via its locked handedness convention); a hand-rolled
    symmetric wobble would have near-zero, phase-dependent net rotation.
    """
    long_a = synthetic.generate_trajectory(
        handedness=+1,
        amplitude_px=10.0,
        noise_sigma_px=0.0,
        n_frames=64,
        track_id=0,
        plant_id=0,
    )
    short = synthetic.generate_trajectory(
        amplitude_px=10.0, noise_sigma_px=0.0, n_frames=15, track_id=1, plant_id=1
    )
    long_b = synthetic.generate_trajectory(
        handedness=-1,
        amplitude_px=10.0,
        noise_sigma_px=0.0,
        n_frames=575,
        track_id=2,
        plant_id=2,
    )
    df = pd.concat([long_a, short, long_b], ignore_index=True)

    result = psi_g.compute(df, cadence_s=300.0)
    assert len(result) == 3
    assert list(result.columns) == list(ROW_IDENTITY_COLUMNS) + list(
        _PSIG_TRAIT_COLUMNS
    )

    by_track = result.set_index("track_id")
    # Short track (id 1): T_psig NaN, raw traits finite.
    assert np.isnan(by_track.loc[1, "T_psig_median_s"])
    assert np.isfinite(by_track.loc[1, "delta_E_amplitude_proxy_px_per_frame"])
    # Long tracks (id 0, 2): T_psig finite; handedness matches each track's sweep.
    assert np.isfinite(by_track.loc[0, "T_psig_median_s"])
    assert np.isfinite(by_track.loc[2, "T_psig_median_s"])
    assert int(by_track.loc[0, "handedness"]) == 1
    assert int(by_track.loc[2, "handedness"]) == -1
