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

from sleap_roots.circumnutation import psi_g
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
    """§3: an even / too-small SG_WINDOW_DETREND override raises ValueError naming it."""
    from sleap_roots.circumnutation._constants import ConstantsT

    df = _make_track_df(n_frames=32)
    override = ConstantsT(SG_WINDOW_DETREND=bad_window)
    with pytest.raises(ValueError, match="SG_WINDOW_DETREND"):
        psi_g.compute(df, cadence_s=300.0, constants=override)


# ===========================================================================
# §4 — raw, CWT-free traits (handedness, delta_E, helix) + conditioning isolation
# ===========================================================================

from sleap_roots.circumnutation import synthetic  # noqa: E402
from sleap_roots.circumnutation._constants import ConstantsT  # noqa: E402


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
