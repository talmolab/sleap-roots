"""Tests for Tier 3c traveling-wave trait emission (PR #10).

Mirrors tests/test_circumnutation_nutation.py / test_circumnutation_psi_g.py
structure. Spec: openspec/changes/add-circumnutation-tier3c-traits/specs/
circumnutation/spec.md (Requirement: Tier 3c traveling-wave trait emission API).
"""

import json
import logging
import warnings
from collections import defaultdict
from pathlib import Path

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


# ---------------------------------------------------------------------------
# Task 3 — per-track spatial chain + error handling
# ---------------------------------------------------------------------------

_SPATIAL_TRAITS = (
    "lambda_spatial_median_px",
    "lambda_spatial_variation",
    "lambda_spatial_mad_px",
)


def _wavy_track_rows(track_id, n_frames=300, amp=20.0, freq=0.3, plate_id="plate"):
    """A genuinely curved oscillating trail that forms a non-degenerate ridge."""
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


def test_healthy_track_yields_finite_spatial_traits():
    df = pd.DataFrame(_wavy_track_rows(0))
    result = traveling_wave.compute(df, cadence_s=300.0)
    row = result.iloc[0]
    for col in _SPATIAL_TRAITS:
        assert np.isfinite(row[col]), col
    assert np.isfinite(row["coi_valid_fraction"])
    assert 0.0 <= row["coi_valid_fraction"] <= 1.0


def test_stationary_track_all_nan_no_runtimewarning():
    rows = []
    for frame in range(64):
        rows.append(
            {
                "series": "test",
                "sample_uid": "test",
                "timepoint": "T0",
                "plate_id": "plate",
                "plant_id": 0,
                "track_id": 0,
                "genotype": np.nan,
                "treatment": np.nan,
                "frame": frame,
                "tip_x": 5.0,
                "tip_y": 5.0,
            }
        )
    df = pd.DataFrame(rows)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = traveling_wave.compute(df, cadence_s=300.0)
    row = result.iloc[0]
    for col in _SPATIAL_TRAITS:
        assert np.isnan(row[col]), col
    assert np.isnan(row["coi_valid_fraction"])  # no ridge formed


def test_short_track_does_not_crash_other_tracks_survive():
    healthy = _wavy_track_rows(0)
    short = [
        {
            "series": "test",
            "sample_uid": "test",
            "timepoint": "T0",
            "plate_id": "plate",
            "plant_id": 1,
            "track_id": 1,
            "genotype": np.nan,
            "treatment": np.nan,
            "frame": f,
            "tip_x": float(f),
            "tip_y": 0.0,
        }
        for f in range(2)
    ]
    df = pd.DataFrame(healthy + short)
    result = traveling_wave.compute(df, cadence_s=300.0).set_index("track_id")
    assert np.isfinite(result.loc[0, "lambda_spatial_median_px"])
    assert np.isnan(result.loc[1, "lambda_spatial_median_px"])
    assert len(result) == 2


def test_all_nan_tip_and_single_frame_tracks_emit_nan_rows():
    nan_track = [
        {
            "series": "test",
            "sample_uid": "test",
            "timepoint": "T0",
            "plate_id": "plate",
            "plant_id": 0,
            "track_id": 0,
            "genotype": np.nan,
            "treatment": np.nan,
            "frame": f,
            "tip_x": np.nan,
            "tip_y": np.nan,
        }
        for f in range(10)
    ]
    single = [
        {
            "series": "test",
            "sample_uid": "test",
            "timepoint": "T0",
            "plate_id": "plate",
            "plant_id": 1,
            "track_id": 1,
            "genotype": np.nan,
            "treatment": np.nan,
            "frame": 0,
            "tip_x": 1.0,
            "tip_y": 2.0,
        }
    ]
    df = pd.DataFrame(nan_track + single)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = traveling_wave.compute(df, cadence_s=300.0)
    assert len(result) == 2
    for col in _SPATIAL_TRAITS + ("coi_valid_fraction",):
        assert result[col].isna().all(), col


# ---------------------------------------------------------------------------
# Task 6 — calibration-table extension (append-only) + in-package literal
# ---------------------------------------------------------------------------

_CALIB_JSON = (
    Path(__file__).parent / "data" / "circumnutation_spatial_cwt_calibration.json"
)

# The original PR #9 18 rows (n, lambda_true) -> ratio. These MUST survive the
# PR #10 append-only extension byte-for-byte (compared by key, not list position).
_ORIGINAL_RATIOS = {
    (200, 20.0): 1.0707850052975358,
    (200, 30.0): 1.0743040511798365,
    (200, 40.0): 1.0947768883704525,
    (200, 50.0): 1.1307412279149716,
    (200, 60.0): 1.155952735350389,
    (200, 80.0): 1.1193063307578321,
    (400, 20.0): 1.0792095307320064,
    (400, 30.0): 1.044290146155988,
    (400, 40.0): 1.068369134605712,
    (400, 50.0): 1.0956767119921316,
    (400, 60.0): 1.1000299725343554,
    (400, 80.0): 1.1253941963571172,
    (600, 20.0): 1.1128336235099439,
    (600, 30.0): 1.0808860017610875,
    (600, 40.0): 1.1419722268051635,
    (600, 50.0): 1.0477828440521288,
    (600, 60.0): 1.0724539886061153,
    (600, 80.0): 1.1330636787960808,
}


def _load_calib_rows():
    return json.loads(_CALIB_JSON.read_text(encoding="utf-8"))["wavelength_calibration"]


def test_calibration_original_rows_preserved_byte_for_byte():
    """The PR #9 18 rows survive the append-only extension unchanged (by key)."""
    rows = {(r["n"], r["lambda_true"]): r["ratio"] for r in _load_calib_rows()}
    for key, ratio in _ORIGINAL_RATIOS.items():
        assert key in rows, key
        assert rows[key] == ratio, key  # exact equality (atol=0)


def test_calibration_extension_covers_real_lambda_for_all_n():
    """The extension adds lambda_true>=140 for all three n (-> lambda_reported>=140)."""
    rows = _load_calib_rows()
    for n in (200, 400, 600):
        lts = {r["lambda_true"] for r in rows if r["n"] == n}
        assert {100.0, 120.0, 140.0, 150.0} <= lts, n
    # n-averaged lambda_reported axis reaches past the observed real lambda ~142.5
    assert max(r["lambda_reported"] for r in rows) >= 140.0


def test_in_package_calibration_literal_matches_n_averaged_json():
    """_CGAU2_LAMBDA_CALIBRATION equals the n-averaged JSON, strictly increasing."""
    rows = _load_calib_rows()
    by_lt = defaultdict(list)
    for r in rows:
        by_lt[r["lambda_true"]].append(r)
    expected = []
    for lt in sorted(by_lt):
        rs = by_lt[lt]
        ratio_mean = sum(r["ratio"] for r in rs) / len(rs)
        lam_rep_mean = sum(r["lambda_reported"] for r in rs) / len(rs)
        expected.append((lam_rep_mean, ratio_mean))
    expected.sort()

    literal = traveling_wave._CGAU2_LAMBDA_CALIBRATION
    assert len(literal) == len(expected)
    for (lr_lit, ra_lit), (lr_exp, ra_exp) in zip(literal, expected):
        assert lr_lit == lr_exp  # atol=0: literal generated from JSON tokens
        assert ra_lit == ra_exp
    # strictly increasing reported axis (well-posed np.interp) covering lambda_true>=140
    axis = [p[0] for p in literal]
    assert all(axis[i + 1] > axis[i] for i in range(len(axis) - 1))
    assert axis[-1] >= 140.0


# ---------------------------------------------------------------------------
# Task 4 — cgau2 calibration consumer + COI gate
# ---------------------------------------------------------------------------


def _make_ridge(in_coi, wavelengths):
    from sleap_roots.circumnutation.spatial_cwt import SpatialRidgeResult

    in_coi = np.asarray(in_coi, dtype=bool)
    wavelengths = np.asarray(wavelengths, dtype=np.float64)
    n = in_coi.size
    return SpatialRidgeResult(
        position_indices=np.arange(n, dtype=np.int64),
        wavelengths_px=wavelengths,
        amplitudes=np.ones(n, dtype=np.float64),
        powers=np.ones(n, dtype=np.float64),
        in_coi=in_coi,
    )


def test_calibrate_recovers_known_knot():
    """Dividing a knot's lambda_reported_mean by its ratio recovers lambda_true."""
    # The lambda_true=80 knot: lambda_reported_mean=90.07371..., ratio=1.12592...
    cal = traveling_wave._calibrate_wavelengths(np.array([90.07371215762747]))
    assert np.allclose(cal, 80.0, atol=1e-6)


def test_coi_gate_exactly_half_does_not_gate():
    """in_coi fraction exactly 0.5 -> coi_valid_fraction 0.5, NOT gated (strict <)."""
    ridge = _make_ridge([True] * 5 + [False] * 5, [90.07371215762747] * 10)
    traits = traveling_wave._ridge_to_traits(ridge, ConstantsT())
    assert traits["coi_valid_fraction"] == 0.5
    assert np.isfinite(traits["lambda_spatial_median_px"])
    assert np.isclose(traits["lambda_spatial_median_px"], 80.0, atol=1e-6)


def test_coi_gate_above_half_in_coi_gates_lambda_but_keeps_fraction():
    """in_coi fraction 0.6 (>COI_FRACTION_MAX) -> lambda NaN, coi_valid_fraction finite."""
    ridge = _make_ridge([True] * 6 + [False] * 4, [90.0] * 10)
    traits = traveling_wave._ridge_to_traits(ridge, ConstantsT())
    assert traits["coi_valid_fraction"] == pytest.approx(0.4)
    assert np.isnan(traits["lambda_spatial_median_px"])
    assert np.isnan(traits["lambda_spatial_variation"])
    assert np.isnan(traits["lambda_spatial_mad_px"])


# ---------------------------------------------------------------------------
# Task 5 — Tier 0/1 composition (5-tuple merge) + lambda_expected/residual
# ---------------------------------------------------------------------------


def _nutating_track_rows(track_id, n_frames=575, plate_id="plate", T_s=3333.0):
    """A nutating lateral oscillation + steady growth drift (synthetic)."""
    from sleap_roots.circumnutation import synthetic

    df = synthetic.generate_trajectory(
        T_nutation_s=T_s,
        n_frames=n_frames,
        cadence_s=300.0,
        amplitude_px=12.0,
        growth_rate_px_per_frame=4.0,
        noise_sigma_px=0.0,
        random_state=0,
    )
    df = df.copy()
    df["series"] = "test"
    df["sample_uid"] = "test"
    df["timepoint"] = "T0"
    df["plate_id"] = plate_id
    df["plant_id"] = track_id
    df["track_id"] = track_id
    df["genotype"] = np.nan
    df["treatment"] = np.nan
    return df


def test_full_emission_on_nutating_synthetic_track():
    """The milestone: a nutating synthetic track emits all 6 finite traits."""
    df = _nutating_track_rows(0)
    result = traveling_wave.compute(df, cadence_s=300.0)
    row = result.iloc[0]
    for col in _TRAIT_COLUMNS:
        assert np.isfinite(row[col]), col
    assert row["traveling_wave_residual"] >= 0.0


def test_multi_plate_float_track_id_uses_correct_plate_operands():
    """Overlapping float64 track_ids across plates: finite operands, no silent NaN."""
    a = _nutating_track_rows(0, plate_id="plateA")
    b = _nutating_track_rows(0, plate_id="plateB", T_s=4000.0)
    df = pd.concat([a, b], ignore_index=True)
    df["track_id"] = df["track_id"].astype(np.float64)  # exercise the int64 coercion
    result = traveling_wave.compute(df, cadence_s=300.0)
    assert len(result) == 2
    # both plates' healthy tracks have FINITE operands (not silent all-NaN)
    assert result["lambda_expected_px"].notna().all()
    # the two plates have different T -> different lambda_expected (correct-plate join)
    exp = result.sort_values("plate_id")["lambda_expected_px"].to_numpy()
    assert not np.isclose(exp[0], exp[1])


def test_non_nutating_track_nans_residual_keeps_spatial_lambda():
    """A noise-only (non-nutating) track: residual + expected NaN; spatial lambda valid."""
    from sleap_roots.circumnutation import synthetic

    df = synthetic.generate_trajectory(
        T_nutation_s=3333.0,
        n_frames=600,
        cadence_s=300.0,
        amplitude_px=0.0,
        growth_rate_px_per_frame=4.0,
        noise_sigma_px=2.0,
        random_state=0,
    ).copy()
    for col, val in [
        ("series", "test"),
        ("sample_uid", "test"),
        ("timepoint", "T0"),
        ("plate_id", "plate"),
        ("plant_id", 0),
        ("track_id", 0),
        ("genotype", np.nan),
        ("treatment", np.nan),
    ]:
        df[col] = val
    result = traveling_wave.compute(df, cadence_s=300.0)
    row = result.iloc[0]
    assert np.isnan(row["traveling_wave_residual"])
    assert np.isnan(row["lambda_expected_px"])
    # pure-spatial trait remains valid (a ridge formed)
    assert np.isfinite(row["lambda_spatial_median_px"])


def test_stationary_track_residual_no_runtimewarning():
    """A stationary track (v~0): lambda_expected/residual NaN, no divide warning."""
    rows = [
        {
            "series": "test",
            "sample_uid": "test",
            "timepoint": "T0",
            "plate_id": "plate",
            "plant_id": 0,
            "track_id": 0,
            "genotype": np.nan,
            "treatment": np.nan,
            "frame": f,
            "tip_x": 5.0,
            "tip_y": 5.0,
        }
        for f in range(64)
    ]
    df = pd.DataFrame(rows)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = traveling_wave.compute(df, cadence_s=300.0)
    row = result.iloc[0]
    assert np.isnan(row["lambda_expected_px"])
    assert np.isnan(row["traveling_wave_residual"])


# ---------------------------------------------------------------------------
# Task 7 — real plate-001 validation + synthetic recovery / noise floor
# ---------------------------------------------------------------------------

_PROOFREAD_FIXTURE = (
    Path(__file__).parent
    / "data"
    / "circumnutation_nipponbare_plate_001"
    / "plate_001_greyscale.tracked_proofread.slp"
)


def _load_plate001_trajectory_df():
    from sleap_roots.series import Series

    series = Series.load(series_name="plate_001", primary_path=str(_PROOFREAD_FIXTURE))
    df = series.get_tracked_tips()
    df["track_id"] = df["track_id"].str.replace("track_", "", regex=False).astype(int)
    df = df.copy()
    df["series"] = "plate_001"
    df["sample_uid"] = "plate_001"
    df["timepoint"] = "T0"
    df["plate_id"] = "plate_001"
    df["plant_id"] = df["track_id"]
    df["genotype"] = "Nipponbare"
    df["treatment"] = "none"
    return df


@pytest.mark.skipif(
    not _PROOFREAD_FIXTURE.exists(),
    reason=f"Git-LFS proofread fixture not present: {_PROOFREAD_FIXTURE}",
)
def test_real_plate001_qpb_result_calibrated():
    """§7: real compute() on the 6 Nipponbare tracks — QPB residual finite & < 0.30."""
    df = _load_plate001_trajectory_df()
    result = traveling_wave.compute(df, cadence_s=300.0)
    assert len(result) == 6

    res = result["traveling_wave_residual"].to_numpy(dtype=np.float64)
    var = result["lambda_spatial_variation"].to_numpy(dtype=np.float64)
    coi = result["coi_valid_fraction"].to_numpy(dtype=np.float64)
    lam = result["lambda_spatial_median_px"].to_numpy(dtype=np.float64)

    print(
        "\nplate-001 traveling_wave: "
        f"lambda_median={np.round(lam, 1).tolist()}; "
        f"residual={np.round(res, 3).tolist()}; "
        f"variation={np.round(var, 3).tolist()}; "
        f"coi_valid={np.round(coi, 3).tolist()}"
    )

    # QPB holds: all 6 residuals finite and within a generous band (the precise
    # endpoints depend on the n-averaged calibration; do NOT pin a tight range).
    assert np.all(np.isfinite(res))
    assert np.all(res < 0.30), res.tolist()
    # all spatial gates pass (in-COI fraction well below COI_FRACTION_MAX=0.5)
    assert np.all(coi >= 0.5), coi.tolist()
    assert np.all(np.isfinite(var))


def test_synthetic_uniform_lambda_recovery_and_noise_floor():
    """Noise-free uniform-λ synthetic: λ recovered a priori AND variation ≈ 0."""
    from sleap_roots.circumnutation import synthetic

    growth = 4.29
    T_s = 3333.0
    cadence = 300.0
    lam_apriori = growth * (T_s / cadence)  # ~47.7 px

    df = synthetic.generate_trajectory(
        T_nutation_s=T_s,
        n_frames=575,
        cadence_s=cadence,
        amplitude_px=2.0,
        growth_rate_px_per_frame=growth,
        noise_sigma_px=0.0,
        random_state=0,
    ).copy()
    for col, val in [
        ("series", "test"),
        ("sample_uid", "test"),
        ("timepoint", "T0"),
        ("plate_id", "plate"),
        ("plant_id", 0),
        ("track_id", 0),
        ("genotype", np.nan),
        ("treatment", np.nan),
    ]:
        df[col] = val
    result = traveling_wave.compute(df, cadence_s=cadence)
    row = result.iloc[0]
    lam = row["lambda_spatial_median_px"]
    print(
        f"\nsynthetic recovery: lambda_apriori={lam_apriori:.1f}, "
        f"lambda_spatial_median_px={lam:.1f}, variation={row['lambda_spatial_variation']:.4f}"
    )
    assert abs(lam - lam_apriori) / lam_apriori < 0.25
    # NO spurious argmax-quantization floor: a uniform-λ trail reads ~0.
    assert row["lambda_spatial_variation"] < 0.05


def test_compute_is_deterministic_across_runs():
    """Two in-process runs are bit-identical on all 6 float trait columns."""
    df = _nutating_track_rows(0)
    r1 = traveling_wave.compute(df, cadence_s=300.0)
    r2 = traveling_wave.compute(df, cadence_s=300.0)
    for col in _TRAIT_COLUMNS:
        np.testing.assert_array_equal(
            r1[col].to_numpy(), r2[col].to_numpy(), err_msg=col
        )


# Cross-OS regression sentinel on a fixed synthetic (amplitude_px=12,
# growth_rate=4, T=3333, noise=0, random_state=0). The spatial-λ columns are
# exact in-process; the v·T-derived columns inherit the Tier-1 scipy tolerance,
# so the full 6-tuple is asserted at atol=1e-6 cross-OS. MAY be re-captured (in a
# follow-up commit) if BLAS/scipy/pywt/numpy semantics legitimately shift.
_CANARY = {
    "lambda_spatial_median_px": 52.89599405686159,
    "lambda_spatial_variation": 0.0,
    "traveling_wave_residual": 0.02539993025468994,
    "lambda_expected_px": 54.27456420220119,
    "lambda_spatial_mad_px": 0.0,
    "coi_valid_fraction": 0.9755671902268761,
}


def test_compute_canary_matches_expected_values():
    df = _nutating_track_rows(0)
    row = traveling_wave.compute(df, cadence_s=300.0).iloc[0]
    for col, expected in _CANARY.items():
        np.testing.assert_allclose(row[col], expected, atol=1e-6, rtol=0.0, err_msg=col)
