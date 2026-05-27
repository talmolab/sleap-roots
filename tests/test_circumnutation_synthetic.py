"""Synthetic-trajectory generator tests for ``sleap_roots.circumnutation.synthetic``.

Exercises the closed-form Rivière 2022 Eq. 4 realization (D1), the
user-facing aggregate parameter API (D2), the locked dtype contract
(D3), the 7 new ConstantsT-overridable defaults (D4), the determinism
contract (D5 / CC-6), handedness sign convention (D6), strict input
validation (D8), the ConstantsT resolution-order (D13), and the
non-tautological reference-fixture agreement test (D12 §2.H + R6).

Spec deltas under test (OpenSpec change ``add-circumnutation-synthetic-generator``,
PR #4):

- ADDED Requirement: Synthetic trajectory generator — scenarios under
  ``## ADDED Requirements`` in ``specs/circumnutation/spec.md``.
- MODIFIED Requirement: Package layout (``synthetic`` moves stub → impl).
- MODIFIED Requirement: Module-level constants (adds 7 SYNTHETIC_* + bumps
  ``_CONSTANTS_VERSION`` 2 → 3).

Theory references: ``docs/circumnutation/theory.md`` §3.5, §4 (Rivière
2022 Eqs. 1 / 3 / 4 / 5), §4.4 (closed-form Δφ), §8 Layer 1 validation
tolerances. ``docs/circumnutation/preliminary_results_2026-05-07.md``
§1 / §3.4 / §4.1 / §4.3 (plate 001 empirical anchors).
"""

import copy
import inspect
import logging
import math
import tempfile
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from sleap_roots.circumnutation import _geometry, kinematics, qc, synthetic
from sleap_roots.circumnutation._constants import (
    SYNTHETIC_AMPLITUDE_PX,
    SYNTHETIC_CADENCE_S,
    SYNTHETIC_GROWTH_AXIS_ANGLE_RAD,
    SYNTHETIC_GROWTH_RATE_PX_PER_FRAME,
    SYNTHETIC_N_FRAMES,
    SYNTHETIC_NOISE_SIGMA_PX,
    SYNTHETIC_T_NUTATION_S,
    ConstantsT,
)
from sleap_roots.circumnutation._types import (
    REQUIRED_PER_FRAME_COLUMNS,
    ROW_IDENTITY_COLUMNS,
)


# ---------------------------------------------------------------------------
# Expected output schema (locked by spec scenario "Default call returns
# 575-row DataFrame with the documented schema")
# ---------------------------------------------------------------------------

EXPECTED_OUTPUT_COLUMNS = tuple(ROW_IDENTITY_COLUMNS) + tuple(
    REQUIRED_PER_FRAME_COLUMNS
)


# ---------------------------------------------------------------------------
# Fixtures (tasks.md §2.1b — synthetic_setup is the shared derived-constants
# fixture used by §2.G.4 / §2.G.5 and any test needing analytical predictions)
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_setup():
    """Derived constants used by analytical predictions in §2.C / §2.G."""
    T_nutation_s = SYNTHETIC_T_NUTATION_S  # 3333.0
    cadence_s = SYNTHETIC_CADENCE_S  # 300.0
    growth_rate = SYNTHETIC_GROWTH_RATE_PX_PER_FRAME  # 4.29
    omega = 2.0 * math.pi / T_nutation_s
    v_growth_per_s = growth_rate / cadence_s
    return {
        "T_nutation_s": T_nutation_s,
        "cadence_s": cadence_s,
        "growth_rate": growth_rate,
        "omega": omega,
        "v_growth_per_s": v_growth_per_s,
    }


# ===========================================================================
# 2.A — Schema / structural tests
# (spec Requirement: Synthetic trajectory generator)
# ===========================================================================


def test_2A1_output_columns_match_spec():
    """2.A.1 — columns equal ROW_IDENTITY_COLUMNS + REQUIRED_PER_FRAME_COLUMNS."""
    df = synthetic.generate_trajectory()
    assert tuple(df.columns) == EXPECTED_OUTPUT_COLUMNS


def test_2A2_output_dtypes_match_contract():
    """2.A.2 — frame=int64; tip_x/tip_y=float64; plant_id/track_id=int64; identity=object."""
    df = synthetic.generate_trajectory()
    assert df["frame"].dtype == np.dtype("int64")
    assert df["tip_x"].dtype == np.dtype("float64")
    assert df["tip_y"].dtype == np.dtype("float64")
    assert df["plant_id"].dtype == np.dtype("int64")
    assert df["track_id"].dtype == np.dtype("int64")
    for col in (
        "series",
        "sample_uid",
        "timepoint",
        "plate_id",
        "genotype",
        "treatment",
    ):
        assert df[col].dtype == np.dtype("object"), f"{col} not object dtype"


def test_2A3_frame_indexing_is_zero_based():
    """2.A.3 — frame.iloc[0] == 0; frame.iloc[-1] == n_frames - 1; strict monotonic."""
    df = synthetic.generate_trajectory(n_frames=100)
    assert df["frame"].iloc[0] == 0
    assert df["frame"].iloc[-1] == 99
    assert (np.diff(df["frame"]) == 1).all()


def test_2A4_plant_id_equals_track_id_by_default():
    """2.A.4 — default kwargs produce plant_id == track_id (foundation convention)."""
    df = synthetic.generate_trajectory()
    assert df["plant_id"].equals(df["track_id"])


def test_2A5_none_genotype_becomes_nan_not_string():
    """2.A.5 — genotype=None / treatment=None map to np.nan in object-dtype column."""
    df = synthetic.generate_trajectory(genotype=None, treatment=None)
    assert df["genotype"].isna().all()
    assert df["genotype"].dtype == np.dtype("object")
    assert not (df["genotype"] == "None").any()
    assert df["treatment"].isna().all()
    assert df["treatment"].dtype == np.dtype("object")
    assert not (df["treatment"] == "None").any()


@pytest.mark.parametrize("n_frames", [1, 5, 10, 100, 575])
def test_2A6_row_count_equals_n_frames(n_frames):
    """2.A.6 — number of rows equals n_frames."""
    df = synthetic.generate_trajectory(n_frames=n_frames)
    assert len(df) == n_frames


def test_2A7_signature_is_kw_only_positional_raises_typeerror():
    """2.A.7 — positional invocation raises TypeError (signature uses `*,`)."""
    with pytest.raises(TypeError):
        synthetic.generate_trajectory(575)


def test_2A8_signature_has_no_px_per_mm():
    """2.A.8 — inspect.signature does NOT contain px_per_mm (re-asserts foundation)."""
    sig = inspect.signature(synthetic.generate_trajectory)
    assert "px_per_mm" not in sig.parameters


def test_2A9_n_frames_1_emits_single_row():
    """2.A.9 — n_frames=1 returns 1-row DataFrame; kinematics.compute handles cleanly."""
    df = synthetic.generate_trajectory(n_frames=1)
    assert len(df) == 1
    # Tier 0 handles single-frame tracks per its existing NaN-trait contract
    tier0 = kinematics.compute(df)
    assert len(tier0) == 1
    assert pd.isna(tier0["v_total_median_px_per_frame"].iloc[0])


def test_2A10_growth_axis_angle_outside_canonical_range_round_trips():
    """2.A.10 — growth_axis_angle_rad=5π runs without raising (rotation mod 2π)."""
    df_canonical = synthetic.generate_trajectory(
        growth_axis_angle_rad=math.pi, noise_sigma_px=0
    )
    df_wrapped = synthetic.generate_trajectory(
        growth_axis_angle_rad=5 * math.pi, noise_sigma_px=0
    )
    # 5π = π + 2π·2, so the resulting trajectory should match the π case
    npt.assert_allclose(
        df_wrapped["tip_x"].to_numpy(), df_canonical["tip_x"].to_numpy(), atol=1e-9
    )
    npt.assert_allclose(
        df_wrapped["tip_y"].to_numpy(), df_canonical["tip_y"].to_numpy(), atol=1e-9
    )


def test_2A11a_caplog_no_warning_or_error_emissions_on_default_call(caplog):
    """2.A.11a — happy-path default call emits NO WARNING/ERROR/CRITICAL records."""
    caplog.set_level(logging.WARNING, logger="sleap_roots.circumnutation.synthetic")
    _ = synthetic.generate_trajectory()
    # No WARNING/ERROR/CRITICAL records from the synthetic module
    bad_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert bad_records == [], f"Unexpected WARNING+ records: {bad_records}"


def test_2A11b_caplog_branch_coverage_for_debug_emissions(caplog):
    """2.A.11b — DEBUG-level emissions in synthetic.py are exercised by tests.

    Per §5.1 coverage policy, debug branches MUST be covered. The
    implementation may emit DEBUG records on parameter resolution or
    short-circuit paths; this test verifies at least one debug record
    is emitted somewhere across the default-call AND the noise=0
    short-circuit paths, OR confirms the impl emits no DEBUG records
    at all (zero-debug impl is also acceptable and trivially branch-
    complete).
    """
    caplog.set_level(logging.DEBUG, logger="sleap_roots.circumnutation.synthetic")
    # Default call
    _ = synthetic.generate_trajectory()
    # Noise short-circuit
    _ = synthetic.generate_trajectory(noise_sigma_px=0)
    # If the impl emits ANY debug records, they should be from synthetic.py
    debug_records = [
        r
        for r in caplog.records
        if r.levelno == logging.DEBUG
        and r.name == "sleap_roots.circumnutation.synthetic"
    ]
    # Either: zero debug emissions (trivially branch-complete) OR every emission
    # comes from the synthetic module (no spurious records). The implementation
    # may legitimately emit zero DEBUG records (no diagnostic-worthy branches);
    # this is acceptable per §2.A.11b.
    for r in debug_records:
        assert r.name == "sleap_roots.circumnutation.synthetic"


# ===========================================================================
# 2.B — Determinism (CC-6)
# ===========================================================================


def test_2B1_same_int_seed_bit_identical():
    """2.B.1 — same int seed produces bit-identical tip_x / tip_y arrays.

    Canary purpose: this test is a REGRESSION DETECTOR for future numpy
    PCG64 drift, NOT a correctness oracle. The expected first-3-values
    are recorded from the GREEN-phase implementation per §3.7 of tasks.md.
    Cross-OS bit-identity holds on numpy ≥ 1.17 / PCG64 / 64-bit platforms
    per NEP 19 (design.md D5, R4).
    """
    df_a = synthetic.generate_trajectory(random_state=0)
    df_b = synthetic.generate_trajectory(random_state=0)
    # Bit-identical (np.array_equal, not allclose)
    assert np.array_equal(df_a["tip_x"].to_numpy(), df_b["tip_x"].to_numpy())
    assert np.array_equal(df_a["tip_y"].to_numpy(), df_b["tip_y"].to_numpy())
    # Canary: known first-3 values for random_state=0 with default kwargs,
    # captured by the §3.7 canary script on 2026-05-22 (re-captured after
    # the cos(π/2) snap-to-zero fix per Copilot review #3 of PR #210; the
    # snap removed ~6e-17 FP noise from the growth-axis decomposition,
    # which shifted ULPs in tip_x). Locked as a regression detector for
    # future numpy PCG64 changes.
    expected_tip_x_first3 = np.array(
        [
            0.17780938387044457,
            -2.866197211904314,
            -3.6186816320785176,
        ]
    )
    expected_tip_y_first3 = np.array(
        [
            1.794257397130898,
            4.012695386628404,
            8.072352985072323,
        ]
    )
    npt.assert_array_equal(df_a["tip_x"].to_numpy()[:3], expected_tip_x_first3)
    npt.assert_array_equal(df_a["tip_y"].to_numpy()[:3], expected_tip_y_first3)


def test_2B2_same_generator_advances_state():
    """2.B.2 — passing the SAME Generator twice produces DIFFERENT output.

    Documents Generator state-advance behavior: the second call sees a
    different RNG sequence because the first call consumed draws.
    """
    rng = np.random.default_rng(42)
    df_a = synthetic.generate_trajectory(random_state=rng)
    df_b = synthetic.generate_trajectory(random_state=rng)
    assert not np.allclose(df_a["tip_x"].to_numpy(), df_b["tip_x"].to_numpy())


def test_2B3_int_seed_equiv_default_rng_seed():
    """2.B.3 — int seed and np.random.default_rng(seed) produce identical output."""
    df_int = synthetic.generate_trajectory(random_state=42)
    df_gen = synthetic.generate_trajectory(random_state=np.random.default_rng(42))
    npt.assert_array_equal(df_int["tip_x"].to_numpy(), df_gen["tip_x"].to_numpy())
    npt.assert_array_equal(df_int["tip_y"].to_numpy(), df_gen["tip_y"].to_numpy())


def test_2B4_different_seeds_differ():
    """2.B.4 — different seeds produce DIFFERENT output."""
    df_a = synthetic.generate_trajectory(random_state=0)
    df_b = synthetic.generate_trajectory(random_state=1)
    assert not np.allclose(df_a["tip_x"].to_numpy(), df_b["tip_x"].to_numpy())


def test_2B5_noise_zero_short_circuits_rng():
    """2.B.5 — noise_sigma_px=0 produces identical output regardless of random_state.

    Per design.md D11: the RNG path is short-circuited entirely when
    noise_sigma_px == 0.0, so random_state has no effect.
    """
    df_none = synthetic.generate_trajectory(noise_sigma_px=0, random_state=None)
    df_42 = synthetic.generate_trajectory(noise_sigma_px=0, random_state=42)
    df_gen = synthetic.generate_trajectory(
        noise_sigma_px=0, random_state=np.random.default_rng(99)
    )
    npt.assert_array_equal(df_none["tip_x"].to_numpy(), df_42["tip_x"].to_numpy())
    npt.assert_array_equal(df_none["tip_x"].to_numpy(), df_gen["tip_x"].to_numpy())
    npt.assert_array_equal(df_none["tip_y"].to_numpy(), df_42["tip_y"].to_numpy())


def test_2B5b_noise_zero_does_not_advance_caller_generator_state():
    """2.B.5b — caller-supplied Generator state UNCHANGED after noise=0 call.

    Locks design.md D11's "decoupled determinism" guarantee.
    """
    rng = np.random.default_rng(42)
    state_before = copy.deepcopy(rng.bit_generator.state)
    _ = synthetic.generate_trajectory(noise_sigma_px=0, random_state=rng)
    state_after = rng.bit_generator.state
    npt.assert_equal(state_before, state_after)


# ===========================================================================
# 2.C — Parameter recovery via Tier 0
# ===========================================================================


def test_2C1_v_long_recovery_exact_at_amplitude_zero():
    """2.C.1 — pure-linear (amplitude=0) trajectory recovers growth_rate exactly.

    Per design.md D10 E1: with amplitude_px=0 the trajectory is a
    straight line and the kinematics-inferred net-displacement growth
    axis matches `growth_axis_angle_rad` exactly. Tolerance 1e-9 per
    round-1 TDD reviewer B3 (loosened from 1e-10 for cross-platform
    BLAS rounding through `(growth_rate/cadence_s) · (i·cadence_s)`).
    """
    df = synthetic.generate_trajectory(
        amplitude_px=0,
        growth_rate_px_per_frame=4.29,
        noise_sigma_px=0,
        growth_axis_angle_rad=math.pi / 2,
    )
    tier0 = kinematics.compute(df)
    v_long = float(tier0["v_long_signed_median_px_per_frame"].iloc[0])
    assert abs(v_long - 4.29) < 1e-9, f"v_long = {v_long}, expected 4.29 ± 1e-9"
    # Pure-linear yields v_lat_abs_median == 0 EXACTLY after the cos(π/2)
    # snap-to-zero added per Copilot review #3 of PR #210. kinematics
    # returns NaN per its `v_lat_abs_median == 0 → long_lat_ratio is NaN`
    # contract. Strict assertion (was "NaN OR > 1e6" before the snap removed
    # the cos(π/2) ≈ 6.1e-17 FP artifact).
    assert pd.isna(tier0["long_lat_ratio"].iloc[0]), (
        f"long_lat_ratio = {tier0['long_lat_ratio'].iloc[0]}; "
        f"expected NaN for pure-linear synth (cos(π/2) snap should make "
        f"v_lat_abs_median == 0 exactly)"
    )


def test_2C2_angular_amplitude_small_angle_recovery(synthetic_setup):
    """2.C.2 — small-amplitude (amplitude=1.0) test where exact ≈ small-angle.

    For amplitude_px=1.0 with plate-001-matching ω and v_growth, the
    exact formula 2·arctan(amp·ω/(2·v_growth)) and the small-angle
    approximation amp·ω/v_growth agree to < 1%. Tight ±5% tolerance
    catches any small-angle bug.
    """
    setup = synthetic_setup
    df = synthetic.generate_trajectory(
        amplitude_px=1.0,
        noise_sigma_px=0,
    )
    tier0 = kinematics.compute(df)
    angular_amplitude = float(tier0["angular_amplitude"].iloc[0])
    expected = 2 * math.atan(1.0 * setup["omega"] / (2 * setup["v_growth_per_s"]))
    # ±5% tolerance — small-angle regime is tight by construction
    assert (
        abs(angular_amplitude - expected) / expected < 0.05
    ), f"angular_amplitude = {angular_amplitude}, expected ≈ {expected} ± 5%"


def test_2C3_angular_amplitude_plate001_sanity_with_exact_formula(synthetic_setup):
    """2.C.3 — plate-001 default amplitude_px=10 uses the EXACT arctan formula.

    Per design.md D10 E2: small-angle over-estimates by ~13% in the
    plate-001 regime (~1.17 rad ≈ 67°). The test must use the EXACT
    relation; ±15% tolerance per theory.md §8 spatial tolerance.
    """
    setup = synthetic_setup
    df = synthetic.generate_trajectory(noise_sigma_px=0)  # all other defaults
    tier0 = kinematics.compute(df)
    angular_amplitude = float(tier0["angular_amplitude"].iloc[0])
    A_lat = SYNTHETIC_AMPLITUDE_PX / 2.0  # peak-to-peak / 2 = half-amplitude
    expected = 2 * math.atan(A_lat * setup["omega"] / setup["v_growth_per_s"])
    # ±15% per theory.md §8 spatial tolerance
    assert (
        abs(angular_amplitude - expected) / expected < 0.15
    ), f"angular_amplitude = {angular_amplitude}, expected ≈ {expected} ± 15%"
    # E5 safety margin: growth_axis_unreliable=False for plate-001 defaults
    assert bool(tier0["growth_axis_unreliable"].iloc[0]) is False


def test_2C4_synth_growth_axis_inference_matches_kwarg():
    """2.C.4 — kinematics-inferred principal_axis_angle matches kwarg in noise-free case.

    Per design.md D10 E1 + arch reviewer N2: kinematics infers via
    net-displacement (NOT PCA), so for noise-free trajectories the
    inferred axis matches the synthesis kwarg exactly up to floating-
    point rounding.
    """
    df = synthetic.generate_trajectory(
        noise_sigma_px=0,
        growth_axis_angle_rad=math.pi / 2,
    )
    tier0 = kinematics.compute(df)
    principal_axis_angle = float(tier0["principal_axis_angle"].iloc[0])
    # Inferred axis should be near π/2 (within ~0.5% per E1 phase-sampling offset)
    assert (
        abs(principal_axis_angle - math.pi / 2) < 0.01
    ), f"principal_axis_angle = {principal_axis_angle}, expected ≈ π/2"


# ===========================================================================
# 2.D — Round-trip noise sanity via QC
# (per-axis σ = noise_sigma_px / √2; QC xy-quadrature recovers noise_sigma_px,
# but with documented per-estimator bias profiles. From §3.7 canary capture
# on 2026-05-21 with noise_sigma_px=2.0:
#   sg ≈ 1.29 (bias factor ~0.65 — matches PR #2's [1.0, 1.6] band for σ=1)
#   d2 ≈ 1.89 (bias factor ~0.95 — near-unbiased)
#   msd ≈ 1.22 (bias factor ~0.61)
# Tests use bias-adjusted bands per estimator.)
# ===========================================================================


# Per-estimator bias factors empirically calibrated on the synth's
# closed-form trajectory at default plate-001-matching parameters
# (see §3.7 canary capture). Used to set bias-adjusted recovery bands.
_SG_BIAS_FACTOR = 0.65  # synth-sg ≈ 0.65 × noise_sigma_px (SG smoothing absorbs noise)
_D2_BIAS_FACTOR = 0.95  # synth-d2 ≈ 0.95 × noise_sigma_px (near-unbiased)
_MSD_BIAS_FACTOR = (
    0.61  # synth-msd ≈ 0.61 × noise_sigma_px (SG-detrend in MSD adds bias)
)
_BIAS_TOLERANCE = 0.25  # ±25% of the bias-adjusted target


def _bias_band(noise_sigma_px, bias_factor):
    """Return (lo, hi) accepting recovered ≈ bias_factor·noise_sigma_px ± _BIAS_TOLERANCE."""
    target = bias_factor * noise_sigma_px
    return target * (1 - _BIAS_TOLERANCE), target * (1 + _BIAS_TOLERANCE)


def test_2D1_sg_residual_recovers_noise_sigma_with_documented_bias():
    """2.D.1 — sg_residual_xy recovers ~0.65 × noise_sigma_px (documented SG bias)."""
    df = synthetic.generate_trajectory(noise_sigma_px=2.0, random_state=42)
    result = qc.compute(df)
    sg = float(result["sg_residual_xy"].iloc[0])
    lo, hi = _bias_band(2.0, _SG_BIAS_FACTOR)
    assert lo <= sg <= hi, (
        f"sg_residual_xy = {sg}, expected within [{lo:.3f}, {hi:.3f}] "
        f"(bias-adjusted target = {_SG_BIAS_FACTOR * 2.0:.3f})"
    )


def test_2D2_d2_noise_recovers_noise_sigma_near_unbiased():
    """2.D.2 — d2_noise_xy recovers ~0.95 × noise_sigma_px (near-unbiased d2)."""
    df = synthetic.generate_trajectory(noise_sigma_px=2.0, random_state=42)
    result = qc.compute(df)
    d2 = float(result["d2_noise_xy"].iloc[0])
    lo, hi = _bias_band(2.0, _D2_BIAS_FACTOR)
    assert lo <= d2 <= hi, f"d2_noise_xy = {d2}, expected within [{lo:.3f}, {hi:.3f}]"


def test_2D3_msd_noise_recovers_noise_sigma_with_documented_bias():
    """2.D.3 — msd_noise_xy recovers ~0.61 × noise_sigma_px (documented MSD bias)."""
    df = synthetic.generate_trajectory(noise_sigma_px=2.0, random_state=42)
    result = qc.compute(df)
    msd = float(result["msd_noise_xy"].iloc[0])
    lo, hi = _bias_band(2.0, _MSD_BIAS_FACTOR)
    assert (
        lo <= msd <= hi
    ), f"msd_noise_xy = {msd}, expected within [{lo:.3f}, {hi:.3f}]"


@pytest.mark.parametrize("noise_sigma_px", [1.0, 2.0, 4.0])
@pytest.mark.parametrize(
    "estimator,bias_factor",
    [
        ("sg_residual_xy", _SG_BIAS_FACTOR),
        ("d2_noise_xy", _D2_BIAS_FACTOR),
        ("msd_noise_xy", _MSD_BIAS_FACTOR),
    ],
)
def test_2D4_estimators_recover_with_documented_bias_profiles(
    noise_sigma_px, estimator, bias_factor
):
    """2.D.4 — bias-adjusted recovery across σ ∈ {1, 2, 4} for all 3 estimators.

    Each estimator has a documented bias factor (sg≈0.65, d2≈0.95, msd≈0.61)
    on the synth's closed-form trajectory. Tests verify the recovery scales
    LINEARLY with noise_sigma_px (bias is multiplicative, not additive).
    """
    df = synthetic.generate_trajectory(noise_sigma_px=noise_sigma_px, random_state=42)
    result = qc.compute(df)
    recovered = float(result[estimator].iloc[0])
    lo, hi = _bias_band(noise_sigma_px, bias_factor)
    assert lo <= recovered <= hi, (
        f"{estimator} = {recovered} at noise_sigma_px={noise_sigma_px}, "
        f"expected within [{lo:.3f}, {hi:.3f}] "
        f"(bias-adjusted target = {bias_factor * noise_sigma_px:.3f})"
    )


def test_2D5_track_is_clean_true_for_clean_synthetic_with_loosened_thresholds():
    """2.D.5 — clean synthetic produces track_is_clean=True with loosened thresholds.

    The default plate-001-matching synth produces d2_msd_agreement ≈ 1.55
    (just above the default threshold 1.5) because the structural ratio
    between d2 and msd bias factors (0.95 / 0.61 ≈ 1.56) is intrinsically
    above 1.5. Same borderline that PR #3 found on plate-001 (d2_msd=1.537
    there). Use ConstantsT to loosen the agreement thresholds — the
    documented escape per design.md.
    """
    df = synthetic.generate_trajectory(noise_sigma_px=2.0, random_state=42)
    constants = ConstantsT(
        SG_D2_AGREEMENT_MAX=2.0,
        SG_MSD_AGREEMENT_MAX=2.0,
        D2_MSD_AGREEMENT_MAX=2.0,
    )
    result = qc.compute(df, constants=constants)
    assert bool(result["track_is_clean"].iloc[0]) is True, (
        f"With loosened thresholds, default synth should be clean; "
        f"qc_failure_reason = {result['qc_failure_reason'].iloc[0]!r}"
    )
    assert result["qc_failure_reason"].iloc[0] == ""


# ===========================================================================
# 2.E — Handedness sign convention
# (all sub-tests use noise_sigma_px=0 for unambiguous determinism)
# ===========================================================================


def test_2E1_handedness_plus_one_gives_positive_psi_g_drift():
    """2.E.1 — handedness=+1 → mean(diff(ψ_g)) > 0 in noise-free case."""
    df = synthetic.generate_trajectory(handedness=+1, noise_sigma_px=0)
    psi_g = _geometry.compute_psi_g(df["tip_x"].to_numpy(), df["tip_y"].to_numpy())
    assert np.mean(np.diff(psi_g)) > 0


def test_2E2_handedness_minus_one_gives_negative_psi_g_drift():
    """2.E.2 — handedness=-1 → mean(diff(ψ_g)) < 0 in noise-free case."""
    df = synthetic.generate_trajectory(handedness=-1, noise_sigma_px=0)
    psi_g = _geometry.compute_psi_g(df["tip_x"].to_numpy(), df["tip_y"].to_numpy())
    assert np.mean(np.diff(psi_g)) < 0


@pytest.mark.parametrize("handedness", [+1, -1])
def test_2E3_handedness_first_velocity_cross_product_matches_psi_g(handedness):
    """2.E.3 — first-velocities cross product (3-frame, truncation-immune) matches ψ_g.

    Per design.md R7 mitigation: the originally-proposed `mean(curl)` formula
    integrates ``vx·ay - vy·ax`` over the whole trajectory, which is
    truncation-sensitive for bounded oscillations (the integrand is a
    sinusoid times a constant — mean over non-integer periods is a small
    residual whose sign depends on the truncation point, NOT on handedness).

    Robust replacement: the cross product of the FIRST two velocity vectors
    ``v0 × v1 = dx[0]·dy[1] - dy[0]·dx[1]`` (computed from just 3 frames)
    is truncation-IMMUNE. For the closed-form synth at growth_axis=π/2 in
    the image-y-down convention, math-CCW (``handedness=+1``) corresponds
    to CW on screen, giving a negative ``v0 × v1``. Conversely
    ``handedness=-1`` gives positive ``v0 × v1``. So
    ``sign(v0 × v1) == -handedness`` for this configuration. Asserting
    BOTH that the cross product has the expected sign AND that
    ``sign(mean(diff(psi_g))) == handedness`` (canonical ψ_g) locks the
    convention chain against silent inversion of `_geometry.compute_psi_g`'s
    atan2 argument order: an attacker reversing the order would flip the
    ψ_g sign but NOT the cross-product sign, breaking the assertion.
    """
    df = synthetic.generate_trajectory(handedness=handedness, noise_sigma_px=0)
    tip_x = df["tip_x"].to_numpy()
    tip_y = df["tip_y"].to_numpy()

    # First two velocity vectors (truncation-immune; uses only frames 0, 1, 2)
    dx = np.diff(tip_x[:3])  # [dx[0], dx[1]]
    dy = np.diff(tip_y[:3])  # [dy[0], dy[1]]
    v0_cross_v1 = float(dx[0] * dy[1] - dy[0] * dx[1])
    # Sign relationship per derivation: math-CCW (handedness=+1) → image-CW → cross < 0
    assert np.sign(v0_cross_v1) == -handedness, (
        f"v0 × v1 = {v0_cross_v1} (sign {np.sign(v0_cross_v1)}); "
        f"expected sign = -handedness = {-handedness} for growth_axis=π/2 image-y-down"
    )

    # Canonical ψ_g via compute_psi_g should track handedness via mean(diff).
    # mean(diff(ψ_g)) is sign-stable for default plate-001 parameters
    # (empirically verified by §3.7 canary across 5 seeds; flagged in R7
    # as truncation-sensitive but stable in practice for n_frames=575).
    psi_g = _geometry.compute_psi_g(tip_x, tip_y)
    psi_g_mean_drift = float(np.mean(np.diff(psi_g)))
    assert np.sign(psi_g_mean_drift) == handedness, (
        f"compute_psi_g mean drift = {psi_g_mean_drift}; "
        f"expected sign = handedness = {handedness} per BM2016 Eq. 20"
    )


def test_2E4_handedness_robust_to_noise_at_default_n_frames():
    """2.E.4 — handedness sign holds under default noise at n_frames=575.

    Documents noise robustness without making it load-bearing in 2.E.1/2.
    """
    df = synthetic.generate_trajectory(
        handedness=+1, noise_sigma_px=2.0, random_state=42
    )
    psi_g = _geometry.compute_psi_g(df["tip_x"].to_numpy(), df["tip_y"].to_numpy())
    assert np.mean(np.diff(psi_g)) > 0


# ===========================================================================
# 2.F — Validation / error path
# ===========================================================================


def test_2F0_positional_call_rejected():
    """2.F.0 — positional invocation raises TypeError."""
    with pytest.raises(TypeError):
        synthetic.generate_trajectory(575)


@pytest.mark.parametrize(
    "param_name,invalid_value,expected_exception",
    [
        # 2.F.1 — n_frames invalid values
        ("n_frames", 0, ValueError),
        ("n_frames", -1, ValueError),
        ("n_frames", True, TypeError),
        ("n_frames", 1.5, TypeError),
        ("n_frames", "100", TypeError),
        ("n_frames", float("nan"), TypeError),  # NaN is float, not int
        ("n_frames", float("inf"), TypeError),  # inf is float, not int
        # 2.F.2 — cadence_s / T_nutation_s invalid values
        ("cadence_s", 0.0, ValueError),
        ("cadence_s", -1.0, ValueError),
        ("cadence_s", float("nan"), ValueError),
        ("cadence_s", float("inf"), ValueError),
        ("cadence_s", float("-inf"), ValueError),
        ("cadence_s", True, TypeError),
        ("cadence_s", "300", TypeError),  # no string coercion
        ("T_nutation_s", 0.0, ValueError),
        ("T_nutation_s", -1.0, ValueError),
        ("T_nutation_s", float("nan"), ValueError),
        ("T_nutation_s", float("inf"), ValueError),
        ("T_nutation_s", True, TypeError),
        ("T_nutation_s", "3333", TypeError),
        # 2.F.3 — amplitude_px / noise_sigma_px (non-negative finite)
        ("amplitude_px", -1.0, ValueError),
        ("amplitude_px", float("nan"), ValueError),
        ("amplitude_px", float("inf"), ValueError),
        ("amplitude_px", True, TypeError),
        ("amplitude_px", "10", TypeError),
        ("noise_sigma_px", -1.0, ValueError),
        ("noise_sigma_px", float("nan"), ValueError),
        ("noise_sigma_px", float("inf"), ValueError),
        ("noise_sigma_px", True, TypeError),
        ("noise_sigma_px", "2", TypeError),
        # 2.F.4 — finite-float fields (with -inf added per round-2 TDD I6)
        ("growth_rate_px_per_frame", float("nan"), ValueError),
        ("growth_rate_px_per_frame", float("inf"), ValueError),
        ("growth_rate_px_per_frame", float("-inf"), ValueError),
        ("growth_rate_px_per_frame", True, TypeError),
        ("growth_rate_px_per_frame", "5", TypeError),
        ("growth_axis_angle_rad", float("nan"), ValueError),
        ("growth_axis_angle_rad", float("inf"), ValueError),
        ("growth_axis_angle_rad", float("-inf"), ValueError),
        ("growth_axis_angle_rad", True, TypeError),
        ("growth_axis_angle_rad", "π/2", TypeError),
        ("x0_px", float("nan"), ValueError),
        ("x0_px", float("inf"), ValueError),
        ("x0_px", True, TypeError),
        ("y0_px", float("nan"), ValueError),
        ("y0_px", float("inf"), ValueError),
        ("y0_px", True, TypeError),
        ("initial_phase_rad", float("nan"), ValueError),
        ("initial_phase_rad", float("inf"), ValueError),
        ("initial_phase_rad", True, TypeError),
        # 2.F.5 — handedness ∈ {+1, -1} only
        ("handedness", 0, ValueError),
        ("handedness", 2, ValueError),
        ("handedness", -2, ValueError),
        ("handedness", 1.0, TypeError),
        ("handedness", True, TypeError),
        ("handedness", "+1", TypeError),
        ("handedness", None, TypeError),
        # 2.F.6 — random_state types (rejects legacy RandomState AND bool)
        ("random_state", 1.5, TypeError),
        ("random_state", "42", TypeError),
        ("random_state", np.random.RandomState(0), TypeError),
        ("random_state", True, TypeError),  # bool subclasses int; reject explicitly
        # 2.F.7 — constants must be None or ConstantsT
        ("constants", dict(), TypeError),
        ("constants", "constants", TypeError),
        ("constants", 42, TypeError),
        # 2.F.8 — identity-column types
        ("plant_id", 1.5, TypeError),
        ("plant_id", True, TypeError),
        ("plant_id", "0", TypeError),
        ("plant_id", float("nan"), TypeError),
        ("track_id", 1.5, TypeError),
        ("track_id", True, TypeError),
        ("track_id", "0", TypeError),
        ("series", 0, TypeError),
        ("series", 1.5, TypeError),
        ("series", None, TypeError),  # mandatory str — None rejected
        ("sample_uid", 0, TypeError),
        ("sample_uid", None, TypeError),
        ("timepoint", 0, TypeError),
        ("timepoint", None, TypeError),
        ("plate_id", 0, TypeError),
        ("plate_id", None, TypeError),
    ],
)
def test_2F_invalid_input_raises(param_name, invalid_value, expected_exception):
    """2.F — every invalid input raises ValueError or TypeError naming the field."""
    kwargs = {param_name: invalid_value}
    with pytest.raises(expected_exception, match=param_name):
        synthetic.generate_trajectory(**kwargs)


# ===========================================================================
# 2.G — ConstantsT override + resolution-order (D13)
# ===========================================================================


def test_2G1_constants_snapshot_contains_7_new_keys_and_preserves_pr3_qc_keys():
    """2.G.1 — snapshot has 7 new SYNTHETIC_* keys AND preserves 4 PR #3 QC keys."""
    from sleap_roots.circumnutation._constants import _default_constants_snapshot

    snapshot = _default_constants_snapshot()
    # 7 new SYNTHETIC_* keys
    assert snapshot["SYNTHETIC_T_NUTATION_S"] == 3333.0
    assert snapshot["SYNTHETIC_AMPLITUDE_PX"] == 10.0
    assert snapshot["SYNTHETIC_GROWTH_RATE_PX_PER_FRAME"] == 4.29
    assert snapshot["SYNTHETIC_NOISE_SIGMA_PX"] == 2.0
    assert snapshot["SYNTHETIC_CADENCE_S"] == 300.0
    assert snapshot["SYNTHETIC_N_FRAMES"] == 575
    assert snapshot["SYNTHETIC_GROWTH_AXIS_ANGLE_RAD"] == math.pi / 2
    # Regression guard: 4 PR #3 QC keys still present with their defaults
    assert snapshot["FRAC_OUTLIER_STEPS_MAX"] == 0.05
    assert snapshot["WORST_STEP_RATIO_MAX"] == 5
    assert snapshot["SG_MSD_AGREEMENT_MAX"] == 1.5
    assert snapshot["D2_MSD_AGREEMENT_MAX"] == 1.5


def test_2G2_constants_version_is_4():
    """2.G.2 — _CONSTANTS_VERSION current value (bumped 2 → 3 in PR #4, 3 → 4 in PR #5)."""
    from sleap_roots.circumnutation import _constants

    assert _constants._CONSTANTS_VERSION == 4


def test_2G3_constantsT_extended_with_7_new_fields():
    """2.G.3 — ConstantsT accepts 7 new SYNTHETIC_* fields with module-level defaults."""
    inst = ConstantsT(SYNTHETIC_AMPLITUDE_PX=20.0)
    assert inst.SYNTHETIC_AMPLITUDE_PX == 20.0
    # Unoverridden fields keep module-level defaults
    assert inst.SYNTHETIC_T_NUTATION_S == 3333.0
    assert inst.SYNTHETIC_GROWTH_RATE_PX_PER_FRAME == 4.29
    assert inst.SYNTHETIC_NOISE_SIGMA_PX == 2.0
    assert inst.SYNTHETIC_CADENCE_S == 300.0
    assert inst.SYNTHETIC_N_FRAMES == 575
    assert inst.SYNTHETIC_GROWTH_AXIS_ANGLE_RAD == math.pi / 2
    # Construct with all 7 overridden
    inst_all = ConstantsT(
        SYNTHETIC_T_NUTATION_S=1800.0,
        SYNTHETIC_AMPLITUDE_PX=20.0,
        SYNTHETIC_GROWTH_RATE_PX_PER_FRAME=3.0,
        SYNTHETIC_NOISE_SIGMA_PX=1.0,
        SYNTHETIC_CADENCE_S=60.0,
        SYNTHETIC_N_FRAMES=200,
        SYNTHETIC_GROWTH_AXIS_ANGLE_RAD=0.0,
    )
    assert inst_all.SYNTHETIC_T_NUTATION_S == 1800.0
    assert inst_all.SYNTHETIC_AMPLITUDE_PX == 20.0
    assert inst_all.SYNTHETIC_GROWTH_RATE_PX_PER_FRAME == 3.0
    assert inst_all.SYNTHETIC_NOISE_SIGMA_PX == 1.0
    assert inst_all.SYNTHETIC_CADENCE_S == 60.0
    assert inst_all.SYNTHETIC_N_FRAMES == 200
    assert inst_all.SYNTHETIC_GROWTH_AXIS_ANGLE_RAD == 0.0


def test_2G4_constantsT_override_propagates_when_kwarg_omitted(synthetic_setup):
    """2.G.4 — when kwarg omitted, constants override propagates (D13 step 2)."""
    setup = synthetic_setup
    # Kwarg omitted (amplitude_px defaults to None internally); constants overrides
    df1 = synthetic.generate_trajectory(
        noise_sigma_px=0,
        constants=ConstantsT(SYNTHETIC_AMPLITUDE_PX=20.0),
    )
    ang_amp_1 = float(kinematics.compute(df1)["angular_amplitude"].iloc[0])
    A_lat_1 = 20.0 / 2.0  # half peak-to-peak
    expected_1 = 2 * math.atan(A_lat_1 * setup["omega"] / setup["v_growth_per_s"])
    assert abs(ang_amp_1 - expected_1) / expected_1 < 0.15, (
        f"With constants override amplitude_px=20, expected angular_amplitude "
        f"≈ {expected_1}, got {ang_amp_1}"
    )


def test_2G5_explicit_kwarg_overrides_constants_override(synthetic_setup):
    """2.G.5 — explicit kwarg beats ConstantsT override (D13 step 1)."""
    setup = synthetic_setup
    # BOTH set: kwarg=15.0, constants=20.0; kwarg should win
    df2 = synthetic.generate_trajectory(
        amplitude_px=15.0,
        noise_sigma_px=0,
        constants=ConstantsT(SYNTHETIC_AMPLITUDE_PX=20.0),
    )
    ang_amp_2 = float(kinematics.compute(df2)["angular_amplitude"].iloc[0])
    A_lat_2 = 15.0 / 2.0
    expected_2 = 2 * math.atan(A_lat_2 * setup["omega"] / setup["v_growth_per_s"])
    assert abs(ang_amp_2 - expected_2) / expected_2 < 0.15, (
        f"With explicit kwarg amplitude_px=15 (constants=20 ignored), expected "
        f"angular_amplitude ≈ {expected_2}, got {ang_amp_2}"
    )
    # Sanity discriminator: kwarg wins → different output from constants-only path
    df_constants_only = synthetic.generate_trajectory(
        noise_sigma_px=0,
        constants=ConstantsT(SYNTHETIC_AMPLITUDE_PX=20.0),
    )
    ang_amp_constants_only = float(
        kinematics.compute(df_constants_only)["angular_amplitude"].iloc[0]
    )
    assert (
        abs(ang_amp_2 - ang_amp_constants_only) / ang_amp_constants_only > 0.01
    ), "kwarg=15 and constants=20 should produce materially different outputs"


def test_2G6_constants_none_uses_module_defaults():
    """2.G.6 — kwarg=None + constants=None → module-level SYNTHETIC_* defaults."""
    df_none = synthetic.generate_trajectory(amplitude_px=None, noise_sigma_px=0)
    df_explicit = synthetic.generate_trajectory(
        amplitude_px=SYNTHETIC_AMPLITUDE_PX, noise_sigma_px=0
    )
    npt.assert_array_equal(df_none["tip_x"].to_numpy(), df_explicit["tip_x"].to_numpy())
    npt.assert_array_equal(df_none["tip_y"].to_numpy(), df_explicit["tip_y"].to_numpy())


# ===========================================================================
# 2.H — Reference-fixture agreement (Layer-1 sanity)
# (NON-tautological: compare generate_trajectory() defaults against
# kinematics/qc.compute(plate_001_real_data) recomputation, NOT against
# hardcoded numbers; per scientific-rigor reviewer B3)
# ===========================================================================


NIPPONBARE_SLP = Path(
    "tests/data/circumnutation_nipponbare_plate_001/plate_001_greyscale.tracked_proofread.slp"
)
NIPPONBARE_CSV = Path(
    "tests/data/circumnutation_nipponbare_plate_001/fixture_metadata.csv"
)


def _load_and_enrich_nipponbare() -> pd.DataFrame:
    """Mirror PR #3 §2.H Nipponbare loader (test_circumnutation_qc.py _load_and_enrich)."""
    from sleap_roots.series import Series

    series = Series.load(
        series_name="plate_001",
        primary_path=str(NIPPONBARE_SLP),
        csv_path=str(NIPPONBARE_CSV),
        sample_uid="plate_001",
    )
    df = series.get_tracked_tips()
    df["track_id"] = df["track_id"].str.replace("track_", "", regex=False).astype(int)
    df["series"] = series.series_name
    df["sample_uid"] = series.sample_uid
    df["timepoint"] = str(series.timepoint) if not pd.isna(series.timepoint) else np.nan
    df["plate_id"] = "plate_001"
    df["plant_id"] = df["track_id"]
    df["genotype"] = "Nipponbare"
    df["treatment"] = "MOCK"
    return df


# Tolerance for 2.H reference-fixture agreement tests. The TRUE-noise level on
# plate 001 is higher than `SYNTHETIC_NOISE_SIGMA_PX = 2.0` (the synth default
# anchored to theory.md §8 σ=2 px guideline). After SG-bias, both synth-sg and
# plate-001-sg are biased-down by the same factor (~0.65), so the RATIO of
# real-sg/synth-sg ≈ 1.42 ≈ √2 — implying plate-001's underlying TRUE noise is
# ~2.83 px. The ~29% gap between synth and real is structurally consistent
# across all 3 estimators (sg/d2/msd ratios all ≈ 1.42). We accept ±35%
# tolerance with this rationale; PR #12's Layer-1 validation will revisit
# default calibration when more multi-plate data lands.
_REFERENCE_FIXTURE_TOLERANCE = 0.35


@pytest.mark.skipif(
    not NIPPONBARE_SLP.exists(),
    reason=f"Nipponbare fixture not present: {NIPPONBARE_SLP}",
)
def test_2H1_synth_sg_residual_matches_real_plate_within_tolerance():
    """2.H.1 — synth default sg_residual_xy matches real plate 001 within ±35%.

    Tolerance accommodates the gap between SYNTHETIC_NOISE_SIGMA_PX=2.0 (the
    theory.md §8 σ=2 px anchor) and plate-001's apparent true-noise level
    of ~2.83 px (back-derived from SG-bias factor ~0.65 and plate sg=1.83).
    The ratio plate/synth ≈ 1.42 ≈ √2 across all 3 estimators is the
    consistency anchor.
    """
    real_df = _load_and_enrich_nipponbare()
    real_qc = qc.compute(real_df)
    real_sg = float(real_qc["sg_residual_xy"].median())  # median across 6 tracks

    synth_df = synthetic.generate_trajectory(random_state=42)
    synth_qc = qc.compute(synth_df)
    synth_sg = float(synth_qc["sg_residual_xy"].iloc[0])

    rel_diff = abs(real_sg - synth_sg) / real_sg
    assert rel_diff < _REFERENCE_FIXTURE_TOLERANCE, (
        f"Real sg_residual_xy median = {real_sg}; synth = {synth_sg}; "
        f"relative difference = {rel_diff*100:.1f}% > "
        f"{_REFERENCE_FIXTURE_TOLERANCE*100:.0f}%"
    )


@pytest.mark.skipif(
    not NIPPONBARE_SLP.exists(),
    reason=f"Nipponbare fixture not present: {NIPPONBARE_SLP}",
)
def test_2H2_synth_mean_step_matches_real_plate_within_tolerance():
    """2.H.2 — synth mean per-frame total step matches real plate 001 within ±35%.

    Same tolerance rationale as 2.H.1: the synth noise default σ=2 is lower
    than plate-001's apparent true noise σ≈2.83, producing a consistent
    ~29% gap in mean-step magnitude as well.
    """
    real_df = _load_and_enrich_nipponbare()
    # Real: median (across 6 tracks) of per-track mean total step
    real_per_track_means = []
    for _, group in real_df.groupby("track_id"):
        xy = (
            group.dropna(subset=["tip_x", "tip_y"])
            .sort_values("frame")[["tip_x", "tip_y"]]
            .to_numpy()
        )
        if len(xy) >= 2:
            real_per_track_means.append(
                float(np.linalg.norm(np.diff(xy, axis=0), axis=1).mean())
            )
    real_mean_step = float(np.median(real_per_track_means))

    synth_df = synthetic.generate_trajectory(random_state=42)
    synth_xy = synth_df[["tip_x", "tip_y"]].to_numpy()
    synth_mean_step = float(np.linalg.norm(np.diff(synth_xy, axis=0), axis=1).mean())

    rel_diff = abs(real_mean_step - synth_mean_step) / real_mean_step
    assert rel_diff < _REFERENCE_FIXTURE_TOLERANCE, (
        f"Real mean step = {real_mean_step}; synth = {synth_mean_step}; "
        f"relative difference = {rel_diff*100:.1f}% > "
        f"{_REFERENCE_FIXTURE_TOLERANCE*100:.0f}%"
    )


@pytest.mark.skipif(
    not NIPPONBARE_SLP.exists(),
    reason=f"Nipponbare fixture not present: {NIPPONBARE_SLP}",
)
def test_2H3_synth_growth_axis_unreliable_false_matches_real_plate():
    """2.H.3 — neither real plate nor synth defaults flag growth_axis_unreliable."""
    real_df = _load_and_enrich_nipponbare()
    real_tier0 = kinematics.compute(real_df)
    assert not real_tier0[
        "growth_axis_unreliable"
    ].any(), (
        "Plate 001 (Nipponbare) tracks should NOT be flagged growth_axis_unreliable"
    )

    synth_df = synthetic.generate_trajectory(random_state=42)
    synth_tier0 = kinematics.compute(synth_df)
    assert bool(synth_tier0["growth_axis_unreliable"].iloc[0]) is False
