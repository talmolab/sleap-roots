"""Tests for ``sleap_roots.circumnutation.temporal_cwt`` (PR #5).

8-section test taxonomy mirroring PR #4's ``test_circumnutation_synthetic.py``:

- §2.A schema/structural (ScaleogramResult / RidgeResult dtype contracts,
  field shapes, frozen attrs class, scalar field values, caplog DEBUG
  emissions on the happy path)
- §2.B determinism (CC-6): same-process bit-identical + cross-OS canary at
  atol=1e-9 against ``synthetic.generate_trajectory(random_state=0,
  n_frames=128, ...)`` input
- §2.C parameter recovery via independent analytical (raw ``np.sin``) and
  synthetic (PR #4) oracles
- §2.D COI mask correctness — cell-by-cell round-trip via the shared
  ``_coi_boundary_samples`` private-but-test-importable helper
- §2.E ridge sanity — single-frequency concentration + COI-interior
  max-fraction dispersion test on pure-noise input
- §2.F validation/errors (parametrized; ids enumerated below)
- §2.G ConstantsT override + 2-tier resolution-order
- §2.H reference-fixture sanity — §2.H.1 proofread-fixture constraint
  satisfaction across 6 tracks; §2.H.2 Layer-1 synthetic sanity at ±10%

Anchors: spec delta at ``openspec/changes/add-circumnutation-temporal-cwt-machinery/specs/circumnutation/spec.md``;
design.md D1–D9 + R1–R6; theory.md §3.4 (H1–H3) + §6.5 (Nyquist) + §7.2
(Tier 1 trait table) + §7.6 (QC ``coi_fraction_t1``); preliminary_results
Summary + §1.2 + §2.1 + §4.4 (Derr Sept-2025 plate-001 empirical anchors).

The Reconciliation Appendix in design.md tracks findings from 3 rounds of
critical review on design.md + 2 rounds of /openspec-review on the OpenSpec
proposal; this test file's 80-id taxonomy is the result of all 5 rounds.
"""

import logging
import math
from pathlib import Path

import attrs
import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from sleap_roots.circumnutation import synthetic
from sleap_roots.circumnutation._constants import (
    ConstantsT,
    _default_constants_snapshot,
    _CONSTANTS_VERSION,
    _SCHEMA_VERSION,
    COI_EFOLDING_FACTOR,
    CWT_SCALE_COUNT_DEFAULT,
    CWT_PERIOD_MIN_NYQUIST_FACTOR,
    CWT_PERIOD_MAX_SIGNAL_FRACTION,
    WAVELET_DEFAULT_TEMPORAL,
    # PR #3 QC constants (§2.G.5 regression guard against accidental deletion
    # during the PR #5 _CONSTANTS_VERSION bump)
    FRAC_OUTLIER_STEPS_MAX,
    WORST_STEP_RATIO_MAX,
    SG_MSD_AGREEMENT_MAX,
    D2_MSD_AGREEMENT_MAX,
)
from sleap_roots.circumnutation.temporal_cwt import (
    compute_scaleogram,
    extract_ridge,
    ScaleogramResult,
    RidgeResult,
    _coi_boundary_samples,  # private-but-test-importable helper (design.md D3)
)
from sleap_roots.series import Series


# Module-logger name PR #5 emits DEBUG records under. Used by §2.A.11 / §2.A.12.
TEMPORAL_CWT_LOGGER = "sleap_roots.circumnutation.temporal_cwt"


# ===========================================================================
# §2.A — Schema / structural tests
# ===========================================================================


def _make_default_x(n_frames: int = 575) -> np.ndarray:
    """Return a deterministic finite float64 array of length ``n_frames``.

    Helper for §2.A / §2.G / §2.B same-process tests where the exact signal
    content does not matter — only that ``compute_scaleogram`` accepts it
    cleanly.
    """
    return np.linspace(0.0, 100.0, n_frames, dtype=np.float64)


def test_2A1_compute_scaleogram_returns_ScaleogramResult():
    """§2.A.1: ``compute_scaleogram`` returns a ``ScaleogramResult`` instance."""
    result = compute_scaleogram(_make_default_x(), 300.0)
    assert isinstance(result, ScaleogramResult)


def test_2A2_ScaleogramResult_field_shapes_and_dtypes():
    """§2.A.2: ScaleogramResult field shapes + dtypes for representative n_frames=575."""
    result = compute_scaleogram(_make_default_x(575), 300.0)
    assert result.scaleogram.shape == (CWT_SCALE_COUNT_DEFAULT, 575)
    assert result.scaleogram.dtype == np.complex128
    assert result.scales.shape == (CWT_SCALE_COUNT_DEFAULT,)
    assert result.scales.dtype == np.float64
    assert result.periods_s.shape == (CWT_SCALE_COUNT_DEFAULT,)
    assert result.periods_s.dtype == np.float64
    assert result.frequencies_hz.shape == (CWT_SCALE_COUNT_DEFAULT,)
    assert result.frequencies_hz.dtype == np.float64
    assert result.coi_mask.shape == result.scaleogram.shape
    assert result.coi_mask.dtype == bool


def test_2A3_scales_strictly_monotonic_increasing():
    """§2.A.3: scales are strictly monotonic increasing."""
    result = compute_scaleogram(_make_default_x(), 300.0)
    assert (np.diff(result.scales) > 0).all()


def test_2A4_periods_s_frequencies_hz_inverse_relation():
    """§2.A.4: frequencies_hz * periods_s == 1 within numerical precision."""
    result = compute_scaleogram(_make_default_x(), 300.0)
    assert np.allclose(result.frequencies_hz * result.periods_s, 1.0, atol=1e-12)


def test_2A5_ScaleogramResult_is_frozen_attrs_class():
    """§2.A.5: ScaleogramResult is a frozen attrs class with the 7 documented fields."""
    assert attrs.has(ScaleogramResult)
    field_names = [f.name for f in attrs.fields(ScaleogramResult)]
    assert field_names == [
        "scaleogram",
        "scales",
        "periods_s",
        "frequencies_hz",
        "coi_mask",
        "cadence_s",
        "wavelet",
    ]
    result = compute_scaleogram(_make_default_x(), 300.0)
    with pytest.raises(attrs.exceptions.FrozenInstanceError):
        result.cadence_s = 999.0


def test_2A6_extract_ridge_returns_RidgeResult():
    """§2.A.6: ``extract_ridge`` returns a ``RidgeResult`` instance."""
    result = compute_scaleogram(_make_default_x(), 300.0)
    ridge = extract_ridge(result)
    assert isinstance(ridge, RidgeResult)


def test_2A7_RidgeResult_field_shapes_and_dtypes():
    """§2.A.7: RidgeResult field shapes + dtypes for n_frames=575."""
    result = compute_scaleogram(_make_default_x(575), 300.0)
    ridge = extract_ridge(result)
    assert ridge.frame_indices.shape == (575,)
    assert ridge.frame_indices.dtype == np.int64
    assert np.array_equal(ridge.frame_indices, np.arange(575, dtype=np.int64))
    assert ridge.periods_s.shape == (575,)
    assert ridge.periods_s.dtype == np.float64
    assert ridge.amplitudes.shape == (575,)
    assert ridge.amplitudes.dtype == np.float64
    assert ridge.powers.shape == (575,)
    assert ridge.powers.dtype == np.float64
    assert ridge.in_coi.shape == (575,)
    assert ridge.in_coi.dtype == bool


def test_2A8_RidgeResult_powers_equals_amplitudes_squared():
    """§2.A.8: ridge.powers equals ridge.amplitudes**2 (redundancy preservation)."""
    result = compute_scaleogram(_make_default_x(), 300.0)
    ridge = extract_ridge(result)
    assert np.allclose(ridge.powers, ridge.amplitudes**2, atol=1e-15)


def test_2A9_RidgeResult_amplitudes_non_negative():
    """§2.A.9: ridge.amplitudes are all non-negative (they are |C|)."""
    result = compute_scaleogram(_make_default_x(), 300.0)
    ridge = extract_ridge(result)
    assert (ridge.amplitudes >= 0).all()


def test_2A10_RidgeResult_is_frozen_attrs_class():
    """§2.A.10: RidgeResult is a frozen attrs class with the 5 documented fields."""
    assert attrs.has(RidgeResult)
    field_names = [f.name for f in attrs.fields(RidgeResult)]
    assert field_names == [
        "frame_indices",
        "periods_s",
        "amplitudes",
        "powers",
        "in_coi",
    ]
    result = compute_scaleogram(_make_default_x(), 300.0)
    ridge = extract_ridge(result)
    with pytest.raises(attrs.exceptions.FrozenInstanceError):
        ridge.amplitudes = np.zeros(575)


def test_2A11_caplog_no_warning_or_error_on_default_call(caplog):
    """§2.A.11: no WARNING/ERROR/CRITICAL records on the happy path from temporal_cwt.

    Filter records by logger name per /copilot-review round-2 C3 — pytest's
    caplog captures globally, so unrelated WARNING+ records from other loggers
    would otherwise produce false-positive failures.
    """
    with caplog.at_level(logging.WARNING, logger=TEMPORAL_CWT_LOGGER):
        result = compute_scaleogram(_make_default_x(), 300.0)
        extract_ridge(result)
    module_records = [r for r in caplog.records if r.name == TEMPORAL_CWT_LOGGER]
    assert all(rec.levelno < logging.WARNING for rec in module_records), (
        f"Unexpected WARNING/ERROR records on {TEMPORAL_CWT_LOGGER}: "
        f"{[(rec.levelname, rec.message) for rec in module_records if rec.levelno >= logging.WARNING]}"
    )


def test_2A12_caplog_debug_messages_contain_required_tokens(caplog):
    """§2.A.12: DEBUG records carry the documented tokens for both functions.

    Filter records by logger name per /copilot-review round-2 C4 — pytest's
    caplog captures globally; only the temporal_cwt module's DEBUG emissions
    are part of the contract.
    """
    with caplog.at_level(logging.DEBUG, logger=TEMPORAL_CWT_LOGGER):
        result = compute_scaleogram(_make_default_x(), 300.0)
        extract_ridge(result)

    debug_messages = [
        rec.message
        for rec in caplog.records
        if rec.name == TEMPORAL_CWT_LOGGER and rec.levelno == logging.DEBUG
    ]
    # compute_scaleogram message
    compute_msgs = [m for m in debug_messages if m.startswith("compute_scaleogram(")]
    assert (
        compute_msgs
    ), "compute_scaleogram emitted no DEBUG record starting with 'compute_scaleogram('"
    cmsg = compute_msgs[0]
    for token in (
        "n_frames=",
        "cadence_s=",
        "n_scales=",
        "period_min_s=",
        "period_max_s=",
        "wavelet=",
    ):
        assert (
            token in cmsg
        ), f"compute_scaleogram DEBUG missing token {token!r}: {cmsg}"
    # extract_ridge message
    ridge_msgs = [m for m in debug_messages if m.startswith("extract_ridge(")]
    assert (
        ridge_msgs
    ), "extract_ridge emitted no DEBUG record starting with 'extract_ridge('"
    rmsg = ridge_msgs[0]
    for token in ("n_scales=", "n_frames="):
        assert token in rmsg, f"extract_ridge DEBUG missing token {token!r}: {rmsg}"


def test_2A13_scalar_field_values():
    """§2.A.13: cadence_s + wavelet scalar field values are exact (per round-1 TDD-I9)."""
    x = np.linspace(0.0, 100.0, 64, dtype=np.float64)
    result = compute_scaleogram(x, 300.0)
    assert result.cadence_s == 300.0
    assert isinstance(result.cadence_s, float)
    assert result.wavelet == "cmor1.5-1.0"


# ===========================================================================
# §2.B — Determinism tests (CC-6)
# ===========================================================================


def test_2B1_compute_scaleogram_same_input_bit_identical_in_process():
    """§2.B.1: same input → bit-identical scaleogram in the same process (atol=0)."""
    x = _make_default_x()
    r1 = compute_scaleogram(x, 300.0)
    r2 = compute_scaleogram(x, 300.0)
    assert np.array_equal(r1.scaleogram, r2.scaleogram)
    assert np.array_equal(r1.scales, r2.scales)
    assert np.array_equal(r1.periods_s, r2.periods_s)
    assert np.array_equal(r1.frequencies_hz, r2.frequencies_hz)
    assert np.array_equal(r1.coi_mask, r2.coi_mask)


def test_2B2_extract_ridge_same_input_bit_identical_in_process():
    """§2.B.2: same ScaleogramResult → bit-identical RidgeResult (atol=0)."""
    x = _make_default_x()
    result = compute_scaleogram(x, 300.0)
    ridge1 = extract_ridge(result)
    ridge2 = extract_ridge(result)
    assert np.array_equal(ridge1.periods_s, ridge2.periods_s)
    assert np.array_equal(ridge1.amplitudes, ridge2.amplitudes)
    assert np.array_equal(ridge1.powers, ridge2.powers)
    assert np.array_equal(ridge1.in_coi, ridge2.in_coi)
    assert np.array_equal(ridge1.frame_indices, ridge2.frame_indices)


# Canary test §2.B.3 — REGRESSION DETECTOR for future pywt / numpy / BLAS drift.
#
# Provenance (captured via scripts/circumnutation/capture_temporal_cwt_canary.py):
#   Capture date (UTC): 2026-05-26T19:04:34Z
#   Platform: Windows 10 (AMD64) / Python 3.11.13
#   numpy 2.3.4 / pywt 1.8.0 / BLAS: scipy-openblas 0.3.30 (USE64BITINT, Haswell)
#   synthetic.py git SHA: fc0e6509298833d6faf90dd0152846a1b039f465
#   Run parameters: random_state=0, n_frames=128, T_nutation_s=3333,
#                   cadence_s=300, noise_sigma_px=0
#   scale_idx_at_target = 39 (period closest to 3333 s)
#   boundary_samples_at_target_scale = 14
#   frame_indices = (16, 64, 112)  — 2 frames inside COI-interior on each
#                                    edge plus the geometric middle
#
# This canary is NOT a correctness oracle. It catches future drift in pywt /
# numpy / BLAS that would silently change scaleogram values, NOT bugs in
# the CWT math (the math is validated by §2.C / §2.E / §2.H). Re-capture
# is appropriate if upstream dependencies legitimately change behavior;
# see design.md R1 for the contingency policy.
_CANARY_FRAME_INDICES = (16, 64, 112)
_CANARY_EXPECTED_VALUES = np.array(
    [
        complex(-4.5563971994174155, -6.3352192703165739),
        complex(8.0782684671402976, -1.5681205841070431),
        complex(-1.8705323728254921, 7.6372716973660646),
    ],
    dtype=np.complex128,
)


def test_2B3_cross_os_canary_at_atol_1e_9():
    """§2.B.3: cross-OS canary at atol=1e-9 against hardcoded expected values.

    Purpose: REGRESSION DETECTOR for future pywt/numpy/BLAS drift (NOT a
    correctness oracle). The 3 hardcoded complex values are captured from
    the GREEN-phase implementation via scripts/circumnutation/capture_temporal_cwt_canary.py.
    Bit-identical reproduction across Ubuntu/Windows/macOS at the time of
    PR merge defines this scenario's contract.

    RED-phase ships with np.nan placeholders so the test fails loudly until
    §3.5 capture lands; this is the documented design.md D5 / round-2 R2-N1
    discipline.
    """
    df = synthetic.generate_trajectory(
        random_state=0,
        n_frames=128,
        T_nutation_s=3333,
        cadence_s=300,
        noise_sigma_px=0,
    )
    x = df["tip_x"].to_numpy()
    result = compute_scaleogram(x, 300.0)
    scale_idx_at_target = int(np.argmin(np.abs(result.periods_s - 3333.0)))
    observed = result.scaleogram[scale_idx_at_target, list(_CANARY_FRAME_INDICES)]
    npt.assert_allclose(observed, _CANARY_EXPECTED_VALUES, atol=1e-9, rtol=0)


# ===========================================================================
# §2.C — Parameter recovery via independent oracles
# ===========================================================================


@pytest.mark.parametrize("T_nutation_s", [1500.0, 3333.0, 7200.0])
def test_2C1_analytical_recovery_at_n_frames_1024(T_nutation_s):
    """§2.C.1: raw np.sin oracle at n_frames=1024 recovers T within ±5%.

    Independent of ``synthetic.generate_trajectory`` (closes the "what if
    the synth has a spectral defect" hole) per /openspec-review round-1
    TDD-I1. n_frames=1024 pinned explicitly per round-3 R3-B2 — at
    n_frames=575 the scale-grid step is ~7% and ±5% recovery fails at
    T=3333 s.
    """
    cadence_s = 300.0
    n_frames = 1024
    t = np.arange(n_frames, dtype=np.float64) * cadence_s
    x = np.sin(2.0 * math.pi * t / T_nutation_s) * 10.0
    result = compute_scaleogram(x, cadence_s)
    ridge = extract_ridge(result)
    interior = ~ridge.in_coi
    assert interior.sum() > 0, "COI saturates the ridge — parameter regime unviable"
    median_period = float(np.median(ridge.periods_s[interior]))
    rel_err = abs(median_period - T_nutation_s) / T_nutation_s
    assert rel_err < 0.05, (
        f"Median ridge period {median_period:.1f}s deviates by {rel_err:.2%} "
        f"from target T={T_nutation_s}s (±5% tolerance)"
    )


@pytest.mark.parametrize("T_nutation_s", [1500.0, 3333.0, 7200.0])
def test_2C2_synthetic_recovery_at_n_frames_575(T_nutation_s):
    """§2.C.2: ``synthetic.generate_trajectory`` at n_frames=575 recovers T within ±10%.

    ±10% absorbs the n=575 scale-grid discreteness. The ±5/n=1024 vs ±10/n=575
    split makes §2.C.1 the discriminating analytical oracle for synth defects.
    """
    cadence_s = 300.0
    df = synthetic.generate_trajectory(
        T_nutation_s=T_nutation_s,
        n_frames=575,
        cadence_s=cadence_s,
        noise_sigma_px=0,
    )
    x = df["tip_x"].to_numpy()
    result = compute_scaleogram(x, cadence_s)
    ridge = extract_ridge(result)
    interior = ~ridge.in_coi
    assert interior.sum() > 0, "COI saturates the ridge — parameter regime unviable"
    median_period = float(np.median(ridge.periods_s[interior]))
    rel_err = abs(median_period - T_nutation_s) / T_nutation_s
    assert rel_err < 0.10, (
        f"Median ridge period {median_period:.1f}s deviates by {rel_err:.2%} "
        f"from target T={T_nutation_s}s (±10% tolerance)"
    )


# ===========================================================================
# §2.D — COI mask correctness
# ===========================================================================


def test_2D1_coi_mask_cell_by_cell_via_shared_helper():
    """§2.D.1: cell-by-cell COI mask round-trip via _coi_boundary_samples (atol=0).

    Test imports the SAME _coi_boundary_samples helper the implementation
    uses, so the integer expression `int(math.ceil(coi_factor * scale))` is
    matched bit-exactly across test and impl. Per /openspec-review round-1
    TDD-I3 + design.md D3.
    """
    n_frames = 512
    x = np.linspace(0.0, 100.0, n_frames, dtype=np.float64)
    result = compute_scaleogram(x, 1.0)  # cadence=1 simplifies the period math
    for i_scale, s in enumerate(result.scales):
        boundary = _coi_boundary_samples(float(s), COI_EFOLDING_FACTOR)
        left = min(boundary, n_frames)
        right = max(0, n_frames - boundary)
        assert result.coi_mask[i_scale, :left].all(), (
            f"Left COI band missing at scale_idx={i_scale}, scale={s}: "
            f"expected first {left} frames True"
        )
        assert result.coi_mask[i_scale, right:].all(), (
            f"Right COI band missing at scale_idx={i_scale}, scale={s}: "
            f"expected last {n_frames - right} frames True"
        )
        if 2 * boundary < n_frames:
            assert not result.coi_mask[i_scale, left:right].any(), (
                f"Interior unexpectedly tainted at scale_idx={i_scale}: "
                f"expected frames [{left}, {right}) all False"
            )


def test_2D2_coi_fraction_increases_with_scale():
    """§2.D.2: coi_mask.mean(axis=1) is monotonically non-decreasing across scales."""
    x = _make_default_x()
    result = compute_scaleogram(x, 300.0)
    coi_fractions = result.coi_mask.mean(axis=1)
    diffs = np.diff(coi_fractions)
    # Allow tiny floating-point fluctuation (none expected, but be permissive)
    assert (diffs >= -1e-12).all(), (
        f"COI fraction NOT monotonically non-decreasing across scales: "
        f"min diff = {diffs.min()}"
    )


def test_2D3_coi_mask_small_at_smallest_scale():
    """§2.D.3: at the smallest scale, COI fraction matches helper-derived expectation."""
    n_frames = 575
    x = _make_default_x(n_frames)
    result = compute_scaleogram(x, 300.0)
    smallest_scale = float(result.scales[0])
    expected_boundary = _coi_boundary_samples(smallest_scale, COI_EFOLDING_FACTOR)
    expected_in_coi_count = min(expected_boundary, n_frames) + max(
        0, n_frames - max(0, n_frames - expected_boundary)
    )
    # The above counts overlap if 2*boundary ≥ n_frames; otherwise it's 2*boundary
    if 2 * expected_boundary < n_frames:
        expected_in_coi_count = 2 * expected_boundary
    else:
        expected_in_coi_count = n_frames
    expected_mean = expected_in_coi_count / n_frames
    actual_mean = float(result.coi_mask[0, :].mean())
    assert actual_mean == pytest.approx(expected_mean, abs=1e-12)
    assert (
        actual_mean < 0.05
    ), f"COI fraction at smallest scale unexpectedly large: {actual_mean:.4%}"


def test_2D4_coi_mask_saturates_at_largest_scale():
    """§2.D.4: at the largest scale, COI fraction is large (mostly unreliable)."""
    n_frames = 575
    x = _make_default_x(n_frames)
    result = compute_scaleogram(x, 300.0)
    largest_idx = result.scales.shape[0] - 1
    actual_mean = float(result.coi_mask[largest_idx, :].mean())
    assert (
        actual_mean > 0.5
    ), f"Largest scale should be mostly COI: got {actual_mean:.4%}"


# ===========================================================================
# §2.E — Ridge sanity
# ===========================================================================


def test_2E1_ridge_concentrated_for_single_frequency_input():
    """§2.E.1: single-frequency input → ridge concentrates at one scale (mode-fraction ≥ 0.85).

    Empirical capture (GREEN-phase): record the measured mode-fraction in
    the assertion failure message so future drift below 0.85 produces an
    informative diff. Per /openspec-review round-1 TDD-I8.
    """
    cadence_s = 300.0
    n_frames = 1024
    T = 3333.0
    t = np.arange(n_frames, dtype=np.float64) * cadence_s
    x = np.sin(2.0 * math.pi * t / T) * 10.0
    result = compute_scaleogram(x, cadence_s)
    ridge = extract_ridge(result)
    interior = ~ridge.in_coi
    assert interior.sum() > 0
    ridge_scale_idx = np.argmax(np.abs(result.scaleogram), axis=0)
    bin_counts = np.bincount(
        ridge_scale_idx[interior], minlength=CWT_SCALE_COUNT_DEFAULT
    )
    mode_fraction = bin_counts.max() / interior.sum()
    assert mode_fraction >= 0.85, (
        f"Single-frequency input did not concentrate the ridge: "
        f"mode_fraction = {mode_fraction:.4f} (expected ≥ 0.85). "
        f"If GREEN-phase measurement consistently exceeds 0.85, lock the "
        f"observed value here as a regression detector."
    )


def test_2E2_ridge_dispersed_for_pure_noise_input():
    """§2.E.2: pure-noise input → ridge dispersed (max-fraction < 0.5).

    Non-degeneracy dispersion test (per round-3 R3-B1 — replaces uniform-null
    chi-square test that fails 100% in practice because CWT of white noise
    is not uniformly distributed across log-spaced scales). Restricted to
    COI-interior frames per round-3 R3-I4.
    """
    n_frames = 1024
    cadence_s = 300.0
    rng = np.random.default_rng(0)
    x = rng.standard_normal(n_frames)
    result = compute_scaleogram(x, cadence_s)
    ridge = extract_ridge(result)
    interior = ~ridge.in_coi
    assert interior.sum() > 0
    ridge_scale_idx = np.argmax(np.abs(result.scaleogram), axis=0)
    bin_counts = np.bincount(
        ridge_scale_idx[interior], minlength=CWT_SCALE_COUNT_DEFAULT
    )
    max_fraction = bin_counts.max() / interior.sum()
    assert max_fraction < 0.5, (
        f"Pure-noise ridge unexpectedly collapsed to one scale: "
        f"max_fraction = {max_fraction:.4f}"
    )


def test_2E3_in_coi_flagged_at_edges():
    """§2.E.3: first/last few frames are in-COI (any scale's COI band ≥ 1)."""
    x = _make_default_x(128)
    result = compute_scaleogram(x, 300.0)
    ridge = extract_ridge(result)
    assert ridge.in_coi[0:3].all(), f"First 3 frames not in-COI: {ridge.in_coi[0:3]}"
    assert ridge.in_coi[-3:].all(), f"Last 3 frames not in-COI: {ridge.in_coi[-3:]}"


def test_2E4_in_coi_consistent_with_scaleogram_coi_mask():
    """§2.E.4: ridge.in_coi[i] == result.coi_mask[ridge_scale_idx[i], i] for all i."""
    x = _make_default_x()
    result = compute_scaleogram(x, 300.0)
    ridge = extract_ridge(result)
    ridge_scale_idx = np.argmax(np.abs(result.scaleogram), axis=0)
    expected_in_coi = result.coi_mask[ridge_scale_idx, np.arange(len(ridge.in_coi))]
    assert np.array_equal(ridge.in_coi, expected_in_coi)


# ===========================================================================
# §2.F — Validation / errors (parametrized; current id count tracked via
# `pytest --collect-only -q tests/test_circumnutation_temporal_cwt.py | grep '::test_2F'`
# rather than a stale hardcoded number — per /copilot-review round-3 on PR #213)
# ===========================================================================


def _make_x_with_nan() -> np.ndarray:
    x = np.linspace(0.0, 100.0, 32, dtype=np.float64)
    x[5] = np.nan
    return x


def _make_x_with_pos_inf() -> np.ndarray:
    x = np.linspace(0.0, 100.0, 32, dtype=np.float64)
    x[5] = np.inf
    return x


def _make_x_with_neg_inf() -> np.ndarray:
    x = np.linspace(0.0, 100.0, 32, dtype=np.float64)
    x[5] = -np.inf
    return x


# §2.F.1 — invalid x (parametrize cases enumerated below; count tracked via
# `pytest --collect-only -q tests/test_circumnutation_temporal_cwt.py::test_2F1_compute_scaleogram_x_invalid`
# rather than a stale hardcoded number — per /copilot-review round-3 on PR #213)
@pytest.mark.parametrize(
    "bad_x",
    [
        pytest.param(_make_x_with_nan(), id="x_contains_nan"),
        pytest.param(_make_x_with_pos_inf(), id="x_contains_pos_inf"),
        pytest.param(_make_x_with_neg_inf(), id="x_contains_neg_inf"),
        pytest.param(
            np.linspace(0, 100, 32, dtype=np.complex128), id="x_complex_dtype"
        ),
        pytest.param(np.zeros((10, 10), dtype=np.float64), id="x_2d_ndarray"),
        pytest.param(np.array([1.0, "a", 2.0], dtype=object), id="x_object_dtype"),
        pytest.param(np.array([], dtype=np.float64), id="x_length_0"),
        pytest.param(np.array([1.0], dtype=np.float64), id="x_length_1"),
        pytest.param(
            np.linspace(0, 1, 8, dtype=np.float64), id="x_length_8_below_floor"
        ),
        pytest.param([1.0, 2.0, 3.0], id="x_list_not_ndarray"),
        pytest.param(
            np.array(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"], dtype="<U1"),
            id="x_string_dtype",
        ),
    ],
)
def test_2F1_compute_scaleogram_x_invalid(bad_x):
    """§2.F.1: invalid x raises ValueError or TypeError naming the field."""
    with pytest.raises(
        (ValueError, TypeError), match=r"x|MIN_FRAMES|dtype|shape|length|finite"
    ):
        compute_scaleogram(bad_x, 300.0)


# §2.F.2 — invalid cadence_s (9 ids; np.bool_ added per round-1 Code-I1)
@pytest.mark.parametrize(
    "bad_cadence,expected_exc",
    [
        pytest.param(0, ValueError, id="cadence_s_zero"),
        pytest.param(-1.0, ValueError, id="cadence_s_negative"),
        pytest.param(float("nan"), ValueError, id="cadence_s_nan"),
        pytest.param(float("inf"), ValueError, id="cadence_s_pos_inf"),
        pytest.param(float("-inf"), ValueError, id="cadence_s_neg_inf"),
        pytest.param(True, TypeError, id="cadence_s_python_bool"),
        pytest.param(np.bool_(True), TypeError, id="cadence_s_numpy_bool"),
        pytest.param("300", TypeError, id="cadence_s_str"),
        pytest.param([300.0], TypeError, id="cadence_s_list"),
    ],
)
def test_2F2_compute_scaleogram_cadence_s_invalid(bad_cadence, expected_exc):
    """§2.F.2: invalid cadence_s raises ValueError/TypeError naming the field."""
    x = np.linspace(0.0, 100.0, 32, dtype=np.float64)
    with pytest.raises(expected_exc, match=r"cadence_s"):
        compute_scaleogram(x, bad_cadence)


# §2.F.3 — invalid constants type (parametrize cases enumerated below)
@pytest.mark.parametrize(
    "bad_constants",
    [
        pytest.param(42, id="constants_int"),
        pytest.param("foo", id="constants_str"),
        pytest.param({}, id="constants_empty_dict"),
    ],
)
def test_2F3_compute_scaleogram_constants_type_invalid(bad_constants):
    """§2.F.3: invalid constants type raises TypeError naming the field."""
    x = np.linspace(0.0, 100.0, 32, dtype=np.float64)
    with pytest.raises(TypeError, match=r"constants"):
        compute_scaleogram(x, 300.0, constants=bad_constants)


# §2.F.4 — invalid constants field values (6 ids: 2 original per round-1 Code-I1 +
# 4 added per /copilot-review PR #213 C1 — CWT_SCALE_COUNT_DEFAULT and
# COI_EFOLDING_FACTOR were not validated)
@pytest.mark.parametrize(
    "bad_constants_obj,expected_field",
    [
        pytest.param(
            ConstantsT(CWT_PERIOD_MAX_SIGNAL_FRACTION=0.0),
            "CWT_PERIOD_MAX_SIGNAL_FRACTION",
            id="signal_fraction_zero",
        ),
        pytest.param(
            ConstantsT(CWT_PERIOD_MIN_NYQUIST_FACTOR=-1.0),
            "CWT_PERIOD_MIN_NYQUIST_FACTOR",
            id="nyquist_factor_negative",
        ),
        pytest.param(
            ConstantsT(CWT_SCALE_COUNT_DEFAULT=0),
            "CWT_SCALE_COUNT_DEFAULT",
            id="scale_count_zero",
        ),
        pytest.param(
            ConstantsT(CWT_SCALE_COUNT_DEFAULT=-5),
            "CWT_SCALE_COUNT_DEFAULT",
            id="scale_count_negative",
        ),
        pytest.param(
            ConstantsT(COI_EFOLDING_FACTOR=0.0),
            "COI_EFOLDING_FACTOR",
            id="coi_factor_zero",
        ),
        pytest.param(
            ConstantsT(COI_EFOLDING_FACTOR=-1.0),
            "COI_EFOLDING_FACTOR",
            id="coi_factor_negative",
        ),
        pytest.param(
            ConstantsT(COI_EFOLDING_FACTOR=float("nan")),
            "COI_EFOLDING_FACTOR",
            id="coi_factor_nan",
        ),
        pytest.param(
            ConstantsT(COI_EFOLDING_FACTOR=float("inf")),
            "COI_EFOLDING_FACTOR",
            id="coi_factor_inf",
        ),
        # Wavelet validation (added per /review-pr round-1 Behavioral-I2 — pywt
        # raises a TypeError/ValueError that does not name `constants.<FIELD>`
        # per the CC-1 naming convention; field-named ValueError preferred).
        pytest.param(
            ConstantsT(WAVELET_DEFAULT_TEMPORAL="garbage_wavelet"),
            "WAVELET_DEFAULT_TEMPORAL",
            id="wavelet_unknown_name",
        ),
        pytest.param(
            ConstantsT(WAVELET_DEFAULT_TEMPORAL=""),
            "WAVELET_DEFAULT_TEMPORAL",
            id="wavelet_empty_string",
        ),
        # bool / np.bool_ for CWT_SCALE_COUNT_DEFAULT (covers the bool-rejection
        # branch — Python bool is an int subclass, so the int-isinstance check
        # would silently pass it as `n_scales=1`/`n_scales=0` if not explicitly
        # guarded).
        pytest.param(
            ConstantsT(CWT_SCALE_COUNT_DEFAULT=True),
            "CWT_SCALE_COUNT_DEFAULT",
            id="scale_count_python_bool",
        ),
        pytest.param(
            ConstantsT(CWT_SCALE_COUNT_DEFAULT=np.bool_(True)),
            "CWT_SCALE_COUNT_DEFAULT",
            id="scale_count_numpy_bool",
        ),
        # Non-numeric type for a 'float' ConstantsT field (per /copilot-review
        # round-2 on PR #213 — attrs does not enforce type annotations at
        # construction, so a caller can pass `ConstantsT(COI_EFOLDING_FACTOR="hello")`;
        # the validator must raise a field-named ValueError, not a
        # `math.isfinite`-internal TypeError).
        pytest.param(
            ConstantsT(COI_EFOLDING_FACTOR="hello"),
            "COI_EFOLDING_FACTOR",
            id="coi_factor_non_numeric_string",
        ),
        pytest.param(
            ConstantsT(CWT_PERIOD_MAX_SIGNAL_FRACTION=None),
            "CWT_PERIOD_MAX_SIGNAL_FRACTION",
            id="signal_fraction_none",
        ),
    ],
)
def test_2F4_compute_scaleogram_constants_field_invalid(
    bad_constants_obj, expected_field
):
    """§2.F.4: invalid ConstantsT field values raise ValueError naming the field."""
    x = np.linspace(0.0, 100.0, 32, dtype=np.float64)
    with pytest.raises(ValueError, match=expected_field):
        compute_scaleogram(x, 300.0, constants=bad_constants_obj)


# §2.F.5 — extract_ridge type-invalid (parametrize cases enumerated below)
@pytest.mark.parametrize(
    "bad_input",
    [
        pytest.param(None, id="extract_ridge_None"),
        pytest.param({}, id="extract_ridge_dict"),
        pytest.param((1, 2, 3), id="extract_ridge_tuple"),
    ],
)
def test_2F5_extract_ridge_input_type_invalid(bad_input):
    """§2.F.5: extract_ridge rejects non-ScaleogramResult input with TypeError."""
    with pytest.raises(TypeError, match=r"ScaleogramResult"):
        extract_ridge(bad_input)


def _make_empty_scaleogram_result(empty_axis: str) -> ScaleogramResult:
    """Build an empty ScaleogramResult fixture for §2.F.6.

    Explicit construction per /openspec-review round-1 TDD-I13 — avoid
    truncated `...` placeholders.
    """
    if empty_axis == "scales":
        return ScaleogramResult(
            scaleogram=np.zeros((0, 10), dtype=np.complex128),
            scales=np.empty(0, dtype=np.float64),
            periods_s=np.empty(0, dtype=np.float64),
            frequencies_hz=np.empty(0, dtype=np.float64),
            coi_mask=np.empty((0, 10), dtype=bool),
            cadence_s=300.0,
            wavelet="cmor1.5-1.0",
        )
    if empty_axis == "frames":
        scales = np.linspace(2.0, 50.0, 10, dtype=np.float64)
        return ScaleogramResult(
            scaleogram=np.zeros((10, 0), dtype=np.complex128),
            scales=scales,
            periods_s=scales * 300.0,
            frequencies_hz=1.0 / (scales * 300.0),
            coi_mask=np.empty((10, 0), dtype=bool),
            cadence_s=300.0,
            wavelet="cmor1.5-1.0",
        )
    raise ValueError(f"Unknown empty_axis: {empty_axis}")


# §2.F.6 — extract_ridge empty (parametrize cases enumerated below)
@pytest.mark.parametrize(
    "empty_axis,match_pattern",
    [
        pytest.param("scales", r"n_scales|empty", id="empty_scales_axis"),
        pytest.param("frames", r"n_frames|empty", id="empty_frames_axis"),
    ],
)
def test_2F6_extract_ridge_empty_scaleogram(empty_axis, match_pattern):
    """§2.F.6: empty ScaleogramResult raises ValueError naming the empty axis."""
    empty = _make_empty_scaleogram_result(empty_axis)
    with pytest.raises(ValueError, match=match_pattern):
        extract_ridge(empty)


# §2.F.7 — extract_ridge constants-invalid (3 ids per round-3 R3-I5)
@pytest.mark.parametrize(
    "bad_constants",
    [
        pytest.param(42, id="extract_ridge_constants_int"),
        pytest.param("foo", id="extract_ridge_constants_str"),
        pytest.param({}, id="extract_ridge_constants_empty_dict"),
    ],
)
def test_2F7_extract_ridge_constants_invalid(bad_constants):
    """§2.F.7: invalid constants in extract_ridge raises TypeError."""
    x = np.linspace(0.0, 100.0, 32, dtype=np.float64)
    result = compute_scaleogram(x, 300.0)
    with pytest.raises(TypeError, match=r"constants"):
        extract_ridge(result, constants=bad_constants)


# §2.F.8 — n=9 exact-floor POSITIVE boundary (per round-1 TDD-I11)
def test_2F8_compute_scaleogram_x_at_min_frames_succeeds():
    """§2.F.8: compute_scaleogram accepts x at the exact MIN_FRAMES_REQUIRED floor."""
    x = np.linspace(0.0, 1.0, 9, dtype=np.float64)
    result = compute_scaleogram(x, 300.0)
    assert result.scaleogram.shape == (CWT_SCALE_COUNT_DEFAULT, 9)


def test_2F9_constants_accepts_numpy_integer_for_scale_count():
    """§2.F.9: ConstantsT(CWT_SCALE_COUNT_DEFAULT=np.int64(32)) is accepted (per /review-pr Behavioral-I1).

    Asymmetry with `_validate_cadence_s` (which accepts `np.integer`) is closed:
    `_validate_cwt_constants` accepts both Python `int` and `numpy.integer` for
    `CWT_SCALE_COUNT_DEFAULT`, rejecting only `bool` / `np.bool_` and non-integers.
    """
    x = np.linspace(0.0, 100.0, 64, dtype=np.float64)
    # np.int64(32) — common when restoring from a JSON sidecar that round-tripped through numpy
    custom = ConstantsT(CWT_SCALE_COUNT_DEFAULT=np.int64(32))
    result = compute_scaleogram(x, 300.0, constants=custom)
    assert result.scaleogram.shape[0] == 32


# ===========================================================================
# §2.G — ConstantsT override + 2-tier resolution-order
# ===========================================================================


def test_2G1_module_default_equals_ConstantsT_default():
    """§2.G.1: module default ≡ ConstantsT() default (2-tier resolution)."""
    x = _make_default_x(256)
    r1 = compute_scaleogram(x, 300.0)
    r2 = compute_scaleogram(x, 300.0, constants=ConstantsT())
    assert np.array_equal(r1.scaleogram, r2.scaleogram)
    assert np.array_equal(r1.scales, r2.scales)
    assert np.array_equal(r1.coi_mask, r2.coi_mask)


def test_2G2_override_CWT_SCALE_COUNT_DEFAULT():
    """§2.G.2: override CWT_SCALE_COUNT_DEFAULT propagates to output shape."""
    x = _make_default_x(256)
    custom = ConstantsT(CWT_SCALE_COUNT_DEFAULT=32)
    result = compute_scaleogram(x, 300.0, constants=custom)
    assert result.scaleogram.shape[0] == 32
    assert result.scales.shape == (32,)


def test_2G3_override_COI_EFOLDING_FACTOR_approximately_doubles_mask():
    """§2.G.3: doubling COI_EFOLDING_FACTOR roughly doubles the COI mask coverage.

    Empirical ratio ≈ 1.94 (not exactly 2) because the mask saturates at the
    highest scales where 2·half_width ≥ n_frames. Per round-3 R3-I2.
    """
    x = _make_default_x(256)
    r_default = compute_scaleogram(x, 300.0)
    custom = ConstantsT(COI_EFOLDING_FACTOR=2.0 * math.sqrt(1.5))
    r_doubled = compute_scaleogram(x, 300.0, constants=custom)
    ratio = r_doubled.coi_mask.mean() / r_default.coi_mask.mean()
    assert 1.7 < ratio < 2.0, (
        f"Expected COI mask coverage ratio in [1.7, 2.0] when doubling "
        f"COI_EFOLDING_FACTOR; got {ratio:.4f}"
    )


def test_2G4_constants_version_is_6():
    """§2.G.4: _CONSTANTS_VERSION == 6 (5 → 6 in PR #9; was 3→4 PR #5, 4→5 PR #6)."""
    assert _CONSTANTS_VERSION == 6
    assert _SCHEMA_VERSION == 1


def test_2G5_default_constants_snapshot_includes_all_required_keys():
    """§2.G.5: snapshot is a set-superset of PR #3 + PR #4 + PR #5 keys."""
    snapshot_keys = set(_default_constants_snapshot().keys())
    required_pr5 = {
        "COI_EFOLDING_FACTOR",
        "CWT_SCALE_COUNT_DEFAULT",
        "CWT_PERIOD_MIN_NYQUIST_FACTOR",
        "CWT_PERIOD_MAX_SIGNAL_FRACTION",
    }
    required_pr4 = {
        "SYNTHETIC_T_NUTATION_S",
        "SYNTHETIC_AMPLITUDE_PX",
        "SYNTHETIC_GROWTH_RATE_PX_PER_FRAME",
        "SYNTHETIC_NOISE_SIGMA_PX",
        "SYNTHETIC_CADENCE_S",
        "SYNTHETIC_N_FRAMES",
        "SYNTHETIC_GROWTH_AXIS_ANGLE_RAD",
    }
    required_pr3 = {
        "FRAC_OUTLIER_STEPS_MAX",
        "WORST_STEP_RATIO_MAX",
        "SG_MSD_AGREEMENT_MAX",
        "D2_MSD_AGREEMENT_MAX",
    }
    assert snapshot_keys >= required_pr5 | required_pr4 | required_pr3


def test_2G6_min_frames_required_tracks_constants_override():
    """§2.G.6: MIN_FRAMES_REQUIRED updates correctly under ConstantsT override."""
    # With CWT_PERIOD_MIN_NYQUIST_FACTOR=4.0 and CWT_PERIOD_MAX_SIGNAL_FRACTION=0.25,
    # floor(4.0/0.25) + 1 = 17 frames required.
    custom = ConstantsT(CWT_PERIOD_MIN_NYQUIST_FACTOR=4.0)
    # Tightened regex per /review-pr round-1 Testing-I2: drop "finite" — that
    # token would also match a validator-misfire path, masking a wrong-error
    # regression. The actual error path for an under-floor `x` length comes
    # from `_validate_x` and names `x` or `MIN_FRAMES`.
    with pytest.raises(ValueError, match=r"x|MIN_FRAMES|length"):
        compute_scaleogram(
            np.linspace(0.0, 1.0, 16, dtype=np.float64), 300.0, constants=custom
        )
    # 17 succeeds
    result = compute_scaleogram(
        np.linspace(0.0, 1.0, 17, dtype=np.float64), 300.0, constants=custom
    )
    assert result.scaleogram.shape[1] == 17


def test_2G7_nyquist_ratio_max_docstring_cross_references_cwt_constant():
    """§2.G.7: _constants.py source contains the cross-reference text per Spec-B2.

    The cross-reference is text in the module source (docstring/comment),
    not a runtime attribute. We grep the source file for the proximity
    of NYQUIST_RATIO_MAX and CWT_PERIOD_MAX_SIGNAL_FRACTION declarations.
    """
    import inspect

    from sleap_roots.circumnutation import _constants as constants_mod

    source = inspect.getsource(constants_mod)
    nyquist_pos = source.find("NYQUIST_RATIO_MAX:")
    cwt_pos = source.find("CWT_PERIOD_MAX_SIGNAL_FRACTION:")
    assert nyquist_pos > -1, "NYQUIST_RATIO_MAX declaration not found"
    assert cwt_pos > -1, "CWT_PERIOD_MAX_SIGNAL_FRACTION declaration not found"
    # Reciprocal cross-reference: each constant's docstring must mention the
    # other. Check a ±800-char window around each declaration.
    nearby_to_nyquist = source[max(0, nyquist_pos - 800) : nyquist_pos + 800]
    assert "CWT_PERIOD_MAX_SIGNAL_FRACTION" in nearby_to_nyquist, (
        "NYQUIST_RATIO_MAX's nearby docstring window does not mention "
        "CWT_PERIOD_MAX_SIGNAL_FRACTION (cross-reference missing)"
    )
    nearby_to_cwt = source[max(0, cwt_pos - 800) : cwt_pos + 800]
    assert "NYQUIST_RATIO_MAX" in nearby_to_cwt, (
        "CWT_PERIOD_MAX_SIGNAL_FRACTION's nearby docstring window does not "
        "mention NYQUIST_RATIO_MAX (reciprocal cross-reference missing)"
    )


# ===========================================================================
# §2.H — Reference-fixture sanity
# ===========================================================================


PROOFREAD_SLP = Path(
    "tests/data/circumnutation_nipponbare_plate_001/plate_001_greyscale.tracked_proofread.slp"
)


def _load_proofread_track_x(track_id: int) -> np.ndarray:
    """Load tip_x for the given track_id from the Nipponbare proofread fixture.

    Returns a deterministic float64 1-D array of length 575 (verified zero-NaN
    and zero frame-gap pre-design across all 6 tracks).
    """
    series = Series.load(
        series_name="plate_001",
        primary_path=str(PROOFREAD_SLP),
    )
    df = series.get_tracked_tips()
    # track_id column from get_tracked_tips() is "track_<i>"; coerce to int.
    df["_track_id_int"] = (
        df["track_id"].str.replace("track_", "", regex=False).astype(int)
    )
    track_df = df[df["_track_id_int"] == track_id].sort_values("frame")
    return track_df["tip_x"].to_numpy(dtype=np.float64)


@pytest.mark.skipif(
    not PROOFREAD_SLP.exists(),
    reason=f"Proofread fixture not present: {PROOFREAD_SLP}",
)
@pytest.mark.parametrize(
    "track_id",
    [pytest.param(t, id=f"track_{t}") for t in range(6)],
)
def test_2H1_proofread_fixture_constraint_satisfaction(track_id):
    """§2.H.1: proofread fixture satisfies all D8 constraints + CWT shape contracts.

    Parametrized across the 6 tracks of plate_001_greyscale.tracked_proofread.slp.
    Pre-design empirical check confirmed: all 6 tracks have 575 finite-float64
    frames with zero NaN and zero frame gaps.
    """
    x = _load_proofread_track_x(track_id)

    # (a) D8 regression guard
    assert np.isfinite(x).all(), f"track_{track_id} contains non-finite values"
    assert len(x) >= 9, f"track_{track_id} has only {len(x)} frames (need ≥ 9)"
    assert (
        x.dtype.kind in "if"
    ), f"track_{track_id} dtype not float-coercible: {x.dtype}"

    # (b) compute_scaleogram does not raise (on raw tip_x — verifies the
    # validation/contract path works against real fixture data)
    result = compute_scaleogram(x, cadence_s=300.0)

    # (c) extract_ridge does not raise
    ridge = extract_ridge(result)

    # (d) scaleogram shape
    assert result.scaleogram.shape == (CWT_SCALE_COUNT_DEFAULT, 575)

    # (e) coi_mask shape matches scaleogram shape
    assert result.coi_mask.shape == result.scaleogram.shape

    # (f) scale range covers biologically-plausible nutation periods
    assert result.periods_s.min() < 1000.0, (
        f"period_min = {result.periods_s.min():.1f}s — scale range does not "
        f"cover sub-1000s periods (biologically expected)"
    )
    assert result.periods_s.max() > 10000.0, (
        f"period_max = {result.periods_s.max():.1f}s — scale range does not "
        f"extend above 10000s (biologically expected)"
    )

    # (g) COI fraction at target scale is well below COI_FRACTION_MAX = 0.5
    scale_idx_at_target = int(np.argmin(np.abs(result.periods_s - 3333.0)))
    coi_fraction_at_target = float(result.coi_mask[scale_idx_at_target, :].mean())
    assert coi_fraction_at_target < 0.10, (
        f"COI fraction at target 3333s scale = {coi_fraction_at_target:.4%} "
        f"(expected < 10%; measured ~4.87% at √1.5 factor — see design.md D3)"
    )

    # (h) Regression-detector check: ridge fields are finite and amplitudes
    # are strictly positive. This catches the "compute_scaleogram returns
    # shape-correct garbage" failure mode (NaN propagation, all-zero
    # scaleogram, ridge collapse to invalid indices) without asserting
    # biological plausibility of the recovered period.
    #
    # Why not assert the ridge period falls in the [1000, 10000] s "nutation"
    # band on raw tip_x: the proofread data has substantial lateral drift
    # (~70-170 px peak-to-peak after linear detrend on tip_x — comparable
    # to or larger than the ~10 px expected nutation amplitude). The CWT
    # correctly identifies that low-frequency drift as the dominant signal
    # and locates the ridge at the longest available scale. That is NOT a
    # bug — it is the expected CWT response to a multi-scale signal where
    # low-frequency content dominates. Proper nutation-period recovery on
    # plate-001 requires the LATERAL coordinate projection per theory.md
    # CC-7, which is PR #6's `coordinate="lateral"` parameter (PR #5 does
    # not own that preprocessing). For PR #5, "machinery produces finite
    # output on real data" is the right scope-disciplined assertion.
    interior = ~ridge.in_coi
    assert interior.sum() > 0, "COI saturates the ridge — fixture-data issue"
    assert np.isfinite(
        ridge.periods_s
    ).all(), f"ridge.periods_s contains non-finite values for track_{track_id}"
    assert np.isfinite(
        ridge.amplitudes
    ).all(), f"ridge.amplitudes contains non-finite values for track_{track_id}"
    assert (
        ridge.amplitudes >= 0
    ).all(), f"ridge.amplitudes contains negative values for track_{track_id}"
    assert (ridge.amplitudes[interior] > 0).all(), (
        f"ridge.amplitudes has zero values in COI-interior for track_{track_id} "
        f"(possible all-zero scaleogram garbage)"
    )


def test_2H2_synthetic_layer_1_sanity():
    """§2.H.2: Layer-1 sanity via synthetic generator at noise_sigma_px=2.

    Recovers T_nutation_s=3333 within ±10%. NOT the Derr forensic match —
    that's PR #6's `derr_match_residual` trait.
    """
    df = synthetic.generate_trajectory(
        T_nutation_s=3333,
        n_frames=575,
        cadence_s=300,
        noise_sigma_px=2,
        random_state=0,
    )
    x = df["tip_x"].to_numpy()
    result = compute_scaleogram(x, 300.0)
    ridge = extract_ridge(result)
    interior = ~ridge.in_coi
    assert interior.sum() > 0
    median_period = float(np.median(ridge.periods_s[interior]))
    rel_err = abs(median_period - 3333.0) / 3333.0
    assert rel_err < 0.10, (
        f"Median ridge period {median_period:.1f}s deviates by {rel_err:.2%} "
        f"from synthetic T=3333s (±10% tolerance for noise_sigma_px=2 case)"
    )
