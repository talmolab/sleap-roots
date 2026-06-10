"""Tests for the Tier 3b spatial CWT machinery (PR #9).

Covers ``spatial_cwt.resample_curvature`` (non-uniform → uniform κ(s) resample),
``compute_scaleogram`` (cgau2 spatial CWT), and ``extract_ridge`` (per-position
λ(s_a) ridge), mirroring the PR #5 ``temporal_cwt`` / PR #8 ``midline``
machinery-test shape. See
``openspec/changes/add-circumnutation-tier3b-spatial-cwt/specs/circumnutation/spec.md``.
"""

import logging
import math

import attrs
import numpy as np
import numpy.testing as npt
import pytest


# ---------------------------------------------------------------------------
# §2 — ResampleResult + resample_curvature
# ---------------------------------------------------------------------------


def _clean_inputs(n=20, ds=5.0, lam=40.0):
    """Finite curvature on a uniform, monotonic arc-length grid (no mask)."""
    arc = np.arange(n, dtype=np.float64) * ds
    kappa = np.sin(2.0 * np.pi * arc / lam).astype(np.float64)
    return kappa, arc


def test_resample_result_is_frozen_attrs_with_six_fields():
    """§2.1: ResampleResult frozen attrs (eq=False), six fields in order."""
    from sleap_roots.circumnutation.spatial_cwt import ResampleResult

    names = tuple(f.name for f in attrs.fields(ResampleResult))
    assert names == (
        "kappa_uniform",
        "s_a_uniform_px",
        "ds",
        "n_unmasked",
        "arc_span_px",
        "is_degenerate",
    )
    r = ResampleResult(
        kappa_uniform=np.zeros(3),
        s_a_uniform_px=np.zeros(3),
        ds=1.0,
        n_unmasked=3,
        arc_span_px=2.0,
        is_degenerate=False,
    )
    with pytest.raises(attrs.exceptions.FrozenInstanceError):
        r.ds = 2.0


def test_resample_curvature_happy_path_shapes_and_apex_origin():
    """§2.3: shapes/dtypes, s_a[0]==0, n_unmasked, arc_span over survivors."""
    from sleap_roots.circumnutation.spatial_cwt import (
        resample_curvature,
        ResampleResult,
    )

    kappa, arc = _clean_inputs(n=20, ds=5.0)
    r = resample_curvature(kappa, arc)
    assert isinstance(r, ResampleResult)
    assert r.is_degenerate is False
    assert r.kappa_uniform.dtype == np.float64
    assert r.s_a_uniform_px.dtype == np.float64
    assert r.kappa_uniform.shape == r.s_a_uniform_px.shape
    assert r.s_a_uniform_px[0] == 0.0
    assert np.all(np.diff(r.s_a_uniform_px) > 0)
    assert r.ds > 0
    assert r.n_unmasked == 20
    assert r.arc_span_px == pytest.approx(arc.max() - arc.min())
    # pinned grid length = floor(arc_span/ds)+1
    assert len(r.kappa_uniform) == int(math.floor(r.arc_span_px / r.ds)) + 1


def test_resample_curvature_drops_masked_frames():
    """§2: n_unmasked counts only ~mask; masked frames contribute no knots."""
    from sleap_roots.circumnutation.spatial_cwt import resample_curvature

    kappa, arc = _clean_inputs(n=20, ds=5.0)
    mask = np.zeros(20, dtype=bool)
    mask[[3, 7, 11]] = True
    r = resample_curvature(kappa, arc, velocity_sub_noise_mask=mask)
    assert r.n_unmasked == 17


def test_resample_curvature_int_mask_coerces_like_bool():
    """§2.7b: an int 0/1 mask yields the same result as the bool mask."""
    from sleap_roots.circumnutation.spatial_cwt import resample_curvature

    kappa, arc = _clean_inputs(n=20, ds=5.0)
    bool_mask = np.zeros(20, dtype=bool)
    bool_mask[[2, 9]] = True
    int_mask = bool_mask.astype(int)
    rb = resample_curvature(kappa, arc, velocity_sub_noise_mask=bool_mask)
    ri = resample_curvature(kappa, arc, velocity_sub_noise_mask=int_mask)
    assert ri.n_unmasked == rb.n_unmasked
    npt.assert_array_equal(ri.kappa_uniform, rb.kappa_uniform)


def test_resample_curvature_drops_non_finite_without_raising(recwarn):
    """§2.7: non-finite (curvature, arc) pairs are DROPPED, not rejected."""
    from sleap_roots.circumnutation.spatial_cwt import resample_curvature

    kappa, arc = _clean_inputs(n=20, ds=5.0)
    kappa = kappa.copy()
    kappa[[4, 5]] = np.nan  # blow-up swept to NaN by PR #8
    r = resample_curvature(kappa, arc)
    assert r.is_degenerate is False
    assert r.n_unmasked == 18
    assert np.isfinite(r.kappa_uniform).all()
    assert len(recwarn) == 0


@pytest.mark.parametrize(
    "bad_kwargs",
    [
        {"curvature": [0.0] * 20},  # not ndarray
        {"curvature_2d": True},  # 2-D
        {"complex": True},  # complex dtype
        {"length_mismatch": True},  # unequal length
    ],
)
def test_resample_curvature_rejects_malformed_inputs(bad_kwargs):
    """§2.5: structural malformation raises TypeError/ValueError naming the field."""
    from sleap_roots.circumnutation.spatial_cwt import resample_curvature

    kappa, arc = _clean_inputs(n=20, ds=5.0)
    if "curvature" in bad_kwargs:
        with pytest.raises(TypeError, match="curvature_px_inv"):
            resample_curvature(bad_kwargs["curvature"], arc)
    elif "curvature_2d" in bad_kwargs:
        with pytest.raises(ValueError, match="curvature_px_inv"):
            resample_curvature(kappa.reshape(4, 5), arc)
    elif "complex" in bad_kwargs:
        with pytest.raises(ValueError, match="curvature_px_inv"):
            resample_curvature(kappa.astype(np.complex128), arc)
    elif "length_mismatch" in bad_kwargs:
        with pytest.raises(ValueError, match="equal length"):
            resample_curvature(kappa, arc[:-1])


def test_resample_curvature_rejects_wrong_length_mask():
    """§2.5: a wrong-length mask raises ValueError naming the field."""
    from sleap_roots.circumnutation.spatial_cwt import resample_curvature

    kappa, arc = _clean_inputs(n=20, ds=5.0)
    with pytest.raises(ValueError, match="velocity_sub_noise_mask"):
        resample_curvature(kappa, arc, velocity_sub_noise_mask=np.zeros(5, dtype=bool))


def test_resample_curvature_rejects_bad_constants():
    """§2.5: invalid constants type raises TypeError naming the field."""
    from sleap_roots.circumnutation.spatial_cwt import resample_curvature

    kappa, arc = _clean_inputs(n=20, ds=5.0)
    with pytest.raises(TypeError, match="constants"):
        resample_curvature(kappa, arc, constants=42)


def test_resample_curvature_too_few_survivors_is_degenerate(recwarn):
    """§2.7: fewer than MIN_SAMPLES survivors → graceful all-NaN, no warning."""
    from sleap_roots.circumnutation.spatial_cwt import resample_curvature

    kappa, arc = _clean_inputs(n=6, ds=5.0)  # 6 < 9
    r = resample_curvature(kappa, arc)
    assert r.is_degenerate is True
    assert r.kappa_uniform.size == 0 or np.isnan(r.kappa_uniform).all()
    assert len(recwarn) == 0


def test_resample_curvature_all_masked_is_degenerate(recwarn):
    """§2.7: fully-masked (n_unmasked==0) → degenerate, no max([])/median([]) crash."""
    from sleap_roots.circumnutation.spatial_cwt import resample_curvature

    kappa, arc = _clean_inputs(n=20, ds=5.0)
    mask = np.ones(20, dtype=bool)
    r = resample_curvature(kappa, arc, velocity_sub_noise_mask=mask)
    assert r.is_degenerate is True
    assert r.n_unmasked == 0
    assert len(recwarn) == 0


def test_resample_curvature_all_equal_arc_is_degenerate(recwarn):
    """§2.7: all-equal arc_length (zero span) → degenerate, no np.median([]) warning."""
    from sleap_roots.circumnutation.spatial_cwt import resample_curvature

    arc = np.full(20, 7.0, dtype=np.float64)
    kappa = np.linspace(0.0, 1.0, 20)
    r = resample_curvature(kappa, arc)
    assert r.is_degenerate is True
    assert len(recwarn) == 0


def test_resample_curvature_skewed_gaps_grid_too_short_is_degenerate():
    """§2.7 Stage-2: ≥9 survivors but skewed gaps → grid length < MIN → degenerate."""
    from sleap_roots.circumnutation.spatial_cwt import resample_curvature

    # 9 survivors, gaps [1,1,1,100,100,100,100,100] → arc_span=503, ds=100, grid len=6
    arc = np.array(
        [0.0, 1.0, 2.0, 3.0, 103.0, 203.0, 303.0, 403.0, 503.0], dtype=np.float64
    )
    kappa = np.sin(arc / 50.0)
    r = resample_curvature(kappa, arc)
    assert r.n_unmasked == 9
    assert r.is_degenerate is True


def test_resample_curvature_deduplicates_duplicate_s_a_deterministically(recwarn):
    """§2.9: duplicate arc_length → finite, deterministic, strictly-increasing xp."""
    from sleap_roots.circumnutation.spatial_cwt import resample_curvature

    arc = np.array(
        [0.0, 5.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0],
        dtype=np.float64,
    )
    kappa = np.sin(arc / 12.0)
    r1 = resample_curvature(kappa, arc)
    r2 = resample_curvature(kappa, arc)
    assert r1.is_degenerate is False
    assert np.isfinite(r1.kappa_uniform).all()
    npt.assert_array_equal(r1.kappa_uniform, r2.kappa_uniform)
    assert len(recwarn) == 0


def test_resample_curvature_value_pins_apex_orientation():
    """§2.11: a feature at the LARGEST arc_length lands at SMALL s_a (apex=0)."""
    from sleap_roots.circumnutation.spatial_cwt import resample_curvature

    n, ds = 20, 5.0
    arc = np.arange(n, dtype=np.float64) * ds
    kappa = np.zeros(n, dtype=np.float64)
    kappa[-1] = 1.0  # spike at the largest arc (the apex, latest tip)
    r = resample_curvature(kappa, arc)
    peak_s_a = r.s_a_uniform_px[int(np.argmax(r.kappa_uniform))]
    # The spike must be near s_a=0 (apex), NOT near s_a=max (base).
    assert peak_s_a < 0.25 * r.arc_span_px


def test_resample_curvature_emits_one_debug_record(caplog):
    """§2.13: exactly one DEBUG record with the documented tokens; no INFO/WARN/ERROR."""
    from sleap_roots.circumnutation.spatial_cwt import resample_curvature

    kappa, arc = _clean_inputs(n=20, ds=5.0)
    with caplog.at_level(
        logging.DEBUG, logger="sleap_roots.circumnutation.spatial_cwt"
    ):
        resample_curvature(kappa, arc)
    records = [
        r for r in caplog.records if r.name == "sleap_roots.circumnutation.spatial_cwt"
    ]
    assert len(records) == 1
    msg = records[0].getMessage()
    assert msg.startswith("resample_curvature(")
    for token in ("n_input=", "n_unmasked=", "ds=", "arc_span_px="):
        assert token in msg
    assert not [r for r in records if r.levelno > logging.DEBUG]
