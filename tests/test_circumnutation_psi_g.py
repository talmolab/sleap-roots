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
