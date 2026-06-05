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

from sleap_roots.circumnutation._geometry import compute_psi_g, compute_signed_area


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
