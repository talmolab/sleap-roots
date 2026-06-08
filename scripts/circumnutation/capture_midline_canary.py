"""Capture deterministic canary values for the PR #8 Tier 3a midline cross-OS test.

Run this on the developer's primary machine to capture ``curvature_px_inv`` and
``arc_length_px`` cells from :func:`sleap_roots.circumnutation.midline.reconstruct`
for two fixed inputs:

1. A pinned closed-form CIRCLE (``R=50``, ``theta=linspace(0, 2π, 128,
   endpoint=False)``) — a self-evident oracle where interior ``κ ≡ 1/R``.
2. The synthetic generator ``generate_trajectory(random_state=0, n_frames=128,
   T_nutation_s=3333, cadence_s=300, noise_sigma_px=0.5)`` — a drift detector.

The captured arrays become the hardcoded ``_MIDLINE_CIRCLE_CANARY`` /
``_MIDLINE_SYNTHETIC_CANARY`` literals in
``tests/test_circumnutation_midline.py``.

**Purpose: REGRESSION DETECTOR for future scipy / numpy / BLAS drift, NOT a
correctness oracle** (the oracle role is the separate ``κ ≈ 1/R`` assertion at a
loose physical tolerance). Bit-identical reproduction across Ubuntu / Windows /
macOS at PR-merge time defines the canary's contract (cross-OS ``atol=1e-9``).
If the canary later fails on a CI runner, FIRST re-run this script on the
failing runner to capture the actual diff; SECOND either pin the offending
dependency OR widen the atol with a reconciliation note.

Invocation::

    uv run python scripts/circumnutation/capture_midline_canary.py
"""

import datetime as _dt
import platform
import sys

import numpy as np
import scipy

from sleap_roots.circumnutation import synthetic
from sleap_roots.circumnutation._constants import _default_constants_snapshot
from sleap_roots.circumnutation.midline import reconstruct


# Canary run parameters — keep in sync with the test setup.
CIRCLE_RADIUS = 50.0
N_FRAMES = 128
CADENCE_S = 300.0
RANDOM_STATE = 0
T_NUTATION_S = 3333
NOISE_SIGMA_PX = 0.5
# Interior frame indices: deliberately well away from the array edges so the
# canary cells come from the Savitzky-Golay INTERIOR (fixed-coefficient FIR
# `correlate1d`) path, NOT the per-edge `mode="interp"` polyfit/lstsq path. This
# is load-bearing for the cross-OS atol=1e-9 floor: the only BLAS/LAPACK-touched
# computation (`savgol_coeffs`) is then data-independent and well-conditioned
# (cond(A)≈11), so the edge lstsq driver variation never enters a canary value.
CANARY_FRAME_INDICES = [20, 64, 100]


def _circle_xy():
    """Return the pinned canary circle (closed loop, endpoint=False)."""
    theta = np.linspace(0.0, 2.0 * np.pi, N_FRAMES, endpoint=False)
    return CIRCLE_RADIUS * np.cos(theta), CIRCLE_RADIUS * np.sin(theta)


def _synthetic_xy():
    """Return the synthetic-generator canary tip coordinates as float64 arrays."""
    df = synthetic.generate_trajectory(
        random_state=RANDOM_STATE,
        n_frames=N_FRAMES,
        T_nutation_s=T_NUTATION_S,
        cadence_s=CADENCE_S,
        noise_sigma_px=NOISE_SIGMA_PX,
    )
    x = df["tip_x"].to_numpy(dtype=np.float64)
    y = df["tip_y"].to_numpy(dtype=np.float64)
    return x, y


def _array_repr(values):
    """Full-precision copy-paste-ready ``np.array([...])`` literal."""
    body = ", ".join(f"{v:.17g}" for v in values)
    return f"np.array([{body}], dtype=np.float64)"


def _capture(label, x, y):
    """Reconstruct and return the canary curvature/arc cells for the given input."""
    result = reconstruct(x, y, cadence_s=CADENCE_S)
    kappa = result.curvature_px_inv[CANARY_FRAME_INDICES]
    arc = result.arc_length_px[CANARY_FRAME_INDICES]
    return label, kappa, arc


def main() -> int:
    """CLI entry point: print provenance + copy-paste-ready canary literals."""
    print("=" * 78)
    print("PR #8 midline canary capture")
    print("=" * 78)
    print(
        f"Capture date (ISO 8601 UTC): {_dt.datetime.now(_dt.timezone.utc).isoformat()}"
    )
    print(f"platform: {platform.system()} {platform.release()} {platform.machine()}")
    print(
        f"python: {sys.version.split()[0]}  numpy: {np.__version__}  scipy: {scipy.__version__}"
    )
    print(f"frame_indices: {CANARY_FRAME_INDICES}")
    print(f"constants_snapshot: {_default_constants_snapshot()}")
    print("-" * 78)

    circle_x, circle_y = _circle_xy()
    synth_x, synth_y = _synthetic_xy()
    for label, x, y in (
        ("circle", circle_x, circle_y),
        ("synthetic", synth_x, synth_y),
    ):
        _, kappa, arc = _capture(label, x, y)
        print(f"# {label} canary — curvature_px_inv at {CANARY_FRAME_INDICES}:")
        print(f"_MIDLINE_{label.upper()}_CANARY_KAPPA = {_array_repr(kappa)}")
        print(f"# {label} canary — arc_length_px at {CANARY_FRAME_INDICES}:")
        print(f"_MIDLINE_{label.upper()}_CANARY_ARC = {_array_repr(arc)}")
        print("-" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
