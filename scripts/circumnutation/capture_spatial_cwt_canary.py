"""Capture deterministic canary values for the PR #9 spatial-CWT cross-OS test.

Run on the developer's primary machine to capture three complex-valued scaleogram
cells at COI-interior positions for a noise-free, RNG-FREE planted sinusoid κ(s).
The captured values become the hardcoded expected array in
``tests/test_circumnutation_spatial_cwt.py::test_compute_scaleogram_cross_os_canary``.

**Purpose: REGRESSION DETECTOR for future pywt / numpy / BLAS drift, NOT a
correctness oracle.** Bit-identical reproduction across Ubuntu / Windows / macOS at
PR-merge defines the contract. ``cgau2`` is unproven cross-OS in this repo; if the
canary later fails on a CI runner, FIRST re-run this script on the failing runner to
capture the diff, THEN either pin the offending dependency OR widen the canary's
atol (per the spec "MAY be re-captured" clause) — do NOT widen the oracle/shape
tests.

The canary input is an RNG-free planted sinusoid, so (unlike the PR #5/#8 canaries)
NO ``synthetic.py`` git SHA is needed; the provenance header still records platform
+ numpy/pywt/BLAS versions.

Invocation::

    uv run python scripts/circumnutation/capture_spatial_cwt_canary.py
    uv run python scripts/circumnutation/capture_spatial_cwt_canary.py --out path.json
"""

import argparse
import datetime as _dt
import io as _io
import json
import platform
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pywt

from sleap_roots.circumnutation._constants import (
    _default_constants_snapshot,
    SPATIAL_COI_EFOLDING_FACTOR,
)
from sleap_roots.circumnutation.spatial_cwt import (
    compute_scaleogram,
    _coi_boundary_samples,
)


# Canary run parameters — keep in sync with the cross-OS canary test setup.
N_SAMPLES = 256
DS = 5.8
LAMBDA_TRUE = 65.0
TARGET_WAVELENGTH = 65.0


def _blas_info_string() -> str:
    buf = _io.StringIO()
    with redirect_stdout(buf):
        np.show_config()
    return buf.getvalue()


def _provenance() -> dict:
    return {
        "capture_date_iso": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python_version": sys.version.split()[0],
        },
        "library_versions": {"numpy": np.__version__, "pywt": pywt.__version__},
        "blas_info": _blas_info_string(),
        "constants_snapshot": _default_constants_snapshot(),
        "run_parameters": {
            "n_samples": N_SAMPLES,
            "ds": DS,
            "lambda_true": LAMBDA_TRUE,
            "target_wavelength": TARGET_WAVELENGTH,
        },
    }


def capture_canary() -> dict:
    """Run the spatial CWT on the planted sinusoid; return COI-interior cells."""
    s_a = np.arange(N_SAMPLES, dtype=np.float64) * DS
    kappa = np.sin(2.0 * np.pi * s_a / LAMBDA_TRUE)
    result = compute_scaleogram(kappa, DS)

    scale_idx = int(np.argmin(np.abs(result.wavelengths_px - TARGET_WAVELENGTH)))
    boundary = _coi_boundary_samples(
        float(result.scales[scale_idx]), SPATIAL_COI_EFOLDING_FACTOR
    )
    positions = [boundary + 2, N_SAMPLES // 2, N_SAMPLES - boundary - 2]
    assert all(
        boundary <= p < N_SAMPLES - boundary for p in positions
    ), f"positions not COI-interior: boundary={boundary}, positions={positions}"
    assert not result.coi_mask[scale_idx, positions].any(), (
        f"positions intersect COI at the target scale: "
        f"{result.coi_mask[scale_idx, positions]}"
    )

    values = result.scaleogram[scale_idx, positions]
    values_repr = "np.array([\n"
    for v in values:
        values_repr += f"        complex({v.real:.17g}, {v.imag:.17g}),\n"
    values_repr += "    ], dtype=np.complex128)"

    return {
        "provenance": _provenance(),
        "scale_idx_at_target": scale_idx,
        "boundary_samples": boundary,
        "positions": positions,
        "values_repr": values_repr,
        "values_list": [[float(v.real), float(v.imag)] for v in values],
    }


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Capture PR #9 spatial-CWT cross-OS canary values."
    )
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON path.")
    args = parser.parse_args()

    captured = capture_canary()
    prov = captured["provenance"]
    print("=" * 78)
    print("PR #9 spatial_cwt canary capture")
    print("=" * 78)
    print(f"Capture date (ISO 8601 UTC): {prov['capture_date_iso']}")
    print(
        f"Platform: {prov['platform']['system']} {prov['platform']['release']} "
        f"({prov['platform']['machine']}); Python {prov['platform']['python_version']}"
    )
    print(
        f"numpy: {prov['library_versions']['numpy']}  pywt: {prov['library_versions']['pywt']}"
    )
    print(f"Run parameters: {prov['run_parameters']}")
    print()
    print(f"scale_idx_at_target = {captured['scale_idx_at_target']}")
    print(f"boundary_samples = {captured['boundary_samples']}")
    print(f"positions = {tuple(captured['positions'])}")
    print()
    print("Captured complex values (paste into the cross-OS canary test):")
    print(captured["values_repr"])

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as f:
            json.dump(captured, f, indent=2, default=str)
        print(f"\nJSON companion written to: {args.out.as_posix()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
