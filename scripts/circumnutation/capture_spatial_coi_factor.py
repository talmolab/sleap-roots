"""Measure cgau2 calibration constants for PR #9 (Tier 3b spatial CWT).

Two measurements, both wavelet-specific to ``cgau2`` (the spatial mother
wavelet, :data:`sleap_roots.circumnutation._constants.WAVELET_DEFAULT_SPATIAL`):

1. **COI e-folding factor** (``SPATIAL_COI_EFOLDING_FACTOR``): the half-width of
   the cone of influence in samples per unit scale, measured by the **impulse
   1/e half-width** method (NOT a literal step — a step is the wrong stimulus for
   an e-folding decay). For ``cmor1.5-1.0`` this factor is ``√1.5 ≈ 1.225`` (the
   temporal ``COI_EFOLDING_FACTOR``); ``cgau2`` differs (its envelope is a
   2nd-derivative Gaussian, not a plain Gaussian) and is measured here.

2. **Wavelength calibration map**: the ``pywt.scale2frequency("cgau2", …)``
   convention over-reports the true spatial wavelength of a planted sinusoid by a
   λ- AND n-dependent band (driven by the discrete log-scale grid + the cgau2
   center-frequency convention). This is NOT a single calibratable constant, so we
   emit a machine-readable ``{n, scale_count, lambda_true, lambda_reported,
   ratio}`` map. The map (a) feeds the λ-recovery oracle tolerances in
   ``tests/test_circumnutation_spatial_cwt.py`` (so they are reproducible, not
   hand-typed) and (b) is the artifact PR #10 consumes to reconcile the offset for
   ``traveling_wave_residual`` / ``lambda_spatial_median`` (see the PR #9
   ``design.md`` "Handoff to PR #10").

Invocation::

    uv run python scripts/circumnutation/capture_spatial_coi_factor.py
    uv run python scripts/circumnutation/capture_spatial_coi_factor.py --out tests/data/circumnutation_spatial_cwt_calibration.json

Output is human-readable on stdout (provenance header + the measured factor +
the calibration table); ``--out`` additionally writes the machine-readable JSON
artifact. The script REPLICATES the planned ``compute_scaleogram`` scale-axis
math (PR #9 spec) because it runs before the implementation lands; keep the two
in sync.
"""

import argparse
import datetime as _dt
import io as _io
import json
import math
import platform
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pywt


WAVELET = "cgau2"
SCALE_COUNT = 64  # CWT_SCALE_COUNT_DEFAULT
WL_MIN_FACTOR = 2.0  # CWT_WAVELENGTH_MIN_NYQUIST_FACTOR
WL_MAX_FRACTION = 0.25  # CWT_WAVELENGTH_MAX_SIGNAL_FRACTION

# Signal lengths and the PR #9 base wavelength grid.
_BASE_NS = (200, 400, 600)
_BASE_LAMBDAS = (20.0, 30.0, 40.0, 50.0, 60.0, 80.0)
# PR #10 append-only extension: new lambda_true knots (measured for ALL _BASE_NS)
# so the consumer's n-average is defined up to lambda_reported ~= 167 px, covering
# the observed real plate-001 lambda ~= 142.5 px without clamped extrapolation.
_EXTENSION_LAMBDAS = (100.0, 120.0, 140.0, 150.0)


def _scale_axis(n: int, ds: float) -> np.ndarray:
    """Replicate the planned spatial scale axis (PR #9 spec)."""
    center_freq = float(pywt.scale2frequency(WAVELET, 1.0))
    wl_min_samples = WL_MIN_FACTOR
    wl_max_samples = WL_MAX_FRACTION * n
    scale_min = wl_min_samples * center_freq
    scale_max = wl_max_samples * center_freq
    return np.logspace(
        math.log10(scale_min), math.log10(scale_max), SCALE_COUNT
    ).astype(np.float64)


def _wavelengths_px(scales: np.ndarray, ds: float) -> np.ndarray:
    """Convention wavelength axis: ds / scale2frequency (px)."""
    freqs = np.asarray(pywt.scale2frequency(WAVELET, scales), dtype=np.float64)
    return ds / freqs


def measure_coi_factor() -> dict:
    """Impulse 1/e half-width COI factor for cgau2, averaged across scales."""
    n = 4001
    impulse = np.zeros(n, dtype=np.float64)
    impulse[n // 2] = 1.0
    scales = np.array([8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0, 96.0, 128.0])
    coefs, _ = pywt.cwt(impulse, scales, WAVELET)
    power = np.abs(coefs)  # (n_scales, n)
    factors = []
    for i, s in enumerate(scales):
        row = power[i]
        peak = row[n // 2]
        if peak <= 0:
            continue
        # Walk right from the center until |W| drops below peak/e.
        thresh = peak / math.e
        half = 0
        for j in range(n // 2, n):
            if row[j] < thresh:
                half = j - n // 2
                break
        if half > 0:
            factors.append(half / s)
    factors = np.asarray(factors, dtype=np.float64)
    return {
        "scales": scales.tolist(),
        "per_scale_factor": factors.tolist(),
        "median": float(np.median(factors)),
        "mean": float(np.mean(factors)),
        "std": float(np.std(factors)),
    }


def measure_wavelength_calibration(lambdas=_BASE_LAMBDAS, ns=_BASE_NS) -> list:
    """Per-(n, lambda_true) reported-vs-true wavelength via the ridge median.

    Args:
        lambdas: The ``lambda_true`` knots to measure (px).
        ns: The signal lengths to measure each knot at.

    Returns:
        A list of ``{n, scale_count, ds, lambda_true, lambda_reported, ratio}``
        rows, one per ``(n, lambda_true)`` pair. Each row is independent (no
        cross-row state), so measuring only new knots reproduces the same values
        for the existing knots in a fixed environment.
    """
    rows = []
    for n in ns:
        ds = 5.8
        s_a = np.arange(n, dtype=np.float64) * ds
        scales = _scale_axis(n, ds)
        wavelengths = _wavelengths_px(scales, ds)
        # crude COI mask (factor ~1.34) to take an interior median
        coi_factor = 1.34
        for lam in lambdas:
            kappa = np.sin(2.0 * np.pi * s_a / lam)
            coefs, _ = pywt.cwt(kappa, scales, WAVELET)
            mag = np.abs(coefs)  # (n_scales, n)
            ridge_scale_idx = np.argmax(mag, axis=0)
            ridge_lambda = wavelengths[ridge_scale_idx]
            # interior positions: outside ceil(coi_factor*max_scale) on each edge
            boundary = int(math.ceil(coi_factor * scales[-1]))
            interior = slice(boundary, n - boundary)
            if n - 2 * boundary < 5:
                interior = slice(n // 4, 3 * n // 4)
            reported = float(np.median(ridge_lambda[interior]))
            rows.append(
                {
                    "n": n,
                    "scale_count": SCALE_COUNT,
                    "ds": ds,
                    "lambda_true": lam,
                    "lambda_reported": reported,
                    "ratio": reported / lam,
                }
            )
    return rows


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
        "wavelet": WAVELET,
        "scale_count": SCALE_COUNT,
        "wl_min_factor": WL_MIN_FACTOR,
        "wl_max_fraction": WL_MAX_FRACTION,
    }


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Measure cgau2 COI e-folding factor + wavelength calibration."
    )
    parser.add_argument("--out", type=Path, default=None, help="JSON artifact path.")
    parser.add_argument(
        "--append",
        action="store_true",
        help=(
            "PR #10 append-only mode: load the existing --out JSON, copy its "
            "provenance + existing wavelength_calibration rows through VERBATIM "
            "(no re-measure), measure ONLY the new _EXTENSION_LAMBDAS knots for "
            "all _BASE_NS, and append them at the END. Makes the existing-rows "
            "freeze mechanical (environment-independent)."
        ),
    )
    args = parser.parse_args()

    if args.append:
        if args.out is None or not args.out.exists():
            print(
                "--append requires an existing --out JSON to extend.", file=sys.stderr
            )
            return 2
        with args.out.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        existing = payload["wavelength_calibration"]
        existing_keys = {(r["n"], r["lambda_true"]) for r in existing}
        new_rows = [
            r
            for r in measure_wavelength_calibration(lambdas=_EXTENSION_LAMBDAS)
            if (r["n"], r["lambda_true"]) not in existing_keys
        ]
        # Append at the END; existing rows + provenance are untouched (verbatim).
        payload["wavelength_calibration"] = existing + new_rows
        coi = payload["coi_efolding_factor"]
        calib = payload["wavelength_calibration"]
        print("=" * 78)
        print(
            f"PR #10 append-only extension: +{len(new_rows)} rows (knots "
            f"{_EXTENSION_LAMBDAS} x n={_BASE_NS}); existing "
            f"{len(existing)} rows + provenance preserved verbatim."
        )
        print("=" * 78)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"JSON artifact updated: {args.out.as_posix()}")
        return 0

    coi = measure_coi_factor()
    calib = measure_wavelength_calibration()
    payload = {
        "provenance": _provenance(),
        "coi_efolding_factor": coi,
        "wavelength_calibration": calib,
    }

    print("=" * 78)
    print("PR #9 spatial_cwt cgau2 calibration capture")
    print("=" * 78)
    prov = payload["provenance"]
    print(f"Capture date (ISO 8601 UTC): {prov['capture_date_iso']}")
    print(
        f"Platform: {prov['platform']['system']} {prov['platform']['release']} "
        f"({prov['platform']['machine']}); Python {prov['platform']['python_version']}"
    )
    print(
        f"numpy: {prov['library_versions']['numpy']}  pywt: {prov['library_versions']['pywt']}"
    )
    print()
    print("COI e-folding factor (impulse 1/e half-width / scale):")
    print(f"  per-scale: {[round(f, 3) for f in coi['per_scale_factor']]}")
    print(f"  median={coi['median']:.4f}  mean={coi['mean']:.4f}  std={coi['std']:.4f}")
    print(f"  -> SPATIAL_COI_EFOLDING_FACTOR default ~= {coi['median']:.3f}")
    print()
    print("Wavelength calibration (scale2frequency convention vs true):")
    print(f"  {'n':>5}{'lambda_true':>13}{'reported':>11}{'ratio':>8}")
    for r in calib:
        print(
            f"  {r['n']:>5}{r['lambda_true']:>13.1f}{r['lambda_reported']:>11.2f}"
            f"{r['ratio']:>8.3f}"
        )
    ratios = [r["ratio"] for r in calib]
    print(f"  ratio band: [{min(ratios):.3f}, {max(ratios):.3f}]")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"\nJSON artifact written to: {args.out.as_posix()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
