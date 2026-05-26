"""Capture deterministic canary values for the PR #5 §2.B.3 cross-OS canary test.

Run this script on the developer's primary machine to capture three
complex-valued scaleogram cells at the resonant scale (period ≈ 3333 s) for
a noise-free synthetic trajectory. The captured values become the hardcoded
expected array in ``tests/test_circumnutation_temporal_cwt.py::test_2B3_cross_os_canary_at_atol_1e_9``.

**Purpose: REGRESSION DETECTOR for future pywt / numpy / BLAS drift, NOT a
correctness oracle.** Bit-identical reproduction across Ubuntu / Windows /
macOS at the time of PR merge defines the canary's contract per the spec
scenario "compute_scaleogram is deterministic across runs". If the canary
later fails on a CI runner, FIRST verify by re-running this script on the
failing runner via a debugging branch to capture the actual diff; SECOND
either pin the offending dependency version OR widen the atol with a
Reconciliation Appendix entry per design.md R1.

Invocation::

    uv run python scripts/circumnutation/capture_temporal_cwt_canary.py

Output is human-readable on stdout: a provenance header followed by the
three captured complex values in copy-paste-ready Python literal format.
Use ``--out path.json`` to additionally write a machine-readable JSON
companion for downstream automation (the JSON header is the same as stdout).

See ``openspec/changes/add-circumnutation-temporal-cwt-machinery/tasks.md``
§3.5 for the integration recipe, and ``design.md`` D5 for the canary
contract.
"""

import argparse
import datetime as _dt
import io as _io
import json
import platform
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pywt

from sleap_roots.circumnutation import synthetic
from sleap_roots.circumnutation.temporal_cwt import (
    compute_scaleogram,
    _coi_boundary_samples,  # private helper imported intentionally for COI-interior derivation
)
from sleap_roots.circumnutation._constants import (
    _default_constants_snapshot,
    COI_EFOLDING_FACTOR,
)


# Canary run parameters — keep in sync with the §2.B.3 test setup.
RANDOM_STATE = 0
N_FRAMES = 128
T_NUTATION_S = 3333
CADENCE_S = 300
NOISE_SIGMA_PX = 0


def _blas_info_string() -> str:
    """Return a short BLAS/LAPACK info string for the provenance header.

    ``numpy.show_config()`` prints to stdout; capture via redirect_stdout.
    """
    buf = _io.StringIO()
    with redirect_stdout(buf):
        np.show_config()
    return buf.getvalue()


def _synthetic_git_sha() -> str:
    """Return the git commit SHA of ``synthetic.py`` for provenance.

    Falls back to ``"unknown"`` if git is not available or the file is
    outside a repo.
    """
    try:
        repo_root = Path(__file__).resolve().parents[2]
        synthetic_path = repo_root / "sleap_roots" / "circumnutation" / "synthetic.py"
        result = subprocess.run(
            ["git", "log", "-1", "--format=%H", "--", str(synthetic_path)],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() or "unknown"
    except Exception:  # pragma: no cover - best-effort provenance only
        return "unknown"


def _make_provenance_header() -> dict:
    """Build the provenance header dict for both stdout + JSON output."""
    return {
        "capture_date_iso": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python_version": sys.version.split()[0],
        },
        "library_versions": {
            "numpy": np.__version__,
            "pywt": pywt.__version__,
        },
        "blas_info": _blas_info_string(),
        "constants_snapshot": _default_constants_snapshot(),
        "run_parameters": {
            "random_state": RANDOM_STATE,
            "n_frames": N_FRAMES,
            "T_nutation_s": T_NUTATION_S,
            "cadence_s": CADENCE_S,
            "noise_sigma_px": NOISE_SIGMA_PX,
        },
        "synthetic_py_git_sha": _synthetic_git_sha(),
    }


def capture_canary() -> dict:
    """Generate the canary input, run the CWT, and return the captured values.

    Returns a dict with keys: ``provenance``, ``scale_idx_at_target``,
    ``frame_indices``, ``values_repr`` (full-precision Python literal),
    ``values_list`` (list of [real, imag] pairs for JSON).
    """
    df = synthetic.generate_trajectory(
        random_state=RANDOM_STATE,
        n_frames=N_FRAMES,
        T_nutation_s=T_NUTATION_S,
        cadence_s=CADENCE_S,
        noise_sigma_px=NOISE_SIGMA_PX,
    )
    x = df["tip_x"].to_numpy()
    result = compute_scaleogram(x, float(CADENCE_S))

    scale_idx_at_target = int(np.argmin(np.abs(result.periods_s - float(T_NUTATION_S))))
    # COI-interior frame indices: 2 frames inside the COI band on each edge
    # plus the geometric middle. Verify each is COI-interior at runtime.
    boundary = _coi_boundary_samples(
        float(result.scales[scale_idx_at_target]), COI_EFOLDING_FACTOR
    )
    frame_indices = [boundary + 2, N_FRAMES // 2, N_FRAMES - boundary - 2]
    # Sanity-check the indices fall in the COI-interior strip.
    assert all(boundary <= idx < N_FRAMES - boundary for idx in frame_indices), (
        f"Canary frame indices not COI-interior: boundary={boundary}, "
        f"indices={frame_indices}"
    )
    assert not result.coi_mask[scale_idx_at_target, frame_indices].any(), (
        f"Canary frame indices intersect the COI band at the target scale: "
        f"in_coi={result.coi_mask[scale_idx_at_target, frame_indices]}"
    )

    values = result.scaleogram[scale_idx_at_target, frame_indices]
    # Format with full float64 precision via repr; ".17g" gives ≥ float64 roundtrip.
    values_repr = "np.array([\n"
    for v in values:
        values_repr += f"    complex({v.real:.17g}, {v.imag:.17g}),\n"
    values_repr += "], dtype=np.complex128)"

    return {
        "provenance": _make_provenance_header(),
        "scale_idx_at_target": scale_idx_at_target,
        "boundary_samples_at_target_scale": boundary,
        "frame_indices": frame_indices,
        "values_repr": values_repr,
        "values_list": [[float(v.real), float(v.imag)] for v in values],
    }


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Capture canary values for the PR #5 §2.B.3 cross-OS deterministic "
            "scaleogram test."
        )
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Optional path to a JSON file where provenance + captured values "
            "will be written (machine-readable companion to stdout)."
        ),
    )
    args = parser.parse_args()

    captured = capture_canary()

    # Pretty-print provenance header to stdout.
    print("=" * 78)
    print("PR #5 temporal_cwt canary capture")
    print("=" * 78)
    prov = captured["provenance"]
    print(f"Capture date (ISO 8601 UTC): {prov['capture_date_iso']}")
    print(f"Platform: {prov['platform']['system']} {prov['platform']['release']} "
          f"({prov['platform']['machine']})")
    print(f"Python: {prov['platform']['python_version']}")
    print(f"numpy: {prov['library_versions']['numpy']}")
    print(f"pywt:  {prov['library_versions']['pywt']}")
    print(f"synthetic.py git SHA: {prov['synthetic_py_git_sha']}")
    print(f"Run parameters: {prov['run_parameters']}")
    print()
    print(f"scale_idx_at_target = {captured['scale_idx_at_target']}")
    print(f"boundary_samples_at_target_scale = {captured['boundary_samples_at_target_scale']}")
    print(f"frame_indices = {tuple(captured['frame_indices'])}")
    print()
    print("Captured complex values (paste into §2.B.3 _CANARY_EXPECTED_VALUES):")
    print(captured["values_repr"])
    print()
    print("BLAS info (numpy.show_config() output):")
    print(prov["blas_info"])

    if args.out is not None:
        # Write JSON companion. blas_info is multi-line — keep it as a string.
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as f:
            json.dump(captured, f, indent=2, default=str)
        print(f"JSON companion written to: {args.out.as_posix()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
