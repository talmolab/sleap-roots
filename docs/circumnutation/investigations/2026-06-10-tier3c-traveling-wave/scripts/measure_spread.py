"""Pass 2: orientation-invariant lambda(s_a) spread + edge-trimming for D4.

Run: uv run python _scratch/2026-06-10-tier3c-traveling-wave-traits/scripts/measure_spread.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sleap_roots.series import Series
from sleap_roots.circumnutation.midline import reconstruct
from sleap_roots.circumnutation.spatial_cwt import (
    resample_curvature,
    compute_scaleogram,
    extract_ridge,
)

CADENCE_S = 300.0
def _find_repo(start):
    """Walk upward to the repo root (the dir containing pyproject.toml)."""
    p = Path(start).resolve()
    for parent in [p, *p.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root (pyproject.toml) not found")


REPO = _find_repo(__file__)
FIXTURE = (
    REPO
    / "tests/data/circumnutation_nipponbare_plate_001/"
    "plate_001_greyscale.tracked_proofread.slp"
)
CALIB = REPO / "tests/data/circumnutation_spatial_cwt_calibration.json"


def load_calibration():
    data = json.loads(CALIB.read_text())
    entries = data["wavelength_calibration"]
    lam_rep = np.array([e["lambda_reported"] for e in entries], dtype=float)
    ratio = np.array([e["ratio"] for e in entries], dtype=float)
    order = np.argsort(lam_rep)
    return lam_rep[order], ratio[order]


def calibrate(lam, lam_rep, ratio):
    return lam / np.interp(lam, lam_rep, ratio)


def load_tracks():
    series = Series.load(series_name="plate_001", primary_path=str(FIXTURE))
    df = series.get_tracked_tips()
    df["track_id"] = df["track_id"].str.replace("track_", "", regex=False).astype(int)
    return df


def robust_stats(lam):
    med = np.median(lam)
    cv = np.std(lam) / np.mean(lam)
    mad = np.median(np.abs(lam - med)) / med
    iqr = (np.quantile(lam, 0.75) - np.quantile(lam, 0.25)) / med
    return cv, mad, iqr


def window_ratio(s, lam, lo, hi):
    """median(lam in [hi,1]) / median(lam in [0,lo]) using s-quantile bands."""
    s_lo_a, s_hi_a = np.quantile(s, 0.0), np.quantile(s, lo)
    s_lo_b, s_hi_b = np.quantile(s, hi), np.quantile(s, 1.0)
    a = lam[(s >= s_lo_a) & (s <= s_hi_a)]
    b = lam[(s >= s_lo_b) & (s <= s_hi_b)]
    return np.median(b) / np.median(a)


def main():
    lam_rep, ratio = load_calibration()
    raw = load_tracks()
    rows = []
    for track_id in sorted(raw.track_id.unique()):
        sub = raw[raw.track_id == track_id].dropna(subset=["tip_x", "tip_y"])
        sub = sub.sort_values("frame")
        x = sub.tip_x.to_numpy(dtype=np.float64)
        y = sub.tip_y.to_numpy(dtype=np.float64)
        mr = reconstruct(x, y, cadence_s=CADENCE_S)
        rs = resample_curvature(
            mr.curvature_px_inv, mr.arc_length_px, mr.velocity_sub_noise_mask
        )
        sr = compute_scaleogram(rs.kappa_uniform, rs.ds)
        rg = extract_ridge(sr)
        interior = ~rg.in_coi
        s = rs.s_a_uniform_px[interior]
        lam = calibrate(rg.wavelengths_px[interior], lam_rep, ratio)
        order = np.argsort(s)
        s, lam = s[order], lam[order]

        # full COI-valid spread
        cv, mad, iqr = robust_stats(lam)
        # edge-trimmed: drop outer 10% of positions (by sorted index)
        n = len(lam)
        k = int(0.10 * n)
        lam_trim = lam[k : n - k]
        cv_t, mad_t, iqr_t = robust_stats(lam_trim)

        rows.append(
            dict(
                track=track_id,
                n=n,
                # full-span symmetric spreads (orientation-invariant)
                cv=round(cv, 3),
                mad=round(mad, 3),
                iqr=round(iqr, 3),
                # edge-trimmed (drop outer 10%)
                cv_trim=round(cv_t, 3),
                mad_trim=round(mad_t, 3),
                iqr_trim=round(iqr_t, 3),
                # directional window ratios (basal/apex), extreme vs interior bands
                wr_0_25=round(window_ratio(s, lam, 0.25, 0.75), 3),
                wr_10_35=round(window_ratio(s, lam, 0.35, 0.65), 3),
            )
        )
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)
    print("\n=== Orientation-invariant spread + edge-trimming ===")
    print(out.to_string(index=False))
    print("\n=== Summary ===")
    for col in ["cv", "mad", "iqr", "cv_trim", "mad_trim", "iqr_trim"]:
        print(f"{col:9s}: median={out[col].median():.3f}, "
              f"range=[{out[col].min():.3f}, {out[col].max():.3f}]")
    print(f"wr_0_25 (extreme bands): range=[{out.wr_0_25.min():.3f}, {out.wr_0_25.max():.3f}]")
    print(f"wr_10_35 (interior bands): range=[{out.wr_10_35.min():.3f}, {out.wr_10_35.max():.3f}]")


if __name__ == "__main__":
    main()
