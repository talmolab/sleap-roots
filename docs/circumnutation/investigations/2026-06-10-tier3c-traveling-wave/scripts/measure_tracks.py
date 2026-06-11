"""Measure Tier 3c traveling-wave operands on the 6 plate-001 proofread tracks.

Run: uv run python _scratch/2026-06-10-tier3c-traveling-wave-traits/scripts/measure_tracks.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sleap_roots.series import Series
from sleap_roots.circumnutation import kinematics, nutation
from sleap_roots.circumnutation.midline import reconstruct
from sleap_roots.circumnutation.spatial_cwt import (
    resample_curvature,
    compute_scaleogram,
    extract_ridge,
)

CADENCE_S = 300.0
APEX_BASAL_FRACTION = 0.25  # window = lowest/highest 25% of s_a among COI-valid

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
    """Build a ratio(lambda_reported) interpolator from the calibration table.

    The table is keyed by (n, ds, lambda_true) with ratio = lambda_reported/lambda_true.
    A consumer only knows lambda_reported, so we interpolate ratio as a function of
    lambda_reported = lambda_true * ratio, then divide observed lambda by that ratio.
    """
    data = json.loads(CALIB.read_text())
    entries = data["wavelength_calibration"]
    lam_rep = np.array([e["lambda_reported"] for e in entries], dtype=float)
    ratio = np.array([e["ratio"] for e in entries], dtype=float)
    order = np.argsort(lam_rep)
    lam_rep, ratio = lam_rep[order], ratio[order]
    print(
        f"calibration: {len(entries)} entries, "
        f"lambda_reported in [{lam_rep.min():.1f}, {lam_rep.max():.1f}], "
        f"ratio in [{ratio.min():.4f}, {ratio.max():.4f}]"
    )
    return lam_rep, ratio


def calibrate(lam_observed, lam_rep, ratio):
    """Return calibrated wavelength(s): lambda_true_est = lambda_obs / ratio(lambda_obs)."""
    r = np.interp(lam_observed, lam_rep, ratio)  # clamps at table ends
    return lam_observed / r


def load_tracks():
    series = Series.load(series_name="plate_001", primary_path=str(FIXTURE))
    df = series.get_tracked_tips()
    df["track_id"] = df["track_id"].str.replace("track_", "", regex=False).astype(int)
    return df


def build_trajectory_df(df):
    """Add the 8 row-identity columns; plant_id == track_id (one track per plant)."""
    df = df.copy()
    df["series"] = "plate_001"
    df["sample_uid"] = "plate_001"
    df["timepoint"] = "T0"
    df["plate_id"] = "plate_001"
    df["plant_id"] = df["track_id"]
    df["genotype"] = "Nipponbare"
    df["treatment"] = "none"
    return df


def main():
    lam_rep, ratio = load_calibration()
    raw = load_tracks()
    traj = build_trajectory_df(raw)

    # Tier 0 + Tier 1 (recompute-internally, as Tier 3c will)
    k_df = kinematics.compute(traj)
    n_df = nutation.compute(traj, CADENCE_S, coordinate="lateral")
    k_by_track = k_df.set_index("track_id")
    n_by_track = n_df.set_index("track_id")

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
        n_int = int(interior.sum())
        s_a = rs.s_a_uniform_px  # apex-origin arc length, s_a[0] == 0
        lam_obs = rg.wavelengths_px
        lam_cal = calibrate(lam_obs, lam_rep, ratio)

        lam_med_raw = float(np.median(lam_obs[interior]))
        lam_med_cal = float(np.median(lam_cal[interior]))
        coi_frac = float(n_int / lam_obs.size)

        # apex (s_a small) vs basal (s_a large) windows among COI-valid positions
        s_int = s_a[interior]
        lam_int_raw = lam_obs[interior]
        lam_int_cal = lam_cal[interior]
        order = np.argsort(s_int)
        s_sorted = s_int[order]
        thr_lo = np.quantile(s_sorted, APEX_BASAL_FRACTION)
        thr_hi = np.quantile(s_sorted, 1 - APEX_BASAL_FRACTION)
        apex_mask = s_int <= thr_lo
        basal_mask = s_int >= thr_hi
        apex_raw = float(np.median(lam_int_raw[apex_mask]))
        basal_raw = float(np.median(lam_int_raw[basal_mask]))
        apex_cal = float(np.median(lam_int_cal[apex_mask]))
        basal_cal = float(np.median(lam_int_cal[basal_mask]))

        v = float(k_by_track.loc[track_id, "v_total_median_px_per_frame"])
        T_s = float(n_by_track.loc[track_id, "T_nutation_median"])
        is_nut = bool(n_by_track.loc[track_id, "is_nutating"])
        T_frames = T_s / CADENCE_S if np.isfinite(T_s) else np.nan
        lam_expected = v * T_frames

        res_raw = abs(lam_med_raw - lam_expected) / lam_expected
        res_cal = abs(lam_med_cal - lam_expected) / lam_expected

        rows.append(
            dict(
                track=track_id,
                n_int=n_int,
                coi_frac=round(coi_frac, 3),
                v=round(v, 3),
                T_s=round(T_s, 1) if np.isfinite(T_s) else np.nan,
                T_fr=round(T_frames, 2) if np.isfinite(T_frames) else np.nan,
                is_nut=is_nut,
                lam_exp=round(lam_expected, 1) if np.isfinite(lam_expected) else np.nan,
                lam_raw=round(lam_med_raw, 1),
                lam_cal=round(lam_med_cal, 1),
                res_raw=round(res_raw, 3) if np.isfinite(res_raw) else np.nan,
                res_cal=round(res_cal, 3) if np.isfinite(res_cal) else np.nan,
                apex_raw=round(apex_raw, 1),
                basal_raw=round(basal_raw, 1),
                ab_ratio_raw=round(basal_raw / apex_raw, 3),
                apex_cal=round(apex_cal, 1),
                basal_cal=round(basal_cal, 1),
                ab_ratio_cal=round(basal_cal / apex_cal, 3),
            )
        )

    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)
    print("\n=== Per-track operands ===")
    print(out.to_string(index=False))

    print("\n=== Summary ===")
    print(f"calibration shifts lambda_median by factor "
          f"{(out.lam_cal / out.lam_raw).mean():.4f} (mean)")
    print(f"traveling_wave_residual raw : median={out.res_raw.median():.3f}, "
          f"range=[{out.res_raw.min():.3f}, {out.res_raw.max():.3f}]")
    print(f"traveling_wave_residual cal : median={out.res_cal.median():.3f}, "
          f"range=[{out.res_cal.min():.3f}, {out.res_cal.max():.3f}]")
    print(f"apex/basal ratio (cal)      : median={out.ab_ratio_cal.median():.3f}, "
          f"range=[{out.ab_ratio_cal.min():.3f}, {out.ab_ratio_cal.max():.3f}]")
    print(f"is_nutating                 : {out.is_nut.sum()}/{len(out)} tracks")

    out.to_csv(Path(__file__).parent.parent / "operands.csv", index=False)
    print("\nwrote operands.csv")


if __name__ == "__main__":
    main()
