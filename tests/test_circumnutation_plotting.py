"""Tests for ``sleap_roots.circumnutation.plotting`` (PR #16 diagnostic plots).

Strategy (design D2): force the headless ``Agg`` backend, split each renderer
into a pure ``_build_*_figure`` (structural assertions) plus the public function
(smoke assertions: returns ``out_path``, PNG exists + non-empty, no exception,
figure closed). No pixel baselines — plots are not bit-reproducible across the
tri-OS CI matrix. Result fixtures are built via the real tier functions so they
never drift from the source dataclasses.
"""

import inspect
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection, QuadMesh  # noqa: E402

from sleap_roots.circumnutation import midline, spatial_cwt, temporal_cwt  # noqa: E402
from sleap_roots.circumnutation import plotting  # noqa: E402
from sleap_roots.circumnutation._types import CircumnutationInputs  # noqa: E402


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _agg_clean_figures():
    """Force Agg and close all figures at BOTH setup and teardown.

    The setup close establishes a clean process-global baseline so a figure
    leaked by a prior module in the same pytest-xdist worker cannot cascade into
    this module's ``plt.get_fignums()`` leak checks.
    """
    matplotlib.use("Agg", force=True)
    plt.close("all")
    yield
    plt.close("all")


def _curved_xy(n=40, seed_phase=0.0):
    """A drifting, laterally-oscillating tip path (non-degenerate)."""
    t = np.linspace(0.0, 1.0, n)
    y = 200.0 * t  # net growth-axis drift along +y
    x = 50.0 + 8.0 * np.sin(2.0 * np.pi * (y / 50.0) + seed_phase)
    return x.astype(np.float64), y.astype(np.float64)


@pytest.fixture
def temporal_result():
    """A real temporal ``(ScaleogramResult, smoothed RidgeResult)`` pair."""
    sig = 8.0 * np.sin(2.0 * np.pi * np.arange(64) / 16.0)
    sg = temporal_cwt.compute_scaleogram(sig, 300.0)
    ridge = temporal_cwt.smooth_ridge(temporal_cwt.extract_ridge(sg))
    return sg, ridge


@pytest.fixture
def spatial_result():
    """A real ``(SpatialScaleogramResult, SpatialRidgeResult)`` pair."""
    kappa = 0.01 * np.sin(2.0 * np.pi * np.arange(64) / 16.0)
    sg = spatial_cwt.compute_scaleogram(kappa, 5.8)
    ridge = spatial_cwt.extract_ridge(sg)
    return sg, ridge


@pytest.fixture
def midline_result():
    """A real, non-degenerate ``MidlineResult``."""
    x, y = _curved_xy()
    return midline.reconstruct(x, y, 300.0)


def _track_rows(track, n=40, phase=0.0):
    x, y = _curved_xy(n, phase)
    rows = []
    for frame in range(n):
        rows.append(
            {
                "series": "plate_test",
                "sample_uid": "plate_test",
                "timepoint": "T0",
                "plate_id": "plate_test",
                "plant_id": track,
                "track_id": track,
                "genotype": "g",
                "treatment": "none",
                "frame": frame,
                "tip_x": float(x[frame]),
                "tip_y": float(y[frame]),
            }
        )
    return rows


@pytest.fixture
def inputs_3plants():
    rows = []
    for tr in range(3):
        rows.extend(_track_rows(tr, phase=0.5 * tr))
    return CircumnutationInputs(trajectory_df=pd.DataFrame(rows), cadence_s=300.0)


# --------------------------------------------------------------------------- #
# scaleogram — structural
# --------------------------------------------------------------------------- #
def test_build_scaleogram_temporal_structure(temporal_result):
    sg, ridge = temporal_result
    fig = plotting._build_scaleogram_figure(sg, ridge_result=ridge)
    try:
        ax = fig.axes[0]
        assert ax.get_xlabel() == "time [s]"
        assert ax.get_ylabel() == "period [s]"
        assert ax.get_yscale() == "log"
        meshes = [c for c in ax.collections if isinstance(c, QuadMesh)]
        assert len(meshes) >= 2  # power heatmap + COI dimming overlay
        # colorbar label present
        assert any("power" in (a.get_ylabel() + a.get_xlabel()) for a in fig.axes)
    finally:
        plt.close(fig)


def test_build_scaleogram_spatial_axes(spatial_result):
    sg, ridge = spatial_result
    fig = plotting._build_scaleogram_figure(sg, ridge_result=ridge)
    try:
        ax = fig.axes[0]
        assert ax.get_xlabel() == "arc length [px]"
        assert ax.get_ylabel() == "wavelength [px]"
        assert "uncalibrated" in ax.get_title(loc="right").lower()
    finally:
        plt.close(fig)


def test_scaleogram_quadmesh_uses_edges(temporal_result):
    sg, _ = temporal_result
    n_scales, n_frames = sg.scaleogram.shape
    fig = plotting._build_scaleogram_figure(sg)
    try:
        mesh = [c for c in fig.axes[0].collections if isinstance(c, QuadMesh)][0]
        coords = mesh.get_coordinates()
        assert coords.shape == (n_scales + 1, n_frames + 1, 2)
    finally:
        plt.close(fig)


def test_scaleogram_lognorm_tolerates_zero_power(temporal_result):
    sg, _ = temporal_result
    # Force exact-zero power cells; the LogNorm floor must not raise.
    z = np.array(sg.scaleogram, dtype=np.complex128)
    z[0, 0] = 0.0
    sg2 = type(sg)(
        scaleogram=z,
        scales=sg.scales,
        periods_s=sg.periods_s,
        frequencies_hz=sg.frequencies_hz,
        coi_mask=sg.coi_mask,
        cadence_s=sg.cadence_s,
        wavelet=sg.wavelet,
    )
    fig = plotting._build_scaleogram_figure(sg2)
    plt.close(fig)


def test_scaleogram_rejects_unsupported_type(tmp_path):
    with pytest.raises(TypeError):
        plotting.scaleogram(object(), tmp_path / "x.png")


def test_scaleogram_rejects_mismatched_ridge(temporal_result, spatial_result, tmp_path):
    sg_t, _ = temporal_result
    _, ridge_s = spatial_result
    with pytest.raises(TypeError):
        plotting.scaleogram(sg_t, tmp_path / "x.png", ridge_result=ridge_s)


def test_scaleogram_rejects_junk_ridge(temporal_result, tmp_path):
    sg_t, _ = temporal_result
    with pytest.raises(TypeError):
        plotting.scaleogram(sg_t, tmp_path / "x.png", ridge_result=object())


# --------------------------------------------------------------------------- #
# scaleogram — smoke
# --------------------------------------------------------------------------- #
def test_scaleogram_smoke(temporal_result, tmp_path):
    sg, ridge = temporal_result
    out = tmp_path / "s.png"
    result = plotting.scaleogram(sg, out, ridge_result=ridge)
    assert result == out
    assert out.exists() and out.stat().st_size > 0
    assert plt.get_fignums() == []


# --------------------------------------------------------------------------- #
# trail_overlay
# --------------------------------------------------------------------------- #
def test_build_trail_structure(midline_result):
    n = midline_result.x_smooth_px.shape[0]
    fig = plotting._build_trail_figure(midline_result)
    try:
        ax = fig.axes[0]
        lcs = [c for c in ax.collections if isinstance(c, LineCollection)]
        assert len(lcs) == 1
        assert lcs[0].get_array().shape[0] == n - 1  # N-1 segments
        # image y-down: ylim descending
        ylo, yhi = ax.get_ylim()
        assert ylo > yhi
        assert ax.get_xlabel() == "x [px]"
    finally:
        plt.close(fig)


def test_trail_does_not_mutate_global_colormap(midline_result, tmp_path):
    before = plt.get_cmap(plotting._CMAP_KAPPA).get_bad().copy()
    plotting.trail_overlay(midline_result, tmp_path / "t.png")
    after = plt.get_cmap(plotting._CMAP_KAPPA).get_bad()
    assert np.allclose(before, after)


def test_trail_handles_all_nan_curvature(tmp_path):
    x, y = _curved_xy()
    mr = midline.reconstruct(x, y, 300.0)
    nan_kappa = np.full_like(mr.curvature_px_inv, np.nan)
    mr2 = type(mr)(
        frame_indices=mr.frame_indices,
        x_smooth_px=mr.x_smooth_px,
        y_smooth_px=mr.y_smooth_px,
        speed_px_per_frame=mr.speed_px_per_frame,
        arc_length_px=mr.arc_length_px,
        curvature_px_inv=nan_kappa,
        velocity_sub_noise_mask=mr.velocity_sub_noise_mask,
        cadence_s=mr.cadence_s,
        sg_window=mr.sg_window,
        sg_degree=mr.sg_degree,
        sigma_v_px_per_frame=mr.sigma_v_px_per_frame,
        noise_mask_k=mr.noise_mask_k,
        is_degenerate=mr.is_degenerate,
    )
    out = plotting.trail_overlay(mr2, tmp_path / "nan.png")
    assert out.exists() and out.stat().st_size > 0


def test_trail_colorbar_extends_both(midline_result):
    fig = plotting._build_trail_figure(midline_result)
    try:
        # A colorbar axes is present (main axes + colorbar axes).
        cbar_axes = [ax for ax in fig.axes if ax.get_label() == "<colorbar>"]
        assert len(cbar_axes) == 1
    finally:
        plt.close(fig)


def test_trail_smoke(midline_result, tmp_path):
    out = plotting.trail_overlay(midline_result, tmp_path / "t.png")
    assert out.exists() and out.stat().st_size > 0
    assert plt.get_fignums() == []


# --------------------------------------------------------------------------- #
# plate_panel
# --------------------------------------------------------------------------- #
def test_build_panel_shared_norm_one_colorbar(midline_result):
    fig = plotting._build_panel_figure([midline_result, midline_result])
    try:
        axes_with_lc = [
            ax
            for ax in fig.axes
            if ax.get_label() != "<colorbar>"
            and any(isinstance(c, LineCollection) for c in ax.collections)
        ]
        assert len(axes_with_lc) == 2
        norms = [
            c.norm
            for ax in axes_with_lc
            for c in ax.collections
            if isinstance(c, LineCollection)
        ]
        assert norms[0].vmin == norms[1].vmin and norms[0].vmax == norms[1].vmax
        # Hidden cells when < 6 plants.
        hidden = [ax for ax in fig.axes if not ax.get_visible()]
        assert len(hidden) >= 1
    finally:
        plt.close(fig)


def test_panel_rejects_more_than_six(midline_result):
    with pytest.raises(ValueError):
        plotting._build_panel_figure([midline_result] * 7)


def test_panel_empty_collection_does_not_crash(tmp_path):
    out = plotting.plate_panel([], tmp_path / "empty_panel.png")
    assert out.exists()


def test_panel_smoke(midline_result, tmp_path):
    out = plotting.plate_panel([midline_result, midline_result], tmp_path / "p.png")
    assert out.exists() and out.stat().st_size > 0
    assert plt.get_fignums() == []


# --------------------------------------------------------------------------- #
# save_plots
# --------------------------------------------------------------------------- #
def test_save_plots_disabled_writes_nothing(inputs_3plants, tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    result = plotting.save_plots(inputs_3plants, out_dir, enabled=False)
    assert result == []
    assert not (out_dir / "plots").exists()


def test_save_plots_requires_existing_out_dir(inputs_3plants, tmp_path):
    with pytest.raises(FileNotFoundError):
        plotting.save_plots(inputs_3plants, tmp_path / "nope")


def test_save_plots_writes_set_and_sidecar(inputs_3plants, tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    written = plotting.save_plots(inputs_3plants, out_dir)
    plots = out_dir / "plots"
    assert plots.is_dir()
    assert len(written) >= 1
    assert all(p.stat().st_size > 0 for p in written)
    # filenames keyed on integer track_id, panel + sidecar present
    names = {p.name for p in plots.iterdir()}
    assert "panel.png" in names
    assert "plots_metadata.json" in names
    assert any(n.startswith("plant0_") for n in names)
    # strict JSON sidecar with run_id key
    meta = json.loads((plots / "plots_metadata.json").read_text(encoding="utf-8"))
    assert meta["constants_version"] == plotting._CONSTANTS_VERSION
    assert "run_id" in meta and "plants" in meta and "files" in meta


def test_save_plots_filenames_use_track_id_not_nan(tmp_path):
    rows = _track_rows(0)
    for r in rows:
        r["plate_id"] = np.nan
        r["plant_id"] = np.nan  # aspirational fields NaN; track_id stays 0
    inputs = CircumnutationInputs(trajectory_df=pd.DataFrame(rows), cadence_s=300.0)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    written = plotting.save_plots(inputs, out_dir)
    assert written
    assert all("nan" not in p.name.lower() for p in written)
    assert any(p.name.startswith("plant0_") for p in written)


def test_save_plots_zero_plants(tmp_path):
    cols = [
        "series",
        "sample_uid",
        "timepoint",
        "plate_id",
        "plant_id",
        "track_id",
        "genotype",
        "treatment",
        "frame",
        "tip_x",
        "tip_y",
    ]
    # one row so the df is non-empty (CircumnutationInputs requires >=1 row),
    # but a single frame degenerates every chain -> no plots.
    df = pd.DataFrame(
        [
            {
                "series": "p",
                "sample_uid": "p",
                "timepoint": "T0",
                "plate_id": "p",
                "plant_id": 0,
                "track_id": 0,
                "genotype": "g",
                "treatment": "n",
                "frame": 0,
                "tip_x": 1.0,
                "tip_y": 1.0,
            }
        ],
        columns=cols,
    )
    inputs = CircumnutationInputs(trajectory_df=df, cadence_s=300.0)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    written = plotting.save_plots(inputs, out_dir)
    assert written == []
    assert not (out_dir / "plots" / "plots_metadata.json").exists()


# --------------------------------------------------------------------------- #
# CC-3 (pure-pixel) + #230 deviation
# --------------------------------------------------------------------------- #
def test_no_px_per_mm_or_mm_anywhere(temporal_result, midline_result):
    for fn in (
        plotting.scaleogram,
        plotting.trail_overlay,
        plotting.plate_panel,
        plotting.save_plots,
    ):
        assert "px_per_mm" not in inspect.signature(fn).parameters
    sg, ridge = temporal_result
    fig = plotting._build_scaleogram_figure(sg, ridge_result=ridge)
    try:
        for ax in fig.axes:
            for label in (ax.get_xlabel(), ax.get_ylabel()):
                assert "mm" not in label.replace("px⁻¹", "")
    finally:
        plt.close(fig)


def test_no_lgz_parameter_anywhere():
    for fn in (
        plotting.scaleogram,
        plotting.trail_overlay,
        plotting.plate_panel,
        plotting.save_plots,
    ):
        params = set(inspect.signature(fn).parameters)
        assert not any(
            "l_gz" in p.lower() or "growth_zone" in p.lower() for p in params
        )


# --------------------------------------------------------------------------- #
# Review-reconciliation edge cases (pre-push /review-pr)
# --------------------------------------------------------------------------- #
def test_save_plots_rejects_non_integer_track_id(tmp_path):
    """Two plants whose track_id truncates to the same int must NOT silently collide."""
    rows = _track_rows(0)
    rows2 = _track_rows(0, phase=0.7)
    for r in rows:
        r["track_id"] = 0.0
    for r in rows2:
        r["track_id"] = 0.4  # distinct group, same int() truncation -> reject
    inputs = CircumnutationInputs(
        trajectory_df=pd.DataFrame(rows + rows2), cadence_s=300.0
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    with pytest.raises(ValueError, match="track_id"):
        plotting.save_plots(inputs, out_dir)


def test_panel_all_nan_curvature_pool(midline_result):
    """A plate_panel whose entire pooled κ is non-finite still renders (no norm crash)."""
    nan_mr = type(midline_result)(
        frame_indices=midline_result.frame_indices,
        x_smooth_px=midline_result.x_smooth_px,
        y_smooth_px=midline_result.y_smooth_px,
        speed_px_per_frame=midline_result.speed_px_per_frame,
        arc_length_px=midline_result.arc_length_px,
        curvature_px_inv=np.full_like(midline_result.curvature_px_inv, np.nan),
        velocity_sub_noise_mask=midline_result.velocity_sub_noise_mask,
        cadence_s=midline_result.cadence_s,
        sg_window=midline_result.sg_window,
        sg_degree=midline_result.sg_degree,
        sigma_v_px_per_frame=midline_result.sigma_v_px_per_frame,
        noise_mask_k=midline_result.noise_mask_k,
        is_degenerate=midline_result.is_degenerate,
    )
    fig = plotting._build_panel_figure([nan_mr, nan_mr])
    plt.close(fig)


def test_helper_skip_paths_return_none():
    """A stationary (zero-net-displacement) track degenerates both tier chains."""
    df = pd.DataFrame(_track_rows(0))
    df["tip_x"] = 5.0  # constant -> zero net displacement
    df["tip_y"] = 5.0
    group = df
    assert (
        plotting._temporal_scaleogram_result(group, 300.0, plotting.ConstantsT())
        is None
    )
    mr, sg, ridge = plotting._spatial_artifacts(group, 300.0, plotting.ConstantsT())
    assert sg is None and ridge is None  # degenerate midline -> no spatial scaleogram


def test_to_jsonable_coerces_path_array_and_na():
    payload = {
        "p": Path("a/b.png"),
        "arr": np.array([1.0, 2.0]),
        "nan": float("nan"),
        "npnan": np.float64("nan"),
        "na": pd.NA,
        "i": np.int64(3),
        "ok": "str",
    }
    out = plotting._to_jsonable(payload)
    # strict JSON must not raise on the coerced payload
    json.dumps(out, allow_nan=False)
    assert out["p"] == "a/b.png"
    assert out["arr"] == [1.0, 2.0]
    assert out["nan"] is None and out["npnan"] is None and out["na"] is None
    assert out["i"] == 3 and out["ok"] == "str"


def test_sidecar_records_resolved_constants(inputs_3plants, tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    plotting.save_plots(inputs_3plants, out_dir)
    meta = json.loads(
        (out_dir / "plots" / "plots_metadata.json").read_text(encoding="utf-8")
    )
    # the full resolved ConstantsT snapshot is recorded (not just the version int)
    assert "constants" in meta and "SG_WINDOW_DETREND" in meta["constants"]
