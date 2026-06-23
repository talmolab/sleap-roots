"""Diagnostic plots for circumnutation traits (PR #16).

Renders the per-plant CWT scaleograms (Tier 1 temporal + Tier 3 spatial), a
κ-color-coded tip-trail overlay, and a 2×3 per-plate panel as PNG files. All
plots are written into a ``plots/`` subdirectory of the pipeline output
directory; generation is suppressible via the ``save_plots(..., enabled=False)``
parameter (the ``--no-plots`` CLI flag that sets it lands in PR #17).

The four public callables are:

- :func:`scaleogram` — a CWT scaleogram heatmap (power ``|C|²``, log color norm,
  cone-of-influence dimming, optional ridge overlay). Polymorphic over the
  temporal :class:`~sleap_roots.circumnutation.temporal_cwt.ScaleogramResult` and
  the spatial
  :class:`~sleap_roots.circumnutation.spatial_cwt.SpatialScaleogramResult`.
- :func:`trail_overlay` — the smoothed tip path colored by the signed
  trajectory curvature ``κ`` (diverging colormap, image-y-down).
- :func:`plate_panel` — a 2×3 grid of trail overlays sharing one κ normalization
  and one colorbar.
- :func:`save_plots` — the orchestrator that re-derives the per-plant Results by
  calling the same tier helpers the analysis uses and writes the standard PNG set
  plus a ``plots_metadata.json`` provenance sidecar.

All axes/colorbars are pure-pixel (CC-3): pixel-native units only, never mm. Plot
display constants (DPI, figure sizes, colormaps, the κ percentile) are
module-level here and deliberately NOT in the ``ConstantsT`` override-bag, so
``_CONSTANTS_VERSION`` is unaffected.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Union

import attrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm, Normalize

from sleap_roots.circumnutation import (
    _noise,
    midline,
    nutation,
    spatial_cwt,
    temporal_cwt,
)
from sleap_roots.circumnutation._constants import _CONSTANTS_VERSION, ConstantsT
from sleap_roots.circumnutation._io import _IDENTITY_5_TUPLE
from sleap_roots.circumnutation.midline import MidlineResult
from sleap_roots.circumnutation.spatial_cwt import (
    SpatialRidgeResult,
    SpatialScaleogramResult,
)
from sleap_roots.circumnutation.temporal_cwt import RidgeResult, ScaleogramResult


logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Module-level display constants (structural, NOT in ConstantsT).
# These are rendering knobs, not analysis parameters, so they do NOT live in the
# typed ConstantsT override-bag and they do NOT bump _CONSTANTS_VERSION.
# --------------------------------------------------------------------------- #
_DPI: int = 150
_FIGSIZE_SCALEOGRAM: tuple = (8.0, 5.0)
_FIGSIZE_TRAIL: tuple = (5.0, 6.0)
_FIGSIZE_PANEL: tuple = (12.0, 8.0)
_CMAP_SCALEOGRAM: str = "viridis"
_CMAP_KAPPA: str = "RdBu_r"
_KAPPA_PCT: float = 98.0
_KAPPA_FALLBACK: float = 1.0  # symmetric ±limit when no finite κ exists
_LOG_EDGE_FACTOR: float = 1.5  # half-decade-ish edge spread for a single log center
_NAN_COLOR: str = "lightgray"
_PANEL_ROWS: int = 2
_PANEL_COLS: int = 3
_PANEL_CAPACITY: int = _PANEL_ROWS * _PANEL_COLS


def _resolve_constants(constants: Optional[ConstantsT]) -> ConstantsT:
    """Return ``constants`` or the default :class:`ConstantsT` when ``None``."""
    return constants if constants is not None else ConstantsT()


# --------------------------------------------------------------------------- #
# pcolormesh cell-edge helpers (centers -> n+1 edges)
# --------------------------------------------------------------------------- #
def _edges_linear(centers: np.ndarray) -> np.ndarray:
    """Build ``len(centers)+1`` linear cell edges from monotonic centers."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.size == 1:
        c = float(centers[0])
        half = 0.5 if c == 0.0 else 0.5 * abs(c)
        return np.array([c - half, c + half])
    mids = 0.5 * (centers[:-1] + centers[1:])
    first = centers[0] - (mids[0] - centers[0])
    last = centers[-1] + (centers[-1] - mids[-1])
    return np.concatenate([[first], mids, [last]])


def _edges_log(centers: np.ndarray) -> np.ndarray:
    """Build ``len(centers)+1`` geometric (log-space) cell edges; centers > 0."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.size == 1:
        c = float(centers[0])
        return np.array([c / _LOG_EDGE_FACTOR, c * _LOG_EDGE_FACTOR])
    return np.exp(_edges_linear(np.log(centers)))


# --------------------------------------------------------------------------- #
# scaleogram
# --------------------------------------------------------------------------- #
def _scaleogram_axes_spec(
    scaleogram_result: Union[ScaleogramResult, SpatialScaleogramResult],
    ridge_result,
) -> dict:
    """Resolve per-type axes/labels for :func:`_build_scaleogram_figure`.

    Returns a dict with the physical x-axis centers, the (log) y-axis centers,
    axis labels, an optional uncalibrated annotation, the ridge y-values, and the
    expected ridge type — dispatching on ``ScaleogramResult`` (temporal) vs
    ``SpatialScaleogramResult`` (spatial). Raises ``TypeError`` for any other
    scaleogram type or a mismatched ``ridge_result`` type.
    """
    if isinstance(scaleogram_result, ScaleogramResult):
        if ridge_result is not None and not isinstance(ridge_result, RidgeResult):
            raise TypeError(
                "ridge_result for a temporal ScaleogramResult must be a "
                "RidgeResult (got "
                f"{type(ridge_result).__name__}); a SpatialRidgeResult or other "
                "type is not allowed."
            )
        n_frames = scaleogram_result.scaleogram.shape[1]
        x_centers = np.arange(n_frames, dtype=np.float64) * float(
            scaleogram_result.cadence_s
        )
        ridge_y = None if ridge_result is None else ridge_result.periods_s
        return {
            "x_centers": x_centers,
            "y_centers": scaleogram_result.periods_s,
            "xlabel": "time [s]",
            "ylabel": "period [s]",
            "annotation": None,
            "ridge_y": ridge_y,
        }
    if isinstance(scaleogram_result, SpatialScaleogramResult):
        if ridge_result is not None and not isinstance(
            ridge_result, SpatialRidgeResult
        ):
            raise TypeError(
                "ridge_result for a SpatialScaleogramResult must be a "
                "SpatialRidgeResult (got "
                f"{type(ridge_result).__name__}); a RidgeResult or other type is "
                "not allowed."
            )
        n_samples = scaleogram_result.scaleogram.shape[1]
        x_centers = np.arange(n_samples, dtype=np.float64) * float(scaleogram_result.ds)
        ridge_y = None if ridge_result is None else ridge_result.wavelengths_px
        return {
            "x_centers": x_centers,
            "y_centers": scaleogram_result.wavelengths_px,
            "xlabel": "arc length [px]",
            "ylabel": "wavelength [px]",
            "annotation": "uncalibrated pywt convention",
            "ridge_y": ridge_y,
        }
    raise TypeError(
        "scaleogram_result must be a ScaleogramResult or SpatialScaleogramResult, "
        f"got {type(scaleogram_result).__name__}."
    )


def _build_scaleogram_figure(
    scaleogram_result: Union[ScaleogramResult, SpatialScaleogramResult],
    ridge_result=None,
) -> plt.Figure:
    """Build (but do not save) the scaleogram diagnostic figure.

    Returns the :class:`matplotlib.figure.Figure` for structural introspection;
    callers are responsible for closing it.
    """
    spec = _scaleogram_axes_spec(scaleogram_result, ridge_result)
    power = np.abs(scaleogram_result.scaleogram) ** 2
    coi_mask = scaleogram_result.coi_mask

    x_edges = _edges_linear(spec["x_centers"])
    y_edges = _edges_log(spec["y_centers"])

    # LogNorm rejects vmin <= 0; floor at the smallest strictly-positive power.
    positive = power[power > 0.0]
    vmin = float(positive.min()) if positive.size else 1e-12
    vmax = float(power.max()) if power.size and np.isfinite(power.max()) else 1.0
    if not (vmax > vmin):
        vmax = vmin * 10.0
    norm = LogNorm(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=_FIGSIZE_SCALEOGRAM, dpi=_DPI)
    mesh = ax.pcolormesh(
        x_edges, y_edges, power, norm=norm, cmap=_CMAP_SCALEOGRAM, shading="flat"
    )
    ax.set_yscale("log")
    ax.set_xlabel(spec["xlabel"])
    ax.set_ylabel(spec["ylabel"])
    if spec["annotation"] is not None:
        ax.set_title(spec["annotation"], fontsize=9, loc="right")

    # Dim the cone of influence (coi_mask True == inside == unreliable).
    coi_overlay = np.ma.masked_where(~coi_mask, np.ones_like(power))
    ax.pcolormesh(
        x_edges,
        y_edges,
        coi_overlay,
        cmap="gray_r",
        alpha=0.45,
        shading="flat",
        vmin=0.0,
        vmax=1.0,
    )

    # Optional ridge overlay: faded over the whole length, solid where reliable
    # (in_coi == False). NaN breaks the solid line inside the COI.
    if spec["ridge_y"] is not None and ridge_result is not None:
        ridge_y = np.asarray(spec["ridge_y"], dtype=np.float64)
        x_centers = spec["x_centers"]
        in_coi = np.asarray(ridge_result.in_coi, dtype=bool)
        ax.plot(x_centers, ridge_y, color="white", lw=1.0, alpha=0.4)
        solid = np.where(in_coi, np.nan, ridge_y)
        ax.plot(x_centers, solid, color="white", lw=1.6)

    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("power |C|²")
    fig.tight_layout()
    return fig


def scaleogram(scaleogram_result, out_path, *, ridge_result=None) -> Path:
    """Render a CWT scaleogram diagnostic plot to a PNG.

    Polymorphic over the temporal
    :class:`~sleap_roots.circumnutation.temporal_cwt.ScaleogramResult` (axes
    ``time [s]`` / ``period [s]``) and the spatial
    :class:`~sleap_roots.circumnutation.spatial_cwt.SpatialScaleogramResult`
    (axes ``arc length [px]`` / ``wavelength [px]``, labeled uncalibrated). The
    heatmap shows power ``|C|²`` on a logarithmic color norm, dims the
    cone-of-influence region, and optionally overlays the per-frame/per-position
    ridge (solid where reliable, faded inside the COI).

    Args:
        scaleogram_result: A ``ScaleogramResult`` or ``SpatialScaleogramResult``.
        out_path: PNG output path; its parent directory must already exist.
        ridge_result: Optional ridge to overlay. Its type MUST agree with the
            scaleogram type (``ScaleogramResult`` with ``RidgeResult``,
            ``SpatialScaleogramResult`` with ``SpatialRidgeResult``).

    Note:
        For a spatial scaleogram the overlaid ridge is the **raw** ridge — the
        ``wavelength [px]`` axis is the uncalibrated pywt convention (annotated as
        such) AND the ridge is **not COI-gated**, so it is a "where the energy is"
        diagnostic, not the authoritative, COI-gated + cgau2-calibrated
        ``lambda_spatial_median_px`` trait (theory.md §7.4). The COI region is
        visually dimmed and the ridge is faded inside it.

    Returns:
        The ``out_path`` (as a :class:`pathlib.Path`) of the written PNG.

    Raises:
        TypeError: If ``scaleogram_result`` is an unsupported type, or
            ``ridge_result`` does not match the scaleogram type.
    """
    out_path = Path(out_path)
    fig = _build_scaleogram_figure(scaleogram_result, ridge_result=ridge_result)
    try:
        fig.savefig(out_path, dpi=_DPI)
    finally:
        plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# trail_overlay
# --------------------------------------------------------------------------- #
def _kappa_segments(midline_result: MidlineResult):
    """Return ``(segments, segment_kappa)`` for the per-segment LineCollection.

    For ``n`` points there are ``n-1`` segments; the per-segment curvature is the
    midpoint average of the two endpoints' ``curvature_px_inv`` (length ``n-1``).
    """
    x = np.asarray(midline_result.x_smooth_px, dtype=np.float64)
    y = np.asarray(midline_result.y_smooth_px, dtype=np.float64)
    kappa = np.asarray(midline_result.curvature_px_inv, dtype=np.float64)
    points = np.column_stack([x, y])
    segments = np.stack([points[:-1], points[1:]], axis=1)
    seg_kappa = 0.5 * (kappa[:-1] + kappa[1:])
    return segments, seg_kappa


def _kappa_limit(values: np.ndarray) -> float:
    """Symmetric κ limit = the 98th percentile of ``|values|`` over finite ones.

    Returns :data:`_KAPPA_FALLBACK` when there is no finite, non-zero value (an
    all-NaN-κ track), so the symmetric norm never gets a NaN/zero bound.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return _KAPPA_FALLBACK
    q = float(np.percentile(np.abs(finite), _KAPPA_PCT))
    return q if q > 0.0 else _KAPPA_FALLBACK


def _kappa_cmap():
    """Return a copied diverging colormap with a ``set_bad`` for NaN segments.

    Copies the global colormap first so the process-global registry entry is
    never mutated (the same hazard class as the Agg backend).
    """
    cmap = plt.get_cmap(_CMAP_KAPPA).copy()
    cmap.set_bad(color=_NAN_COLOR)
    return cmap


def _draw_trail_on_ax(ax, midline_result: MidlineResult, norm, cmap) -> None:
    """Draw one κ-colored tip trail onto ``ax`` using the given norm + cmap."""
    segments, seg_kappa = _kappa_segments(midline_result)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(seg_kappa)
    lc.set_linewidth(2.0)
    ax.add_collection(lc)

    x = np.asarray(midline_result.x_smooth_px, dtype=np.float64)
    y = np.asarray(midline_result.y_smooth_px, dtype=np.float64)
    fx, fy = x[np.isfinite(x)], y[np.isfinite(y)]
    if fx.size and fy.size:
        padx = 0.05 * (fx.max() - fx.min() + 1.0)
        pady = 0.05 * (fy.max() - fy.min() + 1.0)
        ax.set_xlim(fx.min() - padx, fx.max() + padx)
        # y-down image convention: larger y at the bottom (inverted axis).
        ax.set_ylim(fy.max() + pady, fy.min() - pady)
    ax.set_aspect("equal")
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")


def _build_trail_figure(midline_result: MidlineResult) -> plt.Figure:
    """Build (but do not save) the single-plant κ-trail figure."""
    q = _kappa_limit(np.asarray(midline_result.curvature_px_inv, dtype=np.float64))
    norm = Normalize(vmin=-q, vmax=q)
    cmap = _kappa_cmap()

    fig, ax = plt.subplots(figsize=_FIGSIZE_TRAIL, dpi=_DPI)
    _draw_trail_on_ax(ax, midline_result, norm, cmap)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, extend="both")
    cbar.set_label("κ [px⁻¹]")
    fig.tight_layout()
    return fig


def trail_overlay(midline_result, out_path) -> Path:
    """Render the κ-color-coded tip-trail overlay to a PNG.

    The smoothed tip path ``(x_smooth_px, y_smooth_px)`` is drawn as a
    per-segment :class:`~matplotlib.collections.LineCollection` colored by the
    signed trajectory curvature ``curvature_px_inv`` on a diverging colormap
    symmetric about zero (limits ``±`` the 98th percentile of ``|κ|``). NaN
    curvature segments use the colormap's "bad" color; the y-axis is image-down.

    Args:
        midline_result: A
            :class:`~sleap_roots.circumnutation.midline.MidlineResult`.
        out_path: PNG output path; its parent directory must already exist.

    Returns:
        The ``out_path`` (as a :class:`pathlib.Path`) of the written PNG.
    """
    out_path = Path(out_path)
    fig = _build_trail_figure(midline_result)
    try:
        fig.savefig(out_path, dpi=_DPI)
    finally:
        plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# plate_panel
# --------------------------------------------------------------------------- #
def _build_panel_figure(midline_results) -> plt.Figure:
    """Build (but do not save) the 2×3 per-plate panel figure.

    All subplots share a single κ normalization computed over the pooled finite
    curvature of every plant, with one shared colorbar. Empty cells are hidden
    when fewer than six plants are supplied.

    Raises:
        ValueError: If more than six plant results are supplied (the panel is a
            fixed 2×3 grid).
    """
    results = list(midline_results)
    if len(results) > _PANEL_CAPACITY:
        raise ValueError(
            f"plate_panel supports at most {_PANEL_CAPACITY} plants per 2×3 "
            f"panel, got {len(results)}."
        )

    pooled = (
        np.concatenate(
            [np.asarray(r.curvature_px_inv, dtype=np.float64) for r in results]
        )
        if results
        else np.array([], dtype=np.float64)
    )
    q = _kappa_limit(pooled)
    norm = Normalize(vmin=-q, vmax=q)
    cmap = _kappa_cmap()

    fig, axes = plt.subplots(_PANEL_ROWS, _PANEL_COLS, figsize=_FIGSIZE_PANEL, dpi=_DPI)
    flat_axes = np.asarray(axes).ravel()
    for idx, ax in enumerate(flat_axes):
        if idx < len(results):
            _draw_trail_on_ax(ax, results[idx], norm, cmap)
        else:
            ax.set_visible(False)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=flat_axes.tolist(), extend="both")
    cbar.set_label("κ [px⁻¹]")
    return fig


def plate_panel(midline_results, out_path) -> Path:
    """Render a 2×3 per-plate panel of κ-trail overlays to a PNG.

    Each subplot is one plant's trail; all share a single curvature
    normalization (over the pooled finite κ of every plant) and one colorbar, so
    equal color means equal curvature across plants. Empty cells are hidden when
    fewer than six plants are supplied.

    Args:
        midline_results: An ordered collection of
            :class:`~sleap_roots.circumnutation.midline.MidlineResult` (one per
            plant; at most six).
        out_path: PNG output path; its parent directory must already exist.

    Returns:
        The ``out_path`` (as a :class:`pathlib.Path`) of the written PNG.

    Raises:
        ValueError: If more than six plant results are supplied.
    """
    out_path = Path(out_path)
    fig = _build_panel_figure(midline_results)
    try:
        fig.savefig(out_path, dpi=_DPI)
    finally:
        plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# save_plots orchestrator
# --------------------------------------------------------------------------- #
def _to_jsonable(value):
    """Coerce numpy / Path / non-finite values to strict-JSON-safe natives.

    Non-finite floats (``NaN``/``±inf``) — including aspirational identity
    fields like ``plate_id`` that may be ``NaN`` — are mapped to ``None`` so
    ``json.dumps(..., allow_nan=False)`` cannot raise.
    """
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return f if np.isfinite(f) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return _to_jsonable(value.tolist())
    if value is None or isinstance(value, (str, int)):
        return value
    # Scalar NA/NaN/NaT (e.g. a pandas-nullable plate_id) -> JSON null, so a
    # non-finite identity value cannot become the literal string "<NA>".
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _plots_metadata(
    inputs, constants: ConstantsT, identities: list, filenames: list
) -> dict:
    """Assemble the ``plots_metadata.json`` provenance payload."""
    return {
        "constants_version": _CONSTANTS_VERSION,
        # The full resolved ConstantsT snapshot, so two runs with different
        # overrides (but the same constants_version) are distinguishable — the
        # version integer alone is not a reproducibility key.
        "constants": attrs.asdict(constants),
        "display_constants": {
            "dpi": _DPI,
            "figsize_scaleogram": list(_FIGSIZE_SCALEOGRAM),
            "figsize_trail": list(_FIGSIZE_TRAIL),
            "figsize_panel": list(_FIGSIZE_PANEL),
            "cmap_scaleogram": _CMAP_SCALEOGRAM,
            "cmap_kappa": _CMAP_KAPPA,
            "kappa_percentile": _KAPPA_PCT,
        },
        "run_id": inputs.run_id,
        "cadence_s": inputs.cadence_s,
        "R_px": inputs.R_px,
        "run_metadata_ref": "../run_metadata.json",
        "plants": [dict(zip(_IDENTITY_5_TUPLE, identity)) for identity in identities],
        "files": list(filenames),
    }


def _temporal_scaleogram_result(group: pd.DataFrame, cadence_s, constants):
    """Re-derive the Tier 1 ``(ScaleogramResult, RidgeResult)`` for a plant.

    Mirrors ``nutation._compute_one_track`` guard-for-guard: ``_select_signal``
    (raises ``ValueError`` on a non-finite tip column), the two finite-any
    short-circuits, the ``ValueError`` guard around ``compute_scaleogram``, then
    the SMOOTHED ridge (which drives ``T_nutation_median``). Returns ``None`` when
    any guard trips.
    """
    try:
        raw = nutation._select_signal(group, "lateral")
    except ValueError:
        return None
    if not np.isfinite(raw).any():
        return None
    signal = _noise.compute_sg_detrended(
        raw,
        window=int(constants.SG_WINDOW_DETREND),
        polynomial_order=int(constants.SG_DEGREE),
    )
    if not np.isfinite(signal).any():
        return None
    try:
        sg = temporal_cwt.compute_scaleogram(signal, cadence_s, constants=constants)
    except ValueError:
        return None
    raw_ridge = temporal_cwt.extract_ridge(sg, constants=constants)
    ridge = temporal_cwt.smooth_ridge(raw_ridge, constants=constants)
    return sg, ridge


def _spatial_artifacts(group: pd.DataFrame, cadence_s, constants):
    """Re-derive the Tier 3 ``(MidlineResult, spatial scaleogram, ridge)``.

    Mirrors ``traveling_wave._compute_one_track``: frame-sort, finite-tip
    pre-filter (``midline.reconstruct`` rejects non-finite input), reconstruct,
    then the resample → spatial scaleogram → ridge chain. Returns
    ``(mr, sg_or_None, ridge_or_None)`` — ``mr`` is ``None`` only when
    reconstruction itself fails/degenerates; the spatial scaleogram is ``None``
    when the resample or CWT degenerates.
    """
    sub = group.sort_values("frame")
    x = sub["tip_x"].to_numpy(dtype=np.float64)
    y = sub["tip_y"].to_numpy(dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    try:
        mr = midline.reconstruct(x, y, cadence_s=cadence_s, constants=constants)
    except (ValueError, TypeError):
        return None, None, None
    if mr.is_degenerate:
        return None, None, None
    try:
        rs = spatial_cwt.resample_curvature(
            mr.curvature_px_inv,
            mr.arc_length_px,
            mr.velocity_sub_noise_mask,
            constants=constants,
        )
    except (ValueError, TypeError):
        return mr, None, None
    if rs.is_degenerate:
        return mr, None, None
    try:
        sg = spatial_cwt.compute_scaleogram(
            rs.kappa_uniform, rs.ds, constants=constants
        )
        ridge = spatial_cwt.extract_ridge(sg, constants=constants)
    except ValueError:
        return mr, None, None
    return mr, sg, ridge


def save_plots(inputs, out_dir, *, constants=None, enabled=True) -> list[Path]:
    """Write the per-plant diagnostic plot set into a ``plots/`` subdirectory.

    Re-derives the per-plant ``ScaleogramResult`` / ``SpatialScaleogramResult`` /
    ``MidlineResult`` by invoking the same tier helper functions the analysis
    uses (with the supplied ``constants``), so the plotted signal equals the
    analyzed input. For each non-degenerate plant it writes a temporal
    scaleogram, a spatial scaleogram, and a trail overlay; it writes one per-plate
    panel and a ``plots_metadata.json`` provenance sidecar.

    The temporal scaleogram is built from the **lateral** nutation signal (the
    ``nutation.compute`` default that the pipeline uses); it is not faithful to a
    bespoke ``coordinate="x"``/``"y"`` analysis run. This function does not select
    a matplotlib backend — a headless caller (CI, the PR #17 CLI) must set
    ``matplotlib.use("Agg")`` before calling it.

    Args:
        inputs: A
            :class:`~sleap_roots.circumnutation._types.CircumnutationInputs`.
        out_dir: The output directory; it MUST already exist (the ``plots/``
            leaf is created under it). ``save_plots`` is one-plate-per-``out_dir``.
        constants: Optional :class:`ConstantsT` override-bag (defaults applied
            when ``None``).
        enabled: When ``False``, no plots or sidecar are written and an empty
            list is returned (the ``--no-plots`` hook; CLI flag in PR #17).

    Returns:
        The list of written PNG :class:`pathlib.Path` objects (empty when
        ``enabled=False`` or no plant produced a plot).

    Raises:
        FileNotFoundError: If ``out_dir`` does not exist.
    """
    if not enabled:
        logger.info("plotting disabled (enabled=False), skipping all plots")
        return []

    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        raise FileNotFoundError(
            f"save_plots: out_dir does not exist: {out_dir!s} (create it before "
            f"saving; save_plots creates only the plots/ leaf)."
        )

    resolved = _resolve_constants(constants)
    cadence_s = inputs.cadence_s
    df = inputs.trajectory_df

    # Filenames key on the integer `track_id`. `save_plots` is a public entry
    # point that does not run the pipeline's `_validate_integer_identity`, so
    # guard `track_id` here: a fractional float (e.g. 0.0 vs 0.4) would truncate
    # to the same `int()` and silently overwrite another plant's PNGs + corrupt
    # the sidecar. Raise rather than corrupt (the pipeline's contract). Note we
    # validate ONLY `track_id` — `plant_id`/`plate_id` are aspirational and may be
    # NaN by design, which is exactly why filenames use `track_id`.
    # Coerce via pd.to_numeric (errors="coerce") so a non-numeric/object-dtype
    # track_id surfaces as the friendly ValueError below rather than a bare
    # TypeError from np.isfinite.
    track_vals = pd.to_numeric(df["track_id"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    if not np.all(np.isfinite(track_vals)) or not np.array_equal(
        track_vals, track_vals.astype(np.int64)
    ):
        raise ValueError(
            "track_id must be integer-valued and finite for plot filenames; "
            "non-integer (or non-numeric) values would truncate to int64 and "
            "silently overwrite another plant's plots."
        )

    plots_dir = out_dir / "plots"
    written: list[Path] = []
    identities: list = []
    midline_results: list = []

    for key, group in df.groupby(list(_IDENTITY_5_TUPLE), dropna=False, sort=False):
        identity = key if isinstance(key, tuple) else (key,)
        track_label = int(dict(zip(_IDENTITY_5_TUPLE, identity))["track_id"])
        plant_written = False

        # Tier 1 — temporal scaleogram (nutation chain; no frame sort).
        temporal = _temporal_scaleogram_result(group, cadence_s, resolved)
        if temporal is not None:
            sg_t, ridge_t = temporal
            plots_dir.mkdir(exist_ok=True)
            p = scaleogram(
                sg_t,
                plots_dir / f"plant{track_label}_scaleogram_temporal.png",
                ridge_result=ridge_t,
            )
            written.append(p)
            plant_written = True
            logger.debug("wrote temporal scaleogram for plant %s", track_label)
        else:
            logger.debug("skipped temporal scaleogram for plant %s", track_label)

        # Tier 3 — midline (trail + panel) and spatial scaleogram.
        mr, sg_s, ridge_s = _spatial_artifacts(group, cadence_s, resolved)
        if mr is not None:
            plots_dir.mkdir(exist_ok=True)
            p = trail_overlay(mr, plots_dir / f"plant{track_label}_trail.png")
            written.append(p)
            midline_results.append(mr)
            plant_written = True
            logger.debug("wrote trail overlay for plant %s", track_label)
            if sg_s is not None:
                p = scaleogram(
                    sg_s,
                    plots_dir / f"plant{track_label}_scaleogram_spatial.png",
                    ridge_result=ridge_s,
                )
                written.append(p)
                logger.debug("wrote spatial scaleogram for plant %s", track_label)
            else:
                logger.debug("skipped spatial scaleogram for plant %s", track_label)
        else:
            logger.debug("skipped midline/spatial plots for plant %s", track_label)

        if plant_written:
            identities.append(identity)

    # Per-plate panel (from the non-degenerate midlines).
    if midline_results:
        plots_dir.mkdir(exist_ok=True)
        written.append(plate_panel(midline_results, plots_dir / "panel.png"))

    # Provenance sidecar (only when at least one plot was written).
    if written:
        if inputs.run_id is None:
            logger.debug("plots_metadata.json: inputs.run_id is None (no join key)")
        metadata = _plots_metadata(
            inputs, resolved, identities, [p.name for p in written]
        )
        sidecar = plots_dir / "plots_metadata.json"
        sidecar.write_text(
            json.dumps(_to_jsonable(metadata), indent=2, allow_nan=False),
            encoding="utf-8",
        )

    logger.info("wrote %d plots to %s", len(written), plots_dir)
    return written
