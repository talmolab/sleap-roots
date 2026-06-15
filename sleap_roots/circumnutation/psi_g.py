"""Tier 2 — Bastien-Meroz ψ_g trait emission (PR #7, ``add-circumnutation-tier2-psi-g``).

Public callable: :func:`compute` — emits 4 self-contained Tier 2 trait
columns per track from the unwrapped velocity-direction angle ``ψ_g(t)``
(``_geometry.compute_psi_g``, ``atan2(dx, dy)`` per BM2016 Eq. 20 /
``docs/circumnutation/theory.md`` §3.5):

- ``T_psig_median_s`` — COI-masked median of the smoothed-ridge periods of
  the SG-detrended ψ_g (composes ``temporal_cwt.compute_scaleogram`` →
  ``extract_ridge`` → ``smooth_ridge``). The ONLY trait that uses the
  conditioned signal + CWT.
- ``delta_E_amplitude_proxy_px_per_frame`` — ``median(√(dx²+dy²))`` over raw
  finite velocity samples (Eq. 21 amplitude proxy; px/frame, no COI).
- ``handedness`` — ``int(np.sign(ψ_g[-1] − ψ_g[0]))`` over all finite frames
  (net unwrapped rotation; COI-free). ``+1`` ⇔ positive ``dψ_g/dt``.
- ``helix_signed_area_px2`` — y-down Shoelace area
  (``_geometry.compute_signed_area``) of the **growth-detrended** tip
  trajectory (per-axis linear detrend removes the growth ribbon so the
  enclosed area reflects the nutation orbit). ``sign(area)`` confirms
  ``handedness`` on genuinely 2-D-circulating real data (≥5/6 on plate-001;
  raw un-detrended Shoelace agreed only 1/6 — the growth ribbon dominated).

Tier 2 is self-contained: it does NOT consume Tier 1 output. The 5th §7.3
trait ``psig_long_consistency`` (cross-tier ``T_psig ↔ T_nutation``) is
deferred to the roadmapped PR #13 Layer-3 work.

Anchors: spec at
``openspec/changes/add-circumnutation-tier2-psi-g/specs/circumnutation/spec.md``;
design at ``openspec/changes/add-circumnutation-tier2-psi-g/design.md`` +
``docs/superpowers/specs/2026-06-05-add-circumnutation-tier2-psi-g-design.md``
§13; ``docs/circumnutation/theory.md`` §3.5 (ψ_g) + §7.3 (Tier 2 trait table).
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from sleap_roots.circumnutation import _geometry, _noise, temporal_cwt
from sleap_roots.circumnutation._constants import ConstantsT
from sleap_roots.circumnutation._io import (
    _IDENTITY_5_TUPLE,
    _build_per_plant_template_from_df,
)
from sleap_roots.circumnutation._types import (
    ROW_IDENTITY_COLUMNS,
    _validate_trajectory_df,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level contracts
# ---------------------------------------------------------------------------

# 4 trait columns in declared order (per spec ADDED requirement).
_PSIG_TRAIT_COLUMNS: tuple = (
    "T_psig_median_s",
    "delta_E_amplitude_proxy_px_per_frame",
    "handedness",
    "helix_signed_area_px2",
)

# Per-column units for the 4 Tier-2 trait/flag columns (GitHub issue #222
# units-map portion; consumed by the PR #14 pipeline's units-sidecar assembly,
# mirroring the _TIER0_TRAIT_UNITS precedent). ``handedness`` is an integer sign
# in {-1, 0, +1} -> "int" (the type-token convention for non-float columns);
# ``helix_signed_area_px2`` uses the superscript-2 glyph "px²" (the
# PIPELINE_UNIT_VOCABULARY token — a literal ASCII "px2" is not in vocabulary).
# Every value is a member of
# ``sleap_roots.circumnutation._constants.PIPELINE_UNIT_VOCABULARY``.
_PSIG_TRAIT_UNITS: Dict[str, str] = {
    "T_psig_median_s": "s",
    "delta_E_amplitude_proxy_px_per_frame": "px/frame",
    "handedness": "int",
    "helix_signed_area_px2": "px²",
}

# Per-frame tip columns required on trajectory_df (alongside
# ROW_IDENTITY_COLUMNS).
_TIP_X_COLUMN: str = "tip_x"
_TIP_Y_COLUMN: str = "tip_y"

# Minimum finite frames for a non-degenerate ψ_g (needs ≥ 2 velocity samples).
_MIN_FINITE_FRAMES: int = 3

# Numerical-zero guard for handedness (CC-6 determinism hygiene, NOT a physical
# deadband): a |net ψ_g| below this radian threshold collapses to 0. The
# net-rotation endpoint difference is a raw atan2 quantity (~1e-12 reproducible
# cross-OS), well clear of 1e-9.
_HANDEDNESS_ZERO_EPS_RAD: float = 1e-9


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------


def _validate_psig_constants(constants: ConstantsT) -> ConstantsT:
    """Validate the SG-conditioning fields psi_g consumes, with field-named errors.

    psi_g.compute SG-detrends ψ_g with ``SG_WINDOW_DETREND`` / ``SG_DEGREE``
    (``scipy.signal.savgol_filter`` requires an odd ``window_length >
    polyorder``). The CWT-machinery fields are validated downstream by
    :func:`temporal_cwt.compute_scaleogram` (``_validate_cwt_constants``), so
    this checker is deliberately limited to the SG fields — it does NOT call
    ``_validate_nutation_constants`` (Tier-1 fields psi_g does not use).
    """
    swd = constants.SG_WINDOW_DETREND
    sgd = constants.SG_DEGREE
    if (
        not isinstance(swd, (int, np.integer))
        or isinstance(swd, bool)
        or int(swd) < 1
        or int(swd) % 2 == 0
    ):
        raise ValueError(
            f"constants.SG_WINDOW_DETREND must be a positive odd int "
            f"(scipy.signal.savgol_filter requires odd window_length); got "
            f"SG_WINDOW_DETREND={swd!r}"
        )
    if not isinstance(sgd, (int, np.integer)) or isinstance(sgd, bool) or int(sgd) < 0:
        raise ValueError(
            f"constants.SG_DEGREE must be a non-negative int; got SG_DEGREE={sgd!r}"
        )
    if int(swd) <= int(sgd):
        raise ValueError(
            f"constants.SG_WINDOW_DETREND ({swd}) must be greater than "
            f"constants.SG_DEGREE ({sgd}) "
            f"(scipy.signal.savgol_filter requires window_length > polyorder)"
        )
    return constants


def _check_constants(constants: Any) -> ConstantsT:
    """Validate constants as None or ConstantsT; return the resolved instance.

    Mirrors ``nutation._check_constants`` but defers CWT-field validation to
    :func:`temporal_cwt.compute_scaleogram` and validates only the SG fields
    psi_g consumes (via :func:`_validate_psig_constants`).
    """
    if constants is None:
        return _validate_psig_constants(ConstantsT())
    if not isinstance(constants, ConstantsT):
        raise TypeError(
            f"constants must be None or a ConstantsT instance, got "
            f"{type(constants).__name__}"
        )
    return _validate_psig_constants(constants)


def _all_degenerate_traits() -> Dict[str, Any]:
    """Return the all-degenerate trait row (too-short / undefined track).

    ``handedness`` is the neutral ``0`` (no net rotation); the three float
    traits are NaN (undefined). Used by the ``N < 3`` short-circuit.
    """
    return {
        "T_psig_median_s": float("nan"),
        "delta_E_amplitude_proxy_px_per_frame": float("nan"),
        "handedness": 0,
        "helix_signed_area_px2": float("nan"),
    }


def _linear_detrend(a: np.ndarray) -> np.ndarray:
    """Subtract the least-squares linear trend from a 1-D array.

    Removes the dominant linear growth component from a tip-coordinate axis so
    that the residual reflects the nutation orbit rather than the growth ribbon.
    Used by ``helix_signed_area_px2`` (per-axis) so the Shoelace sign confirms
    handedness on real growing-root data. For ``len(a) < 2`` returns a copy
    (no trend to remove).
    """
    n = len(a)
    if n < 2:
        return np.asarray(a, dtype=np.float64).copy()
    t = np.arange(n, dtype=np.float64)
    slope, intercept = np.polyfit(t, a, 1)
    return a - (slope * t + intercept)


def _compute_t_psig_median_s(
    psi: np.ndarray,
    cadence_s: float,
    constants: ConstantsT,
) -> float:
    """COI-masked median period of the SG-detrended ψ_g via the temporal CWT.

    Composes ``compute_scaleogram → extract_ridge → smooth_ridge`` on the
    SG-detrended ψ_g and returns ``nanmedian`` of the smoothed-ridge periods
    over COI-interior frames (``~smooth_ridge.in_coi``). Returns NaN — without
    a ``RuntimeWarning`` — when the track cannot yield a defined period:

    - **Length guard:** ``len(ψ_g) < SG_WINDOW_DETREND`` (``compute_sg_detrended``
      would return all-NaN → ``compute_scaleogram`` would raise).
    - **Zero-energy guard:** a (near-)constant detrended signal (stationary or
      perfectly straight growth). ``compute_scaleogram`` accepts an all-zero
      signal without raising, and ``argmax``-over-zeros would otherwise yield a
      spurious shortest-period ridge (``2·cadence_s``) with no warning.
    - Empty / all-NaN COI interior.
    """
    window = int(constants.SG_WINDOW_DETREND)
    if len(psi) < window:
        return float("nan")
    psi_detrended = _noise.compute_sg_detrended(
        psi, window=window, polynomial_order=int(constants.SG_DEGREE)
    )
    if not np.isfinite(psi_detrended).any():  # pragma: no cover
        # Defensive mirror of nutation.py:418 — unreachable here because the
        # length guard above already excludes the all-NaN short-input path and
        # compute_psi_g is finite for finite tips.
        return float("nan")
    if np.allclose(psi_detrended, 0.0):
        # Zero-energy guard (stationary / straight-growth track).
        return float("nan")
    try:
        scaleogram = temporal_cwt.compute_scaleogram(
            psi_detrended, cadence_s=cadence_s, constants=constants
        )
    except ValueError:  # pragma: no cover
        # Defensive mirror of nutation.py:422-428 — unreachable because
        # len(psi) >= SG_WINDOW_DETREND=23 > MIN_FRAMES_REQUIRED=9 and the
        # detrended signal is finite 1-D float64.
        return float("nan")
    raw_ridge = temporal_cwt.extract_ridge(scaleogram, constants=constants)
    smoothed_ridge = temporal_cwt.smooth_ridge(raw_ridge, constants=constants)
    interior = smoothed_ridge.periods_s[~smoothed_ridge.in_coi]
    if interior.size == 0 or np.all(np.isnan(interior)):
        return float("nan")
    return float(np.nanmedian(interior))


def _compute_one_track(
    group: pd.DataFrame,
    cadence_s: float,
    constants: ConstantsT,
) -> Dict[str, Any]:
    """Compute the 4 Tier 2 ψ_g traits for one track (per design §3.1).

    Conditioning affects ONLY ``T_psig_median_s``; ``handedness`` (raw
    unwrapped ψ_g endpoints), ``delta_E_amplitude_proxy_px_per_frame`` (raw
    velocity samples), and ``helix_signed_area_px2`` (raw coordinates) are
    CWT-free.
    """
    # Frame-order then drop non-finite tip rows (ordering is load-bearing for
    # the diff-based ψ_g; mirrors kinematics.py:171, extended to ±inf since
    # _geometry.compute_psi_g has no finite guard).
    ordered = group.sort_values("frame")
    tip_x = ordered[_TIP_X_COLUMN].to_numpy(dtype=np.float64)
    tip_y = ordered[_TIP_Y_COLUMN].to_numpy(dtype=np.float64)
    finite = np.isfinite(tip_x) & np.isfinite(tip_y)
    x = tip_x[finite]
    y = tip_y[finite]
    n = int(x.size)

    # Degenerate short-circuit: ψ_g needs ≥ 2 velocity samples (≥ 3 frames).
    if n < _MIN_FINITE_FRAMES:
        return _all_degenerate_traits()

    psi = _geometry.compute_psi_g(x, y)  # length n-1, unwrapped

    # handedness: sign of the net unwrapped rotation over all finite frames
    # (COI-free). 1e-9 rad zero-guard collapses straight/stationary float dust.
    net = float(psi[-1] - psi[0])
    if abs(net) < _HANDEDNESS_ZERO_EPS_RAD:
        handedness = 0
    else:
        handedness = int(np.sign(net))

    # delta_E: median velocity magnitude (px/frame; no COI, no /cadence_s).
    dx = np.diff(x)
    dy = np.diff(y)
    delta_E = float(np.median(np.sqrt(dx * dx + dy * dy)))

    # helix: y-down signed area of the GROWTH-DETRENDED tip trajectory. The raw
    # Shoelace area of an open growing root is dominated by the linear-growth
    # ribbon (sign reflects growth+truncation geometry, not circulation —
    # empirically only 1/6 plate-001 tracks agreed with handedness). Subtracting
    # the per-axis linear growth trend isolates the nutation orbit, so the
    # signed area's sign confirms handedness on genuinely 2-D-circulating data
    # (≥5/6 on plate-001). Note: a planar wobble-on-growth (e.g. the synthetic
    # generator) collapses to ~0 area after detrend (no true 2-D circulation).
    helix = _geometry.compute_signed_area(_linear_detrend(x), _linear_detrend(y))

    # T_psig_median_s: CWT path on the SG-detrended ψ_g (the ONLY conditioned
    # trait). Length + zero-energy + COI guards inside the helper.
    t_psig = _compute_t_psig_median_s(psi, cadence_s, constants)

    return {
        "T_psig_median_s": t_psig,
        "delta_E_amplitude_proxy_px_per_frame": delta_E,
        "handedness": handedness,
        "helix_signed_area_px2": helix,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute(
    trajectory_df: pd.DataFrame,
    cadence_s: float,
    constants: Optional[ConstantsT] = None,
) -> pd.DataFrame:
    """Emit Tier 2 ψ_g traits per track (PR #7, theory.md §3.5 + §7.3).

    The 4 trait columns are emitted in declared order: ``T_psig_median_s``
    (s), ``delta_E_amplitude_proxy_px_per_frame`` (px/frame), ``handedness``
    (int64 {−1, 0, +1}), ``helix_signed_area_px2`` (px²). See the module
    docstring for each trait's definition.

    Unlike ``nutation.compute``, there is no ``coordinate`` parameter: ψ_g is
    computed from the raw 2-D tip trajectory ``(tip_x, tip_y)`` via
    ``_geometry.compute_psi_g`` (``atan2(dx, dy)``), so there is no
    1-D-projection choice.

    Args:
        trajectory_df: Per-frame tip-trajectory DataFrame with the eight
            row-identity columns + ``frame``, ``tip_x``, ``tip_y``.
        cadence_s: Frame cadence in seconds. Positive finite float.
        constants: Optional :class:`ConstantsT` override-bag. ``None``
            (default) resolves to module-level defaults.

    Returns:
        Per-plant DataFrame with 1 row per unique
        ``(series, sample_uid, plate_id, plant_id, track_id)`` 5-tuple.
        Columns: 8 row-identity columns + 4 trait columns in declared order
        (3 ``float64`` + 1 ``int64``).
    """
    # Input validation (mirrors nutation.compute boundary). cadence_s reuses
    # the importable temporal_cwt._validate_cadence_s rather than a private copy.
    if not isinstance(trajectory_df, pd.DataFrame):
        raise ValueError(
            f"trajectory_df must be a pandas DataFrame, got "
            f"{type(trajectory_df).__name__}"
        )
    _validate_trajectory_df(None, None, trajectory_df)
    cadence_float = temporal_cwt._validate_cadence_s(cadence_s)
    resolved_constants = _check_constants(constants)

    n_tracks = trajectory_df[list(_IDENTITY_5_TUPLE)].drop_duplicates().shape[0]
    logger.debug("psi_g.compute(n_tracks=%d, cadence_s=%.6f)", n_tracks, cadence_float)

    # Per-track loop (mirrors kinematics.py / qc.py / nutation.py precedent).
    trait_rows: list = []
    for key, group in trajectory_df.groupby(
        list(_IDENTITY_5_TUPLE), dropna=False, sort=False
    ):
        traits = _compute_one_track(
            group, cadence_s=cadence_float, constants=resolved_constants
        )
        identity = dict(zip(_IDENTITY_5_TUPLE, key))
        trait_rows.append({**identity, **traits})

    trait_df = pd.DataFrame(
        trait_rows,
        columns=list(_IDENTITY_5_TUPLE) + list(_PSIG_TRAIT_COLUMNS),
    )

    # Per-plant template via the shared `_io._build_per_plant_template_from_df`
    # helper (mirrors kinematics.compute / nutation.compute). Coerce trait_df's
    # identity dtypes to match the template's int64 keys BEFORE merge so merges
    # don't silently fall through to all-NaN on numeric-string IDs.
    template = _build_per_plant_template_from_df(trajectory_df)
    for col in ("track_id", "plant_id"):
        if col in trait_df.columns and col in template.columns:
            try:
                trait_df[col] = trait_df[col].astype(template[col].dtype)
            except (TypeError, ValueError) as exc:  # pragma: no cover
                raise ValueError(
                    f"trait_df[{col!r}] cannot be cast to template dtype "
                    f"{template[col].dtype!r}; this would silently break the "
                    f"per-plant merge and produce all-NaN traits."
                ) from exc

    result = template.merge(trait_df, on=list(_IDENTITY_5_TUPLE), how="left")

    # Enforce trait dtypes: 3 float64 + 1 int64. handedness uses fillna(0)
    # before astype(int64) (mirrors nutation's is_nutating fillna(False)
    # defensive pattern) — 0 is the documented neutral value, safe only under
    # the 5-tuple merge invariant (template derives from the same
    # trajectory_df → no unmatched rows in practice).
    for col in _PSIG_TRAIT_COLUMNS:
        if col == "handedness":
            result[col] = result[col].fillna(0).astype(np.int64)
        else:
            result[col] = result[col].astype(np.float64)

    # Enforce declared column order: 8 row-identity + 4 trait.
    result = result[list(ROW_IDENTITY_COLUMNS) + list(_PSIG_TRAIT_COLUMNS)]

    return result
