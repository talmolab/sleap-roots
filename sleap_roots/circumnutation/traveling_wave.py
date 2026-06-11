"""Tier 3c — traveling-wave validation traits (PR #10, ``add-circumnutation-tier3c-traits``).

Public callable: :func:`compute` — emits the Tier 3c traveling-wave validation
traits per track, the first consumer of PR #9's spatial-CWT machinery. Its
headline output is ``traveling_wave_residual``, the program's central
falsifiable test of the QPB steady-traveling-wave hypothesis
``lambda_spatial = v * T_nutation`` (``docs/circumnutation/theory.md`` §4.7).

This is a trait-emission module mirroring :mod:`sleap_roots.circumnutation.nutation`
(Tier 1) and :mod:`sleap_roots.circumnutation.psi_g` (Tier 2): the canonical
signature is ``compute(trajectory_df, cadence_s, constants=None) -> pd.DataFrame``
with a per-track 5-tuple groupby and the 8 row-identity columns. It emits 6
float64 trait columns (declared order):

- ``lambda_spatial_median_px`` — median calibrated spatial wavelength along the
  trail (px).
- ``lambda_spatial_variation`` — orientation-robust spread ``MAD/median`` of the
  along-trail wavelength (dimensionless; 0 = uniform). Reads ~0 on a noise-free
  uniform-λ trail; real-data scatter is noise-dependent.
- ``traveling_wave_residual`` — ``|lambda_spatial − v·T_frames| / (v·T_frames)``,
  the QPB falsification test (dimensionless).
- ``lambda_expected_px`` — ``v · T_frames``, ``T_frames = T_nutation_median /
  cadence_s`` (the steady-wave prediction; px).
- ``lambda_spatial_mad_px`` — absolute MAD of the along-trail wavelength (px;
  numerator of ``lambda_spatial_variation``).
- ``coi_valid_fraction`` — fraction of ridge positions outside the cone of
  influence (dimensionless engineering diagnostic).

Reduced scope (PR #9 descope → GitHub issue #230): the ``L_gz``/``L_c``-dependent
traits and the growth-zone mask are NOT emitted (blocked on #230).

Anchors: spec at
``openspec/changes/add-circumnutation-tier3c-traits/specs/circumnutation/spec.md``;
design at ``openspec/changes/add-circumnutation-tier3c-traits/design.md`` +
``docs/superpowers/specs/2026-06-10-add-circumnutation-tier3c-traits-design.md``;
investigation at
``docs/circumnutation/investigations/2026-06-10-tier3c-traveling-wave/report.md``;
``docs/circumnutation/theory.md`` §4.7 (lambda = v*T), §6.4 (frame-domain unit
reconciliation), §7.4 (Tier 3 trait table + handoff) + Appendix B(6).
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from sleap_roots.circumnutation import (
    kinematics,
    midline,
    nutation,
    spatial_cwt,
    temporal_cwt,
)
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

# 6 trait columns in declared order (per spec ADDED requirement). All float64.
_TRAVELING_WAVE_TRAIT_COLUMNS: tuple = (
    "lambda_spatial_median_px",
    "lambda_spatial_variation",
    "traveling_wave_residual",
    "lambda_expected_px",
    "lambda_spatial_mad_px",
    "coi_valid_fraction",
)

# Per-column units (GitHub issue #222); every value is in PIPELINE_UNIT_VOCABULARY.
_TRAVELING_WAVE_TRAIT_UNITS: Dict[str, str] = {
    "lambda_spatial_median_px": "px",
    "lambda_spatial_variation": "—",
    "traveling_wave_residual": "—",
    "lambda_expected_px": "px",
    "lambda_spatial_mad_px": "px",
    "coi_valid_fraction": "—",
}

# cgau2 over-report calibration: a single ``(lambda_reported_mean, ratio_mean)``
# curve obtained by AVERAGING the per-(n) ratios across n in {200, 400, 600} at
# each lambda_true (the ratio scatters ~7% non-monotonically across n; averaging
# avoids the false precision of interpolating that, and books a documented ~±5%
# systematic). Generated VERBATIM (full-precision tokens) from the authoritative
# tests/data/circumnutation_spatial_cwt_calibration.json n-average — the module
# does NOT read tests/data at runtime (wheel-safe). Validated by
# test_in_package_calibration_literal_matches_n_averaged_json. lambda_reported_mean
# is strictly increasing (well-posed np.interp) and covers lambda_true up to 150
# px (lambda_reported ~167 px) so the observed real lambda ~142.5 px is in range.
_CGAU2_LAMBDA_CALIBRATION: tuple = (
    (21.75218773026324, 1.087609386513162),
    (31.99480199096912, 1.0664933996989705),
    (44.06824333041771, 1.1017060832604426),
    (54.570013065987204, 1.091400261319744),
    (66.5687339298172, 1.1094788988302866),
    (90.07371215762747, 1.1259214019703434),
    (109.88388028063677, 1.0988388028063678),
    (131.82390622027648, 1.0985325518356372),
    (157.11793542349366, 1.122270967310669),
    (166.93309102248887, 1.112887273483259),
)

# Precomputed interpolation axes (np.interp clamps beyond the table edges).
_CALIB_AXIS: np.ndarray = np.array(
    [p[0] for p in _CGAU2_LAMBDA_CALIBRATION], dtype=np.float64
)
_CALIB_RATIO: np.ndarray = np.array(
    [p[1] for p in _CGAU2_LAMBDA_CALIBRATION], dtype=np.float64
)


def _calibrate_wavelengths(wavelengths_px: np.ndarray) -> np.ndarray:
    """Convert cgau2-over-reported wavelengths to true px via the n-averaged curve.

    ``lambda_true = lambda_reported / ratio(lambda_reported)``, with ``ratio``
    interpolated on the strictly-increasing ``lambda_reported_mean`` axis of
    :data:`_CGAU2_LAMBDA_CALIBRATION` (clamps at the table edges).
    """
    ratio = np.interp(wavelengths_px, _CALIB_AXIS, _CALIB_RATIO)
    return wavelengths_px / ratio


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------


def _check_constants(constants: Any) -> ConstantsT:
    """Validate constants as None or ConstantsT; return the resolved instance.

    Validates ``COI_FRACTION_MAX`` locally (the field this module consumes) as a
    float in ``(0, 1]`` with a field-named error. The Tier 0/Tier 1 constant
    fields are re-validated by the inner ``kinematics.compute`` /
    ``nutation.compute`` calls.
    """
    if constants is None:
        resolved = ConstantsT()
    elif isinstance(constants, ConstantsT):
        resolved = constants
    else:
        raise TypeError(
            f"constants must be None or a ConstantsT instance, got "
            f"{type(constants).__name__}"
        )
    coi = resolved.COI_FRACTION_MAX
    if not isinstance(coi, (int, float, np.integer, np.floating)) or isinstance(
        coi, (bool, np.bool_)
    ):
        raise ValueError(f"COI_FRACTION_MAX must be a float in (0, 1], got {coi!r}")
    coi_float = float(coi)
    if not np.isfinite(coi_float) or coi_float <= 0.0 or coi_float > 1.0:
        raise ValueError(
            f"COI_FRACTION_MAX must be a float in (0, 1], got {coi_float!r}"
        )
    return resolved


def _all_nan_spatial_traits() -> Dict[str, float]:
    """Return a fully-NaN trait dict (used for tracks with no usable ridge).

    Every one of the 6 trait keys is present so the per-plant merge always
    yields one well-formed row per 5-tuple. ``coi_valid_fraction`` is NaN here
    (no ridge formed); when a ridge DOES form it is populated even if the
    low-COI gate fires (see :func:`_compute_one_track`).
    """
    return {col: float("nan") for col in _TRAVELING_WAVE_TRAIT_COLUMNS}


def _ridge_to_traits(ridge, constants: ConstantsT) -> Dict[str, float]:
    """Gate on the COI and compute the calibrated along-trail wavelength stats.

    A ridge formed, so ``coi_valid_fraction`` is finite. The COI gate NaNs the 3
    wavelength stats when the COI-valid fraction is below ``1 − COI_FRACTION_MAX``
    (strict inequality: an exactly-``COI_FRACTION_MAX`` in-COI fraction does NOT
    gate), while ``coi_valid_fraction`` stays populated so it diagnoses why a row
    gated. Wavelengths are calibrated to true px before the statistics.
    """
    traits = _all_nan_spatial_traits()
    n_total = int(ridge.in_coi.size)
    if n_total == 0:
        return traits
    interior = ~ridge.in_coi
    n_interior = int(interior.sum())
    coi_valid_fraction = n_interior / n_total
    traits["coi_valid_fraction"] = float(coi_valid_fraction)

    if coi_valid_fraction < (1.0 - float(constants.COI_FRACTION_MAX)):
        return traits  # COI gate fired: wavelength stats NaN, fraction finite
    if n_interior == 0:
        return traits

    lam = _calibrate_wavelengths(ridge.wavelengths_px[interior])
    median = float(np.median(lam))
    mad = float(np.median(np.abs(lam - median)))
    traits["lambda_spatial_median_px"] = median
    traits["lambda_spatial_mad_px"] = mad
    traits["lambda_spatial_variation"] = mad / median if median > 0.0 else float("nan")
    return traits


def _compute_one_track(
    group: pd.DataFrame,
    *,
    cadence_s: float,
    constants: ConstantsT,
) -> Dict[str, float]:
    """Compute the spatial Tier 3c traits for a single track's per-frame rows.

    Runs the per-track spatial chain ``midline.reconstruct`` →
    ``resample_curvature`` → ``compute_scaleogram`` → ``extract_ridge``, gates on
    the cone of influence, and computes the along-trail wavelength statistics.
    Always returns a complete 6-key dict and never raises: degenerate, short, or
    all-NaN-tip tracks emit an all-NaN row, and a ridge that forms still
    populates ``coi_valid_fraction`` even when the COI gate fires.

    Wavelengths are calibrated to true px (:func:`_ridge_to_traits`); the
    ``lambda_expected_px`` / ``traveling_wave_residual`` composition is computed
    in :func:`compute` from the joined Tier 0/Tier 1 operands (those two columns
    remain NaN in this per-track dict).
    """
    traits = _all_nan_spatial_traits()

    sub = group.sort_values("frame")
    x = sub["tip_x"].to_numpy(dtype=np.float64)
    y = sub["tip_y"].to_numpy(dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]

    try:
        mr = midline.reconstruct(x, y, cadence_s=cadence_s, constants=constants)
    except (ValueError, TypeError):
        return traits
    if mr.is_degenerate:
        return traits

    rs = spatial_cwt.resample_curvature(
        mr.curvature_px_inv,
        mr.arc_length_px,
        mr.velocity_sub_noise_mask,
        constants=constants,
    )
    if rs.is_degenerate:
        return traits

    try:
        scaleogram = spatial_cwt.compute_scaleogram(
            rs.kappa_uniform, rs.ds, constants=constants
        )
        ridge = spatial_cwt.extract_ridge(scaleogram, constants=constants)
    except ValueError:
        return traits

    # A ridge formed → gate on the COI and compute the calibrated wavelength stats.
    return _ridge_to_traits(ridge, constants)


def compute(
    trajectory_df: pd.DataFrame,
    cadence_s: float,
    constants: Optional[ConstantsT] = None,
) -> pd.DataFrame:
    """Compute Tier 3c traveling-wave validation traits for each track.

    Args:
        trajectory_df: Per-frame tip-trajectory DataFrame containing the eight
            row-identity columns plus ``frame``, ``tip_x``, ``tip_y``.
        cadence_s: Sampling cadence in seconds per frame (positive finite).
        constants: Optional :class:`ConstantsT` override-bag. When ``None``
            (default), module-level defaults are used. ``COI_FRACTION_MAX`` is
            validated locally as a float in ``(0, 1]``.

    Returns:
        Per-plant DataFrame with one row per unique
        ``(series, sample_uid, plate_id, plant_id, track_id)`` 5-tuple: the 8
        row-identity columns followed by the 6 Tier 3c trait columns (all
        float64) in the declared order.

    Raises:
        ValueError: If ``trajectory_df`` is empty / missing required columns,
            ``cadence_s`` is non-positive or non-finite, or ``COI_FRACTION_MAX``
            is outside ``(0, 1]``.
        TypeError: If ``cadence_s`` is a bool/str/list, or ``constants`` is not
            ``None`` / a ``ConstantsT`` instance.
    """
    if not isinstance(trajectory_df, pd.DataFrame):
        raise ValueError(
            f"trajectory_df must be a pandas DataFrame, got "
            f"{type(trajectory_df).__name__}"
        )
    _validate_trajectory_df(None, None, trajectory_df)
    cadence_float = temporal_cwt._validate_cadence_s(cadence_s)
    resolved_constants = _check_constants(constants)

    trait_rows: list = []
    n_tracks = trajectory_df[list(_IDENTITY_5_TUPLE)].drop_duplicates().shape[0]
    logger.debug(
        "traveling_wave.compute(n_tracks=%d, cadence_s=%.6f)",
        n_tracks,
        cadence_float,
    )
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
        columns=list(_IDENTITY_5_TUPLE) + list(_TRAVELING_WAVE_TRAIT_COLUMNS),
    )

    # Per-plant template via the shared helper (mirrors nutation/psi_g). Coerce
    # trait_df identity dtypes to the template's int64 keys before merging so a
    # numeric-string / float key cannot silently fall through to all-NaN.
    template = _build_per_plant_template_from_df(trajectory_df)
    for col in ("track_id", "plant_id"):
        if col in trait_df.columns and col in template.columns:
            try:
                trait_df[col] = trait_df[col].astype(template[col].dtype)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"trait_df[{col!r}] cannot be cast to template dtype "
                    f"{template[col].dtype!r}; this would silently break the "
                    f"per-plant merge and produce all-NaN traits."
                ) from exc

    result = template.merge(trait_df, on=list(_IDENTITY_5_TUPLE), how="left")

    # Compose with Tier 0 (v) and Tier 1 (T_nutation), recomputed self-contained
    # (redundant by design — the PR #14 pipeline DAG MUST dedup Tier 0/Tier 1
    # rather than calling this naively). Join on the FULL _IDENTITY_5_TUPLE with
    # int64 coercion: track_id is not unique across plates, and an int64-vs-float64
    # key mismatch would SILENTLY produce all-NaN operands (not a KeyError).
    tier0 = kinematics.compute(trajectory_df, constants=resolved_constants)
    tier1 = nutation.compute(
        trajectory_df,
        cadence_float,
        coordinate="lateral",
        constants=resolved_constants,
    )
    ops = tier0[list(_IDENTITY_5_TUPLE) + ["v_total_median_px_per_frame"]].merge(
        tier1[list(_IDENTITY_5_TUPLE) + ["T_nutation_median"]],
        on=list(_IDENTITY_5_TUPLE),
        how="outer",
    )
    for frame in (result, ops):
        for col in ("track_id", "plant_id"):
            frame[col] = frame[col].astype(np.int64)
    result = result.merge(ops, on=list(_IDENTITY_5_TUPLE), how="left")

    # lambda_expected_px = v · T_frames, T_frames = T_nutation_median / cadence_s
    # (T_nutation_median is seconds; it is NaN when the track is not nutating, so
    # the residual NaNs naturally for non-nutating tracks). Division guard: the
    # residual is interpretable only when v is finite and the predicted wavelength
    # is positive (a sub-wavelength v·T is an undefined regime, not a QPB result).
    v = result["v_total_median_px_per_frame"].to_numpy(dtype=np.float64)
    t_frames = result["T_nutation_median"].to_numpy(dtype=np.float64) / cadence_float
    expected = v * t_frames
    valid = np.isfinite(v) & np.isfinite(expected) & (expected > 0.0)
    expected = np.where(valid, expected, np.nan)
    median = result["lambda_spatial_median_px"].to_numpy(dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        residual = np.abs(median - expected) / expected
    result["lambda_expected_px"] = expected
    result["traveling_wave_residual"] = residual

    # Enforce trait dtypes: all 6 float64 (no bool/int special case).
    for col in _TRAVELING_WAVE_TRAIT_COLUMNS:
        result[col] = result[col].astype(np.float64)

    # Enforce declared column order: 8 row-identity + 6 trait.
    result = result[list(ROW_IDENTITY_COLUMNS) + list(_TRAVELING_WAVE_TRAIT_COLUMNS)]

    return result
