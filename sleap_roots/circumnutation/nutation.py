"""Tier 1 nutation trait emission (PR #6, ``add-circumnutation-tier1-derr-faithful``).

Public callable: :func:`compute` — emits 8 trait columns per track
(``T_nutation_median``, ``T_nutation_iqr``, ``A_nutation_envelope_max``,
``band_power_ratio``, ``noise_floor_estimate``, ``is_nutating``,
``period_residual_vs_derr_reference``, ``cadence_nyquist_ratio``) by
composing PR #5 CWT primitives with PR #6's new helpers:

- :func:`sleap_roots.circumnutation._geometry.project_to_growth_axis_perpendicular`
  (CC-7 lateral projection of ``(tip_x, tip_y)`` → 1D lateral signal)
- :func:`sleap_roots.circumnutation._noise.compute_sg_detrended` (S1
  round-1: SG-detrend per ``preliminary_results §3.4``; removes
  centerline drift before CWT/FFT)
- :func:`sleap_roots.circumnutation.temporal_cwt.compute_scaleogram` +
  :func:`extract_ridge` (PR #5 primitives)
- :func:`sleap_roots.circumnutation.temporal_cwt.smooth_ridge` (closes
  #214: ``scipy.ndimage.median_filter`` post-filter)
- :func:`sleap_roots.circumnutation._noise.compute_fourier_noise_floor`
  (CC-8: median FFT amplitude in out-of-band region)

Anchors: spec at
``openspec/changes/add-circumnutation-tier1-derr-faithful/specs/circumnutation/spec.md``;
design at ``openspec/changes/add-circumnutation-tier1-derr-faithful/design.md``
D1–D9 + R1–R6 + Reconciliation Appendices for rounds 1 and 2;
``docs/circumnutation/theory.md`` §3.5 (BM2016 Eq. 20 — ψ_g foundation),
§6.5 (Cadence-Nyquist), §7.2 (Tier 1 trait table), §7.6 (QC:
``is_nutating``, ``noise_floor_estimate``); ``docs/circumnutation/roadmap.md``
CC-7 (lateral coordinate) + CC-8 (Fourier noise floor);
``docs/circumnutation/preliminary_results_2026-05-07.md`` §3.4
(SG-detrend prescription) + §4.4 (T_nutation ≈ 3333 s on plate-001).

NaN-gating semantics (S4 round-1 + Sci-I3 round-2): when ``is_nutating ==
False``, 3 strictly biological-meaning-dependent traits become NaN
(``T_nutation_median``, ``T_nutation_iqr``, ``A_nutation_envelope_max``).
5 always-populated traits remain finite: ``is_nutating`` (the gate),
``band_power_ratio`` + ``noise_floor_estimate`` (precursors),
``period_residual_vs_derr_reference`` (ridge-of-noise diagnostic),
``cadence_nyquist_ratio`` (engineering diagnostic: "could we have
observed nutation if it were present?" — answerable regardless of
biological nutation presence).

Closes GitHub issue #214 (ridge-tracking continuity post-filter via
the new :func:`temporal_cwt.smooth_ridge` primitive).
"""

import logging
import math
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import scipy.fft
import scipy.stats

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

# 5-tuple groupby key (subset of ROW_IDENTITY_COLUMNS that uniquely
# identifies a track within a series). Imported from _io to match the
# canonical PR #1 source-of-truth used by kinematics.py / qc.py.

# 8 trait columns in declared order (per spec ADDED requirement).
_NUTATION_TRAIT_COLUMNS: tuple = (
    "T_nutation_median",
    "T_nutation_iqr",
    "A_nutation_envelope_max",
    "band_power_ratio",
    "noise_floor_estimate",
    "is_nutating",
    "period_residual_vs_derr_reference",
    "cadence_nyquist_ratio",
)

# S4 round-1 + Sci-I3 round-2: NaN-gated when is_nutating==False.
_NAN_GATED_TRAITS: tuple = (
    "T_nutation_median",
    "T_nutation_iqr",
    "A_nutation_envelope_max",
)

# Per-frame columns required on trajectory_df (alongside
# ROW_IDENTITY_COLUMNS).
_TIP_X_COLUMN: str = "tip_x"
_TIP_Y_COLUMN: str = "tip_y"

# Valid coordinate= values per CC-7.
_COORDINATE_CHOICES: frozenset = frozenset({"lateral", "x", "y"})


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------


def _check_cadence_s(cadence_s: Any) -> float:
    """Validate cadence_s as positive finite float (PR #5 _validate_cadence_s pattern).

    Accept: Python int/float, numpy integer/floating scalar. Reject: bool
    (Python and numpy), str, list, complex, tuple, NaN/±inf, non-positive.
    """
    if isinstance(cadence_s, bool):
        raise TypeError(
            f"cadence_s must be a positive finite float "
            f"(rejected Python bool {cadence_s!r})"
        )
    if isinstance(cadence_s, np.bool_):
        raise TypeError(
            f"cadence_s must be a positive finite float "
            f"(rejected numpy bool {cadence_s!r})"
        )
    if not isinstance(cadence_s, (int, float, np.integer, np.floating)):
        raise TypeError(
            f"cadence_s must be a positive finite float, got "
            f"{type(cadence_s).__name__}"
        )
    cadence_float = float(cadence_s)
    if not math.isfinite(cadence_float):
        raise ValueError(
            f"cadence_s must be a positive finite float, got {cadence_s!r}"
        )
    if cadence_float <= 0:
        raise ValueError(f"cadence_s must be positive, got cadence_s={cadence_float}")
    return cadence_float


def _check_coordinate(coordinate: Any) -> str:
    """Validate coordinate as one of {'lateral', 'x', 'y'}."""
    if not isinstance(coordinate, str):
        raise ValueError(
            f"coordinate must be a string from "
            f"{sorted(_COORDINATE_CHOICES)!r}, got "
            f"{type(coordinate).__name__}"
        )
    if coordinate not in _COORDINATE_CHOICES:
        raise ValueError(
            f"coordinate must be one of {sorted(_COORDINATE_CHOICES)!r}, "
            f"got coordinate={coordinate!r}"
        )
    return coordinate


def _check_constants(constants: Any) -> ConstantsT:
    """Validate constants as None or ConstantsT; return resolved instance."""
    if constants is None:
        return ConstantsT()
    if not isinstance(constants, ConstantsT):
        raise TypeError(
            f"constants must be None or a ConstantsT instance, got "
            f"{type(constants).__name__}"
        )
    return constants


# ---------------------------------------------------------------------------
# Signal pipeline helpers
# ---------------------------------------------------------------------------


def _select_signal(group: pd.DataFrame, coordinate: str) -> np.ndarray:
    """Dispatch on coordinate ∈ {'lateral', 'x', 'y'} → 1D float64 signal."""
    tip_x = group[_TIP_X_COLUMN].to_numpy(dtype=np.float64)
    tip_y = group[_TIP_Y_COLUMN].to_numpy(dtype=np.float64)
    if coordinate == "x":
        return tip_x
    if coordinate == "y":
        return tip_y
    # coordinate == "lateral"
    return _geometry.project_to_growth_axis_perpendicular(tip_x, tip_y)


def _compute_band_power_traits(
    x: np.ndarray,
    cadence_s: float,
    t_nutation_median_s: float,
    constants: ConstantsT,
) -> tuple:
    """Compute (band_power_ratio, in_band_mean_amplitude) for trait + gate.

    Per design.md D7 + theory.md §7.2 ``[0.5T, 2T]`` band. Uses
    ``scipy.fft.rfft`` for dimensional consistency with
    ``noise_floor_estimate``.

    GREEN-phase Reconciliation (Sci-I1 round-2 empirically confirmed):
    theory.md §7.6's literal gate ``band_power_ratio > BAND_POWER_NOISE_RATIO
    * noise_floor_estimate`` is dimensionally inconsistent — left side is
    a dimensionless ratio in [0, 1]; right side is a constant times an
    amplitude (typically ~1-10 px on plate-001), so the gate NEVER fires
    on legitimate signals. We compute TWO quantities:

    - ``band_power_ratio``: emitted as the trait (per spec, in [0, 1])
    - ``in_band_mean_amplitude``: used ONLY for the gate, in amplitude
      units (same as ``noise_floor_estimate``) → gate is dimensionally
      consistent

    The EMITTED trait matches the spec verbatim. The internal gate uses
    a dimensionally-correct comparison. See GREEN-phase Reconciliation
    Appendix in design.md for the full rationale.
    """
    if len(x) < 2:
        return float("nan"), float("nan")
    if not math.isfinite(t_nutation_median_s) or t_nutation_median_s <= 0:
        return float("nan"), float("nan")
    spectrum = np.abs(scipy.fft.rfft(x))
    freqs = scipy.fft.rfftfreq(len(x), d=cadence_s)
    f_low = constants.BAND_POWER_BAND_LOW_FACTOR / t_nutation_median_s
    f_high = constants.BAND_POWER_BAND_HIGH_FACTOR / t_nutation_median_s
    in_band_mask = (freqs >= f_low) & (freqs <= f_high)
    total_power = float(np.sum(spectrum**2))
    if total_power == 0.0 or not math.isfinite(total_power):
        return float("nan"), float("nan")
    band_power = float(np.sum(spectrum[in_band_mask] ** 2))
    ratio = band_power / total_power
    # Clamp to [0, 1] to absorb any floating-point ulps that push slightly out.
    band_power_ratio = float(min(max(ratio, 0.0), 1.0))
    # In-band mean amplitude for the dimensionally-consistent gate.
    if in_band_mask.any():
        in_band_mean_amplitude = float(np.mean(spectrum[in_band_mask]))
    else:
        in_band_mean_amplitude = float("nan")
    return band_power_ratio, in_band_mean_amplitude


def _all_nan_trait_dict(is_nutating_value: bool = False) -> Dict[str, Any]:
    """Build a fully-NaN/False trait dict (used for stationary tracks per MEC9)."""
    return {
        "T_nutation_median": float("nan"),
        "T_nutation_iqr": float("nan"),
        "A_nutation_envelope_max": float("nan"),
        "band_power_ratio": float("nan"),
        "noise_floor_estimate": float("nan"),
        "is_nutating": bool(is_nutating_value),
        "period_residual_vs_derr_reference": float("nan"),
        "cadence_nyquist_ratio": float("nan"),
    }


def _compute_one_track(
    group: pd.DataFrame,
    cadence_s: float,
    coordinate: str,
    constants: ConstantsT,
) -> Dict[str, Any]:
    """Per-track 9-step pipeline (design.md D5).

    Order:
      0. (MEC9 round-1) All-NaN signal short-circuit BEFORE SG-detrend
      1. Lateral projection (or x / y per coordinate=)
      1b. SG-detrend per preliminary_results §3.4 (S1 round-1)
      2. CWT primitives: compute_scaleogram → extract_ridge → smooth_ridge
      3. T_nutation_median + T_nutation_iqr from COI-masked smoothed periods
      4. A_nutation_envelope_max from COI-masked raw ridge amplitudes
         (MEC8 round-1: empty-slice fallback)
      5. noise_floor_estimate via FFT (CC-8)
      6. band_power_ratio via FFT
      7. is_nutating gate
      8. Derived traits (period_residual_vs_derr_reference, cadence_nyquist_ratio)
      9. NaN-gate 3 traits when is_nutating==False (S4 round-1)
    """
    # Step 1: project to signal axis.
    raw_signal = _select_signal(group, coordinate)

    # Step 0 (MEC9 round-1): all-NaN short-circuit BEFORE SG-detrend.
    # _geometry.project_to_growth_axis_perpendicular returns all-NaN on
    # zero net displacement; downstream savgol_filter on all-NaN
    # propagates NaN AND raises RuntimeWarning. Short-circuit here.
    if not np.isfinite(raw_signal).any():
        return _all_nan_trait_dict(is_nutating_value=False)

    # Step 1b (S1 round-1): SG-detrend the lateral signal per
    # preliminary_results §3.4. Returns all-NaN if len(signal) < window;
    # short-circuit again to avoid downstream NaN propagation.
    signal = _noise.compute_sg_detrended(
        raw_signal,
        window=int(constants.SG_WINDOW_DETREND),
        polynomial_order=int(constants.SG_DEGREE),
    )
    if not np.isfinite(signal).any():
        return _all_nan_trait_dict(is_nutating_value=False)

    # Step 2: CWT primitives (PR #5 + PR #6 smooth_ridge).
    try:
        scaleogram = temporal_cwt.compute_scaleogram(
            signal, cadence_s=cadence_s, constants=constants
        )
    except ValueError:
        # Signal too short for the CWT scale grid → emit NaN row.
        return _all_nan_trait_dict(is_nutating_value=False)
    raw_ridge = temporal_cwt.extract_ridge(scaleogram, constants=constants)
    smoothed_ridge = temporal_cwt.smooth_ridge(raw_ridge, constants=constants)

    # Step 3: COI-masked period statistics from SMOOTHED ridge.
    interior_periods = smoothed_ridge.periods_s[~smoothed_ridge.in_coi]
    if interior_periods.size == 0:
        # Entire ridge in COI — period statistics undefined.
        T_nutation_median_candidate = float("nan")
        T_nutation_iqr_candidate = float("nan")
    else:
        T_nutation_median_candidate = float(np.nanmedian(interior_periods))
        T_nutation_iqr_candidate = float(
            scipy.stats.iqr(interior_periods, nan_policy="omit")
        )

    # Step 4 (MEC8 round-1): COI-masked amplitude peak from RAW ridge,
    # with empty-slice fallback.
    interior_amps = raw_ridge.amplitudes[~raw_ridge.in_coi]
    if interior_amps.size == 0:
        A_nutation_envelope_max_candidate = float("nan")
    else:
        A_nutation_envelope_max_candidate = float(np.max(interior_amps))

    # Step 5: noise_floor_estimate via FFT (CC-8). Uses the CANDIDATE T
    # (pre-NaN-gating) so the noise floor is well-defined regardless of
    # the gate decision.
    noise_floor_estimate = _noise.compute_fourier_noise_floor(
        signal,
        cadence_s=cadence_s,
        t_nutation_median_s=T_nutation_median_candidate,
        factor=float(constants.NOISE_FLOOR_OUT_OF_BAND_FACTOR),
    )

    # Step 6: band_power_ratio (trait) + in_band_mean_amplitude (gate).
    # GREEN-phase Sci-I1 round-2 reconciliation: gate uses amplitude-based
    # comparison so both sides are dimensionally consistent (amplitude
    # units). band_power_ratio remains the spec-defined trait in [0, 1].
    band_power_ratio, in_band_mean_amplitude = _compute_band_power_traits(
        signal,
        cadence_s=cadence_s,
        t_nutation_median_s=T_nutation_median_candidate,
        constants=constants,
    )

    # Step 7: is_nutating gate (dimensionally-consistent variant per
    # GREEN-phase Sci-I1 round-2 reconciliation; see _compute_band_power_traits
    # docstring for the rationale of using in-band-mean-amplitude vs the
    # literal band_power_ratio).
    if (
        math.isfinite(in_band_mean_amplitude)
        and math.isfinite(noise_floor_estimate)
        and noise_floor_estimate >= 0
    ):
        is_nutating = bool(
            in_band_mean_amplitude
            > float(constants.BAND_POWER_NOISE_RATIO) * noise_floor_estimate
        )
    else:
        is_nutating = False

    # Step 8: derived traits — ALWAYS computed (S4 round-1 + Sci-I3 round-2).
    derr_period = float(constants.DERR_EXPECTED_PERIOD_S)
    if math.isfinite(T_nutation_median_candidate) and derr_period > 0:
        period_residual_candidate = float(
            (T_nutation_median_candidate - derr_period) / derr_period
        )
    else:
        period_residual_candidate = float("nan")
    if math.isfinite(T_nutation_median_candidate) and T_nutation_median_candidate > 0:
        cadence_nyquist_ratio_candidate = float(cadence_s / T_nutation_median_candidate)
    else:
        cadence_nyquist_ratio_candidate = float("nan")

    # Step 9: NaN-gate the 3 strictly biological-meaning-dependent traits
    # when is_nutating == False (S4 round-1 + TDD-B1 round-2). The 5
    # always-populated traits remain finite.
    if is_nutating:
        T_nutation_median = T_nutation_median_candidate
        T_nutation_iqr = T_nutation_iqr_candidate
        A_nutation_envelope_max = A_nutation_envelope_max_candidate
    else:
        T_nutation_median = float("nan")
        T_nutation_iqr = float("nan")
        A_nutation_envelope_max = float("nan")

    return {
        "T_nutation_median": T_nutation_median,
        "T_nutation_iqr": T_nutation_iqr,
        "A_nutation_envelope_max": A_nutation_envelope_max,
        "band_power_ratio": float(band_power_ratio),
        "noise_floor_estimate": float(noise_floor_estimate),
        "is_nutating": is_nutating,
        "period_residual_vs_derr_reference": period_residual_candidate,
        "cadence_nyquist_ratio": cadence_nyquist_ratio_candidate,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute(
    trajectory_df: pd.DataFrame,
    cadence_s: float,
    coordinate: str = "lateral",
    constants: Optional[ConstantsT] = None,
) -> pd.DataFrame:
    """Emit Tier 1 nutation traits per track (PR #6, theory.md §7.2 + §7.6 + §6.5).

    The 8 trait columns are emitted in declared order:

    - ``T_nutation_median``: median nutation period (s); NaN-gated when
      ``is_nutating==False``.
    - ``T_nutation_iqr``: IQR of nutation period (s); NaN-gated when
      ``is_nutating==False``.
    - ``A_nutation_envelope_max``: peak envelope amplitude (px);
      NaN-gated when ``is_nutating==False``.
    - ``band_power_ratio``: spectral power in
      ``[BAND_POWER_BAND_LOW_FACTOR/T, BAND_POWER_BAND_HIGH_FACTOR/T]``
      band / total power (dimensionless ∈ [0, 1]); always populated.
    - ``noise_floor_estimate``: median Fourier amplitude over
      ``f > NOISE_FLOOR_OUT_OF_BAND_FACTOR / T`` (amplitude units);
      always populated.
    - ``is_nutating``: ``in_band_mean_amplitude > BAND_POWER_NOISE_RATIO *
      noise_floor_estimate`` (bool); the gate; always populated.
      GREEN-phase Sci-I1 reconciliation: the comparison uses
      ``in_band_mean_amplitude`` (mean of ``|rfft(signal)|`` over the
      ``[BAND_POWER_BAND_LOW_FACTOR/T, BAND_POWER_BAND_HIGH_FACTOR/T]``
      band) for dimensional consistency with ``noise_floor_estimate``
      (both amplitude units). theory.md §7.6's literal formula
      ``band_power_ratio > BAND_POWER_NOISE_RATIO * noise_floor_estimate``
      is dimensionally inconsistent (ratio vs amplitude×constant) and was
      empirically refuted on a pure sinusoid. The emitted
      ``band_power_ratio`` trait remains spec-defined; only the internal
      gate uses ``in_band_mean_amplitude``.
    - ``period_residual_vs_derr_reference``:
      ``(T - DERR_EXPECTED_PERIOD_S) / DERR_EXPECTED_PERIOD_S``
      (dimensionless); ridge-of-noise diagnostic, always populated.
    - ``cadence_nyquist_ratio``: ``cadence_s / T_nutation_median``
      (dimensionless); engineering diagnostic, always populated.

    NaN-gating policy (S4 round-1 + Sci-I3 round-2): when ``is_nutating
    == False``, 3 strictly biological-meaning-dependent traits become
    NaN (T_nutation_median, T_nutation_iqr, A_nutation_envelope_max).
    The 5 always-populated traits (``is_nutating``,
    ``band_power_ratio``, ``noise_floor_estimate``,
    ``period_residual_vs_derr_reference``, ``cadence_nyquist_ratio``)
    are NOT NaN-gated by ``is_nutating`` — but they MAY still be NaN
    when the underlying diagnostic is undefined (e.g., stationary
    tracks, all-COI ridge, empty out-of-band Fourier region). The
    semantic is "not gated by ``is_nutating``", not "guaranteed finite".
    This split lets QC investigations distinguish "no oscillation"
    from "cadence aliasing" from "ridge-of-noise" when the diagnostics
    are defined.

    Pipeline (per design.md D5):

    1. ``_geometry.project_to_growth_axis_perpendicular`` (CC-7 lateral
       projection) → 1D lateral position
    2. ``_noise.compute_sg_detrended`` (preliminary_results §3.4
       prescription: window=SG_WINDOW_DETREND=23, polynomial_order=
       SG_DEGREE=3) → oscillation residual
    3. ``temporal_cwt.compute_scaleogram`` → CWT scaleogram
    4. ``temporal_cwt.extract_ridge`` → per-frame argmax ridge
    5. ``temporal_cwt.smooth_ridge`` → ridge-continuity post-filter
       (closes #214)
    6. ``_noise.compute_fourier_noise_floor`` (CC-8) → noise_floor
    7. Internal band-power FFT integration → band_power_ratio
    8. ``is_nutating = band_power_ratio > BAND_POWER_NOISE_RATIO *
       noise_floor_estimate``
    9. Derived traits (period_residual_vs_derr_reference,
       cadence_nyquist_ratio) ALWAYS computed; 3 NaN-gated when
       is_nutating==False

    Args:
        trajectory_df: Per-frame tip-trajectory DataFrame with the eight
            row-identity columns + ``frame``, ``tip_x``, ``tip_y``.
        cadence_s: Frame cadence in seconds. Positive finite float.
            S8' round-2: explicit positional parameter (NOT carried via
            ``trajectory_df.attrs``) — mirrors
            ``temporal_cwt.compute_scaleogram``'s precedent.
        coordinate: Which 1D time series feeds the temporal CWT. Default
            ``"lateral"`` per CC-7 (perpendicular to growth axis). Also
            accepts ``"x"`` / ``"y"`` for diagnostic raw-coordinate use.
        constants: Optional :class:`ConstantsT` override-bag. ``None``
            (default) resolves to module-level defaults.

    Returns:
        Per-plant DataFrame with 1 row per unique
        ``(series, sample_uid, plate_id, plant_id, track_id)`` 5-tuple.
        Columns: 8 row-identity columns + 8 trait columns in declared
        order (7 ``float64`` + 1 ``bool``).

    Raises:
        ValueError: If ``trajectory_df`` is invalid (delegates to
            ``_validate_trajectory_df``); if ``cadence_s`` is non-positive
            or non-finite; if ``coordinate`` is not in {'lateral', 'x',
            'y'}.
        TypeError: If ``trajectory_df`` is not a ``pd.DataFrame``; if
            ``cadence_s`` is bool/str/list/etc; if ``constants`` is not
            ``None`` or a ``ConstantsT`` instance.
    """
    # Input validation.
    if not isinstance(trajectory_df, pd.DataFrame):
        raise ValueError(
            f"trajectory_df must be a pandas DataFrame, got "
            f"{type(trajectory_df).__name__}"
        )
    _validate_trajectory_df(None, None, trajectory_df)
    cadence_float = _check_cadence_s(cadence_s)
    coordinate_resolved = _check_coordinate(coordinate)
    resolved_constants = _check_constants(constants)

    # Per-track loop (mirrors kinematics.py / qc.py precedent).
    trait_rows: list = []
    n_tracks = trajectory_df[list(_IDENTITY_5_TUPLE)].drop_duplicates().shape[0]
    logger.debug(
        "nutation.compute(n_tracks=%d, coordinate=%r, cadence_s=%.6f)",
        n_tracks,
        coordinate_resolved,
        cadence_float,
    )
    for key, group in trajectory_df.groupby(
        list(_IDENTITY_5_TUPLE), dropna=False, sort=False
    ):
        traits = _compute_one_track(
            group,
            cadence_s=cadence_float,
            coordinate=coordinate_resolved,
            constants=resolved_constants,
        )
        identity = dict(zip(_IDENTITY_5_TUPLE, key))
        trait_rows.append({**identity, **traits})

    trait_df = pd.DataFrame(
        trait_rows,
        columns=list(_IDENTITY_5_TUPLE) + list(_NUTATION_TRAIT_COLUMNS),
    )

    # Per-plant template via the shared `_io._build_per_plant_template_from_df`
    # helper (Copilot review on PR #216, comments at nutation.py:529 + 541):
    # mirrors kinematics.compute / qc.compute precedent. Inherits the helper's
    # validations (NaN track_id rejection + conflicting-per-frame-metadata
    # detection), stable sort, and dtype coercions for identity columns. Coerce
    # trait_df's identity dtypes to match the template's int64 keys BEFORE merge
    # so merges don't silently fall through to all-NaN on numeric-string IDs.
    template = _build_per_plant_template_from_df(trajectory_df)
    for col in ("track_id", "plant_id"):
        if col in trait_df.columns and col in template.columns:
            try:
                trait_df[col] = trait_df[col].astype(template[col].dtype)
            except (TypeError, ValueError):
                pass

    result = template.merge(trait_df, on=list(_IDENTITY_5_TUPLE), how="left")

    # Enforce trait dtypes: 7 float64 + 1 bool.
    for col in _NUTATION_TRAIT_COLUMNS:
        if col == "is_nutating":
            result[col] = result[col].astype(bool)
        else:
            result[col] = result[col].astype(np.float64)

    # Enforce declared column order: 8 row-identity + 8 trait.
    result = result[list(ROW_IDENTITY_COLUMNS) + list(_NUTATION_TRAIT_COLUMNS)]

    return result
