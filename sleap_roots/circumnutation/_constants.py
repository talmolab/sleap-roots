"""Module-level named constants for the circumnutation pipeline.

Defaults sourced from `docs/circumnutation/roadmap.md` cross-cutting
concern CC-2 and the trait-derivation rationale in
`docs/circumnutation/theory.md`. Two version sentinels travel along:
``_SCHEMA_VERSION`` (bumps when the per-plant CSV row-identity columns
or sidecar JSON shapes change) and ``_CONSTANTS_VERSION`` (bumps when
any default in this module changes). Both are emitted into the
``run_metadata.json`` sidecar so a downstream user knows which version
produced any given CSV.

All constants are overridable per-call via the :class:`ConstantsT`
typed override-bag; the module-level names are the canonical defaults.
"""

import logging

import attrs


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema and constants versioning sentinels
# ---------------------------------------------------------------------------

_SCHEMA_VERSION: int = 1
"""Bumped when the per-plant CSV row-identity columns or sidecar JSON shapes change."""

_CONSTANTS_VERSION: int = 1
"""Bumped when any default in this module changes."""


# ---------------------------------------------------------------------------
# Numerical thresholds (theory.md / preliminary_results.md anchors in CC-2)
# ---------------------------------------------------------------------------

NOISE_MASK_K: float = 2
"""Velocity-magnitude noise mask multiplier (theory.md §6.2): mask frames with |v| < k·sigma_v."""

LGZ_STEADY_STATE_RESIDUAL_MAX: float = 0.2
"""Threshold on `L_gz_steady_state_residual / L_gz_estimate` (theory.md §7.4)."""

NYQUIST_RATIO_MAX: float = 0.25
"""Maximum tolerated per-frame-step / spatial-wavelength ratio for spatial CWT (theory.md §6.5)."""

SG_D2_AGREEMENT_MAX: float = 1.5
"""Pairwise agreement threshold between SG and second-difference noise estimators (theory.md §7.6)."""

LGZ_NMIN_RESOLVABLE: int = 5
"""Minimum number of trail frames within the growth zone for L_gz peak resolvability (theory.md §6.4)."""

COI_FRACTION_MAX: float = 0.5
"""Cone-of-influence fraction threshold; above this, scaleogram is unreliable (theory.md §7.6)."""

BAND_POWER_NOISE_RATIO: float = 3
"""Threshold on band-power / out-of-band-noise-floor ratio for `is_nutating` (theory.md §7.6)."""


# ---------------------------------------------------------------------------
# Wavelet basis defaults (forensic match to Derr Sept-2025; Rivière 2022)
# ---------------------------------------------------------------------------

WAVELET_DEFAULT_TEMPORAL: str = "cmor1.5-1.0"
"""Default mother wavelet for temporal CWT — matches Derr's Sept-2025 oracle scaleogram."""

WAVELET_DEFAULT_SPATIAL: str = "cgau2"
"""Default mother wavelet for spatial CWT — matches Rivière 2022 §"Kinematics: fine elongation measurements"."""


# ---------------------------------------------------------------------------
# Smoothing / detrending defaults (preliminary_results.md anchors)
# ---------------------------------------------------------------------------

SG_WINDOW_SHORT: int = 5
"""Short-window Savitzky-Golay window length, in frames (preliminary_results.md §3.3)."""

SG_DEGREE: int = 3
"""Polynomial degree for Savitzky-Golay smoothing (preliminary_results.md §3.3)."""

SG_WINDOW_DETREND: int = 23
"""Long-window Savitzky-Golay window for detrending (≈2 nutation periods; preliminary_results.md §3.4)."""


# ---------------------------------------------------------------------------
# Outlier / reliability thresholds
# ---------------------------------------------------------------------------

OUTLIER_STEP_RATIO: float = 2
"""Multiplier of median-step-magnitude above which a frame is flagged outlier (preliminary_results.md §4.1)."""

GROWTH_AXIS_RELIABILITY_K: float = 10
"""Net displacement / SG-residual ratio below which growth-axis is flagged unreliable (roadmap.md CC-5)."""


# ---------------------------------------------------------------------------
# Unit-string vocabulary for sidecar JSON files
# ---------------------------------------------------------------------------

VALID_UNIT_VOCABULARY: frozenset = frozenset(
    {
        # Pipeline-output (calibration-independent) units
        "px",
        "px²",
        "px/frame",
        "px/hr",
        "px·hr⁻¹",
        # convert_to_mm-output units (downstream converter only — the foundation
        # never emits these directly)
        "mm",
        "mm²",
        "mm/frame",
        "mm/hr",
        "mm·hr⁻¹",
        # Calibration-independent
        "hr",
        "hr⁻¹",
        "s",
        "rad",
        # Sentinels for non-numeric columns
        "bool",
        "int",
        "string",
        # Dimensionless
        "—",
    }
)
"""Closed vocabulary for unit strings used in `traits_per_plant.units.json`.

Pipeline outputs use only the px-based and calibration-independent
forms; the mm-based forms appear only in DataFrames produced by
:func:`sleap_roots.circumnutation.units.convert_to_mm`.
"""


# ---------------------------------------------------------------------------
# Typed override-bag for ergonomic per-call constant overrides
# ---------------------------------------------------------------------------


@attrs.define(slots=False, frozen=True, kw_only=True)
class ConstantsT:
    """Override-bag for circumnutation pipeline constants.

    Construct with keyword overrides for any defaults you want to
    change; unspecified fields keep the module-level defaults. Pass
    instances to functions or pipeline classes that accept a
    ``constants=`` parameter.

    Example:
        >>> custom = ConstantsT(BAND_POWER_NOISE_RATIO=4)
        >>> custom.BAND_POWER_NOISE_RATIO
        4
        >>> custom.NOISE_MASK_K  # default unchanged
        2
    """

    NOISE_MASK_K: float = NOISE_MASK_K
    LGZ_STEADY_STATE_RESIDUAL_MAX: float = LGZ_STEADY_STATE_RESIDUAL_MAX
    NYQUIST_RATIO_MAX: float = NYQUIST_RATIO_MAX
    SG_D2_AGREEMENT_MAX: float = SG_D2_AGREEMENT_MAX
    LGZ_NMIN_RESOLVABLE: int = LGZ_NMIN_RESOLVABLE
    COI_FRACTION_MAX: float = COI_FRACTION_MAX
    BAND_POWER_NOISE_RATIO: float = BAND_POWER_NOISE_RATIO
    WAVELET_DEFAULT_TEMPORAL: str = WAVELET_DEFAULT_TEMPORAL
    WAVELET_DEFAULT_SPATIAL: str = WAVELET_DEFAULT_SPATIAL
    SG_WINDOW_SHORT: int = SG_WINDOW_SHORT
    SG_DEGREE: int = SG_DEGREE
    SG_WINDOW_DETREND: int = SG_WINDOW_DETREND
    OUTLIER_STEP_RATIO: float = OUTLIER_STEP_RATIO
    GROWTH_AXIS_RELIABILITY_K: float = GROWTH_AXIS_RELIABILITY_K


def _default_constants_snapshot() -> dict:
    """Return a serializable mapping of every constant name to its current value.

    Used by :func:`sleap_roots.circumnutation._io.gather_run_metadata` to
    record the constants in effect at the time of a pipeline run, so a
    downstream re-run with different defaults is reproducible.

    Returns:
        Mapping of constant name (str) to value (JSON-serializable scalar).
    """
    return {
        "NOISE_MASK_K": NOISE_MASK_K,
        "LGZ_STEADY_STATE_RESIDUAL_MAX": LGZ_STEADY_STATE_RESIDUAL_MAX,
        "NYQUIST_RATIO_MAX": NYQUIST_RATIO_MAX,
        "SG_D2_AGREEMENT_MAX": SG_D2_AGREEMENT_MAX,
        "LGZ_NMIN_RESOLVABLE": LGZ_NMIN_RESOLVABLE,
        "COI_FRACTION_MAX": COI_FRACTION_MAX,
        "BAND_POWER_NOISE_RATIO": BAND_POWER_NOISE_RATIO,
        "WAVELET_DEFAULT_TEMPORAL": WAVELET_DEFAULT_TEMPORAL,
        "WAVELET_DEFAULT_SPATIAL": WAVELET_DEFAULT_SPATIAL,
        "SG_WINDOW_SHORT": SG_WINDOW_SHORT,
        "SG_DEGREE": SG_DEGREE,
        "SG_WINDOW_DETREND": SG_WINDOW_DETREND,
        "OUTLIER_STEP_RATIO": OUTLIER_STEP_RATIO,
        "GROWTH_AXIS_RELIABILITY_K": GROWTH_AXIS_RELIABILITY_K,
    }
