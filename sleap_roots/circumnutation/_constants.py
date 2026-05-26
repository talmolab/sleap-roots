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
import math

import attrs


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema and constants versioning sentinels
# ---------------------------------------------------------------------------

_SCHEMA_VERSION: int = 1
"""Bumped when the per-plant CSV row-identity columns or sidecar JSON shapes change."""

_CONSTANTS_VERSION: int = 4
"""Bumped when any default in this module changes.

PR #3 (``add-circumnutation-qc-tier``) bumped this 1 → 2 by adding four
new QC-tier threshold constants (``FRAC_OUTLIER_STEPS_MAX``,
``WORST_STEP_RATIO_MAX``, ``SG_MSD_AGREEMENT_MAX``, ``D2_MSD_AGREEMENT_MAX``)
to the overridable defaults set.

PR #4 (``add-circumnutation-synthetic-generator``) bumped this 2 → 3 by
adding seven new synthetic-generator default constants
(``SYNTHETIC_T_NUTATION_S``, ``SYNTHETIC_AMPLITUDE_PX``,
``SYNTHETIC_GROWTH_RATE_PX_PER_FRAME``, ``SYNTHETIC_NOISE_SIGMA_PX``,
``SYNTHETIC_CADENCE_S``, ``SYNTHETIC_N_FRAMES``,
``SYNTHETIC_GROWTH_AXIS_ANGLE_RAD``) to the overridable defaults set.

PR #5 (``add-circumnutation-temporal-cwt-machinery``) bumped this 3 → 4
by adding four new CWT-machinery default constants (``COI_EFOLDING_FACTOR``,
``CWT_SCALE_COUNT_DEFAULT``, ``CWT_PERIOD_MIN_NYQUIST_FACTOR``,
``CWT_PERIOD_MAX_SIGNAL_FRACTION``) to the overridable defaults set.
The ``COI_EFOLDING_FACTOR`` default of ``math.sqrt(1.5)`` = ``√B`` for
cmor1.5-1.0 was empirically verified across cmor0.5/1.0/1.5/2.0 via
step-response measurement (see ``openspec/changes/add-circumnutation-
temporal-cwt-machinery/design.md`` D3).
"""


# ---------------------------------------------------------------------------
# Numerical thresholds (theory.md / preliminary_results.md anchors in CC-2)
# ---------------------------------------------------------------------------

NOISE_MASK_K: float = 2
"""Velocity-magnitude noise mask multiplier (theory.md §6.2): mask frames with |v| < k·sigma_v."""

LGZ_STEADY_STATE_RESIDUAL_MAX: float = 0.2
"""Threshold on `L_gz_steady_state_residual / L_gz_estimate` (theory.md §7.4)."""

NYQUIST_RATIO_MAX: float = 0.25
"""Maximum tolerated per-frame-step / spatial-wavelength ratio for spatial CWT (theory.md §6.5).

Cross-reference (PR #5): numerically equal to :data:`CWT_PERIOD_MAX_SIGNAL_FRACTION`
(both default to ``0.25``) but semantically distinct. ``NYQUIST_RATIO_MAX`` is a
QC alias-protection threshold for PR #6's ``cadence_nyquist_ratio`` trait
(theory.md §6.5 — the per-frame-step / spatial-wavelength ratio for spatial CWT
must stay below this to avoid spatial aliasing). ``CWT_PERIOD_MAX_SIGNAL_FRACTION``
is the CWT scale-range upper bound used by ``compute_scaleogram`` to derive
``period_max_s = fraction * n_frames * cadence_s``. The two constants happen to
share the same numeric default ``0.25`` by coincidence; future tuning may change
either independently.
"""

SG_D2_AGREEMENT_MAX: float = 1.5
"""Pairwise agreement threshold between SG and second-difference noise estimators (theory.md §7.6)."""

SG_MSD_AGREEMENT_MAX: float = 1.5
"""Pairwise agreement threshold between SG and MSD-extrapolation noise estimators (roadmap.md CC-10).

Inherited from :data:`SG_D2_AGREEMENT_MAX`; CC-10 specifies all three
pairwise agreements share the same default threshold. Empirical
validation of whether the three thresholds should diverge is tracked in
GitHub Issue α (PR #3 follow-up).
"""

D2_MSD_AGREEMENT_MAX: float = 1.5
"""Pairwise agreement threshold between d2 and MSD-extrapolation noise estimators (roadmap.md CC-10).

Inherited from :data:`SG_D2_AGREEMENT_MAX`; see :data:`SG_MSD_AGREEMENT_MAX`
for the rationale and empirical-validation follow-up.
"""

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
# Temporal CWT machinery defaults (PR #5; design.md D3/D4/D7)
# ---------------------------------------------------------------------------

COI_EFOLDING_FACTOR: float = math.sqrt(1.5)
"""Cone-of-influence half-width = factor · scale, in samples (PR #5).

Calibrated for ``cmor1.5-1.0`` (the default ``WAVELET_DEFAULT_TEMPORAL``):
the wavelet's Gaussian envelope is ``exp(-t²/B)`` with B=1.5, so the
e-folding time at scale ``s`` is ``s · √B = s · √1.5 ≈ 1.225·s``. This is
the empirically-verified factor (step-response measurement at scales
20/50/100 across cmor0.5/1.0/1.5/2.0 each give factor ≈ √B). See
``openspec/changes/add-circumnutation-temporal-cwt-machinery/design.md`` D3
for the full derivation.

**When overriding ``WAVELET_DEFAULT_TEMPORAL`` to a different wavelet, ALSO
override ``COI_EFOLDING_FACTOR`` to the wavelet-appropriate value.** For
cmor variants the factor is ``math.sqrt(B)``; for ``cgau2`` (PR #9 spatial
sibling) the factor differs and PR #9 will determine it.

NOT to be confused with Torrence & Compo's ``√2`` factor (1998 Eq. 12),
which is the special case for the *standard* Morlet with ω₀=6 (B_equivalent=2).
For cmor1.5-1.0 the analogous factor is ``√1.5``, not ``√2``.
"""

CWT_SCALE_COUNT_DEFAULT: int = 64
"""Number of log-spaced scales returned by :func:`compute_scaleogram` (PR #5).

Derr Sept-2025 pilot used ~64 scales; this density is comparable to
12/octave × ~5-6 octaves (standard CWT-literature target). Overridable via
``ConstantsT(CWT_SCALE_COUNT_DEFAULT=...)`` for callers wanting denser
spectral sampling (at the cost of memory: scales × n_frames complex128).
"""

CWT_PERIOD_MIN_NYQUIST_FACTOR: float = 2.0
"""Multiplier of ``cadence_s`` setting the minimum period in the CWT scale range (PR #5).

``period_min_s = factor · cadence_s = 2 · cadence_s`` is the Nyquist period
(strict mathematical floor below which aliasing is certain). For cmor1.5-1.0
the wavelet support at scale = 2 (period = 2 cadence) is poor; PR #5
documents this caveat (design.md R3) and notes that traits consuming the
scaleogram should COI-mask AND additionally filter to scales with adequate
wavelet support. Tightening to ``factor = 4.0`` is a defensible alternative
default for callers wanting cleaner low-period resolution.
"""

CWT_PERIOD_MAX_SIGNAL_FRACTION: float = 0.25
"""Fraction of ``n_frames · cadence_s`` setting the maximum period in the CWT scale range (PR #5).

``period_max_s = fraction · n_frames · cadence_s = 0.25 · n_frames · cadence_s``
is Torrence & Compo's recommended ``n/4`` upper bound for tractable COI
fraction. Numerically equal to :data:`NYQUIST_RATIO_MAX` (PR #1) at default
``0.25``, but semantically distinct — see that constant's docstring for the
distinction. ``CWT_PERIOD_MAX_SIGNAL_FRACTION`` is the CWT-scale-range upper
bound used by ``compute_scaleogram``; ``NYQUIST_RATIO_MAX`` is the QC alias-
protection threshold for PR #6's ``cadence_nyquist_ratio`` trait. Future
tuning may change either independently.
"""


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

FRAC_OUTLIER_STEPS_MAX: float = 0.05
"""Threshold on the fraction of per-frame steps exceeding :data:`OUTLIER_STEP_RATIO` * median (theory.md §7.6 clean-track clause).

Used by the QC tier's ``track_is_clean`` composite. Empirical validation
of this default is tracked in GitHub Issue β (PR #3 follow-up).
"""

WORST_STEP_RATIO_MAX: float = 5
"""Threshold on the max-step / median-step ratio for the QC tier's ``track_is_clean`` clause (theory.md §7.6).

Empirical validation tracked in GitHub Issue β (PR #3 follow-up).
"""

GROWTH_AXIS_RELIABILITY_K: float = 10
"""Net displacement / SG-residual ratio below which growth-axis is flagged unreliable (roadmap.md CC-5)."""


# ---------------------------------------------------------------------------
# Synthetic-trajectory generator defaults (PR #4; preliminary_results.md
# §1 / §3.4 / §4.1 / §4.3 — plate 001 Nipponbare empirical anchors)
# ---------------------------------------------------------------------------

SYNTHETIC_T_NUTATION_S: float = 3333.0
"""Default nutation period in seconds (Derr Sept-2025 pilot; prelim §3.4)."""

SYNTHETIC_AMPLITUDE_PX: float = 10.0
"""Default peak-to-peak transverse nutation amplitude in px (prelim §4.3 plate 001 detrended)."""

SYNTHETIC_GROWTH_RATE_PX_PER_FRAME: float = 4.29
"""Default apex propagation speed along growth axis in px/frame (prelim §4.1 mean longitudinal step ⟨Δᵍ⟩)."""

SYNTHETIC_NOISE_SIGMA_PX: float = 2.0
"""Default xy-quadrature target noise σ in px (theory.md §8 Layer 1; plate 001 sg ≈ 1.83).

NB: this is the xy-quadrature target so that the QC tier's
``sg_residual_xy`` recovers ``noise_sigma_px`` directly. Per-axis draws
in the closed-form synthesis use ``σ_per_axis = noise_sigma_px / √2``.
"""

SYNTHETIC_CADENCE_S: float = 300.0
"""Default frame cadence in seconds (plate 001 imaging cadence = 5 min)."""

SYNTHETIC_N_FRAMES: int = 575
"""Default number of frames (plate 001 frame count over 47.9 hr at 5-min cadence)."""

SYNTHETIC_GROWTH_AXIS_ANGLE_RAD: float = math.pi / 2
"""Default growth-axis orientation in radians (image-y-down: root growing in +y screen direction)."""


# ---------------------------------------------------------------------------
# Unit-string vocabulary for sidecar JSON files
# ---------------------------------------------------------------------------

PIPELINE_UNIT_VOCABULARY: frozenset = frozenset(
    {
        # Pixel-based length / area / velocity units (what the pipeline emits)
        "px",
        "px²",
        "px/frame",
        "px/hr",
        "px·hr⁻¹",
        # Calibration-independent units (time, angle, rate)
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

The pipeline emits ONLY these forms (no mm-based units), reflecting
the pure-pixel pipeline contract. Use this vocabulary to validate
sidecar JSON files produced by
:func:`sleap_roots.circumnutation._io.write_per_plant_csv`.
"""


CONVERTED_UNIT_VOCABULARY: frozenset = frozenset(
    {
        # Millimeter-based length / area / velocity units (convert_to_mm output range)
        "mm",
        "mm²",
        "mm/frame",
        "mm/hr",
        "mm·hr⁻¹",
    }
)
"""Vocabulary of unit strings produced by :func:`sleap_roots.circumnutation.units.convert_to_mm`.

These units appear ONLY in DataFrames returned by ``convert_to_mm``,
never in pipeline output sidecars. Splitting from
:data:`PIPELINE_UNIT_VOCABULARY` makes the pure-pixel pipeline
contract explicit: a sidecar containing any of these values would
violate the contract.
"""


VALID_UNIT_VOCABULARY: frozenset = PIPELINE_UNIT_VOCABULARY | CONVERTED_UNIT_VOCABULARY
"""Union of :data:`PIPELINE_UNIT_VOCABULARY` and :data:`CONVERTED_UNIT_VOCABULARY`.

Useful for downstream code that needs to accept either a raw pipeline
sidecar or a converted DataFrame. Tests that specifically verify the
pipeline emits no mm units should check against
:data:`PIPELINE_UNIT_VOCABULARY`, not this union.
"""


ROW_IDENTITY_UNITS: dict = {
    "series": "string",
    "sample_uid": "string",
    "timepoint": "string",
    "plate_id": "string",
    "plant_id": "int",
    "track_id": "int",
    "genotype": "string",
    "treatment": "string",
}
"""Canonical unit-string mapping for the eight row-identity columns.

Public so tier modules can import it directly when building their own
per-plant DataFrames and units sidecars; consumed by
:func:`sleap_roots.circumnutation._io.default_units_for_template`.
Every value here is guaranteed to be in :data:`PIPELINE_UNIT_VOCABULARY`.
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
    SG_MSD_AGREEMENT_MAX: float = SG_MSD_AGREEMENT_MAX
    D2_MSD_AGREEMENT_MAX: float = D2_MSD_AGREEMENT_MAX
    LGZ_NMIN_RESOLVABLE: int = LGZ_NMIN_RESOLVABLE
    COI_FRACTION_MAX: float = COI_FRACTION_MAX
    BAND_POWER_NOISE_RATIO: float = BAND_POWER_NOISE_RATIO
    WAVELET_DEFAULT_TEMPORAL: str = WAVELET_DEFAULT_TEMPORAL
    WAVELET_DEFAULT_SPATIAL: str = WAVELET_DEFAULT_SPATIAL
    SG_WINDOW_SHORT: int = SG_WINDOW_SHORT
    SG_DEGREE: int = SG_DEGREE
    SG_WINDOW_DETREND: int = SG_WINDOW_DETREND
    OUTLIER_STEP_RATIO: float = OUTLIER_STEP_RATIO
    FRAC_OUTLIER_STEPS_MAX: float = FRAC_OUTLIER_STEPS_MAX
    WORST_STEP_RATIO_MAX: float = WORST_STEP_RATIO_MAX
    GROWTH_AXIS_RELIABILITY_K: float = GROWTH_AXIS_RELIABILITY_K
    # PR #4 — synthetic-generator defaults
    SYNTHETIC_T_NUTATION_S: float = SYNTHETIC_T_NUTATION_S
    SYNTHETIC_AMPLITUDE_PX: float = SYNTHETIC_AMPLITUDE_PX
    SYNTHETIC_GROWTH_RATE_PX_PER_FRAME: float = SYNTHETIC_GROWTH_RATE_PX_PER_FRAME
    SYNTHETIC_NOISE_SIGMA_PX: float = SYNTHETIC_NOISE_SIGMA_PX
    SYNTHETIC_CADENCE_S: float = SYNTHETIC_CADENCE_S
    SYNTHETIC_N_FRAMES: int = SYNTHETIC_N_FRAMES
    SYNTHETIC_GROWTH_AXIS_ANGLE_RAD: float = SYNTHETIC_GROWTH_AXIS_ANGLE_RAD
    # PR #5 — temporal CWT machinery defaults
    COI_EFOLDING_FACTOR: float = COI_EFOLDING_FACTOR
    CWT_SCALE_COUNT_DEFAULT: int = CWT_SCALE_COUNT_DEFAULT
    CWT_PERIOD_MIN_NYQUIST_FACTOR: float = CWT_PERIOD_MIN_NYQUIST_FACTOR
    CWT_PERIOD_MAX_SIGNAL_FRACTION: float = CWT_PERIOD_MAX_SIGNAL_FRACTION


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
        "SG_MSD_AGREEMENT_MAX": SG_MSD_AGREEMENT_MAX,
        "D2_MSD_AGREEMENT_MAX": D2_MSD_AGREEMENT_MAX,
        "LGZ_NMIN_RESOLVABLE": LGZ_NMIN_RESOLVABLE,
        "COI_FRACTION_MAX": COI_FRACTION_MAX,
        "BAND_POWER_NOISE_RATIO": BAND_POWER_NOISE_RATIO,
        "WAVELET_DEFAULT_TEMPORAL": WAVELET_DEFAULT_TEMPORAL,
        "WAVELET_DEFAULT_SPATIAL": WAVELET_DEFAULT_SPATIAL,
        "SG_WINDOW_SHORT": SG_WINDOW_SHORT,
        "SG_DEGREE": SG_DEGREE,
        "SG_WINDOW_DETREND": SG_WINDOW_DETREND,
        "OUTLIER_STEP_RATIO": OUTLIER_STEP_RATIO,
        "FRAC_OUTLIER_STEPS_MAX": FRAC_OUTLIER_STEPS_MAX,
        "WORST_STEP_RATIO_MAX": WORST_STEP_RATIO_MAX,
        "GROWTH_AXIS_RELIABILITY_K": GROWTH_AXIS_RELIABILITY_K,
        # PR #4 — synthetic-generator defaults
        "SYNTHETIC_T_NUTATION_S": SYNTHETIC_T_NUTATION_S,
        "SYNTHETIC_AMPLITUDE_PX": SYNTHETIC_AMPLITUDE_PX,
        "SYNTHETIC_GROWTH_RATE_PX_PER_FRAME": SYNTHETIC_GROWTH_RATE_PX_PER_FRAME,
        "SYNTHETIC_NOISE_SIGMA_PX": SYNTHETIC_NOISE_SIGMA_PX,
        "SYNTHETIC_CADENCE_S": SYNTHETIC_CADENCE_S,
        "SYNTHETIC_N_FRAMES": SYNTHETIC_N_FRAMES,
        "SYNTHETIC_GROWTH_AXIS_ANGLE_RAD": SYNTHETIC_GROWTH_AXIS_ANGLE_RAD,
        # PR #5 — temporal CWT machinery defaults
        "COI_EFOLDING_FACTOR": COI_EFOLDING_FACTOR,
        "CWT_SCALE_COUNT_DEFAULT": CWT_SCALE_COUNT_DEFAULT,
        "CWT_PERIOD_MIN_NYQUIST_FACTOR": CWT_PERIOD_MIN_NYQUIST_FACTOR,
        "CWT_PERIOD_MAX_SIGNAL_FRACTION": CWT_PERIOD_MAX_SIGNAL_FRACTION,
    }
