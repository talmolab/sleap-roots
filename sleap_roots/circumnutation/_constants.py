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

_CONSTANTS_VERSION: int = 5
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

PR #6 (``add-circumnutation-tier1-derr-faithful``) bumped this 4 → 5 by
adding six new Tier 1 / threshold default constants
(``RIDGE_CONTINUITY_FILTER_WINDOW``, ``NOISE_FLOOR_OUT_OF_BAND_FACTOR``,
``BAND_POWER_BAND_LOW_FACTOR``, ``BAND_POWER_BAND_HIGH_FACTOR``,
``DERR_EXPECTED_PERIOD_S``, ``TEMPORAL_NYQUIST_RATIO_MAX``) to the
overridable defaults set. ``TEMPORAL_NYQUIST_RATIO_MAX = 0.25`` is the
TEMPORAL sibling of the existing :data:`NYQUIST_RATIO_MAX` (SPATIAL);
both default to ``0.25`` per theory.md §6.5's "10-min still works"
empirical anchor — the dimensional separation lives in the constant
NAMES + docstrings, not in different values.
"""


# ---------------------------------------------------------------------------
# Numerical thresholds (theory.md / preliminary_results.md anchors in CC-2)
# ---------------------------------------------------------------------------

NOISE_MASK_K: float = 2
"""Velocity-magnitude noise mask multiplier (theory.md §6.2): mask frames with |v| < k·sigma_v."""

LGZ_STEADY_STATE_RESIDUAL_MAX: float = 0.2
"""Threshold on `L_gz_steady_state_residual / L_gz_estimate` (theory.md §7.4)."""

NYQUIST_RATIO_MAX: float = 0.25
"""Maximum tolerated per-frame-step / spatial-wavelength ratio for SPATIAL CWT (theory.md §6.5).

This is the SPATIAL cadence-Nyquist threshold (px/px, DPI-independent).
The TEMPORAL sibling is :data:`TEMPORAL_NYQUIST_RATIO_MAX`. Both default
to ``0.25`` per theory.md §6.5's empirical anchor ("5-min comfortable,
10-min still works, 30-min aliases"); the dimensional separation lives
in the constant NAMES + docstrings, NOT in different values. PR #9
spatial-CWT machinery consumes this constant; PR #6 (``nutation``
module) consumes ``TEMPORAL_NYQUIST_RATIO_MAX`` for the temporal
``cadence_nyquist_ratio`` trait.

Cross-reference (PR #5): numerically equal to
:data:`CWT_PERIOD_MAX_SIGNAL_FRACTION` (both default to ``0.25``) but
semantically distinct. ``CWT_PERIOD_MAX_SIGNAL_FRACTION`` is the CWT
scale-range upper bound used by ``compute_scaleogram`` to derive
``period_max_s = fraction * n_frames * cadence_s``. The three constants
(``NYQUIST_RATIO_MAX``, ``TEMPORAL_NYQUIST_RATIO_MAX``,
``CWT_PERIOD_MAX_SIGNAL_FRACTION``) happen to share the same numeric
default ``0.25`` by coincidence; future tuning may change each
independently.
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
# Tier 1 / threshold defaults (PR #6; design.md D4/D7/D8/D9)
# ---------------------------------------------------------------------------

RIDGE_CONTINUITY_FILTER_WINDOW: int = 5
"""Median-filter window for ``temporal_cwt.smooth_ridge``, in frames (PR #6).

Issue #214: PR #5's per-frame argmax ridge can hop discontinuously between
adjacent CWT scales when two harmonics have similar amplitude, producing
spurious ``T_nutation_iqr`` inflation. ``smooth_ridge`` applies
``scipy.ndimage.median_filter(periods_s, size=window, mode='nearest')``
to suppress this. Default 5 frames = 25 minutes at plate-001's 300 s
cadence (~0.75% of the 3333 s nutation period) — tightens without
smearing biological signal.

For the empirical acceptance criterion (post-filter IQR < raw IQR on
≥5 of 6 plate-001 tracks), see the spec scenario "GitHub issue #214
acceptance" in the PR #6 spec delta.
"""

NOISE_FLOOR_OUT_OF_BAND_FACTOR: float = 5.0
"""Out-of-band frequency cutoff factor for ``compute_fourier_noise_floor`` (PR #6, CC-8).

The noise floor is the median Fourier amplitude over frequencies
``f > factor / T_nutation_median`` — i.e., the "well above the
candidate-nutation-frequency" region of the spectrum. Default 5.0 per
``roadmap.md`` CC-8 verbatim.

GREEN-phase empirical decision: PR #6 §2.E.7 sensitivity test parametrizes
factor ∈ {3, 5} on plate-001 (factor=7 was removed during GREEN-phase
because at cadence=300 s + T≈3333 s the cutoff 7/3333 ≈ 2.1e-3 Hz
exceeds the Nyquist frequency 1.67e-3 Hz → empty out-of-band → NaN
noise floor by design). The empirically-robust value is recorded in the
GREEN-phase Reconciliation Appendix. If multi-plate work shows factor=5
puts BM2016-predicted second/third harmonics INSIDE the noise band
(inflating the floor and falsely depressing ``band_power_ratio``), a
follow-up roadmap revision may lower this.
"""

BAND_POWER_BAND_LOW_FACTOR: float = 0.5
"""Lower band-edge factor for ``band_power_ratio`` (PR #6, theory.md §7.2).

``f_low = factor / T_nutation_median``. With factor=0.5, the corresponding
PERIOD edge is ``2 * T_nutation_median`` (half the candidate nutation
frequency). Per theory.md §7.2 "spectral power in [0.5T, 2T] band /
total spectral power" — the [0.5T, 2T] period band maps to the
frequency band [1/(2T), 2/T] = [factor_low/T, factor_high/T].
"""

BAND_POWER_BAND_HIGH_FACTOR: float = 2.0
"""Upper band-edge factor for ``band_power_ratio`` (PR #6, theory.md §7.2).

``f_high = factor / T_nutation_median``. With factor=2.0, the
corresponding PERIOD edge is ``T_nutation_median / 2`` (2× the candidate
nutation frequency). Per theory.md §7.2 — see the lower-edge factor
docstring for the period-vs-frequency mapping.
"""

DERR_EXPECTED_PERIOD_S: float = 3333.0
"""Reference rice nutation period (s) for ``period_residual_vs_derr_reference`` (PR #6).

Sourced from Derr Sept-2025 pilot
``5minutes_average_period=3333s.pdf`` (spectral peak at f ≈ 0.0003 Hz,
T ≈ 3333 s ≈ 55.5 min). The trait
``period_residual_vs_derr_reference = (T_nutation_median - DERR_EXPECTED_PERIOD_S) / DERR_EXPECTED_PERIOD_S``
is the FRACTIONAL deviation from this rice-specific reference.

For non-rice species (e.g., Arabidopsis, sunflower per Rivière 2022 §1.2
with nutation period in 5400-14400 s range), override via
``ConstantsT(DERR_EXPECTED_PERIOD_S=<species-appropriate value>)``.
Multi-plate empirical validation of the rice value is tracked in a
follow-up GitHub issue ("circumnutation: validate
DERR_EXPECTED_PERIOD_S and TEMPORAL_NYQUIST_RATIO_MAX from literature
+ multi-plate data") filed alongside PR #6.

When Derr provides raw scaleogram numerics (currently PDF/PNG only),
a future PR may upgrade the trait algorithm from this constant-anchored
residual to a richer spectral-shape distance. The trait CSV column
stays a single float; only the algorithm behind it gets richer.
"""

TEMPORAL_NYQUIST_RATIO_MAX: float = 0.25
"""Maximum tolerated cadence_s / T_nutation_median ratio for TEMPORAL CWT (PR #6).

Conservative cushion below the strict Nyquist limit (0.5) for the
temporal cadence-Nyquist check. The TEMPORAL sibling of
:data:`NYQUIST_RATIO_MAX` (SPATIAL). Both default to ``0.25`` per
theory.md §6.5's empirical anchor: "5-min cadence is comfortable
(ratio ≈ 0.09); 10-min would still work (ratio ≈ 0.18); 30-min would
alias the nutation (ratio ≈ 0.54)". 0.25 sits between "still works"
(0.18) and "aliases" (0.54), biased conservative. The dimensional
separation from ``NYQUIST_RATIO_MAX`` lives in the constant NAMES +
docstrings, not in different values.

PR #6's ``nutation`` module emits the ``cadence_nyquist_ratio`` trait
(``cadence_s / T_nutation_median``) and downstream QC tier will compare
it against this constant. The interim default ``0.25`` is conservative-
defensible per §6.5 but lacks multi-plate empirical validation; the
follow-up GitHub issue "circumnutation: validate
TEMPORAL_NYQUIST_RATIO_MAX from literature + multi-plate data" (filed
alongside PR #6 mirroring the #205-#208 pattern) bundles with the
multi-plate sweep that resolves #202.
"""


# ---------------------------------------------------------------------------
# Spatial CWT machinery defaults (PR #9; design.md + capture_spatial_coi_factor.py)
# ---------------------------------------------------------------------------

SPATIAL_COI_EFOLDING_FACTOR: float = 1.375
"""Cone-of-influence half-width = factor · scale, in samples, for SPATIAL CWT (PR #9).

Calibrated for the ``cgau2`` spatial mother wavelet
(:data:`WAVELET_DEFAULT_SPATIAL`). This is the SPATIAL sibling of
:data:`COI_EFOLDING_FACTOR` (calibrated for ``cmor1.5-1.0`` at ``√1.5 ≈ 1.225``);
``cgau2``'s envelope is a 2nd-derivative Gaussian (not a plain Gaussian), so its
e-folding factor differs and was measured empirically by the **impulse 1/e
half-width** method (median ``≈ 1.375`` across scales 8–128) in
``scripts/circumnutation/capture_spatial_coi_factor.py`` — the cgau2 analog of
the step-response measurement the ``COI_EFOLDING_FACTOR`` docstring explicitly
defers to PR #9.
"""

CWT_WAVELENGTH_MIN_NYQUIST_FACTOR: float = 2.0
"""Multiplier of ``ds`` setting the minimum spatial wavelength in the CWT scale range (PR #9).

``wavelength_min_px = factor · ds = 2 · ds`` is the spatial-Nyquist wavelength
(the strict floor below which spatial aliasing is certain). SPATIAL-domain sibling
of :data:`CWT_PERIOD_MIN_NYQUIST_FACTOR` (TEMPORAL); same numeric default ``2.0``,
the dimensional separation lives in the NAMES + docstrings, not the values.
"""

CWT_WAVELENGTH_MAX_SIGNAL_FRACTION: float = 0.25
"""Fraction of ``n · ds`` setting the maximum spatial wavelength in the CWT scale range (PR #9).

``wavelength_max_px = fraction · n · ds = 0.25 · n · ds`` is Torrence & Compo's
``n/4`` upper bound for a tractable COI fraction, in the spatial domain. SPATIAL
sibling of :data:`CWT_PERIOD_MAX_SIGNAL_FRACTION` (TEMPORAL); same numeric default
``0.25``. The spatial ``MIN_SAMPLES_REQUIRED`` floor derives from these two as
``int(floor(CWT_WAVELENGTH_MIN_NYQUIST_FACTOR / CWT_WAVELENGTH_MAX_SIGNAL_FRACTION)) + 1``
(= ``9`` at defaults), mirroring the temporal derivation.
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
    # PR #6 — Tier 1 / threshold defaults
    RIDGE_CONTINUITY_FILTER_WINDOW: int = RIDGE_CONTINUITY_FILTER_WINDOW
    NOISE_FLOOR_OUT_OF_BAND_FACTOR: float = NOISE_FLOOR_OUT_OF_BAND_FACTOR
    BAND_POWER_BAND_LOW_FACTOR: float = BAND_POWER_BAND_LOW_FACTOR
    BAND_POWER_BAND_HIGH_FACTOR: float = BAND_POWER_BAND_HIGH_FACTOR
    DERR_EXPECTED_PERIOD_S: float = DERR_EXPECTED_PERIOD_S
    TEMPORAL_NYQUIST_RATIO_MAX: float = TEMPORAL_NYQUIST_RATIO_MAX
    # PR #9 — spatial CWT machinery defaults
    SPATIAL_COI_EFOLDING_FACTOR: float = SPATIAL_COI_EFOLDING_FACTOR
    CWT_WAVELENGTH_MIN_NYQUIST_FACTOR: float = CWT_WAVELENGTH_MIN_NYQUIST_FACTOR
    CWT_WAVELENGTH_MAX_SIGNAL_FRACTION: float = CWT_WAVELENGTH_MAX_SIGNAL_FRACTION


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
        # PR #6 — Tier 1 / threshold defaults
        "RIDGE_CONTINUITY_FILTER_WINDOW": RIDGE_CONTINUITY_FILTER_WINDOW,
        "NOISE_FLOOR_OUT_OF_BAND_FACTOR": NOISE_FLOOR_OUT_OF_BAND_FACTOR,
        "BAND_POWER_BAND_LOW_FACTOR": BAND_POWER_BAND_LOW_FACTOR,
        "BAND_POWER_BAND_HIGH_FACTOR": BAND_POWER_BAND_HIGH_FACTOR,
        "DERR_EXPECTED_PERIOD_S": DERR_EXPECTED_PERIOD_S,
        "TEMPORAL_NYQUIST_RATIO_MAX": TEMPORAL_NYQUIST_RATIO_MAX,
    }
