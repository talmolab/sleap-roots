"""Synthetic tip-trajectory generator for Layer-1 validation (PR #4).

Realizes Rivière 2022 Eq. 4 in **parametric closed form** (design.md D1)
using user-facing aggregate parameters (D2): ``amplitude_px``,
``T_nutation_s``, ``growth_rate_px_per_frame``, ``noise_sigma_px``.

Closed-form realization
-----------------------
The apex propagates along the growth axis at velocity ``v_growth_per_s
= growth_rate_px_per_frame / cadence_s``; transverse nutation contributes
``A_lat · sin(handedness · ω · t + initial_phase_rad)`` with
``A_lat = amplitude_px / 2`` and ``ω = 2π / T_nutation_s``; iid Gaussian
localization noise is added per-axis with ``σ_per_axis = noise_sigma_px / √2``
so the QC tier's xy-quadrature noise estimators
(:func:`~sleap_roots.circumnutation._noise.compute_sg_residual_xy` etc.)
recover ``noise_sigma_px`` directly. See ``openspec/changes/.../design.md``
D1 for the equation block and the Rivière correspondence (D2) noting why
the 6-parameter Rivière tuple ``(L_gz, ΔL, δ̇₀, ε̇₀, ω, R)`` collapses to
3 tip-observable aggregates ``(Δφ, v_growth, ω)`` — PR #12 will wrap with
a Rivière-named translation helper when PR #9 / PR #11 land the spatial-
CWT recovery.

Determinism contract (CC-6)
---------------------------
``random_state`` accepts ``int``, ``np.random.Generator``, or ``None``;
a single internal call to ``np.random.default_rng(random_state)`` handles
all three idiomatically. Same int seed → bit-identical ``tip_x`` / ``tip_y``
arrays across runs AND across OSs on 64-bit platforms (numpy's PCG64
stability guarantee per NEP 19; sleap-roots CI is 100% 64-bit).

When ``noise_sigma_px == 0.0`` exactly, the RNG path is SHORT-CIRCUITED:
``np.random.default_rng`` is NOT called, and a caller-supplied ``Generator``
state is unchanged after the call. This decouples output determinism from
``random_state`` in noise-free mode (D11). **Layer-1 caveat**: theory.md
§8 mandates noise for Layer-1 pipeline validation; PR #12's parameterized
suite SHALL use ``noise_sigma_px > 0``. The ``noise_sigma_px=0`` mode is
for closed-form-correctness unit tests only.

Pure-pixel + per-axis noise (σ = noise_sigma_px / √2)
-----------------------------------------------------
All length-bearing outputs are in pixels per CC-3 (no ``px_per_mm`` in
signature). The per-axis Gaussian noise σ is ``noise_sigma_px / √2``,
chosen so that the QC tier's xy-quadrature noise estimators recover the
documented ``noise_sigma_px``. Empirical anchor: plate 001 SG-residual
≈ 1.83 px (preliminary_results §4.2); default ``SYNTHETIC_NOISE_SIGMA_PX
= 2.0``.

Handedness convention
---------------------
``handedness = +1`` (default) = counterclockwise per BM2016 Eq. 20 +
theory.md §3.5 / §7.3. The phase argument is multiplied by ``handedness``
(``sin(handedness · phase)``); flipping handedness flips the sign of the
lateral offset's time derivative, inverting the rotation direction observed
in :func:`~sleap_roots.circumnutation._geometry.compute_psi_g`. Under the
image-y-down convention (theory.md §2.1), the math-CCW sweep displays as
visually clockwise on screen — a rendering detail, not a sign-convention
bug.

ConstantsT resolution-order (D13)
---------------------------------
Seven user-facing parameters default to ``None`` in the signature
(``n_frames``, ``cadence_s``, ``amplitude_px``, ``T_nutation_s``,
``growth_rate_px_per_frame``, ``noise_sigma_px``, ``growth_axis_angle_rad``).
At call time the resolution order is:

1. **Call-site kwarg wins.** If the parameter is explicitly passed (not
   ``None``), use that value.
2. **``constants`` parameter overrides module default.** If the parameter
   is ``None`` AND ``constants`` is a ``ConstantsT`` instance, use
   ``constants.SYNTHETIC_<name>``.
3. **Module-level default.** If both above are absent, fall back to the
   module-level ``SYNTHETIC_<name>`` (which equals the ``ConstantsT()``
   field default).

The remaining parameters (``handedness``, ``x0_px``, ``y0_px``,
``initial_phase_rad``, ``random_state``, ``constants``, and the 8 identity
columns) keep direct literal defaults because they are NOT ConstantsT-
overridable.

Note: divergence from ``CircumnutationInputs.cadence_s`` coercion
----------------------------------------------------------------
Unlike :class:`~sleap_roots.circumnutation._types.CircumnutationInputs`,
which has an attrs converter for ``cadence_s = "300"`` (string → float),
``generate_trajectory`` does NOT coerce string inputs. A string
``cadence_s`` raises ``TypeError`` cleanly. Rationale:
``CircumnutationInputs`` is a downstream-data dataclass that may receive
YAML / JSON-parsed inputs; ``generate_trajectory`` is a programmatic test
fixture that should reject ambiguous types at the call site.

Multi-track plates (idiom)
--------------------------
Each call produces ONE track. For a plate of N tracks with statistically
independent noise streams::

    seed_seq = np.random.SeedSequence(42)
    child_seeds = seed_seq.spawn(6)
    df_plate = pd.concat([
        generate_trajectory(track_id=i, plant_id=i, random_state=child_seeds[i])
        for i in range(6)
    ], ignore_index=True)
"""

import logging
import math
from typing import Any, Optional, Tuple, Union

import numpy as np
import pandas as pd

from sleap_roots.circumnutation._constants import ConstantsT
from sleap_roots.circumnutation._types import (
    REQUIRED_PER_FRAME_COLUMNS,
    ROW_IDENTITY_COLUMNS,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Validation helpers — strict bool-rejecting type checks per design.md D8.
#
# Python convention: bool is a subclass of int, so `isinstance(True, int) == True`.
# We must explicitly reject bool for every numeric field so that True/False
# do not silently pass as 1/0.
# ---------------------------------------------------------------------------


def _check_int_strict(name: str, value: Any, *, min_value: Optional[int] = None) -> int:
    """Validate value is a non-bool int (or numpy.integer); optionally ≥ min_value.

    Returns the value coerced to Python int. Raises:
        TypeError: if value is bool, float, str, None, or other non-integer type.
        ValueError: if value < min_value when min_value is specified.
    """
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a non-bool int, got bool: {value!r}")
    if not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}: {value!r}")
    int_value = int(value)
    if min_value is not None and int_value < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {int_value}")
    return int_value


def _check_float_finite(
    name: str,
    value: Any,
    *,
    positive: bool = False,
    non_negative: bool = False,
) -> float:
    """Validate value is a finite float-like (not bool); optionally positive/non-neg.

    Accepts ``int``, ``np.integer``, ``float``, ``np.floating`` — but explicitly
    rejects ``bool`` (despite being an int-subclass) and rejects string inputs
    (no implicit numeric coercion; design.md D8 documented divergence from
    ``CircumnutationInputs.cadence_s``).

    Returns the value coerced to Python float. Raises:
        TypeError: if value is bool, str, None, or other non-numeric type.
        ValueError: if value is NaN/±inf, or fails the positivity constraint.
    """
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite float, got bool: {value!r}")
    if not isinstance(value, (int, float, np.floating, np.integer)):
        raise TypeError(
            f"{name} must be a float, got {type(value).__name__}: {value!r}"
        )
    float_value = float(value)
    if not math.isfinite(float_value):
        raise ValueError(
            f"{name} must be a finite float (not NaN, +inf, or -inf), "
            f"got {float_value!r}"
        )
    if positive and float_value <= 0:
        raise ValueError(f"{name} must be positive, got {float_value!r}")
    if non_negative and float_value < 0:
        raise ValueError(f"{name} must be non-negative, got {float_value!r}")
    return float_value


def _check_string_or_none(name: str, value: Any, *, allow_none: bool) -> Optional[str]:
    """Validate value is a str (or None if allow_none=True). Reject other types.

    Identity columns: series, sample_uid, timepoint, plate_id are mandatory
    strings (allow_none=False); genotype, treatment are str-or-None
    (allow_none=True; None maps to NaN in the output DataFrame).
    """
    if value is None:
        if allow_none:
            return None
        raise TypeError(f"{name} must be a non-None str, got None")
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a str, got {type(value).__name__}: {value!r}")
    return value


def _check_handedness(value: Any) -> int:
    """Validate handedness is exactly +1 or -1 (int; bool/float rejected)."""
    if isinstance(value, bool):
        raise TypeError(
            f"handedness must be exactly +1 or -1 (int), got bool: {value!r}"
        )
    if not isinstance(value, (int, np.integer)):
        raise TypeError(
            f"handedness must be exactly +1 or -1 (int), got "
            f"{type(value).__name__}: {value!r}"
        )
    int_value = int(value)
    if int_value not in (+1, -1):
        raise ValueError(f"handedness must be exactly +1 or -1, got {int_value!r}")
    return int_value


def _check_random_state(
    value: Any,
) -> Optional[Union[int, np.random.Generator]]:
    """Validate random_state is None | int | np.random.Generator.

    Explicitly rejects the legacy ``np.random.RandomState`` API per
    design.md D5 (modern Generator API only — stability guaranteed by
    NEP 19).
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(
            f"random_state must be int, np.random.Generator, or None; "
            f"got bool: {value!r}"
        )
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, np.random.Generator):
        return value
    raise TypeError(
        f"random_state must be int, np.random.Generator, or None; "
        f"got {type(value).__name__}: {value!r}"
    )


def _check_constants(value: Any) -> Optional[ConstantsT]:
    """Validate constants is None or a ConstantsT instance."""
    if value is None:
        return None
    if not isinstance(value, ConstantsT):
        raise TypeError(
            f"constants must be None or a ConstantsT instance, "
            f"got {type(value).__name__}: {value!r}"
        )
    return value


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_trajectory(
    *,
    # ---- sampling / physics (ConstantsT-overridable; None = "resolve from
    # constants or module-level default" per D13 resolution-order)
    n_frames: Optional[int] = None,
    cadence_s: Optional[float] = None,
    amplitude_px: Optional[float] = None,
    T_nutation_s: Optional[float] = None,
    growth_rate_px_per_frame: Optional[float] = None,
    noise_sigma_px: Optional[float] = None,
    growth_axis_angle_rad: Optional[float] = None,
    # ---- deterministic / geometric (literal defaults; NOT ConstantsT-overridable)
    handedness: int = 1,
    x0_px: float = 0.0,
    y0_px: float = 0.0,
    initial_phase_rad: float = 0.0,
    # ---- determinism + override-bag
    random_state: Optional[Union[int, np.random.Generator]] = None,
    constants: Optional[ConstantsT] = None,
    # ---- identity columns (kw-only defaulted per design.md D9)
    series: str = "synthetic",
    sample_uid: str = "synthetic_001",
    timepoint: str = "synthetic",
    plate_id: str = "synthetic",
    plant_id: int = 0,
    track_id: int = 0,
    genotype: Optional[str] = None,
    treatment: Optional[str] = None,
) -> pd.DataFrame:
    """Generate a single synthetic tip trajectory in pure-pixel coordinates.

    Closed-form realization of Rivière 2022 Eq. 4 using user-facing
    aggregate parameters. See the module docstring for the math and the
    full Rivière-correspondence + determinism + handedness contracts.

    Args:
        n_frames: Number of frames in the output trajectory. ``None``
            resolves to ``constants.SYNTHETIC_N_FRAMES`` (or module
            default ``575``). Must be a non-bool int ≥ 1.
        cadence_s: Frame cadence in seconds. ``None`` resolves to default
            ``300.0`` (5 min). Must be a positive finite float; string
            inputs are NOT coerced (TypeError).
        amplitude_px: Peak-to-peak transverse nutation amplitude in
            pixels. ``None`` resolves to default ``10.0``. Must be a
            non-negative finite float (``0.0`` is valid — produces a
            pure-linear trajectory).
        T_nutation_s: Nutation period in seconds. ``None`` resolves to
            default ``3333.0``. Must be positive finite.
        growth_rate_px_per_frame: Apex propagation speed along the
            growth axis in px/frame. ``None`` resolves to default
            ``4.29``. Must be a finite float (negative IS valid —
            apex moves in -u_g direction).
        noise_sigma_px: Target xy-quadrature noise σ in pixels. Per-axis
            draws use ``σ_per_axis = noise_sigma_px / √2`` so the QC
            tier's xy-quadrature estimators recover this value directly.
            ``None`` resolves to default ``2.0``. Must be a non-negative
            finite float. ``0.0`` short-circuits the RNG path (D11).
        growth_axis_angle_rad: Growth-axis orientation in radians.
            ``None`` resolves to default ``π/2`` (image-y-down: root
            growing in +y screen direction). Must be a finite float;
            values outside ``[-π, π]`` wrap mod 2π via the cos/sin
            evaluation (no explicit modulo).
        handedness: ``+1`` for counterclockwise (math-CCW per BM2016
            Eq. 20; default), ``-1`` for clockwise. Must be exactly
            ``+1`` or ``-1`` (int; bool / float rejected).
        x0_px, y0_px: Initial tip position in pixels. Default ``(0.0, 0.0)``.
            Must be finite floats.
        initial_phase_rad: Initial nutation phase in radians.
            Default ``0.0``. Must be a finite float.
        random_state: Seed (int), pre-built ``np.random.Generator``, or
            ``None`` (fresh non-deterministic Generator). The legacy
            ``np.random.RandomState`` API is explicitly rejected
            (TypeError) per the modern-Generator-only determinism
            contract.
        constants: Optional :class:`ConstantsT` override-bag. Resolution
            order per D13: explicit kwarg > constants.SYNTHETIC_<X> >
            module-level default.
        series, sample_uid, timepoint, plate_id: Mandatory string
            identity columns. Defaults ``"synthetic"`` / ``"synthetic_001"``.
        plant_id, track_id: Mandatory int identity columns. Defaults
            ``0``. ``True``/``False`` rejected (bool ≠ int contract).
        genotype, treatment: Optional string identity columns. Default
            ``None``, mapping to ``np.nan`` in ``object``-dtype output
            columns (NOT literal string ``"None"``).

    Returns:
        ``pandas.DataFrame`` with 11 columns: the 8 row-identity
        columns followed by ``frame`` (int64), ``tip_x`` (float64),
        ``tip_y`` (float64). Exactly ``n_frames`` rows. ``frame`` is
        strict monotonic ascending from ``0`` to ``n_frames - 1``.

    Raises:
        TypeError: If any parameter has the wrong type (bool for numeric
            fields, str for a non-coercible field, ``np.random.RandomState``
            for ``random_state``, etc.). The exception message names the
            offending field.
        ValueError: If any parameter fails the value constraint (n_frames
            < 1, cadence_s ≤ 0, NaN / ±inf, handedness ∉ {+1, -1}, etc.).
            The exception message names the offending field.

    Examples:
        Default plate-001-matching trajectory::

            >>> df = generate_trajectory()
            >>> len(df)
            575
            >>> df["tip_x"].dtype == "float64"
            True

        Deterministic noise::

            >>> df_a = generate_trajectory(random_state=42)
            >>> df_b = generate_trajectory(random_state=42)
            >>> bool((df_a["tip_x"] == df_b["tip_x"]).all())
            True
    """
    # ------------------------------------------------------------------
    # D13 resolution-order: kwarg > constants > module-level default.
    # Sentinel-None args are resolved against constants (or ConstantsT()
    # for module defaults). After resolution, all 7 values are concrete
    # and proceed to validation.
    # ------------------------------------------------------------------
    constants = _check_constants(constants)
    _c = constants if constants is not None else ConstantsT()

    n_frames_resolved = n_frames if n_frames is not None else _c.SYNTHETIC_N_FRAMES
    cadence_s_resolved = cadence_s if cadence_s is not None else _c.SYNTHETIC_CADENCE_S
    amplitude_px_resolved = (
        amplitude_px if amplitude_px is not None else _c.SYNTHETIC_AMPLITUDE_PX
    )
    T_nutation_s_resolved = (
        T_nutation_s if T_nutation_s is not None else _c.SYNTHETIC_T_NUTATION_S
    )
    growth_rate_resolved = (
        growth_rate_px_per_frame
        if growth_rate_px_per_frame is not None
        else _c.SYNTHETIC_GROWTH_RATE_PX_PER_FRAME
    )
    noise_sigma_resolved = (
        noise_sigma_px if noise_sigma_px is not None else _c.SYNTHETIC_NOISE_SIGMA_PX
    )
    growth_axis_resolved = (
        growth_axis_angle_rad
        if growth_axis_angle_rad is not None
        else _c.SYNTHETIC_GROWTH_AXIS_ANGLE_RAD
    )

    # ------------------------------------------------------------------
    # Strict validation per design.md D8. Every field's error message
    # names the offending parameter so test 2.F can pattern-match.
    # ------------------------------------------------------------------
    n_frames_v = _check_int_strict("n_frames", n_frames_resolved, min_value=1)
    cadence_s_v = _check_float_finite("cadence_s", cadence_s_resolved, positive=True)
    amplitude_px_v = _check_float_finite(
        "amplitude_px", amplitude_px_resolved, non_negative=True
    )
    T_nutation_s_v = _check_float_finite(
        "T_nutation_s", T_nutation_s_resolved, positive=True
    )
    growth_rate_v = _check_float_finite(
        "growth_rate_px_per_frame", growth_rate_resolved
    )
    noise_sigma_v = _check_float_finite(
        "noise_sigma_px", noise_sigma_resolved, non_negative=True
    )
    growth_axis_v = _check_float_finite("growth_axis_angle_rad", growth_axis_resolved)
    handedness_v = _check_handedness(handedness)
    x0_v = _check_float_finite("x0_px", x0_px)
    y0_v = _check_float_finite("y0_px", y0_px)
    initial_phase_v = _check_float_finite("initial_phase_rad", initial_phase_rad)
    random_state_v = _check_random_state(random_state)

    series_v = _check_string_or_none("series", series, allow_none=False)
    sample_uid_v = _check_string_or_none("sample_uid", sample_uid, allow_none=False)
    timepoint_v = _check_string_or_none("timepoint", timepoint, allow_none=False)
    plate_id_v = _check_string_or_none("plate_id", plate_id, allow_none=False)
    plant_id_v = _check_int_strict("plant_id", plant_id)
    track_id_v = _check_int_strict("track_id", track_id)
    genotype_v = _check_string_or_none("genotype", genotype, allow_none=True)
    treatment_v = _check_string_or_none("treatment", treatment, allow_none=True)

    # ------------------------------------------------------------------
    # Closed-form trajectory math (design.md D1).
    #
    # Apex propagates along u_g at v_growth_per_s; lateral nutation
    # contributes A_lat·sin(handedness·ω·t + initial_phase_rad) along
    # u_lat = (-u_g[1], u_g[0]) — standard CCW 90° rotation.
    # ------------------------------------------------------------------
    frame_array = np.arange(n_frames_v, dtype=np.int64)
    t_array = frame_array.astype(np.float64) * cadence_s_v
    omega = 2.0 * math.pi / T_nutation_s_v
    v_growth_per_s = growth_rate_v / cadence_s_v
    A_lat = amplitude_px_v / 2.0

    u_g = np.array([math.cos(growth_axis_v), math.sin(growth_axis_v)], dtype=np.float64)
    u_lat = np.array([-u_g[1], u_g[0]], dtype=np.float64)

    phase = handedness_v * omega * t_array + initial_phase_v
    lat_offset = A_lat * np.sin(phase)

    tip_x = x0_v + v_growth_per_s * t_array * u_g[0] + lat_offset * u_lat[0]
    tip_y = y0_v + v_growth_per_s * t_array * u_g[1] + lat_offset * u_lat[1]

    # Additive iid Gaussian noise — per-axis σ = noise_sigma_px / √2 so
    # that the QC tier's xy-quadrature estimators recover noise_sigma_px.
    # Short-circuit when noise == 0 exactly to preserve caller-supplied
    # Generator state per D11.
    if noise_sigma_v > 0.0:
        rng = np.random.default_rng(random_state_v)
        sigma_per_axis = noise_sigma_v / math.sqrt(2.0)
        tip_x = tip_x + rng.normal(0.0, sigma_per_axis, n_frames_v)
        tip_y = tip_y + rng.normal(0.0, sigma_per_axis, n_frames_v)

    # ------------------------------------------------------------------
    # Build the output DataFrame with locked dtypes per design.md D3.
    # Identity columns are constant-valued (one unique 5-tuple per call
    # per D7); use pd.Series(..., dtype=object) for string columns so
    # None becomes np.nan (NOT literal "None").
    # ------------------------------------------------------------------
    n = n_frames_v
    df = pd.DataFrame(
        {
            "series": pd.Series([series_v] * n, dtype=object),
            "sample_uid": pd.Series([sample_uid_v] * n, dtype=object),
            "timepoint": pd.Series([timepoint_v] * n, dtype=object),
            "plate_id": pd.Series([plate_id_v] * n, dtype=object),
            "plant_id": pd.Series([plant_id_v] * n, dtype=np.int64),
            "track_id": pd.Series([track_id_v] * n, dtype=np.int64),
            "genotype": pd.Series(
                [genotype_v if genotype_v is not None else np.nan] * n, dtype=object
            ),
            "treatment": pd.Series(
                [treatment_v if treatment_v is not None else np.nan] * n, dtype=object
            ),
            "frame": frame_array,
            "tip_x": tip_x.astype(np.float64),
            "tip_y": tip_y.astype(np.float64),
        }
    )

    # Enforce declared column order per spec scenario "Default call returns
    # 575-row DataFrame with the documented schema". The column order
    # follows ROW_IDENTITY_COLUMNS + REQUIRED_PER_FRAME_COLUMNS (both
    # imported from _types).
    df = df[list(ROW_IDENTITY_COLUMNS) + list(REQUIRED_PER_FRAME_COLUMNS)]

    return df
