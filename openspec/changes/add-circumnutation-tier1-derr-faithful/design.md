# Design: add-circumnutation-tier1-derr-faithful

## Context

This is PR #6 in the circumnutation analysis program tracked by `docs/circumnutation/roadmap.md`. Foundation (PR #1) shipped contracts, the cmor1.5-1.0 mother-wavelet default, and the locked module callable surface in the `Package layout` requirement. Tier 0 (PR #2) shipped raw kinematic traits + the `_noise.compute_sg_residual_xy` helper + `_geometry.compute_psi_g` helper. QC tier (PR #3) shipped track-level signal-quality traits, deferring `cadence_nyquist_ratio` to PR #6 (this PR) on grounds that the cadence-Nyquist check requires the spectral-peak period from Tier 1. PR #4 shipped the synthetic trajectory generator (`synthetic.generate_trajectory`) which becomes PR #6's primary Layer-1 testbed. PR #5 shipped the temporal CWT primitives (`compute_scaleogram` + `extract_ridge` + `ScaleogramResult` + `RidgeResult`) — but emitted no traits.

This PR replaces the `nutation.compute` / `_geometry.project_to_growth_axis_perpendicular` / `_noise.compute_fourier_noise_floor` / `temporal_cwt.smooth_ridge` surfaces with the first concrete Tier 1 trait emission per `docs/circumnutation/theory.md` §7.2, composing PR #5's CWT primitives + PR #2's growth-axis machinery. PR #6 emits **6 traits** in one new module + closes follow-up issue [#214](https://github.com/talmolab/sleap-roots/issues/214) (ridge-tracking continuity post-filter) + introduces `nutation.py` as the public-API Tier 1 module.

The 6 emitted traits:

| Trait | Source | Units | Type |
|---|---|---|---|
| `T_nutation_median` | theory.md §7.2 | s (NaN if `is_nutating==False`) | float |
| `T_nutation_iqr` | theory.md §7.2 | s (NaN if `is_nutating==False`) | float |
| `A_nutation_envelope_max` | theory.md §7.2 | px (NaN if `is_nutating==False`) | float |
| `band_power_ratio` | theory.md §7.2 + §7.6 | dimensionless (always populated) | float |
| `noise_floor_estimate` | theory.md §7.6 + CC-8 | amplitude units (always populated) | float |
| `is_nutating` | theory.md §7.6 (`band_power_ratio > 3·noise_floor_estimate`) | bool (always populated) | bool |
| `period_residual_vs_derr_reference` | theory.md §7.2 | dimensionless (always populated; deviation from configurable reference) | float |
| `cadence_nyquist_ratio` | theory.md §6.5 (moved from PR #3) | dimensionless (always populated; diagnostic for cadence adequacy) | float |

That is 8 trait columns total (6 "headline" emissions + 2 precursors), matching the "5 Tier 1 + cadence_nyquist_ratio moved from PR #3 + is_nutating from §7.6 + the FFT noise floor as a named precursor" intent.

Theory anchors:

- `docs/circumnutation/theory.md` §7.2 (Tier 1 trait table — 5 traits, all consumed by `nutation.compute`)
- `docs/circumnutation/theory.md` §7.6 (QC table: `noise_floor_estimate`, `is_nutating = band_power_ratio > BAND_POWER_NOISE_RATIO · noise_floor_estimate`, `coi_fraction_t1`)
- `docs/circumnutation/theory.md` §6.5 (Cadence-Nyquist check — temporal cadence vs nutation period; trait moved to PR #6 per the spec deferral)
- `docs/circumnutation/theory.md` §3.5 (BM2016 Eq. 20: `ψ_g(t) = arctan(dx_a/dt, dy_a/dt)` — load-bearing physical justification for tangent-angle decomposition; the lateral-projection helper here is the geometric prerequisite for projecting onto the growth-axis perpendicular)
- `docs/circumnutation/preliminary_results_2026-05-07.md` Summary + §1.2 + §2.1 + §4.4 (empirical anchors; plate 001 Nipponbare T_nutation ≈ 3333 s; ~9.9 px peak-to-peak nutation amplitude; growth-axis drift dominates raw `tip_x` and motivates the lateral-projection requirement)
- Derr Sept-2025 oracle PDF at `c:\vaults\sleap-roots\circumnutation\external_code\derr_wavelets\sept_2025_outputs\5minutes_average_period=3333s.pdf` — the Layer-2 regression target for `period_residual_vs_derr_reference` (the documented spectral peak is 3333 s; per CC-7 the acceptance criterion is `|period_residual_vs_derr_reference| < DERR_MATCH_TOLERANCE = 0.02`, i.e. T = 3333 s ± 2%)
- Cross-cutting concerns touched: **CC-7 (lateral coordinate)**, **CC-8 (Fourier noise floor)**, CC-2 (constants), CC-3 (pure-pixel emission — preserved), CC-6 (determinism), CC-9 (logging)

Predecessor PR #5's GREEN-phase Reconciliation Appendix documented the empirical observation that raw `tip_x` on plate-001 carries ~70-170 px of growth-axis drift dominating the ~10 px nutation signal, justifying the deferral of the lateral-projection to PR #6. This PR ships that projection.

## Goals / Non-Goals

**Goals:**

- Implement `sleap_roots.circumnutation.nutation.compute(trajectory_df, coordinate='lateral', constants=None) -> pd.DataFrame` as the public Tier 1 trait-emission entry point. Mirrors `kinematics.compute` / `qc.compute` precedent (signature + per-track groupby loop + per-track helper).
- Implement `sleap_roots.circumnutation._geometry.project_to_growth_axis_perpendicular(x, y) -> np.ndarray` as the lateral-projection helper. Returns the 1D lateral position time series (length n, dtype float64). Implementation: estimate growth axis from net displacement (matching the convention used by Tier 0's growth-axis reliability gate), construct the perpendicular unit vector, project centered positions.
- Implement `sleap_roots.circumnutation._noise.compute_fourier_noise_floor(x, cadence_s, t_nutation_median_s, factor) -> float` per CC-8. Uses `scipy.fft.rfft` on the 1D input signal; returns median amplitude over frequencies > `factor / t_nutation_median_s`. Single-source-of-truth helper alongside the existing `compute_sg_residual_xy` / `compute_d2_residual_xy` / `compute_msd_residual_xy`.
- Implement `sleap_roots.circumnutation.temporal_cwt.smooth_ridge(ridge_result, window=None, constants=None) -> RidgeResult` as the **third composable CWT primitive**. Closes issue [#214](https://github.com/talmolab/sleap-roots/issues/214). Applies `scipy.ndimage.median_filter(ridge_result.periods_s, size=window)` and returns a new `RidgeResult`. Smooths `periods_s` only; `amplitudes`, `powers`, `in_coi`, `frame_indices` carry through unchanged (rationale in D4).
- Emit the 8 trait columns above per the schema table; honor the **NaN-gating semantics** of D5 (5 meaning-dependent traits become NaN when `is_nutating == False`).
- Add 5 new `ConstantsT`-overridable defaults to `_constants.py`: `RIDGE_CONTINUITY_FILTER_WINDOW=5`, `NOISE_FLOOR_OUT_OF_BAND_FACTOR=5.0`, `BAND_POWER_BAND_LOW_FACTOR=0.5`, `BAND_POWER_BAND_HIGH_FACTOR=2.0`, `DERR_EXPECTED_PERIOD_S=3333.0`. Bump `_CONSTANTS_VERSION` 4 → 5.
- Land a single test file `tests/test_circumnutation_nutation.py` mirroring PR #5's 8-section taxonomy (§2.A schema, §2.B determinism canary, §2.C synthetic parameter recovery, §2.D ridge-continuity post-filter sanity, §2.E `band_power_ratio` + `is_nutating` sanity, §2.F validation/errors, §2.G `ConstantsT` override + resolution-order + `coordinate=` parameter, §2.H reference-fixture sanity including Layer-2 Derr forensic match).
- Foundation-test migration (`tests/test_circumnutation_foundation.py`): add `nutation` to `IMPLEMENTATIONS_WITH_CONSTANTS_KWARG` (4 → 5 entries); bump `_CONSTANTS_VERSION` assertion 4 → 5 (with test rename matching the PR #5 rename pattern); add `nutation` to the `test_module_logger_is_namespaced` parametrize list (per the Copilot-precedent fix from PR #4/PR #5); update the comment block to record PR #6's implementation count.

### Scope discipline — which PR owns what

theory.md §7.2 names 5 Tier 1 traits and §7.6 names 3 QC traits (`coi_fraction_t1`, `is_nutating`, `noise_floor_estimate`) that all consume PR #5's machinery. PR #6 emits the 5 Tier 1 traits + 2 of the 3 §7.6 traits (`is_nutating`, `noise_floor_estimate`) + `cadence_nyquist_ratio` moved from PR #3. PR #6 does NOT emit `coi_fraction_t1` (that stays in QC tier — `mean(coi_mask)` from `ScaleogramResult`; PR #6 doesn't extend qc.py).

| Trait | Owning PR | PR #6's contribution |
|---|---|---|
| `T_nutation_median`, `T_nutation_iqr` | PR #6 | Both: median + IQR of `smooth_ridge(extract_ridge(scaleogram)).periods_s[~in_coi]` |
| `A_nutation_envelope_max` | PR #6 | `max(extract_ridge(scaleogram).amplitudes[~in_coi])` |
| `band_power_ratio` | PR #6 | `power_in_[0.5T, 2T]_band / total_power` from FFT amplitude spectrum |
| `noise_floor_estimate` | PR #6 | `compute_fourier_noise_floor` helper output |
| `is_nutating` | PR #6 | `band_power_ratio > BAND_POWER_NOISE_RATIO * noise_floor_estimate` |
| `period_residual_vs_derr_reference` | PR #6 | `(T_nutation_median - DERR_EXPECTED_PERIOD_S) / DERR_EXPECTED_PERIOD_S` |
| `cadence_nyquist_ratio` | PR #6 (moved from PR #3) | `cadence_s / T_nutation_median` |
| `coi_fraction_t1` | QC tier (future PR) | NOT PR #6; mean of `ScaleogramResult.coi_mask` |

**Non-Goals:**

- **No `coi_fraction_t1` emission.** Per theory.md §7.6 it belongs to QC tier; not in PR #6 scope. A future PR extending qc.py will add it; PR #6 leaves `ScaleogramResult.coi_mask` accessible for that extension.
- **No new Tier 1 trait beyond the 8 listed.** Sub-scale ridge-period refinement (parabolic), harmonic detection, FFT vs CWT alternative spectra for `band_power_ratio` — all deferred. Future-issue notes only.
- **No Mallat 1999 ridge-following algorithm for #214.** PR #6 ships median-filter (option (a) in #214) per Q4. Acceptance is `T_nutation_iqr_post_filter < T_nutation_iqr_raw` on plate-001. Mallat upgrade is deferred to a follow-up issue iff median-filter empirically fails on multi-plate data.
- **No Layer-2 numeric forensic match against Derr's raw scaleogram.** Q6 chose a fractional-period-residual against a CONSTANT (`DERR_EXPECTED_PERIOD_S = 3333.0`) because Derr Sept-2025 outputs are PDF/PNG only — no numeric arrays available. A follow-up issue is filed: "Derr Layer-2 forensic-match upgrade pending raw-data delivery" (see Follow-up Issues).
- **No new fixtures.** §2.H reuses the existing PR #2 Nipponbare proofread fixture (`tests/data/circumnutation_nipponbare_plate_001/plate_001_greyscale.tracked_proofread.slp` — verified zero-NaN, zero-frame-gap across all 6 tracks). PR #4's `synthetic.generate_trajectory` produces Layer-1 ground-truth.
- **No modification of qc.py.** `cadence_nyquist_ratio` is MOVED to PR #6 per the PR #3 spec deferral — but PR #3 never emitted the trait, so there's no removal needed. PR #6 adds it to nutation.csv, not qc.csv.
- **No spec change to `Pure-pixel pipeline output convention` requirement.** All emitted trait values are in pure pixel / dimensionless units; no calibrated-mm interpretation. The `coordinate=` parameter is geometric, not unit-changing.
- **No re-export at the package level.** `nutation.compute` is accessed via `from sleap_roots.circumnutation import nutation`. Mirrors `kinematics`, `qc`, `synthetic`, `temporal_cwt` conventions; top-level `sleap_roots` re-exports stay limited to `CircumnutationInputs` + `convert_to_mm` + `ROW_IDENTITY_COLUMNS`.

## Decisions

### D1. New `nutation.py` module — descriptive-noun naming

Tier 1 trait emission lives in a NEW module `sleap_roots/circumnutation/nutation.py` with the public `compute(trajectory_df, coordinate='lateral', constants=None) -> pd.DataFrame` function. The module name follows the existing descriptive-noun convention (`kinematics`, `qc`, `synthetic`, `temporal_cwt`) — NOT `tier1.py`, which is internal-taxonomy jargon, and NOT `derr.py`, which ties module naming to a person's name rather than the scientific concept.

```python
# sleap_roots/circumnutation/nutation.py
def compute(
    trajectory_df: pd.DataFrame,
    coordinate: str = "lateral",
    constants: Optional[ConstantsT] = None,
) -> pd.DataFrame:
    """Emit Tier 1 nutation traits per theory.md §7.2.

    Returns one row per (series, sample_uid, plate_id, plant_id, track_id),
    with 8 trait columns: T_nutation_median, T_nutation_iqr,
    A_nutation_envelope_max, band_power_ratio, noise_floor_estimate,
    is_nutating, period_residual_vs_derr_reference, cadence_nyquist_ratio.

    Per D5 NaN-gating semantics: when is_nutating == False, the 5
    meaning-dependent traits (T_nutation_median, T_nutation_iqr,
    A_nutation_envelope_max, period_residual_vs_derr_reference, cadence_nyquist_ratio)
    are emitted as NaN.

    Parameters
    ----------
    trajectory_df : pd.DataFrame
        See _types._validate_trajectory_df contract.
    coordinate : {"lateral", "x", "y"}, default "lateral"
        Which 1D time series feeds the temporal CWT. "lateral" projects
        (tip_x, tip_y) onto the growth-axis perpendicular via
        _geometry.project_to_growth_axis_perpendicular; "x" / "y" pass
        raw coordinates (diagnostic use).
    constants : Optional[ConstantsT], default None
        ConstantsT override; standard two-tier resolution.

    Returns
    -------
    pd.DataFrame
        ROW_IDENTITY_COLUMNS + 8 trait columns, in declared order.
    """
```

**Module-level constants** (declared at module top, mirroring `_NUTATION_TRAIT_COLUMNS` from kinematics.py / `_QC_TRAIT_COLUMNS` from qc.py precedent):

```python
_NUTATION_TRAIT_COLUMNS: tuple[str, ...] = (
    "T_nutation_median",
    "T_nutation_iqr",
    "A_nutation_envelope_max",
    "band_power_ratio",
    "noise_floor_estimate",
    "is_nutating",
    "period_residual_vs_derr_reference",
    "cadence_nyquist_ratio",
)
_COORDINATE_CHOICES: frozenset[str] = frozenset({"lateral", "x", "y"})
```

**Per-track loop** (mirrors kinematics.py:395-401 / qc.py:398-403 precedent exactly):

```python
trait_rows: list[dict[str, Any]] = []
for key, group in trajectory_df.groupby(
    list(_IDENTITY_5_TUPLE), dropna=False, sort=False
):
    traits = _compute_one_track(
        group, coordinate=coordinate, constants=resolved_constants
    )
    identity = dict(zip(_IDENTITY_5_TUPLE, key))
    trait_rows.append({**identity, **traits})
```

**Alternatives considered:**

- *`tier1.py`*. Rejected per Q1 user feedback ("tier1 is internal-facing taxonomy, very NOT clean / interpretable / makes the API more difficult").
- *Extend `temporal_cwt.py` with `compute_traits()`*. Rejected per Q1 + PR #5's scope-discipline precedent: temporal_cwt.py owns CWT machinery primitives; mixing trait emission into it conflates two concerns.
- *`derr.py`*. Rejected per Q1: ties module naming to a person rather than the scientific concept.
- *`spectral.py`*. Rejected per Q1: names the technique, not the quantity. Less aligned with the existing `kinematics.py`-noun precedent.

### D2. `coordinate=` parameter + `_geometry.project_to_growth_axis_perpendicular` helper

The `coordinate` parameter accepts one of `{"lateral", "x", "y"}` (per Q2 three-value enum). Default `"lateral"` per CC-7.

- `coordinate="x"` → `signal = group[_TIP_X_COLUMN].to_numpy(dtype=np.float64)`
- `coordinate="y"` → `signal = group[_TIP_Y_COLUMN].to_numpy(dtype=np.float64)`
- `coordinate="lateral"` → `signal = _geometry.project_to_growth_axis_perpendicular(group[_TIP_X_COLUMN].to_numpy(), group[_TIP_Y_COLUMN].to_numpy())`

**The new `_geometry.project_to_growth_axis_perpendicular` helper:**

```python
# sleap_roots/circumnutation/_geometry.py (new function added after compute_psi_g)
def project_to_growth_axis_perpendicular(
    x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """Project (x, y) tip positions onto the growth-axis perpendicular.

    Estimates the growth-axis unit vector from net displacement of
    (x[0], y[0]) to (x[-1], y[-1]), constructs the perpendicular unit
    vector (90° rotation: ``(-u_g[1], u_g[0])``), and projects the
    centered positions onto that perpendicular. Returns the 1D lateral
    position time series.

    Parameters
    ----------
    x, y : np.ndarray
        1D float arrays of equal length, all finite.

    Returns
    -------
    np.ndarray
        1D float64 array of same length as inputs; lateral position
        (px) centered at zero by construction.

    Raises
    ------
    ValueError
        If x and y have different lengths.
        If net displacement length is zero (stationary track —
        growth axis undefined).
        If x or y contain non-finite values.
    """
```

**Algorithm (locked):**

```python
# Validate inputs (length match, finite-only, length ≥ 2).
n = len(x)
dx_net = x[-1] - x[0]
dy_net = y[-1] - y[0]
net_length = math.hypot(dx_net, dy_net)
if net_length == 0.0:
    raise ValueError(
        "project_to_growth_axis_perpendicular: net displacement is "
        "zero; growth axis is undefined for a stationary track."
    )
u_g_x = dx_net / net_length      # growth-axis unit vector x-component
u_g_y = dy_net / net_length      # growth-axis unit vector y-component
u_perp_x = -u_g_y                # 90° rotation: perpendicular x
u_perp_y =  u_g_x                # 90° rotation: perpendicular y
# Center (zero mean) and project onto perpendicular.
x_centered = x - np.mean(x)
y_centered = y - np.mean(y)
lateral = x_centered * u_perp_x + y_centered * u_perp_y
return lateral.astype(np.float64, copy=False)
```

**Why net-displacement (not per-frame tangent or smoothed-velocity):**

- Net-displacement growth axis is what Tier 0's `_geometry.compute_growth_axis_reliability` already uses for the `growth_axis_unreliable` flag. PR #6 reuses the SAME convention for SAME interpretability.
- Per-frame tangent (`ψ_g`) is inherently noisy and changes the projection's geometry every frame. The CWT input would no longer be a "position time series" but a "frame-by-frame-rotated position", which is harder to interpret spectrally.
- Smoothed-velocity is a halfway design; it adds a smoothing-window parameter without a clear benefit over net-displacement for plate-001-scale tracks (~575 frames).

**Why centering before projection:**

- Without centering, the projection inherits the mean position of the trajectory, which adds a DC offset to the lateral signal. DC offset doesn't affect the CWT (cmor1.5-1.0 has zero mean), but it pollutes the FFT amplitude spectrum used by `noise_floor_estimate` and `band_power_ratio`. Centering ensures the FFT-based traits operate on a zero-mean signal.

**Failure mode — stationary track:** if net displacement is zero (plant didn't grow at all over the recording window), the growth axis is undefined. The helper raises `ValueError`. Downstream `nutation.compute` catches and converts to NaN trait emission (no row dropped — the identity row is preserved with all-NaN trait values, matching the kinematics.py / qc.py convention for un-computable tracks).

**Alternatives considered:**

- *Helper in `nutation.py` as `_project_to_lateral`*. Rejected per Q2: violates the `_geometry.py` single-source-of-truth pattern; PR #11 (parametric) will likely need the same projection and would have to duplicate.
- *4-value enum including `"longitudinal"` + symmetric `project_to_growth_axis_parallel`*. Rejected per Q2 (YAGNI): no downstream consumer needs the longitudinal projection in PR #6 scope; can be added when first caller emerges.
- *Per-frame tangent projection*. Rejected (see "Why net-displacement" above).

### D3. `_noise.compute_fourier_noise_floor` per CC-8

```python
# sleap_roots/circumnutation/_noise.py (new function added after compute_msd_residual_xy)
def compute_fourier_noise_floor(
    x: np.ndarray,
    cadence_s: float,
    t_nutation_median_s: float,
    factor: float,
) -> float:
    """Median Fourier amplitude over high-frequency band (CC-8).

    Per theory.md §7.6 + roadmap.md CC-8: the noise-floor estimate is
    the median of the Fourier amplitude spectrum over frequencies
    f > factor / t_nutation_median_s (i.e., out-of-band relative to
    the candidate nutation period).

    Parameters
    ----------
    x : np.ndarray
        1D float array, finite-only. Typically the lateral-projected
        position time series.
    cadence_s : float
        Sample cadence (s). Positive finite.
    t_nutation_median_s : float
        Candidate nutation period (s) used to set the out-of-band
        cutoff. Positive finite.
    factor : float
        Out-of-band cutoff factor. The noise floor is computed over
        frequencies > factor / t_nutation_median_s. Default convention
        in nutation.compute is NOISE_FLOOR_OUT_OF_BAND_FACTOR (= 5.0).

    Returns
    -------
    float
        Median amplitude over the high-frequency band. ``np.nan`` if
        the input is too short (< 2 frames) or if the high-frequency
        band is empty (cutoff exceeds the Nyquist frequency).
    """
    if len(x) < 2:
        logger.debug(
            "compute_fourier_noise_floor: len(x)=%d < 2, returning NaN",
            len(x),
        )
        return float("nan")
    spectrum = np.abs(scipy.fft.rfft(x))
    freqs = scipy.fft.rfftfreq(len(x), d=cadence_s)
    f_cut = factor / t_nutation_median_s
    band_mask = freqs > f_cut
    if not band_mask.any():
        logger.debug(
            "compute_fourier_noise_floor: empty out-of-band region "
            "(f_cut=%.6f > nyquist=%.6f), returning NaN",
            f_cut, freqs[-1],
        )
        return float("nan")
    return float(np.median(spectrum[band_mask]))
```

**Why `scipy.fft.rfft` (not `numpy.fft.rfft`):** scipy.fft is the recommended modern interface (numpy.fft predates rfft canonicalization); scipy.fft is what we use elsewhere in the codebase (PR #1 imported scipy as a runtime dep). Consistent API surface across the package.

**Why `np.abs(rfft)` (amplitude, not PSD):** CC-8 says "median of Tier-1 Fourier AMPLITUDE spectrum". scipy.signal.welch returns PSD (power-spectral-density per Hz) which is dimensionally different. Sticking to amplitude matches CC-8 verbatim AND the `band_power_ratio` integrand (D7) uses amplitude² → no unit-mismatch.

**Why `np.median` (not `np.mean`):** robust to outlier high-frequency components (single-bin spikes don't dominate). Consistent with the existing `_noise.compute_d2_residual_xy` median-based estimator.

**Why **input is `x`** (not the scaleogram or some derived array):** keeps the helper general-purpose. The signal-source decision (lateral vs raw `tip_x`) is made by `nutation.compute` BEFORE calling this helper; the helper just computes a median over an out-of-band FFT slice.

**Edge cases:**

- `len(x) < 2`: `rfft` is undefined → return NaN + debug log
- Cutoff above Nyquist: empty band → return NaN + debug log
- All-zero input: `rfft = [0+0j, 0+0j, ...]` → `np.abs = [0, 0, ...]` → median = 0.0 (legitimate; downstream `is_nutating` will be `band_power_ratio > 0` if any signal at all)
- `t_nutation_median_s` is NaN (e.g., when called BEFORE T is determined — not actually called this way; surfaced as a contract clarity note): if NaN is passed, `f_cut = factor / NaN = NaN`, `freqs > NaN` is all-False → empty band → return NaN

**Alternatives considered:**

- *Reuse PR #5 CWT scaleogram out-of-band columns*. Rejected per Q3: wavelet ≠ Fourier (CC-8 says "Fourier amplitude" verbatim); CWT scale resolution at high frequencies is sparse; would require justification in design.md to deviate from CC-8.
- *Welch PSD*. Rejected per Q3: introduces window-choice tunables; PSD has different units than amplitude; deviates further from CC-8 wording.

### D4. `temporal_cwt.smooth_ridge` as third composable primitive

Per Q4 (Option B: modular + reusable + testable) and issue [#214](https://github.com/talmolab/sleap-roots/issues/214):

```python
# sleap_roots/circumnutation/temporal_cwt.py (new function added after extract_ridge)
def smooth_ridge(
    ridge_result: RidgeResult,
    window: Optional[int] = None,
    constants: Optional[ConstantsT] = None,
) -> RidgeResult:
    """Median-filter ridge periods to suppress scale-hopping artifacts.

    Issue #214: PR #5's per-frame argmax ridge can hop discontinuously
    between adjacent CWT scales at frames where two harmonics have
    similar amplitude. This post-filter applies
    ``scipy.ndimage.median_filter`` over a configurable window to the
    ``periods_s`` field, returning a new RidgeResult with smoothed
    periods. ``amplitudes``, ``powers``, ``in_coi``, and
    ``frame_indices`` are carried through unchanged.

    Parameters
    ----------
    ridge_result : RidgeResult
        The raw ridge from extract_ridge.
    window : Optional[int], default None
        Window size for the median filter (odd integer ≥ 1). If None,
        uses ``resolved_constants.RIDGE_CONTINUITY_FILTER_WINDOW``
        (default 5).
    constants : Optional[ConstantsT], default None
        ConstantsT override; standard two-tier resolution.

    Returns
    -------
    RidgeResult
        New RidgeResult with smoothed periods_s; other fields
        carried through.

    Raises
    ------
    TypeError
        If ridge_result is not a RidgeResult or window is not int.
    ValueError
        If window is even or < 1.
    """
```

**Why smooth `periods_s` only:**

- Issue #214 is specifically about period-IQR being polluted by ridge scale-hopping — the load-bearing acceptance criterion is `T_nutation_iqr_post_filter < T_nutation_iqr_raw`.
- `amplitudes` and `powers` are computed at the ridge cell regardless of which scale it's at. Smoothing amplitudes would lower the peak that `A_nutation_envelope_max` measures — distorting a downstream trait without a corresponding accuracy benefit.
- `in_coi` is a function of ridge_scale_idx, which is implicit in the smoothed periods. Recomputing `in_coi` from the smoothed periods would require access to the COI mask grid — we don't have it from a RidgeResult alone (it lives on ScaleogramResult). Decision: pass `in_coi` through unchanged. The smoothing window is small (5 frames) so the in-coi-vs-out-of-coi boundary stays accurate within ±2 frames.
- `frame_indices` and `powers = amplitudes²` are tautologies that don't change under period-smoothing.

**Why median (not mean) filter:** median is robust to single-frame outliers (the exact failure mode #214 describes — sharp scale-hops). A mean filter would average across the discontinuity, producing intermediate periods that don't exist on the scale grid. Median preserves a valid grid value (or near-grid value within median-of-window precision).

**Why window default = 5:** issue #214's suggested algorithm explicitly references `scipy.ndimage.median_filter(ridge_scale_idx, size=5)`. Five frames at plate-001's 300 s cadence = 25 minutes of smoothing, which is ~0.75% of the 3333 s nutation period — well within "tighten without smearing biological signal" territory.

**Implementation:**

```python
def smooth_ridge(ridge_result, window=None, constants=None):
    resolved = constants if constants is not None else ConstantsT()
    if window is None:
        window = resolved.RIDGE_CONTINUITY_FILTER_WINDOW
    _validate_window(window)  # int type, ≥1, odd
    logger.debug(
        "smooth_ridge(n_frames=%d, window=%d)",
        len(ridge_result.periods_s), window,
    )
    smoothed_periods = scipy.ndimage.median_filter(
        ridge_result.periods_s,
        size=window,
        mode="nearest",  # boundary policy: nearest-neighbor extension
    )
    return RidgeResult(
        frame_indices=ridge_result.frame_indices,
        periods_s=smoothed_periods,
        amplitudes=ridge_result.amplitudes,
        powers=ridge_result.powers,
        in_coi=ridge_result.in_coi,
    )
```

**Boundary policy:** `mode="nearest"` (rather than `"reflect"` or `"constant"`) — preserves the boundary periods rather than reflecting/zeroing them. Boundary frames are already inside-COI in most CWT applications, so they're filtered out downstream by `~in_coi` mask anyway; the choice of boundary policy is essentially diagnostic.

**`constants=None` accepted for forward-compatibility** even though the only currently-used field is `RIDGE_CONTINUITY_FILTER_WINDOW` and the `window=` kwarg already handles the per-call override case. The dual-mechanism (kwarg precedence, then constants, then default) mirrors PR #5's `extract_ridge` two-tier resolution.

**Alternatives considered:**

- *Inline median_filter in nutation.py*. Rejected per Q2 (Elizabeth's preference for modular / reusable / testable functions).
- *`smooth_window` parameter on `extract_ridge`*. Rejected per Q4: bumps PR #5 spec-locked contract; couples ridge extraction to smoothing.
- *Mallat 1999 §4.4.2 ridge-following algorithm*. Rejected per Q4 (YAGNI for PR #6; complexity is unjustified until median-filter is empirically shown to fail). Tracked as a future-possible-issue.
- *Smooth both periods AND amplitudes*. Rejected per the design note above (would distort `A_nutation_envelope_max`).
- *Recompute `in_coi` from smoothed periods*. Rejected — requires access to the COI mask grid that lives on ScaleogramResult; pass-through is correct within smoothing-window precision.

### D5. `is_nutating` ownership in nutation.py + NaN-gating semantics

Per Q5 (NaN-gate ownership) and S4 round-1 reconciliation (which traits actually get NaN-gated): `nutation.py` owns `is_nutating` and uses it as a gate. When `is_nutating == False`, ONLY the 3 strictly biological-meaning-dependent traits become NaN; the 2 diagnostic ratios (`cadence_nyquist_ratio`, `period_residual_vs_derr_reference`) and the 3 precursors stay populated as engineering diagnostics:

| Trait | When `is_nutating==False` | When `is_nutating==True` | Why this gating choice |
|---|---|---|---|
| `T_nutation_median` | NaN | computed | biological-meaning-dependent |
| `T_nutation_iqr` | NaN | computed | biological-meaning-dependent |
| `A_nutation_envelope_max` | NaN | computed | biological-meaning-dependent |
| `period_residual_vs_derr_reference` | populated | populated | diagnostic: "where would the spectral peak land if trusted" |
| `cadence_nyquist_ratio` | populated | populated | diagnostic: engineering-question "could we have observed nutation?" — answer-able regardless of biology (per S4 round-1 reviewer Sci-I3) |
| `band_power_ratio` | populated | populated | precursor to `is_nutating` gate |
| `noise_floor_estimate` | populated | populated | precursor to `is_nutating` gate |
| `is_nutating` | False | True | gate itself |

**Compute-vs-output distinction:** the short-circuit is at the OUTPUT layer, not the COMPUTE layer. `T_nutation_median` MUST be computed first because both `band_power_ratio` (band edges at `[0.5T, 2T]`) and `noise_floor_estimate` (cutoff at `5/T`) depend on it. After computing `is_nutating`, the trait dictionary is constructed with NaN substitutions for the meaning-dependent fields when the gate fails.

**Sequencing inside `_compute_one_track`:**

```python
def _compute_one_track(group, cadence_s, coordinate, constants):
    # S8' round-2: cadence_s is an explicit parameter threaded from
    # nutation.compute (which takes it as a positional argument), NOT
    # via trajectory_df.attrs.
    #
    # 1. Project to signal axis.
    raw_signal = _select_signal(group, coordinate)  # 1D float64

    # 1b. SG-detrend per preliminary_results §3.4 (S1 round-1 addition).
    #     Removes slow centerline drift (which dominates raw `tip_x` by ~20x
    #     per PR #5's GREEN-phase observation). Output is the high-frequency
    #     OSCILLATION component (raw - SG-smoothed-low-pass).
    signal = _noise.compute_sg_detrended(
        raw_signal,
        window=constants.SG_WINDOW_DETREND,        # default 23 (PR #2/#3)
        polynomial_order=constants.SG_DEGREE,      # default 3  (PR #2/#3)
    )

    # 2. CWT primitives.
    scaleogram = temporal_cwt.compute_scaleogram(
        signal, cadence_s=cadence_s, constants=constants)
    raw_ridge = temporal_cwt.extract_ridge(scaleogram, constants=constants)
    smoothed_ridge = temporal_cwt.smooth_ridge(raw_ridge, constants=constants)

    # 3. COI-masked period statistics from smoothed ridge.
    interior_periods = smoothed_ridge.periods_s[~smoothed_ridge.in_coi]
    T_nutation_median = float(np.nanmedian(interior_periods))  # candidate value
    T_nutation_iqr = float(scipy.stats.iqr(
        interior_periods, nan_policy="omit"))  # candidate value

    # 4. COI-masked amplitude peak.
    interior_amps = raw_ridge.amplitudes[~raw_ridge.in_coi]
    A_nutation_envelope_max = float(np.max(interior_amps))  # candidate value

    # 5. FFT noise floor (CC-8).
    noise_floor_estimate = _noise.compute_fourier_noise_floor(
        signal,
        cadence_s=cadence_s,  # threaded as param, not via group.attrs
        t_nutation_median_s=T_nutation_median,
        factor=constants.NOISE_FLOOR_OUT_OF_BAND_FACTOR,
    )

    # 6. Band power ratio.
    band_power_ratio = _compute_band_power_ratio(
        signal,
        cadence_s=cadence_s,  # threaded as param, not via group.attrs
        t_nutation_median_s=T_nutation_median,
        constants=constants,
    )

    # 7. is_nutating gate.
    is_nutating = bool(
        band_power_ratio > constants.BAND_POWER_NOISE_RATIO * noise_floor_estimate
    )

    # 8. Derived traits.
    period_residual_vs_derr_reference = (
        (T_nutation_median - constants.DERR_EXPECTED_PERIOD_S)
        / constants.DERR_EXPECTED_PERIOD_S
    )  # candidate value
    cadence_nyquist_ratio = (
        cadence_s / T_nutation_median  # cadence_s threaded as param
    )  # candidate value

    # 9. NaN-gate ONLY the 3 biological-meaning-dependent traits per S4
    #    (round-1 reconciliation). cadence_nyquist_ratio stays populated as a
    #    diagnostic for *why* is_nutating failed (cadence aliasing vs noise);
    #    period_residual_vs_derr_reference stays populated as a diagnostic for
    #    "where would the spectral peak land if we trusted it" — both useful
    #    in QC investigations regardless of biological nutation presence.
    if not is_nutating:
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
        "period_residual_vs_derr_reference": period_residual_vs_derr_reference,
        "cadence_nyquist_ratio": cadence_nyquist_ratio,
    }
```

**The cadence_s field (S8' round-2 REVERSAL):** the round-1 S8 decision (extend `_validate_trajectory_df` to require `trajectory_df.attrs["cadence_s"]`) was REVERSED in round-2 reconciliation after Architecture-B1/I4 + TDD-I3 found three problems:

1. **Scope was 7× under-estimated**: 68 `kinematics.compute(...)` / `qc.compute(...)` test call sites build `trajectory_df` fixtures without setting attrs; updating all of them (via builder helpers + inline-construction audits + pandas-attrs-propagation verification) is ~25-40 LOC, not the ~10 I estimated.
2. **Contract regression**: `kinematics.py:34` and `qc.py:34` explicitly document "NO `cadence_s` input"; making the shared validator demand an attribute neither module consumes punishes already-merged PRs for PR #6's needs.
3. **Future-tier benefit claim doesn't hold**: PR #7 (`psi_g.compute(x, y, constants=None)`), PR #11 (`parametric.compute(tier3_df, R_px, omega, Delta_phi)`), and PR #14 (`pipeline.compute_traits(inputs, constants=None)` — already reads from `inputs.cadence_s` on `CircumnutationInputs`) do NOT take `cadence_s` via attrs. The "metadata on data" architectural cleanness has no concrete future caller.

**S8' resolution**: `nutation.compute` takes `cadence_s: float` as an explicit positional parameter, matching `temporal_cwt.compute_scaleogram(x, cadence_s, constants=None)`'s PR #5 precedent. Signature is:

```python
def compute(
    trajectory_df: pd.DataFrame,
    cadence_s: float,
    coordinate: str = "lateral",
    constants: Optional[ConstantsT] = None,
) -> pd.DataFrame: ...
```

Internal validation: a private helper `_check_cadence_s(cadence_s)` (mirroring `_noise.compute_sg_residual_xy`'s pattern) validates positive finite float at the top of `compute`. The `_compute_one_track(group, cadence_s, coordinate, constants)` helper takes cadence_s as an explicit kwarg threaded from `compute`'s top. NO pandas `.attrs` access anywhere; NO `_validate_trajectory_df` extension; NO PR #2/PR #3 fixture updates.

**Migration impact of the reversal** (vs the round-1 S8 plan):
- DROPPED: MODIFIED block on "Tier 0 input-validation boundary" requirement (S8' is purely additive at the nutation level).
- DROPPED: PR #2/PR #3 test fixture builder updates.
- DROPPED: Inline-construction audits.
- ADDED: 1 new scenario in the ADDED "Tier 1 nutation trait emission API" requirement: "`nutation.compute` rejects invalid cadence_s (zero/negative/NaN/inf/bool/str) with ValueError/TypeError naming the parameter."

**Stationary-track fallback:** if `_geometry.project_to_growth_axis_perpendicular` raises `ValueError` (zero net displacement → growth axis undefined), `_compute_one_track` catches and emits all-NaN traits (with `is_nutating=False` to honor the "no nutation observable" state). Mirrors kinematics.py / qc.py per-track failure handling.

**Failure-flag column:** unlike qc.py's `track_is_clean` / `qc_failure_reason` composition, nutation.py does NOT introduce a `nutation_failure_reason` text column. Rationale: the 6+2 trait columns already convey state via `is_nutating` boolean + NaN-gated values. Adding a reason text would duplicate information. If a future debugging need arises, the log statements provide the audit trail.

**Alternatives considered:**

- *Always populate all traits regardless of `is_nutating`*. Rejected per Q5: violates scientific-honesty principle (raw CWT-ridge values on noise-only signals are not meaningfully periods).
- *NaN-gate only T_nutation_*\* (keep cadence/derr raw)*. **ACCEPTED via S4 round-1 reconciliation** (reviewer Sci-I3 surfaced that cadence_nyquist_ratio is an engineering-diagnostic for *why* is_nutating failed; period_residual_vs_derr_reference is a "where would the peak land if trusted" diagnostic; both useful regardless of biological nutation). Q5's original Tier 1 NaN-gating of all 5 was refined: NaN-gate only T_nutation_median, T_nutation_iqr, A_nutation_envelope_max (the strictly biological-meaning-dependent traits); keep cadence_nyquist_ratio + period_residual_vs_derr_reference + the 3 precursors always populated.
- *QC tier owns `is_nutating`*. Rejected per Q5: cross-tier ownership for a gate that nutation.py needs to honor is architecturally awkward.

### D6. `period_residual_vs_derr_reference` algorithm — fractional period residual

Per Q6 (Option A — recommended), and accommodating the substantive note from Elizabeth that Derr may share raw data in the future:

```python
period_residual_vs_derr_reference = (
    (T_nutation_median - constants.DERR_EXPECTED_PERIOD_S)
    / constants.DERR_EXPECTED_PERIOD_S
)
```

with `DERR_EXPECTED_PERIOD_S = 3333.0` (s) as a new `ConstantsT`-overridable field.

**Semantic interpretation:** fractional deviation of THIS plate's CWT-extracted nutation period from Derr's reference rice value (~3333 s, per the Sept-2025 pilot PDF). On the Layer-2 acceptance fixture (Nipponbare plate-001), the assertion is `|period_residual_vs_derr_reference| < DERR_MATCH_TOLERANCE = 0.02` (CC-7 ± 2%). On other plates / species, the trait reports the analogous deviation, which is biologically interpretable as "is this plant's nutation period close to or far from the published rice reference?"

**Why a constant, not a per-test literal:** named, ConstantsT-overridable, single-source-of-truth alongside other thresholds. Sidecar metadata records the value as part of the run's reproducibility envelope. Future PR (when Derr supplies raw data) can override `DERR_EXPECTED_PERIOD_S` via ConstantsT in tests OR replace the constant with a fixture-based reference if needed.

**Future upgrade path (filed as a NEW follow-up issue at Stage 6):** when Derr provides the raw input signal array and his published scaleogram numerics, a future PR can replace this constant-based residual with a richer spectral-shape distance (e.g., L2 between normalized amplitude spectra). The Tier 1 schema doesn't change — `period_residual_vs_derr_reference` stays a single float column; only the algorithm behind it gets richer. This is intentional API-stability for the trait CSV.

**NaN-gating:** per D5, `period_residual_vs_derr_reference` is NaN when `is_nutating == False` (since the trait requires `T_nutation_median` to be meaningful).

**Alternatives considered:**

- *Spectral-shape distance against a modeled Derr reference*. Rejected per Q6: requires hand-digitizing FWHM from the PDF; brittle to model choice; harder to test deterministically.
- *Rasterize-and-pixel-diff against a committed PNG*. Rejected per Q6: brittle to colormap/font/axis choices; not biologically meaningful; introduces visual-asset version drift.
- *Absolute residual (not fractional)*. Considered but rejected: fractional form is dimensionless and species-comparable; absolute form would carry units of seconds and would require a separate tolerance per species. The fractional form has the cleaner CC-7 acceptance test (`|residual| < 0.02` independent of period magnitude).

### D7. `band_power_ratio` from FFT amplitude spectrum

Per theory.md §7.2: "spectral power in `[0.5T, 2T]` band / total spectral power". Implementation uses the SAME FFT amplitude spectrum that feeds `noise_floor_estimate` — same input signal, same `scipy.fft.rfft`, same units. This ensures the gate `band_power_ratio > BAND_POWER_NOISE_RATIO * noise_floor_estimate` is dimensionally consistent.

```python
def _compute_band_power_ratio(
    x: np.ndarray,
    cadence_s: float,
    t_nutation_median_s: float,
    constants: ConstantsT,
) -> float:
    if len(x) < 2:
        return float("nan")
    spectrum = np.abs(scipy.fft.rfft(x))
    freqs = scipy.fft.rfftfreq(len(x), d=cadence_s)
    f_low = constants.BAND_POWER_BAND_LOW_FACTOR / t_nutation_median_s
    f_high = constants.BAND_POWER_BAND_HIGH_FACTOR / t_nutation_median_s
    in_band_mask = (freqs >= f_low) & (freqs <= f_high)
    total_power = float(np.sum(spectrum ** 2))
    if total_power == 0.0:
        return float("nan")
    band_power = float(np.sum(spectrum[in_band_mask] ** 2))
    return band_power / total_power
```

**Why power = amplitude², not amplitude:** power-ratio is the standard spectral-power-density concept; "spectral power in band" maps to `Σ|C|²` (energy), not `Σ|C|`. Theory.md §7.2 says "power" verbatim; we match.

**Why both `band_power_ratio` and `noise_floor_estimate` use the FFT (not the CWT scaleogram):** CC-8 anchors `noise_floor_estimate` to "Tier-1 Fourier amplitude". Keeping `band_power_ratio` on the same FFT preserves dimensional consistency with `noise_floor_estimate` AND with the `is_nutating` gate. The CWT scaleogram is a different transform with different spectral resolution; mixing the two would muddy the SNR interpretation.

**Note:** this is an algorithmic CHOICE, not a theory-mandated one. Theory.md §7.2 is silent on whether band_power_ratio is FFT-or-CWT-based. The FFT choice is justified by the SNR-consistency argument above. If round-1 critical review pushes back, the alternative is to use CWT scaleogram column power-summing — but that's still under the SAME spectral-power-density semantics.

**Band edges (theory.md §7.2 `[0.5T, 2T]`):**

- Low edge: `f_low = 0.5 / T_nutation_median = BAND_POWER_BAND_LOW_FACTOR / T_nutation_median` with `BAND_POWER_BAND_LOW_FACTOR = 0.5`
- High edge: `f_high = 2.0 / T_nutation_median = BAND_POWER_BAND_HIGH_FACTOR / T_nutation_median` with `BAND_POWER_BAND_HIGH_FACTOR = 2.0`

Note the factor mapping: `f = factor / T`. `factor=0.5` gives `f = 0.5/T = 1/(2T)` (half-cycle-per-T = period 2T). `factor=2.0` gives `f = 2/T` (period T/2). This means the band `[f_low, f_high]` corresponds to PERIODS in `[T/2, 2T]` — matching theory.md's `[0.5T, 2T]` literally.

**`is_nutating` always populated:** even when `T_nutation_median` is NaN-gated downstream, the gate that produced is_nutating runs on the COMPUTED (pre-gate) T value. So `is_nutating` is always a well-defined boolean reflecting the SNR check against the candidate period.

### D8. `cadence_nyquist_ratio` formula — raw ratio + new TEMPORAL_NYQUIST_RATIO_MAX threshold

Per Q7 (theory.md §6.5 framing) + S2'' round-2 (introducing a separate temporal threshold constant per Sci-B2):

```python
cadence_nyquist_ratio = cadence_s / T_nutation_median  # S8' round-2: cadence_s is explicit param
```

**Threshold (S2'' round-2 separated from spatial):**

- `TEMPORAL_NYQUIST_RATIO_MAX = 0.25` is the threshold for the **temporal** trait (this PR #6 emits `cadence_nyquist_ratio` against this constant).
- `NYQUIST_RATIO_MAX = 0.25` (existing PR #1) is the threshold for the **spatial** check (PR #9 spatial CWT will consume this).
- Both default to 0.25; the SEPARATION lives in the constant NAME + docstring, not in different values. Round-2 Sci-B2 found that theory.md §6.5's "10-min still works" example gives temporal ratio 0.18, which is comfortably under 0.25 — so the conservative cushion holds equally well in both domains, but the EMPIRICAL validation of the temporal threshold is a NEW open question filed as a follow-up issue at Stage 6 (mirrors #205-#208 multi-plate threshold validation pattern).
- Plate-001 examples: temporal `300/3333 ≈ 0.09 < 0.25` ✅; spatial `5.83/65 ≈ 0.09 < 0.25` ✅.

**Why "raw ratio" not strict-Nyquist (`2·cadence_s / T`):**

- Theory.md §6.5 frames it as "step / wavelength = 5.83 / 65 ≈ 9.0%, well below the conservative 25% threshold". The 25% threshold is a CONSERVATIVE cushion below the strict Nyquist limit (which would be 0.5 for the ratio form), not the strict criterion itself.
- The conservative cushion interpretation works identically for both NYQUIST_RATIO_MAX and TEMPORAL_NYQUIST_RATIO_MAX; switching to strict-Nyquist form would force a doubling to 0.5 in both, complicating cross-PR consistency.
- "Cadence-Nyquist" in the trait name is colloquial: it's the ratio THAT GETS CHECKED against a Nyquist-derived threshold, not the Nyquist criterion itself.

**Spatial-version cross-reference:** PR #9 (`add-circumnutation-tier3b-spatial-cwt`) will compute `per_frame_step_px / spatial_wavelength_px` and compare against `NYQUIST_RATIO_MAX`. PR #6 introduces `TEMPORAL_NYQUIST_RATIO_MAX` for its temporal sibling. The dimensional separation is preserved at the CONSTANT layer; the formula structure is identical.

**NaN-gating (S4 round-1 + Sci-I3 reconciliation):** per D5, `cadence_nyquist_ratio` is ALWAYS POPULATED (not NaN-gated) because it's an engineering diagnostic — answers "could we have observed nutation if it were present?" even when biological nutation is absent. Aliasing failure modes need this diagnostic to be visible.

**Alternatives considered:**

- *Nyquist-corrected `2 * cadence_s / T_nutation_median`*. Rejected per Q7: forces re-interpretation of the existing `NYQUIST_RATIO_MAX = 0.25` constant; inconsistent with the spatial-CWT version PR #9 will share.

### D9. New `_constants.py` defaults + `ConstantsT` extension; `_CONSTANTS_VERSION` 4 → 5

Per Q8, 5 new constants are added to `_constants.py` and reflected in `ConstantsT` + `_default_constants_snapshot()`. `_CONSTANTS_VERSION` bumps 4 → 5 (matching PR #5's precedent of bumping per-PR for constant additions).

```python
# sleap_roots/circumnutation/_constants.py (5 new module-level constants added)

#: Window size for the median-filter post-filter applied to RidgeResult.periods_s
#: in temporal_cwt.smooth_ridge. Issue #214: PR #5's per-frame argmax ridge can
#: hop between adjacent scales; this median filter suppresses such hops. Default
#: 5 frames (= 25 minutes at plate-001's 300 s cadence, ~0.75% of the 3333 s
#: nutation period — well within "tighten without smearing biological signal").
RIDGE_CONTINUITY_FILTER_WINDOW: int = 5

#: Out-of-band frequency cutoff factor for compute_fourier_noise_floor (CC-8).
#: The noise floor is the median Fourier amplitude over frequencies f >
#: factor / T_nutation_median. Default 5.0 means "5× the candidate-nutation
#: frequency" — captures genuinely out-of-band noise without contamination
#: from the signal-band tail or harmonics.
NOISE_FLOOR_OUT_OF_BAND_FACTOR: float = 5.0

#: Lower band edge factor for band_power_ratio: f_low = factor / T_nutation_median.
#: With factor=0.5, the corresponding PERIOD edge is 2T (i.e., half the candidate
#: nutation frequency). Per theory.md §7.2 "spectral power in [0.5T, 2T] band".
BAND_POWER_BAND_LOW_FACTOR: float = 0.5

#: Upper band edge factor for band_power_ratio: f_high = factor / T_nutation_median.
#: With factor=2.0, the corresponding PERIOD edge is T/2 (i.e., 2× the candidate
#: nutation frequency). Per theory.md §7.2 "spectral power in [0.5T, 2T] band".
BAND_POWER_BAND_HIGH_FACTOR: float = 2.0

#: Reference rice nutation period (s) for period_residual_vs_derr_reference.
#: Sourced from Derr Sept-2025 pilot: 5minutes_average_period=3333s.pdf
#: (spectral peak at f ≈ 0.0003 Hz, T ≈ 3333 s ≈ 55.5 min).
#: period_residual_vs_derr_reference = (T_nutation_median - DERR_EXPECTED_PERIOD_S) / DERR_EXPECTED_PERIOD_S.
#: When Derr provides raw scaleogram arrays in the future, this constant can
#: be overridden via ConstantsT for richer spectral-shape comparison; the
#: trait column period_residual_vs_derr_reference remains a single float by construction.
DERR_EXPECTED_PERIOD_S: float = 3333.0

# === Module-level version (bumped 4 → 5) ===
_CONSTANTS_VERSION: int = 5
```

**`ConstantsT` extension:**

```python
# Add to attrs.define for ConstantsT:
RIDGE_CONTINUITY_FILTER_WINDOW: int = attrs.field(
    default=RIDGE_CONTINUITY_FILTER_WINDOW,
    validator=attrs.validators.instance_of(int),
)
NOISE_FLOOR_OUT_OF_BAND_FACTOR: float = attrs.field(
    default=NOISE_FLOOR_OUT_OF_BAND_FACTOR,
    validator=attrs.validators.instance_of((int, float)),
    converter=float,
)
BAND_POWER_BAND_LOW_FACTOR: float = attrs.field(
    default=BAND_POWER_BAND_LOW_FACTOR,
    validator=attrs.validators.instance_of((int, float)),
    converter=float,
)
BAND_POWER_BAND_HIGH_FACTOR: float = attrs.field(
    default=BAND_POWER_BAND_HIGH_FACTOR,
    validator=attrs.validators.instance_of((int, float)),
    converter=float,
)
DERR_EXPECTED_PERIOD_S: float = attrs.field(
    default=DERR_EXPECTED_PERIOD_S,
    validator=attrs.validators.instance_of((int, float)),
    converter=float,
)
# S2'' round-2 addition (TEMPORAL_NYQUIST_RATIO_MAX, value 0.25 per theory.md §6.5)
TEMPORAL_NYQUIST_RATIO_MAX: float = attrs.field(
    default=TEMPORAL_NYQUIST_RATIO_MAX,
    validator=attrs.validators.instance_of((int, float)),
    converter=float,
)
```

**`_default_constants_snapshot()` extension** (Architecture-B2 round-2 reconciliation — hard-anchored counts):

Add **6** new entries (RIDGE_CONTINUITY_FILTER_WINDOW, NOISE_FLOOR_OUT_OF_BAND_FACTOR, BAND_POWER_BAND_LOW_FACTOR, BAND_POWER_BAND_HIGH_FACTOR, DERR_EXPECTED_PERIOD_S, TEMPORAL_NYQUIST_RATIO_MAX). Verified-against-source: `_default_constants_snapshot()` currently has **29 entries** (18 pre-PR-#4 baseline + 7 PR #4 + 4 PR #5). Post-PR-#6 count: **35**. PR #3 added 4 ConstantsT fields and 4 module-level constants (FRAC_OUTLIER_STEPS_MAX, WORST_STEP_RATIO_MAX, SG_MSD_AGREEMENT_MAX, D2_MSD_AGREEMENT_MAX) per `_constants.py:32-50` history docstring (round-1's "PR #3 may have added zero" hedge was incorrect). §2.G.6 regression-guard asserts `len(_default_constants_snapshot()) == 35` exactly.

**`DERR_MATCH_TOLERANCE = 0.02` is NOT a runtime constant.** Stays as a test-only literal in `tests/test_circumnutation_nutation.py` (e.g., `_DERR_MATCH_TOLERANCE_FOR_TEST = 0.02`). Rationale: the trait `period_residual_vs_derr_reference` is a pure derivation; "close enough to match" is an assertion bound used only in the Layer-2 acceptance test, not a runtime parameter consumed by any module.

**Alternatives considered:**

- *6 runtime constants (include DERR_MATCH_TOLERANCE)*. Rejected per Q8: YAGNI; no downstream consumer; mixes testing and runtime concerns.
- *4 runtime + hard-coded 3333.0 literal in nutation.py*. Rejected per Q8: violates the established pattern of named, sidecar-versioned, ConstantsT-overridable thresholds.

### D10. Test taxonomy: 8 sections mirroring PR #5

Mirror PR #5's 8-section structure. Target: ~85-100 parametrized ids total. Test file at `tests/test_circumnutation_nutation.py`.

**§2.A Schema** (~14 ids):

- §2.A.1 `nutation.compute` returns `pd.DataFrame` (1 id)
- §2.A.2 Return DataFrame has 8 identity columns + 8 trait columns in declared order (1 id; parametrized by trait name → 8 ids per-column; OR single id with explicit column-list assertion)
- §2.A.3 Trait dtypes: `T_nutation_median, T_nutation_iqr, A_nutation_envelope_max, band_power_ratio, noise_floor_estimate, period_residual_vs_derr_reference, cadence_nyquist_ratio` all `float64`; `is_nutating` is `bool` (8 ids parametrized by column → dtype)
- §2.A.4 Row identity 5-tuple uniqueness (1 id)
- §2.A.5 Logger emits one DEBUG at start of `compute` per the token-containment contract (1 id)
- §2.A.6 No WARNING/ERROR emissions on happy path (1 id; caplog-based)

**§2.B Determinism + canary** (~6 ids):

- §2.B.1 Two-call same-process equality at `atol=0` for all 8 trait columns on a fixed synthetic input (1 id × 8 columns = 8 ids OR 1 id with all-columns check)
- §2.B.2 Cross-OS determinism canary (S6 round-1 reconciliation): hardcoded 3-value tuple for `T_nutation_median`, `band_power_ratio`, `noise_floor_estimate` on synthetic input `(T=3333, σ=0.5, n=575, random_state=0)`, tolerance `atol=1e-6` (S6: loosened from PR #5's 1e-9 because PR #6 adds 3 unverified scipy paths — fft/ndimage/stats — on top of PR #5's verified pywt path; 1e-6 cushion is scientifically irrelevant for these traits per CC-6 "either 1e-9 OR documented looser"). Assertion uses `np.testing.assert_allclose(..., atol=1e-6, equal_nan=False)` so RED-phase NaN placeholder fails (1 id per column = 3 ids)
- §2.B.3 RED-phase placeholder: canary values ship as `np.full(3, np.nan)` with `# TODO: replace via vault capture script on GREEN-phase` comment (1 id)
- §2.B.4 Vault capture script: `c:\vaults\sleap-roots\circumnutation\scripts\capture_nutation_canary.py` (mirrors PR #5 precedent; not committed to repo) (informational, not a test id)

**§2.C Synthetic parameter recovery** (~10 ids):

- §2.C.1 Analytical-oracle Layer-1 (TDD-B2 round-1 + Sci-B1 round-2 reconciliation): `np.sin(2π·t/T)` for `T ∈ {2000, 3333, 4500}` (s), `cadence=300 s`, `n_frames=1024`. Assert `T_nutation_median ≈ T ±5%`, `is_nutating == True` (3 periods × 2 assertions = 6 ids). Two reconciliations applied:
  - **TDD-B2 round-1**: Removed T=1000 (only 1.67× the CWT period_min floor of 600 s at PR #5's `CWT_PERIOD_MIN_NYQUIST_FACTOR=2.0`; would fail recovery the way PR #5 R3-B2 did).
  - **Sci-B1 round-2**: Replaced T=6666 with T=4500. S1's SG-detrending step uses window=23 frames × cadence=300 s = 6900 s smoothing window. T=6666 sits at 96% of that window (T=4500 at 65%); SG-detrend partially suppresses the test signal at the cutoff, dropping `is_nutating` to False even though the period is recovered correctly. T=4500 stays safely below the cutoff while keeping 3-test-point low/mid/high coverage.
- §2.C.2 Synthetic-generator Layer-1: `synthetic.generate_trajectory(T_nutation_s=3333, noise_sigma_px=0.5, n_frames=575)` via the multi-track-style pipeline. Assert `T_nutation_median ≈ 3333 ±10%` (looser tolerance for the noisier synth input) (1 id)
- §2.C.3 Noise-only Layer-1 (TDD-B1 round-1 reconciliation): `synthetic.generate_trajectory(amplitude_px=0.0, noise_sigma_px=1.0, n_frames=1024, cadence_s=300.0, random_state=0)` produces `is_nutating == False`, and the **3** biological-meaning-dependent traits (T_nutation_median, T_nutation_iqr, A_nutation_envelope_max) are NaN per the S4-revised gating (2 ids). Rationale: `T_nutation_s=None` (original draft) resolves to default 3333.0 s in synthetic.py:415-416 — produces a strong sinusoid, NOT noise-only. `amplitude_px=0.0` is the documented noise-only path (synthetic.py:329 docstring confirms 0.0 is valid).
- §2.C.4 Cross-coordinate sanity: on Layer-1 synthetic, `coordinate="lateral"` recovers period; `coordinate="x"` (raw) recovers period (synth is unrotated so raw == lateral up to noise) — verify both pass (2 ids)

**§2.D Ridge-continuity post-filter sanity** (#214 acceptance, ~6 ids):

- §2.D.1 `smooth_ridge` smooths `periods_s` only; `amplitudes`, `powers`, `in_coi`, `frame_indices` carry through unchanged. Asserted on a hand-crafted RidgeResult with known scale-hopping (4 ids per field-pass-through + 1 id for smoothed-periods-different = 5 ids)
- §2.D.2 Acceptance criterion (#214): on the Nipponbare proofread fixture per-track, `T_nutation_iqr_post_filter < T_nutation_iqr_raw`. Empirical, parametrized by track_id (6 ids × 1 assertion = 6 ids), OR aggregated as 1 id with per-track loop. Decision: aggregated 1 id with informative failure message logging per-track values, to keep the test count balanced.

**§2.E `band_power_ratio` + `is_nutating` sanity** (~8 ids):

- §2.E.1 Pure-noise input → `band_power_ratio < BAND_POWER_NOISE_RATIO × noise_floor_estimate` (i.e., `is_nutating == False`). Use `np.random.default_rng(0).standard_normal(575)` (1 id)
- §2.E.2 Pure-sinusoid input → `is_nutating == True` (1 id)
- §2.E.3 `noise_floor_estimate` is always finite and ≥ 0 (1 id)
- §2.E.4 `band_power_ratio` is always finite and in `[0, 1]` (1 id)
- §2.E.5 NaN-gating semantics (S4 round-1 + TDD-B1 round-2 reconciliation): when `is_nutating == False`, the **3** strictly biological-meaning-dependent traits are NaN: `T_nutation_median`, `T_nutation_iqr`, `A_nutation_envelope_max`. Parametrized by trait name (3 ids).
- §2.E.6 Always-populated traits — when `is_nutating == False`, the **5** always-populated traits remain finite: `is_nutating` (False), `band_power_ratio`, `noise_floor_estimate`, `cadence_nyquist_ratio` (engineering diagnostic per Sci-I3), `period_residual_vs_derr_reference` (ridge-of-noise diagnostic per S4). 1 id with 5-column finite-value check.

**§2.F Validation/errors** (~24 ids):

- §2.F.1 `nutation.compute` rejects invalid `trajectory_df` (delegates to `_validate_trajectory_df`; covered indirectly via foundation test, but include 3 representative ids: not-DataFrame, missing-column, dtype-mismatch)
- §2.F.2 `nutation.compute` rejects invalid `coordinate` value (parametrized: "" / "X" / 1 / None / "longitudinal" = 5 ids)
- §2.F.3 `nutation.compute` rejects invalid `constants` (not-None, not-ConstantsT = 2 ids)
- §2.F.4 `_geometry.project_to_growth_axis_perpendicular` rejects length-mismatch (1 id), non-finite (2 ids: NaN-in-x, NaN-in-y), zero-net-displacement (1 id) = 4 ids
- §2.F.5 `_noise.compute_fourier_noise_floor` returns NaN for `len(x) < 2` (1 id), returns NaN for empty out-of-band region (1 id) = 2 ids
- §2.F.6 `temporal_cwt.smooth_ridge` rejects non-`RidgeResult` (1 id), rejects non-odd window (1 id), rejects window < 1 (1 id), accepts default-from-constants (1 id) = 4 ids
- §2.F.7 Stationary-track fallback (TDD-B5 round-1 + TDD-B2 round-2 reconciliation): closed-loop trajectory with `x[-1]==x[0] AND y[-1]==y[0]` but intermediate frames vary (e.g., a circle returning to its start). Passes `_validate_trajectory_df` upstream (all values finite, columns present, S8' cadence_s threaded as kwarg — no attrs check after S8' reversal), then triggers the zero-net-displacement fallback in `_geometry.project_to_growth_axis_perpendicular`. Per Architecture-I3 round-1, the helper returns `np.full(n, np.nan)` rather than raising; downstream `nutation.compute` emits all-NaN trait row with `is_nutating=False`. Assert: no exception bubbled; `is_nutating==False`; 3 NaN-gated traits (T_med, T_iqr, A_max) are NaN; 5 always-populated traits are populated (per S4) — note that on all-NaN input, `period_residual_vs_derr_reference` and `cadence_nyquist_ratio` are also NaN (downstream of the NaN cascade), so the practical assertion is "5 always-populated TRAIT COLUMNS exist in the row, but cadence_nyquist_ratio + period_residual_vs_derr_reference may carry NaN due to upstream propagation". Also asserts no `np.RuntimeWarning("All-NaN slice encountered")` (Architecture-I3 round-2 NaN-cascade concern) — `_compute_one_track` short-circuits the rest of the pipeline when `_geometry.project_to_growth_axis_perpendicular` returns all-NaN. 1 id.

**§2.G `ConstantsT` override + resolution + `coordinate=` parameter** (~12 ids):

- §2.G.1 `ConstantsT()` default produces same output as `constants=None` (1 id)
- §2.G.2 ConstantsT with overridden `RIDGE_CONTINUITY_FILTER_WINDOW` affects `T_nutation_iqr` (verify by parametrize: window=1 vs window=11) (2 ids)
- §2.G.3 ConstantsT with overridden `NOISE_FLOOR_OUT_OF_BAND_FACTOR` shifts the noise band (1 id)
- §2.G.4 ConstantsT with overridden `BAND_POWER_BAND_LOW_FACTOR / HIGH_FACTOR` shifts the band power calculation (2 ids)
- §2.G.5 ConstantsT with overridden `DERR_EXPECTED_PERIOD_S` shifts `period_residual_vs_derr_reference` (1 id)
- §2.G.6 `_default_constants_snapshot()` superset check: new keys present after PR #6 (1 id with set-superset assertion against `{*pr5_keys, *pr6_keys}`)
- §2.G.7 `_CONSTANTS_VERSION == 5` (1 id)
- §2.G.8 `coordinate="lateral"` is default (1 id)
- §2.G.9 `coordinate="x"` produces different output than `coordinate="lateral"` on a non-aligned track (1 id)
- §2.G.10 Foundation-test reach-around verifies `nutation` in `IMPLEMENTATIONS_WITH_CONSTANTS_KWARG` (covered by foundation migration; 1 id reach-around)

**§2.H Reference-fixture sanity** (~14 ids):

- §2.H.1 Nipponbare proofread fixture per-track: `nutation.compute` does not raise and emits 8 trait columns for each of 6 tracks. Parametrized by track_id (6 ids)
- §2.H.2 Nipponbare proofread fixture: plausibility-band median ridge period from `coordinate="lateral"` AFTER SG-detrending (per S1) is in `[1000, 10000]` s. Parametrized by track_id (6 ids) — empirical-correction note: PR #5's GREEN-phase observed centered-but-not-detrended raw `tip_x` could not pass this band; the S1 round-1 reconciliation explicitly adds SG-detrending precisely to make this band achievable on plate-001. If any track fails the plausibility band even after SG-detrending, the failure is INFORMATIVE (signals additional preprocessing is needed) and is recorded in the GREEN-phase Reconciliation Appendix.
- §2.H.3 **Layer-2 Derr forensic match (S5 round-1 + Sci-B3 round-2 reconciliation: two-part assertion)**: on the Nipponbare proofread fixture's `coordinate="lateral"` signal AFTER SG-detrending (per S1), assert TWO conditions:
  1. **CC-7 median enforcement**: `abs(np.median(per_track_residuals)) < 0.02` — the median across all 6 tracks must be within CC-7's stated ±2% target. Median is robust to per-plant variance.
  2. **Per-track count check**: `≥ 4 of 6 tracks satisfy |residual| < 0.05 AND is_nutating == True`. Acknowledges biological variance; allows 2 outlier tracks.

  Rationale (per round-1 TDD-B3 + round-2 Sci-B3 + PR #5 R3-B2 precedent): scale-grid discreteness on n_frames=575 puts a ~5% floor on per-track accuracy; per-plant biological variance adds ~2-5% spread; the median across 6 tracks averages out per-plant noise while still anchoring to CC-7 ±2%. 1 aggregated id with per-track informative-failure messages logging both `is_nutating` and `period_residual_vs_derr_reference` for all 6 tracks (failure message shows median residual + per-track residuals + per-track is_nutating booleans).

  Semantic note (per round-2 Sci-I1): when `is_nutating == False` on a track, `period_residual_vs_derr_reference` is still POPULATED per S4 (it's a "ridge-of-noise" diagnostic, not a forensic match). The per-track AND-conjunction in (2) correctly excludes such tracks from the pass-count. Median in (1) uses ALL 6 residuals regardless of `is_nutating` because the median is robust enough to handle 1-2 "ridge-of-noise" outliers.
- §2.H.4 Issue #214 acceptance criterion check (cross-references §2.D.2): on the proofread fixture, `T_nutation_iqr_post_filter < T_nutation_iqr_raw` for ≥ 5 of 6 tracks (1 id with aggregated count; allows one track to fail without blocking the PR, accommodating tracks where the raw IQR is already minimal due to short COI-interior).

**Foundation-test migration (in `tests/test_circumnutation_foundation.py`):**

- Drop `nutation` from any stub-tracking constants if it appears (it's currently NOT a stub — newly created in PR #6, so this is moot)
- Add `nutation` to `IMPLEMENTATIONS_WITH_CONSTANTS_KWARG` (4 → 5 entries: kinematics, qc, synthetic, temporal_cwt, **nutation**)
- Bump `_CONSTANTS_VERSION` assertion: rename `test_schema_version_is_1_and_constants_version_is_4` → `test_schema_version_is_1_and_constants_version_is_5` (canonical name verified at `tests/test_circumnutation_foundation.py:206`); update the test's docstring (currently says `"_CONSTANTS_VERSION is 4 (bumped in PR #5)"` → `"... is 5 (bumped in PR #6)"`) and the inner assertion `_constants._CONSTANTS_VERSION == 4` → `... == 5`. Per Architecture round-1 B1: PR #5's archived design used the canonical test name; the round-1 PR #6 draft fabricated `test_2G2_*`.
- Add `nutation` to `test_module_logger_is_namespaced` parametrize list (the Copilot-precedent fix; PR #4/PR #5 both added their new module here explicitly)
- Update comment-block-at-lines-26-34 (or wherever the PR-by-PR module count is recorded) to reflect PR #6's addition

**`_constants.py` regression-guard** (`§2.G.6`) (TDD-B3 round-2 reconciliation: TEMPORAL_NYQUIST_RATIO_MAX added; the round-1 5-constant set was stale post-S2): set-superset assertion against `EXPECTED_PR6_CONSTANTS = {*EXPECTED_PR5_CONSTANTS, "RIDGE_CONTINUITY_FILTER_WINDOW", "NOISE_FLOOR_OUT_OF_BAND_FACTOR", "BAND_POWER_BAND_LOW_FACTOR", "BAND_POWER_BAND_HIGH_FACTOR", "DERR_EXPECTED_PERIOD_S", "TEMPORAL_NYQUIST_RATIO_MAX"}` (6 entries). Follows PR #5's per-PR set-splitting precedent. Also assert `len(_default_constants_snapshot()) == 35` post-PR-#6 (Architecture-B2 round-2 reconciliation: 29 baseline + 6 PR #6 = 35).

## Test-file imports

```python
# tests/test_circumnutation_nutation.py
import logging
import math

import attrs
import numpy as np
import pandas as pd
import pytest
import scipy.fft
import scipy.ndimage
import scipy.signal
import scipy.stats

import sleap_roots
from sleap_roots.circumnutation import (
    _constants,
    _geometry,
    _noise,
    _types,
    nutation,
    synthetic,
    temporal_cwt,
)
from sleap_roots.circumnutation._constants import (
    BAND_POWER_BAND_HIGH_FACTOR,
    BAND_POWER_BAND_LOW_FACTOR,
    BAND_POWER_NOISE_RATIO,
    DERR_EXPECTED_PERIOD_S,
    NOISE_FLOOR_OUT_OF_BAND_FACTOR,
    NYQUIST_RATIO_MAX,                  # spatial check (PR #1)
    RIDGE_CONTINUITY_FILTER_WINDOW,
    SG_DEGREE,                          # S1 round-1 (PR #2/#3 carryover)
    SG_WINDOW_DETREND,                  # S1 round-1 (PR #2/#3 carryover)
    TEMPORAL_NYQUIST_RATIO_MAX,         # S2'' round-2: temporal check, value 0.25
    ConstantsT,
)

# Project-local test fixtures (existing PR #2 fixture; no new fixtures).
_NIPPONBARE_PROOFREAD_SLP = (
    "tests/data/circumnutation_nipponbare_plate_001/"
    "plate_001_greyscale.tracked_proofread.slp"
)
_KITAAKEX_BASELINE_SLP = (
    "tests/data/circumnutation_plate/plate_001_greyscale.tracked.slp"
)

# Test-only literal — see D9 rationale.
_DERR_MATCH_TOLERANCE_FOR_TEST: float = 0.02
```

## Risks & Trade-offs

### R1. Median-filter window choice on T_nutation_iqr empirical sensitivity

Default `RIDGE_CONTINUITY_FILTER_WINDOW = 5`. On plate-001 (575 frames at 300 s cadence), this is 25 minutes of smoothing. Risk: if biological period drift over a single track happens on a < 25-minute time scale, the median filter smears it out, lowering both `T_nutation_iqr` (genuinely) AND obscuring real biological variation.

**Mitigation:**

- ConstantsT-overridable: callers needing finer time resolution can override to `window=3` or `window=1` (= no smoothing).
- §2.G.2 tests window=1 vs window=11 to quantify the trait sensitivity.
- §2.H.4 empirically verifies the acceptance criterion (`T_nutation_iqr_post_filter < T_nutation_iqr_raw`) on plate-001, ensuring the default at least preserves the gate.

**Cross-PR awareness:** future multi-plate analyses (#202 K=10 sweep, #205-208 threshold validation) may reveal the default is wrong for their plate population. Re-tuning then is straightforward (single constant change in `_constants.py`, version bump 5 → 6 follows the established pattern).

### R2. NOISE_FLOOR_OUT_OF_BAND_FACTOR sensitivity on `is_nutating` gate

Default `NOISE_FLOOR_OUT_OF_BAND_FACTOR = 5.0`. Means "noise band starts at 5× the candidate nutation frequency". Risk: if the signal has harmonics at 2× or 3× the fundamental (theory: yes, BM2016 §5 predicts second-harmonic content), those harmonics LIVE in the "noise" band by this definition and inflate `noise_floor_estimate`, lowering `band_power_ratio / noise_floor_estimate`, potentially falsely failing `is_nutating == True`.

**Mitigation:**

- ConstantsT-overridable: callers seeing this failure mode can override to factor=3 or factor=2 (a tighter "out-of-band" definition).
- §2.G.3 tests factor sensitivity.
- §2.H.3 Layer-2 forensic match acceptance verifies the default-factor produces `is_nutating == True` on plate-001 (where harmonics are present biologically).

**Open Question (round-1 critical review):** is factor=5 right for the Derr 1D oscillation, where strong second-harmonic content may live at ~6.7e-4 Hz (= 2/3333)? Surface this with the scientific-rigor reviewer.

### R3. DERR_EXPECTED_PERIOD_S = 3333.0 as a species-specific constant

The reference period is rice-specific (per Derr Sept-2025 Nipponbare pilot). Risk: on other species (Arabidopsis, sunflower, etc.), the expected nutation period is different (Rivière 2022 §1.2: nutation period 1.5-4 hours in Averrhoa = 5400-14400 s, a 1.6-4.3× range). Using `DERR_EXPECTED_PERIOD_S = 3333` on those species would produce a large `period_residual_vs_derr_reference` that doesn't actually reflect a failure mode.

**Mitigation:**

- ConstantsT-overridable: pipelines for other species can override `DERR_EXPECTED_PERIOD_S` to the species-appropriate value.
- D6's "biological interpretability" note explicitly clarifies the trait is "fractional deviation from rice reference" — not "match score against this plant's expected period".
- Layer-2 acceptance test is RUN against plate-001 (rice), where the constant applies.
- A future PR may upgrade the trait to be species-aware (taking `species` as a parameter and looking up expected periods from a table). Out of PR #6 scope; not currently filed as an issue.

### R4. Lateral-projection failure mode on stationary tracks

`_geometry.project_to_growth_axis_perpendicular` raises `ValueError` on zero net displacement (D2). Risk: if a real-world track has effectively zero net displacement (e.g., tracking artifact, very short recording, biological non-growth), the whole track's traits become NaN with `is_nutating=False`.

**Mitigation:**

- The PR #3 QC tier's `growth_axis_unreliable` flag already gates these tracks upstream (per PR #2's growth-axis reliability check). Tracks failing the reliability gate should be filtered before reaching `nutation.compute`.
- nutation.py catches the `ValueError` and emits an all-NaN trait row with `is_nutating=False` (gracefully degrades).
- §2.F.7 explicitly tests this fallback.

### R5. Cross-OS determinism of scipy.fft + scipy.ndimage.median_filter

PR #5's cross-OS canary at `atol=1e-9` already passes for `pywt.cwt`. New compute paths in PR #6:

- `scipy.fft.rfft` (used in noise_floor + band_power)
- `scipy.ndimage.median_filter` (used in smooth_ridge)
- `scipy.stats.iqr` (used in T_nutation_iqr)

Risk: any of these may have cross-OS numerical differences > 1e-9. PR #4's synthetic generator established the baseline (numpy `default_rng(0)` reproducibility is exact). PR #5 verified `pywt.cwt` to 1e-9. PR #6 inherits the burden of verifying these scipy paths.

**Mitigation:**

- §2.B canary captures 3 trait values from the GREEN-phase Windows run; CI matrix on Ubuntu / Windows / macOS runs the same canary at `atol=1e-9`.
- If any value fails the cross-OS canary, R5 mitigation falls back to a per-OS-marked canary (matching PR #5's marker pattern, if used) or the tolerance is loosened to `atol=1e-6` with a noted reason in the canary docstring.

### R6. Test runtime + memory budget

The §2.H test suite runs `nutation.compute` 6× on the Nipponbare proofread fixture (6 tracks). Per-track cost: `compute_scaleogram(575 frames, n_scales=64)` ≈ 0.5 s × 64 scales = ~30 s for pywt.cwt. Plus extract_ridge, smooth_ridge, FFT, etc. Estimated per-track total: ~35 s. 6 tracks = ~210 s for §2.H alone. §2.B canary adds ~30 s; §2.C synthetic adds ~6 × ~10 s = 60 s. Total nutation test file: ~5-6 minutes.

**Mitigation:**

- Use `pytest-xdist` for parallelization in CI (already configured).
- Cache the proofread fixture's `nutation.compute` result via `pytest` fixture with `scope="module"` so the §2.H sub-tests share a single computation.
- If runtime is excessive in CI, mark §2.H.3 (Layer-2 forensic match) as `slow` and gate behind `pytest -m slow`.

## Migration Plan

**`sleap_roots/circumnutation/nutation.py` — NEW module (PR #6's headline addition):**

- New file. Module docstring: "Tier 1 nutation trait emission per theory.md §7.2 + §7.6 + §6.5. Composes temporal_cwt CWT primitives + _geometry lateral projection + _noise Fourier noise floor."
- Module-level imports: `import logging`, `import numpy as np`, `import pandas as pd`, `import scipy.fft`, `import scipy.stats`, plus relative imports from `_constants`, `_geometry`, `_noise`, `_types`, `temporal_cwt`.
- Module-level constants: `_NUTATION_TRAIT_COLUMNS` tuple, `_COORDINATE_CHOICES` frozenset, `_IDENTITY_5_TUPLE` (mirroring kinematics/qc).
- Public function: `compute(trajectory_df, coordinate="lateral", constants=None) -> pd.DataFrame`.
- Private helpers: `_compute_one_track(group, coordinate, constants)`, `_select_signal(group, coordinate)`, `_compute_band_power_ratio(signal, cadence_s, t_nut, constants)`, `_check_constants(constants)`, `_check_coordinate(coordinate)`.
- Per-module logger `logger = logging.getLogger(__name__)`.
- **Logger emissions:** at start of `compute` (after input validation): one `logger.debug` whose message MUST start with `"nutation.compute("` AND contain each of `n_tracks=`, `coordinate=`, `cadence_s=`. Suggested f-string: `f"nutation.compute(n_tracks={n_tracks}, coordinate={coordinate!r}, cadence_s={cadence_s:.6f})"`. No INFO/WARNING/ERROR on happy path.

**`sleap_roots/circumnutation/temporal_cwt.py` — MODIFIED (smooth_ridge addition):**

- Add `smooth_ridge(ridge_result, window=None, constants=None) -> RidgeResult` per D4.
- Add private `_validate_window(window)` helper.
- Add `import scipy.ndimage`.
- **Logger emission for smooth_ridge:** one `logger.debug` at start: `f"smooth_ridge(n_frames={n}, window={window})"`. Token-containment contract: starts with `"smooth_ridge("`, contains `n_frames=`, `window=`.

**`sleap_roots/circumnutation/_geometry.py` — MODIFIED (project_to_growth_axis_perpendicular addition):**

- Add `project_to_growth_axis_perpendicular(x, y) -> np.ndarray` per D2.
- Add private `_validate_xy(x, y)` helper if not already present.
- No new imports beyond what's already there (`math`, `numpy`).
- No logger emission on happy path; debug emission on stationary-track ValueError raise.

**`sleap_roots/circumnutation/_noise.py` — MODIFIED (compute_fourier_noise_floor + compute_sg_detrended additions):**

- Add `compute_fourier_noise_floor(x, cadence_s, t_nutation_median_s, factor) -> float` per D3.
- **(S1 round-1 addition)** Add `compute_sg_detrended(x, window, polynomial_order) -> np.ndarray` returning the 1D residual `x - scipy.signal.savgol_filter(x, window_length=window, polyorder=polynomial_order, mode='nearest')`. Reuses existing `SG_WINDOW_DETREND = 23` + `SG_DEGREE = 3` constants (PR #2/#3 carryover — no new constants for this helper). The nutation.compute pipeline calls it after `project_to_growth_axis_perpendicular`, before CWT/FFT, per the preliminary_results §3.4 prescription. Adds ~30 LOC; covered by §2.F validation tests (window even/odd, polynomial_order < window, len(x) < window).
- Add `import scipy.fft`, `import scipy.signal`.
- Debug emission on too-short input or empty out-of-band.

**`sleap_roots/circumnutation/_constants.py` — MODIFIED (5 new constants + version bump):**

- Add **6** new module-level constants per D9 + S2 round-1 reconciliation: `RIDGE_CONTINUITY_FILTER_WINDOW=5`, `NOISE_FLOOR_OUT_OF_BAND_FACTOR=5.0` (S7: kept at CC-8 verbatim, GREEN-phase may revisit), `BAND_POWER_BAND_LOW_FACTOR=0.5`, `BAND_POWER_BAND_HIGH_FACTOR=2.0`, `DERR_EXPECTED_PERIOD_S=3333.0`, **`TEMPORAL_NYQUIST_RATIO_MAX=0.25`** (S2 round-1 addition; the existing `NYQUIST_RATIO_MAX=0.25` is for SPATIAL CWT per theory.md §6.5 — adding a separate temporal constant per theory.md §1.2's "5-10 samples per nutation period" rule that gives ratio ≤ 0.1-0.2; clarifying docstring on `NYQUIST_RATIO_MAX` notes it's the spatial check).
- Extend `ConstantsT` `attrs.define` with 5 new fields.
- Extend `_default_constants_snapshot()` to include 5 new keys.
- Bump `_CONSTANTS_VERSION` 4 → 5.
- Update `_CONSTANTS_VERSION` docstring to note PR #6's contribution.
- Add docstring cross-references: `NYQUIST_RATIO_MAX` ↔ `cadence_nyquist_ratio` (per D8); `BAND_POWER_NOISE_RATIO` ↔ `is_nutating` gate (per D5); `WAVELET_DEFAULT_TEMPORAL` ↔ `RIDGE_CONTINUITY_FILTER_WINDOW` (per D4 default-window-tied-to-wavelet).

**Spec delta (`openspec/changes/add-circumnutation-tier1-derr-faithful/specs/circumnutation/spec.md`):**

Five blocks:

1. **`## MODIFIED Requirements > ### Requirement: Package layout`** (Architecture-B4 round-1 reconciliation) — implementations grow 4 → 5 by ADDITION of `nutation` (NOT by transition from a prior stub; `nutation` is newly created in PR #6 and never existed as a stub). Stub-count claim stays "6 stubs" — `nutation` does not appear in the stub-callable table. Bump implementation-callable enumeration to include `nutation.compute` alongside `kinematics.compute`, `qc.compute`, `synthetic.generate_trajectory`, `temporal_cwt.compute_scaleogram`. Add new scenario: "`nutation.compute(trajectory_df, coordinate='lateral', constants=None)` is callable on a valid trajectory_df without raising (parenthetical: since `nutation` is a NEW module introduced by PR #6, not a transition from a prior stub, it does not appear in the stub-callable table)". Carry the FULL existing requirement text per OpenSpec convention.
2. **`## MODIFIED Requirements > ### Requirement: Module-level constants`** (S2 round-1 reconciliation: 6 new constants, not 5) — extend the required-constants enumeration with the **6** new keys: `RIDGE_CONTINUITY_FILTER_WINDOW`, `NOISE_FLOOR_OUT_OF_BAND_FACTOR`, `BAND_POWER_BAND_LOW_FACTOR`, `BAND_POWER_BAND_HIGH_FACTOR`, `DERR_EXPECTED_PERIOD_S`, **`TEMPORAL_NYQUIST_RATIO_MAX`** (S2 added). Bump the `_CONSTANTS_VERSION` assertion in the existing scenario 4 → 5; add a new scenario "New nutation/Tier 1 constants are overridable via ConstantsT" mirroring the PR #3/#4/#5 scenarios.
3. **(S8' round-2 REVERSAL — this MODIFIED block is DROPPED entirely.)** The round-1 S8 plan to extend "Tier 0 input-validation boundary" with a `trajectory_df.attrs["cadence_s"]` requirement was reversed in round-2 reconciliation (see D5 "cadence_s field" paragraph + Round-2 Reconciliation Appendix). The new ADDED requirement (#4) handles cadence_s validation at the `nutation.compute` boundary via the explicit positional parameter — no spec MODIFIED on Tier 0 needed. The round-1 draft's "Trait CSV row-identity schema" MODIFIED block was ALSO DROPPED earlier per Architecture-B2 round-1 reconciliation (no per-tier trait-column enumeration to extend). Net: spec delta has 1 MODIFIED (Package layout) + 1 MODIFIED (Module-level constants) + 2 ADDED (Tier 1 nutation trait emission API, Temporal CWT ridge-continuity smoothing API). Cleaner footprint than round-1.
4. **`## ADDED Requirements > ### Requirement: Tier 1 nutation trait emission API`** — contract-lock `nutation.compute`:
   - Scenario: `nutation.compute(trajectory_df, coordinate=, constants=None)` returns a `pd.DataFrame` with the 8 documented trait columns + 8 identity columns in declared order.
   - Scenario: `nutation.compute` rejects invalid `coordinate` value with `ValueError`.
   - Scenario: NaN-gating semantics — when `is_nutating == False`, the 5 meaning-dependent traits emit `NaN`; `is_nutating`, `band_power_ratio`, `noise_floor_estimate` always populated.
   - Scenario: `nutation.compute` is deterministic (same input → same output; cross-OS `atol=1e-9`).
   - Scenario: Layer-2 Derr forensic-match acceptance — on the Nipponbare proofread fixture's `coordinate="lateral"` signal, all 6 tracks satisfy `is_nutating == True` AND `|period_residual_vs_derr_reference| < 0.02`. (Reference value 3333.0 s is via `DERR_EXPECTED_PERIOD_S` constant; tolerance 0.02 is a test-only literal.)
   - Scenario: Issue #214 acceptance — `T_nutation_iqr_post_filter < T_nutation_iqr_raw` for ≥ 5 of 6 tracks on the Nipponbare proofread fixture.
5. **`## ADDED Requirements > ### Requirement: Temporal CWT ridge-continuity smoothing API`** — contract-lock `temporal_cwt.smooth_ridge`:
   - Scenario: `smooth_ridge(ridge_result, window=None, constants=None)` returns a `RidgeResult` with smoothed `periods_s`; other fields carried through unchanged.
   - Scenario: `smooth_ridge` rejects non-`RidgeResult` input with `TypeError`.
   - Scenario: `smooth_ridge` rejects non-odd or non-positive `window` with `ValueError`.

**Note on the `Pure-pixel pipeline output convention` and `Units sidecar JSON` requirements:** these are NOT modified. PR #6's traits are dimensionless or pixel-unit; `units.py` mapping carries forward. The `is_nutating` boolean and dimensionless ratios don't trigger any units-conversion considerations.

**`tests/test_circumnutation_foundation.py` changes:**

- Add `"nutation"` to `IMPLEMENTATIONS_WITH_CONSTANTS_KWARG` (4 → 5 entries: `("kinematics", "compute"), ("qc", "compute"), ("synthetic", "generate_trajectory"), ("temporal_cwt", "compute_scaleogram"), ("nutation", "compute")`).
- Add `"nutation"` to `test_module_logger_is_namespaced` parametrize list (per the Copilot precedent from PR #4/PR #5).
- Rename `test_schema_version_is_1_and_constants_version_is_4` → `test_schema_version_is_1_and_constants_version_is_5` (canonical name, NOT `test_2G2_*`).
- Update the comment block at module top (or PR-tracking comment block) to reflect PR #6's addition.
- No removal needed for stubs (`nutation` is not currently a stub).

**New test file `tests/test_circumnutation_nutation.py`:**

- 8 sections (§2.A-§2.H) per D10. ~85-100 parametrized ids total.
- Imports per "Test-file imports" section.
- Uses `synthetic.generate_trajectory` (PR #4) for inputs in §2.B, §2.C, §2.E.
- Uses raw `np.sin` for §2.C.1 (independent analytical oracle).
- Uses the existing Nipponbare proofread fixture for §2.H; no new fixtures.

**New canary capture script `c:\vaults\sleap-roots\circumnutation\scripts\capture_nutation_canary.py`:**

- Stand-alone Python script (vault-only; not committed to repo).
- Invocation: `uv run python c:\vaults\sleap-roots\circumnutation\scripts\capture_nutation_canary.py`
- Generates 3 canary values for the §2.B.2 hardcode: `T_nutation_median`, `band_power_ratio`, `noise_floor_estimate` on `synthetic.generate_trajectory(T_nutation_s=3333, noise_sigma_px=0.5, n_frames=575, random_state=0)`. Includes date-stamp + machine-fingerprint header + ConstantsT snapshot for provenance, mirroring PR #5's capture-script pattern.

**`docs/circumnutation/roadmap.md` updates** (after merge, in cleanup-merged step):

- Row PR #6: status checkbox `⬜` → `✅`. Add issue / PR cross-links.

**`docs/changelog.md` updates** (under `[Unreleased] / ### Added`):

- "circumnutation: Tier 1 nutation trait emission (`sleap_roots.circumnutation.nutation.compute`) producing 8 traits per theory.md §7.2 + §7.6 + §6.5 (T_nutation_median, T_nutation_iqr, A_nutation_envelope_max, band_power_ratio, noise_floor_estimate, is_nutating, period_residual_vs_derr_reference, cadence_nyquist_ratio); composes PR #5 CWT primitives + new `_geometry.project_to_growth_axis_perpendicular` lateral-projection helper + new `_noise.compute_fourier_noise_floor` (CC-8) + new `temporal_cwt.smooth_ridge` ridge-continuity post-filter (closes #214 — Mallat 1999 §4.4.2 inspired); `is_nutating` boolean gates NaN-emission of 5 meaning-dependent traits per scientific-honesty principle; Layer-2 Derr forensic match verified at ±2% on Nipponbare plate-001 proofread fixture. 5 new defaults in `_constants.py` + `ConstantsT`; `_CONSTANTS_VERSION` 4 → 5."

**`sleap_roots/circumnutation/__init__.py`:**

- **No new re-exports.** Match PR #4/PR #5 precedent: `nutation.compute` accessed via `from sleap_roots.circumnutation import nutation`. Top-level `sleap_roots` re-exports stay limited to `CircumnutationInputs` + `convert_to_mm` + `ROW_IDENTITY_COLUMNS`.

**No changes to `sleap_roots/circumnutation/qc.py`:** `cadence_nyquist_ratio` was DEFERRED from PR #3 (never emitted); PR #6 ADDS it to nutation.csv, not qc.csv. The PR #3 spec doesn't enumerate `cadence_nyquist_ratio` in qc.csv columns, so no spec change to QC tier.

## Follow-up Issues

PR #6 files **two new GitHub issues** in addition to the existing open follow-ups.

### NEW #1 (filed as part of PR #6): Derr Layer-2 forensic-match algorithm upgrade

**Title:** "circumnutation: upgrade period_residual_vs_derr_reference to spectral-shape distance when Derr provides raw scaleogram numerics"

**Body:** PR #6's `period_residual_vs_derr_reference` is computed as `(T_nutation_median - DERR_EXPECTED_PERIOD_S) / DERR_EXPECTED_PERIOD_S` (fractional period residual against a single constant). This is the cleanest available algorithm given that Derr Sept-2025 outputs are PDF/PNG only (no raw scaleogram arrays). Elizabeth confirmed during PR #6 brainstorming that Derr can be asked for more data when needed.

When Derr provides:

1. The raw 1D input signal (currently PDF-visualized as `5minutes_sample_data.pdf`).
2. His published scaleogram numerics (currently visualized as `_5minutes_wavelets.png`).
3. Optionally: the FFT amplitude spectrum he derived (visualized as `5minutes_average_period=3333s.pdf`).

The trait can be upgraded from a fractional-period-residual to a richer spectral-shape distance (e.g., L2 between normalized amplitude spectra, restricted to the in-band region). The Tier 1 CSV schema does NOT change: `period_residual_vs_derr_reference` remains a single float column; only the algorithm behind it gets richer.

**Acceptance criteria:**

- Derr's raw input array re-runs through our pipeline and produces `period_residual_vs_derr_reference ≈ 0` (whatever the new algorithm is).
- On the Nipponbare proofread fixture (which is NOT Derr's input), `period_residual_vs_derr_reference` is interpretable and reflects spectral-shape distance from Derr's pilot.
- ConstantsT gains a new `DERR_REFERENCE_FFT_FILE` or similar (whatever the upgrade requires) — keeping the "single source of truth, sidecar-versioned" pattern.

### NEW #2 (filed as part of PR #6): foundation-test EXPECTED_CONSTANTS consolidation

**Title:** "circumnutation: consolidate foundation-test EXPECTED_CONSTANTS per-PR-set splitting into a single canonical set"

**Body:** PR #5 deferred this (R3-N2/N3 acknowledged but not actioned); PR #6 perpetuates the per-PR set-splitting pattern (`EXPECTED_PR1_CONSTANTS | EXPECTED_PR3_CONSTANTS | EXPECTED_PR4_CONSTANTS | EXPECTED_PR5_CONSTANTS | EXPECTED_PR6_CONSTANTS`). This will balloon over the next 12+ PRs.

A future PR should consolidate into a single `EXPECTED_CONSTANTS` set or generate it from `_default_constants_snapshot().keys()` directly. Out of PR #6 scope (would require a foundation-test refactor that touches all merged PRs' test code), but worth tracking.

### Existing open follow-ups (NOT blocked by PR #6)

- **#199 (Python 3.11 + uv modernization)** — independent of PR #6.
- **#202 (K=10 sensitivity sweep on growth-axis reliability gate)** — independent of PR #6 (requires multi-plate data).
- **#205-#208 (α/β/γ/δ QC threshold validation)** — independent of PR #6 (require multi-plate data).
- **#208 specifically (`±inf` detection in QC tier)** — orthogonal to PR #6; the lateral-projection helper rejects non-finite inputs at the `_geometry` boundary (D2), so any `±inf` in `tip_x` / `tip_y` would fail at `nutation.compute`'s input validation before reaching the trait computation. PR #6 does not introduce a similar QC issue.
- **#214 (ridge-tracking continuity)** — **CLOSED BY PR #6** per D4 + §2.D + §2.H.4 empirical acceptance. PR description links to #214.

### Future possible issues (NOT filed yet)

- **Mallat 1999 §4.4.2 ridge-following algorithm.** Filed only IF the median-filter post-filter (D4) empirically fails on multi-plate data.
- **Parabolic ridge refinement** for sub-scale period precision. Filed only IF `T_nutation_iqr` accuracy spec demands sub-grid resolution beyond what the median filter provides.
- **Species-aware `DERR_EXPECTED_PERIOD_S` table.** Filed only IF the per-species deviation tracked via R3 mitigation becomes a practical pain point.

## Open Questions

The following questions are surfaced for the round-1 critical-review subagents to confirm or refute:

- **OQ1 (Scientific rigor):** is the FFT-based `band_power_ratio` the right choice, or should it use the CWT scaleogram column-power-sum? Trade-off: FFT preserves dimensional consistency with `noise_floor_estimate`; CWT uses the time-localized power but introduces a different transform basis.
- **OQ2 (Scientific rigor):** does the `NOISE_FLOOR_OUT_OF_BAND_FACTOR = 5` exclude harmonic content of the fundamental? BM2016 §5 predicts second-harmonic. At factor=5, harmonics at 2/T or 3/T are INSIDE the noise band, which is biologically wrong (they're signal, not noise).
- **OQ3 (Architecture):** should `_geometry.project_to_growth_axis_perpendicular` accept an additional `growth_axis_unit_vector` kwarg for callers who've pre-computed the axis (potential PR #11 reuse)? Or is per-call recomputation acceptable?
- **OQ4 (TDD-testability):** can §2.H.3's Layer-2 forensic-match be verified deterministically against the existing fixture, OR does it require a vault-only fixture (since the lateral projection on plate-001 may not literally exhibit the Derr-paper 3333 s peak at the 2% tolerance)?
- **OQ5 (Cross-PR consistency):** are the 5 new constants placed in `_constants.py` in the right SECTION (after PR #5's CWT-machinery block, before any planned PR #7 ψ_g constants)? Aesthetic question, but matters for `_CONSTANTS_VERSION` regression-test maintainability.

---

## Appendix: Critical-Review Reconciliation

Three parallel critical-review subagents (scientific-rigor / architecture / TDD-testability) reviewed the round-1 design.md and returned **13 BLOCKING + 15 IMPORTANT + 9 NIT findings**, all BLOCKED verdicts (per PR #5 precedent — round-1 BLOCKING is the expected baseline). Reconciliation involved 8 substantive decisions (S1–S8) with the user, plus ~10 surgical inline edits and ~12 deferred mechanical fixes that land in /openspec:proposal scaffolding (Stage 2). Each finding below is quoted, mapped to its resolution, and pointed at the inline location where the fix applies.

### Substantive decisions S1–S8 (user-confirmed in round-1 brainstorming follow-up)

- **S1 — SG-detrending of lateral signal**: ADD new `_noise.compute_sg_detrended(x, window, polynomial_order) -> np.ndarray` helper; reuse existing `SG_WINDOW_DETREND=23` + `SG_DEGREE=3` constants from PR #2/#3. Pipeline becomes: `project_to_growth_axis_perpendicular → compute_sg_detrended → CWT/FFT`. Addresses Sci-B1 ("Detrending omitted — directly contradicts preliminary_results §3.4"). Inline at D5 pseudocode step 1b + Migration Plan `_noise.py` section.
- **S2 — cadence-Nyquist dimensional rework**: ADD new `TEMPORAL_NYQUIST_RATIO_MAX = 0.25` constant per theory.md §1.2's "5-10 samples/period" rule. Existing `NYQUIST_RATIO_MAX = 0.25` stays as the SPATIAL check (per theory.md §6.5); clarifying docstring distinguishes. Formula stays raw `cadence_s / T_nutation_median`. New constant count: 6 (not 5). Addresses Sci-B2. Inline at D8 + Migration Plan `_constants.py` section + Migration Plan spec delta #2.
- **S3 — derr trait rename**: `derr_match_residual` → `period_residual_vs_derr_reference` (more self-documenting; survives `DERR_EXPECTED_PERIOD_S` override on non-rice species). Addresses Sci-B4. Inline at every D-section + §2 test + Migration Plan + Test-file imports (~30 occurrences, applied via `replace_all`).
- **S4 — NaN-gating revision**: NaN-gate ONLY 3 strictly biological-meaning-dependent traits (T_nutation_median, T_nutation_iqr, A_nutation_envelope_max) when `is_nutating==False`. Always populate `cadence_nyquist_ratio` (engineering diagnostic), `period_residual_vs_derr_reference` (spectral-peak diagnostic), `band_power_ratio`, `noise_floor_estimate`, `is_nutating`. Addresses Sci-I3. Inline at D5 NaN-gating table + `_compute_one_track` pseudocode + Context-section trait table.
- **S5 — Layer-2 ±2% strictness**: SOFTEN UPFRONT to "≥4 of 6 tracks within ±5%". Avoids the GREEN-phase reconciliation event PR #5's R3-B2 precedent showed is otherwise guaranteed (scale-grid discreteness alone gives ~5% floor at n_frames=575). Addresses TDD-B3. Inline at §2.H.3.
- **S6 — Canary atol cross-OS**: SET 1e-6 upfront (loosened from PR #5's 1e-9) because PR #6 adds 3 unverified scipy paths (fft/ndimage/stats) on top of PR #5's verified pywt path; 1e-6 cushion is scientifically irrelevant for these traits per CC-6. Addresses TDD-B4. Inline at §2.B.2.
- **S7 — `NOISE_FLOOR_OUT_OF_BAND_FACTOR` default**: KEEP factor=5.0 per CC-8 verbatim in RED-phase; DEFER empirical sensitivity decision to GREEN-phase (measure noise_floor/band_power on plate-001 across factor ∈ {3, 5, 7}, pick value maximizing `is_nutating==True` robustness, record in GREEN-phase Reconciliation Appendix). Addresses Sci-B3. Inline at D4 + R2 + §2.E (sensitivity test parametrize added).
- **S8 — `cadence_s` parameter location**: EXTEND `_types._validate_trajectory_df` to require `trajectory_df.attrs["cadence_s"]` (positive finite float); raises `ValueError` naming the key. Spec delta adds MODIFIED block on "Tier 0 input-validation boundary" requirement. PR #2/#3 test fixtures need ~10 LOC of updates to set `.attrs`. Addresses Arch-B3 (which caught my fabricated "PR #2/#3 precedent" claim). Inline at D5 cadence_s paragraph + Migration Plan spec delta #3 (new MODIFIED block).

### Scientific-rigor reviewer (R1)

- **B1 (BLOCKING)**: "Detrending missing — directly contradicts empirical anchor §1.2 / §3.4." Centering doesn't remove the documented ~20× drift. → **Addressed via S1**: SG-detrending added as Step 1b in `_compute_one_track`; new `_noise.compute_sg_detrended` helper; reuses existing constants.
- **B2 (BLOCKING)**: "cadence_nyquist_ratio mis-anchored to §6.5 (spatial)." Conflates px/px with s/s. → **Addressed via S2**: new `TEMPORAL_NYQUIST_RATIO_MAX = 0.25` constant; clarifying docstring on existing `NYQUIST_RATIO_MAX`.
- **B3 (BLOCKING)**: "OQ2 — second-harmonic contamination at factor=5 is a near-certainty; R2 arithmetic in original draft was wrong (claimed 2/T, 3/T are INSIDE the band, actually OUTSIDE at factor=5; but 5th/6th/7th harmonics are INSIDE)." → **Addressed via S7**: keep factor=5 (CC-8 verbatim) in RED-phase, add §2.E.X sensitivity test parametrized over factor ∈ {3, 5, 7}, GREEN-phase picks empirical value. R2 arithmetic correction noted in design.md (5th/6th/7th harmonics in noise band; lower-order harmonics in signal band).
- **B4 (BLOCKING)**: "`derr_match_residual` semantic — species-specific but trait emitted unconditionally without species metadata." → **Addressed via S3**: rename to `period_residual_vs_derr_reference`; column name now self-documents that it's a deviation from a configurable reference.
- **I1 (IMPORTANT, OQ1 follow-up)**: "`band_power_ratio` Parseval inconsistency on one-sided spectrum." → **Deferred to /openspec:proposal**: D7 docstring will note "one-sided spectral energy"; ratio is well-defined because numerator and denominator share the convention.
- **I2 (IMPORTANT)**: "Net-displacement growth axis biased by mid-trajectory turn." → **Deferred to /openspec:proposal**: D2 will gain a non-goal "PR #6 assumes monotonic growth-axis trajectories; non-monotonic tracks filtered upstream by QC tier's `track_is_clean` gate"; §2.H verifies all 6 plate-001 tracks pass the monotonic check.
- **I3 (IMPORTANT)**: "NaN-gating of `cadence_nyquist_ratio` is scientifically backwards (hides the diagnostic that explains WHY is_nutating failed)." → **Addressed via S4**: ungate `cadence_nyquist_ratio` AND `period_residual_vs_derr_reference`; only 3 strictly biological-meaning-dependent traits stay NaN-gated.
- **N1-N3 (NITs)**: R2 arithmetic, `_noise` median rationale rewrite, dimensional-consistency claim rewrite. → **Deferred to /openspec:proposal**: design.md text refinements applied during proposal scaffolding (not load-bearing for spec correctness).

### Architecture reviewer (R1)

- **B1 (BLOCKING)**: "Foundation test name `test_2G2_constants_version_is_4` is fabricated; actual is `test_schema_version_is_1_and_constants_version_is_4` at line 206." → **Addressed inline**: both Migration Plan occurrences updated to canonical name; docstring + inner assertion update documented.
- **B2 (BLOCKING)**: "`Trait CSV row-identity schema` MODIFIED block — that requirement has NO per-tier trait-column enumeration to extend." → **Addressed inline**: MODIFIED block DROPPED in Migration Plan spec delta; the new ADDED requirement "Tier 1 nutation trait emission API" already documents the 8 trait columns.
- **B3 (BLOCKING)**: "`trajectory_df.attrs['cadence_s']` precedent claim is FABRICATED; kinematics.py / qc.py don't take cadence_s." → **Addressed via S8**: extend `_validate_trajectory_df` to require attrs key; new MODIFIED block on "Tier 0 input-validation boundary" requirement; D5 cadence_s paragraph rewritten with honest scope (PR #2/#3 fixture updates required).
- **B4 (BLOCKING)**: "Package layout MODIFIED conflates ADDITION (new module) vs TRANSITION (stub → impl)." → **Addressed inline**: Migration Plan spec delta #1 rewritten to clarify "implementations grow 4 → 5 by ADDITION of `nutation`; stub-count stays 6; `nutation` not in stub-callable table"; new scenario added.
- **I1 (IMPORTANT)**: "Constants count: design claims 25→30, actually 29→35 (per real `_default_constants_snapshot()` enumeration)." → **Deferred to /openspec:proposal**: counts will be hard-anchored against `len(_default_constants_snapshot())` at impl time (RED-phase test will assert exact count).
- **I2 (IMPORTANT)**: "Explicit foundation-test comment-block edit instructions missing." → **Deferred to /openspec:proposal**: tasks.md will enumerate the comment-block update (add PR #6 line at lines 26-34 of `test_circumnutation_foundation.py`).
- **I3 (IMPORTANT)**: "`project_to_growth_axis_perpendicular` length convention (n) diverges from `compute_psi_g` (n-1); error policy (raise) diverges from compute_psi_g (silent NaN)." → **Partially addressed in /openspec:proposal**: D2 will gain "Why divergent length + error semantics" rationale + change error policy to return `np.full(n, np.nan)` on zero-displacement (matching graceful-NaN pattern; downstream `nutation.compute` propagates the NaN signal through the rest of the pipeline).
- **I4 (IMPORTANT)**: "`_CONSTANTS_VERSION` docstring template not provided." → **Deferred to /openspec:proposal**: tasks.md will provide the exact templated docstring addition.
- **I5 (IMPORTANT)**: "`smooth_ridge` spec missing 2 scenarios (constants resolution + pass-through)." → **Deferred to /openspec:proposal**: spec.md scaffolding will add 2 scenarios per D4 docstring contract.
- **N1-N3 (NITs)**: "6 emitted traits" header misleading (it's 8 columns), unused `attrs`/`scipy.signal` imports, y-image-downward convention. → **Deferred to /openspec:proposal**.

### TDD-testability reviewer (R1)

- **B1 (BLOCKING)**: "§2.C.3 noise-only test uses `T_nutation_s=None` which resolves to default 3333.0 in synthetic.py:415-416 — produces a sinusoid, NOT noise-only. Correct path is `amplitude_px=0.0`." → **Addressed inline**: §2.C.3 rewritten with `amplitude_px=0.0, noise_sigma_px=1.0, n_frames=1024, cadence_s=300.0, random_state=0`.
- **B2 (BLOCKING)**: "§2.C.1 T=1000 fails recovery the way PR #5 R3-B2 did; period_min floor = 2·cadence = 600 s, T=1000 only 1.67× period_min." → **Addressed inline**: §2.C.1 T set changed from `{1000, 3333, 6666}` to `{2000, 3333, 6666}`.
- **B3 (BLOCKING)**: "§2.H.3 6-of-6 Derr forensic match at ±2% likely empirically infeasible." → **Addressed via S5**: softened to "≥4 of 6 within ±5%" upfront, with reconciliation rationale (scale-grid discreteness + per-plant variance).
- **B4 (BLOCKING)**: "Canary atol=1e-9 unverified for scipy paths." → **Addressed via S6**: loosened to 1e-6 upfront with documented rationale; `equal_nan=False` lock added at §2.B.2.
- **B5 (BLOCKING)**: "§2.F.7 stationary-track test can't reach the fallback (fails upstream in `_validate_trajectory_df`)." → **Partially addressed in /openspec:proposal**: §2.F.7 will be rewritten to use a CLOSED-LOOP trajectory where `x[-1]==x[0] AND y[-1]==y[0]` while intermediate frames vary — passes upstream validation but triggers the zero-net-displacement fallback (per Architecture-I3 the helper returns `np.full(n, np.nan)` rather than raising, so the test asserts `is_nutating==False` + NaN-trait emission).
- **I1 (IMPORTANT)**: "§2.F id count too light (24 vs PR #5's 30)." → **Deferred to /openspec:proposal**: tasks.md will enumerate ≥30 ids covering all `_validate_trajectory_df` failure modes per validator (including the new `attrs['cadence_s']` failure modes from S8).
- **I2 (IMPORTANT)**: "§2.G id count light; per-constant override coverage missing." → **Deferred to /openspec:proposal**: tasks.md will enumerate ~16-18 ids, one per new constant × (functional-effect + snapshot-presence). New `TEMPORAL_NYQUIST_RATIO_MAX` (S2) adds 2 more ids.
- **I3 (IMPORTANT)**: "§2.G.9 'non-aligned track' not deterministic." → **Deferred to /openspec:proposal**: §2.G.9 will use `synthetic.generate_trajectory(growth_axis_angle_rad=π/4, ...)` with a derivable projection-factor assertion.
- **I4 (IMPORTANT)**: "§2.B.3 RED-phase NaN canary `equal_nan` lock not explicit." → **Addressed inline at §2.B.2** (the canary section): assertion now spells out `equal_nan=False`.
- **I5 (IMPORTANT)**: "§2.H.4 '≥5 of 6' arbitrary cutoff." → **Deferred to /openspec:proposal**: §2.H.4 will gain a reconciliation-anticipation paragraph: "If <5 tracks improve, record per-track IQR-pre/post deltas in GREEN-phase Reconciliation Appendix and re-anchor to the empirical median or to ≥3 of 6 with documentation."
- **I6 (IMPORTANT)**: "§2.E.1 single-seed noise — seed-stream-coupling per PR #5 R3-B1 precedent." → **Deferred to /openspec:proposal**: §2.E.1 will use n_frames=4096 (suppresses single-seed variance) AND test ≥3 seeds with majority-pass assertion.
- **I7 (IMPORTANT)**: "Foundation-test comment-block at lines 26-34 not enumerated." → **Deferred to /openspec:proposal** (same fix as Architecture-I2).
- **N1-N3 (NITs)**: Unused scipy.signal import (now USED by `compute_sg_detrended` per S1), `_DERR_MATCH_TOLERANCE_FOR_TEST` underscore note, pytest-xdist verification. → **Deferred to /openspec:proposal**.

### Round-1 reconciliation status

- **13 BLOCKING findings**: 13 of 13 resolved (10 via inline edits to design.md; 3 deferred to /openspec:proposal with explicit fix specs).
- **15 IMPORTANT findings**: 5 of 15 addressed inline; 10 deferred to /openspec:proposal with explicit fix specs.
- **9 NIT findings**: 0 of 9 addressed inline; 9 deferred to /openspec:proposal (all minor text refinements).
- **Round-2 review (Stage 1.5d) will**: verify the 10 inline fixes are correct, verify the 23 deferred items are reflected in /openspec:proposal scaffolding when it lands, and surface any NEW findings from the S1–S8 substantive design changes.

---

## Appendix: Critical-Review Reconciliation — Round 2

A second 3-subagent critical-review pass on the round-1-reconciled design.md surfaced **8 BLOCKING + 11 IMPORTANT + 9 NIT findings** (all three reviewers verdict: NEEDS REVISION — a clear improvement over round-1's BLOCKED verdict). Reconciliation involved 4 substantive decisions (S8', S2'', §2.C.1 T-set fix, §2.H.3 two-part assertion) with the user, plus inline edits to fix gaps from round-1 reconciliation that round-2 caught.

### Substantive round-2 decisions

- **S8' (REVERSAL of round-1 S8)**: explicit `cadence_s` positional parameter on `nutation.compute(trajectory_df, cadence_s, coordinate='lateral', constants=None)`. Drops the round-1 plan to extend `_validate_trajectory_df` with `attrs["cadence_s"]` requirement. Rationale (Architecture-B1 + I4 + TDD-I3 round-2): the round-1 estimate of "~10 LOC" of PR #2/PR #3 fixture updates was actually ~25-40 LOC across 68 call sites; the validator extension was a contract regression on cadence-independent tiers; the claimed cross-PR forward-compat benefit doesn't survive checking PR #7/#11/#14 stub signatures (none take cadence_s via attrs). The reversal: drops the Tier 0 input-validation-boundary MODIFIED spec block; drops PR #2/#3 fixture updates entirely; aligns with the `temporal_cwt.compute_scaleogram(x, cadence_s, constants)` PR #5 precedent. Net spec delta: 1 MODIFIED (Package layout) + 1 MODIFIED (Module-level constants) + 2 ADDED (Tier 1 nutation trait emission API, Temporal CWT ridge-continuity smoothing API).
- **S2'' (REVISION of round-1 S2 value)**: `TEMPORAL_NYQUIST_RATIO_MAX = 0.25` (not 0.2). Rationale (Sci-B2 round-2): theory.md §6.5 says "10-min cadence still works" — at T=3333 this is ratio 0.18, comfortably under 0.25 but borderline-fails my proposed 0.2. The 0.2 anchor I cited (§1.2's "5-10 samples/period") was not actually in theory.md verbatim. 0.25 = same conservative cushion as `NYQUIST_RATIO_MAX`; dimensional separation lives in the constant NAME + docstring, not in different values. NEW follow-up issue filed at Stage 6: "circumnutation: validate TEMPORAL_NYQUIST_RATIO_MAX from literature + multi-plate data" (mirrors #205-#208 pattern; bundles with the multi-plate sweep that resolves #202).
- **§2.C.1 T set fix (Sci-B1 round-2)**: T ∈ `{2000, 3333, 4500}` (not `{2000, 3333, 6666}`). Rationale: SG-detrending window = 23 frames × 300 s = 6900 s. T=6666 sits at 96% of the SG-window — partial signal suppression breaks the `is_nutating==True` assertion even when period is recovered correctly. T=4500 (65% of window) stays clear of the cutoff. Mirrors PR #5 R3-B2's "test-vs-resolution-floor" failure mode but at the OPPOSITE end of the band.
- **§2.H.3 two-part assertion fix (Sci-B3 round-2)**: TWO assertions — (1) `abs(median(per_track_residuals)) < 0.02` enforces CC-7 ±2% via the robust median; (2) `count(|residual| < 0.05 AND is_nutating) >= 4` acknowledges per-plant biological variance. Rationale: S5's round-1 "≥4 of 6 within ±5%" softening alone did NOT enforce CC-7's stated ±2% target. The two-part structure makes CC-7 load-bearing (median assertion) while accommodating biological reality (count assertion). Sci-I1 round-2 semantic note (about the AND-conjunction handling false-positives when `is_nutating==False` but `|residual| < 0.05`) is documented inline in §2.H.3.

### Scientific-rigor reviewer (R2)

- **B1 (BLOCKING, S1 × §2.C.1 interaction)**: "SG-detrending will SUPPRESS the T=6666 s analytical-oracle test case at the 6900 s window cutoff." → **Addressed via T set fix**: §2.C.1 now uses `{2000, 3333, 4500}`.
- **B2 (BLOCKING, S2 anchor wrong)**: "TEMPORAL_NYQUIST_RATIO_MAX = 0.2 is anchored to a non-existent §1.2 claim." → **Addressed via S2''**: value revised to 0.25 with §6.5 anchor; NEW follow-up issue filed for empirical validation.
- **B3 (BLOCKING, S5 doesn't enforce CC-7 ±2%)**: "Softening to ±5% on a SUBSET of tracks does NOT enforce CC-7." → **Addressed via §2.H.3 two-part assertion**: median ±2% AND ≥4 of 6 within ±5%.
- **I1 (IMPORTANT, S4 × §2.H.3 conjunction semantics)**: "What does the assertion mean when is_nutating==False but |residual|<0.05?" → **Addressed inline in §2.H.3**: documented as "ridge-of-noise artifact"; AND-conjunction in per-track count correctly excludes; median in the CC-7 assertion uses ALL 6 residuals (robust).
- **I2 (IMPORTANT, S7 × §2.H.3 spec scenario language)**: "factor=5 GREEN-phase decision couples to §2.H.3 pass-rate; scenario language should be 'characterizes' not 'satisfies' until factor is locked." → **Deferred to /openspec:proposal**: spec.md scaffolding will choose scenario language reflecting the GREEN-phase decoupling.
- **I3 (IMPORTANT, rice-tuning of SG_WINDOW_DETREND for non-rice species)**: "Species with T > ~6000 s require both SG_WINDOW_DETREND and DERR_EXPECTED_PERIOD_S override." → **Deferred to /openspec:proposal**: R3 in design.md will gain a cross-link to the new TEMPORAL_NYQUIST_RATIO_MAX follow-up issue; species-specificity is acknowledged but not blocking PR #6 (rice-default is correct for plate-001).
- **N1-N3 (NITs)**: spec-scenario value-pinning for TEMPORAL_NYQUIST_RATIO_MAX; scipy.signal.savgol_filter added to §2.B.2 atol=1e-6 rationale; SG_WINDOW_DETREND docstring should cite preliminary_results §3.4 rationale. → **Deferred to /openspec:proposal**.

### Architecture reviewer (R2)

- **B1 (BLOCKING, S8 scope understated + contract regression)**: "Fixture-update scope is 7× larger than claimed; validator extension contract-regresses PR #2/#3." → **Addressed via S8' REVERSAL**: explicit cadence_s positional parameter; no validator extension; no PR #2/#3 fixture updates.
- **B2 (BLOCKING, stale constants count 25→30)**: "Real count is 29→35 per `_default_constants_snapshot()` enumeration." → **Addressed inline at D9**: replaced parenthetical chain with hard-anchored "29 baseline + 6 PR #6 = 35"; §2.G.6 regression-guard asserts `len(_default_constants_snapshot()) == 35` exactly.
- **I1 (IMPORTANT, TEMPORAL_NYQUIST_RATIO_MAX missing from D8/ConstantsT field block)**: → **Addressed inline at D8 + D9**: D8 now mentions both `NYQUIST_RATIO_MAX` (spatial) and `TEMPORAL_NYQUIST_RATIO_MAX` (temporal) explicitly; D9 `ConstantsT` `attrs.field` block extended with the 6th field.
- **I2 (IMPORTANT, spec-delta scenario mismatch S5)**: "Line 972 still says 'all 6 tracks satisfy' but S5 softened to '≥4 of 6 within ±5%'." → **Deferred to /openspec:proposal**: spec.md scaffolding will use the two-part assertion language from §2.H.3 inline (verifies median ±2% + count ≥4 of 6 ±5%).
- **I3 (IMPORTANT, SG-detrend NaN-cascade hazard)**: "savgol_filter on a single NaN propagates NaN across 22 frames; need NaN-tolerant contract OR short-circuit for all-NaN signals." → **Addressed inline at §2.F.7 + Architecture-I3 round-1**: stationary-track path short-circuits BEFORE `_noise.compute_sg_detrended` when `_geometry.project_to_growth_axis_perpendicular` returns all-NaN; assertion explicitly checks no `np.RuntimeWarning("All-NaN slice encountered")`.
- **I4 (IMPORTANT, S8 cross-PR forward-compat claim doesn't hold)**: → **Addressed via S8' REVERSAL** (this finding was the deciding evidence).
- **N1 (NIT, D9 prose about PR #3 ConstantsT count)**: "PR #3 added 4 ConstantsT fields (FRAC_OUTLIER_STEPS_MAX, WORST_STEP_RATIO_MAX, SG_MSD_AGREEMENT_MAX, D2_MSD_AGREEMENT_MAX), not 'may have added zero'." → **Addressed inline at D9**: hedge replaced with verified count.
- **N2 (NIT, TEMPORAL_NYQUIST_RATIO_MAX value-pin in spec scenario)**: → **Deferred to /openspec:proposal**.
- **N3 (NIT, atol=1e-6 doesn't need spec scenario)**: → **Acknowledged**; no design change needed.

### TDD-testability reviewer (R2)

- **B1 (BLOCKING, §2.E.5 stale post-S4)**: "Still says 5 NaN-gated traits; should be 3 per S4." → **Addressed inline at §2.E.5/§2.E.6**: now correctly enumerates 3 NaN-gated (T_med, T_iqr, A_max) + 5 always-populated (is_nutating, band_power_ratio, noise_floor_estimate, cadence_nyquist_ratio, period_residual_vs_derr_reference).
- **B2 (BLOCKING, §2.F.7 stale text "all-equal")**: → **Addressed inline at §2.F.7**: redefined as closed-loop trajectory with `x[-1]==x[0] AND y[-1]==y[0]` but varying intermediate frames; passes upstream validation (no zero-variance check); triggers zero-net-displacement fallback in `_geometry.project_to_growth_axis_perpendicular` which returns `np.full(n, np.nan)`; downstream emits all-NaN trait row with `is_nutating=False`.
- **B3 (BLOCKING, TEMPORAL_NYQUIST_RATIO_MAX missing from D8 + §2.G.6 + test imports)**: → **Addressed inline**: D8 narrative updated with both NYQUIST_RATIO_MAX (spatial) and TEMPORAL_NYQUIST_RATIO_MAX (temporal); §2.G.6 EXPECTED_PR6_CONSTANTS set extended with TEMPORAL_NYQUIST_RATIO_MAX (6 entries); test-file imports block extended with TEMPORAL_NYQUIST_RATIO_MAX + SG_WINDOW_DETREND + SG_DEGREE (S1 carryovers).
- **I1 (IMPORTANT, S1 × §2.C.2 SG-detrend on pure sinusoid)**: "SG_WINDOW_DETREND=23 × 300 s = 6900 s >> 2× T=3333 s candidate period; savgol_filter removes substantial amplitude from a pure sinusoid; ±10% tolerance may shift after detrending." → **Deferred to /openspec:proposal**: §2.C.2 will either (a) use `constants=ConstantsT(SG_WINDOW_DETREND=1)` to bypass detrending on the synth oracle (cleanest), or (b) use larger noise_sigma_px so the residual carries the period. The /openspec:proposal scaffolding will choose based on empirical RED-phase observation.
- **I2 (IMPORTANT, §2.E factor sensitivity test not enumerated)**: "S7 reconciliation claimed 'inline at §2.E' but no §2.E.X parametrized factor ∈ {3, 5, 7} test exists." → **Deferred to /openspec:proposal**: tasks.md will add §2.E.7 explicitly with parametrize_id derivation.
- **I3 (IMPORTANT, S8 migration scope ~97 occurrences, not ~10 LOC)**: → **OBVIATED by S8' REVERSAL**: no PR #2/#3 fixture changes needed.
- **I4 (IMPORTANT, IMPLEMENTATIONS_WITH_CONSTANTS_KWARG count consistency)**: → **Acknowledged**; no design change needed (4 → 5 is correct).
- **N1 (NIT, R6 runtime estimate)**: "Will balloon to ~15 minutes if §2.E.7 sensitivity test lands." → **Deferred to /openspec:proposal**: R6 will update or mark §2.E.7 as `slow`-gated.
- **N2-N3 (NITs)**: T=2000 empirical-data verification, B4 NaN-canary verification. → **Acknowledged**; no design change needed.

### Round-2 reconciliation status

- **8 BLOCKING findings**: 8 of 8 resolved (6 via inline edits + 2 via substantive decisions S8'/S2''; the inline edits covered §2.C.1 T-set, §2.H.3 two-part assertion, §2.E.5 NaN count, §2.F.7 closed-loop, TEMPORAL_NYQUIST_RATIO_MAX missing-from-D8/§2.G.6/imports, constants count 29→35).
- **11 IMPORTANT findings**: 5 addressed inline (Sci-I1 §2.H.3 docs, Arch-I1 D8 + D9, Arch-I3 NaN-cascade short-circuit, Arch-N1 PR #3 count, TDD-I3 S8' obviated, TDD-I4 acknowledged) + 6 deferred to /openspec:proposal with explicit fix specs.
- **9 NIT findings**: 2 addressed inline (Arch-N1 prose fix, TDD-I4 acknowledged) + 7 deferred to /openspec:proposal.
- **No round-3 critical review planned**: the S8' reversal substantially simplified the design (drops the largest source of architectural complexity); remaining items are mechanical refinements appropriate for /openspec:proposal scaffolding + the 5-subagent /openspec-review pass that effectively replaces a round-3 design review. PR #5 needed 3 rounds because the COI factor √2 → √3 → √1.5 had empirical-correction churn; PR #6 round-2 substantive items were architectural (S8' reversal) and value-precision (S2'' = 0.25), neither of which expect further iteration absent new evidence.

---

---

## Appendix: Critical-Review Reconciliation — Round 3

*(Optional third pass scaffold; populated only if rounds 1-2 fail to converge — see PR #5 precedent where the COI factor went √2 → √3 → √1.5 over 3 rounds.)*

---

## Appendix: GREEN-phase Reconciliation

*(Populated post-TDD-GREEN if implementation deviates from approved design. Per the Stage 5.5 mandate: any deviation requires (a) updating proposal.md / spec.md / tasks.md to reflect reality, and (b) a `### Why X instead of Y?` section here.)*

### Why §X softened from "Y" to "Z"

*(To be filled if applicable.)*

### Other GREEN-phase observations (no deviation)

*(To be filled if applicable.)*

---

## Appendix: OpenSpec /openspec-review Reconciliation

*(To be filled after scaffolding proposal.md / tasks.md / specs/circumnutation/spec.md and running /openspec-review.)*

### Spec quality & OpenSpec best practices reviewer

*(To be filled.)*

### Code & architecture feasibility reviewer

*(To be filled.)*

### Issue alignment reviewer

*(To be filled.)*

### TDD & testing strategy reviewer

*(To be filled.)*

### Scientific rigor & data integrity reviewer

*(To be filled.)*

---

## Appendix: OpenSpec /openspec-review Reconciliation — Round 2

*(Optional second pass if round-1 reconciliation surfaces new findings; scaffold only.)*
