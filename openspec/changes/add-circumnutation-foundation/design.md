# Design: add-circumnutation-foundation

## Context

This PR is foundational — no compute, only contracts. The detailed theoretical content is in `docs/circumnutation/theory.md`; the umbrella roadmap is in `docs/circumnutation/roadmap.md`. This design doc covers only the implementation choices that affect every subsequent tier PR.

## Goals / Non-Goals

**Goals:**
- Establish the package import-tree so subsequent PRs only need to fill in functions, never restructure.
- Lock the **pure-pixel output convention** (CC-3) so the pipeline never depends on calibration. Every length-bearing trait the pipeline emits is in pixels; users compose `convert_to_mm()` downstream when they want mm output.
- Lock the trait CSV row-identity schema (CC-4) so per-genotype / per-plate / per-plant aggregation is feasible without re-engineering.
- Lock the constants pattern (CC-2) — overridable defaults in `_constants.py` with `_CONSTANTS_VERSION` bump rule.
- Lock provenance — `run_metadata.json` sidecar with git SHA, schema/constants versions, input paths, SLEAP and Python versions.
- Lock the logging convention (CC-9).

**Non-Goals:**
- No trait computation. All 10 compute modules are stubs raising `NotImplementedError`.
- No CLI subcommand (lands PR #17).
- No `CircumnutationPipeline` class (lands PR #14).
- No re-implementation of LOESS or wavelet machinery (lands later PRs).
- **No calibration handling in the pipeline.** The pipeline never accepts `px_per_mm` as a parameter; calibration is always a downstream concern.

## Decisions

### D1. Package layout

```
sleap_roots/circumnutation/
├── __init__.py           # public API; re-exports CircumnutationInputs + convert_to_mm
├── _constants.py         # all module-level named constants + _SCHEMA_VERSION + _CONSTANTS_VERSION
├── _types.py             # CircumnutationInputs attrs class + ROW_IDENTITY_COLUMNS tuple
├── _io.py                # units-sidecar writer + run-metadata sidecar writer
├── units.py              # convert_to_mm() pure function (px → mm conversion utility)
├── kinematics.py         # NotImplementedError stub (PR #2)
├── qc.py                 # NotImplementedError stub (PR #3)
├── synthetic.py          # NotImplementedError stub (PR #4)
├── temporal_cwt.py       # NotImplementedError stub (PR #5)
├── psi_g.py              # NotImplementedError stub (PR #7)
├── midline.py            # NotImplementedError stub (PR #8)
├── spatial_cwt.py        # NotImplementedError stub (PR #9)
├── parametric.py         # NotImplementedError stub (PR #11)
├── plotting.py           # NotImplementedError stub (PR #16)
└── pipeline.py           # NotImplementedError stub (PR #14)
```

5 real contract modules + 10 stubs = 15 files. Sub-package precedent: `sleap_roots/viewer/` (4 files). Each stub module exposes one canonical callable raising `NotImplementedError(f"PR #{N} — see docs/circumnutation/roadmap.md")`. Stubs are *importable* — only function call raises.

**Alternatives considered.**

- *Flat layout* — modules at `sleap_roots/circumnutation_<tier>.py`. Rejected: 15 prefixed files clutter the top-level namespace; the `circumnutation_` prefix is redundant once the sub-package exists.
- *Single big file* — `sleap_roots/circumnutation.py` with all tiers inside. Rejected: final size ~1500–2000 LOC; per-tier PR splits would create merge-conflict hell on a single file; per-module test pattern broken.
- *Pipeline class outside the package* (mirroring `sleap_roots/tracked_tip_pipeline.py`). Rejected: this package contains many composable units, not one pipeline; keeping the pipeline inside the package alongside the units it composes is cleaner.

### D2. Pure-pixel pipeline convention; downstream `convert_to_mm()` utility

The pipeline never accepts `px_per_mm` and never emits `[mm]` columns. Internal computation is pixel-native; output is pixel-native. This matches `TrackedTipPipeline`'s `lengths: "pixels"` convention exactly (see `_TRACKED_TIP_UNITS` in `sleap_roots/tracked_tip_pipeline.py`).

The `R` parameter (Bastien-Meroz cross-section radius, used in Tier 4 for Rivière 2022 Eq. 1: `δ̇₀ = ω·R·Δφ / (2·ΔL)`) is a length. The formula's `R/ΔL` ratio is dimensionless, so as long as `R` and `ΔL` use the same length unit, the formula returns the correct `δ̇₀ [hr⁻¹]`. The pipeline accepts `R_px` (pixels); the user converts their physical-unit measurement to pixels using whatever calibration they trust. The pipeline never needs the calibration.

`convert_to_mm()` lives in `sleap_roots/circumnutation/units.py`:

```python
def convert_to_mm(
    traits_df: pd.DataFrame,
    units: dict[str, str],
    px_per_mm: float,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Convert pixel-unit columns of a per-plant traits DataFrame to mm.

    Pure function. Original df and units dict are not mutated. Columns
    whose unit string is "px" or "px²" are scaled and renamed to "_mm" /
    "_mm²" forms; "px/frame" → "mm/frame"; "px/hr" → "mm/hr"; etc. Other
    columns (rad, hr, hr⁻¹, s, bool, int, string, dimensionless) pass
    through unchanged.

    Args:
        traits_df: per-plant trait DataFrame with `_px`-suffixed columns.
        units: column-name → unit-string mapping in the documented vocabulary.
        px_per_mm: positive finite calibration factor.

    Returns:
        (mm_traits_df, mm_units) — copies with calibration applied.
    """
```

The function is implemented in PR #1 because (a) it's small (~30 LOC), (b) the test surface is trivial, (c) having it land with the foundation means the contract is "everything is px; here's how to convert" from day one.

**This decision resolves five review findings at once:** the calibration-pattern blocker (no calibration in the pipeline = nothing to replicate from `TrackedTipPipeline`), the `calibration_present` QC trait (deleted), the calibration_source/confidence provenance fields (moot), the DPI ambiguity (decoupled — pipeline runs without it), and the px_per_mm boundary scenarios (irrelevant — pipeline never takes the parameter).

### D3. Row-identity schema (CC-4)

Trait CSV row-identity columns at the front of every row, in this order:

```
series, sample_uid, timepoint, plate_id, plant_id, track_id, genotype, treatment
```

Today `plant_id == track_id` and both columns are populated identically. Both are reserved so a future divergence (multi-track-per-plant) is non-breaking. `genotype` is populated from `Series` metadata via the `series-metadata` capability (PR #171, merged) where available, NaN otherwise. **`plate_id` and `treatment` are aspirational columns — no upstream produces them today.** They populate as NaN; the schema reserves them so future upstream metadata work (e.g., a follow-up to PR #171 that adds plate-level grouping) is a non-breaking schema extension rather than a breaking redesign.

The trait CSV is sorted via `pandas.DataFrame.sort_values(by=['series', 'sample_uid', 'plate_id', 'plant_id', 'track_id'])` — string columns sort lexicographically, integer columns sort numerically. (Spec scenario explicitly verifies `track_id=2` precedes `track_id=10` after sort.)

### D4. Constants strategy (CC-2)

All defaults live in `_constants.py` as module-level UPPER_SNAKE constants with type hints. Plus two version sentinels:

- `_SCHEMA_VERSION: int = 1` — bumps when row-identity columns or sidecar JSON shapes change.
- `_CONSTANTS_VERSION: int = 1` — bumps when any default in `_constants.py` changes.

Both versions are emitted into `run_metadata.json` so a downstream user knows which version produced a CSV.

Trait functions take an optional `constants: Optional[ConstantsT] = None` parameter (a frozen attrs class) so callers can override per-call without monkey-patching the module. The `Pipeline` class (PR #14) accepts `constants=` at init and passes through.

### D5. Run-metadata sidecar (provenance contract)

Every per-plant CSV gets a sibling `run_metadata.json` with:

- `input_path: str` — absolute path to the input file (.slp or .csv)
- `sleap_roots_git_sha: str` — git SHA of the running sleap-roots
- `sleap_roots_version: str` — `sleap_roots.__version__`
- `sleap_io_version: str` — `sio.__version__`
- `python_version: str` — `sys.version`
- `timestamp: str` — ISO 8601 UTC at write time
- `run_id: Optional[str]` — user-provided
- `_schema_version: int` — from `_constants.py`
- `_constants_version: int` — from `_constants.py`
- `_constants_snapshot: dict[str, Any]` — every name in `_constants.py` and its value at the time of the run

This is the SCIENCE PROVENANCE contract — required at the foundation, not deferred. Without it, every subsequent PR can emit unprovenanced CSVs and the program's reproducibility verdict is compromised.

### D6. Logging (CC-9)

Per-module logger via `logger = logging.getLogger(__name__)`, matching `tracked_tip_pipeline.py`. The package does NOT configure logging at import time — that's the application's responsibility. Tests use `caplog` for verifying messages.

### D7. mkdocs API doc generation for stubs

`docs/gen_ref_pages.py` auto-emits API reference pages for every public `.py` under `sleap_roots/`. The 10 stub modules are public names → mkdocstrings will pull their docstrings. Strategy: every stub callable ships with a complete Google-style docstring (Args, Returns, Raises) declaring the eventual contract even though the body raises. The rendered API pages will be sparse but informative ("Coming in PR #N — see roadmap"). `pydocstyle --convention=google` passes; mkdocs build passes.

### D8. I/O conventions (cross-platform safety)

Match existing repo convention: `pathlib.Path` everywhere, `.as_posix()` at the I/O boundary when passing to APIs that take strings (see `tracked_tip_pipeline.py:337,342,347,486,490,496` and `series.py:132`). All sidecar JSON files written with `encoding="utf-8"`. Non-ASCII unit strings (e.g., `µm`, `hr⁻¹`, `px²`) MUST round-trip without corruption on Windows where the default `open()` encoding is cp1252.

### D9. Test scope for this PR

- Schema tests: row-identity columns exist; correct dtypes; `plant_id == track_id` column-wise; sort order is `(series, sample_uid, plate_id, plant_id, track_id)` with numeric `track_id`.
- `CircumnutationInputs` tests: valid construction; missing row-identity column → ValueError; `cadence_s ≤ 0` → ValueError; `R_px ≤ 0` → ValueError; empty `trajectory_df` → ValueError.
- Stub tests: every stub module imports cleanly; calling its canonical callable raises `NotImplementedError` with message matching regex `r"PR #\d+ — see docs/circumnutation/roadmap\.md"`. Parametrized over all 10 stubs.
- Constants tests: every required name is importable from `_constants` with the documented default; `_SCHEMA_VERSION` and `_CONSTANTS_VERSION` exist as integers.
- Units sidecar tests: file exists, parses, every numeric column has an entry, every value is in the documented vocabulary, UTF-8 roundtrip with non-ASCII string.
- Run-metadata tests: file exists, has all required fields, `_constants_snapshot` matches the loaded `_constants.py` values.
- `convert_to_mm()` tests: `px_per_mm = 47.24` correctly scales `_px` columns; `px_per_mm = 1.0` is identity (column values unchanged but renamed); pure-function (input df not mutated); units dict updated correctly; non-px columns pass through.
- `__init__.py` re-export tests: `from sleap_roots import CircumnutationInputs, convert_to_mm` succeeds.
- Logging test: `getLogger("sleap_roots.circumnutation.<modname>")` returns the module's logger; package import does NOT add handlers to root logger; no records emitted at import time (verified via `caplog`).
- Coverage gate: `--cov=sleap_roots.circumnutation` ≥ 95%.

## Risks / Trade-offs

| Risk | Mitigation |
|---|---|
| Subsequent PRs end up needing to MODIFY (not ADD) the foundational requirements. | Keep this PR's requirements at the *contract* level; tier-specific behavior is ADDED by tier PRs. If a tier PR finds the contract too narrow, surface as a discussion before MODIFYing. |
| Constants chosen here turn out wrong when first used in a tier PR. | Constants are overridable per-call; wrong defaults are tunable, not blockers. `_CONSTANTS_VERSION` bumps capture the change in `run_metadata.json` for traceability. |
| `convert_to_mm()` is implemented before any `_px` column exists. | Tests use a synthetic 1-row trait DataFrame with hand-coded `length_px`, `velocity_px_per_hr` columns to exercise the conversion logic. The function works on any DataFrame matching the unit-string contract; doesn't depend on tier output. |
| Stub `NotImplementedError` raised at module-import time would break `import sleap_roots`. | Stubs raise inside *function bodies*, not at module top-level. Test 3.2b explicitly verifies `import sleap_roots` succeeds. |
| `plate_id` / `treatment` reserved-but-NaN columns confuse downstream users expecting populated values. | Documented behavior: schema is forward-compatible reservation. Future upstream metadata extends; loaders handle NaN gracefully. |

## Migration Plan

This is purely additive. No existing behavior changes. Rollback = revert the PR.

## Open Questions

1. Should `CircumnutationInputs` accept the trajectory DataFrame directly, or a path to a `TrackedTipPipeline` output CSV (let it load)? Strawman: accept either, dispatching by type. Resolved during implementation; not blocking.
2. Should `_constants.py` expose constants as a frozen attrs class (`ConstantsT`) with named fields, or as plain module-level scope plus a separate frozen attrs class for the override-bag? Strawman: both — module-level constants for ergonomic import, attrs class for typed override-bag. Resolved during implementation.
3. Existing issue [#195](https://github.com/talmolab/sleap-roots/issues/195) proposes adding `slp_path` to `TrackedTipPipeline`'s top-level JSON. The run-metadata contract here uses `input_path` for the same purpose; if #195 lands the JSON shape may converge. Track but don't block.
