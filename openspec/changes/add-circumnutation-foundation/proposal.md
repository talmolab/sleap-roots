## Why

This is the first PR in the circumnutation analysis program tracked by `docs/circumnutation/roadmap.md`. It establishes the *contracts* — package layout, dependencies, calibration pattern, row-identity schema, units sidecar, module-constants pattern, logging convention — that every subsequent tier PR builds on. It contains no spectral analysis and no per-tier trait emission; those land in PRs #2–#11 of the roadmap.

Theoretical foundation: `docs/circumnutation/theory.md`. Empirical feasibility: `docs/circumnutation/preliminary_results_2026-05-07.md`. Cross-cutting concerns this PR addresses: CC-2 (constants), CC-3 (calibration pattern), CC-4 (row identity), CC-9 (logging) from the roadmap.

## What Changes

- **NEW capability `circumnutation`** — first set of foundational requirements only (package layout, calibration contract, row-identity schema, units sidecar, constants, logging). Subsequent PRs ADD requirements for each tier.
- **NEW dependency** `pywavelets >= 1.4` (declared in both `[dependencies]` and `[dependency-groups.dev]` / `[optional-dependencies.dev]` per repo convention).
- **NEW dependency** `scipy` made explicit (currently transitive via scikit-image; explicit declaration prevents silent breakage).
- **NEW package** `sleap_roots/circumnutation/` with stub modules (`__init__.py`, `_types.py`, `_constants.py`, `_io.py`) and no compute. The compute modules (`kinematics.py`, `qc.py`, `temporal_cwt.py`, `psi_g.py`, `midline.py`, `spatial_cwt.py`, `parametric.py`, `synthetic.py`, `plotting.py`, `pipeline.py`) are stubbed with `NotImplementedError` so the package import-tree is complete.
- **NEW data class** `CircumnutationInputs` (attrs-based) capturing `(trajectory_df, px_per_mm, cadence_s, R_mm, run_id)` with validation.
- **NEW module-level constants** in `sleap_roots/circumnutation/_constants.py`, listed in `docs/circumnutation/roadmap.md` cross-cutting concern CC-2.
- **NEW CSV row-identity schema** — eight columns `(series, sample_uid, timepoint, plate_id, plant_id, track_id, genotype, treatment)` per cross-cutting concern CC-4. `plant_id == track_id` today; both reserved for future divergence.
- **NEW units sidecar JSON** writer — `_io.write_units_sidecar(out_path, units_dict)`.
- **NEW per-module logger** convention: `logger = logging.getLogger(__name__)` in each module.
- **NEW investigation task** — read `tracked_tip_pipeline.py`, the integration-test fixtures, and the existing trait CSV outputs to determine the exact calibration-serialization pattern, then replicate. Acceptance: a downstream user can load `TrackedTipPipeline` output and `CircumnutationPipeline` output with the same loader. Documented in `_io.py` as the implementation reference.
- **Re-export** `CircumnutationInputs` from `sleap_roots/__init__.py`. The `CircumnutationPipeline` re-export is deferred until PR #14 when it actually exists.
- **No breaking changes** to existing pipelines, traits, or public API.

## Impact

- **Affected specs:** new capability `circumnutation` (this PR is the first to ADD requirements; subsequent PRs accumulate more).
- **Affected code:**
  - `pyproject.toml` (new deps + lock-file refresh)
  - new package `sleap_roots/circumnutation/` (4 real files + 10 stub files)
  - `sleap_roots/__init__.py` (one new re-export)
  - new tests `tests/test_circumnutation_foundation.py`
- **Phase 2 deferred (not in this PR):** all spectral analysis, trait emission, midline reconstruction, plotting, CLI. See roadmap.md PR/issue split.
- **Open scientific blockers:** none directly addressed by this PR; tracked in roadmap.md PRs #19–22.
