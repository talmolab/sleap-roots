## Why

This is the first PR in the circumnutation analysis program tracked by `docs/circumnutation/roadmap.md`. It establishes the *contracts* — package layout, dependencies, pure-pixel output convention, downstream `convert_to_mm` utility, row-identity schema, units sidecar, run-metadata sidecar, module-constants pattern, logging convention — that every subsequent tier PR builds on. It contains no spectral analysis and no per-tier trait emission; those land in PRs #2–#11 of the roadmap.

Theoretical foundation: `docs/circumnutation/theory.md`. Empirical feasibility: `docs/circumnutation/preliminary_results_2026-05-07.md`. Cross-cutting concerns this PR addresses: CC-2 (constants), CC-3 (pure-pixel convention + downstream conversion), CC-4 (row identity), CC-9 (logging) from the roadmap.

## What Changes

- **NEW capability `circumnutation`** — first set of foundational requirements only (package layout, pure-pixel convention, `convert_to_mm` utility, row-identity schema, units sidecar, run-metadata sidecar, constants pattern, logging). Subsequent PRs ADD requirements for each tier.
- **NEW dependency** `pywavelets >= 1.5` (declared in `[project.dependencies]` only — runtime dep). Floor raised to `>=1.5` because earlier pywavelets versions are not numpy-2 ABI-compatible (the lockfile already pulls numpy 2.x via scikit-image).
- **NEW dependency** `scipy` made explicit (currently transitive via scikit-image; explicit declaration prevents silent breakage). Runtime dep, `[project.dependencies]` only.
- **CHANGE `requires-python` from `>=3.7` to `>=3.10`** — the previous `>=3.7` declaration was stale (the lockfile already pulled `numpy>=2.x` which requires Python 3.10+) and CI runs only Python 3.11. Pinning to `>=3.10` declares the floor that matches actual support. Classifiers for 3.7 / 3.8 / 3.9 (all past EOL) removed. `tool.black.target-version = ["py310"]` set explicitly. CI Python (`.python-version`) is unchanged at 3.11.
- **NEW package** `sleap_roots/circumnutation/` (sub-package; precedent: `sleap_roots/viewer/`). Contains `__init__.py`, `_types.py`, `_constants.py`, `_io.py`, `units.py`, plus 10 stub modules whose canonical callables raise `NotImplementedError("PR #<N> — see docs/circumnutation/roadmap.md")` so the import-tree is complete from PR #1 forward.
- **NEW `convert_to_mm` utility** in `sleap_roots/circumnutation/units.py` — pure function that takes a per-plant traits DataFrame + units mapping + `px_per_mm` and returns a copy with `_px` columns multiplied to `_mm`. Calibration is a downstream concern, never a pipeline concern.
- **NEW pure-pixel output convention** — every trait emitted by the pipeline carries pixel units (`px`, `px²`, `px/frame`, `px/hr`, `px·hr⁻¹`) or unit-independent units (`hr`, `hr⁻¹`, `s`, `rad`, dimensionless, boolean, int, string). NO `[mm]` columns ever emitted. Matches `TrackedTipPipeline`'s `lengths: "pixels"` convention exactly.
- **NEW data class** `CircumnutationInputs` (attrs-based) capturing `(trajectory_df, cadence_s, R_px, run_id)` with validation. NO `px_per_mm` parameter. `R_px` is the root cross-section radius in pixels (Bastien-Meroz 2016 Eqs. 20-21 use `R/L` ratios that cancel dimensions, so the user provides `R` in pixels using whatever calibration they trust; the pipeline never needs to know the calibration).
- **NEW module-level constants** in `sleap_roots/circumnutation/_constants.py`. Listed in `roadmap.md` CC-2.
- **NEW CSV row-identity schema** — eight columns `(series, sample_uid, timepoint, plate_id, plant_id, track_id, genotype, treatment)` per CC-4. `plant_id == track_id` today; both reserved for future divergence. `plate_id`, `genotype`, `treatment` are populated from upstream metadata where available, NaN otherwise. Note: `plate_id` and `treatment` are aspirational columns — no upstream produces them today; the schema reserves them so future ingest work is non-breaking.
- **NEW units sidecar JSON** writer — `_io.write_units_sidecar(out_path, units_dict)`. UTF-8 encoded. Vocabulary: `{"px", "px²", "px/frame", "px/hr", "px·hr⁻¹", "hr", "hr⁻¹", "s", "rad", "bool", "int", "string", "—"}`. Every column has an entry.
- **NEW run-metadata sidecar JSON** — captures `(input_path, sleap_roots_git_sha, sleap_roots_version, sleap_io_version, python_version, timestamp, run_id, _schema_version, _constants_version, _constants_snapshot)`. Provenance contract — every output is traceable.
- **NEW schema and constants version sentinels** — `_SCHEMA_VERSION = 1` (bumps when row-identity columns or sidecar shape changes); `_CONSTANTS_VERSION = 1` (bumps when any default in `_constants.py` changes). Both surface in `run_metadata.json`.
- **NEW per-module logger** convention: `logger = logging.getLogger(__name__)` in each module. Package does NOT call `basicConfig` at import time.
- **Re-exports** from `sleap_roots/__init__.py`: `CircumnutationInputs`, `convert_to_mm`. The `CircumnutationPipeline` re-export is deferred until PR #14 when it actually exists.
- **mkdocs API doc handling** — stub modules will produce auto-generated reference pages via `docs/gen_ref_pages.py`. Either (a) write full Google-style docstrings on the stub callables so the rendered pages are sparse but informative, or (b) edit `gen_ref_pages.py` to skip `sleap_roots/circumnutation/` until later PRs. PR #1 chooses (a) — pure scaffolding cost is one good docstring per stub callable.
- **Cross-link** to existing issue [#195 (Provenance: include slp_path in TrackedTipPipeline JSON top level)](https://github.com/talmolab/sleap-roots/issues/195) — the run-metadata sidecar contract here uses similar provenance fields; if #195 lands the JSON shape may converge.
- **No breaking changes** to existing pipelines, traits, or public API.

## Impact

- **Affected specs:** new capability `circumnutation` (this PR is the first to ADD requirements; subsequent PRs accumulate more).
- **Affected code:**
  - `pyproject.toml` (new runtime deps; both deps belong in `[project.dependencies]` only — `[dependency-groups.dev]` and `[optional-dependencies.dev]` are dev-tooling groups, not runtime deps)
  - `uv.lock` (regenerated)
  - new package `sleap_roots/circumnutation/` (5 contract modules: `__init__.py`, `_constants.py`, `_types.py`, `_io.py`, `units.py` + 10 stub modules: `kinematics.py`, `qc.py`, `synthetic.py`, `temporal_cwt.py`, `psi_g.py`, `midline.py`, `spatial_cwt.py`, `parametric.py`, `plotting.py`, `pipeline.py`)
  - `sleap_roots/__init__.py` (two new re-exports: `CircumnutationInputs`, `convert_to_mm`)
  - new tests `tests/test_circumnutation_foundation.py`
  - `docs/gen_ref_pages.py` — no edit; stubs ship with full docstrings
  - `docs/changelog.md` (one entry under "Added" or equivalent — repo convention, see CLAUDE.md)
- **Phase 1 deferred (not in this PR):** all spectral analysis, trait computation, midline reconstruction, plotting, CLI subcommand, `CircumnutationPipeline` class. See roadmap.md PR/issue split.
- **Open scientific blockers:** **none gating this PR** — pure-pixel convention decouples the DPI ambiguity (roadmap PR #19) from the pipeline. The DPI question becomes a downstream-converter input, never a pipeline input.
