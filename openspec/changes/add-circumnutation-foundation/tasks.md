# Tasks for add-circumnutation-foundation

TDD-ordered. Tests precede implementation per `superpowers:test-driven-development`.

## 1. Investigation (must precede 4)

- [ ] 1.1 Read `sleap_roots/tracked_tip_pipeline.py`, its integration tests in `tests/test_tracked_tip_pipeline.py`, and a sample CSV output produced by it. Document the exact `px_per_mm` serialization pattern (location: sidecar JSON, CSV header, attrs metadata, per-row column?) in a docstring comment at the top of `sleap_roots/circumnutation/_io.py`. This output drives the implementation in task 4.

## 2. Dependencies

- [ ] 2.1 Add `pywavelets >= 1.4` and `scipy` (explicit) to `pyproject.toml` `[project.dependencies]`.
- [ ] 2.2 Mirror in `[dependency-groups.dev]` and `[project.optional-dependencies.dev]` as needed (review repo convention).
- [ ] 2.3 Run `uv lock`; commit `uv.lock`.
- [ ] 2.4 Run `uv sync` and confirm `import pywt; import scipy` succeeds in the dev env.

## 3. Tests (write before implementation)

- [ ] 3.1 Create `tests/test_circumnutation_foundation.py`.
- [ ] 3.2 Write test: importing every stub module succeeds (matches spec scenario "All stub modules import cleanly").
- [ ] 3.3 Write test: calling a stub raises `NotImplementedError` containing `"PR #"` and `"docs/circumnutation/roadmap.md"` (matches scenario "Calling a stub raises NotImplementedError").
- [ ] 3.4 Write test: every required constant from the spec is importable from `_constants` with the documented default (matches scenario "All required constants are importable").
- [ ] 3.5 Write test: `CircumnutationInputs` valid construction (matches scenario "Valid construction").
- [ ] 3.6 Write test: `CircumnutationInputs` missing row-identity column raises `ValueError` (matches scenario "Missing row-identity column").
- [ ] 3.7 Write test: `from sleap_roots import CircumnutationInputs` succeeds (matches scenario "Importable from top-level").
- [ ] 3.8 Write test: row-identity schema columns exist with correct dtypes; `plant_id == track_id` column-wise; rows sorted lexicographically (matches scenario "Schema columns exist with correct dtypes").
- [ ] 3.9 Write test: calibration round-trip — write a per-plant CSV with `px_per_mm = 47.24`, load via the foundation's reader (which mirrors `TrackedTipPipeline`'s pattern), recover the value (matches scenario "Calibration provided"). Skip until task 1.1 documents the pattern.
- [ ] 3.10 Write test: calibration omitted — `[mm]` traits are NaN; `calibration_present == False`; no exception (matches scenario "Calibration omitted"). Use a tiny per-plant CSV with one synthetic NaN-emitting trait to exercise the path.
- [ ] 3.11 Write test: units sidecar JSON exists, parses, contains every numeric column, every value in the documented vocabulary (matches scenario "Units sidecar exists and is valid").
- [ ] 3.12 Write test: per-module logger convention — `logging.getLogger("sleap_roots.circumnutation.<modname>")` returns the module's logger object; package import doesn't add root handlers (matches scenario "Module loggers are namespaced").
- [ ] 3.13 Run the test suite — expect all to FAIL with `ImportError` or `AttributeError`. (TDD red phase.)

## 4. Implementation

- [ ] 4.1 Create `sleap_roots/circumnutation/_constants.py` with the constant set listed in the spec, values per `roadmap.md` CC-2.
- [ ] 4.2 Create `sleap_roots/circumnutation/_types.py` with `CircumnutationInputs` (attrs class) and `ROW_IDENTITY_COLUMNS = ("series", "sample_uid", "timepoint", "plate_id", "plant_id", "track_id", "genotype", "treatment")`.
- [ ] 4.3 Create `sleap_roots/circumnutation/_io.py` with:
  - top-of-file docstring comment recording task 1.1's findings
  - `write_per_plant_csv(out_path, df, units, px_per_mm, run_metadata)` matching `TrackedTipPipeline`'s pattern
  - `read_per_plant_csv(in_path) -> (df, units, px_per_mm, run_metadata)` complementary loader
  - `write_units_sidecar(out_path, units_dict)` writing `<out_path>.units.json`
- [ ] 4.4 Create `sleap_roots/circumnutation/__init__.py` re-exporting `CircumnutationInputs` and `ROW_IDENTITY_COLUMNS`.
- [ ] 4.5 Create the 10 stub modules (`kinematics`, `qc`, `synthetic`, `temporal_cwt`, `psi_g`, `midline`, `spatial_cwt`, `parametric`, `plotting`, `pipeline`). Each contains:
  - `import logging; logger = logging.getLogger(__name__)`
  - module docstring naming the PR # that fills it in
  - one or two named callables that raise `NotImplementedError("PR #N — see docs/circumnutation/roadmap.md")` with the appropriate `N`
- [ ] 4.6 Edit `sleap_roots/__init__.py` to add `from sleap_roots.circumnutation import CircumnutationInputs`.

## 5. Verify

- [ ] 5.1 Run `uv run pytest tests/test_circumnutation_foundation.py -v` — all green.
- [ ] 5.2 Run `uv run black --check sleap_roots tests` — passes.
- [ ] 5.3 Run `uv run pydocstyle --convention=google sleap_roots/circumnutation` — passes.
- [ ] 5.4 Run `openspec validate add-circumnutation-foundation --strict` — valid.
- [ ] 5.5 Open PR linking GitHub issue + `add-circumnutation-foundation` change-id.

## 6. Post-merge

- [ ] 6.1 Update `docs/circumnutation/roadmap.md` PR #1 status to ✅.
- [ ] 6.2 Run `openspec archive add-circumnutation-foundation --yes` to fold the foundational requirements into `openspec/specs/circumnutation/spec.md`.
