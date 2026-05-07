# Tasks for add-circumnutation-foundation

TDD-ordered. Tests precede implementation per `superpowers:test-driven-development`. **Do not push commits from section 3 (red phase) without section 4 (implementation) in the same push** — the suite is expected to be red between 3.x and 4.x and the PR is one logical unit.

## 1. Dependencies

- [ ] 1.1 Add `pywavelets >= 1.4` to `[project.dependencies]` in `pyproject.toml` (runtime dep — does NOT belong in `[dependency-groups.dev]` or `[project.optional-dependencies.dev]`, those are dev-tooling groups).
- [ ] 1.2 Add `scipy` to `[project.dependencies]` (currently transitive via scikit-image; explicit declaration prevents silent breakage).
- [ ] 1.3 Run `uv lock`; commit `uv.lock`.
- [ ] 1.4 Run `uv sync` and confirm `import pywt; import scipy` succeeds in the dev env.

## 2. Tests (write before implementation — TDD red phase)

- [ ] 2.1 Create `tests/test_circumnutation_foundation.py` using `tmp_path` fixture for every file-emitting test.
- [ ] 2.2 Write test "all stub modules import cleanly" — explicit import of each of `kinematics`, `qc`, `synthetic`, `temporal_cwt`, `psi_g`, `midline`, `spatial_cwt`, `parametric`, `plotting`, `pipeline`. (Spec scenario "All stub modules import cleanly".)
- [ ] 2.3 Write test "calling each stub raises NotImplementedError with correct PR number" — `pytest.mark.parametrize` over the 10 (module_name, callable_name, expected_pr_number) tuples from spec Requirement: Package layout. For each, import the module, call its canonical callable, assert `NotImplementedError` is raised, assert the message matches regex `r"^PR #(\d+) — see docs/circumnutation/roadmap\.md$"`, assert the captured group equals the expected PR number. (Spec scenario "Calling each stub raises NotImplementedError with the correct PR number".)
- [ ] 2.4 Write test "import sleap_roots succeeds" — `import sleap_roots; assert hasattr(sleap_roots, 'CircumnutationInputs'); assert hasattr(sleap_roots, 'convert_to_mm')`. (Spec scenario "import sleap_roots succeeds without raising".)
- [ ] 2.5 Write test "every required constant is importable" — `from sleap_roots.circumnutation import _constants` then assert each name from spec Requirement: Module-level constants exists with the documented default value. Include `_SCHEMA_VERSION == 1` and `_CONSTANTS_VERSION == 1`. (Spec scenario "All required constants are importable with correct types".)
- [ ] 2.6 Write test "CircumnutationInputs valid construction". (Spec scenario "Valid construction".)
- [ ] 2.7 Write test "missing row-identity column raises ValueError naming the column" — parametrize over all 8 row-identity columns. (Spec scenario "Missing row-identity column".)
- [ ] 2.8 Write test "empty trajectory_df raises ValueError". (Spec scenario "Empty trajectory DataFrame".)
- [ ] 2.9 Write test "invalid cadence_s raises ValueError" — parametrize over `0.0`, `-1.0`, `float('nan')`. (Spec scenario "Invalid cadence_s".)
- [ ] 2.10 Write test "invalid R_px raises ValueError" — parametrize over `0.0`, `-2.4`, `float('nan')`; also assert `R_px=None` succeeds. (Spec scenario "Invalid R_px".)
- [ ] 2.11 Write test "importable from top-level" — `from sleap_roots import CircumnutationInputs, convert_to_mm`. (Spec scenario "Importable from top-level".)
- [ ] 2.12 Write test "schema columns exist with correct dtypes" — build a DataFrame for 6 tracks via the foundation's row builder; assert column order, `track_id` is int, `plant_id` column-wise equals `track_id`, identity columns are object dtype. (Spec scenario "Schema columns exist with correct dtypes".)
- [ ] 2.13 Write test "sort order is numeric for track_id" — DataFrame with `track_id ∈ {2, 10}`; assert row with `track_id=2` precedes row with `track_id=10`. (Spec scenario "Sort order is numeric for track_id".)
- [ ] 2.14 Write test "pipeline output is calibration-independent" — assert no column unit string in the output is mm-bearing. (Spec scenario "Pipeline output is calibration-independent".)
- [ ] 2.15 Write test "convert_to_mm identity at px_per_mm=1.0". (Spec scenario "Identity at px_per_mm = 1.0".)
- [ ] 2.16 Write test "convert_to_mm 1200 DPI conversion". (Spec scenario "1200 DPI conversion".)
- [ ] 2.17 Write test "convert_to_mm velocity unit conversions" — covers `px/hr → mm/hr`, `px/frame → mm/frame`. (Spec scenario "Velocity unit conversions".)
- [ ] 2.18 Write test "convert_to_mm non-px columns pass through". (Spec scenario "Non-px columns pass through".)
- [ ] 2.19 Write test "convert_to_mm invalid px_per_mm" — parametrize over `0.0`, `-1.0`, `float('nan')`. (Spec scenario "Invalid px_per_mm".)
- [ ] 2.20 Write test "units sidecar exists and parses" — write a CSV via `_io.write_per_plant_csv`, assert sibling `traits_per_plant.units.json` exists, parses, every column present, every value in the vocabulary. (Spec scenario "Sidecar exists and parses".)
- [ ] 2.21 Write test "UTF-8 round-trip" — units dict containing `{"helix_signed_area": "px²"}` round-trips byte-for-byte. (Spec scenario "UTF-8 round-trip with non-ASCII unit".)
- [ ] 2.22 Write test "run-metadata sidecar contains required fields" — write a CSV, assert sibling `run_metadata.json` exists with all required keys. (Spec scenario "Run-metadata sidecar contains required fields".)
- [ ] 2.23 Write test "constants snapshot reflects override" — pass a custom `ConstantsT` override; assert snapshot reflects it; default-valued constants remain unmodified. (Spec scenario "Constants snapshot reflects override".)
- [ ] 2.24 Write test "module loggers are namespaced". (Spec scenario "Module loggers are namespaced".)
- [ ] 2.25 Write test "no handlers added on import; no records emitted at import time" — `caplog` captures at `DEBUG` for `sleap_roots.circumnutation`; snapshot root-logger handlers before/after import; assert no records emitted, no new handlers. (Spec scenario "No handlers added on import".)
- [ ] 2.26 Run `uv run pytest tests/test_circumnutation_foundation.py` — expect tests to FAIL (ImportError or AttributeError). This is the TDD red phase.

## 3. Implementation (TDD green phase)

- [ ] 3.1 Create `sleap_roots/circumnutation/_constants.py` with the constant set listed in spec Requirement: Module-level constants. Include `_SCHEMA_VERSION = 1`, `_CONSTANTS_VERSION = 1`. Provide a `ConstantsT` frozen `attrs` class enumerating every overridable parameter with the documented default; this is the typed override-bag.
- [ ] 3.2 Create `sleap_roots/circumnutation/_types.py` with `CircumnutationInputs` (attrs class, no `px_per_mm` parameter) and `ROW_IDENTITY_COLUMNS = ("series", "sample_uid", "timepoint", "plate_id", "plant_id", "track_id", "genotype", "treatment")`. Implement validators for cadence_s, R_px, trajectory_df row-identity columns + non-emptiness.
- [ ] 3.3 Create `sleap_roots/circumnutation/units.py` with `convert_to_mm(traits_df, units, px_per_mm)` per spec Requirement: convert_to_mm utility. Pure function. Validate `px_per_mm`. Use `pathlib`-friendly types where any path-like input would be involved (none here, but sets the convention).
- [ ] 3.4 Create `sleap_roots/circumnutation/_io.py` with:
  - module-level `logger = logging.getLogger(__name__)`
  - `write_per_plant_csv(out_path, df, units, run_metadata)` — writes the CSV + units sidecar + run-metadata sidecar atomically; uses `pathlib.Path` and `.as_posix()` at the I/O boundary; UTF-8 encoding for JSON
  - `read_per_plant_csv(in_path) -> tuple[pd.DataFrame, dict, dict]` — companion loader
  - `write_units_sidecar(out_path, units)` and `read_units_sidecar(in_path)` — explicit JSON helpers, UTF-8
  - `write_run_metadata(out_path, metadata)` and `read_run_metadata(in_path)` — companion helpers
  - `gather_run_metadata(input_path, run_id, constants_snapshot) -> dict` — fills the required fields automatically (git SHA via `subprocess`, versions via `importlib.metadata`, ISO timestamp via `datetime.now(timezone.utc).isoformat()`)
- [ ] 3.5 Create `sleap_roots/circumnutation/__init__.py` re-exporting `CircumnutationInputs`, `ROW_IDENTITY_COLUMNS`, `convert_to_mm`. Module docstring.
- [ ] 3.6 Create the 10 stub modules atomically (all in one commit so test 2.2 doesn't fail partial). Each contains:
  - `import logging; logger = logging.getLogger(__name__)`
  - module docstring naming the PR # that fills it in
  - the canonical callable from the spec table, with full Google-style docstring (Args, Returns, Raises) and body that raises `NotImplementedError(f"PR #{N} — see docs/circumnutation/roadmap.md")`
- [ ] 3.7 Edit `sleap_roots/__init__.py` to add `from sleap_roots.circumnutation import CircumnutationInputs` and `from sleap_roots.circumnutation.units import convert_to_mm`. Match the existing flat re-export convention; no `__all__` to update.
- [ ] 3.8 Add changelog entry to `docs/changelog.md` (lowercase per repo convention).

## 4. Verify

- [ ] 4.1 Run `uv run pytest tests/test_circumnutation_foundation.py -v` — all green.
- [ ] 4.2 Run `uv run pytest tests/ -x` — full suite green (no regression in `test_tracked_tip_pipeline.py`, `test_series.py`, `test_trait_pipelines.py`, etc.).
- [ ] 4.3 Run `uv run pytest tests/test_circumnutation_foundation.py --cov=sleap_roots.circumnutation --cov-report=term-missing` — coverage on the new sub-package ≥ 95%.
- [ ] 4.4 Run `uv lock --check` (or `uv sync --frozen` after a fresh `uv lock`) — confirm `uv.lock` is current.
- [ ] 4.5 Run `uv run black --check sleap_roots tests` — passes.
- [ ] 4.6 Run `uv run pydocstyle --convention=google sleap_roots/` — passes (full sleap_roots/ scope, not just `sleap_roots/circumnutation`, to catch `__init__.py` drift).
- [ ] 4.7 Run `uv run mkdocs build` — passes; rendered API reference includes the 10 stub modules with their stub-callable docstrings.
- [ ] 4.8 Run `openspec validate add-circumnutation-foundation --strict` — valid.
- [ ] 4.9 Open PR linking GitHub epic issue + `add-circumnutation-foundation` change-id.

## 5. Post-merge

- [ ] 5.1 Update `docs/circumnutation/roadmap.md` PR #1 status to ✅; fill in the GitHub issue and PR numbers in the table.
- [ ] 5.2 Run `openspec archive add-circumnutation-foundation --yes` to fold the foundational requirements into `openspec/specs/circumnutation/spec.md`.
