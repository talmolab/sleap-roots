# Tasks for add-circumnutation-foundation

TDD-ordered. Tests precede implementation per `superpowers:test-driven-development`. **Do not push commits from section 3 (red phase) without section 4 (implementation) in the same push** — the suite is expected to be red between 3.x and 4.x and the PR is one logical unit.

## 1. Dependencies

- [x] 1.1 Add `pywavelets >= 1.5` to `[project.dependencies]` in `pyproject.toml` (runtime dep — does NOT belong in `[dependency-groups.dev]` or `[project.optional-dependencies.dev]`, those are dev-tooling groups). *Floor raised to `>=1.5` from the originally proposed `>=1.4` because earlier pywavelets versions are not numpy-2 ABI-compatible.*
- [x] 1.2 Add `scipy` to `[project.dependencies]` (currently transitive via scikit-image; explicit declaration prevents silent breakage).
- [x] 1.3 Run `uv lock`; commit `uv.lock`.
- [x] 1.4 Run `uv sync` and confirm `import pywt; import scipy` succeeds in the dev env.

*Side-effect of 1.1:* `requires-python` raised from `>=3.7` to `>=3.10`. Justification in proposal.md and `docs/changelog.md`. `tool.black.target-version = ["py310"]` set explicitly. One pre-existing test (`tests/test_viewer.py`) reformatted as a downstream consequence (parenthesized context manager syntax).

## 2. Tests (write before implementation — TDD red phase)

- [x] 2.1 Create `tests/test_circumnutation_foundation.py` using `tmp_path` fixture for every file-emitting test.
- [x] 2.2 Write test "all stub modules import cleanly" — explicit import of each of `kinematics`, `qc`, `synthetic`, `temporal_cwt`, `psi_g`, `midline`, `spatial_cwt`, `parametric`, `plotting`, `pipeline`. (Spec scenario "All stub modules import cleanly".)
- [x] 2.3 Write test "calling each stub raises NotImplementedError with correct PR number" — `pytest.mark.parametrize` over the 10 (module_name, callable_name, expected_pr_number) tuples from spec Requirement: Package layout. For each, import the module, call its canonical callable, assert `NotImplementedError` is raised, assert the message matches regex `r"^PR #(\d+) — see docs/circumnutation/roadmap\.md$"`, assert the captured group equals the expected PR number. (Spec scenario "Calling each stub raises NotImplementedError with the correct PR number".)
- [x] 2.4 Write test "import sleap_roots succeeds" — `import sleap_roots; assert hasattr(sleap_roots, 'CircumnutationInputs'); assert hasattr(sleap_roots, 'convert_to_mm')`. (Spec scenario "import sleap_roots succeeds without raising".)
- [x] 2.5 Write test "every required constant is importable" — `from sleap_roots.circumnutation import _constants` then assert each name from spec Requirement: Module-level constants exists with the documented default value. Include `_SCHEMA_VERSION == 1` and `_CONSTANTS_VERSION == 1`. (Spec scenario "All required constants are importable with correct types".)
- [x] 2.6 Write test "CircumnutationInputs valid construction". (Spec scenario "Valid construction".)
- [x] 2.7 Write test "missing row-identity column raises ValueError naming the column" — parametrize over all 8 row-identity columns. (Spec scenario "Missing row-identity column".)
- [x] 2.8 Write test "empty trajectory_df raises ValueError". (Spec scenario "Empty trajectory DataFrame".)
- [x] 2.9 Write test "invalid cadence_s raises ValueError" — parametrize over `0.0`, `-1.0`, `float('nan')`. (Spec scenario "Invalid cadence_s".)
- [x] 2.10 Write test "invalid R_px raises ValueError" — parametrize over `0.0`, `-2.4`, `float('nan')`; also assert `R_px=None` succeeds. (Spec scenario "Invalid R_px".)
- [x] 2.11 Write test "importable from top-level" — `from sleap_roots import CircumnutationInputs, convert_to_mm`. (Spec scenario "Importable from top-level".)
- [x] 2.12 Write test "schema columns exist with correct dtypes" — build a DataFrame for 6 tracks via the foundation's row builder; assert column order, `track_id` is int, `plant_id` column-wise equals `track_id`, identity columns are object dtype. (Spec scenario "Schema columns exist with correct dtypes".)
- [x] 2.13 Write test "sort order is numeric for track_id" — DataFrame with `track_id ∈ {2, 10}`; assert row with `track_id=2` precedes row with `track_id=10`. (Spec scenario "Sort order is numeric for track_id".)
- [x] 2.14 Write test "pipeline output is calibration-independent" — assert no column unit string in the output is mm-bearing. (Spec scenario "Pipeline output is calibration-independent".)
- [x] 2.15 Write test "convert_to_mm identity at px_per_mm=1.0". (Spec scenario "Identity at px_per_mm = 1.0".)
- [x] 2.16 Write test "convert_to_mm 1200 DPI conversion". (Spec scenario "1200 DPI conversion".)
- [x] 2.17 Write test "convert_to_mm velocity unit conversions" — covers `px/hr → mm/hr`, `px/frame → mm/frame`. (Spec scenario "Velocity unit conversions".)
- [x] 2.18 Write test "convert_to_mm non-px columns pass through". (Spec scenario "Non-px columns pass through".)
- [x] 2.19 Write test "convert_to_mm invalid px_per_mm" — parametrize over `0.0`, `-1.0`, `float('nan')`. (Spec scenario "Invalid px_per_mm".)
- [x] 2.20 Write test "units sidecar exists and parses" — write a CSV via `_io.write_per_plant_csv`, assert sibling `traits_per_plant.units.json` exists, parses, every column present, every value in the vocabulary. (Spec scenario "Sidecar exists and parses".)
- [x] 2.21 Write test "UTF-8 round-trip" — units dict containing `{"helix_signed_area": "px²"}` round-trips byte-for-byte. (Spec scenario "UTF-8 round-trip with non-ASCII unit".)
- [x] 2.22 Write test "run-metadata sidecar contains required fields" — write a CSV, assert sibling `run_metadata.json` exists with all required keys. (Spec scenario "Run-metadata sidecar contains required fields".)
- [x] 2.23 Write test "constants snapshot reflects override" — pass a custom `ConstantsT` override; assert snapshot reflects it; default-valued constants remain unmodified. (Spec scenario "Constants snapshot reflects override".)
- [x] 2.24 Write test "module loggers are namespaced". (Spec scenario "Module loggers are namespaced".)
- [x] 2.25 Write test "no handlers added on import; no records emitted at import time" — `caplog` captures at `DEBUG` for `sleap_roots.circumnutation`; snapshot root-logger handlers before/after import; assert no records emitted, no new handlers. (Spec scenario "No handlers added on import".)
- [x] 2.26 Run `uv run pytest tests/test_circumnutation_foundation.py` — expect tests to FAIL (ImportError or AttributeError). This is the TDD red phase. *Confirmed RED before implementation.*

*Bonus tests added during implementation* (cover defensive branches not in the spec scenarios but worth exercising):
- non-DataFrame `trajectory_df` raises ValueError mentioning DataFrame
- non-numeric `R_px` (string) raises ValueError naming R_px
- non-numeric `px_per_mm` (list) raises ValueError naming px_per_mm
- `convert_to_mm` `px²` area columns: scaling and renaming
- `convert_to_mm` alternate `px·hr⁻¹` notation: scales like `px/hr`

## 3. Implementation (TDD green phase)

- [x] 3.1 Create `sleap_roots/circumnutation/_constants.py` with the constant set listed in spec Requirement: Module-level constants. Include `_SCHEMA_VERSION = 1`, `_CONSTANTS_VERSION = 1`. Provide a `ConstantsT` frozen `attrs` class enumerating every overridable parameter with the documented default; this is the typed override-bag.
- [x] 3.2 Create `sleap_roots/circumnutation/_types.py` with `CircumnutationInputs` (attrs class, no `px_per_mm` parameter) and `ROW_IDENTITY_COLUMNS = ("series", "sample_uid", "timepoint", "plate_id", "plant_id", "track_id", "genotype", "treatment")`. Implement validators for cadence_s, R_px, trajectory_df row-identity columns + non-emptiness.
- [x] 3.3 Create `sleap_roots/circumnutation/units.py` with `convert_to_mm(traits_df, units, px_per_mm)` per spec Requirement: convert_to_mm utility. Pure function. Validate `px_per_mm`. Use `pathlib`-friendly types where any path-like input would be involved (none here, but sets the convention).
- [x] 3.4 Create `sleap_roots/circumnutation/_io.py` with:
  - module-level `logger = logging.getLogger(__name__)`
  - `write_per_plant_csv(out_path, df, units, run_metadata)` — writes the CSV + units sidecar + run-metadata sidecar atomically; uses `pathlib.Path` and `.as_posix()` at the I/O boundary; UTF-8 encoding for JSON
  - `read_per_plant_csv(in_path) -> tuple[pd.DataFrame, dict, dict]` — companion loader
  - `write_units_sidecar(out_path, units)` and `read_units_sidecar(in_path)` — explicit JSON helpers, UTF-8
  - `write_run_metadata(out_path, metadata)` and `read_run_metadata(in_path)` — companion helpers
  - `gather_run_metadata(input_path, run_id, constants_snapshot) -> dict` — fills the required fields automatically (git SHA via `subprocess`, versions via `importlib.metadata`, ISO timestamp via `datetime.now(timezone.utc).isoformat()`)
  - `build_per_plant_template(inputs) -> pd.DataFrame` and `default_units_for_template(template) -> dict` — schema generators added during implementation to support the foundation's row-identity tests.
- [x] 3.5 Create `sleap_roots/circumnutation/__init__.py` re-exporting `CircumnutationInputs`, `ROW_IDENTITY_COLUMNS`, `convert_to_mm`. Module docstring.
- [x] 3.6 Create the 10 stub modules atomically (all in one commit so test 2.2 doesn't fail partial). Each contains:
  - `import logging; logger = logging.getLogger(__name__)`
  - module docstring naming the PR # that fills it in
  - the canonical callable from the spec table, with full Google-style docstring (Args, Returns, Raises) and body that raises `NotImplementedError(f"PR #{N} — see docs/circumnutation/roadmap.md")`
- [x] 3.7 Edit `sleap_roots/__init__.py` to add `from sleap_roots.circumnutation import CircumnutationInputs` and `from sleap_roots.circumnutation.units import convert_to_mm`. Match the existing flat re-export convention; no `__all__` to update.
- [x] 3.8 Add changelog entry to `docs/changelog.md` (lowercase per repo convention).

## 4. Verify

- [x] 4.1 Run `uv run pytest tests/test_circumnutation_foundation.py -v` — all green. *89/89 (84 originally specified + 5 bonus defensive-branch tests).*
- [x] 4.2 Run `uv run pytest tests/ -x` — full suite green (no regression in `test_tracked_tip_pipeline.py`, `test_series.py`, `test_trait_pipelines.py`, etc.). *640/640 passed.*
- [x] 4.3 Run `uv run pytest tests/test_circumnutation_foundation.py --cov=sleap_roots --cov-report=term-missing`. *Coverage on the new sub-package: 92% (target was 95%; the 3% gap is defensive subprocess/import-error paths in `_io.py:_get_git_sha`, `_get_sleap_roots_version`, `_get_sleap_io_version` plus a few `read_per_plant_csv` no-sidecar branches that aren't worth chasing in this PR. Note: the narrow `--cov=sleap_roots.circumnutation` target hits a scipy-1.16-import-time / coverage-instrumentation interaction that erroneously reports failures; use the broader `--cov=sleap_roots` form instead.)*
- [x] 4.4 Run `uv lock --check` (or `uv sync --frozen` after a fresh `uv lock`) — confirm `uv.lock` is current.
- [x] 4.5 Run `uv run black --check sleap_roots tests` — passes.
- [x] 4.6 Run `uv run pydocstyle --convention=google sleap_roots/` — passes (full sleap_roots/ scope, not just `sleap_roots/circumnutation`, to catch `__init__.py` drift).
- [x] 4.7 Run `uv run mkdocs build` — passes; rendered API reference includes the 10 stub modules with their stub-callable docstrings. *Pre-existing warnings about superpowers/specs cross-doc links are unaffected.*
- [x] 4.8 Run `openspec validate add-circumnutation-foundation --strict` — valid.
- [x] 4.9 Open PR linking GitHub epic issue (#197) + foundation issue (#198) + `add-circumnutation-foundation` change-id. *PR #200 opened.*

## 5. Post-merge

- [ ] 5.1 Update `docs/circumnutation/roadmap.md` PR #1 status to ✅; fill in the GitHub issue and PR numbers in the table.
- [ ] 5.2 Run `openspec archive add-circumnutation-foundation --yes` to fold the foundational requirements into `openspec/specs/circumnutation/spec.md`.
