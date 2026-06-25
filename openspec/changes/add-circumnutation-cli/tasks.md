# Tasks — add-circumnutation-cli (PR #17)

TDD throughout: every implementation task writes its failing tests first, then the
minimum code to pass. Coverage is CI-enforced (≥84% project; aim ≥90% new code);
locally run via `click.testing.CliRunner` without `--cov` (the local `--cov` path
hits the Windows numpy-reload bug). `black --check` + `pydocstyle
--convention=google sleap_roots/circumnutation/` must stay green.

## 1. Test scaffolding & synthetic-`.slp` fixture helper

- [x] 1.1 Add `tests/test_circumnutation_adapters.py` and `tests/test_circumnutation_cli.py` skeletons.
- [x] 1.2 **Test first:** write a unit test for a `_make_synthetic_tracked_slp(tmp_path, *, n_tracks=2, n_frames=64, noise_sigma_px=2.0)` helper asserting it produces a loadable `.slp` whose `Series.load(...).get_tracked_tips()` returns `n_tracks` tracks × `n_frames` frames.
- [x] 1.3 Implement the helper: `synthetic.generate_trajectory(..., n_frames=n_frames, noise_sigma_px=noise_sigma_px)` (verify these exact kwarg names against `synthetic.py`) → per-row `sio.Instance.from_numpy([[tip_x, tip_y]], skeleton=Skeleton([Node("tip")]), track=Track("track_<i>"))` → `sio.LabeledFrame(video, frame_idx, instances)` over a **PIL-built TIFF** `sio.Video.from_filename(...)` (the TIFF/Video is mandatory — `sio.save_slp` needs a `Video` per `LabeledFrame`) → `sio.Labels(...)` → `sio.save_slp(...)`. `n_frames ≥ 64` clears every tier the full pipeline runs: Tier 1 CWT (min 9), **Tier 2 ψ_g (min 24)**, and the Tier 3 midline/spatial chain — not just Tier 1. `noise_sigma_px > 0` (theory.md §8 mandates noise for Layer-1 validation; avoids degenerate QC/ridge).
- [x] 1.4 **Test first + impl:** add a helper variant (or option) that injects, at one `(track, frame)`, a `sio.PredictedInstance` shadowed by a user-corrected `sio.Instance`, and assert `get_tracked_tips` dedups to one row — so the predicted-vs-user-corrected dedup path is covered WITHOUT the LFS fixture (the plain-`Instance` helper never exercises it).
- [x] 1.5 Add a small **metadata-CSV fixture builder** (writes a temp `plant_qr_code,genotype,treatment,timepoint` CSV) — the adapter metadata-precedence/provenance tests (§2.1) and the CLI `--metadata-csv` tests (§3.1) depend on it. NOTE: `generate_trajectory` is keyword-only (`*`) — pass `n_frames=`/`noise_sigma_px=` as keywords.

## 2. `adapters.series_to_inputs` (TDD)

- [x] 2.1 **Tests first** (`test_circumnutation_adapters.py`), one **distinct test function** per spec scenario of Requirement: Series-to-CircumnutationInputs adapter:
  - returns a 2-tuple `(inputs, identity_provenance)`; `inputs.trajectory_df` has all 8 identity columns; `track_id`/`plant_id` integer, `plant_id == track_id`;
  - prefix-anchored `track_id` strip: `"track_0"`→`0`, a bare `"5"`→`5`;
  - non-integer track name (`"track_2a"`) → `ValueError` naming the offender; interior-`track_` name (`"track_track_1"`) also → `ValueError` (anchored strip leaves `"track_1"`, refuses to silently corrupt to `1` as a global `.replace` would);
  - metadata precedence — flag overrides a real CSV value + INFO override log (one test);
  - CSV value used when no flag, **blank CSV cell + no flag → NO spurious log** (a **separate** test from the next);
  - **neither CSV nor flag → `NaN`** (its own test);
  - `timepoint` from a numeric CSV cell (`0`) → string `"0"` via raw `get_metadata` (not `Series.timepoint`);
  - malformed `--metadata-csv` → clear `ValueError`, not a pandas traceback;
  - **identity provenance**: `identity_provenance["metadata_csv_path"]` resolved-absolute (or `None`); `identity_source` records `flag`/`metadata_csv`/`default` per field.
- [x] 2.2 Implement `sleap_roots/circumnutation/adapters.py::series_to_inputs(...) -> tuple[CircumnutationInputs, dict]` to pass 2.1: returns the inputs + the identity-provenance dict (`metadata_csv_path`, `metadata_csv_sha256` = `hashlib.sha256` of the CSV bytes via `Path.read_bytes`, and the total per-field `identity_source` with the closed label set `{"flag","metadata_csv","default","absent"}`). No `click` import. Google-style docstring.
- [x] 2.3 Add the adapter callability scenario test from the MODIFIED Package layout requirement (`series_to_inputs` on a valid Series returns the tuple, does not raise `NotImplementedError`).

## 2b. Provenance fields in `gather_run_metadata` (TDD)

- [x] 2b.1 **Test first** (extend the `gather_run_metadata` tests in `tests/test_circumnutation_foundation.py`): `gather_run_metadata(..., metadata_csv_path=<abs>, metadata_csv_sha256=<digest>, identity_source={...})` records all three keys; omitting them writes `null` (existing callers unaffected). Assert `identity_source` values are within the closed label set `{"flag","metadata_csv","default","absent"}` (no NaN). Maps to the MODIFIED Run-metadata sidecar scenarios. **Existing-test note:** the current `gather_run_metadata` tests assert keys via `issubset`/`in` (not exact-set), so the new keys do not break them — confirm and, if any exact-key assertion exists, update it here.
- [x] 2b.2 Implement: append `metadata_csv_path=None, metadata_csv_sha256=None, identity_source=None` optional kwargs (AFTER `R_px`) to `sleap_roots/circumnutation/_io.py::gather_run_metadata`, recorded in the returned dict (backward-compatible — existing `save()`/foundation callers, which pass nothing past `input_path` positionally, write `null`).

## 3. `cli.analyze` command (TDD)

- [x] 3.1 **Tests first** (`test_circumnutation_cli.py`), one per spec scenario of Requirement: Circumnutation analyze CLI, all via `CliRunner`:
  - `--help` exits 0; full pipeline happy path (with `--genotype`) writes the `per_plant/` + `per_genotype/` + `plots/` tree; per-plant row count == n_tracks;
  - missing `--cadence-s` / `--sample-uid` → exit 2; nonexistent `SLP_PATH` → exit 2;
  - genotype unresolved (NaN) + aggregation on → exit 1 `ClickException` naming the three flags, no output tree (genotype is filled uniformly per run, so "any NaN" == "all NaN" — there is no partial case in v1);
  - `--no-aggregate` (no genotype) → exit 0, `per_plant/` + `plots/`, no `per_genotype/`;
  - `--no-plots` → no `plots/`; **`--no-plots` + `--no-aggregate` together** → `per_plant/` only;
  - **non-positive `--cadence-s` (0/negative) → exit 1** clean `ClickException`, no traceback; **non-numeric `--cadence-s` → exit 2** (click parse), no traceback (two separate tests);
  - `--metadata-csv` populates `genotype`/`treatment`; `--genotype` override logged;
  - **re-run overwrite**: a second `analyze` into the same `-o` exits 0, overwrites in place (no `FileExistsError`, no appended rows);
  - **default `--output-dir`**: no `-o` → outputs under `./<stem>_circumnutation/` (stem = `Path.stem`, which strips only `.slp`);
  - **`-v` logging**: INFO override notice appears on `result.stderr` (click 8.3 separates stderr; `result.output` is the combined stream), and the result summary is NOT on stderr; at default verbosity no INFO progress lines appear on stderr;
  - top-level `<out>/run_metadata.json` exists and `plots/plots_metadata.json`'s `run_metadata_ref` (`../run_metadata.json`) resolves to it (the dangling-pointer regression guard);
  - `run_metadata.json` (top-level + both subdirs) records resolved-absolute `input_path`, `metadata_csv_path`, `metadata_csv_sha256`, identical `identity_source`, and identical `cadence_s`/`R_px`(`null` when `--r-px` omitted)/`run_id`/`timestamp` across all three;
  - **no-`--metadata-csv` run**: `metadata_csv_path == null`, `metadata_csv_sha256 == null`, `identity_source` total over six fields with no `"metadata_csv"` values;
  - CC-3: `--help` has no `--px-per-mm`, points to `convert_to_mm`, documents the y-down pixel coordinate convention and the `timepoint` string-label note.
- [x] 3.2 Implement `sleap_roots/circumnutation/cli.py`: `@click.group() circumnutation` + `@circumnutation.command() analyze` with the full option set. Lazy imports inside the command body (mirroring `viewer/cli.py`); module top-level stays import-light (only `click`) so `import sleap_roots.cli` is cheap and `import sleap_roots` stays green.
- [x] 3.3 Implement the orchestration body: resolve identity & `--output-dir`; `mkdir` `--output-dir` + `per_plant/` (+ `per_genotype/` when aggregating); `Series.load(csv_path=metadata_csv)` → `inputs, identity_provenance = series_to_inputs(...)`; genotype-missing hard error when aggregation on; `compute_traits`; assemble run-metadata **ONCE** via `gather_run_metadata(input_path=resolved_slp, ..., metadata_csv_path=identity_provenance["metadata_csv_path"], metadata_csv_sha256=identity_provenance["metadata_csv_sha256"], identity_source=identity_provenance["identity_source"])`; write it to the **top-level `<out>/run_metadata.json`** (so the plots sidecar's `../run_metadata.json` ref resolves); `write_per_plant_csv(per_plant/csv, per_plant_df, units, run_metadata)` (NOT `save()` — the CLI shares one provenance snapshot); aggregation path → `aggregate_by_genotype` → `write_per_genotype_csv(per_genotype/csv, ..., run_metadata)` (same dict); plotting path → `matplotlib.use("Agg", force=True)` → lazy import `plotting` → `save_plots(inputs, out_dir=<output-dir>, enabled=True)`.
- [x] 3.4 Implement the error contract: `try/except (ValueError, FileNotFoundError)` → `click.ClickException(str(e))`; no broad catch-all.
- [x] 3.5 Implement `_configure_logging(verbose)` (count → WARNING/INFO/DEBUG, stderr) and the final `click.echo` summary to stdout.

## 4. Register on the root CLI (TDD)

- [x] 4.1 **Test first:** assert `circumnutation` is a registered command on `sleap_roots.cli:main` and that `circumnutation analyze --help` exits 0 (the MODIFIED Package layout `cli` scenario).
- [x] 4.2 Add `from sleap_roots.circumnutation.cli import circumnutation` + `main.add_command(circumnutation)` to `sleap_roots/cli.py`.

## 5. Package-layout & foundation-convention scenarios (TDD)

- [x] 5.1 Cover `import sleap_roots.circumnutation.adapters` and `import sleap_roots.circumnutation.cli` (MODIFIED "All stub modules import cleanly" scenario now lists both). NOTE: there is no single parametrized "import every impl module" test in `tests/test_circumnutation_foundation.py` to extend — the per-module import+callability coverage lives in each module's own test file, so tasks 2.3 (adapters) and 4.1 (cli) already satisfy this scenario. Add an explicit top-level import assertion only if a foundation-level import loop is desired.
- [x] 5.2 Confirm the "Calling each remaining stub" test still asserts exactly 1 remaining stub (`parametric`) and the 14-impl/1-stub accounting holds.
- [x] 5.3 **Convention every prior PR followed:** append `"adapters"` and `"cli"` to the `test_module_logger_is_namespaced` parametrize list in `tests/test_circumnutation_foundation.py` (~lines 773–841) — both new modules must define a namespaced `logger = logging.getLogger(__name__)` (CC-9). Without this the logger-namespace convention silently skips them.

## 6. Real plate-001 end-to-end (skipif-guarded)

- [x] 6.1 **Test:** skipif-guarded on the LFS proofread fixture — `analyze <plate_001.slp> --cadence-s 300 --sample-uid plate_001 --metadata-csv <fixture_metadata.csv> -o <tmp>` exits 0; per-plant CSV carries `genotype=Nipponbare`, `treatment=MOCK`; full tree present; PNGs non-empty. No pixel baselines.

## 7. Docs & verification gates

- [x] 7.1 Ensure mkdocstrings picks up `adapters` / `cli` (Google-style docstrings on every public symbol).
- [x] 7.2 `uv run pytest tests/ -q` green; `black --check`; `pydocstyle --convention=google sleap_roots/circumnutation/`; `uv lock --check` (no-op — `click` already a dep); `uv run mkdocs build`.
- [x] 7.3 `npx openspec validate add-circumnutation-cli --strict` passes.
