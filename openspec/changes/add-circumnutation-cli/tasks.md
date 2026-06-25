# Tasks — add-circumnutation-cli (PR #17)

TDD throughout: every implementation task writes its failing tests first, then the
minimum code to pass. Coverage is CI-enforced (≥84% project; aim ≥90% new code);
locally run via `click.testing.CliRunner` without `--cov` (the local `--cov` path
hits the Windows numpy-reload bug). `black --check` + `pydocstyle
--convention=google sleap_roots/circumnutation/` must stay green.

## 1. Test scaffolding & synthetic-`.slp` fixture helper

- [ ] 1.1 Add `tests/test_circumnutation_adapters.py` and `tests/test_circumnutation_cli.py` skeletons.
- [ ] 1.2 **Test first:** write a unit test for a `_make_synthetic_tracked_slp(tmp_path, *, n_tracks=2, n_frames=64, noise_sigma_px>0)` helper asserting it produces a loadable `.slp` whose `Series.load(...).get_tracked_tips()` returns `n_tracks` tracks × `n_frames` frames.
- [ ] 1.3 Implement the helper: `synthetic.generate_trajectory(...)` → per-row `sio.Instance.from_numpy([[tip_x, tip_y]], skeleton=Skeleton([Node("tip")]), track=Track("track_<i>"))` → `sio.LabeledFrame(video, frame_idx, instances)` over a PIL-built TIFF `sio.Video.from_filename(...)` → `sio.Labels(...)` → `sio.save_slp(...)`. `n_frames ≥ 64`, `noise_sigma_px > 0` so the Tier-1 CWT (min 9 frames) runs non-degenerately.

## 2. `adapters.series_to_inputs` (TDD)

- [ ] 2.1 **Tests first** (`test_circumnutation_adapters.py`), one per spec scenario of Requirement: Series-to-CircumnutationInputs adapter:
  - builds `CircumnutationInputs` with all 8 identity columns; `track_id`/`plant_id` integer, `plant_id == track_id`;
  - prefix-anchored `track_id` strip (incl. interior-`track_` name `"track_track_1"` → `1`);
  - non-integer track name (`"track_2a"`) → `ValueError` naming the offender;
  - metadata precedence: flag overrides a real CSV value + INFO override log; CSV value used when no flag; blank CSV cell + no flag → no spurious log; neither → `NaN`;
  - `timepoint` from a numeric CSV cell (`0`) → string `"0"` via raw `get_metadata` (not `Series.timepoint`);
  - malformed `--metadata-csv` → clear `ValueError`, not a pandas traceback.
- [ ] 2.2 Implement `sleap_roots/circumnutation/adapters.py::series_to_inputs(series, *, cadence_s, sample_uid, series_name=None, timepoint=None, plate_id=None, genotype=None, treatment=None, r_px=None, run_id=None)` to pass 2.1. No `click` import. Google-style docstring.
- [ ] 2.3 Add the adapter callability scenario test from the MODIFIED Package layout requirement (`series_to_inputs` on a valid Series does not raise `NotImplementedError`).

## 3. `cli.analyze` command (TDD)

- [ ] 3.1 **Tests first** (`test_circumnutation_cli.py`), one per spec scenario of Requirement: Circumnutation analyze CLI, all via `CliRunner`:
  - `--help` exits 0; full pipeline happy path (with `--genotype`) writes the `per_plant/` + `per_genotype/` + `plots/` tree; per-plant row count == n_tracks;
  - missing `--cadence-s` / `--sample-uid` → exit 2; nonexistent `SLP_PATH` → exit 2;
  - genotype unresolved + aggregation on → exit 1 `ClickException` naming the three flags, no output tree;
  - `--no-aggregate` (no genotype) → exit 0, `per_plant/` + `plots/`, no `per_genotype/`;
  - `--no-plots` → no `plots/`;
  - bad `--cadence-s` (0 / negative / non-numeric) → non-zero exit, clean message, no traceback;
  - `--metadata-csv` populates `genotype`/`treatment`; `--genotype` override logged;
  - `run_metadata.json` (both subdirs) records resolved-absolute `input_path` + identical `cadence_s`/`R_px`/`run_id`;
  - CC-3: `--help` has no `--px-per-mm` and points to `convert_to_mm`.
- [ ] 3.2 Implement `sleap_roots/circumnutation/cli.py`: `@click.group() circumnutation` + `@circumnutation.command() analyze` with the full option set. Lazy imports inside the command body (mirroring `viewer/cli.py`).
- [ ] 3.3 Implement the orchestration body: resolve identity & `--output-dir`; `mkdir` `per_plant/` (+ `per_genotype/` when aggregating); `Series.load(csv_path=metadata_csv)` → `series_to_inputs(...)`; genotype-missing hard error when aggregation on; `compute_traits` → `save(..., input_path=resolved_slp)`; aggregation path → `aggregate_by_genotype` → `gather_run_metadata` → `write_per_genotype_csv`; plotting path → `matplotlib.use("Agg", force=True)` → lazy import `plotting` → `save_plots(..., enabled=True)`.
- [ ] 3.4 Implement the error contract: `try/except (ValueError, FileNotFoundError)` → `click.ClickException(str(e))`; no broad catch-all.
- [ ] 3.5 Implement `_configure_logging(verbose)` (count → WARNING/INFO/DEBUG, stderr) and the final `click.echo` summary to stdout.

## 4. Register on the root CLI (TDD)

- [ ] 4.1 **Test first:** assert `circumnutation` is a registered command on `sleap_roots.cli:main` and that `circumnutation analyze --help` exits 0 (the MODIFIED Package layout `cli` scenario).
- [ ] 4.2 Add `from sleap_roots.circumnutation.cli import circumnutation` + `main.add_command(circumnutation)` to `sleap_roots/cli.py`.

## 5. Package-layout scenarios (TDD)

- [ ] 5.1 Extend the package-import test so `import sleap_roots.circumnutation.adapters` and `import sleap_roots.circumnutation.cli` are covered (MODIFIED "All stub modules import cleanly" scenario now lists both).
- [ ] 5.2 Confirm the "Calling each remaining stub" test still asserts exactly 1 remaining stub (`parametric`) and the 14-impl/1-stub accounting holds.

## 6. Real plate-001 end-to-end (skipif-guarded)

- [ ] 6.1 **Test:** skipif-guarded on the LFS proofread fixture — `analyze <plate_001.slp> --cadence-s 300 --sample-uid plate_001 --metadata-csv <fixture_metadata.csv> -o <tmp>` exits 0; per-plant CSV carries `genotype=Nipponbare`, `treatment=MOCK`; full tree present; PNGs non-empty. No pixel baselines.

## 7. Docs & verification gates

- [ ] 7.1 Ensure mkdocstrings picks up `adapters` / `cli` (Google-style docstrings on every public symbol).
- [ ] 7.2 `uv run pytest tests/ -q` green; `black --check`; `pydocstyle --convention=google sleap_roots/circumnutation/`; `uv lock --check` (no-op — `click` already a dep); `uv run mkdocs build`.
- [ ] 7.3 `npx openspec validate add-circumnutation-cli --strict` passes.
