# Tasks: add-circumnutation-pipeline (PR #14)

TDD throughout: write the RED test first, watch it fail for the right reason, implement to GREEN,
refactor. Group commits so the first non-raising `pipeline` commit is **atomic** with the
foundation stub→impl migration (Task 3) — otherwise the foundation suite goes red.

## 1. Branch + issue + scaffolding
- [ ] 1.1 Confirm on branch `add-circumnutation-pipeline` (already created off `main`).
- [ ] 1.2 Draft the PR #14 tracking issue to the vault (parent #197; labels
  enhancement/circumnutation/multi-pr); post per the lazy-issue workflow after per-item OK. Capture
  the issue number for `Closes #N`.
- [ ] 1.3 `npx openspec validate add-circumnutation-pipeline --strict` passes on the proposal/specs.

## 2. Additive prerequisites on merged tiers (no behavior change; existing tests stay green)
- [ ] 2.1 **RED** `tests/test_circumnutation_pipeline.py`: assert `nutation._NUTATION_TRAIT_UNITS`
  exists, has one entry per `_NUTATION_TRAIT_COLUMNS` (8), every value in
  `PIPELINE_UNIT_VOCABULARY`.
- [ ] 2.2 **GREEN** Add `_NUTATION_TRAIT_UNITS` to `nutation.py` (co-located with
  `_NUTATION_TRAIT_COLUMNS`); units follow each column's semantics
  (`T_nutation_median`→`s`, `is_nutating`→`bool`, ratios/dimensionless→`—`, `_px`→`px`, …) — derive
  exactly from the column definitions; all values in `PIPELINE_UNIT_VOCABULARY`.
- [ ] 2.3 **RED** assert `psi_g._PSIG_TRAIT_UNITS` exists, one entry per `_PSIG_TRAIT_COLUMNS` (4),
  all in vocabulary.
- [ ] 2.4 **GREEN** Add `_PSIG_TRAIT_UNITS` to `psi_g.py` (e.g. `T_psig_median_s`→`s`,
  `delta_E_amplitude_proxy_px_per_frame`→`px/frame`, `helix_signed_area_px2`→`px²`,
  `handedness`→`—`); all values in vocabulary (verify `px²` is in `PIPELINE_UNIT_VOCABULARY`).
- [ ] 2.5 Confirm `uv run pytest tests/test_circumnutation_nutation.py tests/test_circumnutation_psi_g.py -q`
  still green (additions are purely new constants).

## 3. Dedup fast path on `traveling_wave.compute` (additive; standalone byte-identical)
- [ ] 3.1 **RED** dedup-equivalence test: `traveling_wave.compute(df, 300.0)` (recompute) vs
  `traveling_wave.compute(df, 300.0, tier0_df=kinematics.compute(df), tier1_df=nutation.compute(df, 300.0))`
  produce identical 6 Tier 3c columns (`atol=0`) on a multi-track synthetic.
- [ ] 3.2 **RED** `compute(df, 300.0, tier0_df=...)` with only ONE of the two kwargs → `ValueError`
  (both-or-neither); a `tier0_df` missing `v_total_median_px_per_frame` (or `tier1_df` missing
  `T_nutation_median`, or either missing `_IDENTITY_5_TUPLE`) → `ValueError`.
- [ ] 3.3 **GREEN** Add keyword-only `tier0_df=None` / `tier1_df=None` to `compute`. When both
  given: validate each carries `_IDENTITY_5_TUPLE` + its operand column, then use them in place of
  the internal `kinematics.compute` / `nutation.compute` recompute (current ~lines 367–378). When
  both None: unchanged recompute path. Raise on exactly-one / missing-column.
- [ ] 3.4 Update the `compute` docstring `Note` to document the fast-path kwargs (replace the
  "Batch callers should route through the pipeline" prose).
- [ ] 3.5 Confirm `uv run pytest tests/test_circumnutation_traveling_wave.py -q` (37 tests + canary)
  fully green — standalone path byte-identical.

## 4. `CircumnutationPipeline` — pure `compute_traits` (RED → GREEN), composed schema
- [ ] 4.1 **RED** composed-schema test on a multi-track, ≥2-plate synthetic (overlapping `track_id`,
  float64 `track_id` to exercise the int64-coercion merge guard): `compute_traits(inputs)` returns a
  3-tuple; `per_plant_df` has exactly the **46 columns** in declared tier order (8 identity + Tier 0
  10 + QC 10 + Tier 1 8 + Tier 2 4 + Tier 3c 6); one row per 5-tuple; each tier's columns present;
  bool/string flag dtypes preserved.
- [ ] 4.2 **RED** `growth_axis_unreliable` appears exactly once (no `_x`/`_y`), equals both source
  tiers' values (cross-tier equality), and lives in the Tier 0 block.
- [ ] 4.3 **RED** `units_dict` covers all 46 columns and passes the writer's
  `PIPELINE_UNIT_VOCABULARY` membership; `growth_axis_unreliable` present once.
- [ ] 4.4 **GREEN** Implement `CircumnutationPipeline` (`attrs`, single `constants` field) with
  `compute_traits(self, inputs) -> (per_plant_df, trajectory_df, units_dict)`:
  - call `kinematics.compute`, `qc.compute`, `nutation.compute`, `psi_g.compute` once each (thread
    `cadence_s` to the latter two; pass resolved `constants` to all);
  - drop `growth_axis_unreliable` from the QC frame (Tier 0 owns it); assert equality first;
  - call `traveling_wave.compute(df, cadence_s, constants, tier0_df=tier0, tier1_df=tier1)` (dedup);
  - merge all five on `_IDENTITY_5_TUPLE` (`how="left"` onto `_build_per_plant_template_from_df`,
    int64 coercion-with-raise guard) in fixed tier order;
  - assemble `units_dict` = `ROW_IDENTITY_UNITS` (identity cols) ∪ the five tier `_*_TRAIT_UNITS`;
  - module-level tuples for tier order + the 46-column order; `logger.debug` once at start.
- [ ] 4.5 **GREEN** Module-level `compute_traits(inputs, constants=None)` wrapper =
  `CircumnutationPipeline(constants=constants).compute_traits(inputs)` (preserves the stub
  signature).

## 5. Foundation stub→impl migration (ATOMIC with the first non-raising `pipeline` commit)
- [ ] 5.1 In `tests/test_circumnutation_foundation.py`: remove `("pipeline", "compute_traits", 14)`
  from `STUB_MODULES` (~line 69); remove `("pipeline", "compute_traits")` from
  `STUBS_WITH_CONSTANTS_KWARG` (~line 880); add `("pipeline", "compute_traits")` to
  `IMPLEMENTATIONS_WITH_CONSTANTS_KWARG` (~line 887).
- [ ] 5.2 Add `pipeline` to the explicit `test_module_logger_is_namespaced` list (~lines 759–808,
  the impl modules listed explicitly because they are no longer in `STUB_MODULES`).
- [ ] 5.3 Add a dedicated `pipeline` branch in `test_implementation_accepts_constants_kwarg`
  (~line 915): build a valid `CircumnutationInputs` and call
  `compute_traits(inputs, constants=ConstantsT())` → assert a 3-tuple (the generic `else` path
  passes a bare `trajectory_df`, which `compute_traits` does not accept).
- [ ] 5.4 Add a Package-layout scope-note / counts assertion if the suite checks impl/stub counts
  (verify; mirror the PR #10 transition). Confirm the "2 remaining stubs" enumeration
  (`parametric`, `plotting`) is consistent.
- [ ] 5.5 Confirm `uv run pytest tests/test_circumnutation_foundation.py -q` green.

## 6. `save()` + provenance round-trip
- [ ] 6.1 **RED** `save(out_path, per_plant_df, units, *, input_path, run_id)` writes the CSV +
  `<stem>.units.json` + `run_metadata.json`; `_io.read_per_plant_csv` recovers the DataFrame, units,
  and run_metadata (provenance keys present: git SHA, sleap_roots/sleap_io/numpy/scipy/pandas/python
  versions, platform, ISO timestamp, `_schema_version=1`, `_constants_version=6`,
  `_constants_snapshot`).
- [ ] 6.2 **GREEN** Implement `save` delegating to `_io.gather_run_metadata(input_path, run_id,
  constants)` + `_io.write_per_plant_csv(out_path, per_plant_df, units, run_metadata)`. No I/O in
  `compute_traits`.

## 7. Picklability + determinism
- [ ] 7.1 **RED** `pickle.loads(pickle.dumps(CircumnutationPipeline()))` round-trips and the
  unpickled instance computes an identical `per_plant_df`.
- [ ] 7.2 **RED** two in-process `compute_traits` runs are bit-identical on the float columns
  (`atol=0`).
- [ ] 7.3 **GREEN** Ensure no unpicklable state (only `constants`); fix if any test reds.

## 8. Real plate-001 integration test
- [ ] 8.1 **RED/validation** Load `tests/data/circumnutation_nipponbare_plate_001/plate_001_greyscale.tracked_proofread.slp`
  via the proven inline pattern (`Series.load` → `get_tracked_tips()` → `track_id` int coercion →
  attach the 8 row-identity columns with plate-001 metadata) → `CircumnutationInputs(cadence_s=300.0)`.
  Run the full pipeline (compute → save to a tmp dir → read-back). Assert: 6 rows; the 46-column
  schema in order; all five tiers' columns present; `traveling_wave_residual` finite < 0.30 (the QPB
  band, matching the traveling_wave real-data test); cross-tier `growth_axis_unreliable` equality.
  Skip if the Git-LFS fixture is absent.

## 9. Docs + deviation discipline
- [ ] 9.1 Correct the `pipeline.py` module docstring: sequential merge-orchestrator (not a TraitDef
  DAG); list the composed tiers + the dedup; reference this design.
- [ ] 9.2 `docs/circumnutation/roadmap.md` row #14: ⬜→ in-progress; correct the "TraitDef DAG"
  claim → sequential merge-orchestrator (with the Why); note the Tier 0/1 dedup + the
  nutation/psi_g units-map additions (#222).
- [ ] 9.3 `docs/changelog.md`: PR #14 entry (pipeline composition; dedup; units-map additions).

## 10. Verification gates (all must pass before PR)
- [ ] 10.1 `uv run pytest tests/ -q` fully green; aim ≥90% coverage on `pipeline.py`.
- [ ] 10.2 `black --check sleap_roots tests`.
- [ ] 10.3 `pydocstyle --convention=google sleap_roots/circumnutation/` (read 2–3 existing
  docstrings first; pydocstyle checks Google-section structure, not house formatting).
- [ ] 10.4 `uv lock --check`; `uv run mkdocs build`.
- [ ] 10.5 `npx openspec validate add-circumnutation-pipeline --strict`.
- [ ] 10.6 `/review-pr` (pre-PR) findings reconciled; `/pre-merge`; CI matrix Ubuntu/Windows/macOS
  green; Copilot review (if credits) reconciled.
