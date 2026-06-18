# Tasks — add-circumnutation-per-genotype-aggregation (PR #15)

Each task is RED → GREEN: write the failing test(s) first, then the minimal
implementation to pass. Run the FULL suite (`uv run pytest tests/ -q`) before any
foundation/atomic commit — cross-module reload pollution (the PR #14
PicklingError / isinstance lessons) only surfaces in full-suite runs. Any test
that imports `pipeline` / `ConstantsT` to build a real `per_plant_df` SHALL use
the documented function-local import pattern (as `tests/test_circumnutation_pipeline.py`
already does), not a module-level import.

New aggregation + per-genotype-writer tests live in a NEW
`tests/test_circumnutation_aggregation.py`. The package-layout / logger / writer
guard tests live in the EXISTING `tests/test_circumnutation_foundation.py` and
`tests/test_circumnutation_pipeline.py` (there is no `test_circumnutation_io.py`
or `test_circumnutation_package.py`).

## 0. Follow-up issues (file before the PR, per the lazy-issue workflow)
- [ ] 0.1 Draft + (after user OK) file the issue for the two-level `plant_id`
  collapse generalization of `aggregate_by_genotype` (so it survives a future
  track↔plant divergence instead of raising). Referenced by stable title in
  `aggregation.py`'s 1:1-guard error message + docstring.
- [ ] 0.2 Draft + (after user OK) file the issue for a stem-prefixed
  `run_metadata.json` name to remove the `_io` fixed-name clobber (PR #14
  follow-up b). Referenced by stable title in the `write_per_genotype_csv` docstring.
- [ ] 0.3 Draft + (after user OK) file the issue for a circular-statistics
  per-genotype summary of `principal_axis_angle` (circular mean + resultant
  length); PR #15 excludes it from linear aggregation. Referenced by stable title
  in the `aggregation.py` docstring next to the exclusion.
- [x] 0.4 Until 0.1–0.3 are filed, error messages / docstrings SHALL reference the
  follow-up by stable title (not a hard-coded `#N`), so a deferred filing leaves no
  dangling issue number in shipped code.

## 1. Branch + module scaffolding + package-layout wiring
- [x] 1.1 Confirm on branch `eberrigan/circumnutation-per-genotype-aggregation`.
- [x] 1.2 (TEST, in `tests/test_circumnutation_foundation.py`) Add `"aggregation"`
  to the `test_module_logger_is_namespaced` parametrize list (new impl module —
  NOT a stub; mirrors how `nutation`/`pipeline` were added). Add an import-cleanly
  assertion for `sleap_roots.circumnutation.aggregation`. Do NOT add
  `aggregate_by_genotype` to `IMPLEMENTATIONS_WITH_CONSTANTS_KWARG` (it has no
  `constants=` parameter). (RED)
- [x] 1.3 (TEST, in `tests/test_circumnutation_aggregation.py`) Package-layout
  callability: `aggregate_by_genotype(per_plant_df, units)` returns a 2-tuple
  `(per_genotype_df, dict)` on a valid composed frame without raising. (RED)
- [x] 1.4 Create `sleap_roots/circumnutation/aggregation.py` with the module
  logger (`logger = logging.getLogger(__name__)`, CC-9), the name-tuple
  structural constants (`_AGG_IDENTITY_DROP`, `_AGG_IDENTITY_KEEP`,
  `_AGG_SPECIAL_COLUMNS` — the latter includes `is_nutating`, `handedness`,
  `track_is_clean`, `qc_failure_reason`, `growth_axis_unreliable`,
  `principal_axis_angle`, `helix_signed_area_px2`), and the `aggregate_by_genotype` signature with a
  complete Google-style docstring (no `constants=` parameter). The docstring SHALL
  document: (a) per-trait NaN-skip means per-trait finite n ≤ `n_plants_passing_qc`
  and a NaN `_iqr` beside a finite `_median` means < 2 finite values, not zero
  spread; (b) `frac_nutating`'s denominator (all passing) differs from a per-trait
  median's (finite-only); (c) `_iqr` of an already-dispersion source trait is a
  spread-of-spreads; (d) `principal_axis_angle` is excluded as a wrapping circular
  angle (circular summary deferred — §0.3); (e) the excluded-plant audit recipe
  (re-filter the per-plant CSV by `track_is_clean == False` within the group);
  (f) `handedness_consensus_frac` ≈ 0.5 means the mode is a tie-break artifact;
  (g) `helix_signed_area_px2` is aggregated by magnitude (chirality direction is in
  `handedness_mode`/`consensus_frac`), and the other signed traits
  (`v_long_signed*`, `v_lat_signed*`, `period_residual_vs_derr_reference`) are
  aggregated signed because their sign is per-plant-consistent. (GREEN for 1.2–1.3)

## 2. Input validation (units coverage + 1:1 guard)
- [x] 2.1 (TEST) A `units` mapping that does not 1:1 cover `per_plant_df` columns
  raises `ValueError` naming the offending column, before any output. (RED)
- [x] 2.2 (TEST) The track↔plant 1:1 guard: a `per_plant_df` where one
  `(plate_id, genotype, treatment, plant_id)` maps to ≥2 rows raises `ValueError`
  naming the offending `plant_id`. (RED)
- [x] 2.3 (TEST) No-mutation: `per_plant_df` and `units` are equal to pre-call
  copies after `aggregate_by_genotype` returns. (RED)
- [x] 2.4 Implement input validation (reuse `_validate_units_coverage` from §9 for
  units; a duplicate-`plant_id` check for the 1:1 guard, error message referencing
  the §0.1 follow-up by stable title — not a hard-coded `#N` until filed) and
  operate on a defensive copy (no mutation of inputs). (GREEN for 2.1–2.3)

## 3. Float-trait classification + median/IQR (units-driven)
- [x] 3.1 (TEST) Units-driven float detection: a column is a float trait iff its
  `units` value ∉ {`int`, `bool`, `string`}, it is not a row-identity column, AND
  it is not in `_AGG_SPECIAL_COLUMNS` (so `principal_axis_angle`, despite unit
  `rad`, is NOT a float trait). Assert the derived directly-aggregated float set
  on the known per-plant schema is the 31 numeric-unit traits EXCLUDING both
  `principal_axis_angle` and `helix_signed_area_px2` (the latter handled as a
  magnitude in §4), and INCLUDING `angular_amplitude`. (RED)
- [x] 3.2 (TEST) `<trait>_median` = NaN-skipping median across passing plants;
  `<trait>_iqr` = Q75 − Q25 via `scipy.stats.iqr(nan_policy="omit",
  interpolation="linear")`. Hand-built frame with known values. (RED)
- [x] 3.3 (TEST) IQR = NaN when < 2 finite passing values (n=1 plant case); median
  still emitted at n = 1. (RED)
- [x] 3.4 (TEST) IQR = NaN when a multi-plant group has only 1 *finite* value for a
  trait (others NaN); the distinct code path from 3.3. (RED)
- [x] 3.5 (TEST) scipy guard: assert `scipy.stats.iqr([5.0, float("nan")],
  nan_policy="omit") == 0.0` — the behavior `_iqr_or_nan` depends on — so a scipy
  upgrade that changes NaN/interpolation semantics fails loud. (RED→GREEN)
- [x] 3.6 Implement the `_iqr_or_nan` wrapper (module-qualified `scipy.stats.iqr`;
  `< 2` finite → NaN) and the float median/IQR emission. (GREEN for 3.2–3.4)

## 4. Special-column aggregation
- [x] 4.1 (TEST) `frac_nutating` = mean of `is_nutating` over passing plants;
  NaN when `n_plants_passing_qc == 0`; float64 dtype. (RED)
- [x] 4.2 (TEST) `handedness_mode` from explicit value counts (clear majority
  `[1,1,1,-1]` → 1; consensus_frac 0.75). (RED)
- [x] 4.3 (TEST) `handedness_mode` tie-break: `[1,-1]` → mode −1 (smallest signed
  among tied), consensus_frac 0.5; `[0,0,1,1]` → mode 0. (RED)
- [x] 4.4 (TEST) `growth_axis_unreliable` AND `principal_axis_angle` are neither
  aggregated nor emitted (no `*_median`/`*_iqr` for either); `angular_amplitude`
  IS emitted; the identity sub-columns
  `series`/`sample_uid`/`timepoint`/`plant_id`/`track_id` are dropped. (RED)
- [x] 4.4b (TEST) `helix_signed_area_px2` is aggregated as a magnitude: a group
  with mixed-sign large values (e.g. `[+1000.0, -1000.0]`) yields
  `helix_signed_area_abs_px2_median` ≈ 1000.0 (NOT ≈ 0); no
  `helix_signed_area_px2_median`/`_iqr` column; unit `px²`. (RED)
- [x] 4.5 Implement `frac_nutating`, the explicit-count `handedness_mode` +
  consensus (NOT `Series.mode`), the growth-axis + `principal_axis_angle` drop,
  the `helix_signed_area_px2` → `|·|` magnitude (`helix_signed_area_abs_px2_*`),
  identity-column dropping. Build output rows as a list-of-per-group dicts →
  `pd.DataFrame(rows)` (the `qc.py` pattern, NOT `groupby().apply(Series)`) to
  preserve int64 dtypes; wrap degenerate-group `np.nanmedian` / `scipy.stats.iqr`
  in `np.errstate` / `warnings.catch_warnings` to suppress all-NaN warnings
  (house style). (GREEN for 4.1–4.4b)

## 5. Grouping + QC exclusion + counts + reasons
- [x] 5.1 (TEST) Group by `(plate_id, genotype, treatment)`; rows sorted by the
  key; plates not pooled (two plates of one genotype → two rows). (RED)
- [x] 5.1b (TEST) NaN group key preserved: a plant with NaN `treatment` forms its
  own group (not silently dropped); `Σ(n_plants_passing_qc + n_plants_excluded)`
  over all rows equals `len(per_plant_df)`. Implementation uses
  `groupby(..., dropna=False)`. (RED)
- [x] 5.2 (TEST) Exclusion gate: `track_is_clean == False` plants excluded from
  all stats; `n_plants_passing_qc` / `n_plants_excluded` int counts;
  `n_plants_passing_qc + n_plants_excluded == group size`. (RED)
- [x] 5.3 (TEST) `exclusion_reasons` clause→count string, `_FAILURE_CLAUSE_ORDER`
  order, `"; "`-joined, `""` when none; `qc_failure_reason` split on `", "`;
  `qc_inputs_insufficient` sentinel counted and sorts first; clause-incidence may
  sum > `n_plants_excluded`. (RED)
- [x] 5.4 (TEST) Clause-token safety: assert every member of
  `qc._FAILURE_CLAUSE_ORDER` matches `^[a-z0-9_]+$` (freezes the `:`/`; ` encoding
  against a future clause name with a separator char). (RED→GREEN)
- [x] 5.5 Implement the groupby (`dropna=False`), exclusion filter, counts, and
  the `exclusion_reasons` builder. (GREEN for 5.1–5.4)

## 6. Degenerate groups
- [x] 6.1 (TEST) All-excluded group (`n_plants_passing_qc == 0`): row emitted
  (not dropped), all stats NaN, `frac_nutating`/`handedness_consensus_frac` NaN,
  `handedness_mode == 0`. (RED)
- [x] 6.2 (TEST) Mixed frame — one passing group + one all-excluded group:
  `handedness_mode`, `n_plants_passing_qc`, `n_plants_excluded` stay int dtype
  across both rows (the all-excluded `0` does not upcast the column to float).
  (RED)
- [x] 6.3 (TEST) Empty input: a 0-row `per_plant_df` (full column set + 1:1 units)
  → 0-row `per_genotype_df` with the full expected column set + 1:1 units, no
  raise. Assert the count columns are integer dtype even on the empty frame (a
  bare `pd.DataFrame(columns=[...])` defaults to object — build with explicit
  dtypes). (RED)
- [x] 6.4 Implement the degenerate-group + empty-frame handling (the empty path
  derives columns from the units-driven schema, NOT by iterating groups). (GREEN
  for 6.1–6.3)

## 7. Output schema order + units mapping
- [x] 7.1 (TEST) Column order: `plate_id, genotype, treatment,
  n_plants_passing_qc, n_plants_excluded, exclusion_reasons`, then
  `<trait>_median`/`<trait>_iqr` (input order), `frac_nutating`,
  `handedness_mode`, `handedness_consensus_frac`. Assert the full derived 73-column
  set against an explicit expected list built from the per-plant schema (must
  include literal verbose names like `T_nutation_median_median`,
  `T_nutation_median_iqr`, the helix magnitude `helix_signed_area_abs_px2_median`,
  and `angular_amplitude_median`; must NOT include `principal_axis_angle_median`,
  `principal_axis_angle_iqr`, `helix_signed_area_px2_median`, or
  `helix_signed_area_px2_iqr`). (RED)
- [x] 7.2 (TEST) `per_genotype_units` 1:1 covers the columns; median/IQR carry
  the source unit (incl. the `px²` glyph on `helix_signed_area_abs_px2_*`
  round-tripping through the sidecar); fractions → `—`; `handedness_mode`/counts →
  `int`; `exclusion_reasons`/identity → `string`; every value ∈
  `PIPELINE_UNIT_VOCABULARY`. (RED)
- [x] 7.3 Implement column ordering + the `per_genotype_units` derivation.
  (GREEN for 7.1–7.2)

## 8. Determinism
- [x] 8.1 (TEST) Two invocations on the same inputs compare equal under
  `pandas.testing.assert_frame_equal`. (RED → GREEN once ordering/tie-breaks are
  deterministic.)

## 9. Per-genotype CSV + sidecar I/O (`_io.py`)
- [x] 9.1 (TEST, COMMIT WITH 9.2 — atomic) Refactor guard: after a shared
  `_validate_units_coverage(df, units, *, fn_name)` helper is introduced,
  `write_per_plant_csv` behavior is unchanged — success writes CSV + sidecars;
  the helper still raises `ValueError` BEFORE any file is written AND names the
  offending column(s) (the existing
  `tests/test_circumnutation_foundation.py` writer tests match on `track_id` /
  missing/extra column names and `tests/test_circumnutation_pipeline.py`
  `test_compute_traits_negative_units_coverage_raises` matches on
  `band_power_ratio` — all must stay green). (RED if helper absent)
- [x] 9.2 Extract `_validate_units_coverage` (1:1 coverage only) from
  `write_per_plant_csv` preserving the 3 invariants (raises `ValueError`; before
  any write; names the offending column). Keep the `PIPELINE_UNIT_VOCABULARY`
  membership check INLINE in each writer (not in the shared helper). Commit
  9.1+9.2 together so the suite is never red between them. (GREEN for 9.1 +
  existing per-plant writer tests)
- [x] 9.3 (TEST, in `tests/test_circumnutation_aggregation.py`)
  `write_per_genotype_csv` writes CSV + `<stem>.units.json` + `run_metadata.json`
  (Path-based `.stem`/`.as_posix()` idiom, win32-safe); `read_per_genotype_csv`
  round-trips frame, units, and run-metadata incl. nested `_constants_version == 6`
  / `cadence_s`. (RED)
- [x] 9.4 (TEST) `write_per_genotype_csv` rejects an out-of-vocabulary unit AND a
  non-1:1 units map — `ValueError` raised, NO file written (both error branches,
  for coverage). (RED)
- [x] 9.5 (TEST) `read_per_genotype_csv` on a CSV whose sidecars were removed
  returns `({}, {})` for units + run-metadata. (RED)
- [x] 9.6 Implement `write_per_genotype_csv` / `read_per_genotype_csv` reusing
  `_validate_units_coverage`, `write_units_sidecar`, `write_run_metadata`. Docstring
  documents the fixed-name `run_metadata.json` one-CSV-per-directory constraint
  (point to §0.2 follow-up). (GREEN for 9.3–9.5)

## 10. Real plate-001 integration test (the validation)
- [x] 10.1 (TEST, function-local import of `pipeline`) Reuse PR #14's
  `_load_plate001_inputs` (LFS-gated `pytest.mark.skipif`): run
  `pipeline.compute_traits` → `aggregate_by_genotype` → assert the actual
  empirically-verified plate-001 profile (round-4 ground truth): exactly one
  `(plate_001, Nipponbare, none)` row; `n_plants_passing_qc == 1`,
  `n_plants_excluded == 5`, `n_plants_passing_qc + n_plants_excluded == 6`;
  `exclusion_reasons == "d2_msd_agreement_high:5"`; all aggregated `_median`
  columns finite; all aggregated `_iqr` columns NaN (only 1 passing plant → n < 2
  for every trait); `handedness_mode == 1`, `handedness_consensus_frac == 1.0`,
  `frac_nutating == 1.0`. (If a future fixture/QC change shifts these exact
  numbers, update the assertion — it is a real-data anchor, not a guess.) Then
  round-trip through `write_per_genotype_csv` / `read_per_genotype_csv` writing to
  a DISTINCT tmp subdirectory (so the fixed `run_metadata.json` does not collide
  with any per-plant artifact). NOTE: because plate-001 has only 1 passing plant,
  the multi-plant median/IQR, handedness tie-break, and multi-clause
  `exclusion_reasons` paths are exercised by the §3–§6 SYNTHETIC tests, not here.
  (RED → GREEN)

## 11. Docs + verification gates
- [x] 11.1 Cross-reference §7.7 in the `aggregation.py` module docstring; confirm
  no theory.md change is needed. Word it precisely: PR #15 implements §7.7's
  per-genotype median ± IQR on the current 1:1 track↔plant data (the literal
  two-level per-plant collapse is deferred — §0.1 follow-up — and guarded by the
  1:1 validation), and drops `principal_axis_angle` from the linear aggregation
  (circular summary deferred — §0.3). If any design detail deviates from theory,
  record it in proposal/design/spec + theory Appendix B + changelog with a "Why".
- [x] 11.2 Add a `docs/changelog.md` entry under the `[Unreleased] / Added`
  section, following the house format (lead with the new public symbols). It MUST
  name: `aggregation.aggregate_by_genotype`; the new `_io.write_per_genotype_csv` /
  `read_per_genotype_csv` + the ADDED "Per-genotype trait CSV and sidecar I/O"
  requirement + the behavior-preserving `_validate_units_coverage` refactor; the
  1:1 guard + deferred two-level collapse; the trait-level decisions
  (`principal_axis_angle` dropped + circular summary deferred;
  `helix_signed_area_px2` aggregated by magnitude; `growth_axis_unreliable`
  dropped); and the #222 naming dependency — mirroring how the PR #7/#10 entries
  enumerate their deviations.
- [x] 11.5 (parity with archived PR #14 tasks) Set the PR #15 roadmap row in
  `docs/circumnutation/roadmap.md` to in-flight (🟡) and optionally enrich the
  terse row with the 1:1-guard / `principal_axis_angle` / helix-magnitude notes.
  The ⬜→✅ flip itself stays at archive (post-merge), not in this PR.
- [x] 11.6 (optional) Add a one-line forward-pointer in theory.md §7.7 noting the
  deferred two-level per-plant collapse (§0.1) — no Appendix B entry is required
  (round-4 audit: the PR #15 choices do not stale any theory text).
- [x] 11.3 Run `uv run pytest tests/ -q` (full suite green), `black --check
  sleap_roots tests`, `pydocstyle --convention=google
  sleap_roots/circumnutation/`, `uv lock --check`, `uv run mkdocs build`, and
  `npx openspec validate add-circumnutation-per-genotype-aggregation --strict`.
- [x] 11.4 Confirm `_CONSTANTS_VERSION` is unchanged at 6 (no new scientific
  constant); aim ≥ 90% coverage on `aggregation.py` and the new `_io` writers
  (exercise both validation error branches to hit the lines).
