## Why

The circumnutation pipeline (PR #14) produces a composed **per-plant** trait
table (46 columns, one row per plant) but there is no post-pipeline
aggregation layer. Biologists compare **genotypes**, not individual plants:
they need a per-genotype summary (median ± IQR across plants) with an explicit
count of plants passing QC and the reasons plants were excluded. theory.md
§7.7 specifies this aggregation; PR #15 delivers it as the first post-pipeline
layer, independently of the plotting (PR #16) and CLI (PR #17) work that
follows.

## What Changes

- **ADD** a new implementation module `sleap_roots/circumnutation/aggregation.py`
  exposing the top-level pure function
  `aggregate_by_genotype(per_plant_df, units) -> (per_genotype_df, units)`.
  It groups the PR #14 composed per-plant frame by `(plate_id, genotype,
  treatment)` (preserving NaN group keys) and emits one row per group:
  `n_plants_passing_qc`, `n_plants_excluded`, `exclusion_reasons`, then
  `<trait>_median` / `<trait>_iqr` for 32 numeric traits (`principal_axis_angle`,
  a wrapping circular angle, is excluded; `helix_signed_area_px2`, a signed
  chirality measure, is aggregated by its magnitude as
  `helix_signed_area_abs_px2_median`/`_iqr`), `frac_nutating`, `handedness_mode`,
  and `handedness_consensus_frac`.
- **MODIFY** the `circumnutation` capability's **Package layout** requirement:
  the implementation-module count grows from 10 to **11** by ADDITION of
  `aggregation` (the stub-module count is UNCHANGED at 2), mirroring the
  PR #6 `nutation` / PR #10 `traveling_wave` addition pattern. A callability
  scenario for `aggregation.aggregate_by_genotype` is locked in the MODIFIED
  Package layout requirement.
- **ADD** a **Per-genotype aggregation API** requirement defining the grouping
  keys, per-column aggregation statistics, QC exclusion gate, degenerate-group
  handling, and determinism.
- **ADD** a **Per-genotype trait CSV and sidecar I/O** requirement: new
  `write_per_genotype_csv` / `read_per_genotype_csv` in `_io.py` that reuse the
  existing units-sidecar + run-metadata machinery and the closed
  `PIPELINE_UNIT_VOCABULARY` (no new vocabulary entries). The units-coverage +
  vocabulary validation currently inlined in `write_per_plant_csv` is factored
  into a shared private helper both writers call (behavior-preserving refactor).

## Impact

- Affected specs: `circumnutation` (MODIFIED: Package layout; ADDED: Per-genotype
  aggregation API; ADDED: Per-genotype trait CSV and sidecar I/O).
- Affected code:
  - NEW `sleap_roots/circumnutation/aggregation.py`.
  - `sleap_roots/circumnutation/_io.py` — new `write_per_genotype_csv` /
    `read_per_genotype_csv`; factor `_validate_units_coverage` shared helper
    (behavior-preserving: keep `write_per_plant_csv` raising `ValueError` before
    any write, naming the offending column; keep its existing tests green).
  - NEW `tests/test_circumnutation_aggregation.py` — the aggregation API +
    per-genotype writer tests.
  - `tests/test_circumnutation_foundation.py` — add `"aggregation"` to the
    `test_module_logger_is_namespaced` parametrize list (new impl module, not a
    stub), and DO NOT add `aggregate_by_genotype` to
    `IMPLEMENTATIONS_WITH_CONSTANTS_KWARG` (it takes no `constants=`); the
    existing `write_per_plant_csv` `ValueError` tests guard the
    `_validate_units_coverage` extraction.
  - `tests/test_circumnutation_pipeline.py` — the existing
    `test_compute_traits_units_dict_covers_all_columns_in_vocab` /
    `test_save_*` tests also guard the writer refactor (must stay green).
- Pure-pixel (CC-3): aggregation operates on and emits pure-pixel traits only;
  no `px_per_mm`, no mm-bearing columns.
- `_CONSTANTS_VERSION` is UNCHANGED at 6 — no new scientific/tunable constant
  is introduced (the 25/75 IQR percentile bounds and the min-2-finite-values
  threshold are structural, not tunable).
- Track↔plant 1:1: `aggregate_by_genotype` validates that each `(plate_id,
  genotype, treatment, plant_id)` maps to exactly one row and raises if a future
  divergence violates it (`roadmap.md` §`plant_id` assigns PR #15 the
  `plant_id`-keying responsibility). The two-level `plant_id`-collapse is deferred
  to a follow-up issue (filed before the PR).
- Provenance: `run_metadata.json` is a fixed name per directory; the per-genotype
  writer documents the one-CSV-per-directory constraint (a stem-prefixed name is
  a separate follow-up issue — the PR #14 follow-up (b)).
- Reproducibility: the per-genotype CSV gets its own units sidecar +
  run-metadata bundle, so the aggregated artifact is self-describing and
  traceable like the per-plant CSV.

## References

- Authoritative spec: `docs/circumnutation/theory.md` §7.7 (per-genotype
  aggregation), §7.1–§7.6 (per-track traits being aggregated).
- Roadmap: `docs/circumnutation/roadmap.md` PR #15 row; CC-3 (pure-pixel),
  CC-4 (row identity), CC-9 (logging).
- Input substrate: Requirement: Circumnutation pipeline composition API
  (PR #14, the 46-column per-plant frame); Requirement: Units sidecar JSON;
  Requirement: Run-metadata sidecar.
- Parent epic: #197. Predecessor: PR #14 tracking issue #234 (this PR consumes
  #234's 46-column per-plant frame). Tracking issue: TBD (child of #197; title
  `… [PR #15]`; labels enhancement / circumnutation / multi-pr) — drafted to the
  vault and posted per the lazy-issue workflow before the PR.
- Follow-up issues to file before the PR: (a) the two-level `plant_id`-collapse
  generalization of `aggregate_by_genotype` (so it survives a future track↔plant
  divergence instead of raising); (b) a stem-prefixed `run_metadata.json` name to
  remove the fixed-name clobber in the `_io` writers (the PR #14 follow-up b);
  (c) a circular-statistics per-genotype summary for `principal_axis_angle`
  (circular mean + resultant length), which PR #15 drops rather than aggregate
  linearly.
- Related: #222 (program-wide unit-suffix convention). The aggregated
  `_median` / `_iqr` columns add no new units-sidecar vocabulary, but they are
  *derived* from the source trait names — so when #222 renames source traits to
  carry a terminal unit suffix (`T_nutation_median → …_s`), the derived names
  become medial-suffixed (`…_s_median`); #222 must treat `_median`/`_iqr` as
  stat-suffixes after the unit suffix, or the aggregation naming re-keys then.
  This is a named #222 dependency, not "no bearing".
- Related: #230 (L_gz/L_c research — informational only, does not block PR #15).
