## Context

PR #14 shipped `CircumnutationPipeline.compute_traits(inputs) -> (per_plant_df,
trajectory_df, units_dict)`, producing a composed **per-plant** trait frame of
46 columns: 8 row-identity columns (`series, sample_uid, timepoint, plate_id,
plant_id, track_id, genotype, treatment`) + Tier 0 (10) + QC (10) + Tier 1 (8)
+ Tier 2 (4) + Tier 3c (6). In this codebase `track_id` and `plant_id` are 1:1
(roadmap §Input), so the per-plant frame is already one row per plant.

theory.md §7.7: *"The above are per-track. Per-plant aggregation is median
across the [N] tracks of the same plant. Per-genotype aggregation is median ±
IQR across plants, with explicit `n_plants_passing_qc` count. Plants where
`track_is_clean = False` for all tracks are excluded from aggregation but
flagged in the trait CSV with reason."*

Because track↔plant is 1:1 here, PR #15 aggregates **across plants** per
`(genotype, treatment)` (schema-aware: per `plate_id`), not across tracks within
a plant. This is a NEW capability — there is no aggregation stub in the package
(only `parametric` PR #11 and `plotting` PR #16 remain as stubs).

## Goals / Non-Goals

- **Goals:** a deterministic, pure-pixel per-genotype aggregation that composes
  in-memory after `compute_traits`; explicit QC exclusion with counts and
  reasons; a self-describing per-genotype CSV (units sidecar + run-metadata);
  validated on real plate-001.
- **Non-Goals:** cross-plate pooling (plates are kept separate); plotting
  (PR #16); CLI wiring (PR #17); per-trait finite-count columns (deferred — see
  D2); mm conversion (CC-3 keeps the pipeline pure-pixel).

## Decisions

### D1 — Home, shape, and signature
New module `sleap_roots/circumnutation/aggregation.py` with a top-level pure
function:

```python
def aggregate_by_genotype(
    per_plant_df: pd.DataFrame,   # PR #14 composed 46-col frame
    units: dict,                  # PR #14 units_dict — drives float detection + output units
) -> tuple[pd.DataFrame, dict]:   # (per_genotype_df, per_genotype_units)
```

It consumes the **in-memory** composed `per_plant_df` (not a CSV path), so it
composes directly after `compute_traits`; CSV-driven callers read first via
`read_per_plant_csv`. This mirrors the per-tier `DataFrame → DataFrame` pattern
and keeps `pipeline.py` focused on composition.

**Input validation.** Before aggregating, the function validates (a) that `units`
1:1-covers `per_plant_df`'s columns (reusing the shared `_validate_units_coverage`
helper of D5 — raise `ValueError` naming the offending column), and (b) the
track↔plant **1:1 guard** (see D3). Both fail loud before any output.

**The track↔plant 1:1 guard (review reconciliation).** `roadmap.md` §`plant_id`
explicitly assigns `aggregate_by_genotype` (PR #15) the job of keying on
`plant_id` so aggregation stays correct under three foreseeable track↔plant
divergence cases (track fragmentation, multi-track-per-plant, external
plate-position metadata). In the current data track↔plant is 1:1, so the
composed frame is already one row per plant and grouping by `(plate_id, genotype,
treatment)` treats each row as a plant. Rather than silently relying on 1:1 (the
exact failure the roadmap warns about) or building speculative multi-track
collapse logic that is a no-op today (YAGNI), PR #15 **validates** the invariant:
each `(plate_id, genotype, treatment, plant_id)` combination must map to exactly
one row, else `ValueError` naming the offending `plant_id`. The two-level
`plant_id`-collapse (median across a plant's tracks first, then median ± IQR
across plants — the literal §7.7 shape) is **deferred to a follow-up issue** and
filed before the PR. This mirrors the `_validate_integer_identity` fail-loud
precedent from PR #14.

The guard key `(plate_id, genotype, treatment, plant_id)` **intentionally omits
`series` / `sample_uid`** (and `track_id`), even though `_io._IDENTITY_5_TUPLE`
is `(series, sample_uid, plate_id, plant_id, track_id)`. `plant_id` is unique
only *within* a series/sample, not globally; but the aggregation **group** is
`(plate_id, genotype, treatment)`, and within one group `plant_id` is the plant
key. Two rows in one group that share a `plant_id` (whether from a fragmented
track, a multi-track plant, or two series colliding on `plant_id`) are genuinely
ambiguous for a per-genotype median and SHALL be flagged — adding `series` /
`sample_uid` to the guard key would silently let such collisions through and
defeat the guard. An implementer must NOT "repair" the key by widening it. This
holds for the current single-series-per-call data (plate-001: one `series` /
`sample_uid`, `plant_id == track_id`); a multi-series-per-group frame is exactly
the deferred-collapse case where raising is the intended behavior.

- *Alternatives:* a method on `CircumnutationPipeline` (mixes composition +
  aggregation responsibilities, grows `pipeline.py`); a function in `_io.py`
  (serialization module, not statistics); taking a CSV path (couples to disk,
  not unit-testable on synthetic frames). Rejected.

### D2 — Per-column aggregation statistics
Float/special classification is **units-driven** (D8): a column is a float
trait iff its `units` value is a numeric unit (anything other than `int`,
`bool`, `string`). Special columns are named explicitly.

| Input column(s) | Output | Rule |
|---|---|---|
| every numeric-unit trait NOT in the special set (Tier 0/1/2/3c + QC floats, 32 cols) | `<trait>_median`, `<trait>_iqr` | median + IQR (Q75−Q25) over passing plants, NaN-skipping; per-trait finite n ≤ `n_plants_passing_qc` |
| `is_nutating` (bool) | `frac_nutating` | mean over passing plants |
| `handedness` (int {−1,0,+1}) | `handedness_mode` (int) + `handedness_consensus_frac` (—) | most-frequent sign by explicit value counts (tie-break smallest abs then smallest signed); fraction of passing plants agreeing with the mode |
| `track_is_clean` (bool) | — (drives counts) | the QC gate, not a trait |
| `qc_failure_reason` (string) | — (feeds `exclusion_reasons`) | not a trait |
| `growth_axis_unreliable` (bool) | **dropped** | a `track_is_clean` failure clause (qc.py `_compose_track_is_clean_and_reason`), so always `False` among passing plants → constant-0, no information |
| `principal_axis_angle` (rad, wrapping) | **dropped** | a wrapping circular angle (arctan2 in (−π, π]); linear median/IQR is unsound near ±π, and it is an absolute per-plant reference direction with arbitrary cross-plant orientation. Circular summary deferred to a follow-up issue. (`angular_amplitude`, the other `rad` column, is a non-negative peak-to-peak magnitude → aggregated normally.) |
| `helix_signed_area_px2` (px², signed chirality) | **magnitude** → `helix_signed_area_abs_px2_median` / `_iqr` | the float twin of `handedness`; its sign is chirality, which is bimodal within a genotype, so a signed cross-plant median cancels toward 0 (misreads as "no helix"). Aggregate `median`/`iqr` of `|value|` for helix *strength*; chirality *direction* is carried by `handedness_mode`/`consensus_frac`. (Other signed traits — `v_long_signed*`, `v_lat_signed*`, `period_residual_vs_derr_reference` — have per-plant-consistent sign semantics, verified, and aggregate linearly.) |
| identity (`series, sample_uid, timepoint, plant_id, track_id`) | dropped | vary within group |

Of the 33 numeric-unit columns, `principal_axis_angle` is dropped and `helix_signed_area_px2` is aggregated via its magnitude — so **32 traits get a `_median`/`_iqr` pair** (31 directly + the helix magnitude pair) → 64 stat columns; with the 3 group keys + 3 count/reason columns + `frac_nutating` + `handedness_mode` + `handedness_consensus_frac`, the per-genotype frame has **73 columns**.

`n_plants_passing_qc` is the single per-group count (spec-literal §7.7); no
per-trait finite-count columns are emitted (lean schema; a clean future
addition).

- *Why median+IQR fixed:* §7.7 mandates "median ± IQR". IQR (not MAD/std)
  matches the per-plant tier convention (`T_nutation_iqr` etc.).
- *Why mode for handedness:* handedness is a categorical sign; a numeric mean
  would invent fractional handedness. `handedness_consensus_frac` preserves the
  agreement strength a mode discards.

### D3 — Grouping keys + per-plate reporting
Group key = `(plate_id, genotype, treatment)`. Plates are never pooled (plate is
a batch confound; "reports per-plate_id separately", §7.7). One tidy output
table, one row per group, rows sorted deterministically by the key. A genotype
spanning two plates yields two rows. plate-001 (6 plants, one plate, one
genotype "Nipponbare", treatment "none") → exactly one row.

- *Alternative:* two tables (pooled-across-plates + per-plate breakdown) —
  more artifacts and introduces cross-plate pooling not asked for. Deferred.

### D4 — QC exclusion + flagging
A plant is excluded from all median/IQR computations iff its
`track_is_clean == False`. PR #14's per-plant `track_is_clean` (False ⇒ excluded)
and `qc_failure_reason` columns already satisfy "flagged in the trait CSV with
reason" at the per-plant level; PR #15 **consumes** them and does NOT mutate the
PR #14 contract. The per-genotype output adds:

- `n_plants_passing_qc` (int): count of `track_is_clean == True` plants in the group.
- `n_plants_excluded` (int): count of `track_is_clean == False` plants in the group.
- `exclusion_reasons` (string): clause→count summary over excluded plants,
  ordered by `qc._FAILURE_CLAUSE_ORDER`, e.g.
  `"frac_outlier_steps_high:2; worst_step_ratio_high:1"`; empty `""` when
  nothing is excluded. `qc_failure_reason` is split on `", "` (the qc.py join
  separator); the `qc_inputs_insufficient` sentinel counts as its own clause.
  Counts are clause-incidence (a plant failing two clauses counts in both), so
  they may sum to more than `n_plants_excluded` (documented).

### D5 — Output schema, units, writer
Output column order (73 cols): `plate_id, genotype, treatment,
n_plants_passing_qc, n_plants_excluded, exclusion_reasons`, then
`<trait>_median` / `<trait>_iqr` per aggregated numeric-unit trait (input column
order, `principal_axis_angle` excluded), `frac_nutating`, `handedness_mode`,
`handedness_consensus_frac`.

Aggregated-column units, all already in `PIPELINE_UNIT_VOCABULARY` (**no new
vocabulary**):

| Output column | Unit |
|---|---|
| `<trait>_median`, `<trait>_iqr` | the source trait's unit (median & IQR are same-dimension) |
| `frac_nutating`, `handedness_consensus_frac` | `—` |
| `handedness_mode`, `n_plants_passing_qc`, `n_plants_excluded` | `int` |
| `exclusion_reasons`, `plate_id`, `genotype`, `treatment` | `string` |

Naming: single-underscore `<trait>_median` / `<trait>_iqr` (matches the package
convention). This produces verbose-but-unambiguous names where the source trait
already ends in `_median` (e.g. `T_nutation_median` → `T_nutation_median_median`
/ `T_nutation_median_iqr`); each column is unique and parseable.

Writer: new `write_per_genotype_csv(out_path, df, units, run_metadata)` and
`read_per_genotype_csv(in_path)` in `_io.py`, writing the CSV + `<stem>.units.json`
+ `run_metadata.json` exactly like the per-plant pair. The units-coverage
(1:1 keys↔columns) check currently inlined in `write_per_plant_csv` is factored
into a shared private `_validate_units_coverage(df, units, *, fn_name)` helper.
The helper does **coverage only**; each writer keeps the separate
`PIPELINE_UNIT_VOCABULARY` membership check inline (the aggregation input site
reuses only the coverage helper — per-plant units are pipeline output, already
in-vocabulary, so a vocab check there is unnecessary). The extraction is
**behavior-preserving** and must keep all three invariants the existing per-plant
writer tests assert: it raises `ValueError`, *before any file is written*, and
*names the offending column(s)* in the message (existing tests match on
`track_id`, `band_power_ratio`, the missing/extra column names). Guarded by the
existing per-plant writer tests in `tests/test_circumnutation_foundation.py` and
`tests/test_circumnutation_pipeline.py`.

**`run_metadata.json` fixed-name clobber (review reconciliation).** Like
`write_per_plant_csv`, the new writer writes a fixed-name `run_metadata.json`,
and `CircumnutationPipeline.save` already warns that a second write to one
directory clobbers the first's provenance. PR #15 does not worsen the contract
silently: `write_per_genotype_csv`'s docstring documents that callers SHALL write
at most one CSV artifact per output directory (co-locating per-plant +
per-genotype CSVs overwrites `run_metadata.json`), and the round-trip /
integration tests write the per-genotype artifact to a **distinct tmp
subdirectory**. A stem-prefixed metadata filename that removes the clobber
entirely is filed as a follow-up issue (the PR #14 follow-up (b)), kept out of
PR #15 to preserve per-plant/per-genotype writer symmetry.

### D6 — Degenerate-group handling
The aggregation NEVER drops a group and NEVER raises on degenerate input.

- **All-excluded group** (`n_plants_passing_qc == 0`): the row is still emitted
  with `n_plants_excluded = group size`, `exclusion_reasons` filled, all
  `<trait>_median` / `<trait>_iqr` = NaN, `frac_nutating` = NaN,
  `handedness_consensus_frac` = NaN, and `handedness_mode = 0` (the neutral
  value `psi_g` already uses for the int fill), so `handedness_mode` stays
  `int64`. The `n_plants_passing_qc == 0` count disambiguates a 0 mode here from
  a genuine neutral-handedness majority.
- **Tiny n** (a trait backed by `< 2` finite passing-plant values): `<trait>_iqr`
  = NaN (spread is undefined for n < 2; computed via an internal per-trait
  finite count — still not emitted). `<trait>_median` is emitted for n = 1.
  The `2` threshold is a structural minimum, not a tunable constant — no
  `_CONSTANTS_VERSION` bump.

### D7 — Determinism + tests
- IQR via `scipy.stats.iqr(x, nan_policy="omit", interpolation="linear")`
  (module-qualified scipy) wrapped to enforce the `< 2`-finite → NaN rule.
  `interpolation="linear"` is pinned for cross-version determinism. `scipy.stats.iqr`'s
  `interpolation` kwarg was deprecation-churned historically (→ `method`), and
  the wrapper relies on `iqr([5.0, nan], nan_policy="omit") == 0.0` (a single
  finite value → IQR 0, which the wrapper then converts to NaN). A guard test
  asserts this exact behavior so a future scipy bump that changes NaN /
  interpolation semantics fails loud rather than corrupting published spreads.
- `handedness_mode` is computed from **explicit value counts** (not
  `pandas.Series.mode`, whose `[0]` would silently pick the smallest sign): take
  the sign(s) with the maximum count, break ties by smallest `abs(value)` then
  smallest signed value, so a `{+1, -1}` tie resolves to `-1` and an `{0, +1}`
  tie resolves to `0`. The module docstring notes that `handedness_consensus_frac`
  near `0.5` flags the mode as a tie-break artifact, not a majority.
- `handedness_mode`, `n_plants_passing_qc`, and `n_plants_excluded` are kept
  `int64`; the all-excluded `handedness_mode = 0` fill prevents float promotion.
  A test on a frame **mixing** a normal group and an all-excluded group asserts
  the int dtype survives (a single all-excluded group can pass while the mixed
  case upcasts under some pandas paths).
- **Construction technique (review reconciliation):** build the output as a
  list-of-per-group dicts → `pd.DataFrame(rows)` (the established `qc.py`
  per-track pattern), NOT `groupby().apply(lambda g: pd.Series(...))` — the latter
  packs each group's mixed-dtype values into one homogeneous Series and upcasts
  `handedness_mode`/counts to float64. The mixed-frame dtype test guards this.
- **Grouping uses `dropna=False`** so a plant with a NaN group-key value
  (`genotype`/`treatment`/`plate_id` — the per-plant template permits NaN object
  columns) forms its own group instead of being silently dropped by pandas'
  default `dropna=True`. A NaN-key test asserts `Σ(n_plants_passing_qc +
  n_plants_excluded) == len(per_plant_df)`.
- **Warning hygiene:** the all-NaN-slice `RuntimeWarning` (`np.nanmedian`) and
  `SmallSampleWarning` (`scipy.stats.iqr`) on degenerate groups are suppressed via
  `np.errstate` / `warnings.catch_warnings`, matching the house style in
  `qc.py:309` and `psi_g.py` (which explicitly return NaN without a warning).
- Output rows sorted by `(plate_id, genotype, treatment)`.
- **Integration test (the validation):** run `compute_traits` on real plate-001
  (reusing PR #14's `_load_plate001_inputs`, LFS-gated) → `aggregate_by_genotype`
  → assert the round-4 empirically-verified profile: exactly one
  `(plate_001, Nipponbare, none)` row, `n_plants_passing_qc == 1`,
  `n_plants_excluded == 5`, `exclusion_reasons == "d2_msd_agreement_high:5"`, all
  `_median` finite, all `_iqr` NaN (1 passing plant), `handedness_mode == 1`,
  `handedness_consensus_frac == 1.0`, `frac_nutating == 1.0`; then round-trip
  through `write_per_genotype_csv` / `read_per_genotype_csv` to a distinct tmp
  subdir. The multi-plant median/IQR, handedness tie-break, and multi-clause
  reasons paths are covered by synthetic unit tests (plate-001 has only 1 passing
  plant).
- **Synthetic unit tests** for each branch: float median+IQR, handedness mode +
  tie-break, IQR-NaN at n < 2, QC exclusion + counts, all-excluded empty group,
  clause:count `exclusion_reasons`, units-coverage, determinism
  (`assert_frame_equal` over two runs).

### D8 — Constants, schema derivation, picklability
- `_CONSTANTS_VERSION` stays **6**. Only name-tuple structural constants are
  added to `aggregation.py` (e.g. `_AGG_SPECIAL_COLUMNS`, `_AGG_IDENTITY_KEEP`,
  `_AGG_IDENTITY_DROP`) — column names, like `ROW_IDENTITY_COLUMNS`, not
  versioned scientific constants.
- Float-trait detection is **units-driven**: float traits = input columns whose
  `units` value is not in `{"int", "bool", "string"}`, are not identity keys, and
  are not in the explicit `_AGG_SPECIAL_COLUMNS` set (`is_nutating`, `handedness`,
  `track_is_clean`, `qc_failure_reason`, `growth_axis_unreliable`, the wrapping
  circular `principal_axis_angle`, and the signed-chirality `helix_signed_area_px2`
  — handled via its magnitude). Single source of truth = the per-plant units
  contract; future numeric traits are auto-included unless added to the special
  set. The expected-column-set test pins this so a new numeric column (or a new
  circular/signed-directional trait) surfaces in CI rather than being silently
  linearly aggregated.
- `aggregate_by_genotype` is a top-level pure function (no class), so there is
  no picklability concern (unlike `CircumnutationPipeline`).

## Risks / Trade-offs

- **Verbose stat names** (`T_nutation_median_median`) → accepted as unambiguous
  and convention-consistent; the units sidecar carries no new vocabulary entry.
  **#222 forward-collision (review reconciliation):** #222 will rename source
  traits to carry a terminal unit suffix (`T_nutation_median → T_nutation_median_s`).
  Because PR #15's `_median`/`_iqr` names are *derived* from the source name, that
  rename mechanically yields a *medial* unit suffix (`T_nutation_median_s_median`),
  which violates #222's "unit suffix is terminal" rule and its planned
  foundation-test regex. PR #15 does not pre-empt #222 (the source names are
  unsuffixed today), but the spec/proposal explicitly flag that #222 must treat
  `_median`/`_iqr` as stat-suffixes layered *after* the unit suffix, or the
  aggregation naming will need re-keying when #222 lands. This is a named
  dependency, not "no bearing".
- **Units-driven float detection** could silently aggregate a future numeric
  column inappropriately → mitigated: the test asserts the derived per-genotype
  column set against an explicit expected list built from the known per-plant
  schema, so a schema change surfaces in CI.
- **Signed/directional traits (review reconciliation).** Every signed trait in
  the set was audited against its source sign convention. `principal_axis_angle`
  (arbitrary absolute frame) is dropped; `helix_signed_area_px2` (bimodal
  chirality) is aggregated by magnitude. The remaining signed traits are sound to
  median linearly: `v_long_signed` (sign = growing vs retracting, anchored to the
  net-displacement growth axis — consistent across plants), `v_lat_signed`
  (deterministic 90° lateral convention, ≈0 by symmetry — a sanity check, not an
  arbitrary sign), `period_residual_vs_derr_reference` (sign = slower/faster than
  the rice reference — consistently meaningful). The `_abs` velocity twins are
  unambiguously non-negative.
- **`_validate_units_coverage` refactor** touches `write_per_plant_csv` →
  mitigated: behavior-preserving extraction, guarded by the existing per-plant
  writer tests (must stay green) and the plate-001 integration test.
- **Module-reload pollution** (the PR #14 PicklingError / isinstance lesson):
  tests that exercise cross-module behavior use the documented function-local
  import pattern; run the FULL suite before any foundation commit.

## Migration Plan

Additive only. No existing behavior changes: the per-plant pipeline, its CSV
writer, and the plate-001 integration test remain unchanged. The shared
`_validate_units_coverage` extraction is behavior-preserving. No constants
version bump, so existing run-metadata snapshots remain valid.

## Open Questions

None — D1–D8 resolved and user-approved before scaffolding.
