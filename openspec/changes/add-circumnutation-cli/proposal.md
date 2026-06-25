## Why

The circumnutation pipeline (PR #1–#16) is fully composable in Python —
`Series.load → adapt → compute_traits → save → aggregate_by_genotype → write →
save_plots` — but there is **no user-facing entry point**. A scientist with a
tracked `.slp` must hand-write the `Series → CircumnutationInputs` adapter (today
it exists only as test code, `_load_plate001_inputs()` in
`tests/test_circumnutation_pipeline.py`) and orchestrate every stage by hand. PR
#17 (roadmap row 17, `add-circumnutation-cli`) closes this gap: one
`sleap-roots circumnutation analyze <slp>` command that runs the whole pipeline,
and the formalized `series_to_inputs` adapter the roadmap says lands here.

## What Changes

- **ADD** `sleap_roots/circumnutation/adapters.py` — `series_to_inputs(series, *,
  cadence_s, sample_uid, series_name, timepoint, plate_id, genotype, treatment,
  r_px, run_id) -> tuple[CircumnutationInputs, dict]`. The single bridge from
  `sleap_roots.series` into the pure circumnutation core: `get_tracked_tips()` →
  prefix-anchored `track_id` integer coercion → CSV/flag/NaN identity resolution →
  `CircumnutationInputs` + an **identity-provenance dict** (`metadata_csv_path` +
  per-field `identity_source`). No click dependency (independently testable).
- **ADD** `sleap_roots/circumnutation/cli.py` — a `@click.group() circumnutation`
  holding the `analyze` command, registered on the existing root CLI via
  `main.add_command(circumnutation)` (mirrors `main.add_command(viewer)`).
- **MODIFY** `sleap_roots/cli.py` — add the `main.add_command(circumnutation)`
  line.
- **MODIFY** `sleap_roots/circumnutation/_io.py::gather_run_metadata` — add two
  optional keyword params `metadata_csv_path=None` and `identity_source=None`,
  recorded in `run_metadata.json` (additive, backward-compatible; existing callers
  write `null`).
- **MODIFY** the `circumnutation` spec **Package layout** requirement: grow the
  implementation-module count **12 → 14** by ADDITION of `cli` and `adapters`
  (net-new modules, never stubs; same addition shape as PR #15's `aggregation`);
  stub-module count UNCHANGED at 1 (`parametric`, PR #11, deferred).
- **MODIFY** the `circumnutation` spec **Run-metadata sidecar** requirement: add the
  `metadata_csv_path` + `identity_source` provenance fields.
- **ADD** two `circumnutation` spec requirements: **Series-to-CircumnutationInputs
  adapter** and **Circumnutation analyze CLI**.

Design highlights (full rationale in `design.md` and the brainstorm doc
`docs/superpowers/specs/2026-06-24-add-circumnutation-cli-design.md`):

- **Identity:** `--sample-uid` is **required** (the stable QR/sample id and the
  metadata-CSV join key — never silently synthesized, per CC-4); `--series-name`
  defaults to the `.slp` filename stem.
- **Metadata precedence:** CSV-as-source, flags-as-override. `Series.get_metadata`
  returns `np.nan` indistinguishably for "no CSV"/"missing"/"empty", so the rule is
  `pd.notna`-keyed: a flag always wins; the INFO override-notice fires only when the
  flag shadows a *real, different* CSV value.
- **Gated per-genotype aggregation (`--no-aggregate`):** aggregation is separable
  (`compute_traits`/`save`/`save_plots` never need genotype). When aggregation is
  on (default) and `genotype` is unresolved, `analyze` **hard-errors** (exit 1,
  before any output) rather than emit a vacuous `(NaN, NaN, NaN)` per-genotype CSV;
  `--no-aggregate` opts out (per-plant + plots only, genotype not required).
- **Output tree:** distinct `per_plant/` · `per_genotype/` · `plots/` subdirs —
  **sidesteps #238** (each `_io` writer's fixed-name `run_metadata.json` lives in
  its own dir) without changing the foundation `_io` contract. References #238 but
  does **not** close it.
- **Headless:** `--no-plots` gates the matplotlib import; when plotting,
  `matplotlib.use("Agg", force=True)` runs before `plotting` is imported.
- **Errors:** click handles usage errors (exit 2); the adapter normalizes its
  raisers to `ValueError`; the CLI wraps the pipeline body in `try/except
  (ValueError, FileNotFoundError)` → `ClickException` (exit 1); no catch-all
  (genuine bugs surface tracebacks). Mirrors `viewer/cli.py`.
- **Provenance (full traceability):** the CLI assembles run-metadata **once** via
  a single `gather_run_metadata` (resolved-absolute `input_path`, plus the new
  `metadata_csv_path` and per-field `identity_source` map) and writes it to BOTH
  the per-plant and per-genotype `run_metadata.json` via `write_per_plant_csv` /
  `write_per_genotype_csv` — byte-identical provenance, and a reader can tell
  whether each identity field came from a flag, the metadata CSV, or a default.
  (This supersedes the brainstorm's "keep `save()`" note — a single shared snapshot
  is what makes the identity provenance complete; the CLI does not route the
  per-plant write through `CircumnutationPipeline.save()`.)
- **CC-3 / CC-9:** no `--px-per-mm` (pure-pixel; `--help` points to
  `convert_to_mm`); `-v`/`-vv` set WARNING/INFO/DEBUG to stderr.

## Impact

- **Affected specs:** `circumnutation` (MODIFY Package layout; ADD two
  requirements).
- **Affected code:** `sleap_roots/circumnutation/cli.py` (new),
  `sleap_roots/circumnutation/adapters.py` (new), `sleap_roots/cli.py` (one-line
  registration), `sleap_roots/circumnutation/_io.py` (`gather_run_metadata` gains
  two optional provenance params). No new dependency (`click` is already required).
- **Tests:** `tests/test_circumnutation_cli.py` (new, `click.testing.CliRunner` +
  synthetic-`.slp` round-trip + real plate-001 skipif-guarded e2e),
  `tests/test_circumnutation_adapters.py` (new); plus extending
  `tests/test_circumnutation_foundation.py` (logger-namespace + module-import
  lists) and the `gather_run_metadata` provenance tests.
- **Related issues:** parent #197 (a PR #17 tracking issue is drafted to the vault
  and posted, with its number filled into #197, before merge — `Closes #N` added
  then); #238 (run_metadata clobber — sidestepped, referenced, not closed); #230
  (`L_gz`, informational). A dedicated follow-up issue will track threading
  `input_path` into `save_plots`' own `plots_metadata.json` (the plots sidecar still
  cannot record the source `.slp`; this is NOT what #241 covers — #241 is scoped to
  MCP-serializable `Result` views and recompute removal). PR #18 (user guide)
  consumes this CLI.
