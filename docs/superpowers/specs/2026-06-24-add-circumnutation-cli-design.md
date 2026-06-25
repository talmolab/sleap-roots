# add-circumnutation-cli — design (PR #17)

**Date:** 2026-06-24
**Branch:** `elizabeth/add-circumnutation-cli`
**OpenSpec change-id:** `add-circumnutation-cli`
**Parent issue:** #197 · **Closes:** (PR #17 tracking issue, drafted to vault)
**Roadmap row:** 17 — `sleap-roots circumnutation analyze`

## Summary

PR #17 adds the user-facing CLI for the circumnutation pipeline: a single
`circumnutation analyze` command that composes the whole pipeline on one `.slp`
(load → adapt → `compute_traits` → `save` → `aggregate_by_genotype` → write →
plots). It also formalizes the **`Series` → `CircumnutationInputs` adapter** that
the roadmap says lands here (today it exists only as test code,
`_load_plate001_inputs()` in `tests/test_circumnutation_pipeline.py`).

This is an **addition**, not a stub→impl graduation: there is no `cli` stub (PR
#16 graduated the last one; only the deferred Tier-4 `parametric` stub remains).
PR #17 adds `sleap_roots/circumnutation/cli.py` + `sleap_roots/circumnutation/
adapters.py` and registers the command on the existing click CLI.

### Scope / non-goals

- **In:** `circumnutation` click group + `analyze` command; the `series_to_inputs`
  adapter; output tree (per-plant CSV, per-genotype CSV, plots); pure-pixel
  contract surfaced in `--help`; logging via `-v`.
- **Out (deliberate):** no `--px-per-mm` / calibration surface (CC-3); no
  `ConstantsT` override surface in v1; no `trajectory_per_plant/<id>.csv` dumps
  (the pipeline doesn't emit them today — the roadmap line-16 sketch is
  aspirational); does **not** fix #238 (sidestepped via layout); does **not**
  widen `save_plots` to record `input_path` (noted as a follow-up).

## Architecture

Three new/changed surfaces:

1. **`sleap_roots/circumnutation/adapters.py`** (new) — `series_to_inputs(series,
   *, cadence_s, sample_uid, series_name, timepoint, plate_id, genotype,
   treatment, r_px, run_id) -> CircumnutationInputs`. The single bridge from
   `sleap_roots.series` into the pure circumnutation core. **No click
   dependency** (unit-testable without `CliRunner`); **no `Series` import in
   `_types.py`** (keeps the data class decoupled from the heavier
   `series`/`sleap_io` stack).
2. **`sleap_roots/circumnutation/cli.py`** (new) — a `@click.group()
   circumnutation` holding the `analyze` command. Mirrors
   `sleap_roots/viewer/cli.py`: lazy imports inside the command body, descriptive
   `--help`, `ClickException` for domain errors.
3. **`sleap_roots/cli.py`** (modified) — `main.add_command(circumnutation)`,
   exactly mirroring the existing `main.add_command(viewer)` line.

## D1 — CLI shape & option surface

`circumnutation` group + `analyze` command (mirrors `viewer/cli.py`); chosen over
a flat `circumnutation-analyze` or a bare `analyze` so future subcommands
(`aggregate`, `plot`, `convert-units`) slot in with no restructuring.

| Option | Type | Default / Notes |
|---|---|---|
| `SLP_PATH` (arg) | `click.Path(exists=True, path_type=Path)` | input `.slp` |
| `--cadence-s` | `float`, **required** | feeds all period/freq traits; no default (not recoverable from `.slp`) |
| `--sample-uid` | `str`, **required** | QR / stable sample id; the metadata-CSV join key |
| `--output-dir, -o` | `click.Path(path_type=Path)` | default `./<series_name>_circumnutation/`; `mkdir(parents=True, exist_ok=True)` |
| `--series-name` | `str` | default = `.slp` filename stem (human recording label) |
| `--metadata-csv` | `click.Path(exists=True, path_type=Path)` | optional; `Series` metadata join source |
| `--timepoint` | `str` | default NaN; object/string label (override/fallback) |
| `--plate-id` | `str` | default NaN (NaN-keyed group is allowed) |
| `--genotype` | `str` | **effectively required** via flag *or* `--metadata-csv`; NaN → hard error (see below) |
| `--treatment` | `str` | default NaN (NaN-keyed group is allowed) |
| `--r-px` | `float` | default `None` |
| `--run-id` | `str` | default `None` |
| `--no-plots` | flag | gates plotting + matplotlib import (D4) |
| `--no-aggregate` | flag | skip per-genotype aggregation; `per_genotype/` not written (D3) |
| `-v, --verbose` | count | log level (D8) |

No `--constants` / `--px-per-mm`.

### Identity semantics (the `sample_uid` / `series_name` decision)

`Series.get_metadata` joins `df[df["plant_qr_code"] == self.sample_uid]`, so
`sample_uid` is **not** a display label — it is the stable, unique physical-sample
identifier (the plate's QR code). It must be user-supplied so a pre-authored
metadata CSV can be keyed on it and so re-runs share identity (reproducibility).
A randomly minted UUID would break both. Therefore:

- **`--sample-uid` is required** (no silent synthesis — honors CC-4's "row-identity
  stays explicit, any divergence is a deliberate caller decision", roadmap.md:82).
- **`--series-name` defaults to the `.slp` filename stem** (the human recording
  label; "series" = the time-lapse of frames, the circumnutation analog of the
  RSA multi-scan series).

**Per-genotype aggregation is a gated step; genotype is required only when it
runs.** The per-genotype aggregation (`aggregate_by_genotype`) is fully separable —
`compute_traits` → `save` (per-plant) and `save_plots` never touch genotype, so the
pipeline can produce per-plant traits + plots without any genotype info. The
per-genotype CSV is meaningless without genotype labels, though: an all-NaN
identity would group all plants under one `(NaN, NaN, NaN)` key and emit a
scientifically vacuous file. So aggregation is gated on `--no-aggregate` (symmetric
with `--no-plots`):

- **Aggregation on (default), genotype resolvable** (`--genotype` or
  `--metadata-csv` with a `genotype` column) → full pipeline incl. `per_genotype/`.
- **Aggregation on (default), genotype NaN for any plant** → **hard error** (exit 1)
  *before any output is written*: *"genotype is required for per-genotype
  aggregation; supply --genotype / --metadata-csv, or pass --no-aggregate."* (Per
  Elizabeth's directive — never silently emit a vacuous per-genotype CSV, and never
  auto-skip-and-warn.)
- **`--no-aggregate`** → skip the per-genotype step and the `per_genotype/` dir
  entirely; per-plant + plots still produced; genotype not required.

`--no-aggregate` and `--genotype` are orthogonal — a user may still supply genotype
to populate the per-plant identity columns while skipping the per-genotype CSV.
`plate_id` and `treatment` remain optional (NaN-keyed groups are valid for those —
aspirational per CC-4); only `genotype` gates the aggregation step.

**`--output-dir` default — follows the repo convention.** Other pipelines default
output to cwd and disambiguate runs by embedding `series_name`
(`compute_plant_traits` writes `output_dir="." / f"{series_name}{suffix}"`;
`compute_multiple_dicots_traits_for_groups` defaults `output_dir="grouped_traits"`
with `mkdir(parents=True, exist_ok=True)`; `viewer` defaults to `viewer.html`).
Since `analyze` emits a *tree* rather than a single file, the faithful
translation is to default `--output-dir` to `./<series_name>_circumnutation/` —
cwd-relative, `series_name`-disambiguated (no clobber across `.slp`), and
descriptively self-labeled.

## D2 — the `Series` → `CircumnutationInputs` adapter

**Location:** `adapters.py`, `series_to_inputs(...)`.

**Transform (generalizes `_load_plate001_inputs`).** The adapter formalizes the
blueprint's *mechanical* transform — track_id strip + 8 identity columns — but
**replaces the blueprint's hardcoded test values** (`genotype="Nipponbare"`,
`treatment="none"`, `timepoint="T0"`) with the CSV/flag/NaN sourcing below. Those
literals were ad-hoc test fixtures, not the contract.

1. `df = series.get_tracked_tips()` → columns `["track_id", "frame", "tip_x",
   "tip_y"]`, `track_id` as `"track_<n>"` strings.
2. Derive integer `track_id`: strip a **prefix-anchored** `"track_"`
   (`removeprefix` / `^track_`, *not* the blueprint's global `.replace`) then
   coerce to int. On failure, raise a clear `ValueError` naming the offending
   track name(s): *"Cannot derive an integer track_id from track name(s)
   ['track_2a']; expected 'track_<int>'."*
3. `plant_id = track_id` (today's convention).
4. Resolve the identity fills (see metadata precedence) and set the 8
   `ROW_IDENTITY_COLUMNS`.
5. `return CircumnutationInputs(trajectory_df=df, cadence_s=…, R_px=…, run_id=…)`.

**Metadata precedence — CSV-as-source, flags-as-override (D2c = option A).** For
each of `genotype`/`treatment`/`timepoint`, the adapter resolves a single value
with an explicit, implementable rule (B1 — `Series.get_metadata` returns `np.nan`
*identically* for "no CSV", "missing column", and "empty cell", so the rule keys
on `pd.notna`, not on distinguishing those cases):

```
csv_val = get_metadata(field) if --metadata-csv else np.nan   # raw cell, not Series.<prop>
resolved = flag if flag is not None else csv_val
if flag is not None and pd.notna(csv_val) and str(flag) != str(csv_val):
    log.info("--%s overrides metadata-csv value %r -> %r", field, csv_val, flag)
```
So a flag always wins; the INFO override-notice (CC-9) fires only when the flag
actually shadows a *real* (non-null, different) CSV value — never spuriously.
With neither flag nor CSV value, the field is `NaN`.

**`timepoint` dtype contract (B2).** `--timepoint` is a free-form **object/string**
label. The adapter reads the CSV cell via the **raw `get_metadata("timepoint")`**
(NOT the `Series.timepoint` property, which coerces to `float` and *raises* on a
non-numeric value, series.py:305-330) and `str()`-normalizes both the flag and the
CSV value, so the column is uniformly string-typed (consistent with
`_build_per_plant_template` casting `timepoint` to `object`).

**Malformed `--metadata-csv` (I3).** `get_metadata` does `pd.read_csv` internally;
a non-parseable file raises a pandas error (e.g. `ParserError`), *not* `ValueError`.
The adapter wraps the metadata read and converts any read failure into a clear
`ValueError` (caught by the CLI → clean `ClickException`, D6) rather than leaking a
traceback.

**Edge handling (D2d):**
- **No usable tracks / empty trajectory:** do **not** duplicate validation —
  `get_tracked_tips` already raises `ValueError` for untracked predictions, and
  `CircumnutationInputs` already raises on an empty `trajectory_df`. The CLI layer
  surfaces these as clean `ClickException`s (D6).
- **Non-integer `track_id`:** the adapter validates and raises the friendly
  `ValueError` above rather than letting `astype(int)` throw a cryptic pandas
  error ("raise rather than silently corrupt", matching `_validate_integer_identity`).

## D3 — output layout & the #238 clobber

Both `_io` writers emit a **fixed-name `run_metadata.json`** in the CSV's parent
dir, so two CSVs in one dir collide (#238). PR #17 **sidesteps via distinct
subdirs** (does not fix the underlying `_io` API foot-gun):

```
<output-dir>/
├── per_plant/
│   ├── traits_per_plant.csv
│   ├── traits_per_plant.units.json
│   └── run_metadata.json
├── per_genotype/
│   ├── traits_per_genotype.csv
│   ├── traits_per_genotype.units.json
│   └── run_metadata.json
└── plots/
    ├── *.png
    └── plots_metadata.json
```

`save_plots(inputs, out_dir=<output-dir>, …)` creates the `plots/` leaf. **Zero
changes to the foundation `_io` contract.** The proposal references #238 and
states explicitly that PR #17 sidesteps but does **not** close it.

**Conditional subdirs.** `per_genotype/` is written only when aggregation runs —
`--no-aggregate` omits it entirely (like `--no-plots` omits `plots/`). So the tree
above is the *full* (default, genotype-resolvable) shape; `--no-aggregate` yields
`per_plant/` + `plots/` only.

**The CLI creates `per_plant/` (and `per_genotype/` when aggregating) itself (I1).**
`pipeline.save()` and `write_per_genotype_csv` require their parent dir to *already
exist* — `save()` raises `FileNotFoundError` otherwise (pipeline.py:228-233). So
the CLI `mkdir(parents=True, exist_ok=True)` the `per_plant/` leaf always, and the
`per_genotype/` leaf only on the aggregation path, **before** the writes. (If skipped, the happy path raises
`FileNotFoundError`, which D6 would then mis-surface as a clean exit-1 "domain
error" — masking a CLI bug. Creating the leaves up front prevents that.)

**Re-run overwrite is deliberate (documented).** Re-running `analyze` on the same
`.slp` writes to the same default dir and **overwrites** prior CSVs/sidecars/
`run_metadata.json` (`to_csv` truncates; `mkdir(exist_ok=True)`). This is the
intended behavior — it matches the repo convention (`compute_plant_traits` /
`*_for_groups` / `viewer` all overwrite) and keeps re-runs idempotent. Documented
in `--help` so it is a stated decision, not a silent footgun.

## D4 — `--no-plots` + `matplotlib.use("Agg")`

`save_plots` deliberately does not select a backend (plotting.py:613); the CLI,
which owns the process, makes it headless. `matplotlib.use("Agg")` must run before
`pyplot` is first imported, and nothing in the `analyze` path touches matplotlib
before plotting.

**Gated approach:**
- **Plots enabled (default):** `import matplotlib; matplotlib.use("Agg", force=True)`
  → lazy `from sleap_roots.circumnutation import plotting` → `save_plots(inputs,
  out_dir=<output-dir>, constants=None, enabled=True)`.
- **`--no-plots`:** skip the backend call *and* the `plotting` import entirely;
  `log.info("plotting disabled (--no-plots)")`. Truly avoids importing matplotlib.

**`force=True` is required for test-suite robustness (M).** Under `CliRunner` the
CLI runs *in-process* inside pytest, where an earlier plotting unit test has
already imported `pyplot`. Plain `use("Agg")` after `pyplot` is imported may not
switch the backend and emits a warning; `force=True` makes the switch
deterministic. It is a no-op cost in the real (fresh-process) CLI. (My earlier
"fresh process, force unneeded" rationale was wrong for the shared-interpreter test
environment.)

The `save_plots(enabled=…)` param is still exercised directly in unit tests
(independent of the CLI gating).

## D5 — testing the click CLI

**Primary happy-path mechanism: synthetic tracked `.slp` round-trip.** All
`*.slp` are LFS-tracked, so a fresh CI checkout may have none → real-fixture
tests are skipif-guarded. To carry the coverage gate on every machine, a helper
`_make_synthetic_tracked_slp(tmp_path, *, n_tracks, n_frames, noise_sigma_px)`
builds a tracked `.slp` and the CLI/adapter load it through the *real*
`Series.load → get_tracked_tips` path — the "round-trip through a synthetic `.slp`"
mechanism (not "construct `Labels` in memory and pass to the pipeline").

**Helper construction — the load-bearing detail (B3).** `generate_trajectory`
returns a *DataFrame* (one row per `(track, frame)` with `tip_x`/`tip_y`), **not**
`Labels`. The house idiom (test_tracked_tip_pipeline.py:351-371,
test_series.py:709-798) builds the `.slp` like so:
1. `Image.fromarray(np.zeros((H,W), uint8)).save(tif_path, dpi=(72,72))` (PIL) →
   `video = sio.Video.from_filename(tif_path)`. **A real TIFF is mandatory** —
   `sio.save_slp` needs a `Video` to attach to each `LabeledFrame`. (PIL is already
   a test-path dependency via this idiom.)
2. `skeleton = sio.Skeleton([sio.Node("tip")])`; one `sio.Track(name="track_<i>")`
   per track.
3. For each row, `sio.Instance.from_numpy(np.array([[tip_x, tip_y]]), skeleton,
   track)`; group instances into `sio.LabeledFrame(video, frame_idx, instances)`.
4. `sio.save_slp(sio.Labels(labeled_frames, skeletons=[skeleton], videos=[video],
   tracks=[...]), slp_path)`.
**No image backend is exercised on read** — `Series.load`/`get_tracked_tips` read
only point coordinates, so the synthetic `.slp` round-trips identically on all 3
OSs.

**Minimum frames + noise (B4).** Tier-1 CWT requires `len(x) ≥ MIN_FRAMES_REQUIRED`
(= 9 at defaults; `temporal_cwt._validate_x` raises `ValueError` below it), and
`compute_traits` runs Tier 1 + Tier 3c. So the synthetic happy-path fixture uses
**`n_frames ≥ 64`** (matching the existing `_track_rows(n_frames=64)` pipeline-test
convention) and **`noise_sigma_px > 0`** (theory.md §8 mandates noise for Layer-1
pipeline validation; a noise-free synthetic risks degenerate QC/ridge behavior).
`n_tracks ≥ 2` so per-genotype aggregation has >1 plant.

**`tests/test_circumnutation_cli.py` (CliRunner):**
- `--help` on group + `analyze` → exit 0.
- Missing `--cadence-s` / `--sample-uid` → exit 2 ("Missing option").
- Nonexistent `SLP_PATH` → exit 2 (`Path(exists=True)`).
- Bad `--cadence-s` (`0`, negative, non-numeric) → non-zero exit, clean one-line
  message (D6), no traceback.
- **Happy path on synthetic `.slp`** (**with `--genotype`** supplied) → exit 0;
  assert the D3 output tree; per-plant row count = `n_tracks`; identity columns
  populated; per-genotype CSV has the expected genotype group(s).
- **Genotype-missing hard error** → bare invocation (no `--genotype`, no
  `--metadata-csv`, no `--no-aggregate`) → exit 1, clean `ClickException` naming
  `--genotype` / `--metadata-csv` / `--no-aggregate`; no partial output tree left
  behind.
- **`--no-aggregate` without genotype** → exit 0; `per_plant/` + `plots/` written,
  **no `per_genotype/`** dir; succeeds despite absent genotype.
- `--metadata-csv` (synthetic CSV) → output CSV carries `genotype`/`treatment`;
  `--genotype` flag overrides CSV value (assert override logged via `caplog`).
- `--no-plots` → no `plots/` dir; exit 0.
- **Real plate-001 e2e (skipif-guarded)** → with `--metadata-csv
  fixture_metadata.csv` + `--sample-uid plate_001`, output carries
  `genotype=Nipponbare`, `treatment=MOCK`; full tree; PNGs exist + non-empty.
  **No pixel baselines.**

**`tests/test_circumnutation_adapters.py`:**
- `series_to_inputs` on synthetic Series → 8 identity columns correct, `track_id`
  int, trajectory intact.
- `"track_<n>"` → int; non-integer track name (`"track_2a"`) → clear `ValueError`;
  **interior-`track_` name** (`"track_track_1"`) coerces correctly under the
  prefix-anchored strip (would corrupt under the blueprint's global `.replace`).
- Metadata precedence: CSV value used; flag overrides CSV (+ `caplog` INFO);
  flag fills a *blank* CSV cell → **no** spurious override log; neither → NaN.
- Malformed `--metadata-csv` → clear `ValueError`, not a pandas traceback (I3).
- `--timepoint` from a numeric CSV cell (`0`) → string-normalized `"0"` (B2).
- Real plate-001 (skipif) → matches fixture metadata.

**Coverage:** local `--cov` hits the Windows numpy-reload bug → coverage is
CI-enforced; locally we run via `CliRunner` without `--cov`. Synthetic-`.slp`
tests provide the non-skipped coverage (≥84% project, aim ≥90% new code).

## D6 — error handling / UX

Mirrors `viewer/cli.py`: catch the known domain exception, convert, no catch-all.

- **Click built-ins** → exit 2: missing required options, nonexistent `SLP_PATH`.
- **Pipeline body** (`series_to_inputs` → `CircumnutationInputs` →
  `compute_traits` → `save` → `aggregate` → `write` → `save_plots`) wrapped in one
  `try/except (ValueError, FileNotFoundError)` → `raise click.ClickException(str(e))`
  → **exit 1**, clean message, no traceback. The core already raises descriptive,
  field-naming `ValueError`s, so `str(e)` is user-meaningful.
- **No broad `except Exception`** — unanticipated errors surface as tracebacks
  (debuggable bugs).
- **`cadence_s` (and `R_px`) validated only in `CircumnutationInputs`** — the
  single source of truth (rejects ≤0, NaN, inf, non-numeric, bool) — *not* also
  at the click layer, so every bad-cadence case yields the same exit code (1) and
  messaging.
- **Genotype-missing hard error** (your-call 1, gated): when aggregation is on
  (no `--no-aggregate`), after identity resolution, if any plant's `genotype` is
  NaN, the CLI raises `click.ClickException` (exit 1) naming `--genotype` /
  `--metadata-csv` / `--no-aggregate` — *before* writing any output, so no partial
  tree is left behind. With `--no-aggregate` the check is skipped (genotype not
  required). A deliberate CLI-layer guard, distinct from the core's `ValueError`s.
- **Adapter normalizes its raisers to `ValueError` (I3).** The two non-`ValueError`
  leak surfaces — a malformed `--metadata-csv` (`pd.read_csv` → pandas error) and a
  non-integer track name (`astype(int)` → could surface a pandas error) — are
  caught *inside the adapter* and re-raised as clear `ValueError`s, so the single
  `try/except (ValueError, FileNotFoundError)` at the CLI catches everything
  user-facing. Genuinely-unexpected types still surface as tracebacks by design.

## D7 — provenance: threading `input_path` through `save`

- The CLI computes `slp_path = Path(SLP_PATH).resolve()` once and passes
  `str(slp_path)` as `input_path` into the per-plant writer `pipeline.save(
  per_plant/…csv, …, input_path=slp_path)`, and — **only on the aggregation path
  (not under `--no-aggregate`)** — into the per-genotype writer via
  `gather_run_metadata(input_path=slp_path, run_id=inputs.run_id, constants=None,
  cadence_s=inputs.cadence_s, R_px=inputs.R_px)` → `write_per_genotype_csv`.
  **Resolved-absolute** (a relative path is meaningless in `run_metadata.json`).
- Both subdir `run_metadata.json` files record identical run provenance (same
  `input_path`/`cadence_s`/`R_px`/`run_id`/constants snapshot; only the per-write
  wall-clock `timestamp` differs, within ms). **Deliberate (your-call 3): keep the
  blessed `pipeline.save()` API** for per-plant (it builds provenance internally and
  has the parent-dir guard) plus a second `gather_run_metadata` for per-genotype —
  accepting two `git rev-parse` subprocesses and the ms-apart timestamps as benign,
  rather than bypassing `save()` to share one snapshot. The provenance content is
  identical; only the stamp instant differs.
- **Known gap (follow-up, not fixed here):** `save_plots` takes `inputs`, not
  `input_path`, so `plots_metadata.json` cannot record the source `.slp` path.
  Candidate follow-up: "thread `input_path` into `save_plots` provenance."

## D8 — CC-3 (no `--px-per-mm`) + CC-9 logging (`-v`)

- **CC-3:** no `--px-per-mm`; the `analyze` `--help`/docstring states outputs are
  pixel-native and points to `sleap_roots.circumnutation.units.convert_to_mm` for
  mm conversion. Calibration stays downstream.
- **CC-9:** `-v` is a **count** flag → `0 = WARNING` (quiet default), `-v = INFO`
  (per-plate progress), `-vv = DEBUG` (per-plant detail), matching CC-9's
  INFO/DEBUG split. A `_configure_logging(verbose)` helper maps count → level and
  calls `logging.basicConfig(level=…, format="%(levelname)s %(name)s: %(message)s",
  stream=sys.stderr)`. **Logs → stderr**, keeping stdout for the final
  `click.echo` summary. The D2c override notice is an INFO log.
- **Tracebacks decoupled from `-v`** — known domain errors always become a clean
  `ClickException`; only unanticipated exceptions surface tracebacks. `-v`
  controls log verbosity only.

## OpenSpec deltas (for the proposal phase)

- **MODIFY** the "Package layout" requirement: ADD the new modules (`cli.py` +
  `adapters.py`) to the enumerated impl list; stub table unchanged at 1
  (`parametric`, deferred). The exact before/after impl count must be read off the
  actual Package-layout requirement during `/openspec:proposal` (we add two
  modules, so confirm whether both are roadmap-referenced module names).
- **ADD** a "Circumnutation analyze CLI" requirement (the `analyze` command +
  `series_to_inputs` adapter contract, including the metadata precedence, output
  tree, error contract, and pure-pixel `--help`).

## Verification gates

`uv run pytest tests/ -q` green · ≥84% project coverage (CI) / aim ≥90% new ·
`black --check` · `pydocstyle --convention=google sleap_roots/circumnutation/` ·
`uv lock --check` (click already a dep → no-op) · `uv run mkdocs build` ·
`npx openspec validate add-circumnutation-cli --strict` · tri-OS CI green.

## Relevant issues

Parent #197 · #238 (run_metadata clobber — sidestepped, referenced, not closed) ·
#222 (unit suffixes) · #241 (save_plots MCP-serializable follow-up) · #230 (L_gz,
informational). The `Series → CircumnutationInputs` adapter lands here; PR #18
(user guide) consumes the CLI.

## Critical review reconciliation (2026-06-24)

Three adversarial reviewers (API-fidelity, design-soundness, test/reproducibility)
reviewed the first draft. API-fidelity came back clean (every cited API exists and
behaves as described). Design + test findings and their resolutions:

| # | Finding | Resolution |
|---|---|---|
| B1 | `get_metadata` returns `np.nan` identically for no-CSV / missing-col / empty-cell → override-log unimplementable | Concrete `pd.notna`-keyed precedence + log rule (D2c); flag wins always, log only on real shadow |
| B2 | `--timepoint` (str) vs CSV `timepoint` (numeric) clash; `Series.timepoint` raises on non-numeric | `timepoint` is object/string; adapter reads raw `get_metadata("timepoint")` + `str()`-normalizes (D2) |
| B3 | Synthetic-`.slp` helper omits the mandatory PIL TIFF/Video build step; `generate_trajectory` returns a DataFrame not `Labels` | D5 helper spec rewritten with the full TIFF→Video→`Instance.from_numpy`→`save_slp` idiom |
| B4 | No min `n_frames`; Tier-1 CWT raises below 9 frames | D5 fixture pinned to `n_frames ≥ 64`, `noise_sigma_px > 0`, `n_tracks ≥ 2` |
| I1 | CLI must `mkdir` `per_plant/` + `per_genotype/` leaves; `save()` raises `FileNotFoundError` otherwise | D3 specifies creating both leaves before writing |
| I2 | "Formalizes `_load_plate001_inputs`" conflicts with NaN-defaults (blueprint hardcodes `"none"`/`"T0"`) | D2 clarifies: generalizes the *mechanical* transform, replaces the hardcoded test literals with CSV/flag/NaN sourcing |
| I3 | Malformed `--metadata-csv` → pandas error leaks past `except (ValueError, FileNotFoundError)` | Adapter wraps the metadata read → clear `ValueError` (D2, D6) |
| M | `matplotlib.use("Agg")` without `force=True` is a `CliRunner`-in-process hazard | D4 → `use("Agg", force=True)`; stale contradicting paragraph removed |
| your-call 1 | All-NaN `genotype` default produces a vacuous per-genotype CSV | Per-genotype aggregation is a **gated step** (`--no-aggregate`, parallels `--no-plots`); default-on + genotype NaN → **hard error** (exit 1, before output); `--no-aggregate` → per-plant + plots only, genotype not required. No auto-skip-and-warn. (D1, D2, D3, D5, D6) |
| your-call 2 | Re-run same-`.slp` silent overwrite | Overwrite + documented in `--help` (deliberate; matches repo convention) (D3) |
| your-call 3 | Double `gather_run_metadata` (2 git subprocs, ms-apart stamps) | Keep blessed `save()` API; accept as benign (D7) |
| M3 | Prefix-anchored strip needs an interior-`track_` test | Added `"track_track_1"` adapter test (D5) |
