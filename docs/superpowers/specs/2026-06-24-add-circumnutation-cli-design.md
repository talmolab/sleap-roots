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
| `--timepoint` | `str` | default NaN (override/fallback) |
| `--plate-id` | `str` | default NaN |
| `--genotype` | `str` | default NaN (override/fallback) |
| `--treatment` | `str` | default NaN (override/fallback) |
| `--r-px` | `float` | default `None` |
| `--run-id` | `str` | default `None` |
| `--no-plots` | flag | gates plotting + matplotlib import (D4) |
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

**Transform (formalizes `_load_plate001_inputs`):**
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

**Metadata precedence — CSV-as-source, flags-as-override (D2c = option A):**
- If `--metadata-csv` is given, `genotype`/`treatment`/`timepoint` are pulled via
  `Series.get_metadata` (the CC-4 "populated from `Series` metadata if available"
  path).
- An explicit `--genotype`/`--treatment`/`--timepoint` flag **overrides** the CSV
  value for that field, and the override is **logged at INFO** (CC-9) so a flag
  silently shadowing a CSV value is never invisible.
- With neither CSV nor flag, the field is `NaN`.

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

## D4 — `--no-plots` + `matplotlib.use("Agg")`

`save_plots` deliberately does not select a backend (plotting.py:613); the CLI,
which owns the process, makes it headless. `matplotlib.use("Agg")` must run before
`pyplot` is first imported, and nothing in the `analyze` path touches matplotlib
before plotting.

**Gated approach:**
- **Plots enabled (default):** `import matplotlib; matplotlib.use("Agg")` → lazy
  `from sleap_roots.circumnutation import plotting` → `save_plots(inputs,
  out_dir=<output-dir>, constants=None, enabled=True)`.
- **`--no-plots`:** skip the backend call *and* the `plotting` import entirely;
  `log.info("plotting disabled (--no-plots)")`. Truly avoids importing matplotlib.

Plain `use("Agg")` (no `force=True` — fresh CLI process, set before any pyplot
import). The `save_plots(enabled=…)` param is still exercised directly in unit
tests.

## D5 — testing the click CLI

**Primary happy-path mechanism: synthetic tracked `.slp` round-trip.** All
`*.slp` are LFS-tracked, so a fresh CI checkout may have none → real-fixture
tests are skipif-guarded. To carry the coverage gate on every machine, a helper
`_make_synthetic_tracked_slp(tmp_path, n_tracks, n_frames)` builds a tracked `.slp`
via the house idiom (`synthetic.generate_trajectory` → single-node `"tip"`
skeleton → one `Track(name="track_<i>")` per track → `Instance.from_numpy` →
`sio.Labels` → `sio.save_slp`) and the CLI/adapter load it through the *real*
`Series.load → get_tracked_tips` path. This is the "round-trip through a synthetic
`.slp`" mechanism (not "construct `Labels` in memory and pass to the pipeline").

**`tests/test_circumnutation_cli.py` (CliRunner):**
- `--help` on group + `analyze` → exit 0.
- Missing `--cadence-s` / `--sample-uid` → exit 2 ("Missing option").
- Nonexistent `SLP_PATH` → exit 2 (`Path(exists=True)`).
- Bad `--cadence-s` (`0`, negative, non-numeric) → non-zero exit, clean one-line
  message (D6), no traceback.
- **Happy path on synthetic `.slp`** → exit 0; assert the D3 output tree; row
  count = `n_tracks`; identity columns populated.
- `--metadata-csv` (synthetic CSV) → output CSV carries `genotype`/`treatment`;
  `--genotype` flag overrides CSV value (assert override logged via `caplog`).
- `--no-plots` → no `plots/` dir; exit 0.
- **Real plate-001 e2e (skipif-guarded)** → `genotype=Nipponbare`,
  `treatment=MOCK` from `fixture_metadata.csv`; full tree; PNGs exist + non-empty.
  **No pixel baselines.**

**`tests/test_circumnutation_adapters.py`:**
- `series_to_inputs` on synthetic Series → 8 identity columns correct, `track_id`
  int, trajectory intact.
- `"track_<n>"` → int; non-integer track name → clear `ValueError`.
- Metadata precedence: CSV value used; flag overrides CSV (+ logged); neither → NaN.
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

## D7 — provenance: threading `input_path` through `save`

- The CLI computes `slp_path = Path(SLP_PATH).resolve()` once and passes
  `str(slp_path)` as `input_path` into **both** provenance writers: `pipeline.save(
  per_plant/…csv, …, input_path=slp_path)` and, for the per-genotype artifact,
  `gather_run_metadata(input_path=slp_path, run_id=inputs.run_id, constants=None,
  cadence_s=inputs.cadence_s, R_px=inputs.R_px)` → `write_per_genotype_csv`.
  **Resolved-absolute** (a relative path is meaningless in `run_metadata.json`).
- Both subdir `run_metadata.json` files record identical run provenance (same
  `input_path`/`cadence_s`/`R_px`/`run_id`/constants snapshot; only the per-write
  wall-clock `timestamp` differs, within ms). Uses the public `save()` +
  `gather_run_metadata`/`write_per_genotype_csv` APIs rather than bypassing them.
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

- **MODIFY** the "Package layout" requirement: impl count 12 → 13 by ADDITION
  (`cli.py` + `adapters.py`); stub table unchanged at 1 (`parametric`, deferred).
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
