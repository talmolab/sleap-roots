## Context

PR #17 (roadmap row 17) adds the user-facing CLI for the circumnutation pipeline
and the `Series → CircumnutationInputs` adapter the roadmap says lands here. Full
decision record (D1–D8) and the adversarial-review reconciliation table live in the
brainstorm doc: `docs/superpowers/specs/2026-06-24-add-circumnutation-cli-design.md`.
This `design.md` captures the decisions a reviewer needs without re-deriving them.

## Goals / Non-Goals

- **Goals:** one `circumnutation analyze <slp>` command composing the whole
  pipeline; a reusable, click-free `series_to_inputs` adapter; honest provenance;
  CC-3 (pure-pixel) and CC-9 (logging) compliance; strong non-LFS test coverage.
- **Non-Goals:** no `--px-per-mm`/calibration surface (CC-3 — downstream
  `convert_to_mm`); no `ConstantsT` override surface in v1; no
  `trajectory_per_plant/<id>.csv` dumps (pipeline doesn't emit them); does NOT fix
  #238 (sidestepped via layout); does NOT widen `save_plots` to record
  `input_path` (#241 follow-up).

## Decisions

- **Command shape:** `circumnutation` group + `analyze` command, registered via
  `main.add_command(circumnutation)` — mirrors `viewer/cli.py`; future subcommands
  (`aggregate`, `plot`) slot in without restructuring.
- **`--sample-uid` required (no synthesis):** it is the stable QR/sample id and the
  metadata-CSV join key (`Series.get_metadata` joins `plant_qr_code == sample_uid`),
  so it must be user-supplied and stable (reproducibility + pre-authored CSVs). A
  random UUID would break both. `--series-name` defaults to the `.slp` stem.
  Honors CC-4's "row-identity stays explicit, a deliberate caller decision."
- **Adapter in `adapters.py`** (not `_types.from_series` / a classmethod): keeps
  `_types.py` free of a heavy `Series`/`sleap_io` import and gives the bridge a
  named, click-free, independently-testable home.
- **Metadata precedence — CSV-as-source, flags-as-override, `pd.notna`-keyed.**
  `Series.get_metadata` returns `np.nan` indistinguishably for no-CSV / missing /
  empty, so the override-log fires only when `flag is not None and pd.notna(csv) and
  str(flag) != str(csv)`. `timepoint` is read as the raw cell (not the coercing
  `Series.timepoint` property) and `str`-normalized to object dtype.
- **Gated per-genotype aggregation (`--no-aggregate`).** Aggregation is separable
  (`compute_traits`/`save`/`save_plots` never need genotype). Default-on + genotype
  unresolved → **hard `ClickException` before any output** (never a vacuous
  `(NaN,NaN,NaN)` per-genotype CSV, and explicitly NOT auto-skip-and-warn — per
  Elizabeth's directive). `--no-aggregate` opts out (per-plant + plots only).
- **Output tree — distinct subdirs.** `per_plant/`, `per_genotype/`, `plots/`. Each
  `_io` writer's fixed-name `run_metadata.json` lives in its own dir, so #238's
  clobber is sidestepped with zero `_io` contract change. The CLI `mkdir`s the
  CSV leaves before writing (`save()` requires the parent to exist). References
  #238; does not close it.
- **Headless plotting.** `--no-plots` skips the matplotlib import entirely;
  otherwise `matplotlib.use("Agg", force=True)` runs before `plotting` is imported.
  `force=True` because `CliRunner` runs in-process where `pyplot` may already be
  imported by an earlier test.
- **Error contract.** Click handles usage errors (exit 2); the adapter normalizes
  its raisers to `ValueError`; the CLI wraps the pipeline body in `try/except
  (ValueError, FileNotFoundError)` → `ClickException` (exit 1); no catch-all
  (genuine bugs surface tracebacks). `cadence_s`/`R_px` validated only in
  `CircumnutationInputs` (single source of truth). Mirrors `viewer/cli.py`.
- **Provenance.** Resolved-absolute `.slp` path threaded as `input_path` into
  `save()` and (aggregation path) the per-genotype `gather_run_metadata`. Keep the
  blessed `save()` API (two `gather_run_metadata` calls, ms-apart timestamps —
  benign; content identical).
- **Testing — synthetic-`.slp` round-trip as the primary happy path.** All `*.slp`
  are LFS-tracked → real-fixture tests are skipif-guarded. A
  `_make_synthetic_tracked_slp` helper (PIL TIFF → `sio.Video` →
  `Instance.from_numpy` → `save_slp`) gives a non-skipped, every-OS round-trip
  through the real `Series.load → get_tracked_tips` path; `n_frames ≥ 64`,
  `noise_sigma_px > 0` so the Tier-1 CWT (min 9 frames) runs.

## Risks / Trade-offs

- **Two new modules in one PR** (`cli` + `adapters`) — first time the program adds
  two impl modules at once (12 → 14). Mitigation: the Package-layout MODIFIED
  scope-note documents it explicitly; both gain callability scenarios.
- **`--no-aggregate` + genotype-required is a UX shift** — the bare invocation now
  hard-errors. Mitigation: the error message names all three escape hatches;
  `--metadata-csv` (with the real fixture) is the easy default for real data.
- **#238 sidestepped, not fixed** — the `_io` fixed-name foot-gun persists for any
  caller writing two CSVs to one dir. Mitigation: documented + referenced; the CLI's
  layout never triggers it.

## Migration Plan

Additive only — no existing behavior changes. New modules + one registration line.
Rollback = revert the PR; no data migration.

## Open Questions

None outstanding — all D1–D8 decisions and the three reviewer-surfaced calls
(genotype hard-error → `--no-aggregate`, re-run overwrite, provenance) are resolved
in the brainstorm doc and reflected here.
