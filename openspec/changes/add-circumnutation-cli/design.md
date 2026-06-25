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
  `input_path` in its `plots_metadata.json` (tracked as follow-up #245 — it is NOT
  #241, which is scoped to MCP-serializable `Result` views).

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
- **Output tree — distinct subdirs + a canonical top-level provenance file.**
  `per_plant/`, `per_genotype/`, `plots/`. Each `_io` writer's fixed-name
  `run_metadata.json` lives in its own dir, so #238's clobber is sidestepped with
  zero `_io` contract change. The CLI `mkdir`s the CSV leaves before writing (the
  `_io` writers require the parent to exist). References #238; does not close it.
  The CLI ALSO writes a **top-level `<output-dir>/run_metadata.json`** (the shared
  snapshot) — `save_plots` writes `plots/plots_metadata.json` with a hardcoded
  `run_metadata_ref: "../run_metadata.json"` (plotting.py:521), so without the
  top-level file the `per_plant/` subdir split would dangle that pointer and leave
  plots unprovenanced (caught in review round 2).
- **Headless plotting.** `--no-plots` skips the matplotlib import entirely;
  otherwise `matplotlib.use("Agg", force=True)` runs before `plotting` is imported.
  `force=True` is load-bearing on **every** invocation (not just under `CliRunner`):
  `sleap_roots.series` imports `matplotlib.pyplot` at module top, so by the time
  `analyze` runs, `Series.load` has already imported `pyplot` — a plain `use("Agg")`
  after import would warn and may not switch the backend.
- **Error contract.** Click handles usage errors (exit 2); the adapter normalizes
  its raisers to `ValueError`; the CLI wraps the pipeline body in `try/except
  (ValueError, FileNotFoundError)` → `ClickException` (exit 1); no catch-all
  (genuine bugs surface tracebacks). `cadence_s`/`R_px` validated only in
  `CircumnutationInputs` (single source of truth). Mirrors `viewer/cli.py`.
- **Provenance — full traceability (supersedes the brainstorm's "keep `save()`").**
  Per review, `run_metadata.json` must record not just the `.slp` path but the
  `--metadata-csv` path and a per-field `identity_source` map (`flag` /
  `metadata_csv` / `default`), so a reader knows where each row-identity field came
  from. The adapter computes this (it does the resolution) and returns it alongside
  the inputs. To put a single complete snapshot in all three sidecars (top-level +
  per-plant + per-genotype), the CLI assembles
  run-metadata **once** via `gather_run_metadata` (extended with `metadata_csv_path`
  + `identity_source`) and writes via `write_per_plant_csv` / `write_per_genotype_csv`
  directly — NOT through `CircumnutationPipeline.save()` (which gathers internally
  and can't carry the CLI's identity provenance). This reverses your-call-3 (the
  brainstorm's "keep `save()`") for a good reason: a single shared snapshot is the
  only way to make identity provenance complete and the three sidecars (top-level +
  per-plant + per-genotype) byte-identical (same `timestamp` too).
  `gather_run_metadata` gains three optional, backward-compatible params
  (`metadata_csv_path`, `metadata_csv_sha256`, `identity_source`; existing callers
  write `null`). The `identity_source` map is **total** over the six
  adapter-populated fields with a closed label set — `flag` / `metadata_csv` /
  `default` (the `series` stem) / `absent` (NaN fallback); `default` and `absent`
  are distinct so a reader can tell a derived value from a missing one (review round
  2). `metadata_csv_sha256` hashes the CSV bytes so the join is verifiable even
  though the CSV is an external mutable input recorded by reference (its contents
  aren't snapshotted by value like `cadence_s`). Whole-file hashing is the v1
  primitive: it flags edits to *unrelated* rows as drift (conservative — never a
  false "unchanged"); a tighter resolved-row hash is a possible future refinement.
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

## Implementation reconciliation (2026-06-25)

Two spec scenarios were corrected during TDD implementation (the spec deltas were
updated to match the implemented, correct behavior):

### Why `track_track_1` raises instead of yielding `1`
The round-1 adapter scenario claimed the interior-`track_` name `"track_track_1"`
"yields integer `1`" under a prefix-anchored strip. That is self-contradictory:
`removeprefix("track_")` leaves `"track_1"`, which is **not** a pure integer. The
*correct* behavior — and the program's "raise rather than silently corrupt" rule —
is to **raise** `ValueError` (a global `.replace` would silently corrupt it to `1`;
the anchored strip refuses to). The scenario now asserts it raises.

### Why the "partial genotype" scenario was removed
The adapter fills `genotype` **uniformly** across all tracks of a run
(`df["genotype"] = genotype_val`), and the `--metadata-csv` join keys on a single
`sample_uid` per `Series` — so one plate resolves exactly one genotype. A "partial"
genotype (some tracks resolved, some NaN) is therefore unreachable in v1. The
existing "genotype unresolved with aggregation on" scenario (all-NaN ⟺ any-NaN under
uniform fill) covers the real case; the redundant partial scenario was dropped.

### Pre-PR subagent-review hardening (2026-06-25)
The 5-subagent `/review-pr` self-review (no BLOCKING; code-quality 8/10, testing
100% line+branch on the new modules) surfaced four worthwhile hardenings, applied
before opening the PR:
- **Blank genotype no longer defeats the gate.** `--genotype "  "` / a whitespace
  CSV cell now resolves to `"absent"` (NaN) — `_resolve_field` treats a blank
  (`None`/empty/whitespace) flag *or* CSV cell as not-supplied — so the genotype
  hard error fires instead of forming a degenerate empty-string genotype group.
- **`track_id` strict-digit match.** `_coerce_track_id` now requires
  `re.fullmatch(r"\d+", remainder)`, rejecting names `int()` would silently
  mis-coerce (`"track_1_2"`→12, `"track_-1"`, `"track_ 1"`, `"track_+1"`).
- **Unmatched-`sample_uid` WARNING.** When `--metadata-csv` is supplied but
  `sample_uid` matches no `plant_qr_code` row, the adapter logs a WARNING (the join
  would otherwise silently ship all-NaN metadata for a mistyped `--sample-uid`).
- **Identity-column dtype stability.** The six string identity columns are forced to
  `object` dtype even when all-NaN (was float64 when NaN), via `_PROVENANCE_FIELDS`.
The spec scenarios were extended to lock all four.

### Minor implementation notes (not spec deviations)
- The CLI loads the `.slp` via the `Series.load(primary_path=...)` slot (matching
  the `_load_plate001_inputs` blueprint); `get_tracked_tips` auto-detects the single
  populated path. A future `--root-type` could generalize to lateral/crown `.slp`.
- The default `--output-dir` derives from `Path.stem`, which strips only the final
  `.slp` — a dotted name like `plate_001_greyscale.tracked_proofread.slp` yields
  `plate_001_greyscale.tracked_proofread_circumnutation/`. Users pass `--series-name`
  or `-o` for a cleaner name. Documented in `--help`.
- `click` 8.3's `CliRunner` separates `result.stderr` but `result.output` is the
  combined stream (no `mix_stderr` param); the `-v` test asserts on `result.stderr`.
