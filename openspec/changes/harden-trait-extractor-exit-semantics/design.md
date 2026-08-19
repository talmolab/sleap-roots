## Context

sleap-roots#259 closes the "PID-1 / SIGTERM (deferred)" note and the exit-code "Open Question"
left by the archived `add-trait-extractor-image` design, now that A4's request-driven-pipeline
design (`sleap-roots-pipeline` §8) specifies what it needs from the driver at runtime. The issue is
explicitly paired with `sleap-roots-predict#26` — predict's identical hardening — with the
instruction to reconcile the exit-code and empty-input policy **uniformly across both producers**.

At the time this proposal was first drafted, `sleap-roots-predict#26` had no branch, PR, or commit
(checked via `gh pr list` / `gh api .../branches` against `talmolab/sleap-roots-predict`) — so there
was no existing decision to defer to. This design made the call for `trait_extractor` first and
stated it plainly so the predict-side implementation would adopt the same convention instead of
independently inventing one that then had to be reconciled after the fact.

**Update (2026-08-19, during this proposal's own review):** `sleap-roots-predict`'s session picked
up work in parallel and, as of the round-2/round-3/round-4 cross-repo checks referenced below,
verified directly against its actual (by-then committed) source: a narrow `except
(FileNotFoundError, ValueError): log; raise` wrapper and a `threading.Event`-based SIGTERM handler
checked at the top of `run_batch`'s per-scan loop are real, shipped code in that repo, not
anticipated/invented detail — confirmed by reading `sleap_roots_predict/__main__.py` and
`sleap_roots_predict/batch.py` directly, and by `git log` showing the behavior committed. The two
proposals converged on the identical `0`/`3`/`1`/`2`-reserved/`143` convention.

## Decisions

### Decision 1: Three exit codes, mapped onto the existing isolation boundary

**What:** `0` = full success, `3` = partial (per-scan failures caught inside `extract_batch`'s own
loop), `1` = crash (exception escaped `extract_batch` — invalid manifest, empty input, or an actual
bug). `2` is deliberately left alone.

**Why `3` and not `2` for partial (revised after review):** the first draft of this proposal picked
`2` for "partial." `review-openspec`'s TDD and spec-quality passes both independently caught, and
empirically verified, that `argparse.ArgumentParser.error()` already calls `sys.exit(2)` on a CLI
usage error (`python -m trait_extractor` with missing/extra args exits `2` **today**, before
`extract_batch` ever runs). Reusing `2` for "partial" would make a misconfigured Argo template
`args:` substitution — a real pod-level misconfiguration — indistinguishable from "some scans
isolated-failed, don't retry," which is the exact wrong signal and undermines the whole point of
this proposal for the *other* code (`1`) it was trying to protect. `3` avoids every code already in
use by something else: `0`/`1` (Python/shell defaults), `2` (argparse), `130`/`143` (`128+SIGINT`/
`128+SIGTERM`, conventional and used by Decision 3 below).

**Why this split and not something else:**
- The code already has a real boundary between "isolated per-scan failure" and "escaped exception":
  `extract_batch`'s discovery loop wraps each scan in `except Exception` and records the outcome in
  `BatchResult.failed`; anything that isn't caught there (e.g. `load_run_manifest` raising on an
  invalid top-level manifest) propagates to `__main__.main()` uncaught. The exit-code convention
  reuses that existing structural boundary rather than inventing a parallel failure taxonomy.
  Anything reaching `main()` as an uncaught exception is, by construction, *not* one of the isolated
  per-scan failures the "isolated poison scan" language in the issue is about.
- `3` was picked over reusing `1` for partial, specifically so Argo (or any shell script reading the
  container's exit code) can distinguish the two outcomes without parsing logs. `2` (argparse usage
  errors — see above), `143` (`128+15`, reserved for the SIGTERM path — see Decision 3), and `130`
  (`128+2`, conventionally `SIGINT`) are all avoided so none of the four families collide.
- This is a **driver-only** decision. It intentionally does not implement the A4 Argo template's
  `retryStrategy`/`continueOn` reaction to these codes — the issue itself says "distinct exit codes
  so Argo can distinguish ... or the A4 template handling it via `continueOn`", and explicitly
  scopes the template-side work to the pipeline repo. Emitting the distinct codes here is a
  prerequisite either way (a `continueOn` policy still needs *some* signal to key off) and needs no
  template change to ship.

**Rejected alternative — always exit 0, encode failure only in envelope content:** would make a
broken batch look identical to a successful one at the process-exit boundary, which is exactly the
silent-success failure mode `trait-extractor-image`'s own spec already treats as unacceptable (see
"A failing scan yields a non-zero container exit").

**For `sleap-roots-predict#26`:** adopt the same three codes with the same meaning (`0` full
success, `3` partial/isolated-scan-failures, `1` crash, `2` left alone for argparse usage errors —
verify predict's own CLI parser also uses `argparse` before assuming this applies unchanged).
Predict's per-scan isolation loop in
`run_batch` is structurally the same shape (a top-level try/except per scan), so the same mapping
applies directly. Predict's issue additionally couples atomic writes with checksum-verified resume
(its item 3) — that coupling is orthogonal to the exit-code convention and doesn't change this
mapping.

### Decision 2: Empty-input guard raises, mapping to the "crash" code

**What:** `extract_batch` raises `RuntimeError` when `run_manifest.json` is absent (unscoped mode)
and zero `*.predictions.json` files are found anywhere under `input_dir`.

**Why raise instead of adding a new `BatchResult` state:** an invalid `run_manifest.json` already
raises uncaught today (`load_run_manifest` propagates a `pydantic.ValidationError`,
`test_invalid_manifest_aborts_batch`) — both are "the batch cannot meaningfully run at all"
conditions, discovered before any scan-level work starts, and both should abort before writing
anything. Reusing the same shape (raise, let `__main__` propagate it as exit `1`) keeps a single
"batch cannot run" failure family instead of adding a second, only-slightly-different one.

**Why this doesn't also need to change the scoped (manifest-present) path:** already handled. When
`run_manifest.json` is present, every `scan_key` in its `scan_keys` with no matching
`{scan_key}.predictions.json` is already recorded as a per-scan failure by the existing
"missing_scan_key" bookkeeping loop — `BatchResult.ok` is already `False` in that case (confirmed by
reading `extract_batch` and `test_manifest_declares_scan_key_with_no_predictions_json`). The actual
gap sleap-roots#259 §2 describes is narrower than "empty input in general": it's specifically the
no-manifest fallback path silently succeeding on zero discovered manifests.

**Log-quality parity with `sleap-roots-predict` (added after round-2 cross-repo review):** the
sibling proposal wraps its known staging-error types (`FileNotFoundError`, `ValueError`) in a
narrow `except ...: log; raise` specifically so the exit code stays the default `1` while an
operator still sees one clean logged line instead of a raw traceback. `trait_extractor`'s
`__main__.py` currently has no try/except around `extract_batch(...)` at all, so both this new
`RuntimeError` and the pre-existing `run_manifest.json` failure modes (`pydantic.ValidationError`,
`OSError`, `UnicodeDecodeError` — per `extract_batch`'s own docstring) would surface as a raw
traceback. Adding the equivalent `except (RuntimeError, pydantic.ValidationError, OSError,
UnicodeDecodeError, yaml.YAMLError) as exc: logger.error(...); raise` in `main()` (task 2.2) closes
this gap for parity — the exit code is unaffected, only the log line improves.

**Two implementation gaps found in round-3 review of this exact wrapper, both fixed in tasks.md
task 2.2:** (1) the bare `except (..., pydantic.ValidationError, ...)` clause needs `pydantic`
actually imported in `__main__.py`'s own namespace — it is not transitively available just because
`extractor.py` imports it elsewhere, and omitting it would raise `NameError` (masking the real
exception) the first time the clause is evaluated; (2) `extract_batch` also calls
`load_pipeline_cards()` before `load_run_manifest`, and that function's `yaml.safe_load` on the
packaged `pipeline_selection.yaml` can raise `yaml.YAMLError` — not in the original four-exception
tuple above, and not documented in `extract_batch`'s own `Raises:` section either. Narrow/defensive
(a static packaged file), but added to the tuple (and `import yaml` added alongside `import
pydantic`) for completeness at negligible cost.

### Decision 3: Driver-level SIGTERM handler, no entrypoint/image change

**What:** `main()` installs `signal.signal(signal.SIGTERM, handler)` before running the batch; the
handler logs and calls `sys.exit(143)`.

**Why a driver handler and not `tini`/`--init`:** changing the entrypoint would change the literal
`ENTRYPOINT ["python","-m","trait_extractor"]` that both the deployed `trait-extractor-image` spec
and A4's `args:` rewrite depend on (per the archived design's own reasoning for deferring this in
the first place). A driver-level handler needs no Dockerfile change.

**Why no atomic-write work is needed here (unlike predict):** `envelope.py`'s per-scan write is
already temp-file-then-rename, and the batch is idempotent on retry (skip-if-done via
`idempotency_key` comparison, from `add-run-manifest-consumption`). A SIGKILL following an
unhandled-then-forcibly-delivered SIGTERM can only ever abandon an in-flight temp file, never a
completed `{scan_key}.result.json` — so this really is pure preemption-latency, not the coupled
atomic-write+resume problem predict's item 3 describes.

**Why this handler exits immediately, unlike predict's scan-boundary wait (asymmetry noted after
round-2 cross-repo review):** predict's `SIGTERM` handler sets a `threading.Event` checked at the
top of `run_batch`'s per-scan loop, finishing whatever scan is already in flight before stopping —
because a single scan there is GPU inference with no safe mid-scan interrupt point, and a worse
(SIGKILL-then-truncated) outcome is otherwise possible for its `.slp`/manifest writes. This
driver's `_handle_sigterm` instead calls `sys.exit(143)` directly, with no scan-boundary check, for
two reasons specific to this driver: (1) a single scan's trait computation is CPU-bound and short
(no multi-second uninterruptible native call to wait out), so there's no real latency cost to
skip; (2) it doesn't need one — `envelope.py`'s atomic temp-then-rename means an interrupt at any
point, including mid-scan, can only ever abandon an unfinished temp file, never corrupt a
completed `{scan_key}.result.json` (this is the "no atomic-write work is needed here" point
above). Both handlers reach the same end state (prompt exit, `143`, no corrupted output); they
differ only in how much in-flight work is discarded, which is a direct consequence of each
driver's own per-scan cost and interruptibility, not an oversight to reconcile.

**Test platform note, and avoiding a timing-flaky test (revised after review):** CI runs
`ubuntu-22.04`, `windows-2022`, `macos-14` (`.github/workflows/ci.yml`). Sending a real `SIGTERM` to
a child process and asserting prompt termination is reliable on POSIX; Windows has no equivalent
signal-delivery semantics for `SIGTERM` (Python maps it to forceful termination there, bypassing the
handler).

The first draft of this proposal left the SIGTERM test's timing mechanism undecided ("insert an
artificial delay ... decide the simplest reliable mechanism during implementation"), which
`review-openspec`'s TDD pass correctly flagged as a CI-flake risk — racing a `SIGTERM` against real
scan-processing wall-clock time across three OS runners (macOS especially) is exactly the kind of
test that's green in dev and flaky in CI. Split into two tests instead, per the reviewer's concrete
proposal:
1. A pure, zero-flake, platform-independent unit test that calls `_handle_sigterm(signal.SIGTERM,
   None)` directly and asserts it raises `SystemExit` with `.code == 143` — no subprocess, no
   timing, no OS dependency. Not skipped on Windows (the handler function itself is portable; only
   *delivering* a real signal to a subprocess is not).
2. A subprocess smoke test using a **readiness signal, not a sleep**. Note: `__main__.py`'s `"ok
   "`/`"skip  "`/`"FAIL  "` lines are only printed *after* `extract_batch` returns for the **whole**
   batch (confirmed by reading `main()` — the print loop runs after the single blocking
   `extract_batch` call, not per-scan), so stdout is NOT a usable per-scan readiness signal here.
   `extract_batch` itself, however, already writes each scan's `{scan_key}.result.json` to
   `output_dir` as soon as that scan is computed (`write_envelope`, called once per scan inside the
   discovery loop, before moving to the next) — that's a real, already-existing per-scan signal. The
   test runs over an input with several (many, duplicated — see tasks.md task 3.1) scans, polls
   `output_dir` for the *first* `*.result.json` to appear, sends `SIGTERM` at that point (while
   later scans are still unprocessed), then waits for the process to exit and asserts exit code
   `143` plus that the already-written result file(s) parse as valid `ResultEnvelope`s. Skipped on
   `win32`, consistent with `terminationGracePeriodSeconds`/PID-1 semantics being a Linux-container
   concern in the first place.

   **Timing bounds revised after a further round-2 review pass:** an earlier draft suggested "up
   to a few seconds" for the poll and `proc.wait(timeout=10)` for the exit. A reviewer actually
   measured a cold `python -m trait_extractor` subprocess on an unloaded dev machine: ~3.2s of
   pure interpreter/import startup before any per-scan work begins, ~4.4s total before the first
   result file appears — leaving under 1s of margin against a "~5s" bound, likely to flake from
   startup overhead alone on a loaded or cold CI runner, unrelated to the SIGTERM race the test is
   actually meant to exercise. Use a poll bound of at least 20s and `proc.wait(timeout=30)` —
   generous on purpose, since the test only needs `SIGTERM` to land before all of the duplicated
   scans finish, not to hit a tight deadline.

## Open Questions

- The A4 Argo template's `retryStrategy`/`continueOn` reaction to exit code `3` is left to the
  pipeline-repo's A4-wiring step, as scoped by the original issue. This proposal only guarantees the
  codes exist and are documented.
- No automated test runs the built `trait-extractor` container image itself end-to-end (only
  `docker-trait-extractor.yml`'s build/push, never a `docker run`). The `trait-extractor-image`
  spec's SIGTERM and partial-exit-code scenarios are covered by the driver-level subprocess tests as
  a proxy, not a true container-boundary test. Pre-existing gap, surfaced by `review-openspec`'s
  CI/build and spec-quality passes; not fixed by this proposal.

## Resolved during round-2 review

- **Archive-ordering hazard.** `openspec/changes/add-run-manifest-consumption/` was fully
  implemented and merged (PR #263) but had never been archived, and its own delta specs still
  hardcoded the OLD two-way "exits non-zero" convention in the exact two requirement sections this
  proposal modifies. OpenSpec archiving replaces a requirement's text wholesale from the archived
  change's deltas (not a diff against the live spec), so archiving it *after* this proposal merged
  would have silently clobbered the new three-way wording. Resolved by actually archiving
  `add-run-manifest-consumption` (2026-08-19), ahead of this proposal's own code — see tasks.md
  task-group 0.1 and the Commit plan at the top of tasks.md.

## Resolved during round-3 review

- **This proposal's own spec deltas had gone stale relative to the newly-archived content.**
  Archiving `add-run-manifest-consumption` (above) updated the live specs with content this
  proposal's `specs/*/spec.md` deltas were drafted *before* that archive happened, and therefore
  didn't yet include: the "CLI SHALL report skipped scans..." sentence, the "no CLI flag for
  manifest scoping" parenthetical, the run-manifest-scoping cross-reference, the `and no
  run_manifest.json` scenario qualifier, two entire scenarios ("run_manifest.json absent falls back
  to unscoped discovery", "The CLI prints skipped scans, not just succeeded/failed"), and the
  `0.1.0a3`→`0.1.0a7` version-literal bump in `trait-extractor-image`'s spec. Since a MODIFIED
  requirement's delta replaces the requirement wholesale (not a diff) when later applied, leaving
  any of this out would have silently reverted content the just-archived change correctly added —
  exactly the class of bug the archive-ordering fix above was meant to prevent, just on this
  proposal's own side instead of the sibling's. Fixed by re-reading the live specs after archiving
  and updating both delta files to carry every pre-existing scenario/sentence forward alongside
  this proposal's own additions.
- **`test_module_cli_exits_partial_code_on_isolated_scan_failure`'s log-quality wrapper (Decision 2
  above) had two implementation gaps**, both fixed in tasks.md task 2.2: a missing `import
  pydantic` in `__main__.py` (the bare `pydantic.ValidationError` reference in the except-tuple
  would otherwise raise `NameError`, masking the real exception, the first time any of the wrapped
  exceptions actually fires) and a missing `yaml.YAMLError` case (`extract_batch` calls
  `load_pipeline_cards()`, whose `yaml.safe_load` on the packaged `pipeline_selection.yaml` can
  raise it, before `load_run_manifest` runs).
- Minor prose fixes: an incorrect cross-reference in proposal.md attributed a quoted note to "this
  proposal's own design.md" when it actually only appears in the archived
  `add-trait-extractor-image` design; a scenario title was quoted with a word truncated; tasks.md's
  cross-reference to design.md's exit-code-collision reasoning had the "`2` vs `3`" framing
  inverted relative to design.md's actual heading. All three corrected.
