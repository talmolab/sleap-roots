## Commit plan

Per `review-openspec`'s git-workflow pass: one PR, five ordered commits (numbered here by commit
order — NOT the same as the "## N. ..." task-group numbers below, which are referenced in
parentheses). Task group 2 (exit-code convention) depends on task group 1 (empty-input guard) — a
test added in group 2 (`test_module_cli_exits_crash_code_on_empty_input`) is false without group
1's guard already in place — so those two must land in that relative order (same commit is also
acceptable, but not reversed or independent). Task group 3 (SIGTERM) is independent of 1/2 and
could be reordered, but is kept last to match this file's section order.

Commit 1 (task-group 0.1's archive step) is unrelated housekeeping surfaced by round-2 review — an
already-merged, never-archived sibling change whose stale spec text could otherwise clobber this
proposal's own spec deltas later — and should land first, before any of this proposal's own code.

1. `openspec: archive add-run-manifest-consumption` (task-group 0.1)
2. `openspec: propose harden-trait-extractor-exit-semantics` (this proposal, ahead of any code)
3. `traits: raise on empty unscoped input_dir in extract_batch` (task-group 1)
4. `traits: three-way exit-code convention (0/3/1) for the batch CLI` (task-group 2, depends on
   task-group 1 / commit 3)
5. `traits: handle SIGTERM for graceful shutdown in the batch CLI` (task-group 3)

(`docs: sync trait-extractor exit-code docs + changelog`, task-group 4's remaining items, folds
into whichever of commits 3-5 is smallest to extend, or ships as its own trailing commit if none
fit naturally — implementer's call at write time.)

## 0. Housekeeping — already done, unrelated to this proposal's own behavior change

- [x] 0.1 **Archive-ordering hazard (found in round-2 review) — resolved.**
      `openspec/changes/add-run-manifest-consumption/` was fully implemented and already merged
      (PR #263) but had never been archived. Its own delta specs hardcoded the OLD two-way "exits
      non-zero" convention in the SAME two requirement sections this proposal modifies
      (`specs/trait-extractor-image/spec.md`, `specs/result-envelope-output/spec.md`). OpenSpec
      archiving replaces a requirement's text wholesale from the archived change's own deltas, not
      a diff against the live spec — so archiving it *after* this proposal merged would have
      silently clobbered the new three-way (`0`/`3`/`1`) wording. Resolved by running `openspec
      archive add-run-manifest-consumption --yes` immediately (2026-08-19), as its own commit
      ahead of this proposal's own code, before the hazard could materialize — `openspec validate
      --all --strict` passed 15/15 afterward. The live specs at `openspec/specs/
      result-envelope-output/spec.md` and `openspec/specs/trait-extractor-image/spec.md` now
      correctly show the run-manifest-scoping/skip-if-done behavior from #263, still with the OLD
      exit-code wording pending — exactly as expected until *this* change is itself later applied
      and archived. A round-3 review pass separately confirmed this proposal's own spec deltas
      (task groups' `specs/*/spec.md`) had gone stale relative to the newly-archived content and
      fixed them to include it (the "CLI SHALL report skipped scans" sentence, the
      run-manifest-scoping cross-reference, the `0.1.0a3`→`0.1.0a7` bump, and two entire missing
      scenarios) — see this proposal's own `specs/result-envelope-output/spec.md` and
      `specs/trait-extractor-image/spec.md` for the corrected, current versions. This task's own
      commit (see the Commit plan above) still needs to be made — the working tree currently holds
      the archive as an uncommitted change; committing it is part of landing this proposal, the
      archive operation itself is done.

## 1. Empty-input guard in `extract_batch`

- [x] 1.1 **Test first** (`tests/trait_extractor/test_batch.py`):
      - `test_empty_unscoped_input_dir_raises`: an empty `input_dir` (no `run_manifest.json`, no
        `*.predictions.json` anywhere) -> `extract_batch` raises `RuntimeError` mentioning the
        `input_dir` path; `output_dir` is not created (or contains no `*.result.json`).
      - `test_scoped_input_with_no_matching_files_still_reports_per_scan_failure` (regression
        guard, not new behavior): a `run_manifest.json` present with `scan_keys` but zero matching
        `*.predictions.json` files anywhere -> `BatchResult.ok is False` with every scan_key in
        `failed` (already true today via the existing missing-scan_key bookkeeping; this test
        pins that the new guard does not fire here and does not change this path's behavior).
- [x] 1.2 Implement: after the discovery loop in `extract_batch`, when `scope is None` (no
      `run_manifest.json`) and `seen` is empty, raise `RuntimeError(f"no {_MANIFEST_SUFFIX} files
      found under {Path(input_dir).as_posix()}")` before any `if scope is not None:` bookkeeping.
- [x] 1.3 Run `uv run pytest tests/trait_extractor/test_batch.py -x` to confirm both new tests pass
      and no existing test regresses (in particular the existing scoped-missing-scan_key tests).

## 2. Exit-code convention in `__main__.main()`

`0` = full success, `3` = partial (isolated per-scan failures, caught inside `extract_batch`'s own
loop), `1` = crash (exception escaped `extract_batch` entirely). `2` is deliberately left untouched
— `argparse` already owns it for CLI usage errors; see design.md Decision 1 ("Why `3` and not `2`
for partial") for why `2` was rejected during review.

**Added after round-2 cross-repo review:** the sibling `sleap-roots-predict` proposal wraps its
known staging-error types in a narrow `except ...: log; raise` so an operator sees a clean one-line
message instead of a raw traceback for the crash code, and does so for log-quality parity, not
because the exit code changes. `trait_extractor` should match: today `__main__.py` has no
try/except around `extract_batch(...)` at all (confirmed by reading the file), so both the new
empty-input `RuntimeError` (task 1) and the pre-existing `run_manifest.json` failure modes
(`pydantic.ValidationError`, `OSError`, `UnicodeDecodeError` — per `extract_batch`'s own docstring
`Raises:` section) currently surface as a raw traceback. Task 2.2 below adds the same narrow
log-then-reraise wrapper.

- [x] 2.1 **Test first** (`tests/trait_extractor/test_batch.py`, extending the existing
      `test_module_cli_*` subprocess tests):
      - `test_module_cli_exits_zero_on_full_success` (rename/refine existing
        `test_module_cli_writes_envelopes` if it doesn't already assert `returncode == 0`
        explicitly — confirm current coverage first).
      - `test_module_cli_exits_partial_code_on_isolated_scan_failure`: reuse the existing
        one-good-one-bad-scan fixture setup (`_make_bad_scan_missing_slp`) and assert
        `proc.returncode == 3` (replacing the current `== 1` assertion in
        `test_module_cli_exits_nonzero_on_failure` — rename this test and update its
        docstring/assertion together in this task; do not leave a stale `== 1` assertion for what
        is now the partial case).
      - `test_module_cli_exits_partial_code_on_scoped_missing_scan_key`: a `run_manifest.json`
        declaring a `scan_key` with no matching `*.predictions.json` (mirrors
        `test_manifest_declares_scan_key_with_no_predictions_json`'s setup, run through the CLI
        subprocess) -> `returncode == 3`. This closes a scenario-to-test gap the spec delta already
        describes ("A manifest-scoped scan_key with no matching file is already a reported
        failure") but the original task list never exercised at the CLI/exit-code level.
      - `test_module_cli_exits_crash_code_on_empty_input`: an empty input dir -> `returncode == 1`,
        stderr contains a clean, single logged "Batch aborted: ..." line naming the input
        directory (via the new wrapper in task 2.2) — the exception still propagates afterward
        (a bare `raise`), so a traceback is ALSO present below that line; the fix adds a clean
        message ahead of the traceback, it does not suppress the traceback. (Depends on task 1's
        guard.)
      - `test_module_cli_exits_crash_code_on_invalid_run_manifest`: an invalid (empty `scan_keys`)
        `run_manifest.json` -> `returncode == 1` (this already crashes today via
        `pydantic.ValidationError`; this test pins the exit code explicitly for the first time),
        stderr contains a clean "Batch aborted: ..." line preceding the still-present traceback
        (same wrapper as above).
      - `test_module_cli_usage_error_exits_two_unrelated_to_partial_code` (regression/documentation
        test): invoking `python -m trait_extractor` with a missing required argument -> asserts
        `returncode == 2` and that this is `argparse`'s own pre-existing usage-error code, wholly
        unrelated to (and not colliding in meaning with) the new `3` = partial convention. Exists
        so a future change to the argument parser can't silently blur this boundary unnoticed.
- [x] 2.2 Implement: change `main()`'s return statement from `return 0 if result.ok else 1` to
      `return 0 if result.ok else 3` (the crash case, `1`, is simply Python's default
      uncaught-exception exit code and needs no new code in `main()` for that part — confirm this
      by observing an uncaught exception's default exit code rather than assuming it). Additionally
      (log-quality parity with the sibling `sleap-roots-predict` proposal — see the note above):
      wrap the `extract_batch(...)` call in `except (RuntimeError, pydantic.ValidationError,
      OSError, UnicodeDecodeError, yaml.YAMLError) as exc: logger.error("Batch aborted: %s", exc);
      raise`. **Round-3 review found two concrete implementation gaps in this step, both must be
      addressed:**
      - `__main__.py` currently has NO `import pydantic` (only `argparse`, `sys`,
        `typing.List/Optional`, and `trait_extractor.extractor.extract_batch`) — the bare
        `except (..., pydantic.ValidationError, ...)` clause needs `pydantic` bound in this
        module's own namespace (it is NOT transitively available just because `extractor.py`
        imports it). Add `import pydantic` explicitly, matching `envelope.py`'s existing bare
        `import pydantic` convention. Verified: without this, the clause itself would raise
        `NameError` the first time any of the four/five exception types actually fires — i.e. in
        exactly the new tests below — masking the original exception entirely.
      - `extract_batch` calls `load_pipeline_cards()` (in `pipeline_chooser.py`) before
        `load_run_manifest`, which does `yaml.safe_load(...)` on the packaged
        `pipeline_selection.yaml` — malformed YAML there raises `yaml.YAMLError`, not currently
        listed in `extract_batch`'s own docstring `Raises:` section and not covered by the
        wrapper's exception tuple above (now added). Add `import yaml` to `__main__.py` for the
        same dotted-access reason as `pydantic` above (this is a narrow/defensive case, given
        `pipeline_selection.yaml` is a static packaged file, but costs one line to close).
      Also add `import logging` + a module-level `logger = logging.getLogger(__name__)` to
      `__main__.py`, matching `extractor.py`'s existing pattern (`__main__.py` currently has
      neither). The exception still propagates afterward via the bare `raise`, so the exit code is
      unaffected (still the default `1`) — only the logged message changes, from a raw traceback
      to one clean line.
- [x] 2.3 Update the `main()` docstring's `Returns:` section to document all three driver-owned
      codes (`0`/`3`/`1`) and note that `2` is reserved by `argparse`, not by this driver.
- [x] 2.4 Run `uv run pytest tests/trait_extractor/test_batch.py -x` (full file, including the
      subprocess CLI tests) to confirm the new mapping and no other test hardcodes the old
      `== 1`-for-any-failure assumption.

## 3. SIGTERM handler in `__main__.main()`

- [x] 3.1 **Test first** (`tests/trait_extractor/test_batch.py`), split into two tests:
      - `test_handle_sigterm_raises_systemexit_143`: call the new `_handle_sigterm(signal.SIGTERM,
        None)` function directly (no subprocess) and assert `pytest.raises(SystemExit)` with
        `exc_info.value.code == 143`. Not skipped on any platform — this exercises portable Python,
        not real signal delivery.
      - `test_module_cli_sigterm_exits_promptly_and_preserves_completed_output`
        (`@pytest.mark.skipif(sys.platform == "win32", reason="SIGTERM delivery to a subprocess is
        not POSIX-equivalent on Windows")`): launch the CLI as a subprocess over duplicated fixture
        scans, poll `output_dir` for the first `*.result.json`, send `SIGTERM`, then wait for the
        process to exit, and assert `returncode == 143` and that every `*.result.json` present at
        that point parses as a valid `ResultEnvelope` (none are truncated/corrupt).

      **Duplication recipe (corrected after round-4 review — do NOT reuse
      `_make_bad_scan_missing_slp`'s pattern, it's for a different purpose and will silently break
      this test):** `_make_bad_scan_missing_slp` deliberately omits the `.slp` file (it exists to
      produce a *missing-.slp* failure) and is the wrong template for a *valid* duplicate. For each
      new `scan_key` (e.g. `f"{orig}_{i:03d}"`), in a fresh per-scan directory: (1) copy the
      original fixture scan's `.slp` file(s) verbatim, basenames unchanged — `resolve_artifact_paths`
      resolves `slp_path` as a basename relative to the manifest's own directory, so the `.slp`
      filenames do NOT need to match the new `scan_key`; (2) `json.loads` the original
      `.predictions.json`, set `manifest["scan_key"] = new_key` (leave `artifacts`/`slp_path`
      untouched), write to `{new_key}.predictions.json`; (3) `json.loads` the original
      `.scan_metadata.json`, set `sidecar["scan_key"] = new_key`, write to
      `{new_key}.scan_metadata.json`. Both JSON files' `scan_key` AND the manifest's filename stem
      must all equal `new_key` — forgetting either one does NOT trip the duplicate-scan_key guard
      (each stem is still unique); instead it trips the unrelated stem/scan_key-mismatch guard
      (forgetting the manifest's field) or the sidecar/manifest identity guard (forgetting the
      sidecar's field), and since every duplicate copies the same original manifest, ALL of them
      would fail identically and instantly — no real per-scan compute happens, silently defeating
      the entire point of a long-running batch for `SIGTERM` to land inside.

      **Timing design — REVISED AFTER A REAL CI FAILURE (round-5, this is not theoretical):** an
      earlier draft duplicated the fixture scans ~40x and relied on real per-scan compute time to
      widen the window between "first result written" and "batch fully done," reasoning that more
      duplicates would give more margin. This shipped, opened as PR #266, and **`Test (macos-14)`
      failed on the PR's own first CI run**: `assert 3 == 143` — the batch finished normally
      (`3` = partial, since one duplicated scan happened to trip an unrelated numpy dtype-comparison
      error, isolated as usual) *before* `SIGTERM` could take effect, because macos-14 processed all
      40 real scans faster than the test could react. The reasoning was wrong: the margin that
      matters is "time from first result to full batch completion," which scales with **per-scan
      compute cost** (single-digit milliseconds on a fast runner), not with duplicate count — no
      number of real duplicates reliably fixes a race against a variable that isn't being
      controlled. **Fix:** `extract_batch` (`trait_extractor/extractor.py`) gained a small,
      env-var-gated, no-op-by-default per-scan delay hook
      (`SRT_TRAIT_EXTRACTOR_TEST_SCAN_DELAY_S`), called once per scan after that scan's own
      try/except completes. The test now sets this to `1` (second) and uses only 6 duplicated scans
      (3 iterations × 2 fixtures) — with a deterministic ≥1s gap between every scan, the race margin
      is explicit and runner-speed-independent, not a bet on relative compute speed. This is the
      exact "`time.sleep`/env-var delay hook" an earlier round explicitly rejected in favor of "widen
      via more real duplicates" — that earlier call is reversed here because it's now empirically
      falsified, not just theoretically risky. Manually verified (Windows dev box, so only as a
      smoke test of the delay hook itself, not real signal delivery — `Popen.send_signal(SIGTERM)`
      on Windows maps to `TerminateProcess(handle, 1)` and never invokes the Python handler at all,
      confirmed by the returncode always being exactly `1` regardless of what the handler does) that
      the delay hook doesn't break normal batch operation and that the first result appears at the
      expected time before the delay kicks in. The actual POSIX signal-delivery path can only be
      confirmed via real CI (ubuntu-22.04 / macos-14) — watch the next CI run on this PR.
- [x] 3.2 Implement: in `trait_extractor/__main__.py`, add a module-level `_handle_sigterm(signum,
      frame)` that logs (e.g. via the `logging` module, matching `extractor.py`'s existing
      `logger` pattern) and calls `sys.exit(143)`; register it with `signal.signal(signal.SIGTERM,
      _handle_sigterm)` at the top of `main()`, before calling `extract_batch`.
- [ ] 3.3 Run `uv run pytest tests/trait_extractor/test_batch.py -x` on Linux/macOS (or confirm via
      CI) and separately confirm the full suite still collects cleanly on Windows (the subprocess
      test skips, doesn't error; the direct-call unit test still runs and passes). **This was
      previously checked off based on a local Windows run alone, before the PR's real CI had
      actually run — that was wrong: CI's first real run showed `Test (macos-14)` failing on
      exactly this test. Left unchecked until a genuine green run on ubuntu-22.04 AND macos-14 is
      observed after the round-5 timing fix above.**

## 4. Spec + docs + changelog sync

- [x] 4.1 Confirm `openspec/changes/harden-trait-extractor-exit-semantics/specs/*/spec.md` deltas
      (already drafted in this proposal, using `3` for partial) still match the implemented
      behavior after tasks 1-3; adjust either the code or the deltas if implementation surfaced a
      deviation, per this project's "no silent drift" convention.
- [x] 4.2 Update `docs/dev/trait-extractor-service.md` (currently states "the process exits
      non-zero if any scan failed") to describe the three driver-owned codes and the empty-input
      and SIGTERM behavior.
- [x] 4.3 Add a `docs/changelog.md` `[Unreleased]` entry for this change, matching the format and
      placement the recent `add-run-manifest-consumption` change used.
- [x] 4.4 `openspec validate harden-trait-extractor-exit-semantics --strict` passes.
- [x] 4.5 Run the full local verification suite mirroring what `.github/workflows/ci.yml` actually
      runs (confirm exact commands there first, including argument order — a round-2 review found
      the draft above didn't match CI's exact invocation verbatim): `uv run pytest tests/` (full
      suite, not just `tests/trait_extractor/`, to catch any unrelated regression), `uv run black
      --check sleap_roots tests trait_extractor`, `uv run pydocstyle --convention=google
      sleap_roots trait_extractor`.

## 5. Cross-repo follow-up (tracked, not implemented here — no commits in this repo for these)

- [ ] 5.1 In the PR description, explicitly state the three-exit-code convention (`0` success, `3`
      partial, `1` crash, `2` left to `argparse`) and link `talmolab/sleap-roots-predict#26`, so
      that issue's implementation mirrors this one rather than deciding independently.
- [ ] 5.2 After this PR merges (verify its merge state live, don't assume), update
      `sleap-roots-pipeline`'s `docs/bloom-integration/roadmap.md` A3-traits row: mark
      sleap-roots#259 done with the PR link, note the exit-code convention decided here (for
      A3-predict / A4-wiring to match), and add a status-log entry. This is a change in a
      different repository, not a commit in this PR.
- [ ] 5.3 **Self-archive this change after merge (found in pre-PR review — do NOT skip this).**
      `add-run-manifest-consumption` sat merged-but-unarchived long enough to create the exact
      spec-delta-clobbering hazard task-group 0.1 had to fix; without a tracked follow-up, this
      proposal would repeat that pattern for the NEXT change that touches
      `result-envelope-output`'s "Batch driver and module CLI" or `trait-extractor-image`'s
      "Container image runs the trait extractor..." requirements — `openspec/specs/*.md` will keep
      showing the OLD two-way "exits non-zero" wording until this is done, contradicting the new
      docs this PR ships. Once this PR is confirmed merged: `openspec archive
      harden-trait-extractor-exit-semantics --yes`, verify with `openspec validate --all --strict`,
      and commit as its own small PR — matching this repo's established archive-PR precedent (e.g.
      the archive commits/PRs that followed #254 and #257).
