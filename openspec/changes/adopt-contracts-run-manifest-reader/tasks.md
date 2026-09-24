## Commit plan

One PR, squash-merged, as the repo does for every PR (see #263 for the pin+behavior+openspec
precedent). Per-commit CI state therefore matters for review and bisect on the branch, not on
`main`. Each commit is still kept green.

Every code commit is TDD: its named tests are written first and observed failing, and the
implementation lands in the same commit. The red output is quoted in the commit body. Where a red
step cannot be observed on this Windows machine (for example the symlink test, which needs a
privilege this box lacks), the body says so.

1. `openspec: propose adopt-contracts-run-manifest-reader` covers `openspec/changes/adopt-contracts-run-manifest-reader/**`. CI: green (no code).
2. `deps: bump sleap-roots-contracts 0.1.0a7 -> 0.1.0a9` (task group 1) covers `pyproject.toml`, `uv.lock`, `test_package_boundary.py` and `test_envelope.py`. CI: green.
3. `traits: resolve run identity and load the run manifest once via contracts` (task groups 2, 3 and 5). CI: green.
   - Files: `tests/trait_extractor/conftest.py` (new), `trait_extractor/extractor.py`, `trait_extractor/run_manifest.py`, `trait_extractor/__main__.py`, `test_batch.py`, `test_run_manifest.py`.
   - Group 5 (the CLI tuple) folds into this commit, so no intermediate state has fail-loud aborts without the clean log line.
   - The body notes that the forward still republishes by the legacy name until commit 4.
4. `traits: forward the manifest from the loaded snapshot, atomically, with temp cleanup` (task group 4) covers `trait_extractor/run_manifest.py`, `trait_extractor/extractor.py`, `test_run_manifest.py` and `test_batch.py`. CI: green.
5. `docs: trait-extractor run identity, per-run manifests, Bloom a9 deploy gate` (task group 6) covers `docs/dev/trait-extractor-service.md` and `docs/changelog.md`. CI: no path match.

Commit 2 must land first, because commits 3–4 import names that exist only in 0.1.0a9.

**Squash subject:** `traits: adopt contracts 0.1.0a9 per-run run-manifest reader (srp#71) (#NNN)`.

**Squash body:** a bulleted summary, then a `Deploy note:` paragraph naming the Bloom issue from 0.2. The squash body is the only record on `main`.

## 0. Coordination (not code)

- [x] 0.1 Baseline. `uv pip show sleap-roots-contracts` in the project venv reports `0.1.0a7`. PyPI serves `0.1.0a9`.
- [x] 0.2 (Filed 2026-09-23 as Salk-Harnessing-Plants-Initiative/bloom#895.) **Draft, confirm with the user, then file** a Bloom issue: "`insert_cyl_result_envelope` must accept `contract_version` `0.1.0a9`". It must cover:
  - **The live body.** The target is the 2-arg `insert_cyl_result_envelope(envelope jsonb, p_argo_workflow_name text)` in `supabase/migrations/20260917140000_fix_cyl_redelivery_status_fallback.sql:43-53`, **not** #766's 1-arg body. Re-verify it is still the latest definition when filing.
  - **Vendored contract.** `contracts/pin.json` and `contracts/schema/result_envelope.schema.json` (`$id`).
  - **Options, with evidence (the choice is Bloom's).**
    - **(a) Cutover window.** This is Bloom's established pattern (#766, then pipeline#52). It is loud and recoverable: write-back has no `continueOn`, `ingest.py:528-532` classifies the rejection, and envelopes are re-delivered.
    - **(b) Transitional set `{0.1.0a7, 0.1.0a9}`.** It answers the two prior rejections (`repin-cyl-contract-a3/design.md:24,60-61`, `repin-cyl-contract-a7/design.md:11-22`) with what has changed: an a7 image is live, real a7 rows exist, and the cluster is shared. Per-row `contract_version` keeps provenance.
  - **Cutover guard.** The a7 guard tripped on 10 a3 rows and wedged every staging deploy for six days (bloom#787). It was resolved only by restamping disposable fixture rows. Today's a7 rows are real. Under (a), a guard copied from #766 **will** trip, so it must be removed or redesigned. Under (b), no guard is needed.
  - **Bloom touch list:**
    - the RPC literal;
    - a new rollback file (`supabase/rollbacks/20260917140000_…_rollback.sql` restores a7);
    - `PINNED_VERSION` in `tests/integration/test_cyl_writeback_rpc.py` and `test_cyl_read_path.py`;
    - `contracts/pin.json`, `contracts/schema/result_envelope.schema.json` and `contracts/README.md`.
  - **Evidence of a pure restamp.** The schema diff is `$id`-only. No `ResultEnvelope`/`Provenance`/`TraitValue` hunks. `identity.py`/`hashing.py` are unchanged.
  - **Gate wording.** Applied to the database the cluster write-back targets, not merely merged to `staging`.
  - Then link the issue number from `proposal.md`'s Deploy gate and from the changelog entry.
- [ ] 0.2b Record the Bloom gate in the `sleap-roots-pipeline` roadmap frontier (rows 2/5/6 show none). **Owned by the pipeline session** that wrote the frontier table (it said so in PR #269's review); not done here.
- [x] 0.3 (PR #269.) PR body cross-links. Use `Refs` (not closing keywords) for srp#71 and predict#40, which both stay open. Also link pipeline#82 and the Bloom issue as the deploy gate.
- [x] 0.4 (Filed 2026-09-23 as talmolab/sleap-roots-pipeline#86.) **Draft, confirm with the user, then file** a `sleap-roots-pipeline` issue. Once fail-loud is reachable, traits' exit `1` is retried twice (`retryStrategy: {limit: 2, retryPolicy: Always}`), then `continueOn: failed` runs write-back over `traits/`'s stale manifest. Cross-link it from design Risks.

## 1. Pin bump (commit 2)

- [x] 1.1 **Tests first.**
  - In `test_package_boundary.py`, change the `extractor` literal to `sleap-roots-contracts==0.1.0a9`, both the assertion at `:94` and the message at `:98`.
  - Add `test_all_contracts_pins_agree`.
    - It parses `pyproject.toml` with `tomllib`, as the existing tests do.
    - It collects the string `sleap-roots-contracts` requirements from `[dependency-groups].dev`, `[project.optional-dependencies].dev` and `.extractor`. Skip `{include-group=…}` dict entries.
    - It asserts exactly three, each `==0.1.0a9` with the `python_version >= '3.11'` marker.
  - In `test_envelope.py:41`, change the literal to `"0.1.0a9"`.
  - Run them: all three fail against the a7 pins.
- [x] 1.2 Bump the three pins (`pyproject.toml` lines 52/91/110).
  - Run `uv lock --upgrade-package sleap-roots-contracts`, then `uv lock --check`, then `uv sync`.
  - Judge churn with `git diff --stat uv.lock`, not a raw file compare (autocrlf). Expect only lines 3717/3718/3740/3747/3753/3755, the contracts entries.
  - `[project] dependencies` must be unchanged.
- [x] 1.3 Run the 1.1 tests, then the full `tests/trait_extractor/` suite, **before** any code change. Expect green. A failure is a finding to record, not to patch over.

## 2. Test isolation (commit 3)

- [x] 2.1 Create `tests/trait_extractor/conftest.py` with an autouse fixture that runs `monkeypatch.delenv("ARGO_WORKFLOW_NAME", raising=False)`.
  - Test `test_argo_workflow_name_is_cleared_for_tests` (in `test_batch.py`) asserts the variable is absent.
  - Its red step is manual: run it with `ARGO_WORKFLOW_NAME=x` exported, before the fixture exists. It is vacuous in CI, where the variable is never set. That is accepted and stated in the commit body.
  - Subprocess CLI tests inherit the cleaned `os.environ`: `_run_module_cli` builds `env={**os.environ, …}` at call time.

## 3. Identity resolution + single-snapshot scoping + CLI tuple (commit 3)

Tests go in `tests/trait_extractor/test_batch.py`, over copies of `tests/data/rice_3do_pipeline_output` (`scan0K9E8BI`, `scanYR39SJX`).

Extend the helper to `_write_run_manifest(dir, scan_keys, *, pipeline_run_id="wf-test", filename=None)`. `filename=None` means `RUN_MANIFEST_FILENAME`. An explicit filename lets the file name and the content id differ, which the identity-mismatch tests need. Update the helper docstring.

Unless noted, every scoped test names **one** of the two scans, so that "scoped" is observable against "unscoped". The warning assertions select records from logger `trait_extractor.extractor` at `WARNING` or above. They never assert on `caplog.text` substrings shared with other warnings.

Every `extract_batch` call in these tests uses a fresh `out_dir`. A reused one turns a scoped scan into `skipped`.

- [x] 3.1 **Tests first.** First add a stub to `extractor.py`: the `pipeline_run_id` keyword, ignored, **and** `from sleap_roots_contracts import pipeline_run_id_from_env` (unused until 3.4). With the stub, the tests fail on behavior rather than on `TypeError`, and every spy target exists. Then run each test and watch it fail. The exceptions are the ones marked "regression guard", whose red is manual and is recorded in the commit body.
  - `test_per_run_manifest_wins_over_legacy`
    - Setup: `"wf-a"`. The per-run file names 1 scan; the legacy file names 2.
    - Assert: `succeeded == [that scan]`.
  - `test_concurrent_runs_are_each_scoped_to_their_own_manifest`
    - Setup: `wf-a` and `wf-b` share an input, with separate outputs.
    - Assert **scoping only**: each run's `succeeded` is its own scan. The output-manifest assertions come in 4.2.
  - `test_known_identity_without_manifest_raises_missing`
    - Setup: `"wf-a"` and no manifest.
    - Assert: `RunManifestMissingError`, and no `*.result.json`.
  - `test_known_identity_with_empty_input_raises_missing_not_runtime_error`
    - Setup: an empty `input_dir` and `"wf-a"`.
    - Assert: `RunManifestMissingError`. The `RuntimeError` guard is reachable only with no identity.
  - `test_per_run_manifest_naming_another_run_raises_identity_error`
    - Setup: `filename="run_manifest.wf-a.json"`, content `pipeline_run_id="wf-b"`.
    - Assert: `RunManifestIdentityError`, and no `*.result.json`.
  - `test_unusable_run_id_raises_value_error`
    - Setup: `"../x"`.
    - Assert: `pytest.raises(ValueError, match="not usable as a filename component")`, `type(exc) is ValueError`, and no `*.result.json`.
  - `test_invalid_per_run_manifest_does_not_fall_back_to_valid_legacy`
    - Setup: the fixture scans, `"wf-a"`, and `run_manifest.wf-a.json` with empty `scan_keys` beside a valid legacy file.
    - Assert: `pydantic.ValidationError`, and `not list(out.glob("*.result.json"))`.
    - Also parametrize the existing `test_invalid_manifest_aborts_batch` over the legacy (no identity) and per-run (`"wf-a"`) names.
  - `test_legacy_manifest_naming_another_run_is_honored_and_warned`
    - Setup: `"wf-a"`, and a legacy file naming `wf-old` with 1 scan.
    - Assert: the scope is honored, and exactly one record contains `run_manifest.json`, `wf-old` and `wf-a`.
  - `test_legacy_manifest_naming_this_run_is_not_warned`
    - Setup: the same, but the legacy file names `wf-a`.
    - Assert: no `WARNING` records. This is a regression guard; it passes against the stub by design.
  - `test_per_run_manifests_without_identity_are_ignored_but_warned`
    - Setup: `None`, and only `run_manifest.wf-a.json` naming 1 scan.
    - Assert: both scans are processed, and one `WARNING` names `run_manifest.wf-a.json`.
  - `test_run_identity_defaults_to_environment`
    - Setup: `setenv("ARGO_WORKFLOW_NAME", " wf-a\n")`, a per-run `wf-a` manifest naming 1 scan, and **no** `pipeline_run_id` argument.
    - Assert: the scope is applied (the value is stripped).
    - Then `setenv(..., "   ")` over a legacy file naming 1 scan. Assert: that 1-scan scope is applied (blank → `None` → legacy read).
    - Spy on `trait_extractor.extractor.pipeline_run_id_from_env` with a delegating wrapper, and reset its counter before each call. Assert: it is called exactly once per call that omits the argument, and zero times for an explicit `None` or `"wf-a"`.
  - `test_explicit_none_ignores_environment` (**regression guard**)
    - Setup: `setenv("ARGO_WORKFLOW_NAME", "wf-a")`, `pipeline_run_id=None`, and no manifest.
    - Assert: unscoped success.
    - It passes against the stub by design, since the stub never reads the environment. Its manual red: temporarily resolve the environment when the argument is `None`, and watch it raise `RunManifestMissingError`.
  - `test_run_identity_is_not_stamped_into_envelopes` (spec: Provenance, design D8)
    - Setup: per-run `wf-a` and `wf-b` manifests naming the **same** scan, with separate outputs, plus a no-identity legacy-scoped run over the same scan.
    - Assert: all three `.result.json` files are byte-identical, and `provenance.pipeline_run_id is None`.
    - Also: a `wf-b` run over `wf-a`'s output yields `skipped == [scan]`.
    - This is a regression guard against the stub; the manual red is to stamp the id temporarily.
  - `test_dangling_symlink_manifest_raises`
    - Setup: the fixture scans. Create the link at runtime; on `OSError` (no privilege), call `pytest.skip(...)`. A static `skipif` cannot detect missing privilege.
    - Assert: `FileNotFoundError` matching `re.escape(link.as_posix())`, and `not list(out.glob("*.result.json"))`.
    - The GitHub windows-2022 runners are admin, so the test runs there.
  - `test_manifest_is_loaded_once_with_allow_legacy_true`
    - Setup: a counting wrapper around the real `trait_extractor.extractor.load_run_manifest`.
    - Assert: one call, with positional `(input_dir, "wf-a")` for an explicit id and `(input_dir, None)` when the environment is unset, and `allow_legacy=True`.
    - Its docstring states that this pins the call shape. The "one snapshot" guarantee is pinned by 4.2's batch-level snapshot test, because this count is already 1 today.
- [x] 3.2 **CLI tests first** (task group 5, folded in). Extend `_run_module_cli(repo_root, in_dir, out_dir, extra_env=None)`, which merges `extra_env` over `os.environ`.
  - `test_module_cli_exits_crash_code_on_missing_manifest_for_known_run`: `ARGO_WORKFLOW_NAME=wf-a`, no manifest.
  - `test_module_cli_exits_crash_code_on_identity_mismatch`: `ARGO_WORKFLOW_NAME=wf-a`, and `run_manifest.wf-a.json` naming `wf-b`.
  - `test_module_cli_exits_crash_code_on_unusable_run_id`: `ARGO_WORKFLOW_NAME=../x`.
  - `test_module_cli_exits_crash_code_on_nonexistent_input_dir`.
  - Each asserts exit `1` and `re.search(r"Batch aborted: .*" + re.escape(token), proc.stderr)`. The token is `wf-a`, `wf-b`, `../x`, or `in_dir.as_posix()`. Assert on the log line, not on the traceback text.
  - **When the red is observed:**
    - Against the 3.1 stub, the first three exit `0`: the environment is not read yet, so the run is unscoped.
    - The intended red, "exit `1` with no `Batch aborted:` line", appears after 3.4a and before 3.4b. None of `RunManifestMissingError`, `RunManifestIdentityError` or a bare `ValueError` is in today's tuple.
    - The fourth test raises `RuntimeError` today and `FileNotFoundError` after. Both are caught and name the directory, so it is a regression guard.
- [x] 3.3 **Update existing tests to the inherited behavior.**
  - `test_nonexistent_unscoped_input_dir_raises` now expects `pytest.raises(FileNotFoundError, match=re.escape(in_dir.as_posix()))`. Keep the "nothing written" assertion, and update the docstring.
  - `test_module_cli_exits_crash_code_on_non_utf8_run_manifest`: the assertions are unchanged. The docstring now says `ValidationError` (`json_invalid`), not `UnicodeDecodeError`.
  - `test_main_logs_clean_message_on_os_error_from_run_manifest`: the seam `extractor_module.load_run_manifest` survives unchanged. Confirm it passes.
  - `test_run_manifest.py:10`: change the import to `copy_run_manifest_forward` only, or the module fails at collection. Delete its three `load_run_manifest` tests, which now cover contracts' own behavior (covered by 3.1 plus `test_invalid_manifest_aborts_batch`). Drop the now-unused `import pydantic`.
- [x] 3.4a **Implement the extractor side.** Then run 3.2 and observe its intended red.
  - `extractor.py`:
    - Import `from sleap_roots_contracts import load_run_manifest, pipeline_run_id_from_env`.
    - Add a module `_FROM_ENV = object()` sentinel and the keyword `pipeline_run_id: Union[str, None, object] = _FROM_ENV`.
    - Make one `loaded = load_run_manifest(input_dir, pipeline_run_id, allow_legacy=True)` call, with `scope` from `loaded.manifest.scan_keys`.
    - Add the D5 warning and the D7 warning (`glob("run_manifest.*.json")`, top level, only when `loaded is None`).
    - The forward call is unchanged in this commit (still two arguments).
  - `run_manifest.py`: remove `load_run_manifest` and the unused `RunManifest` import.
- [x] 3.4b **Implement the CLI side.** In `__main__.py`, add `sleap_roots_contracts.RunManifestError` and `ValueError` to the tuple (D6).
  - **Docstrings**: check the format against neighboring Google-style docstrings.
    - `extract_batch`: rewrite the body at `extractor.py:152-160` (per-run and legacy names, identity, fail-loud). Add `pipeline_run_id` to `Args:`. The `Raises:` section lists `RunManifestMissingError`, `RunManifestIdentityError`, `ValueError`, `pydantic.ValidationError`, `OSError`/`FileNotFoundError`, `RuntimeError`, and **keeps** `yaml.YAMLError`. Drop the pointer to the removed `run_manifest.load_run_manifest`.
    - `main()` `Raises:`: add `RunManifestError`, `ValueError` and `FileNotFoundError`. Reword the `UnicodeDecodeError` entry, which is still reachable via `pipeline_selection.yaml`'s `read_text` (`pipeline_chooser.py:65`), not via the manifest.
    - The `__main__` module docstring covers the per-run name.
- [x] 3.5 Run 3.1–3.3 and see them pass. Run the full `tests/trait_extractor/` suite and see it green. All pre-existing scoping/duplicate/orphan/skip tests pass **unchanged**, because they run with no identity.

## 4. Snapshot-based atomic forward with cleanup (commit 4)

Unit tests go in `tests/trait_extractor/test_run_manifest.py` and target `copy_run_manifest_forward(read, input_dir, output_dir)`. Build `read` with contracts' `read_run_manifest(...)` over a real file, never by hand.

**Failure injection must not patch globals blindly.** `monkeypatch.setattr(trait_extractor.run_manifest.os, "replace", …)` patches the **global** `os.replace`, which every `write_envelope` also uses via `Path.replace`. Every fake therefore delegates to the real function and fails only for the manifest's own paths:

```python
real_replace = os.replace
def fake_replace(src, dst, *a, **k):
    if Path(dst).name.startswith("run_manifest"):
        raise OSError("replace")
    return real_replace(src, dst, *a, **k)
```

The same pattern applies to two more fakes:
- **`os.chmod`**: key on `Path(p).name.startswith(".run_manifest")`, and delegate an `int` (fd) argument unchanged.
- **`Path.unlink`**: patch `pathlib.Path.unlink` with `def fake_unlink(self, missing_ok=False)`, keyed on `self.name`, and delegate with `real_unlink(self, missing_ok=missing_ok)`. The implementation must call `os.chmod(tmp_path, read.mode)` and `Path(tmp).unlink(...)`, not an fd variant or `os.unlink`. The fakes therefore pin D3.

Directory-content assertions compare the full listing, `sorted(p.name for p in out.iterdir())`, so any temp-file name is caught.

- [x] 4.1 **Unit tests first.** Run each and watch it fail.
  - `test_forward_publishes_read_bytes_under_read_filename`: a per-run read. The listing is exactly `["run_manifest.wf-a.json"]`, byte-equal to `read.data`.
  - `test_forward_publishes_snapshot_not_current_source`: overwrite the source after the read, then forward. The output holds the original bytes.
  - `test_forward_preserves_source_mode`: `skipif(sys.platform == "win32")`, parametrized over `0o644` (the spec's value) and `0o640`. The latter is neither the umask default nor `mkstemp`'s `0o600`, so it discriminates.
    - `os.chmod(source, mode)` runs **before** `read_run_manifest`, because `read.mode` comes from `fstat` at read time.
    - A selective `os.replace` spy records `os.stat(src).st_mode & 0o777` before delegating. It asserts that equals `read.mode`, which pins "chmod **before** replace".
  - `test_forward_failure_removes_temp_file`: the selective `os.replace` fake. `pytest.raises(OSError)`. The listing is empty.
  - `test_forward_failure_during_chmod_removes_temp_file`: the same, with the selective `os.chmod` fake.
  - `test_forward_systemexit_removes_temp_file_and_propagates`: the selective `os.replace` fake raises `SystemExit(143)`. Assert `pytest.raises(SystemExit)` with `.code == 143`, and that the listing is empty. This pins `except BaseException` (D3).
  - `test_forward_cleanup_failure_does_not_mask_original_error`: `os.replace` raises `"replace"`, and the selective `Path.unlink` raises `"unlink"`.
    - Assert `pytest.raises(OSError, match="^replace$")`.
    - Find the leftover with `[leftover] = [p for p in out.iterdir() if p.name.startswith(".run_manifest")]`, and assert that one `WARNING`'s `getMessage()` contains `leftover.name`. Do not compare `str(path)`, because separators differ on Windows.
  - `test_forward_temp_file_is_dot_prefixed_inside_output_dir`: a delegating spy on `trait_extractor.run_manifest.tempfile.mkstemp`. Assert `Path(dir).resolve() == output_dir.resolve()` and `prefix.startswith(".")`, reading `dir` and `prefix` from kwargs or positional arguments.
  - **Retarget to the new signature:** `writes_into_output_dir`, `overwrites_a_different_prior_manifest`, `noop_when_input_and_output_are_the_same`, and `noop_for_differently_spelled_same_path`. Add a per-run variant of the same-path no-op that also asserts no `.run_manifest*` temp file. **Delete** `noop_when_manifest_absent`: absence is decided before the call (`loaded is None`). Record the deletion in the commit body.
- [x] 4.2 **Batch-level tests first**, in `test_batch.py`.
  - `test_batch_forwards_loaded_snapshot_not_rewritten_source`
    - Wrap `trait_extractor.extractor.load_run_manifest` so it calls the real function, then overwrites the source with a different valid manifest, then returns.
    - Assert: the forwarded bytes equal the original bytes, and a spy on `copy_run_manifest_forward` received the very `loaded.read` object.
  - `test_per_run_manifest_forwarded_under_its_own_name`
    - Setup: `"wf-a"`, per-run plus legacy files present.
    - Assert: `out/run_manifest.wf-a.json` is byte-equal to its source, `out/run_manifest.json` does not exist, and no D5 warning is logged.
  - `test_concurrent_runs_forward_only_their_own_manifest`
    - Setup: separate outputs.
    - Assert: each output holds only its own `run_manifest.<id>.json`.
  - `test_concurrent_runs_sharing_output_dir_do_not_clobber_each_others_manifest`
    - Setup: `wf-a` then `wf-b`, with the same input and the same output.
    - Assert: both per-run files exist, byte-equal to their sources, and neither `succeeded` contains the other's scan. The second run's orphan warning names the first run's scan; pin it.
  - `test_batch_forward_failure_leaves_no_temp_file`
    - Setup: the selective `os.replace` fake during a scoped two-scan batch.
    - Assert: `set(result.succeeded) == {both scans}` (per-scan writes unaffected), the warning names `read.filename` and both directories, and the listing is exactly the `*.result.json` files.
  - `test_per_run_manifest_rerun_skips_and_reforwards`
    - Setup: run `wf-a` twice.
    - Between the runs, write junk to `out/run_manifest.wf-a.json`.
    - Assert: the second run has `skipped == [scan]`, and the forwarded file is restored to the source bytes. This exercises `os.replace` over an existing destination on Windows.
  - The existing `test_copy_forward_failure_*` tests keep patching `trait_extractor.extractor.copy_run_manifest_forward` (their `_boom(*args, **kwargs)` absorbs the new arity) and must pass unchanged.
- [x] 4.3 **Implement** `copy_run_manifest_forward(read, input_dir, output_dir)` per design D3/D4.
  - Use `import os` / `import tempfile` (not `from … import`), a module `logger`, and `Path(tmp).unlink(missing_ok=True)` for cleanup. Remove `shutil`.
  - Order: `mkstemp` → write via `os.fdopen` → close the fd (in `finally`, before anything else) → `os.chmod(tmp_path, read.mode)` → `os.replace`. An fd still open during cleanup would make the Windows `unlink` fail with WinError 32.
  - Rewrite the module docstring (it drops the "Load +" half), the function docstring, and the inline comments at `run_manifest.py:77-93` (the `copyfile`/`SameFileError` backstop and the fixed `.tmp` no longer apply).
  - In `extract_batch`, pass `loaded.read`, and make the best-effort warning `"failed to copy %s from %s to %s: %s"` with `loaded.read.filename`. The existing assertions still match "failed to copy run_manifest.json" for legacy reads.
  - Update `test_manifest_present_input_dir_equals_output_dir_does_not_crash`'s docstring, which cites `shutil.SameFileError`.
- [x] 4.4 Run 4.1 and 4.2 and see them pass. Run the full suite green locally (Windows). The POSIX-only mode test runs on CI's Linux and macOS legs.

## 6. Docs (commit 5; task group 5 was folded into 3.2)

- [x] 6.1 `docs/dev/trait-extractor-service.md`
  - `:75` a7 → a9.
  - `:83-93` identity, per-run vs legacy, `allow_legacy=True` pending pipeline#82, the D5/D7 warnings, and forward-under-read-name.
  - `:105` the exit-`1` list now includes a missing manifest for a known run, an identity mismatch, an unusable `ARGO_WORKFLOW_NAME`, and a missing `input_dir`.
  - `:119-124` note that `-e ARGO_WORKFLOW_NAME` enables fail-loud, and that `:latest` now emits a9 envelopes, which Bloom rejects until the re-pin.
  - `:149-158` rewrite the Downstream note. It is stale today: bloom#685 was closed 2026-09-10, #766 re-pinned to a7, and the cluster runs `sha-689cffb`, an a7 image. The new gate is the 0.2 Bloom issue.
- [x] 6.2 `docs/changelog.md`
  - Add an Unreleased `### Added` bold-titled bullet after `:43`, following the `:42-43` precedent. It covers:
    - the pin bump;
    - identity from the environment;
    - per-run then legacy resolution;
    - fail-loud;
    - the warnings;
    - the snapshot forward with temp cleanup (predict#40 residue);
    - the **Behavior changes** from the proposal table;
    - a **Deploy note** naming the Bloom issue.
  - Annotate the stale deploy note at `:42`: "(Resolved: bloom PR #766 re-pinned to `0.1.0a7`; bloom#685 closed 2026-09-10.)"
  - Annotate `:54`: "(since bumped a3 → a7 → a9; see Added)".
  - Do **not** copy the retracted "worse than the defect" rollout rationale (design Out of scope).
- [x] 6.3 Grep for stale `0.1.0a7` and `0.1.0a3` literals (e.g. changelog `:40`, which should be annotated or justified as historical) and for `run_manifest.json`-only wording in live docs and docstrings. Exclude `openspec/changes/archive/**`, `.worktrees/**`, `sleap_roots.egg-info/**` and `docs/superpowers/specs/**` (historical). Fix or justify each hit.

## 7. Verification and reconciliation (before `/pre-merge-check`)

- [x] 7.1 Run the same commands CI runs:
  - `uv run black --check sleap_roots tests trait_extractor`
  - `uv run pydocstyle --convention=google sleap_roots trait_extractor`
  - `uv run pytest tests/`
  - `uv lock --check`
  - Coverage of the new branches: `uv run pytest --cov=trait_extractor --cov-report=term-missing tests/trait_extractor`. Confirm the cleanup, unlink-failure, D5 and D7 lines are hit. Codecov enforces no threshold.
- [x] 7.2 `openspec validate adopt-contracts-run-manifest-reader --strict`.
- [x] 7.3 Reconcile against the approved proposal, design and deltas.
  - Every named API is what the code calls.
  - Every scenario has a test that exercises it **at the level it is worded**.
  - Each deviation is noted as `### Why N instead of M?`.
- [x] 7.4 Container smoke. Start Docker Desktop, build `trait-extractor.Dockerfile`, and use a fresh `/out` per run. On Git Bash, use `MSYS_NO_PATHCONV=1`.
  - Run (a): the fixture, no `ARGO_WORKFLOW_NAME`. Expect exit `0` and `contract_version == "0.1.0a9"`.
  - Run (b): `-e ARGO_WORKFLOW_NAME=wf-smoke`, no manifest. Expect exit `1` and `Batch aborted:`.
  - Run (c): a copied fixture plus `run_manifest.wf-smoke.json` naming one scan, with `-e ARGO_WORKFLOW_NAME=wf-smoke`. Expect exit `0`, one envelope, and `/out/run_manifest.wf-smoke.json` byte-equal to its source with the source's mode (Linux, in-image).
  - If Docker can't run, say so in the PR; never claim the smoke ran.

## 8. Post-merge (separate PRs)

- [ ] 8.1 Verify the merge live (`gh pr view`). Then open `openspec: archive adopt-contracts-run-manifest-reader after PR #N merge`, run `openspec archive … --yes` and `openspec validate --all --strict`, matching `094992d`/`a98918d`.
- [ ] 8.2 The `sleap-roots-pipeline` traits pin bump is **gated on the Bloom a9 acceptance being applied**. It covers:
  - the image `sha-…@sha256:…`;
  - `SRT_TRAITS_CONTAINER_DIGEST` in the same change (`check_manifests.py`);
  - rewriting the "inert today" `ARGO_WORKFLOW_NAME` comment (template lines 54–58);
  - flipping the pipeline roadmap entry;
  - rewriting the image-history comment at template `:34-42` ("a7-emitting envelopes will be accepted");
  - running `scripts/check_manifests.py` explicitly, since the pipeline repo has no CI for it;
  - `argo template update` for the traits template, then `scripts/check_cluster_drift.sh` → IN SYNC (from WSL: `wsl bash -c`);
  - live acceptance: a Bloom-dispatched run whose new `cyl_trait_sources` rows carry `contract_version: 0.1.0a9`;
  - if option (b) was chosen, a follow-up issue to narrow the accepted set.
  Record in that PR that the bloomctl writer flip (§4 step 3) waits for both the predict and traits templates to be applied.
- [ ] 8.3 **When talmolab/sleap-roots-pipeline#82 lands** (the fleet-wide `allow_legacy=False` flip), make a forward failure under a known identity a `BatchResult.failed` entry (exit `3`), not only a warning. The exit-`0` choice in design D3 is valid only while write-back's legacy fallback is on. Tests first: a batch with an identity plus a failing forward must exit `3`.
