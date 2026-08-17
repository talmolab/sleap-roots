## 1. Bump the contracts pin

**Commit 1.1 and 1.2 as a single atomic commit** — bumping the `pyproject.toml` pins alone (without
updating the two hardcoded test literals) leaves `tests/trait_extractor/` red; these are not
independently safe commit units.

- [x] 1.1 Bump `sleap-roots-contracts==0.1.0a3` to `==0.1.0a7` in all three `pyproject.toml`
      occurrences (dev dependency-group ~line 52, dev optional-dependencies mirror ~line 91,
      `extractor` optional-dependency profile ~line 110 — verify exact lines first, they may have
      shifted). Regenerate `uv.lock` (`uv lock`).
- [x] 1.2 Update every hardcoded old-pin literal in the same commit:
      - `tests/trait_extractor/test_envelope.py`: `assert prov.contract_version == "0.1.0a3"` ->
        `"0.1.0a7"`.
      - `tests/trait_extractor/test_package_boundary.py`: the exact-match guard on
        `sleap-roots-contracts==0.1.0a3` in the `extractor` extra -> `==0.1.0a7` (both the
        comparison AND its assertion-failure message string, which also hardcodes the old
        literal).
      - `docs/dev/trait-extractor-service.md`: line ~75 (`contract_version` documented as
        `0.1.0a3`) and lines ~130-133 (the still-open-blocker note about Bloom's RPC accepting
        `0.1.0a3` / bloom#393) — update the version literal, and add a note that the RPC
        acceptance question has resurfaced for `0.1.0a7` (bloom#393 was closed for the a3 case by
        bloom PR #399, which hard-pins the RPC to exactly `0.1.0a3` — bumping past it reopens the
        same class of blocker; see this change's `design.md` Risks section).
- [x] 1.3 Run `uv run pytest tests/trait_extractor/ -x` to confirm the bump alone (no behavior
      change yet) doesn't break anything, and that `from sleap_roots_contracts import RunManifest,
      RUN_MANIFEST_FILENAME` now imports successfully.

## 2. `run_manifest.py`: load + copy-forward helper (new module)

- [x] 2.1 **Test first** (`tests/trait_extractor/test_run_manifest.py`, new file):
      - `test_load_run_manifest_returns_none_when_absent`: an `input_dir` with no
        `run_manifest.json` -> the loader returns `None` (not an exception).
      - `test_load_run_manifest_parses_valid_file`: a real `run_manifest.json` (pipeline_run_id +
        2 scan_keys) -> returns a validated `RunManifest` with those exact fields.
      - `test_load_run_manifest_raises_on_invalid_manifest`: a `run_manifest.json` failing
        `RunManifest`'s own validation (e.g. empty `scan_keys`) -> raises (loudly, at load time —
        this is a top-level, once-per-batch file, not a per-scan best-effort read).
      - `test_copy_manifest_forward_writes_into_output_dir`: given a loaded manifest's source path,
        copying it forward creates `output_dir/run_manifest.json` with byte-identical content, and
        creates `output_dir` if missing.
- [x] 2.2 Implement `trait_extractor/run_manifest.py`: `load_run_manifest(input_dir) ->
      Optional[RunManifest]` (returns `None` if `RUN_MANIFEST_FILENAME` doesn't exist under
      `input_dir`; raises on a present-but-invalid file) and `copy_run_manifest_forward(input_dir,
      output_dir) -> None` (no-op if the manifest doesn't exist under `input_dir`; a raw file copy,
      not a re-serialization through the frozen `RunManifest` model). Kept separate from
      `manifest.py` (which owns the unrelated per-scan `PredictionManifest`/`ScanMetadata`).
- [x] 2.3 Run the new tests plus the full `tests/trait_extractor/` suite; confirm all green.

## 3. `envelope.py`: read a pre-existing output's idempotency_key and contract_version

- [x] 3.1 **Test first** (extend `tests/trait_extractor/test_envelope.py`):
      - `test_read_existing_identity_returns_none_when_missing`: no `{scan_key}.result.json` in
        `output_dir` -> returns `None`.
      - `test_read_existing_identity_returns_key_and_version_when_present`: a valid pre-written
        envelope -> returns its exact `(idempotency_key, contract_version)` pair.
      - `test_read_existing_identity_returns_none_on_corrupt_file`: a `{scan_key}.result.json`
        containing invalid JSON (and, separately, valid JSON that fails `ResultEnvelope`
        validation) -> returns `None` rather than raising.
      - `test_read_existing_identity_logs_warning_on_corrupt_file`: same corrupt-file case ->
        a warning naming the scan_key is logged (via `caplog` or equivalent).
- [x] 3.2 Implement `read_existing_identity(output_dir, scan_key) -> Optional[Tuple[str, str]]`
      (returning `(idempotency_key, contract_version)`, or a small named tuple/dataclass if
      clearer) in `envelope.py`, alongside the existing `build_provenance`/`build_envelope`/
      `write_envelope`. Log a warning naming the scan_key when the file is **present but
      unreadable/invalid** (corrupt JSON or fails `ResultEnvelope` validation) — **not** when the
      file is simply missing (a plain first-run absence is the normal, expected case and MUST NOT
      warn; only "something unexpected was on disk" should). This package has no prior use of
      Python's `logging` module anywhere (confirmed by grep — the existing convention is plain
      `print(..., file=sys.stderr)` in `__main__.py`); use `logging.getLogger(__name__)` +
      `logger.warning(...)` deliberately as a new, intentional choice for this package (it composes
      better with `caplog`-based testing and Python's zero-config stderr fallback than adding a new
      ad-hoc `print` convention), not because it matches an existing pattern.
- [x] 3.3 Run the new tests plus the full `tests/trait_extractor/` suite; confirm all green.

## 4. `extract_scan`: skip-if-done via early idempotency-key + contract_version comparison

- [x] 4.1 **Test first** (extend `tests/trait_extractor/test_batch.py` or a focused
      `test_skip_if_done.py`, using `shutil.copytree` on the existing `rice_3do_pipeline_output`
      fixture per existing convention — no hand-rolled synthetic JSON):
      - `test_second_call_skips_unchanged_scan`: run `extract_batch` once, then again over
        identical inputs -> the second call's `BatchResult.skipped == [scan_key]` (not
        `succeeded`), and the output file's content is unchanged by the second call.
      - `test_changed_traits_code_sha_forces_recompute`: run once, then again with a different
        `traits_code_sha` passed to `extract_batch` -> the scan reappears in `succeeded` (not
        `skipped`), and the output file's `idempotency_key` changes accordingly.
      - `test_contract_version_change_forces_recompute_even_if_idempotency_key_matches`: run once;
        then `monkeypatch.setattr(trait_extractor.envelope, "contract_version", lambda: "<other
        value>")` (the function object itself) and run again, keeping every other
        idempotency-key input unchanged -> the scan reappears in `succeeded` (not `skipped`), and
        the output file's `contract_version` is updated — proves `idempotency_key`-only comparison
        would have wrongly skipped this case. **Must patch the `contract_version` function object
        directly, NOT `importlib.metadata.version`**: `contract_version()` is
        `@functools.lru_cache(maxsize=1)`-decorated, so once it's been called anywhere in the test
        process, patching the underlying `importlib.metadata.version` lookup has no effect (the
        cached return value is served regardless) — this would make the test either false-negative
        or pass for the wrong, order-dependent reason. Patching the function object itself replaces
        the whole reference and bypasses the cache correctly.
      - `test_corrupt_prior_output_is_recomputed_not_crashed`: pre-seed `output_dir` with a
        corrupt/unparseable `{scan_key}.result.json`, then run `extract_batch` -> the scan
        succeeds (recomputed, output overwritten with a valid envelope), batch does not crash, a
        warning is logged.
      - `test_skip_does_not_invoke_expensive_steps`: patch (`monkeypatch.setattr` or `unittest.mock`)
        `trait_extractor.extractor.load_series` (or another of `choose_pipeline`/
        `check_pipeline_compatible`/`compute_scan_traits`) to raise if called; run `extract_batch`
        twice over identical inputs -> the second run does not raise (proves the patched function
        was never invoked when skipped), while the first run does invoke it. This is the only test
        that actually proves skipping avoids the expensive computation rather than merely producing
        unchanged output (a full, correct recompute would also produce byte-identical output, so
        output-equality alone is not sufficient proof).
- [x] 4.2 Implement in `extractor.py`'s `extract_scan`: move `build_provenance(...)` to run
      immediately after `params = sidecar.to_resolved_params()` (before `load_series`); call
      `read_existing_identity(output_dir, manifest.scan_key)` and compare BOTH
      `provenance.idempotency_key` and `provenance.contract_version` against the returned pair; if
      both match, return a sentinel indicating "skipped" (e.g. `None`) without calling
      `load_series`/`choose_pipeline`/`check_pipeline_compatible`/`compute_scan_traits`/
      `write_envelope`; otherwise proceed exactly as today, reusing the already-built `provenance`
      for the final envelope (don't rebuild it after computing traits).
- [x] 4.3 Add `skipped: List[str] = field(default_factory=list)` to `BatchResult`
      (`extractor.py`); update `extract_batch` to route a skipped `extract_scan` result into
      `result.skipped` instead of `result.succeeded`.
- [x] 4.4 Run the new tests plus the full `tests/trait_extractor/` suite; confirm all green.
      **Specifically re-check `test_envelope.py`'s existing
      `test_extract_scan_emits_valid_byte_stable_envelope`** (calls `extract_scan` twice into the
      same `tmp_path`): after this section, its second call becomes a skip rather than a genuine
      recompute, so its assertions still pass but no longer prove recompute-determinism. Fix via
      **one** of these two (not the "different `traits_code_sha`" option — that would make the
      existing `assert out.read_bytes() == first` fail outright, since `traits_code_sha` feeds
      `Provenance`/the output bytes directly, so it is not a drop-in fix as it might first appear):
      (a) call `extract_scan` twice into two DIFFERENT fresh `tmp_path`-derived output dirs with the
      same inputs (no pre-existing output to skip against on either call, so both are genuine
      recomputes) and assert those two outputs are byte-identical to each other — this preserves the
      original recompute-determinism guarantee without touching the existing assertion's shape; or
      (b) leave the existing test as-is (it now documents skip behavior on its second call) and add
      a NEW, separate test that explicitly forces two genuine recomputes (e.g. via two different
      fresh output dirs, or a real `traits_code_sha` change plus an updated assertion comparing
      trait *values* rather than raw bytes) to keep proving byte-stable re-emission under actual
      recomputation. Do not silently let the original guarantee erode without picking one of these.

## 5. `extract_batch`: manifest-scoped discovery with fallback

- [x] 5.1 **Test first** (extend `tests/trait_extractor/test_batch.py`, new fixture variant: a
      `run_manifest.json` copied alongside the existing 2-scan `rice_3do_pipeline_output` tree, per
      the "copy+mutate the real fixture tree" convention):
      - `test_manifest_scoping_both_scans_in_scope_matches_current_output`: `run_manifest.json`
        present with `scan_keys` = exactly the two fixture scan_keys -> both processed, the two
        `.result.json` files' content is byte-for-byte identical to the no-manifest happy-path
        output (note: `output_dir` as a whole differs by the copied-forward `run_manifest.json` —
        compare the `.result.json` files specifically, not the full directory tree).
      - `test_manifest_scoping_excludes_out_of_scope_scan`: `run_manifest.json` present with
        `scan_keys` = only ONE of the two fixture scan_keys -> only that one appears in
        `succeeded`; the other's `.predictions.json` is untouched and not reported in any
        `BatchResult` bucket at all (verifies contamination-prevention scoping, not just
        "processed fewer scans").
      - `test_manifest_declares_scan_key_with_no_predictions_json`: `run_manifest.json`'s
        `scan_keys` includes a scan_key with no matching `{scan_key}.predictions.json` under
        `input_dir` -> that scan_key appears in `BatchResult.failed` with a message naming the
        missing file; other in-scope scans still process normally.
      - `test_manifest_scoping_duplicate_in_scope_scan_key_is_a_failure`: `run_manifest.json`
        names a scan_key present in two sibling directories under `input_dir` (mirroring
        `test_duplicate_scan_key_across_manifests_reported`'s existing fixture-mutation pattern) ->
        reported as a duplicate-collision failure, not silently resolved by picking either
        candidate.
      - `test_invalid_manifest_aborts_batch`: a `run_manifest.json` present but failing
        `RunManifest` validation (e.g. empty `scan_keys`) -> `extract_batch` raises before
        processing any scan (integration-level check, complementing task 2.1's unit-level test of
        the loader in isolation).
      - `test_no_manifest_falls_back_to_unscoped_rglob`: no `run_manifest.json` anywhere under
        `input_dir` -> behavior and output are byte-identical to `extract_batch`'s current
        (pre-this-change) behavior over the same fixture tree.
      - `test_manifest_copied_forward_into_output_dir`: `run_manifest.json` present -> after
        `extract_batch` returns, `output_dir/run_manifest.json` exists with the same content.
      - `test_manifest_scoped_scan_is_also_skipped_on_second_run`: `run_manifest.json` present,
        run `extract_batch` twice -> the second run's `BatchResult.skipped` contains the
        manifest-scoped scan_key(s) (proves scoping and skip-if-done compose correctly — they are
        built as two independent mechanisms in Sections 3-4 vs. this section, and nothing
        exercises them together otherwise).
- [x] 5.2 Implement in `extractor.py`'s `extract_batch`: call `load_run_manifest(input_dir)` once
      at the top (raises immediately on an invalid manifest, per Section 2). If it returns a
      manifest: keep the existing single `sorted(Path(input_dir).rglob(_MANIFEST_GLOB))` loop and
      its `seen: Dict[str, Path]` duplicate-collision guard structurally unchanged — add exactly
      one new check as the FIRST thing inside the loop body, before the existing duplicate check:
      `if stem not in manifest.scan_keys: continue` (silently skip out-of-scope stems, per the
      "silently excluded" requirement). This means the existing duplicate-detection and per-scan
      `try/except` failure-isolation code apply to the scoped case entirely unchanged — do NOT
      implement this as N separate per-`scan_key` searches (e.g. one `rglob` call per manifest
      `scan_key`); that alternative structure would need its own bespoke duplicate-detection and
      failure-isolation wiring built from scratch and would NOT actually reuse the existing guard,
      contradicting the point of this decision. After the loop completes, compute missing scan_keys
      as `set(manifest.scan_keys) - seen.keys()` and add each as a per-scan failure in
      `BatchResult.failed` naming the missing file. Call `copy_run_manifest_forward(input_dir,
      output_dir)` once after processing. If `load_run_manifest` returned `None`: fall back to
      today's exact `rglob(_MANIFEST_GLOB)` loop, unchanged (no scope-filter `continue`, no
      missing-scan_keys computation).
- [x] 5.3 Run the new tests plus the full `tests/trait_extractor/` suite; confirm all green.

## 6. `__main__.py`: report skipped scans in the CLI

- [x] 6.1 **Test first** (extend `tests/trait_extractor/test_batch.py`'s
      `test_module_cli_writes_envelopes` or add a new CLI test): run `python -m trait_extractor
      <in> <out>` via `subprocess.run` twice over the same inputs -> the second invocation's
      stdout/stderr names the skipped scan(s) distinctly from `ok`/`FAIL` lines, and the summary
      line includes a skipped count.
- [x] 6.2 Implement: update `main()` in `__main__.py` to iterate `result.skipped` (e.g. `print(f"skip
      {scan_key}")`) and include the skipped count in the existing summary line (`"{n} succeeded,
      {n} skipped, {n} failed"` or equivalent) — `result.ok`/exit-code semantics stay exactly as
      today (skipped is not a failure).
- [x] 6.3 Run the new test plus the full `tests/trait_extractor/` suite; confirm all green.

## 7. Full verification

- [x] 7.1 Run the complete `trait_extractor` test suite (`uv run pytest tests/trait_extractor/
      -v`) one final time and confirm all pre-existing tests pass, with the one deliberate,
      accounted-for behavior change from task 4.4 (the byte-stable-envelope test).
- [x] 7.2 Run `uv run black --check trait_extractor tests/trait_extractor` and `uv run pydocstyle
      --convention=google trait_extractor` (match the project's existing docstring conventions —
      read 2-3 existing docstrings in `extractor.py`/`envelope.py` before writing new ones).
- [x] 7.3 Run `openspec validate add-run-manifest-consumption --strict` and resolve any issues.
- [x] 7.4 Update `docs/changelog.md` with an `[Unreleased]` entry describing the scoping +
      skip-if-done addition, and confirm `docs/dev/trait-extractor-service.md` (task 1.2) no longer
      describes only the old unscoped-discovery behavior.
- [x] 7.5 Before opening a PR: confirm with the user/deploy-owner whether a companion Bloom-side PR
      (re-pinning `insert_cyl_result_envelope`'s accepted `contract_version` literal to `0.1.0a7`,
      mirroring bloom PR #399) needs to be filed/sequenced before this image is rebuilt and rolled
      out to the pipeline — this is a deploy-sequencing decision, not something resolved by code in
      this repo (see design.md's Risks section).
