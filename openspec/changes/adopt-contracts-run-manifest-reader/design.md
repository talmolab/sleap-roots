# Design: adopt-contracts-run-manifest-reader

The design of record is `sleap-roots-pipeline/docs/superpowers/specs/2026-09-21-per-run-run-manifest-identity-design.md`,
merged and referenced below as "design §N". This file records only the decisions local to `sleap-roots`.

## Context

`trait_extractor` is both a **reader** (it scopes discovery to `scan_keys`) and a **forwarding
hop** (it republishes the manifest into `output_dir` for write-back). Today it reads the manifest
twice. First `load_run_manifest(input_dir)` at `extractor.py:194` parses it. Then
`copy_run_manifest_forward` re-opens it by name with `shutil.copyfile`. The bytes scoped against
and the bytes forwarded can therefore differ if the file changes between the two opens.

0.1.0a9 adds these primitives. Their signatures were verified against the installed wheel, not
only the design doc, since design §3.1's sketch of `check_run_manifest_identity` still shows a
`filename` parameter where the shipped function takes the whole `read`:

- `pipeline_run_id_from_env(env=None) -> str | None` returns stripped `ARGO_WORKFLOW_NAME`, or
  `None` when it is unset or blank.
- `load_run_manifest(directory, pipeline_run_id, *, allow_legacy) -> LoadedRunManifest | None`
  returns `LoadedRunManifest(manifest, read)`. `read` is a
  `RunManifestRead(filename, data, mode, is_per_run)`.
- It raises `RunManifestMissingError` (a `RunManifestError` and a `LookupError`),
  `RunManifestIdentityError` (a `RunManifestError`, deliberately **not** a `ValueError`),
  `ValueError` (unusable id), pydantic `ValidationError`, and `OSError`. That includes
  `FileNotFoundError` for a missing directory or a dangling-symlink candidate.

## Decisions

### D1. Resolve the id inside `extract_batch`, via a sentinel default

`extract_batch(..., pipeline_run_id=_FROM_ENV)`: if the caller passes nothing, the function calls
`pipeline_run_id_from_env()`. Explicit `None` means "no identity" and explicit `"wf-1"` means that
id, and both are what tests use.

*Rejected:* resolving in `__main__` and defaulting `extract_batch` to `None`. A direct library
caller under Argo would then silently use the legacy name, which is the exact incoherent
"writer has an id, reader doesn't" state design §2.5 rules out. The sentinel makes the safe path
the default one.

*Test isolation:* an autouse fixture in `tests/trait_extractor/conftest.py` runs
`monkeypatch.delenv("ARGO_WORKFLOW_NAME", raising=False)`. A developer or CI shell that happens to
export it therefore cannot flip existing tests into fail-loud mode. The subprocess CLI tests
inherit the cleaned `os.environ`.

### D2. One snapshot drives both scoping and forwarding

`extract_batch` calls contracts' `load_run_manifest` once. `copy_run_manifest_forward`'s signature
becomes `copy_run_manifest_forward(read: RunManifestRead, input_dir, output_dir)`. It never reads
the source again; it publishes `read.data`. Traits' own `load_run_manifest` wrapper in
`trait_extractor/run_manifest.py` is **removed**. Contracts' function is the recommended entry
point, and a local wrapper would be a second definition to drift. `extractor.py` imports it under
the same name, `load_run_manifest`, so the existing monkeypatch-based `main()` test keeps its
seam.

### D3. Forward: `mkstemp` + `chmod(read.mode)` + `os.replace`, with cleanup on failure

This mirrors predict's forwarder (`sleap_roots_predict/run_manifest.py` on `main`), for the
reasons predict documents:

- The temp file lives *inside* `output_dir`. A system-temp file would make `os.replace`
  cross-device (`EXDEV`) on the NFS production mount.
- It is named `tempfile.mkstemp(dir=output_dir, prefix=f".{read.filename}.", suffix=".tmp")`. That
  name is unique, so two writers never share a temp path. This matters for the legacy name, which
  is still shared across runs. The dot prefix hides an orphan left by `SIGKILL` from a
  `run_manifest*` glob.
- The write goes through the `mkstemp` fd, and the fd is closed before the replace. On Windows an
  open handle makes `os.replace` fail with `WinError 32`.
- `os.chmod(tmp, read.mode)` runs **before** the replace. `mkstemp` creates the file at `0600`,
  and write-back runs as a different uid on the same shared mount. Without the chmod, the forwarded
  file is unreadable downstream. Today's `shutil.copyfile` has no mode issue, because it creates
  the file at umask default rather than `0600`. Switching to `mkstemp` introduces the problem, so
  this chmod is required, not optional.
- On **any** exception after `mkstemp` (`except BaseException`, so the `SIGTERM` handler's
  `SystemExit(143)` is covered too), unlink the temp file with `missing_ok=True`, then re-raise.
  An unlink failure is logged at warning and does not mask the original exception. This is one
  deliberate widening beyond predict, which uses `except Exception` and so leaves an orphan on
  SIGTERM. Only `SIGKILL` can still orphan a temp file. `extract_batch`'s
  existing `except OSError` keeps the forward best-effort.

*Rejected:* keeping the fixed `<name>.tmp`. It is simpler, but on the legacy name two concurrent
local runs can still interleave on one temp path.

*Known theoretical edge:* the temp name adds 14 characters to the filename: a leading `.`, a `.`
separator, 8 random characters, and `.tmp`. Any run id of 224 characters or more therefore pushes the temp name past
`NAME_MAX`, and `mkstemp` raises `ENAMETOOLONG`. Contracts' cap is 237. It is an `OSError`, caught
best-effort and logged. Argo's `generateName` ids are about 26 characters, so the edge is not
reachable in practice. Recorded here, and not guarded.

*Windows edge:* if the source manifest is read-only, `chmod(tmp, read.mode)` makes the temp file
read-only too. A failure after that point then makes the cleanup `unlink` fail with
`PermissionError`, which is logged at warning, and the temp file remains. The next forward's
`os.replace` onto a read-only destination can also fail, and is caught best-effort. The cluster is
Linux, so this is recorded and not guarded.

*Not mirrored from predict:* predict removes an `output_dir` it created when the forward fails.
Traits forwards after the per-scan loop, so `output_dir` usually already exists. When it doesn't,
because every in-scope scan failed before writing an envelope, the forward creates it. A leftover
empty or manifest-only `output_dir` is harmless downstream, so there is nothing worth undoing.

*Where the cleanup lives:* the module owns its seams. It uses `import tempfile` and `import os`
(never `from tempfile import mkstemp`, which would bypass a test patch). Cleanup uses
`Path(tmp).unlink(missing_ok=True)`. The module gains a `logger`. The best-effort warning in
`extract_batch` names `read.filename` instead of the hard-coded `run_manifest.json`. Predict's "raise, not best-effort" also stays
unmirrored (user decision, 2026-09-23; the existing spec keeps it). The downstream picture changes
only after two later rollout steps: write-back's reader adopts (design §4 step 3), and the stale
legacy files are deleted (step 5). From then on, a missing forwarded per-run manifest makes
write-back raise `RunManifestMissingError` rather than widen its scope. Until then, write-back's
legacy fallback can still widen it, exactly as today. So best-effort is no worse than today now,
and becomes strictly safer later.

### D4. Same-file no-op stays path-based

Suppose `(input_dir / read.filename).resolve() == (output_dir / read.filename).resolve()`,
typically because `input_dir` and `output_dir` resolve to the same place. Then the forward is
already satisfied, and the forward returns without writing,
as today. This compares paths after symlink resolution; it does not compare inodes. The previous
`shutil.SameFileError` backstop for hardlink aliasing disappears with `copyfile`. For two
hardlinked paths, the replace now atomically swaps in the identical bytes it just read. That
breaks the hardlink but loses no data, which is why no inode check is added. The spec states the
no-op condition as path identity for exactly this reason.

### D5. Warn on a stale legacy read under a known identity

The warning fires when `pipeline_run_id is not None`, `read.is_per_run` is `False`, and
`manifest.pipeline_run_id != pipeline_run_id`. Traits then logs one `WARNING` naming
`read.filename`, the manifest's `pipeline_run_id`, and this process's id. The scope is **still
honored**, and `allow_legacy=True` means exactly that. This is traceability, not policy.

Before `bloomctl` flips (design §4 step 3), the old writer stamps `pipeline_run_id` on every
merge ("newest `pipeline_run_id` wins"). Verified on bloom `staging`:
`download_for_predict.py:455`/`:496`, `os.environ.get("ARGO_WORKFLOW_NAME") or
f"local-{uuid4().hex[:8]}"`. It does not strip the value, but Argo names carry no whitespace. So a
single-chunk legacy read under Argo names the current run, and the warning stays quiet. It also
stays quiet when that legacy file names the current run but still carries the stale 12-key union.
That remains the dominant contamination until design §4 step 5, so the warning's silence is not
evidence of a clean scope.

The warning fires in three cases, and all three are contamination:
- a downloader that failed to write, leaving an older run's manifest in place;
- a concurrent chunk whose downloader merged *after* this one's, so the shared file now names it
  **and carries its `scan_keys`**;
- predict failed before forwarding. The predictor task has `continueOn: failed: true`, so traits
  still runs, and it reads the previous run's `predictions/run_manifest.json`.

This is a warning, not an error, because during the rollout the second case is today's behavior
and design §4 accepts it until bloomctl flips.

### D6. `__main__` crash tuple

The tuple adds `sleap_roots_contracts.RunManifestError` and `ValueError`. `UnicodeDecodeError` is
a `ValueError` subclass; it stays listed explicitly for readability. It can still arise from
paths other than the manifest read. `pydantic.ValidationError` is also a `ValueError`, and it
stays listed too. The exit code is unchanged (Python's default `1`). Only the clean
`Batch aborted:` line is gained.

### D7. Warn when per-run manifests are present but none applies

This covers a run with no identity (`loaded is None`) whose `input_dir` top level holds one or more
`run_manifest.*.json` files. The typical case is a local re-run over a copied cluster tree. The
run falls back to unscoped discovery, which is today's behavior and required by design §2.3.
Before that, it logs one `WARNING` naming the files, so the widening is visible and not silent.

The match is `Path(input_dir).glob("run_manifest.*.json")`, which excludes `run_manifest.json`
itself. It is top-level only, matching the reader. It is a warning, not an error: design §2.3
makes identity-less runs legacy-only on purpose, and a local operator may intend the unscoped run.

### D8. The run identity is not stamped into `Provenance`

`Provenance.pipeline_run_id` stays `None`, as the existing spec requires for this slice. Stamping
it would make envelopes re-emitted by different runs over identical inputs differ, which breaks
byte-stable re-emission and skip-if-done. The spec delta now states this explicitly, because this
change is the first to hold a run identity inside `extract_batch`.

## Risks / Trade-offs

| risk | mitigation |
|---|---|
| Bloom's single-literal pin rejects either a7 or a9 envelopes during the switch, whatever the order | Deploy gate in proposal.md. The Bloom issue (task 0.2) presents a cutover window (Bloom's pattern: loud, recoverable by re-delivery) and a transitional set, and flags that a #766-style guard will trip (bloom#787). The in-cluster template apply waits until Bloom's change is applied |
| First post-deploy run recomputes every scan | Expected: `traits_code_sha` changes per image, and a `contract_version` change forces recompute. Contracts' key derivation is unchanged (`identity.py`/`hashing.py` identical a7→a9) |
| Traits exits `1` on fail-loud, but `continueOn: failed` still runs write-back over `traits/`'s stale manifest | Reachable only once fail-loud can fire (after design §4 step 5); filed against the pipeline (task 0.4) |
| A dev shell with `ARGO_WORKFLOW_NAME` exported makes local runs fail loud | Intended: that shell claims a run identity. Tests are isolated by an autouse `delenv` |
| `allow_legacy=True` still admits a stale 12-key legacy file under Argo | D5 warning; pipeline#82 flips it after the stale files are deleted |
| `os.chmod` is largely a no-op on Windows | The mode test is POSIX-only (`skipif(sys.platform == "win32")`); Windows correctness is covered by the replace/cleanup tests |

## Out of scope

- Flipping `allow_legacy` to `False` (pipeline#82).
- The traits template pin bump in `sleap-roots-pipeline`, after the Bloom a9 acceptance is
  applied. It must co-bump `SRT_TRAITS_CONTAINER_DIGEST`. `check_manifests.py` checks this, but
  only when it is run by hand, since the pipeline repo has no CI for it. It must also
  rewrite the template's "inert today" comment on `ARGO_WORKFLOW_NAME`.
- The a9 wheel's module docstring (`sleap_roots_contracts/run_manifest.py:21-26`) still carries
  the readers-first rationale that pipeline PR #85 retracted. That fix belongs to contracts, and
  nothing here copies the old claim.
- Filtering the per-run manifest out of write-back's own discovery (that is bloomctl's reader).
- GC of accumulated per-run manifests (design §8).
