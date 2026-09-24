# Change: Adopt `sleap-roots-contracts` 0.1.0a9's per-run run-manifest reader in `trait_extractor`

## Why

[talmolab/sleap-roots-pipeline#71](https://github.com/talmolab/sleap-roots-pipeline/issues/71): the
shared `run_manifest.json` accumulates every run's `scan_keys` and is never pruned. It was measured
live: a 1-scan request carried a 12-key manifest, and write-back created `cyl_trait_sources` rows
for 11 unrequested scans. `sleap-roots-contracts` **0.1.0a9** (on PyPI; contracts PRs #38/#39/#41,
all merged) fixes the contract half by naming the manifest per run,
`run_manifest.<pipeline_run_id>.json`. It also adds one reader entry point for all consumers,
`load_run_manifest(directory, pipeline_run_id, *, allow_legacy)`.

The rollout order is normative: **readers before the writer**. That is design of record §4, in
`sleap-roots-pipeline/docs/superpowers/specs/2026-09-21-per-run-run-manifest-identity-design.md`,
as corrected by pipeline PR #85 on 2026-09-23. A writer-first rollout fixes nothing: un-bumped
readers keep scoping to the stale 12-key legacy `run_manifest.json` that still sits in all three
`a4_poc` trees. The per-run name then dies at the first un-adopted hop, and a wrong writer becomes
indistinguishable from an un-adopted hop. Traits is a reader, so it goes now.

The same pass closes the surviving `sleap-roots` residue of
[talmolab/sleap-roots-predict#40](https://github.com/talmolab/sleap-roots-predict/issues/40).
Traits' forward has no temp-file cleanup on failure (design of record §2.1). predict#40 itself
stays open, because its NFS `O_CREAT|O_EXCL` question belongs to the staging lock.

## What Changes

- **Pin bump (BREAKING for the deployed pipeline until Bloom accepts `0.1.0a9`; see Deploy
  gate)**: `sleap-roots-contracts==0.1.0a7` → `==0.1.0a9` in all three `pyproject.toml`
  sites (`[dependency-groups] dev`, `[project.optional-dependencies] dev`, and `extractor`).
  `uv.lock` is re-locked in the same commit, and the test literals move with the pins. A new
  test asserts that all three pins agree.
- **BREAKING (direct `extract_batch` callers): run identity comes from the environment.**
  `extract_batch` gains a keyword `pipeline_run_id`. When a caller omits it, it resolves through
  `sleap_roots_contracts.pipeline_run_id_from_env()`. That is the single "which run am I"
  definition, and design §3.2 requires the writer to use it too. bloomctl adopts it at §4 step 3;
  today it still reads the environment itself, unstripped. Only an explicit `pipeline_run_id=None` opts out. This is the
  first time `trait_extractor` has read `ARGO_WORKFLOW_NAME` (design §2.5). The cluster template
  already sets it; `local-WSL2-*` templates deliberately do not.
- **One snapshot**: `load_run_manifest(input_dir, pipeline_run_id, allow_legacy=True)` is called
  exactly once. `.manifest` scopes discovery, and `.read` (`filename`/`data`/`mode`) drives the
  forward. The second read by name inside the forward goes away. Traits' own `load_run_manifest`
  wrapper is removed.
- **BREAKING: fail loud where the run id is known** (design §2.2). With an identity and no
  manifest under either name, the batch aborts with `RunManifestMissingError` (exit `1`). A
  per-run manifest naming another run raises `RunManifestIdentityError`, and an unusable id raises
  `ValueError`. Both also exit `1`. With no identity, the legacy name is the *correct* name (§2.3,
  §2.9), and its absence keeps today's unscoped behavior.
- **Traceability warnings** (the scope is unchanged by either):
  - A *legacy* manifest read under a known identity, whose `pipeline_run_id` names another run,
    is honored as `allow_legacy=True` requires but logged as a `WARNING` (design D5).
  - Whenever the run is *not* scoped by a per-run manifest (unscoped, or scoped by the legacy
    file) while per-run manifests sit unread beside it, a `WARNING` names them (design D7). This
    is the shape of a copied cluster tree re-run locally.
- **Forward under the name read** (§2.5): the snapshot is republished into `output_dir` as
  `read.filename`, with `read.mode`, atomically. The temp file is removed on failure (design D3).
  It stays **best-effort**, as specified today; that divergence from predict is deliberate.
- **CLI**: `__main__`'s logged-then-reraised tuple gains `RunManifestError` and `ValueError`, so
  every new abort prints `Batch aborted:` before its traceback (design D6).
- **`allow_legacy=True`** at this call site. The flip to `False` is fleet-wide,
  [talmolab/sleap-roots-pipeline#82](https://github.com/talmolab/sleap-roots-pipeline/issues/82),
  after the stale legacy manifests are deleted (design §4 steps 5–6).
- **Docs**: `docs/dev/trait-extractor-service.md` and `docs/changelog.md`. This includes
  correcting two deploy notes that are already stale today, because bloom#685 was resolved.

### Behavior deltas inherited from the contracts reader

| input | before (0.1.0a7 + traits' own reader) | after (0.1.0a9 `load_run_manifest`) |
|---|---|---|
| `input_dir` does not exist | `RuntimeError` from the empty-input guard | **BREAKING:** `FileNotFoundError` naming the directory, before discovery |
| `input_dir` is a regular file | `RuntimeError` from the empty-input guard | **BREAKING:** `NotADirectoryError` naming `<file>/run_manifest[.<id>].json` (POSIX); `FileNotFoundError` naming the directory (Windows) |
| manifest bytes are not UTF-8 | `UnicodeDecodeError` | `pydantic.ValidationError` (`json_invalid`), because it parses from bytes |
| manifest is a dangling symlink | `is_file()` is `False`, so silent unscoped widening | **BREAKING:** `FileNotFoundError` naming the link |
| a *directory* named `run_manifest.json` | `is_file()` is `False`, so silent unscoped widening | **BREAKING:** `IsADirectoryError` (POSIX) / `PermissionError` (Windows) |
| manifest unreadable (`EACCES`) | `OSError` | unchanged |
| identity known, no manifest | unscoped discovery (the id was never read) | **BREAKING:** `RunManifestMissingError` |
| identity known, stale legacy manifest | scoped, silent | scoped, plus a `WARNING` (D5) |

After this change, every row except the last exits `1` with a `Batch aborted:` line. The
missing-directory, regular-file, non-UTF-8 and `EACCES` rows already did. The dangling-symlink,
directory-named-manifest and identity-without-manifest rows previously ran to completion
(exit `0`/`3`) over an unscoped tree.

## Deploy gate (NOT a merge gate): Bloom must accept `0.1.0a9`

Every emitted envelope stamps `provenance.contract_version` from the installed package version, so
this change emits `"0.1.0a9"`. Bloom's write-back RPC checks that stamp against **one** pinned
literal, verified live on 2026-09-23 against bloom `staging`:

- The live body is `insert_cyl_result_envelope(envelope jsonb, p_argo_workflow_name text)` in
  `supabase/migrations/20260917140000_fix_cyl_redelivery_status_fallback.sql:43-53`, with
  `pinned_version constant text := '0.1.0a7'` (`v`-prefix tolerant, otherwise exact).
- It has been redefined since bloom PR #766 (merged to `staging` 2026-09-02; bloom#685 closed
  2026-09-10). A re-pin copied from #766's 1-arg body would recreate a stale overload and regress
  the redelivery fix.
- Bloom also vendors the contract in `contracts/pin.json` and
  `contracts/schema/result_envelope.schema.json`.

**A single literal means some envelopes get rejected whatever the order.** If Bloom re-pins first,
the deployed a7 traits image is rejected until the pin bump. If traits' pin is bumped first, its a9
envelopes are rejected. The Bloom-side choice belongs to Bloom. The issue ([bloom#895](https://github.com/Salk-Harnessing-Plants-Initiative/bloom/issues/895)) presents both
options with evidence:

- **Cutover window (Bloom's established pattern).** Bloom re-pins first, then the traits template
  pin bump follows promptly. This is how #766 and pipeline#52 were sequenced. The window is loud and
  recoverable, not lossy:
  - write-back has no `continueOn`, so a rejection turns the Workflow red;
  - `bloomctl` classifies it as a `contract_version` mismatch (`ingest.py:528-532`);
  - the envelopes stay on disk and are re-delivered on the next run.
- **Transitional accepted set `{0.1.0a7, 0.1.0a9}`, narrowed later.** Bloom has rejected sets twice
  (`repin-cyl-contract-a3/design.md:24,60-61`; `repin-cyl-contract-a7/design.md:11-22`), each time
  because no real acceptance problem existed. That has changed:
  - an a7 image is deployed and emitting;
  - real `0.1.0a7` rows exist;
  - the cluster is shared by staging and production.

  Each row still records its own `contract_version`, so a set loses no provenance.

**The cutover guard.** #766's migration `RAISE`s when rows stamped with the previous literal exist.
Its a7 predecessor tripped on 10 a3 rows (bloom#787). It failed "Apply database migrations" and
cancelled every staging deploy for six days. It was only resolved by restamping rows that happened
to be disposable test fixtures. Today's `0.1.0a7` rows come from real Bloom-dispatched runs. Under
a cutover, a guard copied from #766 **will** trip, so it must be removed or redesigned rather than
restamped around. Under a set, no guard is needed. Filed 2026-09-23 as [bloom#895](https://github.com/Salk-Harnessing-Plants-Initiative/bloom/issues/895).

**Idempotency.** Contracts' key derivation is unchanged: `identity.py` and `hashing.py` are
identical between a7 and a9, and the `ResultEnvelope` schema differs only in `$id`. Keys still
change on deploy, because `traits_code_sha` (baked per image, and an idempotency-key input)
changes with every image. Skip-if-done also recomputes on a `contract_version`-only change. The
first run after deploy therefore recomputes every in-scope scan and inserts new rows (expected).

**Sequencing.**
- Merging is safe. The GHCR `:latest`/`:main` tags float to the a9 image, but nothing consumes
  them: the cluster template is digest-pinned (`sha-689cffb@sha256:…`), `check_manifests.py`
  rejects `:latest`, and the `local-WSL2-*` traits template runs an unrelated legacy image.
- The traits template must not be **applied in-cluster** (`argo template update`) with an a9 image
  until Bloom's a9 acceptance is **applied to the database the cluster write-back targets**. Per
  bloom#863, that is the staging Supabase, even for prod-dispatched runs. "Merged to `staging`" is
  not "applied": bloom#410 and #787 both wedged staging deploys. Under the cutover option, the
  in-cluster apply follows the Bloom apply promptly, so that the rejection window stays short.
- Design §4 step 3, the bloomctl writer flip, is therefore transitively gated on the Bloom change
  too. It must not happen until *both* the predict and traits templates are applied with adopted
  images.
- **The writer can adopt a9 by accident.** `bloomcli/pyproject.toml:28` pins contracts
  `>=0.1.0a7`, unbounded, so any incidental bloomctl image rebuild while traits is held back by
  bloom#895 picks up a9. That alone does not flip the writer, because the per-run filename is a
  code change in `download_for_predict.py`. But treat a bloomctl rebuild during the Bloom gate
  window as a rollout event, not a routine one.
- Predict is not Bloom-gated, because it stamps no `contract_version`. Its gate is the chain in
  design §4:
  1. predict#34 merges (step 0a);
  2. the W&B re-seed (0b);
  3. predict#34 is deployed (0c).
- The next PyPI library release publishes an `extractor` extra that pins `==0.1.0a9`. Anyone who
  runs the extractor locally and then `bloomctl ingest` before Bloom accepts a9 will be rejected.
  The impact is negligible, but it is recorded here.
- The `sleap-roots-pipeline` roadmap frontier (rows 2, 5 and 6) lists traits adoption as unblocked
  and has no Bloom gate. When the Bloom issue is filed (task 0.2), record the gate there.

## Impact

- **Affected specs:**
  - `result-envelope-output` (MODIFIED): "Provenance assembly with deterministic idempotency key"
    (version literal; the run identity SHALL NOT be stamped into `Provenance`), "Batch driver and
    module CLI", and "Run-manifest scoped discovery and copy-forward".
  - `trait-extractor-image` (MODIFIED): "Container image runs the trait extractor over predict
    outputs" and "Slim contracts install via an extractor extra". These carry version literals,
    a no-`ARGO_WORKFLOW_NAME` precondition on the container smoke scenario, and a new normative
    all-pins-agree sentence and scenario.
- **Affected code:** `pyproject.toml`, `uv.lock`, `trait_extractor/run_manifest.py`,
  `trait_extractor/extractor.py`, `trait_extractor/__main__.py`.
- **Affected tests:** `tests/trait_extractor/{conftest.py (new), test_batch.py,
  test_run_manifest.py, test_package_boundary.py, test_envelope.py}`.
- **Affected docs:** `docs/dev/trait-extractor-service.md`, `docs/changelog.md`.
- **Not affected:**
  - the `sleap_roots` library and its runtime dependencies;
  - the trait values;
  - contracts' idempotency-key derivation.
- **Wheel metadata:** only the `dev`/`extractor` extras' `Requires-Dist` moves, at the next
  library release.
- **Not blocked by predict#34 or the W&B re-seed:** 0.1.0a8's shape change was `ModelCard`→`Selector`,
  and `trait_extractor` imports none of it (verified by grep).
- **Related, already filed:** contracts#42, contracts#43, pipeline#82 (`allow_legacy` flip),
  pipeline#83, bloom#894.
- **Follow-ups for the later pipeline pin-bump PR:**
  - co-bump `SRT_TRAITS_CONTAINER_DIGEST`;
  - rewrite the template's "inert today" comment on `ARGO_WORKFLOW_NAME` (lines 54–58), which
    becomes false.
  - Rollout note: once fail-loud is reachable, a traits exit `1` is retried twice. `continueOn:
    failed` then still runs write-back over whatever manifest sits in `traits/`. Filed as [talmolab/sleap-roots-pipeline#86](https://github.com/talmolab/sleap-roots-pipeline/issues/86).
