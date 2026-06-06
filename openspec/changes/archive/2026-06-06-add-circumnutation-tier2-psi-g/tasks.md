# Tasks — add-circumnutation-tier2-psi-g (Tier 2 ψ_g)

Strict TDD: each scope unit ships a **failing test commit** (`test: … (TDD red)`)
then an **implementation commit** (`feat:/fix: … (TDD green)`). No bundling impl +
large test file into one commit (the PR #6 3b7b03e retrospective, #222). Fixup
commits for review findings ship a test alongside any substantive code change.

**Green-between-pairs rule.** After EVERY commit pair below, the full suite stays
green and `black --check` / `pydocstyle` are clean (run the §11 quick gate). This
is load-bearing for the §2 ordering: the foundation migration is **atomic** with
the stub→impl rename so the foundation tests never go red.

## 0. Pre-flight
- [x] 0.1 Confirm branch `add-circumnutation-tier2-psi-g`; `uv sync`; baseline `uv run pytest tests/ -q` green.
- [x] 0.2 Draft the **PR #7 tracking issue** in the vault (`c:\vaults\sleap-roots\circumnutation\`); show the user; do NOT post until OK. **Do NOT file a separate `psig_long_consistency` issue** — it is redundant with the already-roadmapped **PR #13** Layer-3 (`T_nutation ↔ T_psig ±5%`). Instead: (a) annotate the roadmap PR #13 row so it owns BOTH the deferred §7.3 `psig_long_consistency` trait emission AND the consistency test (single-sourced ownership), and (b) record the deferral pointer in this PR's ADDED-requirement spec rationale.

## 1. `_geometry.compute_signed_area` (commit pair — no foundation coupling, safe first)
- [x] 1.1 RED: **absolute anchor** — `compute_signed_area([0,1,1,0],[0,0,1,1]) == -1.0` exactly (y-down negation of standard Shoelace `+1.0`); `< 3` points → `0.0`; non-finite → NaN.
- [x] 1.2 GREEN: implement `compute_signed_area(x, y)` in `_geometry.py` next to `compute_psi_g`; Google docstring documenting the sign convention anchored on the `dψ_g/dt` sign (NOT the ambiguous word "counterclockwise"): `sign(area) == handedness`. Also fix the legacy un-anchored "+1 = counterclockwise" phrasing in the `_geometry.py` MODULE docstring (lines ~17-18) so the locked helper, spec, and theory all use the `dψ_g/dt`-anchored wording.

## 2. `psi_g.compute` schema/structure **+ foundation-test migration (ATOMIC pair)**
> The moment `psi_g.compute` stops raising `NotImplementedError` AND the stub
> callable is renamed `compute_psi_g → compute`, the existing foundation tests
> (`test_stub_callable_raises_with_correct_pr` over `STUB_MODULES`,
> `test_stub_accepts_constants_kwarg` over `STUBS_WITH_CONSTANTS_KWARG`) would go
> red. Therefore the foundation-table migration and the first non-raising
> `psi_g.compute` MUST land in the SAME commit pair.
- [x] 2.1 RED (one commit, two coordinated edits):
  - `tests/test_circumnutation_psi_g.py`: returns `pd.DataFrame`; 8 identity + 4 trait columns in declared order; dtypes 3×float64 + 1×int64; 5-tuple uniqueness. Stub emits the degenerate-row defaults.
  - `tests/test_circumnutation_foundation.py`: remove `psi_g` from `STUB_MODULES` + `STUBS_WITH_CONSTANTS_KWARG`; add `("psi_g","compute")` to `IMPLEMENTATIONS_WITH_CONSTANTS_KWARG` + the matching `elif module_name == "psi_g":` branch (build a **≥24-frame** single-track df and call `fn(df, 300.0, constants=ConstantsT())`). NOTE: this branch only proves **callability** (returns a DataFrame, does not raise) — a straight-line inline df like nutation's would hit the zero-energy guard and emit `T_psig=NaN`, which is fine here; real CWT-path value coverage lives in §5.1. Do NOT add a `T_psig` value assertion to this foundation branch. Also add `psi_g` to the explicit `test_module_logger_is_namespaced` list (removing it from `STUB_MODULES` de-covers it otherwise); the "Calling each remaining stub" enumeration becomes the 5 stubs.
- [x] 2.2 GREEN (one commit): implement the `compute` shell (rename the stub callable) — `groupby` 5-tuple → `_compute_one_track` (returns the degenerate-row defaults for now) → per-plant template merge → declared column order → dtype-enforcement loop (`handedness → fillna(0).astype(int64)`, others `float64`). Suite is GREEN after this pair.
- [x] 2.3 RED+GREEN: units-vocabulary guard test (all 4 trait units ∈ `PIPELINE_UNIT_VOCABULARY`); fix the stale `atan2(dy/dt, dx/dt)` docstring carried by the old stub.

## 3. Input-validation boundary (commit pair)
- [x] 3.1 RED: non-DataFrame `trajectory_df`; invalid `trajectory_df`; bad `cadence_s` value→ValueError / type→TypeError (reuse `temporal_cwt._validate_cadence_s`); `constants` wrong type→TypeError naming the field; invalid SG override (even `SG_WINDOW_DETREND`, or `≤ SG_DEGREE`)→ValueError naming the field.
- [x] 3.2 GREEN: implement `_check_constants` (SG fields only; defer CWT-field validation to `compute_scaleogram`) + wire validators.

## 4. Raw, CWT-free traits (commit pair)
- [x] 4.1 RED: `handedness` keyed off the generator's locked convention — `generate_trajectory(handedness=+1)` → output `handedness==+1`, `handedness=-1` → `−1`; the `1e-9` zero-guard collapses straight-line float dust to 0; `delta_E_amplitude_proxy_px_per_frame` on a known constant-speed synthetic; handedness↔area agreement `int(np.sign(helix)) == handedness`.
- [x] 4.2 GREEN: implement step-4 of `_compute_one_track` (finite mask → ψ_g endpoints → `handedness`; `np.median` speed → `delta_E`; `compute_signed_area` → `helix`). Pass for `N ≥ 3`, independent of the CWT path.
- [x] 4.3 RED+GREEN: **conditioning-isolation** test (spec scenario "conditioning affects only T_psig") — assert the 3 raw traits are bit-identical with vs without the CWT path / under a `SG_WINDOW_DETREND` override (e.g. compare a `3≤N<24` track where the CWT is skipped against the same prefix embedded in an `N≥24` track).

## 5. `T_psig_median_s` CWT path (commit pair)
- [x] 5.1 RED: known-period synthetic (`T ∈ {3333, 4500}`, `n_frames ≥ 575`, `noise_sigma_px=0`) recovers `T_psig_median_s` within ±10% (cite nutation `test_2C2`); assert **no** `RuntimeWarning` via `warnings.catch_warnings()` + `simplefilter("error", RuntimeWarning)` (caplog does NOT capture numpy warnings — nutation `test_2F7` precedent).
- [x] 5.2 GREEN: implement step-5 — length guard (`len(ψ_g) < SG_WINDOW_DETREND` → NaN); `compute_sg_detrended`; **zero-energy guard** (`np.allclose(detrended, 0.0)` → NaN, skip CWT); `try: compute_scaleogram → extract_ridge → smooth_ridge`, COI-interior `~in_coi` nanmedian with empty/all-NaN guard; `except ValueError` + post-detrend finite-check as `# pragma: no cover` defensive mirrors (cite the length/zero-energy guards that make them unreachable).

## 6. Degenerate / edge cases (commit pair)
- [x] 6.1 RED (all assert no `RuntimeWarning` via `simplefilter("error", RuntimeWarning)`): `N < 3` (2-row direct DataFrame) → all-degenerate row; `3 ≤ N < 24` (15-frame) → `T_psig=NaN`, raw traits defined, no exception; **stationary `N≥24`** (`generate_trajectory(amplitude_px=0, growth_rate_px_per_frame=0, noise_sigma_px=0, n_frames=64)`) → `T_psig=NaN` (zero-energy guard) AND no spurious 600 s period; NaN-injection (à la `test_2F7b`) handled gracefully.
- [x] 6.2 GREEN: implement step-1/2 finite-mask + `N<3` short-circuit; confirm the zero-energy and length guards cover the table.

## 7. Determinism + logging (commit pair)
- [x] 7.1 RED: **determinism** — same-process `psi_g.compute` twice: 3 float columns bit-identical at `atol=0`, `handedness` exactly equal; a captured 3-value canary at `atol=1e-6` (cross-OS floor) on a **fixed-seed, noisy** fixture `generate_trajectory(random_state=0, noise_sigma_px=0.5, n_frames=575, T_nutation_s=3333, cadence_s=300)` so the canary actually exercises the scipy float stack (nutation `test_2B2` precedent — `noise_sigma_px=0` would test nothing). **Logging** — `caplog` at DEBUG: exactly one DEBUG record from `sleap_roots.circumnutation.psi_g`, message starts `"psi_g.compute("`, contains `n_tracks=` and `cadence_s=`, has **no** `coordinate=` token, no INFO/WARNING/ERROR.
- [x] 7.2 GREEN: add the single `logger.debug(...)` at the start of `compute` (after validation) — `psi_g.py` already declares `logger = logging.getLogger(__name__)`, so only the call body is new; confirm determinism (no random/ordering nondeterminism).

## 8. Cross-tier consistency (commit pair)
- [x] 8.1 RED: synthetic convention-lock — define `wrap_to_pi`/`circular_mean` as **test-local** helpers (mirror nutation's `_make_hand_crafted_ridge` test-local pattern). Angle-identity fixture (`amplitude_px=0`, θ ∈ {0.3, −2.0}) asserts `abs(wrap_to_pi(circular_mean(ψ_g) − (π/2 − θ))) < 1e-6` and `handedness == 0`; handedness fixture (`amplitude_px>0`, `handedness=±1`) asserts the planted sign.
- [x] 8.2 GREEN: ensure the RED passes (no product code change expected beyond ψ_g exposure already built).
- [x] 8.3 GREEN-phase: plate-001 reconciliation test over the 6 proofread tracks (mirror the test-local `_load_proofread_track_df` loader, `Path()`-based, Git-LFS fixture) — skip NaN `principal_axis_angle` tracks; assert `≥N/6` within `_PSIG_AXIS_RECONCILE_TOL_RAD`; the captured `N`/tolerance MUST clear the pre-committed floor `N ≥ 2` AND `tol ≤ 0.35 rad`; record the observed per-track deviation distribution (max + spread) in the test docstring / commit message; "GREEN-phase Reconciliation" docstring.

## 9. Multi-track integration + docs
- [x] 9.1 RED+GREEN: multi-track integration test (mixed track lengths incl. a `<24` and a `≥24` track in one `trajectory_df`) → correct per-track rows, declared column order, no cross-track contamination.
- [x] 9.2 Patch `docs/circumnutation/theory.md`: §7.3 `handedness` row → "net unwrapped ψ_g rotation over all finite frames (COI-free; +1 ⇔ positive dψ_g/dt)"; §7.3 `delta_E` row → px/frame; §6.3 conditioning note "smooth → SG-detrend (residual)". **Preserve the original wording** via Appendix B (Corrections) entries citing PR #7 + the reasons (do NOT silently overwrite).
- [x] 9.3 `docs/changelog.md` entry; confirm `uv run mkdocs build` renders `psi_g.compute` + `_geometry.compute_signed_area`.

## 10. Verification gates (must pass before requesting merge)
- [x] 10.1 `uv run pytest tests/ -q` green; ≥ 84% project coverage; aim ≥ 90% on `psi_g.py` + the new `_geometry` helper (defensive branches `# pragma: no cover` with invariant comments).
- [x] 10.2 `uv run black --check sleap_roots tests` clean.
- [x] 10.3 `uv run pydocstyle --convention=google sleap_roots/circumnutation/` clean.
- [x] 10.4 `uv lock --check` clean; `uv run mkdocs build` builds.
- [x] 10.5 `npx openspec validate add-circumnutation-tier2-psi-g --strict` valid.
- [x] 10.6 CI matrix Ubuntu/Windows/macOS green on the final commit.
- [x] 10.7 `/copilot-review` + `/review-pr` findings reconciled or deferred to tracked issues.

## 11. Quick gate (run after EVERY commit pair, not just at the end)
- [x] 11.x `uv run pytest tests/test_circumnutation_psi_g.py tests/test_circumnutation_foundation.py -q && uv run black --check sleap_roots tests && uv run pydocstyle --convention=google sleap_roots/circumnutation/` — catches the §2 atomicity hazard and cross-cutting breakage early. After the §1 `_geometry` pair specifically, ALSO run `tests/test_circumnutation_kinematics.py tests/test_circumnutation_synthetic.py` (they exercise `_geometry.compute_psi_g`) — or just run the full suite — so a stray `_geometry.py` edit can't slip through.
