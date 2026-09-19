# Migration Policy — Single Source of Truth (`evalshift.yaml` → push → server)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Do one task at a time; each task is independently shippable. Every logic task is TDD: write the failing test, watch it fail, then code.

**Goal:** A project's migration policy is written once, in `evalshift.yaml`, travels with every pushed run, and is what every surface gates on: the local `compare --policy-gate`, the run's Policy tab, the hosted policy-check endpoint, the PR list "blocked" signal, and the GitHub Action's commit status. The web app displays the policy; it no longer edits it.

**Scope:** Four repositories, in deploy order: `[server]` → `[cli]` → `[client]` → `[action]`. `thresholds` (the free-form, non-gating key) is explicitly **out of scope** and keeps its current push-sync behaviour. *(Superseded 2026-09-19 — Phase 5's `[cli]` item deleted `thresholds` outright; see the note there.)*

**Tech Stack:** server — FastAPI, pydantic, raw SQL via SQLAlchemy `text()`, alembic, pytest on SQLite, `ruff` + `mypy --strict`. cli — Python 3.11+, pydantic, typer/rich, pytest, `mypy --strict`. client — React + TypeScript, vitest + testing-library. action — stdlib Python, pytest.

---

## Part 1 — Why (verified 2026-09-18)

Two independent policy engines gate the same run and can disagree:

| Surface | Policy it reads | Engine | Evidence |
|---|---|---|---|
| `evalshift compare --policy-gate`, run Policy tab, bundle `decision` | `evalshift.yaml:migration_policy` (9 fields, CLI defaults) | `analysis/policy.py:evaluate_migration_policy` | `hosted/bundle.py:135-147`, `cli/commands/compare.py:744-752` |
| `GET /runs/{id}/policy-check`, PR list `latest_status`, GitHub Action commit status | `projects.migration_policy_json` (6 fields, no defaults, web-only editor) | `app/policy/service.py:evaluate_policy` | `app/policy/service.py:895-918`, `app/runs/service.py:1838-1906`, action `scripts/evalshift_action.py:911-964` |

- The CLI never uploads `migration_policy`. Only `thresholds` ride on `POST /runs` (`hosted/client.py:102-128`, `app/runs/schemas.py:30-36`).
- A project nobody configured on the web returns `inconclusive`; the Action treats that as not-a-failure (`_policy_gating`, `status == "inconclusive"` → `should_fail = False`). **The PR gate is silently off** even when the repo's yaml has a strict policy.
- Schema drift: CLI has `max_tool_divergence`, `tool_argument_drift_floor`, `fail_on_dropped_params`; the server model is `extra="forbid"` without them (`app/policy/schemas.py:8-11` cites a `FEATURES_TODO.md` that does not exist in any repo). The server cannot evaluate tool divergence at all — `run_policy_metrics` has no divergence column (`app/policy/service.py:790-792`).
- `PolicyCheckResponse.verdict` vs `.status` is the documented "these can differ" seam (`app/policy/schemas.py:96-99`).

## Part 2 — Decisions

**D1 — The policy rides inside the bundle, as `decision.policy`.**
Not in `POST /runs` metadata. `BUNDLE_SPEC.md:10` and server `CLAUDE.md:12` already say the bundle is the source of truth for policy decisions; a decision that does not say which budgets produced it is incomplete. This also fixes a local gap: `migration_decision.json` becomes self-describing, so `report` can state the budgets the verdict was made under without re-reading the config. `decision.policy` is the **resolved** policy (CLI defaults applied, all nine top-level fields present, `slices` present even if empty) or `null` when `migration_policy` is unset. Server `Decision` is `extra="forbid"` (`app/runs/bundle.py:509`), so the field is added there as optional **before** any CLI emits it.

**D2 — For a run that carries a policy, the server's answer *is* the CLI's decision.**
`policy-check` returns `status = runs.verdict`, the stored `run_budget_results` and `run_regressions`, and the bundle's `reason`, with `policy_source = "run_policy"`. No re-evaluation: the server engine covers six of nine budgets and would disagree with the bundle exactly where the extra three matter. `verdict` and `status` are equal by construction. Existing runs without a snapshot keep today's re-evaluation against the legacy project column, with `policy_source = "project_policy"`. The dual code path is the migration path and is removed once no legacy-policy project has been pushed to in 90 days (tracked as a follow-up, not in this plan).

**D3 — Per-run snapshot, stored in a new nullable `runs.policy_jsonb` column.**
Written at finalize from `decision.policy`. A run is gated on the policy it was pushed with, forever; a later yaml change never rewrites history. Nothing is written to `projects.migration_policy_json` on push. That column becomes **legacy, read-only**: `PATCH /projects/{id}` rejects `migration_policy` with 422, the create endpoint stops accepting it, and the client's edit/create/reset flows are deleted.

**D4 — "The project's policy" for display = the policy of the latest available run on the default branch, else the latest available run, else the legacy column, else none.**
Served by a new `GET /projects/{id}/policy`. The web card shows it read-only, names the run it came from, and offers "Copy as YAML". The legacy case shows a banner telling the owner to move it into `evalshift.yaml`, with the YAML pre-rendered. The empty case shows the starter template as YAML to paste. `GET /policy/template` stays (it feeds that snippet); the client keeps no copy of the numbers.

**D5 — Server `MigrationPolicy` schema is widened, not required-nine.**
`max_tool_divergence`, `tool_argument_drift_floor` (top-level and per-slice) and `fail_on_dropped_params` (top-level only) are added as `Optional`, default `None`. Legacy six-field rows keep validating; a CLI snapshot always fills all nine. Bounds mirror the CLI (`config/models.py:498-525`). `extra="forbid"` stays.

**D6 — No policy is loud, not silent.**
`evalshift push` prints a warning when the run carries no policy. `policy-check` gains `policy_source = "none"` reason text that names the fix. The Action emits a GitHub `::warning::` annotation and a distinct commit-status description when `policy_source` is `none`; it still does not fail the job (behaviour change to failing is a documented opt-in via a new `require-policy` input, default `false`).

**D7 — Adoption hint for projects that only have a web policy.**
`RunUploadResponse` gains `legacy_project_policy: dict | null` (the column value, when set). If the CLI's config has no `migration_policy` and the server reports a legacy one, `push` prints it as a ready-to-paste `migration_policy:` YAML block after the upload. The legacy column is never cleared automatically.

**D8 — Permissions.** Pushing a policy needs only `run:create` — it is evidence about the run, like the rest of the decision block. `policy:configure` remains for `thresholds` only. A PR can loosen its own gate by editing the yaml; that is visible in the diff and is how every CI config works. Org-level floors are a possible later layer, not part of this plan. **[Superseded 2026-09-19:** Phase 5 removed `thresholds` from the CLI, so `policy:configure` now gates nothing the CLI is able to send. A `[server]` cleanup of that permission and of the orphaned `canonical_thresholds` response field is unscheduled.**]**

**Rollout order matters.** A new CLI emitting `decision.policy` against an old server is rejected at finalize (`extra="forbid"`). Ship and deploy `[server]` Phase 1 before releasing `[cli]` Phase 2. Version bumps are deferred to release per project convention.

---

## Phase 1 — `[server]` accept, store, and answer from the run's own policy

### Task 1.1 — Widen `MigrationPolicy` / `SliceMigrationPolicy` to the CLI's field set (D5)

**Files:** `app/policy/schemas.py`, `tests/test_phase3_policy.py`, `tests/test_phase_d2_policy_no_default.py`

- [x] Test: a nine-field policy (the CLI's resolved shape, including `slices` with `max_tool_divergence` / `tool_argument_drift_floor` overrides) validates; a six-field legacy policy still validates with the new fields `None`; an unknown key is still rejected; out-of-bounds values for the new fields (e.g. `tool_argument_drift_floor: 1.5`) are rejected.
- [x] Add to `SliceMigrationPolicy`: `max_tool_divergence: float | None` (0–1), `tool_argument_drift_floor: float | None` (0–1).
- [x] Add to `MigrationPolicy`: the same two, plus `fail_on_dropped_params: bool | None = None`. All optional. Keep the six existing budgets required.
- [x] Rewrite the module docstring: remove the `FEATURES_TODO.md §9a/§9b` references (file does not exist); state that the server **stores** the full CLI shape and **evaluates** only the six for legacy runs (D2).
- [x] `STARTER_POLICY_TEMPLATE` gains the CLI's defaults for the three new fields (`max_tool_divergence=0.20`, `tool_argument_drift_floor=0.9`, `fail_on_dropped_params=False`) so a pasted template is a complete yaml block.
- [x] `make lint && make test`.

### Task 1.2 — Accept `decision.policy` in the bundle (D1)

**Files:** `app/runs/bundle.py`, `schemas/bundle_manifest.schema.json` (via `make export-schemas`), `tests/test_bundle_schema_parity.py`, `BUNDLE_SPEC.md`

- [x] Test (parity): a full bundle with `decision.policy` set to a nine-field policy validates in both validators; a bundle with `decision.policy: null` validates; a bundle **omitting** `decision.policy` validates (old CLIs); a policy with an unknown key is rejected by both.
- [x] Add `policy: MigrationPolicy | None = None` to `Decision`. Import from `app.policy.schemas` — check for an import cycle (`app/policy/schemas.py` imports `BudgetResult`/`BlockingRegression` from `app.runs.bundle`). If cyclic, move `MigrationPolicy`/`SliceMigrationPolicy` into `app/runs/bundle.py` and re-export from `app.policy.schemas`.
- [x] `make export-schemas`; commit the regenerated JSON schema.
- [x] `BUNDLE_SPEC.md`: under Decision, add `policy` — "the resolved migration policy the verdict was computed under; `null` when the CLI had none; absent in bundles from CLI < the release that ships Phase 2. The server stores it verbatim and gates on it (D2)."
- [x] `make lint && make test`.

### Task 1.3 — Migration: `runs.policy_jsonb`

**Files:** `migrations/versions/202609180100_phase17_run_policy_snapshot.py`, `app/runs/service.py` (`RUN_COLUMNS`, `_run_from_row`, the `Run` record model), `tests/test_db_harness.py` or the existing migration-smoke test

- [x] Migration `202609180100`, `down_revision = "202609160100"`: `ALTER TABLE runs ADD COLUMN policy_jsonb JSONB NULL`. Docstring in the repo's style: why per-run (history never rewritten), why nullable (pre-Phase-2 CLIs, and runs with no policy), why the project column is not touched.
- [x] Add `policy_jsonb` to `RUN_COLUMNS`; expose as `policy: dict[str, Any] | None` on the run record; parse with the same `_coerce_json_object` used for `summary_jsonb`.
- [x] Test: the SQLite harness creates the column; `_run_from_row` round-trips a policy dict and a `NULL`.
- [x] `make lint && make test`.

### Task 1.4 — Write the snapshot at finalize

**Files:** `app/runs/service.py` (finalize UPDATE near `:835`), `app/runs/detail_writer.py` (optional: keep it in `service.py` beside `summary_jsonb`), `tests/test_phase4_runs_api.py`

- [x] Test: finalizing a bundle with `decision.policy` stores it; `GET /runs/{id}` (or the run record) exposes it; finalizing a bundle without the field stores `NULL`; re-finalize after `_reset_failed_run` clears it (add `policy_jsonb = NULL` to the reset at `:714`).
- [x] Persist `bundle.decision.policy.model_dump(mode="json")` (or `None`) into `policy_jsonb` in the same UPDATE that writes `verdict` and `summary_jsonb`.
- [x] `make lint && make test`.

### Task 1.5 — `policy-check` answers from the snapshot (D2)

**Files:** `app/policy/service.py`, `app/policy/schemas.py` (`PolicySource`, `PolicyCheckResponse`), `tests/test_phase3_policy.py`, `tests/test_phase_d2_policy_no_default.py`

- [x] Extend `PolicySource = Literal["run_policy", "project_policy", "none"]`.
- [x] Test: a run with a stored snapshot returns `status == run.verdict`, `policy_source == "run_policy"`, `policy` equal to the snapshot, `budgets` from `run_budget_results` (overall scope + slices, as the existing `RunBudgetsResponse` reads them), `blocking_regressions` from `run_regressions`, `reason` from the stored decision. Cover all four verdicts. A **legacy** project policy present on the project must be ignored when a snapshot exists.
- [x] Test: a run without a snapshot on a project with a legacy policy keeps today's behaviour and `policy_source == "project_policy"`.
- [x] Test: a run without a snapshot on a project with no policy → `inconclusive`, `policy_source == "none"`, and `reason` names the fix ("add `migration_policy` to evalshift.yaml and push again").
- [x] Implement `evaluate_run_policy`: branch on `run.policy is not None`. Add `load_stored_verdict_evidence(session, run_id)` that reads `run_budget_results` (all scopes) and `run_regressions`; do not reuse `load_stored_decisions`, whose docstring explicitly excludes limits/pass-fail for the re-evaluation path.
- [x] Store `decision.reason` if it is not already persisted (check `summary_jsonb` / `run_recommendations`; if absent, add `reason` to the finalize UPDATE as a column — `runs.decision_reason TEXT NULL` in the Task 1.3 migration).
- [x] Update `PolicyCheckResponse` docstring: `status` equals `verdict` whenever `policy_source == "run_policy"`.
- [x] `make lint && make test`.

### Task 1.6 — PR list `latest_status` uses the snapshot

**Files:** `app/runs/service.py:list_pull_requests`, `tests/test_phase3_pull_requests.py`

- [x] Test: a PR whose latest run has a snapshot reports `latest_status == latest run's verdict` regardless of the project's legacy policy; a PR whose latest run has no snapshot falls back to the legacy evaluation.
- [x] In the fold, when `record.policy is not None`, set `latest_status = record.verdict` and skip that run in the batched `load_stored_decisions` call.
- [x] `make lint && make test`.

### Task 1.7 — `GET /projects/{id}/policy` (D4)

**Files:** `app/projects/router.py`, `app/projects/schemas.py`, `app/orgs/service.py` or a new `app/projects/policy_service.py`, `tests/test_phase_a3_project_update.py` (or a new `tests/test_phase17_project_policy.py`), `tests/test_phase8_route_authz.py`

- [x] Schema `ProjectPolicyResponse`: `source: Literal["run_policy", "legacy_project_policy", "none"]`, `policy: MigrationPolicy | None`, `run: RunSummary | None` (the run it came from), `template: MigrationPolicy` (the starter, so the client makes one call).
- [x] Test: default-branch run wins over a newer non-default-branch run; with no default-branch run the newest available run wins; with no snapshot runs the legacy column is returned with `source == "legacy_project_policy"`; empty project → `none`; requires `policy:read` (route-authz test).
- [x] Query: `SELECT policy_jsonb, {RUN_COLUMNS} FROM runs WHERE project_id = :p AND status = 'available' AND deleted_at IS NULL AND policy_jsonb IS NOT NULL ORDER BY (branch = :default_branch) DESC, created_at DESC LIMIT 1`.
- [x] `make lint && make test`.

### Task 1.8 — Freeze the legacy column: reject writes (D3)

**Files:** `app/projects/router.py:update_project`, `app/projects/schemas.py:ProjectPatch`, `app/orgs/router.py` / `app/orgs/service.py:create_project`, `tests/test_phase_a3_project_update.py`, `tests/test_phase8_route_authz.py`

- [x] Test: `PATCH /projects/{id}` with `migration_policy` (any value, including `null`) → 422 with detail `"migration_policy is configured in evalshift.yaml and synced on push"`; a PATCH with only `name` still works. `POST /orgs/{org}/projects` with `migration_policy` → 422 same message.
- [x] Remove `migration_policy` from `ProjectPatch` and the create payload (pydantic `extra="forbid"` on those models yields the 422; if they are not `forbid`, add an explicit check so the message is the one above). Delete `update_migration_policy` plumbing from `organizations.update_project`; keep the read path and the audit-diff for the column so history still renders.
- [x] Keep `Project.migration_policy` in the public read model — the client shows it in the legacy banner.
- [x] Remove the `policy:configure` requirement from anything policy-related that remains (there should be nothing left; `thresholds` keeps it). *(Superseded 2026-09-19 — Phase 5's `[cli]` item deleted `thresholds` outright; see the note there.)*
- [x] `make lint && make test`.

### Task 1.9 — Adoption hint in `RunUploadResponse` (D7)

**Files:** `app/runs/schemas.py:RunUploadResponse`, `app/runs/service.py:initiate_run` / `_existing_run_response`, `tests/test_phase4_runs_api.py`

- [x] Test: `POST /runs` on a project with a legacy policy returns `legacy_project_policy` equal to it; on a project without one returns `null`.
- [x] Add `legacy_project_policy: dict[str, Any] | None = None`; populate from `project.migration_policy` in both the fresh and the existing-run responses.
- [x] `make lint && make test`.

### Task 1.10 — Docs `[server]`

**Files:** `BUNDLE_SPEC.md` (done in 1.2), `CLAUDE.md` (only if a workflow note changes), `app/policy/schemas.py` docstrings, OpenAPI descriptions

- [x] Describe `policy_source` values and the D2 rule in the `policy-check` route docstring.
- [x] Note in `BUNDLE_SPEC.md` "Compatibility" section: bundles lacking `decision.policy` are gated on the legacy project policy or not at all.

---

## Phase 2 — `[cli]` put the resolved policy into the decision and the bundle

### Task 2.1 — `MigrationDecision.policy` (D1)

**Files:** `src/evalshift_cli/analysis/policy.py`, `tests/unit/test_policy.py`, `tests/unit/test_analyze_command.py`

- [x] Test: `evaluate_migration_policy(...)` returns a decision whose `policy` equals `policy.model_dump()` (all nine fields + `slices`); `inconclusive_decision(...)` returns `policy is None`; `to_dict()`/`from_dict()` round-trip both; `from_dict()` of a pre-existing `migration_decision.json` without the key still loads (`policy=None`).
- [x] Add `policy: dict[str, Any] | None = None` to the dataclass (dict, not the pydantic model, so `asdict` stays trivial). Populate in both constructors.
- [x] `analyze` already writes the decision to `migration_decision.json`; assert the key is present in the written file.
- [x] `uv run ruff check && uv run mypy && uv run pytest`.

### Task 2.2 — Bundle carries `decision.policy`

**Files:** `src/evalshift_cli/hosted/bundle.py`, `tests/unit/test_bundle_shape.py`, the CLI's copy of `schemas/bundle_manifest.schema.json` if it vendors one (check `grep -rn bundle_manifest.schema src tests`)

- [x] Test: a built bundle with `migration_policy` configured has `decision.policy` with nine top-level keys and `slices`; without it, `decision.policy is None`. If the CLI validates bundles against the vendored JSON schema, copy the regenerated schema from Task 1.2 and assert the built bundle validates.
- [x] No code change should be needed beyond Task 2.1 (`decision.to_dict()` already flows through). Verify, then close.

### Task 2.3 — `push` warnings and the adoption hint (D6, D7)

**Files:** `src/evalshift_cli/hosted/push.py`, `src/evalshift_cli/hosted/client.py` (no change expected), `tests/unit/test_hosted_cli.py`, `tests/unit/test_push_validation.py`

- [x] Test: pushing a bundle whose `decision.policy` is `null` prints `! this run carries no migration policy; the hosted gate will report inconclusive — add migration_policy to evalshift.yaml` (once, yellow, same style as `_warn_threshold_drift`).
- [x] Test: when the config has no `migration_policy` and the initiate response carries `legacy_project_policy`, `push` prints a `migration_policy:` YAML block containing that policy, preceded by `this project has a policy configured in the web app; move it into evalshift.yaml:`; when the config **has** one, nothing is printed even if the server sends a legacy policy.
- [x] Implement `_warn_missing_policy(console, bundle)` and `_print_legacy_policy_hint(console, config_path, response)`. Render YAML with the project's existing YAML writer (the one `init` uses via `render_minimal_config`); do not hand-format. Legacy policies have six fields; validate through `MigrationPolicy` first so the printed block is exactly what the CLI will accept.
- [x] `uv run ruff check && uv run mypy && uv run pytest`.

### Task 2.4 — Docs `[cli]`

**Files:** `docs/hosted.md`, `docs/configuration.md`, `README.md` (policy section, if any), `docs/llms*.txt` if the docs sync script needs re-running

- [x] `docs/configuration.md`: `migration_policy` is the single source of truth; it is snapshotted into every pushed run; the web app shows it and cannot edit it.
- [x] `docs/hosted.md` "What `push` sends": add `decision.policy` to the block table; document the two new warnings and the adoption hint; add a troubleshooting row "`this run carries no migration policy`".
- [x] Run the docs/llms sync if the repo has one (see release notes memory: "docs llms synced").

---

## Phase 3 — `[client]` display, never edit

### Task 3.1 — API layer

**Files:** `src/lib/api.ts`, `src/lib/api.test.ts` (if present)

- [x] Add `ProjectPolicy` type and `api.projectPolicy(projectId)` → `GET /projects/{id}/policy`.
- [x] Extend `MigrationPolicy` type with the three optional fields; extend `PolicyCheck.policy_source` union with `"run_policy"`.
- [x] Remove `migration_policy` from `api.updateProject`'s body type.

### Task 3.2 — Project settings card becomes read-only (D4)

**Files:** `src/pages/app/project/ProjectSettings.tsx`, `src/pages/app/project/policyFields.ts`, delete `src/pages/app/project/EditPolicyDialog.tsx`, `src/pages/app/project/ProjectSettings.test.tsx`

- [x] Tests (replace the edit/reset/create suites at `ProjectSettings.test.tsx:361-510`):
  - `source: "run_policy"` renders all nine budgets, the line "From run `<short id>` on `<branch>`, pushed `<date>`" linking to the run, and no Edit/Create/Reset buttons even for an owner.
  - `source: "legacy_project_policy"` renders the six budgets plus a banner "Configured in the web app. Move it into `evalshift.yaml` — editing here is no longer possible." and a "Copy as YAML" button that writes the yaml block to the clipboard (mock `navigator.clipboard`).
  - `source: "none"` renders the empty state with the starter template rendered as a `migration_policy:` YAML block and a copy button; the text says the gate reports `inconclusive` until a run is pushed with a policy.
  - The card no longer depends on `policy:configure`; a member sees the same content as an owner.
- [x] `policyFields.ts`: extend `POLICY_FIELDS` to nine (add `max_tool_divergence` percent, `tool_argument_drift_floor` percent, `fail_on_dropped_params` boolean → render "yes/no"); delete `validatePolicy`, `toPolicyValues`, `percentMax`, `helpText` and everything only the dialog used. Add `toYaml(policy)` (small hand-rolled renderer for this flat shape plus one level of `slices`; do not add a YAML dependency).
- [x] Delete `EditPolicyDialog.tsx`, the `"policy"` edit target, `onResetPolicy`, `confirmReset`, and the `api.policyTemplate` call (the template now arrives inside `api.projectPolicy`).
- [x] Load `api.projectPolicy` on mount and on project change; loading/error states use the page's existing `LoadingState`/`ErrorState`.
- [x] `npm run lint && npm run typecheck && npm test`.

### Task 3.3 — Run detail Policy tab shows its policy source

**Files:** `src/pages/app/runs/detail/tabs/PolicyTab.tsx`, `src/pages/app/runs/detail/tabs/PolicyTab.test.tsx`, `src/pages/app/runs/detail/fetchers.ts`

- [x] Test: when `policyCheck.policy_source === "run_policy"` the tab header reads "Gated under the policy pushed with this run"; `"project_policy"` reads "Gated under the project's legacy web policy"; `"none"` reads "Not gated — no policy was pushed with this run" with a link to the settings card.
- [x] Add `fetchers.policyCheck` (wire the already-existing `api.policyCheck`, which today has no non-test caller) and render the one-line source header above the budget table. Budgets keep coming from `api.runBudgets`.
- [x] `npm run lint && npm run typecheck && npm test`.

### Task 3.4 — Onboarding checklist copy (only if it mentions the policy)

- [x] `grep -rn "policy" src/pages/app/onboarding src/components/*Checklist*` — if a step says "create a policy in settings", reword to "add `migration_policy` to evalshift.yaml and push".

### Phase 3 landed — deviations

- **3.1 is not independently green.** Dropping `migration_policy` from `api.updateProject`'s body
  type breaks its only two callers, which are exactly what 3.2 deletes and rewrites, so 3.1 and
  3.2 are one commit. `api.policyTemplate` was kept (no caller, like `api.policyCheck` before
  3.3) since it is the endpoint's only client-side name.
- **`MigrationPolicy.slices` is now typed** (`Record<string, SliceMigrationPolicy>`, new exported
  type) rather than `Record<string, unknown>` — `toYaml` needs to walk it.
- **3.3: `RunFetchers.policyCheck` is optional and absent from `sharedRunFetchers`.** The server
  mounts `policy-check` under `/runs/{id}` only; there is no `/share/{token}` counterpart, so the
  share surface would have pointed at a 404. The tab renders no source line there. A failed or
  in-flight policy check degrades to the budget table alone rather than blanking the tab.
- **3.4 was a no-op**, verified: no onboarding or checklist copy mentions creating a policy.
- **Four files outside the task list asserted the old model** and were corrected, since Phase 1/2
  had already made them false:
  - `docs/pages/MigrationPolicy.tsx` — documented the Settings create/edit dialog and claimed
    "editing the policy re-decides *past* runs", which D3 reverses. Rewritten around the yaml →
    push → snapshot model; `#editor`/`#reeval` replaced by `#source`/`#snapshot`/`#display`/
    `#legacy` (no inbound referrers); nav blurb at `docs/data/nav.ts` updated with it.
  - `docs/pages/Verdicts.tsx` — the "server-side enforcement" callout claimed tightening a budget
    can flip a stored run to FAIL.
  - `app/permissionCatalog.ts` — `policy:configure` was labelled "Edit the migration policy"; per
    D8 it now guards `thresholds` only.
  - `app/help/topics/Baselines.tsx` — the in-app guide drew a `DialogFigure` of the deleted
    "Create migration policy" dialog, and imports `POLICY_FIELDS`, so widening it to nine silently
    rendered three blank inputs. The figure is now the `migration_policy:` block itself, rendered
    by the same `toYaml` the settings card copies; the `policy:configure` `CannotNotice` is gone.
- **Not touched, deliberately:** `compare/data/langfuse.ts` had one stale claim (corrected); blog
  posts are dated artifacts and were left alone.
- Acceptance item 2's open question — whether "the previous run's policy is still shown as
  current" confuses — is answered by the card naming the run and branch each policy came from.

---

## Phase 4 — `[action]` loud when ungated (D6)

### Task 4.1 — `policy_source: none` annotation and status text

**Files:** `scripts/evalshift_action.py`, `tests/test_evalshift_action.py`, `action.yml`, `README.md`

- [x] Test: `_policy_gating` with `status == "inconclusive"` and `policy_source == "none"` → `should_fail False`, summary `the gate is off — no migration policy was pushed with this run; add migration_policy to evalshift.yaml`, and a `::warning::` line on stdout (workflow annotation). With `policy_source == "run_policy"` no annotation.
- [x] Test: new input `require-policy: true` makes that same case `should_fail True`, conclusion `failure`; default `false` keeps today's behaviour.
- [x] Implement: read `policy_source` from the payload; add `REQUIRE_POLICY` input plumbing next to `fail-on`; add `require-policy` to `action.yml` (`default: "false"`).
- [x] README: in the `policy` mode table add the `none` row, document `require-policy`, and update line ~200 ("the CLI, the web app and this check all enforce one policy") to say the policy comes from `evalshift.yaml` via the pushed run.
- [x] `uv run pytest` (or the repo's test command) + the pin-consistency test.

---

## Phase 5 — cleanup and follow-ups (not blocking release)

- [ ] `[server]` After 90 days with no `policy_source == "project_policy"` answers in logs: drop `projects.migration_policy_json`, `load_policy`, `evaluate_policy`'s legacy path, and `_effective_slice_policy`. Add a structlog counter now so the decision can be made from data.
- [x] `[cli]` Fold `thresholds` into `migration_policy` or delete it; today it is free-form and gates nothing (`docs/configuration.md:53`).
      **Done 2026-09-19 — deleted outright** (maintainer's call: not folded, no deprecation
      period). Branch `chore/remove-thresholds`. The field, the push sync, `_thresholds_from_config`,
      `_non_empty` and `_warn_threshold_drift` are gone; a config still setting `thresholds:` now
      fails to load with a message naming the removal. Breaking — the repo is 1.0.1, so this
      implies 2.0.0. **Two follow-ups this opened:** (a) **resolved 2026-09-19 — the rule was
      amended, `version:` stays `1`.** The literal marks a config that is still valid but would be
      read with the wrong meaning; a removal that fails the load while naming the key is the
      opposite of that, and bumping would have forced an edit on every config, including the
      majority that never set `thresholds`. `docs/configuration.md`, `DOCS.md`, `llms-full.txt` and
      the CHANGELOG entry now say so. (b) `[server]` `canonical_thresholds` now has no consumer and
      `policy:configure` (D8) guards nothing — **scheduled 2026-09-19 as the `[server]` bullet
      below.**
- [ ] `[server]` Retire the thresholds plumbing the CLI no longer feeds (follow-up (b) above):
      `canonical_thresholds` on the upload response, `_sync_project_thresholds`, and the
      `policy:configure` requirement on `POST /runs`. A CLI older than 2.0.0 still sends
      `thresholds`, so the field keeps being *accepted* — what goes is the sync, the response
      field, and the permission that gated a write nothing performs any more.
- [ ] `[server]` Org-level policy floor (a minimum a pushed policy cannot go below) if governance becomes a customer ask. Design only after D8's acceptance is revisited.

---

## Acceptance (end-to-end, run by hand before release)

1. Fresh project, `evalshift.yaml` with a strict `migration_policy`; `evalshift run` + `push`. Web settings card shows nine budgets "From run …". `policy-check` returns `policy_source: run_policy`, `status == verdict`. Action commit status matches the CLI's `compare --policy-gate` exit code.
2. Same project, delete `migration_policy`, push again. `push` warns; settings card still shows the *previous* run's policy as current (latest run with a snapshot) — verify this is the intended reading of D4 and adjust the card copy if it confuses; Action posts the `::warning::`; run Policy tab says "Not gated".
3. Pre-existing project with a web policy and no yaml policy: `push` prints the YAML hint; settings card shows the legacy banner; `policy-check` for old runs still answers `project_policy`; `PATCH` with `migration_policy` → 422.
4. Old CLI (pre-Phase-2) bundle against new server: finalize succeeds; run has `policy_jsonb NULL`; behaviour identical to (3).
