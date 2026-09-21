# Stale Docs Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove five verified stale or false claims from the shipped docs of `evalshift-action` and `evalshift-cli`, and leave a regression guard behind each one so the same rot cannot recur silently.

**Architecture:** Every fix is a docs edit plus a test that fails on the stale text. Three of the five files are mirrored to the marketing site by a plain `cp` (`npm run sync:llms`), so all source-repo edits land first and a single sync commit re-publishes them last. No product code changes except `scripts/bump_cli_pin.py`, which gains one more self-syncing version site.

**Tech Stack:** Markdown + plain-text docs; `pytest` + `ruff` (both Python repos); `mypy --strict` (CLI only); `vitest` + `tsc` + `eslint` (client). Version-drift guards follow the existing `PIN_SITES` / `ACTION_VERSION_SITES` regex-table pattern in `evalshift-action/scripts/bump_cli_pin.py`.

**Spec:** This plan is its own spec. Every claim below was verified against source on 2026-09-21; the "Evidence" line under each task records what was checked and where.

## Global Constraints

- Three independent git repos under a non-git parent `/home/lukas/repos/evalshift`. **Always pass `-C <repo>`** to git. All three are on `main`, clean, and in sync with `origin` as of 2026-09-21.
- This plan file lives in `evalshift-cli/docs/superpowers/plans/`, which matches a `.gitignore` pattern. Committing it needs **`git add -f`**.
- Conventional Commits with scopes, matching existing history (`fix(docs): ...`, `chore(pin): ...`, `docs: ...`).
- Branch per task, PR per task. Never commit directly to `main`.
- **Gates — `evalshift-action`:** `uv run pytest` · `uv run ruff check .` · `uv run pip-audit`
- **Gates — `evalshift-cli`:** `ruff check .` · `ruff format --check .` · `mypy --strict src/evalshift_cli` · `pytest` (2276 tests, ~19s)
- **Gates — `evalshift-client`:** `npm run lint` · `npm run typecheck` · `npm run build`
- `evalshift-action/llms-full.txt`, `evalshift-cli/llms-full.txt` and `evalshift-sdk/llms-full.txt` are copied verbatim into `evalshift-client/public/` by `npm run sync:llms`. **Never hand-edit the `public/*-llms-full.txt` copies** — fix the source repo, then sync (Task 7).
- **Tasks run in order and each merges before the next starts.** Tasks 2, 3 and 4 all append to the same new file, `evalshift-cli/tests/unit/test_docs_currency.py`, and each branches from a freshly pulled `main`; the expected pass counts quoted in their verification steps assume the earlier task has merged. Tasks 1 and 5 (action) are independent of 2/3/4 (CLI) and may be reordered between repos, but Task 7 is last.
- Do not bump any package version. These are docs-only changes; the action's `pyproject.toml` version stays `0.5.1` and the CLI's stays `1.1.0`.

---

### Task 1: Delete the dead `thresholds:` / `policy:configure` flow from the Action docs

**Why this is first:** it is the only issue where the docs instruct a user to do something that is now impossible. A config carrying `thresholds:` fails to load outright, and the permission the text cites does not exist. The text also ships to AI agents at `evalshift.dev/ci-llms-full.txt`.

**Evidence (verified 2026-09-21):**
- `evalshift-server/app/authz/permissions.py:26` — comment: *"There is no `policy:configure`. It guarded one write — the thresholds sync a run upload used to perform — and went with it (P17 Phase 5)."*
- `evalshift-server/tests/test_phase8_authz.py:164` — asserts `"policy:configure" not in authz.ALL_PERMISSIONS`.
- `evalshift-cli/src/evalshift_cli/config/models.py:738` — a config with `thresholds:` raises *"`thresholds` was removed: it was free-form and gated nothing."*
- The string `Project owner role required` appears **nowhere** in `evalshift-server` (only `Organization owner role required`, in `app/orgs/service.py`, for org-level operations) and nowhere in the action's own `scripts/`. The troubleshooting entry documents an error the product cannot emit, so it is deleted rather than reworded.
- The same README contradicts itself at `README.md:200-204`: *"the action never re-implements a threshold."*

**Files:**
- Modify: `evalshift-action/README.md:110-121`
- Modify: `evalshift-action/DOCS.md:176-188`, `evalshift-action/DOCS.md:870-874`
- Modify: `evalshift-action/llms-full.txt:492-498`, `evalshift-action/llms-full.txt:693-694`
- Create: `evalshift-action/tests/test_docs_currency.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `tests/test_docs_currency.py::RETIRED_TERMS`, a `tuple[str, ...]` of exact substrings that must not appear in the action's three prose files. Task 5 does not touch it; later doc retirements append to it.

- [ ] **Step 1: Create the branch**

```bash
git -C /home/lukas/repos/evalshift/evalshift-action checkout -b fix/drop-dead-thresholds-docs
```

- [ ] **Step 2: Write the failing test**

Create `evalshift-action/tests/test_docs_currency.py`:

```python
"""The docs must not describe flows the product has removed.

`thresholds:` was deleted from `evalshift.yaml` in CLI 1.1.0 -- a config that
still sets it fails to load, by name -- and `policy:configure` was deleted from
the server's permission catalog along with the single write it guarded. This
repo's README, DOCS.md and llms-full.txt described both for five releases after
they were gone, and llms-full.txt is copied verbatim to
https://www.evalshift.dev/ci-llms-full.txt, so the stale instructions were being
served to coding agents as current guidance.

Nothing else noticed, because every existing docs test checks a version literal
rather than a claim. This is the tripwire for claims: a term retired from the
product must not reappear in prose.
"""

from __future__ import annotations

import pytest

from _manifest import REPO_ROOT

#: Exact substrings that named a removed feature. Retiring something else from
#: the product? Append it here in the same commit that removes it.
RETIRED_TERMS: tuple[str, ...] = (
    "policy:configure",
    "thresholds:",
)

PROSE_FILES: tuple[str, ...] = ("README.md", "DOCS.md", "llms-full.txt")


@pytest.mark.parametrize("name", PROSE_FILES)
@pytest.mark.parametrize("term", RETIRED_TERMS)
def test_prose_does_not_describe_a_removed_feature(name: str, term: str) -> None:
    text = (REPO_ROOT / name).read_text(encoding="utf-8")

    assert term not in text, f"{name} still describes the removed {term!r}"
```

- [ ] **Step 3: Run the test to verify it fails**

```bash
cd /home/lukas/repos/evalshift/evalshift-action && uv run pytest tests/test_docs_currency.py -v
```

Expected: 6 tests, all 6 FAIL — each with `<file> still describes the removed '<term>'`.

- [ ] **Step 4: Fix `README.md:110-121`**

Replace this block:

```markdown
Two things a correctly-scoped key deliberately cannot do:

- **Auto-create the hosted project.** `project:create` is an owner permission and
  a service account is never an owner. Create the project once in the web app and
  set `create-project: false`, so a wrong project slug fails as a missing project
  rather than looking like a credential problem.
- **Rewrite the project's gating thresholds.** `evalshift push` sends the
  `thresholds:` block from your `evalshift.yaml` whenever one is present, and
  rewriting a project's gating policy needs `policy:configure` — also owner-only.
  Keep thresholds canonical in the web app and out of the config the CI job runs,
  or the push fails with `Project owner role required`.
```

with:

```markdown
One thing a correctly-scoped key deliberately cannot do:

- **Auto-create the hosted project.** `project:create` is an owner permission and
  a service account is never an owner. Create the project once in the web app and
  set `create-project: false`, so a wrong project slug fails as a missing project
  rather than looking like a credential problem.

The gate itself needs no extra scope. Your `migration_policy` travels inside the
run bundle that `run:create` already uploads, so a member-role key both pushes
the policy and gates on it — see [`fail-on` modes](#fail-on-modes).
```

- [ ] **Step 5: Fix `DOCS.md:176-188`**

Replace this block:

```markdown
Two consequences of a correctly-scoped key, both by design:
```

with:

```markdown
One consequence of a correctly-scoped key, by design:
```

Then delete this bullet entirely:

```markdown
- **It cannot rewrite the project's gating thresholds.** `evalshift push` sends the
  `thresholds:` block from your `evalshift.yaml` whenever one is present, and rewriting a
  project's gating policy needs the owner-only `policy:configure`. Keep thresholds canonical in
  the web app and out of the config the CI job runs, or the push fails with
  `Project owner role required`.
```

and insert, as a new paragraph after the remaining bullet and before the `A denial is self-diagnosing:` paragraph:

```markdown
The gate needs no scope beyond these two. The `migration_policy` block rides inside the run
bundle `run:create` already uploads, so a member-role key both pushes the policy and gates on
it.
```

- [ ] **Step 6: Delete the phantom troubleshooting entry at `DOCS.md:870-874`**

Delete the heading and its body, leaving the surrounding entries untouched:

```markdown
### `Project owner role required`

`evalshift push` tried to rewrite the project's gating thresholds, which needs the owner-only
`policy:configure`. Remove the `thresholds:` block from the config the CI job runs and manage
thresholds in the web app.

```

- [ ] **Step 7: Fix `llms-full.txt:492-498`**

Replace this block:

```text
Out of reach for a scoped key, by design:
- Auto-creating the hosted project (`project:create` is owner-only). Create the project in the
  web app and set `create-project: false`.
- Rewriting gating thresholds (`policy:configure` is owner-only). `evalshift push` sends the
  `thresholds:` block from `evalshift.yaml` whenever one is present, so keep thresholds
  canonical in the web app and out of the CI config, or push fails
  `Project owner role required`.
```

with:

```text
Out of reach for a scoped key, by design:
- Auto-creating the hosted project (`project:create` is owner-only). Create the project in the
  web app and set `create-project: false`.
Gating needs no further scope: `migration_policy` rides in the bundle `run:create` uploads, so a
member-role key both pushes the policy and gates on it.
```

- [ ] **Step 8: Delete the phantom error line at `llms-full.txt:693-694`**

Delete exactly these two lines:

```text
`Project owner role required` → push tried to rewrite gating thresholds (`policy:configure`,
owner-only); drop `thresholds:` from the CI config.
```

- [ ] **Step 9: Run the test to verify it passes**

```bash
cd /home/lukas/repos/evalshift/evalshift-action && uv run pytest tests/test_docs_currency.py -v
```

Expected: 6 passed.

- [ ] **Step 10: Confirm nothing else mentions the retired terms**

```bash
cd /home/lukas/repos/evalshift/evalshift-action && grep -rn "thresholds\|policy:configure" README.md DOCS.md llms-full.txt action.yml
```

Expected: **no output.** (Before this task the same command printed 13 lines.) `README.md:200-204` says "never re-implements a threshold" — singular, no colon — and is correct; it is not matched by `thresholds`.

- [ ] **Step 11: Run the full gate**

```bash
cd /home/lukas/repos/evalshift/evalshift-action && uv run pytest && uv run ruff check .
```

Expected: all tests pass, ruff clean.

- [ ] **Step 12: Commit and open the PR**

```bash
cd /home/lukas/repos/evalshift/evalshift-action
git add README.md DOCS.md llms-full.txt tests/test_docs_currency.py
git commit -m "fix(docs): stop documenting the removed thresholds gate

CLI 1.1.0 deleted \`thresholds:\` from evalshift.yaml and the server deleted
\`policy:configure\` with the one write it guarded, but all three prose files
still told users to keep thresholds in the web app to avoid a
\`Project owner role required\` error the server never emits. The same README
already said the action never re-implements a threshold.

llms-full.txt is mirrored to evalshift.dev/ci-llms-full.txt, so this was being
served to coding agents as current guidance. test_docs_currency.py fails on any
future mention."
git push -u origin fix/drop-dead-thresholds-docs
gh pr create --fill
```

---

### Task 2: Advertise `compare --push`, not the hidden legacy `all --push`

**Evidence (verified 2026-09-21):** `compare` is the registered command (`src/evalshift_cli/cli/main.py:89`). `all` is registered at `main.py:92` with `hidden=True` and prints a rename notice to stderr (`cli/commands/compare.py:322-339`); `LEGACY_COMMAND_NAME = "all"` at `compare.py:319`. The alias is permanent, so nothing is broken — but nine doc sites advertise the hidden name as the thing to type.

**Scope note — five sites must NOT change.** These correctly describe `all` as a legacy alias, or are unrelated:
- `llms-full.txt:75` — "Formerly `all`: hidden alias, still works."
- `llms-full.txt:417` — "`all` -> `compare` in 1.0.0"
- `DOCS.md:353` — a *slice* named `all`. Unrelated to the command.
- `DOCS.md:819` — "Formerly `all`; that name is hidden but still works"
- `DOCS.md:880` — "`evalshift all` became `evalshift compare` in 1.0.0"

The literal `all --push` appears at exactly nine sites and at none of the five above, so a global substring replace is safe. Step 5 proves it.

**Files:**
- Modify: `evalshift-cli/README.md:216`, `:306`
- Modify: `evalshift-cli/DOCS.md:720`
- Modify: `evalshift-cli/llms-full.txt:938`, `:1091`
- Modify: `evalshift-cli/docs/faq.md:15`
- Modify: `evalshift-cli/docs/hosted.md:253`
- Modify: `evalshift-cli/docs/configuration.md:70`
- Modify: `evalshift-cli/docs/index.md:89`
- Create: `evalshift-cli/tests/unit/test_docs_currency.py`

**Interfaces:**
- Consumes: nothing from Task 1 (different repo).
- Produces: `tests/unit/test_docs_currency.py::PROSE_FILES`, a `tuple[str, ...]` of the CLI's seven prose files relative to the repo root. Tasks 3 and 4 add their own test functions to this same module and reuse `PROSE_FILES`.

- [ ] **Step 1: Create the branch**

```bash
git -C /home/lukas/repos/evalshift/evalshift-cli checkout -b fix/advertise-compare-not-all
```

- [ ] **Step 2: Write the failing test**

Create `evalshift-cli/tests/unit/test_docs_currency.py`:

```python
"""The docs must advertise the current command name, not a hidden alias.

`evalshift all` became `evalshift compare` in 1.0.0. The old name stays
registered forever -- scaffolded EVALSHIFT.md files in user repos reference it,
and removing it would itself be breaking -- but it is `hidden=True` and prints a
rename notice. Nine doc sites still told readers to type it, so the docs taught
a name that `evalshift --help` does not list.

Prose that *describes* the alias ("formerly `all`", "`all` -> `compare` in
1.0.0") is correct and deliberately not matched here: the assertion is on the
exact string `all --push`, which only ever appeared as advertised usage.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Every file that documents the CLI in prose, relative to the repo root.
PROSE_FILES: tuple[str, ...] = (
    "README.md",
    "DOCS.md",
    "llms-full.txt",
    "docs/faq.md",
    "docs/hosted.md",
    "docs/configuration.md",
    "docs/index.md",
)


@pytest.mark.parametrize("name", PROSE_FILES)
def test_prose_advertises_compare_not_the_hidden_alias(name: str) -> None:
    text = (REPO_ROOT / name).read_text(encoding="utf-8")

    assert "all --push" not in text, (
        f"{name} advertises the hidden `all` alias; write `compare --push`"
    )
```

- [ ] **Step 3: Run the test to verify it fails**

```bash
pytest tests/unit/test_docs_currency.py -v --no-cov
```

Expected: 7 tests, **7 failed** — `README.md`, `DOCS.md`, `llms-full.txt`, `docs/faq.md`, `docs/hosted.md`, `docs/configuration.md`, `docs/index.md` each report the alias.

- [ ] **Step 4: Replace all nine occurrences**

```bash
cd /home/lukas/repos/evalshift/evalshift-cli
sed -i 's/all --push/compare --push/g' \
  README.md DOCS.md llms-full.txt \
  docs/faq.md docs/hosted.md docs/configuration.md docs/index.md
```

- [ ] **Step 5: Verify the replacement hit nine sites and spared the five alias descriptions**

```bash
cd /home/lukas/repos/evalshift/evalshift-cli
echo "--- should print 9 ---"
grep -rc "compare --push" README.md DOCS.md llms-full.txt docs/*.md | grep -v ':0' | awk -F: '{s+=$2} END {print s}'
echo "--- should print nothing ---"
grep -rn "all --push" README.md DOCS.md llms-full.txt docs/
echo "--- these 5 must still be present ---"
grep -n "Formerly \`all\`" llms-full.txt DOCS.md
grep -n "\`all\` -> \`compare\`\|evalshift all\` became" llms-full.txt DOCS.md
grep -n "\`all\` and any slice" DOCS.md
```

Expected: `9`; no `all --push` hits; and five surviving lines — `llms-full.txt:75`, `DOCS.md:819`, `llms-full.txt:417`, `DOCS.md:880`, `DOCS.md:353`.

- [ ] **Step 6: Run the test to verify it passes**

```bash
cd /home/lukas/repos/evalshift/evalshift-cli && pytest tests/unit/test_docs_currency.py -v --no-cov
```

Expected: 7 passed.

- [ ] **Step 7: Run the full gate**

```bash
cd /home/lukas/repos/evalshift/evalshift-cli
ruff check . && ruff format --check . && mypy --strict src/evalshift_cli && pytest
```

Expected: ruff clean, mypy clean, 2283 passed.

- [ ] **Step 8: Commit and open the PR**

```bash
cd /home/lukas/repos/evalshift/evalshift-cli
git add README.md DOCS.md llms-full.txt docs/ tests/unit/test_docs_currency.py
git commit -m "docs: advertise \`compare --push\`, not the hidden \`all\` alias

\`all\` became \`compare\` in 1.0.0. The alias stays registered forever, but it
is hidden from --help and prints a rename notice, so nine doc sites were
teaching a name the CLI does not advertise. Prose that describes the alias as
an alias is unchanged.

test_docs_currency.py fails on any future \`all --push\` in prose."
git push -u origin fix/advertise-compare-not-all
gh pr create --fill
```

---

### Task 3: Stop capping the provider list at three

**Evidence (verified 2026-09-21):** the CLI dispatches every model call through LiteLLM (`src/evalshift_cli/models/client.py:35`, `litellm.acompletion`). `src/evalshift_cli/models/registry.py:10` states outright: *"the authority on whether a model is callable is **LiteLLM**, not us"*, and `resolve_model` falls back to prefix inference so an unregistered id still dispatches. `Provider = Literal["anthropic", "openai", "google", "other"]` (`registry.py:36`) is a *curated registry* for `doctor` output and report rendering, not a capability boundary. The repo's own `docs/faq.md:50` already answers "which models?" with **"Anything LiteLLM supports."**

Four doc sites present the three names as the boundary, contradicting `faq.md:50`.

**Client scope — decided 2026-09-21.** `evalshift-client` repeats the triple at three sites. The maintainer chose **the two docs mirrors only**: `src/pages/docs/pages/Faq.tsx:21` and `src/pages/docs/pages/WhatGetsUploaded.tsx:94`, which hand-mirror `docs/faq.md` and `docs/hosted.md` and would otherwise contradict the pages they mirror. `src/pages/landing/sections/Hero.tsx:113` ("Works with Anthropic, OpenAI and Google models.", asserted by `src/pages/landing/Landing.test.tsx:253`) stays as marketing copy — **do not touch it**. Those two client edits are executed in **Task 7**, which already owns this plan's client-repo branch; this task stays CLI-only.

**Files:**
- Modify: `evalshift-cli/README.md:301-303`
- Modify: `evalshift-cli/docs/index.md:86-88`
- Modify: `evalshift-cli/docs/faq.md:5-8`
- Modify: `evalshift-cli/docs/hosted.md:246-247`
- Modify: `evalshift-cli/tests/unit/test_docs_currency.py` (add one test)

**Interfaces:**
- Consumes: `tests/unit/test_docs_currency.py::PROSE_FILES` and `REPO_ROOT` from Task 2.
- Produces: nothing later tasks depend on.

- [ ] **Step 1: Create the branch**

```bash
git -C /home/lukas/repos/evalshift/evalshift-cli checkout main && git -C /home/lukas/repos/evalshift/evalshift-cli pull
git -C /home/lukas/repos/evalshift/evalshift-cli checkout -b docs/provider-breadth
```

- [ ] **Step 2: Write the failing test**

Append to `evalshift-cli/tests/unit/test_docs_currency.py`:

```python
#: Files that describe *which providers work*, as opposed to naming three as
#: examples. Each must name LiteLLM, because LiteLLM is the actual boundary:
#: `models/registry.py` says so in its module docstring, and `docs/faq.md`
#: already answers "which models?" with "Anything LiteLLM supports."
PROVIDER_SCOPE_FILES: tuple[str, ...] = (
    "README.md",
    "docs/index.md",
    "docs/faq.md",
    "docs/hosted.md",
)


@pytest.mark.parametrize("name", PROVIDER_SCOPE_FILES)
def test_provider_scope_is_not_capped_at_three(name: str) -> None:
    """The curated registry has three entries; the CLI calls far more than three.

    `Provider` is a Literal of three names plus "other" because those three have
    pricing tables and env-var mappings worth curating. Every call still goes
    through `litellm.acompletion`, and `resolve_model` never raises -- an
    unregistered id is dispatched with a prefix-inferred provider. Prose that
    lists the three without naming LiteLLM reads as a compatibility list and
    undersells the tool.
    """
    text = (REPO_ROOT / name).read_text(encoding="utf-8")

    assert "LiteLLM" in text, f"{name} scopes providers without naming LiteLLM"
```

- [ ] **Step 3: Run the test to verify it fails**

```bash
cd /home/lukas/repos/evalshift/evalshift-cli && pytest tests/unit/test_docs_currency.py -k provider_scope -v --no-cov
```

Expected: 4 tests, **3 failed** (`README.md`, `docs/index.md`, `docs/hosted.md`) and **1 passed** (`docs/faq.md` — it already names LiteLLM at line 50, which is exactly the contradiction being fixed).

- [ ] **Step 4: Fix `README.md:301-303`**

Replace:

```markdown
Your prompts and suite stay local for `doctor`, `run`, `evaluate`, `analyze`,
and `report`. The only outbound calls in local mode are to the LLM providers
you configure (Anthropic, OpenAI, Google) using your own API keys.
```

with:

```markdown
Your prompts and suite stay local for `doctor`, `run`, `evaluate`, `analyze`,
and `report`. The only outbound calls in local mode are to the LLM providers you
configure — any provider LiteLLM supports, called with your own API keys.
Anthropic, OpenAI and Google ids additionally get a curated pricing and
capability entry; everything else is passed through with the provider inferred
from the id.
```

- [ ] **Step 5: Fix `docs/index.md:86-88`**

Replace:

```markdown
The only outbound calls are to the LLM providers you configure (Anthropic,
OpenAI, Google) using your own API keys. `bundle` packages artifacts locally;
hosted upload happens only when you run `push` or `compare --push`.
```

with:

```markdown
The only outbound calls are to the LLM providers you configure — any provider
LiteLLM supports — using your own API keys. `bundle` packages artifacts locally;
hosted upload happens only when you run `push` or `compare --push`.
```

(Note: `compare --push` here assumes Task 2 has merged. If Task 3 runs first, that line still reads `all --push` — leave it alone and let Task 2 fix it.)

- [ ] **Step 6: Fix `docs/faq.md:5-8`**

Replace:

```markdown
**Not during local runs.** `doctor`, `run`, `evaluate`, `analyze`, and
`report` operate locally. Every provider API call goes directly from your
machine to the LLM provider you configured (Anthropic, OpenAI, Google)
using your own API keys.
```

with:

```markdown
**Not during local runs.** `doctor`, `run`, `evaluate`, `analyze`, and
`report` operate locally. Every provider API call goes directly from your
machine to the LLM provider you configured — any provider LiteLLM supports —
using your own API keys.
```

- [ ] **Step 7: Fix `docs/hosted.md:246-247`**

Replace:

```markdown
1. **Your model providers** (Anthropic, OpenAI, Google — whichever you
   configure), using your own API keys: `run` sends the rendered prompts and
```

with:

```markdown
1. **Your model providers** (whichever you configure — any provider LiteLLM
   supports), using your own API keys: `run` sends the rendered prompts and
```

- [ ] **Step 8: Run the test to verify it passes**

```bash
cd /home/lukas/repos/evalshift/evalshift-cli && pytest tests/unit/test_docs_currency.py -v --no-cov
```

Expected: 11 passed (7 from Task 2 + 4 here).

- [ ] **Step 9: Run the full gate**

```bash
cd /home/lukas/repos/evalshift/evalshift-cli
ruff check . && ruff format --check . && mypy --strict src/evalshift_cli && pytest
```

Expected: all clean.

- [ ] **Step 10: Commit and open the PR**

```bash
cd /home/lukas/repos/evalshift/evalshift-cli
git add README.md docs/ tests/unit/test_docs_currency.py
git commit -m "docs: name LiteLLM as the provider boundary, not three brands

models/registry.py says LiteLLM is the authority on whether a model is callable,
resolve_model never raises, and docs/faq.md already answers 'which models?' with
'Anything LiteLLM supports.' Four other sites listed Anthropic/OpenAI/Google as
though that were the compatibility list -- which contradicts faq.md and
undersells the tool. The three keep a curated pricing and capability entry; that
is what the Literal is for.

Known drift: evalshift-client's Faq.tsx and WhatGetsUploaded.tsx hand-mirror
these two pages and still say the old thing."
git push -u origin docs/provider-breadth
gh pr create --fill
```

---

### Task 4: Quote the placeholder `init` actually writes

**Evidence (verified 2026-09-21):** `src/evalshift_cli/cli/commands/init.py:116` contains `content: "{{input}}"`, but `_MINIMAL_YAML_BODY` is a `str.format` template rendered at `init.py:174` — the doubled brace is an escape, so the `evalshift.yaml` on disk contains `{input}`. The existing test `tests/unit/test_init.py::TestInitHappy::test_written_config_parses_via_load_config` already asserts `prompt.content == "{input}"`.

Three doc sites quote the escaped source form and so tell readers — and agents — to write a placeholder the templating engine will never expand. The claim as reported named only `llms-full.txt`; two more were found.

**Files:**
- Modify: `evalshift-cli/llms-full.txt:450`
- Modify: `evalshift-cli/DOCS.md:440`
- Modify: `evalshift-cli/docs/configuration.md:286`
- Modify: `evalshift-cli/tests/unit/test_docs_currency.py` (add one test)

**Interfaces:**
- Consumes: `PROSE_FILES` and `REPO_ROOT` from Task 2.
- Produces: nothing later tasks depend on.

- [ ] **Step 1: Create the branch**

```bash
git -C /home/lukas/repos/evalshift/evalshift-cli checkout main && git -C /home/lukas/repos/evalshift/evalshift-cli pull
git -C /home/lukas/repos/evalshift/evalshift-cli checkout -b fix/docs-quote-rendered-placeholder
```

- [ ] **Step 2: Write the failing test**

Append to `evalshift-cli/tests/unit/test_docs_currency.py`:

```python
@pytest.mark.parametrize("name", PROSE_FILES)
def test_prose_quotes_the_rendered_placeholder(name: str) -> None:
    """Docs must quote the config `init` writes, not the format template.

    `_MINIMAL_YAML_BODY` in `cli/commands/init.py` is passed through
    `str.format`, so its literal `{{input}}` is a brace escape that renders as
    `{input}` on disk -- which is what `test_init.py` asserts the loaded config
    contains. Three doc sites copied the escaped source form verbatim, which
    reads as instructions to write a placeholder `templating.py` will never
    expand.
    """
    text = (REPO_ROOT / name).read_text(encoding="utf-8")

    assert "{{input}}" not in text, (
        f"{name} quotes the escaped `{{{{input}}}}`; `init` writes `{{input}}`"
    )
```

- [ ] **Step 3: Run the test to verify it fails**

```bash
cd /home/lukas/repos/evalshift/evalshift-cli && pytest tests/unit/test_docs_currency.py -k rendered_placeholder -v --no-cov
```

Expected: 7 tests, **3 failed** — `DOCS.md`, `llms-full.txt`, `docs/configuration.md`.

- [ ] **Step 4: Fix all three sites**

```bash
cd /home/lukas/repos/evalshift/evalshift-cli
sed -i 's/{{input}}/{input}/g' llms-full.txt DOCS.md docs/configuration.md
```

- [ ] **Step 5: Verify each site now reads correctly**

```bash
cd /home/lukas/repos/evalshift/evalshift-cli
grep -n "{input}" llms-full.txt DOCS.md docs/configuration.md
```

Expected, exactly:
- `llms-full.txt:123` — `(id: replay, detection: manual, content: "{input}", variables: [input])` *(already correct before this task)*
- `llms-full.txt:450` — ``# `replay` prompt is a passthrough: content "{input}" echoes the promoted capture.``
- `llms-full.txt:1110` — `content: "{input}"` *(already correct before this task)*
- `DOCS.md:440` — ``(`content: "{input}"`)``
- `docs/configuration.md:286` — ``(`content: "{input}"`) is a passthrough``

- [ ] **Step 6: Run the test to verify it passes**

```bash
cd /home/lukas/repos/evalshift/evalshift-cli && pytest tests/unit/test_docs_currency.py -v --no-cov
```

Expected: 18 passed (7 + 4 + 7).

- [ ] **Step 7: Run the full gate**

```bash
cd /home/lukas/repos/evalshift/evalshift-cli
ruff check . && ruff format --check . && mypy --strict src/evalshift_cli && pytest
```

Expected: all clean. `test_init.py::test_written_config_parses_via_load_config` must still pass — it asserts the rendered form and is the reason this fix is in this direction.

- [ ] **Step 8: Commit and open the PR**

```bash
cd /home/lukas/repos/evalshift/evalshift-cli
git add llms-full.txt DOCS.md docs/configuration.md tests/unit/test_docs_currency.py
git commit -m "fix(docs): quote the placeholder init writes, not the format escape

init.py's _MINIMAL_YAML_BODY goes through str.format, so the literal
\`{{input}}\` in the source is a brace escape and the scaffolded evalshift.yaml
contains \`{input}\` -- as test_init.py already asserts. Three doc sites copied
the escaped form, telling readers to write a placeholder templating.py never
expands, while two other sites in the same file showed the correct one."
git push -u origin fix/docs-quote-rendered-placeholder
gh pr create --fill
```

---

### Task 5: Make the example `@vX.Y.Z` tag self-syncing

**Evidence (verified 2026-09-21):** the "pin to an exact tag" example says `@v0.3.0` at `README.md:333`, `DOCS.md:944` and `llms-full.txt:721`. The action's current release is **v0.5.1** (`pyproject.toml:3`; `v0.5.1` exists on `origin`). Nothing breaks — it is illustrative syntax — but it reads as a recommendation two minors stale, and it went stale for the same reason the version headers did: nothing bumped it and nothing checked it.

This repo already solved that class of problem twice. `scripts/bump_cli_pin.py` holds `PIN_SITES` (the CLI pin, sourced from `action.yml`) and `ACTION_VERSION_SITES` (the action's own version, sourced from `pyproject.toml`), and `tests/test_pin_consistency.py` reads the same tables to fail on any stale literal. This task adds a third table rather than hand-editing three files, so the example cannot drift again.

**Files:**
- Modify: `evalshift-action/scripts/bump_cli_pin.py` (add `EXAMPLE_TAG_SITES`; extend `sync_action_version`)
- Modify: `evalshift-action/tests/test_pin_consistency.py` (add two tests)
- Modify: `evalshift-action/README.md:333`, `DOCS.md:944`, `llms-full.txt:721` (rewritten by the script, not by hand)

**Interfaces:**
- Consumes: `bump_cli_pin.VERSION`, `bump_cli_pin.replace_pins`, `bump_cli_pin.find_pins`, `bump_cli_pin.current_action_version`, `_manifest.REPO_ROOT` — all already exist.
- Produces: `bump_cli_pin.EXAMPLE_TAG_SITES: Mapping[str, tuple[str, ...]]`, keyed by the same three filenames. `sync_action_version(root) -> list[Path]` keeps its signature and return type; it now also rewrites these sites.

- [ ] **Step 1: Create the branch**

```bash
git -C /home/lukas/repos/evalshift/evalshift-action checkout main && git -C /home/lukas/repos/evalshift/evalshift-action pull
git -C /home/lukas/repos/evalshift/evalshift-action checkout -b chore/self-syncing-example-tag
```

- [ ] **Step 2: Write the failing tests**

Append to `evalshift-action/tests/test_pin_consistency.py`, and add `EXAMPLE_TAG_SITES` to the existing `from bump_cli_pin import ...` line:

```python
def test_example_tag_sites_cover_the_documented_files() -> None:
    assert set(EXAMPLE_TAG_SITES) == {"README.md", "DOCS.md", "llms-full.txt"}


@pytest.mark.parametrize("name", sorted(EXAMPLE_TAG_SITES))
def test_every_example_tag_matches_pyproject(name: str) -> None:
    """The "pin to an exact tag" example must name a tag that exists and is current.

    It sat at v0.3.0 across five releases -- v0.3.x through v0.5.1 -- because it
    was hand-written prose that no bump touched. Advice to pin is advice to pin
    to *something*; two minors behind, the example reads as the recommendation.
    """
    released = current_action_version(REPO_ROOT)
    text = (REPO_ROOT / name).read_text(encoding="utf-8")

    stale = [
        found for found in find_pins(text, EXAMPLE_TAG_SITES[name], label=name) if found != released
    ]

    assert stale == [], (
        f"{name}'s example tag is @v{sorted(set(stale))}; pyproject.toml says {released}"
    )
```

- [ ] **Step 3: Run the tests to verify they fail**

```bash
cd /home/lukas/repos/evalshift/evalshift-action && uv run pytest tests/test_pin_consistency.py -v
```

Expected: collection ERROR — `ImportError: cannot import name 'EXAMPLE_TAG_SITES' from 'bump_cli_pin'`.

- [ ] **Step 4: Add the table to `scripts/bump_cli_pin.py`**

Insert immediately after the `ACTION_VERSION_SITES` block:

```python
# The "pin to an exact tag" example in the versioning prose. Same source of truth as
# ACTION_VERSION_SITES -- pyproject.toml -- in a different shape: a `@vX.Y.Z` git tag
# rather than a bare version. It sat at `@v0.3.0` from v0.3.x through v0.5.1 because
# no bump touched it and no test read it. Advice to pin has to name a tag that exists.
EXAMPLE_TAG_SITES: Mapping[str, tuple[str, ...]] = {
    "README.md": (rf"exact tag such as `@v{VERSION}`",),
    "DOCS.md": (rf"exact tag such as `@v{VERSION}`",),
    "llms-full.txt": (rf"^`@v0` tracks the latest v0\.x\. `@v{VERSION}` pins exactly\.",),
}
```

- [ ] **Step 5: Extend `sync_action_version` to rewrite them**

Replace the body of `sync_action_version` with:

```python
def sync_action_version(root: Path = REPO_ROOT) -> list[Path]:
    """Rewrite every advertised action version to match ``pyproject.toml``.

    Covers both shapes the version is written in: the prose version headers
    (``ACTION_VERSION_SITES``) and the ``@vX.Y.Z`` example tag
    (``EXAMPLE_TAG_SITES``). Call this AFTER ``pyproject.toml`` is written, so
    both follow the bump. Returns the files actually changed, in order, deduped.
    """
    released = current_action_version(root)
    changed: list[Path] = []
    for table in (ACTION_VERSION_SITES, EXAMPLE_TAG_SITES):
        for name, patterns in table.items():
            path = root / name
            before = path.read_text(encoding="utf-8")
            after = replace_pins(before, patterns, released, label=name)
            if after == before:
                continue
            path.write_text(after, encoding="utf-8")
            # Both tables name the same three files, so a path can already be here.
            if path not in changed:
                changed.append(path)
    return changed
```

Also update the module docstring's last paragraph, replacing `` ``ACTION_VERSION_SITES`` enumerates those headers and `` … with:

```
``ACTION_VERSION_SITES`` enumerates those headers, ``EXAMPLE_TAG_SITES`` the ``@vX.Y.Z``
example tag that drifted the same way, and ``sync_action_version`` rewrites both from
``pyproject.toml`` on every bump, so none of them can disagree.
```

- [ ] **Step 6: Run the tests to verify they now fail on the stale literal (not on the import)**

```bash
cd /home/lukas/repos/evalshift/evalshift-action && uv run pytest tests/test_pin_consistency.py -v
```

Expected: `test_example_tag_sites_cover_the_documented_files` PASSES; the three `test_every_example_tag_matches_pyproject` cases FAIL with ``example tag is @v['0.3.0']; pyproject.toml says 0.5.1``.

- [ ] **Step 7: Rewrite the three docs by running the syncer**

```bash
cd /home/lukas/repos/evalshift/evalshift-action
uv run python -c "
import sys; sys.path.insert(0, 'scripts')
from bump_cli_pin import REPO_ROOT, sync_action_version
for p in sync_action_version(REPO_ROOT): print(p.relative_to(REPO_ROOT))
"
```

Expected output: `README.md`, `DOCS.md`, `llms-full.txt`.

- [ ] **Step 8: Verify the edit hit only the example tag**

```bash
cd /home/lukas/repos/evalshift/evalshift-action && git diff --stat && git diff -U0 | grep '^[-+]' | grep -v '^[-+][-+]'
```

Expected: exactly three changed lines, each `v0.3.0` → `v0.5.1`. Every `uses: babaliauskas/evalshift-action@v0` line must be **unchanged** — the floating major tag is correct and the regexes are scoped to the prose sentence, not to `@v`.

- [ ] **Step 9: Run the full gate**

```bash
cd /home/lukas/repos/evalshift/evalshift-action && uv run pytest && uv run ruff check .
```

Expected: all pass, including the pre-existing `test_bump_cli_pin.py` suite.

- [ ] **Step 10: Commit and open the PR**

```bash
cd /home/lukas/repos/evalshift/evalshift-action
git add scripts/bump_cli_pin.py tests/test_pin_consistency.py README.md DOCS.md llms-full.txt
git commit -m "chore(docs): sync the example \`@vX.Y.Z\` tag from pyproject

The 'pin to an exact tag' example said @v0.3.0 from v0.3.x through v0.5.1 --
hand-written prose no bump touched and no test read, which is exactly how the
version headers drifted before PR #13. Rather than edit it again, add
EXAMPLE_TAG_SITES alongside ACTION_VERSION_SITES so sync_action_version rewrites
it on every bump and test_pin_consistency fails on any stale literal.

The floating \`@v0\` in the usage examples is unchanged -- it is correct."
git push -u origin chore/self-syncing-example-tag
gh pr create --fill
```

---

### Task 6: Make the Status section's claims checkable — **needs a decision first**

Two claims sit in `evalshift-cli/README.md:67-70`. They are different kinds of problem and only one has a mechanical fix.

**6a — the coverage number is wrong and unguarded.** README says *"the test suite covers 92% of the source."* Measured on `main` at 2026-09-21: **94%** (`TOTAL 9219 434 2642 202 94%`, 2276 passed in 18.5s). There is no `fail_under` in `[tool.coverage.report]` (`pyproject.toml:113-121`), so nothing checks the number and it will drift again. The fix is to stop quoting a literal that rots: set a floor and describe the floor.

**6b — decided 2026-09-21: keep "Stable and in production use." exactly as written.** It was raised as a judgment call, not an error: `llms-full.txt` is mirrored to `evalshift.dev/cli-llms-full.txt`, so AI engines will repeat the phrase as a customer claim rather than as a self-description. The maintainer is comfortable with that. **Preserve the sentence byte-for-byte** — this task ships 6a alone.

**Files:**
- Modify: `evalshift-cli/README.md:67-70`
- Modify: `evalshift-cli/pyproject.toml` (`[tool.coverage.report]`, add `fail_under`)

**Interfaces:** none — nothing depends on this task, and nothing it depends on.

- [ ] **Step 1: No decision pending.** 6b was answered on 2026-09-21: keep the sentence as-is. Proceed straight to Step 2 and change only the coverage claim.

- [ ] **Step 2: Create the branch**

```bash
git -C /home/lukas/repos/evalshift/evalshift-cli checkout main && git -C /home/lukas/repos/evalshift/evalshift-cli pull
git -C /home/lukas/repos/evalshift/evalshift-cli checkout -b docs/checkable-status-claims
```

- [ ] **Step 3: Add the coverage floor to `pyproject.toml`**

In `[tool.coverage.report]`, add above `exclude_lines`:

```toml
# The README quotes this floor rather than a measured percentage, so the claim
# is enforced instead of hand-maintained. Measured 94% on 2026-09-21; the floor
# is set below that deliberately, so an honest refactor does not fail CI while
# a real drop still does. Raise it when the margin gets comfortable.
fail_under = 90
```

- [ ] **Step 4: Verify the floor holds and actually bites**

```bash
cd /home/lukas/repos/evalshift/evalshift-cli
pytest 2>&1 | tail -3
```

Expected: `TOTAL ... 94%` and `2276 passed` with **no** `FAIL Required test coverage of 90% not reached`.

Then prove the gate is live:

```bash
cd /home/lukas/repos/evalshift/evalshift-cli
pytest tests/unit/test_init.py --cov-fail-under=99 2>&1 | tail -2
```

Expected: `FAIL Required test coverage of 99% not reached`. This confirms `fail_under` is wired, not inert.

- [ ] **Step 5: Rewrite `README.md:67-70`**

Replace:

```markdown
**Stable and in production use.** Every command in the pipeline is shipped and
the test suite covers 92% of the source. The CLI is published on PyPI as
`evalshift`, the capture SDK as `evalshift-sdk`, and the hosted service runs at
`api.evalshift.dev`.
```

with exactly this — the lead sentence is unchanged per the 6b decision, and only
the coverage clause moves from a literal to the enforced floor:

```markdown
**Stable and in production use.** Every command in the pipeline is shipped and
CI enforces a 90% coverage floor on the source. The CLI is published on PyPI as
`evalshift`, the capture SDK as `evalshift-sdk`, and the hosted service runs at
`api.evalshift.dev`.
```

- [ ] **Step 6: Run the full gate**

```bash
cd /home/lukas/repos/evalshift/evalshift-cli
ruff check . && ruff format --check . && mypy --strict src/evalshift_cli && pytest
```

Expected: all clean.

- [ ] **Step 7: Commit and open the PR**

```bash
cd /home/lukas/repos/evalshift/evalshift-cli
git add README.md pyproject.toml
git commit -m "docs: enforce the coverage claim instead of hand-writing it

README said 92%; the suite measures 94%, and nothing checked either number
because [tool.coverage.report] had no fail_under. Quote an enforced floor
rather than a literal that rots between releases."
git push -u origin docs/checkable-status-claims
gh pr create --fill
```

---

### Task 7: Re-publish the mirrored `llms-full.txt` files to the site

**Run this only after Tasks 1–6 have merged to `main` in their source repos.** `npm run sync:llms` is a plain three-way `cp` (`evalshift-client/package.json:17`) with no test guarding drift, which is how the Task 1 text reached `evalshift.dev/ci-llms-full.txt` in the first place: the 2026-09-20 sync faithfully copied a stale source.

This task also carries the two client-side docs mirrors the maintainer approved for Task 3 (decision of 2026-09-21): they live in this repo, and this is the plan's only client-repo branch.

**Files:**
- Modify: `evalshift-client/public/ci-llms-full.txt` (from `evalshift-action`, Tasks 1 + 5)
- Modify: `evalshift-client/public/cli-llms-full.txt` (from `evalshift-cli`, Tasks 2 + 3 + 4)
- Modify: `evalshift-client/src/pages/docs/pages/Faq.tsx:21`
- Modify: `evalshift-client/src/pages/docs/pages/WhatGetsUploaded.tsx:94`
- `evalshift-client/public/sdk-llms-full.txt` — expected unchanged; no task touched the SDK.
- **Do not touch** `src/pages/landing/sections/Hero.tsx` or `src/pages/landing/Landing.test.tsx` — the hero's three-brand line is deliberate marketing copy.

**Interfaces:**
- Consumes: the merged `llms-full.txt` of `evalshift-action` and `evalshift-cli`.
- Produces: nothing.

- [ ] **Step 1: Confirm every source repo is on a clean, merged `main`**

```bash
for r in evalshift-action evalshift-cli evalshift-sdk; do
  echo "== $r"
  git -C /home/lukas/repos/evalshift/$r checkout main -q && git -C /home/lukas/repos/evalshift/$r pull -q
  git -C /home/lukas/repos/evalshift/$r status --short --branch
done
```

Expected: each reports `## main...origin/main` with no trailing `[ahead/behind]` and no modified files.

- [ ] **Step 2: Create the branch and sync**

```bash
cd /home/lukas/repos/evalshift/evalshift-client
git checkout main && git pull
git checkout -b chore/sync-llms-after-docs-cleanup
npm run sync:llms
```

- [ ] **Step 3: Verify the copies now match their sources exactly**

```bash
cd /home/lukas/repos/evalshift/evalshift-client
diff ../evalshift-action/llms-full.txt public/ci-llms-full.txt && echo "ci OK"
diff ../evalshift-cli/llms-full.txt    public/cli-llms-full.txt && echo "cli OK"
diff ../evalshift-sdk/llms-full.txt    public/sdk-llms-full.txt && echo "sdk OK"
```

Expected: three `OK` lines, no diff output.

- [ ] **Step 4: Verify the stale claims are gone from what the site serves**

```bash
cd /home/lukas/repos/evalshift/evalshift-client
echo "--- all four greps must print nothing ---"
grep -n "policy:configure\|thresholds:" public/ci-llms-full.txt
grep -n "@v0\.3\.0"                      public/ci-llms-full.txt
grep -n "all --push"                     public/cli-llms-full.txt
grep -n "{{input}}"                      public/cli-llms-full.txt
```

Expected: no output from any of them.

- [ ] **Step 5: Fix the two hand-written docs mirrors**

These mirror `docs/faq.md` and `docs/hosted.md`, which Task 3 rewrote. Read each
file first — the surrounding JSX differs from the markdown and the replacement has
to fit the existing element structure, so match the file's own wrapping and
`<strong>`/`<code>` usage rather than pasting markdown prose.

In `src/pages/docs/pages/Faq.tsx:21`, the phrase `providers you configured
(Anthropic, OpenAI, Google) using your` becomes the equivalent of *"providers you
configured — any provider LiteLLM supports — using your"*, matching how Task 3
reworded `docs/faq.md:5-8`.

In `src/pages/docs/pages/WhatGetsUploaded.tsx:94`, `<strong>Your model providers</strong>
(Anthropic, OpenAI, Google — whichever you` becomes the equivalent of *"(whichever
you configure — any provider LiteLLM supports)"*, matching Task 3's `docs/hosted.md:246-247`.

- [ ] **Step 6: Confirm the hero is untouched and the mirrors changed**

```bash
cd /home/lukas/repos/evalshift/evalshift-client
echo "--- hero must still say the old thing (deliberate) ---"
grep -n "Works with Anthropic, OpenAI and Google models" src/pages/landing/sections/Hero.tsx
echo "--- both mirrors must now name LiteLLM ---"
grep -c "LiteLLM" src/pages/docs/pages/Faq.tsx src/pages/docs/pages/WhatGetsUploaded.tsx
echo "--- landing test must be unmodified ---"
git diff --name-only | grep -c "Landing.test.tsx" || echo "0 (correct)"
```

Expected: the hero line present; `1` for each mirror; `0 (correct)` for the landing test.

- [ ] **Step 7: Run the full gate**

```bash
cd /home/lukas/repos/evalshift/evalshift-client
npm run lint && npm run typecheck && npm run build && npx vitest run
```

Expected: all pass, including `Landing.test.tsx` — its assertion on the hero string is
untouched and must stay green. The `public/*.txt` files are served statically and are not
parsed by the build; that part is a regression check on the site, not on the copies.

- [ ] **Step 8: Commit**

```bash
cd /home/lukas/repos/evalshift/evalshift-client
git add public/ci-llms-full.txt public/cli-llms-full.txt src/pages/docs/pages/
git commit -m "chore: sync llms-full after the docs cleanup

Picks up the action's removed thresholds/policy:configure guidance and current
example tag, and the CLI's compare --push, LiteLLM provider scope and {input}
placeholder. sync:llms is a plain cp with no drift guard, so the 2026-09-20 sync
faithfully republished stale sources; the guards now live in the source repos."
git push -u origin chore/sync-llms-after-docs-cleanup
gh pr create --fill
```

- [ ] **Step 9: Confirm the deploy served the new text** *(after the user pushes and merges)*

After the PR merges and the site deploys:

```bash
curl -s https://www.evalshift.dev/ci-llms-full.txt | grep -c "policy:configure"
curl -s https://www.evalshift.dev/cli-llms-full.txt | grep -c "all --push"
```

Expected: `0` from both.

---

## Commit the plan itself

```bash
cd /home/lukas/repos/evalshift/evalshift-cli
git add -f docs/superpowers/plans/2026-09-21-stale-docs-cleanup.md
git commit -m "docs(plan): stale docs cleanup across action, cli and site"
```

The `-f` is required: `docs/superpowers/plans/` matches a `.gitignore` pattern, but the plan files are tracked.

## Out of scope, recorded deliberately

- **`evalshift-client`'s landing hero.** `src/pages/landing/sections/Hero.tsx:113` ("Works with Anthropic, OpenAI and Google models.", asserted by `Landing.test.tsx:253`) keeps the three-brand line: the maintainer ruled on 2026-09-21 that naming three recognisable brands is a legitimate marketing choice, distinct from a docs page stating a capability boundary. The two docs mirrors were brought in scope by the same decision and are executed in Task 7.
- **A cross-repo drift guard for `sync:llms`.** The honest guard — a client-side test diffing `public/*-llms-full.txt` against `../evalshift-*/llms-full.txt` — cannot run in the client's CI, where the sibling repos are not checked out. The guards this plan adds live in the source repos instead, which is where the rot starts.
- **`projects.thresholds`.** The server still stores a per-project `thresholds` blob, edited through `PATCH /projects/{id}` under `project:update` (`app/authz/permissions.py:30-32`). Only the push-time sync and its permission were removed. Nothing in this plan touches that field.
