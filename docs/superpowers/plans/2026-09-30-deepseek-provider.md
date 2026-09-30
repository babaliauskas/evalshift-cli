# DeepSeek as a Supported Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make DeepSeek a first-class EvalShift provider: CLI runs, captures replays, tool-call evals, `init`, doctor and the report handle it correctly. Every public surface that lists providers says so: CLI/SDK/action docs, the website docs and the landing page.

**Architecture:** The CLI already sends every call through LiteLLM, which knows the `deepseek/` provider. The work is mostly teaching EvalShift's own provider taxonomy about DeepSeek: registry, key pre-check, tool-response parsing, pricing lookup and `init`. There is also one small module for DeepSeek's thinking-mode behaviour. The SDK needs no code change, because `wrap_openai(OpenAI(base_url=...))` already captures DeepSeek. The action, SDK and client changes are docs and copy only.

**Tech Stack:** Python 3.11+, LiteLLM (`litellm>=1.77,<2`, 1.100.0 installed), Typer, pytest, mypy strict, ruff (CLI/SDK/action). Vite + React 19 + TypeScript + Vitest (client).

**Spec:** This plan's own **Findings** and **Design decisions** sections below. They come from the 2026-09-30 audit, which included a live LiteLLM probe run in `evalshift-cli/.venv`. There is no separate spec doc.

---

## Findings (why each task exists)

Verified on 2026-09-30 against the checked-out code and LiteLLM 1.100.0:

| # | Symptom today | Root cause |
|---|---|---|
| F1 | `deepseek/deepseek-flash` works for plain completions only by accident: provider is `"other"` | `registry._infer_provider_and_canonical` has no DeepSeek rule |
| F2 | A capture made with `OpenAI(base_url="https://api.deepseek.com")` records the bare id `deepseek-flash`. Replaying it fails: LiteLLM raises `BadRequestError: LLM Provider NOT provided` | Bare `deepseek-*` ids are not prefixed with `deepseek/` |
| F3 | Every tool-call eval on a DeepSeek arm raises `ModelError` | `tool_parser.detect_provider` raises `ToolParseError` for any id that isn't Anthropic, OpenAI or Gemini (`evaluators/tool_parser.py:39-64`) |
| F4 | A missing `DEEPSEEK_API_KEY` is not caught before a run. `doctor` doesn't show the key | `PROVIDER_ENV_VARS` has no DeepSeek entry, and `"other"` skips the pre-check (`cli/commands/run.py:222`) |
| F5 | A DeepSeek judge grading DeepSeek arms gets no self-preference warning | `models/family.py:65`: `"other"` never matches |
| F6 | `capture sync` prices a bare `deepseek-flash` call at $0 | `utils/cost._price_table_key` returns the bare key first. `litellm.cost_per_token("deepseek-flash")` raises (no provider); only `deepseek/deepseek-flash` prices |
| F7 | **Temperature is silently ignored**, and nothing warns | DeepSeek's current models (`deepseek-flash`, `deepseek-v4-pro`) run in *thinking mode by default*. Thinking mode "does not support `temperature`, `presence_penalty`, `frequency_penalty` … setting them will not trigger an error but will also have no effect" ([docs](https://api-docs.deepseek.com/guides/thinking_mode/)). LiteLLM still lists `temperature` as supported, so `honors_temperature` returns `True` |
| F8 | **Multi-round tool replay against DeepSeek returns HTTP 400** | With `tools` on the request, DeepSeek requires the `reasoning_content` of every earlier assistant turn, or it returns 400. `orchestrator.build_round_messages` (`runner/orchestrator.py:966-985`) and `_dispatch_message` (`:1031-1047`) build assistant turns without it. LiteLLM backfills a `" "` placeholder only when `thinking={"type":"enabled"}` is passed explicitly (`litellm/llms/deepseek/chat/transformation.py:214-227`). EvalShift never passes it, and our floor `litellm>=1.77` may predate that backfill anyway |
| F9 | The landing page says "Works with Anthropic, OpenAI and Google models." The CLI, action and site docs list only three key env vars | Copy |

DeepSeek API model ids as of 2026-09-30 ([pricing page](https://api-docs.deepseek.com/quick_start/pricing)):
- `deepseek-flash` (DeepSeek-V4.1-Flash, 1M context)
- `deepseek-v4-pro` (DeepSeek-V4-Pro-0813, 1M context)

The legacy `deepseek-v4-flash` is still accepted and served by Flash. LiteLLM 1.100.0 prices both current ids, reports `supports_reasoning=True` and `supports_function_calling=True`, and reads `DEEPSEEK_API_KEY` / `DEEPSEEK_API_BASE`.

## Design decisions

1. **DeepSeek becomes a registry `Provider` (`"deepseek"`).** It is attributed only for the `deepseek/` prefix and bare `deepseek-*` ids. The same weights served by another host stay `"other"`, because they authenticate with that host's keys, not `DEEPSEEK_API_KEY`. Examples: `azure_ai/deepseek-v4-pro`, `bedrock/…deepseek…`, `hosted_vllm/deepseek-ai/…`, `openrouter/deepseek/…`.
2. **Tool parsing treats DeepSeek as OpenAI-shaped, wherever it is hosted.** `detect_provider` returns `"openai"` for any id containing `deepseek`. Its `Provider` literal names a *response shape*, not a vendor. LiteLLM normalises DeepSeek tool calls to OpenAI's `tool_calls`, which is exactly what `_parse_gemini` already relies on for Gemini.
3. **EvalShift never switches thinking off.** A capture cannot record `thinking` (it is not in the SDK's `GENERATION_KEYS`), so the app's own setting is unknown. The API default is thinking on. Instead:
   - `honors_temperature` returns `False` for thinking-by-default DeepSeek models, so the report's existing non-determinism banner fires (fixes F7).
   - The client backfills `reasoning_content: " "` on assistant turns that lack it, for those models only (fixes F8). This is the same placeholder LiteLLM uses, done in EvalShift so it doesn't depend on the LiteLLM version or on an explicit `thinking` flag.
4. **"Thinking by default" means provider `deepseek` and `litellm.supports_reasoning(model=<canonical>)` is `True`.** Any exception or `False` reads as "not thinking". That matches `capabilities.py`'s rule that uncertainty reads as honoured (no false banner). Caveat (accepted): LiteLLM's reasoning flag is also `True` for DeepSeek models whose thinking is opt-in (e.g. `deepseek/deepseek-v3.2`), so those get a false non-determinism banner and a harmless `reasoning_content` placeholder. The current API ids `deepseek-flash` / `deepseek-v4-pro` think by default.
5. **Pricing lookup tries the canonical (provider-prefixed) id first.** This fixes F6. Measured effect on existing ids (accepted, ruling R10): identical for every text model — `openai/gpt-4o-mini` and `anthropic/claude-*` are not table keys, so they still fall through to the bare/stripped form. A few niche Gemini keys do change: `gemini-exp-1206` now prices at $0 because its `gemini/` entry has zero prices, and image models' cache-read price and the image-preview >200k-token tiers differ between the prefixed and bare entries.
6. **`init --provider deepseek`** scaffolds:
   - `deepseek-flash` as the source model
   - `deepseek-v4-pro` as the target hint and judge
   - the semantic block commented out, because DeepSeek has no embedding endpoint (same as Anthropic)

   `gemini` stays the default.
7. **Docs name the curated providers alphabetically** ("Anthropic, DeepSeek, Google and OpenAI"). This keeps `test_provider_scope_is_not_capped_at_three`'s regex guarding the old three-brand phrasing without editing it. A new docs-currency test makes every registry key env var and every `init --provider` choice appear in the reference docs, so the next provider can't be half-documented.

### Maintainer decisions to confirm

The plan proceeds with the default shown; change any of these before execution if you disagree.

- **D1 – doctor row.** `DEEPSEEK_API_KEY` gets an always-shown doctor row, like the other three. A non-DeepSeek user sees one more yellow `✗`. *Alternative:* show the row only when the key is set or the config names a DeepSeek model (more code, and it breaks the "one row per provider" rule).
- **D2 – landing copy.** `Works with Anthropic, OpenAI, Google and DeepSeek models.` This reverses the 2026-09-21 "do not touch Hero.tsx:113" note, because you asked for it. *Alternative:* `Works with Anthropic, OpenAI, Google, DeepSeek and any model LiteLLM supports.`
- **D3 – blog posts stay as published.** `llm-regression-testing-in-ci.md:40-42` lists three keys. Blog posts are dated articles, the same policy used for `all --push` on 2026-09-21.
- **D4 – release.** The CLI ships this as **1.2.0** (a new provider and a new `init` choice is a minor bump). The SDK change is docs-only and needs no PyPI release; the PyPI README catches up at the next SDK release. The client copy ships only after CLI 1.2.0 is on PyPI.

## Global Constraints

- Work happens in per-repo worktrees already cut from fresh `origin/main`: `/home/lukas/repos/evalshift/evalshift-{cli,sdk,action,client}-wt-deepseek` on branches `feat/deepseek-provider`, `docs/deepseek`, `docs/deepseek-key`, `feat/deepseek-copy`. Never touch the original checkouts. Always run git as `git -C <worktree>`. The plan file is gitignored-but-tracked: commit it with `git add -f`.
- CLI gates (all must pass before each commit): `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy --strict src/evalshift_cli`, `uv run pytest --cov-fail-under=90`, which together are `make ci`.
- Conventional Commits. Commit trailer: `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.
- Every user-visible CLI change updates `DOCS.md`, `llms-full.txt`, the matching `docs/` page and `CHANGELOG.md` under `## [Unreleased]` (per `evalshift-cli/CLAUDE.md:77`). The SDK has the same CHANGELOG rule. The action, server and client have no CHANGELOG.
- No version bumps inside feature PRs; they happen in the release commit (Task E1).
- `llms-full.txt` is hand-maintained in each repo. `evalshift-client/public/{cli,sdk,ci}-llms-full.txt` are **never** hand-edited; they are refreshed by `npm run sync:llms` (Task E2).
- Enumerate doc sites with a repo-wide `git grep` before editing, not only from the lists in this plan. Two past plans miscounted hand-written site lists.
- Model ids used everywhere: `deepseek-flash`, `deepseek-v4-pro`. Canonical: `deepseek/deepseek-flash`, `deepseek/deepseek-v4-pro`. Env var: `DEEPSEEK_API_KEY`.

## Review Focus

These are the five inputs most likely to hurt a DeepSeek user that no single task's happy path covers. Each has a pinning test in the task named after the arrow.

1. **A bare id recorded by a capture** (`deepseek-flash` with no prefix) must replay, price, pre-check its key and parse tool calls. → A1 (resolve), A2 (price), A3 (tool parse via the resolved id)
2. **A multi-round tool replay whose assistant turns came from another model** must not 400 on DeepSeek. → A4 client test with an assistant `tool_calls` turn and no `reasoning_content`; confirmed live in A7
3. **A DeepSeek arm must not be reported as deterministic.** → A4 capabilities test: `honors_temperature` is `False` even though LiteLLM lists `temperature`
4. **DeepSeek hosted elsewhere** (`azure_ai/deepseek-v4-pro`, `hosted_vllm/deepseek-ai/…`) must parse tool calls, but must not demand `DEEPSEEK_API_KEY`. → A1 (provider stays `"other"`) + A3 (parses as OpenAI shape)
5. **Non-DeepSeek providers must be byte-for-byte unaffected.** No `reasoning_content` on Gemini/OpenAI messages; unchanged pricing keys for `gpt-4o-mini` / `gemini-2.5-flash`. → A4 negative client test; A2 regression test

---

## File map

**evalshift-cli** (branch `feat/deepseek-provider`)
- Modify `src/evalshift_cli/models/registry.py`: `Provider`, `PROVIDER_ENV_VARS`, `_MODELS`, prefix inference
- Modify `src/evalshift_cli/utils/cost.py`: `_price_table_key` order
- Modify `src/evalshift_cli/evaluators/tool_parser.py`: `detect_provider`
- Create `src/evalshift_cli/models/deepseek.py`: thinking-mode quirks (one responsibility: DeepSeek API behaviour)
- Modify `src/evalshift_cli/models/capabilities.py`: `honors_temperature`
- Modify `src/evalshift_cli/models/client.py`: backfill in `_dispatch_with_retry`
- Modify `src/evalshift_cli/cli/commands/init.py`, `_scaffold.py`, `_agents.py`
- Modify `scripts/smoke_live_tools.py`
- Docs: `README.md`, `DOCS.md`, `llms-full.txt`, `docs/getting-started.md`, `docs/faq.md`, `docs/configuration.md`, `CHANGELOG.md`, `pyproject.toml` (keywords)
- Tests:
  - `tests/unit/test_model_registry.py`, `test_model_family.py`, `test_doctor.py`, `test_run_command.py`, `test_cost.py`
  - `test_tool_parser.py` (+ fixture `tests/unit/fixtures/tool_responses/deepseek/single_tool_call.json`)
  - new `test_deepseek.py`, plus `test_model_capabilities.py`, `test_model_client.py`, `test_init.py`, `test_docs_currency.py`

**evalshift-sdk** (branch `docs/deepseek`): `README.md`, `DOCS.md`, `llms-full.txt`, `CHANGELOG.md`

**evalshift-action** (branch `docs/deepseek-key`): `README.md`, `DOCS.md`, `llms-full.txt`, `tests/test_evalshift_action.py`

**evalshift-client** (branch `feat/deepseek-copy`):
- `src/pages/landing/sections/Hero.tsx`, `src/pages/landing/Landing.test.tsx`
- `src/pages/docs/pages/{CliCommands,Cli,Faq,Action,Configuration,GettingStarted,SdkAdapters}.tsx`

**evalshift-server:** no change. `source_model`/`target_model` are free strings (`app/runs/bundle.py:321-322`), and `judge_family_overlap` has no server-side enum.

---

## Phase A: evalshift-cli

### Task A0: Branch

- [ ] **Step 1: Cut the branch from fresh main**

```bash
git -C /home/lukas/repos/evalshift/evalshift-cli-wt-deepseek fetch origin
git -C /home/lukas/repos/evalshift/evalshift-cli-wt-deepseek switch -c feat/deepseek-provider origin/main
cd /home/lukas/repos/evalshift/evalshift-cli-wt-deepseek && uv sync
git -C /home/lukas/repos/evalshift/evalshift-cli-wt-deepseek add -f docs/superpowers/plans/2026-09-30-deepseek-provider.md
git -C /home/lukas/repos/evalshift/evalshift-cli-wt-deepseek commit -m "docs(plan): DeepSeek as a supported provider" -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task A1: Registry knows DeepSeek (fixes F1, F2, F4, F5)

**Files:**
- Modify: `src/evalshift_cli/models/registry.py:36-44` (`Provider`, `PROVIDER_ENV_VARS`), `:88-131` (`_MODELS`), `:237-263` (`_infer_provider_and_canonical` + docstrings at `:18-22`, `:66`)
- Test: `tests/unit/test_model_registry.py`, `tests/unit/test_model_family.py`, `tests/unit/test_doctor.py`, `tests/unit/test_run_command.py`

**Interfaces:**
- Produces: `Provider = Literal["anthropic", "openai", "google", "deepseek", "other"]`; `PROVIDER_ENV_VARS["deepseek"] == ("DEEPSEEK_API_KEY",)`; `resolve_model("deepseek-flash").id == "deepseek/deepseek-flash"`; `resolve_model(<bare deepseek-*>)` → `("deepseek/<id>", "deepseek")`. Doctor, `run`/`compare` pre-checks, `insights/stage.py` and `models/family.py` pick this up with no code change.

- [ ] **Step 1: Write the failing registry tests**

In `tests/unit/test_model_registry.py`, replace the body of `test_every_provider_represented`:

```python
    def test_every_provider_represented(self) -> None:
        providers = {m.provider for m in list_supported()}
        assert providers == {"anthropic", "openai", "google", "deepseek"}
```

Append to `class TestResolveModel`:

```python
    def test_bare_deepseek_alias_uses_registry(self) -> None:
        meta = resolve_model("deepseek-flash")
        assert meta.id == "deepseek/deepseek-flash"
        assert meta.provider == "deepseek"
        assert "(passthrough)" not in meta.display_name

    def test_unknown_deepseek_prefix_inferred(self) -> None:
        # What a capture records when the app called api.deepseek.com through
        # the OpenAI client: the bare id, which LiteLLM cannot route alone.
        meta = resolve_model("deepseek-v5-preview")
        assert meta.id == "deepseek/deepseek-v5-preview"
        assert meta.provider == "deepseek"
        assert meta.display_name.endswith("(passthrough)")

    def test_prefixed_deepseek_id_passes_through(self) -> None:
        meta = resolve_model("deepseek/deepseek-v5-preview")
        assert meta.id == "deepseek/deepseek-v5-preview"
        assert meta.provider == "deepseek"

    @pytest.mark.parametrize(
        "model_id",
        [
            "azure_ai/deepseek-v4-pro",
            "hosted_vllm/deepseek-ai/DeepSeek-V4-Flash",
            "openrouter/deepseek/deepseek-v4-pro",
        ],
    )
    def test_deepseek_on_another_host_is_not_the_deepseek_provider(self, model_id: str) -> None:
        # Another host authenticates with its own keys, never DEEPSEEK_API_KEY,
        # so it must not be attributed to the deepseek provider's key check.
        assert resolve_model(model_id).provider == "other"


class TestProviderEnvVars:
    def test_deepseek_key(self) -> None:
        assert PROVIDER_ENV_VARS["deepseek"] == ("DEEPSEEK_API_KEY",)
```

Add `PROVIDER_ENV_VARS` to the file's `from evalshift_cli.models.registry import (...)` block, and add `import pytest` if it is not already imported.

- [ ] **Step 2: Write the failing family, doctor and pre-check tests**

Append to `class TestSharedJudgeFamily` in `tests/unit/test_model_family.py`:

```python
    def test_deepseek_judge_on_a_deepseek_arm_is_one_family(self) -> None:
        roles = shared_judge_family(
            judge_model="deepseek-v4-pro",
            source_model="gpt-5.4-mini",
            target_model="deepseek-flash",
        )
        assert roles == ["target"]
```

In `tests/unit/test_doctor.py`, extend `TestRunChecksAPIKeys.test_partial_keys` with one more line at the end:

```python
        assert _by_name(results, "DEEPSEEK_API_KEY").status == "warn"
```

In `tests/unit/test_run_command.py`, add `"DEEPSEEK_API_KEY",` to the tuple in `TestRunApiKeyPrecheck._clear_keys`. Then append this method to the class:

```python
    def test_deepseek_arm_without_key_is_caught_before_the_run(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _scaffold(tmp_path)
        monkeypatch.chdir(tmp_path)
        self._clear_keys(monkeypatch)

        result = runner.invoke(
            app, ["run", "--from", "deepseek-flash", "--to", "deepseek-v4-pro", "--yes"]
        )
        assert result.exit_code == 1
        assert "missing API key" in result.stdout
        assert "DEEPSEEK_API_KEY" in result.stdout
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_model_registry.py tests/unit/test_model_family.py tests/unit/test_doctor.py::TestRunChecksAPIKeys tests/unit/test_run_command.py::TestRunApiKeyPrecheck -q`
Expected: FAIL. `providers` is missing `"deepseek"`, `resolve_model("deepseek-flash").provider == "other"`, `KeyError: 'deepseek'`, the family roles are `[]`, there is no `DEEPSEEK_API_KEY` row, and the run pre-check passes the DeepSeek arms.

- [ ] **Step 4: Implement**

In `registry.py`:

```python
Provider = Literal["anthropic", "openai", "google", "deepseek", "other"]

# Env vars LiteLLM reads to authenticate each provider, in preference
# order (primary first; the second entry is an accepted alias).
PROVIDER_ENV_VARS: Final[dict[Provider, tuple[str, ...]]] = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "google": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "deepseek": ("DEEPSEEK_API_KEY",),
}
```

Update the `ModelMetadata.provider` docstring line to ``"anthropic"`` | ``"openai"`` | ``"google"`` | ``"deepseek"``.

Append to `_MODELS`, after the Google block:

```python
    # ---- DeepSeek --------------------------------------------------------
    # The two ids DeepSeek's API serves as of 2026-09. Both run in thinking
    # mode by default, which ignores temperature — see models/deepseek.py.
    ModelMetadata(
        id="deepseek/deepseek-flash",
        provider="deepseek",
        display_name="DeepSeek V4.1 Flash",
        aliases=("deepseek-flash",),
    ),
    ModelMetadata(
        id="deepseek/deepseek-v4-pro",
        provider="deepseek",
        display_name="DeepSeek V4 Pro",
        aliases=("deepseek-v4-pro",),
    ),
```

In `_infer_provider_and_canonical`, add `"deepseek": "deepseek",` to `prefix_to_provider`. Add this branch after the `gpt-`/`o1-`/`o3-` branch:

```python
    if id_or_alias.startswith("deepseek-"):
        return f"deepseek/{id_or_alias}", "deepseek"
```

Add a bullet to its docstring decision tree ("If it starts with ``deepseek-`` → deepseek, prefix ``deepseek/``."). Add the same rule to the module docstring's prefix list (`:18-22`).

- [ ] **Step 5: Run the tests to verify they pass, then the full gate**

Run: `uv run pytest tests/unit/test_model_registry.py tests/unit/test_model_family.py tests/unit/test_doctor.py tests/unit/test_run_command.py -q`, then `make ci`
Expected: PASS. mypy may flag a `match`/`dict` over `Provider` that is now non-exhaustive. Fix any such site by adding the `"deepseek"` case, not by casting.

- [ ] **Step 6: Commit**

```bash
git -C /home/lukas/repos/evalshift/evalshift-cli-wt-deepseek add src/evalshift_cli/models/registry.py tests/unit/test_model_registry.py tests/unit/test_model_family.py tests/unit/test_doctor.py tests/unit/test_run_command.py
git -C /home/lukas/repos/evalshift/evalshift-cli-wt-deepseek commit -m "feat(models): register DeepSeek as a provider" -m "Bare deepseek-* ids now resolve to deepseek/…, DEEPSEEK_API_KEY is pre-checked and shown by doctor, and a DeepSeek judge on a DeepSeek arm is flagged as one family." -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task A2: Price a bare id under its provider-prefixed key (fixes F6)

**Files:**
- Modify: `src/evalshift_cli/utils/cost.py:138-153` (`_price_table_key`)
- Test: `tests/unit/test_cost.py` (`class TestEstimateCallCost`)

**Interfaces:**
- Consumes: `resolve_model("deepseek-flash").id == "deepseek/deepseek-flash"` (A1)
- Produces: `estimate_call_cost("deepseek-flash", …) > 0` with the real LiteLLM table

- [ ] **Step 1: Write the failing test**

Append to `class TestEstimateCallCost`:

```python
    def test_bare_id_is_priced_under_its_provider_prefixed_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # litellm's table holds DeepSeek under both the bare and the prefixed
        # key, but cost_per_token can only infer a provider from the prefixed
        # one — asked about the bare id it raises, which used to price a
        # DeepSeek capture at $0.
        monkeypatch.setattr(
            cost_module.litellm,
            "model_cost",
            {"deepseek-flash": {}, "deepseek/deepseek-flash": {}},
        )

        def priced(*, model: str, prompt_tokens: int, completion_tokens: int) -> tuple[float, float]:
            if model != "deepseek/deepseek-flash":
                raise ValueError(f"LLM Provider NOT provided: {model}")
            return prompt_tokens * 0.001, completion_tokens * 0.002

        monkeypatch.setattr(cost_module.litellm, "cost_per_token", priced)

        assert cost_module.estimate_call_cost("deepseek-flash", 1000, 100) == pytest.approx(1.2)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/unit/test_cost.py::TestEstimateCallCost -q`
Expected: the new test FAILS (`0.0 != 1.2`). The other five pass.

- [ ] **Step 3: Implement**

In `_price_table_key`, change the loop to try the canonical form first, and update the docstring:

```python
    """The key litellm's price table holds ``model_id`` under, or ``None``.

    Tries the registry's canonical (provider-prefixed) form first, then the id
    as recorded, then the canonical form with the prefix stripped. The order
    matters: litellm keys some providers under both spellings but can only
    price the prefixed one (``deepseek-flash`` is a key, yet
    ``cost_per_token`` cannot infer its provider), while most first-party
    entries exist only bare (``gpt-4o-mini``) and fall through to the second
    or third candidate. A pure dict lookup: ``litellm.model_cost`` is the
    bundled table, so a miss costs nothing and touches nothing.
    """
    canonical = resolve_model(model_id).id
    _, _, stripped = canonical.partition("/")
    table = litellm.model_cost
    for candidate in (canonical, model_id, stripped):
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/unit/test_cost.py tests/unit/test_captures_promote.py -q`
Expected: PASS. `test_resolves_registry_aliases_and_provider_prefixes` still passes because `gemini/gemini-2.5-flash` is not in its stub table.

- [ ] **Step 5: Confirm against the real table (no network)**

Run: `uv run python -c "from evalshift_cli.utils.cost import estimate_call_cost as c; print([c(m,1000,1000) for m in ('deepseek-flash','gpt-4o-mini','gemini-2.5-flash','claude-sonnet-4-5')])"`
Expected: four non-zero numbers.

- [ ] **Step 6: Commit**

```bash
git -C /home/lukas/repos/evalshift/evalshift-cli-wt-deepseek add src/evalshift_cli/utils/cost.py tests/unit/test_cost.py
git -C /home/lukas/repos/evalshift/evalshift-cli-wt-deepseek commit -m "fix(cost): price recorded calls under the provider-prefixed key first" -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task A3: Parse DeepSeek tool calls as OpenAI-shaped (fixes F3)

**Files:**
- Modify: `src/evalshift_cli/evaluators/tool_parser.py:1-11` (module docstring), `:39-64` (`detect_provider`)
- Create: `tests/unit/fixtures/tool_responses/deepseek/single_tool_call.json`
- Test: `tests/unit/test_tool_parser.py` (`TestDetectProvider`, new `TestParseDeepSeek`), `tests/unit/test_model_client.py` (`TestCompleteWithTools`)

**Interfaces:**
- Consumes: A1's canonical ids
- Produces: `detect_provider(<any id containing "deepseek">) == "openai"`. `ModelClient.complete_with_tools` serialises tools with `to_openai()` for DeepSeek and parses with `_parse_openai`

- [ ] **Step 1: Create the fixture**

`tests/unit/fixtures/tool_responses/deepseek/single_tool_call.json` is a LiteLLM-normalised DeepSeek thinking-mode response. It carries `reasoning_content`, which the parser must ignore:

```json
{
  "id": "chatcmpl_synthetic_deepseek_single",
  "object": "chat.completion",
  "model": "deepseek-flash",
  "choices": [
    {
      "index": 0,
      "finish_reason": "tool_calls",
      "message": {
        "role": "assistant",
        "content": "",
        "reasoning_content": "The user wants ACME's Q3 records; search the database first.",
        "tool_calls": [
          {
            "id": "call_00_synthetic",
            "type": "function",
            "function": {
              "name": "search_db",
              "arguments": "{\"query\": \"ACME Q3\"}"
            }
          }
        ]
      }
    }
  ],
  "usage": {"prompt_tokens": 60, "completion_tokens": 40, "total_tokens": 100}
}
```

- [ ] **Step 2: Write the failing tests**

In `TestDetectProvider.test_known_models`'s parametrize list, add:

```python
            ("deepseek/deepseek-flash", "openai"),
            ("deepseek-v4-pro", "openai"),
            # Same weights, other hosts: LiteLLM returns OpenAI's shape for all.
            ("azure_ai/deepseek-v4-pro", "openai"),
            ("hosted_vllm/deepseek-ai/DeepSeek-V4-Flash", "openai"),
```

Add after the OpenAI parser tests:

```python
class TestParseDeepSeek:
    def test_single_tool_call_ignores_reasoning_content(self) -> None:
        raw = _load("deepseek", "single_tool_call")
        model_id = "deepseek/deepseek-flash"
        trace = parse_response_to_trace(
            raw, provider=detect_provider(model_id), model_id=model_id
        )
        assert trace.call_count == 1
        assert trace.calls[0].tool_name == "search_db"
        assert trace.calls[0].arguments == {"query": "ACME Q3"}
        assert trace.calls[0].call_id == "call_00_synthetic"
        # The reasoning chain is not the answer.
        assert not trace.final_text
```

Append to `TestCompleteWithTools` in `tests/unit/test_model_client.py`:

```python
    async def test_deepseek_bare_id_dispatches_openai_shaped_tools(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = _patch_tools_acompletion(monkeypatch, _OPENAI_SINGLE_RESPONSE)
        result = await ModelClient().complete_with_tools(
            model="deepseek-flash",
            prompt="hi",
            tools=[_DEMO_TOOL],
        )
        assert result.model_id == "deepseek/deepseek-flash"
        assert result.trace.calls[0].tool_name == "search_db"
        kwargs = captured["kwargs"]
        assert kwargs["model"] == "deepseek/deepseek-flash"
        assert kwargs["tools"][0]["type"] == "function"
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_tool_parser.py tests/unit/test_model_client.py::TestCompleteWithTools -q`
Expected: FAIL with `ToolParseError: unknown: cannot detect provider for model id: 'deepseek/deepseek-flash'`.

- [ ] **Step 4: Implement**

In `detect_provider`, insert this before the final `raise`:

```python
    if "deepseek" in lowered:
        # DeepSeek's API is OpenAI-compatible, and LiteLLM hands its tool calls
        # back in OpenAI's ``tool_calls`` shape wherever the model is hosted
        # (deepseek/, azure_ai/, bedrock/, hosted_vllm/, openrouter/ ...).
        return "openai"
```

Update `detect_provider`'s docstring. Its `Raises:` paragraph should say the id "can't be mapped to one of the response shapes we parse (Anthropic, OpenAI, Gemini)". Add a sentence saying the return value names a **response shape**, so DeepSeek maps to `"openai"`. Add the same note to the module docstring.

- [ ] **Step 5: Run the tests, then the gate**

Run: `uv run pytest tests/unit/test_tool_parser.py tests/unit/test_model_client.py tests/unit/test_model_capabilities.py -q`, then `make ci`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git -C /home/lukas/repos/evalshift/evalshift-cli-wt-deepseek add src/evalshift_cli/evaluators/tool_parser.py tests/unit/fixtures/tool_responses/deepseek tests/unit/test_tool_parser.py tests/unit/test_model_client.py
git -C /home/lukas/repos/evalshift/evalshift-cli-wt-deepseek commit -m "feat(tools): parse DeepSeek tool calls as OpenAI-shaped" -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task A4: DeepSeek thinking mode: banner the ignored temperature, backfill reasoning (fixes F7, F8)

**Files:**
- Create: `src/evalshift_cli/models/deepseek.py`
- Modify: `src/evalshift_cli/models/capabilities.py:164-185` (`honors_temperature`) and its module docstring; `src/evalshift_cli/models/client.py:684-730` (`_dispatch_with_retry` entry)
- Test: create `tests/unit/test_deepseek.py`; extend `tests/unit/test_model_capabilities.py` and `tests/unit/test_model_client.py`

**Interfaces:**
- Consumes: `resolve_model(...).provider == "deepseek"` (A1)
- Produces:
  - `evalshift_cli.models.deepseek.REASONING_PLACEHOLDER: Final = " "`
  - `thinking_by_default(model_id: str) -> bool`
  - `backfill_reasoning_content(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]`
  - `honors_temperature("deepseek-flash") is False` whenever LiteLLM reports reasoning support

- [ ] **Step 1: Write the failing module tests**

`tests/unit/test_deepseek.py`:

```python
"""Tests for :mod:`evalshift_cli.models.deepseek`.

LiteLLM's reasoning flag is stubbed in every test: the bundled table is
upstream data, and a LiteLLM upgrade must not flip these tests.
"""

from __future__ import annotations

from typing import Any

import pytest

from evalshift_cli.models import deepseek as deepseek_module
from evalshift_cli.models.deepseek import (
    REASONING_PLACEHOLDER,
    backfill_reasoning_content,
    thinking_by_default,
)


def _stub_reasoning(monkeypatch: pytest.MonkeyPatch, result: bool | Exception) -> list[str]:
    seen: list[str] = []

    def fake(*, model: str, **_: Any) -> bool:
        seen.append(model)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(deepseek_module.litellm, "supports_reasoning", fake)
    return seen


class TestThinkingByDefault:
    def test_reasoning_deepseek_model_thinks_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen = _stub_reasoning(monkeypatch, True)
        assert thinking_by_default("deepseek-flash") is True
        # Asked about the canonical id, which is what LiteLLM keys on.
        assert seen == ["deepseek/deepseek-flash"]

    def test_non_reasoning_deepseek_model_does_not(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_reasoning(monkeypatch, False)
        assert thinking_by_default("deepseek/deepseek-chat") is False

    def test_uncertain_answer_reads_as_not_thinking(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_reasoning(monkeypatch, RuntimeError("no model info"))
        assert thinking_by_default("deepseek-v5-preview") is False

    @pytest.mark.parametrize(
        "model_id", ["gemini/gemini-2.5-flash", "gpt-4o", "azure_ai/deepseek-v4-pro"]
    )
    def test_other_providers_never_ask(
        self, monkeypatch: pytest.MonkeyPatch, model_id: str
    ) -> None:
        seen = _stub_reasoning(monkeypatch, True)
        assert thinking_by_default(model_id) is False
        assert seen == []


class TestBackfillReasoningContent:
    def test_assistant_turn_without_reasoning_gets_the_placeholder(self) -> None:
        messages = [
            {"role": "user", "content": "find ACME"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call_r0_0"}]},
            {"role": "tool", "tool_call_id": "call_r0_0", "content": "{}"},
        ]
        out = backfill_reasoning_content(messages)
        assert out[1]["reasoning_content"] == REASONING_PLACEHOLDER == " "
        assert "reasoning_content" not in out[0]
        assert "reasoning_content" not in out[2]

    def test_recorded_reasoning_is_kept(self) -> None:
        messages = [{"role": "assistant", "content": "x", "reasoning_content": "because"}]
        assert backfill_reasoning_content(messages)[0]["reasoning_content"] == "because"

    def test_input_is_not_mutated(self) -> None:
        turn = {"role": "assistant", "content": ""}
        backfill_reasoning_content([turn])
        assert "reasoning_content" not in turn
```

- [ ] **Step 2: Write the failing capability and client tests**

Append to `tests/unit/test_model_capabilities.py`:

```python
class TestDeepSeekThinkingMode:
    """Thinking mode accepts ``temperature`` and ignores it; LiteLLM still lists it."""

    def test_thinking_deepseek_model_does_not_honour_temperature(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_params(monkeypatch, _WITH_TEMPERATURE)
        monkeypatch.setattr(litellm, "supports_reasoning", lambda **_: True)
        assert honors_temperature("deepseek-flash") is False

    def test_non_thinking_deepseek_model_falls_back_to_the_probe(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_params(monkeypatch, _WITH_TEMPERATURE)
        monkeypatch.setattr(litellm, "supports_reasoning", lambda **_: False)
        assert honors_temperature("deepseek/deepseek-chat") is True
```

Append to `tests/unit/test_model_client.py` (import `litellm` at the top if it isn't already):

```python
# ---------------------------------------------------------------------------
# DeepSeek thinking mode
# ---------------------------------------------------------------------------

_REPLAYED_ROUND: list[dict[str, Any]] = [
    {"role": "user", "content": "find ACME"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_r0_0",
                "type": "function",
                "function": {"name": "search_db", "arguments": '{"query": "ACME"}'},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_r0_0", "content": '{"rows": []}'},
]


class TestDeepSeekReasoningBackfill:
    async def test_replayed_assistant_turn_carries_placeholder_reasoning(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # DeepSeek 400s a tools request whose earlier assistant turns lack
        # reasoning_content; a teacher-forced round never has DeepSeek's own.
        monkeypatch.setattr(litellm, "supports_reasoning", lambda **_: True)
        captured = _patch_tools_acompletion(monkeypatch, _OPENAI_SINGLE_RESPONSE)
        await ModelClient().complete_messages_with_tools(
            model="deepseek-flash",
            messages=[dict(m) for m in _REPLAYED_ROUND],
            tools=[_DEMO_TOOL],
        )
        sent = captured["kwargs"]["messages"]
        assert sent[1]["reasoning_content"] == " "
        assert "reasoning_content" not in sent[0]

    async def test_other_providers_are_sent_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(litellm, "supports_reasoning", lambda **_: True)
        captured = _patch_tools_acompletion(monkeypatch, _OPENAI_SINGLE_RESPONSE)
        await ModelClient().complete_messages_with_tools(
            model="gpt-4o",
            messages=[dict(m) for m in _REPLAYED_ROUND],
            tools=[_DEMO_TOOL],
        )
        assert captured["kwargs"]["messages"] == _REPLAYED_ROUND
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_deepseek.py tests/unit/test_model_capabilities.py tests/unit/test_model_client.py -q`
Expected: `ModuleNotFoundError: evalshift_cli.models.deepseek`. The capability test returns `True`, and the client sends no `reasoning_content`.

- [ ] **Step 4: Create `src/evalshift_cli/models/deepseek.py`**

```python
"""DeepSeek API behaviour EvalShift has to account for, in one place.

DeepSeek's current API models (``deepseek-flash``, ``deepseek-v4-pro``) run in
*thinking mode* unless the request turns it off
(https://api-docs.deepseek.com/guides/thinking_mode/). Thinking mode changes
two things the replay path depends on:

* ``temperature`` is accepted and ignored — "setting them will not trigger an
  error but will also have no effect". LiteLLM still lists ``temperature`` as
  supported, so :func:`~evalshift_cli.models.capabilities.honors_temperature`
  asks :func:`thinking_by_default` before trusting LiteLLM's answer, and the
  report's non-determinism banner covers DeepSeek arms.
* A request carrying ``tools`` must pass back the ``reasoning_content`` of
  every earlier assistant turn, or the API answers 400. A teacher-forced round
  is rebuilt from the *recording* — often another model's — so there is no
  DeepSeek reasoning to pass. :func:`backfill_reasoning_content` supplies the
  single-space placeholder the API accepts. LiteLLM does the same, but only
  when the caller passes ``thinking`` explicitly, which EvalShift never does
  (a capture cannot record it); doing it here also keeps the fix independent
  of the installed LiteLLM version.

EvalShift never switches thinking off: the application under test runs with
DeepSeek's default, and replaying a different configuration would measure
the wrong thing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

import litellm

from evalshift_cli.models.registry import resolve_model

#: The minimum ``reasoning_content`` DeepSeek accepts on an assistant turn —
#: an empty reasoning chain, the same value LiteLLM injects.
REASONING_PLACEHOLDER: Final = " "


def thinking_by_default(model_id: str) -> bool:
    """Report whether ``model_id`` is a DeepSeek API model that thinks by default.

    Args:
        model_id: Any user-supplied model id or alias; resolved first, so a
            bare id recorded by a capture works.

    Returns:
        ``True`` only for the ``deepseek`` provider when LiteLLM positively
        reports reasoning support for the canonical id. Every other provider,
        DeepSeek weights on another host, and every uncertain LiteLLM answer
        return ``False`` — the same "uncertainty reads as honoured" rule as
        :mod:`evalshift_cli.models.capabilities`.
    """
    meta = resolve_model(model_id)
    if meta.provider != "deepseek":
        return False
    try:
        return bool(litellm.supports_reasoning(model=meta.id))
    except Exception:
        return False


def backfill_reasoning_content(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return ``messages`` with :data:`REASONING_PLACEHOLDER` on bare assistant turns.

    Assistant turns that already carry a non-empty ``reasoning_content`` keep
    it; every other role passes through. The input is not mutated.
    """
    out: list[dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") == "assistant" and not msg.get("reasoning_content"):
            out.append({**msg, "reasoning_content": REASONING_PLACEHOLDER})
        else:
            out.append(dict(msg))
    return out


__all__ = ["REASONING_PLACEHOLDER", "backfill_reasoning_content", "thinking_by_default"]
```

- [ ] **Step 5: Wire it into capabilities and the client**

In `capabilities.py`, import `from evalshift_cli.models.deepseek import thinking_by_default` and change `honors_temperature`'s body:

```python
    if thinking_by_default(model_id):
        # Accepted and ignored by DeepSeek thinking mode; LiteLLM cannot see
        # that. See evalshift_cli.models.deepseek.
        return False
    return not unsupported_params(model_id, ["temperature"])
```

Extend its docstring's `Returns:` paragraph with one sentence naming this exception. Also add a bullet to the module docstring's list of exceptions to "LiteLLM's answer is the authority".

In `client.py`, import `from evalshift_cli.models.deepseek import backfill_reasoning_content, thinking_by_default`. Insert this at the top of `_dispatch_with_retry`, before the existing `if canonical in self._temperature_rejected:` line:

```python
        if "messages" in kwargs and thinking_by_default(canonical):
            # DeepSeek thinking mode 400s a tools request whose earlier
            # assistant turns lack reasoning_content; see models/deepseek.py.
            kwargs["messages"] = backfill_reasoning_content(kwargs["messages"])
```

Add one sentence about this to the `_dispatch_with_retry` docstring's "adaptation" paragraph.

- [ ] **Step 6: Run the tests, then the gate**

Run: `uv run pytest tests/unit/test_deepseek.py tests/unit/test_model_capabilities.py tests/unit/test_model_client.py tests/unit/test_orchestrator.py tests/unit/test_tool_rounds.py -q`, then `make ci`
Expected: PASS. If an orchestrator test asserts on exact dispatched `messages` for a DeepSeek id, it doesn't exist today, so no fixture should change. Any diff means a non-DeepSeek path was touched, which is a bug.

- [ ] **Step 7: Commit**

```bash
git -C /home/lukas/repos/evalshift/evalshift-cli-wt-deepseek add src/evalshift_cli/models/deepseek.py src/evalshift_cli/models/capabilities.py src/evalshift_cli/models/client.py tests/unit/test_deepseek.py tests/unit/test_model_capabilities.py tests/unit/test_model_client.py
git -C /home/lukas/repos/evalshift/evalshift-cli-wt-deepseek commit -m "feat(models): account for DeepSeek thinking mode" -m "Thinking mode ignores temperature, so DeepSeek arms now get the non-determinism banner; replayed assistant turns get the placeholder reasoning_content DeepSeek requires on tools requests." -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task A5: `evalshift init --provider deepseek`

**Files:**
- Modify: `src/evalshift_cli/cli/commands/init.py:45` (`PROVIDERS`), `:51-70` (`_PROVIDER_MODELS`), `:83-90` (`_SEMANTIC_BLOCK_DISABLED`)
- Modify: `src/evalshift_cli/cli/commands/_scaffold.py:109-114` (`PROVIDER_API_KEY_ENVS`)
- Modify: `src/evalshift_cli/cli/commands/_agents.py:86-88`
- Test: `tests/unit/test_init.py` (`TestInitProvider`, the workflow class holding `test_provider_key_matches_provider`)

**Interfaces:**
- Consumes: `PROVIDER_ENV_VARS`, `resolve_model` (A1)
- Produces: `PROVIDERS == ("gemini", "openai", "anthropic", "deepseek")`, which A6's docs test reads as `"|".join(PROVIDERS)`

- [ ] **Step 1: Write the failing tests**

In `TestInitProvider`, change the `test_every_provider_config_round_trips` loop to `for provider in PROVIDERS:`. Import `PROVIDERS` from `evalshift_cli.cli.commands.init`, and import `PROVIDER_API_KEY_ENVS` from `evalshift_cli.cli.commands._scaffold` if it isn't imported already. Then append:

```python
    def test_deepseek_provider_writes_deepseek_ids_and_comments_out_semantic(
        self, in_tmp: Path
    ) -> None:
        result = runner.invoke(app, ["init", "--provider", "deepseek"])
        assert result.exit_code == 0, result.stdout
        cfg = load_config(in_tmp / CONFIG_FILENAME)
        assert cfg.defaults.source_model == "deepseek-flash"
        assert cfg.evaluators.llm_judge[0].judge_model == "deepseek-v4-pro"
        # DeepSeek has no embedding endpoint — semantic ships commented out.
        assert cfg.evaluators.semantic is None
        body = (in_tmp / CONFIG_FILENAME).read_text(encoding="utf-8")
        assert "# semantic:" in body
        assert "Anthropic has no embedding" not in body
        assert "DEEPSEEK_API_KEY" in result.stdout

    def test_every_init_provider_key_is_the_registry_key(self) -> None:
        # init and the run pre-check must agree on which env var authenticates
        # a scaffold's models, or `init` tells the user to export the wrong one.
        for provider in PROVIDERS:
            source = _PROVIDER_MODELS[provider]["source_model"]
            registry_provider = resolve_model(source).provider
            assert PROVIDER_API_KEY_ENVS[provider] == PROVIDER_ENV_VARS[registry_provider][0]
```

Imports for that test: `_PROVIDER_MODELS` from `evalshift_cli.cli.commands.init`; `PROVIDER_ENV_VARS, resolve_model` from `evalshift_cli.models.registry`.

In the workflow test class, next to `test_provider_key_matches_provider`, add:

```python
    def test_deepseek_workflow_uses_the_deepseek_key(self, in_tmp: Path) -> None:
        body, _ = self._workflow(in_tmp, "--provider", "deepseek")
        assert "DEEPSEEK_API_KEY" in body
        assert "GEMINI_API_KEY" not in body
```

Also change the comment at `test_init.py:269` to `# No Anthropic embedding endpoint — semantic ships commented out.`, because the scaffold wording is changing.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_init.py -q`
Expected: FAIL. `--provider deepseek` exits 2 ("unknown provider"), and `PROVIDER_API_KEY_ENVS` has no `"deepseek"` entry.

- [ ] **Step 3: Implement**

`init.py`:

```python
PROVIDERS: Final = ("gemini", "openai", "anthropic", "deepseek")
```

Add to `_PROVIDER_MODELS`:

```python
    "deepseek": {
        "source_model": "deepseek-flash",
        "target_hint": "deepseek-v4-pro",
        "judge_model": "deepseek-v4-pro",
        "embedding_model": "",  # no DeepSeek embedding endpoint
    },
```

Change the first two comment lines of `_SEMANTIC_BLOCK_DISABLED` to:

```
  # Embedding-based drift score (advisory). This provider has no embedding
  # endpoint — uncomment and set an OpenAI or Gemini embedding model (and
```

`_scaffold.py`: add `"deepseek": "DEEPSEEK_API_KEY",` to `PROVIDER_API_KEY_ENVS`.

`_agents.py:87`: change it to ``  (e.g. `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`) matching the``. Re-wrap the line if ruff's line length requires it.

- [ ] **Step 4: Find tests that pin the changed text**

Run: `git -C /home/lukas/repos/evalshift/evalshift-cli-wt-deepseek grep -n "Anthropic has no embedding\|ANTHROPIC_API_KEY\`, \`OPENAI_API_KEY\`)" -- tests src`
Expected: no hits outside the lines just edited. Update any hit to the new wording.

- [ ] **Step 5: Run the tests, then the gate**

Run: `uv run pytest tests/unit/test_init.py tests/unit/test_agents.py -q`, then `make ci`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git -C /home/lukas/repos/evalshift/evalshift-cli-wt-deepseek add src/evalshift_cli/cli/commands/init.py src/evalshift_cli/cli/commands/_scaffold.py src/evalshift_cli/cli/commands/_agents.py tests/unit/test_init.py
git -C /home/lukas/repos/evalshift/evalshift-cli-wt-deepseek commit -m "feat(init): scaffold DeepSeek projects with --provider deepseek" -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task A6: CLI docs, a docs-currency guard, CHANGELOG

**Files:**
- Modify: `tests/unit/test_docs_currency.py`
- Modify:
  - `README.md:302-306`
  - `DOCS.md:72-80, 136, 385, 803, 888-890` + a new FAQ answer after `:931`
  - `llms-full.txt:120-121, 425-427, 615-618` + the same FAQ answer
  - `docs/getting-started.md:43-48`
  - `docs/faq.md:50-56` + new question
  - `docs/configuration.md:595-597`
  - `pyproject.toml:14`
  - `CHANGELOG.md`

**Interfaces:**
- Consumes: `PROVIDER_ENV_VARS` (A1), `PROVIDERS` (A5)

- [ ] **Step 1: Write the failing guard**

Append to `tests/unit/test_docs_currency.py` (with the imports at the top of the file):

```python
from evalshift_cli.cli.commands.init import PROVIDERS
from evalshift_cli.models.registry import PROVIDER_ENV_VARS

#: Files that tell a user which env var authenticates which provider. A
#: provider the registry can authenticate but these files never name is a
#: provider whose users are told nothing — the state DeepSeek support shipped
#: into on 2026-09-30.
KEY_TABLE_FILES: tuple[str, ...] = ("DOCS.md", "llms-full.txt", "docs/getting-started.md")

#: Files that spell out `init --provider`'s choices.
INIT_PROVIDER_FILES: tuple[str, ...] = ("DOCS.md", "llms-full.txt")


@pytest.mark.parametrize("name", KEY_TABLE_FILES)
@pytest.mark.parametrize("env_var", sorted(aliases[0] for aliases in PROVIDER_ENV_VARS.values()))
def test_key_docs_name_every_registry_provider(name: str, env_var: str) -> None:
    text = (REPO_ROOT / name).read_text(encoding="utf-8")

    assert env_var in text, f"{name} never tells users about {env_var}"


@pytest.mark.parametrize("name", INIT_PROVIDER_FILES)
def test_docs_list_every_init_provider(name: str) -> None:
    text = (REPO_ROOT / name).read_text(encoding="utf-8")

    assert f"--provider {'|'.join(PROVIDERS)}" in text, (
        f"{name} lists `init --provider` choices that differ from init.PROVIDERS"
    )
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/unit/test_docs_currency.py -q`
Expected: FAIL on `DEEPSEEK_API_KEY` in all three files, and on `--provider gemini|openai|anthropic|deepseek` in both.

- [ ] **Step 3: Enumerate every site first**

Run: `git -C /home/lukas/repos/evalshift/evalshift-cli-wt-deepseek grep -nE "GEMINI_API_KEY|gemini\|openai\|anthropic|Claude, GPT|\(\`anthropic\`|gemini-\*. → Google|Anthropic, OpenAI and Google" -- README.md DOCS.md llms-full.txt docs AGENTS.md`
Expected: the sites listed in this task's **Files** block. Treat any extra hit as in scope unless it is a Gemini-only example (`examples/`, the `init --provider gemini` sample output) or tool/wire-shape prose.

- [ ] **Step 4: Edit the key and choice sites**

Use these exact edits:

- `DOCS.md:72-80`: add `export DEEPSEEK_API_KEY=...` as a fourth line in the export block.
- `DOCS.md:888-890`: add the row ``| `DEEPSEEK_API_KEY` | — | DeepSeek auth |``.
- `llms-full.txt:425-427`: add the row `| DEEPSEEK_API_KEY | — | DeepSeek auth |`.
- `docs/getting-started.md:46-48`: add `export DEEPSEEK_API_KEY=<deepseek-api-key>`.
- `DOCS.md:136`: replace the `--provider` bullet with:

  ``- `--provider gemini|openai|anthropic|deepseek` — which provider's model ids the scaffold uses (prompted on a TTY; defaults to `gemini` otherwise). Gemini and OpenAI scaffolds include an embedding-based semantic evaluator; the Anthropic and DeepSeek scaffolds comment it out (no embedding endpoint).``
- `DOCS.md:803`: `--provider gemini|openai|anthropic` → `--provider gemini|openai|anthropic|deepseek`.
- `llms-full.txt:121`: `[--provider gemini|openai|anthropic]` → `[--provider gemini|openai|anthropic|deepseek]`.

- [ ] **Step 5: Edit the provider-scope prose (alphabetical, see Design decision 7)**

- `README.md:304`: `Anthropic, OpenAI and Google ids additionally get a curated pricing and` → `Anthropic, DeepSeek, Google and OpenAI ids additionally get a curated pricing and`.
- `docs/faq.md:51-52`: `(Claude, GPT,` / `Gemini)` → `(Claude, DeepSeek,` / `Gemini, GPT)`.
- `DOCS.md:385` and `llms-full.txt:617`: in the prefix-inference list, add `` `deepseek-*` → DeepSeek`` in DOCS.md and `deepseek-* -> deepseek` in llms-full.txt, after the OpenAI entry.
- `docs/configuration.md:596`: `(`anthropic`, `openai`, `google`)` → `(`anthropic`, `deepseek`, `google`, `openai`)`.
- `pyproject.toml:14`: add `"deepseek"` to `keywords` after `"gemini"`.

- [ ] **Step 6: Add the DeepSeek answer**

Add this as a new question in `docs/faq.md`, placed after "what models does EvalShift support?". Add the same text to `DOCS.md` after line 931's answer, as a `###`-level entry matching its neighbours. Add it to `llms-full.txt` after line 618, re-wrapped in that file's terse style but keeping every fact:

```markdown
### Does EvalShift work with DeepSeek?

Yes. Export `DEEPSEEK_API_KEY` and use DeepSeek's API ids, `deepseek-flash`
or `deepseek-v4-pro`. A bare `deepseek-*` id (what a capture records when your
app calls `api.deepseek.com` through the OpenAI client) gets the `deepseek/`
prefix automatically. `evalshift init --provider deepseek` scaffolds a
DeepSeek project. Three things differ from other providers:

- **Sampling is not controlled.** Both models run in thinking mode by
  default, which accepts `temperature` and ignores it. EvalShift keeps thinking
  on, because that is what your application runs, so DeepSeek arms are marked
  non-deterministic in the report. Raise `defaults.samples_per_example` when
  the verdict matters.
- **Replayed tool rounds carry an empty reasoning chain.** DeepSeek requires
  the `reasoning_content` of earlier assistant turns on any request with
  tools. A teacher-forced round comes from the recording, not from DeepSeek,
  so EvalShift sends the single-space placeholder the API accepts.
- **No embeddings.** DeepSeek has no embedding endpoint. The `semantic`
  evaluator needs an OpenAI or Gemini embedding model and its key, which is
  why the DeepSeek scaffold ships it commented out.

DeepSeek served by another host (self-hosted open weights, or a cloud region
of your choice) goes through that host's LiteLLM prefix (`hosted_vllm/`,
`azure_ai/`, `bedrock/`, ...) and its environment variables. Tool calls parse
the same way, but the key pre-check and the notes above apply to the
`deepseek/` API only. LiteLLM also reads `DEEPSEEK_API_BASE` to point the
`deepseek/` provider at a DeepSeek-compatible endpoint.
```

- [ ] **Step 7: Add the CHANGELOG entry**

Under `## [Unreleased]`, add an `### Added` section above the existing `### Fixed` if there is none:

```markdown
### Added

- DeepSeek is a supported provider. `deepseek-flash` and `deepseek-v4-pro`
  are in the model registry; bare `deepseek-*` ids (as recorded by a capture
  of an app calling `api.deepseek.com` through the OpenAI client) resolve to
  `deepseek/…`; `DEEPSEEK_API_KEY` is checked before a run and shown by
  `evalshift doctor`; tool-call evals parse DeepSeek responses; a DeepSeek
  judge grading a DeepSeek arm gets the judge-family warning; and
  `evalshift init --provider deepseek` scaffolds a DeepSeek project. DeepSeek's
  default thinking mode ignores `temperature`, so DeepSeek arms carry the
  report's non-determinism banner, and replayed assistant turns are sent with
  the placeholder `reasoning_content` DeepSeek requires on tool requests.

### Fixed

- `capture sync` priced calls recorded under a bare id at $0 when LiteLLM
  keys the model under both spellings but can only price the provider-prefixed
  one (DeepSeek). The price lookup now tries the provider-prefixed id first.
```

Merge the Fixed bullet into the existing `### Fixed` list rather than adding a second heading.

- [ ] **Step 8: Run the docs tests, then the gate**

Run: `uv run pytest tests/unit/test_docs_currency.py tests/unit/test_init.py -q`, then `make ci`
Expected: PASS, including `test_provider_scope_is_not_capped_at_three`, since the alphabetical lists don't match its regex.

- [ ] **Step 9: Commit**

```bash
git -C /home/lukas/repos/evalshift/evalshift-cli-wt-deepseek add tests/unit/test_docs_currency.py README.md DOCS.md llms-full.txt docs pyproject.toml CHANGELOG.md
git -C /home/lukas/repos/evalshift/evalshift-cli-wt-deepseek commit -m "docs: document DeepSeek and guard key/provider lists against the registry" -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task A7: Live verification with a real DeepSeek key (gate for the PR)

This task needs `DEEPSEEK_API_KEY`, which the maintainer supplies. It costs a few cents. **Do not open the PR until every check below passes.** If one fails, stop and report the exact error; do not work around it.

**Files:**
- Modify: `scripts/smoke_live_tools.py:74-77` (`MODELS`), `:105-113` (`_provider_or_skip`), plus a new multi-round check and a text-only chat-history check

- [ ] **Step 1: Make the smoke script provider-generic and add a multi-round check**

Replace `_provider_or_skip` so it reads keys from the registry and keeps DeepSeek fixtures out of `openai/`:

```python
def _provider_or_skip(model: str) -> str | None:
    """Return the fixture directory for ``model`` iff its provider's key is set."""
    meta = resolve_model(model)
    keys = PROVIDER_ENV_VARS.get(meta.provider, ())
    if not any(os.environ.get(k) for k in keys):
        return None
    # detect_provider names a response shape; DeepSeek shares OpenAI's, but its
    # live captures must not overwrite the OpenAI ones.
    return "deepseek" if meta.provider == "deepseek" else detect_provider(model)
```

Import `from evalshift_cli.models.registry import PROVIDER_ENV_VARS, resolve_model  # noqa: E402`. Set:

```python
MODELS = [
    "gemini/gemini-2.5-flash",
    "gemini/gemini-3.1-flash-lite-preview",
    "deepseek-flash",
    "deepseek-v4-pro",
]
```

Add this multi-round check, called once per model at the end of the model's loop body inside `main()`. It counts as a failure on exception:

```python
async def _multi_round(client: ModelClient, model: str) -> None:
    """Round 1 of a teacher-forced replay: an assistant turn DeepSeek never wrote."""
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Look up ACME's Q3 revenue."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_r0_0",
                    "type": "function",
                    "function": {"name": "search_db", "arguments": '{"query": "ACME Q3"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_r0_0", "content": '{"revenue_musd": 12.4}'},
    ]
    result = await client.complete_messages_with_tools(model=model, messages=messages, tools=TOOLS)
    print(f"  round 1: calls={result.trace.tool_names} text={bool(result.trace.final_text)}")
```

After it, also once per model and counted the same way (`  history FAILED: ...` on exception), add a text-only chat-history check. The backfill puts the placeholder on this assistant turn too, and there are no tools:

```python
async def _chat_history(client: ModelClient, model: str) -> None:
    """A text-only replayed history: an assistant turn with no reasoning_content."""
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Hi, I ordered a kettle last week."},
        {"role": "assistant", "content": "Thanks! How can I help with your kettle order?"},
        {"role": "user", "content": "What's your standard refund policy, in one sentence?"},
    ]
    result = await client.complete_messages(model=model, messages=messages)
    print(f"  history: text={bool(result.text)}")
```

- [ ] **Step 2: Run the smoke script**

Run: `DEEPSEEK_API_KEY=... uv run python scripts/smoke_live_tools.py` (the Gemini models skip when their key is unset)
Expected, for both `deepseek-flash` and `deepseek-v4-pro`:
- `single_tool` / `parallel` prompts print non-empty `calls:` and a cost above `$0.000000`.
- `text_only` prints no calls.
- `round 1:` prints with **no 400**. This is the live proof of F8's fix.
- `history: text=True` prints with no error. This proves the placeholder is harmless on a text-only request.
- New files appear under `tests/unit/fixtures/tool_responses/deepseek/*_live.json`.

- [ ] **Step 3: Check the 400 really is what the backfill prevents**

Run this once with the backfill disabled. Temporarily comment out the three inserted lines in `_dispatch_with_retry`, rerun only `deepseek-flash`, and confirm `round 1` FAILS with DeepSeek's "reasoning_content … must be passed back" 400. Then restore the lines and confirm it passes again.
Expected: FAIL without the backfill, PASS with it. If it passes without the backfill too, DeepSeek has relaxed the rule. Keep the backfill anyway (it's harmless), but record the observation in the PR description.

- [ ] **Step 4: End-to-end through the CLI**

Run:
```bash
cd /home/lukas/repos/evalshift/evalshift-cli-wt-deepseek
DEEPSEEK_API_KEY=... uv run evalshift test-call --model deepseek-flash --max-tokens 2048
cd examples/agent && DEEPSEEK_API_KEY=... GEMINI_API_KEY=... uv run evalshift compare --from deepseek-flash --to deepseek-v4-pro --yes
```
Expected:
- `test-call` prints a reply and a non-zero cost.
- `compare` finishes and writes a report.
- The report shows the non-determinism banner naming both DeepSeek arms.
- `report.json`'s tool-call scores are populated, not errored.
- `evalshift doctor` in the same shell shows `DEEPSEEK_API_KEY  set`.

Then check a DeepSeek judge end to end. Copy `examples/agent` to a scratch directory, add an `llm_judge` entry with `judge_model: deepseek-v4-pro` (the judge `init --provider deepseek` scaffolds), and rerun the same `compare`.
Expected:
- The judge's verdicts are populated in `report.json`, not errored or unmeasured.
- The non-determinism banner also names `deepseek/deepseek-v4-pro` as the judge. It must be listed once, even though it is also an arm.

- [ ] **Step 5: Commit the script and the live fixtures**

```bash
git -C /home/lukas/repos/evalshift/evalshift-cli-wt-deepseek add scripts/smoke_live_tools.py tests/unit/fixtures/tool_responses/deepseek
git -C /home/lukas/repos/evalshift/evalshift-cli-wt-deepseek commit -m "test(smoke): cover DeepSeek and a replayed tool round in the live smoke script" -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task A8: Open the CLI PR

- [ ] **Step 1: Push and open the PR**

```bash
git -C /home/lukas/repos/evalshift/evalshift-cli-wt-deepseek push -u origin feat/deepseek-provider
cd /home/lukas/repos/evalshift/evalshift-cli-wt-deepseek && gh pr create --title "feat: DeepSeek as a supported provider" --body "$(cat <<'EOF'
Adds DeepSeek (`deepseek-flash`, `deepseek-v4-pro`) as a first-class provider: registry + key pre-check + doctor row, tool-call parsing, judge-family detection, `init --provider deepseek`, correct pricing of bare recorded ids, and handling for DeepSeek's default thinking mode (non-determinism banner; placeholder `reasoning_content` on replayed tool rounds, which DeepSeek otherwise rejects with a 400).

Plan: `docs/superpowers/plans/2026-09-30-deepseek-provider.md`. Live-verified against the DeepSeek API (Task A7): <paste smoke + compare summary>.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Phase B: evalshift-sdk (docs only)

The SDK needs no code change. `wrap_openai` records whatever `model` the app sent, and Phase A makes that bare id replayable.

### Task B1: Document DeepSeek capture

**Files:** `README.md:66-68`, `DOCS.md:794`, `llms-full.txt:307-308, 501`, `CHANGELOG.md`

- [ ] **Step 1: Branch**

```bash
git -C /home/lukas/repos/evalshift/evalshift-sdk-wt-deepseek fetch origin
git -C /home/lukas/repos/evalshift/evalshift-sdk-wt-deepseek switch -c docs/deepseek origin/main
```

- [ ] **Step 2: Enumerate the sites**

Run: `git -C /home/lukas/repos/evalshift/evalshift-sdk-wt-deepseek grep -n "Groq" -- README.md DOCS.md llms-full.txt llms.txt docs`
Expected: `README.md:68`, `DOCS.md:794`, `llms-full.txt:307-308, 501`, and `docs/DECISIONS.md:506`, which is a historical decision record: leave it.

- [ ] **Step 3: Edit**

- `README.md:68` comment: `# OpenAI(base_url=...) covers DeepSeek, Ollama, vLLM, Groq, ...`
- `DOCS.md:794`: insert DeepSeek at the start of the list (`DeepSeek, Ollama, vLLM, llama.cpp server, …`). Append this sentence to the paragraph: ``For DeepSeek, `wrap_openai(OpenAI(base_url="https://api.deepseek.com", api_key=os.environ["DEEPSEEK_API_KEY"]))` records `model_id` as the bare id you pass (`deepseek-flash`); the CLI (1.2.0+) resolves it to `deepseek/deepseek-flash` on replay.``
- `llms-full.txt:307-308`: add `DeepSeek,` to the parenthesised list. `:501`: `# OpenAI(base_url=...) for DeepSeek / Ollama / vLLM / Groq ...`.

- [ ] **Step 4: CHANGELOG**

Add under `## [Unreleased]` → `### Changed` (create the heading if needed):

```markdown
- Docs: DeepSeek is listed among the OpenAI-compatible APIs `wrap_openai`
  captures unchanged, with the exact client construction. Replaying those
  captures needs EvalShift CLI 1.2.0 or later.
```

- [ ] **Step 5: Run the SDK gate, commit, push, open a PR**

Run (from `evalshift-sdk/`, mirroring `.github/workflows/ci.yml`): `uv run ruff check && uv run ruff format --check && uv run mypy && uv run pytest`
Expected: all green (docs-only change; this is a sanity run).

```bash
git -C /home/lukas/repos/evalshift/evalshift-sdk-wt-deepseek add README.md DOCS.md llms-full.txt CHANGELOG.md
git -C /home/lukas/repos/evalshift/evalshift-sdk-wt-deepseek commit -m "docs: capture DeepSeek through wrap_openai" -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
git -C /home/lukas/repos/evalshift/evalshift-sdk-wt-deepseek push -u origin docs/deepseek
```

Open the PR with `gh pr create` (body ends with the Claude Code line). **Merge it after the CLI 1.2.0 release**, because it names that version.

---

## Phase C: evalshift-action (docs + one pinning test)

`action.yml` has no provider inputs. Keys flow through the job `env:`, and `_secret_values` (`scripts/evalshift_action.py:826-834`) already redacts anything ending in `API_KEY`.

### Task C1: Provider key tables and a redaction test

**Files:** `README.md:137-160`, `DOCS.md:204-225`, `llms-full.txt:515-526`, `tests/test_evalshift_action.py`

- [ ] **Step 1: Branch**

```bash
git -C /home/lukas/repos/evalshift/evalshift-action-wt-deepseek fetch origin
git -C /home/lukas/repos/evalshift/evalshift-action-wt-deepseek switch -c docs/deepseek-key origin/main
```

- [ ] **Step 2: Write the pinning test**

Append to `tests/test_evalshift_action.py`:

```python
def test_provider_keys_are_redacted_including_deepseek() -> None:
    # Keys reach the CLI through the job env untouched; the log redactor
    # matches on the `_API_KEY` suffix, so a new provider needs no code change.
    env = {
        "ANTHROPIC_API_KEY": "sk-ant-secret",
        "DEEPSEEK_API_KEY": "sk-deepseek-secret",
        "HOME": "/home/runner",
    }
    assert sorted(action._secret_values(env)) == ["sk-ant-secret", "sk-deepseek-secret"]
```

Run (from `evalshift-action/`): `uv run pytest tests/test_evalshift_action.py -q -k deepseek`
Expected: PASS immediately. This test pins existing behaviour; it does not drive a change.

- [ ] **Step 3: Edit the three tables**

After the Google row in each file, add:
- `README.md` (`## Model provider API keys` table): ``| DeepSeek  | `DEEPSEEK_API_KEY`                    |``
- `DOCS.md:211-215`: the same row.
- `llms-full.txt:517-521`: `| DeepSeek | DEEPSEEK_API_KEY |`

Keep the column padding consistent with neighbouring rows. Then run `git -C /home/lukas/repos/evalshift/evalshift-action-wt-deepseek grep -n "GEMINI_API_KEY or GOOGLE_API_KEY"` and confirm all three tables were covered.

- [ ] **Step 4: Run the action's gate, commit, push, open the PR**

Run (mirroring `.github/workflows/ci.yml`): `uv run pytest && uv run ruff check .`

```bash
git -C /home/lukas/repos/evalshift/evalshift-action-wt-deepseek add README.md DOCS.md llms-full.txt tests/test_evalshift_action.py
git -C /home/lukas/repos/evalshift/evalshift-action-wt-deepseek commit -m "docs: list DEEPSEEK_API_KEY among provider keys" -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
git -C /home/lukas/repos/evalshift/evalshift-action-wt-deepseek push -u origin docs/deepseek-key
```

**Merge it after** the automatic CLI-pin PR has moved the action to CLI ≥ 1.2.0. Before that, the pinned CLI cannot run DeepSeek tool evals.

---

## Phase D: evalshift-client (landing + site docs)

Ship **after CLI 1.2.0 is on PyPI** (D4). Stack: Vite + React 19 + TS; tests are Vitest next to the source.

### Task D1: Landing copy

**Files:** `src/pages/landing/sections/Hero.tsx:112-114`, `src/pages/landing/Landing.test.tsx:253-255`

- [ ] **Step 1: Branch**

```bash
git -C /home/lukas/repos/evalshift/evalshift-client-wt-deepseek fetch origin
git -C /home/lukas/repos/evalshift/evalshift-client-wt-deepseek switch -c feat/deepseek-copy origin/main
```

- [ ] **Step 2: Update the test first**

`Landing.test.tsx:253-255`:

```tsx
    expect(
      screen.getByText(/Works with Anthropic, OpenAI, Google and DeepSeek models\./),
    ).toBeInTheDocument()
```

Run: `npx vitest run src/pages/landing/Landing.test.tsx`
Expected: FAIL, because the text is not found.

- [ ] **Step 3: Update the copy (D2)**

`Hero.tsx:113`: `Works with Anthropic, OpenAI, Google and DeepSeek models.`

Run: `npx vitest run src/pages/landing/Landing.test.tsx`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git -C /home/lukas/repos/evalshift/evalshift-client-wt-deepseek add src/pages/landing/sections/Hero.tsx src/pages/landing/Landing.test.tsx
git -C /home/lukas/repos/evalshift/evalshift-client-wt-deepseek commit -m "feat(landing): list DeepSeek among supported providers" -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task D2: Site docs pages

These are hand-written TSX mirrors of the CLI/SDK/action docs. No guard exists, so enumerate first.

**Files:** `src/pages/docs/pages/{CliCommands,Cli,Faq,Action,Configuration,GettingStarted,SdkAdapters}.tsx`

- [ ] **Step 1: Enumerate**

Run: `git -C /home/lukas/repos/evalshift/evalshift-client-wt-deepseek grep -nE "GEMINI_API_KEY|gemini\|openai\|anthropic|Claude, GPT|gpt-\*|\(<Code>anthropic</Code>|Ollama, vLLM" -- src/pages/docs`
Expected: the sites below. Leave `Changelog.tsx` alone (Task E1 handles it), and leave the tool/wire-shape pages (`SdkCapture.tsx`, `GoldenSuite.tsx`, `Agents.tsx`).

- [ ] **Step 2: Edit**

- `CliCommands.tsx:11`: `--provider gemini|openai|anthropic` → `--provider gemini|openai|anthropic|deepseek`
- `CliCommands.tsx:127-130` (`ENV_VARS`): add `{ name: "DEEPSEEK_API_KEY", detail: "DeepSeek auth" },` after the Anthropic entry.
- `Cli.tsx:221-222`: after `<Code>gpt-*</Code>/<Code>o1-*</Code>/<Code>o3-*</Code> → OpenAI`, add `, <Code>deepseek-*</Code> → DeepSeek` inside the parenthesis.
- `Faq.tsx:112`: `(Claude, GPT, Gemini)` → `(Claude, DeepSeek, Gemini, GPT)`. Also add a "Does EvalShift work with DeepSeek?" entry mirroring the CLI FAQ answer from Task A6 Step 6, in this file's existing Q/A component shape.
- `Action.tsx:183-185`: add `, <Code>DEEPSEEK_API_KEY</Code>` after `<Code>GEMINI_API_KEY</Code>/<Code>GOOGLE_API_KEY</Code>`.
- `Configuration.tsx:690-691`: `(<Code>anthropic</Code>, <Code>openai</Code>, <Code>google</Code>)` → `(<Code>anthropic</Code>, <Code>deepseek</Code>, <Code>google</Code>, <Code>openai</Code>)`.
- `GettingStarted.tsx:25`: `# or OPENAI_API_KEY / ANTHROPIC_API_KEY` → `# or OPENAI_API_KEY / ANTHROPIC_API_KEY / DEEPSEEK_API_KEY`. Keep the column alignment of the code block.
- `SdkAdapters.tsx:24`: `(Ollama, vLLM, Groq, OpenRouter, ...)` → `(DeepSeek, Ollama, vLLM, Groq, OpenRouter, ...)`. At `:205-206`, add DeepSeek at the start of the same list.

- [ ] **Step 3: Run the client gate**

Run: `npm run lint && npm run typecheck && npx vitest run && npm run build`
Expected: all green. Prerender succeeds for the docs routes.

- [ ] **Step 4: Commit, push, PR**

```bash
git -C /home/lukas/repos/evalshift/evalshift-client-wt-deepseek add src/pages/docs/pages
git -C /home/lukas/repos/evalshift/evalshift-client-wt-deepseek commit -m "docs(site): document DeepSeek keys, ids and init choice" -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
git -C /home/lukas/repos/evalshift/evalshift-client-wt-deepseek push -u origin feat/deepseek-copy
```

Open the PR with `gh pr create`. The body ends with the Claude Code line. Vercel deploys from `main` on merge.

---

## Phase E: Release and sync (maintainer-driven)

### Task E1: Release

Order: SDK → CLI → action pin PR → client llms sync.

- [ ] **Step 1: CLI 1.2.0.** After the A8 PR merges, cut the release commit the usual way (`chore(release): 1.2.0`). It bumps `pyproject.toml` and moves `## [Unreleased]` to `## [1.2.0]`. Tag it; the trusted-publishing workflow ships it.
- [ ] **Step 2: Action.** Wait for `bump-cli-pin.yml`'s daily PyPI poll (04:23 UTC) to open the pin PR, then merge it. Then merge the C1 PR.
- [ ] **Step 3: SDK.** Merge the B1 PR. No PyPI release is needed (D4).
- [ ] **Step 4: Client changelog.** In the D PR (or a follow-up), add the 1.2.0 headline to `src/pages/docs/pages/Changelog.tsx` and bump `src/lib/version.ts`. This follows the precedent of client commit `b1330d9` ("advertise CLI 1.1.0").

### Task E2: Refresh the site's llms mirrors

- [ ] **Step 1: Pull all three source repos to their merged `main`, then sync**

```bash
for r in evalshift-cli evalshift-sdk evalshift-action; do git -C /home/lukas/repos/evalshift/$r switch main && git -C /home/lukas/repos/evalshift/$r reset --hard origin/main; done
cd /home/lukas/repos/evalshift/evalshift-client-wt-deepseek && npm run sync:llms
grep -c DEEPSEEK_API_KEY public/cli-llms-full.txt public/ci-llms-full.txt
grep -c DeepSeek public/sdk-llms-full.txt
```

Expected: every count is ≥ 1.

- [ ] **Step 2: Commit and push to `main`** (the established flow for the llms sync)

```bash
git -C /home/lukas/repos/evalshift/evalshift-client-wt-deepseek add public/cli-llms-full.txt public/sdk-llms-full.txt public/ci-llms-full.txt
git -C /home/lukas/repos/evalshift/evalshift-client-wt-deepseek commit -m "docs: sync llms-full mirrors for DeepSeek support" -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

## Out of scope (follow-ups worth filing)

- **Keeping a capture's real `reasoning_content`.** Captures of DeepSeek apps carry the model's own reasoning on assistant turns. The CLI's `ChatMessage` drops it at promotion, so replayed *history* gets the placeholder too. Preserving it would change the suite schema.
- **`thinking` / `reasoning_effort` in the SDK's `GENERATION_KEYS`.** This would let a replay honour an app that turned thinking off. The allow-list is frozen public API ("widening this is a deliberate decision").
- **`top_p` below 0.95 is clamped in thinking mode.** This is a value constraint the dropped-params probe cannot see, like `temperature`'s reasoning-tier rejections.
- **A per-model `api_base` in `evalshift.yaml`** for EU/self-hosted endpoints. Today it works through LiteLLM's env vars (`DEEPSEEK_API_BASE`, `HOSTED_VLLM_API_BASE`, ...), which Task A6 documents.
- **Stale server README** (`evalshift-server/README.md:163-164` references a nonexistent `app/evaluations/runner.py`). Unrelated to providers; noted by the sweep.
