# Temperature Value Rejection — Runtime Adaptation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When a provider 400s the *value* of `temperature` (reasoning-tier models accept only their default), the client drops the parameter, redispatches, memoizes per model, and the run reports the model in the existing `non_deterministic_models` banner.

**Architecture:** One detection predicate + adaptation inside `ModelClient._dispatch_with_retry` (the single choke point under all four `complete*` methods). Two merge points carry the client's rejected-model set into `state.non_deterministic_models`: the orchestrator's final state write (run phase) and the evaluate command's state write (judge phase). No cache, config, or registry changes.

**Tech Stack:** Python 3.11+, litellm, pydantic, pytest (+pytest-asyncio), `mypy --strict`, ruff.

**Spec:** `docs/superpowers/specs/2026-08-23-temperature-value-rejection-design.md` — read it first.

## Global Constraints

- `mypy --strict` clean; ruff clean (`make ci` is the gate, and the pre-push hook).
- Tests first — watch each fail before implementing (TDD).
- Conventional Commits.
- Detection predicate must require ALL of: exception type name `BadRequestError`, message contains `temperature` (case-insensitive), `"temperature" in kwargs` at the call site. Anything less is not intercepted.
- Adaptation must NOT consume a retry attempt. `AuthError` short-circuit, backoff, and exhaustion behavior unchanged.
- Cache keys unchanged.
- Exception matching is by type NAME (string), matching the existing `_map_exception` idiom at `src/evalshift/models/client.py:678` — do not import litellm exception classes.

---

### Task 1: Client-level detection, adaptation, memoization

**Files:**
- Modify: `src/evalshift/models/client.py` (constructor ~:229, kwargs comment blocks ~:331 and ~:455, `_dispatch_with_retry` ~:470-529, helpers section)
- Modify: `src/evalshift/models/capabilities.py` (module docstring lines 20-23)
- Test: `tests/unit/test_model_client.py`

**Interfaces:**
- Consumes: existing `_patch_acompletion(monkeypatch, handler)` harness and `_FakeResponse` in `tests/unit/test_model_client.py`; existing `RetryPolicy`, `_map_exception`.
- Produces: `ModelClient.temperature_rejected_models` property → `frozenset[str]` of canonical model ids (Tasks 2 and 3 read it). Module-level helper `_is_temperature_value_rejection(exc: BaseException) -> bool`.

- [x] **Step 1: Write the failing tests**

Append to `tests/unit/test_model_client.py` (reuse the file's existing fakes/harness; `RateLimitError` import already exists at the top):

```python
# ---------------------------------------------------------------------------
# Temperature value rejection (reasoning-tier models)
# ---------------------------------------------------------------------------

# Name-based to match _map_exception's idiom: production raises
# litellm.BadRequestError; tests only need the type NAME to match.
_BadRequestError = type("BadRequestError", (Exception,), {})

_TEMP_400_MSG = (
    "OpenAIException - Unsupported value: 'temperature' does not support 0.0 "
    "with this model. Only the default (1) value is supported."
)


class TestTemperatureValueRejection:
    @pytest.mark.asyncio
    async def test_adapts_resends_without_temperature_and_records_model(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        calls: list[dict[str, Any]] = []

        def handler(**kwargs: Any) -> Any:
            calls.append(dict(kwargs))
            if "temperature" in kwargs:
                raise _BadRequestError(_TEMP_400_MSG)
            return _FakeResponse("adapted ok")

        _patch_acompletion(monkeypatch, handler)
        client = ModelClient(retry_policy=RetryPolicy(max_attempts=2))
        with caplog.at_level("WARNING"):
            result = await client.complete(model="gpt-4o", prompt="hi")

        assert result.text == "adapted ok"
        assert len(calls) == 2
        assert "temperature" in calls[0]
        assert "temperature" not in calls[1]
        assert client.temperature_rejected_models == frozenset({"openai/gpt-4o"})
        warnings = [r for r in caplog.records if "temperature" in r.getMessage()]
        assert len(warnings) == 1

    @pytest.mark.asyncio
    async def test_adaptation_does_not_consume_a_retry_attempt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # max_attempts=2. Sequence: temp-400 (adaptation), transient timeout
        # (attempt 1), success (attempt 2). Only passes if the adaptation
        # left the full retry budget intact.
        calls: list[dict[str, Any]] = []
        timeout_error = type("Timeout", (Exception,), {})

        def handler(**kwargs: Any) -> Any:
            calls.append(dict(kwargs))
            if "temperature" in kwargs:
                raise _BadRequestError(_TEMP_400_MSG)
            if len(calls) == 2:
                raise timeout_error("transient")
            return _FakeResponse("ok")

        _patch_acompletion(monkeypatch, handler)
        client = ModelClient(retry_policy=RetryPolicy(max_attempts=2))
        result = await client.complete(model="gpt-4o", prompt="hi")
        assert result.text == "ok"
        assert len(calls) == 3

    @pytest.mark.asyncio
    async def test_later_calls_omit_temperature_preemptively(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[dict[str, Any]] = []

        def handler(**kwargs: Any) -> Any:
            calls.append(dict(kwargs))
            if "temperature" in kwargs:
                raise _BadRequestError(_TEMP_400_MSG)
            return _FakeResponse("ok")

        _patch_acompletion(monkeypatch, handler)
        client = ModelClient()
        await client.complete(model="gpt-4o", prompt="first")
        await client.complete(model="gpt-4o", prompt="second")
        # first call: 2 dispatches (reject + adapted); second call: exactly 1.
        assert len(calls) == 3
        assert "temperature" not in calls[2]

    @pytest.mark.asyncio
    async def test_non_temperature_400_keeps_existing_retry_then_raise(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def handler(**kwargs: Any) -> Any:
            raise _BadRequestError("Unsupported value: 'tool_choice'")

        _patch_acompletion(monkeypatch, handler)
        client = ModelClient(retry_policy=RetryPolicy(max_attempts=2))
        with pytest.raises(ModelError):
            await client.complete(model="gpt-4o", prompt="hi")
        assert client.temperature_rejected_models == frozenset()

    @pytest.mark.asyncio
    async def test_temperature_400_without_temperature_in_kwargs_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A model that keeps 400ing about temperature even after the parameter
        # is gone must surface as ModelError, not loop. After the one
        # adaptation, kwargs no longer carry temperature, so the predicate's
        # kwargs leg fails and the normal path takes over.
        def handler(**kwargs: Any) -> Any:
            raise _BadRequestError(_TEMP_400_MSG)

        _patch_acompletion(monkeypatch, handler)
        client = ModelClient(retry_policy=RetryPolicy(max_attempts=2))
        with pytest.raises(ModelError):
            await client.complete(model="gpt-4o", prompt="hi")
        # The adaptation itself still fired once and recorded the model.
        assert client.temperature_rejected_models == frozenset({"openai/gpt-4o"})
```

Note: check how the file's existing async tests are decorated (`@pytest.mark.asyncio` vs asyncio_mode config) and match; adjust `_FakeResponse`/import names only if they differ from the harness shown at the top of the file.

- [x] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_model_client.py::TestTemperatureValueRejection -v
```

Expected: all 5 FAIL — `AttributeError: ... no attribute 'temperature_rejected_models'` (and/or `ModelError` from the unadapted 400).

- [x] **Step 3: Implement in `src/evalshift/models/client.py`**

Constructor (~:229) — add the set:

```python
    def __init__(self, *, retry_policy: RetryPolicy | None = None) -> None:
        self._retry = retry_policy or RetryPolicy()
        # Canonical ids of models that 400ed the VALUE of ``temperature``
        # (reasoning-tier models accept only their default). Once listed, a
        # model's calls omit the parameter entirely -- one failed call per
        # model per process. See _is_temperature_value_rejection.
        self._temperature_rejected: set[str] = set()
```

Property, right after `__init__`:

```python
    @property
    def temperature_rejected_models(self) -> frozenset[str]:
        """Canonical ids that rejected every non-default ``temperature`` value.

        Populated at dispatch time, from the provider's own 400. The
        orchestrator and the evaluate command merge this into the run
        state's ``non_deterministic_models`` so the report's banner covers
        runtime discoveries as well as the run-start capability probe.
        """
        return frozenset(self._temperature_rejected)
```

Module-level helper, in the Helpers section next to `_map_exception` (~:678):

```python
def _is_temperature_value_rejection(exc: BaseException) -> bool:
    """Report whether ``exc`` is a provider 400 rejecting ``temperature``'s value.

    Matched by type NAME like :func:`_map_exception`, so litellm's
    ``BadRequestError`` is caught without importing its class. The caller
    must additionally check that the outgoing kwargs actually carried
    ``temperature`` -- a temperature-flavoured 400 on a call that never sent
    the parameter is somebody else's bug and must surface.
    """
    return type(exc).__name__ == "BadRequestError" and "temperature" in str(exc).lower()
```

Rewrite `_dispatch_with_retry` (~:470-529). The `for` loop becomes a `while` so the adaptation can rewind the attempt counter; everything else keeps its exact semantics:

```python
    async def _dispatch_with_retry(
        self,
        canonical: str,
        kwargs: dict[str, Any],
        *,
        log_suffix: str,
    ) -> tuple[Any, int]:
        # (keep the existing docstring; append:)
        #
        # One adaptation is layered on top of the retry policy: a provider
        # 400 that rejects the VALUE of ``temperature`` (reasoning-tier
        # models accept only their default) pops the parameter and
        # redispatches immediately, without consuming a retry attempt. The
        # model id is memoized on the client so later calls omit the
        # parameter before dispatch.
        if canonical in self._temperature_rejected:
            kwargs.pop("temperature", None)
        last_exc: Exception | None = None
        attempt = 0
        while True:
            attempt += 1
            start = time.perf_counter()
            try:
                response = await litellm.acompletion(**kwargs)
            except Exception as exc:
                if "temperature" in kwargs and _is_temperature_value_rejection(exc):
                    kwargs.pop("temperature")
                    if canonical not in self._temperature_rejected:
                        self._temperature_rejected.add(canonical)
                        log.warning(
                            "model %s rejects non-default temperature values; "
                            "resending without temperature — sampling for this "
                            "model is not controlled and outputs are "
                            "non-deterministic",
                            canonical,
                        )
                    attempt -= 1  # adaptation, not a retry
                    continue
                mapped = _map_exception(exc)
                # Auth errors are deterministic; don't waste retries on them.
                if isinstance(mapped, AuthError):
                    raise mapped from exc
                last_exc = mapped
                if attempt >= self._retry.max_attempts:
                    raise mapped from exc
                delay = self._retry.delay(attempt)
                log.warning(
                    "model %s%s attempt %d failed (%s); retrying in %.2fs",
                    canonical,
                    log_suffix,
                    attempt,
                    mapped.__class__.__name__,
                    delay,
                )
                await asyncio.sleep(delay)
                continue
            latency_ms = int((time.perf_counter() - start) * 1000)
            return response, latency_ms
```

Delete the now-unreachable trailing `raise ModelError(f"exhausted retries...")` and the `# Loop exits cleanly...` comment (the `while True` only exits via `return`/`raise`; keep `last_exc` only if something still reads it — if nothing does, remove it too).

Termination argument (for the reviewer): the adaptation branch requires `"temperature" in kwargs` and unconditionally pops it, so it fires at most once per call — no infinite loop.

- [x] **Step 4: Fix the two false comments**

`client.py` kwargs blocks (~:331 and ~:455) — replace the four-line comment above `"drop_params": True` in BOTH places with:

```python
            # drop_params only saves models LiteLLM's configs special-case
            # (o-series names). Other reasoning-tier models reject
            # temperature != 1 at the API with a 400; that case is handled
            # at dispatch — see _is_temperature_value_rejection and the
            # adaptation in _dispatch_with_retry.
            "drop_params": True,
```

`capabilities.py` module docstring (lines 20-23) — replace the paragraph
"Note this detects *withdrawal*, not value constraints. ... already handled by ``drop_params`` in :mod:`evalshift.models.client`." with:

```
Note this detects *withdrawal*, not value constraints. Reasoning-tier models
such as ``gpt-5.6-terra`` advertise ``temperature`` while rejecting every
value except their default; ``drop_params`` does not cover them (LiteLLM
special-cases only o-series names). That case is detected from the
provider's own 400 at dispatch time and adapted per model — see
``ModelClient._dispatch_with_retry`` in :mod:`evalshift.models.client`.
```

- [x] **Step 5: Run the tests**

```bash
pytest tests/unit/test_model_client.py -v
```

Expected: new class 5/5 PASS, every pre-existing test in the file still PASS (retry, auth short-circuit, mapping tests prove the unchanged semantics).

- [x] **Step 6: Static checks**

```bash
ruff check src/evalshift/models/ tests/unit/test_model_client.py && mypy --strict src/evalshift
```

Expected: clean.

- [x] **Step 7: Commit**

```bash
git add src/evalshift/models/client.py src/evalshift/models/capabilities.py tests/unit/test_model_client.py
git commit -m "fix(client): adapt to models that reject temperature values

Reasoning-tier models (gpt-5.6-terra and kin) advertise temperature but
400 every value except the default; drop_params only saves o-series
names. Detect the provider's 400 at dispatch, resend without the
parameter, and memoize per model so later calls skip it preemptively.
The adaptation does not consume a retry attempt. Also corrects the two
comments that claimed drop_params already covered this."
```

---

### Task 2: Orchestrator merges runtime rejections into run state

**Files:**
- Modify: `src/evalshift/runner/orchestrator.py` (`_process_work` final-state block, ~:814-818)
- Test: `tests/unit/test_orchestrator.py`

**Interfaces:**
- Consumes: `ModelClient.temperature_rejected_models` (Task 1); `_process_work`'s existing `client: ModelClient` parameter and `state` object; `touch_checkpoint`; `RunState.non_deterministic_models` (`src/evalshift/runner/models.py:159`).
- Produces: final `state.json` whose `non_deterministic_models` is the probe result plus runtime discoveries, deduplicated, probe-order first then sorted runtime additions.

- [x] **Step 1: Write the failing tests**

Append to `tests/unit/test_orchestrator.py`, using the file's existing fixtures (`cache`, `_config`, `_suite`, `_writeable_paths`, `_make_fake_client`) and its existing state-reading idiom (the file already imports the checkpoint module — reuse its import name for `read_state`):

```python
class TestRuntimeTemperatureRejectionReporting:
    @pytest.mark.asyncio
    async def test_rejected_models_land_in_final_state(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        cache: CacheStore,
    ) -> None:
        _make_fake_client(monkeypatch)
        config_path, suite_path, runs_base = _writeable_paths(tmp_path)
        client = ModelClient()
        # White-box seed: simulates a mid-run provider rejection without
        # needing a live 400 through the faked complete().
        client._temperature_rejected.add("openai/gpt-5.6-terra")
        result = await run_orchestrator(
            config=_config(),
            suite=_suite(),
            config_path=config_path,
            suite_path=suite_path,
            runs_base=runs_base,
            source="gpt-4o",
            target="gpt-4o-mini",
            yes=True,
            client=client,
            cache=cache,
        )
        state = read_state(result.run_dir)
        assert "openai/gpt-5.6-terra" in state.non_deterministic_models

    @pytest.mark.asyncio
    async def test_merge_deduplicates_against_probe_result(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        cache: CacheStore,
    ) -> None:
        _make_fake_client(monkeypatch)
        monkeypatch.setattr(
            orchestrator_module,
            "detect_non_deterministic_models",
            lambda *, source, target: ["openai/gpt-5.6-terra"],
        )
        config_path, suite_path, runs_base = _writeable_paths(tmp_path)
        client = ModelClient()
        client._temperature_rejected.add("openai/gpt-5.6-terra")
        result = await run_orchestrator(
            config=_config(),
            suite=_suite(),
            config_path=config_path,
            suite_path=suite_path,
            runs_base=runs_base,
            source="gpt-4o",
            target="gpt-4o-mini",
            yes=True,
            client=client,
            cache=cache,
        )
        state = read_state(result.run_dir)
        assert state.non_deterministic_models.count("openai/gpt-5.6-terra") == 1
```

Match the surrounding tests for the exact `run_orchestrator(...)` keyword set — copy the call from `test_single_run_writes_state_and_raw` (~:156) and change only what these tests need. Import `ModelClient` and the orchestrator module alias if the file lacks them.

- [x] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_orchestrator.py -k RuntimeTemperatureRejection -v
```

Expected: FAIL — `"openai/gpt-5.6-terra" not in []` (first test); dedup test fails the same way until the merge exists.

- [x] **Step 3: Implement the merge**

In `_process_work` (`src/evalshift/runner/orchestrator.py` ~:814-818), replace:

```python
    # Final checkpoint + status flip.
    final_state = touch_checkpoint(state, completed).model_copy(
        update={"status": "completed"},
    )
```

with:

```python
    # Final checkpoint + status flip. Runtime-discovered temperature
    # rejections join the probe-detected list here so the report's
    # non-determinism banner covers both. Probe entries keep their order;
    # runtime additions follow, sorted, deduplicated.
    runtime_nondet = sorted(
        set(client.temperature_rejected_models) - set(state.non_deterministic_models)
    )
    final_state = touch_checkpoint(state, completed).model_copy(
        update={
            "status": "completed",
            "non_deterministic_models": [
                *state.non_deterministic_models,
                *runtime_nondet,
            ],
        },
    )
```

(Resume caveat, acceptable and documented by this comment: a resumed run re-discovers rejections on its live calls; a fully-cached resume makes no live calls and adds nothing new — the prior final state already carried them.)

- [x] **Step 4: Run tests**

```bash
pytest tests/unit/test_orchestrator.py -v
```

Expected: both new tests PASS, all existing orchestrator tests PASS.

- [x] **Step 5: Commit**

```bash
git add src/evalshift/runner/orchestrator.py tests/unit/test_orchestrator.py
git commit -m "feat(runner): report runtime temperature rejections in run state

Models the client discovered mid-run join non_deterministic_models at
the final state write, deduplicated against the run-start probe, so the
report banner and JSON cover both detection paths."
```

---

### Task 3: Evaluate command shares one judge client and merges its discoveries

**Files:**
- Modify: `src/evalshift/cli/commands/evaluate.py` (caller ~:167, `_build_evaluators` ~:283-334, `write_state` ~:207)
- Test: `tests/unit/test_evaluate_command.py`

**Interfaces:**
- Consumes: `ModelClient.temperature_rejected_models` (Task 1); `PairwiseJudgeEvaluator(..., client=...)` (existing parameter, `src/evalshift/evaluators/llm_judge.py:65`).
- Produces: `_build_evaluators(cfg, project_root, judge_client: ModelClient)` — new required keyword; the state written after scoring carries judge-phase rejections.

- [x] **Step 1: Write the failing tests**

Add to `tests/unit/test_evaluate_command.py` (follow the file's existing config-building idiom for a cfg with one `llm_judge` entry; adapt fixture names to what the file already uses):

```python
def test_build_evaluators_threads_shared_judge_client(tmp_path: Path) -> None:
    cfg = _cfg_with_judge()  # file-local helper; build EvalShiftConfig with one llm_judge entry
    judge_client = ModelClient()
    evaluators = _build_evaluators(cfg, tmp_path, judge_client=judge_client)
    judges = [e for e in evaluators if isinstance(e, PairwiseJudgeEvaluator)]
    assert judges, "config should have produced a judge"
    assert all(j._client is judge_client for j in judges)
```

And the merge behavior at the state write — if the file has a command-level harness that runs scoring end-to-end, add an assertion there that a pre-seeded `judge_client._temperature_rejected` entry appears in the re-read state's `non_deterministic_models`; if no such harness exists, test the merge expression through the command function with mocked `_score_everything` following the file's established mocking pattern. The assertion that matters:

```python
    state_after = read_state(run_dir)
    assert "openai/gpt-5.6-terra" in state_after.non_deterministic_models
    assert state_after.evaluator_coverage == coverage  # both updates land in ONE write
```

- [x] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_evaluate_command.py -k "judge_client or temperature" -v
```

Expected: FAIL — `_build_evaluators() got an unexpected keyword argument 'judge_client'`.

- [x] **Step 3: Implement**

`src/evalshift/cli/commands/evaluate.py`:

Add import (top of file, with the other model imports):

```python
from evalshift.models.client import ModelClient
```

Caller (~:167):

```python
    # One client shared by every judge so temperature rejections discovered
    # while judging are collected in one place and merged into state below.
    judge_client = ModelClient()
    evaluators = _build_evaluators(cfg, project_root, judge_client=judge_client)
```

`_build_evaluators` signature (~:283):

```python
def _build_evaluators(
    cfg: EvalShiftConfig, project_root: Path, *, judge_client: ModelClient
) -> list[Evaluator]:
```

Judge construction (~:326-333):

```python
    for j in cfg.evaluators.llm_judge:
        _add(
            PairwiseJudgeEvaluator(
                criterion_name=j.criterion_name,
                criterion_prompt=j.criterion_prompt,
                judge_model=j.judge_model,
                client=judge_client,
            ),
            blocking=j.blocking,
        )
```

State write (~:207) — replace:

```python
    write_state(run_dir, state.model_copy(update={"evaluator_coverage": coverage}))
```

with:

```python
    # Judge calls can discover temperature-rejecting models after the run
    # phase already wrote its state; merge them here so the report banner
    # covers the judge model too.
    runtime_nondet = sorted(
        set(judge_client.temperature_rejected_models)
        - set(state.non_deterministic_models)
    )
    write_state(
        run_dir,
        state.model_copy(
            update={
                "evaluator_coverage": coverage,
                "non_deterministic_models": [
                    *state.non_deterministic_models,
                    *runtime_nondet,
                ],
            },
        ),
    )
```

Fix every other `_build_evaluators(` call site the same way (grep for it; `evalshift all` may reach it through this module only — verify).

- [x] **Step 4: Run tests**

```bash
pytest tests/unit/test_evaluate_command.py tests/unit/test_all_command.py -v
```

Expected: new tests PASS; existing evaluate/all command tests PASS (they exercise `_build_evaluators` and will catch a missed call site).

- [x] **Step 5: Commit**

```bash
git add src/evalshift/cli/commands/evaluate.py tests/unit/test_evaluate_command.py
git commit -m "feat(evaluate): share one judge client and report its rejections

All PairwiseJudgeEvaluators now dispatch through a single ModelClient;
temperature rejections it discovers merge into non_deterministic_models
alongside evaluator_coverage in the existing post-scoring state write."
```

---

### Task 4: Judge integration test — verdict instead of EvaluatorError

**Files:**
- Test: `tests/unit/test_evaluators.py` (judge section, near the `PairwiseJudgeEvaluator` tests ~:616)

**Interfaces:**
- Consumes: Task 1's adaptation through the real `ModelClient`; the file's existing judge test idiom (`criterion_name`/`criterion_prompt`, `score(...)` signature at ~:625-631) and `rng` seam.

- [x] **Step 1: Write the test (should pass immediately — it locks the end-to-end behavior)**

```python
    @pytest.mark.asyncio
    async def test_judge_survives_temperature_value_rejection(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # End-to-end through the real ModelClient: first dispatch 400s the
        # temperature value, the client adapts, the judge gets its verdict.
        # Before the adaptation existed this raised EvaluatorError and a
        # blocking judge poisoned the gate.
        bad_request = type("BadRequestError", (Exception,), {})
        calls: list[dict[str, Any]] = []

        async def fake_acompletion(**kwargs: Any) -> Any:
            calls.append(dict(kwargs))
            if "temperature" in kwargs:
                raise bad_request(
                    "'temperature' does not support 0.0 with this model."
                )
            return _fake_llm_response('{"winner": "A"}')  # reuse/adapt the file's response fake

        monkeypatch.setattr(client_module.litellm, "acompletion", fake_acompletion)
        monkeypatch.setattr(
            client_module.litellm, "completion_cost", lambda **_: 0.0
        )
        rng = random.Random(0)
        rng.random = lambda: 0.0  # type: ignore[method-assign]  # target shown as A
        ev = PairwiseJudgeEvaluator(
            criterion_name="equivalence",
            criterion_prompt="Which is better?",
            judge_model="gpt-4o",
            client=ModelClient(),
            rng=rng,
        )
        score = await ev.score(
            prompt_id="p",
            example_id="e",
            input_vars={},
            source_output="src",
            target_output="tgt",
        )
        assert score.target_score == 1.0
        assert len(calls) == 2 and "temperature" not in calls[1]
```

Reuse the file's existing fake-response helper for the judge (`_client_returning` internals show the shape); import `client as client_module` from `evalshift.models` and `random` if not present. `completion_cost` signature: match how `tests/unit/test_model_client.py` patches it (`lambda completion_response=None, **_: 0.0`).

- [x] **Step 2: Run it**

```bash
pytest tests/unit/test_evaluators.py -k temperature -v
```

Expected: PASS. If it fails, Task 1 has a bug — fix there, not here.

- [x] **Step 3: Commit**

```bash
git add tests/unit/test_evaluators.py
git commit -m "test(judge): lock verdict delivery through temperature adaptation"
```

---

### Task 5: Documentation

**Files:**
- Modify: `DOCS.md:587` (Sampling control paragraph)
- Modify: `CHANGELOG.md` (Unreleased)
- Modify: `llms-full.txt:467` area (determinism text)

**Interfaces:** none — prose only. Check whether `llms-full.txt` is generated (`grep -rn "llms-full" Makefile scripts/`); if generated, update the source and regenerate instead of editing the copy.

- [x] **Step 1: DOCS.md** — extend the Sampling-control paragraph (`DOCS.md:587`). After the sentence ending "…the report shows a banner above the verdict plus a methodology note.", insert:

```markdown
A second failure mode is caught at call time rather than run start: reasoning-tier models (for example `gpt-5.6-terra`) advertise `temperature` but reject every value except their default with a 400. The first such rejection makes EvalShift resend the call without `temperature` and stop sending it to that model for the rest of the process; the model joins `non_deterministic_models` and the same banner. One call per affected model fails and is retried adapted — nothing is lost, but sampling for that model is provider-default, not controlled.
```

- [x] **Step 2: CHANGELOG.md** — under `## [Unreleased]`, add:

```markdown
### Fixed

- Models that reject non-default `temperature` values (reasoning-tier models
  such as `gpt-5.6-terra`) no longer fail every call. The client detects the
  provider's 400, resends without `temperature`, memoizes the model for the
  rest of the process, and reports it in `non_deterministic_models` alongside
  probe-detected models. Previously a `blocking: true` judge on such a model
  errored on every example after burning the full retry budget per call.
```

- [x] **Step 3: llms-full.txt** — update the determinism passage (~:467) with the same two-mode story (withdrawal probed at run start; value rejection adapted at call time), matching the file's prevailing terseness. If generated, regenerate.

- [x] **Step 4: Commit**

```bash
git add DOCS.md CHANGELOG.md llms-full.txt
git commit -m "docs: document temperature value-rejection adaptation"
```

---

### Task 6: Full verification gate

- [x] **Step 1: Run the CI mirror**

```bash
make ci
```

Expected: ruff, `mypy --strict`, and the full pytest suite all green. Fix anything red before claiming done (superpowers:verification-before-completion).

- [x] **Step 2: Reproduce the original failure shape once, manually** — optional sanity: `evalshift test-call` (or the judge path) against a faked reject is already covered by Task 4; live verification happens in the personalButler CI once a release ships.
