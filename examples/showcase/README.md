# EvalShift showcase

Four small, runnable scenarios that demonstrate what EvalShift catches.
Use these to demo, build intuition, or sanity-check a release.

## Run them all

```bash
# Live (uses your API keys; default):
./scripts/run_showcase.sh

# Offline (deterministic, no keys, replays canned fixtures):
./scripts/run_showcase.sh --offline

# One scenario only:
./scripts/run_showcase.sh --only fail-dropped-tool

# Open the last report at the end:
./scripts/run_showcase.sh --offline --open
```

Each scenario produces a full HTML report at
`examples/showcase/<name>/.evalshift/runs/<run-id>/report.html`.

## Scenarios

| name | what it shows |
|---|---|
| `pass-clean` | Source and target both behave correctly. Green report end-to-end. |
| `fail-dropped-tool` | Target drops `notify_security_team` on security queries — high-severity regression. |
| `fail-argument-drift` | Same tool called, but the refund `amount_usd` is halved. The `tool_arguments` evaluator (numeric strategy, 5% tolerance) catches it. |
| `mix` | Parallel tool calls + refusals + structural (length, regex) + tool evaluators in one run. |

## Live vs offline — what's reliable

- **Offline mode** is fully deterministic. The fixtures shipped in each
  scenario engineer specific source/target behaviour; the report will
  always look the same.
- **Live mode** depends on real model behaviour:
  - `pass-clean` and `mix` are stable on top-tier models.
  - `fail-argument-drift` engineers the drift through the prompt itself
    (which tells the model to cap refunds at 50%); both source and
    target will likely drift in live mode, so you'll see absolute
    failure on `amount_usd` rather than a clean source-vs-target delta.
    The offline path shows the clean delta.
  - `fail-dropped-tool` uses a deliberately weakened prompt. Smaller
    target models more readily skip the security tool, but there's some
    jitter — repeat runs may differ.

If you need a deterministic demo, use `--offline`. If you want to see
real model behaviour, use the live path.

## Refreshing fixtures

If you edit a prompt or add an example, regenerate the fixtures from a
live run:

```bash
cd examples/showcase/pass-clean
evalshift run --yes
RUN_ID=$(ls -t .evalshift/runs | head -1)
python ../../../scripts/capture_fixtures.py "$RUN_ID" \
    --suite golden.jsonl --out fixtures.jsonl
```

The capture script extracts the per-call `model_id` and a substring
match (default: each example's `query` input) from the run's
`raw.jsonl` and writes one fixture per call.

## Adding a new scenario

1. Create `examples/showcase/<name>/` with `evalshift.yaml`,
   `golden.jsonl`, and `fixtures.jsonl`.
2. Reference shared assets via relative paths:
   `path: ../shared/prompts/<name>.py` and
   `tools_path: ../shared/tools.yaml`.
3. Set `defaults.cache: false` so `--offline` always hits the replay
   client.
4. Add the scenario to `ALL_SCENARIOS` in `scripts/run_showcase.sh`.
