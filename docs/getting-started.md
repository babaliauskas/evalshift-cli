# Getting started

This page walks you from a fresh shell to a working `evalshift.yaml` in
under a minute.

## 1. Install

EvalShift is a Python 3.14+ package. We recommend [`uv`][uv] for dependency
management, but `pip` works too.

```bash
# uv (recommended)
uv pip install evalshift

# or pip
pip install evalshift
```

Verify the install:

```bash
evalshift --version
```

## 2. Set provider API keys

EvalShift calls Anthropic, OpenAI, and Google directly using your own keys.
Nothing is sent to any EvalShift-operated server — every API call is
client-side, and responses are cached locally in `~/.evalshift/cache.db`.

Set whichever providers you intend to use:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
export GOOGLE_API_KEY=...
```

## 3. Scaffold a starter project

```bash
mkdir my-eval
cd my-eval
evalshift init
```

This writes four files (a runnable customer-support agent project):

| File              | Purpose                                               |
| ----------------- | ----------------------------------------------------- |
| `evalshift.yaml`  | Run configuration (prompts, evaluators, slices).      |
| `prompts.py`      | Agent system prompt discovered by AST-walking.        |
| `tools.yaml`      | Six tool specs the agent can call.                    |
| `golden.jsonl`    | 40 example suite rows across 5 slices.                |

If any of those files already exist, `init` refuses to clobber them. Pass
`--force` to overwrite, or `--directory my-eval/` to scaffold into a
different folder.

## 4. Verify your environment

```bash
evalshift doctor
```

You'll see a short table:

* Green ✓ — check passes.
* Yellow ✗ — informational warning (e.g. an unset API key, or no
  `evalshift.yaml` here yet). Doctor still exits 0.
* Red ✗ — hard failure (e.g. an `evalshift.yaml` that doesn't validate).
  Doctor exits 1.

If everything is green or yellow, you're ready to run.

## 5. What's next

The `run`, `evaluate`, `analyze`, and `report` commands are still being
built — see `MVP_TODO.md` in the repo for the up-to-date status of the
build. Once they land you'll be able to do:

```bash
evalshift run --from claude-4.5-sonnet --to claude-5-sonnet
evalshift report <run-id>
```

[uv]: https://docs.astral.sh/uv/
