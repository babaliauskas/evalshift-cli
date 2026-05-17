# EvalShift

> Run your prompts on two LLMs and find out, with statistical confidence,
> what regressed.

EvalShift is a local-first CLI that helps engineering teams migrate
safely between LLM versions. Point it at your prompts and a golden
suite of inputs; it runs both models, scores the outputs with
structural / semantic / LLM-as-judge / tool-call evaluators, and
produces a single-file HTML report with **defensible statistics**:
paired tests, Cohen's d, 95% CIs, and Benjamini-Hochberg correction
across every comparison.

Hosted private-alpha commands are available when you explicitly log in
and push a run. Local runs remain local by default.

## The pipeline

The fast path is one command:

```
evalshift init        # scaffold a project
evalshift all         # doctor → run → evaluate → analyze → report
```

Or drive each stage by hand:

```
evalshift init        # scaffold a project
evalshift doctor      # verify env + config
evalshift run         # call both models on every example
evalshift evaluate    # score every (source, target) pair
evalshift analyze     # paired tests + BH correction
evalshift report      # single-file HTML report
```

Each stage writes its artefact under `.evalshift/runs/<run-id>/`:

| Stage      | Artefact         |
| ---------- | ---------------- |
| `run`      | `raw.jsonl`      |
| `evaluate` | `scores.jsonl`   |
| `analyze`  | `analysis.json`  |
| `report`   | `report.html` + `report.json` |

Hosted commands add optional sharing and CI workflows:

```
evalshift login       # store a hosted API token
evalshift whoami      # show hosted identity and org roles
evalshift bundle      # package a completed local run
evalshift push        # upload a bundle to hosted EvalShift
evalshift all --push  # run locally, then push
```

## Local-first by design

For local commands, your prompts and your suite never leave your machine.
The only outbound calls are to the LLM providers you configure (Anthropic,
OpenAI, Google) using your own API keys. `bundle` packages artifacts locally;
hosted upload happens only when you run `push` or `all --push`.

## Where to next

* [Getting started](getting-started.md) — a 60-second walkthrough.
* [Hosted alpha](hosted.md) — login, bundle, push, thresholds, privacy.
* [GitHub Action](github-action.md) — PR comments and hosted gates.
* [Configuration](configuration.md) — every `evalshift.yaml` field.
* [Evaluators](evaluators.md) — when to use each evaluator family.
* [Methodology](methodology.md) — the statistical machinery.
* [FAQ](faq.md) — common questions.
