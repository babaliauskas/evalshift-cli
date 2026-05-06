# Evaluators

An evaluator scores the (source_output, target_output) pair for one
example. Every evaluator returns a `PairedScore` with both halves and
a `delta = target_score - source_score`. Negative deltas mean the
target regressed; positive deltas mean it improved.

The MVP ships three families:

## Structural (deterministic, free)

Fast checks that depend only on the output shape — no API calls.

* **`json_schema`** — does the output parse as JSON and validate
  against a schema you provide? 1.0 / 0.0.
* **`regex`** — does the output match a pattern? 1.0 / 0.0.
* **`length`** — is the output within `[min_chars, max_chars]`?
  1.0 inside, distance-decayed outside.

Use these whenever you can — they're free, deterministic, and cover
most real regressions.

## Semantic (cheap)

* **`semantic.cosine`** — embed both outputs with a configurable
  embedding model, compute cosine similarity, then frame the result
  as a target-preservation score: source = 1.0, target = similarity.
  Delta < 0 means the target drifted in meaning from the source.

Use when:
* You don't have a clean structural check.
* You want to detect "wandered off" outputs that still look fine
  syntactically.

Don't use when:
* The target is intentionally meant to differ from the source (e.g.
  you're migrating from a verbose model to a terse one). The
  similarity will look low and you'll get a confusing "regression"
  signal.

## LLM-as-judge (most expensive)

* **`llm_judge.<criterion>`** — ask a strong model "which output
  better satisfies this criterion?" with random A/B ordering to
  reduce positional bias. Verdict maps cleanly to (source, target)
  scores.

Use when you can articulate the difference you care about as a
sentence ("which output preserves more factual detail?"). Multiple
`llm_judge` entries are allowed — each becomes its own evaluator.

## Mixing evaluators

You can configure several at once. They all run on every (prompt,
example) pair, and the analysis layer treats each as a separate
comparison (so BH correction adjusts for the multiple-test count
correctly).

A typical migration uses:
* 1–2 structural evaluators (cheap baseline checks)
* 1 `semantic.cosine` (catches semantic drift)
* 1 `llm_judge` per criterion the team cares about

## Cost considerations

Per (prompt, example) pair, each evaluator means:

| Evaluator     | Cost                                    |
| ------------- | --------------------------------------- |
| structural.*  | $0 (no calls)                           |
| semantic      | 2 embedding calls                       |
| llm_judge     | 1 judge model completion                |

A 100-example suite with 1 prompt and 4 evaluators (2 structural +
1 semantic + 1 judge) is:

* Run: 200 model calls (100 × 2 models)
* Evaluate: 200 embedding calls + 100 judge calls

LiteLLM's pricing data drives the pre-flight estimate; the local
SQLite cache absorbs identical re-runs.
