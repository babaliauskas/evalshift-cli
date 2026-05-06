# Configuration reference

Every AIMigrate run is driven by a single `aimigrate.yaml` file. This
page documents every field — types, defaults, and what they do.

`aimigrate init` writes a heavily-commented starter you can edit to
your needs. Below is the canonical reference.

## Top-level shape

```yaml
version: 1                # required, must be 1
prompts: [...]            # required, at least one
defaults: {...}           # optional
evaluators: {...}         # optional (but at least one is needed for `evaluate`)
slices: [...]             # optional
```

Unknown keys are rejected (`extra: forbid` everywhere) so typos fail
fast instead of silently dropping.

## `prompts`

A list of prompt definitions. Each entry has:

| Field         | Type    | Required                          | Description |
| ------------- | ------- | --------------------------------- | ----------- |
| `id`          | string  | yes                               | Stable identifier surfaced in reports. Must be unique within the file. |
| `detection`   | enum    | yes                               | `manual` or `python_string`. |
| `content`     | string  | when `detection: manual`          | Inline prompt body. Forbidden when `detection: python_string`. |
| `path`        | string  | when `detection: python_string`   | Relative or absolute path to a `.py` file. Resolved against the directory containing `aimigrate.yaml`. |
| `variable`    | string  | when `detection: python_string`   | Module-level variable name holding the prompt string. |
| `variables`   | list    | optional                          | Names of `{template}` placeholders the prompt expects. Used by the pre-flight compatibility check. |

### Two prompt-detection modes

* `manual` — write the prompt body inline:
  ```yaml
  - id: greet
    detection: manual
    content: "Hello {name}"
    variables: [name]
  ```
* `python_string` — point at an existing module-level string in your
  codebase:
  ```yaml
  - id: greet
    detection: python_string
    path: src/prompts/greet.py
    variable: GREET_PROMPT
    variables: [name]
  ```
  AIMigrate AST-walks the file and extracts the string literal. **It
  does not run user code.** F-strings, concatenations, `.format()`
  calls, and other dynamic forms are explicitly rejected.

## `defaults`

| Field           | Type   | Default                       | Description |
| --------------- | ------ | ----------------------------- | ----------- |
| `source_model`  | string | (none)                        | Default `--from` model id (or alias). |
| `target_model`  | string | (none)                        | Default `--to` model id (or alias). |
| `judge_model`   | string | `claude-5-sonnet-20260101`    | Default LLM-as-judge model. |
| `concurrency`   | int    | 10 (1 ≤ x ≤ 64)               | Max in-flight LLM calls during `aimigrate run`. |
| `cache`         | bool   | `true`                        | Read/write the local SQLite cache at `~/.aimigrate/cache.db`. |
| `max_cost_usd`  | float  | 50.0                          | Soft ceiling reserved for future enforcement. The pre-flight cost prompt currently triggers above $10 (skip with `--yes`). |

## `evaluators`

Three sub-keys, all optional. **At least one evaluator must be
configured for `aimigrate evaluate` to do anything.**

### `evaluators.structural`

A list. Each entry has a `type` and the fields that type needs.

| `type`        | Required fields                          | Behaviour |
| ------------- | ---------------------------------------- | --------- |
| `json_schema` | `schema_path` (string)                   | Each output is parsed as JSON; score 1.0 if it validates against the schema, 0.0 otherwise. |
| `regex`       | `pattern` (string)                       | Score 1.0 if the regex matches anywhere in the output, 0.0 otherwise. |
| `length`      | `min_chars` and/or `max_chars` (int)     | Score 1.0 inside the bounds, distance-decayed outside. |

Optional `applies_to: ["prompt-id-glob", ...]` (default `["*"]`) for
future per-prompt scoping.

### `evaluators.semantic`

A single object (not a list).

| Field             | Type   | Default                  | Description |
| ----------------- | ------ | ------------------------ | ----------- |
| `embedding_model` | string | `text-embedding-3-small` | LiteLLM-compatible embedding model id. Use a Gemini one (e.g. `gemini/text-embedding-004`) if you don't have an OpenAI key. |

The semantic evaluator scores the **target's similarity to the source**:
target_score = cosine(source, target), source_score = 1.0. A
negative `delta` means the target drifted from the source's meaning.

### `evaluators.llm_judge`

A list of pairwise judges. Each entry has:

| Field              | Type   | Required | Description |
| ------------------ | ------ | -------- | ----------- |
| `criterion_name`   | string | yes      | Short id surfaced in reports. |
| `criterion_prompt` | string | yes      | Free-form criterion the judge applies (e.g. "which output preserves more factual detail?"). |
| `judge_model`      | string | optional | Model used as the judge. |

The judge sees both outputs (with random A/B order to defang positional
bias) and produces strict-JSON `{"winner": "A"|"B"|"tie", "reason":
"..."}`. Target wins → `(0.0, 1.0)`; tie → `(0.5, 0.5)`; source wins →
`(1.0, 0.0)`. Malformed responses degrade to `(0.5, 0.5)` with the
error preserved.

## `slices`

A list of named subsets used for slice-level statistical analysis.
The implicit `"all"` slice always exists.

| Field         | Type   | Required | Description |
| ------------- | ------ | -------- | ----------- |
| `name`        | string | yes      | Slice name surfaced in reports. |
| `filter`      | string | yes      | A tag string. In MVP the filter is a literal tag — examples whose `tags` list contains the value land in this slice. |
| `applies_to`  | list   | optional | Glob list of prompt ids this slice applies to (default `["*"]`). |

## Suite (`golden.jsonl`) shape

The suite is JSON Lines — one example per non-blank line. Each row:

| Field      | Type    | Required | Description |
| ---------- | ------- | -------- | ----------- |
| `id`       | string  | yes      | Unique within the suite. |
| `inputs`   | object  | yes      | Mapping of template-variable name to value. |
| `tags`     | list    | optional | Slice tags. |
| `expected` | object  | optional | Reference output (unused by most evaluators). |

Unknown keys are rejected (typos fail fast).
