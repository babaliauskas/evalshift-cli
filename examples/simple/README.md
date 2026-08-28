# Simple example

A minimal EvalShift project with one prompt and one length evaluator.

## Local run

```bash
cd examples/simple
export GOOGLE_API_KEY=<google-api-key>

evalshift run --yes --from gemini-2.5-flash --to gemini-2.5-pro
RUN_ID=$(ls .evalshift/runs/ | head -1)
evalshift evaluate "$RUN_ID"
evalshift analyze "$RUN_ID"
evalshift report "$RUN_ID" --open
```

The config file is `evalshift.yaml`; the suite is `golden.jsonl`.

## Hosted push

```bash
evalshift login --token <hosted-api-token> --host <hosted-api-url>
evalshift push "$RUN_ID" --project acme/model-migration
```

Or add `project: acme/model-migration` to `evalshift.yaml` and run:

```bash
evalshift all --yes --push
```
