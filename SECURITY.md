# Security policy

## Reporting a vulnerability

If you believe you've found a security vulnerability in EvalShift,
please **do not open a public GitHub issue**. Instead, email the
maintainer directly at:

> **l.babaliauskas@gmail.com**

Include:

1. A short description of the issue and its impact.
2. Steps to reproduce.
3. Any proof-of-concept code, if applicable.

You should expect an acknowledgement within 72 hours and a fuller
response within 7 days. We'll work with you on a coordinated
disclosure timeline.

## Scope

EvalShift is a local-first CLI with optional hosted private-alpha upload
commands. The local threat model is centred on:

* **Untrusted project files.** `evalshift.yaml`, `prompts.py`, and
  the suite JSONL come from the user's project. The Python-string
  parser AST-walks `prompts.py` rather than executing it; suite and
  config parsing reject unknown keys; EvalShift never `eval`s user
  data.
* **Outbound API calls** go directly to the LLM provider you
  configured. Local runs do not phone home, send telemetry, or upload
  prompts to EvalShift-operated services.
* **The local SQLite cache** at `~/.evalshift/cache.db` is the only
  persistent local response cache; nothing outside `~/.evalshift/` and
  the project directory is touched for local runs.
* **Hosted credentials** are stored at `~/.evalshift/credentials` as
  JSON with owner-only `0600` permissions. CI should pass
  `EVALSHIFT_TOKEN` through secrets, not command-line logs.
* **Hosted uploads are opt-in.** `evalshift bundle` packages completed run
  artifacts locally into `run_bundle.json.gz`. `evalshift push` and
  `evalshift all --push` upload that bundle to the hosted backend configured
  by `--host` or `EVALSHIFT_HOST`.

Do not include hosted API tokens, provider API keys, OAuth codes, signed
upload URLs, or private repository details in bug reports or public issues.

Out of scope for this CLI security policy: the hosted backend's deployment
security controls, billing, enterprise SSO, and provider-key hosting.
