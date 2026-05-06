# Security policy

## Reporting a vulnerability

If you believe you've found a security vulnerability in AIMigrate,
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

AIMigrate is a local-first CLI; the threat model is centred on:

* **Untrusted project files.** `aimigrate.yaml`, `prompts.py`, and
  the suite JSONL come from the user's project. The Python-string
  parser AST-walks `prompts.py` rather than executing it; suite and
  config parsing reject unknown keys; AIMigrate never `eval`s user
  data.
* **Outbound API calls** go directly to the LLM provider you
  configured. AIMigrate does not phone home, send telemetry, or
  upload your prompts anywhere else.
* **The local SQLite cache** at `~/.aimigrate/cache.db` is the only
  persistent storage; nothing outside `~/.aimigrate/` and the project
  directory is touched.

Out of scope (for now): hosted backends, multi-tenant deployments,
any "AIMigrate cloud" — none of which exist in the MVP.
