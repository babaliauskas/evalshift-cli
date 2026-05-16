# Developer convenience targets for evalshift-cli.

SERVER_SCHEMA ?= ../evalshift-server/schemas/bundle_manifest.schema.json
CLI_SCHEMA := src/evalshift/hosted/bundle_manifest.schema.json

.PHONY: help test lint format typecheck sync-schema check-schema

help:
	@echo "Targets:"
	@echo "  test           Run pytest"
	@echo "  lint           Ruff check + format-check + mypy strict"
	@echo "  format         Ruff format"
	@echo "  typecheck      mypy strict"
	@echo "  sync-schema    Copy the bundle JSON schema from evalshift-server"
	@echo "  check-schema   Fail if the CLI schema differs from the server export"
	@echo
	@echo "Override SERVER_SCHEMA=path/to/bundle_manifest.schema.json if the"
	@echo "server repo lives somewhere other than ../evalshift-server."

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy --strict src/evalshift

format:
	uv run ruff format .

typecheck:
	uv run mypy --strict src/evalshift

sync-schema:
	@test -f "$(SERVER_SCHEMA)" || (echo "server schema not found at $(SERVER_SCHEMA); set SERVER_SCHEMA=..." && exit 1)
	cp "$(SERVER_SCHEMA)" "$(CLI_SCHEMA)"
	@echo "synced $(CLI_SCHEMA) from $(SERVER_SCHEMA)"

check-schema:
	@test -f "$(SERVER_SCHEMA)" || (echo "server schema not found at $(SERVER_SCHEMA); set SERVER_SCHEMA=..." && exit 1)
	@diff -q "$(SERVER_SCHEMA)" "$(CLI_SCHEMA)" >/dev/null || ( \
	    echo "✗ CLI bundle schema is out of sync with the server export."; \
	    echo "  Run 'make sync-schema' (or override SERVER_SCHEMA=...) and commit the diff."; \
	    exit 1 )
	@echo "✓ CLI schema matches server export"
