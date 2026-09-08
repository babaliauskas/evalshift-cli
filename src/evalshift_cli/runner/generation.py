"""Translate a recorded generation_config into ModelClient dispatch kwargs.

The dict arrives verbatim from a capture (recorded by the evalshift-sdk under the
model_call event's ``metadata["generation_config"]`` and promoted onto
``SuiteExample.generation_config``). Only well-known keys are honoured; everything
else is ignored with a debug log — a bad or foreign recorded config must never
block a suite run.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

#: Keys this translator consumes; anything else is debug-logged and skipped.
_HANDLED_KEYS = frozenset(
    {"temperature", "response_format", "response_mime_type", "response_schema"}
)


def translate_generation_config(
    config: dict[str, Any] | None,
) -> tuple[float | None, dict[str, Any] | None]:
    """``(temperature_override, extra_kwargs)`` for ModelClient from a recorded config.

    * ``temperature`` (int/float, not bool) → temperature override.
    * a litellm-shaped ``response_format`` (dict) → passed through as-is.
    * else ``response_mime_type == "application/json"`` → ``response_format``:
      ``json_schema`` when a dict ``response_schema`` is present, else ``json_object``.
    * anything else is ignored (debug-logged) — never an error.
    """
    if not isinstance(config, dict) or not config:
        return None, None

    temperature: float | None = None
    raw_temp = config.get("temperature")
    if isinstance(raw_temp, (int, float)) and not isinstance(raw_temp, bool):
        temperature = float(raw_temp)

    extra: dict[str, Any] | None = None
    response_format = config.get("response_format")
    if isinstance(response_format, dict):
        extra = {"response_format": response_format}
    elif config.get("response_mime_type") == "application/json":
        schema = config.get("response_schema")
        if isinstance(schema, dict):
            extra = {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "captured_schema", "schema": schema},
                }
            }
        else:
            extra = {"response_format": {"type": "json_object"}}

    ignored = [k for k in config if k not in _HANDLED_KEYS]
    if ignored:
        log.debug("generation_config keys ignored at dispatch: %s", sorted(ignored))
    return temperature, extra


__all__ = ["translate_generation_config"]
