"""Fixture for the non-literal-prompt rejection test.

The prompt is a computed f-string, which the EvalShift parser refuses to
evaluate at parse time for safety.
"""

NAME = "world"
GREET_PROMPT = f"Hello {NAME}"
