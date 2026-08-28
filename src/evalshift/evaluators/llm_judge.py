"""LLM-as-judge evaluator: pairwise comparison via a prompted model."""

from __future__ import annotations

import json
import logging
import random
import re
from typing import Any

from evalshift.cache.store import CacheStore, cache_key
from evalshift.evaluators.base import EvaluatorError, PairedScore
from evalshift.models.client import ModelClient

log = logging.getLogger(__name__)


#: Cap on the rendered transcript's length (characters). Keeps multi-turn
#: judge prompts bounded regardless of how long the recorded conversation is.
MAX_TRANSCRIPT_CHARS = 4000

_TRUNCATION_MARKER = "[... truncated ...]"


JUDGE_PROMPT_TEMPLATE = """\
You are an impartial judge comparing two AI assistant outputs against a criterion.

{context_section}Criterion: {criterion_prompt}

Output A:
\"\"\"
{output_a}
\"\"\"

Output B:
\"\"\"
{output_b}
\"\"\"

Decide which output better satisfies the criterion. Reply with strict JSON only,
no prose, on a single line:
{{"winner": "A" | "B" | "tie", "reason": "<one short sentence>"}}
"""


class PairwiseJudgeEvaluator:
    """Ask a model which of two outputs better satisfies a criterion.

    Order randomization (target sometimes shown as A, sometimes as B)
    mitigates positional bias common in LLM judges. The mapping back to
    target-vs-source happens after the verdict so the caller never sees
    the randomization.
    """

    #: Stable evaluator-type slug the analysis layer selects policy rows
    #: on — never the user-chosen ``name``, which is free to change.
    kind = "llm_judge"

    def __init__(
        self,
        *,
        criterion_name: str,
        criterion_prompt: str,
        judge_model: str = "gemini/gemini-3.1-flash-lite-preview",
        client: ModelClient | None = None,
        rng: random.Random | None = None,
        name: str | None = None,
        cache: CacheStore | None = None,
    ) -> None:
        """Build the evaluator.

        Args:
            criterion_name: Short id for the criterion, used in the
                evaluator name and the cache key.
            criterion_prompt: The criterion shown to the judge.
            judge_model: Model that renders the verdict.
            client: Override the model client (tests).
            rng: Override the A/B randomization source (tests).
            name: Evaluator name; defaults to ``llm_judge.<criterion>``.
            cache: Optional store for verdict reuse. Judge calls are the
                slowest part of scoring and are deterministic given the
                criterion and the two outputs, so caching makes a repeat
                ``evaluate`` over the same run essentially free. ``None``
                disables caching.
        """
        self.criterion_name = criterion_name
        self.criterion_prompt = criterion_prompt
        self.judge_model = judge_model
        self.name = name or f"llm_judge.{criterion_name}"
        self._client = client or ModelClient()
        self._rng = rng or random.Random()
        self.cache = cache

    async def score(
        self,
        *,
        prompt_id: str,
        example_id: str,
        input_vars: dict[str, Any],
        source_output: str,
        target_output: str,
        history: list[dict[str, Any]] | None = None,
    ) -> PairedScore | None:
        if not source_output.strip() and not target_output.strip():
            # Tool-only turn: both models answered with calls, not prose.
            # Asking a judge to compare two empty strings costs a call and
            # returns a meaningless tie, so bail before the cache lookup and
            # before the A/B draw — and report nothing, because the tie this
            # used to fabricate was indistinguishable from a judged one. One
            # empty side still goes to the judge: a target that went silent
            # is a real regression.
            return None

        context_section = ""
        if history is not None:
            current_input = _current_input(input_vars)
            context_section = _format_transcript(history, current_input) + "\n\n"

        key = self._cache_key(
            context_section=context_section,
            source_output=source_output,
            target_output=target_output,
        )
        if self.cache is not None:
            hit = await self.cache.get(key)
            if hit is not None:
                replayed = _decode_cached_verdict(hit.response_text)
                if replayed is not None:
                    verdict, target_is_a = replayed
                    return _score_from_verdict(verdict, target_is_a=target_is_a)

        target_is_a = self._rng.random() < 0.5
        if target_is_a:
            output_a, output_b = target_output, source_output
        else:
            output_a, output_b = source_output, target_output

        judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
            context_section=context_section,
            criterion_prompt=self.criterion_prompt,
            output_a=output_a,
            output_b=output_b,
        )

        try:
            response = await self._client.complete(
                model=self.judge_model,
                prompt=judge_prompt,
                temperature=0.0,
            )
            verdict = _parse_verdict(response.text)
        except Exception as exc:
            # A judge that can't produce a verdict is a broken measurement,
            # not a tie. Raise so the harness records the error and the
            # analysis layer excludes the row — neutral-scoring here would
            # silently count every failed judge call as "equivalent".
            log.warning("judge call failed for %s/%s: %s", prompt_id, example_id, exc)
            raise EvaluatorError(
                f"judge call failed for {prompt_id}/{example_id}: {exc}",
            ) from exc

        if self.cache is not None:
            # Store the orientation alongside the verdict: the verdict is a
            # letter, so it's only interpretable together with which side
            # was shown as A.
            await self.cache.put(
                key,
                model_id=self.judge_model,
                prompt_text=judge_prompt,
                inputs={"criterion_name": self.criterion_name},
                response_text=json.dumps({"verdict": verdict, "target_was_a": target_is_a}),
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                cost_usd=response.cost_usd,
                latency_ms=response.latency_ms,
            )

        return _score_from_verdict(verdict, target_is_a=target_is_a)

    def _cache_key(
        self,
        *,
        context_section: str,
        source_output: str,
        target_output: str,
    ) -> str:
        """Key a verdict by criterion + model + the two outputs.

        Built from a *canonical* prompt (source always as A, target always
        as B) rather than the prompt actually sent: the A/B orientation is
        randomized per call, so keying on the sent prompt would miss on
        roughly half of all re-runs. The randomization still happens on the
        live path — it's recorded in the cached payload and replayed
        verbatim on a hit.
        """
        return cache_key(
            model_id=self.judge_model,
            prompt_text=JUDGE_PROMPT_TEMPLATE.format(
                context_section=context_section,
                criterion_prompt=self.criterion_prompt,
                output_a=source_output,
                output_b=target_output,
            ),
            inputs={"criterion_name": self.criterion_name},
            temperature=0.0,
            max_tokens=0,
        )


def _score_from_verdict(verdict: str, *, target_is_a: bool) -> PairedScore:
    """Map a judge verdict letter back to target-vs-source scores."""
    if verdict == "tie":
        return PairedScore(
            source_score=0.5,
            target_score=0.5,
            explanation="tie",
            metadata={"target_was_a": target_is_a, "verdict": "tie"},
        )

    a_won = verdict == "A"
    target_won = a_won == target_is_a
    return PairedScore(
        source_score=0.0 if target_won else 1.0,
        target_score=1.0 if target_won else 0.0,
        explanation=f"target {'won' if target_won else 'lost'}",
        metadata={"target_was_a": target_is_a, "verdict": verdict},
    )


def _decode_cached_verdict(payload: str) -> tuple[str, bool] | None:
    """Decode a cached ``(verdict, target_was_a)`` pair.

    Anything unreadable is treated as a miss — the cache is a disposable
    optimisation and must never fail a run.
    """
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(decoded, dict):
        return None
    verdict = decoded.get("verdict")
    target_was_a = decoded.get("target_was_a")
    if verdict not in ("A", "B", "tie") or not isinstance(target_was_a, bool):
        return None
    return verdict, target_was_a


def _parse_verdict(text: str) -> str:
    """Pull a winner label out of the judge's response.

    Tolerant: tries strict JSON first, then a regex JSON-object scan,
    then a final keyword fallback.
    """
    stripped = text.strip()
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*?\}", stripped, re.DOTALL)
        if not match:
            raise ValueError(
                f"no JSON object found in judge response: {stripped[:200]!r}"
            ) from None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"judge response contained malformed JSON: {match.group(0)[:200]!r}"
            ) from exc

    if not isinstance(data, dict):
        raise ValueError(f"judge response wasn't a JSON object: {data!r}")

    winner = str(data.get("winner", "")).strip().upper()
    if winner == "A":
        return "A"
    if winner == "B":
        return "B"
    if winner in ("TIE", "DRAW", "EQUAL"):
        return "tie"
    raise ValueError(f"invalid winner value: {winner!r}")


def _current_input(input_vars: dict[str, Any]) -> str:
    """Render the current turn's input for the transcript's final ``[user]`` line.

    Single-variable prompts (the common case) just use that value verbatim.
    Multi-variable prompts don't have one obvious "the user said this" field,
    so we fall back to a compact JSON dump of the whole ``input_vars`` dict —
    simple and lossless, at the cost of being less prose-like.
    """
    if len(input_vars) == 1:
        return str(next(iter(input_vars.values())))
    return json.dumps(input_vars)


def _tool_names_by_call_id(history: list[dict[str, Any]]) -> dict[str, str]:
    """Map each recorded tool-call id to the tool it invoked."""
    names: dict[str, str] = {}
    for msg in history:
        for call in msg.get("tool_calls") or []:
            call_id = call.get("id")
            if call_id:
                names[str(call_id)] = str(call.get("name", ""))
    return names


def _transcript_line(msg: dict[str, Any], tool_names: dict[str, str]) -> str:
    """Render one history message as a single labelled transcript line.

    An ``assistant`` turn that called tools is rendered with a compact
    ``→ name(args)`` suffix and a ``tool`` result is labelled with the tool
    it answers, so a judge reading the prefix sees the agent loop rather
    than a run of blank assistant turns.
    """
    role = str(msg.get("role", ""))
    content = str(msg.get("content", ""))
    calls = msg.get("tool_calls") or []
    if role == "assistant" and calls:
        rendered = " ".join(
            f"→ {c.get('name', '')}({json.dumps(c.get('arguments') or {})})" for c in calls
        )
        return f"[assistant] {content} {rendered}" if content else f"[assistant] {rendered}"
    if role == "tool":
        name = tool_names.get(str(msg.get("tool_call_id", "")), "")
        return f"[tool {name}] {content}" if name else f"[tool] {content}"
    return f"[{role}] {content}"


def _format_transcript(history: list[dict[str, Any]], current_input: str) -> str:
    """Render a conversation prefix + the current turn as a labelled transcript.

    ``history`` is the verbatim prefix (``{"role", "content"}`` dicts, plus
    ``tool_calls`` / ``tool_call_id`` on agent turns); ``current_input``
    becomes the final ``[user]`` line, since both the source and target
    outputs are replies to that same message. If the rendered block would
    exceed :data:`MAX_TRANSCRIPT_CHARS`, it's truncated to the system message
    (if present) plus the tail of the remaining turns, with a marker line
    noting the cut.
    """
    tool_names = _tool_names_by_call_id(history)
    lines = [_transcript_line(msg, tool_names) for msg in history]
    lines.append(f"[user] {current_input}")
    body = "\n".join(lines)
    text = (
        f'Conversation context (both outputs respond to the final user message):\n"""\n{body}\n"""'
    )

    if len(text) <= MAX_TRANSCRIPT_CHARS:
        return text

    system_head: list[str] = []
    rest = lines
    if history and history[0]["role"] == "system":
        system_head = [lines[0]]
        rest = lines[1:]

    header = 'Conversation context (both outputs respond to the final user message):\n"""\n'
    footer = '\n"""'
    fixed_overhead = len(header) + len(footer)
    fixed_overhead += sum(len(line) + 1 for line in system_head)  # +1 per newline
    fixed_overhead += len(_TRUNCATION_MARKER) + 1  # marker line + its newline

    budget = max(0, MAX_TRANSCRIPT_CHARS - fixed_overhead)
    tail: list[str] = []
    used = 0
    # Walk backwards keeping whole lines until the budget runs out; the
    # current-input line is last so it's always the first one kept.
    for line in reversed(rest):
        cost = len(line) + 1
        if used + cost > budget and tail:
            break
        tail.insert(0, line)
        used += cost
        if used > budget:
            break

    # Guard rail: even a single kept line (typically the current-input
    # line) can alone exceed the budget for a pathologically long turn.
    # Hard-clip it so the "total under the cap" contract always holds.
    if len(tail) == 1 and len(tail[0]) + 1 > budget:
        tail[0] = tail[0][: max(0, budget - 1)]

    kept_body_lines = [*system_head, _TRUNCATION_MARKER, *tail]
    truncated_body = "\n".join(kept_body_lines)
    return f"{header}{truncated_body}{footer}"


__all__ = ["MAX_TRANSCRIPT_CHARS", "PairwiseJudgeEvaluator"]
