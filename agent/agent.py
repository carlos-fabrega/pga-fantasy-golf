"""
The analyst agent's loop, written by hand rather than with a framework or the
SDK's tool runner, so every message, tool call and result is explicit:

  1. messages = [question]; call Claude with the system prompt and tool schemas
  2. if it asks for tools, run each one (agent/tools.py), append the results, repeat
  3. stop on a final text answer, or after MAX_ITERATIONS calls, or past the timeout

Every step is appended to traces/YYYY-MM-DD.jsonl: the model's output, each
tool's input and output, tokens and latency.

Pass the previous turn's AgentResult.messages as `history` to continue a
conversation (agent/cli.py does this, so the agent can ask a clarifying question).
"""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import anthropic

from agent.db import connect
from agent.prompts import SYSTEM_PROMPT
from agent.tools import TOOLS, run_tool
from src.paths import TRACES_DIR

MODEL = "claude-opus-5-5"
EFFORT = "medium"
MAX_ITERATIONS = 8        # model calls per question
TIMEOUT_S = 180.0         # wall-clock budget per question
MAX_TOKENS = 16000

# Server-side refusal fallback: a declined request is re-run on a fallback model
# inside the same call. Sent through extra_body because SDK 0.72 (the newest on
# Python 3.8) predates these parameters.
BETAS = ["server-side-fallback-2026-07-01"]
EXTRA_BODY = {"fallbacks": "default", "output_config": {"effort": EFFORT}}

# Block types sent back to the API on the next turn. Thinking blocks must be
# returned unchanged; anything else (e.g. fallback markers the old SDK can't
# parse) is dropped.
_RESEND = {"text", "thinking", "redacted_thinking", "tool_use"}


@dataclass
class ToolCall:
    name: str
    input: dict
    output: dict


@dataclass
class AgentResult:
    answer: str
    stop_reason: str          # end_turn | max_iterations | timeout | max_tokens | refusal | ...
    iterations: int
    tool_calls: List[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    run_id: str = ""
    messages: List[dict] = field(default_factory=list)   # conversation so far, for the next turn


class Tracer:
    """Appends one JSON line per step to traces/YYYY-MM-DD.jsonl."""

    def __init__(self, run_id: str, trace_dir: Path = TRACES_DIR):
        self.run_id = run_id
        self.path = trace_dir / f"{dt.date.today().isoformat()}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, **data) -> None:
        record = {"run_id": self.run_id, "ts": dt.datetime.now().isoformat(timespec="milliseconds"),
                  "event": event, **data}
        with self.path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")


def _block_summary(block) -> dict:
    """What the trace keeps of one content block (thinking text is empty by default)."""
    if block.type == "text":
        return {"type": "text", "text": block.text}
    if block.type == "tool_use":
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    return {"type": block.type}


def _resend(content) -> List[dict]:
    return [b.to_dict() for b in content if b.type in _RESEND]


def run(
    question: str,
    history: Optional[List[dict]] = None,
    client: Optional[anthropic.Anthropic] = None,
    con: Optional[sqlite3.Connection] = None,
    model: str = MODEL,
    max_iterations: int = MAX_ITERATIONS,
    timeout_s: float = TIMEOUT_S,
    trace_dir: Path = TRACES_DIR,
    conversation_id: Optional[str] = None,
) -> AgentResult:
    """Answer one question, calling tools as Claude asks for them."""
    client = client or anthropic.Anthropic(timeout=timeout_s)
    con = con or connect()
    run_id = uuid.uuid4().hex[:12]
    tracer = Tracer(run_id, trace_dir)
    tracer.log("start", question=question, conversation_id=conversation_id, history_messages=len(history or []),
               model=model, effort=EFFORT, max_iterations=max_iterations)

    messages: List[dict] = list(history or []) + [{"role": "user", "content": question}]
    result = AgentResult(answer="", stop_reason="", iterations=0, run_id=run_id, messages=messages)
    deadline = time.monotonic() + timeout_s

    while True:
        if result.iterations >= max_iterations:
            result.stop_reason = "max_iterations"
            break
        if time.monotonic() >= deadline:
            result.stop_reason = "timeout"
            break

        result.iterations += 1
        t0 = time.monotonic()
        response = client.beta.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
            betas=BETAS,
            extra_body=EXTRA_BODY,
        )
        usage = response.usage
        result.input_tokens += usage.input_tokens
        result.output_tokens += usage.output_tokens
        tracer.log("model_call", iteration=result.iterations, model=response.model,
                   stop_reason=response.stop_reason,
                   latency_ms=round((time.monotonic() - t0) * 1000),
                   input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
                   cache_read_tokens=getattr(usage, "cache_read_input_tokens", None) or 0,
                   content=[_block_summary(b) for b in response.content])

        result.answer = "\n".join(b.text for b in response.content if b.type == "text" and b.text).strip()
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if response.stop_reason != "tool_use" or not tool_uses:
            result.stop_reason = response.stop_reason
            # Keep the final turn for follow-ups, minus any tool_use left without a result.
            final = [b for b in _resend(response.content) if b["type"] != "tool_use"]
            if final:
                messages.append({"role": "assistant", "content": final})
            break

        messages.append({"role": "assistant", "content": _resend(response.content)})
        tool_results = []
        for block in tool_uses:
            t0 = time.monotonic()
            output = run_tool(con, block.name, dict(block.input))
            is_error = "error" in output
            tracer.log("tool_call", iteration=result.iterations, tool_use_id=block.id,
                       name=block.name, input=block.input, output=output, is_error=is_error,
                       latency_ms=round((time.monotonic() - t0) * 1000))
            result.tool_calls.append(ToolCall(block.name, dict(block.input), output))
            tool_results.append({"type": "tool_result", "tool_use_id": block.id,
                                 "content": json.dumps(output), "is_error": is_error})
        # All results for one turn go back in a single user message.
        messages.append({"role": "user", "content": tool_results})

    if result.stop_reason in ("max_iterations", "timeout"):
        result.answer = (f"I stopped before finishing ({result.stop_reason.replace('_', ' ')} after "
                         f"{result.iterations} model calls), so I don't have a complete answer.")
    elif result.stop_reason == "refusal":
        result.answer = "The model declined to answer this question."
    tracer.log("end", stop_reason=result.stop_reason, iterations=result.iterations,
               tool_calls=len(result.tool_calls), input_tokens=result.input_tokens,
               output_tokens=result.output_tokens, answer=result.answer)
    return result

