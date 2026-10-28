"""
Command line for the PGA analyst agent.

    python -m agent.cli                                      # conversation: /new starts over, /quit exits
    python -m agent.cli "Who is ranked first this week?"     # one question, then exit
    python -m agent.cli --model claude-haiku-5-5             # cheaper model for smoke tests

Follow-ups stay in one conversation, so when the agent asks which player you
mean, your reply answers it. Every turn is logged to traces/ (see agent.agent).
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from collections import Counter
from typing import List, Optional

import anthropic

from agent.agent import MODEL, TIMEOUT_S, AgentResult, run
from agent.db import connect

HELP = "Ask about the model's rankings, features or explanations. /new starts over, /quit exits."


def _footer(res: AgentResult) -> str:
    counts = Counter(c.name for c in res.tool_calls)
    tools = ", ".join(f"{n} x{k}" if k > 1 else n for n, k in counts.items()) or "no tools"
    stop = "" if res.stop_reason == "end_turn" else f"stopped: {res.stop_reason}; "
    return f"[{stop}{tools}; {res.input_tokens:,} in / {res.output_tokens:,} out tokens; trace {res.run_id}]"


def ask(question: str, history: List[dict], **kwargs) -> Optional[AgentResult]:
    """Run one turn and print it. Returns None (history unchanged) if it didn't complete."""
    try:
        res = run(question, history=history, **kwargs)
    except anthropic.APIError as e:
        print(f"API error: {e}")
        return None
    except KeyboardInterrupt:
        print("\nCancelled.")
        return None
    print(f"\n{res.answer}\n\n{_footer(res)}")
    return res


def repl(**kwargs) -> None:
    print(HELP)
    history: List[dict] = []
    conversation_id = uuid.uuid4().hex[:12]
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        if line in ("/quit", "/exit"):
            return
        if line == "/new":
            history, conversation_id = [], uuid.uuid4().hex[:12]
            print("Started a new conversation.")
            continue
        if line.startswith("/"):
            print(HELP)
            continue
        res = ask(line, history, conversation_id=conversation_id, **kwargs)
        if res is not None:
            history = res.messages


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Ask the PGA analyst agent questions")
    parser.add_argument("question", nargs="?", help="Ask one question and exit")
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args(argv)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set. Export it in your shell profile and open a new terminal.")
        return 1
    kwargs = dict(client=anthropic.Anthropic(timeout=TIMEOUT_S), con=connect(), model=args.model)
    if args.question:
        return 0 if ask(args.question, [], **kwargs) else 1
    repl(**kwargs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
