"""
Command line for the PGA analyst agent.

    python -m agent.cli                                      # conversation: /new starts over, /quit exits
    python -m agent.cli "Who is ranked first this week?"     # one question, then exit
    python -m agent.cli --model claude-haiku-5-5             # cheaper model for smoke tests

Follow-ups stay in one conversation, so when the agent asks which player you
mean, your reply answers it. Every turn is logged to traces/ (see agent.agent).

Lineups need your approval: after any turn where the agent proposed lineups,
the CLI shows them (from the tool output, not the agent's summary) and asks
Approve? [y/N]. Only a yes saves them, to the event's data/processed folder as
lineups_approved_{proposal_id}.csv. The agent has no save tool.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import uuid
from collections import Counter
from pathlib import Path
from typing import List, Optional

import anthropic
import pandas as pd

from agent.agent import MODEL, TIMEOUT_S, AgentResult, Tracer, run
from agent.db import connect
from src.paths import PROCESSED_DIR, TRACES_DIR

HELP = ("Ask about rankings, features, explanations, value plays, lineups or model performance. "
        "/new starts over, /quit exits.")


def _footer(res: AgentResult) -> str:
    counts = Counter(c.name for c in res.tool_calls)
    tools = ", ".join(f"{n} x{k}" if k > 1 else n for n, k in counts.items()) or "no tools"
    stop = "" if res.stop_reason == "end_turn" else f"stopped: {res.stop_reason}; "
    return f"[{stop}{tools}; {res.input_tokens:,} in / {res.output_tokens:,} out tokens; trace {res.run_id}]"


def _show(p: dict) -> None:
    ev = p["event"]
    print(f"\nProposal {p['proposal_id']}: {ev['name']} {ev['season']}, {p['model_version']} model, "
          f"budget ${p['budget']:,}")
    if p["locks"] or p["excludes"]:
        print(f"  Locked: {', '.join(p['locks']) or 'none'}. Excluded: {', '.join(p['excludes']) or 'none'}.")
    for k, lu in enumerate(p["lineups"], 1):
        print(f"  Lineup {k}: ${lu['total_salary']:,} (${lu['budget_left']:,} left), score {lu['total_score']}")
        for pl in lu["players"]:
            print(f"    {pl['name']:<26} {'$' + format(pl['salary'], ','):>10}   model rank {pl['model_rank']}")


def _save(p: dict) -> Path:
    """Write an approved proposal next to that week's other lineups."""
    ev_id = p["event"]["event_id"]
    folder = next(iter(sorted(PROCESSED_DIR.glob(f"{ev_id}_*"))), PROCESSED_DIR / ev_id)
    folder.mkdir(parents=True, exist_ok=True)
    approved_at = dt.datetime.now().isoformat(timespec="seconds")
    rows = [{"lineup": k, **{f: pl[f] for f in ("player_id", "name", "salary", "model_rank", "score")},
             "proposal_id": p["proposal_id"], "model_version": p["model_version"],
             "budget": p["budget"], "approved_at": approved_at}
            for k, lu in enumerate(p["lineups"], 1) for pl in lu["players"]]
    path = folder / f"lineups_approved_{p['proposal_id']}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def review(res: AgentResult) -> None:
    """Ask the user to approve each lineup proposal from this turn; save only on yes."""
    proposals = [c.output for c in res.tool_calls if c.name == "propose_lineups" and "error" not in c.output]
    if not proposals:
        return
    tracer = Tracer(res.run_id, TRACES_DIR)
    for p in proposals:
        _show(p)
        try:
            answer = input(f"Approve and save proposal {p['proposal_id']}? [y/N] ")
        except (EOFError, KeyboardInterrupt):
            answer = ""
            print()
        approved = answer.strip().lower() in ("y", "yes")
        path = _save(p) if approved else None
        print(f"Saved to {path}" if path else "Not saved.")
        tracer.log("approval", proposal_id=p["proposal_id"], approved=approved,
                   path=str(path) if path else None)


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
    review(res)
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
