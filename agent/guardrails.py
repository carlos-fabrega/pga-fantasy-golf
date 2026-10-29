"""
Deterministic guardrails for the analyst agent (PGA_AGENT_PLAN.md §7): plain
Python checks, no LLM.

- check_numbers: every number in an answer must appear in a tool result, the
  user's own words, or the system prompt and tool descriptions we wrote,
  allowing for rounding. "Top 10" framing and table row numbers are skipped.
  agent.run applies it to each answer, and the CLI shows what it flags.
- validate_proposal: re-checks a propose_lineups result against the database
  (roster size, salary cap, locks, excludes, distinct lineups) before the CLI
  lets the user approve it. It reads salaries itself rather than trusting the
  totals in the proposal.

The other §7 guardrail, player-name resolution, is the resolve_player tool
plus the system prompt's rule to ask when a name is ambiguous.
"""

from __future__ import annotations

import json
import re
import sqlite3
from typing import Iterator, List, Tuple

from agent.prompts import SYSTEM_PROMPT
from agent.tools import TOOLS, _ids
from src.optimizer import DEFAULT_BUDGET, N_PLAYERS

# A number as an answer writes it: optional sign and $, thousands commas,
# decimals, K/M suffix, %, ordinal ending. Digits inside identifiers
# (R2026060, sg_putt_per_round_roll16, proposal ids) are skipped.
_NUMBER = re.compile(
    r"(?<![\w.$])(?P<sign>[-−+])?(?P<cur>\$)?"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?P<suf>[KkMm](?![A-Za-z]))?(?P<pct>%)?(?:st|nd|rd|th)?(?!\w)")
_LIST_MARKER = re.compile(r"^\s*\d+[.)]\s", re.MULTILINE)   # "1. " starting a list item
_TABLE_ROW_NUMBER = re.compile(r"^(\s*\|)\s*\d+\s*(?=\|)", re.MULTILINE)   # "| 2 | Webb Simpson |"
_TOP_N = re.compile(r"\btop[- ]\d+\b", re.IGNORECASE)   # framing ("the rest of the top 10"), not a claim
# Text we wrote, so its numbers ("$1,650,000" default budget, "2018-2022") are fair to repeat.
_REFERENCE = SYSTEM_PROMPT + json.dumps(TOOLS)
# Tool messages write money with thousands commas ("$1,898,000"); JSON lists use ", ".
_SOURCE_NUMBER = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")


def _claims(answer: str) -> Iterator[Tuple[str, float, float, bool]]:
    """(as written, absolute value, rounding tolerance, is a percentage) per number."""
    text = _TOP_N.sub(" ", _TABLE_ROW_NUMBER.sub(r"\1 ", _LIST_MARKER.sub(" ", answer)))
    for m in _NUMBER.finditer(text):
        num = m["num"].replace(",", "")
        decimals = len(num.split(".")[1]) if "." in num else 0
        # K/M only scale money: "$1.65M" is 1,650,000, but "3M Open" is a tournament.
        mult = {"k": 1e3, "m": 1e6}.get((m["suf"] or "").lower(), 1.0) if m["cur"] else 1.0
        yield m.group(0), float(num) * mult, 0.5 * 10 ** -decimals * mult, bool(m["pct"])


def _sources(messages: List[dict]) -> Iterator[str]:
    """Text the answer may draw numbers from: our prompt and tool docs, the user's words, tool inputs and results."""
    yield _REFERENCE
    for msg in messages:
        content = msg["content"]
        if isinstance(content, str):
            if msg["role"] == "user":
                yield content
            continue
        for block in content:
            if block.get("type") == "tool_result":
                yield block["content"]
            elif block.get("type") == "tool_use":
                yield json.dumps(block["input"])
            elif block.get("type") == "text" and msg["role"] == "user":
                yield block["text"]


def check_numbers(answer: str, messages: List[dict]) -> List[str]:
    """
    Numbers in the answer that no source supports. A number is supported if a
    source value rounds to it at the precision written ("0.34" from 0.3394,
    "$1.65M" from 1650000), ignoring sign, or as a percentage of a share
    ("2.5%" from 0.025). Derived numbers (sums, differences the model worked
    out itself) are flagged too: the rule is that every number comes from a tool.
    """
    known = [float(x.replace(",", "")) for text in _sources(messages) for x in _SOURCE_NUMBER.findall(text)]
    unsupported: List[str] = []
    for raw, value, tol, pct in _claims(answer):
        ok = any(abs(value - s) <= tol + 1e-9 for s in known) or (
            pct and any(abs(value / 100 - s) <= tol / 100 + 1e-9 for s in known))
        if not ok and raw not in unsupported:
            unsupported.append(raw)
    return unsupported


def validate_proposal(con: sqlite3.Connection, args: dict, proposal: dict) -> List[str]:
    """
    Problems with a propose_lineups result, checked against the request (args)
    and the salaries table. An empty list means every lineup is valid.
    """
    if "error" in proposal:
        return []
    event_id = proposal["event"]["event_id"]
    budget = int(args.get("budget", DEFAULT_BUDGET))
    locks, excludes = set(_ids(args.get("locks"))), set(_ids(args.get("excludes")))
    salary = {pid: int(round(s)) for pid, s in con.execute(
        "SELECT player_id, salary FROM salaries WHERE event_id = ?", (event_id,))}

    problems = []
    if proposal["budget"] != budget:
        problems.append(f"The proposal's budget (${proposal['budget']:,}) isn't the requested ${budget:,}.")
    if len(proposal["lineups"]) > int(args.get("n_lineups", 1)):
        problems.append(f"{len(proposal['lineups'])} lineups returned for {args.get('n_lineups', 1)} requested.")
    seen: List[set] = []
    for k, lu in enumerate(proposal["lineups"], 1):
        tag = f"Lineup {k}"
        names = {p["player_id"]: p["name"] for p in lu["players"]}
        ids = [p["player_id"] for p in lu["players"]]
        if len(ids) != N_PLAYERS:
            problems.append(f"{tag} has {len(ids)} players, not {N_PLAYERS}.")
        if len(set(ids)) != len(ids):
            problems.append(f"{tag} lists a player more than once.")
        unpriced = [names[i] for i in ids if i not in salary]
        if unpriced:
            problems.append(f"{tag} includes players with no salary at {event_id}: {', '.join(unpriced)}.")
        total = sum(salary.get(i, 0) for i in ids)
        if total > budget:
            problems.append(f"{tag} costs ${total:,}, over the ${budget:,} cap.")
        if lu["total_salary"] != total:
            problems.append(f"{tag} reports ${lu['total_salary']:,} in salary; the salaries add up to ${total:,}.")
        wrong = [p["name"] for p in lu["players"] if p["player_id"] in salary and p["salary"] != salary[p["player_id"]]]
        if wrong:
            problems.append(f"{tag} shows the wrong salary for {', '.join(wrong)}.")
        if locks - set(ids):
            problems.append(f"{tag} is missing locked players: {', '.join(sorted(locks - set(ids)))}.")
        if excludes & set(ids):
            problems.append(f"{tag} includes excluded players: {', '.join(names[i] for i in sorted(excludes & set(ids)))}.")
        if set(ids) in seen:
            problems.append(f"{tag} repeats an earlier lineup.")
        seen.append(set(ids))
    return problems
