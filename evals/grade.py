"""
Deterministic graders for eval cases: no LLM judge. Each takes the case's
`expected` block, the agent's answer and its tool calls, and returns
(passed, reason). See evals/generate.py for what each type means.

Text matching is deliberately simple (normalized substrings, keyword lists), so
a failure can mean the grader missed a phrasing. Read the answer in the report
before changing the agent, and widen the grader if that's what went wrong.
"""

from __future__ import annotations

import re
import sqlite3
from typing import Callable, Dict, List, Tuple

from agent.agent import ToolCall
from agent.db import match_key
from agent.guardrails import _claims, validate_proposal

Result = Tuple[bool, str]

# Plain-English words for each feature family, for explanations that describe a
# driver ("putting over the last 16 starts") instead of naming it.
_SG_WORDS = {
    "total": ["total", "overall"],
    "t2g": ["tee-to-green", "tee to green", "t2g", "ball-striking", "ball striking"],
    "app": ["approach", "iron"],
    "arg": ["around the green", "around-the-green", "short game", "chipping", "arg"],
    "ott": ["off the tee", "off-the-tee", "driving", "tee shot", "ott"],
    "putt": ["putt"],
}
_OTHER_WORDS = {
    "playoff_sg_hist": ["playoff"], "playoff_starts": ["playoff"],
    "course_sg_hist": ["course"], "course_starts": ["course"],
    "links_sg_hist": ["links"], "links_starts": ["links"],
    "starts_last8": ["starts"],
}
_UNAVAILABLE_CUES = ["not available", "unavailable", "isn't available", "aren't available", "not in the database",
                     "no salar", "can't", "cannot", "couldn't", "could not", "don't have", "doesn't have",
                     "no data", "no record", "not found", "no longer", "retrained", "changed", "no event",
                     "isn't in", "aren't in", "not have", "doesn't include", "no results", "no explanation"]
_PLAYER_TOOLS = {"get_player_features", "explain_prediction"}


def _has_name(answer: str, name: str) -> bool:
    return f" {match_key(name)} " in f" {match_key(answer)} "


def _sentences(answer: str) -> List[str]:
    return [s.lower() for s in re.split(r"(?<=[.!?])\s+|\n+", answer) if s.strip()]


def feature_mentioned(feature: str, answer: str) -> bool:
    """The feature's id appears, or a sentence names its family (and its window, for rolling ones)."""
    if feature in answer:
        return True
    m = re.fullmatch(r"sg_(\w+?)_per_round_roll(\d+)", feature)
    if m:
        words, window = _SG_WORDS[m.group(1)], m.group(2)
        return any(any(w in s for w in words) and re.search(rf"(?<!\d){window}(?!\d)", s)
                   for s in _sentences(answer))
    m = re.fullmatch(r"made_cut_rate_roll(\d+)", feature)
    if m:
        return any("cut" in s and re.search(rf"(?<!\d){m.group(1)}(?!\d)", s) for s in _sentences(answer))
    return any(w in answer.lower() for w in _OTHER_WORDS.get(feature, [feature]))


def _numbers_in(answer: str) -> List[float]:
    return [value for _, value, _, _ in _claims(answer)]


def _outputs(calls: List[ToolCall], tool: str) -> List[dict]:
    return [c.output for c in calls if c.name == tool]


# ── Graders by expected type ───────────────────────────────────────────────────

def _exact(exp: dict, answer: str, calls, con) -> Result:
    return (True, "") if _has_name(answer, exp["value"]) else (False, f"doesn't name {exp['value']}")


def _number(exp: dict, answer: str, calls, con) -> Result:
    found = _numbers_in(answer)
    missing = [v for v in exp["values"] if not any(abs(abs(x) - abs(v)) <= exp["tol"] + 1e-9 for x in found)]
    return (False, f"missing {missing}") if missing else (True, "")


def _drivers(exp: dict, answer: str, calls, con) -> Result:
    missing = [f for f in exp["must_mention"] if not feature_mentioned(f, answer)]
    return (False, f"doesn't mention {missing}") if missing else (True, "")


def _topset(exp: dict, answer: str, calls, con) -> Result:
    hits = [n for n in exp["names"] if _has_name(answer, n)]
    ok = len(hits) >= exp["min_overlap"]
    return ok, "" if ok else f"names {len(hits)} of {exp['names']} (needs {exp['min_overlap']})"


def _lineup_valid(exp: dict, answer: str, calls, con) -> Result:
    good = [c for c in calls if c.name == "propose_lineups" and "error" not in c.output]
    if not good:
        return False, "no successful propose_lineups call"
    # Validate the last proposal against what the user asked for, not what the agent passed.
    wanted = {k: exp[k] for k in ("n_lineups", "locks", "excludes", "budget")}
    problems = validate_proposal(con, wanted, good[-1].output)
    if len(good[-1].output["lineups"]) != exp["n_lineups"]:
        problems.append(f"{len(good[-1].output['lineups'])} lineups, wanted {exp['n_lineups']}")
    return (False, "; ".join(problems)) if problems else (True, "")


def _infeasible(exp: dict, answer: str, calls, con) -> Result:
    outs = _outputs(calls, "propose_lineups")
    if any("error" not in o for o in outs):
        return False, "proposed a lineup anyway"
    if not any(o.get("error") == "infeasible" for o in outs):
        return False, "never got the optimizer's infeasible result"
    said = any(w in answer.lower() for w in ["budget", "cap", "can't", "cannot", "exceed", "over", "infeasible"])
    return (True, "") if said else (False, "doesn't say the locks don't fit the budget")


def _no_player_data(calls) -> bool:
    return not any(c.name in _PLAYER_TOOLS and "error" not in c.output for c in calls)


def _clarify(exp: dict, answer: str, calls, con) -> Result:
    resolved = _outputs(calls, "resolve_player")
    ambiguous = [o for o in resolved if o.get("status") == "ambiguous"]
    if not ambiguous:
        return False, "resolve_player never came back ambiguous"
    named = {c["name"] for o in ambiguous for c in o["candidates"] if _has_name(answer, c["name"])}
    if "?" not in answer or len(named) < 2:
        return False, f"doesn't ask which player (names {len(named)} candidates)"
    return (True, "") if _no_player_data(calls) else (False, "used one candidate's data without asking")


def _refusal_or_clarify(exp: dict, answer: str, calls, con) -> Result:
    if not _no_player_data(calls):
        return False, "returned player data for a name that doesn't exist"
    said = "?" in answer or any(w in answer.lower() for w in _UNAVAILABLE_CUES + ["no player", "no golfer"])
    return (True, "") if said else (False, "doesn't say the player wasn't found or ask")


def _unavailable(exp: dict, answer: str, calls, con) -> Result:
    if not any(c.output.get("error") == exp["error"] for c in calls):
        return False, f"never hit the {exp['error']} error"
    said = any(w in answer.lower() for w in _UNAVAILABLE_CUES)
    return (True, "") if said else (False, "doesn't say the data is unavailable")


GRADERS: Dict[str, Callable[..., Result]] = {
    "exact": _exact, "number": _number, "drivers": _drivers, "topset": _topset,
    "lineup_valid": _lineup_valid, "infeasible": _infeasible, "clarify": _clarify,
    "refusal_or_clarify": _refusal_or_clarify, "unavailable": _unavailable,
}


def grade(case: dict, answer: str, calls: List[ToolCall], con: sqlite3.Connection) -> Result:
    exp = case["expected"]
    return GRADERS[exp["type"]](exp, answer, calls, con)


def tools_ok(case: dict, calls: List[ToolCall]) -> bool:
    """Tool-call correctness: every expected tool was called at least once."""
    return set(case["expected_tools"]) <= {c.name for c in calls}
