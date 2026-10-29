"""
Tests for the deterministic guardrails: the numeric-claim checker and the
lineup validator (the latter against the tools' fixture database).

Run:  python -m pytest tests/test_guardrails.py
"""

from __future__ import annotations

import copy
import json

import pytest

from agent.guardrails import check_numbers, validate_proposal
from agent.tools import run_tool
from tests.test_tools import SALARY, con, model   # noqa: F401  (pytest fixtures)


# ── check_numbers ──────────────────────────────────────────────────────────────

def _conversation(question, *outputs, tool_input=None):
    return [
        {"role": "user", "content": question},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "x", "input": tool_input or {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": json.dumps(o)}
                                     for o in outputs]},
        {"role": "assistant", "content": [{"type": "text", "text": "The gap is 9.99."}]},   # never a source
    ]


TOOL_OUTPUT = {"score_gap": -0.3394, "contribution": 0.194, "salary": 1650000, "rank": 5,
               "prize_pct": 0.0255, "date": "2026-08-27", "feature": "sg_putt_per_round_roll16"}


@pytest.mark.parametrize("answer", [
    "The gap is 0.3394.",              # exact
    "The gap is 0.34.",                # rounded
    "Kim leads by −0.339.",            # unicode minus, rounded
    "a gap of 0.3394 in Kim's favour", # sign dropped in prose
    "It adds +0.194.",
    "The cap is $1,650,000.",
    "The cap is $1.65M.",
    "The cap is $1650K.",
    "He's ranked 5th.",
    "He won 2.55% of the purse.",      # share shown as a percentage
    "on 2026-08-27",
    "over his last 16 starts",         # backed by the feature name
    "Show me the top 3",               # (the question's own number, below)
])
def test_supported_numbers_pass(answer):
    assert check_numbers(answer, _conversation("Show me the top 3", TOOL_OUTPUT)) == []


def test_unsupported_numbers_are_flagged_once_each():
    answer = "Scheffler gains +0.34 but loses −0.73, and −0.73 again; overall 0.28 is typical. Rank 7."
    assert check_numbers(answer, _conversation("Why?", TOOL_OUTPUT)) == ["−0.73", "0.28", "7"]


def test_numbers_inside_tool_messages_count():
    err = {"error": "infeasible", "detail": "The locked players cost $1,898,000, over the $1,000,000 budget."}
    convo = _conversation("Lock them", err)
    assert check_numbers("They cost $1,898,000 against a $1,000,000 cap.", convo) == []
    assert check_numbers("That's $898,000 over.", convo) == ["$898,000"]          # derived
    assert check_numbers("Ranks 1, 2 and 3.", _conversation("q", {"ranks": [1, 2, 3]})) == []


def test_framing_table_rows_and_our_own_docs_are_not_claims():
    convo = _conversation("Value plays?", {"players": [{"name": "Webb Simpson", "value": 91}]})
    table = "| # | Player | Value |\n|---|---|---|\n| 1 | Davis Chatfield | +103 |\n| 2 | Webb Simpson | +91 |"
    assert check_numbers(table, convo) == ["+103"]                     # the row numbers 1 and 2 are skipped
    assert check_numbers("I can show the rest of the top 10, or the Top-25.", convo) == []
    # The default budget and backtest years come from the tool descriptions we wrote.
    assert check_numbers("The default cap is $1,650,000; compare the 2018–2022 backtest.", convo) == []
    assert check_numbers("He's 12th of 30.", convo) == ["12th", "30"]   # a real claim is still checked


def test_a_bare_m_suffix_is_not_millions():
    convo = _conversation("q", {"events": ["3M Open 2026"]})
    assert check_numbers("Salaries exist for the 3M Open.", convo) == []


def test_the_models_own_earlier_text_is_not_a_source():
    assert check_numbers("As I said, 9.99.", _conversation("Why?", TOOL_OUTPUT)) == ["9.99"]


def test_identifiers_list_markers_and_tool_inputs():
    answer = "1. R2026060 (`sg_app_per_round_roll8`), proposal f57ec4c8: top 30 shown."
    convo = _conversation("Why?", TOOL_OUTPUT, tool_input={"top_n": 30})
    assert check_numbers(answer, convo) == []


# ── validate_proposal ──────────────────────────────────────────────────────────

ARGS = {"n_lineups": 2, "budget": 3_000_000, "locks": ["37455"], "excludes": ["46046"]}


@pytest.fixture()
def proposal(con):
    out = run_tool(con, "propose_lineups", ARGS)
    assert "error" not in out
    return out


def _break(proposal, fn):
    bad = copy.deepcopy(proposal)
    fn(bad)
    return bad


def test_optimizer_output_is_valid(con, proposal):
    assert validate_proposal(con, ARGS, proposal) == []
    assert validate_proposal(con, ARGS, {"error": "infeasible"}) == []


def _swap_in(pid):
    def fn(p):
        p["lineups"][0]["players"][-1].update(player_id=pid, name=pid, salary=SALARY.get(pid, 0))
    return fn


@pytest.mark.parametrize("tamper, problem", [
    (lambda p: p["lineups"][0]["players"].pop(), "has 5 players"),
    (lambda p: p["lineups"][0]["players"].append(dict(p["lineups"][0]["players"][0])), "more than once"),
    (_swap_in("46046"), "includes excluded players"),
    (lambda p: [lu["players"].sort(key=lambda x: x["player_id"] == "37455") or lu["players"].pop()
                for lu in p["lineups"][:1]], "missing locked players: 37455"),
    (_swap_in("46464"), "no salary at R2026060"),
    (lambda p: p["lineups"][0].update(total_salary=1), "reports $1 in salary"),
    (lambda p: p["lineups"][0]["players"][0].update(salary=1), "wrong salary"),
    (lambda p: p["lineups"].__setitem__(1, p["lineups"][0]), "repeats an earlier lineup"),
    (lambda p: p.update(budget=9_999_999), "isn't the requested"),
    (lambda p: p["lineups"].append(p["lineups"][0]), "3 lineups returned for 2"),
])
def test_tampered_proposals_are_caught(con, proposal, tamper, problem):
    problems = validate_proposal(con, ARGS, _break(proposal, tamper))
    assert any(problem in x for x in problems), problems


def test_over_the_cap(con, proposal):
    tight = {**ARGS, "budget": 2_000_000}
    bad = _break(proposal, lambda p: p.update(budget=2_000_000))
    assert any("over the $2,000,000 cap" in x for x in validate_proposal(con, tight, bad))
