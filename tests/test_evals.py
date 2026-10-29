"""
Tests for the eval harness: the case file, the graders, and the runner (with
the agent stubbed, so no API calls).

Run:  python -m pytest tests/test_evals.py
"""

from __future__ import annotations

import json
from collections import Counter

import pytest

from agent.agent import AgentResult, ToolCall
from agent.tools import TOOLS, run_tool
from evals import run_evals
from evals.grade import GRADERS, feature_mentioned, grade, tools_ok
from tests.test_tools import con, model   # noqa: F401  (pytest fixtures)


def _case(type_, **exp):
    return {"id": "x", "category": "lookup", "question": "q", "expected_tools": [], "expected": {"type": type_, **exp}}


def _call(tool, output, **inp):
    return ToolCall(tool, inp, output)


# ── The committed case file ────────────────────────────────────────────────────

def test_case_file_is_well_formed():
    cases = run_evals.load_cases()
    assert len({c["id"] for c in cases}) == len(cases) == 44
    assert Counter(c["category"] for c in cases) == {
        "lookup": 11, "explanation": 10, "value": 5, "lineup": 8, "guardrail": 10}
    assert sum(c.get("hard", False) for c in cases) == 14
    tool_names = {t["name"] for t in TOOLS}

    def types(exp):
        return [exp["type"]] + [t for c in exp.get("checks", []) for t in types(c)]
    for c in cases:
        assert c["question"] and set(c["expected_tools"]) <= tool_names
        for key in ("expected", "expected_first"):
            if key in c:
                assert set(types(c[key])) <= set(GRADERS)


# ── Graders ────────────────────────────────────────────────────────────────────

def test_exact_ignores_case_and_accents():
    assert grade(_case("exact", value="Ludvig Åberg"), "LUDVIG ABERG is first.", [], None)[0]
    assert not grade(_case("exact", value="Si Woo Kim"), "Tom Kim is first.", [], None)[0]


def test_number_needs_every_value_within_tolerance():
    case = _case("number", values=[6, 23], tol=0)
    assert grade(case, "Model rank 6th vs salary rank 23.", [], None)[0]
    ok, reason = grade(case, "Model rank 6.", [], None)
    assert not ok and "23" in reason
    assert grade(_case("number", values=[196000], tol=0), "His salary is $196K.", [], None)[0]
    assert grade(_case("number", values=[-0.252], tol=0.0015), "It was -0.252.", [], None)[0]
    assert not grade(_case("number", values=[2.604], tol=0.0015), "About 2.6.", [], None)[0]


@pytest.mark.parametrize("feature, text, hit", [
    ("sg_putt_per_round_roll16", "Putting over the last 16 starts costs him 0.145.", True),
    ("sg_putt_per_round_roll16", "`sg_putt_per_round_roll16` is the biggest driver.", True),
    ("sg_putt_per_round_roll16", "Putting over the last 8 starts hurts. So does 16-start approach.", False),
    ("sg_app_per_round_roll8", "His 8-start iron play leads the field.", True),
    ("sg_ott_per_round_roll24", "Driving across 24 starts helps.", True),
    ("playoff_sg_hist", "His playoff record is strong.", True),
    ("sg_total_per_round_roll24", "Strong strokes-gained over the last 24 starts.", True),
    ("sg_total_per_round_roll24", "Strokes gained putting over the last 24 starts.", False),
    ("course_starts", "Nothing about it.", False),
])
def test_feature_mentioned(feature, text, hit):
    assert feature_mentioned(feature, text) is hit


def test_topset_and_drivers():
    names = ["A One", "B Two", "C Three", "D Four", "E Five"]
    assert grade(_case("topset", names=names, min_overlap=4), "A One, B Two, C Three and D Four.", [], None)[0]
    assert not grade(_case("topset", names=names, min_overlap=4), "A One and B Two.", [], None)[0]
    case = _case("drivers", must_mention=["playoff_sg_hist", "sg_total_per_round_roll24"])
    assert grade(case, "Playoff history helps. Total strokes gained over 24 starts too.", [], None)[0]
    assert not grade(case, "Playoff history helps.", [], None)[0]


AMBIGUOUS = {"status": "ambiguous", "candidates": [{"name": "Si Woo Kim"}, {"name": "Tom Kim"}]}


def test_clarify():
    calls = [_call("resolve_player", AMBIGUOUS, name="Kim")]
    assert grade(_case("clarify"), "Do you mean Si Woo Kim or Tom Kim?", calls, None)[0]
    assert not grade(_case("clarify"), "Si Woo Kim is 1st.", calls, None)[0]
    used = calls + [_call("explain_prediction", {"player": {}}, player_id="37455")]
    assert not grade(_case("clarify"), "Si Woo Kim or Tom Kim?", used, None)[0]
    assert not grade(_case("clarify"), "Which Kim?", [], None)[0]          # never looked it up


def test_refusal_and_unavailable():
    nobody = [_call("resolve_player", {"error": "player_not_found"}, name="Zylo")]
    assert grade(_case("refusal_or_clarify"), "I couldn't find a player by that name.", nobody, None)[0]
    assert not grade(_case("refusal_or_clarify"), "He should do well.", nobody, None)[0]

    changed = [_call("explain_prediction", {"error": "model_changed"})]
    case = _case("unavailable", error="model_changed")
    assert grade(case, "That explanation isn't available: the model was retrained.", changed, None)[0]
    assert not grade(case, "He leads on approach play.", changed, None)[0]
    assert not grade(case, "That isn't available.", [], None)[0]


def test_infeasible():
    err = [_call("propose_lineups", {"error": "infeasible", "detail": "over budget"})]
    assert grade(_case("infeasible"), "Those three cost more than the $1,000,000 budget.", err, None)[0]
    assert not grade(_case("infeasible"), "Here's a lineup.", err + [_call("propose_lineups", {"lineups": []})], None)[0]
    assert not grade(_case("infeasible"), "It's over budget.", [], None)[0]


def test_lineup_valid_checks_the_users_constraints_not_the_agents(con):
    wanted = {"n_lineups": 2, "locks": ["37455"], "excludes": [], "budget": 3_000_000}
    case = _case("lineup_valid", **wanted)
    out = run_tool(con, "propose_lineups", {"n_lineups": 2, "locks": ["37455"], "budget": 3_000_000})
    assert grade(case, "", [_call("propose_lineups", out)], con)[0]
    # The agent forgot the lock: the lineups may still be valid for what it asked, but not for the user.
    forgot = run_tool(con, "propose_lineups", {"n_lineups": 2, "budget": 3_000_000, "excludes": ["37455"]})
    ok, reason = grade(case, "", [_call("propose_lineups", forgot)], con)
    assert not ok and "missing locked" in reason
    one = run_tool(con, "propose_lineups", {"n_lineups": 1, "locks": ["37455"], "budget": 3_000_000})
    assert "wanted 2" in grade(case, "", [_call("propose_lineups", one)], con)[1]


def test_text_no_unsupported_and_all():
    case = _case("text", any_of=["start"], none_of_regex=[r"(?<!\d)16[- ]rounds?\b"])
    assert grade(case, "Over the last 16 starts.", [], None)[0]
    assert not grade(case, "Over 16 rounds.", [], None)[0]
    assert not grade(case, "Recently.", [], None)[0]
    assert grade(_case("no_unsupported"), "x", [], None, unsupported=[])[0]
    assert "0.52" in grade(_case("no_unsupported"), "x", [], None, unsupported=["0.52"])[1]
    both = _case("all", checks=[{"type": "exact", "value": "Si Woo Kim"},
                                {"type": "number", "values": [5], "tol": 0}])
    assert grade(both, "Si Woo Kim is #1; Scheffler is 5th.", [], None)[0]
    ok, reason = grade(both, "Scheffler is 5th.", [], None)
    assert not ok and "Si Woo Kim" in reason


def test_tools_ok():
    case = {**_case("exact", value="x"), "expected_tools": ["resolve_player", "get_rankings"]}
    assert tools_ok(case, [_call("resolve_player", {}), _call("get_rankings", {}), _call("get_rankings", {})])
    assert not tools_ok(case, [_call("get_rankings", {})])


# ── Runner ─────────────────────────────────────────────────────────────────────

def test_runner_writes_reports(tmp_path, monkeypatch):
    cases = [
        {"id": "a", "category": "lookup", "question": "Who is #1?", "expected_tools": ["get_rankings"],
         "expected": {"type": "exact", "value": "Si Woo Kim"}},
        {"id": "b", "category": "guardrail", "question": "How is Kim?", "expected_tools": ["resolve_player"],
         "expected": {"type": "clarify"}},
    ]
    qfile = tmp_path / "questions.jsonl"
    qfile.write_text("\n".join(json.dumps(c) for c in cases))
    monkeypatch.setattr(run_evals, "EVAL_QUESTIONS", qfile)
    monkeypatch.setattr(run_evals, "EVAL_RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(run_evals, "connect", lambda: None)
    monkeypatch.setattr(run_evals.anthropic, "Anthropic", lambda **kw: None)

    def fake_run(question, **kwargs):
        calls = [ToolCall("get_rankings", {}, {"players": []})]
        return AgentResult(answer="Si Woo Kim is #1 with 0.98.", stop_reason="end_turn", iterations=2,
                           tool_calls=calls, input_tokens=10_000, output_tokens=1_000, run_id="r",
                           unsupported_numbers=["0.98"])
    monkeypatch.setattr(run_evals, "run", fake_run)

    assert run_evals.main(["--model", "claude-opus-5-5"]) == 0
    (report,) = (tmp_path / "results").glob("*.json")
    data = json.loads(report.read_text())
    s = data["summary"]
    assert (s["cases"], s["passed"], s["accuracy"]) == (2, 1, 0.5)
    assert s["by_category"] == {"lookup": {"cases": 1, "accuracy": 1.0}, "guardrail": {"cases": 1, "accuracy": 0.0}}
    assert (s["tool_call_correctness"], s["unsupported_number_rate"]) == (0.5, 1.0)
    assert s["avg_cost_usd"] == pytest.approx((10_000 * 4 + 1_000 * 20) / 1e6)
    md = report.with_suffix(".md").read_text()
    assert "1/2 passed (50%)" in md and "### b: How is Kim?" in md and "**FAIL**" in md


def test_multi_turn_case_carries_history_and_checks_the_first_reply(monkeypatch):
    case = {"id": "m", "category": "guardrail", "question": "Where is Kim?", "follow_ups": ["Tom"],
            "expected_tools": ["resolve_player", "get_rankings"], "hard": True,
            "expected_first": {"type": "clarify"}, "expected": {"type": "number", "values": [4], "tol": 0}}
    seen = []

    def fake_run(question, history=None, **kwargs):
        seen.append((question, len(history)))
        if question == "Where is Kim?":
            calls = [ToolCall("resolve_player", {}, AMBIGUOUS)]
            answer = first_answer[0]
        else:
            calls, answer = [ToolCall("get_rankings", {}, {})], "Tom Kim is 4th."
        return AgentResult(answer=answer, stop_reason="end_turn", iterations=1, tool_calls=calls,
                           input_tokens=1, output_tokens=1, run_id=question[:3],
                           messages=list(history) + [{"role": "user"}, {"role": "assistant"}])
    monkeypatch.setattr(run_evals, "run", fake_run)

    first_answer = ["Do you mean Si Woo Kim or Tom Kim?"]
    row = run_evals.run_case(case, None, None, "claude-haiku-5-5", "s")
    assert seen == [("Where is Kim?", 0), ("Tom", 2)]
    assert row["passed"] and row["tools_ok"] and row["iterations"] == 2 and row["hard"]
    assert row["run_ids"] == ["Whe", "Tom"] and "[Tom]\nTom Kim is 4th." in row["answer"]

    first_answer[0] = "Si Woo Kim is 1st."           # answered without asking
    row = run_evals.run_case(case, None, None, "claude-haiku-5-5", "s")
    assert not row["passed"] and row["reason"].startswith("turn 1:")
