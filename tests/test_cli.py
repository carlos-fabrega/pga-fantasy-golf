"""
Tests for the agent CLI, with the agent loop replaced by a stub.

Run:  python -m pytest tests/test_cli.py
"""

from __future__ import annotations

import json

import anthropic
import httpx
import pandas as pd
import pytest

from agent import cli
from agent.agent import AgentResult, ToolCall


@pytest.fixture()
def turns(monkeypatch):
    """Stub agent.run: record each call, reply with a two-message turn."""
    calls = []

    def fake_run(question, history=None, conversation_id=None, **kwargs):
        calls.append({"question": question, "history": list(history), "conversation_id": conversation_id})
        if question == "boom":
            raise anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com"))
        return AgentResult(answer=f"answer to {question}", stop_reason="end_turn", iterations=1,
                           tool_calls=[ToolCall("resolve_player", {}, {}), ToolCall("resolve_player", {}, {})],
                           run_id="r1", messages=list(history) + [{"role": "user"}, {"role": "assistant"}])

    monkeypatch.setattr(cli, "run", fake_run)
    return calls


def _type(monkeypatch, *lines):
    it = iter(lines)

    def fake_input(prompt=""):
        try:
            return next(it)
        except StopIteration:
            raise EOFError
    monkeypatch.setattr("builtins.input", fake_input)


def test_follow_ups_share_history_until_new(monkeypatch, capsys, turns):
    _type(monkeypatch, "How is Smith doing?", "Cameron", "/new", "Top 5?", "/quit", "never asked")
    cli.repl()
    assert [c["question"] for c in turns] == ["How is Smith doing?", "Cameron", "Top 5?"]
    assert [len(c["history"]) for c in turns] == [0, 2, 0]
    ids = [c["conversation_id"] for c in turns]
    assert ids[0] == ids[1] != ids[2]
    out = capsys.readouterr().out
    assert "answer to Cameron" in out and "resolve_player x2" in out


def test_api_error_is_reported_and_history_kept(monkeypatch, capsys, turns):
    _type(monkeypatch, "Top 5?", "boom", "And 6th?")       # then EOF ends the loop
    cli.repl()
    assert [len(c["history"]) for c in turns] == [0, 2, 2]
    assert "API error" in capsys.readouterr().out


def test_unknown_command_prints_help(monkeypatch, capsys, turns):
    _type(monkeypatch, "/help")
    cli.repl()
    assert turns == [] and capsys.readouterr().out.count("/new starts over") == 2


def test_missing_api_key(monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert cli.main(["Top 5?"]) == 1
    assert "ANTHROPIC_API_KEY is not set" in capsys.readouterr().out


# ── Lineup approval ────────────────────────────────────────────────────────────

PROPOSAL = {
    "proposal_id": "abc12345", "status": "proposed: not saved",
    "event": {"event_id": "R2026060", "name": "TOUR Championship", "season": 2026},
    "model_version": "nocut", "budget": 1_650_000, "locks": ["Scottie Scheffler"], "excludes": [],
    "lineups": [{"players": [{"player_id": str(i), "name": f"Player {i}", "salary": 200_000 + i,
                              "model_rank": i, "score": 0.5} for i in range(6)],
                 "total_salary": 1_200_015, "budget_left": 449_985, "total_score": 3.0}] * 2,
}


@pytest.fixture()
def folders(tmp_path, monkeypatch):
    week = tmp_path / "processed" / "R2026060_tour_championship"
    week.mkdir(parents=True)
    monkeypatch.setattr(cli, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(cli, "TRACES_DIR", tmp_path / "traces")
    monkeypatch.setattr(cli, "validate_proposal", lambda con, args, proposal: [])   # tested in test_guardrails
    return week, tmp_path / "traces"


def _proposing(monkeypatch, *outputs):
    def fake_run(question, history=None, **kwargs):
        return AgentResult(answer="Here are 2 proposed lineups.", stop_reason="end_turn", iterations=2,
                           tool_calls=[ToolCall("propose_lineups", {}, o) for o in outputs],
                           run_id="r9", messages=[])
    monkeypatch.setattr(cli, "run", fake_run)


def _approvals(traces):
    lines = [json.loads(l) for f in traces.glob("*.jsonl") for l in f.read_text().splitlines()]
    return [(e["proposal_id"], e["approved"]) for e in lines if e["event"] == "approval"]


def test_approved_proposal_is_saved_in_the_event_folder(monkeypatch, capsys, folders):
    week, traces = folders
    _proposing(monkeypatch, PROPOSAL)
    _type(monkeypatch, "Build 2 lineups with Scheffler", "y")
    cli.repl()
    saved = pd.read_csv(week / "lineups_approved_abc12345.csv", dtype={"player_id": str})
    assert len(saved) == 12 and sorted(saved["lineup"].unique()) == [1, 2]
    assert set(saved["player_id"]) == {str(i) for i in range(6)}
    assert (saved["model_version"] == "nocut").all() and saved["approved_at"].notna().all()
    out = capsys.readouterr().out
    assert "Lineup 2: $1,200,015" in out and "Locked: Scottie Scheffler" in out
    assert _approvals(traces) == [("abc12345", True)]


@pytest.mark.parametrize("reply", ["n", "", "maybe"])
def test_anything_but_yes_saves_nothing(monkeypatch, capsys, folders, reply):
    week, traces = folders
    _proposing(monkeypatch, PROPOSAL)
    _type(monkeypatch, "Build lineups", reply)
    cli.repl()
    assert list(week.iterdir()) == []
    assert "Not saved." in capsys.readouterr().out
    assert _approvals(traces) == [("abc12345", False)]


def test_end_of_input_at_the_prompt_saves_nothing(monkeypatch, folders):
    week, _ = folders
    _proposing(monkeypatch, PROPOSAL)
    _type(monkeypatch, "Build lineups")            # EOF when asked to approve
    cli.repl()
    assert list(week.iterdir()) == []


def test_failed_proposals_are_not_offered(monkeypatch, capsys, folders):
    _, traces = folders
    _proposing(monkeypatch, {"error": "infeasible", "detail": "Over budget."})
    _type(monkeypatch, "Build lineups")
    cli.repl()
    assert "Approve" not in capsys.readouterr().out and not traces.exists()


def test_invalid_proposal_is_shown_but_never_offered(monkeypatch, capsys, folders):
    week, traces = folders
    monkeypatch.setattr(cli, "validate_proposal", lambda con, args, proposal: ["Lineup 1 costs too much."])
    _proposing(monkeypatch, PROPOSAL)
    _type(monkeypatch, "Build lineups")
    cli.repl()
    out = capsys.readouterr().out
    assert "failed validation" in out and "Lineup 1 costs too much." in out and "Approve" not in out
    assert list(week.iterdir()) == [] and _approvals(traces) == [("abc12345", False)]


def test_unsupported_numbers_are_shown(monkeypatch, capsys, turns):
    def fake_run(question, history=None, **kwargs):
        return AgentResult(answer="Gap is 0.73.", stop_reason="end_turn", iterations=1,
                           run_id="r1", unsupported_numbers=["0.73"])
    monkeypatch.setattr(cli, "run", fake_run)
    _type(monkeypatch, "Why?")
    cli.repl()
    assert "[check: not found in any tool output: 0.73]" in capsys.readouterr().out


def test_one_shot_mode_asks_too(monkeypatch, folders):
    week, _ = folders
    _proposing(monkeypatch, PROPOSAL)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setattr(cli, "connect", lambda: None)
    _type(monkeypatch, "yes")
    assert cli.main(["Build lineups"]) == 0
    assert (week / "lineups_approved_abc12345.csv").exists()
