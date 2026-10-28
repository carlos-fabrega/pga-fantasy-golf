"""
Tests for the agent CLI, with the agent loop replaced by a stub.

Run:  python -m pytest tests/test_cli.py
"""

from __future__ import annotations

import anthropic
import httpx
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
