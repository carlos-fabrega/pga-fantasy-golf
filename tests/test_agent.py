"""
Tests for the agent loop, with a scripted fake client in place of the API.

Run:  python -m pytest tests/test_agent.py
"""

from __future__ import annotations

import json

import pytest
from anthropic.types.beta import (
    BetaMessage, BetaTextBlock, BetaThinkingBlock, BetaToolUseBlock, BetaUsage,
)

from agent import agent
from agent.prompts import SYSTEM_PROMPT
from agent.tools import TOOLS


def _msg(*blocks, stop="end_turn", tokens=(100, 20)):
    return BetaMessage.construct(
        id="msg", type="message", role="assistant", model="claude-opus-5-5",
        content=list(blocks), stop_reason=stop, stop_sequence=None,
        usage=BetaUsage.construct(input_tokens=tokens[0], output_tokens=tokens[1]),
    )


def _text(t):
    return BetaTextBlock.construct(type="text", text=t, citations=None)


def _tool(id_, tool, **inp):
    return BetaToolUseBlock.construct(type="tool_use", id=id_, name=tool, input=inp)


class FakeClient:
    """Returns scripted responses in order and records every request."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.beta = self
        self.messages = self

    def create(self, **kwargs):
        # Snapshot messages: the loop keeps appending to the same list.
        self.requests.append({**kwargs, "messages": json.loads(json.dumps(kwargs["messages"]))})
        if len(self.responses) > 1:
            return self.responses.pop(0)
        return self.responses[0]   # last response repeats


@pytest.fixture()
def tool_log(monkeypatch):
    calls = []

    def fake_run_tool(con, name, args):
        calls.append((name, args))
        if name == "resolve_player" and args["name"] == "Nobody":
            return {"error": "player_not_found", "detail": "No player matches 'Nobody'."}
        return {"ok": name, "args": args}

    monkeypatch.setattr(agent, "run_tool", fake_run_tool)
    return calls


def _run(client, tmp_path, **kw):
    return agent.run("Who is ranked first?", client=client, con=object(), trace_dir=tmp_path, **kw)


def _trace(tmp_path):
    (path,) = tmp_path.glob("*.jsonl")
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_tool_call_then_answer(tmp_path, tool_log):
    client = FakeClient([
        _msg(_text("Checking."), _tool("t1", "get_rankings", top_n=1), stop="tool_use"),
        _msg(_text("Si Woo Kim is first."), tokens=(300, 10)),
    ])
    res = _run(client, tmp_path)

    assert (res.answer, res.stop_reason, res.iterations) == ("Si Woo Kim is first.", "end_turn", 2)
    assert tool_log == [("get_rankings", {"top_n": 1})]
    assert [c.name for c in res.tool_calls] == ["get_rankings"]
    assert (res.input_tokens, res.output_tokens) == (400, 30)

    # Second request: question, assistant turn with the tool_use, then its result.
    msgs = client.requests[1]["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert msgs[1]["content"][1]["id"] == "t1"
    (result,) = msgs[2]["content"]
    assert result["tool_use_id"] == "t1" and result["is_error"] is False
    assert json.loads(result["content"]) == {"ok": "get_rankings", "args": {"top_n": 1}}


def test_request_carries_prompt_tools_and_config(tmp_path, tool_log):
    client = FakeClient([_msg(_text("Hi."))])
    _run(client, tmp_path, model="claude-test")
    req = client.requests[0]
    assert req["model"] == "claude-test"
    assert req["system"] == SYSTEM_PROMPT and req["tools"] == TOOLS
    assert req["betas"] == agent.BETAS
    assert req["extra_body"]["output_config"] == {"effort": agent.EFFORT}


def test_parallel_tool_calls_return_in_one_message_with_errors_flagged(tmp_path, tool_log):
    client = FakeClient([
        _msg(_tool("a", "resolve_player", name="Scheffler"),
             _tool("b", "resolve_player", name="Nobody"), stop="tool_use"),
        _msg(_text("Done.")),
    ])
    _run(client, tmp_path)
    results = client.requests[1]["messages"][2]["content"]
    assert [r["tool_use_id"] for r in results] == ["a", "b"]
    assert [r["is_error"] for r in results] == [False, True]


def test_thinking_blocks_are_sent_back_and_unknown_blocks_dropped(tmp_path, tool_log):
    thinking = BetaThinkingBlock.construct(type="thinking", thinking="", signature="sig123")
    unknown = BetaTextBlock.construct(type="fallback", text=None, citations=None)
    client = FakeClient([
        _msg(thinking, unknown, _tool("t1", "get_rankings"), stop="tool_use"),
        _msg(_text("Done.")),
    ])
    _run(client, tmp_path)
    sent = client.requests[1]["messages"][1]["content"]
    assert [b["type"] for b in sent] == ["thinking", "tool_use"]
    assert sent[0]["signature"] == "sig123"


def test_iteration_cap(tmp_path, tool_log):
    client = FakeClient([_msg(_tool("t", "get_rankings"), stop="tool_use")])
    res = _run(client, tmp_path, max_iterations=3)
    assert (res.stop_reason, res.iterations, len(client.requests)) == ("max_iterations", 3, 3)
    assert "stopped before finishing" in res.answer


def test_timeout_stops_before_calling(tmp_path, tool_log):
    client = FakeClient([_msg(_text("Hi."))])
    res = _run(client, tmp_path, timeout_s=0)
    assert (res.stop_reason, res.iterations, client.requests) == ("timeout", 0, [])


def test_trace_logs_every_step(tmp_path, tool_log):
    client = FakeClient([
        _msg(_tool("t1", "get_rankings", top_n=2), stop="tool_use"),
        _msg(_text("Answer.")),
    ])
    res = _run(client, tmp_path)
    events = _trace(tmp_path)
    assert [e["event"] for e in events] == ["start", "model_call", "tool_call", "model_call", "end"]
    assert {e["run_id"] for e in events} == {res.run_id}
    assert events[0]["question"] == "Who is ranked first?"
    tool = events[2]
    assert (tool["name"], tool["input"], tool["is_error"]) == ("get_rankings", {"top_n": 2}, False)
    assert tool["output"] == {"ok": "get_rankings", "args": {"top_n": 2}}
    assert all("latency_ms" in e for e in events if e["event"] in ("model_call", "tool_call"))
    assert events[-1]["answer"] == "Answer."


def test_history_carries_the_conversation_into_the_next_turn(tmp_path, tool_log):
    client = FakeClient([_msg(_text("Which Smith: Cameron or Jordan?")), _msg(_text("Cameron is 4th."))])
    first = agent.run("How is Smith doing?", client=client, con=object(), trace_dir=tmp_path)
    assert [m["role"] for m in first.messages] == ["user", "assistant"]

    second = agent.run("Cameron", history=first.messages, client=client, con=object(), trace_dir=tmp_path)
    sent = client.requests[1]["messages"]
    assert [m["role"] for m in sent] == ["user", "assistant", "user"]
    assert sent[1]["content"][0]["text"] == "Which Smith: Cameron or Jordan?"
    assert sent[2]["content"] == "Cameron"
    assert len(second.messages) == 4
    assert len(first.messages) == 2          # the caller's history isn't mutated


def test_unanswered_tool_use_is_left_out_of_history(tmp_path, tool_log):
    # A response cut off at max_tokens mid tool call must not leave a tool_use
    # without a tool_result, or the next turn's request is invalid.
    client = FakeClient([_msg(_text("Let me check."), _tool("t1", "get_rankings"), stop="max_tokens")])
    res = _run(client, tmp_path)
    assert res.stop_reason == "max_tokens" and tool_log == []
    assert [b["type"] for b in res.messages[-1]["content"]] == ["text"]
