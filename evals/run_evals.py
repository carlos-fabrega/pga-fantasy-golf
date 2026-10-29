"""
Run the agent on every eval case, grade it, and write a report.

    python -m evals.run_evals                              # all cases, default model (Opus 5.5)
    python -m evals.run_evals --model claude-haiku-5-5     # cheaper model
    python -m evals.run_evals --ids rank-01,lu-02          # just these cases
    python -m evals.run_evals --category lineup            # one category
    python -m evals.run_evals --rescore evals/results/20261028-220001_claude-opus-5-5.json

Every run calls the API once per case (several model calls each), so it costs
money: about $0.0014 a case on Haiku 5.5 and $0.055 on Opus 5.5 (measured,
Oct 2026). Reports land in evals/results/{timestamp}_{model}.json and .md; each
case's trace is in traces/ under conversation_id eval-{timestamp}-{case id}.

--rescore re-grades a saved run from its traces with the current graders and
numeric checker, without calling the API, and writes {report}_rescored.json/.md.
Use it after changing a grader or the checker, so the change is measured on the
same answers rather than on a fresh, noisy run.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import anthropic

from agent.agent import MODEL, TIMEOUT_S, ToolCall, run
from agent.db import connect
from agent.guardrails import check_numbers
from evals.grade import grade, tools_ok
from src.paths import EVAL_QUESTIONS, EVAL_RESULTS_DIR, TRACES_DIR

# $ per million input / output tokens (Claude API list prices, Oct 2026). Haiku's
# rate is for prompts up to 100K tokens; these runs stay far below that.
PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}
CATEGORIES = ["lookup", "explanation", "value", "lineup", "guardrail"]


def load_cases() -> List[dict]:
    with open(EVAL_QUESTIONS) as f:
        return [json.loads(line) for line in f if line.strip()]


def _cost(model: str, tokens_in: int, tokens_out: int) -> Optional[float]:
    if model not in PRICES:
        return None
    p_in, p_out = PRICES[model]
    return (tokens_in * p_in + tokens_out * p_out) / 1e6


def _grade_turns(case: dict, answers: List[str], turn_calls: List[List[ToolCall]],
                 flags: List[List[str]], con) -> tuple:
    """Grade the last answer (with every turn's tool calls), plus the first reply if the case checks it."""
    calls = [c for tc in turn_calls for c in tc]
    passed, reason = grade(case, answers[-1], calls, con, flags[-1])
    if "expected_first" in case:   # e.g. the agent must ask "which Kim?" before answering
        ok, why = grade(case, answers[0], turn_calls[0], con, flags[0], key="expected_first")
        if not ok:
            passed, reason = False, "; ".join(x for x in (f"turn 1: {why}", reason) if x)
    return passed, reason


def run_case(case: dict, client, con, model: str, stamp: str) -> dict:
    """Run a case's question and any follow_ups as one conversation, then grade the last answer."""
    t0 = time.monotonic()
    row = {"id": case["id"], "category": case["category"], "type": case["expected"]["type"],
           "hard": case.get("hard", False), "question": case["question"]}
    turns = [case["question"], *case.get("follow_ups", [])]
    history, calls, answers, results = [], [], [], []
    try:
        for q in turns:
            res = run(q, history=history, client=client, con=con, model=model,
                      conversation_id=f"eval-{stamp}-{case['id']}")
            history = res.messages
            calls += res.tool_calls
            answers.append(res.answer)
            results.append(res)
    except anthropic.APIError as e:
        return {**row, "passed": False, "reason": f"API error: {e}", "error": True,
                "latency_s": round(time.monotonic() - t0, 1)}
    last = results[-1]
    passed, reason = _grade_turns(case, answers, [r.tool_calls for r in results],
                                  [r.unsupported_numbers for r in results], con)
    tokens_in, tokens_out = sum(r.input_tokens for r in results), sum(r.output_tokens for r in results)
    return {**row, "passed": passed, "reason": reason, "error": False,
            "tools_ok": tools_ok(case, calls), "tools": [c.name for c in calls],
            "stop_reason": last.stop_reason, "iterations": sum(r.iterations for r in results),
            "latency_s": round(time.monotonic() - t0, 1),
            "input_tokens": tokens_in, "output_tokens": tokens_out,
            "cost_usd": _cost(model, tokens_in, tokens_out),
            "unsupported_numbers": [n for r in results for n in r.unsupported_numbers],
            "run_ids": [r.run_id for r in results],
            "answer": "\n\n".join(f"[{q}]\n{a}" for q, a in zip(turns, answers)) if len(turns) > 1 else last.answer}


def _load_traces(run_ids: List[str]) -> Dict[str, List[dict]]:
    wanted, events = set(run_ids), defaultdict(list)
    for path in sorted(TRACES_DIR.glob("*.jsonl")):
        with path.open() as f:
            for line in f:
                e = json.loads(line)
                if e["run_id"] in wanted:
                    events[e["run_id"]].append(e)
    return events


def rescore_row(case: Optional[dict], row: dict, events: Dict[str, List[dict]], con) -> dict:
    """Re-grade one saved case from its traces with today's graders and checker. No API calls."""
    if row["error"]:
        return row
    if case is None or case["question"] != row["question"]:
        return {**row, "reason": f"{row['reason']} (case changed since this run; not re-scored)".strip()}
    run_ids = row.get("run_ids") or [row["run_id"]]
    if not all(any(e["event"] == "end" for e in events.get(r, [])) for r in run_ids):
        return {**row, "reason": f"{row['reason']} (trace missing; not re-scored)".strip()}
    # Rebuild what the checker saw live: every question, tool input and tool result so far.
    messages: List[dict] = []
    answers, turn_calls, flags = [], [], []
    for rid in run_ids:
        calls = []
        for e in events[rid]:
            if e["event"] == "start":
                messages.append({"role": "user", "content": e["question"]})
            elif e["event"] == "tool_call":
                messages.append({"role": "assistant", "content": [{"type": "tool_use", "input": e["input"]}]})
                messages.append({"role": "user", "content": [{"type": "tool_result",
                                                              "content": json.dumps(e["output"])}]})
                calls.append(ToolCall(e["name"], e["input"], e["output"]))
            elif e["event"] == "end":
                answers.append(e["answer"])
                flags.append([] if e["stop_reason"] in ("max_iterations", "timeout", "refusal")
                             else check_numbers(e["answer"], messages))
        turn_calls.append(calls)
    passed, reason = _grade_turns(case, answers, turn_calls, flags, con)
    all_calls = [c for tc in turn_calls for c in tc]
    return {**row, "passed": passed, "reason": reason, "tools_ok": tools_ok(case, all_calls),
            "unsupported_numbers": [n for f in flags for n in f]}


def rescore(report_path: Path) -> Path:
    """Write {report}_rescored.json/.md next to a saved report and return the .md path."""
    old = json.loads(report_path.read_text())
    cases = {c["id"]: c for c in load_cases()}
    events = _load_traces([r for row in old["cases"] for r in (row.get("run_ids") or [row.get("run_id")]) if r])
    con = connect()
    rows = [rescore_row(cases.get(row["id"]), row, events, con) for row in old["cases"]]
    for before, after in zip(old["cases"], rows):
        if before["passed"] != after["passed"] or before.get("unsupported_numbers") != after.get("unsupported_numbers"):
            print(f"{after['id']:<9} {'pass' if before['passed'] else 'FAIL'} -> {'pass' if after['passed'] else 'FAIL'}"
                  f"  flags {before.get('unsupported_numbers')} -> {after.get('unsupported_numbers')}")
    meta = {k: v for k, v in old.items() if k not in ("summary", "cases")}
    meta["rescored_from"] = report_path.name
    summary = summarize(rows)
    base = report_path.with_name(report_path.stem + "_rescored")
    base.with_suffix(".json").write_text(json.dumps({**meta, "summary": summary, "cases": rows}, indent=1))
    base.with_suffix(".md").write_text(report_md(meta, summary, rows))
    print(f"{old['summary']['passed']}/{old['summary']['cases']} -> {summary['passed']}/{summary['cases']} passed; "
          f"unsupported-number rate {_pct(old['summary']['unsupported_number_rate'])} -> "
          f"{_pct(summary['unsupported_number_rate'])}. Report: {base.with_suffix('.md')}")
    return base.with_suffix(".md")


def _mean(xs) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def summarize(rows: List[dict]) -> dict:
    ran = [r for r in rows if not r["error"]]
    by_cat: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    costs = [r.get("cost_usd") for r in ran]
    return {
        "cases": len(rows), "passed": sum(r["passed"] for r in rows), "api_errors": len(rows) - len(ran),
        "accuracy": _mean([r["passed"] for r in rows]),
        "by_category": {c: {"cases": len(rs), "accuracy": _mean([r["passed"] for r in rs])}
                        for c in CATEGORIES for rs in [by_cat.get(c, [])] if rs},
        "hard": {"cases": sum(r["hard"] for r in rows), "accuracy": _mean([r["passed"] for r in rows if r["hard"]])},
        "tool_call_correctness": _mean([r["tools_ok"] for r in ran]),
        # Share of answers where the numeric-claim checker flagged at least one number.
        "unsupported_number_rate": _mean([bool(r["unsupported_numbers"]) for r in ran]),
        "avg_iterations": _mean([r["iterations"] for r in ran]),
        "avg_latency_s": _mean([r["latency_s"] for r in ran]),
        "avg_cost_usd": _mean(costs),
        "total_cost_usd": sum(c for c in costs if c is not None) if any(c is not None for c in costs) else None,
    }


def _pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def report_md(meta: dict, summary: dict, rows: List[dict]) -> str:
    s = summary
    money = lambda x: "n/a" if x is None else f"${x:.4f}"   # noqa: E731
    calls = "n/a" if s["avg_iterations"] is None else f"{s['avg_iterations']:.1f}"
    latency = "n/a" if s["avg_latency_s"] is None else f"{s['avg_latency_s']:.1f}s"
    lines = [
        f"# Eval run {meta['stamp']}: {meta['model']}",
        "",
        *([f"Re-scored from `{meta['rescored_from']}` with the current graders and checker (same answers).", ""]
          if meta.get("rescored_from") else []),
        f"{s['passed']}/{s['cases']} passed ({_pct(s['accuracy'])})"
        + (f", {s['api_errors']} API errors" if s["api_errors"] else "") + ".",
        "",
        "| Metric | Value |", "|---|---|",
        f"| Accuracy | {_pct(s['accuracy'])} |",
        *[f"| {c} | {_pct(v['accuracy'])} ({v['cases']} cases) |" for c, v in s["by_category"].items()],
        f"| hard cases | {_pct(s['hard']['accuracy'])} ({s['hard']['cases']} cases) |",
        f"| Tool-call correctness | {_pct(s['tool_call_correctness'])} |",
        f"| Answers with unsupported numbers | {_pct(s['unsupported_number_rate'])} |",
        f"| Avg model calls | {calls} |",
        f"| Avg latency | {latency} |",
        f"| Avg cost per question | {money(s['avg_cost_usd'])} |",
        f"| Total cost | {money(s['total_cost_usd'])} |",
        "",
        "## Cases",
        "",
        "| Case | Result | Tools OK | Unsupported numbers | Calls | Reason |", "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['id']}{' (hard)' if r['hard'] else ''} | {'pass' if r['passed'] else '**FAIL**'} | "
                     f"{'' if r['error'] else ('yes' if r['tools_ok'] else 'no')} | "
                     f"{', '.join(r.get('unsupported_numbers') or [])} | {r.get('iterations', '')} | "
                     f"{r['reason']} |")
    failures = [r for r in rows if not r["passed"]]
    if failures:
        lines += ["", "## Failures", ""]
        for r in failures:
            lines += [f"### {r['id']}: {r['question']}", "",
                      f"**Why:** {r['reason']}. Tools: {', '.join(r.get('tools', [])) or 'none'}. "
                      f"Traces: `{', '.join(r.get('run_ids', []))}`.", "", "> " + (r.get("answer") or "").replace("\n", "\n> "), ""]
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the PGA agent evals")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--ids", help="Comma-separated case ids")
    parser.add_argument("--category", choices=CATEGORIES)
    parser.add_argument("--rescore", type=Path, metavar="REPORT_JSON",
                        help="Re-grade a saved run from its traces (no API calls)")
    args = parser.parse_args(argv)
    if args.rescore:
        rescore(args.rescore)
        return 0

    cases = load_cases()
    if args.ids:
        wanted = set(args.ids.split(","))
        cases = [c for c in cases if c["id"] in wanted]
    if args.category:
        cases = [c for c in cases if c["category"] == args.category]
    if not cases:
        print("No cases match.")
        return 1

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    client, con = anthropic.Anthropic(timeout=TIMEOUT_S), connect()
    rows = []
    for i, case in enumerate(cases, 1):
        row = run_case(case, client, con, args.model, stamp)
        rows.append(row)
        print(f"[{i}/{len(cases)}] {case['id']:<9} {'pass' if row['passed'] else 'FAIL'}  {row['reason']}")

    summary = summarize(rows)
    meta = {"stamp": stamp, "model": args.model, "questions": str(EVAL_QUESTIONS.name),
            "case_ids": [c["id"] for c in cases]}
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    base = EVAL_RESULTS_DIR / f"{stamp}_{args.model}"
    base.with_suffix(".json").write_text(json.dumps({**meta, "summary": summary, "cases": rows}, indent=1))
    base.with_suffix(".md").write_text(report_md(meta, summary, rows))
    print(f"\n{summary['passed']}/{summary['cases']} passed. Report: {base.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
