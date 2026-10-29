"""
Run the agent on every eval case, grade it, and write a report.

    python -m evals.run_evals                              # all cases, default model (Opus 5.5)
    python -m evals.run_evals --model claude-haiku-5-5     # cheaper model
    python -m evals.run_evals --ids rank-01,lu-02          # just these cases
    python -m evals.run_evals --category lineup            # one category

Every run calls the API once per case (several model calls each), so it costs
money: about $0.0013 a case on Haiku 5.5 (measured, Oct 2026) and roughly
$0.05-0.10 on Opus 5.5 (estimated from single traced runs). Reports land in evals/results/{timestamp}_{model}.json and .md; each
case's trace is in traces/ under conversation_id eval-{timestamp}-{case id}.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from collections import defaultdict
from typing import Dict, List, Optional

import anthropic

from agent.agent import MODEL, TIMEOUT_S, run
from agent.db import connect
from evals.grade import grade, tools_ok
from src.paths import EVAL_QUESTIONS, EVAL_RESULTS_DIR

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


def run_case(case: dict, client, con, model: str, stamp: str) -> dict:
    t0 = time.monotonic()
    row = {"id": case["id"], "category": case["category"], "type": case["expected"]["type"],
           "question": case["question"]}
    try:
        res = run(case["question"], client=client, con=con, model=model,
                  conversation_id=f"eval-{stamp}-{case['id']}")
    except anthropic.APIError as e:
        return {**row, "passed": False, "reason": f"API error: {e}", "error": True,
                "latency_s": round(time.monotonic() - t0, 1)}
    passed, reason = grade(case, res.answer, res.tool_calls, con)
    return {**row, "passed": passed, "reason": reason, "error": False,
            "tools_ok": tools_ok(case, res.tool_calls), "tools": [c.name for c in res.tool_calls],
            "stop_reason": res.stop_reason, "iterations": res.iterations,
            "latency_s": round(time.monotonic() - t0, 1),
            "input_tokens": res.input_tokens, "output_tokens": res.output_tokens,
            "cost_usd": _cost(model, res.input_tokens, res.output_tokens),
            "unsupported_numbers": res.unsupported_numbers, "run_id": res.run_id, "answer": res.answer}


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
        f"{s['passed']}/{s['cases']} passed ({_pct(s['accuracy'])})"
        + (f", {s['api_errors']} API errors" if s["api_errors"] else "") + ".",
        "",
        "| Metric | Value |", "|---|---|",
        f"| Accuracy | {_pct(s['accuracy'])} |",
        *[f"| {c} | {_pct(v['accuracy'])} ({v['cases']} cases) |" for c, v in s["by_category"].items()],
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
        lines.append(f"| {r['id']} | {'pass' if r['passed'] else '**FAIL**'} | "
                     f"{'' if r['error'] else ('yes' if r['tools_ok'] else 'no')} | "
                     f"{', '.join(r.get('unsupported_numbers') or [])} | {r.get('iterations', '')} | "
                     f"{r['reason']} |")
    failures = [r for r in rows if not r["passed"]]
    if failures:
        lines += ["", "## Failures", ""]
        for r in failures:
            lines += [f"### {r['id']}: {r['question']}", "",
                      f"**Why:** {r['reason']}. Tools: {', '.join(r.get('tools', [])) or 'none'}. "
                      f"Trace: `{r.get('run_id', '')}`.", "", "> " + (r.get("answer") or "").replace("\n", "\n> "), ""]
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the PGA agent evals")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--ids", help="Comma-separated case ids")
    parser.add_argument("--category", choices=CATEGORIES)
    args = parser.parse_args(argv)

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
