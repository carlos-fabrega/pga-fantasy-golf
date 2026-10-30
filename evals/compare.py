"""
Compare eval runs before and after a change, averaged over repeated runs.

    python -m evals.compare --before evals/results/A.json evals/results/B.json \
                            --after  evals/results/C.json evals/results/D.json

Prints a Markdown summary (mean accuracy, hard-case accuracy, unsupported-number
rate, cost, latency) and every case whose pass count changed, so a one-case
improvement can be told apart from run-to-run noise. Paste it into
evals/CHANGELOG.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional


def _load(paths: List[Path]) -> List[dict]:
    return [json.loads(p.read_text()) for p in paths]


def _mean(xs) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _passes(runs: List[dict]) -> Dict[str, int]:
    """How many of the runs each case passed in."""
    counts: Dict[str, int] = {}
    for run in runs:
        for r in run["cases"]:
            counts[r["id"]] = counts.get(r["id"], 0) + bool(r["passed"])
    return counts


def compare(before: List[dict], after: List[dict]) -> str:
    def metric(runs, fn):
        return _mean([fn(r["summary"]) for r in runs])

    rows = [
        ("Accuracy", lambda s: s["accuracy"], "{:.1%}"),
        ("Hard cases", lambda s: s["hard"]["accuracy"], "{:.1%}"),
        ("Answers with unsupported numbers", lambda s: s["unsupported_number_rate"], "{:.1%}"),
        ("Avg model calls", lambda s: s["avg_iterations"], "{:.2f}"),
        ("Avg latency (s)", lambda s: s["avg_latency_s"], "{:.1f}"),
        ("Avg cost per question", lambda s: s["avg_cost_usd"], "${:.4f}"),
    ]
    model = {r["model"] for r in before + after}
    lines = [f"Model: {', '.join(sorted(model))}. Means over {len(before)} runs before and {len(after)} after.", "",
             "| Metric | Before | After |", "|---|---|---|"]
    for name, fn, fmt in rows:
        b, a = metric(before, fn), metric(after, fn)
        lines.append(f"| {name} | {'n/a' if b is None else fmt.format(b)} | {'n/a' if a is None else fmt.format(a)} |")
    pb, pa = _passes(before), _passes(after)
    changed = [(cid, pb.get(cid, 0), pa.get(cid, 0)) for cid in sorted(set(pb) | set(pa))
               if pb.get(cid, 0) / len(before) != pa.get(cid, 0) / len(after)]
    lines += ["", "Cases whose pass count changed:", ""]
    lines += ([f"- {cid}: {b}/{len(before)} → {a}/{len(after)}" for cid, b, a in changed] or ["- none"])
    lines += ["", "Runs: before " + ", ".join(f"`{r['stamp']}`" for r in before)
              + "; after " + ", ".join(f"`{r['stamp']}`" for r in after) + "."]
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Compare eval runs before and after a change")
    parser.add_argument("--before", nargs="+", type=Path, required=True)
    parser.add_argument("--after", nargs="+", type=Path, required=True)
    args = parser.parse_args(argv)
    print(compare(_load(args.before), _load(args.after)), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
