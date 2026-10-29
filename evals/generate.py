"""
Generate the eval cases (evals/questions.jsonl) from the agent database.

Expected answers come from the tools themselves, run against the current
database, so the cases stay correct when new weeks are added: re-run this after
`python -m agent.db`. The tools have their own unit tests; these cases test
whether the agent uses them and reports what they return. The generator stops
with an error if any case's premise doesn't hold (e.g. an "infeasible" lineup
that turns out to fit).

    python -m evals.generate

Case format (one JSON object per line):
    id, category, question, expected_tools (all must be called), expected:
      exact          value: player name the answer must contain
      number         values: numbers the answer must all contain, within tol
      drivers        must_mention: features the explanation must name (or describe)
      topset         names, min_overlap: how many of these names the answer must contain
      lineup_valid   n_lineups, locks, excludes, budget: a proposal meeting them, validated
      infeasible     a propose_lineups 'infeasible' error, reported as such
      clarify        the answer asks which player, naming 2+ candidates
      refusal_or_clarify   no player data for an unknown name; says so or asks
      unavailable    error: the tool error code the answer must report as unavailable
"""

from __future__ import annotations

import json
import sqlite3
from typing import Dict, List

from agent.db import connect
from agent.tools import run_tool
from src.optimizer import DEFAULT_BUDGET
from src.paths import EVAL_QUESTIONS


def _ok(con: sqlite3.Connection, tool: str, **args) -> dict:
    out = run_tool(con, tool, args)
    if "error" in out:
        raise RuntimeError(f"{tool}({args}) failed while generating cases: {out}")
    return out


def _player_id(con: sqlite3.Connection, name: str) -> str:
    out = _ok(con, "resolve_player", name=name)
    if out["status"] != "resolved":
        raise RuntimeError(f"{name!r} doesn't resolve to one player: {out}")
    return out["player_id"]


def _top_drivers(out: dict, n: int = 2) -> List[str]:
    drivers = out["difference"]["top_drivers"] if "difference" in out else out["player"]["top_drivers"]
    return [d["feature"] for d in drivers[:n]]


def build(con: sqlite3.Connection) -> List[Dict]:
    cases: List[Dict] = []

    def add(id_, category, question, tools, **expected):
        cases.append({"id": id_, "category": category, "question": question,
                      "expected_tools": tools, "expected": expected})

    week = _ok(con, "get_rankings", top_n=30)                       # "this week"
    week_name = week["event"]["name"]
    wyndham = _ok(con, "get_rankings", event="Wyndham Championship", season=2026, top_n=5)
    open_ = _ok(con, "get_rankings", event="The Open Championship", season=2026, top_n=5)
    rank = {p["name"]: p for p in week["players"]}
    top1, top2 = week["players"][0], week["players"][1]
    scheffler = _player_id(con, "Scottie Scheffler")

    # ── Lookups ────────────────────────────────────────────────────────────────
    add("rank-01", "lookup", "Who is ranked #1 this week?", ["get_rankings"],
        type="exact", value=top1["name"])
    add("rank-02", "lookup", "Who did the model rank first at the 2026 Wyndham Championship?",
        ["get_rankings"], type="exact", value=wyndham["players"][0]["name"])
    usopen = _ok(con, "get_rankings", event="U.S. Open", season=2019, top_n=1)
    add("rank-03", "lookup", "Who did the model rank first at the 2019 U.S. Open?", ["get_rankings"],
        type="exact", value=usopen["players"][0]["name"])
    add("rank-04", "lookup", "Where is Scottie Scheffler ranked this week?", ["resolve_player", "get_rankings"],
        type="number", values=[rank["Scottie Scheffler"]["rank"]], tol=0)

    f = _ok(con, "get_player_features", player_id=scheffler)["features"]
    add("feat-01", "lookup",
        "What is Scottie Scheffler's strokes gained total per round over the last 16 starts, going into this week?",
        ["resolve_player", "get_player_features"],
        type="number", values=[f["sg_total_per_round_roll16"]], tol=0.0015)
    wy_top = wyndham["players"][0]
    f = _ok(con, "get_player_features", player_id=wy_top["player_id"], event=wyndham["event"]["event_id"])["features"]
    add("feat-02", "lookup",
        f"What was {wy_top['name']}'s putting strokes gained per round over the last 8 starts, "
        f"going into the 2026 Wyndham Championship?",
        ["resolve_player", "get_player_features"],
        type="number", values=[f["sg_putt_per_round_roll8"]], tol=0.0015)

    value_week = _ok(con, "value_report", top_n=50)["players"]
    tom = next(p for p in value_week if p["name"] == "Tom Kim")
    add("sal-01", "lookup", "What is Tom Kim's Buzz salary this week?", ["resolve_player", "value_report"],
        type="number", values=[tom["salary"]], tol=0)
    perf = _ok(con, "model_performance", event=wyndham["event"]["event_id"])
    add("perf-01", "lookup", "What was the model's Spearman rank correlation at the 2026 Wyndham Championship?",
        ["model_performance"], type="number", values=[perf["spearman"]], tol=0.0015)

    # ── Explanations (live weeks whose saved model still reproduces the scores) ─
    def gap_case(id_, question, a, b, event=None):
        out = _ok(con, "explain_prediction", player_id=_player_id(con, a), compare_to=_player_id(con, b),
                  **({"event": event} if event else {}))
        add(id_, "explanation", question, ["resolve_player", "explain_prediction"],
            type="drivers", must_mention=_top_drivers(out))

    def single_case(id_, question, name, event=None):
        out = _ok(con, "explain_prediction", player_id=_player_id(con, name), **({"event": event} if event else {}))
        add(id_, "explanation", question, ["resolve_player", "explain_prediction"],
            type="drivers", must_mention=_top_drivers(out))

    gap_case("expl-01", "Why is Scheffler ranked below Si Woo Kim this week?", "Scottie Scheffler", "Si Woo Kim")
    gap_case("expl-02", "Why does the model rank Patrick Cantlay above Matt Fitzpatrick this week?",
             "Patrick Cantlay", "Matt Fitzpatrick")
    single_case("expl-03", f"Why is {top1['name']} the model's top pick this week?", top1["name"])
    single_case("expl-04", "What's driving Tom Kim's score this week?", "Tom Kim")
    o1, o3 = open_["players"][0]["name"], open_["players"][2]["name"]
    open_id = open_["event"]["event_id"]
    single_case("expl-05", f"Why was {o1} ranked #1 at the 2026 Open Championship?", o1, event=open_id)
    gap_case("expl-06", f"Why did the model rank {o1} above {o3} at the 2026 Open Championship?", o1, o3,
             event=open_id)

    # ── Value ──────────────────────────────────────────────────────────────────
    add("val-01", "value", "Who are the best value plays this week?", ["value_report"],
        type="topset", names=[p["name"] for p in value_week[:5]], min_overlap=4)
    wy_value = _ok(con, "value_report", event=wyndham["event"]["event_id"], top_n=5)["players"]
    add("val-02", "value", "Who were the top 5 value plays at the 2026 Wyndham Championship?", ["value_report"],
        type="topset", names=[p["name"] for p in wy_value], min_overlap=4)
    rocket = _ok(con, "value_report", event="Rocket Classic", season=2026, top_n=1)["players"][0]
    add("val-03", "value", "Who was the single best value play at the 2026 Rocket Classic?", ["value_report"],
        type="exact", value=rocket["name"])
    smalley = next(p for p in value_week if p["name"] == "Alex Smalley")
    add("val-04", "value", "How does Alex Smalley's model rank compare with the salary rank this week?",
        ["resolve_player", "value_report"],
        type="number", values=[smalley["model_rank"], smalley["salary_rank"]], tol=0)

    # ── Lineups ────────────────────────────────────────────────────────────────
    def lineup_case(id_, question, n=1, locks=(), excludes=(), budget=DEFAULT_BUDGET, event=None):
        args = {"n_lineups": n, "locks": list(locks), "excludes": list(excludes), "budget": budget,
                **({"event": event} if event else {})}
        out = _ok(con, "propose_lineups", **args)
        if len(out["lineups"]) < n:
            raise RuntimeError(f"{id_}: only {len(out['lineups'])} of {n} lineups fit")
        add(id_, "lineup", question, ["propose_lineups"], type="lineup_valid", n_lineups=n,
            locks=list(locks), excludes=list(excludes), budget=budget)

    lineup_case("lu-01", "Build me a lineup for this week.")
    lineup_case("lu-02", "Build 2 lineups for this week with Scottie Scheffler locked.", n=2, locks=[scheffler])
    wy1 = wyndham["players"][0]
    lineup_case("lu-03", f"Build 3 lineups for the 2026 Wyndham Championship without {wy1['name']}.",
                n=3, excludes=[wy1["player_id"]], event=wyndham["event"]["event_id"])
    rk = _ok(con, "get_rankings", event="Rocket Classic", season=2026, top_n=3)["players"]
    lineup_case("lu-04", f"Build 2 lineups for the 2026 Rocket Classic locking {rk[0]['name']} and "
                         f"{rk[2]['name']}, and leaving out {rk[1]['name']}.",
                n=2, locks=[rk[0]["player_id"], rk[2]["player_id"]], excludes=[rk[1]["player_id"]],
                event="Rocket Classic")
    lineup_case("lu-05", "Build a lineup for this week with a $1,500,000 budget.", budget=1_500_000)
    pricey = sorted(value_week, key=lambda p: -p["salary"])[:3]
    out = run_tool(con, "propose_lineups", {"locks": [p["player_id"] for p in pricey], "budget": 1_000_000})
    if out.get("error") != "infeasible":
        raise RuntimeError(f"lu-06 should be infeasible: {out}")
    add("lu-06", "lineup", f"Build a lineup for this week locking {', '.join(p['name'] for p in pricey)} "
                           f"with a $1,000,000 budget.", ["propose_lineups"], type="infeasible")

    # ── Guardrails ─────────────────────────────────────────────────────────────
    if run_tool(con, "resolve_player", {"name": "Zylo Brackenridge"}).get("error") != "player_not_found":
        raise RuntimeError("guard-01's made-up player now matches someone")
    add("guard-01", "guardrail", "How will Zylo Brackenridge do this week?", ["resolve_player"],
        type="refusal_or_clarify")
    for id_, name, question in [("guard-02", "Smith", "How is Smith doing this week?"),
                                ("guard-03", "Kim", "Where is Kim ranked this week?")]:
        out = _ok(con, "resolve_player", name=name)
        if out["status"] != "ambiguous":
            raise RuntimeError(f"{id_}: {name!r} is no longer ambiguous")
        add(id_, "guardrail", question, ["resolve_player"], type="clarify")

    def unavailable_case(id_, question, tools, tool, **args):
        code = run_tool(con, tool, args).get("error")
        if code is None:
            raise RuntimeError(f"{id_}: {tool}({args}) no longer fails")
        add(id_, "guardrail", question, tools, type="unavailable", error=code)

    unavailable_case("guard-04", f"Why is {wy1['name']} ranked first at the 2026 Wyndham Championship?",
                     ["explain_prediction"], "explain_prediction", player_id=wy1["player_id"],
                     event=wyndham["event"]["event_id"])
    unavailable_case("guard-05", "Who were the best value plays at the 2019 U.S. Open?", ["value_report"],
                     "value_report", event="U.S. Open", season=2019)
    unavailable_case("guard-06", "Who does the model rank first at the 2026 Masters?", ["get_rankings"],
                     "get_rankings", event="Masters", season=2026)

    print(f"'This week' is the {week_name} ({week['event']['event_id']}); {top2['name']} is 2nd.")
    return cases


def main() -> None:
    cases = build(connect())
    EVAL_QUESTIONS.parent.mkdir(parents=True, exist_ok=True)
    with EVAL_QUESTIONS.open("w") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")
    counts: Dict[str, int] = {}
    for c in cases:
        counts[c["category"]] = counts.get(c["category"], 0) + 1
    print(f"Wrote {len(cases)} cases to {EVAL_QUESTIONS}: {counts}")


if __name__ == "__main__":
    main()
