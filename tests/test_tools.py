"""
Tests for the agent's tools, against a small fixture database and model.

Run:  python -m pytest tests/test_tools.py
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from agent import tools
from agent.db import FEATURES, connect
from agent.tools import TOOLS, run_tool
from src.model import VARIANT_FEATURES

BASE = VARIANT_FEATURES["base"]

PLAYERS = [
    # player_id, name, aliases
    ("46046", "Scottie Scheffler", None),
    ("35891", "Cameron Smith", None),
    ("46464", "Jordan Smith", None),
    ("50525", "Collin Morikawa", None),
    ("H1646", "Chad Collins", None),
    ("37455", "Si Woo Kim", None),
    ("52955", "Ludvig Åberg", "Ludvig Aberg"),
    ("29725", "Tony Finau", None),
    ("39546", "Max Homa", None),
    ("39971", "Sungjae Im", None),
]
# A bigger backtest field for 2254, since Spearman needs 20+ players (as in model CV).
BACKTEST = [(f"G{i:02d}", f"Golfer {i:02d}", None) for i in range(22)]
EVENTS = [
    # event_id, tournament_name, course, season, date, source
    ("2254", "Wyndham Championship", "Sedgefield", 2019, "2019-08-04", "training"),
    ("R2026541", "Genesis Scottish Open", "Renaissance", 2026, "2026-07-09", "weekly"),
    ("R2026013", "Wyndham Championship", "Sedgefield", 2026, "2026-08-06", "weekly"),
    ("R2026060", "TOUR Championship", "East Lake", 2026, "2026-08-27", "weekly"),
]
FIELD = ["46046", "35891", "50525", "37455", "52955", "29725", "39546", "39971"]   # all but J. Smith / C. Collins
SALARY = {pid: 100_000 * (i + 1) for i, pid in enumerate(FIELD)}   # $100K ... $800K


@pytest.fixture(scope="module")
def model():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(60, len(BASE))), columns=BASE)
    y = rng.integers(0, 4, size=60)
    m = xgb.XGBRanker(n_estimators=20, max_depth=3, objective="rank:ndcg")
    m.fit(X, y, qid=np.repeat(np.arange(6), 10))
    return m


@pytest.fixture()
def con(tmp_path, model, monkeypatch):
    monkeypatch.setattr(tools, "_load_model", lambda variant: model)
    rng = np.random.default_rng(1)

    feats = []
    for event_id, *_ in EVENTS:
        for pid in FIELD:
            row = {"player_id": pid, "event_id": event_id, "owgr": float(FIELD.index(pid) + 1)}
            row.update({c: np.nan for c in FEATURES})
            row.update(dict(zip(BASE, rng.normal(size=len(BASE)))))
            feats.append(row)
    features = pd.DataFrame(feats)
    features.loc[features["player_id"] == "52955", "course_sg_hist"] = np.nan   # one gap to report

    def _scores(event_id, version, offset=0.0):
        f = features[features["event_id"] == event_id]
        s = model.predict(f[BASE].astype(float)) + offset
        return pd.DataFrame({"player_id": f["player_id"].values, "event_id": event_id,
                             "model_version": version, "score": s,
                             "rank": pd.Series(s).rank(ascending=False, method="min").astype(int).values})

    backtest = pd.DataFrame({"player_id": [p for p, *_ in BACKTEST], "event_id": "2254",
                             "model_version": "base_oof", "score": np.linspace(1, -1, len(BACKTEST))})
    backtest["rank"] = np.arange(1, len(BACKTEST) + 1)
    predictions = pd.concat([
        backtest,
        _scores("R2026541", "base"),                   # links event that only has base scores
        _scores("R2026013", "base"),
        _scores("R2026060", "nocut"),
        _scores("R2026060", "base", offset=0.5),       # scored by a model since retrained
    ], ignore_index=True)

    db = tmp_path / "agent.sqlite"
    w = sqlite3.connect(db)
    pd.DataFrame(PLAYERS + BACKTEST, columns=["player_id", "name", "aliases"]).to_sql("players", w, index=False)
    pd.DataFrame(EVENTS, columns=["event_id", "tournament_name", "course", "season", "date", "source"]
                 ).to_sql("events", w, index=False)
    features.to_sql("features", w, index=False)
    predictions.to_sql("predictions", w, index=False)
    pd.DataFrame([{"player_id": pid, "event_id": e, "salary": SALARY[pid]}
                  for e in ("R2026013", "R2026060") for pid in FIELD]).to_sql("salaries", w, index=False)
    # Results: the backtest finishes in model order, so its Spearman is 1; live
    # weeks finish in field order (FIELD[0] wins).
    results = [{"player_id": p, "event_id": "2254", "finish": i + 1, "prize_pct": 0.2 / (i + 1)}
               for i, (p, *_) in enumerate(BACKTEST)]
    results += [{"player_id": pid, "event_id": e, "finish": i + 1, "prize_pct": 0.2 / (i + 1)}
                for e in ("R2026013", "R2026060") for i, pid in enumerate(FIELD)]
    pd.DataFrame(results).to_sql("results", w, index=False)
    w.close()
    return connect(db)


# ── resolve_player ─────────────────────────────────────────────────────────────

def test_resolve_exact_ignores_case_order_accents_and_matches_aliases(con):
    for name in ("SCOTTIE SCHEFFLER", "Scheffler, Scottie"):
        out = run_tool(con, "resolve_player", {"name": name})
        assert (out["status"], out["player_id"], out["match"]) == ("resolved", "46046", "exact")
    assert run_tool(con, "resolve_player", {"name": "ludvig aberg"})["name"] == "Ludvig Åberg"


def test_resolve_partial_names(con):
    out = run_tool(con, "resolve_player", {"name": "Morikawa"})
    assert (out["status"], out["player_id"], out["match"]) == ("resolved", "50525", "last_name")
    out = run_tool(con, "resolve_player", {"name": "Si Woo"})
    assert (out["player_id"], out["match"]) == ("37455", "leading_words")
    out = run_tool(con, "resolve_player", {"name": "Cam Smith"})
    assert (out["player_id"], out["match"]) == ("35891", "first_name_prefix")


def test_resolve_shared_name_is_ambiguous_most_recent_first(con):
    out = run_tool(con, "resolve_player", {"name": "Smith"})
    assert out["status"] == "ambiguous"
    # Cameron Smith has events in the fixture, Jordan Smith none.
    assert [c["player_id"] for c in out["candidates"]] == ["35891", "46464"]


def test_resolve_fuzzy_never_auto_resolves(con):
    out = run_tool(con, "resolve_player", {"name": "Schefler"})
    assert out["status"] == "ambiguous" and out["match"] == "fuzzy"
    assert [c["player_id"] for c in out["candidates"]] == ["46046"]
    # Spelling alone puts Collins nearer than Collin — so the user must choose.
    out = run_tool(con, "resolve_player", {"name": "Colin"})
    assert out["status"] == "ambiguous"
    assert {c["player_id"] for c in out["candidates"]} == {"50525", "H1646"}


def test_resolve_not_found(con):
    assert run_tool(con, "resolve_player", {"name": "Zzyzx Qwerty"})["error"] == "player_not_found"


# ── get_rankings ───────────────────────────────────────────────────────────────

def test_rankings_default_to_latest_event_and_its_variant(con):
    out = run_tool(con, "get_rankings", {"top_n": 3})
    assert out["event"]["event_id"] == "R2026060"
    assert out["model_version"] == "nocut"          # TOUR Championship auto-selects no-cut
    assert out["available_model_versions"] == ["base", "nocut"]
    assert out["field_size"] == len(FIELD) and len(out["players"]) == 3
    scores = [p["score"] for p in out["players"]]
    assert scores == sorted(scores, reverse=True)
    assert [p["rank"] for p in out["players"]] == [1, 2, 3]


def test_rankings_fall_back_to_base_when_variant_was_not_run(con):
    out = run_tool(con, "get_rankings", {"event": "Scottish"})
    assert out["model_version"] == "base"


def test_rankings_event_by_name_and_season(con):
    assert run_tool(con, "get_rankings", {"event": "wyndham"})["event"]["event_id"] == "R2026013"
    out = run_tool(con, "get_rankings", {"event": "Wyndham", "season": 2019})
    assert (out["event"]["event_id"], out["model_version"]) == ("2254", "base_oof")


def test_rankings_event_errors(con):
    out = run_tool(con, "get_rankings", {"event": "Championship"})
    assert out["error"] == "ambiguous_event"
    assert set(out["candidates"]) == {"TOUR Championship", "Wyndham Championship"}
    assert run_tool(con, "get_rankings", {"event": "Masters"})["error"] == "event_not_found"
    out = run_tool(con, "get_rankings", {"event": "R2026013", "model_version": "links"})
    assert out["error"] == "no_such_model_version" and out["available"] == ["base"]


# ── get_player_features ────────────────────────────────────────────────────────

def test_features_are_the_events_model_columns(con):
    out = run_tool(con, "get_player_features", {"player_id": "52955"})
    assert out["model_version"] == "nocut"
    cols = set(out["features"]) | set(out["missing"])
    assert cols == set(VARIANT_FEATURES["nocut"])
    assert not any(c.startswith("made_cut_rate") for c in cols)
    assert "course_sg_hist" in out["missing"]

    out = run_tool(con, "get_player_features", {"player_id": "52955", "event": "R2026013"})
    assert "made_cut_rate_roll8" in out["features"]


def test_features_errors(con):
    out = run_tool(con, "get_player_features", {"player_id": "46464"})
    assert out["error"] == "not_in_field"
    assert run_tool(con, "get_player_features", {"player_id": "999"})["error"] == "unknown_player"


# ── explain_prediction ─────────────────────────────────────────────────────────

def test_explanation_reproduces_the_stored_score(con):
    out = run_tool(con, "explain_prediction", {"player_id": "46046", "top_k": len(BASE)})
    p = out["player"]
    total = sum(d["contribution"] for d in p["top_drivers"]) + p["other_features_total"] + p["baseline"]
    assert total == pytest.approx(p["score"], abs=1e-3)
    mags = [abs(d["contribution"]) for d in p["top_drivers"]]
    assert mags == sorted(mags, reverse=True)


def test_comparison_gap_adds_up(con):
    out = run_tool(con, "explain_prediction",
                   {"player_id": "46046", "compare_to": "35891", "top_k": 3})
    d = out["difference"]
    assert len(d["top_drivers"]) == 3
    gap = sum(t["contribution_gap"] for t in d["top_drivers"]) + d["other_features_total"]
    assert gap == pytest.approx(d["score_gap"], abs=1e-3)
    assert d["score_gap"] == pytest.approx(out["player"]["score"] - out["compare_to"]["score"], abs=1e-3)


def test_explanation_refuses_when_model_no_longer_matches(con):
    out = run_tool(con, "explain_prediction", {"player_id": "46046", "model_version": "base"})
    assert out["error"] == "model_changed"


def test_explanation_unsupported_for_backtest_events(con):
    out = run_tool(con, "explain_prediction", {"player_id": "46046", "event": "2254"})
    assert out["error"] == "unsupported"


# ── value_report ───────────────────────────────────────────────────────────────

def test_value_is_salary_rank_minus_model_rank(con):
    out = run_tool(con, "value_report", {"event": "Wyndham", "top_n": len(FIELD)})
    assert out["priced_players"] == len(FIELD)
    players = out["players"]
    by_score = sorted(players, key=lambda p: -p["score"])
    for p in players:
        assert p["salary"] == SALARY[p["player_id"]]
        assert p["salary_rank"] == len(FIELD) - FIELD.index(p["player_id"])    # priciest = 1
        assert p["model_rank"] == by_score.index(p) + 1
        assert p["value"] == p["salary_rank"] - p["model_rank"]
    assert [p["value"] for p in players] == sorted((p["value"] for p in players), reverse=True)


def test_value_needs_salaries(con):
    out = run_tool(con, "value_report", {"event": "Wyndham", "season": 2019})
    assert out["error"] == "no_salaries"
    assert out["events_with_salaries"] == ["TOUR Championship 2026", "Wyndham Championship 2026"]


# ── propose_lineups ────────────────────────────────────────────────────────────

def test_lineups_respect_budget_locks_excludes_and_differ(con):
    budget = 3_000_000   # Si Woo ($400K) + 5 of the other 6 fits only by leaving out a $500K+ player
    out = run_tool(con, "propose_lineups", {"n_lineups": 3, "budget": budget,
                                            "locks": ["37455"], "excludes": ["46046"]})
    assert out["status"].startswith("proposed") and out["proposal_id"]
    assert (out["locks"], out["excludes"]) == (["Si Woo Kim"], ["Scottie Scheffler"])
    seen = []
    for lu in out["lineups"]:
        ids = {p["player_id"] for p in lu["players"]}
        assert len(ids) == 6 and "37455" in ids and "46046" not in ids
        assert lu["total_salary"] == sum(SALARY[i] for i in ids) <= budget
        assert lu["budget_left"] == budget - lu["total_salary"]
        assert ids not in seen
        seen.append(ids)
    assert len(seen) == 3


def test_lineups_report_when_fewer_fit(con):
    # Excluding 2 of 8 leaves exactly one possible lineup.
    out = run_tool(con, "propose_lineups", {"n_lineups": 2, "budget": 10_000_000,
                                            "excludes": ["46046", "35891"]})
    assert len(out["lineups"]) == 1 and "Only 1 of 2" in out["note"]


def test_lineup_errors(con):
    def err(**a):
        return run_tool(con, "propose_lineups", a)["error"]
    assert err(locks=["46464"]) == "lock_not_available"                 # Jordan Smith isn't in the field
    assert err(locks=["46046"], excludes=["46046"]) == "bad_input"
    assert err(locks=["39971", "39546"], budget=1_000_000) == "infeasible"   # $1.5M locked
    assert err(budget=1_000_000) == "infeasible"                         # cheapest six cost $2.1M
    assert err(n_lineups=6) == "bad_input"
    assert err(locks=["999"]) == "unknown_player"
    assert err(event="Wyndham", season=2019) == "no_salaries"


# ── model_performance ──────────────────────────────────────────────────────────

def test_performance_for_one_event(con):
    out = run_tool(con, "model_performance", {"event": "Wyndham", "season": 2019})
    assert (out["model_version"], out["players"], out["spearman"]) == ("base_oof", 22, 1.0)
    assert out["top10_hits"] == 10 and out["winner"] == {"name": "Golfer 00", "model_rank": 1}

    out = run_tool(con, "model_performance", {"event": "Wyndham"})       # live week, 8 players
    assert out["spearman"] is None                                     # under 20: skipped, as in CV
    assert out["winner"]["name"] == "Scottie Scheffler"                # FIELD[0] finishes first
    assert out["top10_hits"] == len(FIELD)                             # whole field is "top 10"


def test_performance_by_season_and_overall(con):
    out = run_tool(con, "model_performance", {"season": 2026})
    assert out["events"] == 2 and [e["event_id"] for e in out["by_event"]] == ["R2026013", "R2026060"]
    assert out["by_event"][1]["model_version"] == "nocut"
    out = run_tool(con, "model_performance", {})
    assert [(s["season"], s["events"]) for s in out["seasons"]] == [(2019, 1), (2026, 2)]
    assert out["seasons"][0]["mean_spearman"] == 1.0
    assert run_tool(con, "model_performance", {"season": 2001})["error"] == "no_results"


# ── schemas + dispatch ─────────────────────────────────────────────────────────

def test_every_schema_has_a_function_and_valid_required_args():
    for t in TOOLS:
        assert t["name"] in tools._FUNCS and t["description"]
        props = t["input_schema"]["properties"]
        assert set(t["input_schema"].get("required", [])) <= set(props)


def test_run_tool_returns_errors_instead_of_raising(con):
    assert run_tool(con, "drop_tables", {})["error"] == "unknown_tool"
    assert run_tool(con, "get_rankings", {"bogus": 1})["error"] == "bad_input"
    assert run_tool(con, "get_player_features", {})["error"] == "bad_input"
