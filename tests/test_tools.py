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
]
EVENTS = [
    # event_id, tournament_name, course, season, date, source
    ("2254", "Wyndham Championship", "Sedgefield", 2019, "2019-08-04", "training"),
    ("R2026541", "Genesis Scottish Open", "Renaissance", 2026, "2026-07-09", "weekly"),
    ("R2026013", "Wyndham Championship", "Sedgefield", 2026, "2026-08-06", "weekly"),
    ("R2026060", "TOUR Championship", "East Lake", 2026, "2026-08-27", "weekly"),
]
FIELD = ["46046", "35891", "50525", "37455", "52955"]   # everyone but Jordan Smith / Chad Collins


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

    predictions = pd.concat([
        _scores("2254", "base_oof"),
        _scores("R2026541", "base"),                   # links event that only has base scores
        _scores("R2026013", "base"),
        _scores("R2026060", "nocut"),
        _scores("R2026060", "base", offset=0.5),       # scored by a model since retrained
    ], ignore_index=True)

    db = tmp_path / "agent.sqlite"
    w = sqlite3.connect(db)
    pd.DataFrame(PLAYERS, columns=["player_id", "name", "aliases"]).to_sql("players", w, index=False)
    pd.DataFrame(EVENTS, columns=["event_id", "tournament_name", "course", "season", "date", "source"]
                 ).to_sql("events", w, index=False)
    features.to_sql("features", w, index=False)
    predictions.to_sql("predictions", w, index=False)
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
