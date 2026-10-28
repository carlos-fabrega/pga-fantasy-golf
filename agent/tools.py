"""
Tools the analyst agent can call, plus their JSON schemas.

Each tool takes a read-only connection from agent.db.connect() and plain JSON
arguments, and returns compact JSON-able dicts. Expected failures (unknown
player, ambiguous event, ...) come back as {"error": code, "detail": ...}
rather than raising, so the agent can read them and recover. run_tool() is the
single entry point the agent loop uses.

Model scores are raw XGBRanker outputs: higher is better, and they are only
comparable within one event and one model version. (The projections CSVs call
this column projected_prize_pct, but it is the ranker's margin, not a share of
the purse.)
"""

from __future__ import annotations

import contextlib
import difflib
import io
import sqlite3
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
import xgboost as xgb

from agent.db import match_key
from src.model import VARIANT_FEATURES, load
from src.predict import align_features, select_variant

# Contributions + baseline must reproduce the stored score to within this, or
# the saved model is not the one that produced the ranking.
_REPRODUCE_TOL = 1e-4


class ToolError(Exception):
    def __init__(self, code: str, detail: str, **extra):
        super().__init__(detail)
        self.code, self.detail, self.extra = code, detail, extra

    def as_dict(self) -> dict:
        return {"error": self.code, "detail": self.detail, **self.extra}


def _load_model(variant: str):
    """Saved model for a variant. A seam so tests can swap in a small model."""
    return load(variant=variant)


def _r(x, nd: int = 4):
    """Round for output; NaN/None → None so the JSON stays clean."""
    return None if x is None or pd.isna(x) else round(float(x), nd)


# ── Shared lookups ─────────────────────────────────────────────────────────────

def _event_info(ev: sqlite3.Row) -> dict:
    return {"event_id": ev["event_id"], "name": ev["tournament_name"],
            "season": ev["season"], "date": ev["date"], "course": ev["course"]}


def _resolve_event(con: sqlite3.Connection, event: Optional[str] = None,
                   season: Optional[int] = None) -> sqlite3.Row:
    """
    event can be an event_id ("R2026060", "2254") or part of a tournament name
    ("Wyndham"). A name resolves to its latest edition, or the one in season.
    None means the most recent event that has predictions ("this week").
    """
    if event is None:
        row = con.execute(
            "SELECT * FROM events WHERE event_id IN (SELECT DISTINCT event_id FROM predictions) "
            "ORDER BY date DESC LIMIT 1").fetchone()
        if row is None:
            raise ToolError("no_data", "The database has no scored events.")
        return row

    row = con.execute("SELECT * FROM events WHERE event_id = ?", (str(event),)).fetchone()
    if row is not None:
        return row

    sql, params = "SELECT * FROM events WHERE tournament_name LIKE ?", [f"%{event}%"]
    if season is not None:
        sql, params = sql + " AND season = ?", params + [int(season)]
    rows = con.execute(sql + " ORDER BY date DESC", params).fetchall()
    if not rows:
        raise ToolError("event_not_found", f"No event matches {event!r}"
                        + (f" in season {season}." if season is not None else "."))
    names = list(dict.fromkeys(r["tournament_name"] for r in rows))
    if len(names) > 1:
        exact = [r for r in rows if r["tournament_name"].lower() == str(event).lower()]
        if exact:
            return exact[0]
        raise ToolError("ambiguous_event", f"{event!r} matches several tournaments; ask which one.",
                        candidates=names[:8])
    return rows[0]


def _model_version(con: sqlite3.Connection, ev: sqlite3.Row,
                   requested: Optional[str] = None) -> Tuple[str, List[str]]:
    """
    The model version to report for an event, and every version available.
    Default: the out-of-fold backtest for training events; for live weeks, the
    variant predict would auto-select (links/nocut/base), falling back to base
    when that variant wasn't run.
    """
    available = [r[0] for r in con.execute(
        "SELECT DISTINCT model_version FROM predictions WHERE event_id = ? ORDER BY 1",
        (ev["event_id"],))]
    if not available:
        raise ToolError("no_predictions", f"No model predictions for {ev['event_id']}.")
    if requested is not None:
        if requested not in available:
            raise ToolError("no_such_model_version",
                            f"{requested!r} was not run for {ev['event_id']}.", available=available)
        return requested, available
    preferred = "base_oof" if ev["source"] == "training" else select_variant(ev["tournament_name"])
    if preferred in available:
        return preferred, available
    return ("base" if "base" in available else available[0]), available


def _player(con: sqlite3.Connection, player_id: str) -> sqlite3.Row:
    row = con.execute("SELECT * FROM players WHERE player_id = ?", (str(player_id),)).fetchone()
    if row is None:
        raise ToolError("unknown_player", f"No player_id {player_id!r}; call resolve_player first.")
    return row


def _last_event_dates(con: sqlite3.Connection, player_ids: List[str]) -> Dict[str, str]:
    marks = ",".join("?" * len(player_ids))
    return dict(con.execute(
        f"SELECT f.player_id, MAX(e.date) FROM features f JOIN events e USING (event_id) "
        f"WHERE f.player_id IN ({marks}) GROUP BY f.player_id", player_ids).fetchall())


def _feature_row(con: sqlite3.Connection, player_id: str, ev: sqlite3.Row) -> pd.Series:
    df = pd.read_sql("SELECT * FROM features WHERE player_id = ? AND event_id = ?",
                     con, params=(player_id, ev["event_id"]))
    if df.empty:
        last = _last_event_dates(con, [player_id]).get(player_id)
        raise ToolError("not_in_field", f"Player {player_id} has no data for {ev['event_id']} "
                        f"({ev['tournament_name']} {ev['season']}).", last_event_date=last)
    return df.iloc[0]


# ── 1. resolve_player ──────────────────────────────────────────────────────────

def resolve_player(con: sqlite3.Connection, name: str) -> dict:
    """
    Turn a name as the user typed it into a player_id. Stages, first hit wins:
    exact (any spelling, ignoring accents/spacing/punctuation) → same last name
    with a first-name prefix ("Matt Fitz…" style) → last name alone → fuzzy
    (similarity ≥ 0.8). One match resolves; several come back as candidates
    for the agent to ask about.
    """
    key = match_key(name)
    if not key:
        raise ToolError("bad_input", "Empty player name.")
    spellings = []   # (player_id, canonical name, key of one spelling)
    for r in con.execute("SELECT player_id, name, aliases FROM players"):
        for s in [r["name"]] + (r["aliases"].split("|") if r["aliases"] else []):
            spellings.append((r["player_id"], r["name"], match_key(s)))
    tokens = key.split()

    def _first_last(k: str) -> Tuple[str, str]:
        p = k.split()
        return p[0], (p[-1] if len(p) > 1 else "")

    stages: List[Tuple[str, Callable[[str], bool]]] = [
        ("exact", lambda k: k.replace(" ", "") == key.replace(" ", "")),
    ]
    if len(tokens) > 1:
        qf, ql = tokens[0], tokens[-1]
        stages.append(("first_name_prefix", lambda k: _first_last(k)[1] == ql and (
            _first_last(k)[0].startswith(qf) or qf.startswith(_first_last(k)[0]))))
        # Leading whole words, for multi-word first names: "Si Woo" → "Si Woo Kim".
        stages.append(("leading_words", lambda k: k.startswith(key + " ")))
    else:
        stages.append(("last_name", lambda k: _first_last(k)[1] == key))
        stages.append(("first_name", lambda k: _first_last(k)[0] == key))

    names = {}
    for match, hit in stages:
        names = {pid: n for pid, n, k in spellings if hit(k)}
        if names:
            break
    else:
        # A one-word query is a first or last name, so compare it with those alone.
        match = "fuzzy"
        sim: Dict[str, float] = {}
        for pid, _, k in spellings:
            targets = _first_last(k) if len(tokens) == 1 else (k,)
            best = max(difflib.SequenceMatcher(None, key, t).ratio() for t in targets)
            sim[pid] = max(sim.get(pid, 0.0), best)
        ranked = sorted((s, pid) for pid, s in sim.items() if s >= 0.8)[::-1]
        canon = {pid: n for pid, n, _ in spellings}
        names = {pid: canon[pid] for _, pid in ranked}

    if not names:
        raise ToolError("player_not_found", f"No player matches {name!r}.")
    last = _last_event_dates(con, list(names))
    found = [{"player_id": pid, "name": n, "last_event_date": last.get(pid)} for pid, n in names.items()]
    # A fuzzy hit is a guess ("Colin" is nearer "Collins" than "Collin" by spelling), so
    # even a single one comes back for the user to confirm.
    if len(found) == 1 and match != "fuzzy":
        return {"status": "resolved", "match": match, **found[0]}
    if match != "fuzzy":
        found.sort(key=lambda c: c["last_event_date"] or "", reverse=True)
    return {"status": "ambiguous", "match": match, "candidates": found[:8],
            "detail": ("No exact match; ask the user whether they meant one of these."
                       if match == "fuzzy" else "Several players match; ask the user which one.")}


# ── 2. get_rankings ────────────────────────────────────────────────────────────

def get_rankings(con: sqlite3.Connection, event: Optional[str] = None, season: Optional[int] = None,
                 top_n: int = 10, model_version: Optional[str] = None) -> dict:
    ev = _resolve_event(con, event, season)
    version, available = _model_version(con, ev, model_version)
    rows = con.execute(
        "SELECT p.rank, p.player_id, pl.name, p.score FROM predictions p "
        "JOIN players pl USING (player_id) WHERE p.event_id = ? AND p.model_version = ? "
        "ORDER BY p.rank, p.score DESC", (ev["event_id"], version)).fetchall()
    top_n = max(1, int(top_n))
    return {
        "event": _event_info(ev),
        "model_version": version,
        "available_model_versions": available,
        "field_size": len(rows),
        "players": [{"rank": r["rank"], "player_id": r["player_id"], "name": r["name"],
                     "score": _r(r["score"])} for r in rows[:top_n]],
    }


# ── 3. get_player_features ─────────────────────────────────────────────────────

def _variant(version: str) -> str:
    return version.replace("_oof", "")


def get_player_features(con: sqlite3.Connection, player_id: str, event: Optional[str] = None,
                        season: Optional[int] = None, model_version: Optional[str] = None) -> dict:
    pl = _player(con, player_id)
    ev = _resolve_event(con, event, season)
    version, _ = _model_version(con, ev, model_version)
    row = _feature_row(con, pl["player_id"], ev)
    cols = VARIANT_FEATURES[_variant(version)]
    return {
        "event": _event_info(ev),
        "player": {"player_id": pl["player_id"], "name": pl["name"]},
        "model_version": version,
        "owgr": _r(row["owgr"], 0),
        "features": {c: _r(row[c], 3) for c in cols if pd.notna(row[c])},
        "missing": [c for c in cols if pd.isna(row[c])],
    }


# ── 4. explain_prediction ──────────────────────────────────────────────────────

def _explain_one(con: sqlite3.Connection, model, ev: sqlite3.Row, version: str,
                 player_id: str) -> Tuple[dict, pd.Series, pd.Series]:
    """SHAP contributions for one player, checked against the stored score."""
    pl = _player(con, player_id)
    pred = con.execute("SELECT rank, score FROM predictions WHERE player_id = ? AND event_id = ? "
                       "AND model_version = ?", (pl["player_id"], ev["event_id"], version)).fetchone()
    if pred is None:
        raise ToolError("not_in_field", f"{pl['name']} has no {version} prediction for {ev['event_id']}.")
    row = _feature_row(con, pl["player_id"], ev)
    with contextlib.redirect_stdout(io.StringIO()):   # align_features notes absent columns
        X = align_features(row.to_frame().T, model)
    contribs = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[0]
    contrib = pd.Series(contribs[:-1], index=X.columns)
    baseline = float(contribs[-1])
    if abs(contrib.sum() + baseline - pred["score"]) > _REPRODUCE_TOL:
        raise ToolError(
            "model_changed",
            f"The saved {version} model no longer reproduces the stored scores for "
            f"{ev['event_id']} (it was retrained after these projections were made), so an "
            f"explanation would describe a different model than the one that ranked this field.")
    info = {"player_id": pl["player_id"], "name": pl["name"], "rank": pred["rank"],
            "score": _r(pred["score"]), "baseline": _r(baseline)}
    return info, contrib, X.iloc[0]


def _top(contrib: pd.Series, k: int) -> List[str]:
    return contrib.abs().sort_values(ascending=False, kind="stable").index[:k].tolist()


def explain_prediction(con: sqlite3.Connection, player_id: str, event: Optional[str] = None,
                       season: Optional[int] = None, compare_to: Optional[str] = None,
                       model_version: Optional[str] = None, top_k: int = 5) -> dict:
    ev = _resolve_event(con, event, season)
    version, _ = _model_version(con, ev, model_version)
    if ev["source"] == "training":
        raise ToolError("unsupported", "Explanations are only available for live weekly events; "
                        "backtest scores come from per-season models that aren't saved.")
    model = _load_model(version)
    top_k = max(1, int(top_k))

    info, contrib, values = _explain_one(con, model, ev, version, player_id)
    top = _top(contrib, top_k)
    out = {
        "event": _event_info(ev),
        "model_version": version,
        "player": {**info,
                   "top_drivers": [{"feature": f, "value": _r(values[f], 3),
                                    "contribution": _r(contrib[f])} for f in top],
                   "other_features_total": _r(contrib.drop(top).sum())},
    }
    if compare_to is not None:
        info_b, contrib_b, values_b = _explain_one(con, model, ev, version, compare_to)
        gap = contrib - contrib_b
        top_gap = _top(gap, top_k)
        out["compare_to"] = info_b
        out["difference"] = {
            "score_gap": _r(info["score"] - info_b["score"]),
            "top_drivers": [{"feature": f, "value": _r(values[f], 3), "value_compare": _r(values_b[f], 3),
                             "contribution_gap": _r(gap[f])} for f in top_gap],
            "other_features_total": _r(gap.drop(top_gap).sum()),
        }
    return out


# ── Schemas + dispatch ─────────────────────────────────────────────────────────

_EVENT_PROPS = {
    "event": {"type": "string",
              "description": "Event id (e.g. 'R2026060') or part of a tournament name (e.g. 'Wyndham'). "
                             "Omit for the most recent scored event (this week)."},
    "season": {"type": "integer", "description": "Season year, to pick an edition when event is a name."},
}
_VERSION_PROP = {
    "model_version": {"type": "string",
                      "description": "Override the model version (base, links, nocut, base_oof). "
                                     "Omit to use the event's default."},
}
_PLAYER_ID = {"type": "string", "description": "player_id from resolve_player."}

TOOLS: List[dict] = [
    {
        "name": "resolve_player",
        "description": (
            "Look up a golfer's player_id from a name as the user wrote it (full name, last name, "
            "nickname or misspelling). Call this before any tool that takes a player_id. Returns "
            "status 'resolved' with one player, or 'ambiguous' with candidates — then ask the user "
            "which one; never pick for them."),
        "input_schema": {"type": "object", "properties": {"name": {"type": "string"}},
                         "required": ["name"]},
    },
    {
        "name": "get_rankings",
        "description": (
            "The model's ranking of an event's field: rank, player and score. Live 2026 weeks use "
            "the weekly projection (links model for links events, nocut model for no-cut events); "
            "2018-2022 events use honest out-of-fold backtest scores (base_oof). Scores are raw "
            "ranking outputs, comparable only within one event and model version — not "
            "probabilities or prize shares."),
        "input_schema": {"type": "object", "properties": {
            **_EVENT_PROPS,
            "top_n": {"type": "integer", "description": "How many players to return (default 10)."},
            **_VERSION_PROP}},
    },
    {
        "name": "get_player_features",
        "description": (
            "A player's model inputs for one event (only the features that event's model uses). "
            "sg_* are strokes gained per round averaged over the last 8/16/24 starts (roll8/16/24); "
            "made_cut_rate_* the share of those starts making the cut; starts_last8 starts in the "
            "last 8 events; course_* / links_* / playoff_* history at this course or event type. "
            "Every feature only uses events before this one. 'missing' lists features with no data."),
        "input_schema": {"type": "object", "properties": {
            "player_id": _PLAYER_ID, **_EVENT_PROPS, **_VERSION_PROP}, "required": ["player_id"]},
    },
    {
        "name": "explain_prediction",
        "description": (
            "Why the model scored a player as it did at a live weekly event: exact per-feature "
            "SHAP contributions from the XGBoost model (contributions + baseline = score). Pass "
            "compare_to with a second player_id to get the features that drive the gap between "
            "them. Not available for 2018-2022 backtest events."),
        "input_schema": {"type": "object", "properties": {
            "player_id": _PLAYER_ID, **_EVENT_PROPS,
            "compare_to": {"type": "string", "description": "Optional second player_id to compare against."},
            "top_k": {"type": "integer", "description": "How many top features to list (default 5)."},
            **_VERSION_PROP}, "required": ["player_id"]},
    },
]

_FUNCS: Dict[str, Callable[..., dict]] = {
    "resolve_player": resolve_player,
    "get_rankings": get_rankings,
    "get_player_features": get_player_features,
    "explain_prediction": explain_prediction,
}
_SCHEMAS = {t["name"]: t["input_schema"] for t in TOOLS}


def run_tool(con: sqlite3.Connection, name: str, args: dict) -> dict:
    """Run a tool by name; every failure comes back as an {"error": ...} dict."""
    if name not in _FUNCS:
        return {"error": "unknown_tool", "detail": f"No tool named {name!r}."}
    schema = _SCHEMAS[name]
    unknown = set(args) - set(schema["properties"])
    missing = set(schema.get("required", [])) - set(args)
    if unknown or missing:
        return {"error": "bad_input",
                "detail": f"unknown args {sorted(unknown)}, missing args {sorted(missing)}"}
    try:
        return _FUNCS[name](con, **args)
    except ToolError as e:
        return e.as_dict()
    except Exception as e:   # surfaced to the agent rather than crashing the loop
        return {"error": "internal_error", "detail": f"{type(e).__name__}: {e}"}
