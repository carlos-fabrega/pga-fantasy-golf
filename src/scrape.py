"""
PGA Tour data fetcher — uses the orchestrator.pgatour.com GraphQL API.

The original stats-download CSV endpoint is no longer functional (returns empty
bodies as of mid-2025). This module uses GraphQL queries confirmed to work.

Key endpoints:
  - upcomingSchedule: next N PGA Tour events with course/date
  - field(id):        players entered in a tournament
  - tournamentPastResults: full leaderboard for a completed tournament

Confirmed SG stat IDs (from schema introspection):
  02567  SG: Off The Tee
  02568  SG: Approach to Green
  02569  SG: Around The Green
  02564  SG: Putting
  02675  SG: Total

Usage — build weekly feature matrix:
  python -m src.scrape
"""

from __future__ import annotations

import os
import time
from typing import List, Optional

import pandas as pd
import requests

_URL = "https://orchestrator.pgatour.com/graphql"
# PGA Tour's public web-client API key — it ships in their site bundle, so it's not a
# secret. Overridable via PGA_API_KEY in case they rotate it.
_KEY = os.environ.get("PGA_API_KEY", "da2-gsrx5bibzbb4njvhl7t37wqyl4")


def _post(query: str, variables: Optional[dict] = None) -> dict:
    body = {"query": query}
    if variables:
        body["variables"] = variables
    # A backfill makes thousands of calls; one dropped connection or 5xx shouldn't end it.
    for attempt in range(3):
        try:
            resp = requests.post(
                _URL, json=body,
                headers={"x-api-key": _KEY, "Content-Type": "application/json"},
                timeout=30,
            )
            if resp.status_code < 500:
                break
        except (requests.ConnectionError, requests.Timeout):
            if attempt == 2:
                raise
        time.sleep(2 ** attempt)
    resp.raise_for_status()
    return resp.json().get("data", {})


# ── Schedule ───────────────────────────────────────────────────────────────────

def get_upcoming_tournaments(n: int = 3) -> List[dict]:
    """
    Return the next N upcoming PGA Tour events.
    Each dict has: id, name, course, date (string like 'Jun 18 - 21').
    """
    data = _post("""
        query {
            upcomingSchedule(tourCode: "R") {
                tournaments { id tournamentName courseName date startDate }
            }
        }
    """)
    ts = data.get("upcomingSchedule", {}).get("tournaments", [])
    return [
        {
            "tournament_id": t["id"],
            "tournament_name": t["tournamentName"],
            "course": t["courseName"],
            "date": t["date"],
            "start_date": t.get("startDate"),
        }
        for t in ts[:n]
    ]


def get_tournaments_for_year(year: int) -> List[dict]:
    """
    Return all tournaments in a given PGA Tour season.
    Each dict has: tournament_id, display_name.
    """
    data = _post(
        """
        query StatDetails($year: Int) {
            statDetails(tourCode: R, statId: "02675", year: $year) {
                tournamentPills { tournamentId displayName }
            }
        }
        """,
        {"year": year},
    )
    pills = data.get("statDetails", {}).get("tournamentPills", [])
    return [
        {"tournament_id": p["tournamentId"], "display_name": p["displayName"]}
        for p in pills
    ]


def get_completed_schedule(year: int) -> List[dict]:
    """
    Return every completed PGA Tour event in a season, oldest first.
    Each dict has: tournament_id, tournament_name, start_date (Timestamp),
    course, purse (display string, e.g. "$8,000,000").
    """
    data = _post("""
        query {
            schedule(tourCode: "R", year: "%d") {
                completed { tournaments { id tournamentName startDate courseName purse } }
            }
        }
    """ % year)
    groups = (data.get("schedule") or {}).get("completed", [])
    rows = [
        {
            "tournament_id": t["id"],
            "tournament_name": t["tournamentName"],
            "start_date": pd.to_datetime(t["startDate"], unit="ms"),
            "course": t.get("courseName"),
            "purse": t.get("purse"),
        }
        for g in groups
        for t in (g.get("tournaments") or [])
    ]
    return sorted(rows, key=lambda r: r["start_date"])


# ── Field ──────────────────────────────────────────────────────────────────────

def get_field(tournament_id: str) -> List[dict]:
    """
    Return the player field for a tournament.
    Each dict has: player_id, player_name, country, owgr, alternate, withdrawn.
    """
    data = _post(
        """
        query Field($id: ID!) {
            field(id: $id) {
                players { id displayName country owgr alternate withdrawn }
            }
        }
        """,
        {"id": tournament_id},
    )
    players = data.get("field", {}).get("players", [])
    return [
        {
            "player_id": p["id"],
            "player_name": p["displayName"],
            "country": p.get("country"),
            "owgr": p.get("owgr"),
            "alternate": p.get("alternate", False),
            "withdrawn": p.get("withdrawn", False),
        }
        for p in players
        if not p.get("alternate") and not p.get("withdrawn")
    ]


# ── Past tournament results ────────────────────────────────────────────────────

def get_tournament_results(tournament_id: str, year: int, season_code: Optional[int] = None) -> pd.DataFrame:
    """
    Return the leaderboard for a completed tournament.
    Columns: player_id, player_name, finish_num, total_strokes, rounds_played,
             par_relative_score, winnings.
    Note: does NOT include SG stats (src.backfill adds those).

    The API's `year` is a season code, not a calendar year: year*10, except
    that events played twice in the 2022-23 wraparound season use 20230 for
    the fall-2022 playing and 20231 for the fall-2023 one (pass season_code
    for those; see backfill.season_code). An unrecognised value is ignored and
    the most recent edition comes back instead, so a response for a different
    season is rejected. Returns an empty DataFrame in that case.
    """
    past = _past_results(tournament_id, season_code or year * 10)
    if str(past.get("id", ""))[:5] != tournament_id[:5]:
        return pd.DataFrame()
    players = past.get("players", [])
    rows = []
    for p in players:
        rounds_played = len([r for r in p.get("rounds", []) if r.get("score")])
        par_str = p.get("parRelativeScore", "0").replace("E", "0").replace("+", "")
        rows.append({
            "player_id": p.get("id"),
            "player_name": p.get("player", {}).get("displayName"),
            "tournament_id": tournament_id,
            "season": year,
            "finish_num": pd.to_numeric(
                str(p.get("position", "")).replace("T", ""), errors="coerce"
            ),
            "total_strokes": pd.to_numeric(p.get("total"), errors="coerce"),
            "rounds_played": rounds_played,
            "par_relative_score": pd.to_numeric(par_str, errors="coerce"),
            "made_cut": rounds_played >= 3,
            "winnings": p.get("additionalData", [None, None])[1],
        })
    return pd.DataFrame(rows)


def _past_results(tournament_id: str, season_code: int) -> dict:
    """Raw tournamentPastResults payload ({} if none) for one season code."""
    data = _post(
        """
        query TournamentPastResults($id: ID!, $year: Int) {
            tournamentPastResults(id: $id, year: $year) {
                id
                players {
                    id
                    player { id displayName }
                    rounds { score parRelativeScore }
                    total
                    position
                    parRelativeScore
                    additionalData
                }
            }
        }
        """,
        {"id": tournament_id, "year": season_code},
    )
    return (data or {}).get("tournamentPastResults") or {}


# ── Weekly feature matrix ──────────────────────────────────────────────────────

def build_weekly_features(
    tournament_id: str,
    history: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build the feature matrix for one tournament's field from the local history
    (historical CSV + backfilled events), using features.build_features — the
    same code that builds the training matrix. The field is appended to the
    history as rows with no results, so every shift(1) feature on those rows
    sees exactly the events before this one.

    The history must be current: run `python -m src.backfill` first (predict
    does). Only the field and the schedule come from the API.

    Args:
        tournament_id:  PGA Tour tournament ID, e.g. 'R2026026'
        history:        ingest.combine() output; loaded from disk if omitted

    Returns:
        DataFrame with one row per player, feature columns matching training schema.
    """
    from src.features import build_features, feature_columns
    from src.ingest import combine, course_key, load_historical, load_scraped
    from src.paths import HISTORICAL_CSV, SCRAPED_DIR

    print(f"Fetching field for {tournament_id} ...")
    players = get_field(tournament_id)
    if not players:
        raise ValueError(f"No players found in field for {tournament_id}")
    print(f"  {len(players)} players in field")

    if history is None:
        history = combine(load_historical(HISTORICAL_CSV), load_scraped(SCRAPED_DIR))

    # Name, course and start date: from the upcoming schedule, or from the local
    # history when re-scoring an event that has already been played.
    meta = next((t for t in get_upcoming_tournaments(n=10)
                 if t["tournament_id"] == tournament_id), None)
    if meta:
        tournament_name, course = meta["tournament_name"], meta["course"]
        date = pd.to_datetime(meta["start_date"], unit="ms")
    else:
        past = history[history["tournament_id"] == tournament_id]
        if past.empty:
            raise ValueError(f"{tournament_id} is neither upcoming nor in the local history")
        tournament_name, course, date = past.iloc[0][["tournament_name", "course", "date"]]

    history = history[history["date"] < date]
    print(f"  Local history through {history['date'].max():%Y-%m-%d} "
          f"({history['tournament_name'].iloc[-1]})")

    field = pd.DataFrame({
        "player_id": [str(p["player_id"]) for p in players],
        "player_name": [p["player_name"] for p in players],
        "owgr": pd.to_numeric([p.get("owgr") for p in players], errors="coerce"),
        "tournament_id": tournament_id,
        "tournament_name": tournament_name,
        "course": course,
        "course_key": course_key(course),
        "date": date,
        "source": "field",
    })
    df = build_features(pd.concat([history, field], ignore_index=True))

    extra = ["links_sg_hist", "links_starts", "playoff_sg_hist", "playoff_starts"]
    cols = list(field.columns[:6]) + feature_columns(df) + extra
    return df.loc[df["source"] == "field", cols].reset_index(drop=True)


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from src.paths import PROCESSED_DIR

    upcoming = get_upcoming_tournaments(1)
    if not upcoming:
        print("No upcoming tournaments found.")
        sys.exit(0)

    next_t = upcoming[0]
    print(f"\nNext tournament: {next_t['tournament_name']} @ {next_t['course']}")
    print(f"Date: {next_t['date']}")
    print(f"ID:   {next_t['tournament_id']}\n")

    df = build_weekly_features(next_t["tournament_id"])
    out = PROCESSED_DIR / "weekly_features.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"\nSaved {len(df)} rows → {out}")
    print(df.head(3).to_string())
