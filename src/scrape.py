"""
PGA Tour data fetcher — uses the orchestrator.pgatour.com GraphQL API.

The original stats-download CSV endpoint is no longer functional (returns empty
bodies as of mid-2025). This module uses GraphQL queries confirmed to work.

Key endpoints:
  - upcomingSchedule: next N PGA Tour events with course/date
  - field(id):        players entered in a tournament
  - playerFinishStats: last 10 per-tournament values of any stat for one player
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
from typing import Dict, List, Optional

import pandas as pd
import requests

_URL = "https://orchestrator.pgatour.com/graphql"
# PGA Tour's public web-client API key — it ships in their site bundle, so it's not a
# secret. Overridable via PGA_API_KEY in case they rotate it.
_KEY = os.environ.get("PGA_API_KEY", "da2-gsrx5bibzbb4njvhl7t37wqyl4")

SG_STAT_IDS: Dict[str, str] = {
    "sg_ott":   "02567",
    "sg_app":   "02568",
    "sg_arg":   "02569",
    "sg_putt":  "02564",
    "sg_total": "02675",
}


def _post(query: str, variables: Optional[dict] = None) -> dict:
    body = {"query": query}
    if variables:
        body["variables"] = variables
    resp = requests.post(
        _URL, json=body,
        headers={"x-api-key": _KEY, "Content-Type": "application/json"},
        timeout=30,
    )
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


# ── Player cut history ─────────────────────────────────────────────────────────

_CUT_HISTORY_QUERY = """
query {
    playerProfileTournamentResults(playerId: "%s", tourCode: R) {
        tournaments {
            tournaments { startDate position }
        }
    }
}
"""

_INVALID_POSITIONS = {"W/D", "DQ", "MDF", "998", None}


def get_player_cut_history(player_id: str, n_events: int = 24) -> List[bool]:
    """
    Return a list of bool (True = made cut) for the player's last n_events
    PGA Tour starts, ordered from oldest to newest.
    Withdrawals and non-standard events (Ryder Cup etc.) are excluded.
    """
    try:
        data = _post(_CUT_HISTORY_QUERY % player_id)
    except Exception:
        return []

    if data is None:
        return []

    groups = (data.get("playerProfileTournamentResults") or {}).get("tournaments", [])
    events = []
    for g in groups:
        for t in (g.get("tournaments") or []):
            pos = t.get("position")
            start = t.get("startDate")
            if pos in _INVALID_POSITIONS or pos == "998":
                continue
            events.append({"start": start or "", "made_cut": pos != "CUT"})

    events.sort(key=lambda x: x["start"])
    return [e["made_cut"] for e in events[-n_events:]]


# ── Player form ────────────────────────────────────────────────────────────────

_FINISH_STAT_QUERY = """
query PlayerFinishStats($playerId: ID!, $statId: String!, $tourCode: TourCode!) {
    playerFinishStats(playerId: $playerId, statId: $statId, tourCode: $tourCode) {
        statName playerAvg
        statValues { ... on FinishStatValue { value date tournamentName } }
    }
}
"""


def get_player_sg_history(
    player_id: str,
    stat_ids: Dict[str, str] = SG_STAT_IDS,
    n_events: int = 10,
) -> pd.DataFrame:
    """
    Fetch the last N tournament SG values for a player across all SG categories.
    Returns a DataFrame with one row per tournament, columns named sg_*.
    The API returns at most 10 events per stat.
    """
    per_stat: Dict[str, pd.DataFrame] = {}

    for stat_name, stat_id in stat_ids.items():
        try:
            data = _post(
                _FINISH_STAT_QUERY,
                {"playerId": player_id, "statId": stat_id, "tourCode": "R"},
            )
        except Exception:
            continue

        if data is None:
            continue

        vals = (
            (data.get("playerFinishStats") or {})
            .get("statValues", [])
        )
        if not vals:
            continue

        rows = [
            {
                "tournament_name": v.get("tournamentName"),
                "date": pd.to_datetime(v["date"], unit="s", utc=True).tz_localize(None),
                stat_name: float(v["value"]),
            }
            for v in vals[-n_events:]
            if v.get("value") is not None
        ]
        if rows:
            per_stat[stat_name] = pd.DataFrame(rows).set_index("date")

        time.sleep(0.15)

    if not per_stat:
        return pd.DataFrame()

    # Merge all stats on date index (outer join keeps all events)
    combined = pd.concat(per_stat.values(), axis=1)
    # Keep the tournament_name from whichever stat had a value for that date
    tn_series = [df[["tournament_name"]] for df in per_stat.values()]
    combined["tournament_name"] = pd.concat(tn_series, axis=1).bfill(axis=1).iloc[:, 0]

    combined["player_id"] = player_id
    return combined.reset_index().rename(columns={"index": "date"})


# ── Past tournament results ────────────────────────────────────────────────────

def get_tournament_results(tournament_id: str, year: int) -> pd.DataFrame:
    """
    Return the leaderboard for a completed tournament.
    Columns: player_id, player_name, finish_num, total_strokes, rounds_played,
             par_relative_score, winnings.
    Note: does NOT include SG stats (use get_player_sg_history for that).
    """
    data = _post(
        """
        query TournamentPastResults($id: ID!, $year: Int) {
            tournamentPastResults(id: $id, year: $year) {
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
        {"id": tournament_id, "year": year},
    )
    players = data.get("tournamentPastResults", {}).get("players", [])
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


# ── Weekly feature matrix ──────────────────────────────────────────────────────

def build_weekly_features(
    tournament_id: str,
    historical_df: Optional[pd.DataFrame] = None,
    n_events: int = 10,
) -> pd.DataFrame:
    """
    Build a feature matrix for the upcoming tournament.
    For each active player in the field, fetches their last N tournament SG values
    and computes rolling averages (matching the training feature schema).

    Args:
        tournament_id:  PGA Tour tournament ID, e.g. 'R2026026'
        historical_df:  processed historical DataFrame (for course history lookup)
        n_events:       how many past events to pull per player (max 10 from API)

    Returns:
        DataFrame with one row per player, feature columns matching training schema.
    """
    print(f"Fetching field for {tournament_id} ...")
    players = get_field(tournament_id)
    if not players:
        raise ValueError(f"No players found in field for {tournament_id}")
    print(f"  {len(players)} players in field")

    # Get upcoming tournament meta (for course name)
    upcoming = get_upcoming_tournaments(n=10)
    meta = next((t for t in upcoming if t["tournament_id"] == tournament_id), {})
    course = meta.get("course")
    tournament_name = meta.get("tournament_name")

    # Pre-normalize course names in historical data for fuzzy matching.
    # Historical CSV stores: "Muirfield Village Golf Club - Dublin, OH"
    # API returns:           "Muirfield Village Golf Club"
    # Normalize: take text before " - " or "," and lowercase.
    if historical_df is not None and "course" in historical_df.columns:
        def _normalize_course(name: str) -> str:
            return str(name).split(" - ")[0].split(",")[0].strip().lower()
        historical_df = historical_df.copy()
        historical_df["_course_norm"] = historical_df["course"].apply(_normalize_course)
        course_norm = _normalize_course(course) if course else ""

    rows = []
    for i, p in enumerate(players):
        pid = p["player_id"]
        if (i + 1) % 20 == 0:
            print(f"  Fetching form: {i+1}/{len(players)}")

        history = get_player_sg_history(pid, n_events=n_events)
        cut_history = get_player_cut_history(pid, n_events=24)
        time.sleep(0.1)

        row = {
            "player_id": pid,
            "player_name": p["player_name"],
            "owgr": pd.to_numeric(p.get("owgr"), errors="coerce"),
            "tournament_id": tournament_id,
            "tournament_name": tournament_name,
            "course": course,
        }

        # ── SG rolling features ───────────────────────────────────────────────
        if not history.empty:
            for stat in SG_STAT_IDS:
                if stat not in history.columns:
                    continue
                vals = history[stat].dropna().values
                for w, label in [(8, "roll8"), (16, "roll16"), (24, "roll24")]:
                    n = min(w, len(vals))
                    row[f"{stat}_per_round_{label}"] = (
                        float(vals[-n:].mean()) if n > 0 else None
                    )

            # Derive sg_t2g from its three components (API doesn't expose it).
            for label in ("roll8", "roll16", "roll24"):
                parts = [row.get(f"sg_{c}_per_round_{label}") for c in ("ott", "app", "arg")]
                if all(v is not None for v in parts):
                    row[f"sg_t2g_per_round_{label}"] = sum(parts)

            row["starts_last8"] = len(history)

        # ── Made-cut rate ─────────────────────────────────────────────────────
        if cut_history:
            cuts = cut_history  # list of bool, oldest → newest
            for w, label in [(8, "roll8"), (16, "roll16"), (24, "roll24")]:
                n = min(w, len(cuts))
                row[f"made_cut_rate_{label}"] = float(sum(cuts[-n:])) / n if n > 0 else None

        # ── Course history ────────────────────────────────────────────────────
        if historical_df is not None and course_norm:
            player_course = historical_df[
                (historical_df["player_id"] == pid)
                & (historical_df["_course_norm"] == course_norm)
            ]
            if not player_course.empty:
                row["course_sg_hist"] = player_course["sg_total_per_round"].mean()
                row["course_starts"] = len(player_course)

        # ── Links history ─────────────────────────────────────────────────────
        # Combine API history (last 10 events) with historical CSV data.
        # The API captures recent Scottish Opens and The Open; the historical CSV
        # has The Open entries but sg_total_per_round is NaN there, so net signal
        # comes almost entirely from the API.
        from src.features import is_links_tournament
        links_sg_vals = []
        if not history.empty and "tournament_name" in history.columns and "sg_total" in history.columns:
            # tournament_name may be a duplicate column (DataFrame), so squeeze to Series
            t_names = history["tournament_name"]
            if isinstance(t_names, pd.DataFrame):
                t_names = t_names.iloc[:, 0]
            sg_totals = history["sg_total"]
            for name, sg in zip(t_names.tolist(), sg_totals.tolist()):
                if is_links_tournament(str(name)) and pd.notna(sg):
                    links_sg_vals.append(float(sg))
        if historical_df is not None and "tournament_name" in historical_df.columns:
            player_hist = historical_df[historical_df["player_id"] == pid]
            hist_links_mask = player_hist["tournament_name"].apply(is_links_tournament)
            hist_links_sg = player_hist.loc[hist_links_mask, "sg_total_per_round"].dropna()
            links_sg_vals.extend(hist_links_sg.tolist())
        if links_sg_vals:
            row["links_sg_hist"] = float(sum(links_sg_vals) / len(links_sg_vals))
            row["links_starts"]  = len(links_sg_vals)

        # ── Playoff history ───────────────────────────────────────────────────
        from src.features import is_playoff_tournament
        playoff_sg_vals = []
        if not history.empty and "tournament_name" in history.columns and "sg_total" in history.columns:
            t_names = history["tournament_name"]
            if isinstance(t_names, pd.DataFrame):
                t_names = t_names.iloc[:, 0]
            sg_totals = history["sg_total"]
            for name, sg in zip(t_names.tolist(), sg_totals.tolist()):
                if is_playoff_tournament(str(name)) and pd.notna(sg):
                    playoff_sg_vals.append(float(sg))
        if historical_df is not None and "tournament_name" in historical_df.columns:
            player_hist = historical_df[historical_df["player_id"] == pid]
            hist_playoff_mask = player_hist["tournament_name"].apply(is_playoff_tournament)
            hist_playoff_sg = player_hist.loc[hist_playoff_mask, "sg_total_per_round"].dropna()
            playoff_sg_vals.extend(hist_playoff_sg.tolist())
        if playoff_sg_vals:
            row["playoff_sg_hist"] = float(sum(playoff_sg_vals) / len(playoff_sg_vals))
            row["playoff_starts"]  = len(playoff_sg_vals)

        rows.append(row)

    return pd.DataFrame(rows)


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from src.ingest import load_historical
    from src.paths import HISTORICAL_CSV, PROCESSED_DIR

    hist = load_historical(HISTORICAL_CSV) if HISTORICAL_CSV.exists() else None

    upcoming = get_upcoming_tournaments(1)
    if not upcoming:
        print("No upcoming tournaments found.")
        sys.exit(0)

    next_t = upcoming[0]
    print(f"\nNext tournament: {next_t['tournament_name']} @ {next_t['course']}")
    print(f"Date: {next_t['date']}")
    print(f"ID:   {next_t['tournament_id']}\n")

    df = build_weekly_features(next_t["tournament_id"], historical_df=hist)
    out = PROCESSED_DIR / "weekly_features.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"\nSaved {len(df)} rows → {out}")
    print(df.head(3).to_string())
