"""
Backfill mid-2022+ per-tournament data from the PGA Tour GraphQL API.

The old stats-download CSV endpoint died mid-2025, leaving data/raw/scraped/
empty — so the model had been training on 2015-2022 only. This module rebuilds
the scraped feed directly from GraphQL, one CSV per completed event:

  - Event list, start date, course and purse from schedule(year).completed.
  - Finish position + made-cut flag for the whole field (incl. missed cuts) via
    get_tournament_results().
  - Per-tournament Strokes Gained (all 5 categories) via
    statDetails(statId=..., eventQuery={tournamentId, queryType: EVENT_ONLY}).
    The API rejects statId as a GraphQL *variable*, so it is inlined per query.
    Each row's "Avg" field is the player's per-round SG for that category. The
    API has SG only for players who made the cut, and none at all for events
    without ShotLink (the majors).

sg_t2g is derived as ott+app+arg (the SG identity), matching how scrape.py
reconstructs it at inference. ingest.load_scraped() consumes these files.

Usage:
    python -m src.backfill                              # 2022 through this year
    python -m src.backfill --years 2024 2025
    python -m src.backfill --tournament R2024493        # (re)fetch a single event
"""

from __future__ import annotations

import argparse
import datetime as dt
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.paths import SCRAPED_DIR
from src.scrape import get_completed_schedule, get_tournament_results, _post

# Per-round SG statId → column name. sg_t2g is derived, not fetched.
_SG_STATS: Dict[str, str] = {
    "sg_ott":   "02567",
    "sg_app":   "02568",
    "sg_arg":   "02569",
    "sg_putt":  "02564",
    "sg_total": "02675",
}

# Not individual stroke play, or not a Tour event with a purse: no usable
# finish/prize for the model.
_SKIP_PATTERNS = ("presidents cup", "ryder cup", "zurich classic", "match play", "q-school")

# The orchestrator API throttles aggressively: spacing calls < ~1s returns
# truncated row sets (silently — a partial 200, not an error). ~1.2s between
# calls empirically returns the full field, so we trade runtime for completeness.
_SLEEP = 1.2         # delay between per-category calls
_RETRIES = 5
_RETRY_SLEEP = 4.0

# An event whose SG rows cover less than this share of the made-cut players is
# reported: the leaderboard and the SG probably came from different editions.
_MIN_SG_COVERAGE = 0.8


def season_code(year: int, start_date: pd.Timestamp) -> int:
    """
    The season code tournamentPastResults expects. The 2022-23 wraparound
    season held its fall events twice (fall 2022 and fall 2023); the second
    playing is 20231. Every other edition is year*10.
    """
    if year == 2023 and start_date >= pd.Timestamp("2023-09-01"):
        return 20231
    return year * 10


def _event_sg_one_stat(tournament_id: str, year: int, stat_id: str) -> Dict[str, float]:
    """
    Return {player_id: per_round_sg} for one stat category at one event.
    statId must be inlined (the API rejects it as a variable).
    """
    query = (
        'query SD($year: Int, $eventQuery: StatDetailEventQuery) {'
        '  statDetails(tourCode: R, statId: "%s", year: $year, eventQuery: $eventQuery) {'
        '    rows { ... on StatDetailsPlayer { playerId stats { statName statValue } } }'
        '  }'
        '}' % stat_id
    )
    variables = {
        "year": year,
        "eventQuery": {"tournamentId": tournament_id, "queryType": "EVENT_ONLY"},
    }

    for _ in range(_RETRIES):
        try:
            data = _post(query, variables)
            sd = (data or {}).get("statDetails")
            if sd is not None:
                out: Dict[str, float] = {}
                for r in sd.get("rows", []):
                    stats = {s["statName"]: s["statValue"] for s in r["stats"]}
                    avg = stats.get("Avg")
                    if avg not in (None, ""):
                        try:
                            out[str(r["playerId"])] = float(avg)
                        except (TypeError, ValueError):
                            pass
                return out
        except Exception:
            pass
        time.sleep(_RETRY_SLEEP)
    return {}


def build_event(event: dict, year: int) -> pd.DataFrame:
    """
    Build one standardized per-player DataFrame for a single completed event
    (a get_completed_schedule() row): finish/cut for every player (incl. missed
    cuts), SG left-joined on made-cut players. Returns an empty DataFrame if
    the event has no results.
    """
    tournament_id = event["tournament_id"]
    results = get_tournament_results(
        tournament_id, year, season_code=season_code(year, event["start_date"]))
    if results.empty:
        return pd.DataFrame()
    results["player_id"] = results["player_id"].astype(str)

    # SG per category (made-cut players only), merged by player_id.
    sg_by_player: Dict[str, Dict[str, float]] = {}
    for col, stat_id in _SG_STATS.items():
        vals = _event_sg_one_stat(tournament_id, year, stat_id)
        for pid, v in vals.items():
            sg_by_player.setdefault(pid, {})[f"{col}_per_round"] = v
        time.sleep(_SLEEP)

    sg_df = (
        pd.DataFrame.from_dict(sg_by_player, orient="index")
        .rename_axis("player_id")
        .reset_index()
        if sg_by_player else pd.DataFrame(columns=["player_id"])
    )

    df = results.merge(sg_df, on="player_id", how="left")

    # Derive sg_t2g from its three components (SG identity), matching inference.
    parts = ["sg_ott_per_round", "sg_app_per_round", "sg_arg_per_round"]
    if all(p in df.columns for p in parts):
        df["sg_t2g_per_round"] = df[parts].sum(axis=1, min_count=3)

    df["tournament_name"] = event["tournament_name"]
    df["course"] = event.get("course")
    df["date"] = pd.Timestamp(event["start_date"]).strftime("%Y-%m-%d")
    df["purse"] = event.get("purse")
    df["source"] = "scraped"
    return df


def run(years: List[int], only_tournament: Optional[str] = None, output_dir: Path = SCRAPED_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for year in years:
        events = [e for e in get_completed_schedule(year)
                  if not any(p in e["tournament_name"].lower() for p in _SKIP_PATTERNS)]
        if only_tournament:
            events = [e for e in events if e["tournament_id"] == only_tournament]
        print(f"\n=== {year}: {len(events)} events ===", flush=True)

        for i, ev in enumerate(events, 1):
            tid = ev["tournament_id"]
            tag = f"  [{i}/{len(events)}] {tid} {ev['tournament_name'][:40]:40}"
            out_path = output_dir / f"{tid}.csv"
            if out_path.exists() and not only_tournament:
                print(f"{tag} — exists, skip", flush=True)
                continue
            try:
                df = build_event(ev, year)
            except Exception as e:
                print(f"{tag} — ERROR {e}", flush=True)
                continue
            if df.empty:
                print(f"{tag} — no results, skip", flush=True)
                continue
            has_sg = df["sg_total_per_round"].notna() if "sg_total_per_round" in df.columns else pd.Series(False, index=df.index)
            coverage = has_sg[df["made_cut"]].mean() if df["made_cut"].any() else 0.0
            flag = "  ⚠ LOW SG COVERAGE" if 0 < has_sg.sum() and coverage < _MIN_SG_COVERAGE else ""
            df.to_csv(out_path, index=False)
            print(f"{tag} — {len(df):3d} players, {int(has_sg.sum()):3d} w/ SG "
                  f"({coverage:.0%} of made-cut){flag}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="+", type=int,
                        default=list(range(2022, dt.date.today().year + 1)))
    parser.add_argument("--tournament", default=None, help="Backfill only this tournament id")
    parser.add_argument("--dir", default=str(SCRAPED_DIR))
    args = parser.parse_args()
    run(args.years, only_tournament=args.tournament, output_dir=Path(args.dir))
