"""
Backfill 2023+ per-tournament data from the PGA Tour GraphQL API.

The old stats-download CSV endpoint died mid-2025, leaving data/raw/scraped/
empty — so the model has been training on 2015-2022 only. This module rebuilds
the scraped feed directly from GraphQL:

  - Per-tournament Strokes Gained (all 5 categories) via
    statDetails(statId=..., eventQuery={tournamentId, queryType: EVENT_ONLY}).
    The API rejects statId as a GraphQL *variable*, so it is inlined per query.
    Each row's "Avg" field is the player's per-round SG for that category.
  - Finish position + made-cut flag (incl. missed-cut players, which the SG
    endpoint omits) via get_tournament_results().

One CSV per event is written to data/raw/scraped/ with a fixed schema that
ingest.load_scraped() consumes. sg_t2g is derived as ott+app+arg (the SG
identity), matching how features/scrape.py reconstruct it at inference.

Usage:
    python -m src.backfill --years 2023 2024 2025 2026
    python -m src.backfill --years 2024 --tournament R2024493   # single event
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.scrape import get_tournaments_for_year, get_tournament_results, _post

# Per-round SG statId → column name. sg_t2g is derived, not fetched.
_SG_STATS: Dict[str, str] = {
    "sg_ott":   "02567",
    "sg_app":   "02568",
    "sg_arg":   "02569",
    "sg_putt":  "02564",
    "sg_total": "02675",
}

OUTPUT_DIR = Path("data/raw/scraped")

# The orchestrator API throttles aggressively: spacing calls < ~1s returns
# truncated row sets (silently — a partial 200, not an error). ~1.2s between
# calls empirically returns the full field, so we trade runtime for completeness.
_SLEEP = 1.2         # delay between per-category calls
_RETRIES = 5
_RETRY_SLEEP = 4.0


def _event_sg_one_stat(tournament_id: str, year: int, stat_id: str, col: str) -> Dict[str, float]:
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


def build_event(tournament_id: str, year: int, tournament_name: str = "") -> pd.DataFrame:
    """
    Build one standardized per-player DataFrame for a single event:
    finish/cut for every player (incl. missed cuts), SG left-joined on made-cut
    players. Returns empty DataFrame if the event has no results.
    """
    results = get_tournament_results(tournament_id, year)
    if results.empty:
        return pd.DataFrame()
    results["player_id"] = results["player_id"].astype(str)

    # SG per category (made-cut players only), merged by player_id.
    sg_by_player: Dict[str, Dict[str, float]] = {}
    for col, stat_id in _SG_STATS.items():
        vals = _event_sg_one_stat(tournament_id, year, stat_id, col)
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

    df["tournament_id"] = tournament_id
    df["tournament_name"] = tournament_name
    df["season"] = year
    df["source"] = "scraped"
    return df


def run(years: List[int], only_tournament: Optional[str] = None, output_dir: Path = OUTPUT_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for year in years:
        events = get_tournaments_for_year(year)
        if only_tournament:
            events = [e for e in events if e["tournament_id"] == only_tournament]
        print(f"\n=== {year}: {len(events)} events ===")

        for i, ev in enumerate(events, 1):
            tid = ev["tournament_id"]
            name = ev.get("display_name", "")
            out_path = output_dir / f"{tid}.csv"
            if out_path.exists() and not only_tournament:
                print(f"  [{i}/{len(events)}] {tid} {name[:40]:40} — exists, skip")
                continue
            try:
                df = build_event(tid, year, name)
            except Exception as e:
                print(f"  [{i}/{len(events)}] {tid} {name[:40]:40} — ERROR {e}")
                continue
            if df.empty:
                print(f"  [{i}/{len(events)}] {tid} {name[:40]:40} — no results, skip")
                continue
            n_sg = df["sg_total_per_round"].notna().sum() if "sg_total_per_round" in df.columns else 0
            df.to_csv(out_path, index=False)
            print(f"  [{i}/{len(events)}] {tid} {name[:40]:40} — {len(df):3d} players, {n_sg:3d} w/ SG → {out_path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="+", type=int, default=[2023, 2024, 2025, 2026])
    parser.add_argument("--tournament", default=None, help="Backfill only this tournament id")
    parser.add_argument("--dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()
    run(args.years, only_tournament=args.tournament, output_dir=Path(args.dir))
