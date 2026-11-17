"""
Official World Golf Ranking snapshots from the PGA Tour GraphQL API.

The OWGR stat (statId 186) can be queried "through" any past PGA Tour event,
which gives the top ~1,000 players' rank and average points as of that event's
final day. features.add_owgr() joins each tournament to the latest snapshot
dated before it starts.

Do NOT use the `owgr` on a past event's field listing for history: the API
returns the ranking from after the event there, so it leaks the result.

Two kinds of file under data/raw/owgr/:
  {season}.csv           every "through event" snapshot the API has for a season
  week_{YYYY-MM-DD}.csv  the current ranking, saved each time predict runs

The API's through-event history is patchy for recent seasons (22 snapshots for
2025, 4 for 2026), so the weekly files are what keep the feature fresh from now on.

Usage:
    python -m src.owgr                    # 2014 through this year; skips past seasons already on disk
    python -m src.owgr --years 2025 2026
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
from pathlib import Path
from typing import List

import pandas as pd

from src.paths import OWGR_DIR
from src.scrape import _post, get_tournaments_for_year

_COLS = ["snapshot_date", "player_id", "owgr_rank", "owgr_points"]
_MONTHS = {m: i + 1 for i, m in enumerate("Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split())}
_QUERY = """
    query($year: Int, $eq: StatDetailEventQuery) {
        statDetails(tourCode: R, statId: "186", year: $year, eventQuery: $eq) {
            lastProcessed
            rows { ... on StatDetailsPlayer { playerId rank stats { statValue } } }
        }
    }
"""


def _snapshot(year: int, tournament_id: str = None) -> dict:
    eq = {"queryType": "THROUGH_EVENT", "tournamentId": tournament_id} if tournament_id else None
    return _post(_QUERY, {"year": year, "eq": eq})["statDetails"]


def _frame(date: pd.Timestamp, rows: List[dict]) -> pd.DataFrame:
    return pd.DataFrame({
        "snapshot_date": date,
        "player_id": [str(r["playerId"]) for r in rows],
        "owgr_rank": [int(r["rank"]) for r in rows],
        "owgr_points": [float(r["stats"][0]["statValue"]) for r in rows],   # average points
    })


def fetch_season(year: int) -> pd.DataFrame:
    """Every through-event snapshot the API has for one season, oldest first."""
    found = []
    for t in reversed(get_tournaments_for_year(year)):          # the API lists newest first
        d = _snapshot(year, t["tournament_id"])
        # "Through the Sony Open in Hawaii, Jan 14 2024"; seasons before 2024 omit the year.
        m = re.search(r", (\w{3}) (\d+)(?: (\d{4}))?\s*$", d["lastProcessed"] or "")
        if m and d["rows"]:
            found.append((_MONTHS[m.group(1)], int(m.group(2)), m.group(3), d["rows"]))
    if not found:
        return pd.DataFrame(columns=_COLS)
    # Without a year in the label, walk the season in order: a wraparound season
    # (up to 2023) opens in the previous autumn, and the year turns when the month drops.
    cal = year - 1 if year <= 2023 and found[0][0] >= 9 else year
    frames, prev = [], found[0][0]
    for month, day, stated, rows in found:
        if month < prev:
            cal += 1
        prev = month
        frames.append(_frame(pd.Timestamp(int(stated) if stated else cal, month, day), rows))
    return pd.concat(frames, ignore_index=True)


def fetch_latest(output_dir: Path = OWGR_DIR) -> None:
    """Save the current ranking as week_{date}.csv (one request)."""
    d = _snapshot(dt.date.today().year)
    label = d["lastProcessed"] or ""
    full = re.search(r"(\d{2})-(\d{2})-(\d{4})", label)      # "Through Week Ending: 12-29-2025"
    short = re.search(r", (\w{3}) (\d+)\s*$", label)         # "Through the Sony Open in Hawaii, Jan 14"
    if not d["rows"] or not (full or short):
        print("  OWGR: no current ranking returned")
        return
    if full:
        date = pd.Timestamp(int(full.group(3)), int(full.group(1)), int(full.group(2)))
    else:
        today = pd.Timestamp(dt.date.today())
        date = pd.Timestamp(today.year, _MONTHS[short.group(1)], int(short.group(2)))
        if date > today:                                     # a December label read in January
            date = date.replace(year=today.year - 1)
    output_dir.mkdir(parents=True, exist_ok=True)
    _frame(date, d["rows"]).to_csv(output_dir / f"week_{date:%Y-%m-%d}.csv", index=False)
    print(f"  OWGR: current ranking through {date:%Y-%m-%d}")


def run(years: List[int], output_dir: Path = OWGR_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    this_year = dt.date.today().year
    for year in years:
        path = output_dir / f"{year}.csv"
        if path.exists() and year < this_year:
            continue
        df = fetch_season(year)
        df.to_csv(path, index=False)
        print(f"  OWGR {year}: {df['snapshot_date'].nunique()} snapshots")
    fetch_latest(output_dir)


def load_snapshots(input_dir: Path = OWGR_DIR) -> pd.DataFrame:
    """All snapshots on disk: snapshot_date, player_id (API id, str), owgr_rank, owgr_points."""
    files = sorted(Path(input_dir).glob("*.csv"))
    if not files:
        return pd.DataFrame(columns=_COLS)
    df = pd.concat([pd.read_csv(f, dtype={"player_id": str}, parse_dates=["snapshot_date"]) for f in files])
    return df.drop_duplicates(["snapshot_date", "player_id"]).reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="+", type=int,
                        default=list(range(2014, dt.date.today().year + 1)))
    args = parser.parse_args()
    run(args.years)
