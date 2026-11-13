"""
Load and standardize raw data from both sources:

  - Historical CSV (2014-2022): one row per player-tournament, includes purse and finish pos
  - Scraped CSVs (mid-2022+, from src.backfill): one row per player-tournament for the
    whole field; SG for made-cut players only

Both are normalized to a common schema before feature engineering.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

from src.crosswalk import build_players, match_key

# ── PGA Tour payout table ──────────────────────────────────────────────────────
# Payout as a fraction of total purse by finish position.
# Source: standard PGA Tour distribution (approximate).
# Missed cuts receive 0. Positions beyond the table receive the minimum entry.
_PAYOUT_PCTS = {
     1: 0.1800,  2: 0.1090,  3: 0.0690,  4: 0.0490,  5: 0.0410,
     6: 0.0370,  7: 0.0340,  8: 0.0310,  9: 0.0290, 10: 0.0270,
    11: 0.0250, 12: 0.0235, 13: 0.0220, 14: 0.0205, 15: 0.0190,
    16: 0.0175, 17: 0.0165, 18: 0.0155, 19: 0.0145, 20: 0.0135,
    21: 0.0125, 22: 0.0115, 23: 0.0105, 24: 0.00975,25: 0.00900,
    26: 0.00845,27: 0.00790,28: 0.00745,29: 0.00700,30: 0.00655,
    31: 0.00615,32: 0.00575,33: 0.00540,34: 0.00505,35: 0.00475,
    36: 0.00450,37: 0.00425,38: 0.00400,39: 0.00375,40: 0.00350,
    41: 0.00330,42: 0.00310,43: 0.00290,44: 0.00270,45: 0.00255,
    46: 0.00240,47: 0.00225,48: 0.00215,49: 0.00205,50: 0.00195,
    51: 0.00185,52: 0.00180,53: 0.00175,54: 0.00170,55: 0.00165,
    56: 0.00160,57: 0.00155,58: 0.00150,59: 0.00145,60: 0.00140,
    61: 0.00135,62: 0.00130,63: 0.00125,64: 0.00120,65: 0.00115,
}
_PAYOUT_MIN = 0.00110  # applied to positions beyond 65 that still made the cut


def pos_to_prize_pct(pos: float, made_cut: bool) -> float:
    """Convert numeric finish position to prize share (fraction of purse)."""
    if not made_cut or pd.isna(pos):
        return 0.0
    p = int(round(pos))
    return _PAYOUT_PCTS.get(p, _PAYOUT_MIN)

# ── Historical CSV ─────────────────────────────────────────────────────────────

_HIST_RENAME = {
    "Player_initial_last": "player_abbr",
    "tournament id": "tournament_id",
    "player id": "player_id",
    "hole_par": "total_par",
    "strokes": "total_strokes",
    "n_rounds": "rounds_played",
    "made_cut": "made_cut",
    "pos": "pos",
    "total_DKP": "dk_points",
    "total_FDP": "fd_points",
    "player": "player_name",
    "tournament name": "tournament_name",
    "course": "course",
    "date": "date",
    "purse": "purse",
    "season": "season",
    "no_cut": "no_cut",
    "Finish": "finish",
    "sg_putt": "sg_putt",
    "sg_arg": "sg_arg",
    "sg_app": "sg_app",
    "sg_ott": "sg_ott",
    "sg_t2g": "sg_t2g",
    "sg_total": "sg_total",
}

_HIST_DROP = {
    "hole_DKP", "hole_FDP", "hole_SDP",
    "streak_DKP", "streak_FDP", "streak_SDP",
    "finish_DKP", "finish_FDP", "finish_SDP",
    "total_SDP",
    "Unnamed: 2", "Unnamed: 3", "Unnamed: 4",
}

# In the historical CSV, sg_* columns are per-round averages.
_HIST_SG_COLS = ["sg_putt", "sg_arg", "sg_app", "sg_ott", "sg_t2g", "sg_total"]


def load_historical(path: str | Path) -> pd.DataFrame:
    """
    Load the 2014-2022 historical CSV and return a standardized DataFrame.
    SG columns in this file are per-round averages; we store them as-is and also
    compute per-round equivalents so they match the scraped format.
    """
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in _HIST_DROP if c in df.columns])
    df = df.rename(columns=_HIST_RENAME)

    df["date"] = pd.to_datetime(df["date"])
    df["season"] = df["season"].astype(int)
    df["player_id"] = df["player_id"].astype(str)
    df["tournament_id"] = df["tournament_id"].astype(str)
    df["made_cut"] = df["made_cut"].astype(bool)
    df["rounds_played"] = df["rounds_played"].astype(float)

    # Numeric finish position (strip leading "T" from ties)
    df["finish_num"] = (
        df["finish"].astype(str)
        .str.replace("T", "", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
    )

    # Historical SG values are already per-round averages.
    # Rename to *_per_round for schema consistency.
    for col in _HIST_SG_COLS:
        if col in df.columns:
            df[f"{col}_per_round"] = df[col]

    # Prize money as a fraction of purse — primary model target.
    # Purse is in millions in the CSV; prize_pct is dimensionless.
    df["prize_pct"] = [
        pos_to_prize_pct(pos, mc)
        for pos, mc in zip(df["finish_num"], df["made_cut"])
    ]

    df["source"] = "historical"
    return df.sort_values("date", kind="stable").reset_index(drop=True)


# ── Scraped CSVs ───────────────────────────────────────────────────────────────

def _parse_usd(val) -> float:
    s = str(val).replace("$", "").replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return float("nan")


def load_scraped(scraped_dir: str | Path) -> pd.DataFrame:
    """
    Load every R{year}{id}.csv that src.backfill wrote to scraped_dir and return
    a DataFrame aligned with load_historical(): the whole field per event
    (missed cuts included), per-round SG where the API has it (made-cut players
    at ShotLink events), and prize_pct from the same payout table as the
    historical data so the target means the same thing in both eras.
    """
    frames = []
    for f in sorted(Path(scraped_dir).glob("R*.csv")):
        # player_id as str: API ids can have leading zeros ("01706").
        frames.append(pd.read_csv(f, dtype={"player_id": str, "tournament_id": str}))
    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df["season"] = df["season"].astype(int)
    df["made_cut"] = df["made_cut"].astype(bool)
    df["rounds_played"] = df["rounds_played"].astype(float)
    df["purse"] = df["purse"].map(_parse_usd) / 1e6      # millions, like the historical CSV
    df["prize_pct"] = [
        pos_to_prize_pct(pos, mc) for pos, mc in zip(df["finish_num"], df["made_cut"])
    ]
    df["source"] = "scraped"
    return df.sort_values("date").reset_index(drop=True)


# ── Combine ────────────────────────────────────────────────────────────────────

# All columns present in the final combined DataFrame (NaN where not available).
_SCHEMA = [
    "player_id", "player_name", "player_abbr",
    "tournament_id", "tournament_name", "course",
    "date", "season",
    "rounds_played", "made_cut", "finish_num",
    "total_par", "total_strokes",
    "sg_total_per_round", "sg_t2g_per_round", "sg_putt_per_round",
    "sg_ott_per_round", "sg_app_per_round", "sg_arg_per_round",
    "purse", "prize_pct", "dk_points", "fd_points",
    "source",
]


def course_key(course) -> str:
    """
    Course name reduced to a join key shared by both sources: the historical
    CSV says "Muirfield Village Golf Club - Dublin, OH", the API says
    "Muirfield Village Golf Club".
    """
    if pd.isna(course):
        return ""
    return str(course).split(" - ")[0].split(",")[0].strip().lower()


def _course_bridge(historical: pd.DataFrame, scraped: pd.DataFrame) -> dict:
    """
    {historical course_key: API course_key}, learned from events both sources
    cover (the API reaches back before the historical CSV ends). The two name
    the same course too differently to normalise ("Bay Hill - Orlando, FL" vs
    "Arnold Palmer's Bay Hill Club & Lodge"), so an event is paired across
    sources by its winner and date instead, and the course names follow.
    """
    def winners(df: pd.DataFrame) -> pd.DataFrame:
        w = df[df["finish_num"] == 1]
        return pd.DataFrame({"winner": w["player_name"].map(match_key), "date": w["date"],
                             "course_key": w["course"].map(course_key)})

    pairs = winners(historical).merge(winners(scraped), on="winner", suffixes=("_h", "_s"))
    pairs = pairs[(pairs["date_h"] - pairs["date_s"]).abs() <= pd.Timedelta(days=7)]
    return dict(zip(pairs["course_key_h"], pairs["course_key_s"]))


def combine(historical: pd.DataFrame, scraped: pd.DataFrame) -> pd.DataFrame:
    """
    Merge historical and scraped DataFrames into one canonical DataFrame.

    The two sources use different player ids for the same golfer, so when both
    are present every row is put on one id: the API id where the crosswalk
    links the golfer (that is what live data uses), else "H" + the historical
    id so the two id spaces can't collide. The original id of a historical row
    is kept in hist_player_id. Without this a golfer's rolling history would
    restart from nothing at the 2022 boundary.

    Scraped events that started on or before the last historical date are
    dropped: the API covers those too, under different tournament ids, so they
    would otherwise appear twice. They are first used to learn which course
    names in the two sources are the same course (_course_bridge).
    """
    frames = []
    if not historical.empty:
        frames.append(historical)
    if not scraped.empty:
        frames.append(scraped)

    if not frames:
        raise ValueError("Both historical and scraped DataFrames are empty.")

    bridge = {}
    if not historical.empty and not scraped.empty:
        bridge = _course_bridge(historical, scraped)
        scraped = scraped[scraped["date"] > historical["date"].max()]
        players, _ = build_players(
            historical[["player_id", "player_name"]], scraped[["player_id", "player_name"]])
        from_hist = players.dropna(subset=["hist_id"])     # linked → API id, unlinked → "H" + id
        to_canonical = dict(zip(from_hist["hist_id"], from_hist["player_id"]))
        historical = historical.assign(
            hist_player_id=historical["player_id"],
            player_id=historical["player_id"].map(to_canonical),
        )
        frames = [historical, scraped]

    df = pd.concat(frames, ignore_index=True)
    df["course_key"] = df["course"].map(course_key)
    df["course_key"] = df["course_key"].map(bridge).fillna(df["course_key"])

    # Keep historical rows when there's overlap (historical has more columns).
    df["_source_rank"] = df["source"].map({"historical": 0, "scraped": 1}).fillna(1)
    # Stable sort: the historical CSV has a few player-events twice with conflicting
    # rows, and which one survives must not depend on how many rows are loaded.
    df = df.sort_values("_source_rank", kind="stable").drop_duplicates(
        subset=["player_id", "tournament_id", "season"], keep="first"
    ).drop(columns=["_source_rank"])

    # Reorder to schema; add missing columns as NaN.
    for col in _SCHEMA:
        if col not in df.columns:
            df[col] = np.nan

    df = df[_SCHEMA + [c for c in df.columns if c not in _SCHEMA]]

    # Final sort: chronological when we have dates, otherwise by season + tournament.
    if df["date"].notna().any():
        df = df.sort_values(["date", "player_id"]).reset_index(drop=True)
    else:
        df = df.sort_values(["season", "tournament_id", "player_id"]).reset_index(drop=True)

    return df
