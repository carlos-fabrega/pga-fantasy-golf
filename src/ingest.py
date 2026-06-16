"""
Load and standardize raw data from both sources:

  - Historical CSV (2014-2022): one row per player-tournament, includes DK points
  - Scraped CSVs (2023+): one row per player-tournament, SG only, made-cut players only

Both are normalized to a common schema before feature engineering.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

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

    df["source"] = "historical"
    return df.sort_values("date").reset_index(drop=True)


# ── Scraped CSVs ───────────────────────────────────────────────────────────────

# Column mapping for the stats-download CSV (stat 02675 / SG Total).
# The endpoint returns totals across measured rounds; we divide to get per-round.
_SCRAPE_RENAME = {
    "RANK": "finish_num",
    "PLAYER_ID": "player_id",
    "PLAYER": "player_name",
    "AVG": "sg_total_per_round",       # already per-round
    "TOTAL SG:T": "sg_total_event",    # total for the event
    "TOTAL SG:T2G": "sg_t2g_event",
    "TOTAL SG:P": "sg_putt_event",
    "MEASURED ROUNDS": "rounds_played",
}


def load_scraped(scraped_dir: str | Path) -> pd.DataFrame:
    """
    Load all R{year}{id}.csv files from scraped_dir (output from the PGA scraper).
    Returns a standardized DataFrame aligned with load_historical().
    Only contains players who made the cut (that's all the scraper captures).
    """
    scraped_dir = Path(scraped_dir)
    frames = []

    for f in sorted(scraped_dir.glob("R????.csv")):
        # Filename: R{year}{tournament_id}.csv  e.g. R2024003.csv
        stem = f.stem          # "R2024003"
        year = int(stem[1:5])  # 2024
        tid = stem[5:]         # "003"

        try:
            df = pd.read_csv(f)
        except Exception:
            continue

        df = df.rename(columns=_SCRAPE_RENAME)
        df["player_id"] = df["player_id"].astype(str)
        df["tournament_id"] = tid
        df["season"] = year
        df["made_cut"] = True

        # Compute per-round SG columns
        rounds = df["rounds_played"].replace(0, np.nan)
        if "sg_total_event" in df.columns:
            df["sg_total_per_round"] = df["sg_total_event"] / rounds
        if "sg_t2g_event" in df.columns:
            df["sg_t2g_per_round"] = df["sg_t2g_event"] / rounds
        if "sg_putt_event" in df.columns:
            df["sg_putt_per_round"] = df["sg_putt_event"] / rounds

        df["source"] = "scraped"
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    return combined


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
    "dk_points", "fd_points",
    "source",
]


def combine(historical: pd.DataFrame, scraped: pd.DataFrame) -> pd.DataFrame:
    """
    Merge historical and scraped DataFrames into one canonical DataFrame.
    Deduplicates on (player_id, tournament_id, season), preferring historical rows
    where both sources overlap.
    """
    frames = []
    if not historical.empty:
        frames.append(historical)
    if not scraped.empty:
        frames.append(scraped)

    if not frames:
        raise ValueError("Both historical and scraped DataFrames are empty.")

    df = pd.concat(frames, ignore_index=True)

    # Keep historical rows when there's overlap (historical has more columns).
    df["_source_rank"] = df["source"].map({"historical": 0, "scraped": 1}).fillna(1)
    df = df.sort_values("_source_rank").drop_duplicates(
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
