"""
Feature engineering for the PGA Tour prediction model.

All rolling/historical features are computed with shift(1) before the window,
so for tournament T a player's feature only uses data from T-1 and earlier.
This prevents any look-ahead leakage into the training set.

Input DataFrame must have been standardized by ingest.combine().
"""

from __future__ import annotations

from typing import Sequence, List

import numpy as np
import pandas as pd

# Substrings matched against lowercase tournament_name to identify links events.
# The Open Championship is the canonical links event; Scottish Open and Dunhill
# Links are also genuine links courses.  Irish Open venue varies but is often links.
# NOTE: historical CSV has sg_total_per_round = NaN for The Open, so this feature
# only carries signal at inference time (via the API's playerFinishStats history).
_LINKS_PATTERNS = [
    "the open",          # The Open Championship (all venues)
    "scottish open",     # Genesis/Aberdeen Scottish Open (Renaissance Club etc.)
    "dunhill links",     # Alfred Dunhill Links Championship
    "irish open",        # Dubai Duty Free Irish Open (links venues)
]


def is_links_tournament(tournament_name: str) -> bool:
    """Return True if the tournament name matches a known links event."""
    name = str(tournament_name).lower()
    return any(p in name for p in _LINKS_PATTERNS)


# Substrings matched against lowercase tournament_name to identify FedEx Cup
# Playoff events (small, no-cut, elite-only fields). Covers the pre-2019
# 4-event format (Barclays/Deutsche Bank or Dell Technologies/BMW/Tour
# Championship) and the WGC-branded 2019 St. Jude edition, all of which are
# safe historical-CSV matches. "fedex st. jude championship" is included for
# live/scraped data only — that exact name (post-2022 rebrand as the modern
# playoff opener) never appears in the 2014-2022 historical CSV, so it can't
# collide with the unrelated regular-season "FedEx St. Jude Classic" rows.
_PLAYOFF_PATTERNS = [
    "bmw championship",
    "tour championship",
    "the barclays",
    "dell technologies championship",
    "deutsche bank championship",
    "wgc-fedex st. jude classic",
    "fedex st. jude championship",
]


def is_playoff_tournament(tournament_name: str) -> bool:
    """Return True if the tournament name matches a known FedEx Cup Playoff event."""
    name = str(tournament_name).lower()
    return any(p in name for p in _PLAYOFF_PATTERNS)


# SG columns that exist in the historical data (per-round averages).
# Some (ott, app, arg) are absent from scraped data; we skip them gracefully.
_SG_COLS = [
    "sg_total_per_round",
    "sg_t2g_per_round",
    "sg_putt_per_round",
    "sg_ott_per_round",
    "sg_app_per_round",
    "sg_arg_per_round",
]

# Rolling window sizes measured in "tournament starts" (not weeks).
# min_periods = window // 4 so a player with fewer starts still gets a value.
_WINDOWS = [8, 16, 24]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _sort_key(df: pd.DataFrame) -> pd.Series:
    """
    Return a per-row integer sort key for temporal ordering.
    Prefers the 'date' column; falls back to (season * 1000 + intra-season rank).
    The fallback treats tournament_id as a string and ranks them within the season,
    which is approximately chronological since IDs increment through the year.
    """
    if "date" in df.columns and df["date"].notna().mean() > 0.5:
        return pd.to_datetime(df["date"]).astype("int64")

    # Rank tournament_id within each season (ties broken arbitrarily but consistently).
    intra = df.groupby("season")["tournament_id"].rank(method="dense").astype(int)
    return df["season"].astype(int) * 10_000 + intra


def _rolling_mean(series: pd.Series, window: int) -> pd.Series:
    """Leakage-safe flat rolling mean: shift by 1 before the window."""
    return series.shift(1).rolling(window, min_periods=max(1, window // 4)).mean()


def _ewm_mean(series: pd.Series, window: int) -> pd.Series:
    """
    Leakage-safe exponentially weighted mean: shift by 1 then ewm with span=window.
    More recent tournaments get higher weight than older ones.
    """
    return series.shift(1).ewm(span=window, min_periods=max(1, window // 4)).mean()


def _trailing_valid_mean(series: pd.Series, window: int) -> pd.Series:
    """
    Leakage-safe flat mean of up to the `window` most-recent NON-NULL values that
    occur strictly before each row.

    This matches inference exactly: scrape.build_weekly_features computes SG
    rolling features as the flat mean of a player's last N *SG-recorded* (made-cut)
    events. Rolling over the raw positional series instead would average over
    calendar starts and let missed-cut NaN gaps skew the window, diverging from
    what the live API can produce. See tests/test_feature_parity.py.
    """
    valid = series.dropna()
    trailing = valid.rolling(window, min_periods=1).mean().shift(1)
    # Map back to every row; carry the last known trailing mean across missed-cut
    # rows (which have no SG of their own) so they still carry recent form.
    return trailing.reindex(series.index).ffill()


# ── Rolling SG averages ────────────────────────────────────────────────────────

def add_rolling_sg(
    df: pd.DataFrame,
    sg_cols: Sequence[str] = _SG_COLS,
    windows: Sequence[int] = _WINDOWS,
) -> pd.DataFrame:
    """
    For each player, add rolling means of their SG stats over the last N tournaments.
    Only SG columns that exist in df are processed (missing ones are silently skipped).
    """
    df = df.copy()
    df["_order"] = _sort_key(df)
    df = df.sort_values(["player_id", "_order"])

    present_sg = [c for c in sg_cols if c in df.columns]

    for col in present_sg:
        for w in windows:
            feature_name = f"{col}_roll{w}"
            # Flat trailing mean over SG-recorded events, matching inference.
            df[feature_name] = df.groupby("player_id")[col].transform(
                lambda x, w=w: _trailing_valid_mean(x, w)
            )

    return df.drop(columns=["_order"]).reset_index(drop=True)


# ── Course history ─────────────────────────────────────────────────────────────

def add_course_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each player-tournament, add their historical average SG total at that course
    and the count of prior starts there — using only pre-tournament data.
    Rows where 'course' is null are skipped (no course history assigned).
    """
    df = df.copy()

    if "course" not in df.columns or df["course"].isna().all():
        df["course_sg_hist"] = np.nan
        df["course_starts"] = np.nan
        return df

    df["_order"] = _sort_key(df)
    df = df.sort_values(["player_id", "course", "_order"])

    def _expanding_mean_shifted(x: pd.Series) -> pd.Series:
        return x.shift(1).expanding().mean()

    def _expanding_count_shifted(x: pd.Series) -> pd.Series:
        return x.shift(1).expanding().count()

    df["course_sg_hist"] = df.groupby(["player_id", "course"])[
        "sg_total_per_round"
    ].transform(_expanding_mean_shifted)

    df["course_starts"] = df.groupby(["player_id", "course"])[
        "sg_total_per_round"
    ].transform(_expanding_count_shifted)

    return df.drop(columns=["_order"]).reset_index(drop=True)


# ── Links history ─────────────────────────────────────────────────────────────

def add_links_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each player-tournament, add their historical average SG total per round
    across all prior links events, and the count of those starts.

    links_sg_hist: expanding mean of sg_total_per_round at links venues (shifted
                   by 1 so the current tournament is never included).
    links_starts:  count of prior links starts with valid SG data.

    Limitation: the historical CSV has sg_total_per_round = NaN for The Open
    Championship, so this column is mostly NaN in training data.  It carries
    signal at inference time via the API's playerFinishStats history (which
    includes recent Scottish Opens and Opens where SG was recorded).
    """
    df = df.copy()

    if "tournament_name" not in df.columns:
        df["links_sg_hist"] = np.nan
        df["links_starts"]  = np.nan
        return df

    df["_order"]    = _sort_key(df)
    df["_links_sg"] = df["sg_total_per_round"].where(
        df["tournament_name"].apply(is_links_tournament)
    )
    df = df.sort_values(["player_id", "_order"])

    df["links_sg_hist"] = df.groupby("player_id")["_links_sg"].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    df["links_starts"] = df.groupby("player_id")["_links_sg"].transform(
        lambda x: x.shift(1).expanding().count()
    )

    return df.drop(columns=["_order", "_links_sg"]).reset_index(drop=True)


# ── Playoff history ────────────────────────────────────────────────────────────

def add_playoff_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each player-tournament, add their historical average SG total per round
    across all prior FedEx Cup Playoff events, and the count of those starts.

    playoff_sg_hist: expanding mean of sg_total_per_round at playoff events
                     (shifted by 1 so the current tournament is never included).
    playoff_starts:  count of prior playoff starts with valid SG data.
    """
    df = df.copy()

    if "tournament_name" not in df.columns:
        df["playoff_sg_hist"] = np.nan
        df["playoff_starts"]  = np.nan
        return df

    df["_order"]      = _sort_key(df)
    df["_playoff_sg"] = df["sg_total_per_round"].where(
        df["tournament_name"].apply(is_playoff_tournament)
    )
    df = df.sort_values(["player_id", "_order"])

    df["playoff_sg_hist"] = df.groupby("player_id")["_playoff_sg"].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    df["playoff_starts"] = df.groupby("player_id")["_playoff_sg"].transform(
        lambda x: x.shift(1).expanding().count()
    )

    return df.drop(columns=["_order", "_playoff_sg"]).reset_index(drop=True)


# ── Made-cut rate ──────────────────────────────────────────────────────────────

def add_made_cut_rate(
    df: pd.DataFrame,
    windows: Sequence[int] = _WINDOWS,
) -> pd.DataFrame:
    """Rolling fraction of starts where the player made the cut."""
    df = df.copy()
    df["_order"] = _sort_key(df)
    df = df.sort_values(["player_id", "_order"])

    for w in windows:
        df[f"made_cut_rate_roll{w}"] = df.groupby("player_id")["made_cut"].transform(
            lambda x, w=w: _rolling_mean(x.astype(float), w)
        )

    return df.drop(columns=["_order"]).reset_index(drop=True)


# ── Recent starts ──────────────────────────────────────────────────────────────

def add_recent_starts(df: pd.DataFrame, window: int = 8) -> pd.DataFrame:
    """
    Count of tournaments a player has started in the last N events.
    Low values flag players returning from injury or with thin history.
    """
    df = df.copy()
    df["_order"] = _sort_key(df)
    df = df.sort_values(["player_id", "_order"])

    df[f"starts_last{window}"] = df.groupby("player_id")["rounds_played"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).count()
    )

    return df.drop(columns=["_order"]).reset_index(drop=True)


# ── Master feature builder ─────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps in order.
    Returns the enriched DataFrame; original columns are preserved.
    """
    df = add_rolling_sg(df)
    df = add_course_history(df)
    df = add_links_history(df)
    df = add_playoff_history(df)
    df = add_made_cut_rate(df)
    df = add_recent_starts(df)
    return df


# ── Feature column catalog ─────────────────────────────────────────────────────

def feature_columns(df: pd.DataFrame) -> List[str]:
    """Return the list of engineered feature columns present in df."""
    roll_pattern = "_roll"
    other = ["course_sg_hist", "course_starts", "starts_last8"]
    return [
        c for c in df.columns
        if roll_pattern in c or c in other
    ]
