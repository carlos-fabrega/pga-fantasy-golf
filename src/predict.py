"""
Score the upcoming tournament field with the trained model.

Usage:
    python -m src.predict --tournament R2026030          # base model
    python -m src.predict --tournament R2026100 --links  # links-tuned model
    python -m src.predict --top 20                       # re-score most recent saved features

Output is saved to data/processed/{tournament_id}_{slug}/projections.csv
For links model: projections_links.csv in the same folder.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from src.model import FEATURE_COLS, FEATURE_COLS_LINKS, load
from src.paths import PROCESSED_DIR, HISTORICAL_CSV


def _tournament_dir(tournament_id: str, tournament_name: str) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "_", tournament_name.lower()).strip("_")
    return PROCESSED_DIR / f"{tournament_id}_{slug}"


def _latest_features() -> Optional[Path]:
    """Return the most recently modified weekly_features.parquet across all tournament folders."""
    candidates = sorted(
        PROCESSED_DIR.glob("*/weekly_features.parquet"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _latest_projections() -> Optional[Path]:
    """Return the most recently modified projections.csv across all tournament folders."""
    candidates = sorted(
        PROCESSED_DIR.glob("*/projections.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def predict(
    weekly_df: pd.DataFrame,
    links: bool = False,
) -> pd.DataFrame:
    """
    Score a weekly feature DataFrame.
    links=True uses the links-tuned model and FEATURE_COLS_LINKS.
    Returns a DataFrame with player_id, player_name, projected_prize_pct, rank.
    """
    model = load(links=links)
    feature_cols = FEATURE_COLS_LINKS if links else FEATURE_COLS

    trained_features = list(model.feature_names_in_)
    missing = [c for c in trained_features if c not in weekly_df.columns]

    if missing:
        print(f"Note: {len(missing)} training features absent from weekly data "
              f"(will be NaN): {missing}")
        weekly_df = weekly_df.copy()
        for col in missing:
            weekly_df[col] = float("nan")

    X = weekly_df[trained_features].astype(float)
    preds = model.predict(X)

    out = weekly_df[["player_id", "player_name"]].copy()
    for col in ("owgr", "course", "tournament_id", "tournament_name"):
        if col in weekly_df.columns:
            out[col] = weekly_df[col]

    out["projected_prize_pct"] = preds.round(6)
    out["rank"] = out["projected_prize_pct"].rank(ascending=False).astype(int)
    out = out.sort_values("rank").reset_index(drop=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tournament", default=None,
        help="Tournament ID to scrape a fresh field for (e.g. R2026030). "
             "If omitted, re-scores the most recently saved features."
    )
    parser.add_argument("--top",   type=int, default=None, help="Show only top N players")
    parser.add_argument("--links", action="store_true",
                        help="Use the links-tuned model (model_links.pkl)")
    args = parser.parse_args()

    if args.tournament:
        from src.ingest import load_historical
        from src.scrape import build_weekly_features

        hist = load_historical(HISTORICAL_CSV) if HISTORICAL_CSV.exists() else None
        weekly_df = build_weekly_features(args.tournament, historical_df=hist)

        t_name = weekly_df["tournament_name"].iloc[0] if "tournament_name" in weekly_df.columns else args.tournament
        out_dir = _tournament_dir(args.tournament, t_name)
        out_dir.mkdir(parents=True, exist_ok=True)

        features_path = out_dir / "weekly_features.parquet"
        weekly_df.to_parquet(features_path, index=False)
        print(f"Saved fresh weekly features → {features_path}\n")
    else:
        features_path = _latest_features()
        legacy = PROCESSED_DIR / "weekly_features.parquet"
        if features_path is None and legacy.exists():
            features_path = legacy

        if features_path is None:
            print("No weekly features found. Run with --tournament <id> first.")
            return
        weekly_df = pd.read_parquet(features_path)

        t_name = weekly_df["tournament_name"].iloc[0] if "tournament_name" in weekly_df.columns else "unknown"
        t_id   = weekly_df["tournament_id"].iloc[0]   if "tournament_id"   in weekly_df.columns else "unknown"
        out_dir = _tournament_dir(t_id, t_name)
        out_dir.mkdir(parents=True, exist_ok=True)

    # Drop players with no recent SG data
    feature_cols = FEATURE_COLS_LINKS if args.links else FEATURE_COLS
    sg_cols = [c for c in feature_cols if "sg_" in c and c in weekly_df.columns]
    has_data = weekly_df[sg_cols].notna().any(axis=1) if sg_cols else pd.Series(True, index=weekly_df.index)
    dropped = (~has_data).sum()
    if dropped:
        print(f"Excluding {dropped} players with no recent SG data (amateurs/qualifiers).")
    weekly_df = weekly_df[has_data].reset_index(drop=True)

    projections = predict(weekly_df, links=args.links)

    proj_filename = "projections_links.csv" if args.links else "projections.csv"
    proj_path = out_dir / proj_filename
    projections.to_csv(proj_path, index=False)

    # Print
    model_label = "[LINKS MODEL]" if args.links else "[BASE MODEL]"
    display = projections if args.top is None else projections.head(args.top)
    tournament = projections["tournament_name"].iloc[0] if "tournament_name" in projections.columns else "Unknown"
    course     = projections["course"].iloc[0]          if "course"           in projections.columns else ""
    print(f"\n{'='*64}")
    print(f"  {tournament}  |  {course}  {model_label}")
    print(f"{'='*64}")
    print(f"{'Rank':<6}{'Player':<30}{'OWGR':<8}{'Proj. Score':<14}")
    print("-" * 60)
    for _, row in display.iterrows():
        owgr = f"#{int(row['owgr'])}" if pd.notna(row.get("owgr")) else "—"
        print(f"  {row['rank']:<4}  {row['player_name']:<28}  {owgr:<6}  {row['projected_prize_pct']*100:>7.2f}%")

    print(f"\nFull projections saved → {proj_path}")


if __name__ == "__main__":
    main()
