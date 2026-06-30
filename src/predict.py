"""
Score the upcoming tournament field with the trained model.

Usage:
    python -m src.predict --tournament R2026030   # scrape fresh field then score
    python -m src.predict --top 20                # re-score most recent saved features
    python -m src.predict --tournament R2026030 --top 30

Output is saved to data/processed/{tournament_id}_{slug}/projections.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from src.model import FEATURE_COLS, MODEL_PATH, load

PROCESSED_DIR = Path("data/processed")


def _tournament_dir(tournament_id: str, tournament_name: str) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "_", tournament_name.lower()).strip("_")
    return PROCESSED_DIR / f"{tournament_id}_{slug}"


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
    model_path: Path = MODEL_PATH,
) -> pd.DataFrame:
    """
    Score a weekly feature DataFrame.
    Returns a DataFrame with player_id, player_name, projected_dk_points, rank.
    """
    model = load(model_path)

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

    out["projected_dk_points"] = preds.round(2)
    out["rank"] = out["projected_dk_points"].rank(ascending=False).astype(int)
    out = out.sort_values("rank").reset_index(drop=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tournament", default=None,
        help="Tournament ID to scrape a fresh field for (e.g. R2026030). "
             "If omitted, re-scores the most recently saved features."
    )
    parser.add_argument("--top", type=int, default=None, help="Show only top N players")
    args = parser.parse_args()

    if args.tournament:
        from src.ingest import load_historical
        from src.scrape import build_weekly_features, get_upcoming_tournaments

        hist_path = Path("14-22 practice data.csv")
        hist = load_historical(hist_path) if hist_path.exists() else None
        weekly_df = build_weekly_features(args.tournament, historical_df=hist)

        # Derive output directory from tournament metadata
        t_name = weekly_df["tournament_name"].iloc[0] if "tournament_name" in weekly_df.columns else args.tournament
        out_dir = _tournament_dir(args.tournament, t_name)
        out_dir.mkdir(parents=True, exist_ok=True)

        features_path = out_dir / "weekly_features.parquet"
        weekly_df.to_parquet(features_path, index=False)
        print(f"Saved fresh weekly features → {features_path}\n")
    else:
        latest = _latest_projections()
        # Try to find matching features parquet
        features_path = None
        if latest:
            candidate = latest.parent / "weekly_features.parquet"
            if candidate.exists():
                features_path = candidate
        # Fall back to legacy flat path
        legacy = PROCESSED_DIR / "weekly_features.parquet"
        if features_path is None and legacy.exists():
            features_path = legacy

        if features_path is None:
            print("No weekly features found. Run with --tournament <id> first.")
            return
        weekly_df = pd.read_parquet(features_path)

        t_name = weekly_df["tournament_name"].iloc[0] if "tournament_name" in weekly_df.columns else "unknown"
        t_id = weekly_df["tournament_id"].iloc[0] if "tournament_id" in weekly_df.columns else "unknown"
        out_dir = _tournament_dir(t_id, t_name)
        out_dir.mkdir(parents=True, exist_ok=True)

    # Drop players with no recent SG data (all-NaN → unreliable model output)
    sg_cols = [c for c in FEATURE_COLS if "sg_" in c and c in weekly_df.columns]
    has_data = weekly_df[sg_cols].notna().any(axis=1) if sg_cols else pd.Series(True, index=weekly_df.index)
    dropped = (~has_data).sum()
    if dropped:
        print(f"Excluding {dropped} players with no recent SG data (amateurs/qualifiers).")
    weekly_df = weekly_df[has_data].reset_index(drop=True)

    projections = predict(weekly_df)

    proj_path = out_dir / "projections.csv"
    projections.to_csv(proj_path, index=False)

    # Print
    display = projections if args.top is None else projections.head(args.top)
    tournament = projections["tournament_name"].iloc[0] if "tournament_name" in projections.columns else "Unknown"
    course = projections["course"].iloc[0] if "course" in projections.columns else ""
    print(f"\n{'='*60}")
    print(f"  {tournament}  |  {course}")
    print(f"{'='*60}")
    print(f"{'Rank':<6}{'Player':<30}{'OWGR':<8}{'Proj. DK Pts':<14}")
    print("-" * 58)
    for _, row in display.iterrows():
        owgr = f"#{int(row['owgr'])}" if pd.notna(row.get("owgr")) else "—"
        print(f"  {row['rank']:<4}  {row['player_name']:<28}  {owgr:<6}  {row['projected_dk_points']:>6.1f}")

    print(f"\nFull projections saved → {proj_path}")


if __name__ == "__main__":
    main()
