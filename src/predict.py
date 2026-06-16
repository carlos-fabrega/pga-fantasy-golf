"""
Score the upcoming tournament field with the trained model.

Usage:
    python -m src.predict                         # score weekly_features.parquet
    python -m src.predict --tournament R2026026   # scrape fresh field then score
    python -m src.predict --top 20                # show top 20 only

Output is printed to stdout and saved to data/processed/projections.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.model import FEATURE_COLS, MODEL_PATH, load


WEEKLY_FEATURES_PATH = Path("data/processed/weekly_features.parquet")
OUTPUT_PATH = Path("data/processed/projections.csv")


def predict(
    weekly_df: pd.DataFrame,
    model_path: Path = MODEL_PATH,
) -> pd.DataFrame:
    """
    Score a weekly feature DataFrame.
    Returns a DataFrame with player_id, player_name, projected_dk_points, rank.
    """
    model = load(model_path)

    # Use only the features the model was trained on, in the right order.
    trained_features = list(model.feature_names_in_)
    present = [c for c in trained_features if c in weekly_df.columns]
    missing = [c for c in trained_features if c not in weekly_df.columns]

    if missing:
        print(f"Note: {len(missing)} training features absent from weekly data "
              f"(will be NaN): {missing}")
        for col in missing:
            weekly_df = weekly_df.copy()
            weekly_df[col] = float("nan")

    X = weekly_df[trained_features].astype(float)
    preds = model.predict(X)

    out = weekly_df[["player_id", "player_name"]].copy()
    if "owgr" in weekly_df.columns:
        out["owgr"] = weekly_df["owgr"]
    if "course" in weekly_df.columns:
        out["course"] = weekly_df["course"]
    if "tournament_name" in weekly_df.columns:
        out["tournament_name"] = weekly_df["tournament_name"]

    out["projected_dk_points"] = preds.round(2)
    out["rank"] = out["projected_dk_points"].rank(ascending=False).astype(int)
    out = out.sort_values("rank").reset_index(drop=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tournament", default=None,
        help="Tournament ID to scrape a fresh field for (e.g. R2026026). "
             "If omitted, uses weekly_features.parquet."
    )
    parser.add_argument("--top", type=int, default=None, help="Show only top N players")
    args = parser.parse_args()

    if args.tournament:
        from src.ingest import load_historical
        from src.scrape import build_weekly_features

        hist_path = Path("14-22 practice data.csv")
        hist = load_historical(hist_path) if hist_path.exists() else None
        weekly_df = build_weekly_features(args.tournament, historical_df=hist)
        weekly_df.to_parquet(WEEKLY_FEATURES_PATH, index=False)
        print(f"Saved fresh weekly features → {WEEKLY_FEATURES_PATH}\n")
    else:
        if not WEEKLY_FEATURES_PATH.exists():
            print(f"No weekly features found at {WEEKLY_FEATURES_PATH}.")
            print("Run `python -m src.scrape` first, or use --tournament <id>")
            return
        weekly_df = pd.read_parquet(WEEKLY_FEATURES_PATH)

    # Drop players with no recent SG data (all-NaN features → unreliable model output)
    from src.model import FEATURE_COLS
    sg_cols = [c for c in FEATURE_COLS if "sg_" in c and c in weekly_df.columns]
    has_data = weekly_df[sg_cols].notna().any(axis=1) if sg_cols else pd.Series(True, index=weekly_df.index)
    dropped = (~has_data).sum()
    if dropped:
        print(f"Excluding {dropped} players with no recent SG data (amateurs/qualifiers).")
    weekly_df = weekly_df[has_data].reset_index(drop=True)

    projections = predict(weekly_df)

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    projections.to_csv(OUTPUT_PATH, index=False)

    # Print
    display = projections if args.top is None else projections.head(args.top)
    tournament = weekly_df.get("tournament_name", pd.Series()).iloc[0] if len(weekly_df) else "Unknown"
    course = weekly_df.get("course", pd.Series()).iloc[0] if len(weekly_df) else ""
    print(f"\n{'='*60}")
    print(f"  {tournament}  |  {course}")
    print(f"{'='*60}")
    print(f"{'Rank':<6}{'Player':<30}{'OWGR':<8}{'Proj. DK Pts':<14}")
    print("-" * 58)
    for _, row in display.iterrows():
        owgr = f"#{int(row['owgr'])}" if pd.notna(row.get("owgr")) else "—"
        print(f"  {row['rank']:<4}  {row['player_name']:<28}  {owgr:<6}  {row['projected_dk_points']:>6.1f}")

    print(f"\nFull projections saved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
