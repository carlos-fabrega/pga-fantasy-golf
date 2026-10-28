"""
Score the upcoming tournament field with the trained model.

Usage:
    python -m src.predict                       # auto-detect next event; links model auto-selected
    python -m src.predict --tournament R2026030 # score a specific event
    python -m src.predict --no-links            # force the base model on a links event
    python -m src.predict --no-nocut            # force the base model on a no-cut event
    python -m src.predict --latest              # re-score most recent saved features (no scrape)

Output is saved to data/processed/{tournament_id}_{slug}/projections.csv
For links model: projections_links.csv in the same folder.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from src.model import VARIANT_FEATURES, load
from src.features import is_links_tournament, is_nocut_tournament
from src.paths import PROCESSED_DIR, HISTORICAL_CSV


def tournament_dir(tournament_id: str, tournament_name: str) -> Path:
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


def select_variant(
    tournament_name: str,
    links: Optional[bool] = None,
    nocut: Optional[bool] = None,
    playoff: Optional[bool] = None,
) -> str:
    """
    Decide base vs links vs nocut vs playoff. An explicit flag always wins
    (True forces that variant on; False forces "base"); None means auto-detect
    links (Open/Scottish/Dunhill/Irish) and no-cut (Tour Championship/BMW/
    Sentry/WGCs — see features._NOCUT_PATTERNS).

    Links is checked first, but the two sets do not currently overlap: every
    links event on the calendar has a 36-hole cut.
    NOTE: the playoff variant is NOT auto-selected for BMW/Tour Championship —
    backtesting (see src/model.py FEATURE_COLS_PLAYOFF comment) showed it
    underperforms the base model on held-out playoff seasons (too little
    training data: ~23 historical playoff editions). Pass playoff=True
    explicitly to use it anyway.
    """
    forced = {"links": links, "playoff": playoff, "nocut": nocut}
    if any(v is True for v in forced.values()):
        return next(k for k, v in forced.items() if v is True)
    if any(v is False for v in forced.values()):
        return "base"
    if is_links_tournament(tournament_name):
        return "links"
    if is_nocut_tournament(tournament_name):
        return "nocut"
    return "base"


def align_features(weekly_df: pd.DataFrame, model) -> pd.DataFrame:
    """
    Return the model's input matrix: its trained feature columns, in training
    order, with any column absent from weekly_df filled as NaN. Shared by
    scoring and explanation so both see exactly the same matrix.
    """
    trained_features = list(model.feature_names_in_)
    missing = [c for c in trained_features if c not in weekly_df.columns]

    if missing:
        print(f"Note: {len(missing)} training features absent from weekly data "
              f"(will be NaN): {missing}")
        weekly_df = weekly_df.copy()
        for col in missing:
            weekly_df[col] = float("nan")

    return weekly_df[trained_features].astype(float)


def predict(
    weekly_df: pd.DataFrame,
    variant: str = "base",
) -> pd.DataFrame:
    """
    Score a weekly feature DataFrame.
    variant selects the model/feature set ("base", "links", or "playoff").
    Returns a DataFrame with player_id, player_name, projected_prize_pct, rank.
    """
    model = load(variant=variant)
    X = align_features(weekly_df, model)
    preds = model.predict(X)

    out = weekly_df[["player_id", "player_name"]].copy()
    for col in ("owgr", "course", "tournament_id", "tournament_name"):
        if col in weekly_df.columns:
            out[col] = weekly_df[col]

    out["projected_prize_pct"] = preds.round(6)
    out["rank"] = out["projected_prize_pct"].rank(ascending=False).astype(int)
    out = out.sort_values("rank").reset_index(drop=True)
    return out


def score_field(weekly_df: pd.DataFrame, variant: str = "base") -> pd.DataFrame:
    """
    Drop players with no recent SG data (amateurs/qualifiers), then score the
    rest with predict(). Returns the projections DataFrame.
    """
    feature_cols = VARIANT_FEATURES[variant]
    sg_cols = [c for c in feature_cols if "sg_" in c and c in weekly_df.columns]
    has_data = weekly_df[sg_cols].notna().any(axis=1) if sg_cols else pd.Series(True, index=weekly_df.index)
    dropped = (~has_data).sum()
    if dropped:
        print(f"Excluding {dropped} players with no recent SG data (amateurs/qualifiers).")
    weekly_df = weekly_df[has_data].reset_index(drop=True)

    return predict(weekly_df, variant=variant)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tournament", default=None,
        help="Tournament ID to scrape a fresh field for (e.g. R2026030). "
             "If omitted, auto-detects the next upcoming event."
    )
    parser.add_argument("--top",   type=int, default=None, help="Show only top N players")
    # Tri-state links/playoff: explicit flag forces on/off, neither = auto-detect.
    parser.add_argument("--links",    dest="links", action="store_true",
                        help="Force the links-tuned model on.")
    parser.add_argument("--no-links", dest="links", action="store_false",
                        help="Force the base model, even on a links event.")
    parser.set_defaults(links=None)
    parser.add_argument("--playoff",    dest="playoff", action="store_true",
                        help="Force the FedEx Cup Playoff-tuned model on.")
    parser.add_argument("--no-playoff", dest="playoff", action="store_false",
                        help="Force the base model, even on a playoff event.")
    parser.set_defaults(playoff=None)
    parser.add_argument("--nocut", dest="nocut", action="store_true",
                        help="Force the no-cut model (base features minus made_cut_rate_*) on.")
    parser.add_argument("--no-nocut", dest="nocut", action="store_false",
                        help="Force the base model, even on a no-cut event.")
    parser.set_defaults(nocut=None)
    parser.add_argument("--latest", action="store_true",
                        help="Re-score the most recently saved weekly features instead of "
                             "scraping a fresh field.")
    args = parser.parse_args()

    # Auto-detect the next upcoming event when no tournament is given and we're not re-scoring.
    if not args.tournament and not args.latest:
        from src.scrape import get_upcoming_tournaments
        upcoming = get_upcoming_tournaments(1)
        if not upcoming:
            print("No upcoming tournaments found. Pass --tournament <id> explicitly.")
            return
        args.tournament = upcoming[0]["tournament_id"]
        print(f"Auto-detected next event: {upcoming[0]['tournament_name']} "
              f"@ {upcoming[0]['course']}  ({args.tournament})\n")

    if args.tournament:
        from src.ingest import load_historical
        from src.scrape import build_weekly_features

        hist = load_historical(HISTORICAL_CSV) if HISTORICAL_CSV.exists() else None
        weekly_df = build_weekly_features(args.tournament, historical_df=hist)

        t_name = weekly_df["tournament_name"].iloc[0] if "tournament_name" in weekly_df.columns else args.tournament
        out_dir = tournament_dir(args.tournament, t_name)
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
        out_dir = tournament_dir(t_id, t_name)
        out_dir.mkdir(parents=True, exist_ok=True)

    t_name = weekly_df["tournament_name"].iloc[0] if "tournament_name" in weekly_df.columns else ""
    variant = select_variant(t_name, links=args.links, nocut=args.nocut, playoff=args.playoff)

    # Announce only auto-detected variants; an explicit flag needs no explanation.
    auto = args.links is None and args.nocut is None and args.playoff is None
    if auto and variant == "links":
        print(f"'{t_name}' is a links event → using the links-tuned model.\n")
    elif auto and variant == "nocut":
        print(f"'{t_name}' is a no-cut event → using the no-cut model "
              f"(made_cut_rate_* dropped). Pass --no-nocut to override.\n")

    projections = score_field(weekly_df, variant=variant)

    proj_filename = "projections.csv" if variant == "base" else f"projections_{variant}.csv"
    proj_path = out_dir / proj_filename
    projections.to_csv(proj_path, index=False)

    # Print
    model_label = f"[{variant.upper()} MODEL]"
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
