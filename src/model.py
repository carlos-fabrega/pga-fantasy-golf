"""
Model training, temporal cross-validation, evaluation, and serialization.

Design choices:
  - XGBoost with NaN pass-through (no imputation needed — missing history is
    naturally encoded as NaN and XGBoost splits on it correctly)
  - Temporal expanding-window CV: train on seasons up to year N, test on N+1.
    Random splits are forbidden here because future SG data would leak into
    rolling features computed from the same pool.
  - Target: prize_pct (player's share of tournament purse, e.g. 0.18 for a win)
  - Primary eval metric: per-tournament Spearman rank correlation — this is
    what matters for lineup selection, not raw RMSE.

Usage:
    python -m src.model          # train, evaluate, save model
    python -m src.model --cv     # show full expanding-window CV results
"""

from __future__ import annotations

import argparse
import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from xgboost import XGBRanker

from src.paths import MODEL_BASE_PATH, MODEL_LINKS_PATH, FEATURE_MATRIX

# ── Feature / target config ────────────────────────────────────────────────────

FEATURE_COLS: List[str] = [
    # Rolling SG averages — all six categories × three windows
    "sg_total_per_round_roll8",  "sg_total_per_round_roll16",  "sg_total_per_round_roll24",
    "sg_t2g_per_round_roll8",    "sg_t2g_per_round_roll16",    "sg_t2g_per_round_roll24",
    "sg_app_per_round_roll8",    "sg_app_per_round_roll16",    "sg_app_per_round_roll24",
    "sg_arg_per_round_roll8",    "sg_arg_per_round_roll16",    "sg_arg_per_round_roll24",
    "sg_ott_per_round_roll8",    "sg_ott_per_round_roll16",    "sg_ott_per_round_roll24",
    "sg_putt_per_round_roll8",   "sg_putt_per_round_roll16",   "sg_putt_per_round_roll24",
    # Course and consistency features
    "course_sg_hist",
    "course_starts",
    "made_cut_rate_roll8",  "made_cut_rate_roll16",  "made_cut_rate_roll24",
    "starts_last8",
]

# Links model adds historical performance across all links courses as a signal.
# links_sg_hist: player's average SG total per round at links venues (all-time).
# links_starts:  number of prior links starts (confidence weight for the above).
FEATURE_COLS_LINKS: List[str] = FEATURE_COLS + ["links_sg_hist", "links_starts"]

TARGET_COL = "prize_pct"

# Relevance grades for rank:ndcg (must be non-negative integers).
# Mapped from prize_pct so NDCG's 2^grade-1 gain mirrors the prize distribution.
#   0 → missed cut          (prize_pct = 0)
#   1 → made cut, pos 25+   (prize_pct ~0.001–0.009)
#   2 → pos 12–24           (prize_pct ~0.009–0.025)
#   3 → top 11, pos 6–11    (prize_pct ~0.025–0.041)
#   4 → top 5               (prize_pct ~0.041–0.109)
#   5 → win                 (prize_pct ≥ 0.109)
def _to_grade(pct: float) -> int:
    if pct <= 0.0:    return 0
    if pct < 0.009:   return 1
    if pct < 0.025:   return 2
    if pct < 0.041:   return 3
    if pct < 0.109:   return 4
    return 5


# Backwards-compat alias used by predict.py when no --links flag is passed
MODEL_PATH = MODEL_BASE_PATH


# ── XGBoost config ─────────────────────────────────────────────────────────────

def _make_model(n_estimators: int = 800) -> XGBRanker:
    # LambdaRank (rank:ndcg) directly optimises ranking rather than regressing
    # on prize_pct.  prize_pct is used as the relevance score: winner (0.18)
    # is most relevant, missed cut (0.0) is least.  This avoids the target-scale
    # regularisation problem that plagued the regression approach.
    return XGBRanker(
        objective="rank:ndcg",
        n_estimators=n_estimators,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )


# ── Data prep ──────────────────────────────────────────────────────────────────

def prepare(df: pd.DataFrame, links: bool = False) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """
    Return (X, y, groups) for the LambdaRank model.

    links=True uses FEATURE_COLS_LINKS (adds links_sg_hist, links_starts).
    All rows with a non-null target and at least one non-null SG feature are
    included — missed-cut players (prize_pct=0) are valid lowest-relevance
    examples that help the model learn to rank them below made-cut players.

    Data is sorted by tournament_id so XGBRanker's group array is contiguous.
    groups[i] = number of rows belonging to tournament i.
    """
    feature_cols = FEATURE_COLS_LINKS if links else FEATURE_COLS

    has_target = df[TARGET_COL].notna()
    has_any_sg = df[[c for c in feature_cols if "sg_" in c and c in df.columns]].notna().any(axis=1)
    mask = has_target & has_any_sg

    sub = df[mask].copy()
    # Must be sorted by tournament so groups are contiguous blocks
    sub = sub.sort_values(["season", "tournament_id"])

    present_features = [c for c in feature_cols if c in df.columns]
    X = sub[present_features].astype(float)
    # Convert continuous prize_pct to integer relevance grades for rank:ndcg.
    # The original prize_pct is retained in sub for Spearman evaluation.
    y = sub[TARGET_COL].apply(_to_grade).astype(int)
    groups = sub.groupby("tournament_id", sort=False).size().values
    return X, y, groups


# ── Evaluation ─────────────────────────────────────────────────────────────────

def _spearman_by_tournament(
    df_test: pd.DataFrame,
    y_pred: np.ndarray,
    min_players: int = 20,
) -> pd.Series:
    """
    Per-tournament Spearman rank correlation between predicted scores and prize_pct.
    Skips tournaments with fewer than min_players or constant actual/predicted values
    (e.g. no-cut events where all payouts are identical).
    """
    tmp = df_test[["tournament_id", TARGET_COL]].copy()
    tmp["pred"] = y_pred
    results = {}
    for tid, grp in tmp.groupby("tournament_id"):
        valid = grp.dropna(subset=[TARGET_COL, "pred"])
        if len(valid) < min_players:
            continue
        if valid[TARGET_COL].std() == 0 or valid["pred"].std() == 0:
            continue
        r, _ = spearmanr(valid[TARGET_COL], valid["pred"])
        results[tid] = r
    return pd.Series(results)


def evaluate(
    model: XGBRanker,
    df_test: pd.DataFrame,
    label: str = "Test",
    links: bool = False,
) -> Dict[str, float]:
    """
    Evaluate a fitted model on df_test.
    Returns a dict of scalar metrics and prints a summary.
    """
    X_test, y_test, _ = prepare(df_test, links=links)
    if X_test.empty:
        print(f"[{label}] No valid rows to evaluate.")
        return {}

    y_pred = model.predict(X_test)
    residuals = y_test.values - y_pred

    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae  = float(np.mean(np.abs(residuals)))

    spear_by_t = _spearman_by_tournament(
        df_test.loc[X_test.index], y_pred
    )

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "spearman_mean": float(spear_by_t.mean()),
        "spearman_median": float(spear_by_t.median()),
        "spearman_p25": float(spear_by_t.quantile(0.25)),
        "spearman_p75": float(spear_by_t.quantile(0.75)),
        "n_tournaments": len(spear_by_t),
        "n_rows": len(y_test),
    }

    print(f"\n── {label} ({metrics['n_rows']:,} rows, {metrics['n_tournaments']} tournaments) ──")
    print(f"  RMSE : {rmse:.2f}  |  MAE : {mae:.2f}")
    print(f"  Spearman ρ  mean={metrics['spearman_mean']:+.3f}  "
          f"median={metrics['spearman_median']:+.3f}  "
          f"[{metrics['spearman_p25']:+.3f}, {metrics['spearman_p75']:+.3f}]")
    return metrics


# ── Temporal cross-validation ──────────────────────────────────────────────────

def temporal_cv(df: pd.DataFrame, min_train_seasons: int = 3, links: bool = False) -> List[Dict]:
    """
    Expanding-window temporal CV.
    For each season S (starting after min_train_seasons), train on all seasons
    < S and test on season S. Reports metrics per fold.
    """
    seasons = sorted(df["season"].unique())
    results = []

    for i, test_season in enumerate(seasons):
        if i < min_train_seasons:
            continue  # need enough history to train

        train_seasons = seasons[:i]
        df_train = df[df["season"].isin(train_seasons)]
        df_test  = df[df["season"] == test_season]

        X_train, y_train, groups_train = prepare(df_train, links=links)
        if X_train.empty:
            continue

        model = _make_model()
        model.fit(X_train, y_train, group=groups_train)

        metrics = evaluate(model, df_test, label=f"Test {test_season} (train {train_seasons[0]}-{train_seasons[-1]})", links=links)
        metrics["test_season"] = test_season
        metrics["train_seasons"] = train_seasons
        results.append(metrics)

    return results


# ── Training & saving ──────────────────────────────────────────────────────────

def train_final(df: pd.DataFrame, links: bool = False) -> XGBRanker:
    """Train on all available seasons and return the fitted model."""
    X, y, groups = prepare(df, links=links)
    model = _make_model()
    model.fit(X, y, group=groups)
    return model


def save(model: XGBRanker, links: bool = False) -> None:
    path = MODEL_LINKS_PATH if links else MODEL_BASE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved → {path}")


def load(links: bool = False) -> XGBRanker:
    path = MODEL_LINKS_PATH if links else MODEL_BASE_PATH
    if not path.exists() and links:
        raise FileNotFoundError(
            f"Links model not found at {path}. "
            "Run `python -m src.model --links` to train it first."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Feature importances ────────────────────────────────────────────────────────

def feature_importance(model: XGBRanker, top_n: int = 20) -> pd.Series:
    present = [c for c in FEATURE_COLS if c in model.feature_names_in_]
    imp = pd.Series(
        model.feature_importances_,
        index=model.feature_names_in_,
    ).loc[present].sort_values(ascending=False)
    return imp.head(top_n)


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv",    action="store_true", help="Run temporal CV")
    parser.add_argument("--links", action="store_true",
                        help="Train/evaluate the links-tuned model (adds links_sg_hist, links_starts)")
    parser.add_argument(
        "--data", default=str(FEATURE_MATRIX),
        help="Path to feature matrix parquet"
    )
    args = parser.parse_args()

    label_suffix = " [LINKS]" if args.links else " [BASE]"
    df = pd.read_parquet(args.data)
    seasons = sorted(df["season"].unique())
    print(f"Loaded {len(df):,} rows | seasons {seasons[0]}–{seasons[-1]}{label_suffix}")

    if args.cv:
        print(f"\n=== Temporal Cross-Validation (expanding window){label_suffix} ===")
        cv_results = temporal_cv(df, links=args.links)
        if cv_results:
            cv_df = pd.DataFrame(cv_results)
            print(f"\nCV Summary across {len(cv_df)} folds:")
            for col in ["rmse", "mae", "spearman_mean", "spearman_median"]:
                print(f"  {col:<22} mean={cv_df[col].mean():.3f}  std={cv_df[col].std():.3f}")
    else:
        # Train/test split: train on all but last season, test on last season
        train_seasons = seasons[:-1]
        test_season   = seasons[-1]

        df_train = df[df["season"].isin(train_seasons)]
        df_test  = df[df["season"] == test_season]

        print(f"\nTrain: {train_seasons[0]}–{train_seasons[-1]} ({len(df_train):,} rows)")
        print(f"Test:  {test_season} ({len(df_test):,} rows)")

        X_train, y_train, groups_train = prepare(df_train, links=args.links)
        print(f"Training on {len(X_train):,} rows × {len(X_train.columns)} features ...")
        model = _make_model()
        model.fit(X_train, y_train, group=groups_train)

        evaluate(model, df_train, label=f"Train {train_seasons[0]}–{train_seasons[-1]}", links=args.links)
        evaluate(model, df_test,  label=f"Test {test_season}", links=args.links)

        print("\n── Feature Importances (top 15) ──")
        imp = feature_importance(model, top_n=15)
        for feat, score in imp.items():
            bar = "█" * int(score * 200)
            print(f"  {feat:<45} {score:.4f}  {bar}")

        # Retrain on all seasons for production use
        print(f"\nRetraining on all {len(seasons)} seasons for production ...")
        model_final = train_final(df, links=args.links)
        save(model_final, links=args.links)
