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

from src.features import is_playoff_tournament
from src.paths import (
    MODEL_BASE_PATH, MODEL_LINKS_PATH, MODEL_PLAYOFF_PATH, MODEL_NOCUT_PATH,
    FEATURE_MATRIX,
)

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
    # Historical performance at FedEx Cup Playoff events (BMW, Tour Championship,
    # pre-2019 Barclays/Dell Technologies/Deutsche Bank, WGC-FedEx St. Jude).
    # Included in the base feature set (available for every prediction, not just
    # playoff weeks) since it's a cheap, always-computable signal.
    "playoff_sg_hist", "playoff_starts",
]

# Links model adds historical performance across all links courses as a signal.
# links_sg_hist: player's average SG total per round at links venues (all-time).
# links_starts:  number of prior links starts (confidence weight for the above).
FEATURE_COLS_LINKS: List[str] = FEATURE_COLS + ["links_sg_hist", "links_starts"]

# Playoff model reuses the base feature set — playoff_sg_hist/playoff_starts are
# already in FEATURE_COLS above. What distinguishes this variant is the training
# *rows*: prepare() restricts to FedEx Cup Playoff tournaments only (see below),
# unlike the links model which trains on the full dataset with links features as
# just two extra optional columns.
FEATURE_COLS_PLAYOFF: List[str] = FEATURE_COLS

# No-cut model drops the made_cut_rate_* features. At no-cut events (Tour
# Championship, limited-field invitationals) every player completes four rounds
# and earns a cheque, so made-cut rate encodes a downside risk that does not
# exist that week — it penalises high-SG players who happen to have missed cuts
# in ordinary full-field events. Same training rows as base; two fewer columns.
FEATURE_COLS_NOCUT: List[str] = [
    c for c in FEATURE_COLS if not c.startswith("made_cut_rate")
]

# Maps a variant name to its feature set / model path, used throughout this module.
VARIANT_FEATURES: Dict[str, List[str]] = {
    "base": FEATURE_COLS,
    "links": FEATURE_COLS_LINKS,
    "playoff": FEATURE_COLS_PLAYOFF,
    "nocut": FEATURE_COLS_NOCUT,
}
VARIANT_PATHS = {
    "base": MODEL_BASE_PATH,
    "links": MODEL_LINKS_PATH,
    "playoff": MODEL_PLAYOFF_PATH,
    "nocut": MODEL_NOCUT_PATH,
}

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

def prepare(df: pd.DataFrame, variant: str = "base") -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """
    Return (X, y, groups) for the LambdaRank model.

    variant="links" uses FEATURE_COLS_LINKS (adds links_sg_hist, links_starts),
    trained on the full dataset like "base" — the links columns are just two
    extra optional features.

    variant="playoff" additionally restricts training rows to FedEx Cup Playoff
    tournaments only (BMW Championship, Tour Championship, etc.) — a genuine
    row subset, not just extra columns, since the goal is a model calibrated to
    playoff-strength fields specifically.

    All rows with a non-null target and at least one non-null SG feature are
    included — missed-cut players (prize_pct=0) are valid lowest-relevance
    examples that help the model learn to rank them below made-cut players.

    Data is sorted by tournament_id so XGBRanker's group array is contiguous.
    groups[i] = number of rows belonging to tournament i.
    """
    feature_cols = VARIANT_FEATURES[variant]

    has_target = df[TARGET_COL].notna()
    has_any_sg = df[[c for c in feature_cols if "sg_" in c and c in df.columns]].notna().any(axis=1)
    mask = has_target & has_any_sg
    if variant == "playoff":
        mask &= df["tournament_name"].apply(is_playoff_tournament)

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
    variant: str = "base",
) -> Dict[str, float]:
    """
    Evaluate a fitted model on df_test.
    Returns a dict of scalar metrics and prints a summary.
    """
    X_test, y_test, _ = prepare(df_test, variant=variant)
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

def _temporal_folds(df: pd.DataFrame, min_train_seasons: int, variant: str):
    """
    Yield (test_season, train_seasons, model, df_test) for each expanding-window
    fold: the model is fitted on all seasons < test_season only.
    """
    seasons = sorted(df["season"].unique())

    for i, test_season in enumerate(seasons):
        if i < min_train_seasons:
            continue  # need enough history to train

        train_seasons = seasons[:i]
        df_train = df[df["season"].isin(train_seasons)]
        df_test  = df[df["season"] == test_season]

        X_train, y_train, groups_train = prepare(df_train, variant=variant)
        if X_train.empty:
            continue

        model = _make_model()
        model.fit(X_train, y_train, group=groups_train)
        yield test_season, train_seasons, model, df_test


def temporal_cv(df: pd.DataFrame, min_train_seasons: int = 3, variant: str = "base") -> List[Dict]:
    """
    Expanding-window temporal CV.
    For each season S (starting after min_train_seasons), train on all seasons
    < S and test on season S. Reports metrics per fold.

    For variant="playoff", both train and test rows are restricted to playoff
    tournaments by prepare(), so folds cover far fewer rows/tournaments than
    base or links — expect noisier per-fold metrics.
    """
    results = []
    for test_season, train_seasons, model, df_test in _temporal_folds(df, min_train_seasons, variant):
        metrics = evaluate(model, df_test, label=f"Test {test_season} (train {train_seasons[0]}-{train_seasons[-1]})", variant=variant)
        metrics["test_season"] = test_season
        metrics["train_seasons"] = train_seasons
        results.append(metrics)

    return results


def oof_predictions(df: pd.DataFrame, min_train_seasons: int = 3, variant: str = "base") -> pd.DataFrame:
    """
    Out-of-fold scores from the same expanding-window folds as temporal_cv():
    every row of season S is scored by a model trained only on seasons < S, so
    these are honest backtest predictions (unlike scoring past events with the
    final all-seasons model, which saw them in training).

    Returns one row per scored player-tournament with player_id, player_name,
    tournament_id, tournament_name, season, finish_num, made_cut, prize_pct,
    score (raw LambdaRank margin) and rank (1 = best within the tournament).
    The first min_train_seasons seasons have no predictions.
    """
    keep = ["player_id", "player_name", "tournament_id", "tournament_name",
            "season", "finish_num", "made_cut", TARGET_COL]
    frames = []
    for _, _, model, df_test in _temporal_folds(df, min_train_seasons, variant):
        X_test, _, _ = prepare(df_test, variant=variant)
        if X_test.empty:
            continue
        out = df_test.loc[X_test.index, [c for c in keep if c in df_test.columns]].copy()
        out["score"] = model.predict(X_test)
        frames.append(out)

    if not frames:
        return pd.DataFrame(columns=keep + ["score", "rank"])
    oof = pd.concat(frames).reset_index(drop=True)
    oof["rank"] = oof.groupby("tournament_id")["score"].rank(ascending=False, method="min").astype(int)
    return oof


# ── Training & saving ──────────────────────────────────────────────────────────

def train_final(df: pd.DataFrame, variant: str = "base") -> XGBRanker:
    """Train on all available seasons and return the fitted model."""
    X, y, groups = prepare(df, variant=variant)
    model = _make_model()
    model.fit(X, y, group=groups)
    return model


def save(model: XGBRanker, variant: str = "base") -> None:
    path = VARIANT_PATHS[variant]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved → {path}")


def load(variant: str = "base") -> XGBRanker:
    path = VARIANT_PATHS[variant]
    if not path.exists() and variant != "base":
        raise FileNotFoundError(
            f"{variant.capitalize()} model not found at {path}. "
            f"Run `python -m src.model --{variant}` to train it first."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Feature importances ────────────────────────────────────────────────────────

def feature_importance(model: XGBRanker, variant: str = "base", top_n: int = 20) -> pd.Series:
    present = [c for c in VARIANT_FEATURES[variant] if c in model.feature_names_in_]
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
    parser.add_argument("--playoff", action="store_true",
                        help="Train/evaluate the FedEx Cup Playoff-tuned model "
                             "(trained only on BMW/Tour Championship/etc. rows)")
    parser.add_argument("--nocut", action="store_true",
                        help="Train/evaluate the no-cut model (base features minus "
                             "made_cut_rate_*), for no-cut/limited-field events")
    parser.add_argument(
        "--data", default=str(FEATURE_MATRIX),
        help="Path to feature matrix parquet"
    )
    args = parser.parse_args()
    if sum([args.links, args.playoff, args.nocut]) > 1:
        parser.error("--links, --playoff and --nocut are mutually exclusive.")
    variant = ("links" if args.links else "playoff" if args.playoff
               else "nocut" if args.nocut else "base")

    label_suffix = f" [{variant.upper()}]"
    df = pd.read_parquet(args.data)
    seasons = sorted(df["season"].unique())
    print(f"Loaded {len(df):,} rows | seasons {seasons[0]}–{seasons[-1]}{label_suffix}")

    if args.cv:
        print(f"\n=== Temporal Cross-Validation (expanding window){label_suffix} ===")
        cv_results = temporal_cv(df, variant=variant)
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

        X_train, y_train, groups_train = prepare(df_train, variant=variant)
        print(f"Training on {len(X_train):,} rows × {len(X_train.columns)} features ...")
        model = _make_model()
        model.fit(X_train, y_train, group=groups_train)

        evaluate(model, df_train, label=f"Train {train_seasons[0]}–{train_seasons[-1]}", variant=variant)
        evaluate(model, df_test,  label=f"Test {test_season}", variant=variant)

        print("\n── Feature Importances (top 15) ──")
        imp = feature_importance(model, variant=variant, top_n=15)
        for feat, score in imp.items():
            bar = "█" * int(score * 200)
            print(f"  {feat:<45} {score:.4f}  {bar}")

        # Retrain on all seasons for production use
        print(f"\nRetraining on all {len(seasons)} seasons for production ...")
        model_final = train_final(df, variant=variant)
        save(model_final, variant=variant)
