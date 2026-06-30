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
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from xgboost import XGBRegressor

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

TARGET_COL = "prize_pct"
MODEL_PATH = Path("data/processed/model.pkl")


# ── XGBoost config ─────────────────────────────────────────────────────────────

def _make_model(n_estimators: int = 800) -> XGBRegressor:
    # Hyperparameters are tuned for the prize_pct target (range ~0.001–0.18).
    # gamma and reg_lambda must be scaled relative to the target's magnitude —
    # the same values that worked for dk_points (range 0–200) would suppress
    # all splits here since XGBoost gain scales with (target values)².
    return XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.0,
        reg_lambda=0.01,
        tree_method="hist",
        enable_categorical=False,
        random_state=42,
        n_jobs=-1,
    )


# ── Data prep ──────────────────────────────────────────────────────────────────

def prepare(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Return (X, y) for rows that have a non-null target and at least one
    non-null SG rolling feature (XGBoost can handle rows where ALL features
    are NaN, but those rows carry no information).

    When target is prize_pct, restrict to made-cut rows only.  All missed-cut
    rows share prize_pct=0.0 exactly, forming a degenerate mass that prevents
    the model from finding useful splits.  The model still ranks likely
    missed-cut players lower at inference time via their weak SG features.
    """
    has_target = df[TARGET_COL].notna()
    has_any_sg = df[[c for c in FEATURE_COLS if "sg_" in c]].notna().any(axis=1)
    mask = has_target & has_any_sg

    if TARGET_COL == "prize_pct" and "made_cut" in df.columns:
        mask = mask & df["made_cut"].fillna(False).astype(bool)

    present_features = [c for c in FEATURE_COLS if c in df.columns]
    X = df.loc[mask, present_features].astype(float)
    y = df.loc[mask, TARGET_COL].astype(float)
    return X, y


# ── Evaluation ─────────────────────────────────────────────────────────────────

def _spearman_by_tournament(
    df_test: pd.DataFrame,
    y_pred: np.ndarray,
    min_players: int = 20,
) -> pd.Series:
    """
    Per-tournament Spearman rank correlation between predicted and actual DK points.
    Only include tournaments where at least min_players have valid predictions.
    """
    tmp = df_test[["tournament_id", TARGET_COL]].copy()
    tmp["pred"] = y_pred
    results = {}
    for tid, grp in tmp.groupby("tournament_id"):
        valid = grp.dropna(subset=[TARGET_COL, "pred"])
        if len(valid) < min_players:
            continue
        r, _ = spearmanr(valid[TARGET_COL], valid["pred"])
        results[tid] = r
    return pd.Series(results)


def evaluate(
    model: XGBRegressor,
    df_test: pd.DataFrame,
    label: str = "Test",
) -> Dict[str, float]:
    """
    Evaluate a fitted model on df_test.
    Returns a dict of scalar metrics and prints a summary.
    """
    X_test, y_test = prepare(df_test)
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

def temporal_cv(df: pd.DataFrame, min_train_seasons: int = 3) -> List[Dict]:
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

        X_train, y_train = prepare(df_train)
        if X_train.empty:
            continue

        model = _make_model()
        model.fit(X_train, y_train)

        metrics = evaluate(model, df_test, label=f"Test {test_season} (train {train_seasons[0]}-{train_seasons[-1]})")
        metrics["test_season"] = test_season
        metrics["train_seasons"] = train_seasons
        results.append(metrics)

    return results


# ── Training & saving ──────────────────────────────────────────────────────────

def train_final(df: pd.DataFrame) -> XGBRegressor:
    """Train on all available seasons and return the fitted model."""
    X, y = prepare(df)
    model = _make_model()
    model.fit(X, y)
    return model


def save(model: XGBRegressor, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved → {path}")


def load(path: Path = MODEL_PATH) -> XGBRegressor:
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Feature importances ────────────────────────────────────────────────────────

def feature_importance(model: XGBRegressor, top_n: int = 20) -> pd.Series:
    present = [c for c in FEATURE_COLS if c in model.feature_names_in_]
    imp = pd.Series(
        model.feature_importances_,
        index=model.feature_names_in_,
    ).loc[present].sort_values(ascending=False)
    return imp.head(top_n)


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv", action="store_true", help="Run temporal CV")
    parser.add_argument(
        "--data", default="data/processed/feature_matrix.parquet",
        help="Path to feature matrix parquet"
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.data)
    seasons = sorted(df["season"].unique())
    print(f"Loaded {len(df):,} rows | seasons {seasons[0]}–{seasons[-1]}")

    if args.cv:
        print("\n=== Temporal Cross-Validation (expanding window) ===")
        cv_results = temporal_cv(df)
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

        X_train, y_train = prepare(df_train)
        print(f"Training on {len(X_train):,} rows × {len(X_train.columns)} features ...")
        model = _make_model()
        model.fit(X_train, y_train)

        evaluate(model, df_train, label=f"Train {train_seasons[0]}–{train_seasons[-1]}")
        evaluate(model, df_test,  label=f"Test {test_season}")

        print("\n── Feature Importances (top 15) ──")
        imp = feature_importance(model, top_n=15)
        for feat, score in imp.items():
            bar = "█" * int(score * 200)
            print(f"  {feat:<45} {score:.4f}  {bar}")

        # Retrain on all seasons for production use
        print(f"\nRetraining on all {len(seasons)} seasons for production ...")
        model_final = train_final(df)
        save(model_final)
