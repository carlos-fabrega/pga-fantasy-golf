"""
Train/serve feature-parity check for the SG rolling features.

Background: training builds SG rolling features in features.add_rolling_sg;
inference rebuilds them in scrape.build_weekly_features from the live GraphQL
API. If the two use different math, the model scores production rows on a
different feature distribution than it trained on — silent, and it was real:
training used an EWM over the full career while inference used a flat mean of
the last <=10 events.

This script reconstructs, from the processed feature_matrix, what inference
WOULD compute for each row (flat mean of the player's last N SG-recorded events
strictly before that event) and compares it to the stored training feature.

It reports two things separately:
  1. METHOD parity  — inference logic with NO event cap vs the training column.
     After the EWM->flat fix these must match to floating-point tolerance.
  2. PRODUCTION parity — inference logic capped at 10 events (the live API's hard
     limit) vs the training column. roll8 still matches; roll16/roll24 diverge
     for players with >10 prior starts. That residual is the window-saturation
     skew, only closeable by feeding stored history at inference (needs backfill).

Run:  python -m tests.test_feature_parity
Exit code is nonzero if METHOD parity fails (the part we control).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import _sort_key

MATRIX = Path("data/processed/feature_matrix.parquet")
SG_COL = "sg_total_per_round"
WINDOWS = [8, 16, 24]
API_CAP = 10          # playerFinishStats returns at most 10 events
TOL = 1e-9
N_PLAYERS = 40        # sample size for the report


def _inference_value(prior_sg_desc: list[float], window: int, cap: int | None) -> float:
    """
    Replicate scrape.build_weekly_features: flat mean of the last `window`
    SG-recorded events. `prior_sg_desc` is newest-first. `cap` mirrors the live
    API limit on how many events are even visible.
    """
    vals = prior_sg_desc if cap is None else prior_sg_desc[:cap]
    if not vals:
        return np.nan
    n = min(window, len(vals))
    return float(np.mean(vals[:n]))


def main() -> int:
    if not MATRIX.exists():
        print(f"Missing {MATRIX}; run `python -m src.pipeline` first.")
        return 2

    df = pd.read_parquet(MATRIX)
    df = df.copy()
    df["_order"] = _sort_key(df)

    # Players with enough SG history to exercise the long windows.
    sg_counts = df[df[SG_COL].notna()].groupby("player_id").size()
    rich = sg_counts[sg_counts >= 30].index.tolist()
    sample = rich[:N_PLAYERS]

    method_err = {w: [] for w in WINDOWS}     # uncapped vs training column
    prod_err = {w: [] for w in WINDOWS}       # capped-at-10 vs training column

    for pid in sample:
        g = df[df["player_id"] == pid].sort_values("_order")
        sg_series = g[SG_COL].values
        # Compare on made-cut rows (rows that actually have SG), where training
        # and inference are both well-defined.
        for i in range(len(g)):
            if not np.isfinite(sg_series[i]):
                continue
            prior = [v for v in sg_series[:i] if np.isfinite(v)]
            prior_desc = prior[::-1]
            if len(prior_desc) < 12:          # need history past the API cap to be meaningful
                continue
            for w in WINDOWS:
                train_v = g.iloc[i][f"{SG_COL}_roll{w}"]
                if not np.isfinite(train_v):
                    continue
                method_v = _inference_value(prior_desc, w, cap=None)
                prod_v = _inference_value(prior_desc, w, cap=API_CAP)
                method_err[w].append(abs(train_v - method_v))
                prod_err[w].append(abs(train_v - prod_v))

    print(f"Sampled {len(sample)} players, comparing {SG_COL}_roll{{8,16,24}}\n")
    print(f"{'window':<8}{'METHOD max|Δ|':<18}{'PROD max|Δ|':<16}{'PROD mean|Δ|':<14}{'n':<8}")
    print("-" * 60)
    method_ok = True
    for w in WINDOWS:
        me = np.array(method_err[w]); pe = np.array(prod_err[w])
        if len(me) == 0:
            print(f"{w:<8}{'(no samples)':<18}")
            continue
        m_max = me.max()
        print(f"{w:<8}{m_max:<18.2e}{pe.max():<16.4f}{pe.mean():<14.4f}{len(me):<8}")
        if m_max > TOL:
            method_ok = False

    print()
    if method_ok:
        print("PASS  METHOD parity: training math == inference math (EWM->flat fix verified).")
    else:
        print("FAIL  METHOD parity: training and inference compute SG rolls differently.")

    print("NOTE  PROD parity: roll8 ~0 (API cap >= 8). roll16/roll24 diverge because the")
    print("      live API returns <=10 SG events — residual window-saturation skew that")
    print("      needs stored history at inference (blocked on the data backfill).")

    return 0 if method_ok else 1


if __name__ == "__main__":
    sys.exit(main())
